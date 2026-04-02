package org.beehive.gpullama3.tornadovm.layers.type.fp16;

import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerBatchPrefillKernels;
import org.beehive.gpullama3.tornadovm.layerplanner.WorkerGridFactory;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.List;
import java.util.stream.IntStream;

/**
 * Builds per-layer batch prefill TaskGraphs for Phase 4 GPU batched prefill.
 *
 * <p>One {@link ImmutableTaskGraph} per transformer layer, each processing
 * {@code batchSize} tokens simultaneously via {@link TransformerBatchPrefillKernels}.</p>
 *
 * <p>KV cache ({@code wrapKeyCache}, {@code wrapValueCache}) is persisted on device
 * after every layer so the subsequent single-token decode layers can consume it.</p>
 */
public class LlamaFP16BatchPrefillLayers {

    // Matches the local workgroup size used by the single-token kernels.
    static final int LOCAL_WORK_GROUP_SIZE = 32;

    private final LlamaState state;
    private final LlamaTornadoWeights weights;
    private final LlamaConfiguration config;
    private final KernelContext context = new KernelContext();
    private final int batchSize;
    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    public LlamaFP16BatchPrefillLayers(LlamaState state, LlamaTornadoWeights weights,
                                        LlamaConfiguration config, int batchSize) {
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.layerITGs = IntStream.range(0, config.numberOfLayers())
                .mapToObj(this::createBatchLayerTaskGraph)
                .map(TaskGraph::snapshot)
                .toList();
    }

    // @formatter:off
    private TaskGraph createBatchLayerTaskGraph(int layerIndex) {
        String graphName = "batchLayer_" + layerIndex;
        if (layerIndex == config.numberOfLayers() - 1) lastLayerTaskGraphID = graphName;

        TaskGraph layer = new TaskGraph(graphName);

        // ── Data Transfers ─────────────────────────────────────────────────────
        if (layerIndex == 0) {
            // batchStartPosHolder is set by host before each chunk → EVERY_EXECUTION
            layer.transferToDevice(DataTransferMode.EVERY_EXECUTION, state.batchStartPosHolder);
            // Allocate persistent GPU-side intermediates once
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.attnScaleBatch, state.ffnScaleBatch,
                    state.wrapXbFP16Batch,
                    state.wrapQBatch, state.wrapKBatch, state.wrapVBatch,
                    state.wrapXbBatch,
                    state.wrapHbBatch,
                    state.wrapKeyCache, state.wrapValueCache);
            // wrapXBatch produced by the batch activation graph
            layer.consumeFromDevice(state.wrapXBatch);
        } else {
            layer.consumeFromDevice(
                    context,
                    state.wrapXBatch,
                    state.wrapXbFP16Batch,
                    state.wrapQBatch, state.wrapKBatch, state.wrapVBatch,
                    state.wrapXbBatch,
                    state.wrapHbBatch,
                    state.wrapKeyCache, state.wrapValueCache,
                    state.batchStartPosHolder,
                    state.attnScaleBatch, state.ffnScaleBatch);
        }

        // Per-layer weights: upload once
        layer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asHalfFloatArray(),
                weights.wkLayered[layerIndex].asHalfFloatArray(),
                weights.wvLayered[layerIndex].asHalfFloatArray(),
                weights.woLayered[layerIndex].asHalfFloatArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].asHalfFloatArray(),
                weights.w2Layered[layerIndex].asHalfFloatArray(),
                weights.w3Layered[layerIndex].asHalfFloatArray());

        int dim      = config.dim();
        int kvDim    = config.kvDim();
        int hidDim   = config.hiddenDim();

        // ── Attention Block ────────────────────────────────────────────────────
        layer.task("batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduce,
                context, state.wrapXBatch, state.attnScaleBatch,
                dim, config.rmsNormEps());

        layer.task("batch_attn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP16,
                context, state.wrapXbFP16Batch, state.wrapXBatch,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.attnScaleBatch, dim);

        layer.task("batch_qkv",
                TransformerBatchPrefillKernels::batchedFusedQKVMatmul,
                context,
                state.wrapXbFP16Batch,
                state.wrapQBatch, state.wrapKBatch, state.wrapVBatch,
                weights.wqLayered[layerIndex].asHalfFloatArray(),
                weights.wkLayered[layerIndex].asHalfFloatArray(),
                weights.wvLayered[layerIndex].asHalfFloatArray(),
                dim, kvDim, LOCAL_WORK_GROUP_SIZE);

        layer.task("batch_rope_kv",
                TransformerBatchPrefillKernels::batchedRopeWithKVCache,
                context, state.batchStartPosHolder,
                state.wrapQBatch, state.wrapKBatch, state.wrapVBatch,
                state.wrapKeyCache, state.wrapValueCache,
                kvDim, config.headSize(), layerIndex, config.contextLength(), dim);

        layer.task("batch_attention",
                TransformerBatchPrefillKernels::batchedFlashAttention,
                context, state.batchStartPosHolder,
                state.wrapQBatch, state.wrapKeyCache, state.wrapValueCache,
                state.wrapXbBatch,
                config.numberOfHeads(), config.headSize(),
                kvDim, config.kvMul(), layerIndex, config.contextLength(), dim);

        layer.task("batch_attn_out",
                TransformerBatchPrefillKernels::batchedMatVecWithResidual,
                context, state.wrapXbBatch, state.wrapXBatch,
                weights.woLayered[layerIndex].asHalfFloatArray(),
                dim, dim, LOCAL_WORK_GROUP_SIZE);

        // ── FFN Block ──────────────────────────────────────────────────────────
        layer.task("batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedFFNRmsReduce,
                context, state.wrapXBatch, state.ffnScaleBatch,
                dim, config.rmsNormEps());

        layer.task("batch_ffn_gate_up",
                TransformerBatchPrefillKernels::batchedFusedRmsNormFFNGateUp,
                context, state.wrapXBatch, state.wrapHbBatch,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.ffnScaleBatch,
                weights.w1Layered[layerIndex].asHalfFloatArray(),
                weights.w3Layered[layerIndex].asHalfFloatArray(),
                dim, hidDim, LOCAL_WORK_GROUP_SIZE);

        layer.task("batch_ffn_down",
                TransformerBatchPrefillKernels::batchedMatVecWithResidual,
                context, state.wrapHbBatch, state.wrapXBatch,
                weights.w2Layered[layerIndex].asHalfFloatArray(),
                hidDim, dim, LOCAL_WORK_GROUP_SIZE);

        // Persist wrapXBatch for the next layer, and KV cache so the decode
        // layers can consume it via the activation graph pass-through.
        layer.persistOnDevice(state.wrapXBatch, state.wrapKeyCache, state.wrapValueCache);

        return layer;
    }
    // @formatter:on

    /** Registers all batch layer workers in the shared {@link GridScheduler}. */
    public void updateGridScheduler(GridScheduler scheduler) {
        int dim     = config.dim();
        int kvDim   = config.kvDim();
        int hidDim  = config.hiddenDim();
        int nHeads  = config.numberOfHeads();
        int headSz  = config.headSize();

        // RMS: one thread per batch token
        WorkerGrid rmsWorker = WorkerGridFactory.genericWorker(batchSize, 1);

        // RMS apply: B*dim threads, local=256 (dim is always a multiple of 256 for LLaMA)
        WorkerGrid rmsApplyWorker = WorkerGridFactory.genericWorker(batchSize * dim, 256);

        // QKV: B*(dim+2*kvDim) workgroups × LOCAL_WORK_GROUP_SIZE
        int qkvRows = dim + 2 * kvDim;
        WorkerGrid qkvWorker = WorkerGridFactory.genericWorker(
                batchSize * qkvRows * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);

        // RoPE+KV cache: B*(dim/2) threads, local=512
        int ropeGlobal = batchSize * (dim / 2);
        int ropeLocal  = Math.min(512, ropeGlobal);
        while (ropeLocal > 1 && ropeGlobal % ropeLocal != 0) ropeLocal--;
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(ropeGlobal, ropeLocal);

        // Attention (flash): B*nHeads workgroups × optimalLocalSize
        int optLocal = findOptimalLocalSize(headSz);
        WorkerGrid attnWorker = WorkerGridFactory.genericWorker(
                batchSize * nHeads * optLocal, optLocal);

        // Mat-vec (Wo, W2): B*d workgroups × LOCAL_WORK_GROUP_SIZE
        WorkerGrid matVecDimWorker = WorkerGridFactory.genericWorker(
                batchSize * dim * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);
        WorkerGrid matVecHidWorker = WorkerGridFactory.genericWorker(
                batchSize * hidDim * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);

        for (int i = 0; i < config.numberOfLayers(); i++) {
            String p = "batchLayer_" + i + ".";
            scheduler.addWorkerGrid(p + "batch_attn_rms",     rmsWorker);
            scheduler.addWorkerGrid(p + "batch_attn_rms_apply", rmsApplyWorker);
            scheduler.addWorkerGrid(p + "batch_qkv",          qkvWorker);
            scheduler.addWorkerGrid(p + "batch_rope_kv",      ropeWorker);
            scheduler.addWorkerGrid(p + "batch_attention",    attnWorker);
            scheduler.addWorkerGrid(p + "batch_attn_out",     matVecDimWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_rms",      rmsWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_gate_up",  matVecHidWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_down",     matVecDimWorker);
        }
    }

    private static int findOptimalLocalSize(int size) {
        int optimal = Math.min(size, 64);
        if (size % optimal != 0) {
            for (int s = 64; s >= 1; s--) {
                if (size % s == 0) { optimal = s; break; }
            }
        }
        return optimal;
    }

    public List<ImmutableTaskGraph> getLayerImmutableTaskGraphs() { return layerITGs; }
    public String getLastLayerTaskGraphID()                        { return lastLayerTaskGraphID; }
    public KernelContext getContext()                               { return context; }
}
