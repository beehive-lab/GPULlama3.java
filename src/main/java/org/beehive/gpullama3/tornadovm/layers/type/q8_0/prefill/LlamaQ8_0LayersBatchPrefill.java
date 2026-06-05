package org.beehive.gpullama3.tornadovm.layers.type.q8_0.prefill;

import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerBatchPrefillKernels;
import org.beehive.gpullama3.tornadovm.scheduling.WorkerGridFactory;
import org.beehive.gpullama3.tornadovm.layers.BatchPrefillTransformerLayerTaskGraphs;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.List;
import java.util.stream.IntStream;

/**
 * Batched-prefill transformer-layer TaskGraphs for the unified batched prefill-decode plan (Q8_0).
 *
 * <p>Mirrors {@link org.beehive.gpullama3.tornadovm.layers.type.fp16.prefill.LlamaFP16LayersBatchPrefill}
 * but uses Q8_0 kernels with inline dequantization. Key differences from the FP16 path:</p>
 * <ul>
 *   <li>{@code wrapXBatch} is filled with dequantized FP32 embeddings by the host before
 *       the activation graph runs (no on-device FP16→FP32 conversion).</li>
 *   <li>{@code wrapXbBatch} (FP32) is reused as the normalized xb intermediate: written
 *       by {@code batchedRmsApplyFP32}, read by {@code batchedFusedQKVMatmulQ8}, then
 *       overwritten by flash attention output.</li>
 *   <li>{@code wrapXbFP16Batch} is not used.</li>
 *   <li>Weight matrices are {@code ByteArray} (Q8_0 format).</li>
 * </ul>
 */
public class LlamaQ8_0LayersBatchPrefill implements BatchPrefillTransformerLayerTaskGraphs {

    static final int LOCAL_WORK_GROUP_SIZE = 32;

    private final LlamaState state;
    private final LlamaTornadoWeights weights;
    private final LlamaConfiguration config;
    private final KernelContext context = new KernelContext();
    private final int batchSize;
    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    public LlamaQ8_0LayersBatchPrefill(LlamaState state, LlamaTornadoWeights weights,
                                        LlamaConfiguration config, int batchSize) {
        this.state     = state;
        this.weights   = weights;
        this.config    = config;
        this.batchSize = batchSize;
        this.layerITGs = IntStream.range(0, config.numberOfLayers())
                .mapToObj(this::createBatchPrefillLayerTaskGraph)
                .map(TaskGraph::snapshot)
                .toList();
    }

    // @formatter:off
    private TaskGraph createBatchPrefillLayerTaskGraph(int layerIndex) {
        String graphName = "batchPrefillLayer_" + layerIndex;
        if (layerIndex == config.numberOfLayers() - 1) lastLayerTaskGraphID = graphName;

        TaskGraph layer = new TaskGraph(graphName);

        // ── Data Transfers ─────────────────────────────────────────────────────
        if (layerIndex == 0) {
            // batchStartPosHolder is set by host before each chunk → EVERY_EXECUTION
            layer.transferToDevice(DataTransferMode.EVERY_EXECUTION, state.batchStartPosHolder);
            // Allocate GPU-side batch intermediates once.
            // wrapXBatch is filled with dequantized FP32 by the host, persisted by prefillActivation.
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.attnScaleBatch, state.ffnScaleBatch,
                    state.wrapXbBatch,
                    state.wrapQBatch, state.wrapKBatch, state.wrapVBatch,
                    state.wrapHbBatch,
                    state.wrapKeyCache, state.wrapValueCache);
            layer.consumeFromDevice("prefillActivation", state.wrapXBatch);
        } else {
            String pred = "batchPrefillLayer_" + (layerIndex - 1);
            layer.consumeFromDevice(pred,
                    context,
                    state.wrapXBatch,
                    state.wrapXbBatch,
                    state.wrapQBatch, state.wrapKBatch, state.wrapVBatch,
                    state.wrapHbBatch,
                    state.wrapKeyCache, state.wrapValueCache,
                    state.batchStartPosHolder,
                    state.attnScaleBatch, state.ffnScaleBatch);
        }

        // Per-layer weights: upload once (Q8_0 format)
        layer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                weights.woLayered[layerIndex].asByteArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].asByteArray(),
                weights.w2Layered[layerIndex].asByteArray(),
                weights.w3Layered[layerIndex].asByteArray());

        int dim    = config.dim();
        int kvDim  = config.kvDim();
        int hidDim = config.hiddenDim();

        // ── Attention Block ────────────────────────────────────────────────────
        layer.task("batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduce,
                context, state.wrapXBatch, state.attnScaleBatch,
                dim, config.rmsNormEps());

        // Writes FP32 normalized xb into wrapXbBatch (reused later by flash attention)
        layer.task("batch_attn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP32,
                context, state.wrapXbBatch, state.wrapXBatch,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.attnScaleBatch, dim);

        layer.task("batch_qkv",
                TransformerBatchPrefillKernels::batchedFusedQKVMatmulQ8,
                context,
                state.wrapXbBatch,
                state.wrapQBatch, state.wrapKBatch, state.wrapVBatch,
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                dim, kvDim, LOCAL_WORK_GROUP_SIZE);

        layer.task("batch_rope_kv",
                TransformerBatchPrefillKernels::batchedRopeWithKVCache,
                context, state.batchStartPosHolder,
                state.wrapQBatch, state.wrapKBatch, state.wrapVBatch,
                state.wrapKeyCache, state.wrapValueCache,
                kvDim, config.headSize(), layerIndex, config.contextLength(), dim);

        // Overwrites wrapXbBatch with attention output
        layer.task("batch_attention",
                TransformerBatchPrefillKernels::batchedFlashAttention,
                context, state.batchStartPosHolder,
                state.wrapQBatch, state.wrapKeyCache, state.wrapValueCache,
                state.wrapXbBatch,
                config.numberOfHeads(), config.headSize(),
                kvDim, config.kvMul(), layerIndex, config.contextLength(), dim);

        layer.task("batch_attn_out",
                TransformerBatchPrefillKernels::batchedMatVecWithResidualQ8,
                context, state.wrapXbBatch, state.wrapXBatch,
                weights.woLayered[layerIndex].asByteArray(),
                dim, dim, LOCAL_WORK_GROUP_SIZE);

        // ── FFN Block ──────────────────────────────────────────────────────────
        layer.task("batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedFFNRmsReduce,
                context, state.wrapXBatch, state.ffnScaleBatch,
                dim, config.rmsNormEps());

        layer.task("batch_ffn_gate_up",
                TransformerBatchPrefillKernels::batchedFusedRmsNormFFNGateUpQ8,
                context, state.wrapXBatch, state.wrapHbBatch,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.ffnScaleBatch,
                weights.w1Layered[layerIndex].asByteArray(),
                weights.w3Layered[layerIndex].asByteArray(),
                dim, hidDim, LOCAL_WORK_GROUP_SIZE);

        layer.task("batch_ffn_down",
                TransformerBatchPrefillKernels::batchedMatVecWithResidualQ8,
                context, state.wrapHbBatch, state.wrapXBatch,
                weights.w2Layered[layerIndex].asByteArray(),
                hidDim, dim, LOCAL_WORK_GROUP_SIZE);

        layer.persistOnDevice(state.wrapXBatch, state.wrapKeyCache, state.wrapValueCache);

        return layer;
    }
    // @formatter:on

    public void updateGridScheduler(GridScheduler scheduler) {
        int dim    = config.dim();
        int kvDim  = config.kvDim();
        int hidDim = config.hiddenDim();
        int nHeads = config.numberOfHeads();
        int headSz = config.headSize();

        WorkerGrid rmsWorker       = WorkerGridFactory.genericWorker(batchSize, 1);
        WorkerGrid rmsApplyWorker  = WorkerGridFactory.genericWorker(batchSize * dim, 256);
        int qkvRows = dim + 2 * kvDim;
        WorkerGrid qkvWorker       = WorkerGridFactory.genericWorker(
                batchSize * qkvRows * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);
        int ropeGlobal = batchSize * (dim / 2);
        int ropeLocal  = Math.min(512, ropeGlobal);
        while (ropeLocal > 1 && ropeGlobal % ropeLocal != 0) ropeLocal--;
        WorkerGrid ropeWorker      = WorkerGridFactory.genericWorker(ropeGlobal, ropeLocal);
        int optLocal = findOptimalLocalSize(headSz);
        WorkerGrid attnWorker      = WorkerGridFactory.genericWorker(
                batchSize * nHeads * optLocal, optLocal);
        WorkerGrid matVecDimWorker = WorkerGridFactory.genericWorker(
                batchSize * dim * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);
        WorkerGrid matVecHidWorker = WorkerGridFactory.genericWorker(
                batchSize * hidDim * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);

        for (int i = 0; i < config.numberOfLayers(); i++) {
            String p = "batchPrefillLayer_" + i + ".";
            scheduler.addWorkerGrid(p + "batch_attn_rms",       rmsWorker);
            scheduler.addWorkerGrid(p + "batch_attn_rms_apply", rmsApplyWorker);
            scheduler.addWorkerGrid(p + "batch_qkv",            qkvWorker);
            scheduler.addWorkerGrid(p + "batch_rope_kv",        ropeWorker);
            scheduler.addWorkerGrid(p + "batch_attention",      attnWorker);
            scheduler.addWorkerGrid(p + "batch_attn_out",       matVecDimWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_rms",        rmsWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_gate_up",    matVecHidWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_down",       matVecDimWorker);
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
