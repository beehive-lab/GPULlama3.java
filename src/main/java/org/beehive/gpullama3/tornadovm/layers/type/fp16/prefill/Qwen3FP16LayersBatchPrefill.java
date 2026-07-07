package org.beehive.gpullama3.tornadovm.layers.type.fp16.prefill;

import org.beehive.gpullama3.inference.state.Qwen3State;
import org.beehive.gpullama3.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
import org.beehive.gpullama3.tornadovm.kernels.Qwen3Kernels;
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
 * Batched-prefill transformer-layer TaskGraphs for the Qwen3 FP16 unified batched prefill-decode plan.
 *
 * <p>Mirrors {@link LlamaFP16LayersBatchPrefill} but adapts to Qwen3's GQA layout and
 * Qwen3-specific kernels (fused Q/K RMSNorm, RoPE theta = 1 000 000). Avoids any calls to
 * {@code Qwen3Configuration.headSize()}, {@code kvDim()}, or {@code kvMul()} which throw.</p>
 */
public class Qwen3FP16LayersBatchPrefill implements BatchPrefillTransformerLayerTaskGraphs {

    static final int LOCAL_WORK_GROUP_SIZE = 32;

    private final Qwen3State state;
    private final Qwen3TornadoWeights weights;
    private final Qwen3Configuration config;
    private final KernelContext context = new KernelContext();
    private final int batchSize;
    private final int nHeadKv;
    private final int nEmbdHeadK;
    private final int nEmbdHeadV;
    private final int nEmbdHead;
    private final int qDim;
    private final int kvDim;
    private final int gqa;
    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    public Qwen3FP16LayersBatchPrefill(Qwen3State state, Qwen3TornadoWeights weights,
                                       Qwen3Configuration config, int batchSize) {
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.nHeadKv = config.numberOfKeyValueHeads();
        this.nEmbdHeadK = config.numberOfHeadsKey();
        this.nEmbdHeadV = config.numberOfHeadsValue();
        this.nEmbdHead = nEmbdHeadV;
        this.qDim = nEmbdHeadK * config.numberOfHeads();
        this.kvDim = nEmbdHeadV * nHeadKv;
        this.gqa = config.numberOfHeads() / nHeadKv;
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
        int dim    = config.dim();
        int hidDim = config.hiddenDim();

        // ── Data Transfers ─────────────────────────────────────────────────────
        if (layerIndex == 0) {
            layer.transferToDevice(DataTransferMode.EVERY_EXECUTION, state.batchStartPosHolder);
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.attnScaleBatch, state.ffnScaleBatch,
                    state.wrapXbFP16Batch,
                    state.wrapQBatch, state.wrapKBatch, state.wrapVBatch,
                    state.wrapXbBatch,
                    state.wrapHbBatch,
                    state.wrapKeyCache, state.wrapValueCache);
            layer.consumeFromDevice("prefillActivation", state.wrapXBatch);
        } else {
            String pred = "batchPrefillLayer_" + (layerIndex - 1);
            layer.consumeFromDevice(pred,
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
                weights.rms_att_QNormLayered[layerIndex].asFloatArray(),
                weights.rms_att_KNormLayered[layerIndex].asFloatArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].asHalfFloatArray(),
                weights.w2Layered[layerIndex].asHalfFloatArray(),
                weights.w3Layered[layerIndex].asHalfFloatArray());

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
                Qwen3Kernels::batchedFusedQKVMatmulFP16,
                context,
                state.wrapXbFP16Batch,
                state.wrapQBatch, state.wrapKBatch, state.wrapVBatch,
                weights.wqLayered[layerIndex].asHalfFloatArray(),
                weights.wkLayered[layerIndex].asHalfFloatArray(),
                weights.wvLayered[layerIndex].asHalfFloatArray(),
                dim, qDim, kvDim, LOCAL_WORK_GROUP_SIZE);

        layer.task("batch_qk_rmsnorm",
                Qwen3Kernels::batchedFusedQKRmsNorm,
                context,
                state.wrapQBatch, state.wrapKBatch,
                weights.rms_att_QNormLayered[layerIndex].asFloatArray(),
                weights.rms_att_KNormLayered[layerIndex].asFloatArray(),
                config.numberOfHeads(), nHeadKv, nEmbdHead,
                qDim, kvDim, config.rmsNormEps());

        layer.task("batch_rope_kv",
                Qwen3Kernels::batchedRopeWithKVCacheQwen3,
                context, state.batchStartPosHolder,
                state.wrapQBatch, state.wrapKBatch, state.wrapVBatch,
                state.wrapKeyCache, state.wrapValueCache,
                kvDim, nEmbdHead, layerIndex, config.contextLength(), qDim);

        // Reuses batchedFlashAttention: passes qDim as the 'dim' stride parameter.
        // Valid because qDim == dim for all standard Qwen3 models (nEmbdHeadK = dim/nHeads).
        layer.task("batch_attention",
                TransformerBatchPrefillKernels::batchedFlashAttention,
                context, state.batchStartPosHolder,
                state.wrapQBatch, state.wrapKeyCache, state.wrapValueCache,
                state.wrapXbBatch,
                config.numberOfHeads(), nEmbdHead,
                kvDim, gqa, layerIndex, config.contextLength(), qDim);

        // Output projection: n=qDim (input), d=dim (output)
        layer.task("batch_attn_out",
                TransformerBatchPrefillKernels::batchedMatVecWithResidual,
                context, state.wrapXbBatch, state.wrapXBatch,
                weights.woLayered[layerIndex].asHalfFloatArray(),
                qDim, dim, LOCAL_WORK_GROUP_SIZE);

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

        layer.persistOnDevice(state.wrapXBatch, state.wrapKeyCache, state.wrapValueCache);

        return layer;
    }
    // @formatter:on

    public void updateGridScheduler(GridScheduler scheduler) {
        int dim    = config.dim();
        int hidDim = config.hiddenDim();

        WorkerGrid rmsWorker        = WorkerGridFactory.genericWorker(batchSize, 1);
        WorkerGrid rmsApplyWorker   = WorkerGridFactory.genericWorker(batchSize * dim, 256);

        int qkvRows = qDim + 2 * kvDim;
        WorkerGrid qkvWorker = WorkerGridFactory.genericWorker(
                batchSize * qkvRows * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);

        WorkerGrid qkRmsNormWorker = WorkerGridFactory.genericWorker(
                batchSize * (config.numberOfHeads() + nHeadKv) * nEmbdHead, nEmbdHead);

        int ropeGlobal = batchSize * (qDim / 2);
        int ropeLocal  = Math.min(512, ropeGlobal);
        while (ropeLocal > 1 && ropeGlobal % ropeLocal != 0) ropeLocal--;
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(ropeGlobal, ropeLocal);

        int optLocal = findOptimalLocalSize(nEmbdHead);
        WorkerGrid attnWorker = WorkerGridFactory.genericWorker(
                batchSize * config.numberOfHeads() * optLocal, optLocal);

        // Wo: B*dim output rows (n=qDim, d=dim)
        WorkerGrid matVecDimWorker = WorkerGridFactory.genericWorker(
                batchSize * dim * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);
        WorkerGrid matVecHidWorker = WorkerGridFactory.genericWorker(
                batchSize * hidDim * LOCAL_WORK_GROUP_SIZE, LOCAL_WORK_GROUP_SIZE);

        for (int i = 0; i < config.numberOfLayers(); i++) {
            String p = "batchPrefillLayer_" + i + ".";
            scheduler.addWorkerGrid(p + "batch_attn_rms",       rmsWorker);
            scheduler.addWorkerGrid(p + "batch_attn_rms_apply", rmsApplyWorker);
            scheduler.addWorkerGrid(p + "batch_qkv",            qkvWorker);
            scheduler.addWorkerGrid(p + "batch_qk_rmsnorm",     qkRmsNormWorker);
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
