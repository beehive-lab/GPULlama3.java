package org.beehive.gpullama3.tornadovm.layers.type.q8_0.prefill;

import org.beehive.gpullama3.inference.state.Qwen3State;
import org.beehive.gpullama3.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
import org.beehive.gpullama3.tornadovm.kernels.Qwen3Kernels;
import org.beehive.gpullama3.tornadovm.kernels.TransformerBatchPrefillKernels;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.prefill.LlamaFP16LayersBatchPrefillMMA;
import org.beehive.gpullama3.tornadovm.scheduling.WorkerGridFactory;
import org.beehive.gpullama3.tornadovm.layers.BatchPrefillTransformerLayerTaskGraphs;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.List;
import java.util.stream.IntStream;

/**
 * Qwen3 batched-prefill transformer-layer TaskGraphs on the tensor-core (MMA)
 * pipeline. Mirrors {@link LlamaFP16LayersBatchPrefillMMA} with the Qwen3
 * architectural additions: per-head Q/K RMS normalization between the QKV
 * projection and RoPE, split-half RoPE pairing, and qDim-shaped attention
 * (qDim = nHeads * headDim, which may differ from the model dim).
 *
 * <p>Tensor-core layer pipeline (13 tasks, all GEMMs on MMA, Q8_0 weights dequantized
 * to FP16 in the GEMM staging registers — W8A16):</p>
 * <pre>
 *   batch_attn_rms        parallel RMS square-sum reduction (256 thr/token)
 *   batch_attn_rms_apply  RMS apply + FP16 quantize → wrapXbFP16Batch
 *   qkvProj               ONE fused MMA GEMM → packed qkvResultBatch [q|k|v]
 *   batch_qk_rmsnorm      per-head Q/K RMS norm over the packed buffer
 *   batch_rope_kv         split-half RoPE over the packed buffer + KV cache write
 *   batch_attention       flash attention (register-partitioned P·V) → attnOutFP16
 *   woProj                MMA GEMM [dim × qDim] → woOut
 *   batch_ffn_rms         parallel RMS reduce FUSED with x += woOut
 *   batch_ffn_rms_apply   RMS apply + FP16 quantize → normedXFFNFP16
 *   gateUpProj            ONE fused MMA GEMM → packed gateUpResultBatch [gate|up]
 *   swiglu                SiLU(gate)*up over packed buffer → wrapHbFP16Batch
 *   w2Proj                MMA GEMM → w2Out
 *   w2Resid               x += w2Out
 * </pre>
 *
 * <p>Requires dim, qDim, kvDim, and hidDim to be multiples of 128
 * (holds for all standard Qwen3 checkpoints).</p>
 */
public class Qwen3Q8_0LayersBatchPrefillMMA implements BatchPrefillTransformerLayerTaskGraphs {

    // Local size for the parallel RMS reductions (one workgroup per token).
    static final int RMS_LOCAL_SIZE = 256;

    private final Qwen3State state;
    private final Qwen3TornadoWeights weights;
    private final Qwen3Configuration config;
    private final KernelContext context = new KernelContext();
    private final int batchSize;
    // GEMM M dimension rounded up to whole 128-row tiles (BM); see the Llama
    // planner for the padding rationale. Non-GEMM kernels use the true batchSize.
    private final int paddedBatch;
    private final int nHeadKv;
    private final int nEmbdHead;
    private final int qDim;
    private final int kvDim;
    private final int gqa;
    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    public Qwen3Q8_0LayersBatchPrefillMMA(Qwen3State state, Qwen3TornadoWeights weights,
                                          Qwen3Configuration config, int batchSize) {
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.paddedBatch = (batchSize + 127) & ~127;
        if (batchSize % 128 != 0) {
            System.out.printf("[GPULlama3] prefill batch %d padded to %d for tensor-core tiles; "
                    + "GEMM efficiency is %d/%d — use a multiple of 128 for best throughput.%n",
                    batchSize, paddedBatch, batchSize, paddedBatch);
        }
        this.nHeadKv = config.numberOfKeyValueHeads();
        this.nEmbdHead = config.numberOfHeadsValue();
        this.qDim = config.numberOfHeadsKey() * config.numberOfHeads();
        this.kvDim = config.numberOfHeadsValue() * nHeadKv;
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

        TaskGraph batchPrefillLayer = new TaskGraph(graphName);
        int dim    = config.dim();
        int hidDim = config.hiddenDim();

        // ── Data Transfers ─────────────────────────────────────────────────────
        if (layerIndex == 0) {
            batchPrefillLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION, state.batchStartPosHolder);
            batchPrefillLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.attnScaleBatch, state.ffnScaleBatch,
                    state.wrapXbFP16Batch,
                    state.qkvResultBatch,
                    state.wrapKeyCache, state.wrapValueCache,
                    state.normedXFFNFP16, state.gateUpResultBatch,
                    state.attnOutFP16, state.woOut, state.wrapHbFP16Batch, state.w2Out);
            batchPrefillLayer.consumeFromDevice("prefillActivation", state.wrapXBatch);
        } else {
            String pred = "batchPrefillLayer_" + (layerIndex - 1);
            batchPrefillLayer.consumeFromDevice(pred,
                    context,
                    state.wrapXBatch,
                    state.batchStartPosHolder,
                    state.attnScaleBatch, state.ffnScaleBatch,
                    state.wrapXbFP16Batch,
                    state.qkvResultBatch,
                    state.wrapKeyCache, state.wrapValueCache,
                    state.normedXFFNFP16, state.gateUpResultBatch,
                    state.attnOutFP16, state.woOut, state.wrapHbFP16Batch, state.w2Out);
        }

        // Per-layer weights: upload once
        batchPrefillLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                weights.woLayered[layerIndex].asByteArray(),
                weights.rms_att_QNormLayered[layerIndex].asFloatArray(),
                weights.rms_att_KNormLayered[layerIndex].asFloatArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].asByteArray(),
                weights.w2Layered[layerIndex].asByteArray(),
                weights.w3Layered[layerIndex].asByteArray());

        // ── Attention Block ────────────────────────────────────────────────────
        batchPrefillLayer.task("batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context, state.wrapXBatch, state.attnScaleBatch,
                dim, config.rmsNormEps(), RMS_LOCAL_SIZE);

        batchPrefillLayer.task("batch_attn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP16,
                context, state.wrapXbFP16Batch, state.wrapXBatch,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.attnScaleBatch, dim);

        // Q, K, V in ONE tensor-core launch → packed [q|k|v] rows (stride qDim+2*kvDim)
        batchPrefillLayer.task("qkvProj",
                TransformerBatchPrefillKernels::gemmMMAQKVQ8,
                context, state.wrapXbFP16Batch,
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                state.qkvResultBatch, paddedBatch, qDim, kvDim, dim);

        // Qwen3: per-head RMS norm on Q and K before RoPE
        batchPrefillLayer.task("batch_qk_rmsnorm",
                Qwen3Kernels::batchedFusedQKRmsNormPacked,
                context, state.qkvResultBatch,
                weights.rms_att_QNormLayered[layerIndex].asFloatArray(),
                weights.rms_att_KNormLayered[layerIndex].asFloatArray(),
                config.numberOfHeads(), nHeadKv, nEmbdHead,
                qDim, kvDim, config.rmsNormEps());

        batchPrefillLayer.task("batch_rope_kv",
                Qwen3Kernels::batchedRopeWithKVCacheQwen3Packed,
                context, state.batchStartPosHolder,
                state.qkvResultBatch,
                state.wrapKeyCache, state.wrapValueCache,
                kvDim, nEmbdHead, layerIndex, config.contextLength(), qDim);

        // Register-partitioned flash attention over the packed buffer.
        // The 'dim' parameter doubles as the packed-Q stride base and the
        // attnOutFP16 row width — both are qDim for Qwen3.
        batchPrefillLayer.task("batch_attention",
                TransformerBatchPrefillKernels::batchedFlashAttentionFP16Out,
                context, state.batchStartPosHolder,
                state.qkvResultBatch, state.wrapKeyCache, state.wrapValueCache,
                state.attnOutFP16,
                config.numberOfHeads(), nEmbdHead,
                kvDim, gqa, layerIndex, config.contextLength(), qDim);

        // Output projection: [M=batch, N=dim, K=qDim]
        batchPrefillLayer.task("woProj", TransformerBatchPrefillKernels::gemmMMAQ8,
                context, state.attnOutFP16,
                weights.woLayered[layerIndex].asByteArray(),
                state.woOut, paddedBatch, dim, qDim);

        // ── FFN Block ──────────────────────────────────────────────────────────
        batchPrefillLayer.task("batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceFusedResidual,
                context, state.wrapXBatch, state.woOut, state.ffnScaleBatch,
                dim, config.rmsNormEps(), RMS_LOCAL_SIZE);

        batchPrefillLayer.task("batch_ffn_rms_apply",
                TransformerBatchPrefillKernels::batchedFFNRmsApplyFP16,
                context, state.normedXFFNFP16, state.wrapXBatch,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.ffnScaleBatch, dim);

        batchPrefillLayer.task("gateUpProj",
                TransformerBatchPrefillKernels::gemmMMAGateUpQ8,
                context, state.normedXFFNFP16,
                weights.w1Layered[layerIndex].asByteArray(),
                weights.w3Layered[layerIndex].asByteArray(),
                state.gateUpResultBatch, paddedBatch, hidDim, dim);

        batchPrefillLayer.task("swiglu",
                TransformerBatchPrefillKernels::batchedFFNSwiGLUFP16Packed,
                        context, state.wrapHbFP16Batch, state.gateUpResultBatch, hidDim)
                .task("w2Proj", TransformerBatchPrefillKernels::gemmMMAQ8,
                        context, state.wrapHbFP16Batch,
                        weights.w2Layered[layerIndex].asByteArray(),
                        state.w2Out, paddedBatch, dim, hidDim)
                .task("w2Resid", TransformerBatchPrefillKernels::batchedResidualAddFP32,
                        context, state.wrapXBatch, state.w2Out);

        batchPrefillLayer.persistOnDevice(state.wrapXBatch, state.wrapKeyCache, state.wrapValueCache);

        return batchPrefillLayer;
    }
    // @formatter:on

    // gemmMMA family: 256 threads/block (1D within block), grid over M- and N-blocks.
    static WorkerGrid mmaGrid(int paddedM, int N) {
        int mBlocks = paddedM / 128;   // BM
        int nBlocks = N / 128;         // BN
        WorkerGrid2D g = new WorkerGrid2D(mBlocks * 256, nBlocks);
        g.setLocalWork(256, 1, 1);
        return g;
    }

    static WorkerGrid elementwiseGrid(int n) {   // n must be a multiple of 256
        WorkerGrid1D g = new WorkerGrid1D(n);
        g.setLocalWork(256, 1, 1);
        return g;
    }

    /** Registers all batch layer workers in the shared {@link GridScheduler}. */
    public void updateGridScheduler(GridScheduler scheduler) {
        int dim     = config.dim();
        int hidDim  = config.hiddenDim();
        int nHeads  = config.numberOfHeads();

        WorkerGrid rmsWorker = WorkerGridFactory.genericWorker(
                batchSize * RMS_LOCAL_SIZE, RMS_LOCAL_SIZE);

        WorkerGrid rmsApplyWorker    = WorkerGridFactory.genericWorker(batchSize * dim, 256);
        WorkerGrid ffnRmsApplyWorker = WorkerGridFactory.genericWorker(batchSize * dim, 256);

        // Q/K per-head RMS norm: one nEmbdHead-thread workgroup per (token, head)
        WorkerGrid qkRmsNormWorker = WorkerGridFactory.genericWorker(
                batchSize * (nHeads + nHeadKv) * nEmbdHead, nEmbdHead);

        // Split-half RoPE: B*(qDim/2) threads
        int ropeGlobal = batchSize * (qDim / 2);
        int ropeLocal  = Math.min(512, ropeGlobal);
        while (ropeLocal > 1 && ropeGlobal % ropeLocal != 0) ropeLocal--;
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(ropeGlobal, ropeLocal);

        // Attention: B*nHeads workgroups × min(nEmbdHead,128) threads
        int attnLocal = Math.min(nEmbdHead, 128);
        WorkerGrid attnWorker = WorkerGridFactory.genericWorker(
                batchSize * nHeads * attnLocal, attnLocal);

        // MMA grids
        WorkerGrid mmaQkvWorker    = mmaGrid(paddedBatch, qDim + 2 * kvDim);  // fused QKV
        WorkerGrid mmaDimWorker    = mmaGrid(paddedBatch, dim);               // woProj, w2Proj
        WorkerGrid mmaGateUpWorker = mmaGrid(paddedBatch, 2 * hidDim);        // fused W1/W3

        WorkerGrid ewDimWorker = elementwiseGrid(batchSize * dim);     // w2Resid
        WorkerGrid ewHidWorker = elementwiseGrid(batchSize * hidDim);  // swiglu

        for (int i = 0; i < config.numberOfLayers(); i++) {
            String p = "batchPrefillLayer_" + i + ".";
            scheduler.addWorkerGrid(p + "batch_attn_rms",       rmsWorker);
            scheduler.addWorkerGrid(p + "batch_attn_rms_apply", rmsApplyWorker);
            scheduler.addWorkerGrid(p + "qkvProj",              mmaQkvWorker);
            scheduler.addWorkerGrid(p + "batch_qk_rmsnorm",     qkRmsNormWorker);
            scheduler.addWorkerGrid(p + "batch_rope_kv",        ropeWorker);
            scheduler.addWorkerGrid(p + "batch_attention",      attnWorker);
            scheduler.addWorkerGrid(p + "woProj",               mmaDimWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_rms",        rmsWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_rms_apply",  ffnRmsApplyWorker);
            scheduler.addWorkerGrid(p + "gateUpProj",           mmaGateUpWorker);
            scheduler.addWorkerGrid(p + "swiglu",               ewHidWorker);
            scheduler.addWorkerGrid(p + "w2Proj",               mmaDimWorker);
            scheduler.addWorkerGrid(p + "w2Resid",              ewDimWorker);
        }
    }

    public List<ImmutableTaskGraph> getLayerImmutableTaskGraphs() { return layerITGs; }
    public String getLastLayerTaskGraphID()                        { return lastLayerTaskGraphID; }
    public KernelContext getContext()                               { return context; }
}
