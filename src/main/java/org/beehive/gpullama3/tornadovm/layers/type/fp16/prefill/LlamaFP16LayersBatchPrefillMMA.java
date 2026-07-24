package org.beehive.gpullama3.tornadovm.layers.type.fp16.prefill;

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
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.List;
import java.util.stream.IntStream;

/**
 * Batched-prefill transformer-layer TaskGraphs for the unified batched prefill-decode plan
 * ({@link org.beehive.gpullama3.tornadovm.TornadoVMMasterPlanBatchPrefillDecode}).
 *
 * <p>One {@link ImmutableTaskGraph} per transformer layer, each processing
 * {@code batchSize} tokens simultaneously via {@link TransformerBatchPrefillKernels}.</p>
 *
 * <p>Tensor-core layer pipeline (12 tasks, all GEMMs on MMA):</p>
 * <pre>
 *   batch_attn_rms        parallel RMS square-sum reduction (256 thr/token)
 *   batch_attn_rms_apply  RMS apply + FP16 quantize → wrapXbFP16Batch
 *   qkvProj               ONE fused MMA GEMM → packed qkvResultBatch [q|k|v]
 *   batch_rope_kv         RoPE over the packed buffer + KV cache write
 *   batch_attention       flash attention (register-partitioned P·V) → attnOutFP16
 *   woProj                MMA GEMM → woOut
 *   batch_ffn_rms         parallel RMS reduce FUSED with x += woOut
 *   batch_ffn_rms_apply   RMS apply + FP16 quantize → normedXFFNFP16
 *   gateUpProj            ONE fused MMA GEMM → packed gateUpResultBatch [gate|up]
 *   swiglu                SiLU(gate)*up over packed buffer → wrapHbFP16Batch
 *   w2Proj                MMA GEMM → w2Out
 *   w2Resid               x += w2Out
 * </pre>
 *
 * <p>KV cache ({@code wrapKeyCache}, {@code wrapValueCache}) is persisted on device
 * after every layer so the subsequent single-token decode layers can consume it.</p>
 */
public class LlamaFP16LayersBatchPrefillMMA implements BatchPrefillTransformerLayerTaskGraphs {

    // Local size for the parallel RMS reductions (one workgroup per token).
    static final int RMS_LOCAL_SIZE = 256;

    private final LlamaState state;
    private final LlamaTornadoWeights weights;
    private final LlamaConfiguration config;
    private final KernelContext context = new KernelContext();
    private final int batchSize;
    // GEMM M dimension rounded up to whole 128-row tiles (BM). The GEMM-adjacent
    // buffers in State are allocated at this padded size; rows >= batchSize are
    // computed but never consumed. All non-GEMM kernels use the true batchSize.
    private final int paddedBatch;
    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    public LlamaFP16LayersBatchPrefillMMA(LlamaState state, LlamaTornadoWeights weights,
                                          LlamaConfiguration config, int batchSize) {
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

        // ── Data Transfers ─────────────────────────────────────────────────────
        if (layerIndex == 0) {
            // batchStartPosHolder is set by host before each chunk → EVERY_EXECUTION
            batchPrefillLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION, state.batchStartPosHolder);
            // Allocate persistent GPU-side intermediates once
            batchPrefillLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.attnScaleBatch, state.ffnScaleBatch,
                    state.wrapXbFP16Batch,
                    state.qkvResultBatch,
                    state.wrapKeyCache, state.wrapValueCache,
                    state.normedXFFNFP16, state.gateUpResultBatch,
                    state.attnOutFP16, state.woOut, state.wrapHbFP16Batch, state.w2Out);
            // wrapXBatch produced by the prefillActivation graph and persists in device memory
            // to consume it from there we should use the explicit uniqueTaskGraph name
            // the no-arg form would use current graph name, which causes NPE without CUDA Graphs
            batchPrefillLayer.consumeFromDevice("prefillActivation", state.wrapXBatch);
        } else {
            // for the same reasons as above, we should use the explicit uniqueTaskGraph name to consume
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
        batchPrefillLayer.task("batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context, state.wrapXBatch, state.attnScaleBatch,
                dim, config.rmsNormEps(), RMS_LOCAL_SIZE);

        batchPrefillLayer.task("batch_attn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP16,
                context, state.wrapXbFP16Batch, state.wrapXBatch,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.attnScaleBatch, dim);

        // Q, K, V in ONE tensor-core launch → packed [q|k|v] rows.
        // Grid spans (dim + 2*kvDim)/128 N-blocks: no grid starvation on the
        // skinny GQA projections, and the A operand is read once, not thrice.
        batchPrefillLayer.task("qkvProj",
                TransformerBatchPrefillKernels::gemmMMAQKV,
                context, state.wrapXbFP16Batch,
                weights.wqLayered[layerIndex].asHalfFloatArray(),
                weights.wkLayered[layerIndex].asHalfFloatArray(),
                weights.wvLayered[layerIndex].asHalfFloatArray(),
                state.qkvResultBatch, paddedBatch, dim, kvDim, dim);

        batchPrefillLayer.task("batch_rope_kv",
                TransformerBatchPrefillKernels::batchedRopeWithKVCachePacked,
                context, state.batchStartPosHolder,
                state.qkvResultBatch,
                state.wrapKeyCache, state.wrapValueCache,
                kvDim, config.headSize(), layerIndex, config.contextLength(), dim);

        // Register-partitioned P·V accumulation + direct FP16 emission
        // (replaces batchedFlashAttention + attnCast).
        batchPrefillLayer.task("batch_attention",
                TransformerBatchPrefillKernels::batchedFlashAttentionFP16Out,
                context, state.batchStartPosHolder,
                state.qkvResultBatch, state.wrapKeyCache, state.wrapValueCache,
                state.attnOutFP16,
                config.numberOfHeads(), config.headSize(),
                kvDim, config.kvMul(), layerIndex, config.contextLength(), dim);

        batchPrefillLayer.task("woProj", TransformerBatchPrefillKernels::gemmMMA,
                context, state.attnOutFP16,
                weights.woLayered[layerIndex].asHalfFloatArray(),
                state.woOut, paddedBatch, dim, dim);

        // ── FFN Block ──────────────────────────────────────────────────────────
        // x += woOut is fused into the FFN RMS reduction (drops the woResid task).
        batchPrefillLayer.task("batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceFusedResidual,
                context, state.wrapXBatch, state.woOut, state.ffnScaleBatch,
                dim, config.rmsNormEps(), RMS_LOCAL_SIZE);

        batchPrefillLayer.task("batch_ffn_rms_apply",
                TransformerBatchPrefillKernels::batchedFFNRmsApplyFP16,
                context, state.normedXFFNFP16, state.wrapXBatch,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.ffnScaleBatch, dim);

        // W1 and W3 in ONE tensor-core launch → packed [gate|up] rows.
        batchPrefillLayer.task("gateUpProj",
                TransformerBatchPrefillKernels::gemmMMAGateUp,
                context, state.normedXFFNFP16,
                weights.w1Layered[layerIndex].asHalfFloatArray(),
                weights.w3Layered[layerIndex].asHalfFloatArray(),
                state.gateUpResultBatch, paddedBatch, hidDim, dim);

        batchPrefillLayer.task("swiglu",
                TransformerBatchPrefillKernels::batchedFFNSwiGLUFP16Packed,
                        context, state.wrapHbFP16Batch, state.gateUpResultBatch, hidDim)
                .task("w2Proj", TransformerBatchPrefillKernels::gemmMMA,
                        context, state.wrapHbFP16Batch,
                        weights.w2Layered[layerIndex].asHalfFloatArray(),
                        state.w2Out, batchSize, dim, hidDim)
                .task("w2Resid", TransformerBatchPrefillKernels::batchedResidualAddFP32,
                        context, state.wrapXBatch, state.w2Out);

        // Persist wrapXBatch for the next layer, and KV cache so the decode
        // layers can consume it via the activation graph pass-through.
        batchPrefillLayer.persistOnDevice(state.wrapXBatch, state.wrapKeyCache, state.wrapValueCache);

        return batchPrefillLayer;
    }
    // @formatter:on

    // gemmMMA family: 256 threads/block (1D within block), grid over M- and N-blocks.
    static WorkerGrid mmaGrid(int paddedM, int N) {
        int mBlocks = paddedM / 128;   // BM
        int nBlocks = N / 128;         // BN
        WorkerGrid2D g = new WorkerGrid2D(mBlocks * 256, nBlocks);
        g.setLocalWork(256, 1, 1);     // groupIdx∈[0,mBlocks), groupIdy∈[0,nBlocks)
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
        int kvDim   = config.kvDim();
        int hidDim  = config.hiddenDim();
        int nHeads  = config.numberOfHeads();
        int headSz  = config.headSize();

        // Parallel RMS reductions: one 256-thread workgroup per batch token
        WorkerGrid rmsWorker = WorkerGridFactory.genericWorker(
                batchSize * RMS_LOCAL_SIZE, RMS_LOCAL_SIZE);

        // RMS apply: B*dim threads, local=256 (dim is always a multiple of 256 for LLaMA)
        WorkerGrid rmsApplyWorker    = WorkerGridFactory.genericWorker(batchSize * dim, 256);
        WorkerGrid ffnRmsApplyWorker = WorkerGridFactory.genericWorker(batchSize * dim, 256);

        // RoPE+KV cache: B*(dim/2) threads, local=512
        int ropeGlobal = batchSize * (dim / 2);
        int ropeLocal  = Math.min(512, ropeGlobal);
        while (ropeLocal > 1 && ropeGlobal % ropeLocal != 0) ropeLocal--;
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(ropeGlobal, ropeLocal);

        // Attention (flash): B*nHeads workgroups × min(headSize,128) threads.
        // The kernel requires headSize <= 2*localSize.
        int attnLocal = Math.min(headSz, 128);
        WorkerGrid attnWorker = WorkerGridFactory.genericWorker(
                batchSize * nHeads * attnLocal, attnLocal);

        // MMA grids
        WorkerGrid mmaQkvWorker    = mmaGrid(paddedBatch, dim + 2 * kvDim);  // fused QKV
        WorkerGrid mmaDimWorker    = mmaGrid(paddedBatch, dim);              // woProj, w2Proj
        WorkerGrid mmaGateUpWorker = mmaGrid(paddedBatch, 2 * hidDim);       // fused W1/W3

        // Elementwise grids (one thread per valid element; dim & hidDim are mult. of 256)
        WorkerGrid ewDimWorker = elementwiseGrid(batchSize * dim);     // w2Resid
        WorkerGrid ewHidWorker = elementwiseGrid(batchSize * hidDim);  // swiglu

        for (int i = 0; i < config.numberOfLayers(); i++) {
            String p = "batchPrefillLayer_" + i + ".";
            scheduler.addWorkerGrid(p + "batch_attn_rms",       rmsWorker);
            scheduler.addWorkerGrid(p + "batch_attn_rms_apply", rmsApplyWorker);
            scheduler.addWorkerGrid(p + "qkvProj",              mmaQkvWorker);
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
