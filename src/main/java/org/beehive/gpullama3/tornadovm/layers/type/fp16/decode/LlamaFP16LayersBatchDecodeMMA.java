package org.beehive.gpullama3.tornadovm.layers.type.fp16.decode;

import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerBatchPrefillKernels;
import org.beehive.gpullama3.tornadovm.scheduling.WorkerGridFactory;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.List;
import java.util.stream.IntStream;

/**
 * Batched-DECODE transformer-layer TaskGraphs: B independent sequences advancing
 * one token per step, each with its own KV region and its own position.
 *
 * <p>Structurally identical to {@code LlamaFP16LayersBatchPrefillMMA} (same 12-task
 * MMA pipeline: RMS → fused QKV GEMM → RoPE+KV → flash attention → Wo → FFN RMS →
 * fused gate/up GEMM → SwiGLU → W2 → residual). The only differences are the two
 * KV-addressing kernels, which take a per-slot position array and a B-sized KV
 * cache ({@code batchIdx} stride = {@code numLayers*contextLength*kvDim}):</p>
 * <ul>
 *   <li>{@code batch_rope_kv} → {@link TransformerBatchPrefillKernels#batchedDecodeRopeWithKVCachePacked}</li>
 *   <li>{@code batch_attention} → {@link TransformerBatchPrefillKernels#batchedDecodeAttentionFP16Out}</li>
 * </ul>
 *
 * <p>The KV cache and {@code seqPositions} are supplied by the engine (not State),
 * so the context length can be capped independently to fit B×L×ctx×kvDim in VRAM.</p>
 */
public class LlamaFP16LayersBatchDecodeMMA {

    static final int RMS_LOCAL_SIZE = 256;

    private final LlamaState state;
    private final LlamaTornadoWeights weights;
    private final LlamaConfiguration config;
    private final KernelContext context = new KernelContext();
    private final int batchSize;
    private final int paddedBatch;
    private final int decodeCtx;
    private final FloatArray keyCacheBatch;
    private final FloatArray valueCacheBatch;
    private final IntArray seqPositions;
    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    public LlamaFP16LayersBatchDecodeMMA(LlamaState state, LlamaTornadoWeights weights,
                                         LlamaConfiguration config, int batchSize, int decodeCtx,
                                         FloatArray keyCacheBatch, FloatArray valueCacheBatch,
                                         IntArray seqPositions) {
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.paddedBatch = (batchSize + 127) & ~127;
        this.decodeCtx = decodeCtx;
        this.keyCacheBatch = keyCacheBatch;
        this.valueCacheBatch = valueCacheBatch;
        this.seqPositions = seqPositions;
        this.layerITGs = IntStream.range(0, config.numberOfLayers())
                .mapToObj(this::createLayerTaskGraph)
                .map(TaskGraph::snapshot)
                .toList();
    }

    // @formatter:off
    private TaskGraph createLayerTaskGraph(int layerIndex) {
        String graphName = "batchDecodeLayer_" + layerIndex;
        if (layerIndex == config.numberOfLayers() - 1) lastLayerTaskGraphID = graphName;

        TaskGraph g = new TaskGraph(graphName);

        if (layerIndex == 0) {
            // Per-slot positions change every decode step.
            g.transferToDevice(DataTransferMode.EVERY_EXECUTION, seqPositions);
            g.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.attnScaleBatch, state.ffnScaleBatch,
                    state.wrapXbFP16Batch,
                    state.qkvResultBatch,
                    keyCacheBatch, valueCacheBatch,
                    state.normedXFFNFP16, state.gateUpResultBatch,
                    state.attnOutFP16, state.woOut, state.wrapHbFP16Batch, state.w2Out);
            g.consumeFromDevice("prefillActivation", state.wrapXBatch);
        } else {
            String pred = "batchDecodeLayer_" + (layerIndex - 1);
            g.consumeFromDevice(pred,
                    context,
                    state.wrapXBatch,
                    seqPositions,
                    state.attnScaleBatch, state.ffnScaleBatch,
                    state.wrapXbFP16Batch,
                    state.qkvResultBatch,
                    keyCacheBatch, valueCacheBatch,
                    state.normedXFFNFP16, state.gateUpResultBatch,
                    state.attnOutFP16, state.woOut, state.wrapHbFP16Batch, state.w2Out);
        }

        g.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asHalfFloatArray(),
                weights.wkLayered[layerIndex].asHalfFloatArray(),
                weights.wvLayered[layerIndex].asHalfFloatArray(),
                weights.woLayered[layerIndex].asHalfFloatArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].asHalfFloatArray(),
                weights.w2Layered[layerIndex].asHalfFloatArray(),
                weights.w3Layered[layerIndex].asHalfFloatArray());

        int dim    = config.dim();
        int kvDim  = config.kvDim();
        int hidDim = config.hiddenDim();

        g.task("batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context, state.wrapXBatch, state.attnScaleBatch,
                dim, config.rmsNormEps(), RMS_LOCAL_SIZE);

        g.task("batch_attn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP16,
                context, state.wrapXbFP16Batch, state.wrapXBatch,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.attnScaleBatch, dim);

        g.task("qkvProj",
                TransformerBatchPrefillKernels::gemmMMAQKV,
                context, state.wrapXbFP16Batch,
                weights.wqLayered[layerIndex].asHalfFloatArray(),
                weights.wkLayered[layerIndex].asHalfFloatArray(),
                weights.wvLayered[layerIndex].asHalfFloatArray(),
                state.qkvResultBatch, paddedBatch, dim, kvDim, dim);

        // DECODE: per-slot position + per-slot KV region.
        g.task("batch_rope_kv",
                TransformerBatchPrefillKernels::batchedDecodeRopeWithKVCachePacked,
                context, seqPositions,
                state.qkvResultBatch,
                keyCacheBatch, valueCacheBatch,
                kvDim, config.headSize(), layerIndex, config.numberOfLayers(), decodeCtx, dim);

        g.task("batch_attention",
                TransformerBatchPrefillKernels::batchedDecodeAttentionFP16Out,
                context, seqPositions,
                state.qkvResultBatch, keyCacheBatch, valueCacheBatch,
                state.attnOutFP16,
                config.numberOfHeads(), config.headSize(),
                kvDim, config.kvMul(), layerIndex, config.numberOfLayers(), decodeCtx, dim);

        g.task("woProj", TransformerBatchPrefillKernels::gemmMMA,
                context, state.attnOutFP16,
                weights.woLayered[layerIndex].asHalfFloatArray(),
                state.woOut, paddedBatch, dim, dim);

        g.task("batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceFusedResidual,
                context, state.wrapXBatch, state.woOut, state.ffnScaleBatch,
                dim, config.rmsNormEps(), RMS_LOCAL_SIZE);

        g.task("batch_ffn_rms_apply",
                TransformerBatchPrefillKernels::batchedFFNRmsApplyFP16,
                context, state.normedXFFNFP16, state.wrapXBatch,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.ffnScaleBatch, dim);

        g.task("gateUpProj",
                TransformerBatchPrefillKernels::gemmMMAGateUp,
                context, state.normedXFFNFP16,
                weights.w1Layered[layerIndex].asHalfFloatArray(),
                weights.w3Layered[layerIndex].asHalfFloatArray(),
                state.gateUpResultBatch, paddedBatch, hidDim, dim);

        g.task("swiglu",
                TransformerBatchPrefillKernels::batchedFFNSwiGLUFP16Packed,
                        context, state.wrapHbFP16Batch, state.gateUpResultBatch, hidDim)
         .task("w2Proj", TransformerBatchPrefillKernels::gemmMMA,
                        context, state.wrapHbFP16Batch,
                        weights.w2Layered[layerIndex].asHalfFloatArray(),
                        state.w2Out, batchSize, dim, hidDim)
         .task("w2Resid", TransformerBatchPrefillKernels::batchedResidualAddFP32,
                        context, state.wrapXBatch, state.w2Out);

        g.persistOnDevice(state.wrapXBatch, keyCacheBatch, valueCacheBatch);
        return g;
    }
    // @formatter:on

    static WorkerGrid mmaGrid(int paddedM, int N) {
        int mBlocks = paddedM / 128;
        int nBlocks = N / 128;
        WorkerGrid2D grid = new WorkerGrid2D(mBlocks * 256, nBlocks);
        grid.setLocalWork(256, 1, 1);
        return grid;
    }

    static WorkerGrid elementwiseGrid(int n) {
        WorkerGrid1D grid = new WorkerGrid1D(n);
        grid.setLocalWork(256, 1, 1);
        return grid;
    }

    public void updateGridScheduler(GridScheduler scheduler) {
        int dim    = config.dim();
        int kvDim  = config.kvDim();
        int hidDim = config.hiddenDim();
        int nHeads = config.numberOfHeads();
        int headSz = config.headSize();

        WorkerGrid rmsWorker = WorkerGridFactory.genericWorker(batchSize * RMS_LOCAL_SIZE, RMS_LOCAL_SIZE);
        WorkerGrid rmsApplyWorker    = WorkerGridFactory.genericWorker(batchSize * dim, 256);
        WorkerGrid ffnRmsApplyWorker = WorkerGridFactory.genericWorker(batchSize * dim, 256);

        int ropeGlobal = batchSize * (dim / 2);
        int ropeLocal  = Math.min(512, ropeGlobal);
        while (ropeLocal > 1 && ropeGlobal % ropeLocal != 0) ropeLocal--;
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(ropeGlobal, ropeLocal);

        int attnLocal = Math.min(headSz, 128);
        WorkerGrid attnWorker = WorkerGridFactory.genericWorker(batchSize * nHeads * attnLocal, attnLocal);

        WorkerGrid mmaQkvWorker    = mmaGrid(paddedBatch, dim + 2 * kvDim);
        WorkerGrid mmaDimWorker    = mmaGrid(paddedBatch, dim);
        WorkerGrid mmaGateUpWorker = mmaGrid(paddedBatch, 2 * hidDim);

        WorkerGrid ewDimWorker = elementwiseGrid(batchSize * dim);
        WorkerGrid ewHidWorker = elementwiseGrid(batchSize * hidDim);

        for (int i = 0; i < config.numberOfLayers(); i++) {
            String p = "batchDecodeLayer_" + i + ".";
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
    public KernelContext getContext()                              { return context; }
}
