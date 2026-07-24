package org.beehive.gpullama3.tornadovm.layers.type.q8_0;

import org.beehive.gpullama3.inference.state.Gemma4State;
import org.beehive.gpullama3.inference.weights.tornado.Gemma4TornadoWeights;
import org.beehive.gpullama3.model.gemma4.Gemma4Configuration;
import org.beehive.gpullama3.tensor.tornado.TornadoTensor;
import org.beehive.gpullama3.tornadovm.kernels.Gemma4Kernels;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernelsLayered;
import org.beehive.gpullama3.tornadovm.scheduling.WorkerGridFactory;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;
import org.beehive.gpullama3.tornadovm.layers.AbstractTransformerLayerTaskGraphs;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Gemma4Q8_0FFNLayers: Q8_0 transformer-layer task graphs for the Gemma 4 architecture.
 *
 * <p>Structurally identical to {@code Gemma4FP16FFNLayers}. The main attention/FFN projections
 * (Q/K/V/O, FFN gate/up/down) are Q8_0 byte arrays consumed via {@code matrixVectorGenericQ8Byte} /
 * {@link Gemma4Kernels#fusedGateUpGeGLUQ8}. The per-layer-embedding (PLE) projections
 * ({@code inp_gate}, {@code proj}, {@code per_layer_model_proj}) are <em>not</em> uniformly Q8_0 in
 * a Q8_0 GGUF -- they may be stored at F32 or F16 -- so they are routed through
 * {@link #addProjection} which selects the matmul kernel from each tensor's {@link TornadoTensor#type()}.
 * The (un-quantized) norm and scale weights remain F32.</p>
 *
 * <p>Gemma 4's layers differ enough from the "Llama-like" models that nothing here is fused the way
 * {@code Qwen3FP16FFNLayers} is -- each layer carries its own Q/K-norm and a "sandwich" of pre/post
 * normalization around both attention and FFN, attention head dimensions and RoPE tables differ
 * between sliding-window and full-attention layers (and are baked into each layer's task graph as
 * compile-time constants -- see {@link Gemma4Configuration#headDim}), some layers reuse an earlier
 * layer's KV cache instead of computing their own, the FFN uses GeGLU, and every layer mixes in a
 * per-layer embedding (PLE) contribution. See {@link org.beehive.gpullama3.inference.InferenceCore#forwardJavaGemma4}
 * for the reference computation each task mirrors.</p>
 *
 * <p>Layer 0's task graph additionally carries one-time-per-token setup that the reference
 * implementation performs before the layer loop: scaling the token embedding by {@code sqrt(dim)},
 * and computing the per-layer-embedding inputs ({@code perLayerInputs}) from the per-layer model
 * projection and the (host-gathered) per-layer token embedding row -- see {@link #appendPLESetupTasks}.</p>
 */
public class Gemma4Q8_0FFNLayers extends AbstractTransformerLayerTaskGraphs<Gemma4TornadoWeights, Gemma4Configuration> {

    /** Local memory size for per-head Q/K/V-norm reductions; must evenly divide both head dimensions (256, 512). */
    private static final int HEAD_NORM_LOCAL_SIZE = 64;

    private final Gemma4State gemma4State;
    private final int nHead;
    private final int nHeadKv;
    private final int kvMul;
    private final int dim;
    private final int nEmbdPerLayer;
    private final int perLayerTotal;
    private final float embedScale;
    private final float perLayerTokEmbedScale;
    private final float perLayerProjScale;
    private final float perLayerInputScale;

    public Gemma4Q8_0FFNLayers(String taskGraphName, Gemma4State state, Gemma4TornadoWeights weights, Gemma4Configuration config, SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config, schedulerType);
        this.gemma4State = state;
        this.nHead = config.numberOfHeads();
        this.nHeadKv = config.numberOfKeyValueHeads();
        this.kvMul = config.kvMul();
        this.dim = config.dim();
        this.nEmbdPerLayer = config.embeddingLengthPerLayer();
        this.perLayerTotal = config.numberOfLayers() * nEmbdPerLayer;
        this.embedScale = (float) Math.sqrt(dim);
        this.perLayerTokEmbedScale = (float) Math.sqrt(nEmbdPerLayer);
        this.perLayerProjScale = (float) (1.0 / Math.sqrt(dim));
        this.perLayerInputScale = (float) (1.0 / Math.sqrt(2.0));
        setupFFNLayers();
    }

    // ═══════════════════════════════════════════════════════════════════════════════════
    //                                  TASK GRAPH
    // ═══════════════════════════════════════════════════════════════════════════════════

    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        var taskGraphName = "layer_" + layerIndex;
        final int headDim = config.headDim(layerIndex);
        final boolean isSwa = config.isSwa(layerIndex);
        final boolean hasOwnKv = config.hasOwnKv(layerIndex);
        final int qDim = nHead * headDim;
        final int kvDim = nHeadKv * headDim;
        final int ffnLen = config.feedForwardLength(layerIndex);
        final int cacheBaseOffset = gemma4State.cacheLayerBaseOffset[layerIndex];
        final int windowSize = isSwa ? config.slidingWindowSize() : config.contextLength();
        final var freqCisReal = (isSwa ? weights.freqCisRealSwa : weights.freqCisRealFull).asFloatArray();
        final var freqCisImag = (isSwa ? weights.freqCisImagSwa : weights.freqCisImagFull).asFloatArray();
        final int peOffset = layerIndex * nEmbdPerLayer;

        var unifiedLayer = new TaskGraph(taskGraphName);
        unifiedLayer.consumeFromDevice(gemma4State.wrapX);
        unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                weights.woLayered[layerIndex].asByteArray(),
                weights.attnQNorm[layerIndex].asFloatArray(),
                weights.attnKNorm[layerIndex].asFloatArray(),
                weights.attnPostNorm[layerIndex].asFloatArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].asByteArray(),
                weights.w3Layered[layerIndex].asByteArray(),
                weights.w2Layered[layerIndex].asByteArray(),
                weights.ffnPostNorm[layerIndex].asFloatArray(),
                weightArray(weights.perLayerInpGate[layerIndex]),
                weightArray(weights.perLayerProj[layerIndex]),
                weights.perLayerPostNorm[layerIndex].asFloatArray());
        if (weights.layerOutputScale[layerIndex] != null) {
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, weights.layerOutputScale[layerIndex].asFloatArray());
        }
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);

        if (layerIndex == 0) {
            appendPLESetupTasks(unifiedLayer);
        }

        // ═══════════════════════════════════ ATTENTION ═══════════════════════════════════
        unifiedLayer.task("attn_norm_reduce",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                context, gemma4State.temp, gemma4State.wrapX, dim, config.rmsNormEps(), gemma4State.localSize);
        if (shouldUseFinalNormalization()) {
            unifiedLayer.task("attn_norm_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context, gemma4State.temp, dim, config.rmsNormEps());
        }
        unifiedLayer.task("attn_norm_apply",
                Gemma4Kernels::applyRmsNorm,
                context, gemma4State.wrapXb, gemma4State.wrapX, weights.rms_att_weightLayered[layerIndex].asFloatArray(), gemma4State.temp, dim);

        unifiedLayer.task("q_proj",
                TransformerComputeKernelsLayered::matrixVectorGenericQ8Byte,
                context, gemma4State.wrapXb, gemma4State.wrapQ, weights.wqLayered[layerIndex].asByteArray(), dim, qDim, LOCAL_WORK_GROUP_SIZE_ALLOC);
        unifiedLayer.task("q_norm",
                Gemma4Kernels::rmsNormPerHead,
                context, gemma4State.wrapQ, weights.attnQNorm[layerIndex].asFloatArray(), nHead, headDim, HEAD_NORM_LOCAL_SIZE, config.rmsNormEps());

        if (hasOwnKv) {
            unifiedLayer.task("k_proj",
                    TransformerComputeKernelsLayered::matrixVectorGenericQ8Byte,
                    context, gemma4State.wrapXb, gemma4State.wrapK, weights.wkLayered[layerIndex].asByteArray(), dim, kvDim, LOCAL_WORK_GROUP_SIZE_ALLOC);
            unifiedLayer.task("k_norm",
                    Gemma4Kernels::rmsNormPerHead,
                    context, gemma4State.wrapK, weights.attnKNorm[layerIndex].asFloatArray(), nHeadKv, headDim, HEAD_NORM_LOCAL_SIZE, config.rmsNormEps());
            unifiedLayer.task("v_proj",
                    TransformerComputeKernelsLayered::matrixVectorGenericQ8Byte,
                    context, gemma4State.wrapXb, gemma4State.wrapV, weights.wvLayered[layerIndex].asByteArray(), dim, kvDim, LOCAL_WORK_GROUP_SIZE_ALLOC);
            unifiedLayer.task("v_norm",
                    Gemma4Kernels::rmsNormPerHeadNoWeight,
                    context, gemma4State.wrapV, nHeadKv, headDim, HEAD_NORM_LOCAL_SIZE, config.rmsNormEps());
            unifiedLayer.task("rope_and_cache",
                    Gemma4Kernels::ropeNeoxRotateAndCacheCopy,
                    context, gemma4State.positionHolder, gemma4State.wrapQ, gemma4State.wrapK, gemma4State.wrapV,
                    gemma4State.wrapKeyCache, gemma4State.wrapValueCache, freqCisReal, freqCisImag,
                    nHeadKv, headDim, kvDim, cacheBaseOffset);
        } else {
            unifiedLayer.task("rope_q_only",
                    Gemma4Kernels::ropeNeoxRotateQOnly,
                    context, gemma4State.positionHolder, gemma4State.wrapQ, freqCisReal, freqCisImag, headDim);
        }

        unifiedLayer.task("attention",
                Gemma4Kernels::attentionWithSlidingWindow,
                gemma4State.wrapQ, gemma4State.wrapKeyCache, gemma4State.wrapValueCache, gemma4State.wrapXb, gemma4State.wrapAtt,
                nHead, headDim, kvDim, kvMul, gemma4State.positionHolder, cacheBaseOffset, windowSize, config.contextLength());

        unifiedLayer.task("wo_proj",
                TransformerComputeKernelsLayered::matrixVectorGenericQ8Byte,
                context, gemma4State.wrapXb, gemma4State.wrapXb2, weights.woLayered[layerIndex].asByteArray(), qDim, dim, LOCAL_WORK_GROUP_SIZE_ALLOC);

        unifiedLayer.task("post_attn_reduce",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                context, gemma4State.tempPostAttn, gemma4State.wrapXb2, dim, config.rmsNormEps(), gemma4State.localSize);
        if (shouldUseFinalNormalization()) {
            unifiedLayer.task("post_attn_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context, gemma4State.tempPostAttn, dim, config.rmsNormEps());
        }
        unifiedLayer.task("post_attn_apply",
                Gemma4Kernels::rmsNormApplyWithResidual,
                context, gemma4State.wrapX, gemma4State.wrapXb2, weights.attnPostNorm[layerIndex].asFloatArray(), gemma4State.tempPostAttn, dim);

        // ═══════════════════════════════════════ FFN ═════════════════════════════════════
        unifiedLayer.task("ffn_norm_reduce",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                context, gemma4State.tempFFN, gemma4State.wrapX, dim, config.rmsNormEps(), gemma4State.localSize);
        if (shouldUseFinalNormalization()) {
            unifiedLayer.task("ffn_norm_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context, gemma4State.tempFFN, dim, config.rmsNormEps());
        }
        unifiedLayer.task("ffn_norm_apply",
                Gemma4Kernels::applyRmsNorm,
                context, gemma4State.wrapXb, gemma4State.wrapX, weights.rms_ffn_weightLayered[layerIndex].asFloatArray(), gemma4State.tempFFN, dim);

        unifiedLayer.task("ffn_gate_up",
                Gemma4Kernels::fusedGateUpGeGLUQ8,
                context, gemma4State.wrapXb, gemma4State.wrapHb, weights.w1Layered[layerIndex].asByteArray(), weights.w3Layered[layerIndex].asByteArray(),
                dim, ffnLen, LOCAL_WORK_GROUP_SIZE_ALLOC);
        unifiedLayer.task("ffn_down_proj",
                TransformerComputeKernelsLayered::matrixVectorGenericQ8Byte,
                context, gemma4State.wrapHb, gemma4State.wrapXb2, weights.w2Layered[layerIndex].asByteArray(), ffnLen, dim, LOCAL_WORK_GROUP_SIZE_ALLOC);

        unifiedLayer.task("post_ffn_reduce",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                context, gemma4State.tempPostFfn, gemma4State.wrapXb2, dim, config.rmsNormEps(), gemma4State.localSize);
        if (shouldUseFinalNormalization()) {
            unifiedLayer.task("post_ffn_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context, gemma4State.tempPostFfn, dim, config.rmsNormEps());
        }
        unifiedLayer.task("post_ffn_apply",
                Gemma4Kernels::rmsNormApplyWithResidual,
                context, gemma4State.wrapX, gemma4State.wrapXb2, weights.ffnPostNorm[layerIndex].asFloatArray(), gemma4State.tempPostFfn, dim);

        // ═══════════════════════════ PER-LAYER EMBEDDING (PLE) ═══════════════════════════
        addProjection(unifiedLayer, "ple_gate_proj", gemma4State.wrapX, gemma4State.wrapPerLayerGate, weights.perLayerInpGate[layerIndex], dim, nEmbdPerLayer);
        unifiedLayer.task("ple_gate_gelu_mul",
                Gemma4Kernels::pleGateGeluMul,
                context, gemma4State.wrapPerLayerGate, gemma4State.wrapPerLayerInputs, peOffset, nEmbdPerLayer);
        addProjection(unifiedLayer, "ple_proj", gemma4State.wrapPerLayerGate, gemma4State.wrapPerLayerOut, weights.perLayerProj[layerIndex], nEmbdPerLayer, dim);

        unifiedLayer.task("ple_post_reduce",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                context, gemma4State.tempPostPle, gemma4State.wrapPerLayerOut, dim, config.rmsNormEps(), gemma4State.localSize);
        if (shouldUseFinalNormalization()) {
            unifiedLayer.task("ple_post_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context, gemma4State.tempPostPle, dim, config.rmsNormEps());
        }
        unifiedLayer.task("ple_post_apply",
                Gemma4Kernels::rmsNormApplyWithResidual,
                context, gemma4State.wrapX, gemma4State.wrapPerLayerOut, weights.perLayerPostNorm[layerIndex].asFloatArray(), gemma4State.tempPostPle, dim);

        if (weights.layerOutputScale[layerIndex] != null) {
            unifiedLayer.task("layer_output_scale",
                    Gemma4Kernels::scaleInPlaceFromTensor,
                    context, gemma4State.wrapX, weights.layerOutputScale[layerIndex].asFloatArray(), dim);
        }

        unifiedLayer.persistOnDevice(gemma4State.wrapX);
        return unifiedLayer;
    }

    /**
     * One-time-per-token setup tasks, prepended to layer 0's graph: scales the token embedding by
     * {@code sqrt(dim)} (Gemma4 scales embeddings on input -- the generic {@link org.beehive.gpullama3.tornadovm.layers.Activation}
     * task graph that produced {@code wrapX} doesn't know about this), then computes the per-layer
     * embedding inputs from the per-layer model projection and the (host-gathered) per-token
     * per-layer-token-embedding row. Mirrors steps 1-2 of {@link org.beehive.gpullama3.inference.InferenceCore#forwardJavaGemma4}.
     */
    private void appendPLESetupTasks(TaskGraph unifiedLayer) {
        unifiedLayer.task("scale_embedding",
                Gemma4Kernels::scaleInPlace,
                context, gemma4State.wrapX, embedScale, dim);

        addProjection(unifiedLayer, "ple_model_proj", gemma4State.wrapX, gemma4State.wrapPerLayerProjScratch, weights.perLayerModelProj, dim, perLayerTotal);
        unifiedLayer.task("ple_proj_scale_norm",
                Gemma4Kernels::pleProjScaleAndNormalize,
                context, gemma4State.wrapPerLayerProjScratch, weights.perLayerProjNorm.asFloatArray(), nEmbdPerLayer, HEAD_NORM_LOCAL_SIZE, perLayerProjScale, config.rmsNormEps());
        unifiedLayer.task("ple_merge",
                Gemma4Kernels::addAndScale,
                context, gemma4State.wrapPerLayerInputs, gemma4State.wrapPerLayerProjScratch, gemma4State.wrapPerLayerTokenEmbedRow, perLayerInputScale, perLayerTotal);
    }

    /**
     * Configure data transfers for first and subsequent layers.
     */
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        if (layerIndex == 0) {
            unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                    gemma4State.positionHolder, gemma4State.wrapPerLayerTokenEmbedRow,
                    gemma4State.temp, gemma4State.tempFFN, gemma4State.tempPostAttn, gemma4State.tempPostFfn, gemma4State.tempPostPle);
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    weightArray(weights.perLayerModelProj), weights.perLayerProjNorm.asFloatArray(),
                    weights.freqCisRealSwa.asFloatArray(), weights.freqCisImagSwa.asFloatArray(),
                    weights.freqCisRealFull.asFloatArray(), weights.freqCisImagFull.asFloatArray());
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    context, gemma4State.wrapXb, gemma4State.wrapXb2,
                    gemma4State.wrapQ, gemma4State.wrapK, gemma4State.wrapV,
                    gemma4State.wrapKeyCache, gemma4State.wrapValueCache,
                    gemma4State.wrapAtt, gemma4State.wrapHb,
                    gemma4State.wrapPerLayerInputs, gemma4State.wrapPerLayerProjScratch,
                    gemma4State.wrapPerLayerGate, gemma4State.wrapPerLayerOut);
        } else {
            unifiedLayer.consumeFromDevice(context, gemma4State.wrapXb, gemma4State.wrapXb2,
                    gemma4State.wrapQ, gemma4State.wrapK, gemma4State.wrapV,
                    gemma4State.wrapKeyCache, gemma4State.wrapValueCache,
                    gemma4State.wrapAtt, gemma4State.wrapHb,
                    gemma4State.wrapPerLayerInputs, gemma4State.wrapPerLayerGate, gemma4State.wrapPerLayerOut,
                    gemma4State.positionHolder);
        }
        return unifiedLayer;
    }

    // ═══════════════════════════════════════════════════════════════════════════════════
    //                          MIXED-PRECISION PROJECTION HELPERS
    // ═══════════════════════════════════════════════════════════════════════════════════

    /**
     * Adds a matrix-vector projection task, selecting the matmul kernel from the weight tensor's
     * stored precision. The per-layer-embedding projections are not uniformly Q8_0 in a Q8_0 GGUF
     * (they may be F32 or F16), so the kernel/accessor must be chosen per tensor.
     */
    private void addProjection(TaskGraph tg, String taskName, FloatArray in, FloatArray out, TornadoTensor w, int n, int d) {
        switch (w.type()) {
            case Q8_0 -> tg.task(taskName, TransformerComputeKernelsLayered::matrixVectorGenericQ8Byte,
                    context, in, out, w.asByteArray(), n, d, LOCAL_WORK_GROUP_SIZE_ALLOC);
            case F16 -> tg.task(taskName, TransformerComputeKernelsLayered::matrixVectorGeneric,
                    context, in, out, w.asHalfFloatArray(), n, d, LOCAL_WORK_GROUP_SIZE_ALLOC);
            case F32 -> tg.task(taskName, TransformerComputeKernelsLayered::matrixVectorGeneric,
                    context, in, out, w.asFloatArray(), n, d, LOCAL_WORK_GROUP_SIZE_ALLOC);
            default -> throw new UnsupportedOperationException("Unsupported projection weight type: " + w.type());
        }
    }

    /** Returns the device-resident native array backing a weight tensor (for {@code transferToDevice}), matching {@link #addProjection}'s dispatch. */
    private static Object weightArray(TornadoTensor w) {
        return switch (w.type()) {
            case Q8_0 -> w.asByteArray();
            case F16 -> w.asHalfFloatArray();
            case F32 -> w.asFloatArray();
            default -> throw new UnsupportedOperationException("Unsupported projection weight type: " + w.type());
        };
    }

    // ═══════════════════════════════════════════════════════════════════════════════════
    //                                 GRID SCHEDULER
    // ═══════════════════════════════════════════════════════════════════════════════════

    @Override
    public GridScheduler updateGridScheduler(GridScheduler gridScheduler) {
        WorkerGrid rmsNormWorker = WorkerGridFactory.createRmsNormWorker(dim, gemma4State.localSize);
        WorkerGrid dimElementWiseWorker = WorkerGridFactory.genericWorker(dim, LOCAL_WORK_GROUP_SIZE_ALLOC);
        WorkerGrid woProjWorker = WorkerGridFactory.genericWorker(dim * LOCAL_WORK_GROUP_SIZE_ALLOC, LOCAL_WORK_GROUP_SIZE_ALLOC);
        WorkerGrid pleGateProjWorker = WorkerGridFactory.genericWorker(nEmbdPerLayer * LOCAL_WORK_GROUP_SIZE_ALLOC, LOCAL_WORK_GROUP_SIZE_ALLOC);
        WorkerGrid pleGateGeluWorker = WorkerGridFactory.genericWorker(nEmbdPerLayer, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // === Layer-0 PLE setup ===
        gridScheduler.addWorkerGrid("layer_0.scale_embedding", dimElementWiseWorker);
        gridScheduler.addWorkerGrid("layer_0.ple_model_proj", WorkerGridFactory.genericWorker(perLayerTotal * LOCAL_WORK_GROUP_SIZE_ALLOC, LOCAL_WORK_GROUP_SIZE_ALLOC));
        gridScheduler.addWorkerGrid("layer_0.ple_proj_scale_norm", WorkerGridFactory.genericWorker(config.numberOfLayers() * HEAD_NORM_LOCAL_SIZE, HEAD_NORM_LOCAL_SIZE));
        gridScheduler.addWorkerGrid("layer_0.ple_merge", WorkerGridFactory.genericWorker(perLayerTotal, LOCAL_WORK_GROUP_SIZE_ALLOC));

        for (int i = 0; i < config.numberOfLayers(); i++) {
            String prefix = "layer_" + i + ".";
            int headDim = config.headDim(i);
            boolean hasOwnKv = config.hasOwnKv(i);
            int qDim = nHead * headDim;
            int kvDim = nHeadKv * headDim;
            int ffnLen = config.feedForwardLength(i);

            WorkerGrid headNormWorker = WorkerGridFactory.genericWorker(nHead * HEAD_NORM_LOCAL_SIZE, HEAD_NORM_LOCAL_SIZE);
            WorkerGrid kvHeadNormWorker = WorkerGridFactory.genericWorker(nHeadKv * HEAD_NORM_LOCAL_SIZE, HEAD_NORM_LOCAL_SIZE);
            WorkerGrid ropeWorker = WorkerGridFactory.createRoPEWorker(nHead, headDim);
            WorkerGrid attentionWorker = WorkerGridFactory.createAttentionWorker(nHead, headDim);
            WorkerGrid qProjWorker = WorkerGridFactory.genericWorker(qDim * LOCAL_WORK_GROUP_SIZE_ALLOC, LOCAL_WORK_GROUP_SIZE_ALLOC);
            WorkerGrid kvProjWorker = WorkerGridFactory.genericWorker(kvDim * LOCAL_WORK_GROUP_SIZE_ALLOC, LOCAL_WORK_GROUP_SIZE_ALLOC);
            WorkerGrid ffnGateUpWorker = WorkerGridFactory.genericWorker(ffnLen * LOCAL_WORK_GROUP_SIZE_ALLOC, LOCAL_WORK_GROUP_SIZE_ALLOC);

            gridScheduler.addWorkerGrid(prefix + "attn_norm_reduce", rmsNormWorker);
            gridScheduler.addWorkerGrid(prefix + "attn_norm_apply", dimElementWiseWorker);
            gridScheduler.addWorkerGrid(prefix + "q_proj", qProjWorker);
            gridScheduler.addWorkerGrid(prefix + "q_norm", headNormWorker);
            if (hasOwnKv) {
                gridScheduler.addWorkerGrid(prefix + "k_proj", kvProjWorker);
                gridScheduler.addWorkerGrid(prefix + "k_norm", kvHeadNormWorker);
                gridScheduler.addWorkerGrid(prefix + "v_proj", kvProjWorker);
                gridScheduler.addWorkerGrid(prefix + "v_norm", kvHeadNormWorker);
                gridScheduler.addWorkerGrid(prefix + "rope_and_cache", ropeWorker);
            } else {
                gridScheduler.addWorkerGrid(prefix + "rope_q_only", ropeWorker);
            }
            gridScheduler.addWorkerGrid(prefix + "attention", attentionWorker);
            gridScheduler.addWorkerGrid(prefix + "wo_proj", woProjWorker);
            gridScheduler.addWorkerGrid(prefix + "post_attn_reduce", rmsNormWorker);
            gridScheduler.addWorkerGrid(prefix + "post_attn_apply", dimElementWiseWorker);

            gridScheduler.addWorkerGrid(prefix + "ffn_norm_reduce", rmsNormWorker);
            gridScheduler.addWorkerGrid(prefix + "ffn_norm_apply", dimElementWiseWorker);
            gridScheduler.addWorkerGrid(prefix + "ffn_gate_up", ffnGateUpWorker);
            gridScheduler.addWorkerGrid(prefix + "ffn_down_proj", woProjWorker);
            gridScheduler.addWorkerGrid(prefix + "post_ffn_reduce", rmsNormWorker);
            gridScheduler.addWorkerGrid(prefix + "post_ffn_apply", dimElementWiseWorker);

            gridScheduler.addWorkerGrid(prefix + "ple_gate_proj", pleGateProjWorker);
            gridScheduler.addWorkerGrid(prefix + "ple_gate_gelu_mul", pleGateGeluWorker);
            gridScheduler.addWorkerGrid(prefix + "ple_proj", woProjWorker);
            gridScheduler.addWorkerGrid(prefix + "ple_post_reduce", rmsNormWorker);
            gridScheduler.addWorkerGrid(prefix + "ple_post_apply", dimElementWiseWorker);

            if (shouldUseFinalNormalization()) {
                gridScheduler.addWorkerGrid(prefix + "attn_norm_finalize", rmsNormWorker);
                gridScheduler.addWorkerGrid(prefix + "post_attn_finalize", rmsNormWorker);
                gridScheduler.addWorkerGrid(prefix + "ffn_norm_finalize", rmsNormWorker);
                gridScheduler.addWorkerGrid(prefix + "post_ffn_finalize", rmsNormWorker);
                gridScheduler.addWorkerGrid(prefix + "ple_post_finalize", rmsNormWorker);
            }
            if (weights.layerOutputScale[i] != null) {
                gridScheduler.addWorkerGrid(prefix + "layer_output_scale", dimElementWiseWorker);
            }
        }
        return gridScheduler;
    }
}
