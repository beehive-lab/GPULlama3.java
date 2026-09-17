package org.beehive.jllm.backend.tornado.layers;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jllm.backend.tornado.kernels.Gemma4BatchPrefillKernels;
import org.beehive.jllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jllm.inference.state.Gemma4State;
import org.beehive.jllm.inference.weights.tornado.Gemma4TornadoWeights;
import org.beehive.jllm.model.gemma4.Gemma4Configuration;
import org.beehive.jllm.runtime.tensor.DataType;
import org.beehive.jllm.backend.tornado.tensor.TornadoTensor;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

// @formatter:off
/**
 * The batched-prefill transformer layers for Gemma 4, Q8_0 weights, on the tensor cores.
 *
 * <p><b>Why this exists.</b> Until it did, this family's prompt processing was its decode loop run
 * once per prompt token, so every token read every weight: five hundred and twelve sweeps of the
 * model to ingest five hundred and twelve tokens. A chunk-wide graph reads each weight once for the
 * whole chunk, which is the entire difference between a matrix-vector product and a matrix-matrix
 * one, and it is why the reference implementation is two orders of magnitude faster at this and not
 * at decode.
 *
 * <p><b>What is this family's and what is not.</b> The GEMMs are the repository's own — {@code
 * gemmMMAQ8}, {@code gemmMMAQKVQ8}, {@code gemmMMAGateUpQ8} and {@code gemmMMA} are written in
 * terms of M, N and K and know nothing about any architecture — and so are the RMS reductions and
 * the FP32-to-FP16 cast. Everything that is Gemma 4's is in {@link Gemma4BatchPrefillKernels}: the
 * sandwich norms, the query/key/value head norms, the table-driven NeoX rotation into this family's
 * flat KV cache, the sliding window, the GeGLU, and the per-layer-embedding block.
 *
 * <p><b>Two head widths and two feed-forward widths.</b> Four layers in five attend through a
 * 256-wide head and a 512-position window; the fifth attends through a 512-wide head over the whole
 * context. Blocks 0-14 have a 6144-wide feed-forward and blocks 15-34 a 12288-wide one. Every
 * chunk-wide buffer is therefore allocated at the widest and addressed with the layer's own stride,
 * and every worker grid is built per layer rather than once — a grid keyed on a task name is wrong
 * the moment one name maps to two shapes.
 *
 * <p><b>Twenty layers project no key and no value.</b> {@code shared_kv_layers} is 20, so fifteen
 * layers own a cache and the rest read an earlier layer's. Those twenty run the query projection
 * alone, rotate the query alone, and their attention addresses the cache their source layer filled.
 *
 * <p><b>Numerically this is not the decode path.</b> The GEMM operands are FP16 with FP32
 * accumulation, and the two per-layer-embedding projections — which this file's weights hold in
 * FP32 — are narrowed to FP16 once at construction so they can be a GEMM's B operand at all. That
 * is a new approximation on the prefill path and it is gated as one, at the decision level, not by
 * a bound that was widened to admit it.
 */
// @formatter:on
public class Gemma4BatchPrefillLayers implements BatchPrefillTransformerLayerTaskGraphs {

    /** One workgroup per token for the RMS reductions, as the other MMA prefill families use. */
    private static final int RMS_LOCAL_SIZE = 256;

    /** Lanes per (row, head) in the per-head norms and in attention. */
    private static final int HEAD_LOCAL_SIZE = 128;

    private final Gemma4State state;
    private final Gemma4TornadoWeights weights;
    private final Gemma4Configuration config;
    private final KernelContext context = new KernelContext();
    private final int batchSize;
    private final int paddedBatch;

    private final int nHead;
    private final int nHeadKv;
    private final int kvMul;
    private final int dim;
    private final int nEmbdPerLayer;
    private final int perLayerTotal;
    private final float embedScale;
    private final float perLayerProjScale;
    private final float perLayerInputScale;

    /**
     * The two per-layer-embedding projections in FP16.
     *
     * <p>This file holds {@code inp_gate} and {@code proj} in FP32, and the tensor-core GEMM's B
     * operand is FP16. They are narrowed once here rather than per chunk: together they are 786,432
     * elements a layer, 55 MB across the trunk in FP16, which is less than one chunk's worth of
     * re-narrowing traffic and is paid at plan construction instead of inside the timed window.
     */
    private final HalfFloatArray[] pleGateF16;

    private final HalfFloatArray[] pleProjF16;

    private final List<ImmutableTaskGraph> layerITGs;
    private String lastLayerTaskGraphID;

    public Gemma4BatchPrefillLayers(
            Gemma4State state,
            Gemma4TornadoWeights weights,
            Gemma4Configuration config,
            int batchSize) {
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.paddedBatch = (batchSize + 127) & ~127;
        this.nHead = config.numberOfHeads();
        this.nHeadKv = config.numberOfKeyValueHeads();
        this.kvMul = config.kvMul();
        this.dim = config.dim();
        this.nEmbdPerLayer = config.embeddingLengthPerLayer();
        this.perLayerTotal = config.numberOfLayers() * nEmbdPerLayer;
        this.embedScale = (float) Math.sqrt(dim);
        this.perLayerProjScale = (float) (1.0 / Math.sqrt(dim));
        this.perLayerInputScale = (float) (1.0 / Math.sqrt(2.0));

        int layers = config.numberOfLayers();
        this.pleGateF16 = new HalfFloatArray[layers];
        this.pleProjF16 = new HalfFloatArray[layers];
        for (int l = 0; l < layers; l++) {
            int pleElements = dim * nEmbdPerLayer;
            pleGateF16[l] =
                    narrowToF16(weights.perLayerInpGate[l], pleElements, "blk." + l + ".inp_gate");
            pleProjF16[l] = narrowToF16(weights.perLayerProj[l], pleElements, "blk." + l + ".proj");
        }

        List<ImmutableTaskGraph> graphs = new ArrayList<>(layers);
        for (int l = 0; l < layers; l++) {
            graphs.add(createBatchPrefillLayerTaskGraph(l).snapshot());
        }
        this.layerITGs = List.copyOf(graphs);
    }

    // @formatter:off
    /**
     * The B operand of a tensor-core GEMM in FP16, from whatever the file holds.
     *
     * <p>Refused by name rather than by a fallthrough: reading a Q8_0 block layout as FP16 halves
     * is a plausible-looking activation and wrong output, and a quantization this method does not
     * know is a missing case and not a reason to guess.
     */
    // @formatter:on
    private static HalfFloatArray narrowToF16(TornadoTensor t, int elements, String name) {
        HalfFloatArray out = new HalfFloatArray(elements);
        switch (t.dataType()) {
            case F16 -> {
                var src = t.asHalfFloatArray();
                for (int i = 0; i < elements; i++) {
                    out.set(i, src.get(i));
                }
            }
            case F32 -> {
                var src = t.asFloatArray();
                for (int i = 0; i < elements; i++) {
                    out.set(i, new HalfFloat(src.get(i)));
                }
            }
            default ->
                    throw new UnsupportedOperationException(
                            "gemma4 batched prefill has no tensor-core operand for "
                                    + t.dataType()
                                    + " ("
                                    + name
                                    + "); the per-layer-embedding projections must be F32 or F16");
        }
        return out;
    }

    private static void requireQ8(TornadoTensor t, String name) {
        if (t.dataType() != DataType.Q8_0) {
            throw new UnsupportedOperationException(
                    "gemma4 batched prefill is Q8_0-only for the trunk; "
                            + name
                            + " is "
                            + t.dataType());
        }
    }

    /** The packed [q|k|v] row stride this layer's projection writes. */
    private int qkvStride(int layerIndex) {
        int headDim = config.headDim(layerIndex);
        int qDim = nHead * headDim;
        return config.hasOwnKv(layerIndex) ? qDim + 2 * nHeadKv * headDim : qDim;
    }

    // @formatter:off
    private TaskGraph createBatchPrefillLayerTaskGraph(int layerIndex) {
        String graphName = "batchPrefillLayer_" + layerIndex;
        if (layerIndex == config.numberOfLayers() - 1) {
            lastLayerTaskGraphID = graphName;
        }
        TaskGraph layer = new TaskGraph(graphName);

        final int headDim = config.headDim(layerIndex);
        final boolean isSwa = config.isSwa(layerIndex);
        final boolean hasOwnKv = config.hasOwnKv(layerIndex);
        final int qDim = nHead * headDim;
        final int kvDim = nHeadKv * headDim;
        final int ffnLen = config.feedForwardLength(layerIndex);
        final int cacheBaseOffset = state.cacheLayerBaseOffset[layerIndex];
        final int windowSize = isSwa ? config.slidingWindowSize() : config.contextLength();
        final var freqCisReal =
                (isSwa ? weights.freqCisRealSwa : weights.freqCisRealFull).asFloatArray();
        final var freqCisImag =
                (isSwa ? weights.freqCisImagSwa : weights.freqCisImagFull).asFloatArray();
        final int peOffset = layerIndex * nEmbdPerLayer;
        final int stride = qkvStride(layerIndex);

        requireQ8(weights.wqLayered[layerIndex], "blk." + layerIndex + ".attn_q");
        requireQ8(weights.woLayered[layerIndex], "blk." + layerIndex + ".attn_output");
        requireQ8(weights.w1Layered[layerIndex], "blk." + layerIndex + ".ffn_gate");
        requireQ8(weights.w3Layered[layerIndex], "blk." + layerIndex + ".ffn_up");
        requireQ8(weights.w2Layered[layerIndex], "blk." + layerIndex + ".ffn_down");

        // ── Transfers ──────────────────────────────────────────────────────────
        if (layerIndex == 0) {
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapPerLayerTokenEmbedRowBatch);
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.branchScaleBatch,
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.qkvResultBatch,
                    state.workspace.attnOutFP16,
                    state.workspace.attnScoresBatch,
                    state.workspace.woOut,
                    state.workspace.normedXFFNFP16,
                    state.workspace.gateUpResultBatch,
                    state.workspace.wrapHbFP16Batch,
                    state.workspace.w2Out);
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.wrapXFP16Batch,
                    state.workspace.wrapPerLayerInputsBatch,
                    state.workspace.wrapPerLayerProjScratchBatch,
                    state.workspace.wrapPerLayerGateBatch,
                    state.workspace.wrapPerLayerGateFP16Batch,
                    state.workspace.wrapPerLayerOutBatch);
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    weights.perLayerModelProj.asHalfFloatArray(),
                    weights.perLayerProjNorm.asFloatArray(),
                    weights.freqCisRealSwa.asFloatArray(),
                    weights.freqCisImagSwa.asFloatArray(),
                    weights.freqCisRealFull.asFloatArray(),
                    weights.freqCisImagFull.asFloatArray());
            layer.consumeFromDevice("prefillActivation", state.workspace.wrapXBatch);
        } else {
            String pred = "batchPrefillLayer_" + (layerIndex - 1);
            layer.consumeFromDevice(
                    pred,
                    context,
                    state.workspace.wrapXBatch,
                    state.workspace.batchStartPosHolder,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.branchScaleBatch,
                    state.workspace.wrapXbFP16Batch,
                    state.workspace.qkvResultBatch,
                    state.workspace.attnOutFP16,
                    state.workspace.attnScoresBatch,
                    state.workspace.woOut,
                    state.workspace.normedXFFNFP16,
                    state.workspace.gateUpResultBatch,
                    state.workspace.wrapHbFP16Batch,
                    state.workspace.w2Out);
            layer.consumeFromDevice(
                    pred,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.wrapXFP16Batch,
                    state.workspace.wrapPerLayerInputsBatch,
                    state.workspace.wrapPerLayerProjScratchBatch,
                    state.workspace.wrapPerLayerGateBatch,
                    state.workspace.wrapPerLayerGateFP16Batch,
                    state.workspace.wrapPerLayerOutBatch);
        }

        layer.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asByteArray(),
                weights.woLayered[layerIndex].asByteArray(),
                weights.attnQNorm[layerIndex].asFloatArray(),
                weights.attnPostNorm[layerIndex].asFloatArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].asByteArray(),
                weights.w3Layered[layerIndex].asByteArray(),
                weights.w2Layered[layerIndex].asByteArray(),
                weights.ffnPostNorm[layerIndex].asFloatArray(),
                pleGateF16[layerIndex],
                pleProjF16[layerIndex],
                weights.perLayerPostNorm[layerIndex].asFloatArray());
        if (hasOwnKv) {
            requireQ8(weights.wkLayered[layerIndex], "blk." + layerIndex + ".attn_k");
            requireQ8(weights.wvLayered[layerIndex], "blk." + layerIndex + ".attn_v");
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    weights.wkLayered[layerIndex].asByteArray(),
                    weights.wvLayered[layerIndex].asByteArray(),
                    weights.attnKNorm[layerIndex].asFloatArray());
        }
        if (weights.layerOutputScale[layerIndex] != null) {
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    weights.layerOutputScale[layerIndex].asFloatArray());
        }

        if (layerIndex == 0) {
            appendPleSetup(layer);
        }

        // ── Attention ──────────────────────────────────────────────────────────
        layer.task(
                "batch_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context,
                state.workspace.wrapXBatch,
                state.workspace.attnScaleBatch,
                dim,
                config.rmsNormEps(),
                RMS_LOCAL_SIZE);
        layer.task(
                "batch_attn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP16,
                context,
                state.workspace.wrapXbFP16Batch,
                state.workspace.wrapXBatch,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                state.workspace.attnScaleBatch,
                dim);

        if (hasOwnKv) {
            layer.task(
                    "qkvProj",
                    TransformerBatchPrefillKernels::gemmMMAQKVQ8,
                    context,
                    state.workspace.wrapXbFP16Batch,
                    weights.wqLayered[layerIndex].asByteArray(),
                    weights.wkLayered[layerIndex].asByteArray(),
                    weights.wvLayered[layerIndex].asByteArray(),
                    state.workspace.qkvResultBatch,
                    paddedBatch,
                    qDim,
                    kvDim,
                    dim);
            layer.task(
                    "batch_qkv_norm",
                    Gemma4BatchPrefillKernels::batchedQkvHeadNorms,
                    context,
                    state.workspace.qkvResultBatch,
                    weights.attnQNorm[layerIndex].asFloatArray(),
                    weights.attnKNorm[layerIndex].asFloatArray(),
                    nHead,
                    nHeadKv,
                    headDim,
                    stride,
                    HEAD_LOCAL_SIZE,
                    config.rmsNormEps());
            layer.task(
                    "batch_rope_kv",
                    Gemma4BatchPrefillKernels::batchedRopeAndCache,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.qkvResultBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    freqCisReal,
                    freqCisImag,
                    nHead,
                    nHeadKv,
                    headDim,
                    kvDim,
                    stride,
                    cacheBaseOffset);
        } else {
            layer.task(
                    "qkvProj",
                    TransformerBatchPrefillKernels::gemmMMAQ8,
                    context,
                    state.workspace.wrapXbFP16Batch,
                    weights.wqLayered[layerIndex].asByteArray(),
                    state.workspace.qkvResultBatch,
                    paddedBatch,
                    qDim,
                    dim);
            layer.task(
                    "batch_qkv_norm",
                    Gemma4BatchPrefillKernels::batchedQHeadNorm,
                    context,
                    state.workspace.qkvResultBatch,
                    weights.attnQNorm[layerIndex].asFloatArray(),
                    nHead,
                    headDim,
                    stride,
                    HEAD_LOCAL_SIZE,
                    config.rmsNormEps());
            layer.task(
                    "batch_rope_kv",
                    Gemma4BatchPrefillKernels::batchedRopeQOnly,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.qkvResultBatch,
                    freqCisReal,
                    freqCisImag,
                    nHead,
                    headDim,
                    stride);
        }

        layer.task(
                "batch_attention",
                Gemma4BatchPrefillKernels::batchedSlidingWindowAttention,
                context,
                state.workspace.batchStartPosHolder,
                state.workspace.qkvResultBatch,
                state.workspace.wrapKeyCache,
                state.workspace.wrapValueCache,
                state.workspace.attnOutFP16,
                state.workspace.attnScoresBatch,
                nHead,
                headDim,
                kvDim,
                kvMul,
                stride,
                cacheBaseOffset,
                windowSize,
                config.contextLength(),
                HEAD_LOCAL_SIZE);

        layer.task(
                "woProj",
                TransformerBatchPrefillKernels::gemmMMAQ8,
                context,
                state.workspace.attnOutFP16,
                weights.woLayered[layerIndex].asByteArray(),
                state.workspace.woOut,
                paddedBatch,
                dim,
                qDim);
        layer.task(
                "batch_post_attn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context,
                state.workspace.woOut,
                state.workspace.branchScaleBatch,
                dim,
                config.rmsNormEps(),
                RMS_LOCAL_SIZE);
        layer.task(
                "batch_post_attn_apply",
                Gemma4BatchPrefillKernels::batchedRmsApplyWithResidual,
                context,
                state.workspace.wrapXBatch,
                state.workspace.woOut,
                weights.attnPostNorm[layerIndex].asFloatArray(),
                state.workspace.branchScaleBatch,
                dim);

        // ── Feed-forward ───────────────────────────────────────────────────────
        layer.task(
                "batch_ffn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context,
                state.workspace.wrapXBatch,
                state.workspace.ffnScaleBatch,
                dim,
                config.rmsNormEps(),
                RMS_LOCAL_SIZE);
        layer.task(
                "batch_ffn_rms_apply",
                TransformerBatchPrefillKernels::batchedRmsApplyFP16,
                context,
                state.workspace.normedXFFNFP16,
                state.workspace.wrapXBatch,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.workspace.ffnScaleBatch,
                dim);
        layer.task(
                "gateUpProj",
                TransformerBatchPrefillKernels::gemmMMAGateUpQ8,
                context,
                state.workspace.normedXFFNFP16,
                weights.w1Layered[layerIndex].asByteArray(),
                weights.w3Layered[layerIndex].asByteArray(),
                state.workspace.gateUpResultBatch,
                paddedBatch,
                ffnLen,
                dim);
        layer.task(
                "batch_geglu",
                Gemma4BatchPrefillKernels::batchedGeGLUFP16Packed,
                context,
                state.workspace.wrapHbFP16Batch,
                state.workspace.gateUpResultBatch,
                ffnLen);
        layer.task(
                "w2Proj",
                TransformerBatchPrefillKernels::gemmMMAQ8,
                context,
                state.workspace.wrapHbFP16Batch,
                weights.w2Layered[layerIndex].asByteArray(),
                state.workspace.w2Out,
                paddedBatch,
                dim,
                ffnLen);
        layer.task(
                "batch_post_ffn_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context,
                state.workspace.w2Out,
                state.workspace.branchScaleBatch,
                dim,
                config.rmsNormEps(),
                RMS_LOCAL_SIZE);
        layer.task(
                "batch_post_ffn_apply",
                Gemma4BatchPrefillKernels::batchedRmsApplyWithResidual,
                context,
                state.workspace.wrapXBatch,
                state.workspace.w2Out,
                weights.ffnPostNorm[layerIndex].asFloatArray(),
                state.workspace.branchScaleBatch,
                dim);

        // ── Per-layer embedding ────────────────────────────────────────────────
        layer.task(
                "batch_x_cast",
                TransformerBatchPrefillKernels::batchedConvertFP32toFP16,
                context,
                state.workspace.wrapXBatch,
                state.workspace.wrapXFP16Batch);
        layer.task(
                "pleGateProj",
                TransformerBatchPrefillKernels::gemmMMA,
                context,
                state.workspace.wrapXFP16Batch,
                pleGateF16[layerIndex],
                state.workspace.wrapPerLayerGateBatch,
                paddedBatch,
                nEmbdPerLayer,
                dim);
        layer.task(
                "batch_ple_gate_gelu",
                Gemma4BatchPrefillKernels::batchedPleGateGeluMul,
                context,
                state.workspace.wrapPerLayerGateFP16Batch,
                state.workspace.wrapPerLayerGateBatch,
                state.workspace.wrapPerLayerInputsBatch,
                peOffset,
                nEmbdPerLayer,
                perLayerTotal);
        layer.task(
                "pleProj",
                TransformerBatchPrefillKernels::gemmMMA,
                context,
                state.workspace.wrapPerLayerGateFP16Batch,
                pleProjF16[layerIndex],
                state.workspace.wrapPerLayerOutBatch,
                paddedBatch,
                dim,
                nEmbdPerLayer);
        layer.task(
                "batch_ple_post_rms",
                TransformerBatchPrefillKernels::batchedRmsReduceParallel,
                context,
                state.workspace.wrapPerLayerOutBatch,
                state.workspace.branchScaleBatch,
                dim,
                config.rmsNormEps(),
                RMS_LOCAL_SIZE);
        layer.task(
                "batch_ple_post_apply",
                Gemma4BatchPrefillKernels::batchedRmsApplyWithResidual,
                context,
                state.workspace.wrapXBatch,
                state.workspace.wrapPerLayerOutBatch,
                weights.perLayerPostNorm[layerIndex].asFloatArray(),
                state.workspace.branchScaleBatch,
                dim);

        if (weights.layerOutputScale[layerIndex] != null) {
            layer.task(
                    "batch_layer_output_scale",
                    Gemma4BatchPrefillKernels::batchedScaleInPlaceFromTensor,
                    context,
                    state.workspace.wrapXBatch,
                    weights.layerOutputScale[layerIndex].asFloatArray());
        }

        layer.persistOnDevice(
                state.workspace.wrapXBatch,
                state.workspace.wrapKeyCache,
                state.workspace.wrapValueCache);
        return layer;
    }

    /**
     * The chunk-wide form of layer 0's one-time-per-token per-layer-embedding setup: scale the
     * embeddings, project them to the per-layer inputs, normalize each layer's segment, and merge
     * with the host-gathered per-layer token embedding rows.
     */
    private void appendPleSetup(TaskGraph layer) {
        layer.task(
                "batch_scale_embedding",
                Gemma4BatchPrefillKernels::batchedScaleInPlace,
                context,
                state.workspace.wrapXBatch,
                embedScale);
        layer.task(
                "batch_embed_cast",
                TransformerBatchPrefillKernels::batchedConvertFP32toFP16,
                context,
                state.workspace.wrapXBatch,
                state.workspace.wrapXFP16Batch);
        layer.task(
                "pleModelProj",
                TransformerBatchPrefillKernels::gemmMMA,
                context,
                state.workspace.wrapXFP16Batch,
                weights.perLayerModelProj.asHalfFloatArray(),
                state.workspace.wrapPerLayerProjScratchBatch,
                paddedBatch,
                perLayerTotal,
                dim);
        layer.task(
                "batch_ple_proj_scale_norm",
                Gemma4BatchPrefillKernels::batchedPleProjScaleAndNormalize,
                context,
                state.workspace.wrapPerLayerProjScratchBatch,
                weights.perLayerProjNorm.asFloatArray(),
                nEmbdPerLayer,
                HEAD_LOCAL_SIZE,
                perLayerProjScale,
                config.rmsNormEps());
        layer.task(
                "batch_ple_merge",
                Gemma4BatchPrefillKernels::batchedAddAndScale,
                context,
                state.workspace.wrapPerLayerInputsBatch,
                state.workspace.wrapPerLayerProjScratchBatch,
                state.workspace.wrapPerLayerTokenEmbedRowBatch,
                perLayerInputScale);
    }

    // @formatter:on

    /** The {@code gemmMMA} family: 256 threads per block, one block per (M-tile, N-tile). */
    private static WorkerGrid mmaGrid(int paddedM, int n) {
        WorkerGrid2D g = new WorkerGrid2D((paddedM / 128) * 256, n / 128);
        g.setLocalWork(256, 1, 1);
        return g;
    }

    private static WorkerGrid elementwise(int n, int local) {
        int l = Math.min(local, n);
        while (l > 1 && n % l != 0) {
            l--;
        }
        WorkerGrid1D g = new WorkerGrid1D(n);
        g.setLocalWork(l, 1, 1);
        return g;
    }

    // @formatter:off
    /**
     * Worker grids, built per layer.
     *
     * <p>Per layer and not once: this family's head width, feed-forward width and whether a layer
     * projects a key at all differ between layers sharing a task name, and a grid keyed on a task
     * name is wrong the moment one name maps to two shapes.
     */
    // @formatter:on
    public void updateGridScheduler(GridScheduler scheduler) {
        WorkerGrid rmsWorker =
                WorkerGridFactory.genericWorker(batchSize * RMS_LOCAL_SIZE, RMS_LOCAL_SIZE);
        WorkerGrid dimApplyWorker = elementwise(batchSize * dim, 256);
        WorkerGrid mmaDimWorker = mmaGrid(paddedBatch, dim);
        WorkerGrid pleGateWorker = mmaGrid(paddedBatch, nEmbdPerLayer);
        WorkerGrid pleGateGeluWorker = elementwise(batchSize * nEmbdPerLayer, 256);

        for (int l = 0; l < config.numberOfLayers(); l++) {
            String p = "batchPrefillLayer_" + l + ".";
            int headDim = config.headDim(l);
            int qDim = nHead * headDim;
            int kvDim = nHeadKv * headDim;
            int ffnLen = config.feedForwardLength(l);
            boolean hasOwnKv = config.hasOwnKv(l);

            scheduler.addWorkerGrid(p + "batch_attn_rms", rmsWorker);
            scheduler.addWorkerGrid(p + "batch_attn_rms_apply", dimApplyWorker);
            scheduler.addWorkerGrid(
                    p + "qkvProj", mmaGrid(paddedBatch, hasOwnKv ? qDim + 2 * kvDim : qDim));

            int normSlots = hasOwnKv ? nHead + 2 * nHeadKv : nHead;
            scheduler.addWorkerGrid(
                    p + "batch_qkv_norm",
                    WorkerGridFactory.genericWorker(
                            batchSize * normSlots * HEAD_LOCAL_SIZE, HEAD_LOCAL_SIZE));
            scheduler.addWorkerGrid(
                    p + "batch_rope_kv", elementwise(batchSize * nHead * (headDim / 2), 256));
            scheduler.addWorkerGrid(
                    p + "batch_attention",
                    WorkerGridFactory.genericWorker(
                            batchSize * nHead * HEAD_LOCAL_SIZE, HEAD_LOCAL_SIZE));
            scheduler.addWorkerGrid(p + "woProj", mmaDimWorker);
            scheduler.addWorkerGrid(p + "batch_post_attn_rms", rmsWorker);
            scheduler.addWorkerGrid(p + "batch_post_attn_apply", dimApplyWorker);

            scheduler.addWorkerGrid(p + "batch_ffn_rms", rmsWorker);
            scheduler.addWorkerGrid(p + "batch_ffn_rms_apply", dimApplyWorker);
            scheduler.addWorkerGrid(p + "gateUpProj", mmaGrid(paddedBatch, 2 * ffnLen));
            scheduler.addWorkerGrid(p + "batch_geglu", elementwise(batchSize * ffnLen, 256));
            scheduler.addWorkerGrid(p + "w2Proj", mmaDimWorker);
            scheduler.addWorkerGrid(p + "batch_post_ffn_rms", rmsWorker);
            scheduler.addWorkerGrid(p + "batch_post_ffn_apply", dimApplyWorker);

            scheduler.addWorkerGrid(p + "batch_x_cast", dimApplyWorker);
            scheduler.addWorkerGrid(p + "pleGateProj", pleGateWorker);
            scheduler.addWorkerGrid(p + "batch_ple_gate_gelu", pleGateGeluWorker);
            scheduler.addWorkerGrid(p + "pleProj", mmaDimWorker);
            scheduler.addWorkerGrid(p + "batch_ple_post_rms", rmsWorker);
            scheduler.addWorkerGrid(p + "batch_ple_post_apply", dimApplyWorker);
            if (weights.layerOutputScale[l] != null) {
                scheduler.addWorkerGrid(p + "batch_layer_output_scale", dimApplyWorker);
            }
        }

        String p0 = "batchPrefillLayer_0.";
        scheduler.addWorkerGrid(p0 + "batch_scale_embedding", dimApplyWorker);
        scheduler.addWorkerGrid(p0 + "batch_embed_cast", dimApplyWorker);
        scheduler.addWorkerGrid(p0 + "pleModelProj", mmaGrid(paddedBatch, perLayerTotal));
        scheduler.addWorkerGrid(
                p0 + "batch_ple_proj_scale_norm",
                WorkerGridFactory.genericWorker(
                        batchSize * config.numberOfLayers() * HEAD_LOCAL_SIZE, HEAD_LOCAL_SIZE));
        scheduler.addWorkerGrid(
                p0 + "batch_ple_merge", elementwise(batchSize * perLayerTotal, 256));
    }

    public List<ImmutableTaskGraph> getLayerImmutableTaskGraphs() {
        return layerITGs;
    }

    public String getLastLayerTaskGraphID() {
        return lastLayerTaskGraphID;
    }

    public KernelContext getContext() {
        return context;
    }
}
