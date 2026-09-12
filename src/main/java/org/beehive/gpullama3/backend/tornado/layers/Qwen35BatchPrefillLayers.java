package org.beehive.gpullama3.backend.tornado.layers;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;
import org.beehive.gpullama3.backend.tornado.TensorCoreSupport;
import org.beehive.gpullama3.backend.tornado.kernels.Qwen35BatchKernels;
import org.beehive.gpullama3.backend.tornado.kernels.Qwen35MMAKernels;
import org.beehive.gpullama3.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.gpullama3.backend.tornado.kernels.TransformerComputeKernelsQ4_0;
import org.beehive.gpullama3.backend.tornado.kernels.TransformerComputeKernelsQ4_1;
import org.beehive.gpullama3.backend.tornado.kernels.TransformerComputeKernelsQ5_K;
import org.beehive.gpullama3.backend.tornado.plan.FusedOperandSupport;
import org.beehive.gpullama3.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.gpullama3.backend.tornado.tensor.TornadoTensor;
import org.beehive.gpullama3.inference.state.Qwen35State;
import org.beehive.gpullama3.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.gpullama3.model.qwen35.Qwen35Configuration;
import org.beehive.gpullama3.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

// @formatter:off
/**
 * The {@code qwen35} transformer layers over a chunk of prompt tokens, one task graph per block.
 *
 * <p>The same computation the single-token graphs perform, with a row index — except in the two
 * places where a row depends on the row before it. There the kernel scans the chunk itself, in
 * token order, one lane per convolution channel or per delta-net value column. See {@code
 * Qwen35BatchKernels} for why that is exact and why it makes the result independent of the chunk
 * size.
 *
 * <p>Every weight-reading task is still selected by that tensor's own representation, and the fused
 * gate/up task still requires its two operands to agree. Nothing is materialized for the batched
 * path that is not materialized for the single-token one, which is to say nothing at all.
 *
 * <p>Graph names are {@code batchLayer_i}, distinct from the decode graphs' {@code layer_i}: the
 * batched plan holds both families at once, and the decode ones consume what these produced.
 */
// @formatter:on
public class Qwen35BatchPrefillLayers implements BatchPrefillTransformerLayerTaskGraphs {

    // @formatter:off
    /**
     * Whether the Q4_0 projections that read the normed chunk run on the tensor cores.
     *
     * <p>Off by default, and the reason is a trade rather than a defect. The kernel is correct — it
     * is held against the host tensor in {@code Qwen35MMAProjectionAccelTest} — and it is worth
     * about 2.8% of prompt processing. But FP16 multiplicands move the logits enough to break three
     * of this family's parity bounds, including the absolute ceiling by a factor of five, and 2.8%
     * does not buy a change to the numerical contract.
     *
     * <p>What would: the same treatment for the fused gate/up and {@code ffn_down}, which are
     * another 57% of the profile between them. Then the contract moves once, for a number worth
     * moving it for.
     */
    // @formatter:on
    private static final boolean TENSOR_CORES = Boolean.getBoolean("llama.qwen35.tensorCores");

    private static final int MATVEC_LOCAL = 128;
    private static final int ELEMENTWISE_LOCAL = 128;

    /** Lanes per attention workgroup. One workgroup handles one (row, head). */
    private static final int ATTENTION_LOCAL = 128;

    private final Qwen35State state;
    private final Qwen35TornadoWeights weights;
    private final Qwen35Configuration config;
    private final int batchSize;
    private final KernelContext context = new KernelContext();

    private final List<ImmutableTaskGraph> graphs;
    private String lastLayerTaskGraphID;

    /**
     * How many prompt rows each task's grid covers per workgroup — one, unless it is tiled.
     *
     * <p>Recorded while the graphs are built, from the kernel's own tile constant, because the
     * worker grid has to match the kernel the dispatch chose — and the choice is per tensor, not
     * per task name. {@code ffn_down} is Q4_1 on this model's first eight blocks and Q4_0 on the
     * rest, so the same name can be tiled differently in different layers; keying on the name alone
     * gave the untiled kernel a tiled grid, and every row past the first read the wrong activation.
     * Taking the number from the kernel that was selected means a tile constant can only ever be
     * changed in one place.
     */
    private final java.util.Map<String, Integer> rowTiles = new java.util.LinkedHashMap<>();

    /**
     * How many <b>output</b> rows each task's grid covers per workgroup — one, unless the kernel
     * also tiles that axis. Recorded the same way and for the same reason as {@link #rowTiles}.
     */
    private final java.util.Map<String, Integer> colTiles = new java.util.LinkedHashMap<>();

    /** Tasks that run on the tensor cores, whose grid is a warp per (16 x 8) output tile. */
    private final java.util.Map<String, Integer> mmaTasks = new java.util.LinkedHashMap<>();

    /**
     * Whether a Q4_0 projection over this shape can run on the tensor cores.
     *
     * <p>The chunk has to fill whole 16-row MMA tiles, because the store writes a whole tile and
     * the output buffer is not padded; the reduction dimension has to be a whole number of Q4_0
     * blocks; and the output has to be a whole number of the 8-column tile.
     */
    private boolean mmaEligible(int n, int d) {
        return TENSOR_CORES
                && TensorCoreSupport.isTensorCoreCapableBackend()
                && batchSize % Qwen35MMAKernels.BM == 0
                && n % 32 == 0
                && d % Qwen35MMAKernels.BN == 0;
    }

    public Qwen35BatchPrefillLayers(
            Qwen35State state,
            Qwen35TornadoWeights weights,
            Qwen35Configuration config,
            int batchSize) {
        this.state = state;
        this.weights = weights;
        this.config = config;
        this.batchSize = batchSize;
        this.graphs =
                IntStream.range(0, config.numberOfLayers())
                        .mapToObj(this::buildLayer)
                        .map(TaskGraph::snapshot)
                        .toList();
        this.lastLayerTaskGraphID = "batchLayer_" + (config.numberOfLayers() - 1);
    }

    @Override
    public List<ImmutableTaskGraph> getLayerImmutableTaskGraphs() {
        return graphs;
    }

    @Override
    public String getLastLayerTaskGraphID() {
        return lastLayerTaskGraphID;
    }

    // ── validation and dispatch ───────────────────────────────────────────────

    /** Whether this state's key/value store is half precision. */
    private boolean fp16Kv() {
        return state.usesFp16KeyValueCache();
    }

    /** The key store the graphs bind, whichever precision it is in. */
    private Object keyStore() {
        return fp16Kv() ? state.workspace.wrapKeyCacheFP16 : state.workspace.wrapKeyCache;
    }

    private Object valueStore() {
        return fp16Kv() ? state.workspace.wrapValueCacheFP16 : state.workspace.wrapValueCache;
    }

    private TornadoTensor require(TornadoTensor[] tensors, int layer, String role) {
        TornadoTensor tensor = tensors == null || layer >= tensors.length ? null : tensors[layer];
        if (tensor == null) {
            throw new IllegalStateException(
                    "qwen35 block "
                            + layer
                            + " is a "
                            + (config.isRecurrentLayer(layer) ? "recurrent" : "attention")
                            + " layer and must carry "
                            + role
                            + ", which this model does not hold for it");
        }
        return tensor;
    }

    /**
     * {@code out[b] = w · x[b]}, or {@code +=}, by the representation {@code w} is in.
     *
     * <p>The batched counterpart of the single-token dispatch, and refused the same way: a
     * representation with no batched kernel is named rather than converted.
     */
    private void matVecBatch(
            TaskGraph graph,
            int layer,
            String task,
            String role,
            TornadoTensor w,
            FloatArray xBatch,
            FloatArray outBatch,
            int n,
            int d,
            boolean residual) {
        requireWholeBlocks(layer, task, role, w.dataType(), n);
        switch (w.dataType()) {
            case F32 -> {
                if (residual) {
                    throw unsupported(layer, task, role, w.dataType(), "an accumulating");
                }
                graph.task(
                        task,
                        TransformerBatchPrefillKernels::batchedMatVecF32,
                        context,
                        xBatch,
                        outBatch,
                        w.asFloatArray(),
                        n,
                        d,
                        batchSize,
                        MATVEC_LOCAL);
            }
            case Q4_0 -> {
                // The normed chunk is also staged as FP16 right after the norm, so a projection
                // reading it can run on the tensor cores rather than as a scalar matrix-vector.
                // Anything else — a residual form, another input, a shape the MMA tiles do not
                // divide — takes the tiled scalar kernel below.
                if (!residual && xBatch == state.workspace.wrapNormedBatch && mmaEligible(n, d)) {
                    mmaTasks.put("batchLayer_" + layer + "." + task, d);
                    graph.task(
                            task,
                            Qwen35MMAKernels::projectionMMAQ4_0,
                            context,
                            state.workspace.wrapNormedFP16Batch,
                            w.asByteArray(),
                            outBatch,
                            batchSize,
                            d,
                            n);
                    return;
                }
                // Tiled: one workgroup per (tile of rows, output row), decoding each weight once
                // for the tile. A quantized projection is memory-bound, and a chunk is only worth
                // scheduling if it reuses the weights it reads.
                rowTiles.put(
                        "batchLayer_" + layer + "." + task,
                        TransformerComputeKernelsQ4_0.rowTile());
                colTiles.put(
                        "batchLayer_" + layer + "." + task,
                        TransformerComputeKernelsQ4_0.colTile());
                if (residual) {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ4_0::matrixVectorTiledBatchWithResidualQ4_0,
                            context,
                            xBatch,
                            outBatch,
                            w.asByteArray(),
                            n,
                            d,
                            batchSize,
                            MATVEC_LOCAL);
                } else {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ4_0::matrixVectorTiledBatchQ4_0,
                            context,
                            xBatch,
                            outBatch,
                            w.asByteArray(),
                            n,
                            d,
                            batchSize,
                            MATVEC_LOCAL);
                }
            }
            case Q4_1 -> {
                rowTiles.put(
                        "batchLayer_" + layer + "." + task,
                        TransformerComputeKernelsQ4_1.rowTile());
                colTiles.put(
                        "batchLayer_" + layer + "." + task,
                        TransformerComputeKernelsQ4_1.colTile());
                if (residual) {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ4_1::matrixVectorTiledBatchWithResidualQ4_1,
                            context,
                            xBatch,
                            outBatch,
                            w.asByteArray(),
                            n,
                            d,
                            batchSize,
                            MATVEC_LOCAL);
                } else {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ4_1::matrixVectorTiledBatchQ4_1,
                            context,
                            xBatch,
                            outBatch,
                            w.asByteArray(),
                            n,
                            d,
                            batchSize,
                            MATVEC_LOCAL);
                }
            }
            case Q5_K -> {
                rowTiles.put(
                        "batchLayer_" + layer + "." + task,
                        TransformerComputeKernelsQ5_K.rowTile());
                colTiles.put(
                        "batchLayer_" + layer + "." + task,
                        TransformerComputeKernelsQ5_K.colTile());
                if (residual) {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ5_K::matrixVectorTiledBatchWithResidualQ5_K,
                            context,
                            xBatch,
                            outBatch,
                            w.asByteArray(),
                            n,
                            d,
                            batchSize,
                            MATVEC_LOCAL);
                } else {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ5_K::matrixVectorTiledBatchQ5_K,
                            context,
                            xBatch,
                            outBatch,
                            w.asByteArray(),
                            n,
                            d,
                            batchSize,
                            MATVEC_LOCAL);
                }
            }
            default -> throw unsupported(layer, task, role, w.dataType(), "a batched");
        }
    }

    private UnsupportedOperationException unsupported(
            int layer, String task, String role, DataType type, String article) {
        return new UnsupportedOperationException(
                "qwen35 batch-prefill layer "
                        + layer
                        + " task '"
                        + task
                        + "' reads "
                        + role
                        + " as "
                        + type
                        + ", for which this backend has no "
                        + article
                        + " matrix-vector kernel. It is not converted to another representation to"
                        + " get one.");
    }

    private void requireWholeBlocks(int layer, String task, String role, DataType type, int n) {
        int blockSize =
                switch (type) {
                    case Q4_0, Q4_1, Q8_0 -> 32;
                    case Q4_K, Q5_K, Q6_K -> 256;
                    default -> 1;
                };
        if (n % blockSize != 0) {
            throw new UnsupportedOperationException(
                    "qwen35 batch-prefill layer "
                            + layer
                            + " task '"
                            + task
                            + "' reads "
                            + role
                            + " as "
                            + type
                            + " with a row of "
                            + n
                            + " weights, which is not a whole number of "
                            + blockSize
                            + "-weight blocks.");
        }
    }

    private void fusedGateUpBatch(
            TaskGraph graph, int layer, TornadoTensor gate, TornadoTensor up, FloatArray xBatch) {
        FusedOperandSupport.requireUniform(
                "qwen35 batch-prefill layer " + layer + " fused gate/up feed-forward",
                List.of("ffn_gate", "ffn_up"),
                gate,
                up);
        if (gate.dataType() != DataType.Q4_0) {
            throw unsupported(
                    layer, "ffn_gate_up", "ffn_gate|ffn_up", gate.dataType(), "a batched");
        }
        if (mmaEligible(config.dim(), config.hiddenDim())) {
            // Two single-panel projections rather than one two-panel kernel. Same M, N and K, the
            // same FP16 chunk, the same two destination buffers and the same SwiGLU task after
            // them; each call stages the activation tile for its own panel, which the fused form
            // staged once. The kernel is the one every other projection already uses.
            mmaTasks.put("batchLayer_" + layer + ".ffn_gate_proj", config.hiddenDim());
            mmaTasks.put("batchLayer_" + layer + ".ffn_up_proj", config.hiddenDim());
            graph.task(
                    "ffn_gate_proj",
                    Qwen35MMAKernels::projectionMMAQ4_0,
                    context,
                    state.workspace.wrapNormedFP16Batch,
                    gate.asByteArray(),
                    state.workspace.wrapGateBatch,
                    batchSize,
                    config.hiddenDim(),
                    config.dim());
            graph.task(
                    "ffn_up_proj",
                    Qwen35MMAKernels::projectionMMAQ4_0,
                    context,
                    state.workspace.wrapNormedFP16Batch,
                    up.asByteArray(),
                    state.workspace.wrapUpBatch,
                    batchSize,
                    config.hiddenDim(),
                    config.dim());
            graph.task(
                    "ffn_swiglu",
                    Qwen35MMAKernels::swiGLUBatch,
                    context,
                    state.workspace.wrapGateBatch,
                    state.workspace.wrapUpBatch,
                    state.workspace.wrapHbBatch);
            return;
        }
        rowTiles.put(
                "batchLayer_" + layer + ".ffn_gate_up", TransformerComputeKernelsQ4_0.ffnRowTile());
        colTiles.put(
                "batchLayer_" + layer + ".ffn_gate_up", TransformerComputeKernelsQ4_0.ffnColTile());
        graph.task(
                "ffn_gate_up",
                TransformerComputeKernelsQ4_0::fusedFFNGateUpSiLUTiledBatchQ4_0,
                context,
                xBatch,
                state.workspace.wrapHbBatch,
                gate.asByteArray(),
                up.asByteArray(),
                config.dim(),
                config.hiddenDim(),
                batchSize,
                MATVEC_LOCAL);
    }

    // ── the layer graphs ──────────────────────────────────────────────────────

    private TaskGraph buildLayer(int layerIndex) {
        TaskGraph layer = new TaskGraph("batchLayer_" + layerIndex);

        String predecessor =
                layerIndex == 0 ? "prefillActivation" : "batchLayer_" + (layerIndex - 1);
        layer.consumeFromDevice(predecessor, state.workspace.wrapXBatch);
        configureTransfers(layer, layerIndex, predecessor);
        transferLayerWeights(layer, layerIndex);

        normalize(
                layer,
                "attn_rms_reduce",
                "attn_rms_apply",
                state.workspace.attnScaleBatch,
                require(weights.rms_att_weightLayered, layerIndex, "attn_norm"));

        if (config.isRecurrentLayer(layerIndex)) {
            deltaNetBranch(layer, layerIndex);
        } else {
            attentionBranch(layer, layerIndex);
        }

        normalize(
                layer,
                "ffn_rms_reduce",
                "ffn_rms_apply",
                state.workspace.ffnScaleBatch,
                require(weights.rms_ffn_weightLayered, layerIndex, "post_attention_norm"));

        fusedGateUpBatch(
                layer,
                layerIndex,
                require(weights.w1Layered, layerIndex, "ffn_gate"),
                require(weights.w3Layered, layerIndex, "ffn_up"),
                state.workspace.wrapNormedBatch);
        TornadoTensor down = require(weights.w2Layered, layerIndex, "ffn_down");
        // Both representations this family's ffn_down comes in. The Q4_1 kernel below was written
        // and tested with the Q4_0 one, and then never reached: this condition asked for Q4_0 and
        // the choice of kernel underneath it asked whether the tensor was Q4_1, so the first eight
        // blocks -- the Q4_1 ones -- fell through to the scalar path, which is what the profile
        // showed still running.
        boolean downOnTensorCores =
                (down.dataType() == DataType.Q4_0 || down.dataType() == DataType.Q4_1)
                        && mmaEligible(config.hiddenDim(), config.dim());
        if (downOnTensorCores) {
            // The tensor-core store overwrites, so the residual is a pass of its own. Its input is
            // SwiGLU's output rather than a normed chunk, so that is converted here too.
            mmaTasks.put("batchLayer_" + layerIndex + ".ffn_down_proj", config.dim());
            layer.task(
                    "ffn_down_fp16",
                    Qwen35MMAKernels::convertToFP16,
                    context,
                    state.workspace.wrapHbBatch,
                    state.workspace.wrapHbFP16BatchMMA);
            layer.task(
                    "ffn_down_proj",
                    down.dataType() == DataType.Q4_1
                            ? Qwen35MMAKernels::projectionMMAQ4_1
                            : Qwen35MMAKernels::projectionMMAQ4_0,
                    context,
                    state.workspace.wrapHbFP16BatchMMA,
                    down.asByteArray(),
                    state.workspace.wrapFFNDownBatch,
                    batchSize,
                    config.dim(),
                    config.hiddenDim());
            layer.task(
                    "ffn_down_residual",
                    Qwen35MMAKernels::residualAdd,
                    context,
                    state.workspace.wrapXBatch,
                    state.workspace.wrapFFNDownBatch);
        } else {
            matVecBatch(
                    layer,
                    layerIndex,
                    "ffn_down_proj",
                    "ffn_down",
                    down,
                    state.workspace.wrapHbBatch,
                    state.workspace.wrapXBatch,
                    config.hiddenDim(),
                    config.dim(),
                    true);
        }

        layer.persistOnDevice(
                state.workspace.wrapXBatch,
                keyStore(),
                valueStore(),
                state.workspace.wrapBlockTable,
                state.workspace.wrapConvState,
                state.workspace.wrapDeltaState);
        return layer;
    }

    /** {@code normed[b] = weight ⊙ rms(x[b])} — a scale per row, then the apply. */
    private void normalize(
            TaskGraph layer,
            String reduce,
            String apply,
            FloatArray scaleBatch,
            TornadoTensor weight) {
        layer.task(
                reduce,
                TransformerBatchPrefillKernels::batchedRmsReduce,
                context,
                state.workspace.wrapXBatch,
                scaleBatch,
                config.dim(),
                config.rmsNormEps());
        layer.task(
                apply,
                TransformerBatchPrefillKernels::batchedRmsApplyFP32,
                context,
                state.workspace.wrapNormedBatch,
                state.workspace.wrapXBatch,
                weight.asFloatArray(),
                scaleBatch,
                config.dim());
        if (TENSOR_CORES && TensorCoreSupport.isTensorCoreCapableBackend()) {
            layer.task(
                    apply + "_fp16",
                    Qwen35MMAKernels::convertNormedToFP16,
                    context,
                    state.workspace.wrapNormedBatch,
                    state.workspace.wrapNormedFP16Batch,
                    config.dim(),
                    state.workspace.batchStartPosHolder);
        }
    }

    private void attentionBranch(TaskGraph layer, int layerIndex) {
        final int headDim = config.numberOfHeadsKey();
        final int kvDim = config.kvDim();
        final int attnDim = config.attentionOutputInputDim();
        final int kvLayer = config.keyValueLayerIndex(layerIndex);

        matVecBatch(
                layer,
                layerIndex,
                "attn_q_proj",
                "attn_q",
                require(weights.wqLayered, layerIndex, "attn_q"),
                state.workspace.wrapNormedBatch,
                state.workspace.wrapQGateBatch,
                config.dim(),
                config.queryGateDim(),
                false);
        matVecBatch(
                layer,
                layerIndex,
                "attn_k_proj",
                "attn_k",
                require(weights.wkLayered, layerIndex, "attn_k"),
                state.workspace.wrapNormedBatch,
                state.workspace.wrapKBatch,
                config.dim(),
                kvDim,
                false);
        matVecBatch(
                layer,
                layerIndex,
                "attn_v_proj",
                "attn_v",
                require(weights.wvLayered, layerIndex, "attn_v"),
                state.workspace.wrapNormedBatch,
                state.workspace.wrapVBatch,
                config.dim(),
                kvDim,
                false);

        layer.task(
                "attn_split_query_gate",
                Qwen35BatchKernels::splitQueryGateBatch,
                context,
                state.workspace.wrapQGateBatch,
                state.workspace.wrapAttnQBatch,
                state.workspace.wrapAttnGateBatch,
                config.numberOfHeads(),
                headDim,
                state.workspace.batchStartPosHolder);

        layer.task(
                "attn_qk_norm",
                Qwen35BatchKernels::fusedQKRmsNormBatch,
                context,
                state.workspace.wrapAttnQBatch,
                state.workspace.wrapKBatch,
                require(weights.attnQNorm, layerIndex, "attn_q_norm").asFloatArray(),
                require(weights.attnKNorm, layerIndex, "attn_k_norm").asFloatArray(),
                config.numberOfHeads(),
                config.numberOfKeyValueHeads(),
                headDim,
                config.rmsNormEps(),
                state.workspace.batchStartPosHolder);

        layer.task(
                "attn_rope",
                Qwen35BatchKernels::ropeNeoxPartialBatch,
                context,
                state.workspace.batchStartPosHolder,
                state.workspace.wrapAttnQBatch,
                state.workspace.wrapKBatch,
                weights.freq_cis_realFlat.asFloatArray(),
                weights.freq_cis_imagFlat.asFloatArray(),
                config.numberOfHeads(),
                config.numberOfKeyValueHeads(),
                headDim,
                config.ropeDimensionCount());

        // Every row's key and value written before any row attends: a row may read an earlier
        // row's entry, and the append is what puts it there.
        if (fp16Kv()) {
            layer.task(
                    "attn_kv_append",
                    Qwen35BatchKernels::appendKeyValueBatchFP16Paged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    state.workspace.wrapBlockTable,
                    kvDim,
                    kvLayer,
                    state.kvBlockCfg,
                    state.kvBlockStride);
        } else {
            layer.task(
                    "attn_kv_append",
                    Qwen35BatchKernels::appendKeyValueBatchPaged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.wrapBlockTable,
                    kvDim,
                    kvLayer,
                    state.kvBlockCfg,
                    state.kvBlockStride);
        }

        if (fp16Kv()) {
            layer.task(
                    "attention",
                    Qwen35BatchKernels::attentionBatchFP16Paged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapAttnQBatch,
                    state.workspace.wrapKeyCacheFP16,
                    state.workspace.wrapValueCacheFP16,
                    state.workspace.wrapXbBatch,
                    config.numberOfHeads(),
                    headDim,
                    kvDim,
                    config.kvMul(),
                    kvLayer,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    ATTENTION_LOCAL);
        } else {
            layer.task(
                    "attention",
                    Qwen35BatchKernels::attentionBatchPaged,
                    context,
                    state.workspace.batchStartPosHolder,
                    state.workspace.wrapAttnQBatch,
                    state.workspace.wrapKeyCache,
                    state.workspace.wrapValueCache,
                    state.workspace.wrapXbBatch,
                    config.numberOfHeads(),
                    headDim,
                    kvDim,
                    config.kvMul(),
                    kvLayer,
                    state.workspace.wrapBlockTable,
                    state.kvBlockCfg,
                    state.kvBlockStride,
                    ATTENTION_LOCAL);
        }

        layer.task(
                "attn_output_gate",
                Qwen35BatchKernels::applyOutputGateBatch,
                context,
                state.workspace.wrapXbBatch,
                state.workspace.wrapAttnGateBatch,
                attnDim,
                state.workspace.batchStartPosHolder);

        matVecBatch(
                layer,
                layerIndex,
                "attn_output_proj",
                "attn_output",
                require(weights.woLayered, layerIndex, "attn_output"),
                state.workspace.wrapXbBatch,
                state.workspace.wrapXBatch,
                attnDim,
                config.dim(),
                true);
    }

    private void deltaNetBranch(TaskGraph layer, int layerIndex) {
        final int convDim = config.deltaNetConvDim();
        final int keyDim = config.deltaNetKeyDim();
        final int valueDim = config.deltaNetValueDim();
        final int valueHeads = config.numberOfValueHeads();
        final int headK = config.headKeyDim();
        final int headV = config.headValueDim();
        final int recurrent = config.recurrentLayerIndex(layerIndex);

        matVecBatch(
                layer,
                layerIndex,
                "ssm_qkv_proj",
                "attn_qkv",
                require(weights.ssmQkv, layerIndex, "attn_qkv"),
                state.workspace.wrapNormedBatch,
                state.workspace.wrapSsmQkvBatch,
                config.dim(),
                convDim,
                false);
        matVecBatch(
                layer,
                layerIndex,
                "ssm_gate_proj",
                "attn_gate",
                require(weights.ssmGate, layerIndex, "attn_gate"),
                state.workspace.wrapNormedBatch,
                state.workspace.wrapSsmZBatch,
                config.dim(),
                valueDim,
                false);
        matVecBatch(
                layer,
                layerIndex,
                "ssm_beta_proj",
                "ssm_beta",
                require(weights.ssmBeta, layerIndex, "ssm_beta"),
                state.workspace.wrapNormedBatch,
                state.workspace.wrapSsmBetaBatch,
                config.dim(),
                valueHeads,
                false);
        matVecBatch(
                layer,
                layerIndex,
                "ssm_alpha_proj",
                "ssm_alpha",
                require(weights.ssmAlpha, layerIndex, "ssm_alpha"),
                state.workspace.wrapNormedBatch,
                state.workspace.wrapSsmAlphaBatch,
                config.dim(),
                valueHeads,
                false);

        layer.task(
                "ssm_decay_beta",
                Qwen35BatchKernels::decayAndBetaBatch,
                context,
                state.workspace.wrapSsmAlphaBatch,
                state.workspace.wrapSsmBetaBatch,
                require(weights.ssmDtBias, layerIndex, "ssm_dt.bias").asFloatArray(),
                require(weights.ssmA, layerIndex, "ssm_a").asFloatArray(),
                valueHeads,
                state.workspace.batchStartPosHolder);

        // The scan: one lane per channel, walking the chunk in token order.
        layer.task(
                "ssm_conv",
                Qwen35BatchKernels::causalConv1dScan,
                context,
                state.workspace.wrapSsmQkvBatch,
                require(weights.ssmConv1d, layerIndex, "ssm_conv1d").asFloatArray(),
                state.workspace.wrapConvState,
                state.workspace.wrapSsmConvOutBatch,
                convDim,
                config.ssmConvKernel(),
                recurrent * config.convStateSize(),
                state.workspace.batchStartPosHolder);
        layer.task(
                "ssm_conv_silu",
                Qwen35BatchKernels::siluInPlaceBatch,
                context,
                state.workspace.wrapSsmConvOutBatch,
                convDim,
                state.workspace.batchStartPosHolder);

        layer.task(
                "ssm_split_qkv",
                Qwen35BatchKernels::splitThreeWayBatch,
                context,
                state.workspace.wrapSsmConvOutBatch,
                state.workspace.wrapSsmQBatch,
                state.workspace.wrapSsmKBatch,
                state.workspace.wrapSsmVBatch,
                keyDim,
                keyDim,
                valueDim,
                state.workspace.batchStartPosHolder);

        layer.task(
                "ssm_l2norm_q",
                Qwen35BatchKernels::l2NormPerHeadBatch,
                context,
                state.workspace.wrapSsmQBatch,
                config.numberOfKeyHeads(),
                headK,
                config.rmsNormEps(),
                state.workspace.batchStartPosHolder);
        layer.task(
                "ssm_l2norm_k",
                Qwen35BatchKernels::l2NormPerHeadBatch,
                context,
                state.workspace.wrapSsmKBatch,
                config.numberOfKeyHeads(),
                headK,
                config.rmsNormEps(),
                state.workspace.batchStartPosHolder);
        layer.task(
                "ssm_scale_q",
                Qwen35BatchKernels::scaleInPlaceBatch,
                context,
                state.workspace.wrapSsmQBatch,
                (float) (1.0 / Math.sqrt(headK)),
                keyDim,
                state.workspace.batchStartPosHolder);

        // The other scan: one lane per (value head, value column), same order.
        layer.task(
                "ssm_delta_rule",
                Qwen35BatchKernels::deltaRuleScan,
                context,
                state.workspace.wrapSsmQBatch,
                state.workspace.wrapSsmKBatch,
                state.workspace.wrapSsmVBatch,
                state.workspace.wrapSsmAlphaBatch,
                state.workspace.wrapSsmBetaBatch,
                state.workspace.wrapDeltaState,
                state.workspace.wrapSsmOutBatch,
                valueHeads,
                config.numberOfKeyHeads(),
                headV,
                recurrent * config.deltaNetStateSize(),
                state.workspace.batchStartPosHolder);

        layer.task(
                "ssm_gated_norm",
                Qwen35BatchKernels::gatedNormPerHeadBatch,
                context,
                state.workspace.wrapSsmOutBatch,
                state.workspace.wrapSsmZBatch,
                require(weights.ssmNorm, layerIndex, "ssm_norm").asFloatArray(),
                valueHeads,
                headV,
                config.rmsNormEps(),
                state.workspace.batchStartPosHolder);

        TornadoTensor ssmOut = require(weights.ssmOut, layerIndex, "ssm_out");
        if (ssmOut.dataType() == DataType.Q5_K && mmaEligible(valueDim, config.dim())) {
            // Same shape as ffn_down: convert the readout, project, then add the residual back,
            // because a tensor-core store overwrites.
            mmaTasks.put("batchLayer_" + layerIndex + ".ssm_out_proj", config.dim());
            layer.task(
                    "ssm_out_fp16",
                    Qwen35MMAKernels::convertToFP16,
                    context,
                    state.workspace.wrapSsmOutBatch,
                    state.workspace.wrapSsmOutFP16Batch);
            layer.task(
                    "ssm_out_proj",
                    Qwen35MMAKernels::projectionMMAQ5_K,
                    context,
                    state.workspace.wrapSsmOutFP16Batch,
                    ssmOut.asByteArray(),
                    state.workspace.wrapFFNDownBatch,
                    batchSize,
                    config.dim(),
                    valueDim);
            layer.task(
                    "ssm_out_residual",
                    Qwen35MMAKernels::residualAdd,
                    context,
                    state.workspace.wrapXBatch,
                    state.workspace.wrapFFNDownBatch);
        } else {
            matVecBatch(
                    layer,
                    layerIndex,
                    "ssm_out_proj",
                    "ssm_out",
                    ssmOut,
                    state.workspace.wrapSsmOutBatch,
                    state.workspace.wrapXBatch,
                    valueDim,
                    config.dim(),
                    true);
        }
    }

    // ── transfers ─────────────────────────────────────────────────────────────

    private void transferLayerWeights(TaskGraph layer, int layerIndex) {
        List<Object> tensors = new ArrayList<>();
        tensors.add(weights.rms_att_weightLayered[layerIndex].asFloatArray());
        tensors.add(weights.rms_ffn_weightLayered[layerIndex].asFloatArray());
        tensors.add(deviceArray(require(weights.w1Layered, layerIndex, "ffn_gate")));
        tensors.add(deviceArray(require(weights.w2Layered, layerIndex, "ffn_down")));
        tensors.add(deviceArray(require(weights.w3Layered, layerIndex, "ffn_up")));
        if (config.isRecurrentLayer(layerIndex)) {
            tensors.add(deviceArray(require(weights.ssmQkv, layerIndex, "attn_qkv")));
            tensors.add(deviceArray(require(weights.ssmGate, layerIndex, "attn_gate")));
            tensors.add(deviceArray(require(weights.ssmAlpha, layerIndex, "ssm_alpha")));
            tensors.add(deviceArray(require(weights.ssmBeta, layerIndex, "ssm_beta")));
            tensors.add(deviceArray(require(weights.ssmOut, layerIndex, "ssm_out")));
            tensors.add(require(weights.ssmConv1d, layerIndex, "ssm_conv1d").asFloatArray());
            tensors.add(require(weights.ssmDtBias, layerIndex, "ssm_dt.bias").asFloatArray());
            tensors.add(require(weights.ssmA, layerIndex, "ssm_a").asFloatArray());
            tensors.add(require(weights.ssmNorm, layerIndex, "ssm_norm").asFloatArray());
        } else {
            tensors.add(deviceArray(require(weights.wqLayered, layerIndex, "attn_q")));
            tensors.add(deviceArray(require(weights.wkLayered, layerIndex, "attn_k")));
            tensors.add(deviceArray(require(weights.wvLayered, layerIndex, "attn_v")));
            tensors.add(deviceArray(require(weights.woLayered, layerIndex, "attn_output")));
            tensors.add(require(weights.attnQNorm, layerIndex, "attn_q_norm").asFloatArray());
            tensors.add(require(weights.attnKNorm, layerIndex, "attn_k_norm").asFloatArray());
        }
        layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, tensors.toArray());
    }

    private static Object deviceArray(TornadoTensor tensor) {
        return switch (tensor.dataType()) {
            case F32 -> tensor.asFloatArray();
            case F16 -> tensor.asHalfFloatArray();
            default -> tensor.asByteArray();
        };
    }

    // @formatter:off
    /**
     * The chunk's scratch and its persistent state.
     *
     * <p>The recurrent state is uploaded once and never read back: it is the session's own history,
     * advanced in place by these graphs and then by the decode graphs. Uploading it per chunk would
     * overwrite what the previous chunk wrote with the host's stale zeros, which is the one way a
     * chunk boundary could change the answer.
     */
    // @formatter:on
    private void configureTransfers(TaskGraph layer, int layerIndex, String predecessor) {
        if (layerIndex == 0) {
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.batchStartPosHolder);
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION, state.workspace.wrapBlockTable);
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.workspace.wrapNormedBatch,
                    state.workspace.wrapNormedFP16Batch,
                    state.workspace.wrapGateBatch,
                    state.workspace.wrapUpBatch,
                    state.workspace.wrapHbFP16BatchMMA,
                    state.workspace.wrapFFNDownBatch,
                    state.workspace.wrapSsmOutFP16Batch,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapQGateBatch,
                    state.workspace.wrapAttnQBatch,
                    state.workspace.wrapAttnGateBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapHbBatch,
                    keyStore(),
                    valueStore());
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    state.workspace.wrapSsmQkvBatch,
                    state.workspace.wrapSsmConvOutBatch,
                    state.workspace.wrapSsmZBatch,
                    state.workspace.wrapSsmAlphaBatch,
                    state.workspace.wrapSsmBetaBatch,
                    state.workspace.wrapSsmQBatch,
                    state.workspace.wrapSsmKBatch,
                    state.workspace.wrapSsmVBatch,
                    state.workspace.wrapSsmOutBatch,
                    state.workspace.wrapConvState,
                    state.workspace.wrapDeltaState);
        } else {
            layer.consumeFromDevice(
                    predecessor,
                    context,
                    state.workspace.wrapNormedBatch,
                    state.workspace.wrapNormedFP16Batch,
                    state.workspace.wrapGateBatch,
                    state.workspace.wrapUpBatch,
                    state.workspace.wrapHbFP16BatchMMA,
                    state.workspace.wrapFFNDownBatch,
                    state.workspace.wrapSsmOutFP16Batch,
                    state.workspace.attnScaleBatch,
                    state.workspace.ffnScaleBatch,
                    state.workspace.wrapQGateBatch,
                    state.workspace.wrapAttnQBatch,
                    state.workspace.wrapAttnGateBatch,
                    state.workspace.wrapKBatch,
                    state.workspace.wrapVBatch,
                    state.workspace.wrapXbBatch,
                    state.workspace.wrapHbBatch,
                    keyStore(),
                    valueStore(),
                    state.workspace.batchStartPosHolder);
            layer.consumeFromDevice(predecessor, state.workspace.wrapBlockTable);
            layer.consumeFromDevice(
                    predecessor,
                    state.workspace.wrapSsmQkvBatch,
                    state.workspace.wrapSsmConvOutBatch,
                    state.workspace.wrapSsmZBatch,
                    state.workspace.wrapSsmAlphaBatch,
                    state.workspace.wrapSsmBetaBatch,
                    state.workspace.wrapSsmQBatch,
                    state.workspace.wrapSsmKBatch,
                    state.workspace.wrapSsmVBatch,
                    state.workspace.wrapSsmOutBatch,
                    state.workspace.wrapConvState,
                    state.workspace.wrapDeltaState);
        }
    }

    // ── worker grids ──────────────────────────────────────────────────────────

    @Override
    public void updateGridScheduler(GridScheduler scheduler) {
        final int headDim = config.numberOfHeadsKey();

        WorkerGrid rmsReduce = WorkerGridFactory.genericWorker(batchSize, 1);
        WorkerGrid rmsApply =
                WorkerGridFactory.genericWorker(batchSize * config.dim(), ELEMENTWISE_LOCAL);
        // One lane per element of the padded chunk.
        WorkerGrid fp16Convert =
                WorkerGridFactory.genericWorker(
                        ((batchSize + Qwen35MMAKernels.BM - 1) / Qwen35MMAKernels.BM)
                                * Qwen35MMAKernels.BM
                                * config.dim(),
                        ELEMENTWISE_LOCAL);
        WorkerGrid ssmOutFP16Convert =
                WorkerGridFactory.genericWorker(
                        ((batchSize + Qwen35MMAKernels.BM - 1) / Qwen35MMAKernels.BM)
                                * Qwen35MMAKernels.BM
                                * config.deltaNetValueDim(),
                        ELEMENTWISE_LOCAL);
        WorkerGrid hbFP16Convert =
                WorkerGridFactory.genericWorker(
                        ((batchSize + Qwen35MMAKernels.BM - 1) / Qwen35MMAKernels.BM)
                                * Qwen35MMAKernels.BM
                                * config.hiddenDim(),
                        ELEMENTWISE_LOCAL);
        WorkerGrid residualAdd =
                WorkerGridFactory.genericWorker(batchSize * config.dim(), ELEMENTWISE_LOCAL);
        WorkerGrid swiglu =
                WorkerGridFactory.genericWorker(batchSize * config.hiddenDim(), ELEMENTWISE_LOCAL);
        WorkerGrid queryGate =
                WorkerGridFactory.genericWorker(
                        batchSize * config.attentionOutputInputDim(), ELEMENTWISE_LOCAL);
        WorkerGrid qkNorm =
                WorkerGridFactory.genericWorker(
                        batchSize * (config.numberOfHeads() + config.numberOfKeyValueHeads()), 1);
        WorkerGrid rope =
                WorkerGridFactory.genericWorker(
                        batchSize * config.numberOfHeads() * (config.ropeDimensionCount() / 2), 32);
        WorkerGrid kvAppend =
                WorkerGridFactory.genericWorker(batchSize * config.kvDim(), ELEMENTWISE_LOCAL);
        // One workgroup per (row, head); the workgroup's lanes split the causal range.
        WorkerGrid attention =
                WorkerGridFactory.genericWorker(
                        batchSize * config.numberOfHeads() * ATTENTION_LOCAL, ATTENTION_LOCAL);
        WorkerGrid outputGate =
                WorkerGridFactory.genericWorker(
                        batchSize * config.attentionOutputInputDim(), ELEMENTWISE_LOCAL);

        WorkerGrid convDim =
                WorkerGridFactory.genericWorker(
                        batchSize * config.deltaNetConvDim(), ELEMENTWISE_LOCAL);
        // The scans launch one lane per channel or per state column — not per row. The chunk is
        // the loop inside the lane.
        WorkerGrid convChannels =
                WorkerGridFactory.genericWorker(config.deltaNetConvDim(), ELEMENTWISE_LOCAL);
        WorkerGrid deltaColumns =
                WorkerGridFactory.genericWorker(
                        config.numberOfValueHeads() * config.headValueDim(), ELEMENTWISE_LOCAL);
        WorkerGrid keyDim =
                WorkerGridFactory.genericWorker(
                        batchSize * config.deltaNetKeyDim(), ELEMENTWISE_LOCAL);
        WorkerGrid keyHeads =
                WorkerGridFactory.genericWorker(batchSize * config.numberOfKeyHeads(), 1);
        WorkerGrid valueHeads =
                WorkerGridFactory.genericWorker(batchSize * config.numberOfValueHeads(), 1);

        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            String prefix = "batchLayer_" + layer + ".";
            scheduler.addWorkerGrid(prefix + "attn_rms_reduce", rmsReduce);
            scheduler.addWorkerGrid(prefix + "ffn_rms_reduce", rmsReduce);
            scheduler.addWorkerGrid(prefix + "attn_rms_apply", rmsApply);
            scheduler.addWorkerGrid(prefix + "ffn_rms_apply", rmsApply);
            if (TENSOR_CORES && TensorCoreSupport.isTensorCoreCapableBackend()) {
                scheduler.addWorkerGrid(prefix + "attn_rms_apply_fp16", fp16Convert);
                scheduler.addWorkerGrid(prefix + "ffn_rms_apply_fp16", fp16Convert);
            }
            if (mmaTasks.containsKey(prefix + "ffn_gate_proj")) {
                scheduler.addWorkerGrid(
                        prefix + "ffn_gate_proj",
                        matVecWorker(prefix + "ffn_gate_proj", config.hiddenDim()));
                scheduler.addWorkerGrid(
                        prefix + "ffn_up_proj",
                        matVecWorker(prefix + "ffn_up_proj", config.hiddenDim()));
                scheduler.addWorkerGrid(prefix + "ffn_swiglu", swiglu);
            } else {
                scheduler.addWorkerGrid(
                        prefix + "ffn_gate_up",
                        matVecWorker(prefix + "ffn_gate_up", config.hiddenDim()));
            }
            scheduler.addWorkerGrid(
                    prefix + "ffn_down_proj", matVecWorker(prefix + "ffn_down_proj", config.dim()));
            if (mmaTasks.containsKey(prefix + "ffn_down_proj")) {
                scheduler.addWorkerGrid(prefix + "ffn_down_fp16", hbFP16Convert);
                scheduler.addWorkerGrid(prefix + "ffn_down_residual", residualAdd);
            }

            if (config.isRecurrentLayer(layer)) {
                scheduler.addWorkerGrid(
                        prefix + "ssm_qkv_proj",
                        matVecWorker(prefix + "ssm_qkv_proj", config.deltaNetConvDim()));
                scheduler.addWorkerGrid(
                        prefix + "ssm_gate_proj",
                        matVecWorker(prefix + "ssm_gate_proj", config.deltaNetValueDim()));
                scheduler.addWorkerGrid(
                        prefix + "ssm_beta_proj",
                        matVecWorker(prefix + "ssm_beta_proj", config.numberOfValueHeads()));
                scheduler.addWorkerGrid(
                        prefix + "ssm_alpha_proj",
                        matVecWorker(prefix + "ssm_alpha_proj", config.numberOfValueHeads()));
                scheduler.addWorkerGrid(prefix + "ssm_decay_beta", valueHeads);
                scheduler.addWorkerGrid(prefix + "ssm_conv", convChannels);
                scheduler.addWorkerGrid(prefix + "ssm_conv_silu", convDim);
                scheduler.addWorkerGrid(prefix + "ssm_split_qkv", convDim);
                scheduler.addWorkerGrid(prefix + "ssm_l2norm_q", keyHeads);
                scheduler.addWorkerGrid(prefix + "ssm_l2norm_k", keyHeads);
                scheduler.addWorkerGrid(prefix + "ssm_scale_q", keyDim);
                scheduler.addWorkerGrid(prefix + "ssm_delta_rule", deltaColumns);
                scheduler.addWorkerGrid(prefix + "ssm_gated_norm", valueHeads);
                scheduler.addWorkerGrid(
                        prefix + "ssm_out_proj",
                        matVecWorker(prefix + "ssm_out_proj", config.dim()));
                if (mmaTasks.containsKey(prefix + "ssm_out_proj")) {
                    scheduler.addWorkerGrid(prefix + "ssm_out_fp16", ssmOutFP16Convert);
                    scheduler.addWorkerGrid(prefix + "ssm_out_residual", residualAdd);
                }
            } else {
                scheduler.addWorkerGrid(
                        prefix + "attn_q_proj",
                        matVecWorker(prefix + "attn_q_proj", config.queryGateDim()));
                scheduler.addWorkerGrid(
                        prefix + "attn_k_proj",
                        matVecWorker(prefix + "attn_k_proj", config.kvDim()));
                scheduler.addWorkerGrid(
                        prefix + "attn_v_proj",
                        matVecWorker(prefix + "attn_v_proj", config.kvDim()));
                scheduler.addWorkerGrid(prefix + "attn_split_query_gate", queryGate);
                scheduler.addWorkerGrid(prefix + "attn_qk_norm", qkNorm);
                scheduler.addWorkerGrid(prefix + "attn_rope", rope);
                scheduler.addWorkerGrid(prefix + "attn_kv_append", kvAppend);
                scheduler.addWorkerGrid(prefix + "attention", attention);
                scheduler.addWorkerGrid(prefix + "attn_output_gate", outputGate);
                scheduler.addWorkerGrid(
                        prefix + "attn_output_proj",
                        matVecWorker(prefix + "attn_output_proj", config.dim()));
            }
        }
    }

    /** One workgroup per (row, output row), or per (row tile, output row) where tiled. */
    private WorkerGrid matVecWorker(String qualifiedTask, int rows) {
        Integer mmaCols = mmaTasks.get(qualifiedTask);
        if (mmaCols != null) {
            int rowTilesMma = batchSize / Qwen35MMAKernels.BM;
            int colTilesMma = mmaCols / Qwen35MMAKernels.BN;
            return WorkerGridFactory.genericWorker(
                    rowTilesMma * colTilesMma * Qwen35MMAKernels.LOCAL, Qwen35MMAKernels.LOCAL);
        }
        int tileRows = rowTiles.getOrDefault(qualifiedTask, 1);
        int tileCols = colTiles.getOrDefault(qualifiedTask, 1);
        int rowGroups = (batchSize + tileRows - 1) / tileRows;
        int colGroups = (rows + tileCols - 1) / tileCols;
        return WorkerGridFactory.genericWorker(rowGroups * colGroups * MATVEC_LOCAL, MATVEC_LOCAL);
    }
}
