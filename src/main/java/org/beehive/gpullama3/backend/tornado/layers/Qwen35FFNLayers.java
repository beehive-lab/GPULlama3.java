package org.beehive.gpullama3.backend.tornado.layers;

import java.util.ArrayList;
import java.util.List;
import org.beehive.gpullama3.backend.tornado.device.TornadoDevices;
import org.beehive.gpullama3.backend.tornado.kernels.Qwen35AttentionKernels;
import org.beehive.gpullama3.backend.tornado.kernels.Qwen35DeltaNetKernels;
import org.beehive.gpullama3.backend.tornado.kernels.Qwen3Kernels;
import org.beehive.gpullama3.backend.tornado.kernels.TransformerComputeKernels;
import org.beehive.gpullama3.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.gpullama3.backend.tornado.kernels.TransformerComputeKernelsQ4_0;
import org.beehive.gpullama3.backend.tornado.kernels.TransformerComputeKernelsQ4_1;
import org.beehive.gpullama3.backend.tornado.kernels.TransformerComputeKernelsQ4_K;
import org.beehive.gpullama3.backend.tornado.kernels.TransformerComputeKernelsQ5_K;
import org.beehive.gpullama3.backend.tornado.kernels.TransformerComputeKernelsQ6_K;
import org.beehive.gpullama3.backend.tornado.kernels.TransformerPagedKvKernels;
import org.beehive.gpullama3.backend.tornado.plan.FusedOperandSupport;
import org.beehive.gpullama3.backend.tornado.scheduling.SchedulerType;
import org.beehive.gpullama3.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.gpullama3.backend.tornado.tensor.TornadoTensor;
import org.beehive.gpullama3.inference.state.Qwen35State;
import org.beehive.gpullama3.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.gpullama3.model.qwen35.Qwen35Configuration;
import org.beehive.gpullama3.runtime.tensor.DataType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

// @formatter:off
/**
 * The {@code qwen35} transformer layers, one task graph per block.
 *
 * <h2>Two layer kinds, one graph shape</h2>
 *
 * <p>Three quarters of the blocks mix with a Gated Delta Net and one quarter with attention, and
 * which is which is {@link Qwen35Configuration#isRecurrentLayer(int)} — the same answer the host
 * path uses. Both kinds then run the same dense SwiGLU feed-forward, and both have the same outer
 * residual shape. The per-layer arrays are indexed by <b>absolute</b> block, and are {@code null}
 * at blocks of the other kind; a layer validates the tensors its own kind needs before it builds a
 * single task, so a mis-shaped file fails naming the layer, the role and the representation rather
 * than by dereferencing null somewhere inside TornadoVM.
 *
 * <h2>Every task is selected by the representation of the tensor it reads</h2>
 *
 * <p>This is the first family whose model is genuinely mixed: Q4_0 projections and embeddings, Q4_1
 * {@code ffn_down} on the first eight blocks, Q5_K {@code ssm_out}, a Q6_K vocabulary projection,
 * F32 norms and SSM parameters. Nothing is materialized to a common representation, so a task is
 * bound to a format-specific kernel chosen <b>here</b>, before compilation — never to a kernel that
 * switches on a dtype inside its inner loop, which would cost the compiler the fixed addressing
 * that makes the loop worth writing.
 *
 * <p>The one task that reads two weights at once is the fused gate/up feed-forward. It states that
 * its operands must share a representation and refuses a mixture by name; it does not convert one
 * of them, and it does not read one block layout as another.
 *
 * <h2>What it does not do</h2>
 *
 * <p>Single-token decode only. There are no prefill or batch variants of these graphs, and the
 * provider declares none. The MTP block past the trunk is not built here: it is a draft head, not
 * part of ordinary generation, and {@code numberOfLayers()} excludes it.
 */
// @formatter:on
public class Qwen35FFNLayers
        extends AbstractTransformerLayerTaskGraphs<Qwen35TornadoWeights, Qwen35Configuration> {

    /** Lanes per workgroup for a matrix-vector task: one workgroup reduces one output row. */
    private static final int MATVEC_LOCAL = 128;

    /** Lanes per workgroup for an elementwise task. */
    private static final int ELEMENTWISE_LOCAL = 128;

    private final Qwen35State qwen35State;

    /**
     * What each weight-reading task was bound to, in construction order.
     *
     * <p>Recorded rather than derived: which kernel a task got is what changes when a tensor's
     * representation changes, and it is the only thing about a built plan a test can compare
     * without executing it. {@code Qwen35GraphTopologyTest} asserts over this.
     */
    private final List<Dispatch> dispatches = new ArrayList<>();

    // @formatter:off
    /**
     * One weight-reading task: where it is, what it reads, which kernel decodes it, and whether its
     * <b>activation</b> reached it quantized.
     *
     * <p>The last of those is not derivable from the other three. Whether a projection takes the
     * packed-integer path depends on what the buffer it reads happens to hold at that point in the
     * layer, which is a fact about ordering rather than about the tensor — so it is recorded here
     * and asserted, not inferred.
     */
    // @formatter:on
    public record Dispatch(
            int layer,
            String task,
            String role,
            DataType representation,
            boolean quantizedActivation) {}

    /**
     * The graph layer 0 consumes its activation from.
     *
     * <p>{@code activationUpdate} in the single-token plan and {@code decodeActivation} in the
     * prefill/decode one. The layer computation is identical in both — sequential prefill is these
     * graphs with the logits graph skipped — so the plan shape is the only thing that differs, and
     * it differs by a name.
     */
    private final String activationGraphName;

    public Qwen35FFNLayers(
            String taskGraphName,
            Qwen35State state,
            Qwen35TornadoWeights weights,
            Qwen35Configuration config,
            SchedulerType schedulerType) {
        this(taskGraphName, state, weights, config, schedulerType, "activationUpdate");
    }

    public Qwen35FFNLayers(
            String taskGraphName,
            Qwen35State state,
            Qwen35TornadoWeights weights,
            Qwen35Configuration config,
            SchedulerType schedulerType,
            String activationGraphName) {
        super(taskGraphName, state, weights, config, schedulerType);
        this.qwen35State = state;
        this.activationGraphName = activationGraphName;
        setupFFNLayers();
    }

    /** The dispatch inventory, in construction order. */
    public List<Dispatch> dispatchInventory() {
        return List.copyOf(dispatches);
    }

    // ── validation ────────────────────────────────────────────────────────────

    /**
     * The tensor a layer's role must carry, or a failure naming what is missing.
     *
     * <p>A {@code null} here means the file disagrees with the topology the metadata declares — a
     * block the configuration calls recurrent that carries attention tensors, or the reverse. That
     * is a load-time fact, and the message says which block and which role rather than leaving a
     * null to surface as a NullPointerException inside graph construction.
     */
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

    // ── per-representation matrix-vector dispatch ─────────────────────────────

    /**
     * {@code out = w · x}, or {@code out += w · x}, by the representation {@code w} is in.
     *
     * <p>The selection happens here, at plan construction, so each task is compiled against one
     * block layout with fixed addressing. A representation with no kernel for this shape is refused
     * by name: converting it would double what it occupies and hide the gap.
     */
    private void matVec(
            TaskGraph graph,
            int layer,
            String task,
            String role,
            TornadoTensor w,
            FloatArray x,
            FloatArray out,
            int n,
            int d,
            boolean residual) {
        requireWholeBlocks(layer, task, role, w.dataType(), n);
        boolean packed =
                w.dataType() == DataType.Q4_0
                        && ((!residual && x == state.workspace.wrapXb && normedActivationQuantized)
                                || (residual
                                        && x == state.workspace.wrapHb
                                        && hiddenActivationQuantized));
        dispatches.add(new Dispatch(layer, task, role, w.dataType(), packed));
        switch (w.dataType()) {
            case F32 -> {
                if (residual) {
                    throw unsupported(layer, task, role, w.dataType(), "an accumulating");
                }
                graph.task(
                        task,
                        TransformerComputeKernelsLayered::matrixVectorGeneric,
                        context,
                        x,
                        out,
                        w.asFloatArray(),
                        n,
                        d,
                        MATVEC_LOCAL);
            }
            case F16 -> {
                if (residual) {
                    graph.task(
                            task,
                            TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                            context,
                            x,
                            out,
                            w.asHalfFloatArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                } else {
                    graph.task(
                            task,
                            TransformerComputeKernelsLayered::matrixVectorGeneric,
                            context,
                            x,
                            out,
                            w.asHalfFloatArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                }
            }
            case Q8_0 -> {
                if (residual) {
                    graph.task(
                            task,
                            TransformerComputeKernelsLayered
                                    ::matrixVectorGenericWithResidualQ8_0Byte,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                } else {
                    graph.task(
                            task,
                            TransformerComputeKernelsLayered::matrixVectorGenericQ8Byte,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                }
            }
            case Q4_0 -> {
                if (residual && x == state.workspace.wrapHb && hiddenActivationQuantized) {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ4_0::matrixVectorGenericWithResidualQ4_0DP4A,
                            context,
                            state.workspace.wrapXbQuants,
                            state.workspace.wrapXbScales,
                            state.workspace.wrapXbSums,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                } else if (!residual && x == state.workspace.wrapXb && normedActivationQuantized) {
                    // The packed-integer path, for a projection whose input the branch already
                    // quantized. Weights stay Q4_0; the activation is what changed representation.
                    graph.task(
                            task,
                            TransformerComputeKernelsQ4_0::matrixVectorGenericQ4_0DP4A,
                            context,
                            state.workspace.wrapXbQuants,
                            state.workspace.wrapXbScales,
                            state.workspace.wrapXbSums,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                } else if (residual) {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ4_0::matrixVectorGenericWithResidualQ4_0,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                } else {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ4_0::matrixVectorGenericQ4_0,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                }
            }
            case Q4_1 -> {
                if (residual) {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ4_1::matrixVectorGenericWithResidualQ4_1,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                } else {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ4_1::matrixVectorGenericQ4_1,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                }
            }
            case Q4_K -> {
                if (residual) {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ4_K::matrixVectorGenericWithResidualQ4_K,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                } else {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ4_K::matrixVectorGenericQ4_K,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                }
            }
            case Q5_K -> {
                if (residual) {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ5_K::matrixVectorGenericWithResidualQ5_K,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                } else {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ5_K::matrixVectorGenericQ5_K,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                }
            }
            case Q6_K -> {
                if (residual) {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ6_K::matrixVectorGenericWithResidualQ6_K,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                } else {
                    graph.task(
                            task,
                            TransformerComputeKernelsQ6_K::matrixVectorGenericQ6_K,
                            context,
                            x,
                            out,
                            w.asByteArray(),
                            n,
                            d,
                            MATVEC_LOCAL);
                }
            }
            default -> throw unsupported(layer, task, role, w.dataType(), "a");
        }
    }

    // @formatter:off
    /**
     * A quantized row must be a whole number of blocks.
     *
     * <p>Every block-decoding kernel here addresses a row as {@code row * blocksPerRow} blocks, so
     * it assumes each row starts on a block boundary. That holds for every projection in a real
     * file — a quantizer will not split a block across rows — and when it does not hold the kernel
     * reads a neighbouring row's blocks, producing weights of plausible magnitude and fluent, wrong
     * output. Checked here because the alternative is discovering it as a numerical disagreement on
     * a model that takes minutes to load.
     */
    // @formatter:on
    private void requireWholeBlocks(int layer, String task, String role, DataType type, int n) {
        int blockSize =
                switch (type) {
                    case Q4_0, Q4_1, Q8_0 -> 32;
                    case Q4_K, Q5_K, Q6_K -> 256;
                    default -> 1;
                };
        if (n % blockSize != 0) {
            throw new UnsupportedOperationException(
                    "qwen35 layer "
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
                            + "-weight blocks. Every block-decoding kernel addresses a row by its"
                            + " block offset, so a partial row would read the next row's blocks.");
        }
    }

    private UnsupportedOperationException unsupported(
            int layer, String task, String role, DataType type, String article) {
        return new UnsupportedOperationException(
                "qwen35 layer "
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
                        + " get one: that would hide a missing kernel behind a memory cost and a"
                        + " silent change of arithmetic.");
    }

    /**
     * The fused gate/up feed-forward, by the representation both weights share.
     *
     * <p>The one task here that decodes two weight matrices in one pass, so it is the one task
     * whose operands must agree. They are checked rather than assumed: a Q4_0 kernel handed a Q5_K
     * operand reads 176-byte super-blocks as ten 18-byte blocks and produces weights of plausible
     * magnitude.
     */
    private void fusedGateUp(
            TaskGraph graph, int layer, TornadoTensor gate, TornadoTensor up, FloatArray x) {
        FusedOperandSupport.requireUniform(
                "qwen35 layer " + layer + " fused gate/up feed-forward",
                List.of("ffn_gate", "ffn_up"),
                gate,
                up);
        boolean packed =
                gate.dataType() == DataType.Q4_0
                        && x == qwen35State.workspace.wrapXb
                        && normedActivationQuantized;
        dispatches.add(
                new Dispatch(layer, "ffn_gate_up", "ffn_gate|ffn_up", gate.dataType(), packed));
        if (packed) {
            graph.task(
                    "ffn_gate_up",
                    TransformerComputeKernelsQ4_0::fusedFFNGateUpSiLUQ4_0DP4A,
                    context,
                    qwen35State.workspace.wrapXbQuants,
                    qwen35State.workspace.wrapXbScales,
                    qwen35State.workspace.wrapXbSums,
                    qwen35State.workspace.wrapHb,
                    gate.asByteArray(),
                    up.asByteArray(),
                    config.dim(),
                    config.hiddenDim(),
                    MATVEC_LOCAL);
            return;
        }
        switch (gate.dataType()) {
            case Q4_0 ->
                    graph.task(
                            "ffn_gate_up",
                            TransformerComputeKernelsQ4_0::fusedFFNGateUpSiLUQ4_0,
                            context,
                            x,
                            qwen35State.workspace.wrapHb,
                            gate.asByteArray(),
                            up.asByteArray(),
                            config.dim(),
                            config.hiddenDim(),
                            MATVEC_LOCAL);
            case Q4_K ->
                    graph.task(
                            "ffn_gate_up",
                            TransformerComputeKernelsQ4_K::fusedFFNGateUpSiLUQ4_K,
                            context,
                            x,
                            qwen35State.workspace.wrapHb,
                            gate.asByteArray(),
                            up.asByteArray(),
                            config.dim(),
                            config.hiddenDim(),
                            MATVEC_LOCAL);
            case Q8_0 ->
                    graph.task(
                            "ffn_gate_up",
                            TransformerComputeKernelsLayered
                                    ::fusedFeedForwardWithSiLUAndGLUActivationQ8_0Byte,
                            context,
                            x,
                            qwen35State.workspace.wrapHb,
                            gate.asByteArray(),
                            up.asByteArray(),
                            config.dim(),
                            config.hiddenDim(),
                            MATVEC_LOCAL);
            case F16 ->
                    graph.task(
                            "ffn_gate_up",
                            TransformerComputeKernelsLayered
                                    ::fusedFeedForwardWithSiLUAndGLUActivation,
                            context,
                            x,
                            qwen35State.workspace.wrapHb,
                            gate.asHalfFloatArray(),
                            up.asHalfFloatArray(),
                            config.dim(),
                            config.hiddenDim(),
                            MATVEC_LOCAL);
            default ->
                    throw unsupported(
                            layer, "ffn_gate_up", "ffn_gate|ffn_up", gate.dataType(), "a");
        }
    }

    // ── the layer graphs ──────────────────────────────────────────────────────

    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        TaskGraph layer = new TaskGraph("layer_" + layerIndex);

        String predecessor = layerIndex == 0 ? activationGraphName : "layer_" + (layerIndex - 1);
        layer.consumeFromDevice(predecessor, qwen35State.workspace.wrapX);
        configureLayerDataTransfers(layer, layerIndex);
        transferLayerWeights(layer, layerIndex);

        // Input normalization, shared by both mixers: xb = attn_norm ⊙ rms(x).
        normalize(
                layer,
                "attn_rms_reduce",
                "attn_rms_finalize",
                "attn_rms_apply",
                qwen35State.workspace.temp,
                require(weights.rms_att_weightLayered, layerIndex, "attn_norm"));

        if (DP4A) {
            // Once per branch, for every Q4_0 projection below that reads the normed activation.
            layer.task(
                    "xb_quantize",
                    TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                    context,
                    qwen35State.workspace.wrapXb,
                    qwen35State.workspace.wrapXbQuants,
                    qwen35State.workspace.wrapXbScales,
                    qwen35State.workspace.wrapXbSums);
        }
        normedActivationQuantized = DP4A;

        if (config.isRecurrentLayer(layerIndex)) {
            deltaNetBranch(layer, layerIndex);
        } else {
            attentionBranch(layer, layerIndex);
        }

        // The feed-forward's input norm is the file's post_attention_norm; there is no ffn_norm.
        // It writes over wrapXb, so whatever was quantized from it no longer describes it.
        normedActivationQuantized = false;
        normalize(
                layer,
                "ffn_rms_reduce",
                "ffn_rms_finalize",
                "ffn_rms_apply",
                qwen35State.workspace.tempFFN,
                require(weights.rms_ffn_weightLayered, layerIndex, "post_attention_norm"));

        if (DP4A) {
            // Its own quantization, of the feed-forward's own activation. The branch's quants
            // describe the attention norm's output, which this is not. The scratch is the same
            // three arrays: the branch's projections are all behind us in this graph, so the
            // buffers are free, and a second set would cost memory to say the same thing.
            layer.task(
                    "ffn_xb_quantize",
                    TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                    context,
                    qwen35State.workspace.wrapXb,
                    qwen35State.workspace.wrapXbQuants,
                    qwen35State.workspace.wrapXbScales,
                    qwen35State.workspace.wrapXbSums);
            normedActivationQuantized = true;
        }

        fusedGateUp(
                layer,
                layerIndex,
                require(weights.w1Layered, layerIndex, "ffn_gate"),
                require(weights.w3Layered, layerIndex, "ffn_up"),
                qwen35State.workspace.wrapXb);
        hiddenActivationQuantized = false;
        if (DP4A && PACKED_FFN_DOWN) {
            // SwiGLU's output, quantized fresh. It is neither of the activations quantized
            // earlier in this layer, and the scratch it shares with them is sized for it.
            layer.task(
                    "ffn_down_quantize",
                    TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                    context,
                    qwen35State.workspace.wrapHb,
                    qwen35State.workspace.wrapXbQuants,
                    qwen35State.workspace.wrapXbScales,
                    qwen35State.workspace.wrapXbSums);
            hiddenActivationQuantized = true;
        }

        matVec(
                layer,
                layerIndex,
                "ffn_down_proj",
                "ffn_down",
                require(weights.w2Layered, layerIndex, "ffn_down"),
                qwen35State.workspace.wrapHb,
                qwen35State.workspace.wrapX,
                config.hiddenDim(),
                config.dim(),
                true);

        layer.persistOnDevice(
                qwen35State.workspace.wrapX,
                keyStore(),
                valueStore(),
                qwen35State.workspace.wrapConvState,
                qwen35State.workspace.wrapDeltaState);
        return layer;
    }

    /** Whether this state's key/value store is half precision. */
    protected boolean fp16Kv() {
        return state.usesFp16KeyValueCache();
    }

    /** The key store the graphs bind, whichever precision it is in. */
    protected Object keyStore() {
        return fp16Kv() ? state.workspace.wrapKeyCacheFP16 : state.workspace.wrapKeyCache;
    }

    protected Object valueStore() {
        return fp16Kv() ? state.workspace.wrapValueCacheFP16 : state.workspace.wrapValueCache;
    }

    /** {@code xb = weight ⊙ rms(x)} — the reduction, its finalize where needed, and the apply. */
    private void normalize(
            TaskGraph layer,
            String reduce,
            String finalize,
            String apply,
            FloatArray scratch,
            TornadoTensor weight) {
        layer.task(
                reduce,
                rmsReduceKernel(),
                context,
                scratch,
                qwen35State.workspace.wrapX,
                config.dim(),
                config.rmsNormEps(),
                qwen35State.localSize);
        if (shouldUseFinalNormalization()) {
            layer.task(
                    finalize,
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    scratch,
                    config.dim(),
                    config.rmsNormEps());
        }
        layer.task(
                apply,
                TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                context,
                qwen35State.workspace.wrapXb,
                qwen35State.workspace.wrapX,
                weight.asFloatArray(),
                scratch);
    }

    // @formatter:off
    /**
     * A full-attention block, from the normalized {@code wrapXb} back into {@code wrapX}.
     *
     * <p>Follows the host branch operation for operation. Three things separate it from Qwen3's:
     * the query projection is twice as wide and carries an interleaved output gate; the rotation
     * covers 64 of a 256-wide head; and the attention result is gated by a logistic before the
     * output projection.
     */
    // @formatter:on
    // @formatter:off
    /**
     * Whether this device's Q4_0 projections read a quantized activation and a packed integer dot
     * product rather than a floating-point one.
     *
     * <p>A device fact, not a user choice: the packed path needs {@code dp4a} to be lowered, and it
     * is worth taking only where that has been measured. Everything else about the decision is a
     * property of the projection — Q4_0 weights, no residual, and an input the branch has already
     * quantized — and is decided where the task is built.
     *
     * <p>Measured on the 27B at chunk 32, interleaved: {@code tg128} 14.73 and 14.72 t/s against
     * 13.70 and 13.61, and {@code tg128@d381} 11.23 against 10.67 and 10.50. Prefill is untouched
     * and unchanged. What it costs is in the parity record: the activation is quantized to eight
     * bits, so the logits move by far more than the floating-point path's bounds allow — relative
     * L2 2.45e-2 against a 1e-4 bound, cosine 0.99970 — while the decisions did not, at 0/63 argmax
     * disagreements and token-identical greedy output over 120 tokens.
     */
    // @formatter:on
    // @formatter:off
    /**
     * Whether {@code wrapXb} still holds the activation {@code xb_quantize} quantized.
     *
     * <p>The buffer is reused inside a layer — the attention branch writes its gated output into
     * it, and the feed-forward norm writes over it again — so the identity of the array a
     * projection reads says nothing about <b>which</b> activation is in it. This says. It is set
     * where the quantization is emitted and cleared at every point the contents change, and it is
     * what the packed-integer dispatch consults; without it, a projection reading {@code wrapXb}
     * after either of those writes would silently consume the previous activation's quants. Today
     * the one projection that would — {@code attn_output_proj} — is excluded for the unrelated
     * reason that it folds a residual, which is not a property worth depending on.
     */
    // @formatter:on
    private boolean normedActivationQuantized;

    /**
     * Whether {@code wrapHb} holds the activation the feed-forward's own quantization describes.
     *
     * <p>Separate from {@link #normedActivationQuantized} because it is a different buffer holding
     * a different activation: SwiGLU's output, which only {@code ffn_down} reads.
     */
    private boolean hiddenActivationQuantized;

    /**
     * TEMPORARY, for the {@code ffn_down} evaluation. Not the accepted default; removed when the
     * tradeoff is decided either way.
     */
    private static final boolean PACKED_FFN_DOWN = Boolean.getBoolean("llama.qwen35.packedFfnDown");

    private static final boolean DP4A =
            TornadoDevices.current()
                            .capabilities()
                            .supports(
                                    org.beehive.gpullama3.runtime.backend.DeviceCapability
                                            .PACKED_INTEGER_DOT)
                    // The escape hatch is for the tests whose subject is addressing rather than
                    // arithmetic: they compare the device against the host exactly, which a
                    // quantized activation cannot do. Not a user option, and not a CLI flag.
                    && !"false"
                            .equalsIgnoreCase(
                                    System.getProperty("llama.qwen35.packedIntegerDot", "true"));

    private void attentionBranch(TaskGraph layer, int layerIndex) {
        final int headDim = config.numberOfHeadsKey();
        final int kvDim = config.kvDim();
        final int attnDim = config.attentionOutputInputDim();

        matVec(
                layer,
                layerIndex,
                "attn_q_proj",
                "attn_q",
                require(weights.wqLayered, layerIndex, "attn_q"),
                qwen35State.workspace.wrapXb,
                qwen35State.workspace.wrapQ,
                config.dim(),
                config.queryGateDim(),
                false);
        matVec(
                layer,
                layerIndex,
                "attn_k_proj",
                "attn_k",
                require(weights.wkLayered, layerIndex, "attn_k"),
                qwen35State.workspace.wrapXb,
                qwen35State.workspace.wrapK,
                config.dim(),
                kvDim,
                false);
        matVec(
                layer,
                layerIndex,
                "attn_v_proj",
                "attn_v",
                require(weights.wvLayered, layerIndex, "attn_v"),
                qwen35State.workspace.wrapXb,
                qwen35State.workspace.wrapV,
                config.dim(),
                kvDim,
                false);

        layer.task(
                "attn_split_query_gate",
                Qwen35AttentionKernels::splitQueryGate,
                context,
                qwen35State.workspace.wrapQ,
                qwen35State.workspace.wrapAttnQ,
                qwen35State.workspace.wrapAttnGate,
                config.numberOfHeads(),
                headDim);

        layer.task(
                "attn_qk_norm",
                Qwen3Kernels::fusedQKRmsNorm,
                context,
                qwen35State.workspace.wrapAttnQ,
                qwen35State.workspace.wrapK,
                require(weights.attnQNorm, layerIndex, "attn_q_norm").asFloatArray(),
                require(weights.attnKNorm, layerIndex, "attn_k_norm").asFloatArray(),
                config.numberOfHeads(),
                config.numberOfKeyValueHeads(),
                headDim,
                headDim,
                config.rmsNormEps());

        layer.task(
                "attn_rope",
                Qwen35AttentionKernels::ropeNeoxPartial,
                context,
                qwen35State.workspace.positionHolder,
                qwen35State.workspace.wrapAttnQ,
                qwen35State.workspace.wrapK,
                weights.freq_cis_realFlat.asFloatArray(),
                weights.freq_cis_imagFlat.asFloatArray(),
                config.numberOfHeads(),
                config.numberOfKeyValueHeads(),
                headDim,
                config.ropeDimensionCount());

        // The key/value store is sized by the blocks that attend, so this layer addresses it by
        // its dense index. Its own index would run four times past the end of the store.
        int kvLayer = config.keyValueLayerIndex(layerIndex);
        if (fp16Kv()) {
            layer.task(
                    "attn_kv_append",
                    Qwen35AttentionKernels::appendKeyValueFP16Paged,
                    context,
                    qwen35State.workspace.positionHolder,
                    qwen35State.workspace.wrapK,
                    qwen35State.workspace.wrapV,
                    qwen35State.workspace.wrapKeyCacheFP16,
                    qwen35State.workspace.wrapValueCacheFP16,
                    qwen35State.workspace.wrapBlockTable,
                    kvDim,
                    kvLayer,
                    qwen35State.kvBlockCfg,
                    qwen35State.kvBlockStride);
        } else {
            layer.task(
                    "attn_kv_append",
                    Qwen35AttentionKernels::appendKeyValuePaged,
                    context,
                    qwen35State.workspace.positionHolder,
                    qwen35State.workspace.wrapK,
                    qwen35State.workspace.wrapV,
                    qwen35State.workspace.wrapKeyCache,
                    qwen35State.workspace.wrapValueCache,
                    qwen35State.workspace.wrapBlockTable,
                    kvDim,
                    kvLayer,
                    qwen35State.kvBlockCfg,
                    qwen35State.kvBlockStride);
        }

        // @formatter:off
        // The single-workgroup online-softmax kernel, on every backend.
        //
        // Not the split-KV (flash-decoding) kernel every other family decodes with: that one
        // stages the query and a per-thread accumulator in local arrays fixed at 128 floats per
        // head, and this family's head is 256 wide. Handing it a 256-wide head reads and writes
        // past those arrays — an illegal address on CUDA, which surfaces as a poisoned context and
        // an allocation failure several calls later rather than as a fault in the kernel that
        // caused it. This kernel sizes its shared memory from the head width it is given.
        //
        // The cost is the parallelism the splits would have given: one workgroup per head rather
        // than eight per head. A split-KV variant whose local arrays are sized from parameters
        // would recover it, and belongs with a measurement rather than ahead of one.
        // @formatter:on
        if (fp16Kv()) {
            layer.task(
                    "attention",
                    TransformerPagedKvKernels::processHeadsFlashAttentionFP16Paged,
                    context,
                    qwen35State.workspace.wrapAttnQ,
                    qwen35State.workspace.wrapKeyCacheFP16,
                    qwen35State.workspace.wrapValueCacheFP16,
                    qwen35State.workspace.wrapXb,
                    config.numberOfHeads(),
                    headDim,
                    kvDim,
                    config.kvMul(),
                    qwen35State.workspace.positionHolder,
                    kvLayer,
                    qwen35State.workspace.wrapBlockTable,
                    qwen35State.kvBlockCfg,
                    qwen35State.kvBlockStride);
        } else {
            layer.task(
                    "attention",
                    TransformerPagedKvKernels::processHeadsFlashAttentionPaged,
                    context,
                    qwen35State.workspace.wrapAttnQ,
                    qwen35State.workspace.wrapKeyCache,
                    qwen35State.workspace.wrapValueCache,
                    qwen35State.workspace.wrapXb,
                    config.numberOfHeads(),
                    headDim,
                    kvDim,
                    config.kvMul(),
                    qwen35State.workspace.positionHolder,
                    kvLayer,
                    qwen35State.workspace.wrapBlockTable,
                    qwen35State.kvBlockCfg,
                    qwen35State.kvBlockStride);
        }

        // A logistic, not a SiLU: reusing the SwiGLU kernel would multiply by the gate twice.
        // The gated attention output lands in wrapXb, over the activation that was quantized.
        normedActivationQuantized = false;
        layer.task(
                "attn_output_gate",
                Qwen35AttentionKernels::applyOutputGate,
                context,
                qwen35State.workspace.wrapXb,
                qwen35State.workspace.wrapAttnGate,
                attnDim);

        matVec(
                layer,
                layerIndex,
                "attn_output_proj",
                "attn_output",
                require(weights.woLayered, layerIndex, "attn_output"),
                qwen35State.workspace.wrapXb,
                qwen35State.workspace.wrapX,
                attnDim,
                config.dim(),
                true);
    }

    // @formatter:off
    /**
     * A Gated Delta Net block, from the normalized {@code wrapXb} back into {@code wrapX}.
     *
     * <p>Follows the host branch operation for operation, including the two orderings that are not
     * interchangeable: the fused projection is convolved <b>before</b> it is split, and the queries
     * and keys are normalized <b>after</b> the convolution.
     *
     * <p>The convolution window and the delta-net state are per-layer slices of one array each, so
     * a layer passes its offset rather than binding its own buffer — 48 buffers per kind would be
     * 48 transfers to arrange and keep resident.
     */
    // @formatter:on
    private void deltaNetBranch(TaskGraph layer, int layerIndex) {
        final int convDim = config.deltaNetConvDim();
        final int keyDim = config.deltaNetKeyDim();
        final int valueDim = config.deltaNetValueDim();
        final int valueHeads = config.numberOfValueHeads();
        final int headK = config.headKeyDim();
        final int headV = config.headValueDim();
        final int recurrent = config.recurrentLayerIndex(layerIndex);

        matVec(
                layer,
                layerIndex,
                "ssm_qkv_proj",
                "attn_qkv",
                require(weights.ssmQkv, layerIndex, "attn_qkv"),
                qwen35State.workspace.wrapXb,
                qwen35State.workspace.wrapSsmQkv,
                config.dim(),
                convDim,
                false);
        matVec(
                layer,
                layerIndex,
                "ssm_gate_proj",
                "attn_gate",
                require(weights.ssmGate, layerIndex, "attn_gate"),
                qwen35State.workspace.wrapXb,
                qwen35State.workspace.wrapSsmZ,
                config.dim(),
                valueDim,
                false);
        matVec(
                layer,
                layerIndex,
                "ssm_beta_proj",
                "ssm_beta",
                require(weights.ssmBeta, layerIndex, "ssm_beta"),
                qwen35State.workspace.wrapXb,
                qwen35State.workspace.wrapSsmBeta,
                config.dim(),
                valueHeads,
                false);
        matVec(
                layer,
                layerIndex,
                "ssm_alpha_proj",
                "ssm_alpha",
                require(weights.ssmAlpha, layerIndex, "ssm_alpha"),
                qwen35State.workspace.wrapXb,
                qwen35State.workspace.wrapSsmAlpha,
                config.dim(),
                valueHeads,
                false);

        layer.task(
                "ssm_decay_beta",
                Qwen35DeltaNetKernels::decayAndBeta,
                context,
                qwen35State.workspace.wrapSsmAlpha,
                qwen35State.workspace.wrapSsmBeta,
                require(weights.ssmDtBias, layerIndex, "ssm_dt.bias").asFloatArray(),
                require(weights.ssmA, layerIndex, "ssm_a").asFloatArray(),
                valueHeads);

        layer.task(
                "ssm_conv",
                Qwen35DeltaNetKernels::causalConv1d,
                context,
                qwen35State.workspace.wrapSsmQkv,
                require(weights.ssmConv1d, layerIndex, "ssm_conv1d").asFloatArray(),
                qwen35State.workspace.wrapConvState,
                qwen35State.workspace.wrapSsmConvOut,
                convDim,
                config.ssmConvKernel(),
                recurrent * config.convStateSize());
        layer.task(
                "ssm_conv_silu",
                Qwen35DeltaNetKernels::siluInPlace,
                context,
                qwen35State.workspace.wrapSsmConvOut,
                convDim);

        layer.task(
                "ssm_split_qkv",
                TransformerComputeKernels::splitThreeWay,
                context,
                qwen35State.workspace.wrapSsmConvOut,
                qwen35State.workspace.wrapSsmQ,
                qwen35State.workspace.wrapSsmK,
                qwen35State.workspace.wrapSsmV,
                keyDim,
                keyDim,
                valueDim);

        layer.task(
                "ssm_l2norm_q",
                Qwen35DeltaNetKernels::l2NormPerHead,
                context,
                qwen35State.workspace.wrapSsmQ,
                config.numberOfKeyHeads(),
                headK,
                config.rmsNormEps());
        layer.task(
                "ssm_l2norm_k",
                Qwen35DeltaNetKernels::l2NormPerHead,
                context,
                qwen35State.workspace.wrapSsmK,
                config.numberOfKeyHeads(),
                headK,
                config.rmsNormEps());
        layer.task(
                "ssm_scale_q",
                TransformerComputeKernels::scaleInPlace,
                context,
                qwen35State.workspace.wrapSsmQ,
                (float) (1.0 / Math.sqrt(headK)),
                keyDim);

        layer.task(
                "ssm_delta_rule",
                Qwen35DeltaNetKernels::deltaRule,
                context,
                qwen35State.workspace.wrapSsmQ,
                qwen35State.workspace.wrapSsmK,
                qwen35State.workspace.wrapSsmV,
                qwen35State.workspace.wrapSsmAlpha,
                qwen35State.workspace.wrapSsmBeta,
                qwen35State.workspace.wrapDeltaState,
                qwen35State.workspace.wrapSsmOut,
                valueHeads,
                config.numberOfKeyHeads(),
                headV,
                recurrent * config.deltaNetStateSize());

        layer.task(
                "ssm_gated_norm",
                Qwen35DeltaNetKernels::gatedNormPerHead,
                context,
                qwen35State.workspace.wrapSsmOut,
                qwen35State.workspace.wrapSsmZ,
                require(weights.ssmNorm, layerIndex, "ssm_norm").asFloatArray(),
                valueHeads,
                headV,
                config.rmsNormEps());

        matVec(
                layer,
                layerIndex,
                "ssm_out_proj",
                "ssm_out",
                require(weights.ssmOut, layerIndex, "ssm_out"),
                qwen35State.workspace.wrapSsmOut,
                qwen35State.workspace.wrapX,
                valueDim,
                config.dim(),
                true);
    }

    // ── transfers ─────────────────────────────────────────────────────────────

    /**
     * This layer's weights, uploaded once in the graph that first reads them.
     *
     * <p>A weight array bound with {@code transferToDevice} in two graphs of one execution plan
     * gets a device buffer in each, so a layer uploads only its own and never another's.
     */
    // @formatter:off
    /**
     * The graph that has already uploaded this layer's weights, or {@code null} to upload them
     * here.
     *
     * <p>A weight array bound with {@code transferToDevice} in two graphs of one execution plan
     * gets a device buffer in each, so a plan holding both a batch-prefill and a decode family
     * would hold the model twice. The decode family consumes what the batch family uploaded.
     */
    // @formatter:on
    protected String weightSourceGraphName(int layerIndex) {
        return null;
    }

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
        String source = weightSourceGraphName(layerIndex);
        if (source != null) {
            layer.consumeFromDevice(source, tensors.toArray());
        } else {
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, tensors.toArray());
        }
    }

    /** A tensor's device array, in whatever representation it is retained in. */
    private static Object deviceArray(TornadoTensor tensor) {
        return switch (tensor.dataType()) {
            case F32 -> tensor.asFloatArray();
            case F16 -> tensor.asHalfFloatArray();
            default -> tensor.asByteArray();
        };
    }

    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph layer, int layerIndex) {
        if (layerIndex == 0) {
            layer.transferToDevice(
                    DataTransferMode.EVERY_EXECUTION,
                    qwen35State.workspace.positionHolder,
                    qwen35State.workspace.temp,
                    qwen35State.workspace.tempFFN,
                    qwen35State.workspace.wrapBlockTable);
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    context,
                    qwen35State.workspace.wrapXb,
                    qwen35State.workspace.wrapQ,
                    qwen35State.workspace.wrapAttnQ,
                    qwen35State.workspace.wrapAttnGate,
                    qwen35State.workspace.wrapK,
                    qwen35State.workspace.wrapV,
                    keyStore(),
                    valueStore(),
                    qwen35State.workspace.wrapAtt,
                    qwen35State.workspace.wrapHb);
            if (DP4A) {
                layer.transferToDevice(
                        DataTransferMode.FIRST_EXECUTION,
                        qwen35State.workspace.wrapXbQuants,
                        qwen35State.workspace.wrapXbScales,
                        qwen35State.workspace.wrapXbSums);
            }
            // The recurrent state persists across tokens and is updated in place, so it is
            // uploaded once — zeroed — and never read back. Uploading it every execution would
            // overwrite the device's own history with the host's stale copy.
            layer.transferToDevice(
                    DataTransferMode.FIRST_EXECUTION,
                    qwen35State.workspace.wrapSsmQkv,
                    qwen35State.workspace.wrapSsmConvOut,
                    qwen35State.workspace.wrapSsmZ,
                    qwen35State.workspace.wrapSsmAlpha,
                    qwen35State.workspace.wrapSsmBeta,
                    qwen35State.workspace.wrapSsmQ,
                    qwen35State.workspace.wrapSsmK,
                    qwen35State.workspace.wrapSsmV,
                    qwen35State.workspace.wrapSsmOut,
                    qwen35State.workspace.wrapConvState,
                    qwen35State.workspace.wrapDeltaState);
        } else {
            String predecessor = "layer_" + (layerIndex - 1);
            layer.consumeFromDevice(
                    predecessor,
                    context,
                    qwen35State.workspace.wrapXb,
                    qwen35State.workspace.wrapQ,
                    qwen35State.workspace.wrapAttnQ,
                    qwen35State.workspace.wrapAttnGate,
                    qwen35State.workspace.wrapK,
                    qwen35State.workspace.wrapV,
                    keyStore(),
                    valueStore(),
                    qwen35State.workspace.wrapAtt,
                    qwen35State.workspace.wrapHb,
                    qwen35State.workspace.positionHolder);
            layer.consumeFromDevice(predecessor, qwen35State.workspace.wrapBlockTable);
            if (DP4A) {
                layer.consumeFromDevice(
                        predecessor,
                        qwen35State.workspace.wrapXbQuants,
                        qwen35State.workspace.wrapXbScales,
                        qwen35State.workspace.wrapXbSums);
            }
            layer.consumeFromDevice(
                    predecessor,
                    qwen35State.workspace.temp,
                    qwen35State.workspace.tempFFN,
                    qwen35State.workspace.wrapSsmQkv,
                    qwen35State.workspace.wrapSsmConvOut,
                    qwen35State.workspace.wrapSsmZ,
                    qwen35State.workspace.wrapSsmAlpha,
                    qwen35State.workspace.wrapSsmBeta,
                    qwen35State.workspace.wrapSsmQ,
                    qwen35State.workspace.wrapSsmK,
                    qwen35State.workspace.wrapSsmV,
                    qwen35State.workspace.wrapSsmOut,
                    qwen35State.workspace.wrapConvState,
                    qwen35State.workspace.wrapDeltaState);
        }
        return layer;
    }

    // ── worker grids ──────────────────────────────────────────────────────────

    @Override
    public GridScheduler updateGridScheduler(GridScheduler scheduler) {
        WorkerGrid rmsReduce =
                rmsReduceWorker(
                        WorkerGridFactory.createRmsNormWorker(config.dim(), state.localSize));
        WorkerGrid rmsApply = WorkerGridFactory.createRmsNormWorker(config.dim(), state.localSize);
        WorkerGrid rmsFinalize =
                WorkerGridFactory.createRmsNormWorker(config.dim(), state.localSize);

        final int headDim = config.numberOfHeadsKey();
        WorkerGrid queryGate =
                WorkerGridFactory.genericWorker(config.queryGateDim(), ELEMENTWISE_LOCAL);
        WorkerGrid qkNorm =
                WorkerGridFactory.genericWorker(
                        (config.numberOfHeads() + config.numberOfKeyValueHeads()) * headDim,
                        headDim);
        WorkerGrid rope =
                WorkerGridFactory.genericWorker(
                        config.numberOfHeads() * (config.ropeDimensionCount() / 2), 32);
        WorkerGrid kvAppend = WorkerGridFactory.genericWorker(config.kvDim(), ELEMENTWISE_LOCAL);
        WorkerGrid attention =
                WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), headDim);
        WorkerGrid outputGate =
                WorkerGridFactory.genericWorker(
                        config.attentionOutputInputDim(), ELEMENTWISE_LOCAL);

        WorkerGrid convDim =
                WorkerGridFactory.genericWorker(config.deltaNetConvDim(), ELEMENTWISE_LOCAL);
        WorkerGrid keyDim =
                WorkerGridFactory.genericWorker(config.deltaNetKeyDim(), ELEMENTWISE_LOCAL);
        WorkerGrid keyHeads =
                WorkerGridFactory.genericWorker(
                        config.numberOfKeyHeads(), config.numberOfKeyHeads());
        WorkerGrid valueHeads =
                WorkerGridFactory.genericWorker(
                        config.numberOfValueHeads(), config.numberOfValueHeads());
        WorkerGrid deltaRule =
                WorkerGridFactory.genericWorker(
                        config.numberOfValueHeads() * config.headValueDim(), ELEMENTWISE_LOCAL);

        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            String prefix = "layer_" + layer + ".";
            scheduler.addWorkerGrid(prefix + "attn_rms_reduce", rmsReduce);
            scheduler.addWorkerGrid(prefix + "ffn_rms_reduce", rmsReduce);
            if (shouldUseFinalNormalization()) {
                scheduler.addWorkerGrid(prefix + "attn_rms_finalize", rmsFinalize);
                scheduler.addWorkerGrid(prefix + "ffn_rms_finalize", rmsFinalize);
            }
            scheduler.addWorkerGrid(prefix + "attn_rms_apply", rmsApply);
            if (DP4A) {
                WorkerGrid quantize = WorkerGridFactory.genericWorker(config.dim(), 32);
                scheduler.addWorkerGrid(prefix + "xb_quantize", quantize);
                scheduler.addWorkerGrid(
                        prefix + "ffn_xb_quantize",
                        WorkerGridFactory.genericWorker(config.dim(), 32));
                if (PACKED_FFN_DOWN) {
                    scheduler.addWorkerGrid(
                            prefix + "ffn_down_quantize",
                            WorkerGridFactory.genericWorker(config.hiddenDim(), 32));
                }
            }
            scheduler.addWorkerGrid(prefix + "ffn_rms_apply", rmsApply);
            scheduler.addWorkerGrid(prefix + "ffn_gate_up", matVecWorker(config.hiddenDim()));
            scheduler.addWorkerGrid(prefix + "ffn_down_proj", matVecWorker(config.dim()));

            if (config.isRecurrentLayer(layer)) {
                scheduler.addWorkerGrid(
                        prefix + "ssm_qkv_proj", matVecWorker(config.deltaNetConvDim()));
                scheduler.addWorkerGrid(
                        prefix + "ssm_gate_proj", matVecWorker(config.deltaNetValueDim()));
                scheduler.addWorkerGrid(
                        prefix + "ssm_beta_proj", matVecWorker(config.numberOfValueHeads()));
                scheduler.addWorkerGrid(
                        prefix + "ssm_alpha_proj", matVecWorker(config.numberOfValueHeads()));
                scheduler.addWorkerGrid(prefix + "ssm_decay_beta", valueHeads);
                scheduler.addWorkerGrid(prefix + "ssm_conv", convDim);
                scheduler.addWorkerGrid(prefix + "ssm_conv_silu", convDim);
                scheduler.addWorkerGrid(prefix + "ssm_split_qkv", convDim);
                scheduler.addWorkerGrid(prefix + "ssm_l2norm_q", keyHeads);
                scheduler.addWorkerGrid(prefix + "ssm_l2norm_k", keyHeads);
                scheduler.addWorkerGrid(prefix + "ssm_scale_q", keyDim);
                scheduler.addWorkerGrid(prefix + "ssm_delta_rule", deltaRule);
                scheduler.addWorkerGrid(prefix + "ssm_gated_norm", valueHeads);
                scheduler.addWorkerGrid(prefix + "ssm_out_proj", matVecWorker(config.dim()));
            } else {
                scheduler.addWorkerGrid(
                        prefix + "attn_q_proj", matVecWorker(config.queryGateDim()));
                scheduler.addWorkerGrid(prefix + "attn_k_proj", matVecWorker(config.kvDim()));
                scheduler.addWorkerGrid(prefix + "attn_v_proj", matVecWorker(config.kvDim()));
                scheduler.addWorkerGrid(prefix + "attn_split_query_gate", queryGate);
                scheduler.addWorkerGrid(prefix + "attn_qk_norm", qkNorm);
                scheduler.addWorkerGrid(prefix + "attn_rope", rope);
                scheduler.addWorkerGrid(prefix + "attn_kv_append", kvAppend);
                scheduler.addWorkerGrid(prefix + "attention", attention);
                scheduler.addWorkerGrid(prefix + "attn_output_gate", outputGate);
                scheduler.addWorkerGrid(prefix + "attn_output_proj", matVecWorker(config.dim()));
            }
        }
        return scheduler;
    }

    /** One workgroup per output row, which is how every matrix-vector kernel here is written. */
    private static WorkerGrid matVecWorker(int rows) {
        return WorkerGridFactory.genericWorker(rows * MATVEC_LOCAL, MATVEC_LOCAL);
    }
}
