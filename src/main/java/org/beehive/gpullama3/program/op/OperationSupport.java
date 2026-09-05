package org.beehive.gpullama3.program.op;

import java.util.EnumSet;
import java.util.Objects;
import java.util.Set;
import org.beehive.gpullama3.runtime.tensor.DataType;
import org.beehive.gpullama3.runtime.tensor.ExecutionTarget;

/**
 * Which representations each execution target can actually run each operation at.
 *
 * <p>{@link #require} throws naming the operation, the representation and the target. It never
 * falls back to another representation, never converts silently, and never lets an unsupported pair
 * reach a kernel. That is the same refusal {@code ForwardPlanFactory} already makes for a
 * format-decoded dtype, moved to where the vocabulary can state it.
 */
public final class OperationSupport {

    private OperationSupport() {}

    /**
     * The representations {@code target} can run {@code kind} at.
     *
     * <p>An empty set means the target does not implement the operation at all, which is a
     * different thing from implementing it at no useful representation — {@link
     * OperationKind#MAT_MUL} on the CPU is the case: the host's batched prefill holds an array of
     * per-row tensors and multiplies them one row at a time, so what it has is {@link
     * OperationKind#MAT_VEC} repeated, not a matrix-matrix product.
     *
     * @return an immutable set; never {@code null}
     */
    public static Set<DataType> supported(OperationKind kind, ExecutionTarget target) {
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(target, "target");
        return switch (target) {
            case CPU -> cpu(kind);
            case GPU -> gpu(kind);
        };
    }

    /** Whether {@code target} can run {@code kind} at {@code dataType}. */
    public static boolean supports(OperationKind kind, DataType dataType, ExecutionTarget target) {
        Objects.requireNonNull(dataType, "dataType");
        return supported(kind, target).contains(dataType);
    }

    /**
     * Refuses an operation the target cannot run, naming everything the reader needs.
     *
     * @throws UnsupportedOperationException if {@code target} cannot run {@code operation} at its
     *     {@linkplain Operation#dataType() representation}
     */
    public static void require(Operation operation, ExecutionTarget target) {
        Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(target, "target");
        if (!supports(operation.kind(), operation.dataType(), target)) {
            Set<DataType> available = supported(operation.kind(), target);
            throw new UnsupportedOperationException(
                    operation.kind()
                            + " is not supported for "
                            + operation.dataType()
                            + " on "
                            + target
                            + (available.isEmpty()
                                    ? "; "
                                            + target
                                            + " does not implement "
                                            + operation.kind()
                                            + " at all"
                                    : "; " + target + " supports " + available));
        }
    }

    /**
     * The host.
     *
     * <p>Quantized weights are read where they lie: {@code Q4_0}, {@code Q4_K}, {@code Q5_K} and
     * {@code Q6_K} are decoded inside the dot product by the corresponding {@code *FloatTensor},
     * which is why they appear here and nowhere in {@link #gpu}. There is no decode step before the
     * multiply and no decode operation to place there.
     *
     * <p>Activations are {@code F32} on this path — the host accumulates in float whatever the
     * weights are stored as.
     */
    private static Set<DataType> cpu(OperationKind kind) {
        return switch (kind) {
                // Weight-bearing: dispatch follows the representation the weights are read in.
            case MAT_VEC, EMBEDDING_LOOKUP, VOCAB_PROJECTION ->
                    Set.of(
                            DataType.F32,
                            DataType.F16,
                            DataType.BF16,
                            DataType.Q8_0,
                            DataType.Q4_0,
                            DataType.Q4_K,
                            DataType.Q5_K,
                            DataType.Q6_K);

                // Activation-only: the host computes in float.
            case RMS_NORM,
                            ROPE,
                            KV_APPEND,
                            ATTENTION,
                            SOFTMAX,
                            SWIGLU,
                            GEGLU,
                            RESIDUAL_ADD,
                            BIAS_ADD,
                            SCALE,
                            SPLIT_FUSED_QKV,
                            SPLIT_GATE_UP,
                            LOGIT_SOFT_CAP,
                            ARG_MAX,
                            SAMPLE ->
                    Set.of(DataType.F32);

                // Mixture of experts: routing and accumulation are float work; expert weights carry
                // whatever representation the stacked tensors were materialized in.
            case MOE_ROUTER, WEIGHTED_ACCUMULATE -> Set.of(DataType.F32);
            case EXPERT_FEED_FORWARD ->
                    Set.of(
                            DataType.F32,
                            DataType.F16,
                            DataType.BF16,
                            DataType.Q8_0,
                            DataType.Q4_0,
                            DataType.Q4_K,
                            DataType.Q5_K,
                            DataType.Q6_K);

                // No matrix-matrix implementation: batched prefill repeats single rows.
            case MAT_MUL -> Set.of();
        };
    }

    /**
     * The accelerator, through TornadoVM.
     *
     * <p>Weights arrive here already materialized in a representation a kernel exists for, which is
     * why the format-decoded types are absent rather than refused case by case: the loader turned
     * them into {@code Q8_0} before dispatch ever saw them, and {@code BF16} was narrowed to {@code
     * F16}.
     *
     * <p>{@link OperationKind#ATTENTION} carries {@code F32} and {@code F16} because the key/value
     * cache has both representations today — the FP16 cache is this parameter, not a separate
     * operation. Sampling is {@code F32} and is here at all because sampling is an operation and
     * may execute on the device (Rule 8b), which is what makes the existing device-sample path
     * expressible rather than a special case.
     */
    private static Set<DataType> gpu(OperationKind kind) {
        return switch (kind) {
                // Weight-bearing: the two representations plans are built for today.
            case MAT_VEC, MAT_MUL, EMBEDDING_LOOKUP, VOCAB_PROJECTION ->
                    Set.of(DataType.F16, DataType.Q8_0);

                // Key/value representation, F32 by default and F16 behind the FP16 cache.
            case KV_APPEND, ATTENTION -> Set.of(DataType.F32, DataType.F16);

                // Activations, in float or half depending on the kernel variant in use.
            case RMS_NORM,
                            ROPE,
                            SOFTMAX,
                            SWIGLU,
                            GEGLU,
                            RESIDUAL_ADD,
                            BIAS_ADD,
                            SCALE,
                            SPLIT_FUSED_QKV,
                            SPLIT_GATE_UP ->
                    Set.of(DataType.F32, DataType.F16);

                // Logits are float on every path that samples on the device.
            case LOGIT_SOFT_CAP, ARG_MAX, SAMPLE -> Set.of(DataType.F32);

            case MOE_ROUTER, EXPERT_FEED_FORWARD, WEIGHTED_ACCUMULATE -> Set.of();
        };
    }

    /**
     * Every representation any target runs any operation at — the union of the tables above.
     *
     * <p>Useful to a test that wants to assert something about the whole surface without
     * hard-coding it a second time.
     */
    public static Set<DataType> everySupportedDataType() {
        EnumSet<DataType> all = EnumSet.noneOf(DataType.class);
        for (OperationKind kind : OperationKind.values()) {
            for (ExecutionTarget target : ExecutionTarget.values()) {
                all.addAll(supported(kind, target));
            }
        }
        return Set.copyOf(all);
    }
}
