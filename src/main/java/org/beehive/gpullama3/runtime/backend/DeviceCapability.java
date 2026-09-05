package org.beehive.gpullama3.runtime.backend;

import java.util.Locale;
import java.util.Objects;
import org.beehive.gpullama3.api.Experimental;

/**
 * One thing a device can do that changes what is lowered onto it.
 *
 * <p>A value rather than an enum, for {@link BackendId}'s reason: a backend that gains a capability
 * should not require an edit in the layer above it. The constants below are the ones the tree
 * actually branches on today.
 *
 * <p>The bar for adding one, from {@code ProgramCacheKey}: <b>if changing it can change task count,
 * task names, kernels, grid entries or bindings, it belongs here</b> — and therefore in the cache
 * key. A capability that changes nothing observable does not need naming.
 */
@Experimental
public final class DeviceCapability {

    /**
     * Warp/sub-group shuffle reductions ({@code KernelContext.simdShuffleDown}) produce correct
     * results. PTX does; the OpenCL backend compiles the shuffle and computes the wrong answer, so
     * the warp-shuffle GEMV kernels fall back to the shared-memory variants there.
     */
    public static final DeviceCapability WARP_SHUFFLE = of("warp-shuffle");

    /**
     * Tensor-core MMA intrinsics ({@code mmaLoadA/B}, {@code mma}, {@code mmaStore}) lower. CUDA
     * only.
     */
    public static final DeviceCapability TENSOR_CORE_MMA = of("tensor-core-mma");

    /**
     * The multi-workgroup split-KV flash-decoding attention kernel JITs. Metal fails to, so Qwen3
     * falls back to the single-workgroup-per-head online-softmax kernel there.
     */
    public static final DeviceCapability SPLIT_KV_ATTENTION = of("split-kv-attention");

    /**
     * Root-mean-square normalization completes in one pass. Where it does not, lowering emits an
     * extra {@code *_rms_finalize} task per block — the same program, a different task set, which
     * is exactly why capabilities are in the cache key.
     */
    public static final DeviceCapability SINGLE_PASS_RMS = of("single-pass-rms");

    /**
     * A 32-wide subgroup butterfly reduction over {@code KernelContext.simdShuffleDown} produces
     * correct results for the fused Q/K/V projection kernel family.
     *
     * <p><b>Deliberately narrower than {@link #WARP_SHUFFLE}.</b> That capability is PTX's
     * shuffle-reduction correctness, verified wrong on OpenCL and never measured on Metal at all —
     * granting it to Metal would silently change every other call site gated on it (Qwen3's GEMV
     * kernel selection among them), none of which this capability's verification covers. This one
     * names exactly what was measured: {@code fusedQKVMatmulXSimd32}'s five-step 32-lane butterfly
     * (shuffle widths 16, 8, 4, 2, 1) against a CPU reference, isolated in its own minimal task
     * graph, with no rounding ambiguity in the inputs — exact agreement, no poisoned output
     * remaining, on Metal (Apple Pro, TornadoVM 5.2.0-jdk21). Not evaluated on OpenCL or PTX, where
     * {@link #WARP_SHUFFLE} already answers the equivalent question for the kernels gated on it. A
     * capability that changes nothing observable does not need naming — this one selects between
     * {@code fusedQKVMatmulX} (shared-memory reduction, works everywhere) and {@code
     * fusedQKVMatmulXSimd32} (32-lane shuffle reduction) for the QKV projection task, so it belongs
     * here by this file's own bar.
     */
    public static final DeviceCapability SUBGROUP_SHUFFLE_32 = of("subgroup-shuffle-32");

    /**
     * The doubly-rounded FP16 QKV projection is accurate enough on this device.
     *
     * <p>{@code fusedQKVMatmulX} multiplies a packed FP16 weight pair by a packed FP16 activation
     * pair, so the product is rounded to FP16 before the FP32 accumulator sees it — once per term,
     * over a projection row, in a direction that does not cancel. Withheld on OpenCL, where that
     * cost the Llama-shaped FP16 families their CPU parity (worst relative L2 0.0109 against CUDA's
     * 0.0011 for the identical kernel); where it is withheld, the projection widens each pair
     * before multiplying instead.
     *
     * <p><b>Narrower than the name suggests.</b> Other kernels multiply packed pairs and hold
     * parity on OpenCL — {@code fusedRmsNormFFNGateUp}, which Qwen3 shares, is one. What is
     * specific to the QKV projection is that <i>both</i> operands are FP16: the weights, and an
     * activation {@code mapContextWithQuantize} has already rounded to FP16. Read this as a
     * statement about that combination, not about packed arithmetic generally. Metal keeps the
     * packed path and has not been measured against this question.
     */
    public static final DeviceCapability PACKED_HALF2_MATH = of("packed-half2-math");

    private final String name;

    private DeviceCapability(String name) {
        this.name = name;
    }

    public static DeviceCapability of(String name) {
        Objects.requireNonNull(name, "name");
        String canonical = name.trim().toLowerCase(Locale.ROOT);
        if (canonical.isEmpty()) {
            throw new IllegalArgumentException("a capability name must not be blank");
        }
        return new DeviceCapability(canonical);
    }

    public String name() {
        return name;
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof DeviceCapability that && name.equals(that.name);
    }

    @Override
    public int hashCode() {
        return name.hashCode();
    }

    @Override
    public String toString() {
        return name;
    }
}
