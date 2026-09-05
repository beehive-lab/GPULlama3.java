package org.beehive.gpullama3.runtime.tensor;

import org.beehive.gpullama3.api.Experimental;

/**
 * How values are represented where the engine actually computes with them.
 *
 * <p>This is the runtime's own vocabulary, deliberately smaller than the file format's. GGUF's
 * {@code GGMLType} describes what is in a file; {@code DataType} describes what executes. The two
 * are related by an explicit mapping in the format layer rather than by being the same enum, which
 * is what lets the runtime, program, operation and backend layers stay free of format types (Rule
 * 4).
 *
 * <h2>What is not here</h2>
 *
 * <p>{@link #Q4_0}, {@link #Q4_K}, {@link #Q5_K} and {@link #Q6_K} are here for one specific
 * reason: the CPU path executes them directly, decoding blocks inside the dot product. They are
 * {@linkplain #isFormatDecoded() format-decoded} — no target materializes a tensor in them, and the
 * GPU path does not execute them at all: the loader materializes {@link #Q8_0} instead. That is why
 * the format mapping takes a load target rather than being a global function.
 *
 * <p>Nothing depends on this type yet; it is introduced ahead of the descriptors and the mapping
 * that use it. Its public exposure arrives with the façade's dtype accessors.
 */
@Experimental
public enum DataType {

    /** 32-bit float. What the CPU accumulates in, whatever the weights are stored as. */
    F32(false, false),

    /** 16-bit float. Stored and computed with directly on the GPU path. */
    F16(false, false),

    /**
     * 16-bit brain float: the same exponent range as {@link #F32} with fewer mantissa bits.
     *
     * <p>Not {@linkplain #isFormatDecoded() format-decoded}: the CPU materializes a tensor in this
     * representation and reads it directly. The GPU does not execute it, and converts to {@link
     * #F16} at load instead — a narrowing that loses exponent range, which is why the conversion is
     * stated in {@link #materializedFallback()} rather than left implicit. The device type exists
     * (TornadoVM's {@code BFloat16Array}, 5.2.0), so this is today's behaviour and not a permanent
     * limit.
     */
    BF16(false, false),

    /** 8-bit block quantization: signed 8-bit values with a per-block scale. */
    Q8_0(true, false),

    /**
     * 4-bit block quantization, 32 values to a block with one scale. Like the K-quants it is
     * decoded during compute on the CPU and materialized as {@link #Q8_0} for the GPU.
     */
    Q4_0(true, true),

    /**
     * 4-bit K-quantization. CPU only, decoded during compute; the GPU materializes {@link #Q8_0}.
     */
    Q4_K(true, true),

    /**
     * 5-bit K-quantization. CPU only, decoded during compute; the GPU materializes {@link #Q8_0}.
     */
    Q5_K(true, true),

    /**
     * 6-bit K-quantization. CPU only, decoded during compute; the GPU materializes {@link #Q8_0}.
     */
    Q6_K(true, true);

    private final boolean quantized;
    private final boolean formatDecoded;

    DataType(boolean quantized, boolean formatDecoded) {
        this.quantized = quantized;
        this.formatDecoded = formatDecoded;
    }

    /** Whether values are stored in blocks with scales rather than as plain floats. */
    public boolean isQuantized() {
        return quantized;
    }

    /**
     * Whether this representation is only ever <i>decoded</i> during compute, never materialized as
     * a tensor of its own.
     *
     * <p>The distinction is not decorative. A format-decoded type has no storage form a backend can
     * be asked to allocate, so it can never be the target of a materialization, and a plan cannot
     * be built for it on a backend that does not decode it. The K-quants are the case: the CPU
     * decodes them in the dot product, and the GPU never sees them because the loader turns them
     * into {@link #Q8_0} first.
     */
    public boolean isFormatDecoded() {
        return formatDecoded;
    }

    /**
     * The representation a target must materialize to run this one, when it cannot execute it
     * directly.
     *
     * <p>Stated here for the format-decoded types and for {@link #BF16}, and only as what it is —
     * the fallback that exists today. <b>Which target uses it is not this type's business</b>: that
     * is the mapping's, which takes a load target precisely because the answer differs between CPU
     * and GPU.
     *
     * @return the fallback representation, or this type when it needs none
     */
    public DataType materializedFallback() {
        return switch (this) {
            case Q4_0, Q4_K, Q5_K, Q6_K -> Q8_0; // format-decoded: the GPU never sees them
            case BF16 -> F16; // narrowed at load; no BF16 kernels yet
            default -> this;
        };
    }
}
