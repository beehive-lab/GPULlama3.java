package org.beehive.gpullama3.runtime.tensor;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * A read-only tensor whose element count exceeds what an {@code int} can index.
 *
 * <p>The engine's {@code FloatTensor} and the device array types are int-indexed, which caps them
 * at 2^31 elements. Gemma-4's per-layer token embedding table is about 2.35 billion elements, so it
 * cannot be wrapped in either. It also never needs to be: only one embedding row is read per token.
 *
 * <p>This type is that case and only that case — long-indexed, element-at-a-time, no bulk
 * materialization. It exists so weight sets can hold such a tensor without holding a {@code
 * GGMLTensorEntry}, which would put GGUF's vocabulary in the runtime (Rule 4). The dequantization
 * lives here, keyed on {@link DataType}, rather than on the file's type.
 *
 * <p>Not a general tensor abstraction, and deliberately not on the {@code TensorDescriptor} path:
 * it describes storage the runtime reads, not a tensor a backend allocates.
 */
public final class LongIndexedTensor {

    private final MemorySegment data;
    private final DataType dataType;

    public LongIndexedTensor(MemorySegment data, DataType dataType) {
        this.data = data;
        this.dataType = dataType;
    }

    /** How the values are represented. */
    public DataType dataType() {
        return dataType;
    }

    /**
     * The value at an absolute element index, converted to float.
     *
     * <p>Element-at-a-time on purpose: callers read a single row out of a tensor far too large to
     * copy, so there is nothing to amortize a bulk path over.
     *
     * @param elementIndex index into the flattened, row-major tensor
     */
    public float valueAt(long elementIndex) {
        return switch (dataType) {
            case F32 -> data.get(ValueLayout.JAVA_FLOAT_UNALIGNED, elementIndex * Float.BYTES);
            case F16 ->
                    Float.float16ToFloat(
                            data.get(ValueLayout.JAVA_SHORT_UNALIGNED, elementIndex * Short.BYTES));
                // BF16 is the top 16 bits of the F32 bit pattern.
            case BF16 ->
                    Float.intBitsToFloat(
                            ((int)
                                            data.get(
                                                    ValueLayout.JAVA_SHORT_UNALIGNED,
                                                    elementIndex * Short.BYTES))
                                    << 16);
            case Q8_0 -> q8_0ValueAt(elementIndex);
            default ->
                    throw new UnsupportedOperationException(
                            "LongIndexedTensor does not read "
                                    + dataType
                                    + "; it is only used for embedding"
                                    + " tables, which are never stored in the format-decoded types");
        };
    }

    /** Q8_0: 32 signed 8-bit values behind one FP16 scale, blocks tiling the row-major data. */
    private float q8_0ValueAt(long elementIndex) {
        final int blockSize = 32;
        final int blockBytes = Short.BYTES + blockSize; // FP16 scale + 32 quants
        long blockOffset = (elementIndex / blockSize) * blockBytes;
        int withinBlock = (int) (elementIndex % blockSize);
        float scale = Float.float16ToFloat(data.get(ValueLayout.JAVA_SHORT_UNALIGNED, blockOffset));
        byte quant = data.get(ValueLayout.JAVA_BYTE, blockOffset + Short.BYTES + withinBlock);
        return scale * quant;
    }
}
