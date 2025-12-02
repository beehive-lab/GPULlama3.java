package org.beehive.gpullama3.tensor.tornado;

import org.beehive.gpullama3.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.tensor.standard.FloatTensor;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.*;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.concurrent.*;
import java.util.stream.IntStream;

public class Q8_0TornadoTensor extends TornadoTensor {

    private final int size;
    private final HalfFloatArray scales;  // One per 32-element block
    private final Int8Array quants;       // Quantized int8 values
    private MemorySegment segment;

    public Q8_0TornadoTensor(int size, HalfFloatArray scales, Int8Array quants, MemorySegment segment) {
        this.size = size;
        this.scales = scales;
        this.quants = quants;
        this.segment = segment;
    }

    public int getSize() {
        return size;
    }

    /**
     * Returns the scale factors for GPU kernels.
     *
     * @return HalfFloatArray containing fp16 scale factors
     */
    public HalfFloatArray getScales() {
        return scales;
    }

    /**
     * Returns the quantized values for GPU kernels.
     *
     * @return Int8Array containing quantized int8 values
     */
    public Int8Array getQuants() {
        return quants;
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q8_0;
    }

    public MemorySegment asMemorySegment() {
        return segment;
    }

    /**
     * Dequantizes and returns a single float value.
     *
     * @param index Element index
     * @return Dequantized float value
     */
    public float getFloat(int index) {
        assert 0 <= index;
        int blockIdx = index / GGMLType.Q8_0.getBlockSize();
        float scale = scales.get(blockIdx).getFloat32();
        byte quant = quants.get(index);
        return quant * scale;
    }

    /**
     * Creates a Q8_0TornadoTensor from a GGMLTensorEntry (original implementation).
     */
    public static Q8_0TornadoTensor createAsQ8_0(GGMLTensorEntry entry) {
        if (entry.ggmlType() != GGMLType.Q8_0) {
            throw new IllegalArgumentException("Expected Q8_0 tensor, got: " + entry.ggmlType() + " for tensor: " + entry.name());
        }

        int[] shape = entry.shape();
        int size = FloatTensor.numberOfElements(shape);
        int numBlocks = size / GGMLType.Q8_0.getBlockSize();

        if (size % GGMLType.Q8_0.getBlockSize() != 0) {
            throw new IllegalArgumentException("Q8_0 tensor size must be multiple of " + GGMLType.Q8_0.getBlockSize() + ", got: " + size + " for tensor: " + entry.name());
        }

        // TODO: fix Q8_0 loading in tornado layoyt
        //  currently we end up to hack it by removing
        //  tornado header from memory segment
        MemorySegment q8Segment = entry.memorySegment().asSlice(TornadoNativeArray.ARRAY_HEADER);

        // allocate the arrays for quantized data (int8) and scales (fp16)
        HalfFloatArray scales = new HalfFloatArray(numBlocks);
        Int8Array quants = new Int8Array(size);

        // unpack Q8_0 blocks: [2 bytes fp16 scale][32 bytes int8 quants]
        ValueLayout.OfShort shortLayout = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
        ValueLayout.OfByte byteLayout = ValueLayout.JAVA_BYTE;

        // element-wise copy and unpack from MemorySegment to HalfFloatArray scales and Int8Array quants
        // use parallel streams and unroll inner loop for better performance
        IntStream.range(0, numBlocks)
                .parallel()
                .forEach(block -> {
                    // TODO: use GGML type method for the 34L size
                    long blockOffset = block * 34L;  // 34 bytes per block

                    // read fp16 scale (first 2 bytes of block)
                    short scaleRaw = q8Segment.get(shortLayout, blockOffset);
                    scales.set(block, new HalfFloat(scaleRaw));
                    int blockStart = block * 32;

                    // read 32 int8 quantized values (remaining bytes of block)
                    // TODO: use GGML type method for the 32 size
                    for (int i = 0; i < 32; i += 4) {
                        // unroll inner loop for better performance
                        byte q0 = q8Segment.get(byteLayout, blockOffset + 2 + i);
                        byte q1 = q8Segment.get(byteLayout, blockOffset + 2 + i + 1);
                        byte q2 = q8Segment.get(byteLayout, blockOffset + 2 + i + 2);
                        byte q3 = q8Segment.get(byteLayout, blockOffset + 2 + i + 3);

                        quants.set(blockStart + i,     q0);
                        quants.set(blockStart + i + 1, q1);
                        quants.set(blockStart + i + 2, q2);
                        quants.set(blockStart + i + 3, q3);
                    }
                });

        return new Q8_0TornadoTensor(size, scales, quants, q8Segment);
    }

    /**
     * Creates a Q8_0TornadoTensor formulated as FP32TornadoTensor object from a GGMLTensorEntry.
     * NOTE: Hack implementation to comply with FP32 inference.
     */
    public static FP32TornadoTensor createAsFP32(GGMLTensorEntry entry) {
        if (entry.ggmlType() != GGMLType.Q8_0) {
            throw new IllegalArgumentException("Expected Q8_0 tensor, got: " + entry.ggmlType() + " for tensor: " + entry.name());
        }

        int[] shape = entry.shape();
        int size = FloatTensor.numberOfElements(shape);
        int numBlocks = size / GGMLType.Q8_0.getBlockSize();

        if (size % GGMLType.Q8_0.getBlockSize() != 0) {
            throw new IllegalArgumentException("Q8_0 tensor size must be multiple of " + GGMLType.Q8_0.getBlockSize() + ", got: " + size + " for tensor: " + entry.name());
        }

        // TODO: fix Q8_0 loading in tornado layoyt
        //  currently we end up to hack it by removing
        //  tornado header from memory segment
        MemorySegment q8Segment = entry.memorySegment().asSlice(TornadoNativeArray.ARRAY_HEADER);

        // allocate the FloatArray to store the result
        FloatArray floatArray = new FloatArray(size);

        // unpack Q8_0 blocks: [2 bytes fp16 scale][32 bytes int8 quants]
        ValueLayout.OfShort shortLayout = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
        ValueLayout.OfByte byteLayout = ValueLayout.JAVA_BYTE;

        // element-wise dequantization and copy from MemorySegment to FloatArray
        // use parallel streams and unroll inner loop for better performance
        IntStream.range(0, numBlocks)
                .parallel()
                .forEach(block -> {
                    // TODO: use GGML type method for the 34L size
                    long blockOffset = block * 34L;  // 34 bytes per block

                    // read fp16 scale (first 2 bytes of block) and convert to float
                    short scaleRaw = q8Segment.get(shortLayout, blockOffset);
                    float scale = Float.float16ToFloat(scaleRaw);
                    int blockStart = block * 32;

                    // read 32 int8 quantized values (remaining bytes of block)
                    // TODO: use GGML type method for the 32 size
                    for (int i = 0; i < 32; i += 4) {
                        // unroll inner loop for better performance
                        byte q0 = q8Segment.get(byteLayout, blockOffset + 2 + i);
                        byte q1 = q8Segment.get(byteLayout, blockOffset + 2 + i + 1);
                        byte q2 = q8Segment.get(byteLayout, blockOffset + 2 + i + 2);
                        byte q3 = q8Segment.get(byteLayout, blockOffset + 2 + i + 3);

                        floatArray.set(blockStart + i,     q0 * scale);
                        floatArray.set(blockStart + i + 1, q1 * scale);
                        floatArray.set(blockStart + i + 2, q2 * scale);
                        floatArray.set(blockStart + i + 3, q3 * scale);
                    }
                });

        return new FP32TornadoTensor(floatArray);
    }
}
