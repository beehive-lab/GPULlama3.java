package org.beehive.gpullama3.tensor.tornado;

import org.beehive.gpullama3.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.tensor.standard.FloatTensor;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

public class Q4_0TornadoTensor extends TornadoTensor {

    private final HalfFloatArray scales;  // One per 32-element block
    private final ByteArray quants;       // Packed 4-bit quantized values (2 per byte)
    private MemorySegment segment;

    public Q4_0TornadoTensor(int size, HalfFloatArray scales, ByteArray quants, MemorySegment segment) {
        super(size);
        this.scales = scales;
        this.quants = quants;
        this.segment = segment;
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
     * @return ByteArray containing packed 4-bit quantized values
     */
    public ByteArray getQuants() {
        return quants;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q4_0;
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
        assert 0 <= index && index < size;
        int blockIdx = index / GGMLType.Q4_0.getBlockSize();
        int withinBlockIdx = index % GGMLType.Q4_0.getBlockSize();

        float scale = scales.get(blockIdx).getFloat32();

        // Each byte contains 2 4-bit values
        int byteIdx = withinBlockIdx / 2;
        byte packedByte = quants.get(blockIdx * 16 + byteIdx);

        // Extract the 4-bit value (lower or upper nibble)
        byte quant;
        if (withinBlockIdx % 2 == 0) {
            // Lower 4 bits
            quant = (byte) (packedByte & 0x0F);
        } else {
            // Upper 4 bits
            quant = (byte) ((packedByte >>> 4) & 0x0F);
        }

        // Offset by -8 (same as Q8_0)
        quant -= 8;

        return quant * scale;
    }

    public static Q4_0TornadoTensor create(GGMLTensorEntry entry) {
        if (entry.ggmlType() != GGMLType.Q4_0) {
            throw new IllegalArgumentException("Expected Q4_0 tensor, got: " + entry.ggmlType() + " for tensor: " + entry.name());
        }

        int[] shape = entry.shape();
        int size = FloatTensor.numberOfElements(shape);
        int numBlocks = size / GGMLType.Q4_0.getBlockSize();

        if (size % GGMLType.Q4_0.getBlockSize() != 0) {
            throw new IllegalArgumentException("Q4_0 tensor size must be multiple of " + GGMLType.Q4_0.getBlockSize() + ", got: " + size + " for tensor: " + entry.name());
        }

        MemorySegment q4Segment = entry.memorySegment();

        // allocate the arrays for quantized data (packed 4-bit) and scales (fp16)
        HalfFloatArray scales = new HalfFloatArray(numBlocks);
        ByteArray quants = new ByteArray(numBlocks * 16);  // 32 4-bit values = 16 bytes per block

        // unpack Q4_0 blocks: [2 bytes fp16 scale][16 bytes packed 4-bit quants]
        ValueLayout.OfShort shortLayout = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
        ValueLayout.OfByte byteLayout = ValueLayout.JAVA_BYTE;

        for (int block = 0; block < numBlocks; block++) {
            long blockOffset = block * GGMLType.Q4_0.getTypeSize();  // 18 bytes per block

            // read fp16 scale (first 2 bytes of block)
            short scaleRaw = q4Segment.get(shortLayout, blockOffset);
            scales.set(block, new HalfFloat(scaleRaw));

            // read 16 bytes of packed 4-bit quantized values (remaining bytes of block)
            for (int i = 0; i < 16; i++) {
                byte quantValue = q4Segment.get(byteLayout, blockOffset + 2 + i);
                quants.set(block * 16 + i, quantValue);
            }
        }

        return new Q4_0TornadoTensor(size, scales, quants, q4Segment);
    }
}
