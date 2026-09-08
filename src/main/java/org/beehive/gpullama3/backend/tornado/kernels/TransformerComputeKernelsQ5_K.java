package org.beehive.gpullama3.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Device kernels that read {@code Q5_K} weights in the file's own representation.
 *
 * <p>Shaped like the {@code Q4_K} kernels beside them, and sharing their scale/minimum unpacking
 * exactly — Q5_K's twelve scale bytes are packed identically. What Q5_K adds is a fifth bit per
 * weight, held in a separate 32-byte plane rather than beside its nibble.
 *
 * <h2>The super-block</h2>
 *
 * <pre>
 *   offset   0   d       (fp16)    super-block scale for the quantized scales
 *   offset   2   dmin    (fp16)    super-block scale for the quantized minima
 *   offset   4   scales  (12 B)    eight 6-bit scale/min pairs
 *   offset  16   qh      (32 B)    the fifth bit of each of the 256 weights
 *   offset  48   qs     (128 B)    the low four bits
 * </pre>
 *
 * <p>A weight is {@code d * scale(sub) * q - dmin * min(sub)}, {@code q} five bits wide. The 256
 * weights are four pairs of 64; within a pair the first 32 take the low nibble of a byte and the
 * next 32 the high nibble of the same byte, and the fifth bit comes from bit {@code 2*pair} or
 * {@code 2*pair + 1} of the {@code qh} byte for that position.
 *
 * <p><b>Two things here are wrong-but-plausible if mis-addressed.</b> Taking the fifth bit from the
 * wrong bit position shifts a weight by sixteen quantization steps, and the {@code subBlock >= 4}
 * branch of the scale unpacking straddles two bytes. Neither shows up as anything but slightly
 * worse output. {@code Q5_KDecodeTest} holds this against {@code Q5_KFloatTensor} on adversarial
 * blocks rather than on values that happen to avoid those paths.
 */
public final class TransformerComputeKernelsQ5_K {

    /** Weights per super-block. */
    private static final int QK_K = 256;

    /** Bytes per super-block. */
    private static final int BLOCK_BYTES = 176;

    /** Byte offset of the packed 6-bit scale/min pairs. */
    private static final int SCALES_OFFSET = 4;

    /** Byte offset of the fifth-bit plane. */
    private static final int QH_OFFSET = 16;

    /** Byte offset of the low four bits. */
    private static final int QS_OFFSET = 48;

    private TransformerComputeKernelsQ5_K() {}

    /**
     * One weight, decoded from its super-block.
     *
     * <p>Package-private so the decode test can hold it against the host tensor directly on the
     * same bytes. TornadoVM inlines it.
     *
     * @param w the whole weight matrix, as the file stores it
     * @param blockByteOffset byte offset of this element's super-block
     * @param withinBlock the element's index inside the super-block, 0..255
     */
    static float decode(ByteArray w, int blockByteOffset, int withinBlock) {
        float d = w.getHalfFloat(blockByteOffset).getFloat32();
        float dmin = w.getHalfFloat(blockByteOffset + 2).getFloat32();

        int pairIndex = withinBlock / 64; // 0..3
        int posInPair = withinBlock - pairIndex * 64; // 0..63
        int highNibble = posInPair / 32; // 0 for the first 32, 1 for the next
        int subBlock = pairIndex * 2 + highNibble;
        int posInHalf = posInPair - highNibble * 32; // 0..31

        int qsByte = w.get(blockByteOffset + QS_OFFSET + pairIndex * 32 + posInHalf) & 0xFF;
        int q = (highNibble == 0) ? (qsByte & 0xF) : ((qsByte >> 4) & 0xF);

        // The fifth bit is indexed by position within the pair's half, and by which nibble the
        // element came from — not by the element's index in the super-block.
        int qhByte = w.get(blockByteOffset + QH_OFFSET + posInHalf) & 0xFF;
        int highBit = (qhByte >> (pairIndex * 2 + highNibble)) & 1;
        q += highBit * 16;

        int scalesBase = blockByteOffset + SCALES_OFFSET;
        int sc;
        int m;
        if (subBlock < 4) {
            sc = w.get(scalesBase + subBlock) & 63;
            m = w.get(scalesBase + subBlock + 4) & 63;
        } else {
            int lowScale = w.get(scalesBase + subBlock + 4) & 0xFF;
            int highScale = w.get(scalesBase + subBlock - 4) & 0xFF;
            sc = (lowScale & 0xF) | ((highScale >> 6) << 4);
            m = ((lowScale >> 4) & 0xF) | (((w.get(scalesBase + subBlock) & 0xFF) >> 6) << 4);
        }
        return d * sc * q - dmin * m;
    }

    /** One row's dot product against {@code x}, reduced across a 32-lane subgroup. */
    private static float rowDotSimd32(
            KernelContext context, FloatArray x, ByteArray w, int n, int rowId) {
        int localId = context.localIdx;
        int blocksPerRow = (n + QK_K - 1) / QK_K;
        int rowBlockOffset = rowId * blocksPerRow;

        float partialSum = 0.0f;
        for (int j = localId; j < n; j += 32) {
            int blockIdx = j / QK_K;
            int withinBlock = j - blockIdx * QK_K;
            int blockByteOffset = (rowBlockOffset + blockIdx) * BLOCK_BYTES;
            partialSum += decode(w, blockByteOffset, withinBlock) * x.get(j);
        }

        partialSum += context.simdShuffleDown(partialSum, 16);
        partialSum += context.simdShuffleDown(partialSum, 8);
        partialSum += context.simdShuffleDown(partialSum, 4);
        partialSum += context.simdShuffleDown(partialSum, 2);
        partialSum += context.simdShuffleDown(partialSum, 1);
        return partialSum;
    }

    /** One row's dot product against {@code x}, reduced through shared memory. */
    private static float rowDotShared(
            KernelContext context, int localSize, FloatArray x, ByteArray w, int n, int rowId) {
        int localId = context.localIdx;
        float[] localSums = context.allocateFloatLocalArray(localSize);

        int blocksPerRow = (n + QK_K - 1) / QK_K;
        int rowBlockOffset = rowId * blocksPerRow;

        float partialSum = 0.0f;
        for (int j = localId; j < n; j += localSize) {
            int blockIdx = j / QK_K;
            int withinBlock = j - blockIdx * QK_K;
            int blockByteOffset = (rowBlockOffset + blockIdx) * BLOCK_BYTES;
            partialSum += decode(w, blockByteOffset, withinBlock) * x.get(j);
        }

        localSums[localId] = partialSum;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }
        return localSums[0];
    }

    /** {@code output[row] = w[row]·x}. */
    public static void matrixVectorGenericQ5_K(
            KernelContext context,
            FloatArray x,
            FloatArray output,
            ByteArray w,
            int n,
            int d,
            int localWorkGroupSize) {
        int rowId = context.groupIdx;
        if (rowId >= d) {
            return;
        }
        float sum = rowDotShared(context, localWorkGroupSize, x, w, n, rowId);
        if (context.localIdx == 0) {
            output.set(rowId, sum);
        }
    }

    /** Subgroup-shuffle variant of {@link #matrixVectorGenericQ5_K}. */
    public static void matrixVectorGenericQ5_KSimd32(
            KernelContext context, FloatArray x, FloatArray output, ByteArray w, int n, int d) {
        int rowId = context.groupIdx;
        if (rowId >= d) {
            return;
        }
        float sum = rowDotSimd32(context, x, w, n, rowId);
        if (context.localIdx == 0) {
            output.set(rowId, sum);
        }
    }

    /** {@code hb[row] += w[row]·x}. */
    public static void matrixVectorGenericWithResidualQ5_K(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            ByteArray w,
            int n,
            int d,
            int localWorkGroupSize) {
        int rowId = context.groupIdx;
        if (rowId >= d) {
            return;
        }
        float sum = rowDotShared(context, localWorkGroupSize, x, w, n, rowId);
        if (context.localIdx == 0) {
            hb.set(rowId, hb.get(rowId) + sum);
        }
    }

    /** Subgroup-shuffle variant of {@link #matrixVectorGenericWithResidualQ5_K}. */
    public static void matrixVectorGenericWithResidualQ5_KSimd32(
            KernelContext context, FloatArray x, FloatArray hb, ByteArray w, int n, int d) {
        int rowId = context.groupIdx;
        if (rowId >= d) {
            return;
        }
        float sum = rowDotSimd32(context, x, w, n, rowId);
        if (context.localIdx == 0) {
            hb.set(rowId, hb.get(rowId) + sum);
        }
    }
}
