package org.beehive.gpullama3.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Device kernels that read {@code Q4_0} weights <b>in the file's own representation</b>.
 *
 * <p>The Q4_K family's siblings, and shaped identically to them on purpose — same signatures, same
 * workgroup-per-row structure, same reductions — so a layer differs only in which method reference
 * it names. What differs is the decode, and Q4_0's is the simplest of the quantizations that reach
 * the device.
 *
 * <h2>The block</h2>
 *
 * <p>32 weights in 18 bytes: {@code d} (fp16) at 0, then sixteen bytes of packed nibbles. A weight
 * is {@code d * (q - 8)} — an <b>unsigned</b> nibble recentred by eight, with a single scale and no
 * minimum, which is what separates it from {@code Q4_1} and from the K-quants' per-sub-block
 * scales. Element {@code i} below 16 is the low nibble of byte {@code i}; element {@code i} at 16
 * or above is the high nibble of byte {@code i - 16}.
 *
 * <h2>Why it exists</h2>
 *
 * <p>Q4_0 used to be materialized as Q8_0 at load, which roughly doubles a model's device
 * footprint: 4.5 bits per weight become 8.5. That is the same problem retaining Q4_K solved for
 * Devstral, and it is what puts a Q4_0 file's own size — rather than twice it — against the
 * device's memory.
 *
 * <p>The unpacking below is the same arithmetic as the host's {@code Q4_0FloatTensor}, which is the
 * reference it was written against. {@code Q4_0DecodeTest} holds the two against each other on the
 * same bytes rather than trusting the restatement.
 */
public final class TransformerComputeKernelsQ4_0 {

    /** Weights per block. */
    private static final int QK = 32;

    /** Bytes per block: 2 (d) + 16 (packed nibbles). */
    private static final int BLOCK_BYTES = 18;

    /** Byte offset of the packed nibbles within a block. */
    private static final int QS_OFFSET = 2;

    private TransformerComputeKernelsQ4_0() {}

    /**
     * One weight, decoded from its block.
     *
     * <p>Package-private, like its Q4_K counterpart, so the decode test can hold it against the
     * host tensor directly on the same bytes. TornadoVM inlines it.
     *
     * @param w the whole weight matrix, as the file stores it
     * @param blockByteOffset byte offset of this element's block
     * @param withinBlock the element's index inside the block, 0..31
     */
    static float decode(ByteArray w, int blockByteOffset, int withinBlock) {
        float d = w.getHalfFloat(blockByteOffset).getFloat32();
        int half = withinBlock / 16; // 0 for the low nibble, 1 for the high
        int byteIndex = withinBlock - half * 16;
        int packed = w.get(blockByteOffset + QS_OFFSET + byteIndex) & 0xFF;
        int q = (half == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);
        return d * (q - 8);
    }

    /**
     * One row's dot product against {@code x}, reduced across a 32-lane subgroup.
     *
     * <p>Used where {@code DeviceCapability.SUBGROUP_SHUFFLE_32} holds.
     */
    private static float rowDotSimd32(
            KernelContext context, FloatArray x, ByteArray w, int n, int rowId) {
        int localId = context.localIdx;
        int blocksPerRow = (n + QK - 1) / QK;
        int rowBlockOffset = rowId * blocksPerRow;

        float partialSum = 0.0f;
        for (int j = localId; j < n; j += 32) {
            int blockIdx = j / QK;
            int withinBlock = j - blockIdx * QK;
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

        int blocksPerRow = (n + QK - 1) / QK;
        int rowBlockOffset = rowId * blocksPerRow;

        float partialSum = 0.0f;
        for (int j = localId; j < n; j += localSize) {
            int blockIdx = j / QK;
            int withinBlock = j - blockIdx * QK;
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

    /** {@code output[row] = w[row]·x}. Q4_0 counterpart of {@code matrixVectorGenericQ8Byte}. */
    public static void matrixVectorGenericQ4_0(
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

    /** Subgroup-shuffle variant of {@link #matrixVectorGenericQ4_0}. */
    public static void matrixVectorGenericQ4_0Simd32(
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

    /**
     * {@code hb[row] += w[row]·x}. Q4_0 counterpart of {@code
     * matrixVectorGenericWithResidualQ8_0Byte}.
     */
    public static void matrixVectorGenericWithResidualQ4_0(
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

    /** Subgroup-shuffle variant of {@link #matrixVectorGenericWithResidualQ4_0}. */
    public static void matrixVectorGenericWithResidualQ4_0Simd32(
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

    /**
     * Fused query/key/value projection, one workgroup per output row.
     *
     * <p>Stated in terms of {@code qDim} and {@code kvDim} rather than assuming the query width is
     * {@code dim}, so it serves a family whose head dimension is independent of {@code dim /
     * heads} as well as one where they agree. The row's identity selects which projection it
     * belongs to.
     */
    public static void fusedQKVMatmulQ4_0(
            KernelContext context,
            FloatArray x,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            ByteArray wq,
            ByteArray wk,
            ByteArray wv,
            int dim,
            int qDim,
            int kvDim,
            int localWorkGroupSize) {
        int rowId = context.groupIdx;
        int totalRows = qDim + 2 * kvDim;
        if (rowId >= totalRows) {
            return;
        }

        if (rowId < qDim) {
            float sum = rowDotShared(context, localWorkGroupSize, x, wq, dim, rowId);
            if (context.localIdx == 0) {
                q.set(rowId, sum);
            }
        } else if (rowId < qDim + kvDim) {
            int row = rowId - qDim;
            float sum = rowDotShared(context, localWorkGroupSize, x, wk, dim, row);
            if (context.localIdx == 0) {
                k.set(row, sum);
            }
        } else {
            int row = rowId - qDim - kvDim;
            float sum = rowDotShared(context, localWorkGroupSize, x, wv, dim, row);
            if (context.localIdx == 0) {
                v.set(row, sum);
            }
        }
    }

    /**
     * Fused feed-forward gate/up projection with SwiGLU, over an already-normalized activation.
     *
     * <p>The normalization is dtype-independent and its existing task is reused rather than folded
     * in here, which is the same trade the Q4_K path makes: one more task per layer, one fewer
     * kernel to keep correct.
     */
    public static void fusedFFNGateUpSiLUQ4_0(
            KernelContext context,
            FloatArray x,
            FloatArray hb,
            ByteArray w1,
            ByteArray w3,
            int n,
            int d,
            int localWorkGroupSize) {
        int rowId = context.groupIdx;
        int localId = context.localIdx;
        if (rowId >= d) {
            return;
        }

        // Both projections in one pass over one local array: two calls to a helper that allocates
        // local memory and barriers would allocate twice and reduce twice, and the two reductions
        // would have to interleave their barriers correctly to be safe. Gate occupies the first
        // half, up the second, and one tree reduces both.
        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize * 2);
        int blocksPerRow = (n + QK - 1) / QK;
        int rowBlockOffset = rowId * blocksPerRow;

        float gate = 0.0f;
        float up = 0.0f;
        for (int j = localId; j < n; j += localWorkGroupSize) {
            int blockIdx = j / QK;
            int withinBlock = j - blockIdx * QK;
            int blockByteOffset = (rowBlockOffset + blockIdx) * BLOCK_BYTES;
            float activation = x.get(j);
            gate += decode(w1, blockByteOffset, withinBlock) * activation;
            up += decode(w3, blockByteOffset, withinBlock) * activation;
        }
        localSums[localId] = gate;
        localSums[localWorkGroupSize + localId] = up;
        context.localBarrier();

        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
                localSums[localWorkGroupSize + localId] +=
                        localSums[localWorkGroupSize + localId + stride];
            }
            context.localBarrier();
        }

        if (localId == 0) {
            float gateSum = localSums[0];
            float silu = gateSum / (1.0f + TornadoMath.exp(-gateSum));
            hb.set(rowId, silu * localSums[localWorkGroupSize]);
        }
    }
}
