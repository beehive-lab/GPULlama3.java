package org.beehive.gpullama3.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Device kernels that read {@code Q4_1} weights in the file's own representation.
 *
 * <p>Shaped like the {@code Q4_0}, {@code Q4_K} and {@code Q6_K} kernels beside them — same
 * signatures, same workgroup-per-row structure, same reductions — so a layer differs only in which
 * method reference it names. What differs is the decode.
 *
 * <h2>The block</h2>
 *
 * <p>32 weights in 20 bytes: {@code d} (fp16) at 0, {@code m} (fp16) at 2, then sixteen bytes of
 * packed nibbles. A weight is {@code d * q + m} with an <b>unsigned</b> nibble.
 *
 * <p>That is the whole difference from Q4_0, which is {@code d * (q - 8)} — one recentring and one
 * extra half of header. Applying Q4_0's arithmetic to a Q4_1 block, or Q4_1's offsets to a Q4_0
 * one, gives weights of entirely plausible magnitude, so neither mistake announces itself. {@code
 * Q4_1DecodeTest} holds this against {@code Q4_1FloatTensor} on the same bytes.
 */
public final class TransformerComputeKernelsQ4_1 {

    /** Weights per block. */
    private static final int QK = 32;

    /** Bytes per block: 2 (d) + 2 (m) + 16 (packed nibbles). */
    private static final int BLOCK_BYTES = 20;

    /** Byte offset of the packed nibbles within a block. */
    private static final int QS_OFFSET = 4;

    /** Prompt rows a tiled batch workgroup covers, as in {@code TransformerComputeKernelsQ4_0}. */
    private static final int ROW_TILE = 8;

    private TransformerComputeKernelsQ4_1() {}

    /**
     * One weight, decoded from its block.
     *
     * @param w the whole weight matrix, as the file stores it
     * @param blockByteOffset byte offset of this element's block
     * @param withinBlock the element's index inside the block, 0..31
     */
    static float decode(ByteArray w, int blockByteOffset, int withinBlock) {
        float d = w.getHalfFloat(blockByteOffset).getFloat32();
        float m = w.getHalfFloat(blockByteOffset + 2).getFloat32();
        int half = withinBlock / 16; // 0 for the low nibble, 1 for the high
        int byteIndex = withinBlock - half * 16;
        int packed = w.get(blockByteOffset + QS_OFFSET + byteIndex) & 0xFF;
        int q = (half == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);
        return d * q + m;
    }

    /** One row's dot product against {@code x}, reduced across a 32-lane subgroup. */
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
        return rowDotShared(context, localSize, x, 0, w, n, rowId);
    }

    /**
     * The same reduction over a row of a <b>batch</b> of activations.
     *
     * <p>{@code xOffset} is where this row's activation starts. Everything else — the block
     * addressing, the decode, the reduction — is the single-token path's, so a batched projection
     * is the same arithmetic in the same order over a different input offset.
     */
    private static float rowDotShared(
            KernelContext context,
            int localSize,
            FloatArray x,
            int xOffset,
            ByteArray w,
            int n,
            int rowId) {
        int localId = context.localIdx;
        float[] localSums = context.allocateFloatLocalArray(localSize);

        int blocksPerRow = (n + QK - 1) / QK;
        int rowBlockOffset = rowId * blocksPerRow;

        float partialSum = 0.0f;
        for (int j = localId; j < n; j += localSize) {
            int blockIdx = j / QK;
            int withinBlock = j - blockIdx * QK;
            int blockByteOffset = (rowBlockOffset + blockIdx) * BLOCK_BYTES;
            partialSum += decode(w, blockByteOffset, withinBlock) * x.get(xOffset + j);
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
    public static void matrixVectorGenericQ4_1(
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

    /** Subgroup-shuffle variant of {@link #matrixVectorGenericQ4_1}. */
    public static void matrixVectorGenericQ4_1Simd32(
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
    public static void matrixVectorGenericWithResidualQ4_1(
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

    /** Subgroup-shuffle variant of {@link #matrixVectorGenericWithResidualQ4_1}. */
    public static void matrixVectorGenericWithResidualQ4_1Simd32(
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

    /** Prompt rows a tiled batch workgroup covers. */
    public static int rowTile() {
        return ROW_TILE;
    }

    // @formatter:off
    /**
     * {@code out[b][row] = w[row]·x[b]} for a tile of up to {@link #ROW_TILE} prompt rows, one
     * workgroup per (row tile, output row).
     *
     * <p>Q4_0's tiled kernel with Q4_1's decode. The untiled batch kernel beside this one reads the
     * weight row once per prompt row, which is what running the rows separately reads; this one
     * reads and decodes each weight once for the whole tile.
     */
    // @formatter:on
    public static void matrixVectorTiledBatchQ4_1(
            KernelContext context,
            FloatArray xBatch,
            FloatArray outBatch,
            ByteArray w,
            int n,
            int d,
            int activeRows,
            int localWorkGroupSize) {
        int groupId = context.groupIdx;
        int tile = groupId / d;
        int rowId = groupId - tile * d;
        int firstRow = tile * ROW_TILE;
        if (firstRow >= activeRows) {
            return;
        }
        int localId = context.localIdx;

        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize * ROW_TILE);
        tileDot(context, xBatch, w, n, activeRows, localWorkGroupSize, rowId, firstRow, localSums);

        if (localId == 0) {
            for (int t = 0; t < ROW_TILE; t++) {
                int row = firstRow + t;
                if (row < activeRows) {
                    outBatch.set(row * d + rowId, localSums[t * localWorkGroupSize]);
                }
            }
        }
    }

    /**
     * {@code out[b][row] += w[row]·x[b]} for a tile of rows. See {@link
     * #matrixVectorTiledBatchQ4_1}.
     */
    public static void matrixVectorTiledBatchWithResidualQ4_1(
            KernelContext context,
            FloatArray xBatch,
            FloatArray outBatch,
            ByteArray w,
            int n,
            int d,
            int activeRows,
            int localWorkGroupSize) {
        int groupId = context.groupIdx;
        int tile = groupId / d;
        int rowId = groupId - tile * d;
        int firstRow = tile * ROW_TILE;
        if (firstRow >= activeRows) {
            return;
        }
        int localId = context.localIdx;

        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize * ROW_TILE);
        tileDot(context, xBatch, w, n, activeRows, localWorkGroupSize, rowId, firstRow, localSums);

        if (localId == 0) {
            for (int t = 0; t < ROW_TILE; t++) {
                int row = firstRow + t;
                if (row < activeRows) {
                    int index = row * d + rowId;
                    outBatch.set(index, outBatch.get(index) + localSums[t * localWorkGroupSize]);
                }
            }
        }
    }

    /** The tile's eight dot products, reduced into {@code localSums[t * localSize]}. */
    private static void tileDot(
            KernelContext context,
            FloatArray xBatch,
            ByteArray w,
            int n,
            int activeRows,
            int localSize,
            int rowId,
            int firstRow,
            float[] localSums) {
        int localId = context.localIdx;
        int blocksPerRow = (n + QK - 1) / QK;
        int rowBlockOffset = rowId * blocksPerRow;

        float a0 = 0.0f;
        float a1 = 0.0f;
        float a2 = 0.0f;
        float a3 = 0.0f;
        float a4 = 0.0f;
        float a5 = 0.0f;
        float a6 = 0.0f;
        float a7 = 0.0f;

        for (int j = localId; j < n; j += localSize) {
            int blockIdx = j / QK;
            int withinBlock = j - blockIdx * QK;
            int blockByteOffset = (rowBlockOffset + blockIdx) * BLOCK_BYTES;
            float weight = decode(w, blockByteOffset, withinBlock);
            a0 += weight * xBatch.get(firstRow * n + j);
            if (firstRow + 1 < activeRows) {
                a1 += weight * xBatch.get((firstRow + 1) * n + j);
            }
            if (firstRow + 2 < activeRows) {
                a2 += weight * xBatch.get((firstRow + 2) * n + j);
            }
            if (firstRow + 3 < activeRows) {
                a3 += weight * xBatch.get((firstRow + 3) * n + j);
            }
            if (firstRow + 4 < activeRows) {
                a4 += weight * xBatch.get((firstRow + 4) * n + j);
            }
            if (firstRow + 5 < activeRows) {
                a5 += weight * xBatch.get((firstRow + 5) * n + j);
            }
            if (firstRow + 6 < activeRows) {
                a6 += weight * xBatch.get((firstRow + 6) * n + j);
            }
            if (firstRow + 7 < activeRows) {
                a7 += weight * xBatch.get((firstRow + 7) * n + j);
            }
        }

        localSums[localId] = a0;
        localSums[localSize + localId] = a1;
        localSums[2 * localSize + localId] = a2;
        localSums[3 * localSize + localId] = a3;
        localSums[4 * localSize + localId] = a4;
        localSums[5 * localSize + localId] = a5;
        localSums[6 * localSize + localId] = a6;
        localSums[7 * localSize + localId] = a7;
        context.localBarrier();

        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                for (int t = 0; t < ROW_TILE; t++) {
                    localSums[t * localSize + localId] +=
                            localSums[t * localSize + localId + stride];
                }
            }
            context.localBarrier();
        }
    }
}
