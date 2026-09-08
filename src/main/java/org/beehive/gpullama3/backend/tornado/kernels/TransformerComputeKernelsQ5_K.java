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
     * The fp16 at {@code index}, assembled from two plain byte loads.
     *
     * <p>Deliberately not {@code ByteArray.getHalfFloat}: that call is what TornadoVM 5.2.0's
     * sketcher chokes on in this decode ("Unable to build sketch for method: fillInStackTrace"),
     * while whole-byte loads compile. Reading the two bytes and widening the half here keeps the
     * kernel on constructs the sketcher handles.
     *
     * <p>Subnormals are handled explicitly; infinities and NaNs are not, because a quantized block
     * scale is neither.
     */
    private static float halfFromBytes(ByteArray w, int index) {
        int lo = w.get(index) & 0xFF;
        int hi = w.get(index + 1) & 0xFF;
        int h = (hi << 8) | lo;
        int mantissa = h & 0x3FF;
        int exponent = (h >>> 10) & 0x1F;
        float magnitude;
        if (exponent == 0) {
            magnitude = mantissa * 5.9604645E-8f; // 2^-24, the subnormal step
        } else {
            // 2^(exponent-15) without a bit-pattern reinterpret and without a data-dependent
            // loop. Float.intBitsToFloat reaches the Metal backend as a node its LIR builder does
            // not implement ("TornadoInternalError: unimplemented" in MetalNodeLIRBuilder.doBlock),
            // and a counted loop over the exponent compiled but took minutes and produced zero.
            // The exponent is 1.30, so |e| < 16 and four fixed tests cover every case.
            int e = exponent - 15;
            int magnitudeOfE = e;
            if (e < 0) {
                magnitudeOfE = -e;
            }
            float scale = 1.0f;
            if ((magnitudeOfE & 1) != 0) {
                scale *= 2.0f;
            }
            if ((magnitudeOfE & 2) != 0) {
                scale *= 4.0f;
            }
            if ((magnitudeOfE & 4) != 0) {
                scale *= 16.0f;
            }
            if ((magnitudeOfE & 8) != 0) {
                scale *= 256.0f;
            }
            if (e < 0) {
                scale = 1.0f / scale;
            }
            magnitude = (1.0f + mantissa * (1.0f / 1024.0f)) * scale;
        }
        if ((h & 0x8000) != 0) {
            return -magnitude;
        }
        return magnitude;
    }

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
        // Byte-assembled, like the kernel: the parity test calls this, and a decode that read its
        // scale differently from the one the device runs would be testing something else.
        float d = halfFromBytes(w, blockByteOffset);
        float dmin = halfFromBytes(w, blockByteOffset + 2);

        int pairIndex = withinBlock / 64; // 0..3
        int posInPair = withinBlock - pairIndex * 64; // 0..63
        int highNibble = posInPair / 32; // 0 for the first 32, 1 for the next
        int subBlock = pairIndex * 2 + highNibble;
        int posInHalf = posInPair - highNibble * 32; // 0..31

        int qsByte = w.get(blockByteOffset + QS_OFFSET + pairIndex * 32 + posInHalf) & 0xFF;
        int q = (highNibble == 0) ? (qsByte & 0xF) : ((qsByte >> 4) & 0xF);

        // The fifth bit is indexed by position within the pair's half, and by which nibble the
        // element came from — not by the element's index in the super-block. Its bit position is
        // `pairIndex * 2 + highNibble`.
        //
        // Written as three index-derived branches with constant shift amounts, which is neither
        // the obvious formulation nor an arbitrary one. Two shorter versions do not survive
        // TornadoVM's CUDA backend:
        //
        //   (qh >>> shift) & 1            a variable shift amount, which makes it emit
        //                                 deoptimization scaffolding it cannot declare —
        //                                 "identifier 'context' is undefined", "identifier
        //                                 'slots' is undefined"
        //   (qh & mask) == 0 ? 0 : 16     a conditional move, which asserts inside
        //                                 CUDALIRGenerator.emitIntegerTestMove
        //
        // Branching on a *loaded* value fails the same way as the variable shift. Branching on an
        // index does not — Q4_K's own `subBlock < 4` does exactly that and compiles. So the shift
        // is decomposed against the index: 4 for the high pair, 2 for an odd pair, 1 for the high
        // nibble, summing to the same amount, with the value's dataflow branch-free.
        int qhByte = w.get(blockByteOffset + QH_OFFSET + posInHalf) & 0xFF;
        int bits = qhByte;
        if (pairIndex >= 2) {
            bits = bits >> 4;
        }
        if ((pairIndex & 1) == 1) {
            bits = bits >> 2;
        }
        if (highNibble == 1) {
            bits = bits >> 1;
        }
        q += (bits & 1) * 16;

        int packed = scaleAndMin(w, blockByteOffset + SCALES_OFFSET, subBlock);
        return d * (packed >> 8) * q - dmin * (packed & 0xFF);
    }

    /**
     * A sub-block's 6-bit scale and minimum, packed as {@code (scale << 8) | min}.
     *
     * <p>Its own method rather than eight lines inside {@link #decode}, and that is a code
     * generation constraint rather than a style choice. Inlined, the whole decode was large enough
     * that TornadoVM's CUDA backend emitted a kernel referring to an undeclared {@code context} and
     * an undeclared local array — {@code identifier "context" is undefined}, {@code identifier
     * "slots" is undefined} — for the shared-memory reduction that calls it. The decode compiles
     * perfectly well on its own; it is the combination with the reduction that broke, and keeping
     * this branch in a separate method is what fixes it.
     *
     * <p>Both values are six bits, so they pack into one int with room to spare and no information
     * is lost. The {@code subBlock >= 4} case is the one where a 6-bit value straddles two bytes.
     */
    private static int scaleAndMin(ByteArray w, int scalesBase, int subBlock) {
        if (subBlock < 4) {
            return ((w.get(scalesBase + subBlock) & 63) << 8) | (w.get(scalesBase + subBlock + 4) & 63);
        }
        int lowScale = w.get(scalesBase + subBlock + 4) & 0xFF;
        int highScale = w.get(scalesBase + subBlock - 4) & 0xFF;
        int sc = (lowScale & 0xF) | ((highScale >> 6) << 4);
        int m = ((lowScale >> 4) & 0xF) | (((w.get(scalesBase + subBlock) & 0xFF) >> 6) << 4);
        return (sc << 8) | m;
    }

    /** One row's dot product against {@code x}, reduced across a 32-lane subgroup. */
    private static float rowDotSimd32(
            KernelContext context, FloatArray x, ByteArray w, int n, int rowId) {
        int localId = context.localIdx;
        int blocksPerRow = (n + QK_K - 1) / QK_K;
        int rowBlockOffset = rowId * blocksPerRow;

        int subBlocks = n / 32;
        float partialSum = 0.0f;
        for (int sb = localId; sb < subBlocks; sb += 32) {
            partialSum += laneSum(x, w, n, rowBlockOffset, sb);
        }

        partialSum += context.simdShuffleDown(partialSum, 16);
        partialSum += context.simdShuffleDown(partialSum, 8);
        partialSum += context.simdShuffleDown(partialSum, 4);
        partialSum += context.simdShuffleDown(partialSum, 2);
        partialSum += context.simdShuffleDown(partialSum, 1);
        return partialSum;
    }

    /**
     * One lane's contribution to a row, walking whole 32-element sub-blocks.
     *
     * <p>Not the element-strided loop its Q4_K sibling uses, and the reason is a code generation
     * one as much as an efficiency one. Q5_K's per-element decode is large enough that TornadoVM's
     * CUDA backend emitted deoptimization scaffolding it cannot declare when the decode was inlined
     * into a loop — {@code identifier "context" is undefined}, {@code identifier "slots" is
     * undefined} — with the same decode compiling perfectly well outside a loop, and with both the
     * shared-memory and the shuffle reduction affected, so it was never about local memory.
     *
     * <p>Walking sub-blocks makes the per-element body small: a sub-block's scale, minimum, nibble
     * plane and fifth-bit position are all constant across its 32 elements, so they are computed
     * once instead of thirty-two times. That is the shape a Q5_K kernel should have had anyway.
     */
    private static float laneSum(FloatArray x, ByteArray w, int n, int rowBlockOffset, int subBlockIndex) {
        return laneSum(x, 0, w, n, rowBlockOffset, subBlockIndex);
    }

    /** The same sub-block, over a batch of activations starting at {@code xOffset}. */
    private static float laneSum(
            FloatArray x,
            int xOffset,
            ByteArray w,
            int n,
            int rowBlockOffset,
            int subBlockIndex) {
        int block = subBlockIndex / 8;
        int subInBlock = subBlockIndex - block * 8;
        int blockByteOffset = (rowBlockOffset + block) * BLOCK_BYTES;

        float d = halfFromBytes(w, blockByteOffset);
        float dmin = halfFromBytes(w, blockByteOffset + 2);
        int packed = scaleAndMin(w, blockByteOffset + SCALES_OFFSET, subInBlock);
        float scale = d * (packed >> 8);
        float minimum = dmin * (packed & 0xFF);

        int pairIndex = subInBlock >> 1;
        int highNibble = subInBlock & 1;
        int qsBase = blockByteOffset + QS_OFFSET + pairIndex * 32;
        int qhBase = blockByteOffset + QH_OFFSET;
        // Loop-invariant: the fifth bit's position depends only on the sub-block.
        int bitShift = pairIndex * 2 + highNibble;
        int elementBase = subBlockIndex * 32;

        float sum = 0.0f;
        for (int t = 0; t < 32; t++) {
            int qsByte = w.get(qsBase + t) & 0xFF;
            int low = qsByte & 0xF;
            if (highNibble == 1) {
                low = (qsByte >> 4) & 0xF;
            }
            int qhByte = w.get(qhBase + t) & 0xFF;
            int high = (qhByte >> bitShift) & 1;
            sum += (scale * (low + high * 16) - minimum) * x.get(xOffset + elementBase + t);
        }
        return sum;
    }

    /** One row's dot product against {@code x}, reduced through shared memory. */
    private static float rowDotShared(
            KernelContext context, int localSize, FloatArray x, ByteArray w, int n, int rowId) {
        return rowDotShared(context, localSize, x, 0, w, n, rowId);
    }

    /** The same reduction over one row of a batch of activations. */
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

        int blocksPerRow = (n + QK_K - 1) / QK_K;
        int rowBlockOffset = rowId * blocksPerRow;
        int subBlocks = n / 32;

        float partialSum = 0.0f;
        for (int sb = localId; sb < subBlocks; sb += localSize) {
            partialSum += laneSum(x, xOffset, w, n, rowBlockOffset, sb);
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

    // @formatter:off
    /**
     * {@code out[b][row] = w[row]·x[b]} over a chunk of activations, one workgroup per (row,
     * output row). Padding rows return before reading anything.
     */
    // @formatter:on
    public static void matrixVectorBatchQ5_K(
            KernelContext context,
            FloatArray xBatch,
            FloatArray outBatch,
            ByteArray w,
            int n,
            int d,
            int activeRows,
            int localWorkGroupSize) {
        int groupId = context.groupIdx;
        int batchIdx = groupId / d;
        int rowId = groupId - batchIdx * d;
        if (batchIdx >= activeRows) {
            return;
        }
        float sum = rowDotShared(context, localWorkGroupSize, xBatch, batchIdx * n, w, n, rowId);
        if (context.localIdx == 0) {
            outBatch.set(batchIdx * d + rowId, sum);
        }
    }

    /** {@code out[b][row] += w[row]·x[b]}, the residual form. */
    public static void matrixVectorBatchWithResidualQ5_K(
            KernelContext context,
            FloatArray xBatch,
            FloatArray outBatch,
            ByteArray w,
            int n,
            int d,
            int activeRows,
            int localWorkGroupSize) {
        int groupId = context.groupIdx;
        int batchIdx = groupId / d;
        int rowId = groupId - batchIdx * d;
        if (batchIdx >= activeRows) {
            return;
        }
        float sum = rowDotShared(context, localWorkGroupSize, xBatch, batchIdx * n, w, n, rowId);
        if (context.localIdx == 0) {
            int index = batchIdx * d + rowId;
            outBatch.set(index, outBatch.get(index) + sum);
        }
    }
}
