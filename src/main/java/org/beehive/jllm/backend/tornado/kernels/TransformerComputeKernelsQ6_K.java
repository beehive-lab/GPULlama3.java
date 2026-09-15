package org.beehive.gpullama3.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Device kernels that read {@code Q6_K} weights in the file's own representation.
 *
 * <p>The companion to {@link TransformerComputeKernelsQ4_K}, and the reason it has one: a "Q4_K_M"
 * file is not uniformly Q4_K. It mixes per tensor and per layer — Devstral's holds {@code attn_v}
 * and {@code ffn_down} as Q6_K in half its layers and Q4_K in the other half — so retaining Q4_K
 * alone still left those tensors materialized as Q8_0 at roughly 1.9x their size. With both, the
 * whole file stays in its own representation.
 *
 * <h2>The super-block</h2>
 *
 * <p>256 weights in 210 bytes: {@code ql} (128 B, the low 4 bits) at 0, {@code qh} (64 B, the high
 * 2 bits) at 128, sixteen signed per-16 scales at 192, and {@code d} (fp16) at 208. A weight is
 * {@code d * scale(sub) * (q - 32)}, where {@code q} is 6 bits assembled from the two planes. The
 * block is read as two halves of 128, each of four groups of 32, and which nibble and which pair of
 * {@code qh} bits a weight uses depends on its group — the arithmetic below is the host's {@code
 * Q6_KFloatTensor}, which {@code Q4_KDecodeTest} holds it against element by element.
 */
public final class TransformerComputeKernelsQ6_K {

    /** Weights per super-block. */
    private static final int QK_K = 256;

    /** Bytes per super-block: 128 (ql) + 64 (qh) + 16 (scales) + 2 (d). */
    private static final int BLOCK_BYTES = 210;

    private static final int QH_OFFSET = 128;
    private static final int SCALES_OFFSET = 192;
    private static final int D_OFFSET = 208;

    private TransformerComputeKernelsQ6_K() {}

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
     * <p>Package-private so {@code Q4_KDecodeTest} can hold it against the host's {@code
     * Q6_KFloatTensor} directly, on the same bytes.
     */
    static float decode(ByteArray w, int blockByteOffset, int withinBlock) {
        float d = halfFromBytes(w, blockByteOffset + D_OFFSET);

        int halfIndex = withinBlock / 128; // 0 or 1
        int posInHalf = withinBlock - halfIndex * 128; // 0..127
        int groupInHalf = posInHalf / 32; // 0..3
        int posInGroup = posInHalf - groupInHalf * 32; // 0..31

        int qlBase = blockByteOffset + halfIndex * 64;
        int qhBase = blockByteOffset + QH_OFFSET + halfIndex * 32;
        int scBase = blockByteOffset + SCALES_OFFSET + halfIndex * 8;

        int is = posInGroup / 16; // 0 or 1
        int qh = w.get(qhBase + posInGroup) & 0xFF;

        int secondPlane = groupInHalf & 1;
        int highNibble = groupInHalf >> 1;
        int ql = w.get(qlBase + secondPlane * 32 + posInGroup) & 0xFF;
        int lowBits = ql & 0xF;
        if (highNibble == 1) {
            lowBits = (ql >> 4) & 0xF;
        }
        int highPair = qh & 3;
        if (groupInHalf == 1) {
            highPair = (qh >> 2) & 3;
        } else if (groupInHalf == 2) {
            highPair = (qh >> 4) & 3;
        } else if (groupInHalf == 3) {
            highPair = (qh >> 6) & 3;
        }
        int qValue = (lowBits | (highPair << 4)) - 32;

        int scale = w.get(scBase + is + groupInHalf * 2) & 0xFF;
        if (scale > 127) {
            scale = scale - 256;
        }
        return d * scale * qValue;
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
    public static void matrixVectorGenericQ6_K(
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

    /** {@code hb[row] += w[row]·x}. */
    public static void matrixVectorGenericWithResidualQ6_K(
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

    /**
     * {@code hb[i] = silu(hb[i]) * hb2[i]} — the SwiGLU half of the FFN, as its own task.
     *
     * <p>Separate from the projections because the gate and up weights of a mixed K-quant file can
     * be different representations, so they are computed by two independently-selected matvecs
     * rather than one fused kernel. Elementwise and dtype-independent; it lives here because this
     * is the path that needs it.
     */
    public static void siluAndMultiply(
            KernelContext context, FloatArray hb, FloatArray hb2, int d) {
        int i = context.globalIdx;
        if (i >= d) {
            return;
        }
        float gate = hb.get(i);
        hb.set(i, (gate / (1.0f + TornadoMath.exp(-gate))) * hb2.get(i));
    }
}
