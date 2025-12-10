package org.beehive.gpullama3.tornadovm.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

public class TransformerComputeKernels {

    /**
     * Default constructor for the TransformerComputeKernels class.
     */
    public TransformerComputeKernels() {
    }

    public static void emptyTaskToForceCopyIn(FloatArray buffer) {
        float dummy = buffer.get(0);
        if (dummy > Float.MAX_VALUE) {
            buffer.set(0, dummy);
        }
    }

    public static void convertFP16toFP32(KernelContext context, HalfFloatArray x, FloatArray wrapX) {
        int i = context.globalIdx;
        wrapX.set(i, x.get(i).getFloat32());
    }

    public static void convertQ8_0toFP32(KernelContext context, ByteArray x, FloatArray wrapX) {
        int globalId = context.globalIdx;
        int totalElements = wrapX.getSize();

        if (globalId >= totalElements) {
            return;
        }

        // Q8_0 block structure constants
        int blockSize = 32;
        int Q8_0_BLOCK_BYTES = 34; // 2 bytes scale + 32 bytes quants

        // Calculate which block and position within block
        int blockIdx = globalId / blockSize;
        int withinBlockIdx = globalId % blockSize;

        // Calculate byte offset for this Q8_0 block
        int blockByteOffset = blockIdx * Q8_0_BLOCK_BYTES;

        // Load scale (first 2 bytes of block as HalfFloat)
        HalfFloat scale = x.getHalfFloat(blockByteOffset);
        float scaleFloat = scale.getFloat32();

        // Load quantized value (skip 2-byte scale, then index within block)
        byte quantValue = x.get(blockByteOffset + 2 + withinBlockIdx);

        // Dequantize: float_value = quantized_value * scale
        float dequantizedValue = ((float) quantValue) * scaleFloat;

        // Store result in output FloatArray
        wrapX.set(globalId, dequantizedValue);
    }

    public static void convertFP32toFP16(KernelContext context,  FloatArray wrapX, HalfFloatArray x) {
        int i = context.globalIdx;
        float valInput = wrapX.get(i);
        HalfFloat val = new HalfFloat(valInput);
        x.set(i,val);
    }

    /**
     * Performs RMS (Root Mean Square) normalization using parallel reduction.
     * This is a two-phase reduction: first within work groups, then across work groups.
     *
     * Phase 1: Each work group computes a partial sum of squares
     * Phase 2: First thread combines all partial sums and computes normalization factor
     *
     * @param context Kernel execution context
     * @param output Array to store partial sums and final normalization factor
     * @param x Input array to normalize
     * @param size Number of elements to process
     * @param ermsNorm Epsilon value for numerical stability (epsilon * epsilon)
     * @param localMemSize Size of local memory allocation (work group size)
     */
    public static void reductionOneBlockWithLayer(KernelContext context, FloatArray output, FloatArray x, int size, float ermsNorm, int localMemSize) {
        int gid = context.globalIdx;
        int lid = context.localIdx;
        int groupId = context.groupIdx;
        int groupSize = context.localGroupSizeX;

        // Allocate local memory with the provided size
        float[] localX = context.allocateFloatLocalArray(localMemSize);

        // Load input value and compute square
        if (gid < size) {
            localX[lid] = x.get(gid);
            localX[lid] = localX[lid] * localX[lid];
        } else {
            localX[lid] = 0.0f;
        }

        // Perform parallel reduction within the work group
        for (int stride = (groupSize / 2); stride > 0; stride /= 2) {
            context.localBarrier();
            if (lid < stride) {
                localX[lid] += localX[lid + stride];
            }
        }

        // Each workgroup stores its partial sum in a different location
        if (lid == 0) {
            // Store the partial sum from each workgroup
            output.set(groupId + 1, localX[0]);
        }

        // Only the first thread in the first workgroup computes the final normalization factor
        if (gid == 0) {
            // Combine partial sums from all workgroups
            float ss = 0.0f;
            for (int i = 1; i <= (size / localMemSize); i++) {  // Assuming 8 workgroups
                ss += output.get(i);
            }

            ss /= size;
            ss += ermsNorm;
            ss = 1.0f / TornadoMath.sqrt(ss);
            output.set(0, ss);  // Store the final scale factor
        }
    }

    /**
     * Applies the computed normalization factor to scale weights.
     * This is the second phase of RMS normalization.
     *
     * @param context Kernel execution context
     * @param output Array for normalized output
     * @param weights Weight values to normalize
     * @param temp Temporary array containing a normalization factor at index 0
     */
    public static void reductionOneBlock2WithLogits(KernelContext context, FloatArray output, FloatArray weights, FloatArray temp) {
        int gid = context.globalIdx;
        float ss = temp.get(0);
        output.set(gid, weights.get(gid) * (ss * output.get(gid)));
    }

}
