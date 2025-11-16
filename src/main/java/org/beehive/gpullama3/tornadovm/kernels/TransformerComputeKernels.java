package org.beehive.gpullama3.tornadovm.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

public class TransformerComputeKernels {

    /**
     * Default constructor for the TransformerComputeKernels class.
     */
    public TransformerComputeKernels() {
    }

    public static void copyInEmbeddingActivation(FloatArray buffer) {
        float dummy = buffer.get(0);
        if (dummy > Float.MAX_VALUE) {
            buffer.set(0, dummy);
        }
    }

    public static void copyInEmbeddingActivationFP16(HalfFloatArray buffer) {
        float dummy = buffer.get(0).getFloat32();
        if (dummy > Float.MAX_VALUE) {
            buffer.set(0, new HalfFloat(dummy));
        }
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
    public static void reductionOneBlockWithLayer(KernelContext context, FloatArray output, HalfFloatArray x, int size, float ermsNorm, int localMemSize) {
        int gid = context.globalIdx;
        int lid = context.localIdx;
        int groupId = context.groupIdx;
        int groupSize = context.localGroupSizeX;

        // Allocate local memory with the provided size
        float[] localX = context.allocateFloatLocalArray(localMemSize);

        // Load input value and compute square
        if (gid < size) {
            localX[lid] = x.get(gid).getFloat32();
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
     *
     *
     */

    public static void copyHack(HalfFloatArray x, HalfFloatArray hackX) {
        for (@Parallel int i = 0; i < x.getSize(); i++) {
            hackX.set(i, x.get(i));
        }
    }

    public static void reductionOneBlock2WithLogits(KernelContext context, HalfFloatArray output, FloatArray weights, FloatArray temp) {
//        int gid = context.globalIdx;
//        float ss = temp.get(0);
//        output.set(gid, new HalfFloat((weights.get(gid) * (ss * output.get(gid).getFloat32()))));


        int gid = context.globalIdx;

        // Step 1: read normalization scalar
        float ss = temp.get(0);

        // Step 2: read current output value as float
        HalfFloat hf = output.get(gid);
        float out_f = hf.getFloat32();

        // Step 3: read weight
//        float w = weights.get(gid);

        // Step 4: compute scaled output
        float scaled = ss * out_f;

        // Step 5: multiply by weight
        float prod = weights.get(gid) * scaled;

        // Step 6: create HalfFloat result
        HalfFloat result = new HalfFloat(prod);

        // Step 7: write back
        output.set(gid, result);
    }


    public static void reductionOneBlock2WithLogits2(KernelContext context, HalfFloatArray input, HalfFloatArray output, FloatArray weights, FloatArray temp) {
        int gid = context.globalIdx;
        float ss = temp.get(0);
        float inter = ss * input.get(gid).getHalfFloatValue();
        HalfFloat x = new HalfFloat((weights.get(gid) * inter));
        output.set(gid, x);
    }

}
