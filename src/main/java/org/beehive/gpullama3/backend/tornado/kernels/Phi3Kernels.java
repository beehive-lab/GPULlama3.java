package org.beehive.gpullama3.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * Phi3Kernels: Optimized GPU kernels for Phi3 model family.
 *
 * <p>Key differences from Qwen/Llama kernels:
 *
 * <ul>
 *   <li>Generic fused RMS + matmul (single output matrix)
 *   <li>Phi3 RoPE with headSize/2 offset pattern
 *   <li>Combined gate/up structure support
 * </ul>
 */
public class Phi3Kernels {

    /**
     * Fused RMSNorm apply + single matrix-vector multiplication.
     *
     * <p>Combines RMS normalization application with a generic matmul in one kernel, reducing
     * memory bandwidth by avoiding intermediate storage.
     *
     * <p>Formula: output[row] = sum_j(W[row,j] * rmsWeight[j] * scale * x[j])
     *
     * <p>Use cases:
     *
     * <ul>
     *   <li>Phi3 combined QKV projection (output = wqkv · RMSNorm(x))
     *   <li>Phi3 combined gate/up projection (output = wUp · RMSNorm(x))
     *   <li>Any single-matrix projection after RMSNorm
     * </ul>
     *
     * @param context Kernel execution context
     * @param x Input hidden state (FP32) [dim]
     * @param output Output buffer (FP32) [outputDim]
     * @param rmsWeights RMS normalization weights (FP32) [dim]
     * @param rmsScale Precomputed RMS scale factor [1] (from reduction kernel)
     * @param w Weight matrix (FP16) [outputDim × dim]
     * @param inputDim Input dimension (dim)
     * @param outputDim Output dimension
     * @param localWorkGroupSize Local work group size for reduction
     */
    public static void fusedRmsNormMatmul(
            KernelContext context,
            FloatArray x, // input (FP32)
            FloatArray output, // output (FP32)
            FloatArray rmsWeights, // RMS norm weights
            FloatArray rmsScale, // temp[0] = scale factor
            HalfFloatArray w, // weight matrix
            int inputDim, // input dimension
            int outputDim, // output dimension
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;

        if (rowId >= outputDim) {
            return;
        }

        float scale = rmsScale.get(0);

        // Allocate shared memory for reduction
        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);

        int rowOffset = rowId * inputDim;

        // Each thread computes partial dot product with inline normalization
        float partialSum = 0.0f;
        for (int j = localId; j < inputDim; j += localWorkGroupSize) {
            float normalized = rmsWeights.get(j) * scale * x.get(j);
            partialSum += w.get(rowOffset + j).getFloat32() * normalized;
        }

        localSum[localId] = partialSum;
        context.localBarrier();

        // Parallel reduction within workgroup
        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }

        // Thread 0 writes final result
        if (localId == 0) {
            output.set(rowId, localSum[0]);
        }
    }

    /**
     * Fused RMSNorm apply + QKV projection with direct output to separate Q, K, V buffers.
     *
     * <p>Eliminates the need for a separate splitQKV kernel by routing outputs directly based on
     * row index:
     *
     * <ul>
     *   <li>Rows [0, dim): Q projection
     *   <li>Rows [dim, dim+kvDim): K projection
     *   <li>Rows [dim+kvDim, dim+2*kvDim): V projection
     * </ul>
     *
     * <p>Formula: output[row] = sum_j(Wqkv[row,j] * rmsWeight[j] * scale * x[j])
     *
     * @param context Kernel execution context
     * @param x Input hidden state (FP32) [dim]
     * @param q Output Q buffer (FP32) [dim]
     * @param k Output K buffer (FP32) [kvDim]
     * @param v Output V buffer (FP32) [kvDim]
     * @param rmsWeights RMS normalization weights (FP32) [dim]
     * @param rmsScale Precomputed RMS scale factor [1]
     * @param wqkv Combined QKV weight matrix (FP16) [opSize × dim]
     * @param dim Model dimension (Q output size)
     * @param kvDim KV dimension (K/V output size)
     * @param localWorkGroupSize Local work group size for reduction
     */
    public static void fusedRmsNormQKVMatmulDirect(
            KernelContext context,
            FloatArray x, // input (FP32)
            FloatArray q, // output Q (FP32)
            FloatArray k, // output K (FP32)
            FloatArray v, // output V (FP32)
            FloatArray rmsWeights, // RMS norm weights
            FloatArray rmsScale, // temp[0] = scale factor
            HalfFloatArray wqkv, // combined QKV weight matrix
            int dim, // input dim and Q output dim
            int kvDim, // K/V output dim
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;

        // Total rows = dim (Q) + kvDim (K) + kvDim (V)
        int totalRows = dim + 2 * kvDim;
        if (rowId >= totalRows) {
            return;
        }

        float scale = rmsScale.get(0);

        // Allocate shared memory for reduction
        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);

        int rowOffset = rowId * dim;

        // Each thread computes partial dot product with inline normalization
        float partialSum = 0.0f;
        for (int j = localId; j < dim; j += localWorkGroupSize) {
            float normalized = rmsWeights.get(j) * scale * x.get(j);
            partialSum += wqkv.get(rowOffset + j).getFloat32() * normalized;
        }

        localSum[localId] = partialSum;
        context.localBarrier();

        // Parallel reduction within workgroup
        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }

        // Thread 0 writes to appropriate output buffer
        if (localId == 0) {
            float result = localSum[0];

            if (rowId < dim) {
                // Q projection: rows [0, dim)
                q.set(rowId, result);
            } else if (rowId < dim + kvDim) {
                // K projection: rows [dim, dim+kvDim)
                int kIdx = rowId - dim;
                k.set(kIdx, result);
            } else {
                // V projection: rows [dim+kvDim, dim+2*kvDim)
                int vIdx = rowId - dim - kvDim;
                v.set(vIdx, result);
            }
        }
    }

    /**
     * Fused RMSNorm apply + Gate/Up projection + SiLU + GLU in one kernel.
     *
     * <p>Eliminates the need for separate gateUpSiLU kernel by computing both gate and up
     * projections per workgroup and applying activation inline.
     *
     * <p>For each output index i:
     *
     * <ul>
     *   <li>gate[i] = dot(wUp[i], RMSNorm(x))
     *   <li>up[i] = dot(wUp[hiddenDim + i], RMSNorm(x))
     *   <li>output[i] = SiLU(gate[i]) × up[i]
     * </ul>
     *
     * @param context Kernel execution context
     * @param x Input hidden state (FP32) [dim]
     * @param output Output buffer (FP32) [hiddenDim] - final FFN result
     * @param rmsWeights RMS normalization weights (FP32) [dim]
     * @param rmsScale Precomputed RMS scale factor [1]
     * @param wUp Combined gate+up weight matrix (FP16) [2×hiddenDim × dim]
     * @param dim Input dimension
     * @param hiddenDim Hidden dimension (output size)
     * @param localWorkGroupSize Local work group size for reduction
     */
    public static void fusedRmsNormFFNGateUpSiLU(
            KernelContext context,
            FloatArray x, // input (FP32)
            FloatArray output, // output (FP32) [hiddenDim]
            FloatArray rmsWeights, // RMS norm weights
            FloatArray rmsScale, // temp[0] = scale factor
            HalfFloatArray wUp, // combined gate+up weights [2×hiddenDim × dim]
            int dim, // input dimension
            int hiddenDim, // output dimension
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;

        if (rowId >= hiddenDim) {
            return;
        }

        float scale = rmsScale.get(0);

        // Allocate shared memory for reduction
        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);

        // === Compute GATE (row i) ===
        int gateRowOffset = rowId * dim;

        float gatePartialSum = 0.0f;
        for (int j = localId; j < dim; j += localWorkGroupSize) {
            float normalized = rmsWeights.get(j) * scale * x.get(j);
            gatePartialSum += wUp.get(gateRowOffset + j).getFloat32() * normalized;
        }

        localSum[localId] = gatePartialSum;
        context.localBarrier();

        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }

        float gateResult = localSum[0];

        // === Compute UP (row hiddenDim + i) ===
        int upRowOffset = (hiddenDim + rowId) * dim;

        float upPartialSum = 0.0f;
        for (int j = localId; j < dim; j += localWorkGroupSize) {
            float normalized = rmsWeights.get(j) * scale * x.get(j);
            upPartialSum += wUp.get(upRowOffset + j).getFloat32() * normalized;
        }

        localSum[localId] = upPartialSum;
        context.localBarrier();

        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }

        float upResult = localSum[0];

        // === Apply SiLU(gate) × up ===
        if (localId == 0) {
            float silu = gateResult / (1.0f + TornadoMath.exp(-gateResult));
            output.set(rowId, silu * upResult);
        }
    }
}
