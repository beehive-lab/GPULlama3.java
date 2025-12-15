package org.beehive.gpullama3.tornadovm.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Phi3Kernels: Optimized GPU kernels for Phi3 model family.
 *
 * <p>Key differences from Qwen/Llama kernels:</p>
 * <ul>
 *   <li>Generic fused RMS + matmul (single output matrix)</li>
 *   <li>Phi3 RoPE with headSize/2 offset pattern</li>
 *   <li>Combined gate/up structure support</li>
 * </ul>
 */
public class Phi3Kernels {

    /**
     * Fused RMSNorm apply + single matrix-vector multiplication.
     *
     * <p>Combines RMS normalization application with a generic matmul in one kernel,
     * reducing memory bandwidth by avoiding intermediate storage.</p>
     *
     * <p>Formula: output[row] = sum_j(W[row,j] * rmsWeight[j] * scale * x[j])</p>
     *
     * <p>Use cases:</p>
     * <ul>
     *   <li>Phi3 combined QKV projection (output = wqkv · RMSNorm(x))</li>
     *   <li>Phi3 combined gate/up projection (output = wUp · RMSNorm(x))</li>
     *   <li>Any single-matrix projection after RMSNorm</li>
     * </ul>
     *
     * @param context         Kernel execution context
     * @param x               Input hidden state (FP32) [dim]
     * @param output          Output buffer (FP32) [outputDim]
     * @param rmsWeights      RMS normalization weights (FP32) [dim]
     * @param rmsScale        Precomputed RMS scale factor [1] (from reduction kernel)
     * @param w               Weight matrix (FP16) [outputDim × dim]
     * @param inputDim        Input dimension (dim)
     * @param outputDim       Output dimension
     * @param localWorkGroupSize  Local work group size for reduction
     */
    public static void fusedRmsNormMatmul(
            KernelContext context,
            FloatArray x,               // input (FP32)
            FloatArray output,          // output (FP32)
            FloatArray rmsWeights,      // RMS norm weights
            FloatArray rmsScale,        // temp[0] = scale factor
            HalfFloatArray w,           // weight matrix
            int inputDim,               // input dimension
            int outputDim,              // output dimension
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
     * Phi3 RoPE rotation with fused KV cache copy.
     *
     * <p>Phi3 uses a different RoPE pattern than Llama/Qwen:</p>
     * <ul>
     *   <li>Pairs elements with offset headSize/2 (not adjacent pairs)</li>
     *   <li>Each thread processes one dimension pair across all heads</li>
     *   <li>Iterates over heads internally</li>
     * </ul>
     *
     * <p>This fused kernel combines:</p>
     * <ul>
     *   <li>Phi3-style RoPE rotation for Q and K</li>
     *   <li>Direct cache write for rotated K</li>
     *   <li>Direct cache copy for V (no rotation)</li>
     * </ul>
     *
     * @param context         Kernel execution context
     * @param positionHolder  Current position in sequence [1]
     * @param sq              Query vectors (in/out, rotated) [dim]
     * @param sk              Key vectors (in/out, rotated) [kvDim]
     * @param sv              Value vectors (in only) [kvDim]
     * @param keyCache        Key cache (out) [layers × contextLength × kvDim]
     * @param valueCache      Value cache (out) [layers × contextLength × kvDim]
     * @param nHeadKv         Number of KV heads
     * @param headSize        Dimension per head
     * @param kvDim           Total KV dimension (nHeadKv × headSize)
     * @param layer           Current layer index
     * @param contextLength   Maximum sequence length
     */
    public static void ropeRotationWithCacheCopyPhi3(
            KernelContext context,
            IntArray positionHolder,
            FloatArray sq,              // Q vector (in/out)
            FloatArray sk,              // K vector (in/out)
            FloatArray sv,              // V vector (in only)
            FloatArray keyCache,        // Key cache (out)
            FloatArray valueCache,      // Value cache (out)
            int nHeadKv,
            int headSize,
            int kvDim,
            int layer,
            int contextLength) {

        int idx = context.globalIdx;
        int dimHalf = headSize / 2;

        // Each thread processes one dimension pair
        if (idx >= dimHalf) {
            return;
        }

        int pos = positionHolder.get(0);
        int cacheOffset = layer * contextLength * kvDim + pos * kvDim;

        // Calculate frequency for this dimension
        float freq = 1.0f / TornadoMath.pow(10000.0f, (float) (idx * 2) / (float) headSize);
        float val = pos * freq;
        float fcr = TornadoMath.cos(val);
        float fci = TornadoMath.sin(val);

        // Process Q: all heads (dim = nHeads × headSize)
        int totalDimQ = sq.getSize();
        for (int base = 0; base < totalDimQ; base += headSize) {
            if (base + idx >= totalDimQ || base + idx + dimHalf >= totalDimQ) {
                break;
            }

            // Rotate Q with offset pattern
            float v0 = sq.get(base + idx);
            float v1 = sq.get(base + idx + dimHalf);
            sq.set(base + idx, v0 * fcr - v1 * fci);
            sq.set(base + idx + dimHalf, v0 * fci + v1 * fcr);
        }

        // Process K: only kvDim elements, with cache write
        for (int base = 0; base < kvDim; base += headSize) {
            if (base + idx >= kvDim || base + idx + dimHalf >= kvDim) {
                break;
            }

            // Rotate K with offset pattern
            float k0 = sk.get(base + idx);
            float k1 = sk.get(base + idx + dimHalf);
            float rotated0 = k0 * fcr - k1 * fci;
            float rotated1 = k0 * fci + k1 * fcr;

            // Write rotated K back
            sk.set(base + idx, rotated0);
            sk.set(base + idx + dimHalf, rotated1);

            // Fused cache write for K
            keyCache.set(cacheOffset + base + idx, rotated0);
            keyCache.set(cacheOffset + base + idx + dimHalf, rotated1);

            // Fused cache copy for V (no rotation needed)
            valueCache.set(cacheOffset + base + idx, sv.get(base + idx));
            valueCache.set(cacheOffset + base + idx + dimHalf, sv.get(base + idx + dimHalf));
        }
    }

    /**
     * Fused RMSNorm apply + QKV projection with direct output to separate Q, K, V buffers.
     *
     * <p>Eliminates the need for a separate splitQKV kernel by routing outputs
     * directly based on row index:</p>
     * <ul>
     *   <li>Rows [0, dim): Q projection</li>
     *   <li>Rows [dim, dim+kvDim): K projection</li>
     *   <li>Rows [dim+kvDim, dim+2*kvDim): V projection</li>
     * </ul>
     *
     * <p>Formula: output[row] = sum_j(Wqkv[row,j] * rmsWeight[j] * scale * x[j])</p>
     *
     * @param context           Kernel execution context
     * @param x                 Input hidden state (FP32) [dim]
     * @param q                 Output Q buffer (FP32) [dim]
     * @param k                 Output K buffer (FP32) [kvDim]
     * @param v                 Output V buffer (FP32) [kvDim]
     * @param rmsWeights        RMS normalization weights (FP32) [dim]
     * @param rmsScale          Precomputed RMS scale factor [1]
     * @param wqkv              Combined QKV weight matrix (FP16) [opSize × dim]
     * @param dim               Model dimension (Q output size)
     * @param kvDim             KV dimension (K/V output size)
     * @param localWorkGroupSize Local work group size for reduction
     */
    public static void fusedRmsNormQKVMatmulDirect(
            KernelContext context,
            FloatArray x,               // input (FP32)
            FloatArray q,               // output Q (FP32)
            FloatArray k,               // output K (FP32)
            FloatArray v,               // output V (FP32)
            FloatArray rmsWeights,      // RMS norm weights
            FloatArray rmsScale,        // temp[0] = scale factor
            HalfFloatArray wqkv,        // combined QKV weight matrix
            int dim,                    // input dim and Q output dim
            int kvDim,                  // K/V output dim
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
     * <p>Eliminates the need for separate gateUpSiLU kernel by computing both
     * gate and up projections per workgroup and applying activation inline.</p>
     *
     * <p>For each output index i:</p>
     * <ul>
     *   <li>gate[i] = dot(wUp[i], RMSNorm(x))</li>
     *   <li>up[i] = dot(wUp[hiddenDim + i], RMSNorm(x))</li>
     *   <li>output[i] = SiLU(gate[i]) × up[i]</li>
     * </ul>
     *
     * @param context           Kernel execution context
     * @param x                 Input hidden state (FP32) [dim]
     * @param output            Output buffer (FP32) [hiddenDim] - final FFN result
     * @param rmsWeights        RMS normalization weights (FP32) [dim]
     * @param rmsScale          Precomputed RMS scale factor [1]
     * @param wUp               Combined gate+up weight matrix (FP16) [2×hiddenDim × dim]
     * @param dim               Input dimension
     * @param hiddenDim         Hidden dimension (output size)
     * @param localWorkGroupSize Local work group size for reduction
     */
    public static void fusedRmsNormFFNGateUpSiLU(
            KernelContext context,
            FloatArray x,               // input (FP32)
            FloatArray output,          // output (FP32) [hiddenDim]
            FloatArray rmsWeights,      // RMS norm weights
            FloatArray rmsScale,        // temp[0] = scale factor
            HalfFloatArray wUp,         // combined gate+up weights [2×hiddenDim × dim]
            int dim,                    // input dimension
            int hiddenDim,              // output dimension
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