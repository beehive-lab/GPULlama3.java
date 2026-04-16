package org.beehive.gpullama3.tornadovm.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
public class Gemma4Kernels {

    // ═══════════════════════════════════════════════════════════════════════
    //  4a. Fused RMS Norm + FFN Gate/Up + GELU (instead of SiLU) for Q8_0
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Fused RMSNorm apply + Gate/Up projection + GELU + GLU for Q8_0 weights.
     * Same structure as TransformerComputeKernelsLayered.fusedRmsNormFFNGateUpQ8_0
     * but uses GELU activation instead of SiLU.
     */
    public static void fusedRmsNormFFNGateUpGeluQ8_0(
            KernelContext context,
            FloatArray x,               // raw input (FP32)
            FloatArray hb,              // output: GELU(x·W1) * (x·W3)
            FloatArray rmsWeights,      // RMS norm weights
            FloatArray rmsScale,        // tempFFN[0] = scale factor
            ByteArray w1,               // W1 (gate) Q8_0 weights
            ByteArray w3,               // W3 (up) Q8_0 weights
            int inputDim,               // input dimension
            int hiddenDim,              // hidden dimension
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;

        if (rowId >= hiddenDim) {
            return;
        }

        float scale = rmsScale.get(0);
        final int blockSize = 32;
        final int Q8_0_BLOCK_BYTES = 34;

        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize);

        int blocksPerRow = (inputDim + blockSize - 1) / blockSize;
        int w1RowBlockOffset = rowId * blocksPerRow;
        int w3RowBlockOffset = rowId * blocksPerRow;

        // ========== W1 computation with inline RMS normalization ==========
        float partialSum1_1 = 0.0f, partialSum1_2 = 0.0f, partialSum1_3 = 0.0f, partialSum1_4 = 0.0f;

        for (int j = localId * 4; j < inputDim - 3; j += localWorkGroupSize * 4) {
            int blockIdx = j / blockSize;
            int withinBlockIdx = j % blockSize;

            int w1BlockByteOffset = (w1RowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;
            HalfFloat w1Scale = w1.getHalfFloat(w1BlockByteOffset);
            float w1ScaleFloat = w1Scale.getFloat32();

            int w1QuantsOffset = w1BlockByteOffset + 2 + withinBlockIdx;
            byte w1Quant1 = w1.get(w1QuantsOffset);
            byte w1Quant2 = w1.get(w1QuantsOffset + 1);
            byte w1Quant3 = w1.get(w1QuantsOffset + 2);
            byte w1Quant4 = w1.get(w1QuantsOffset + 3);

            float norm1 = rmsWeights.get(j) * (scale * x.get(j));
            float norm2 = rmsWeights.get(j + 1) * (scale * x.get(j + 1));
            float norm3 = rmsWeights.get(j + 2) * (scale * x.get(j + 2));
            float norm4 = rmsWeights.get(j + 3) * (scale * x.get(j + 3));

            partialSum1_1 += ((float) w1Quant1 * w1ScaleFloat) * norm1;
            partialSum1_2 += ((float) w1Quant2 * w1ScaleFloat) * norm2;
            partialSum1_3 += ((float) w1Quant3 * w1ScaleFloat) * norm3;
            partialSum1_4 += ((float) w1Quant4 * w1ScaleFloat) * norm4;
        }

        float partialSum1 = partialSum1_1 + partialSum1_2 + partialSum1_3 + partialSum1_4;

        for (int j = ((inputDim / 4) * 4) + localId; j < inputDim; j += localWorkGroupSize) {
            int blockIdx = j / blockSize;
            int withinBlockIdx = j % blockSize;

            int w1BlockByteOffset = (w1RowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;
            HalfFloat w1Scale = w1.getHalfFloat(w1BlockByteOffset);
            float w1ScaleFloat = w1Scale.getFloat32();

            byte w1Quant = w1.get(w1BlockByteOffset + 2 + withinBlockIdx);
            float normalized = rmsWeights.get(j) * (scale * x.get(j));

            partialSum1 += ((float) w1Quant * w1ScaleFloat) * normalized;
        }

        localSums[localId] = partialSum1;
        context.localBarrier();

        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }

        float sum1 = localSums[0];

        // ========== W3 computation with inline RMS normalization ==========
        float partialSum3_1 = 0.0f, partialSum3_2 = 0.0f, partialSum3_3 = 0.0f, partialSum3_4 = 0.0f;

        for (int j = localId * 4; j < inputDim - 3; j += localWorkGroupSize * 4) {
            int blockIdx = j / blockSize;
            int withinBlockIdx = j % blockSize;

            int w3BlockByteOffset = (w3RowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;
            HalfFloat w3Scale = w3.getHalfFloat(w3BlockByteOffset);
            float w3ScaleFloat = w3Scale.getFloat32();

            int w3QuantsOffset = w3BlockByteOffset + 2 + withinBlockIdx;
            byte w3Quant1 = w3.get(w3QuantsOffset);
            byte w3Quant2 = w3.get(w3QuantsOffset + 1);
            byte w3Quant3 = w3.get(w3QuantsOffset + 2);
            byte w3Quant4 = w3.get(w3QuantsOffset + 3);

            float norm1 = rmsWeights.get(j) * (scale * x.get(j));
            float norm2 = rmsWeights.get(j + 1) * (scale * x.get(j + 1));
            float norm3 = rmsWeights.get(j + 2) * (scale * x.get(j + 2));
            float norm4 = rmsWeights.get(j + 3) * (scale * x.get(j + 3));

            partialSum3_1 += ((float) w3Quant1 * w3ScaleFloat) * norm1;
            partialSum3_2 += ((float) w3Quant2 * w3ScaleFloat) * norm2;
            partialSum3_3 += ((float) w3Quant3 * w3ScaleFloat) * norm3;
            partialSum3_4 += ((float) w3Quant4 * w3ScaleFloat) * norm4;
        }

        float partialSum3 = partialSum3_1 + partialSum3_2 + partialSum3_3 + partialSum3_4;

        for (int j = ((inputDim / 4) * 4) + localId; j < inputDim; j += localWorkGroupSize) {
            int blockIdx = j / blockSize;
            int withinBlockIdx = j % blockSize;

            int w3BlockByteOffset = (w3RowBlockOffset + blockIdx) * Q8_0_BLOCK_BYTES;
            HalfFloat w3Scale = w3.getHalfFloat(w3BlockByteOffset);
            float w3ScaleFloat = w3Scale.getFloat32();

            byte w3Quant = w3.get(w3BlockByteOffset + 2 + withinBlockIdx);
            float normalized = rmsWeights.get(j) * (scale * x.get(j));

            partialSum3 += ((float) w3Quant * w3ScaleFloat) * normalized;
        }

        localSums[localId] = partialSum3;
        context.localBarrier();

        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }

        float sum3 = localSums[0];

        // === GELU + GLU ===
        if (localId == 0) {
            float gelu = TransformerComputeKernelsLayered.geluActivation(sum1);
            hb.set(rowId, gelu * sum3);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  4b. Q8_0 matrix-vector multiplication (write, no residual add)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Q8_0 matrix-vector multiplication that writes the result to output (overwrite, no residual).
     * Computes: output[row] = (W · x)[row]
     */
    public static void matrixVectorWriteQ8_0(
            KernelContext context,
            FloatArray x,              // input vector
            FloatArray output,         // output vector (overwritten)
            ByteArray w,               // weight matrix (Q8_0)
            int n,                     // input dimension
            int d,                     // output dimension
            int localWorkGroupSize) {

        int rowId = context.groupIdx;
        int localId = context.localIdx;

        if (rowId >= d) {
            return;
        }

        float sum = TransformerComputeKernelsLayered.matrixVectorRowMajorOptimizedQ8_0Byte(
                context, localWorkGroupSize, x, w, n);

        if (localId == 0) {
            output.set(rowId, sum);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  4b2. RMS norm + weighted residual add
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Apply RMS normalization (using precomputed scale from temp[0]) with learned weights,
     * then add to residual. Matches CPU: residual[i] += normWeights[i] * scale * buffer[i]
     * where scale = 1/sqrt(mean(buffer^2) + eps), precomputed by reductionOneBlockWithLayer.
     */
    public static void rmsNormWeightedResidual(
            FloatArray residual,       // x (in/out) - residual accumulator
            FloatArray buffer,         // matmul output to be normalized
            FloatArray normWeights,    // post-norm learned weights
            FloatArray temp,           // temp[0] = precomputed RMS scale factor
            int dim) {
        float scale = temp.get(0);
        for (@Parallel int i = 0; i < dim; i++) {
            residual.set(i, residual.get(i) + normWeights.get(i) * scale * buffer.get(i));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  4b3. Bare RMS norm per head (for V normalization, no learned weights)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Apply bare RMS normalization (no learned weights) independently per KV head.
     * Gemma4 normalizes V to unit RMS per head before caching.
     * Computes: v[h*headSize+i] *= 1/sqrt(mean(v[h*headSize .. (h+1)*headSize-1]^2) + eps)
     */
    public static void bareRmsNormPerHead(
            KernelContext context,
            FloatArray v,              // V vector (in/out)
            int nKvHeads,
            int headSize,
            float rmsNormEps) {

        int h = context.groupIdx;        // head index
        int tid = context.localIdx;      // thread within workgroup
        int localSize = context.localGroupSizeX;

        if (h >= nKvHeads) {
            return;
        }

        int offset = h * headSize;

        // Allocate local memory for parallel reduction
        float[] localSums = context.allocateFloatLocalArray(localSize);

        // Step 1: Compute partial sum of squares
        float partialSum = 0.0f;
        for (int i = tid; i < headSize; i += localSize) {
            float val = v.get(offset + i);
            partialSum += val * val;
        }

        localSums[tid] = partialSum;
        context.localBarrier();

        // Parallel reduction
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                localSums[tid] += localSums[tid + stride];
            }
            context.localBarrier();
        }

        // Compute scale: 1/sqrt(mean(v^2) + eps)
        float ss = localSums[0];
        ss = ss / headSize + rmsNormEps;
        ss = 1.0f / TornadoMath.sqrt(ss);

        context.localBarrier();

        // Step 2: Apply bare normalization in-place (no learned weights)
        for (int i = tid; i < headSize; i += localSize) {
            v.set(offset + i, ss * v.get(offset + i));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  4c. Apply per-layer output scale
    // ═══════════════════════════════════════════════════════════════════════

    public static void applyLayerOutputScale(FloatArray x, float scale, int dim) {
        for (@Parallel int i = 0; i < dim; i++) {
            x.set(i, x.get(i) * scale);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  4d. RoPE rotation with KV cache copy using precomputed frequencies
    //      and per-layer cache offset (supports SWA)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Fused RoPE rotation with KV cache copy for Gemma4.
     * Uses precomputed RoPE frequency arrays (not inline theta computation).
     * Supports per-layer cache offsets and SWA sliding window indexing.
     *
     * @param flags bit 0 (& 1) = isSWA (1=SWA, 0=full attention),
     *              bit 1 (& 2) = writeCache (2=write K/V to cache, 0=skip for shared KV layers)
     */
    public static void ropeRotationWithCacheCopyGemma4(
            KernelContext context,
            IntArray positionHolder,
            FloatArray q,              // Q vector (in/out)
            FloatArray k,              // K vector (in/out)
            FloatArray v,              // V vector (in only)
            FloatArray keyCache,       // Key cache (out)
            FloatArray valueCache,     // Value cache (out)
            FloatArray freqCisReal,    // precomputed cos values
            FloatArray freqCisImag,    // precomputed sin values
            int numberOfKeyValueHeads,
            int headSize,
            int kvDim,
            int cacheOffset,           // per-layer offset into flat cache
            int slidingWindow,
            int flags) {               // bit0=isSWA, bit1=writeCache

        int h = context.globalIdx;     // head index
        int ic = context.globalIdy;    // half-head dimension index

        int isSWA = flags & 1;
        int writeCache = (flags >> 1) & 1;

        int pos = positionHolder.get(0);
        int poffset = h * headSize;
        int halfHead = headSize / 2;

        // Look up precomputed RoPE frequencies
        int freqIdx = pos * halfHead + ic;
        float fcr = freqCisReal.get(freqIdx);
        float fci = freqCisImag.get(freqIdx);

        // Rotate Q (all heads, always)
        float v0q = q.get(poffset + ic);
        float v1q = q.get(poffset + ic + halfHead);
        q.set(poffset + ic, v0q * fcr - v1q * fci);
        q.set(poffset + ic + halfHead, v0q * fci + v1q * fcr);

        // Rotate K and copy K/V to cache (only for KV heads, only if writeCache)
        if (writeCache == 1 && h < numberOfKeyValueHeads && (poffset + ic + halfHead) < k.getSize()) {
            float v0k = k.get(poffset + ic);
            float v1k = k.get(poffset + ic + halfHead);
            float rotatedK0 = v0k * fcr - v1k * fci;
            float rotatedK1 = v0k * fci + v1k * fcr;

            k.set(poffset + ic, rotatedK0);
            k.set(poffset + ic + halfHead, rotatedK1);

            // Cache position: SWA uses modular indexing, full uses direct position
            int cachePos = (isSWA == 1) ? (pos & (slidingWindow - 1)) : pos;
            int cacheIdx = cacheOffset + cachePos * kvDim;
            int kvIdx = h * headSize;

            keyCache.set(cacheIdx + kvIdx + ic, rotatedK0);
            keyCache.set(cacheIdx + kvIdx + ic + halfHead, rotatedK1);

            valueCache.set(cacheIdx + kvIdx + ic, v.get(poffset + ic));
            valueCache.set(cacheIdx + kvIdx + ic + halfHead, v.get(poffset + ic + halfHead));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  4e. Flash Attention for Gemma4
    //      - No attention scaling (score = Q · K, not score /= sqrt(headSize))
    //      - SWA window: only attend to recent positions
    //      - Per-layer cache offset
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Flash Attention for Gemma4 with per-layer cache offset and optional SWA window.
     * No attention scaling applied (Gemma4 uses scale=1.0).
     */
    public static void processHeadsFlashAttentionGemma4(
            KernelContext context,
            FloatArray q,
            FloatArray keyCache,
            FloatArray valueCache,
            FloatArray xb,
            int nHeads,
            int headSize,
            int kvDim,
            int kvMul,
            IntArray positionHolder,
            int cacheOffset,           // per-layer offset into flat cache
            int contextLength,
            int isSWA,                 // 1 = SWA, 0 = full
            int slidingWindow) {

        int tid = context.localIdx;
        int h = context.groupIdx;
        int localSize = context.localGroupSizeX;

        if (h >= nHeads) {
            return;
        }

        int pos = positionHolder.get(0);
        int kvHeadIdx = h / kvMul;
        // Adaptive tile size to stay within shared memory limits (~48KB)
        // K+V tiles: 2 * BLOCK_SIZE_C * headSize * 4 bytes
        // For headSize=512: BLOCK_SIZE_C=8 → 32KB (OK), BLOCK_SIZE_C=16 → 64KB (exceeds 48KB)
        int BLOCK_SIZE_C = headSize <= 256 ? 16 : 8;

        // Determine attention window
        int startPos = (isSWA == 1) ? Math.max(0, pos - slidingWindow + 1) : 0;

        float[] q_shared = context.allocateFloatLocalArray(headSize);
        float[] k_tile = context.allocateFloatLocalArray(BLOCK_SIZE_C * headSize);
        float[] v_tile = context.allocateFloatLocalArray(BLOCK_SIZE_C * headSize);
        float[] s_tile = context.allocateFloatLocalArray(BLOCK_SIZE_C);
        float[] shared_tile_max_holder = context.allocateFloatLocalArray(1);

        float maxScore = Float.NEGATIVE_INFINITY;
        float sumExp = 0.0f;

        float[] output = new float[headSize];
        for (int i = 0; i < headSize; i++) {
            output[i] = 0.0f;
        }

        // Load query vector into shared memory
        for (int i = tid; i < headSize; i += localSize) {
            q_shared[i] = q.get(h * headSize + i);
        }

        context.localBarrier();

        // Process sequence in tiles
        for (int tileC = startPos; tileC <= pos; tileC += BLOCK_SIZE_C) {
            int tileEnd = Math.min(tileC + BLOCK_SIZE_C - 1, pos);

            // Load K and V tiles from per-layer cache
            for (int tIdxInSeq = tileC + tid; tIdxInSeq <= tileEnd; tIdxInSeq += localSize) {
                int k_v_idx_in_tile = tIdxInSeq - tileC;
                int tileMemOffset = k_v_idx_in_tile * headSize;

                // For SWA, map the actual position to the circular cache position
                int cachePos = (isSWA == 1) ? (tIdxInSeq & (slidingWindow - 1)) : tIdxInSeq;

                for (int d = 0; d < headSize; d++) {
                    int kvOffset = cacheOffset + cachePos * kvDim + kvHeadIdx * headSize + d;
                    k_tile[tileMemOffset + d] = keyCache.get(kvOffset);
                    v_tile[tileMemOffset + d] = valueCache.get(kvOffset);
                }
            }

            context.localBarrier();

            // Compute attention scores (NO scaling - Gemma4 uses scale=1.0)
            for (int tIdxInSeq = tileC + tid; tIdxInSeq <= tileEnd; tIdxInSeq += localSize) {
                int score_idx_in_tile = tIdxInSeq - tileC;

                float score = 0.0f;
                for (int d = 0; d < headSize; d++) {
                    score += q_shared[d] * k_tile[score_idx_in_tile * headSize + d];
                }
                // No scaling: Gemma4 uses attention_scale=1.0
                s_tile[score_idx_in_tile] = score;
            }

            context.localBarrier();

            // Find tile max
            float tileLocalMax = Float.NEGATIVE_INFINITY;
            for (int i = 0; i <= tileEnd - tileC; i++) {
                if (s_tile[i] > tileLocalMax) {
                    tileLocalMax = s_tile[i];
                }
            }

            if (tid == 0) {
                shared_tile_max_holder[0] = tileLocalMax;
            }
            context.localBarrier();
            float currentTileMax = shared_tile_max_holder[0];

            // Rescale previous results if needed
            float newMax = Math.max(maxScore, currentTileMax);
            if (newMax != maxScore && maxScore != Float.NEGATIVE_INFINITY) {
                float rescale = TornadoMath.exp(maxScore - newMax);
                sumExp *= rescale;
                for (int d = 0; d < headSize; d++) {
                    output[d] *= rescale;
                }
            }
            maxScore = newMax;

            // Accumulate weighted values
            for (int t_idx = 0; t_idx <= tileEnd - tileC; t_idx++) {
                float expScore = TornadoMath.exp(s_tile[t_idx] - maxScore);
                sumExp += expScore;

                for (int d = 0; d < headSize; d++) {
                    output[d] += expScore * v_tile[t_idx * headSize + d];
                }
            }
            context.localBarrier();
        }

        // Normalize and write results
        float normFactor = (sumExp > 0.0f) ? (1.0f / sumExp) : 0.0f;
        for (int d = tid; d < headSize; d += localSize) {
            xb.set(h * headSize + d, output[d] * normFactor);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  4f. Add per-layer embedding contribution
    // ═══════════════════════════════════════════════════════════════════════

    public static void addPerLayerEmbedding(FloatArray wrapX, FloatArray perLayerContribs, int layerOffset, int dim) {
        for (@Parallel int i = 0; i < dim; i++) {
            wrapX.set(i, wrapX.get(i) + perLayerContribs.get(layerOffset + i));
        }
    }
}
// @formatter:on
