package org.beehive.gpullama3.tornadovm.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * GPU kernels for batched prefill (Phase 4).
 *
 * <p>Each kernel processes {@code batchSize} tokens simultaneously.
 * Batch tensors are flat: element [b][i] lives at index {@code b*stride + i}.
 * Worker-grid sizes are scaled by {@code batchSize} vs the single-token kernels.</p>
 *
 * <p>These kernels are meant to be registered in {@link TornadoVMMasterPlanWithPrefillDecode}
 * batch task graphs; they are NOT invoked directly.</p>
 */
public final class TransformerBatchPrefillKernels {

    private TransformerBatchPrefillKernels() {}

    // ── Activation ────────────────────────────────────────────────────────────

    /**
     * Converts B×dim FP16 token embeddings to FP32.
     * Worker: B*dim global threads, localSize=128.
     */
    public static void batchEmbeddingToFP32(KernelContext context,
                                             HalfFloatArray embeddingXBatch,
                                             FloatArray wrapXBatch) {
        int gid = context.globalIdx;
        wrapXBatch.set(gid, embeddingXBatch.get(gid).getFloat32());
    }

    // ── RMS Norm (attention) ─────────────────────────────────────────────────

    /**
     * Sequential RMS reduction — one thread per batch item.
     *
     * <p>Each thread computes the RMS scale factor for its token:
     * {@code scale[b] = 1 / sqrt( mean(x[b]²) + eps )}</p>
     *
     * Worker: batchSize global threads, localSize=1.
     */
    public static void batchedRmsReduce(KernelContext context,
                                         FloatArray wrapXBatch,
                                         FloatArray attnScaleBatch,
                                         int dim, float eps) {
        int b = context.globalIdx;
        int base = b * dim;
        float ss = 0.0f;
        for (int i = 0; i < dim; i++) {
            float val = wrapXBatch.get(base + i);
            ss += val * val;
        }
        ss /= dim;
        ss += eps;
        attnScaleBatch.set(b, 1.0f / TornadoMath.sqrt(ss));
    }

    /**
     * Applies RMS normalization and FP16-quantizes the result.
     *
     * <p>{@code xbFP16Batch[b*dim+i] = FP16( rmsWeights[i] * scale[b] * x[b*dim+i] )}</p>
     *
     * Worker: B*dim global threads, localSize=256.
     */
    public static void batchedRmsApplyFP16(KernelContext context,
                                            HalfFloatArray xbFP16Batch,
                                            FloatArray wrapXBatch,
                                            FloatArray rmsWeights,
                                            FloatArray attnScaleBatch,
                                            int dim) {
        int gid = context.globalIdx;
        int b = gid / dim;
        int i = gid % dim;
        float scale = attnScaleBatch.get(b);
        float result = rmsWeights.get(i) * scale * wrapXBatch.get(gid);
        xbFP16Batch.set(gid, new HalfFloat(result));
    }

    // ── QKV Projection ────────────────────────────────────────────────────────

    /**
     * Fused batched QKV projection (FP16 weights, FP16 input).
     *
     * <p>One workgroup per (batchIdx, outputRow) pair.
     * globalGroupIdx = batchIdx * (dim + 2*kvDim) + rowIdx.</p>
     *
     * Worker: B*(dim+2*kvDim) workgroups × localWorkGroupSize threads.
     */
    public static void batchedFusedQKVMatmul(KernelContext context,
                                              HalfFloatArray xbFP16Batch,
                                              FloatArray wrapQBatch,
                                              FloatArray wrapKBatch,
                                              FloatArray wrapVBatch,
                                              HalfFloatArray wq,
                                              HalfFloatArray wk,
                                              HalfFloatArray wv,
                                              int dim, int kvDim,
                                              int localWorkGroupSize) {
        int groupId = context.groupIdx;
        int localId = context.localIdx;
        int totalRows = dim + 2 * kvDim;
        int batchIdx = groupId / totalRows;
        int rowIdx   = groupId % totalRows;
        int inputOff = batchIdx * dim;

        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);

        if (rowIdx < dim) {
            int rowOff = rowIdx * dim;
            float partial = 0.0f;
            for (int j = localId; j < dim; j += localWorkGroupSize) {
                partial += wq.get(rowOff + j).getFloat32() * xbFP16Batch.get(inputOff + j).getFloat32();
            }
            localSum[localId] = partial;
            context.localBarrier();
            for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
                if (localId < s) localSum[localId] += localSum[localId + s];
                context.localBarrier();
            }
            if (localId == 0) wrapQBatch.set(batchIdx * dim + rowIdx, localSum[0]);

        } else if (rowIdx < dim + kvDim) {
            int kRow = rowIdx - dim;
            int rowOff = kRow * dim;
            float partial = 0.0f;
            for (int j = localId; j < dim; j += localWorkGroupSize) {
                partial += wk.get(rowOff + j).getFloat32() * xbFP16Batch.get(inputOff + j).getFloat32();
            }
            localSum[localId] = partial;
            context.localBarrier();
            for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
                if (localId < s) localSum[localId] += localSum[localId + s];
                context.localBarrier();
            }
            if (localId == 0) wrapKBatch.set(batchIdx * kvDim + kRow, localSum[0]);

        } else {
            int vRow = rowIdx - dim - kvDim;
            int rowOff = vRow * dim;
            float partial = 0.0f;
            for (int j = localId; j < dim; j += localWorkGroupSize) {
                partial += wv.get(rowOff + j).getFloat32() * xbFP16Batch.get(inputOff + j).getFloat32();
            }
            localSum[localId] = partial;
            context.localBarrier();
            for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
                if (localId < s) localSum[localId] += localSum[localId + s];
                context.localBarrier();
            }
            if (localId == 0) wrapVBatch.set(batchIdx * kvDim + vRow, localSum[0]);
        }
    }

    // ── RoPE + KV Cache ───────────────────────────────────────────────────────

    /**
     * Fused batched RoPE rotation + KV cache write.
     *
     * <p>globalIdx encodes (batchIdx, pairIdx) as {@code batchIdx*(dim/2) + pairIdx}.
     * Position for token b = {@code startPos + b}.</p>
     *
     * Worker: B*(dim/2) global threads, localSize=512 (or less if B*dim/2 < 512).
     */
    public static void batchedRopeWithKVCache(KernelContext context,
                                               IntArray batchStartPosHolder,
                                               FloatArray wrapQBatch,
                                               FloatArray wrapKBatch,
                                               FloatArray wrapVBatch,
                                               FloatArray wrapKeyCache,
                                               FloatArray wrapValueCache,
                                               int kvDim, int headSize,
                                               int layerIndex, int contextLength, int dim) {
        int globalIdx = context.globalIdx;
        int halfDim   = dim / 2;
        int batchIdx  = globalIdx / halfDim;
        int pairIdx   = globalIdx % halfDim;
        int i         = pairIdx * 2;

        int pos     = batchStartPosHolder.get(0) + batchIdx;
        int qOffset = batchIdx * dim;
        int kOffset = batchIdx * kvDim;

        if (i + 1 < dim) {
            int head_dim = i % headSize;
            float freq = 1.0f / TornadoMath.pow(50000.0f, head_dim / (float) headSize);
            float val  = pos * freq;
            float fcr  = TornadoMath.cos(val);
            float fci  = TornadoMath.sin(val);

            // Rotate Q
            float v0q = wrapQBatch.get(qOffset + i);
            float v1q = wrapQBatch.get(qOffset + i + 1);
            wrapQBatch.set(qOffset + i,     v0q * fcr - v1q * fci);
            wrapQBatch.set(qOffset + i + 1, v0q * fci + v1q * fcr);

            // Rotate K and write K,V to cache
            if (i + 1 < kvDim) {
                float v0k    = wrapKBatch.get(kOffset + i);
                float v1k    = wrapKBatch.get(kOffset + i + 1);
                float rotK0  = v0k * fcr - v1k * fci;
                float rotK1  = v0k * fci + v1k * fcr;
                wrapKBatch.set(kOffset + i,     rotK0);
                wrapKBatch.set(kOffset + i + 1, rotK1);

                int cacheOff = layerIndex * contextLength * kvDim + pos * kvDim;
                wrapKeyCache.set(cacheOff + i,     rotK0);
                wrapKeyCache.set(cacheOff + i + 1, rotK1);
                wrapValueCache.set(cacheOff + i,     wrapVBatch.get(kOffset + i));
                wrapValueCache.set(cacheOff + i + 1, wrapVBatch.get(kOffset + i + 1));
            }
        }
    }

    // ── Attention ─────────────────────────────────────────────────────────────

    /**
     * Batched causal flash attention.
     *
     * <p>One workgroup per (batchIdx, headIdx) pair:
     * {@code groupIdx = batchIdx * nHeads + headIdx}.
     * Token b attends to positions 0..{@code startPos + b} (causal).</p>
     *
     * Worker: B*nHeads workgroups × optimalLocalSize threads.
     */
    public static void batchedFlashAttention(KernelContext context,
                                              IntArray batchStartPosHolder,
                                              FloatArray wrapQBatch,
                                              FloatArray wrapKeyCache,
                                              FloatArray wrapValueCache,
                                              FloatArray wrapXbBatch,
                                              int nHeads, int headSize,
                                              int kvDim, int kvMul,
                                              int layerIndex, int contextLength, int dim) {
        int tid      = context.localIdx;
        int groupId  = context.groupIdx;
        int localSz  = context.localGroupSizeX;

        int batchIdx  = groupId / nHeads;
        int h         = groupId % nHeads;
        int pos       = batchStartPosHolder.get(0) + batchIdx;
        int loff      = layerIndex * contextLength * kvDim;
        int kvHeadIdx = h / kvMul;
        int BLOCK_C   = 16;

        float[] qShared   = context.allocateFloatLocalArray(headSize);
        float[] kTile     = context.allocateFloatLocalArray(BLOCK_C * headSize);
        float[] vTile     = context.allocateFloatLocalArray(BLOCK_C * headSize);
        float[] sTile     = context.allocateFloatLocalArray(BLOCK_C);
        float[] maxHolder = context.allocateFloatLocalArray(1);

        // Load Q into shared memory
        int qOffset = batchIdx * dim + h * headSize;
        for (int i = tid; i < headSize; i += localSz) {
            qShared[i] = wrapQBatch.get(qOffset + i);
        }
        context.localBarrier();

        float maxScore = Float.NEGATIVE_INFINITY;
        float sumExp   = 0.0f;
        float[] output = new float[headSize];
        for (int i = 0; i < headSize; i++) output[i] = 0.0f;

        for (int tileC = 0; tileC <= pos; tileC += BLOCK_C) {
            int tileEnd = Math.min(tileC + BLOCK_C - 1, pos);

            // Load K/V tile
            for (int t = tileC + tid; t <= tileEnd; t += localSz) {
                int tInTile  = t - tileC;
                int tileMOff = tInTile * headSize;
                for (int d = 0; d < headSize; d++) {
                    int kvOff = loff + t * kvDim + kvHeadIdx * headSize + d;
                    kTile[tileMOff + d] = wrapKeyCache.get(kvOff);
                    vTile[tileMOff + d] = wrapValueCache.get(kvOff);
                }
            }
            context.localBarrier();

            // Compute attention scores
            for (int t = tileC + tid; t <= tileEnd; t += localSz) {
                int tInTile = t - tileC;
                float score = 0.0f;
                for (int d = 0; d < headSize; d++) {
                    score += qShared[d] * kTile[tInTile * headSize + d];
                }
                sTile[tInTile] = score / TornadoMath.sqrt(headSize);
            }
            context.localBarrier();

            // Tile max
            float tileMax = Float.NEGATIVE_INFINITY;
            for (int t = 0; t <= tileEnd - tileC; t++) {
                if (sTile[t] > tileMax) tileMax = sTile[t];
            }
            if (tid == 0) maxHolder[0] = tileMax;
            context.localBarrier();
            float curTileMax = maxHolder[0];

            float newMax = Math.max(maxScore, curTileMax);
            if (newMax != maxScore && maxScore != Float.NEGATIVE_INFINITY) {
                float scale = TornadoMath.exp(maxScore - newMax);
                sumExp *= scale;
                for (int d = 0; d < headSize; d++) output[d] *= scale;
            }
            maxScore = newMax;

            for (int t = 0; t <= tileEnd - tileC; t++) {
                float expScore = TornadoMath.exp(sTile[t] - maxScore);
                sumExp += expScore;
                for (int d = 0; d < headSize; d++) {
                    output[d] += expScore * vTile[t * headSize + d];
                }
            }
            context.localBarrier();
        }

        float norm = (sumExp > 0.0f) ? (1.0f / sumExp) : 0.0f;
        int xbOffset = batchIdx * dim + h * headSize;
        for (int d = tid; d < headSize; d += localSz) {
            wrapXbBatch.set(xbOffset + d, output[d] * norm);
        }
    }

    // ── Output / FFN Projections ─────────────────────────────────────────────

    /**
     * Batched matrix-vector multiply with residual add.
     *
     * <p>Used for both the attention output projection (Wo) and the FFN down
     * projection (W2). One workgroup per (batchIdx, outputRow):
     * {@code groupIdx = batchIdx * d + rowIdx}.</p>
     *
     * <ul>
     *   <li>Wo: inputBatch=xbBatch (B×dim), outputBatch=xBatch (B×dim), n=dim, d=dim</li>
     *   <li>W2: inputBatch=hbBatch (B×hiddenDim), outputBatch=xBatch (B×dim), n=hiddenDim, d=dim</li>
     * </ul>
     *
     * Worker: B*d workgroups × localWorkGroupSize threads.
     */
    public static void batchedMatVecWithResidual(KernelContext context,
                                                  FloatArray inputBatch,
                                                  FloatArray outputBatch,
                                                  HalfFloatArray w,
                                                  int n, int d,
                                                  int localWorkGroupSize) {
        int groupId  = context.groupIdx;
        int localId  = context.localIdx;
        int batchIdx = groupId / d;
        int rowIdx   = groupId % d;

        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);
        int inputOff = batchIdx * n;
        int rowOff   = rowIdx * n;

        float partial = 0.0f;
        for (int j = localId; j < n; j += localWorkGroupSize) {
            partial += w.get(rowOff + j).getFloat32() * inputBatch.get(inputOff + j);
        }
        localSum[localId] = partial;
        context.localBarrier();
        for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
            if (localId < s) localSum[localId] += localSum[localId + s];
            context.localBarrier();
        }
        if (localId == 0) {
            int outIdx = batchIdx * d + rowIdx;
            outputBatch.set(outIdx, outputBatch.get(outIdx) + localSum[0]);
        }
    }

    // ── FFN RMS Norm ─────────────────────────────────────────────────────────

    /**
     * Sequential FFN RMS reduction — one thread per batch item.
     * Worker: batchSize global threads, localSize=1.
     */
    public static void batchedFFNRmsReduce(KernelContext context,
                                            FloatArray wrapXBatch,
                                            FloatArray ffnScaleBatch,
                                            int dim, float eps) {
        int b    = context.globalIdx;
        int base = b * dim;
        float ss = 0.0f;
        for (int i = 0; i < dim; i++) {
            float val = wrapXBatch.get(base + i);
            ss += val * val;
        }
        ss /= dim;
        ss += eps;
        ffnScaleBatch.set(b, 1.0f / TornadoMath.sqrt(ss));
    }

    // ── FFN SwiGLU ───────────────────────────────────────────────────────────

    /**
     * Batched fused RMS-apply + W1/W3 gate-up projections + SiLU + GLU.
     *
     * <p>One workgroup per (batchIdx, hiddenRow):
     * {@code groupIdx = batchIdx * hiddenDim + rowIdx}.</p>
     *
     * Worker: B*hiddenDim workgroups × localWorkGroupSize threads.
     */
    public static void batchedFusedRmsNormFFNGateUp(KernelContext context,
                                                      FloatArray wrapXBatch,
                                                      FloatArray wrapHbBatch,
                                                      FloatArray rmsFFNWeights,
                                                      FloatArray ffnScaleBatch,
                                                      HalfFloatArray w1,
                                                      HalfFloatArray w3,
                                                      int dim, int hiddenDim,
                                                      int localWorkGroupSize) {
        int groupId  = context.groupIdx;
        int localId  = context.localIdx;
        int batchIdx = groupId / hiddenDim;
        int rowIdx   = groupId % hiddenDim;

        float scale    = ffnScaleBatch.get(batchIdx);
        int inputOff   = batchIdx * dim;
        int rowOff     = rowIdx * dim;

        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);

        // W1 matmul with inline RMS apply
        float sum1 = 0.0f;
        for (int j = localId; j < dim; j += localWorkGroupSize) {
            float normed = rmsFFNWeights.get(j) * scale * wrapXBatch.get(inputOff + j);
            sum1 += w1.get(rowOff + j).getFloat32() * normed;
        }
        localSum[localId] = sum1;
        context.localBarrier();
        for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
            if (localId < s) localSum[localId] += localSum[localId + s];
            context.localBarrier();
        }
        float result1 = localSum[0];

        // W3 matmul with inline RMS apply
        float sum3 = 0.0f;
        for (int j = localId; j < dim; j += localWorkGroupSize) {
            float normed = rmsFFNWeights.get(j) * scale * wrapXBatch.get(inputOff + j);
            sum3 += w3.get(rowOff + j).getFloat32() * normed;
        }
        localSum[localId] = sum3;
        context.localBarrier();
        for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
            if (localId < s) localSum[localId] += localSum[localId + s];
            context.localBarrier();
        }
        float result3 = localSum[0];

        // SiLU(W1·x) × (W3·x)
        if (localId == 0) {
            float silu = result1 / (1.0f + TornadoMath.exp(-result1));
            wrapHbBatch.set(batchIdx * hiddenDim + rowIdx, silu * result3);
        }
    }
}
