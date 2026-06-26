package org.beehive.gpullama3.tornadovm.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * GPU kernels for batched prefill.
 *
 * <p>Each kernel processes {@code batchSize} tokens simultaneously.
 * Batch tensors are flat: element [b][i] lives at index {@code b*stride + i}.
 * Worker-grid sizes are scaled by {@code batchSize} vs the single-token kernels.</p>
 *
 * <p>These kernels are meant to be registered in
 * {@link org.beehive.gpullama3.tornadovm.TornadoVMMasterPlanBatchPrefillDecode} TaskGraphs.</p>
 */
public final class TransformerBatchPrefillKernels {

    // @formatter:off
    private TransformerBatchPrefillKernels() {
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
        int rowIdx = groupId % totalRows;
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
                if (localId < s) {
                    localSum[localId] += localSum[localId + s];
                }
                context.localBarrier();
            }
            if (localId == 0) {
                wrapQBatch.set(batchIdx * dim + rowIdx, localSum[0]);
            }

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
                if (localId < s) {
                    localSum[localId] += localSum[localId + s];
                }
                context.localBarrier();
            }
            if (localId == 0) {
                wrapKBatch.set(batchIdx * kvDim + kRow, localSum[0]);
            }

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
                if (localId < s) {
                    localSum[localId] += localSum[localId + s];
                }
                context.localBarrier();
            }
            if (localId == 0) {
                wrapVBatch.set(batchIdx * kvDim + vRow, localSum[0]);
            }
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
        int halfDim = dim / 2;
        int batchIdx = globalIdx / halfDim;
        int pairIdx = globalIdx % halfDim;
        int i = pairIdx * 2;

        int pos = batchStartPosHolder.get(0) + batchIdx;
        int qOffset = batchIdx * dim;
        int kOffset = batchIdx * kvDim;

        if (i + 1 < dim) {
            int head_dim = i % headSize;
            float freq = 1.0f / TornadoMath.pow(50000.0f, head_dim / (float) headSize);
            float val = pos * freq;
            float fcr = TornadoMath.cos(val);
            float fci = TornadoMath.sin(val);

            // Rotate Q
            float v0q = wrapQBatch.get(qOffset + i);
            float v1q = wrapQBatch.get(qOffset + i + 1);
            wrapQBatch.set(qOffset + i, v0q * fcr - v1q * fci);
            wrapQBatch.set(qOffset + i + 1, v0q * fci + v1q * fcr);

            // Rotate K and write K,V to cache
            if (i + 1 < kvDim) {
                float v0k = wrapKBatch.get(kOffset + i);
                float v1k = wrapKBatch.get(kOffset + i + 1);
                float rotK0 = v0k * fcr - v1k * fci;
                float rotK1 = v0k * fci + v1k * fcr;
                wrapKBatch.set(kOffset + i, rotK0);
                wrapKBatch.set(kOffset + i + 1, rotK1);

                int cacheOff = layerIndex * contextLength * kvDim + pos * kvDim;
                wrapKeyCache.set(cacheOff + i, rotK0);
                wrapKeyCache.set(cacheOff + i + 1, rotK1);
                wrapValueCache.set(cacheOff + i, wrapVBatch.get(kOffset + i));
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
        int tid = context.localIdx;
        int groupId = context.groupIdx;
        int localSz = context.localGroupSizeX;

        int batchIdx = groupId / nHeads;
        int h = groupId % nHeads;
        int pos = batchStartPosHolder.get(0) + batchIdx;
        int loff = layerIndex * contextLength * kvDim;
        int kvHeadIdx = h / kvMul;
        int BLOCK_C = 16;

        float[] qShared = context.allocateFloatLocalArray(headSize);
        float[] kTile = context.allocateFloatLocalArray(BLOCK_C * headSize);
        float[] vTile = context.allocateFloatLocalArray(BLOCK_C * headSize);
        float[] sTile = context.allocateFloatLocalArray(BLOCK_C);
        float[] maxHolder = context.allocateFloatLocalArray(1);

        // Load Q into shared memory
        int qOffset = batchIdx * dim + h * headSize;
        for (int i = tid; i < headSize; i += localSz) {
            qShared[i] = wrapQBatch.get(qOffset + i);
        }
        context.localBarrier();

        float maxScore = Float.NEGATIVE_INFINITY;
        float sumExp = 0.0f;
        float[] output = new float[headSize];
        for (int i = 0; i < headSize; i++) {
            output[i] = 0.0f;
        }

        for (int tileC = 0; tileC <= pos; tileC += BLOCK_C) {
            int tileEnd = Math.min(tileC + BLOCK_C - 1, pos);

            // Load K/V tile
            for (int t = tileC + tid; t <= tileEnd; t += localSz) {
                int tInTile = t - tileC;
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
                if (sTile[t] > tileMax) {
                    tileMax = sTile[t];
                }
            }
            if (tid == 0) {
                maxHolder[0] = tileMax;
            }
            context.localBarrier();
            float curTileMax = maxHolder[0];

            float newMax = Math.max(maxScore, curTileMax);
            if (newMax != maxScore && maxScore != Float.NEGATIVE_INFINITY) {
                float scale = TornadoMath.exp(maxScore - newMax);
                sumExp *= scale;
                for (int d = 0; d < headSize; d++) {
                    output[d] *= scale;
                }
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
        int rowIdx = groupId % d;

        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);
        int inputOff = batchIdx * n;
        int rowOff = rowIdx * n;

        float partial = 0.0f;
        for (int j = localId; j < n; j += localWorkGroupSize) {
            partial += w.get(rowOff + j).getFloat32() * inputBatch.get(inputOff + j);
        }
        localSum[localId] = partial;
        context.localBarrier();
        for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
            if (localId < s) {
                localSum[localId] += localSum[localId + s];
            }
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
        int b = context.globalIdx;
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
        int groupId = context.groupIdx;
        int localId = context.localIdx;
        int batchIdx = groupId / hiddenDim;
        int rowIdx = groupId % hiddenDim;

        float scale = ffnScaleBatch.get(batchIdx);
        int inputOff = batchIdx * dim;
        int rowOff = rowIdx * dim;

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
            if (localId < s) {
                localSum[localId] += localSum[localId + s];
            }
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
            if (localId < s) {
                localSum[localId] += localSum[localId + s];
            }
            context.localBarrier();
        }
        float result3 = localSum[0];

        // SiLU(W1·x) × (W3·x)
        if (localId == 0) {
            float silu = result1 / (1.0f + TornadoMath.exp(-result1));
            wrapHbBatch.set(batchIdx * hiddenDim + rowIdx, silu * result3);
        }
    }

    // ── Q8_0 Batch Kernels ───────────────────────────────────────────────────

    /**
     * No-op kernel for Q8_0 batch activation graph.
     * The host fills wrapXBatch with dequantized FP32 embeddings before execution.
     * Worker: 1 global thread.
     */
    public static void batchPassthrough(KernelContext context, FloatArray wrapXBatch) {
        if (context.globalIdx == 0) {
            wrapXBatch.set(0, wrapXBatch.get(0));
        }
    }

    /**
     * Applies RMS normalization to FP32 — Q8_0 variant.
     * Writes normalized FP32 to wrapXbBatch (reused as xb intermediate before QKV).
     * Worker: B*dim global threads, localSize=256.
     */
    public static void batchedRmsApplyFP32(KernelContext context,
                                            FloatArray wrapXbBatch,
                                            FloatArray wrapXBatch,
                                            FloatArray rmsWeights,
                                            FloatArray attnScaleBatch,
                                            int dim) {
        int gid = context.globalIdx;
        int b = gid / dim;
        int i = gid % dim;
        wrapXbBatch.set(gid, rmsWeights.get(i) * attnScaleBatch.get(b) * wrapXBatch.get(gid));
    }

    /**
     * Fused batched QKV projection with Q8_0 weight dequantization.
     * Input wrapXbBatch is FP32 (written by batchedRmsApplyFP32).
     * groupIdx = batchIdx * (dim + 2*kvDim) + rowIdx.
     * Worker: B*(dim+2*kvDim) workgroups × localWorkGroupSize threads.
     */
    public static void batchedFusedQKVMatmulQ8(KernelContext context,
                                                FloatArray wrapXbBatch,
                                                FloatArray wrapQBatch,
                                                FloatArray wrapKBatch,
                                                FloatArray wrapVBatch,
                                                ByteArray wq,
                                                ByteArray wk,
                                                ByteArray wv,
                                                int dim, int kvDim,
                                                int localWorkGroupSize) {
        int groupId = context.groupIdx;
        int localId = context.localIdx;
        int totalRows = dim + 2 * kvDim;
        int batchIdx = groupId / totalRows;
        int rowIdx = groupId % totalRows;
        int inputOff = batchIdx * dim;

        int blockSize = 32;
        int Q8_0_BLOCK_BYTES = 34;
        int blocksPerRow = (dim + blockSize - 1) / blockSize;

        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);

        if (rowIdx < dim) {
            int rowBlockOffset = rowIdx * blocksPerRow;
            float partial = 0.0f;
            for (int j = localId; j < dim; j += localWorkGroupSize) {
                int blockByteOffset = (rowBlockOffset + j / blockSize) * Q8_0_BLOCK_BYTES;
                float scale = wq.getHalfFloat(blockByteOffset).getFloat32();
                float quant = wq.get(blockByteOffset + 2 + j % blockSize);
                partial += quant * scale * wrapXbBatch.get(inputOff + j);
            }
            localSum[localId] = partial;
            context.localBarrier();
            for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
                if (localId < s) {
                    localSum[localId] += localSum[localId + s];
                }
                context.localBarrier();
            }
            if (localId == 0) {
                wrapQBatch.set(batchIdx * dim + rowIdx, localSum[0]);
            }

        } else if (rowIdx < dim + kvDim) {
            int kRow = rowIdx - dim;
            int rowBlockOffset = kRow * blocksPerRow;
            float partial = 0.0f;
            for (int j = localId; j < dim; j += localWorkGroupSize) {
                int blockByteOffset = (rowBlockOffset + j / blockSize) * Q8_0_BLOCK_BYTES;
                float scale = wk.getHalfFloat(blockByteOffset).getFloat32();
                float quant = wk.get(blockByteOffset + 2 + j % blockSize);
                partial += quant * scale * wrapXbBatch.get(inputOff + j);
            }
            localSum[localId] = partial;
            context.localBarrier();
            for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
                if (localId < s) {
                    localSum[localId] += localSum[localId + s];
                }
                context.localBarrier();
            }
            if (localId == 0) {
                wrapKBatch.set(batchIdx * kvDim + kRow, localSum[0]);
            }

        } else {
            int vRow = rowIdx - dim - kvDim;
            int rowBlockOffset = vRow * blocksPerRow;
            float partial = 0.0f;
            for (int j = localId; j < dim; j += localWorkGroupSize) {
                int blockByteOffset = (rowBlockOffset + j / blockSize) * Q8_0_BLOCK_BYTES;
                float scale = wv.getHalfFloat(blockByteOffset).getFloat32();
                float quant = wv.get(blockByteOffset + 2 + j % blockSize);
                partial += quant * scale * wrapXbBatch.get(inputOff + j);
            }
            localSum[localId] = partial;
            context.localBarrier();
            for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
                if (localId < s) {
                    localSum[localId] += localSum[localId + s];
                }
                context.localBarrier();
            }
            if (localId == 0) {
                wrapVBatch.set(batchIdx * kvDim + vRow, localSum[0]);
            }
        }
    }

    /**
     * Batched matrix-vector multiply with residual add (Q8_0 weights).
     * Used for attention output (Wo) and FFN down (W2) projections.
     * groupIdx = batchIdx * d + rowIdx.
     * Worker: B*d workgroups × localWorkGroupSize threads.
     */
    public static void batchedMatVecWithResidualQ8(KernelContext context,
                                                    FloatArray inputBatch,
                                                    FloatArray outputBatch,
                                                    ByteArray w,
                                                    int n, int d,
                                                    int localWorkGroupSize) {
        int groupId = context.groupIdx;
        int localId = context.localIdx;
        int batchIdx = groupId / d;
        int rowIdx = groupId % d;

        int blockSize = 32;
        int Q8_0_BLOCK_BYTES = 34;
        int blocksPerRow = (n + blockSize - 1) / blockSize;
        int rowBlockOffset = rowIdx * blocksPerRow;
        int inputOff = batchIdx * n;

        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);

        float partial = 0.0f;
        for (int j = localId; j < n; j += localWorkGroupSize) {
            int blockByteOffset = (rowBlockOffset + j / blockSize) * Q8_0_BLOCK_BYTES;
            float scale = w.getHalfFloat(blockByteOffset).getFloat32();
            float quant = w.get(blockByteOffset + 2 + j % blockSize);
            partial += quant * scale * inputBatch.get(inputOff + j);
        }
        localSum[localId] = partial;
        context.localBarrier();
        for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
            if (localId < s) {
                localSum[localId] += localSum[localId + s];
            }
            context.localBarrier();
        }
        if (localId == 0) {
            int outIdx = batchIdx * d + rowIdx;
            outputBatch.set(outIdx, outputBatch.get(outIdx) + localSum[0]);
        }
    }

    /**
     * Batched fused RMS-apply + W1/W3 gate-up projections + SiLU + GLU (Q8_0 weights).
     * groupIdx = batchIdx * hiddenDim + rowIdx.
     * Worker: B*hiddenDim workgroups × localWorkGroupSize threads.
     */
    public static void batchedFusedRmsNormFFNGateUpQ8(KernelContext context,
                                                        FloatArray wrapXBatch,
                                                        FloatArray wrapHbBatch,
                                                        FloatArray rmsFFNWeights,
                                                        FloatArray ffnScaleBatch,
                                                        ByteArray w1,
                                                        ByteArray w3,
                                                        int dim, int hiddenDim,
                                                        int localWorkGroupSize) {
        int groupId = context.groupIdx;
        int localId = context.localIdx;
        int batchIdx = groupId / hiddenDim;
        int rowIdx = groupId % hiddenDim;

        float scale = ffnScaleBatch.get(batchIdx);
        int inputOff = batchIdx * dim;

        int blockSize = 32;
        int Q8_0_BLOCK_BYTES = 34;
        int blocksPerRow = (dim + blockSize - 1) / blockSize;
        int rowBlockOffset = rowIdx * blocksPerRow;

        float[] localSum = context.allocateFloatLocalArray(localWorkGroupSize);

        float sum1 = 0.0f;
        for (int j = localId; j < dim; j += localWorkGroupSize) {
            int blockByteOffset = (rowBlockOffset + j / blockSize) * Q8_0_BLOCK_BYTES;
            float w1Scale = w1.getHalfFloat(blockByteOffset).getFloat32();
            float w1Quant = w1.get(blockByteOffset + 2 + j % blockSize);
            float normed = rmsFFNWeights.get(j) * scale * wrapXBatch.get(inputOff + j);
            sum1 += w1Quant * w1Scale * normed;
        }
        localSum[localId] = sum1;
        context.localBarrier();
        for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
            if (localId < s) {
                localSum[localId] += localSum[localId + s];
            }
            context.localBarrier();
        }
        float result1 = localSum[0];

        float sum3 = 0.0f;
        for (int j = localId; j < dim; j += localWorkGroupSize) {
            int blockByteOffset = (rowBlockOffset + j / blockSize) * Q8_0_BLOCK_BYTES;
            float w3Scale = w3.getHalfFloat(blockByteOffset).getFloat32();
            float w3Quant = w3.get(blockByteOffset + 2 + j % blockSize);
            float normed = rmsFFNWeights.get(j) * scale * wrapXBatch.get(inputOff + j);
            sum3 += w3Quant * w3Scale * normed;
        }
        localSum[localId] = sum3;
        context.localBarrier();
        for (int s = localWorkGroupSize / 2; s > 0; s >>= 1) {
            if (localId < s) {
                localSum[localId] += localSum[localId + s];
            }
            context.localBarrier();
        }
        float result3 = localSum[0];

        if (localId == 0) {
            float silu = result1 / (1.0f + TornadoMath.exp(-result1));
            wrapHbBatch.set(batchIdx * hiddenDim + rowIdx, silu * result3);
        }
    }

    /**
     * RMS-apply for FFN, writing FP16. Mirrors batchedRmsApplyFP16 but pulls
     * scale from ffnScaleBatch. Output is the A operand for the W1/W3 MMA tasks.
     *
     * Worker: B*dim global threads, localSize=256.
     */
    public static void batchedFFNRmsApplyFP16(KernelContext context,
                                              HalfFloatArray normedXFFNFP16,
                                              FloatArray wrapXBatch,
                                              FloatArray rmsFFNWeights,
                                              FloatArray ffnScaleBatch,
                                              int dim) {
        int gid = context.globalIdx;
        int b = gid / dim;
        int i = gid % dim;
        float scale = ffnScaleBatch.get(b);
        float result = rmsFFNWeights.get(i) * scale * wrapXBatch.get(gid);
        normedXFFNFP16.set(gid, new HalfFloat(result));
    }

    /**
     * Fused SiLU(gate) * up after the two FFN matmuls.
     * Operates on FP32 inputs (MMA writes FP32).
     *
     * Worker: B*hiddenDim global threads, localSize=256.
     */
    public static void batchedFFNSwiGLU(KernelContext context,
                                        FloatArray wrapHbBatch,
                                        FloatArray ffnGateResult,
                                        FloatArray ffnUpResult,
                                        int hiddenDim) {
        int gid = context.globalIdx;
        float g = ffnGateResult.get(gid);
        float u = ffnUpResult.get(gid);
        float silu = g / (1.0f + TornadoMath.exp(-g));
        wrapHbBatch.set(gid, silu * u);
    }

    private static final int WARP_SIZE = 32;
    private static final int BM = 128, BN = 128, BK = 16;
    private static final int WARPS_M = 4, WARPS_N = 2;
    private static final int WARPS_PER_BLOCK = WARPS_M * WARPS_N;
    private static final int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    private static final int WM = BM / WARPS_M;
    private static final int WN = BN / WARPS_N;
    private static final int B_SUBTILE_BYTES = 256;

    public static void gemmMMA(KernelContext ctx,
                                  HalfFloatArray A, HalfFloatArray B, FloatArray C,
                                  int M, int N, int K) {
        int tid = ctx.localIdx;
        int warpId = tid / WARP_SIZE;
        int warpM = warpId / WARPS_N;
        int warpN = warpId % WARPS_N;
        int blockRow = BM * ctx.groupIdx;
        int blockCol = BN * ctx.groupIdy;

        int[] aTile = ctx.allocateIntLocalArray(BM * BK / 2);
        int[] bTile = ctx.allocateIntLocalArray(BK * BN / 2);

        float[] c00 = ctx.mmaFragment(0.0f); float[] c01 = ctx.mmaFragment(0.0f);
        float[] c02 = ctx.mmaFragment(0.0f); float[] c03 = ctx.mmaFragment(0.0f);
        float[] c04 = ctx.mmaFragment(0.0f); float[] c05 = ctx.mmaFragment(0.0f);
        float[] c06 = ctx.mmaFragment(0.0f); float[] c07 = ctx.mmaFragment(0.0f);
        float[] c10 = ctx.mmaFragment(0.0f); float[] c11 = ctx.mmaFragment(0.0f);
        float[] c12 = ctx.mmaFragment(0.0f); float[] c13 = ctx.mmaFragment(0.0f);
        float[] c14 = ctx.mmaFragment(0.0f); float[] c15 = ctx.mmaFragment(0.0f);
        float[] c16 = ctx.mmaFragment(0.0f); float[] c17 = ctx.mmaFragment(0.0f);

        int numKSteps = K / BK;
        for (int kStep = 0; kStep < numKSteps; kStep++) {
            int kBase = kStep * BK;
            // A load: A is [M, K] row-major (unchanged from gemmMMA)
            for (int idx = tid; idx < BM * BK / 2; idx += THREADS_PER_BLOCK) {
                int m_row = idx / (BK / 2);
                int k_pair = idx % (BK / 2);
                int k_base = k_pair * 2;
                int gA = (blockRow + m_row) * K + (kBase + k_base);
                int lo = A.get(gA).getHalfFloatValue() & 0xFFFF;
                int hi = A.get(gA + 1).getHalfFloatValue() & 0xFFFF;
                aTile[m_row * (BK / 2) + k_pair] = lo | (hi << 16);
            }
            // B load: B is [N, K] row-major.
            // The shared-memory layout in bTile is unchanged; we read from
            // a different global layout and pack into the same shared positions
            // so that mmaLoadB, mma fragments, and stores all operate identically.
            for (int idx = tid; idx < BK * BN / 2; idx += THREADS_PER_BLOCK) {
                int subTileId = idx / 64;
                int intInSub = idx % 64;
                int k_row = intInSub / 4;
                int j_pair = intInSub % 4;
                int j_base = j_pair * 2;
                int col_in_block = subTileId * 8 + j_base;
                // B[col, k] is at col * K + k in [N, K] row-major.
                int gB0 = (blockCol + col_in_block)     * K + (kBase + k_row);
                int gB1 = (blockCol + col_in_block + 1) * K + (kBase + k_row);
                int lo = B.get(gB0).getHalfFloatValue() & 0xFFFF;
                int hi = B.get(gB1).getHalfFloatValue() & 0xFFFF;
                bTile[idx] = lo | (hi << 16);
            }
            ctx.localBarrier();

            int aOff0 = warpM * 1024;
            int aOff1 = warpM * 1024 + 512;
            HalfFloat[] a0 = ctx.mmaLoadA(aTile, BK, aOff0);
            HalfFloat[] a1 = ctx.mmaLoadA(aTile, BK, aOff1);
            int bBase = warpN * 8;
            HalfFloat[] b0 = ctx.mmaLoadB(bTile, BK, (bBase + 0) * B_SUBTILE_BYTES);
            HalfFloat[] b1 = ctx.mmaLoadB(bTile, BK, (bBase + 1) * B_SUBTILE_BYTES);
            HalfFloat[] b2 = ctx.mmaLoadB(bTile, BK, (bBase + 2) * B_SUBTILE_BYTES);
            HalfFloat[] b3 = ctx.mmaLoadB(bTile, BK, (bBase + 3) * B_SUBTILE_BYTES);
            HalfFloat[] b4 = ctx.mmaLoadB(bTile, BK, (bBase + 4) * B_SUBTILE_BYTES);
            HalfFloat[] b5 = ctx.mmaLoadB(bTile, BK, (bBase + 5) * B_SUBTILE_BYTES);
            HalfFloat[] b6 = ctx.mmaLoadB(bTile, BK, (bBase + 6) * B_SUBTILE_BYTES);
            HalfFloat[] b7 = ctx.mmaLoadB(bTile, BK, (bBase + 7) * B_SUBTILE_BYTES);

            c00 = ctx.mma(a0, b0, c00, MMAShape.M16N8K16);
            c01 = ctx.mma(a0, b1, c01, MMAShape.M16N8K16);
            c02 = ctx.mma(a0, b2, c02, MMAShape.M16N8K16);
            c03 = ctx.mma(a0, b3, c03, MMAShape.M16N8K16);
            c04 = ctx.mma(a0, b4, c04, MMAShape.M16N8K16);
            c05 = ctx.mma(a0, b5, c05, MMAShape.M16N8K16);
            c06 = ctx.mma(a0, b6, c06, MMAShape.M16N8K16);
            c07 = ctx.mma(a0, b7, c07, MMAShape.M16N8K16);
            c10 = ctx.mma(a1, b0, c10, MMAShape.M16N8K16);
            c11 = ctx.mma(a1, b1, c11, MMAShape.M16N8K16);
            c12 = ctx.mma(a1, b2, c12, MMAShape.M16N8K16);
            c13 = ctx.mma(a1, b3, c13, MMAShape.M16N8K16);
            c14 = ctx.mma(a1, b4, c14, MMAShape.M16N8K16);
            c15 = ctx.mma(a1, b5, c15, MMAShape.M16N8K16);
            c16 = ctx.mma(a1, b6, c16, MMAShape.M16N8K16);
            c17 = ctx.mma(a1, b7, c17, MMAShape.M16N8K16);
            ctx.localBarrier();
        }

        int rBase = blockRow + warpM * WM;
        int cBase = blockCol + warpN * WN;
        ctx.mmaStore(c00, C, rBase + 0,  cBase + 0,  N);
        ctx.mmaStore(c01, C, rBase + 0,  cBase + 8,  N);
        ctx.mmaStore(c02, C, rBase + 0,  cBase + 16, N);
        ctx.mmaStore(c03, C, rBase + 0,  cBase + 24, N);
        ctx.mmaStore(c04, C, rBase + 0,  cBase + 32, N);
        ctx.mmaStore(c05, C, rBase + 0,  cBase + 40, N);
        ctx.mmaStore(c06, C, rBase + 0,  cBase + 48, N);
        ctx.mmaStore(c07, C, rBase + 0,  cBase + 56, N);
        ctx.mmaStore(c10, C, rBase + 16, cBase + 0,  N);
        ctx.mmaStore(c11, C, rBase + 16, cBase + 8,  N);
        ctx.mmaStore(c12, C, rBase + 16, cBase + 16, N);
        ctx.mmaStore(c13, C, rBase + 16, cBase + 24, N);
        ctx.mmaStore(c14, C, rBase + 16, cBase + 32, N);
        ctx.mmaStore(c15, C, rBase + 16, cBase + 40, N);
        ctx.mmaStore(c16, C, rBase + 16, cBase + 48, N);
        ctx.mmaStore(c17, C, rBase + 16, cBase + 56, N);
    }

    // ── Residual add (FP32) ───────────────────────────────────────────────
// gemmMMA overwrites C, but Wo and W2 both need x = x + W·a.
// Worker: B*dim global threads (valid rows only), localSize=256.
    public static void batchedResidualAddFP32(KernelContext context,
                                              FloatArray residual,   // x (in/out)
                                              FloatArray delta) {     // GEMM output
        int gid = context.globalIdx;
        residual.set(gid, residual.get(gid) + delta.get(gid));
    }

    // ── SwiGLU emitting FP16 ──────────────────────────────────────────────
// Replaces batchedFFNSwiGLU. Output is the A operand for the W2 GEMM.
// Worker: B*hiddenDim global threads, localSize=256.
    public static void batchedFFNSwiGLUFP16(KernelContext context,
                                            HalfFloatArray wrapHbFP16Batch,
                                            FloatArray ffnGateResult,
                                            FloatArray ffnUpResult,
                                            int hiddenDim) {
        int gid = context.globalIdx;
        float g = ffnGateResult.get(gid);
        float u = ffnUpResult.get(gid);
        float silu = g / (1.0f + TornadoMath.exp(-g));
        wrapHbFP16Batch.set(gid, new HalfFloat(silu * u));
    }

    // ── FP32 → FP16 cast (Option B only, see Wo below) ────────────────────
// Worker: B*dim global threads, localSize=256.
    public static void batchedConvertFP32toFP16(KernelContext context,
                                                FloatArray in,
                                                HalfFloatArray out) {
        int gid = context.globalIdx;
        out.set(gid, new HalfFloat(in.get(gid)));
    }

    // @formatter:on
}
