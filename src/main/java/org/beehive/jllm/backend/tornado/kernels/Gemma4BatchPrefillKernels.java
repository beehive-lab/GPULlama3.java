package org.beehive.jllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.HalfFloat;

// @formatter:off
/**
 * The batched-prefill kernels this family does not share with any other.
 *
 * <p>Every kernel here is the chunk-wide form of one {@link Gemma4Kernels} kernel: the same
 * arithmetic in the same order, with a row index added. A row is one prompt token; {@code
 * startPosHolder} carries the chunk's first sequence position at index 0 and the number of
 * <i>real</i> rows at index 1, because the grids launch a padded row count and the padding rows
 * must not rotate, must not write a key or a value, and must not address past the end of a layer's
 * KV slice.
 *
 * <p>They are a separate class rather than additions to {@link TransformerBatchPrefillKernels}
 * because the differences are this architecture's and not parameters: sandwich norms that normalize
 * a branch output and add it onto the residual, a per-head norm on the query, the key <i>and</i>
 * the value, a GeGLU where every other family has SwiGLU, an attention scale of exactly one, a
 * sliding window on four layers out of five, and the per-layer embedding block that has no
 * counterpart anywhere else in the repository. The generic kernels that <i>do</i> fit — the RMS
 * reductions, the FP32→FP16 cast and the whole {@code gemmMMA} family — are called from the layer
 * graph unchanged, not copied here.
 */
// @formatter:on
public final class Gemma4BatchPrefillKernels {

    private Gemma4BatchPrefillKernels() {}

    // ── Norms ────────────────────────────────────────────────────────────────

    // @formatter:off
    /**
     * Sandwich norm with residual, chunk-wide: {@code x[b,i] += weight[i] * (scale[b] * delta[b,i])}.
     *
     * <p>The row-wise form of {@link Gemma4Kernels#rmsNormApplyWithResidual}, and the reason this
     * family cannot use {@code batchedRmsReduceFusedResidual}: the scale is the RMS of the branch
     * output {@code delta}, not of the residual {@code x} it is added to, so the reduce runs over a
     * different buffer than the add updates and the two cannot be one pass.
     *
     * <p>Worker: {@code B*size} threads, local 256.
     */
    // @formatter:on
    public static void batchedRmsApplyWithResidual(
            KernelContext context,
            FloatArray x,
            FloatArray delta,
            FloatArray weight,
            FloatArray scaleBatch,
            int size) {
        int gid = context.globalIdx;
        int b = gid / size;
        int i = gid - b * size;
        float scale = scaleBatch.get(b);
        x.set(gid, x.get(gid) + weight.get(i) * (scale * delta.get(gid)));
    }

    // @formatter:off
    /**
     * The query, key and value per-head norms over the packed {@code [q|k|v]} projection output,
     * in one launch.
     *
     * <p>One workgroup per (row, head slot), where the slots run {@code q} heads, then {@code k}
     * heads, then {@code v} heads — which is exactly the packed row's own order, so a slot's base
     * is {@code slot * headDim} from the row start and no per-kind branch on the address is needed.
     * Only the weight differs by kind, and Gemma 4 normalizes V without one; that is the whole
     * reason this is a single kernel with a three-way select rather than three launches.
     *
     * <p>Worker: {@code B*(nHeads + 2*nHeadKv)} workgroups of {@code localMemSize} lanes.
     */
    // @formatter:on
    public static void batchedQkvHeadNorms(
            KernelContext context,
            FloatArray qkv,
            FloatArray qNormWeight,
            FloatArray kNormWeight,
            int nHeads,
            int nHeadKv,
            int headDim,
            int qkvStride,
            int localMemSize,
            float rmsNormEps) {
        int group = context.groupIdx;
        int localId = context.localIdx;
        int localSize = context.localGroupSizeX;

        int slotsPerRow = nHeads + 2 * nHeadKv;
        int b = group / slotsPerRow;
        int slot = group - b * slotsPerRow;
        int base = b * qkvStride + slot * headDim;

        float[] localSum = context.allocateFloatLocalArray(localMemSize);
        float partial = 0.0f;
        for (int i = localId; i < headDim; i += localSize) {
            float v = qkv.get(base + i);
            partial += v * v;
        }
        localSum[localId] = partial;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }
        float ss = localSum[0] / headDim + rmsNormEps;
        ss = 1.0f / TornadoMath.sqrt(ss);
        context.localBarrier();

        if (slot < nHeads) {
            for (int i = localId; i < headDim; i += localSize) {
                qkv.set(base + i, qNormWeight.get(i) * (ss * qkv.get(base + i)));
            }
        } else if (slot < nHeads + nHeadKv) {
            for (int i = localId; i < headDim; i += localSize) {
                qkv.set(base + i, kNormWeight.get(i) * (ss * qkv.get(base + i)));
            }
        } else {
            for (int i = localId; i < headDim; i += localSize) {
                qkv.set(base + i, ss * qkv.get(base + i));
            }
        }
    }

    // @formatter:off
    /**
     * The query-only per-head norm, for the twenty layers that reuse an earlier layer's KV and
     * therefore project no key and no value.
     *
     * <p>Worker: {@code B*nHeads} workgroups of {@code localMemSize} lanes.
     */
    // @formatter:on
    public static void batchedQHeadNorm(
            KernelContext context,
            FloatArray qkv,
            FloatArray qNormWeight,
            int nHeads,
            int headDim,
            int qkvStride,
            int localMemSize,
            float rmsNormEps) {
        int group = context.groupIdx;
        int localId = context.localIdx;
        int localSize = context.localGroupSizeX;

        int b = group / nHeads;
        int h = group - b * nHeads;
        int base = b * qkvStride + h * headDim;

        float[] localSum = context.allocateFloatLocalArray(localMemSize);
        float partial = 0.0f;
        for (int i = localId; i < headDim; i += localSize) {
            float v = qkv.get(base + i);
            partial += v * v;
        }
        localSum[localId] = partial;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }
        float ss = localSum[0] / headDim + rmsNormEps;
        ss = 1.0f / TornadoMath.sqrt(ss);
        context.localBarrier();
        for (int i = localId; i < headDim; i += localSize) {
            qkv.set(base + i, qNormWeight.get(i) * (ss * qkv.get(base + i)));
        }
    }

    // ── RoPE and the KV cache ────────────────────────────────────────────────

    // @formatter:off
    /**
     * NeoX RoPE over the packed {@code [q|k|v]} rows, fused with the key/value cache write.
     *
     * <p>The chunk-wide form of {@link Gemma4Kernels#ropeNeoxRotateAndCacheCopy}, reading its angle
     * from the same precomputed tables at {@code startPos + b} rather than at a single position.
     * The cache is this family's flat one addressed by {@code cacheBaseOffset}, not a paged one: the
     * decode graphs that run after prefill read it that way, and a prefill that wrote pages would
     * leave them reading a cache nobody filled.
     *
     * <p>Padding rows return before writing anything. A padded row's position is past the chunk and
     * would be a valid index into the next layer's KV slice, so the guard is what keeps a padded
     * launch from corrupting a neighbouring layer rather than merely wasting work.
     *
     * <p>Worker: {@code B*nHeads*(headDim/2)} threads.
     */
    // @formatter:on
    public static void batchedRopeAndCache(
            KernelContext context,
            IntArray startPosHolder,
            FloatArray qkv,
            FloatArray keyCache,
            FloatArray valueCache,
            FloatArray freqCisReal,
            FloatArray freqCisImag,
            int nHeads,
            int nHeadKv,
            int headDim,
            int kvDim,
            int qkvStride,
            int cacheBaseOffset) {
        int gid = context.globalIdx;
        int half = headDim / 2;
        int perRow = nHeads * half;
        int b = gid / perRow;
        int rem = gid - b * perRow;
        int h = rem / half;
        int ic = rem - h * half;

        if (b >= startPosHolder.get(1)) {
            return;
        }
        int pos = startPosHolder.get(0) + b;
        float fcr = freqCisReal.get(pos * half + ic);
        float fci = freqCisImag.get(pos * half + ic);

        int rowBase = b * qkvStride;
        int qBase = rowBase + h * headDim;
        float v0q = qkv.get(qBase + ic);
        float v1q = qkv.get(qBase + ic + half);
        qkv.set(qBase + ic, v0q * fcr - v1q * fci);
        qkv.set(qBase + ic + half, v0q * fci + v1q * fcr);

        if (h < nHeadKv) {
            int kBase = rowBase + nHeads * headDim + h * headDim;
            int vBase = kBase + nHeadKv * headDim;
            float v0k = qkv.get(kBase + ic);
            float v1k = qkv.get(kBase + ic + half);
            float rotatedK0 = v0k * fcr - v1k * fci;
            float rotatedK1 = v0k * fci + v1k * fcr;

            int cacheOffset = cacheBaseOffset + pos * kvDim + h * headDim;
            keyCache.set(cacheOffset + ic, rotatedK0);
            keyCache.set(cacheOffset + ic + half, rotatedK1);
            valueCache.set(cacheOffset + ic, qkv.get(vBase + ic));
            valueCache.set(cacheOffset + ic + half, qkv.get(vBase + ic + half));
        }
    }

    // @formatter:off
    /**
     * NeoX RoPE on the query alone, for the layers that reuse an earlier layer's KV cache.
     *
     * <p>Worker: {@code B*nHeads*(headDim/2)} threads.
     */
    // @formatter:on
    public static void batchedRopeQOnly(
            KernelContext context,
            IntArray startPosHolder,
            FloatArray qkv,
            FloatArray freqCisReal,
            FloatArray freqCisImag,
            int nHeads,
            int headDim,
            int qkvStride) {
        int gid = context.globalIdx;
        int half = headDim / 2;
        int perRow = nHeads * half;
        int b = gid / perRow;
        int rem = gid - b * perRow;
        int h = rem / half;
        int ic = rem - h * half;

        if (b >= startPosHolder.get(1)) {
            return;
        }
        int pos = startPosHolder.get(0) + b;
        float fcr = freqCisReal.get(pos * half + ic);
        float fci = freqCisImag.get(pos * half + ic);

        int qBase = b * qkvStride + h * headDim;
        float v0 = qkv.get(qBase + ic);
        float v1 = qkv.get(qBase + ic + half);
        qkv.set(qBase + ic, v0 * fcr - v1 * fci);
        qkv.set(qBase + ic + half, v0 * fci + v1 * fcr);
    }

    // ── Attention ────────────────────────────────────────────────────────────

    // @formatter:off
    /**
     * Causal attention over a (possibly sliding) window, one workgroup per (row, head), with the
     * result emitted in FP16 for the output projection's tensor-core GEMM.
     *
     * <p>The chunk-wide form of {@link Gemma4Kernels#attentionWithSlidingWindowParallel}: three
     * phases, the score dot product and the weighted sum each serial and in dimension order, only
     * the softmax maximum and the sum of exponentials as trees. Row {@code b} attends positions
     * {@code [max(0, pos-windowSize+1), pos]} of the shared sequence cache with {@code pos =
     * startPos + b}, which is what makes the chunk causal without a mask: a row simply never reads
     * past its own position. Gemma 4's attention scale is one, so no score is divided.
     *
     * <p>The scores go to a global scratch rather than to shared memory because the window is up to
     * the whole context and a workgroup's shared memory is not. Each (row, head) owns {@code
     * scoreStride} floats of it and writes only inside its own slice.
     *
     * <p>Padding rows write zeros into their output and return: those rows still pass through the
     * output projection, and an uninitialized FP16 row would put NaNs into a GEMM that shares no
     * lanes with the real rows but does share its warps' execution.
     *
     * <p>Worker: {@code B*nHeads} workgroups of {@code localMemSize} lanes.
     */
    // @formatter:on
    public static void batchedSlidingWindowAttention(
            KernelContext context,
            IntArray startPosHolder,
            FloatArray qkv,
            FloatArray keyCache,
            FloatArray valueCache,
            HalfFloatArray out,
            FloatArray scores,
            int nHeads,
            int headDim,
            int kvDim,
            int kvMul,
            int qkvStride,
            int cacheBaseOffset,
            int windowSize,
            int scoreStride,
            int localMemSize) {
        int tid = context.localIdx;
        int group = context.groupIdx;
        int localSize = context.localGroupSizeX;

        int b = group / nHeads;
        int h = group - b * nHeads;
        int outBase = b * (nHeads * headDim) + h * headDim;

        if (b >= startPosHolder.get(1)) {
            for (int i = tid; i < headDim; i += localSize) {
                out.set(outBase + i, new HalfFloat(0.0f));
            }
            return;
        }

        int pos = startPosHolder.get(0) + b;
        int windowStart = Math.max(0, pos - windowSize + 1);
        int scoreBase = group * scoreStride;
        int kvHeadIdx = h / kvMul;
        int qOffset = b * qkvStride + h * headDim;

        float[] qShared = context.allocateFloatLocalArray(headDim);
        float[] reduce = context.allocateFloatLocalArray(localMemSize);

        for (int i = tid; i < headDim; i += localSize) {
            qShared[i] = qkv.get(qOffset + i);
        }
        context.localBarrier();

        for (int t = windowStart + tid; t <= pos; t += localSize) {
            int keyOffset = cacheBaseOffset + t * kvDim + kvHeadIdx * headDim;
            float score = 0.0f;
            for (int i = 0; i < headDim; i++) {
                score += qShared[i] * keyCache.get(keyOffset + i);
            }
            scores.set(scoreBase + (t - windowStart), score);
        }
        context.localBarrier();

        float localMax = Float.NEGATIVE_INFINITY;
        for (int t = windowStart + tid; t <= pos; t += localSize) {
            float v = scores.get(scoreBase + (t - windowStart));
            if (v > localMax) {
                localMax = v;
            }
        }
        reduce[tid] = localMax;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float other = reduce[tid + stride];
                if (other > reduce[tid]) {
                    reduce[tid] = other;
                }
            }
            context.localBarrier();
        }
        float maxScore = reduce[0];
        context.localBarrier();

        float localSum = 0.0f;
        for (int t = windowStart + tid; t <= pos; t += localSize) {
            float e = TornadoMath.exp(scores.get(scoreBase + (t - windowStart)) - maxScore);
            scores.set(scoreBase + (t - windowStart), e);
            localSum += e;
        }
        reduce[tid] = localSum;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            context.localBarrier();
        }
        float sum = reduce[0];
        float normFactor = (sum > 0.0f) ? (1.0f / sum) : (1.0f / (pos - windowStart + 1));
        context.localBarrier();

        for (int t = windowStart + tid; t <= pos; t += localSize) {
            scores.set(scoreBase + (t - windowStart), scores.get(scoreBase + (t - windowStart)) * normFactor);
        }
        context.localBarrier();

        for (int i = tid; i < headDim; i += localSize) {
            float weightedSum = 0.0f;
            for (int t = windowStart; t <= pos; t++) {
                int valueOffset = cacheBaseOffset + t * kvDim + kvHeadIdx * headDim;
                weightedSum += scores.get(scoreBase + (t - windowStart)) * valueCache.get(valueOffset + i);
            }
            out.set(outBase + i, new HalfFloat(weightedSum));
        }
    }

    // ── Feed-forward ─────────────────────────────────────────────────────────

    // @formatter:off
    /**
     * GeGLU over the packed {@code [gate|up]} rows the fused gate/up GEMM produced, emitted in FP16
     * for the down projection's GEMM: {@code hb[b,i] = gelu(gateUp[b,i]) * gateUp[b,ffnLen+i]}.
     *
     * <p>The GeGLU, not a SwiGLU with the activation swapped: {@link
     * TransformerComputeKernelsLayered#geluActivation} is the same function {@link
     * Gemma4Kernels#fusedGateUpGeGLUQ8} applies on the single-token path, so the two paths differ
     * in where the products come from and not in what is computed from them.
     *
     * <p>Worker: {@code B*ffnLen} threads, local 256.
     */
    // @formatter:on
    public static void batchedGeGLUFP16Packed(
            KernelContext context, HalfFloatArray hbFP16, FloatArray gateUp, int ffnLen) {
        int gid = context.globalIdx;
        int b = gid / ffnLen;
        int i = gid - b * ffnLen;
        int rowBase = b * 2 * ffnLen;
        float gate = gateUp.get(rowBase + i);
        float up = gateUp.get(rowBase + ffnLen + i);
        hbFP16.set(gid, new HalfFloat(TransformerComputeKernelsLayered.geluActivation(gate) * up));
    }

    // ── Per-layer embeddings ─────────────────────────────────────────────────

    // @formatter:off
    /**
     * The per-layer-embedding gate, chunk-wide, emitted in FP16 for the projection that follows:
     * {@code out[b,i] = gelu(gate[b,i]) * perLayerInputs[b, peOffset + i]}.
     *
     * <p>{@code perLayerInputs} is one row of {@code numLayers*segmentSize} per prompt token, so
     * {@code peOffset} selects this layer's segment within a row exactly as it selects it within
     * the single vector on the decode path.
     *
     * <p>Worker: {@code B*segmentSize} threads.
     */
    // @formatter:on
    public static void batchedPleGateGeluMul(
            KernelContext context,
            HalfFloatArray out,
            FloatArray gate,
            FloatArray perLayerInputs,
            int peOffset,
            int segmentSize,
            int perLayerTotal) {
        int gid = context.globalIdx;
        int b = gid / segmentSize;
        int i = gid - b * segmentSize;
        float gated = TransformerComputeKernelsLayered.geluActivation(gate.get(gid));
        out.set(gid, new HalfFloat(gated * perLayerInputs.get(b * perLayerTotal + peOffset + i)));
    }

    // @formatter:off
    /**
     * The per-layer projection's pre-scale and per-segment RMS norm, chunk-wide.
     *
     * <p>The chunk-wide form of {@link Gemma4Kernels#pleProjScaleAndNormalize}: a row holds {@code
     * numLayers} segments of {@code segmentSize}, one workgroup normalizes one segment of one row,
     * and the single learned weight is reused for every segment as it is on the decode path.
     *
     * <p>Worker: {@code B*numLayers} workgroups of {@code localMemSize} lanes.
     */
    // @formatter:on
    public static void batchedPleProjScaleAndNormalize(
            KernelContext context,
            FloatArray x,
            FloatArray weight,
            int segmentSize,
            int localMemSize,
            float preScale,
            float rmsNormEps) {
        int segIdx = context.groupIdx;
        int localId = context.localIdx;
        int localSize = context.localGroupSizeX;
        int base = segIdx * segmentSize;

        float[] localSum = context.allocateFloatLocalArray(localMemSize);
        float partial = 0.0f;
        for (int i = localId; i < segmentSize; i += localSize) {
            float v = x.get(base + i) * preScale;
            x.set(base + i, v);
            partial += v * v;
        }
        localSum[localId] = partial;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSum[localId] += localSum[localId + stride];
            }
            context.localBarrier();
        }
        float ss = localSum[0] / segmentSize + rmsNormEps;
        ss = 1.0f / TornadoMath.sqrt(ss);
        context.localBarrier();
        for (int i = localId; i < segmentSize; i += localSize) {
            x.set(base + i, weight.get(i) * (ss * x.get(base + i)));
        }
    }

    // ── Elementwise ──────────────────────────────────────────────────────────

    /** {@code out[i] = (a[i] + b[i]) * scale} — the chunk-wide {@link Gemma4Kernels#addAndScale}. */
    public static void batchedAddAndScale(
            KernelContext context, FloatArray out, FloatArray a, FloatArray b, float scale) {
        int gid = context.globalIdx;
        out.set(gid, (a.get(gid) + b.get(gid)) * scale);
    }

    /** {@code x[i] *= scale} — the embedding scale Gemma 4 applies on input. */
    public static void batchedScaleInPlace(KernelContext context, FloatArray x, float scale) {
        int gid = context.globalIdx;
        x.set(gid, x.get(gid) * scale);
    }

    /**
     * {@code x[i] *= scaleTensor[0]} — the chunk-wide {@link Gemma4Kernels#scaleInPlaceFromTensor},
     * for the layers that carry a learned output scale.
     */
    public static void batchedScaleInPlaceFromTensor(
            KernelContext context, FloatArray x, FloatArray scaleTensor) {
        int gid = context.globalIdx;
        x.set(gid, x.get(gid) * scaleTensor.get(0));
    }
}
