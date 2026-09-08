package org.beehive.gpullama3.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The {@code qwen35} mixer kernels over a chunk of prompt tokens.
 *
 * <h2>Two kinds of kernel, and the difference is the whole design</h2>
 *
 * <p>Most of what a layer does is per token: a projection, a norm, a rotation, a gate. Those become
 * batched by adding a row index, and the rows are independent.
 *
 * <p>Two are not. A recurrent layer's convolution keeps a rolling window per channel and its
 * delta-net keeps a matrix per value head, and token {@code t} reads what token {@code t-1} wrote.
 * Those two kernels <b>scan</b>: a lane walks the chunk in order, in one loop, updating the state
 * it alone owns. That is exact rather than approximate — a convolution channel's window is private
 * to that channel and a delta-net value column's state is private to that column, so the
 * sequential dependency lives entirely inside one lane and needs no barrier and no ordering
 * between lanes.
 *
 * <p>It is also why the result cannot depend on the chunk size: the arithmetic per token is the
 * same expression in the same order as the single-token kernel performs, and the batch width only
 * decides how many iterations a lane runs before returning.
 *
 * <h2>Padding</h2>
 *
 * <p>Every kernel launches a fixed number of rows and is told how many are active. An inactive row
 * computes nothing, writes no key/value entry, and — in the scans — is never reached, because the
 * loop bound is the active count rather than the launch width.
 *
 * <p>Bodies are lifted into lane methods wherever the arithmetic is worth checking on the host,
 * for the reason {@code Qwen35DeltaNetKernels} gives: a method taking a {@link KernelContext} can
 * only be exercised by running it on a device.
 */
// @formatter:on
public final class Qwen35BatchKernels {

    private Qwen35BatchKernels() {}

    // ---- the recurrent scans -------------------------------------------------

    /**
     * One channel of the depthwise causal convolution, walked across the chunk in token order.
     *
     * <p>The single-token kernel's body, in a loop: for each token it convolves the window with
     * this channel's taps, writes that token's output, and shifts the window. A lane owns one
     * channel's window slice, so nothing here races and nothing needs a barrier.
     *
     * @param channel the lane
     * @param activeRows how many of the chunk's rows carry a real token
     */
    static void causalConv1dScanLane(
            FloatArray inputBatch,
            FloatArray weight,
            FloatArray window,
            FloatArray outBatch,
            int channels,
            int kernel,
            int windowOffset,
            int activeRows,
            int channel) {
        int history = kernel - 1;
        int wBase = channel * kernel;
        int hBase = windowOffset + channel * history;

        for (int row = 0; row < activeRows; row++) {
            float x = inputBatch.get(row * channels + channel);

            float sum = 0.0f;
            for (int t = 0; t < history; t++) {
                sum += weight.get(wBase + t) * window.get(hBase + t);
            }
            sum += weight.get(wBase + history) * x;
            outBatch.set(row * channels + channel, sum);

            for (int t = 0; t + 1 < history; t++) {
                window.set(hBase + t, window.get(hBase + t + 1));
            }
            if (history > 0) {
                window.set(hBase + history - 1, x);
            }
        }
    }

    /** One lane per channel; each walks the whole chunk. */
    public static void causalConv1dScan(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray weight,
            FloatArray window,
            FloatArray outBatch,
            int channels,
            int kernel,
            int windowOffset,
            IntArray batchInfo) {
        int channel = context.globalIdx;
        if (channel >= channels) {
            return;
        }
        causalConv1dScanLane(
                inputBatch,
                weight,
                window,
                outBatch,
                channels,
                kernel,
                windowOffset,
                batchInfo.get(1),
                channel);
    }

    /**
     * One value column of one head, walked across the chunk in token order.
     *
     * <p>Per token: decay this column, predict, correct, accumulate the rank-one update, read out.
     * The same two passes the single-token kernel makes, and the same modulo mapping from a value
     * head to its key head — the reference repeats key heads by tiling, and dividing instead pairs
     * every value head with the wrong key.
     *
     * @param lane {@code head * stateDim + column}
     */
    static void deltaRuleScanLane(
            FloatArray qBatch,
            FloatArray kBatch,
            FloatArray vBatch,
            FloatArray decayBatch,
            FloatArray betaBatch,
            FloatArray state,
            FloatArray outBatch,
            int valueHeads,
            int keyHeads,
            int stateDim,
            int stateOffset,
            int activeRows,
            int lane) {
        int head = lane / stateDim;
        int column = lane - head * stateDim;

        int stateBase = stateOffset + head * stateDim * stateDim;
        int keyBase = (head % keyHeads) * stateDim;
        int valueBase = head * stateDim;
        int keyRowStride = keyHeads * stateDim;
        int valueRowStride = valueHeads * stateDim;

        for (int row = 0; row < activeRows; row++) {
            float g = decayBatch.get(row * valueHeads + head);
            float b = betaBatch.get(row * valueHeads + head);
            int keyRow = row * keyRowStride + keyBase;
            int valueRow = row * valueRowStride + valueBase;

            float prediction = 0.0f;
            for (int i = 0; i < stateDim; i++) {
                int index = stateBase + i * stateDim + column;
                float decayed = state.get(index) * g;
                state.set(index, decayed);
                prediction += decayed * kBatch.get(keyRow + i);
            }

            float correction = (vBatch.get(valueRow + column) - prediction) * b;

            float readout = 0.0f;
            for (int i = 0; i < stateDim; i++) {
                int index = stateBase + i * stateDim + column;
                float updated = state.get(index) + kBatch.get(keyRow + i) * correction;
                state.set(index, updated);
                readout += updated * qBatch.get(keyRow + i);
            }
            outBatch.set(valueRow + column, readout);
        }
    }

    /** One lane per (value head, value column); each walks the whole chunk. */
    public static void deltaRuleScan(
            KernelContext context,
            FloatArray qBatch,
            FloatArray kBatch,
            FloatArray vBatch,
            FloatArray decayBatch,
            FloatArray betaBatch,
            FloatArray state,
            FloatArray outBatch,
            int valueHeads,
            int keyHeads,
            int stateDim,
            int stateOffset,
            IntArray batchInfo) {
        int lane = context.globalIdx;
        if (lane >= valueHeads * stateDim) {
            return;
        }
        deltaRuleScanLane(
                qBatch,
                kBatch,
                vBatch,
                decayBatch,
                betaBatch,
                state,
                outBatch,
                valueHeads,
                keyHeads,
                stateDim,
                stateOffset,
                batchInfo.get(1),
                lane);
    }

    // ---- per-token kernels, with a row index ---------------------------------

    /** SiLU over a chunk, in place. One lane per element of the chunk. */
    public static void siluInPlaceBatch(
            KernelContext context, FloatArray values, int count, IntArray batchInfo) {
        int lane = context.globalIdx;
        if (lane >= count * batchInfo.get(1)) {
            return;
        }
        float v = values.get(lane);
        values.set(lane, v / (1.0f + TornadoMath.exp(-v)));
    }

    /** {@code x *= scale} over a chunk, in place. */
    public static void scaleInPlaceBatch(
            KernelContext context, FloatArray values, float scale, int count, IntArray batchInfo) {
        int lane = context.globalIdx;
        if (lane >= count * batchInfo.get(1)) {
            return;
        }
        values.set(lane, values.get(lane) * scale);
    }

    /**
     * The fused {@code q ‖ k ‖ v} split over a chunk.
     *
     * <p>Row-major throughout: the source row is {@code dimA + dimB + dimC} wide and each
     * destination row is its own width, so the split is a change of stride as well as of offset.
     */
    static void splitThreeWayBatchLane(
            FloatArray fusedBatch,
            FloatArray a,
            FloatArray b,
            FloatArray c,
            int dimA,
            int dimB,
            int dimC,
            int lane) {
        int fusedWidth = dimA + dimB + dimC;
        int row = lane / fusedWidth;
        int element = lane - row * fusedWidth;
        float value = fusedBatch.get(lane);
        if (element < dimA) {
            a.set(row * dimA + element, value);
        } else if (element < dimA + dimB) {
            b.set(row * dimB + (element - dimA), value);
        } else {
            c.set(row * dimC + (element - dimA - dimB), value);
        }
    }

    /** One lane per element of the chunk's fused rows. */
    public static void splitThreeWayBatch(
            KernelContext context,
            FloatArray fusedBatch,
            FloatArray a,
            FloatArray b,
            FloatArray c,
            int dimA,
            int dimB,
            int dimC,
            IntArray batchInfo) {
        int lane = context.globalIdx;
        if (lane >= (dimA + dimB + dimC) * batchInfo.get(1)) {
            return;
        }
        splitThreeWayBatchLane(fusedBatch, a, b, c, dimA, dimB, dimC, lane);
    }

    /** One head of one row scaled to unit length. One lane per (row, head). */
    public static void l2NormPerHeadBatch(
            KernelContext context,
            FloatArray values,
            int heads,
            int headDim,
            float eps,
            IntArray batchInfo) {
        int lane = context.globalIdx;
        if (lane >= heads * batchInfo.get(1)) {
            return;
        }
        int base = lane * headDim;
        float ss = 0.0f;
        for (int i = 0; i < headDim; i++) {
            float v = values.get(base + i);
            ss += v * v;
        }
        float inv = 1.0f / TornadoMath.max(TornadoMath.sqrt(ss), eps);
        for (int i = 0; i < headDim; i++) {
            values.set(base + i, values.get(base + i) * inv);
        }
    }

    /** One row's decay and write strength. One lane per (row, value head). */
    public static void decayAndBetaBatch(
            KernelContext context,
            FloatArray alphaBatch,
            FloatArray betaBatch,
            FloatArray dtBias,
            FloatArray a,
            int valueHeads,
            IntArray batchInfo) {
        int lane = context.globalIdx;
        if (lane >= valueHeads * batchInfo.get(1)) {
            return;
        }
        int head = lane % valueHeads;

        float raw = betaBatch.get(lane);
        betaBatch.set(lane, 1.0f / (1.0f + TornadoMath.exp(-raw)));

        float biased = alphaBatch.get(lane) + dtBias.get(head);
        float softplus = biased > 20.0f ? biased : TornadoMath.log(1.0f + TornadoMath.exp(biased));
        alphaBatch.set(lane, TornadoMath.exp(a.get(head) * softplus));
    }

    /** {@code rms_norm(values, weight) * silu(gate)} per head. One lane per (row, head). */
    public static void gatedNormPerHeadBatch(
            KernelContext context,
            FloatArray values,
            FloatArray gate,
            FloatArray weight,
            int heads,
            int headDim,
            float eps,
            IntArray batchInfo) {
        int lane = context.globalIdx;
        if (lane >= heads * batchInfo.get(1)) {
            return;
        }
        int base = lane * headDim;

        float ss = 0.0f;
        for (int i = 0; i < headDim; i++) {
            float v = values.get(base + i);
            ss += v * v;
        }
        float inv = 1.0f / TornadoMath.sqrt(ss / headDim + eps);

        for (int i = 0; i < headDim; i++) {
            float z = gate.get(base + i);
            float silu = z / (1.0f + TornadoMath.exp(-z));
            values.set(base + i, weight.get(i) * (inv * values.get(base + i)) * silu);
        }
    }

    // ---- attention-layer kernels ---------------------------------------------

    /** The interleaved query/gate projection, separated per row. */
    public static void splitQueryGateBatch(
            KernelContext context,
            FloatArray fusedBatch,
            FloatArray queryBatch,
            FloatArray gateBatch,
            int heads,
            int headDim,
            IntArray batchInfo) {
        int lane = context.globalIdx;
        int width = heads * headDim;
        if (lane >= width * batchInfo.get(1)) {
            return;
        }
        int row = lane / width;
        int within = lane - row * width;
        int head = within / headDim;
        int element = within - head * headDim;
        int fusedBase = row * 2 * width + head * 2 * headDim;
        queryBatch.set(lane, fusedBatch.get(fusedBase + element));
        gateBatch.set(lane, fusedBatch.get(fusedBase + headDim + element));
    }

    /**
     * Partial NeoX rotation over a chunk, at each row's own position.
     *
     * <p>{@code batchInfo[0]} is the position of row 0, so row {@code r} rotates at {@code
     * batchInfo[0] + r}. A single scalar position would rotate the whole chunk as though it were
     * one token, which is the defect this signature exists to prevent.
     */
    public static void ropeNeoxPartialBatch(
            KernelContext context,
            IntArray batchInfo,
            FloatArray queryBatch,
            FloatArray keyBatch,
            FloatArray freqCisReal,
            FloatArray freqCisImag,
            int heads,
            int keyValueHeads,
            int headDim,
            int rotaryDim) {
        int lane = context.globalIdx;
        int half = rotaryDim / 2;
        int perRow = heads * half;
        if (lane >= perRow * batchInfo.get(1)) {
            return;
        }
        int row = lane / perRow;
        int within = lane - row * perRow;
        int head = within / half;
        int ic = within - head * half;
        int position = batchInfo.get(0) + row;

        float fcr = freqCisReal.get(position * half + ic);
        float fci = freqCisImag.get(position * half + ic);

        int queryBase = row * heads * headDim + head * headDim;
        float q0 = queryBatch.get(queryBase + ic);
        float q1 = queryBatch.get(queryBase + ic + half);
        queryBatch.set(queryBase + ic, q0 * fcr - q1 * fci);
        queryBatch.set(queryBase + ic + half, q0 * fci + q1 * fcr);

        if (head < keyValueHeads) {
            int keyBase = row * keyValueHeads * headDim + head * headDim;
            float k0 = keyBatch.get(keyBase + ic);
            float k1 = keyBatch.get(keyBase + ic + half);
            keyBatch.set(keyBase + ic, k0 * fcr - k1 * fci);
            keyBatch.set(keyBase + ic + half, k0 * fci + k1 * fcr);
        }
    }

    /**
     * Each row's key and value written into the paged store, at that row's own position.
     *
     * <p>Exactly one write per (row, element): the store is left in the state sequential ingestion
     * would leave it in, which is what decode then reads.
     */
    public static void appendKeyValueBatchPaged(
            KernelContext context,
            IntArray batchInfo,
            FloatArray keyBatch,
            FloatArray valueBatch,
            FloatArray keyCache,
            FloatArray valueCache,
            IntArray blockTable,
            int kvDim,
            int layer,
            int blockCfg,
            int blockStride) {
        int lane = context.globalIdx;
        if (lane >= kvDim * batchInfo.get(1)) {
            return;
        }
        int row = lane / kvDim;
        int element = lane - row * kvDim;
        int position = batchInfo.get(0) + row;
        int slot = batchInfo.get(2);

        int cacheOffset =
                KvBlockAddress.offset(
                        blockTable,
                        slot,
                        position,
                        KvBlockAddress.layerOffset(layer, kvDim, blockCfg),
                        kvDim,
                        blockCfg,
                        blockStride);
        keyCache.set(cacheOffset + element, keyBatch.get(lane));
        valueCache.set(cacheOffset + element, valueBatch.get(lane));
    }

    /**
     * Causal attention for one (row, head), over the paged store.
     *
     * <p>One workgroup per (row, head), with the query staged and the tiles held in local memory
     * sized from the head width — this family's head is 256 wide, and the shared split-KV kernel's
     * arrays are fixed at 128.
     *
     * <p>The mask is the loop bound. A row reads positions {@code 0..startPos + row} inclusive and
     * cannot reach a later row, whose key/value entries the append above has already written.
     */
    public static void attentionBatchPaged(
            KernelContext context,
            IntArray batchInfo,
            FloatArray queryBatch,
            FloatArray keyCache,
            FloatArray valueCache,
            FloatArray outBatch,
            int heads,
            int headSize,
            int kvDim,
            int kvMul,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int localWorkGroupSize) {
        int tid = context.localIdx;
        // The workgroup width as a parameter, not as context.localGroupSizeX: a local array's
        // extent has to be a compile-time constant on CUDA, and a value read from the context is
        // not one ("expression must have a constant value" from nvrtc, on the __shared__ decl).
        int localSize = localWorkGroupSize;
        int group = context.groupIdx;
        int row = group / heads;
        int head = group - row * heads;
        if (row >= batchInfo.get(1)) {
            return;
        }

        int position = batchInfo.get(0) + row;
        int slot = batchInfo.get(2);
        int layerOff = KvBlockAddress.layerOffset(layer, kvDim, blockCfg);
        int kvHead = head / kvMul;
        float invSqrt = 1.0f / TornadoMath.sqrt(headSize);

        float[] qShared = context.allocateFloatLocalArray(headSize);
        float[] partialMax = context.allocateFloatLocalArray(localWorkGroupSize);
        float[] partialSum = context.allocateFloatLocalArray(localWorkGroupSize);
        float[] reduced = context.allocateFloatLocalArray(2);

        int queryBase = row * heads * headSize + head * headSize;
        for (int i = tid; i < headSize; i += localSize) {
            qShared[i] = queryBatch.get(queryBase + i);
        }
        context.localBarrier();

        // Pass 1: this lane's slice of the causal range, tracking a running maximum and sum.
        float maxScore = Float.NEGATIVE_INFINITY;
        for (int p = tid; p <= position; p += localSize) {
            int base =
                    KvBlockAddress.offset(
                                    blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                            + kvHead * headSize;
            float score = 0.0f;
            for (int d = 0; d < headSize; d++) {
                score += qShared[d] * keyCache.get(base + d);
            }
            score *= invSqrt;
            maxScore = TornadoMath.max(maxScore, score);
        }
        partialMax[tid] = maxScore;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partialMax[tid] = TornadoMath.max(partialMax[tid], partialMax[tid + stride]);
            }
            context.localBarrier();
        }
        if (tid == 0) {
            reduced[0] = partialMax[0];
        }
        context.localBarrier();
        float globalMax = reduced[0];

        // Pass 2: the denominator, against the settled maximum.
        float sum = 0.0f;
        for (int p = tid; p <= position; p += localSize) {
            int base =
                    KvBlockAddress.offset(
                                    blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                            + kvHead * headSize;
            float score = 0.0f;
            for (int d = 0; d < headSize; d++) {
                score += qShared[d] * keyCache.get(base + d);
            }
            sum += TornadoMath.exp(score * invSqrt - globalMax);
        }
        partialSum[tid] = sum;
        context.localBarrier();
        for (int stride = localSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partialSum[tid] += partialSum[tid + stride];
            }
            context.localBarrier();
        }
        if (tid == 0) {
            reduced[1] = partialSum[0];
        }
        context.localBarrier();
        float denominator = reduced[1];

        // Pass 3: one output element per lane, weighted over the same range.
        int outBase = row * heads * headSize + head * headSize;
        for (int d = tid; d < headSize; d += localSize) {
            float accumulated = 0.0f;
            for (int p = 0; p <= position; p++) {
                int base =
                        KvBlockAddress.offset(
                                        blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                                + kvHead * headSize;
                float score = 0.0f;
                for (int i = 0; i < headSize; i++) {
                    score += qShared[i] * keyCache.get(base + i);
                }
                float weight = TornadoMath.exp(score * invSqrt - globalMax);
                accumulated += weight * valueCache.get(base + d);
            }
            outBatch.set(outBase + d, accumulated / denominator);
        }
    }

    /** The attention result gated by the logistic of its gate, over a chunk. */
    public static void applyOutputGateBatch(
            KernelContext context,
            FloatArray values,
            FloatArray gate,
            int count,
            IntArray batchInfo) {
        int lane = context.globalIdx;
        if (lane >= count * batchInfo.get(1)) {
            return;
        }
        float g = gate.get(lane);
        values.set(lane, values.get(lane) * (1.0f / (1.0f + TornadoMath.exp(-g))));
    }

    /**
     * The per-head query and key RMS norms over a chunk, one lane per (row, head).
     *
     * <p>A lane rather than a workgroup with a reduction: the head is 256 wide and there are 28 of
     * them per row, so the serial sum costs less than the barriers would. The learned scales are
     * one head wide and shared by every head and every row.
     *
     * @param heads query heads; lanes past them address key heads
     */
    public static void fusedQKRmsNormBatch(
            KernelContext context,
            FloatArray queryBatch,
            FloatArray keyBatch,
            FloatArray queryWeights,
            FloatArray keyWeights,
            int heads,
            int keyValueHeads,
            int headDim,
            float eps,
            IntArray batchInfo) {
        int lane = context.globalIdx;
        int perRow = heads + keyValueHeads;
        if (lane >= perRow * batchInfo.get(1)) {
            return;
        }
        int row = lane / perRow;
        int head = lane - row * perRow;

        if (head < heads) {
            normalizeHead(queryBatch, queryWeights, row * heads * headDim + head * headDim, headDim, eps);
        } else {
            int keyHead = head - heads;
            normalizeHead(
                    keyBatch,
                    keyWeights,
                    row * keyValueHeads * headDim + keyHead * headDim,
                    headDim,
                    eps);
        }
    }

    private static void normalizeHead(
            FloatArray values, FloatArray weights, int base, int headDim, float eps) {
        float ss = 0.0f;
        for (int i = 0; i < headDim; i++) {
            float v = values.get(base + i);
            ss += v * v;
        }
        float inv = 1.0f / TornadoMath.sqrt(ss / headDim + eps);
        for (int i = 0; i < headDim; i++) {
            values.set(base + i, weights.get(i) * (inv * values.get(base + i)));
        }
    }
}
