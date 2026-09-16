package org.beehive.jllm.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
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
 * to that channel and a delta-net value column's state is private to that column, so the sequential
 * dependency lives entirely inside one lane and needs no barrier and no ordering between lanes.
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
 * <p>Bodies are lifted into lane methods wherever the arithmetic is worth checking on the host, for
 * the reason {@code Qwen35DeltaNetKernels} gives: a method taking a {@link KernelContext} can only
 * be exercised by running it on a device.
 */
// @formatter:on
public final class Qwen35BatchKernels {

    /** Positions whose weights the workgroup computes together in the value pass. */
    private static final int ATTENTION_TILE = 16;

    /**
     * Output elements one lane carries in the value pass.
     *
     * <p>{@code headSize / localSize}, rounded up, for this family's 256-wide head against a
     * 128-lane workgroup. A constant because a private array's extent has to be one.
     */
    private static final int ATTENTION_SLOTS = 4;

    /** Head dimensions one staged key tile of the first pass holds. */
    private static final int ATTENTION_STAGE_DIMS = 32;

    /**
     * Positions the wide value pass takes at a time: one per lane of the 128-lane workgroup, so
     * every lane computes one weight and the two barriers are paid per 128 positions. The
     * candidate's own constant; {@link #ATTENTION_TILE} stays the reference kernels'.
     */
    private static final int ATTENTION_VALUE_TILE_WIDE = 128;

    /**
     * Lanes the staged first pass is written for: its load mapping puts one position's 32
     * dimensions on one warp and covers a 128-position tile with 128 lanes.
     */
    public static final int ATTENTION_STAGE_LANES = 128;

    /**
     * Row pitch of the transposed key tile, in floats: the 128 positions of a tile plus one, so
     * that the 32 dimensions of one position — stored by one warp in one instruction — land in 32
     * distinct banks.
     */
    private static final int ATTENTION_STAGE_LD = 129;

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

    /** Columns one workgroup of {@link #deltaRuleScanShared} owns: one lane each. */
    public static final int DELTA_SHARED_COLUMNS = 32;

    /** State rows the shared tile is sized for: this family's 128-wide state. */
    public static final int DELTA_SHARED_STATE_DIM = 128;

    /**
     * Whether the state geometry can run the shared-state scan: the tile is sized for a 128-wide
     * state, whose columns fall into whole groups of 32.
     */
    public static boolean deltaSharedEligible(int stateDim) {
        return stateDim == DELTA_SHARED_STATE_DIM;
    }

    // @formatter:off
    /**
     * {@link #deltaRuleScan} with the state column held in shared memory across the chunk.
     *
     * <p>One 32-lane workgroup owns 32 adjacent value columns of one head; lane {@code l} owns
     * column {@code col0 + l}. The 128 x 32 state slice is staged once from the persistent state —
     * for each state row, the 32 lanes read 32 consecutive floats — into {@code tile[i * 32 +
     * lane]}, so a lane's column lives in one bank and no lane ever touches another lane's
     * elements; the scan then walks the active tokens in order doing exactly the per-token decay,
     * prediction, correction, update and readout of the reference lane, in the reference order,
     * reading and writing the column through the tile, and writes the column back once at the end.
     * No consumer reads the persistent state between tokens of a chunk: the scan is the only writer
     * in the batched graph and the next reader is the next chunk's scan (or decode's), so the
     * intermediate values are private to the lane either way.
     *
     * <p>Worker: {@code valueHeads * (stateDim / 32)} groups of 32 lanes. Requires {@code stateDim
     * % 32 == 0}, which {@link #deltaSharedEligible(int)} decides.
     */
    // @formatter:on
    public static void deltaRuleScanShared(
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
        int lane = context.localIdx;
        int group = context.groupIdx;
        int columnGroups = stateDim / DELTA_SHARED_COLUMNS;
        int head = group / columnGroups;
        int column = (group - head * columnGroups) * DELTA_SHARED_COLUMNS + lane;
        int activeRows = batchInfo.get(1);

        int stateBase = stateOffset + head * stateDim * stateDim;
        int keyBase = (head % keyHeads) * stateDim;
        int valueBase = head * stateDim;
        int keyRowStride = keyHeads * stateDim;
        int valueRowStride = valueHeads * stateDim;

        // The state column: tile[i * 32 + lane] is element i of this lane's column.
        float[] tile =
                context.allocateFloatLocalArray(DELTA_SHARED_STATE_DIM * DELTA_SHARED_COLUMNS);
        for (int i = 0; i < stateDim; i++) {
            tile[i * DELTA_SHARED_COLUMNS + lane] = state.get(stateBase + i * stateDim + column);
        }

        for (int row = 0; row < activeRows; row++) {
            float g = decayBatch.get(row * valueHeads + head);
            float b = betaBatch.get(row * valueHeads + head);
            int keyRow = row * keyRowStride + keyBase;
            int valueRow = row * valueRowStride + valueBase;

            float prediction = 0.0f;
            for (int i = 0; i < stateDim; i++) {
                int index = i * DELTA_SHARED_COLUMNS + lane;
                float decayed = tile[index] * g;
                tile[index] = decayed;
                prediction += decayed * kBatch.get(keyRow + i);
            }

            float correction = (vBatch.get(valueRow + column) - prediction) * b;

            float readout = 0.0f;
            for (int i = 0; i < stateDim; i++) {
                int index = i * DELTA_SHARED_COLUMNS + lane;
                float updated = tile[index] + kBatch.get(keyRow + i) * correction;
                tile[index] = updated;
                readout += updated * qBatch.get(keyRow + i);
            }
            outBatch.set(valueRow + column, readout);
        }

        for (int i = 0; i < stateDim; i++) {
            state.set(stateBase + i * stateDim + column, tile[i * DELTA_SHARED_COLUMNS + lane]);
        }
    }

    /** Warps per workgroup of {@link #deltaRuleScanWarp}: each owns one value column. */
    public static final int DELTA_WARP_COLUMNS_PER_GROUP = 4;

    /** Lanes of {@link #deltaRuleScanWarp}'s workgroup. */
    public static final int DELTA_WARP_LOCAL = DELTA_WARP_COLUMNS_PER_GROUP * 32;

    /**
     * Whether the state geometry can run the warp-per-column scan: a 128-wide state, four rows a
     * lane.
     */
    public static boolean deltaWarpEligible(int stateDim) {
        return stateDim == DELTA_SHARED_STATE_DIM;
    }

    /**
     * The sum of {@code value} over the 32 lanes of the warp, folded with five shuffle-down steps
     * (lane {@code l} adds lane {@code l + 16}, then {@code l + 8}, ... ), and broadcast from lane
     * zero so every lane holds it. A fixed association: {@code ((v0 + v16) + (v8 + v24)) + ...},
     * the same on every call.
     */
    private static float warpSumBroadcast(KernelContext context, float value) {
        float sum = value;
        sum += context.simdShuffleDown(sum, 16);
        sum += context.simdShuffleDown(sum, 8);
        sum += context.simdShuffleDown(sum, 4);
        sum += context.simdShuffleDown(sum, 2);
        sum += context.simdShuffleDown(sum, 1);
        return context.simdBroadcastFirst(sum);
    }

    // @formatter:off
    /**
     * The batched delta-rule scan with one warp per value column and the column's 128 state
     * elements spread over the warp's lanes in registers, four a lane.
     *
     * <p>Lane {@code l} holds state rows {@code l, l + 32, l + 64, l + 96} of its warp's column,
     * loaded once from the persistent {@code [head][row][column]} layout (a strided gather, since a
     * lane's rows are a state width apart) and written back once. Per token, in order: every lane
     * decays each of its four elements, multiplies each by that row's key and folds the four
     * products; the warp sums the 32 partials for the prediction; every lane computes the same
     * correction from the broadcast prediction, adds {@code key * correction} to each element and
     * folds each updated element times that row's query; the warp sums the partials for the
     * readout, which lane zero stores. The head-to-key-head mapping, the token order, the query
     * scaling and the decay input are the per-lane scan's.
     *
     * <p><b>Not bit-preserving.</b> The per-lane scan sums a column's 128 products in row order in
     * one accumulator; here each lane sums four and the warp folds the 32 partials as a tree, so
     * the prediction and the readout are reassociated. The decay is applied to each element before
     * its product, as in the per-lane scan, and not factored out of the sum.
     *
     * <p>Worker: {@code valueHeads * stateDim} lanes in groups of {@link #DELTA_WARP_LOCAL}: four
     * warps a group, one column each.
     */
    // @formatter:on
    public static void deltaRuleScanWarp(
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
        int local = context.localIdx;
        int lane = local & 31;
        int warp = local >> 5;
        int column0 = context.groupIdx * DELTA_WARP_COLUMNS_PER_GROUP + warp;
        int head = column0 / stateDim;
        int column = column0 - head * stateDim;
        int activeRows = batchInfo.get(1);

        int stateBase = stateOffset + head * stateDim * stateDim;
        int keyBase = (head % keyHeads) * stateDim;
        int valueBase = head * stateDim;
        int keyRowStride = keyHeads * stateDim;
        int valueRowStride = valueHeads * stateDim;

        // This lane's four state rows: lane, lane + 32, lane + 64, lane + 96.
        int row0 = lane;
        int row1 = lane + 32;
        int row2 = lane + 64;
        int row3 = lane + 96;
        float s0 = state.get(stateBase + row0 * stateDim + column);
        float s1 = state.get(stateBase + row1 * stateDim + column);
        float s2 = state.get(stateBase + row2 * stateDim + column);
        float s3 = state.get(stateBase + row3 * stateDim + column);

        for (int row = 0; row < activeRows; row++) {
            float g = decayBatch.get(row * valueHeads + head);
            float b = betaBatch.get(row * valueHeads + head);
            int keyRow = row * keyRowStride + keyBase;
            int valueRow = row * valueRowStride + valueBase;
            float k0 = kBatch.get(keyRow + row0);
            float k1 = kBatch.get(keyRow + row1);
            float k2 = kBatch.get(keyRow + row2);
            float k3 = kBatch.get(keyRow + row3);

            // Decay each element, then its product with the key; fold the four in row order.
            s0 = s0 * g;
            s1 = s1 * g;
            s2 = s2 * g;
            s3 = s3 * g;
            float partial = s0 * k0;
            partial += s1 * k1;
            partial += s2 * k2;
            partial += s3 * k3;
            float prediction = warpSumBroadcast(context, partial);

            float correction = (vBatch.get(valueRow + column) - prediction) * b;

            s0 = s0 + k0 * correction;
            s1 = s1 + k1 * correction;
            s2 = s2 + k2 * correction;
            s3 = s3 + k3 * correction;
            float readoutPartial = s0 * qBatch.get(keyRow + row0);
            readoutPartial += s1 * qBatch.get(keyRow + row1);
            readoutPartial += s2 * qBatch.get(keyRow + row2);
            readoutPartial += s3 * qBatch.get(keyRow + row3);
            float readout = warpSumBroadcast(context, readoutPartial);
            if (lane == 0) {
                outBatch.set(valueRow + column, readout);
            }
        }

        state.set(stateBase + row0 * stateDim + column, s0);
        state.set(stateBase + row1 * stateDim + column, s1);
        state.set(stateBase + row2 * stateDim + column, s2);
        state.set(stateBase + row3 * stateDim + column, s3);
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

        // Pass 3: the weighted value sum, over the same range.
        //
        // A position's score does not depend on the output element, so it is computed once per
        // position and shared, rather than once per (position, output element). Writing the loops
        // the other way round — an output element outside, a position inside, a dot product
        // innermost — costs `headSize` times the arithmetic of pass 1 for the same answer, which
        // is what this kernel used to do and what made attention the second-largest item in the
        // prefill profile.
        //
        // Positions are taken a tile at a time: the workgroup computes the tile's weights
        // cooperatively, then every lane sweeps its own output elements over that tile.
        int outBase = row * heads * headSize + head * headSize;
        float[] weights = context.allocateFloatLocalArray(ATTENTION_TILE);
        float[] accumulated = new float[ATTENTION_SLOTS];
        for (int t = 0; t < ATTENTION_SLOTS; t++) {
            accumulated[t] = 0.0f;
        }

        for (int tileStart = 0; tileStart <= position; tileStart += ATTENTION_TILE) {
            int tileEnd = tileStart + ATTENTION_TILE - 1;
            if (tileEnd > position) {
                tileEnd = position;
            }

            for (int p = tileStart + tid; p <= tileEnd; p += localSize) {
                int base =
                        KvBlockAddress.offset(
                                        blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                                + kvHead * headSize;
                float score = 0.0f;
                for (int i = 0; i < headSize; i++) {
                    score += qShared[i] * keyCache.get(base + i);
                }
                weights[p - tileStart] = TornadoMath.exp(score * invSqrt - globalMax);
            }
            context.localBarrier();

            int slotIndex = 0;
            for (int d = tid; d < headSize; d += localSize) {
                float partial = accumulated[slotIndex];
                for (int p = tileStart; p <= tileEnd; p++) {
                    int base =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            p,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + kvHead * headSize;
                    partial += weights[p - tileStart] * valueCache.get(base + d);
                }
                accumulated[slotIndex] = partial;
                slotIndex++;
            }
            context.localBarrier();
        }

        int slotIndex = 0;
        for (int d = tid; d < headSize; d += localSize) {
            outBatch.set(outBase + d, accumulated[slotIndex] / denominator);
            slotIndex++;
        }
    }

    /** {@link #appendKeyValueBatchPaged} into a half-precision store. */
    public static void appendKeyValueBatchFP16Paged(
            KernelContext context,
            IntArray batchInfo,
            FloatArray keyBatch,
            FloatArray valueBatch,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
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
        keyCache.set(cacheOffset + element, new HalfFloat(keyBatch.get(lane)));
        valueCache.set(cacheOffset + element, new HalfFloat(valueBatch.get(lane)));
    }

    /**
     * {@link #attentionBatchPaged} over a half-precision store.
     *
     * <p>Entries are widened as they are read and every accumulation stays FP32, so the half
     * precision is in the store and nowhere else.
     */
    public static void attentionBatchFP16Paged(
            KernelContext context,
            IntArray batchInfo,
            FloatArray queryBatch,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
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
                score += qShared[d] * keyCache.get(base + d).getFloat32();
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
                score += qShared[d] * keyCache.get(base + d).getFloat32();
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

        // Pass 3: the weighted value sum, over the same range.
        //
        // A position's score does not depend on the output element, so it is computed once per
        // position and shared, rather than once per (position, output element). Writing the loops
        // the other way round — an output element outside, a position inside, a dot product
        // innermost — costs `headSize` times the arithmetic of pass 1 for the same answer, which
        // is what this kernel used to do and what made attention the second-largest item in the
        // prefill profile.
        //
        // Positions are taken a tile at a time: the workgroup computes the tile's weights
        // cooperatively, then every lane sweeps its own output elements over that tile.
        int outBase = row * heads * headSize + head * headSize;
        float[] weights = context.allocateFloatLocalArray(ATTENTION_TILE);
        float[] accumulated = new float[ATTENTION_SLOTS];
        for (int t = 0; t < ATTENTION_SLOTS; t++) {
            accumulated[t] = 0.0f;
        }

        for (int tileStart = 0; tileStart <= position; tileStart += ATTENTION_TILE) {
            int tileEnd = tileStart + ATTENTION_TILE - 1;
            if (tileEnd > position) {
                tileEnd = position;
            }

            for (int p = tileStart + tid; p <= tileEnd; p += localSize) {
                int base =
                        KvBlockAddress.offset(
                                        blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                                + kvHead * headSize;
                float score = 0.0f;
                for (int i = 0; i < headSize; i++) {
                    score += qShared[i] * keyCache.get(base + i).getFloat32();
                }
                weights[p - tileStart] = TornadoMath.exp(score * invSqrt - globalMax);
            }
            context.localBarrier();

            int slotIndex = 0;
            for (int d = tid; d < headSize; d += localSize) {
                float partial = accumulated[slotIndex];
                for (int p = tileStart; p <= tileEnd; p++) {
                    int base =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            p,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + kvHead * headSize;
                    partial += weights[p - tileStart] * valueCache.get(base + d).getFloat32();
                }
                accumulated[slotIndex] = partial;
                slotIndex++;
            }
            context.localBarrier();
        }

        int slotIndex = 0;
        for (int d = tid; d < headSize; d += localSize) {
            outBatch.set(outBase + d, accumulated[slotIndex] / denominator);
            slotIndex++;
        }
    }

    // @formatter:off
    /**
     * {@link #attentionBatchFP16Paged} with each causal query-key dot product computed once.
     *
     * <p>The reference kernel walks the causal range three times and recomputes every dot product
     * on each walk: once for the maximum, once for the denominator, once for the value weights.
     * This form computes it in the first walk, stores the <b>unscaled</b> FP32 sum in {@code
     * scores} and reads it back in the other two. The score is stored before {@code invSqrt} is
     * applied, so the scaled expression the later passes evaluate — {@code score * invSqrt -
     * globalMax} — is the same expression over the same operand bits, and the compiler's
     * contraction of it is the same in every pass. The dot product's own accumulation order, the
     * reductions, the exponentials and the weighted value sum are the reference kernel's.
     *
     * <p>{@code scores} is scratch indexed by {@code ((row * heads) + head) * scoreStride + p}, so
     * every (row, head) workgroup of a launch owns a disjoint span and no launch depends on what an
     * earlier one left: every position a workgroup reads in passes two and three is one its own
     * first pass wrote — the first walk covers {@code p = tid, tid + localSize, ...} up to the
     * row's position, which is exactly the range the later walks read — and the workgroup barriers
     * between the passes are what make one lane's global stores visible to the lanes that read them
     * in the value pass, where a position belongs to a different lane.
     *
     * @param scores per-launch scratch, at least {@code rows * heads * scoreStride} floats
     * @param scoreStride the stride between (row, head) spans; at least the largest position + 1
     */
    // @formatter:on
    public static void attentionBatchFP16PagedScored(
            KernelContext context,
            IntArray batchInfo,
            FloatArray queryBatch,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            FloatArray outBatch,
            int heads,
            int headSize,
            int kvDim,
            int kvMul,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int localWorkGroupSize,
            FloatArray scores,
            int scoreStride) {
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

        int scoreBase = (row * heads + head) * scoreStride;

        // Pass 1: this lane's slice of the causal range, tracking a running maximum; the unscaled
        // dot product is kept for the other two passes.
        float maxScore = Float.NEGATIVE_INFINITY;
        for (int p = tid; p <= position; p += localSize) {
            int base =
                    KvBlockAddress.offset(
                                    blockTable, slot, p, layerOff, kvDim, blockCfg, blockStride)
                            + kvHead * headSize;
            float score = 0.0f;
            for (int d = 0; d < headSize; d++) {
                score += qShared[d] * keyCache.get(base + d).getFloat32();
            }
            scores.set(scoreBase + p, score);
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

        // Pass 2: the denominator, against the settled maximum, from the stored dot products.
        float sum = 0.0f;
        for (int p = tid; p <= position; p += localSize) {
            float score = scores.get(scoreBase + p);
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

        // Pass 3: the weighted value sum, a tile of positions at a time, as in the reference.
        int outBase = row * heads * headSize + head * headSize;
        float[] weights = context.allocateFloatLocalArray(ATTENTION_TILE);
        float[] accumulated = new float[ATTENTION_SLOTS];
        for (int t = 0; t < ATTENTION_SLOTS; t++) {
            accumulated[t] = 0.0f;
        }

        for (int tileStart = 0; tileStart <= position; tileStart += ATTENTION_TILE) {
            int tileEnd = tileStart + ATTENTION_TILE - 1;
            if (tileEnd > position) {
                tileEnd = position;
            }

            for (int p = tileStart + tid; p <= tileEnd; p += localSize) {
                float score = scores.get(scoreBase + p);
                weights[p - tileStart] = TornadoMath.exp(score * invSqrt - globalMax);
            }
            context.localBarrier();

            int slotIndex = 0;
            for (int d = tid; d < headSize; d += localSize) {
                float partial = accumulated[slotIndex];
                for (int p = tileStart; p <= tileEnd; p++) {
                    int base =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            p,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + kvHead * headSize;
                    partial += weights[p - tileStart] * valueCache.get(base + d).getFloat32();
                }
                accumulated[slotIndex] = partial;
                slotIndex++;
            }
            context.localBarrier();
        }

        int slotIndex = 0;
        for (int d = tid; d < headSize; d += localSize) {
            outBatch.set(outBase + d, accumulated[slotIndex] / denominator);
            slotIndex++;
        }
    }

    // @formatter:off
    /**
     * {@link #attentionBatchFP16PagedScored} with the first pass reading keys through a transposed
     * shared-memory tile.
     *
     * <p>The reference first pass has lane {@code p} read key row {@code p} on its own: across a
     * warp the addresses are a key row apart, so every load instruction touches thirty-two separate
     * segments. Here positions are taken 128 at a time and dimensions 32 at a time: the 128 lanes
     * load the tile's 128 x 32 halves with consecutive lanes reading consecutive dimensions of one
     * position (coalesced), widen them to FP32 (exact) and store them transposed, {@code
     * keyTile[dim * 129 + positionInTile]}, so that a lane's later reads of its own position's
     * dimensions are consecutive across the warp and the stores of one position's dimensions fall
     * in distinct banks (the padding to 129 is what separates them). Lane {@code tid} owns position
     * {@code tileStart + tid} — the reference's lane-to-position assignment — and accumulates its
     * one FP32 score over the dimensions in increasing order, tile after tile, the reference's
     * order; the unscaled score is stored and scaled exactly as before.
     *
     * <p>Every lane runs every tile's barriers; a lane whose position lies past the causal range
     * skips only its loads (storing zeros), its accumulation and its stores.
     *
     * <p>Shared memory: {@code 32 * 129} floats for the tile, plus the reference kernel's own. The
     * remaining passes, the score scratch, the launch geometry and the value pass are the reference
     * kernel's.
     */
    // @formatter:on
    public static void attentionBatchFP16PagedScoredStaged(
            KernelContext context,
            IntArray batchInfo,
            FloatArray queryBatch,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            FloatArray outBatch,
            int heads,
            int headSize,
            int kvDim,
            int kvMul,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int localWorkGroupSize,
            FloatArray scores,
            int scoreStride) {
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

        int scoreBase = (row * heads + head) * scoreStride;

        // Pass 1 through the staged tiles: this lane's positions, one per 128-position tile.
        float[] keyTile =
                context.allocateFloatLocalArray(ATTENTION_STAGE_DIMS * ATTENTION_STAGE_LD);
        float maxScore = Float.NEGATIVE_INFINITY;
        int loadPos0 = tid >> 5;
        int loadDim = tid & 31;
        for (int tileStart = 0; tileStart <= position; tileStart += localSize) {
            int p = tileStart + tid;
            float score = 0.0f;
            for (int dimStart = 0; dimStart < headSize; dimStart += ATTENTION_STAGE_DIMS) {
                // Stage: lane tid loads dimension (dimStart + tid % 32) of positions
                // tileStart + tid / 32 + 4i. A warp's 32 lanes read one position's 32
                // consecutive halves.
                for (int i = 0; i < localSize / 4; i++) {
                    int posInTile = loadPos0 + 4 * i;
                    int loadPos = tileStart + posInTile;
                    float value = 0.0f;
                    if (loadPos <= position) {
                        int base =
                                KvBlockAddress.offset(
                                                blockTable,
                                                slot,
                                                loadPos,
                                                layerOff,
                                                kvDim,
                                                blockCfg,
                                                blockStride)
                                        + kvHead * headSize;
                        value = keyCache.get(base + dimStart + loadDim).getFloat32();
                    }
                    keyTile[loadDim * ATTENTION_STAGE_LD + posInTile] = value;
                }
                context.localBarrier();
                if (p <= position) {
                    for (int d = 0; d < ATTENTION_STAGE_DIMS; d++) {
                        score += qShared[dimStart + d] * keyTile[d * ATTENTION_STAGE_LD + tid];
                    }
                }
                context.localBarrier();
            }
            if (p <= position) {
                scores.set(scoreBase + p, score);
                score *= invSqrt;
                maxScore = TornadoMath.max(maxScore, score);
            }
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

        // Pass 2: the denominator, against the settled maximum, from the stored dot products.
        float sum = 0.0f;
        for (int p = tid; p <= position; p += localSize) {
            float score = scores.get(scoreBase + p);
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

        // Pass 3: the weighted value sum, a tile of positions at a time, as in the reference.
        int outBase = row * heads * headSize + head * headSize;
        float[] weights = context.allocateFloatLocalArray(ATTENTION_TILE);
        float[] accumulated = new float[ATTENTION_SLOTS];
        for (int t = 0; t < ATTENTION_SLOTS; t++) {
            accumulated[t] = 0.0f;
        }

        for (int tileStart = 0; tileStart <= position; tileStart += ATTENTION_TILE) {
            int tileEnd = tileStart + ATTENTION_TILE - 1;
            if (tileEnd > position) {
                tileEnd = position;
            }

            for (int p = tileStart + tid; p <= tileEnd; p += localSize) {
                float score = scores.get(scoreBase + p);
                weights[p - tileStart] = TornadoMath.exp(score * invSqrt - globalMax);
            }
            context.localBarrier();

            int slotIndex = 0;
            for (int d = tid; d < headSize; d += localSize) {
                float partial = accumulated[slotIndex];
                for (int p = tileStart; p <= tileEnd; p++) {
                    int base =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            p,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + kvHead * headSize;
                    partial += weights[p - tileStart] * valueCache.get(base + d).getFloat32();
                }
                accumulated[slotIndex] = partial;
                slotIndex++;
            }
            context.localBarrier();
        }

        int slotIndex = 0;
        for (int d = tid; d < headSize; d += localSize) {
            outBatch.set(outBase + d, accumulated[slotIndex] / denominator);
            slotIndex++;
        }
    }

    /**
     * {@link #attentionBatchFP16PagedScoredStaged} with the value pass taking positions 128 at a
     * time instead of 16.
     *
     * <p>Only the third pass changes: its weights array holds 128 exponentials, all 128 lanes
     * compute one each, and a lane sweeps its output elements over the 128 positions of the tile —
     * in increasing position order, the same order the 16-position tiles produce end to end,
     * including the partial last tile — before the next tile is prepared. The two barriers around
     * the shared weights are kept, so they are paid once per 128 positions rather than once per 16.
     * The staged first pass, the stored unscaled scores, the maximum, the denominator, the
     * exponential expression, the causal range, the paged addressing, the head mapping and the grid
     * are the staged kernel's.
     */
    // @formatter:on
    public static void attentionBatchFP16PagedScoredStagedWide(
            KernelContext context,
            IntArray batchInfo,
            FloatArray queryBatch,
            HalfFloatArray keyCache,
            HalfFloatArray valueCache,
            FloatArray outBatch,
            int heads,
            int headSize,
            int kvDim,
            int kvMul,
            int layer,
            IntArray blockTable,
            int blockCfg,
            int blockStride,
            int localWorkGroupSize,
            FloatArray scores,
            int scoreStride) {
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

        int scoreBase = (row * heads + head) * scoreStride;

        // Pass 1 through the staged tiles: this lane's positions, one per 128-position tile.
        float[] keyTile =
                context.allocateFloatLocalArray(ATTENTION_STAGE_DIMS * ATTENTION_STAGE_LD);
        float maxScore = Float.NEGATIVE_INFINITY;
        int loadPos0 = tid >> 5;
        int loadDim = tid & 31;
        for (int tileStart = 0; tileStart <= position; tileStart += localSize) {
            int p = tileStart + tid;
            float score = 0.0f;
            for (int dimStart = 0; dimStart < headSize; dimStart += ATTENTION_STAGE_DIMS) {
                // Stage: lane tid loads dimension (dimStart + tid % 32) of positions
                // tileStart + tid / 32 + 4i. A warp's 32 lanes read one position's 32
                // consecutive halves.
                for (int i = 0; i < localSize / 4; i++) {
                    int posInTile = loadPos0 + 4 * i;
                    int loadPos = tileStart + posInTile;
                    float value = 0.0f;
                    if (loadPos <= position) {
                        int base =
                                KvBlockAddress.offset(
                                                blockTable,
                                                slot,
                                                loadPos,
                                                layerOff,
                                                kvDim,
                                                blockCfg,
                                                blockStride)
                                        + kvHead * headSize;
                        value = keyCache.get(base + dimStart + loadDim).getFloat32();
                    }
                    keyTile[loadDim * ATTENTION_STAGE_LD + posInTile] = value;
                }
                context.localBarrier();
                if (p <= position) {
                    for (int d = 0; d < ATTENTION_STAGE_DIMS; d++) {
                        score += qShared[dimStart + d] * keyTile[d * ATTENTION_STAGE_LD + tid];
                    }
                }
                context.localBarrier();
            }
            if (p <= position) {
                scores.set(scoreBase + p, score);
                score *= invSqrt;
                maxScore = TornadoMath.max(maxScore, score);
            }
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

        // Pass 2: the denominator, against the settled maximum, from the stored dot products.
        float sum = 0.0f;
        for (int p = tid; p <= position; p += localSize) {
            float score = scores.get(scoreBase + p);
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

        // Pass 3: the weighted value sum, a tile of positions at a time, as in the reference.
        int outBase = row * heads * headSize + head * headSize;
        float[] weights = context.allocateFloatLocalArray(ATTENTION_VALUE_TILE_WIDE);
        float[] accumulated = new float[ATTENTION_SLOTS];
        for (int t = 0; t < ATTENTION_SLOTS; t++) {
            accumulated[t] = 0.0f;
        }

        for (int tileStart = 0; tileStart <= position; tileStart += ATTENTION_VALUE_TILE_WIDE) {
            int tileEnd = tileStart + ATTENTION_VALUE_TILE_WIDE - 1;
            if (tileEnd > position) {
                tileEnd = position;
            }

            for (int p = tileStart + tid; p <= tileEnd; p += localSize) {
                float score = scores.get(scoreBase + p);
                weights[p - tileStart] = TornadoMath.exp(score * invSqrt - globalMax);
            }
            context.localBarrier();

            int slotIndex = 0;
            for (int d = tid; d < headSize; d += localSize) {
                float partial = accumulated[slotIndex];
                for (int p = tileStart; p <= tileEnd; p++) {
                    int base =
                            KvBlockAddress.offset(
                                            blockTable,
                                            slot,
                                            p,
                                            layerOff,
                                            kvDim,
                                            blockCfg,
                                            blockStride)
                                    + kvHead * headSize;
                    partial += weights[p - tileStart] * valueCache.get(base + d).getFloat32();
                }
                accumulated[slotIndex] = partial;
                slotIndex++;
            }
            context.localBarrier();
        }

        int slotIndex = 0;
        for (int d = tid; d < headSize; d += localSize) {
            outBatch.set(outBase + d, accumulated[slotIndex] / denominator);
            slotIndex++;
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
            normalizeHead(
                    queryBatch, queryWeights, row * heads * headDim + head * headDim, headDim, eps);
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
