package org.beehive.gpullama3.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * A tensor-core projection that reads {@code Q4_0} weights in the file's own representation.
 *
 * <h2>Why this exists</h2>
 *
 * <p>The scalar batched projections around this one run at about 12% of the device's FP32 rate, 14%
 * of its DRAM rate and well inside its L2 rate — near none of the three, which is what an
 * instruction-bound kernel looks like. The count is roughly two and a half instructions per useful
 * multiply-add: a weight decode per element, eight activation loads, sixteen scalar FMAs. One
 * tensor-core MMA instruction retires 512 multiply-accumulates.
 *
 * <p>This repository already had FP16 tensor-core GEMMs — {@code gemmMMA} and its fused siblings —
 * but only the FP16 families reached them. The family whose whole point is native quantized storage
 * was the one running scalar code.
 *
 * <h2>What it does differently from {@code gemmMMA}</h2>
 *
 * <p>Two things. The weight operand is <b>not</b> materialized: a {@code Q4_0} block is decoded
 * into the shared tile as it is staged, so each weight is decoded once per tile rather than once
 * per output element, and the model stays 4-bit in memory. And the shape is skinny — {@code M} is a
 * chunk of prompt rows, sixteen or thirty-two, not the hundreds {@code gemmMMA}'s 128-row block
 * assumes.
 *
 * <p>{@code gemmMMA} computes {@code C[M,N] = A[M,K] x B[N,K]} with {@code B} row-major over {@code
 * N}. That is already how this project stores a weight matrix — one row per output — so nothing is
 * transposed here; only the decode is new.
 *
 * <h2>Precision</h2>
 *
 * <p>Weights are decoded to FP16, activations are converted to FP16, and the accumulation is FP32,
 * which is what the tensor core does natively. A {@code Q4_0} weight is a 4-bit integer times an
 * FP16 block scale, so the decoded value is very nearly exact in FP16; the activations are what
 * loses a little. Against the scalar path this is strictly less precision in the multiplicands and
 * the same in the accumulator.
 *
 * <h2>Shape requirements</h2>
 *
 * <ul>
 *   <li>{@code K % 32 == 0} — a whole number of {@code Q4_0} blocks;
 *   <li>{@code N % 128 == 0} — the column block;
 *   <li>{@code M} padded up to a multiple of 16, which is why the FP16 activation buffer is
 *       allocated with padded rows. Rows past the chunk's active count are computed and stored, and
 *       never read.
 * </ul>
 *
 * <p>Worker: {@code WorkerGrid1D(ceil(M/16) * (N/128) * 128)}, local 128.
 */
// @formatter:on
public final class Qwen35MMAKernels {

    /** Weights per Q4_0 block. */
    private static final int QK = 32;

    /** Bytes per Q4_0 block: 2 (fp16 scale) + 16 (packed nibbles). */
    private static final int BLOCK_BYTES = 18;

    private static final int WARP_SIZE = 32;

    /** Rows of the output tile — one MMA tile. */
    public static final int BM = 16;

    /**
     * Columns of one MMA panel — the shape's N. The swizzled B load takes no offset, so a wider
     * tile is several panels in several tiles rather than one tile with an offset.
     */
    private static final int PANEL = 8;

    /**
     * Panels one warp carries.
     *
     * <p>One. Four was tried, on the reasoning that the A tile and the two barriers would then be
     * paid once for thirty-two columns instead of eight; it measured 50.7 t/s against 62.1 at
     * pp381. The swizzled B load takes no offset, so more panels means more separate shared tiles
     * and a branch per stored element to pick between them, and that costs more than the
     * amortization returns.
     */
    private static final int PANELS = 1;

    /** Columns of the output tile. */
    public static final int BN = PANEL * PANELS;

    /** The K step, in elements. Half a Q4_0 block, so a block spans two steps. */
    private static final int BK = 16;

    /** Threads per workgroup: one warp, which is what an m16n8k16 tile is. */
    public static final int LOCAL = 32;

    /** Bytes of one eight-column, {@code BK}-deep B panel. */
    private static final int B_SUBTILE_BYTES = 256;

    /** Bytes of one {@code BM}-row, {@code BK}-deep A panel: {@code BM * BK} halves, int-packed. */
    private static final int A_SUBTILE_BYTES = BM * BK * 2;

    private Qwen35MMAKernels() {}

    /**
     * {@code fp16[row][j] = fp32[row][j]} over the chunk, zeroing the rows that pad the last MMA
     * row tile.
     *
     * <p>Those rows are multiplied and stored like any other; they are never read back. Zeroing
     * them keeps whatever the buffer happened to hold out of a tensor-core multiply.
     */
    public static void convertNormedToFP16(
            KernelContext ctx, FloatArray in, HalfFloatArray out, int n, IntArray batchInfo) {
        int index = ctx.globalIdx;
        int row = index / n;
        float value = 0.0f;
        if (row < batchInfo.get(1)) {
            value = in.get(index);
        }
        out.set(index, new HalfFloat(value));
    }

    /**
     * The fp16 at {@code index}, from two byte loads. See {@code TransformerComputeKernelsQ5_K}.
     */
    private static float halfFromBytes(ByteArray w, int index) {
        int lo = w.get(index) & 0xFF;
        int hi = w.get(index + 1) & 0xFF;
        int h = (hi << 8) | lo;
        int mantissa = h & 0x3FF;
        int exponent = (h >>> 10) & 0x1F;
        float magnitude;
        if (exponent == 0) {
            magnitude = mantissa * 5.9604645E-8f;
        } else {
            int e = exponent - 15;
            int magnitudeOfE = e;
            if (e < 0) {
                magnitudeOfE = -e;
            }
            float scale = 1.0f;
            if ((magnitudeOfE & 1) != 0) {
                scale *= 2.0f;
            }
            if ((magnitudeOfE & 2) != 0) {
                scale *= 4.0f;
            }
            if ((magnitudeOfE & 4) != 0) {
                scale *= 16.0f;
            }
            if ((magnitudeOfE & 8) != 0) {
                scale *= 256.0f;
            }
            if (e < 0) {
                scale = 1.0f / scale;
            }
            magnitude = (1.0f + mantissa * (1.0f / 1024.0f)) * scale;
        }
        if ((h & 0x8000) != 0) {
            return -magnitude;
        }
        return magnitude;
    }

    /** One {@code Q4_0} weight of row {@code row}, element {@code k}, as a float. */
    private static float weightAt(ByteArray w, int row, int k, int blocksPerRow) {
        int block = k >> 5;
        int within = k & 31;
        int base = (row * blocksPerRow + block) * BLOCK_BYTES;
        float d = halfFromBytes(w, base);
        int packed = w.get(base + 2 + (within & 15)) & 0xFF;
        int q = packed & 0xF;
        if (within >= 16) {
            q = (packed >> 4) & 0xF;
        }
        return d * (q - 8);
    }

    // @formatter:off
    /**
     * {@code out[M,N] = A[M,K] x W[N,K]}, with {@code W} held as {@code Q4_0} and decoded into a
     * shared tile as it is staged.
     *
     * <p>One warp per output tile of sixteen rows by eight columns — the {@code m16n8k16} shape
     * itself. The B tile is staged through {@code swizzleStoreFp16Stride32} and read with {@code
     * mmaLoadBSwizzled}, which is the route that avoids ever reading a half's bits in the kernel;
     * two implementations that did read them are recorded in {@code
     * docs/architecture/tornadovm-issues}, one refusing to compile and one crashing the compiler.
     *
     * @param aFP16 activations, {@code [M padded to 16][K]}, FP16
     * @param w the weight matrix, as the file stores it, {@code [N][K]} in {@code Q4_0}
     * @param out {@code [M][N]}, FP32
     */
    // @formatter:on
    public static void projectionMMAQ4_0(
            KernelContext ctx,
            HalfFloatArray aFP16,
            ByteArray w,
            FloatArray out,
            int m,
            int n,
            int k) {
        int lane = ctx.localIdx;
        int colTiles = n / BN;
        int group = ctx.groupIdx;
        int rowTile = group / colTiles;
        int colTile = group - rowTile * colTiles;
        int blockRow = rowTile * BM;
        int blockCol = colTile * BN;
        int blocksPerRow = k / QK;

        // One allocation for both A panels, as for B: the first at byte offset zero, the second at
        // A_SUBTILE_BYTES. The A load applies no swizzle — its per-lane address is
        // (row << 5) + col with row < 16 and col in {0, 16}, so a panel reaches at most byte 496
        // and stays inside its own 512 — and the offset-aware load adds the base afterwards, so
        // each panel sees exactly the layout it had as its own array.
        int[] aTile = ctx.allocateIntLocalArray(2 * BM * BK / 2);
        // One allocation for both B panels: the low half at byte offset zero, the high half at
        // B_SUBTILE_BYTES. The offset-aware store and load apply the swizzle to the in-panel
        // address first and add the offset afterwards, so each panel keeps exactly the layout it
        // had as its own array, and the two cannot overlap -- a panel's swizzled address stays
        // inside its own 256 bytes.
        HalfFloat[] bTile = ctx.allocateHalfFloatLocalArray(2 * PANEL * BK);

        float[] acc = ctx.mmaFragment(0.0f);

        // One staging round per Q4_0 block, not per MMA step. A block is 32 elements and the MMA
        // step is 16, so a round stages two tiles and issues two MMAs: the block scale is read
        // once for all 32 of its weights instead of once per weight, both nibble halves of each
        // packed byte are used, and the two barriers are paid per 32 elements rather than per 16.
        //
        // A lane owns eight consecutive elements of one column. Which half of the block those
        // eight fall in is fixed by the lane, so the choice of destination tile is loop-invariant
        // rather than a branch per element.
        int stageCol = lane >> 2;
        int stageQuarter = lane & 3;
        int stageFirst = stageQuarter * 8;
        int stageByte = stageFirst & 15;
        boolean stageHighNibble = stageFirst >= 16;
        boolean stageHighHalf = stageFirst >= 16;
        // Which panel this lane stages, as a byte offset rather than a choice of array.
        int stageOffset = 0;
        if (stageHighHalf) {
            stageOffset = B_SUBTILE_BYTES;
        }
        int stageK = stageFirst & 15;

        int numBlocks = k / QK;
        for (int blockIndex = 0; blockIndex < numBlocks; blockIndex++) {
            int kBase = blockIndex * QK;

            // A: 256 ints over 32 lanes, eight each — two MMA steps' worth. Int i holds row i/8 at
            // element pair (i%8)*2 for the first step, and the same for the second.
            for (int slot = 0; slot < 8; slot++) {
                int i = lane + slot * WARP_SIZE;
                int half = i >>> 7;
                int j = i & 127;
                int row = j >>> 3;
                int kk = (j & 7) << 1;
                int base = (blockRow + row) * k + kBase + half * BK + kk;
                // The same two adjacent halves, the same destination slot, packed the same way —
                // src[base] | src[base + 1] << 16 — but copied global-to-shared without the
                // register round-trip. `base` is even (k is a whole number of Q4_0 blocks, and
                // kBase, half * BK and kk are all even), so the source byte address is
                // header + 2 * base and four-byte aligned, which is what cp.async requires.
                ctx.asyncCopyToLocal(aTile, half * (BM * BK / 2) + j, aFP16, base);
            }

            // B: this lane's column, one scale, eight contiguous packed bytes.
            int base = ((blockCol + stageCol) * blocksPerRow + blockIndex) * BLOCK_BYTES;
            // Read through the array's own half accessor rather than assembling the half from two
            // bytes: the block stride is 18, so every block scale is two-byte aligned, and this
            // lowers to one hardware conversion where halfFromBytes lowers to a ten-branch
            // software expansion. Same bytes, same interpretation, same value. The other kernels
            // in this file keep halfFromBytes.
            float scale = w.getHalfFloat(base).getFloat32();
            for (int t = 0; t < 8; t++) {
                int packed = w.get(base + 2 + stageByte + t) & 0xFF;
                int q = packed & 0xF;
                if (stageHighNibble) {
                    q = (packed >> 4) & 0xF;
                }
                HalfFloat value = new HalfFloat(scale * (q - 8));
                // (k index, column, columns per row) — the order the swizzled load expects.
                ctx.mmaStoreBSwizzled(bTile, stageK + t, stageCol, PANEL, value, stageOffset);
            }
            // Commit and wait before the barrier that publishes both tiles: every lane issues
            // its own eight copies -- i = lane + slot * 32 covers 0..255 exactly once across the
            // warp -- and every lane waits, so no MMA reads a slot whose copy is still in flight.
            // The trailing barrier below keeps the next round's copies out of a tile this round is
            // still reading.
            ctx.asyncCopyCommit();
            ctx.asyncCopyWaitGroup(0);
            ctx.localBarrier();

            acc =
                    ctx.mma(
                            ctx.mmaLoadA(aTile, BK, 0),
                            ctx.mmaLoadBSwizzled(bTile, BK, 0),
                            acc,
                            MMAShape.M16N8K16);
            acc =
                    ctx.mma(
                            ctx.mmaLoadA(aTile, BK, A_SUBTILE_BYTES),
                            ctx.mmaLoadBSwizzled(bTile, BK, B_SUBTILE_BYTES),
                            acc,
                            MMAShape.M16N8K16);
            ctx.localBarrier();
        }

        ctx.mmaStore(acc, out, blockRow, blockCol, n);
    }

    // @formatter:off

    /** {@code hb = silu(gate) * up} over the chunk. One lane per element. */
    public static void swiGLUBatch(
            KernelContext ctx, FloatArray gate, FloatArray up, FloatArray hb) {
        int index = ctx.globalIdx;
        float g = gate.get(index);
        hb.set(index, (g / (1.0f + TornadoMath.exp(-g))) * up.get(index));
    }

    /** {@code fp16[i] = fp32[i]} over a chunk that already fills whole MMA row tiles. */
    public static void convertToFP16(KernelContext ctx, FloatArray in, HalfFloatArray out) {
        int index = ctx.globalIdx;
        out.set(index, new HalfFloat(in.get(index)));
    }

    /**
     * {@code x[i] += delta[i]} — the residual a tensor-core projection cannot fold into its store.
     */
    public static void residualAdd(KernelContext ctx, FloatArray x, FloatArray delta) {
        int index = ctx.globalIdx;
        x.set(index, x.get(index) + delta.get(index));
    }

    // ---- Q5_K ---------------------------------------------------------------

    /** Weights per Q5_K super-block. */
    private static final int QK_K = 256;

    /** Bytes per Q5_K super-block. */
    private static final int K_BLOCK_BYTES = 176;

    private static final int K_SCALES_OFFSET = 4;

    private static final int K_QH_OFFSET = 16;

    private static final int K_QS_OFFSET = 48;

    /**
     * A sub-block's 6-bit scale and minimum, packed as {@code (scale << 8) | min}.
     *
     * <p>Its own method for the reason {@code TransformerComputeKernelsQ5_K} gives: inlined, the
     * decode grew large enough that TornadoVM's CUDA backend emitted a kernel referring to an
     * undeclared {@code context}.
     */
    private static int scaleAndMin(ByteArray w, int scalesBase, int subBlock) {
        if (subBlock < 4) {
            return ((w.get(scalesBase + subBlock) & 63) << 8)
                    | (w.get(scalesBase + subBlock + 4) & 63);
        }
        int lowScale = w.get(scalesBase + subBlock + 4) & 0xFF;
        int highScale = w.get(scalesBase + subBlock - 4) & 0xFF;
        int sc = (lowScale & 0xF) | ((highScale >> 6) << 4);
        int m = ((lowScale >> 4) & 0xF) | (((w.get(scalesBase + subBlock) & 0xFF) >> 6) << 4);
        return (sc << 8) | m;
    }

    // @formatter:off
    /**
     * {@code out[M,N] = A[M,K] x W[N,K]} for {@code Q5_K} weights.
     *
     * <p>The same staging shape as the {@code Q4_0} form: a round is 32 elements, which for Q5_K is
     * one sub-block of a 256-weight super-block, so the sub-block's scale and minimum are computed
     * once per round and a lane owns eight consecutive elements of one column.
     */
    // @formatter:on
    public static void projectionMMAQ5_K(
            KernelContext ctx,
            HalfFloatArray aFP16,
            ByteArray w,
            FloatArray out,
            int m,
            int n,
            int k) {
        int lane = ctx.localIdx;
        int colTiles = n / BN;
        int group = ctx.groupIdx;
        int rowTile = group / colTiles;
        int colTile = group - rowTile * colTiles;
        int blockRow = rowTile * BM;
        int blockCol = colTile * BN;
        int superBlocksPerRow = k / QK_K;

        // One allocation per operand, the second panel at a byte offset, as in
        // projectionMMAQ4_0. The tile geometry and the addressing are the same here: a B panel's
        // in-panel address is at most 254 before the swizzle, which permutes within the same 256
        // bytes, and an A panel's per-lane address reaches at most 496 of its 512. The A load
        // applies no swizzle; the B store and load apply it before adding the offset.
        int[] aTile = ctx.allocateIntLocalArray(2 * BM * BK / 2);
        HalfFloat[] bTile = ctx.allocateHalfFloatLocalArray(2 * PANEL * BK);

        float[] acc = ctx.mmaFragment(0.0f);

        int stageCol = lane >> 2;
        int stageFirst = (lane & 3) * 8;
        boolean stageHighHalf = stageFirst >= 16;
        int stageOffset = 0;
        if (stageHighHalf) {
            stageOffset = B_SUBTILE_BYTES;
        }
        int stageK = stageFirst & 15;

        int numRounds = k / QK;
        for (int round = 0; round < numRounds; round++) {
            int kBase = round * QK;

            for (int slot = 0; slot < 8; slot++) {
                int i = lane + slot * WARP_SIZE;
                int half = i >>> 7;
                int j = i & 127;
                int row = j >>> 3;
                int kk = (j & 7) << 1;
                int base = (blockRow + row) * k + kBase + half * BK + kk;
                // The same two adjacent halves into the same slot, packed the same way, copied
                // global-to-shared without the register round-trip. `base` is even -- the dispatch
                // guard makes k a whole number of blocks, and kBase, half * BK and kk are even --
                // so the source byte address is header + 2 * base and four-byte aligned.
                ctx.asyncCopyToLocal(aTile, half * (BM * BK / 2) + j, aFP16, base);
            }

            int superBlock = round >> 3;
            int subBlock = round & 7;
            int base = ((blockCol + stageCol) * superBlocksPerRow + superBlock) * K_BLOCK_BYTES;
            // As in the Q4_1 kernel: the super-block stride is 176 and d and dmin sit at offsets 0
            // and 2, so both addresses are two-byte aligned. The six-bit sub-block scales at
            // K_SCALES_OFFSET stay byte reads -- they are not halves.
            float d = w.getHalfFloat(base).getFloat32();
            float dmin = w.getHalfFloat(base + 2).getFloat32();
            int packedScale = scaleAndMin(w, base + K_SCALES_OFFSET, subBlock);
            float scale = d * (packedScale >> 8);
            float minimum = dmin * (packedScale & 0xFF);

            int pairIndex = subBlock >> 1;
            int highNibble = subBlock & 1;
            int qsBase = base + K_QS_OFFSET + pairIndex * 32;
            int qhBase = base + K_QH_OFFSET;
            int bitShift = pairIndex * 2 + highNibble;

            for (int t = 0; t < 8; t++) {
                // Q5_K indexes its packed byte by the element's position in the whole sub-block,
                // 0..31 — the nibble half is a property of the sub-block, not of the element, which
                // is what makes this different from Q4_0's byte-and-nibble split.
                int posInSub = stageFirst + t;
                int qsByte = w.get(qsBase + posInSub) & 0xFF;
                int low = qsByte & 0xF;
                if (highNibble == 1) {
                    low = (qsByte >> 4) & 0xF;
                }
                int qhByte = w.get(qhBase + posInSub) & 0xFF;
                int high = (qhByte >> bitShift) & 1;
                HalfFloat value = new HalfFloat(scale * (low + high * 16) - minimum);
                ctx.mmaStoreBSwizzled(bTile, stageK + t, stageCol, PANEL, value, stageOffset);
            }
            // Commit and wait before the barrier that publishes both tiles: every lane issues
            // its own eight copies -- i = lane + slot * 32 covers 0..255 exactly once across the
            // warp -- and every lane waits, so no MMA reads a slot whose copy is still in flight.
            // The trailing barrier below keeps the next round's copies out of a tile this round is
            // still reading.
            ctx.asyncCopyCommit();
            ctx.asyncCopyWaitGroup(0);
            ctx.localBarrier();

            acc =
                    ctx.mma(
                            ctx.mmaLoadA(aTile, BK, 0),
                            ctx.mmaLoadBSwizzled(bTile, BK, 0),
                            acc,
                            MMAShape.M16N8K16);
            acc =
                    ctx.mma(
                            ctx.mmaLoadA(aTile, BK, A_SUBTILE_BYTES),
                            ctx.mmaLoadBSwizzled(bTile, BK, B_SUBTILE_BYTES),
                            acc,
                            MMAShape.M16N8K16);
            ctx.localBarrier();
        }

        ctx.mmaStore(acc, out, blockRow, blockCol, n);
    }

    // ---- Q4_1 ---------------------------------------------------------------

    /** Bytes per Q4_1 block: 2 (d) + 2 (m) + 16 packed nibbles. */
    private static final int BLOCK_BYTES_Q4_1 = 20;

    // @formatter:off
    /**
     * {@code out[M,N] = A[M,K] x W[N,K]} for {@code Q4_1} weights — this model's {@code ffn_down}
     * on the first eight blocks.
     *
     * <p>{@link #projectionMMAQ4_0}'s staging with Q4_1's decode: two fp16 headers rather than one,
     * an unsigned nibble, and {@code d * q + m} rather than {@code d * (q - 8)}.
     */
    // @formatter:on
    public static void projectionMMAQ4_1(
            KernelContext ctx,
            HalfFloatArray aFP16,
            ByteArray w,
            FloatArray out,
            int m,
            int n,
            int k) {
        int lane = ctx.localIdx;
        int colTiles = n / BN;
        int group = ctx.groupIdx;
        int rowTile = group / colTiles;
        int colTile = group - rowTile * colTiles;
        int blockRow = rowTile * BM;
        int blockCol = colTile * BN;
        int blocksPerRow = k / QK;

        // One allocation per operand, the second panel at a byte offset, as in
        // projectionMMAQ4_0. Same tile geometry and same addressing: a B panel's in-panel address
        // is at most 254 before the swizzle, which permutes within the same 256 bytes, and an A
        // panel's per-lane address reaches at most 496 of its 512. The A load applies no swizzle;
        // the B store and load apply it before adding the offset.
        int[] aTile = ctx.allocateIntLocalArray(2 * BM * BK / 2);
        HalfFloat[] bTile = ctx.allocateHalfFloatLocalArray(2 * PANEL * BK);

        float[] acc = ctx.mmaFragment(0.0f);

        int stageCol = lane >> 2;
        int stageFirst = (lane & 3) * 8;
        int stageByte = stageFirst & 15;
        boolean stageHighNibble = stageFirst >= 16;
        boolean stageHighHalf = stageFirst >= 16;
        int stageOffset = 0;
        if (stageHighHalf) {
            stageOffset = B_SUBTILE_BYTES;
        }
        int stageK = stageFirst & 15;

        int numBlocks = k / QK;
        for (int blockIndex = 0; blockIndex < numBlocks; blockIndex++) {
            int kBase = blockIndex * QK;

            for (int slot = 0; slot < 8; slot++) {
                int i = lane + slot * WARP_SIZE;
                int half = i >>> 7;
                int j = i & 127;
                int row = j >>> 3;
                int kk = (j & 7) << 1;
                int base = (blockRow + row) * k + kBase + half * BK + kk;
                // The same two adjacent halves into the same slot, packed the same way, copied
                // global-to-shared without the register round-trip. `base` is even -- the dispatch
                // guard makes k a whole number of blocks, and kBase, half * BK and kk are even --
                // so the source byte address is header + 2 * base and four-byte aligned.
                ctx.asyncCopyToLocal(aTile, half * (BM * BK / 2) + j, aFP16, base);
            }

            int base = ((blockCol + stageCol) * blocksPerRow + blockIndex) * BLOCK_BYTES_Q4_1;
            // Both fields through the array's own half accessor rather than assembled from two
            // bytes: the block stride is 20 and the fields sit at offsets 0 and 2, so every address
            // is two-byte aligned, and this lowers to one hardware conversion where halfFromBytes
            // lowers to a ten-branch software expansion. Same bytes, and the same value on this
            // little-endian CUDA target, where the byte pair and the native short agree.
            float scale = w.getHalfFloat(base).getFloat32();
            float minimum = w.getHalfFloat(base + 2).getFloat32();
            for (int t = 0; t < 8; t++) {
                int packed = w.get(base + 4 + stageByte + t) & 0xFF;
                int q = packed & 0xF;
                if (stageHighNibble) {
                    q = (packed >> 4) & 0xF;
                }
                HalfFloat value = new HalfFloat(scale * q + minimum);
                ctx.mmaStoreBSwizzled(bTile, stageK + t, stageCol, PANEL, value, stageOffset);
            }
            // Commit and wait before the barrier that publishes both tiles: every lane issues
            // its own eight copies -- i = lane + slot * 32 covers 0..255 exactly once across the
            // warp -- and every lane waits, so no MMA reads a slot whose copy is still in flight.
            // The trailing barrier below keeps the next round's copies out of a tile this round is
            // still reading.
            ctx.asyncCopyCommit();
            ctx.asyncCopyWaitGroup(0);
            ctx.localBarrier();

            acc =
                    ctx.mma(
                            ctx.mmaLoadA(aTile, BK, 0),
                            ctx.mmaLoadBSwizzled(bTile, BK, 0),
                            acc,
                            MMAShape.M16N8K16);
            acc =
                    ctx.mma(
                            ctx.mmaLoadA(aTile, BK, A_SUBTILE_BYTES),
                            ctx.mmaLoadBSwizzled(bTile, BK, B_SUBTILE_BYTES),
                            acc,
                            MMAShape.M16N8K16);
            ctx.localBarrier();
        }

        ctx.mmaStore(acc, out, blockRow, blockCol, n);
    }
}
