package org.beehive.jllm.backend.tornado.kernels;

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
     * two implementations that did read them were tried, one refusing to compile and one crashing
     * the compiler.
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
    /**
     * {@link #projectionMMAQ4_0} with each lane's eight packed nibble bytes read as four aligned
     * 16-bit words instead of eight bytes.
     *
     * <p>Everything else is the one-panel kernel: the same tile, grid, A staging, barriers, MMA
     * order, scale conversion and store. The bytes a lane decodes are the same bytes in the same
     * order — a word is its low byte then its high byte, little-endian, which is how {@code
     * getHalfFloat} on a {@code ByteArray} reads — so the nibbles, the decode and the accumulation
     * are unchanged and the output is bit-identical.
     *
     * <p>Alignment: a block is 18 bytes, so every block base is even; the packed bytes start at
     * {@code base + 2}; and a lane's first byte within them, {@code stageByte}, is 0 or 8. Each of
     * the four words therefore starts at an even offset and the last ends at {@code base + 2 +
     * stageByte + 7 <= base + 17}, inside the block.
     *
     * <p>One warp per output tile of sixteen rows by eight columns — the {@code m16n8k16} shape
     * itself. The B tile is staged through {@code swizzleStoreFp16Stride32} and read with {@code
     * mmaLoadBSwizzled}, which is the route that avoids ever reading a half's bits in the kernel;
     * two implementations that did read them were tried, one refusing to compile and one crashing
     * the compiler.
     *
     * @param aFP16 activations, {@code [M padded to 16][K]}, FP16
     * @param w the weight matrix, as the file stores it, {@code [N][K]} in {@code Q4_0}
     * @param out {@code [M][N]}, FP32
     */
    // @formatter:on
    public static void projectionMMAQ4_0Paired(
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
            int packedBase = base + 2 + stageByte;
            for (int pair = 0; pair < 4; pair++) {
                // Raw bits of bytes (2 * pair, 2 * pair + 1): the low byte is the first of the two.
                int word = w.getHalfFloat(packedBase + 2 * pair).getHalfFloatValue() & 0xFFFF;
                int packedLow = word & 0xFF;
                int qLow = packedLow & 0xF;
                if (stageHighNibble) {
                    qLow = (packedLow >> 4) & 0xF;
                }
                HalfFloat valueLow = new HalfFloat(scale * (qLow - 8));
                // (k index, column, columns per row) — the order the swizzled load expects.
                ctx.mmaStoreBSwizzled(
                        bTile, stageK + 2 * pair, stageCol, PANEL, valueLow, stageOffset);
                int packedHigh = (word >> 8) & 0xFF;
                int qHigh = packedHigh & 0xF;
                if (stageHighNibble) {
                    qHigh = (packedHigh >> 4) & 0xF;
                }
                HalfFloat valueHigh = new HalfFloat(scale * (qHigh - 8));
                ctx.mmaStoreBSwizzled(
                        bTile, stageK + 2 * pair + 1, stageCol, PANEL, valueHigh, stageOffset);
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
    /**
     * {@link #projectionMMAQ4_0Paired} with the next block's weight loads issued a round early.
     *
     * <p>Per lane the loads are a scale and four packed words; here they are hoisted into private
     * scalars one block ahead — a prologue loads block zero, and each round, having copied the
     * values it will decode, issues the loads for the block after it before staging its own tiles,
     * so that decode, the copy wait and the barrier can overlap the loads' latency if the compiler
     * keeps that order. The last round issues nothing: the guard is {@code blockIndex + 1 <
     * numBlocks}, so every block is loaded exactly once and nothing past the row's last block is
     * read. What is decoded, stored and multiplied is the paired kernel's in the paired kernel's
     * order, so the outputs are bit-identical.
     */
    // @formatter:on
    public static void projectionMMAQ4_0Prefetch(
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
        int laneBlockStride = BLOCK_BYTES;
        int laneRowBase = (blockCol + stageCol) * blocksPerRow * BLOCK_BYTES + 2 + stageByte;

        // Prologue: block zero's scale and packed words.
        int nextBase = (blockCol + stageCol) * blocksPerRow * BLOCK_BYTES;
        float scaleNext = w.getHalfFloat(nextBase).getFloat32();
        int wordNext0 = w.getHalfFloat(laneRowBase).getHalfFloatValue() & 0xFFFF;
        int wordNext1 = w.getHalfFloat(laneRowBase + 2).getHalfFloatValue() & 0xFFFF;
        int wordNext2 = w.getHalfFloat(laneRowBase + 4).getHalfFloatValue() & 0xFFFF;
        int wordNext3 = w.getHalfFloat(laneRowBase + 6).getHalfFloatValue() & 0xFFFF;

        for (int blockIndex = 0; blockIndex < numBlocks; blockIndex++) {
            int kBase = blockIndex * QK;

            // This round's operands, then the next block's loads before any of this round's
            // staging.
            float scale = scaleNext;
            int word0 = wordNext0;
            int word1 = wordNext1;
            int word2 = wordNext2;
            int word3 = wordNext3;
            if (blockIndex + 1 < numBlocks) {
                int base = nextBase + laneBlockStride;
                int packed = laneRowBase + (blockIndex + 1) * laneBlockStride;
                scaleNext = w.getHalfFloat(base).getFloat32();
                wordNext0 = w.getHalfFloat(packed).getHalfFloatValue() & 0xFFFF;
                wordNext1 = w.getHalfFloat(packed + 2).getHalfFloatValue() & 0xFFFF;
                wordNext2 = w.getHalfFloat(packed + 4).getHalfFloatValue() & 0xFFFF;
                wordNext3 = w.getHalfFloat(packed + 6).getHalfFloatValue() & 0xFFFF;
                nextBase = base;
            }

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

            // B: this lane's column from the values loaded a round ago, one word per pair.
            for (int pair = 0; pair < 4; pair++) {
                int word = word0;
                if (pair == 1) {
                    word = word1;
                } else if (pair == 2) {
                    word = word2;
                } else if (pair == 3) {
                    word = word3;
                }
                int packedLow = word & 0xFF;
                int qLow = packedLow & 0xF;
                if (stageHighNibble) {
                    qLow = (packedLow >> 4) & 0xF;
                }
                HalfFloat valueLow = new HalfFloat(scale * (qLow - 8));
                // (k index, column, columns per row) — the order the swizzled load expects.
                ctx.mmaStoreBSwizzled(
                        bTile, stageK + 2 * pair, stageCol, PANEL, valueLow, stageOffset);
                int packedHigh = (word >> 8) & 0xFF;
                int qHigh = packedHigh & 0xF;
                if (stageHighNibble) {
                    qHigh = (packedHigh >> 4) & 0xF;
                }
                HalfFloat valueHigh = new HalfFloat(scale * (qHigh - 8));
                ctx.mmaStoreBSwizzled(
                        bTile, stageK + 2 * pair + 1, stageCol, PANEL, valueHigh, stageOffset);
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

    // ---- Q4_0 dequantization to FP16 (experiment) --------------------------------

    /**
     * {@code out[row][e] = fp16(scale * (q - 8))} for a whole {@code Q4_0} matrix, one lane per
     * element, decoded with the expression the tensor-core kernels stage into their shared tiles so
     * the halves carry the same bits those kernels multiply.
     *
     * <p>The nibble is chosen by a branch on a lane-dependent value, as in the projection kernels:
     * with the choice foldable at compile time the high nibble's recentring is stamped unsigned and
     * every value below eight decodes to infinity (see
     * everyNibbleDecodesWithTheSignItsScaleGivesIt).
     *
     * <p>Worker: {@code n * k} lanes.
     */
    public static void dequantizeQ4_0ToFP16(
            KernelContext ctx, ByteArray w, HalfFloatArray out, int n, int k) {
        int lane = ctx.globalIdx;
        int blocksPerRow = k / QK;
        int row = lane / k;
        int element = lane - row * k;
        int block = element >> 5;
        int within = element & 31;
        int base = (row * blocksPerRow + block) * BLOCK_BYTES;
        float scale = w.getHalfFloat(base).getFloat32();
        int packed = w.get(base + 2 + (within & 15)) & 0xFF;
        int q = packed & 0xF;
        if (within >= 16) {
            q = (packed >> 4) & 0xF;
        }
        out.set(lane, new HalfFloat(scale * (q - 8)));
    }

    /**
     * {@code out[row][e] = fp16(scale * (q5 + high * 16) - minimum)} for a whole {@code Q5_K}
     * matrix, one lane per element: the sub-block's scale and minimum through {@link #scaleAndMin},
     * the low nibble chosen by a branch on the sub-block's parity, the high bit from the {@code qh}
     * plane — the expression {@link #projectionMMAQ5_KPaired} stages, so the halves carry the bits
     * it multiplies.
     *
     * <p>Worker: {@code n * k} lanes.
     */
    public static void dequantizeQ5_KToFP16(
            KernelContext ctx, ByteArray w, HalfFloatArray out, int n, int k) {
        int lane = ctx.globalIdx;
        int superBlocksPerRow = k / QK_K;
        int row = lane / k;
        int element = lane - row * k;
        int superBlock = element >> 8;
        int inSuper = element & 255;
        int subBlock = inSuper >> 5;
        int posInSub = inSuper & 31;
        int base = (row * superBlocksPerRow + superBlock) * K_BLOCK_BYTES;
        float d = w.getHalfFloat(base).getFloat32();
        float dmin = w.getHalfFloat(base + 2).getFloat32();
        int packedScale = scaleAndMin(w, base + K_SCALES_OFFSET, subBlock);
        float scale = d * (packedScale >> 8);
        float minimum = dmin * (packedScale & 0xFF);
        int pairIndex = subBlock >> 1;
        int highNibble = subBlock & 1;
        int qsByte = w.get(base + K_QS_OFFSET + pairIndex * 32 + posInSub) & 0xFF;
        int low = qsByte & 0xF;
        if (highNibble == 1) {
            low = (qsByte >> 4) & 0xF;
        }
        int qhByte = w.get(base + K_QH_OFFSET + posInSub) & 0xFF;
        int high = (qhByte >> (pairIndex * 2 + highNibble)) & 1;
        out.set(lane, new HalfFloat(scale * (low + high * 16) - minimum));
    }

    /**
     * {@code out[row][e] = fp16(scale * q + minimum)} for a whole {@code Q4_1} matrix, one lane per
     * element: the block's two halves, the unsigned nibble chosen by a branch on a lane-dependent
     * value (the branchless form is stamped unsigned and decodes wrongly, as the Q4_0 decoder
     * records), the expression {@link #projectionMMAQ4_1} stages, so the halves carry the bits it
     * multiplies.
     *
     * <p>Worker: {@code n * k} lanes.
     */
    public static void dequantizeQ4_1ToFP16(
            KernelContext ctx, ByteArray w, HalfFloatArray out, int n, int k) {
        int lane = ctx.globalIdx;
        int blocksPerRow = k / QK;
        int row = lane / k;
        int element = lane - row * k;
        int block = element >> 5;
        int within = element & 31;
        int base = (row * blocksPerRow + block) * BLOCK_BYTES_Q4_1;
        float scale = w.getHalfFloat(base).getFloat32();
        float minimum = w.getHalfFloat(base + 2).getFloat32();
        int packed = w.get(base + 4 + (within & 15)) & 0xFF;
        int q = packed & 0xF;
        if (within >= 16) {
            q = (packed >> 4) & 0xF;
        }
        out.set(lane, new HalfFloat(scale * q + minimum));
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

    // @formatter:off
    /**
     * {@link #projectionMMAQ5_K} with each lane's eight {@code qs} bytes and eight {@code qh} bytes
     * read as four aligned 16-bit words from each plane.
     *
     * <p>Everything else is the byte-load kernel: the tile, grid, A staging, barriers, MMA order,
     * {@code scaleAndMin}, the scale and minimum arithmetic, the nibble and high-bit selection and
     * the FP16 conversion. A word is its low byte then its high byte, little-endian, which is how
     * {@code getHalfFloat} on a {@code ByteArray} reads, so element {@code 2p} takes the low byte
     * and {@code 2p + 1} the high byte — the same bytes in the same order, so the output is
     * bit-identical.
     *
     * <p>Alignment and bounds: a super-block is 176 bytes, so every super-block base is even; the
     * {@code qs} plane starts at byte 48 and a sub-block pair's 32 bytes at {@code 48 + 32p}, the
     * {@code qh} plane at byte 16; a lane's first element {@code stageFirst} is 0, 8, 16 or 24; so
     * each word starts at an even offset, the last {@code qs} byte read is {@code 48 + 96 + 31 =
     * 175} and the last {@code qh} byte {@code 16 + 31 = 47}, both inside the super-block.
     */
    // @formatter:on
    public static void projectionMMAQ5_KPaired(
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

            for (int pair = 0; pair < 4; pair++) {
                // Q5_K indexes its packed byte by the element's position in the whole sub-block,
                // 0..31 — the nibble half is a property of the sub-block, not of the element. One
                // word of each plane covers elements 2 * pair and 2 * pair + 1 of this lane's
                // eight.
                int posInSub = stageFirst + 2 * pair;
                int qsWord = w.getHalfFloat(qsBase + posInSub).getHalfFloatValue() & 0xFFFF;
                int qhWord = w.getHalfFloat(qhBase + posInSub).getHalfFloatValue() & 0xFFFF;

                int qsByte = qsWord & 0xFF;
                int low = qsByte & 0xF;
                if (highNibble == 1) {
                    low = (qsByte >> 4) & 0xF;
                }
                int qhByte = qhWord & 0xFF;
                int high = (qhByte >> bitShift) & 1;
                HalfFloat value = new HalfFloat(scale * (low + high * 16) - minimum);
                ctx.mmaStoreBSwizzled(
                        bTile, stageK + 2 * pair, stageCol, PANEL, value, stageOffset);

                int qsByteHigh = (qsWord >> 8) & 0xFF;
                int lowHigh = qsByteHigh & 0xF;
                if (highNibble == 1) {
                    lowHigh = (qsByteHigh >> 4) & 0xF;
                }
                int qhByteHigh = (qhWord >> 8) & 0xFF;
                int highHigh = (qhByteHigh >> bitShift) & 1;
                HalfFloat valueHigh = new HalfFloat(scale * (lowHigh + highHigh * 16) - minimum);
                ctx.mmaStoreBSwizzled(
                        bTile, stageK + 2 * pair + 1, stageCol, PANEL, valueHigh, stageOffset);
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

    // ---- Q4_0 decoded straight into the GEMM's B-tile order (experiment) ---------------

    /** The tiled GEMM's geometry, restated here so this file's loader agrees with it by name. */
    private static final int GEMM_BM = 128;

    private static final int GEMM_BN = 128;
    private static final int GEMM_BK = 16;
    private static final int GEMM_WARPS_N = 2;
    private static final int GEMM_WM = 32;
    private static final int GEMM_WN = 64;

    /** Ints in one B tile: {@code GEMM_BK * GEMM_BN / 2}. */
    private static final int GEMM_B_TILE_INTS = GEMM_BK * GEMM_BN / 2;

    // @formatter:off
    /**
     * {@code out = fp16(scale * (q - 8))} for a whole {@code Q4_0} matrix, one lane per element,
     * the decode expression of {@link #dequantizeQ4_0ToFP16}, written not row-major but in the
     * order the tiled GEMM stages its B tile in shared memory, so the GEMM can copy each tile
     * global-to-shared as contiguous four-byte words.
     *
     * <p><b>Layout.</b> The matrix is {@code n} rows (output columns of the projection) by {@code
     * k}. It is cut into tiles of 128 rows by 16 k, numbered {@code tile = (row / 128) * (k / 16) +
     * kk / 16}: all of a row block's k-steps in order, then the next row block. A tile holds 1024
     * packed pairs; pair {@code idx} (0..1023) holds rows {@code (idx >>> 6) * 8 + (idx & 3) * 2}
     * and that plus one at k {@code (idx & 63) >>> 2} within the tile — the index {@code gemmMMA}
     * gives {@code bTile[idx]} — with the even row in the low half. Half position {@code h = tile *
     * 2048 + idx * 2 + (row & 1)}.
     *
     * <p><b>Inverse.</b> From {@code h}: {@code idx = (h >>> 1) & 1023}, {@code tile = h >>> 11},
     * {@code row = (tile / (k / 16)) * 128 + ((idx >>> 6) << 3) + ((idx & 3) << 1) + (h & 1)},
     * {@code kk = (tile % (k / 16)) * 16 + ((idx & 63) >>> 2)}. Every half position maps to one
     * element and back; {@code n % 128 == 0} and {@code k % 16 == 0}, as the GEMM requires.
     *
     * <p>Worker: {@code n * k} lanes; the lane order is a fixed permutation of the half positions
     * (see the body), so every half is written exactly once.
     */
    // @formatter:on
    public static void dequantizeQ4_0ToFP16Tiled(
            KernelContext ctx, ByteArray w, HalfFloatArray out, int n, int k) {
        int lane = ctx.globalIdx;
        int kSteps = k / GEMM_BK;
        // Lane to half position: not the identity. A warp of 32 lanes covers four rows by eight k
        // (four blocks read, four 32-byte sectors written) rather than the eight rows by four k
        // the identity would give (eight blocks read, two sectors written); measured 2-3% faster
        // over the production shapes. Lane bits, low to high: row parity, the low pair bit, three
        // low k bits, the high pair bit, the high k bit, then the sub-tile and tile.
        int parity = lane & 1;
        int pairInSub = ((lane >>> 1) & 1) | (((lane >>> 5) & 1) << 1);
        int kk = ((lane >>> 2) & 7) | (((lane >>> 6) & 1) << 3);
        int pair = ((lane >>> 7) << 6) + (kk << 2) + pairInSub;
        int tile = pair >>> 10;
        int idx = pair & (GEMM_B_TILE_INTS - 1);
        int rowBlock = tile / kSteps;
        int kStep = tile - rowBlock * kSteps;
        int row = rowBlock * GEMM_BN + ((idx >>> 6) << 3) + ((idx & 3) << 1) + parity;
        int element = kStep * GEMM_BK + ((idx & 63) >>> 2);
        int blocksPerRow = k / QK;
        int block = element >> 5;
        int within = element & 31;
        int base = (row * blocksPerRow + block) * BLOCK_BYTES;
        float scale = w.getHalfFloat(base).getFloat32();
        int packed = w.get(base + 2 + (within & 15)) & 0xFF;
        int q = packed & 0xF;
        if (within >= 16) {
            q = (packed >> 4) & 0xF;
        }
        out.set((pair << 1) + parity, new HalfFloat(scale * (q - 8)));
    }

    // @formatter:off
    /**
     * {@link #dequantizeQ4_0ToFP16Tiled} with both nibbles of a packed byte decoded by one lane:
     * one scale and one byte read, two halves written, in the same tiled layout.
     *
     * <p><b>Address mapping.</b> Byte {@code t} (0..15) of block {@code b} of row {@code r} holds
     * elements {@code 32b + t} (low nibble) and {@code 32b + t + 16} (high nibble): the same row,
     * the same position {@code t} within a 16-wide k-tile, in k-tiles {@code 2b} and {@code 2b +
     * 1}. In the tiled layout those are tiles {@code T} and {@code T + 1} of the same row block
     * ({@code K % 32 == 0}, so a row block's tiles come in whole pairs) at the same in-tile index,
     * so the high half's position is the low half's plus 2048. A lane therefore enumerates the low
     * halves — every half position whose k-tile is even, in the retained decoder's lane order with
     * the tile-parity bit removed — and writes the high half at {@code + 2048}. Lane bits, low to
     * high: row parity, the low pair bit, three low k bits, the high pair bit, the high k bit, the
     * sub-tile (four bits), then the tile pair. {@code n * k / 2} lanes cover every low half once,
     * every high half once, and the largest address written is {@code n * k - 1}.
     *
     * <p><b>Arithmetic.</b> Both elements are {@code fp16(scale * (q - 8))} with the same scale,
     * each nibble recentred as an int and converted independently; no packed arithmetic. The high
     * nibble is taken as {@code (packed & 0xF0) >>> 4} rather than {@code (packed >>> 4) & 0xF}:
     * the latter is stamped unsigned by the CUDA lowering and its recentring decodes to infinity
     * (see everyNibbleDecodesWithTheSignItsScaleGivesIt); the test checks every nibble's bits.
     *
     * <p>Worker: {@code n * k / 2} lanes.
     */
    // @formatter:on
    public static void dequantizeQ4_0ToFP16TiledPairs(
            KernelContext ctx, ByteArray w, HalfFloatArray out, int n, int k) {
        int lane = ctx.globalIdx;
        int kSteps = k / GEMM_BK;
        int parity = lane & 1;
        int pairInSub = ((lane >>> 1) & 1) | (((lane >>> 5) & 1) << 1);
        int kk = ((lane >>> 2) & 7) | (((lane >>> 6) & 1) << 3);
        int sub = (lane >>> 7) & 15;
        int tilePair = lane >>> 11;
        int tile = tilePair << 1;
        int idx = (sub << 6) + (kk << 2) + pairInSub;
        int rowBlock = tile / kSteps;
        int kStep = tile - rowBlock * kSteps;
        int row = rowBlock * GEMM_BN + (sub << 3) + (pairInSub << 1) + parity;
        int element = kStep * GEMM_BK + kk;
        int blocksPerRow = k / QK;
        int block = element >> 5;
        int within = element & 15;
        int base = (row * blocksPerRow + block) * BLOCK_BYTES;
        float scale = w.getHalfFloat(base).getFloat32();
        int packed = w.get(base + 2 + within) & 0xFF;
        int low = packed & 0xF;
        int high = (packed & 0xF0) >>> 4;
        int lowHalf = (tile << 11) + (idx << 1) + parity;
        out.set(lowHalf, new HalfFloat(scale * (low - 8)));
        out.set(lowHalf + 2 * GEMM_B_TILE_INTS, new HalfFloat(scale * (high - 8)));
    }

    /** {@code lo | hi << 16} of two halves: the packing of the GEMM's shared tiles. */
    private static int packHalvesGemm(HalfFloatArray src, int idxLo, int idxHi) {
        int lo = src.get(idxLo).getHalfFloatValue() & 0xFFFF;
        int hi = src.get(idxHi).getHalfFloatValue() & 0xFFFF;
        return lo | (hi << 16);
    }

    // @formatter:off
    /**
     * {@code TransformerBatchPrefillKernels.gemmMMA} with its B operand in the tile order {@link
     * #dequantizeQ4_0ToFP16Tiled} writes: {@code C[M,N] (FP32) = A[M,K] (FP16, row-major) x B},
     * where B's tile {@code (blockCol / 128) * (K / 16) + kStep} is the 1024 ints of this K-step's
     * shared tile in shared-tile order. The A staging, the fragment loads, the MMA sequence, the
     * FP32 accumulation, the store and the tile geometry are those of {@code gemmMMA}; only the B
     * staging differs: each lane's four ints of the tile are copied global-to-shared with {@code
     * cp.async}, four-byte words from contiguous addresses, in place of eight two-byte loads a K
     * apart and four shared stores through registers.
     *
     * <p>Same synchronisation shape: the next step's B copies are issued after the barrier that
     * ends this step's fragment loads, overlap the MMAs, and are waited for before the barrier that
     * publishes the tile.
     *
     * <p>Requires M % 128 == 0, N % 128 == 0, K % 16 == 0. Worker: WorkerGrid2D((M/128)*256,
     * N/128), local (256,1,1).
     */
    // @formatter:on
    public static void gemmMMATiledB(
            KernelContext ctx,
            HalfFloatArray A,
            HalfFloatArray B,
            FloatArray C,
            int M,
            int N,
            int K) {
        int tid = ctx.localIdx;
        int warpId = tid / WARP_SIZE;
        int warpM = warpId / GEMM_WARPS_N;
        int warpN = warpId % GEMM_WARPS_N;
        int blockRow = GEMM_BM * ctx.groupIdx;
        int blockCol = GEMM_BN * ctx.groupIdy;

        int[] aTile = ctx.allocateIntLocalArray(GEMM_BM * GEMM_BK / 2);
        // Two B tiles: the copy for the next step lands in the one this step is not reading.
        int[] bTile = ctx.allocateIntLocalArray(2 * GEMM_B_TILE_INTS);

        float[] c00 = ctx.mmaFragment(0.0f);
        float[] c01 = ctx.mmaFragment(0.0f);
        float[] c02 = ctx.mmaFragment(0.0f);
        float[] c03 = ctx.mmaFragment(0.0f);
        float[] c04 = ctx.mmaFragment(0.0f);
        float[] c05 = ctx.mmaFragment(0.0f);
        float[] c06 = ctx.mmaFragment(0.0f);
        float[] c07 = ctx.mmaFragment(0.0f);
        float[] c10 = ctx.mmaFragment(0.0f);
        float[] c11 = ctx.mmaFragment(0.0f);
        float[] c12 = ctx.mmaFragment(0.0f);
        float[] c13 = ctx.mmaFragment(0.0f);
        float[] c14 = ctx.mmaFragment(0.0f);
        float[] c15 = ctx.mmaFragment(0.0f);
        float[] c16 = ctx.mmaFragment(0.0f);
        float[] c17 = ctx.mmaFragment(0.0f);

        int aIdx0 = tid;
        int gA0 = (blockRow + (aIdx0 >>> 3)) * K + ((aIdx0 & 7) << 1);
        int aIdx1 = tid + 256;
        int gA1 = (blockRow + (aIdx1 >>> 3)) * K + ((aIdx1 & 7) << 1);
        int aIdx2 = tid + 512;
        int gA2 = (blockRow + (aIdx2 >>> 3)) * K + ((aIdx2 & 7) << 1);
        int aIdx3 = tid + 768;
        int gA3 = (blockRow + (aIdx3 >>> 3)) * K + ((aIdx3 & 7) << 1);
        // B: this block's column tile sequence starts at pair (groupIdy * (K / 16)) * 1024; each
        // K-step's tile is the next 1024 pairs, and shared int idx is global half 2 * (base + idx).
        int numKSteps = K / GEMM_BK;
        int bTileBase = ctx.groupIdy * numKSteps * GEMM_B_TILE_INTS;
        int bIdx0 = tid;
        int bIdx1 = tid + 256;
        int bIdx2 = tid + 512;
        int bIdx3 = tid + 768;

        int aReg0 = packHalvesGemm(A, gA0, gA0 + 1);
        int aReg1 = packHalvesGemm(A, gA1, gA1 + 1);
        int aReg2 = packHalvesGemm(A, gA2, gA2 + 1);
        int aReg3 = packHalvesGemm(A, gA3, gA3 + 1);
        aTile[aIdx0] = aReg0;
        aTile[aIdx1] = aReg1;
        aTile[aIdx2] = aReg2;
        aTile[aIdx3] = aReg3;
        int gB = (bTileBase) << 1;
        ctx.asyncCopyToLocal(bTile, bIdx0, B, gB + (bIdx0 << 1));
        ctx.asyncCopyToLocal(bTile, bIdx1, B, gB + (bIdx1 << 1));
        ctx.asyncCopyToLocal(bTile, bIdx2, B, gB + (bIdx2 << 1));
        ctx.asyncCopyToLocal(bTile, bIdx3, B, gB + (bIdx3 << 1));
        ctx.asyncCopyCommit();
        ctx.asyncCopyWaitGroup(0);
        ctx.localBarrier();

        for (int kStep = 0; kStep < numKSteps; kStep++) {
            int bufThis = (kStep & 1) * GEMM_B_TILE_INTS;
            int bufNext = GEMM_B_TILE_INTS - bufThis;
            if (kStep + 1 < numKSteps) {
                int kOff = (kStep + 1) * GEMM_BK;
                aReg0 = packHalvesGemm(A, gA0 + kOff, gA0 + kOff + 1);
                aReg1 = packHalvesGemm(A, gA1 + kOff, gA1 + kOff + 1);
                aReg2 = packHalvesGemm(A, gA2 + kOff, gA2 + kOff + 1);
                aReg3 = packHalvesGemm(A, gA3 + kOff, gA3 + kOff + 1);
                // The other B buffer was last read a step ago, before that step's closing
                // barrier: free. Issue now, so the copy overlaps this whole step.
                int gBNext = (bTileBase + (kStep + 1) * GEMM_B_TILE_INTS) << 1;
                ctx.asyncCopyToLocal(bTile, bufNext + bIdx0, B, gBNext + (bIdx0 << 1));
                ctx.asyncCopyToLocal(bTile, bufNext + bIdx1, B, gBNext + (bIdx1 << 1));
                ctx.asyncCopyToLocal(bTile, bufNext + bIdx2, B, gBNext + (bIdx2 << 1));
                ctx.asyncCopyToLocal(bTile, bufNext + bIdx3, B, gBNext + (bIdx3 << 1));
                ctx.asyncCopyCommit();
            }

            int aOff0 = warpM * 1024;
            int aOff1 = warpM * 1024 + 512;
            HalfFloat[] a0 = ctx.mmaLoadA(aTile, GEMM_BK, aOff0);
            HalfFloat[] a1 = ctx.mmaLoadA(aTile, GEMM_BK, aOff1);
            int bBase = warpN * 8 + (bufThis >>> 6);
            HalfFloat[] b0 = ctx.mmaLoadB(bTile, GEMM_BK, (bBase + 0) * B_SUBTILE_BYTES);
            HalfFloat[] b1 = ctx.mmaLoadB(bTile, GEMM_BK, (bBase + 1) * B_SUBTILE_BYTES);
            HalfFloat[] b2 = ctx.mmaLoadB(bTile, GEMM_BK, (bBase + 2) * B_SUBTILE_BYTES);
            HalfFloat[] b3 = ctx.mmaLoadB(bTile, GEMM_BK, (bBase + 3) * B_SUBTILE_BYTES);
            HalfFloat[] b4 = ctx.mmaLoadB(bTile, GEMM_BK, (bBase + 4) * B_SUBTILE_BYTES);
            HalfFloat[] b5 = ctx.mmaLoadB(bTile, GEMM_BK, (bBase + 5) * B_SUBTILE_BYTES);
            HalfFloat[] b6 = ctx.mmaLoadB(bTile, GEMM_BK, (bBase + 6) * B_SUBTILE_BYTES);
            HalfFloat[] b7 = ctx.mmaLoadB(bTile, GEMM_BK, (bBase + 7) * B_SUBTILE_BYTES);
            ctx.localBarrier();

            if (kStep + 1 < numKSteps) {
                aTile[aIdx0] = aReg0;
                aTile[aIdx1] = aReg1;
                aTile[aIdx2] = aReg2;
                aTile[aIdx3] = aReg3;
            }

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
            ctx.asyncCopyWaitGroup(0);
            ctx.localBarrier();
        }

        int rBase = blockRow + warpM * GEMM_WM;
        int cBase = blockCol + warpN * GEMM_WN;
        ctx.mmaStore(c00, C, rBase + 0, cBase + 0, N);
        ctx.mmaStore(c01, C, rBase + 0, cBase + 8, N);
        ctx.mmaStore(c02, C, rBase + 0, cBase + 16, N);
        ctx.mmaStore(c03, C, rBase + 0, cBase + 24, N);
        ctx.mmaStore(c04, C, rBase + 0, cBase + 32, N);
        ctx.mmaStore(c05, C, rBase + 0, cBase + 40, N);
        ctx.mmaStore(c06, C, rBase + 0, cBase + 48, N);
        ctx.mmaStore(c07, C, rBase + 0, cBase + 56, N);
        ctx.mmaStore(c10, C, rBase + 16, cBase + 0, N);
        ctx.mmaStore(c11, C, rBase + 16, cBase + 8, N);
        ctx.mmaStore(c12, C, rBase + 16, cBase + 16, N);
        ctx.mmaStore(c13, C, rBase + 16, cBase + 24, N);
        ctx.mmaStore(c14, C, rBase + 16, cBase + 32, N);
        ctx.mmaStore(c15, C, rBase + 16, cBase + 40, N);
        ctx.mmaStore(c16, C, rBase + 16, cBase + 48, N);
        ctx.mmaStore(c17, C, rBase + 16, cBase + 56, N);
    }
}
