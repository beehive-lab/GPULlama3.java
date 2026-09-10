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

        int[] aTileLo = ctx.allocateIntLocalArray(BM * BK / 2);
        int[] aTileHi = ctx.allocateIntLocalArray(BM * BK / 2);
        HalfFloat[] bTileLo = ctx.allocateHalfFloatLocalArray(PANEL * BK);
        HalfFloat[] bTileHi = ctx.allocateHalfFloatLocalArray(PANEL * BK);

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
                int value =
                        (aFP16.get(base).getHalfFloatValue() & 0xFFFF)
                                | ((aFP16.get(base + 1).getHalfFloatValue() & 0xFFFF) << 16);
                if (half == 0) {
                    aTileLo[j] = value;
                } else {
                    aTileHi[j] = value;
                }
            }

            // B: this lane's column, one scale, eight contiguous packed bytes.
            int base = ((blockCol + stageCol) * blocksPerRow + blockIndex) * BLOCK_BYTES;
            float scale = halfFromBytes(w, base);
            for (int t = 0; t < 8; t++) {
                int packed = w.get(base + 2 + stageByte + t) & 0xFF;
                int q = packed & 0xF;
                if (stageHighNibble) {
                    q = (packed >> 4) & 0xF;
                }
                HalfFloat value = new HalfFloat(scale * (q - 8));
                // (k index, column, columns per row) — the order the swizzled load expects.
                if (stageHighHalf) {
                    ctx.swizzleStoreFp16Stride32(bTileHi, stageK + t, stageCol, PANEL, value);
                } else {
                    ctx.swizzleStoreFp16Stride32(bTileLo, stageK + t, stageCol, PANEL, value);
                }
            }
            ctx.localBarrier();

            acc =
                    ctx.mma(
                            ctx.mmaLoadA(aTileLo, BK),
                            ctx.mmaLoadBSwizzled(bTileLo, BK),
                            acc,
                            MMAShape.M16N8K16);
            acc =
                    ctx.mma(
                            ctx.mmaLoadA(aTileHi, BK),
                            ctx.mmaLoadBSwizzled(bTileHi, BK),
                            acc,
                            MMAShape.M16N8K16);
            ctx.localBarrier();
        }

        ctx.mmaStore(acc, out, blockRow, blockCol, n);
    }

    // @formatter:off
    /**
     * The fused gate/up projection on the tensor cores: {@code gate = A x W1}, {@code up = A x W3},
     * both {@code Q4_0}, sharing one staged activation tile.
     *
     * <p>Two weight matrices against the same activations, so staging A once and keeping two
     * accumulators pays for itself directly. SwiGLU is not applied here — combining the two
     * accumulators elementwise would mean reading fragment elements, which the API does not offer —
     * so the two results are written out and {@link #swiGLUBatch} combines them, which is the same
     * split the FP16 families use between {@code gemmMMAGateUp} and {@code batchedFFNSwiGLU}.
     */
    // @formatter:on
    public static void projectionMMAQ4_0GateUp(
            KernelContext ctx,
            HalfFloatArray aFP16,
            ByteArray w1,
            ByteArray w3,
            FloatArray gateOut,
            FloatArray upOut,
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

        int[] aTileLo = ctx.allocateIntLocalArray(BM * BK / 2);
        int[] aTileHi = ctx.allocateIntLocalArray(BM * BK / 2);
        HalfFloat[] gateLo = ctx.allocateHalfFloatLocalArray(PANEL * BK);
        HalfFloat[] gateHi = ctx.allocateHalfFloatLocalArray(PANEL * BK);
        HalfFloat[] upLo = ctx.allocateHalfFloatLocalArray(PANEL * BK);
        HalfFloat[] upHi = ctx.allocateHalfFloatLocalArray(PANEL * BK);

        float[] accGate = ctx.mmaFragment(0.0f);
        float[] accUp = ctx.mmaFragment(0.0f);

        int stageCol = lane >> 2;
        int stageFirst = (lane & 3) * 8;
        int stageByte = stageFirst & 15;
        boolean stageHighNibble = stageFirst >= 16;
        boolean stageHighHalf = stageFirst >= 16;
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
                int value =
                        (aFP16.get(base).getHalfFloatValue() & 0xFFFF)
                                | ((aFP16.get(base + 1).getHalfFloatValue() & 0xFFFF) << 16);
                if (half == 0) {
                    aTileLo[j] = value;
                } else {
                    aTileHi[j] = value;
                }
            }

            int base = ((blockCol + stageCol) * blocksPerRow + blockIndex) * BLOCK_BYTES;
            float gateScale = halfFromBytes(w1, base);
            float upScale = halfFromBytes(w3, base);
            for (int t = 0; t < 8; t++) {
                int gatePacked = w1.get(base + 2 + stageByte + t) & 0xFF;
                int upPacked = w3.get(base + 2 + stageByte + t) & 0xFF;
                int gateQ = gatePacked & 0xF;
                int upQ = upPacked & 0xF;
                if (stageHighNibble) {
                    gateQ = (gatePacked >> 4) & 0xF;
                    upQ = (upPacked >> 4) & 0xF;
                }
                HalfFloat gateValue = new HalfFloat(gateScale * (gateQ - 8));
                HalfFloat upValue = new HalfFloat(upScale * (upQ - 8));
                if (stageHighHalf) {
                    ctx.swizzleStoreFp16Stride32(gateHi, stageK + t, stageCol, PANEL, gateValue);
                    ctx.swizzleStoreFp16Stride32(upHi, stageK + t, stageCol, PANEL, upValue);
                } else {
                    ctx.swizzleStoreFp16Stride32(gateLo, stageK + t, stageCol, PANEL, gateValue);
                    ctx.swizzleStoreFp16Stride32(upLo, stageK + t, stageCol, PANEL, upValue);
                }
            }
            ctx.localBarrier();

            HalfFloat[] fragALo = ctx.mmaLoadA(aTileLo, BK);
            HalfFloat[] fragAHi = ctx.mmaLoadA(aTileHi, BK);
            accGate =
                    ctx.mma(fragALo, ctx.mmaLoadBSwizzled(gateLo, BK), accGate, MMAShape.M16N8K16);
            accGate =
                    ctx.mma(fragAHi, ctx.mmaLoadBSwizzled(gateHi, BK), accGate, MMAShape.M16N8K16);
            accUp = ctx.mma(fragALo, ctx.mmaLoadBSwizzled(upLo, BK), accUp, MMAShape.M16N8K16);
            accUp = ctx.mma(fragAHi, ctx.mmaLoadBSwizzled(upHi, BK), accUp, MMAShape.M16N8K16);
            ctx.localBarrier();
        }

        ctx.mmaStore(accGate, gateOut, blockRow, blockCol, n);
        ctx.mmaStore(accUp, upOut, blockRow, blockCol, n);
    }

    /** {@code hb = silu(gate) * up} over the chunk. One lane per element. */
    public static void swiGLUBatch(
            KernelContext ctx, FloatArray gate, FloatArray up, FloatArray hb) {
        int index = ctx.globalIdx;
        float g = gate.get(index);
        hb.set(index, (g / (1.0f + TornadoMath.exp(-g))) * up.get(index));
    }
}
