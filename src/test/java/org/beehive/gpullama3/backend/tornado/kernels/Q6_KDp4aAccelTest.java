package org.beehive.gpullama3.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.beehive.gpullama3.tensor.standard.Q6_KFloatTensor;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

// @formatter:off
/**
 * The packed-integer {@code Q6_K} projection, against the host tensor that defines the format.
 *
 * <p>Q6_K is the most intricate layout that reaches the device here: 256 weights per super-block
 * with one fp16 scale, sixteen signed byte scales, and each quantum split between a nibble of
 * {@code ql} and a pair of bits of {@code qh}, with which nibble and which pair depending on where
 * the element sits in the block. Getting that mapping wrong produces weights of plausible
 * magnitude, so the first case checks the <b>unpacking itself</b> rather than a dot product over
 * it.
 *
 * <p>The activation is one-hot, so each output is one decoded weight and can be held against {@link
 * Q6_KFloatTensor}. One row per position covers every group, both nibbles, every {@code qh} shift
 * and every scale index in the block; the weights carry all sixty-four raw values and both signs of
 * byte scale.
 */
// @formatter:on
public class Q6_KDp4aAccelTest {

    private static final int QK_K = 256;
    private static final int BLOCK_BYTES = 210;
    private static final int QH_OFFSET = 128;
    private static final int SCALES_OFFSET = 192;
    private static final int D_OFFSET = 208;
    private static final int QK = 32;
    private static final int LOCAL = 128;

    /** One super-block wide, one row per element in it. */
    private static final int N = QK_K;

    private static final int D = QK_K;

    /**
     * Weights covering all sixty-four raw values and both signs of scale.
     *
     * <p>Row {@code r} shifts the pattern so a given position sees a different raw value in every
     * row, and the byte scales alternate sign across the sixteen of a block.
     */
    private static byte[] weights() {
        byte[] raw = new byte[D * BLOCK_BYTES];
        for (int row = 0; row < D; row++) {
            int base = row * BLOCK_BYTES;
            // d: about 0.0625, positive, so the byte scales carry the sign.
            raw[base + D_OFFSET] = 0x00;
            raw[base + D_OFFSET + 1] = 0x2C;
            for (int i = 0; i < 16; i++) {
                int magnitude = 1 + ((row + i) % 7);
                raw[base + SCALES_OFFSET + i] = (byte) ((i % 2 == 0) ? magnitude : -magnitude);
            }
            // Every element of the block gets a raw value; the walk covers 0..63 as row varies.
            for (int e = 0; e < QK_K; e++) {
                int value = (e + row) % 64;
                writeQuantum(raw, base, e, value);
            }
        }
        return raw;
    }

    /** Places a six-bit value at element {@code e}, in the layout the host tensor reads. */
    private static void writeQuantum(byte[] raw, int base, int e, int value) {
        int half = e / 128;
        int posInHalf = e % 128;
        int groupInHalf = posInHalf / 32;
        int posInGroup = posInHalf % 32;
        int qlAt = base + half * 64 + ((groupInHalf & 1) * 32) + posInGroup;
        int qhAt = base + QH_OFFSET + half * 32 + posInGroup;
        int low = value & 0xF;
        int high = (value >> 4) & 3;
        if (groupInHalf >= 2) {
            raw[qlAt] = (byte) ((raw[qlAt] & 0x0F) | (low << 4));
        } else {
            raw[qlAt] = (byte) ((raw[qlAt] & 0xF0) | low);
        }
        int shift = 2 * groupInHalf;
        raw[qhAt] = (byte) ((raw[qhAt] & ~(3 << shift)) | (high << shift));
    }

    @Test
    public void everyQuantumUnpacksAsTheHostTensorReadsIt() throws Exception {
        byte[] raw = weights();

        // One-hot: row r selects element r, so out[r] is that row's weight r.
        float[] host = new float[N];
        FloatArray x = new FloatArray(N);
        IntArray quants = new IntArray(N / 4);
        FloatArray scales = new FloatArray(N / QK);
        IntArray sums = new IntArray(N / QK);
        FloatArray out = new FloatArray(D);
        ByteArray w = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            w.set(i, raw[i]);
        }

        float[] decoded = new float[D];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(raw.length);
            MemorySegment.copy(raw, 0, segment, ValueLayout.JAVA_BYTE, 0, raw.length);
            Q6_KFloatTensor reference = new Q6_KFloatTensor(D * N, segment);
            for (int row = 0; row < D; row++) {
                decoded[row] = reference.getFloat(row * N + row);
            }
        }

        for (int row = 0; row < D; row++) {
            java.util.Arrays.fill(host, 0.0f);
            host[row] = 1.0f;
            for (int i = 0; i < N; i++) {
                x.set(i, host[i]);
            }
            quants.init(0);
            scales.init(0.0f);
            sums.init(0);
            out.init(0.0f);

            TaskGraph graph =
                    new TaskGraph("q6k")
                            .transferToDevice(
                                    DataTransferMode.EVERY_EXECUTION,
                                    x,
                                    w,
                                    quants,
                                    scales,
                                    sums,
                                    out)
                            .task(
                                    "quantize",
                                    TransformerComputeKernelsQ4_0::quantizeActivationQ8Blocks,
                                    new KernelContext(),
                                    x,
                                    quants,
                                    scales,
                                    sums)
                            .task(
                                    "matvec",
                                    TransformerComputeKernelsQ6_K::matrixVectorGenericQ6_KDP4A,
                                    new KernelContext(),
                                    quants,
                                    scales,
                                    sums,
                                    out,
                                    w,
                                    N,
                                    D,
                                    LOCAL)
                            .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
            GridScheduler scheduler = new GridScheduler();
            WorkerGrid1D blocks = new WorkerGrid1D(N);
            blocks.setLocalWork(QK, 1, 1);
            scheduler.addWorkerGrid("q6k.quantize", blocks);
            WorkerGrid1D rows = new WorkerGrid1D(D * LOCAL);
            rows.setLocalWork(LOCAL, 1, 1);
            scheduler.addWorkerGrid("q6k.matvec", rows);
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
                plan.withGridScheduler(scheduler).execute();
            }

            float got = out.get(row);
            assertTrue("row " + row + " is " + got, Float.isFinite(got));
            // The one-hot block quantizes to a scale of 1/127 and a quant of 127, so the product
            // is the decoded weight exactly but for that round trip.
            assertEquals(
                    "element " + row + " of its own row, raw value " + ((row + row) % 64),
                    decoded[row],
                    got,
                    Math.max(1e-4f, Math.abs(decoded[row]) * 1e-3f));
        }
    }
}
