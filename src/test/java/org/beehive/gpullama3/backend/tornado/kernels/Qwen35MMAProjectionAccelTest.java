package org.beehive.gpullama3.backend.tornado.kernels;

import static org.junit.Assert.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.beehive.gpullama3.format.GGMLType;
import org.beehive.gpullama3.tensor.standard.FloatTensor;
import org.beehive.gpullama3.tensor.standard.Q4_0FloatTensor;
import org.beehive.gpullama3.tensor.standard.Q4_1FloatTensor;
import org.beehive.gpullama3.tensor.standard.Q5_KFloatTensor;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

// @formatter:off
/**
 * The tensor-core {@code Q4_0} projection against the host tensor's own dot product.
 *
 * <p>Deliberately not a bit-parity test. The kernel decodes the weights to FP16, converts the
 * activations to FP16 and accumulates in FP32 on the tensor core; the scalar path it replaces
 * multiplies in FP32. The numbers differ by what sixteen-bit multiplicands cost, and the bound
 * below is sized for that — a few parts in a thousand of the largest output. An addressing defect
 * in the tile packing, which is the thing that can actually go wrong here, lands orders of
 * magnitude outside it.
 */
// @formatter:on
public class Qwen35MMAProjectionAccelTest {

    /** K: a whole number of Q4_0 blocks, and of the 16-element MMA step. */
    private static final int K = 512;

    /** N: a whole number of the 128-column block. */
    private static final int N = 256;

    /** M: one and two MMA row tiles. */
    private static final int[] ROWS = {16, 32};

    private static byte[] randomWeights(long seed) {
        int blocks = N * (K / GGMLType.Q4_0.getBlockSize());
        byte[] raw = new byte[blocks * GGMLType.Q4_0.getTypeSize()];
        new Random(seed).nextBytes(raw);
        for (int b = 0; b < blocks; b++) {
            int base = b * GGMLType.Q4_0.getTypeSize();
            raw[base] = (byte) (b & 0xFF);
            raw[base + 1] = 0x2C; // ~0.06: finite, modest, different per block
        }
        return raw;
    }

    private static float activation(int row, int i) {
        return (float) Math.sin(0.31 * i + 0.7 * row);
    }

    @Test
    public void theTensorCoreProjectionMatchesTheHost() throws Exception {
        byte[] raw = randomWeights(20260910L);

        for (int m : ROWS) {
            HalfFloatArray a = new HalfFloatArray(m * K);
            for (int r = 0; r < m; r++) {
                for (int i = 0; i < K; i++) {
                    a.set(r * K + i, new HalfFloat(activation(r, i)));
                }
            }
            ByteArray w = new ByteArray(raw.length);
            for (int i = 0; i < raw.length; i++) {
                w.set(i, raw[i]);
            }
            FloatArray out = new FloatArray(m * N);
            out.init(0.0f);

            TaskGraph graph = new TaskGraph("mmaq40_" + m);
            graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, a, w, out);
            graph.task(
                    "projection",
                    Qwen35MMAKernels::projectionMMAQ4_0,
                    new KernelContext(),
                    a,
                    w,
                    out,
                    m,
                    N,
                    K);
            graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);

            int rowTiles = (m + Qwen35MMAKernels.BM - 1) / Qwen35MMAKernels.BM;
            int colBlocks = N / Qwen35MMAKernels.BN;
            WorkerGrid worker = new WorkerGrid1D(rowTiles * colBlocks * Qwen35MMAKernels.LOCAL);
            worker.setLocalWork(Qwen35MMAKernels.LOCAL, 1, 1);
            GridScheduler scheduler = new GridScheduler();
            scheduler.addWorkerGrid("mmaq40_" + m + ".projection", worker);
            try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
                plan.withGridScheduler(scheduler).execute();
            }

            float[] expected = new float[m * N];
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment segment = arena.allocate(raw.length);
                MemorySegment.copy(raw, 0, segment, ValueLayout.JAVA_BYTE, 0, raw.length);
                FloatTensor host = new Q4_0FloatTensor(N * K, segment);
                for (int r = 0; r < m; r++) {
                    for (int col = 0; col < N; col++) {
                        float sum = 0f;
                        for (int i = 0; i < K; i++) {
                            sum += host.getFloat(col * K + i) * activation(r, i);
                        }
                        expected[r * N + col] = sum;
                    }
                }
            }

            double largest = 0;
            double worst = 0;
            int worstAt = -1;
            for (int i = 0; i < m * N; i++) {
                largest = Math.max(largest, Math.abs(expected[i]));
            }
            for (int i = 0; i < m * N; i++) {
                double err = Math.abs(expected[i] - out.get(i));
                if (err > worst) {
                    worst = err;
                    worstAt = i;
                }
            }
            assertTrue(
                    "m="
                            + m
                            + " worst absolute error "
                            + worst
                            + " at row "
                            + (worstAt / N)
                            + " col "
                            + (worstAt % N)
                            + " (expected "
                            + expected[worstAt]
                            + ", got "
                            + out.get(worstAt)
                            + "), largest output "
                            + largest,
                    worst < 0.01 * largest);
        }
    }

    /** The fused gate/up form: two weight matrices against one staged activation tile. */
    @Test
    public void theTensorCoreGateUpMatchesTheHost() throws Exception {
        byte[] rawGate = randomWeights(20260910L);
        byte[] rawUp = randomWeights(777L);
        int m = 32;

        HalfFloatArray a = activations(m);
        ByteArray w1 = toDevice(rawGate);
        ByteArray w3 = toDevice(rawUp);
        FloatArray gateOut = new FloatArray(m * N);
        FloatArray upOut = new FloatArray(m * N);
        gateOut.init(0.0f);
        upOut.init(0.0f);

        TaskGraph graph = new TaskGraph("mmaGateUp");
        graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, a, w1, w3, gateOut, upOut);
        graph.task(
                "gateUp",
                Qwen35MMAKernels::projectionMMAQ4_0GateUp,
                new KernelContext(),
                a,
                w1,
                w3,
                gateOut,
                upOut,
                m,
                N,
                K);
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, gateOut, upOut);
        execute(graph, "mmaGateUp.gateUp", m);

        assertMatches("gate", GGMLType.Q4_0, rawGate, gateOut, m);
        assertMatches("up", GGMLType.Q4_0, rawUp, upOut, m);
    }

    /** The Q4_1 form — this model's ffn_down on the first eight blocks. */
    @Test
    public void theTensorCoreQ4_1ProjectionMatchesTheHost() throws Exception {
        int blocks = N * (K / GGMLType.Q4_1.getBlockSize());
        byte[] raw = new byte[blocks * GGMLType.Q4_1.getTypeSize()];
        new Random(99L).nextBytes(raw);
        for (int b = 0; b < blocks; b++) {
            int base = b * GGMLType.Q4_1.getTypeSize();
            for (int offset : new int[] {0, 2}) {
                raw[base + offset] = (byte) (b & 0xFF);
                raw[base + offset + 1] = 0x2C;
            }
        }
        int m = 32;

        HalfFloatArray a = activations(m);
        ByteArray w = toDevice(raw);
        FloatArray out = new FloatArray(m * N);
        out.init(0.0f);

        TaskGraph graph = new TaskGraph("mmaQ41");
        graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, a, w, out);
        graph.task(
                "projection",
                Qwen35MMAKernels::projectionMMAQ4_1,
                new KernelContext(),
                a,
                w,
                out,
                m,
                N,
                K);
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        execute(graph, "mmaQ41.projection", m);

        assertMatches("q41", GGMLType.Q4_1, raw, out, m);
    }

    /** The Q5_K form, against the host tensor over the same bytes. */
    @Test
    public void theTensorCoreQ5_KProjectionMatchesTheHost() throws Exception {
        int blocks = N * (K / GGMLType.Q5_K.getBlockSize());
        byte[] raw = new byte[blocks * GGMLType.Q5_K.getTypeSize()];
        new Random(4242L).nextBytes(raw);
        for (int b = 0; b < blocks; b++) {
            int base = b * GGMLType.Q5_K.getTypeSize();
            for (int offset : new int[] {0, 2}) {
                raw[base + offset] = (byte) (b & 0xFF);
                raw[base + offset + 1] = 0x2C;
            }
        }
        int m = 32;

        HalfFloatArray a = activations(m);
        ByteArray w = toDevice(raw);
        FloatArray out = new FloatArray(m * N);
        out.init(0.0f);

        TaskGraph graph = new TaskGraph("mmaQ5K");
        graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, a, w, out);
        graph.task(
                "projection",
                Qwen35MMAKernels::projectionMMAQ5_K,
                new KernelContext(),
                a,
                w,
                out,
                m,
                N,
                K);
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        execute(graph, "mmaQ5K.projection", m);

        assertMatches("q5k", GGMLType.Q5_K, raw, out, m);
    }

    private static HalfFloatArray activations(int m) {
        HalfFloatArray a = new HalfFloatArray(m * K);
        for (int r = 0; r < m; r++) {
            for (int i = 0; i < K; i++) {
                a.set(r * K + i, new HalfFloat(activation(r, i)));
            }
        }
        return a;
    }

    private static ByteArray toDevice(byte[] raw) {
        ByteArray w = new ByteArray(raw.length);
        for (int i = 0; i < raw.length; i++) {
            w.set(i, raw[i]);
        }
        return w;
    }

    private static void execute(TaskGraph graph, String qualifiedTask, int m) throws Exception {
        WorkerGrid worker =
                new WorkerGrid1D(
                        (m / Qwen35MMAKernels.BM)
                                * (N / Qwen35MMAKernels.BN)
                                * Qwen35MMAKernels.LOCAL);
        worker.setLocalWork(Qwen35MMAKernels.LOCAL, 1, 1);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid(qualifiedTask, worker);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }
    }

    private static void assertMatches(
            String what, GGMLType type, byte[] raw, FloatArray got, int m) {
        float[] expected = new float[m * N];
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment segment = arena.allocate(raw.length);
            MemorySegment.copy(raw, 0, segment, ValueLayout.JAVA_BYTE, 0, raw.length);
            FloatTensor host =
                    switch (type) {
                        case Q5_K -> new Q5_KFloatTensor(N * K, segment);
                        case Q4_1 -> new Q4_1FloatTensor(N * K, segment);
                        default -> new Q4_0FloatTensor(N * K, segment);
                    };
            for (int r = 0; r < m; r++) {
                for (int col = 0; col < N; col++) {
                    float sum = 0f;
                    for (int i = 0; i < K; i++) {
                        sum += host.getFloat(col * K + i) * activation(r, i);
                    }
                    expected[r * N + col] = sum;
                }
            }
        }
        double largest = 0;
        double worst = 0;
        int worstAt = -1;
        for (int i = 0; i < m * N; i++) {
            largest = Math.max(largest, Math.abs(expected[i]));
        }
        for (int i = 0; i < m * N; i++) {
            double err = Math.abs(expected[i] - got.get(i));
            if (err > worst) {
                worst = err;
                worstAt = i;
            }
        }
        assertTrue(
                what
                        + ": worst absolute error "
                        + worst
                        + " at row "
                        + (worstAt / N)
                        + " col "
                        + (worstAt % N)
                        + " (expected "
                        + expected[worstAt]
                        + ", got "
                        + got.get(worstAt)
                        + "), largest "
                        + largest,
                worst < 0.01 * largest);
    }
}
