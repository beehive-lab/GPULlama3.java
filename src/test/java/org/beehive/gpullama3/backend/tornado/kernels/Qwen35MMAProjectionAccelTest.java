package org.beehive.gpullama3.backend.tornado.kernels;

import static org.junit.Assert.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.beehive.gpullama3.format.GGMLType;
import org.beehive.gpullama3.tensor.standard.FloatTensor;
import org.beehive.gpullama3.tensor.standard.Q4_0FloatTensor;
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
}
