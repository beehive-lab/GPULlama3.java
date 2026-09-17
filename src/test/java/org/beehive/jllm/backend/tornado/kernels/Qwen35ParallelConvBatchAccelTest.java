package org.beehive.jllm.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * {@code causalConv1dBatch} + {@code causalConv1dWindowUpdate} against {@code causalConv1dScan}:
 * outputs and the final window raw-bit equal over chunks of 1, 2, 3, 300 and 2048 active rows in a
 * 2048-row buffer (inactive rows NaN-poisoned and untouched by both), a nonzero window offset,
 * random taps and a random initial window. A negative control (the update run before the
 * convolution) differs.
 */
public class Qwen35ParallelConvBatchAccelTest {

    private static final int CHANNELS = 10240;
    private static final int KERNEL = 4;
    private static final int ROWS = 2048;

    private static FloatArray random(int n, long seed) {
        FloatArray a = new FloatArray(n);
        Random rng = new Random(seed);
        for (int i = 0; i < n; i++) {
            a.set(i, rng.nextFloat() * 2.0f - 1.0f);
        }
        return a;
    }

    private static FloatArray input(int active, long seed) {
        FloatArray a = random(ROWS * CHANNELS, seed);
        for (int i = active * CHANNELS; i < ROWS * CHANNELS; i++) {
            a.set(i, Float.NaN);
        }
        return a;
    }

    private static int mismatches(String what, FloatArray a, FloatArray b) {
        int n = 0;
        String first = null;
        for (int i = 0; i < a.getSize(); i++) {
            if (Float.floatToRawIntBits(a.get(i)) != Float.floatToRawIntBits(b.get(i))) {
                if (first == null) {
                    first = "index " + i + ": " + a.get(i) + " vs " + b.get(i);
                }
                n++;
            }
        }
        if (n > 0) {
            System.out.println(what + ": " + n + " differ, first at " + first);
        }
        return n;
    }

    private static WorkerGrid lanes(int count, int local) {
        WorkerGrid g = new WorkerGrid1D(count);
        g.setLocalWork(local, 1, 1);
        return g;
    }

    /** Returns {outputs differing, window slots differing}. */
    private static int[] compare(int active, boolean updateFirst) throws Exception {
        int offset = 3 * CHANNELS * (KERNEL - 1);
        int windowSize = offset + CHANNELS * (KERNEL - 1) + 7;
        FloatArray in = input(active, 11L + active);
        FloatArray taps = random(CHANNELS * KERNEL, 5L);
        FloatArray window1 = random(windowSize, 6L);
        FloatArray window2 = random(windowSize, 6L);
        FloatArray out1 = new FloatArray(ROWS * CHANNELS);
        FloatArray out2 = new FloatArray(ROWS * CHANNELS);
        out1.init(Float.NaN);
        out2.init(Float.NaN);
        IntArray info = new IntArray(4);
        info.set(1, active);
        TaskGraph g =
                new TaskGraph("cv")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                in,
                                taps,
                                window1,
                                window2,
                                out1,
                                out2,
                                info)
                        .task(
                                "scan",
                                Qwen35BatchKernels::causalConv1dScan,
                                new KernelContext(),
                                in,
                                taps,
                                window1,
                                out1,
                                CHANNELS,
                                KERNEL,
                                offset,
                                info);
        if (updateFirst) {
            g.task(
                    "window",
                    Qwen35BatchKernels::causalConv1dWindowUpdate,
                    new KernelContext(),
                    in,
                    window2,
                    CHANNELS,
                    KERNEL,
                    offset,
                    info);
        }
        g.task(
                "batch",
                Qwen35BatchKernels::causalConv1dBatch,
                new KernelContext(),
                in,
                taps,
                window2,
                out2,
                CHANNELS,
                KERNEL,
                offset,
                info);
        if (!updateFirst) {
            g.task(
                    "window",
                    Qwen35BatchKernels::causalConv1dWindowUpdate,
                    new KernelContext(),
                    in,
                    window2,
                    CHANNELS,
                    KERNEL,
                    offset,
                    info);
        }
        g.transferToHost(DataTransferMode.EVERY_EXECUTION, window1, window2, out1, out2);
        GridScheduler s = new GridScheduler();
        s.addWorkerGrid("cv.scan", lanes(CHANNELS, 128));
        s.addWorkerGrid("cv.batch", lanes(ROWS * CHANNELS, 128));
        s.addWorkerGrid("cv.window", lanes(CHANNELS, 128));
        try (TornadoExecutionPlan p = new TornadoExecutionPlan(g.snapshot())) {
            p.withGridScheduler(s).execute();
        }
        for (int i = 0; i < active * CHANNELS; i++) {
            assertTrue("scan output not finite at " + i, !Float.isNaN(out1.get(i)));
        }
        for (int i = active * CHANNELS; i < ROWS * CHANNELS; i++) {
            assertTrue("batch wrote an inactive row at " + i, Float.isNaN(out2.get(i)));
        }
        return new int[] {
            mismatches("outputs active=" + active, out1, out2),
            mismatches("window active=" + active, window1, window2)
        };
    }

    @Test
    public void outputsAndWindowAreRawBitEqualToTheScan() throws Exception {
        for (int active : new int[] {1, 2, 3, 300, 2048}) {
            int[] d = compare(active, false);
            assertEquals("outputs differ at " + active + " rows", 0, d[0]);
            assertEquals("window differs at " + active + " rows", 0, d[1]);
        }
    }

    @Test
    public void theWindowUpdateBeforeTheConvolutionDiffers() throws Exception {
        int[] d = compare(300, true);
        assertTrue("updating the window first agreed", d[0] > 0);
    }
}
