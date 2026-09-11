package org.beehive.gpullama3.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/** Minimal reproducer: is a private array inside a kernel zero-initialized, as Java requires? */
public class PrivateArrayZeroInitAccelTest {

    private static final int THREADS = 256;
    private static final int SLOTS = 8;

    /** Accumulates into a freshly allocated array without writing it first. */
    public static void accumulateWithoutZeroing(KernelContext context, FloatArray out) {
        int id = context.globalIdx;
        float[] acc = new float[SLOTS];
        for (int t = 0; t < SLOTS; t++) {
            acc[t] += 1.0f;
        }
        float sum = 0.0f;
        for (int t = 0; t < SLOTS; t++) {
            sum += acc[t];
        }
        out.set(id, sum);
    }

    @Test
    public void aFreshPrivateArrayIsZero() throws Exception {
        FloatArray out = new FloatArray(THREADS);
        out.init(-1.0f);

        TaskGraph graph = new TaskGraph("priv");
        graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, out);
        graph.task("k", PrivateArrayZeroInitAccelTest::accumulateWithoutZeroing,
                new KernelContext(), out);
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);

        WorkerGrid worker = new WorkerGrid1D(THREADS);
        worker.setLocalWork(64, 1, 1);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("priv.k", worker);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }

        for (int i = 0; i < THREADS; i++) {
            assertEquals("thread " + i, (float) SLOTS, out.get(i), 0.0f);
        }
    }
}
