package org.beehive.gpullama3.tornadovmissues;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * {@code (byte & 0xFF) >> 4 & 0xF} is stamped unsigned, and subtracting from it wraps.
 *
 * <p>Sixteen bytes holding nibble values 0..15 in their high half. Each is recentred by eight, as
 * every Q4_0 decode does, and the expected results are -8..7. Not part of the build.
 */
public class UnsignedNibbleRecentringAccelTest {

    /** The straight-line form: no branch between the shift and the subtraction. */
    public static void recentre(KernelContext ctx, ByteArray packed, FloatArray out) {
        int i = ctx.globalIdx;
        int q = (packed.get(i) & 0xFF) >> 4 & 0xF;
        out.set(i, q - 8);
    }

    @Test
    public void aRecentredHighNibbleIsSigned() throws Exception {
        ByteArray packed = new ByteArray(16);
        for (int i = 0; i < 16; i++) {
            packed.set(i, (byte) (i << 4));
        }
        FloatArray out = new FloatArray(16);
        out.init(0.0f);

        TaskGraph graph =
                new TaskGraph("nibble")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, packed, out)
                        .task(
                                "k",
                                UnsignedNibbleRecentringAccelTest::recentre,
                                new KernelContext(),
                                packed,
                                out)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out);
        WorkerGrid1D worker = new WorkerGrid1D(16);
        worker.setLocalWork(16, 1, 1);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("nibble.k", worker);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }

        for (int i = 0; i < 16; i++) {
            assertEquals("nibble " + i, (float) (i - 8), out.get(i), 0.0f);
        }
    }
}
