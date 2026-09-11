package org.beehive.gpullama3.tornadovmissues;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.MMAShape;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * SCRATCH — is an MMA accumulator fragment readable at a constant index from Java?
 *
 * <p>The question this answers: can a per-K-block scale be applied to the accumulator between MMA
 * steps, the way llama.cpp's integer MMQ does. If it can, integer MMA over native Q4_0 is
 * expressible; if it cannot, only a store/reload round-trip is.
 *
 * <p>Answer, on 6.0.1-jdk21-dev: neither. Any Java-level index of the value {@code ctx.mma}
 * returns fails to lower, reads and writes alike. Not part of the build; see the README.
 */
public class MmaFragmentAccessAccelTest {

    private static final int M = 16;
    private static final int N = 8;
    private static final int K = 16;

    /** out = (A x B) * 2, with the doubling applied to the fragment rather than to the store. */
    public static void scaleFragment(
            KernelContext ctx, HalfFloatArray a, HalfFloatArray b, FloatArray out) {
        int lane = ctx.localIdx;
        int[] aTile = ctx.allocateIntLocalArray(M * K / 2);
        HalfFloat[] bTile = ctx.allocateHalfFloatLocalArray(N * K);

        for (int slot = 0; slot < 4; slot++) {
            int i = lane + slot * 32;
            int row = i >>> 3;
            int kk = (i & 7) << 1;
            int base = row * K + kk;
            aTile[i] =
                    (a.get(base).getHalfFloatValue() & 0xFFFF)
                            | ((a.get(base + 1).getHalfFloatValue() & 0xFFFF) << 16);
        }
        for (int slot = 0; slot < 4; slot++) {
            int i = lane + slot * 32;
            int col = i >>> 4;
            int kk = i & 15;
            ctx.swizzleStoreFp16Stride32(bTile, kk, col, N, b.get(col * K + kk));
        }
        ctx.localBarrier();

        float[] acc = ctx.mmaFragment(0.0f);
        acc = ctx.mma(ctx.mmaLoadA(aTile, K), ctx.mmaLoadBSwizzled(bTile, K), acc, MMAShape.M16N8K16);

        // THE PROBE: constant-index READS of the accumulator, written to plain global memory.
        out.set(lane * 4, acc[0] * 2.0f);
        out.set(lane * 4 + 1, acc[1] * 2.0f);
        out.set(lane * 4 + 2, acc[2] * 2.0f);
        out.set(lane * 4 + 3, acc[3] * 2.0f);
    }

    @Test
    public void anAccumulatorFragmentCanBeReadAtAConstantIndex() throws Exception {
        HalfFloatArray a = new HalfFloatArray(M * K);
        HalfFloatArray b = new HalfFloatArray(N * K);
        float[] ah = new float[M * K];
        float[] bh = new float[N * K];
        for (int i = 0; i < M * K; i++) {
            ah[i] = ((i * 7) % 13 - 6) * 0.25f;
            a.set(i, new HalfFloat(ah[i]));
        }
        for (int i = 0; i < N * K; i++) {
            bh[i] = ((i * 5) % 11 - 5) * 0.5f;
            b.set(i, new HalfFloat(bh[i]));
        }
        FloatArray out = new FloatArray(M * N);
        out.init(0.0f);

        KernelContext ctx = new KernelContext();
        TaskGraph graph =
                new TaskGraph("probe")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, a, b, out)
                        .task("k", MmaFragmentAccessAccelTest::scaleFragment, ctx, a, b, out)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out);

        WorkerGrid1D worker = new WorkerGrid1D(32);
        worker.setLocalWork(32, 1, 1);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("probe.k", worker);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }

        for (int lane = 0; lane < 32; lane++) {
            for (int i = 0; i < 4; i++) {
                int m = lane / 4 + 8 * (i / 2);
                int n = (lane % 4) * 2 + (i % 2);
                float expected = 0;
                for (int k = 0; k < K; k++) {
                    expected += ah[m * K + k] * bh[n * K + k];
                }
                assertEquals(
                        "lane " + lane + " elem " + i,
                        2.0f * expected,
                        out.get(lane * 4 + i),
                        1e-2f);
            }
        }
    }
}
