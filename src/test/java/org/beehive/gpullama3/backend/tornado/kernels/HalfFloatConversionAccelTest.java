package org.beehive.gpullama3.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeTrue;

import org.beehive.gpullama3.backend.tornado.device.TornadoDevices;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * What a device does when it rounds an FP32 value to FP16 and reads it back.
 *
 * <p>This is not an abstract question. The Llama-shaped FP16 layer graphs write the normalized
 * activation to an {@code HalfFloatArray} before the QKV projection ({@code
 * mapContextWithQuantize}), so every layer of every token passes through this conversion. Qwen3
 * does not — its RMS norm and QKV projection are fused and stay in FP32 — which is why a defect
 * here shows up as "Llama and Granite disagree with the CPU, Qwen3 agrees" and looks like a family
 * problem rather than a conversion one.
 *
 * <p>The reference is {@link Float#floatToFloat16}, which is IEEE 754 round-to-nearest-even. A
 * backend that truncates instead loses up to one ULP per conversion in a consistent direction,
 * which accumulates across layers rather than cancelling.
 */
public class HalfFloatConversionAccelTest {

    private static final int N = 4096;

    /** Writes each input to FP16 storage; the read-back is what the comparison sees. */
    public static void roundTrip(KernelContext context, FloatArray in, HalfFloatArray out) {
        int i = context.globalIdx;
        out.set(i, new HalfFloat(in.get(i)));
    }

    @Test
    public void fp32ToFp16RoundingMatchesIeeeRoundToNearestEven() throws Exception {
        assumeTrue("environment absent: no accelerator", acceleratorPresent());

        FloatArray in = new FloatArray(N);
        HalfFloatArray out = new HalfFloatArray(N);
        // Values spanning the range the normalized activations actually occupy, deliberately off
        // the representable grid so rounding direction is observable.
        for (int i = 0; i < N; i++) {
            double t = (i - N / 2.0) / (N / 8.0);
            in.set(i, (float) (t * 1.0000305175781));
        }

        TaskGraph graph =
                new TaskGraph("conv")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, in)
                        .task(
                                "roundTrip",
                                HalfFloatConversionAccelTest::roundTrip,
                                new KernelContext(),
                                in,
                                out)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, out);

        WorkerGrid worker = new WorkerGrid1D(N);
        worker.setLocalWork(32, 1, 1);
        GridScheduler scheduler = new GridScheduler("conv.roundTrip", worker);

        ImmutableTaskGraph immutable = graph.snapshot();
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(immutable)) {
            plan.withGridScheduler(scheduler).execute();
        }

        int mismatches = 0;
        int firstMismatch = -1;
        double worstUlp = 0;
        for (int i = 0; i < N; i++) {
            float expected = Float.float16ToFloat(Float.floatToFloat16(in.get(i)));
            float actual = out.get(i).getFloat32();
            if (Float.compare(expected, actual) != 0) {
                if (firstMismatch < 0) {
                    firstMismatch = i;
                }
                mismatches++;
                double ulp = Math.abs(expected - actual) / Math.max(Math.ulp(expected), 1e-30);
                worstUlp = Math.max(worstUlp, ulp);
            }
        }

        if (mismatches > 0) {
            System.out.printf(
                    "backend %s: %d/%d conversions differ from round-to-nearest-even,"
                            + " worst %.2f ULP; first at i=%d in=%.9g expected=%.9g actual=%.9g%n",
                    TornadoDevices.current().id().backend(),
                    mismatches,
                    N,
                    worstUlp,
                    firstMismatch,
                    in.get(firstMismatch),
                    Float.float16ToFloat(Float.floatToFloat16(in.get(firstMismatch))),
                    out.get(firstMismatch).getFloat32());
        }
        assertEquals(
                "FP32->FP16 conversion must round to nearest even, as Float.floatToFloat16 does."
                        + " Every Llama-shaped FP16 layer writes its normalized activation through this"
                        + " conversion, so a backend that rounds differently loses precision once per"
                        + " layer per token, in a consistent direction",
                0,
                mismatches);
    }

    private static boolean acceleratorPresent() {
        try {
            return TornadoDevices.current() != null
                    && !"cpu".equals(TornadoDevices.current().id().backend().toString());
        } catch (RuntimeException | LinkageError e) {
            return false;
        }
    }
}
