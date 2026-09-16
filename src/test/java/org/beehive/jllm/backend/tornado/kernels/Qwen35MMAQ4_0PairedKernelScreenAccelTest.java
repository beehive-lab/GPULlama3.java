package org.beehive.jllm.backend.tornado.kernels;

import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.util.Arrays;
import java.util.Locale;
import java.util.Random;
import org.junit.Test;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoExecutionResult;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.enums.ProfilerMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * Kernel-level screen: device kernel time of the byte-load and paired-load {@code Q4_0} projections
 * on the production shapes, from the TornadoVM profiler's per-execution event timing, launched
 * alternately after a warm-up.
 *
 * <p>A measurement tool, not a gate: it prints the distributions and asserts nothing about them.
 * Opt in with {@code -Djllm.kernelScreen=true} or {@code JLLM_KERNEL_SCREEN=true}; otherwise it is
 * skipped, so a routine accelerator run carries no timing threshold.
 */
public class Qwen35MMAQ4_0PairedKernelScreenAccelTest {

    private static final int M = 32;
    private static final int WARMUP = 5;
    private static final int SAMPLES = 25;

    private static final int[][] SHAPES = {
        {17408, 5120}, // ffn_gate / ffn_up
        {5120, 17408}, // ffn_down
        {1024, 5120}, // attn_k / attn_v
    };

    @Test
    public void screen() throws Exception {
        // The surefire argLine is fixed by the accel profile, so the environment is the way in.
        assumeTrue(
                "opt in with -Djllm.kernelScreen=true or JLLM_KERNEL_SCREEN=true",
                Boolean.getBoolean("jllm.kernelScreen")
                        || "true".equals(System.getenv("JLLM_KERNEL_SCREEN")));
        for (int[] shape : SHAPES) {
            int n = shape[0];
            int k = shape[1];
            Random rng = new Random(n * 31L + k);
            HalfFloatArray a = new HalfFloatArray(M * k);
            for (int i = 0; i < M * k; i++) {
                a.set(i, new HalfFloat(rng.nextFloat() * 2.0f - 1.0f));
            }
            byte[] raw = new byte[n * (k / 32) * 18];
            rng.nextBytes(raw);
            for (int b = 0; b < n * (k / 32); b++) {
                raw[b * 18] = (byte) (b & 0xFF);
                raw[b * 18 + 1] = (byte) ((b & 1) == 0 ? 0x2C : 0xAC);
            }
            ByteArray w = new ByteArray(raw.length);
            for (int i = 0; i < raw.length; i++) {
                w.set(i, raw[i]);
            }
            FloatArray outA = new FloatArray(M * n);
            FloatArray outB = new FloatArray(M * n);

            WorkerGrid grid =
                    new WorkerGrid1D(
                            (M / Qwen35MMAKernels.BM)
                                    * (n / Qwen35MMAKernels.BN)
                                    * Qwen35MMAKernels.LOCAL);
            grid.setLocalWork(Qwen35MMAKernels.LOCAL, 1, 1);

            TaskGraph original =
                    new TaskGraph("orig")
                            .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, w)
                            .task(
                                    "p",
                                    Qwen35MMAKernels::projectionMMAQ4_0,
                                    new KernelContext(),
                                    a,
                                    w,
                                    outA,
                                    M,
                                    n,
                                    k)
                            .transferToHost(DataTransferMode.UNDER_DEMAND, outA);
            TaskGraph paired =
                    new TaskGraph("pair")
                            .transferToDevice(DataTransferMode.FIRST_EXECUTION, a, w)
                            .task(
                                    "p",
                                    Qwen35MMAKernels::projectionMMAQ4_0Paired,
                                    new KernelContext(),
                                    a,
                                    w,
                                    outB,
                                    M,
                                    n,
                                    k)
                            .transferToHost(DataTransferMode.UNDER_DEMAND, outB);
            GridScheduler sa = new GridScheduler();
            sa.addWorkerGrid("orig.p", grid);
            GridScheduler sb = new GridScheduler();
            sb.addWorkerGrid("pair.p", grid);

            try (TornadoExecutionPlan planA = new TornadoExecutionPlan(original.snapshot());
                    TornadoExecutionPlan planB = new TornadoExecutionPlan(paired.snapshot())) {
                planA.withGridScheduler(sa).withProfiler(ProfilerMode.SILENT);
                planB.withGridScheduler(sb).withProfiler(ProfilerMode.SILENT);
                for (int i = 0; i < WARMUP; i++) {
                    planA.execute();
                    planB.execute();
                }
                long[] tA = new long[SAMPLES];
                long[] tB = new long[SAMPLES];
                for (int i = 0; i < SAMPLES; i++) {
                    // Alternate the order each sample so neither kernel always follows the other.
                    if ((i & 1) == 0) {
                        tA[i] = kernelNs(planA.execute());
                        tB[i] = kernelNs(planB.execute());
                    } else {
                        tB[i] = kernelNs(planB.execute());
                        tA[i] = kernelNs(planA.execute());
                    }
                }
                report("original n=" + n + " k=" + k, tA);
                report("paired   n=" + n + " k=" + k, tB);
                assertTrue(tA[0] > 0 && tB[0] > 0);
            }
        }
    }

    private static long kernelNs(TornadoExecutionResult result) {
        return result.getProfilerResult().getDeviceKernelTime();
    }

    private static void report(String label, long[] ns) {
        long[] sorted = ns.clone();
        Arrays.sort(sorted);
        double mean = Arrays.stream(ns).average().orElse(0);
        StringBuilder samples = new StringBuilder();
        for (long v : ns) {
            samples.append(String.format(Locale.ROOT, "%.1f;", v / 1e3));
        }
        System.out.printf(
                Locale.ROOT,
                "[screen] %-28s min %.1f  median %.1f  mean %.1f  max %.1f us  samples(us) %s%n",
                label,
                sorted[0] / 1e3,
                sorted[sorted.length / 2] / 1e3,
                mean / 1e3,
                sorted[sorted.length - 1] / 1e3,
                samples);
    }
}
