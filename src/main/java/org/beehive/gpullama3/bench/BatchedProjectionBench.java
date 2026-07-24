package org.beehive.gpullama3.bench;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.util.Random;

/**
 * Shows the projection batching win directly: a projection C[B,N] = X[B,K] · Wᵀ
 * where W is [N,K]. In batched form each weight row W[n] is loaded into shared
 * memory once per workgroup and reused across all B tokens (arithmetic intensity
 * ~B → compute-bound); the single-token path re-reads W[n] for every token
 * (memory-bound). This is why batching turns decode matvecs into a compute-bound
 * win — the opposite regime from attention (memory-bound, ~2×).
 *
 * Synthetic FP32, Llama-1B-ish projection shape (K=N=2048). No model load.
 * Run: ... BatchedProjectionBench [B] [K] [N]
 */
public class BatchedProjectionBench {

    static final int LOCAL = 128;

    /** One workgroup per output column n: load W[n] to shared, apply to all B tokens. */
    public static void batchedProjection(KernelContext ctx, FloatArray x, FloatArray w, FloatArray c, int B, int K, int N) {
        int n = ctx.groupIdx;         // output column
        int tid = ctx.localIdx;
        int localSz = ctx.localGroupSizeX;
        float[] wRow = ctx.allocateFloatLocalArray(2048); // K <= 2048
        for (int i = tid; i < K; i += localSz) {
            wRow[i] = w.get(n * K + i);
        }
        ctx.localBarrier();
        for (int b = tid; b < B; b += localSz) {
            float acc = 0.0f;
            int xoff = b * K;
            for (int i = 0; i < K; i++) {
                acc += wRow[i] * x.get(xoff + i);
            }
            c.set(b * N + n, acc);
        }
    }

    /** Single-token matvec: one workgroup per output column, W[n] re-read from global. */
    public static void singleProjection(KernelContext ctx, FloatArray x, FloatArray w, FloatArray c, int K, int N) {
        int n = ctx.groupIdx;
        int tid = ctx.localIdx;
        int localSz = ctx.localGroupSizeX;
        float[] partial = ctx.allocateFloatLocalArray(LOCAL);
        float acc = 0.0f;
        for (int i = tid; i < K; i += localSz) {
            acc += w.get(n * K + i) * x.get(i);
        }
        partial[tid] = acc;
        ctx.localBarrier();
        for (int s = localSz / 2; s > 0; s >>= 1) {
            if (tid < s) {
                partial[tid] += partial[tid + s];
            }
            ctx.localBarrier();
        }
        if (tid == 0) {
            c.set(n, partial[0]);
        }
    }

    public static void main(String[] args) throws TornadoExecutionPlanException {
        int B = args.length > 0 ? Integer.parseInt(args[0]) : 32;
        int K = args.length > 1 ? Integer.parseInt(args[1]) : 2048;
        int N = args.length > 2 ? Integer.parseInt(args[2]) : 2048;
        int iters = 100;
        Random rnd = new Random(1);

        FloatArray w = new FloatArray(N * K);
        for (int i = 0; i < N * K; i++) {
            w.set(i, rnd.nextFloat() - 0.5f);
        }
        FloatArray xB = new FloatArray(B * K);
        for (int i = 0; i < B * K; i++) {
            xB.set(i, rnd.nextFloat() - 0.5f);
        }
        FloatArray cB = new FloatArray(B * N);

        KernelContext ctx = new KernelContext();
        WorkerGrid1D wg = new WorkerGrid1D(N * LOCAL);
        wg.setLocalWork(LOCAL, 1, 1);
        GridScheduler gs = new GridScheduler("p.proj", wg);
        TaskGraph tg = new TaskGraph("p") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, xB, w) //
                .task("proj", BatchedProjectionBench::batchedProjection, ctx, xB, w, cB, B, K, N) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, cB);

        double batchedMs;
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(tg.snapshot())) {
            plan.withGridScheduler(gs);
            for (int i = 0; i < 10; i++) {
                plan.execute();
            }
            // correctness (row 0..N, token 0)
            double maxRel = 0.0;
            for (int n = 0; n < Math.min(N, 256); n++) {
                float ref = 0.0f;
                for (int i = 0; i < K; i++) {
                    ref += w.get(n * K + i) * xB.get(i);
                }
                float got = cB.get(n); // token 0 output
                maxRel = Math.max(maxRel, Math.abs(ref - got) / Math.max(1e-2, Math.abs(ref)));
            }
            long s = System.nanoTime();
            for (int i = 0; i < iters; i++) {
                plan.execute();
            }
            batchedMs = (System.nanoTime() - s) / 1e6 / iters;
            System.out.printf("Batched projection K=%d N=%d B=%d  correctness maxRel(tok0)=%.4f%n", K, N, B, maxRel);
        }

        // single-token matvec
        FloatArray x1 = new FloatArray(K);
        for (int i = 0; i < K; i++) {
            x1.set(i, rnd.nextFloat() - 0.5f);
        }
        FloatArray c1 = new FloatArray(N);
        KernelContext ctx1 = new KernelContext();
        WorkerGrid1D wg1 = new WorkerGrid1D(N * LOCAL);
        wg1.setLocalWork(LOCAL, 1, 1);
        GridScheduler gs1 = new GridScheduler("s.proj", wg1);
        TaskGraph tg1 = new TaskGraph("s") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, x1, w) //
                .task("proj", BatchedProjectionBench::singleProjection, ctx1, x1, w, c1, K, N) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, c1);
        double singleMs;
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(tg1.snapshot())) {
            plan.withGridScheduler(gs1);
            for (int i = 0; i < 10; i++) {
                plan.execute();
            }
            long s = System.nanoTime();
            for (int i = 0; i < iters; i++) {
                plan.execute();
            }
            singleMs = (System.nanoTime() - s) / 1e6 / iters;
        }
        System.out.printf("  batched %d tokens: %.4f ms  |  1 token: %.4f ms  |  B*single=%.4f ms%n",
                B, batchedMs, singleMs, singleMs * B);
        System.out.printf("  projection speedup (B matvecs -> 1 batched GEMM): %.2fx  (%.0f vs %.0f tok/s)%n",
                (singleMs * B) / batchedMs, B / (batchedMs / 1000.0), 1.0 / (singleMs / 1000.0));
    }
}
