package org.beehive.gpullama3.bench;

import org.beehive.gpullama3.tornadovm.kernels.TransformerBatchPrefillKernels;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.exceptions.TornadoExecutionPlanException;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.Random;

/**
 * Standalone validation + benchmark for {@link TransformerBatchPrefillKernels#batchedDecodeAttention}:
 * B independent sequences each with their own KV cache, one query token each,
 * every slot attending 0..pos of its own cache. Synthetic Llama-3.2-1B dims (no
 * model load). Verifies correctness against a CPU reference and reports the
 * batched attention throughput vs running the same slots one-at-a-time.
 *
 * Run: java ... org.beehive.gpullama3.bench.BatchedDecodeAttentionBench [B] [seqLen]
 */
public class BatchedDecodeAttentionBench {

    // Llama-3.2-1B geometry.
    static final int DIM = 2048;
    static final int N_HEADS = 32;
    static final int HEAD_SIZE = 64;
    static final int N_KV_HEADS = 8;
    static final int KV_DIM = N_KV_HEADS * HEAD_SIZE;   // 512
    static final int KV_MUL = N_HEADS / N_KV_HEADS;     // 4
    static final int N_LAYERS = 1;                      // single layer for the microbench
    static final int CTX = 1024;
    static final int LAYER = 0;
    static final int LOCAL = HEAD_SIZE;                 // threads per (slot,head) workgroup

    public static void main(String[] args) throws TornadoExecutionPlanException {
        int B = args.length > 0 ? Integer.parseInt(args[0]) : 32;
        int seqLen = args.length > 1 ? Integer.parseInt(args[1]) : 256; // each slot attends 0..seqLen-1
        int iterations = 100;
        Random rnd = new Random(42);

        FloatArray q = new FloatArray(B * DIM);
        FloatArray keyCache = new FloatArray(B * N_LAYERS * CTX * KV_DIM);
        FloatArray valueCache = new FloatArray(B * N_LAYERS * CTX * KV_DIM);
        FloatArray xb = new FloatArray(B * DIM);
        IntArray seqPos = new IntArray(B);

        for (int i = 0; i < B * DIM; i++) {
            q.set(i, rnd.nextFloat() - 0.5f);
        }
        for (int b = 0; b < B; b++) {
            seqPos.set(b, seqLen - 1);
            long base = (long) b * N_LAYERS * CTX * KV_DIM;
            for (int t = 0; t < seqLen; t++) {
                for (int d = 0; d < KV_DIM; d++) {
                    keyCache.set((int) (base + (long) t * KV_DIM + d), rnd.nextFloat() - 0.5f);
                    valueCache.set((int) (base + (long) t * KV_DIM + d), rnd.nextFloat() - 0.5f);
                }
            }
        }

        KernelContext context = new KernelContext();
        WorkerGrid1D worker = new WorkerGrid1D(B * N_HEADS * LOCAL);
        worker.setLocalWork(LOCAL, 1, 1);
        GridScheduler grid = new GridScheduler("bench.attn", worker);

        TaskGraph tg = new TaskGraph("bench") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, q, keyCache, valueCache, seqPos) //
                .task("attn", TransformerBatchPrefillKernels::batchedDecodeAttention, //
                        context, seqPos, q, keyCache, valueCache, xb, //
                        N_HEADS, HEAD_SIZE, KV_DIM, KV_MUL, LAYER, N_LAYERS, CTX, DIM) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, xb);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(tg.snapshot())) {
            plan.withGridScheduler(grid);
            // warmup
            for (int i = 0; i < 10; i++) {
                plan.execute();
            }
            // correctness
            float[] ref = cpuReference(q, keyCache, valueCache, seqPos, B, seqLen);
            double maxRel = 0.0;
            int bad = 0;
            for (int i = 0; i < B * DIM; i++) {
                float e = ref[i];
                float a = xb.get(i);
                double rel = Math.abs(e - a) / Math.max(1e-3, Math.abs(e));
                maxRel = Math.max(maxRel, rel);
                if (rel > 2e-2) {
                    bad++;
                }
            }
            System.out.printf("Batched decode attention: B=%d seqLen=%d  correctness: maxRel=%.4f, out-of-tol=%d/%d%n",
                    B, seqLen, maxRel, bad, B * DIM);

            long start = System.nanoTime();
            for (int i = 0; i < iterations; i++) {
                plan.execute();
            }
            double batchedMs = (System.nanoTime() - start) / 1e6 / iterations;

            // Per-slot one-at-a-time (B=1) for comparison.
            double singleMs = benchSingleSlot(seqLen, iterations);
            System.out.printf("  batched %d slots: %.4f ms/step  |  1 slot: %.4f ms  |  B*single=%.4f ms%n",
                    B, batchedMs, singleMs, singleMs * B);
            System.out.printf("  speedup (B sequential attn -> one batched launch): %.2fx%n", (singleMs * B) / batchedMs);
        }
    }

    private static double benchSingleSlot(int seqLen, int iterations) throws TornadoExecutionPlanException {
        Random rnd = new Random(7);
        FloatArray q = new FloatArray(DIM);
        FloatArray keyCache = new FloatArray(N_LAYERS * CTX * KV_DIM);
        FloatArray valueCache = new FloatArray(N_LAYERS * CTX * KV_DIM);
        FloatArray xb = new FloatArray(DIM);
        IntArray seqPos = new IntArray(1);
        seqPos.set(0, seqLen - 1);
        for (int i = 0; i < DIM; i++) {
            q.set(i, rnd.nextFloat() - 0.5f);
        }
        for (int i = 0; i < N_LAYERS * CTX * KV_DIM; i++) {
            keyCache.set(i, rnd.nextFloat() - 0.5f);
            valueCache.set(i, rnd.nextFloat() - 0.5f);
        }
        KernelContext context = new KernelContext();
        WorkerGrid1D worker = new WorkerGrid1D(N_HEADS * LOCAL);
        worker.setLocalWork(LOCAL, 1, 1);
        GridScheduler grid = new GridScheduler("single.attn", worker);
        TaskGraph tg = new TaskGraph("single") //
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, q, keyCache, valueCache, seqPos) //
                .task("attn", TransformerBatchPrefillKernels::batchedDecodeAttention, //
                        context, seqPos, q, keyCache, valueCache, xb, //
                        N_HEADS, HEAD_SIZE, KV_DIM, KV_MUL, LAYER, N_LAYERS, CTX, DIM) //
                .transferToHost(DataTransferMode.EVERY_EXECUTION, xb);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(tg.snapshot())) {
            plan.withGridScheduler(grid);
            for (int i = 0; i < 10; i++) {
                plan.execute();
            }
            long start = System.nanoTime();
            for (int i = 0; i < iterations; i++) {
                plan.execute();
            }
            return (System.nanoTime() - start) / 1e6 / iterations;
        }
    }

    private static float[] cpuReference(FloatArray q, FloatArray keyCache, FloatArray valueCache, IntArray seqPos, int B, int seqLen) {
        float[] out = new float[B * DIM];
        float scale = (float) (1.0 / Math.sqrt(HEAD_SIZE));
        for (int b = 0; b < B; b++) {
            int pos = seqPos.get(b);
            long base = (long) b * N_LAYERS * CTX * KV_DIM;
            for (int h = 0; h < N_HEADS; h++) {
                int kvHead = h / KV_MUL;
                int qOff = b * DIM + h * HEAD_SIZE;
                float[] scores = new float[pos + 1];
                float max = Float.NEGATIVE_INFINITY;
                for (int t = 0; t <= pos; t++) {
                    float s = 0.0f;
                    for (int d = 0; d < HEAD_SIZE; d++) {
                        s += q.get(qOff + d) * keyCache.get((int) (base + (long) t * KV_DIM + kvHead * HEAD_SIZE + d));
                    }
                    s *= scale;
                    scores[t] = s;
                    max = Math.max(max, s);
                }
                float sum = 0.0f;
                for (int t = 0; t <= pos; t++) {
                    scores[t] = (float) Math.exp(scores[t] - max);
                    sum += scores[t];
                }
                float inv = sum > 0 ? 1.0f / sum : 0.0f;
                for (int d = 0; d < HEAD_SIZE; d++) {
                    float acc = 0.0f;
                    for (int t = 0; t <= pos; t++) {
                        acc += scores[t] * inv * valueCache.get((int) (base + (long) t * KV_DIM + kvHead * HEAD_SIZE + d));
                    }
                    out[qOff + d] = acc;
                }
            }
        }
        return out;
    }
}
