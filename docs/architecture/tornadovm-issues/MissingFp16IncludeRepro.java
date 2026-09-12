package tornadovmissues;

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
 * Standalone reproducer: a kernel that declares a half-precision shared tile fails to compile when
 * no other fp16 spelling survives code generation, because the {@code cuda_fp16.h} include is
 * decided by scanning the emitted text for {@code __half}, {@code half2} or {@code 2half}.
 *
 * <p>No model, no GGUF, no engine code — one kernel, three arrays, a single MMA step.
 *
 * <pre>
 *   export TORNADOVM_HOME=/path/to/tornadovm-6.0.1-jdk21-dev-cuda
 *   mkdir -p /tmp/fp16repro
 *   javac -cp "$TORNADOVM_HOME/share/java/tornado/*" -d /tmp/fp16repro MissingFp16IncludeRepro.java
 *   tornado --jvm "-Dtornado.recover.bailout=False" \
 *           -cp /tmp/fp16repro tornadovmissues.MissingFp16IncludeRepro
 * </pre>
 *
 * <p>Expected: the kernel compiles and the program prints a finite first output element. Actual:
 * NVRTC rejects the generated source with {@code identifier "half" is undefined} on the shared tile
 * declaration.
 *
 * <p>Pass {@code trigger} as the single argument to run the same kernel with one extra read —
 * {@code weights.getHalfFloat(0).getFloat32()}, whose lowering contains {@code __half2float} — added
 * to the result. That version compiles and runs, which is the whole point: the difference between
 * the two is one fp16 <b>spelling</b> in the emitted text, not the presence of fp16 storage.
 */
public final class MissingFp16IncludeRepro {

    private static final int M = 16;
    private static final int N = 8;
    private static final int K = 16;
    private static final int LOCAL = 32;

    private MissingFp16IncludeRepro() {}

    /** Declares a half tile, writes it with the swizzled store, reads it back into an MMA. */
    public static void halfTileNoSpelling(
            KernelContext ctx, HalfFloatArray a, FloatArray scale, FloatArray out) {
        int lane = ctx.localIdx;
        int[] aTile = ctx.allocateIntLocalArray(M * K / 2);
        HalfFloat[] bTile = ctx.allocateHalfFloatLocalArray(N * K);

        // A: filled by cp.async, so no half is ever read into a register here.
        for (int slot = 0; slot < 4; slot++) {
            int i = lane + slot * LOCAL;
            int row = i >>> 3;
            int kk = (i & 7) << 1;
            ctx.asyncCopyToLocal(aTile, i, a, row * K + kk);
        }

        // B: written from a value computed at run time, so the store lowers to a bare
        // ((half *) tile)[...] = <float> assignment. A compile-time constant would instead emit
        // __float2half(...), which is one of the spellings the include scan looks for -- that is
        // the difference between reproducing and not.
        for (int t = 0; t < 4; t++) {
            int row = (lane & 3) * 4 + t;
            int col = lane >> 2;
            ctx.mmaStoreBSwizzled(bTile, row, col, N, new HalfFloat(scale.get(0) + t), 0);
        }

        ctx.asyncCopyCommit();
        ctx.asyncCopyWaitGroup(0);
        ctx.localBarrier();

        float[] acc = ctx.mmaFragment(0.0f);
        acc =
                ctx.mma(
                        ctx.mmaLoadA(aTile, K, 0),
                        ctx.mmaLoadBSwizzled(bTile, K, 0),
                        acc,
                        MMAShape.M16N8K16);
        ctx.mmaStore(acc, out, 0, 0, N);
    }

    /** The same kernel plus one {@code getHalfFloat} read, which emits {@code __half2float}. */
    public static void halfTileWithSpelling(
            KernelContext ctx, HalfFloatArray a, FloatArray scale, HalfFloatArray weights,
            FloatArray out) {
        int lane = ctx.localIdx;
        int[] aTile = ctx.allocateIntLocalArray(M * K / 2);
        HalfFloat[] bTile = ctx.allocateHalfFloatLocalArray(N * K);

        for (int slot = 0; slot < 4; slot++) {
            int i = lane + slot * LOCAL;
            int row = i >>> 3;
            int kk = (i & 7) << 1;
            ctx.asyncCopyToLocal(aTile, i, a, row * K + kk);
        }
        for (int t = 0; t < 4; t++) {
            int row = (lane & 3) * 4 + t;
            int col = lane >> 2;
            ctx.mmaStoreBSwizzled(bTile, row, col, N, new HalfFloat(scale.get(0) + t), 0);
        }

        ctx.asyncCopyCommit();
        ctx.asyncCopyWaitGroup(0);
        ctx.localBarrier();

        float[] acc = ctx.mmaFragment(0.0f);
        acc =
                ctx.mma(
                        ctx.mmaLoadA(aTile, K, 0),
                        ctx.mmaLoadBSwizzled(bTile, K, 0),
                        acc,
                        MMAShape.M16N8K16);
        ctx.mmaStore(acc, out, 0, 0, N);
        if (lane == 0) {
            out.set(0, out.get(0) + weights.get(0).getFloat32());
        }
    }

    public static void main(String[] args) throws Exception {
        boolean trigger = args.length > 0 && "trigger".equals(args[0]);

        HalfFloatArray a = new HalfFloatArray(M * K);
        for (int i = 0; i < M * K; i++) {
            a.set(i, new HalfFloat(0.25f));
        }
        HalfFloatArray weights = new HalfFloatArray(2);
        weights.set(0, new HalfFloat(1.0f));
        weights.set(1, new HalfFloat(1.0f));
        FloatArray scale = new FloatArray(1);
        scale.set(0, 0.5f);
        FloatArray out = new FloatArray(M * N);
        out.init(0.0f);

        TaskGraph graph = new TaskGraph("repro");
        graph.transferToDevice(DataTransferMode.EVERY_EXECUTION, a, weights, scale, out);
        if (trigger) {
            graph.task(
                    "mma",
                    MissingFp16IncludeRepro::halfTileWithSpelling,
                    new KernelContext(),
                    a,
                    scale,
                    weights,
                    out);
        } else {
            graph.task(
                    "mma",
                    MissingFp16IncludeRepro::halfTileNoSpelling,
                    new KernelContext(),
                    a,
                    scale,
                    out);
        }
        graph.transferToHost(DataTransferMode.EVERY_EXECUTION, out);

        GridScheduler scheduler = new GridScheduler();
        WorkerGrid1D worker = new WorkerGrid1D(LOCAL);
        worker.setLocalWork(LOCAL, 1, 1);
        scheduler.addWorkerGrid("repro.mma", worker);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }
        System.out.printf(
                "%s: out[0] = %s%n", trigger ? "with fp16 spelling" : "no fp16 spelling", out.get(0));
    }
}
