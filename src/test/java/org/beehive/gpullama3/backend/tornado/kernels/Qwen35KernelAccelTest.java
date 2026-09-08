package org.beehive.gpullama3.backend.tornado.kernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;
import org.beehive.gpullama3.inference.op.CpuOperations;
import org.beehive.gpullama3.tensor.standard.ArrayFloatTensor;
import org.beehive.gpullama3.tensor.standard.FloatTensor;
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
 * The {@code qwen35} mixer kernels, <b>compiled and executed on a device</b>, against the host
 * operations.
 *
 * <p>The parity tests beside this one run every lane on the host. That settles the arithmetic and
 * the addressing and says nothing about whether TornadoVM can compile the kernel — a distinction
 * that is not academic here. Q5_K's matrix-vector kernel passed its host parity test and then
 * failed to compile in seven successive formulations, the cause being one method call inlined into
 * a loop. These kernels use nested loops, a retained state array written in place, and
 * {@code TornadoMath} transcendentals, none of which the host tests exercise as device code.
 *
 * <p>Dimensions are the 27B's own where it matters — a 128-wide delta-net head, 48 value heads
 * against 16 key heads — and small in the ways that do not, so the test stays quick.
 */
public class Qwen35KernelAccelTest {

    private static final int STATE_DIM = 128;
    private static final int VALUE_HEADS = 48;
    private static final int KEY_HEADS = 16;
    private static final int KEY_DIM = KEY_HEADS * STATE_DIM;
    private static final int VALUE_DIM = VALUE_HEADS * STATE_DIM;
    private static final int CONV_DIM = 10240;
    private static final int CONV_KERNEL = 4;
    private static final float EPS = 1e-6f;

    private final Random random = new Random(20260908L);

    private float[] noise(int n, float scale) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) random.nextGaussian() * scale;
        }
        return v;
    }

    private static FloatArray toDevice(float[] v) {
        FloatArray a = new FloatArray(v.length);
        for (int i = 0; i < v.length; i++) {
            a.set(i, v[i]);
        }
        return a;
    }

    private static void run(TaskGraph graph, String taskName, int global, int local)
            throws Exception {
        WorkerGrid worker = new WorkerGrid1D(global);
        worker.setLocalWork(local, 1, 1);
        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid(graph.getTaskGraphName() + "." + taskName, worker);
        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }
    }

    private static void assertClose(String what, FloatTensor host, FloatArray device) {
        for (int i = 0; i < host.size(); i++) {
            float expected = host.getFloat(i);
            assertEquals(
                    what + "[" + i + "]",
                    expected,
                    device.get(i),
                    Math.max(1e-4f, Math.abs(expected) * 1e-4f));
        }
    }

    @Test
    public void theCausalConvolutionRunsOnTheDevice() throws Exception {
        float[] input = noise(CONV_DIM, 1.0f);
        float[] weight = noise(CONV_DIM * CONV_KERNEL, 0.5f);
        float[] window = noise(CONV_DIM * (CONV_KERNEL - 1), 1.0f);

        FloatTensor hostWindow = new ArrayFloatTensor(window.clone());
        FloatTensor hostOut = ArrayFloatTensor.allocate(CONV_DIM);
        CpuOperations.causalConv1d(
                new ArrayFloatTensor(input.clone()),
                new ArrayFloatTensor(weight.clone()),
                hostWindow,
                hostOut,
                CONV_DIM,
                CONV_KERNEL);

        FloatArray deviceWindow = toDevice(window);
        FloatArray deviceOut = new FloatArray(CONV_DIM);
        KernelContext context = new KernelContext();
        FloatArray deviceInput = toDevice(input);
        FloatArray deviceWeight = toDevice(weight);
        TaskGraph graph =
                new TaskGraph("conv")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                deviceInput,
                                deviceWeight,
                                deviceWindow,
                                deviceOut)
                        .task(
                                "k",
                                Qwen35DeltaNetKernels::causalConv1d,
                                context,
                                deviceInput,
                                deviceWeight,
                                deviceWindow,
                                deviceOut,
                                CONV_DIM,
                                CONV_KERNEL,
                                0)
                        .transferToHost(
                                DataTransferMode.EVERY_EXECUTION, deviceOut, deviceWindow);
        run(graph, "k", CONV_DIM, 128);

        assertClose("conv out", hostOut, deviceOut);
        assertClose("conv window", hostWindow, deviceWindow);
    }

    @Test
    public void theDeltaRuleRunsOnTheDevice() throws Exception {
        float[] q = noise(KEY_DIM, 0.1f);
        float[] k = noise(KEY_DIM, 0.1f);
        float[] v = noise(VALUE_DIM, 1.0f);
        float[] state = noise(VALUE_HEADS * STATE_DIM * STATE_DIM, 0.05f);
        float[] decay = new float[VALUE_HEADS];
        float[] beta = new float[VALUE_HEADS];
        for (int h = 0; h < VALUE_HEADS; h++) {
            decay[h] = 0.5f + 0.5f * random.nextFloat();
            beta[h] = random.nextFloat();
        }

        FloatTensor hostState = new ArrayFloatTensor(state.clone());
        FloatTensor hostOut = ArrayFloatTensor.allocate(VALUE_DIM);
        CpuOperations.deltaRuleUpdate(
                new ArrayFloatTensor(q.clone()),
                new ArrayFloatTensor(k.clone()),
                new ArrayFloatTensor(v.clone()),
                new ArrayFloatTensor(decay.clone()),
                new ArrayFloatTensor(beta.clone()),
                hostState,
                hostOut,
                VALUE_HEADS,
                KEY_HEADS,
                STATE_DIM);

        FloatArray dq = toDevice(q);
        FloatArray dk = toDevice(k);
        FloatArray dv = toDevice(v);
        FloatArray ddecay = toDevice(decay);
        FloatArray dbeta = toDevice(beta);
        FloatArray dstate = toDevice(state);
        FloatArray dout = new FloatArray(VALUE_DIM);
        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph("delta")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                dq, dk, dv, ddecay, dbeta, dstate, dout)
                        .task(
                                "k",
                                Qwen35DeltaNetKernels::deltaRule,
                                context,
                                dq, dk, dv, ddecay, dbeta, dstate, dout,
                                VALUE_HEADS,
                                KEY_HEADS,
                                STATE_DIM,
                                0)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, dout, dstate);
        run(graph, "k", VALUE_HEADS * STATE_DIM, 128);

        assertClose("delta readout", hostOut, dout);
        assertClose("delta state", hostState, dstate);
    }

    @Test
    public void theGatedNormAndL2NormRunOnTheDevice() throws Exception {
        float[] values = noise(VALUE_DIM, 1.0f);
        float[] gate = noise(VALUE_DIM, 1.0f);
        float[] weight = noise(STATE_DIM, 1.0f);

        FloatTensor hostGated = new ArrayFloatTensor(values.clone());
        CpuOperations.gatedNorm(
                hostGated,
                new ArrayFloatTensor(gate.clone()),
                new ArrayFloatTensor(weight.clone()),
                VALUE_HEADS,
                STATE_DIM,
                EPS);

        FloatArray dvalues = toDevice(values);
        FloatArray dgate = toDevice(gate);
        FloatArray dweight = toDevice(weight);
        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph("gated")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, dvalues, dgate, dweight)
                        .task(
                                "k",
                                Qwen35DeltaNetKernels::gatedNormPerHead,
                                context,
                                dvalues,
                                dgate,
                                dweight,
                                VALUE_HEADS,
                                STATE_DIM,
                                EPS)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, dvalues);
        run(graph, "k", VALUE_HEADS, 16);
        assertClose("gated norm", hostGated, dvalues);

        float[] keys = noise(KEY_DIM, 1.0f);
        FloatTensor hostL2 = new ArrayFloatTensor(keys.clone());
        for (int head = 0; head < KEY_HEADS; head++) {
            CpuOperations.l2Norm(hostL2, head * STATE_DIM, STATE_DIM, EPS);
        }
        FloatArray dkeys = toDevice(keys);
        KernelContext l2Context = new KernelContext();
        TaskGraph l2 =
                new TaskGraph("l2")
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, dkeys)
                        .task(
                                "k",
                                Qwen35DeltaNetKernels::l2NormPerHead,
                                l2Context,
                                dkeys,
                                KEY_HEADS,
                                STATE_DIM,
                                EPS)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, dkeys);
        run(l2, "k", KEY_HEADS, 16);
        assertClose("l2", hostL2, dkeys);
    }

    @Test
    public void theDecayAndBetaRunOnTheDevice() throws Exception {
        float[] alpha = noise(VALUE_HEADS, 1.0f);
        float[] beta = noise(VALUE_HEADS, 1.0f);
        float[] dtBias = noise(VALUE_HEADS, 0.5f);
        float[] a = new float[VALUE_HEADS];
        for (int h = 0; h < VALUE_HEADS; h++) {
            a[h] = -(0.5f + random.nextFloat());
        }

        float[] expectedDecay = new float[VALUE_HEADS];
        float[] expectedBeta = new float[VALUE_HEADS];
        for (int h = 0; h < VALUE_HEADS; h++) {
            expectedBeta[h] = CpuOperations.logistic(beta[h]);
            expectedDecay[h] =
                    (float) Math.exp(a[h] * CpuOperations.softplus(alpha[h] + dtBias[h]));
        }

        FloatArray dalpha = toDevice(alpha);
        FloatArray dbeta = toDevice(beta);
        FloatArray ddt = toDevice(dtBias);
        FloatArray da = toDevice(a);
        KernelContext context = new KernelContext();
        TaskGraph graph =
                new TaskGraph("decay")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, dalpha, dbeta, ddt, da)
                        .task(
                                "k",
                                Qwen35DeltaNetKernels::decayAndBeta,
                                context,
                                dalpha,
                                dbeta,
                                ddt,
                                da,
                                VALUE_HEADS)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, dalpha, dbeta);
        run(graph, "k", VALUE_HEADS, 16);

        for (int h = 0; h < VALUE_HEADS; h++) {
            assertEquals("beta[" + h + "]", expectedBeta[h], dbeta.get(h), 1e-5f);
            assertEquals("decay[" + h + "]", expectedDecay[h], dalpha.get(h), 1e-5f);
        }
    }

    @Test
    public void theAttentionKernelsRunOnTheDevice() throws Exception {
        int heads = 24;
        int kvHeads = 4;
        int headDim = 256;
        int rotaryDim = 64;
        int queryDim = heads * headDim;
        int kvDim = kvHeads * headDim;
        int position = 5;

        float[] fused = noise(queryDim * 2, 1.0f);
        FloatArray dfused = toDevice(fused);
        FloatArray dquery = new FloatArray(queryDim);
        FloatArray dgate = new FloatArray(queryDim);
        KernelContext splitContext = new KernelContext();
        TaskGraph split =
                new TaskGraph("split")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, dfused, dquery, dgate)
                        .task(
                                "k",
                                Qwen35AttentionKernels::splitQueryGate,
                                splitContext,
                                dfused,
                                dquery,
                                dgate,
                                heads,
                                headDim)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, dquery, dgate);
        run(split, "k", queryDim, 128);
        for (int head = 0; head < heads; head++) {
            for (int i = 0; i < headDim; i += 37) {
                assertEquals(
                        "query", fused[head * 2 * headDim + i], dquery.get(head * headDim + i), 0f);
                assertEquals(
                        "gate",
                        fused[head * 2 * headDim + headDim + i],
                        dgate.get(head * headDim + i),
                        0f);
            }
        }

        var freqs =
                org.beehive.gpullama3.model.loader.RopeFrequencies.precomputeFreqsCis(
                        64, rotaryDim, 1e7f, false, 0, 0, 0, 0);
        float[] query = noise(queryDim, 1.0f);
        float[] key = noise(kvDim, 1.0f);
        FloatTensor hostQuery = new ArrayFloatTensor(query.clone());
        FloatTensor hostKey = new ArrayFloatTensor(key.clone());
        CpuOperations.ropeNeoxPartial(
                hostQuery,
                heads,
                headDim,
                rotaryDim,
                position,
                new ArrayFloatTensor(freqs.first()),
                new ArrayFloatTensor(freqs.second()));
        CpuOperations.ropeNeoxPartial(
                hostKey,
                kvHeads,
                headDim,
                rotaryDim,
                position,
                new ArrayFloatTensor(freqs.first()),
                new ArrayFloatTensor(freqs.second()));

        FloatArray dq = toDevice(query);
        FloatArray dk = toDevice(key);
        IntArray positionHolder = new IntArray(2);
        positionHolder.set(0, position);
        positionHolder.set(1, 0);
        KernelContext ropeContext = new KernelContext();
        FloatArray dreal = toDevice(freqs.first());
        FloatArray dimag = toDevice(freqs.second());
        TaskGraph rope =
                new TaskGraph("rope")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION,
                                positionHolder, dq, dk, dreal, dimag)
                        .task(
                                "k",
                                Qwen35AttentionKernels::ropeNeoxPartial,
                                ropeContext,
                                positionHolder,
                                dq,
                                dk,
                                dreal,
                                dimag,
                                heads,
                                kvHeads,
                                headDim,
                                rotaryDim)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, dq, dk);
        run(rope, "k", heads * (rotaryDim / 2), 32);
        assertClose("rope query", hostQuery, dq);
        assertClose("rope key", hostKey, dk);

        // The output gate.
        float[] values = noise(queryDim, 1.0f);
        float[] gateValues = noise(queryDim, 1.0f);
        FloatArray dvalues = toDevice(values);
        FloatArray dgateValues = toDevice(gateValues);
        KernelContext gateContext = new KernelContext();
        TaskGraph gate =
                new TaskGraph("gate")
                        .transferToDevice(
                                DataTransferMode.EVERY_EXECUTION, dvalues, dgateValues)
                        .task(
                                "k",
                                Qwen35AttentionKernels::applyOutputGate,
                                gateContext,
                                dvalues,
                                dgateValues,
                                queryDim)
                        .transferToHost(DataTransferMode.EVERY_EXECUTION, dvalues);
        run(gate, "k", queryDim, 128);
        for (int i = 0; i < queryDim; i += 53) {
            float expected = values[i] * CpuOperations.logistic(gateValues[i]);
            assertTrue(
                    "gated[" + i + "] " + expected + " vs " + dvalues.get(i),
                    Math.abs(expected - dvalues.get(i)) <= Math.max(1e-5f, Math.abs(expected) * 1e-5f));
        }
    }
}
