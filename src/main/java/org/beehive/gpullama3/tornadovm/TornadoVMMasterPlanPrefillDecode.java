package org.beehive.gpullama3.tornadovm;

import org.beehive.gpullama3.auxiliary.RunMetrics;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.tornadovm.plan.ForwardPlanFactory;
import org.beehive.gpullama3.tornadovm.plan.PrefillDecodeForwardPlan;
import org.beehive.gpullama3.tornadovm.plan.layout.PrefillDecodeForwardTaskGraphLayout;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * GPU execution plan for sequential (single-token) prefill/decode separation.
 *
 * <p>A single {@link TornadoExecutionPlan} holds all graphs so that the KV cache
 * ({@code wrapKeyCache}, {@code wrapValueCache}) is allocated once and remains on
 * device across both phases.  Prefill and decode reuse the same N layer graphs;
 * only the logits graph is skipped during prefill.</p>
 *
 * <p>Graph layout (N+2 graphs total):</p>
 * <pre>
 *   [0]      decodeActivation    single-token FP16 → FP32; KV-cache allocated on first execution
 *   [1..N]   layer_0..layer_N-1  transformer layers (attention + FFN)
 *   [N+1]    logits              final RMSNorm + wcls matmul
 * </pre>
 *
 * <p>Two forward passes:</p>
 * <ul>
 *   <li>{@link #tornadoVMForwardPrefill} — graphs 0..N (activation + layers), logits skipped.
 *       Called once per prompt token; populates the KV cache.</li>
 *   <li>{@link #tornadoVMForwardDecode} — full pass including logits.
 *       Called once per generated token; returns logits for sampling.</li>
 * </ul>
 */
public class TornadoVMMasterPlanPrefillDecode implements TornadoVMMasterPlan {

    private final State            state;
    private final Model            model;
    private final Configuration    config;

    PrefillDecodeForwardPlan prefillDecodeForwardPlan;
    PrefillDecodeForwardTaskGraphLayout taskGraphLayout;
    public TornadoExecutionPlan executionPlan;

    // ── Construction ─────────────────────────────────────────────────────────
    TornadoVMMasterPlanPrefillDecode(State state, Model model) {
        if (ENABLE_TORNADOVM_INIT_TIME) {
            System.err.println("\nStarting TornadoVM initialization...");
        }

        this.state  = state;
        this.model  = model;
        this.config = model.configuration();

        long startTime = System.nanoTime();
        this.executionPlan = createExecutionPlan();
        long planCreationTime = System.nanoTime();

        if (CUDA_GRAPHS) executionPlan.withAllGraphs().withCUDAGraph();
        executionPlan.withPreCompilation();
        long warmupTime = System.nanoTime();

        forceCopyInReadOnlyData();
        long copyTime = System.nanoTime();

        RunMetrics.setTornadoMetrics(planCreationTime - startTime, warmupTime - planCreationTime, copyTime - warmupTime);
    }

    // ── Plan construction ─────────────────────────────────────────────────────

    @Override
    public TornadoExecutionPlan createExecutionPlan() {
        GGMLType weightType = model.weights().getWeightType();
        this.prefillDecodeForwardPlan = ForwardPlanFactory.createPrefillDecode(weightType, state, model);
        this.taskGraphLayout = prefillDecodeForwardPlan.getTaskGraphLayout();
        var taskGraphs = prefillDecodeForwardPlan.getImmutableTaskGraphs();
        return new TornadoExecutionPlan(taskGraphs.toArray(new ImmutableTaskGraph[0]));
    }

    // ── Initialisation ────────────────────────────────────────────────────────

    /** Runs all graphs once to trigger FIRST_EXECUTION uploads and warm up CUDA graphs. */
    @Override
    public void forceCopyInReadOnlyData() {
        state.wrapX.clear();
        state.positionHolder.init(0);

        for (int i = 0; i <= taskGraphLayout.logitsIdx(); i++) {
            var g = executionPlan.withGraph(i)
                    .withGridScheduler(prefillDecodeForwardPlan.getGridScheduler());
            if (CUDA_GRAPHS) g.withCUDAGraph();
            g.execute();
        }
    }

    // ── Forward passes ────────────────────────────────────────────────────────

    /**
     * GPU prefill forward: activation + all transformer layers, logits skipped.
     *
     * @param position sequence position being processed
     */
    public void tornadoVMForwardPrefill(int position) {
        var act = executionPlan.withGraph(taskGraphLayout.activationIdx())
                .withGridScheduler(prefillDecodeForwardPlan.getGridScheduler());
        if (CUDA_GRAPHS) act.withCUDAGraph();
        act.execute();

        state.positionHolder.set(0, position);
        state.temp.clear();
        state.tempFFN.clear();

        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            var l = executionPlan.withGraph(taskGraphLayout.layerIdx(layer))
                    .withGridScheduler(prefillDecodeForwardPlan.getGridScheduler());
            if (CUDA_GRAPHS) l.withCUDAGraph();
            l.execute();
        }
    }

    /**
     * GPU decode forward: full execution including logits.
     *
     * @param position sequence position being processed
     * @return logits array for token sampling
     */
    public FloatArray tornadoVMForwardDecode(int position) {
        return tornadoVMExecuteForward(position);
    }

    @Override
    public FloatArray tornadoVMExecuteForward(int position) {
        var act = executionPlan.withGraph(taskGraphLayout.activationIdx())
                .withGridScheduler(prefillDecodeForwardPlan.getGridScheduler());
        if (CUDA_GRAPHS) act.withCUDAGraph();
        act.execute();

        state.positionHolder.set(0, position);
        state.temp.clear();
        state.tempFFN.clear();

        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            var l = executionPlan.withGraph(taskGraphLayout.layerIdx(layer))
                    .withGridScheduler(prefillDecodeForwardPlan.getGridScheduler());
            if (CUDA_GRAPHS) l.withCUDAGraph();
            l.execute();
        }

        state.tempLogits.clear();
        state.wrapLogits.clear();
        var logits = executionPlan.withGraph(taskGraphLayout.logitsIdx())
                .withGridScheduler(prefillDecodeForwardPlan.getGridScheduler());
        if (CUDA_GRAPHS) logits.withCUDAGraph();
        logits.execute();

        return state.wrapLogits;
    }

    @Override
    public void freeTornadoExecutionPlan() {
        executionPlan.freeDeviceMemory();
    }
}
