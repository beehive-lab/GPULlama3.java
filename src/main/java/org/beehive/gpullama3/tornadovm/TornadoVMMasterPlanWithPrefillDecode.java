package org.beehive.gpullama3.tornadovm;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Wraps {@link TornadoVMMasterPlanStandard} and adds a prefill-only GPU forward pass.
 *
 * <p>Parallel to {@link TornadoVMMasterPlanStandard} — does NOT modify it.</p>
 *
 * <p>The existing execution plan has this graph layout:</p>
 * <pre>
 *   graph 0         : preprocessing (embedding setup)
 *   graphs 1..N     : transformer layers
 *   graph N+1       : logits projection (final RMSNorm + wcls matmul)
 * </pre>
 *
 * <p>{@link #tornadoVMForwardPrefill} executes graphs 0..N and deliberately
 * skips graph N+1. The KV cache is populated correctly by the layer graphs;
 * the logits are not needed for prefill positions so the projection is wasted
 * work that we avoid.</p>
 *
 * <p>For decode, {@link #tornadoVMForwardDecode} delegates to the wrapped
 * plan's {@code tornadoVMForwardExecuteLayered}, preserving identical behaviour
 * to the baseline GPU path.</p>
 *
 * <p>Implements {@link TornadoVMMasterPlan} so it can be returned by the factory
 * and stored in the model; {@link #tornadoVMForwardExecuteLayered} delegates to
 * {@link #tornadoVMForwardDecode}.</p>
 */
public class TornadoVMMasterPlanWithPrefillDecode implements TornadoVMMasterPlan {

    private final TornadoVMMasterPlanStandard plan;
    private final State state;
    private final Configuration config;

    public TornadoVMMasterPlanWithPrefillDecode(TornadoVMMasterPlanStandard plan, State state, Model model) {
        this.plan = plan;
        this.state = state;
        this.config = model.configuration();
    }

    /** Factory: initializes the inner standard plan then wraps it. */
    public static TornadoVMMasterPlanWithPrefillDecode initialize(State state, Model model) {
        TornadoVMMasterPlanStandard inner = TornadoVMMasterPlanStandard.initialize(state, model);
        return new TornadoVMMasterPlanWithPrefillDecode(inner, state, model);
    }

    /**
     * GPU prefill forward: runs preprocessing + all transformer layers, skips logits.
     *
     * <p>Mirrors {@link TornadoVMMasterPlan#tornadoVMForwardExecuteLayered} except
     * the final logits graph (graph {@code numberOfLayers + 1}) is not executed.</p>
     *
     * @param position sequence position being processed
     */
    public void tornadoVMForwardPrefill(int position) {
        // Graph 0: preprocessing
        plan.executionPlan.withGraph(0)
                .withGridScheduler(plan.tornadoVMLayerPlanner.getGridScheduler())
                .execute();

        state.positionHolder.set(0, position);
        state.temp.clear();
        state.tempFFN.clear();

        // Graphs 1..N: transformer layers
        for (int layer = 1; layer <= config.numberOfLayers(); layer++) {
            plan.executionPlan.withGraph(layer)
                    .withGridScheduler(plan.tornadoVMLayerPlanner.getGridScheduler())
                    .execute();
        }

        // Graph N+1 (logits) intentionally skipped — not needed for prefill positions.
    }

    /**
     * GPU decode forward: full execution including logits.
     * Delegates to {@link TornadoVMMasterPlan#tornadoVMForwardExecuteLayered}.
     *
     * @param position sequence position being processed
     * @return logits array for token sampling
     */
    public FloatArray tornadoVMForwardDecode(int position) {
        return plan.tornadoVMForwardExecuteLayered(position);
    }

    /** Delegates to the wrapped plan's full forward pass (used by the standard decode path). */
    @Override
    public FloatArray tornadoVMForwardExecuteLayered(int position) {
        return tornadoVMForwardDecode(position);
    }

    @Override
    public void freeTornadoExecutionPlan() {
        plan.freeTornadoExecutionPlan();
    }
}
