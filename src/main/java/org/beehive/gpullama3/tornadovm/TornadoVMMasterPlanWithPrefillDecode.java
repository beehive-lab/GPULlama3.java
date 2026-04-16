package org.beehive.gpullama3.tornadovm;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.tornadovm.layerplanner.GenericLayerPlanner;
import org.beehive.gpullama3.tornadovm.layerplanner.QuantizationPlannerFactory;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * GPU execution plan for single-token prefill/decode separation.
 *
 * <p>Uses the same single-token execution plan as {@link TornadoVMMasterPlanStandard}
 * but exposes two distinct forward passes:</p>
 * <ul>
 *   <li>{@link #tornadoVMForwardPrefill} — runs graphs 0..N, skips the logits graph.
 *       Called for each prompt token; KV cache is populated but logits are discarded.</li>
 *   <li>{@link #tornadoVMForwardDecode} — full execution including logits.
 *       Called for each generated token.</li>
 * </ul>
 *
 * <p>Graph layout (same as {@link TornadoVMMasterPlanStandard}):</p>
 * <pre>
 *   graph 0         : preprocessing (embedding setup)
 *   graphs 1..N     : transformer layers
 *   graph N+1       : logits projection (final RMSNorm + wcls matmul)
 * </pre>
 */
public class TornadoVMMasterPlanWithPrefillDecode implements TornadoVMMasterPlan {

    private final State state;
    private final Model model;
    private final Configuration config;

    GenericLayerPlanner tornadoVMLayerPlanner;
    public TornadoExecutionPlan executionPlan;

    public TornadoVMMasterPlanWithPrefillDecode(State state, Model model) {
        long startTime = System.nanoTime();
        long planCreationTime = 0;
        long warmupTime = 0;

        if (ENABLE_TORNADOVM_INIT_TIME) {
            System.err.println("\nStarting TornadoVM initialization...");
        }

        this.state = state;
        this.model = model;
        this.config = model.configuration();

        this.executionPlan = createExecutionPlan();

        if (ENABLE_TORNADOVM_INIT_TIME) {
            planCreationTime = System.nanoTime();
            System.err.printf("TornadoVM GPU single-token prefill/decode execution plan creation: %.2f ms\n", (planCreationTime - startTime) / 1_000_000.0);
        }

        if (CUDA_GRAPHS) executionPlan.withAllGraphs().withCUDAGraph();
        executionPlan.withPreCompilation();

        if (ENABLE_TORNADOVM_INIT_TIME) {
            warmupTime = System.nanoTime();
            System.err.printf("Java to GPU JIT compiler warmup: %.2f ms\n", (warmupTime - planCreationTime) / 1_000_000.0);
        }

        forceCopyInReadOnlyData();

        if (ENABLE_TORNADOVM_INIT_TIME) {
            long copyTime = System.nanoTime();
            System.err.printf("Transfer read-only weights to GPU: %.2f ms\n", (copyTime - warmupTime) / 1_000_000.0);
            System.err.printf("Finished TornadoVM initialization...\n \n");
        }
    }

    /**
     * Creates the {@link TornadoExecutionPlan} for forward pass with *prefill/decode separation*.
     * Prefill is token-by-token but does not compute logits.
     */
    @Override
    public TornadoExecutionPlan createExecutionPlan() {
        GGMLType weightType = model.weights().getWeightType();
        this.tornadoVMLayerPlanner = QuantizationPlannerFactory.create(weightType, state, model);
        var taskGraphs = tornadoVMLayerPlanner.getImmutableTaskGraphs();
        var taskGraphArray = taskGraphs.toArray(new ImmutableTaskGraph[taskGraphs.size()]);
        return new TornadoExecutionPlan(taskGraphArray);
    }

    @Override
    public void forceCopyInReadOnlyData() {
        state.wrapX.clear();
        state.positionHolder.init(0);

        executionPlan.withGraph(0).withGridScheduler(tornadoVMLayerPlanner.getGridScheduler()).execute();

        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            executionPlan.withGraph(layer + 1).withGridScheduler(tornadoVMLayerPlanner.getGridScheduler()).execute();
        }

        executionPlan.withGraph(config.numberOfLayers() + 1).withGridScheduler(tornadoVMLayerPlanner.getGridScheduler()).execute();
    }

    /**
     * GPU prefill forward: runs preprocessing + all transformer layers, skips logits.
     * KV cache is populated; logits projection is intentionally omitted.
     *
     * @param position sequence position being processed
     */
    public void tornadoVMForwardPrefill(int position) {
        // Graph 0: preprocessing
        executionPlan.withGraph(0)
                .withGridScheduler(tornadoVMLayerPlanner.getGridScheduler())
                .execute();

        state.positionHolder.set(0, position);
        state.temp.clear();
        state.tempFFN.clear();

        // Graphs 1..N: transformer layers (logits graph N+1 intentionally skipped)
        for (int layer = 1; layer <= config.numberOfLayers(); layer++) {
            executionPlan.withGraph(layer)
                    .withGridScheduler(tornadoVMLayerPlanner.getGridScheduler())
                    .execute();
        }
    }

    /**
     * GPU decode forward: full execution including logits.
     *
     * @param position sequence position being processed
     * @return logits array for token sampling
     */
    public FloatArray tornadoVMForwardDecode(int position) {
        return tornadoVMForwardExecuteLayered(position);
    }

    @Override
    public FloatArray tornadoVMForwardExecuteLayered(int position) {
        var preGraph = executionPlan.withGraph(0)
                .withGridScheduler(tornadoVMLayerPlanner.getGridScheduler());
        if (CUDA_GRAPHS) preGraph.withCUDAGraph();
        preGraph.execute();

        state.positionHolder.set(0, position);
        state.temp.clear();
        state.tempFFN.clear();

        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            executionPlan.withGraph(1 + layer)
                    .withGridScheduler(tornadoVMLayerPlanner.getGridScheduler())
                    .execute();
        }

        state.tempLogits.clear();
        state.wrapLogits.clear();
        var logitsGraph = executionPlan.withGraph(config.numberOfLayers() + 1)
                .withGridScheduler(tornadoVMLayerPlanner.getGridScheduler());
        if (CUDA_GRAPHS) logitsGraph.withCUDAGraph();
        logitsGraph.execute();

        return state.wrapLogits;
    }

    @Override
    public void freeTornadoExecutionPlan() {
        executionPlan.freeDeviceMemory();
    }
}
