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
 * Standard (single-token) GPU execution plan.
 *
 * <p>Processes one token at a time through preprocessing + N transformer layers +
 * logits projection.  Used for both the baseline GPU path and the Phase 2
 * sequential prefill/decode path.</p>
 */
public class TornadoVMMasterPlanStandard implements TornadoVMMasterPlan {

    private final State state;
    private final Model model;
    private final Configuration config;

    GenericLayerPlanner tornadoVMLayerPlanner;
    public TornadoExecutionPlan executionPlan;

    public TornadoVMMasterPlanStandard(State state, Model model) {
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
            System.err.printf("TornadoVM GPU standard execution plan creation: %.2f ms\n", (planCreationTime - startTime) / 1_000_000.0);
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

//    @Override
//    public GenericLayerPlanner createPlanner() {
//        GGMLType weightType = model.weights().getWeightType();
//        return QuantizationPlannerFactory.create(weightType, state, model);
//    }

    /**
     * Creates the {@link TornadoExecutionPlan} for *simple/standard* single-token forward pass.
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
    public FloatArray tornadoVMForwardExecuteLayered(int position) {
        // @formatter:off
        var preGraph = executionPlan.withGraph(getPreprocessingGraphIndex())
                .withGridScheduler(tornadoVMLayerPlanner.getGridScheduler());
        if (CUDA_GRAPHS) preGraph.withCUDAGraph();
        preGraph.execute();

        state.positionHolder.set(0, position);
        state.temp.clear();
        state.tempFFN.clear();

        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            executionPlan.withGraph(getLayerGraphIndex(layer))
                    .withGridScheduler(tornadoVMLayerPlanner.getGridScheduler())
                    //.withCUDAGraph()
                    .execute();
        }
        state.tempLogits.clear();
        state.wrapLogits.clear();
        var logitsGraph = executionPlan.withGraph(getFinalLogitsGraphIndex())
                .withGridScheduler(tornadoVMLayerPlanner.getGridScheduler());
        if (CUDA_GRAPHS) logitsGraph.withCUDAGraph();
        logitsGraph.execute();
        // @formatter:on
        return state.wrapLogits;
    }

    private int getPreprocessingGraphIndex() {
        return 0;
    }

    private int getLayerGraphIndex(int layerIndex) {
        return 1 + layerIndex;
    }

    private int getFinalLogitsGraphIndex() {
        return tornadoVMLayerPlanner.getImmutableTaskGraphs().size() - 1;
    }

    @Override
    public void forceCopyInReadOnlyData() {
        state.wrapX.clear();
        state.positionHolder.init(0);

        //executionPlan.withGraph(0).withGridScheduler(tornadoVMLayerPlanner.getGridScheduler()).withCUDAGraph().execute();
        executionPlan.withGraph(0).withGridScheduler(tornadoVMLayerPlanner.getGridScheduler()).execute();

        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            //executionPlan.withGraph(layer + 1).withGridScheduler(tornadoVMLayerPlanner.getGridScheduler()).withCUDAGraph().execute();
            executionPlan.withGraph(layer + 1).withGridScheduler(tornadoVMLayerPlanner.getGridScheduler()).execute();
        }

        //executionPlan.withGraph(config.numberOfLayers() + 1).withGridScheduler(tornadoVMLayerPlanner.getGridScheduler()).withCUDAGraph().execute();
        executionPlan.withGraph(config.numberOfLayers() + 1).withGridScheduler(tornadoVMLayerPlanner.getGridScheduler()).execute();
    }

    @Override
    public void freeTornadoExecutionPlan() {
        executionPlan.freeDeviceMemory();
    }
}
