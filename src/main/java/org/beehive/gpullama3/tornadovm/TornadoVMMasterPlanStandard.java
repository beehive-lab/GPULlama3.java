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

    public static final boolean ENABLE_TORNADOVM_INIT_TIME = Boolean.parseBoolean(System.getProperty("llama.EnableTimingForTornadoVMInit", "False"));

    private final State state;
    private final Configuration config;
    public TornadoExecutionPlan executionPlan;
    GenericLayerPlanner tornadoVMLayerPlanner;

    public TornadoVMMasterPlanStandard(State state, Model model) {
        this.tornadoVMLayerPlanner = createPlanner(state, model);
        this.executionPlan = createExecutionPlan();
        this.state = state;
        this.config = model.configuration();
    }

    /**
     * Initializes and warms up the standard TornadoVM plan.
     *
     * @param state the model state containing KV cache
     * @param model the model instance
     * @return the initialized plan ready for inference
     */
    static TornadoVMMasterPlanStandard initialize(State state, Model model) {
        long startTime = System.nanoTime();
        long planCreationTime = 0;
        long warmupTime = 0;

        if (ENABLE_TORNADOVM_INIT_TIME) {
            System.err.println("\nStarting TornadoVM initialization...");
        }

        TornadoVMMasterPlanStandard tornadoVMPlan = new TornadoVMMasterPlanStandard(state, model);

        if (ENABLE_TORNADOVM_INIT_TIME) {
            planCreationTime = System.nanoTime();
            System.err.printf("TornadoVM GPU execution plan creation: %.2f ms\n", (planCreationTime - startTime) / 1_000_000.0);
        }

        if (CUDA_GRAPHS) tornadoVMPlan.executionPlan.withAllGraphs().withCUDAGraph();
        tornadoVMPlan.executionPlan.withPreCompilation();

        if (ENABLE_TORNADOVM_INIT_TIME) {
            warmupTime = System.nanoTime();
            System.err.printf("Java to GPU JIT compiler warmup: %.2f ms\n", (warmupTime - planCreationTime) / 1_000_000.0);
        }

        tornadoVMPlan.forceCopyInReadOnlyDataLayered();

        if (ENABLE_TORNADOVM_INIT_TIME) {
            long copyTime = System.nanoTime();
            System.err.printf("Transfer read-only weights to GPU: %.2f ms\n", (copyTime - warmupTime) / 1_000_000.0);
            System.err.printf("Finished TornadoVM initialization...\n \n");
        }

        return tornadoVMPlan;
    }

    private TornadoExecutionPlan createExecutionPlan() {
        var taskGraphs = tornadoVMLayerPlanner.getImmutableTaskGraphs();
        var taskGraphArray = taskGraphs.toArray(new ImmutableTaskGraph[taskGraphs.size()]);
        return new TornadoExecutionPlan(taskGraphArray);
    }

    private GenericLayerPlanner createPlanner(State state, Model model) {
        GGMLType weightType = model.weights().getWeightType();
        return QuantizationPlannerFactory.create(weightType, state, model);
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

    public void forceCopyInReadOnlyDataLayered() {
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
