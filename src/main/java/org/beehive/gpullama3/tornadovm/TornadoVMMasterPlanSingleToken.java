package org.beehive.gpullama3.tornadovm;

import org.beehive.gpullama3.auxiliary.RunMetrics;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.tornadovm.plan.ForwardPlanFactory;
import org.beehive.gpullama3.tornadovm.plan.SingleTokenForwardPlan;
import org.beehive.gpullama3.tornadovm.plan.layout.SingleTokenForwardTaskGraphLayout;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Standard (single-token) GPU execution plan.
 *
 * <p>Processes one token at a time through preprocessing + N transformer layers +
 * logits projection.
 * </p>
 */
public class TornadoVMMasterPlanSingleToken implements TornadoVMMasterPlan {

    private final State state;
    private final Model model;
    private final Configuration config;

    SingleTokenForwardPlan tornadoVMForwardPlan;
    SingleTokenForwardTaskGraphLayout taskGraphLayout;
    public TornadoExecutionPlan executionPlan;

    public TornadoVMMasterPlanSingleToken(State state, Model model) {
        if (ENABLE_TORNADOVM_INIT_TIME) {
            System.err.println("\nStarting TornadoVM initialization...");
        }

        this.state = state;
        this.model = model;
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

    @Override
    public TornadoExecutionPlan createExecutionPlan() {
        GGMLType weightType = model.weights().getWeightType();
        this.tornadoVMForwardPlan = ForwardPlanFactory.createSingleToken(weightType, state, model);
        this.taskGraphLayout = tornadoVMForwardPlan.getTaskGraphLayout();
        var taskGraphs = tornadoVMForwardPlan.getImmutableTaskGraphs();
        return new TornadoExecutionPlan(taskGraphs.toArray(new ImmutableTaskGraph[0]));
    }

    @Override
    public FloatArray tornadoVMForwardDecode(int position) {
        // @formatter:off
        var preGraph = executionPlan.withGraph(taskGraphLayout.activationIdx())
                .withGridScheduler(tornadoVMForwardPlan.getGridScheduler());
        if (CUDA_GRAPHS) preGraph.withCUDAGraph();
        preGraph.execute();

        state.positionHolder.set(0, position);
        state.temp.clear();
        state.tempFFN.clear();

        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            executionPlan.withGraph(taskGraphLayout.layerIdx(layer))
                    .withGridScheduler(tornadoVMForwardPlan.getGridScheduler())
                    .execute();
        }
        state.tempLogits.clear();
        state.wrapLogits.clear();
        var logitsGraph = executionPlan.withGraph(taskGraphLayout.logitsIdx())
                .withGridScheduler(tornadoVMForwardPlan.getGridScheduler());
        if (CUDA_GRAPHS) logitsGraph.withCUDAGraph();
        logitsGraph.execute();
        // @formatter:on
        return state.wrapLogits;
    }

    @Override
    public void forceCopyInReadOnlyData() {
        state.wrapX.clear();
        state.positionHolder.init(0);

        executionPlan.withGraph(taskGraphLayout.activationIdx()).withGridScheduler(tornadoVMForwardPlan.getGridScheduler()).execute();

        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            executionPlan.withGraph(taskGraphLayout.layerIdx(layer)).withGridScheduler(tornadoVMForwardPlan.getGridScheduler()).execute();
        }

        executionPlan.withGraph(taskGraphLayout.logitsIdx()).withGridScheduler(tornadoVMForwardPlan.getGridScheduler()).execute();
    }

    @Override
    public void freeTornadoExecutionPlan() {
        executionPlan.freeDeviceMemory();
    }
}
