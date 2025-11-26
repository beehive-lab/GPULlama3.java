package org.beehive.gpullama3.tornadovm.layers;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernels;
import org.beehive.gpullama3.tornadovm.layerplanner.WorkerGridFactory;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

public class Activation extends AbstractLayer {
    private final TaskGraph activationUpdate;

    public Activation(String taskGraphHandle, State state, Weights weights, Configuration config) {
        super(taskGraphHandle, state, weights, config);

        KernelContext kernelContext = new KernelContext();
        // @formatter:off
        this.activationUpdate = new TaskGraph(taskGraphHandle)
            .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.embeddingX)
            .task("updateX", TransformerComputeKernels::convertFP16toFP32, kernelContext, state.embeddingX, state.wrapX)
            .persistOnDevice(state.wrapX);
        // @formatter:on
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler scheduler) {
        WorkerGrid worker = new WorkerGrid1D(config.dim());
        worker.setLocalWork(128, 1, 1);
        scheduler.addWorkerGrid("activationUpdate.updateX", worker);
        return scheduler;
    }

    @Override
    public GridScheduler getGridScheduler() {
        return null;
    }

    @Override
    public TaskGraph getTaskGraph() {
        return activationUpdate;
    }

    @Override
    public ImmutableTaskGraph getImmutableTaskGraph() {
        return activationUpdate.snapshot();
    }

}

