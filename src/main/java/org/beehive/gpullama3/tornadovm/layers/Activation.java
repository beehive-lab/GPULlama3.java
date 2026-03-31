package org.beehive.gpullama3.tornadovm.layers;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernels;
import org.beehive.gpullama3.tornadovm.layerplanner.WorkerGridFactory;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

public class Activation extends AbstractLayer {
    private final TaskGraph activationTaskGraph;

    public Activation(String name, State state, Weights weights, Configuration config) {
        super(name, state, weights, config);
        this.activationTaskGraph = setupActivationTaskGraph(name);
    }

    // @formatter:off
    protected TaskGraph setupActivationTaskGraph(String name) {
        return switch (config.quantization()) {
            case "FP16" -> new TaskGraph(name)
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.embeddingX)
                    .task("updateX", TransformerComputeKernels::convertFP16toFP32, context, (HalfFloatArray) state.embeddingX, state.wrapX)
                    .persistOnDevice(state.wrapX);
            case "Q8_0" -> new TaskGraph(name)
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.embeddingX)
                    .task("updateX", TransformerComputeKernels::convertQ8_0toFP32, context, (ByteArray) state.embeddingX, state.wrapX)
                    .persistOnDevice(state.wrapX);
            default -> throw new UnsupportedOperationException("Unsupported quantization format: " + config.quantization());
        };
    }
    // @formatter:on

    @Override
    public GridScheduler updateGridScheduler(GridScheduler scheduler) {
        WorkerGrid worker = WorkerGridFactory.genericWorker(config.dim(), 128);
        scheduler.addWorkerGrid(activationTaskGraph.getTaskGraphName() + ".updateX", worker);
        return scheduler;
    }

    public TaskGraph getTaskGraph() {
        return activationTaskGraph;
    }

    public ImmutableTaskGraph getImmutableTaskGraph() {
        return activationTaskGraph.snapshot();
    }

}

