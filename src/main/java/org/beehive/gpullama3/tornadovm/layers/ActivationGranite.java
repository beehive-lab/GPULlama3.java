package org.beehive.gpullama3.tornadovm.layers;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.granite.GraniteConfiguration;
import org.beehive.gpullama3.tornadovm.kernels.GraniteKernels;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

public class ActivationGranite extends Activation {
    private final TaskGraph activationUpdate;

    // Granite is a special case where activation X is scaled by embedding scale float value that inside model.
    public ActivationGranite(String taskGraphHandle, State state, Weights weights, GraniteConfiguration config) {
        super(taskGraphHandle, state, weights, config);

        KernelContext kernelContext = new KernelContext();

        // @formatter:off
        switch (config.quantization()) {
            case "FP16" -> {
                this.activationUpdate = new TaskGraph(taskGraphHandle)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.embeddingX)
                        .task("updateX", GraniteKernels::convertFP16toFP32withGraniteScale, kernelContext, (HalfFloatArray) state.embeddingX, state.wrapX,  config.embeddingScale())
                        .persistOnDevice(state.wrapX);
            }
            case "Q8_0" -> {
                this.activationUpdate = new TaskGraph(taskGraphHandle)
                        .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.embeddingX)
                        .task("updateX", GraniteKernels::convertQ8_0toFP32withGraniteScale, kernelContext, (ByteArray) state.embeddingX, state.wrapX, config.embeddingScale())
                        .persistOnDevice(state.wrapX);
            }
            default -> throw new UnsupportedOperationException("Unsupported quantization format: " + config.quantization());
        }
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
