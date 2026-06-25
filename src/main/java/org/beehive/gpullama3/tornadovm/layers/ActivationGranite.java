package org.beehive.gpullama3.tornadovm.layers;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.granite.GraniteConfiguration;
import org.beehive.gpullama3.tornadovm.kernels.GraniteKernels;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * Granite-specific activation: applies an embedding scale factor during the FP32 conversion.
 * Overrides only the TaskGraph builder; all other behaviour is inherited from Activation.
 */
public class ActivationGranite extends Activation {

    // Granite is a special case where activation X is scaled by embedding scale float value that inside model.
    public ActivationGranite(String taskGraphHandle, State state, Weights weights, GraniteConfiguration config) {
        super(taskGraphHandle, state, weights, config);
    }

    // @formatter:off
    @Override
    protected TaskGraph setupActivationTaskGraph(String handle) {
        GraniteConfiguration cfg = (GraniteConfiguration) config;
        return switch (config.quantization()) {
            case "FP16" -> new TaskGraph(handle)
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.embeddingX)
                    .task("updateX", GraniteKernels::convertFP16toFP32withGraniteScale, context, (HalfFloatArray) state.embeddingX, state.wrapX, cfg.embeddingScale())
                    .persistOnDevice(state.wrapX);
            case "Q8_0" -> new TaskGraph(handle)
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.embeddingX)
                    .task("updateX", GraniteKernels::convertQ8_0toFP32withGraniteScale, context, (ByteArray) state.embeddingX, state.wrapX, cfg.embeddingScale())
                    .persistOnDevice(state.wrapX);
            default -> throw new UnsupportedOperationException("Unsupported quantization format: " + config.quantization());
        };
    }
    // @formatter:on
}
