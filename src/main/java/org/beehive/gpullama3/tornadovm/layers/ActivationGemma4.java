package org.beehive.gpullama3.tornadovm.layers;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.gemma4.Gemma4Configuration;
import org.beehive.gpullama3.tornadovm.kernels.GraniteKernels;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * Gemma4-specific activation: applies sqrt(dim) embedding scale during FP32 conversion.
 * Reuses GraniteKernels' scale-aware conversion kernels.
 */
public class ActivationGemma4 extends Activation {

    public ActivationGemma4(String taskGraphHandle, State state, Weights weights, Gemma4Configuration config) {
        super(taskGraphHandle, state, weights, config);
    }

    // @formatter:off
    @Override
    protected TaskGraph setupActivationTaskGraph(String handle) {
        Gemma4Configuration cfg = (Gemma4Configuration) config;
        float embeddingScale = (float) Math.sqrt(cfg.dim());
        return switch (config.quantization()) {
            case "FP16" -> new TaskGraph(handle)
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.embeddingX)
                    .task("updateX", GraniteKernels::convertFP16toFP32withGraniteScale, context, (HalfFloatArray) state.embeddingX, state.wrapX, embeddingScale)
                    .persistOnDevice(state.wrapX);
            case "Q8_0" -> new TaskGraph(handle)
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.embeddingX)
                    .task("updateX", GraniteKernels::convertQ8_0toFP32withGraniteScale, context, (ByteArray) state.embeddingX, state.wrapX, embeddingScale)
                    .persistOnDevice(state.wrapX);
            default -> throw new UnsupportedOperationException("Unsupported quantization format: " + config.quantization());
        };
    }
    // @formatter:on
}
