package org.beehive.gpullama3.tornadovm.layers.type.q8_0;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.granite.GraniteConfiguration;
import org.beehive.gpullama3.tornadovm.kernels.GraniteKernels;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernels;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernelsLayered;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Granite-specific Q8_0 logits layer.
 * Identical to LogitsQ8_0Layer except vocab_proj uses a scaled kernel (logitScale).
 */
public class LogitsGraniteQ8_0Layer extends LogitsQ8_0Layer {

    public LogitsGraniteQ8_0Layer(String name, State state, Weights weights, Configuration config,
            String lastTaskGraphID, SchedulerType schedulerType) {
        super(name, state, weights, config, lastTaskGraphID, schedulerType);
    }

    // @formatter:off
    @Override
    protected TaskGraph setupLogitsTaskGraph(TornadoWeights weights, Configuration config) {
        GraniteConfiguration graniteCfg = (GraniteConfiguration) config;
        var logits = new TaskGraph("logits");

        // === Data Setup ===
        logits.consumeFromDevice(lastTaskGraphID, state.wrapX);
        logits.transferToDevice(DataTransferMode.EVERY_EXECUTION, state.tempLogits);
        logits.transferToDevice(DataTransferMode.FIRST_EXECUTION, context,
                                                                  state.wrapLogits,
                                                                  weights.wclsByteArray.asByteArray(),
                                                                  weights.rms_final_weight_as_floatArray);
        // === Final RMS Normalization ===
        logits.task("rms_reduce",
                TransformerComputeKernels::reductionOneBlockWithLayer,
                context,
                state.tempLogits,
                state.wrapX,
                config.dim(),
                config.rmsNormEps(),
                state.localSize);

        if (schedulerType == SchedulerType.NON_NVIDIA) {
            logits.task("rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    state.tempLogits,
                    config.dim(),
                    config.rmsNormEps());
        }

        logits.task("mapContextLogits",
                TransformerComputeKernels::reductionOneBlock2WithLogits,
                context,
                state.wrapX,
                weights.rms_final_weight_as_floatArray.asFloatArray(),
                state.tempLogits);

        // === Vocabulary Projection (Granite: scaled by logitScale) ===
        logits.task("vocab_proj",
                GraniteKernels::matrixVectorGenericQ8ByteWithGraniteScale,
                context,
                state.wrapX,
                state.wrapLogits,
                weights.wclsByteArray.asByteArray(),
                config.dim(),
                config.vocabularySize(),
                LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS,
                graniteCfg.logitScale());

        logits.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);
        return logits;
    }
    // @formatter:on
}
