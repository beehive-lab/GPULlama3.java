package org.beehive.gpullama3.tornadovm.layers.type.fp16;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.gemma4.Gemma4Configuration;
import org.beehive.gpullama3.tornadovm.kernels.Gemma4Kernels;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernels;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernelsLayered;
import org.beehive.gpullama3.tornadovm.scheduling.WorkerGridFactory;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Gemma4-specific FP16 logits layer.
 *
 * Identical to {@link LogitsFP16Layer} except for one addition: Gemma4 applies a final
 * logit soft-cap, {@code logits = softcap * tanh(logits / softcap)}, after the vocabulary
 * projection (see {@code gemma4.final_logit_softcapping} and
 * {@link org.beehive.gpullama3.inference.InferenceCore#forwardJavaGemma4}).
 */
public class Gemma4LogitsFP16Layer extends LogitsFP16Layer {

    private static final String SOFTCAP_TASK = "logit_softcap";

    public Gemma4LogitsFP16Layer(String name, State state, Weights weights, Configuration config,
            String lastTaskGraphID, SchedulerType schedulerType) {
        super(name, state, weights, config, lastTaskGraphID, schedulerType);
    }

    private float softcap() {
        return ((Gemma4Configuration) config).finalLogitSoftcapping();
    }

    // @formatter:off
    @Override
    protected TaskGraph setupLogitsTaskGraph(TornadoWeights weights, Configuration config) {
        var logits = new TaskGraph("logits");
        // === Data Setup ===
        logits.consumeFromDevice(lastTaskGraphID, state.wrapX);
        logits.transferToDevice(DataTransferMode.EVERY_EXECUTION, state.tempLogits);
        logits.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                context,
                state.wrapLogits,
                state.wrapXbFP16,
                weights.wclsByteArray.asHalfFloatArray(),
                weights.rms_final_weight_as_floatArray.asFloatArray());

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

        logits.task("rms_apply_fp16",
                TransformerComputeKernels::mapContextWithQuantizeLogits,
                context,
                state.wrapXbFP16,
                state.wrapX,
                weights.rms_final_weight_as_floatArray.asFloatArray(),
                state.tempLogits);

        // === Vocabulary Projection ===
        logits.task("vocab_proj",
                TransformerComputeKernelsLayered::matrixVectorGeneric,
                context,
                state.wrapXbFP16,
                state.wrapLogits,
                weights.wclsByteArray.asHalfFloatArray(),
                config.dim(),
                config.vocabularySize(),
                LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS);

        // === Final logit soft-capping (Gemma4-specific) ===
        if (softcap() != 0.0f) {
            logits.task(SOFTCAP_TASK,
                    Gemma4Kernels::applyLogitSoftcap,
                    context,
                    state.wrapLogits,
                    softcap(),
                    config.vocabularySize());
        }

        // === Transfer Results to Host ===
        logits.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);
        return logits;
    }
    // @formatter:on

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        var scheduler = super.updateGridScheduler(tornadoForwardScheduler);
        if (softcap() != 0.0f) {
            scheduler.addWorkerGrid("logits." + SOFTCAP_TASK, WorkerGridFactory.genericWorker(config.vocabularySize(), LOCAL_WORK_GROUP_SIZE_ALLOC));
        }
        return scheduler;
    }
}
