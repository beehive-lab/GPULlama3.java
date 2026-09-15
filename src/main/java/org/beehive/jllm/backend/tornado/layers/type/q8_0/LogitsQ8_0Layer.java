package org.beehive.gpullama3.backend.tornado.layers.type.q8_0;

import org.beehive.gpullama3.backend.tornado.kernels.TransformerComputeKernels;
import org.beehive.gpullama3.backend.tornado.kernels.TransformerComputeKernelsLayered;
import org.beehive.gpullama3.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.gpullama3.backend.tornado.scheduling.SchedulerType;
import org.beehive.gpullama3.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.tornado.Qwen2TornadoWeights;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

public class LogitsQ8_0Layer extends AbstractLogitsTaskGraph {

    // @formatter:off
    public LogitsQ8_0Layer(
            String name,
            State state,
            Weights weights,
            Configuration config,
            String lastTaskGraphID,
            SchedulerType schedulerType) {
        super(name, state, weights, config, lastTaskGraphID, schedulerType);
    }

    // @formatter:on

    protected void configureAdditionalConsumes(TaskGraph logits) {}

    protected void configureAdditionalPersists(TaskGraph logits) {}

    // @formatter:off
    @Override
    protected TaskGraph setupLogitsTaskGraph(TornadoWeights weights, Configuration config) {
        var logits = new TaskGraph("logits");

        // === Data Setup ===
        configureAdditionalConsumes(logits);
        logits.consumeFromDevice(lastTaskGraphID, state.workspace.wrapX);
        logits.transferToDevice(DataTransferMode.EVERY_EXECUTION, state.workspace.tempLogits);
        logits.transferToDevice(
                DataTransferMode.FIRST_EXECUTION,
                context,
                state.workspace.wrapLogits,
                weights.wclsByteArray.asByteArray(),
                weights.rms_final_weight_as_floatArray);
        // === Final RMS Normalization ===
        logits.task(
                "rms_reduce",
                rmsReduceKernel(),
                context,
                state.workspace.tempLogits, // output: partial sums + final scale factor
                state.workspace.wrapX, // input: hidden state
                config.dim(),
                config.rmsNormEps(),
                state.localSize);

        if (schedulerType == SchedulerType.NON_NVIDIA) {
            logits.task(
                    "rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    state.workspace.tempLogits,
                    config.dim(),
                    config.rmsNormEps());
        }

        logits.task(
                "mapContextLogits",
                TransformerComputeKernels::reductionOneBlock2WithLogits,
                context,
                state.workspace.wrapX,
                weights.rms_final_weight_as_floatArray.asFloatArray(),
                state.workspace.tempLogits);

        // === Vocabulary Projection ===
        logits.task(
                "vocab_proj",
                TransformerComputeKernelsLayered::matrixVectorGenericQ8Byte,
                context,
                state.workspace.wrapX,
                state.workspace.wrapLogits,
                weights.wclsByteArray.asByteArray(),
                config.dim(),
                config.vocabularySize(),
                LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS);

        logits.transferToHost(DataTransferMode.EVERY_EXECUTION, state.workspace.wrapLogits);
        configureAdditionalPersists(logits);
        return logits;
    }

    // @formatter:on

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        var logitsRMS = WorkerGridFactory.createRmsNormWorker(config.dim(), rmsLocalSize());
        var vocabSizeRowMajor =
                config.vocabularySize() * LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS;
        var vocabWorker = new WorkerGrid1D(vocabSizeRowMajor);
        vocabWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS, 1, 1);
        tornadoForwardScheduler.addWorkerGrid("logits.vocab_proj", vocabWorker);
        tornadoForwardScheduler.addWorkerGrid("logits.rms_reduce", rmsReduceWorker(logitsRMS));
        tornadoForwardScheduler.addWorkerGrid("logits.mapContextLogits", logitsRMS);
        return tornadoForwardScheduler;
    }

    /** Local workgroup size for RMS norm. Qwen2 requires a smaller group (32 vs 256). */
    protected int rmsLocalSize() {
        return weights instanceof Qwen2TornadoWeights ? 32 : 256;
    }
}
