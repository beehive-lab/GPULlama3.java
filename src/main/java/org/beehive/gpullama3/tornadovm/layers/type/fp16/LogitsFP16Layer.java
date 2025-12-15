package org.beehive.gpullama3.tornadovm.layers.type.fp16;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.tornado.Qwen2TornadoWeights;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernels;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernelsLayered;
import org.beehive.gpullama3.tornadovm.layerplanner.WorkerGridFactory;
import org.beehive.gpullama3.tornadovm.layerplanner.strategy.SchedulerType;
import org.beehive.gpullama3.tornadovm.layers.AbstractLayer;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

public class LogitsFP16Layer extends AbstractLayer {

    private String lastTaskGraphID;
    private TaskGraph logitsTaskGraph;
    private ImmutableTaskGraph immutableLogitsGraph;
    private GridScheduler scheduler;
    private SchedulerType schedulerType;

    public LogitsFP16Layer(String name, State state, Weights weights, Configuration config, String lastTaskGraphID, SchedulerType schedulerType) {
        super(name, state, weights, config);
        this.lastTaskGraphID = lastTaskGraphID;
        this.schedulerType = schedulerType;
        var tornadoWeights = requireWeightsType(weights, TornadoWeights.class, "LogitsFP16Layer", "TornadoTensor");
        this.logitsTaskGraph = setupLogitsTaskGraph(tornadoWeights, config);
    }

    // @formatter:off
    private TaskGraph setupLogitsTaskGraph(TornadoWeights weights, Configuration config) {
        var logits = new TaskGraph("logits");
        // === Data Setup ===
        logits.consumeFromDevice(lastTaskGraphID, state.wrapX);
        logits.transferToDevice(DataTransferMode.EVERY_EXECUTION, state.tempLogits);
        logits.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                // Kernel context
                context,
                // Output buffer
                state.wrapLogits,
                // Intermediate FP16 buffer
                state.wrapXbFP16,
                // Weights
                weights.wclsByteArray.asHalfFloatArray(),
                weights.rms_final_weight_as_floatArray.asFloatArray());

        // === Final RMS Normalization ===
        logits.task("rms_reduce",
                TransformerComputeKernels::reductionOneBlockWithLayer,
                context,
                state.tempLogits,        // output: partial sums + final scale factor
                state.wrapX,             // input: hidden state
                config.dim(),            // dimension
                config.rmsNormEps(),     // epsilon for numerical stability
                state.localSize);        // local workgroup size

        if (schedulerType == SchedulerType.NON_NVIDIA) {
            logits.task("rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    state.tempLogits,    // in/out: combines partial sums
                    config.dim(),        // dimension
                    config.rmsNormEps()); // epsilon
        }

        logits.task("rms_apply_fp16",
                TransformerComputeKernels::mapContextWithQuantizeLogits,
                context,
                state.wrapXbFP16,        // output: normalized (FP16)
                state.wrapX,             // input: hidden state
                weights.rms_final_weight_as_floatArray.asFloatArray(),  // RMS weights
                state.tempLogits);       // scale factor from reduction

        // === Vocabulary Projection ===
        logits.task("vocab_proj",
                TransformerComputeKernelsLayered::matrixVectorGeneric,
                context,
                state.wrapXbFP16,                              // input (FP16)
                state.wrapLogits,                              // output
                weights.wclsByteArray.asHalfFloatArray(),      // vocabulary weights
                config.dim(),                                  // input dimension
                config.vocabularySize(),                       // output dimension
                LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS);

        // === Transfer Results to Host ===
        logits.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);
        return logits;
    }
    // @formatter:on

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        WorkerGrid logitsRMS = WorkerGridFactory.createRmsNormWorker(config.dim(), weights instanceof Qwen2TornadoWeights ? 32 : 256);
        var vocabSizeRowMajor = config.vocabularySize() * LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS;
        var vocabWorker = new WorkerGrid1D(vocabSizeRowMajor);
        vocabWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS, 1, 1);
        tornadoForwardScheduler.addWorkerGrid("logits.rms_reduce", logitsRMS);
        tornadoForwardScheduler.addWorkerGrid("logits.rms_apply_fp16", logitsRMS);
        tornadoForwardScheduler.addWorkerGrid("logits.vocab_proj", vocabWorker);
        return tornadoForwardScheduler;
    }

    @Override
    public GridScheduler getGridScheduler() {
        return scheduler;
    }

    @Override
    public TaskGraph getTaskGraph() {
        return logitsTaskGraph;
    }

    @Override
    public ImmutableTaskGraph getImmutableTaskGraph() {
        return immutableLogitsGraph;
    }
}
