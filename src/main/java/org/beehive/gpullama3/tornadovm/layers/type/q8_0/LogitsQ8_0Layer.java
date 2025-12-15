package org.beehive.gpullama3.tornadovm.layers.type.q8_0;

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
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

public class LogitsQ8_0Layer extends AbstractLayer {

    private String lastTaskGraphID;
    private TaskGraph logitsTaskGraph;
    private ImmutableTaskGraph immutableLogitsGraph;
    private GridScheduler scheduler;
    private SchedulerType schedulerType;

    public LogitsQ8_0Layer(String taskGraphName, State state, Weights weights, Configuration config, String lastTaskGraphID, SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config);
        this.lastTaskGraphID = lastTaskGraphID;
        var tornadoWeights = requireWeightsType(weights, TornadoWeights.class, "LogitsQ8_0Layer", "TornadoTensor");
        this.logitsTaskGraph = setupLogitsTaskGraph(tornadoWeights, config);
        this.schedulerType = schedulerType;
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        var logitsRMS = WorkerGridFactory.createRmsNormWorker(config.dim(), weights instanceof Qwen2TornadoWeights ? 32 : 256);
        var vocabSizeRowMajor = config.vocabularySize() * LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS;
        var vocabWorker = new WorkerGrid1D(vocabSizeRowMajor);
        vocabWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS, 1, 1);
        tornadoForwardScheduler.addWorkerGrid("logits.vocab_proj", vocabWorker);
        tornadoForwardScheduler.addWorkerGrid("logits.rms_reduce", logitsRMS);
        tornadoForwardScheduler.addWorkerGrid("logits.mapContextLogits", logitsRMS);
        return tornadoForwardScheduler;
    }

    // @formatter:off
    private TaskGraph setupLogitsTaskGraph(TornadoWeights weights, Configuration config) {
        var logits = new TaskGraph("logits");
        // === Data Setup ===
        logits.consumeFromDevice(lastTaskGraphID, state.wrapX);
        logits.transferToDevice(DataTransferMode.EVERY_EXECUTION, state.tempLogits);
        logits.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        context, //
                        state.wrapLogits,  //
                        weights.wclsByteArray.asByteArray(), //
                        weights.rms_final_weight_as_floatArray);

        // === Final RMS Normalization ===
        logits.task("rms_reduce",
                TransformerComputeKernels::reductionOneBlockWithLayer,
                context,
                state.tempLogits,  // output: partial sums + final scale factor
                state.wrapX,        // input: hidden state
                config.dim(),        // dimension
                config.rmsNormEps(),   // epsilon for numerical stability
                state.localSize);    // local workgroup size

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
        
        // === Vocabulary vocab_proj ===
        logits.task("vocab_proj", TransformerComputeKernelsLayered::matrixVectorGenericQ8Byte,  //
                context, 
                state.wrapX, 
                state.wrapLogits, 
                weights.wclsByteArray.asByteArray(), 
                config.dim(), 
                config.vocabularySize(), 
                LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS);

        // === Transfer Results to Host ===
        logits.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);
        return logits;
    }
    // @formatter:on

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
