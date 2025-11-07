package org.beehive.gpullama3.tornadovm.layers.type.q8_0;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.tornado.q8_0.LlamaTornadoWeightsQ8_0;
import org.beehive.gpullama3.inference.weights.tornado.q8_0.Qwen3TornadoWeightsQ8_0;
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

public class LogitsQ8_0Layer extends AbstractLayer {

    private String lastTaskGraphID;
    private TaskGraph logitsTaskGraph;
    private ImmutableTaskGraph immutableLogitsGraph;
    private GridScheduler scheduler;
    private final SchedulerType schedulerType;

    public LogitsQ8_0Layer(String taskGraphName, State state, Weights weights, Configuration config, String lastTaskGraphID, SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config);
        this.lastTaskGraphID = lastTaskGraphID;
        this.schedulerType = schedulerType;
        state.tempLogits.init(0.0f);
        var q8_0Weights = requireWeightsType(weights, LlamaTornadoWeightsQ8_0.class, "LogitsQ8_0Layer", "Q8_0");
        this.logitsTaskGraph = setupLogitsTaskGraph(q8_0Weights, config);
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        WorkerGrid logitsRMS;
        if (weights instanceof Qwen3TornadoWeightsQ8_0) {
            logitsRMS = WorkerGridFactory.createRmsNormWorker(config.dim(), 32);
        } else {
            logitsRMS = WorkerGridFactory.createRmsNormWorker(config.dim(), 256);
        }

        var vocabSizeRowMajor = config.vocabularySize() * LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS;
        WorkerGrid vocabWorker = new WorkerGrid1D(vocabSizeRowMajor);
        vocabWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS, 1, 1);

        tornadoForwardScheduler.addWorkerGrid("logits.projection", vocabWorker);
        tornadoForwardScheduler.addWorkerGrid("logits.reductionsOneBlockLogits", logitsRMS);
        tornadoForwardScheduler.addWorkerGrid("logits.mapContextLogits", logitsRMS);
        return tornadoForwardScheduler;
    }

    private TaskGraph setupLogitsTaskGraph(LlamaTornadoWeightsQ8_0 weights, Configuration config) {
        TaskGraph logits = new TaskGraph("logits");
        logits.consumeFromDevice(lastTaskGraphID, state.wrapX).transferToDevice(DataTransferMode.EVERY_EXECUTION, state.tempLogits)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, context, state.wrapLogits, weights.wclsHalfFloat.getQuants(), weights.wclsHalfFloat.getScales(),
                        weights.rms_final_weight_as_floatArray);
        logits.task("reductionsOneBlockLogits", TransformerComputeKernels::reductionOneBlockWithLayer, context, state.tempLogits, state.wrapX, config.dim(), config.rmsNormEps(), state.localSize);
        if (schedulerType == SchedulerType.NON_NVIDIA) {
            logits.task("reductionFinalNormalizationLogits", TransformerComputeKernelsLayered::reductionFinalNormalization, context, state.tempLogits,
                    config.dim(), config.rmsNormEps());
        }
        logits.task("mapContextLogits", TransformerComputeKernels::reductionOneBlock2WithLogits, context, state.wrapX, weights.rms_final_weight_as_floatArray, state.tempLogits)
                .task("projection", TransformerComputeKernelsLayered::matrixVectorGeneric,
                        context, state.wrapX, state.wrapLogits, weights.wclsHalfFloat.getQuants(), weights.wclsHalfFloat.getScales(),
                        config.dim(), config.vocabularySize(), LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);
        return logits;
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
