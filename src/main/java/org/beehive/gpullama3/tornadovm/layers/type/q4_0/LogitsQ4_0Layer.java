package org.beehive.gpullama3.tornadovm.layers.type.q4_0;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.inference.weights.tornado.Qwen2TornadoWeights;
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

import java.util.SequencedCollection;

public class LogitsQ4_0Layer extends AbstractLayer {

    private String lastTaskGraphID;
    private TaskGraph logitsTaskGraph;
    private ImmutableTaskGraph immutableLogitsGraph;
    private GridScheduler scheduler;
    private SchedulerType schedulerType;

    public LogitsQ4_0Layer(String taskGraphName, State state, Weights weights, Configuration config, String lastTaskGraphID, SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config);
        this.lastTaskGraphID = lastTaskGraphID;
        state.tempLogits.init(0.0f);
        var tornadoWeights = requireWeightsType(weights, TornadoWeights.class, "LogitsQ4_0Layer", "TornadoTensor");
        this.logitsTaskGraph = setupLogitsTaskGraph(tornadoWeights, config);
        this.schedulerType = schedulerType;
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        WorkerGrid logitsRMS;
        if (weights instanceof Qwen2TornadoWeights) {
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

    private TaskGraph setupLogitsTaskGraph(TornadoWeights weights, Configuration config) {
        TaskGraph logits = new TaskGraph("logits");
        logits.consumeFromDevice(lastTaskGraphID, state.wrapX).transferToDevice(DataTransferMode.EVERY_EXECUTION, state.tempLogits)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, context, state.wrapLogits, weights.wclsByteArray.getQuants(), weights.wclsByteArray.getScales(),
                        weights.rms_final_weight_as_floatArray)
                .task("reductionsOneBlockLogits", TransformerComputeKernels::reductionOneBlockWithLayer, context, state.tempLogits, state.wrapX, config.dim(), config.rmsNormEps(), state.localSize);
                if (schedulerType == SchedulerType.NON_NVIDIA) {
                    logits.task("reductionFinalNormalizationLogits", TransformerComputeKernelsLayered::reductionFinalNormalization, context, state.tempLogits, config.dim(), config.rmsNormEps());
                }
                logits.task("mapContextLogits", TransformerComputeKernels::reductionOneBlock2WithLogits, context, state.wrapX, weights.rms_final_weight_as_floatArray.asFloatArray(), state.tempLogits)
                .task("projection", TransformerComputeKernelsLayered::matrixVectorGeneric,  //
                        context, state.wrapX, state.wrapLogits, weights.wclsByteArray.getQuants(), weights.wclsByteArray.getScales(), //
                        config.dim(), config.vocabularySize(), LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS) //
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
