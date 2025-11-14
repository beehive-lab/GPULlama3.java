package org.beehive.gpullama3.tornadovm.layers.type.q4_0;

import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernelsLayered;
import org.beehive.gpullama3.tornadovm.layerplanner.WorkerGridFactory;
import org.beehive.gpullama3.tornadovm.layerplanner.strategy.SchedulerType;
import org.beehive.gpullama3.tornadovm.layers.AbstractFFNLayers;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.List;
import java.util.stream.IntStream;

public class LlamaQ4_0FFNLayers extends AbstractFFNLayers {

    GridScheduler scheduler;
    List<ImmutableTaskGraph> ffnLayerTaskGraphs;

    public LlamaQ4_0FFNLayers(String taskGraphName, LlamaState state, LlamaTornadoWeights weights, Configuration config, SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config, schedulerType);
        ffnLayerTaskGraphs = setupFFNLayered();
    }

    @Override
    public GridScheduler getGridScheduler() {
        return scheduler;
    }

    @Override
    public TaskGraph getTaskGraph() {
        return null;
    }

    @Override
    public ImmutableTaskGraph getImmutableTaskGraph() {
        return null;
    }

    List<ImmutableTaskGraph> setupFFNLayered() {
        state.temp.init(0.0f);
        state.tempFFN.init(0.0f);
        var numLayers = config.numberOfLayers();

        return IntStream.range(0, numLayers).mapToObj(i -> {
            var ffnLayer = setupSingleFFNLayer((LlamaTornadoWeights) weights, config, i);
            if (i == numLayers - 1) {
                setupLastID(ffnLayer.getTaskGraphName());
            }
            return ffnLayer.snapshot();
        }).toList();
    }

    TaskGraph setupSingleFFNLayer(LlamaTornadoWeights weights, Configuration config, int layerIndex) {
        var layerTaskGraphName = "layer_" + layerIndex;
        TaskGraph unifiedLayer = new TaskGraph(layerTaskGraphName);
        unifiedLayer.consumeFromDevice(state.wrapX);
        unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                //Copy-in weights per layer for batched-layered layout
                weights.rms_att_weightLayered[layerIndex].asFloatArray(), weights.wqLayered[layerIndex].getQuants(), weights.wqLayered[layerIndex].getScales(), weights.wkLayered[layerIndex].getQuants(),
                weights.wkLayered[layerIndex].getScales(), weights.wvLayered[layerIndex].getQuants(), weights.wvLayered[layerIndex].getScales(), weights.woLayered[layerIndex].getQuants(),
                weights.woLayered[layerIndex].getScales(), weights.rms_ffn_weightLayered[layerIndex].asFloatArray(), weights.w1Layered[layerIndex].getQuants(), weights.w1Layered[layerIndex].getScales(),
                weights.w2Layered[layerIndex].getQuants(), weights.w2Layered[layerIndex].getScales(), weights.w3Layered[layerIndex].getQuants(), weights.w3Layered[layerIndex].getScales());
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);
        unifiedLayer.task("reductionsOneBlock", TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, state.temp, state.wrapX, config.dim(), config.rmsNormEps(), state.localSize);
                if (shouldUseFinalNormalization()) {
                    unifiedLayer.task("reductionFinalNormalization", TransformerComputeKernelsLayered::reductionFinalNormalization, context, state.temp,
                            config.dim(), config.rmsNormEps());
                }
        unifiedLayer.task("mapContext", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, state.wrapXb, state.wrapX, weights.rms_att_weightLayered[layerIndex].asFloatArray(), state.temp)
                .task("qmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context, state.wrapXb, state.wrapQ, weights.wqLayered[layerIndex].getQuants(),
                        weights.wqLayered[layerIndex].getScales(), config.dim(), config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("kmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context, state.wrapXb, state.wrapK, weights.wkLayered[layerIndex].getQuants(),
                        weights.wkLayered[layerIndex].getScales(), config.dim(), config.kvDim(), LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("vmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context, state.wrapXb, state.wrapV, weights.wvLayered[layerIndex].getQuants(),
                        weights.wvLayered[layerIndex].getScales(), config.dim(), config.kvDim(), LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("rope", TransformerComputeKernelsLayered::ropeRotation, context, state.positionHolder, state.wrapQ, state.wrapK, config.kvDim(), config.headSize())
                .task("copyToCaches", TransformerComputeKernelsLayered::copyToCache, state.wrapKeyCache, state.wrapK, state.wrapValueCache, state.wrapV, state.positionHolder, config.kvDim(),
                        layerIndex, config.contextLength());
                configureAttention(unifiedLayer, layerIndex);
        unifiedLayer.task("matmul1", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context, state.wrapXb, state.wrapX, weights.woLayered[layerIndex].getQuants(),
                        weights.woLayered[layerIndex].getScales(), config.dim(), config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("reductionsOneBlockFFN", TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, state.tempFFN, state.wrapX, config.dim(), config.rmsNormEps(), state.localSize);
                if (shouldUseFinalNormalization()) {
                    unifiedLayer.task("reductionFinalNormalizationFFN", TransformerComputeKernelsLayered::reductionFinalNormalization, context, state.tempFFN,
                            config.dim(), config.rmsNormEps());
                }
                unifiedLayer.task("mapContextFFN", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, state.wrapXb, state.wrapX, weights.rms_ffn_weightLayered[layerIndex].asFloatArray(), state.tempFFN)
                .task("fused_ffn_w1_w3", TransformerComputeKernelsLayered::fusedFeedForwardWithSiLUAndGLUActivation, context, state.wrapXb, state.wrapHb, weights.w1Layered[layerIndex].getQuants(),
                        weights.w1Layered[layerIndex].getScales(), weights.w3Layered[layerIndex].getQuants(), weights.w3Layered[layerIndex].getScales(), config.dim(), config.hiddenDim(),
                        LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("projectionTwo", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context, state.wrapHb, state.wrapX, weights.w2Layered[layerIndex].getQuants(),
                        weights.w2Layered[layerIndex].getScales(), config.hiddenDim(), config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC).persistOnDevice(state.wrapX);
        return unifiedLayer;
    }

    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        // First layer: Transfer initial data to device (one-time transfer)
        if (layerIndex == 0) {
            // Transfer all attention-related data: query, key, value matrices and their caches
            unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION, state.positionHolder, state.temp, state.tempFFN); //
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, //
                    context, state.wrapXb, state.wrapXb2, //
                    state.wrapQ, state.wrapK, state.wrapV, //
                    state.wrapKeyCache, state.wrapValueCache, //
                    state.wrapAtt, state.wrapHb); //
        } else {
            // Subsequent layers: Consume data already on device from previous layer
            unifiedLayer.consumeFromDevice(context, state.wrapXb, state.wrapXb2, //
                    state.wrapQ, state.wrapK, state.wrapV, //
                    state.wrapKeyCache, state.wrapValueCache, //
                    state.wrapAtt, state.wrapHb, //
                    state.positionHolder //
            );
        }
        return unifiedLayer;
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(config.dim() / 2, 128);
        WorkerGrid rmsNormWorker = WorkerGridFactory.createRmsNormWorker(config.dim(), 256);

        int configDimRowMajorGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configDimRowMajorGlobalWorker = WorkerGridFactory.genericWorker(configDimRowMajorGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int configKvDimRowMajorGlobal = config.kvDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configKvDimRowMajorGlobalWorker = WorkerGridFactory.genericWorker(configKvDimRowMajorGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int configHiddenDimRowMajor = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configHiddenDimRowMajorWorker = WorkerGridFactory.genericWorker(configHiddenDimRowMajor, LOCAL_WORK_GROUP_SIZE_ALLOC);
        WorkerGrid parallelAttentionWorker = WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), config.headSize());
        WorkerGrid copyToCachesWorker = WorkerGridFactory.genericWorker(config.dim(), 128);

        for (int i = 0; i < config.numberOfLayers(); i++) {
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".qmatmul", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".kmatmul", configKvDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".vmatmul", configKvDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rope", ropeWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".matmul1", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".projectionTwo", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".fused_ffn_w1_w3", configHiddenDimRowMajorWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlock", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".mapContext", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlockFFN", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".mapContextFFN", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".parallel-attention", parallelAttentionWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".copyToCaches", copyToCachesWorker);
        }
        return tornadoForwardScheduler;
    }

    public List<ImmutableTaskGraph> getFfnLayerTaskGraphs() {
        return ffnLayerTaskGraphs;
    }

    private TaskGraph configureAttention(TaskGraph unifiedLayer, int layerIndex) {
        if (schedulerType == SchedulerType.NVIDIA) {
            return unifiedLayer.task("parallel-attention", TransformerComputeKernelsLayered::processHeadsFlashAttention,
                    context, state.wrapQ, state.wrapKeyCache, state.wrapValueCache, state.wrapXb,
                    config.numberOfHeads(), config.headSize(), config.kvDim(), config.kvMul(),
                    state.positionHolder, layerIndex, config.contextLength());
        } else {
            return unifiedLayer.task("parallel-attention", TransformerComputeKernelsLayered::processHeadsParallel,
                    state.wrapQ, state.wrapKeyCache, state.wrapValueCache, state.wrapXb,
                    config.numberOfHeads(), config.headSize(), config.kvDim(), config.kvMul(), config.contextLength(),
                    state.positionHolder, state.wrapAtt, layerIndex, config.contextLength());
        }
    }
}
