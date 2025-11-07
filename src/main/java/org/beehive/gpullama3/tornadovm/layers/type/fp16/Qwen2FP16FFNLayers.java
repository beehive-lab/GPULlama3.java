package org.beehive.gpullama3.tornadovm.layers.type.fp16;

import org.beehive.gpullama3.inference.state.Qwen2State;
import org.beehive.gpullama3.inference.weights.tornado.fp16.Qwen2TornadoWeights;
import org.beehive.gpullama3.model.qwen2.Qwen2Configuration;
import org.beehive.gpullama3.tornadovm.kernels.Qwen2Kernels;
import org.beehive.gpullama3.tornadovm.kernels.Qwen3Kernels;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernelsLayered;
import org.beehive.gpullama3.tornadovm.layerplanner.WorkerGridFactory;
import org.beehive.gpullama3.tornadovm.layers.AbstractFFNLayers;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.WorkerGrid2D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.ArrayList;
import java.util.List;

/**
 * Qwen2FP16FFNLayers: FP16 FFN layers for Qwen2 with Group Query Attention (GQA) support.
 *
 * Key Differences from Qwen3: - No tempQcur/tempKcur fields in Qwen2State - Includes bias terms for Q, K, V projections - Standard GQA (no parallel offset RMSNorm) - Uses
 * Qwen2Kernels::processHeadsFlashAttention for attention computation - Uses Qwen3Kernels::ropeRotation for position embeddings - Simpler matrix dimensions (uses config.dim() and config.kvDim()
 * directly)
 *
 * Works directly with Qwen2State to access and mutate Qwen2-specific state fields.
 */
public class Qwen2FP16FFNLayers extends AbstractFFNLayers {

    // Typed references to Qwen2-specific state and config
    private final Qwen2State qwen2State;
    private final Qwen2Configuration qwen2Config;
    TaskGraph ffnLayerTaskGraph;
    GridScheduler scheduler;
    List<ImmutableTaskGraph> ffnLayerTaskGraphs;

    public Qwen2FP16FFNLayers(String taskGraphName, Qwen2State state, Qwen2TornadoWeights weights, Qwen2Configuration config, org.beehive.gpullama3.tornadovm.layerplanner.strategy.SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config, schedulerType);
        this.qwen2State = state;
        this.qwen2Config = config;
        ffnLayerTaskGraphs = setupFFNLayered();
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        int h = config.numberOfHeads();
        int ic = config.headSize() / 2;
        WorkerGrid ropeWorker = new WorkerGrid2D(h, ic);
        ropeWorker.setGlobalWork(h, ic, 1);
        ropeWorker.setLocalWork(1, 1, 1);

        int configDimRowMajorGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configDimRowMajorGlobalWorker = new WorkerGrid1D(configDimRowMajorGlobal);
        configDimRowMajorGlobalWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        int configKvDimRowMajorGlobal = config.kvDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configKvDimRowMajorGlobalWorker = new WorkerGrid1D(configKvDimRowMajorGlobal);
        configKvDimRowMajorGlobalWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        WorkerGrid qBiasWorker = new WorkerGrid1D(config.dim());
        qBiasWorker.setGlobalWork(config.dim(), 1, 1);
        qBiasWorker.setLocalWork(config.dim() / 8, 1, 1);
        WorkerGrid kvBiasWorker = new WorkerGrid1D(config.kvDim());
        kvBiasWorker.setGlobalWork(config.kvDim(), 1, 1);
        kvBiasWorker.setLocalWork(32, 1, 1);

        int configHiddenDimRowMajor = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configHiddenDimRowMajorWorker = new WorkerGrid1D(configHiddenDimRowMajor);
        configHiddenDimRowMajorWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        WorkerGrid rmsNormWorker = WorkerGridFactory.createRmsNormWorker(config.dim(), 32);

        // Parallel attention worker configuration
        // Calculate optimal local work size based on head dimension
        int optimalLocalSize = Math.min(config.headSize(), 64); // Start with 64 threads per head
        if (config.headSize() % optimalLocalSize != 0) {
            // Find largest divisor of headSize <= 64
            for (int size = 64; size >= 1; size--) {
                if (config.headSize() % size == 0) {
                    optimalLocalSize = size;
                    break;
                }
            }
        }

        WorkerGrid parallelAttentionWorker = new WorkerGrid1D(config.numberOfHeads());
        parallelAttentionWorker.setGlobalWork(config.numberOfHeads() * optimalLocalSize, 1, 1);
        parallelAttentionWorker.setLocalWork(optimalLocalSize, 1, 1);

        WorkerGrid copyToCachesWorker = new WorkerGrid1D(config.kvDim());
        copyToCachesWorker.setGlobalWork(config.kvDim(), 1, 1);
        copyToCachesWorker.setLocalWork(32, 1, 1); // Set local work size to 32 (for copying to caches)

        // Map workers to tasks
        for (int i = 0; i < config.numberOfLayers(); i++) {
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".qmatmul", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".kmatmul", configKvDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".vmatmul", configKvDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".qbias", qBiasWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".kbias", kvBiasWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".vbias", kvBiasWorker);
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

    @Override
    public GridScheduler getGridScheduler() {
        return scheduler;
    }

    @Override
    public TaskGraph getTaskGraph() {
        return ffnLayerTaskGraph;
    }

    @Override
    public ImmutableTaskGraph getImmutableTaskGraph() {
        return null;
    }

    public List<ImmutableTaskGraph> getFfnLayerTaskGraphs() {
        return ffnLayerTaskGraphs;
    }

    /**
     * Setup all FFN layers for all transformer layers
     */
    List<ImmutableTaskGraph> setupFFNLayered() {
        List<ImmutableTaskGraph> ffnGraphs = new ArrayList<>();

        qwen2State.temp.init(0.0f);
        qwen2State.tempFFN.init(0.0f);

        for (int layerIndex = 0; layerIndex < qwen2Config.numberOfLayers(); layerIndex++) {
            TaskGraph ffnLayer = setupSingleQwen2FFNLayer((Qwen2TornadoWeights) weights, layerIndex);
            if (layerIndex == qwen2Config.numberOfLayers() - 1) {
                setupLastID(ffnLayer.getTaskGraphName());
            }
            ffnGraphs.add(ffnLayer.snapshot());
        }

        return ffnGraphs;
    }

    /**
     * Setup a single transformer layer for Qwen2 with GQA
     */
    TaskGraph setupSingleQwen2FFNLayer(Qwen2TornadoWeights weights, int layerIndex) {
        var taskGraphName = "layer_" + layerIndex;
        TaskGraph unifiedLayer = new TaskGraph(taskGraphName);
        unifiedLayer.consumeFromDevice(state.wrapX);
        unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, //
                weights.rms_att_weightLayered[layerIndex], //
                weights.wqLayered[layerIndex], //
                weights.wkLayered[layerIndex], //
                weights.wvLayered[layerIndex], //
                weights.woLayered[layerIndex], //
                weights.q_biasLayered[layerIndex], //
                weights.k_biasLayered[layerIndex], //
                weights.v_biasLayered[layerIndex], //
                weights.rms_ffn_weightLayered[layerIndex], //
                weights.w1Layered[layerIndex], //
                weights.w2Layered[layerIndex], //
                weights.w3Layered[layerIndex]); //
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex); //

        unifiedLayer.task("reductionsOneBlock", TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, qwen2State.temp, qwen2State.wrapX, config.dim(), config.rmsNormEps(),
                        qwen2State.localSize)
                .task("mapContext", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, qwen2State.wrapXb, qwen2State.wrapX, weights.rms_att_weightLayered[layerIndex],
                        qwen2State.temp)
                .task("qmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context, qwen2State.wrapXb, qwen2State.wrapQ, weights.wqLayered[layerIndex], config.dim(), config.dim(),
                        LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("kmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context, qwen2State.wrapXb, qwen2State.wrapK, weights.wkLayered[layerIndex], config.dim(), config.kvDim(),
                        LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("vmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context, qwen2State.wrapXb, qwen2State.wrapV, weights.wvLayered[layerIndex], config.dim(), config.kvDim(),
                        LOCAL_WORK_GROUP_SIZE_ALLOC).task("qbias", TransformerComputeKernelsLayered::addInPlace, qwen2State.wrapQ, weights.q_biasLayered[layerIndex], config.dim())
                .task("kbias", TransformerComputeKernelsLayered::addInPlace, qwen2State.wrapK, weights.k_biasLayered[layerIndex], config.kvDim())
                .task("vbias", TransformerComputeKernelsLayered::addInPlace, qwen2State.wrapV, weights.v_biasLayered[layerIndex], config.kvDim())
                .task("rope", Qwen3Kernels::ropeRotation, context, qwen2State.positionHolder, qwen2State.wrapQ, qwen2State.wrapK, config.numberOfKeyValueHeads(), config.headSize())
                .task("copyToCaches", TransformerComputeKernelsLayered::copyToCache, qwen2State.wrapKeyCache, qwen2State.wrapK, qwen2State.wrapValueCache, qwen2State.wrapV, qwen2State.positionHolder,
                        config.kvDim(), layerIndex, config.contextLength())
                .task("parallel-attention", Qwen2Kernels::processHeadsFlashAttention, context, qwen2State.wrapQ, qwen2State.wrapKeyCache, qwen2State.wrapValueCache, qwen2State.wrapXb,
                        config.numberOfHeads(), config.headSize(), config.kvDim(), config.kvMul(), qwen2State.positionHolder, layerIndex, config.contextLength())
                .task("matmul1", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context, qwen2State.wrapXb, qwen2State.wrapX, weights.woLayered[layerIndex], config.dim(),
                        config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("reductionsOneBlockFFN", TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, qwen2State.tempFFN, qwen2State.wrapX, config.dim(), config.rmsNormEps(),
                        qwen2State.localSize)
                .task("mapContextFFN", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, qwen2State.wrapXb, qwen2State.wrapX, weights.rms_ffn_weightLayered[layerIndex],
                        qwen2State.tempFFN)
                .task("fused_ffn_w1_w3", TransformerComputeKernelsLayered::fusedFeedForwardWithSiLUAndGLUActivation, context, qwen2State.wrapXb, qwen2State.wrapHb, weights.w1Layered[layerIndex],
                        weights.w3Layered[layerIndex], config.dim(), config.hiddenDim(), LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("projectionTwo", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context, qwen2State.wrapHb, qwen2State.wrapX, weights.w2Layered[layerIndex],
                        config.hiddenDim(), config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC).persistOnDevice(state.wrapX);

        return unifiedLayer;
    }

    /**
     * Configure data transfers for first and subsequent layers
     */
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        // First layer: Transfer initial data to device (one-time transfer)
        if (layerIndex == 0) {
            // Transfer all attention-related data: query, key, value matrices and their caches
            unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION, qwen2State.positionHolder, qwen2State.temp, qwen2State.tempFFN); //
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, //
                    context, qwen2State.wrapXb, qwen2State.wrapXb2, //
                    qwen2State.wrapQ, qwen2State.wrapK, qwen2State.wrapV, //
                    qwen2State.wrapKeyCache, qwen2State.wrapValueCache, //
                    qwen2State.wrapAtt, qwen2State.wrapHb); //
        } else {
            // Subsequent layers: Consume data already on device from previous layer
            unifiedLayer.consumeFromDevice( //
                    context, qwen2State.wrapXb, qwen2State.wrapXb2, //
                    qwen2State.wrapQ, qwen2State.wrapK, qwen2State.wrapV, //
                    qwen2State.wrapKeyCache, qwen2State.wrapValueCache, //
                    qwen2State.wrapAtt, qwen2State.wrapHb, //
                    qwen2State.positionHolder //
            );
        }
        return unifiedLayer;
    }

}
