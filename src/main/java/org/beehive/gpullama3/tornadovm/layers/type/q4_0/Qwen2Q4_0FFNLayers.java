package org.beehive.gpullama3.tornadovm.layers.type.q4_0;

import org.beehive.gpullama3.inference.state.Qwen2State;
import org.beehive.gpullama3.inference.weights.tornado.Qwen2TornadoWeights;
import org.beehive.gpullama3.model.qwen2.Qwen2Configuration;
import org.beehive.gpullama3.tornadovm.kernels.Qwen2Kernels;
import org.beehive.gpullama3.tornadovm.kernels.Qwen3Kernels;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernelsLayered;
import org.beehive.gpullama3.tornadovm.layerplanner.WorkerGridFactory;
import org.beehive.gpullama3.tornadovm.layerplanner.strategy.SchedulerType;
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
 * Qwen2Q4_0FFNLayers: Q4_0-quantized FFN layers for Qwen2 with Group Query Attention (GQA) support.
 *
 * Key Differences from Qwen2FP16FFNLayers:
 * - Uses Q4_0-quantized weights (getQuants() and getScales())
 * - Same attention and RoPE kernels as FP16 version
 * - 4-bit integer computations with dequantization
 * - 4x memory compression vs FP16, 2x vs Q8_0
 * - Includes bias terms for Q, K, V projections
 *
 * Works directly with Qwen2State to access and mutate Qwen2-specific state fields.
 */
public class Qwen2Q4_0FFNLayers extends AbstractFFNLayers {

    TaskGraph ffnLayerTaskGraph;
    GridScheduler scheduler;
    List<ImmutableTaskGraph> ffnLayerTaskGraphs;

    // Typed references to Qwen2-specific state and config
    private final Qwen2State qwen2State;
    private final Qwen2Configuration qwen2Config;

    public Qwen2Q4_0FFNLayers(String taskGraphName, Qwen2State state, Qwen2TornadoWeights weights, Qwen2Configuration config, SchedulerType schedulerType) {
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
            TaskGraph ffnLayer = setupSingleQwen2Q4_0FFNLayer((Qwen2TornadoWeights) weights, layerIndex);
            if (layerIndex == qwen2Config.numberOfLayers() - 1) {
                setupLastID(ffnLayer.getTaskGraphName());
            }
            ffnGraphs.add(ffnLayer.snapshot());
        }
        return ffnGraphs;
    }

    /**
     * Setup a single transformer layer for Qwen2 with Q4_0 quantization and GQA
     */
    TaskGraph setupSingleQwen2Q4_0FFNLayer(Qwen2TornadoWeights weights, int layerIndex) {
      TaskGraph  unifiedLayer = new TaskGraph("layer_" + layerIndex);
        unifiedLayer.consumeFromDevice(state.wrapX);
        unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                //Copy-in weights per layer for batched-layered layout
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].getScales(),
                weights.wqLayered[layerIndex].getQuants(),
                weights.wkLayered[layerIndex].getScales(),
                weights.wkLayered[layerIndex].getQuants(),
                weights.wvLayered[layerIndex].getScales(),
                weights.wvLayered[layerIndex].getQuants(),
                weights.woLayered[layerIndex].getScales(),
                weights.woLayered[layerIndex].getQuants(),
                weights.q_biasLayered[layerIndex].asFloatArray(),
                weights.k_biasLayered[layerIndex].asFloatArray(),
                weights.v_biasLayered[layerIndex].asFloatArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].getScales(),
                weights.w1Layered[layerIndex].getQuants(),
                weights.w2Layered[layerIndex].getScales(),
                weights.w2Layered[layerIndex].getQuants(),
                weights.w3Layered[layerIndex].getScales(),
                weights.w3Layered[layerIndex].getQuants()
        );
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);

        unifiedLayer.task("reductionsOneBlock" , TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, state.temp,
                        state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                .task("mapContext", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, state.wrapXb,
                        state.wrapX, weights.rms_att_weightLayered[layerIndex].asFloatArray(), state.temp)
                .task("qmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context,
                        state.wrapXb,  state.wrapQ, weights.wqLayered[layerIndex].getQuants(), weights.wqLayered[layerIndex].getScales(), config.dim(), config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("kmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context,
                        state.wrapXb,  state.wrapK, weights.wkLayered[layerIndex].getQuants(), weights.wkLayered[layerIndex].getScales(), config.dim(), config.kvDim(), LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("vmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context,
                        state.wrapXb,   state.wrapV, weights.wvLayered[layerIndex].getQuants(), weights.wvLayered[layerIndex].getScales(), config.dim(), config.kvDim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("qbias", TransformerComputeKernelsLayered::addInPlace, state.wrapQ, weights.q_biasLayered[layerIndex].asFloatArray(), config.dim())
                .task("kbias", TransformerComputeKernelsLayered::addInPlace, state.wrapK, weights.k_biasLayered[layerIndex].asFloatArray(), config.kvDim())
                .task("vbias", TransformerComputeKernelsLayered::addInPlace, state.wrapV, weights.v_biasLayered[layerIndex].asFloatArray(), config.kvDim())
                .task("rope", Qwen3Kernels::ropeRotation,context, state.positionHolder, state.wrapQ, state.wrapK, config.numberOfKeyValueHeads(),
                        config.headSize())
                .task("copyToCaches", TransformerComputeKernelsLayered::copyToCache,
                        state.wrapKeyCache, state.wrapK,  state.wrapValueCache, state.wrapV, state.positionHolder, config.kvDim(), layerIndex, config.contextLength())
                .task("parallel-attention", Qwen2Kernels::processHeadsFlashAttention, context,
                        state.wrapQ, state.wrapKeyCache, state.wrapValueCache, state.wrapXb,
                        config.numberOfHeads(), config.headSize(), config.kvDim(), config.kvMul(),
                        state.positionHolder, layerIndex, config.contextLength())
                .task("matmul1", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context,
                        state.wrapXb,  state.wrapX, weights.woLayered[layerIndex].getQuants(), weights.woLayered[layerIndex].getScales(), config.dim(), config.dim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("reductionsOneBlockFFN", TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, state.tempFFN,
                        state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                .task("mapContextFFN", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, state.wrapXb,
                        state.wrapX, weights.rms_ffn_weightLayered[layerIndex].asFloatArray(), state.tempFFN)
                .task("fused_ffn_w1_w3", TransformerComputeKernelsLayered::fusedFeedForwardWithSiLUAndGLUActivation, context,
                        state.wrapXb,   state.wrapHb, weights.w1Layered[layerIndex].getQuants(), weights.w1Layered[layerIndex].getScales(), weights.w3Layered[layerIndex].getQuants(), weights.w3Layered[layerIndex].getScales(), config.dim(), config.hiddenDim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("projectionTwo", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context,
                        state.wrapHb, state.wrapX, weights.w2Layered[layerIndex].getQuants(), weights.w2Layered[layerIndex].getScales(), config.hiddenDim(), config.dim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                .persistOnDevice(
                        state.wrapX
                );
        return unifiedLayer;

    }

    /**
     * Configure data transfers for first and subsequent layers
     */
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        if (layerIndex == 0) {
            // First layer: Transfer temporary buffers and QKV state every execution
            unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                    qwen2State.positionHolder, qwen2State.temp, qwen2State.tempFFN);
            // First execution: allocate workspace buffers
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    context, qwen2State.wrapXb, qwen2State.wrapXb2,
                    qwen2State.wrapQ, qwen2State.wrapK, qwen2State.wrapV,
                    qwen2State.wrapKeyCache, qwen2State.wrapValueCache,
                    qwen2State.wrapAtt, qwen2State.wrapHb);
        } else {
            // Subsequent layers: Consume data from previous layer
            unifiedLayer.consumeFromDevice(context, qwen2State.wrapXb, qwen2State.wrapXb2,
                    qwen2State.wrapQ, qwen2State.wrapK, qwen2State.wrapV,
                    qwen2State.wrapKeyCache, qwen2State.wrapValueCache,
                    qwen2State.wrapAtt, qwen2State.wrapHb, qwen2State.positionHolder);
        }
        return unifiedLayer;
    }

}
