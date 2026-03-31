package org.beehive.gpullama3.tornadovm.layers.type.q8_0;

import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.mistral.MistralConfiguration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernelsLayered;
import org.beehive.gpullama3.tornadovm.layerplanner.WorkerGridFactory;
import org.beehive.gpullama3.tornadovm.layerplanner.strategy.SchedulerType;
import org.beehive.gpullama3.tornadovm.layers.AbstractFFNLayers;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

public class MistralQ8_0FFNLayers extends AbstractFFNLayers<LlamaTornadoWeights, MistralConfiguration> {

    public MistralQ8_0FFNLayers(String taskGraphName, LlamaState state, LlamaTornadoWeights weights, MistralConfiguration config, SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config, schedulerType);
        setupFFNLayers();
    }

    // @formatter:off
    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        var layerTaskGraphName = "layer_" + layerIndex;
        TaskGraph unifiedLayer = new TaskGraph(layerTaskGraphName);

        // === Data Setup ===
        unifiedLayer.consumeFromDevice(state.wrapX);
        unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                weights.woLayered[layerIndex].asByteArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].asByteArray(),
                weights.w2Layered[layerIndex].asByteArray(),
                weights.w3Layered[layerIndex].asByteArray());
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);

        // === Attention Block ===
        unifiedLayer.task("attn_rms_reduce",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                context, state.temp, state.wrapX,
                config.dim(), config.rmsNormEps(), state.localSize);

        if (shouldUseFinalNormalization()) {
            unifiedLayer.task("attn_rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context, state.temp, config.dim(), config.rmsNormEps());
        }

        unifiedLayer.task("attn_rms_apply",
                TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                context, state.wrapXb, state.wrapX,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(), state.temp);

        unifiedLayer.task("qkv_projection",
                TransformerComputeKernelsLayered::fusedQKVMatmulQ8,
                context,
                state.wrapXb,
                state.wrapQ, state.wrapK, state.wrapV,
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                config.dim(), config.kvDim(), LOCAL_WORK_GROUP_SIZE_ALLOC);

        unifiedLayer.task("rope_and_kv_cache",
                TransformerComputeKernelsLayered::ropeRotationWithCacheCopy,
                context,
                state.positionHolder,
                state.wrapQ, state.wrapK, state.wrapV,
                state.wrapKeyCache, state.wrapValueCache,
                config.kvDim(), config.headSize(), layerIndex, config.contextLength());

        configureAttention(unifiedLayer, layerIndex);

        unifiedLayer.task("attn_output_proj",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                context, state.wrapXb, state.wrapX,
                weights.woLayered[layerIndex].asByteArray(),
                config.dim(), config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC);

        // === FFN Block ===
        unifiedLayer.task("ffn_rms_reduce",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                context, state.tempFFN, state.wrapX,
                config.dim(), config.rmsNormEps(), state.localSize);

        if (shouldUseFinalNormalization()) {
            unifiedLayer.task("ffn_rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context, state.tempFFN, config.dim(), config.rmsNormEps());
        }

        unifiedLayer.task("rms_ffn_gate_up",
                TransformerComputeKernelsLayered::fullyFusedRmsNormFFNGateUpQ8,
                context,
                state.wrapX, state.wrapHb,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].asByteArray(),
                weights.w3Layered[layerIndex].asByteArray(),
                config.dim(), config.hiddenDim(), LOCAL_WORK_GROUP_SIZE_ALLOC);

        unifiedLayer.task("ffn_down_proj",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                context, state.wrapHb, state.wrapX,
                weights.w2Layered[layerIndex].asByteArray(),
                config.hiddenDim(), config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC);

        unifiedLayer.persistOnDevice(state.wrapX);

        return unifiedLayer;
    }

    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        if (layerIndex == 0) {
            unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                    state.positionHolder, state.temp, state.tempFFN);
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.wrapXb, state.wrapXb2,
                    state.wrapQ, state.wrapK, state.wrapV,
                    state.wrapKeyCache, state.wrapValueCache,
                    state.wrapAtt, state.wrapHb);
        } else {
            unifiedLayer.consumeFromDevice(
                    context,
                    state.wrapXb, state.wrapXb2,
                    state.wrapQ, state.wrapK, state.wrapV,
                    state.wrapKeyCache, state.wrapValueCache,
                    state.wrapAtt, state.wrapHb,
                    state.positionHolder);
        }
        return unifiedLayer;
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        WorkerGrid rmsNormWorker = WorkerGridFactory.createRmsNormWorker(config.dim(), 256);

        int configDimRowMajorGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configDimRowMajorGlobalWorker = WorkerGridFactory.genericWorker(configDimRowMajorGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int configHiddenDimRowMajor = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configHiddenDimRowMajorWorker = WorkerGridFactory.genericWorker(configHiddenDimRowMajor, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int fusedQkvGlobal = (config.dim() + 2 * config.kvDim()) * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedQkvWorker = WorkerGridFactory.genericWorker(fusedQkvGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        WorkerGrid ropeWithCacheWorker = WorkerGridFactory.genericWorker(config.dim() / 2, 512);
        WorkerGrid parallelAttentionWorker = WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), config.headSize());

        for (int i = 0; i < config.numberOfLayers(); i++) {
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".attn_rms_reduce", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".attn_rms_apply", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".qkv_projection", fusedQkvWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rope_and_kv_cache", ropeWithCacheWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".attention", parallelAttentionWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".attn_output_proj", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".ffn_rms_reduce", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rms_ffn_gate_up", configHiddenDimRowMajorWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".ffn_down_proj", configDimRowMajorGlobalWorker);
        }

        return tornadoForwardScheduler;
    }

    private TaskGraph configureAttention(TaskGraph unifiedLayer, int layerIndex) {
        if (schedulerType == SchedulerType.NVIDIA) {
            return unifiedLayer.task("attention",
                TransformerComputeKernelsLayered::processHeadsFlashAttention,
                context,
                state.wrapQ, state.wrapKeyCache, state.wrapValueCache, state.wrapXb,
                config.numberOfHeads(), config.headSize(),
                config.kvDim(), config.kvMul(),
                state.positionHolder, layerIndex, config.contextLength());
        } else {
            return unifiedLayer.task("attention",
                TransformerComputeKernelsLayered::processHeadsParallel,
                state.wrapQ, state.wrapKeyCache, state.wrapValueCache, state.wrapXb,
                config.numberOfHeads(), config.headSize(),
                config.kvDim(), config.kvMul(), config.contextLength(),
                state.positionHolder, state.wrapAtt, layerIndex, config.contextLength());
        }
    }
    // @formatter:on
}
