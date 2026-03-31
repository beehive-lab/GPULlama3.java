package org.beehive.gpullama3.tornadovm.layers.type.fp16;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.mistral.MistralConfiguration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernels;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernelsLayered;
import org.beehive.gpullama3.tornadovm.layerplanner.WorkerGridFactory;
import org.beehive.gpullama3.tornadovm.layerplanner.strategy.SchedulerType;
import org.beehive.gpullama3.tornadovm.layers.AbstractFFNLayers;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

public class MistralFP16FFNLayers extends AbstractFFNLayers<LlamaTornadoWeights, MistralConfiguration> {

    public MistralFP16FFNLayers(String taskGraph, State state, LlamaTornadoWeights weights, MistralConfiguration config, SchedulerType schedulerType) {
        super(taskGraph, state, weights, config, schedulerType);
        setupFFNLayers();
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        WorkerGrid rmsNormWorker = WorkerGridFactory.createRmsNormWorker(config.dim(), 256);

        int configDimRowMajorGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configDimRowMajorGlobalWorker = WorkerGridFactory.genericWorker(configDimRowMajorGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int configHiddenDimRowMajor = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configHiddenDimRowMajorWorker = WorkerGridFactory.genericWorker(configHiddenDimRowMajor, LOCAL_WORK_GROUP_SIZE_ALLOC);

        WorkerGrid parallelAttentionWorker = WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), config.headSize());

        int fusedQKVRows = config.dim() + 2 * config.kvDim();
        int fusedQKVGlobal = fusedQKVRows * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedQKVWorker = WorkerGridFactory.genericWorker(fusedQKVGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);
        WorkerGrid ropeWithCacheWorker = WorkerGridFactory.genericWorker(config.dim() / 2, 512);

        // Map workers to tasks
        for (int i = 0; i < config.numberOfLayers(); i++) {
            // === Attention Block ===
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".attn_rms_reduce", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".attn_rms_apply_fp16", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".qkv_projection", fusedQKVWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rope_and_kv_cache", ropeWithCacheWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".attention", parallelAttentionWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".attn_output_proj", configDimRowMajorGlobalWorker);
            // === FFN Block ===
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".ffn_rms_reduce", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rms_ffn_gate_up", configHiddenDimRowMajorWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".ffn_down_proj", configDimRowMajorGlobalWorker);
        }
        return tornadoForwardScheduler;
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
                weights.wqLayered[layerIndex].asHalfFloatArray(),
                weights.wkLayered[layerIndex].asHalfFloatArray(),
                weights.wvLayered[layerIndex].asHalfFloatArray(),
                weights.woLayered[layerIndex].asHalfFloatArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].asHalfFloatArray(),
                weights.w2Layered[layerIndex].asHalfFloatArray(),
                weights.w3Layered[layerIndex].asHalfFloatArray());
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

        unifiedLayer.task("attn_rms_apply_fp16",
                TransformerComputeKernels::mapContextWithQuantize,
                context, state.wrapXbFP16, state.wrapX,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(), state.temp);

        unifiedLayer.task("qkv_projection",
                TransformerComputeKernelsLayered::fusedQKVMatmulX,
                context,
                state.wrapXbFP16,
                state.wrapQ,
                state.wrapK,
                state.wrapV,
                weights.wqLayered[layerIndex].asHalfFloatArray(),
                weights.wkLayered[layerIndex].asHalfFloatArray(),
                weights.wvLayered[layerIndex].asHalfFloatArray(),
                config.dim(),
                config.kvDim(),
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        unifiedLayer.task("rope_and_kv_cache",
                TransformerComputeKernelsLayered::ropeRotationWithCacheCopy,
                context,
                state.positionHolder,
                state.wrapQ,
                state.wrapK,
                state.wrapV,
                state.wrapKeyCache,
                state.wrapValueCache,
                config.kvDim(),
                config.headSize(),
                layerIndex,
                config.contextLength());

        configureAttention(unifiedLayer, layerIndex);

        unifiedLayer.task("attn_output_proj",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                context, state.wrapXb, state.wrapX,
                weights.woLayered[layerIndex].asHalfFloatArray(),
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
                TransformerComputeKernelsLayered::fusedRmsNormFFNGateUp,
                context,
                state.wrapX,
                state.wrapHb,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                state.tempFFN,
                weights.w1Layered[layerIndex].asHalfFloatArray(),
                weights.w3Layered[layerIndex].asHalfFloatArray(),
                config.dim(),
                config.hiddenDim(),
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        unifiedLayer.task("ffn_down_proj",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                context, state.wrapHb, state.wrapX,
                weights.w2Layered[layerIndex].asHalfFloatArray(),
                config.hiddenDim(), config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC);

        unifiedLayer.persistOnDevice(state.wrapX);

        return unifiedLayer;
    }

    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        if (layerIndex == 0) {
            unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                    state.positionHolder,
                    state.temp, state.tempFFN);
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.wrapXb, state.wrapXb2,
                    state.wrapQ, state.wrapK, state.wrapV,
                    state.wrapKeyCache, state.wrapValueCache,
                    state.wrapAtt, state.wrapHb, state.wrapXbFP16);
        } else {
            unifiedLayer.consumeFromDevice(
                    context,
                    state.wrapXb, state.wrapXb2,
                    state.wrapQ, state.wrapK, state.wrapV,
                    state.wrapKeyCache, state.wrapValueCache,
                    state.wrapAtt, state.wrapHb,
                    state.positionHolder, state.wrapXbFP16);
        }
        return unifiedLayer;
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
