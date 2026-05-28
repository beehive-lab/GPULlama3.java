package org.beehive.gpullama3.tornadovm.layers.type.q8_0;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.devstral.DevstralConfiguration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernelsLayered;
import org.beehive.gpullama3.tornadovm.scheduling.WorkerGridFactory;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;
import org.beehive.gpullama3.tornadovm.layers.AbstractTransformerLayerTaskGraphs;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Q8_0 FFN layers for Devstral 2 models.
 * Uses precomputed RoPE frequencies (YaRN scaling) instead of on-the-fly computation.
 */
public class DevstralQ8_0FFNLayers extends AbstractTransformerLayerTaskGraphs<LlamaTornadoWeights, DevstralConfiguration> {

    public DevstralQ8_0FFNLayers(String taskGraphName, State state, LlamaTornadoWeights weights, DevstralConfiguration config, SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config, schedulerType);
        setupFFNLayers();
    }

    // @formatter:off
    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        var layerTaskGraphName = "layer_" + layerIndex;
        TaskGraph unifiedLayer = new TaskGraph(layerTaskGraphName);

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
                TransformerComputeKernelsLayered::fusedQKVMatmulQ8NonSquare,
                context,
                state.wrapXb,
                state.wrapQ, state.wrapK, state.wrapV,
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                config.dim(), config.qDim(), config.kvDim(), LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Use precomputed RoPE frequencies (YaRN-scaled)
        unifiedLayer.task("rope_and_kv_cache",
                TransformerComputeKernelsLayered::ropeRotationWithCacheCopyPrecomputed,
                context,
                state.positionHolder,
                state.wrapQ, state.wrapK, state.wrapV,
                state.wrapKeyCache, state.wrapValueCache,
                weights.freq_cis_realFlat.asFloatArray(),
                weights.freq_cis_imagFlat.asFloatArray(),
                config.kvDim(), config.headSize(), layerIndex, config.contextLength());

        configureAttention(unifiedLayer, layerIndex);

        // O projection: n=qDim (input), d=dim (output)
        unifiedLayer.task("attn_output_proj",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                context, state.wrapXb, state.wrapX,
                weights.woLayered[layerIndex].asByteArray(),
                config.qDim(), config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC);

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
                    state.wrapAtt, state.wrapHb,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray());
        } else {
            unifiedLayer.consumeFromDevice(
                    context,
                    state.wrapXb, state.wrapXb2,
                    state.wrapQ, state.wrapK, state.wrapV,
                    state.wrapKeyCache, state.wrapValueCache,
                    state.wrapAtt, state.wrapHb,
                    state.positionHolder,
                    weights.freq_cis_realFlat.asFloatArray(),
                    weights.freq_cis_imagFlat.asFloatArray());
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

        int fusedQkvGlobal = (config.qDim() + 2 * config.kvDim()) * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedQkvWorker = WorkerGridFactory.genericWorker(fusedQkvGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        WorkerGrid ropeWithCacheWorker = WorkerGridFactory.genericWorker(config.qDim() / 2, 512);
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
