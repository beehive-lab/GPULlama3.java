package org.beehive.gpullama3.tornadovm.layers.type.q8_0;

import org.beehive.gpullama3.inference.state.Phi3State;
import org.beehive.gpullama3.inference.weights.tornado.Phi3TornadoWeights;
import org.beehive.gpullama3.model.phi3.Phi3Configuration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernelsLayered;
import org.beehive.gpullama3.tornadovm.layerplanner.WorkerGridFactory;
import org.beehive.gpullama3.tornadovm.layerplanner.strategy.SchedulerType;
import org.beehive.gpullama3.tornadovm.layers.AbstractFFNLayers;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.ArrayList;
import java.util.List;

/**
 * Phi3Q8_0FFNLayers: Q8_0-quantized FFN layers for Phi3 with Group Query Attention (GQA) support.
 *
 * Key Differences from Phi3FP16FFNLayers:
 * - Uses Q8_0-quantized weights (getQuants() and getScales())
 * - Same attention and RoPE kernels as FP16 version
 * - 8-bit integer computations with dequantization
 * - 2x memory compression vs FP16
 * - Same combined QKV and gate/up FFN structure
 *
 * Works directly with Phi3State to access and mutate Phi3-specific state fields.
 */
public class Phi3Q8_0FFNLayers extends AbstractFFNLayers {

    TaskGraph ffnLayerTaskGraph;
    GridScheduler scheduler;
    List<ImmutableTaskGraph> ffnLayerTaskGraphs;

    // Typed references to Phi3-specific state and config
    private final Phi3State phi3State;
    private final Phi3Configuration phi3Config;

    // Phi3-specific dimension for combined QKV buffer
    private final int opSize;

    public Phi3Q8_0FFNLayers(String taskGraphName, Phi3State state, Phi3TornadoWeights weights, Phi3Configuration config, SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config, schedulerType);
        this.phi3State = state;
        this.phi3Config = config;
        this.opSize = config.dim() + 2 * (config.numberOfKeyValueHeads() * config.headSize());
        ffnLayerTaskGraphs = setupFFNLayered();
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        WorkerGrid rmsNormWorker = WorkerGridFactory.createRmsNormWorker(config.dim(), 256);
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(config.dim() / 2, 128);

        int configDimRowMajorGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configDimRowMajorGlobalWorker = WorkerGridFactory.genericWorker(configDimRowMajorGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        final int opSize = config.dim() + 2 * (config.numberOfKeyValueHeads() * config.headSize());

        int qkvmatmulDimRowMajorGlobal = opSize * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid qkvDimRowMajorGlobalWorker = WorkerGridFactory.genericWorker(qkvmatmulDimRowMajorGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int wgetUPDimRowMajor = 2 * config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid wgetHiddenDimRowMajorWorker = WorkerGridFactory.genericWorker(wgetUPDimRowMajor, LOCAL_WORK_GROUP_SIZE_ALLOC);

        WorkerGrid parallelAttentionWorker = WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), config.headSize());
        WorkerGrid copyToCachesWorker = WorkerGridFactory.genericWorker(config.dim(), 128);
        WorkerGrid splitGateUpSiLUWorker = WorkerGridFactory.genericWorker(config.hiddenDim(), 128);
        WorkerGrid splitQKVWorker = WorkerGridFactory.genericWorker(opSize, 128);
        for (int i = 0; i < config.numberOfLayers(); i++) {
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".qkvmatmul", qkvDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".splitQKV", splitQKVWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rope", ropeWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".matmul1", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".wDown", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".wGateUp", wgetHiddenDimRowMajorWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlock", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".mapContext", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlockFFN", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".mapContextFFN", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".parallel-attention", parallelAttentionWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".copyToCaches", copyToCachesWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".gateUpSiLU", splitGateUpSiLUWorker);
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

        // Initialize buffers using Phi3State directly
        phi3State.temp.init(0.0f);
        phi3State.tempFFN.init(0.0f);

        for (int layerIndex = 0; layerIndex < phi3Config.numberOfLayers(); layerIndex++) {
            TaskGraph ffnLayer = setupSinglePhi3Q8_0FFNLayer((Phi3TornadoWeights) weights, layerIndex);
            if (layerIndex == phi3Config.numberOfLayers() - 1) {
                setupLastID(ffnLayer.getTaskGraphName());
            }
            ffnGraphs.add(ffnLayer.snapshot());
        }

        return ffnGraphs;
    }

    /**
     * Setup a single transformer layer for Phi3 with Q8_0 quantization, combined QKV and gate/up FFN
     */
    TaskGraph setupSinglePhi3Q8_0FFNLayer(Phi3TornadoWeights weights, int layerIndex) {

        TaskGraph unifiedLayer = new TaskGraph("layer_" + layerIndex);
        unifiedLayer.consumeFromDevice(phi3State.wrapX);
        unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                // Copy-in quantized weights per layer
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqkvLayered[layerIndex].asByteArray(),
                weights.woLayered[layerIndex].asByteArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.wUpLayered[layerIndex].asByteArray(),
                weights.wDownLayered[layerIndex].asByteArray()
        );
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);

        // RMSNorm for attention input
        unifiedLayer.task("reductionsOneBlock",
                        TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                        context,
                        phi3State.temp,
                        phi3State.wrapX,
                        phi3Config.dim(),
                        phi3Config.rmsNormEps(),
                        phi3State.localSize)
                .task("mapContext",
                        TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                        context,
                        phi3State.wrapXb,
                        phi3State.wrapX,
                        weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                        phi3State.temp);

        // Combined QKV projection (quantized)
        unifiedLayer.task("qkvmatmul",
                        TransformerComputeKernelsLayered::matrixVectorGenericQ8Byte,
                        context,
                        phi3State.wrapXb,
                        phi3State.wrapQkv,
                        weights.wqkvLayered[layerIndex].asByteArray(),
                        phi3Config.dim(),
                        opSize,
                        LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("splitQKV",
                        TransformerComputeKernelsLayered::splitQKV,
                        phi3State.wrapQkv,
                        phi3State.wrapQ,
                        phi3State.wrapK,
                        phi3State.wrapV,
                        phi3Config.dim(),
                        phi3Config.headSize() * phi3Config.numberOfKeyValueHeads());

        // RoPE rotation (Phi3-specific kernel)
        unifiedLayer.task("rope",
                TransformerComputeKernelsLayered::ropeRotationPhi3,
                context,
                phi3State.positionHolder,
                phi3State.wrapQ,
                phi3State.wrapK,
                phi3Config.kvDim(),
                phi3Config.headSize());

        // Copy to caches
        unifiedLayer.task("copyToCaches",
                TransformerComputeKernelsLayered::copyToCache,
                phi3State.wrapKeyCache,
                phi3State.wrapK,
                phi3State.wrapValueCache,
                phi3State.wrapV,
                phi3State.positionHolder,
                phi3Config.kvDim(),
                layerIndex,
                phi3Config.contextLength());

        // Parallel attention
        unifiedLayer.task("parallel-attention",
                TransformerComputeKernelsLayered::processHeadsFlashAttention,
                context,
                phi3State.wrapQ,
                phi3State.wrapKeyCache,
                phi3State.wrapValueCache,
                phi3State.wrapXb,
                phi3Config.numberOfHeads(),
                phi3Config.headSize(),
                phi3Config.kvDim(),
                phi3Config.kvMul(),
                phi3State.positionHolder,
                layerIndex,
                phi3Config.contextLength());

        // Output projection (quantized)
        unifiedLayer.task("matmul1",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                context,
                phi3State.wrapXb,
                phi3State.wrapX,
                weights.woLayered[layerIndex].asByteArray(),
                phi3Config.dim(),
                phi3Config.dim(),
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // FFN section: RMSNorm
        unifiedLayer.task("reductionsOneBlockFFN",
                        TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                        context,
                        phi3State.tempFFN,
                        phi3State.wrapX,
                        phi3Config.dim(),
                        phi3Config.rmsNormEps(),
                        phi3State.localSize)
                .task("mapContextFFN",
                        TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                        context,
                        phi3State.wrapXb,
                        phi3State.wrapX,
                        weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                        phi3State.tempFFN);

        // FFN: combined Up and Gate projection (outputs 2 * hiddenDim, quantized)
        unifiedLayer.task("wGateUp",
                        TransformerComputeKernelsLayered::matrixVectorGenericQ8Byte,
                        context,
                        phi3State.wrapXb,
                        phi3State.wrapHb,
                        weights.wUpLayered[layerIndex].asByteArray(),
                        phi3Config.dim(),
                        2 * phi3Config.hiddenDim(),
                        LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("gateUpSiLU",
                        TransformerComputeKernelsLayered::splitGateUpAndSiLU,
                        phi3State.wrapHb,
                        phi3State.wrapHbG,
                        phi3State.wrapHbU,
                        phi3Config.hiddenDim());

        // FFN: Down projection with residual (quantized)
        unifiedLayer.task("wDown",
                        TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                        context,
                        phi3State.wrapHbU,
                        phi3State.wrapX,
                        weights.wDownLayered[layerIndex].asByteArray(),
                        phi3Config.hiddenDim(),
                        phi3Config.dim(),
                        LOCAL_WORK_GROUP_SIZE_ALLOC)
                .persistOnDevice(
                        phi3State.wrapX
                );
        return unifiedLayer;
    }

    /**
     * Configure data transfers for first and subsequent layers
     */
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        // First layer: Transfer initial data to device (one-time transfer)
        if (layerIndex == 0) {
            // Transfer all attention-related data: query, key, value matrices and their caches
            unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION, state.positionHolder, state.temp, state.tempFFN); //
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, //
                    context, state.wrapXb, state.wrapXb2, //
                    state.wrapQ, state.wrapK, state.wrapV, //
                    state.wrapKeyCache, state.wrapValueCache, //
                    state.wrapAtt, state.wrapHb, //
                    phi3State.wrapHbG, phi3State.wrapHbU, phi3State.wrapQkv); //
        } else {
            // Subsequent layers: Consume data already on device from previous layer
            unifiedLayer.consumeFromDevice(context, state.wrapXb, state.wrapXb2, //
                    state.wrapQ, state.wrapK, state.wrapV, //
                    state.wrapKeyCache, state.wrapValueCache, //
                    state.wrapAtt, state.wrapHb, //
                    state.positionHolder, // /
                    phi3State.wrapHbG, phi3State.wrapHbU, phi3State.wrapQkv);
        }
        return unifiedLayer;
    }

}
