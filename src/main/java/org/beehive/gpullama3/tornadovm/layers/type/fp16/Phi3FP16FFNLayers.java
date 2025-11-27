package org.beehive.gpullama3.tornadovm.layers.type.fp16;

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
 * Phi3FP16FFNLayers: FP16 FFN layers for Phi3 with Group Query Attention (GQA) support.
 *
 * Key Differences from Qwen2/Qwen3:
 * - Uses combined QKV matrix (wqkv) instead of separate Q, K, V matrices
 * - Includes splitQKV task to separate combined buffer
 * - Uses ropeRotationPhi3 kernel for position embeddings
 * - FFN uses single wUp matrix that outputs both Gate and Up (2 * hiddenDim)
 * - Includes splitGateUpAndSiLU task for FFN activation
 * - Uses wDown for final FFN projection
 * - No Q, K, V bias terms
 *
 * Works directly with Phi3State to access and mutate Phi3-specific state fields.
 */
public class Phi3FP16FFNLayers extends AbstractFFNLayers {

    TaskGraph ffnLayerTaskGraph;
    GridScheduler scheduler;
    List<ImmutableTaskGraph> ffnLayerTaskGraphs;

    // Typed references to Phi3-specific state and config
    private final Phi3State phi3State;
    private final Phi3Configuration phi3Config;

    // Phi3-specific dimension for combined QKV buffer
    private final int opSize;

    public Phi3FP16FFNLayers(String taskGraphName, Phi3State state, Phi3TornadoWeights weights, Phi3Configuration config, SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config,schedulerType);
        this.phi3State = state;
        this.phi3Config = config;

        // Ensure we have Phi3-specific weights
        if (!(weights instanceof Phi3TornadoWeights phi3Weights)) {
            throw new IllegalArgumentException("Phi3FP16FFNLayers requires Phi3TornadoWeights with TornadoTensor layout");
        }

        // Calculate opSize for combined QKV buffer
        // opSize = num_heads * head_dim + 2 * (num_key_value_heads * head_dim) = dim + 2 * kvDim
        this.opSize = config.dim() + 2 * (config.numberOfKeyValueHeads() * config.headSize());
        ffnLayerTaskGraphs = setupFFNLayered();
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler gridScheduler) {
        // RMS norm worker
        WorkerGrid rmsNormWorker = WorkerGridFactory.createRmsNormWorker(config.dim(), state.localSize);

        // Combined QKV matmul worker
        int matmulQkvGlobal = opSize * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmulQkvRowMajorWorker = WorkerGridFactory.genericWorker(matmulQkvGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // RoPE worker (2D: heads x embedding_head/2)
        int ic = config.headSize() / 2;
        WorkerGrid ropeWorker = WorkerGridFactory.createRoPEWorker(config.numberOfHeads(), config.headSize());

        // Copy to cache worker
        WorkerGrid copyToCachesWorker = WorkerGridFactory.genericWorker(config.kvDim(), 32);

        // Parallel attention worker
        WorkerGrid parallelAttentionWorker = WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), config.headSize());

        // Matmul1 worker (output projection)
        int matmul1Global = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmul1Worker = WorkerGridFactory.genericWorker(matmul1Global, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // FFN workers
        int ffnUpGlobal = (2 * config.hiddenDim()) * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid ffnUpWorker = WorkerGridFactory.genericWorker(ffnUpGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int ffnDownGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid ffnDownWorker = WorkerGridFactory.genericWorker(ffnDownGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Map workers to tasks for each layer
        for (int i = 0; i < config.numberOfLayers(); i++) {
            gridScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlock", rmsNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".mapContext", rmsNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".qkvmatmul", matmulQkvRowMajorWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".rope", ropeWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".copyToCaches", copyToCachesWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".parallel-attention", parallelAttentionWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".matmul1", matmul1Worker);
            gridScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlockFFN", rmsNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".mapContextFFN", rmsNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".wGateUp", ffnUpWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".wDown", ffnDownWorker);
        }

        return gridScheduler;
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
            TaskGraph ffnLayer = setupSinglePhi3FFNLayer((Phi3TornadoWeights) weights, layerIndex);
            if (layerIndex == phi3Config.numberOfLayers() - 1) {
                setupLastID(ffnLayer.getTaskGraphName());
            }
            ffnGraphs.add(ffnLayer.snapshot());
        }

        return ffnGraphs;
    }

    /**
     * Setup a single transformer layer for Phi3 with combined QKV and gate/up FFN
     */
    TaskGraph setupSinglePhi3FFNLayer(Phi3TornadoWeights weights, int layerIndex) {

        TaskGraph unifiedLayer = new TaskGraph("layer_" + layerIndex);
        unifiedLayer.consumeFromDevice(phi3State.wrapX);
        unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                // Copy-in weights per layer for batched-layered layout
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqkvLayered[layerIndex].asHalfFloatArray(),
                weights.woLayered[layerIndex].asHalfFloatArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.wUpLayered[layerIndex].asHalfFloatArray(),
                weights.wDownLayered[layerIndex].asHalfFloatArray()
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

        // Combined QKV projection
        unifiedLayer.task("qkvmatmul",
                        TransformerComputeKernelsLayered::matrixVectorGeneric,
                        context,
                        phi3State.wrapXb,
                        phi3State.wrapQkv,
                        weights.wqkvLayered[layerIndex].asHalfFloatArray(),
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

        // Output projection
        unifiedLayer.task("matmul1",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                context,
                phi3State.wrapXb,
                phi3State.wrapX,
                weights.woLayered[layerIndex].asHalfFloatArray(),
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

        // FFN: combined Up and Gate projection (outputs 2 * hiddenDim)
        unifiedLayer.task("wGateUp",
                        TransformerComputeKernelsLayered::matrixVectorGeneric,
                        context,
                        phi3State.wrapXb,
                        phi3State.wrapHb,
                        weights.wUpLayered[layerIndex].asHalfFloatArray(),
                        phi3Config.dim(),
                        2 * phi3Config.hiddenDim(),
                        LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("gateUpSiLU",
                        TransformerComputeKernelsLayered::splitGateUpAndSiLU,
                        phi3State.wrapHb,
                        phi3State.wrapHbG,
                        phi3State.wrapHbU,
                        phi3Config.hiddenDim());

        // FFN: Down projection with residual
        unifiedLayer.task("wDown",
                        TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                        context,
                        phi3State.wrapHbU,
                        phi3State.wrapX,
                        weights.wDownLayered[layerIndex].asHalfFloatArray(),
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
