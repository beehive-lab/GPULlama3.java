package org.beehive.gpullama3.tornadovm.layers.type.q8_0;

import org.beehive.gpullama3.inference.state.Qwen3State;
import org.beehive.gpullama3.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
import org.beehive.gpullama3.tornadovm.kernels.Qwen3Kernels;
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
 * Qwen3Q8_0FFNLayers: Q8_0-quantized FFN layers for Qwen3 with Group Query Attention (GQA) support.
 *
 * Key Differences from Qwen3FP16FFNLayers:
 * - Uses Q8_0-quantized weights (getQuants() and getScales())
 * - Same Qwen3Kernels for RMSNorm and RoPE
 * - 8-bit integer computations with dequantization
 * - 2x memory compression vs FP16
 *
 * Works directly with Qwen3State to access and mutate Qwen3-specific state fields
 * like tempQcur and tempKcur.
 */
public class Qwen3Q8_0FFNLayers extends AbstractFFNLayers {

    String lastTaskGraphID;
    TaskGraph ffnLayerTaskGraph;
    GridScheduler scheduler;
    List<ImmutableTaskGraph> ffnLayerTaskGraphs;

    // Typed references to Qwen3-specific state and config
    private final Qwen3State qwen3State;
    private final Qwen3Configuration qwen3Config;

    // Qwen3-specific GQA parameters
    private final int nHeadKv;
    private final int nEmbdHeadK;
    private final int nEmbdHeadV;
    private final int nEmbdVGqa;
    private final int nEmbdHead;
    private final int nEmbdGqa;
    private final int gqa;

    public Qwen3Q8_0FFNLayers(String taskGraphName, Qwen3State state, Qwen3TornadoWeights weights, Qwen3Configuration config, SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config, schedulerType);
        this.qwen3State = state;
        this.qwen3Config = config;
        this.nHeadKv = config.numberOfKeyValueHeads();
        this.nEmbdHeadK = config.numberOfHeadsKey();
        this.nEmbdHeadV = config.numberOfHeadsValue();
        this.nEmbdVGqa = nEmbdHeadV * nHeadKv;
        this.nEmbdHead = nEmbdHeadV;
        this.nEmbdGqa = nEmbdVGqa;
        this.gqa = config.numberOfHeads() / config.numberOfKeyValueHeads();
        ffnLayerTaskGraphs = setupFFNLayered();
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        WorkerGrid rmsNormWorker = WorkerGridFactory.createRmsNormWorker(config.dim(), state.localSize);

        int matmulQGlobal = nEmbdHeadK * config.numberOfHeads() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmulQRowMajorWorker = WorkerGridFactory.genericWorker(matmulQGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int matmulKVGlobal = nEmbdGqa * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmulKVRowMajorWorker = WorkerGridFactory.genericWorker(matmulKVGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        WorkerGrid qCurWorker = WorkerGridFactory.genericWorker(config.numberOfHeads() * nEmbdHead, nEmbdHead);
        WorkerGrid kCurWorker = WorkerGridFactory.genericWorker(config.numberOfKeyValueHeads() * nEmbdHead, nEmbdHead);

        int h = config.numberOfHeads();
        int ic = nEmbdHead / 2;
        WorkerGrid ropeWorker = WorkerGridFactory.createRoPEWorker(h, nEmbdHead);
        WorkerGrid copyToCachesWorker = WorkerGridFactory.genericWorker(nEmbdGqa, 128);
        WorkerGrid parallelAttentionWorker = WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), nEmbdHead);

        int matmul1Global = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmul1Worker = WorkerGridFactory.genericWorker(matmul1Global, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int fusedFFNW1W3Global = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedFFNW1W3Worker = WorkerGridFactory.genericWorker(fusedFFNW1W3Global, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int projectionTwoGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid projectionTwoWorker = WorkerGridFactory.genericWorker(projectionTwoGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        for (int i = 0; i < config.numberOfLayers(); i++) {
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlock", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".mapContext", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".qmatmul", matmulQRowMajorWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".kmatmul", matmulKVRowMajorWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".vmatmul", matmulKVRowMajorWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rmsnormReduction_Qcur", qCurWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rmsnormMapIndexInPlace_Qcur", qCurWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rmsnormReduction_Kcur", kCurWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rmsnormMapIndexInPlace_Kcur", kCurWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".ropeRotation", ropeWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".copyToCaches", copyToCachesWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".parallel-attention", parallelAttentionWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".matmul1", matmul1Worker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlockFFN", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".mapContextFFN", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".fused_ffn_w1_w3", fusedFFNW1W3Worker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".projectionTwo", projectionTwoWorker);
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
        qwen3State.temp.init(0.0f);
        qwen3State.tempFFN.init(0.0f);
        qwen3State.tempQcur.init(0.0f);
        qwen3State.tempKcur.init(0.0f);

        for (int layerIndex = 0; layerIndex < qwen3Config.numberOfLayers(); layerIndex++) {
            TaskGraph ffnLayer = setupSingleQwen3FFNLayer((Qwen3TornadoWeights) weights, layerIndex);
            if (layerIndex == qwen3Config.numberOfLayers() - 1) {
                setupLastID(ffnLayer.getTaskGraphName());
            }
            ffnGraphs.add(ffnLayer.snapshot());
        }
        return ffnGraphs;
    }

    /**
     * Setup a single transformer layer for Qwen3 with GQA (Q8_0 quantized)
     */
    TaskGraph setupSingleQwen3FFNLayer(Qwen3TornadoWeights weights, int layerIndex) {

        var unifiedLayerName = "layer_" + layerIndex;
        TaskGraph unifiedLayer = new TaskGraph(unifiedLayerName);
        unifiedLayer.consumeFromDevice(qwen3State.wrapX);
        // Transfer Q8_0 weights for this layer (quants and scales)
        unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(), //
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                weights.woLayered[layerIndex].asByteArray(),
                weights.rms_att_KNormLayered[layerIndex].asFloatArray(), //
                weights.rms_att_QNormLayered[layerIndex].asFloatArray(),//
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(), //
                weights.w1Layered[layerIndex].asByteArray(),
                weights.w2Layered[layerIndex].asByteArray(),
                weights.w3Layered[layerIndex].asByteArray());

        // Configure layer data transfers (EVERY_EXECUTION and device persistence)
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);


        // RMS norm for attention input
        unifiedLayer.task("reductionsOneBlock",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                context, qwen3State.temp, qwen3State.wrapX, config.dim(), config.rmsNormEps(), qwen3State.localSize)
                .task("mapContext",
                        TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                        context, qwen3State.wrapXb, qwen3State.wrapX, weights.rms_att_weightLayered[layerIndex].asFloatArray(), qwen3State.temp);

        // QKV projections with Qwen3 GQA dimensions
        // Q8_0 weights pass both quants and scales
        int qDim0 = nEmbdHeadK * config.numberOfHeads();  // Query dimension
        int kvDim0 = nEmbdGqa;                             // KV dimension (smaller due to GQA)
        int qkvDim1 = config.dim();                        // Input dimension

        unifiedLayer.task("qmatmul",
                TransformerComputeKernelsLayered::matrixVectorGenericQ8Byte,
                context, qwen3State.wrapXb, qwen3State.wrapQ,
                weights.wqLayered[layerIndex].asByteArray(),
                qkvDim1, qDim0, LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("kmatmul",
                        TransformerComputeKernelsLayered::matrixVectorGenericQ8Byte,
                        context, qwen3State.wrapXb, qwen3State.wrapK,
                        weights.wkLayered[layerIndex].asByteArray(),
                        qkvDim1, kvDim0, LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("vmatmul",
                        TransformerComputeKernelsLayered::matrixVectorGenericQ8Byte,
                        context, qwen3State.wrapXb, qwen3State.wrapV,
                        weights.wvLayered[layerIndex].asByteArray(),
                        qkvDim1, kvDim0, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Qcur: RMS norm with parallel offset for Query
        Qwen3State qwen3State = (Qwen3State) state;
        unifiedLayer.task("rmsnormReduction_Qcur",
                Qwen3Kernels::rmsnormWithParallelOffset,
                context, qwen3State.tempQcur, qwen3State.wrapQ, qwen3State.localSize, nEmbdHead, config.rmsNormEps())
                .task("rmsnormMapIndexInPlace_Qcur",
                        Qwen3Kernels::rmsnormMapIndexInPlaceWithParallelOffset,
                        context, qwen3State.wrapQ, weights.rms_att_QNormLayered[layerIndex].asFloatArray(), nEmbdHead, qwen3State.tempQcur);

        // Kcur: RMS norm with parallel offset for Key
        unifiedLayer.task("rmsnormReduction_Kcur",
                Qwen3Kernels::rmsnormWithParallelOffset,
                context, qwen3State.tempKcur, qwen3State.wrapK, qwen3State.localSize, nEmbdHead, config.rmsNormEps())
                .task("rmsnormMapIndexInPlace_Kcur",
                        Qwen3Kernels::rmsnormMapIndexInPlaceWithParallelOffset,
                        context, qwen3State.wrapK, weights.rms_att_KNormLayered[layerIndex].asFloatArray(), nEmbdHead, qwen3State.tempKcur);

        // RoPE rotation (Qwen3 variant)
        unifiedLayer.task("ropeRotation",
                Qwen3Kernels::ropeRotation,
                context, qwen3State.positionHolder, qwen3State.wrapQ, qwen3State.wrapK,
                config.numberOfKeyValueHeads(), nEmbdHead);

        // Copy to KV cache
        unifiedLayer.task("copyToCaches",
                TransformerComputeKernelsLayered::copyToCache,
                qwen3State.wrapKeyCache, qwen3State.wrapK, qwen3State.wrapValueCache, qwen3State.wrapV,
                qwen3State.positionHolder, nEmbdGqa, layerIndex, config.contextLength());

        // Parallel attention (with GQA support)
        unifiedLayer.task("parallel-attention",
                TransformerComputeKernelsLayered::processHeadsFlashAttentionOpt,
                context, qwen3State.wrapQ, qwen3State.wrapKeyCache, qwen3State.wrapValueCache, qwen3State.wrapXb,
                config.numberOfHeads(), nEmbdHead, nEmbdGqa, gqa, qwen3State.positionHolder, layerIndex, config.contextLength());

        // Output projection (Q8_0 weights)
        unifiedLayer.task("matmul1",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                context, qwen3State.wrapXb, qwen3State.wrapX,
                weights.woLayered[layerIndex].asByteArray(),
                qDim0, config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC);

        // ========== FEED-FORWARD BLOCK ==========

        // RMS norm for FFN input
        unifiedLayer.task("reductionsOneBlockFFN",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                context, qwen3State.tempFFN, qwen3State.wrapX, config.dim(), config.rmsNormEps(), qwen3State.localSize)
                .task("mapContextFFN",
                        TransformerComputeKernelsLayered::reductionOneBlock2WithLayer,
                        context, qwen3State.wrapXb, qwen3State.wrapX, weights.rms_ffn_weightLayered[layerIndex].asFloatArray(), qwen3State.tempFFN);

        // Fused FFN: w1(x) ⊗ w3(x) with SiLU activation (Q8_0 weights)
        unifiedLayer.task("fused_ffn_w1_w3",
                TransformerComputeKernelsLayered::fusedFeedForwardWithSiLUAndGLUActivationQ8_0Byte,
                context, qwen3State.wrapXb, qwen3State.wrapHb,
                weights.w1Layered[layerIndex].asByteArray(),
                weights.w3Layered[layerIndex].asByteArray(),
                config.dim(), config.hiddenDim(), LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("projectionTwo",
                        TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                        context, qwen3State.wrapHb, qwen3State.wrapX,
                        weights.w2Layered[layerIndex].asByteArray(),
                        config.hiddenDim(), config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC)
                .persistOnDevice(state.wrapX);

        return unifiedLayer;
    }

    /**
     * Configure data transfers for first and subsequent layers
     */
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        if (layerIndex == 0) {
            // First layer: Transfer temporary buffers and QKV state every execution
            unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                    qwen3State.positionHolder, qwen3State.temp, qwen3State.tempFFN);

            Qwen3State qwen3State = (Qwen3State) state;
            unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                    qwen3State.tempQcur, qwen3State.tempKcur);

            // First execution: allocate workspace buffers
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, //
                    context, qwen3State.wrapXb, qwen3State.wrapXb2,  //
                    qwen3State.wrapQ, qwen3State.wrapK, qwen3State.wrapV, //
                    qwen3State.wrapKeyCache, qwen3State.wrapValueCache, //
                    qwen3State.wrapAtt, qwen3State.wrapHb); //
        } else {
            // Subsequent layers: Consume data from previous layer
            unifiedLayer.consumeFromDevice(context, qwen3State.wrapXb, qwen3State.wrapXb2, //
                    qwen3State.wrapQ, qwen3State.wrapK, qwen3State.wrapV, //
                    qwen3State.wrapKeyCache, qwen3State.wrapValueCache, //
                    qwen3State.wrapAtt, qwen3State.wrapHb, qwen3State.positionHolder); //

            Qwen3State qwen3State = (Qwen3State) state;
            unifiedLayer.consumeFromDevice(qwen3State.tempQcur, qwen3State.tempKcur); //
        }
        return unifiedLayer;
    }

}