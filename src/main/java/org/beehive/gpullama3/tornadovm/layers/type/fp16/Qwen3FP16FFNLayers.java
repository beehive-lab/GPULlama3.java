package org.beehive.gpullama3.tornadovm.layers.type.fp16;

import org.beehive.gpullama3.inference.state.Qwen3State;
import org.beehive.gpullama3.inference.weights.tornado.fp16.Qwen3TornadoWeights;
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
 * Qwen3FP16FFNLayers: FP16 FFN layers for Qwen3 with Group Query Attention (GQA) support.
 *
 * Key Differences from Llama: - Supports GQA with separate KV heads (nHeadKv) - Uses Qwen3Kernels for RMSNorm with parallel offset - Custom RoPE rotation for Qwen3 - Different attention computation
 * due to GQA structure
 *
 * Works directly with Qwen3State to access and mutate Qwen3-specific state fields like tempQcur and tempKcur.
 */
public class Qwen3FP16FFNLayers extends AbstractFFNLayers {

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
    TaskGraph ffnLayerTaskGraph;
    GridScheduler scheduler;
    List<ImmutableTaskGraph> ffnLayerTaskGraphs;

    public Qwen3FP16FFNLayers(String taskGraphName, Qwen3State state, Qwen3TornadoWeights weights, Qwen3Configuration config, SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config, schedulerType);
        this.qwen3State = state;
        this.qwen3Config = config;

        // Initialize GQA parameters from Qwen3Config
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
    public GridScheduler updateGridScheduler(GridScheduler gridScheduler) {
        WorkerGrid rmsNormWorker = WorkerGridFactory.createRmsNormWorker(config.dim(), state.localSize);

        // Q matmul worker (GQA: full query heads)
        int matmulQGlobal = nEmbdHeadK * config.numberOfHeads() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmulQRowMajorWorker = WorkerGridFactory.genericWorker(matmulQGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // KV matmul worker (GQA: reduced KV heads)
        int matmulKVGlobal = nEmbdGqa * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmulKVRowMajorWorker = WorkerGridFactory.genericWorker(matmulKVGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Current embedding head worker
        WorkerGrid curWorker = WorkerGridFactory.createRmsNormWorker(nEmbdHead, 128);

        // Q current worker
        WorkerGrid qCurWorker = WorkerGridFactory.genericWorker(config.numberOfHeads() * nEmbdHead, nEmbdHead);

        // K current worker
        WorkerGrid kCurWorker = WorkerGridFactory.genericWorker(config.numberOfKeyValueHeads() * nEmbdHead, nEmbdHead);

        // RoPE worker (2D: heads x embedding_head/2)
        int ic = nEmbdHead / 2;
        WorkerGrid ropeWorker = WorkerGridFactory.createRoPEWorker(config.numberOfHeads(), nEmbdHead);

        // Copy to cache worker
        WorkerGrid copyToCachesWorker = WorkerGridFactory.genericWorker(nEmbdGqa, 128);

        // Parallel attention worker
        WorkerGrid parallelAttentionWorker = WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), nEmbdHead);

        // Matmul1 worker (output projection)
        int matmul1Global = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmul1Worker = WorkerGridFactory.genericWorker(matmul1Global, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // FFN workers
        int fusedFFNW1W3Global = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedFFNW1W3Worker = WorkerGridFactory.genericWorker(fusedFFNW1W3Global, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int projectionTwoGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid projectionTwoWorker = WorkerGridFactory.genericWorker(projectionTwoGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Map workers to tasks for each layer
        for (int i = 0; i < config.numberOfLayers(); i++) {
            gridScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlock", rmsNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".mapContext", rmsNormWorker);

            gridScheduler.addWorkerGrid("layer_" + i + ".qmatmul", matmulQRowMajorWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".kmatmul", matmulKVRowMajorWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".vmatmul", matmulKVRowMajorWorker);

            gridScheduler.addWorkerGrid("layer_" + i + ".rmsnormReduction_Qcur", qCurWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".rmsnormMapIndexInPlace_Qcur", qCurWorker);

            gridScheduler.addWorkerGrid("layer_" + i + ".rmsnormReduction_Kcur", kCurWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".rmsnormMapIndexInPlace_Kcur", kCurWorker);

            gridScheduler.addWorkerGrid("layer_" + i + ".ropeRotation", ropeWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".copyToCaches", copyToCachesWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".parallel-attention", parallelAttentionWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".matmul1", matmul1Worker);

            gridScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlockFFN", rmsNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".mapContextFFN", rmsNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".fused_ffn_w1_w3", fusedFFNW1W3Worker);
            gridScheduler.addWorkerGrid("layer_" + i + ".projectionTwo", projectionTwoWorker);
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
     * Setup a single transformer layer for Qwen3 with GQA
     */
    TaskGraph setupSingleQwen3FFNLayer(Qwen3TornadoWeights weights, int layerIndex) {
        var taskGraphName = "ffn_layer_" + layerIndex;
        TaskGraph unifiedLayer = new TaskGraph(taskGraphName);
        unifiedLayer.consumeFromDevice(state.wrapX);
        unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, //
                //Copy-in weights per layer for batched-layered layout
                weights.rms_att_weightLayered[layerIndex], //
                weights.wqLayered[layerIndex], //
                weights.wkLayered[layerIndex], //
                weights.wvLayered[layerIndex], //
                weights.woLayered[layerIndex], //
                //rms_att_KNormLayered
                weights.rms_att_KNormLayered[layerIndex], //
                //rms_att_QNormLayered
                weights.rms_att_QNormLayered[layerIndex], //
                weights.rms_ffn_weightLayered[layerIndex], //
                weights.w1Layered[layerIndex], //
                weights.w2Layered[layerIndex], //
                weights.w3Layered[layerIndex] //
        );
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);
        unifiedLayer.task("reductionsOneBlock", TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, qwen3State.temp, qwen3State.wrapX, // in
                qwen3Config.dim(), qwen3Config.rmsNormEps(), qwen3State.localSize).task("mapContext", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, qwen3State.wrapXb, // out
                qwen3State.wrapX, weights.rms_att_weightLayered[layerIndex], qwen3State.temp);

        int qDim0 = nEmbdHeadK * qwen3Config.numberOfHeads();
        int kvDim0 = nEmbdGqa;
        int qkvDim1 = qwen3Config.dim();
        unifiedLayer.task("qmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context, qwen3State.wrapXb, qwen3State.wrapQ,                    // output
                        weights.wqLayered[layerIndex], qkvDim1, qDim0, LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("kmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context, qwen3State.wrapXb, qwen3State.wrapK,        // output
                        weights.wkLayered[layerIndex], qkvDim1, kvDim0, LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("vmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context, qwen3State.wrapXb, qwen3State.wrapV,        // output
                        weights.wvLayered[layerIndex], qkvDim1, kvDim0, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Qcur rmsnorm
        unifiedLayer.task("rmsnormReduction_Qcur", Qwen3Kernels::rmsnormWithParallelOffset, context, qwen3State.tempQcur,         // output
                        qwen3State.wrapQ,            // input
                        qwen3State.localSize,        // currently 128, should be variable of global nEmbHead
                        nEmbdHead,                   // for normalization
                        qwen3Config.rmsNormEps())    // for normalization
                .task("rmsnormMapIndexInPlace_Qcur", Qwen3Kernels::rmsnormMapIndexInPlaceWithParallelOffset, context, qwen3State.wrapQ,        // output
                        weights.rms_att_QNormLayered[layerIndex], nEmbdHead, qwen3State.tempQcur);

        // Kcur rmsnorm
        unifiedLayer.task("rmsnormReduction_Kcur", Qwen3Kernels::rmsnormWithParallelOffset, context, qwen3State.tempKcur,         // output
                        qwen3State.wrapK,            // input
                        qwen3State.localSize,        // currently 128, should be variable of global nEmbHead
                        nEmbdHead,                   // for normalization
                        qwen3Config.rmsNormEps())    // for normalization
                .task("rmsnormMapIndexInPlace_Kcur", Qwen3Kernels::rmsnormMapIndexInPlaceWithParallelOffset, context, qwen3State.wrapK,        // output
                        weights.rms_att_KNormLayered[layerIndex], nEmbdHead, qwen3State.tempKcur);

        // rope rotation task graph
        unifiedLayer.task("ropeRotation", Qwen3Kernels::ropeRotation, context, qwen3State.positionHolder, qwen3State.wrapQ,            // out
                qwen3State.wrapK,            // out
                qwen3Config.numberOfKeyValueHeads(), nEmbdHead);

        unifiedLayer.task("copyToCaches", TransformerComputeKernelsLayered::copyToCache, qwen3State.wrapKeyCache,         // out
                qwen3State.wrapK,                // in
                qwen3State.wrapValueCache,       // out
                qwen3State.wrapV,                // in
                qwen3State.positionHolder, nEmbdGqa, layerIndex, qwen3Config.contextLength());

        unifiedLayer.task("parallel-attention", TransformerComputeKernelsLayered::processHeadsFlashAttentionOpt, context, qwen3State.wrapQ, qwen3State.wrapKeyCache, qwen3State.wrapValueCache,
                qwen3State.wrapXb,               // out
                qwen3Config.numberOfHeads(), nEmbdHead, nEmbdGqa, gqa, qwen3State.positionHolder, layerIndex, qwen3Config.contextLength());

        unifiedLayer.task("matmul1", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context, qwen3State.wrapXb,                           // vector
                qwen3State.wrapX,                            // out, should be [1024]
                weights.woLayered[layerIndex],               // matrix
                nEmbdHeadK * qwen3Config.numberOfHeads(),    // dim1 = 2048
                qwen3Config.dim(),                           // dim0 = 1024
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        unifiedLayer.task("reductionsOneBlockFFN", TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, qwen3State.tempFFN, qwen3State.wrapX, qwen3Config.dim(),
                        qwen3Config.rmsNormEps(), qwen3State.localSize)
                .task("reductionFinalNormalizationFFN", TransformerComputeKernelsLayered::reductionFinalNormalization, context, qwen3State.tempFFN, qwen3Config.dim(), qwen3Config.rmsNormEps())
                .task("mapContextFFN", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, qwen3State.wrapXb, qwen3State.wrapX, weights.rms_ffn_weightLayered[layerIndex],
                        qwen3State.tempFFN);

        unifiedLayer.task("fused_ffn_w1_w3", TransformerComputeKernelsLayered::fusedFeedForwardWithSiLUAndGLUActivation, context, qwen3State.wrapXb, qwen3State.wrapHb, weights.w1Layered[layerIndex],
                        weights.w3Layered[layerIndex], qwen3Config.dim(), qwen3Config.hiddenDim(), LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("projectionTwo", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context, qwen3State.wrapHb, qwen3State.wrapX, weights.w2Layered[layerIndex],
                        qwen3Config.hiddenDim(), qwen3Config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC).persistOnDevice(qwen3State.wrapX);
        return unifiedLayer;
    }

    /**
     * Configure data transfers for first and subsequent layers
     */
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        if (layerIndex == 0) {
            // First layer: Transfer temporary buffers and QKV state every execution
            unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION, qwen3State.positionHolder, qwen3State.temp, qwen3State.tempFFN);
            unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION, qwen3State.tempQcur, qwen3State.tempKcur);
            // First execution: allocate workspace buffers
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, //
                    context, qwen3State.wrapXb, qwen3State.wrapXb2,  //
                    qwen3State.wrapQ, qwen3State.wrapK, qwen3State.wrapV, //
                    qwen3State.wrapKeyCache, qwen3State.wrapValueCache,  //
                    qwen3State.wrapAtt, qwen3State.wrapHb);
        } else {
            // Subsequent layers: Consume data from previous layer
            unifiedLayer.consumeFromDevice(context, qwen3State.wrapXb, qwen3State.wrapXb2, //
                    qwen3State.wrapQ, qwen3State.wrapK,  //
                    qwen3State.wrapV, qwen3State.wrapKeyCache, //
                    qwen3State.wrapValueCache, qwen3State.wrapAtt, //
                    qwen3State.wrapHb, qwen3State.positionHolder); //

            unifiedLayer.consumeFromDevice(qwen3State.tempQcur, qwen3State.tempKcur);
        }
        return unifiedLayer;
    }

}