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
    public GridScheduler updateGridScheduler(GridScheduler gridScheduler) {
        WorkerGrid rmsNormWorker = WorkerGridFactory.createRmsNormWorker(config.dim(), state.localSize);

        int qkRmsNormGroups = config.numberOfHeads() + config.numberOfKeyValueHeads();
        WorkerGrid qkRmsNormWorker = WorkerGridFactory.genericWorker(qkRmsNormGroups * nEmbdHead, nEmbdHead);

        WorkerGrid ropeWorker = WorkerGridFactory.createRoPEWorker(config.numberOfHeads(), nEmbdHead);
        WorkerGrid parallelAttentionWorker = WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), nEmbdHead);
        // attn_output_proj worker (output projection)
        int matmul1Global = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmul1Worker = WorkerGridFactory.genericWorker(matmul1Global, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int fusedFFNW1W3Global = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedFFNW1W3Worker = WorkerGridFactory.genericWorker(fusedFFNW1W3Global, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int projectionTwoGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid projectionTwoWorker = WorkerGridFactory.genericWorker(projectionTwoGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int qDim0 = nEmbdHeadK * qwen3Config.numberOfHeads();
        int kvDim0 = nEmbdGqa;
        int fusedQKVRows = qDim0 + 2 * kvDim0;  // Q rows + K rows + V rows
        int fusedQKVGlobal = fusedQKVRows * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedQKVWorker = WorkerGridFactory.genericWorker(fusedQKVGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        for (int i = 0; i < config.numberOfLayers(); i++) {
            gridScheduler.addWorkerGrid("layer_" + i + ".attn_rms_reduce", rmsNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".attn_rms_qkv_projection", fusedQKVWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".qk_rmsnorm", qkRmsNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".rope_and_kv_cache", ropeWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".attention", parallelAttentionWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".attn_output_proj", matmul1Worker);
            gridScheduler.addWorkerGrid("layer_" + i + ".ffn_rms_reduce", rmsNormWorker);
            if (shouldUseFinalNormalization()) {
                gridScheduler.addWorkerGrid("layer_" + i + ".ffn_rms_finalize", rmsNormWorker);
            }
            gridScheduler.addWorkerGrid("layer_" + i + ".rms_ffn_gate_up", fusedFFNW1W3Worker);
            gridScheduler.addWorkerGrid("layer_" + i + ".ffn_down_proj", projectionTwoWorker);
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
     * Setup a single transformer layer for Qwen3 with GQA (Q8_0 quantized)
     */
    TaskGraph setupSingleQwen3FFNLayer(Qwen3TornadoWeights weights, int layerIndex) {
        var taskGraphName = "layer_" + layerIndex;

        // === Dimension Parameters ===
        int qDim = nEmbdHeadK * qwen3Config.numberOfHeads();  // Q output size (full heads)
        int kvDim = nEmbdGqa;                                  // K/V output size (reduced for GQA)
        int inputDim = qwen3Config.dim();                      // Model dimension

        var unifiedLayer = new TaskGraph(taskGraphName);

        // === Data Setup ===
        unifiedLayer.consumeFromDevice(qwen3State.wrapX);
        unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                // Attention weights
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),   // RMS norm weights
                weights.wqLayered[layerIndex].asByteArray(),                // Q projection
                weights.wkLayered[layerIndex].asByteArray(),                // K projection
                weights.wvLayered[layerIndex].asByteArray(),                // V projection
                weights.woLayered[layerIndex].asByteArray(),                // Output projection
                // Qwen3-specific Q/K norm weights
                weights.rms_att_KNormLayered[layerIndex].asFloatArray(),    // K RMSNorm weights
                weights.rms_att_QNormLayered[layerIndex].asFloatArray(),    // Q RMSNorm weights
                // FFN weights
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),   // FFN RMSNorm weights
                weights.w1Layered[layerIndex].asByteArray(),                // FFN gate projection
                weights.w2Layered[layerIndex].asByteArray(),                // FFN down projection
                weights.w3Layered[layerIndex].asByteArray());               // FFN up projection
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);

        // ═══════════════════════════════════════════════════════════════════════
        //                           ATTENTION BLOCK
        // ═══════════════════════════════════════════════════════════════════════

        // RMS Normalization - compute scale factor
        unifiedLayer.task("attn_rms_reduce",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                context,
                qwen3State.temp,              // output: scale factor
                qwen3State.wrapX,             // input: hidden state
                config.dim(),                 // dimension
                config.rmsNormEps(),          // epsilon
                qwen3State.localSize);        // local memory size

        if (shouldUseFinalNormalization()) {
            unifiedLayer.task("attn_rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    state.temp,
                    config.dim(),
                    config.rmsNormEps());
        }

        // Fused RMS Apply + QKV Projection
        unifiedLayer.task("attn_rms_qkv_projection",
                Qwen3Kernels::fusedRmsNormQKVMatmulQ8_0,
                context,
                qwen3State.wrapX,             // input: raw hidden state (FP32)
                qwen3State.wrapQ,             // output: Q vectors
                qwen3State.wrapK,             // output: K vectors
                qwen3State.wrapV,             // output: V vectors
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),  // RMS weights
                qwen3State.temp,              // RMS scale factor from reduction
                weights.wqLayered[layerIndex].asByteArray(),               // Wq (Q8_0)
                weights.wkLayered[layerIndex].asByteArray(),               // Wk (Q8_0)
                weights.wvLayered[layerIndex].asByteArray(),               // Wv (Q8_0)
                inputDim,                     // input dimension
                qDim,                         // Q output dimension
                kvDim,                        // K/V output dimension (GQA: reduced)
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Fused Q/K RMSNorm (Qwen3-specific)
        unifiedLayer.task("qk_rmsnorm",
                Qwen3Kernels::fusedQKRmsNorm,
                context,
                qwen3State.wrapQ,             // Q vectors (in/out)
                qwen3State.wrapK,             // K vectors (in/out)
                weights.rms_att_QNormLayered[layerIndex].asFloatArray(),   // Q norm weights
                weights.rms_att_KNormLayered[layerIndex].asFloatArray(),   // K norm weights
                qwen3Config.numberOfHeads(),           // nHeads (Q heads)
                qwen3Config.numberOfKeyValueHeads(),   // nHeadKv (K/V heads, GQA)
                nEmbdHead,                    // head dimension
                nEmbdHead,                    // local memory size
                qwen3Config.rmsNormEps());    // epsilon

        // Fused RoPE Rotation + KV Cache Write
        unifiedLayer.task("rope_and_kv_cache",
                Qwen3Kernels::ropeRotationWithCacheCopy,
                context,
                qwen3State.positionHolder,    // current position
                qwen3State.wrapQ,             // Q vectors (in/out, rotated)
                qwen3State.wrapK,             // K vectors (in/out, rotated)
                qwen3State.wrapV,             // V vectors (in only)
                qwen3State.wrapKeyCache,      // key cache (out)
                qwen3State.wrapValueCache,    // value cache (out)
                qwen3Config.numberOfKeyValueHeads(),   // nHeadKv
                nEmbdHead,                    // head dimension
                nEmbdGqa,                     // kvDim
                layerIndex,                   // layer index for cache offset
                qwen3Config.contextLength()); // max sequence length

        // Flash Attention
        unifiedLayer.task("attention",
                TransformerComputeKernelsLayered::processHeadsFlashAttention,
                context,
                qwen3State.wrapQ,             // query vectors
                qwen3State.wrapKeyCache,      // key cache
                qwen3State.wrapValueCache,    // value cache
                qwen3State.wrapXb,            // output: attention result
                qwen3Config.numberOfHeads(),  // nHeads
                nEmbdHead,                    // headSize
                nEmbdGqa,                     // kvDim
                gqa,                          // kvMul (nHeads / nHeadKv)
                qwen3State.positionHolder,    // position
                layerIndex,                   // layer index
                qwen3Config.contextLength()); // context length

        // Output Projection with Residual
        unifiedLayer.task("attn_output_proj",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                context,
                qwen3State.wrapXb,  // input: attention output
                qwen3State.wrapX,   // output: wrapX += Wo · wrapXb
                weights.woLayered[layerIndex].asByteArray(),    // Wo [dim x qDim]
                nEmbdHeadK * qwen3Config.numberOfHeads(),       // input dim (qDim)
                config.dim(),       // output dim
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // ═══════════════════════════════════════════════════════════════════════
        //                              FFN BLOCK
        // ═══════════════════════════════════════════════════════════════════════

        // RMS Normalization - compute scale factor
        unifiedLayer.task("ffn_rms_reduce",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                context,
                qwen3State.tempFFN,           // output: scale factor
                qwen3State.wrapX,             // input: hidden state
                qwen3Config.dim(),            // dimension
                qwen3Config.rmsNormEps(),     // epsilon
                qwen3State.localSize);        // local memory size

        // Final normalization (non-NVIDIA only)
        if (shouldUseFinalNormalization()) {
            unifiedLayer.task("ffn_rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    qwen3State.tempFFN,       // scale factor (in/out)
                    qwen3Config.dim(),        // dimension
                    qwen3Config.rmsNormEps()); // epsilon
        }

        // Fused RMS Apply + Gate/Up Projection + SiLU + GLU
        unifiedLayer.task("rms_ffn_gate_up",
                TransformerComputeKernelsLayered::fusedRmsNormFFNGateUpQ8_0,
                context,
                qwen3State.wrapX,             // input: raw hidden state (FP32)
                qwen3State.wrapHb,            // output: SiLU(x·W1) ⊙ (x·W3)
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),  // RMS weights
                qwen3State.tempFFN,           // RMS scale factor
                weights.w1Layered[layerIndex].asByteArray(),               // W1 (gate) Q8_0
                weights.w3Layered[layerIndex].asByteArray(),               // W3 (up) Q8_0
                qwen3Config.dim(),            // input dimension
                qwen3Config.hiddenDim(),      // hidden dimension
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Down Projection with Residual
        unifiedLayer.task("ffn_down_proj",
                        TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                        context,
                        qwen3State.wrapHb,      // input: FFN intermediate
                        qwen3State.wrapX,       // output: wrapX += W2 · wrapHb
                        weights.w2Layered[layerIndex].asByteArray(),  // W2 (down)
                        config.hiddenDim(),     // input dim
                        config.dim(),           // output dim
                        LOCAL_WORK_GROUP_SIZE_ALLOC);

        unifiedLayer.persistOnDevice(state.wrapX);

        return unifiedLayer;
    }
    // @formatter:on

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