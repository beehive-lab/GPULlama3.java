package org.beehive.gpullama3.tornadovm.layers.type.q8_0;

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
 * Qwen2Q8_0FFNLayers: Q8_0-quantized FFN layers for Qwen2 with Group Query Attention (GQA) support.
 *
 * Key Differences from Qwen2FP16FFNLayers:
 * - Uses Q8_0-quantized weights (getQuants() and getScales())
 * - Same attention and RoPE kernels as FP16 version
 * - 8-bit integer computations with dequantization
 * - 2x memory compression vs FP16
 * - Includes bias terms for Q, K, V projections
 *
 * Works directly with Qwen2State to access and mutate Qwen2-specific state fields.
 */
public class Qwen2Q8_0FFNLayers extends AbstractFFNLayers {

    TaskGraph ffnLayerTaskGraph;
    GridScheduler scheduler;
    List<ImmutableTaskGraph> ffnLayerTaskGraphs;

    // Typed references to Qwen2-specific state and config
    private final Qwen2State qwen2State;
    private final Qwen2Configuration qwen2Config;

    public Qwen2Q8_0FFNLayers(String taskGraphName, Qwen2State state, Qwen2TornadoWeights weights, Qwen2Configuration config, SchedulerType schedulerType) {
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

        int fusedQKVGlobal = (config.dim() + 2 * config.kvDim()) * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedQKVWorker = new WorkerGrid1D(fusedQKVGlobal);
        fusedQKVWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

        // WorkerGrid for fused QKV bias addition (dimension is dimQ)
        WorkerGrid fusedQKVBiasWorker = new WorkerGrid1D(config.dim());
        fusedQKVBiasWorker.setGlobalWork(config.dim(), 1, 1);
        fusedQKVBiasWorker.setLocalWork(32, 1, 1);

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
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".attn_rms_reduce", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".attn_rms_qkv_projection", fusedQKVWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".fused_qkv_bias", fusedQKVBiasWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rope_and_kv_cache", ropeWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".attention", parallelAttentionWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".attn_output_proj", configDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".ffn_rms_reduce", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rms_ffn_gate_up", configHiddenDimRowMajorWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".ffn_down_proj", configDimRowMajorGlobalWorker);
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
            TaskGraph ffnLayer = setupSingleQwen2Q8_0FFNLayer((Qwen2TornadoWeights) weights, layerIndex);
            if (layerIndex == qwen2Config.numberOfLayers() - 1) {
                setupLastID(ffnLayer.getTaskGraphName());
            }
            ffnGraphs.add(ffnLayer.snapshot());
        }
        return ffnGraphs;
    }

    /**
     * Setup a single transformer layer for Qwen2 with Q8_0 quantization and GQA
     */
    TaskGraph setupSingleQwen2Q8_0FFNLayer(Qwen2TornadoWeights weights, int layerIndex) {
      TaskGraph  unifiedLayer = new TaskGraph("layer_" + layerIndex);
        unifiedLayer.consumeFromDevice(state.wrapX);
        unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                // Attention weights
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                weights.woLayered[layerIndex].asByteArray(),
                // Qwen2-specific bias terms
                weights.q_biasLayered[layerIndex].asFloatArray(),
                weights.k_biasLayered[layerIndex].asFloatArray(),
                weights.v_biasLayered[layerIndex].asFloatArray(),
                // FFN weights
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].asByteArray(),
                weights.w2Layered[layerIndex].asByteArray(),
                weights.w3Layered[layerIndex].asByteArray()
        );
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);

        // ═══════════════════════════════════════════════════════════════════════
        //                           ATTENTION BLOCK
        // ═══════════════════════════════════════════════════════════════════════

        // RMS Normalization - compute scale factor
        unifiedLayer.task("attn_rms_reduce",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                context,
                qwen2State.temp,              // output: scale factor
                qwen2State.wrapX,             // input: hidden state
                config.dim(),                 // dimension
                config.rmsNormEps(),          // epsilon
                qwen2State.localSize);        // local memory size

        unifiedLayer.task("attn_rms_qkv_projection",
                Qwen3Kernels::fusedRmsNormQKVMatmulQ8_0,
                context,
                qwen2State.wrapX,       // input: raw hidden state (FP32)
                qwen2State.wrapQ,       // output: Q vectors
                qwen2State.wrapK,       // output: K vectors
                qwen2State.wrapV,       // output: V vectors
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),   // RMS weights
                qwen2State.temp,        // RMS scale factor from reduction
                weights.wqLayered[layerIndex].asByteArray(),    // Wq (Q8_0)
                weights.wkLayered[layerIndex].asByteArray(),    // Wk (Q8_0)
                weights.wvLayered[layerIndex].asByteArray(),    // Wv (Q8_0)
                config.dim(),           // input dimension
                config.dim(),           // Q output dimension
                config.kvDim(),         // K/V output dimension (GQA: reduced)
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Fused Q/K/V Bias Addition (3→1 kernel fusion)
        unifiedLayer.task("fused_qkv_bias",
                TransformerComputeKernelsLayered::fusedQKvBiasAddition,
                context,
                qwen2State.wrapQ,             // Q (in/out)
                qwen2State.wrapK,             // K (in/out)
                weights.q_biasLayered[layerIndex].asFloatArray(),   // Q bias
                qwen2State.wrapV,             // V (in/out)
                weights.k_biasLayered[layerIndex].asFloatArray(),   // K bias
                weights.v_biasLayered[layerIndex].asFloatArray(),   // V bias
                config.dim(),                 // dimQ
                config.kvDim());              // dimKV

        // Fused RoPE Rotation + KV Cache Write
        unifiedLayer.task("rope_and_kv_cache",
                Qwen3Kernels::ropeRotationWithCacheCopy,
                context,
                qwen2State.positionHolder,    // current sequence position
                qwen2State.wrapQ,             // Q (rotated in-place)
                qwen2State.wrapK,             // K (rotated in-place)
                qwen2State.wrapV,             // V (copied to cache)
                qwen2State.wrapKeyCache,      // key cache (write)
                qwen2State.wrapValueCache,    // value cache (write)
                config.numberOfKeyValueHeads(), // nHeadKv
                config.headSize(),            // per-head dimension
                config.kvDim(),               // kvDim
                layerIndex,                   // layer offset
                config.contextLength());      // max sequence length

        // Flash Attention
        unifiedLayer.task("attention",
                Qwen2Kernels::processHeadsFlashAttention,
                context,
                qwen2State.wrapQ,             // query vectors
                qwen2State.wrapKeyCache,      // key cache
                qwen2State.wrapValueCache,    // value cache
                qwen2State.wrapXb,            // output: attention result
                config.numberOfHeads(),       // nHeads
                config.headSize(),            // headSize
                config.kvDim(),               // kvDim
                config.kvMul(),               // kvMul (nHeads / nHeadKv)
                qwen2State.positionHolder,    // position
                layerIndex,                   // layer index
                config.contextLength());      // context length

        // Output Projection with Residual
        unifiedLayer.task("attn_output_proj",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                context,
                qwen2State.wrapXb,            // input: attention output
                qwen2State.wrapX,             // output: wrapX += Wo · wrapXb
                weights.woLayered[layerIndex].asByteArray(),  // Wo
                config.dim(),                 // input dim
                config.dim(),                 // output dim
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // ═══════════════════════════════════════════════════════════════════════
        //                              FFN BLOCK
        // ═══════════════════════════════════════════════════════════════════════

        // RMS Normalization - compute scale factor
        unifiedLayer.task("ffn_rms_reduce",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                context,
                qwen2State.tempFFN,           // output: scale factor
                qwen2State.wrapX,             // input: hidden state
                config.dim(),                 // dimension
                config.rmsNormEps(),          // epsilon
                qwen2State.localSize);        // local memory size

        // Final normalization (non-NVIDIA only)
        if (shouldUseFinalNormalization()) {
            unifiedLayer.task("ffn_rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    qwen2State.tempFFN,       // scale factor (in/out)
                    config.dim(),             // dimension
                    config.rmsNormEps());     // epsilon
        }

        // Fused RMS Apply + Gate/Up Projection + SiLU + GLU
        // (Replaces mapContextFFN + fusedFeedForwardWithSiLUAndGLUActivation)
        unifiedLayer.task("rms_ffn_gate_up",
                TransformerComputeKernelsLayered::fusedRmsNormFFNGateUpQ8_0,
                context,
                qwen2State.wrapX,             // input: raw hidden state (FP32)
                qwen2State.wrapHb,            // output: SiLU(x·W1) ⊙ (x·W3)
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),  // RMS weights
                qwen2State.tempFFN,           // RMS scale factor
                weights.w1Layered[layerIndex].asByteArray(),          // W1 (gate)
                weights.w3Layered[layerIndex].asByteArray(),          // W3 (up)
                config.dim(),                 // input dimension
                config.hiddenDim(),           // hidden dimension
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Down Projection with Residual
        unifiedLayer.task("ffn_down_proj",
                        TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                        context,
                        qwen2State.wrapHb,            // input: FFN intermediate
                        qwen2State.wrapX,             // output: wrapX += W2 · wrapHb
                        weights.w2Layered[layerIndex].asByteArray(),  // W2 (down)
                        config.hiddenDim(),           // input dim
                        config.dim(),                 // output dim
                        LOCAL_WORK_GROUP_SIZE_ALLOC);

        unifiedLayer.persistOnDevice(state.wrapX);

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
