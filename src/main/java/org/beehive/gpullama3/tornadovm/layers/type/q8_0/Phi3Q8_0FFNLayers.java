package org.beehive.gpullama3.tornadovm.layers.type.q8_0;

import org.beehive.gpullama3.inference.state.Phi3State;
import org.beehive.gpullama3.inference.weights.tornado.Phi3TornadoWeights;
import org.beehive.gpullama3.model.phi3.Phi3Configuration;
import org.beehive.gpullama3.tornadovm.kernels.Phi3Kernels;
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
 * Key Differences from Phi3FP16FFNLayers: - Uses Q8_0-quantized weights (getQuants() and getScales()) - Same attention and RoPE kernels as FP16 version - 8-bit integer computations with
 * dequantization - 2x memory compression vs FP16 - Same combined QKV and gate/up FFN structure
 *
 * Works directly with Phi3State to access and mutate Phi3-specific state fields.
 */
public class Phi3Q8_0FFNLayers extends AbstractFFNLayers {

    // Typed references to Phi3-specific state and config
    private final Phi3State phi3State;
    private final Phi3Configuration phi3Config;
    // Phi3-specific dimension for combined QKV buffer
    private final int opSize;
    TaskGraph ffnLayerTaskGraph;
    GridScheduler scheduler;
    List<ImmutableTaskGraph> ffnLayerTaskGraphs;

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

        WorkerGrid parallelAttentionWorker = WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), config.headSize());
        int ffnFusedGlobal = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid ffnFusedWorker = WorkerGridFactory.genericWorker(ffnFusedGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        for (int i = 0; i < config.numberOfLayers(); i++) {
            //                           ATTENTION BLOCK
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".attn_rms_reduce", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".attn_rms_qkv_projection_q8", qkvDimRowMajorGlobalWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rope_and_kv_cache", ropeWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".attention", parallelAttentionWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".attn_output_proj", configDimRowMajorGlobalWorker);
            //                              FFN BLOCK
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".ffn_rms_reduce", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".ffn_rms_gateup_silu_q8", ffnFusedWorker);
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
        for (int layerIndex = 0; layerIndex < phi3Config.numberOfLayers(); layerIndex++) {
            TaskGraph ffnLayer = setupSinglePhi3Q8_0FFNLayer((Phi3TornadoWeights) weights, layerIndex);
            if (layerIndex == phi3Config.numberOfLayers() - 1) {
                setupLastID(ffnLayer.getTaskGraphName());
            }
            ffnGraphs.add(ffnLayer.snapshot());
        }
        return ffnGraphs;
    }

    // @formatter:off
    /**
     * Transformer Layer Task Flow (Phi3Q8_0FFNLayers - Fully Optimized)
     *
     * ══════════════════════════════════════════════════════════════════════════════
     *                              ATTENTION BLOCK
     * ══════════════════════════════════════════════════════════════════════════════
     *
     *   wrapX (FP32)
     *      │
     *      ▼
     *  ┌─────────────────┐
     *  │ attn_rms_reduce │──▶ temp (scale factor for RMSNorm)
     *  └────────┬────────┘
     *           │
     *           ▼
     *  ┌─────────────────────────────┐
     *  │ attn_rms_qkv_projection_q8  │──▶ wrapQ, wrapK, wrapV (direct output)
     *  └────────────┬────────────────┘    (fused: RMS apply + Q8 QKV matmul + split)
     *               │
     *               ▼
     *  ┌───────────────────┐   ┌─────────────────────────────────────┐
     *  │ rope_and_kv_cache │───▶│ Q,K rotated + KeyCache, ValueCache │
     *  └─────────┬─────────┘   └─────────────────────────────────────┘
     *            │                (fused: Phi3 RoPE + cache write)
     *            ▼
     *  ┌───────────┐
     *  │ attention │──▶ wrapXb (attention output)
     *  └─────┬─────┘
     *        │
     *        ▼
     *  ┌──────────────────┐
     *  │ attn_output_proj │──▶ wrapX += Wo · wrapXb (residual, Q8 dequant)
     *  └────────┬─────────┘
     *           │
     * ══════════╪═══════════════════════════════════════════════════════════════════
     *           │                    FFN BLOCK
     * ══════════╪═══════════════════════════════════════════════════════════════════
     *           │
     *           ▼
     *  ┌────────────────┐
     *  │ ffn_rms_reduce │──▶ tempFFN (scale factor)
     *  └───────┬────────┘
     *          │
     *          ▼ (optional: NON_NVIDIA only)
     *  ┌──────────────────┐
     *  │ ffn_rms_finalize │──▶ tempFFN (final scale)
     *  └────────┬─────────┘
     *           │
     *           ▼
     *  ┌───────────────────────┐
     *  │ ffn_rms_gateup_silu_q8│──▶ wrapHbU = SiLU(RMSNorm(x)·Wgate) ⊙ (RMSNorm(x)·Wup)
     *  └───────────┬───────────┘    (fused: RMS apply + Q8 gate/up matmul + SiLU + GLU)
     *              │
     *              ▼
     *  ┌───────────────┐
     *  │ ffn_down_proj │──▶ wrapX += wDown · wrapHbU (residual, Q8 dequant)
     *  └───────┬───────┘
     *          │
     *          ▼
     *      wrapX (FP32) ──▶ [next layer or logits]
     *
     * ══════════════════════════════════════════════════════════════════════════════
     *
     * Task Count: 8 tasks (NVIDIA) / 9 tasks (non-NVIDIA)
     * Original:   13 tasks
     * Reduction:  5 tasks eliminated (38% fewer kernel launches)
     *
     * Data Flow Summary:
     *   Input:  wrapX (FP32) - hidden state from previous layer
     *   Output: wrapX (FP32) - updated hidden state with residual connections
     *
     * Key Fusion Points (vs original 13 tasks):
     *   • attn_rms_qkv_projection_q8: Fused RMS apply + Q8 QKV matmul + direct split (3→1 kernel)
     *   • rope_and_kv_cache:          Fused Phi3 RoPE rotation + cache write (2→1 kernel)
     *   • ffn_rms_gateup_silu_q8:     Fused RMS apply + Q8 gate/up matmul + SiLU + GLU (3→1 kernel)
     *
     * Q8_0 Quantization Details:
     *   • Block size: 32 elements per quantization block
     *   • Block format: 2-byte FP16 scale + 32 signed int8 quantized values (34 bytes total)
     *   • Dequantization: value = quant * scale (inline during matmul)
     *   • Memory savings: ~2x compression vs FP16, ~4x vs FP32
     *
     * Phi3-Specific:
     *   • Combined wqkv: Single Q8 [opSize × dim] matrix for Q+K+V projection
     *   • Direct QKV output: No intermediate buffer, routes by row index
     *   • Phi3 RoPE: Uses headSize/2 offset pattern (different from Llama/Qwen)
     *   • Combined wUp: Single Q8 [2×hiddenDim × dim] matrix for gate+up
     *   • Inline SiLU+GLU: No intermediate wrapHb buffer needed
     *
     * Buffers Eliminated by Fusion:
     *   • wrapXb (attention path): Not needed between RMS and QKV projection
     *   • wrapQkv: Direct output to Q/K/V buffers
     *   • wrapHb: Not needed between gate/up matmul and SiLU
     *   • wrapHbG: Gate output merged into final computation
     *
     */
    TaskGraph setupSinglePhi3Q8_0FFNLayer(Phi3TornadoWeights weights, int layerIndex) {
        var taskGraphName = "layer_" + layerIndex;
        var unifiedLayer = new TaskGraph(taskGraphName);
        unifiedLayer.consumeFromDevice(phi3State.wrapX);
        unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                // Copy-in quantized weights per layer (Q8_0 format: ByteArray)
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqkvLayered[layerIndex].asByteArray(),
                weights.woLayered[layerIndex].asByteArray(),
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.wUpLayered[layerIndex].asByteArray(),
                weights.wDownLayered[layerIndex].asByteArray()
        );
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);

        // ═══════════════════════════════════════════════════════════════════════
        //                           ATTENTION BLOCK
        // ═══════════════════════════════════════════════════════════════════════

        // RMS Normalization - compute scale factor
        unifiedLayer.task("attn_rms_reduce",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                context,
                phi3State.temp,               // output: scale factor
                phi3State.wrapX,              // input: hidden state
                phi3Config.dim(),             // dimension
                phi3Config.rmsNormEps(),      // epsilon
                phi3State.localSize);         // local memory size

        if (shouldUseFinalNormalization()) {
            unifiedLayer.task("attn_rms_finalize",
                TransformerComputeKernelsLayered::reductionFinalNormalization,
                context,
                state.temp,
                config.dim(),
                config.rmsNormEps());
        }

        // Fused: RMS apply + Q8 QKV matmul + direct Q/K/V split
        unifiedLayer.task("attn_rms_qkv_projection_q8",
                TransformerComputeKernelsLayered::fusedRmsNormQKVMatmulQ8,
                context,
                phi3State.wrapX,              // input: hidden state
                phi3State.wrapQ,              // output Q [dim]
                phi3State.wrapK,              // output K [kvDim]
                phi3State.wrapV,              // output V [kvDim]
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),  // RMS weights
                phi3State.temp,               // RMS scale (precomputed)
                weights.wqkvLayered[layerIndex].asByteArray(),  // Q8 combined QKV [opSize × dim]
                phi3Config.dim(),             // input dim
                phi3Config.kvDim(),           // K/V output dim
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Fused Phi3 RoPE Rotation + KV Cache Write
        unifiedLayer.task("rope_and_kv_cache", Phi3Kernels::ropeRotationWithCacheCopyPhi3,
                context,
                phi3State.positionHolder,     // current position
                phi3State.wrapQ,              // Q vectors (in/out, rotated)
                phi3State.wrapK,              // K vectors (in/out, rotated)
                phi3State.wrapV,              // V vectors (in only)
                phi3State.wrapKeyCache,       // key cache (out)
                phi3State.wrapValueCache,     // value cache (out)
                phi3Config.numberOfKeyValueHeads(),  // nHeadKv
                phi3Config.headSize(),        // head dimension
                phi3Config.kvDim(),           // kvDim
                layerIndex,                   // layer index for cache offset
                phi3Config.contextLength());  // max sequence length

        // Flash Attention
        unifiedLayer.task("attention",
                TransformerComputeKernelsLayered::processHeadsFlashAttention,
                context,
                phi3State.wrapQ,              // query vectors
                phi3State.wrapKeyCache,       // key cache
                phi3State.wrapValueCache,     // value cache
                phi3State.wrapXb,             // output: attention result
                phi3Config.numberOfHeads(),   // nHeads
                phi3Config.headSize(),        // headSize
                phi3Config.kvDim(),           // kvDim
                phi3Config.kvMul(),           // kvMul (nHeads / nHeadKv)
                phi3State.positionHolder,     // position
                layerIndex,                   // layer index
                phi3Config.contextLength());  // context length

        // Output Projection with Residual (Q8 dequantization)
        unifiedLayer.task("attn_output_proj",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                context,
                phi3State.wrapXb,             // input: attention output
                phi3State.wrapX,              // output: wrapX += Wo · wrapXb
                weights.woLayered[layerIndex].asByteArray(),  // Q8 Wo [dim × dim]
                phi3Config.dim(),             // input dim
                phi3Config.dim(),             // output dim
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // ═══════════════════════════════════════════════════════════════════════
        //                              FFN BLOCK
        // ═══════════════════════════════════════════════════════════════════════

        // RMS Normalization - compute scale factor
        unifiedLayer.task("ffn_rms_reduce",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                context,
                phi3State.tempFFN,            // output: scale factor
                phi3State.wrapX,              // input: hidden state
                phi3Config.dim(),             // dimension
                phi3Config.rmsNormEps(),      // epsilon
                phi3State.localSize);         // local memory size

        // Final normalization (non-NVIDIA only)
        if (shouldUseFinalNormalization()) {
            unifiedLayer.task("ffn_rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    phi3State.tempFFN,        // scale factor (in/out)
                    phi3Config.dim(),         // dimension
                    phi3Config.rmsNormEps()); // epsilon
        }

        // Fused: RMS apply + Q8 gate/up matmul + SiLU activation + GLU
        unifiedLayer.task("ffn_rms_gateup_silu_q8",
                TransformerComputeKernelsLayered::fusedRmsNormFFNGateUpSiLUQ8,
                context,
                phi3State.wrapX,              // input: hidden state
                phi3State.wrapHbU,            // output: SiLU(gate) ⊙ up [hiddenDim]
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),  // RMS weights
                phi3State.tempFFN,            // RMS scale (precomputed)
                weights.wUpLayered[layerIndex].asByteArray(),  // Q8 combined gate+up [2×hiddenDim × dim]
                phi3Config.dim(),             // input dim
                phi3Config.hiddenDim(),       // output dim
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Down Projection with Residual (Q8 dequantization)
        unifiedLayer.task("ffn_down_proj",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                context,
                phi3State.wrapHbU,            // input: FFN intermediate
                phi3State.wrapX,              // output: wrapX += wDown · wrapHbU
                weights.wDownLayered[layerIndex].asByteArray(),  // Q8 wDown [dim × hiddenDim]
                phi3Config.hiddenDim(),       // input dim
                phi3Config.dim(),             // output dim
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        unifiedLayer.persistOnDevice(phi3State.wrapX);
        return unifiedLayer;
    }
    // @formatter:on

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
    // @formatter:on

}
