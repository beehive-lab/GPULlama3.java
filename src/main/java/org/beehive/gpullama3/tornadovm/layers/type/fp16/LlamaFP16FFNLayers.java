package org.beehive.gpullama3.tornadovm.layers.type.fp16;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernels;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernelsLayered;
import org.beehive.gpullama3.tornadovm.layerplanner.WorkerGridFactory;
import org.beehive.gpullama3.tornadovm.layerplanner.strategy.SchedulerType;
import org.beehive.gpullama3.tornadovm.layers.AbstractFFNLayers;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

public class LlamaFP16FFNLayers extends AbstractFFNLayers<LlamaTornadoWeights, LlamaConfiguration> {

    public LlamaFP16FFNLayers(String taskGraph, State state, LlamaTornadoWeights weights, LlamaConfiguration config, SchedulerType schedulerType) {
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
    /**
     * Transformer Layer Task Flow (LlamaFP16FFNLayers)
     *
     * ══════════════════════════════════════════════════════════════════════════════
     *                              ATTENTION BLOCK
     * ══════════════════════════════════════════════════════════════════════════════
     *
     *   wrapX (FP32)
     *      │
     *      ▼
     *  ┌─────────────────┐
     *  │ attn_rms_reduce │──▶ temp (partial sums)
     *  └────────┬────────┘
     *           │
     *           ▼ (optional: NON_NVIDIA only)
     *  ┌──────────────────┐
     *  │ attn_rms_finalize│──▶ temp (final scale)
     *  └────────┬─────────┘
     *           │
     *           ▼
     *  ┌─────────────────────┐
     *  │ attn_rms_apply_fp16 │──▶ wrapXbFP16 (normalized, FP16)
     *  └──────────┬──────────┘
     *             │
     *             ▼
     *  ┌────────────────┐      ┌─────────────────────────────┐
     *  │ qkv_projection │──────▶│ wrapQ, wrapK, wrapV (FP32) │
     *  └───────┬────────┘      └─────────────────────────────┘
     *          │
     *          ▼
     *  ┌───────────────────┐   ┌─────────────────────────────────────┐
     *  │ rope_and_kv_cache │───▶│ Q,K rotated + KeyCache, ValueCache │
     *  └─────────┬─────────┘   └─────────────────────────────────────┘
     *            │
     *            ▼
     *  ┌───────────┐
     *  │ attention │──▶ wrapXb (attention output)
     *  └─────┬─────┘
     *        │
     *        ▼
     *  ┌──────────────────┐
     *  │ attn_output_proj │──▶ wrapX += Wo · wrapXb (residual connection)
     *  └────────┬─────────┘
     *           │
     * ══════════╪═══════════════════════════════════════════════════════════════════
     *           │                    FFN BLOCK
     * ══════════╪═══════════════════════════════════════════════════════════════════
     *           │
     *           ▼
     *  ┌────────────────┐
     *  │ ffn_rms_reduce │──▶ tempFFN (partial sums)
     *  └───────┬────────┘
     *          │
     *          ▼ (optional: NON_NVIDIA only)
     *  ┌─────────────────┐
     *  │ ffn_rms_finalize│──▶ tempFFN (final scale)
     *  └────────┬────────┘
     *           │
     *           ▼
     *  ┌─────────────────┐
     *  │ rms_ffn_gate_up │──▶ wrapHb = SiLU(RMSNorm(x)·W1) ⊙ (RMSNorm(x)·W3)
     *  └────────┬────────┘    (fused: RMS apply + W1/W3 matmuls + SiLU + GLU)
     *           │
     *           ▼
     *  ┌──────────────┐
     *  │ ffn_down_proj│──▶ wrapX += W2 · wrapHb (residual connection)
     *  └──────┬───────┘
     *         │
     *         ▼
     *     wrapX (FP32) ──▶ [next layer or logits]
     *
     * ══════════════════════════════════════════════════════════════════════════════
     *
     * Task Count: 9 tasks (7 if NVIDIA, skipping rms_finalize steps)
     *
     * Data Flow Summary:
     *   Input:  wrapX (FP32) - hidden state from previous layer
     *   Output: wrapX (FP32) - updated hidden state with residual connections
     *
     * Key Fusion Points:
     *   • qkv_projection:   Fused Q/K/V matmuls (3→1 kernel)
     *   • rope_and_kv_cache: Fused RoPE rotation + cache write (2→1 kernel)
     *   • rms_ffn_gate_up:  Fused RMS apply + W1/W3 matmuls + SiLU + GLU (4→1 kernel)
     *
     */
    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        var layerTaskGraphName = "layer_" + layerIndex;
        TaskGraph unifiedLayer = new TaskGraph(layerTaskGraphName);

        // === Data Setup ===
        // consumeFromDevice for wrapX: the no-arg form uses the current graph's own name as the
        // source key, which works in CUDA-graph mode (pointers are frozen) but fails in interpreter
        // mode (updatePersistedObjectState looks up the predecessor's name, not the current name).
        // Subclasses that receive wrapX across a graph boundary override predecessorGraphName() to
        // return the correct predecessor graph name so the XPUBuffer is propagated in both modes.
        String wrapXSrc = predecessorGraphName(layerIndex);
        if (wrapXSrc != null) {
            unifiedLayer.consumeFromDevice(wrapXSrc, state.wrapX);
        } else {
            unifiedLayer.consumeFromDevice(state.wrapX);
        }
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
        // RMS Normalization
        unifiedLayer.task("attn_rms_reduce",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayerFuseFP16,
                context, state.wrapXbFP16, state.wrapX, weights.rms_att_weightLayered[layerIndex].asFloatArray(), state.temp,
                config.dim(), config.rmsNormEps(), state.localSize);

        // QKV Projection (fused)
        unifiedLayer.task("qkv_projection",
                TransformerComputeKernelsLayered::fusedQKVMatmulX,
                context,
                state.wrapXbFP16,                                          // input (FP32)
                state.wrapQ,                                          // output Q
                state.wrapK,                                          // output K
                state.wrapV,                                          // output V
                weights.wqLayered[layerIndex].asHalfFloatArray(),     // Wq
                weights.wkLayered[layerIndex].asHalfFloatArray(),     // Wk
                weights.wvLayered[layerIndex].asHalfFloatArray(),     // Wv
                config.dim(),                                         // dim
                config.kvDim(),                                       // kvDim
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // RoPE + KV Cache
        unifiedLayer.task("rope_and_kv_cache",
                TransformerComputeKernelsLayered::ropeRotationWithCacheCopy,
                context,
                state.positionHolder,
                state.wrapQ,                 // Q (in/out)
                state.wrapK,                 // K (in/out)
                state.wrapV,                 // V (in only)
                state.wrapKeyCache,          // Key cache (out)
                state.wrapValueCache,        // Value cache (out)
                config.kvDim(),
                config.headSize(),
                layerIndex,
                config.contextLength());
        // Attention
        configureAttention(unifiedLayer, layerIndex);
        // Output Projection (Wo) with residual
        unifiedLayer.task("attn_output_proj",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                context, state.wrapXb, state.wrapX,
                weights.woLayered[layerIndex].asHalfFloatArray(),
                config.dim(), config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC);

        // === FFN Block ===
        // RMS Normalization
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
                state.wrapX,                                              // raw input (FP32)
                state.wrapHb,                                             // output
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(), // RMS weights
                state.tempFFN,                                            // RMS scale factor
                weights.w1Layered[layerIndex].asHalfFloatArray(),         // W1
                weights.w3Layered[layerIndex].asHalfFloatArray(),         // W3
                config.dim(),                                             // input dimension
                config.hiddenDim(),                                       // output dimension
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Down projection (W2) with residual
        unifiedLayer.task("ffn_down_proj",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                context, state.wrapHb, state.wrapX,
                weights.w2Layered[layerIndex].asHalfFloatArray(),
                config.hiddenDim(), config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC);

        unifiedLayer.persistOnDevice(state.wrapX, state.wrapKeyCache,
                state.wrapValueCache);

        return unifiedLayer;
    }

    /**
     * Returns the name of the predecessor task graph from which {@code wrapX} should be consumed,
     * or {@code null} to fall back to the no-arg form (source key = own graph name).
     *
     * <p>The no-arg form is safe in CUDA-graph mode (device pointers are frozen at capture time)
     * but fails in interpreter mode: {@code updatePersistedObjectState} looks up the predecessor's
     * graph name, not the current graph's name, so the XPUBuffer is never propagated and
     * {@code executeAlloc} NPEs on a null buffer.</p>
     *
     * <p>Override in subclasses that receive {@code wrapX} from a named predecessor graph:</p>
     * <ul>
     *   <li>layer 0: return the activation graph name (e.g. {@code "activationUpdate"})</li>
     *   <li>layer k &gt; 0: return {@code "layer_" + (k-1)}</li>
     * </ul>
     */
    protected String predecessorGraphName(int layerIndex) {
        return (layerIndex == 0) ? "activationUpdate" : "layer_" + (layerIndex - 1);
    }

    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        if (layerIndex == 0) {
            // First layer: Transfer initial data to device (one-time transfer)
            unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                    state.positionHolder,
                    state.temp, state.tempFFN
            );
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    // Kernel context
                    context,
                    // Intermediate buffers
                    state.wrapXb, state.wrapXb2,
                    // QKV vectors
                    state.wrapQ, state.wrapK, state.wrapV,
                    // KV cache
                    state.wrapKeyCache, state.wrapValueCache,
                    // Attention & FFN buffers
                    state.wrapAtt, state.wrapHb, state.wrapXbFP16);
        } else {
            // Subsequent layers: consume from the previous layer graph by name.
            // The no-arg consumeFromDevice form uses the current graph's own name as source key,
            // which never matches the predecessor in interpreter mode (no CUDA graphs).
            String pred = "layer_" + (layerIndex - 1);
            unifiedLayer.consumeFromDevice(pred,
                    // Kernel context
                    context,
                    // Intermediate buffers
                    state.wrapXb, state.wrapXb2,
                    // QKV vectors
                    state.wrapQ, state.wrapK, state.wrapV,
                    // KV cache
                    state.wrapKeyCache, state.wrapValueCache,
                    // Attention & FFN buffers
                    state.wrapAtt, state.wrapHb,
                    // Position & misc
                    state.positionHolder, state.wrapXbFP16);
        }
        return unifiedLayer;
    }

    private TaskGraph configureAttention(TaskGraph unifiedLayer, int layerIndex) {
        if (schedulerType == SchedulerType.NVIDIA) {
            // Flash Attention (optimized for NVIDIA GPUs)
            return unifiedLayer.task("attention",
                    TransformerComputeKernelsLayered::processHeadsFlashAttention,
                    context,
                    state.wrapQ,              // Query
                    state.wrapKeyCache,       // Key cache
                    state.wrapValueCache,     // Value cache
                    state.wrapXb,             // Output
                    config.numberOfHeads(),
                    config.headSize(),
                    config.kvDim(),
                    config.kvMul(),
                    state.positionHolder,
                    layerIndex,
                    config.contextLength());
        } else {
            // Standard parallel attention (for non-NVIDIA backends)
            return unifiedLayer.task("attention",
                    TransformerComputeKernelsLayered::processHeadsParallel,
                    state.wrapQ,              // Query
                    state.wrapKeyCache,       // Key cache
                    state.wrapValueCache,     // Value cache
                    state.wrapXb,             // Output
                    config.numberOfHeads(),
                    config.headSize(),
                    config.kvDim(),
                    config.kvMul(),
                    config.contextLength(),   // seqLen parameter
                    state.positionHolder,
                    state.wrapAtt,            // Attention weights buffer
                    layerIndex,
                    config.contextLength());
        }
    }
    // @formatter:on

}
