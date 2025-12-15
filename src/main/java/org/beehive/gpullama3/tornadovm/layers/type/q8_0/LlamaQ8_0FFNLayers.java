package org.beehive.gpullama3.tornadovm.layers.type.q8_0;

import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernelsLayered;
import org.beehive.gpullama3.tornadovm.layerplanner.WorkerGridFactory;
import org.beehive.gpullama3.tornadovm.layerplanner.strategy.SchedulerType;
import org.beehive.gpullama3.tornadovm.layers.AbstractFFNLayers;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.List;
import java.util.stream.IntStream;

public class LlamaQ8_0FFNLayers extends AbstractFFNLayers {

    GridScheduler scheduler;
    List<ImmutableTaskGraph> ffnLayerTaskGraphs;

    public LlamaQ8_0FFNLayers(String taskGraphName, LlamaState state, LlamaTornadoWeights weights, Configuration config, SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config, schedulerType);
        ffnLayerTaskGraphs = setupFFNLayered();
    }

    @Override
    public GridScheduler getGridScheduler() {
        return scheduler;
    }

    @Override
    public TaskGraph getTaskGraph() {
        return null;
    }

    @Override
    public ImmutableTaskGraph getImmutableTaskGraph() {
        return null;
    }

    List<ImmutableTaskGraph> setupFFNLayered() {
        return IntStream.range(0, config.numberOfLayers()).mapToObj(i -> {
            var ffnLayer = setupSingleFFNLayer((LlamaTornadoWeights) weights, config, i);
            if (i == config.numberOfLayers() - 1) {
                setupLastID(ffnLayer.getTaskGraphName());
            }
            return ffnLayer.snapshot();
        }).toList();
    }

    // @formatter:off
    /**
     * Transformer Layer Task Flow (LlamaQ8FFNLayers)
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
     *  ┌────────────────┐
     *  │ attn_rms_apply │──▶ wrapXb (normalized, FP32)
     *  └───────┬────────┘
     *          │
     *          ▼
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
     *  └────────┬────────┘    (fully fused: RMS reduce/apply + W1/W3 matmuls + SiLU + GLU)
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
     *   • qkv_projection:   Fused Q/K/V matmuls with Q8 dequantization (3→1 kernel)
     *   • rope_and_kv_cache: Fused RoPE rotation + cache write (2→1 kernel)
     *   • rms_ffn_gate_up:  Fully fused RMS norm + W1/W3 matmuls + SiLU + GLU (5→1 kernel)
     *
     * Quantization: Q8_0 format (8-bit weights with block-wise scaling)
     *
     */
    TaskGraph setupSingleFFNLayer(LlamaTornadoWeights weights, Configuration config, int layerIndex) {
        var layerTaskGraphName = "layer_" + layerIndex;
        TaskGraph unifiedLayer = new TaskGraph(layerTaskGraphName);

        // === Data Setup ===
        unifiedLayer.consumeFromDevice(state.wrapX);
        unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                // Copy-in weights per layer for batched-layered layout (Q8 format)
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

        // === Attention Block ===
        // RMS Normalization
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

        // QKV Projection (fused with Q8 dequantization)
        unifiedLayer.task("qkv_projection",
                TransformerComputeKernelsLayered::fusedQKVMatmulQ8,
                context,
                state.wrapXb,                                         // input (FP32)
                state.wrapQ,                                          // output Q
                state.wrapK,                                          // output K
                state.wrapV,                                          // output V
                weights.wqLayered[layerIndex].asByteArray(),          // Wq (Q8)
                weights.wkLayered[layerIndex].asByteArray(),          // Wk (Q8)
                weights.wvLayered[layerIndex].asByteArray(),          // Wv (Q8)
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

        // Output Projection (Wo) with residual (Q8 dequantization)
        unifiedLayer.task("attn_output_proj",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                context, state.wrapXb, state.wrapX,
                weights.woLayered[layerIndex].asByteArray(),
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

        // Fully fused: RMS apply + Gate/Up projections + SiLU + GLU (Q8 dequantization)
        unifiedLayer.task("rms_ffn_gate_up",
                TransformerComputeKernelsLayered::fullyFusedRmsNormFFNGateUpQ8,
                context,
                state.wrapX,                                              // raw input (FP32)
                state.wrapHb,                                             // output
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(), // RMS weights
                weights.w1Layered[layerIndex].asByteArray(),              // W1 (Q8)
                weights.w3Layered[layerIndex].asByteArray(),              // W3 (Q8)
                config.dim(),                                             // input dimension
                config.hiddenDim(),                                       // output dimension
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Down projection (W2) with residual (Q8 dequantization)
        unifiedLayer.task("ffn_down_proj",
                TransformerComputeKernelsLayered::matrixVectorGenericWithResidualQ8_0Byte,
                context, state.wrapHb, state.wrapX,
                weights.w2Layered[layerIndex].asByteArray(),
                config.hiddenDim(), config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Keep activation X on device for next layer
        unifiedLayer.persistOnDevice(state.wrapX);

        return unifiedLayer;
    }

    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        // First layer: Transfer initial data to device (one-time transfer)
        if (layerIndex == 0) {
            // Transfer all attention-related data: query, key, value matrices and their caches
            unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                    state.positionHolder,
                    state.temp, state.tempFFN); //
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, //
                    context,
                    state.wrapXb, state.wrapXb2, //
                    state.wrapQ, state.wrapK, state.wrapV, //
                    state.wrapKeyCache, state.wrapValueCache, //
                    state.wrapAtt, state.wrapHb); //
        } else {
            // Subsequent layers: Consume data already on device from previous layer
            unifiedLayer.consumeFromDevice(
                    context,
                    state.wrapXb, state.wrapXb2, //
                    state.wrapQ, state.wrapK, state.wrapV, //
                    state.wrapKeyCache, state.wrapValueCache, //
                    state.wrapAtt, state.wrapHb, //
                    state.positionHolder //
            );
        }
        return unifiedLayer;
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        // === Worker Grid Definitions ===
        WorkerGrid rmsNormWorker = WorkerGridFactory.createRmsNormWorker(config.dim(), 256);

        int configDimRowMajorGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configDimRowMajorGlobalWorker = WorkerGridFactory.genericWorker(configDimRowMajorGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        int configHiddenDimRowMajor = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid configHiddenDimRowMajorWorker = WorkerGridFactory.genericWorker(configHiddenDimRowMajor, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Fused QKV: dim rows for Q + kvDim rows for K + kvDim rows for V
        int fusedQkvGlobal = (config.dim() + 2 * config.kvDim()) * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedQkvWorker = WorkerGridFactory.genericWorker(fusedQkvGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        WorkerGrid ropeWithCacheWorker = WorkerGridFactory.genericWorker(config.dim() / 2, 512);

        WorkerGrid parallelAttentionWorker = WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), config.headSize());

        // === Per-Layer Grid Assignments (ordered by task graph flow) ===
        for (int i = 0; i < config.numberOfLayers(); i++) {
            // --- Attention Block ---
            // RMS Normalization
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".attn_rms_reduce", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".attn_rms_apply", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".qkv_projection", fusedQkvWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rope_and_kv_cache", ropeWithCacheWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".attention", parallelAttentionWorker);
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".attn_output_proj", configDimRowMajorGlobalWorker);
            // --- FFN Block ---
            // RMS Normalization
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".ffn_rms_reduce", rmsNormWorker);
            // Fused RMS + Gate/Up Projections
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rms_ffn_gate_up", configHiddenDimRowMajorWorker);
            // Down Projection
            tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".ffn_down_proj", configDimRowMajorGlobalWorker);
        }

        return tornadoForwardScheduler;
    }

    public List<ImmutableTaskGraph> getFfnLayerTaskGraphs() {
        return ffnLayerTaskGraphs;
    }

    private TaskGraph configureAttention(TaskGraph unifiedLayer, int layerIndex) {
        if (schedulerType == SchedulerType.NVIDIA) {
            return unifiedLayer.task("attention",
                TransformerComputeKernelsLayered::processHeadsFlashAttention,
                context,
                state.wrapQ, state.wrapKeyCache,
                state.wrapValueCache, state.wrapXb,
                config.numberOfHeads(), config.headSize(),
                config.kvDim(), config.kvMul(),
                state.positionHolder, layerIndex,
                config.contextLength());
        } else {
            return unifiedLayer.task("attention",
                TransformerComputeKernelsLayered::processHeadsParallel,
                state.wrapQ, state.wrapKeyCache,
                state.wrapValueCache, state.wrapXb,
                config.numberOfHeads(), config.headSize(),
                config.kvDim(), config.kvMul(), config.contextLength(),
                state.positionHolder, state.wrapAtt, layerIndex,
                config.contextLength());
        }
    }
    // @formatter:on
}
