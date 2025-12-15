package org.beehive.gpullama3.tornadovm.layers.type.fp16;

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
 * Phi3FP16FFNLayers: FP16 FFN layers for Phi3 with Group Query Attention (GQA) support.
 *
 * Key Differences from Qwen2/Qwen3: - Uses combined QKV matrix (wqkv) instead of separate Q, K, V matrices - Includes splitQKV task to separate combined buffer - Uses ropeRotationPhi3 kernel for
 * position embeddings - FFN uses single wUp matrix that outputs both Gate and Up (2 * hiddenDim) - Includes splitGateUpAndSiLU task for FFN activation - Uses wDown for final FFN projection - No Q, K,
 * V bias terms
 *
 * Works directly with Phi3State to access and mutate Phi3-specific state fields.
 */
public class Phi3FP16FFNLayers extends AbstractFFNLayers {

    // Typed references to Phi3-specific state and config
    private final Phi3State phi3State;
    private final Phi3Configuration phi3Config;
    // Phi3-specific dimension for combined QKV buffer
    private final int opSize;
    TaskGraph ffnLayerTaskGraph;
    GridScheduler scheduler;
    List<ImmutableTaskGraph> ffnLayerTaskGraphs;

    public Phi3FP16FFNLayers(String taskGraphName, Phi3State state, Phi3TornadoWeights weights, Phi3Configuration config, SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config, schedulerType);
        this.phi3State = state;
        this.phi3Config = config;
        this.opSize = config.dim() + 2 * (config.numberOfKeyValueHeads() * config.headSize());
        ffnLayerTaskGraphs = setupFFNLayered();
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler gridScheduler) {
        // RMS norm worker
        WorkerGrid rmsNormWorker = WorkerGridFactory.createRmsNormWorker(config.dim(), state.localSize);

        // Fused RMS + QKV matmul worker
        int fusedQkvGlobal = opSize * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedQkvWorker = WorkerGridFactory.genericWorker(fusedQkvGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Fused RoPE + cache copy worker (Phi3 uses dim/2 pattern)
        WorkerGrid ropeWorker = WorkerGridFactory.genericWorker(config.dim() / 2, 128);

        // Parallel attention worker
        WorkerGrid parallelAttentionWorker = WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), config.headSize());

        // Output projection worker
        int matmul1Global = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid matmul1Worker = WorkerGridFactory.genericWorker(matmul1Global, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Fused RMS + FFN gate/up worker
        int fusedFFNGlobal = (2 * config.hiddenDim()) * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid fusedFFNWorker = WorkerGridFactory.genericWorker(fusedFFNGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

        // FFN down projection worker
        int ffnDownGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
        WorkerGrid ffnDownWorker = WorkerGridFactory.genericWorker(ffnDownGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);
        // Same worker as before - total rows = dim + 2*kvDim = opSize

        for (int i = 0; i < config.numberOfLayers(); i++) {
            // === Attention Block ===
            gridScheduler.addWorkerGrid("layer_" + i + ".attn_rms_reduce", rmsNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".attn_rms_qkv_projection", fusedQkvWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".rope_and_kv_cache", ropeWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".attention", parallelAttentionWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".attn_output_proj", matmul1Worker);
            // === FFN Block ===
            gridScheduler.addWorkerGrid("layer_" + i + ".ffn_rms_reduce", rmsNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".rms_ffn_silu", fusedFFNWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".ffn_down_proj", ffnDownWorker);
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
        for (int layerIndex = 0; layerIndex < phi3Config.numberOfLayers(); layerIndex++) {
            TaskGraph ffnLayer = setupSinglePhi3FFNLayer((Phi3TornadoWeights) weights, layerIndex);
            if (layerIndex == phi3Config.numberOfLayers() - 1) {
                setupLastID(ffnLayer.getTaskGraphName());
            }
            ffnGraphs.add(ffnLayer.snapshot());
        }
        return ffnGraphs;
    }

    // @formatter:off
    /**
     * Transformer Layer Task Flow (Phi3FP16FFNLayers - Fully Optimized)
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
     *  ┌────────────────────────┐
     *  │ attn_rms_qkv_projection│──▶ wrapQ, wrapK, wrapV (direct output)
     *  └───────────┬────────────┘    (fused: RMS apply + QKV matmul + split)
     *              │
     *              ▼
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
     *  │ attn_output_proj │──▶ wrapX += Wo · wrapXb (residual connection)
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
     *  ┌──────────────┐
     *  │ rms_ffn_silu │──▶ wrapHbU = SiLU(RMSNorm(x)·Wgate) ⊙ (RMSNorm(x)·Wup)
     *  └──────┬───────┘    (fused: RMS apply + gate/up matmul + SiLU + GLU)
     *         │
     *         ▼
     *  ┌──────────────┐
     *  │ ffn_down_proj│──▶ wrapX += wDown · wrapHbU (residual connection)
     *  └──────┬───────┘
     *         │
     *         ▼
     *     wrapX (FP32) ──▶ [next layer or logits]
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
     *   • attn_rms_qkv_projection: Fused RMS apply + QKV matmul + direct split (3→1 kernel)
     *   • rope_and_kv_cache:       Fused Phi3 RoPE rotation + cache write (2→1 kernel)
     *   • rms_ffn_silu:            Fused RMS apply + gate/up matmul + SiLU + GLU (3→1 kernel)
     *
     * Phi3-Specific:
     *   • Combined wqkv: Single [opSize × dim] matrix for Q+K+V projection
     *   • Direct QKV output: No intermediate buffer, routes by row index
     *   • Phi3 RoPE: Uses headSize/2 offset pattern (different from Llama/Qwen)
     *   • Combined wUp: Single [2×hiddenDim × dim] matrix for gate+up
     *   • Inline SiLU+GLU: No intermediate wrapHb buffer needed
     *
     */
    TaskGraph setupSinglePhi3FFNLayer(Phi3TornadoWeights weights, int layerIndex) {
        var taskGraphName = "layer_" + layerIndex;
        var unifiedLayer = new TaskGraph(taskGraphName);
        unifiedLayer.consumeFromDevice(phi3State.wrapX);
        unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                // Attention weights
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqkvLayered[layerIndex].asHalfFloatArray(),
                weights.woLayered[layerIndex].asHalfFloatArray(),
                // FFN weights
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.wUpLayered[layerIndex].asHalfFloatArray(),
                weights.wDownLayered[layerIndex].asHalfFloatArray());
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);

        // ═══════════════════════════════════════════════════════════════════════
        //                           ATTENTION BLOCK
        // ═══════════════════════════════════════════════════════════════════════

        // RMS Normalization - compute scale factor
        unifiedLayer.task("attn_rms_reduce", TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                context, phi3State.temp,               // output: scale factor
                phi3State.wrapX,              // input: hidden state
                phi3Config.dim(),             // dimension
                phi3Config.rmsNormEps(),      // epsilon
                phi3State.localSize);         // local memory size

        if (shouldUseFinalNormalization()) {
            unifiedLayer.task("attn_rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context, state.temp, config.dim(), config.rmsNormEps());
        }

        unifiedLayer.task("attn_rms_qkv_projection", Phi3Kernels::fusedRmsNormQKVMatmulDirect,
                context, phi3State.wrapX,              // input
                phi3State.wrapQ,              // output Q
                phi3State.wrapK,              // output K
                phi3State.wrapV,              // output V
                weights.rms_att_weightLayered[layerIndex].asFloatArray(), phi3State.temp,               // RMS scale
                weights.wqkvLayered[layerIndex].asHalfFloatArray(), phi3Config.dim(),             // dim
                phi3Config.kvDim(),           // kvDim
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Fused Phi3 RoPE Rotation + KV Cache Write
        unifiedLayer.task("rope_and_kv_cache", Phi3Kernels::ropeRotationWithCacheCopyPhi3,
                context, phi3State.positionHolder,     // current position
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
        unifiedLayer.task("attention", TransformerComputeKernelsLayered::processHeadsFlashAttention,
                context, phi3State.wrapQ,              // query vectors
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

        // Output Projection with Residual
        unifiedLayer.task("attn_output_proj", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                context, phi3State.wrapXb,             // input: attention output
                phi3State.wrapX,              // output: wrapX += Wo · wrapXb
                weights.woLayered[layerIndex].asHalfFloatArray(),  // Wo [dim × dim]
                phi3Config.dim(),             // input dim
                phi3Config.dim(),             // output dim
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // ═══════════════════════════════════════════════════════════════════════
        //                              FFN BLOCK
        // ═══════════════════════════════════════════════════════════════════════

        // RMS Normalization - compute scale factor
        unifiedLayer.task("ffn_rms_reduce", TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                context, phi3State.tempFFN,            // output: scale factor
                phi3State.wrapX,              // input: hidden state
                phi3Config.dim(),             // dimension
                phi3Config.rmsNormEps(),      // epsilon
                phi3State.localSize);         // local memory size

        // Final normalization (non-NVIDIA only)
        if (shouldUseFinalNormalization()) {
            unifiedLayer.task("ffn_rms_finalize", TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context, phi3State.tempFFN,        // scale factor (in/out)
                    phi3Config.dim(),         // dimension
                    phi3Config.rmsNormEps()); // epsilon
        }

        unifiedLayer.task("rms_ffn_silu", Phi3Kernels::fusedRmsNormFFNGateUpSiLU,
                context, phi3State.wrapX,              // input
                phi3State.wrapHbU,            // output (direct to final FFN buffer)
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(), phi3State.tempFFN,            // RMS scale
                weights.wUpLayered[layerIndex].asHalfFloatArray(), phi3Config.dim(),             // input dim
                phi3Config.hiddenDim(),       // output dim (hiddenDim, not 2×hiddenDim!)
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // Down Projection with Residual
        unifiedLayer.task("ffn_down_proj", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual,
                context, phi3State.wrapHbU,            // input: FFN intermediate
                phi3State.wrapX,              // output: wrapX += wDown · wrapHbU
                weights.wDownLayered[layerIndex].asHalfFloatArray(),  // wDown [dim × hiddenDim]
                phi3Config.hiddenDim(),       // input dim
                phi3Config.dim(),             // output dim
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        unifiedLayer.persistOnDevice(phi3State.wrapX);
        return unifiedLayer;
    }

    /**
     * Configure data transfers for first and subsequent layers
     */
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        if (layerIndex == 0) {
            // First layer: Transfer temporary buffers and state every execution
            unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION, phi3State.positionHolder,
                    phi3State.temp, phi3State.tempFFN);
            // First execution: allocate workspace buffers
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    context, phi3State.wrapXb, phi3State.wrapXb2,
                    phi3State.wrapQ, phi3State.wrapK, phi3State.wrapV,
                    phi3State.wrapKeyCache, phi3State.wrapValueCache,
                    phi3State.wrapAtt, phi3State.wrapHb,  phi3State.wrapHbG,
                    phi3State.wrapHbU, phi3State.wrapQkv);
        } else {
            // Subsequent layers: Consume data from previous layer
            unifiedLayer.consumeFromDevice(context, phi3State.wrapXb, phi3State.wrapXb2,
                    phi3State.wrapQ, phi3State.wrapK, phi3State.wrapV, phi3State.wrapKeyCache,
                    phi3State.wrapValueCache, phi3State.wrapAtt, phi3State.wrapHb, phi3State.positionHolder,
                    phi3State.wrapHbG, phi3State.wrapHbU, phi3State.wrapQkv);
        }
        return unifiedLayer;
    }
    // @formatter:on

}
