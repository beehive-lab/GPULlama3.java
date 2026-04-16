package org.beehive.gpullama3.tornadovm.layers.type.q8_0;

import org.beehive.gpullama3.inference.state.Gemma4State;
import org.beehive.gpullama3.inference.weights.tornado.Gemma4TornadoWeights;
import org.beehive.gpullama3.model.gemma4.Gemma4Configuration;
import org.beehive.gpullama3.tornadovm.kernels.Gemma4Kernels;
import org.beehive.gpullama3.tornadovm.kernels.Qwen3Kernels;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernelsLayered;
import org.beehive.gpullama3.tornadovm.layerplanner.WorkerGridFactory;
import org.beehive.gpullama3.tornadovm.layerplanner.strategy.SchedulerType;
import org.beehive.gpullama3.tornadovm.layers.AbstractFFNLayers;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Gemma4 Q8_0-quantized FFN layers with per-layer dimensions, GELU activation,
 * post-norms, dual RoPE, SWA-aware caching, and layer output scaling.
 */
public class Gemma4Q8_0FFNLayers extends AbstractFFNLayers<Gemma4TornadoWeights, Gemma4Configuration> {

    private final Gemma4State gemma4State;

    public Gemma4Q8_0FFNLayers(String taskGraphName, Gemma4State state, Gemma4TornadoWeights weights,
                                Gemma4Configuration config, SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config, schedulerType);
        this.gemma4State = state;
        setupFFNLayers();
    }

    // @formatter:off
    @Override
    public GridScheduler updateGridScheduler(GridScheduler gridScheduler) {
        for (int i = 0; i < config.numberOfLayers(); i++) {
            int headSize = config.headSize(i);
            int kvHeads = config.numberOfKeyValueHeads(i);
            int kvDim = config.kvDim(i);
            int queryDim = config.queryDim(i);
            int hiddenDim = config.feedForwardLength()[i];

            // RMS norm worker (always uses full dim)
            WorkerGrid rmsNormWorker = WorkerGridFactory.createRmsNormWorker(config.dim(), state.localSize);

            // Fused QKV projection: Q rows + K rows + V rows
            int fusedQKVRows = queryDim + 2 * kvDim;
            int fusedQKVGlobal = fusedQKVRows * LOCAL_WORK_GROUP_SIZE_ALLOC;
            WorkerGrid fusedQKVWorker = WorkerGridFactory.genericWorker(fusedQKVGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

            // Q/K RMS norm: (nHeads + kvHeads) workgroups, each of headSize threads
            int qkRmsNormGroups = config.numberOfHeads() + kvHeads;
            WorkerGrid qkRmsNormWorker = WorkerGridFactory.genericWorker(qkRmsNormGroups * headSize, headSize);

            // V bare RMS norm: kvHeads workgroups
            int vNormLocalSize = Math.min(headSize, 256);
            WorkerGrid vNormWorker = WorkerGridFactory.genericWorker(kvHeads * vNormLocalSize, vNormLocalSize);

            // RoPE worker
            WorkerGrid ropeWorker = WorkerGridFactory.createRoPEWorker(config.numberOfHeads(), headSize);

            // Attention worker
            WorkerGrid attentionWorker = WorkerGridFactory.createAttentionWorker(config.numberOfHeads(), headSize);

            // Attention output matmul: dim output rows
            int attnOutGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
            WorkerGrid attnOutWorker = WorkerGridFactory.genericWorker(attnOutGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

            // Post-norm residual: dim elements
            WorkerGrid normResidualWorker = WorkerGridFactory.genericWorker(config.dim(), 128);

            // FFN gate/up: hiddenDim rows
            int ffnGateUpGlobal = hiddenDim * LOCAL_WORK_GROUP_SIZE_ALLOC;
            WorkerGrid ffnGateUpWorker = WorkerGridFactory.genericWorker(ffnGateUpGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

            // FFN down matmul: dim output rows
            int ffnDownGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
            WorkerGrid ffnDownWorker = WorkerGridFactory.genericWorker(ffnDownGlobal, LOCAL_WORK_GROUP_SIZE_ALLOC);

            // Layer output scale: dim elements
            WorkerGrid scaleWorker = WorkerGridFactory.genericWorker(config.dim(), 128);

            // === Attention block ===
            gridScheduler.addWorkerGrid("layer_" + i + ".attn_rms_reduce", rmsNormWorker);
            if (shouldUseFinalNormalization()) {
                gridScheduler.addWorkerGrid("layer_" + i + ".attn_rms_finalize", rmsNormWorker);
            }
            gridScheduler.addWorkerGrid("layer_" + i + ".attn_rms_qkv_projection", fusedQKVWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".qk_rmsnorm", qkRmsNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".v_bare_rmsnorm", vNormWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".rope_and_kv_cache", ropeWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".attention", attentionWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".attn_output_matmul", attnOutWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".attn_output_postnorm_reduce", rmsNormWorker);
            if (shouldUseFinalNormalization()) {
                gridScheduler.addWorkerGrid("layer_" + i + ".attn_output_postnorm_finalize", rmsNormWorker);
            }
            gridScheduler.addWorkerGrid("layer_" + i + ".attn_output_postnorm_residual", normResidualWorker);

            // === FFN block ===
            gridScheduler.addWorkerGrid("layer_" + i + ".ffn_rms_reduce", rmsNormWorker);
            if (shouldUseFinalNormalization()) {
                gridScheduler.addWorkerGrid("layer_" + i + ".ffn_rms_finalize", rmsNormWorker);
            }
            gridScheduler.addWorkerGrid("layer_" + i + ".rms_ffn_gate_up", ffnGateUpWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".ffn_down_matmul", ffnDownWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".ffn_down_postnorm_reduce", rmsNormWorker);
            if (shouldUseFinalNormalization()) {
                gridScheduler.addWorkerGrid("layer_" + i + ".ffn_down_postnorm_finalize", rmsNormWorker);
            }
            gridScheduler.addWorkerGrid("layer_" + i + ".ffn_down_postnorm_residual", normResidualWorker);
            gridScheduler.addWorkerGrid("layer_" + i + ".layer_output_scale", scaleWorker);
        }
        return gridScheduler;
    }

    @Override
    protected TaskGraph createFFNLayerTaskGraph(int layerIndex) {
        var taskGraphName = "layer_" + layerIndex;

        // === Per-layer dimension parameters ===
        int headSize = config.headSize(layerIndex);
        int kvHeads = config.numberOfKeyValueHeads(layerIndex);
        int kvDim = config.kvDim(layerIndex);
        int queryDim = config.queryDim(layerIndex);
        int hiddenDim = config.feedForwardLength()[layerIndex];
        int inputDim = config.dim();
        int gqa = config.numberOfHeads() / kvHeads;
        boolean isSWA = config.isSWA()[layerIndex];
        boolean hasKv = config.hasKv(layerIndex);
        int kvSourceLayer = config.kvSourceLayer(layerIndex);
        int cacheOffset = gemma4State.kvCacheLayerOffset[kvSourceLayer];

        // Select RoPE frequencies based on layer type
        var freqReal = isSWA ? weights.freq_cis_real_swa.asFloatArray() : weights.freq_cis_realFlat.asFloatArray();
        var freqImag = isSWA ? weights.freq_cis_imag_swa.asFloatArray() : weights.freq_cis_imagFlat.asFloatArray();

        var unifiedLayer = new TaskGraph(taskGraphName);

        // === Data Setup ===
        unifiedLayer.consumeFromDevice(gemma4State.wrapX);
        unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                // Attention weights
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                weights.woLayered[layerIndex].asByteArray(),
                // Q/K norm weights
                weights.rms_att_QNormLayered[layerIndex].asFloatArray(),
                weights.rms_att_KNormLayered[layerIndex].asFloatArray(),
                // Post-attention norm
                weights.postAttentionNormLayered[layerIndex].asFloatArray(),
                // FFN weights
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                weights.w1Layered[layerIndex].asByteArray(),
                weights.w2Layered[layerIndex].asByteArray(),
                weights.w3Layered[layerIndex].asByteArray(),
                // Post-FFN norm
                weights.postFfwNormLayered[layerIndex].asFloatArray(),
                // RoPE frequencies for this layer type
                freqReal, freqImag);
        unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);

        // ═══════════════════════════════════════════════════════════════
        //                       ATTENTION BLOCK
        // ═══════════════════════════════════════════════════════════════

        // 1. RMS Normalization - compute scale factor for attention input
        unifiedLayer.task("attn_rms_reduce",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                context,
                gemma4State.temp,
                gemma4State.wrapX,
                inputDim,
                config.rmsNormEps(),
                gemma4State.localSize);

        if (shouldUseFinalNormalization()) {
            unifiedLayer.task("attn_rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    gemma4State.temp,
                    inputDim,
                    config.rmsNormEps());
        }

        // 2. Fused RMS Apply + QKV Projection (Q8_0)
        unifiedLayer.task("attn_rms_qkv_projection",
                Qwen3Kernels::fusedRmsNormQKVMatmulQ8_0,
                context,
                gemma4State.wrapX,
                gemma4State.wrapQ,
                gemma4State.wrapK,
                gemma4State.wrapV,
                weights.rms_att_weightLayered[layerIndex].asFloatArray(),
                gemma4State.temp,
                weights.wqLayered[layerIndex].asByteArray(),
                weights.wkLayered[layerIndex].asByteArray(),
                weights.wvLayered[layerIndex].asByteArray(),
                inputDim,
                queryDim,
                kvDim,
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // 3. Fused Q/K RMSNorm (per-head normalization with learned weights)
        unifiedLayer.task("qk_rmsnorm",
                Qwen3Kernels::fusedQKRmsNorm,
                context,
                gemma4State.wrapQ,
                gemma4State.wrapK,
                weights.rms_att_QNormLayered[layerIndex].asFloatArray(),
                weights.rms_att_KNormLayered[layerIndex].asFloatArray(),
                config.numberOfHeads(),
                kvHeads,
                headSize,
                headSize,
                config.rmsNormEps());

        // 4. V bare RMS norm (per-head, no learned weights - Gemma4 specific)
        unifiedLayer.task("v_bare_rmsnorm",
                Gemma4Kernels::bareRmsNormPerHead,
                context,
                gemma4State.wrapV,
                kvHeads,
                headSize,
                config.rmsNormEps());

        // 5. Fused RoPE Rotation + KV Cache Write
        //    writeCache=0 for shared KV layers to avoid overwriting source layer's cache
        //    flags: bit0=isSWA, bit1=writeCache
        int ropeFlags = (isSWA ? 1 : 0) | (hasKv ? 2 : 0);
        unifiedLayer.task("rope_and_kv_cache",
                Gemma4Kernels::ropeRotationWithCacheCopyGemma4,
                context,
                gemma4State.positionHolder,
                gemma4State.wrapQ,
                gemma4State.wrapK,
                gemma4State.wrapV,
                gemma4State.wrapKeyCache,
                gemma4State.wrapValueCache,
                freqReal, freqImag,
                kvHeads,
                headSize,
                kvDim,
                cacheOffset,
                config.slidingWindow(),
                ropeFlags);

        // 6. Flash Attention (no scaling, SWA-aware)
        unifiedLayer.task("attention",
                Gemma4Kernels::processHeadsFlashAttentionGemma4,
                context,
                gemma4State.wrapQ,
                gemma4State.wrapKeyCache,
                gemma4State.wrapValueCache,
                gemma4State.wrapXb,
                config.numberOfHeads(),
                headSize,
                kvDim,
                gqa,
                gemma4State.positionHolder,
                cacheOffset,
                config.contextLength(),
                isSWA ? 1 : 0,
                config.slidingWindow());

        // 7. Attention Output: matmul → RMS norm → residual add
        //    Step 7a: Wo matmul (write to wrapXb2)
        unifiedLayer.task("attn_output_matmul",
                Gemma4Kernels::matrixVectorWriteQ8_0,
                context,
                gemma4State.wrapXb,
                gemma4State.wrapXb2,
                weights.woLayered[layerIndex].asByteArray(),
                queryDim,
                inputDim,
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        //    Step 7b: RMS reduction on matmul output (reuse temp - attn input RMS is consumed)
        unifiedLayer.task("attn_output_postnorm_reduce",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                context,
                gemma4State.temp,
                gemma4State.wrapXb2,
                inputDim,
                config.rmsNormEps(),
                gemma4State.localSize);

        if (shouldUseFinalNormalization()) {
            unifiedLayer.task("attn_output_postnorm_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    gemma4State.temp,
                    inputDim,
                    config.rmsNormEps());
        }

        //    Step 7c: Apply post-attention norm weights + residual
        unifiedLayer.task("attn_output_postnorm_residual",
                Gemma4Kernels::rmsNormWeightedResidual,
                gemma4State.wrapX,
                gemma4State.wrapXb2,
                weights.postAttentionNormLayered[layerIndex].asFloatArray(),
                gemma4State.temp,
                inputDim);

        // ═══════════════════════════════════════════════════════════════
        //                          FFN BLOCK
        // ═══════════════════════════════════════════════════════════════

        // 8. RMS Normalization - compute scale factor for FFN input
        unifiedLayer.task("ffn_rms_reduce",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                context,
                gemma4State.tempFFN,
                gemma4State.wrapX,
                inputDim,
                config.rmsNormEps(),
                gemma4State.localSize);

        if (shouldUseFinalNormalization()) {
            unifiedLayer.task("ffn_rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    gemma4State.tempFFN,
                    inputDim,
                    config.rmsNormEps());
        }

        // 9. Fused RMS Apply + Gate/Up Projection + GELU + GLU
        unifiedLayer.task("rms_ffn_gate_up",
                Gemma4Kernels::fusedRmsNormFFNGateUpGeluQ8_0,
                context,
                gemma4State.wrapX,
                gemma4State.wrapHb,
                weights.rms_ffn_weightLayered[layerIndex].asFloatArray(),
                gemma4State.tempFFN,
                weights.w1Layered[layerIndex].asByteArray(),
                weights.w3Layered[layerIndex].asByteArray(),
                inputDim,
                hiddenDim,
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        // 10. FFN Down: matmul → RMS norm → residual add
        //     Step 10a: W2 matmul (write to wrapXb)
        unifiedLayer.task("ffn_down_matmul",
                Gemma4Kernels::matrixVectorWriteQ8_0,
                context,
                gemma4State.wrapHb,
                gemma4State.wrapXb,
                weights.w2Layered[layerIndex].asByteArray(),
                hiddenDim,
                inputDim,
                LOCAL_WORK_GROUP_SIZE_ALLOC);

        //     Step 10b: RMS reduction on matmul output (reuse tempFFN - FFN input RMS is consumed)
        unifiedLayer.task("ffn_down_postnorm_reduce",
                TransformerComputeKernelsLayered::reductionOneBlockWithLayer,
                context,
                gemma4State.tempFFN,
                gemma4State.wrapXb,
                inputDim,
                config.rmsNormEps(),
                gemma4State.localSize);

        if (shouldUseFinalNormalization()) {
            unifiedLayer.task("ffn_down_postnorm_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    gemma4State.tempFFN,
                    inputDim,
                    config.rmsNormEps());
        }

        //     Step 10c: Apply post-FFN norm weights + residual
        unifiedLayer.task("ffn_down_postnorm_residual",
                Gemma4Kernels::rmsNormWeightedResidual,
                gemma4State.wrapX,
                gemma4State.wrapXb,
                weights.postFfwNormLayered[layerIndex].asFloatArray(),
                gemma4State.tempFFN,
                inputDim);

        // 11. Layer Output Scale
        unifiedLayer.task("layer_output_scale",
                Gemma4Kernels::applyLayerOutputScale,
                gemma4State.wrapX,
                weights.layerOutputScale[layerIndex],
                inputDim);

        unifiedLayer.persistOnDevice(state.wrapX);

        return unifiedLayer;
    }
    // @formatter:on

    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        if (layerIndex == 0) {
            // First layer: transfer temporary buffers every execution
            unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                    gemma4State.positionHolder, gemma4State.temp, gemma4State.tempFFN);

            // First execution: allocate workspace buffers
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    context, gemma4State.wrapXb, gemma4State.wrapXb2,
                    gemma4State.wrapQ, gemma4State.wrapK, gemma4State.wrapV,
                    gemma4State.wrapKeyCache, gemma4State.wrapValueCache,
                    gemma4State.wrapAtt, gemma4State.wrapHb);
        } else {
            // Subsequent layers: consume from previous layer
            unifiedLayer.consumeFromDevice(context, gemma4State.wrapXb, gemma4State.wrapXb2,
                    gemma4State.wrapQ, gemma4State.wrapK, gemma4State.wrapV,
                    gemma4State.wrapKeyCache, gemma4State.wrapValueCache,
                    gemma4State.wrapAtt, gemma4State.wrapHb, gemma4State.positionHolder);
        }
        return unifiedLayer;
    }
}
