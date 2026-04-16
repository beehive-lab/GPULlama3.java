package org.beehive.gpullama3.inference.weights.standard;

import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.tensor.standard.FloatTensor;

import java.lang.foreign.MemorySegment;
import java.nio.FloatBuffer;

/**
 * Gemma 4 specific weights extending StandardWeights with additional tensors
 * for per-head normalization, post-attention/FFN norms, dual RoPE, per-layer
 * embeddings, and MoE support.
 */
public class Gemma4StandardWeights extends StandardWeights {

    // Per-head Q/K normalization (similar to Qwen3)
    public final FloatBuffer[] attnQNorm;          // blk.X.attn_q_norm.weight
    public final FloatBuffer[] attnKNorm;           // blk.X.attn_k_norm.weight

    // Post-attention and post-FFN normalization
    public final FloatBuffer[] postAttentionNorm;   // blk.X.post_attention_norm.weight
    public final FloatBuffer[] postFfwNorm;          // blk.X.post_ffw_norm.weight

    // Per-layer output scaling
    public final float[] layerOutputScale;           // blk.X.layer_output_scale.weight

    // Dual RoPE frequencies (SWA vs full attention)
    public final FloatTensor freq_cis_real_swa;
    public final FloatTensor freq_cis_imag_swa;
    public final FloatTensor freq_cis_real_full;
    public final FloatTensor freq_cis_imag_full;

    // Per-layer embeddings (optional, null if embeddingLengthPerLayer == 0)
    // perLayerTokenEmbd stored as raw MemorySegment because it can exceed 2B elements
    public final MemorySegment perLayerTokenEmbdSegment; // per_layer_token_embd.weight
    public final GGMLType perLayerTokenEmbdType;         // quantization type of per_layer_token_embd
    public final FloatTensor perLayerModelProj;          // per_layer_model_proj.weight
    public final FloatBuffer perLayerProjNorm;       // per_layer_proj_norm.weight
    public final FloatTensor[] perLayerInpGate;      // blk.X.inp_gate.weight
    public final FloatTensor[] perLayerProj;         // blk.X.proj.weight
    public final FloatBuffer[] perLayerPostNorm;     // blk.X.post_norm.weight

    // MoE weights (optional, null if dense model)
    public final FloatTensor[] ffnGateInp;           // blk.X.ffn_gate_inp.weight
    public final FloatBuffer[] ffnGateInpScale;      // blk.X.ffn_gate_inp.scale
    public final FloatTensor[] ffnGateUpExps;        // blk.X.ffn_gate_up_exps.weight
    public final FloatTensor[] ffnDownExps;          // blk.X.ffn_down_exps.weight
    public final FloatBuffer[] ffnDownExpsScale;     // blk.X.ffn_down_exps.scale

    // MoE-specific norms
    public final FloatBuffer[] ffnPostNorm1;         // blk.X.post_ffw_norm_1.weight
    public final FloatBuffer[] preFfwNorm2;          // blk.X.pre_ffw_norm_2.weight
    public final FloatBuffer[] ffnPostNorm2;         // blk.X.post_ffw_norm_2.weight

    // @formatter:off
    public Gemma4StandardWeights(
            FloatTensor token_embedding_table,
            FloatTensor[] rms_att_weight,
            FloatTensor[] wq, FloatTensor[] wk, FloatTensor[] wv, FloatTensor[] wo,
            FloatBuffer[] attnQNorm, FloatBuffer[] attnKNorm,
            FloatBuffer[] postAttentionNorm,
            FloatTensor[] rms_ffn_weight,
            FloatTensor[] w1, FloatTensor[] w2, FloatTensor[] w3,
            FloatBuffer[] postFfwNorm,
            FloatTensor rms_final_weight,
            float[] layerOutputScale,
            FloatTensor freq_cis_real_full, FloatTensor freq_cis_imag_full,
            FloatTensor freq_cis_real_swa, FloatTensor freq_cis_imag_swa,
            FloatTensor wcls,
            MemorySegment perLayerTokenEmbdSegment, GGMLType perLayerTokenEmbdType,
            FloatTensor perLayerModelProj,
            FloatBuffer perLayerProjNorm,
            FloatTensor[] perLayerInpGate, FloatTensor[] perLayerProj,
            FloatBuffer[] perLayerPostNorm,
            FloatTensor[] ffnGateInp, FloatBuffer[] ffnGateInpScale,
            FloatTensor[] ffnGateUpExps, FloatTensor[] ffnDownExps,
            FloatBuffer[] ffnDownExpsScale,
            FloatBuffer[] ffnPostNorm1, FloatBuffer[] preFfwNorm2,
            FloatBuffer[] ffnPostNorm2,
            GGMLType weightType) {
        super(token_embedding_table, rms_att_weight, wq, wk, wv, wo,
              rms_ffn_weight, w1, w2, w3, rms_final_weight,
              freq_cis_real_full, freq_cis_imag_full, // parent's freq_cis aliased to full
              wcls, weightType);

        this.attnQNorm = attnQNorm;
        this.attnKNorm = attnKNorm;
        this.postAttentionNorm = postAttentionNorm;
        this.postFfwNorm = postFfwNorm;
        this.layerOutputScale = layerOutputScale;
        this.freq_cis_real_full = freq_cis_real_full;
        this.freq_cis_imag_full = freq_cis_imag_full;
        this.freq_cis_real_swa = freq_cis_real_swa;
        this.freq_cis_imag_swa = freq_cis_imag_swa;
        this.perLayerTokenEmbdSegment = perLayerTokenEmbdSegment;
        this.perLayerTokenEmbdType = perLayerTokenEmbdType;
        this.perLayerModelProj = perLayerModelProj;
        this.perLayerProjNorm = perLayerProjNorm;
        this.perLayerInpGate = perLayerInpGate;
        this.perLayerProj = perLayerProj;
        this.perLayerPostNorm = perLayerPostNorm;
        this.ffnGateInp = ffnGateInp;
        this.ffnGateInpScale = ffnGateInpScale;
        this.ffnGateUpExps = ffnGateUpExps;
        this.ffnDownExps = ffnDownExps;
        this.ffnDownExpsScale = ffnDownExpsScale;
        this.ffnPostNorm1 = ffnPostNorm1;
        this.preFfwNorm2 = preFfwNorm2;
        this.ffnPostNorm2 = ffnPostNorm2;
    }
    // @formatter:on

    @Override
    public GGMLType getWeightType() {
        return weightType;
    }
}
