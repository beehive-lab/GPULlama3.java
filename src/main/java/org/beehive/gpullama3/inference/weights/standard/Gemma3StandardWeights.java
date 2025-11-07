package org.beehive.gpullama3.inference.weights.standard;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;

/**
 * Gemma3-specific weights with "sandwich normalization" architecture.
 * Gemma3 has 4 normalization layers per block:
 * - attn_norm (pre-attention)
 * - post_attention_norm (after attention, before residual)
 * - ffn_norm (pre-FFN)
 * - post_ffw_norm (after FFN, before residual)
 * Plus Q/K normalization within attention.
 */
public class Gemma3StandardWeights extends StandardWeightsWithQKNorm {
    public final FloatTensor[] postAttentionNorm;  // post-attention normalization
    public final FloatTensor[] postFFNNorm;        // post-FFN normalization

    // @formatter:off
    public Gemma3StandardWeights(
            FloatTensor token_embedding_table,
            FloatTensor[] rms_att_weight,
            FloatTensor[] wq,
            FloatTensor[] wk,
            FloatTensor[] wv,
            FloatTensor[] wo,
            FloatTensor[] attnKNorm,
            FloatTensor[] attnQNorm,
            FloatTensor[] postAttentionNorm,
            FloatTensor[] rms_ffn_weight,
            FloatTensor[] w1,
            FloatTensor[] w2,
            FloatTensor[] w3,
            FloatTensor[] postFFNNorm,
            FloatTensor rms_final_weight,
            FloatTensor freq_cis_real,
            FloatTensor freq_cis_imag,
            FloatTensor wcls,
            GGMLType weightType) {
        super(token_embedding_table, rms_att_weight, wq, wk, wv, wo,
              attnKNorm, attnQNorm, rms_ffn_weight, w1, w2, w3, rms_final_weight,
              freq_cis_real, freq_cis_imag, wcls, weightType);
        this.postAttentionNorm = postAttentionNorm;
        this.postFFNNorm = postFFNNorm;
    }
    // @formatter:on

    @Override
    public GGMLType getWeightType() {
        return weightType;
    }
}
