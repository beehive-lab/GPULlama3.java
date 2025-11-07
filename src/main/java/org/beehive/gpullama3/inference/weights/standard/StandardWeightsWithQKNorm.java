package org.beehive.gpullama3.inference.weights.standard;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;

/**
 * Base class for model weights that include Q/K normalization layers.
 * Used by models like Qwen3 and Gemma3 that apply normalization to attention queries and keys.
 */
public abstract class StandardWeightsWithQKNorm extends StandardWeights {
    public final FloatTensor[] attnKNorm, attnQNorm;

    // @formatter:off
    public StandardWeightsWithQKNorm(
            FloatTensor token_embedding_table,
            FloatTensor[] rms_att_weight,
            FloatTensor[] wq,
            FloatTensor[] wk,
            FloatTensor[] wv,
            FloatTensor[] wo,
            FloatTensor[] attnKNorm,
            FloatTensor[] attnQNorm,
            FloatTensor[] rms_ffn_weight,
            FloatTensor[] w1,
            FloatTensor[] w2,
            FloatTensor[] w3,
            FloatTensor rms_final_weight,
            FloatTensor freq_cis_real,
            FloatTensor freq_cis_imag,
            FloatTensor wcls,
            GGMLType weightType) {
        super(token_embedding_table, rms_att_weight, wq, wk, wv, wo,
              rms_ffn_weight, w1, w2, w3, rms_final_weight,
              freq_cis_real, freq_cis_imag, wcls, weightType);
        this.attnKNorm = attnKNorm;
        this.attnQNorm = attnQNorm;
    }
    // @formatter:on
}
