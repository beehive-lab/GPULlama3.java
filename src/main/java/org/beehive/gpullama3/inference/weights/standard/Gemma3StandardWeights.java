package org.beehive.gpullama3.inference.weights.standard;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;

/**
 * Weight class for Google Gemma 3 models (CPU inference).
 *
 * <p>Gemma 3 uses "sandwich normalization" with 4 norm layers per block:</p>
 * <ul>
 *   <li>Pre-attention norm (rms_att_weight)</li>
 *   <li>Post-attention norm (postAttentionNorm)</li>
 *   <li>Pre-FFN norm (rms_ffn_weight)</li>
 *   <li>Post-FFN norm (postFFNNorm)</li>
 * </ul>
 *
 * <p>It also includes Q/K normalization like Qwen3.</p>
 */
public class Gemma3StandardWeights extends Qwen3StandardWeights {

    // Additional Gemma3-specific norm layers (sandwich normalization)
    public final FloatTensor[] postAttentionNorm;  // Post-attention normalization
    public final FloatTensor[] postFFNNorm;        // Post-FFN normalization

    // @formatter:off
    /**
     * Constructor for {@code Gemma3StandardWeights}.
     *
     * @param token_embedding_table The token embedding table, used to map tokens to embeddings.
     * @param rms_att_weight        The array of Root Mean Square (RMS) attention weights (pre-attention norm).
     * @param wq                    The array of query weight tensors for attention layers.
     * @param wk                    The array of key weight tensors for attention layers.
     * @param wv                    The array of value weight tensors for attention layers.
     * @param wo                    The array of output weight tensors for attention layers.
     * @param attnKNorm             The array of normalization tensors for attention keys.
     * @param attnQNorm             The array of normalization tensors for attention queries.
     * @param postAttentionNorm     The array of post-attention normalization tensors.
     * @param rms_ffn_weight        The array of RMS weights for feed-forward neural network layers (pre-FFN norm).
     * @param w1                    The array of first weight tensors for feed-forward layers.
     * @param w2                    The array of second weight tensors for feed-forward layers.
     * @param w3                    The array of third weight tensors for feed-forward layers.
     * @param postFFNNorm           The array of post-FFN normalization tensors.
     * @param rms_final_weight      The RMS weight used for final output normalization.
     * @param freq_cis_real         The real part of the frequency position encodings.
     * @param freq_cis_imag         The imaginary part of the frequency position encodings.
     * @param wcls                  The weight tensor for the classification head.
     * @param weightType            The type of the weights, defined as {@link GGMLType}.
     */
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
        // Call Qwen3StandardWeights constructor (which has Q/K norm)
        super(token_embedding_table,
                rms_att_weight,
                wq,
                wk,
                wv,
                wo,
                attnKNorm,
                attnQNorm,
                rms_ffn_weight,
                w1,
                w2,
                w3,
                rms_final_weight,
                freq_cis_real,
                freq_cis_imag,
                wcls,
                weightType);

        // Initialize Gemma3-specific sandwich normalization fields
        this.postAttentionNorm = postAttentionNorm;
        this.postFFNNorm = postFFNNorm;
    }
    // @formatter:on

    @Override
    public GGMLType getWeightType() {
        return weightType;
    }
}
