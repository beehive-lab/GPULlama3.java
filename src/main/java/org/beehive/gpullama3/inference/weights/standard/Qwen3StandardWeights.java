package org.beehive.gpullama3.inference.weights.standard;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;

/**
 * A model-specific implementation of {@link StandardWeightsWithQKNorm} for the Qwen-3 model.
 * This class defines the weights required for performing inference
 * using the Qwen-3 model in the standard CPU-based format.
 */
public class Qwen3StandardWeights extends StandardWeightsWithQKNorm {

    // @formatter:off
    /**
     * Constructor for {@code Qwen3StandardWeights}.
     *
     * @param token_embedding_table The token embedding table, used to map tokens to embeddings.
     * @param rms_att_weight        The array of Root Mean Square (RMS) attention weights.
     * @param wq                    The array of query weight tensors for attention layers.
     * @param wk                    The array of key weight tensors for attention layers.
     * @param wv                    The array of value weight tensors for attention layers.
     * @param wo                    The array of output weight tensors for attention layers.
     * @param attnKNorm             The array of normalization tensors for attention keys.
     * @param attnQNorm             The array of normalization tensors for attention queries.
     * @param rms_ffn_weight        The array of RMS weights for feed-forward neural network layers.
     * @param w1                    The array of first weight tensors for feed-forward layers.
     * @param w2                    The array of second weight tensors for feed-forward layers.
     * @param w3                    The array of third weight tensors for feed-forward layers.
     * @param rms_final_weight      The RMS weight used for final output normalization.
     * @param freq_cis_real         The real part of the frequency position encodings.
     * @param freq_cis_imag         The imaginary part of the frequency position encodings.
     * @param wcls                  The weight tensor for the classification head.
     * @param weightType            The type of the weights, defined as {@link GGMLType}.
     */
    public Qwen3StandardWeights(
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
        // call to StandardWeightsWithQKNorm constructor
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
    }
    // @formatter:on

    @Override
    public GGMLType getWeightType() {
        return weightType;
    }
}
