package org.beehive.gpullama3.inference.weights.tornado;

import org.beehive.gpullama3.core.model.GGMLType;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * Gemma3-specific weights for TornadoVM (GPU) execution.
 * Gemma3 has "sandwich normalization" with 4 normalization layers per block
 * plus Q/K normalization within attention.
 */
public class Gemma3TornadoWeights extends FP16Weights {
    public FloatArray[] rms_att_KNormLayered;        // attnKNorm
    public FloatArray[] rms_att_QNormLayered;        // attnQNorm
    public FloatArray[] postAttentionNormLayered;    // post-attention normalization
    public FloatArray[] postFFNNormLayered;          // post-FFN normalization

    // @formatter:off
    public Gemma3TornadoWeights(
            FloatArray tokenEmbeddingTable,
            FloatArray[] rms_att_weightLayered,
            HalfFloatArray[] wqLayered,
            HalfFloatArray[] wkLayered,
            HalfFloatArray[] wvLayered,
            HalfFloatArray[] woLayered,
            FloatArray[] rms_att_KNormLayered,
            FloatArray[] rms_att_QNormLayered,
            FloatArray[] postAttentionNormLayered,
            FloatArray[] rms_ffn_weightLayered,
            HalfFloatArray[] w1Layered,
            HalfFloatArray[] w2Layered,
            HalfFloatArray[] w3Layered,
            FloatArray[] postFFNNormLayered,
            FloatArray rms_final_weight_as_floatArray,
            FloatArray freq_cis_realFlat,
            FloatArray freq_cis_imagFlat,
            HalfFloatArray wclsByteArray,
            GGMLType weightType) {
        // call to FP16Weights constructor
        super(tokenEmbeddingTable,
                rms_att_weightLayered,
                wqLayered,
                wkLayered,
                wvLayered,
                woLayered,
                rms_ffn_weightLayered,
                w1Layered,
                w2Layered,
                w3Layered,
                rms_final_weight_as_floatArray,
                freq_cis_realFlat,
                freq_cis_imagFlat,
                wclsByteArray,
                weightType);
        // init Gemma3-specific fields
        this.rms_att_KNormLayered = rms_att_KNormLayered;
        this.rms_att_QNormLayered = rms_att_QNormLayered;
        this.postAttentionNormLayered = postAttentionNormLayered;
        this.postFFNNormLayered = postFFNNormLayered;
    }
    // @formatter:on
}
