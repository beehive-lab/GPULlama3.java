package org.beehive.gpullama3.inference.weights.tornado;

import org.beehive.gpullama3.core.model.GGMLType;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

public class Qwen2TornadoWeights extends TornadoWeights {

    // Qwen2-specific tornado weights
    FloatArray[] q_biasLayered;
    FloatArray[] k_biasLayered;
    FloatArray[] v_biasLayered;

    public Qwen2TornadoWeights(FloatArray tokenEmbeddingTable, FloatArray[] rms_att_weightLayered, HalfFloatArray[] wqLayered, HalfFloatArray[] wkLayered, HalfFloatArray[] wvLayered,
            FloatArray[] wqBiasLayered,
            FloatArray[] wkBiasLayered,
            FloatArray[] wvBiasLayered,
            HalfFloatArray[] woLayered, FloatArray[] rms_ffn_weightLayered, HalfFloatArray[] w1Layered,
            HalfFloatArray[] w2Layered, HalfFloatArray[] w3Layered, FloatArray rms_final_weight_as_floatArray, FloatArray freq_cis_realFlat, FloatArray freq_cis_imagFlat, HalfFloatArray wclsByteArray,
            GGMLType weightType) {
        // call to TornadoWeights constructor
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
        // init qwen2-specific fields
        this.q_biasLayered = wqBiasLayered;
        this.k_biasLayered = wkBiasLayered;
        this.v_biasLayered = wvBiasLayered;
    }
}
