package com.example.inference.weights.tornado;

import com.example.core.model.GGMLType;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

public class LlamaTornadoWeights extends TornadoWeights{

    public LlamaTornadoWeights(FloatArray tokenEmbeddingTable, FloatArray[] rms_att_weightLayered, HalfFloatArray[] wqLayered, HalfFloatArray[] wkLayered, HalfFloatArray[] wvLayered,
            HalfFloatArray[] woLayered, FloatArray[] rms_ffn_weightLayered, HalfFloatArray[] w1Layered, HalfFloatArray[] w2Layered, HalfFloatArray[] w3Layered,
            FloatArray rms_final_weight_as_floatArray, FloatArray freq_cis_realFlat, FloatArray freq_cis_imagFlat, HalfFloatArray wclsByteArray, GGMLType weightType) {
        super(tokenEmbeddingTable, rms_att_weightLayered, wqLayered, wkLayered, wvLayered, woLayered, rms_ffn_weightLayered, w1Layered, w2Layered, w3Layered, rms_final_weight_as_floatArray,
                freq_cis_realFlat, freq_cis_imagFlat, wclsByteArray, weightType);
    }
}
