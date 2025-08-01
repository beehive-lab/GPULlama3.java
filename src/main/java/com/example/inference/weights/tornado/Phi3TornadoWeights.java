package com.example.inference.weights.tornado;

import com.example.core.model.GGMLType;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

public class Phi3TornadoWeights extends TornadoWeights {

    // Phi3-specific weight arrays
    public HalfFloatArray[] wqkvLayered;    // Combined QKV weights: (layer, op_size, dim) where op_size = dim + 2 * (n_kv_heads * head_dim)
    public HalfFloatArray[] wDownLayered;   // FFN down projection: (layer, dim, hidden_dim)
    public HalfFloatArray[] wUpLayered;     // FFN up projection: (layer, hidden_dim, dim)

    // @formatter:off
    public Phi3TornadoWeights(
            FloatArray tokenEmbeddingTable,
            FloatArray[] rms_att_weightLayered,
            HalfFloatArray[] wqkvLayered,        // Combined QKV weights for Phi3
            HalfFloatArray[] woLayered,
            FloatArray[] rms_ffn_weightLayered,
            HalfFloatArray[] wDownLayered,       // FFN down weights
            HalfFloatArray[] wUpLayered,         // FFN up weights
            FloatArray rms_final_weight_as_floatArray,
            FloatArray freq_cis_realFlat,
            FloatArray freq_cis_imagFlat,
            HalfFloatArray wclsByteArray,
            GGMLType weightType) {

        // Call to TornadoWeights constructor with null values for unused standard weights
        super(tokenEmbeddingTable,
                rms_att_weightLayered,
                null,  // wqLayered - not used in Phi3, using combined wqkv instead
                null,  // wkLayered - not used in Phi3, using combined wqkv instead
                null,  // wvLayered - not used in Phi3, using combined wqkv instead
                woLayered,
                rms_ffn_weightLayered,
                null,  // w1Layered - not used in Phi3, using wUp instead
                null,  // w2Layered - not used in Phi3, using wDown instead
                null,  // w3Layered - not used in Phi3, using wUp instead
                rms_final_weight_as_floatArray,
                freq_cis_realFlat,
                freq_cis_imagFlat,
                wclsByteArray,
                weightType);

        // Initialize Phi3-specific fields
        this.wqkvLayered = wqkvLayered;
        this.wDownLayered = wDownLayered;
        this.wUpLayered = wUpLayered;
    }
    // @formatter:on
}