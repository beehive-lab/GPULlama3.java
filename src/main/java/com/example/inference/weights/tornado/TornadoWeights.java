package com.example.inference.weights.tornado;

import com.example.core.model.GGMLType;
import com.example.inference.weights.Weights;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

public abstract class TornadoWeights implements Weights {

    public FloatArray[] rms_att_weightLayered;          // (layer, dim) rmsnorm weights
    public HalfFloatArray[] wqLayered;                  // (layer, n_heads * head_size)
    public HalfFloatArray[] wkLayered;                  // (layer, n_kv_heads, head_size)
    public HalfFloatArray[] wvLayered;                  // (layer, n_kv_heads * head_size)
    public HalfFloatArray[] woLayered;                  // (layer, n_heads * head_size, dim)
    public FloatArray[] rms_ffn_weightLayered;          // (layer, dim)
    public HalfFloatArray[] w1Layered;                  // (layer, hidden_dim, dim)
    public HalfFloatArray[] w2Layered;                  // (layer, dim, hidden_dim)
    public HalfFloatArray[] w3Layered;                  // (layer, hidden_dim, dim)
    public FloatArray rms_final_weight_as_floatArray;
    public FloatArray tokenEmbeddingTable;              // (vocab_size, dim)
    public FloatArray freq_cis_realFlat;                // (seq_len, head_size/2)
    public FloatArray freq_cis_imagFlat;                // (seq_len, head_size/2)
    public HalfFloatArray wclsHalfFloat;

    // (optional) classifier weights for the logits, on the last layer
    protected final GGMLType weightType;

    protected TornadoWeights(
            FloatArray tokenEmbeddingTable,
            FloatArray[] rms_att_weightLayered,
            HalfFloatArray[] wqLayered,
            HalfFloatArray[] wkLayered,
            HalfFloatArray[] wvLayered,
            HalfFloatArray[] woLayered,
            FloatArray[] rms_ffn_weightLayered,
            HalfFloatArray[] w1Layered,
            HalfFloatArray[] w2Layered,
            HalfFloatArray[] w3Layered,
            FloatArray rms_final_weight_as_floatArray,
            FloatArray freq_cis_realFlat,
            FloatArray freq_cis_imagFlat,
            HalfFloatArray wclsByteArray,
            GGMLType weightType) {
        // TornadoVM format
        this.tokenEmbeddingTable = tokenEmbeddingTable;
        this.rms_att_weightLayered = rms_att_weightLayered;
        this.wqLayered = wqLayered;
        this.wkLayered = wkLayered;
        this.wvLayered = wvLayered;
        this.woLayered = woLayered;
        this.rms_ffn_weightLayered = rms_ffn_weightLayered;
        this.w1Layered = w1Layered;
        this.w2Layered = w2Layered;
        this.w3Layered = w3Layered;
        this.rms_final_weight_as_floatArray = rms_final_weight_as_floatArray;
        this.freq_cis_realFlat = freq_cis_realFlat;
        this.freq_cis_imagFlat = freq_cis_imagFlat;
        this.wclsHalfFloat = wclsByteArray;
        this.weightType = weightType;
    }

    @Override
    public GGMLType getWeightType() {
        return weightType;
    }


}
