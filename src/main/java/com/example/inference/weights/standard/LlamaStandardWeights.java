package com.example.inference.weights.standard;

import com.example.core.model.GGMLType;
import com.example.core.model.tensor.FloatTensor;

public class LlamaStandardWeights extends StandardWeights {

    public LlamaStandardWeights(FloatTensor token_embedding_table, FloatTensor[] rms_att_weight, FloatTensor[] wq, FloatTensor[] wk, FloatTensor[] wv, FloatTensor[] wo, FloatTensor[] rms_ffn_weight,
            FloatTensor[] w1, FloatTensor[] w2, FloatTensor[] w3, FloatTensor rms_final_weight, FloatTensor freq_cis_real, FloatTensor freq_cis_imag, FloatTensor wcls, GGMLType weightType) {
        super(token_embedding_table, rms_att_weight, wq, wk, wv, wo, rms_ffn_weight, w1, w2, w3, rms_final_weight, freq_cis_real, freq_cis_imag, wcls, weightType);
    }

    @Override
    public GGMLType getWeightType() {
        return weightType;
    }
}
