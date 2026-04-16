package org.beehive.gpullama3.inference.weights.tornado;

import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.tensor.tornado.TornadoTensor;

public class Gemma4TornadoWeights extends TornadoWeights {

    // Per-head Q/K normalization (like Qwen3)
    public final TornadoTensor[] rms_att_QNormLayered;
    public final TornadoTensor[] rms_att_KNormLayered;

    // Post-attention and post-FFN norms (element-wise, F32)
    public final TornadoTensor[] postAttentionNormLayered;
    public final TornadoTensor[] postFfwNormLayered;

    // Per-layer output scale
    public final float[] layerOutputScale;

    // Dual RoPE: SWA frequencies (parent's freq_cis_realFlat/imagFlat store full-attention RoPE)
    public final TornadoTensor freq_cis_real_swa;
    public final TornadoTensor freq_cis_imag_swa;

    // @formatter:off
    public Gemma4TornadoWeights(
            TornadoTensor tokenEmbeddingTable,
            TornadoTensor[] rmsAttWeight,
            TornadoTensor[] wq,
            TornadoTensor[] wk,
            TornadoTensor[] wv,
            TornadoTensor[] wo,
            TornadoTensor[] rms_att_QNormLayered,
            TornadoTensor[] rms_att_KNormLayered,
            TornadoTensor[] postAttentionNormLayered,
            TornadoTensor[] rmsFFNWeight,
            TornadoTensor[] w1,
            TornadoTensor[] w2,
            TornadoTensor[] w3,
            TornadoTensor[] postFfwNormLayered,
            TornadoTensor rmsFinalWeight,
            float[] layerOutputScale,
            TornadoTensor freqCisRealFull,
            TornadoTensor freqCisImagFull,
            TornadoTensor freqCisRealSwa,
            TornadoTensor freqCisImagSwa,
            TornadoTensor wCls,
            GGMLType weightType) {
        super(tokenEmbeddingTable, rmsAttWeight, wq, wk, wv, wo,
                rmsFFNWeight, w1, w2, w3, rmsFinalWeight,
                freqCisRealFull, freqCisImagFull, wCls, weightType);
        this.rms_att_QNormLayered = rms_att_QNormLayered;
        this.rms_att_KNormLayered = rms_att_KNormLayered;
        this.postAttentionNormLayered = postAttentionNormLayered;
        this.postFfwNormLayered = postFfwNormLayered;
        this.layerOutputScale = layerOutputScale;
        this.freq_cis_real_swa = freqCisRealSwa;
        this.freq_cis_imag_swa = freqCisImagSwa;
    }
    // @formatter:on
}
