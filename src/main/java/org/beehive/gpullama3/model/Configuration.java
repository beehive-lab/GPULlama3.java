package org.beehive.gpullama3.model;

import org.beehive.gpullama3.runtime.tensor.DataType;

public interface Configuration {

    /**
     * @deprecated Use {@link #activationType()}. This string comes from GGUF's {@code
     *     general.file_type} and describes the <i>file</i>, which is the least reliable of the
     *     three notions of "the model's type" the code carries: a K-quant file reports {@code
     *     "Q8_0"} because that is what its activations end up as, not because that is what is in
     *     it. The per-tensor {@code DataType} on a descriptor is the truth.
     */
    @Deprecated
    String quantization();

    /**
     * The representation activations are held in.
     *
     * <p>Not the weights' type: an FP16 model keeps FP16 activations, and everything quantized
     * quantizes its activations to Q8_0 to match the kernels that consume them.
     */
    default DataType activationType() {
        return "FP16".equals(quantization()) ? DataType.F16 : DataType.Q8_0;
    }

    /** Transformer embedding dimension */
    int dim();

    /** Hidden dimension size for feed-forward network layers */
    int hiddenDim();

    /** Number of transformer layers in the model */
    int numberOfLayers();

    /** Number of attention heads for queries */
    int numberOfHeads();

    /** Number of key/value heads (can be fewer than query heads in multi-query attention) */
    int numberOfKeyValueHeads();

    int numberOfHeadsKey();

    /** Size of the vocabulary (token set) */
    int vocabularySize();

    /** Maximum sequence length the model can process */
    int contextLength();

    /** Max sequence length in model */
    int contextLengthModel();

    /** Epsilon value for RMSNorm layers (stabilizes normalization) */
    float rmsNormEps();

    /** Base value for RoPE (Rotary Position Embedding) calculations */
    float ropeTheta();

    int headSize();

    int kvDim();

    int kvMul();
}
