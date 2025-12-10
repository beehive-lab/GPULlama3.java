package org.beehive.gpullama3.model;

public interface Configuration {

    String quantization();

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
