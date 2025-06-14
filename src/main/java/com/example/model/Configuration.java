package com.example.model;

public interface Configuration {

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

    /** Size of the vocabulary (token set) */
    int vocabularySize();

    /** Maximum sequence length the model can process */
    int contextLength();

    /** Epsilon value for RMSNorm layers (stabilizes normalization) */
    float rmsNormEps();

    /** Base value for RoPE (Rotary Position Embedding) calculations */
    float ropeTheta();

    /** Size of each attention head (derived from dim / numberOfHeads) */
    int headSize();

    int kvDim();

    int kvMul();
}
