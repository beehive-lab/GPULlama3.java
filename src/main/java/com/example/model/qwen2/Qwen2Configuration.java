package com.example.model.qwen2;

import com.example.model.Configuration;

public record Qwen2Configuration(int dim,
                                 int hiddenDim,
                                 int numberOfLayers,
                                 int numberOfHeads,
                                 int numberOfKeyValueHeads,
                                 int numberOfHeadsKey,
                                 int numberOfHeadsValue,
                                 int vocabularySize,
                                 int contextLengthModel,
                                 int contextLength,
                                 boolean sharedWeights,
                                 float rmsNormEps,
                                 float ropeTheta) implements Configuration {
    @Override
    public int headSize() {
        return dim / numberOfHeads;
    }

    @Override
    public int kvDim() {
        return (dim * numberOfKeyValueHeads) / numberOfHeads;
    }

    @Override
    public int kvMul() {
        throw new UnsupportedOperationException("Not supported for Qwen2.");
    }

    @Override
    public int contextLengthModel() {
        return contextLengthModel;
    }
}
