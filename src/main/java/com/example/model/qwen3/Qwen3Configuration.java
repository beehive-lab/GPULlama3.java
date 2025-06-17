package com.example.model.qwen3;

import com.example.model.Configuration;

public record Qwen3Configuration(int dim,
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
        throw new UnsupportedOperationException("Not supported for Qwen3.");
    }

    @Override
    public int kvDim() {
        throw new UnsupportedOperationException("Not supported for Qwen3.");
    }

    @Override
    public int kvMul() {
        throw new UnsupportedOperationException("Not supported for Qwen3.");
    }

    @Override
    public int contextLengthModel() {
        return contextLengthModel;
    }
}
