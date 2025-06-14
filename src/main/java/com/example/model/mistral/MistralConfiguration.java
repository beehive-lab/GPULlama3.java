package com.example.model.mistral;

import com.example.model.Configuration;

public record MistralConfiguration(int dim, int hiddenDim, int numberOfLayers, int numberOfHeads, int numberOfKeyValueHeads, int vocabularySize, int contextLength, boolean sharedWeights,
                                   float rmsNormEps, float ropeTheta) implements Configuration {

    public int kvDim() {
        return dim * numberOfKeyValueHeads / numberOfHeads;
    }

    public int kvMul() {
        return numberOfHeads / numberOfKeyValueHeads;
    }

    public int headSize() {
        return dim / numberOfHeads;
    }
}

