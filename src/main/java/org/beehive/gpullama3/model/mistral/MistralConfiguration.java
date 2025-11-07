package org.beehive.gpullama3.model.mistral;

import org.beehive.gpullama3.model.Configuration;

// @formatter:off
public record MistralConfiguration(int dim,
                                   int hiddenDim,
                                   int numberOfLayers,
                                   int numberOfHeads,
                                   int numberOfKeyValueHeads,
                                   int vocabularySize,
                                   int contextLength,
                                   boolean sharedWeights,
                                   float rmsNormEps,
                                   float ropeTheta) implements Configuration {

    public int kvDim() {
        return dim * numberOfKeyValueHeads / numberOfHeads;
    }

    public int kvMul() {
        return numberOfHeads / numberOfKeyValueHeads;
    }

    @Override
    public int numberOfHeadsKey() {
        throw new UnsupportedOperationException("Not supported for Mistral.");
    }

    @Override
    public int numberOfHeadsValue() {
        throw new UnsupportedOperationException("Not supported for Mistral.");
    }

    @Override
    public int contextLengthModel() {
        throw new UnsupportedOperationException("Not supported for Mistral.");
    }

    public int headSize() {
        return dim / numberOfHeads;
    }
}

