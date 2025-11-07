package org.beehive.gpullama3.model.gemma3;

import org.beehive.gpullama3.model.Configuration;

// @formatter:off
public record Gemma3Configuration(int dim,
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
        throw new UnsupportedOperationException("Not supported for Gemma3.");
    }

    @Override
    public int kvDim() {
        throw new UnsupportedOperationException("Not supported for Gemma3.");
    }

    @Override
    public int kvMul() {
        throw new UnsupportedOperationException("Not supported for Gemma3.");
    }

    @Override
    public int contextLengthModel() {
        return contextLengthModel;
    }

    @Override
    public int numberOfHeadsKey() {
        return numberOfHeadsKey;
    }

    /**
     * Creates a new Configuration with a different context length.
     *
     * @param newContextLength The new context length to use
     * @return A new Configuration instance with updated context length,
     *         or the current instance if newContextLength is negative
     */
    // @formatter:off
    public Gemma3Configuration withContextLength(int newContextLength) {
        if (newContextLength < 0) {
            return this; // no change
        }
        return new Gemma3Configuration(
                this.dim,
                this.hiddenDim,
                this.numberOfLayers,
                this.numberOfHeads,
                this.numberOfKeyValueHeads,
                this.numberOfHeadsKey,
                this.numberOfHeadsValue,
                this.vocabularySize,
                this.contextLengthModel,
                newContextLength,
                this.sharedWeights,
                this.rmsNormEps,
                this.ropeTheta
        );
    }
    // @formatter:on
}
