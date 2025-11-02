package org.beehive.gpullama3.model.gemma3;

import org.beehive.gpullama3.model.Configuration;

/**
 * Configuration for Google Gemma 3 models.
 *
 * Gemma 3 uses:
 * - Sandwich normalization (4 norm layers per block: pre/post for attention and FFN)
 * - Q/K normalization (per-head normalization of query and key vectors)
 * - Embedding scaling by sqrt(dim)
 * - SentencePiece tokenization with byte-level fallback
 */
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
                                  float ropeTheta,
                                  float attentionScale) implements Configuration {
    @Override
    public int headSize() {
        throw new UnsupportedOperationException("Not supported for Gemma3. Use numberOfHeadsKey for Q/K norm.");
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
}
// @formatter:on
