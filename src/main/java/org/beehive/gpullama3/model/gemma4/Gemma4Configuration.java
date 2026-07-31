package org.beehive.gpullama3.model.gemma4;

import org.beehive.gpullama3.model.Configuration;

/**
 * Configuration for the Gemma 4 architecture (e.g. Gemma-4-E2B-It).
 *
 * <p>Gemma 4 alternates sliding-window and full (global) attention layers, each with their
 * own head dimensions, RoPE base/scaling, and a subset of layers reusing the KV cache produced
 * by an earlier layer ("shared KV layers"). It also augments every layer with a per-layer
 * embedding (PLE) mechanism and applies a final logit soft-cap.</p>
 */
// @formatter:off
public record Gemma4Configuration(String quantization,
                                  int dim,
                                  int numberOfLayers,
                                  int numberOfHeads,
                                  int numberOfKeyValueHeads,
                                  int headDimSwa,
                                  int headDimFull,
                                  int[] feedForwardLength,
                                  boolean[] slidingWindowPattern,
                                  int slidingWindowSize,
                                  int sharedKvLayers,
                                  int embeddingLengthPerLayer,
                                  int vocabularySize,
                                  int contextLengthModel,
                                  int contextLength,
                                  float rmsNormEps,
                                  float ropeTheta,
                                  float ropeThetaSwa,
                                  float finalLogitSoftcapping) implements Configuration {

    @Override
    public String quantization() {
        return quantization;
    }

    @Override
    public int hiddenDim() {
        throw new UnsupportedOperationException("Gemma4 has per-layer feed-forward dimensions; use feedForwardLength(layer).");
    }

    @Override
    public int numberOfHeadsKey() {
        throw new UnsupportedOperationException("Gemma4 has per-layer head dimensions; use headDim(layer).");
    }

    @Override
    public int headSize() {
        throw new UnsupportedOperationException("Gemma4 has per-layer head dimensions; use headDim(layer).");
    }

    @Override
    public int kvDim() {
        throw new UnsupportedOperationException("Gemma4 has per-layer head dimensions; use headDim(layer) * numberOfKeyValueHeads().");
    }

    @Override
    public int kvMul() {
        return numberOfHeads / numberOfKeyValueHeads;
    }

    @Override
    public int contextLengthModel() {
        return contextLengthModel;
    }

    /** Returns the feed-forward (FFN hidden) dimension for the given layer. */
    public int feedForwardLength(int layer) {
        return feedForwardLength[layer];
    }

    /** Whether the given layer uses sliding-window (local) attention as opposed to full (global) attention. */
    public boolean isSwa(int layer) {
        return slidingWindowPattern[layer];
    }

    /** Returns the attention head dimension for the given layer (depends on whether it is a sliding-window or full layer). */
    public int headDim(int layer) {
        return isSwa(layer) ? headDimSwa : headDimFull;
    }

    /** The maximum head dimension across all layers; used to size shared scratch buffers. */
    public int maxHeadDim() {
        return Math.max(headDimSwa, headDimFull);
    }

    /** The maximum feed-forward dimension across all layers; used to size shared scratch buffers. */
    public int maxFeedForwardLength() {
        int max = 0;
        for (int ff : feedForwardLength) {
            max = Math.max(max, ff);
        }
        return max;
    }

    /** Number of (initial) layers that own and populate their own KV cache; later layers reuse one of these. */
    public int nLayerKvFromStart() {
        return numberOfLayers - sharedKvLayers;
    }

    /** Whether the given layer computes and stores its own K/V (as opposed to reusing an earlier layer's KV cache). */
    public boolean hasOwnKv(int layer) {
        return layer < nLayerKvFromStart();
    }

    /** Returns the index of the layer whose KV cache this layer reuses, or -1 if this layer owns its KV cache. */
    public int kvReuseLayer(int layer) {
        if (hasOwnKv(layer)) {
            return -1;
        }
        return nLayerKvFromStart() - (isSwa(layer) ? 2 : 1);
    }
}
// @formatter:on
