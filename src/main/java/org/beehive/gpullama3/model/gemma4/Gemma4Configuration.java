package org.beehive.gpullama3.model.gemma4;

import org.beehive.gpullama3.model.Configuration;

import java.util.Arrays;

// @formatter:off
public record Gemma4Configuration(
        String quantization,
        int dim,                          // embedding_length
        int numberOfLayers,               // block_count
        int numberOfHeads,                // attention.head_count
        int headSizeFull,                 // attention.key_length (e.g., 512)
        int headSizeSWA,                  // attention.key_length_swa (e.g., 256)
        int slidingWindow,                // attention.sliding_window
        boolean[] isSWA,                  // per-layer SWA pattern
        int[] numberOfKeyValueHeadsPerLayer,  // derived from tensor shapes
        int[] feedForwardLength,          // per-layer FFN hidden dims
        int vocabularySize,
        int contextLength,
        int contextLengthModel,
        float rmsNormEps,
        float ropeTheta,                  // full attention RoPE theta
        float ropeThetaSWA,               // SWA RoPE theta
        float logitSoftcapping,           // logit softcapping (0 = disabled)
        int nLayerKvFromStart,            // shared KV cache boundary
        int embeddingLengthPerLayer,      // per-layer embedding dim (0 = disabled)
        int expertCount,                  // MoE expert count (0 = dense)
        int expertUsedCount,              // top-k experts
        int expertFeedForwardLength       // expert FFN dim
) implements Configuration {

    @Override
    public int hiddenDim() {
        return maxHiddenDim();
    }

    @Override
    public int numberOfKeyValueHeads() {
        throw new UnsupportedOperationException("Gemma4 has per-layer KV head counts, use numberOfKeyValueHeads(layer)");
    }

    @Override
    public int numberOfHeadsKey() {
        return headSizeFull;
    }

    @Override
    public int headSize() {
        return headSizeFull;
    }

    @Override
    public int kvDim() {
        throw new UnsupportedOperationException("Gemma4 has per-layer KV dims, use kvDim(layer)");
    }

    @Override
    public int kvMul() {
        throw new UnsupportedOperationException("Gemma4 has per-layer KV muls");
    }

    // Gemma4-specific helpers

    public boolean isMoE() {
        return expertCount > 0;
    }

    public int maxHiddenDim() {
        return Arrays.stream(feedForwardLength).max().orElseThrow();
    }

    public int headSize(int layer) {
        return isSWA[layer] ? headSizeSWA : headSizeFull;
    }

    public int numberOfKeyValueHeads(int layer) {
        return numberOfKeyValueHeadsPerLayer[layer];
    }

    public int kvDim(int layer) {
        return numberOfKeyValueHeadsPerLayer[layer] * headSize(layer);
    }

    public int queryDim(int layer) {
        return numberOfHeads * headSize(layer);
    }

    public int kvCachePositions(int layer) {
        return isSWA[layer] ? Math.min(contextLength, slidingWindow) : contextLength;
    }

    public int kvCacheIndex(int layer, int position) {
        return isSWA[layer] ? (position & (slidingWindow - 1)) : position;
    }

    public boolean hasKv(int layer) {
        return layer < nLayerKvFromStart;
    }

    public int kvSourceLayer(int layer) {
        if (layer < nLayerKvFromStart) return layer;
        return nLayerKvFromStart - (isSWA[layer] ? 2 : 1);
    }
}
// @formatter:on
