package org.beehive.gpullama3.tornadovm.plan.components;

/**
 * Prepares embedding vectors in host memory before GPU activation graphs run.
 *
 * <p>Concrete implementations are format-specific (FP16 byte copy vs Q8_0 CPU dequantization)
 * and live in the component-provider classes.</p>
 */
public interface EmbeddingPreparer {
    /** Clears the batch input buffer and resets the batch-start position holder to zero. */
    void initBatchState();

    /** Copies or dequantizes embeddings for {@code chunkSize} tokens into the batch buffer and sets the batch-start position holder. */
    void copyBatchEmbeddings(int[] tokenIds, int startPos, int chunkSize);

    /** Copies or dequantizes the embedding for a single decode token into the single-token buffer. */
    void copyDecodeEmbedding(int token);
}
