package org.beehive.gpullama3.model;

import org.beehive.gpullama3.runtime.tensor.DataType;

public interface Configuration {

    /**
     * @deprecated Use {@link #activationType()}. This string comes from GGUF's {@code
     *     general.file_type} and describes the <i>file</i>, which is the least reliable of the
     *     three notions of "the model's type" the code carries: a K-quant file reports {@code
     *     "Q8_0"} because that is what its activations end up as, not because that is what is in
     *     it. The per-tensor {@code DataType} on a descriptor is the truth.
     */
    @Deprecated
    String quantization();

    /**
     * The representation activations are held in.
     *
     * <p>Not the weights' type: an FP16 model keeps FP16 activations, and everything quantized
     * quantizes its activations to Q8_0 to match the kernels that consume them.
     */
    default DataType activationType() {
        return "FP16".equals(quantization()) ? DataType.F16 : DataType.Q8_0;
    }

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

    int numberOfHeadsKey();

    // @formatter:off
    /**
     * How many layers hold key/value entries.
     *
     * <p>Every layer, for a stack that is attention throughout — which is every family but one.
     * {@code qwen35} attends in one layer of four and mixes the rest with a recurrence that
     * retains nothing per position, so its key/value store is sized by this rather than by the
     * layer count, and a memory prediction built from {@link #numberOfLayers()} over-predicts it
     * fourfold.
     */
    // @formatter:on
    default int keyValueLayerCount() {
        return numberOfLayers();
    }

    // @formatter:off
    /**
     * Bytes of per-session state that is neither key/value cache nor scratch.
     *
     * <p>Zero for a stack that is attention throughout. A recurrent layer keeps its history in a
     * fixed-size state instead of in a growing cache — a convolution window and a delta-net matrix
     * per head — which persists across tokens, is updated in place, and is sized from the
     * configuration rather than from the context length. It is not scratch, so a workspace figure
     * derived from the transformer's dimensions does not include it.
     */
    // @formatter:on
    default long recurrentStateBytes() {
        return 0L;
    }

    /** Size of the vocabulary (token set) */
    int vocabularySize();

    /** Maximum sequence length the model can process */
    int contextLength();

    /** Max sequence length in model */
    int contextLengthModel();

    /** Epsilon value for RMSNorm layers (stabilizes normalization) */
    float rmsNormEps();

    /** Base value for RoPE (Rotary Position Embedding) calculations */
    float ropeTheta();

    int headSize();

    int kvDim();

    int kvMul();
}
