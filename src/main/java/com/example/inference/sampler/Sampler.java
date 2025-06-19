package com.example.inference.sampler;

import com.example.core.model.tensor.FloatTensor;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Generic interface for sampling tokens from probability distributions.
 * Supports both FloatTensor and FloatArray tensor implementations.
 */
@FunctionalInterface
public interface Sampler {
    /**
     * Sample a token from the provided tensor.
     *
     * @param tensor The tensor containing probabilities/logits
     * @return The selected token index
     */
    int sampleToken(Object tensor);

    /**
     * Argmax implementation for FloatTensor.
     */
    Sampler TENSOR_ARGMAX = tensor -> {
        if (tensor instanceof FloatTensor) {
            return ((FloatTensor) tensor).argmax();
        } else if (tensor instanceof FloatArray) {
            return argmaxFloatArray((FloatArray) tensor);
        }
        throw new IllegalArgumentException("Unsupported tensor type: " +
                (tensor != null ? tensor.getClass().getName() : "null"));
    };

    /**
     * Legacy ARGMAX for backward compatibility.
     * @deprecated Use TENSOR_ARGMAX instead
     */
    @Deprecated
    Sampler ARGMAX = TENSOR_ARGMAX;

    /**
     * Find the index of the maximum value in a FloatArray.
     *
     * @param array The FloatArray to find the maximum value in
     * @return The index of the maximum value
     */
    static int argmaxFloatArray(FloatArray array) {
        float maxValue = Float.NEGATIVE_INFINITY;
        int maxIndex = 0;

        for (int i = 0; i < array.getSize(); i++) {
            float value = array.get(i);
            if (value > maxValue) {
                maxValue = value;
                maxIndex = i;
            }
        }

        return maxIndex;
    }
}