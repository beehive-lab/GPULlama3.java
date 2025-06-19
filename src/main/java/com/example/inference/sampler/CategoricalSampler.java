package com.example.inference.sampler;

import com.example.core.model.tensor.FloatTensor;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.util.random.RandomGenerator;

/**
 * A sampler that samples from a categorical distribution.
 * Supports both FloatTensor and FloatArray implementations.
 */
public record CategoricalSampler(RandomGenerator rng) implements Sampler {

    @Override
    public int sampleToken(Object tensor) {
        if (tensor instanceof FloatTensor) {
            return sampleFromFloatTensor((FloatTensor) tensor);
        } else if (tensor instanceof FloatArray) {
            return sampleFromFloatArray((FloatArray) tensor);
        }
        throw new IllegalArgumentException("Unsupported tensor type: " +
                (tensor != null ? tensor.getClass().getName() : "null"));
    }

    /**
     * Sample from a FloatTensor probability distribution.
     *
     * @param logits The FloatTensor containing probabilities
     * @return The sampled token index
     */
    private int sampleFromFloatTensor(FloatTensor logits) {
        // sample index from probabilities (they must sum to 1!)
        float random0to1 = rng.nextFloat(1f);
        float cdf = 0.0f;
        for (int i = 0; i < logits.size(); i++) {
            cdf += logits.getFloat(i);
            if (random0to1 < cdf) {
                return i;
            }
        }
        return logits.size() - 1; // in case of rounding errors
    }

    /**
     * Sample from a FloatArray probability distribution.
     *
     * @param logits The FloatArray containing probabilities
     * @return The sampled token index
     */
    private int sampleFromFloatArray(FloatArray logits) {
        // sample index from probabilities (they must sum to 1!)
        float random0to1 = rng.nextFloat(1f);
        float cdf = 0.0f;
        for (int i = 0; i < logits.getSize(); i++) {
            cdf += logits.get(i);
            if (random0to1 < cdf) {
                return i;
            }
        }
        return logits.getSize() - 1; // in case of rounding errors
    }
}