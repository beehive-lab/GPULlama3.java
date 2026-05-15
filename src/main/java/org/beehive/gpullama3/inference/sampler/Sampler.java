package org.beehive.gpullama3.inference.sampler;

import org.beehive.gpullama3.Options;
import org.beehive.gpullama3.tensor.standard.FloatTensor;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tornadovm.utils.FloatArrayUtils;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Generic interface for sampling tokens from probability distributions. Supports both FloatTensor and FloatArray tensor implementations.
 */
@FunctionalInterface
public interface Sampler {

    /**
     * Argmax implementation for FloatTensor.
     */
    Sampler TENSOR_ARGMAX = tensor -> {
        if (tensor instanceof FloatTensor) {
            return ((FloatTensor) tensor).argmax();
        } else if (tensor instanceof FloatArray) {
            return argmaxFloatArray((FloatArray) tensor);
        }
        throw new IllegalArgumentException("Unsupported tensor type: " + (tensor != null ? tensor.getClass().getName() : "null"));
    };
    /**
     * Legacy ARGMAX for backward compatibility.
     *
     * @deprecated Use TENSOR_ARGMAX instead
     */
    @Deprecated
    Sampler ARGMAX = TENSOR_ARGMAX;

    /**
     * Creates and configures a sampler for token generation based on specified parameters.
     *
     * <p>This method selects an appropriate sampling strategy for next-token prediction
     * in language model inference. It supports several sampling approaches:</p>
     *
     * <ul>
     *   <li>Greedy sampling (temperature = 0): Always selects the most probable token</li>
     *   <li>Temperature sampling: Adjusts probability distribution sharpness</li>
     *   <li>Top-p (nucleus) sampling: Considers only tokens comprising the top p probability mass</li>
     * </ul>
     *
     * <p>The method handles both {@link FloatTensor} and {@link FloatArray} logits types
     * to support both CPU and GPU execution paths.</p>
     *
     * @param vocabularySize
     *         The size of the model's vocabulary
     * @param temperature
     *         A value controlling randomness in sampling:
     *         <ul>
     *           <li>0.0f: No randomness (greedy sampling)</li>
     *           <li>1.0f: Standard sampling from unmodified distribution</li>
     *           <li>&lt;1.0f: More deterministic (sharper distribution)</li>
     *           <li>&gt;1.0f: More random (flatter distribution)</li>
     *         </ul>
     * @param topp
     *         The cumulative probability threshold for nucleus sampling (0.0-1.0).
     *         <ul>
     *           <li>Values ≤0 or ≥1: Disables top-p sampling</li>
     *           <li>Values in (0,1): Restricts sampling to tokens comprising the top p probability mass</li>
     *         </ul>
     * @param rngSeed
     *         Seed value for the random number generator to ensure reproducibility
     * @return A configured {@link Sampler} that implements the selected sampling strategy and handles both tensor and array-based logits
     * @throws IllegalArgumentException
     *         if logits are of an unsupported type
     */
    static Sampler selectSampler(int vocabularySize, float temperature, float topp, long rngSeed) {
        Sampler sampler;
        if (temperature == 0.0f) {
            // greedy argmax sampling: take the token with the highest probability
            sampler = Sampler.TENSOR_ARGMAX; // Use TENSOR_ARGMAX instead of ARGMAX
        } else {
            // we sample from this distribution to get the next token
            RandomGenerator rng = RandomGeneratorFactory.getDefault().create(rngSeed);
            Sampler innerSampler;
            // Determine whether to use top-p (nucleus) sampling
            if (topp <= 0 || topp >= 1) {
                // If topp is outside (0,1), use standard categorical sampling
                // This samples directly from the probability distribution
                innerSampler = new CategoricalSampler(rng);
            } else {
                // Use top-p (nucleus) sampling with the specified threshold
                // This restricts sampling to only the most likely tokens that
                // cumulatively comprise the top p probability mass
                innerSampler = new ToppSampler(vocabularySize, topp, rng);
            }

            // Create a sampler that:
            // 1. Applies temperature scaling to the logits
            // 2. Converts logits to probabilities using softmax
            // 3. Delegates the actual sampling to the appropriate inner sampler
            sampler = logits -> {
                // Handle different logits formats to support both CPU and GPU paths
                if (logits instanceof FloatTensor) {
                    // For CPU path using FloatTensor
                    FloatTensor tensorLogits = (FloatTensor) logits;
                    // Apply temperature scaling - lower values make distribution more peaked
                    tensorLogits.divideInPlace(0, tensorLogits.size(), temperature);
                    // Convert logits to probabilities using softmax
                    tensorLogits.softmaxInPlace(0, tensorLogits.size());
                } else if (logits instanceof FloatArray) {
                    // For GPU path using FloatArray
                    FloatArray arrayLogits = (FloatArray) logits;
                    // Apply the same operations but using FloatArray-specific methods for TornadoVM data types
                    FloatArrayUtils.divideInPlace(arrayLogits, 0, arrayLogits.getSize(), temperature);
                    FloatArrayUtils.softmaxInPlace(arrayLogits, 0, arrayLogits.getSize());
                } else {
                    // If logits are neither FloatTensor nor FloatArray, throw an exception
                    throw new IllegalArgumentException("Unsupported logits type: " + (logits != null ? logits.getClass().getName() : "null"));
                }
                return innerSampler.sampleToken(logits);
            };
        }
        return sampler;
    }

    static Sampler createSampler(Model model, Options options) {
        float temperature = Float.isNaN(options.temperature())
                ? (float) model.chatFormat().defaultTemperature()
                : options.temperature();
        float topp = Float.isNaN(options.topp())
                ? (float) model.chatFormat().defaultTopP()
                : options.topp();
        return selectSampler(model.configuration().vocabularySize(), temperature, topp, options.seed());
    }

    /**
     * Find the index of the maximum value in a FloatArray.
     *
     * @param array
     *         The FloatArray to find the maximum value in
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

    /**
     * Sample a token from the provided tensor.
     *
     * @param tensor
     *         The tensor containing probabilities/logits
     * @return The selected token index
     */
    int sampleToken(Object tensor);
}