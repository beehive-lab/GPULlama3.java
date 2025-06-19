package com.example;

import com.example.aot.AOT;
import com.example.core.model.tensor.FloatTensor;
import com.example.inference.sampler.CategoricalSampler;
import com.example.inference.sampler.Sampler;
import com.example.inference.sampler.ToppSampler;
import com.example.loader.weights.ModelLoader;
import com.example.model.Model;
import com.example.tornadovm.FloatArrayUtils;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.io.IOException;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

public class LlamaApp {
    // Configuration flags for hardware acceleration and optimizations
    public static final boolean USE_VECTOR_API = Boolean.parseBoolean(System.getProperty("llama.VectorAPI", "true"));   // Enable Java Vector API for CPU acceleration
    public static final boolean USE_AOT = Boolean.parseBoolean(System.getProperty("llama.AOT", "false"));               // Use Ahead-of-Time compilation
    public static final boolean USE_TORNADOVM = Boolean.parseBoolean(System.getProperty("use.tornadovm", "false"));     // Use TornadoVM for GPU acceleration
    public static final boolean SHOW_PERF_INTERACTIVE = Boolean.parseBoolean(System.getProperty("llama.ShowPerfInteractive", "true")); // Show performance metrics in interactive mode

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
     * @param vocabularySize The size of the model's vocabulary
     * @param temperature A value controlling randomness in sampling:
     *                   <ul>
     *                     <li>0.0f: No randomness (greedy sampling)</li>
     *                     <li>1.0f: Standard sampling from unmodified distribution</li>
     *                     <li>&lt;1.0f: More deterministic (sharper distribution)</li>
     *                     <li>&gt;1.0f: More random (flatter distribution)</li>
     *                   </ul>
     * @param topp The cumulative probability threshold for nucleus sampling (0.0-1.0).
     *            <ul>
     *              <li>Values ≤0 or ≥1: Disables top-p sampling</li>
     *              <li>Values in (0,1): Restricts sampling to tokens comprising the top p probability mass</li>
     *            </ul>
     * @param rngSeed Seed value for the random number generator to ensure reproducibility
     *
     * @return A configured {@link Sampler} that implements the selected sampling strategy
     *         and handles both tensor and array-based logits
     *
     * @throws IllegalArgumentException if logits are of an unsupported type
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

    /**
     * Loads the language model based on the given options.
     * <p>
     * If Ahead-of-Time (AOT) mode is enabled, attempts to use a pre-loaded compiled model.
     * Otherwise, loads the model from the specified path using the model loader.
     * </p>
     *
     * @param options the parsed CLI options containing model path and max token limit
     * @return the loaded {@link Model} instance
     * @throws IOException if the model fails to load
     * @throws IllegalStateException if AOT loading is enabled but the preloaded model is unavailable
     */
    private static Model loadModel(Options options) throws IOException {
        if (USE_AOT) {
            Model model = AOT.tryUsePreLoaded(options.modelPath(), options.maxTokens());
            if (model == null) {
                throw new IllegalStateException("Failed to load precompiled AOT model.");
            }
            return model;
        }
        return ModelLoader.loadModel(options.modelPath(), options.maxTokens(), true);
    }

    private static Sampler createSampler(Model model, Options options) {
        return selectSampler(model.configuration().vocabularySize(), options.temperature(), options.topp(), options.seed());
    }

    /**
     * Entry point for running the LLaMA-based model with provided command-line arguments.
     *
     * <p>Initializes model options, loads the appropriate model (either AOT or on-demand),
     * configures the sampler, and runs either in interactive or single-instruction mode
     * based on the input options.</p>
     *
     * @param args command-line arguments used to configure model path, temperature, seed, etc.
     * @throws IOException if model loading or file operations fail.
     */
    public static void main(String[] args) throws IOException {
        Options options = Options.parseOptions(args);
        Model model = loadModel(options);
        Sampler sampler = createSampler(model, options);

        if (options.interactive()) {
            model.runInteractive(sampler, options);
        } else {
            model.runInstructOnce(sampler, options);
        }
    }
}



