package com.example.inference;

import com.example.auxiliary.LastRunMetrics;
import com.example.inference.sampler.Sampler;
import com.example.loader.weights.State;
import com.example.model.Configuration;
import com.example.model.Model;
import com.example.tokenizer.impl.Tokenizer;
import com.example.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

/**
 * Main entry point for LLM token generation.
 *
 * <p>
 * Orchestrates the complete inference process: ingests prompt tokens, then generates
 * new tokens until a stop condition is met. Supports both CPU and GPU execution.
 */
public final class InferenceEngine {

    private InferenceEngine() {
        //prevent instantiation
    }

    /**
     * LLM generation entry point, ingest prompt tokens and generates new tokens.
     *
     * <p>
     * All prompt tokens are ingested first, then inference starts, until a stop token is found.
     * The returned tokens only include generated/inferred tokens.
     *
     * @param model            model to run inference (including weights, configuration, tokenizer ...)
     * @param state            state of the model e.g. key/value caches ... this is mutated by this call
     * @param startPosition    start prompt ingestion + inference at this position in the context e.g. useful if state was kept across calls (chained generation). 0 implies run with no previous context.
     * @param promptTokens     prompt tokens to ingest, all the prompt tokens will be ingested, given there's enough capacity left in the context
     * @param stopTokens       set of tokens that abort generation during inference, stop tokens do not affect prompt ingestion
     * @param maxTokens        maximum number of tokens (can go up to {@link Configuration#contextLength context length}
     *                         if this value is negative or greater than {@link Configuration#contextLength context length}
     * @param sampler          {@link Sampler strategy} used to select tokens
     * @param echo             debugging flag, prints ALL, prompt and inferred tokens, to {@link System#err stderr}
     * @param onTokenGenerated callback, if non-null, it's called every time a token is inferred e.g. it's not called when ingesting prompt tokens
     * @return list of generated/inferred tokens, including the stop token, if any e.g. does not include any token from the prompt
     */
    public static List<Integer> generateTokens(Model model, State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated) {
        // Start timing the whole process
        long startNanos = System.nanoTime();
        long inferenceStartNanos = 0;

        Object logits;
        // Validate and adjust maxTokens if necessary
        if (maxTokens < 0 || model.configuration().contextLength() < maxTokens) {
            maxTokens = model.configuration().contextLength();
        }

        // Storage for generated tokens
        List<Integer> generatedTokens = new ArrayList<>();

        // Initialize token variables
        int currentToken = state.latestToken;
        int nextToken;
        int promptIndex = 0;
        int pos = startPosition;

        while (pos < maxTokens) {

            logits = InferenceCore.forwardJava(model, state, currentToken, pos);

            // Handle token processing
            if (promptIndex < promptTokens.size()) {
                // We're still processing the prompt tokens
                nextToken = promptTokens.get(promptIndex++);
                if (echo) {
                    System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
                }
            } else {
                // Mark the start of actual generation (after prompt processing)
                if (inferenceStartNanos == 0) {
                    inferenceStartNanos = System.nanoTime();
                }

                // Sample the next token
                nextToken = sampler.sampleToken(logits);

                // Output the token if echo is enabled
                if (echo) {
                    System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
                }

                // Track the generated token
                generatedTokens.add(nextToken);

                // Notify via callback if provided
                if (onTokenGenerated != null) {
                    onTokenGenerated.accept(nextToken);
                }

                // Check for stop condition
                if (stopTokens.contains(nextToken)) {
                    break;
                }
            }

            // Update for next iteration
            currentToken = nextToken;
            state.latestToken = currentToken;
            pos++;
        }

        // Calculate and print performance metrics
        long endNanos = System.nanoTime();
        double totalTimeSeconds = (endNanos - startNanos) / 1_000_000_000.0;
        int totalTokens = promptIndex + generatedTokens.size();

        LastRunMetrics.setMetrics(totalTokens, totalTimeSeconds);

        return generatedTokens;
    }

    public static List<Integer> generateTokensGPU(Model model, State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated, TornadoVMMasterPlan tornadoVMPlan) {
        // === Setup and Initialization ===
        long startNanos = System.nanoTime();
        long inferenceStartNanos = 0;

        // Pre-validate the max tokens to avoid checking in the loop
        int actualMaxTokens = Math.min(maxTokens > 0 ? maxTokens : model.configuration().contextLength(), model.configuration().contextLength());

        // Preallocate with expected capacity to avoid resizing
        List<Integer> generatedTokens = new ArrayList<>(Math.min(256, actualMaxTokens - promptTokens.size())); // Conservative estimate

        // === Token Generation Loop ===
        int currentToken = state.latestToken;
        int nextToken;
        int promptIndex = 0;
        int pos = startPosition;

        // Use more efficient direct array access for prompt tokens if possible
        int[] promptTokenArray = null;
        if (promptTokens instanceof ArrayList) {
            // Try to extract the underlying array for faster access
            try {
                // This is a performance optimization that may not work on all JVMs
                promptTokenArray = promptTokens.stream().mapToInt(Integer::intValue).toArray();
            } catch (Exception e) {
                // Fall back to list access
            }
        }

        // Main generation loop
        while (pos < actualMaxTokens) {
            // GPU Forward Pass - No conditional check since we know we're using GPU
            FloatArray logits = InferenceCore.forwardTornadoVM(model, state, currentToken, pos, tornadoVMPlan);

            // Process prompt tokens if still remaining
            if (promptIndex < promptTokens.size()) {
                // Get next prompt token (using array access if available)
                nextToken = promptTokenArray != null ? promptTokenArray[promptIndex++] : promptTokens.get(promptIndex++);

                if (echo) {
                    // Decode and output token
                    System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
                }
            } else {
                // Mark first inference token
                if (inferenceStartNanos == 0) {
                    inferenceStartNanos = System.nanoTime();
                }

                // Sample next token - use GPU sampling if available
                nextToken = sampler.sampleToken(logits);

                // Add token consumer support
                if (onTokenGenerated != null) {
                    onTokenGenerated.accept(nextToken);
                }

                // Output if needed
                if (echo && onTokenGenerated == null) {
                    System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
                }

                // Store token
                generatedTokens.add(nextToken);

                // Check stop condition
                if (stopTokens.contains(nextToken)) {
                    break;
                }
            }

            // Update for next iteration
            currentToken = nextToken;
            state.latestToken = currentToken;
            pos++;
        }

        // === Performance Metrics ===
        long endNanos = System.nanoTime();
        double totalSeconds = (endNanos - startNanos) / 1_000_000_000.0;
        int totalTokens = promptIndex + generatedTokens.size();

        // Set metrics for tokens achieved
        LastRunMetrics.setMetrics(totalTokens, totalSeconds);

        return generatedTokens;
    }
}
