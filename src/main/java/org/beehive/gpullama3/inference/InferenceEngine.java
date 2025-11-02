package org.beehive.gpullama3.inference;

import org.beehive.gpullama3.auxiliary.LastRunMetrics;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

/**
 * Main entry point for LLM token generation.
 *
 * <p>
 * Orchestrates the complete inference process: ingests prompt tokens, then generates new tokens until a stop condition is met. Supports both CPU and GPU execution.
 * </p>
 *
 * <p>
 * It provides unified logic for the following methods:
 * <ul>
 *     <li>{@link #generateTokensLlama}     – for LLaMA and Mistral models running on CPU</li>
 *     <li>{@link #generateTokensGPULlama}  – for LLaMA and Mistral models executed on GPU</li>
 *     <li>{@link #generateTokensQwen3}     – for Qwen3 models running on CPU</li>
 *     <li>{@link #generateTokensGPUQwen3}  – for Qwen3 models executed on GPU</li>
 * </ul>
 * </p>
 */
public final class InferenceEngine {

    private InferenceEngine() {
        //prevent instantiation
    }

    /**
     * Apply repetition penalty to logits of recently generated tokens.
     * This prevents the model from getting stuck in loops by penalizing tokens that were recently generated.
     *
     * @param logits the logits tensor to modify
     * @param recentTokens set of recently generated token IDs to penalize
     * @param penaltyFactor factor to divide logits by (> 1.0 to reduce probability)
     */
    private static void applyRepetitionPenalty(Object logits, Set<Integer> recentTokens, float penaltyFactor) {
        if (recentTokens == null || recentTokens.isEmpty() || penaltyFactor <= 1.0f) {
            return; // No penalty to apply
        }

        for (int token : recentTokens) {
            if (logits instanceof org.beehive.gpullama3.core.model.tensor.FloatTensor) {
                org.beehive.gpullama3.core.model.tensor.FloatTensor floatTensor = (org.beehive.gpullama3.core.model.tensor.FloatTensor) logits;
                if (token >= 0 && token < floatTensor.size()) {
                    float currentLogit = floatTensor.getFloat(token);
                    floatTensor.setFloat(token, currentLogit / penaltyFactor);
                }
            } else if (logits instanceof FloatArray) {
                FloatArray floatArray = (FloatArray) logits;
                if (token >= 0 && token < floatArray.getSize()) {
                    float currentLogit = floatArray.get(token);
                    floatArray.set(token, currentLogit / penaltyFactor);
                }
            }
        }
    }

    /**
     * LLM generation entry point, ingest prompt tokens and generates new tokens.
     *
     * <p>
     * All prompt tokens are ingested first, then inference starts, until a stop token is found. The returned tokens only include generated/inferred tokens.
     *
     * @param model
     *         model to run inference (including weights, configuration, tokenizer ...)
     * @param state
     *         state of the model e.g. key/value caches ... this is mutated by this call
     * @param startPosition
     *         start prompt ingestion + inference at this position in the context e.g. useful if state was kept across calls (chained generation). 0 implies run with no previous context.
     * @param promptTokens
     *         prompt tokens to ingest, all the prompt tokens will be ingested, given there's enough capacity left in the context
     * @param stopTokens
     *         set of tokens that abort generation during inference, stop tokens do not affect prompt ingestion
     * @param maxTokens
     *         maximum number of tokens (can go up to {@link Configuration#contextLength context length} if this value is negative or greater than {@link Configuration#contextLength context length}
     * @param sampler
     *         {@link Sampler strategy} used to select tokens
     * @param echo
     *         debugging flag, prints ALL, prompt and inferred tokens, to {@link System#err stderr}
     * @param onTokenGenerated
     *         callback, if non-null, it's called every time a token is inferred e.g. it's not called when ingesting prompt tokens
     * @return list of generated/inferred tokens, including the stop token, if any e.g. does not include any token from the prompt
     */
    public static List<Integer> generateTokensLlama(Model model, State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
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

        // Repetition penalty tracking: keep track of last 5 generated tokens
        final int REPETITION_PENALTY_WINDOW = 5;
        final float REPETITION_PENALTY_FACTOR = 3.0f; // Penalize by dividing logits by 3.0 (stronger penalty)
        LinkedList<Integer> recentTokens = new LinkedList<>();
        Set<Integer> recentTokensSet = new HashSet<>();

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

                // Apply repetition penalty to prevent token loops
                applyRepetitionPenalty(logits, recentTokensSet, REPETITION_PENALTY_FACTOR);

                // Sample the next token
                nextToken = sampler.sampleToken(logits);

                // Output the token if echo is enabled
                if (echo) {
                    System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
                }

                // Track the generated token
                generatedTokens.add(nextToken);

                // Track token for repetition penalty
                recentTokens.addLast(nextToken);
                recentTokensSet.add(nextToken);

                // Keep window size limited
                if (recentTokens.size() > REPETITION_PENALTY_WINDOW) {
                    int removedToken = recentTokens.removeFirst();
                    // Only remove from set if this token isn't elsewhere in the window
                    if (!recentTokens.contains(removedToken)) {
                        recentTokensSet.remove(removedToken);
                    }
                }

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

    public static List<Integer> generateTokensQwen3(Model model, State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated) {
        // Start timing the whole process
        long startNanos = System.nanoTime();
        long inferenceStartNanos = 0;

        // Validate and adjust maxTokens if necessary
        if (maxTokens < 0 || model.configuration().contextLength() < maxTokens) {
            maxTokens = model.configuration().contextLength();
        }

        // Storage for generated tokens
        List<Integer> generatedTokens = new ArrayList<>();

        // Initialize token variables
        int currentToken = state.latestToken; // BOS?
        int nextToken = 0;
        int promptIndex = 0;

        // Repetition penalty tracking: keep track of last 5 generated tokens
        final int REPETITION_PENALTY_WINDOW = 5;
        final float REPETITION_PENALTY_FACTOR = 3.0f; // Penalize by dividing logits by 3.0 (stronger penalty)
        LinkedList<Integer> recentTokens = new LinkedList<>();
        Set<Integer> recentTokensSet = new HashSet<>();

        // FIX: Loop must run for prompt processing + token generation
        int totalIterations = promptTokens.size() + maxTokens;

        for (int position = startPosition; position < totalIterations; ++position) {

            // Handle token processing
            if (promptIndex < promptTokens.size()) {
                // We're still processing the prompt tokens
                final int token = promptTokens.get(promptIndex);

                model.forward(state, token, position);

                promptIndex++;
                if (promptIndex < promptTokens.size()) {
                    continue;
                }
                if (echo) {
                    System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
                }
                // We have reached the last prompt token and computed the first response-token.
                // BUG FIX: Don't manually increment position - the for loop will do it!
                // The old code did position++ here, causing the loop's ++position to skip a position
            } else {
                // Mark the start of actual generation (after prompt processing)
                if (inferenceStartNanos == 0) {
                    inferenceStartNanos = System.nanoTime();
                }

                model.forward(state, currentToken, position);
            }

            // Apply repetition penalty to prevent token loops
            applyRepetitionPenalty(state.logits, recentTokensSet, REPETITION_PENALTY_FACTOR);

            // Sample the next token
            nextToken = sampler.sampleToken(state.logits);

            // DEBUG: Log what token was selected at each position
            System.err.printf(">>> Position %d (promptIndex=%d): Selected token %d\n", position, promptIndex, nextToken);

            if (position < 5) {
                // Also log top-5 tokens for position 0 to understand if it's selecting the right token
                if (position == 0 || position == 1) {
                    // Find top 5 logits - check first few values for debugging
                    System.err.printf("    Top logits at position %d (first 10): ", position);
                    for (int t = 0; t < 10; t++) {
                        System.err.printf("[%d]=%.2f ", t, state.logits.getFloat(t));
                    }
                    System.err.println();
                }
            }

            // Output the token if echo is enabled
            if (echo) {
                System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
            }

            // Track the generated token
            generatedTokens.add(nextToken);

            // Track token for repetition penalty (only after prompt phase)
            if (promptIndex >= promptTokens.size()) {
                recentTokens.addLast(nextToken);
                recentTokensSet.add(nextToken);

                // Keep window size limited
                if (recentTokens.size() > REPETITION_PENALTY_WINDOW) {
                    int removedToken = recentTokens.removeFirst();
                    // Only remove from set if this token isn't elsewhere in the window
                    if (!recentTokens.contains(removedToken)) {
                        recentTokensSet.remove(removedToken);
                    }
                }
            }

            // Notify via callback if provided
            if (onTokenGenerated != null) {
                onTokenGenerated.accept(nextToken);
            }

            // Check for stop condition
            if (stopTokens.contains(nextToken)) {
                break;
            }

            // Update for next iteration
            state.latestToken = currentToken = nextToken;
        }

        // Calculate and print performance metrics
        long endNanos = System.nanoTime();
        double totalTimeSeconds = (endNanos - startNanos) / 1_000_000_000.0;
        int totalTokens = promptIndex + generatedTokens.size();

        LastRunMetrics.setMetrics(totalTokens, totalTimeSeconds);

        return generatedTokens;
    }

    public static List<Integer> generateTokensPhi3(Model model, State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated) {

        long startNanos = System.nanoTime();
        if (maxTokens < 0 || model.configuration().contextLength() < maxTokens) {
            maxTokens = model.configuration().contextLength();
        }
        List<Integer> generatedTokens = new ArrayList<>(maxTokens);
        int token = state.latestToken; // BOS?
        int nextToken;
        int promptIndex = 0;
        ByteArrayOutputStream baos = new ByteArrayOutputStream(5);
        for (int position = startPosition; position < maxTokens; ++position) {

            model.forward(state, token, position);
            if (promptIndex < promptTokens.size()) {
                // Force-pick token from prompt.
                nextToken = promptTokens.get(promptIndex++);
                if (echo) {
                    System.out.println("NextToken: " + nextToken);
                    String decoded = model.tokenizer().decode(List.of(nextToken));
                    System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
                }
            } else {
                nextToken = sampler.sampleToken(state.logits);
                if (echo) {
                    // log inferred token
                    System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
                }
                generatedTokens.add(nextToken);
                if (onTokenGenerated != null) {
                    onTokenGenerated.accept(nextToken);
                }
                if (stopTokens.contains(nextToken)) {
                    break;
                }
            }
            state.latestToken = token = nextToken;
            if (position == 2000) {
                break;
            }
        }

        // Calculate and print performance metrics
        long endNanos = System.nanoTime();
        double totalTimeSeconds = (endNanos - startNanos) / 1_000_000_000.0;
        int totalTokens = promptIndex + generatedTokens.size();

        LastRunMetrics.setMetrics(totalTokens, totalTimeSeconds);

        return generatedTokens;

    }

    public static List<Integer> generateTokensGPULlama(Model model, State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
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
            //System.out.println("currentToken: " + currentToken);
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

    public static List<Integer> generateTokensGPUQwen3(Model model, State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated, TornadoVMMasterPlan tornadoVMPlan) {
        // Start timing the whole process
        long startNanos = System.nanoTime();
        long inferenceStartNanos = 0;

        // Pre-validate the max tokens to avoid checking in the loop
        int actualMaxTokens = Math.min(maxTokens > 0 ? maxTokens : model.configuration().contextLength(), model.configuration().contextLength());

        // Preallocate with expected capacity to avoid resizing
        List<Integer> generatedTokens = new ArrayList<>(Math.min(256, actualMaxTokens - promptTokens.size())); // Conservative estimate

        // Initialize token variables
        int currentToken = state.latestToken; // BOS?
        int nextToken = 0;
        int promptIndex = 0;

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

        for (int position = startPosition; position < maxTokens; ++position) {

            // Handle token processing
            if (promptIndex < promptTokens.size()) {
                // We're still processing the prompt tokens
                final int token = promptTokens.get(promptIndex);

                //System.out.println("Token: " + token);
                model.forward(state, token, position);

                promptIndex++;
                if (promptIndex < promptTokens.size()) {
                    continue;
                }
                if (echo) {
                    System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
                }
                // We have reached the last prompt token and computed the first response-token.
                // BUG FIX: Don't manually increment position - the for loop will do it!
                // The old code did position++ here, causing the loop's ++position to skip a position
            } else {
                // Mark the start of actual generation (after prompt processing)
                if (inferenceStartNanos == 0) {
                    inferenceStartNanos = System.nanoTime();
                }

                model.forward(state, currentToken, position);
            }

            // Sample the next token
            nextToken = sampler.sampleToken(state.wrapLogits);

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

            // Update for next iteration
            state.latestToken = currentToken = nextToken;
        }

        // Calculate and print performance metrics
        long endNanos = System.nanoTime();
        double totalTimeSeconds = (endNanos - startNanos) / 1_000_000_000.0;
        int totalTokens = promptIndex + generatedTokens.size();

        LastRunMetrics.setMetrics(totalTokens, totalTimeSeconds);

        return generatedTokens;
    }

    public static List<Integer> generateTokensGPUPhi3(Model model, State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated, TornadoVMMasterPlan tornadoVMPlan) {
        // Start timing the whole process
        long startNanos = System.nanoTime();
        long inferenceStartNanos = 0;

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
            // GPU Forward Pass
            FloatArray logits = InferenceCore.forwardTornadoVM(model, state, currentToken, pos, tornadoVMPlan);

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
}