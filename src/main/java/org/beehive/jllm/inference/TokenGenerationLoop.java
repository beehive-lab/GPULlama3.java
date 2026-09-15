package org.beehive.gpullama3.inference;

import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import org.beehive.gpullama3.auxiliary.RunMetrics;
import org.beehive.gpullama3.backend.cpu.InferenceCore;
import org.beehive.gpullama3.backend.cpu.InferenceCoreBatchPrefillDecode;
import org.beehive.gpullama3.backend.cpu.InferenceCoreWithPrefillDecode;
import org.beehive.gpullama3.backend.tornado.TornadoVMMasterPlan;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tokenizer.Tokenizer;

/**
 * The token-generation loop: prompt ingestion, then generation until a stop condition.
 *
 * <h2>The name</h2>
 *
 * <p>This was {@code InferenceEngine}, and "engine" had come to mean three different things in one
 * codebase. It is <b>none</b> of the other two:
 *
 * <ul>
 *   <li>{@code engine.LLMEngine} is the request scheduler — admission, queueing, batching across
 *       requests. It owns <i>who runs next</i>.
 *   <li>{@code generation.ModelGeneration} is generation policy — prompts, chat formatting, console
 *       I/O, the CLI's turn structure. It owns <i>what a turn is</i>.
 *   <li>This class owns <b>one sequence's loop</b>: ingest its prompt, forward, sample, stop. It
 *       schedules nothing and formats nothing.
 * </ul>
 *
 * <p>It provides unified logic for the following methods:
 *
 * <ul>
 *   <li>{@link #generateTokensLlama} – for LLaMA and Mistral models running on CPU
 *   <li>{@link #generateTokensGPULlama} – for LLaMA and Mistral models executed on GPU
 *   <li>{@link #generateTokensQwen3} – for Qwen3 models running on CPU
 *   <li>{@link #generateTokensGPUQwen3} – for Qwen3 models executed on GPU
 * </ul>
 */
public final class TokenGenerationLoop {

    /**
     * The host forward pass for a model, resolved through provider discovery.
     *
     * <p>This layer selects the backend — it always did, it just used to let the model re-derive
     * the choice from {@code plan == null}. The host sites call this; the accelerator sites call
     * {@code TornadoForwardPass} with the plan they already hold.
     *
     * <p>Cached per model, because resolution walks the discovered providers and the answer cannot
     * change for a loaded model. Nothing here runs per token beyond a map lookup on a small map,
     * and the resolved pass is what the token loop calls.
     */
    private static final java.util.Map<
                    org.beehive.gpullama3.runtime.model.ArchitectureId, ForwardPass>
            HOST_FORWARD = new java.util.concurrent.ConcurrentHashMap<>();

    /** The loop's logits as a sampler sees them. */
    private static Logits asLogits(Object logits) {
        if (logits instanceof org.beehive.gpullama3.tensor.standard.FloatTensor tensor) {
            return Logits.of(tensor);
        }
        throw new IllegalArgumentException(
                "Unsupported logits type: "
                        + (logits != null ? logits.getClass().getName() : "null"));
    }

    private static ForwardPass hostForward(Model model) {
        return HOST_FORWARD.computeIfAbsent(
                model.architectureId(),
                org.beehive.gpullama3.backend.cpu.CpuForwardPasses::forArchitecture);
    }

    /**
     * Benchmarking aid: keep decoding past the stop token so every run generates the same token
     * count.
     */
    private static final boolean IGNORE_EOS = Boolean.getBoolean("llama.bench.ignoreEos");

    private TokenGenerationLoop() {
        // prevent instantiation
    }

    /**
     * Greedy next-token pick for the GPU FP16 path. When the session's policy puts sampling on the
     * device, the argmax already ran there and the token id sits in {@code
     * state.workspace.sampledToken}; read it directly (the full logits row never left the GPU).
     * Otherwise fall back to the host sampler over the transferred logits.
     *
     * <p>{@code deviceSample} is passed in rather than read here: it is resolved <b>once per
     * generation</b>, outside the loop. Reading the policy per token would be the regression this
     * migration is measured against.
     */
    private static int sampleTokenGpu(
            State state,
            Sampler sampler,
            Logits logits,
            org.beehive.gpullama3.backend.tornado.lowering.InvocationBoundary.Result result,
            boolean deviceSample) {
        if (result != null) {
            // A lowered invocation already carries whatever the device decided, in storage this
            // session owns. Reading the domain's sampled-token array here would read a domain-owned
            // array after
            // the lock was released — exactly the escape the boundary exists to prevent.
            return result.hasSampledToken() ? result.sampledToken() : sampler.sampleToken(logits);
        }
        if (deviceSample) {
            return state.workspace.deviceSampledToken();
        }
        return sampler.sampleToken(logits);
    }

    /**
     * LLM generation entry point, ingest prompt tokens and generates new tokens.
     *
     * <p>All prompt tokens are ingested first, then inference starts, until a stop token is found.
     * The returned tokens only include generated/inferred tokens.
     *
     * @param model model to run inference (including weights, configuration, tokenizer.)
     * @param state state of the model e.g. key/value caches. this is mutated by this call
     * @param startPosition start prompt ingestion + inference at this position in the context e.g.
     *     useful if state was kept across calls (chained generation). 0 implies run with no
     *     previous context.
     * @param promptTokens prompt tokens to ingest, all the prompt tokens will be ingested, given
     *     there's enough capacity left in the context
     * @param stopTokens set of tokens that abort generation during inference, stop tokens do not
     *     affect prompt ingestion
     * @param maxTokens maximum number of tokens (can go up to {@link Configuration#contextLength
     *     context length} if this value is negative or greater than {@link
     *     Configuration#contextLength context length}
     * @param sampler {@link Sampler strategy} used to select tokens
     * @param echo debugging flag, prints ALL, prompt and inferred tokens, to {@link System#err
     *     stderr}
     * @param onTokenGenerated callback, if non-null, it's called every time a token is inferred
     *     e.g. it's not called when ingesting prompt tokens
     * @return list of generated/inferred tokens, including the stop token, if any e.g. does not
     *     include any token from the prompt
     */
    /** The host loop, with the prompt ingestion selected from the session's policy. */
    public static List<Integer> generateTokensLlamaForPolicy(
            Model model,
            State state,
            int startPosition,
            List<Integer> promptTokens,
            Set<Integer> stopTokens,
            int maxTokens,
            Sampler sampler,
            boolean echo,
            IntConsumer onTokenGenerated) {
        if (state.executionPolicy().phaseStrategy()
                == org.beehive.gpullama3.runtime.policy.ExecutionPolicy.PhaseStrategy
                        .PREFILL_DECODE) {
            return state.executionPolicy().prefillBatchSize() > 1
                    ? generateLlamaCpuWithBatchPrefill(
                            model,
                            state,
                            startPosition,
                            promptTokens,
                            stopTokens,
                            maxTokens,
                            sampler,
                            echo,
                            onTokenGenerated)
                    : generateLlamaCpuWithPrefill(
                            model,
                            state,
                            startPosition,
                            promptTokens,
                            stopTokens,
                            maxTokens,
                            sampler,
                            echo,
                            onTokenGenerated);
        }
        return generateTokensLlama(
                model,
                state,
                startPosition,
                promptTokens,
                stopTokens,
                maxTokens,
                sampler,
                echo,
                onTokenGenerated);
    }

    /**
     * Unchanged. It exists — a real host implementation, not a stub — which is worth saying because
     * collapsing the family's dispatch into "batched prefill has no CPU path" would have been a
     * regression, and nearly was.
     */
    private static List<Integer> generateLlamaCpuWithBatchPrefill(
            Model model,
            State state,
            int startPosition,
            List<Integer> promptTokens,
            Set<Integer> stopTokens,
            int maxTokens,
            Sampler sampler,
            boolean echo,
            IntConsumer onTokenGenerated) {

        long startNanos = System.nanoTime();

        final Configuration config = model.configuration();
        int actualMaxTokens =
                (maxTokens < 0 || config.contextLength() < maxTokens)
                        ? config.contextLength()
                        : maxTokens;
        // Resolved once per generation from the session's policy, not from a class constant
        // The plan was built for this value; reading a different one here would batch
        // against a graph sized for something else.
        final int batchSize = state.executionPolicy().prefillBatchSize();

        List<Integer> generatedTokens = new ArrayList<>();

        PromptIngestion ingestion = PromptIngestion.of(state, promptTokens, startPosition);
        int currentToken = ingestion.firstToken(); // BOS
        int pos = startPosition;
        int N = promptTokens.size();

        // ── Prefill ───────────────────────────────────────────────────────────
        if (N > 0 && pos < actualMaxTokens) {
            // Build the token sequence at positions [startPosition. startPosition+N-1].
            //
            // Two shapes, decided by PromptIngestion. When the state's seed is the token the
            // prompt already opens with, the sequence is simply the prompt — feeding the seed as
            // well is the duplication PromptIngestion exists to remove. Otherwise the seed is a
            // real first token (a start header, for families that use one) and leads the prompt.
            int[] prefillSeq = new int[N];
            if (ingestion.firstIndex() == 1) {
                for (int i = 0; i < N; i++) {
                    prefillSeq[i] = promptTokens.get(i);
                }
            } else {
                prefillSeq[0] = currentToken;
                for (int i = 1; i < N; i++) {
                    prefillSeq[i] = promptTokens.get(i - 1);
                }
            }

            for (int chunkStart = 0;
                    chunkStart < N && pos + chunkStart < actualMaxTokens;
                    chunkStart += batchSize) {
                int chunkEnd = Math.min(Math.min(chunkStart + batchSize, N), actualMaxTokens - pos);
                int chunkSize = chunkEnd - chunkStart;
                int[] chunk = Arrays.copyOfRange(prefillSeq, chunkStart, chunkEnd);

                if (chunkSize == 1) {
                    InferenceCoreWithPrefillDecode.forwardJavaPrefill(
                            model, state, chunk[0], pos + chunkStart);
                } else {
                    InferenceCoreBatchPrefillDecode.batchForwardJavaPrefill(
                            model, state, chunk, pos + chunkStart, chunkSize);
                }

                if (echo) {
                    for (int b = 0; b < chunkSize; b++) {
                        int echoed = promptTokens.get(Math.min(chunkStart + b, N - 1));
                        System.err.print(
                                Tokenizer.replaceControlCharacters(
                                        model.tokenizer().decode(List.of(echoed))));
                    }
                }
            }

            currentToken = promptTokens.get(N - 1);
            pos = startPosition + N;
        }

        state.latestToken = currentToken;
        long decodeStartNanos = System.nanoTime();

        // ── Decode ────────────────────────────────────────────────────────────
        while (pos < actualMaxTokens) {
            var logits = InferenceCore.forwardJava(model, state, currentToken, pos);
            int nextToken = sampler.sampleToken(asLogits(logits));

            if (echo) {
                System.err.print(
                        Tokenizer.replaceControlCharacters(
                                model.tokenizer().decode(List.of(nextToken))));
            }

            generatedTokens.add(nextToken);

            if (onTokenGenerated != null) {
                onTokenGenerated.accept(nextToken);
            }

            if (!IGNORE_EOS && stopTokens.contains(nextToken)) {
                break;
            }

            currentToken = nextToken;
            state.latestToken = currentToken;
            pos++;
        }

        long endNanos = System.nanoTime();
        RunMetrics.setInferenceMetrics(
                promptTokens.size(),
                decodeStartNanos - startNanos,
                generatedTokens.size(),
                endNanos - decodeStartNanos,
                endNanos - startNanos);
        RunMetrics.setHasPrefillPhase(true);

        return generatedTokens;
    }

    private static List<Integer> generateLlamaCpuWithPrefill(
            Model model,
            State state,
            int startPosition,
            List<Integer> promptTokens,
            Set<Integer> stopTokens,
            int maxTokens,
            Sampler sampler,
            boolean echo,
            IntConsumer onTokenGenerated) {

        long startNanos = System.nanoTime();

        final Configuration config = model.configuration();
        int actualMaxTokens =
                (maxTokens < 0 || config.contextLength() < maxTokens)
                        ? config.contextLength()
                        : maxTokens;

        List<Integer> generatedTokens = new ArrayList<>();

        PromptIngestion ingestion = PromptIngestion.of(state, promptTokens, startPosition);
        int currentToken = ingestion.firstToken(); // BOS
        int pos = startPosition;
        int N = promptTokens.size();

        // ── Prefill ───────────────────────────────────────────────────────────
        if (N > 0 && pos < actualMaxTokens) {
            for (int promptIndex = ingestion.firstIndex();
                    promptIndex < N && pos < actualMaxTokens;
                    promptIndex++) {
                InferenceCoreWithPrefillDecode.forwardJavaPrefill(model, state, currentToken, pos);
                currentToken = promptTokens.get(promptIndex);
                if (echo) {
                    System.err.print(
                            Tokenizer.replaceControlCharacters(
                                    model.tokenizer().decode(List.of(currentToken))));
                }
                pos++;
            }
        }

        state.latestToken = currentToken;
        long decodeStartNanos = System.nanoTime();

        // ── Decode ────────────────────────────────────────────────────────────
        while (pos < actualMaxTokens) {
            var logits = InferenceCore.forwardJava(model, state, currentToken, pos);
            int nextToken = sampler.sampleToken(asLogits(logits));

            if (echo) {
                System.err.print(
                        Tokenizer.replaceControlCharacters(
                                model.tokenizer().decode(List.of(nextToken))));
            }

            generatedTokens.add(nextToken);

            if (onTokenGenerated != null) {
                onTokenGenerated.accept(nextToken);
            }

            if (!IGNORE_EOS && stopTokens.contains(nextToken)) {
                break;
            }

            currentToken = nextToken;
            state.latestToken = currentToken;
            pos++;
        }

        long endNanos = System.nanoTime();
        RunMetrics.setInferenceMetrics(
                promptTokens.size(),
                decodeStartNanos - startNanos,
                generatedTokens.size(),
                endNanos - decodeStartNanos,
                endNanos - startNanos);
        RunMetrics.setHasPrefillPhase(true);

        return generatedTokens;
    }

    public static List<Integer> generateTokensLlama(
            Model model,
            State state,
            int startPosition,
            List<Integer> promptTokens,
            Set<Integer> stopTokens,
            int maxTokens,
            Sampler sampler,
            boolean echo,
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
        PromptIngestion ingestion = PromptIngestion.of(state, promptTokens, startPosition);
        int currentToken = ingestion.firstToken();
        int nextToken;
        int promptIndex = ingestion.firstIndex();
        int pos = startPosition;

        while (pos < maxTokens) {

            // Through the provider SPI, as generateTokensQwen3 and generateTokensPhi3 already do.
            // This loop serves llama, mistral and devstral; llama's and mistral's providers both
            // resolve to InferenceCore::forwardJava, the method that was called directly here, so
            // neither changes. Devstral's resolves to forwardJavaDevstral — and calling forwardJava
            // for it was wrong: Devstral's head dimension is independent of dim/numberOfHeads
            // (5120 embedding, 32 heads, 128 head_dim), so Llama's forward pass wrote a dim-sized
            // projection into the qDim-sized buffer and died with
            // "Index 4640 out of bounds for length 4096" on the first real fixture (Metal parity
            // task 12). The registered DevstralCpuForwardProvider was never consulted on this path.
            hostForward(model).forward(model, state, currentToken, pos);

            // Handle token processing
            if (promptIndex < promptTokens.size()) {
                // We're still processing the prompt tokens
                nextToken = promptTokens.get(promptIndex++);
                if (echo) {
                    System.err.print(
                            Tokenizer.replaceControlCharacters(
                                    model.tokenizer().decode(List.of(nextToken))));
                }
            } else {
                // Mark the start of actual generation (after prompt processing)
                if (inferenceStartNanos == 0) {
                    inferenceStartNanos = System.nanoTime();
                }

                // Sample the next token
                nextToken = sampler.sampleToken(asLogits(state.logits));

                // Output the token if echo is enabled
                if (echo) {
                    System.err.print(
                            Tokenizer.replaceControlCharacters(
                                    model.tokenizer().decode(List.of(nextToken))));
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
        long decodeStart = inferenceStartNanos > 0 ? inferenceStartNanos : endNanos;
        RunMetrics.setInferenceMetrics(
                promptIndex,
                decodeStart - startNanos,
                generatedTokens.size(),
                endNanos - decodeStart,
                endNanos - startNanos);

        return generatedTokens;
    }

    public static List<Integer> generateTokensQwen3(
            Model model,
            State state,
            int startPosition,
            List<Integer> promptTokens,
            Set<Integer> stopTokens,
            int maxTokens,
            Sampler sampler,
            boolean echo,
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
        int generatedTokenBudget = Math.max(0, maxTokens - startPosition - promptTokens.size());

        // Initialize token variables
        PromptIngestion ingestion = PromptIngestion.of(state, promptTokens, startPosition);
        int currentToken = ingestion.firstToken(); // BOS?
        int nextToken = 0;
        int promptIndex = 0;

        for (int position = startPosition; position < maxTokens; ++position) {

            // Handle token processing
            if (promptIndex < promptTokens.size()) {
                // We're still processing the prompt tokens
                final int token = promptTokens.get(promptIndex);

                hostForward(model).forward(model, state, token, position);

                promptIndex++;
                if (promptIndex < promptTokens.size()) {
                    continue;
                }
                if (echo) {
                    System.err.print(
                            Tokenizer.replaceControlCharacters(
                                    model.tokenizer().decode(List.of(nextToken))));
                }
                // The last prompt token produced the first response-token logits.
                // The for-loop advances to the next sequence position.
                if (generatedTokenBudget == 0) {
                    break;
                }
            } else {
                // Mark the start of actual generation (after prompt processing)
                if (inferenceStartNanos == 0) {
                    inferenceStartNanos = System.nanoTime();
                }

                hostForward(model).forward(model, state, currentToken, position);
            }

            // Sample the next token
            nextToken = sampler.sampleToken(asLogits(state.logits));

            // Output the token if echo is enabled
            if (echo) {
                System.err.print(
                        Tokenizer.replaceControlCharacters(
                                model.tokenizer().decode(List.of(nextToken))));
            }

            // Track the generated token
            generatedTokens.add(nextToken);

            // Notify via callback if provided
            if (onTokenGenerated != null) {
                onTokenGenerated.accept(nextToken);
            }

            // Check for stop condition
            if (generatedTokens.size() >= generatedTokenBudget || stopTokens.contains(nextToken)) {
                break;
            }

            // Update for next iteration
            state.latestToken = currentToken = nextToken;
        }

        // Calculate and print performance metrics
        long endNanos = System.nanoTime();
        long decodeStart = inferenceStartNanos > 0 ? inferenceStartNanos : endNanos;
        RunMetrics.setInferenceMetrics(
                promptIndex,
                decodeStart - startNanos,
                generatedTokens.size(),
                endNanos - decodeStart,
                endNanos - startNanos);

        return generatedTokens;
    }

    public static List<Integer> generateTokensPhi3(
            Model model,
            State state,
            int startPosition,
            List<Integer> promptTokens,
            Set<Integer> stopTokens,
            int maxTokens,
            Sampler sampler,
            boolean echo,
            IntConsumer onTokenGenerated) {

        long startNanos = System.nanoTime();
        long inferenceStartNanos = 0;
        if (maxTokens < 0 || model.configuration().contextLength() < maxTokens) {
            maxTokens = model.configuration().contextLength();
        }
        List<Integer> generatedTokens = new ArrayList<>(maxTokens);
        int token = state.latestToken; // BOS?
        int nextToken;
        int promptIndex = 0;
        ByteArrayOutputStream baos = new ByteArrayOutputStream(5);
        for (int position = startPosition; position < maxTokens; ++position) {

            hostForward(model).forward(model, state, token, position);
            if (promptIndex < promptTokens.size()) {
                // Force-pick token from prompt.
                nextToken = promptTokens.get(promptIndex++);
                if (echo) {
                    System.out.println("NextToken: " + nextToken);
                    String decoded = model.tokenizer().decode(List.of(nextToken));
                    System.err.print(
                            Tokenizer.replaceControlCharacters(
                                    model.tokenizer().decode(List.of(nextToken))));
                }
            } else {
                if (inferenceStartNanos == 0) {
                    inferenceStartNanos = System.nanoTime();
                }
                nextToken = sampler.sampleToken(asLogits(state.logits));
                if (echo) {
                    // log inferred token
                    System.err.print(
                            Tokenizer.replaceControlCharacters(
                                    model.tokenizer().decode(List.of(nextToken))));
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
        long decodeStart = inferenceStartNanos > 0 ? inferenceStartNanos : endNanos;
        RunMetrics.setInferenceMetrics(
                promptIndex,
                decodeStart - startNanos,
                generatedTokens.size(),
                endNanos - decodeStart,
                endNanos - startNanos);

        return generatedTokens;
    }

    /**
     * The legacy entry point: the cursor is the state's own {@code latestToken}.
     *
     * <p>Unchanged behaviour for every path but the lowered one, including families whose initial
     * seed is not repeated in the prompt — they depend on the seed being whatever {@code
     * createNewState} put in the state, and {@link GenerationCursor#forState} is exactly that.
     */
    public static List<Integer> generateTokensGPULlama(
            Model model,
            State state,
            int startPosition,
            List<Integer> promptTokens,
            Set<Integer> stopTokens,
            int maxTokens,
            Sampler sampler,
            boolean echo,
            IntConsumer onTokenGenerated,
            TornadoVMMasterPlan tornadoVMPlan) {
        return generateTokensGPULlama(
                model,
                state,
                GenerationCursor.forState(state),
                startPosition,
                promptTokens,
                stopTokens,
                maxTokens,
                sampler,
                echo,
                onTokenGenerated,
                tornadoVMPlan);
    }

    /**
     * As above, with the continuation seed and the produced token carried by a {@link
     * GenerationCursor} rather than by the state.
     *
     * <p>That is what lets a session execute in a workspace it shares: its conversation history
     * travels with it instead of living in storage the next session overwrites.
     */
    public static List<Integer> generateTokensGPULlama(
            Model model,
            State state,
            GenerationCursor cursor,
            int startPosition,
            List<Integer> promptTokens,
            Set<Integer> stopTokens,
            int maxTokens,
            Sampler sampler,
            boolean echo,
            IntConsumer onTokenGenerated,
            TornadoVMMasterPlan tornadoVMPlan) {
        if (state.executionPolicy().phaseStrategy()
                == org.beehive.gpullama3.runtime.policy.ExecutionPolicy.PhaseStrategy
                        .PREFILL_DECODE) {
            return generateWithPrefill(
                    model,
                    state,
                    cursor,
                    startPosition,
                    promptTokens,
                    stopTokens,
                    maxTokens,
                    sampler,
                    echo,
                    onTokenGenerated,
                    tornadoVMPlan);
        }
        return generateInterleaved(
                model,
                state,
                cursor,
                startPosition,
                promptTokens,
                stopTokens,
                maxTokens,
                sampler,
                echo,
                onTokenGenerated,
                tornadoVMPlan);
    }

    /**
     * Prompt ingestion as a separate phase, then decode — the shape both deprecated {@code
     * InferenceEngineWith*} classes had.
     *
     * <p>They differed only in how the prompt is ingested: one token at a time through {@code
     * TornadoPrefillPass.prefill}, or in chunks of {@code prefillBatchSize} through {@code
     * TornadoBatchPrefillPass.batchPrefill}. <b>The decode loop was copied</b>, and the copies had
     * drifted: the batch one carries a generated-token budget the sequential one does not, and only
     * it handles the Qwen2-MoE seed. Both behaviours are preserved here, in one place, where the
     * difference is visible instead of being spread across two files.
     */
    private static List<Integer> generateWithPrefill(
            Model model,
            State state,
            GenerationCursor cursor,
            int startPosition,
            List<Integer> promptTokens,
            Set<Integer> stopTokens,
            int maxTokens,
            Sampler sampler,
            boolean echo,
            IntConsumer onTokenGenerated,
            TornadoVMMasterPlan tornadoVMPlan) {
        long startNanos = System.nanoTime();
        final Configuration config = model.configuration();
        int actualMaxTokens =
                (maxTokens < 0 || config.contextLength() < maxTokens)
                        ? config.contextLength()
                        : maxTokens;

        int batchSize = state.executionPolicy().prefillBatchSize();
        boolean batched = batchSize > 1;
        List<Integer> generatedTokens = new ArrayList<>();

        PromptIngestion ingestion = PromptIngestion.of(cursor.seed(), promptTokens, startPosition);
        int currentToken = ingestion.firstToken();
        int pos = startPosition;
        int promptSize = promptTokens.size();
        int generatedTokenBudget = Integer.MAX_VALUE;

        if (batched) {
            var plan =
                    (org.beehive.gpullama3.backend.tornado.TornadoVMMasterPlanBatchPrefillDecode)
                            tornadoVMPlan;
            // This branch prefilled all N whenever the seed was the prompt's own first token, then
            // decoded starting from the last prompt token again. The model therefore saw that
            // token twice: once in the key/value cache at position N-1, and once more as the
            // decode input at position N. Every compared row was wrong from the first, at any
            // batch size and on both kernel families, because the *sequence* was wrong rather than
            // the arithmetic.
            //
            // Qwen2-MoE already had the right shape for its own reason — its seed repeats the
            // prompt's first token — so the two cases are one case, and `seedIsPromptHead` is the
            // condition both were expressing.
            boolean qwen2MoE =
                    model.getModelType() == org.beehive.gpullama3.model.ModelType.QWEN_2_MOE;
            boolean seedIsPromptHead = qwen2MoE || ingestion.firstIndex() == 1;
            int prefillTokenCount = seedIsPromptHead ? Math.max(0, promptSize - 1) : promptSize;
            int[] prefillSeq = new int[prefillTokenCount];
            if (seedIsPromptHead) {
                for (int i = 0; i < prefillTokenCount; i++) {
                    prefillSeq[i] = promptTokens.get(i);
                }
            } else {
                // The seed is a real first token the prompt does not repeat, so it leads: the
                // prefill is seed + tokens[0. N-2], again stopping one short.
                prefillSeq[0] = currentToken;
                for (int i = 1; i < promptSize; i++) {
                    prefillSeq[i] = promptTokens.get(i - 1);
                }
            }

            for (int chunkStart = 0;
                    chunkStart < prefillTokenCount && pos + chunkStart < actualMaxTokens;
                    chunkStart += batchSize) {
                int chunkEnd =
                        Math.min(
                                Math.min(chunkStart + batchSize, prefillTokenCount),
                                actualMaxTokens - pos);
                int chunkSize = chunkEnd - chunkStart;
                int[] chunk = java.util.Arrays.copyOfRange(prefillSeq, chunkStart, chunkEnd);
                org.beehive.gpullama3.backend.tornado.TornadoBatchPrefillPass.batchPrefill(
                        model, state, chunk, pos + chunkStart, chunkSize, plan);
                if (echo) {
                    for (int b = 0; b < chunkSize; b++) {
                        int echoed = promptTokens.get(Math.min(chunkStart + b, promptSize - 1));
                        System.err.print(
                                Tokenizer.replaceControlCharacters(
                                        model.tokenizer().decode(List.of(echoed))));
                    }
                }
            }
            currentToken = promptTokens.get(promptSize - 1);
            // Decode resumes at the position after the last one prefilled — derived from what was
            // actually prefilled rather than from the prompt size, which is what let the two drift
            // apart in the first place.
            pos = startPosition + prefillTokenCount;
            // Derived from the same count, so the budget stops one short of nothing: the decode
            // loop now runs to actualMaxTokens exactly as the sequential branch does, and a batched
            // run emits the same number of tokens as a single-token run instead of one fewer. That
            // missing token was the same off-by-one, seen from the other end.
            generatedTokenBudget = Math.max(0, actualMaxTokens - startPosition - prefillTokenCount);
        } else {
            var plan =
                    (org.beehive.gpullama3.backend.tornado.TornadoVMMasterPlanPrefillDecode)
                            tornadoVMPlan;
            for (int promptIndex = ingestion.firstIndex();
                    promptIndex < promptSize && pos < actualMaxTokens;
                    promptIndex++) {
                org.beehive.gpullama3.backend.tornado.TornadoPrefillPass.prefill(
                        model, state, currentToken, pos, plan);
                currentToken = promptTokens.get(promptIndex);
                if (echo) {
                    System.err.print(
                            Tokenizer.replaceControlCharacters(
                                    model.tokenizer().decode(List.of(currentToken))));
                }
                pos++;
            }
        }

        cursor.advance(currentToken);
        long decodeStartNanos = System.nanoTime();

        // ── Decode: one loop, whichever prefill ran ───────────────────────────
        boolean deviceSample =
                state.executionPolicy().samplingResidency()
                        == org.beehive.gpullama3.runtime.policy.ExecutionPolicy.SamplingResidency
                                .DEVICE;
        while (pos < actualMaxTokens && generatedTokens.size() < generatedTokenBudget) {
            Logits logits =
                    batched
                            ? org.beehive.gpullama3.backend.tornado.TornadoBatchPrefillPass.decode(
                                    model,
                                    state,
                                    currentToken,
                                    pos,
                                    (org.beehive.gpullama3.backend.tornado
                                                    .TornadoVMMasterPlanBatchPrefillDecode)
                                            tornadoVMPlan)
                            : org.beehive.gpullama3.backend.tornado.TornadoForwardPass.forward(
                                    model, state, currentToken, pos, tornadoVMPlan);
            int nextToken = sampleTokenGpu(state, sampler, logits, null, deviceSample);

            if (echo) {
                System.err.print(
                        Tokenizer.replaceControlCharacters(
                                model.tokenizer().decode(List.of(nextToken))));
            }
            generatedTokens.add(nextToken);
            if (onTokenGenerated != null) {
                onTokenGenerated.accept(nextToken);
            }
            if (!IGNORE_EOS && stopTokens.contains(nextToken)) {
                break;
            }
            currentToken = nextToken;
            cursor.advance(currentToken);
            pos++;
        }

        long endNanos = System.nanoTime();
        RunMetrics.setInferenceMetrics(
                promptTokens.size(),
                decodeStartNanos - startNanos,
                generatedTokens.size(),
                endNanos - decodeStartNanos,
                endNanos - startNanos);
        RunMetrics.setHasPrefillPhase(true);
        return generatedTokens;
    }

    /** The single-token shape: prompt and generation in one loop. */
    private static List<Integer> generateInterleaved(
            Model model,
            State state,
            GenerationCursor cursor,
            int startPosition,
            List<Integer> promptTokens,
            Set<Integer> stopTokens,
            int maxTokens,
            Sampler sampler,
            boolean echo,
            IntConsumer onTokenGenerated,
            TornadoVMMasterPlan tornadoVMPlan) {
        // === Setup and Initialization ===
        long startNanos = System.nanoTime();
        long inferenceStartNanos = 0;

        // Pre-validate the max tokens to avoid checking in the loop
        int actualMaxTokens =
                Math.min(
                        maxTokens > 0 ? maxTokens : model.configuration().contextLength(),
                        model.configuration().contextLength());

        // Preallocate with expected capacity to avoid resizing. Clamp at 0: when the
        // prompt is longer than the token budget (actualMaxTokens), the difference is
        // negative and would throw IllegalArgumentException("Illegal Capacity").
        List<Integer> generatedTokens =
                new ArrayList<>(
                        Math.max(
                                0,
                                Math.min(
                                        256,
                                        actualMaxTokens
                                                - promptTokens.size()))); // Conservative estimate

        // === Token Generation Loop ===
        PromptIngestion ingestion = PromptIngestion.of(cursor.seed(), promptTokens, startPosition);
        int currentToken = ingestion.firstToken();
        int nextToken;
        int promptIndex = ingestion.firstIndex();
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

        // Resolved once for the whole generation, never per token. It was
        // a static final read from a system property at class initialization, which is what made
        // the CLI clear the property before the layer class could load.
        boolean deviceSample =
                state.executionPolicy().samplingResidency()
                        == org.beehive.gpullama3.runtime.policy.ExecutionPolicy.SamplingResidency
                                .DEVICE;

        // Main generation loop
        while (pos < actualMaxTokens) {
            // GPU Forward Pass - No conditional check since we know we're using GPU
            // System.out.println("currentToken: " + currentToken);
            Logits logits =
                    org.beehive.gpullama3.backend.tornado.TornadoForwardPass.forward(
                            model, state, currentToken, pos, tornadoVMPlan);

            // Process prompt tokens if still remaining
            if (promptIndex < promptTokens.size()) {
                // Get next prompt token (using array access if available)
                nextToken =
                        promptTokenArray != null
                                ? promptTokenArray[promptIndex++]
                                : promptTokens.get(promptIndex++);

                if (echo) {
                    // Decode and output token
                    System.err.print(
                            Tokenizer.replaceControlCharacters(
                                    model.tokenizer().decode(List.of(nextToken))));
                }
            } else {
                // Mark first inference token
                if (inferenceStartNanos == 0) {
                    inferenceStartNanos = System.nanoTime();
                }

                // Sample next token - use GPU sampling if available
                nextToken = sampleTokenGpu(state, sampler, logits, null, deviceSample);

                // Add token consumer support
                if (onTokenGenerated != null) {
                    onTokenGenerated.accept(nextToken);
                }

                // Output if needed
                if (echo && onTokenGenerated == null) {
                    System.err.print(
                            Tokenizer.replaceControlCharacters(
                                    model.tokenizer().decode(List.of(nextToken))));
                }

                // Store token
                generatedTokens.add(nextToken);

                // Check stop condition
                if (!IGNORE_EOS && stopTokens.contains(nextToken)) {
                    break;
                }
            }

            // Update for next iteration
            currentToken = nextToken;
            cursor.advance(currentToken);
            pos++;
        }

        // === Performance Metrics ===
        long endNanos = System.nanoTime();
        long decodeStart = inferenceStartNanos > 0 ? inferenceStartNanos : endNanos;
        RunMetrics.setInferenceMetrics(
                promptIndex,
                decodeStart - startNanos,
                generatedTokens.size(),
                endNanos - decodeStart,
                endNanos - startNanos);

        return generatedTokens;
    }

    public static List<Integer> generateTokensGPUQwen3(
            Model model,
            State state,
            int startPosition,
            List<Integer> promptTokens,
            Set<Integer> stopTokens,
            int maxTokens,
            Sampler sampler,
            boolean echo,
            IntConsumer onTokenGenerated,
            TornadoVMMasterPlan tornadoVMPlan) {
        // Start timing the whole process
        long startNanos = System.nanoTime();
        long inferenceStartNanos = 0;

        // Pre-validate the max tokens to avoid checking in the loop
        int actualMaxTokens =
                Math.min(
                        maxTokens > 0 ? maxTokens : model.configuration().contextLength(),
                        model.configuration().contextLength());

        // Preallocate with expected capacity to avoid resizing. Clamp at 0: when the
        // prompt is longer than the token budget (actualMaxTokens), the difference is
        // negative and would throw IllegalArgumentException("Illegal Capacity").
        List<Integer> generatedTokens =
                new ArrayList<>(
                        Math.max(
                                0,
                                Math.min(
                                        256,
                                        actualMaxTokens
                                                - promptTokens.size()))); // Conservative estimate
        int generatedTokenBudget =
                Math.max(0, actualMaxTokens - startPosition - promptTokens.size());

        // Initialize token variables
        PromptIngestion ingestion = PromptIngestion.of(state, promptTokens, startPosition);
        int currentToken = ingestion.firstToken(); // BOS?
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

        var boundary =
                tornadoVMPlan
                                instanceof
                                org.beehive.gpullama3.backend.tornado.lowering.InvocationBoundary b
                        ? b
                        : null;
        org.beehive.gpullama3.backend.tornado.lowering.InvocationBoundary.Result lastResult = null;
        Logits lastLogits = null;

        // Resolved once for the whole generation, never per token.
        boolean deviceSample =
                state.executionPolicy().samplingResidency()
                        == org.beehive.gpullama3.runtime.policy.ExecutionPolicy.SamplingResidency
                                .DEVICE;

        for (int position = startPosition; position < maxTokens; ++position) {

            // Handle token processing
            if (promptIndex < promptTokens.size()) {
                // We're still processing the prompt tokens
                final int token = promptTokens.get(promptIndex);

                // System.out.println("Token: " + token);
                if (boundary != null) {
                    lastResult = boundary.invoke(token, position);
                    lastLogits = lastResult.logits();
                } else {
                    lastLogits =
                            org.beehive.gpullama3.backend.tornado.TornadoForwardPass.forward(
                                    model, state, token, position, tornadoVMPlan);
                }

                promptIndex++;
                if (promptIndex < promptTokens.size()) {
                    continue;
                }
                if (echo) {
                    System.err.print(
                            Tokenizer.replaceControlCharacters(
                                    model.tokenizer().decode(List.of(nextToken))));
                }
                // The last prompt token produced the first response-token logits.
                // The for-loop advances to the next sequence position.
                if (generatedTokenBudget == 0) {
                    break;
                }
            } else {
                // Mark the start of actual generation (after prompt processing)
                if (inferenceStartNanos == 0) {
                    inferenceStartNanos = System.nanoTime();
                }

                if (boundary != null) {
                    lastResult = boundary.invoke(currentToken, position);
                    lastLogits = lastResult.logits();
                } else {
                    lastLogits =
                            org.beehive.gpullama3.backend.tornado.TornadoForwardPass.forward(
                                    model, state, currentToken, position, tornadoVMPlan);
                }
            }

            nextToken = sampleTokenGpu(state, sampler, lastLogits, lastResult, deviceSample);

            // Output the token if echo is enabled
            if (echo) {
                System.err.print(
                        Tokenizer.replaceControlCharacters(
                                model.tokenizer().decode(List.of(nextToken))));
            }

            // Track the generated token
            generatedTokens.add(nextToken);

            // Notify via callback if provided
            if (onTokenGenerated != null) {
                onTokenGenerated.accept(nextToken);
            }

            // Check for stop condition
            // The budget is a hard limit and is never bypassed; only the stop token is, so that
            // an ignore-EOS benchmark run still terminates — on the budget instead of on EOS.
            if (generatedTokens.size() >= generatedTokenBudget
                    || (!IGNORE_EOS && stopTokens.contains(nextToken))) {
                break;
            }

            // Update for next iteration
            state.latestToken = currentToken = nextToken;
        }

        // Calculate and print performance metrics
        long endNanos = System.nanoTime();
        long decodeStart = inferenceStartNanos > 0 ? inferenceStartNanos : endNanos;
        RunMetrics.setInferenceMetrics(
                promptIndex,
                decodeStart - startNanos,
                generatedTokens.size(),
                endNanos - decodeStart,
                endNanos - startNanos);

        return generatedTokens;
    }

    public static List<Integer> generateTokensGPUPhi3(
            Model model,
            State state,
            int startPosition,
            List<Integer> promptTokens,
            Set<Integer> stopTokens,
            int maxTokens,
            Sampler sampler,
            boolean echo,
            IntConsumer onTokenGenerated,
            TornadoVMMasterPlan tornadoVMPlan) {
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
        PromptIngestion ingestion = PromptIngestion.of(state, promptTokens, startPosition);
        int currentToken = ingestion.firstToken();
        int nextToken;
        int promptIndex = ingestion.firstIndex();
        int pos = startPosition;

        while (pos < maxTokens) {
            // GPU Forward Pass
            Logits logits =
                    org.beehive.gpullama3.backend.tornado.TornadoForwardPass.forward(
                            model, state, currentToken, pos, tornadoVMPlan);

            // Handle token processing
            if (promptIndex < promptTokens.size()) {
                // We're still processing the prompt tokens
                nextToken = promptTokens.get(promptIndex++);
                if (echo) {
                    System.err.print(
                            Tokenizer.replaceControlCharacters(
                                    model.tokenizer().decode(List.of(nextToken))));
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
                    System.err.print(
                            Tokenizer.replaceControlCharacters(
                                    model.tokenizer().decode(List.of(nextToken))));
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
        long decodeStart = inferenceStartNanos > 0 ? inferenceStartNanos : endNanos;
        RunMetrics.setInferenceMetrics(
                promptIndex,
                decodeStart - startNanos,
                generatedTokens.size(),
                endNanos - decodeStart,
                endNanos - startNanos);

        return generatedTokens;
    }

    /**
     * Generates tokens using the Granite model with CPU inference. Identical pattern to
     * generateTokensLlama but calls forwardGranite.
     */
    public static List<Integer> generateTokensGranite(
            Model model,
            State state,
            int startPosition,
            List<Integer> promptTokens,
            Set<Integer> stopTokens,
            int maxTokens,
            Sampler sampler,
            boolean echo,
            IntConsumer onTokenGenerated) {
        long startNanos = System.nanoTime();
        long inferenceStartNanos = 0;

        Object logits;
        if (maxTokens < 0 || model.configuration().contextLength() < maxTokens) {
            maxTokens = model.configuration().contextLength();
        }

        List<Integer> generatedTokens = new ArrayList<>();

        PromptIngestion ingestion = PromptIngestion.of(state, promptTokens, startPosition);
        int currentToken = ingestion.firstToken();
        int nextToken;
        int promptIndex = ingestion.firstIndex();
        int pos = startPosition;

        while (pos < maxTokens) {
            // Call Granite-specific forward pass
            logits = InferenceCore.forwardGranite(model, state, currentToken, pos);

            if (promptIndex < promptTokens.size()) {
                nextToken = promptTokens.get(promptIndex++);
                if (echo) {
                    System.err.print(
                            Tokenizer.replaceControlCharacters(
                                    model.tokenizer().decode(List.of(nextToken))));
                }
            } else {
                if (inferenceStartNanos == 0) {
                    inferenceStartNanos = System.nanoTime();
                }

                nextToken = sampler.sampleToken(asLogits(logits));

                if (echo) {
                    System.err.print(
                            Tokenizer.replaceControlCharacters(
                                    model.tokenizer().decode(List.of(nextToken))));
                }

                generatedTokens.add(nextToken);

                if (onTokenGenerated != null) {
                    onTokenGenerated.accept(nextToken);
                }

                if (stopTokens.contains(nextToken)) {
                    break;
                }
            }

            currentToken = nextToken;
            state.latestToken = currentToken;
            pos++;
        }

        long endNanos = System.nanoTime();
        long decodeStart = inferenceStartNanos > 0 ? inferenceStartNanos : endNanos;
        RunMetrics.setInferenceMetrics(
                promptIndex,
                decodeStart - startNanos,
                generatedTokens.size(),
                endNanos - decodeStart,
                endNanos - startNanos);

        return generatedTokens;
    }

    /**
     * Generates tokens using the Granite model with GPU (TornadoVM) inference. Identical pattern to
     * generateTokensGPULlama.
     */
    public static List<Integer> generateTokensGPUGranite(
            Model model,
            State state,
            int startPosition,
            List<Integer> promptTokens,
            Set<Integer> stopTokens,
            int maxTokens,
            Sampler sampler,
            boolean echo,
            IntConsumer onTokenGenerated,
            TornadoVMMasterPlan tornadoVMMasterPlan) {
        long startNanos = System.nanoTime();
        long inferenceStartNanos = 0;

        Logits logits;
        if (maxTokens < 0 || model.configuration().contextLength() < maxTokens) {
            maxTokens = model.configuration().contextLength();
        }

        List<Integer> generatedTokens = new ArrayList<>();

        PromptIngestion ingestion = PromptIngestion.of(state, promptTokens, startPosition);
        int currentToken = ingestion.firstToken();
        int nextToken;
        int promptIndex = ingestion.firstIndex();
        int pos = startPosition;

        while (pos < maxTokens) {
            // Call TornadoVM forward pass (same as Llama for now)
            logits =
                    org.beehive.gpullama3.backend.tornado.TornadoForwardPass.forward(
                            model, state, currentToken, pos, tornadoVMMasterPlan);

            if (promptIndex < promptTokens.size()) {
                nextToken = promptTokens.get(promptIndex++);
                if (echo) {
                    System.err.print(
                            Tokenizer.replaceControlCharacters(
                                    model.tokenizer().decode(List.of(nextToken))));
                }
            } else {
                if (inferenceStartNanos == 0) {
                    inferenceStartNanos = System.nanoTime();
                }

                nextToken = sampler.sampleToken(logits);

                if (echo) {
                    System.err.print(
                            Tokenizer.replaceControlCharacters(
                                    model.tokenizer().decode(List.of(nextToken))));
                }

                generatedTokens.add(nextToken);

                if (onTokenGenerated != null) {
                    onTokenGenerated.accept(nextToken);
                }

                if (stopTokens.contains(nextToken)) {
                    break;
                }
            }

            currentToken = nextToken;
            state.latestToken = currentToken;
            pos++;
        }

        long endNanos = System.nanoTime();
        long decodeStart = inferenceStartNanos > 0 ? inferenceStartNanos : endNanos;
        RunMetrics.setInferenceMetrics(
                promptIndex,
                decodeStart - startNanos,
                generatedTokens.size(),
                endNanos - decodeStart,
                endNanos - startNanos);

        return generatedTokens;
    }
}
