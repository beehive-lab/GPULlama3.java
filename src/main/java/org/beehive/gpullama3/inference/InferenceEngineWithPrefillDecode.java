package org.beehive.gpullama3.inference;

import org.beehive.gpullama3.auxiliary.RunMetrics;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tokenizer.Tokenizer;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlanPrefillDecode;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

/**
 * Token generation entry point for the sequential prefill/decode inference path (Phase 1/2).
 *
 * <p>Parallel to {@link InferenceEngine} — does NOT modify it.</p>
 *
 * <p>The split loop runs two phases:</p>
 * <ol>
 * <li><b>Prefill</b> (positions 0..N-1): calls
 * {@link InferenceCoreWithPrefillDecode#forwardJavaPrefill} for every
 * prompt token. Vocabulary projection is skipped because these logits
 * are discarded. KV cache is populated identically to the baseline.</li>
 * <li><b>Decode</b> (position N onward): calls
 * {@link InferenceCore#forwardJava} per generated token.
 * Behaviour is identical to the baseline decode path.</li>
 * </ol>
 *
 * <p>Activated by {@code -Dllama.withPrefillDecode=true} with
 * {@code llama.prefillBatchSize == 1} (default). For batch sizes {@code > 1},
 * see {@link InferenceEngineWithBatchPrefillDecode}.</p>
 */
public final class InferenceEngineWithPrefillDecode {

    /** Benchmarking aid: keep decoding past the stop token so every run generates the same token count. */
    private static final boolean IGNORE_EOS = Boolean.getBoolean("llama.bench.ignoreEos");

    private InferenceEngineWithPrefillDecode() {
    }

    /**
     * LLaMA token generation with sequential prefill/decode separation (CPU, Phase 1).
     *
     * <p>Drop-in replacement for {@link InferenceEngine#generateTokensLlama}.</p>
     */
    // @formatter:off
    public static List<Integer> generateTokensLlama(Model model,
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
        int actualMaxTokens = (maxTokens < 0 || config.contextLength() < maxTokens)
                ? config.contextLength() : maxTokens;

        List<Integer> generatedTokens = new ArrayList<>();

        int currentToken = state.latestToken; // BOS
        int pos = startPosition;
        int N = promptTokens.size();

        // ── Prefill ───────────────────────────────────────────────────────────
        if (N > 0 && pos < actualMaxTokens) {
            for (int promptIndex = 0; promptIndex < N && pos < actualMaxTokens; promptIndex++) {
                InferenceCoreWithPrefillDecode.forwardJavaPrefill(model, state, currentToken, pos);
                currentToken = promptTokens.get(promptIndex);
                if (echo) {
                    System.err.print(Tokenizer.replaceControlCharacters(
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
            int nextToken = sampler.sampleToken(logits);

            if (echo) {
                System.err.print(Tokenizer.replaceControlCharacters(
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
        RunMetrics.setInferenceMetrics(promptTokens.size(), decodeStartNanos - startNanos,
                generatedTokens.size(), endNanos - decodeStartNanos, endNanos - startNanos);
        RunMetrics.setHasPrefillPhase(true);

        return generatedTokens;
    }

    /**
     * LLaMA GPU token generation with sequential prefill/decode separation (Phase 2).
     *
     * <p>FP16 only; Q8_0 throws {@link UnsupportedOperationException}.</p>
     *
     * <p>Split loop:</p>
     * <ul>
     * <li><b>Prefill</b> (0..N-1): {@link InferenceCoreWithPrefillDecode#forwardTornadoVMPrefill}
     * — layer graphs execute, logits graph is skipped.</li>
     * <li><b>Decode</b> (N onward): {@link InferenceCore#forwardTornadoVM}
     * — identical to the baseline GPU decode path.</li>
     * </ul>
     */
    public static List<Integer> generateTokensGPULlama(Model model,
                                                       State state,
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
        int actualMaxTokens = (maxTokens < 0 || config.contextLength() < maxTokens)
                ? config.contextLength() : maxTokens;

        TornadoVMMasterPlanPrefillDecode prefillPlan = (TornadoVMMasterPlanPrefillDecode) tornadoVMPlan;

        List<Integer> generatedTokens = new ArrayList<>();

        int currentToken = state.latestToken; // BOS
        int pos = startPosition;

        // ── Prefill (GPU, no logits) ──────────────────────────────────────────
        for (int promptIndex = 0; promptIndex < promptTokens.size() && pos < actualMaxTokens; promptIndex++) {
            InferenceCoreWithPrefillDecode.forwardTornadoVMPrefill(model, state, currentToken, pos, prefillPlan);
            currentToken = promptTokens.get(promptIndex);
            if (echo) {
                System.err.print(Tokenizer.replaceControlCharacters(
                        model.tokenizer().decode(List.of(currentToken))));
            }
            pos++;
        }

        state.latestToken = currentToken;
        long decodeStartNanos = System.nanoTime();

        // ── Decode (GPU, with logits) ─────────────────────────────────────────
        while (pos < actualMaxTokens) {
            var logits = InferenceCore.forwardTornadoVM(model, state, currentToken, pos, tornadoVMPlan);
            int nextToken = sampler.sampleToken(logits);

            if (echo) {
                System.err.print(Tokenizer.replaceControlCharacters(
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
        RunMetrics.setInferenceMetrics(promptTokens.size(), decodeStartNanos - startNanos,
                generatedTokens.size(), endNanos - decodeStartNanos, endNanos - startNanos);
        RunMetrics.setHasPrefillPhase(true);

        return generatedTokens;
    }
    // @formatter:on
}
