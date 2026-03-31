package org.beehive.gpullama3.inference;

import org.beehive.gpullama3.auxiliary.LastRunMetrics;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tokenizer.Tokenizer;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlanWithPrefillDecode;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

/**
 * Token generation entry point for the prefill/decode separated inference path.
 *
 * <p>Parallel to {@link InferenceEngine} — does NOT modify it.</p>
 *
 * <p>The split loop runs two phases:</p>
 * <ol>
 *   <li><b>Prefill</b> (positions 0..N-1): calls
 *       {@link InferenceCoreWithPrefillDecode#forwardJavaPrefill} for every
 *       prompt token. Vocabulary projection is skipped because these logits
 *       are discarded. KV cache is populated identically to the baseline.</li>
 *   <li><b>Decode</b> (position N onward): calls
 *       {@link InferenceCore#forwardJava} per generated token.
 *       Behaviour is identical to the baseline decode path.</li>
 * </ol>
 *
 * <p>Activated by {@code -Dllama.batchedPrefill=true} (set via
 * {@code --batched-prefill} in the Python launcher).</p>
 */
public final class InferenceEngineWithPrefillDecode {

    private InferenceEngineWithPrefillDecode() {}

    /**
     * LLaMA token generation with prefill/decode separation (CPU, Phase 1).
     *
     * <p>Drop-in replacement for
     * {@link InferenceEngine#generateTokensLlama} when the batched-prefill
     * flag is enabled. Only the CPU path is implemented here; GPU support
     * is added in a later phase.</p>
     */
    public static List<Integer> generateTokensLlama(
            Model model, State state, int startPosition,
            List<Integer> promptTokens, Set<Integer> stopTokens,
            int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated) {

        long startNanos = System.nanoTime();

        final Configuration config = model.configuration();
        if (maxTokens < 0 || config.contextLength() < maxTokens) {
            maxTokens = config.contextLength();
        }

        List<Integer> generatedTokens = new ArrayList<>();

        int currentToken = state.latestToken; // BOS
        int pos = startPosition;

        // ── Phase 1: Prefill ──────────────────────────────────────────────────
        // Run all prompt tokens through the forward pass without computing
        // logits. The KV cache is populated at each position, which is all
        // that matters. After this loop:
        //   currentToken == promptTokens.getLast()
        //   pos          == startPosition + promptTokens.size()
        for (int promptIndex = 0; promptIndex < promptTokens.size(); promptIndex++) {
            InferenceCoreWithPrefillDecode.forwardJavaPrefill(model, state, currentToken, pos);
            currentToken = promptTokens.get(promptIndex);
            if (echo) {
                System.err.print(Tokenizer.replaceControlCharacters(
                        model.tokenizer().decode(List.of(currentToken))));
            }
            pos++;
        }

        state.latestToken = currentToken;

        // ── Phase 2: Decode ───────────────────────────────────────────────────
        // Standard single-token forward with logits.  Behaviour is identical
        // to the baseline InferenceEngine decode path.
        long inferenceStartNanos = 0;
        while (pos < maxTokens) {
            if (inferenceStartNanos == 0) {
                inferenceStartNanos = System.nanoTime();
            }

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

            if (stopTokens.contains(nextToken)) {
                break;
            }

            currentToken = nextToken;
            state.latestToken = currentToken;
            pos++;
        }

        long endNanos = System.nanoTime();
        int totalTokens = promptTokens.size() + generatedTokens.size();
        LastRunMetrics.setMetrics(totalTokens, (endNanos - startNanos) / 1_000_000_000.0);

        return generatedTokens;
    }

    /**
     * LLaMA GPU token generation with prefill/decode separation (Phase 2).
     *
     * <p>Drop-in replacement for
     * {@link InferenceEngine#generateTokensGPULlama} when the batched-prefill
     * flag is enabled. FP16 only; Q8_0 throws {@link UnsupportedOperationException}.</p>
     *
     * <p>Split loop:</p>
     * <ul>
     *   <li><b>Prefill</b> (0..N-1): {@link InferenceCoreWithPrefillDecode#forwardTornadoVMPrefill}
     *       — layer graphs execute, logits graph is skipped.</li>
     *   <li><b>Decode</b> (N onward): {@link InferenceCore#forwardTornadoVM}
     *       — identical to the baseline GPU decode path.</li>
     * </ul>
     */
    public static List<Integer> generateTokensGPULlama(
            Model model, State state, int startPosition,
            List<Integer> promptTokens, Set<Integer> stopTokens,
            int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated, TornadoVMMasterPlan tornadoVMPlan) {

        // Q8_0 GPU prefill not implemented yet
        if (((TornadoWeights) model.weights()).getWeightType() == GGMLType.Q8_0) {
            // TODO Phase 4: implement Q8_0 GPU batched prefill kernels
            throw new UnsupportedOperationException(
                    "GPU prefill/decode path not yet implemented for Q8_0 weights");
        }

        long startNanos = System.nanoTime();

        final Configuration config = model.configuration();
        int actualMaxTokens = (maxTokens < 0 || config.contextLength() < maxTokens)
                ? config.contextLength() : maxTokens;

        List<Integer> generatedTokens = new ArrayList<>();

        int currentToken = state.latestToken; // BOS
        int pos = startPosition;

        // Thin wrapper: no new TornadoVM plan created, just holds the reference
        TornadoVMMasterPlanWithPrefillDecode prefillPlan =
                new TornadoVMMasterPlanWithPrefillDecode(tornadoVMPlan, state, model);

        // ── Phase 1: Prefill (GPU, no logits) ────────────────────────────────
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

        // ── Phase 2: Decode (GPU, with logits) ───────────────────────────────
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

            if (stopTokens.contains(nextToken)) {
                break;
            }

            currentToken = nextToken;
            state.latestToken = currentToken;
            pos++;
        }

        long endNanos = System.nanoTime();
        int totalTokens = promptTokens.size() + generatedTokens.size();
        LastRunMetrics.setMetrics(totalTokens, (endNanos - startNanos) / 1_000_000_000.0);

        return generatedTokens;
    }


}
