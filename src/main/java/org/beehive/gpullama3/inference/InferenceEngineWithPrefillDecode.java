package org.beehive.gpullama3.inference;

import org.beehive.gpullama3.auxiliary.LastRunMetrics;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tokenizer.Tokenizer;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlanBatchPrefill;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlanStandard;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlanWithPrefillDecode;

import java.util.ArrayList;
import java.util.Arrays;
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

    /** Prefill chunk size. 1 = sequential (Phase 1 behaviour), >1 = batched (Phase 3/4). */
    static final int PREFILL_BATCH_SIZE = Integer.getInteger("llama.prefillBatchSize", 1);

    /**
     * LLaMA token generation with prefill/decode separation (CPU).
     *
     * <p>When {@code llama.prefillBatchSize > 1} (Phase 3), prompt tokens are
     * processed in chunks of that size using batch matmul, which traverses each
     * weight matrix once per chunk instead of once per token.</p>
     *
     * <p>When {@code llama.prefillBatchSize == 1} (Phase 1), falls back to
     * sequential single-token prefill (skip logits only).</p>
     *
     * <p>Drop-in replacement for {@link InferenceEngine#generateTokensLlama}.</p>
     */
    public static List<Integer> generateTokensLlama(
            Model model, State state, int startPosition,
            List<Integer> promptTokens, Set<Integer> stopTokens,
            int maxTokens, Sampler sampler, boolean echo,
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
            if (PREFILL_BATCH_SIZE > 1) {
                // Phase 3: batch prefill — process tokens in chunks of PREFILL_BATCH_SIZE.
                // Build the token sequence at positions [startPosition .. startPosition+N-1]:
                //   position startPosition+0 : currentToken (BOS)
                //   position startPosition+1 : promptTokens[0]
                //   ...
                //   position startPosition+N-1: promptTokens[N-2]
                int[] prefillSeq = new int[N];
                prefillSeq[0] = currentToken;
                for (int i = 1; i < N; i++) prefillSeq[i] = promptTokens.get(i - 1);

                for (int chunkStart = 0; chunkStart < N && pos + chunkStart < actualMaxTokens; chunkStart += PREFILL_BATCH_SIZE) {
                    int chunkEnd  = Math.min(Math.min(chunkStart + PREFILL_BATCH_SIZE, N), actualMaxTokens - pos);
                    int chunkSize = chunkEnd - chunkStart;
                    int[] chunk   = Arrays.copyOfRange(prefillSeq, chunkStart, chunkEnd);

                    if (chunkSize == 1) {
                        InferenceCoreWithPrefillDecode.forwardJavaPrefill(model, state, chunk[0], pos + chunkStart);
                    } else {
                        InferenceCoreWithPrefillDecode.batchForwardJavaPrefill(model, state, chunk, pos + chunkStart, chunkSize);
                    }

                    if (echo) {
                        for (int b = 0; b < chunkSize; b++) {
                            int echoed = promptTokens.get(Math.min(chunkStart + b, N - 1));
                            System.err.print(Tokenizer.replaceControlCharacters(
                                    model.tokenizer().decode(List.of(echoed))));
                        }
                    }
                }

                currentToken = promptTokens.get(N - 1);
                pos = startPosition + N;
            } else {
                // Phase 1: sequential prefill — single token, no logits
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
        }

        state.latestToken = currentToken;

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

        if (PREFILL_BATCH_SIZE > 1) {
            // ── Phase 4: Batch GPU Prefill ────────────────────────────────────
            // Plan was pre-initialized in Model.runInstructOnce/runInteractive
            // as a TornadoVMMasterPlanBatchPrefill by TornadoVMMasterPlan.initializeTornadoVMPlan.
            TornadoVMMasterPlanBatchPrefill plan = (TornadoVMMasterPlanBatchPrefill) tornadoVMPlan;

            int N = promptTokens.size();

            // Build the token sequence at positions [startPosition .. startPosition+N-1]:
            //   position startPosition+0 : currentToken (BOS/previous token)
            //   position startPosition+1 : promptTokens[0]
            //   ...
            //   position startPosition+N-1: promptTokens[N-2]
            int[] prefillSeq = new int[N];
            prefillSeq[0] = currentToken;
            for (int i = 1; i < N; i++) prefillSeq[i] = promptTokens.get(i - 1);

            for (int chunkStart = 0; chunkStart < N && pos + chunkStart < actualMaxTokens; chunkStart += PREFILL_BATCH_SIZE) {
                int chunkEnd  = Math.min(Math.min(chunkStart + PREFILL_BATCH_SIZE, N), actualMaxTokens - pos);
                int chunkSize = chunkEnd - chunkStart;
                int[] chunk   = Arrays.copyOfRange(prefillSeq, chunkStart, chunkEnd);

                if (chunkSize == 1) {
                    // Single-token chunk: use decode path (includes logits skip is not needed
                    // here, but we need the KV cache populated — use batch prefill with size 1)
                    plan.tornadoVMForwardBatchPrefill(chunk, pos + chunkStart, model, 1);
                } else {
                    plan.tornadoVMForwardBatchPrefill(chunk, pos + chunkStart, model, chunkSize);
                }

                if (echo) {
                    for (int b = 0; b < chunkSize; b++) {
                        int echoed = promptTokens.get(Math.min(chunkStart + b, N - 1));
                        System.err.print(Tokenizer.replaceControlCharacters(
                                model.tokenizer().decode(List.of(echoed))));
                    }
                }
            }

            currentToken = promptTokens.get(N - 1);
            pos = startPosition + N;
            state.latestToken = currentToken;

            // ── Phase 4: Decode (GPU, with logits, via unified plan) ──────────
            while (pos < actualMaxTokens) {
                var logits = plan.tornadoVMForwardDecode(currentToken, pos, model);
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

        } else {
        // ── Phase 2: Sequential GPU Prefill + Decode ─────────────────────────

        // Thin wrapper: no new TornadoVM plan created, just holds the reference
        // Plan is a TornadoVMMasterPlanStandard when PREFILL_BATCH_SIZE == 1.
        TornadoVMMasterPlanWithPrefillDecode prefillPlan =
                new TornadoVMMasterPlanWithPrefillDecode(
                        (TornadoVMMasterPlanStandard) tornadoVMPlan, state, model);

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

        } // end else (Phase 2)

        long endNanos = System.nanoTime();
        int totalTokens = promptTokens.size() + generatedTokens.size();
        LastRunMetrics.setMetrics(totalTokens, (endNanos - startNanos) / 1_000_000_000.0);

        return generatedTokens;
    }


}
