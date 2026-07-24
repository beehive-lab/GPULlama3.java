package org.beehive.gpullama3.inference;

import org.beehive.gpullama3.auxiliary.RunMetrics;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tokenizer.Tokenizer;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlanBatchPrefillDecode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

/**
 * Token generation entry point for the batched prefill/decode inference path (Phase 3/4).
 *
 * <p>Parallel to {@link InferenceEngineWithPrefillDecode} — does NOT modify it.</p>
 *
 * <p>The split loop runs two phases:</p>
 * <ol>
 * <li><b>Prefill</b> (positions 0..N-1): processes prompt tokens in chunks of
 * {@link TornadoVMMasterPlan#PREFILL_BATCH_SIZE} using
 * {@link InferenceCoreBatchPrefillDecode#batchForwardJavaPrefill} (CPU) or
 * {@link InferenceCoreBatchPrefillDecode#batchForwardTornadoVMPrefill} (GPU).
 * Logits are discarded; only the KV cache is populated.</li>
 * <li><b>Decode</b> (position N onward): calls {@link InferenceCore#forwardJava} (CPU) or
 * {@link InferenceCoreBatchPrefillDecode#forwardTornadoVMDecode} (GPU) per token.</li>
 * </ol>
 *
 * <p>Activated when both {@code -Dllama.withPrefillDecode=true} and
 * {@code -Dllama.prefillBatchSize > 1} are set.</p>
 */
public final class InferenceEngineWithBatchPrefillDecode {

    /** Benchmarking aid: keep decoding past the stop token so every run generates the same token count. */
    private static final boolean IGNORE_EOS = Boolean.getBoolean("llama.bench.ignoreEos");

    private InferenceEngineWithBatchPrefillDecode() {
    }

    /**
     * LLaMA batched prefill token generation (CPU, Phase 3).
     *
     * <p>Prompt tokens are processed in chunks of {@link TornadoVMMasterPlan#PREFILL_BATCH_SIZE}
     * using batch matmul ({@link InferenceCoreBatchPrefillDecode#batchForwardJavaPrefill}),
     * which traverses each weight matrix once per chunk instead of once per token.
     * A remainder chunk of size 1 falls back to the sequential prefill path.</p>
     *
     * <p>Drop-in replacement for {@link InferenceEngine#generateTokensLlama} when batching
     * is enabled.</p>
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
        final int batchSize = TornadoVMMasterPlan.PREFILL_BATCH_SIZE;

        List<Integer> generatedTokens = new ArrayList<>();

        int currentToken = state.latestToken; // BOS
        int pos = startPosition;
        int N = promptTokens.size();

        // ── Prefill ───────────────────────────────────────────────────────────
        if (N > 0 && pos < actualMaxTokens) {
            // Build the token sequence at positions [startPosition .. startPosition+N-1]:
            //   position startPosition+0 : currentToken (BOS)
            //   position startPosition+k : promptTokens[k-1]
            int[] prefillSeq = new int[N];
            prefillSeq[0] = currentToken;
            for (int i = 1; i < N; i++) {
                prefillSeq[i] = promptTokens.get(i - 1);
            }

            for (int chunkStart = 0; chunkStart < N && pos + chunkStart < actualMaxTokens; chunkStart += batchSize) {
                int chunkEnd = Math.min(Math.min(chunkStart + batchSize, N), actualMaxTokens - pos);
                int chunkSize = chunkEnd - chunkStart;
                int[] chunk = Arrays.copyOfRange(prefillSeq, chunkStart, chunkEnd);

                if (chunkSize == 1) {
                    InferenceCoreWithPrefillDecode.forwardJavaPrefill(model, state, chunk[0], pos + chunkStart);
                } else {
                    InferenceCoreBatchPrefillDecode.batchForwardJavaPrefill(model, state, chunk, pos + chunkStart, chunkSize);
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
    // @formatter:on

    /**
     * LLaMA batched GPU prefill token generation (GPU, Phase 4).
     *
     * <p>FP16 only; Q8_0 throws {@link UnsupportedOperationException}.</p>
     *
     * <p>Split loop:</p>
     * <ul>
     * <li><b>Prefill</b>: {@link InferenceCoreBatchPrefillDecode#batchForwardTornadoVMPrefill}
     * processes each chunk (including size-1 remainder) via the batch GPU kernels.</li>
     * <li><b>Decode</b>: {@link InferenceCoreBatchPrefillDecode#forwardTornadoVMDecode}
     * per generated token.</li>
     * </ul>
     */
    // @formatter:off
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
        final int batchSize = TornadoVMMasterPlan.PREFILL_BATCH_SIZE;

        TornadoVMMasterPlanBatchPrefillDecode plan = (TornadoVMMasterPlanBatchPrefillDecode) tornadoVMPlan;

        List<Integer> generatedTokens = new ArrayList<>();

        int currentToken = state.latestToken; // BOS
        int pos = startPosition;
        int N = promptTokens.size();

        // ── Prefill ───────────────────────────────────────────────────────────
        // Build the token sequence at positions [startPosition .. startPosition+N-1]:
        //   position startPosition+0 : currentToken (BOS/previous token)
        //   position startPosition+k : promptTokens[k-1]
        int[] prefillSeq = new int[N];
        prefillSeq[0] = currentToken;
        for (int i = 1; i < N; i++) {
            prefillSeq[i] = promptTokens.get(i - 1);
        }

        for (int chunkStart = 0; chunkStart < N && pos + chunkStart < actualMaxTokens; chunkStart += batchSize) {
            int chunkEnd = Math.min(Math.min(chunkStart + batchSize, N), actualMaxTokens - pos);
            int chunkSize = chunkEnd - chunkStart;
            int[] chunk = Arrays.copyOfRange(prefillSeq, chunkStart, chunkEnd);

            InferenceCoreBatchPrefillDecode.batchForwardTornadoVMPrefill(model, state, chunk, pos + chunkStart, chunkSize, plan);

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
        long decodeStartNanos = System.nanoTime();

        // ── Decode ────────────────────────────────────────────────────────────
        while (pos < actualMaxTokens) {
            var logits = InferenceCoreBatchPrefillDecode.forwardTornadoVMDecode(model, state, currentToken, pos, plan);
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
