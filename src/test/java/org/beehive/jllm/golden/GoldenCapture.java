package org.beehive.gpullama3.golden;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.beehive.gpullama3.backend.tornado.TornadoVMMasterPlan;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.loader.ModelLoader;

/**
 * Runs the pinned fixture and captures one logits row per generated token.
 *
 * <p>The hook is the {@link Sampler}: it receives the logits row for every generated position, so
 * capturing needs no production change. Sampling stays greedy (argmax), which makes the seed
 * irrelevant and the token sequence deterministic.
 *
 * <p>Requires {@code -Dllama.deviceSample=false} (the default). With on-device sampling the argmax
 * runs on the GPU and only the token id crosses to the host, so there would be no logits row to
 * capture — {@link #assertHostLogitsAvailable()} makes that explicit rather than silently producing
 * empty goldens.
 */
public final class GoldenCapture {

    /** Compared rows: one per generated token. Stated verbatim in the golden metadata. */
    public static final int TOKENS = 64;

    /** The fixed prompt. Any change to this invalidates every committed golden. */
    public static final String PROMPT = "Explain what a matrix multiplication is in one paragraph.";

    public static final int CONTEXT_LENGTH = 512;

    public static final class Result {
        public final List<float[]> rows = new ArrayList<>();
        public final List<Integer> tokenIds = new ArrayList<>();
    }

    private GoldenCapture() {}

    public static void assertHostLogitsAvailable() {
        if (Boolean.getBoolean("llama.deviceSample")) {
            throw new IllegalStateException(
                    "llama.deviceSample=true keeps the logits row on the device; goldens must run with it false");
        }
    }

    public static Result capture(Path ggufPath, boolean useGpu) throws Exception {
        return capture(ggufPath, useGpu, null);
    }

    /**
     * @param forcedTokens when non-null, the sampler returns these tokens instead of its own argmax
     *     ("teacher forcing").
     *     <p>This is what makes a cross-path comparison meaningful. Greedy decoding is
     *     autoregressive, so the first near-tie that tips differently sends the two paths into
     *     different contexts, and every row after that compares unrelated states. Forcing both
     *     paths along the same token sequence keeps the context identical at every position, so a
     *     difference in logits is a difference in arithmetic rather than a difference in history.
     */
    public static Result capture(Path ggufPath, boolean useGpu, List<Integer> forcedTokens)
            throws Exception {
        return capture(ggufPath, useGpu, forcedTokens, 1);
    }

    /**
     * The same capture, driven through batched prefill when {@code prefillBatchSize} is above one.
     *
     * <p>The execution policy is read from system properties when the {@link State} is built, and
     * the batch width has to be known then too because the prefill workspace is sized from it. So
     * the properties are set around construction and restored after: this harness is the only place
     * that knows the mode is being forced, and a leaked property would silently change every later
     * test in the same JVM.
     *
     * <p>Nothing else changes. The same prompt, the same forced token sequence and the same
     * capturing sampler, so a difference in the rows is a difference in the arithmetic of the
     * batched kernels and not a difference in what was asked of them.
     */
    public static Result capture(
            Path ggufPath, boolean useGpu, List<Integer> forcedTokens, int prefillBatchSize)
            throws Exception {
        assertHostLogitsAvailable();

        String previousPrefill = System.getProperty("llama.withPrefillDecode");
        String previousBatch = System.getProperty("llama.prefillBatchSize");
        if (prefillBatchSize > 1) {
            System.setProperty("llama.withPrefillDecode", "true");
            System.setProperty("llama.prefillBatchSize", String.valueOf(prefillBatchSize));
        }
        try {
            return captureWithCurrentPolicy(ggufPath, useGpu, forcedTokens, prefillBatchSize);
        } finally {
            restore("llama.withPrefillDecode", previousPrefill);
            restore("llama.prefillBatchSize", previousBatch);
        }
    }

    private static void restore(String key, String previous) {
        if (previous == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, previous);
        }
    }

    private static Result captureWithCurrentPolicy(
            Path ggufPath, boolean useGpu, List<Integer> forcedTokens, int prefillBatchSize)
            throws Exception {
        Model model = ModelLoader.loadModel(ggufPath, CONTEXT_LENGTH, true, useGpu);
        State state =
                prefillBatchSize > 1
                        ? State.withPrefillBatchSize(prefillBatchSize, () -> model.createNewState())
                        : model.createNewState();
        ChatFormat chatFormat = model.chatFormat();

        List<Integer> promptTokens = new ArrayList<>();
        if (model.shouldAddBeginOfText()) {
            promptTokens.add(chatFormat.getBeginOfText());
        }
        promptTokens.addAll(
                chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, PROMPT)));
        promptTokens.addAll(
                chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

        Result result = new Result();
        Sampler capturing =
                tensor -> {
                    result.rows.add(toFloatArray(tensor));
                    int token = Sampler.TENSOR_ARGMAX.sampleToken(tensor);
                    result.tokenIds.add(token);
                    int step = result.rows.size() - 1;
                    if (forcedTokens != null && step < forcedTokens.size()) {
                        return forcedTokens.get(step);
                    }
                    return token;
                };

        // No stop tokens: a golden must always compare the same number of rows, so generation is
        // bounded only by TOKENS.
        Set<Integer> stopTokens = Set.of();

        // The budget counts every forward, ingestion included, and ingestion is one shorter for a
        // family whose seed the prompt already carries — see PromptIngestion. Deriving the
        // adjustment from the same source keeps the row count at TOKENS for every family, instead
        // of encoding one family's arithmetic as a constant that quietly rots.
        int skippedSeed =
                org.beehive.gpullama3.inference.PromptIngestion.of(state, promptTokens, 0)
                        .firstIndex();
        int budget = promptTokens.size() + TOKENS - skippedSeed;

        TornadoVMMasterPlan plan = null;
        try {
            if (useGpu) {
                // The factory consults the lowering's opt-in itself, so this harness needs no
                // branch of its own — it needed one while the branch lived at each construction
                // site, and that duplication is what let the CLI and the benchmark script miss it.
                // Callers still assert on LoweredPlanSelection.loweredPlanCount(), never on the
                // property: the question is whether the lowering ran, not whether it was asked for.
                plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);
                model.generateTokensGPU(
                        state, 0, promptTokens, stopTokens, budget, capturing, false, null, plan);
            } else {
                model.generateTokens(
                        state, 0, promptTokens, stopTokens, budget, capturing, false, null);
            }
        } finally {
            if (plan != null) {
                plan.freeTornadoExecutionPlan();
            }
        }
        return result;
    }

    private static float[] toFloatArray(org.beehive.gpullama3.inference.Logits logits) {
        float[] out = new float[logits.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = logits.get(i);
        }
        return out;
    }
}
