package org.beehive.gpullama3.golden;

import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.loader.ModelLoader;
import org.beehive.gpullama3.tensor.standard.FloatTensor;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Runs the pinned fixture and captures one logits row per generated token.
 *
 * <p>The hook is the {@link Sampler}: it receives the logits row for every generated position, so
 * capturing needs no production change. Sampling stays greedy (argmax), which makes the seed
 * irrelevant and the token sequence deterministic.
 *
 * <p>Requires {@code -Dllama.deviceSample=false} (the default). With on-device sampling the argmax
 * runs on the GPU and only the token id crosses to the host, so there would be no logits row to
 * capture — {@link #assertHostLogitsAvailable()} makes that explicit rather than silently
 * producing empty goldens.
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

    private GoldenCapture() {
    }

    public static void assertHostLogitsAvailable() {
        if (Boolean.getBoolean("llama.deviceSample")) {
            throw new IllegalStateException(
                    "llama.deviceSample=true keeps the logits row on the device; goldens must run with it false");
        }
    }

    public static Result capture(Path ggufPath, boolean useGpu) throws Exception {
        assertHostLogitsAvailable();

        Model model = ModelLoader.loadModel(ggufPath, CONTEXT_LENGTH, true, useGpu);
        State state = model.createNewState();
        ChatFormat chatFormat = model.chatFormat();

        List<Integer> promptTokens = new ArrayList<>();
        if (model.shouldAddBeginOfText()) {
            promptTokens.add(chatFormat.getBeginOfText());
        }
        promptTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, PROMPT)));
        promptTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

        Result result = new Result();
        Sampler capturing = tensor -> {
            result.rows.add(toFloatArray(tensor));
            int token = Sampler.TENSOR_ARGMAX.sampleToken(tensor);
            result.tokenIds.add(token);
            return token;
        };

        // No stop tokens: a golden must always compare the same number of rows, so generation is
        // bounded only by TOKENS.
        Set<Integer> stopTokens = Set.of();

        TornadoVMMasterPlan plan = null;
        try {
            if (useGpu) {
                plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);
                model.generateTokensGPU(state, 0, promptTokens, stopTokens,
                        promptTokens.size() + TOKENS, capturing, false, null, plan);
            } else {
                model.generateTokens(state, 0, promptTokens, stopTokens,
                        promptTokens.size() + TOKENS, capturing, false, null);
            }
        } finally {
            if (plan != null) {
                plan.freeTornadoExecutionPlan();
            }
        }
        return result;
    }

    private static float[] toFloatArray(Object tensor) {
        if (tensor instanceof FloatArray fa) {
            float[] out = new float[fa.getSize()];
            for (int i = 0; i < out.length; i++) {
                out[i] = fa.get(i);
            }
            return out;
        }
        if (tensor instanceof FloatTensor ft) {
            float[] out = new float[ft.size()];
            for (int i = 0; i < out.length; i++) {
                out[i] = ft.getFloat(i);
            }
            return out;
        }
        throw new IllegalArgumentException("unsupported logits type: "
                + (tensor == null ? "null" : tensor.getClass().getName()));
    }
}
