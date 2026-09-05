package org.beehive.gpullama3.golden;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.beehive.gpullama3.backend.tornado.TornadoVMMasterPlan;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.loader.ModelLoader;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Prints raw logit values for the first few generated positions, so CPU and GPU output can be
 * compared element by element. Diagnostic only.
 *
 * <p>Sampling is greedy argmax, so temperature, top-p and seed are fixed and have no effect — the
 * only varying inputs are the prompt and the context length, held identical across configs.
 */
public final class LogitDump {

    public static void main(String[] args) throws Exception {
        Path model = Paths.get(System.getProperty("dump.model"));
        boolean gpu = Boolean.parseBoolean(System.getProperty("dump.gpu", "true"));
        int positions = Integer.getInteger("dump.positions", 4);
        int cols = Integer.getInteger("dump.cols", 6);
        String label = System.getProperty("dump.label", "");
        // Teacher forcing: every configuration is fed exactly these token ids, so the state
        // progression is identical everywhere and any logit difference is arithmetic, not history.
        String forcedProp = System.getProperty("dump.forced", "");
        List<Integer> forced = new ArrayList<>();
        if (!forcedProp.isBlank()) {
            for (String t : forcedProp.split(",")) {
                forced.add(Integer.parseInt(t.trim()));
            }
        }

        Model m = ModelLoader.loadModel(model, 512, true, gpu);
        State state = m.createNewState();
        ChatFormat cf = m.chatFormat();

        List<Integer> prompt = new ArrayList<>();
        if (m.shouldAddBeginOfText()) {
            prompt.add(cf.getBeginOfText());
        }
        prompt.addAll(
                cf.encodeMessage(
                        new ChatFormat.Message(ChatFormat.Role.USER, GoldenCapture.PROMPT)));
        prompt.addAll(cf.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

        List<float[]> rows = new ArrayList<>();
        List<Integer> toks = new ArrayList<>();
        Sampler capture =
                t -> {
                    rows.add(toArray(t));
                    int tok = Sampler.TENSOR_ARGMAX.sampleToken(t);
                    toks.add(tok);
                    int step = rows.size() - 1;
                    if (step < forced.size()) {
                        return forced.get(step);
                    }
                    return tok;
                };

        TornadoVMMasterPlan plan = null;
        try {
            if (gpu) {
                plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, m);
                m.generateTokensGPU(
                        state,
                        0,
                        prompt,
                        Set.of(),
                        prompt.size() + positions,
                        capture,
                        false,
                        null,
                        plan);
            } else {
                m.generateTokens(
                        state,
                        0,
                        prompt,
                        Set.of(),
                        prompt.size() + positions,
                        capture,
                        false,
                        null);
            }
        } finally {
            if (plan != null) {
                plan.freeTornadoExecutionPlan();
            }
        }

        System.out.println("### " + label);
        if (Boolean.getBoolean("dump.header")) {
            System.out.println("prompt text : " + GoldenCapture.PROMPT);
            System.out.println("prompt ids  : " + prompt);
        }
        System.out.println("forced ids  : " + (forced.isEmpty() ? "(none - greedy)" : forced));
        System.out.println("argmax ids  : " + toks.subList(0, Math.min(positions, toks.size())));
        for (int p = 0; p < Math.min(positions, rows.size()); p++) {
            StringBuilder sb = new StringBuilder();
            sb.append("pos ").append(p).append("  ");
            for (int i = 0; i < cols; i++) {
                sb.append(String.format("%14.7f", rows.get(p)[i]));
            }
            System.out.println(sb);
        }
    }

    private static float[] toArray(Object t) {
        if (t instanceof FloatArray fa) {
            float[] out = new float[fa.getSize()];
            for (int i = 0; i < out.length; i++) {
                out[i] = fa.get(i);
            }
            return out;
        }
        org.beehive.gpullama3.tensor.standard.FloatTensor ft =
                (org.beehive.gpullama3.tensor.standard.FloatTensor) t;
        float[] out = new float[ft.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ft.getFloat(i);
        }
        return out;
    }

    private LogitDump() {}
}
