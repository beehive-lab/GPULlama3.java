package org.beehive.gpullama3.golden;

import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.loader.ModelLoader;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Separates runtime non-determinism from compilation non-determinism.
 *
 * <p>Every earlier probe rebuilt the {@link TornadoVMMasterPlan} per run, which recompiles the task
 * graphs — so "run-to-run" drift could have been two different kernels rather than one kernel
 * behaving differently. This runs the same input twice through <b>one</b> plan (A vs B), then once
 * more through a <b>freshly built</b> plan (C).
 *
 * <p>A≠B ⇒ runtime. A=B and A≠C ⇒ codegen differs per compilation.
 */
public final class OnePlanTest {

    public static void main(String[] args) throws Exception {
        Path modelPath = Paths.get(System.getProperty("plan.model"));
        int tokens = Integer.getInteger("plan.tokens", 8);
        List<Integer> forced = new ArrayList<>();
        for (String t : System.getProperty("plan.forced", "6828,47544,374,264").split(",")) {
            forced.add(Integer.parseInt(t.trim()));
        }

        Model model = ModelLoader.loadModel(modelPath, 512, true, true);
        State state = model.createNewState();
        List<Integer> prompt = buildPrompt(model);

        TornadoVMMasterPlan plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);
        List<float[]> a;
        List<float[]> b;
        try {
            a = run(model, state, prompt, plan, tokens, forced);
            // Re-running the whole sequence is NOT a clean repeat: State carries over (proved by
            // Q8_0 producing a bit-identical 19.37 delta twice, which a race cannot do). Instead
            // re-execute the SAME token at the SAME position, which rewrites that position's KV
            // with the same values and advances nothing.
            int pos = prompt.size();
            int tok = forced.get(forced.size() - 1);
            float[] first = snap(org.beehive.gpullama3.inference.InferenceCore
                    .forwardTornadoVM(model, state, tok, pos, plan));
            for (int k = 0; k < 5; k++) {
                float[] again = snap(org.beehive.gpullama3.inference.InferenceCore
                        .forwardTornadoVM(model, state, tok, pos, plan));
                int diff = 0;
                double worst = 0;
                for (int i = 0; i < first.length; i++) {
                    if (Float.floatToRawIntBits(first[i]) != Float.floatToRawIntBits(again[i])) {
                        diff++;
                        worst = Math.max(worst, Math.abs(first[i] - again[i]));
                    }
                }
                System.out.printf("same plan, same token+position, repeat %d : changed=%d/%d worstAbs=%.7f  %s%n",
                        k + 1, diff, first.length, worst, diff == 0 ? "IDENTICAL" : "*** DIVERGES ***");
            }
            b = a;
        } finally {
            plan.freeTornadoExecutionPlan();
        }

        // Fresh state and a fresh plan: this recompiles the task graphs.
        State state2 = model.createNewState();
        TornadoVMMasterPlan plan2 = TornadoVMMasterPlan.initializeTornadoVMPlan(state2, model);
        List<float[]> c;
        try {
            c = run(model, state2, prompt, plan2, tokens, forced);
        } finally {
            plan2.freeTornadoExecutionPlan();
        }

        report("A vs C  (rebuilt plan, recompiled)   ", a, c);
    }

    private static float[] snap(FloatArray fa) {
        float[] out = new float[fa.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = fa.get(i);
        }
        return out;
    }

    private static List<Integer> buildPrompt(Model model) {
        ChatFormat cf = model.chatFormat();
        List<Integer> prompt = new ArrayList<>();
        if (model.shouldAddBeginOfText()) {
            prompt.add(cf.getBeginOfText());
        }
        prompt.addAll(cf.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, GoldenCapture.PROMPT)));
        prompt.addAll(cf.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
        return prompt;
    }

    private static List<float[]> run(Model model, State state, List<Integer> prompt,
            TornadoVMMasterPlan plan, int tokens, List<Integer> forced) {
        List<float[]> rows = new ArrayList<>();
        Sampler capture = t -> {
            rows.add(toArray(t));
            int tok = Sampler.TENSOR_ARGMAX.sampleToken(t);
            int step = rows.size() - 1;
            return step < forced.size() ? forced.get(step) : tok;
        };
        model.generateTokensGPU(state, 0, prompt, Set.of(), prompt.size() + tokens,
                capture, false, null, plan);
        return rows;
    }

    private static void report(String label, List<float[]> x, List<float[]> y) {
        int rows = Math.min(x.size(), y.size());
        int divergent = 0;
        int firstBad = -1;
        double worst = 0;
        for (int r = 0; r < rows; r++) {
            boolean differs = false;
            for (int i = 0; i < x.get(r).length; i++) {
                if (Float.floatToRawIntBits(x.get(r)[i]) != Float.floatToRawIntBits(y.get(r)[i])) {
                    differs = true;
                    worst = Math.max(worst, Math.abs(x.get(r)[i] - y.get(r)[i]));
                }
            }
            if (differs) {
                divergent++;
                if (firstBad < 0) {
                    firstBad = r;
                }
            }
        }
        System.out.printf("%s : divergentRows=%d/%d firstDivergentRow=%d worstAbs=%.7f  %s%n",
                label, divergent, rows, firstBad, worst,
                divergent == 0 ? "IDENTICAL" : "*** DIVERGES ***");
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

    private OnePlanTest() {
    }
}
