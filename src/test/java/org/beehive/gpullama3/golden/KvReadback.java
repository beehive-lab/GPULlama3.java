package org.beehive.gpullama3.golden;

import org.beehive.gpullama3.inference.InferenceCore;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.loader.ModelLoader;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Diagnostic: characterises the wrapKeyCache host readback, which differs on every iteration under
 * -Dgpullama3.diag.transfers even when the logits are bit-identical. Reports where in the buffer
 * the differences sit and whether the changing region is ever non-zero.
 */
public final class KvReadback {

    public static void main(String[] args) throws Exception {
        Path modelPath = Path.of(System.getProperty("rate.model"));
        int iterations = Integer.getInteger("rate.iterations", 10);

        Model model = ModelLoader.loadModel(modelPath, 512, true, true);
        State state = model.createNewState();

        ChatFormat cf = model.chatFormat();
        List<Integer> prompt = new ArrayList<>();
        if (model.shouldAddBeginOfText()) {
            prompt.add(cf.getBeginOfText());
        }
        prompt.addAll(cf.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, GoldenCapture.PROMPT)));
        prompt.addAll(cf.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

        TornadoVMMasterPlan plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);
        try {
            List<Integer> sink = new ArrayList<>();
            Sampler greedy = t -> {
                int tok = Sampler.TENSOR_ARGMAX.sampleToken(t);
                sink.add(tok);
                return tok;
            };
            model.generateTokensGPU(state, 0, prompt, Set.of(), prompt.size() + 1, greedy, false, null, plan);

            int pos = prompt.size();
            int tok = sink.get(0);
            int kvDim = model.configuration().numberOfKeyValueHeads() * model.configuration().headSize();
            System.out.printf("kvCache size=%d kvDim=%d layers=%d promptLen=%d pos=%d%n",
                    state.wrapKeyCache.getSize(), kvDim, model.configuration().numberOfLayers(),
                    prompt.size(), pos);

            float[] prev = snap(InferenceCore.forwardTornadoVM(model, state, tok, pos, plan));
            float[] prevKv = snap(state.wrapKeyCache);
            for (int i = 1; i <= iterations; i++) {
                float[] logits = snap(InferenceCore.forwardTornadoVM(model, state, tok, pos, plan));
                float[] kv = snap(state.wrapKeyCache);

                boolean logitsSame = same(prev, logits);
                int diff = 0, first = -1, last = -1, nonZeroBoth = 0, zeroToValue = 0, valueToZero = 0;
                double worst = 0;
                for (int j = 0; j < kv.length; j++) {
                    if (Float.floatToRawIntBits(prevKv[j]) != Float.floatToRawIntBits(kv[j])) {
                        diff++;
                        if (first < 0) {
                            first = j;
                        }
                        last = j;
                        worst = Math.max(worst, Math.abs(prevKv[j] - kv[j]));
                        if (prevKv[j] == 0f && kv[j] != 0f) {
                            zeroToValue++;
                        } else if (prevKv[j] != 0f && kv[j] == 0f) {
                            valueToZero++;
                        } else {
                            nonZeroBoth++;
                        }
                    }
                }
                System.out.printf("iter=%2d logitsIdentical=%-5s kvDiffering=%d/%d first=%d(layer %d, pos %d) last=%d worst=%.6g "
                                + "[0->v=%d v->0=%d v->v=%d]%n",
                        i, logitsSame, diff, kv.length, first,
                        first < 0 ? -1 : first / (kvDim * 512), first < 0 ? -1 : (first % (kvDim * 512)) / kvDim,
                        last, worst, zeroToValue, valueToZero, nonZeroBoth);
                prevKv = kv;
            }
        } finally {
            plan.freeTornadoExecutionPlan();
        }
    }

    private static boolean same(float[] a, float[] b) {
        for (int i = 0; i < a.length; i++) {
            if (Float.floatToRawIntBits(a[i]) != Float.floatToRawIntBits(b[i])) {
                return false;
            }
        }
        return true;
    }

    private static float[] snap(FloatArray fa) {
        float[] out = new float[fa.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = fa.get(i);
        }
        return out;
    }

    private KvReadback() {
    }
}
