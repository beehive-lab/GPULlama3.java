package org.beehive.gpullama3;

import static org.beehive.gpullama3.inference.sampler.Sampler.createSampler;
import static org.beehive.gpullama3.model.loader.ModelLoader.loadModel;

import java.io.IOException;
import org.beehive.gpullama3.auxiliary.RunMetrics;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.model.Model;

public class LlamaApp {
    // Configuration flags for hardware acceleration and optimizations
    public static final boolean USE_VECTOR_API =
            Boolean.parseBoolean(
                    System.getProperty(
                            "llama.VectorAPI",
                            "true")); // Enable Java Vector API for CPU acceleration
    public static final boolean SHOW_PERF_INTERACTIVE =
            Boolean.parseBoolean(
                    System.getProperty(
                            "llama.ShowPerfInteractive",
                            "true")); // Show performance metrics in interactive mode

    /**
     * On-device greedy sampling ({@code -Dllama.deviceSample=true}) keeps the logits on the GPU and
     * returns only the argmax token id. It is only valid on the GPU FP16 greedy path for the models
     * whose decode loop reads {@code state.workspace.sampledToken} (Llama / Mistral / Qwen3). For
     * any other configuration the host still needs the full logits row, so the flag is cleared
     * here.
     */
    private static void guardDeviceSample(Model model, Options options) {
        if (!Boolean.getBoolean("llama.deviceSample")) {
            return;
        }
        boolean greedy = options.temperature() == 0.0f;
        boolean fp16 = "FP16".equals(model.configuration().quantization());
        var mt = model.getModelType();
        boolean wiredLoop =
                mt == org.beehive.gpullama3.model.ModelType.LLAMA_3
                        || mt == org.beehive.gpullama3.model.ModelType.MISTRAL
                        || mt == org.beehive.gpullama3.model.ModelType.QWEN_3;
        if (!(options.useTornadovm() && greedy && fp16 && wiredLoop)) {
            System.err.println(
                    "[deviceSample] ignored — requires GPU + greedy (temperature 0) + FP16 + Llama/Mistral/Qwen3");
            System.clearProperty("llama.deviceSample");
        }
    }

    private static void runSingleInstruction(Model model, Sampler sampler, Options options) {
        // The CLI is generation policy, so it calls the generation package directly rather than
        // through the deprecated bridge on Model (Rule 8a).
        String response =
                org.beehive.gpullama3.generation.ModelGeneration.runInstructOnce(
                        model, sampler, options);
        System.out.println(response);
        if (SHOW_PERF_INTERACTIVE) {
            RunMetrics.printMetrics();
        }
    }

    /**
     * Entry point for running the LLaMA-based model with provided command-line arguments.
     *
     * <p>Initializes model options, loads the appropriate model (either AOT or on-demand),
     * configures the sampler, and runs either in interactive or single-instruction mode based on
     * the input options.
     *
     * @param args command-line arguments used to configure model path, temperature, seed, etc.
     * @throws IOException if model loading or file operations fail.
     */
    static void main(String[] args) throws IOException {
        Options options = Options.parseOptions(args);
        Model model = loadModel(options);
        guardDeviceSample(model, options);
        Sampler sampler = createSampler(model, options);

        if (options.interactive()) {
            org.beehive.gpullama3.generation.ModelGeneration.runInteractive(
                    model, sampler, options);
        } else {
            runSingleInstruction(model, sampler, options);
        }
    }
}
