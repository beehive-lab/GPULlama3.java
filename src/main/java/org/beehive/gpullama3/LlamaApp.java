package org.beehive.gpullama3;

import org.beehive.gpullama3.auxiliary.RunMetrics;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.model.Model;

import java.io.IOException;

import static org.beehive.gpullama3.inference.sampler.Sampler.createSampler;
import static org.beehive.gpullama3.model.loader.ModelLoader.loadModel;

public class LlamaApp {
    // Configuration flags for hardware acceleration and optimizations
    public static final boolean USE_VECTOR_API = Boolean.parseBoolean(System.getProperty("llama.VectorAPI", "true"));   // Enable Java Vector API for CPU acceleration
    public static final boolean SHOW_PERF_INTERACTIVE = Boolean.parseBoolean(System.getProperty("llama.ShowPerfInteractive", "true")); // Show performance metrics in interactive mode

    private static void runSingleInstruction(Model model, Sampler sampler, Options options) {
        String response = model.runInstructOnce(sampler, options);
        System.out.println(response);
        if (SHOW_PERF_INTERACTIVE) {
            RunMetrics.printMetrics();
        }
    }

    /**
     * Entry point for running the LLaMA-based model with provided command-line arguments.
     *
     * <p>Initializes model options, loads the appropriate model (either AOT or on-demand),
     * configures the sampler, and runs either in interactive or single-instruction mode based on the input options.</p>
     *
     * @param args
     *         command-line arguments used to configure model path, temperature, seed, etc.
     * @throws IOException
     *         if model loading or file operations fail.
     */
    static void main(String[] args) throws IOException {
        Options options = Options.parseOptions(args);
        long loadStart = System.nanoTime();
        Model model = loadModel(options);
        RunMetrics.setLoadDuration(System.nanoTime() - loadStart);
        Sampler sampler = createSampler(model, options);

        if (options.interactive()) {
            model.runInteractive(sampler, options);
        } else {
            runSingleInstruction(model, sampler, options);
        }
    }
}



