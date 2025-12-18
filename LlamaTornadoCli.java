//JAVA 21
//PREVIEW
//DEPS io.github.beehive-lab:gpu-llama3:0.3.2-dev
//DEPS io.github.beehive-lab:tornado-api:2.2.0
//DEPS io.github.beehive-lab:tornado-runtime:2.2.0

//SOURCES TornadoFlags.java
// === Set to not get annoying warnings about annotation processing
//JAVAC_OPTIONS -proc:full

// Compiler options
//JAVAC_OPTIONS --enable-preview
//JAVAC_OPTIONS --add-modules=jdk.incubator.vector

// JVM options for basic setup
//JAVA_OPTIONS --enable-preview
//JAVA_OPTIONS --add-modules=jdk.incubator.vector

package org.beehive.gpullama3.cli;

import org.beehive.gpullama3.Options;
import org.beehive.gpullama3.auxiliary.LastRunMetrics;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.model.Model;

import java.io.IOException;

import static org.beehive.gpullama3.inference.sampler.Sampler.createSampler;
import static org.beehive.gpullama3.model.loader.ModelLoader.loadModel;

/**
 * LlamaTornadoCli - Pure Java CLI for running llama-tornado models
 *
 * This class provides a standalone command-line interface for running LLaMA models
 * with TornadoVM acceleration. It can be executed directly with JBang or as a
 * compiled Java application.
 *
 * Usage with JBang:
 *   jbang LlamaTornadoCli.java --model path/to/model.gguf --prompt "Your prompt here"
 *
 * Usage as compiled application:
 *   java --enable-preview --add-modules jdk.incubator.vector \
 *        -cp target/gpu-llama3-0.3.1.jar \
 *        org.beehive.gpullama3.cli.LlamaTornadoCli \
 *        --model path/to/model.gguf --prompt "Your prompt here"
 *
 * Examples:
 *   # Interactive chat mode
 *   jbang LlamaTornadoCli.java -m model.gguf --interactive
 *
 *   # Single instruction mode
 *   jbang LlamaTornadoCli.java -m model.gguf -p "Explain quantum computing"
 *
 *   # With TornadoVM acceleration
 *   jbang LlamaTornadoCli.java -m model.gguf -p "Hello" --use-tornadovm true
 *
 *   # Custom temperature and sampling
 *   jbang LlamaTornadoCli.java -m model.gguf -p "Tell me a story" \
 *        --temperature 0.7 --top-p 0.9 --max-tokens 512
 */
public class LlamaTornadoCli {

    // Configuration flags
    public static final boolean USE_VECTOR_API = Boolean.parseBoolean(
        System.getProperty("llama.VectorAPI", "true"));
    public static final boolean SHOW_PERF_INTERACTIVE = Boolean.parseBoolean(
        System.getProperty("llama.ShowPerfInteractive", "true"));

    /**
     * Run a single instruction and display the response
     */
    private static void runSingleInstruction(Model model, Sampler sampler, Options options) {
        String response = model.runInstructOnce(sampler, options);
        System.out.println(response);
        if (SHOW_PERF_INTERACTIVE) {
            LastRunMetrics.printMetrics();
        }
    }

    /**
     * Main entry point for the CLI application
     *
     * @param args command-line arguments (see Options.parseOptions for details)
     * @throws IOException if model loading fails
     */
    public static void main(String[] args) throws IOException {
        // Print banner
        printBanner();

        // Check if help requested
        if (args.length == 0 || hasHelpFlag(args)) {
            Options.printUsage(System.out);
            System.exit(0);
        }

        try {
            // Parse options
            Options options = Options.parseOptions(args);

            // Load model
            Model model = loadModel(options);

            // Create sampler
            Sampler sampler = createSampler(model, options);

            // Run in interactive or single-instruction mode
            if (options.interactive()) {
                System.out.println("Starting interactive chat mode...");
                System.out.println("Type your messages below (Ctrl+C to exit):");
                System.out.println();
                model.runInteractive(sampler, options);
            } else {
                runSingleInstruction(model, sampler, options);
            }
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Check if help flag is present in arguments
     */
    private static boolean hasHelpFlag(String[] args) {
        for (String arg : args) {
            if (arg.equals("--help") || arg.equals("-h")) {
                return true;
            }
        }
        return false;
    }

    /**
     * Print ASCII banner
     */
    private static void printBanner() {
        System.out.println("""
            ╔══════════════════════════════════════════════════════════╗
            ║        Llama-Tornado CLI - GPU-Accelerated LLM           ║
            ║           Powered by TornadoVM & Java 21                 ║
            ╚══════════════════════════════════════════════════════════╝
            """);
    }
}
