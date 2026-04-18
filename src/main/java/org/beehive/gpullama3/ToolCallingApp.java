package org.beehive.gpullama3;

import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.format.ToolCallExtract;
import org.beehive.gpullama3.model.format.ToolCallParserUtils;
import org.beehive.gpullama3.tools.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.beehive.gpullama3.inference.sampler.Sampler.createSampler;
import static org.beehive.gpullama3.model.loader.ModelLoader.loadModel;

/**
 * Standalone tool-calling entry point for GPULlama3.java.
 *
 * Uses the same command-line flags as {@link LlamaApp} plus the tool-calling loop
 * provided by {@link ToolCallingSession}. A {@code listDirectory} tool is registered
 * as a built-in demo; extend {@link #buildRegistry()} to add more tools.
 *
 * <pre>
 * java @$TORNADOVM_HOME/tornado-argfile \
 *     --add-modules jdk.incubator.vector --enable-preview \
 *     -cp gpu-llama3.jar org.beehive.gpullama3.ToolCallingApp \
 *     --model /path/to/model.gguf \
 *     --prompt "Show me what is inside /tmp" \
 *     --use-tornadovm true
 * </pre>
 */
public class ToolCallingApp {

    public static void main(String[] args) throws IOException {
        Options options = Options.parseOptions(args);
        Model model = loadModel(options);
        Sampler sampler = createSampler(model, options);

        ToolRegistry registry = buildRegistry();
        ToolCallingOptions tcOptions = ToolCallingOptions.from(options);
        ToolCallingSession session = new ToolCallingSession(model, sampler, registry, tcOptions);

        ToolCallingResult result = session.run(options.systemPrompt(), options.prompt());

        if (!tcOptions.verbose()) {
            // verbose=false means ToolCallingSession did not stream tokens — print the answer now
            System.out.println(result.finalAnswer());
        }
    }

    // ── Tool registry ─────────────────────────────────────────────────────────

    private static ToolRegistry buildRegistry() {
        ToolRegistry registry = new ToolRegistry();
        registry.register(listDirectoryDefinition(), ToolCallingApp::listDirectory);
        return registry;
    }

    private static ToolDefinition listDirectoryDefinition() {
        return new ToolDefinition(
                "listDirectory",
                "Lists the contents of a directory on the local filesystem. " +
                "Returns file names and metadata. " +
                "Use this when the user asks what is inside a directory or folder.",
                """
                {"type":"object","properties":{"path":{"type":"string",\
                "description":"Absolute path of the directory to list, e.g. /tmp or /home/orion"}},\
                "required":["path"]}""");
    }

    private static ToolResult listDirectory(ToolCallExtract call) {
        String path = ToolCallParserUtils.extractStringValue(call.argumentsJson(), "path");
        if (path == null || path.isBlank()) {
            return ToolResult.failure("listDirectory",
                    "Could not extract 'path' from arguments: " + call.argumentsJson());
        }
        Path dir = Path.of(path);
        if (!dir.isAbsolute())
            return ToolResult.failure("listDirectory", "Path must be absolute. Got: " + path);
        if (!Files.exists(dir))
            return ToolResult.failure("listDirectory", "Path does not exist: " + path);
        if (!Files.isDirectory(dir))
            return ToolResult.failure("listDirectory", "Not a directory: " + path);
        try {
            StringBuilder sb = new StringBuilder("Contents of ").append(path).append(":\n");
            try (var stream = Files.list(dir)) {
                stream.sorted().forEach(entry -> {
                    boolean isDir = Files.isDirectory(entry);
                    long size = 0;
                    try { if (!isDir) size = Files.size(entry); } catch (IOException ignored) {}
                    sb.append(isDir ? "dir:  " : "file: ")
                      .append(entry.getFileName())
                      .append(isDir ? "" : ", " + size + " bytes")
                      .append("\n");
                });
            }
            return ToolResult.success("listDirectory", sb.toString());
        } catch (IOException e) {
            return ToolResult.failure("listDirectory", e.getMessage());
        }
    }
}
