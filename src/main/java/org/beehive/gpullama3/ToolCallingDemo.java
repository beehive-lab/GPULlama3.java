package org.beehive.gpullama3;

import org.beehive.gpullama3.auxiliary.LastRunMetrics;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.format.ToolCallExtract;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.function.IntConsumer;

import static org.beehive.gpullama3.model.loader.ModelLoader.loadModel;

/**
 * Standalone demo that exercises tool calling end-to-end directly against the
 * GPULlama3.java inference engine — no Quarkus or LangChain4J required.
 *
 * Usage:
 *   ./llamaTornado --model /path/to/model.gguf --tool-demo
 *   ./llamaTornado --model /path/to/model.gguf --tool-demo --gpu --opencl
 */
public class ToolCallingDemo {

    private static final String TOOLS_JSON = """
            [
              {
                "type": "function",
                "function": {
                  "name": "list_directory",
                  "description": "List files and directories at the given path using ls",
                  "parameters": {
                    "type": "object",
                    "properties": {
                      "path": {
                        "type": "string",
                        "description": "Directory path to list. Defaults to current directory if omitted."
                      }
                    },
                    "required": []
                  }
                }
              }
            ]""";

    private static final String USER_PROMPT =
            "Use the list_directory tool to list the files in the current directory.";

    public static void main(String[] args) throws IOException {
        Options options = Options.parseOptions(ensurePrompt(args));

        System.out.println("=== GPULlama3 Tool Calling Demo ===");
        System.out.println("Model : " + options.modelPath());
        System.out.println("GPU   : " + options.useTornadovm());
        System.out.println("Prompt: " + USER_PROMPT);
        System.out.println();

        Model model = loadModel(options);
        Sampler sampler = Sampler.createSampler(model, options);
        ChatFormat chatFormat = model.chatFormat();

        // ── Build prompt ──────────────────────────────────────────────────────
        String toolSuffix = chatFormat.toolSystemPromptSuffix(TOOLS_JSON);
        List<Integer> promptTokens = new ArrayList<>();

        if (model.shouldAddBeginOfText()) {
            promptTokens.add(chatFormat.getBeginOfText());
        }
        if (model.shouldAddSystemPrompt()) {
            // Keep the system message minimal — just the tool instructions.
            // A preamble like "You are a helpful assistant" can cause the model
            // to answer from knowledge instead of calling the tool.
            promptTokens.addAll(chatFormat.encodeMessage(
                    new ChatFormat.Message(ChatFormat.Role.SYSTEM, toolSuffix.stripLeading())));
        }
        promptTokens.addAll(chatFormat.encodeMessage(
                new ChatFormat.Message(ChatFormat.Role.USER, USER_PROMPT)));
        promptTokens.addAll(chatFormat.encodeHeader(
                new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

        // ── First inference turn ──────────────────────────────────────────────
        Set<Integer> stopTokens = chatFormat.getToolAwareStopTokens();
        State state = model.createNewState();

        System.out.println("--- Raw model output ---");
        List<Integer> responseTokens = generateTokens(
                model, options, state, promptTokens, stopTokens, options.maxTokens(), sampler);
        System.out.println("\n--- End raw output ---\n");

        if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
            responseTokens.removeLast();
        }
        String rawResponse = model.tokenizer().decode(responseTokens);

        // ── Detect and execute tool call ──────────────────────────────────────
        Optional<ToolCallExtract> toolCall = chatFormat.extractToolCall(rawResponse);
        if (toolCall.isPresent()) {
            ToolCallExtract tc = toolCall.get();
            System.out.println("✓ Tool call detected!");
            System.out.println("  Function : " + tc.name());
            System.out.println("  Arguments: " + tc.argumentsJson());

            String toolResult = executeTool(tc);
            System.out.println("  Result   :\n" + toolResult);

            // ── Feed result back and get final answer ─────────────────────────
            List<Integer> continuation = new ArrayList<>(promptTokens);
            continuation.addAll(chatFormat.encodeToolCallAssistantTurn(tc));
            continuation.addAll(chatFormat.encodeToolResultTurn(null, tc.name(), toolResult));
            // Ask the model to summarise in plain text — prevents small models from
            // looping back into another tool call when they see tool defs in the system prompt.
            continuation.addAll(chatFormat.encodeMessage(
                    new ChatFormat.Message(ChatFormat.Role.USER,
                            "Based on the tool result above, please answer my question in plain text.")));
            continuation.addAll(chatFormat.encodeHeader(
                    new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

            State state2 = model.createNewState();
            System.out.println("--- Final answer ---");
            generateTokens(model, options, state2, continuation,
                    chatFormat.getStopTokens(), options.maxTokens(), sampler);
            System.out.println();

        } else {
            System.out.println("✗ No tool call detected.");
            System.out.println("  The model responded with plain text instead of a tool call.");
            System.out.println("  Note: reliable tool calling typically requires a 7B+ model.");
            System.out.println("\n  Full decoded response:\n" + rawResponse);
        }

        LastRunMetrics.printMetrics();
    }

    // ── Tool executor ─────────────────────────────────────────────────────────

    /** Maximum characters of tool output fed back into the prompt. */
    private static final int MAX_TOOL_RESULT_CHARS = 600;

    private static String executeTool(ToolCallExtract tc) {
        return switch (tc.name()) {
            case "list_directory" -> {
                String path = extractStringArg(tc.argumentsJson(), "path", ".");
                yield truncate(runProcess("ls", "-la", path));
            }
            default -> "Unknown tool: " + tc.name();
        };
    }

    private static String runProcess(String... command) {
        try {
            var process = new ProcessBuilder(command)
                    .redirectErrorStream(true)
                    .start();
            String output = new String(process.getInputStream().readAllBytes());
            process.waitFor();
            return output;
        } catch (Exception e) {
            return "Error executing command: " + e.getMessage();
        }
    }

    private static String truncate(String text) {
        if (text.length() <= MAX_TOOL_RESULT_CHARS) return text;
        return text.substring(0, MAX_TOOL_RESULT_CHARS) + "\n... (truncated)";
    }

    /**
     * Extracts a string value from a flat JSON object by key.
     * Falls back to {@code defaultValue} if the key is absent or if the value is a
     * nested object (which happens when a small model echoes the schema definition
     * instead of supplying an actual argument value).
     */
    private static String extractStringArg(String json, String key, String defaultValue) {
        String marker = "\"" + key + "\":";
        int idx = json.indexOf(marker);
        if (idx == -1) return defaultValue;
        int pos = idx + marker.length();
        while (pos < json.length() && Character.isWhitespace(json.charAt(pos))) pos++;
        if (pos >= json.length()) return defaultValue;
        // Nested object instead of a plain string — model echoed the schema, use default
        if (json.charAt(pos) == '{') return defaultValue;
        int valStart = json.indexOf('"', pos);
        if (valStart == -1) return defaultValue;
        int valEnd = json.indexOf('"', valStart + 1);
        if (valEnd == -1) return defaultValue;
        return json.substring(valStart + 1, valEnd);
    }

    // ── Inference helpers ─────────────────────────────────────────────────────

    private static List<Integer> generateTokens(
            Model model, Options options, State state,
            List<Integer> promptTokens, Set<Integer> stopTokens,
            int maxTokens, Sampler sampler) {

        IntConsumer tokenConsumer = token -> {
            if (model.tokenizer().shouldDisplayToken(token)) {
                System.out.print(model.tokenizer().decode(List.of(token)));
                System.out.flush();
            }
        };

        if (options.useTornadovm()) {
            TornadoVMMasterPlan plan = TornadoVMMasterPlan.initializeTornadoVMPlan(state, model);
            try {
                return model.generateTokensGPU(
                        state, 0, promptTokens, stopTokens, maxTokens, sampler,
                        false, tokenConsumer, plan);
            } finally {
                plan.freeTornadoExecutionPlan();
            }
        } else {
            return model.generateTokens(
                    state, 0, promptTokens, stopTokens, maxTokens, sampler,
                    false, tokenConsumer);
        }
    }

    private static String[] ensurePrompt(String[] args) {
        for (String arg : args) {
            if (arg.equals("--prompt") || arg.equals("-p")) return args;
        }
        String[] extended = Arrays.copyOf(args, args.length + 2);
        extended[args.length]     = "--prompt";
        extended[args.length + 1] = USER_PROMPT;
        return extended;
    }
}
