package org.beehive.gpullama3.tools;

import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.format.ToolCallExtract;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.function.IntConsumer;

/**
 * Framework-agnostic orchestrator for the tool-calling loop:
 * <pre>
 *   prompt → first generation → extract tool call? → execute → feed result → final answer
 * </pre>
 *
 * Supports Llama 3.1, Llama 3.2, and Qwen3 via the {@link ChatFormat} abstraction.
 * The session is single-use; create a new instance per request.
 */
public class ToolCallingSession {

    private final Model model;
    private final Sampler sampler;
    private final ToolRegistry registry;
    private final ToolCallingOptions options;

    public ToolCallingSession(Model model, Sampler sampler, ToolRegistry registry, ToolCallingOptions options) {
        this.model = model;
        this.sampler = sampler;
        this.registry = registry;
        this.options = options;
    }

    /** Run with no custom system prompt (the tool definitions become the system message). */
    public ToolCallingResult run(String userPrompt) {
        return run(null, userPrompt);
    }

    /**
     * Run with an optional system prompt prefix.  Tool definitions are appended to it.
     *
     * @param systemPrompt base system prompt, or {@code null} for tools-only
     * @param userPrompt   the user's request
     */
    public ToolCallingResult run(String systemPrompt, String userPrompt) {
        ChatFormat chatFormat = model.chatFormat();
        String toolsJson = registry.toToolsJson();

        // Build effective system and user messages according to the model's tool injection strategy.
        // Formats like Llama 3.2 inject tool definitions into the first user message
        // (injectsToolsInUserMessage() == true); others (Qwen3, Mistral) append to the system message.
        String effectiveSystem;
        String effectiveUser;
        if (chatFormat.injectsToolsInUserMessage()) {
            String prefix = chatFormat.toolSystemMessagePrefix();
            effectiveSystem = systemPrompt == null
                    ? prefix.stripLeading()
                    : prefix + systemPrompt;
            effectiveUser = chatFormat.toolFirstUserMessagePrefix(toolsJson) + userPrompt;
        } else {
            String toolSuffix = chatFormat.toolSystemPromptSuffix(toolsJson);
            effectiveSystem = systemPrompt == null
                    ? toolSuffix.stripLeading()
                    : systemPrompt + toolSuffix;
            effectiveUser = userPrompt;
        }

        log("\n[DEBUG] model: %s", model.getClass().getSimpleName());
        log("[DEBUG] chatFormat: %s", chatFormat.getClass().getSimpleName());
        log("[DEBUG] shouldAddSystemPrompt: %s", model.shouldAddSystemPrompt());
        log("[DEBUG] toolAwareStopTokens: %s", chatFormat.getToolAwareStopTokens());
        log("[DEBUG] ── effective system prompt ──────────────────────────");
        log("%s", effectiveSystem);
        log("[DEBUG] ── end system prompt ─────────────────────────────────");

        // ── Build initial prompt tokens ───────────────────────────────────────
        List<Integer> promptTokens = new ArrayList<>();
        if (model.shouldAddBeginOfText()) {
            promptTokens.add(chatFormat.getBeginOfText());
        }
        if (model.shouldAddSystemPrompt() && !effectiveSystem.isBlank()) {
            promptTokens.addAll(chatFormat.encodeMessage(
                    new ChatFormat.Message(ChatFormat.Role.SYSTEM, effectiveSystem)));
        }
        promptTokens.addAll(chatFormat.encodeMessage(
                new ChatFormat.Message(ChatFormat.Role.USER, effectiveUser)));
        promptTokens.addAll(chatFormat.encodeHeader(
                new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

        Set<Integer> toolStopTokens = chatFormat.getToolAwareStopTokens();

        List<ToolCallExtract> callsMade = new ArrayList<>();
        List<ToolResult> toolResults = new ArrayList<>();

        State state = model.createNewState();
        TornadoVMMasterPlan plan = options.useGpu()
                ? TornadoVMMasterPlan.initializeTornadoVMPlan(state, model)
                : null;

        try {
        // ── Tool round-trip loop ──────────────────────────────────────────────
        for (int round = 0; round < options.maxRoundTrips(); round++) {
            log("\n--- First generation (round %d) ---", round + 1);
            List<Integer> responseTokens = generateTokens(state, plan, promptTokens, toolStopTokens);

            // strip trailing stop token
            if (!responseTokens.isEmpty() && toolStopTokens.contains(responseTokens.getLast())) {
                responseTokens.removeLast();
            }
            String rawResponse = model.tokenizer().decode(responseTokens);

            log("[DEBUG] raw response (round %d): >>>%s<<<", round + 1, rawResponse);
            log("[DEBUG] response tokens: %d  (stop token stripped: %s)",
                    responseTokens.size(),
                    !responseTokens.isEmpty() && toolStopTokens.contains(responseTokens.getLast()) ? "yes" : "no — stop token was removed before decoding");

            List<ToolCallExtract> batchCalls = chatFormat.extractAllToolCalls(rawResponse);
            log("[DEBUG] extractAllToolCalls found: %d call(s)", batchCalls.size());
            if (batchCalls.isEmpty()) {
                log("\n--- No tool call detected; returning plain text response ---");
                return new ToolCallingResult(rawResponse, callsMade, toolResults, false);
            }

            // ── Execute all tool calls found in this response ─────────────────
            promptTokens = new ArrayList<>(promptTokens);
            for (ToolCallExtract call : batchCalls) {
                callsMade.add(call);
                log("\n[Tool call] %s(%s)", call.name(), call.argumentsJson());

                ToolResult result = registry.execute(call);
                toolResults.add(result);
                log("[Tool result] %s", result.isError() ? "ERROR: " + result.error() : truncate(result.resultText()));

                String feedbackContent = result.isError()
                        ? "Tool '" + call.name() + "' failed: " + result.error()
                        : "Tool '" + call.name() + "' returned:\n" + truncate(result.resultText());

                promptTokens.addAll(chatFormat.encodeToolCallAssistantTurn(call));
                promptTokens.addAll(chatFormat.encodeToolResultTurn(null, call.name(), feedbackContent));
            }
            promptTokens.addAll(chatFormat.encodeMessage(
                    new ChatFormat.Message(ChatFormat.Role.USER,
                            "Using the tool results above, call the next required tool or answer the user's original question if all steps are complete.")));
            promptTokens.addAll(chatFormat.encodeHeader(
                    new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
        }

        // ── Final answer after all tool round-trips ───────────────────────────
        log("\n--- Final generation ---");
        List<Integer> finalTokens = generateTokens(state, plan, promptTokens, chatFormat.getStopTokens());
        if (!finalTokens.isEmpty() && chatFormat.getStopTokens().contains(finalTokens.getLast())) {
            finalTokens.removeLast();
        }
        String finalAnswer = model.tokenizer().decode(finalTokens);

        boolean hitLimit = callsMade.size() >= options.maxRoundTrips()
                && chatFormat.extractToolCall(finalAnswer).isPresent();

        return new ToolCallingResult(finalAnswer, callsMade, toolResults, hitLimit);

        } finally {
            if (plan != null) plan.freeTornadoExecutionPlan();
        }
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    private List<Integer> generateTokens(State state, TornadoVMMasterPlan plan,
                                          List<Integer> prompt, Set<Integer> stopTokens) {
        IntConsumer tokenConsumer = options.verbose() ? this::printToken : null;

        if (options.useGpu()) {
            return model.generateTokensGPU(
                    state, 0, prompt, stopTokens, options.maxTokens(), sampler,
                    false, tokenConsumer, plan);
        } else {
            return model.generateTokens(
                    state, 0, prompt, stopTokens, options.maxTokens(), sampler,
                    false, tokenConsumer);
        }
    }

    private void printToken(int token) {
        if (model.tokenizer().shouldDisplayToken(token)) {
            System.out.print(model.tokenizer().decode(List.of(token)));
            System.out.flush();
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private String truncate(String text) {
        if (text == null) return "";
        if (text.length() <= options.maxToolResultChars()) return text;
        return text.substring(0, options.maxToolResultChars()) + "\n... (truncated)";
    }

    private void log(String fmt, Object... args) {
        if (options.verbose()) {
            System.out.printf((fmt) + "%n", args);
        }
    }
}
