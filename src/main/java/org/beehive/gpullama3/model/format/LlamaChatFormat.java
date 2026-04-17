package org.beehive.gpullama3.model.format;

import org.beehive.gpullama3.tokenizer.LlamaTokenizer;
import org.beehive.gpullama3.tokenizer.Tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public class LlamaChatFormat implements ChatFormat {

    protected final Tokenizer tokenizer;
    protected final int beginOfText;
    protected final int endHeader;
    protected final int startHeader;
    protected final int endOfTurn;
    protected final int endOfText;
    protected final int endOfMessage;
    protected final int pythonTag;
    protected final Set<Integer> stopTokens;

    public LlamaChatFormat(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();
        this.beginOfText = specialTokens.get("<|begin_of_text|>");
        this.startHeader = specialTokens.get("<|start_header_id|>");
        this.endHeader = specialTokens.get("<|end_header_id|>");
        this.endOfTurn = specialTokens.get("<|eot_id|>");
        this.endOfText = specialTokens.get("<|end_of_text|>");
        this.endOfMessage = specialTokens.getOrDefault("<|eom_id|>", -1); // only in 3.1
        this.pythonTag = specialTokens.getOrDefault("<|python_tag|>", -1);  // only in 3.1
        this.stopTokens = Set.of(endOfText, endOfTurn);
    }

    @Override
    public int getBeginOfText() {
        return beginOfText;
    }

    @Override
    public Set<Integer> getStopTokens() {
        return stopTokens;
    }

    @Override
    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(startHeader);
        tokens.addAll(tokenizer.encodeAsList(message.role().name()));
        tokens.add(endHeader);
        tokens.addAll(tokenizer.encodeAsList("\n"));
        return tokens;
    }

    @Override
    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = encodeHeader(message);
        tokens.addAll(tokenizer.encodeAsList(message.content().strip()));
        tokens.add(endOfTurn);
        return tokens;
    }

    public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<Message> dialog) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(beginOfText);
        for (Message message : dialog) {
            tokens.addAll(encodeMessage(message));
        }
        if (appendAssistantTurn) {
            // Add the start of an assistant message for the model to complete.
            tokens.addAll(encodeHeader(new Message(ChatFormat.Role.ASSISTANT, "")));
        }
        return tokens;
    }

    // ── Tool calling ──────────────────────────────────────────────────────────

    /**
     * LLaMA 3.1 tool calling system prompt suffix.
     * Instructs the model to respond with JSON using the {"name":…,"parameters":{…}} format.
     */
    @Override
    public String toolSystemPromptSuffix(String toolsJson) {
        return "\n\nGiven the following functions, please respond with a JSON for a function call "
                + "with its proper arguments that best answers the given prompt.\n\n"
                + "Respond in the format {\"name\": function name, \"parameters\": dictionary of "
                + "argument name and its value}. Do not use variables.\n\n"
                + toolsJson;
    }

    /**
     * Re-encodes a prior assistant tool-call turn for multi-turn history.
     * Format: {@code <|start_header_id|>assistant<|end_header_id|>\n<|python_tag|>JSON<|eom_id|>}
     */
    @Override
    public List<Integer> encodeToolCallAssistantTurn(ToolCallExtract toolCall) {
        List<Integer> tokens = new ArrayList<>(encodeHeader(new Message(Role.ASSISTANT, "")));
        if (pythonTag != -1) {
            tokens.add(pythonTag);
        }
        String json = "{\"name\":\"" + toolCall.name() + "\",\"parameters\":" + toolCall.argumentsJson() + "}";
        tokens.addAll(tokenizer.encodeAsList(json));
        if (endOfMessage != -1) {
            tokens.add(endOfMessage);
        } else {
            tokens.add(endOfTurn);
        }
        return tokens;
    }

    /**
     * Encodes a tool result using the LLaMA "ipython" role.
     * Format: {@code <|start_header_id|>ipython<|end_header_id|>\nresult<|eot_id|>}
     */
    @Override
    public List<Integer> encodeToolResultTurn(String toolCallId, String toolName, String result) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(startHeader);
        tokens.addAll(tokenizer.encodeAsList("ipython"));
        tokens.add(endHeader);
        tokens.addAll(tokenizer.encodeAsList("\n"));
        tokens.addAll(tokenizer.encodeAsList(result));
        tokens.add(endOfTurn);
        return tokens;
    }

    /**
     * Detects a tool call in the decoded response text.
     *
     * <p>Two formats are recognised:</p>
     * <ol>
     *   <li><strong>LLaMA 3.1 native</strong>: {@code <|python_tag|>{"name":…,"parameters":{…}}}</li>
     *   <li><strong>Fallback</strong>: raw JSON object (possibly wrapped in markdown code fences),
     *       as produced by smaller models that follow the system-prompt instructions but skip
     *       the special token prefix.</li>
     * </ol>
     */
    @Override
    public Optional<ToolCallExtract> extractToolCall(String responseText) {
        // ── 1. Native LLaMA 3.1 format: <|python_tag|>{...} ──────────────────
        int idx = responseText.indexOf("<|python_tag|>");
        if (idx != -1) {
            String json = responseText.substring(idx + "<|python_tag|>".length()).strip();
            return parseToolCallJson(json);
        }

        // ── 2. Fallback: raw JSON, possibly inside markdown code fences ───────
        String stripped = stripMarkdownFences(responseText.strip());
        if (stripped.startsWith("{")) {
            return parseToolCallJson(stripped);
        }

        return Optional.empty();
    }

    /**
     * Adds {@code <|eom_id|>} to the stop tokens when tools are enabled.
     * LLaMA 3.1 ends tool-call turns with {@code <|eom_id|>} instead of {@code <|eot_id|>}.
     */
    @Override
    public Set<Integer> getToolAwareStopTokens() {
        if (endOfMessage != -1) {
            return Set.of(endOfText, endOfTurn, endOfMessage);
        }
        return stopTokens;
    }

    /**
     * Strips surrounding markdown code fences (``` or ```json / ```python etc.) if present.
     */
    private static String stripMarkdownFences(String text) {
        if (!text.startsWith("```")) {
            return text;
        }
        int firstNewline = text.indexOf('\n');
        if (firstNewline == -1) {
            return text;
        }
        String body = text.substring(firstNewline + 1);
        if (body.endsWith("```")) {
            body = body.substring(0, body.length() - 3).stripTrailing();
        }
        return body.strip();
    }

    /**
     * Parses a tool call JSON object into a {@link ToolCallExtract}.
     *
     * <p>Accepts both formats produced by LLaMA variants:</p>
     * <ul>
     *   <li>{@code {"name":"fn","parameters":{…}}} — LLaMA 3.1 native</li>
     *   <li>{@code {"function":"fn","parameters":{…}}} — produced by some fine-tunes</li>
     *   <li>{@code {"name":"fn","arguments":{…}}} — alternative key</li>
     * </ul>
     * Uses brace-counting to extract nested argument objects correctly.
     */
    private static Optional<ToolCallExtract> parseToolCallJson(String json) {
        // ── extract tool name: try "name" then "function" ─────────────────────
        String name = extractStringValue(json, "name");
        if (name == null) {
            name = extractStringValue(json, "function");
        }
        if (name == null) {
            return Optional.empty();
        }

        // ── extract arguments object: try "parameters" then "arguments" ───────
        String argsJson = extractNestedObject(json, "parameters");
        if (argsJson == null) {
            argsJson = extractNestedObject(json, "arguments");
        }
        if (argsJson == null) {
            argsJson = "{}"; // tool call with no arguments
        }

        return Optional.of(new ToolCallExtract(name, argsJson));
    }

    /** Extracts the string value for {@code "key":"<value>"} from a JSON object. */
    private static String extractStringValue(String json, String key) {
        String marker = "\"" + key + "\":";
        int markerIdx = json.indexOf(marker);
        if (markerIdx == -1) return null;
        int quoteStart = json.indexOf('"', markerIdx + marker.length());
        if (quoteStart == -1) return null;
        int quoteEnd = json.indexOf('"', quoteStart + 1);
        if (quoteEnd == -1) return null;
        return json.substring(quoteStart + 1, quoteEnd);
    }

    /**
     * Extracts the JSON object value for {@code "key":{…}} using brace-counting,
     * so nested objects are handled correctly regardless of what follows.
     */
    private static String extractNestedObject(String json, String key) {
        String marker = "\"" + key + "\":";
        int markerIdx = json.indexOf(marker);
        if (markerIdx == -1) return null;
        int braceStart = json.indexOf('{', markerIdx + marker.length());
        if (braceStart == -1) return null;
        int depth = 0;
        for (int i = braceStart; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '{') depth++;
            else if (c == '}') {
                depth--;
                if (depth == 0) return json.substring(braceStart, i + 1);
            }
        }
        return null; // unbalanced JSON
    }
}