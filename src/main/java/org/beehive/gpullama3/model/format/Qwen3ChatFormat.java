package org.beehive.gpullama3.model.format;

import org.beehive.gpullama3.tokenizer.Qwen3Tokenizer;

import java.util.*;

/**
 * Utility tailored for the Chat Markup Language (ChatML) prompt format.
 */
public class Qwen3ChatFormat implements ChatFormat {

    protected final int beginOfText;
    protected final int startHeader;
    protected final int endHeader;
    protected final int endOfTurn;
    protected final int endOfText;
    protected final int endOfMessage;
    protected final int endOfTextFim;
    protected final int imStart; // beginOfText
    protected final int imEnd; // endOfText
    protected final int fimPrefix;
    protected final int fimSuffix;
    protected final int fimMiddle;
    protected Qwen3Tokenizer tokenizer;
    protected ChatTokens chatTokens;

    public Qwen3ChatFormat(Qwen3Tokenizer tokenizer, ChatTokens chatTokens) {
        this.tokenizer = tokenizer;
        this.chatTokens = chatTokens;
        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();
        this.beginOfText = specialTokens.getOrDefault("", -1);
        this.startHeader = specialTokens.getOrDefault(chatTokens.tStartHeader(), -1);
        this.endHeader = specialTokens.getOrDefault(chatTokens.tEndHeader(), -1);
        this.endOfTurn = specialTokens.getOrDefault(chatTokens.tEndOfTurn(), -1);
        this.endOfText = specialTokens.getOrDefault(chatTokens.tEndOfText(), -1);
        this.endOfTextFim = specialTokens.getOrDefault(chatTokens.tEndOfTextFim(), -1);
        this.endOfMessage = specialTokens.getOrDefault("", -1); // Use default value if key not found

        this.imStart = startHeader;
        this.imEnd = endHeader;

        this.fimPrefix = specialTokens.getOrDefault("<|fim_prefix|>", -1);
        this.fimSuffix = specialTokens.getOrDefault("<|fim_suffix|>", -1);
        this.fimMiddle = specialTokens.getOrDefault("<|fim_middle|>", -1);
    }

    public ChatTokens chatTokens() {
        return chatTokens;
    }

    @Override
    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();
        if (endHeader == -1) {
            // DeepSeek-R1
            String sToken = switch (message.role().name()) {
                case "system" -> null;
                case "user" -> "<｜User｜>";
                case "assistant" -> "<｜Assistant｜>";
                case "fim_prefix" -> "<|fim_prefix|>";
                case "fim_middle" -> "<|fim_middle|>";
                case "fim_suffix" -> "<|fim_suffix|>";
                default -> null;
            };
            if (sToken != null) {
                Integer token = tokenizer.getSpecialTokens().get(sToken);
                if (token == null) {
                    throw new IllegalStateException(String.format("Unknown token '%s'", sToken));
                }
                tokens.add(token);
            }
        } else if (Role.FIM_PREFIX.equals(message.role())) {
            // fill-in-the-middle, token fim_prefix.
            tokens.add(fimPrefix);
        } else if (Role.FIM_SUFFIX.equals(message.role())) {
            tokens.add(fimSuffix);
        } else if (Role.FIM_MIDDLE.equals(message.role())) {
            tokens.add(fimMiddle);
        } else {
            // Add the special token directly, don't try to encode it
            tokens.add(imStart);
            // Encode the role name as ordinary text (no special tokens in role names)
            tokens.addAll(this.tokenizer.encodeOrdinaryAsList(message.role().name()));
            tokens.addAll(this.tokenizer.encodeOrdinaryAsList("\n"));
        }
        return tokens;
    }

    @Override
    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = this.encodeHeader(message);
        // Encode message content as ordinary text
        tokens.addAll(this.tokenizer.encodeOrdinaryAsList(message.content().strip()));
        boolean isFim = Role.FIM_PREFIX.equals(message.role()) || Role.FIM_SUFFIX.equals(message.role()) || Role.FIM_MIDDLE.equals(message.role());
        if (imEnd != -1 && !isFim) {
            // Add the end token directly
            tokens.add(imEnd);
        }
        return tokens;
    }

    @Override
    public int getBeginOfText() {
        if (beginOfText == -1) {
            // deepseek-r1
            return startHeader;
        } else {
            return beginOfText;
        }
    }

    @Override
    public Set<Integer> getStopTokens() {
        if (imEnd == -1 && endOfText == -1) {
            throw new IllegalStateException("No stop token is defined.");
        }

        // Only add valid token IDs (not -1)
        Set<Integer> stopTokens = new HashSet<>();
        if (imEnd != -1) {
            stopTokens.add(imEnd);
        }
        if (endOfText != -1) {
            stopTokens.add(endOfText);
        }
        if (endOfTextFim != -1) {
            stopTokens.add(endOfTextFim);
        }

        return stopTokens;
    }

    // ── Tool calling ──────────────────────────────────────────────────────────

    /**
     * Qwen3 tool calling system prompt suffix.
     * Appended to the system message; instructs the model to wrap tool calls in
     * {@code <tool_call>…</tool_call>} XML tags.
     */
    @Override
    public String toolSystemPromptSuffix(String toolsJson) {
        return "\n\n# Tools\n\n"
                + "You may call one or more functions to assist with the user query.\n\n"
                + "You are provided with function signatures within <tools></tools> XML tags:\n"
                + "<tools>\n"
                + toolsJson
                + "\n</tools>\n\n"
                + "For each function call, return a json object with function name and arguments "
                + "within <tool_call></tool_call> XML tags:\n"
                + "<tool_call>\n"
                + "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
                + "</tool_call>";
    }

    /**
     * Re-encodes a prior assistant tool-call turn for multi-turn history.
     * Format: {@code <|im_start|>assistant\n<tool_call>\nJSON\n</tool_call><|im_end|>}
     */
    @Override
    public List<Integer> encodeToolCallAssistantTurn(ToolCallExtract toolCall) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(imStart);
        tokens.addAll(tokenizer.encodeOrdinaryAsList("assistant\n"));
        String json = "{\"name\":\"" + toolCall.name() + "\",\"arguments\":" + toolCall.argumentsJson() + "}";
        tokens.addAll(tokenizer.encodeOrdinaryAsList("<tool_call>\n" + json + "\n</tool_call>"));
        if (imEnd != -1) {
            tokens.add(imEnd);
        }
        return tokens;
    }

    /**
     * Encodes a tool result using the Qwen3 "tool" role.
     * Format: {@code <|im_start|>tool\nresult<|im_end|>}
     */
    @Override
    public List<Integer> encodeToolResultTurn(String toolCallId, String toolName, String result) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(imStart);
        tokens.addAll(tokenizer.encodeOrdinaryAsList("tool\n"));
        tokens.addAll(tokenizer.encodeOrdinaryAsList(result));
        if (imEnd != -1) {
            tokens.add(imEnd);
        }
        return tokens;
    }

    /**
     * Detects a tool call enclosed in {@code <tool_call>…</tool_call>} tags.
     */
    @Override
    public Optional<ToolCallExtract> extractToolCall(String responseText) {
        int start = responseText.indexOf("<tool_call>");
        int end = responseText.lastIndexOf("</tool_call>");
        if (start == -1 || end == -1) {
            return Optional.empty();
        }
        String json = responseText.substring(start + "<tool_call>".length(), end).strip();
        return parseToolCallJson(json, "arguments");
    }

    /**
     * Parses {@code name} and the value of {@code argsKey} out of a tool-call JSON object.
     * Uses brace-counting to extract nested argument objects correctly.
     * Avoids a JSON-library dependency.
     */
    private static Optional<ToolCallExtract> parseToolCallJson(String json, String argsKey) {
        // extract "name"
        String name = extractStringValue(json, "name");
        if (name == null) {
            return Optional.empty();
        }

        // extract arguments object using brace-counting
        String argsJson = extractNestedObject(json, argsKey);
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
