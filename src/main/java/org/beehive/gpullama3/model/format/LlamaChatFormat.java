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
        return "\n\n# Tools\n\n"
                + "You may call one or more functions to assist with the user query.\n\n"
                + "You are provided with function signatures within <tools></tools> XML tags:\n\n"
                + "<tools>\n" + toolsJson + "\n</tools>\n\n"
                + "IMPORTANT: the \"name\" field in your tool call MUST be exactly one of the function names "
                + "listed inside <tools> above — not a path, not a word from the user's message.\n\n"
                + "For each function call, return a json object with function name and arguments "
                + "within <tool_call></tool_call> XML tags:\n\n"
                + "<tool_call>\n"
                + "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
                + "</tool_call>";
    }

    /**
     * Re-encodes a prior assistant tool-call turn for multi-turn history.
     * Format: {@code <|start_header_id|>assistant<|end_header_id|>\n<|python_tag|>JSON<|eom_id|>}
     */
    @Override
    public List<Integer> encodeToolCallAssistantTurn(ToolCallExtract toolCall) {
        List<Integer> tokens = new ArrayList<>(encodeHeader(new Message(Role.ASSISTANT, "")));
        String json = "<tool_call>\n{\"name\":\"" + toolCall.name() + "\",\"arguments\":" + toolCall.argumentsJson() + "}\n</tool_call>";
        tokens.addAll(tokenizer.encodeAsList(json));
        tokens.add(endOfTurn);
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
     * Supports LLaMA 3.1 (native {@code <|python_tag|>} + {@code "parameters"} key),
     * LLaMA 3.2 ({@code "arguments"} key, tag often absent), and a raw-JSON fallback
     * for smaller models. Delegates to {@link ToolCallParserUtils#parseLlamaResponse}.
     */
    @Override
    public Optional<ToolCallExtract> extractToolCall(String responseText) {
        return ToolCallParserUtils.parseLlamaResponse(responseText);
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

}