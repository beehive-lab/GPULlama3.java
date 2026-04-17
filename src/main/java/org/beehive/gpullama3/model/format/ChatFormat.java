package org.beehive.gpullama3.model.format;

import org.beehive.gpullama3.tokenizer.DevstralTokenizer;
import org.beehive.gpullama3.tokenizer.GraniteTokenizer;
import org.beehive.gpullama3.tokenizer.LlamaTokenizer;
import org.beehive.gpullama3.tokenizer.MistralTokenizer;
import org.beehive.gpullama3.tokenizer.Phi3Tokenizer;
import org.beehive.gpullama3.tokenizer.Qwen3Tokenizer;

import java.util.List;
import java.util.Optional;
import java.util.Set;

public interface ChatFormat {

    static ChatFormat create(Object tokenizer, ChatTokens chatTokens) {
        return switch (tokenizer) {
            case DevstralTokenizer devstralTokenizer -> new DevstralChatFormat(devstralTokenizer);
            case GraniteTokenizer graniteTokenizer -> new GraniteChatFormat(graniteTokenizer);
            case LlamaTokenizer llamaTokenizer -> new LlamaChatFormat(llamaTokenizer);
            case MistralTokenizer mistralTokenizer -> new MistralChatFormat(mistralTokenizer);
            case Qwen3Tokenizer qwen3Tokenizer -> new Qwen3ChatFormat(qwen3Tokenizer, chatTokens);
            case Phi3Tokenizer phi3Tokenizer -> new Phi3ChatFormat(phi3Tokenizer, chatTokens);
            default -> throw new IllegalArgumentException("Unsupported tokenizer type: " + tokenizer.getClass().getName());
        };
    }

    default ChatTokens chatTokens() {
        throw new UnsupportedOperationException("ChatFormat for Llama and Mistral does not support chatTokens");
    }

    List<Integer> encodeHeader(Message message);

    List<Integer> encodeMessage(Message message);

    int getBeginOfText();

    Set<Integer> getStopTokens();

    /**
     * Returns plain text to append to the system message content when tools are available.
     * The returned string is concatenated to the system message before encoding, so the
     * normal {@link #encodeMessage} path handles tokenization.
     *
     * @param toolsJson JSON array of tool definitions, e.g.
     *                  {@code [{"type":"function","function":{...}}]}
     */
    default String toolSystemPromptSuffix(String toolsJson) {
        throw new UnsupportedOperationException("Tool calling not supported for: " + getClass().getSimpleName());
    }

    /**
     * Re-encodes a prior assistant tool-call turn into the conversation token stream.
     * Used when replaying multi-turn history that contains a previous tool call.
     *
     * @param toolCall the tool call to encode (name + raw arguments JSON)
     */
    default List<Integer> encodeToolCallAssistantTurn(ToolCallExtract toolCall) {
        throw new UnsupportedOperationException("Tool calling not supported for: " + getClass().getSimpleName());
    }

    /**
     * Encodes a tool execution result message in the model-native format.
     *
     * @param toolCallId the ID of the originating tool call (may be ignored by some formats)
     * @param toolName   the name of the tool that was called
     * @param result     the result content string
     */
    default List<Integer> encodeToolResultTurn(String toolCallId, String toolName, String result) {
        throw new UnsupportedOperationException("Tool calling not supported for: " + getClass().getSimpleName());
    }

    /**
     * Detects and extracts a tool call from fully decoded model response text.
     * Returns {@link Optional#empty()} when the response is a plain text answer.
     *
     * @param responseText the fully decoded response from the model
     */
    default Optional<ToolCallExtract> extractToolCall(String responseText) {
        return Optional.empty();
    }

    /**
     * Stop tokens to use when tool calling is enabled.
     * Some models (LLaMA 3.1+) use a different end-of-turn token ({@code <|eom_id|>})
     * when emitting a tool call instead of a regular response.
     */
    default Set<Integer> getToolAwareStopTokens() {
        return getStopTokens();
    }

    record ChatTokens(String tStartHeader, String tEndHeader, String tEndOfTurn, String tEndOfText, String tEndOfTextFim) {
    }

    /**
     * Represents a single message in a LLM chat session.
     *
     * Each message is associated with a specific role (system, user, or assistant) and contains the textual content of that message.
     *
     * @param role
     *         the participant who issued the message (SYSTEM, USER, or ASSISTANT).
     * @param content
     *         the textual content of the message
     */
    record Message(Role role, String content) {
    }

    /**
     * Represents the role of a participant in a LLM chat conversation
     *
     * There are three standard roles:
     * <ul>
     * <li><strong>SYSTEM</strong> - sets the behavior and context of the assistant at the start of the conversation.</li>
     * <li><strong>USER</strong> - represents input from the human user.</li>
     * <li><strong>ASSISTANT</strong> - represents output from the AI assistant.</li>
     * </ul>
     *
     * @param name
     *         the string representation of the role
     */
    record Role(String name) {
        public static Role SYSTEM = new Role("system");
        public static Role USER = new Role("user");
        public static Role ASSISTANT = new Role("assistant");
        public static Role FIM_PREFIX = new ChatFormat.Role("fim_prefix");
        public static Role FIM_SUFFIX = new ChatFormat.Role("fim_suffix");
        public static Role FIM_MIDDLE = new ChatFormat.Role("fim_middle");

        @Override
        public String toString() {
            return name;
        }
    }

}