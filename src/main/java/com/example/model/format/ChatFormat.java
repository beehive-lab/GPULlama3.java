package com.example.model.format;

import com.example.tokenizer.impl.LlamaTokenizer;
import com.example.tokenizer.impl.MistralTokenizer;
import com.example.tokenizer.impl.Qwen3Tokenizer;

import java.util.List;
import java.util.Set;

public interface ChatFormat {

    default ChatTokens chatTokens() {
        throw new UnsupportedOperationException("ChatFormat for Llama and Mistral does not support chatTokens");
    }

    public record ChatTokens(String tStartHeader, String tEndHeader, String tEndOfTurn, String tEndOfText, String tEndOfTextFim) { }

    static ChatFormat create(Object tokenizer, ChatTokens chatTokens) {
        if (tokenizer instanceof LlamaTokenizer llamaTokenizer) {
            return new LlamaChatFormat(llamaTokenizer);
        } else if (tokenizer instanceof MistralTokenizer mistralTokenizer) {
            return new MistralChatFormat(mistralTokenizer);
        } else if (tokenizer instanceof Qwen3Tokenizer qwen3Tokenizer) {
            return new Qwen3ChatFormat(qwen3Tokenizer, chatTokens);
        } else {
            throw new IllegalArgumentException("Unsupported tokenizer type: " + tokenizer.getClass().getName());
        }
    }

    List<Integer> encodeHeader(Message message);

    List<Integer> encodeMessage(Message message);

    int getBeginOfText();

    Set<Integer> getStopTokens();

    /**
     * Represents a single message in a LLM chat session.
     *
     * Each message is associated with a specific role (system, user, or assistant)
     * and contains the textual content of that message.
     *
     * @param role the participant who issued the message (SYSTEM, USER, or ASSISTANT).
     * @param content the textual content of the message
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
     * @param name the string representation of the role
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