package com.example.model.format;

import com.example.tokenizer.impl.LlamaTokenizer;
import com.example.tokenizer.impl.MistralTokenizer;

import java.util.List;
import java.util.Set;

public interface ChatFormat {

    static ChatFormat create(Object tokenizer) {
        if (tokenizer instanceof LlamaTokenizer llamaTokenizer) {
            return new LlamaChatFormat(llamaTokenizer);
        } else if (tokenizer instanceof MistralTokenizer mistralTokenizer) {
            return new MistralChatFormat(mistralTokenizer);
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

        @Override
        public String toString() {
            return name;
        }
    }

}