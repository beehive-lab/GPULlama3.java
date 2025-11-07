package org.beehive.gpullama3.model.format;

import org.beehive.gpullama3.tokenizer.impl.Gemma3Tokenizer;

import java.util.*;

/**
 * Chat format implementation for Gemma 3 models.
 *
 * Gemma 3 uses the following prompt template:
 * <bos><start_of_turn>user
 * {user_message}<end_of_turn>
 * <start_of_turn>model
 * {model_message}<end_of_turn>
 */
public class Gemma3ChatFormat implements ChatFormat {

    protected final int beginOfText;        // <bos>
    protected final int startOfTurn;        // <start_of_turn>
    protected final int endOfTurn;          // <end_of_turn>
    protected final int endOfText;          // <eos>
    protected final Set<Integer> stopTokens;
    protected Gemma3Tokenizer tokenizer;

    public Gemma3ChatFormat(Gemma3Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();

        this.beginOfText = specialTokens.getOrDefault("<bos>", -1);
        this.startOfTurn = specialTokens.getOrDefault("<start_of_turn>", -1);
        this.endOfTurn = specialTokens.getOrDefault("<end_of_turn>", -1);
        this.endOfText = specialTokens.getOrDefault("<eos>", -1);

        // Stop tokens for Gemma 3
        this.stopTokens = new HashSet<>();
        if (endOfTurn != -1) {
            stopTokens.add(endOfTurn);
        }
        if (endOfText != -1) {
            stopTokens.add(endOfText);
        }
    }

    @Override
    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();

        // Add <start_of_turn> token
        tokens.add(startOfTurn);

        // Add role name (user, model, system, etc.)
        // Convert role to appropriate text for Gemma (assistant -> model)
        String roleName = message.role().name();
        if (roleName.equals("assistant")) {
            roleName = "model";
        }

        tokens.addAll(this.tokenizer.encodeAsList(roleName));
        tokens.addAll(this.tokenizer.encodeAsList("\n"));

        return tokens;
    }

    @Override
    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = this.encodeHeader(message);

        // Add message content (trimmed)
        tokens.addAll(this.tokenizer.encodeAsList(message.content().strip()));

        // Add <end_of_turn> token followed by newline
        tokens.add(endOfTurn);
        tokens.addAll(this.tokenizer.encodeAsList("\n"));

        return tokens;
    }

    @Override
    public int getBeginOfText() {
        return beginOfText;
    }

    @Override
    public Set<Integer> getStopTokens() {
        if (stopTokens.isEmpty()) {
            throw new IllegalStateException("No stop tokens defined for Gemma3.");
        }
        return stopTokens;
    }
}
