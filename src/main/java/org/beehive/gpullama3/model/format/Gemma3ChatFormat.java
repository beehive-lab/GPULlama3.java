package org.beehive.gpullama3.model.format;

import org.beehive.gpullama3.tokenizer.impl.Gemma3Tokenizer;

import java.util.*;

/**
 * Chat format for Google Gemma 3 models.
 *
 * <p>Gemma 3 chat template:</p>
 * <pre>
 * &lt;bos&gt;&lt;start_of_turn&gt;user
 * {user_message}&lt;end_of_turn&gt;
 * &lt;start_of_turn&gt;model
 * {model_message}&lt;end_of_turn&gt;
 * </pre>
 *
 * <p>Stop tokens: &lt;end_of_turn&gt;, &lt;eos&gt;</p>
 */
public class Gemma3ChatFormat implements ChatFormat {

    protected final int beginOfText;
    protected final int endOfText;
    protected final int startOfTurn;
    protected final int endOfTurn;
    protected Gemma3Tokenizer tokenizer;

    public Gemma3ChatFormat(Gemma3Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();

        // Load special tokens
        this.beginOfText = specialTokens.getOrDefault("<bos>", -1);
        this.endOfText = specialTokens.getOrDefault("<eos>", -1);
        this.startOfTurn = specialTokens.getOrDefault("<start_of_turn>", -1);
        this.endOfTurn = specialTokens.getOrDefault("<end_of_turn>", -1);
    }

    @Override
    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();

        // Add <start_of_turn> token
        if (startOfTurn != -1) {
            tokens.add(startOfTurn);
        }

        // Encode the role name (user, model, system, etc.)
        tokens.addAll(this.tokenizer.encodeOrdinaryAsList(message.role().name()));

        // Add newline after role
        tokens.addAll(this.tokenizer.encodeOrdinaryAsList("\n"));

        return tokens;
    }

    @Override
    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = this.encodeHeader(message);

        // Encode message content as ordinary text
        tokens.addAll(this.tokenizer.encodeOrdinaryAsList(message.content().strip()));

        // Add <end_of_turn> token
        if (endOfTurn != -1) {
            tokens.add(endOfTurn);
        }

        // Add newline after end_of_turn
        tokens.addAll(this.tokenizer.encodeOrdinaryAsList("\n"));

        return tokens;
    }

    @Override
    public int getBeginOfText() {
        return beginOfText;
    }

    @Override
    public Set<Integer> getStopTokens() {
        Set<Integer> stopTokens = new HashSet<>();

        // Add end_of_turn as primary stop token
        if (endOfTurn != -1) {
            stopTokens.add(endOfTurn);
        }

        // Add eos as secondary stop token
        if (endOfText != -1) {
            stopTokens.add(endOfText);
        }

        if (stopTokens.isEmpty()) {
            throw new IllegalStateException("No stop tokens defined for Gemma3");
        }

        return stopTokens;
    }
}
