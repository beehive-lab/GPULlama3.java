package org.beehive.gpullama3.model.format;

import org.beehive.gpullama3.tokenizer.Gemma4Tokenizer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Chat format for Gemma 4 models.
 * Turn structure: <|turn>role\n[content]<turn|>\n
 */
public class Gemma4ChatFormat implements ChatFormat {

    private final Gemma4Tokenizer tokenizer;
    private final int beginOfSentence; // <bos>
    private final int startOfTurn;     // <|turn>
    private final int endOfTurn;       // <turn|>
    private final int endOfSentence;   // <eos>

    public Gemma4ChatFormat(Gemma4Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();
        this.beginOfSentence = specialTokens.get("<bos>");
        this.startOfTurn = specialTokens.get("<|turn>");
        this.endOfTurn = specialTokens.get("<turn|>");
        this.endOfSentence = specialTokens.get("<eos>");
    }

    @Override
    public ChatTokens chatTokens() {
        return new ChatTokens("<|turn>", "<turn|>", "", "<eos>", "");
    }

    @Override
    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(startOfTurn);
        // Gemma uses "model" instead of "assistant"
        String roleName = message.role().name();
        if ("assistant".equals(roleName)) {
            roleName = "model";
        }
        tokens.addAll(tokenizer.encode(roleName));
        tokens.addAll(tokenizer.encode("\n"));
        return tokens;
    }

    @Override
    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = encodeHeader(message);
        tokens.addAll(tokenizer.encode(message.content().strip()));
        tokens.add(endOfTurn);
        tokens.addAll(tokenizer.encode("\n"));
        return tokens;
    }

    @Override
    public int getBeginOfText() {
        return beginOfSentence;
    }

    @Override
    public Set<Integer> getStopTokens() {
        Set<Integer> tokens = new HashSet<>();
        tokens.add(endOfSentence);
        tokens.add(endOfTurn);
        return tokens;
    }
}
