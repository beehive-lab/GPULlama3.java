package com.example.model.format;

import com.example.tokenizer.impl.Qwen3Tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Utility tailored for the Chat Markup Language (ChatML) prompt format.
 */
public class Qwen3ChatFormat implements ChatFormat {

    protected Qwen3Tokenizer tokenizer;
    protected ChatTokens chatTokens;

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

        fimPrefix = specialTokens.getOrDefault("<|fim_prefix|>", -1);
        fimSuffix = specialTokens.getOrDefault("<|fim_suffix|>", -1);
        fimMiddle = specialTokens.getOrDefault("<|fim_middle|>", -1);
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
                Integer token = tokenizer.getSpecialTokens().get("<｜User｜>");
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
            tokens.add(imStart);
            tokens.addAll(this.tokenizer.encodeAsList(message.role().name()));
            tokens.addAll(this.tokenizer.encodeAsList("\n"));
        }
        return tokens;
    }

    @Override
    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = this.encodeHeader(message);
        tokens.addAll(this.tokenizer.encodeAsList(message.content().strip()));
        boolean isFim = Role.FIM_PREFIX.equals(message.role()) || Role.FIM_SUFFIX.equals(message.role()) || Role.FIM_MIDDLE.equals(message.role());
        if (imEnd != -1 && !isFim) {
            tokens.add(imEnd);
        }
        return tokens;
    }

    @Override
    public int getBeginOfText() {
        return beginOfText;
    }



    @Override
    public Set<Integer> getStopTokens() {
        if (imEnd == -1 && endOfText == -1) {
            throw new IllegalStateException("No stop token is defined.");
        }
        if (imEnd == -1) {

            return Set.of(endOfText);
        }
        return Set.of(imEnd, endOfText, endOfTextFim);
    }
}
