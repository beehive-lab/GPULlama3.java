package org.beehive.gpullama3.tokenizer;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Gemma 4 tokenizer: SentencePiece-style BPE with Unicode block character space replacement.
 * Space is replaced with '\u2581' before encoding.
 * Byte fallback uses <0xHH> tokens.
 */
public class Gemma4Tokenizer implements Tokenizer {

    private final Vocabulary vocabulary;
    private final Map<String, Integer> specialTokens;
    private final int[] tokenType;
    private final int byte0; // index of <0x00> token

    public Gemma4Tokenizer(Vocabulary vocabulary, int[] tokenType) {
        this.vocabulary = vocabulary;
        this.tokenType = tokenType.clone();
        // Mark tokens up to and including <turn|> as type 6 (displayable control)
        int endOfTurn = vocabulary.getIndex("<turn|>").orElseThrow();
        for (int i = 0; i <= endOfTurn; ++i) {
            if (this.tokenType[i] == 1) {
                this.tokenType[i] = 6;
            }
        }
        this.byte0 = vocabulary.getIndex("<0x00>").orElseThrow();
        this.specialTokens = buildSpecialTokens(this.tokenType)
                .stream()
                .collect(Collectors.toMap(t -> vocabulary.get(t), t -> t));
    }

    private List<Integer> buildSpecialTokens(int[] tokenType) {
        return IntStream.range(0, tokenType.length)
                .filter(t -> tokenType[t] != 1)
                .boxed()
                .toList();
    }

    @Override
    public Map<String, Integer> getSpecialTokens() {
        return specialTokens;
    }

    @Override
    public boolean isSpecialToken(int tokenIndex) {
        return tokenType[tokenIndex] != 1;
    }

    @Override
    public boolean shouldDisplayToken(int token) {
        int tt = tokenType[token];
        return tt == 1 || tt == 6;
    }

    @Override
    public String regexPattern() {
        return null;
    }

    @Override
    public List<Integer> encode(String text, Set<String> allowedSpecial) {
        // Simple implementation: replace spaces and encode
        return encodeImpl(text.replace(' ', '\u2581'));
    }

    public List<Integer> encode(String text) {
        return encodeImpl(text.replace(' ', '\u2581'));
    }

    private List<Integer> encodeImpl(String text) {
        List<Integer> tokens = new ArrayList<>();

        for (int i = 0, cpi; i < text.length(); i += Character.charCount(cpi)) {
            cpi = text.codePointAt(i);
            String singleCodepoint = Character.toString(cpi);
            int id = vocabulary.getIndex(singleCodepoint).orElse(-1);
            if (id != -1) {
                tokens.add(id);
            } else {
                // Byte fallback
                for (byte b : singleCodepoint.getBytes(StandardCharsets.UTF_8)) {
                    tokens.add(Byte.toUnsignedInt(b) + byte0);
                }
            }
        }

        // Greedy BPE merge
        while (true) {
            float bestScore = -1e10f;
            int bestId = -1;
            int bestIdx = -1;

            for (int i = 0; i < tokens.size() - 1; ++i) {
                String merged = vocabulary.get(tokens.get(i)) + vocabulary.get(tokens.get(i + 1));
                int id = vocabulary.getIndex(merged).orElse(-1);
                if (id != -1 && vocabulary.getScore(id) > bestScore) {
                    bestScore = vocabulary.getScore(id);
                    bestId = id;
                    bestIdx = i;
                }
            }

            if (bestIdx == -1) {
                break;
            }

            tokens.set(bestIdx, bestId);
            tokens.remove(bestIdx + 1);
        }

        return tokens;
    }

    @Override
    public List<Integer> encodeAsList(String text) {
        return encode(text);
    }

    @Override
    public String decode(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            String tokenString = vocabulary.get(token);
            if (isSpecialToken(token)) {
                // Handle byte tokens like <0xHH>
                String prefix = "<0x";
                String suffix = ">";
                if (tokenString.length() == 6 && tokenString.startsWith(prefix) && tokenString.endsWith(suffix)) {
                    String code = tokenString.substring(prefix.length(), tokenString.length() - suffix.length());
                    int cp = Integer.parseInt(code, 16);
                    tokenString = Character.toString(cp);
                }
            } else {
                tokenString = tokenString.replace('\u2581', ' ');
            }
            sb.append(tokenString);
        }
        return sb.toString();
    }
}
