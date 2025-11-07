package org.beehive.gpullama3.tokenizer.impl;

import org.beehive.gpullama3.core.types.Pair;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;

import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * BPE tokenizer for Gemma 3 models.
 * Similar to Llama tokenizer but adapted for Gemma's vocabulary and special tokens.
 */
public class Gemma3Tokenizer implements Tokenizer {
    // Gemma uses a similar pattern to Llama 3
    private static final String GEMMA3_PATTERN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    private final Pattern compiledPattern;
    private final Vocabulary vocabulary;
    private final Map<Pair<Integer, Integer>, Integer> merges;
    private final Map<String, Integer> specialTokens;
    private final int[] tokenTypes;

    public String regexPattern() {
        if (compiledPattern == null) {
            return null;
        }
        return compiledPattern.pattern();
    }

    public String getTokenString(int tokenId) {
        return vocabulary.get(tokenId);
    }

    @Override
    public Map<String, Integer> getSpecialTokens() {
        return specialTokens;
    }

    @Override
    public boolean isSpecialToken(int tokenIndex) {
        return specialTokens.containsValue(tokenIndex);
    }

    @Override
    public boolean shouldDisplayToken(int token) {
        return !isSpecialToken(token);
    }

    public Gemma3Tokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        // Load token types
        int[] tokenTypes = (int[]) metadata.get("tokenizer.ggml.token_type");
        this.tokenTypes = tokenTypes;

        // Load merges from metadata
        String[] mergeLines = (String[]) metadata.get("tokenizer.ggml.merges");
        List<Pair<Integer, Integer>> merges;

        if (mergeLines == null || mergeLines.length == 0) {
            // If no merges, use empty list (some tokenizers might not use BPE merges)
            merges = List.of();
        } else {
            merges = Arrays.stream(mergeLines)
                    .map(line -> line.split(" "))
                    .map(parts -> new Pair<>(
                            vocabulary.getIndex(parts[0]).orElseThrow(),
                            vocabulary.getIndex(parts[1]).orElseThrow()))
                    .toList();
        }

        // Detect special tokens
        // For Gemma, special tokens typically include: <bos>, <eos>, <start_of_turn>, <end_of_turn>
        // NOTE: Byte tokens (type 3) with placeholder names like <unused78> are NOT special tokens!
        Map<String, Integer> specialTokens = new HashMap<>();

        // Scan vocabulary for special tokens (those starting with '<' and ending with '>')
        // but exclude byte placeholder tokens and hex byte tokens (pattern: <0xHH>)
        for (int i = 0; i < vocabulary.size(); i++) {
            String token = vocabulary.get(i);
            if (token.startsWith("<") && token.endsWith(">")) {
                // Check if it's a hex byte token (format: <0xHH>)
                if (token.matches("<0x[0-9a-fA-F]{2}>")) {
                    continue; // Skip hex byte tokens
                }
                // Check if it's a byte placeholder token (format: <unusedNN>)
                if (token.matches("<unused\\d+>")) {
                    continue; // Skip byte placeholder tokens
                }
                // This is a real special token like <bos>, <eos>, <start_of_turn>, etc.
                specialTokens.put(token, i);
            }
        }

        // Initialize fields
        this.vocabulary = vocabulary;
        this.compiledPattern = Pattern.compile(GEMMA3_PATTERN);
        this.specialTokens = specialTokens;
        this.merges = new HashMap<>();

        for (Pair<Integer, Integer> pair : merges) {
            int firstIndex = pair.first();
            int secondIndex = pair.second();
            this.merges.put(new Pair<>(firstIndex, secondIndex), vocabulary.size() + this.merges.size());
        }
    }

    @Override
    public List<Integer> encode(String text, Set<String> allowedSpecial) {
        // Handle special tokens
        if (allowedSpecial.isEmpty()) {
            return encodeOrdinary(text);
        }

        // Split text by special tokens
        String specialPattern = allowedSpecial
                .stream()
                .map(java.util.regex.Pattern::quote)
                .collect(java.util.stream.Collectors.joining("|", "(", ")"));

        String[] specialChunks = text.split(specialPattern);
        List<Integer> ids = new ArrayList<>();

        for (String part : specialChunks) {
            if (allowedSpecial.contains(part)) {
                ids.add(specialTokens.get(part));
            } else {
                ids.addAll(encodeOrdinary(part));
            }
        }
        return ids;
    }

    @Override
    public List<Integer> encodeAsList(String text) {
        return encode(text, Set.of());
    }

    private List<Integer> encodeOrdinary(String text) {
        if (text == null || text.isEmpty()) {
            return List.of();
        }

        List<Integer> tokens = new ArrayList<>();
        Matcher matcher = compiledPattern.matcher(text);

        while (matcher.find()) {
            String piece = matcher.group();
            byte[] bytes = piece.getBytes(StandardCharsets.UTF_8);

            // Convert bytes to token indices
            List<Integer> pieceTokens = new ArrayList<>();
            for (byte b : bytes) {
                String byteStr = vocabulary.get(Byte.toUnsignedInt(b));
                pieceTokens.add(vocabulary.getIndex(byteStr).orElseThrow());
            }

            // Apply BPE merges
            while (pieceTokens.size() >= 2) {
                Pair<Integer, Integer> bestPair = null;
                int bestIndex = -1;
                int bestMergeIndex = Integer.MAX_VALUE;

                for (int i = 0; i < pieceTokens.size() - 1; i++) {
                    Pair<Integer, Integer> pair = new Pair<>(pieceTokens.get(i), pieceTokens.get(i + 1));
                    Integer mergeIndex = merges.get(pair);
                    if (mergeIndex != null && mergeIndex < bestMergeIndex) {
                        bestPair = pair;
                        bestIndex = i;
                        bestMergeIndex = mergeIndex;
                    }
                }

                if (bestPair == null) {
                    break;
                }

                // Merge the pair
                pieceTokens.set(bestIndex, merges.get(bestPair));
                pieceTokens.remove(bestIndex + 1);
            }

            tokens.addAll(pieceTokens);
        }

        return tokens;
    }

    @Override
    public String decode(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            // Check if it's a byte token (type 3: token IDs 0-255)
            if (tokenTypes != null && tokenTypes[token] == 3) {
                // Decode byte token: use token ID as byte value
                sb.append((char) token);
                continue;
            }

            String tokenString = vocabulary.get(token);

            // Check if it's a hex byte token (format: <0xHH>)
            if (tokenString.matches("<0x[0-9a-fA-F]{2}>")) {
                // Decode hex byte token: extract hex value and convert to character
                String code = tokenString.substring(3, tokenString.length() - 1);
                int byteValue = Integer.parseInt(code, 16);
                tokenString = Character.toString(byteValue);
            } else if (isSpecialToken(token)) {
                // Skip actual special tokens (like <bos>, <eos>, <start_of_turn>, etc.)
                continue;
            } else {
                // SentencePiece uses ▁ (U+2581) to represent spaces
                tokenString = tokenString.replace('▁', ' ');
            }
            sb.append(tokenString);
        }
        return sb.toString();
    }
}
