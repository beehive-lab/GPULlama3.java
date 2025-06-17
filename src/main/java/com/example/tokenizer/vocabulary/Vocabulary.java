package com.example.tokenizer.vocabulary;

import java.util.Arrays;
import java.util.Map;
import java.util.OptionalInt;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public record Vocabulary(String[] tokens, float[] scores, Map<String, Integer> tokenToIndex) {
    private static final String TOKENIZER_LLAMA_3_MODEL = "gpt2";

    // @formatter:off
    public Vocabulary(String[] vocabulary, float[] scores) {
        this(vocabulary, scores,
                IntStream.range(0, vocabulary.length)
                        .boxed()
                        .collect(Collectors.toMap(i -> vocabulary[i], i -> i))
        );
    }
    // @formatter:on

    public String get(int tokenIndex) {
        return tokens[tokenIndex];
    }

    public OptionalInt getIndex(String token) {
        Integer value = tokenToIndex.get(token);
        return value != null ? OptionalInt.of(value) : OptionalInt.empty();
    }

    public static Vocabulary loadLlamaVocabulary(Map<String, Object> metadata) {
        String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
        return new Vocabulary(tokens, null);
    }

    public static Vocabulary loadMistralVocabulary(Map<String, Object> metadata) {
        String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
        float[] scores = (float[]) metadata.get("tokenizer.ggml.scores");
        Vocabulary v = new Vocabulary(tokens, scores);
        return v;
    }

    public static Vocabulary loadQwen3Vocabulary(Map<String, Object> metadata) {
        String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
        float[] scores = (float[]) metadata.get("tokenizer.ggml.scores");
        return new Vocabulary(tokens, scores);
    }

    public int size() {
        return tokens.length;
    }

    /**
     * Only for Mistral.
     */
    public float getScore(int tokenIndex) {
        return scores[tokenIndex];
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Vocabulary:\n");
        sb.append("Tokens: ").append(Arrays.toString(tokens)).append("\n");
        sb.append("Scores: ").append(Arrays.toString(scores)).append("\n");
        sb.append("Token to Index Map:\n");
        tokenToIndex.forEach((token, index) -> sb.append("  ").append(token).append(" -> ").append(index).append("\n"));
        return sb.toString();
    }
}