package com.example.tokenizer.impl;

import com.example.tokenizer.vocabulary.Vocabulary;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

class MistralTokenizerTest {

    private Vocabulary vocabulary;
    private MistralTokenizer tokenizer;

    @BeforeEach
    void setup() {
        List<String> baseTokens = List.of("▁h", "e", "l", "o", "▁", "▁hello");
        List<String> byteFallbackTokens = new ArrayList<>();

        for (int i = 0; i < 256; i++) {
            byteFallbackTokens.add(String.format("<0x%02X>", i));
        }

        List<String> allTokens = new ArrayList<>();
        allTokens.addAll(baseTokens);
        allTokens.addAll(byteFallbackTokens);

        String[] tokens = allTokens.toArray(new String[0]);
        float[] scores = new float[tokens.length];
        Arrays.fill(scores, 0.0f); // dummy scores

        int[] tokenTypes = new int[tokens.length];
        Arrays.fill(tokenTypes, 1); // mark all normal
        tokenTypes[baseTokens.size()] = 0; // mark <0x00> as special

        Map<String, Object> metadata = new HashMap<>();
        metadata.put("tokenizer.ggml.token_type", tokenTypes);

        vocabulary = new Vocabulary(tokens, scores);
        tokenizer = new MistralTokenizer(metadata, vocabulary);
    }

    @Test
    void testEncodeSimpleText() {
        List<Integer> tokens = tokenizer.encodeAsList("hello");
        assertNotNull(tokens);
        assertFalse(tokens.isEmpty());
    }

    @Test
    void testRegexPatternReturnsNull() {
        assertNull(tokenizer.regexPattern());
    }

    @Test
    void testSpecialTokenDetection() {
        assertTrue(tokenizer.isSpecialToken(6));
        assertFalse(tokenizer.isSpecialToken(0));
    }

    @Test
    void testShouldDisplayToken() {
        assertTrue(tokenizer.shouldDisplayToken(0));
        assertFalse(tokenizer.shouldDisplayToken(6));
    }

    @Test
    void testDecodeSpecialByteFallbackToken() {
        List<Integer> tokens = List.of(6); // token <0x00>
        String result = tokenizer.decode(tokens);
        assertEquals("\u0000", result); // ASCII for <0x00>
    }

    @Test
    void testEncodeEmptyInput() {
        List<Integer> tokens = tokenizer.encodeAsList("");
        assertTrue(tokens.isEmpty(), "Should return empty token list for empty input");
    }
}