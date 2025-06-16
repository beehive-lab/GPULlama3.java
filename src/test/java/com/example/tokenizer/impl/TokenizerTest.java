package com.example.tokenizer.impl;

import com.example.core.types.Pair;
import com.example.tokenizer.vocabulary.Vocabulary;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class TokenizerTest {

    private Tokenizer tokenizer;

    @BeforeEach
    void setup() {
        String[] tokens = {"H", "e", "l", "o", " ", "He", "lo"};
        float[] scores = new float[tokens.length];

        // Create token to index mapping
        Vocabulary vocab = new Vocabulary(tokens, scores);

        List<Pair<Integer, Integer>> merges = List.of(
                new Pair<>(0, 1), // H + e → He
                new Pair<>(2, 3)  // l + o → lo
        );

        String regex = "[A-Za-z ]+";

        Map<String, Integer> specialTokens = Map.of("<PAD>", 100, "<EOS>", 101);

        tokenizer = new Tokenizer(vocab, merges, regex, specialTokens);
    }

    @Test
    void testEncodeOrdinary() {
        List<Integer> result = tokenizer.encodeOrdinary("Hello");
        assertNotNull(result);
        assertIterableEquals(List.of(5, 2, 6), result);
    }

    @Test
    void testEncodeWithManualSplit() {
        List<Integer> result = new ArrayList<>();
        result.addAll(tokenizer.encodeOrdinary("Hello"));
        result.add(tokenizer.getSpecialTokens().get("<EOS>"));
        assertTrue(result.contains(101));
    }

    @Test
    void testRegexPattern() {
        assertEquals("[A-Za-z ]+", tokenizer.regexPattern());
    }

    @Test
    void testDecode() {
        String input = "He lo";
        List<Integer> ids = tokenizer.encodeOrdinary(input);
        String decoded = tokenizer.decodeImpl(ids);
        assertEquals("He lo", decoded);
    }

    @Test
    void testSpecialTokenCheck() {
        assertTrue(tokenizer.isSpecialToken(100));
        assertFalse(tokenizer.isSpecialToken(999));
    }

    //Edge cases
    @Test
    void testEncodeOrdinaryWithEmptyString() {
        List<Integer> result = tokenizer.encodeOrdinary("");
        assertNotNull(result, "Result should not be null for empty input");
        assertTrue(result.isEmpty(), "Result should be empty for empty input");
    }

    @Test
    void testEncodeOrdinaryWithNull() {
        assertThrows(NullPointerException.class, () -> tokenizer.encodeOrdinary(null));
    }
}