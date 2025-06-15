package com.example.tokenizer.impl;

import com.example.core.types.Pair;
import com.example.tokenizer.vocabulary.Vocabulary;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

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
        assertTrue(result.contains(5)); // He
        assertTrue(result.contains(6)); // lo
    }

    @Test
    void testEncodeWithSpecialToken() {
        String input = "Hello<EOS>";
        List<Integer> result = tokenizer.encode(input, Set.of("<EOS>"));
        assertTrue(result.contains(101)); // <EOS>
    }

    @Test
    void testRegexPattern() {
        assertEquals("[A-Za-z ]+", tokenizer.regexPattern());
    }

    @Test
    void testDecode() {
        String input = "He lo";
        List<Integer> ids = tokenizer.encodeOrdinary(input);
        String decoded = tokenizer.decode(ids);
        assertEquals(input, decoded);
    }

    @Test
    void testSpecialTokenCheck() {
        assertTrue(tokenizer.isSpecialToken(100));
        assertFalse(tokenizer.isSpecialToken(999));
    }
}