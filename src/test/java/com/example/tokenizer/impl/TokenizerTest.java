package com.example.tokenizer.impl;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TokenizerInterfaceTest {

    @Test
    void testReplaceControlCharactersWithCodePoints() {
        int[] input = {'H', 'e', '\n', 0x07, 'l', 'o'}; // 0x07 = BEL (control character)
        String result = Tokenizer.replaceControlCharacters(input);

        assertEquals("He\n\\u0007lo", result); // \n allowed, BEL escaped
    }

    @Test
    void testReplaceControlCharactersWithString() {
        String input = "He\n\u0007lo"; // \u0007 is a bell character (non-printable control char)
        String result = Tokenizer.replaceControlCharacters(input);

        assertEquals("He\n\\u0007lo", result);
    }

    @Test
    void testReplaceControlCharactersWithOnlyPrintableChars() {
        String input = "Hello, World!";
        String result = Tokenizer.replaceControlCharacters(input);

        assertEquals(input, result);
    }

    @Test
    void testReplaceControlCharactersWithMultipleControlChars() {
        String input = "\u0001\u0002A\nB\u0003"; // \u0001, \u0002, \u0003 are control chars
        String result = Tokenizer.replaceControlCharacters(input);

        assertEquals("\\u0001\\u0002A\nB\\u0003", result);
    }

    @Test
    void testReplaceControlCharactersEmptyInput() {
        String input = "";
        String result = Tokenizer.replaceControlCharacters(input);

        assertEquals("", result);
    }

    @Test
    void testReplaceControlCharactersNullSafe() {
        // Add this test if you plan to make it null-safe.
        assertThrows(NullPointerException.class, () -> {
            Tokenizer.replaceControlCharacters((String) null);
        });
    }
}