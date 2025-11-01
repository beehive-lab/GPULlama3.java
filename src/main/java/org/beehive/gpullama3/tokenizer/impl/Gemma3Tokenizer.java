package org.beehive.gpullama3.tokenizer.impl;

import org.beehive.gpullama3.auxiliary.Utf8Mask;
import org.beehive.gpullama3.core.types.Pair;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Tokenizer for Google Gemma 3 models.
 *
 * <p>Gemma 3 uses SentencePiece tokenization with:</p>
 * <ul>
 *   <li>Byte-level encoding for first 256 tokens (type 3)</li>
 *   <li>Space represented as ▁ (U+2581)</li>
 *   <li>Byte fallback encoding with offset 217</li>
 *   <li>Special tokens like <bos>, <eos>, <start_of_turn>, <end_of_turn></li>
 * </ul>
 */
public class Gemma3Tokenizer implements Tokenizer {
    private static final String GEMMA3_PATTERN = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    private static final String SPIECE_UNDERLINE = "▁";
    private static final int BYTE_FALLBACK_OFFSET = 217;

    private final Pattern compiledPattern;
    private final Vocabulary vocabulary;
    private final Map<Pair<Integer, Integer>, Integer> merges;
    private final Map<String, Integer> specialTokens;
    private final int[] tokenTypes;

    /** buffer to store incomplete UTF-8 sequence */
    private final byte[] bufUtf8 = new byte[4];
    /** index in UTF-8 buffer */
    private int currUtf8Index = 0;
    /** current UTF-8 mask */
    private Utf8Mask currUtf8Mask;

    @Override
    public String regexPattern() {
        if (compiledPattern == null) {
            return null;
        }
        return compiledPattern.pattern();
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
        // Display regular tokens (type 1) and special reasoning tokens if present
        int tokenType = getTokenType(token);
        return tokenType == 1 || tokenType == 6;
    }

    public int getTokenType(int tokenIndex) {
        if (tokenTypes == null || tokenIndex >= tokenTypes.length) {
            return 1; // Default to normal token
        }
        return tokenTypes[tokenIndex];
    }

    // @formatter:off
    public Gemma3Tokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        this.vocabulary = vocabulary;
        this.compiledPattern = Pattern.compile(GEMMA3_PATTERN);

        // Load token types if available
        this.tokenTypes = (int[]) metadata.get("tokenizer.ggml.token_type");

        // Load merges if available
        String[] mergeLines = (String[]) metadata.get("tokenizer.ggml.merges");
        this.merges = new HashMap<>();
        if (mergeLines != null) {
            List<Pair<Integer, Integer>> mergesList = Arrays.stream(mergeLines)
                    .map(line -> line.split(" "))
                    .map(parts ->
                            new Pair<>(
                                    vocabulary.getIndex(parts[0]).orElseThrow(),
                                    vocabulary.getIndex(parts[1]).orElseThrow())
                    ).toList();

            for (Pair<Integer, Integer> pair : mergesList) {
                int firstIndex = pair.first();
                int secondIndex = pair.second();
                int mergeIndex = vocabulary.getIndex(vocabulary.get(firstIndex) + vocabulary.get(secondIndex)).orElseThrow();
                this.merges.put(pair, mergeIndex);
            }
        }

        // Identify special tokens
        // Gemma special tokens typically include: <bos>, <eos>, <start_of_turn>, <end_of_turn>, <pad>
        this.specialTokens = new HashMap<>();
        for (int i = 0; i < vocabulary.size(); i++) {
            String token = vocabulary.get(i);
            if (isSpecialTokenPattern(token)) {
                specialTokens.put(token, i);
            }
        }
    }
    // @formatter:on

    private boolean isSpecialTokenPattern(String token) {
        // Special tokens start and end with angle brackets
        // But exclude <unusedNN> and <0xHH> patterns which are byte tokens
        if (token.startsWith("<") && token.endsWith(">")) {
            // Exclude byte tokens
            if (token.matches("<0x[0-9a-fA-F]{2}>")) {
                return false;
            }
            if (token.matches("<unused\\d+>")) {
                return false;
            }
            return true;
        }
        return false;
    }

    private int[] encodeImpl(String text) {
        return encode(text, Set.of()).stream().mapToInt(i -> i).toArray();
    }

    static List<String> findAll(Pattern pattern, String text) {
        List<String> allMatches = new ArrayList<>();
        Matcher matcher = pattern.matcher(text);
        while (matcher.find()) {
            allMatches.add(matcher.group());
        }
        return allMatches;
    }

    /**
     * Encoding that ignores any special tokens.
     */
    public List<Integer> encodeOrdinary(String text) {
        // split text into chunks of text by categories defined in regex pattern
        List<String> textChunks = findAll(compiledPattern, text);
        // all chunks of text are encoded separately, then results are joined
        List<Integer> ids = new ArrayList<>();
        for (String chunk : textChunks) {
            List<Integer> chunkIds = encodeChunk(chunk);
            ids.addAll(chunkIds);
        }
        return ids;
    }

    private Map<Pair<Integer, Integer>, Integer> getStats(List<Integer> ids) {
        Map<Pair<Integer, Integer>, Integer> map = new HashMap<>();
        for (int i = 0; i + 1 < ids.size(); i++) {
            Pair<Integer, Integer> key = new Pair<>(ids.get(i), ids.get(i + 1));
            map.put(key, map.getOrDefault(key, 0) + 1);
        }
        return map;
    }

    private List<Integer> encodeChunk(String chunk) {
        // Convert chunk to token IDs using vocabulary
        List<Integer> ids = new ArrayList<>();
        for (int b : chunk.toCharArray()) {
            int tokenIndex = this.vocabulary.getIndex(String.valueOf((char) b)).orElseThrow();
            ids.add(tokenIndex);
        }

        // Apply BPE merges if available
        if (!merges.isEmpty()) {
            while (ids.size() >= 2) {
                Map<Pair<Integer, Integer>, Integer> stats = getStats(ids);
                Pair<Integer, Integer> pair = stats.keySet().stream()
                        .min(Comparator.comparingInt(key -> this.merges.getOrDefault(key, Integer.MAX_VALUE)))
                        .orElseThrow();

                if (!this.merges.containsKey(pair)) {
                    break; // nothing else can be merged anymore
                }

                int idx = this.merges.get(pair);
                ids = merge(ids, pair, idx);
            }
        }
        return ids;
    }

    static List<Integer> merge(List<Integer> ids, Pair<Integer, Integer> pair, int idx) {
        List<Integer> newids = new ArrayList<>();
        int i = 0;
        while (i < ids.size()) {
            if (ids.get(i).equals(pair.first()) && i < ids.size() - 1 && ids.get(i + 1).equals(pair.second())) {
                newids.add(idx);
                i += 2;
            } else {
                newids.add(ids.get(i));
                i += 1;
            }
        }
        return newids;
    }

    // @formatter:off
    static Map<Integer, Integer> bytesToUnicode() {
        List<Integer> bs = new ArrayList<>();
        IntStream.rangeClosed('!', '~').forEach(bs::add);
        IntStream.rangeClosed('¡', '¬').forEach(bs::add);
        IntStream.rangeClosed('®', 'ÿ').forEach(bs::add);

        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n += 1;
            }
        }

        return IntStream.range(0, bs.size())
                .boxed()
                .collect(Collectors.toMap(bs::get, cs::get));
    }
    // @formatter:on

    static final Map<Integer, Integer> BYTE_ENCODER = bytesToUnicode();
    static final Map<Integer, Integer> BYTE_DECODER = BYTE_ENCODER.entrySet().stream()
            .collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey));

    public int[] encode(String text) {
        StringBuilder sb = new StringBuilder();
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        for (byte b : bytes) {
            sb.appendCodePoint(BYTE_ENCODER.get(Byte.toUnsignedInt(b)));
        }
        return encodeImpl(sb.toString());
    }

    @Override
    public List<Integer> encode(String text, Set<String> allowedSpecial) {
        if (allowedSpecial.isEmpty()) {
            return encodeOrdinary(text);
        }

        String specialPattern = allowedSpecial
                .stream()
                .map(Pattern::quote)
                .collect(Collectors.joining("|", "(", ")"));

        String[] specialChunks = text.split(specialPattern);
        List<Integer> ids = new ArrayList<>();
        for (String part : specialChunks) {
            if (allowedSpecial.contains(part)) {
                ids.add(getSpecialTokens().get(part));
            } else {
                ids.addAll(encodeOrdinary(part));
            }
        }
        return ids;
    }

    public List<Integer> encodeOrdinaryAsList(String text) {
        StringBuilder sb = new StringBuilder();
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        for (byte b : bytes) {
            sb.appendCodePoint(BYTE_ENCODER.get(Byte.toUnsignedInt(b)));
        }
        return encodeOrdinary(sb.toString());
    }

    @Override
    public List<Integer> encodeAsList(String text) {
        return Arrays.stream(encode(text)).boxed().toList();
    }

    public String decodeImpl(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            String tokenString = vocabulary.get(token);
            sb.append(tokenString);
        }
        return sb.toString();
    }

    @Override
    public String decode(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();

        for (int token : tokens) {
            // Type 3: Byte tokens (IDs 0-255 or with fallback offset) - decode as raw bytes
            if (tokenTypes != null && token < tokenTypes.length && tokenTypes[token] == 3) {
                // Handle byte fallback encoding
                if (token >= BYTE_FALLBACK_OFFSET && token < 256 + BYTE_FALLBACK_OFFSET) {
                    sb.append((char) (token - BYTE_FALLBACK_OFFSET));
                } else if (token < 256) {
                    sb.append((char) token);
                }
                continue;
            }

            String tokenString = vocabulary.get(token);

            // Handle hex byte tokens like <0x12>
            if (tokenString.matches("<0x[0-9a-fA-F]{2}>")) {
                String code = tokenString.substring(3, tokenString.length() - 1);
                int byteValue = Integer.parseInt(code, 16);
                tokenString = Character.toString(byteValue);
            } else if (isSpecialToken(token)) {
                // Skip special tokens in output
                continue;
            } else {
                // SentencePiece: replace ▁ with space
                tokenString = tokenString.replace(SPIECE_UNDERLINE, " ");
            }

            sb.append(tokenString);
        }

        // Handle any remaining UTF-8 decoding
        String decoded = sb.toString();
        int[] decodedBytesAsInts = decoded.codePoints()
                .map(cp -> cp <= 512 ? BYTE_DECODER.getOrDefault(cp, cp) : cp)
                .toArray();

        byte[] rawBytes = new byte[decodedBytesAsInts.length + 3];
        int indexRawByte = 0;

        loopDecoded:
        for (int i = 0; i < decoded.length(); i++) {
            byte b = (byte) decodedBytesAsInts[i];
            if (currUtf8Index == 0) {
                for (Utf8Mask utf8Mask : Utf8Mask.MASKS) {
                    if ((b & utf8Mask.mask()) == utf8Mask.pattern()) {
                        currUtf8Mask = utf8Mask;
                        bufUtf8[currUtf8Index++] = b;
                        continue loopDecoded;
                    }
                }
            }
            if (currUtf8Index > 0 && currUtf8Mask != null) {
                bufUtf8[currUtf8Index++] = b;
                if (currUtf8Index == currUtf8Mask.len()) {
                    System.arraycopy(bufUtf8, 0, rawBytes, indexRawByte, currUtf8Mask.len());
                    indexRawByte += currUtf8Mask.len();
                    currUtf8Index = 0;
                    currUtf8Mask = null;
                }
                continue;
            }
            rawBytes[indexRawByte++] = b;
        }

        return new String(rawBytes, 0, indexRawByte, StandardCharsets.UTF_8);
    }
}
