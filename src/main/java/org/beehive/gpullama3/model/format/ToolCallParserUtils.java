package org.beehive.gpullama3.model.format;

import java.util.Optional;

/**
 * Pure-string tool-call extraction for Llama and Qwen3 response formats.
 *
 * All methods are stateless and do not require any model or tokenizer instance,
 * making them directly unit-testable.
 */
public final class ToolCallParserUtils {

    private ToolCallParserUtils() {}

    // ── Llama ─────────────────────────────────────────────────────────────────

    /**
     * Extracts a tool call from a LLaMA 3.1 or 3.2 model response.
     *
     * Recognised formats:
     *  1. {@code <|python_tag|>{"name":…,"parameters":{…}}} — LLaMA 3.1 native, also accepted by 3.2
     *  2. Raw JSON with {@code "arguments"} key instead of {@code "parameters"} — LLaMA 3.2 instruction format
     *  3. Raw JSON object optionally inside markdown code fences — fallback for models that
     *     follow system-prompt instructions but omit the special-token prefix
     *
     * Both {@code "parameters"} and {@code "arguments"} are tried so a single implementation
     * handles the 3.1 and 3.2 variants transparently.
     */
    public static Optional<ToolCallExtract> parseLlamaResponse(String responseText) {
        // 1. Native LLaMA 3.1 format: <|python_tag|>{...}
        int idx = responseText.indexOf("<|python_tag|>");
        if (idx != -1) {
            String json = responseText.substring(idx + "<|python_tag|>".length()).strip();
            return parseLlamaJson(json);
        }

        // 2. LLaMA 3.2 format: <tool_call>...</tool_call>
        int tcStart = responseText.indexOf("<tool_call>");
        int tcEnd   = responseText.lastIndexOf("</tool_call>");
        if (tcStart != -1 && tcEnd != -1 && tcEnd > tcStart) {
            String json = responseText.substring(tcStart + "<tool_call>".length(), tcEnd).strip();
            return parseLlamaJson(json);
        }

        // 3. Fallback: raw JSON, possibly inside markdown code fences
        String stripped = stripMarkdownFences(responseText.strip());
        if (stripped.startsWith("{")) {
            return parseLlamaJson(stripped);
        }

        return Optional.empty();
    }

    /**
     * Parses a LLaMA-style tool call JSON object.
     * Accepts {@code {"name":…,"parameters":{…}}}, {@code {"function":…,"parameters":{…}}},
     * and {@code {"name":…,"arguments":{…}}}.
     */
    private static Optional<ToolCallExtract> parseLlamaJson(String json) {
        String name = extractStringValue(json, "name");
        if (name == null) {
            name = extractStringValue(json, "function");
        }
        if (name == null) return Optional.empty();

        String argsJson = extractNestedObject(json, "parameters");
        if (argsJson == null) argsJson = extractNestedObject(json, "arguments");
        if (argsJson == null) argsJson = "{}";

        return Optional.of(new ToolCallExtract(name, argsJson));
    }

    // ── Qwen3 ─────────────────────────────────────────────────────────────────

    /**
     * Extracts a tool call enclosed in {@code <tool_call>…</tool_call>} tags
     * as produced by Qwen3 models.
     */
    public static Optional<ToolCallExtract> parseQwen3Response(String responseText) {
        int start = responseText.indexOf("<tool_call>");
        int end   = responseText.lastIndexOf("</tool_call>");
        if (start == -1 || end == -1 || end <= start) return Optional.empty();

        String json = responseText.substring(start + "<tool_call>".length(), end).strip();

        String name = extractStringValue(json, "name");
        if (name == null) return Optional.empty();

        String argsJson = extractNestedObject(json, "arguments");
        if (argsJson == null) argsJson = "{}";

        return Optional.of(new ToolCallExtract(name, argsJson));
    }

    // ── Shared helpers ────────────────────────────────────────────────────────

    /** Strips surrounding markdown code fences (```…```) if present. */
    public static String stripMarkdownFences(String text) {
        if (!text.startsWith("```")) return text;
        int firstNewline = text.indexOf('\n');
        if (firstNewline == -1) return text;
        String body = text.substring(firstNewline + 1);
        if (body.endsWith("```")) body = body.substring(0, body.length() - 3).stripTrailing();
        return body.strip();
    }

    /** Extracts the string value for {@code "key": "<value>"} from a JSON object. Tolerates whitespace around {@code :}. */
    public static String extractStringValue(String json, String key) {
        String marker = "\"" + key + "\"";
        int markerIdx = json.indexOf(marker);
        if (markerIdx == -1) return null;
        int colonIdx = json.indexOf(':', markerIdx + marker.length());
        if (colonIdx == -1) return null;
        int quoteStart = json.indexOf('"', colonIdx + 1);
        if (quoteStart == -1) return null;
        int quoteEnd = json.indexOf('"', quoteStart + 1);
        if (quoteEnd == -1) return null;
        return json.substring(quoteStart + 1, quoteEnd);
    }

    /**
     * Extracts the JSON object value for {@code "key": {…}} using brace-counting.
     * Handles nested objects and tolerates whitespace around {@code :}.
     */
    public static String extractNestedObject(String json, String key) {
        String marker = "\"" + key + "\"";
        int markerIdx = json.indexOf(marker);
        if (markerIdx == -1) return null;
        int colonIdx = json.indexOf(':', markerIdx + marker.length());
        if (colonIdx == -1) return null;
        int braceStart = json.indexOf('{', colonIdx + 1);
        if (braceStart == -1) return null;
        int depth = 0;
        for (int i = braceStart; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '{') depth++;
            else if (c == '}') {
                if (--depth == 0) return json.substring(braceStart, i + 1);
            }
        }
        return null; // unbalanced
    }
}
