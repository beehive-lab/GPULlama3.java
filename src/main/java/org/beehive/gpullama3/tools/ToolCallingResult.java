package org.beehive.gpullama3.tools;

import org.beehive.gpullama3.model.format.ToolCallExtract;

import java.util.List;

/**
 * The outcome of a complete tool-calling session (prompt → [tool round-trips] → answer).
 *
 * @param finalAnswer         the model's final plain-text answer
 * @param callsMade           tool calls that were extracted and executed (may be empty)
 * @param results             corresponding tool results (same order as callsMade)
 * @param reachedMaxRoundTrips true when the session stopped because maxRoundTrips was hit
 */
public record ToolCallingResult(
        String finalAnswer,
        List<ToolCallExtract> callsMade,
        List<ToolResult> results,
        boolean reachedMaxRoundTrips) {

    public boolean hadToolCalls() {
        return !callsMade.isEmpty();
    }

    /** Returns a result representing a plain-text (no-tool) response. */
    public static ToolCallingResult plainText(String answer) {
        return new ToolCallingResult(answer, List.of(), List.of(), false);
    }
}
