package org.beehive.gpullama3.tools;

/**
 * The result of executing a single tool call.
 *
 * @param toolName    name of the tool that was invoked
 * @param resultText  the tool's output (may be JSON or plain text)
 * @param error       non-null when execution failed; contains the error message
 */
public record ToolResult(String toolName, String resultText, String error) {

    /** Returns true when the tool execution failed. */
    public boolean isError() {
        return error != null;
    }

    public static ToolResult success(String toolName, String resultText) {
        return new ToolResult(toolName, resultText, null);
    }

    public static ToolResult failure(String toolName, String errorMessage) {
        return new ToolResult(toolName, null, errorMessage);
    }
}
