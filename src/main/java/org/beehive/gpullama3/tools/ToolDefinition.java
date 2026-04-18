package org.beehive.gpullama3.tools;

/**
 * Framework-agnostic description of a tool available to the model.
 *
 * @param name            unique tool name
 * @param description     human-readable description used in the model's system prompt
 * @param parametersJson  JSON Schema object for the tool's parameters, e.g.
 *                        {@code {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}
 */
public record ToolDefinition(String name, String description, String parametersJson) {

    /** Convenience factory for a tool with no parameters. */
    public static ToolDefinition noArgs(String name, String description) {
        return new ToolDefinition(name, description, "{\"type\":\"object\",\"properties\":{}}");
    }
}
