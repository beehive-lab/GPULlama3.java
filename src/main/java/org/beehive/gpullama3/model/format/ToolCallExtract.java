package org.beehive.gpullama3.model.format;

/**
 * Represents a single tool call extracted from a model response.
 * Contains the raw strings — JSON parsing of arguments is left to the caller.
 *
 * @param name          the tool/function name to invoke
 * @param argumentsJson the arguments as a JSON object string, e.g. {"location":"Boston"}
 */
public record ToolCallExtract(String name, String argumentsJson) {
}
