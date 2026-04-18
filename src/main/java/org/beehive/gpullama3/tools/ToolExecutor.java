package org.beehive.gpullama3.tools;

import org.beehive.gpullama3.model.format.ToolCallExtract;

/**
 * Executes a single tool call and returns the result.
 * Implementations are responsible for parsing {@link ToolCallExtract#argumentsJson()}
 * and performing the actual action.
 */
@FunctionalInterface
public interface ToolExecutor {
    ToolResult execute(ToolCallExtract call);
}
