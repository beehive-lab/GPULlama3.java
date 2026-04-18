package org.beehive.gpullama3.tools;

import org.beehive.gpullama3.model.format.ToolCallExtract;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Holds available tool definitions and their executors.
 * Registration order is preserved for deterministic JSON output.
 */
public class ToolRegistry {

    private final Map<String, Entry> entries = new LinkedHashMap<>();

    private record Entry(ToolDefinition definition, ToolExecutor executor) {}

    public ToolRegistry register(ToolDefinition definition, ToolExecutor executor) {
        entries.put(definition.name(), new Entry(definition, executor));
        return this;
    }

    public Optional<ToolDefinition> getDefinition(String name) {
        Entry e = entries.get(name);
        return e == null ? Optional.empty() : Optional.of(e.definition());
    }

    public Optional<ToolExecutor> getExecutor(String name) {
        Entry e = entries.get(name);
        return e == null ? Optional.empty() : Optional.of(e.executor());
    }

    public List<ToolDefinition> definitions() {
        return entries.values().stream().map(Entry::definition).toList();
    }

    public boolean isEmpty() {
        return entries.isEmpty();
    }

    /**
     * Executes the named tool, returning a failure result for unknown tools or
     * executor exceptions. Never throws.
     */
    public ToolResult execute(ToolCallExtract call) {
        Optional<ToolExecutor> executor = getExecutor(call.name());
        if (executor.isEmpty()) {
            return ToolResult.failure(call.name(), "Unknown tool: " + call.name());
        }
        try {
            return executor.get().execute(call);
        } catch (Exception e) {
            return ToolResult.failure(call.name(), "Tool execution failed: " + e.getMessage());
        }
    }

    /**
     * Serialises all registered tools to the flat JSON array expected by
     * {@code LlamaChatFormat.toolSystemPromptSuffix()} and
     * {@code Qwen3ChatFormat.toolSystemPromptSuffix()}.
     *
     * Format: {@code [{"name":…,"description":…,"parameters":{…}}]}
     */
    public String toToolsJson() {
        List<ToolDefinition> defs = definitions();
        if (defs.isEmpty()) return "[]";

        StringBuilder sb = new StringBuilder("[\n");
        for (int i = 0; i < defs.size(); i++) {
            ToolDefinition d = defs.get(i);
            sb.append("  {\n");
            sb.append("    \"name\": \"").append(escapeJson(d.name())).append("\",\n");
            sb.append("    \"description\": \"").append(escapeJson(d.description())).append("\",\n");
            sb.append("    \"parameters\": ").append(d.parametersJson()).append("\n");
            sb.append("  }");
            if (i < defs.size() - 1) sb.append(",");
            sb.append("\n");
        }
        sb.append("]");
        return sb.toString();
    }

    private static String escapeJson(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }
}
