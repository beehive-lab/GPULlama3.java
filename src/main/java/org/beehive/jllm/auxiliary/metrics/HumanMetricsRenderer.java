package org.beehive.gpullama3.auxiliary.metrics;

/**
 * Renders metrics in human-readable format to {@code stderr}.
 *
 * <p>This is the default renderer — no configuration needed. To enable explicitly:</p>
 * <pre>
 *   -Dllama.metrics.format=human   (default, can be omitted)
 *   -Dllama.metrics.output=stderr  (default, can be omitted)
 * </pre>
 *
 * <p>To also print TornadoVM initialisation timings (plan creation, JIT, weight copy-in),
 * additionally set:</p>
 * <pre>
 *   -Dllama.EnableTimingForTornadoVMInit=true
 * </pre>
 */
public final class HumanMetricsRenderer implements MetricsRenderer {

    @Override
    public String render(RunMetricsSnapshot s) {
        StringBuilder sb = new StringBuilder();
        sb.append("\n==== Performance Metrics ====\n");

        if (s.hasPrefillPhase()) {
            sb.append(String.format(
                    "Total achieved tok/s: %.2f. Tokens: %d, seconds: %.2f%n" +
                    "¬Prefill achieved tok/s: %.2f. Tokens: %d, seconds: %.2f%n" +
                    "¬Decode achieved tok/s: %.2f. Tokens: %d, seconds: %.2f%n",
                    s.totalRate(),      s.totalCount(),      s.totalDuration()      / 1e9,
                    s.promptEvalRate(), s.promptEvalCount(), s.promptEvalDuration() / 1e9,
                    s.evalRate(),       s.evalCount(),       s.evalDuration()       / 1e9));
        } else {
            sb.append(String.format("achieved tok/s: %.2f. Tokens: %d, seconds: %.2f%n",
                    s.totalRate(), s.totalCount(), s.totalDuration() / 1e9));
        }

        if (Boolean.parseBoolean(System.getProperty("llama.EnableTimingForTornadoVMInit", "false"))
                && s.tornadoPlanCreationDuration() > 0) {
            sb.append(String.format(
                    "GGUF Model Load: %.2f ms%n" +
                    "Compilation & CodeGen: %.2f ms%n" +
                    "Warmup: %.2f ms%n" +
                    "Read-only weights Copy-in: %.2f ms%n",
                    s.loadDuration()                          / 1_000_000.0,
                    s.tornadoPlanCreationDuration()           / 1_000_000.0,
                    s.tornadoJitDuration()                    / 1_000_000.0,
                    s.tornadoReadOnlyWeightsCopyInDuration()  / 1_000_000.0));
        }

        return sb.toString();
    }
}
