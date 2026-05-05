package org.beehive.gpullama3.auxiliary;

/**
 * Singleton that accumulates fine-grained performance metrics across one inference run.
 *
 * <p>Metrics are set incrementally by different layers of the stack:</p>
 * <ul>
 *   <li>{@link #setLoadDuration} — called from {@code LlamaApp} around model file loading</li>
 *   <li>{@link #setTornadoMetrics} — called from TornadoVM plan constructors</li>
 *   <li>{@link #setInferenceMetrics} — called from InferenceEngine variants at end of generation</li>
 *   <li>{@link #setHasPrefillPhase} — called from prefill-decode engine variants</li>
 * </ul>
 *
 * <p>All durations are stored in nanoseconds. {@link #printMetrics()} prints throughput only:</p>
 * <ul>
 *   <li>Standard engine: {@code Total: X tok/s}</li>
 *   <li>Prefill-decode engines: {@code Prefill: X tok/s | Decode: Y tok/s | Total: Z tok/s}</li>
 * </ul>
 */
public final class RunMetrics {

    // ── Core metrics (nanoseconds) ────────────────────────────────────────────
    private long totalDurationNs;
    private long loadDurationNs;
    private int  promptEvalCount;
    private long promptEvalDurationNs;
    private int  evalCount;
    private long evalDurationNs;
    private boolean hasPrefillPhase;

    // ── TornadoVM-specific metrics (nanoseconds) ──────────────────────────────
    private long tornadoCompileDurationNs;
    private long tornadoWarmupDurationNs;

    // ── Singleton ─────────────────────────────────────────────────────────────
    private static final RunMetrics INSTANCE = new RunMetrics();

    private RunMetrics() {}

    // ── Setters ───────────────────────────────────────────────────────────────

    /** Records the time spent loading the model file (not including TornadoVM initialisation). */
    public static void setLoadDuration(long ns) {
        INSTANCE.loadDurationNs = ns;
    }

    /**
     * Records TornadoVM-specific initialisation durations.
     *
     * @param compileNs plan-graph construction + JIT compilation ({@code withPreCompilation()})
     * @param warmupNs  first-execution weight upload ({@code forceCopyInReadOnlyData()})
     */
    public static void setTornadoMetrics(long compileNs, long warmupNs) {
        INSTANCE.tornadoCompileDurationNs = compileNs;
        INSTANCE.tornadoWarmupDurationNs  = warmupNs;
    }

    /**
     * Records inference-phase durations at the end of a generation run.
     *
     * @param promptCount    number of prompt tokens processed (prefill)
     * @param prefillNs      wall-clock time spent in the prefill phase
     * @param generatedCount number of tokens generated (decode)
     * @param decodeNs       wall-clock time spent in the decode phase
     * @param totalNs        total wall-clock time for the full inference call
     */
    public static void setInferenceMetrics(int promptCount, long prefillNs,
                                           int generatedCount, long decodeNs,
                                           long totalNs) {
        INSTANCE.promptEvalCount      = promptCount;
        INSTANCE.promptEvalDurationNs = prefillNs;
        INSTANCE.evalCount            = generatedCount;
        INSTANCE.evalDurationNs       = decodeNs;
        INSTANCE.totalDurationNs      = totalNs;
    }

    /**
     * Signals that prefill and decode are distinct timed phases.
     * Called by {@code InferenceEngineWithPrefillDecode} and
     * {@code InferenceEngineWithBatchPrefillDecode} before returning.
     */
    public static void setHasPrefillPhase(boolean value) {
        INSTANCE.hasPrefillPhase = value;
    }

    // ── Output ────────────────────────────────────────────────────────────────

    /** Prints throughput metrics to {@code stderr}. */
    public static void printMetrics() {
        RunMetrics m = INSTANCE;

        int    totalTokens  = m.promptEvalCount + m.evalCount;
        double totalSecs    = m.totalDurationNs / 1e9;
        double totalRate    = totalSecs > 0 ? totalTokens / totalSecs : 0;

        System.err.println();
        System.err.println("==== Performance Metrics ====");
        if (m.hasPrefillPhase) {
            double prefillSecs = m.promptEvalDurationNs / 1e9;
            double decodeSecs  = m.evalDurationNs / 1e9;
            double prefillRate = (prefillSecs > 0 && m.promptEvalCount > 0)
                    ? m.promptEvalCount / prefillSecs : 0;
            double decodeRate  = (decodeSecs > 0 && m.evalCount > 0)
                    ? m.evalCount / decodeSecs : 0;
            System.err.printf(
                    "Total achieved tok/s: %.2f. Tokens: %d, seconds: %.2f%n" +
                    "¬Prefill achieved tok/s: %.2f. Tokens: %d, seconds: %.2f%n" +
                    "¬Decode achieved tok/s: %.2f. Tokens: %d, seconds: %.2f%n",
                    totalRate, totalTokens, totalSecs,
                    prefillRate, m.promptEvalCount, prefillSecs,
                    decodeRate,  m.evalCount,       decodeSecs);
        } else {
            System.err.printf("achieved tok/s: %.2f. Tokens: %d, seconds: %.2f%n",
                    totalRate, totalTokens, totalSecs);
        }
        System.err.println();
    }
}
