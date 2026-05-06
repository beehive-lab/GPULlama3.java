package org.beehive.gpullama3.auxiliary;

import org.beehive.gpullama3.auxiliary.metrics.GitHubMetricsRenderer;
import org.beehive.gpullama3.auxiliary.metrics.HumanMetricsRenderer;
import org.beehive.gpullama3.auxiliary.metrics.JsonMetricsRenderer;
import org.beehive.gpullama3.auxiliary.metrics.MetricsRenderer;
import org.beehive.gpullama3.auxiliary.metrics.RunMetricsSnapshot;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Singleton that accumulates fine-grained performance metrics across one inference run.
 *
 * <p>Metrics are set incrementally by different layers of the stack:</p>
 * <ul>
 *   <li>{@link #setLoadDuration} — called from {@code ModelLoader}</li>
 *   <li>{@link #setTornadoMetrics} — called from TornadoVM plan constructors</li>
 *   <li>{@link #setInferenceMetrics} — called from InferenceEngine variants at end of generation</li>
 *   <li>{@link #setHasPrefillPhase} — called from prefill-decode engine variants</li>
 * </ul>
 *
 * <p>All durations are stored in nanoseconds. {@link #printMetrics()} builds an immutable
 * {@link RunMetricsSnapshot}, selects a {@link MetricsRenderer}, and writes to the configured sink.</p>
 *
 * <p>Configurable via system properties:</p>
 * <ul>
 *   <li>{@code llama.metrics.format} — {@code human} (default) | {@code json} | {@code github}</li>
 *   <li>{@code llama.metrics.output} — {@code stderr} (default) | {@code stdout} | {@code file}</li>
 *   <li>{@code llama.metrics.file}   — target path when {@code output=file}</li>
 * </ul>
 */
public final class RunMetrics {

    // ── Core metrics (nanoseconds) ────────────────────────────────────────────
    private long    totalDurationNs;
    private long    loadDurationNs;
    private int     promptEvalCount;
    private long    promptEvalDurationNs;
    private int     evalCount;
    private long    evalDurationNs;
    private boolean hasPrefillPhase;

    // ── TornadoVM-specific metrics (nanoseconds) ──────────────────────────────
    private long tornadoPlanCreationNs;
    private long tornadoJitNs;
    private long readOnlyWeightsCopyInNs;

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
     * @param planCreationNs  task-graph construction ({@code createExecutionPlan()})
     * @param jitNs           JIT compilation ({@code withPreCompilation()})
     * @param weightCopyNs    first-execution weight upload ({@code forceCopyInReadOnlyData()})
     */
    public static void setTornadoMetrics(long planCreationNs, long jitNs, long weightCopyNs) {
        INSTANCE.tornadoPlanCreationNs   = planCreationNs;
        INSTANCE.tornadoJitNs            = jitNs;
        INSTANCE.readOnlyWeightsCopyInNs = weightCopyNs;
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

    // ── Snapshot ──────────────────────────────────────────────────────────────

    /** Returns an immutable snapshot of all currently collected metrics. */
    public static RunMetricsSnapshot snapshot() {
        RunMetrics m = INSTANCE;
        return RunMetricsSnapshot.of(
                m.totalDurationNs,      m.loadDurationNs,
                m.promptEvalCount,      m.promptEvalDurationNs,
                m.evalCount,            m.evalDurationNs,
                m.hasPrefillPhase,
                m.tornadoPlanCreationNs, m.tornadoJitNs,
                m.readOnlyWeightsCopyInNs);
    }

    // ── Output ────────────────────────────────────────────────────────────────

    /**
     * Builds a snapshot, selects a renderer based on {@code llama.metrics.format},
     * and writes the result to the sink configured by {@code llama.metrics.output}.
     */
    public static void printMetrics() {
        RunMetricsSnapshot snap = snapshot();

        MetricsRenderer renderer = switch (System.getProperty("llama.metrics.format", "human").toLowerCase()) {
            case "json"   -> new JsonMetricsRenderer();
            case "github" -> new GitHubMetricsRenderer();
            default       -> new HumanMetricsRenderer();
        };

        String rendered = renderer.render(snap);

        switch (System.getProperty("llama.metrics.output", "stderr").toLowerCase()) {
            case "stdout" -> System.out.print(rendered);
            case "file"   -> writeToFile(rendered);
            default       -> System.err.print(rendered);
        }
    }

    private static void writeToFile(String content) {
        String filePath = System.getProperty("llama.metrics.file");
        if (filePath == null || filePath.isBlank()) {
            throw new IllegalStateException(
                    "llama.metrics.output=file requires llama.metrics.file to be set");
        }
        Path path = Path.of(filePath);
        try {
            Path parent = path.getParent();
            if (parent != null) Files.createDirectories(parent);
            Files.writeString(path, content);
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to write metrics to " + filePath, e);
        }
    }
}
