package org.beehive.gpullama3.auxiliary.metrics;

/**
 * Immutable snapshot of all performance metrics for a single inference run.
 *
 * <p>Raw duration fields are in nanoseconds. Derived rate fields are in tokens per second.
 * Construct via {@link #of} — it computes the derived values from the raw inputs.</p>
 */
public record RunMetricsSnapshot(
        long    totalDuration,
        long    loadDuration,
        int     promptEvalCount,
        long    promptEvalDuration,
        int     evalCount,
        long    evalDuration,
        boolean hasPrefillPhase,
        long    tornadoPlanCreationDuration,
        long    tornadoJitDuration,
        long    tornadoReadOnlyWeightsCopyInDuration,
        // derived
        int     totalCount,
        double  promptEvalRate,
        double  evalRate,
        double  totalRate
) {
    public static RunMetricsSnapshot of(
            long totalDuration,    long loadDuration,
            int  promptEvalCount,  long promptEvalDuration,
            int  evalCount,        long evalDuration,
            boolean hasPrefillPhase,
            long tornadoPlanCreationDuration,
            long tornadoJitDuration,
            long tornadoReadOnlyWeightsCopyInDuration) {

        int    totalCount     = promptEvalCount + evalCount;
        double promptEvalRate = tokensPerSecond(promptEvalCount, promptEvalDuration);
        double evalRate       = tokensPerSecond(evalCount,       evalDuration);
        double totalRate      = tokensPerSecond(totalCount,      totalDuration);

        return new RunMetricsSnapshot(
                totalDuration,    loadDuration,
                promptEvalCount,  promptEvalDuration,
                evalCount,        evalDuration,
                hasPrefillPhase,
                tornadoPlanCreationDuration, tornadoJitDuration,
                tornadoReadOnlyWeightsCopyInDuration,
                totalCount, promptEvalRate, evalRate, totalRate);
    }

    private static double tokensPerSecond(int tokens, long durationNs) {
        return (durationNs > 0 && tokens > 0) ? tokens / (durationNs / 1e9) : 0.0;
    }
}
