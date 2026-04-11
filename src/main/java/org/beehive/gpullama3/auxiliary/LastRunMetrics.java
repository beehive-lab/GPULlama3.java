package org.beehive.gpullama3.auxiliary;

/**
 * Record to store metrics from the last model run.
 * @param totalTokens The total number of tokens processed
 * @param totalNanos The total time in nanoseconds
 * @param promptEvalCount Number of tokens in the prompt
 * @param promptNanos Time to process the prompt in nanoseconds
 * @param inferenceEvalCount Number of tokens in the model's response
 * @param inferenceNanos Time to output the response in nanoseconds
 * @param tornadoCompileNanos Time to compile the tornado task graph in nanoseconds
 * @param tornadoWarmupNanos Time spent warming up until steady state in nanoseconds
 */
public record LastRunMetrics(int totalTokens, long totalNanos, int promptEvalCount, long promptNanos, int inferenceEvalCount, long inferenceNanos, long tornadoCompileNanos, long tornadoWarmupNanos) {
    /**
     * Singleton instance to store the latest metrics
     */
    private static LastRunMetrics latestMetrics;
    // Load duration is static because it only occurs once
    private static long loadDurationNanos = 0;

    /**
     * Sets the metrics for the latest run
     * @param totalTokens The total number of tokens processed
     * @param totalNanos The total time in nanoseconds
     * @param promptEvalCount Number of tokens in the prompt
     * @param promptNanos Time to process the prompt in nanoseconds
     * @param inferenceEvalCount Number of tokens in the model's response
     * @param inferenceNanos Time to output the response in nanoseconds
     * @param tornadoCompileNanos Time to compile the tornado task graph in nanoseconds
     * @param tornadoWarmupNanos Time spent warming up until steady state in nanoseconds
     */
    public static void setMetrics(int totalTokens, long totalNanos, int promptEvalCount, long promptNanos, int inferenceEvalCount, long inferenceNanos, long tornadoCompileNanos, long tornadoWarmupNanos) {
        latestMetrics = new LastRunMetrics(totalTokens, totalNanos, promptEvalCount, promptNanos, inferenceEvalCount, inferenceNanos, tornadoCompileNanos, tornadoWarmupNanos);
    }
    /**
     * @param nanos Time it takes to load the model in nanoseconds; only changes when new model is loaded
     */
    public static void setModelLoadDuration(long nanos){
        loadDurationNanos = nanos;
    }

    /**
     * Prints the metrics from the latest run to stderr
     */
    public static void printMetrics() {
        if (latestMetrics != null) {
            
            // If statements to only print model load once, and only print TornadoVM metrics if using GPU
            if (loadDurationNanos > 0) {
                double loadSeconds = loadDurationNanos / 1_000_000_000.0;
                System.err.printf("Model load time: %d ns (%.2f s)\n", loadDurationNanos, loadSeconds);
            }
            if (latestMetrics.tornadoCompileNanos() > 0) {
                double compileSeconds = latestMetrics.tornadoCompileNanos() / 1_000_000_000.0;
                System.err.printf("TornadoVM Compile Time: %d ns (%.2f s)\n", latestMetrics.tornadoCompileNanos(), compileSeconds);
            }
            if (latestMetrics.tornadoWarmupNanos() > 0) {
                double warmupSeconds = latestMetrics.tornadoWarmupNanos() / 1_000_000_000.0;
                System.err.printf("TornadoVM Warmup Time: %d ns (%.2f s)\n", latestMetrics.tornadoWarmupNanos(), warmupSeconds);
            }
            
            double totalSeconds = latestMetrics.totalNanos() / 1_000_000_000.0;
            // Time to First Token = Load Time + Prompt Eval time (which includes first decode step)
            long ttftNanos = loadDurationNanos + latestMetrics.promptNanos();
            double ttftSeconds = ttftNanos / 1_000_000_000.0;
            double promptSeconds = latestMetrics.promptNanos() / 1_000_000_000.0;
            double prefillThroughput = latestMetrics.promptEvalCount() / promptSeconds;
            double inferenceSeconds = latestMetrics.inferenceNanos() / 1_000_000_000.0;
            double decodeThroughput = latestMetrics.inferenceEvalCount() / inferenceSeconds;
            double tokensPerSecond = latestMetrics.totalTokens() / totalSeconds;
            System.err.printf("\n\nAchieved tok/s: %.2f. Total tokens: %d, Total time: %d ns (%.2f s), Time to first Token: %d ns (%.2f s)\nPrefill throughput: %.2f tok/s, Prompt tokens: %d, Prompt time: %d ns (%.2f s)\nDecode throughput: %.2f tok/s, Inference tokens: %d, Inference time: %d ns (%.2f s)\n", tokensPerSecond, latestMetrics.totalTokens(), latestMetrics.totalNanos(), totalSeconds, ttftNanos, ttftSeconds, prefillThroughput, latestMetrics.promptEvalCount(), latestMetrics.promptNanos(), promptSeconds, decodeThroughput, latestMetrics.inferenceEvalCount(), latestMetrics.inferenceNanos(), inferenceSeconds);
            
            loadDurationNanos = 0;
        }
    }
}
