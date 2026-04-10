package org.beehive.gpullama3.auxiliary;

/**
 * Record to store metrics from the last model run.
 * @param totalTokens The total number of tokens processed
 * @param totalNanos The total time in nanoseconds
 * @param promptEvalCount Number of tokens in the prompt
 * @param promptNanos Time to process the prompt in nanoseconds
 * @param inferenceEvalCount Number of tokens in the model's response
 * @param inferenceNanos Time to output the response in nanoseconds
 */
public record LastRunMetrics(int totalTokens, long totalNanos, int promptEvalCount, long promptNanos, int inferenceEvalCount, long inferenceNanos) {
    /**
     * Singleton instance to store the latest metrics
     */
    private static LastRunMetrics latestMetrics;

    /**
     * Sets the metrics for the latest run
     * @param totalTokens The total number of tokens processed
     * @param totalNanos The total time in nanoseconds
     * @param promptEvalCount Number of tokens in the prompt
     * @param promptNanos Time to process the prompt in nanoseconds
     * @param inferenceEvalCount Number of tokens in the model's response
     * @param inferenceNanos Time to output the response in nanoseconds
     */
    public static void setMetrics(int totalTokens, long totalNanos, int promptEvalCount, long promptNanos, int inferenceEvalCount, long inferenceNanos) {
        latestMetrics = new LastRunMetrics(totalTokens, totalNanos, promptEvalCount, promptNanos, inferenceEvalCount, inferenceNanos);
    }

    /**
     * Prints the metrics from the latest run to stderr
     */
    public static void printMetrics() {
        if (latestMetrics != null) {
            double totalSeconds = latestMetrics.totalNanos() / 1_000_000_000.0;
            double promptSeconds = latestMetrics.promptNanos() / 1_000_000_000.0;
            double prefillThroughput = latestMetrics.promptEvalCount() / promptSeconds;
            double inferenceSeconds = latestMetrics.inferenceNanos() / 1_000_000_000.0;
            double decodeThroughput = latestMetrics.inferenceEvalCount() / inferenceSeconds;
            double tokensPerSecond = latestMetrics.totalTokens() / totalSeconds;
            System.err.printf("\n\nAchieved tok/s: %.2f. Total tokens: %d, Total time: %d ns (%.2f s)\nPrefill throughput: %.2f tok/s, Prompt tokens: %d, Prompt time: %d ns (%.2f s)\nDecode throughput: %.2f tok/s, Inference tokens: %d, Inference time: %d ns (%.2f s)\n", tokensPerSecond, latestMetrics.totalTokens(), latestMetrics.totalNanos(), totalSeconds, prefillThroughput, latestMetrics.promptEvalCount(), latestMetrics.promptNanos(), promptSeconds, decodeThroughput, latestMetrics.inferenceEvalCount(), latestMetrics.inferenceNanos(), inferenceSeconds);
        }
    }
}
