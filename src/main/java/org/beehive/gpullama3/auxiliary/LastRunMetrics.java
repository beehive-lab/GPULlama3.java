package org.beehive.gpullama3.auxiliary;

/**
 * Record to store metrics from the last model run.
 * @param totalTokens The total number of tokens processed
 * @param totalSeconds The total time in seconds
 */
public record LastRunMetrics(int totalTokens, double totalSeconds) {
    /**
     * Singleton instance to store the latest metrics
     */
    private static LastRunMetrics latestMetrics;

    /**
     * Sets the metrics for the latest run
     *
     * @param tokens The total number of tokens processed
     * @param seconds The total time in seconds
     */
    public static void setMetrics(int tokens, double seconds) {
        latestMetrics = new LastRunMetrics(tokens, seconds);
    }

    public static LastRunMetrics getMetrics() {
        return latestMetrics;
    }

    public static String getMetricsString() {
        double tokensPerSecond = latestMetrics.totalTokens() / latestMetrics.totalSeconds();
        return String.format("\n\nachieved tok/s: %.2f. Tokens: %d, seconds: %.2f\n", tokensPerSecond, latestMetrics.totalTokens(), latestMetrics.totalSeconds());
    }

    /**
     * Prints the metrics from the latest run to stderr
     */
    public static void printMetrics() {
        if (latestMetrics != null) {
            double tokensPerSecond = latestMetrics.totalTokens() / latestMetrics.totalSeconds();
            System.err.printf("\n\nachieved tok/s: %.2f. Tokens: %d, seconds: %.2f\n", tokensPerSecond, latestMetrics.totalTokens(), latestMetrics.totalSeconds());
        }
    }
}
