package org.beehive.gpullama3.tools;

/**
 * Tuning parameters for a {@link ToolCallingSession}.
 *
 * @param maxTokens          max tokens per inference call
 * @param maxRoundTrips      max tool → result → re-inference cycles (default 1)
 * @param maxToolResultChars tool output is truncated to this length before feeding back
 * @param verbose            print step-by-step output to stdout
 * @param useGpu             use TornadoVM GPU path for inference
 */
public record ToolCallingOptions(
        int maxTokens,
        int maxRoundTrips,
        int maxToolResultChars,
        boolean verbose,
        boolean useGpu) {

    public static ToolCallingOptions defaults() {
        return new ToolCallingOptions(1024, 1, 2000, true, false);
    }

    public static ToolCallingOptions from(org.beehive.gpullama3.Options options) {
        return new ToolCallingOptions(
                options.maxTokens(),
                1,
                2000,
                true,
                options.useTornadovm());
    }
}
