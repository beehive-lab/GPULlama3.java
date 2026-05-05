package org.beehive.gpullama3.auxiliary.metrics;

/** Renders a {@link RunMetricsSnapshot} to a string for a specific output format. */
@FunctionalInterface
public interface MetricsRenderer {
    String render(RunMetricsSnapshot snapshot);
}
