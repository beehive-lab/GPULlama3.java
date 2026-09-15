package org.beehive.gpullama3.backend.tornado.plan.layout;

import org.beehive.gpullama3.backend.tornado.plan.ExecutionMode;

/**
 * How many layer graph families each execution mode builds — read from the layouts themselves.
 *
 * <p><b>Derived, not declared.</b> The count comes from the layout records that already describe
 * each topology, so the answer lives beside the graph indices rather than in a table someone must
 * remember to update. {@link #verify} then checks each layout's own arithmetic — {@code totalGraphs
 * == families × N + nonLayerGraphs} — which is what makes an added family a test failure instead of
 * a silent under-prediction.
 *
 * <p><b>Exhaustive by construction.</b> The switch has no {@code default}, so a new {@link
 * ExecutionMode} does not compile until it states its answer here.
 */
public final class TornadoGraphTopology {

    /** An arbitrary layer count for structural checks; the identity holds for every N. */
    private static final int PROBE_LAYERS = 16;

    private TornadoGraphTopology() {}

    /** Layer graph families for {@code mode}. */
    public static int layerGraphFamilies(ExecutionMode mode, int layers) {
        return switch (mode) {
            case STANDARD -> new SingleTokenForwardTaskGraphLayout(layers).layerGraphFamilies();
            case PREFILL_DECODE ->
                    new PrefillDecodeForwardTaskGraphLayout(layers).layerGraphFamilies();
            case BATCH_PREFILL_DECODE ->
                    new BatchPrefillDecodeForwardTaskGraphLayout(layers).layerGraphFamilies();
        };
    }

    /** Graphs that are not per-layer, for {@code mode}. */
    public static int nonLayerGraphs(ExecutionMode mode, int layers) {
        return switch (mode) {
            case STANDARD -> new SingleTokenForwardTaskGraphLayout(layers).nonLayerGraphs();
            case PREFILL_DECODE -> new PrefillDecodeForwardTaskGraphLayout(layers).nonLayerGraphs();
            case BATCH_PREFILL_DECODE ->
                    new BatchPrefillDecodeForwardTaskGraphLayout(layers).nonLayerGraphs();
        };
    }

    /** Total graphs for {@code mode}. */
    public static int totalGraphs(ExecutionMode mode, int layers) {
        return switch (mode) {
            case STANDARD -> new SingleTokenForwardTaskGraphLayout(layers).totalGraphs();
            case PREFILL_DECODE -> new PrefillDecodeForwardTaskGraphLayout(layers).totalGraphs();
            case BATCH_PREFILL_DECODE ->
                    new BatchPrefillDecodeForwardTaskGraphLayout(layers).totalGraphs();
        };
    }

    /** Whether a mode's declared family count agrees with the graphs it actually lays out. */
    public static boolean verify(ExecutionMode mode, int layers) {
        return totalGraphs(mode, layers)
                == layerGraphFamilies(mode, layers) * layers + nonLayerGraphs(mode, layers);
    }

    /** Every selectable mode agrees with its own layout arithmetic. */
    public static boolean verifyAll() {
        for (ExecutionMode mode : ExecutionMode.values()) {
            if (!verify(mode, PROBE_LAYERS)) {
                return false;
            }
        }
        return true;
    }
}
