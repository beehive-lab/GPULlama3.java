package org.beehive.gpullama3.backend.tornado.plan.layout;

// @formatter:off
/**
 * Graph-index arithmetic for the 2N+3 batch-prefill/decode forward plan.
 *
 * <pre>
 *   [0]         batchPrefillActivation
 *   [1.N]      batchPrefillLayer_0. batchPrefillLayer_{N-1}
 *   [N+1]       decodeActivation    (consumes + re-persists KV cache)
 *   [N+2.2N+1] decodeLayer_0. decodeLayer_{N-1}
 *   [2N+2]      logits
 * </pre>
 */
public record BatchPrefillDecodeForwardTaskGraphLayout(int N) {
    public int batchActivationIdx() {
        return 0;
    }

    public int batchLayerIdx(int i) {
        return 1 + i;
    }

    public int decodeActivationIdx() {
        return N + 1;
    }

    public int decodeLayerIdx(int i) {
        return N + 2 + i;
    }

    public int logitsIdx() {
        return 2 * N + 2;
    }

    /**
     * How many distinct <b>layer graph families</b> this topology builds.
     *
     * <p>Not the graph count. A family is a set of per-layer graphs that each bind the layer's
     * weights, and the Tornado runtime allocates a device buffer per graph — so this, not {@link
     * #totalGraphs()}, is the multiplier on per-layer weight memory. This layout has 2.
     *
     * <p>{@code TornadoGraphTopology} asserts that {@code totalGraphs() == layerGraphFamilies() * N
     * + nonLayerGraphs()}, so adding a family here without updating this method fails that check
     * rather than silently under-predicting memory.
     */
    public int layerGraphFamilies() {
        return 2;
    }

    /** Graphs that are not per-layer: batch activation, decode activation and logits. */
    public int nonLayerGraphs() {
        return 3;
    }

    public int totalGraphs() {
        return 2 * N + 3;
    }
}
// @formatter:on
