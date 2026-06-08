package org.beehive.gpullama3.tornadovm.plan.layout;

// @formatter:off
/**
 * Graph-index arithmetic for the 2N+3 batch-prefill/decode forward plan.
 *
 * <pre>
 *   [0]         batchPrefillActivation
 *   [1..N]      batchPrefillLayer_0 .. batchPrefillLayer_{N-1}
 *   [N+1]       decodeActivation    (consumes + re-persists KV cache)
 *   [N+2..2N+1] decodeLayer_0 .. decodeLayer_{N-1}
 *   [2N+2]      logits
 * </pre>
 */
public record BatchPrefillDecodeForwardTaskGraphLayout(int N) {
    public int batchActivationIdx()   { return 0; }
    public int batchLayerIdx(int i)   { return 1 + i; }
    public int decodeActivationIdx()  { return N + 1; }
    public int decodeLayerIdx(int i)  { return N + 2 + i; }
    public int logitsIdx()            { return 2 * N + 2; }
    public int totalGraphs()          { return 2 * N + 3; }
}
// @formatter:on