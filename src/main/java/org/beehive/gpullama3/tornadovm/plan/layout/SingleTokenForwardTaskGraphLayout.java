package org.beehive.gpullama3.tornadovm.plan.layout;

/**
 * Graph-index arithmetic for the N+2 single-token forward plan.
 *
 * <pre>
 *   [0]      activation
 *   [1..N]   layer_0 .. layer_{N-1}
 *   [N+1]    logits
 * </pre>
 */
public record SingleTokenForwardTaskGraphLayout(int N) {
    public int activationIdx()  { return 0; }
    public int layerIdx(int i)  { return 1 + i; }
    public int logitsIdx()      { return N + 1; }
    public int totalGraphs()    { return N + 2; }
}
