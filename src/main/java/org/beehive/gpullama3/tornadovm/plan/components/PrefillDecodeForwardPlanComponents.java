package org.beehive.gpullama3.tornadovm.plan.components;

import org.beehive.gpullama3.tornadovm.layers.AbstractLogitsLayer;
import org.beehive.gpullama3.tornadovm.layers.ActivationGraph;
import org.beehive.gpullama3.tornadovm.layers.TransformerLayerTaskGraphs;

/**
 * Components for the N+2 prefill/decode forward plan.
 *
 * <p>Extends {@link SingleTokenForwardPlanComponents} with the decode-phase
 * activation, KV-cache-aware layer group, and decode logits.</p>
 */
public interface PrefillDecodeForwardPlanComponents extends SingleTokenForwardPlanComponents {

    ActivationGraph decodeActivation();

    TransformerLayerTaskGraphs prefillDecodeLayers();

    AbstractLogitsLayer decodeLogits(String previousGraphId);
}
