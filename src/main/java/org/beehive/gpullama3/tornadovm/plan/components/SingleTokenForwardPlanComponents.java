package org.beehive.gpullama3.tornadovm.plan.components;

import org.beehive.gpullama3.tornadovm.layers.AbstractLogitsLayer;
import org.beehive.gpullama3.tornadovm.layers.ActivationTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.TransformerLayerTaskGraphs;

/**
 * Components for the single-token forward pass.
 *
 * <p>All model+quantization combinations implement this interface.
 * Models that support prefill/decode modes implement the richer
 * {@link PrefillDecodeForwardPlanComponents} or {@link BatchPrefillDecodeForwardPlanComponents}.</p>
 */
public interface SingleTokenForwardPlanComponents {

    ActivationTaskGraph standardActivation();

    TransformerLayerTaskGraphs standardLayers();

    AbstractLogitsLayer standardLogits(String previousGraphId);
}
