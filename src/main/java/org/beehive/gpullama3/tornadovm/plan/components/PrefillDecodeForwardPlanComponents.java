package org.beehive.gpullama3.tornadovm.plan.components;

import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlanPrefillDecode;
import org.beehive.gpullama3.tornadovm.layers.AbstractLogitsTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.ActivationTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.TransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.plan.PrefillDecodeForwardPlan;

/**
 * The necessary components that any model+quantization combination
 * should implement to support *prefill-decode inference*.
 * <p>
 *    Prefill-decode inference with TornadoVM is implemented by {@link TornadoVMMasterPlanPrefillDecode}.
 *    It employs a {@link PrefillDecodeForwardPlan} instance to represent the complete
 *    prefill-decode forward operation as a chain of distinct TornadoVM TaskGraphs.
 *    The components of this chain are represented by the following components:
 *    <ul>
 *      <li>{@link #prefillDecodeActivation()} — KV-cache-aware activation (graph 0)</li>
 *      <li>{@link #prefillDecodeTransformerLayers()} — N KV-cache-aware transformer layer task graphs (graphs 1..N)</li>
 *      <li>{@link #decodeLogits(String)} — final RMSNorm + vocabulary projection (graph N+1)</li>
 *    </ul>
 * </p>
 *
 * Note: Consult also the {@link org.beehive.gpullama3.tornadovm.plan.layout.PrefillDecodeForwardTaskGraphLayout}
 *
 */
public interface PrefillDecodeForwardPlanComponents extends SingleTokenForwardPlanComponents {

    ActivationTaskGraph prefillDecodeActivation();

    TransformerLayerTaskGraphs prefillDecodeTransformerLayers();

    AbstractLogitsTaskGraph decodeLogits(String previousGraphId);
}
