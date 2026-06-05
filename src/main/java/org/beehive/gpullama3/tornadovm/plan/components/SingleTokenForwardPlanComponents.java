package org.beehive.gpullama3.tornadovm.plan.components;

import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlanSingleToken;
import org.beehive.gpullama3.tornadovm.layers.AbstractLogitsTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.ActivationTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.TransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.plan.SingleTokenForwardPlan;

/**
 * The necessary components that any model+quantization combination
 * should implement to support *single-token inference*.
 * <p>
 *    Single-token inference with TornadoVM is implemented by {@link TornadoVMMasterPlanSingleToken}.
 *    It employees a {@link SingleTokenForwardPlan} instance to represent the complete
 *    single-token forward operation as a chain of distinct TornadoVM TaskGraphs.
 *    The components of this chain are represented by the following components:
 *    <ul>
 *      <li>{@link #singleTokenActivation()} — embedding lookup → FP32 activation (graph 0)</li>
 *      <li>{@link #singleTokenTransformerLayers()} — N transformer layer TaskGraphs (graphs 1..N)</li>
 *      <li>{@link #singleTokenLogits(String)} — final RMSNorm + vocabulary projection (graph N+1)</li>
 *    </ul>
 * </p>
 *
 * Note: Consult also the {@link org.beehive.gpullama3.tornadovm.plan.layout.SingleTokenForwardTaskGraphLayout}
 */
public interface SingleTokenForwardPlanComponents {

    ActivationTaskGraph singleTokenActivation();

    TransformerLayerTaskGraphs singleTokenTransformerLayers();

    AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId);
}
