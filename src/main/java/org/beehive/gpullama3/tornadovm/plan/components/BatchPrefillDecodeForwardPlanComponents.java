package org.beehive.gpullama3.tornadovm.plan.components;

import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlanBatchPrefillDecode;
import org.beehive.gpullama3.tornadovm.layers.ActivationTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.layers.TransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.plan.BatchPrefillDecodeForwardPlan;

/**
 * The necessary components that any model+quantization combination
 * should implement to support *batch-prefill/decode inference*.
 * <p>
 *    Batch-prefill/decode inference with TornadoVM is implemented by {@link TornadoVMMasterPlanBatchPrefillDecode}.
 *    It employs a {@link BatchPrefillDecodeForwardPlan} instance to represent the complete
 *    batch-prefill/decode forward operation as a chain of distinct TornadoVM TaskGraphs.
 *    The components of this chain are represented by the following components:
 *    <ul>
 *      <li>{@link #batchPrefillActivation(int)} — B×dim embedding → FP32 batch activation (graph 0)</li>
 *      <li>{@link #batchPrefillTransformerLayers(int)} — N batch transformer layer task graphs (graphs 1..N)</li>
 *      <li>{@link #batchDecodeActivation(String)} — single-token decode activation (graph N+1)</li>
 *      <li>{@link #batchDecodeTransformerLayers()} — N decode transformer layer task graphs (graphs N+2..2N+1)</li>
 *      <li>{@link #decodeLogits(String)} (inherited) — final RMSNorm + vocabulary projection (graph 2N+2)</li>
 *    </ul>
 * </p>
 *
 * Note: Consult also the {@link org.beehive.gpullama3.tornadovm.plan.layout.BatchPrefillDecodeForwardTaskGraphLayout}
 */
public interface BatchPrefillDecodeForwardPlanComponents extends PrefillDecodeForwardPlanComponents {

    ActivationTaskGraph batchPrefillActivation(int batchSize);

    ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId);

    TransformerLayerTaskGraphs batchDecodeTransformerLayers();

    BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(int batchSize);

}
