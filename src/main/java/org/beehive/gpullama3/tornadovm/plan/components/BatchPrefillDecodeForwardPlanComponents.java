package org.beehive.gpullama3.tornadovm.plan.components;

import org.beehive.gpullama3.tornadovm.layers.ActivationTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.layers.TransformerLayerTaskGraphs;

/**
 * Components for the 2N+3 batch-prefill/decode forward plan.
 *
 * <p>Extends {@link PrefillDecodeForwardPlanComponents} with the batch-prefill
 * activation, batch layer group, KV-cache decode activation, and the
 * host-side embedding preparer.</p>
 */
public interface BatchPrefillDecodeForwardPlanComponents extends PrefillDecodeForwardPlanComponents {

    ActivationTaskGraph batchPrefillActivation(int batchSize);

    ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId);

    TransformerLayerTaskGraphs batchDecodeLayers();

    BatchPrefillTransformerLayerTaskGraphs batchPrefillLayers(int batchSize);

    EmbeddingPreparer embeddingPreparer();
}
