package org.beehive.gpullama3.tornadovm.layers;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

import java.util.List;

/**
 * Interface for a group of N batched-prefill transformer-layer TornadoVM TaskGraphs.
 *
 * <p>Implemented by {@code LlamaFP16LayersBatchPrefillMMA} and {@code LlamaQ8_0LayersBatchPrefillMMA}.</p>
 */
public interface BatchPrefillTransformerLayerTaskGraphs {
    List<ImmutableTaskGraph> getLayerImmutableTaskGraphs();

    void updateGridScheduler(GridScheduler scheduler);

    String getLastLayerTaskGraphID();
}
