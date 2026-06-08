package org.beehive.gpullama3.tornadovm.layers;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

import java.util.List;

/**
 * Interface for a group of N transformer-layer TornadoVM TaskGraphs (standard or prefill-decode variants).
 *
 * <p>Implemented by {@link AbstractTransformerLayerTaskGraphs} and its subclasses.</p>
 */
public interface TransformerLayerTaskGraphs {
    List<ImmutableTaskGraph> getFFNLayerImmutableTaskGraphs();

    GridScheduler updateGridScheduler(GridScheduler scheduler);

    String getLastFFNLayerTaskGraphID();
}
