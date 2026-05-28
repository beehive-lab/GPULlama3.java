package org.beehive.gpullama3.tornadovm.layers;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

/**
 * Common interface for a single activation task graph (standard, decode, and batch-prefill variants).
 *
 * <p>Implemented by {@link Activation} and custom activation wrappers used by
 * {@link org.beehive.gpullama3.tornadovm.plan.components.SingleTokenForwardPlanComponents}.</p>
 */
public interface ActivationTaskGraph {
    ImmutableTaskGraph getImmutableTaskGraph();
    GridScheduler updateGridScheduler(GridScheduler scheduler);
}
