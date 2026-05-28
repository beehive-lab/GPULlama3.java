package org.beehive.gpullama3.tornadovm.layers;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

/**
 * Common interface for single activation graphs (standard, decode, batch-prefill variants).
 *
 * <p>Implemented by {@link Activation} and custom activation wrappers used by
 * {@link org.beehive.gpullama3.tornadovm.plan.components.SingleTokenForwardPlanComponents}.</p>
 */
public interface ActivationGraph {
    ImmutableTaskGraph getImmutableTaskGraph();
    GridScheduler updateGridScheduler(GridScheduler scheduler);
}
