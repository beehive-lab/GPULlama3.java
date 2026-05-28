package org.beehive.gpullama3.tornadovm.layers;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * Abstract base for all logits layers (final vocabulary projection step).
 *
 * Holds the shared fields and calls the protected buildLogitsTaskGraph() hook once
 * during construction. Subclasses implement buildLogitsTaskGraph() to define the
 * quantization-specific task sequence; Granite variants override it to swap in
 * their scaled kernel.
 */
public abstract class AbstractLogitsLayer extends AbstractLayer {

    protected final String lastTaskGraphID;
    protected final SchedulerType schedulerType;
    private final TaskGraph logitsTaskGraph;

    protected AbstractLogitsLayer(String name, State state, Weights weights, Configuration config,
            String lastTaskGraphID, SchedulerType schedulerType) {
        super(name, state, weights, config);
        this.lastTaskGraphID = lastTaskGraphID;
        this.schedulerType = schedulerType;
        TornadoWeights tornadoWeights = requireWeightsType(weights, TornadoWeights.class,
                getClass().getSimpleName(), "TornadoTensor");
        this.logitsTaskGraph = setupLogitsTaskGraph(tornadoWeights, config);
    }

    protected abstract TaskGraph setupLogitsTaskGraph(TornadoWeights weights, Configuration config);

    public final TaskGraph getTaskGraph() {
        return logitsTaskGraph;
    }

    public final ImmutableTaskGraph getImmutableTaskGraph() {
        return logitsTaskGraph.snapshot();
    }
}
