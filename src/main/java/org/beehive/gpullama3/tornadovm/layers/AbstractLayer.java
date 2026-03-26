package org.beehive.gpullama3.tornadovm.layers;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * Abstract base class for Activations, FFN Layers, and Logits.
 */
public abstract class AbstractLayer {

    /** Common constants used in tasks & worker-grid sizing. */
    protected static final int LOCAL_WORK_GROUP_SIZE_ALLOC = 32;
    protected static final int THREAD_SCALE_FOR_LOGITS = 8;

    protected final Weights weights;
    protected final Configuration config;
    protected final State state;
    protected final KernelContext context = new KernelContext();

    protected AbstractLayer(String taskGraphName, State state, Weights weights, Configuration config) {
        this.state = state;
        this.weights = weights;
        this.config = config;
    }

    @SuppressWarnings("unchecked")
    protected static <T> T requireWeightsType(Object weights, Class<T> expectedType, String layerName, String layout) {
        if (expectedType.isInstance(weights)) {
            return (T) weights;
        }
        throw new IllegalArgumentException(layerName + " requires " + expectedType.getSimpleName() + " with " + layout + " layout");
    }

    public abstract GridScheduler updateGridScheduler(GridScheduler scheduler);

    /** Allow subclasses to override if they need custom data transfers. */
    protected TaskGraph configureLayerDataTransfers(TaskGraph tg, int layerIndex) {
        return tg;
    }
}
