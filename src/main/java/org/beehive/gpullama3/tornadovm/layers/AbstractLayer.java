package org.beehive.gpullama3.tornadovm.layers;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;

import java.util.ArrayList;
import java.util.List;

/**
 * Minimal base with common fields/utilities so subclasses compile cleanly. Adjust or remove fields if they already exist in your project.
 */
public abstract class AbstractLayer {

    /** Common constants used in tasks & worker-grid sizing. */
    protected static final int LOCAL_WORK_GROUP_SIZE_ALLOC = 32;
    protected static final int THREAD_SCALE_FOR_LOGITS = 8;
    protected static String lastTaskGraphID;
    protected final Weights weights;
    protected final Configuration config;
    /** Often a small context/config buffer passed into kernels. Use your real type if available. */
    protected final KernelContext context = new KernelContext();
    /** Collected snapshots for scheduling / debugging. */
    protected final List<ImmutableTaskGraph> taskGraphs = new ArrayList<>();
    /** Optional: track the "main" task graph for the layer if one exists. */
    protected TaskGraph taskGraph;
    /** Shared runtime objects (exposed because kernels expect them). */
    protected State state;

    protected AbstractLayer(String taskGraphName, State state, Weights weights, Configuration config) {
        this.taskGraph = null;
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

    public abstract GridScheduler getGridScheduler();

    public abstract TaskGraph getTaskGraph();

    public abstract ImmutableTaskGraph getImmutableTaskGraph();

    /** Allow subclasses to override if they need custom transfers. */
    protected TaskGraph configureLayerDataTransfers(TaskGraph tg, int layerIndex) {
        return tg;
    }

    public String getLastTaskGraphID() {
        return lastTaskGraphID;
    }

    public void setupLastID(String taskGraphID) {
        if (lastTaskGraphID == null) {
            lastTaskGraphID = taskGraphID;
        } else if (!lastTaskGraphID.equals(taskGraphID)) {
            throw new IllegalStateException("Task graph IDs do not match: " + lastTaskGraphID + " vs " + taskGraphID);
        }
    }
}
