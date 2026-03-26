package org.beehive.gpullama3.tornadovm.layers;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.tornadovm.layerplanner.strategy.SchedulerType;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

import java.util.List;

/**
 * Abstract base class for all FFN (Feed-Forward Network) layer implementations.
 *
 * Each subclass builds N ImmutableTaskGraphs (one per FFN layer) via
 * {@link #setupFFNLayerTaskGraphs}, covering RMSNorm, Attention, and FFN computations.
 *
 * Model-specific subclasses: Llama, Qwen2, Qwen3, Phi3, Granite — each in FP16 and Q8_0 variants.
 */
public abstract class AbstractFFNLayers extends AbstractLayer {

    protected String lastFFNLayerTaskGraphID;
    protected final SchedulerType schedulerType;


    /**
     * Constructor for FFN layers.
     *
     * @param taskGraphName
     *         Name for the task graph
     * @param state
     *         Runtime state (LlamaState, Qwen2State, etc.)
     * @param weights
     *         Model weights (FP16Weights, Q8_0Weights, etc.)
     * @param config
     *         Model configuration
     */
    protected AbstractFFNLayers(String taskGraphName, State state, Weights weights, Configuration config, SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config);
        this.schedulerType = schedulerType;
    }

    /**
     * Creates the TornadoVM {@link uk.ac.manchester.tornado.api.TaskGraph} for the FFN layers.
     * It creates one TaskGraph per layer and snapshots it to produce an {@link ImmutableTaskGraph} per layer.
     * It also stores the TaskGraph ID of the last FFN layer for use by the {@link AbstractLogitsLayer}.
     */
    protected abstract List<ImmutableTaskGraph> setupFFNLayerTaskGraphs();

    /**
     * Returns all task graphs for the FFN layers.
     *
     * For a model with N transformer layers, this returns N ImmutableTaskGraphs, one for each layer (containing RMSNorm, Attention, FFN computations).
     *
     * @return List of immutable task graphs (one per transformer layer)
     */
    public abstract List<ImmutableTaskGraph> getFFNLayerTaskGraphs();

    /**
     * Returns the TaskGraph ID of the last FFN layer.
     * Used by the logits layer to chain its consumeFromDevice call.
     */
    public String getLastFFNLayerTaskGraphID() {
        return lastFFNLayerTaskGraphID;
    }

    /**
     * Configures the attention mechanism based on hardware scheduler type.
     *
     * - NVIDIA hardware: Uses Flash Attention for optimized performance
     * - NON_NVIDIA hardware: Uses parallel head processing
     *
     * This method should be called during task graph setup in subclasses.
     *
     * @return true if final normalization step should be used (NON_NVIDIA), false otherwise
     */
    protected boolean shouldUseFinalNormalization() {
        return schedulerType == SchedulerType.NON_NVIDIA;
    }
}