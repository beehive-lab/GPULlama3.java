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
 * Extends AbstractLayer and adds FFN-specific methods: - getFfnLayerTaskGraphs(): Returns task graphs for all transformer layers - getLastTaskGraphID(): Tracks the ID of the last task graph
 *
 * All model-specific FFN layers extend this: - LlamaFP16FFNLayers, Qwen2FP16FFNLayers, Qwen3FP16FFNLayers, Phi3FP16FFNLayers - LlamaQ8_0FFNLayers, Qwen2Q8_0FFNLayers, Qwen3Q8_0FFNLayers,
 * Phi3Q8_0FFNLayers
 *
 * Used by FP16LayerPlanner and Q8_0LayerPlanner template methods for type-safe polymorphic access to any FFN layer implementation.
 */
public abstract class AbstractFFNLayers extends AbstractLayer {

    protected String lastTaskGraphID;
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
     * @param schedulerType
     *         Scheduler type (NVIDIA or NON_NVIDIA) for hardware-specific optimizations
     */
    protected AbstractFFNLayers(String taskGraphName, State state, Weights weights, Configuration config, SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config);
        this.schedulerType = schedulerType;
    }

    /**
     * Returns all task graphs for the FFN layers.
     *
     * For a model with N transformer layers, this returns N ImmutableTaskGraphs, one for each layer (containing RMSNorm, Attention, FFN computations).
     *
     * @return List of immutable task graphs (one per transformer layer)
     */
    public abstract List<ImmutableTaskGraph> getFfnLayerTaskGraphs();

    /**
     * Get the ID of the last task graph.
     *
     * Used by LogitsLayer to know where to attach the final logits computation. The last transformer layer's task graph ID is needed to chain the logits computation after all FFN layers complete.
     *
     * @return Task graph ID of the last FFN layer
     */
    @Override
    public String getLastTaskGraphID() {
        return lastTaskGraphID;
    }

    /**
     * Clear the last task graph ID.
     *
     * Used for resetting state if needed.
     */
    public void clearLastTaskGraphID() {
        lastTaskGraphID = null;
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