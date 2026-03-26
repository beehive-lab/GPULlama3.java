package org.beehive.gpullama3.tornadovm.layers;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.tornadovm.layerplanner.strategy.SchedulerType;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;

import java.util.List;
import java.util.stream.IntStream;

/**
 * Abstract base class for all FFN (Feed-Forward Network) layer implementations.
 * Extended by model and quantization-specific subclasses that provide specific implementations.
 */
public abstract class AbstractFFNLayers<W extends Weights, C extends Configuration> extends AbstractLayer {

    /**
     * List of TornadoVM {@link ImmutableTaskGraph}s, one per FFN layer.
     * Build by {@link #setupFFNLayers()}.
     */
    private List<ImmutableTaskGraph> ffnLayerITGs;
    protected final W weights;
    protected final C config;

    protected String lastFFNLayerTaskGraphID;
    protected final SchedulerType schedulerType;

    protected AbstractFFNLayers(String taskGraphName, State state, W weights, C config, SchedulerType schedulerType) {
        super(taskGraphName, state, weights, config);
        this.weights = weights;
        this.config = config;
        this.schedulerType = schedulerType;
        // the ffnLayerITGs is initialized on subclasses
        // due to some model-specific values (i.e. in Qwen3)
    }

    /**
     * Creates the {@link ImmutableTaskGraph} list for each FFN layer.
     */
    protected void setupFFNLayers() {
        int numLayers = config.numberOfLayers();

        this.ffnLayerITGs = IntStream.range(0, numLayers)
                .mapToObj(this::setupFFNLayer)
                .toList();
    }

    /**
     * Creates the TaskGraph for a specific FFN layer and produces the {@link ImmutableTaskGraph}.
     * In addition, it stores the TaskGraph ID of the last FFN layer for use by the {@link AbstractLogitsLayer}.
     */
    private ImmutableTaskGraph setupFFNLayer(int layerIndex) {
        TaskGraph tg = createFFNLayerTaskGraph(layerIndex);

        if (layerIndex == config.numberOfLayers() - 1) {
            lastFFNLayerTaskGraphID = tg.getTaskGraphName();
        }

        return tg.snapshot();
    }

    /**
     * Model and quantization-specific implementation of the FFN layer task graph.
     */
    protected abstract TaskGraph createFFNLayerTaskGraph(int layerIndex);

    public List<ImmutableTaskGraph> getFFNLayerImmutableTaskGraphs() {
        return ffnLayerITGs;
    }

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