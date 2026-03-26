package org.beehive.gpullama3.tornadovm.layerplanner.quantization;

import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tornadovm.layerplanner.base.QuantizedLayerPlanner;
import org.beehive.gpullama3.tornadovm.layers.AbstractFFNLayers;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.LogitsQ8_0Layer;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

import java.util.ArrayList;
import java.util.List;

/**
 * Base for all Q8_0-quantized layer planners.
 *
 * Subclasses: LlamaQ8_0LayerPlanner, Qwen2Q8_0LayerPlanner, etc.
 *
 * Q8_0 Specific: - Uses 8-bit integer quantization with uniform scaling per 32-element block - Weights: weights.xxxByteArray arrays - Compute: dequantize on-the-fly during matmul - Memory: 2x
 * compression vs FP16
 */
public abstract class Q8_0LayerPlanner<S extends State, C extends Configuration, W extends TornadoWeights> extends QuantizedLayerPlanner<S, C, W> {

    protected Activation activationLayer;
    protected AbstractFFNLayers<?,?> ffnLayers;
    protected LogitsQ8_0Layer logitsLayer;

    // Cache for task graphs and scheduler (set once, reused)
    protected List<ImmutableTaskGraph> cachedTaskGraphs;
    protected GridScheduler cachedScheduler;

    protected Q8_0LayerPlanner(S state, Model model) {
        super(state, model);
        initializeLayerComponents();
    }

    @Override
    protected void validateQuantizationType() {
        if (this.weights.getWeightType() != GGMLType.Q8_0) {
            throw new IllegalArgumentException("Q8_0LayerPlanner requires GGMLType.Q8_0, got: " + this.weights.getWeightType());
        }
    }

    @Override
    protected void initializeLayerComponents() {
        // Override in subclasses (LlamaQ8_0LayerPlanner, etc.)
    }

    protected final void setupTornadoForwardPlan() {
        List<ImmutableTaskGraph> allTaskGraphs = new ArrayList<>();
        GridScheduler masterScheduler = new GridScheduler();

        // 1. Activation layer (common to all models)
        allTaskGraphs.add(activationLayer.getImmutableTaskGraph());
        activationLayer.updateGridScheduler(masterScheduler);

        // 2. FFN layers (N transformer layers - model-specific)
        allTaskGraphs.addAll(ffnLayers.getFFNLayerImmutableTaskGraphs());
        ffnLayers.updateGridScheduler(masterScheduler);

        // 3. Logits layer (common to all models)
        allTaskGraphs.add(logitsLayer.getTaskGraph().snapshot());
        logitsLayer.updateGridScheduler(masterScheduler);

        // Cache for future retrievals
        this.cachedTaskGraphs = allTaskGraphs;
        this.cachedScheduler = masterScheduler;
    }

    /**
     * Returns cached task graphs (used by hardware strategy pattern).
     *
     * Removed from all model-specific planners - centralized here.
     */
    public final List<ImmutableTaskGraph> getImmutableTaskGraphs() {
        return this.cachedTaskGraphs;
    }

    /**
     * Returns cached scheduler (used by hardware strategy pattern).
     *
     * Removed from all model-specific planners - centralized here.
     */
    @Override
    public final GridScheduler getGridScheduler() {
        return this.cachedScheduler;
    }

}