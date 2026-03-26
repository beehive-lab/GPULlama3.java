package org.beehive.gpullama3.tornadovm.layerplanner.quantization;

import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tornadovm.layerplanner.base.QuantizedLayerPlanner;
import org.beehive.gpullama3.tornadovm.layers.AbstractFFNLayers;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LogitsFP16Layer;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

import java.util.ArrayList;
import java.util.List;

/**
 * Base for all FP16-quantized layer planners.
 *
 * Subclasses: LlamaFP16LayerPlanner, Qwen2FP16LayerPlanner, etc.
 *
 * FP16 Specific: - Uses half-precision floating point kernels - Weights: weights.xxxHalfFloat arrays - Compute: 2x faster than FP32 on modern GPUs
 */
public abstract class FP16LayerPlanner<S extends State, C extends Configuration, W extends TornadoWeights> extends QuantizedLayerPlanner<S, C, W> {

    protected Activation activationLayer;
    protected AbstractFFNLayers<?,?> ffnLayers;
    protected LogitsFP16Layer logitsLayer;

    protected List<ImmutableTaskGraph> immutableTaskGraphs;
    protected GridScheduler gridScheduler ;

    protected FP16LayerPlanner(S state, Model model) {
        super(state, model);
        initializeLayerComponents();
    }

    @Override
    protected void validateQuantizationType() {
        if (this.weights.getWeightType() != GGMLType.F16) {
            throw new IllegalArgumentException("FP16LayerPlanner requires GGMLType.F16, got: " + this.weights.getWeightType());
        }
    }

    @Override
    protected void initializeLayerComponents() {
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
        this.immutableTaskGraphs = allTaskGraphs;
        this.gridScheduler = masterScheduler;
    }

    /**
     * Returns cached task graphs (used by hardware strategy pattern).
     *
     * Removed from all model-specific planners - centralized here.
     */
    public final List<ImmutableTaskGraph> getImmutableTaskGraphs() {
        return this.immutableTaskGraphs;
    }

    /**
     * Returns cached scheduler (used by hardware strategy pattern).
     *
     * Removed from all model-specific planners - centralized here.
     */
    @Override
    public final GridScheduler getGridScheduler() {
        return this.gridScheduler;
    }

}