package org.beehive.gpullama3.tornadovm.layerplanner.model.q8_0;

import org.beehive.gpullama3.inference.state.Qwen3State;
import org.beehive.gpullama3.inference.weights.tornado.q8_0.Qwen3TornadoWeightsQ8_0;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
import org.beehive.gpullama3.tornadovm.layerplanner.quantization.Q8_0LayerPlanner;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.LogitsQ8_0Layer;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.Qwen3Q8_0FFNLayers;

/**
 * Qwen3Q8_0LayerPlanner: Qwen3 model with Q8_0-quantized weights.
 *
 * Follows the same pattern as LlamaQ8_0LayerPlanner but with: - Qwen3-specific FFN layers (supports GQA) - Qwen3Q8_0TornadoWeights (8-bit integer quantization) - Qwen3Configuration - 2x memory
 * compression vs FP16
 *
 * Inherits from Q8_0LayerPlanner<Qwen3State, Qwen3Configuration, Qwen3Q8_0TornadoWeights>
 */
public class Qwen3Q8_0LayerPlanner extends Q8_0LayerPlanner<Qwen3State, Qwen3Configuration, Qwen3TornadoWeightsQ8_0> {

    public Qwen3Q8_0LayerPlanner(Qwen3State state, Model model) {
        super(state, model);
        validateQuantizationType();
        setupTornadoForwardPlan();
    }

    @Override
    protected void initializeLayerComponents() {
        this.activationLayer = new Activation("activationUpdate", this.state, this.weights, this.config);
        this.ffnLayers = new Qwen3Q8_0FFNLayers("qwen3FFN", this.state, this.weights, this.config, this.schedulerType);
        this.logitsLayer = new LogitsQ8_0Layer("qwen3Logits", this.state, this.weights, this.config, ffnLayers.getLastTaskGraphID(), this.schedulerType);
    }
}