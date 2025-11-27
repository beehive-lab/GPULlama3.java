package org.beehive.gpullama3.tornadovm.layerplanner.model.q4_0;

import org.beehive.gpullama3.inference.state.Qwen3State;
import org.beehive.gpullama3.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
import org.beehive.gpullama3.tornadovm.layerplanner.quantization.Q4_0LayerPlanner;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.type.q4_0.Qwen3Q4_0FFNLayers;
import org.beehive.gpullama3.tornadovm.layers.type.q4_0.LogitsQ4_0Layer;

/**
 * Qwen3Q4_0LayerPlanner: Qwen3 model with Q4_0-quantized weights.
 *
 * Follows the same pattern as LlamaQ4_0LayerPlanner but with:
 * - Qwen3-specific FFN layers (supports GQA)
 * - Qwen3TornadoWeights (4-bit integer quantization)
 * - Qwen3Configuration
 * - 4x memory compression vs FP16, 2x vs Q8_0
 *
 * Inherits from Q4_0LayerPlanner<Qwen3State, Qwen3Configuration, Qwen3TornadoWeights>
 */
public class Qwen3Q4_0LayerPlanner extends Q4_0LayerPlanner<Qwen3State, Qwen3Configuration, Qwen3TornadoWeights> {

    public Qwen3Q4_0LayerPlanner(Qwen3State state, Model model) {
        super(state, model);
        validateQuantizationType();
        setupTornadoForwardPlan();
    }

    @Override
    protected void initializeLayerComponents() {
        this.activationLayer = new Activation("activationUpdate", this.state, this.weights, this.config);
        this.ffnLayers = new Qwen3Q4_0FFNLayers("qwen3FFN", this.state, this.weights, this.config, this.schedulerType);
        this.logitsLayer = new LogitsQ4_0Layer("qwen3Logits", this.state, this.weights, this.config, ffnLayers.getLastTaskGraphID(),this.schedulerType);
    }
}
