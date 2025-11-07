package org.beehive.gpullama3.tornadovm.layerplanner.model.q8_0;

import org.beehive.gpullama3.inference.state.Phi3State;
import org.beehive.gpullama3.inference.weights.tornado.q8_0.Phi3TornadoWeightsQ8_0;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.phi3.Phi3Configuration;
import org.beehive.gpullama3.tornadovm.layerplanner.quantization.Q8_0LayerPlanner;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.LogitsQ8_0Layer;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.Phi3Q8_0FFNLayers;

/**
 * Phi3Q8_0LayerPlanner: Phi3 model with Q8_0-quantized weights.
 *
 * Follows the same pattern as Qwen3Q8_0LayerPlanner but with: - Phi3-specific FFN layers (combined QKV + gate/up FFN) - Phi3TornadoWeightsQ8_0 (8-bit integer quantization) - Phi3Configuration - 2x
 * memory compression vs FP16
 *
 * Inherits from Q8_0LayerPlanner<Phi3State, Phi3Configuration, Phi3TornadoWeightsQ8_0>
 */
public class Phi3Q8_0LayerPlanner extends Q8_0LayerPlanner<Phi3State, Phi3Configuration, Phi3TornadoWeightsQ8_0> {

    public Phi3Q8_0LayerPlanner(Phi3State state, Model model) {
        super(state, model);
        validateQuantizationType();
        setupTornadoForwardPlan();
    }

    @Override
    protected void initializeLayerComponents() {
        this.activationLayer = new Activation("activationUpdate", this.state, this.weights, this.config);
        this.ffnLayers = new Phi3Q8_0FFNLayers("phi3FFN", this.state, this.weights, this.config, this.schedulerType);
        this.logitsLayer = new LogitsQ8_0Layer("phi3Logits", this.state, this.weights, this.config, ffnLayers.getLastTaskGraphID(), this.schedulerType);
    }

}
