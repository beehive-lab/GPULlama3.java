package org.beehive.gpullama3.tornadovm.layerplanner.model.q4_0;

import org.beehive.gpullama3.inference.state.Phi3State;
import org.beehive.gpullama3.inference.weights.tornado.Phi3TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.phi3.Phi3Configuration;
import org.beehive.gpullama3.tornadovm.layerplanner.quantization.Q4_0LayerPlanner;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.type.q4_0.Phi3Q4_0FFNLayers;
import org.beehive.gpullama3.tornadovm.layers.type.q4_0.LogitsQ4_0Layer;

/**
 * Phi3Q4_0LayerPlanner: Phi3 model with Q4_0-quantized weights.
 *
 * Follows the same pattern as Qwen3Q4_0LayerPlanner but with:
 * - Phi3-specific FFN layers (combined QKV + gate/up FFN)
 * - Phi3TornadoWeights (4-bit integer quantization)
 * - Phi3Configuration
 * - 4x memory compression vs FP16, 2x vs Q8_0
 *
 * Inherits from Q4_0LayerPlanner<Phi3State, Phi3Configuration, Phi3TornadoWeights>
 */
public class Phi3Q4_0LayerPlanner extends Q4_0LayerPlanner<Phi3State, Phi3Configuration, Phi3TornadoWeights> {

    public Phi3Q4_0LayerPlanner(Phi3State state, Model model) {
        super(state, model);
        validateQuantizationType();
        setupTornadoForwardPlan();
    }

    @Override
    protected void initializeLayerComponents() {
        this.activationLayer = new Activation("activationUpdate", this.state, this.weights, this.config);
        this.ffnLayers = new Phi3Q4_0FFNLayers("phi3FFN", this.state, this.weights, this.config, this.schedulerType);
        this.logitsLayer = new LogitsQ4_0Layer("phi3Logits", this.state, this.weights, this.config, ffnLayers.getLastTaskGraphID(), this.schedulerType);
    }

}
