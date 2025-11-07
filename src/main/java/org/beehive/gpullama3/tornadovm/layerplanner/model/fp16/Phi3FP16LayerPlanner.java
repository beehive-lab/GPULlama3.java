package org.beehive.gpullama3.tornadovm.layerplanner.model.fp16;

import org.beehive.gpullama3.inference.state.Phi3State;
import org.beehive.gpullama3.inference.weights.tornado.fp16.Phi3TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.phi3.Phi3Configuration;
import org.beehive.gpullama3.tornadovm.layerplanner.quantization.FP16LayerPlanner;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LogitsFP16Layer;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.Phi3FP16FFNLayers;

/**
 * Phi3FP16LayerPlanner: Phi3 model with FP16 weights.
 *
 * Follows the same pattern as Qwen3FP16LayerPlanner but with: - Phi3-specific FFN layers (combined QKV + gate/up FFN) - Phi3TornadoWeights - Phi3Configuration
 *
 * Inherits from FP16LayerPlanner<Phi3State, Phi3Configuration, Phi3TornadoWeights>
 */
public class Phi3FP16LayerPlanner extends FP16LayerPlanner<Phi3State, Phi3Configuration, Phi3TornadoWeights> {

    public Phi3FP16LayerPlanner(Phi3State state, Model model) {
        super(state, model);
        validateQuantizationType();
        setupTornadoForwardPlan();
    }

    @Override
    protected void initializeLayerComponents() {
        this.activationLayer = new Activation("activationUpdate", this.state, this.weights, this.config);
        this.ffnLayers = new Phi3FP16FFNLayers("phi3FFN", this.state, this.weights, this.config, this.schedulerType);
        this.logitsLayer = new LogitsFP16Layer("phi3Logits", this.state, this.weights, this.config, ffnLayers.getLastTaskGraphID(), this.schedulerType);
    }

}
