package org.beehive.gpullama3.tornadovm.layerplanner.model.q4_0;

import org.beehive.gpullama3.inference.state.Qwen2State;
import org.beehive.gpullama3.inference.weights.tornado.Qwen2TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.qwen2.Qwen2Configuration;
import org.beehive.gpullama3.tornadovm.layerplanner.quantization.Q4_0LayerPlanner;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.type.q4_0.Qwen2Q4_0FFNLayers;
import org.beehive.gpullama3.tornadovm.layers.type.q4_0.LogitsQ4_0Layer;

public class Qwen2Q4_0LayerPlanner extends Q4_0LayerPlanner<Qwen2State, Qwen2Configuration, Qwen2TornadoWeights> {

    public Qwen2Q4_0LayerPlanner(Qwen2State state, Model model) {
        super(state, model);
        validateQuantizationType();
        setupTornadoForwardPlan();
    }

    @Override
    protected void initializeLayerComponents() {
        this.activationLayer = new Activation("activationUpdate", this.state, this.weights, this.config);
        this.ffnLayers = new Qwen2Q4_0FFNLayers("qwen2FFN", this.state, this.weights, this.config, this.schedulerType);
        this.logitsLayer = new LogitsQ4_0Layer("qwen2Logits", this.state, this.weights, this.config, ffnLayers.getLastTaskGraphID(), this.schedulerType);
    }

}
