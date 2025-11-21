package org.beehive.gpullama3.tornadovm.layerplanner.model.q4_0;

import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import org.beehive.gpullama3.tornadovm.layerplanner.quantization.Q4_0LayerPlanner;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.type.q4_0.LlamaQ4_0FFNLayers;
import org.beehive.gpullama3.tornadovm.layers.type.q4_0.LogitsQ4_0Layer;

public class LlamaQ4_0LayerPlanner extends Q4_0LayerPlanner<LlamaState, LlamaConfiguration, LlamaTornadoWeights> {

    public LlamaQ4_0LayerPlanner(LlamaState state, Model model) {
        super(state, model);
        validateQuantizationType();
        setupTornadoForwardPlan();
    }

    @Override
    protected void initializeLayerComponents() {
        this.activationLayer = new Activation("activationUpdate", this.state, this.weights, this.config);
        this.ffnLayers = new LlamaQ4_0FFNLayers("llamaFFN", this.state, this.weights, this.config, this.schedulerType);
        this.logitsLayer = new LogitsQ4_0Layer("llamaLogits", this.state, this.weights, this.config, ffnLayers.getLastTaskGraphID(), this.schedulerType);
    }

}
