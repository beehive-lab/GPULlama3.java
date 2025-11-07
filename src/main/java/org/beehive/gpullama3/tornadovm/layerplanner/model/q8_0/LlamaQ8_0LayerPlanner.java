package org.beehive.gpullama3.tornadovm.layerplanner.model.q8_0;

import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.weights.tornado.q8_0.LlamaTornadoWeightsQ8_0;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import org.beehive.gpullama3.tornadovm.layerplanner.quantization.Q8_0LayerPlanner;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.LlamaQ8_0FFNLayers;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.LogitsQ8_0Layer;

public class LlamaQ8_0LayerPlanner extends Q8_0LayerPlanner<LlamaState, LlamaConfiguration, LlamaTornadoWeightsQ8_0> {

    public LlamaQ8_0LayerPlanner(LlamaState state, Model model) {
        super(state, model);
        validateQuantizationType();
        setupTornadoForwardPlan();
    }

    @Override
    protected void initializeLayerComponents() {
        this.activationLayer = new Activation("activationUpdate", this.state, this.weights, this.config);
        this.ffnLayers = new LlamaQ8_0FFNLayers("llamaFFN", this.state, this.weights, this.config, this.schedulerType);
        this.logitsLayer = new LogitsQ8_0Layer("llamaLogits", this.state, this.weights, this.config, ffnLayers.getLastTaskGraphID(), this.schedulerType);
    }

}