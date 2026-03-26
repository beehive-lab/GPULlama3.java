package org.beehive.gpullama3.tornadovm.layerplanner.model.q8_0;

import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import org.beehive.gpullama3.tornadovm.layerplanner.quantization.Q8_0LayerPlanner;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.LlamaQ8_0FFNLayers;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.LogitsQ8_0Layer;

public class LlamaQ8_0LayerPlanner extends Q8_0LayerPlanner<LlamaState, LlamaConfiguration, LlamaTornadoWeights> {

    public LlamaQ8_0LayerPlanner(LlamaState state, Model model) {
        super(state, model);
        this.activationLayer = new Activation("activationUpdate", state, weights, config);
        this.ffnLayers = new LlamaQ8_0FFNLayers("llamaFFN", state, weights, config, schedulerType);
        this.logitsLayer = new LogitsQ8_0Layer("logits", state, weights, config, ffnLayers.getLastFFNLayerTaskGraphID(), schedulerType);
        createTornadoInferencePlan();
    }
}
