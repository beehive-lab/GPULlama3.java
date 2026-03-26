package org.beehive.gpullama3.tornadovm.layerplanner.model.fp16;

import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LlamaFP16FFNLayers;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LogitsFP16Layer;

public class LlamaFP16LayerPlanner extends FP16LayerPlanner<LlamaState, LlamaConfiguration, LlamaTornadoWeights> {

    public LlamaFP16LayerPlanner(LlamaState state, Model model) {
        super(state, model);
        this.activationLayer = new Activation("activationUpdate", state, weights, config);
        this.ffnLayers = new LlamaFP16FFNLayers("llamaFFN", state, weights, config, schedulerType);
        this.logitsLayer = new LogitsFP16Layer("logits", state, weights, config, ffnLayers.getLastFFNLayerTaskGraphID(), schedulerType);
        createTornadoInferencePlan();
    }
}
