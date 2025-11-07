package org.beehive.gpullama3.tornadovm.layerplanner.model.fp16;

import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.weights.tornado.fp16.LlamaTornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import org.beehive.gpullama3.tornadovm.layerplanner.quantization.FP16LayerPlanner;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LlamaFP16FFNLayers;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LogitsFP16Layer;

public class LlamaFP16LayerPlanner extends FP16LayerPlanner<LlamaState, LlamaConfiguration, LlamaTornadoWeights> {

    public LlamaFP16LayerPlanner(LlamaState state, Model model) {
        super(state, model);
        validateQuantizationType();
        setupTornadoForwardPlan();
    }

    @Override
    protected void initializeLayerComponents() {
        this.activationLayer = new Activation("activationUpdate", this.state, this.weights, this.config);
        this.ffnLayers = new LlamaFP16FFNLayers("llamaFFN", this.state, this.weights, this.config, this.schedulerType);
        this.logitsLayer = new LogitsFP16Layer("llamaLogits", this.state, this.weights, this.config, ffnLayers.getLastTaskGraphID(), this.schedulerType);
    }

}