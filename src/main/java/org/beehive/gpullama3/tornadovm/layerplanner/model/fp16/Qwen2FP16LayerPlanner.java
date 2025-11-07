package org.beehive.gpullama3.tornadovm.layerplanner.model.fp16;

import org.beehive.gpullama3.inference.state.Qwen2State;
import org.beehive.gpullama3.inference.weights.tornado.fp16.Qwen2TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.qwen2.Qwen2Configuration;
import org.beehive.gpullama3.tornadovm.layerplanner.quantization.FP16LayerPlanner;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LogitsFP16Layer;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.Qwen2FP16FFNLayers;

/**
 * Qwen2FP16LayerPlanner: Qwen2 model with FP16 weights.
 *
 * Follows the same pattern as LlamaFP16LayerPlanner but with: - Qwen2-specific FFN layers (supports GQA with bias terms) - Qwen2TornadoWeights - Qwen2Configuration
 *
 * Inherits from FP16LayerPlanner<Qwen2State, Qwen2Configuration, Qwen2TornadoWeights>
 */
public class Qwen2FP16LayerPlanner extends FP16LayerPlanner<Qwen2State, Qwen2Configuration, Qwen2TornadoWeights> {

    public Qwen2FP16LayerPlanner(Qwen2State state, Model model) {
        super(state, model);
        validateQuantizationType();
        setupTornadoForwardPlan();
    }

    @Override
    protected void initializeLayerComponents() {
        this.activationLayer = new Activation("activationUpdate", this.state, this.weights, this.config);
        this.ffnLayers = new Qwen2FP16FFNLayers("qwen2FFN", this.state, this.weights, this.config, this.schedulerType);
        this.logitsLayer = new LogitsFP16Layer("qwen2Logits", this.state, this.weights, this.config, ffnLayers.getLastTaskGraphID(), this.schedulerType);
    }
}
