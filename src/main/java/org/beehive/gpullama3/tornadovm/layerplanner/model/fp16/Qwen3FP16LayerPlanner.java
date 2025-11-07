package org.beehive.gpullama3.tornadovm.layerplanner.model.fp16;

import org.beehive.gpullama3.inference.state.Qwen3State;
import org.beehive.gpullama3.inference.weights.tornado.fp16.Qwen3TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
import org.beehive.gpullama3.tornadovm.layerplanner.quantization.FP16LayerPlanner;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LogitsFP16Layer;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.Qwen3FP16FFNLayers;

/**
 * Qwen3FP16LayerPlanner: Qwen3 model with FP16 weights.
 *
 * Follows the same pattern as LlamaFP16LayerPlanner but with: - Qwen3-specific FFN layers (supports GQA) - Qwen3TornadoWeights - Qwen3Configuration
 *
 * Inherits from FP16LayerPlanner<Qwen3State, Qwen3Configuration, Qwen3TornadoWeights>
 */
public class Qwen3FP16LayerPlanner extends FP16LayerPlanner<Qwen3State, Qwen3Configuration, Qwen3TornadoWeights> {

    public Qwen3FP16LayerPlanner(Qwen3State state, Model model) {
        super(state, model);
        validateQuantizationType();
        setupTornadoForwardPlan();
    }

    @Override
    protected void initializeLayerComponents() {
        this.activationLayer = new Activation("activationUpdate", this.state, this.weights, this.config);
        this.ffnLayers = new Qwen3FP16FFNLayers("qwen3FFN", this.state, this.weights, this.config, this.schedulerType);
        this.logitsLayer = new LogitsFP16Layer("qwen3Logits", this.state, this.weights, this.config, ffnLayers.getLastTaskGraphID(), this.schedulerType);
    }

}