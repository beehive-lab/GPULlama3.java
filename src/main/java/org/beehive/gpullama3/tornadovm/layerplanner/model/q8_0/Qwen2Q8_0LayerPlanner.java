package org.beehive.gpullama3.tornadovm.layerplanner.model.q8_0;

import org.beehive.gpullama3.inference.state.Qwen2State;
import org.beehive.gpullama3.inference.weights.tornado.Qwen2TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.qwen2.Qwen2Configuration;
import org.beehive.gpullama3.tornadovm.layerplanner.quantization.Q8_0LayerPlanner;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.LogitsQ8_0Layer;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.Qwen2Q8_0FFNLayers;

/**
 * Qwen2Q8_0LayerPlanner: Qwen2 model with Q8_0-quantized weights.
 *
 * Follows the same pattern as LlamaQ8_0LayerPlanner but with: - Qwen2-specific FFN layers (supports GQA with bias terms) - Qwen2TornadoWeights (8-bit integer quantization) - Qwen2Configuration -
 * 2x memory compression vs FP16
 *
 * Inherits from Q8_0LayerPlanner<Qwen2State, Qwen2Configuration, Qwen2TornadoWeights>
 */
public class Qwen2Q8_0LayerPlanner extends Q8_0LayerPlanner<Qwen2State, Qwen2Configuration, Qwen2TornadoWeights> {

    public Qwen2Q8_0LayerPlanner(Qwen2State state, Model model) {
        super(state, model);
        this.activationLayer = new Activation("activationUpdate", state, weights, config);
        this.ffnLayers = new Qwen2Q8_0FFNLayers("qwen2FFN", state, weights, config, schedulerType);
        this.logitsLayer = new LogitsQ8_0Layer("logits", state, weights, config, ffnLayers.getLastFFNLayerTaskGraphID(), schedulerType);
        createTornadoInferencePlan();
    }
}
