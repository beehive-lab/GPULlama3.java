package org.beehive.gpullama3.tornadovm.layerplanner.model.q8_0;

import org.beehive.gpullama3.inference.state.Gemma4State;
import org.beehive.gpullama3.inference.weights.tornado.Gemma4TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.gemma4.Gemma4Configuration;
import org.beehive.gpullama3.tornadovm.layers.ActivationGemma4;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.Gemma4Q8_0FFNLayers;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.LogitsQ8_0Layer;

/**
 * Gemma4 Q8_0 layer planner: orchestrates activation, FFN layers, and logits
 * for GPU inference with Gemma4-specific per-layer dimensions and dual RoPE.
 */
public class Gemma4Q8_0LayerPlanner extends Q8_0LayerPlanner<Gemma4State, Gemma4Configuration, Gemma4TornadoWeights> {

    public Gemma4Q8_0LayerPlanner(Gemma4State state, Model model) {
        super(state, model);
        this.activationLayer = new ActivationGemma4("activationUpdate", state, weights, config);
        this.ffnLayers = new Gemma4Q8_0FFNLayers("gemma4FFN", state, weights, config, schedulerType);
        this.logitsLayer = new LogitsQ8_0Layer("logits", state, weights, config,
                ffnLayers.getLastFFNLayerTaskGraphID(), schedulerType);
        createTornadoInferencePlan();
    }
}
