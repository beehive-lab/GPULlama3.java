package org.beehive.gpullama3.tornadovm.layerplanner.model.fp16;

import org.beehive.gpullama3.inference.state.Gemma4State;
import org.beehive.gpullama3.inference.weights.tornado.Gemma4TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.gemma4.Gemma4Configuration;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.Gemma4LogitsFP16Layer;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.Gemma4FP16FFNLayers;

/**
 * Gemma4FP16LayerPlanner: Gemma 4 model with FP16 weights.
 *
 * Follows the same pattern as Qwen3FP16LayerPlanner: wires together the (model-agnostic) Activation
 * layer, Gemma4-specific FFN layers, and a Gemma4-specific logits layer (which adds the final
 * logit soft-cap), then assembles the inference plan.
 *
 * Inherits from FP16LayerPlanner<Gemma4State, Gemma4Configuration, Gemma4TornadoWeights>
 */
public class Gemma4FP16LayerPlanner extends FP16LayerPlanner<Gemma4State, Gemma4Configuration, Gemma4TornadoWeights> {

    public Gemma4FP16LayerPlanner(Gemma4State state, Model model) {
        super(state, model);
        this.activationLayer = new Activation("activationUpdate", state, weights, config);
        this.ffnLayers = new Gemma4FP16FFNLayers("gemma4FFN", state, weights, config, schedulerType);
        this.logitsLayer = new Gemma4LogitsFP16Layer("logits", state, weights, config, ffnLayers.getLastFFNLayerTaskGraphID(), schedulerType);
        createTornadoInferencePlan();
    }
}
