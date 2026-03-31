package org.beehive.gpullama3.tornadovm.layerplanner.model.fp16;

import org.beehive.gpullama3.inference.state.GraniteState;
import org.beehive.gpullama3.inference.weights.tornado.GraniteTornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.granite.GraniteConfiguration;
import org.beehive.gpullama3.tornadovm.layers.ActivationGranite;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.GraniteFP16FFNLayers;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LogitsGraniteFP16Layer;

public class GraniteFP16LayerPlanner extends FP16LayerPlanner<GraniteState, GraniteConfiguration, GraniteTornadoWeights> {

    public GraniteFP16LayerPlanner(GraniteState state, Model model) {
        super(state, model);
        this.activationLayer = new ActivationGranite("activationUpdate", state, weights, config);
        this.ffnLayers = new GraniteFP16FFNLayers("graniteFFN", state, weights, config, schedulerType);
        this.logitsLayer = new LogitsGraniteFP16Layer("logits", state, weights, config, ffnLayers.getLastFFNLayerTaskGraphID(), schedulerType);
        createTornadoInferencePlan();
    }
}
