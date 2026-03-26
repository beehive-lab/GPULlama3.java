package org.beehive.gpullama3.tornadovm.layerplanner.model.q8_0;

import org.beehive.gpullama3.inference.state.GraniteState;
import org.beehive.gpullama3.inference.weights.tornado.GraniteTornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.granite.GraniteConfiguration;
import org.beehive.gpullama3.tornadovm.layerplanner.quantization.Q8_0LayerPlanner;
import org.beehive.gpullama3.tornadovm.layers.ActivationGranite;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.GraniteQ8_0FFNLayers;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.LogitsGraniteQ8_0Layer;

public class GraniteQ8_0LayerPlanner extends Q8_0LayerPlanner<GraniteState, GraniteConfiguration, GraniteTornadoWeights> {

    public GraniteQ8_0LayerPlanner(GraniteState state, Model model) {
        super(state, model);
        this.activationLayer = new ActivationGranite("activationUpdate", state, weights, config);
        this.ffnLayers = new GraniteQ8_0FFNLayers("graniteFFN", state, weights, config, schedulerType);
        this.logitsLayer = new LogitsGraniteQ8_0Layer("logits", state, weights, config, ffnLayers.getLastFFNLayerTaskGraphID(), schedulerType);
        createTornadoInferencePlan();
    }
}
