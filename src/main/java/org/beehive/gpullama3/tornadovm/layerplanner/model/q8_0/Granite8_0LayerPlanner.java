package org.beehive.gpullama3.tornadovm.layerplanner.model.q8_0;

import org.beehive.gpullama3.inference.state.GraniteState;
import org.beehive.gpullama3.inference.weights.tornado.GraniteTornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.granite.GraniteConfiguration;
import org.beehive.gpullama3.tornadovm.layerplanner.quantization.Q8_0LayerPlanner;
import org.beehive.gpullama3.tornadovm.layers.ActivationGranite;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.GraniteQ8_0FFNLayers;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.LogitsGraniteQ8_0Layer;

public class Granite8_0LayerPlanner extends Q8_0LayerPlanner<GraniteState, GraniteConfiguration, GraniteTornadoWeights> {

    public Granite8_0LayerPlanner(GraniteState state, Model model) {
        super(state, model);
        validateQuantizationType();
        setupTornadoForwardPlan();
    }

    @Override
    protected void initializeLayerComponents() {
        this.activationLayer = new ActivationGranite("activationUpdate", this.state, this.weights, this.config);
        this.ffnLayers = new GraniteQ8_0FFNLayers("graniteFFN", this.state, this.weights, this.config, this.schedulerType);
        this.logitsLayer = new LogitsGraniteQ8_0Layer("graniteLogits", this.state, this.weights, this.config, ffnLayers.getLastTaskGraphID(), this.schedulerType);
    }
}
