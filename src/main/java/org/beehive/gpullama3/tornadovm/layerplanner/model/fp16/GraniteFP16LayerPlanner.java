package org.beehive.gpullama3.tornadovm.layerplanner.model.fp16;

import org.beehive.gpullama3.inference.state.GraniteState;
import org.beehive.gpullama3.inference.weights.tornado.GraniteTornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.granite.GraniteConfiguration;
import org.beehive.gpullama3.tornadovm.layerplanner.quantization.FP16LayerPlanner;
import org.beehive.gpullama3.tornadovm.layers.ActivationGranite;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.GraniteFP16FFNLayers;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LogitsGraniteFP16Layer;

public class GraniteFP16LayerPlanner extends FP16LayerPlanner<GraniteState, GraniteConfiguration, GraniteTornadoWeights> {
    public GraniteFP16LayerPlanner(GraniteState state, Model model) {
        super(state, model);
        validateQuantizationType();
        setupTornadoForwardPlan();
    }

    @Override
    protected void initializeLayerComponents() {
        this.activationLayer = new ActivationGranite("activationUpdate", this.state, this.weights, this.config);
        this.ffnLayers = new GraniteFP16FFNLayers("graniteFFN", this.state, this.weights, this.config, this.schedulerType);
        this.logitsLayer = new LogitsGraniteFP16Layer("graniteLogits", this.state, this.weights, this.config, ffnLayers.getLastTaskGraphID(), this.schedulerType);
    }

}
