package org.beehive.gpullama3.tornadovm.layerplanner.model.fp16;

import org.beehive.gpullama3.inference.state.DevstralState;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.devstral.DevstralConfiguration;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.DevstralFP16FFNLayers;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LogitsFP16Layer;

public class DevstralFP16LayerPlanner extends FP16LayerPlanner<DevstralState, DevstralConfiguration, LlamaTornadoWeights> {

    public DevstralFP16LayerPlanner(DevstralState state, Model model) {
        super(state, model);
        this.activationLayer = new Activation("activationUpdate", state, weights, config);
        this.ffnLayers = new DevstralFP16FFNLayers("devstralFFN", state, weights, config, schedulerType);
        this.logitsLayer = new LogitsFP16Layer("logits", state, weights, config, ffnLayers.getLastFFNLayerTaskGraphID(), schedulerType);
        createTornadoInferencePlan();
    }
}
