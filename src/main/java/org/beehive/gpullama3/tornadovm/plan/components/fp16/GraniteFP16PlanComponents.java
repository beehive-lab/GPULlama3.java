package org.beehive.gpullama3.tornadovm.plan.components.fp16;

import org.beehive.gpullama3.inference.state.GraniteState;
import org.beehive.gpullama3.inference.weights.tornado.GraniteTornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.granite.GraniteConfiguration;
import org.beehive.gpullama3.tornadovm.layers.AbstractLogitsLayer;
import org.beehive.gpullama3.tornadovm.layers.ActivationGranite;
import org.beehive.gpullama3.tornadovm.layers.ActivationTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.TransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.GraniteFP16FFNLayers;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LogitsGraniteFP16Layer;
import org.beehive.gpullama3.tornadovm.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerDetectionService;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;

public class GraniteFP16PlanComponents implements SingleTokenForwardPlanComponents {

    private final GraniteState state;
    private final GraniteTornadoWeights weights;
    private final GraniteConfiguration config;
    private final SchedulerType schedulerType;

    public GraniteFP16PlanComponents(GraniteState state, Model model) {
        this.state = state;
        this.config = (GraniteConfiguration) model.configuration();
        this.weights = (GraniteTornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override public ActivationTaskGraph standardActivation() {
        return new ActivationGranite("activationUpdate", state, weights, config);
    }

    @Override public TransformerLayerTaskGraphs standardLayers() {
        return new GraniteFP16FFNLayers("graniteFFN", state, weights, config, schedulerType);
    }

    @Override public AbstractLogitsLayer standardLogits(String previousGraphId) {
        return new LogitsGraniteFP16Layer("logits", state, weights, config, previousGraphId, schedulerType);
    }
}
