package org.beehive.gpullama3.tornadovm.plan.components.q8_0;

import org.beehive.gpullama3.inference.state.Phi3State;
import org.beehive.gpullama3.inference.weights.tornado.Phi3TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.phi3.Phi3Configuration;
import org.beehive.gpullama3.tornadovm.layers.AbstractLogitsTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.ActivationTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.TransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.LogitsQ8_0Layer;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.Phi3Q8_0FFNLayers;
import org.beehive.gpullama3.tornadovm.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerDetectionService;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;

public class Phi3Q8_0PlanComponents implements SingleTokenForwardPlanComponents {

    private final Phi3State state;
    private final Phi3TornadoWeights weights;
    private final Phi3Configuration config;
    private final SchedulerType schedulerType;

    public Phi3Q8_0PlanComponents(Phi3State state, Model model) {
        this.state = state;
        this.config = (Phi3Configuration) model.configuration();
        this.weights = (Phi3TornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new Activation("activationUpdate", state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new Phi3Q8_0FFNLayers("phi3FFN", state, weights, config, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new LogitsQ8_0Layer("logits", state, weights, config, previousGraphId, schedulerType);
    }
}
