package org.beehive.gpullama3.tornadovm.plan.components.q8_0;

import org.beehive.gpullama3.inference.state.Qwen2State;
import org.beehive.gpullama3.inference.weights.tornado.Qwen2TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.qwen2.Qwen2Configuration;
import org.beehive.gpullama3.tornadovm.layers.AbstractLogitsTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.ActivationTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.TransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.LogitsQ8_0Layer;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.Qwen2Q8_0FFNLayers;
import org.beehive.gpullama3.tornadovm.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerDetectionService;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;

public class Qwen2Q8_0PlanComponents implements SingleTokenForwardPlanComponents {

    private final Qwen2State state;
    private final Qwen2TornadoWeights weights;
    private final Qwen2Configuration config;
    private final SchedulerType schedulerType;

    public Qwen2Q8_0PlanComponents(Qwen2State state, Model model) {
        this.state = state;
        this.config = (Qwen2Configuration) model.configuration();
        this.weights = (Qwen2TornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override public ActivationTaskGraph singleTokenActivation() {
        return new Activation("activationUpdate", state, weights, config);
    }

    @Override public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new Qwen2Q8_0FFNLayers("qwen2FFN", state, weights, config, schedulerType);
    }

    @Override public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new LogitsQ8_0Layer("logits", state, weights, config, previousGraphId, schedulerType);
    }
}
