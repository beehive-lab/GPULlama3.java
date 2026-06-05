package org.beehive.gpullama3.tornadovm.plan.components.fp16;

import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.mistral.MistralConfiguration;
import org.beehive.gpullama3.tornadovm.layers.AbstractLogitsTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.ActivationTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.TransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LogitsFP16Layer;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.MistralFP16FFNLayers;
import org.beehive.gpullama3.tornadovm.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerDetectionService;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;

public class MistralFP16PlanComponents implements SingleTokenForwardPlanComponents {

    private final LlamaState state;
    private final LlamaTornadoWeights weights;
    private final MistralConfiguration config;
    private final SchedulerType schedulerType;

    public MistralFP16PlanComponents(LlamaState state, Model model) {
        this.state = state;
        this.config = (MistralConfiguration) model.configuration();
        this.weights = (LlamaTornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override public ActivationTaskGraph singleTokenActivation() {
        return new Activation("activationUpdate", state, weights, config);
    }

    @Override public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new MistralFP16FFNLayers("mistralFFN", state, weights, config, schedulerType);
    }

    @Override public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new LogitsFP16Layer("logits", state, weights, config, previousGraphId, schedulerType);
    }
}
