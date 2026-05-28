package org.beehive.gpullama3.tornadovm.plan.components.fp16;

import org.beehive.gpullama3.inference.state.Qwen3State;
import org.beehive.gpullama3.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
import org.beehive.gpullama3.tornadovm.layers.AbstractLogitsTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.ActivationTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.TransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LogitsFP16Layer;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.Qwen3FP16FFNLayers;
import org.beehive.gpullama3.tornadovm.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerDetectionService;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;

public class Qwen3FP16PlanComponents implements SingleTokenForwardPlanComponents {

    private final Qwen3State state;
    private final Qwen3TornadoWeights weights;
    private final Qwen3Configuration config;
    private final SchedulerType schedulerType;

    public Qwen3FP16PlanComponents(Qwen3State state, Model model) {
        this.state = state;
        this.config = (Qwen3Configuration) model.configuration();
        this.weights = (Qwen3TornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override public ActivationTaskGraph standardActivation() {
        return new Activation("activationUpdate", state, weights, config);
    }

    @Override public TransformerLayerTaskGraphs standardLayers() {
        return new Qwen3FP16FFNLayers("qwen3FFN", state, weights, config, schedulerType);
    }

    @Override public AbstractLogitsTaskGraph standardLogits(String previousGraphId) {
        return new LogitsFP16Layer("logits", state, weights, config, previousGraphId, schedulerType);
    }
}
