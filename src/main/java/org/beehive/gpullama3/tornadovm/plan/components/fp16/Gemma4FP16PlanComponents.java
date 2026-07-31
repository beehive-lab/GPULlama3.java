package org.beehive.gpullama3.tornadovm.plan.components.fp16;

import org.beehive.gpullama3.inference.state.Gemma4State;
import org.beehive.gpullama3.inference.weights.tornado.Gemma4TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.gemma4.Gemma4Configuration;
import org.beehive.gpullama3.tornadovm.layers.AbstractLogitsTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.ActivationTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.TransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.Gemma4FP16FFNLayers;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.Gemma4LogitsFP16Layer;
import org.beehive.gpullama3.tornadovm.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerDetectionService;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;

/**
 * FP16 single-token plan components for the Gemma 4 architecture.
 *
 * <p>Wires together the (model-agnostic) {@link Activation} task graph, the Gemma4-specific
 * {@link Gemma4FP16FFNLayers transformer layers} (Q/K-norm, pre/post normalization sandwich,
 * sliding-window attention, GeGLU FFN, per-layer embeddings) and the Gemma4-specific
 * {@link Gemma4LogitsFP16Layer logits layer} (final logit soft-cap). Gemma 4 currently supports
 * the STANDARD execution mode only, so this implements {@link SingleTokenForwardPlanComponents}.</p>
 */
public class Gemma4FP16PlanComponents implements SingleTokenForwardPlanComponents {

    private final Gemma4State state;
    private final Gemma4TornadoWeights weights;
    private final Gemma4Configuration config;
    private final SchedulerType schedulerType;

    public Gemma4FP16PlanComponents(Gemma4State state, Model model) {
        this.state = state;
        this.config = (Gemma4Configuration) model.configuration();
        this.weights = (Gemma4TornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new Activation("activationUpdate", state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new Gemma4FP16FFNLayers("gemma4FFN", state, weights, config, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new Gemma4LogitsFP16Layer("logits", state, weights, config, previousGraphId, schedulerType);
    }
}
