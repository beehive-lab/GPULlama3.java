package org.beehive.gpullama3.backend.tornado.plan.components;

import org.beehive.gpullama3.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.gpullama3.backend.tornado.layers.Activation;
import org.beehive.gpullama3.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.gpullama3.backend.tornado.layers.Qwen35FFNLayers;
import org.beehive.gpullama3.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.gpullama3.backend.tornado.layers.type.q8_0.LogitsQ8_0Layer;
import org.beehive.gpullama3.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.gpullama3.backend.tornado.scheduling.SchedulerType;
import org.beehive.gpullama3.inference.state.Qwen35State;
import org.beehive.gpullama3.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.qwen35.Qwen35Configuration;

/**
 * The {@code qwen35} single-token plan: an activation graph, one graph per trunk layer, and a
 * logits graph.
 *
 * <p>Not in a per-representation package, because there is no single representation to name it
 * after. The activation and logits graphs already dispatch on the tensor they read — the embedding
 * table and the vocabulary projection — and the layer graphs do the same per weight, so one set of
 * components serves the whole family rather than one per dtype.
 *
 * <p>Sequential prefill reuses the same layer graphs: prompt ingestion is the decode computation
 * with the logits graph skipped, and the recurrent state it advances is the same device buffer
 * decode continues from. Only the graph layer 0 consumes from differs.
 */
public class Qwen35PlanComponents implements PrefillDecodeForwardPlanComponents {

    private final Qwen35State state;
    private final Qwen35TornadoWeights weights;
    private final Qwen35Configuration config;
    private final SchedulerType schedulerType;

    public Qwen35PlanComponents(Qwen35State state, Model model) {
        this.state = state;
        this.config = (Qwen35Configuration) model.configuration();
        this.weights = (Qwen35TornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new Activation("activationUpdate", state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new Qwen35FFNLayers("qwen35FFN", state, weights, config, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new LogitsQ8_0Layer(
                "logits", state, weights, config, previousGraphId, schedulerType);
    }

    // ── Sequential prefill/decode ─────────────────────────────────────────────

    @Override
    public ActivationTaskGraph prefillDecodeActivation() {
        return new Activation("decodeActivation", state, weights, config);
    }

    @Override
    public TransformerLayerTaskGraphs prefillDecodeTransformerLayers() {
        return new Qwen35FFNLayers(
                "qwen35FFN", state, weights, config, schedulerType, "decodeActivation");
    }

    @Override
    public AbstractLogitsTaskGraph decodeLogits(String previousGraphId) {
        return singleTokenLogits(previousGraphId);
    }
}
