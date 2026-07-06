package org.beehive.gpullama3.tornadovm.plan.components.fp16;

import org.beehive.gpullama3.inference.state.Qwen3State;
import org.beehive.gpullama3.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
import org.beehive.gpullama3.tornadovm.layers.AbstractLogitsTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.ActivationTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.layers.TransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LogitsFP16Layer;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.Qwen3FP16FFNLayers;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.decode.LogitsFP16LayerDecode;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.decode.Qwen3FP16FFNLayersDecode;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.decode.Qwen3FP16FFNLayersPrefillDecode;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.prefill.Qwen3FP16LayersBatchPrefill;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.prefill.Qwen3FP16LayersBatchPrefillGeneric;
import org.beehive.gpullama3.tornadovm.TensorCoreSupport;
import org.beehive.gpullama3.tornadovm.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.gpullama3.tornadovm.plan.components.activation.BatchDecodeActivation;
import org.beehive.gpullama3.tornadovm.plan.components.activation.BatchPrefillActivation;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerDetectionService;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;

public class Qwen3FP16PlanComponents implements BatchPrefillDecodeForwardPlanComponents {

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

    // ── Activations ───────────────────────────────────────────────────────────

    @Override
    public ActivationTaskGraph singleTokenActivation() {
        return new Activation("activationUpdate", state, weights, config);
    }

    @Override
    public ActivationTaskGraph prefillDecodeActivation() {
        return new Activation("decodeActivation", state, weights, config);
    }

    @Override
    public ActivationTaskGraph batchPrefillActivation(int batchSize) {
        return new BatchPrefillActivation(state, config, batchSize, false);
    }

    @Override
    public ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId) {
        return new BatchDecodeActivation(state, config, lastBatchLayerId, false);
    }

    // ── Transformer layer TaskGraphs ──────────────────────────────────────────

    @Override
    public TransformerLayerTaskGraphs singleTokenTransformerLayers() {
        return new Qwen3FP16FFNLayers("qwen3FFN", state, weights, config, schedulerType);
    }

    @Override
    public TransformerLayerTaskGraphs prefillDecodeTransformerLayers() {
        return new Qwen3FP16FFNLayersPrefillDecode("decode", state, weights, config, schedulerType);
    }

    @Override
    public TransformerLayerTaskGraphs batchDecodeTransformerLayers() {
        return new Qwen3FP16FFNLayersDecode("decode", state, weights, config, schedulerType);
    }

    @Override
    public BatchPrefillTransformerLayerTaskGraphs batchPrefillTransformerLayers(int batchSize) {
        if (TensorCoreSupport.isTensorCoreCapableBackend()) {
            return new Qwen3FP16LayersBatchPrefill(state, weights, config, batchSize);
        }
        return new Qwen3FP16LayersBatchPrefillGeneric(state, weights, config, batchSize);
    }

    // ── Logits layers ─────────────────────────────────────────────────────────

    @Override
    public AbstractLogitsTaskGraph singleTokenLogits(String previousGraphId) {
        return new LogitsFP16Layer("logits", state, weights, config, previousGraphId, schedulerType);
    }

    @Override
    public AbstractLogitsTaskGraph decodeLogits(String previousGraphId) {
        return new LogitsFP16LayerDecode("logits", state, weights, config, previousGraphId, schedulerType);
    }
}
