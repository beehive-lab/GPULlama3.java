package org.beehive.gpullama3.tornadovm.plan.components.fp16;

import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import org.beehive.gpullama3.tornadovm.layers.AbstractLogitsTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.ActivationTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.layers.TransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LlamaFP16FFNLayers;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LogitsFP16Layer;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.decode.LlamaFP16FFNLayersDecode;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.decode.LlamaFP16FFNLayersPrefillDecode;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.decode.LogitsFP16LayerDecode;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.prefill.LlamaFP16LayersBatchPrefill;
import org.beehive.gpullama3.tornadovm.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.gpullama3.tornadovm.plan.components.activation.BatchDecodeActivation;
import org.beehive.gpullama3.tornadovm.plan.components.activation.BatchPrefillActivation;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerDetectionService;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;


/**
 * {@link BatchPrefillDecodeForwardPlanComponents} for Llama + FP16.
 *
 * <p>Provides all layer objects and the FP16 embedding preparer (raw half-float byte copy).</p>
 */
public class LlamaFP16PlanComponents implements BatchPrefillDecodeForwardPlanComponents {

    private final LlamaState state;
    private final LlamaTornadoWeights weights;
    private final LlamaConfiguration config;
    private final SchedulerType schedulerType;

    public LlamaFP16PlanComponents(LlamaState state, Model model) {
        this.state = state;
        this.config = (LlamaConfiguration) model.configuration();
        this.weights = (LlamaTornadoWeights) model.weights();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
    }

    // ── Activations ───────────────────────────────────────────────────────────

    @Override public ActivationTaskGraph standardActivation() {
        return new Activation("activationUpdate", state, weights, config);
    }

    @Override public ActivationTaskGraph decodeActivation() {
        return new Activation("decodeActivation", state, weights, config);
    }

    @Override public ActivationTaskGraph batchPrefillActivation(int batchSize) {
        return new BatchPrefillActivation(state, config, batchSize, false);
    }

    @Override public ActivationTaskGraph batchDecodeActivation(String lastBatchLayerId) {
        return new BatchDecodeActivation(state, config, lastBatchLayerId, false);
    }

    // ── Transformer layer task graphs ──────────────────────────────────────────────────────

    @Override public TransformerLayerTaskGraphs standardLayers() {
        return new LlamaFP16FFNLayers("llamaFFN", state, weights, config, schedulerType);
    }

    @Override public TransformerLayerTaskGraphs prefillDecodeLayers() {
        return new LlamaFP16FFNLayersPrefillDecode("decode", state, weights, config, schedulerType);
    }

    @Override public TransformerLayerTaskGraphs batchDecodeLayers() {
        return new LlamaFP16FFNLayersDecode("decode", state, weights, config, schedulerType);
    }

    @Override public BatchPrefillTransformerLayerTaskGraphs batchPrefillLayers(int batchSize) {
        return new LlamaFP16LayersBatchPrefill(state, weights, config, batchSize);
    }

    // ── Logits layers ─────────────────────────────────────────────────────────

    @Override public AbstractLogitsTaskGraph standardLogits(String previousGraphId) {
        return new LogitsFP16Layer("logits", state, weights, config, previousGraphId, schedulerType);
    }

    @Override public AbstractLogitsTaskGraph decodeLogits(String previousGraphId) {
        return new LogitsFP16LayerDecode("logits", state, weights, config, previousGraphId, schedulerType);
    }

}
