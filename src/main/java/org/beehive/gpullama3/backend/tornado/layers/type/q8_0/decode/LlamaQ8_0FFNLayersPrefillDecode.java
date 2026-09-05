package org.beehive.gpullama3.backend.tornado.layers.type.q8_0.decode;

import org.beehive.gpullama3.backend.tornado.layers.type.q8_0.LlamaQ8_0FFNLayers;
import org.beehive.gpullama3.backend.tornado.scheduling.SchedulerType;
import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * Decode transformer-layer TaskGraphs for the single-token prefill/decode plan ({@link
 * org.beehive.gpullama3.backend.tornado.TornadoVMMasterPlanPrefillDecode}).
 *
 * <p>Layer 0 delegates to {@link LlamaQ8_0FFNLayers#configureLayerDataTransfers} which includes
 * {@code FIRST_EXECUTION} for {@code wrapKeyCache} and {@code wrapValueCache}, allocating the KV
 * cache on the very first forward pass. Layers 1+ use explicit predecessor names for all consumed
 * objects, required by TornadoVM's interpreter mode.
 */
public class LlamaQ8_0FFNLayersPrefillDecode extends LlamaQ8_0FFNLayers {

    // @formatter:off
    public LlamaQ8_0FFNLayersPrefillDecode(
            String taskGraph,
            LlamaState state,
            LlamaTornadoWeights weights,
            LlamaConfiguration config,
            SchedulerType schedulerType) {
        super(taskGraph, state, weights, config, schedulerType);
    }

    // @formatter:on

    @Override
    protected String predecessorGraphName(int layerIndex) {
        return (layerIndex == 0) ? "decodeActivation" : "layer_" + (layerIndex - 1);
    }

    // @formatter:off
    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph layer, int layerIndex) {
        if (layerIndex == 0) {
            return super.configureLayerDataTransfers(layer, 0);
        }
        String pred = "layer_" + (layerIndex - 1);
        layer.consumeFromDevice(
                pred,
                context,
                state.workspace.wrapXb,
                state.workspace.wrapXb2,
                state.workspace.wrapQ,
                state.workspace.wrapK,
                state.workspace.wrapV,
                state.workspace.wrapKeyCache,
                state.workspace.wrapValueCache,
                state.workspace.wrapAtt,
                state.workspace.wrapHb,
                state.workspace.positionHolder);
        layer.consumeFromDevice(pred, state.workspace.wrapBlockTable);
        return layer;
    }
    // @formatter:on
}
