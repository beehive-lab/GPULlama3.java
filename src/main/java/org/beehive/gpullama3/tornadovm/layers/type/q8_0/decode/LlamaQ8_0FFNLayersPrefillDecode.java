package org.beehive.gpullama3.tornadovm.layers.type.q8_0.decode;

import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.LlamaQ8_0FFNLayers;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * Decode transformer-layer task graphs for the single-token prefill/decode plan
 * ({@link org.beehive.gpullama3.tornadovm.TornadoVMMasterPlanPrefillDecode}).
 *
 * <p>Layer 0 delegates to {@link LlamaQ8_0FFNLayers#configureLayerDataTransfers} which
 * includes {@code FIRST_EXECUTION} for {@code wrapKeyCache} and {@code wrapValueCache},
 * allocating the KV cache on the very first forward pass. Layers 1+ use explicit
 * predecessor names for all consumed objects, required by TornadoVM's interpreter mode.</p>
 */
public class LlamaQ8_0FFNLayersPrefillDecode extends LlamaQ8_0FFNLayers {

    public LlamaQ8_0FFNLayersPrefillDecode(String taskGraph, LlamaState state,
                                            LlamaTornadoWeights weights, LlamaConfiguration config,
                                            SchedulerType schedulerType) {
        super(taskGraph, state, weights, config, schedulerType);
    }

    @Override
    protected String predecessorGraphName(int layerIndex) {
        return (layerIndex == 0) ? "decodeActivation" : "layer_" + (layerIndex - 1);
    }

    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph layer, int layerIndex) {
        if (layerIndex == 0) {
            return super.configureLayerDataTransfers(layer, 0);
        }
        String pred = "layer_" + (layerIndex - 1);
        layer.consumeFromDevice(pred,
                context,
                state.wrapXb, state.wrapXb2,
                state.wrapQ, state.wrapK, state.wrapV,
                state.wrapKeyCache, state.wrapValueCache,
                state.wrapAtt, state.wrapHb,
                state.positionHolder);
        return layer;
    }
}
