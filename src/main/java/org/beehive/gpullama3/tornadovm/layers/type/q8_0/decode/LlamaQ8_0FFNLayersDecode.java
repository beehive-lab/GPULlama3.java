package org.beehive.gpullama3.tornadovm.layers.type.q8_0.decode;

import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.LlamaQ8_0FFNLayers;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Decode transformer-layer task graphs for the unified batched prefill-decode plan
 * ({@link org.beehive.gpullama3.tornadovm.TornadoVMMasterPlanBatchPrefillDecode}).
 *
 * <p>Layer 0 consumes the KV cache from device (passed through by the decode activation
 * graph, which relays it from the last batch prefill layer). No FIRST_EXECUTION allocation
 * for the KV cache — it was already allocated in the batch prefill phase.</p>
 */
public class LlamaQ8_0FFNLayersDecode extends LlamaQ8_0FFNLayers {

    public LlamaQ8_0FFNLayersDecode(String taskGraph, LlamaState state,
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
            layer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                    state.positionHolder, state.temp, state.tempFFN);
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.wrapXb, state.wrapXb2,
                    state.wrapQ, state.wrapK, state.wrapV,
                    state.wrapAtt, state.wrapHb);
            layer.consumeFromDevice("decodeActivation", state.wrapKeyCache, state.wrapValueCache);
        } else {
            String pred = "layer_" + (layerIndex - 1);
            layer.consumeFromDevice(pred,
                    context,
                    state.wrapXb, state.wrapXb2,
                    state.wrapQ, state.wrapK, state.wrapV,
                    state.wrapKeyCache, state.wrapValueCache,
                    state.wrapAtt, state.wrapHb,
                    state.positionHolder);
        }
        return layer;
    }
}
