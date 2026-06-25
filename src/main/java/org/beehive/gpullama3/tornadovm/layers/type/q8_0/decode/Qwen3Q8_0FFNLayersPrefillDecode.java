package org.beehive.gpullama3.tornadovm.layers.type.q8_0.decode;

import org.beehive.gpullama3.inference.state.Qwen3State;
import org.beehive.gpullama3.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.Qwen3Q8_0FFNLayers;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * Decode transformer-layer TaskGraphs for the single-token prefill/decode plan (Qwen3 Q8_0).
 *
 * <p>Layer 0 delegates to the base-class which allocates wrapKeyCache/wrapValueCache with
 * FIRST_EXECUTION. Layers 1+ consume all live buffers from the explicit predecessor graph.</p>
 */
public class Qwen3Q8_0FFNLayersPrefillDecode extends Qwen3Q8_0FFNLayers {

    public Qwen3Q8_0FFNLayersPrefillDecode(String taskGraph, Qwen3State state,
                                           Qwen3TornadoWeights weights, Qwen3Configuration config,
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
                qwen3State.wrapXb, qwen3State.wrapXb2,
                qwen3State.wrapQ, qwen3State.wrapK, qwen3State.wrapV,
                qwen3State.wrapKeyCache, qwen3State.wrapValueCache,
                qwen3State.wrapAtt, qwen3State.wrapHb,
                qwen3State.positionHolder);
        layer.consumeFromDevice(qwen3State.tempQcur, qwen3State.tempKcur);
        layer.consumeFromDevice(pred, qwen3State.wrapAttSplit);
        return layer;
    }
}
