package org.beehive.gpullama3.tornadovm.layers.type.fp16.decode;

import org.beehive.gpullama3.inference.state.Qwen3State;
import org.beehive.gpullama3.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.Qwen3FP16FFNLayers;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * Decode transformer-layer TaskGraphs for the single-token prefill/decode plan (Qwen3 FP16).
 *
 * <p>Layer 0 delegates to the base-class {@link Qwen3FP16FFNLayers#configureLayerDataTransfers}
 * which allocates wrapKeyCache/wrapValueCache with FIRST_EXECUTION. Layers 1+ consume all
 * live buffers from the explicit predecessor graph to satisfy TornadoVM interpreter mode.</p>
 *
 * <p>Note: Qwen3FP16FFNLayers does not use wrapXbFP16 in any kernel task, so it is
 * intentionally excluded from the consume list.</p>
 */
public class Qwen3FP16FFNLayersPrefillDecode extends Qwen3FP16FFNLayers {

    public Qwen3FP16FFNLayersPrefillDecode(String taskGraph, Qwen3State state,
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
        layer.consumeFromDevice(pred, qwen3State.wrapAttSplit);
        return layer;
    }
}
