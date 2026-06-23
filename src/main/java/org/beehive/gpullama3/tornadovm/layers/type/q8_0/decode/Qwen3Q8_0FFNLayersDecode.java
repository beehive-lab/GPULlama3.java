package org.beehive.gpullama3.tornadovm.layers.type.q8_0.decode;

import org.beehive.gpullama3.inference.state.Qwen3State;
import org.beehive.gpullama3.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.Qwen3Q8_0FFNLayers;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Decode transformer-layer TaskGraphs for the unified batched prefill-decode plan (Qwen3 Q8_0).
 *
 * <p>Layer 0: KV cache consumed from "decodeActivation" (already allocated by batch prefill).
 * Layers 1+: all consumed objects use explicit predecessor name for interpreter mode.</p>
 */
public class Qwen3Q8_0FFNLayersDecode extends Qwen3Q8_0FFNLayers {

    public Qwen3Q8_0FFNLayersDecode(String taskGraph, Qwen3State state,
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
            layer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                    qwen3State.positionHolder, qwen3State.temp, qwen3State.tempFFN);
            layer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                    qwen3State.tempQcur, qwen3State.tempKcur);
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    context,
                    qwen3State.wrapXb, qwen3State.wrapXb2,
                    qwen3State.wrapQ, qwen3State.wrapK, qwen3State.wrapV,
                    qwen3State.wrapAtt, qwen3State.wrapHb);
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, qwen3State.wrapAttSplit);
            // KV cache already allocated by batch prefill; relay from decode activation graph.
            layer.consumeFromDevice("decodeActivation",
                    qwen3State.wrapKeyCache, qwen3State.wrapValueCache);
        } else {
            String pred = "layer_" + (layerIndex - 1);
            layer.consumeFromDevice(pred,
                    context,
                    qwen3State.wrapXb, qwen3State.wrapXb2,
                    qwen3State.wrapQ, qwen3State.wrapK, qwen3State.wrapV,
                    qwen3State.wrapKeyCache, qwen3State.wrapValueCache,
                    qwen3State.wrapAtt, qwen3State.wrapHb,
                    qwen3State.positionHolder,
                    qwen3State.temp, qwen3State.tempFFN);
            layer.consumeFromDevice(qwen3State.tempQcur, qwen3State.tempKcur);
            layer.consumeFromDevice(pred, qwen3State.wrapAttSplit);
        }
        return layer;
    }
}
