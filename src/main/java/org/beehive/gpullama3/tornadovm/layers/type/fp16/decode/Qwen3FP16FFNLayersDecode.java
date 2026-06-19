package org.beehive.gpullama3.tornadovm.layers.type.fp16.decode;

import org.beehive.gpullama3.inference.state.Qwen3State;
import org.beehive.gpullama3.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.Qwen3FP16FFNLayers;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Decode transformer-layer TaskGraphs for the unified batched prefill-decode plan (Qwen3 FP16).
 *
 * <p>Layer 0: KV cache is consumed from "decodeActivation" (already allocated by the batch
 * prefill phase). Working buffers get FIRST_EXECUTION allocation. Layers 1+: all consumed
 * objects use the explicit predecessor name to satisfy TornadoVM interpreter mode.</p>
 *
 * <p>Qwen3FP16FFNLayers does not use wrapXbFP16 in any task, so it is excluded.</p>
 */
public class Qwen3FP16FFNLayersDecode extends Qwen3FP16FFNLayers {

    public Qwen3FP16FFNLayersDecode(String taskGraph, Qwen3State state,
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
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    context,
                    qwen3State.wrapXb, qwen3State.wrapXb2,
                    qwen3State.wrapQ, qwen3State.wrapK, qwen3State.wrapV,
                    qwen3State.wrapAtt, qwen3State.wrapHb);
            if (useSplitKVAttention) {
                layer.transferToDevice(DataTransferMode.FIRST_EXECUTION, qwen3State.wrapAttSplit);
            }
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
            if (useSplitKVAttention) {
                layer.consumeFromDevice(pred, qwen3State.wrapAttSplit);
            }
        }
        return layer;
    }
}
