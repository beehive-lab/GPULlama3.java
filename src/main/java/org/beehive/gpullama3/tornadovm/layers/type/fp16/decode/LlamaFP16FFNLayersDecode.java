package org.beehive.gpullama3.tornadovm.layers.type.fp16.decode;

import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import org.beehive.gpullama3.tornadovm.layerplanner.strategy.SchedulerType;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LlamaFP16FFNLayers;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Identical to {@link LlamaFP16FFNLayers} except decode layer 0 uses
 * {@code consumeFromDevice} for the KV cache instead of {@code FIRST_EXECUTION}.
 *
 * <p>This ensures decode layer 0 receives the KV-cache device pointer that was
 * persisted by the last batch prefill layer and passed through the decode
 * activation graph.</p>
 */
public class LlamaFP16FFNLayersDecode extends LlamaFP16FFNLayers {
    public LlamaFP16FFNLayersDecode(String taskGraph, LlamaState state,
                                    LlamaTornadoWeights weights, LlamaConfiguration config,
                                    SchedulerType schedulerType) {
        super(taskGraph, state, weights, config, schedulerType);
    }

    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph layer, int layerIndex) {
        if (layerIndex == 0) {
            // Same as parent layer 0 BUT wrapKeyCache/wrapValueCache come
            // from device (passed through by the decode activation graph).
            layer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                    state.positionHolder, state.temp, state.tempFFN);
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.wrapXb, state.wrapXb2,
                    state.wrapQ, state.wrapK, state.wrapV,
                    state.wrapAtt, state.wrapHb, state.wrapXbFP16);
            // KV cache: consume from device (device pointer supplied by
            // decode activation's pass-through from last batch layer).
            layer.consumeFromDevice(state.wrapKeyCache, state.wrapValueCache);
        } else {
            // Identical to parent for layers 1+ (already uses consumeFromDevice).
            return super.configureLayerDataTransfers(layer, layerIndex);
        }
        return layer;
    }
}
