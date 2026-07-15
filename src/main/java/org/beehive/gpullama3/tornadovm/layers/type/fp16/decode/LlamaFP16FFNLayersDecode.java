package org.beehive.gpullama3.tornadovm.layers.type.fp16.decode;

import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LlamaFP16FFNLayers;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Decode transformer-layer TaskGraphs of the unified batched prefill-decode plan
 * ({@link org.beehive.gpullama3.tornadovm.TornadoVMMasterPlanBatchPrefillDecode}).
 *
 * <p>Overrides data-transfer declarations so that all cross-graph boundaries use
 * the explicit-source form of {@code consumeFromDevice}.  The no-arg form (used by
 * the base class) passes the <em>current</em> graph's own name as the source key.
 * In CUDA-graph mode this is harmless (device pointers are frozen at capture time),
 * but in interpreter mode {@code updatePersistedObjectState} looks up the
 * <em>predecessor's</em> name, so the lookup always misses and the XPUBuffer is
 * never propagated — causing either a null-pointer crash or a silent re-upload
 * from host (zeros), corrupting the hidden state and KV cache.</p>
 *
 */
public class LlamaFP16FFNLayersDecode extends LlamaFP16FFNLayers {
    public LlamaFP16FFNLayersDecode(String taskGraph, LlamaState state,
                                    LlamaTornadoWeights weights, LlamaConfiguration config,
                                    SchedulerType schedulerType) {
        super(taskGraph, state, weights, config, schedulerType);
    }

    /**
     * Supplies the correct predecessor graph name for {@code consumeFromDevice(wrapX)}.
     *
     * <p>Layer 0 receives {@code wrapX} from the decode activation graph;
     * layers 1+ receive it from the previous decode layer.
     * Must match the {@code TaskGraph} names used in
     * {@code buildDecodeActivationGraph()} and {@code createFFNLayerTaskGraph()}.</p>
     */
    /**
     * The prefill/decode graph variants share the FP32 KV cache with the batch-prefill layers,
     * so the FP16 KV cache path (standard single-token mode only) is disabled here.
     */
    @Override
    protected boolean useFp16KVCache() {
        return false;
    }

    @Override
    protected String predecessorGraphName(int layerIndex) {
        return (layerIndex == 0) ? "decodeActivation" : "layer_" + (layerIndex - 1);
    }

    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph layer, int layerIndex) {
        if (layerIndex == 0) {
            // Same as parent layer 0, but wrapKeyCache/wrapValueCache come from device
            // (passed through by the decode activation graph, which relays them from
            // the last batch prefill layer).  No FIRST_EXECUTION for KV cache here.
            layer.transferToDevice(DataTransferMode.EVERY_EXECUTION,
                    state.positionHolder, state.temp, state.tempFFN);
            layer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    context,
                    state.wrapXb, state.wrapXb2,
                    state.wrapQ, state.wrapK, state.wrapV,
                    state.wrapAtt, state.wrapHb, state.wrapXbFP16);
            // Explicit source — must match the TaskGraph name in buildDecodeActivationGraph().
            layer.consumeFromDevice("decodeActivation", state.wrapKeyCache, state.wrapValueCache);
        } else {
            // Layers 1+: use explicit predecessor name for ALL consumed objects.
            // Calling super here would use the no-arg form (source key = own graph name),
            // which silently fails in interpreter mode and causes re-upload from host.
            String pred = "layer_" + (layerIndex - 1);
            layer.consumeFromDevice(pred,
                    context,
                    state.wrapXb, state.wrapXb2,
                    state.wrapQ, state.wrapK, state.wrapV,
                    state.wrapKeyCache, state.wrapValueCache,
                    state.wrapAtt, state.wrapHb,
                    state.positionHolder, state.wrapXbFP16);
        }
        return layer;
    }
}
