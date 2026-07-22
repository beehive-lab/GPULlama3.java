package org.beehive.gpullama3.tornadovm.layers.type.fp16.decode;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LogitsFP16Layer;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * Logits layer of the unified batched prefill-decode plan
 *  * ({@link org.beehive.gpullama3.tornadovm.TornadoVMMasterPlanBatchPrefillDecode}).
 *
 * <p>Extends {@link LogitsFP16Layer} with KV-cache pass-through so the device
 * pointers for {@code wrapKeyCache} and {@code wrapValueCache} survive the
 * logits → decode-activation boundary across decode tokens.</p>
 *
 * <p>In interpreter (non-CUDA-graph) mode, {@code updatePersistedObjectState()}
 * propagates device pointers from the predecessor graph's persisted set. After the
 * last decode token the predecessor of the next decode-activation graph is the
 * logits graph. Without the pass-through here, the KV-cache pointer is absent from
 * the logits persisted set, cleared to null, and the first decode layer crashes with
 * an NPE in {@code executeAlloc}.</p>
 *
 * <p>Bytecode order matters: {@code consumeFromDevice} must precede task declarations,
 * and {@code persistOnDevice} must follow {@code transferToHost}. The hooks in
 * {@link LogitsFP16Layer} guarantee this ordering.</p>
 */
public class LogitsFP16LayerDecode extends LogitsFP16Layer {

    public LogitsFP16LayerDecode(String name, State state, Weights weights, Configuration config,
            String lastTaskGraphID, SchedulerType schedulerType) {
        super(name, state, weights, config, lastTaskGraphID, schedulerType);
    }

    /** The KV cache objects the decode layers actually persist (FP16 when that path is active). */
    private Object keyCache() {
        return (State.USE_FP16_KV && state.wrapKeyCacheFP16 != null) ? state.wrapKeyCacheFP16 : state.wrapKeyCache;
    }

    private Object valueCache() {
        return (State.USE_FP16_KV && state.wrapValueCacheFP16 != null) ? state.wrapValueCacheFP16 : state.wrapValueCache;
    }

    /**
     * Prepends {@code consumeFromDevice(lastTaskGraphID, keyCache, valueCache)} before all tasks.
     *
     * <p>Must use the named-source form so that {@code updatePersistedObjectState()} adds the KV cache
     * to the source-keyed map. Without the source name, the fallback in {@code updatePersistedObjectState}
     * uses the current graph's general persisted list, which causes the XPUBuffer from the predecessor
     * (last decode layer) to never be propagated into the logits graph's device state.</p>
     */
    @Override
    protected void configureAdditionalConsumes(TaskGraph logits) {
        logits.consumeFromDevice(lastTaskGraphID, keyCache(), valueCache());
    }

    /** Appends {@code persistOnDevice(keyCache, valueCache)} after {@code transferToHost}. */
    @Override
    protected void configureAdditionalPersists(TaskGraph logits) {
        logits.persistOnDevice(keyCache(), valueCache());
    }
}
