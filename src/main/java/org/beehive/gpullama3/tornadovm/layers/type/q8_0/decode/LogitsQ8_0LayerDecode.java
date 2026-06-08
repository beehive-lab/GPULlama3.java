package org.beehive.gpullama3.tornadovm.layers.type.q8_0.decode;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.LogitsQ8_0Layer;
import uk.ac.manchester.tornado.api.TaskGraph;

/**
 * Logits layer for the unified batched prefill-decode plan (Q8_0).
 *
 * <p>Extends {@link LogitsQ8_0Layer} with KV-cache pass-through so the device pointers for
 * {@code wrapKeyCache} and {@code wrapValueCache} survive the logits → decode-activation
 * boundary between decode tokens. Without the pass-through, the KV-cache pointer is absent
 * from the logits persisted set, cleared to null, and the first decode layer crashes with
 * an NPE in {@code executeAlloc}.</p>
 */
public class LogitsQ8_0LayerDecode extends LogitsQ8_0Layer {

    // @formatter:off
    public LogitsQ8_0LayerDecode(String name, State state, Weights weights, Configuration config,
            String lastTaskGraphID, SchedulerType schedulerType) {
        super(name, state, weights, config, lastTaskGraphID, schedulerType);
    }
    // @formatter:on

    @Override
    protected void configureAdditionalConsumes(TaskGraph logits) {
        logits.consumeFromDevice(lastTaskGraphID, state.wrapKeyCache, state.wrapValueCache);
    }

    @Override
    protected void configureAdditionalPersists(TaskGraph logits) {
        logits.persistOnDevice(state.wrapKeyCache, state.wrapValueCache);
    }
}
