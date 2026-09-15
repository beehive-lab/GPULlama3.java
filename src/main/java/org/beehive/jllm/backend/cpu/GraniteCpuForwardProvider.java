package org.beehive.gpullama3.backend.cpu;

import org.beehive.gpullama3.inference.ForwardPass;
import org.beehive.gpullama3.runtime.model.ArchitectureId;

/** The host forward pass for {@code granite}. */
public final class GraniteCpuForwardProvider implements CpuForwardProvider {

    private static final ArchitectureId ARCHITECTURE = ArchitectureId.of("granite");

    @Override
    public ArchitectureId architecture() {
        return ARCHITECTURE;
    }

    @Override
    public ForwardPass create() {
        return (model, state, token, position) ->
                InferenceCore.forwardGranite(model, state, token, position);
    }
}
