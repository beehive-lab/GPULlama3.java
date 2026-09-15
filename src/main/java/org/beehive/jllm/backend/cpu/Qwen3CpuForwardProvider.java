package org.beehive.gpullama3.backend.cpu;

import org.beehive.gpullama3.inference.ForwardPass;
import org.beehive.gpullama3.runtime.model.ArchitectureId;

/** The host forward pass for {@code qwen3}. */
public final class Qwen3CpuForwardProvider implements CpuForwardProvider {

    private static final ArchitectureId ARCHITECTURE = ArchitectureId.of("qwen3");

    @Override
    public ArchitectureId architecture() {
        return ARCHITECTURE;
    }

    @Override
    public ForwardPass create() {
        return (model, state, token, position) ->
                InferenceCore.forwardJavaQwen3(model, state, token, position);
    }
}
