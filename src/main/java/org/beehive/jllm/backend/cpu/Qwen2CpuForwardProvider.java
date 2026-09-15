package org.beehive.gpullama3.backend.cpu;

import org.beehive.gpullama3.inference.ForwardPass;
import org.beehive.gpullama3.runtime.model.ArchitectureId;

/** The host forward pass for {@code qwen2}. */
public final class Qwen2CpuForwardProvider implements CpuForwardProvider {

    private static final ArchitectureId ARCHITECTURE = ArchitectureId.of("qwen2");

    @Override
    public ArchitectureId architecture() {
        return ARCHITECTURE;
    }

    @Override
    public ForwardPass create() {
        return (model, state, token, position) ->
                InferenceCore.forwardJavaQwen2(model, state, token, position);
    }
}
