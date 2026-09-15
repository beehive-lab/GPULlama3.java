package org.beehive.gpullama3.backend.cpu;

import org.beehive.gpullama3.inference.ForwardPass;
import org.beehive.gpullama3.runtime.model.ArchitectureId;

/** The host forward pass for {@code qwen2-moe}. */
public final class Qwen2MoECpuForwardProvider implements CpuForwardProvider {

    private static final ArchitectureId ARCHITECTURE = ArchitectureId.of("qwen2-moe");

    @Override
    public ArchitectureId architecture() {
        return ARCHITECTURE;
    }

    @Override
    public ForwardPass create() {
        return (model, state, token, position) ->
                InferenceCore.forwardJavaQwen2MoE(model, state, token, position);
    }
}
