package org.beehive.gpullama3.backend.cpu;

import org.beehive.gpullama3.inference.ForwardPass;
import org.beehive.gpullama3.runtime.model.ArchitectureId;

/** The host forward pass for {@code qwen35}. */
public final class Qwen35CpuForwardProvider implements CpuForwardProvider {

    private static final ArchitectureId ARCHITECTURE = ArchitectureId.of("qwen35");

    @Override
    public ArchitectureId architecture() {
        return ARCHITECTURE;
    }

    @Override
    public ForwardPass create() {
        return Qwen35Forward::forward;
    }
}
