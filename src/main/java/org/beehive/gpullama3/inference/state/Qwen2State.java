package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.model.Configuration;

/**
 * Represents the state of the Qwen2 model during inference.
 * This class extends {@link State} to include model-specific functionalities
 * and configurations tailored for the Qwen2 model.
 *
 * <p><b>Note:</b> Qwen2State uses a smaller localSize (32) compared to other models (256).</p>
 */
public class Qwen2State extends State {

    public Qwen2State(Configuration config, int batchsize) {
        super(config, batchsize);
        this.localSize = 32;
    }

    @Override
    protected StateFields createStateFields(Configuration config) {
        StateFieldAllocator allocator = new Qwen2StateFieldAllocator(config, localSize);
        return allocator.allocateFields();
    }
}
