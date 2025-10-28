package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.model.Configuration;

/**
 * Represents the state of the Llama model during inference.
 * This class extends {@link State} to include model-specific functionalities
 * and configurations tailored for the Llama model.
 *
 * <p><b>Note 1:</b> LlamaState contains additional fields for TornadoVM wrappers
 * to enable GPU-accelerated processing of the model.</p>
 *
 * <p><b>Note 2:</b> This state implementation is also used for the Mistral model.</p>
 */
public final class LlamaState extends State {

    public LlamaState(Configuration config, int batchsize) {
        super(config, batchsize);
    }

    @Override
    protected StateFields createStateFields(Configuration config) {
        StateFieldAllocator allocator = new LlamaStateFieldAllocator(config, localSize);
        return allocator.allocateFields();
    }
}
