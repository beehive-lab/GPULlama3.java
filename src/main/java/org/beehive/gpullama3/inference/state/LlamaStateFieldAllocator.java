package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.model.Configuration;

/**
 * StateFieldAllocator for Llama and Mistral models.
 *
 * <p>Both Llama and Mistral use identical state dimensions and allocation patterns,
 * so they share the same allocator implementation.</p>
 */
public class LlamaStateFieldAllocator extends StateFieldAllocator {

    public LlamaStateFieldAllocator(Configuration config, int localSize) {
        super(config, localSize);
    }

    @Override
    protected int getDimX() {
        return config.dim();
    }

    @Override
    protected int getDimXb() {
        return config.dim();
    }

    @Override
    protected int getDimXb2() {
        return config.dim();
    }

    @Override
    protected int getDimHb() {
        return config.hiddenDim();
    }

    @Override
    protected int getDimHb2() {
        return config.hiddenDim();
    }

    @Override
    protected int getDimQ() {
        return config.dim();
    }

    @Override
    protected int getDimK() {
        return config.dim();
    }

    @Override
    protected int getDimV() {
        return config.dim();
    }

    @Override
    protected int getKvCacheDim() {
        return (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
    }
}
