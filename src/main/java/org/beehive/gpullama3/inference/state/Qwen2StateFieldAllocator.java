package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.qwen2.Qwen2Configuration;

/**
 * StateFieldAllocator for Qwen2 models.
 *
 * <p>Qwen2 uses different dimensions for K and V tensors compared to Llama/Mistral,
 * specifically using kvDim() for K and V instead of the full dim().</p>
 */
public class Qwen2StateFieldAllocator extends StateFieldAllocator {

    private final Qwen2Configuration qwen2Config;

    public Qwen2StateFieldAllocator(Configuration config, int localSize) {
        super(config, localSize);
        this.qwen2Config = (Qwen2Configuration) config;
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
        return qwen2Config.kvDim();
    }

    @Override
    protected int getDimV() {
        return qwen2Config.kvDim();
    }

    @Override
    protected int getKvCacheDim() {
        return qwen2Config.kvDim();
    }
}
