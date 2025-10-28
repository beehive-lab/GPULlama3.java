package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;

/**
 * StateFieldAllocator for Qwen3 models.
 *
 * <p>Qwen3 has unique dimension calculations based on separate key and value head dimensions.
 * It uses nEmbdHeadK and nEmbdHeadV for calculating the actual dimensions of Q, K, V tensors.</p>
 */
public class Qwen3StateFieldAllocator extends StateFieldAllocator {

    private final Qwen3Configuration qwen3Config;
    private final int nEmbdHeadK;
    private final int nEmbdKGqa;
    private final int nEmbdGqa;

    public Qwen3StateFieldAllocator(Configuration config, int localSize) {
        super(config, localSize);
        this.qwen3Config = (Qwen3Configuration) config;

        // Qwen3-specific dimension calculations
        int nHeadKv = qwen3Config.numberOfKeyValueHeads();
        this.nEmbdHeadK = qwen3Config.numberOfHeadsKey();
        this.nEmbdKGqa = nEmbdHeadK * nHeadKv;

        int nEmbdHeadV = qwen3Config.numberOfHeadsValue();
        int nEmbdVGqa = nEmbdHeadV * nHeadKv;
        this.nEmbdGqa = nEmbdVGqa;
    }

    @Override
    protected int getDimX() {
        return config.dim();
    }

    @Override
    protected int getDimXb() {
        return nEmbdHeadK * config.numberOfHeads();
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
        return nEmbdHeadK * config.numberOfHeads();
    }

    @Override
    protected int getDimK() {
        return nEmbdKGqa;
    }

    @Override
    protected int getDimV() {
        return nEmbdKGqa;
    }

    @Override
    protected int getKvCacheDim() {
        return nEmbdGqa;
    }
}
