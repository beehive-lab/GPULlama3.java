package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.phi3.Phi3Configuration;

/**
 * StateFieldAllocator for Phi3 models.
 *
 * <p>Phi3 has unique state requirements:</p>
 * <ul>
 *   <li>hb is twice hiddenDim (for combined gate/up buffer)</li>
 *   <li>Uses explicit kvDim calculation</li>
 *   <li>tempFFN uses hiddenDim instead of dim</li>
 * </ul>
 */
public class Phi3StateFieldAllocator extends StateFieldAllocator {

    private final Phi3Configuration phi3Config;
    private final int kvDim;

    public Phi3StateFieldAllocator(Configuration config, int localSize) {
        super(config, localSize);
        this.phi3Config = (Phi3Configuration) config;

        // Phi3-specific KV dimension calculation
        int dim = phi3Config.dim();
        int nHeads = phi3Config.numberOfHeads();
        int nKvHeads = phi3Config.numberOfKeyValueHeads();
        this.kvDim = (dim * nKvHeads) / nHeads;
    }

    @Override
    protected int getDimX() {
        return phi3Config.dim();
    }

    @Override
    protected int getDimXb() {
        return phi3Config.dim();
    }

    @Override
    protected int getDimXb2() {
        return phi3Config.dim();
    }

    @Override
    protected int getDimHb() {
        // Phi3 uses 2x hiddenDim for combined gate/up buffer
        return 2 * phi3Config.hiddenDim();
    }

    @Override
    protected int getDimHb2() {
        return phi3Config.hiddenDim();
    }

    @Override
    protected int getDimQ() {
        return phi3Config.dim();
    }

    @Override
    protected int getDimK() {
        return kvDim;
    }

    @Override
    protected int getDimV() {
        return kvDim;
    }

    @Override
    protected int getKvCacheDim() {
        return kvDim;
    }

    @Override
    protected int getTempLogitsReductionDim() {
        // Phi3 uses vocabularySize for tempLogits
        return phi3Config.vocabularySize();
    }
}
