package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Represents the state of the Qwen3 model during inference.
 * This class extends {@link State} to include model-specific functionalities
 * and configurations tailored for the Qwen3 model.
 *
 * <p><b>Note 1:</b> Qwen3State contains additional fields for TornadoVM wrappers
 * to enable GPU-accelerated processing of the model.</p>
 *
 */
public final class Qwen3State extends State {

    // Qwen3 specific fields
    // Temporary buffers for intermediate calculations.
    public FloatArray tempQcur;
    public FloatArray tempKcur;

    public Qwen3State(Configuration config, int batchsize) {
        super(config, batchsize);
        // Initialize Qwen3-specific fields
        Qwen3Configuration qwen3config = (Qwen3Configuration) config;
        int nEmbdHead = qwen3config.numberOfHeads();
        this.tempQcur = new FloatArray(nEmbdHead);
        this.tempKcur = new FloatArray(nEmbdHead);
    }

    @Override
    protected StateFields createStateFields(Configuration config) {
        StateFieldAllocator allocator = new Qwen3StateFieldAllocator(config, localSize);
        return allocator.allocateFields();
    }
}
