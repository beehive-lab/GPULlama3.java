package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.phi3.Phi3Configuration;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Represents the state of the Phi3 model during inference.
 * This class extends {@link State} to include model-specific functionalities
 * and configurations tailored for the Phi3 model.
 *
 * <p><b>Note:</b> Phi3State has additional model-specific buffers for QKV processing
 * and FFN gate/up states.</p>
 */
public class Phi3State extends State {
    // Phi3-specific fields for QKV processing
    public FloatTensor qkv; // Combined QKV buffer: op_size = dim + 2 * (n_kv_heads * head_dim)

    // Phi3-specific fields for FFN gate/up processing
    public FloatTensor hbG; // Gate states buffer
    public FloatTensor hbU; // Up states buffer

    public FloatArray wrapQkv; // TornadoVM wrapper for QKV buffer
    public FloatArray wrapHbG; // TornadoVM wrapper for gate states
    public FloatArray wrapHbU; // TornadoVM wrapper for up states

    public Phi3State(Configuration config, int batchsize) {
        super(config, batchsize);

        // Initialize Phi3-specific fields
        Phi3Configuration phi3Config = (Phi3Configuration) config;

        // QKV buffer size: op_size = num_heads * head_dim + 2 * (num_key_value_heads * head_dim)
        int opSize = phi3Config.dim() + 2 * (phi3Config.numberOfKeyValueHeads() * phi3Config.headSize());
        this.qkv = ArrayFloatTensor.allocate(opSize);

        // FFN gate and up state buffers
        this.hbG = ArrayFloatTensor.allocate(phi3Config.hiddenDim());
        this.hbU = ArrayFloatTensor.allocate(phi3Config.hiddenDim());

        // TornadoVM wrappers for GPU acceleration
        this.wrapQkv = new FloatArray(opSize);
        this.wrapHbG = new FloatArray(phi3Config.hiddenDim());
        this.wrapHbU = new FloatArray(phi3Config.hiddenDim());
    }

    @Override
    protected StateFields createStateFields(Configuration config) {
        StateFieldAllocator allocator = new Phi3StateFieldAllocator(config, localSize);
        return allocator.allocateFields();
    }
}
