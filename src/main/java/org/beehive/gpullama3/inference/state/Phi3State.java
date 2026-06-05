package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.tensor.standard.ArrayFloatTensor;
import org.beehive.gpullama3.tensor.standard.FloatTensor;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.phi3.Phi3Configuration;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.stream.Stream;

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
        StateFields fields = new StateFields();

        Phi3Configuration phi3Config = (Phi3Configuration) config;

        // Phi3-specific dimensions
        int dim = phi3Config.dim();
        int headSize = phi3Config.headSize();
        int nHeads = phi3Config.numberOfHeads();
        int nKvHeads = phi3Config.numberOfKeyValueHeads();
        int kvDim = (dim * nKvHeads) / nHeads;
        int hiddenDim = phi3Config.hiddenDim();
        int contextLength = phi3Config.contextLength();
        int vocabSize = phi3Config.vocabularySize();
        int nLayers = phi3Config.numberOfLayers();

        // Standard tensor allocations for Phi3
        fields.x = ArrayFloatTensor.allocate(dim);
        fields.xb = ArrayFloatTensor.allocate(dim); // Used for attention output
        fields.xb2 = ArrayFloatTensor.allocate(dim); // Used for residual connections
        fields.hb = ArrayFloatTensor.allocate(2 * hiddenDim); // Combined gate/up buffer
        fields.hb2 = ArrayFloatTensor.allocate(hiddenDim); // FFN output buffer

        // Attention-related tensors
        fields.q = ArrayFloatTensor.allocate(dim); // Query states
        fields.k = ArrayFloatTensor.allocate(kvDim); // Key states
        fields.v = ArrayFloatTensor.allocate(kvDim); // Value states
        fields.att = ArrayFloatTensor.allocate(nHeads, contextLength); // Attention scores

        // Output logits
        fields.logits = ArrayFloatTensor.allocate(vocabSize);

        // Key-value cache with Phi3 dimensions
        fields.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(contextLength, kvDim)).limit(nLayers).toArray(FloatTensor[]::new);
        fields.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(contextLength, kvDim)).limit(nLayers).toArray(FloatTensor[]::new);

        // TornadoVM wrapper arrays for GPU acceleration
        switch (config.quantization()) {
            case "FP16" -> fields.createActivationFP16(config.dim());
            case "Q8_0" -> fields.createActivationQ8_0(config.dim());
            default -> throw new UnsupportedOperationException("Unsupported quantization format: " + config.quantization());
        }
        fields.wrapX = new FloatArray(dim);
        fields.wrapXb = new FloatArray(dim);
        fields.wrapXFP16 = new HalfFloatArray(dim);
        fields.wrapXbFP16 = new HalfFloatArray(dim);
        fields.wrapXb2 = new FloatArray(dim);
        fields.wrapHb = new FloatArray(2 * hiddenDim);
        fields.wrapHb2 = new FloatArray(hiddenDim);
        fields.wrapLogits = new FloatArray(vocabSize);
        fields.wrapQ = new FloatArray(dim);
        fields.wrapK = new FloatArray(kvDim);
        fields.wrapV = new FloatArray(kvDim);

        // KV cache wrappers
        fields.wrapKeyCache = new FloatArray(contextLength * kvDim * nLayers);
        fields.wrapValueCache = new FloatArray(contextLength * kvDim * nLayers);
        fields.wrapKeyCache.init(0.f);
        fields.wrapValueCache.init(0.f);

        // Attention wrapper
        fields.wrapAtt = new FloatArray(nHeads * contextLength);

        // Position holder for GPU operations
        fields.positionHolder = new IntArray(1);

        // Temporary arrays for reductions and operations
        fields.temp = new FloatArray(1 + ((dim + localSize - 1) / localSize));
        fields.tempFFN = new FloatArray(1 + ((hiddenDim + localSize - 1) / localSize));
        fields.tempLogits = new FloatArray(1 + ((vocabSize + localSize - 1) / localSize));

        return fields;
    }
}
