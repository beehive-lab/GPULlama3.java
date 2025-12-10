package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.tensor.standard.FloatTensor;
import org.beehive.gpullama3.model.Configuration;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.*;

/**
 * Represents the base state structure used during LLM inference.
 * This class provides a common foundation for handling state-related data and functionalities
 * that can be extended by model-specific implementations.
 *
 * <p><b>Key Responsibilities:</b></p>
 * <ul>
 *   <li>Defines core structures to store and access model state data required for computation.</li>
 *   <li>Can be extended by model-specific state classes (e.g., {@link LlamaState}, {@link Qwen3State}).</li>
 * </ul>
 *
 * <p><b>Usage:</b> Extend `State` to implement model-specific state configurations
 * while reusing the common structure and functionality provided by this class.</p>
 *
 * <p><b>Note:</b> This class is designed to be generic and does not include any
 * model-specific behavior or fields. Those should be implemented in subclasses.</p>
 */
public abstract class State {

    // current wave of activations
    public final FloatTensor x;         // activation at current time stamp (dim,)
    public final FloatTensor xb;        // same, but inside a residual branch (dim,)
    public final FloatTensor xb2;       // an additional buffer just for convenience (dim,)
    public final FloatTensor hb;        // buffer for hidden dimension in the ffn (hidden_dim,)
    public final FloatTensor hb2;       // buffer for hidden dimension in the ffn (hidden_dim,)
    public final FloatTensor q;         // query (dim,)
    public final FloatTensor k;         // key (dim,)
    public final FloatTensor v;         // value (dim,)
    public final FloatTensor att;       // buffer for scores/attention values (n_heads, seq_len)
    public final FloatTensor logits;    // output logits
    public final int batchsize;

    // kv cache
    public final FloatTensor[] keyCache;   // (n_layer, seq_len, kv_dim)
    public final FloatTensor[] valueCache; // (n_layer, seq_len, kv_dim)

    // Wrappers for TornadoVM compatibility (FloatArray data structure for TornadoVM acceleration)
    // TornadoVM uses FloatArray for more efficient handling of data, particularly when running on GPU or other accelerators.
    public final FloatArray wrapLogits;     // FloatArray wrapper for the logits tensor, compatible with TornadoVM for GPU execution.
    public final FloatArray wrapXb;         // FloatArray wrapper for xb (residual branch activation), optimized for TornadoVM usage.
    public final FloatArray wrapXb2;        // FloatArray wrapper for xb2, another residual buffer to aid in computations with TornadoVM.
    public final FloatArray wrapHb;         // FloatArray wrapper for hb (hidden dimension buffer for FFN), optimized for TornadoVM.
    public final FloatArray wrapHb2;        // FloatArray wrapper for hb2, additional hidden buffer for FFN, for compatibility with TornadoVM.
    public final FloatArray wrapX;          // FloatArray wrapper for the current activation tensor, optimized for TornadoVM.
    public final FloatArray wrapQ;          // FloatArray wrapper for the query tensor, optimized for TornadoVM.
    public final FloatArray wrapK;          // FloatArray wrapper for the key tensor, optimized for TornadoVM.
    public final FloatArray wrapV;          // FloatArray wrapper for the value tensor, optimized for TornadoVM.
    public final FloatArray wrapAtt;        // FloatArray wrapper for the attention scores, optimized for TornadoVM.
    public final FloatArray wrapKeyCache;   // FloatArray wrapper for the key cache, optimized for TornadoVM.
    public final FloatArray wrapValueCache; // FloatArray wrapper for the value cache, optimized for TornadoVM.
    public final IntArray positionHolder;

    public TornadoNativeArray embeddingX;
    // store inter
    public int localSize;
    public FloatArray temp;         // Temporary buffer for intermediate calculations, size adjusted for local workgroup size.
    public FloatArray tempFFN;      // Temporary buffer for feed-forward network calculations, size adjusted for local workgroup size.
    public FloatArray tempLogits;   // Temporary buffer for logits calculations, size adjusted for local workgroup size.
    public int latestToken;         // Keeps track of the most recent token processed by the model. Useful for stateful or autoregressive models.

    /** last index in previous block */

    protected State(Configuration config, int batchsize) {
        this.batchsize = -1;
        this.latestToken = -1;
        this.localSize = 256;

        // Initialize all fields through the creation method
        StateFields fields = createStateFields(config);

        this.x = fields.x;
        this.xb = fields.xb;
        this.xb2 = fields.xb2;
        this.hb = fields.hb;
        this.hb2 = fields.hb2;
        this.q = fields.q;
        this.k = fields.k;
        this.v = fields.v;
        this.att = fields.att;
        this.logits = fields.logits;
        //int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        this.keyCache = fields.keyCache;
        this.valueCache = fields.valueCache;

        this.embeddingX = fields.embeddingX;
        this.wrapX = fields.wrapX;
        this.wrapXb = fields.wrapXb;
        this.wrapXb2 = fields.wrapXb2;
        this.wrapHb = fields.wrapHb;
        this.wrapHb2 = fields.wrapHb2;
        this.wrapLogits = fields.wrapLogits;
        this.wrapQ = fields.wrapQ;
        this.wrapK = fields.wrapK;
        this.wrapV = fields.wrapV;

        // dim vs kvdim
        this.wrapKeyCache = fields.wrapKeyCache;
        this.wrapValueCache = fields.wrapValueCache;
        this.wrapAtt = fields.wrapAtt;
        this.positionHolder = fields.positionHolder;

        // You need at least 9 elements: 1 for the final result + 8 for the workgroup partial sums
        this.temp = fields.temp;
        this.tempFFN = fields.tempFFN;
        this.tempLogits = fields.tempLogits;
    }

    // Abstract method - subclasses implement their specific allocation logic and sizes
    protected abstract StateFields createStateFields(Configuration config);

    // Helper class to hold all the state fields during construction
    protected static class StateFields {
        public FloatTensor x, xb, xb2, hb, hb2, q, k, v, att, logits;
        public FloatTensor[] keyCache, valueCache;
        public FloatArray wrapX, wrapXb, wrapXb2, wrapHb, wrapHb2, wrapLogits;
        public FloatArray wrapQ, wrapK, wrapV, wrapAtt, wrapKeyCache, wrapValueCache;
        public IntArray positionHolder;
        public FloatArray temp, tempFFN, tempLogits;
        public TornadoNativeArray embeddingX;

        public void createActivationFP16(int size) {
            this.embeddingX = new HalfFloatArray(size);
        }

        public void createActivationQ8_0(int size) {
            int blockSize = 32;
            int Q8_0_BLOCK_BYTES = 34; // 2 bytes scale + 32 bytes quants
            int blocksNeeded = (size + blockSize - 1) / blockSize;
            int q8BytesNeeded = blocksNeeded * Q8_0_BLOCK_BYTES;
            this.embeddingX = new ByteArray(q8BytesNeeded);
        }
    }

    @Override
    public State clone() throws CloneNotSupportedException {
        return (State) super.clone();
    }
}