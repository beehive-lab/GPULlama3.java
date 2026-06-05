package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.tensor.standard.ArrayFloatTensor;
import org.beehive.gpullama3.tensor.standard.FloatTensor;
import org.beehive.gpullama3.model.Configuration;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.stream.Stream;

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

    // ── Batch-prefill GPU buffers ─────────────────────────────────────────────
    // Allocated when llama.prefillBatchSize > 1; null otherwise.
    // Layout: flat [B × stride], element [b][i] at index b*stride + i.
    public final HalfFloatArray embeddingXBatch;  // B × dim  (FP16 input)
    public final FloatArray     wrapXBatch;        // B × dim  (live activations)
    public final HalfFloatArray wrapXbFP16Batch;  // B × dim  (RMSNorm output, FP16)
    public final FloatArray     wrapQBatch;        // B × dim
    public final FloatArray     wrapKBatch;        // B × kvDim
    public final FloatArray     wrapVBatch;        // B × kvDim
    public final FloatArray     wrapXbBatch;       // B × dim  (attention output)
    public final FloatArray     wrapHbBatch;       // B × hiddenDim
    public final FloatArray     attnScaleBatch;    // B        (per-token RMS scale, attn)
    public final FloatArray     ffnScaleBatch;     // B        (per-token RMS scale, FFN)
    public final IntArray       batchStartPosHolder; // 1      (start position of chunk)

    public LlamaState(Configuration config, int batchsize) {
        super(config, batchsize);
        int gpuBatchSize = Integer.getInteger("llama.prefillBatchSize", 1);
        if (gpuBatchSize > 1) {
            int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
            this.embeddingXBatch   = new HalfFloatArray(gpuBatchSize * config.dim());
            this.wrapXBatch        = new FloatArray(gpuBatchSize * config.dim());
            this.wrapXbFP16Batch   = new HalfFloatArray(gpuBatchSize * config.dim());
            this.wrapQBatch        = new FloatArray(gpuBatchSize * config.dim());
            this.wrapKBatch        = new FloatArray(gpuBatchSize * kvDim);
            this.wrapVBatch        = new FloatArray(gpuBatchSize * kvDim);
            this.wrapXbBatch       = new FloatArray(gpuBatchSize * config.dim());
            this.wrapHbBatch       = new FloatArray(gpuBatchSize * config.hiddenDim());
            this.attnScaleBatch    = new FloatArray(gpuBatchSize);
            this.ffnScaleBatch     = new FloatArray(gpuBatchSize);
            this.batchStartPosHolder = new IntArray(1);
        } else {
            this.embeddingXBatch   = null;
            this.wrapXBatch        = null;
            this.wrapXbFP16Batch   = null;
            this.wrapQBatch        = null;
            this.wrapKBatch        = null;
            this.wrapVBatch        = null;
            this.wrapXbBatch       = null;
            this.wrapHbBatch       = null;
            this.attnScaleBatch    = null;
            this.ffnScaleBatch     = null;
            this.batchStartPosHolder = null;
        }
    }

    @Override
    protected StateFields createStateFields(Configuration config) {
        StateFields fields = new StateFields();

        // Allocation with Llama/Mistral dimensions
        fields.x = ArrayFloatTensor.allocate(config.dim());
        fields.xb = ArrayFloatTensor.allocate(config.dim());
        fields.xb2 = ArrayFloatTensor.allocate(config.dim());
        fields.hb = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.hb2 = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.q = ArrayFloatTensor.allocate(config.dim());
        fields.k = ArrayFloatTensor.allocate(config.dim());
        fields.v = ArrayFloatTensor.allocate(config.dim());
        fields.att = ArrayFloatTensor.allocate(config.numberOfHeads(), config.contextLength());
        fields.logits = ArrayFloatTensor.allocate(config.vocabularySize());

        // Key-value cache with Llama/Mistral dimensions
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        fields.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength(), kvDim)).limit(config.numberOfLayers()).toArray(FloatTensor[]::new);
        fields.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength(), kvDim)).limit(config.numberOfLayers()).toArray(FloatTensor[]::new);

        // TornadoVM wrappers with Llama/Mistral dimensions
        fields.wrapX = new FloatArray(config.dim());
        fields.wrapXb = new FloatArray(config.dim());
        fields.wrapXb2 = new FloatArray(config.dim());
        fields.wrapHb = new FloatArray(config.hiddenDim());
        fields.wrapHb2 = new FloatArray(config.hiddenDim());

        switch (config.quantization()) {
            case "FP16" -> fields.createActivationFP16(config.dim());
            case "Q8_0" -> fields.createActivationQ8_0(config.dim());
            default -> throw new UnsupportedOperationException("Unsupported quantization format: " + config.quantization());
        }
        fields.wrapLogits = new FloatArray(config.vocabularySize());
        fields.wrapQ = new FloatArray(config.dim());
        fields.wrapK = new FloatArray(config.dim());
        fields.wrapV = new FloatArray(config.dim());

        fields.wrapXFP16 = new HalfFloatArray(config.dim());
        fields.wrapXbFP16 = new HalfFloatArray(config.dim());
        // dim vs kvdim
        fields.wrapKeyCache = new FloatArray(config.contextLength() * kvDim * config.numberOfLayers());
        fields.wrapValueCache = new FloatArray(config.contextLength() * kvDim * config.numberOfLayers());
        fields.wrapValueCache.init(0.f);
        fields.wrapKeyCache.init(0.f);
        fields.wrapAtt = new FloatArray(config.numberOfHeads() * config.contextLength());
        fields.positionHolder = new IntArray(1);

        // Temporary arrays
        fields.temp = new FloatArray(1 + ((config.dim() + localSize - 1) / localSize));
        fields.tempFFN = new FloatArray(1 + ((config.dim() + localSize - 1) / localSize));
        fields.tempLogits = new FloatArray(1 + ((config.dim() + localSize - 1) / localSize));

        return fields;
    }

    @Override
    public void resetBatchActivationBuffers() {
        wrapXBatch.clear();
        batchStartPosHolder.init(0);
    }
}
