package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.tensor.standard.ArrayFloatTensor;
import org.beehive.gpullama3.tensor.standard.FloatTensor;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.stream.Stream;

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

    // Split-KV (flash-decoding) attention scratch: per (head, split) partial numerator [headSize] plus
    // block max and block sum. Sized exactly to the configured split count so the split-KV kernel writes
    // every element each layer (a partially-written large buffer is not reliably synced between the
    // split and combine tasks by TornadoVM). See processHeadsFlashAttentionSplitKV / combineSplitKVAttention.
    public FloatArray wrapAttSplit;

    /** Number of KV splits per head for split-KV (flash-decoding) decode attention. */
    public static final int SPLIT_KV = 8;

    public Qwen3State(Configuration config, int batchsize) {
        super(config, batchsize);
        // Initialize Qwen3-specific fields
        Qwen3Configuration qwen3config = (Qwen3Configuration) config;
        int nEmbdHead = qwen3config.numberOfHeads();
        this.tempQcur = new FloatArray(nEmbdHead);
        this.tempKcur = new FloatArray(nEmbdHead);

        int headSizeV = qwen3config.numberOfHeadsValue();
        this.wrapAttSplit = new FloatArray(qwen3config.numberOfHeads() * SPLIT_KV * (headSizeV + 2));
    }

    @Override
    protected int batchQDim(Configuration config) {
        Qwen3Configuration q3 = (Qwen3Configuration) config;
        return q3.numberOfHeadsKey() * q3.numberOfHeads();
    }

    @Override
    protected int batchKvDim(Configuration config) {
        Qwen3Configuration q3 = (Qwen3Configuration) config;
        return q3.numberOfHeadsValue() * q3.numberOfKeyValueHeads();
    }

    @Override
    protected StateFields createStateFields(Configuration configuration) {
        StateFields fields = new StateFields();

        Qwen3Configuration config = (Qwen3Configuration) configuration;

        // Qwen3-specific sizes
        int nHeadKv = config.numberOfKeyValueHeads();
        int nEmbdHeadK = config.numberOfHeadsKey();
        int nEmbdKGqa = nEmbdHeadK * nHeadKv;
        int nEmbdHeadV = config.numberOfHeadsValue();
        int nEmbdVGqa = nEmbdHeadV * nHeadKv;
        int nEmbdGqa = nEmbdVGqa;

        // Qwen3-specific allocation logic
        fields.x = ArrayFloatTensor.allocate(config.dim());
        fields.xb = ArrayFloatTensor.allocate(nEmbdHeadK * config.numberOfHeads());
        fields.xb2 = ArrayFloatTensor.allocate(config.dim());
        fields.hb = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.hb2 = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.q = ArrayFloatTensor.allocate(nEmbdHeadK * config.numberOfHeads());
        fields.k = ArrayFloatTensor.allocate(nEmbdKGqa);
        fields.v = ArrayFloatTensor.allocate(nEmbdKGqa);
        fields.att = ArrayFloatTensor.allocate(config.numberOfHeads(), config.contextLength());
        fields.logits = ArrayFloatTensor.allocate(config.vocabularySize());

        // Key-value cache with Qwen3 dimensions
        fields.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength(), nEmbdGqa)).limit(config.numberOfLayers()).toArray(FloatTensor[]::new);
        fields.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength(), nEmbdGqa)).limit(config.numberOfLayers()).toArray(FloatTensor[]::new);

        // TornadoVM wrappers with Qwen3-specific sizes

        switch (config.quantization()) {
            case "FP16" -> fields.createActivationFP16(config.dim());
            case "Q8_0" -> fields.createActivationQ8_0(config.dim());
            default -> throw new UnsupportedOperationException("Unsupported quantization format: " + config.quantization());
        }

        fields.wrapX = new FloatArray(config.dim());
        fields.wrapXb = new FloatArray(nEmbdHeadK * config.numberOfHeads());
        fields.wrapXbFP16 = new HalfFloatArray(nEmbdHeadK * config.numberOfHeads());

        fields.wrapXb2 = new FloatArray(config.dim());
        fields.wrapHb = new FloatArray(config.hiddenDim());
        fields.wrapHb2 = new FloatArray(config.hiddenDim());
        fields.wrapLogits = new FloatArray(config.vocabularySize());
        fields.wrapQ = new FloatArray(nEmbdHeadK * config.numberOfHeads());
        fields.wrapK = new FloatArray(nEmbdKGqa);
        fields.wrapV = new FloatArray(nEmbdKGqa);
        fields.wrapKeyCache = new FloatArray(config.contextLength() * nEmbdGqa * config.numberOfLayers());
        fields.wrapValueCache = new FloatArray(config.contextLength() * nEmbdGqa * config.numberOfLayers());
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
}
