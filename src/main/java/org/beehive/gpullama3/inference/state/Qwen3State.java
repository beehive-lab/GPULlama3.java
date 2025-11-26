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

    public Qwen3State(Configuration config, int batchsize) {
        super(config, batchsize);
        // Initialize Qwen3-specific fields
        Qwen3Configuration qwen3config = (Qwen3Configuration) config;
        int nEmbdHead = qwen3config.numberOfHeads();
        this.tempQcur = new FloatArray(nEmbdHead);
        this.tempKcur = new FloatArray(nEmbdHead);
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

        fields.embeddingX = new HalfFloatArray(config.dim());
        fields.wrapX = new FloatArray(config.dim());
        fields.wrapXb = new FloatArray(nEmbdHeadK * config.numberOfHeads());
        fields.wrapXb2 = new FloatArray(config.dim());
        fields.wrapHb = new FloatArray(config.hiddenDim());
        fields.wrapHb2 = new FloatArray(config.hiddenDim());
        fields.wrapLogits = new FloatArray(config.vocabularySize());
        fields.wrapQ = new FloatArray(nEmbdHeadK * config.numberOfHeads());
        fields.wrapK = new FloatArray(nEmbdKGqa);
        fields.wrapV = new FloatArray(nEmbdKGqa);
        fields.embeddingX = new HalfFloatArray(config.dim());
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
