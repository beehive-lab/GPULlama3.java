package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.gemma3.Gemma3Configuration;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.stream.Stream;

/**
 * Represents the state of the Gemma3 model during inference.
 * This class extends {@link State} to include model-specific functionalities
 * and configurations tailored for the Gemma3 model.
 *
 * <p><b>Note:</b> Gemma3State contains additional fields for TornadoVM wrappers
 * to enable GPU-accelerated processing of the model. It supports Q/K normalization
 * similar to Qwen3.</p>
 */
public final class Gemma3State extends State {

    // Gemma3 specific fields
    // Temporary buffers for intermediate calculations
    public FloatArray tempQcur;
    public FloatArray tempKcur;

    public Gemma3State(Configuration config, int batchsize) {
        super(config, batchsize);
        // Initialize Gemma3-specific fields
        Gemma3Configuration gemma3config = (Gemma3Configuration) config;
        int nEmbdHead = gemma3config.numberOfHeads();
        this.tempQcur = new FloatArray(nEmbdHead);
        this.tempKcur = new FloatArray(nEmbdHead);
    }

    @Override
    protected StateFields createStateFields(Configuration configuration) {
        StateFields fields = new StateFields();

        Gemma3Configuration config = (Gemma3Configuration) configuration;

        // Gemma3-specific sizes
        int nHeadKv = config.numberOfKeyValueHeads();
        int nEmbdHeadK = config.numberOfHeadsKey();
        int nEmbdKGqa = nEmbdHeadK * nHeadKv;
        int nEmbdHeadV = config.numberOfHeadsValue();
        int nEmbdVGqa = nEmbdHeadV * nHeadKv;
        int nEmbdGqa = nEmbdVGqa;

        // Gemma3-specific allocation logic
        fields.x = ArrayFloatTensor.allocate(config.dim());
        // Note: For Gemma3, xb needs to hold the full dim after normalization
        fields.xb = ArrayFloatTensor.allocate(config.dim());
        fields.xb2 = ArrayFloatTensor.allocate(config.dim());
        fields.hb = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.hb2 = ArrayFloatTensor.allocate(config.hiddenDim());
        // Q uses nEmbdHeadK * nHeads (weight matrix output size)
        fields.q = ArrayFloatTensor.allocate(nEmbdHeadK * config.numberOfHeads());
        fields.k = ArrayFloatTensor.allocate(nEmbdKGqa);
        fields.v = ArrayFloatTensor.allocate(nEmbdKGqa);
        fields.att = ArrayFloatTensor.allocate(config.numberOfHeads(), config.contextLength());
        fields.logits = ArrayFloatTensor.allocate(config.vocabularySize());

        // Key-value cache with Gemma3 dimensions
        fields.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength(), nEmbdGqa)).limit(config.numberOfLayers()).toArray(FloatTensor[]::new);
        fields.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength(), nEmbdGqa)).limit(config.numberOfLayers()).toArray(FloatTensor[]::new);

        // TornadoVM wrappers with Gemma3-specific sizes
        fields.wrapX = new FloatArray(config.dim());
        fields.wrapXb = new FloatArray(config.dim());
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
