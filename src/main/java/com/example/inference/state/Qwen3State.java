package com.example.inference.state;

import com.example.core.model.tensor.ArrayFloatTensor;
import com.example.core.model.tensor.FloatTensor;
import com.example.model.Configuration;
import com.example.model.qwen3.Qwen3Configuration;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.stream.Stream;

public final class Qwen3State extends State {

    // Qwen3-specific field
    public final FloatTensor kq;

    public Qwen3State(Configuration config, int batchsize) {
        super(config, batchsize);
        // Initialize Qwen3-specific field
        this.kq = ArrayFloatTensor.allocate(config.numberOfHeads(), 32, 15);
    }

    @Override
    protected StateFields createStateFields(Configuration configuration) {
        StateFields fields = new StateFields();

        Qwen3Configuration config = (Qwen3Configuration) configuration;

        // Qwen3-specific calculations
        int nHeadKv = config.numberOfKeyValueHeads();
        int nEmbdHeadK = config.numberOfHeadsKey();
        int nEmbdKGqa = nEmbdHeadK * nHeadKv;

        // Qwen3-specific allocation logic
        fields.x = ArrayFloatTensor.allocate(config.dim());
        fields.xb = ArrayFloatTensor.allocate(nEmbdHeadK * config.numberOfHeads());
        fields.xb2 = ArrayFloatTensor.allocate(config.dim());
        fields.hb = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.hb2 = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.q = ArrayFloatTensor.allocate(nEmbdHeadK * config.numberOfHeads());
        fields.k = ArrayFloatTensor.allocate(nEmbdKGqa);  // Different from Llama!
        fields.v = ArrayFloatTensor.allocate(nEmbdKGqa);  // Different from Llama!
        fields.att = ArrayFloatTensor.allocate(config.numberOfHeads(), config.contextLength());
        fields.logits = ArrayFloatTensor.allocate(config.vocabularySize());

        // Key-value cache with Qwen3 dimensions
        int kvDim = nEmbdKGqa;
        fields.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength(), kvDim))
                .limit(config.numberOfLayers()).toArray(FloatTensor[]::new);
        fields.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength(), kvDim))
                .limit(config.numberOfLayers()).toArray(FloatTensor[]::new);

        // TornadoVM wrappers with Qwen3-specific sizes
        fields.wrapX = new FloatArray(config.dim());
        fields.wrapXb = new FloatArray(nEmbdHeadK * config.numberOfHeads());  // Different from Llama!
        fields.wrapXb2 = new FloatArray(config.dim());
        fields.wrapHb = new FloatArray(config.hiddenDim());
        fields.wrapHb2 = new FloatArray(config.hiddenDim());
        fields.wrapLogits = new FloatArray(config.vocabularySize());
        fields.wrapQ = new FloatArray(nEmbdHeadK * config.numberOfHeads());   // Different from Llama!
        fields.wrapK = new FloatArray(nEmbdKGqa);  // Different from Llama!
        fields.wrapV = new FloatArray(nEmbdKGqa);  // Different from Llama!

        fields.wrapKeyCache = new FloatArray(config.contextLength() * kvDim * config.numberOfLayers());
        fields.wrapValueCache = new FloatArray(config.contextLength() * kvDim * config.numberOfLayers());
        fields.wrapValueCache.init(0.f);
        fields.wrapKeyCache.init(0.f);
        fields.wrapAtt = new FloatArray(config.numberOfHeads() * config.contextLength());
        fields.positionHolder = new IntArray(1);

        // Temporary arrays
        fields.temp = new FloatArray(1 + ((config.dim() + localSize-1) / localSize));
        fields.tempFFN = new FloatArray(1 + ((config.dim() + localSize-1) / localSize));
        fields.tempLogits = new FloatArray(1 + ((config.dim() + localSize-1) / localSize));

        return fields;
    }
}
