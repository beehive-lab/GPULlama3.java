package com.example.inference.state;

import com.example.core.model.tensor.ArrayFloatTensor;
import com.example.core.model.tensor.FloatTensor;
import com.example.model.Configuration;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.stream.Stream;

public final class LlamaState extends State {

    public LlamaState(Configuration config, int batchsize) {
        super(config, batchsize);
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

        fields.wrapLogits = new FloatArray(config.vocabularySize());
        fields.wrapQ = new FloatArray(config.dim());
        fields.wrapK = new FloatArray(config.dim());
        fields.wrapV = new FloatArray(config.dim());

        // dim vs kvdim
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
