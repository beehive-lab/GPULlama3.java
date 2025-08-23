package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.qwen2.Qwen2Configuration;

import java.util.stream.Stream;

public class Qwen2State extends State {

    //Qwen2 specific fields TODO

    public Qwen2State(Configuration config, int batchsize) {
        super(config, batchsize);
        // Initialize Qwen2-specific fields TODO
        Qwen2Configuration qwen2Config = (Qwen2Configuration) config;
    }
    @Override
    protected StateFields createStateFields(Configuration configuration) {
        StateFields fields = new StateFields();

        Qwen2Configuration config = (Qwen2Configuration) configuration;

        int nEmbdGqa = config.kvDim();

        // with Qwen2-specific sizes
        fields.x = ArrayFloatTensor.allocate(config.dim());
        fields.xb = ArrayFloatTensor.allocate(config.dim());
        fields.xb2 = ArrayFloatTensor.allocate(config.dim());
        fields.hb = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.hb2 = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.q = ArrayFloatTensor.allocate(config.dim());
        fields.k = ArrayFloatTensor.allocate(config.kvDim());
        fields.v = ArrayFloatTensor.allocate(config.kvDim());
        fields.att = ArrayFloatTensor.allocate(config.numberOfHeads(), config.contextLength());
        fields.logits = ArrayFloatTensor.allocate(config.vocabularySize());

        // Key-value cache with Qwen2 dimensions
        fields.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength(), nEmbdGqa)).limit(config.numberOfLayers()).toArray(FloatTensor[]::new);
        fields.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength(), nEmbdGqa)).limit(config.numberOfLayers()).toArray(FloatTensor[]::new);

        return fields;

    }
}
