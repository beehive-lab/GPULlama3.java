package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.tensor.standard.ArrayFloatTensor;
import org.beehive.gpullama3.tensor.standard.FloatTensor;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.devstral.DevstralConfiguration;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.stream.Stream;

/**
 * State for Devstral 2 models where head_dim != dim/num_heads.
 * Allocates Q with qDim (num_heads * head_dim) and K/V with kvDim (num_kv_heads * head_dim).
 */
public final class DevstralState extends State {

    public DevstralState(Configuration config, int batchsize) {
        super(config, batchsize);
    }

    @Override
    protected StateFields createStateFields(Configuration config) {
        DevstralConfiguration dc = (DevstralConfiguration) config;
        StateFields fields = new StateFields();

        int qDim = dc.qDim();
        int kvDim = dc.kvDim();

        fields.x = ArrayFloatTensor.allocate(dc.dim());
        fields.xb = ArrayFloatTensor.allocate(dc.dim());
        fields.xb2 = ArrayFloatTensor.allocate(dc.dim());
        fields.hb = ArrayFloatTensor.allocate(dc.hiddenDim());
        fields.hb2 = ArrayFloatTensor.allocate(dc.hiddenDim());
        fields.q = ArrayFloatTensor.allocate(qDim);
        fields.k = ArrayFloatTensor.allocate(kvDim);
        fields.v = ArrayFloatTensor.allocate(kvDim);
        fields.att = ArrayFloatTensor.allocate(dc.numberOfHeads(), dc.contextLength());
        fields.logits = ArrayFloatTensor.allocate(dc.vocabularySize());

        fields.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(dc.contextLength(), kvDim)).limit(dc.numberOfLayers()).toArray(FloatTensor[]::new);
        fields.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(dc.contextLength(), kvDim)).limit(dc.numberOfLayers()).toArray(FloatTensor[]::new);

        // TornadoVM wrappers
        fields.wrapX = new FloatArray(dc.dim());
        fields.wrapXb = new FloatArray(dc.dim());
        fields.wrapXb2 = new FloatArray(dc.dim());
        fields.wrapHb = new FloatArray(dc.hiddenDim());
        fields.wrapHb2 = new FloatArray(dc.hiddenDim());

        switch (dc.quantization()) {
            case "FP16" -> fields.createActivationFP16(dc.dim());
            case "Q8_0" -> fields.createActivationQ8_0(dc.dim());
            default -> throw new UnsupportedOperationException("Unsupported quantization format: " + dc.quantization());
        }
        fields.wrapLogits = new FloatArray(dc.vocabularySize());
        fields.wrapQ = new FloatArray(qDim);
        fields.wrapK = new FloatArray(kvDim);
        fields.wrapV = new FloatArray(kvDim);

        fields.wrapXFP16 = new HalfFloatArray(dc.dim());
        fields.wrapXbFP16 = new HalfFloatArray(dc.dim());
        fields.wrapKeyCache = new FloatArray(dc.contextLength() * kvDim * dc.numberOfLayers());
        fields.wrapValueCache = new FloatArray(dc.contextLength() * kvDim * dc.numberOfLayers());
        fields.wrapValueCache.init(0.f);
        fields.wrapKeyCache.init(0.f);
        fields.wrapAtt = new FloatArray(dc.numberOfHeads() * dc.contextLength());
        fields.positionHolder = new IntArray(1);

        fields.temp = new FloatArray(1 + ((dc.dim() + localSize - 1) / localSize));
        fields.tempFFN = new FloatArray(1 + ((dc.dim() + localSize - 1) / localSize));
        fields.tempLogits = new FloatArray(1 + ((dc.dim() + localSize - 1) / localSize));

        return fields;
    }
}
