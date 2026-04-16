package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.gemma4.Gemma4Configuration;
import org.beehive.gpullama3.tensor.standard.ArrayFloatTensor;
import org.beehive.gpullama3.tensor.standard.FloatTensor;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.util.stream.IntStream;

/**
 * Gemma 4 model state with dual head sizes, SWA-aware KV caches,
 * per-layer embedding buffers, and MoE buffers.
 */
public final class Gemma4State extends State {

    // Additional buffer for attention output before O projection (max queryDim)
    public final FloatTensor xb_k;

    // Per-layer KV cache offsets into the flat TornadoVM cache arrays
    public int[] kvCacheLayerOffset;

    // Per-layer embedding buffers (null if not used)
    public final FloatTensor perLayerInputs;
    public final FloatTensor plGate;
    public final FloatTensor plProj;

    // MoE buffers (null if dense model)
    public final FloatTensor routerLogits;
    public final FloatTensor moeInput;
    public final FloatTensor moeOutput;
    public final FloatTensor expertGateUp;
    public final FloatTensor expertDown;

    public Gemma4State(Gemma4Configuration config, int batchsize) {
        super(config, batchsize);

        int maxQueryDim = config.numberOfHeads() * config.headSizeFull();
        this.xb_k = ArrayFloatTensor.allocate(maxQueryDim);

        // Per-layer embedding buffers
        int plDim = config.embeddingLengthPerLayer();
        if (plDim > 0) {
            this.perLayerInputs = ArrayFloatTensor.allocate(plDim * config.numberOfLayers());
            this.plGate = ArrayFloatTensor.allocate(plDim);
            this.plProj = ArrayFloatTensor.allocate(config.dim());
        } else {
            this.perLayerInputs = null;
            this.plGate = null;
            this.plProj = null;
        }

        // MoE buffers
        if (config.isMoE()) {
            this.routerLogits = ArrayFloatTensor.allocate(config.expertCount());
            this.moeInput = ArrayFloatTensor.allocate(config.dim());
            this.moeOutput = ArrayFloatTensor.allocate(config.dim());
            this.expertGateUp = ArrayFloatTensor.allocate(2 * config.expertFeedForwardLength());
            this.expertDown = ArrayFloatTensor.allocate(config.dim());
        } else {
            this.routerLogits = null;
            this.moeInput = null;
            this.moeOutput = null;
            this.expertGateUp = null;
            this.expertDown = null;
        }
    }

    @Override
    protected StateFields createStateFields(Configuration configuration) {
        StateFields fields = new StateFields();
        Gemma4Configuration config = (Gemma4Configuration) configuration;

        int dim = config.dim();
        int maxQueryDim = config.numberOfHeads() * config.headSizeFull();
        int maxKVDim = IntStream.range(0, config.numberOfLayers()).map(config::kvDim).max().orElse(0);
        int maxHiddenDim = config.maxHiddenDim();

        fields.x = ArrayFloatTensor.allocate(dim);
        fields.xb = ArrayFloatTensor.allocate(dim);
        fields.xb2 = ArrayFloatTensor.allocate(dim);
        fields.hb = ArrayFloatTensor.allocate(maxHiddenDim);
        fields.hb2 = ArrayFloatTensor.allocate(maxHiddenDim);
        fields.q = ArrayFloatTensor.allocate(maxQueryDim);
        fields.k = ArrayFloatTensor.allocate(maxKVDim);
        fields.v = ArrayFloatTensor.allocate(maxKVDim);
        fields.att = ArrayFloatTensor.allocate(config.numberOfHeads(), config.contextLength());
        fields.logits = ArrayFloatTensor.allocate(config.vocabularySize());

        // KV cache: only allocate for layers that have their own KV (not shared)
        int nLayerKv = config.nLayerKvFromStart();
        fields.keyCache = new FloatTensor[nLayerKv];
        fields.valueCache = new FloatTensor[nLayerKv];
        for (int l = 0; l < nLayerKv; l++) {
            int kvDim = config.kvDim(l);
            int kvPositions = config.kvCachePositions(l);
            fields.keyCache[l] = ArrayFloatTensor.allocate(kvPositions, kvDim);
            fields.valueCache[l] = ArrayFloatTensor.allocate(kvPositions, kvDim);
        }

        // TornadoVM wrappers (minimal allocation for CPU-first)
        switch (config.quantization()) {
            case "FP16" -> fields.createActivationFP16(dim);
            case "Q8_0" -> fields.createActivationQ8_0(dim);
            default -> throw new UnsupportedOperationException("Unsupported quantization format: " + config.quantization());
        }

        fields.wrapX = new FloatArray(dim);
        fields.wrapXb = new FloatArray(maxQueryDim);
        fields.wrapXb2 = new FloatArray(dim);
        fields.wrapHb = new FloatArray(maxHiddenDim);
        fields.wrapHb2 = new FloatArray(maxHiddenDim);
        fields.wrapLogits = new FloatArray(config.vocabularySize());
        fields.wrapQ = new FloatArray(maxQueryDim);
        fields.wrapK = new FloatArray(maxKVDim);
        fields.wrapV = new FloatArray(maxKVDim);
        // KV cache: compute per-layer offsets and total flat cache size
        this.kvCacheLayerOffset = new int[nLayerKv];
        int totalCacheSize = 0;
        for (int l = 0; l < nLayerKv; l++) {
            kvCacheLayerOffset[l] = totalCacheSize;
            totalCacheSize += config.kvCachePositions(l) * config.kvDim(l);
        }
        fields.wrapKeyCache = new FloatArray(totalCacheSize);
        fields.wrapValueCache = new FloatArray(totalCacheSize);
        fields.wrapAtt = new FloatArray(config.numberOfHeads() * config.contextLength());
        fields.positionHolder = new IntArray(1);

        fields.temp = new FloatArray(1 + ((dim + localSize - 1) / localSize));
        fields.tempFFN = new FloatArray(1 + ((dim + localSize - 1) / localSize));
        fields.tempLogits = new FloatArray(1 + ((dim + localSize - 1) / localSize));

        return fields;
    }
}
