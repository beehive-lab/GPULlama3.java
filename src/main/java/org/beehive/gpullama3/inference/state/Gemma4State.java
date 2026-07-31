package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.tensor.standard.ArrayFloatTensor;
import org.beehive.gpullama3.tensor.standard.FloatTensor;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.gemma4.Gemma4Configuration;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Inference state for Gemma 4 models.
 *
 * <p>In addition to the common buffers, Gemma 4 needs scratch space for its per-layer
 * embedding (PLE) mechanism. Buffers that vary in size across layers (Q/K/V, attention
 * output, FFN hidden state) are sized to the maximum across all layers. The KV cache is
 * allocated per "physical" layer: layers that reuse an earlier layer's KV cache (Gemma4's
 * "shared KV layers" feature) simply alias that layer's cache arrays.</p>
 *
 * <p>The TornadoVM (GPU) wrapper buffers mirror the same scheme: {@link #wrapKeyCache}/
 * {@link #wrapValueCache} are laid out back-to-back only for layers that own a KV cache
 * (see {@link #cacheLayerBaseOffset}), and the per-layer-embedding scratch buffers are
 * exposed as flat {@link FloatArray}s for transfer to the GPU.</p>
 */
public final class Gemma4State extends State {

    /** Per-layer projected input embeddings (PLE), laid out as [layer][embeddingLengthPerLayer]. */
    public final FloatTensor perLayerInputs;
    /** Scratch buffer for the per-layer model projection output, same layout as {@link #perLayerInputs}. */
    public final FloatTensor perLayerProjScratch;
    /** Scratch buffer for a single layer's gated per-layer-embedding contribution. */
    public final FloatTensor perLayerGate;
    /** Scratch buffer for a single layer's projected per-layer-embedding output (dim-sized). */
    public final FloatTensor perLayerOut;

    /**
     * For each layer {@code l}, the base element offset of its KV-cache slot inside
     * {@link #wrapKeyCache}/{@link #wrapValueCache} (GPU path) and {@link #keyCache}/{@link #valueCache}
     * (CPU path, where it doubles as the "physical" layer index used for cache aliasing). Layers that
     * reuse an earlier layer's cache share that layer's offset, so attention kernels can address the
     * (possibly shared) cache uniformly via {@code cacheLayerBaseOffset[l]} without branching on reuse.
     */
    public final int[] cacheLayerBaseOffset;

    // GPU (TornadoVM) per-layer-embedding scratch buffers; mirror perLayerInputs/perLayerProjScratch/perLayerGate/perLayerOut.
    public final FloatArray wrapPerLayerInputs;
    public final FloatArray wrapPerLayerProjScratch;
    public final FloatArray wrapPerLayerGate;
    public final FloatArray wrapPerLayerOut;
    /** Holds the current token's per-layer-token-embedding row (gathered on the host each step, then transferred to the GPU). */
    public final FloatArray wrapPerLayerTokenEmbedRow;

    // Extra RMSNorm reduction scratch buffers (GPU path): Gemma4's "sandwich norm" pattern needs five
    // independent reductions per layer (attn-norm uses the inherited `temp`, FFN-norm `tempFFN`); each
    // of the others gets its own buffer so consecutive reduce/apply pairs never alias.
    public final FloatArray tempPostAttn;
    public final FloatArray tempPostFfn;
    public final FloatArray tempPostPle;

    public Gemma4State(Configuration config, int batchsize) {
        super(config, batchsize);

        Gemma4Configuration gemma4config = (Gemma4Configuration) config;
        int perLayerTotal = gemma4config.numberOfLayers() * gemma4config.embeddingLengthPerLayer();
        this.perLayerInputs = ArrayFloatTensor.allocate(perLayerTotal);
        this.perLayerProjScratch = ArrayFloatTensor.allocate(perLayerTotal);
        this.perLayerGate = ArrayFloatTensor.allocate(gemma4config.embeddingLengthPerLayer());
        this.perLayerOut = ArrayFloatTensor.allocate(gemma4config.dim());

        this.cacheLayerBaseOffset = computeCacheLayerBaseOffsets(gemma4config);

        this.wrapPerLayerInputs = new FloatArray(perLayerTotal);
        this.wrapPerLayerProjScratch = new FloatArray(perLayerTotal);
        this.wrapPerLayerGate = new FloatArray(gemma4config.embeddingLengthPerLayer());
        this.wrapPerLayerOut = new FloatArray(gemma4config.dim());
        this.wrapPerLayerTokenEmbedRow = new FloatArray(perLayerTotal);

        int tempSize = 1 + ((gemma4config.dim() + localSize - 1) / localSize);
        this.tempPostAttn = new FloatArray(tempSize);
        this.tempPostFfn = new FloatArray(tempSize);
        this.tempPostPle = new FloatArray(tempSize);
    }

    /**
     * Computes, for each layer, the base element offset of its KV-cache slot in a flat buffer that
     * back-to-back concatenates only the caches of layers that own one ({@link Gemma4Configuration#hasOwnKv}).
     * Reusing layers inherit their source layer's offset (and -- by construction -- its head dimension,
     * since {@link Gemma4Configuration#kvReuseLayer} only ever points to a layer with the same {@code isSwa}-ness).
     */
    private static int[] computeCacheLayerBaseOffsets(Gemma4Configuration config) {
        int nHeadKv = config.numberOfKeyValueHeads();
        int[] offsets = new int[config.numberOfLayers()];
        int running = 0;
        for (int l = 0; l < config.numberOfLayers(); l++) {
            int reuse = config.kvReuseLayer(l);
            if (reuse < 0) {
                offsets[l] = running;
                running += config.contextLength() * (nHeadKv * config.headDim(l));
            } else {
                offsets[l] = offsets[reuse];
            }
        }
        return offsets;
    }

    /** Total number of elements needed for the (deduplicated) flat KV cache buffer. */
    private static int totalCacheElements(Gemma4Configuration config, int[] cacheLayerBaseOffset) {
        int nHeadKv = config.numberOfKeyValueHeads();
        int total = 0;
        for (int l = 0; l < config.numberOfLayers(); l++) {
            if (config.hasOwnKv(l)) {
                total = Math.max(total, cacheLayerBaseOffset[l] + config.contextLength() * (nHeadKv * config.headDim(l)));
            }
        }
        return total;
    }

    @Override
    protected StateFields createStateFields(Configuration configuration) {
        StateFields fields = new StateFields();

        Gemma4Configuration config = (Gemma4Configuration) configuration;

        int dim = config.dim();
        int nHead = config.numberOfHeads();
        int nHeadKv = config.numberOfKeyValueHeads();
        int maxHeadDim = config.maxHeadDim();
        int maxFFN = config.maxFeedForwardLength();

        int qSize = nHead * maxHeadDim;
        int kvSize = nHeadKv * maxHeadDim;

        fields.x = ArrayFloatTensor.allocate(dim);
        fields.xb = ArrayFloatTensor.allocate(Math.max(dim, qSize));
        fields.xb2 = ArrayFloatTensor.allocate(dim);
        fields.hb = ArrayFloatTensor.allocate(maxFFN);
        fields.hb2 = ArrayFloatTensor.allocate(maxFFN);
        fields.q = ArrayFloatTensor.allocate(qSize);
        fields.k = ArrayFloatTensor.allocate(kvSize);
        fields.v = ArrayFloatTensor.allocate(kvSize);
        fields.att = ArrayFloatTensor.allocate(nHead, config.contextLength());
        fields.logits = ArrayFloatTensor.allocate(config.vocabularySize());

        // KV cache: layers that own their KV get a fresh cache; layers that reuse an earlier
        // layer's KV (Gemma4's "shared KV layers") alias that layer's arrays directly.
        FloatTensor[] keyCache = new FloatTensor[config.numberOfLayers()];
        FloatTensor[] valueCache = new FloatTensor[config.numberOfLayers()];
        for (int l = 0; l < config.numberOfLayers(); l++) {
            int reuse = config.kvReuseLayer(l);
            if (reuse < 0) {
                int layerKvDim = config.headDim(l) * nHeadKv;
                keyCache[l] = ArrayFloatTensor.allocate(config.contextLength(), layerKvDim);
                valueCache[l] = ArrayFloatTensor.allocate(config.contextLength(), layerKvDim);
            } else {
                keyCache[l] = keyCache[reuse];
                valueCache[l] = valueCache[reuse];
            }
        }
        fields.keyCache = keyCache;
        fields.valueCache = valueCache;

        switch (config.quantization()) {
            case "FP16" -> fields.createActivationFP16(dim);
            case "Q8_0" -> fields.createActivationQ8_0(dim);
            default -> throw new UnsupportedOperationException("Unsupported quantization format: " + config.quantization());
        }

        fields.wrapX = new FloatArray(dim);
        fields.wrapXb = new FloatArray(Math.max(dim, qSize));
        fields.wrapXbFP16 = new HalfFloatArray(Math.max(dim, qSize));
        fields.wrapXb2 = new FloatArray(dim);
        fields.wrapHb = new FloatArray(maxFFN);
        fields.wrapHb2 = new FloatArray(maxFFN);
        fields.wrapLogits = new FloatArray(config.vocabularySize());
        fields.wrapQ = new FloatArray(qSize);
        fields.wrapK = new FloatArray(kvSize);
        fields.wrapV = new FloatArray(kvSize);

        // Flat GPU KV cache: back-to-back slots only for layers that own a cache (see cacheLayerBaseOffset).
        int[] gpuCacheLayerBaseOffset = computeCacheLayerBaseOffsets(config);
        int totalCacheElements = Math.max(1, totalCacheElements(config, gpuCacheLayerBaseOffset));
        fields.wrapKeyCache = new FloatArray(totalCacheElements);
        fields.wrapValueCache = new FloatArray(totalCacheElements);
        fields.wrapValueCache.init(0.f);
        fields.wrapKeyCache.init(0.f);
        fields.wrapAtt = new FloatArray(nHead * config.contextLength());
        fields.positionHolder = new IntArray(1);

        fields.temp = new FloatArray(1 + ((dim + localSize - 1) / localSize));
        fields.tempFFN = new FloatArray(1 + ((dim + localSize - 1) / localSize));
        fields.tempLogits = new FloatArray(1 + ((dim + localSize - 1) / localSize));

        return fields;
    }
}
