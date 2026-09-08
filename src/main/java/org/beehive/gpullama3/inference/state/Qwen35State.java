package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.qwen35.Qwen35Configuration;
import org.beehive.gpullama3.tensor.standard.ArrayFloatTensor;
import org.beehive.gpullama3.tensor.standard.FloatTensor;

/**
 * Session state for the {@code qwen35} architecture.
 *
 * <p>Two things distinguish it from every other family's state.
 *
 * <h2>Key/value storage exists for a quarter of the layers</h2>
 *
 * <p>Only the attention layers have keys and values to retain. The 48 recurrent layers of the 27B
 * would otherwise each be handed a full context-length cache they never write — at 8k context and
 * this model's 1024-wide KV that is 2.5 GB of untouched arrays. {@link #keyCache} and {@link
 * #valueCache} are therefore {@code null} at a recurrent layer's index, and the arrays stay indexed
 * by absolute layer so no caller keeps a second numbering.
 *
 * <h2>Recurrent layers hold state instead, and it is not a cache</h2>
 *
 * <p>{@link #convState} and {@link #deltaState} are fixed-size per layer and independent of
 * position: a delta-net layer's entire history is summed into a {@code head_v_dim × head_v_dim}
 * matrix per value head, and the convolution keeps the last {@code kernel - 1} inputs. That is why
 * they are session state and not KV storage — nothing about them can be paged, evicted, shared or
 * leased, and the engine's cache manager would have nothing to manage.
 *
 * <p>It is also why {@link #resetSequenceState()} matters here and is a no-op elsewhere. A key/value
 * cache does not need clearing between sequences because attention only reads up to the current
 * position; a recurrent state has no such mask, and a stale one silently conditions the new
 * sequence on the old one.
 */
public final class Qwen35State extends State {

    /**
     * Rolling convolution history per recurrent layer, {@code convDim * (kernel - 1)} elements,
     * channel-major and oldest tap first: element {@code c * (kernel - 1) + t} is channel {@code c}
     * as it stood {@code kernel - 1 - t} steps ago. Channel-major to match how GGUF lays out {@code
     * ssm_conv1d}, so the convolution reads both operands the same way. {@code null} at an
     * attention layer.
     */
    public final FloatTensor[] convState;

    /**
     * Delta-net state per recurrent layer: {@code numberOfValueHeads} matrices of {@code
     * headValueDim × headValueDim}, indexed {@code (h * S + i) * S + j} for key row {@code i} and
     * value column {@code j}. {@code null} at an attention layer.
     */
    public final FloatTensor[] deltaState;

    // ---- scratch, sized once ------------------------------------------------

    /** The fused {@code q ‖ k ‖ v} projection of a recurrent layer, before the convolution. */
    public final FloatTensor ssmQkv;

    /** The same after the depthwise convolution and its SiLU. */
    public final FloatTensor ssmConvOut;

    /** The {@code z} gate the delta-net output is normalized against. */
    public final FloatTensor ssmZ;

    /** Per-value-head decay, from {@code ssm_alpha} through its bias, softplus and {@code ssm_a}. */
    public final FloatTensor ssmAlpha;

    /** Per-value-head write strength, from {@code ssm_beta} through the logistic. */
    public final FloatTensor ssmBeta;

    /** The recurrent branch's readout, {@code value_dim} wide, before the gated norm. */
    public final FloatTensor ssmOut;

    /** The gated norm's result, feeding the recurrent output projection. */
    public final FloatTensor ssmNormed;

    /**
     * The trunk's final hidden state, after {@code output_norm} and before the LM head.
     *
     * <p>Kept because the MTP block consumes it: its input is this vector concatenated with the
     * drafted token's embedding. Written by every trunk forward pass whether or not speculation is
     * running, because the cost is one copy of {@code dim} floats and the alternative is a forward
     * pass whose behaviour depends on a mode flag.
     */
    public final FloatTensor hNextn;

    /** The MTP block's concatenated {@code [enorm(embedding) ‖ hnorm(hidden)]}, {@code 2 * dim}. */
    public final FloatTensor nextnConcat;

    /** The query half of an attention layer's fused query/gate projection, de-interleaved. */
    public final FloatTensor attnQ;

    /** Its gate half. Applied through a logistic to the attention result. */
    public final FloatTensor attnGate;

    /** The convolved query slice of a recurrent layer, {@code keyHeads * headKeyDim}. */
    public final FloatTensor ssmQ;

    /** Its key slice, the same width. */
    public final FloatTensor ssmK;

    /** Its value slice, {@code valueHeads * headValueDim}. */
    public final FloatTensor ssmV;

    public Qwen35State(Configuration config, int batchsize) {
        this(config, batchsize, null);
    }

    public Qwen35State(
            Configuration config, int batchsize, org.beehive.gpullama3.runtime.kv.KvLease lease) {
        super(config, batchsize, lease);
        Qwen35Configuration c = (Qwen35Configuration) config;

        this.convState = new FloatTensor[c.numberOfLayers()];
        this.deltaState = new FloatTensor[c.numberOfLayers()];
        for (int l = 0; l < c.numberOfLayers(); l++) {
            if (c.isRecurrentLayer(l)) {
                this.convState[l] = ArrayFloatTensor.allocate(c.convStateSize());
                this.deltaState[l] = ArrayFloatTensor.allocate(c.deltaNetStateSize());
            }
        }

        this.ssmQkv = ArrayFloatTensor.allocate(c.deltaNetConvDim());
        this.ssmConvOut = ArrayFloatTensor.allocate(c.deltaNetConvDim());
        this.ssmZ = ArrayFloatTensor.allocate(c.deltaNetValueDim());
        this.ssmAlpha = ArrayFloatTensor.allocate(c.numberOfValueHeads());
        this.ssmBeta = ArrayFloatTensor.allocate(c.numberOfValueHeads());
        this.ssmOut = ArrayFloatTensor.allocate(c.deltaNetValueDim());
        this.ssmNormed = ArrayFloatTensor.allocate(c.deltaNetValueDim());
        this.hNextn = ArrayFloatTensor.allocate(c.dim());
        this.nextnConcat = ArrayFloatTensor.allocate(2 * c.dim());
        this.attnQ = ArrayFloatTensor.allocate(c.attentionOutputInputDim());
        this.attnGate = ArrayFloatTensor.allocate(c.attentionOutputInputDim());
        this.ssmQ = ArrayFloatTensor.allocate(c.deltaNetKeyDim());
        this.ssmK = ArrayFloatTensor.allocate(c.deltaNetKeyDim());
        this.ssmV = ArrayFloatTensor.allocate(c.deltaNetValueDim());
    }

    /**
     * Zeroes the recurrent state so a reused session does not continue the previous sequence.
     *
     * <p>The key/value caches are deliberately left alone: attention reads only up to the current
     * position, so rewinding the position is enough for them, and clearing them would cost a
     * context-length write for no effect.
     */
    @Override
    public void resetSequenceState() {
        for (int l = 0; l < convState.length; l++) {
            if (convState[l] != null) {
                convState[l].fillInPlace(0, convState[l].size(), 0f);
                deltaState[l].fillInPlace(0, deltaState[l].size(), 0f);
            }
        }
    }

    @Override
    protected int batchQDim(Configuration config) {
        return ((Qwen35Configuration) config).attentionOutputInputDim();
    }

    @Override
    protected int batchKvDim(Configuration config) {
        return ((Qwen35Configuration) config).kvDim();
    }

    @Override
    protected StateFields createStateFields(Configuration configuration) {
        Qwen35Configuration config = (Qwen35Configuration) configuration;
        StateFields fields = new StateFields();

        int kvDim = config.kvDim();

        fields.x = ArrayFloatTensor.allocate(config.dim());
        // Wide enough for the attention branch's concatenated heads, which exceed dim here
        // (24 heads of 256 against a 5120 embedding), and reused by the feed-forward branch.
        fields.xb = ArrayFloatTensor.allocate(Math.max(config.attentionOutputInputDim(), config.dim()));
        fields.xb2 = ArrayFloatTensor.allocate(config.dim());
        fields.hb = ArrayFloatTensor.allocate(config.hiddenDim());
        fields.hb2 = ArrayFloatTensor.allocate(config.hiddenDim());
        // Query and output gate arrive fused, interleaved per head, and stay that way.
        fields.q = ArrayFloatTensor.allocate(config.queryGateDim());
        fields.k = ArrayFloatTensor.allocate(kvDim);
        fields.v = ArrayFloatTensor.allocate(kvDim);
        fields.att = ArrayFloatTensor.allocate(config.numberOfHeads(), config.contextLength());
        fields.logits = ArrayFloatTensor.allocate(config.vocabularySize());

        // One cache per block that actually attends; null elsewhere, indexed by absolute block.
        int blocks = config.numberOfBlocks();
        fields.keyCache = new FloatTensor[blocks];
        fields.valueCache = new FloatTensor[blocks];
        for (int l = 0; l < blocks; l++) {
            if (!config.isRecurrentLayer(l)) {
                fields.keyCache[l] =
                        ArrayFloatTensor.allocate(config.contextLength(), kvDim);
                fields.valueCache[l] =
                        ArrayFloatTensor.allocate(config.contextLength(), kvDim);
            }
        }

        // No device workspace: no backend claims this architecture yet, and allocating buffers
        // for a plan nobody builds would reserve memory this model has none to spare.
        fields.kvBlockCfg = 0;
        fields.kvBlockStride = 0;
        return fields;
    }
}
