package org.beehive.gpullama3.inference.weights.tornado;

import org.beehive.gpullama3.backend.tornado.tensor.TornadoTensor;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.runtime.tensor.DataType;

/**
 * Device weights for the {@code qwen35} architecture.
 *
 * <p>Implements {@link Weights} directly rather than extending {@link TornadoWeights}, for the
 * reason {@code Qwen35StandardWeights} does not extend {@code StandardWeights}: that base class
 * assumes every layer has a query, key and value projection, and here three layers in four have
 * none. Extending it would mean 48 nulls per array and a base class whose fields lie about the
 * model.
 *
 * <p>Every per-layer array is indexed by <b>absolute block index</b>, with {@code null} where the
 * block is of the other kind — the same convention the host weights use, and for the same reason:
 * a second, compacted numbering is the kind of off-by-one that produces fluent, wrong text. The
 * dense indices that do exist ({@code keyValueLayerIndex}, {@code recurrentLayerIndex}) address
 * <i>state</i>, not weights, and are the configuration's business rather than this class's.
 */
public final class Qwen35TornadoWeights implements Weights {

    /** Trunk layers plus MTP blocks; the length of every per-layer array here. */
    public final int blockCount;

    // ---- shared by both layer kinds ---------------------------------------

    public final TornadoTensor tokenEmbeddingTable;

    /** Input norm of the mixer branch, every block. */
    public final TornadoTensor[] attnNorm;

    /** Input norm of the feed-forward branch. Named {@code post_attention_norm} in the file. */
    public final TornadoTensor[] ffnNorm;

    public final TornadoTensor[] ffnGate;
    public final TornadoTensor[] ffnDown;
    public final TornadoTensor[] ffnUp;

    public final TornadoTensor outputNorm;
    public final TornadoTensor output;

    /** RoPE tables, precomputed over {@code rope.dimension_count} rather than the head width. */
    public final TornadoTensor freqCisReal;

    public final TornadoTensor freqCisImag;

    // ---- attention blocks --------------------------------------------------

    /** The fused query/gate projection: per head a query slice then a gate slice. */
    public final TornadoTensor[] wq;

    public final TornadoTensor[] wk;
    public final TornadoTensor[] wv;
    public final TornadoTensor[] wo;
    public final TornadoTensor[] attnQNorm;
    public final TornadoTensor[] attnKNorm;

    // ---- recurrent blocks --------------------------------------------------

    /** Fused {@code q ‖ k ‖ v} projection feeding the depthwise convolution. */
    public final TornadoTensor[] ssmQkv;

    /** The {@code z} gate the delta-net output is normalized against. */
    public final TornadoTensor[] ssmGate;

    /** Depthwise causal convolution kernel, {@code conv_kernel} taps per channel. */
    public final TornadoTensor[] ssmConv1d;

    /** Per-value-head decay projection, before the bias, softplus and {@code ssmA}. */
    public final TornadoTensor[] ssmAlpha;

    /** Per-value-head write-strength projection, before the logistic. */
    public final TornadoTensor[] ssmBeta;

    /** Bias added to the decay projection before the softplus. */
    public final TornadoTensor[] ssmDtBias;

    /** {@code -exp(A_log)}: multiplies the softplus to give the log decay. */
    public final TornadoTensor[] ssmA;

    /** Gated RMS norm scale over one value head's width. */
    public final TornadoTensor[] ssmNorm;

    /** Output projection of the recurrent branch. */
    public final TornadoTensor[] ssmOut;

    private final DataType weightType;

    // @formatter:off
    public Qwen35TornadoWeights(
            int blockCount,
            TornadoTensor tokenEmbeddingTable,
            TornadoTensor[] attnNorm,
            TornadoTensor[] ffnNorm,
            TornadoTensor[] ffnGate,
            TornadoTensor[] ffnDown,
            TornadoTensor[] ffnUp,
            TornadoTensor outputNorm,
            TornadoTensor output,
            TornadoTensor freqCisReal,
            TornadoTensor freqCisImag,
            TornadoTensor[] wq,
            TornadoTensor[] wk,
            TornadoTensor[] wv,
            TornadoTensor[] wo,
            TornadoTensor[] attnQNorm,
            TornadoTensor[] attnKNorm,
            TornadoTensor[] ssmQkv,
            TornadoTensor[] ssmGate,
            TornadoTensor[] ssmConv1d,
            TornadoTensor[] ssmAlpha,
            TornadoTensor[] ssmBeta,
            TornadoTensor[] ssmDtBias,
            TornadoTensor[] ssmA,
            TornadoTensor[] ssmNorm,
            TornadoTensor[] ssmOut,
            DataType weightType) {
        this.blockCount = blockCount;
        this.tokenEmbeddingTable = tokenEmbeddingTable;
        this.attnNorm = attnNorm;
        this.ffnNorm = ffnNorm;
        this.ffnGate = ffnGate;
        this.ffnDown = ffnDown;
        this.ffnUp = ffnUp;
        this.outputNorm = outputNorm;
        this.output = output;
        this.freqCisReal = freqCisReal;
        this.freqCisImag = freqCisImag;
        this.wq = wq;
        this.wk = wk;
        this.wv = wv;
        this.wo = wo;
        this.attnQNorm = attnQNorm;
        this.attnKNorm = attnKNorm;
        this.ssmQkv = ssmQkv;
        this.ssmGate = ssmGate;
        this.ssmConv1d = ssmConv1d;
        this.ssmAlpha = ssmAlpha;
        this.ssmBeta = ssmBeta;
        this.ssmDtBias = ssmDtBias;
        this.ssmA = ssmA;
        this.ssmNorm = ssmNorm;
        this.ssmOut = ssmOut;
        this.weightType = weightType;
    }
    // @formatter:on

    @Override
    public DataType dataType() {
        return weightType;
    }
}
