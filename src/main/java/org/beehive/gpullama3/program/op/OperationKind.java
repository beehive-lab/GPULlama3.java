package org.beehive.gpullama3.program.op;

/**
 * The closed set of operations transformer inference needs.
 *
 * <p>Closed on purpose, and closed the same way {@code TensorRole} is: a free-form operation name
 * becomes a dispatch key that architecture-specific code can branch on, which is what Rule 15
 * exists to prevent. <b>Adding a model family must not mean adding a kind</b> — it means assembling
 * a program out of these.
 *
 * <p>A kind is added only when the work genuinely differs, not when a family arranges the same work
 * differently. Grouped query attention is not a second {@link #ATTENTION}; it is {@link #ATTENTION}
 * with fewer key/value heads.
 */
public enum OperationKind {

    /** Root-mean-square normalization with a learned scale. */
    RMS_NORM,

    /** Rotary position embedding applied to the query and key projections. */
    ROPE,

    /** Matrix times vector: one row of activations against a weight matrix. */
    MAT_VEC,

    /** Matrix times matrix: a batch of activation rows against a weight matrix. */
    MAT_MUL,

    /** Writing this step's key and value into the retained key/value store. */
    KV_APPEND,

    /** Scaled dot-product attention over a set of keys and values. */
    ATTENTION,

    /** Softmax over the last dimension. */
    SOFTMAX,

    /** The SwiGLU feed-forward activation: {@code silu(gate) * up}. */
    SWIGLU,

    /** The GeGLU feed-forward activation: {@code gelu(gate) * up}. */
    GEGLU,

    /** Residual addition. */
    RESIDUAL_ADD,

    /**
     * Adding a learned bias vector to a projection.
     *
     * <p>Separate from {@link #RESIDUAL_ADD} despite the identical arithmetic: its right-hand side
     * is a model weight rather than a branch result, and it sits after a projection rather than at
     * the end of a block. Qwen2 and Qwen2-MoE carry QKV biases; Llama does not.
     */
    BIAS_ADD,

    /**
     * Separating a fused query/key/value projection into its three parts.
     *
     * <p>Phi3 projects all three with one matrix and reads the parts out of the result. A logical
     * operation rather than materialization, because it transforms a per-invocation activation
     * after the projection has run, where materialization is fixed once at load. A backend is still
     * free to fuse it into the projection and produce no intermediate.
     */
    SPLIT_FUSED_QKV,

    /**
     * Separating a fused gate/up feed-forward projection into its two halves.
     *
     * <p>{@link #SPLIT_FUSED_QKV}'s twin, for the second fused projection Phi3 has, added by the
     * same decision. A backend may fuse the projection, the split and the {@link #SWIGLU} into one
     * kernel; the program still says the split happens.
     */
    SPLIT_GATE_UP,

    /**
     * Multiplying every element by one scalar from the model configuration.
     *
     * <p>Granite's µP factors and Gemma's embedding scale. Not folded into the neighbouring
     * operation: a {@code MatVec} with an optional output scale would give every family a parameter
     * only one of them uses.
     */
    SCALE,

    /** Reading token embeddings out of the embedding table. */
    EMBEDDING_LOOKUP,

    /** Projecting the final hidden state onto the vocabulary. */
    VOCAB_PROJECTION,

    /**
     * Smoothly bounding the logits: {@code cap * tanh(x / cap)}.
     *
     * <p>Forward computation rather than generation policy, so its placement is the same whether
     * sampling runs on the host or the device. A model without soft-capping omits it.
     */
    LOGIT_SOFT_CAP,

    /** Choosing which experts a token is routed to, and with what weight. */
    MOE_ROUTER,

    /** One expert's gated feed-forward pass, indexed out of stacked expert tensors. */
    EXPERT_FEED_FORWARD,

    /**
     * Accumulating a branch into the residual stream with a scalar weight.
     *
     * <p>Separate from {@link #RESIDUAL_ADD}: in a mixture of experts this <i>is</i> the residual
     * connection, and the order these run in is the order the sum is formed.
     */
    WEIGHTED_ACCUMULATE,

    /** Greedy selection of the highest-scoring token. */
    ARG_MAX,

    /** Stochastic selection, with its parameters supplied per invocation. */
    SAMPLE
}
