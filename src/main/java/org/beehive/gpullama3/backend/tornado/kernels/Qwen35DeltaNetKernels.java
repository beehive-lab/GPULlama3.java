package org.beehive.gpullama3.backend.tornado.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Device kernels for the Gated Delta Net mixer — the recurrent three quarters of a {@code qwen35}
 * stack.
 *
 * <h2>Every kernel here is one lane, written as a function</h2>
 *
 * <p>Each kernel body is a static method taking an explicit lane index, and the kernel itself is a
 * two-line wrapper that passes {@code context.globalIdx}. That is not decoration: a method taking
 * {@link KernelContext} cannot be called on the host, so a kernel written directly against it can
 * only be checked by running it, on a device, inside a model. Lifting the arithmetic out means
 * {@code Qwen35DeltaNetKernelParityTest} can run every lane on the host and compare it to {@code
 * CpuOperations} element by element — which is the check that catches an indexing mistake, and the
 * one that is otherwise impossible to write.
 *
 * <p>It costs nothing at run time. TornadoVM inlines the helper.
 *
 * <h2>Why no reductions and no barriers</h2>
 *
 * <p>The delta rule looks like it needs one — it is a matrix-vector product per head — and it does
 * not. Give a lane one <b>value column</b> {@code j} of a head's state and every quantity it needs
 * is its own: the decayed column, the prediction {@code Sᵀk} for that column, the correction, the
 * rank-one update to that column, and the readout {@code Sᵀq} for that column. Nothing is shared
 * between lanes, so there is no barrier, no local memory, and no cross-lane reduction.
 *
 * <p>The state layout makes that coalesced rather than merely correct. State is {@code (h * S + i)
 * * S + j} — key row {@code i}, value column {@code j} — so at each step of the inner loop the
 * {@code S} lanes of a head read {@code S} consecutive floats. Per-lane the access is strided; what
 * a GPU is paid for is the access being contiguous <i>across</i> lanes, and it is.
 *
 * <p>This is the same layout the host path uses, deliberately. Two layouts would mean the parity
 * test compares a transpose against a transpose and proves nothing about either.
 */
public final class Qwen35DeltaNetKernels {

    private Qwen35DeltaNetKernels() {}

    // ---- causal convolution --------------------------------------------------

    /**
     * One channel of a depthwise causal convolution, advancing that channel's window.
     *
     * <p>Depthwise means the channels never mix, so a channel is a lane and there is nothing to
     * reduce. The window is state and this owns advancing it; each lane touches only its own slice
     * of it, so the in-place shift races with nothing.
     *
     * <p>Both the kernel and the window are channel-major — a channel's taps are contiguous, oldest
     * first — matching how GGUF stores {@code ssm_conv1d} and how the host path reads it.
     *
     * <p>{@code windowOffset} is where this layer's window starts. Every recurrent layer's window
     * lives in one array — a device buffer per layer would be 48 of them to transfer and persist —
     * so the layer is an offset rather than a separate allocation. It is a parameter and not
     * derived from anything, because deriving it would mean the kernel knowing how many layers
     * there are.
     *
     * @param channel the lane: which of {@code channels} this call computes
     */
    static void causalConv1dLane(
            FloatArray input,
            FloatArray weight,
            FloatArray window,
            FloatArray out,
            int kernel,
            int windowOffset,
            int channel) {
        int history = kernel - 1;
        int wBase = channel * kernel;
        int hBase = windowOffset + channel * history;
        float x = input.get(channel);

        float sum = 0.0f;
        for (int t = 0; t < history; t++) {
            sum += weight.get(wBase + t) * window.get(hBase + t);
        }
        sum += weight.get(wBase + history) * x;
        out.set(channel, sum);

        for (int t = 0; t + 1 < history; t++) {
            window.set(hBase + t, window.get(hBase + t + 1));
        }
        if (history > 0) {
            window.set(hBase + history - 1, x);
        }
    }

    /** One lane per channel. */
    public static void causalConv1d(
            KernelContext context,
            FloatArray input,
            FloatArray weight,
            FloatArray window,
            FloatArray out,
            int channels,
            int kernel,
            int windowOffset) {
        int channel = context.globalIdx;
        if (channel >= channels) {
            return;
        }
        causalConv1dLane(input, weight, window, out, kernel, windowOffset, channel);
    }

    /**
     * SiLU over the convolved result, in place. One lane per channel.
     *
     * <p>A separate kernel rather than folded into the convolution: the host path applies it to the
     * whole vector after the convolution, and keeping the boundary in the same place keeps the two
     * comparable term by term.
     */
    public static void siluInPlace(KernelContext context, FloatArray values, int count) {
        int i = context.globalIdx;
        if (i >= count) {
            return;
        }
        float v = values.get(i);
        values.set(i, v / (1.0f + TornadoMath.exp(-v)));
    }

    // ---- L2 normalization ----------------------------------------------------

    /**
     * One head scaled to unit length.
     *
     * <p>A lane per head rather than a workgroup per head with a reduction: a head is 128 wide
     * here and there are 16 of them, so the reduction would cost more in barriers than the serial
     * loop costs in arithmetic. Epsilon floors the divisor rather than being added under the root,
     * which is what {@code ggml_l2_norm} does and what the host path was written against.
     *
     * @param head the lane
     */
    static void l2NormLane(FloatArray values, int headDim, float eps, int head) {
        int base = head * headDim;
        float ss = 0.0f;
        for (int i = 0; i < headDim; i++) {
            float v = values.get(base + i);
            ss += v * v;
        }
        float inv = 1.0f / TornadoMath.max(TornadoMath.sqrt(ss), eps);
        for (int i = 0; i < headDim; i++) {
            values.set(base + i, values.get(base + i) * inv);
        }
    }

    /** One lane per head. */
    public static void l2NormPerHead(
            KernelContext context, FloatArray values, int heads, int headDim, float eps) {
        int head = context.globalIdx;
        if (head >= heads) {
            return;
        }
        l2NormLane(values, headDim, eps, head);
    }

    // ---- the decay and write strengths ---------------------------------------

    /**
     * One value head's decay and write strength, from their raw projections.
     *
     * <p>{@code beta = sigmoid(betaRaw)} and {@code decay = exp(a * softplus(alphaRaw + dtBias))},
     * both in place. Folded into one lane because they are consumed together and each is a handful
     * of operations on one element — two kernels over 48 elements would be two launches to save
     * nothing.
     *
     * <p>{@code a} is {@code -exp(A_log)} as the file stores it, so the product is a log decay and
     * its exponential lands in {@code (0, 1)}: the state is forgotten, never amplified.
     */
    static void decayAndBetaLane(
            FloatArray alpha, FloatArray beta, FloatArray dtBias, FloatArray a, int head) {
        float raw = beta.get(head);
        beta.set(head, 1.0f / (1.0f + TornadoMath.exp(-raw)));

        float biased = alpha.get(head) + dtBias.get(head);
        // softplus, guarded the way the host path guards it: for a large argument log1p(exp(x))
        // is x to within float precision, and exp(x) alone would overflow.
        float softplus = biased > 20.0f ? biased : TornadoMath.log(1.0f + TornadoMath.exp(biased));
        alpha.set(head, TornadoMath.exp(a.get(head) * softplus));
    }

    /** One lane per value head. */
    public static void decayAndBeta(
            KernelContext context,
            FloatArray alpha,
            FloatArray beta,
            FloatArray dtBias,
            FloatArray a,
            int valueHeads) {
        int head = context.globalIdx;
        if (head >= valueHeads) {
            return;
        }
        decayAndBetaLane(alpha, beta, dtBias, a, head);
    }

    // ---- the delta rule ------------------------------------------------------

    /**
     * One value column of one head: decay, correct, accumulate, read back.
     *
     * <pre>
     *   s[i]  = S[i][j] * decay          // forget
     *   sk    = Σ s[i]·k[i]              // what the state already predicts for this column
     *   d     = (v[j] - sk) * beta       // the part it does not
     *   S[i][j] = s[i] + k[i]·d          // write the correction
     *   out[j]  = Σ S[i][j]·q[i]         // read it back
     * </pre>
     *
     * <p>Two passes over the column rather than one, because the readout must see the updated
     * state and the update needs the whole prediction first. Both passes are over the same 128
     * values, so the second finds them in cache.
     *
     * <p><b>A value head reads key head {@code h % keyHeads}.</b> Modulo, not division: the
     * reference repeats the key heads by tiling. Dividing pairs every value head with the wrong
     * key and produces fluent, slowly degrading output — the defect this port already made once on
     * the host, which is why it is stated here rather than left to the caller.
     *
     * <p>{@code stateOffset} is where this layer's state starts, for the reason the convolution's
     * window offset exists: every recurrent layer's state lives in one array, and 48 separate
     * device buffers would be 48 transfers to arrange and keep resident.
     *
     * @param lane {@code head * stateDim + column}
     */
    static void deltaRuleLane(
            FloatArray q,
            FloatArray k,
            FloatArray v,
            FloatArray decay,
            FloatArray beta,
            FloatArray state,
            FloatArray out,
            int keyHeads,
            int stateDim,
            int stateOffset,
            int lane) {
        int head = lane / stateDim;
        int column = lane - head * stateDim;

        int stateBase = stateOffset + head * stateDim * stateDim;
        int kvBase = (head % keyHeads) * stateDim;
        int valueBase = head * stateDim;

        float g = decay.get(head);
        float b = beta.get(head);

        float prediction = 0.0f;
        for (int i = 0; i < stateDim; i++) {
            int index = stateBase + i * stateDim + column;
            float decayed = state.get(index) * g;
            state.set(index, decayed);
            prediction += decayed * k.get(kvBase + i);
        }

        float correction = (v.get(valueBase + column) - prediction) * b;

        float readout = 0.0f;
        for (int i = 0; i < stateDim; i++) {
            int index = stateBase + i * stateDim + column;
            float updated = state.get(index) + k.get(kvBase + i) * correction;
            state.set(index, updated);
            readout += updated * q.get(kvBase + i);
        }
        out.set(valueBase + column, readout);
    }

    /** One lane per (value head, value column) — {@code valueHeads * stateDim} of them. */
    public static void deltaRule(
            KernelContext context,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            FloatArray decay,
            FloatArray beta,
            FloatArray state,
            FloatArray out,
            int valueHeads,
            int keyHeads,
            int stateDim,
            int stateOffset) {
        int lane = context.globalIdx;
        if (lane >= valueHeads * stateDim) {
            return;
        }
        deltaRuleLane(q, k, v, decay, beta, state, out, keyHeads, stateDim, stateOffset, lane);
    }

    // ---- the gated norm ------------------------------------------------------

    /**
     * One head of {@code rms_norm(values, weight) * silu(gate)}, in place on {@code values}.
     *
     * <p>A lane per head, for the reason {@link #l2NormLane} is: the head is narrow and there are
     * few of them. The learned scale is one head wide and shared by every head.
     */
    static void gatedNormLane(
            FloatArray values,
            FloatArray gate,
            FloatArray weight,
            int headDim,
            float eps,
            int head) {
        int base = head * headDim;

        float ss = 0.0f;
        for (int i = 0; i < headDim; i++) {
            float v = values.get(base + i);
            ss += v * v;
        }
        float inv = 1.0f / TornadoMath.sqrt(ss / headDim + eps);

        for (int i = 0; i < headDim; i++) {
            float z = gate.get(base + i);
            float silu = z / (1.0f + TornadoMath.exp(-z));
            values.set(base + i, weight.get(i) * (inv * values.get(base + i)) * silu);
        }
    }

    /** One lane per head. */
    public static void gatedNormPerHead(
            KernelContext context,
            FloatArray values,
            FloatArray gate,
            FloatArray weight,
            int heads,
            int headDim,
            float eps) {
        int head = context.globalIdx;
        if (head >= heads) {
            return;
        }
        gatedNormLane(values, gate, weight, headDim, eps, head);
    }
}
