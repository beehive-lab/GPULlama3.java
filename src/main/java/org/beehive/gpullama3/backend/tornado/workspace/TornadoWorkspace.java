package org.beehive.gpullama3.backend.tornado.workspace;

import org.beehive.gpullama3.inference.Logits;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

/**
 * The device arrays one session executes against — activations, attention scratch, the key/value
 * views, the control and result carriers, and the batched-prefill workspace.
 *
 * <p><b>Not neutral, and not an interface.</b> Nothing outside {@code backend.tornado} names this
 * type. A neutral interface carrying {@code FloatArray} accessors would be the same violation
 * wearing a different name, which is why there is no property bag, no {@code Object}-typed
 * workspace, and no widened accessor.
 *
 * <p><b>Identity is fixed for its life</b> [C1]: a captured CUDA graph replays against the
 * addresses it captured, so nothing here is reallocated while a plan exists. A shared workspace
 * belongs to its binding domain and is borrowed under that domain's invocation lock.
 *
 * <p>Fields, not accessors, exactly as before: the per-token path reads them directly and must not
 * gain a call.
 */
public final class TornadoWorkspace {

    public FloatArray wrapAttSplit;
    public FloatArray
            wrapLogits; // FloatArray wrapper for the logits tensor, compatible with TornadoVM for
    // GPU execution.
    public FloatArray
            wrapXb; // FloatArray wrapper for xb (residual branch activation), optimized for
    // TornadoVM usage.
    public FloatArray
            wrapXb2; // FloatArray wrapper for xb2, another residual buffer to aid in computations
    // with TornadoVM.
    public FloatArray
            wrapHb; // FloatArray wrapper for hb (hidden dimension buffer for FFN), optimized for
    // TornadoVM.
    public FloatArray wrapHb2; // FloatArray wrapper for hb2, additional hidden buffer for FFN, for
    // compatibility with TornadoVM.
    public FloatArray
            wrapX; // FloatArray wrapper for the current activation tensor, optimized for TornadoVM.
    public FloatArray wrapQ; // FloatArray wrapper for the query tensor, optimized for TornadoVM.
    public FloatArray wrapK; // FloatArray wrapper for the key tensor, optimized for TornadoVM.
    public FloatArray wrapV; // FloatArray wrapper for the value tensor, optimized for TornadoVM.
    public FloatArray
            wrapAtt; // FloatArray wrapper for the attention scores, optimized for TornadoVM.
    public FloatArray
            wrapKeyCache; // FloatArray wrapper for the key cache, optimized for TornadoVM.
    public FloatArray
            wrapValueCache; // FloatArray wrapper for the value cache, optimized for TornadoVM.
    public HalfFloatArray
            wrapKeyCacheFP16; // Optional half-precision key cache (see USE_FP16_KV); null unless
    // enabled.
    public HalfFloatArray
            wrapValueCacheFP16; // Optional half-precision value cache (see USE_FP16_KV); null
    // unless enabled.
    public IntArray positionHolder;
    public IntArray wrapBlockTable;
    public TornadoNativeArray embeddingX;
    public HalfFloatArray
            wrapXbFP16; // FloatArray wrapper for xb (residual branch activation), optimized for
    // TornadoVM usage.
    public FloatArray
            temp; // Temporary buffer for intermediate calculations, size adjusted for local
    // workgroup size.
    public FloatArray
            tempFFN; // Temporary buffer for feed-forward network calculations, size adjusted for
    // local workgroup size.
    public FloatArray
            tempLogits; // Temporary buffer for logits calculations, size adjusted for local
    // workgroup size.
    public HalfFloatArray wrapXFP16;
    public HalfFloatArray embeddingXBatch; // B × dim  (FP16 input)
    public FloatArray wrapXBatch; // B × dim  (live activations / Q8_0 dequant)
    public HalfFloatArray wrapXbFP16Batch; // B × dim  (RMSNorm output, FP16)
    public FloatArray wrapQBatch; // B × qDim (Q projection)
    public FloatArray wrapKBatch; // B × kvDim
    public FloatArray wrapVBatch; // B × kvDim
    public FloatArray wrapXbBatch; // B × qDim  (attention output)
    public FloatArray wrapHbBatch; // B × hiddenDim
    public FloatArray attnScaleBatch; // B        (per-token RMS scale, attn)
    public FloatArray ffnScaleBatch; // B        (per-token RMS scale, FFN)
    public IntArray batchStartPosHolder; // 1      (start position of chunk)
    public HalfFloatArray normedXFFNFP16;
    public FloatArray ffnGateResult;
    public FloatArray ffnUpResult;
    public HalfFloatArray xbFP16Batch;
    public HalfFloatArray attnOutFP16;
    public FloatArray woOut;
    public HalfFloatArray wrapHbFP16Batch;
    public FloatArray w2Out;
    public FloatArray qkvResultBatch; // B × (dim + 2*kvDim), packed [q|k|v] rows
    public FloatArray gateUpResultBatch; // B × 2*hiddenDim, packed [gate|up] rows

    // ── Family-specific device arrays ──────────────────────────────────────────────
    //
    // Qwen3, Phi3, Gemma4 and Qwen2-MoE each declared these on their own State subtype, which is
    // why those four were Rule 1 entries after the base class stopped being one. They are device
    // arrays like every other field here; that one family uses a buffer and another does not is not
    // a reason for it to live somewhere else.
    public FloatArray wrapPerLayerInputs;
    public FloatArray wrapPerLayerProjScratch;
    public FloatArray wrapPerLayerGate;
    public FloatArray wrapPerLayerOut;
    public FloatArray wrapPerLayerTokenEmbedRow;
    public FloatArray tempPostAttn;
    public FloatArray tempPostFfn;
    public FloatArray tempPostPle;
    public FloatArray wrapQkv; // TornadoVM wrapper for QKV buffer
    public FloatArray wrapHbG; // TornadoVM wrapper for gate states
    public FloatArray wrapHbU; // TornadoVM wrapper for up states
    public FloatArray wrapRouterLogits;
    public IntArray wrapSelectedExperts;
    public FloatArray wrapRoutingWeights;
    public FloatArray wrapExpertGate;
    public FloatArray wrapSharedGate;
    public FloatArray wrapSharedOutput;
    public FloatArray wrapRouterLogitsBatch;
    public IntArray activeBatchSizeHolder;
    public IntArray wrapSelectedExpertsBatch;
    public FloatArray wrapRoutingWeightsBatch;
    public IntArray wrapGroupedAssignmentIds;
    public IntArray wrapGroupedPositionByAssignment;
    public FloatArray wrapGroupedExpertHidden;
    public FloatArray wrapGroupedExpertDown;
    public FloatArray wrapSharedHiddenBatch;
    public FloatArray wrapSharedWeightBatch;
    public FloatArray tempQcur;
    public FloatArray tempKcur;

    /** One slot: the token an on-device sampler wrote, read by the loop. */
    public IntArray sampledToken = new IntArray(1);

    /**
     * The token an on-device sampler wrote, as an {@code int}.
     *
     * <p>The generation loop needs the value, not the carrier. Reading {@code
     * workspace.sampledToken.get(0)} from the loop is what made {@code TokenGenerationLoop} name
     * {@code IntArray}; this returns the same slot without naming it.
     */
    public int deviceSampledToken() {
        return sampledToken.get(0);
    }

    private FloatArray logitsViewTarget;
    private Logits logitsView;

    /**
     * {@code array} as the neutral {@link Logits} view.
     *
     * <p>A one-element identity cache, and that is the whole of it. The view is always over the
     * array the plan actually returned — a legacy plan hands back this workspace's {@code
     * wrapLogits}, a lowered one hands back the session's own copy — so the identity is checked
     * rather than assumed. Steady state allocates nothing: a decode loop asks for the same array
     * every token, which is why this is not built fresh per call.
     */
    public Logits logitsView(FloatArray array) {
        if (array != logitsViewTarget) {
            logitsViewTarget = array;
            logitsView = TornadoLogits.of(array);
        }
        return logitsView;
    }
}
