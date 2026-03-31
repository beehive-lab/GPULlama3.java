package org.beehive.gpullama3.inference;

import org.beehive.gpullama3.auxiliary.Parallel;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.standard.StandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tensor.standard.ArrayFloatTensor;
import org.beehive.gpullama3.tensor.standard.FloatTensor;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlanWithPrefillDecode;

import java.lang.foreign.MemorySegment;

/**
 * Low-level forward passes for the prefill/decode separated inference path.
 *
 * <p>Parallel to {@link InferenceCore} — does NOT modify it.</p>
 *
 * <p>The key addition is {@link #forwardJavaPrefill}, which runs a full
 * transformer forward pass but skips the final RMSNorm and vocabulary
 * projection (wcls matmul). This is correct for all prefill positions
 * because their logits are discarded anyway; only the KV-cache update
 * matters. Skipping the projection saves one large matmul
 * (vocabularySize × dim) per prefill token.</p>
 */
public final class InferenceCoreWithPrefillDecode {

    private InferenceCoreWithPrefillDecode() {}

    /**
     * Prefill-only forward pass for LLaMA (CPU, FP32 weights).
     *
     * <p>Identical to {@link InferenceCore#forwardJava} except the final
     * RMSNorm and vocabulary projection are omitted. The KV cache is
     * populated correctly at {@code position}.</p>
     *
     * @param model    the LLaMA model (must carry {@link StandardWeights})
     * @param state    mutable inference state (KV cache, activations …)
     * @param token    input token id
     * @param position sequence position being processed
     */
    public static void forwardJavaPrefill(Model model, State state, int token, int position) {
        final Configuration config = model.configuration();
        final StandardWeights weights = (StandardWeights) model.weights();
        int dim = config.dim();
        int headSize = config.headSize();
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        int kvMul = config.numberOfHeads() / config.numberOfKeyValueHeads();
        float sqrtHeadSize = (float) Math.sqrt(headSize);

        // Token embedding
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

        // Transformer layers
        for (int l = 0; l < config.numberOfLayers(); l++) {
            // Attention RMSNorm
            InferenceCore.rmsnorm(state.xb, state.x, weights.rms_att_weight[l], 0, dim, config.rmsNormEps());

            // QKV projections
            weights.wq[l].matmul(state.xb, state.q, dim, dim);
            weights.wk[l].matmul(state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(state.xb, state.v, kvDim, dim);

            // RoPE
            for (int i = 0; i < dim; i += 2) {
                int head_dim = i % headSize;
                float fcr = weights.freq_cis_real.getFloat(position * (headSize / 2) + (head_dim / 2));
                float fci = weights.freq_cis_imag.getFloat(position * (headSize / 2) + (head_dim / 2));
                int rotn = i < kvDim ? 2 : 1;
                for (int v = 0; v < rotn; v++) {
                    FloatTensor vec = v == 0 ? state.q : state.k;
                    float v0 = vec.getFloat(i);
                    float v1 = vec.getFloat(i + 1);
                    vec.setFloat(i, v0 * fcr - v1 * fci);
                    vec.setFloat(i + 1, v0 * fci + v1 * fcr);
                }
            }

            // KV cache update
            state.k.copyTo(0, state.keyCache[l], position * kvDim, kvDim);
            state.v.copyTo(0, state.valueCache[l], position * kvDim, kvDim);

            // Multi-head attention
            int curLayer = l;
            Parallel.parallelFor(0, config.numberOfHeads(), h -> {
                int qOffset = h * headSize;
                int attOffset = h * config.contextLength();

                for (int t = 0; t <= position; t++) {
                    int keyCacheOffset = t * kvDim + (h / kvMul) * headSize;
                    float score = state.q.dot(qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
                    score /= sqrtHeadSize;
                    state.att.setFloat(attOffset + t, score);
                }

                state.att.softmaxInPlace(attOffset, position + 1);

                int xbOffset = h * headSize;
                state.xb.fillInPlace(xbOffset, headSize, 0f);
                for (int t = 0; t <= position; t++) {
                    int vOffset = t * kvDim + (h / kvMul) * headSize;
                    float a = state.att.getFloat(attOffset + t);
                    state.xb.saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                }
            });

            // Attention output projection + residual
            weights.wo[l].matmul(state.xb, state.xb2, dim, dim);
            state.x.addInPlace(state.xb2);

            // FFN RMSNorm
            InferenceCore.rmsnorm(state.xb, state.x, weights.rms_ffn_weight[l], 0, dim, config.rmsNormEps());

            // FFN (SwiGLU)
            weights.w1[l].matmul(state.xb, state.hb, config.hiddenDim(), dim);
            weights.w3[l].matmul(state.xb, state.hb2, config.hiddenDim(), dim);
            state.hb.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));
            state.hb.multiplyInPlace(state.hb2);
            weights.w2[l].matmul(state.hb, state.xb, dim, config.hiddenDim());

            // FFN residual
            state.x.addInPlace(state.xb);
        }

        // Final RMSNorm and vocab projection intentionally omitted:
        // logits are not needed for prefill positions — only the KV cache matters.
    }

    /**
     * CPU batched prefill forward pass for LLaMA (Phase 3).
     *
     * <p>Processes {@code batchSize} prompt tokens simultaneously through all
     * transformer layers. For each layer, Q/K/V projections, output projection,
     * and FFN projections are computed via batch matmul
     * ({@link FloatTensor#matmul(int, FloatTensor[], FloatTensor[], int, int)}),
     * which parallelises over both output dimension and batch simultaneously.
     * Attention reuses {@code state.att} sequentially per token (parallel per
     * head within each token), keeping memory overhead minimal.</p>
     *
     * <p>The logits layer is intentionally omitted — only the KV cache matters
     * for prefill positions.</p>
     *
     * @param model     the LLaMA model (must carry {@link StandardWeights})
     * @param state     mutable inference state (KV cache, att buffer …)
     * @param tokens    input token ids, {@code tokens[b]} at position {@code startPos+b}
     * @param startPos  sequence position of {@code tokens[0]}
     * @param batchSize number of tokens in this chunk ({@code tokens.length})
     */
    public static void batchForwardJavaPrefill(Model model, State state, int[] tokens, int startPos, int batchSize) {
        final Configuration config = model.configuration();
        final StandardWeights weights = (StandardWeights) model.weights();
        int dim = config.dim();
        int headSize = config.headSize();
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        int kvMul = config.numberOfHeads() / config.numberOfKeyValueHeads();
        float sqrtHeadSize = (float) Math.sqrt(headSize);

        // ── Batch activation tensors (allocated once per chunk) ───────────────
        FloatTensor[] x   = new FloatTensor[batchSize];
        FloatTensor[] xb  = new FloatTensor[batchSize];
        FloatTensor[] xb2 = new FloatTensor[batchSize];
        FloatTensor[] q   = new FloatTensor[batchSize];
        FloatTensor[] k   = new FloatTensor[batchSize];
        FloatTensor[] v   = new FloatTensor[batchSize];
        FloatTensor[] hb  = new FloatTensor[batchSize];
        FloatTensor[] hb2 = new FloatTensor[batchSize];
        for (int b = 0; b < batchSize; b++) {
            x[b]   = ArrayFloatTensor.allocate(dim);
            xb[b]  = ArrayFloatTensor.allocate(dim);
            xb2[b] = ArrayFloatTensor.allocate(dim);
            q[b]   = ArrayFloatTensor.allocate(dim);
            k[b]   = ArrayFloatTensor.allocate(kvDim);
            v[b]   = ArrayFloatTensor.allocate(kvDim);
            hb[b]  = ArrayFloatTensor.allocate(config.hiddenDim());
            hb2[b] = ArrayFloatTensor.allocate(config.hiddenDim());
        }

        // ── Token embeddings ──────────────────────────────────────────────────
        Parallel.parallelFor(0, batchSize, b ->
                weights.token_embedding_table.copyTo(tokens[b] * dim, x[b], 0, dim));

        // ── Transformer layers ────────────────────────────────────────────────
        for (int l = 0; l < config.numberOfLayers(); l++) {
            final int layer = l;

            // Attention RMSNorm (parallel per b)
            Parallel.parallelFor(0, batchSize, b ->
                    InferenceCore.rmsnorm(xb[b], x[b], weights.rms_att_weight[layer], 0, dim, config.rmsNormEps()));

            // QKV projections — batch matmul parallelises over (dim × batchSize)
            weights.wq[l].matmul(batchSize, xb, q,   dim,   dim);
            weights.wk[l].matmul(batchSize, xb, k,   kvDim, dim);
            weights.wv[l].matmul(batchSize, xb, v,   kvDim, dim);

            // RoPE + KV cache store (parallel per b — different positions, no conflict)
            Parallel.parallelFor(0, batchSize, b -> {
                int pos = startPos + b;
                for (int i = 0; i < dim; i += 2) {
                    int head_dim = i % headSize;
                    float fcr = weights.freq_cis_real.getFloat(pos * (headSize / 2) + (head_dim / 2));
                    float fci = weights.freq_cis_imag.getFloat(pos * (headSize / 2) + (head_dim / 2));
                    int rotn = i < kvDim ? 2 : 1;
                    for (int vv = 0; vv < rotn; vv++) {
                        FloatTensor vec = vv == 0 ? q[b] : k[b];
                        float v0 = vec.getFloat(i);
                        float v1 = vec.getFloat(i + 1);
                        vec.setFloat(i,     v0 * fcr - v1 * fci);
                        vec.setFloat(i + 1, v0 * fci + v1 * fcr);
                    }
                }
                k[b].copyTo(0, state.keyCache[layer],   pos * kvDim, kvDim);
                v[b].copyTo(0, state.valueCache[layer],  pos * kvDim, kvDim);
            });

            // Attention — sequential per b (state.att is shared), parallel per head
            for (int b = 0; b < batchSize; b++) {
                final int pos_b   = startPos + b;
                final int bFinal  = b;
                Parallel.parallelFor(0, config.numberOfHeads(), h -> {
                    int qOffset  = h * headSize;
                    int attOffset = h * config.contextLength();

                    for (int t = 0; t <= pos_b; t++) {
                        int keyCacheOffset = t * kvDim + (h / kvMul) * headSize;
                        float score = q[bFinal].dot(qOffset, state.keyCache[layer], keyCacheOffset, headSize) / sqrtHeadSize;
                        state.att.setFloat(attOffset + t, score);
                    }
                    state.att.softmaxInPlace(attOffset, pos_b + 1);

                    int xbOffset = h * headSize;
                    xb[bFinal].fillInPlace(xbOffset, headSize, 0f);
                    for (int t = 0; t <= pos_b; t++) {
                        int vOffset = t * kvDim + (h / kvMul) * headSize;
                        float a = state.att.getFloat(attOffset + t);
                        xb[bFinal].saxpyInPlace(xbOffset, state.valueCache[layer], vOffset, headSize, a);
                    }
                });
            }

            // Output projection — batch matmul
            weights.wo[l].matmul(batchSize, xb, xb2, dim, dim);

            // Residual + FFN RMSNorm (parallel per b)
            Parallel.parallelFor(0, batchSize, b -> {
                x[b].addInPlace(xb2[b]);
                InferenceCore.rmsnorm(xb[b], x[b], weights.rms_ffn_weight[layer], 0, dim, config.rmsNormEps());
            });

            // FFN projections — batch matmul
            weights.w1[l].matmul(batchSize, xb, hb,  config.hiddenDim(), dim);
            weights.w3[l].matmul(batchSize, xb, hb2, config.hiddenDim(), dim);

            // SwiGLU (parallel per b)
            Parallel.parallelFor(0, batchSize, b -> {
                hb[b].mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));
                hb[b].multiplyInPlace(hb2[b]);
            });

            // w2 projection — batch matmul (output reuses xb)
            weights.w2[l].matmul(batchSize, hb, xb, dim, config.hiddenDim());

            // FFN residual (parallel per b)
            Parallel.parallelFor(0, batchSize, b -> x[b].addInPlace(xb[b]));
        }

        // Final RMSNorm and vocab projection intentionally omitted —
        // logits are not needed for any token in a prefill batch.
    }

    /**
     * GPU prefill-only forward pass for LLaMA (FP16, TornadoVM).
     *
     * <p>Copies the token embedding into {@code state.embeddingX} (same as
     * {@link InferenceCore#forwardTornadoVM}) then delegates to
     * {@link TornadoVMMasterPlanWithPrefillDecode#tornadoVMForwardPrefill},
     * which executes preprocessing + layer graphs but skips the logits graph.</p>
     *
     * @param model       the LLaMA model (must carry {@link TornadoWeights}, FP16 only)
     * @param state       mutable inference state
     * @param token       input token id
     * @param position    sequence position being processed
     * @param prefillPlan the prefill/decode plan wrapper
     * @throws UnsupportedOperationException if the model uses Q8_0 weights
     */
    public static void forwardTornadoVMPrefill(Model model, State state, int token, int position,
            TornadoVMMasterPlanWithPrefillDecode prefillPlan) {
        final Configuration configuration = model.configuration();
        final TornadoWeights weights = (TornadoWeights) model.weights();

        switch (weights.getWeightType()) {
            case F16 -> {
                MemorySegment tokenEmbeddings = weights.getTokenEmbeddingTable().asHalfFloatArray().getSegment();
                int bytes = Short.BYTES;
                MemorySegment.copy(tokenEmbeddings, (long) token * configuration.dim() * bytes,
                        state.embeddingX.getSegment(), 0, (long) configuration.dim() * bytes);
            }
            case Q8_0 -> throw new UnsupportedOperationException(
                    // TODO Phase 4: implement Q8_0 GPU batched prefill kernels
                    "GPU prefill/decode path not yet implemented for Q8_0 weights");
            default -> throw new IllegalArgumentException("Unsupported weight type: " + weights.getWeightType());
        }

        prefillPlan.tornadoVMForwardPrefill(position);
    }
}
