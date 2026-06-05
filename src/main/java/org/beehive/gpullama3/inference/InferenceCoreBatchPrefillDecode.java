package org.beehive.gpullama3.inference;

import org.beehive.gpullama3.auxiliary.Parallel;
import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.standard.StandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tensor.standard.ArrayFloatTensor;
import org.beehive.gpullama3.tensor.standard.FloatTensor;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlanBatchPrefillDecode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.lang.foreign.MemorySegment;

/**
 * Low-level forward passes for the batched prefill/decode inference path (Phase 3/4).
 *
 * <p>Parallel to {@link InferenceCoreWithPrefillDecode} — does NOT modify it.</p>
 *
 * <p>Provides three operations:</p>
 * <ul>
 *   <li>{@link #batchForwardJavaPrefill} — CPU batch prefill: processes a chunk of
 *       prompt tokens in one pass using batch matmul, avoiding redundant weight
 *       traversals. Only the KV cache is populated; logits are intentionally omitted.</li>
 *   <li>{@link #batchForwardTornadoVMPrefill} — GPU batch prefill: copies batch embeddings
 *       into device-visible state buffers then runs the batch activation + layer graphs.</li>
 *   <li>{@link #forwardTornadoVMDecode} — GPU decode: copies the decode token embedding
 *       then runs the decode activation + layer + logits graphs.</li>
 * </ul>
 */
public final class InferenceCoreBatchPrefillDecode {

    private InferenceCoreBatchPrefillDecode() {}

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

            Parallel.parallelFor(0, batchSize, b ->
                    InferenceCore.rmsnorm(xb[b], x[b], weights.rms_att_weight[layer], 0, dim, config.rmsNormEps()));

            weights.wq[l].matmul(batchSize, xb, q,   dim,   dim);
            weights.wk[l].matmul(batchSize, xb, k,   kvDim, dim);
            weights.wv[l].matmul(batchSize, xb, v,   kvDim, dim);

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

            for (int b = 0; b < batchSize; b++) {
                final int pos_b  = startPos + b;
                final int bFinal = b;
                Parallel.parallelFor(0, config.numberOfHeads(), h -> {
                    int qOffset   = h * headSize;
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

            weights.wo[l].matmul(batchSize, xb, xb2, dim, dim);

            Parallel.parallelFor(0, batchSize, b -> {
                x[b].addInPlace(xb2[b]);
                InferenceCore.rmsnorm(xb[b], x[b], weights.rms_ffn_weight[layer], 0, dim, config.rmsNormEps());
            });

            weights.w1[l].matmul(batchSize, xb, hb,  config.hiddenDim(), dim);
            weights.w3[l].matmul(batchSize, xb, hb2, config.hiddenDim(), dim);

            Parallel.parallelFor(0, batchSize, b -> {
                hb[b].mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));
                hb[b].multiplyInPlace(hb2[b]);
            });

            weights.w2[l].matmul(batchSize, hb, xb, dim, config.hiddenDim());

            Parallel.parallelFor(0, batchSize, b -> x[b].addInPlace(xb[b]));
        }
        // Final RMSNorm and vocab projection intentionally omitted —
        // logits are not needed for any token in a prefill batch.
    }

    private static final int Q8_0_BLOCK_SIZE  = 32;
    private static final int Q8_0_BLOCK_BYTES = 34;

    /**
     * GPU batched prefill forward pass (Phase 4).
     *
     * <p>Copies {@code chunkSize} token embeddings into device-visible state buffers,
     * then delegates graph execution to the plan.</p>
     *
     * @param model     the LLaMA model
     * @param state     mutable inference state (must be a {@link LlamaState})
     * @param tokens    token ids for this chunk
     * @param startPos  sequence position of {@code tokens[0]}
     * @param chunkSize number of tokens in this chunk
     * @param plan      the batched prefill/decode GPU plan
     */
    public static void batchForwardTornadoVMPrefill(Model model, State state, int[] tokens, int startPos,
            int chunkSize, TornadoVMMasterPlanBatchPrefillDecode plan) {
        final Configuration config = model.configuration();
        final TornadoWeights weights = (TornadoWeights) model.weights();
        final LlamaState llamaState = (LlamaState) state;

        llamaState.batchStartPosHolder.set(0, startPos);

        switch (weights.getWeightType()) {
            case F16 -> {
                MemorySegment embTable = weights.getTokenEmbeddingTable().asHalfFloatArray().getSegment();
                long dimBytes = (long) config.dim() * Short.BYTES;
                for (int b = 0; b < chunkSize; b++) {
                    MemorySegment.copy(embTable, (long) tokens[b] * dimBytes,
                            llamaState.embeddingXBatch.getSegment(), (long) b * dimBytes, dimBytes);
                }
            }
            case Q8_0 -> {
                var embTable = weights.getTokenEmbeddingTable().asByteArray();
                int dim = config.dim();
                int blocksPerRow = (dim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
                for (int b = 0; b < chunkSize; b++) {
                    int tokenId = tokens[b];
                    for (int j = 0; j < dim; j++) {
                        int blockByteOffset = (tokenId * blocksPerRow + j / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES;
                        float scale = embTable.getHalfFloat(blockByteOffset).getFloat32();
                        float quant = embTable.get(blockByteOffset + 2 + j % Q8_0_BLOCK_SIZE);
                        llamaState.wrapXBatch.set(b * dim + j, quant * scale);
                    }
                }
            }
            default -> throw new IllegalArgumentException("Unsupported weight type: " + weights.getWeightType());
        }

        plan.tornadoVMForwardBatchPrefill();
    }

    /**
     * GPU decode forward pass (Phase 4).
     *
     * <p>Copies the token embedding into device-visible state, then delegates
     * graph execution to the plan.</p>
     *
     * @param model    the LLaMA model
     * @param state    mutable inference state
     * @param token    current token id
     * @param position sequence position
     * @param plan     the batched prefill/decode GPU plan
     * @return logits array for token sampling
     */
    public static FloatArray forwardTornadoVMDecode(Model model, State state, int token, int position,
            TornadoVMMasterPlanBatchPrefillDecode plan) {
        final Configuration config = model.configuration();
        final TornadoWeights weights = (TornadoWeights) model.weights();

        switch (weights.getWeightType()) {
            case F16 -> {
                MemorySegment embTable = weights.getTokenEmbeddingTable().asHalfFloatArray().getSegment();
                MemorySegment.copy(embTable, (long) token * config.dim() * Short.BYTES,
                        state.embeddingX.getSegment(), 0L, (long) config.dim() * Short.BYTES);
            }
            case Q8_0 -> {
                MemorySegment embTable = weights.getTokenEmbeddingTable().asByteArray().getSegment();
                int blocksPerToken = (config.dim() + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
                long bytesPerToken = (long) blocksPerToken * Q8_0_BLOCK_BYTES;
                MemorySegment.copy(embTable, (long) token * bytesPerToken,
                        state.embeddingX.getSegment(), 0L, bytesPerToken);
            }
            default -> throw new IllegalArgumentException("Unsupported weight type: " + weights.getWeightType());
        }

        return plan.tornadoVMForwardDecode(position);
    }
}
