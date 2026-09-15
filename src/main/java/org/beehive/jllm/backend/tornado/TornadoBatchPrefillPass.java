package org.beehive.gpullama3.backend.tornado;

import java.lang.foreign.MemorySegment;
import org.beehive.gpullama3.inference.Logits;
import org.beehive.gpullama3.inference.state.Qwen2MoEState;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;

/** The accelerated <b>batched prefill</b> pass and its decode step. */
public final class TornadoBatchPrefillPass {

    private static final int Q8_0_BLOCK_SIZE = 32;
    private static final int Q8_0_BLOCK_BYTES = 34;

    private TornadoBatchPrefillPass() {}

    /**
     * Stages {@code chunkSize} token embeddings into the session's device batch carrier, then runs
     * the batch activation and layer graphs. The logits graph is skipped: no token in a prefill
     * batch needs its logits.
     *
     * @param model the model
     * @param state the session's state
     * @param tokens token ids for this chunk
     * @param startPos sequence position of {@code tokens[0]}
     * @param chunkSize number of tokens in this chunk
     * @param plan the batched prefill/decode GPU plan
     */
    public static void batchPrefill(
            Model model,
            State state,
            int[] tokens,
            int startPos,
            int chunkSize,
            TornadoVMMasterPlanBatchPrefillDecode plan) {
        final Configuration config = model.configuration();
        final TornadoWeights weights = (TornadoWeights) model.weights();

        state.workspace.batchStartPosHolder.set(0, startPos);
        // The kernels launch a fixed batchSize rows; this tells them how many are real, so the
        // padding rows do not rotate, do not write KV, and cannot run past this layer's KV slice.
        state.workspace.batchStartPosHolder.set(1, chunkSize);
        // The KV slot travels with the chunk, the way it travels with the position on the
        // single-token path. Forgetting it would address slot 0 — another session's KV.
        if (state.workspace.batchStartPosHolder.getSize() > 2) {
            state.workspace.batchStartPosHolder.set(2, state.kvSlot);
        }
        if (state instanceof Qwen2MoEState moeState
                && moeState.workspace.activeBatchSizeHolder != null) {
            moeState.workspace.activeBatchSizeHolder.set(0, chunkSize);
        }

        switch (weights.dataType()) {
            case F16 -> {
                MemorySegment embTable =
                        weights.getTokenEmbeddingTable().asHalfFloatArray().getSegment();
                long dimBytes = (long) config.dim() * Short.BYTES;
                for (int b = 0; b < chunkSize; b++) {
                    MemorySegment.copy(
                            embTable,
                            (long) tokens[b] * dimBytes,
                            state.workspace.embeddingXBatch.getSegment(),
                            (long) b * dimBytes,
                            dimBytes);
                }
            }
            case Q8_0 -> {
                var embTable = weights.getTokenEmbeddingTable().asByteArray();
                int dim = config.dim();
                int blocksPerRow = (dim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
                for (int b = 0; b < chunkSize; b++) {
                    int tokenId = tokens[b];
                    for (int j = 0; j < dim; j++) {
                        int blockByteOffset =
                                (tokenId * blocksPerRow + j / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES;
                        float scale = embTable.getHalfFloat(blockByteOffset).getFloat32();
                        float quant = embTable.get(blockByteOffset + 2 + j % Q8_0_BLOCK_SIZE);
                        state.workspace.wrapXBatch.set(b * dim + j, quant * scale);
                    }
                }
            }
            default ->
                    throw new IllegalArgumentException(
                            "Unsupported weight type: " + weights.dataType());
        }

        plan.tornadoVMForwardBatchPrefill();
    }

    /**
     * The decode step of the batched path: stage one token's embedding, then run the decode
     * activation, layer and logits graphs.
     *
     * <p>Returns the neutral {@link Logits} view over the array the plan produced. The logits stay
     * <b>device-resident</b> exactly as before — the view reads the same {@code FloatArray} in
     * place, and no readback, copy or synchronization is added or removed.
     *
     * @param model the model
     * @param state the session's state
     * @param token current token id
     * @param position sequence position
     * @param plan the batched prefill/decode GPU plan
     * @return the logits this invocation produced, for sampling
     */
    public static Logits decode(
            Model model,
            State state,
            int token,
            int position,
            TornadoVMMasterPlanBatchPrefillDecode plan) {
        final Configuration config = model.configuration();
        final TornadoWeights weights = (TornadoWeights) model.weights();

        switch (weights.dataType()) {
            case F16 -> {
                MemorySegment embTable =
                        weights.getTokenEmbeddingTable().asHalfFloatArray().getSegment();
                MemorySegment.copy(
                        embTable,
                        (long) token * config.dim() * Short.BYTES,
                        state.workspace.embeddingX.getSegment(),
                        0L,
                        (long) config.dim() * Short.BYTES);
            }
            case Q8_0 -> {
                MemorySegment embTable =
                        weights.getTokenEmbeddingTable().asByteArray().getSegment();
                int blocksPerToken = (config.dim() + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
                long bytesPerToken = (long) blocksPerToken * Q8_0_BLOCK_BYTES;
                MemorySegment.copy(
                        embTable,
                        (long) token * bytesPerToken,
                        state.workspace.embeddingX.getSegment(),
                        0L,
                        bytesPerToken);
            }
            default ->
                    throw new IllegalArgumentException(
                            "Unsupported weight type: " + weights.dataType());
        }

        return state.workspace.logitsView(plan.tornadoVMForwardDecode(position));
    }
}
