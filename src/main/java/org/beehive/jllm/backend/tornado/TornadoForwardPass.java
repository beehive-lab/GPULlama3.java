package org.beehive.gpullama3.backend.tornado;

import java.lang.foreign.MemorySegment;
import org.beehive.gpullama3.inference.Logits;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;

/**
 * The accelerated forward pass: stage the token's embedding, then run the plan.
 *
 * <p>Inside {@code backend.tornado}, where naming a plan and a device array is the job.
 */
public final class TornadoForwardPass {

    private TornadoForwardPass() {}

    /**
     * Performs the initial embedding lookup and triggers the TornadoVM accelerated forward pass for
     * an LLM token.
     *
     * <p>This method handles the first phase of processing a token through the transformer model:
     *
     * <ol>
     *   <li>Copies the token embedding from the model's embedding table to the state's buffer
     *   <li>Delegates the transformer layer processing to TornadoVM through the master plan
     * </ol>
     *
     * <p>The token embedding lookup happens on the CPU using {@link MemorySegment} operations,
     * while the subsequent transformer layers processing is offloaded to the accelerator through
     * TornadoVM for improved performance.
     *
     * @param model The Llama model containing weights and configuration parameters
     * @param state The current execution state holding input/output tensors and temporary buffers
     * @param token The input token ID to process
     * @param position The position of this token in the sequence context window
     * @param tornadoVMMasterPlan The execution plan for TornadoVM acceleration
     * @return the output logits for token prediction, as the neutral view
     */
    public static Logits forward(
            Model model,
            State state,
            int token,
            int position,
            TornadoVMMasterPlan tornadoVMMasterPlan) {
        if (tornadoVMMasterPlan
                instanceof
                org.beehive.gpullama3.backend.tornado.lowering.InvocationBoundary boundary) {
            return boundary.invoke(token, position).logits();
        }

        final Configuration configuration = model.configuration();
        final TornadoWeights weights = (TornadoWeights) model.weights();

        // The *embedding tensor's own* representation, not the model-wide weight type. They are
        // the same for a uniform F16 or Q8_0 file, but a K-quant file is mixed: Devstral's Q4_K_M
        // holds its blk.* weights as Q4_K and other tensors as Q6_K/Q8_0, so asking the model-wide
        // type here staged the wrong number of bytes -- or, before this, refused a model whose
        // embedding it could actually copy.
        switch (weights.getTokenEmbeddingTable().dataType()) {
            case F16 -> {
                MemorySegment tokenEmbeddings =
                        weights.getTokenEmbeddingTable().asHalfFloatArray().getSegment();
                int bytes = Short.BYTES;
                MemorySegment.copy(
                        tokenEmbeddings,
                        (long) token * configuration.dim() * bytes,
                        state.workspace.embeddingX.getSegment(),
                        0,
                        (long) configuration.dim() * bytes);
            }
            case Q8_0 -> {
                MemorySegment tokenEmbeddings =
                        weights.getTokenEmbeddingTable().asByteArray().getSegment();
                int blockSize = 32;
                int Q8_0_BLOCK_BYTES = 34; // 2 bytes scale + 32 bytes quants
                int blocksPerToken =
                        (configuration.dim() + blockSize - 1) / blockSize; // Ceiling division
                long bytesPerToken = (long) blocksPerToken * Q8_0_BLOCK_BYTES;

                MemorySegment.copy(
                        tokenEmbeddings,
                        (long) token * bytesPerToken,
                        state.workspace.embeddingX.getSegment(),
                        0,
                        bytesPerToken);
            }
            default ->
                    throw new IllegalArgumentException(
                            "Unsupported embedding weight type: "
                                    + weights.getTokenEmbeddingTable().dataType());
        }

        return state.workspace.logitsView(tornadoVMMasterPlan.tornadoVMForwardDecode(position));
    }
}
