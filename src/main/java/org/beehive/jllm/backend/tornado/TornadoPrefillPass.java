package org.beehive.gpullama3.backend.tornado;

import java.lang.foreign.MemorySegment;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;

/**
 * The accelerated <b>sequential prefill</b> pass: stage one token's embedding, then run the plan's
 * prefill graphs.
 *
 * <p>The body is the one that ran before, moved rather than rewritten: the same two data-type
 * cases, the same byte arithmetic, the same delegation to {@link
 * TornadoVMMasterPlanPrefillDecode#tornadoVMForwardPrefill}. Sequential prefill stays selectable
 * and behaves identically.
 */
public final class TornadoPrefillPass {

    private TornadoPrefillPass() {}

    /**
     * Stages {@code token}'s embedding into the session's device carrier, then executes the
     * preprocessing and layer graphs — the logits graph is deliberately skipped, because a prefill
     * position's logits are discarded and only its KV entry matters.
     *
     * @param model the model (must carry {@link TornadoWeights})
     * @param state the session's state, whose workspace holds the embedding carrier
     * @param token input token id
     * @param position sequence position being processed
     * @param prefillPlan the prefill/decode plan wrapper
     */
    public static void prefill(
            Model model,
            State state,
            int token,
            int position,
            TornadoVMMasterPlanPrefillDecode prefillPlan) {
        final Configuration configuration = model.configuration();
        final TornadoWeights weights = (TornadoWeights) model.weights();

        switch (weights.dataType()) {
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
                int blocksPerToken = (configuration.dim() + 31) / 32;
                long bytesPerToken = (long) blocksPerToken * 34;
                MemorySegment.copy(
                        tokenEmbeddings,
                        (long) token * bytesPerToken,
                        state.workspace.embeddingX.getSegment(),
                        0,
                        bytesPerToken);
            }
            default ->
                    throw new IllegalArgumentException(
                            "Unsupported weight type: " + weights.dataType());
        }

        prefillPlan.tornadoVMForwardPrefill(position);
    }
}
