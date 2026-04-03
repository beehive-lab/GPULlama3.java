package org.beehive.gpullama3.tornadovm;

import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Model;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Common contract for all TornadoVM GPU execution plans.
 *
 * <p>Two concrete implementations exist:</p>
 * <ul>
 *   <li>{@link TornadoVMMasterPlanStandard} — single-token forward pass; used for the
 *       baseline GPU path and Phase 2 sequential prefill/decode.</li>
 *   <li>{@link TornadoVMMasterPlanBatchPrefill} — unified plan for Phase 4 batched
 *       prefill + single-token decode within one {@code TornadoExecutionPlan}.</li>
 * </ul>
 *
 * <p>The {@link #initializeTornadoVMPlan} factory selects the appropriate implementation
 * based on {@code llama.prefillBatchSize}: if {@code > 1}, returns a
 * {@link TornadoVMMasterPlanBatchPrefill}; otherwise returns a
 * {@link TornadoVMMasterPlanStandard}.</p>
 */
public interface TornadoVMMasterPlan {

    boolean ENABLE_TORNADOVM_INIT_TIME = Boolean.parseBoolean(
            System.getProperty("llama.EnableTimingForTornadoVMInit", "False"));

    /** When {@code false}, {@code withCUDAGraph()} is never called — useful for debugging. */
    boolean CUDA_GRAPHS = Boolean.parseBoolean(
            System.getProperty("llama.cudaGraphs", "true"));

    /**
     * Single-token forward pass returning output logits.
     *
     * <p>Used by the standard GPU path ({@link org.beehive.gpullama3.inference.InferenceCore#forwardTornadoVM})
     * and the Phase 2 sequential decode path. Not applicable to
     * {@link TornadoVMMasterPlanBatchPrefill} — that plan uses its own typed methods.</p>
     *
     * @param position sequence position of the current token
     * @return logits array for token sampling
     */
    FloatArray tornadoVMForwardExecuteLayered(int position);

    /** Releases all device memory held by this plan. */
    void freeTornadoExecutionPlan();

    /**
     * Factory: creates, JIT-compiles, and warms up the appropriate plan.
     *
     * <p>When {@code llama.prefillBatchSize > 1} a {@link TornadoVMMasterPlanBatchPrefill}
     * is returned; otherwise a {@link TornadoVMMasterPlanStandard} is returned.</p>
     *
     * @param state the model state (must be {@link LlamaState} when batch size {@code > 1})
     * @param model the model instance
     * @return the initialized plan, also stored via {@link Model#setTornadoVMPlan}
     */
    static TornadoVMMasterPlan initializeTornadoVMPlan(State state, Model model) {
        int batchSize = Integer.getInteger("llama.prefillBatchSize", 1);
        TornadoVMMasterPlan plan;
        if (batchSize > 1) {
            plan = TornadoVMMasterPlanBatchPrefill.initializeUnifiedPlan(
                    (LlamaState) state, model, batchSize);
        } else {
            plan = TornadoVMMasterPlanStandard.initialize(state, model);
        }
        model.setTornadoVMPlan(plan);
        return plan;
    }
}
