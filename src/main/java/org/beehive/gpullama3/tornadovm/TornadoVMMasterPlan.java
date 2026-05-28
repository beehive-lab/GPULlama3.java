package org.beehive.gpullama3.tornadovm;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Model;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Common contract for all TornadoVM GPU execution plans.
 *
 * <p>Three concrete implementations exist:</p>
 * <ul>
 *   <li>{@link TornadoVMMasterPlanSingleToken} — baseline single-token forward pass
 *       (preprocessing + N layers + logits).</li>
 *   <li>{@link TornadoVMMasterPlanPrefillDecode} — sequential prefill/decode separation;
 *       reuses the same N layer graphs for both phases, skipping logits during prefill.</li>
 *   <li>{@link TornadoVMMasterPlanBatchPrefillDecode} — batched prefill + single-token
 *       decode; holds 2N+3 graphs in one plan to keep the KV cache on device across phases.</li>
 * </ul>
 *
 * <p>The {@link #initializeTornadoVMPlan} factory selects the implementation based on
 * {@code llama.withPrefillDecode} and {@code llama.prefillBatchSize}:</p>
 * <ul>
 *   <li>{@code withPrefillDecode=false} → {@link TornadoVMMasterPlanSingleToken}</li>
 *   <li>{@code withPrefillDecode=true}, {@code prefillBatchSize=1} → {@link TornadoVMMasterPlanPrefillDecode}</li>
 *   <li>{@code withPrefillDecode=true}, {@code prefillBatchSize>1} → {@link TornadoVMMasterPlanBatchPrefillDecode}</li>
 * </ul>
 */
public interface TornadoVMMasterPlan {

    boolean ENABLE_TORNADOVM_INIT_TIME = Boolean.parseBoolean(
            System.getProperty("llama.EnableTimingForTornadoVMInit", "False"));

    /** When {@code true}, {@code withCUDAGraph()} is called — PTX/CUDA backend only. */
    boolean CUDA_GRAPHS = Boolean.parseBoolean(
            System.getProperty("llama.cudaGraphs", "false"));

    boolean WITH_PREFILL_DECODE = Boolean.getBoolean("llama.withPrefillDecode");

    int PREFILL_BATCH_SIZE = Integer.getInteger("llama.prefillBatchSize", 1);

    /**
     * Factory: creates, JIT-compiles, and warms up the appropriate TornadoVMMasterPlan.
     *
     * <p>When {@code llama.withPrefillDecode=true} and {@code llama.prefillBatchSize > 1},
     * a {@link TornadoVMMasterPlanBatchPrefillDecode} is returned.
     * Otherwise a {@link TornadoVMMasterPlanSingleToken} is returned (used for the baseline
     * path and the sequential prefill/decode path when batch size is 1).</p>
     *
     * @param state the model state
     * @param model the model instance
     * @return the initialized plan, also stored via {@link Model#setTornadoVMPlan}
     */
    static TornadoVMMasterPlan initializeTornadoVMPlan(State state, Model model) {
        TornadoVMMasterPlan plan;

        if (WITH_PREFILL_DECODE && PREFILL_BATCH_SIZE > 1) {
            // GPU path with batched prefill/decode
            plan = new TornadoVMMasterPlanBatchPrefillDecode(state, model);
        } else if (WITH_PREFILL_DECODE) {
            // GPU path with simple prefill/decode
            plan = new TornadoVMMasterPlanPrefillDecode(state, model);
        } else {
            // GPU path with no prefill/decode
            plan = new TornadoVMMasterPlanSingleToken(state, model);
        }
        model.setTornadoVMPlan(plan);
        return plan;
    }

    /**
     * Creates the appropriate {@link TornadoExecutionPlan} instance
     * for the given {@link Model} and {@link State}.
     */
    TornadoExecutionPlan createExecutionPlan();

    void forceCopyInReadOnlyData();

    FloatArray tornadoVMExecuteForward(int position);

    /** Releases all device memory held by this plan. */
    void freeTornadoExecutionPlan();
}
