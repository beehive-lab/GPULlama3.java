package org.beehive.gpullama3.backend.tornado;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.runtime.metrics.MetricsSink;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

// @formatter:off
/**
 * Common contract for all TornadoVM GPU execution plans.
 *
 * <p>Three concrete implementations exist:
 *
 * <ul>
 *   <li>{@link TornadoVMMasterPlanSingleToken} — baseline single-token forward pass (preprocessing
 *       + N layers + logits).
 *   <li>{@link TornadoVMMasterPlanPrefillDecode} — sequential prefill/decode separation; reuses the
 *       same N layer graphs for both phases, skipping logits during prefill.
 *   <li>{@link TornadoVMMasterPlanBatchPrefillDecode} — batched prefill + single-token decode;
 *       holds 2N+3 graphs in one plan to keep the KV cache on device across phases.
 * </ul>
 *
 * <p>The {@link #initializeTornadoVMPlan} factory selects the implementation based on {@code
 * llama.withPrefillDecode} and {@code llama.prefillBatchSize}:
 *
 * <ul>
 *   <li>{@code withPrefillDecode=false} → {@link TornadoVMMasterPlanSingleToken}
 *   <li>{@code withPrefillDecode=true}, {@code prefillBatchSize=1} → {@link
 *       TornadoVMMasterPlanPrefillDecode}
 *   <li>{@code withPrefillDecode=true}, {@code prefillBatchSize>1} → {@link
 *       TornadoVMMasterPlanBatchPrefillDecode}
 * </ul>
 */
public interface TornadoVMMasterPlan {

    boolean ENABLE_TORNADOVM_INIT_TIME =
            Boolean.parseBoolean(System.getProperty("llama.EnableTimingForTornadoVMInit", "False"));

    /** When {@code true}, {@code withCUDAGraph()} is called — PTX/CUDA backend only. */
    boolean CUDA_GRAPHS = Boolean.parseBoolean(System.getProperty("llama.cudaGraphs", "false"));

    /**
     * @deprecated Replaced by {@code state.executionPolicy()}. This constant survives only for
     *     {@code State} and {@code Qwen2MoEState}, which use the batch size to <b>size arrays</b> —
     *     a capacity input, not policy — and for the bench harness. It is not read on any execution
     *     path.
     */
    @Deprecated boolean WITH_PREFILL_DECODE = Boolean.getBoolean("llama.withPrefillDecode");

    /**
     * @deprecated see {@link #WITH_PREFILL_DECODE}. Capacity input only.
     */
    @Deprecated int PREFILL_BATCH_SIZE = Integer.getInteger("llama.prefillBatchSize", 1);

    /**
     * Factory: creates, JIT-compiles, and warms up the appropriate TornadoVMMasterPlan.
     *
     * <p>When {@code llama.withPrefillDecode=true} and {@code llama.prefillBatchSize > 1}, a {@link
     * TornadoVMMasterPlanBatchPrefillDecode} is returned. Otherwise a {@link
     * TornadoVMMasterPlanSingleToken} is returned (used for the baseline path and the sequential
     * prefill/decode path when batch size is 1).
     *
     * @param state the model state
     * @param model the model instance
     * @return the initialized plan, also stored via {@link Model#setTornadoVMPlan}
     */
    static TornadoVMMasterPlan initializeTornadoVMPlan(State state, Model model) {
        return initializeTornadoVMPlan(state, model, MetricsSink.disabled());
    }

    /**
     * As {@link #initializeTornadoVMPlan(State, Model)}, with the plan reporting device-side
     * measurements to {@code sink}.
     *
     * <p>The sink is taken here rather than installed later because TornadoVM's profiler must be
     * switched on before the plan compiles and executes. With the default disabled sink the
     * profiler is never enabled and the reporting is one boolean test per execution.
     */
    static TornadoVMMasterPlan initializeTornadoVMPlan(State state, Model model, MetricsSink sink) {
        // Plan construction is where the device buffers are actually allocated, so it is where an
        // over-budget configuration fails — on the lowered path and the legacy one alike, which is
        // why the boundary is the whole method rather than the construction sites inside it.
        try {
            return buildPlan(state, model, sink);
        } catch (RuntimeException failure) {
            if (DeviceMemoryExhaustedException.isDeviceExhaustion(failure)) {
                throw DeviceMemoryExhaustedException.wrap(failure);
            }
            throw failure;
        }
    }

    private static TornadoVMMasterPlan buildPlan(State state, Model model, MetricsSink sink) {
        TornadoVMMasterPlan plan;

        // The lowering's opt-in is consulted here, in the one factory every caller reaches, rather
        // than at each construction site. It was branched at two sites before — the API session and
        // the golden harness — which is why the CLI, the server and the benchmark script silently
        // ran the legacy path however the flag was set: a paired A/B taken through the script was
        // measuring legacy against legacy. `handles` answers false unless the opt-in is set and the
        // tuple is the one the slice implements, so this costs a boolean read otherwise.
        if (org.beehive.gpullama3.backend.tornado.lowering.LoweredPlanSelection.handles(
                model, state)) {
            reportPath(org.beehive.gpullama3.runtime.backend.ExecutionPath.LOWERED, model, state);
            return org.beehive.gpullama3.backend.tornado.lowering.LoweredPlanSelection.lower(
                    model, state, sink);
        }
        reportPath(org.beehive.gpullama3.runtime.backend.ExecutionPath.LEGACY, model, state);

        // Resolved from the session's policy, once, here — not from a class constant read at
        // initialization.
        var policy = state.executionPolicy();
        boolean prefillDecode =
                policy.phaseStrategy()
                        == org.beehive.gpullama3.runtime.policy.ExecutionPolicy.PhaseStrategy
                                .PREFILL_DECODE;
        if (prefillDecode && policy.prefillBatchSize() > 1) {
            // GPU path with batched prefill/decode
            plan = new TornadoVMMasterPlanBatchPrefillDecode(state, model, sink);
        } else if (prefillDecode) {
            // GPU path with simple prefill/decode
            plan = new TornadoVMMasterPlanPrefillDecode(state, model, sink);
        } else {
            // GPU path with no prefill/decode
            plan = new TornadoVMMasterPlanSingleToken(state, model, sink);
        }
        // Deliberately not stored on the model. A plan belongs to the session that built it;
        // parking it on the shared model meant the second session to start silently replaced the
        // first session's plan, and a later CPU call would then take the GPU branch against a
        // plan bound to someone else's buffers.
        return plan;
    }

    // @formatter:on

    /**
     * Creates the appropriate {@link TornadoExecutionPlan} instance for the given {@link Model} and
     * {@link State}.
     */
    TornadoExecutionPlan createExecutionPlan();

    void forceCopyInReadOnlyData();

    FloatArray tornadoVMForwardDecode(int position);

    /** Releases all device memory held by this plan. */
    void freeTornadoExecutionPlan();

    /**
     * Records which path this session took, and the exact combination [D-7].
     *
     * <p>Reported <b>here</b>, in the one factory every caller reaches, for the same reason the
     * lowering opt-in is consulted here: it was branched at two sites once, and the CLI, the server
     * and the benchmark script all silently ran the legacy path however the flag was set. A report
     * emitted anywhere else would have the same hole.
     */
    private static void reportPath(
            org.beehive.gpullama3.runtime.backend.ExecutionPath path,
            org.beehive.gpullama3.model.Model model,
            org.beehive.gpullama3.inference.state.State state) {
        var combination =
                org.beehive.gpullama3.backend.tornado.lowering.LoweredPlanSelection.combinationOf(
                        model, state);
        boolean qualified =
                org.beehive.gpullama3.backend.tornado.lowering.LoweringQualification.isQualified(
                        combination.architecture(), combination.dtype(), combination.mode());
        org.beehive.gpullama3.auxiliary.RunMetrics.setExecutionPath(
                path.reportName(),
                combination.toString(),
                qualified,
                org.beehive.gpullama3.backend.tornado.lowering.LoweredPlanSelection.mode()
                        .name()
                        .toLowerCase(java.util.Locale.ROOT));
    }
}
