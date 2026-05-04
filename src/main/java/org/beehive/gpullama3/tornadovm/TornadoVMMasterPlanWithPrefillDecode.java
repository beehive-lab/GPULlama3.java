package org.beehive.gpullama3.tornadovm;

import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernels;
import org.beehive.gpullama3.tornadovm.layerplanner.WorkerGridFactory;
import org.beehive.gpullama3.tornadovm.layerplanner.strategy.SchedulerDetectionService;
import org.beehive.gpullama3.tornadovm.layerplanner.strategy.SchedulerType;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.decode.LlamaFP16FFNLayersPrefillDecode;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.decode.LogitsFP16LayerDecode;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

import java.util.ArrayList;
import java.util.List;

/**
 * GPU execution plan for sequential (single-token) prefill/decode separation.
 *
 * <p>A single {@link TornadoExecutionPlan} holds all graphs so that the KV cache
 * ({@code wrapKeyCache}, {@code wrapValueCache}) is allocated once and remains on
 * device across both phases.  Prefill and decode reuse the same N layer graphs;
 * only the logits graph is skipped during prefill.</p>
 *
 * <p>Graph layout (N+2 graphs total):</p>
 * <pre>
 *   [0]      decodeActivation    single-token FP16 → FP32; KV-cache allocated on first execution
 *   [1..N]   layer_0..layer_N-1  transformer layers (attention + FFN)
 *   [N+1]    logits              final RMSNorm + wcls matmul
 * </pre>
 *
 * <p>Two forward passes:</p>
 * <ul>
 *   <li>{@link #tornadoVMForwardPrefill} — graphs 0..N (activation + layers), logits skipped.
 *       Called once per prompt token; populates the KV cache.</li>
 *   <li>{@link #tornadoVMForwardDecode} — full pass including logits.
 *       Called once per generated token; returns logits for sampling.</li>
 * </ul>
 */
public class TornadoVMMasterPlanWithPrefillDecode implements TornadoVMMasterPlan {

    private final LlamaState         state;
    private final Model              model;
    private final LlamaConfiguration config;
    private final int                N;   // numberOfLayers
    private final TornadoExecutionPlan executionPlan;
    private final GridScheduler        gridScheduler;

    // ── Graph-index helpers ───────────────────────────────────────────────────
    private int activationIdx()    { return 0; }
    private int layerIdx(int i)    { return 1 + i; }
    private int logitsIdx()        { return N + 1; }

    // ── Construction ─────────────────────────────────────────────────────────
    TornadoVMMasterPlanWithPrefillDecode(State initialState, Model model) {
        long startTime = System.nanoTime();
        long planCreationTime = 0;
        long warmupTime = 0;

        if (ENABLE_TORNADOVM_INIT_TIME) {
            System.err.println("\nStarting TornadoVM initialization...");
        }

        this.state  = (LlamaState) initialState;
        this.model  = model;
        this.config = (LlamaConfiguration) model.configuration();
        this.N      = config.numberOfLayers();
        this.gridScheduler = new GridScheduler();
        this.executionPlan = createExecutionPlan();

        if (ENABLE_TORNADOVM_INIT_TIME) {
            planCreationTime = System.nanoTime();
            System.err.printf("TornadoVM GPU single-token prefill/decode execution plan creation: %.2f ms\n",
                    (planCreationTime - startTime) / 1_000_000.0);
        }

        if (CUDA_GRAPHS) executionPlan.withAllGraphs().withCUDAGraph();
        executionPlan.withPreCompilation();

        if (ENABLE_TORNADOVM_INIT_TIME) {
            warmupTime = System.nanoTime();
            System.err.printf("Java to GPU JIT compiler warmup: %.2f ms\n",
                    (warmupTime - planCreationTime) / 1_000_000.0);
        }

        forceCopyInReadOnlyData();

        if (ENABLE_TORNADOVM_INIT_TIME) {
            long copyTime = System.nanoTime();
            System.err.printf("Transfer read-only weights to GPU: %.2f ms\n",
                    (copyTime - warmupTime) / 1_000_000.0);
            System.err.printf("Finished TornadoVM initialization...\n \n");
        }
    }

    // ── Activation graph ─────────────────────────────────────────────────────

    /**
     * Graph 0: single-token FP16 → FP32.
     *
     * <p>Outputs {@code wrapX} (FP32 hidden state) and persists it on device so that
     * decode layer 0 can pick it up via {@code consumeFromDevice("decodeActivation", wrapX)}.
     * The KV cache is <em>not</em> managed here — it is allocated on the first forward pass
     * by decode layer 0 via {@code FIRST_EXECUTION}.</p>
     */
    private TaskGraph buildActivationGraph(KernelContext ctx) {
        return new TaskGraph("decodeActivation")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.embeddingX)
                .task("updateX", TransformerComputeKernels::convertFP16toFP32,
                        ctx, (HalfFloatArray) state.embeddingX, state.wrapX)
                .persistOnDevice(state.wrapX);
    }

    // ── Plan construction ─────────────────────────────────────────────────────
    /**
     * Creates the {@link TornadoExecutionPlan} for forward pass with *prefill/decode separation*.
     * Prefill is token-by-token but does not compute logits.
     *
     * TODO: support Q8_0 weights
     * To implement this, consult how {@link TornadoVMMasterPlanStandard} uses the {@link QuantizationPlannerFactory}
     */
    @Override
    public TornadoExecutionPlan createExecutionPlan() {
        GGMLType weightType = model.weights().getWeightType();
        switch (weightType) {
            case F16 -> { /* supported — continue below */ }
            case Q8_0 -> throw new UnsupportedOperationException(
                    "Prefill/decode GPU path not yet implemented for Q8_0 weights");
            default -> throw new UnsupportedOperationException(
                    "Prefill/decode GPU path not supported for weight type: " + weightType);
        }

        LlamaTornadoWeights weights      = (LlamaTornadoWeights) model.weights();
        SchedulerType       schedulerType = SchedulerDetectionService.determineSchedulerType(model);

        List<ImmutableTaskGraph> all = new ArrayList<>(N + 2);

        // [0] Activation ──────────────────────────────────────────────────────
        KernelContext actCtx = new KernelContext();
        all.add(buildActivationGraph(actCtx).snapshot());
        gridScheduler.addWorkerGrid("decodeActivation.updateX",
                WorkerGridFactory.genericWorker(config.dim(), 128));

        // [1..N] Decode layer graphs ──────────────────────────────────────────
        // Layer 0: FIRST_EXECUTION for KV cache + consumeFromDevice("decodeActivation", wrapX).
        // Layers 1+: consumeFromDevice with explicit predecessor names for interpreter mode.
        LlamaFP16FFNLayersPrefillDecode decodeLayers =
                new LlamaFP16FFNLayersPrefillDecode("decode", state, weights, config, schedulerType);
        all.addAll(decodeLayers.getFFNLayerImmutableTaskGraphs());
        decodeLayers.updateGridScheduler(gridScheduler);

        // [N+1] Logits ────────────────────────────────────────────────────────
        // LogitsFP16LayerDecode re-persists the KV cache so the pointer survives
        // the logits → layer_0 KV-cache FIRST_EXECUTION boundary across decode tokens.
        LogitsFP16LayerDecode logitsLayer = new LogitsFP16LayerDecode(
                "logits", state, weights, config,
                decodeLayers.getLastFFNLayerTaskGraphID(), schedulerType);
        all.add(logitsLayer.getImmutableTaskGraph());
        logitsLayer.updateGridScheduler(gridScheduler);

        return new TornadoExecutionPlan(all.toArray(new ImmutableTaskGraph[0]));
    }

    // ── Initialisation ────────────────────────────────────────────────────────

    /** Runs all graphs once to trigger FIRST_EXECUTION uploads and warm up CUDA graphs. */
    @Override
    public void forceCopyInReadOnlyData() {
        state.wrapX.clear();
        state.positionHolder.init(0);

        for (int i = 0; i <= logitsIdx(); i++) {
            var g = executionPlan.withGraph(i).withGridScheduler(gridScheduler);
            if (CUDA_GRAPHS) g.withCUDAGraph();
            g.execute();
        }
    }

    // ── Forward passes ────────────────────────────────────────────────────────

    /**
     * GPU prefill forward: activation + all transformer layers, logits skipped.
     * KV cache is populated for each prompt token.
     *
     * @param position sequence position being processed
     */
    public void tornadoVMForwardPrefill(int position) {
        var prefillActivation = executionPlan.withGraph(activationIdx()).withGridScheduler(gridScheduler);
        if (CUDA_GRAPHS) prefillActivation.withCUDAGraph();
        prefillActivation.execute();

        state.positionHolder.set(0, position);
        state.temp.clear();
        state.tempFFN.clear();

        for (int layer = 0; layer < N; layer++) {
            var prefillLayer = executionPlan.withGraph(layerIdx(layer)).withGridScheduler(gridScheduler);
            if (CUDA_GRAPHS) prefillLayer.withCUDAGraph();
            prefillLayer.execute();
        }
    }

    /**
     * GPU decode forward: full execution including logits.
     *
     * @param position sequence position being processed
     * @return logits array for token sampling
     */
    public FloatArray tornadoVMForwardDecode(int position) {
        return tornadoVMForwardExecuteLayered(position);
    }

    @Override
    public FloatArray tornadoVMForwardExecuteLayered(int position) {
        var act = executionPlan.withGraph(activationIdx()).withGridScheduler(gridScheduler);
        if (CUDA_GRAPHS) act.withCUDAGraph();
        act.execute();

        state.positionHolder.set(0, position);
        state.temp.clear();
        state.tempFFN.clear();

        for (int layer = 0; layer < N; layer++) {
            var l = executionPlan.withGraph(layerIdx(layer)).withGridScheduler(gridScheduler);
            if (CUDA_GRAPHS) l.withCUDAGraph();
            l.execute();
        }

        state.tempLogits.clear();
        state.wrapLogits.clear();
        var logits = executionPlan.withGraph(logitsIdx()).withGridScheduler(gridScheduler);
        if (CUDA_GRAPHS) logits.withCUDAGraph();
        logits.execute();

        return state.wrapLogits;
    }

    @Override
    public void freeTornadoExecutionPlan() {
        executionPlan.freeDeviceMemory();
    }
}
