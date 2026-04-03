package org.beehive.gpullama3.tornadovm;

import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernels;
import org.beehive.gpullama3.tornadovm.layerplanner.WorkerGridFactory;
import org.beehive.gpullama3.tornadovm.layerplanner.strategy.SchedulerDetectionService;
import org.beehive.gpullama3.tornadovm.layerplanner.strategy.SchedulerType;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.decode.LlamaFP16FFNLayersDecode;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.prefill.LlamaFP16LayersBatchPrefill;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LogitsFP16Layer;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

/**
 * Unified GPU execution plan for Phase 4: batched prefill + single-token decode.
 *
 * <p>A single {@link TornadoExecutionPlan} holds all graphs so that the KV cache
 * ({@code wrapKeyCache}, {@code wrapValueCache}) is shared on device via
 * {@code persistOnDevice}/{@code consumeFromDevice}.  Two separate plans would
 * allocate independent device buffers and lose the prefill KV state.</p>
 *
 * <p>Graph layout (2N+3 graphs total):</p>
 * <pre>
 *   [0]         batch activation     B×dim FP16 → FP32
 *   [1..N]      batch layer graphs   B tokens, all transformer ops
 *   [N+1]       decode activation    single-token FP16 → FP32 + KV-cache pass-through
 *   [N+2..2N+1] decode layer graphs  single-token, standard kernels
 *   [2N+2]      logits graph
 * </pre>
 *
 * <p>KV cache pointer chain across phases:</p>
 * <pre>
 *   batchLayer[N-1]  --persistOnDevice(wrapKeyCache)-→
 *   decodeActivation --consumeFromDevice(wrapKeyCache)-→  (pass-through)
 *   decodeLayer[0]   --consumeFromDevice(wrapKeyCache)-→  (used by attention)
 * </pre>
 */
public class TornadoVMMasterPlanBatchPrefill implements TornadoVMMasterPlan {

    private static final boolean ENABLE_TIMING =
            Boolean.parseBoolean(System.getProperty("llama.EnableTimingForTornadoVMInit", "False"));

    private final LlamaState         state;
    private final LlamaConfiguration config;
    private final int                batchSize;
    private final int                N;   // numberOfLayers
    private final TornadoExecutionPlan executionPlan;
    private final GridScheduler        gridScheduler;

    // ── Graph-index helpers ───────────────────────────────────────────────────
    private int batchActivationIdx()       { return 0; }
    private int batchLayerIdx(int i)       { return 1 + i; }
    private int decodeActivationIdx()      { return N + 1; }
    private int decodeLayerIdx(int i)      { return N + 2 + i; }
    private int logitsIdx()                { return 2 * N + 2; }

    // ── Construction ─────────────────────────────────────────────────────────
    private TornadoVMMasterPlanBatchPrefill(LlamaState state, Model model, int batchSize) {
        this.state     = state;
        this.config    = (LlamaConfiguration) model.configuration();
        this.batchSize = batchSize;
        this.N         = config.numberOfLayers();

        LlamaTornadoWeights weights       = (LlamaTornadoWeights) model.weights();
        SchedulerType       schedulerType = SchedulerDetectionService.determineSchedulerType(model);

        List<ImmutableTaskGraph> all       = new ArrayList<>(2 * N + 3);
        GridScheduler            scheduler = new GridScheduler();

        // [0] Batch prefill activation ────────────────────────────────────────────────
        KernelContext batchActCtx = new KernelContext();
        all.add(buildBatchPrefillActivationGraph(batchActCtx).snapshot());
        scheduler.addWorkerGrid("batchActivation.batchUpdateX",
                WorkerGridFactory.genericWorker(batchSize * config.dim(), 128));

        // [1..N] Batch prefill layer graphs ───────────────────────────────────────────
        LlamaFP16LayersBatchPrefill batchLayers =
                new LlamaFP16LayersBatchPrefill(state, weights, config, batchSize);
        all.addAll(batchLayers.getLayerImmutableTaskGraphs());
        batchLayers.updateGridScheduler(scheduler);

        // [N+1] Decode activation (with KV-cache pass-through) ────────────────
        KernelContext decodeActCtx = new KernelContext();
        all.add(buildDecodeActivationGraph(decodeActCtx).snapshot());
        scheduler.addWorkerGrid("activationUpdate.updateX",
                WorkerGridFactory.genericWorker(config.dim(), 128));

        // [N+2..2N+1] Decode layer graphs  ────────────────────────────────────
        // Layer 0 uses consumeFromDevice for KV cache (no FIRST_EXECUTION upload).
        LlamaFP16FFNLayersDecode decodeLayers =
                new LlamaFP16FFNLayersDecode(
                        "llamaFFNDecode", state, weights, config, schedulerType);
        all.addAll(decodeLayers.getFFNLayerImmutableTaskGraphs());
        decodeLayers.updateGridScheduler(scheduler);

        // [2N+2] Logits ───────────────────────────────────────────────────────
        LogitsFP16Layer logitsLayer = new LogitsFP16Layer("logits", state, weights, config,
                decodeLayers.getLastFFNLayerTaskGraphID(), schedulerType);
        all.add(logitsLayer.getImmutableTaskGraph());
        logitsLayer.updateGridScheduler(scheduler);

        this.gridScheduler  = scheduler;
        this.executionPlan  = new TornadoExecutionPlan(all.toArray(new ImmutableTaskGraph[0]));
    }

    // ── Batch Prefill Activation graphs ─────────────────────────────────────────────────────

    /** Graph 0: B×dim FP16 embeddings → FP32 wrapXBatch. */
    private TaskGraph buildBatchPrefillActivationGraph(KernelContext ctx) {
        return new TaskGraph("batchActivation")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, ctx, state.wrapXBatch)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.embeddingXBatch)
                .task("batchUpdateX",
                        (KernelContext c, HalfFloatArray src, FloatArray dst) ->
                                dst.set(c.globalIdx, src.get(c.globalIdx).getFloat32()),
                        ctx, state.embeddingXBatch, state.wrapXBatch)
                .persistOnDevice(state.wrapXBatch);
    }

    /**
     * Graph N+1: single-token FP16 → FP32.
     *
     * <p>Receives the KV-cache device pointer from batch layer N via
     * {@code consumeFromDevice}, then re-emits it via {@code persistOnDevice} so
     * that {@code updatePersistedObjectState()} can propagate it to decode layer 0.
     * Both halves of the chain are required; without the re-persist the pointer is
     * not forwarded in interpreter (non-CUDA-graph) mode.</p>
     */
    private TaskGraph buildDecodeActivationGraph(KernelContext ctx) {
        return new TaskGraph("activationUpdate")
                .consumeFromDevice(state.wrapKeyCache, state.wrapValueCache)   // KV pass-through
//                .transferToDevice(DataTransferMode.EVERY_EXECUTION,
//                        state.wrapKeyCache,
//                        state.wrapValueCache)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, ctx, state.wrapX)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.embeddingX)
                .task("updateX",
                        TransformerComputeKernels::convertFP16toFP32,
                        ctx, (HalfFloatArray) state.embeddingX, state.wrapX)
                // wrapX persisted for decode layer 0; wrapKeyCache/wrapValueCache
                // re-persisted so updatePersistedObjectState() propagates the device
                // pointer to decode layer 0's consumeFromDevice without CUDA graphs.
                .persistOnDevice(state.wrapX, state.wrapKeyCache, state.wrapValueCache);
    }

    // ── Static factory ────────────────────────────────────────────────────────

    /**
     * Creates, JIT-compiles, and warms up the unified plan.
     * Mirrors {@link TornadoVMMasterPlan#initializeTornadoVMPlan}.
     */
    public static TornadoVMMasterPlanBatchPrefill initializeUnifiedPlan(
            LlamaState state, Model model, int batchSize) {

        long t0 = System.nanoTime();
        TornadoVMMasterPlanBatchPrefill plan =
                new TornadoVMMasterPlanBatchPrefill(state, model, batchSize);

        if (ENABLE_TIMING)
            System.err.printf("[BatchPlan] Graph construction: %.2f ms%n",
                    (System.nanoTime() - t0) / 1e6);

        if (CUDA_GRAPHS) plan.executionPlan.withAllGraphs().withCUDAGraph();
        plan.executionPlan.withPreCompilation();

        if (ENABLE_TIMING)
            System.err.printf("[BatchPlan] JIT compilation: %.2f ms%n",
                    (System.nanoTime() - t0) / 1e6);

        plan.forceCopyInReadOnlyData();

        if (ENABLE_TIMING)
            System.err.printf("[BatchPlan] Init complete: %.2f ms%n",
                    (System.nanoTime() - t0) / 1e6);

        return plan;
    }

    /** Runs all graphs once to trigger FIRST_EXECUTION uploads and warm up CUDA graphs. */
    private void forceCopyInReadOnlyData() {
        state.wrapXBatch.clear();
        state.wrapX.clear();
        state.positionHolder.init(0);
        state.batchStartPosHolder.init(0);

        for (int i = 0; i <= logitsIdx(); i++) {
            var g = executionPlan.withGraph(i).withGridScheduler(gridScheduler);
            if (CUDA_GRAPHS) g.withCUDAGraph();
            g.execute();
        }
    }

    // ── Forward passes ────────────────────────────────────────────────────────

    /**
     * Batch prefill: runs graphs 0..N (activation + N layers), skips logits.
     *
     * @param tokenIds  token IDs for this chunk (length == batchSize, or tail)
     * @param startPos  sequence position of tokenIds[0]
     * @param model     model (for embedding table)
     * @param chunkSize actual number of tokens in this chunk (≤ batchSize)
     */
    public void tornadoVMForwardBatchPrefill(int[] tokenIds, int startPos, Model model, int chunkSize) {
        LlamaTornadoWeights weights = (LlamaTornadoWeights) model.weights();
        MemorySegment embTable = weights.getTokenEmbeddingTable().asHalfFloatArray().getSegment();
        int bytes = Short.BYTES;
        int dim   = config.dim();

        // Copy B embeddings into embeddingXBatch
        for (int b = 0; b < chunkSize; b++) {
            MemorySegment.copy(embTable, (long) tokenIds[b] * dim * bytes,
                    state.embeddingXBatch.getSegment(), (long) b * dim * bytes,
                    (long) dim * bytes);
        }
        state.batchStartPosHolder.set(0, startPos);

        // Graph 0: batch activation
        var batchAct = executionPlan.withGraph(batchActivationIdx()).withGridScheduler(gridScheduler);
        if (CUDA_GRAPHS) batchAct.withCUDAGraph();
        batchAct.execute();

        // Graphs 1..N: batch transformer layers
        for (int l = 0; l < N; l++) {
            var batchLayer = executionPlan.withGraph(batchLayerIdx(l)).withGridScheduler(gridScheduler);
            if (CUDA_GRAPHS) batchLayer.withCUDAGraph();
            batchLayer.execute();
        }
        //System.err.println("[DEBUG] last batch layer done, about to return from prefill");
        // Logits skipped — not needed for prefill positions.
    }

    /**
     * Single-token decode: runs graphs N+1..2N+2 (activation + N layers + logits).
     *
     * @param token    token ID to process
     * @param position sequence position
     * @param model    model (for embedding table)
     * @return logits array for sampling
     */
    public FloatArray tornadoVMForwardDecode(int token, int position, Model model) {
        LlamaTornadoWeights weights = (LlamaTornadoWeights) model.weights();
        MemorySegment embTable = weights.getTokenEmbeddingTable().asHalfFloatArray().getSegment();
        int bytes = Short.BYTES;
        int dim   = config.dim();

        MemorySegment.copy(embTable, (long) token * dim * bytes,
                state.embeddingX.getSegment(), 0L, (long) dim * bytes);

        state.positionHolder.set(0, position);
        state.temp.clear();
        state.tempFFN.clear();

        // Graph N+1: decode activation
        var decodeAct = executionPlan.withGraph(decodeActivationIdx()).withGridScheduler(gridScheduler);
        if (CUDA_GRAPHS) decodeAct.withCUDAGraph();
        //System.err.println("[DEBUG] about to execute decode activation (graph " + decodeActivationIdx() + "--)");
        decodeAct.execute();

        // Graphs N+2..2N+1: decode transformer layers
        for (int l = 0; l < N; l++) {
            var decodeLayer = executionPlan.withGraph(decodeLayerIdx(l)).withGridScheduler(gridScheduler);
            if (CUDA_GRAPHS) decodeLayer.withCUDAGraph();
            //System.err.println("[DEBUG] about to execute decode transformer layer (graph " + decodeLayerIdx(l) + "--)");
            decodeLayer.execute();
        }

        state.tempLogits.clear();
        state.wrapLogits.clear();

        // Graph 2N+2: logits
        var logits = executionPlan.withGraph(logitsIdx()).withGridScheduler(gridScheduler);
        if (CUDA_GRAPHS) logits.withCUDAGraph();
        logits.execute();

        return state.wrapLogits;
    }

    @Override
    public FloatArray tornadoVMForwardExecuteLayered(int position) {
        throw new UnsupportedOperationException(
                "Use tornadoVMForwardBatchPrefill / tornadoVMForwardDecode for batch plan");
    }

    @Override
    public void freeTornadoExecutionPlan() {
        executionPlan.freeDeviceMemory();
    }

    // ── Inner class: decode layer 0 with consumeFromDevice for KV cache ───────
// moved to package
//
//    private static final class LlamaFP16FFNLayersForUnifiedDecode extends LlamaFP16FFNLayers {
//
//
//    }
}
