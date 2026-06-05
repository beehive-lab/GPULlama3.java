package org.beehive.gpullama3.tornadovm;

import org.beehive.gpullama3.auxiliary.RunMetrics;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.tornadovm.plan.BatchPrefillDecodeForwardPlan;
import org.beehive.gpullama3.tornadovm.plan.ForwardPlanFactory;
import org.beehive.gpullama3.tornadovm.plan.layout.BatchPrefillDecodeForwardTaskGraphLayout;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * GPU execution plan for batched prefill + single-token decode.
 *
 * <p>A single {@link TornadoExecutionPlan} holds all task graphs for
 * batched prefill and single-token decode phases:</p>
 *
 * <p>TaskGraph layout (2N+3 TaskGraphs total):</p>
 * <pre>
 *   [0]         batchPrefillActivation  B×dim embeddings → FP32 wrapXBatch
 *   [1..N]      batch-prefill layers    B tokens, all transformer ops
 *   [N+1]       decodeActivation        single-token embedding → FP32 + KV-cache pass-through
 *   [N+2..2N+1] decode layers           single-token, standard kernels
 *   [2N+2]      logits
 * </pre>
 */
public class TornadoVMMasterPlanBatchPrefillDecode implements TornadoVMMasterPlan {

    private final State         state;
    private final Model         model;
    private final Configuration config;

    BatchPrefillDecodeForwardPlan batchPrefillDecodeForwardPlan;
    BatchPrefillDecodeForwardTaskGraphLayout taskGraphLayout;
    public TornadoExecutionPlan executionPlan;

    // ── Construction ─────────────────────────────────────────────────────────
    TornadoVMMasterPlanBatchPrefillDecode(State initialState, Model model) {
        if (ENABLE_TORNADOVM_INIT_TIME) {
            System.err.println("\nStarting TornadoVM initialization...");
        }

        this.state  = initialState;
        this.model  = model;
        this.config = model.configuration();

        long startTime = System.nanoTime();
        this.executionPlan = createExecutionPlan();
        long planCreationTime = System.nanoTime();

        if (CUDA_GRAPHS) executionPlan.withAllGraphs().withCUDAGraph();
        executionPlan.withPreCompilation();
        long warmupTime = System.nanoTime();

        forceCopyInReadOnlyData();
        long copyTime = System.nanoTime();

        RunMetrics.setTornadoMetrics(planCreationTime - startTime, warmupTime - planCreationTime, copyTime - warmupTime);
    }

    // ── Plan construction ─────────────────────────────────────────────────────

    @Override
    public TornadoExecutionPlan createExecutionPlan() {
        GGMLType weightType = model.weights().getWeightType();
        this.batchPrefillDecodeForwardPlan =
                ForwardPlanFactory.createBatchPrefillDecode(weightType, state, model);
        this.taskGraphLayout = batchPrefillDecodeForwardPlan.getTaskGraphLayout();
        var taskGraphs = batchPrefillDecodeForwardPlan.getImmutableTaskGraphs();
        return new TornadoExecutionPlan(taskGraphs.toArray(new ImmutableTaskGraph[0]));
    }

    // ── Initialisation ────────────────────────────────────────────────────────

    @Override
    public void forceCopyInReadOnlyData() {
        batchPrefillDecodeForwardPlan.getEmbeddingPreparer().initBatchState();
        state.wrapX.clear();
        state.positionHolder.init(0);

        for (int i = 0; i <= taskGraphLayout.logitsIdx(); i++) {
            var g = executionPlan.withGraph(i)
                    .withGridScheduler(batchPrefillDecodeForwardPlan.getGridScheduler());
            if (CUDA_GRAPHS) g.withCUDAGraph();
            g.execute();
        }
    }

    // ── Forward passes ────────────────────────────────────────────────────────

    /**
     * Batch prefill: runs graphs 0..N (activation + N layers), skips logits.
     * Caller is responsible for copying batch embeddings into state before calling this.
     */
    public void tornadoVMForwardBatchPrefill() {
        var batchAct = executionPlan.withGraph(taskGraphLayout.batchActivationIdx())
                .withGridScheduler(batchPrefillDecodeForwardPlan.getGridScheduler());
        if (CUDA_GRAPHS) batchAct.withCUDAGraph();
        batchAct.execute();

        for (int l = 0; l < config.numberOfLayers(); l++) {
            var batchLayer = executionPlan.withGraph(taskGraphLayout.batchLayerIdx(l))
                    .withGridScheduler(batchPrefillDecodeForwardPlan.getGridScheduler());
            if (CUDA_GRAPHS) batchLayer.withCUDAGraph();
            batchLayer.execute();
        }
    }

    /**
     * Single-token decode: runs graphs N+1..2N+2 (activation + N layers + logits).
     * Caller is responsible for copying the decode embedding into state before calling this.
     *
     * @param position sequence position
     * @return logits array for sampling
     */
    @Override
    public FloatArray tornadoVMForwardDecode(int position) {
        state.positionHolder.set(0, position);
        state.temp.clear();
        state.tempFFN.clear();

        var decodeAct = executionPlan.withGraph(taskGraphLayout.decodeActivationIdx())
                .withGridScheduler(batchPrefillDecodeForwardPlan.getGridScheduler());
        if (CUDA_GRAPHS) decodeAct.withCUDAGraph();
        decodeAct.execute();

        for (int l = 0; l < config.numberOfLayers(); l++) {
            var decodeLayer = executionPlan.withGraph(taskGraphLayout.decodeLayerIdx(l))
                    .withGridScheduler(batchPrefillDecodeForwardPlan.getGridScheduler());
            if (CUDA_GRAPHS) decodeLayer.withCUDAGraph();
            decodeLayer.execute();
        }

        state.tempLogits.clear();
        state.wrapLogits.clear();

        var logits = executionPlan.withGraph(taskGraphLayout.logitsIdx())
                .withGridScheduler(batchPrefillDecodeForwardPlan.getGridScheduler());
        if (CUDA_GRAPHS) logits.withCUDAGraph();
        logits.execute();

        return state.wrapLogits;
    }

    @Override
    public FloatArray tornadoVMExecuteForward(int position) {
        throw new UnsupportedOperationException(
                "Use tornadoVMForwardBatchPrefill / tornadoVMForwardDecode for batch plan");
    }

    @Override
    public void freeTornadoExecutionPlan() {
        executionPlan.freeDeviceMemory();
    }
}
