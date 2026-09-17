package org.beehive.jllm.backend.tornado.plan.components.activation;

import org.beehive.jllm.backend.tornado.kernels.TransformerComputeKernels;
import org.beehive.jllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.model.Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;

// @formatter:off
/**
 * This family's decode activation graph with the key/value pass-through ("decodeActivation").
 *
 * <p>The counterpart of {@link BatchDecodeActivation} for a family whose cache is flat rather than
 * paged: there is no block table to relay, because there are no blocks — this family addresses one
 * contiguous buffer through a per-layer base offset, and twenty of its thirty-five layers address
 * an earlier layer's slot rather than one of their own.
 *
 * <p><b>The pass-through is host-side state aliasing, not device work.</b> This graph's only task is
 * the embedding conversion; the caches are arguments to no task in it. What the consume/persist
 * pairs do is make TornadoVM point this graph's buffer state for them at the producing graph's, so
 * <em>last batch-prefill layer → decodeActivation → decode layer 0</em> resolves to one live
 * buffer. They look inert in a bytecode trace and are not.
 */
// @formatter:on
public class Gemma4BatchDecodeActivation implements ActivationTaskGraph {

    private final ImmutableTaskGraph itg;
    private final int dim;

    public Gemma4BatchDecodeActivation(
            State state, Configuration config, String lastBatchLayerId) {
        this.dim = config.dim();
        KernelContext ctx = new KernelContext();
        this.itg = buildGraph(ctx, state, lastBatchLayerId).snapshot();
    }

    private TaskGraph buildGraph(KernelContext ctx, State state, String lastBatchLayerId) {
        return new TaskGraph("decodeActivation")
                .consumeFromDevice(
                        lastBatchLayerId,
                        state.workspace.wrapKeyCache,
                        state.workspace.wrapValueCache)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.workspace.embeddingX)
                .task(
                        "updateX",
                        TransformerComputeKernels::convertQ8_0toFP32,
                        ctx,
                        (ByteArray) state.workspace.embeddingX,
                        state.workspace.wrapX)
                .persistOnDevice(
                        state.workspace.wrapX,
                        state.workspace.wrapKeyCache,
                        state.workspace.wrapValueCache);
    }

    @Override
    public ImmutableTaskGraph getImmutableTaskGraph() {
        return itg;
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler scheduler) {
        scheduler.addWorkerGrid(
                "decodeActivation.updateX", WorkerGridFactory.genericWorker(dim, 128));
        return scheduler;
    }
}
