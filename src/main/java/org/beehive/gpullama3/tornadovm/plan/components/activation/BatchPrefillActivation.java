package org.beehive.gpullama3.tornadovm.plan.components.activation;

import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerBatchPrefillKernels;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernels;
import org.beehive.gpullama3.tornadovm.layers.ActivationTaskGraph;
import org.beehive.gpullama3.tornadovm.scheduling.WorkerGridFactory;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Batch-prefill activation graph ("prefillActivation").
 *
 * <ul>
 *   <li>FP16: uploads {@code embeddingXBatch} (EVERY_EXECUTION) and converts to
 *       FP32 {@code wrapXBatch}.</li>
 *   <li>Q8_0: uploads FP32 {@code wrapXBatch} pre-filled by host (EVERY_EXECUTION)
 *       and runs a no-op {@code batchPassthrough} so TornadoVM has at least one task.</li>
 * </ul>
 */
public class BatchPrefillActivation implements ActivationTaskGraph {

    private final ImmutableTaskGraph itg;
    private final boolean isQ8;
    private final int batchSize;
    private final int dim;

    public BatchPrefillActivation(LlamaState state, LlamaConfiguration config, int batchSize, boolean isQ8) {
        this.isQ8 = isQ8;
        this.batchSize = batchSize;
        this.dim = config.dim();
        KernelContext ctx = new KernelContext();
        this.itg = buildGraph(ctx, state).snapshot();
    }

    private TaskGraph buildGraph(KernelContext ctx, LlamaState state) {
        if (isQ8) {
            return new TaskGraph("prefillActivation")
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.wrapXBatch)
                    .task("batchPassthrough", TransformerBatchPrefillKernels::batchPassthrough,
                            ctx, state.wrapXBatch)
                    .persistOnDevice(state.wrapXBatch);
        }
        return new TaskGraph("prefillActivation")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, ctx, state.wrapXBatch)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.embeddingXBatch)
                .task("updateX", TransformerComputeKernels::convertFP16toFP32,
                        ctx, state.embeddingXBatch, state.wrapXBatch)
                .persistOnDevice(state.wrapXBatch);
    }

    @Override
    public ImmutableTaskGraph getImmutableTaskGraph() {
        return itg;
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler scheduler) {
        if (isQ8) {
            scheduler.addWorkerGrid("prefillActivation.batchPassthrough",
                    WorkerGridFactory.genericWorker(1, 1));
        } else {
            scheduler.addWorkerGrid("prefillActivation.updateX",
                    WorkerGridFactory.genericWorker(batchSize * dim, 128));
        }
        return scheduler;
    }
}
