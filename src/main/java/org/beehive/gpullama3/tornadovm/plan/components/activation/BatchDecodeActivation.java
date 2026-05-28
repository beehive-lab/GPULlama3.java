package org.beehive.gpullama3.tornadovm.plan.components.activation;

import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernels;
import org.beehive.gpullama3.tornadovm.layers.ActivationTaskGraph;
import org.beehive.gpullama3.tornadovm.scheduling.WorkerGridFactory;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;

/**
 * Decode activation graph with KV-cache pass-through ("decodeActivation").
 *
 * <p>Used in the 2N+3 batch-prefill/decode plan. Consumes
 * {@code wrapKeyCache}/{@code wrapValueCache} from the last batch-prefill layer,
 * converts the single-token embedding to FP32, then re-persists the KV cache so
 * that decode layer 0 can consume it.</p>
 */
public class BatchDecodeActivation implements ActivationTaskGraph {

    private final ImmutableTaskGraph itg;
    private final int dim;

    public BatchDecodeActivation(LlamaState state, LlamaConfiguration config,
                                 String lastBatchLayerId, boolean isQ8) {
        this.dim = config.dim();
        KernelContext ctx = new KernelContext();
        this.itg = buildGraph(ctx, state, lastBatchLayerId, isQ8).snapshot();
    }

    private TaskGraph buildGraph(KernelContext ctx, LlamaState state,
                                 String lastBatchLayerId, boolean isQ8) {
        TaskGraph tg = new TaskGraph("decodeActivation")
                .consumeFromDevice(lastBatchLayerId, state.wrapKeyCache, state.wrapValueCache)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.embeddingX);
        if (isQ8) {
            tg.task("updateX", TransformerComputeKernels::convertQ8_0toFP32,
                    ctx, (ByteArray) state.embeddingX, state.wrapX);
        } else {
            tg.task("updateX", TransformerComputeKernels::convertFP16toFP32,
                    ctx, (HalfFloatArray) state.embeddingX, state.wrapX);
        }
        return tg.persistOnDevice(state.wrapX, state.wrapKeyCache, state.wrapValueCache);
    }

    @Override
    public ImmutableTaskGraph getImmutableTaskGraph() {
        return itg;
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler scheduler) {
        scheduler.addWorkerGrid("decodeActivation.updateX",
                WorkerGridFactory.genericWorker(dim, 128));
        return scheduler;
    }
}
