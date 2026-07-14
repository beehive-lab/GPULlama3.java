package org.beehive.gpullama3.tornadovm.layers.type.fp16;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernels;
import org.beehive.gpullama3.tornadovm.kernels.TransformerComputeKernelsLayered;
import org.beehive.gpullama3.inference.weights.tornado.Qwen2TornadoWeights;
import org.beehive.gpullama3.tornadovm.scheduling.WorkerGridFactory;
import org.beehive.gpullama3.tornadovm.scheduling.SchedulerType;
import org.beehive.gpullama3.tornadovm.layers.AbstractLogitsTaskGraph;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

public class LogitsFP16Layer extends AbstractLogitsTaskGraph {

    /** Local workgroup size for the single-workgroup on-device argmax. */
    private static final int SAMPLE_LOCAL = 256;

    /**
     * On-device greedy sampling: append a GPU argmax over the logits and transfer only the
     * sampled token id (1 int) to the host instead of the full vocab logits row. Opt-in via
     * {@code -Dllama.deviceSample=true}; only valid for greedy decoding (LlamaApp disables it
     * for temperature &gt; 0 and non-FP16 models, which still need the full logits host-side).
     */
    public static final boolean DEVICE_SAMPLE = Boolean.getBoolean("llama.deviceSample");

    public LogitsFP16Layer(String name, State state, Weights weights, Configuration config,
            String lastTaskGraphID, SchedulerType schedulerType) {
        super(name, state, weights, config, lastTaskGraphID, schedulerType);
    }

    /**
     * Hook called before any data transfers or tasks. Override to prepend
     * {@code consumeFromDevice} declarations that must precede the bytecode
     * (e.g. KV-cache pass-through in the Phase 4 unified plan).
     */
    protected void configureAdditionalConsumes(TaskGraph logits) {}

    /**
     * Hook called after {@code transferToHost}. Override to append
     * {@code persistOnDevice} declarations (e.g. KV-cache pass-through in Phase 4).
     */
    protected void configureAdditionalPersists(TaskGraph logits) {}

    // @formatter:off
    @Override
    protected TaskGraph setupLogitsTaskGraph(TornadoWeights weights, Configuration config) {
        var logits = new TaskGraph("logits");
        // === Data Setup ===
        configureAdditionalConsumes(logits);
        logits.consumeFromDevice(lastTaskGraphID, state.wrapX);
        logits.transferToDevice(DataTransferMode.EVERY_EXECUTION, state.tempLogits);
        logits.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                // Kernel context
                context,
                // Output buffer
                state.wrapLogits,
                // Intermediate FP16 buffer
                state.wrapXbFP16,
                // Weights
                weights.wclsByteArray.asHalfFloatArray(),
                weights.rms_final_weight_as_floatArray.asFloatArray());

        // === Final RMS Normalization ===
        logits.task("rms_reduce",
                TransformerComputeKernels::reductionOneBlockWithLayer,
                context,
                state.tempLogits,        // output: partial sums + final scale factor
                state.wrapX,             // input: hidden state
                config.dim(),            // dimension
                config.rmsNormEps(),     // epsilon for numerical stability
                state.localSize);        // local workgroup size

        if (schedulerType == SchedulerType.NON_NVIDIA) {
            logits.task("rms_finalize",
                    TransformerComputeKernelsLayered::reductionFinalNormalization,
                    context,
                    state.tempLogits,    // in/out: combines partial sums
                    config.dim(),        // dimension
                    config.rmsNormEps()); // epsilon
        }

        logits.task("rms_apply_fp16",
                TransformerComputeKernels::mapContextWithQuantizeLogits,
                context,
                state.wrapXbFP16,        // output: normalized (FP16)
                state.wrapX,             // input: hidden state
                weights.rms_final_weight_as_floatArray.asFloatArray(),  // RMS weights
                state.tempLogits);       // scale factor from reduction

        // === Vocabulary Projection ===
        logits.task("vocab_proj",
                TransformerComputeKernelsLayered::matrixVectorGeneric,
                context,
                state.wrapXbFP16,                              // input (FP16)
                state.wrapLogits,                              // output
                weights.wclsByteArray.asHalfFloatArray(),      // vocabulary weights
                config.dim(),                                  // input dimension
                config.vocabularySize(),                       // output dimension
                LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS);

        // === Sampling / result transfer ===
        if (DEVICE_SAMPLE) {
            // Greedy argmax on the GPU; only the token id crosses to the host (the full
            // vocab logits row stays device-side — no big D2H copy, no host scan).
            logits.transferToDevice(DataTransferMode.FIRST_EXECUTION, state.sampledToken);
            logits.task("argmax_sample",
                    TransformerComputeKernels::argmaxLogits,
                    context,
                    state.wrapLogits,
                    state.sampledToken,
                    config.vocabularySize(),
                    SAMPLE_LOCAL);
            logits.transferToHost(DataTransferMode.EVERY_EXECUTION, state.sampledToken);
        } else {
            logits.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);
        }
        configureAdditionalPersists(logits);
        return logits;
    }
    // @formatter:on

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        var logitsRMS = WorkerGridFactory.createRmsNormWorker(config.dim(), rmsLocalSize());
        var vocabSizeRowMajor = config.vocabularySize() * LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS;
        var vocabWorker = new WorkerGrid1D(vocabSizeRowMajor);
        vocabWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS, 1, 1);
        tornadoForwardScheduler.addWorkerGrid("logits.rms_reduce", logitsRMS);
        tornadoForwardScheduler.addWorkerGrid("logits.rms_apply_fp16", logitsRMS);
        tornadoForwardScheduler.addWorkerGrid("logits.vocab_proj", vocabWorker);
        if (DEVICE_SAMPLE) {
            var argmaxWorker = new WorkerGrid1D(SAMPLE_LOCAL);   // one workgroup
            argmaxWorker.setLocalWork(SAMPLE_LOCAL, 1, 1);
            tornadoForwardScheduler.addWorkerGrid("logits.argmax_sample", argmaxWorker);
        }
        return tornadoForwardScheduler;
    }

    /** Local workgroup size for RMS norm. Qwen2 requires a smaller group (32 vs 256). */
    protected int rmsLocalSize() {
        return weights instanceof Qwen2TornadoWeights ? 32 : 256;
    }
}
