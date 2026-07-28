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
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.cublas.CuBlas;
import uk.ac.manchester.tornado.cublas.CuBlasLt;
import uk.ac.manchester.tornado.cublas.enums.CuBlasOperation;

public class LogitsFP16Layer extends AbstractLogitsTaskGraph {

    /**
     * Vocabulary-projection backend for the logits matvec (hybrid-library experiment):
     * {@code jit} (default: TornadoVM JIT matvec), {@code gemmEx} (cuBLAS GemmEx, FP16 in /
     * FP32 out) or {@code lt} (cuBLASLt heuristic-selected FP16 matmul + FP16→FP32 copy task).
     */
    static final String LOGITS_LIB = System.getProperty("llama.logitsLib", "jit");

    /** FP16 logits buffer for the cuBLASLt path (Lt FP16 matmul writes half precision). */
    private HalfFloatArray wrapLogitsFP16;

    public LogitsFP16Layer(String name, State state, Weights weights, Configuration config,
            String lastTaskGraphID, SchedulerType schedulerType) {
        super(name, state, weights, config, lastTaskGraphID, schedulerType);
    }

    /** Elementwise FP16→FP32 copy (cuBLASLt path: the sampler reads FP32 logits on the host). */
    public static void halfToFloat(FloatArray out, HalfFloatArray in, int size) {
        for (@Parallel int i = 0; i < size; i++) {
            float v = in.get(i).getFloat32();
            out.set(i, v);
        }
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
        // Row-major logits(vocab) = Wcls(vocab x dim) · x(dim). cuBLAS is column-major, so
        // Wcls stored row-major is a (dim x vocab) column-major matrix and op(A)=TRANSPOSE
        // recovers Wcls: gemm(OP_T, OP_N, m=vocab, n=1, k=dim, A=Wcls lda=dim, B=x ldb=dim,
        // C=logits ldc=vocab).
        switch (LOGITS_LIB) {
            case "gemmEx" -> logits.libraryTask("vocab_proj", CuBlas::cublasGemmExFP16FP32,
                    CuBlasOperation.CUBLAS_OP_T.operation(), CuBlasOperation.CUBLAS_OP_N.operation(),
                    config.vocabularySize(), 1, config.dim(),
                    1.0f, weights.wclsByteArray.asHalfFloatArray(), config.dim(),
                    state.wrapXbFP16, config.dim(),
                    0.0f, state.wrapLogits, config.vocabularySize());
            case "lt" -> {
                wrapLogitsFP16 = new HalfFloatArray(config.vocabularySize());
                logits.transferToDevice(DataTransferMode.FIRST_EXECUTION, wrapLogitsFP16);
                logits.libraryTask("vocab_proj", CuBlasLt::ltMatmulFP16,
                        CuBlasOperation.CUBLAS_OP_T.operation(), CuBlasOperation.CUBLAS_OP_N.operation(),
                        config.vocabularySize(), 1, config.dim(),
                        1.0f, weights.wclsByteArray.asHalfFloatArray(), config.dim(),
                        state.wrapXbFP16, config.dim(),
                        0.0f, wrapLogitsFP16, config.vocabularySize());
                logits.task("logits_f32", LogitsFP16Layer::halfToFloat,
                        state.wrapLogits, wrapLogitsFP16, config.vocabularySize());
            }
            default -> logits.task("vocab_proj",
                    TransformerComputeKernelsLayered::matrixVectorGeneric,
                    context,
                    state.wrapXbFP16,                              // input (FP16)
                    state.wrapLogits,                              // output
                    weights.wclsByteArray.asHalfFloatArray(),      // vocabulary weights
                    config.dim(),                                  // input dimension
                    config.vocabularySize(),                       // output dimension
                    LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS);
        }

        // === Transfer Results to Host ===
        logits.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);
        configureAdditionalPersists(logits);
        return logits;
    }
    // @formatter:on

    @Override
    public GridScheduler updateGridScheduler(GridScheduler tornadoForwardScheduler) {
        var logitsRMS = WorkerGridFactory.createRmsNormWorker(config.dim(), rmsLocalSize());
        tornadoForwardScheduler.addWorkerGrid("logits.rms_reduce", logitsRMS);
        tornadoForwardScheduler.addWorkerGrid("logits.rms_apply_fp16", logitsRMS);
        if (LOGITS_LIB.equals("jit")) {
            var vocabSizeRowMajor = config.vocabularySize() * LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS;
            var vocabWorker = new WorkerGrid1D(vocabSizeRowMajor);
            vocabWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS, 1, 1);
            tornadoForwardScheduler.addWorkerGrid("logits.vocab_proj", vocabWorker);
        }
        return tornadoForwardScheduler;
    }

    /** Local workgroup size for RMS norm. Qwen2 requires a smaller group (32 vs 256). */
    protected int rmsLocalSize() {
        return weights instanceof Qwen2TornadoWeights ? 32 : 256;
    }
}
