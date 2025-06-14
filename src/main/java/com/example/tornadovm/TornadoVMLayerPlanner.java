package com.example.tornadovm;

import com.example.auxiliary.Tuple2;
import com.example.model.Configuration;
import com.example.model.Model;
import com.example.loader.weights.State;
import com.example.loader.weights.Weights;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.WorkerGrid;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.ArrayList;
import java.util.List;

// @formatter:off
    /**
     * TornadoVMLayerPlanner orchestrates the execution planning for transformer model inference
     * on GPU using the TornadoVM framework.
     *
     * This class is responsible for:
     * - Creating task graphs for each layer of the neural network
     * - Managing GPU memory transfers between layers
     * - Configuring worker grids for optimal GPU utilization
     * - Setting up the execution schedule for the entire forward pass
     *
     * The planner implements a layered approach where:
     * - Each layer is represented as a separate TaskGraph
     * - Data transfers are optimized to minimize host-device communication
     * - Worker grids are configured for different types of operations (attention, FFN, etc.)
     * - The entire pipeline is scheduled to run efficiently on GPU
     *
     * Key optimizations include:
     * - One-time transfer of static data (weights, caches)
     * - Per-execution transfer of dynamic data (position, activations)
     * - Device-to-device data consumption between layers
     * - Parallelized attention computation across heads
     *
     * @see TaskGraph
     * @see GridScheduler
     * @see Llama
     */
    // @formatter:on
    public class TornadoVMLayerPlanner {
        private static final int LOCAL_WORK_GROUP_SIZE_ALLOC = 32;
        private static final int THREAD_SCALE_FOR_LOGITS = 8;

        private final State state;
        private final Configuration config;
        private final Weights weights;
        private final KernelContext context;

        /**
         * Constructs a TornadoVMLayerPlanner for the given Llama model.
         *
         * @param state
         *         The state object containing model tensors and buffers
         * @param model
         *         The Llama model instance containing configuration and weights
         */
        public TornadoVMLayerPlanner(State state, Model model) {
            this.state = state;
            this.config = model.configuration();
            this.weights = model.weights();
            this.context = new KernelContext();
        }

        public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlanLayered() {
            List<ImmutableTaskGraph> taskGraphs = new ArrayList<>();

            state.temp.init(0.0f);
            state.tempFFN.init(0.0f);
            state.tempLogits.init(0.0f);

            // @formatter:off
            TaskGraph activationUpdate = new TaskGraph("activationUpdate")
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.wrapX)
                    .task("updateX", TransformerComputeKernels::emptyTaskToForceCopyIn, state.wrapX)
                    .persistOnDevice(state.wrapX);
            taskGraphs.add(activationUpdate.snapshot());

        TaskGraph unifiedLayer = null;
        for (int layerIndex =0; layerIndex < config.numberOfLayers(); layerIndex++) {
            unifiedLayer = new TaskGraph("layer_" + layerIndex);
            unifiedLayer.consumeFromDevice(state.wrapX);
            unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                    //Copy-in weights per layer for batched-layered layout
                    weights.rms_att_weightLayered[layerIndex],
                    weights.wqLayered[layerIndex],
                    weights.wkLayered[layerIndex],
                    weights.wvLayered[layerIndex],
                    weights.woLayered[layerIndex],
                    weights.rms_ffn_weightLayered[layerIndex],
                    weights.w1Layered[layerIndex],
                    weights.w2Layered[layerIndex],
                    weights.w3Layered[layerIndex]
            );
            unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);
            unifiedLayer.task("reductionsOneBlock" , TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, state.temp,
                        state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                .task("mapContext", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, state.wrapXb,
                        state.wrapX, weights.rms_att_weightLayered[layerIndex], state.temp)
                .task("qmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context,
                        state.wrapXb,  state.wrapQ, weights.wqLayered[layerIndex], config.dim(), config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("kmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context,
                        state.wrapXb,  state.wrapK, weights.wkLayered[layerIndex], config.dim(), config.kvDim(), LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("vmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context,
                        state.wrapXb,   state.wrapV, weights.wvLayered[layerIndex], config.dim(), config.kvDim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("rope", TransformerComputeKernelsLayered::ropeRotation,context,
                        state.positionHolder, state.wrapQ, state.wrapK, config.kvDim(),
                        config.headSize())
                .task("copyToCaches", TransformerComputeKernelsLayered::copyToCache,
                        state.wrapKeyCache, state.wrapK,  state.wrapValueCache, state.wrapV, state.positionHolder, config.kvDim(), layerIndex, config.contextLength())
                .task("parallel-attention", TransformerComputeKernelsLayered::processHeadsFlashAttention, context,
                        state.wrapQ, state.wrapKeyCache, state.wrapValueCache, state.wrapXb,
                        config.numberOfHeads(), config.headSize(), config.kvDim(), config.kvMul(),
                        state.positionHolder, layerIndex, config.contextLength())
                .task("matmul1", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context,
                        state.wrapXb,  state.wrapX, weights.woLayered[layerIndex], config.dim(), config.dim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("reductionsOneBlockFFN", TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, state.tempFFN,
                        state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                .task("mapContextFFN", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, state.wrapXb,
                        state.wrapX, weights.rms_ffn_weightLayered[layerIndex], state.tempFFN)
                .task("fused_ffn_w1_w3", TransformerComputeKernelsLayered::fusedFeedForwardWithSiLUAndGLUActivation, context,
                        state.wrapXb,   state.wrapHb, weights.w1Layered[layerIndex], weights.w3Layered[layerIndex], config.dim(), config.hiddenDim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                .task("projectionTwo", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context,
                        state.wrapHb, state.wrapX, weights.w2Layered[layerIndex], config.hiddenDim(), config.dim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                .persistOnDevice(
                        state.wrapX
                );
            taskGraphs.add(unifiedLayer.snapshot());
        }

        TaskGraph lastUnifiedLayer = unifiedLayer;
        TaskGraph logits = new TaskGraph("logits")
                .consumeFromDevice(lastUnifiedLayer.getTaskGraphName(),
                        state.wrapX
                )
                .transferToDevice(DataTransferMode.EVERY_EXECUTION,
                        state.tempLogits
                )
                .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        context,
                        state.wrapLogits,
                        weights.wclsHalfFloat,
                        weights.rms_final_weight_as_floatArray
                )
                .task("reductionsOneBlockLogits", TransformerComputeKernels::reductionOneBlockWithLayer, context, state.tempLogits,
                        state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                .task("mapContextLogits", TransformerComputeKernels::reductionOneBlock2WithLogits, context, state.wrapX,
                        weights.rms_final_weight_as_floatArray, state.tempLogits);
                logits = configureQuantizedMatrixVectorFinalWeight(logits);
                logits.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);
        taskGraphs.add(logits.snapshot());
        // @formatter:on

            return new Tuple2<>(taskGraphs, setupGridSchedulersLayered());
        }

        // @formatter:off
        /**
         * Configures the final projection layer in the task graph based on weight quantization type.
         *
         * This method adds a "projection" task to compute the final logits by performing a
         * matrix-vector multiplication between the model's output embeddings and the classifier
         * weights (wcls). The computation kernel used depends on the quantization format.
         *
         * Supported quantization types:
         * - Q8_0: 8-bit quantization with uniform scaling per 32-element block
         * - Q4_0: 4-bit quantization with uniform scaling per 32-element block
         *
         * The task multiplies:
         * - weights.wclsByteArray: Quantized classifier weights (vocab_size x dim)
         * - state.wrapX: Current layer output (dim)
         * - Result: state.wrapLogits: Raw logits (vocab_size)
         *
         * @param logits The existing task graph to extend with the projection operation
         * @return The modified task graph with the projection task added
         * @throws UnsupportedOperationException If weights.weightType is not Q8_0 or Q4_0
         */
        // @formatter:on
        private TaskGraph configureQuantizedMatrixVectorFinalWeight(TaskGraph logits) {
            switch (weights.weightType) {
                case F16:
                case Q8_0:
                case Q4_0:
                    logits.task("projection", TransformerComputeKernelsLayered::matrixVectorGeneric,  //
                            context, state.wrapX, state.wrapLogits, weights.wclsHalfFloat, //
                            config.dim(), config.vocabularySize(), LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS); //
                    break;
                default:
                    throw new UnsupportedOperationException("Unsupported weight quantization type: " + weights.weightType + ". Only Q8_0 and Q4_0 are supported.");
            }
            return logits;
        }

        /**
         * Configures data transfer operations for a specific layer in the neural network task graph.
         *
         * This method manages GPU memory transfers with optimized data movement strategies:
         * This optimization pattern minimizes data movement by:
         * 1. Using one-time transfers for static data
         * 2. Reusing intermediate results already on GPU from previous layers
         * 3. Only transferring //
         * dynamic data that changes per execution
         *
         * @param unifiedLayer
         *         The task graph representing this layer's operations
         * @param layerIndex
         *         Index of the current layer (0-based)
         * @return The configured task graph with appropriate data transfer operations
         */
        private TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
            // First layer: Transfer initial data to device (one-time transfer)
            if (layerIndex == 0) {
                // Transfer all attention-related data: query, key, value matrices and their caches
                unifiedLayer.transferToDevice(DataTransferMode.EVERY_EXECUTION, state.positionHolder, state.temp, state.tempFFN); //
                unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION, //
                        context, state.wrapXb, state.wrapXb2, //
                        state.wrapQ, state.wrapK, state.wrapV, //
                        state.wrapKeyCache, state.wrapValueCache, //
                        state.wrapAtt, state.wrapHb); //
            } else {
                // Subsequent layers: Consume data already on device from previous layer
                unifiedLayer.consumeFromDevice(context, state.wrapXb, state.wrapXb2, //
                        state.wrapQ, state.wrapK, state.wrapV, //
                        state.wrapKeyCache, state.wrapValueCache, //
                        state.wrapAtt, state.wrapHb, //
                        state.positionHolder //
                );
            }
            return unifiedLayer;
        }

        // @formatter:off
        /**
         * Sets up the grid scheduler configuration for a layered neural network forward pass.
         *
         * This method creates and configures worker grids for different types of GPU operations
         * in the transformer/ML model pipeline. Each worker grid defines how work should be
         * distributed across GPU threads (OpenCL work-items or CUDA threads).
         *
         * The method creates several worker profiles:
         * - Single thread operations (activation updates)
         * - RoPE (Rotary Position Embedding) operations
         * - Matrix multiplications with different dimensions
         * - RMS normalization operations
         * - Parallel attention computations
         * - Cache copying operations
         * - Vocabulary projections
         *
         * Each worker grid maps to equivalent OpenCL NDRange or CUDA grid/block configurations:
         * - setGlobalWork() ≈ OpenCL global_work_size ≈ CUDA grid dimensions × block dimensions
         * - setLocalWork() ≈ OpenCL local_work_size ≈ CUDA block dimensions
         *
         * @return GridScheduler configured with all necessary worker grids for the model layers
         */
        // @formatter:on
        private GridScheduler setupGridSchedulersLayered() {
            GridScheduler tornadoForwardScheduler = new GridScheduler();

            // Single worker for tasks running with a single thread
            // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[1,1,1], localWorkSize=[1,1,1])
            // CUDA equivalent: kernel<<<dim3(1,1,1), dim3(1,1,1)>>>
            WorkerGrid singleWorker = new WorkerGrid1D(1);
            singleWorker.setGlobalWork(1, 1, 1);
            singleWorker.setLocalWork(1, 1, 1);

            // config.dim / 2 Worker for RoPE
            // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.dim/2,1,1], localWorkSize=[128,1,1])
            // CUDA equivalent: kernel<<<dim3((config.dim/2+127)/128,1,1), dim3(128,1,1)>>>
            WorkerGrid ropeWorker = new WorkerGrid1D(config.dim() / 2);
            ropeWorker.setGlobalWork(config.dim() / 2, 1, 1);
            ropeWorker.setLocalWork(128, 1, 1);

            // config.dim Worker for Row major access
            // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.dim*LOCAL_WORK_GROUP_SIZE_ALLOC,1,1], localWorkSize=[LOCAL_WORK_GROUP_SIZE_ALLOC,1,1])
            // CUDA equivalent: kernel<<<dim3(config.dim,1,1), dim3(LOCAL_WORK_GROUP_SIZE_ALLOC,1,1)>>>
            int configDimRowMajorGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
            WorkerGrid configDimRowMajorGlobalWorker = new WorkerGrid1D(configDimRowMajorGlobal);
            configDimRowMajorGlobalWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

            // config.kvDim Worker for Row major access
            // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.kvDim*LOCAL_WORK_GROUP_SIZE_ALLOC,1,1], localWorkSize=[LOCAL_WORK_GROUP_SIZE_ALLOC,1,1])
            // CUDA equivalent: kernel<<<dim3(config.kvDim,1,1), dim3(LOCAL_WORK_GROUP_SIZE_ALLOC,1,1)>>>
            int configKvDimRowMajorGlobal = config.kvDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
            WorkerGrid configKvDimRowMajorGlobalWorker = new WorkerGrid1D(configKvDimRowMajorGlobal);
            configKvDimRowMajorGlobalWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

            // config.hiddenDim * 32 Worker for Row major access
            // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.hiddenDim*LOCAL_WORK_GROUP_SIZE_ALLOC,1,1], localWorkSize=[LOCAL_WORK_GROUP_SIZE_ALLOC,1,1])
            // CUDA equivalent: kernel<<<dim3(config.hiddenDim,1,1), dim3(LOCAL_WORK_GROUP_SIZE_ALLOC,1,1)>>>
            int configHiddenDimRowMajor = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
            WorkerGrid configHiddenDimRowMajorWorker = new WorkerGrid1D(configHiddenDimRowMajor);
            configHiddenDimRowMajorWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

            // RMSNorm worker configuration
            // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.dim,1,1], localWorkSize=[256,1,1])
            // CUDA equivalent: kernel<<<dim3((config.dim+255)/256,1,1), dim3(256,1,1)>>>
            WorkerGrid rmsNormWorker = new WorkerGrid1D(config.dim());
            rmsNormWorker.setGlobalWork(config.dim(), 1, 1);  // Set global work size to total dimension
            rmsNormWorker.setLocalWork(256, 1, 1);         // Set local work size to 256 (standard efficient size)

            // Parallel attention worker configuration
            // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.numberOfHeads,1,1], localWorkSize=[4,1,1])
            // CUDA equivalent: kernel<<<dim3((config.numberOfHeads+3)/4,1,1), dim3(4,1,1)>>>
            WorkerGrid parallelAttentionWorker = new WorkerGrid1D(config.numberOfHeads());
            // the global group work size is numberOfHeads * localWorkGroupSize, where the localWorkGroupSize is currently 4
            parallelAttentionWorker.setGlobalWork(config.numberOfHeads() * 8, 1, 1);
            parallelAttentionWorker.setLocalWork(8, 1, 1); // Set local work size to 4 (for parallel attention)

            // Copy to caches worker configuration
            // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.dim,1,1], localWorkSize=[128,1,1])
            // CUDA equivalent: kernel<<<dim3((config.dim+127)/128,1,1), dim3(128,1,1)>>>
            WorkerGrid copyToCachesWorker = new WorkerGrid1D(config.kvDim());
            copyToCachesWorker.setGlobalWork(config.dim(), 1, 1);
            copyToCachesWorker.setLocalWork(128, 1, 1); // Set local work size to 32 (for copying to caches)

            // Map workers to tasks
            tornadoForwardScheduler.addWorkerGrid("activationUpdate.updateX", singleWorker);
            for (int i = 0; i < config.numberOfLayers(); i++) {
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".qmatmul", configDimRowMajorGlobalWorker);
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".kmatmul", configKvDimRowMajorGlobalWorker);
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".vmatmul", configKvDimRowMajorGlobalWorker);
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rope", ropeWorker);
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".matmul1", configDimRowMajorGlobalWorker);
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".projectionTwo", configDimRowMajorGlobalWorker);
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".fused_ffn_w1_w3", configHiddenDimRowMajorWorker);
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlock", rmsNormWorker);
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".mapContext", rmsNormWorker);
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlockFFN", rmsNormWorker);
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".mapContextFFN", rmsNormWorker);
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".parallel-attention", parallelAttentionWorker);
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".copyToCaches", copyToCachesWorker);
            }

            // Vocabulary worker configuration
            // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.vocabularySize,1,1], localWorkSize=[16,1,1])
            // CUDA equivalent: kernel<<<dim3((config.vocabularySize+15)/16,1,1), dim3(16,1,1)>>>
            int vocabSizeRowMajor = config.vocabularySize() * LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS;
            WorkerGrid vocabWorker = new WorkerGrid1D(vocabSizeRowMajor);
            vocabWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS, 1, 1);

            tornadoForwardScheduler.addWorkerGrid("logits.projection", vocabWorker);
            tornadoForwardScheduler.addWorkerGrid("logits.reductionsOneBlockLogits", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("logits.mapContextLogits", rmsNormWorker);

            return tornadoForwardScheduler;
        }

        private GridScheduler setupGridSchedulersLayeredNonNvidia() {
            GridScheduler tornadoForwardScheduler = new GridScheduler();

            // Single worker for tasks running with a single thread
            // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[1,1,1], localWorkSize=[1,1,1])
            // CUDA equivalent: kernel<<<dim3(1,1,1), dim3(1,1,1)>>>
            WorkerGrid singleWorker = new WorkerGrid1D(1);
            singleWorker.setGlobalWork(1, 1, 1);
            singleWorker.setLocalWork(1, 1, 1);

            // config.dim / 2 Worker for RoPE
            // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.dim/2,1,1], localWorkSize=[128,1,1])
            // CUDA equivalent: kernel<<<dim3((config.dim/2+127)/128,1,1), dim3(128,1,1)>>>
            WorkerGrid ropeWorker = new WorkerGrid1D(config.dim() / 2);
            ropeWorker.setGlobalWork(config.dim() / 2, 1, 1);
            ropeWorker.setLocalWork(128, 1, 1);

            // config.dim Worker for Row major access
            // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.dim*LOCAL_WORK_GROUP_SIZE_ALLOC,1,1], localWorkSize=[LOCAL_WORK_GROUP_SIZE_ALLOC,1,1])
            // CUDA equivalent: kernel<<<dim3(config.dim,1,1), dim3(LOCAL_WORK_GROUP_SIZE_ALLOC,1,1)>>>
            int configDimRowMajorGlobal = config.dim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
            WorkerGrid configDimRowMajorGlobalWorker = new WorkerGrid1D(configDimRowMajorGlobal);
            configDimRowMajorGlobalWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

            // config.kvDim Worker for Row major access
            // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.kvDim*LOCAL_WORK_GROUP_SIZE_ALLOC,1,1], localWorkSize=[LOCAL_WORK_GROUP_SIZE_ALLOC,1,1])
            // CUDA equivalent: kernel<<<dim3(config.kvDim,1,1), dim3(LOCAL_WORK_GROUP_SIZE_ALLOC,1,1)>>>
            int configKvDimRowMajorGlobal = config.kvDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
            WorkerGrid configKvDimRowMajorGlobalWorker = new WorkerGrid1D(configKvDimRowMajorGlobal);
            configKvDimRowMajorGlobalWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

            // config.hiddenDim * 32 Worker for Row major access
            // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.hiddenDim*LOCAL_WORK_GROUP_SIZE_ALLOC,1,1], localWorkSize=[LOCAL_WORK_GROUP_SIZE_ALLOC,1,1])
            // CUDA equivalent: kernel<<<dim3(config.hiddenDim,1,1), dim3(LOCAL_WORK_GROUP_SIZE_ALLOC,1,1)>>>
            int configHiddenDimRowMajor = config.hiddenDim() * LOCAL_WORK_GROUP_SIZE_ALLOC;
            WorkerGrid configHiddenDimRowMajorWorker = new WorkerGrid1D(configHiddenDimRowMajor);
            configHiddenDimRowMajorWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC, 1, 1);

            // RMSNorm worker configuration
            // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.dim,1,1], localWorkSize=[256,1,1])
            // CUDA equivalent: kernel<<<dim3((config.dim+255)/256,1,1), dim3(256,1,1)>>>
            WorkerGrid rmsNormWorker = new WorkerGrid1D(config.dim());
            rmsNormWorker.setGlobalWork(config.dim(), 1, 1);  // Set global work size to total dimension
            rmsNormWorker.setLocalWork(256, 1, 1);         // Set local work size to 256 (standard efficient size)

            // Parallel attention worker configuration
            // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.numberOfHeads,1,1], localWorkSize=[4,1,1])
            // CUDA equivalent: kernel<<<dim3((config.numberOfHeads+3)/4,1,1), dim3(4,1,1)>>>
            WorkerGrid parallelAttentionWorker = new WorkerGrid1D(config.numberOfHeads());
            // the global group work size is numberOfHeads * localWorkGroupSize, where the localWorkGroupSize is currently 4
            parallelAttentionWorker.setGlobalWork(config.numberOfHeads() * 8, 1, 1);
            parallelAttentionWorker.setLocalWork(8, 1, 1); // Set local work size to 4 (for parallel attention)

            // Copy to caches worker configuration
            // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.dim,1,1], localWorkSize=[128,1,1])
            // CUDA equivalent: kernel<<<dim3((config.dim+127)/128,1,1), dim3(128,1,1)>>>
            WorkerGrid copyToCachesWorker = new WorkerGrid1D(config.kvDim());
            copyToCachesWorker.setGlobalWork(config.dim(), 1, 1);
            copyToCachesWorker.setLocalWork(128, 1, 1); // Set local work size to 32 (for copying to caches)

            // Map workers to tasks
            tornadoForwardScheduler.addWorkerGrid("activationUpdate.updateX", singleWorker);
            for (int i = 0; i < config.numberOfLayers(); i++) {
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".qmatmul", configDimRowMajorGlobalWorker);
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".kmatmul", configKvDimRowMajorGlobalWorker);
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".vmatmul", configKvDimRowMajorGlobalWorker);
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".rope", ropeWorker);
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".matmul1", configDimRowMajorGlobalWorker);
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".projectionTwo", configDimRowMajorGlobalWorker);
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".fused_ffn_w1_w3", configHiddenDimRowMajorWorker);
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlock", rmsNormWorker);
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".mapContext", rmsNormWorker);
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".reductionsOneBlockFFN", rmsNormWorker);
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".mapContextFFN", rmsNormWorker);
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".parallel-attention", parallelAttentionWorker);
                tornadoForwardScheduler.addWorkerGrid("layer_" + i + ".copyToCaches", copyToCachesWorker);
            }

            // Vocabulary worker configuration
            // OpenCL equivalent: clEnqueueNDRangeKernel(globalWorkSize=[config.vocabularySize,1,1], localWorkSize=[16,1,1])
            // CUDA equivalent: kernel<<<dim3((config.vocabularySize+15)/16,1,1), dim3(16,1,1)>>>
            int vocabSizeRowMajor = config.vocabularySize() * LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS;
            WorkerGrid vocabWorker = new WorkerGrid1D(vocabSizeRowMajor);
            vocabWorker.setLocalWork(LOCAL_WORK_GROUP_SIZE_ALLOC * THREAD_SCALE_FOR_LOGITS, 1, 1);

            tornadoForwardScheduler.addWorkerGrid("logits.projection", vocabWorker);
            tornadoForwardScheduler.addWorkerGrid("logits.reductionsOneBlockLogits", rmsNormWorker);
            tornadoForwardScheduler.addWorkerGrid("logits.mapContextLogits", rmsNormWorker);

            return tornadoForwardScheduler;
        }

        public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlanLayeredNonNvidia() {
            List<ImmutableTaskGraph> taskGraphs = new ArrayList<>();

            state.temp.init(0.0f);
            state.tempFFN.init(0.0f);
            state.tempLogits.init(0.0f);

            // @formatter:off
            TaskGraph activationUpdate = new TaskGraph("activationUpdate")
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.wrapX)
                    .task("updateX", TransformerComputeKernels::emptyTaskToForceCopyIn, state.wrapX)
                    .persistOnDevice(state.wrapX);
            taskGraphs.add(activationUpdate.snapshot());

            TaskGraph unifiedLayer = null;
            for (int layerIndex =0; layerIndex < config.numberOfLayers(); layerIndex++) {
                unifiedLayer = new TaskGraph("layer_" + layerIndex);
                unifiedLayer.consumeFromDevice(state.wrapX);
                unifiedLayer.transferToDevice(DataTransferMode.FIRST_EXECUTION,
                        //Copy-in weights per layer for batched-layered layout
                        weights.rms_att_weightLayered[layerIndex],
                        weights.wqLayered[layerIndex],
                        weights.wkLayered[layerIndex],
                        weights.wvLayered[layerIndex],
                        weights.woLayered[layerIndex],
                        weights.rms_ffn_weightLayered[layerIndex],
                        weights.w1Layered[layerIndex],
                        weights.w2Layered[layerIndex],
                        weights.w3Layered[layerIndex]
                );
                unifiedLayer = configureLayerDataTransfers(unifiedLayer, layerIndex);
                unifiedLayer.task("reductionsOneBlock" , TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, state.temp,
                                state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                        .task("reductionFinalNormalization" , TransformerComputeKernelsLayered::reductionFinalNormalization, context, state.temp,
                                config.dim(), config.rmsNormEps())
                        .task("mapContext", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, state.wrapXb,
                                state.wrapX, weights.rms_att_weightLayered[layerIndex], state.temp)
                        .task("qmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context,
                                state.wrapXb,  state.wrapQ, weights.wqLayered[layerIndex], config.dim(), config.dim(), LOCAL_WORK_GROUP_SIZE_ALLOC)
                        .task("kmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context,
                                state.wrapXb,  state.wrapK, weights.wkLayered[layerIndex], config.dim(), config.kvDim(), LOCAL_WORK_GROUP_SIZE_ALLOC)
                        .task("vmatmul", TransformerComputeKernelsLayered::matrixVectorGeneric, context,
                                state.wrapXb,   state.wrapV, weights.wvLayered[layerIndex], config.dim(), config.kvDim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                        .task("rope", TransformerComputeKernelsLayered::ropeRotation,context,
                                state.positionHolder, state.wrapQ, state.wrapK, config.kvDim(),
                                config.headSize())
                        .task("copyToCaches", TransformerComputeKernelsLayered::copyToCache,
                                state.wrapKeyCache, state.wrapK,  state.wrapValueCache, state.wrapV, state.positionHolder, config.kvDim(), layerIndex, config.contextLength())
                        .task("parallel-attention", TransformerComputeKernelsLayered::processHeadsParallel,
                                state.wrapQ, state.wrapKeyCache, state.wrapValueCache, state.wrapXb,
                                config.numberOfHeads(), config.headSize(), config.kvDim(), config.kvMul(), config.vocabularySize(),
                                state.positionHolder, state.wrapAtt, layerIndex, config.contextLength())
                        .task("matmul1", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context,
                                state.wrapXb,  state.wrapX, weights.woLayered[layerIndex], config.dim(), config.dim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                        .task("reductionsOneBlockFFN", TransformerComputeKernelsLayered::reductionOneBlockWithLayer, context, state.tempFFN,
                                state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                        .task("reductionFinalNormalizationFFN" , TransformerComputeKernelsLayered::reductionFinalNormalization, context, state.tempFFN,
                                config.dim(), config.rmsNormEps())
                        .task("mapContextFFN", TransformerComputeKernelsLayered::reductionOneBlock2WithLayer, context, state.wrapXb,
                                state.wrapX, weights.rms_ffn_weightLayered[layerIndex], state.tempFFN)
                        .task("fused_ffn_w1_w3", TransformerComputeKernelsLayered::fusedFeedForwardWithSiLUAndGLUActivation, context,
                                state.wrapXb,   state.wrapHb, weights.w1Layered[layerIndex], weights.w3Layered[layerIndex], config.dim(), config.hiddenDim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                        .task("projectionTwo", TransformerComputeKernelsLayered::matrixVectorGenericWithResidual, context,
                                state.wrapHb, state.wrapX, weights.w2Layered[layerIndex], config.hiddenDim(), config.dim(),  LOCAL_WORK_GROUP_SIZE_ALLOC)
                        .persistOnDevice(
                                state.wrapX
                        );
                taskGraphs.add(unifiedLayer.snapshot());
            }

            TaskGraph lastUnifiedLayer = unifiedLayer;
            TaskGraph logits = new TaskGraph("logits")
                    .consumeFromDevice(lastUnifiedLayer.getTaskGraphName(),
                            state.wrapX
                    )
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION,
                            state.tempLogits
                    )
                    .transferToDevice(DataTransferMode.FIRST_EXECUTION,
                            context,
                            state.wrapLogits,
                            weights.wclsHalfFloat,
                            weights.rms_final_weight_as_floatArray
                    )
                    .task("reductionsOneBlockLogits", TransformerComputeKernels::reductionOneBlockWithLayer, context, state.tempLogits,
                            state.wrapX, config.dim(), config.rmsNormEps(), state.localSize)
                    .task("reductionFinalNormalizationLogits" , TransformerComputeKernelsLayered::reductionFinalNormalization, context, state.tempLogits,
                            config.dim(), config.rmsNormEps())
                    .task("mapContextLogits", TransformerComputeKernels::reductionOneBlock2WithLogits, context, state.wrapX,
                            weights.rms_final_weight_as_floatArray, state.tempLogits);
            logits = configureQuantizedMatrixVectorFinalWeight(logits);
            logits.transferToHost(DataTransferMode.EVERY_EXECUTION, state.wrapLogits);
            taskGraphs.add(logits.snapshot());
            // @formatter:on

            return new Tuple2<>(taskGraphs, setupGridSchedulersLayeredNonNvidia());
        }

    }
