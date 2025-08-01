package com.example.tornadovm;

import com.example.auxiliary.Tuple2;
import com.example.inference.state.Phi3State;
import com.example.inference.state.Qwen3State;
import com.example.inference.state.State;
import com.example.model.Configuration;
import com.example.model.Model;
import com.example.model.ModelType;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.TornadoRuntime;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.util.List;
import java.util.Locale;

public class TornadoVMMasterPlan {
    private static final boolean ENABLE_TORNADOVM_INIT_TIME = Boolean.parseBoolean(System.getProperty("llama.EnableTimingForTornadoVMInit", "False"));

    private final State state;
    private final Configuration config;
    public GridScheduler scheduler;
    public TornadoExecutionPlan executionPlan;
    List<ImmutableTaskGraph> taskGraphs;

    public TornadoVMMasterPlan(State state, Model model) {
        TornadoVMLayerPlanner tornadoVMLayerPlanner = createPlanner(state, model);
        Tuple2<List<ImmutableTaskGraph>, GridScheduler> tornadoVMPlan = shouldUseNvidiaScheduler(model)
                ? tornadoVMLayerPlanner.setupTornadoForwardPlanLayered()
                : tornadoVMLayerPlanner.setupTornadoForwardPlanLayeredNonNvidia();
        this.taskGraphs = tornadoVMPlan.getFirst();
        this.scheduler = tornadoVMPlan.getSecond();
        this.state = state;
        this.config = model.configuration();
        this.executionPlan = new TornadoExecutionPlan(taskGraphs.toArray(new ImmutableTaskGraph[taskGraphs.size()]));
    }

    /**
     * Initializes the TornadoVM plan for GPU acceleration with optional timing.
     * This method handles:
     * 1. Creation of the TornadoVM master plan
     * 2. Warming up the JIT compiler for better performance
     * 3. Copying read-only model weights to the GPU
     *
     * @param state The model state containing KV cache
     * @param model The Llama model instance
     * @return The initialized TornadoVMMasterPlan ready for inference
     */
    public static TornadoVMMasterPlan initializeTornadoVMPlan(State state, Model model) {
        // Initialize timing variables outside conditional blocks to avoid scope issues
        long startTime = System.nanoTime();
        long planCreationTime = 0;
        long warmupTime = 0;

        // Start a timing message if enabled
        if (ENABLE_TORNADOVM_INIT_TIME) {
            System.err.println("\nStarting TornadoVM initialization...");
        }

        // 1. Pre-allocate the TornadoVM plan
        TornadoVMMasterPlan tornadoVMPlan = new TornadoVMMasterPlan(state, model);

        // Record time after plan creation
        if (ENABLE_TORNADOVM_INIT_TIME) {
            planCreationTime = System.nanoTime();
            System.err.printf("TornadoVM GPU execution plan creation: %.2f ms\n", (planCreationTime - startTime) / 1_000_000.0);
        }

        // 2. Perform warmup with extra iterations to ensure JIT compilation is complete
        tornadoVMPlan.executionPlan.withPreCompilation(); // Force JIT compilation from Java to GPU code

        // Record time after warmup
        if (ENABLE_TORNADOVM_INIT_TIME) {
            warmupTime = System.nanoTime();
            System.err.printf("Java to GPU JIT compiler warmup: %.2f ms\n", (warmupTime - planCreationTime) / 1_000_000.0);
        }

        // 3. Perform copy-in of read-only weights and objects
        tornadoVMPlan.forceCopyInReadOnlyDataLayered(); // Force copy-in read-only weights

        // Record final timing information
        if (ENABLE_TORNADOVM_INIT_TIME) {
            long copyTime = System.nanoTime();
            System.err.printf("Transfer read-only weights to GPU: %.2f ms\n", (copyTime - warmupTime) / 1_000_000.0);
            System.err.printf("Finished TornadoVM initialization...\n \n");
        }

        model.setTornadoVMPlan(tornadoVMPlan);

        return tornadoVMPlan;
    }

    /**
     * Dispatcher method to select the TornadoVMLayerPlanner for the model.
     */
    TornadoVMLayerPlanner createPlanner(State state, Model model) {
        return switch (model.getModelType()) {
            case LLAMA_3, MISTRAL -> new TornadoVMLayerPlanner(state, model);
            case QWEN_3 -> new Qwen3TornadoVMLayerPlanner((Qwen3State) state, model);
            case PHI_3 -> new Phi3TornadoVMLayerPlanner((Phi3State) state, model);
            case UNKNOWN -> throw new UnsupportedOperationException("Unknown model type");
        };
    }

    /**
     * Determines whether the NVIDIA-specific scheduler should be used based on the current
     * hardware backend and the model type.
     * <p>
     * The scheduler is used only if the runtime is targeting an NVIDIA backend and the model
     * is not of type {@code MISTRAL}. If either the hardware is not NVIDIA or the model is
     * {@code MISTRAL}, the NVIDIA-specific scheduler should not be used.
     *
     * @param model the model whose type may affect the scheduler decision
     * @return {@code true} if the NVIDIA-specific scheduler should be used; {@code false} otherwise
     */
    public static boolean shouldUseNvidiaScheduler(Model model) {
        TornadoRuntime runtime = TornadoRuntimeProvider.getTornadoRuntime();
        String platformName = runtime.getBackend(0).getDefaultDevice().getPlatformName().toLowerCase(Locale.ROOT);

        boolean isNvidia = platformName.contains("nvidia");
        boolean isNotMistral = model.getModelType() != ModelType.MISTRAL;

        boolean result = isNvidia && isNotMistral;

        return result;
    }

    /**
     * Executes the forward pass of a LLaMA transformer model using TornadoVM acceleration.
     *This method processes the transformer layers in sequence for a particular token position in the context
     * window.
     *
     * <p>The execution happens in three phases:
     * <ol>
     *   <li>Initial token embedding lookup (already done before calling this method)</li>
     *   <li>Sequential processing through each transformer layer using TornadoVM</li>
     *   <li>Final projection to logits using TornadoVM</li>
     * </ol>
     *
     *
     * @param position
     *         The current position in the sequence being processed
     * @return FloatTensor containing the output logits for token prediction
     */

    public FloatArray tornadoVMForwardExecuteLayered(int position) {
        // @formatter:off
        // 1. Execute the preprocessing graph (e.g., input preparation, memory initialization)
        executionPlan.withGraph(getPreprocessingGraphIndex())
                .withGridScheduler(scheduler)
                .execute();

        // Set the position in the state object (used by attention layers)
        state.positionHolder.set(0, position);

        // 2. Execute each transformer layer graph sequentially
        // Each graph computes attention and feed-forward transformations for one layer
        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            executionPlan.withGraph(getLayerGraphIndex(layer))
                    .withGridScheduler(scheduler)
                    .execute();
        }

        // 3. Execute the final graph that projects the last hidden state to output logits
        executionPlan.withGraph(getFinalLogitsGraphIndex())
                .withGridScheduler(scheduler)
                .execute();

        // @formatter:on
        // Return the logits (used for token prediction)
        return state.wrapLogits;
    }

    /**
     * Returns the graph index for the pre-processing step (e.g., token embedding).
     */
    private int getPreprocessingGraphIndex() {
        return 0;
    }

    /**
     * Returns the graph index for the given transformer layer.
     * @param layerIndex Index of the transformer layer (0-based)
     */
    private int getLayerGraphIndex(int layerIndex) {
        return 1 + layerIndex;
    }

    /**
     * Returns the graph index for the final projection to logits.
     */
    private int getFinalLogitsGraphIndex() {
        return taskGraphs.size() - 1;
    }

    /// Execute the forward pass of the LLaMA transformer model using TornadoVM acceleration
    /// just once to copy the data into the read-only data layer.
    public void forceCopyInReadOnlyDataLayered() {
        // Execute all TornadoVM graphs
        state.wrapX.init(0.0f);
        state.positionHolder.init(0);

        // Execute activation update graph
        executionPlan.withGraph(0).withGridScheduler(scheduler).execute();

        // Execute layer processing graphs
        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            executionPlan.withGraph(layer + 1).withGridScheduler(scheduler).execute();
        }

        // Execute logits graph
        executionPlan.withGraph(config.numberOfLayers() + 1).withGridScheduler(scheduler).execute();
    }

    /**
     * Frees the device memory allocated for the TornadoVM execution plan.
     * This method should be called when the execution plan is no longer needed
     * to release resources and avoid memory leaks.
     */
    public void freeTornadoExecutionPlan() {
        executionPlan.freeDeviceMemory();
    }
}
