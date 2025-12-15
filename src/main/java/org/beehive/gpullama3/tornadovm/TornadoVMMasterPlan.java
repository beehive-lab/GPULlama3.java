package org.beehive.gpullama3.tornadovm;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.tornadovm.layerplanner.base.QuantizationPlannerFactory;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class TornadoVMMasterPlan {
    public static final boolean ENABLE_TORNADOVM_INIT_TIME = Boolean.parseBoolean(System.getProperty("llama.EnableTimingForTornadoVMInit", "False"));

    private final State state;
    private final Configuration config;
    public TornadoExecutionPlan executionPlan;
    GenericLayerPlanner tornadoVMLayerPlanner;

    public TornadoVMMasterPlan(State state, Model model) {
        this.tornadoVMLayerPlanner = createPlanner(state, model);
        this.executionPlan = createExecutionPlan();
        this.state = state;
        this.config = model.configuration();
    }

    /**
     * Initializes the TornadoVM plan for GPU acceleration with optional timing. This method handles: 1. Creation of the TornadoVM master plan 2. Warming up the JIT compiler for better performance 3.
     * Copying read-only model weights to the GPU
     *
     * @param state
     *         The model state containing KV cache
     * @param model
     *         The Llama model instance
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

    private TornadoExecutionPlan createExecutionPlan() {
        var taskGraphs = tornadoVMLayerPlanner.getImmutableTaskGraphs();
        var taskGraphArray = taskGraphs.toArray(new ImmutableTaskGraph[taskGraphs.size()]);
        return new TornadoExecutionPlan(taskGraphArray);
    }

    private GenericLayerPlanner createPlanner(State state, Model model) {
        // ========== STEP 1: Detect Quantization Type ==========
        GGMLType weightType = model.weights().getWeightType();

        // ========== STEP 2: Route via Factory ==========
        // Factory handles all model × quantization combinations
        GenericLayerPlanner basePlanner = QuantizationPlannerFactory.create(weightType, state, model);

        return basePlanner;
    }

    /**
     * Determines whether the NVIDIA-specific scheduler should be used based on the current
     * hardware backend and the model type.
     * <p>
     * The scheduler is used only if the runtime is targeting an NVIDIA backend and the model is not of type {@code MISTRAL}. If either the hardware is not NVIDIA or the model is {@code MISTRAL}, the
     * NVIDIA-specific scheduler should not be used.
     *
     * @param model
     *         the model whose type may affect the scheduler decision
     * @return {@code true} if the NVIDIA-specific scheduler should be used; {@code false} otherwise
     */

    /**
     * Executes the forward pass of a LLaMA transformer model using TornadoVM acceleration. This method processes the transformer layers in sequence for a particular token position in the context
     * window.
     *
     * <p>The execution happens in three phases:
     * <ol>
     *   <li>Initial token embedding lookup (already done before calling this method)</li>
     *   <li>Sequential processing through each transformer layer using TornadoVM</li>
     *   <li>Final projection to logits using TornadoVM</li>
     * </ol>
     *
     * @param position
     *         The current position in the sequence being processed
     * @return FloatTensor containing the output logits for token prediction
     */

    // int pos, ModelPlanner
    public FloatArray tornadoVMForwardExecuteLayered(int position) {
        // @formatter:off
        // 1. Execute the preprocessing graph (e.g., input preparation, memory initialization)
        executionPlan.withGraph(getPreprocessingGraphIndex())
                .withGridScheduler(tornadoVMLayerPlanner.getGridScheduler())
                .execute();

        // Set the position in the state object (used by attention layers)
        state.positionHolder.set(0, position);
        state.temp.clear();
        state.tempFFN.clear();

        // 2. Execute each transformer layer graph sequentially
        // Each graph computes attention and feed-forward transformations for one layer
        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            executionPlan.withGraph(getLayerGraphIndex(layer))
                    .withGridScheduler(tornadoVMLayerPlanner.getGridScheduler())
                    .execute();
        }
        state.tempLogits.clear(); // Clear the intermediate logits tensor -> set to 0f
        state.wrapLogits.clear(); // Clear the output logits tensor -> set to 0f
        // 3. Execute the final graph that projects the last hidden state to output logits
        executionPlan.withGraph(getFinalLogitsGraphIndex())
                .withGridScheduler(tornadoVMLayerPlanner.getGridScheduler())
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
     *
     * @param layerIndex
     *         Index of the transformer layer (0-based)
     */
    private int getLayerGraphIndex(int layerIndex) {
        return 1 + layerIndex;
    }

    /**
     * Returns the graph index for the final projection to logits.
     */
    private int getFinalLogitsGraphIndex() {
        return tornadoVMLayerPlanner.getImmutableTaskGraphs().size() - 1;
    }

    /// Execute the forward pass of the LLaMA transformer model using TornadoVM acceleration just once to copy the data into the read-only data layer.
    public void forceCopyInReadOnlyDataLayered() {
        // Execute all TornadoVM graphs
        state.wrapX.clear();
        state.positionHolder.init(0);

        // Execute activation update graph
        executionPlan.withGraph(0).withGridScheduler(tornadoVMLayerPlanner.getGridScheduler()).execute();

        // Execute layer processing graphs
        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            executionPlan.withGraph(layer + 1).withGridScheduler(tornadoVMLayerPlanner.getGridScheduler()).execute();
        }

        // Execute logits graph
        executionPlan.withGraph(config.numberOfLayers() + 1).withGridScheduler(tornadoVMLayerPlanner.getGridScheduler()).execute();
    }

    /**
     * Frees the device memory allocated for the TornadoVM execution plan. This method should be called when the execution plan is no longer needed to release resources and avoid memory leaks.
     */
    public void freeTornadoExecutionPlan() {
        executionPlan.freeDeviceMemory();
    }
}
