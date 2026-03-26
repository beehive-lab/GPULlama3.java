package org.beehive.gpullama3.tornadovm.layerplanner.base;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tornadovm.layerplanner.GenericLayerPlanner;
import org.beehive.gpullama3.tornadovm.layerplanner.strategy.SchedulerDetectionService;
import org.beehive.gpullama3.tornadovm.layerplanner.strategy.SchedulerType;
import org.beehive.gpullama3.tornadovm.layers.AbstractFFNLayers;
import org.beehive.gpullama3.tornadovm.layers.AbstractLogitsLayer;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;

import java.util.ArrayList;
import java.util.List;

/**
 * Abstract base for all quantization-specific planners.
 *
 * Extracts common state from the model, detects the hardware scheduler type,
 * and assembles the full execution plan via createTornadoInferencePlan().
 * Subclasses (FP16LayerPlanner, Q8_0LayerPlanner) only provide quantization validation.
 */
public abstract class QuantizedLayerPlanner<S extends State, C extends Configuration, W extends Weights>
        implements GenericLayerPlanner {

    protected final S state;
    protected final C config;
    protected final W weights;
    protected final KernelContext context;
    protected final Model model;
    protected final SchedulerType schedulerType;

    protected Activation activationLayer;
    protected AbstractFFNLayers<W, C> ffnLayers;
    protected AbstractLogitsLayer logitsLayer;

    private List<ImmutableTaskGraph> immutableTaskGraphs;
    private GridScheduler gridScheduler;

    @SuppressWarnings("unchecked")
    protected QuantizedLayerPlanner(S state, Model model) {
        this.state = state;
        this.model = model;
        this.config = (C) model.configuration();
        this.weights = (W) model.weights();
        this.context = new KernelContext();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);
        validateQuantizationType();
    }

    /** Validates that the model weights match the expected quantization type. */
    protected abstract void validateQuantizationType();

    /**
     * Creates the TornadoVM inference execution pipeline.
     * It represents the entire Feed-Forward Network (FFN) and consists of:
     * <ul>
     *     <li>Activation layer</li>
     *     <li>FFN layers (N transformer layers, model-specific)</li>
     *     <li>Logits layer</li>
     * </ul>
     * <p>
     * Each component is represented as an {@link ImmutableTaskGraph}, along with a
     * corresponding {@link GridScheduler} configuration that defines how tasks are
     * mapped on the GPU.
     * </p>
     * This method assembles all components into a unified execution pipeline and
     * caches the resulting task graphs and scheduler for reuse across inference runs.
     */
    protected final void createTornadoInferencePlan() {
        List<ImmutableTaskGraph> allTaskGraphs = new ArrayList<>();
        GridScheduler masterScheduler = new GridScheduler();

        // 1. Activation layer (common to all models)
        allTaskGraphs.add(activationLayer.getImmutableTaskGraph());
        activationLayer.updateGridScheduler(masterScheduler);

        // 2. FFN layers (N transformer layers - model-specific)
        allTaskGraphs.addAll(ffnLayers.getFFNLayerImmutableTaskGraphs());
        ffnLayers.updateGridScheduler(masterScheduler);

        // 3. Logits layer (common to all models)
        allTaskGraphs.add(logitsLayer.getImmutableTaskGraph());
        logitsLayer.updateGridScheduler(masterScheduler);

        // Cache for future retrievals
        this.immutableTaskGraphs = allTaskGraphs;
        this.gridScheduler = masterScheduler;
    }

    @Override
    public final List<ImmutableTaskGraph> getImmutableTaskGraphs() {
        return this.immutableTaskGraphs;
    }

    @Override
    public final GridScheduler getGridScheduler() {
        return this.gridScheduler;
    }
}
