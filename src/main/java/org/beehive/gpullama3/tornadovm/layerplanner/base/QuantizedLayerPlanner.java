package org.beehive.gpullama3.tornadovm.layerplanner.base;

import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tornadovm.GenericLayerPlanner;
import org.beehive.gpullama3.tornadovm.layerplanner.strategy.SchedulerDetectionService;
import org.beehive.gpullama3.tornadovm.layerplanner.strategy.SchedulerType;
import uk.ac.manchester.tornado.api.KernelContext;

/**
 * Abstract base for all quantization-specific planners.
 *
 * Contains shared logic that works regardless of model type but depends on quantization. Subclasses: FP16LayerPlanner, Q8_0LayerPlanner, etc.
 */
public abstract class QuantizedLayerPlanner<S extends State, C extends Configuration, W extends Weights> implements GenericLayerPlanner {

    // Common state for all quantizations
    protected static final int LOCAL_WORK_GROUP_SIZE_ALLOC = 32;
    protected static final int THREAD_SCALE_FOR_LOGITS = 8;

    protected final S state;
    protected final C config;
    protected final W weights;
    protected final KernelContext context;
    protected final Model model;
    protected final SchedulerType schedulerType;

    /**
     * Constructor: validate quantization type, extract model components
     */
    protected QuantizedLayerPlanner(S state, Model model) {
        this.state = state;
        this.model = model;
        this.config = (C) model.configuration();
        this.weights = (W) model.weights();
        this.context = new KernelContext();
        this.schedulerType = SchedulerDetectionService.determineSchedulerType(model);

        validateQuantizationType();
    }

    /**
     * Override in subclasses to validate correct quantization format. E.g., FP16LayerPlanner checks: weights instanceof FP16Weights
     */
    protected abstract void validateQuantizationType();

    /**
     * Override in subclasses for model-specific initialization
     */
    protected abstract void initializeLayerComponents();

    // Common helper methods for all quantizations
    protected C getConfig() {
        return config;
    }

    protected W getWeights() {
        return weights;
    }

    protected S getState() {
        return state;
    }
}