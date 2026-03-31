package org.beehive.gpullama3.tornadovm.layerplanner.model.fp16;

import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tornadovm.layerplanner.QuantizedLayerPlanner;

/**
 * Base for all FP16-quantized layer planners.
 */
public abstract class FP16LayerPlanner<S extends State, C extends Configuration, W extends TornadoWeights> extends QuantizedLayerPlanner<S, C, W> {

    protected FP16LayerPlanner(S state, Model model) {
        super(state, model);
    }

    @Override
    protected void validateQuantizationType() {
        if (this.weights.getWeightType() != GGMLType.F16) {
            throw new IllegalArgumentException("FP16LayerPlanner requires GGMLType.F16, got: " + this.weights.getWeightType());
        }
    }
}
