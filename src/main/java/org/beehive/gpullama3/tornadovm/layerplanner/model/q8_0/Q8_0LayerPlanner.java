package org.beehive.gpullama3.tornadovm.layerplanner.model.q8_0;

import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tornadovm.layerplanner.QuantizedLayerPlanner;

/**
 * Base for all Q8_0-quantized layer planners.
 */
public abstract class Q8_0LayerPlanner<S extends State, C extends Configuration, W extends TornadoWeights>
        extends QuantizedLayerPlanner<S, C, W> {

    protected Q8_0LayerPlanner(S state, Model model) {
        super(state, model);
    }

    @Override
    protected void validateQuantizationType() {
        if (this.weights.getWeightType() != GGMLType.Q8_0) {
            throw new IllegalArgumentException("Q8_0LayerPlanner requires GGMLType.Q8_0, got: " + this.weights.getWeightType());
        }
    }
}
