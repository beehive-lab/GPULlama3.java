package org.beehive.gpullama3.tornadovm.layerplanner.quantization;

import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tornadovm.layerplanner.base.QuantizedLayerPlanner;
import org.beehive.gpullama3.tornadovm.layers.AbstractFFNLayers;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.LogitsQ8_0Layer;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

import java.util.ArrayList;
import java.util.List;

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
