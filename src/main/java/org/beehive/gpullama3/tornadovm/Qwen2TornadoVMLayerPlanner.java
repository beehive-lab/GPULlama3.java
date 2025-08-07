package org.beehive.gpullama3.tornadovm;

import org.beehive.gpullama3.auxiliary.Tuple2;
import org.beehive.gpullama3.inference.state.Qwen2State;
import org.beehive.gpullama3.inference.weights.tornado.Qwen2TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.qwen2.Qwen2Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;

import java.util.List;

public class Qwen2TornadoVMLayerPlanner extends TornadoVMLayerPlanner<Qwen2State, Qwen2Configuration, Qwen2TornadoWeights> {

    /**
     * Constructs a TornadoVMLayerPlanner for the given Llama model.
     *
     * @param state
     *         The state object containing model tensors and buffers
     * @param model
     *         The Llama model instance containing configuration and weights
     */
    public Qwen2TornadoVMLayerPlanner(Qwen2State state, Model model) {
        super(state, model);
    }

    @Override
    protected TaskGraph configureLayerDataTransfers(TaskGraph unifiedLayer, int layerIndex) {
        throw new UnsupportedOperationException("configureLayerDataTransfers Not supported yet.");
    }

    @Override
    public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlanLayered() {
        throw new UnsupportedOperationException("setupTornadoForwardPlanLayered Not supported yet.");
    }

    @Override
    public Tuple2<List<ImmutableTaskGraph>, GridScheduler> setupTornadoForwardPlanLayeredNonNvidia() {
        return setupTornadoForwardPlanLayered();
    }

    private GridScheduler setupQwen2GridSchedulersLayeredNonNvidia() {
        throw new UnsupportedOperationException("setupQwen2GridSchedulersLayeredNonNvidia Not supported yet.");
    }
}
