package org.beehive.gpullama3.tornadovm.plan;

import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tornadovm.layers.AbstractLogitsTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.ActivationTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.TransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.gpullama3.tornadovm.plan.layout.SingleTokenForwardTaskGraphLayout;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

import java.util.ArrayList;
import java.util.List;

/**
 * Topology plan for the N+2 single-token forward pass.
 *
 * <p>Graph layout:</p>
 * <pre>
 *   [0]      activation   ← singleTokenActivation()
 *   [1..N]   layers       ← singleTokenTransformerLayers()
 *   [N+1]    logits       ← singleTokenLogits(String)
 * </pre>
 */
public class SingleTokenForwardPlan extends ForwardPlan {

    private final SingleTokenForwardTaskGraphLayout taskGraphLayout;

    public SingleTokenForwardPlan(Model model, SingleTokenForwardPlanComponents components) {
        int N = model.configuration().numberOfLayers();
        this.taskGraphLayout = new SingleTokenForwardTaskGraphLayout(N);

        List<ImmutableTaskGraph> all = new ArrayList<>(N + 2);
        GridScheduler scheduler = new GridScheduler();

        ActivationTaskGraph act = components.singleTokenActivation();
        all.add(act.getImmutableTaskGraph());
        act.updateGridScheduler(scheduler);

        TransformerLayerTaskGraphs layers = components.singleTokenTransformerLayers();
        all.addAll(layers.getFFNLayerImmutableTaskGraphs());
        layers.updateGridScheduler(scheduler);

        AbstractLogitsTaskGraph logits = components.singleTokenLogits(layers.getLastFFNLayerTaskGraphID());
        all.add(logits.getImmutableTaskGraph());
        logits.updateGridScheduler(scheduler);

        setGraphs(all, scheduler);
    }

    public SingleTokenForwardTaskGraphLayout getTaskGraphLayout() {
        return taskGraphLayout;
    }
}
