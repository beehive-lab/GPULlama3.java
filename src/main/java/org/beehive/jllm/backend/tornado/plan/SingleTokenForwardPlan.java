package org.beehive.gpullama3.backend.tornado.plan;

import java.util.ArrayList;
import java.util.List;
import org.beehive.gpullama3.backend.tornado.layers.AbstractLogitsTaskGraph;
import org.beehive.gpullama3.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.gpullama3.backend.tornado.layers.TransformerLayerTaskGraphs;
import org.beehive.gpullama3.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.gpullama3.backend.tornado.plan.layout.SingleTokenForwardTaskGraphLayout;
import org.beehive.gpullama3.model.Model;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

// @formatter:off
/**
 * Topology plan for the N+2 single-token forward pass.
 *
 * <p>Graph layout:
 *
 * <pre>
 *   [0]      activation   ← singleTokenActivation()
 *   [1.N]   layers       ← singleTokenTransformerLayers()
 *   [N+1]    logits       ← singleTokenLogits(String)
 * </pre>
 */
// @formatter:on
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

        AbstractLogitsTaskGraph logits =
                components.singleTokenLogits(layers.getLastFFNLayerTaskGraphID());
        all.add(logits.getImmutableTaskGraph());
        logits.updateGridScheduler(scheduler);

        setGraphs(all, scheduler);
    }

    public SingleTokenForwardTaskGraphLayout getTaskGraphLayout() {
        return taskGraphLayout;
    }
}
