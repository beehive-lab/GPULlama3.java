package org.beehive.gpullama3.tornadovm.plan;

import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tornadovm.layers.AbstractLogitsTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.ActivationTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.TransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.plan.components.PrefillDecodeForwardPlanComponents;
import org.beehive.gpullama3.tornadovm.plan.layout.PrefillDecodeForwardTaskGraphLayout;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

import java.util.ArrayList;
import java.util.List;

// @formatter:off
/**
 * Topology plan for the N+2 prefill/decode forward pass.
 *
 * <p>Graph layout:</p>
 * <pre>
 *   [0]      activation   ← prefillDecodeActivation()
 *   [1..N]   layers       ← prefillDecodeTransformerLayers()
 *   [N+1]    logits       ← decodeLogits(String)
 * </pre>
 *
 * <p>During prefill, the master plan executes graphs 0..N (skipping logits).
 * During decode, all N+2 graphs run.</p>
 */
// @formatter:on
public class PrefillDecodeForwardPlan extends ForwardPlan {

    private final PrefillDecodeForwardTaskGraphLayout taskGraphLayout;

    public PrefillDecodeForwardPlan(Model model, PrefillDecodeForwardPlanComponents components) {
        int N = model.configuration().numberOfLayers();
        this.taskGraphLayout = new PrefillDecodeForwardTaskGraphLayout(N);

        List<ImmutableTaskGraph> all = new ArrayList<>(N + 2);
        GridScheduler scheduler = new GridScheduler();

        ActivationTaskGraph act = components.prefillDecodeActivation();
        all.add(act.getImmutableTaskGraph());
        act.updateGridScheduler(scheduler);

        TransformerLayerTaskGraphs layers = components.prefillDecodeTransformerLayers();
        all.addAll(layers.getFFNLayerImmutableTaskGraphs());
        layers.updateGridScheduler(scheduler);

        AbstractLogitsTaskGraph logits = components.decodeLogits(layers.getLastFFNLayerTaskGraphID());
        all.add(logits.getImmutableTaskGraph());
        logits.updateGridScheduler(scheduler);

        setGraphs(all, scheduler);
    }

    public PrefillDecodeForwardTaskGraphLayout getTaskGraphLayout() {
        return taskGraphLayout;
    }
}
