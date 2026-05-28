package org.beehive.gpullama3.tornadovm.plan;

import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tornadovm.layers.AbstractLogitsLayer;
import org.beehive.gpullama3.tornadovm.layers.ActivationTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.TransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.plan.components.PrefillDecodeForwardPlanComponents;
import org.beehive.gpullama3.tornadovm.plan.layout.PrefillDecodeForwardTaskGraphLayout;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

import java.util.ArrayList;
import java.util.List;

/**
 * Topology plan for the N+2 prefill/decode forward pass.
 *
 * <p>Graph layout:</p>
 * <pre>
 *   [0]      decodeActivation
 *   [1..N]   decode transformer layers
 *   [N+1]    logits
 * </pre>
 *
 * <p>During prefill, the master plan executes graphs 0..N (skipping logits).
 * During decode, all N+2 graphs run.</p>
 */
public class PrefillDecodeForwardPlan extends ForwardPlan {

    private final PrefillDecodeForwardTaskGraphLayout taskGraphLayout;

    public PrefillDecodeForwardPlan(Model model, PrefillDecodeForwardPlanComponents components) {
        int N = model.configuration().numberOfLayers();
        this.taskGraphLayout = new PrefillDecodeForwardTaskGraphLayout(N);

        List<ImmutableTaskGraph> all = new ArrayList<>(N + 2);
        GridScheduler scheduler = new GridScheduler();

        ActivationTaskGraph act = components.decodeActivation();
        all.add(act.getImmutableTaskGraph());
        act.updateGridScheduler(scheduler);

        TransformerLayerTaskGraphs layers = components.prefillDecodeLayers();
        all.addAll(layers.getFFNLayerImmutableTaskGraphs());
        layers.updateGridScheduler(scheduler);

        AbstractLogitsLayer logits = components.decodeLogits(layers.getLastFFNLayerTaskGraphID());
        all.add(logits.getImmutableTaskGraph());
        logits.updateGridScheduler(scheduler);

        setGraphs(all, scheduler);
    }

    public PrefillDecodeForwardTaskGraphLayout getTaskGraphLayout() {
        return taskGraphLayout;
    }
}
