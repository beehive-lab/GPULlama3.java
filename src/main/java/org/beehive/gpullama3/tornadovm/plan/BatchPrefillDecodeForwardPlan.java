package org.beehive.gpullama3.tornadovm.plan;

import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tornadovm.layers.AbstractLogitsTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.ActivationTaskGraph;
import org.beehive.gpullama3.tornadovm.layers.BatchPrefillTransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.layers.TransformerLayerTaskGraphs;
import org.beehive.gpullama3.tornadovm.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.gpullama3.tornadovm.plan.layout.BatchPrefillDecodeForwardTaskGraphLayout;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

import java.util.ArrayList;
import java.util.List;

/**
 * Topology plan for the 2N+3 batch-prefill/decode forward pass.
 *
 * <p>Graph layout:</p>
 * <pre>
 *   [0]         batch activation    ← batchPrefillActivation(int)
 *   [1..N]      batch layers        ← batchPrefillTransformerLayers(int)
 *   [N+1]       decode activation   ← batchDecodeActivation(String)
 *   [N+2..2N+1] decode layers       ← batchDecodeTransformerLayers()
 *   [2N+2]      logits              ← decodeLogits(String)
 * </pre>
 *
 * <p>During batch prefill, the master plan executes graphs 0..N.
 * During decode, graphs N+1..2N+2 run.</p>
 */
public class BatchPrefillDecodeForwardPlan extends ForwardPlan {

    private final BatchPrefillDecodeForwardTaskGraphLayout taskGraphLayout;

    public BatchPrefillDecodeForwardPlan(Model model, BatchPrefillDecodeForwardPlanComponents components, int batchSize) {
        int N = model.configuration().numberOfLayers();
        this.taskGraphLayout = new BatchPrefillDecodeForwardTaskGraphLayout(N);

        List<ImmutableTaskGraph> all = new ArrayList<>(2 * N + 3);
        GridScheduler scheduler = new GridScheduler();

        ActivationTaskGraph batchAct = components.batchPrefillActivation(batchSize);
        all.add(batchAct.getImmutableTaskGraph());
        batchAct.updateGridScheduler(scheduler);

        BatchPrefillTransformerLayerTaskGraphs batchLayers = components.batchPrefillTransformerLayers(batchSize);
        all.addAll(batchLayers.getLayerImmutableTaskGraphs());
        batchLayers.updateGridScheduler(scheduler);

        ActivationTaskGraph decodeAct = components.batchDecodeActivation(batchLayers.getLastLayerTaskGraphID());
        all.add(decodeAct.getImmutableTaskGraph());
        decodeAct.updateGridScheduler(scheduler);

        TransformerLayerTaskGraphs decodeLayers = components.batchDecodeTransformerLayers();
        all.addAll(decodeLayers.getFFNLayerImmutableTaskGraphs());
        decodeLayers.updateGridScheduler(scheduler);

        AbstractLogitsTaskGraph logits = components.decodeLogits(decodeLayers.getLastFFNLayerTaskGraphID());
        all.add(logits.getImmutableTaskGraph());
        logits.updateGridScheduler(scheduler);

        setGraphs(all, scheduler);
    }

    public BatchPrefillDecodeForwardTaskGraphLayout getTaskGraphLayout() {
        return taskGraphLayout;
    }
}
