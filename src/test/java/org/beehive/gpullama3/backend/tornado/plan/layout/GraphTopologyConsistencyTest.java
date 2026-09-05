package org.beehive.gpullama3.backend.tornado.plan.layout;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.beehive.gpullama3.backend.tornado.plan.ExecutionMode;
import org.junit.Test;

/**
 * The memory model multiplies per-layer weight bytes by the number of layer graph families, because
 * the Tornado runtime holds object state per task graph and so allocates a device buffer per graph
 * that binds an array. Getting that count wrong under-predicts by roughly the size of the model —
 * measured at 1856 MiB for Llama-3.2-1B-F16 — and an under-prediction admits a load that then dies
 * part-allocated.
 *
 * <p><b>What this test is for.</b> The count used to be a literal in the memory model. A new mode,
 * or a layout that grew a second family, would have kept reporting {@code EXACT} while predicting
 * the memory of a different plan. Three things now prevent that, and this pins all three:
 *
 * <ol>
 *   <li>each layout declares its own family count beside its graph indices;
 *   <li>{@code totalGraphs == families x N + nonLayerGraphs} is checked, so a layout cannot grow a
 *       family without either updating the count or failing here;
 *   <li>{@link TornadoGraphTopology}'s switches have no {@code default}, so a new {@link
 *       ExecutionMode} does not compile until it answers.
 * </ol>
 */
public class GraphTopologyConsistencyTest {

    /** Every selectable topology's declared families agree with the graphs it lays out. */
    @Test
    public void everySelectableTopologyIsSelfConsistent() {
        for (ExecutionMode mode : ExecutionMode.values()) {
            for (int layers : new int[] {1, 16, 32, 80}) {
                assertTrue(
                        mode
                                + " at "
                                + layers
                                + " layers: totalGraphs ("
                                + TornadoGraphTopology.totalGraphs(mode, layers)
                                + ") must equal families ("
                                + TornadoGraphTopology.layerGraphFamilies(mode, layers)
                                + ") x layers + nonLayerGraphs ("
                                + TornadoGraphTopology.nonLayerGraphs(mode, layers)
                                + "). A layout that grew a layer family without updating its"
                                + " layerGraphFamilies() fails here rather than under-predicting"
                                + " memory silently.",
                        TornadoGraphTopology.verify(mode, layers));
            }
        }
        assertTrue(TornadoGraphTopology.verifyAll());
    }

    /**
     * The family counts themselves, pinned per mode.
     *
     * <p>Exact equalities rather than "batched is larger", because the number is the multiplier on
     * the largest component of the memory plan. It is also the measurement's own conclusion:
     * batched prefill costs 1872 MiB more than single-token against 1856 MiB of per-layer weights,
     * which is one extra family and not two.
     */
    @Test
    public void theFamilyCountsAreTheMeasuredOnes() {
        assertEquals(
                "single-token binds each layer's weights once",
                1,
                TornadoGraphTopology.layerGraphFamilies(ExecutionMode.STANDARD, 16));
        assertEquals(
                "sequential prefill reuses the decode layer graphs, so also once",
                1,
                TornadoGraphTopology.layerGraphFamilies(ExecutionMode.PREFILL_DECODE, 16));
        assertEquals(
                "batched prefill builds a second layer family and binds the weights twice",
                2,
                TornadoGraphTopology.layerGraphFamilies(ExecutionMode.BATCH_PREFILL_DECODE, 16));
    }

    /**
     * Sequential prefill lays out the same number of graphs as single-token.
     *
     * <p>Recorded because it was got wrong once: an earlier note claimed sequential prefill had the
     * same 2N+3 graphs as batched prefill and explained its equal memory cost by the families
     * differing. The graph counts differ too — N+2 against 2N+3 — and the measurement (2365 MiB
     * against 2362) is consistent with the simpler explanation.
     */
    @Test
    public void sequentialPrefillLaysOutTheSameGraphsAsSingleToken() {
        assertEquals(
                TornadoGraphTopology.totalGraphs(ExecutionMode.STANDARD, 16),
                TornadoGraphTopology.totalGraphs(ExecutionMode.PREFILL_DECODE, 16));
        assertEquals(
                "batched prefill is the topology that differs",
                2 * 16 + 3,
                TornadoGraphTopology.totalGraphs(ExecutionMode.BATCH_PREFILL_DECODE, 16));
    }

    /**
     * Every mode is answered.
     *
     * <p>A guard against the switches gaining a {@code default} in a future tidy-up, which would
     * turn "a new mode does not compile" into "a new mode silently reports one family".
     */
    @Test
    public void everyModeIsAnswered() {
        for (ExecutionMode mode : ExecutionMode.values()) {
            assertTrue(
                    mode + " must report at least one layer family",
                    TornadoGraphTopology.layerGraphFamilies(mode, 16) >= 1);
            assertTrue(
                    mode + " must lay out more graphs than it has layers",
                    TornadoGraphTopology.totalGraphs(mode, 16) > 16);
        }
        assertEquals(
                "ExecutionMode gained a constant; TornadoGraphTopology and the memory model"
                        + " must state its layer-family count",
                3,
                ExecutionMode.values().length);
    }
}
