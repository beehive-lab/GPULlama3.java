package org.beehive.gpullama3.backend.tornado;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.beehive.gpullama3.backend.cpu.Qwen35Forward;
import org.beehive.gpullama3.backend.tornado.tensor.FP32TornadoTensor;
import org.beehive.gpullama3.backend.tornado.tensor.Q4_0TornadoTensor;
import org.beehive.gpullama3.backend.tornado.tensor.Q4_1TornadoTensor;
import org.beehive.gpullama3.backend.tornado.tensor.Q5_KTornadoTensor;
import org.beehive.gpullama3.backend.tornado.tensor.Q6_KTornadoTensor;
import org.beehive.gpullama3.backend.tornado.tensor.TornadoTensor;
import org.beehive.gpullama3.inference.state.Qwen35State;
import org.beehive.gpullama3.inference.weights.standard.Qwen35StandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.Qwen35TornadoWeights;
import org.beehive.gpullama3.model.qwen35.Qwen35;
import org.beehive.gpullama3.model.qwen35.Qwen35Configuration;
import org.beehive.gpullama3.runtime.metrics.MetricsSink;
import org.beehive.gpullama3.runtime.tensor.DataType;
import org.beehive.gpullama3.tensor.standard.ArrayFloatTensor;
import org.beehive.gpullama3.tensor.standard.FloatTensor;
import org.beehive.gpullama3.tensor.standard.Q4_0FloatTensor;
import org.beehive.gpullama3.tensor.standard.Q4_1FloatTensor;
import org.beehive.gpullama3.tensor.standard.Q5_KFloatTensor;
import org.beehive.gpullama3.tensor.standard.Q6_KFloatTensor;
import org.junit.Test;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

// @formatter:off
/**
 * A whole {@code qwen35} forward pass on the device against the same one on the host, on a model
 * small enough to build in a test.
 *
 * <p>The kernel tests settle each operation on its own and the topology test settles the plan's
 * shape. Neither can see the defects that only appear when the parts are wired together: a buffer
 * one task writes and the next does not read, a layer consuming the wrong predecessor's output, a
 * recurrent state that does not carry from one token to the next, an operand bound in the wrong
 * order. Those produce fluent, wrong text on the real model, and a multi-minute load before you
 * find out.
 *
 * <p>Both paths read the <b>same bytes</b>: each weight is generated once and wrapped twice, as the
 * host tensor and as the device tensor for that representation. So a disagreement here is the
 * engine's, not the fixture's.
 *
 * <p>The model is mixed the way the real one is — Q4_1 down projections on the early blocks, Q5_K
 * recurrent outputs, a Q6_K vocabulary projection, Q4_0 elsewhere, F32 norms and SSM parameters —
 * and it has both layer kinds, so both mixers run.
 */
// @formatter:on
public class Qwen35SyntheticParityAccelTest {

    private static void assertFinite(String what, FloatTensor values) {
        for (int i = 0; i < values.size(); i++) {
            float v = values.getFloat(i);
            assertTrue(what + "[" + i + "] is " + v, Float.isFinite(v));
        }
    }

    // @formatter:off
    /**
     * Two positions of the same sequence, host against device.
     *
     * <p>Two rather than one because the first says nothing about the recurrence: at position 0 the
     * convolution window and the delta-net state are zero, so a layer that failed to carry them
     * forward would still agree. The second position is the one that reads what the first wrote.
     */
    // @formatter:on
    @Test
    public void theWholeForwardPassAgreesWithTheHost() throws Exception {
        String previous = System.getProperty("use.tornadovm");
        System.setProperty("use.tornadovm", "true");
        try (Arena owned = Arena.ofShared()) {
            Qwen35Configuration config = Qwen35SyntheticModel.config();
            Qwen35SyntheticModel.Weights both = new Qwen35SyntheticModel(owned).weights(config);

            Qwen35 hostModel = new Qwen35(config, null, both.host(), null);
            Qwen35 deviceModel = new Qwen35(config, null, both.device(), null);
            Qwen35State hostState = new Qwen35State(config, -1);
            Qwen35State deviceState = new Qwen35State(config, -1);

            TornadoVMMasterPlanSingleToken plan =
                    new TornadoVMMasterPlanSingleToken(
                            deviceState, deviceModel, MetricsSink.disabled());
            try {
                int[] tokens = {7, 91};
                for (int position = 0; position < tokens.length; position++) {
                    FloatTensor expected =
                            Qwen35Forward.forward(
                                    hostModel, hostState, tokens[position], position);
                    assertFinite("host logits at position " + position, expected);

                    var actual =
                            TornadoForwardPass.forward(
                                    deviceModel,
                                    deviceState,
                                    tokens[position],
                                    position,
                                    plan);

                    float maxAbs = 0f;
                    for (int i = 0; i < Qwen35SyntheticModel.VOCAB; i++) {
                        maxAbs = Math.max(maxAbs, Math.abs(expected.getFloat(i)));
                    }
                    for (int i = 0; i < Qwen35SyntheticModel.VOCAB; i++) {
                        float device = actual.get(i);
                        assertTrue(
                                "device logit " + i + " at position " + position + " is " + device,
                                Float.isFinite(device));
                        // Relative to the row's own scale: the device reduces in a different order
                        // and through a different attention decomposition, so this is not bit
                        // equality — but it is far tighter than a defect would survive.
                        assertEquals(
                                "logit " + i + " at position " + position,
                                expected.getFloat(i),
                                device,
                                Math.max(1e-3f, maxAbs * 3e-4f));
                    }
                }
            } finally {
                plan.freeTornadoExecutionPlan();
            }
        } finally {
            if (previous == null) {
                System.clearProperty("use.tornadovm");
            } else {
                System.setProperty("use.tornadovm", previous);
            }
        }
    }
}
