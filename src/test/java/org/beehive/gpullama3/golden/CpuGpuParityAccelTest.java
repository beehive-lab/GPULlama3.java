package org.beehive.gpullama3.golden;

import org.beehive.gpullama3.golden.GoldenFixture.Fixture;
import org.junit.Test;

import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

/**
 * T1.5 — CPU↔GPU parity, Class B.
 *
 * <p>Runs the same fixture and prompt through the CPU path and the GPU path and compares the
 * logits. The CPU path is the reference, as specified in
 * {@code verification-gates.md} §CPU↔GPU parity. Cross-backend comparison is never bit-exact —
 * different orders of the same arithmetic — so it uses a tolerance, and NaN/Inf on either side
 * fails.
 *
 * <p><b>Tolerance, and a documented deviation.</b> The specified bound is
 * {@code |got − ref| ≤ 1e-2·Σ|wᵢaᵢ| + 1e-3}. The {@code Σ|wᵢaᵢ|} term is the absolute sum of the
 * products forming each output element — it measures how much cancellation the dot product
 * involved. It is not computable from a test: the weights and the final hidden state are
 * device-resident and never transferred to the host, and reaching into them from here would mean
 * a production change. This test therefore uses the row's own magnitude as the scale surrogate:
 * {@code 1e-2·max|ref| + 1e-3}. That is the same shape of bound and is stricter than the
 * specified one wherever {@code Σ|wᵢaᵢ| ≥ max|ref|}, which holds whenever the dot product does not
 * cancel — the normal case. Tightening this to the exact bound needs the ops vocabulary of M8,
 * where per-operation absolute sums become observable.
 */
public class CpuGpuParityAccelTest {

    private static final double REL_SCALE = 1e-2;
    private static final double ABS_FLOOR = 1e-3;

    @Test
    public void llama3_2_1b_q8_0_cpuGpuParity() throws Exception {
        assertParity(Fixture.LLAMA_3_2_1B_Q8_0);
    }

    @Test
    public void llama3_2_1b_f16_cpuGpuParity() throws Exception {
        assertParity(Fixture.LLAMA_3_2_1B_F16);
    }

    /** Gap between the best and second-best logit — how close the decision was. */
    private static double top1MinusTop2(float[] v) {
        double best = Double.NEGATIVE_INFINITY;
        double second = Double.NEGATIVE_INFINITY;
        for (float f : v) {
            if (f > best) {
                second = best;
                best = f;
            } else if (f > second) {
                second = f;
            }
        }
        return best - second;
    }

    private void assertParity(Fixture fixture) throws Exception {
        Path model = GoldenFixture.locate(fixture);
        if (model == null) {
            System.out.println("[SKIP] environment absent — " + GoldenFixture.absentMessage(fixture));
            assumeTrue("environment absent: fixture " + fixture.fileName, false);
        }
        if (!TupleInfo.acceleratorPresent()) {
            System.out.println("[SKIP] environment absent — no TornadoVM device");
            assumeTrue("environment absent: no accelerator", false);
        }

        GoldenCapture.Result cpu = GoldenCapture.capture(model, false);
        // Teacher-force the GPU along the CPU's tokens so every compared row shares the CPU's
        // context. Without this the comparison is meaningless past the first near-tie: greedy
        // decoding is autoregressive, so one tipped tie sends the paths into different histories
        // and later rows compare unrelated states rather than arithmetic.
        GoldenCapture.Result gpu = GoldenCapture.capture(model, true, cpu.tokenIds);

        assertEquals("compared row count", cpu.rows.size(), gpu.rows.size());

        for (int r = 0; r < cpu.rows.size(); r++) {
            assertFalse("NaN/Inf in CPU logits, row " + r, Envelope.hasNonFinite(cpu.rows.get(r)));
            assertFalse("NaN/Inf in GPU logits, row " + r, Envelope.hasNonFinite(gpu.rows.get(r)));
        }

        // Report where the paths would have diverged if left to decode freely. This is expected
        // to be non-zero on near-ties and is information, not a failure — the tolerance below is
        // the actual gate.
        int argmaxDisagreements = 0;
        for (int r = 0; r < cpu.rows.size(); r++) {
            if (Envelope.argmax(cpu.rows.get(r)) != Envelope.argmax(gpu.rows.get(r))) {
                argmaxDisagreements++;
            }
        }
        System.out.printf("[PARITY] %s argmax disagreements under teacher forcing: %d/%d%n",
                fixture.quantization, argmaxDisagreements, cpu.rows.size());

        // Record the top-1/top-2 margins at each disagreement. A reversal whose margin is far
        // below the observed drift is a near-tie tipping, which is expected across numerical
        // paths; a reversal with a wide margin would be a genuine parity defect. Reported, not
        // asserted -- the disagreements stay unresolved rather than being tolerated away.
        for (int r = 0; r < cpu.rows.size(); r++) {
            float[] ref = cpu.rows.get(r);
            float[] got = gpu.rows.get(r);
            int aRef = Envelope.argmax(ref);
            int aGot = Envelope.argmax(got);
            if (aRef == aGot) {
                continue;
            }
            System.out.printf("  [MARGIN] row %d: cpu picks %d (cpu margin %.6g), gpu picks %d "
                            + "(gpu margin %.6g); cpu logits[%d]=%.6g logits[%d]=%.6g; "
                            + "gpu logits[%d]=%.6g logits[%d]=%.6g%n",
                    r, aRef, top1MinusTop2(ref), aGot, top1MinusTop2(got),
                    aRef, ref[aRef], aGot, ref[aGot],
                    aRef, got[aRef], aGot, got[aGot]);
        }

        int worstRow = -1;
        double worstExcess = 0;
        double worstDiff = 0;
        double worstTol = 0;
        for (int r = 0; r < cpu.rows.size(); r++) {
            float[] ref = cpu.rows.get(r);
            float[] got = gpu.rows.get(r);
            double scale = 0;
            for (float v : ref) {
                scale = Math.max(scale, Math.abs(v));
            }
            double tol = REL_SCALE * scale + ABS_FLOOR;
            for (int i = 0; i < ref.length; i++) {
                double d = Math.abs((double) ref[i] - (double) got[i]);
                if (d - tol > worstExcess) {
                    worstExcess = d - tol;
                    worstRow = r;
                    worstDiff = d;
                    worstTol = tol;
                }
            }
        }

        System.out.printf("[PARITY] %s worst |cpu-gpu|=%.6g against tolerance %.6g (row %d)%n",
                fixture.quantization,
                worstRow < 0 ? 0.0 : worstDiff,
                worstRow < 0 ? 0.0 : worstTol,
                worstRow);

        assertTrue(String.format(
                        "CPU/GPU parity exceeded for %s: |cpu-gpu|=%.6g > tolerance %.6g at row %d",
                        fixture.quantization, worstDiff, worstTol, worstRow),
                worstExcess <= 0);
    }
}
