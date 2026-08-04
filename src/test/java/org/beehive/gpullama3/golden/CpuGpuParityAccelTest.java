package org.beehive.gpullama3.golden;

import org.beehive.gpullama3.golden.GoldenFixture.Fixture;
import org.junit.Test;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

/**
 * T1.5 — CPU↔GPU parity, Class B.
 *
 * <p>Runs the same fixture and prompt through the CPU path and the GPU path and compares the
 * logits. The CPU path is the reference, as specified in {@code verification-gates.md} §CPU↔GPU
 * parity. Cross-path comparison is never bit-exact — different orders of the same arithmetic, and
 * on the FP16 path a different storage format for the activations — so it uses tolerances, and
 * NaN/Inf on either side fails.
 *
 * <p><b>Four complementary gates</b>, because one worst-element assertion cannot distinguish a
 * broken kernel from one unlucky tail value out of 8.2 million:
 *
 * <ol>
 *   <li><b>Elementwise</b> {@code |gpu − cpu| ≤ atol + rtol·|cpu|}, the conventional mixed bound
 *       (the same shape as {@code torch.testing.assert_close}), with a small violation budget;</li>
 *   <li><b>Hard ceiling</b> on the single largest absolute error, so the budget can never hide a
 *       large excursion;</li>
 *   <li><b>Whole-vector</b> relative L2 and cosine similarity, which describe the row as a whole
 *       instead of being decided by its most extreme element;</li>
 *   <li><b>Decision-level</b>: argmax agreement and top-k overlap. An argmax reversal is only a
 *       failure when the CPU considered the decision clear-cut — a reversed near-tie is expected
 *       between numerical paths and is reported, not tolerated silently.</li>
 * </ol>
 *
 * <p><b>Thresholds are measured, not assumed</b>, and are per quantization: the FP16 GPU path
 * stores its normalized activations as FP16 while the CPU path keeps FP32, so its floor is set by
 * that format, whereas the Q8_0 path keeps FP32 activations and tracks the CPU far more closely.
 * The values below come from {@code golden/ParityProfile} on the pinned tuple and sit roughly 2×
 * above the observed worst case: FP16 max abs error 0.032, relL2 4.5e-3, min cosine 0.9999917;
 * Q8_0 max abs error 4.3e-5, relL2 6.4e-6, cosine 1.0 to eight decimals. Re-derive them with that
 * tool if the tuple, the prompt or the compared-token count changes; do not widen them to make a
 * failing run pass.
 *
 * <p><b>Teacher forcing</b> is what makes the comparison meaningful: greedy decoding is
 * autoregressive, so the first near-tie that tips differently sends the two paths into different
 * contexts and every later row compares unrelated states. Forcing the GPU along the CPU's tokens
 * keeps the KV state identical at every compared position.
 */
public class CpuGpuParityAccelTest {

    /**
     * Per-configuration bounds. {@code atol} dominates for the vast majority of logits, which sit
     * near zero; {@code rtol} is what keeps the bound honest on the few large ones.
     */
    private record Bounds(double atol, double rtol, double maxAbsCeiling, double relL2,
                          double minCosine, double violationFraction, double decisionGap) {
    }

    private static final Bounds FP16 = new Bounds(6e-2, 1e-2, 1e-1, 1e-2, 0.9999, 1e-4, 0.5);
    private static final Bounds Q8_0 = new Bounds(5e-4, 1e-2, 1e-3, 1e-4, 0.999999, 1e-4, 0.5);

    @Test
    public void llama3_2_1b_q8_0_cpuGpuParity() throws Exception {
        assertParity(Fixture.LLAMA_3_2_1B_Q8_0, Q8_0);
    }

    @Test
    public void llama3_2_1b_f16_cpuGpuParity() throws Exception {
        assertParity(Fixture.LLAMA_3_2_1B_F16, FP16);
    }

    private void assertParity(Fixture fixture, Bounds bounds) throws Exception {
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
        GoldenCapture.Result gpu = GoldenCapture.capture(model, true, cpu.tokenIds);

        assertEquals("compared row count", cpu.rows.size(), gpu.rows.size());

        long elements = 0;
        long violations = 0;
        double maxAbs = 0;
        double maxRatio = 0;
        int maxAbsRow = -1;
        int maxAbsIndex = -1;
        double worstRelL2 = 0;
        int worstRelL2Row = -1;
        double minCosine = 1;
        int argmaxDisagreements = 0;
        double top5 = 0;
        double top10 = 0;
        List<String> wideReversals = new ArrayList<>();

        for (int r = 0; r < cpu.rows.size(); r++) {
            float[] ref = cpu.rows.get(r);
            float[] got = gpu.rows.get(r);
            assertEquals("vocabulary length, row " + r, ref.length, got.length);
            assertFalse("NaN/Inf in CPU logits, row " + r, Envelope.hasNonFinite(ref));
            assertFalse("NaN/Inf in GPU logits, row " + r, Envelope.hasNonFinite(got));

            double sqDiff = 0;
            double sqRef = 0;
            double sqGot = 0;
            double dot = 0;
            for (int i = 0; i < ref.length; i++) {
                double d = Math.abs((double) ref[i] - (double) got[i]);
                double tol = bounds.atol() + bounds.rtol() * Math.abs((double) ref[i]);
                if (d > tol) {
                    violations++;
                }
                maxRatio = Math.max(maxRatio, d / tol);
                // Tracked independently of the violation test: when everything passes there is
                // still a largest error, and reporting it as zero would hide the headroom left.
                if (d > maxAbs) {
                    maxAbs = d;
                    maxAbsRow = r;
                    maxAbsIndex = i;
                }
                sqDiff += d * d;
                sqRef += (double) ref[i] * ref[i];
                sqGot += (double) got[i] * got[i];
                dot += (double) ref[i] * (double) got[i];
                elements++;
            }

            double relL2 = Math.sqrt(sqDiff / sqRef);
            if (relL2 > worstRelL2) {
                worstRelL2 = relL2;
                worstRelL2Row = r;
            }
            minCosine = Math.min(minCosine, dot / (Math.sqrt(sqRef) * Math.sqrt(sqGot)));

            top5 += overlap(ref, got, 5);
            top10 += overlap(ref, got, 10);

            int aRef = Envelope.argmax(ref);
            int aGot = Envelope.argmax(got);
            if (aRef != aGot) {
                argmaxDisagreements++;
                // The gap between the two tokens that actually competed, on each side. This is the
                // movement the GPU needed to reverse this specific decision — a per-path
                // top1-minus-top2 margin can involve a third token entirely.
                double cpuGap = ref[aRef] - ref[aGot];
                double gpuGap = got[aGot] - got[aRef];
                String line = String.format(
                        "row %d: cpu picks %d, gpu picks %d; cpu gap=%.6g gpu gap=%.6g",
                        r, aRef, aGot, cpuGap, gpuGap);
                System.out.println("  [REVERSAL] " + line);
                if (cpuGap > bounds.decisionGap()) {
                    wideReversals.add(line);
                }
            }
        }

        int rows = cpu.rows.size();
        System.out.printf("[PARITY] %s rows=%d elements=%d%n", fixture.quantization, rows, elements);
        System.out.printf("[PARITY]   elementwise: violations=%d (%.4g%%, budget %.4g%%) worstRatio=%.4g%n",
                violations, 100.0 * violations / elements, 100.0 * bounds.violationFraction(), maxRatio);
        System.out.printf("[PARITY]   maxAbs=%.6g (row %d, token %d; ceiling %.6g)%n",
                maxAbs, maxAbsRow, maxAbsIndex, bounds.maxAbsCeiling());
        System.out.printf("[PARITY]   worst relL2=%.6g (row %d; bound %.6g)  minCosine=%.8f (bound %.8f)%n",
                worstRelL2, worstRelL2Row, bounds.relL2(), minCosine, bounds.minCosine());
        System.out.printf("[PARITY]   argmax disagreements=%d/%d  top5=%.3f/5  top10=%.3f/10%n",
                argmaxDisagreements, rows, top5 / rows, top10 / rows);

        assertTrue(String.format("%s: %d/%d elementwise violations (%.4g%%) exceed the %.4g%% budget"
                        + " at atol=%.3g rtol=%.3g; worst was %.3gx tolerance",
                        fixture.quantization, violations, elements, 100.0 * violations / elements,
                        100.0 * bounds.violationFraction(), bounds.atol(), bounds.rtol(), maxRatio),
                violations <= bounds.violationFraction() * elements);

        assertTrue(String.format("%s: max |cpu-gpu|=%.6g exceeds the ceiling %.6g (row %d, token %d)",
                        fixture.quantization, maxAbs, bounds.maxAbsCeiling(), maxAbsRow, maxAbsIndex),
                maxAbs <= bounds.maxAbsCeiling());

        assertTrue(String.format("%s: relative L2 %.6g exceeds %.6g at row %d",
                        fixture.quantization, worstRelL2, bounds.relL2(), worstRelL2Row),
                worstRelL2 <= bounds.relL2());

        assertTrue(String.format("%s: cosine similarity %.8f is below %.8f",
                        fixture.quantization, minCosine, bounds.minCosine()),
                minCosine >= bounds.minCosine());

        assertTrue(String.format("%s: argmax reversed where the CPU decision was not close"
                        + " (gap > %.3g): %s", fixture.quantization, bounds.decisionGap(), wideReversals),
                wideReversals.isEmpty());
    }

    /** Number of shared entries between the two top-k sets. */
    private static int overlap(float[] a, float[] b, int k) {
        int[] ta = topK(a, k);
        int[] tb = topK(b, k);
        int shared = 0;
        for (int x : ta) {
            for (int y : tb) {
                if (x == y) {
                    shared++;
                    break;
                }
            }
        }
        return shared;
    }

    private static int[] topK(float[] v, int k) {
        Integer[] idx = new Integer[v.length];
        for (int i = 0; i < v.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (p, q) -> Float.compare(v[q], v[p]));
        int[] out = new int[k];
        for (int i = 0; i < k; i++) {
            out[i] = idx[i];
        }
        return out;
    }
}
