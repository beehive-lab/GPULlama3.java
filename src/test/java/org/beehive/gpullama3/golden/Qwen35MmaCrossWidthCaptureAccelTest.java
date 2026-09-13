package org.beehive.gpullama3.golden;

import org.beehive.gpullama3.backend.tornado.PlanDispatchEvidence;

// @formatter:off
/**
 * {@link Qwen35CrossWidthCaptureAccelTest} with the tensor-core path selected.
 *
 * <p>Same capture tool, same two-run contract — it writes one width's logits and asserts nothing
 * unless driven twice — but built against the MMA kernels. Earlier cross-width comparisons in this
 * repository were taken without the property and therefore compared the scalar path against itself.
 *
 * <p>The property selects the path; {@link #verifyDispatch} checks the scheduler of the plan this
 * capture was taken on, so a file written from a scalar plan fails here rather than being compared
 * later as MMA evidence.
 *
 * <pre>
 * for W in 32 64; do
 *   JAVA_TOOL_OPTIONS="-Dllama.crossWidth.width=$W -Dllama.crossWidth.out=/tmp/mma-cw-$W.bin" \
 *     ./mvnw -o verify -Paccel-tests -Dtest='Qwen35MmaCrossWidthCaptureAccelTest'
 * done
 * </pre>
 *
 * Compare the fields, not the tail: a 12-byte header, then rows x vocabulary floats, then a token
 * count and that many ids. The width word differs by design.
 */
// @formatter:on
public class Qwen35MmaCrossWidthCaptureAccelTest extends Qwen35CrossWidthCaptureAccelTest {

    static {
        System.setProperty("llama.qwen35.tensorCores", "true");
    }

    @Override
    protected void verifyDispatch(GoldenCapture.Result result, int width) {
        PlanDispatchEvidence.assertQwen35AttentionOutputOnTensorCores(
                result.gridScheduler, width, result.dim);
    }
}
