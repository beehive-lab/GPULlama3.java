package org.beehive.jllm.quality;

import org.beehive.jllm.backend.tornado.PlanDispatchEvidence;
import uk.ac.manchester.tornado.api.GridScheduler;

// @formatter:off
/**
 * {@link Qwen35NllScreenAccelTest} with the tensor-core batched prefill selected.
 *
 * <p>The same passages and the same helper: the unscored prefix is ingested through the batched
 * path and the identical teacher-forced continuation positions are scored. What differs from the
 * parent is only which kernels the prefix went through — `jllm.qwen35.tensorCores` is set here,
 * before this JVM touches the layer class, so the plan is the MMA one, and {@link #verifyDispatch}
 * checks that against the scheduler of the plan this screen scored with, not against the property.
 *
 * <p>Drive it with {@code -Djllm.nllScreen.batch=32} for the batched prefix, and with {@code
 * -Djllm.nllScreen.out=<file>} to write the per-passage report. Without the batch width the plan is
 * the single-token one, which has no batched projection to check, and this class fails rather than
 * scoring a path its name does not describe.
 *
 * <p><b>This is the repository's reused development screen, not independent quality validation.</b>
 * Five passages from this repository's own files, one register, one domain, and it has driven
 * several accept/reject decisions already.
 */
// @formatter:on
public class Qwen35MmaNllScreenAccelTest extends Qwen35NllScreenAccelTest {

    static {
        System.setProperty("jllm.qwen35.tensorCores", "true");
    }

    @Override
    protected void verifyDispatch(GridScheduler grids, int batch, int dim) {
        // At the widths that fill whole GEMM tiles the projection runs as the dequantize-then-GEMM
        // pair; below them, as the direct tensor-core kernel. Either way it is on the tensor cores.
        if (org.beehive.jllm.model.qwen35.Qwen35Configuration.dequantGemmWidth(batch)) {
            PlanDispatchEvidence.assertQwen35AttentionOutputOnDequantGemm(grids, batch, dim, 6144);
        } else {
            PlanDispatchEvidence.assertQwen35AttentionOutputOnTensorCores(grids, batch, dim);
        }
    }
}
