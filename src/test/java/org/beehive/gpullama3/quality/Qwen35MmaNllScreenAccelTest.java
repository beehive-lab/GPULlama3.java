package org.beehive.gpullama3.quality;

import org.beehive.gpullama3.backend.tornado.PlanDispatchEvidence;
import uk.ac.manchester.tornado.api.GridScheduler;

// @formatter:off
/**
 * {@link Qwen35NllScreenAccelTest} with the tensor-core batched prefill selected.
 *
 * <p>The same passages and the same helper: the unscored prefix is ingested through the batched
 * path and the identical teacher-forced continuation positions are scored. What differs from the
 * parent is only which kernels the prefix went through — `llama.qwen35.tensorCores` is set here,
 * before this JVM touches the layer class, so the plan is the MMA one, and {@link #verifyDispatch}
 * checks that against the scheduler of the plan this screen scored with, not against the property.
 *
 * <p>Drive it with {@code -Dllama.nllScreen.batch=32} for the batched prefix, and with {@code
 * -Dllama.nllScreen.out=<file>} to write the per-passage report. Without the batch width the plan
 * is the single-token one, which has no batched projection to check, and this class fails rather
 * than scoring a path its name does not describe.
 *
 * <p><b>This is the repository's reused development screen, not independent quality validation.</b>
 * Five passages from this repository's own files, one register, one domain, and it has driven
 * several accept/reject decisions already.
 */
// @formatter:on
public class Qwen35MmaNllScreenAccelTest extends Qwen35NllScreenAccelTest {

    static {
        System.setProperty("llama.qwen35.tensorCores", "true");
    }

    @Override
    protected void verifyDispatch(GridScheduler grids, int batch, int dim) {
        PlanDispatchEvidence.assertQwen35AttentionOutputOnTensorCores(grids, batch, dim);
    }
}
