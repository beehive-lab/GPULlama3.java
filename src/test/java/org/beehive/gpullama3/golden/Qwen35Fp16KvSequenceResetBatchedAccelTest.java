package org.beehive.gpullama3.golden;

import org.beehive.gpullama3.backend.tornado.PlanDispatchEvidence;
import org.beehive.gpullama3.backend.tornado.TornadoVMMasterPlan;
import org.junit.Test;

/**
 * {@link Qwen35SequenceResetBatchedAccelTest} with a half-precision key/value cache.
 *
 * <p>The batched prefill/decode plan reaches the same decode attention, so this is the reset case
 * for split-KV under the mode the benchmarks run.
 */
public class Qwen35Fp16KvSequenceResetBatchedAccelTest extends Qwen35SequenceReset {

    static {
        System.setProperty("llama.kvcache.fp16", "true");
    }

    @Override
    void verifyAttentionDispatch(TornadoVMMasterPlan plan) {
        PlanDispatchEvidence.assertQwen35SplitKvAttention(
                PlanDispatchEvidence.gridSchedulerIfAvailable(plan));
    }

    @Test
    public void qwen3_8_27b_q4_0_resetRestoresTheSequenceBatchedOnFp16Kv() throws Exception {
        assertResetRestoresTheSequence(32);
    }
}
