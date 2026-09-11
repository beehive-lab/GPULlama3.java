package org.beehive.gpullama3.golden;

import org.beehive.gpullama3.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Qwen3.8-27B with the prompt ingested in chunks, against the CPU reference.
 *
 * <p>Its own class, and therefore its own JVM: this fixture holds 15.5 GiB on the device and
 * TornadoVM returns freed device memory to its own provider rather than to the driver, so two of
 * these in one process exhaust the card partway through the second.
 *
 * <p>Same bounds as the single-token path. This family's batched projections accumulate in FP32
 * like its single-token ones — it does not use the shared Q8_0 tensor-core GEMM, whose FP16
 * accumulation is why batched prefill cannot meet these bounds for the families that do.
 */
public class Qwen35BatchedPrefillParityAccelTest extends CpuGpuParity {

    @Test
    public void qwen3_8_27b_q4_0_batchedPrefillParity() throws Exception {
        assertParityBatched(Fixture.QWEN3_8_27B_Q4_0, Q8_0_PACKED_DECODE, 32);
    }
}
