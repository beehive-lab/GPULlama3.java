package org.beehive.gpullama3.golden;

import org.beehive.gpullama3.golden.GoldenFixture.Fixture;
import org.junit.Test;

/** Qwen3's logits against the CPU reference. See {@link CpuGpuParity}. */
public class Qwen3CpuGpuParityAccelTest extends CpuGpuParity {

    @Test
    public void qwen3_0_6b_f16_cpuGpuParity() throws Exception {
        assertParity(Fixture.QWEN3_0_6B_F16, FP16);
    }

    @Test
    public void qwen3_0_6b_q8_0_cpuGpuParity() throws Exception {
        assertParity(Fixture.QWEN3_0_6B_Q8_0, Q8_0);
    }
}
