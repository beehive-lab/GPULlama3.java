package org.beehive.gpullama3.golden;

import org.beehive.gpullama3.golden.GoldenFixture.Fixture;
import org.junit.Test;

/** Phi3's logits against the CPU reference. See {@link CpuGpuParity}. */
public class Phi3CpuGpuParityAccelTest extends CpuGpuParity {

    @Test
    public void phi3_mini_4k_f16_cpuGpuParity() throws Exception {
        assertParity(Fixture.PHI3_MINI_4K_F16, FP16);
    }

    @Test
    public void phi3_mini_4k_q8_0_cpuGpuParity() throws Exception {
        assertParity(Fixture.PHI3_MINI_4K_Q8_0, Q8_0);
    }
}
