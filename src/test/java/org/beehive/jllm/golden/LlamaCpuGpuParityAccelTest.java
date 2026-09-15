package org.beehive.gpullama3.golden;

import org.beehive.gpullama3.golden.GoldenFixture.Fixture;
import org.junit.Test;

/** Llama's logits against the CPU reference. See {@link CpuGpuParity}. */
public class LlamaCpuGpuParityAccelTest extends CpuGpuParity {

    @Test
    public void llama3_2_1b_q8_0_cpuGpuParity() throws Exception {
        assertParity(Fixture.LLAMA_3_2_1B_Q8_0, Q8_0);
    }

    @Test
    public void llama3_2_1b_f16_cpuGpuParity() throws Exception {
        assertParity(Fixture.LLAMA_3_2_1B_F16, FP16);
    }
}
