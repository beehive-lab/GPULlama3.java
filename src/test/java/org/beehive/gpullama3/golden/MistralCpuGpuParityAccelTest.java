package org.beehive.gpullama3.golden;

import org.beehive.gpullama3.golden.GoldenFixture.Fixture;
import org.junit.Test;

/** Mistral's logits against the CPU reference. See {@link CpuGpuParity}. */
public class MistralCpuGpuParityAccelTest extends CpuGpuParity {

    @Test
    public void mistral_7b_q8_0_cpuGpuParity() throws Exception {
        assertParity(Fixture.MISTRAL_7B_Q8_0, Q8_0);
    }
}
