package org.beehive.gpullama3.golden;

import org.beehive.gpullama3.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Qwen3.8-27B ingested in chunks of 64, against the CPU reference.
 *
 * <p>Wider than the prompt: a single chunk, most of it inactive. The active count comes from the chunk rather than from the launch width, and this is where that distinction is load-bearing.
 *
 * <p>One width per class, and therefore per JVM. This fixture holds 15.5 GiB on the device and
 * TornadoVM returns freed device memory to its own provider rather than to the driver, so a second
 * plan in the same process exhausts the card. The widths are separated rather than looped.
 */
public class Qwen35BatchedPrefillWidth64ParityAccelTest extends CpuGpuParity {

    @Test
    public void qwen3_8_27b_q4_0_batchedPrefillParityAt64() throws Exception {
        assertParityBatched(Fixture.QWEN3_8_27B_Q4_0, Q8_0, 64);
    }
}
