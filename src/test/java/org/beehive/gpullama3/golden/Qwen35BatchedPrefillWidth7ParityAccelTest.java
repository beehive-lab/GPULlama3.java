package org.beehive.gpullama3.golden;

import org.beehive.gpullama3.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Qwen3.8-27B ingested in chunks of 7, against the CPU reference.
 *
 * <p>A width that divides nothing. The prompt is not a multiple of it, so the last chunk is partially active and the padding rows have to contribute nothing.
 *
 * <p>One width per class, and therefore per JVM. This fixture holds 15.5 GiB on the device and
 * TornadoVM returns freed device memory to its own provider rather than to the driver, so a second
 * plan in the same process exhausts the card. The widths are separated rather than looped.
 */
public class Qwen35BatchedPrefillWidth7ParityAccelTest extends CpuGpuParity {

    @Test
    public void qwen3_8_27b_q4_0_batchedPrefillParityAt7() throws Exception {
        assertParityBatched(Fixture.QWEN3_8_27B_Q4_0, Q8_0, 7);
    }
}
