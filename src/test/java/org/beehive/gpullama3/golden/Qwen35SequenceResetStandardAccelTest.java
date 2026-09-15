package org.beehive.gpullama3.golden;

import org.junit.Test;

/**
 * The same property on the single-token path, whose layer graphs bind the same recurrent buffers.
 *
 * <p>Its own class for the same device-memory reason as the batched case.
 */
public class Qwen35SequenceResetStandardAccelTest extends Qwen35SequenceReset {

    @Test
    public void qwen3_8_27b_q4_0_resetRestoresTheSequenceStandard() throws Exception {
        assertResetRestoresTheSequence(1);
    }
}
