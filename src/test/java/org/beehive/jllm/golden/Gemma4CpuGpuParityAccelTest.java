package org.beehive.jllm.golden;

import org.beehive.jllm.golden.GoldenFixture.Fixture;
import org.junit.Test;

/**
 * Gemma 4's logits against the CPU reference, one case per representation. See {@link
 * CpuGpuParity}.
 *
 * <p>This family reached an accelerator before it reached this gate: it was verified on the CPU and
 * on Apple-Silicon OpenCL, and never on CUDA. A GPU-versus-GPU comparison cannot see a defect that
 * moves the whole GPU, so the CPU is the only reference that can.
 */
public class Gemma4CpuGpuParityAccelTest extends CpuGpuParity {

    @Test
    public void gemma4E2bQ8_0CpuGpuParity() throws Exception {
        assertParity(Fixture.GEMMA_4_E2B_Q8_0, Q8_0);
    }

    /** The file is BF16; the plan it selects is the FP16 one, so these are the FP16 bounds. */
    @Test
    public void gemma4E2bBf16CpuGpuParity() throws Exception {
        assertParity(Fixture.GEMMA_4_E2B_BF16, FP16);
    }

    // No Q4_0 case: Gemma4PlanProvider declares F16 and Q8_0 only, and the file does not load at
    // all yet -- its per_layer_token_embd is Q5_K, which LongIndexedTensor does not read. Testing a
    // representation the provider does not claim would be asserting a capability nobody offers.
}
