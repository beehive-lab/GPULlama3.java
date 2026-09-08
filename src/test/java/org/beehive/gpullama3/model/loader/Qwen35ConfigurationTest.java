package org.beehive.gpullama3.model.loader;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.beehive.gpullama3.model.qwen35.Qwen35Configuration;
import org.junit.Test;

/**
 * The derived geometry of a {@code qwen35} configuration, checked against Qwen3.8-27B's actual
 * metadata.
 *
 * <p>Every number here is one the file states and something else derives, and the two must agree.
 * The delta-net widths in particular appear in no metadata key: they come out of the SSM block, and
 * a wrong derivation gives a model that loads and mis-strides every recurrent layer.
 */
public class Qwen35ConfigurationTest {

    /** Qwen3.8-27B-Q4_0.gguf's metadata block, transcribed. */
    private static Qwen35Configuration qwen38_27b() {
        return new Qwen35Configuration(
                "Q8_0",
                /* dim */ 5120,
                /* hiddenDim */ 17408,
                /* numberOfLayers */ 64,
                /* numberOfNextnLayers */ 1,
                /* numberOfHeads */ 24,
                /* numberOfKeyValueHeads */ 4,
                /* numberOfHeadsKey */ 256,
                /* numberOfHeadsValue */ 256,
                /* fullAttentionInterval */ 4,
                /* ssmConvKernel */ 4,
                /* ssmStateSize */ 128,
                /* ssmGroupCount */ 16,
                /* ssmTimeStepRank */ 48,
                /* ssmInnerSize */ 6144,
                /* ropeDimensionCount */ 64,
                /* vocabularySize */ 248320,
                /* contextLengthModel */ 262144,
                /* contextLength */ 4096,
                /* rmsNormEps */ 1e-6f,
                /* ropeTheta */ 1e7f);
    }

    @Test
    public void attentionHeadWidthIsStatedNotDerived() {
        Qwen35Configuration c = qwen38_27b();
        assertEquals(256, c.headSize());
        // 5120 / 24 is 213, which is what deriving it would give — and would mis-address every
        // head. That the two differ is the whole point of the check.
        assertFalse(c.headSize() == c.dim() / c.numberOfHeads());
        assertEquals(24 * 256 * 2, c.queryGateDim());
        assertEquals(24 * 256, c.attentionOutputInputDim());
        assertEquals(4 * 256, c.kvDim());
        assertEquals(6, c.kvMul());
    }

    @Test
    public void deltaNetWidthsComeOutOfTheSsmBlock() {
        Qwen35Configuration c = qwen38_27b();
        assertEquals(128, c.headKeyDim());
        assertEquals(16, c.numberOfKeyHeads());
        assertEquals(48, c.numberOfValueHeads());
        assertEquals(128, c.headValueDim());
        assertEquals(2048, c.deltaNetKeyDim());
        assertEquals(6144, c.deltaNetValueDim());
        // The convolution runs over q ‖ k ‖ v, and this is attn_qkv's stated width.
        assertEquals(10240, c.deltaNetConvDim());
        assertEquals(3, c.valueHeadsPerKeyHead());
        assertEquals(48 * 128 * 128, c.deltaNetStateSize());
        assertEquals(3 * 10240, c.convStateSize());
    }

    @Test
    public void threeLayersInFourRecur() {
        Qwen35Configuration c = qwen38_27b();
        assertEquals(48, c.numberOfLayers() - c.numberOfAttentionLayers());
        assertEquals(16, c.numberOfAttentionLayers());
        assertEquals(65, c.numberOfBlocks());

        // The interval counts from one: 3, 7, 11 … attend.
        assertTrue(c.isRecurrentLayer(0));
        assertTrue(c.isRecurrentLayer(2));
        assertFalse(c.isRecurrentLayer(3));
        assertFalse(c.isRecurrentLayer(63));
        // The MTP block is a full attention block and is never recurrent.
        assertFalse(c.isRecurrentLayer(64));
    }

    @Test
    public void rotaryWidthIsSmallerThanTheHead() {
        Qwen35Configuration c = qwen38_27b();
        assertEquals(64, c.ropeDimensionCount());
        assertTrue(c.ropeDimensionCount() < c.headSize());
    }
}
