package org.beehive.gpullama3.runtime.tensor;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import java.util.EnumSet;
import java.util.Set;
import org.junit.Test;

public class DataTypeTest {

    /**
     * The value set is the decision. A new constant appearing here without a runtime that stores or
     * computes with it is the drift this test exists to catch — that is how an execution vocabulary
     * turns back into a copy of the file format's.
     */
    @Test
    public void theValueSetIsWhatTheRuntimeExecutes() {
        assertEquals(
                EnumSet.of(
                        DataType.F32,
                        DataType.F16,
                        DataType.BF16,
                        DataType.Q8_0,
                        DataType.Q4_0,
                        DataType.Q4_K,
                        DataType.Q5_K,
                        DataType.Q6_K),
                EnumSet.allOf(DataType.class));
    }

    @Test
    public void theBlockQuantsTheDeviceCannotExecuteAreTheFormatDecodedOnes() {
        Set<DataType> formatDecoded = EnumSet.noneOf(DataType.class);
        for (DataType type : DataType.values()) {
            if (type.isFormatDecoded()) {
                formatDecoded.add(type);
            }
        }
        assertEquals(
                "the CPU-decoded quantizations, and only those, are format-decoded",
                EnumSet.of(DataType.Q4_0, DataType.Q4_K, DataType.Q5_K, DataType.Q6_K),
                formatDecoded);
    }

    @Test
    public void aFormatDecodedTypeMaterializesAsQ8_0() {
        assertSame(DataType.Q8_0, DataType.Q4_0.materializedFallback());
        assertSame(DataType.Q8_0, DataType.Q4_K.materializedFallback());
        assertSame(DataType.Q8_0, DataType.Q5_K.materializedFallback());
        assertSame(DataType.Q8_0, DataType.Q6_K.materializedFallback());
    }

    /**
     * BF16 is not format-decoded — the CPU materializes a tensor in it and reads it directly — but
     * the GPU has no BF16 kernels, so materializing for the device narrows to F16. That makes it
     * the one type whose fallback is neither itself nor {@code Q8_0}, which is why the fallback is
     * stated per constant rather than derived from {@code isFormatDecoded()}.
     */
    @Test
    public void bf16NarrowsToF16WhenMaterialized() {
        assertSame(DataType.F16, DataType.BF16.materializedFallback());
        assertFalse(
                "BF16 is materialized on the CPU, not decoded during compute",
                DataType.BF16.isFormatDecoded());
        assertFalse(
                "BF16 is a float representation, not a block quantization",
                DataType.BF16.isQuantized());
    }

    /** A type a target can execute needs no fallback, and must not claim one. */
    @Test
    public void anExecutableTypeIsItsOwnMaterialization() {
        assertSame(DataType.F32, DataType.F32.materializedFallback());
        assertSame(DataType.F16, DataType.F16.materializedFallback());
        assertSame(DataType.Q8_0, DataType.Q8_0.materializedFallback());
    }

    @Test
    public void quantizationIsAPropertyOfTheRepresentation() {
        assertFalse(DataType.F32.isQuantized());
        assertFalse(DataType.F16.isQuantized());
        assertTrue(DataType.Q8_0.isQuantized());
        assertTrue(DataType.Q4_K.isQuantized());
    }

    /**
     * Block size and scale layout belong to {@code TensorLayout}. If they ever appear here, {@code
     * Q8_0} with one scale arrangement becomes a different data type from {@code Q8_0} with
     * another, and operations can no longer be parameterized by dtype alone.
     */
    @Test
    public void theTypeCarriesNoStorageDetail() {
        Set<String> methods = new java.util.TreeSet<>();
        for (java.lang.reflect.Method method : DataType.class.getDeclaredMethods()) {
            if (method.getDeclaringClass() == DataType.class && !method.isSynthetic()) {
                methods.add(method.getName());
            }
        }
        assertEquals(
                Set.of(
                        "values",
                        "valueOf",
                        "isQuantized",
                        "isFormatDecoded",
                        "materializedFallback"),
                methods);
    }

    /** A format-decoded type has no storage form, so it can never be a materialization target. */
    @Test
    public void noFormatDecodedTypeIsAFallback() {
        for (DataType type : DataType.values()) {
            assertFalse(
                    type + " falls back to a type nothing can allocate",
                    type.materializedFallback().isFormatDecoded());
        }
    }
}
