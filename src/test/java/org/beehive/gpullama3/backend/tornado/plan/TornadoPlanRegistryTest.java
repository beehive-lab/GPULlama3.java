package org.beehive.gpullama3.backend.tornado.plan;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Set;
import org.beehive.gpullama3.runtime.model.ArchitectureId;
import org.beehive.gpullama3.runtime.tensor.DataType;
import org.junit.Test;

/**
 * Families that have not migrated must be <b>absent</b>, not present-and-empty: absence is what
 * makes {@code ForwardPlanFactory} keep answering for them, and a provider declaring no modes would
 * silently take that answer away.
 */
public class TornadoPlanRegistryTest {

    @Test
    public void theMigratedProvidersDeclareTheInventoriedMatrix() {
        var llama = provider("llama");
        assertEquals(
                "both representations",
                Set.of(DataType.F16, DataType.Q8_0),
                llama.supportedDataTypes());
        assertEquals(
                "all three plan shapes", Set.of(ExecutionMode.values()), llama.supportedModes());

        var mistral = provider("mistral");
        assertEquals(
                "both representations",
                Set.of(DataType.F16, DataType.Q8_0),
                mistral.supportedDataTypes());
        assertEquals(
                "STANDARD only, as the factory has always enforced",
                Set.of(ExecutionMode.STANDARD),
                mistral.supportedModes());
    }

    /**
     * A STANDARD-only family declares exactly that, for every such family rather than one of them.
     *
     * <p>Devstral's {@code PREFILL_DECODE not yet supported for DEVSTRAL_2 + Q4_K} was read as a
     * possible policy-resolution or diagnostic defect — a request for PREFILL_DECODE coming back
     * labelled BATCH_PREFILL_DECODE. It is not: each mode reaches the registry through its own
     * factory entry point and the refusal names the mode that was actually requested. This pins
     * that, so a future change cannot quietly make one mode answer for another.
     *
     * <p>It also pins the scope. STANDARD_ONLY is a family-level declaration shared by Devstral,
     * Qwen2 and Mistral; it is not specific to Devstral, and not specific to Q4_K — the dtype
     * appears in the message text only because the message names the whole tuple.
     */
    @Test
    public void everyStandardOnlyFamilyDeclaresStandardOnly() {
        for (String architecture : Set.of("devstral", "qwen2", "mistral")) {
            assertEquals(
                    architecture + " is STANDARD-only",
                    Set.of(ExecutionMode.STANDARD),
                    provider(architecture).supportedModes());
        }
    }

    /** Every architecture the factory used to dispatch on now has a provider. */
    @Test
    public void everyArchitectureInTheMatrixIsRegistered() {
        var registered = TornadoPlanRegistry.registered();
        for (String architecture :
                new String[] {
                    "llama",
                    "mistral",
                    "devstral",
                    "qwen2",
                    "deepseek-r1-distill-qwen",
                    "qwen2-moe",
                    "qwen3",
                    "gemma4",
                    "phi3",
                    "granite"
                }) {
            assertTrue(
                    architecture
                            + " must resolve through a provider now that the factory's"
                            + " switches are gone",
                    registered.contains(ArchitectureId.of(architecture)));
        }
        assertEquals("the matrix has ten architectures", 10, registered.size());
    }

    /**
     * Qwen2-MoE: registered, {@code Q8_0} only, and {@code F16} is a refusal rather than a gap.
     *
     * <p>The shape that made the registry's protocol matter. If an unsupported dtype returned an
     * empty result, this family would have looked unmigrated and — while the switch still existed —
     * the switch would have answered for it. Now there is no switch, so an empty result here would
     * surface as "no provider registered", which is a different and false statement.
     */
    @Test
    public void aRegisteredProviderWithAnUnsupportedDtypeIsNotAnUnmigratedArchitecture() {
        var moe = provider("qwen2-moe");
        assertEquals("Q8_0 only", Set.of(DataType.Q8_0), moe.supportedDataTypes());
        assertEquals(
                "STANDARD and batch prefill, but not sequential prefill",
                Set.of(ExecutionMode.STANDARD, ExecutionMode.BATCH_PREFILL_DECODE),
                moe.supportedModes());
        assertTrue(
                "it is registered, which is what distinguishes refusal from absence",
                TornadoPlanRegistry.registered().contains(ArchitectureId.of("qwen2-moe")));
    }

    private static TornadoPlanProvider provider(String architecture) {
        ArchitectureId id = ArchitectureId.of(architecture);
        return TornadoPlanRegistry.discover().stream()
                .filter(p -> p.architecture().equals(id))
                .findFirst()
                .orElseThrow(() -> new AssertionError("no plan provider for " + id));
    }
}
