package org.beehive.gpullama3.backend.tornado.device;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import org.beehive.gpullama3.backend.tornado.TensorCoreSupport;
import org.beehive.gpullama3.backend.tornado.scheduling.SchedulerDetectionService;
import org.beehive.gpullama3.runtime.backend.Device;
import org.beehive.gpullama3.runtime.backend.DeviceCapability;
import org.junit.Test;

/**
 * These run wherever the unit gate runs, including on a machine with no accelerator: resolution
 * never throws, and the no-device answer is a normal one. What they cannot assert is <i>which</i>
 * capabilities a given machine has — that is a property of the machine — so they assert the
 * invariants instead: the answer is stable, the predicates agree with the resolved capabilities,
 * and the cache-key label is the one it always was.
 */
public class TornadoDevicesTest {

    @Test
    public void resolutionNeverThrowsAndIsStable() {
        Device first = TornadoDevices.current();
        assertNotNull(first);
        // Stability is the point, not the speed: a cache-key component that changes underneath the
        // cache produces one compiled program per lookup.
        assertSame(first, TornadoDevices.current());
        assertEquals(first.id(), TornadoDevices.current().id());
    }

    @Test
    public void theIdentityIsTheBackendAndTheDisplayNameTogether() {
        Device device = TornadoDevices.current();
        assertEquals(device.backend(), device.id().backend());
        assertEquals(device.displayName(), device.id().handle());
        assertTrue(
                "a display name must be usable in an error message",
                !device.displayName().isBlank());
    }

    @Test
    public void tensorCoreSupportAgreesWithTheResolvedCapability() {
        assertEquals(
                TornadoDevices.current().capabilities().supports(DeviceCapability.TENSOR_CORE_MMA),
                TensorCoreSupport.isTensorCoreCapableBackend());
    }

    @Test
    public void warpShuffleSupportAgreesWithTheResolvedCapability() {
        assertEquals(
                TornadoDevices.current().capabilities().supports(DeviceCapability.WARP_SHUFFLE),
                SchedulerDetectionService.isWarpShuffleSupported());
    }

    /**
     * Metal parity, task 5→6 follow-up. Deliberately independent of {@link
     * #warpShuffleSupportAgreesWithTheResolvedCapability} — {@code SUBGROUP_SHUFFLE_32} and {@code
     * WARP_SHUFFLE} are verified for different kernels on different backends and must not be
     * conflated (see {@code DeviceCapability.SUBGROUP_SHUFFLE_32}'s javadoc).
     */
    @Test
    public void subgroupShuffle32SupportAgreesWithTheResolvedCapability() {
        assertEquals(
                TornadoDevices.current()
                        .capabilities()
                        .supports(DeviceCapability.SUBGROUP_SHUFFLE_32),
                SchedulerDetectionService.isSubgroupShuffle32Supported());
    }

    /**
     * The specific, narrow claim this capability is allowed to make: granted only where this
     * project measured it (Metal), never inferred from "not NVIDIA" or any other broad rule. A
     * device that is neither NVIDIA-class nor Metal (an OpenCL platform, say) must not pick up this
     * capability by accident.
     */
    @Test
    public void subgroupShuffle32IsNeverGrantedToADeviceThatIsNotMetal() {
        Device device = TornadoDevices.current();
        boolean isMetal =
                "Apple Metal".equals(device.displayName())
                        || device.id().backend().toString().equals("metal");
        if (!isMetal) {
            assertFalse(
                    "SUBGROUP_SHUFFLE_32 is verified for Metal only",
                    device.capabilities().supports(DeviceCapability.SUBGROUP_SHUFFLE_32));
        }
    }

    @Test
    public void theMetalPredicateIsTheAbsenceOfSplitKvAttention() {
        // It reads as a backend question and is a capability question: Metal is where the split-KV
        // kernel fails to JIT, and that is what every call site actually branches on.
        assertEquals(
                !TornadoDevices.current()
                        .capabilities()
                        .supports(DeviceCapability.SPLIT_KV_ATTENTION),
                SchedulerDetectionService.isMetalBackend());
    }

    @Test
    public void aMachineWithoutAnAcceleratorStillHasAStableAnswer() {
        Device device = TornadoDevices.current();
        if ("unavailable".equals(device.displayName())) {
            assertTrue(
                    "no device means no optional capabilities",
                    device.capabilities().asSet().isEmpty());
            assertEquals("none", device.capabilities().fingerprint());
        }
    }
}
