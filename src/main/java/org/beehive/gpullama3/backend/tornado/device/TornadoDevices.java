package org.beehive.gpullama3.backend.tornado.device;

import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import org.beehive.gpullama3.runtime.backend.BackendId;
import org.beehive.gpullama3.runtime.backend.Device;
import org.beehive.gpullama3.runtime.backend.DeviceCapabilities;
import org.beehive.gpullama3.runtime.backend.DeviceCapability;
import org.beehive.gpullama3.runtime.backend.DeviceId;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

/**
 * The one place this process asks TornadoVM what it is running on.
 *
 * <p>Before this, four call sites asked independently — {@code LoweredPlanSelection}'s device label
 * for the cache key, {@code SchedulerDetectionService} for the scheduler type and two backend
 * predicates, {@code TensorCoreSupport} for MMA, and {@code LlamaBench} inline for a report heading
 * — and all four pinned {@code getBackend(0).getDefaultDevice()}. Four answers that must agree,
 * derived four times, is a disagreement waiting for a machine with two backends installed.
 *
 * <p><b>Resolved once, lazily, and cached.</b> That is not an optimization: a cache-key component
 * must not change underneath the cache. {@code LoweredPlanSelection} learned this the expensive way
 * — it used to read {@code System.getProperty("tornado.device")}, which TornadoVM sets while
 * initializing, so it read {@code "default"} before the first plan and {@code "0"} after, and the
 * same program keyed differently in the first session than in every later one. Asking the runtime
 * rather than the property, and asking once, is the fix.
 *
 * <p><b>No accelerator is a normal answer.</b> Resolution never throws: a machine without a device
 * gets a stable placeholder identity and an empty capability set, so a caller that only wants a
 * label for a log line does not have to handle an exception, and a program compiled without a
 * device is one nothing will execute anyway.
 */
public final class TornadoDevices {

    /**
     * What the label was before a device could be resolved. Kept verbatim: it reaches cache keys.
     */
    private static final String UNAVAILABLE = "unavailable";

    private TornadoDevices() {}

    /** The device this process executes on. Resolved on first use and stable thereafter. */
    public static Device current() {
        return Holder.DEVICE;
    }

    private static final class Holder {
        private static final Device DEVICE = resolve();

        private static Device resolve() {
            try {
                var backend = TornadoRuntimeProvider.getTornadoRuntime().getBackend(0);
                TornadoVMBackendType type = backend.getBackendType();
                String platformName = backend.getDefaultDevice().getPlatformName();
                return new ResolvedDevice(
                        backendId(type),
                        platformName,
                        capabilitiesOf(type, platformName),
                        TornadoNativeArray.ARRAY_HEADER);
            } catch (RuntimeException | LinkageError e) {
                // No accelerator present. The identity still has to be stable and comparable.
                // No accelerator: no native-array header either, which is what a caller mapping
                // for it should reserve.
                return new ResolvedDevice(BackendId.CPU, UNAVAILABLE, DeviceCapabilities.NONE, 0L);
            }
        }
    }

    /**
     * TornadoVM's backend type as a {@link BackendId}. An unrecognised type keeps its own name
     * rather than collapsing to a default — a backend this project has not heard of is still a
     * distinct backend, and merging it into another would make two of them share cache entries.
     *
     * <p><b>PTX is deliberately not a case.</b> TornadoVM removed {@code TornadoVMBackendType.PTX}
     * in favour of CUDA, so naming the constant pins this project to the 5.2.0 release line and
     * fails to compile against anything newer — which is exactly what broke the build. The default
     * branch already produces the identical value: {@link BackendId#of} lower-cases its argument,
     * so {@code of("PTX")} is {@link BackendId#PTX}. Removing the case changes no behaviour on a
     * TornadoVM that still has the constant, and lets the code build against one that does not.
     */
    private static BackendId backendId(TornadoVMBackendType type) {
        return switch (type) {
            case OPENCL -> BackendId.OPENCL;
            case METAL -> BackendId.METAL;
            default -> BackendId.of(type.name());
        };
    }

    /**
     * The four device facts the tree branches on, each preserving exactly the predicate it
     * replaces.
     *
     * <ul>
     *   <li><b>warp shuffle</b> — PTX only. The OpenCL backend compiles {@code
     *       KernelContext.simdShuffleDown} and computes the wrong answer, so the warp GEMVs fall
     *       back to the shared-memory variants there. (The CUDA backend is expected to qualify once
     *       it merges; the old predicate carried that as a TODO and this keeps the same behaviour
     *       rather than pre-enabling it.)
     *   <li><b>tensor-core MMA</b> — CUDA only; TornadoVM lowers the MMA intrinsics nowhere else.
     *   <li><b>split-KV attention</b> — everywhere except Metal, which fails to JIT {@code
     *       processHeadsFlashAttentionSplitKV}.
     *   <li><b>single-pass RMS</b> — the device half of the scheduler type: an NVIDIA platform.
     *       Elsewhere lowering emits an extra {@code *_rms_finalize} task per block. The model half
     *       of that decision is not a device fact and stays in {@code SchedulerDetectionService}.
     *   <li><b>32-wide subgroup shuffle</b> — Metal only, and deliberately not the same grant as
     *       warp shuffle above: verified for exactly the fused Q/K/V projection kernel's five-step
     *       butterfly reduction (Metal parity task, {@code DeviceCapability.SUBGROUP_SHUFFLE_32}),
     *       not for warp shuffle in general. PTX and OpenCL are unaffected by this grant.
     * </ul>
     */
    private static DeviceCapabilities capabilitiesOf(
            TornadoVMBackendType type, String platformName) {
        Set<DeviceCapability> capabilities = new HashSet<>();
        String name = platformName.toLowerCase(Locale.ROOT);
        if ("PTX".equals(type.name())) {
            capabilities.add(DeviceCapability.WARP_SHUFFLE);
        }
        if (type == TornadoVMBackendType.CUDA) {
            capabilities.add(DeviceCapability.TENSOR_CORE_MMA);
        }
        if (type != TornadoVMBackendType.METAL) {
            capabilities.add(DeviceCapability.SPLIT_KV_ATTENTION);
        }
        if (name.contains("nvidia") || name.contains("cuda") || name.contains("ptx")) {
            capabilities.add(DeviceCapability.SINGLE_PASS_RMS);
        }
        if (type == TornadoVMBackendType.METAL) {
            capabilities.add(DeviceCapability.SUBGROUP_SHUFFLE_32);
        }
        // Withheld on OpenCL only: the packed FP16 multiply rounds every product to FP16 before
        // it is accumulated, and on OpenCL that costs enough accuracy for the Llama-shaped FP16
        // QKV projection to fail CPU parity. The identical kernel holds parity on CUDA, so this
        // is a device property, not a kernel defect, and every other backend keeps the fast path.
        if (type != TornadoVMBackendType.OPENCL) {
            capabilities.add(DeviceCapability.PACKED_HALF2_MATH);
        }
        return DeviceCapabilities.of(capabilities);
    }

    private record ResolvedDevice(
            DeviceId id,
            String displayName,
            DeviceCapabilities capabilities,
            long nativeArrayHeaderBytes)
            implements Device {

        ResolvedDevice(
                BackendId backend,
                String platformName,
                DeviceCapabilities capabilities,
                long nativeArrayHeaderBytes) {
            this(
                    DeviceId.of(backend, platformName),
                    platformName,
                    capabilities,
                    nativeArrayHeaderBytes);
        }
    }
}
