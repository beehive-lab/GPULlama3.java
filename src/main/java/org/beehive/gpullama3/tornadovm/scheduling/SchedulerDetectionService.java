package org.beehive.gpullama3.tornadovm.scheduling;

import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.ModelType;
import uk.ac.manchester.tornado.api.TornadoRuntime;
import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;

import java.util.Locale;

public class SchedulerDetectionService {

    /**
     * Whether the active TornadoVM backend evaluates warp/sub-group shuffle reductions
     * ({@code KernelContext.simdShuffleDown}) correctly. PTX/CUDA do; the OpenCL backend
     * compiles the shuffle but produces incorrect results, so the warp-shuffle GEMV kernels
     * must only run on PTX/CUDA (other backends fall back to the shared-memory GEMVs).
     */
    public static boolean isWarpShuffleSupported() {
        TornadoVMBackendType backendType = TornadoRuntimeProvider.getTornadoRuntime()
                .getBackend(0)
                .getBackendType();
        String backendName = backendType.name();
        return "PTX".equals(backendName);
        // TODO replace lines 23-24 with 26 when CUDA backend is merged
        // return backendType == TornadoVMBackendType.PTX || backendType == TornadoVMBackendType.CUDA;
    }

    /**
     * Whether the active TornadoVM backend is Metal. The Metal backend fails to JIT the
     * multi-workgroup split-KV flash-decoding attention kernel ({@code processHeadsFlashAttentionSplitKV}),
     * so Qwen3 layers must fall back to the single-workgroup-per-head online-softmax kernel there.
     */
    public static boolean isMetalBackend() {
        TornadoVMBackendType backendType = TornadoRuntimeProvider.getTornadoRuntime()
                .getBackend(0)
                .getBackendType();
        return backendType == TornadoVMBackendType.METAL;
    }

    // @formatter:off
    public static SchedulerType determineSchedulerType(Model model) {
        TornadoRuntime tornadoRuntime = TornadoRuntimeProvider.getTornadoRuntime();
        String platformName = tornadoRuntime.getBackend(0)
                .getDefaultDevice()
                .getPlatformName()
                .toLowerCase(Locale.ROOT);

        boolean isNvidia = platformName.contains("nvidia") ||
                platformName.contains("cuda") ||
                platformName.contains("ptx");
        boolean isNotMistral = model.getModelType() != ModelType.MISTRAL;

        return (isNvidia && isNotMistral) ? SchedulerType.NVIDIA : SchedulerType.NON_NVIDIA;
    }
    // @formatter:on
}
