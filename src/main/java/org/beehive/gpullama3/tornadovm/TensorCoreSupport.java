package org.beehive.gpullama3.tornadovm;

import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;

/**
 * Detects whether the active TornadoVM backend can execute the tensor-core
 * (MMA) batch-prefill kernels. TornadoVM lowers the MMA intrinsics
 * ({@code mmaLoadA/B}, {@code mma}, {@code mmaStore}) only on the NVIDIA
 * PTX and CUDA backends; on OpenCL, SPIR-V, and Metal the batch-prefill
 * planners fall back to the portable matvec pipeline ({@code *Generic}
 * planner classes).
 */
public final class TensorCoreSupport {

    private static boolean notified = false;

    private TensorCoreSupport() {
    }

    public static synchronized boolean isTensorCoreCapableBackend() {
        TornadoVMBackendType backendType = TornadoRuntimeProvider.getTornadoRuntime()
                .getBackend(0)
                .getBackendType();
        boolean capable = backendType == TornadoVMBackendType.PTX || backendType == TornadoVMBackendType.CUDA;
        if (!capable && !notified) {
            notified = true;
        }
        return capable;
    }
}
