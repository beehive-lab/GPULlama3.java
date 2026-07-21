package org.beehive.gpullama3.tornadovm;

import uk.ac.manchester.tornado.api.enums.TornadoVMBackendType;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;

/**
 * Detects whether the active TornadoVM backend can execute the tensor-core
 * (MMA) batch-prefill kernels. TornadoVM lowers the MMA intrinsics
 * ({@code mmaLoadA/B}, {@code mma}, {@code mmaStore}) only on the NVIDIA
 * CUDA backend.
 */
public final class TensorCoreSupport {

    public static boolean isTensorCoreCapableBackend() {
        TornadoVMBackendType backendType = TornadoRuntimeProvider.getTornadoRuntime()
                .getBackend(0)
                .getBackendType();
        return backendType == TornadoVMBackendType.CUDA;
    }
}
