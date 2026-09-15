package org.beehive.gpullama3.backend.tornado;

import org.beehive.gpullama3.backend.tornado.device.TornadoDevices;
import org.beehive.gpullama3.runtime.backend.DeviceCapability;

/**
 * Whether the active device can execute the tensor-core (MMA) batch-prefill kernels. TornadoVM
 * lowers the MMA intrinsics ({@code mmaLoadA/B}, {@code mma}, {@code mmaStore}) only on the NVIDIA
 * CUDA backend.
 */
public final class TensorCoreSupport {

    private TensorCoreSupport() {}

    public static boolean isTensorCoreCapableBackend() {
        return TornadoDevices.current().capabilities().supports(DeviceCapability.TENSOR_CORE_MMA);
    }
}
