package org.beehive.gpullama3.golden;

import uk.ac.manchester.tornado.api.TornadoDeviceMap;
import uk.ac.manchester.tornado.api.common.TornadoDevice;

/**
 * The pinned tuple a golden is valid on: device, driver, TornadoVM version, backend and the build
 * flags that change numerics. Recorded into every golden's metadata and re-read when comparing, so
 * a run on a different machine downgrades instead of producing a false failure.
 */
public final class TupleInfo {

    private TupleInfo() {
    }

    public static String tornadoVmVersion() {
        Package p = TornadoDeviceMap.class.getPackage();
        String v = p == null ? null : p.getImplementationVersion();
        if (v != null && !v.isBlank()) {
            return v;
        }
        return System.getProperty("tornado.version", "unknown");
    }

    public static TornadoDevice defaultDevice() {
        try {
            TornadoDeviceMap map = new TornadoDeviceMap();
            if (map.getNumBackends() == 0) {
                return null;
            }
            return map.getAllBackends().get(0).getDevice(0);
        } catch (RuntimeException | Error e) {
            return null;
        }
    }

    public static String deviceName() {
        TornadoDevice d = defaultDevice();
        return d == null ? "" : d.getDeviceName();
    }

    public static String backend() {
        TornadoDevice d = defaultDevice();
        return d == null ? "" : d.getTornadoVMBackend().name();
    }

    /**
     * There is no driver-version accessor on {@code TornadoDevice}, so the platform name plus the
     * device's OpenCL C version stand in as the driver half of the tuple.
     */
    public static String driver() {
        TornadoDevice d = defaultDevice();
        return d == null ? "" : d.getPlatformName() + " / " + d.getDeviceOpenCLCVersion();
    }

    /** True when a TornadoVM device is actually usable in this JVM. */
    public static boolean acceleratorPresent() {
        return defaultDevice() != null;
    }
}
