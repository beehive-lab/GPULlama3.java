package org.beehive.gpullama3.tornadovm.layerplanner.strategy;

import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.ModelType;
import uk.ac.manchester.tornado.api.TornadoRuntime;
import uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider;

import java.util.Locale;

public class SchedulerDetectionService {

    public static SchedulerType determineSchedulerType(Model model) {
        TornadoRuntime tornadoRuntime = TornadoRuntimeProvider.getTornadoRuntime();
        String platformName = tornadoRuntime.getBackend(0).getDefaultDevice().getPlatformName().toLowerCase(Locale.ROOT);

        boolean isNvidia = platformName.contains("nvidia") || platformName.contains("cuda") || platformName.contains("ptx");
        boolean isNotMistral = model.getModelType() != ModelType.MISTRAL;

        return (isNvidia && isNotMistral) ? SchedulerType.NVIDIA : SchedulerType.NON_NVIDIA;
    }
}