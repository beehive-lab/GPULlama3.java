package org.beehive.gpullama3.backend.tornado.plan;

import java.util.Set;
import org.beehive.gpullama3.backend.tornado.lowering.TornadoSupportSets;
import org.beehive.gpullama3.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.gpullama3.backend.tornado.plan.components.fp16.Qwen3FP16PlanComponents;
import org.beehive.gpullama3.backend.tornado.plan.components.q8_0.Qwen3Q8_0PlanComponents;
import org.beehive.gpullama3.inference.state.Qwen3State;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.runtime.model.ArchitectureId;
import org.beehive.gpullama3.runtime.tensor.DataType;

/** Qwen3's plan components — all three plan shapes. */
public final class Qwen3PlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("qwen3");

    @Override
    public ArchitectureId architecture() {
        return ID;
    }

    @Override
    public Set<DataType> supportedDataTypes() {
        return TornadoSupportSets.BOTH_REPRESENTATIONS;
    }

    @Override
    public Set<ExecutionMode> supportedModes() {
        return TornadoSupportSets.EVERY_MODE;
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        Qwen3State typed = PlanStates.expect(Qwen3State.class, state, ID);
        return weights == DataType.F16
                ? new Qwen3FP16PlanComponents(typed, model)
                : new Qwen3Q8_0PlanComponents(typed, model);
    }
}
