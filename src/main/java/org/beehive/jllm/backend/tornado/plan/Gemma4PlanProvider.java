package org.beehive.gpullama3.backend.tornado.plan;

import java.util.Set;
import org.beehive.gpullama3.backend.tornado.lowering.TornadoSupportSets;
import org.beehive.gpullama3.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.gpullama3.backend.tornado.plan.components.fp16.Gemma4FP16PlanComponents;
import org.beehive.gpullama3.backend.tornado.plan.components.q8_0.Gemma4Q8_0PlanComponents;
import org.beehive.gpullama3.inference.state.Gemma4State;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.runtime.model.ArchitectureId;
import org.beehive.gpullama3.runtime.tensor.DataType;

/**
 * Gemma4's plan components. No program description exists for it yet, which is a separate fact: the
 * legacy plan is what it has always run.
 */
public final class Gemma4PlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("gemma4");

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
        return TornadoSupportSets.STANDARD_ONLY;
    }

    @Override
    public SingleTokenForwardPlanComponents components(DataType weights, State state, Model model) {
        Gemma4State typed = PlanStates.expect(Gemma4State.class, state, ID);
        return weights == DataType.F16
                ? new Gemma4FP16PlanComponents(typed, model)
                : new Gemma4Q8_0PlanComponents(typed, model);
    }
}
