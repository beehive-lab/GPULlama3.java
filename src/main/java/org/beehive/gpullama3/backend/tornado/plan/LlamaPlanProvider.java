package org.beehive.gpullama3.backend.tornado.plan;

import java.util.Set;
import org.beehive.gpullama3.backend.tornado.lowering.TornadoSupportSets;
import org.beehive.gpullama3.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.gpullama3.backend.tornado.plan.components.fp16.LlamaFP16PlanComponents;
import org.beehive.gpullama3.backend.tornado.plan.components.q8_0.LlamaQ8_0PlanComponents;
import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.runtime.model.ArchitectureId;
import org.beehive.gpullama3.runtime.tensor.DataType;

/**
 * Llama's plan components — all three plan shapes, both representations.
 *
 * <p>A file of its own, like every provider: adding an architecture must not mean editing a file
 * that contains other families.
 */
public final class LlamaPlanProvider implements TornadoPlanProvider {

    private static final ArchitectureId ID = ArchitectureId.of("llama");

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
        LlamaState typed = PlanStates.expect(LlamaState.class, state, ID);
        return weights == DataType.F16
                ? new LlamaFP16PlanComponents(typed, model)
                : new LlamaQ8_0PlanComponents(typed, model);
    }
}
