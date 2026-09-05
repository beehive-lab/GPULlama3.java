package org.beehive.gpullama3.backend.tornado.plan;

import java.util.Set;
import org.beehive.gpullama3.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.runtime.model.ArchitectureId;
import org.beehive.gpullama3.runtime.tensor.DataType;

/**
 * One architecture's <b>legacy plan components</b>, as this backend builds them.
 *
 * <p>The twin of {@code TornadoLoweringProvider}, for the path that is still the default. The
 * lowering answers "build this program"; this answers "build the plan components this family has
 * always used". Both are registered per identity and discovered, and neither is listed centrally.
 *
 * <p>The concrete {@code State} cast that {@code ForwardPlanFactory} used to do per family moves in
 * here, which is where a family's own file can do it safely. That cast is the reason the factory's
 * switch could not simply be deleted (Rule 15's note on {@code ForwardPlanFactory$1}).
 */
public interface TornadoPlanProvider {

    /** The identity these components implement. Two providers claiming one identity is an error. */
    ArchitectureId architecture();

    /** The materialized weight representations this family has components for. */
    Set<DataType> supportedDataTypes();

    /** The execution modes this family has plans for. */
    Set<ExecutionMode> supportedModes();

    /**
     * Builds the components for a representation this provider supports.
     *
     * @throws IllegalArgumentException if the state is not this family's, which is the cast that
     *     used to live in the central factory
     */
    SingleTokenForwardPlanComponents components(DataType weights, State state, Model model);
}
