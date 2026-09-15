package org.beehive.gpullama3.backend.tornado.plan;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.ServiceLoader;
import org.beehive.gpullama3.backend.tornado.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.gpullama3.backend.tornado.plan.components.PrefillDecodeForwardPlanComponents;
import org.beehive.gpullama3.backend.tornado.plan.components.SingleTokenForwardPlanComponents;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.runtime.model.ArchitectureId;
import org.beehive.gpullama3.runtime.tensor.DataType;

/**
 * Finds the plan provider for an architecture — discovery, index and validation only.
 *
 * <p>Holds no family name, for the reason {@code TornadoBackendSupport} holds none: a table of
 * families is a switch with different syntax, and the addition test is written to catch either.
 *
 * <p>{@link #create} returns {@code null} for an architecture with no provider, which is how {@code
 * ForwardPlanFactory} migrates one slice at a time without the matrix changing under it. That is
 * deliberately a different answer from "unsupported": a family whose provider exists and whose mode
 * is not supported gets the existing error, unchanged.
 */
public final class TornadoPlanRegistry {

    private TornadoPlanRegistry() {}

    public static List<TornadoPlanProvider> discover() {
        return discover(Thread.currentThread().getContextClassLoader());
    }

    public static List<TornadoPlanProvider> discover(ClassLoader classLoader) {
        List<TornadoPlanProvider> providers = new ArrayList<>();
        ServiceLoader.load(TornadoPlanProvider.class, classLoader).forEach(providers::add);
        providers.sort(Comparator.comparing(provider -> provider.getClass().getName()));
        return providers;
    }

    /**
     * @throws IllegalStateException if two providers claim one identity, naming both classes
     */
    static Map<ArchitectureId, TornadoPlanProvider> index(List<TornadoPlanProvider> providers) {
        Map<ArchitectureId, TornadoPlanProvider> byId = new LinkedHashMap<>();
        for (TornadoPlanProvider provider : providers) {
            TornadoPlanProvider previous = byId.put(provider.architecture(), provider);
            if (previous != null) {
                List<String> both =
                        new ArrayList<>(
                                List.of(
                                        previous.getClass().getName(),
                                        provider.getClass().getName()));
                both.sort(Comparator.naturalOrder());
                throw new IllegalStateException(
                        "Two plan providers claim '"
                                + provider.architecture()
                                + "' on the tornado backend: "
                                + String.join(", ", both)
                                + ". Exactly one must.");
            }
        }
        return byId;
    }

    private static final class Index {
        private static final Map<ArchitectureId, TornadoPlanProvider> BY_ID = index(discover());
    }

    /** The registered identities, for a message that must say what is available. */
    public static String registeredNames() {
        return Index.BY_ID.isEmpty()
                ? "nothing"
                : Index.BY_ID.keySet().stream()
                        .map(ArchitectureId::name)
                        .sorted()
                        .collect(java.util.stream.Collectors.joining(", "));
    }

    /** Which architectures have migrated to a registered plan provider. */
    public static java.util.Set<ArchitectureId> registered() {
        return new java.util.LinkedHashSet<>(Index.BY_ID.keySet());
    }

    /**
     * The plan for a migrated architecture.
     *
     * <p>It does <b>not</b> mean "unsupported". A registered provider that does not support the
     * dtype or the mode throws the named error here rather than returning empty — falling through
     * to the family switch would have made a deliberate refusal indistinguishable from a family
     * nobody had migrated, and the switch would then have answered for a family that had already
     * moved. That is the Qwen2-MoE shape: registered, {@code Q8_0} only, and {@code F16} is a
     * refusal rather than a gap.
     *
     * <p>The unsupported messages are the factory's own, word for word, so a caller cannot tell
     * whether a family has migrated by reading its error.
     */
    static Optional<ForwardPlan> create(
            DataType quantization, ExecutionMode mode, State state, Model model) {
        TornadoPlanProvider provider = Index.BY_ID.get(model.architectureId());
        if (provider == null) {
            return Optional.empty();
        }
        if (!provider.supportedDataTypes().contains(quantization)) {
            throw new UnsupportedOperationException(
                    quantization + " not supported for model: " + model.getModelType());
        }
        if (!provider.supportedModes().contains(mode)) {
            throw new UnsupportedOperationException(
                    mode + " not yet supported for " + model.getModelType() + " + " + quantization);
        }

        SingleTokenForwardPlanComponents components =
                provider.components(quantization, state, model);
        return Optional.of(
                switch (mode) {
                    case STANDARD -> new SingleTokenForwardPlan(model, components);
                    case PREFILL_DECODE ->
                            new PrefillDecodeForwardPlan(
                                    model, (PrefillDecodeForwardPlanComponents) components);
                    case BATCH_PREFILL_DECODE ->
                            new BatchPrefillDecodeForwardPlan(
                                    model,
                                    (BatchPrefillDecodeForwardPlanComponents) components,
                                    state.executionPolicy().prefillBatchSize());
                });
    }
}
