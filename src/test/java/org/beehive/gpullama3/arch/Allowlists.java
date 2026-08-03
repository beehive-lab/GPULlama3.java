package org.beehive.gpullama3.arch;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Enumerated allowlists for the rules that cannot pass on today's code.
 *
 * <p>Policy ({@code dependency-rules.md} §Allowlist policy), enforced by
 * {@link DependencyRulesTest}:
 * <ol>
 *   <li>fully qualified names only — <b>never</b> a wildcard or a package;</li>
 *   <li>every entry names the milestone that removes it;</li>
 *   <li>an allowlist may shrink in any PR; it may not grow without an ADR or a recorded
 *       maintainer decision;</li>
 *   <li>a rule with an empty allowlist has its allowlist deleted, not left empty — which is
 *       why Rules 7 and 11 have no list here;</li>
 *   <li>CI fails both on a new violation and on a stale entry that no longer violates.</li>
 * </ol>
 */
public final class Allowlists {

    /**
     * Rule 1 — classes outside the Tornado backend that import {@code uk.ac.manchester.tornado}.
     * Removal order per the rule text: sampler, then inference, then loaders, then state, then
     * the tornado tensors.
     */
    public static final Set<String> RULE_1 = frozen(
            // M6 — bench harness drives the plan directly; follows the engine tier
            "org.beehive.gpullama3.bench.LlamaBench",

            // M8/M9 — InferenceCore* take and return FloatArray; ops vocabulary replaces this
            "org.beehive.gpullama3.inference.InferenceCore",
            "org.beehive.gpullama3.inference.InferenceCoreBatchPrefillDecode",
            "org.beehive.gpullama3.inference.InferenceCoreWithPrefillDecode",
            "org.beehive.gpullama3.inference.InferenceEngine",

            // M8 (Rule 8b) — Sampler is typed against FloatArray; sampling becomes an operation
            "org.beehive.gpullama3.inference.sampler.CategoricalSampler",
            "org.beehive.gpullama3.inference.sampler.Sampler",
            "org.beehive.gpullama3.inference.sampler.ToppSampler",

            // M6 — State subclasses hold device arrays; split into session state + leased storage
            "org.beehive.gpullama3.inference.state.DevstralState",
            "org.beehive.gpullama3.inference.state.GraniteState",
            "org.beehive.gpullama3.inference.state.LlamaState",
            "org.beehive.gpullama3.inference.state.Phi3State",
            "org.beehive.gpullama3.inference.state.Qwen2State",
            "org.beehive.gpullama3.inference.state.Qwen3State",
            "org.beehive.gpullama3.inference.state.State",
            "org.beehive.gpullama3.inference.state.State$StateFields",

            // M4 — loaders materialize device storage; becomes format-neutral descriptors
            "org.beehive.gpullama3.model.loader.DevstralModelLoader",
            "org.beehive.gpullama3.model.loader.GraniteLoader",
            "org.beehive.gpullama3.model.loader.LlamaModelLoader",
            "org.beehive.gpullama3.model.loader.MistralModelLoader",
            "org.beehive.gpullama3.model.loader.ModelLoader",
            "org.beehive.gpullama3.model.loader.Phi3ModelLoader",
            "org.beehive.gpullama3.model.loader.Qwen2ModelLoader",
            "org.beehive.gpullama3.model.loader.Qwen3ModelLoader",

            // M4 — GGUF is a format concern (Rule 4) and must not know device array types
            "org.beehive.gpullama3.tensor.GGUF",

            // M12 — tornado tensors become backend-owned storage behind the backend SPI
            "org.beehive.gpullama3.tensor.tornado.FP16TornadoTensor",
            "org.beehive.gpullama3.tensor.tornado.FP32TornadoTensor",
            "org.beehive.gpullama3.tensor.tornado.Q8_0TornadoTensor",
            "org.beehive.gpullama3.tensor.tornado.TornadoTensor");

    /**
     * Rule 2 — model packages depending on TornadoVM or on the Tornado backend package.
     * The concrete model types are here for {@code TornadoVMMasterPlan} in
     * {@code generateTokensGPU}; that leaves with the session split (ADR-001, M6).
     */
    public static final Set<String> RULE_2 = frozen(
            // M6 (ADR-001) — the plan moves off the model onto a session
            "org.beehive.gpullama3.model.AbstractModel",
            "org.beehive.gpullama3.model.Model",
            "org.beehive.gpullama3.model.devstral.Devstral",
            "org.beehive.gpullama3.model.granite.Granite",
            "org.beehive.gpullama3.model.llama.Llama",
            "org.beehive.gpullama3.model.mistral.Mistral",
            "org.beehive.gpullama3.model.phi3.Phi3",
            "org.beehive.gpullama3.model.qwen2.Qwen2",
            "org.beehive.gpullama3.model.qwen3.Qwen3",

            // M4 — loaders build device tensors directly
            "org.beehive.gpullama3.model.loader.AbstractModelLoader",
            "org.beehive.gpullama3.model.loader.DevstralModelLoader",
            "org.beehive.gpullama3.model.loader.GraniteLoader",
            "org.beehive.gpullama3.model.loader.LlamaModelLoader",
            "org.beehive.gpullama3.model.loader.MistralModelLoader",
            "org.beehive.gpullama3.model.loader.ModelLoader",
            "org.beehive.gpullama3.model.loader.Phi3ModelLoader",
            "org.beehive.gpullama3.model.loader.Qwen2ModelLoader",
            "org.beehive.gpullama3.model.loader.Qwen3ModelLoader");

    /**
     * Rule 5 — loaded-model types with non-final fields.
     *
     * <p>{@code AbstractModel.tokenizer/weights/chatFormat} are assigned once in the constructor
     * and become final in <b>T1.8</b>, which is ordered after the goldens land. {@code plan}
     * cannot become final until sessions exist (M6). The per-family {@code configuration} fields
     * are the same shape.
     */
    public static final Set<String> RULE_5 = frozen(
            // T1.8 for tokenizer/weights/chatFormat; M6 for plan
            "org.beehive.gpullama3.model.AbstractModel",

            // M6 — configuration is set post-construction by the loaders
            "org.beehive.gpullama3.model.devstral.Devstral",
            "org.beehive.gpullama3.model.llama.Llama",
            "org.beehive.gpullama3.model.mistral.Mistral",
            "org.beehive.gpullama3.model.phi3.Phi3",
            "org.beehive.gpullama3.model.qwen2.Qwen2",
            "org.beehive.gpullama3.model.qwen3.Qwen3");

    /**
     * Rule 8a — lower layers reaching generation policy (the CLI, the options record, the server).
     *
     * <p>Nothing under {@code tornadovm} is listed, and {@link DependencyRulesTest} asserts that
     * stays true: on-device sampling is an <b>operation</b>, not generation policy (Rule 8b), so a
     * future device sampler must not be waved through here.
     */
    public static final Set<String> RULE_8A = frozen(
            // M6 — the bench harness drives the generation loop directly
            "org.beehive.gpullama3.bench.LlamaBench",

            // M8 (Rule 8b) — Sampler.createSampler(Model, Options) depends on the CLI record
            "org.beehive.gpullama3.inference.sampler.Sampler",

            // M3/M10 — Model.runInteractive/runInstructOnce own the generation loop and take Options
            "org.beehive.gpullama3.model.Model",

            // M4 — ModelLoader reads CLI options to decide what to load
            "org.beehive.gpullama3.model.loader.ModelLoader",

            // M10 — reads LlamaApp.USE_VECTOR_API as a config flag; becomes execution policy
            "org.beehive.gpullama3.tensor.standard.Q4_0FloatTensor");

    /**
     * Rule 16 — console I/O in library code.
     *
     * <p>{@code LlamaApp} and {@code Options} are today's CLI integration and are excluded by the
     * rule rather than listed here: printing is the CLI's job. The rule document counts 20
     * printing files; the two CLI types account for the difference.
     */
    public static final Set<String> RULE_16 = frozen(
            // M2 — progress and timing output belongs behind the metrics sink
            "org.beehive.gpullama3.auxiliary.RunMetrics",
            "org.beehive.gpullama3.auxiliary.Timer$1",

            // M6 — bench harness prints its own report
            "org.beehive.gpullama3.bench.LlamaBench",

            // M2/M10 — the generation loop streams tokens straight to stdout
            "org.beehive.gpullama3.inference.InferenceEngine",
            "org.beehive.gpullama3.inference.InferenceEngineWithBatchPrefillDecode",
            "org.beehive.gpullama3.inference.InferenceEngineWithPrefillDecode",
            "org.beehive.gpullama3.model.Model",

            // M4 — loaders print progress while reading GGUF
            "org.beehive.gpullama3.model.loader.AbstractModelLoader",
            "org.beehive.gpullama3.model.loader.ModelLoader",
            "org.beehive.gpullama3.model.loader.Phi3ModelLoader",

            // M2 — plan setup prints init/JIT timings (the --verbose-init output)
            "org.beehive.gpullama3.tornadovm.TornadoVMMasterPlanBatchPrefillDecode",
            "org.beehive.gpullama3.tornadovm.TornadoVMMasterPlanPrefillDecode",
            "org.beehive.gpullama3.tornadovm.TornadoVMMasterPlanSingleToken",

            // M2 — MMA prefill layers print shape/dispatch diagnostics
            "org.beehive.gpullama3.tornadovm.layers.type.fp16.prefill.LlamaFP16LayersBatchPrefillMMA",
            "org.beehive.gpullama3.tornadovm.layers.type.fp16.prefill.Qwen3FP16LayersBatchPrefillMMA",
            "org.beehive.gpullama3.tornadovm.layers.type.q8_0.prefill.LlamaQ8_0LayersBatchPrefillMMA",
            "org.beehive.gpullama3.tornadovm.layers.type.q8_0.prefill.Qwen3Q8_0LayersBatchPrefillMMA");

    // Rule 7 and Rule 11 have no allowlist: they pass on today's code (policy item 4).

    private Allowlists() {
    }

    private static Set<String> frozen(String... names) {
        return Set.copyOf(new LinkedHashSet<>(Set.of(names)));
    }
}
