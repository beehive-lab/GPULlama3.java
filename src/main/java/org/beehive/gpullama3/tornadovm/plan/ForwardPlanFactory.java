package org.beehive.gpullama3.tornadovm.plan;

import org.beehive.gpullama3.inference.state.DevstralState;
import org.beehive.gpullama3.inference.state.GraniteState;
import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.state.Phi3State;
import org.beehive.gpullama3.inference.state.Qwen2State;
import org.beehive.gpullama3.inference.state.Qwen3State;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import org.beehive.gpullama3.tornadovm.plan.components.BatchPrefillDecodeForwardPlanComponents;
import org.beehive.gpullama3.tornadovm.plan.components.fp16.DevstralFP16PlanComponents;
import org.beehive.gpullama3.tornadovm.plan.components.fp16.GraniteFP16PlanComponents;
import org.beehive.gpullama3.tornadovm.plan.components.fp16.LlamaFP16PlanComponents;
import org.beehive.gpullama3.tornadovm.plan.components.fp16.MistralFP16PlanComponents;
import org.beehive.gpullama3.tornadovm.plan.components.fp16.Phi3FP16PlanComponents;
import org.beehive.gpullama3.tornadovm.plan.components.fp16.Qwen2FP16PlanComponents;
import org.beehive.gpullama3.tornadovm.plan.components.fp16.Qwen3FP16PlanComponents;
import org.beehive.gpullama3.tornadovm.plan.components.q8_0.DevstralQ8_0PlanComponents;
import org.beehive.gpullama3.tornadovm.plan.components.q8_0.GraniteQ8_0PlanComponents;
import org.beehive.gpullama3.tornadovm.plan.components.q8_0.LlamaQ8_0PlanComponents;
import org.beehive.gpullama3.tornadovm.plan.components.q8_0.MistralQ8_0PlanComponents;
import org.beehive.gpullama3.tornadovm.plan.components.q8_0.Phi3Q8_0PlanComponents;
import org.beehive.gpullama3.tornadovm.plan.components.q8_0.Qwen2Q8_0PlanComponents;
import org.beehive.gpullama3.tornadovm.plan.components.q8_0.Qwen3Q8_0PlanComponents;

/**
 * Factory for {@link ForwardPlan} instances.
 *
 * <p>Dispatches across three axes in order:
 * <ol>
 *   <li>Quantization ({@link GGMLType})</li>
 *   <li>Model family ({@link org.beehive.gpullama3.model.ModelType})</li>
 *   <li>Execution mode ({@link ExecutionMode})</li>
 * </ol>
 *
 * <p>Use the typed convenience methods when the execution mode is known at the call site:</p>
 * <ul>
 *   <li>{@link #createSingleToken} — returns {@link SingleTokenForwardPlan}</li>
 *   <li>{@link #createPrefillDecode} — returns {@link PrefillDecodeForwardPlan}</li>
 *   <li>{@link #createBatchPrefillDecode} — returns {@link BatchPrefillDecodeForwardPlan}</li>
 * </ul>
 */
public class ForwardPlanFactory {

    private ForwardPlanFactory() {}

    // ── Typed public API ──────────────────────────────────────────────────────

    public static SingleTokenForwardPlan createSingleToken(GGMLType quantization, State state, Model model) {
        ForwardPlan plan = create(quantization, ExecutionMode.STANDARD, state, model);
        if (plan instanceof SingleTokenForwardPlan singleToken) return singleToken;
        throw new IllegalStateException("Expected SingleTokenForwardPlan for STANDARD mode but got " + plan.getClass().getSimpleName());
    }

    public static PrefillDecodeForwardPlan createPrefillDecode(GGMLType quantization, State state, Model model) {
        ForwardPlan plan = create(quantization, ExecutionMode.PREFILL_DECODE, state, model);
        if (plan instanceof PrefillDecodeForwardPlan prefillDecode) return prefillDecode;
        throw new IllegalStateException("Expected PrefillDecodeForwardPlan for PREFILL_DECODE mode but got " + plan.getClass().getSimpleName());
    }

    public static BatchPrefillDecodeForwardPlan createBatchPrefillDecode(GGMLType quantization, State state, Model model) {
        ForwardPlan plan = create(quantization, ExecutionMode.BATCH_PREFILL_DECODE, state, model);
        if (plan instanceof BatchPrefillDecodeForwardPlan batchPrefillDecode) return batchPrefillDecode;
        throw new IllegalStateException("Expected BatchPrefillDecodeForwardPlan for BATCH_PREFILL_DECODE mode but got " + plan.getClass().getSimpleName());
    }

    // ── Generic dispatch ──────────────────────────────────────────────────────

    static ForwardPlan create(GGMLType quantization, ExecutionMode mode, State state, Model model) {
        return switch (quantization) {
            case F16  -> createFP16Plan(mode, state, model);
            case Q8_0 -> createQ8_0Plan(mode, state, model);
            case F32  -> throw new UnsupportedOperationException("F32 plans not yet implemented");
            case Q4_0 -> throw new UnsupportedOperationException("Q4_0 plans not yet implemented");
            default   -> throw new UnsupportedOperationException("Quantization not supported: " + quantization);
        };
    }

    // ── FP16 branch ───────────────────────────────────────────────────────────

    private static ForwardPlan createFP16Plan(ExecutionMode mode, State state, Model model) {
        return switch (model.getModelType()) {
            case LLAMA_3                  -> createLlamaFP16Plan(mode, (LlamaState)    state, model);
            case MISTRAL                  -> createMistralFP16Plan(mode, (LlamaState)    state, model);
            case DEVSTRAL_2               -> createDevstralFP16Plan(mode, (DevstralState) state, model);
            case QWEN_2                   -> createQwen2FP16Plan(mode, (Qwen2State)    state, model);
            case QWEN_3                   -> createQwen3FP16Plan(mode, (Qwen3State)    state, model);
            case PHI_3                    -> createPhi3FP16Plan(mode, (Phi3State)     state, model);
            case GRANITE                  -> createGraniteFP16Plan(mode, (GraniteState)  state, model);
            case DEEPSEEK_R1_DISTILL_QWEN -> createQwen2FP16Plan(mode, (Qwen2State)    state, model);
            default -> throw new UnsupportedOperationException("F16 not supported for model: " + model.getModelType());
        };
    }

    // ── Q8_0 branch ───────────────────────────────────────────────────────────

    private static ForwardPlan createQ8_0Plan(ExecutionMode mode, State state, Model model) {
        return switch (model.getModelType()) {
            case LLAMA_3                  -> createLlamaQ8_0Plan(mode, (LlamaState)    state, model);
            case MISTRAL                  -> createMistralQ8_0Plan(mode, (LlamaState)    state, model);
            case DEVSTRAL_2               -> createDevstralQ8_0Plan(mode, (DevstralState) state, model);
            case QWEN_2                   -> createQwen2Q8_0Plan(mode, (Qwen2State)    state, model);
            case QWEN_3                   -> createQwen3Q8_0Plan(mode, (Qwen3State)    state, model);
            case PHI_3                    -> createPhi3Q8_0Plan(mode, (Phi3State)     state, model);
            case GRANITE                  -> createGraniteQ8_0Plan(mode, (GraniteState)  state, model);
            case DEEPSEEK_R1_DISTILL_QWEN -> createQwen2Q8_0Plan(mode, (Qwen2State)    state, model);
            default -> throw new UnsupportedOperationException("Q8_0 not supported for model: " + model.getModelType());
        };
    }

    // ── Model+quant helpers — Llama (all 3 modes supported) ──────────────────

    private static ForwardPlan createLlamaFP16Plan(ExecutionMode mode, LlamaState state, Model model) {
        BatchPrefillDecodeForwardPlanComponents components = new LlamaFP16PlanComponents(state, model);
        return switch (mode) {
            case STANDARD             -> new SingleTokenForwardPlan(model, components);
            case PREFILL_DECODE       -> new PrefillDecodeForwardPlan(model, components);
            case BATCH_PREFILL_DECODE -> new BatchPrefillDecodeForwardPlan(model, components, TornadoVMMasterPlan.PREFILL_BATCH_SIZE);
        };
    }

    private static ForwardPlan createLlamaQ8_0Plan(ExecutionMode mode, LlamaState state, Model model) {
        BatchPrefillDecodeForwardPlanComponents components = new LlamaQ8_0PlanComponents(state, model);
        return switch (mode) {
            case STANDARD             -> new SingleTokenForwardPlan(model, components);
            case PREFILL_DECODE       -> new PrefillDecodeForwardPlan(model, components);
            case BATCH_PREFILL_DECODE -> new BatchPrefillDecodeForwardPlan(model, components, TornadoVMMasterPlan.PREFILL_BATCH_SIZE);
        };
    }

    // ── Model+quant helpers — STANDARD only ──────────────────────────────────

    private static ForwardPlan createMistralFP16Plan(ExecutionMode mode, LlamaState state, Model model) {
        if (mode != ExecutionMode.STANDARD)
            throw new UnsupportedOperationException(mode + " not yet supported for MISTRAL + F16");
        return new SingleTokenForwardPlan(model, new MistralFP16PlanComponents(state, model));
    }

    private static ForwardPlan createMistralQ8_0Plan(ExecutionMode mode, LlamaState state, Model model) {
        if (mode != ExecutionMode.STANDARD)
            throw new UnsupportedOperationException(mode + " not yet supported for MISTRAL + Q8_0");
        return new SingleTokenForwardPlan(model, new MistralQ8_0PlanComponents(state, model));
    }

    private static ForwardPlan createDevstralFP16Plan(ExecutionMode mode, DevstralState state, Model model) {
        if (mode != ExecutionMode.STANDARD)
            throw new UnsupportedOperationException(mode + " not yet supported for DEVSTRAL_2 + F16");
        return new SingleTokenForwardPlan(model, new DevstralFP16PlanComponents(state, model));
    }

    private static ForwardPlan createDevstralQ8_0Plan(ExecutionMode mode, DevstralState state, Model model) {
        if (mode != ExecutionMode.STANDARD)
            throw new UnsupportedOperationException(mode + " not yet supported for DEVSTRAL_2 + Q8_0");
        return new SingleTokenForwardPlan(model, new DevstralQ8_0PlanComponents(state, model));
    }

    private static ForwardPlan createQwen2FP16Plan(ExecutionMode mode, Qwen2State state, Model model) {
        if (mode != ExecutionMode.STANDARD)
            throw new UnsupportedOperationException(mode + " not yet supported for QWEN_2 + F16");
        return new SingleTokenForwardPlan(model, new Qwen2FP16PlanComponents(state, model));
    }

    private static ForwardPlan createQwen2Q8_0Plan(ExecutionMode mode, Qwen2State state, Model model) {
        if (mode != ExecutionMode.STANDARD)
            throw new UnsupportedOperationException(mode + " not yet supported for QWEN_2 + Q8_0");
        return new SingleTokenForwardPlan(model, new Qwen2Q8_0PlanComponents(state, model));
    }

    private static ForwardPlan createQwen3FP16Plan(ExecutionMode mode, Qwen3State state, Model model) {
        if (mode != ExecutionMode.STANDARD)
            throw new UnsupportedOperationException(mode + " not yet supported for QWEN_3 + F16");
        return new SingleTokenForwardPlan(model, new Qwen3FP16PlanComponents(state, model));
    }

    private static ForwardPlan createQwen3Q8_0Plan(ExecutionMode mode, Qwen3State state, Model model) {
        if (mode != ExecutionMode.STANDARD)
            throw new UnsupportedOperationException(mode + " not yet supported for QWEN_3 + Q8_0");
        return new SingleTokenForwardPlan(model, new Qwen3Q8_0PlanComponents(state, model));
    }

    private static ForwardPlan createPhi3FP16Plan(ExecutionMode mode, Phi3State state, Model model) {
        if (mode != ExecutionMode.STANDARD)
            throw new UnsupportedOperationException(mode + " not yet supported for PHI_3 + F16");
        return new SingleTokenForwardPlan(model, new Phi3FP16PlanComponents(state, model));
    }

    private static ForwardPlan createPhi3Q8_0Plan(ExecutionMode mode, Phi3State state, Model model) {
        if (mode != ExecutionMode.STANDARD)
            throw new UnsupportedOperationException(mode + " not yet supported for PHI_3 + Q8_0");
        return new SingleTokenForwardPlan(model, new Phi3Q8_0PlanComponents(state, model));
    }

    private static ForwardPlan createGraniteFP16Plan(ExecutionMode mode, GraniteState state, Model model) {
        if (mode != ExecutionMode.STANDARD)
            throw new UnsupportedOperationException(mode + " not yet supported for GRANITE + F16");
        return new SingleTokenForwardPlan(model, new GraniteFP16PlanComponents(state, model));
    }

    private static ForwardPlan createGraniteQ8_0Plan(ExecutionMode mode, GraniteState state, Model model) {
        if (mode != ExecutionMode.STANDARD)
            throw new UnsupportedOperationException(mode + " not yet supported for GRANITE + Q8_0");
        return new SingleTokenForwardPlan(model, new GraniteQ8_0PlanComponents(state, model));
    }
}
