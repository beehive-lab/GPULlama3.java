package org.beehive.gpullama3.tornadovm.layerplanner.base;

import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.state.Phi3State;
import org.beehive.gpullama3.inference.state.Qwen2State;
import org.beehive.gpullama3.inference.state.Qwen3State;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tornadovm.GenericLayerPlanner;
import org.beehive.gpullama3.tornadovm.layerplanner.model.fp16.LlamaFP16LayerPlanner;
import org.beehive.gpullama3.tornadovm.layerplanner.model.fp16.Phi3FP16LayerPlanner;
import org.beehive.gpullama3.tornadovm.layerplanner.model.fp16.Qwen2FP16LayerPlanner;
import org.beehive.gpullama3.tornadovm.layerplanner.model.fp16.Qwen3FP16LayerPlanner;
import org.beehive.gpullama3.tornadovm.layerplanner.model.q8_0.LlamaQ8_0LayerPlanner;
import org.beehive.gpullama3.tornadovm.layerplanner.model.q8_0.Phi3Q8_0LayerPlanner;
import org.beehive.gpullama3.tornadovm.layerplanner.model.q8_0.Qwen2Q8_0LayerPlanner;
import org.beehive.gpullama3.tornadovm.layerplanner.model.q8_0.Qwen3Q8_0LayerPlanner;
import org.beehive.gpullama3.tornadovm.layerplanner.model.q4_0.LlamaQ4_0LayerPlanner;
import org.beehive.gpullama3.tornadovm.layerplanner.model.q4_0.Phi3Q4_0LayerPlanner;
import org.beehive.gpullama3.tornadovm.layerplanner.model.q4_0.Qwen2Q4_0LayerPlanner;
import org.beehive.gpullama3.tornadovm.layerplanner.model.q4_0.Qwen3Q4_0LayerPlanner;

/**
 * Factory class responsible for creating appropriate layer planners based on model type and quantization.
 * <p>
 * The factory follows a routing logic:
 * <ol>
 *   <li>Determine quantization type from {@link GGMLType}</li>
 *   <li>Determine model type from {@link Model}</li>
 *   <li>Instantiate appropriate planner implementation</li>
 * </ol>
 * <p>
 * Examples:
 * <ul>
 *   <li>{@code QuantizationType.FP16 + ModelType.LLAMA_3 → LlamaFP16LayerPlanner}</li>
 *   <li>{@code QuantizationType.Q8_0 + ModelType.QWEN_2 → Qwen2Q8_0LayerPlanner}</li>
 * </ul>
 */
public class QuantizationPlannerFactory {

    /**
     * Main factory method: create planner for given model + quantization
     */
    public static GenericLayerPlanner create(GGMLType quantization, State state, Model model) {
        return switch (quantization) {
            case F32 -> createFP32Planner(state, model);
            case F16 -> createFP16Planner(state, model);
            case Q8_0 -> createQ8_0Planner(state, model);
            case Q4_0  -> createQ4_0Planner(state, model);
            default -> throw new UnsupportedOperationException("Quantization not supported: " + quantization);
        };
    }

    // ============ FP16 Planners ============
    private static GenericLayerPlanner createFP16Planner(State state, Model model) {
        return switch (model.getModelType()) {
            case LLAMA_3, MISTRAL -> new LlamaFP16LayerPlanner((LlamaState) state, model);
            case QWEN_2 -> new Qwen2FP16LayerPlanner((Qwen2State) state, model);
            case QWEN_3 -> new Qwen3FP16LayerPlanner((Qwen3State) state, model);
            case PHI_3 -> new Phi3FP16LayerPlanner((Phi3State) state, model);
            case DEEPSEEK_R1_DISTILL_QWEN -> new Qwen2FP16LayerPlanner((Qwen2State) state, model);
            default -> throw new UnsupportedOperationException("FP16 not supported for model: " + model.getModelType());
        };
    }

    // ============ Q8_0 Planners ============
    private static GenericLayerPlanner createQ8_0Planner(State state, Model model) {
        return switch (model.getModelType()) {
            case LLAMA_3, MISTRAL -> new LlamaQ8_0LayerPlanner((LlamaState) state, model);
            case QWEN_2 -> new Qwen2Q8_0LayerPlanner((Qwen2State) state, model);
            case QWEN_3 -> new Qwen3Q8_0LayerPlanner((Qwen3State) state, model);
            case PHI_3 -> new Phi3Q8_0LayerPlanner((Phi3State) state, model);
            case DEEPSEEK_R1_DISTILL_QWEN -> new Qwen2Q8_0LayerPlanner((Qwen2State) state, model);
            default -> throw new UnsupportedOperationException("Q8_0 not supported for model: " + model.getModelType());
        };
    }

    // ============ FP32 Planners (FUTURE) ============
    private static GenericLayerPlanner createFP32Planner(State state, Model model) {
        throw new UnsupportedOperationException("FP32 planners not yet implemented");
    }

    // ============ Q4_0 Planners ============
    private static GenericLayerPlanner createQ4_0Planner(State state, Model model) {
        return switch (model.getModelType()) {
            case LLAMA_3, MISTRAL -> new LlamaQ4_0LayerPlanner((LlamaState) state, model);
            case QWEN_2 -> new Qwen2Q4_0LayerPlanner((Qwen2State) state, model);
            case QWEN_3 -> new Qwen3Q4_0LayerPlanner((Qwen3State) state, model);
            case PHI_3 -> new Phi3Q4_0LayerPlanner((Phi3State) state, model);
            case DEEPSEEK_R1_DISTILL_QWEN -> new Qwen2Q4_0LayerPlanner((Qwen2State) state, model);
            default -> throw new UnsupportedOperationException("Q4_0 not supported for model: " + model.getModelType());
        };
    }

}
