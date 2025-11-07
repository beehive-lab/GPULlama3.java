package org.beehive.gpullama3.model.gemma3;

import org.beehive.gpullama3.inference.InferenceCore;
import org.beehive.gpullama3.inference.InferenceEngine;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.Gemma3State;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.AbstractModel;
import org.beehive.gpullama3.model.ModelType;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.tokenizer.impl.Gemma3Tokenizer;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

/**
 * Gemma3 model implementation for inference.
 * Gemma3 uses a similar architecture to Llama but with a unique prompt template format:
 * <bos><start_of_turn>user
 * {user_message}<end_of_turn>
 * <start_of_turn>model
 * {model_message}<end_of_turn>
 */
public class Gemma3 extends AbstractModel {

    Gemma3Configuration configuration;

    public Gemma3(Gemma3Configuration configuration, Tokenizer tokenizer, Weights weights, ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat, null);
        this.configuration = configuration;
    }

    public Gemma3Configuration configuration() {
        return configuration;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.GEMMA_3;
    }

    public Gemma3Tokenizer tokenizer() {
        return (Gemma3Tokenizer) tokenizer;
    }

    @Override
    public State createNewState() {
        State state = new Gemma3State(configuration(), -1);
        // Gemma3 uses <bos> token to start
        state.latestToken = tokenizer.getSpecialTokens().get("<bos>");
        return state;
    }

    @Override
    public State createNewState(int batchsize) {
        State state = new Gemma3State(configuration(), batchsize);
        state.latestToken = tokenizer.getSpecialTokens().get("<bos>");
        return state;
    }

    /**
     * Gemma3 uses <bos> at the beginning of text.
     */
    @Override
    public boolean shouldAddBeginOfText() {
        return true;
    }

    @Override
    public void forward(State state, int token, int position) {
        if (plan == null) {
            // Gemma3 has unique "sandwich normalization" architecture
            InferenceCore.forwardJavaGemma3(this, state, token, position);
        } else {
            InferenceCore.forwardTornadoVM(this, state, token, position, tornadoVMPlan());
        }
    }

    @Override
    public List<Integer> generateTokens(State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated) {
        // Gemma uses Qwen3-style generation (with Q/K normalization)
        return InferenceEngine.generateTokensQwen3(this, state, startPosition, promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated);
    }

    @Override
    public List<Integer> generateTokensGPU(State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated, TornadoVMMasterPlan tornadoVMPlan) {
        // Gemma uses Qwen3-style GPU generation (with Q/K normalization)
        return InferenceEngine.generateTokensGPUQwen3(this, state, startPosition, promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated, tornadoVMPlan);
    }
}
