package org.beehive.gpullama3.model.granite;

import org.beehive.gpullama3.inference.InferenceCore;
import org.beehive.gpullama3.inference.InferenceEngine;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.GraniteState;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.AbstractModel;
import org.beehive.gpullama3.model.ModelType;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.tokenizer.GraniteTokenizer;
import org.beehive.gpullama3.tokenizer.Tokenizer;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

public class Granite extends AbstractModel {

    private final GraniteConfiguration configuration;

    public Granite(GraniteConfiguration configuration, Tokenizer tokenizer, Weights weights, ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat, null);
        this.configuration = configuration;
    }

    @Override
    public GraniteConfiguration configuration() {
        return configuration;
    }

    @Override
    public GraniteTokenizer tokenizer() {
        return (GraniteTokenizer) tokenizer;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.GRANITE;
    }

    @Override
    public State createNewState() {
        State state = new GraniteState(configuration(), -1);
        // Granite uses token 0 (<|end_of_text|>) as BOS - it's multi-purpose
        // Token 0 is the default BOS for Granite
        state.latestToken = 0;
        return state;
    }

    @Override
    public State createNewState(int batchsize) {
        State state = new GraniteState(configuration(), batchsize);
        // Token 0 is the default BOS for Granite
        state.latestToken = 0;
        return state;
    }

    @Override
    public void forward(State state, int token, int position) {
        // Uses Granite-specific forward with scaling factors
        InferenceCore.forwardGranite(this, state, token, position);
    }

    @Override
    public List<Integer> generateTokens(State state, int startPosition, List<Integer> promptTokens,
            Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated) {
        return InferenceEngine.generateTokensGranite(this, state, startPosition, promptTokens,
                stopTokens, maxTokens, sampler, echo, onTokenGenerated);
    }

    @Override
    public List<Integer> generateTokensGPU(State state, int startPosition, List<Integer> promptTokens,
            Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated, TornadoVMMasterPlan tornadoVMPlan) {
        return InferenceEngine.generateTokensGPUGranite(this, state, startPosition, promptTokens,
                stopTokens, maxTokens, sampler, echo, onTokenGenerated, tornadoVMPlan);
    }

    // Convenience accessors for scaling factors (used in forward pass)
    public float embeddingScale() {
        return configuration.embeddingScale();
    }

    public float residualScale() {
        return configuration.residualScale();
    }

    public float attentionScale() {
        return configuration.attentionScale();
    }

    public float logitScale() {
        return configuration.logitScale();
    }
}