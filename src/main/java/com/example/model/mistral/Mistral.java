package com.example.model.mistral;

import com.example.inference.InferenceCore;
import com.example.inference.InferenceEngine;
import com.example.inference.sampler.Sampler;
import com.example.loader.weights.LlamaState;
import com.example.loader.weights.State;
import com.example.loader.weights.Weights;
import com.example.model.Model;
import com.example.model.ModelType;
import com.example.model.format.ChatFormat;
import com.example.tokenizer.impl.MistralTokenizer;
import com.example.tokenizer.impl.Tokenizer;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

public record Mistral(MistralConfiguration configuration, Tokenizer tokenizer, Weights weights, ChatFormat chatFormat) implements Model {

    /* For explicit use */
    private MistralTokenizer getAsMistralTokenizer() {
        return (MistralTokenizer) tokenizer;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.MISTRAL;
    }

    public State createNewState() {
        State state = new LlamaState(configuration(), -1);
        state.latestToken = tokenizer.getSpecialTokens().get("<s>");
        return state;
    }

    public State createNewState(int batchsize) {
        State state = new LlamaState(configuration(), batchsize);
        state.latestToken = tokenizer.getSpecialTokens().get("<s>");
        return state;
    }

    @Override
    public void forward(State state, int token, int position) {
        InferenceCore.forwardJava(this, state, token, position);
    }

    @Override
    public List<Integer> generateTokens(State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo, IntConsumer onTokenGenerated) {
        return InferenceEngine.generateTokensLlama(this, state, startPosition, promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated);
    }

}
