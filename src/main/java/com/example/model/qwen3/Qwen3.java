package com.example.model.qwen3;

import com.example.inference.InferenceCore;
import com.example.inference.InferenceEngine;
import com.example.inference.sampler.Sampler;
import com.example.loader.weights.Qwen3State;
import com.example.loader.weights.State;
import com.example.loader.weights.Weights;
import com.example.model.Model;
import com.example.model.ModelType;
import com.example.model.format.ChatFormat;
import com.example.tokenizer.impl.Tokenizer;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

public record Qwen3(Qwen3Configuration configuration, Tokenizer tokenizer, Weights weights, ChatFormat chatFormat) implements Model {

    @Override
    public ModelType getModelType() {
        return ModelType.QWEN_3;
    }

    @Override
    public State createNewState() {
        State state = new Qwen3State(configuration(), -1);
        state.latestToken = tokenizer.getSpecialTokens().get(chatFormat.chatTokens().tStartHeader());
        return state;
    }

    @Override
    public State createNewState(int batchsize) {
        State state = new Qwen3State(configuration(), batchsize);
        state.latestToken = tokenizer.getSpecialTokens().get(chatFormat.chatTokens().tStartHeader());
        return state;
    }

    @Override
    public void forward(State state, int token, int position) {
        InferenceCore.forwardJavaQwen3(this, state, token, position);
    }

    @Override
    public List<Integer> generateTokens(State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo, IntConsumer onTokenGenerated) {
        return InferenceEngine.generateTokensQwen3(this, state, startPosition, promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated);
    }

}
