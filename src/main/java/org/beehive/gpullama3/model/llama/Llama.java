package org.beehive.gpullama3.model.llama;

import org.beehive.gpullama3.inference.InferenceCore;
import org.beehive.gpullama3.inference.InferenceEngine;
import org.beehive.gpullama3.inference.InferenceEngineWithBatchPrefillDecode;
import org.beehive.gpullama3.inference.InferenceEngineWithPrefillDecode;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.AbstractModel;
import org.beehive.gpullama3.model.ModelType;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.tokenizer.LlamaTokenizer;
import org.beehive.gpullama3.tokenizer.Tokenizer;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

public class Llama extends AbstractModel {

    static final boolean WITH_PREFILL_DECODE = Boolean.getBoolean("llama.withPrefillDecode");

    LlamaConfiguration configuration;

    public Llama(LlamaConfiguration configuration, Tokenizer tokenizer, Weights weights, ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat, null);
        this.configuration = configuration;
    }

    @Override
    public LlamaConfiguration configuration() {
        return configuration;
    }

    @Override
    public LlamaTokenizer tokenizer() {
        return (LlamaTokenizer) tokenizer;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.LLAMA_3;
    }

    @Override
    public State createNewState() {
        State state = new LlamaState(configuration(), -1);
        state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        return state;
    }

    @Override
    public State createNewState(int batchsize) {
        State state = new LlamaState(configuration(), batchsize);
        state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        return state;
    }

    @Override
    public void forward(State state, int token, int position) {
        InferenceCore.forwardJava(this, state, token, position);
    }

    @Override
    public List<Integer> generateTokens(State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated) {
        if (WITH_PREFILL_DECODE && TornadoVMMasterPlan.PREFILL_BATCH_SIZE > 1) {
            return InferenceEngineWithBatchPrefillDecode.generateTokensLlama(this, state, startPosition, promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated);
        }
        if (WITH_PREFILL_DECODE) {
            return InferenceEngineWithPrefillDecode.generateTokensLlama(this, state, startPosition, promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated);
        }
        return InferenceEngine.generateTokensLlama(this, state, startPosition, promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated);
    }

    @Override
    public List<Integer> generateTokensGPU(State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated, TornadoVMMasterPlan tornadoVMPlan) {
        if (WITH_PREFILL_DECODE && TornadoVMMasterPlan.PREFILL_BATCH_SIZE > 1) {
            return InferenceEngineWithBatchPrefillDecode.generateTokensGPULlama(this, state, startPosition, promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated, tornadoVMPlan);
        }
        if (WITH_PREFILL_DECODE) {
            return InferenceEngineWithPrefillDecode.generateTokensGPULlama(this, state, startPosition, promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated, tornadoVMPlan);
        }
        return InferenceEngine.generateTokensGPULlama(this, state, startPosition, promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated, tornadoVMPlan);
    }
}

