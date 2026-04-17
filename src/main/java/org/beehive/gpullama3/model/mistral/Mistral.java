package org.beehive.gpullama3.model.mistral;

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
import org.beehive.gpullama3.tokenizer.MistralTokenizer;
import org.beehive.gpullama3.tokenizer.Tokenizer;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

import static org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan.WITH_PREFILL_DECODE;

public class Mistral extends AbstractModel {

    MistralConfiguration configuration;

    public Mistral(MistralConfiguration configuration, Tokenizer tokenizer, Weights weights, ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat, null);
        this.configuration = configuration;
    }

    @Override
    public MistralConfiguration configuration() {
        return configuration;
    }

    @Override
    public MistralTokenizer tokenizer() {
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
    public List<Integer> generateTokens(State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated) {
        if (WITH_PREFILL_DECODE && TornadoVMMasterPlan.PREFILL_BATCH_SIZE > 1) {
            throw new UnsupportedOperationException("Batch prefill/decode on CPU not yet implemented for Mistral");
        }
        if (WITH_PREFILL_DECODE) {
            throw new UnsupportedOperationException("Prefill/decode on CPU not yet implemented for Mistral");
        }
        return InferenceEngine.generateTokensLlama(this, state, startPosition, promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated);
    }

    @Override
    public List<Integer> generateTokensGPU(State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated, TornadoVMMasterPlan tornadoVMPlan) {
        if (WITH_PREFILL_DECODE && TornadoVMMasterPlan.PREFILL_BATCH_SIZE > 1) {
            throw new UnsupportedOperationException("Batch prefill/decode on GPU not yet implemented for Mistral");
        }
        if (WITH_PREFILL_DECODE) {
            throw new UnsupportedOperationException("Prefill/decode on GPU not yet implemented for Mistral");
        }
        return InferenceEngine.generateTokensGPULlama(this, state, startPosition, promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated, tornadoVMPlan);
    }

}
