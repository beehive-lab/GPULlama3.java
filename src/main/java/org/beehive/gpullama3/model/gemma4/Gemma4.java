package org.beehive.gpullama3.model.gemma4;

import org.beehive.gpullama3.inference.InferenceCore;
import org.beehive.gpullama3.inference.InferenceEngine;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.Gemma4State;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.AbstractModel;
import org.beehive.gpullama3.model.ModelType;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.tokenizer.Gemma4Tokenizer;
import org.beehive.gpullama3.tokenizer.Tokenizer;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

public class Gemma4 extends AbstractModel {

    private final Gemma4Configuration configuration;

    public Gemma4(Gemma4Configuration configuration, Tokenizer tokenizer, Weights weights, ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat, null);
        this.configuration = configuration;
    }

    @Override
    public Gemma4Configuration configuration() {
        return configuration;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.GEMMA_4;
    }

    @Override
    public Gemma4Tokenizer tokenizer() {
        return (Gemma4Tokenizer) tokenizer;
    }

    @Override
    public State createNewState() {
        State state = new Gemma4State(configuration, -1);
        state.latestToken = tokenizer.getSpecialTokens().get("<bos>");
        return state;
    }

    @Override
    public State createNewState(int batchsize) {
        State state = new Gemma4State(configuration, batchsize);
        state.latestToken = tokenizer.getSpecialTokens().get("<bos>");
        return state;
    }

    @Override
    public boolean shouldAddBeginOfText() {
        return true;
    }

    @Override
    public void forward(State state, int token, int position) {
        InferenceCore.forwardJavaGemma4(this, (Gemma4State) state, token, position);
    }

    @Override
    public List<Integer> generateTokens(State state, int startPosition, List<Integer> promptTokens,
                                         Set<Integer> stopTokens, int maxTokens, Sampler sampler,
                                         boolean echo, IntConsumer onTokenGenerated) {
        return InferenceEngine.generateTokensGemma4(this, state, startPosition, promptTokens,
                stopTokens, maxTokens, sampler, echo, onTokenGenerated);
    }

    @Override
    public List<Integer> generateTokensGPU(State state, int startPosition, List<Integer> promptTokens,
                                            Set<Integer> stopTokens, int maxTokens, Sampler sampler,
                                            boolean echo, IntConsumer onTokenGenerated,
                                            TornadoVMMasterPlan tornadoVMPlan) {
        return InferenceEngine.generateTokensGPUGemma4(this, state, startPosition, promptTokens,
                stopTokens, maxTokens, sampler, echo, onTokenGenerated, tornadoVMPlan);
    }
}
