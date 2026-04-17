package org.beehive.gpullama3.model.phi3;

import org.beehive.gpullama3.inference.InferenceCore;
import org.beehive.gpullama3.inference.InferenceEngine;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.Phi3State;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.AbstractModel;
import org.beehive.gpullama3.model.ModelType;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.tokenizer.Phi3Tokenizer;
import org.beehive.gpullama3.tokenizer.Tokenizer;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

import static org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan.WITH_PREFILL_DECODE;

public class Phi3 extends AbstractModel {

    Phi3Configuration configuration;

    public Phi3(Phi3Configuration configuration, Tokenizer tokenizer, Weights weights, ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat, null);
        this.configuration = configuration;
    }

    public Phi3Configuration configuration() {
        return configuration;
    }

    public Phi3Tokenizer tokenizer() {
        return (Phi3Tokenizer) tokenizer;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.PHI_3;
    }

    @Override
    public State createNewState() {
        State state = new Phi3State(configuration(), -1);
        state.latestToken = tokenizer.getSpecialTokens().get(chatFormat.chatTokens().tStartHeader());
        return state;
    }

    @Override
    public State createNewState(int batchsize) {
        State state = new Phi3State(configuration(), batchsize);
        state.latestToken = tokenizer.getSpecialTokens().get(chatFormat.chatTokens().tStartHeader());
        return state;
    }

    /**
     * No begin of text needed for Phi3 models.
     */
    @Override
    public boolean shouldAddBeginOfText() {
        return false;
    }

    @Override
    public void forward(State state, int token, int position) {
        if (plan == null) {
            InferenceCore.forwardJavaPhi3( this, (Phi3State) state, token, position);
        } else {
            InferenceCore.forwardTornadoVM(this, state, token, position, tornadoVMPlan());
        }
    }

    @Override
    public List<Integer> generateTokens(State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated) {
        if (WITH_PREFILL_DECODE && TornadoVMMasterPlan.PREFILL_BATCH_SIZE > 1) {
            throw new UnsupportedOperationException("Batch prefill/decode on CPU not yet implemented for Phi3");
        }
        if (WITH_PREFILL_DECODE) {
            throw new UnsupportedOperationException("Prefill/decode on CPU not yet implemented for Phi3");
        }
        return InferenceEngine.generateTokensPhi3(this, state, startPosition, promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated);
    }

    @Override
    public List<Integer> generateTokensGPU(State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated, TornadoVMMasterPlan tornadoVMPlan) {
        if (WITH_PREFILL_DECODE && TornadoVMMasterPlan.PREFILL_BATCH_SIZE > 1) {
            throw new UnsupportedOperationException("Batch prefill/decode on GPU not yet implemented for Phi3");
        }
        if (WITH_PREFILL_DECODE) {
            throw new UnsupportedOperationException("Prefill/decode on GPU not yet implemented for Phi3");
        }
        return InferenceEngine.generateTokensGPUPhi3(this, state, startPosition, promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated, tornadoVMPlan);
    }
}
