package org.beehive.gpullama3.model.mistral;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import org.beehive.gpullama3.backend.tornado.TornadoVMMasterPlan;
import org.beehive.gpullama3.inference.TokenGenerationLoop;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.LlamaState;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.AbstractModel;
import org.beehive.gpullama3.model.ModelType;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.runtime.policy.ExecutionPolicy.PhaseStrategy;
import org.beehive.gpullama3.tokenizer.MistralTokenizer;
import org.beehive.gpullama3.tokenizer.Tokenizer;

public class Mistral extends AbstractModel {

    MistralConfiguration configuration;

    public Mistral(
            MistralConfiguration configuration,
            Tokenizer tokenizer,
            Weights weights,
            ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat);
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

    /** Block-table KV addressing, on the same flag every other family follows. */
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
    public List<Integer> generateTokens(
            State state,
            int startPosition,
            List<Integer> promptTokens,
            Set<Integer> stopTokens,
            int maxTokens,
            Sampler sampler,
            boolean echo,
            IntConsumer onTokenGenerated) {
        if (state.executionPolicy().phaseStrategy() == PhaseStrategy.PREFILL_DECODE
                && state.executionPolicy().prefillBatchSize() > 1) {
            throw new UnsupportedOperationException(
                    "Batch prefill/decode on CPU not yet implemented for Mistral");
        }
        if (state.executionPolicy().phaseStrategy() == PhaseStrategy.PREFILL_DECODE) {
            throw new UnsupportedOperationException(
                    "Prefill/decode on CPU not yet implemented for Mistral");
        }
        return TokenGenerationLoop.generateTokensLlama(
                this,
                state,
                startPosition,
                promptTokens,
                stopTokens,
                maxTokens,
                sampler,
                echo,
                onTokenGenerated);
    }

    @Override
    public List<Integer> generateTokensGPU(
            State state,
            int startPosition,
            List<Integer> promptTokens,
            Set<Integer> stopTokens,
            int maxTokens,
            Sampler sampler,
            boolean echo,
            IntConsumer onTokenGenerated,
            TornadoVMMasterPlan tornadoVMPlan) {
        if (state.executionPolicy().phaseStrategy() == PhaseStrategy.PREFILL_DECODE
                && state.executionPolicy().prefillBatchSize() > 1) {
            throw new UnsupportedOperationException(
                    "Batch prefill/decode on GPU not yet implemented for Mistral");
        }
        if (state.executionPolicy().phaseStrategy() == PhaseStrategy.PREFILL_DECODE) {
            throw new UnsupportedOperationException(
                    "Prefill/decode on GPU not yet implemented for Mistral");
        }
        return TokenGenerationLoop.generateTokensGPULlama(
                this,
                state,
                startPosition,
                promptTokens,
                stopTokens,
                maxTokens,
                sampler,
                echo,
                onTokenGenerated,
                tornadoVMPlan);
    }

    @Override
    public boolean supportsSharedKvStorage() {
        return true;
    }

    /** A state whose KV lives in the lease's shared storage. */
    @Override
    public State createNewState(org.beehive.gpullama3.runtime.kv.KvLease lease) {
        if (lease == null || lease.storage() == null) {
            return createNewState();
        }
        State state = new LlamaState(configuration(), -1, lease);
        state.latestToken = tokenizer.getSpecialTokens().get("<s>");
        return state;
    }

    /** Its own identity, stated rather than derived. */
    @Override
    public org.beehive.gpullama3.runtime.model.ArchitectureId architectureId() {
        return org.beehive.gpullama3.runtime.model.ArchitectureId.of("mistral");
    }
}
