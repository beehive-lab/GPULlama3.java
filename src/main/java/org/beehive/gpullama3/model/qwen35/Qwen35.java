package org.beehive.gpullama3.model.qwen35;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import org.beehive.gpullama3.backend.tornado.TornadoVMMasterPlan;
import org.beehive.gpullama3.inference.TokenGenerationLoop;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.Qwen35State;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.AbstractModel;
import org.beehive.gpullama3.model.ModelType;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.runtime.model.ArchitectureId;
import org.beehive.gpullama3.tokenizer.Qwen35Tokenizer;
import org.beehive.gpullama3.tokenizer.Tokenizer;

/**
 * A loaded model of the {@code qwen35} architecture — the hybrid attention/delta-net stack behind
 * the Qwen3.5, 3.6 and 3.8 releases.
 *
 * <p>Host execution only. No {@code TornadoPlanProvider} claims this architecture, so a request for
 * an accelerator fails by name rather than silently running something that is not this model.
 */
public class Qwen35 extends AbstractModel {

    private static final ArchitectureId ARCHITECTURE = ArchitectureId.of("qwen35");

    private final Qwen35Configuration configuration;

    public Qwen35(
            Qwen35Configuration configuration,
            Tokenizer tokenizer,
            Weights weights,
            ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat);
        this.configuration = configuration;
    }

    @Override
    public Qwen35Configuration configuration() {
        return configuration;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.QWEN_3_5;
    }

    @Override
    public Qwen35Tokenizer tokenizer() {
        return (Qwen35Tokenizer) tokenizer;
    }

    @Override
    public State createNewState() {
        State state = new Qwen35State(configuration(), -1);
        state.latestToken = chatFormat.getBeginOfText();
        return state;
    }

    @Override
    public State createNewState(int batchsize) {
        State state = new Qwen35State(configuration(), batchsize);
        state.latestToken = chatFormat.getBeginOfText();
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
        return TokenGenerationLoop.generateTokensQwen3(
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

    /**
     * Refused rather than approximated.
     *
     * <p>Reached only if a device was resolved for this model, which no provider allows today. The
     * message names the cause because the alternative — running the host path while the caller
     * believes it asked for a GPU — is the failure mode that makes a wrong benchmark look like a
     * fast one.
     */
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
        throw new UnsupportedOperationException(
                "qwen35 has no accelerator path: its delta-net layers have no kernels, and its"
                        + " weights would be materialized as Q8_0, which this architecture's sizes"
                        + " do not fit in device memory. Run it on the CPU.");
    }

    @Override
    public ArchitectureId architectureId() {
        return ARCHITECTURE;
    }
}
