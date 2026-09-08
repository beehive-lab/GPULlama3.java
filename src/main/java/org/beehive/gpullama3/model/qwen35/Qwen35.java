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
 * <p>Runs on the host and, in single-token decode, on an accelerator. Its weights are retained in
 * the representations the file holds them in — Q4_0 projections, Q4_1 down projections on the early
 * blocks, Q5_K recurrent outputs, a Q6_K vocabulary projection, F32 norms and SSM parameters — and
 * each device task decodes the representation of the tensor it reads. Prefill, batched decode and
 * device-side drafting are not implemented, and a request for one fails by name.
 */
public class Qwen35 extends AbstractModel {

    private static final ArchitectureId ARCHITECTURE = ArchitectureId.of("qwen35");

    /**
     * Whether to drive generation through the MTP draft head.
     *
     * <p>Default off, and it stays off until a backend can verify several positions in one forward
     * pass: without that, an accepted draft saves no work and the draft head's own block is added
     * cost. Read once, here, so a session cannot change it halfway through a sequence.
     */
    private static final boolean SPECULATIVE = Boolean.getBoolean("llama.qwen35.speculative");

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
        if (SPECULATIVE && configuration.numberOfNextnLayers() > 0) {
            return TokenGenerationLoop.generateTokensQwen35(
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

    // @formatter:off
    /**
     * Single-token decode on the accelerator, through the shared generation loop.
     *
     * <p>The loop is the one Qwen3 uses. Nothing in it is family-specific: it stages the token's
     * embedding, runs the plan's graphs in order and samples, and everything that makes this model
     * what it is lives in the graphs the plan holds.
     *
     * <p>The plan is single-token only. A caller who asked for sequential or batched prefill has
     * already been refused by the provider, which declares neither.
     */
    // @formatter:on
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
        return TokenGenerationLoop.generateTokensGPUQwen3(
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
    public ArchitectureId architectureId() {
        return ARCHITECTURE;
    }
}
