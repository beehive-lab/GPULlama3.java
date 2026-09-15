package org.beehive.jllm.model.gemma4;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;
import org.beehive.jllm.backend.tornado.TornadoVMMasterPlan;
import org.beehive.jllm.inference.TokenGenerationLoop;
import org.beehive.jllm.inference.sampler.Sampler;
import org.beehive.jllm.inference.state.Gemma4State;
import org.beehive.jllm.inference.state.State;
import org.beehive.jllm.inference.weights.Weights;
import org.beehive.jllm.inference.weights.tornado.Gemma4TornadoWeights;
import org.beehive.jllm.model.AbstractModel;
import org.beehive.jllm.model.ModelType;
import org.beehive.jllm.model.format.ChatFormat;
import org.beehive.jllm.tokenizer.Gemma4Tokenizer;
import org.beehive.jllm.tokenizer.Tokenizer;

public class Gemma4 extends AbstractModel {

    Gemma4Configuration configuration;

    public Gemma4(
            Gemma4Configuration configuration,
            Tokenizer tokenizer,
            Weights weights,
            ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat);
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
        State state = new Gemma4State(configuration(), -1);
        state.latestToken = chatFormat.getBeginOfText();
        return state;
    }

    @Override
    public State createNewState(int batchsize) {
        State state = new Gemma4State(configuration(), batchsize);
        state.latestToken = chatFormat.getBeginOfText();
        return state;
    }

    /**
     * Gathers the current token's row out of {@code per_layer_token_embd} (~2.35 billion elements
     * -- far too large to keep resident on the GPU, see {@link
     * Gemma4TornadoWeights#perLayerTokenEmbd}) directly into {@link
     * Gemma4State#wrapPerLayerTokenEmbedRow}, pre-scaled by {@code sqrt(embeddingLengthPerLayer)}
     * (mirroring step 2 of {@code InferenceCore.forwardJavaGemma4}), ready for transfer to the GPU
     * as part of layer 0's per-layer-embedding setup.
     */
    private void gatherPerLayerTokenEmbeddingRow(Gemma4State state, int token) {
        Gemma4TornadoWeights gemma4Weights = (Gemma4TornadoWeights) weights;
        int nEmbdPerLayer = configuration.embeddingLengthPerLayer();
        int perLayerTotal = configuration.numberOfLayers() * nEmbdPerLayer;
        float scale = (float) Math.sqrt(nEmbdPerLayer);
        org.beehive.jllm.backend.tornado.tensor.TornadoTensorLoader.copyEmbeddingRowToFloatArray(
                gemma4Weights.perLayerTokenEmbd,
                token,
                perLayerTotal,
                state.workspace.wrapPerLayerTokenEmbedRow,
                scale);
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

    /** Its own identity, stated rather than derived. */
    @Override
    public org.beehive.jllm.runtime.model.ArchitectureId architectureId() {
        return org.beehive.jllm.runtime.model.ArchitectureId.of("gemma4");
    }
}
