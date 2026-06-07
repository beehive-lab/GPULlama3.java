package org.beehive.gpullama3.model.gemma4;

import org.beehive.gpullama3.inference.InferenceCore;
import org.beehive.gpullama3.inference.InferenceEngine;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.Gemma4State;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.tornado.Gemma4TornadoWeights;
import org.beehive.gpullama3.model.AbstractModel;
import org.beehive.gpullama3.model.ModelType;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.loader.ModelLoader;
import org.beehive.gpullama3.tokenizer.Gemma4Tokenizer;
import org.beehive.gpullama3.tokenizer.Tokenizer;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

public class Gemma4 extends AbstractModel {

    Gemma4Configuration configuration;

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

    @Override
    public void forward(State state, int token, int position) {
        if (plan == null) {
            InferenceCore.forwardJavaGemma4(this, state, token, position);
        } else {
            gatherPerLayerTokenEmbeddingRow((Gemma4State) state, token);
            InferenceCore.forwardTornadoVM(this, state, token, position, tornadoVMPlan());
        }
    }

    /**
     * Gathers the current token's row out of {@code per_layer_token_embd} (~2.35 billion elements --
     * far too large to keep resident on the GPU, see {@link Gemma4TornadoWeights#perLayerTokenEmbd})
     * directly into {@link Gemma4State#wrapPerLayerTokenEmbedRow}, pre-scaled by {@code sqrt(embeddingLengthPerLayer)}
     * (mirroring step 2 of {@link InferenceCore#forwardJavaGemma4}), ready for transfer to the GPU as
     * part of layer 0's per-layer-embedding setup.
     */
    private void gatherPerLayerTokenEmbeddingRow(Gemma4State state, int token) {
        Gemma4TornadoWeights gemma4Weights = (Gemma4TornadoWeights) weights;
        int nEmbdPerLayer = configuration.embeddingLengthPerLayer();
        int perLayerTotal = configuration.numberOfLayers() * nEmbdPerLayer;
        float scale = (float) Math.sqrt(nEmbdPerLayer);
        ModelLoader.copyEmbeddingRowToFloatArray(gemma4Weights.perLayerTokenEmbd, token, perLayerTotal, state.wrapPerLayerTokenEmbedRow, scale);
    }

    @Override
    public List<Integer> generateTokens(State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated) {
        return InferenceEngine.generateTokensQwen3(this, state, startPosition, promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated);
    }

    @Override
    public List<Integer> generateTokensGPU(State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated, TornadoVMMasterPlan tornadoVMPlan) {
        return InferenceEngine.generateTokensGPUQwen3(this, state, startPosition, promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated, tornadoVMPlan);
    }
}
