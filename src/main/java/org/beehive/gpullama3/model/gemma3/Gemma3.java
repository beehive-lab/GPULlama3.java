package org.beehive.gpullama3.model.gemma3;

import org.beehive.gpullama3.inference.InferenceCore;
import org.beehive.gpullama3.inference.InferenceEngine;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.Gemma3State;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.AbstractModel;
import org.beehive.gpullama3.model.ModelType;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.tokenizer.impl.Gemma3Tokenizer;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

/**
 * Google Gemma 3 model implementation.
 *
 * <p>Key features of Gemma 3:</p>
 * <ul>
 *   <li>Sandwich normalization: 4 norm layers per block (pre/post for attention and FFN)</li>
 *   <li>Q/K normalization: Per-head normalization of query and key vectors</li>
 *   <li>Embedding scaling: Embeddings multiplied by √dim</li>
 *   <li>SentencePiece tokenization with byte-level fallback</li>
 * </ul>
 */
public class Gemma3 extends AbstractModel {

    Gemma3Configuration configuration;

    public Gemma3(Gemma3Configuration configuration, Tokenizer tokenizer, Weights weights, ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat, null);
        this.configuration = configuration;
    }

    public Gemma3Configuration configuration() {
        return configuration;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.GEMMA_3;
    }

    public Gemma3Tokenizer tokenizer() {
        return (Gemma3Tokenizer) tokenizer;
    }

    @Override
    public State createNewState() {
        State state = new Gemma3State(configuration(), -1);
        // Initialize with <bos> token
        Integer bosToken = tokenizer.getSpecialTokens().get("<bos>");
        state.latestToken = (bosToken != null) ? bosToken : 0;
        return state;
    }

    @Override
    public State createNewState(int batchsize) {
        State state = new Gemma3State(configuration(), batchsize);
        // Initialize with <bos> token
        Integer bosToken = tokenizer.getSpecialTokens().get("<bos>");
        state.latestToken = (bosToken != null) ? bosToken : 0;
        return state;
    }

    /**
     * Gemma 3 uses <bos> token at the beginning.
     */
    @Override
    public boolean shouldAddBeginOfText() {
        return true;
    }

    @Override
    public void forward(State state, int token, int position) {
        if (plan == null) {
            // CPU inference path
            InferenceCore.forwardJavaGemma3(this, state, token, position);
        } else {
            // GPU inference path (can reuse Qwen3 planner for Q/K norm support)
            InferenceCore.forwardTornadoVM(this, state, token, position, tornadoVMPlan());
        }
    }

    @Override
    public List<Integer> generateTokens(State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated) {
        // Use Qwen3 generation method since both have Q/K normalization
        return InferenceEngine.generateTokensQwen3(this, state, startPosition, promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated);
    }

    @Override
    public List<Integer> generateTokensGPU(State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated, TornadoVMMasterPlan tornadoVMPlan) {
        return InferenceEngine.generateTokensGPUQwen3(this, state, startPosition, promptTokens, stopTokens, maxTokens, sampler, echo, onTokenGenerated, tornadoVMPlan);
    }
}
