package com.example.model.llama;

import com.example.auxiliary.Timer;
import com.example.core.model.GGUF;
import com.example.core.model.tensor.GGMLTensorEntry;
import com.example.model.Model;
import com.example.loader.weights.State;
import com.example.loader.weights.Weights;
import com.example.model.ModelType;
import com.example.tokenizer.impl.LlamaTokenizer;
import com.example.tokenizer.impl.Tokenizer;
import com.example.tokenizer.vocabulary.Vocabulary;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.Map;

import static com.example.loader.weights.ModelLoader.loadWeights;

public record Llama(LlamaConfiguration configuration, Tokenizer tokenizer, Weights weights) implements Model {
    private static final int BATCH_SIZE = Integer.getInteger("llama.BatchSize", 16);

    /* For explicit use */
    private LlamaTokenizer getAsLlamaTokenizer() {
        return (LlamaTokenizer) tokenizer;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.LLAMA_3;
    }

    @Override
    public State createNewState() {
        State state = new State(configuration(), -1);
        state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        return state;
    }

    @Override
    public State createNewState(int batchsize) {
        State state = new State(configuration(), batchsize);
        state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        return state;
    }

    // @formatter:off
    public static Llama loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights) {
        try (var ignored = Timer.log("Load LlaMa model")) {
            Map<String, Object> metadata = gguf.getMetadata();

            Vocabulary vocabulary = Vocabulary.loadLlamaVocabulary(metadata);
            Tokenizer tokenizer = new LlamaTokenizer(metadata, vocabulary);

            LlamaConfiguration config = new LlamaConfiguration(
                    (int) metadata.get("llama.embedding_length"),
                    (int) metadata.get("llama.feed_forward_length"),
                    (int) metadata.get("llama.block_count"),
                    (int) metadata.get("llama.attention.head_count"),

                    metadata.containsKey("llama.attention.head_count_kv") ?
                            (int) metadata.get("llama.attention.head_count_kv") :
                            (int) metadata.get("llama.attention.head_count"),

                    vocabulary.size(),
                    (int) metadata.get("llama.context_length"),
                    (float) metadata.getOrDefault("llama.attention.layer_norm_rms_epsilon", 1e-5f),
                    (float) metadata.getOrDefault("llama.rope.freq_base", 10000f)
            ).withContextLength(contextLength);

            Weights weights = null;
            if (loadWeights) {
                Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());
                weights = loadWeights(tensorEntries, config);
            }
            return new Llama(config, tokenizer, weights);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    // @formatter:on

}

