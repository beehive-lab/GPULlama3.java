package com.example.model.mistral;

import com.example.auxiliary.Timer;
import com.example.core.model.GGUF;
import com.example.core.model.tensor.GGMLTensorEntry;
import com.example.model.Model;
import com.example.loader.weights.State;
import com.example.loader.weights.Weights;
import com.example.model.ModelType;
import com.example.tokenizer.impl.MistralTokenizer;
import com.example.tokenizer.impl.Tokenizer;
import com.example.tokenizer.vocabulary.Vocabulary;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.Map;

import static com.example.loader.weights.ModelLoader.loadWeights;

public record Mistral(MistralConfiguration configuration, Tokenizer tokenizer, Weights weights) implements Model {

    /* For explicit use */
    private MistralTokenizer getAsMistralTokenizer() {
        return (MistralTokenizer) tokenizer;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.MISTRAL;
    }

    public State createNewState() {
        State state = new State(configuration(), -1);
        state.latestToken = tokenizer.getSpecialTokens().get("<s>");
        return state;
    }

    public State createNewState(int batchsize) {
        State state = new State(configuration(), batchsize);
        state.latestToken = tokenizer.getSpecialTokens().get("<s>");
        return state;
    }

    // @formatter:off
    public static Mistral loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights) {
        try (var ignored = Timer.log("Load Mistral model")) {
            Map<String, Object> metadata = gguf.getMetadata();

            Vocabulary vocabulary = Vocabulary.loadMistralVocabulary(metadata);
            Tokenizer tokenizer = new MistralTokenizer(metadata, vocabulary);

            int modelContextLength = (int) metadata.get("llama.context_length");
            if (contextLength < 0 || modelContextLength < contextLength) {
                contextLength = modelContextLength;
            }

            MistralConfiguration config = new MistralConfiguration(
                    (int) metadata.get("llama.embedding_length"),
                    (int) metadata.get("llama.feed_forward_length"),
                    (int) metadata.get("llama.block_count"),
                    (int) metadata.get("llama.attention.head_count"),

                    metadata.containsKey("llama.attention.head_count_kv")
                            ? (int) metadata.get("llama.attention.head_count_kv")
                            : (int) metadata.get("llama.attention.head_count"),

                    vocabulary.size(),
                    contextLength,
                    false,
                    (float) metadata.getOrDefault("llama.attention.layer_norm_rms_epsilon", 1e-5f),
                    (float) metadata.getOrDefault("llama.rope.freq_base", 10000f)
            );

            Weights weights = null;
            if (loadWeights) {
                Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());
                weights = loadWeights(tensorEntries, config);
            }
            return new Mistral(config, tokenizer, weights);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    // @formatter:on

}
