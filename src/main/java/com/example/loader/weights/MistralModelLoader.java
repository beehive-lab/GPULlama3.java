package com.example.loader.weights;

import com.example.auxiliary.Timer;
import com.example.core.model.GGUF;
import com.example.core.model.tensor.GGMLTensorEntry;
import com.example.model.format.ChatFormat;
import com.example.model.mistral.Mistral;
import com.example.model.mistral.MistralConfiguration;
import com.example.tokenizer.impl.MistralTokenizer;
import com.example.tokenizer.impl.Tokenizer;
import com.example.tokenizer.vocabulary.Vocabulary;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.Map;

public class MistralModelLoader extends ModelLoader {

    public MistralModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights) {
        super(fileChannel, gguf, contextLength, loadWeights);
    }

    // @formatter:off
    @Override
    public Mistral loadModel() {
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
            return new Mistral(config, tokenizer, weights, ChatFormat.create(tokenizer, null));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    // @formatter:on
}
