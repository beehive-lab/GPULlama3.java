package com.example.loader.weights;

import com.example.auxiliary.Timer;
import com.example.core.model.GGUF;
import com.example.core.model.tensor.GGMLTensorEntry;
import com.example.model.format.ChatFormat;
import com.example.model.llama.Llama;
import com.example.model.llama.LlamaConfiguration;
import com.example.tokenizer.impl.LlamaTokenizer;
import com.example.tokenizer.impl.Tokenizer;
import com.example.tokenizer.vocabulary.Vocabulary;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.Map;

public class LlamaModelLoader extends ModelLoader {

    public LlamaModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights) {
        super(fileChannel, gguf, contextLength, loadWeights);
    }

    // @formatter:off
    @Override
    public Llama loadModel() {
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
            return new Llama(config, tokenizer, weights, ChatFormat.create(tokenizer, null));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    // @formatter:on
}
