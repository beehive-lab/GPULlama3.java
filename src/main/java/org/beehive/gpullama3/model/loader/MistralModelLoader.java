package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.auxiliary.Timer;
import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.mistral.Mistral;
import org.beehive.gpullama3.model.mistral.MistralConfiguration;
import org.beehive.gpullama3.tokenizer.impl.MistralTokenizer;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.Map;

public class MistralModelLoader extends ModelLoader {

    public MistralModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, loadWeights, useTornadovm);
    }

    // @formatter:off
    @Override
    public Mistral loadModel() {
        try {
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
