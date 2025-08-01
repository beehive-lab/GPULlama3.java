package com.example.model.loader;

import com.example.auxiliary.Timer;
import com.example.core.model.GGUF;
import com.example.core.model.tensor.ArrayFloatTensor;
import com.example.core.model.tensor.GGMLTensorEntry;
import com.example.core.types.Pair;
import com.example.inference.weights.Weights;
import com.example.inference.weights.standard.Qwen2StandardWeights;
import com.example.model.Configuration;
import com.example.model.Model;
import com.example.model.format.ChatFormat;
import com.example.model.format.ChatFormat.ChatTokens;
import com.example.model.qwen2.Qwen2;
import com.example.model.qwen2.Qwen2Configuration;
import com.example.tokenizer.impl.Qwen3Tokenizer;
import com.example.tokenizer.impl.Tokenizer;
import com.example.tokenizer.vocabulary.Vocabulary;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.Map;

import static com.example.tokenizer.vocabulary.Vocabulary.loadQwen3Vocabulary;

public class Qwen2ModelLoader extends ModelLoader {

    public Qwen2ModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights) {
        super(fileChannel, gguf, contextLength, loadWeights);
    }

    @Override
    public Model loadModel() {
        try (var ignored = Timer.log("Load Qwen2 model")) {
            Map<String, Object> metadata = gguf.getMetadata();

            // reuse method of Qwen3
            Vocabulary vocabulary = loadQwen3Vocabulary(metadata);
            boolean isDeepSeekR1DistillQwen = "DeepSeek-R1-Distill-Qwen".equals(metadata.get("general.basename"));
            Tokenizer tokenizer = new Qwen3Tokenizer(metadata, vocabulary, isDeepSeekR1DistillQwen);

            int modelContextLength = (int) metadata.get("qwen2.context_length");
            if (contextLength < 0 || modelContextLength < contextLength) {
                contextLength = modelContextLength;
            }

            int numberOfKeyValueHeads = metadata.containsKey("qwen2.attention.head_count_kv")
                    ? (int) metadata.get("qwen2.attention.head_count_kv")
                    : (int) metadata.get("qwen2.attention.head_count");
            Qwen2Configuration config = new Qwen2Configuration(
                    (int) metadata.get("qwen2.embedding_length"),       // dim
                    (int) metadata.get("qwen2.feed_forward_length"),    // hiddendim
                    (int) metadata.get("qwen2.block_count"),            // numberOfLayers
                    (int) metadata.get("qwen2.attention.head_count"),   // numberOfHeads

                    numberOfKeyValueHeads, // numberOfKeyValueHeads
                    numberOfKeyValueHeads, // numberOfHeadsKey
                    numberOfKeyValueHeads, // numberOfHeadsValue

                    vocabulary.size(),
                    modelContextLength, contextLength,
                    false,
                    (float) metadata.get("qwen2.attention.layer_norm_rms_epsilon"),
                    (float) metadata.get("qwen2.rope.freq_base")
            );

            Weights weights = null;
            if (loadWeights) {
                Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());
                weights = loadWeights(tensorEntries, config);
            }
            // Qwen2.5-Coder uses <|endoftext|> as stop-token.
            ChatTokens chatTokens = isDeepSeekR1DistillQwen ?
                    new ChatTokens( "<｜begin▁of▁sentence｜>", "", "", "<｜end▁of▁sentence｜>", "") :
                    new ChatTokens( "<|im_start|>", "<|im_end|>", "", "<|end_of_text|>", "<|endoftext|>");
            return new Qwen2(config, tokenizer, weights, ChatFormat.create(tokenizer, chatTokens));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public Weights createStandardWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        return new Qwen2StandardWeights(
                loadQuantized(tokenEmbeddings),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.weight")),

                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.bias")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.bias")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.bias")),

                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                loadQuantized(tensorEntries.get("output_norm.weight")),
                new ArrayFloatTensor(ropeFreqs.first()),
                new ArrayFloatTensor(ropeFreqs.second()),
                loadQuantized(outputWeight),
                outputWeight.ggmlType());
    }
}
