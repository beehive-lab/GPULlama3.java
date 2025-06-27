package com.example.model.loader;

import com.example.LlamaApp;
import com.example.auxiliary.Timer;
import com.example.core.model.GGMLType;
import com.example.core.model.GGUF;
import com.example.core.model.tensor.ArrayFloatTensor;
import com.example.core.model.tensor.GGMLTensorEntry;
import com.example.core.types.Pair;
import com.example.inference.operation.RoPE;
import com.example.inference.weights.standard.Qwen3StandardWeights;
import com.example.inference.weights.Weights;
import com.example.model.Configuration;
import com.example.model.format.ChatFormat;
import com.example.model.format.ChatFormat.ChatTokens;
import com.example.model.qwen3.Qwen3;
import com.example.model.qwen3.Qwen3Configuration;
import com.example.tokenizer.impl.Qwen3Tokenizer;
import com.example.tokenizer.impl.Tokenizer;
import com.example.tokenizer.vocabulary.Vocabulary;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.Map;

import static com.example.tokenizer.vocabulary.Vocabulary.loadQwen3Vocabulary;

public class Qwen3ModelLoader extends ModelLoader {

    public Qwen3ModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights) {
        super(fileChannel, gguf, contextLength, loadWeights);
    }

    @Override
    public Qwen3 loadModel() {
        try (var ignored = Timer.log("Load Qwen3 model")) {
            Map<String, Object> metadata = gguf.getMetadata();

            Vocabulary vocabulary = loadQwen3Vocabulary(metadata);
            boolean isDeepSeekR1DistillQwen = "DeepSeek-R1-Distill-Qwen".equals(metadata.get("general.basename"));
            Tokenizer tokenizer = new Qwen3Tokenizer(metadata, vocabulary, isDeepSeekR1DistillQwen);

            int modelContextLength = (int) metadata.get("qwen3.context_length");
            if (contextLength < 0 || modelContextLength < contextLength) {
                contextLength = modelContextLength;
            }

            //String modelName = ggufPath.getFileName().toString();
            Qwen3Configuration config = new Qwen3Configuration(
                    //modelName,
                    (int) metadata.get("qwen3.embedding_length"),
                    (int) metadata.get("qwen3.feed_forward_length"),
                    (int) metadata.get("qwen3.block_count"),
                    (int) metadata.get("qwen3.attention.head_count"),

                    metadata.containsKey("qwen3.attention.head_count_kv")
                            ? (int) metadata.get("qwen3.attention.head_count_kv")
                            : (int) metadata.get("qwen3.attention.head_count"),
                    (int) metadata.get("qwen3.attention.key_length"),
                    (int) metadata.get("qwen3.attention.value_length"),

                    vocabulary.size(),
                    modelContextLength, contextLength,
                    false,
                    (float) metadata.get("qwen3.attention.layer_norm_rms_epsilon"),
                    (float) metadata.get("qwen3.rope.freq_base")
            );

            Weights weights = null;
            if (loadWeights) {
                Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());
                weights = loadWeights(tensorEntries, config);
            }
            // Qwen2.5-coder uses <|endoftext|> as stop-token.
            ChatTokens chatTokens = isDeepSeekR1DistillQwen ?
                    new ChatTokens( "<｜begin▁of▁sentence｜>", "", "", "<｜end▁of▁sentence｜>", "") :
                    new ChatTokens( "<|im_start|>", "<|im_end|>", "", "<|end_of_text|>", "<|endoftext|>");
            return new Qwen3(config, tokenizer, weights, ChatFormat.create(tokenizer, chatTokens));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public Weights loadWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config) {
        Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(
                config.contextLengthModel(),
                config.numberOfHeadsKey(),
                config.ropeTheta(),
                false,
                0,
                0,
                0,
                0
        );

        GGMLTensorEntry tokenEmbeddings = tensorEntries.get("token_embd.weight");
        GGMLTensorEntry outputWeight = tensorEntries.getOrDefault("output.weight", tokenEmbeddings);

        if (LlamaApp.USE_TORNADOVM) {
            System.out.println("Loading model weights in TornadoVM format (loading " + outputWeight.ggmlType() + " -> " + GGMLType.F16 + ")");
            return createTornadoVMWeights(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
        } else {
            return createStandardWeights(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
        }
    }

    @Override
    public Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public Weights createStandardWeights(Map<String, GGMLTensorEntry> tensorEntries,
                                         Configuration config,
                                         Pair<float[], float[]> ropeFreqs,
                                         GGMLTensorEntry tokenEmbeddings,
                                         GGMLTensorEntry outputWeight) {
        float[] ropeFreqsReal = ropeFreqs.first();
        float[] ropeFreqsImag = ropeFreqs.second();
        return new Qwen3StandardWeights(
                loadQuantized(tokenEmbeddings),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),    // rms_att_weight
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.weight")),       // wq
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.weight")),       // wk
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.weight")),       // wv
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_output.weight")),  // wo

                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k_norm.weight")),  // attnKNorm
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q_norm.weight")),  // attnQNorm

                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),     //rms_ffn_weight
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),     // w1
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),     // w2
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),       // w3
                loadQuantized(tensorEntries.get("output_norm.weight")), // rms_final_weight
                new ArrayFloatTensor(ropeFreqsReal),
                new ArrayFloatTensor(ropeFreqsImag),
                tensorEntries.containsKey("output.weight")
                        ? ModelLoader.loadQuantized(tensorEntries.get("output.weight"))
                        : loadQuantized(tokenEmbeddings), // weights are shared
                null
        );
    }
}
