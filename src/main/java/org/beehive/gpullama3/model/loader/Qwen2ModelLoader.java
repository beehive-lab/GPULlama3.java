package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.LlamaApp;
import org.beehive.gpullama3.auxiliary.Timer;
import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.core.types.Pair;
import org.beehive.gpullama3.inference.operation.RoPE;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.standard.Qwen2StandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.Qwen2TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.format.ChatFormat.ChatTokens;
import org.beehive.gpullama3.model.qwen2.Qwen2;
import org.beehive.gpullama3.model.qwen2.Qwen2Configuration;
import org.beehive.gpullama3.tokenizer.impl.Qwen3Tokenizer;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.Map;

import static org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary.loadQwen3Vocabulary;

public class Qwen2ModelLoader extends ModelLoader {

    public Qwen2ModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights) {
        super(fileChannel, gguf, contextLength, loadWeights);
    }

    @Override
    public Model loadModel() {
        Map<String, Object> metadata = gguf.getMetadata();
        String basename = (String) metadata.get("general.basename");

        String modelName = "DeepSeek-R1-Distill-Qwen".equals(basename)
                ? "DeepSeek-R1-Distill-Qwen"
                : "Qwen2.5";

        try (var ignored = Timer.log("Load " + modelName + " model")) {
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

    // @formatter:off
    @Override
    public Weights loadWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config) {
        Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(
                config.contextLengthModel(),
                config.headSize(),
                config.ropeTheta(),
                false,
                8,
                1,
                3,
                8192
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

    @Override
    public Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        return new Qwen2TornadoWeights(
                loadTensorAsFloatArray(tokenEmbeddings),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                // Qwen2-specific: qkv bias
                loadArrayAsFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.bias")),
                loadArrayAsFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.bias")),
                loadArrayAsFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.bias")),

                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),            // w1
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),            // w2
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),              // w3
                floatBufferToFloatArray(tensorEntries.get("output_norm.weight")),
                FloatArray.fromArray(ropeFreqs.first()),
                FloatArray.fromArray(ropeFreqs.second()),
                loadTensorAsHalfFloatArray(outputWeight),
                outputWeight.ggmlType()
        );
    }
    // @formatter:on

}
