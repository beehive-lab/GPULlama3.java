package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.Options;
import org.beehive.gpullama3.auxiliary.Timer;
import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.core.types.Pair;
import org.beehive.gpullama3.inference.operation.RoPE;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.standard.Qwen3StandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.Qwen3TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.format.ChatFormat.ChatTokens;
import org.beehive.gpullama3.model.qwen3.Qwen3;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
import org.beehive.gpullama3.tokenizer.impl.Qwen3Tokenizer;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.Map;

import static org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary.loadQwen3Vocabulary;

public class Qwen3ModelLoader extends ModelLoader {

    public Qwen3ModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, loadWeights, useTornadovm);
    }

    // @formatter:off
    @Override
    public Qwen3 loadModel() {
        try  {
            Map<String, Object> metadata = gguf.getMetadata();

            Vocabulary vocabulary = loadQwen3Vocabulary(metadata);
            boolean isDeepSeekR1DistillQwen = "DeepSeek-R1-Distill-Qwen".equals(metadata.get("general.basename"));
            Tokenizer tokenizer = new Qwen3Tokenizer(metadata, vocabulary, isDeepSeekR1DistillQwen);

            int modelContextLength = (int) metadata.get("qwen3.context_length");
            if (contextLength < 0 || modelContextLength < contextLength) {
                contextLength = modelContextLength;
            }

            Qwen3Configuration config = new Qwen3Configuration(
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
    // @formatter:on

    // @formatter:off
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

        if (useTornadovm) {
            if (TornadoVMMasterPlan.ENABLE_TORNADOVM_INIT_TIME) {
                System.out.println("Loading model weights in TornadoVM format (loading " + outputWeight.ggmlType() + " -> " + GGMLType.F16 + ")");
            }
            return createTornadoVMWeights(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
        } else {
            return createStandardWeights(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
        }
    }
    // @formatter:on

    // @formatter:off
    @Override
    public Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        return new Qwen3TornadoWeights(
                loadTensorAsFloatArray(tokenEmbeddings),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k_norm.weight")),   // attnKNorm
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q_norm.weight")),   // attnQNorm
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

    // @formatter:off
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
    // @formatter:on
}
