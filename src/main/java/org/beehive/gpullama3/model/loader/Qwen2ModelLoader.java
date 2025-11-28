package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.tensor.GGUF;
import org.beehive.gpullama3.tensor.standard.ArrayFloatTensor;
import org.beehive.gpullama3.tensor.tornado.FP32TornadoTensor;
import org.beehive.gpullama3.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.auxiliary.Pair;
import org.beehive.gpullama3.inference.operation.RoPE;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.standard.Qwen2StandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.Qwen2TornadoWeights;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.format.ChatFormat.ChatTokens;
import org.beehive.gpullama3.model.qwen2.Qwen2;
import org.beehive.gpullama3.model.qwen2.Qwen2Configuration;
import org.beehive.gpullama3.tokenizer.Qwen3Tokenizer;
import org.beehive.gpullama3.tokenizer.Tokenizer;
import org.beehive.gpullama3.tokenizer.Vocabulary;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.nio.channels.FileChannel;
import java.util.Map;

import static org.beehive.gpullama3.model.loader.ModelLoader.*;

public class Qwen2ModelLoader extends AbstractModelLoader<Qwen2, Qwen2Configuration> {

    public Qwen2ModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, useTornadovm);
    }

    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        return Vocabulary.loadQwen3Vocabulary(metadata);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        boolean isDeepSeekR1DistillQwen = "DeepSeek-R1-Distill-Qwen".equals(metadata.get("general.basename"));
        return new Qwen3Tokenizer(metadata, vocabulary, isDeepSeekR1DistillQwen);
    }

    // @formatter:off
    @Override
    protected Qwen2Configuration createConfiguration(Map<String, Object> metadata) {
        int modelContextLength = (int) metadata.get("qwen2.context_length");
        int finalContextLength = (contextLength < 0 || modelContextLength < contextLength) ? modelContextLength : contextLength;

        int numberOfKeyValueHeads = metadata.containsKey("qwen2.attention.head_count_kv") ? (int) metadata.get("qwen2.attention.head_count_kv") : (int) metadata.get("qwen2.attention.head_count");
        int vocabSize = vocabulary.size();

        return new Qwen2Configuration(
                (int) metadata.get("qwen2.embedding_length"),       // dim
                (int) metadata.get("qwen2.feed_forward_length"),    // hiddendim
                (int) metadata.get("qwen2.block_count"),            // numberOfLayers
                (int) metadata.get("qwen2.attention.head_count"),   // numberOfHeads

                numberOfKeyValueHeads, // numberOfKeyValueHeads
                numberOfKeyValueHeads, // numberOfHeadsKey
                numberOfKeyValueHeads, // numberOfHeadsValue

                vocabSize,
                modelContextLength,
                finalContextLength,
                false,
                (float) metadata.get("qwen2.attention.layer_norm_rms_epsilon"),
                (float) metadata.get("qwen2.rope.freq_base")
        );
    }
    // @formatter:on

    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(Qwen2Configuration config) {
        return RoPE.precomputeFreqsCis(config.contextLengthModel(), config.headSize(), config.ropeTheta(), false, 8, 1, 3, 8192);
    }

    // @formatter:off
    @Override
    protected Qwen2 createModel(Qwen2Configuration config, Tokenizer tokenizer, Weights weights) {
        Map<String, Object> metadata = gguf.getMetadata();
        boolean isDeepSeekR1DistillQwen = "DeepSeek-R1-Distill-Qwen".equals(metadata.get("general.basename"));
        // Qwen2.5-Coder uses <|endoftext|> as stop-token.
        ChatTokens chatTokens = isDeepSeekR1DistillQwen ? new ChatTokens("<｜begin▁of▁sentence｜>", "", "", "<｜end▁of▁sentence｜>", "")
                : new ChatTokens("<|im_start|>", "<|im_end|>", "", "<|end_of_text|>", "<|endoftext|>");
        return new Qwen2(config, tokenizer, weights, ChatFormat.create(tokenizer, chatTokens));
    }
    // @formatter:on

    // @formatter:off
    @Override
    protected Weights createStandardWeights(Map<String, GGMLTensorEntry> tensorEntries, Qwen2Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
                                            GGMLTensorEntry outputWeight) {

        final int nl = config.numberOfLayers();

        return new Qwen2StandardWeights(
                loadTensor(tokenEmbeddings),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_q.bias")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_k.bias")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_v.bias")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                loadTensor(tensorEntries.get("output_norm.weight")),
                new ArrayFloatTensor(ropeFreqs.first()),
                new ArrayFloatTensor(ropeFreqs.second()),
                loadTensor(outputWeight),
                outputWeight.ggmlType()
        );
    }
    // @formatter:on

    // @formatter:off
    @Override
    protected Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries, Qwen2Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
                                             GGMLTensorEntry outputWeight) {
        GGMLType ggmlType = outputWeight.ggmlType();

        if (TornadoVMMasterPlan.ENABLE_TORNADOVM_INIT_TIME) {
            System.out.println("Loading model weights in TornadoVM format (loading " + ggmlType + ")");
        }

        // Validate supported types
        if (ggmlType != GGMLType.F16 && ggmlType != GGMLType.Q8_0) {
            throw new UnsupportedOperationException("Type: " + ggmlType + " currently not supported for TornadoVM weights.");
        }

        final int nl = config.numberOfLayers();

        // Load all tensors uniformly as TornadoTensor hierarchy
        return new Qwen2TornadoWeights(
                loadTornadoTensor(tokenEmbeddings),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),    // fp32
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                // Qwen2-specific: qkv bias (always F32)
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_q.bias")),         // fp32
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_k.bias")),         // fp32
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_v.bias")),         // fp32
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),     // fp32
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                loadTornadoTensor(tensorEntries.get("output_norm.weight")),                                     // fp32
                new FP32TornadoTensor(FloatArray.fromArray(ropeFreqs.first())),
                new FP32TornadoTensor(FloatArray.fromArray(ropeFreqs.second())),
                loadTornadoTensor(outputWeight),
                ggmlType
        );

    }
    // @formatter:off
}
