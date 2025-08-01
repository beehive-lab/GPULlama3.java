package com.example.model.loader;

import com.example.LlamaApp;
import com.example.auxiliary.Timer;
import com.example.core.model.GGMLType;
import com.example.core.model.GGUF;
import com.example.core.model.tensor.ArrayFloatTensor;
import com.example.core.model.tensor.GGMLTensorEntry;
import com.example.core.types.Pair;
import com.example.inference.operation.RoPE;
import com.example.inference.weights.Weights;
import com.example.inference.weights.standard.Phi3StandardWeights;
import com.example.inference.weights.tornado.Phi3TornadoWeights;
import com.example.model.Configuration;
import com.example.model.Model;
import com.example.model.format.ChatFormat;
import com.example.model.phi3.Phi3;
import com.example.model.phi3.Phi3Configuration;
import com.example.tokenizer.impl.Phi3Tokenizer;
import com.example.tokenizer.impl.Tokenizer;
import com.example.tokenizer.vocabulary.Vocabulary;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.Map;

public class Phi3ModelLoader extends ModelLoader {
    public Phi3ModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights) {
        super(fileChannel, gguf, contextLength, loadWeights);
    }

    // @formatter:off
    @Override
    public Phi3 loadModel() {
        try (var ignored = Timer.log("Load Phi3 model")) {
            Map<String, Object> metadata = gguf.getMetadata();
            final String modelPrefix = "phi3.";

            Vocabulary vocabulary = Vocabulary.loadPhi3Vocabulary(metadata);
            Tokenizer tokenizer = new Phi3Tokenizer(metadata, vocabulary);
            System.out.println("Tokenizer: " + tokenizer.getClass().getSimpleName());

            int modelContextLength = (int) metadata.get(modelPrefix + "context_length");
            if (contextLength < 0 || modelContextLength < contextLength) {
                contextLength = modelContextLength;
            }

            Phi3Configuration config = new Phi3Configuration(
                    (int) metadata.get(modelPrefix + "embedding_length"),           // dim
                    (int) metadata.get(modelPrefix + "feed_forward_length"),        // hidden_dim
                    (int) metadata.get(modelPrefix + "block_count"),                // n_layers
                    (int) metadata.get(modelPrefix + "attention.head_count"),       // n_heads

                    metadata.containsKey(modelPrefix + "attention.head_count_kv")
                            ? (int) metadata.get(modelPrefix + "attention.head_count_kv")
                            : (int) metadata.get(modelPrefix + "attention.head_count"), // n_kv_heads

                    vocabulary.size(),                                              // vocab_size
                    contextLength,                                                  // context_length (user-specified, not model)
                    (float) metadata.getOrDefault(modelPrefix + "attention.layer_norm_rms_epsilon", 1e-5f), // rms_norm_eps
                    (float) metadata.getOrDefault(modelPrefix + "rope.freq_base", 10000f)           // rope_theta
            );

            Weights weights = null;
            if (loadWeights) {
                Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());
                weights = loadWeights(tensorEntries, config, modelContextLength);
            }

            // Phi3 chat tokens
            ChatFormat.ChatTokens chatTokens = new ChatFormat.ChatTokens(
                    "<|system|>", "<|end|>", "<|user|>", "<|end|>", "<|assistant|>"
            );

            return new Phi3(config, tokenizer, weights, ChatFormat.create(tokenizer, chatTokens));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    // @formatter:on

    // @formatter:off
    private Weights loadWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config, int modelContextLength) {
        // Calculate head size from dim and numberOfHeads
        int headSize = config.dim() / config.numberOfHeads();

        Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(
                modelContextLength,    // Use model context length for RoPE precomputation
                headSize,              // Calculated head size
                config.ropeTheta(),
                false,                 // Phi3 uses standard RoPE, not neox-style based on reference
                8, 1, 3, 8192         // Additional RoPE parameters from reference
        );

        GGMLTensorEntry tokenEmbeddings = tensorEntries.get("token_embd.weight");
        GGMLTensorEntry outputWeight = tensorEntries.get("output.weight"); // Phi3 always has separate output weight

        if (LlamaApp.USE_TORNADOVM) {
            System.out.println("Loading model weights in TornadoVM format (loading " + outputWeight.ggmlType() + " -> " + GGMLType.F16 + ")");
            return createTornadoVMWeights(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
        } else {
            return createStandardWeights(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
        }
    }
    // @formatter:on

    // @formatter:off
    @Override
    public Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config,
            Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        return new Phi3TornadoWeights(
                loadTensorAsFloatArray(tokenEmbeddings),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_qkv.weight")),      // Combined QKV
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_output.weight")),   // wo
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),      // wDown
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),        // wUp (not combined in reference)
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

        return new Phi3StandardWeights(
                loadQuantized(tokenEmbeddings),                                                                               // token_embedding_table
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),    // rms_att_weight (as FloatTensor[])
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_qkv.weight")),     // wqkv (combined)
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_output.weight")),  // wo
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),     // rms_ffn_weight (as FloatTensor[])
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),     // wDown
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),       // wUp (separate, not combined)
                loadQuantized(tensorEntries.get("output_norm.weight")),                                                      // rms_final_weight (as FloatTensor)
                new ArrayFloatTensor(ropeFreqsReal),                                                                         // freq_cis_real
                new ArrayFloatTensor(ropeFreqsImag),                                                                         // freq_cis_imag
                loadQuantized(outputWeight),                                                                                 // wcls
                outputWeight.ggmlType()                                                                                      // weightType
        );
    }

    // @formatter:on
}
