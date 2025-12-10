package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.tensor.GGUF;
import org.beehive.gpullama3.tensor.standard.ArrayFloatTensor;
import org.beehive.gpullama3.tensor.tornado.FP32TornadoTensor;
import org.beehive.gpullama3.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.auxiliary.Pair;
import org.beehive.gpullama3.inference.operation.RoPE;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.standard.Phi3StandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.Phi3TornadoWeights;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.phi3.Phi3;
import org.beehive.gpullama3.model.phi3.Phi3Configuration;
import org.beehive.gpullama3.tokenizer.Phi3Tokenizer;
import org.beehive.gpullama3.tokenizer.Tokenizer;
import org.beehive.gpullama3.tokenizer.Vocabulary;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.nio.channels.FileChannel;
import java.util.Map;

import static org.beehive.gpullama3.model.loader.ModelLoader.*;

public class Phi3ModelLoader extends AbstractModelLoader<Phi3, Phi3Configuration> {
    private int modelContextLength;

    public Phi3ModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, useTornadovm);
    }

    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        return Vocabulary.loadPhi3Vocabulary(metadata);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        if (TornadoVMMasterPlan.ENABLE_TORNADOVM_INIT_TIME) {
            Tokenizer tokenizer = new Phi3Tokenizer(metadata, vocabulary);
            System.out.println("Tokenizer: " + tokenizer.getClass().getSimpleName());
            return tokenizer;
        }
        return new Phi3Tokenizer(metadata, vocabulary);
    }

    // @formatter:off
    @Override
    protected Phi3Configuration createConfiguration(Map<String, Object> metadata) {
        final String modelPrefix = "phi3.";

        var config = new Phi3Configuration(
                getModelQuantization(metadata),
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
        return config;
    }
    // @formatter:off

    // @formatter:off
    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(Phi3Configuration config) {
        // Calculate head size from dim and numberOfHeads
        int headSize = config.dim() / config.numberOfHeads();

        return RoPE.precomputeFreqsCis(
                modelContextLength,     // Use model context length for RoPE precomputation
                headSize,               // Calculated head size
                config.ropeTheta(),
                false,      // Phi3 uses standard RoPE, not neox-style based on reference
                8,
                1,
                3,
                8192                    // Additional RoPE parameters from reference
        );
    }
    // @formatter:off

    @Override
    protected Phi3 createModel(Phi3Configuration config, Tokenizer tokenizer, Weights weights) {
        // Phi3 chat tokens
        ChatFormat.ChatTokens chatTokens = new ChatFormat.ChatTokens("<|system|>", "<|end|>", "<|user|>", "<|end|>", "<|assistant|>");

        return new Phi3(config, tokenizer, weights, ChatFormat.create(tokenizer, chatTokens));
    }

    // @formatter:off
    @Override
    protected Weights createStandardWeights(Map<String, GGMLTensorEntry> tensorEntries, Phi3Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings, GGMLTensorEntry outputWeight) {
        float[] ropeFreqsReal = ropeFreqs.first();
        float[] ropeFreqsImag = ropeFreqs.second();

        final int nl = config.numberOfLayers();

        return new Phi3StandardWeights(
                loadTensor(tokenEmbeddings),                                                             // token_embedding_table
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),    // rms_att_weight (as FloatTensor[])
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_qkv.weight")),     // wqkv (combined)
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),  // wo
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),     // rms_ffn_weight (as FloatTensor[])
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),     // wDown
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),       // wUp (separate, not combined)
                loadTensor(tensorEntries.get("output_norm.weight")),                                     // rms_final_weight (as FloatTensor)
                new ArrayFloatTensor(ropeFreqsReal),                                                     // freq_cis_real
                new ArrayFloatTensor(ropeFreqsImag),                                                     // freq_cis_imag
                loadTensor(outputWeight),                                                                // wcls
                outputWeight.ggmlType()                                                                  // weightType
        );
    }
    // @formatter:on

    // @formatter:off
    @Override
    protected Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries, Phi3Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings, GGMLTensorEntry outputWeight) {
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
        return new Phi3TornadoWeights(
                loadTornadoTensor(tokenEmbeddings),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),    // fp32
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_qkv.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),     // fp32
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                loadTornadoTensor(tensorEntries.get("output_norm.weight")),                                     // fp32
                new FP32TornadoTensor(FloatArray.fromArray(ropeFreqs.first())),
                new FP32TornadoTensor(FloatArray.fromArray(ropeFreqs.second())),
                loadTornadoTensor(outputWeight),
                ggmlType
        );
    }
    // @formatter:on
}
