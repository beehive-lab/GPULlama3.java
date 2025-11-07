package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.core.types.Pair;
import org.beehive.gpullama3.inference.operation.RoPE;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.standard.Gemma3StandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.Gemma3TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.gemma3.Gemma3;
import org.beehive.gpullama3.model.gemma3.Gemma3Configuration;
import org.beehive.gpullama3.tokenizer.impl.Gemma3Tokenizer;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.Map;

public class Gemma3ModelLoader extends ModelLoader {

    public Gemma3ModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadoVM) {
        super(fileChannel, gguf, contextLength, loadWeights, useTornadoVM);
    }

    // @formatter:off
    @Override
    public Gemma3 loadModel() {
        try  {
            Map<String, Object> metadata = gguf.getMetadata();

            // Load vocabulary - Gemma uses similar format to Llama
            Vocabulary vocabulary = Vocabulary.loadLlamaVocabulary(metadata);
            Tokenizer tokenizer = new Gemma3Tokenizer(metadata, vocabulary);

            // Gemma models can use different architecture prefixes in GGUF:
            // - "gemma." for Gemma 1
            // - "gemma2." for Gemma 2
            // - "llama." if converted using certain tools (since Gemma is based on Llama)
            // Try to detect the correct prefix
            String prefix;
            String architecture = (String) metadata.getOrDefault("general.architecture", "unknown");

            if (metadata.containsKey("gemma3.embedding_length")) {
                prefix = "gemma3.";
            } else if (metadata.containsKey("gemma2.embedding_length")) {
                prefix = "gemma2.";
            } else if (metadata.containsKey("gemma.embedding_length")) {
                prefix = "gemma.";
            } else if (metadata.containsKey("llama.embedding_length")) {
                // Some Gemma models use llama. prefix
                prefix = "llama.";
            } else {
                throw new RuntimeException("Unknown Gemma model architecture '" + architecture + "'. Cannot find embedding_length in metadata.");
            }

            int dim = (int) metadata.get(prefix + "embedding_length");
            int hiddenDim = (int) metadata.get(prefix + "feed_forward_length");
            int nLayers = (int) metadata.get(prefix + "block_count");
            int nHeads = (int) metadata.get(prefix + "attention.head_count");
            int nKVHeads = metadata.containsKey(prefix + "attention.head_count_kv") ?
                            (int) metadata.get(prefix + "attention.head_count_kv") :
                            nHeads;

            // Gemma uses separate key/value head dimensions like Qwen3
            int nHeadsKey = metadata.containsKey(prefix + "attention.key_length") ?
                            (int) metadata.get(prefix + "attention.key_length") :
                            dim / nHeads;  // Default to headSize
            int nHeadsValue = metadata.containsKey(prefix + "attention.value_length") ?
                            (int) metadata.get(prefix + "attention.value_length") :
                            dim / nHeads;  // Default to headSize

            int ctxLength = (int) metadata.get(prefix + "context_length");
            float rmsNormEps = (float) metadata.getOrDefault(prefix + "attention.layer_norm_rms_epsilon", 1e-6f);
            float ropeTheta = (float) metadata.getOrDefault(prefix + "rope.freq_base", 10000f);
            boolean sharedWeights = false;  // Gemma typically doesn't share weights

            // Load tensor entries first to get actual vocab size from embeddings
            Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());

            // Debug: Uncomment to print all tensor names
            // System.err.println("=== Gemma Tensor Names (ALL) ===");
            // tensorEntries.keySet().stream().sorted().forEach(System.err::println);
            // System.err.println("=== End Tensor Names ===");

            GGMLTensorEntry tokenEmbeddings = tensorEntries.get("token_embd.weight");

            // Get actual vocab size from token embeddings tensor shape
            // Embedding tensors are typically [vocab_size, embedding_dim]
            int[] embShape = tokenEmbeddings.shape();
            System.err.printf("Token embedding shape: %s%n", java.util.Arrays.toString(embShape));
            // For GGUF, embeddings are stored as [embedding_dim, vocab_size], so use shape[1]
            int vocabSize = embShape.length > 1 ? embShape[1] : embShape[0];

            System.err.printf("Gemma3 Config: dim=%d, hiddenDim=%d, layers=%d, heads=%d, kvHeads=%d, vocab=%d, ctx=%d%n",
                    dim, hiddenDim, nLayers, nHeads, nKVHeads, vocabSize, ctxLength);

            // Use user-specified context length if provided, otherwise use model's default
            int actualContextLength = (contextLength < 0) ? ctxLength : contextLength;

            Gemma3Configuration config = new Gemma3Configuration(
                    dim,
                    hiddenDim,
                    nLayers,
                    nHeads,
                    nKVHeads,
                    nHeadsKey,        // Key head dimension for Q/K norm
                    nHeadsValue,      // Value head dimension for Q/K norm
                    vocabSize,
                    ctxLength,        // Model's original context length
                    actualContextLength,    // Effective context length to use
                    sharedWeights,    // Whether embeddings/output weights are shared
                    rmsNormEps,
                    ropeTheta
            );

            Weights weights = null;
            if (loadWeights) {
                weights = loadWeights(tensorEntries, config);
            }
            return new Gemma3(config, tokenizer, weights, ChatFormat.create(tokenizer, null));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    // @formatter:on

    // @formatter:off
    @Override
    public Weights loadWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config) {
        // Gemma uses RoPE like Qwen3 with key head dimension
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
                System.out.println("Loading Gemma3 weights in TornadoVM format (loading " + outputWeight.ggmlType() + ")");
            }
            return createTornadoVMWeights(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
        } else {
            return createStandardWeights(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
        }
    }

    @Override
    public Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config,
                                          Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
                                          GGMLTensorEntry outputWeight) {
        return new Gemma3TornadoWeights(
                loadTensorAsFloatArray(tokenEmbeddings),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k_norm.weight")),   // attnKNorm
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q_norm.weight")),   // attnQNorm
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".post_attention_norm.weight")),  // postAttentionNorm
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),            // w1
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),            // w2
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),              // w3
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".post_ffw_norm.weight")),  // postFFNNorm
                floatBufferToFloatArray(tensorEntries.get("output_norm.weight")),
                FloatArray.fromArray(ropeFreqs.first()),
                FloatArray.fromArray(ropeFreqs.second()),
                loadTensorAsHalfFloatArray(outputWeight),
                outputWeight.ggmlType()
        );
    }

    @Override
    public Weights createStandardWeights(Map<String, GGMLTensorEntry> tensorEntries,
                                         Configuration config,
                                         Pair<float[], float[]> ropeFreqs,
                                         GGMLTensorEntry tokenEmbeddings,
                                         GGMLTensorEntry outputWeight) {
        return new Gemma3StandardWeights(
                loadQuantized(tokenEmbeddings),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),    // rms_att_weight
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.weight")),       // wq
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.weight")),       // wk
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.weight")),       // wv
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_output.weight")),  // wo

                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k_norm.weight")),  // attnKNorm
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q_norm.weight")),  // attnQNorm
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".post_attention_norm.weight")),  // postAttentionNorm

                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),     // rms_ffn_weight
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),     // w1
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),     // w2
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),       // w3
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".post_ffw_norm.weight")),  // postFFNNorm

                loadQuantized(tensorEntries.get("output_norm.weight")),                                                      // rms_final_weight
                new ArrayFloatTensor(ropeFreqs.first()),
                new ArrayFloatTensor(ropeFreqs.second()),
                tensorEntries.containsKey("output.weight")
                        ? ModelLoader.loadQuantized(tensorEntries.get("output.weight"))
                        : loadQuantized(tokenEmbeddings),                                                                    // wcls (weights are shared)
                outputWeight.ggmlType()
        );
    }
    // @formatter:on
}
