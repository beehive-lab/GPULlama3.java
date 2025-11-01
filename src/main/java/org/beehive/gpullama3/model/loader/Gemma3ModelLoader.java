package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.core.types.Pair;
import org.beehive.gpullama3.inference.operation.RoPE;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.standard.Gemma3StandardWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.format.Gemma3ChatFormat;
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

import static org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary.loadLlamaVocabulary;

/**
 * Model loader for Google Gemma 3 models.
 *
 * <p>Loads Gemma 3 models from GGUF format with support for:</p>
 * <ul>
 *   <li>FP16 and Q8_0 quantization</li>
 *   <li>CPU and GPU (TornadoVM) inference</li>
 *   <li>Sandwich normalization (4 norm layers per block)</li>
 *   <li>Q/K normalization</li>
 * </ul>
 */
public class Gemma3ModelLoader extends ModelLoader {

    public Gemma3ModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, loadWeights, useTornadovm);
    }

    // @formatter:off
    @Override
    public Gemma3 loadModel() {
        try {
            Map<String, Object> metadata = gguf.getMetadata();

            // Load vocabulary (Gemma uses similar vocabulary to Llama)
            Vocabulary vocabulary = loadLlamaVocabulary(metadata);
            Tokenizer tokenizer = new Gemma3Tokenizer(metadata, vocabulary);

            // Detect metadata prefix (try gemma3, gemma2, gemma, then llama)
            String prefix;
            if (metadata.containsKey("gemma3.embedding_length")) {
                prefix = "gemma3.";
            } else if (metadata.containsKey("gemma2.embedding_length")) {
                prefix = "gemma2.";
            } else if (metadata.containsKey("gemma.embedding_length")) {
                prefix = "gemma.";
            } else if (metadata.containsKey("llama.embedding_length")) {
                prefix = "llama.";
            } else {
                throw new RuntimeException("Unknown Gemma3 architecture - cannot find metadata with prefix gemma3/gemma2/gemma/llama");
            }

            // Load configuration from metadata
            int modelContextLength = (int) metadata.get(prefix + "context_length");
            if (contextLength < 0 || modelContextLength < contextLength) {
                contextLength = modelContextLength;
            }

            int dim = (int) metadata.get(prefix + "embedding_length");
            int hiddenDim = (int) metadata.get(prefix + "feed_forward_length");
            int nLayers = (int) metadata.get(prefix + "block_count");
            int nHeads = (int) metadata.get(prefix + "attention.head_count");
            int nKVHeads = metadata.containsKey(prefix + "attention.head_count_kv")
                    ? (int) metadata.get(prefix + "attention.head_count_kv")
                    : nHeads;

            // Gemma3 specific: key and value head dimensions
            int nHeadsKey = metadata.containsKey(prefix + "attention.key_length")
                    ? (int) metadata.get(prefix + "attention.key_length")
                    : (dim / nHeads);
            int nHeadsValue = metadata.containsKey(prefix + "attention.value_length")
                    ? (int) metadata.get(prefix + "attention.value_length")
                    : (dim / nHeads);

            float rmsNormEps = metadata.containsKey(prefix + "attention.layer_norm_rms_epsilon")
                    ? (float) metadata.get(prefix + "attention.layer_norm_rms_epsilon")
                    : 1e-6f;
            float ropeTheta = metadata.containsKey(prefix + "rope.freq_base")
                    ? (float) metadata.get(prefix + "rope.freq_base")
                    : 10000.0f;

            // Determine vocabulary size from token embeddings tensor
            Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());
            GGMLTensorEntry tokenEmbeddings = tensorEntries.get("token_embd.weight");
            int[] embShape = tokenEmbeddings.shape();
            int vocabSize = embShape.length > 1 ? embShape[1] : embShape[0];

            // Check if weights are shared between embeddings and output
            boolean sharedWeights = !tensorEntries.containsKey("output.weight");

            // Debug output
            System.err.println("DEBUG Gemma3 config loading:");
            System.err.println("  dim=" + dim + ", hiddenDim=" + hiddenDim + ", nLayers=" + nLayers);
            System.err.println("  nHeads=" + nHeads + ", nKVHeads=" + nKVHeads);
            System.err.println("  nHeadsKey=" + nHeadsKey + ", nHeadsValue=" + nHeadsValue);
            System.err.println("  dim / nHeads = " + (dim / nHeads));
            System.err.println("  nHeadsKey * nHeads = " + (nHeadsKey * nHeads));

            // Debug: check tensor sizes
            GGMLTensorEntry wqTensor = tensorEntries.get("blk.0.attn_q.weight");
            GGMLTensorEntry woTensor = tensorEntries.get("blk.0.attn_output.weight");
            if (wqTensor != null) {
                System.err.println("  wq shape: " + java.util.Arrays.toString(wqTensor.shape()));
            }
            if (woTensor != null) {
                int[] woShape = woTensor.shape();
                System.err.println("  wo shape: " + java.util.Arrays.toString(woShape));
                int woSize = 1;
                for (int s : woShape) woSize *= s;
                System.err.println("  wo size: " + woSize + ", wo projects from " + woShape[1] + " to " + woShape[0]);
            }

            Gemma3Configuration config = new Gemma3Configuration(
                    dim,
                    hiddenDim,
                    nLayers,
                    nHeads,
                    nKVHeads,
                    nHeadsKey,
                    nHeadsValue,
                    vocabSize,
                    modelContextLength,
                    contextLength,
                    sharedWeights,
                    rmsNormEps,
                    ropeTheta
            );

            Weights weights = null;
            if (loadWeights) {
                weights = loadWeights(tensorEntries, config);
            }

            return new Gemma3(config, tokenizer, weights, new Gemma3ChatFormat((Gemma3Tokenizer) tokenizer));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    // @formatter:on

    // @formatter:off
    @Override
    public Weights loadWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config) {
        // Compute RoPE frequencies using key head size
        Gemma3Configuration gemma3Config = (Gemma3Configuration) config;
        Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(
                config.contextLengthModel(),
                gemma3Config.numberOfHeadsKey(),
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
                System.out.println("Loading Gemma3 model weights in TornadoVM format (loading " + outputWeight.ggmlType() + ")");
            }
            // GPU path - TODO: implement Gemma3TornadoWeights
            // For now, we'll focus on CPU implementation
            throw new UnsupportedOperationException("TornadoVM GPU support for Gemma3 not yet implemented. Use CPU mode (remove --gpu flag).");
        } else {
            return createStandardWeights(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
        }
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

        return new Gemma3StandardWeights(
                loadQuantized(tokenEmbeddings),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),         // pre-attention norm
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.weight")),            // wq
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.weight")),            // wk
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.weight")),            // wv
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_output.weight")),       // wo

                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k_norm.weight")),       // attnKNorm (Q/K norm)
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q_norm.weight")),       // attnQNorm (Q/K norm)
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".post_attention_norm.weight")), // postAttentionNorm (sandwich norm)

                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),          // pre-FFN norm
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),          // w1
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),          // w2
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),            // w3
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".post_ffw_norm.weight")),     // postFFNNorm (sandwich norm)

                loadQuantized(tensorEntries.get("output_norm.weight")),                                                           // rms_final_weight
                new ArrayFloatTensor(ropeFreqsReal),
                new ArrayFloatTensor(ropeFreqsImag),
                tensorEntries.containsKey("output.weight")
                        ? ModelLoader.loadQuantized(tensorEntries.get("output.weight"))
                        : loadQuantized(tokenEmbeddings), // weights are shared
                outputWeight.ggmlType()
        );
    }
    // @formatter:on
}
