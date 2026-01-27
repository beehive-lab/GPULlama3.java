package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.auxiliary.Pair;
import org.beehive.gpullama3.inference.operation.RoPE;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.standard.GraniteStandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.GraniteTornadoWeights;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.granite.Granite;
import org.beehive.gpullama3.model.granite.GraniteConfiguration;
import org.beehive.gpullama3.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.tensor.GGUF;
import org.beehive.gpullama3.tensor.standard.ArrayFloatTensor;
import org.beehive.gpullama3.tensor.tornado.FP32TornadoTensor;
import org.beehive.gpullama3.tokenizer.GraniteTokenizer;
import org.beehive.gpullama3.tokenizer.Tokenizer;
import org.beehive.gpullama3.tokenizer.Vocabulary;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.nio.channels.FileChannel;
import java.util.Map;

import static org.beehive.gpullama3.model.loader.ModelLoader.loadArrayOfTensors;
import static org.beehive.gpullama3.model.loader.ModelLoader.loadArrayOfTornadoTensors;
import static org.beehive.gpullama3.model.loader.ModelLoader.loadTensor;
import static org.beehive.gpullama3.model.loader.ModelLoader.loadTornadoTensor;

public class GraniteLoader extends AbstractModelLoader<Granite, GraniteConfiguration> {

    public GraniteLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, useTornadovm);
    }

    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        // Granite uses the same token format as Llama
        return Vocabulary.loadLlamaVocabulary(metadata);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        return new GraniteTokenizer(metadata, vocabulary);
    }

    // @formatter:off
        @Override
        protected GraniteConfiguration createConfiguration(Map<String, Object> metadata) {
            int vocabSize = metadata.containsKey("granite.vocab_size")
                    ? (int) metadata.get("granite.vocab_size")
                    : (int) metadata.get("tokenizer.ggml.tokens.length");

            // Extract Granite-specific metadata keys
            float embeddingScale = (float) metadata.getOrDefault("granite.embedding_scale", 12.0f);
            float residualScale = (float) metadata.getOrDefault("granite.residual_scale", 0.22f);
            float attentionScale = (float) metadata.getOrDefault("granite.attention.scale", 0.0078125f);
            float logitScale = (float) metadata.getOrDefault("granite.logit_scale", 16.0f);

            int kvHeads;
            Object kvHeadsObj = metadata.get("granite.attention.head_count_kv");
            if (kvHeadsObj instanceof int[] kvHeadsArray) {
                // Granite 4.0: per-layer array - take first value (assuming uniform for now)
                kvHeads = kvHeadsArray[0];
            } else if (kvHeadsObj instanceof Integer) {
                // Granite 3.3: scalar value
                kvHeads = (Integer) kvHeadsObj;
            } else {
                // Fallback to head count (no GQA)
                kvHeads = (int) metadata.get("granite.attention.head_count");
            }

            return new GraniteConfiguration(
                    getModelQuantization(metadata),
                    (int) metadata.get("granite.embedding_length"),
                    (int) metadata.get("granite.feed_forward_length"),
                    (int) metadata.get("granite.block_count"),
                    (int) metadata.get("granite.attention.head_count"),
                    kvHeads,
                    vocabSize,
                    (int) metadata.get("granite.context_length"),
                    (float) metadata.getOrDefault("granite.attention.layer_norm_rms_epsilon", 1e-5f),
                    (float) metadata.getOrDefault("granite.rope.freq_base", 10000f),
                    embeddingScale,
                    residualScale,
                    attentionScale,
                    logitScale,
                    true  // Granite ties word embeddings
            ).withContextLength(contextLength);
        }

    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(GraniteConfiguration config) {
        return RoPE.precomputeFreqsCis(config.contextLength(), config.dim() / config.numberOfHeads(), config.ropeTheta(),
                false, 1.0f, 1.0f, 1.0f, config.contextLength());
    }
    // @formatter:on

    @Override
    protected Granite createModel(GraniteConfiguration config, Tokenizer tokenizer, Weights weights) {
        return new Granite(config, tokenizer, weights, ChatFormat.create(tokenizer, null));
    }

    // @formatter:off
        @Override
        protected Weights createStandardWeights(Map<String, GGMLTensorEntry> tensorEntries,
                                                GraniteConfiguration config,
                                                Pair<float[], float[]> ropeFreqs,
                                                GGMLTensorEntry tokenEmbeddings,
                                                GGMLTensorEntry outputWeight) {
            final int nl = config.numberOfLayers();

            return new GraniteStandardWeights(
                loadTensor(tokenEmbeddings),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                loadTensor(tensorEntries.get("output_norm.weight")),
                new ArrayFloatTensor(ropeFreqs.first()),
                new ArrayFloatTensor(ropeFreqs.second()),
                loadTensor(outputWeight),
                outputWeight.ggmlType());
        }
        // @formatter:on

    // @formatter:off
        @Override
        protected Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries,
                                                 GraniteConfiguration config,
                                                 Pair<float[], float[]> ropeFreqs,
                                                 GGMLTensorEntry tokenEmbeddings,
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
            return new GraniteTornadoWeights(
                loadTornadoTensor(tokenEmbeddings),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                loadTornadoTensor(tensorEntries.get("output_norm.weight")),
                new FP32TornadoTensor(FloatArray.fromArray(ropeFreqs.first())),
                new FP32TornadoTensor(FloatArray.fromArray(ropeFreqs.second())),
                loadTornadoTensor(outputWeight),
                ggmlType
            );
        }
        // @formatter:on
}
