package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.tensor.GGUF;
import org.beehive.gpullama3.tensor.standard.ArrayFloatTensor;
import org.beehive.gpullama3.tensor.tornado.FP32TornadoTensor;
import org.beehive.gpullama3.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.auxiliary.Pair;
import org.beehive.gpullama3.inference.operation.RoPE;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.standard.LlamaStandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.devstral.Devstral;
import org.beehive.gpullama3.model.devstral.DevstralConfiguration;
import org.beehive.gpullama3.tokenizer.DevstralTokenizer;
import org.beehive.gpullama3.tokenizer.Tokenizer;
import org.beehive.gpullama3.tokenizer.Vocabulary;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.nio.channels.FileChannel;
import java.util.Map;

import static org.beehive.gpullama3.model.loader.ModelLoader.*;

public class DevstralModelLoader extends AbstractModelLoader<Devstral, DevstralConfiguration> {

    public DevstralModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, useTornadovm);
    }

    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        return Vocabulary.loadDevstralVocabulary(metadata);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        return new DevstralTokenizer(metadata, vocabulary);
    }

    // @formatter:off
    @Override
    protected DevstralConfiguration createConfiguration(Map<String, Object> metadata) {
        String prefix = "mistral3";

        int modelContextLength = (int) metadata.get(prefix + ".context_length");
        int finalContextLength = (contextLength < 0 || modelContextLength < contextLength) ? modelContextLength : contextLength;

        int vocabSize = metadata.containsKey(prefix + ".vocab_size") ? (int) metadata.get(prefix + ".vocab_size") : (int) metadata.get("tokenizer.ggml.tokens.length");

        // Devstral 2 has independent head dimension (head_dim != dim/num_heads)
        int headDim = (int) metadata.get(prefix + ".attention.key_length");

        return new DevstralConfiguration(
                getModelQuantization(metadata),
                (int) metadata.get(prefix + ".embedding_length"),
                (int) metadata.get(prefix + ".feed_forward_length"),
                (int) metadata.get(prefix + ".block_count"),
                (int) metadata.get(prefix + ".attention.head_count"),
                metadata.containsKey(prefix + ".attention.head_count_kv") ?
                        (int) metadata.get(prefix + ".attention.head_count_kv")
                        : (int) metadata.get(prefix + ".attention.head_count"),
                headDim,
                vocabSize,
                finalContextLength,
                (float) metadata.getOrDefault(prefix + ".attention.layer_norm_rms_epsilon", 1e-5f),
                (float) metadata.getOrDefault(prefix + ".rope.freq_base", 10000f)
        );
    }
    // @formatter:on

    // @formatter:off
    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(DevstralConfiguration config) {
        Map<String, Object> metadata = gguf.getMetadata();
        String prefix = "mistral3";

        String ropeScalingType = (String) metadata.getOrDefault(prefix + ".rope.scaling.type", "");
        if ("yarn".equals(ropeScalingType)) {
            float factor = (float) metadata.get(prefix + ".rope.scaling.factor");
            float betaFast = (float) metadata.get(prefix + ".rope.scaling.yarn_beta_fast");
            float betaSlow = (float) metadata.get(prefix + ".rope.scaling.yarn_beta_slow");
            float logMultiplier = (float) metadata.getOrDefault(prefix + ".rope.scaling.yarn_log_multiplier", 0.0f);
            int originalContextLength = (int) metadata.get(prefix + ".rope.scaling.original_context_length");

            return RoPE.precomputeFreqsCisYaRN(
                    config.contextLength(),
                    config.headDim(),
                    config.ropeTheta(),
                    factor,
                    betaFast,
                    betaSlow,
                    logMultiplier,
                    originalContextLength
            );
        }

        return RoPE.precomputeFreqsCis(
                config.contextLength(),
                config.headDim(),
                config.ropeTheta(),
                false,
                1.0f,
                1.0f,
                1.0f,
                config.contextLength()
        );
    }
    // @formatter:on

    @Override
    protected Devstral createModel(DevstralConfiguration config, Tokenizer tokenizer, Weights weights) {
        return new Devstral(config, tokenizer, weights, ChatFormat.create(tokenizer, null));
    }

    // @formatter:off
    @Override
    protected Weights createStandardWeights(Map<String, GGMLTensorEntry> tensorEntries, DevstralConfiguration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings, GGMLTensorEntry outputWeight) {

        final int nl = config.numberOfLayers();

        return new LlamaStandardWeights(
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
    protected Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries, DevstralConfiguration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings, GGMLTensorEntry outputWeight) {
        GGMLType ggmlType = outputWeight.ggmlType();

        if (TornadoVMMasterPlan.ENABLE_TORNADOVM_INIT_TIME) {
            System.out.println("Loading model weights in TornadoVM format (loading " + ggmlType + ")");
        }

        if (ggmlType != GGMLType.F16 && ggmlType != GGMLType.Q8_0) {
            throw new UnsupportedOperationException("Type: " + ggmlType + " currently not supported for TornadoVM weights.");
        }

        final int nl = config.numberOfLayers();

        return new LlamaTornadoWeights(
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
