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
import org.beehive.gpullama3.model.llama.Llama;
import org.beehive.gpullama3.model.llama.LlamaConfiguration;
import org.beehive.gpullama3.tokenizer.LlamaTokenizer;
import org.beehive.gpullama3.tokenizer.Tokenizer;
import org.beehive.gpullama3.tokenizer.Vocabulary;

import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;


import java.nio.channels.FileChannel;
import java.util.Map;

import static org.beehive.gpullama3.model.loader.ModelLoader.*;

public class LlamaModelLoader extends AbstractModelLoader<Llama, LlamaConfiguration> {

    public LlamaModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, loadWeights, useTornadovm);
    }

    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        return Vocabulary.loadLlamaVocabulary(metadata);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        return new LlamaTokenizer(metadata, vocabulary);
    }

    @Override
    protected LlamaConfiguration createConfiguration(Map<String, Object> metadata) {
        int vocabSize = metadata.containsKey("llama.vocab_size") ? (int) metadata.get("llama.vocab_size") : (int) metadata.get("tokenizer.ggml.tokens.length");

        return new LlamaConfiguration(
                (int) metadata.get("llama.embedding_length"),
                (int) metadata.get("llama.feed_forward_length"),
                (int) metadata.get("llama.block_count"),
                (int) metadata.get("llama.attention.head_count"),
                metadata.containsKey("llama.attention.head_count_kv") ?
                        (int) metadata.get("llama.attention.head_count_kv")
                        : (int) metadata.get("llama.attention.head_count"),
                vocabSize,
                (int) metadata.get("llama.context_length"),
                (float) metadata.getOrDefault("llama.attention.layer_norm_rms_epsilon", 1e-5f),
                (float) metadata.getOrDefault("llama.rope.freq_base", 10000f)).withContextLength(contextLength);
    }

    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(LlamaConfiguration config) {
        return RoPE.precomputeFreqsCis(config.contextLength(), config.dim() / config.numberOfHeads(), config.ropeTheta(), false, 1.0f, 1.0f, 1.0f, config.contextLength()
        );
    }

    @Override
    protected Llama createModel(LlamaConfiguration config, Tokenizer tokenizer, Weights weights) {
        return new Llama(config, tokenizer, weights, ChatFormat.create(tokenizer, null));
    }

    @Override
    protected Weights createStandardWeights(Map<String, GGMLTensorEntry> tensorEntries, LlamaConfiguration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {

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

    @Override
    protected Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries,
                                             LlamaConfiguration config,
                                             Pair<float[], float[]> ropeFreqs,
                                             GGMLTensorEntry tokenEmbeddings,
                                             GGMLTensorEntry outputWeight) {
        GGMLType ggmlType = outputWeight.ggmlType();

        if (TornadoVMMasterPlan.ENABLE_TORNADOVM_INIT_TIME) {
            System.out.println("Loading model weights in TornadoVM format (loading " + ggmlType + ")");
        }

        // Validate supported types
        if (ggmlType != GGMLType.F16 && ggmlType != GGMLType.Q8_0 && ggmlType != GGMLType.Q4_0) {
            throw new UnsupportedOperationException("Type: " + ggmlType + " currently not supported for TornadoVM weights.");
        }

        final int nl = config.numberOfLayers();

        // Load all tensors uniformly as TornadoTensor hierarchy
        return new LlamaTornadoWeights(
                loadTornadoTensorAsFP32(tokenEmbeddings),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfTornadoTensorsAsFP32(nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                loadTornadoTensorAsFP32(tensorEntries.get("output_norm.weight")),
                new FP32TornadoTensor(FloatArray.fromArray(ropeFreqs.first())),
                new FP32TornadoTensor(FloatArray.fromArray(ropeFreqs.second())),
                loadTornadoTensor(outputWeight),
                ggmlType
        );
    }
}
