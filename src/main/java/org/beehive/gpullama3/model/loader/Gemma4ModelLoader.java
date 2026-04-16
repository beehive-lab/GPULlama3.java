package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.auxiliary.Pair;
import org.beehive.gpullama3.inference.operation.RoPE;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.standard.Gemma4StandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.Gemma4TornadoWeights;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.format.Gemma4ChatFormat;
import org.beehive.gpullama3.model.gemma4.Gemma4;
import org.beehive.gpullama3.model.gemma4.Gemma4Configuration;
import org.beehive.gpullama3.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.tensor.GGUF;
import org.beehive.gpullama3.tensor.standard.ArrayFloatTensor;
import org.beehive.gpullama3.tensor.tornado.FP32TornadoTensor;
import org.beehive.gpullama3.tokenizer.Gemma4Tokenizer;
import org.beehive.gpullama3.tokenizer.Tokenizer;
import org.beehive.gpullama3.tokenizer.Vocabulary;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.Map;

import static org.beehive.gpullama3.model.loader.ModelLoader.loadArrayOfTensors;
import static org.beehive.gpullama3.model.loader.ModelLoader.loadArrayOfTornadoTensors;
import static org.beehive.gpullama3.model.loader.ModelLoader.loadTensor;
import static org.beehive.gpullama3.model.loader.ModelLoader.loadTornadoTensor;

public class Gemma4ModelLoader extends AbstractModelLoader<Gemma4, Gemma4Configuration> {

    public Gemma4ModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, useTornadovm);
    }

    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        return loadGemma4Vocabulary(metadata);
    }

    public static Vocabulary loadGemma4Vocabulary(Map<String, Object> metadata) {
        String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
        float[] scores = (float[]) metadata.get("tokenizer.ggml.scores");
        return new Vocabulary(tokens, scores);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        int[] tokenTypes = (int[]) metadata.get("tokenizer.ggml.token_type");
        return new Gemma4Tokenizer(vocabulary, tokenTypes);
    }

    // @formatter:off
    @Override
    protected Gemma4Configuration createConfiguration(Map<String, Object> metadata) {
        int modelContextLength = (int) metadata.get("gemma4.context_length");
        int finalContextLength = (contextLength < 0 || modelContextLength < contextLength)
                ? modelContextLength : contextLength;

        int embeddingLength = (int) metadata.get("gemma4.embedding_length");
        int numberOfHeads = (int) metadata.get("gemma4.attention.head_count");
        int numberOfLayers = (int) metadata.get("gemma4.block_count");
        int headSizeFull = (int) metadata.get("gemma4.attention.key_length");
        int headSizeSWA = (int) metadata.get("gemma4.attention.key_length_swa");
        int slidingWindow = (int) metadata.get("gemma4.attention.sliding_window");

        float logitSoftcapping = (float) metadata.getOrDefault("gemma4.final_logit_softcapping", 0f);
        float rmsNormEps = (float) metadata.getOrDefault("gemma4.attention.layer_norm_rms_epsilon", 1e-6f);
        float ropeTheta = (float) metadata.getOrDefault("gemma4.rope.freq_base", 1000000f);
        float ropeThetaSWA = (float) metadata.getOrDefault("gemma4.rope.freq_base_swa", 10000f);

        // MoE parameters
        int expertCount = (int) metadata.getOrDefault("gemma4.expert_count", 0);
        int expertUsedCount = (int) metadata.getOrDefault("gemma4.expert_used_count", 0);
        int expertFeedForwardLength = (int) metadata.getOrDefault("gemma4.expert_feed_forward_length", 0);

        // Per-layer feed forward lengths
        int[] feedForwardLength;
        Object ffnRaw = metadata.get("gemma4.feed_forward_length");
        if (ffnRaw instanceof int[] arr) {
            feedForwardLength = arr;
        } else {
            feedForwardLength = new int[numberOfLayers];
            Arrays.fill(feedForwardLength, (int) ffnRaw);
        }

        Map<String, GGUF.GGUFTensorInfo> tensorInfos = gguf.getTensorInfos();

        // Derive isSWA per layer from Q norm weight size
        boolean[] isSWA;
        Object swaRaw = metadata.get("gemma4.attention.sliding_window_pattern");
        if (swaRaw instanceof boolean[] arr) {
            isSWA = arr;
        } else {
            isSWA = new boolean[numberOfLayers];
            for (int i = 0; i < numberOfLayers; i++) {
                GGUF.GGUFTensorInfo qNorm = tensorInfos.get("blk." + i + ".attn_q_norm.weight");
                if (qNorm != null) {
                    long qNormSize = 1;
                    for (int d : qNorm.dimensions()) qNormSize *= d;
                    isSWA[i] = (qNormSize == headSizeSWA);
                } else {
                    isSWA[i] = (i % 5 != 4); // fallback pattern
                }
            }
        }

        // Derive per-layer KV head count from K weight shapes
        int[] numberOfKeyValueHeadsPerLayer = new int[numberOfLayers];
        for (int i = 0; i < numberOfLayers; i++) {
            GGUF.GGUFTensorInfo kWeight = tensorInfos.get("blk." + i + ".attn_k.weight");
            int headSize = isSWA[i] ? headSizeSWA : headSizeFull;
            if (kWeight != null) {
                long kDim = kWeight.dimensions()[1];
                numberOfKeyValueHeadsPerLayer[i] = (int) (kDim / headSize);
            } else {
                numberOfKeyValueHeadsPerLayer[i] = numberOfHeads;
            }
        }

        // Shared KV layers
        int sharedKvLayers = (int) metadata.getOrDefault("gemma4.attention.shared_kv_layers", 0);
        int nLayerKvFromStart = numberOfLayers - sharedKvLayers;

        int embeddingLengthPerLayer = (int) metadata.getOrDefault("gemma4.embedding_length_per_layer_input", 0);

        return new Gemma4Configuration(
                getModelQuantization(metadata),
                embeddingLength,
                numberOfLayers,
                numberOfHeads,
                headSizeFull,
                headSizeSWA,
                slidingWindow,
                isSWA,
                numberOfKeyValueHeadsPerLayer,
                feedForwardLength,
                vocabulary.size(),
                finalContextLength,
                modelContextLength,
                rmsNormEps,
                ropeTheta,
                ropeThetaSWA,
                logitSoftcapping,
                nLayerKvFromStart,
                embeddingLengthPerLayer,
                expertCount,
                expertUsedCount,
                expertFeedForwardLength
        );
    }
    // @formatter:on

    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(Gemma4Configuration config) {
        // SWA frequencies are simple
        // Full frequencies need model's rope_freqs.weight - handled in loadWeights
        return RoPE.precomputeFreqsCis(config.contextLengthModel(), config.headSizeSWA(),
                config.ropeThetaSWA(), false, 0, 0, 0, 0);
    }

    @Override
    public Weights loadWeights(Map<String, GGMLTensorEntry> tensorEntries, Gemma4Configuration config) {
        // Precompute SWA RoPE frequencies
        Pair<float[], float[]> ropeFreqsSWA = RoPE.precomputeFreqsCis(
                config.contextLengthModel(), config.headSizeSWA(), config.ropeThetaSWA(),
                false, 0, 0, 0, 0);

        // Precompute full attention RoPE frequencies using model-provided freq factors
        GGMLTensorEntry ropeFreqsEntry = tensorEntries.get("rope_freqs.weight");
        Pair<float[], float[]> ropeFreqsFull;
        if (ropeFreqsEntry != null) {
            FloatBuffer ropeFreqsBuf = ropeFreqsEntry.memorySegment()
                    .asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            float[] modelRopeFreqs = new float[ropeFreqsBuf.remaining()];
            ropeFreqsBuf.get(modelRopeFreqs);
            ropeFreqsFull = RoPE.precomputeFreqsCisFromFreqs(
                    config.contextLengthModel(), config.headSizeFull(), config.ropeTheta(), modelRopeFreqs);
        } else {
            // Fallback: standard RoPE for full attention
            ropeFreqsFull = RoPE.precomputeFreqsCis(
                    config.contextLengthModel(), config.headSizeFull(), config.ropeTheta(),
                    false, 0, 0, 0, 0);
        }

        GGMLTensorEntry tokenEmbeddings = getTokenEmbeddings(tensorEntries);
        GGMLTensorEntry outputWeight = getOutputWeight(tensorEntries, tokenEmbeddings);

        if (useTornadovm) {
            return createTornadoVMWeights(tensorEntries, config, null, tokenEmbeddings, outputWeight);
        }

        return createStandardWeightsWithDualRoPE(tensorEntries, config, ropeFreqsSWA, ropeFreqsFull,
                tokenEmbeddings, outputWeight);
    }

    private Weights createStandardWeightsWithDualRoPE(
            Map<String, GGMLTensorEntry> tensorEntries, Gemma4Configuration config,
            Pair<float[], float[]> ropeFreqsSWA, Pair<float[], float[]> ropeFreqsFull,
            GGMLTensorEntry tokenEmbeddings, GGMLTensorEntry outputWeight) {

        final int nl = config.numberOfLayers();

        // Load per-layer output scale
        float[] layerOutputScale = new float[nl];
        for (int i = 0; i < nl; i++) {
            GGMLTensorEntry scaleEntry = tensorEntries.get("blk." + i + ".layer_output_scale.weight");
            if (scaleEntry != null) {
                layerOutputScale[i] = scaleEntry.memorySegment()
                        .asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(0);
            } else {
                layerOutputScale[i] = 1.0f;
            }
        }

        // Load V weights (nullable: layers without V use K as V)
        var wv = new org.beehive.gpullama3.tensor.standard.FloatTensor[nl];
        for (int i = 0; i < nl; i++) {
            GGMLTensorEntry vEntry = tensorEntries.get("blk." + i + ".attn_v.weight");
            wv[i] = vEntry != null ? loadTensor(vEntry) : null;
        }

        // Load per-layer embedding weights (optional)
        java.lang.foreign.MemorySegment perLayerTokenEmbdSegment = null;
        org.beehive.gpullama3.tensor.GGMLType perLayerTokenEmbdType = null;
        org.beehive.gpullama3.tensor.standard.FloatTensor perLayerModelProj = null;
        FloatBuffer perLayerProjNorm = null;
        org.beehive.gpullama3.tensor.standard.FloatTensor[] perLayerInpGate = null;
        org.beehive.gpullama3.tensor.standard.FloatTensor[] perLayerProj = null;
        FloatBuffer[] perLayerPostNorm = null;

        if (config.embeddingLengthPerLayer() > 0 && tensorEntries.containsKey("per_layer_token_embd.weight")) {
            GGMLTensorEntry plEntry = tensorEntries.get("per_layer_token_embd.weight");
            perLayerTokenEmbdSegment = plEntry.memorySegment();
            perLayerTokenEmbdType = plEntry.ggmlType();
            perLayerModelProj = loadTensor(tensorEntries.get("per_layer_model_proj.weight"));
            perLayerProjNorm = toFloatBuffer(tensorEntries.get("per_layer_proj_norm.weight"));
            perLayerInpGate = loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".inp_gate.weight"));
            perLayerProj = loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".proj.weight"));
            perLayerPostNorm = loadArrayOfFloatBuffer(nl, i -> tensorEntries.get("blk." + i + ".post_norm.weight"));
        }

        // Load MoE weights (optional)
        org.beehive.gpullama3.tensor.standard.FloatTensor[] ffnGateInp = null;
        FloatBuffer[] ffnGateInpScale = null;
        org.beehive.gpullama3.tensor.standard.FloatTensor[] ffnGateUpExps = null;
        org.beehive.gpullama3.tensor.standard.FloatTensor[] ffnDownExps = null;
        FloatBuffer[] ffnDownExpsScale = null;
        FloatBuffer[] ffnPostNorm1 = null;
        FloatBuffer[] preFfwNorm2 = null;
        FloatBuffer[] ffnPostNorm2 = null;

        if (config.isMoE()) {
            ffnGateInp = loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_gate_inp.weight"));
            ffnGateInpScale = loadArrayOfFloatBuffer(nl, i -> tensorEntries.get("blk." + i + ".ffn_gate_inp.scale"));
            ffnGateUpExps = loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_gate_up_exps.weight"));
            ffnDownExps = loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_down_exps.weight"));
            ffnDownExpsScale = loadArrayOfFloatBuffer(nl, i -> tensorEntries.get("blk." + i + ".ffn_down_exps.scale"));
            ffnPostNorm1 = loadArrayOfFloatBuffer(nl, i -> tensorEntries.get("blk." + i + ".post_ffw_norm_1.weight"));
            preFfwNorm2 = loadArrayOfFloatBuffer(nl, i -> tensorEntries.get("blk." + i + ".pre_ffw_norm_2.weight"));
            ffnPostNorm2 = loadArrayOfFloatBuffer(nl, i -> tensorEntries.get("blk." + i + ".post_ffw_norm_2.weight"));
        }

        return new Gemma4StandardWeights(
                loadTensor(tokenEmbeddings),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                wv,
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfFloatBuffer(nl, i -> tensorEntries.get("blk." + i + ".attn_q_norm.weight")),
                loadArrayOfFloatBuffer(nl, i -> tensorEntries.get("blk." + i + ".attn_k_norm.weight")),
                loadArrayOfFloatBuffer(nl, i -> tensorEntries.get("blk." + i + ".post_attention_norm.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                loadArrayOfFloatBuffer(nl, i -> tensorEntries.get("blk." + i + ".post_ffw_norm.weight")),
                loadTensor(tensorEntries.get("output_norm.weight")),
                layerOutputScale,
                new ArrayFloatTensor(ropeFreqsFull.first()),
                new ArrayFloatTensor(ropeFreqsFull.second()),
                new ArrayFloatTensor(ropeFreqsSWA.first()),
                new ArrayFloatTensor(ropeFreqsSWA.second()),
                tensorEntries.containsKey("output.weight")
                        ? loadTensor(tensorEntries.get("output.weight"))
                        : loadTensor(tokenEmbeddings),
                perLayerTokenEmbdSegment, perLayerTokenEmbdType,
                perLayerModelProj, perLayerProjNorm,
                perLayerInpGate, perLayerProj, perLayerPostNorm,
                ffnGateInp, ffnGateInpScale, ffnGateUpExps, ffnDownExps, ffnDownExpsScale,
                ffnPostNorm1, preFfwNorm2, ffnPostNorm2,
                null // GGMLType determined later
        );
    }

    // Helper: load array of FloatBuffers from F32 tensor entries
    private static FloatBuffer[] loadArrayOfFloatBuffer(int size, java.util.function.IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatBuffer[] array = new FloatBuffer[size];
        for (int i = 0; i < size; i++) {
            array[i] = toFloatBuffer(getTensorEntry.apply(i));
        }
        return array;
    }

    private static FloatBuffer toFloatBuffer(GGMLTensorEntry tensorEntry) {
        return tensorEntry.memorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
    }

    @Override
    protected Gemma4 createModel(Gemma4Configuration config, Tokenizer tokenizer, Weights weights) {
        ChatFormat chatFormat = new Gemma4ChatFormat((Gemma4Tokenizer) tokenizer);
        return new Gemma4(config, tokenizer, weights, chatFormat);
    }

    @Override
    protected Weights createStandardWeights(Map<String, GGMLTensorEntry> tensorEntries,
                                            Gemma4Configuration config, Pair<float[], float[]> ropeFreqs,
                                            GGMLTensorEntry tokenEmbeddings, GGMLTensorEntry outputWeight) {
        // Not used directly - loadWeights is overridden for dual RoPE
        throw new UnsupportedOperationException("Use loadWeights instead");
    }

    @Override
    protected Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries,
                                             Gemma4Configuration config, Pair<float[], float[]> ropeFreqs,
                                             GGMLTensorEntry tokenEmbeddings, GGMLTensorEntry outputWeight) {
        if (TornadoVMMasterPlan.ENABLE_TORNADOVM_INIT_TIME) {
            System.out.println("Loading Gemma4 model weights in TornadoVM format (loading " + outputWeight.ggmlType() + ")");
        }

        GGMLType ggmlType = outputWeight.ggmlType();
        final int nl = config.numberOfLayers();

        // Compute dual RoPE frequencies (same as loadWeights)
        Pair<float[], float[]> ropeFreqsSWA = RoPE.precomputeFreqsCis(
                config.contextLengthModel(), config.headSizeSWA(), config.ropeThetaSWA(),
                false, 0, 0, 0, 0);

        GGMLTensorEntry ropeFreqsEntry = tensorEntries.get("rope_freqs.weight");
        Pair<float[], float[]> ropeFreqsFull;
        if (ropeFreqsEntry != null) {
            FloatBuffer ropeFreqsBuf = ropeFreqsEntry.memorySegment()
                    .asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            float[] modelRopeFreqs = new float[ropeFreqsBuf.remaining()];
            ropeFreqsBuf.get(modelRopeFreqs);
            ropeFreqsFull = RoPE.precomputeFreqsCisFromFreqs(
                    config.contextLengthModel(), config.headSizeFull(), config.ropeTheta(), modelRopeFreqs);
        } else {
            ropeFreqsFull = RoPE.precomputeFreqsCis(
                    config.contextLengthModel(), config.headSizeFull(), config.ropeTheta(),
                    false, 0, 0, 0, 0);
        }

        // Load per-layer output scale
        float[] layerOutputScale = new float[nl];
        for (int i = 0; i < nl; i++) {
            GGMLTensorEntry scaleEntry = tensorEntries.get("blk." + i + ".layer_output_scale.weight");
            if (scaleEntry != null) {
                layerOutputScale[i] = scaleEntry.memorySegment()
                        .asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(0);
            } else {
                layerOutputScale[i] = 1.0f;
            }
        }

        return new Gemma4TornadoWeights(
                loadTornadoTensor(tokenEmbeddings),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                // Q/K norm weights (F32)
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_q_norm.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_k_norm.weight")),
                // Post-attention norm (F32)
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".post_attention_norm.weight")),
                // FFN weights
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                // Post-FFN norm (F32)
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".post_ffw_norm.weight")),
                // Final norm
                loadTornadoTensor(tensorEntries.get("output_norm.weight")),
                // Per-layer output scale
                layerOutputScale,
                // Dual RoPE as FP32TornadoTensors (full = parent's freq_cis)
                new FP32TornadoTensor(FloatArray.fromArray(ropeFreqsFull.first())),
                new FP32TornadoTensor(FloatArray.fromArray(ropeFreqsFull.second())),
                new FP32TornadoTensor(FloatArray.fromArray(ropeFreqsSWA.first())),
                new FP32TornadoTensor(FloatArray.fromArray(ropeFreqsSWA.second())),
                // Output weights
                tensorEntries.containsKey("output.weight")
                        ? loadTornadoTensor(tensorEntries.get("output.weight"))
                        : loadTornadoTensor(tokenEmbeddings),
                ggmlType
        );
    }
}
