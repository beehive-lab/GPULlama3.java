package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.auxiliary.Pair;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.standard.Gemma4StandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.Gemma4TornadoWeights;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.gemma4.Gemma4;
import org.beehive.gpullama3.model.gemma4.Gemma4Configuration;
import org.beehive.gpullama3.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.tensor.GGUF;
import org.beehive.gpullama3.tensor.GGUF.GGUFTensorInfo;
import org.beehive.gpullama3.tensor.standard.ArrayFloatTensor;
import org.beehive.gpullama3.tensor.tornado.FP32TornadoTensor;
import org.beehive.gpullama3.tensor.tornado.TornadoTensor;
import org.beehive.gpullama3.tokenizer.Gemma4Tokenizer;
import org.beehive.gpullama3.tokenizer.Tokenizer;
import org.beehive.gpullama3.tokenizer.Vocabulary;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.Map;
import java.util.function.IntFunction;

import static org.beehive.gpullama3.model.loader.ModelLoader.loadArrayOfTensors;
import static org.beehive.gpullama3.model.loader.ModelLoader.loadArrayOfTornadoTensors;
import static org.beehive.gpullama3.model.loader.ModelLoader.loadTensor;
import static org.beehive.gpullama3.model.loader.ModelLoader.loadTornadoTensor;

/**
 * Loader for Gemma 4 models (e.g. Gemma-4-E2B-It).
 *
 * <p>Gemma 4 needs two distinct precomputed RoPE tables (sliding-window vs. full/global attention
 * layers use different bases and head dimensions, and full-attention layers additionally apply a
 * per-dimension frequency scaling stored in the {@code rope_freqs} tensor), so RoPE frequencies are
 * computed directly here -- where the tensor entries are available -- rather than through the
 * generic {@link #precomputeRopeFrequencies} hook.</p>
 */
public class Gemma4ModelLoader extends AbstractModelLoader<Gemma4, Gemma4Configuration> {

    public Gemma4ModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, useTornadovm);
    }

    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        return Vocabulary.loadGemma4Vocabulary(metadata);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        return new Gemma4Tokenizer(metadata, vocabulary);
    }

    // @formatter:off
    @Override
    protected Gemma4Configuration createConfiguration(Map<String, Object> metadata) {
        int modelContextLength = (int) metadata.get("gemma4.context_length");
        int finalContextLength = (contextLength < 0 || modelContextLength < contextLength) ? modelContextLength : contextLength;
        int numberOfLayers = (int) metadata.get("gemma4.block_count");

        return new Gemma4Configuration(
                getModelQuantization(metadata),
                (int) metadata.get("gemma4.embedding_length"),
                numberOfLayers,
                (int) metadata.get("gemma4.attention.head_count"),
                (int) metadata.get("gemma4.attention.head_count_kv"),
                (int) metadata.get("gemma4.attention.key_length_swa"),
                (int) metadata.get("gemma4.attention.key_length"),
                (int[]) metadata.get("gemma4.feed_forward_length"),
                (boolean[]) metadata.get("gemma4.attention.sliding_window_pattern"),
                (int) metadata.get("gemma4.attention.sliding_window"),
                (int) metadata.get("gemma4.attention.shared_kv_layers"),
                (int) metadata.get("gemma4.embedding_length_per_layer_input"),
                vocabulary.size(),
                modelContextLength,
                finalContextLength,
                (float) metadata.get("gemma4.attention.layer_norm_rms_epsilon"),
                (float) metadata.get("gemma4.rope.freq_base"),
                (float) metadata.get("gemma4.rope.freq_base_swa"),
                (float) metadata.get("gemma4.final_logit_softcapping")
        );
    }
    // @formatter:on

    /** Gemma4 needs two RoPE tables computed with tensor data (rope_freqs); see {@link #ropeTables}. */
    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(Gemma4Configuration config) {
        return null;
    }

    @Override
    protected Gemma4 createModel(Gemma4Configuration config, Tokenizer tokenizer, Weights weights) {
        return new Gemma4(config, tokenizer, weights, ChatFormat.create(tokenizer, null));
    }

    // @formatter:off
    @Override
    protected Weights createStandardWeights(Map<String, GGMLTensorEntry> tensorEntries, Gemma4Configuration config, Pair<float[], float[]> ropeFreqs,
                                             GGMLTensorEntry tokenEmbeddings, GGMLTensorEntry outputWeight) {
        final int nl = config.numberOfLayers();
        RopeTables ropeTables = computeRopeTables(tensorEntries, config);

        return new Gemma4StandardWeights(
                loadTensor(tokenEmbeddings),
                tensorEntries.containsKey("output.weight") ? loadTensor(tensorEntries.get("output.weight")) : loadTensor(tokenEmbeddings),
                loadTensor(tensorEntries.get("output_norm.weight")),

                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_q_norm.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_k_norm.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".post_attention_norm.weight")),

                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".post_ffw_norm.weight")),

                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".inp_gate.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".proj.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".post_norm.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".layer_output_scale.weight")),

                tensorEntries.get("per_layer_token_embd.weight"),
                loadTensor(tensorEntries.get("per_layer_model_proj.weight")),
                loadTensor(tensorEntries.get("per_layer_proj_norm.weight")),

                new ArrayFloatTensor(ropeTables.realSwa),
                new ArrayFloatTensor(ropeTables.imagSwa),
                new ArrayFloatTensor(ropeTables.realFull),
                new ArrayFloatTensor(ropeTables.imagFull),

                null
        );
    }
    // @formatter:on

    // @formatter:off
    @Override
    protected Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries, Gemma4Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        final int nl = config.numberOfLayers();
        GGMLType ggmlType = effectiveGpuWeightType(outputWeight.ggmlType());
        RopeTables ropeTables = computeRopeTables(tensorEntries, config);

        return new Gemma4TornadoWeights(
                loadTornadoTensor(tokenEmbeddings),

                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_q_norm.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_k_norm.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".post_attention_norm.weight")),

                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".post_ffw_norm.weight")),

                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".inp_gate.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".proj.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".post_norm.weight")),
                loadArrayOfTornadoTensorsNullable(nl, i -> tensorEntries.get("blk." + i + ".layer_output_scale.weight")),

                stripTornadoArrayHeader(tensorEntries.get("per_layer_token_embd.weight")),
                loadTornadoTensor(tensorEntries.get("per_layer_model_proj.weight")),
                loadTornadoTensor(tensorEntries.get("per_layer_proj_norm.weight")),
                loadTornadoTensor(tensorEntries.get("output_norm.weight")),

                new FP32TornadoTensor(FloatArray.fromArray(ropeTables.realSwa)),
                new FP32TornadoTensor(FloatArray.fromArray(ropeTables.imagSwa)),
                new FP32TornadoTensor(FloatArray.fromArray(ropeTables.realFull)),
                new FP32TornadoTensor(FloatArray.fromArray(ropeTables.imagFull)),

                tensorEntries.containsKey("output.weight") ? loadTornadoTensor(tensorEntries.get("output.weight")) : loadTornadoTensor(tokenEmbeddings),
                ggmlType
        );
    }
    // @formatter:on

    /**
     * Tensor entries produced by {@link GGUF#loadTensorsTornado} prefix every {@code memorySegment()} with a
     * 16-byte {@link TornadoNativeArray#ARRAY_HEADER} (so the data can be wrapped as a TornadoVM native array
     * without copying) -- but {@code per_layer_token_embd} is kept as a raw entry and addressed with
     * byte-offset arithmetic that assumes the segment starts at the tensor's actual data (see
     * {@link ModelLoader#copyEmbeddingRowToFloatArray}, mirroring the CPU path's {@link ModelLoader#copyEmbeddingRow}
     * over a {@link GGUF#loadTensorsStandard}-produced entry, which has no such header). Slice past the
     * header here so both code paths see the same layout.
     */
    private static GGMLTensorEntry stripTornadoArrayHeader(GGMLTensorEntry entry) {
        long headerBytes = TornadoNativeArray.ARRAY_HEADER;
        return new GGMLTensorEntry(entry.mappedFile(), entry.name(), entry.ggmlType(), entry.shape(), entry.memorySegment().asSlice(headerBytes));
    }

    /** Like {@link ModelLoader#loadArrayOfTornadoTensors}, but tolerates missing entries (Gemma4's optional per-layer output scale). */
    private static TornadoTensor[] loadArrayOfTornadoTensorsNullable(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        TornadoTensor[] array = new TornadoTensor[size];
        for (int i = 0; i < size; i++) {
            GGMLTensorEntry entry = getTensorEntry.apply(i);
            array[i] = (entry == null) ? null : loadTornadoTensor(entry);
        }
        return array;
    }

    private record RopeTables(float[] realSwa, float[] imagSwa, float[] realFull, float[] imagFull) {
    }

    /**
     * Computes the two RoPE frequency tables Gemma4 needs.
     * <p>
     * Sliding-window layers use {@code rope_theta_swa} with {@code headDimSwa} and no extra scaling.
     * Full/global-attention layers use {@code rope_theta} with {@code headDimFull}, additionally
     * dividing each rotation angle by the corresponding entry of the (single, shared) {@code
     * rope_freqs} tensor -- this is how the GGUF encodes "partial RoPE" (entries are 1.0 for the
     * active low-frequency dimensions and effectively infinite for the inactive ones, which zeroes
     * out their rotation).
     */
    private RopeTables computeRopeTables(Map<String, GGMLTensorEntry> tensorEntries, Gemma4Configuration config) {
        Pair<float[], float[]> swa = precomputeFreqsCisWithFactors(config.contextLengthModel(), config.headDimSwa(), config.ropeThetaSwa(), null);

        // rope_freqs.weight is intentionally excluded from tensorEntries by GGUF.loadTensorsStandard/
        // loadTensorsTornado (it isn't needed by most architectures), so read it directly here.
        float[] freqFactors = readFloat32TensorDirect("rope_freqs.weight");
        Pair<float[], float[]> full = precomputeFreqsCisWithFactors(config.contextLengthModel(), config.headDimFull(), config.ropeTheta(), freqFactors);

        return new RopeTables(swa.first(), swa.second(), full.first(), full.second());
    }

    /** Reads a small F32 tensor's raw data directly from the GGUF file, bypassing the {@code tensorEntries} map. */
    private float[] readFloat32TensorDirect(String tensorName) {
        GGUFTensorInfo info = gguf.getTensorInfos().get(tensorName);
        if (info == null) {
            return null;
        }
        if (info.ggmlType() != GGMLType.F32) {
            throw new UnsupportedOperationException("Expected F32 tensor for " + tensorName + ", got " + info.ggmlType());
        }

        int numberOfElements = 1;
        for (int dimension : info.dimensions()) {
            numberOfElements *= dimension;
        }

        long byteOffset = gguf.getTensorDataOffset() + info.offset();
        ByteBuffer buffer = ByteBuffer.allocate(numberOfElements * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        try {
            while (buffer.hasRemaining()) {
                if (gguf.getFileChannel().read(buffer, byteOffset + buffer.position()) < 0) {
                    throw new EOFException("Unexpected end of file while reading " + tensorName);
                }
            }
        } catch (IOException e) {
            throw new ModelLoadException("Failed to read " + tensorName + " from GGUF file", e);
        }
        buffer.flip();

        float[] result = new float[numberOfElements];
        buffer.asFloatBuffer().get(result);
        return result;
    }

    /** Like {@link org.beehive.gpullama3.inference.operation.RoPE#precomputeFreqsCis}, but allows dividing each pair's frequency by a per-dimension scaling factor (NeoX-style RoPE). */
    private static Pair<float[], float[]> precomputeFreqsCisWithFactors(int contextLength, int headSize, double theta, float[] freqFactors) {
        assert headSize % 2 == 0;
        float[] cr = new float[contextLength * (headSize / 2)];
        float[] ci = new float[contextLength * (headSize / 2)];
        int n = 0;
        for (int pos = 0; pos < contextLength; ++pos) {
            for (int i = 0; i < headSize; i += 2) {
                int pairIndex = i / 2;
                float freq = (float) (1.0 / Math.pow(theta, i / (double) headSize));
                if (freqFactors != null) {
                    freq = freq / freqFactors[pairIndex];
                }
                float val = pos * freq;
                cr[n] = (float) Math.cos(val);
                ci[n] = (float) Math.sin(val);
                n++;
            }
        }
        assert contextLength * (headSize / 2) == n;
        return new Pair<>(cr, ci);
    }
}
