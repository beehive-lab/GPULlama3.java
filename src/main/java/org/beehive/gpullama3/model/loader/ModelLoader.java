package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.Options;
import org.beehive.gpullama3.aot.AOT;
import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.core.model.tensor.ArrayFloatTensor;
import org.beehive.gpullama3.core.model.tensor.F16FloatTensor;
import org.beehive.gpullama3.core.model.tensor.F32FloatTensor;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.core.model.tensor.Q4_0FloatTensor;
import org.beehive.gpullama3.core.model.tensor.Q8_0FloatTensor;
import org.beehive.gpullama3.core.model.tensor.Q8_0QuantizedTensor;
import org.beehive.gpullama3.core.types.Pair;
import org.beehive.gpullama3.inference.operation.RoPE;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.standard.LlamaStandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.LlamaTornadoWeights;
import org.beehive.gpullama3.inference.weights.tornado.Q8_0Weights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.ModelType;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.Int8Array;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.function.IntFunction;

public abstract class ModelLoader {

    public static final boolean USE_AOT = Boolean.parseBoolean(System.getProperty("llama.AOT", "false"));               // Use Ahead-of-Time compilation

    protected FileChannel fileChannel;
    protected GGUF gguf;
    protected int contextLength;
    protected boolean loadWeights;
    protected boolean useTornadovm;

    public ModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights, boolean useTornadovm) {
        this.fileChannel = fileChannel;
        this.gguf = gguf;
        this.contextLength = contextLength;
        this.loadWeights = loadWeights;
        this.useTornadovm = useTornadovm;
    }

    private static ModelType detectModelType(Map<String, Object> metadata) {
        String name = (String) metadata.get("general.name");
        String tokenizerModel = (String) metadata.get("tokenizer.ggml.model");
        Integer vocabSize = (Integer) metadata.get("llama.vocab_size");

        // Check by name first
        if (name != null) {
            String lowerName = name.toLowerCase();
            if (lowerName.contains("mistral")) {
                return ModelType.MISTRAL;
            } else if (lowerName.contains("llama")) {
                return ModelType.LLAMA_3;
            } else if (lowerName.contains("qwen2")) {
                return ModelType.QWEN_2;
            } else if (lowerName.contains("qwen3")) {
                return ModelType.QWEN_3;
            } else if (lowerName.contains("deepseek r1 distill")) {
                return ModelType.DEEPSEEK_R1_DISTILL_QWEN;
            } else if (lowerName.contains("phi3")) {
                return ModelType.PHI_3;
            } else if (lowerName.contains("gemma")) {
                return ModelType.GEMMA_3;
            }

        }

        return ModelType.UNKNOWN;
    }

    /**
     * Loads the language model based on the given options.
     * <p>
     * If Ahead-of-Time (AOT) mode is enabled, attempts to use a pre-loaded compiled model. Otherwise, loads the model from the specified path using the model loader.
     * </p>
     *
     * @param options
     *         the parsed CLI options containing model path and max token limit
     * @return the loaded {@link Model} instance
     * @throws IOException
     *         if the model fails to load
     * @throws IllegalStateException
     *         if AOT loading is enabled but the preloaded model is unavailable
     */
    public static Model loadModel(Options options) throws IOException {
        if (USE_AOT) {
            Model model = AOT.tryUsePreLoaded(options.modelPath(), options.maxTokens());
            if (model == null) {
                throw new IllegalStateException("Failed to load precompiled AOT model.");
            }
            return model;
        }
        return ModelLoader.loadModel(options.modelPath(), options.maxTokens(), true, options.useTornadovm());
    }

    public static Model loadModel(Path ggufPath, int contextLength, boolean loadWeights, boolean useTornadovm) throws IOException {
        // initial load of metadata from gguf file
        GGUF gguf = GGUF.loadModel(ggufPath);
        FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ);
        // detect model type
        ModelType modelType = detectModelType(gguf.getMetadata());
        // model type-specific load
        return modelType.loadModel(fileChannel, gguf, contextLength, loadWeights, useTornadovm);
    }

    public static FloatTensor loadQuantized(GGMLTensorEntry entry) {
        GGMLType ggmlType = entry.ggmlType();
        return switch (ggmlType) {
            case F32 -> new F32FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q8_0 -> new Q8_0FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q4_0 -> new Q4_0FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case F16 -> new F16FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            default -> throw new UnsupportedOperationException("Quantization format " + ggmlType);
        };
    }

    public static FloatArray[] loadArrayAsFloatArray(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatArray[] array = new FloatArray[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadTensorAsFloatArray(getTensorEntry.apply(i));
        }
        return array;
    }
    //@formatter:on

    public static HalfFloatArray[] loadArrayAsHalfFloatArray(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        HalfFloatArray[] array = new HalfFloatArray[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadTensorAsHalfFloatArray(getTensorEntry.apply(i));
        }
        return array;
    }

    public static Q8_0QuantizedTensor[] loadArrayAsQ8_0QuantizedTensor(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        Q8_0QuantizedTensor[] array = new Q8_0QuantizedTensor[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadQ8_0QuantizedTensor(getTensorEntry.apply(i));
        }
        return array;
    }

    //@formatter:off

    public static FloatArray floatBufferToFloatArray(GGMLTensorEntry tensorEntry) {
        if (tensorEntry.ggmlType() == GGMLType.F32) {
            FloatBuffer buffer = tensorEntry.memorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            return FloatArray.fromFloatBuffer(buffer);
        } else {
            throw new UnsupportedOperationException("Conversion to FloatArray from " + tensorEntry.ggmlType());
        }
    }
    //@formatter:on

    public static FloatArray[] loadArrayAsFloatArrayFromBuffer(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatArray[] array = new FloatArray[size];
        for (int i = 0; i < size; i++) {
            array[i] = floatBufferToFloatArray(getTensorEntry.apply(i));
        }
        return array;
    }

    public static ByteArray createByteArrayFromTensor(GGMLTensorEntry entry) {
        FloatTensor tensor = loadQuantized(entry);
        return ByteArray.fromSegment(tensor.asMemorySegment());
    }

    public static FloatArray loadTensorAsFloatArray(GGMLTensorEntry entry) {
        if (entry.ggmlType() == GGMLType.F32) {
            // For F32, we can directly create FloatArray from memory
            FloatBuffer buffer = entry.memorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            FloatArray array = new FloatArray(buffer.remaining());
            for (int i = 0; i < buffer.remaining(); i++) {
                array.set(i, buffer.get());
            }
            return array;
        } else {
            // For quantized formats, we need to load through FloatTensor
            FloatTensor tensor = loadQuantized(entry);
            FloatArray array = new FloatArray(tensor.size());
            for (int i = 0; i < tensor.size(); i++) {
                array.set(i, tensor.getFloat(i));
            }
            return array;
        }
    }

    public static HalfFloatArray loadTensorAsHalfFloatArray(GGMLTensorEntry entry) {
        if (entry.ggmlType() == GGMLType.F32) {
            System.out.println("Loading F32 tensor as HalfFloatArray");
            return null;
        } else {
            // For quantized formats, we need to load through FloatTensor
            FloatTensor tensor = loadQuantized(entry);
            HalfFloatArray array = new HalfFloatArray(tensor.size());
            for (int i = 0; i < tensor.size(); i++) {
                HalfFloat x = new HalfFloat(tensor.getFloat(i));
                array.set(i, x);
            }
            return array;
        }
    }

    public static Q8_0QuantizedTensor loadQ8_0QuantizedTensor(GGMLTensorEntry entry) {
        if (entry.ggmlType() != GGMLType.Q8_0) {
            throw new IllegalArgumentException("Expected Q8_0 tensor, got: " + entry.ggmlType() + " for tensor: " + entry.name());
        }

        int[] shape = entry.shape();
        int size = FloatTensor.numberOfElements(shape);
        int numBlocks = size / GGMLType.Q8_0.getBlockSize();

        if (size % GGMLType.Q8_0.getBlockSize() != 0) {
            throw new IllegalArgumentException("Q8_0 tensor size must be multiple of " + GGMLType.Q8_0.getBlockSize() + ", got: " + size + " for tensor: " + entry.name());
        }

        MemorySegment q8Segment = entry.memorySegment();

        // allocate the arrays for quantized data (int8) and scales (fp16)
        HalfFloatArray scales = new HalfFloatArray(numBlocks);
        Int8Array quants = new Int8Array(size);

        // unpack Q8_0 blocks: [2 bytes fp16 scale][32 bytes int8 quants]
        ValueLayout.OfShort shortLayout = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
        ValueLayout.OfByte byteLayout = ValueLayout.JAVA_BYTE;

        for (int block = 0; block < numBlocks; block++) {
            long blockOffset = block * 34L;  // 34 bytes per block

            // read fp16 scale (first 2 bytes of block)
            short scaleRaw = q8Segment.get(shortLayout, blockOffset);
            scales.set(block, new HalfFloat(scaleRaw));

            // read 32 int8 quantized values (remaining bytes of block)
            for (int i = 0; i < 32; i++) {
                byte quantValue = q8Segment.get(byteLayout, blockOffset + 2 + i);
                quants.set(block * 32 + i, quantValue);
            }
        }

        return new Q8_0QuantizedTensor(size, scales, quants, q8Segment);
    }

    public static FloatTensor[] loadArrayOfQuantized(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatTensor[] array = new FloatTensor[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadQuantized(getTensorEntry.apply(i));
        }
        return array;
    }

    public static FloatBuffer[] loadArrayOfFloatBuffer(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatBuffer[] array = new FloatBuffer[size];
        for (int i = 0; i < size; i++) {
            array[i] = toFloatBuffer(getTensorEntry.apply(i));
        }
        return array;
    }

    public static FloatBuffer toFloatBuffer(GGMLTensorEntry tensorEntry) {
        GGMLType ggmlType = tensorEntry.ggmlType();
        return switch (ggmlType) {
            case F32 -> tensorEntry.memorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            default -> throw new UnsupportedOperationException("Conversion to " + ggmlType);
        };
    }

    public abstract Model loadModel();

    //@formatter:off
    public Weights loadWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config) {
        boolean ropeScaling = tensorEntries.containsKey("rope_freqs");
        RopeConfig ropeConfig = new RopeConfig(8.0f,         // scaleFactor
                1.0f,                    // loFreqFactor
                3.0f,                    // hiFreqFactor
                8192                     // oldContextLength
        );

        Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(
                config.contextLength(),         // Maximum sequence length the model can process
                config.headSize(),              // Dimension of each attention head
                config.ropeTheta(),             // Base frequency parameter (typically 10000.0)
                ropeScaling,                    // Whether to apply frequency scaling (determined by model type)
                ropeConfig.scaleFactor,         // Scale factor for extending context length (NTK-aware scaling)
                ropeConfig.loFreqFactor,        // Low frequency scaling factor for better long-range dependencies
                ropeConfig.hiFreqFactor,        // High frequency scaling factor for preserving local precision
                ropeConfig.oldContextLength     // Original context length the model was trained with
        );

        GGMLTensorEntry tokenEmbeddings = tensorEntries.get("token_embd.weight");
        GGMLTensorEntry outputWeight = tensorEntries.getOrDefault("output.weight", tokenEmbeddings);

        if (useTornadovm) {
            if (TornadoVMMasterPlan.ENABLE_TORNADOVM_INIT_TIME) {
                System.out.println("Loading model weights in TornadoVM format (loading " + outputWeight.ggmlType() + ")");
            }

            if (outputWeight.ggmlType() == GGMLType.Q8_0) {
                return createTornadoVMWeightsQ8_0(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
            } else {
                return createTornadoVMWeights(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
             }
        } else {
            return createStandardWeights(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
        }
    }

    public Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        return new LlamaTornadoWeights(
                // Load directly to TornadoVM format
                loadTensorAsFloatArray(tokenEmbeddings), loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayAsHalfFloatArray(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_up.weight")), floatBufferToFloatArray(tensorEntries.get("output_norm.weight")),
                FloatArray.fromArray(ropeFreqs.first()), FloatArray.fromArray(ropeFreqs.second()), loadTensorAsHalfFloatArray(outputWeight), outputWeight.ggmlType()) {
        };
    }

    private Q8_0Weights createTornadoVMWeightsQ8_0(Map<String, GGMLTensorEntry> tensorEntries, Configuration config,  Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings, GGMLTensorEntry outputWeight) {
        return new Q8_0Weights(
                loadTensorAsFloatArray(tokenEmbeddings),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayAsQ8_0QuantizedTensor(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayAsQ8_0QuantizedTensor(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayAsQ8_0QuantizedTensor(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayAsQ8_0QuantizedTensor(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayAsFloatArrayFromBuffer(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayAsQ8_0QuantizedTensor(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayAsQ8_0QuantizedTensor(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayAsQ8_0QuantizedTensor(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                floatBufferToFloatArray(tensorEntries.get("output_norm.weight")),
                FloatArray.fromArray(ropeFreqs.first()),
                FloatArray.fromArray(ropeFreqs.second()),
                loadQ8_0QuantizedTensor(outputWeight),
                outputWeight.ggmlType()
        );
    }

    /**
     * Creates weights in standard format only
     */
    public Weights createStandardWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
            GGMLTensorEntry outputWeight) {
        return new LlamaStandardWeights(
                loadQuantized(tokenEmbeddings),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfQuantized(config.numberOfLayers(), i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                loadQuantized(tensorEntries.get("output_norm.weight")),
                new ArrayFloatTensor(ropeFreqs.first()),
                new ArrayFloatTensor(ropeFreqs.second()),
                loadQuantized(outputWeight),
                outputWeight.ggmlType());
    }

    // Helper class to encapsulate RoPE configuration parameters
    private static class RopeConfig {
        final float scaleFactor;
        final float loFreqFactor;
        final float hiFreqFactor;
        final int oldContextLength;

        RopeConfig(float scaleFactor, float loFreqFactor, float hiFreqFactor, int oldContextLength) {
            this.scaleFactor = scaleFactor;
            this.loFreqFactor = loFreqFactor;
            this.hiFreqFactor = hiFreqFactor;
            this.oldContextLength = oldContextLength;
        }
    }

}
