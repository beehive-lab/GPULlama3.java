package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.Options;
import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.core.model.tensor.*;
import org.beehive.gpullama3.core.model.tensor.F16FloatTensor;
import org.beehive.gpullama3.core.model.tensor.F32FloatTensor;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.core.model.tensor.Q4_0FloatTensor;
import org.beehive.gpullama3.core.model.tensor.Q8_0FloatTensor;
import org.beehive.gpullama3.core.model.tensor.Q8_0QuantizedTensor;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.ModelType;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.*;

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
            } else if (lowerName.contains("phi3") || lowerName.contains("phi-3")) {
                return ModelType.PHI_3;
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

    /**
     * Dispatcher method for loading a standard (non-tornado) tensor based on type.
     * Used in CPU-path.
     */
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

    /**
     * Dispatcher method for loading a standard tensor array based on type.
     * Used in CPU-path.
     */
    public static FloatTensor[] loadArrayOfQuantized(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatTensor[] array = new FloatTensor[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadQuantized(getTensorEntry.apply(i));
        }
        return array;
    }

    /**
     * [WIP]
     * Dispatcher method for loading a TornadoVM tensor based on type.
     * Used in GPU-path.
     *
     * TODO: fix this to follow loadQuantized logic
     */
    public static FloatTensor loadTornadoTensor(GGMLTensorEntry entry) {
        GGMLType ggmlType = entry.ggmlType();
        int size = FloatTensor.numberOfElements(entry.shape());
        return switch (ggmlType) {
//            case F32 -> new F32QuantizedTensor(size, entry.memorySegment());
            case Q8_0 -> loadQ8_0QuantizedTensor(entry);
//            case Q4_0 -> throw new UnsupportedOperationException("Not yet implemented");
//            //FloatTensor.numberOfElements(entry.shape()), entry.memorySegment()
//            case F16 -> new F16QuantizedTensor(size, entry.memorySegment());
//            /*{
//                HalfFloatArray array = new HalfFloatArray();
//                array.getSegment().copyFrom(entry.memorySegment());
//                // or array.getSegmentWithHeader() ?
//            }*/
            default -> throw new UnsupportedOperationException("Quantization format " + ggmlType);
        };
    }

    /**
     * Dispatcher method for loading a TornadoVM tensor array based on type.
     * Used in GPU-path.
     */
    public static FloatTensor[] loadTornadoTensorArray(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatTensor[] array = new FloatTensor[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadTornadoTensor(getTensorEntry.apply(i));
        }
        return array;
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

    // TODO: rename to loadQ8_0Tensor
    // move to a utils class
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
            // TODO: use GGML type method for the 34L size
            long blockOffset = block * 34L;  // 34 bytes per block

            // read fp16 scale (first 2 bytes of block)
            short scaleRaw = q8Segment.get(shortLayout, blockOffset);
            scales.set(block, new HalfFloat(scaleRaw));

            // read 32 int8 quantized values (remaining bytes of block)
            // TODO: use GGML type method for the 32 size
            for (int i = 0; i < 32; i++) {
                byte quantValue = q8Segment.get(byteLayout, blockOffset + 2 + i);
                quants.set(block * 32 + i, quantValue);
            }
        }

        return new Q8_0QuantizedTensor(size, scales, quants, q8Segment);
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
