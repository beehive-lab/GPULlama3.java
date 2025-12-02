package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.Options;
import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.tensor.GGUF;
import org.beehive.gpullama3.tensor.*;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.ModelType;
import org.beehive.gpullama3.tensor.standard.*;
import org.beehive.gpullama3.tensor.tornado.FP16TornadoTensor;
import org.beehive.gpullama3.tensor.tornado.FP32TornadoTensor;
import org.beehive.gpullama3.tensor.tornado.Q8_0TornadoTensor;
import org.beehive.gpullama3.tensor.tornado.TornadoTensor;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.*;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;
import java.util.function.IntFunction;
import java.util.stream.Collectors;

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
     *
     * <p>If Ahead-of-Time (AOT) mode is enabled, attempts to use a pre-loaded compiled model.
     * Otherwise, loads the model from the specified path using the model loader.
     *
     * @param options the parsed CLI options containing model path and max token limit
     * @return the loaded {@link Model} instance
     * @throws IOException           if the model fails to load
     * @throws IllegalStateException if AOT loading is enabled but the preloaded model is unavailable
     */
    public static Model loadModel(Options options) throws IOException {
        Path ggufPath = options.modelPath();
        int contextLength = options.maxTokens();
        boolean useTornadovm = options.useTornadovm();

        // initial load of metadata from gguf file
        GGUF gguf = GGUF.loadGGUFMetadata(ggufPath);
        // detect model type
        ModelType modelType = detectModelType(gguf.getMetadata());
        // model type-specific load
        return modelType.loadModel(gguf.getFileChannel(), gguf, contextLength, useTornadovm);
    }

    /**
     * Dispatcher method for loading a standard (non-tornado) tensor based on GGML type.
     * Used in CPU-path.
     */
    public static FloatTensor loadTensor(GGMLTensorEntry entry) {
        GGMLType ggmlType = entry.ggmlType();
        return switch (ggmlType) {
            case F32 -> new FP32FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q8_0 -> new Q8_0FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q4_0 -> new Q4_0FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case F16 -> new FP16FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            default -> throw new UnsupportedOperationException("Quantization format " + ggmlType);
        };
    }

    /**
     * Dispatcher method for loading a standard tensor array based on type.
     * Used in CPU-path.
     */
    public static FloatTensor[] loadArrayOfTensors(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatTensor[] array = new FloatTensor[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadTensor(getTensorEntry.apply(i));
        }
        return array;
    }

    /**
     * Dispatcher method for loading a TornadoVM-compatible tensor based on GGML type.
     * Used in GPU-path.
     */
    public static TornadoTensor loadTornadoTensor(GGMLTensorEntry entry) {
        GGMLType ggmlType = entry.ggmlType();
        int size = FloatTensor.numberOfElements(entry.shape());
        return switch (ggmlType) {
            case F32 -> FP32TornadoTensor.fromTornadoMemorySegment(entry.memorySegment());
            case F16 -> FP16TornadoTensor.fromTornadoMemorySegment(entry.memorySegment());
            case Q8_0 -> Q8_0TornadoTensor.createAsQ8_0(entry);
            case Q4_0 -> throw new UnsupportedOperationException("Q4 format not supported yet");
            default -> throw new UnsupportedOperationException("Quantization format " + ggmlType);
        };
    }

    /**
     * Dispatcher method for loading a TornadoVM tensor array based on type.
     * Used in GPU-path.
     */
    public static TornadoTensor[] loadArrayOfTornadoTensors(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        TornadoTensor[] array = new TornadoTensor[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadTornadoTensor(getTensorEntry.apply(i));
        }
        return array;
    }

    /**
     * Load a tensor and manually convert to FP32 (FloatArray).
     * Used for embeddings that currently are treated as FP32.
     * TODO: it is ultra-slow and should be removed
     */
    public static TornadoTensor loadTornadoTensorAsFP32(GGMLTensorEntry entry) {
        TornadoTensor tensor = loadTornadoTensor(entry);
        return switch (tensor.type()) {
            case F32 -> tensor;
            case F16 -> {
                HalfFloatArray tensorHFA = tensor.asHalfFloatArray();
                int numOfElements = tensorHFA.getSize();
                FloatArray tensorFA = new FloatArray(numOfElements);
                for (int i = 0; i < numOfElements; i++) {
                    tensorFA.set(i, tensorHFA.get(i).getFloat32());
                }
                yield new FP32TornadoTensor(tensorFA);
            }
            case Q8_0 -> Q8_0TornadoTensor.createAsFP32(entry);
            default -> {
                throw new UnsupportedOperationException("Unsupported tensor type: " + tensor.type());
            }
        };
    }

    // Helper methods

    public static FloatArray[] loadArrayAsFloatArray(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatArray[] array = new FloatArray[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadTensorAsFloatArray(getTensorEntry.apply(i));
        }
        return array;
    }

    public static HalfFloatArray[] loadArrayAsHalfFloatArray(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        HalfFloatArray[] array = new HalfFloatArray[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadTensorAsHalfFloatArray(getTensorEntry.apply(i));
        }
        return array;
    }

    public static Q8_0TornadoTensor[] loadArrayAsQ8_0TornadoTensor(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        Q8_0TornadoTensor[] array = new Q8_0TornadoTensor[size];
        for (int i = 0; i < size; i++) {
            array[i] = Q8_0TornadoTensor.createAsQ8_0(getTensorEntry.apply(i));
        }
        return array;
    }

    public static FloatArray floatBufferToFloatArray(GGMLTensorEntry tensorEntry) {
        if (tensorEntry.ggmlType() == GGMLType.F32) {
            FloatBuffer buffer = tensorEntry.memorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            return FloatArray.fromFloatBuffer(buffer);
        } else {
            throw new UnsupportedOperationException("Conversion to FloatArray from " + tensorEntry.ggmlType());
        }
    }

    public static FloatArray[] loadArrayAsFloatArrayFromBuffer(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatArray[] array = new FloatArray[size];
        for (int i = 0; i < size; i++) {
            array[i] = floatBufferToFloatArray(getTensorEntry.apply(i));
        }
        return array;
    }

    public static ByteArray createByteArrayFromTensor(GGMLTensorEntry entry) {
        FloatTensor tensor = loadTensor(entry);
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
            FloatTensor tensor = loadTensor(entry);
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
            FloatTensor tensor = loadTensor(entry);
            HalfFloatArray array = new HalfFloatArray(tensor.size());
            for (int i = 0; i < tensor.size(); i++) {
                HalfFloat x = new HalfFloat(tensor.getFloat(i));
                array.set(i, x);
            }
            return array;
        }
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
}
