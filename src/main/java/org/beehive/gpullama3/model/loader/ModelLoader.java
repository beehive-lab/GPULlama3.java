package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.Options;
import org.beehive.gpullama3.auxiliary.RunMetrics;
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
import java.lang.foreign.Arena;
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
            if (lowerName.contains("granite")) {
                return ModelType.GRANITE;
            } else if (lowerName.contains("gemma-4") || lowerName.contains("gemma 4")) {
                return ModelType.GEMMA_4;
            } else if (lowerName.contains("devstral")) {
                return ModelType.DEVSTRAL_2;
            } else if (lowerName.contains("mistral")) {
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

        // Alternative: check by metadata keys if name-based detection fails
        if (metadata.containsKey("granite.block_count")) {
            return ModelType.GRANITE;
        }
        if ("gemma4".equals(metadata.get("general.architecture")) || metadata.containsKey("gemma4.block_count")) {
            return ModelType.GEMMA_4;
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

        long start = System.nanoTime();
        GGUF gguf = GGUF.loadGGUFMetadata(ggufPath);
        ModelType modelType = detectModelType(gguf.getMetadata());
        Model model = modelType.loadModel(gguf.getFileChannel(), gguf, contextLength, useTornadovm);
        RunMetrics.setLoadDuration(System.nanoTime() - start);
        return model;
    }

    /**
     * For compatibility with langchain4j and quarkus.
     */
    public static Model loadModel(Path ggufPath, int contextLength, boolean loadWeights, boolean useTornadovm) throws IOException {
        long start = System.nanoTime();
        GGUF gguf = GGUF.loadGGUFMetadata(ggufPath);
        ModelType modelType = detectModelType(gguf.getMetadata());
        Model model = modelType.loadModel(gguf.getFileChannel(), gguf, contextLength, useTornadovm);
        RunMetrics.setLoadDuration(System.nanoTime() - start);
        return model;
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
            case Q4_K -> new Q4_KFloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q5_K -> new Q5_KFloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q6_K -> new Q6_KFloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case F16 -> new FP16FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case BF16 -> new BF16FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
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
            case BF16 -> convertBF16ToFP16TornadoTensor(entry);
            case Q8_0 -> Q8_0TornadoTensor.fromTornadoMemorySegment(entry.memorySegment());
            case Q4_K, Q5_K, Q6_K -> dequantizeToQ8_0TornadoTensor(entry);
            case Q4_0 -> throw new UnsupportedOperationException("Q4_0 format not supported for TornadoVM yet");
            default -> throw new UnsupportedOperationException("Quantization format " + ggmlType);
        };
    }

    /**
     * Dequantizes a K-quant tensor (Q4_K, Q5_K, Q6_K) to Q8_0 format for TornadoVM/GPU execution.
     * This is a load-time conversion that allows K-quant models to run on GPU with existing Q8_0 kernels.
     */
    private static Q8_0TornadoTensor dequantizeToQ8_0TornadoTensor(GGMLTensorEntry entry) {
        // The entry's memorySegment includes a TornadoVM ARRAY_HEADER prefix (16 bytes of zeros).
        // Slice past it so the K-quant FloatTensor reads raw tensor data starting at byte 0.
        long headerBytes = TornadoNativeArray.ARRAY_HEADER;
        GGMLTensorEntry dataEntry = new GGMLTensorEntry(
                entry.mappedFile(), entry.name(), entry.ggmlType(), entry.shape(),
                entry.memorySegment().asSlice(headerBytes));
        FloatTensor sourceTensor = loadTensor(dataEntry);
        int numElements = sourceTensor.size();
        int blockSize = 32;
        int blocksNeeded = (numElements + blockSize - 1) / blockSize;
        int q8BlockBytes = 34; // 2 bytes scale + 32 bytes quants
        int q8BytesNeeded = blocksNeeded * q8BlockBytes;

        byte[] q8Data = new byte[q8BytesNeeded];

        for (int b = 0; b < blocksNeeded; b++) {
            int start = b * blockSize;
            int end = Math.min(start + blockSize, numElements);

            // Find max absolute value for scale
            float maxAbs = 0;
            for (int i = start; i < end; i++) {
                maxAbs = Math.max(maxAbs, Math.abs(sourceTensor.getFloat(i)));
            }
            float scale = maxAbs / 127.0f;

            // Write scale as fp16 (little-endian)
            short scaleF16 = Float.floatToFloat16(scale);
            int blockOff = b * q8BlockBytes;
            q8Data[blockOff] = (byte) (scaleF16 & 0xFF);
            q8Data[blockOff + 1] = (byte) ((scaleF16 >> 8) & 0xFF);

            // Quantize values
            float invScale = scale != 0 ? 1.0f / scale : 0;
            for (int i = start; i < end; i++) {
                int qi = Math.round(sourceTensor.getFloat(i) * invScale);
                qi = Math.max(-128, Math.min(127, qi));
                q8Data[blockOff + 2 + (i - start)] = (byte) qi;
            }
        }

        // Allocate native memory with TornadoNativeArray header, matching GGUF.loadTensorsTornado layout
        MemorySegment nativeSegment = Arena.ofAuto().allocate(headerBytes + q8BytesNeeded, 4);
        // Zero out the header
        for (int i = 0; i < headerBytes; i++) {
            nativeSegment.set(ValueLayout.JAVA_BYTE, i, (byte) 0);
        }
        // Copy Q8_0 data after header
        MemorySegment.copy(MemorySegment.ofArray(q8Data), 0, nativeSegment, headerBytes, q8BytesNeeded);
        return Q8_0TornadoTensor.fromTornadoMemorySegment(nativeSegment);
    }

    /**
     * Converts a BF16 tensor to an FP16 {@link FP16TornadoTensor} for TornadoVM/GPU execution.
     * TornadoVM has no native BF16 kernel support, so weights are widened to FP32 (a lossless,
     * simple bit-shift for BF16) and narrowed to IEEE FP16 at load time -- the same representation
     * the existing FP16 GPU kernels already expect (see {@link #loadTornadoTensor}).
     */
    private static FP16TornadoTensor convertBF16ToFP16TornadoTensor(GGMLTensorEntry entry) {
        long headerBytes = TornadoNativeArray.ARRAY_HEADER;
        GGMLTensorEntry dataEntry = new GGMLTensorEntry(
                entry.mappedFile(), entry.name(), entry.ggmlType(), entry.shape(),
                entry.memorySegment().asSlice(headerBytes));
        FloatTensor source = loadTensor(dataEntry);
        int numElements = source.size();

        MemorySegment nativeSegment = Arena.ofAuto().allocate(headerBytes + (long) numElements * Short.BYTES, 4);
        for (long i = 0; i < headerBytes; i++) {
            nativeSegment.set(ValueLayout.JAVA_BYTE, i, (byte) 0);
        }
        for (int i = 0; i < numElements; i++) {
            short f16Bits = Float.floatToFloat16(source.getFloat(i));
            nativeSegment.set(ValueLayout.JAVA_SHORT_UNALIGNED, headerBytes + (long) i * Short.BYTES, f16Bits);
        }
        return FP16TornadoTensor.fromTornadoMemorySegment(nativeSegment);
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
     * Copies a single {@code rowSize}-element row (selected by {@code rowIndex}) out of a -- possibly
     * very large -- embedding-table tensor directly into {@code dest}, converting each element to float
     * on the fly.
     *
     * <p>Some tensors (e.g. Gemma4's {@code per_layer_token_embd}, with ~2.35 billion elements) exceed
     * {@link Integer#MAX_VALUE} elements/bytes, which would overflow the int-based
     * {@link FloatTensor#numberOfElements} / {@link GGMLType#byteSizeFor} used to wrap a tensor entry in
     * a {@link FloatTensor}. Such tensors are kept as raw {@link GGMLTensorEntry}s and addressed here with
     * {@code long} byte offsets instead -- since only single-row (embedding lookup) access is needed.</p>
     */
    public static void copyEmbeddingRow(GGMLTensorEntry entry, long rowIndex, int rowSize, FloatTensor dest, int destOffset) {
        GGMLType type = entry.ggmlType();
        if (type.getBlockSize() != 1) {
            throw new UnsupportedOperationException("copyEmbeddingRow only supports unblocked (per-element) types, got " + type);
        }
        MemorySegment segment = entry.memorySegment();
        long elementBytes = type.getTypeSize();
        long rowByteOffset = rowIndex * rowSize * elementBytes;
        for (int i = 0; i < rowSize; i++) {
            long byteOffset = rowByteOffset + (long) i * elementBytes;
            float value = switch (type) {
                case F32 -> segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, byteOffset);
                case F16 -> Float.float16ToFloat(segment.get(ValueLayout.JAVA_SHORT_UNALIGNED, byteOffset));
                case BF16 -> Float.intBitsToFloat(((int) segment.get(ValueLayout.JAVA_SHORT_UNALIGNED, byteOffset)) << 16);
                default -> throw new UnsupportedOperationException("copyEmbeddingRow: unsupported type " + type);
            };
            dest.setFloat(destOffset + i, value);
        }
    }

    /**
     * Like {@link #copyEmbeddingRow(GGMLTensorEntry, long, int, FloatTensor, int)}, but writes into a
     * TornadoVM {@link FloatArray} (optionally scaling each element) -- used by the GPU path to gather
     * a per-token embedding row directly into a buffer ready for transfer to the device.
     */
    public static void copyEmbeddingRowToFloatArray(GGMLTensorEntry entry, long rowIndex, int rowSize, FloatArray dest, float scale) {
        GGMLType type = entry.ggmlType();
        if (type.getBlockSize() != 1) {
            throw new UnsupportedOperationException("copyEmbeddingRowToFloatArray only supports unblocked (per-element) types, got " + type);
        }
        MemorySegment segment = entry.memorySegment();
        long elementBytes = type.getTypeSize();
        long rowByteOffset = rowIndex * rowSize * elementBytes;
        for (int i = 0; i < rowSize; i++) {
            long byteOffset = rowByteOffset + (long) i * elementBytes;
            float value = switch (type) {
                case F32 -> segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, byteOffset);
                case F16 -> Float.float16ToFloat(segment.get(ValueLayout.JAVA_SHORT_UNALIGNED, byteOffset));
                case BF16 -> Float.intBitsToFloat(((int) segment.get(ValueLayout.JAVA_SHORT_UNALIGNED, byteOffset)) << 16);
                default -> throw new UnsupportedOperationException("copyEmbeddingRowToFloatArray: unsupported type " + type);
            };
            dest.set(i, value * scale);
        }
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
