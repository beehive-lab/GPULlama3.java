package org.beehive.gpullama3.model.loader;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.Map;
import java.util.function.IntFunction;
import org.beehive.gpullama3.Options;
import org.beehive.gpullama3.auxiliary.RunMetrics;
import org.beehive.gpullama3.backend.tornado.tensor.FP16TornadoTensor;
import org.beehive.gpullama3.backend.tornado.tensor.FP32TornadoTensor;
import org.beehive.gpullama3.backend.tornado.tensor.Q8_0TornadoTensor;
import org.beehive.gpullama3.backend.tornado.tensor.TornadoTensor;
import org.beehive.gpullama3.backend.tornado.tensor.TornadoTensorLoader;
import org.beehive.gpullama3.format.*;
import org.beehive.gpullama3.format.GGMLType;
import org.beehive.gpullama3.format.GGUF;
import org.beehive.gpullama3.format.TensorDescriptors;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.ModelType;
import org.beehive.gpullama3.model.provider.ModelProvider;
import org.beehive.gpullama3.model.provider.ModelProviders;
import org.beehive.gpullama3.runtime.backend.BackendId;
import org.beehive.gpullama3.runtime.tensor.ExecutionTarget;
import org.beehive.gpullama3.runtime.tensor.TensorDescriptor;
import org.beehive.gpullama3.tensor.standard.*;

public abstract class ModelLoader {

    /**
     * Rule 16: loading is library code. This one is a genuine diagnostic rather than progress — an
     * F32 tensor reaching the FP16 path is a case this loader does not handle — so it is a warning,
     * not an info line.
     */
    private static final System.Logger LOGGER = System.getLogger(ModelLoader.class.getName());

    protected FileChannel fileChannel;
    protected GGUF gguf;
    protected int contextLength;
    protected boolean loadWeights;
    protected boolean useTornadovm;

    public ModelLoader(
            FileChannel fileChannel,
            GGUF gguf,
            int contextLength,
            boolean loadWeights,
            boolean useTornadovm) {
        this.fileChannel = fileChannel;
        this.gguf = gguf;
        this.contextLength = contextLength;
        this.loadWeights = loadWeights;
        this.useTornadovm = useTornadovm;
    }

    private static ModelType detectModelType(Map<String, Object> metadata) {
        // Architecture key is authoritative (set by llama.cpp conversion) and doesn't
        // depend on how the model happens to be named, unlike general.name below.
        if ("qwen2moe".equals(metadata.get("general.architecture"))) {
            return ModelType.QWEN_2_MOE;
        }

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
        if ("gemma4".equals(metadata.get("general.architecture"))
                || metadata.containsKey("gemma4.block_count")) {
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
     * @throws IOException if the model fails to load
     * @throws IllegalStateException if AOT loading is enabled but the preloaded model is
     *     unavailable
     */
    public static Model loadModel(Options options) throws IOException {
        return loadModel(options.modelPath(), options.maxTokens(), true, options.useTornadovm());
    }

    /**
     * Whether discovered providers do the loading. Defaults to true; {@code
     * -Dllama.providers=false} selects the {@code ModelType} dispatch this replaced.
     *
     * <p>The fallback exists for one release, so that a model which loads differently through a
     * provider has a way to be compared rather than a way to be stuck.
     */
    private static boolean providersEnabled() {
        return !"false".equalsIgnoreCase(System.getProperty("llama.providers", "true"));
    }

    /** For compatibility with langchain4j and quarkus. */
    public static Model loadModel(
            Path ggufPath, int contextLength, boolean loadWeights, boolean useTornadovm)
            throws IOException {
        long start = System.nanoTime();
        ModelSource source = ModelSource.ofFile(ggufPath);
        Model model =
                providersEnabled()
                        ? loadThroughProvider(source, contextLength, useTornadovm)
                        : loadThroughModelType(source, contextLength, useTornadovm);
        RunMetrics.setLoadDuration(System.nanoTime() - start);
        return model;
    }

    /**
     * The discovered provider loads it. Recognition happens once, in the provider that claims the
     * source, and the architecture identity it chooses is the one everything downstream uses.
     */
    private static Model loadThroughProvider(
            ModelSource source, int contextLength, boolean useTornadovm) throws IOException {
        ModelProvider provider = ModelProviders.select(source);
        return provider.load(source, providerBackend(useTornadovm), contextLength);
    }

    /**
     * The identity a provider and its diagnostics see.
     *
     * <p>Was a hardcoded {@code BackendId.CUDA} for every non-CPU load — wrong on Metal, PTX and
     * OpenCL alike, and a defect this method's callers ({@code FamilyProviders} and {@code
     * Gemma4Provider}, both only ever comparing the value to {@code BackendId.CPU}) could not
     * observe. When a real accelerator is resolved, this reports it truthfully instead.
     *
     * <p><b>The no-accelerator corner case keeps the old placeholder.</b> If {@code
     * useTornadovm=true} and nothing resolves an accelerator at all — {@code
     * TornadoDevices.current()} itself falls back to the {@code BackendId.CPU} placeholder —
     * reporting that placeholder here would flip every provider's {@code
     * !BackendId.CPU.equals(backend)} check to {@code false} and silently take the CPU path for a
     * caller that asked for the GPU one: a new silent fallback, introduced by fixing an old
     * mislabel. Keeping {@code BackendId.CUDA} here in that one case is exactly the pre-existing
     * behaviour {@code useTornadovm}'s boolean callers already have — unchanged, not newly correct,
     * because correctness for that case is {@code LocalModels}'s explicit-accelerator validation
     * territory, not this compatibility path's.
     *
     * <p>Package-visible for {@code ModelLoaderProviderBackendTest} and its accelerator sibling.
     */
    static BackendId providerBackend(boolean useTornadovm) {
        if (!useTornadovm) {
            return BackendId.CPU;
        }
        BackendId resolved =
                org.beehive.gpullama3.backend.tornado.device.TornadoDevices.current()
                        .id()
                        .backend();
        return BackendId.CPU.equals(resolved) ? BackendId.CUDA : resolved;
    }

    /** The dispatch providers replaced, kept selectable for one release. */
    private static Model loadThroughModelType(
            ModelSource source, int contextLength, boolean useTornadovm) {
        ModelType modelType = detectModelType(source.metadata());
        return modelType.loadModel(
                source.gguf().getFileChannel(), source.gguf(), contextLength, useTornadovm);
    }

    /**
     * Loads a host tensor: describe, then materialize.
     *
     * <p>The descriptor is metadata — it allocates nothing and copies nothing — but it is where the
     * element count is validated, so a tensor too large for an int-indexed array fails naming
     * itself rather than wrapping around into a smaller one.
     */
    /**
     * The weight footprint of a model file, from its descriptors alone.
     *
     * <p>Reads tensor metadata, never tensor data — this is what lets a preflight answer "will it
     * fit" without a multi-gigabyte upload.
     *
     * <p>Lives here because Rule 4 permits the loaders to name GGUF and forbids it to the runtime
     * and the backends. What leaves this method is a neutral {@link
     * org.beehive.gpullama3.runtime.memory.WeightFootprint}.
     *
     * <p>The per-layer / global split follows the GGUF convention that a layer's tensors are named
     * {@code blk.N.*}. That is the same convention every loader in this package already relies on
     * to find them, so the two cannot disagree about what a layer owns.
     *
     * <p><b>Sized as the accelerator materializes it, not as the file stores it.</b> This is a
     * device memory plan, and {@link #loadTornadoTensor} materializes a representation the device
     * has no kernel for as Q8_0 — so a Q4_K file's weights occupy roughly twice their file size
     * once loaded. Measuring the file type here under-predicted exactly those models, in the one
     * direction a preflight must never be wrong: {@code MemoryPlanAccuracyAccelTest} exists because
     * a prediction below what is actually allocated admits a load that then dies part-allocated.
     * Found on the real Devstral fixture (Metal parity task 12): a Q4_K 24B predicted 13.5 GiB and
     * died materializing Q8_0 — {@code OutOfMemoryError: Cannot reserve. direct buffer memory at
     * TornadoTensorLoader.dequantizeToQ8_0}. The materialized type comes from {@code
     * DataTypeMapping.materializedType}, the same function {@link #loadTornadoTensor}'s descriptor
     * uses, so the prediction and the allocation cannot disagree about what a tensor becomes. F16,
     * Q8_0 and F32 materialize as themselves, so every tuple measured on CUDA is predicted
     * byte-for-byte as before.
     */
    public static org.beehive.gpullama3.runtime.memory.WeightFootprint weightFootprint(
            Path ggufPath) throws IOException {
        GGUF gguf = GGUF.loadGGUFMetadata(ggufPath);
        long perLayer = 0;
        long global = 0;
        int perLayerTensors = 0;
        int globalTensors = 0;
        for (GGUF.GGUFTensorInfo info : gguf.getTensorInfos().values()) {
            if (info.name().equals("rope_freqs.weight")) {
                continue; // not materialized — every loader skips it
            }
            long elements = 1L;
            for (int d : info.dimensions()) {
                elements *= d;
            }
            org.beehive.gpullama3.runtime.tensor.DataType materialized =
                    org.beehive.gpullama3.format.DataTypeMapping.materializedType(
                            info.ggmlType(),
                            org.beehive.gpullama3.runtime.tensor.ExecutionTarget.GPU);
            long bytes =
                    org.beehive.gpullama3.format.TensorDescriptors.layoutOf(materialized)
                            .byteSize(elements);
            if (info.name().startsWith("blk.")) {
                perLayer += bytes;
                perLayerTensors++;
            } else {
                global += bytes;
                globalTensors++;
            }
        }
        return new org.beehive.gpullama3.runtime.memory.WeightFootprint(
                perLayer, perLayerTensors, global, globalTensors);
    }

    public static FloatTensor loadTensor(GGMLTensorEntry entry) {
        TensorDescriptor descriptor = TensorDescriptors.describeSource(entry);
        int size = descriptor.shape().elementCountAsInt(descriptor.name());
        MemorySegment data = entry.memorySegment();
        return switch (entry.ggmlType()) {
            case F32 -> new FP32FloatTensor(size, data);
            case Q8_0 -> new Q8_0FloatTensor(size, data);
            case Q4_0 -> new Q4_0FloatTensor(size, data);
            case Q4_K -> new Q4_KFloatTensor(size, data);
            case Q5_K -> new Q5_KFloatTensor(size, data);
            case Q6_K -> new Q6_KFloatTensor(size, data);
            case F16 -> new FP16FloatTensor(size, data);
            case BF16 -> new BF16FloatTensor(size, data);
            default ->
                    throw new UnsupportedOperationException(
                            "Quantization format " + entry.ggmlType());
        };
    }

    /** Dispatcher method for loading a standard tensor array based on type. Used in CPU-path. */
    public static FloatTensor[] loadArrayOfTensors(
            int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatTensor[] array = new FloatTensor[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadTensor(getTensorEntry.apply(i));
        }
        return array;
    }

    // Helper methods

    public static FloatBuffer toFloatBuffer(GGMLTensorEntry tensorEntry) {
        GGMLType ggmlType = tensorEntry.ggmlType();
        return switch (ggmlType) {
            case F32 ->
                    tensorEntry
                            .memorySegment()
                            .asByteBuffer()
                            .order(ByteOrder.LITTLE_ENDIAN)
                            .asFloatBuffer();
            default -> throw new UnsupportedOperationException("Conversion to " + ggmlType);
        };
    }

    /** Loads a GGUF tensor as this backend's device tensor. */
    /** Loads a tensor for the device <b>retaining Q4_K</b> rather than materializing it as Q8_0. */
    public static TornadoTensor loadTornadoTensorRetainingQ4_K(GGMLTensorEntry entry) {
        if (entry.ggmlType() == GGMLType.Q4_K) {
            return org.beehive.gpullama3.backend.tornado.tensor.Q4_KTornadoTensor
                    .fromTornadoMemorySegment(entry.memorySegment());
        }
        if (entry.ggmlType() == GGMLType.Q6_K) {
            return org.beehive.gpullama3.backend.tornado.tensor.Q6_KTornadoTensor
                    .fromTornadoMemorySegment(entry.memorySegment());
        }
        return loadTornadoTensor(entry);
    }

    public static TornadoTensor loadTornadoTensor(GGMLTensorEntry entry) {
        // Describe first: the descriptor states what this tensor becomes on the device — including
        // the Q8_0 materialization for representations with no kernel — and validates the element
        // count before any storage is touched. It holds no data, so nothing is copied for it.
        TensorDescriptor descriptor = TensorDescriptors.describe(entry, ExecutionTarget.GPU);
        descriptor.shape().elementCountAsInt(descriptor.name());
        return switch (entry.ggmlType()) {
            case F32 -> FP32TornadoTensor.fromTornadoMemorySegment(entry.memorySegment());
            case F16 -> FP16TornadoTensor.fromTornadoMemorySegment(entry.memorySegment());
            case BF16 -> TornadoTensorLoader.convertBF16ToFP16(rawTensorData(entry));
            case Q8_0 -> Q8_0TornadoTensor.fromTornadoMemorySegment(entry.memorySegment());
                // A representation the device has no kernel for is materialized as Q8_0 at load
                // The conversion reads through the CPU tensor for that format, which
                // already knows how to decode it.
            case Q4_0, Q4_K, Q5_K, Q6_K ->
                    TornadoTensorLoader.dequantizeToQ8_0(rawTensorData(entry));
            default ->
                    throw new UnsupportedOperationException(
                            "Quantization format " + entry.ggmlType());
        };
    }

    /**
     * The entry's tensor data, read as a CPU tensor, past the device array header.
     *
     * <p>An entry loaded for the device is prefixed with TornadoVM's array header; how wide that is
     * is the backend's knowledge, so the slice comes from there.
     */
    private static FloatTensor rawTensorData(GGMLTensorEntry entry) {
        GGMLTensorEntry dataEntry =
                new GGMLTensorEntry(
                        entry.mappedFile(),
                        entry.name(),
                        entry.ggmlType(),
                        entry.shape(),
                        TornadoTensorLoader.withoutArrayHeader(entry.memorySegment()));
        return loadTensor(dataEntry);
    }

    /** Dispatcher for an array of device tensors. Used in the GPU path. */
    /**
     * {@link #loadArrayOfTornadoTensors} that retains Q4_K — see {@link
     * #loadTornadoTensorRetainingQ4_K}. For the per-layer weights of a family that has Q4_K
     * kernels.
     */
    public static TornadoTensor[] loadArrayOfTornadoTensorsRetainingQ4_K(
            int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        TornadoTensor[] array = new TornadoTensor[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadTornadoTensorRetainingQ4_K(getTensorEntry.apply(i));
        }
        return array;
    }

    public static TornadoTensor[] loadArrayOfTornadoTensors(
            int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        TornadoTensor[] array = new TornadoTensor[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadTornadoTensor(getTensorEntry.apply(i));
        }
        return array;
    }
}
