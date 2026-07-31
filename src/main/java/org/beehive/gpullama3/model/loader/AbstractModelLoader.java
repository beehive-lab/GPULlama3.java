package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.tensor.GGUF;
import org.beehive.gpullama3.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.auxiliary.Pair;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.tokenizer.Tokenizer;
import org.beehive.gpullama3.tokenizer.Vocabulary;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.Map;

/**
 * Abstract base class for model loaders using Template Method pattern. Provides common loading flow with extension points for model-specific logic.
 *
 * @param <M> The specific Model type to load
 * @param <C> The specific Configuration type for the model
 */
public abstract class AbstractModelLoader<M extends Model, C extends Configuration> {

    protected final FileChannel fileChannel;
    protected final GGUF gguf;
    protected final int contextLength;
    protected final boolean useTornadovm;

    protected Vocabulary vocabulary;

    protected AbstractModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
        this.fileChannel = fileChannel;
        this.gguf = gguf;
        this.contextLength = contextLength;
        this.useTornadovm = useTornadovm;
    }

    protected String getModelQuantization(Map<String, Object> metadata) {
        int modelQuantizationAsInt = (int) metadata.get("general.file_type");
        return switch (modelQuantizationAsInt) {
            case 1 -> "FP16";
            case 32 -> "FP16"; // MOSTLY_BF16 (treated like FP16 for activation buffers)
            case 7 -> "Q8_0";
            case 14, 15 -> "Q8_0"; // Q4_K_S, Q4_K_M (K-quants use Q8_0 activations)
            case 16, 17 -> "Q8_0"; // Q5_K_S, Q5_K_M
            case 18 -> "Q8_0";     // Q6_K
            default -> throw new UnsupportedOperationException("Unsupported quantization format: " + modelQuantizationAsInt + " (as int).");
        };
    }

    /**
     * Returns the effective GPU weight type for TornadoVM execution.
     * K-quant types (Q4_K, Q5_K, Q6_K) are dequantized to Q8_0 at load time.
     */
    protected static GGMLType effectiveGpuWeightType(GGMLType ggmlType) {
        return switch (ggmlType) {
            case F16, F32, Q8_0 -> ggmlType;
            case BF16 -> GGMLType.F16; // widened to FP16 at load time; see ModelLoader#loadTornadoTensor
            case Q4_K, Q5_K, Q6_K -> GGMLType.Q8_0;
            default -> ggmlType;
        };
    }

    private static String fileTypeName(int fileType) {
        return switch (fileType) {
            case 0 -> "F32";
            case 1 -> "F16";
            case 32 -> "BF16";
            case 7 -> "Q8_0";
            case 14 -> "Q4_K_S";
            case 15 -> "Q4_K_M";
            case 16 -> "Q5_K_S";
            case 17 -> "Q5_K_M";
            case 18 -> "Q6_K";
            default -> "type_" + fileType;
        };
    }

    /**
     * Template method that defines the model loading workflow. Subclasses should not override this method.
     *
     * @return The loaded model instance
     */
    public final M loadModel() {
        try {
            Map<String, Object> metadata = gguf.getMetadata();

            // Step 1: Load vocabulary
            this.vocabulary = loadVocabulary(metadata);

            // Step 2: Create tokenizer
            Tokenizer tokenizer = createTokenizer(metadata, vocabulary);

            // Step 3: Create configuration
            C config = createConfiguration(metadata);

            // Step 4: Load tensor entries
            Map<String, GGMLTensorEntry> tensorEntries;
            if (useTornadovm) {
                tensorEntries = GGUF.loadTensorsTornado(fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());
            } else {
                tensorEntries = GGUF.loadTensorsStandard(fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());
            }

            // Step 4: Load weights
            Weights weights = loadWeights(tensorEntries, config);

            // Step 5: Create and return model instance
            return createModel(config, tokenizer, weights);

        } catch (IOException e) {
            throw new ModelLoadException("Failed to load model", e);
        }
    }

    /**
     * Load the vocabulary from GGUF metadata. Model-specific implementations should override this method.
     *
     * @param metadata The GGUF metadata map
     * @return The loaded Vocabulary
     */
    protected abstract Vocabulary loadVocabulary(Map<String, Object> metadata);

    /**
     * Create a tokenizer instance for this model.
     *
     * @param metadata   The GGUF metadata map
     * @param vocabulary The loaded vocabulary
     * @return The tokenizer instance
     */
    protected abstract Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary);

    /**
     * Create a configuration instance from GGUF metadata.
     *
     * @param metadata The GGUF metadata map
     * @return The configuration instance
     */
    protected abstract C createConfiguration(Map<String, Object> metadata);

    /**
     * Load model weights from tensor entries. Default implementation handles common weight loading logic.
     *
     * @param tensorEntries Map of tensor names to tensor entries
     * @param config        The model configuration
     * @return The loaded weights
     */
    public Weights loadWeights(Map<String, GGMLTensorEntry> tensorEntries, C config) {
        // Precompute RoPE frequencies
        Pair<float[], float[]> ropeFreqs = precomputeRopeFrequencies(config);

        // Get token embeddings and output weights
        GGMLTensorEntry tokenEmbeddings = getTokenEmbeddings(tensorEntries);
        GGMLTensorEntry outputWeight = getOutputWeight(tensorEntries, tokenEmbeddings);

        // Delegate to specific implementation
        if (useTornadovm) {
            GGMLType gpuType = effectiveGpuWeightType(outputWeight.ggmlType());
            if (TornadoVMMasterPlan.ENABLE_TORNADOVM_INIT_TIME) {
                int fileType = (int) gguf.getMetadata().get("general.file_type");
                System.out.println("Loading model weights in TornadoVM format (" + fileTypeName(fileType) + " -> " + gpuType + ")");
            }
            return createTornadoVMWeights(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
        } else {
            return createStandardWeights(tensorEntries, config, ropeFreqs, tokenEmbeddings, outputWeight);
        }
    }

    /**
     * Create the final model instance.
     *
     * @param config    The model configuration
     * @param tokenizer The tokenizer
     * @param weights   The loaded weights
     * @return The model instance
     */
    protected abstract M createModel(C config, Tokenizer tokenizer, Weights weights);

    /**
     * Precompute RoPE frequencies for this model. Default implementation can be overridden for custom RoPE configurations.
     */
    protected abstract Pair<float[], float[]> precomputeRopeFrequencies(C config);

    /**
     * Get token embeddings tensor entry. Default implementation can be overridden for different tensor naming.
     */
    protected GGMLTensorEntry getTokenEmbeddings(Map<String, GGMLTensorEntry> tensorEntries) {
        return tensorEntries.get("token_embd.weight");
    }

    /**
     * Get output weight tensor entry. Default implementation falls back to token embeddings if output.weight not found.
     */
    protected GGMLTensorEntry getOutputWeight(Map<String, GGMLTensorEntry> tensorEntries, GGMLTensorEntry tokenEmbeddings) {
        return tensorEntries.getOrDefault("output.weight", tokenEmbeddings);
    }

    /**
     * Create standard (CPU) weights.
     */
    protected abstract Weights createStandardWeights(Map<String, GGMLTensorEntry> tensorEntries, C config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings, GGMLTensorEntry outputWeight);

    /**
     * Create TornadoVM (GPU) weights.
     */
    protected abstract Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries, C config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings, GGMLTensorEntry outputWeight);
}