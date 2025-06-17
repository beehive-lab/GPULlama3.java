package com.example.aot;

import com.example.auxiliary.Timer;
import com.example.core.model.GGUF;
import com.example.core.model.tensor.GGMLTensorEntry;
import com.example.loader.weights.LlamaModelLoader;
import com.example.model.Model;
import com.example.Options;
import com.example.model.format.LlamaChatFormat;
import com.example.model.llama.Llama;
import com.example.loader.weights.ModelLoader;
import com.example.loader.weights.Weights;
import com.example.tokenizer.impl.LlamaTokenizer;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.Objects;

/**
 * Support for AOT preloading of GGUF metadata with GraalVM's Native Image.
 *
 * <p>
 * To preload a model at build time, pass {@code -Dllama.PreloadGGUF=/path/to/model.gguf}
 * to the native-image builder command. At runtime, the preloaded model will be used
 * iff the specified and preloaded file names (base name) match.
 */
public final class AOT {
    AOT.PartialModel preLoaded = AOT.PRELOADED_GGUF;

    static LlamaModelLoader modelLoader;


    record PartialModel(String modelFileName, Llama model, long tensorDataOffset, Map<String, GGUF.GGUFTensorInfo> tensorInfos) {}

    private static final PartialModel PRELOADED_GGUF = preLoadGGUF(System.getProperty("llama.PreloadGGUF"));

    private static PartialModel preLoadGGUF(String modelPath) {
        if (modelPath == null || modelPath.isEmpty()) {
            return null;
        }
        try {
            Path path = Path.of(modelPath);
            if (!Files.exists(path) || !Files.isRegularFile(path)) {
                throw new IllegalArgumentException("Cannot pre-load model: " + path);
            }
            GGUF gguf = GGUF.loadModel(path);
            try (FileChannel fileChannel = FileChannel.open(path, StandardOpenOption.READ)) {
                modelLoader = new LlamaModelLoader(fileChannel, gguf, Options.DEFAULT_MAX_TOKENS, false);
                return new PartialModel(
                        path.getFileName().toString(),
                        modelLoader.loadModel(), // TODO: needs proper handling for AOT
                        gguf.getTensorDataOffset(),
                        gguf.getTensorInfos()
                );
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Tries to reuse a compatible AOT preloaded model.
     * The file name (base name) must match with the preloaded file name.
     * No checksum/hash is checked for performance reasons.
     */
    public static Model tryUsePreLoaded(Path modelPath, int contextLength) throws IOException {
        AOT.PartialModel preLoaded = AOT.PRELOADED_GGUF;
        if (preLoaded == null) {
            return null; // no pre-loaded model stored
        }
        String optionsModel = modelPath.getFileName().toString();
        String preLoadedModel = preLoaded.modelFileName();
        if (!Objects.equals(optionsModel, preLoadedModel)) {
            // Preloaded and specified model file names didn't match.
            return null;
        }
        Llama baseModel = preLoaded.model();
        try (var timer = Timer.log("Load tensors from pre-loaded model");
                var fileChannel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            // Load only the tensors (mmap slices).
            Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, preLoaded.tensorDataOffset(), preLoaded.tensorInfos());
            Weights weights = modelLoader.loadWeights(tensorEntries, baseModel.configuration());
            return new Llama(baseModel.configuration().withContextLength(contextLength), baseModel.tokenizer(), weights, new LlamaChatFormat((LlamaTokenizer) baseModel.tokenizer()));
        }
    }
}

