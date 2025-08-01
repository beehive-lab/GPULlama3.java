package com.example.model;

import com.example.core.model.GGUF;
import com.example.model.loader.LlamaModelLoader;
import com.example.model.loader.MistralModelLoader;
import com.example.model.loader.Phi3ModelLoader;
import com.example.model.loader.Qwen3ModelLoader;

import java.nio.channels.FileChannel;

/**
 * Enumerates the different types of models supported by GPULlama3.java.
 * This enum helps in categorizing and handling model-specific logic
 * based on the type of model being used.
 *
 * <p><b>Usage:</b> Use {@code ModelType} to specify or retrieve the type of
 * large language model (LLM), such as Llama or Qwen3. This ensures clean and
 * structured handling of model behaviors and configurations by dispatching
 * calls to the appropriate model loader for each specific model type.</p>
 *
 * <p>Each enum value represents a distinct model type, which might be used for
 * conditional logic, initialization, or resource allocation within GPULlama3.java.</p>
 */
public enum ModelType {
    LLAMA_3 {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights) {
            return new LlamaModelLoader(fileChannel, gguf, contextLength, loadWeights).loadModel();
        }
    },

    MISTRAL {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights) {
            return new MistralModelLoader(fileChannel, gguf, contextLength, loadWeights).loadModel();
        }
    },

    QWEN_3 {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights) {
            return new Qwen3ModelLoader(fileChannel, gguf, contextLength, loadWeights).loadModel();
        }
    },

    PHI_3 {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights) {
            return new Phi3ModelLoader(fileChannel, gguf, contextLength, loadWeights).loadModel();
        }
    },

    UNKNOWN {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights) {
            throw new UnsupportedOperationException("Cannot load unknown model type");
        }
    };

    // Abstract method that each enum constant must implement
    public abstract Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights);
}
