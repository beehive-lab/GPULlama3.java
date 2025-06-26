package com.example.model;

import com.example.core.model.GGUF;
import com.example.model.loader.LlamaModelLoader;
import com.example.model.loader.MistralModelLoader;
import com.example.model.loader.Qwen3ModelLoader;

import java.nio.channels.FileChannel;

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

    UNKNOWN {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights) {
            throw new UnsupportedOperationException("Cannot load unknown model type");
        }
    };

    // Abstract method that each enum constant must implement
    public abstract Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights);
}
