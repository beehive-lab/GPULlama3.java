package com.example.model;

import com.example.core.model.GGUF;
import com.example.model.llama.Llama;
import com.example.model.mistral.Mistral;

import java.nio.channels.FileChannel;

public enum ModelType {
    LLAMA_3 {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights) {
            return Llama.loadModel(fileChannel, gguf, contextLength, loadWeights);
        }
    },

    MISTRAL {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights) {
            return Mistral.loadModel(fileChannel, gguf, contextLength, loadWeights);
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
