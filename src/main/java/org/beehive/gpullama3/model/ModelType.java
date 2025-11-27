package org.beehive.gpullama3.model;

import org.beehive.gpullama3.tensor.GGUF;
import org.beehive.gpullama3.model.loader.LlamaModelLoader;
import org.beehive.gpullama3.model.loader.MistralModelLoader;
import org.beehive.gpullama3.model.loader.Phi3ModelLoader;
import org.beehive.gpullama3.model.loader.Qwen2ModelLoader;
import org.beehive.gpullama3.model.loader.Qwen3ModelLoader;

import java.nio.channels.FileChannel;

/**
 * Enumerates the different types of models supported by GPULlama3.java. This enum helps in categorizing and handling model-specific
 * logic based on the type of model being used.
 *
 * <p><b>Usage:</b> Use {@code ModelType} to specify or retrieve the type of
 * large language model (LLM), such as Llama or Qwen3. This ensures clean and structured handling of model behaviors and configurations by
 * dispatching calls to the appropriate model loader for each
 * model type.</p>
 *
 * <p>Each enum value represents a distinct model type, which might be used for
 * conditional logic, initialization, or resource allocation within GPULlama3.java.</p>
 */
public enum ModelType {
    LLAMA_3 {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
            return new LlamaModelLoader(fileChannel, gguf, contextLength, useTornadovm).loadModel();
        }
    },

    MISTRAL {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
            return new MistralModelLoader(fileChannel, gguf, contextLength, useTornadovm).loadModel();
        }
    },

    QWEN_2 {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
            return new Qwen2ModelLoader(fileChannel, gguf, contextLength, useTornadovm).loadModel();
        }
    },

    QWEN_3 {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
            return new Qwen3ModelLoader(fileChannel, gguf, contextLength, useTornadovm).loadModel();
        }
    },

    DEEPSEEK_R1_DISTILL_QWEN {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
            return new Qwen2ModelLoader(fileChannel, gguf, contextLength, useTornadovm).loadModel();
        }
    },

    PHI_3 {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
            return new Phi3ModelLoader(fileChannel, gguf, contextLength, useTornadovm).loadModel();
        }
    },

    UNKNOWN {
        @Override
        public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
            throw new UnsupportedOperationException("Cannot load unknown model type");
        }
    };

    // Abstract method that each enum constant must implement
    public abstract Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm);

    public boolean isDeepSeekR1() {
        return this == DEEPSEEK_R1_DISTILL_QWEN;
    }
}
