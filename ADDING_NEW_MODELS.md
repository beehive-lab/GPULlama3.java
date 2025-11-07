# Guide: Adding New Models to GPULlama3.java

This comprehensive guide explains how to add support for new transformer-based language models to GPULlama3.java.

**Last Updated**: November 1, 2025
**Example Model**: Google Gemma 3
**Difficulty**: Advanced (requires understanding of transformer architectures)

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Architecture Analysis](#architecture-analysis)
3. [Step-by-Step Implementation](#step-by-step-implementation)
4. [Testing and Debugging](#testing-and-debugging)
5. [Common Patterns](#common-patterns)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Knowledge Requirements
- ✅ Java programming (records, interfaces, generics)
- ✅ Transformer architecture basics (attention, FFN, normalization)
- ✅ Model formats (GGUF, safetensors)
- ✅ Tokenization (BPE, SentencePiece, WordPiece)

### Tools Needed
- Java 21+ with preview features enabled
- Maven build system
- GGUF model files
- (Optional) TornadoVM for GPU support

### Existing Codebase Familiarity
Study these existing implementations:
1. **Simple**: Llama (standard transformer)
2. **With GQA**: Mistral (grouped-query attention)
3. **With Q/K Norm**: Qwen3 (query/key normalization)
4. **Complex**: Gemma3 (sandwich normalization)

---

## Architecture Analysis

### Step 1: Research the Model Architecture

#### 1.1 Identify Key Characteristics
Research and document:
- [ ] **Model family**: Llama-based, GPT-based, custom?
- [ ] **Architecture variants**: Standard, MoE, multimodal?
- [ ] **Normalization type**: LayerNorm, RMSNorm, custom?
- [ ] **Attention mechanism**: MHA, GQA, MQA?
- [ ] **Special features**: Rope, ALiBi, sliding window, etc.

#### 1.2 Find Reference Implementations
Look for:
- Official HuggingFace transformers code
- llama.cpp implementation (C++)
- GGML format documentation
- Academic papers or blog posts

**Example Resources**:
```bash
# llama.cpp docs
https://github.com/ggml-org/llama.cpp/tree/master/docs

# HuggingFace model card
https://huggingface.co/[organization]/[model-name]

# Architecture diagrams
https://github.com/[org]/[repo]/blob/main/architecture.md
```

#### 1.3 Create Architecture Comparison

Compare with existing models:

| Feature | Llama | Mistral | Qwen3 | Your Model |
|---------|-------|---------|-------|------------|
| Norm layers per block | 2 | 2 | 2 | ? |
| Attention type | MHA | GQA | GQA | ? |
| Q/K normalization | ❌ | ❌ | ✅ | ? |
| Embedding scaling | ❌ | ❌ | ❌ | ? |
| Special tokens | 3 | 5 | 4 | ? |
| Context window | 128K | 32K | 131K | ? |

---

## Step-by-Step Implementation

### Phase 1: Configuration and State (30-60 minutes)

#### Step 2.1: Create Model Configuration

**File**: `src/main/java/org/beehive/gpullama3/model/{modelname}/{ModelName}Configuration.java`

```java
package org.beehive.gpullama3.model.{modelname};

import org.beehive.gpullama3.model.Configuration;

public record {ModelName}Configuration(
    // Core dimensions
    int dim,                          // Model dimension
    int hiddenDim,                    // FFN hidden dimension
    int numberOfLayers,               // Number of transformer blocks
    int numberOfHeads,                // Number of attention heads
    int numberOfKeyValueHeads,        // For GQA (use numberOfHeads if MHA)

    // Vocabulary and context
    int vocabularySize,               // Size of vocabulary
    int contextLength,                // Maximum sequence length

    // Normalization
    float rmsNormEps,                 // RMSNorm epsilon (usually 1e-5 or 1e-6)

    // Position encoding
    float ropeTheta                   // RoPE theta (usually 10000 or 500000)

    // Add model-specific fields here:
    // - int numberOfHeadsKey (if using Q/K norm like Qwen3/Gemma3)
    // - int numberOfHeadsValue (if using Q/K norm)
    // - boolean sharedWeights (if embeddings/output weights shared)
    // - int slidingWindow (for Mistral)
) implements Configuration {

    @Override
    public int headSize() {
        return dim / numberOfHeads;
    }

    // Implement other Configuration interface methods
    @Override
    public int contextLength() { return contextLength; }

    @Override
    public int dim() { return dim; }

    // ... etc
}
```

**Decision Points**:
- ❓ Does the model use Grouped-Query Attention? → Add `numberOfKeyValueHeads`
- ❓ Does it have Q/K normalization? → Add `numberOfHeadsKey`, `numberOfHeadsValue`
- ❓ Are output and embedding weights shared? → Add `sharedWeights` boolean
- ❓ Does it use sliding window attention? → Add `slidingWindow` int

#### Step 2.2: Create Model State

**File**: `src/main/java/org/beehive/gpullama3/inference/state/{ModelName}State.java`

```java
package org.beehive.gpullama3.inference.state;

import org.beehive.gpullama3.model.Configuration;

public class {ModelName}State extends State {

    public {ModelName}State(Configuration config, int batchSize) {
        super(config, batchSize);

        // Add model-specific state buffers here if needed
        // Most models can use the base State class
    }
}
```

**When to extend**:
- Only create custom state if you need additional buffers
- Most models can use base `State` class directly

---

### Phase 2: Model Class (30 minutes)

#### Step 2.3: Create Main Model Class

**File**: `src/main/java/org/beehive/gpullama3/model/{modelname}/{ModelName}.java`

```java
package org.beehive.gpullama3.model.{modelname};

import org.beehive.gpullama3.inference.InferenceCore;
import org.beehive.gpullama3.inference.InferenceEngine;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.AbstractModel;
import org.beehive.gpullama3.model.ModelType;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

public class {ModelName} extends AbstractModel {

    private final {ModelName}Configuration configuration;

    public {ModelName}({ModelName}Configuration configuration,
                       Tokenizer tokenizer,
                       Weights weights,
                       ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat, null);
        this.configuration = configuration;
    }

    @Override
    public {ModelName}Configuration configuration() {
        return configuration;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.{MODEL_NAME};
    }

    @Override
    public State createNewState() {
        State state = new {ModelName}State(configuration(), -1);
        // Set initial token (usually BOS token)
        state.latestToken = tokenizer.getSpecialTokens().get("<bos>");
        return state;
    }

    @Override
    public State createNewState(int batchSize) {
        State state = new {ModelName}State(configuration(), batchSize);
        state.latestToken = tokenizer.getSpecialTokens().get("<bos>");
        return state;
    }

    @Override
    public boolean shouldAddBeginOfText() {
        return true;  // Most models use BOS token
    }

    @Override
    public void forward(State state, int token, int position) {
        if (plan == null) {
            // CPU inference path
            InferenceCore.forwardJava{ModelName}(this, state, token, position);
        } else {
            // GPU inference path
            InferenceCore.forwardTornadoVM(this, state, token, position, tornadoVMPlan());
        }
    }

    @Override
    public List<Integer> generateTokens(State state, int startPosition,
                                        List<Integer> promptTokens,
                                        Set<Integer> stopTokens, int maxTokens,
                                        Sampler sampler, boolean echo,
                                        IntConsumer onTokenGenerated) {
        // Choose generation method based on architecture similarity:
        // - Standard: InferenceEngine.generateTokensLlama()
        // - With Q/K norm: InferenceEngine.generateTokensQwen3()
        return InferenceEngine.generateTokensLlama(this, state, startPosition,
                                                   promptTokens, stopTokens,
                                                   maxTokens, sampler, echo,
                                                   onTokenGenerated);
    }

    @Override
    public List<Integer> generateTokensGPU(State state, int startPosition,
                                           List<Integer> promptTokens,
                                           Set<Integer> stopTokens, int maxTokens,
                                           Sampler sampler, boolean echo,
                                           IntConsumer onTokenGenerated,
                                           TornadoVMMasterPlan tornadoVMPlan) {
        return InferenceEngine.generateTokensGPULlama(this, state, startPosition,
                                                      promptTokens, stopTokens,
                                                      maxTokens, sampler, echo,
                                                      onTokenGenerated, tornadoVMPlan);
    }
}
```

---

### Phase 3: Tokenizer (1-2 hours)

#### Step 2.4: Implement Tokenizer

**File**: `src/main/java/org/beehive/gpullama3/tokenizer/impl/{ModelName}Tokenizer.java`

```java
package org.beehive.gpullama3.tokenizer.impl;

import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;
import java.util.*;

public class {ModelName}Tokenizer implements Tokenizer {

    private final Vocabulary vocabulary;
    private final Map<String, Integer> specialTokens;

    public {ModelName}Tokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        this.vocabulary = vocabulary;

        // Load special tokens from vocabulary
        this.specialTokens = new HashMap<>();

        // Scan vocabulary for special tokens
        for (int i = 0; i < vocabulary.size(); i++) {
            String token = vocabulary.get(i);
            if (isSpecialTokenPattern(token)) {
                specialTokens.put(token, i);
            }
        }
    }

    private boolean isSpecialTokenPattern(String token) {
        // Define what makes a token "special" for your model
        // Common patterns: <bos>, <eos>, <pad>, etc.
        return token.startsWith("<") && token.endsWith(">") &&
               !token.matches("<0x[0-9a-fA-F]{2}>") &&  // Not byte tokens
               !token.matches("<unused\\d+>");           // Not placeholders
    }

    @Override
    public List<Integer> encodeAsList(String text) {
        // Implement encoding logic
        // For most models, can delegate to existing tokenizer
        // or implement BPE/SentencePiece algorithm
        return List.of(); // TODO: Implement
    }

    @Override
    public String decode(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            // Handle special cases:
            // 1. Byte tokens (if model uses them)
            // 2. Special tokens (skip)
            // 3. Regular tokens

            String tokenString = vocabulary.get(token);

            if (isSpecialToken(token)) {
                continue; // Skip special tokens
            }

            // Handle model-specific replacements
            // Examples:
            // - SentencePiece: replace ▁ with space
            // - Some models: decode hex bytes

            sb.append(tokenString);
        }
        return sb.toString();
    }

    @Override
    public Map<String, Integer> getSpecialTokens() {
        return specialTokens;
    }

    @Override
    public boolean isSpecialToken(int tokenIndex) {
        return specialTokens.containsValue(tokenIndex);
    }

    @Override
    public boolean shouldDisplayToken(int token) {
        return !isSpecialToken(token);
    }
}
```

**Key Decisions**:
1. **Tokenization Algorithm**: BPE, SentencePiece, WordPiece?
2. **Byte-Level Encoding**: Does the model use raw bytes for first 256 tokens?
3. **Special Characters**: How are spaces represented? (▁ in SentencePiece)
4. **Metadata Keys**: Where are merges, vocab, and scores stored in GGUF?

---

### Phase 4: Chat Format (30 minutes)

#### Step 2.5: Create Chat Format

**File**: `src/main/java/org/beehive/gpullama3/model/format/{ModelName}ChatFormat.java`

```java
package org.beehive.gpullama3.model.format;

import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import java.util.*;

public class {ModelName}ChatFormat implements ChatFormat {

    private final int beginOfText;
    private final int endOfText;
    private final Set<Integer> stopTokens;
    private final Tokenizer tokenizer;

    public {ModelName}ChatFormat(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();

        // Load special tokens
        this.beginOfText = specialTokens.getOrDefault("<bos>", -1);
        this.endOfText = specialTokens.getOrDefault("<eos>", -1);

        // Define stop tokens
        this.stopTokens = new HashSet<>();
        if (endOfText != -1) {
            stopTokens.add(endOfText);
        }
        // Add model-specific stop tokens
    }

    @Override
    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();

        // Encode role header
        // Example: <|start_header_id|>user<|end_header_id|>

        return tokens;
    }

    @Override
    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = new ArrayList<>();

        // Encode complete message with header and content
        // Follow the model's specific chat template

        tokens.addAll(encodeHeader(message));
        tokens.addAll(tokenizer.encodeAsList(message.content().strip()));
        // Add end-of-message tokens

        return tokens;
    }

    @Override
    public int getBeginOfText() {
        return beginOfText;
    }

    @Override
    public Set<Integer> getStopTokens() {
        return stopTokens;
    }
}
```

**Chat Template Research**:
1. Check model card on HuggingFace for `tokenizer_config.json`
2. Look for `chat_template` field in GGUF metadata
3. Reference implementations in transformers library

**Common Templates**:
- **Llama 3**: `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{msg}<|eot_id|>`
- **Gemma**: `<bos><start_of_turn>user\n{msg}<end_of_turn>\n<start_of_turn>model\n`
- **ChatML**: `<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n`

---

### Phase 5: Weights (1-2 hours)

#### Step 2.6: Create Weight Classes

**CPU Weights** - `src/main/java/org/beehive/gpullama3/inference/weights/standard/{ModelName}StandardWeights.java`:

```java
package org.beehive.gpullama3.inference.weights.standard;

import org.beehive.gpullama3.core.model.GGMLType;
import org.beehive.gpullama3.core.model.tensor.FloatTensor;

public class {ModelName}StandardWeights extends StandardWeights {

    // Add model-specific weight fields
    // Example for sandwich normalization:
    // public final FloatTensor[] postAttentionNorm;
    // public final FloatTensor[] postFFNNorm;

    public {ModelName}StandardWeights(
            FloatTensor token_embedding_table,
            FloatTensor[] rms_att_weight,
            FloatTensor[] wq,
            FloatTensor[] wk,
            FloatTensor[] wv,
            FloatTensor[] wo,
            FloatTensor[] rms_ffn_weight,
            FloatTensor[] w1,
            FloatTensor[] w2,
            FloatTensor[] w3,
            FloatTensor rms_final_weight,
            FloatTensor freq_cis_real,
            FloatTensor freq_cis_imag,
            FloatTensor wcls,
            GGMLType ggmlType
            // Add custom parameters
    ) {
        super(token_embedding_table, rms_att_weight, wq, wk, wv, wo,
              rms_ffn_weight, w1, w2, w3, rms_final_weight,
              freq_cis_real, freq_cis_imag, wcls, ggmlType);

        // Initialize custom fields
    }
}
```

**GPU Weights** - `src/main/java/org/beehive/gpullama3/inference/weights/tornado/{ModelName}TornadoWeights.java`:

```java
package org.beehive.gpullama3.inference.weights.tornado;

import org.beehive.gpullama3.core.model.GGMLType;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

public class {ModelName}TornadoWeights extends FP16Weights {

    // Add model-specific weight arrays
    // Use FloatArray for GPU memory

    public {ModelName}TornadoWeights(/* parameters */) {
        super(/* base parameters */);
        // Initialize custom fields
    }
}
```

---

### Phase 6: Model Loader (2-3 hours)

#### Step 2.7: Create Model Loader

**File**: `src/main/java/org/beehive/gpullama3/model/loader/{ModelName}ModelLoader.java`

```java
package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.core.model.GGUF;
import org.beehive.gpullama3.core.model.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.core.types.Pair;
import org.beehive.gpullama3.inference.operation.RoPE;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.{modelname}.*;
import org.beehive.gpullama3.tokenizer.impl.Tokenizer;
import org.beehive.gpullama3.tokenizer.vocabulary.Vocabulary;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.Map;

public class {ModelName}ModelLoader extends ModelLoader {

    public {ModelName}ModelLoader(FileChannel fileChannel, GGUF gguf,
                                  int contextLength, boolean loadWeights,
                                  boolean useTornadoVM) {
        super(fileChannel, gguf, contextLength, loadWeights, useTornadoVM);
    }

    @Override
    public {ModelName} loadModel() {
        try {
            Map<String, Object> metadata = gguf.getMetadata();

            // 1. LOAD VOCABULARY
            Vocabulary vocabulary = Vocabulary.loadLlamaVocabulary(metadata);
            Tokenizer tokenizer = new {ModelName}Tokenizer(metadata, vocabulary);

            // 2. DETECT METADATA PREFIX
            // Try different prefixes: {model}. or llama. or mistral.
            String prefix;
            if (metadata.containsKey("{model}.embedding_length")) {
                prefix = "{model}.";
            } else if (metadata.containsKey("llama.embedding_length")) {
                prefix = "llama.";
            } else {
                throw new RuntimeException("Unknown architecture");
            }

            // 3. LOAD CONFIGURATION FROM METADATA
            int dim = (int) metadata.get(prefix + "embedding_length");
            int hiddenDim = (int) metadata.get(prefix + "feed_forward_length");
            int nLayers = (int) metadata.get(prefix + "block_count");
            int nHeads = (int) metadata.get(prefix + "attention.head_count");
            int nKVHeads = metadata.containsKey(prefix + "attention.head_count_kv")
                    ? (int) metadata.get(prefix + "attention.head_count_kv")
                    : nHeads;
            int ctxLength = (int) metadata.get(prefix + "context_length");
            float rmsNormEps = (float) metadata.getOrDefault(
                    prefix + "attention.layer_norm_rms_epsilon", 1e-6f);
            float ropeTheta = (float) metadata.getOrDefault(
                    prefix + "rope.freq_base", 10000f);

            // 4. LOAD TENSOR ENTRIES
            Map<String, GGMLTensorEntry> tensorEntries =
                    GGUF.loadTensors(fileChannel, gguf.getTensorDataOffset(),
                                   gguf.getTensorInfos());

            // 5. GET VOCAB SIZE FROM EMBEDDINGS TENSOR
            GGMLTensorEntry tokenEmbeddings = tensorEntries.get("token_embd.weight");
            int[] embShape = tokenEmbeddings.shape();
            int vocabSize = embShape.length > 1 ? embShape[1] : embShape[0];

            // 6. CREATE CONFIGURATION
            int actualContextLength = (contextLength < 0) ? ctxLength : contextLength;
            {ModelName}Configuration config = new {ModelName}Configuration(
                    dim, hiddenDim, nLayers, nHeads, nKVHeads,
                    vocabSize, actualContextLength, rmsNormEps, ropeTheta
                    // Add model-specific parameters
            );

            // 7. LOAD WEIGHTS
            Weights weights = null;
            if (loadWeights) {
                weights = loadWeights(tensorEntries, config);
            }

            // 8. RETURN MODEL
            return new {ModelName}(config, tokenizer, weights,
                                  ChatFormat.create(tokenizer, null));

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public Weights loadWeights(Map<String, GGMLTensorEntry> tensorEntries,
                               Configuration config) {
        // Precompute RoPE frequencies
        Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(
                config.contextLength(),
                config.headSize(),
                config.ropeTheta(),
                false, 0, 0, 0, 0
        );

        GGMLTensorEntry tokenEmbeddings = tensorEntries.get("token_embd.weight");
        GGMLTensorEntry outputWeight = tensorEntries.getOrDefault(
                "output.weight", tokenEmbeddings);

        if (useTornadovm) {
            return createTornadoVMWeights(tensorEntries, config, ropeFreqs,
                                         tokenEmbeddings, outputWeight);
        } else {
            return createStandardWeights(tensorEntries, config, ropeFreqs,
                                        tokenEmbeddings, outputWeight);
        }
    }

    @Override
    public Weights createStandardWeights(Map<String, GGMLTensorEntry> tensorEntries,
                                        Configuration config,
                                        Pair<float[], float[]> ropeFreqs,
                                        GGMLTensorEntry tokenEmbeddings,
                                        GGMLTensorEntry outputWeight) {
        // Load all weight tensors
        // Pattern: "blk.{layer}.{component}.weight"

        return new {ModelName}StandardWeights(
                loadQuantized(tokenEmbeddings),
                loadArrayOfQuantized(config.numberOfLayers(),
                        i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfQuantized(config.numberOfLayers(),
                        i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfQuantized(config.numberOfLayers(),
                        i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayOfQuantized(config.numberOfLayers(),
                        i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayOfQuantized(config.numberOfLayers(),
                        i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                // ... load all tensors
                loadQuantized(tensorEntries.get("output_norm.weight")),
                new ArrayFloatTensor(ropeFreqs.first()),
                new ArrayFloatTensor(ropeFreqs.second()),
                loadQuantized(outputWeight),
                outputWeight.ggmlType()
        );
    }

    @Override
    public Weights createTornadoVMWeights(/* ... */) {
        // Similar to createStandardWeights but using FloatArray
        // Use loadTensorAsFloatArray() and loadArrayAsFloatArrayFromBuffer()
        return new {ModelName}TornadoWeights(/* ... */);
    }
}
```

**Debugging Tips**:
- Print all tensor names: `tensorEntries.keySet().stream().sorted().forEach(System.err::println);`
- Check tensor shapes: `System.err.println("Shape: " + Arrays.toString(tensor.shape()));`
- Verify metadata keys: `metadata.keySet().stream().filter(k -> k.startsWith("llama")).forEach(System.err::println);`

---

### Phase 7: Inference Implementation (3-5 hours)

#### Step 2.8: Implement Forward Pass

**File**: `src/main/java/org/beehive/gpullama3/inference/InferenceCore.java`

Add method:

```java
public static FloatTensor forwardJava{ModelName}(Model model, State state,
                                                  int token, int position) {
    Configuration config = model.configuration();
    {ModelName}StandardWeights weights = ({ModelName}StandardWeights) model.weights();
    int dim = config.dim();
    int kvDim = config.kvDim();
    int kvMul = config.kvMul();
    int headSize = config.headSize();
    int hiddenDim = config.hiddenDim();

    // 1. COPY TOKEN EMBEDDING
    weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

    // 2. APPLY EMBEDDING SCALING (if model requires)
    // Example for Gemma:
    // float embeddingScale = (float) Math.sqrt(dim);
    // for (int i = 0; i < dim; i++) {
    //     state.x.setFloat(i, state.x.getFloat(i) * embeddingScale);
    // }

    // 3. FORWARD THROUGH ALL LAYERS
    for (int l = 0; l < config.numberOfLayers(); l++) {
        int curLayer = l;

        // ===== ATTENTION BLOCK =====

        // 3.1 Pre-normalization
        rmsnorm(state.xb, state.x, weights.rms_att_weight[curLayer],
                dim, config.rmsNormEps());

        // 3.2 QKV projections
        weights.wq[l].matmul(state.xb, state.q, dim, dim);
        weights.wk[l].matmul(state.xb, state.k, dim, kvDim);
        weights.wv[l].matmul(state.xb, state.v, dim, kvDim);

        // 3.3 Apply Q/K normalization (if model uses it)
        // rmsnorm(state.q, state.q, weights.attnQNorm[curLayer], ...);
        // rmsnorm(state.k, state.k, weights.attnKNorm[curLayer], ...);

        // 3.4 Apply RoPE
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % headSize;
            float fcr = weights.freq_cis_real.getFloat(position * (dim / 2) + i / 2);
            float fci = weights.freq_cis_imag.getFloat(position * (dim / 2) + i / 2);

            float q0 = state.q.getFloat(i);
            float q1 = state.q.getFloat(i + 1);
            state.q.setFloat(i, q0 * fcr - q1 * fci);
            state.q.setFloat(i + 1, q0 * fci + q1 * fcr);
        }
        // Apply RoPE to keys similarly

        // 3.5 Store KV in cache
        int loff = l * config.contextLength() * kvDim;
        state.k.copyTo(0, state.key_cache, loff + position * kvDim, kvDim);
        state.v.copyTo(0, state.value_cache, loff + position * kvDim, kvDim);

        // 3.6 Multi-head attention
        for (int h = 0; h < config.numberOfHeads(); h++) {
            // Compute attention for this head
            // See existing implementations for detailed attention logic
        }

        // 3.7 Output projection
        weights.wo[l].matmul(state.xb, state.xb2, dim, dim);

        // 3.8 Apply post-attention normalization (if model uses it)
        // rmsnorm(state.xb2, state.xb2, weights.postAttentionNorm[curLayer], ...);

        // 3.9 Residual connection
        state.x.addInPlace(state.xb2);

        // ===== FFN BLOCK =====

        // 3.10 Pre-normalization
        rmsnorm(state.xb, state.x, weights.rms_ffn_weight[curLayer],
                dim, config.rmsNormEps());

        // 3.11 FFN computation (SwiGLU activation)
        weights.w1[l].matmul(state.xb, state.hb, dim, hiddenDim);
        weights.w3[l].matmul(state.xb, state.hb2, dim, hiddenDim);

        // Apply activation
        for (int i = 0; i < hiddenDim; i++) {
            float val = state.hb.getFloat(i);
            val = val / (1.0f + (float) Math.exp(-val)); // Swish
            val *= state.hb2.getFloat(i);                 // Gate
            state.hb.setFloat(i, val);
        }

        // 3.12 Output projection
        weights.w2[l].matmul(state.hb, state.xb2, hiddenDim, dim);

        // 3.13 Apply post-FFN normalization (if model uses it)
        // rmsnorm(state.xb2, state.xb2, weights.postFFNNorm[curLayer], ...);

        // 3.14 Residual connection
        state.x.addInPlace(state.xb2);
    }

    // 4. FINAL LAYER NORM
    rmsnorm(state.x, state.x, weights.rms_final_weight, dim, config.rmsNormEps());

    // 5. CLASSIFIER
    weights.wcls.matmul(state.x, state.logits, dim, config.vocabularySize());

    return state.logits;
}
```

**Key Considerations**:
1. **Normalization**: RMSNorm, LayerNorm, or custom?
2. **Activation**: SwiGLU, GELU, ReLU?
3. **Attention**: Standard, GQA, sliding window?
4. **Special operations**: Embedding scaling, rope scaling, etc.

---

### Phase 8: Integration (30 minutes)

#### Step 2.9: Update ModelType Enum

**File**: `src/main/java/org/beehive/gpullama3/model/ModelType.java`

```java
{MODEL_NAME} {
    @Override
    public Model loadModel(FileChannel fileChannel, GGUF gguf,
                          int contextLength, boolean loadWeights,
                          boolean useTornadovm) {
        return new {ModelName}ModelLoader(fileChannel, gguf, contextLength,
                                         loadWeights, useTornadovm).loadModel();
    }
}
```

#### Step 2.10: Update Model Detection

**File**: `src/main/java/org/beehive/gpullama3/model/loader/ModelLoader.java`

```java
else if (lowerName.contains("{model}")) {
    return ModelType.{MODEL_NAME};
}
```

#### Step 2.11: Update TornadoVM Planner (if needed)

**File**: `src/main/java/org/beehive/gpullama3/tornadovm/TornadoVMMasterPlan.java`

```java
case {MODEL_NAME} -> createLlamaPlanner(state, model);  // or createQWEN3Planner
```

**Planner Selection**:
- Use `createLlamaPlanner` for standard transformers
- Use `createQWEN3Planner` for models with Q/K normalization
- Create custom planner if architecture is significantly different

---

## Testing and Debugging

### Phase 9: Testing (Ongoing)

#### Step 3.1: Unit Tests

Create test file: `src/test/java/org/beehive/gpullama3/model/{modelname}/{ModelName}Test.java`

```java
@Test
public void testTokenization() {
    // Test basic tokenization
}

@Test
public void testChatFormatting() {
    // Test chat template
}

@Test
public void testModelLoading() {
    // Test GGUF loading
}
```

#### Step 3.2: Integration Testing

```bash
# 1. Test model loading
./llama-tornado --model {model}.gguf --prompt "test"

# 2. Test with different quantizations
./llama-tornado --model {model}-Q8_0.gguf --prompt "Hello"
./llama-tornado --model {model}-f16.gguf --prompt "Hello"

# 3. Test CPU vs GPU
./llama-tornado --model {model}.gguf --prompt "test"  # CPU
./llama-tornado --model {model}.gguf --prompt "test" --gpu  # GPU

# 4. Test interactive mode
./llama-tornado --model {model}.gguf -i

# 5. Test with system prompt
./llama-tornado --model {model}.gguf --prompt "test" -sp "You are a helpful assistant"
```

#### Step 3.3: Debugging Checklist

- [ ] **Model loads without errors**
  - Check metadata keys match expected names
  - Verify all tensors are found

- [ ] **Vocabulary size matches**
  - Compare GGUF vocab size with config
  - Check embedding tensor shape

- [ ] **Tokenization works**
  - Test encode/decode round-trip
  - Verify special tokens are recognized

- [ ] **Generates tokens**
  - Not just stop tokens immediately
  - Token IDs are within vocabulary range

- [ ] **Output is readable**
  - Not garbled or nonsensical
  - Follows prompt context

- [ ] **Performance is reasonable**
  - CPU: 5-20 tok/s depending on size
  - GPU: 50-200 tok/s depending on size

---

## Common Patterns

### Pattern 1: Standard Transformer (like Llama)
- 2 norm layers per block
- Standard multi-head attention
- SwiGLU activation
- RoPE position encoding

**Reuse**:
- `StandardWeights` class
- `forwardJavaLlama` inference
- `LlamaChatFormat` (with modifications)

### Pattern 2: Grouped-Query Attention (like Mistral)
- Fewer KV heads than Q heads
- Otherwise similar to Llama

**Reuse**:
- Same as Llama
- Adjust `numberOfKeyValueHeads` in config

### Pattern 3: With Q/K Normalization (like Qwen3)
- Per-head normalization of Q and K
- May use separate head dimensions

**Reuse**:
- `StandardWeightsWithQKNorm` base class
- `forwardJavaQwen3` inference
- `generateTokensQwen3` generation method

### Pattern 4: Sandwich Normalization (like Gemma3)
- 4 norm layers per block
- Pre and post normalization

**New Implementation Required**:
- Custom weights class with 4 norm arrays
- Custom forward pass with extra norm steps

---

## Troubleshooting

### Issue: Model doesn't load

**Symptoms**: Exception during model loading

**Debug Steps**:
1. Print all metadata keys:
   ```java
   metadata.keySet().forEach(System.err::println);
   ```
2. Check architecture name:
   ```java
   String arch = (String) metadata.get("general.architecture");
   System.err.println("Architecture: " + arch);
   ```
3. Try different prefixes (llama., mistral., {model}.)

### Issue: Immediate stop token generation

**Symptoms**: Model generates stop token as first token

**Possible Causes**:
- Chat format is wrong (missing model turn setup)
- Normalization epsilon is incorrect
- Embedding scaling is missing or wrong
- Weights are loaded incorrectly

**Debug**:
1. Enable echo mode to see what's generated
2. Check prompt token IDs are correct
3. Verify chat template matches model's expected format
4. Add debug prints in forward pass to check tensor values

### Issue: Garbage output

**Symptoms**: Nonsensical or random characters

**Possible Causes**:
- Tokenizer decode logic is wrong
- Byte tokens not handled correctly
- Special tokens not filtered
- Wrong vocabulary

**Debug**:
1. Print token IDs being generated
2. Check token ID → string mapping
3. Verify byte token handling
4. Test with known-good prompts

### Issue: Slow performance

**Symptoms**: Much slower than expected

**Possible Causes**:
- Not using vectorization (Java Vector API)
- Memory layout inefficient
- Missing optimizations in matmul

**Solutions**:
- Check `USE_VECTOR_API` flag is enabled
- Profile with JMH
- Compare with reference implementation speeds

### Issue: GPU doesn't work

**Symptoms**: GPU mode crashes or falls back to CPU

**Possible Causes**:
- TornadoVM not installed correctly
- Wrong planner selected
- Memory insufficient

**Debug**:
1. Check TornadoVM installation: `tornado --devices`
2. Try with smaller model first
3. Enable verbose logging: `--verbose-init`

---

## Validation Checklist

Before considering implementation complete:

### Functionality
- [ ] Model loads from GGUF file
- [ ] Tokenization encode/decode works
- [ ] Chat format is correct
- [ ] Generates coherent output
- [ ] Stop tokens work correctly
- [ ] Special tokens are handled
- [ ] Multiple quantization types work (Q8_0, F16)

### Performance
- [ ] CPU inference speed is reasonable
- [ ] GPU inference works (if applicable)
- [ ] Memory usage is acceptable
- [ ] No memory leaks

### Code Quality
- [ ] Follows existing code style
- [ ] Has inline documentation
- [ ] Complex logic is commented
- [ ] No debug prints in production code
- [ ] Exception handling is proper

### Testing
- [ ] Manual testing with various prompts
- [ ] Tested with different quantization formats
- [ ] Tested in interactive mode
- [ ] Tested with system prompts
- [ ] Compared output with reference implementation

### Documentation
- [ ] Changes documented in CHANGES.md
- [ ] Added model to README.md
- [ ] Chat template documented
- [ ] Any quirks or limitations noted

---

## Additional Resources

### HuggingFace
- Model cards with architecture details
- `config.json` for hyperparameters
- `tokenizer_config.json` for tokenization

### llama.cpp
- Reference C++ implementations
- GGUF format documentation
- Performance benchmarks

### Papers
- Original model papers
- Architecture variants
- Tokenization methods

### Community
- GitHub issues for similar models
- Discord/forums for Q&A
- Existing PRs as examples

---

## Example: Quick Reference Commands

```bash
# Download model from HuggingFace
huggingface-cli download {org}/{model}-GGUF {file}.gguf --local-dir .

# Build project
make clean && make

# Test basic inference
./llama-tornado --model {model}.gguf --prompt "Hello, how are you?"

# Test with echo to see tokens
./llama-tornado --model {model}.gguf --prompt "test" --echo true

# Interactive mode
./llama-tornado --model {model}.gguf -i

# GPU mode
./llama-tornado --model {model}.gguf --prompt "test" --gpu --gpu-memory 8GB

# Debug vocabulary
./llama-tornado --model {model}.gguf --prompt "test" 2>&1 | grep -i vocab
```

---

## Conclusion

Adding a new model requires:
1. **Understanding** the architecture deeply
2. **Implementing** 8-10 core classes
3. **Testing** thoroughly
4. **Debugging** patiently

**Estimated Time**: 1-3 days for experienced developers

**Difficulty Factors**:
- Standard transformer: ⭐⭐ (Easy)
- With GQA: ⭐⭐⭐ (Medium)
- With Q/K norm: ⭐⭐⭐⭐ (Hard)
- Completely custom: ⭐⭐⭐⭐⭐ (Expert)

Good luck! 🚀
