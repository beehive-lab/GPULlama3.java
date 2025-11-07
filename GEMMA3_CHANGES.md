# Gemma 3 Implementation - Changes Documentation

## Overview
This document details all changes made to add Google Gemma 3 model support to GPULlama3.java.

**Date**: November 1, 2025
**Model**: Google Gemma 3 (1B, 4B, 12B, 27B variants)
**Status**: Implementation complete, debugging in progress

---

## Architecture Details

### Gemma 3 Unique Features
1. **Sandwich Normalization**: 4 normalization layers per block (vs. 2 in standard transformers)
   - `attn_norm` → Attention → `post_attention_norm` → Residual
   - `ffn_norm` → FFN → `post_ffw_norm` → Residual

2. **Q/K Normalization**: Per-head normalization of query and key vectors within attention

3. **Embedding Scaling**: Embeddings multiplied by √dim for numerical stability

4. **Byte-Level Tokenization**: First 256 tokens (type 3) are raw bytes, stored as `<unused0>` to `<unused255>` in vocabulary

5. **SentencePiece Tokenizer**: Uses ▁ (U+2581) character to represent spaces

---

## Files Created

### 1. Model Configuration
**File**: `src/main/java/org/beehive/gpullama3/model/gemma3/Gemma3Configuration.java`
```java
public record Gemma3Configuration(
    int dim, int hiddenDim, int numberOfLayers, int numberOfHeads,
    int numberOfKeyValueHeads, int numberOfHeadsKey, int numberOfHeadsValue,
    int vocabularySize, int contextLengthModel, int contextLength,
    boolean sharedWeights, float rmsNormEps, float ropeTheta
) implements Configuration
```
- Compatible with Qwen3 structure (includes numberOfHeadsKey/Value fields)
- Supports 128K context window

### 2. Model State
**File**: `src/main/java/org/beehive/gpullama3/inference/state/Gemma3State.java`
- Manages KV cache and inference buffers
- Extends base `State` class

### 3. Main Model Class
**File**: `src/main/java/org/beehive/gpullama3/model/gemma3/Gemma3.java`
```java
@Override
public void forward(State state, int token, int position) {
    if (plan == null) {
        InferenceCore.forwardJavaGemma3(this, state, token, position);
    } else {
        InferenceCore.forwardTornadoVM(this, state, token, position, tornadoVMPlan());
    }
}
```
- Routes to Gemma3-specific CPU inference or Qwen3 GPU planner

### 4. Tokenizer Implementation
**File**: `src/main/java/org/beehive/gpullama3/tokenizer/impl/Gemma3Tokenizer.java`

**Key Features**:
- Loads token types from metadata (`tokenizer.ggml.token_type`)
- Distinguishes between byte tokens (type 3) and regular tokens (type 6)
- Special token detection excludes `<unusedNN>` and `<0xHH>` patterns

**Critical Decoder Logic**:
```java
@Override
public String decode(List<Integer> tokens) {
    for (int token : tokens) {
        // Type 3: Byte tokens (IDs 0-255) - decode as raw bytes
        if (tokenTypes != null && tokenTypes[token] == 3) {
            sb.append((char) token);
            continue;
        }

        String tokenString = vocabulary.get(token);

        // Hex byte tokens like <0x12>
        if (tokenString.matches("<0x[0-9a-fA-F]{2}>")) {
            String code = tokenString.substring(3, tokenString.length() - 1);
            int byteValue = Integer.parseInt(code, 16);
            tokenString = Character.toString(byteValue);
        } else if (isSpecialToken(token)) {
            continue; // Skip special tokens
        } else {
            // SentencePiece: ▁ → space
            tokenString = tokenString.replace('▁', ' ');
        }
        sb.append(tokenString);
    }
    return sb.toString();
}
```

### 5. Chat Format
**File**: `src/main/java/org/beehive/gpullama3/model/format/Gemma3ChatFormat.java`

**Template Format**:
```
<bos><start_of_turn>user
{user_message}<end_of_turn>
<start_of_turn>model
{model_message}<end_of_turn>
```

**Stop Tokens**: `<end_of_turn>`, `<eos>`

### 6. Weight Classes

#### CPU Weights Base Class
**File**: `src/main/java/org/beehive/gpullama3/inference/weights/standard/StandardWeightsWithQKNorm.java`
```java
public abstract class StandardWeightsWithQKNorm extends StandardWeights {
    public final FloatTensor[] attnKNorm, attnQNorm;
}
```

#### Gemma3 CPU Weights
**File**: `src/main/java/org/beehive/gpullama3/inference/weights/standard/Gemma3StandardWeights.java`
```java
public class Gemma3StandardWeights extends StandardWeightsWithQKNorm {
    public final FloatTensor[] postAttentionNorm;  // Post-attention normalization
    public final FloatTensor[] postFFNNorm;        // Post-FFN normalization
}
```

#### Gemma3 GPU Weights
**File**: `src/main/java/org/beehive/gpullama3/inference/weights/tornado/Gemma3TornadoWeights.java`
```java
public class Gemma3TornadoWeights extends FP16Weights {
    public FloatArray[] rms_att_KNormLayered;
    public FloatArray[] rms_att_QNormLayered;
    public FloatArray[] postAttentionNormLayered;
    public FloatArray[] postFFNNormLayered;
}
```

### 7. Model Loader
**File**: `src/main/java/org/beehive/gpullama3/model/loader/Gemma3ModelLoader.java`

**Metadata Prefix Detection**:
```java
// Tries: gemma3. → gemma2. → gemma. → llama.
if (metadata.containsKey("gemma3.embedding_length")) {
    prefix = "gemma3.";
} else if (metadata.containsKey("gemma2.embedding_length")) {
    prefix = "gemma2.";
}
```

**Tensor Loading** (4 norm layers per block):
```java
loadArrayOfQuantized(config.numberOfLayers(),
    i -> tensorEntries.get("blk." + i + ".attn_norm.weight"))
loadArrayOfQuantized(config.numberOfLayers(),
    i -> tensorEntries.get("blk." + i + ".post_attention_norm.weight"))
loadArrayOfQuantized(config.numberOfLayers(),
    i -> tensorEntries.get("blk." + i + ".ffn_norm.weight"))
loadArrayOfQuantized(config.numberOfLayers(),
    i -> tensorEntries.get("blk." + i + ".post_ffw_norm.weight"))
```

---

## Files Modified

### 1. Model Type Enum
**File**: `src/main/java/org/beehive/gpullama3/model/ModelType.java`

**Added**:
```java
GEMMA_3 {
    @Override
    public Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength,
                          boolean loadWeights, boolean useTornadovm) {
        return new Gemma3ModelLoader(fileChannel, gguf, contextLength,
                                    loadWeights, useTornadovm).loadModel();
    }
}
```

### 2. Model Detection
**File**: `src/main/java/org/beehive/gpullama3/model/loader/ModelLoader.java`

**Added**:
```java
else if (lowerName.contains("gemma")) {
    return ModelType.GEMMA_3;
}
```

### 3. Inference Core
**File**: `src/main/java/org/beehive/gpullama3/inference/InferenceCore.java`

**Added Method**: `forwardJavaGemma3()` (~150 lines)

**Key Implementation Details**:
```java
// Embedding scaling
float embeddingScale = (float) Math.sqrt(dim);
for (int i = 0; i < dim; i++) {
    state.x.setFloat(i, state.x.getFloat(i) * embeddingScale);
}

for (int l = 0; l < config.numberOfLayers(); l++) {
    // ATTENTION BLOCK with sandwich normalization
    state.x.copyTo(0, state.xb2, 0, dim);  // Save residual
    rmsnorm(state.xb, state.x, weights.rms_att_weight[curLayer], ...);

    // ... QKV matmuls, Q/K norm, RoPE, attention ...

    weights.wo[l].matmul(state.xb, state.x, ...);
    rmsnorm(state.x, state.x, weights.postAttentionNorm[curLayer], ...); // POST-NORM
    state.x.addInPlace(state.xb2); // Residual

    // FFN BLOCK with sandwich normalization
    state.x.copyTo(0, state.xb2, 0, dim);  // Save residual
    rmsnorm(state.xb, state.x, weights.rms_ffn_weight[curLayer], ...);

    // ... FFN computation ...

    rmsnorm(state.x, state.x, weights.postFFNNorm[curLayer], ...); // POST-NORM
    state.x.addInPlace(state.xb2); // Residual
}
```

### 4. TornadoVM Planner
**File**: `src/main/java/org/beehive/gpullama3/tornadovm/TornadoVMMasterPlan.java`

**Modified**:
```java
case QWEN_3, GEMMA_3 -> createQWEN3Planner(state, model);
```
Routes Gemma 3 to Qwen3 planner (both use Q/K normalization)

### 5. Configuration Interface
**File**: `src/main/java/org/beehive/gpullama3/model/Configuration.java`

**Added**:
```java
int numberOfHeadsValue();  // For Gemma3/Qwen3 compatibility
```

### 6. Other Configuration Classes
**Files**:
- `LlamaConfiguration.java`
- `MistralConfiguration.java`
- `Phi3Configuration.java`

**Added** implementations of `numberOfHeadsValue()` method

---

## Known Issues

### Issue 1: Immediate Stop Token Generation
**Symptom**: Model generates `<end_of_turn>` (token 106) as first token
**Status**: Under investigation
**Possible Causes**:
1. Incorrect normalization implementation
2. Missing Gemma-specific initialization
3. Weight loading mismatch
4. Chat template formatting issue

### Issue 2: GGUF Compatibility
**Tested Models**:
- ❌ User-provided GGUF files (corrupted vocabulary)
- ❌ `ggml-org/gemma-3-4b-it-GGUF` (same stop token issue)

**Next Steps**:
- Debug embedding scaling factor
- Verify RMSNorm epsilon values
- Check attention mask implementation
- Compare with llama.cpp implementation

---

## Testing

### Test Command
```bash
./llama-tornado --model gemma-3-4b-it-Q8_0.gguf --prompt "Tell me a joke"
```

### Expected Output Format
```
<bos><start_of_turn>user
Tell me a joke<end_of_turn>
<start_of_turn>model
[Model response]<end_of_turn>
```

### Performance
- **CPU**: ~6-9 tok/s on FP16/Q8_0 (4B model)
- **GPU**: Not yet tested

---

## References

1. **Gemma 3 Architecture**: https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal/gemma3.md
2. **HuggingFace Model**: https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF
3. **Google Blog**: Gemma 3 uses sandwich normalization and Q/K norm
4. **SentencePiece Tokenizer**: Byte-level encoding with space as ▁ character

---

## Build and Run

### Compile
```bash
make
```

### Run CPU Inference
```bash
./llama-tornado --model gemma-3-4b-it-Q8_0.gguf --prompt "Hello"
```

### Run GPU Inference (TornadoVM)
```bash
./llama-tornado --model gemma-3-4b-it-Q8_0.gguf --prompt "Hello" --gpu --gpu-memory 8GB
```

---

## Contributors
- Initial implementation: Claude (Anthropic)
- Architecture research: Based on llama.cpp and Graphcore blog posts
