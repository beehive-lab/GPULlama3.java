# GPULlama3.java - Architecture Documentation

**Version**: 0.2.2
**Last Updated**: 2025-10-28
**Repository**: https://github.com/beehive-lab/GPULlama3.java

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technology Stack](#technology-stack)
3. [Directory Structure](#directory-structure)
4. [Core Components](#core-components)
5. [Architecture Patterns](#architecture-patterns)
6. [Data Flow](#data-flow)
7. [GPU Acceleration](#gpu-acceleration)
8. [Build System](#build-system)
9. [Performance Characteristics](#performance-characteristics)
10. [Supported Models](#supported-models)

---

## Project Overview

GPULlama3.java is a **native Java implementation of Llama3 and other large language models (LLMs)** with automatic GPU acceleration via TornadoVM. It represents the first complete Java implementation of Llama3 inference with JVM-to-GPU compilation, enabling managed languages to leverage GPU acceleration without manual kernel development.

### Key Features

- **Pure Java Implementation**: No C/C++ or CUDA required
- **Automatic GPU Acceleration**: JIT compilation from Java to GPU kernels
- **Multi-Backend Support**: OpenCL, PTX/CUDA, SPIR-V
- **Multiple Model Architectures**: Llama3, Mistral, Qwen2/3, Phi3, DeepSeek-R1-Distill
- **Quantization Support**: FP32, FP16, Q8_0, Q4_0
- **GGUF Format**: Compatible with llama.cpp ecosystem
- **LangChain4j Integration**: Official model provider (v1.7.1+)

### Project Statistics

| Metric | Value |
|--------|-------|
| Total Java Files | 83 |
| Lines of Code | ~5,266 |
| Base Package | `org.beehive.gpullama3` |
| Java Version | 21+ (with preview features) |
| License | MIT |

---

## Technology Stack

### Core Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| **Java** | 21+ | Primary implementation language |
| **TornadoVM** | 1.1.2-dev | GPU acceleration framework |
| **Maven** | 3.x | Build system and dependency management |
| **GGUF Format** | v2/v3 | Model file format |

### Java Features Used

- **Vector API** (Preview): SIMD CPU acceleration
- **Foreign Function & Memory API** (Preview): Efficient memory-mapped file access
- **Records**: Immutable data classes (Options, Configuration)
- **Sealed Classes**: Type-safe model hierarchies

### GPU Backends

| Backend | Supported Hardware | Status |
|---------|-------------------|--------|
| **PTX (CUDA)** | NVIDIA GPUs | Production |
| **OpenCL** | NVIDIA, AMD, Intel, Apple | Production |
| **SPIR-V** | Future Vulkan support | Planned |

---

## Directory Structure

```
GPULlama3.java/
├── src/main/java/org/beehive/gpullama3/
│   ├── LlamaApp.java                    # Application entry point
│   ├── Options.java                     # CLI argument parser
│   │
│   ├── aot/                             # AOT compilation support
│   │   └── AOT.java
│   │
│   ├── auxiliary/                       # Utility classes
│   │   ├── LastRunMetrics.java          # Performance metrics
│   │   ├── Timer.java                   # Timing utilities
│   │   ├── Parallel.java                # Parallel processing
│   │   ├── Tuple2.java                  # Generic tuple
│   │   └── Utf8Mask.java                # UTF-8 encoding utilities
│   │
│   ├── core/                            # Core data structures
│   │   ├── model/                       # Model file format
│   │   │   ├── GGUF.java                # GGUF parser
│   │   │   ├── GGMLType.java            # Data type definitions
│   │   │   └── tensor/                  # Tensor implementations
│   │   │       ├── FloatTensor.java     # Base interface
│   │   │       ├── ArrayFloatTensor.java
│   │   │       ├── F16FloatTensor.java  # FP16 tensors
│   │   │       ├── F32FloatTensor.java  # FP32 tensors
│   │   │       ├── Q8_0FloatTensor.java # 8-bit quantized
│   │   │       ├── Q4_0FloatTensor.java # 4-bit quantized
│   │   │       └── GGMLTensorEntry.java
│   │   │
│   │   └── types/                       # Core type definitions
│   │       ├── Float16.java
│   │       ├── MetadataValueType.java
│   │       └── Pair.java
│   │
│   ├── inference/                       # Inference pipeline
│   │   ├── InferenceEngine.java         # Token generation orchestrator
│   │   ├── InferenceCore.java           # Forward pass implementation
│   │   │
│   │   ├── operation/
│   │   │   └── RoPE.java                # Rotary Position Embeddings
│   │   │
│   │   ├── sampler/                     # Token sampling
│   │   │   ├── Sampler.java
│   │   │   ├── CategoricalSampler.java
│   │   │   └── ToppSampler.java         # Nucleus sampling
│   │   │
│   │   ├── state/                       # Model state during inference
│   │   │   ├── State.java               # Base state with KV cache
│   │   │   ├── StateFieldAllocator.java # Abstract factory for state allocation
│   │   │   ├── LlamaState.java
│   │   │   ├── LlamaStateFieldAllocator.java
│   │   │   ├── Qwen2State.java
│   │   │   ├── Qwen2StateFieldAllocator.java
│   │   │   ├── Qwen3State.java
│   │   │   ├── Qwen3StateFieldAllocator.java
│   │   │   ├── Phi3State.java
│   │   │   └── Phi3StateFieldAllocator.java
│   │   │
│   │   └── weights/                     # Weight storage
│   │       ├── Weights.java             # Base interface
│   │       ├── standard/                # CPU implementations
│   │       │   ├── StandardWeights.java
│   │       │   ├── LlamaStandardWeights.java
│   │       │   ├── Qwen2StandardWeights.java
│   │       │   ├── Qwen3StandardWeights.java
│   │       │   └── Phi3StandardWeights.java
│   │       │
│   │       └── tornado/                 # GPU implementations
│   │           ├── TornadoWeights.java
│   │           ├── LlamaTornadoWeights.java
│   │           ├── Qwen2TornadoWeights.java
│   │           ├── Qwen3TornadoWeights.java
│   │           └── Phi3TornadoWeights.java
│   │
│   ├── model/                           # Model abstractions
│   │   ├── Model.java                   # Core interface
│   │   ├── AbstractModel.java           # Base implementation
│   │   ├── Configuration.java           # Model configuration
│   │   ├── ModelType.java               # Enum for model types
│   │   │
│   │   ├── format/                      # Chat formatting
│   │   │   ├── ChatFormat.java
│   │   │   ├── LlamaChatFormat.java
│   │   │   ├── MistralChatFormat.java
│   │   │   ├── Qwen3ChatFormat.java
│   │   │   └── Phi3ChatFormat.java
│   │   │
│   │   ├── llama/                       # Llama implementation
│   │   │   ├── Llama.java
│   │   │   └── LlamaConfiguration.java
│   │   │
│   │   ├── mistral/                     # Mistral implementation
│   │   │   ├── Mistral.java
│   │   │   └── MistralConfiguration.java
│   │   │
│   │   ├── qwen2/                       # Qwen2 implementation
│   │   │   ├── Qwen2.java
│   │   │   └── Qwen2Configuration.java
│   │   │
│   │   ├── qwen3/                       # Qwen3 implementation
│   │   │   ├── Qwen3.java
│   │   │   └── Qwen3Configuration.java
│   │   │
│   │   ├── phi3/                        # Phi3 implementation
│   │   │   ├── Phi3.java
│   │   │   └── Phi3Configuration.java
│   │   │
│   │   └── loader/                      # Model loaders
│   │       ├── ModelLoader.java         # Utility methods for loading
│   │       ├── AbstractModelLoader.java # Template method base class
│   │       ├── ModelLoadException.java  # Model loading exception
│   │       ├── LlamaModelLoader.java
│   │       ├── MistralModelLoader.java
│   │       ├── Qwen2ModelLoader.java
│   │       ├── Qwen3ModelLoader.java
│   │       └── Phi3ModelLoader.java
│   │
│   ├── tokenizer/                       # Tokenization
│   │   ├── impl/
│   │   │   ├── Tokenizer.java           # Base interface
│   │   │   ├── LlamaTokenizer.java
│   │   │   ├── MistralTokenizer.java
│   │   │   ├── Qwen3Tokenizer.java
│   │   │   └── Phi3Tokenizer.java
│   │   │
│   │   └── vocabulary/
│   │       └── Vocabulary.java
│   │
│   └── tornadovm/                       # GPU acceleration
│       ├── TornadoVMMasterPlan.java     # GPU execution orchestrator
│       ├── TornadoVMLayerPlanner.java   # Base layer planner
│       ├── Qwen2TornadoVMLayerPlanner.java
│       ├── Qwen3TornadoVMLayerPlanner.java
│       ├── Phi3TornadoVMLayerPlanner.java
│       ├── TransformerComputeKernels.java
│       ├── TransformerComputeKernelsLayered.java
│       ├── Qwen2Kernels.java
│       ├── Qwen3Kernels.java
│       └── FloatArrayUtils.java
│
├── external/
│   └── tornadovm/                       # TornadoVM submodule
│
├── scripts/                             # Helper scripts
│   ├── all.sh
│   └── example-argfile
│
├── docs/                                # Documentation
│   ├── TORNADOVM_TRANSFORMER_OPTIMIZATIONS.md
│   ├── GPULlama3_ROADMAP.md
│   └── performance.png
│
├── pom.xml                              # Maven configuration
├── Makefile                             # Build shortcuts
├── set_paths / set_paths.cmd            # Environment setup
├── README.md
├── CONTRIBUTING.md
├── LICENSE
└── CITATION.cff
```

---

## Core Components

### 1. Entry Point & CLI

#### LlamaApp.java
**Purpose**: Application bootstrap and execution orchestration

**Responsibilities**:
- Parse command-line arguments via Options
- Load model via ModelType factory
- Initialize Sampler (Categorical or Top-p)
- Route to interactive or instruct mode
- Manage execution lifecycle

**Key Configuration Flags**:
```java
USE_VECTOR_API       // Enable Java Vector API for CPU SIMD
SHOW_PERF_INTERACTIVE // Show performance metrics in interactive mode
```

#### Options.java
**Purpose**: CLI configuration and validation

**Fields**:
```java
record Options(
    String modelPath,        // Path to GGUF model file
    String prompt,           // Input prompt (instruct mode)
    boolean interactive,     // Enable interactive/chat mode
    float temperature,       // Sampling temperature (0.1-2.0)
    float topP,              // Top-p nucleus sampling (0-1)
    int seed,                // Random seed for reproducibility
    boolean stream,          // Stream tokens as generated
    int maxTokens,           // Maximum tokens to generate
    boolean useTornadovm,    // Enable GPU acceleration
    boolean echo             // Echo input prompt
)
```

---

### 2. Model Abstraction Layer

#### Model.java (Interface)
**Purpose**: Core contract for all model implementations

**Key Methods**:
```java
Configuration configuration()    // Model architecture parameters
Tokenizer tokenizer()            // Text <-> token conversion
Weights weights()                // Model weights (CPU or GPU)
ChatFormat chatFormat()          // Chat/instruct formatting
void forward(int token, int pos, State state)  // Single forward pass
String generateTokens(...)       // Token generation pipeline
```

#### Configuration.java (Interface)
**Purpose**: Model architecture parameters

**Key Parameters**:
```java
int dim()                  // Model dimension (embedding size)
int hidden_dim()           // FFN hidden dimension
int n_heads()              // Number of attention heads
int n_layers()             // Number of transformer layers
int vocab_size()           // Vocabulary size
int context_length()       // Maximum context length
int n_kv_heads()           // Number of key-value heads (MQA/GQA)
float norm_eps()           // RMS normalization epsilon
```

#### ModelType.java (Enum)
**Purpose**: Factory pattern for model instantiation

**Supported Types**:
```java
LLAMA_3,
MISTRAL,
QWEN_2,
QWEN_3,
PHI_3,
DEEPSEEK_R1_DISTILL_QWEN
```

**Factory Method**:
```java
Model loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, Options options)
```

---

### 3. Model Implementations

Each model architecture has 4 key components:

#### Model Class (e.g., Llama.java)
- Implements `Model` interface
- Holds configuration, tokenizer, weights
- Delegates forward pass to InferenceCore or TornadoVM
- Manages state creation

#### Configuration Class (e.g., LlamaConfiguration.java)
- Immutable record of architecture parameters
- Loaded from GGUF metadata

#### ModelLoader Class (e.g., LlamaModelLoader.java)
- Extends `AbstractModelLoader` with Template Method pattern
- Implements model-specific methods:
  - `loadVocabulary()` - Load tokenizer vocabulary
  - `createTokenizer()` - Create model-specific tokenizer
  - `createConfiguration()` - Parse model configuration
  - `precomputeRopeFrequencies()` - Calculate RoPE frequencies
  - `createStandardWeights()` - Create CPU weights
  - `createTornadoVMWeights()` - Create GPU weights
- Eliminates code duplication across loaders (~60% reduction)

#### ChatFormat Class (e.g., LlamaChatFormat.java)
- Prompt templating for chat/instruct modes
- System prompt injection
- User/assistant markers

---

### 4. GGUF Model Format Support

#### GGUF.java
**Purpose**: Parse GGUF (GPT-Generated Unified Format) files

**Process**:
1. Read magic number (0x46554747)
2. Parse version and alignment
3. Extract metadata key-value pairs
4. Map tensor information (name, type, dimensions)
5. Load tensor data via memory-mapped files

**Supported Versions**: GGUF v2, v3

#### GGMLType.java
**Purpose**: Data type definitions and dequantization

**Supported Types**:
- `F32`: 32-bit float
- `F16`: 16-bit float
- `Q8_0`: 8-bit quantized (block-based)
- `Q4_0`: 4-bit quantized (block-based)
- `Q4_1`, `Q5_0`, `Q5_1`: Other quantization formats

#### Tensor Hierarchy

```
FloatTensor (interface)
    ├── ArrayFloatTensor     # In-memory array
    ├── F32FloatTensor       # FP32 storage
    ├── F16FloatTensor       # FP16 storage
    ├── Q8_0FloatTensor      # 8-bit quantized (dequantized on access)
    └── Q4_0FloatTensor      # 4-bit quantized (dequantized on access)
```

---

### 5. Inference Pipeline

#### InferenceEngine.java
**Purpose**: Token generation orchestrator

**Key Methods**:
```java
generateTokensLlama(...)     // Llama/Mistral generation
generateTokensQwen3(...)     // Qwen3 generation
generateTokensGPU(...)       // GPU-accelerated generation
```

**Generation Process**:
1. **Ingestion Phase**: Process all prompt tokens, populate KV cache
2. **Generation Phase**: Auto-regressive token generation
3. **Termination**: Stop at EOS token or max tokens
4. **Callback**: Stream tokens via callback function

#### InferenceCore.java
**Purpose**: Low-level forward pass implementation

**CPU Forward Pass**:
```java
forwardJava(...)         // Llama/Mistral CPU forward pass
forwardJavaQwen3(...)    // Qwen3 CPU forward pass
```

**GPU Forward Pass**:
```java
forwardTornadoVM(...)    // TornadoVM GPU execution
```

**Operations**:
- **RMS Normalization**: `rmsnorm()`
- **Matrix Multiplication**: Weight-activation products
- **Attention**: Scaled dot-product with softmax
- **Feedforward**: SwiGLU/GELU activation
- **RoPE**: Rotary Position Embeddings

---

### 6. State Management

#### State.java (Abstract Base)
**Purpose**: Manage activation buffers and KV cache during inference

**Activation Buffers**:
```java
float[] x, xb, xb2      // Activations at different stages
float[] hb, hb2         // Hidden dimension buffers (FFN)
float[] q, k, v         // Query, Key, Value tensors
float[] att             // Attention scores
float[] logits          // Output logits (vocab_size)
```

**KV Cache**:
```java
float[][][] keyCache[layer][position][kv_dim]
float[][][] valueCache[layer][position][kv_dim]
```

**TornadoVM Wrappers** (for GPU):
```java
FloatArray wrapLogits, wrapX, wrapQ, wrapK, wrapV, wrapAtt, ...
```

#### Model-Specific States
Each state class uses the **StateFieldAllocator** factory pattern:
- **LlamaState**: Llama/Mistral models → `LlamaStateFieldAllocator`
- **Qwen2State**: Qwen2 models → `Qwen2StateFieldAllocator`
- **Qwen3State**: Qwen3 models → `Qwen3StateFieldAllocator`
- **Phi3State**: Phi3 models → `Phi3StateFieldAllocator`

**StateFieldAllocator Pattern**:
- Abstract factory for state field allocation
- Eliminates ~200 lines of duplication (60-66% reduction per state)
- Centralizes allocation logic in factory classes
- Model-specific dimensions defined in allocator subclasses

---

### 7. Tokenization

#### Tokenizer.java (Interface)
**Purpose**: Text <-> Token ID conversion

**Key Methods**:
```java
List<Integer> encode(String text)        // Text -> Token IDs
String decode(List<Integer> tokens)      // Token IDs -> Text
String regexPattern()                    // Tokenization regex
int specialToken(String token)           // Get special token ID
```

#### Model-Specific Tokenizers
- **LlamaTokenizer**: "cl100k_base" encoding
- **MistralTokenizer**: Mistral-specific boundaries
- **Qwen3Tokenizer**: Qwen3 encoding
- **Phi3Tokenizer**: Phi3 encoding

#### Vocabulary.java
**Purpose**: Token-to-string mapping and BPE support

---

### 8. Sampling Strategies

#### Sampler.java (Interface)
**Purpose**: Next-token selection strategy

#### CategoricalSampler
**Strategy**: Temperature-based random sampling

**Process**:
1. Apply temperature scaling to logits
2. Convert logits to probabilities (softmax)
3. Sample from categorical distribution

#### ToppSampler
**Strategy**: Top-p (nucleus) sampling

**Process**:
1. Apply temperature scaling
2. Sort tokens by probability
3. Select smallest set with cumulative probability ≥ p
4. Sample from filtered distribution

---

### 9. Weight Storage

#### StandardWeights (CPU)
**Storage**: `FloatTensor[]` (supports all quantization formats)

**Access**: Direct memory access, dequantization on load

#### TornadoWeights (GPU)
**Storage**: `FloatArray[]` (TornadoVM native type)

**Access**: Pre-allocated on GPU device memory

#### Model-Specific Weights

**Llama/Mistral Weights**:
```java
token_embedding_table    // Embedding matrix [vocab_size x dim]
rms_att_weight           // Attention RMS norm [n_layers x dim]
wq, wk, wv, wo           // Attention weights [n_layers x ...]
rms_ffn_weight           // FFN RMS norm [n_layers x dim]
w1, w2, w3               // FFN weights [n_layers x ...]
rms_final_weight         // Final RMS norm [dim]
wcls                     // Output projection [vocab_size x dim]
```

---

### 10. GPU Acceleration (TornadoVM)

#### TornadoVMMasterPlan.java
**Purpose**: Orchestrate GPU execution

**Initialization Steps**:
1. **Plan Creation**: Build TornadoVM task graphs (~100-500ms)
2. **JIT Compilation**: Java -> GPU kernel compilation (~500-2000ms)
3. **Weight Transfer**: Copy read-only weights to GPU memory
4. **Warmup**: Pre-execute to stabilize performance

**Device Selection**:
- Nvidia GPUs: PTX backend (optimized)
- Non-Nvidia GPUs: OpenCL backend

#### TornadoVMLayerPlanner.java
**Purpose**: Convert transformer layers to TornadoVM task graphs

**Model-Specific Planners**:
- `Qwen2TornadoVMLayerPlanner`
- `Qwen3TornadoVMLayerPlanner`
- `Phi3TornadoVMLayerPlanner`

#### TransformerComputeKernels.java
**Purpose**: GPU kernel implementations

**Optimized Kernels**:
- **RMS Normalization**: Two-phase parallel reduction
- **Matrix Multiplication**: BLAS-optimized
- **RoPE**: Precomputed lookup tables
- **Attention**: Scaled dot-product with softmax
- **SwiGLU/GELU**: Fused activation functions

---

## Architecture Patterns

### 1. Factory Pattern
**Location**: `ModelType.java`, `StateFieldAllocator.java`

**Purpose**: Type-safe model instantiation and state allocation

**Model Loading Factory**:
```java
ModelType.LLAMA_3.loadModel(fileChannel, gguf, contextLength, options)
    -> LlamaModelLoader.loadModel()
    -> new Llama(config, tokenizer, weights)
```

**State Allocation Factory** (NEW):
```java
// Abstract Factory + Template Method for state field allocation
StateFieldAllocator allocator = new LlamaStateFieldAllocator(config, localSize);
StateFields fields = allocator.allocateFields();
    -> Allocates tensors, KV cache, TornadoVM wrappers
    -> Model-specific dimensions via abstract methods
```

### 2. Strategy Pattern
**Location**: `Sampler.java` hierarchy

**Purpose**: Swappable sampling strategies

```java
Sampler sampler = useToppSampling
    ? new ToppSampler(temperature, topP, seed)
    : new CategoricalSampler(temperature, seed);
```

### 3. Adapter Pattern
**Location**: `Weights` hierarchy

**Purpose**: Unified interface for CPU/GPU weights

```java
Weights weights = options.useTornadovm()
    ? new LlamaTornadoWeights(...)
    : new LlamaStandardWeights(...);
```

### 4. Template Method Pattern
**Location**: `InferenceCore.java`, `AbstractModelLoader.java` (NEW)

**Purpose**: Extensible forward pass and model loading

**Inference Template**:
```java
abstract void forward(token, pos, state)
    ├── forwardJava(...)         // CPU implementation
    └── forwardTornadoVM(...)    // GPU implementation
```

**Model Loader Template** (NEW):
```java
// AbstractModelLoader defines loading skeleton
Model loadModel() {
    Vocabulary vocab = loadVocabulary(metadata);           // Abstract
    Tokenizer tokenizer = createTokenizer(metadata, vocab); // Abstract
    Config config = createConfiguration(metadata);          // Abstract
    Pair<float[], float[]> rope = precomputeRopeFrequencies(config); // Abstract
    Weights weights = loadWeights(...);                     // Template uses abstract methods
    return createModel(config, tokenizer, weights);         // Abstract
}
```

### 5. Builder Pattern
**Location**: `TornadoVMMasterPlan.java`

**Purpose**: Complex GPU execution plan construction

```java
TornadoVMMasterPlan.builder()
    .withPreCompilation()
    .withDeviceMemory("7GB")
    .build();
```

### 6. Visitor Pattern
**Location**: `ModelType.java` enum

**Purpose**: Model-specific operations dispatch

```java
enum ModelType {
    LLAMA_3 {
        public Model loadModel(...) { /* Llama-specific logic */ }
    },
    QWEN_3 {
        public Model loadModel(...) { /* Qwen3-specific logic */ }
    }
}
```

---

## Data Flow

### Inference Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. MODEL LOADING                                                │
├─────────────────────────────────────────────────────────────────┤
│ GGUF File → Parser → Metadata + Tensors                         │
│   ├─ Detect Model Type (from metadata)                          │
│   ├─ Create Configuration                                       │
│   ├─ Deserialize Weights (Standard or Tornado)                  │
│   └─ Initialize Tokenizer                                       │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. INPUT PREPARATION                                            │
├─────────────────────────────────────────────────────────────────┤
│ Text Prompt → Tokenizer.encode() → Token IDs                    │
│   ├─ Apply ChatFormat (if instruct mode)                        │
│   ├─ Add system prompt (if provided)                            │
│   └─ Create token sequence                                      │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. GPU INITIALIZATION (if enabled)                              │
├─────────────────────────────────────────────────────────────────┤
│ TornadoVMMasterPlan                                             │
│   ├─ Create Task Graphs (layer by layer)                        │
│   ├─ JIT Compile Java → GPU Kernels                             │
│   ├─ Transfer Weights to GPU Memory                             │
│   └─ Warmup Execution                                           │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. TOKEN INGESTION (Prompt Processing)                          │
├─────────────────────────────────────────────────────────────────┤
│ For each prompt token:                                          │
│   Model.forward(token, position, state)                         │
│     ├─ Token Embedding Lookup                                   │
│     ├─ For each layer:                                          │
│     │   ├─ RMS Norm (input)                                     │
│     │   ├─ Self-Attention (Q, K, V)                             │
│     │   │   ├─ Linear projections (Wq, Wk, Wv)                 │
│     │   │   ├─ RoPE (rotary position embeddings)               │
│     │   │   ├─ Store K, V in cache                              │
│     │   │   ├─ Attention scores (Q @ K^T)                       │
│     │   │   └─ Weighted sum of V                                │
│     │   ├─ Residual connection                                  │
│     │   ├─ RMS Norm (FFN)                                       │
│     │   ├─ Feedforward (SwiGLU)                                 │
│     │   │   ├─ W1, W3 projections                               │
│     │   │   ├─ SwiGLU activation                                │
│     │   │   └─ W2 projection                                    │
│     │   └─ Residual connection                                  │
│     ├─ Final RMS Norm                                           │
│     └─ Output projection (Wcls) → Logits                        │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. AUTO-REGRESSIVE GENERATION                                   │
├─────────────────────────────────────────────────────────────────┤
│ Repeat until EOS or max_tokens:                                 │
│   ├─ Model.forward(prev_token, position, state)                 │
│   ├─ Extract logits (vocab_size)                                │
│   ├─ Sampler.sample(logits) → next_token                        │
│   ├─ Tokenizer.decode([next_token]) → text                      │
│   ├─ Callback(text) → Stream to user                            │
│   └─ Update position                                            │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. METRICS & CLEANUP                                            │
├─────────────────────────────────────────────────────────────────┤
│ LastRunMetrics.report()                                         │
│   ├─ Tokens/second                                              │
│   ├─ Total latency                                              │
│   └─ KV cache usage                                             │
└─────────────────────────────────────────────────────────────────┘
```

### CPU vs GPU Execution Paths

#### CPU Path
```
InferenceEngine.generateTokensLlama()
    ↓
InferenceCore.forwardJava(token, pos, state)
    ↓
StandardWeights (FloatTensor[])
    ↓
Java operations (loops, arrays)
```

#### GPU Path
```
InferenceEngine.generateTokensGPU()
    ↓
TornadoVMMasterPlan.execute(taskIndex)
    ↓
TornadoVM Task Graph (layer-based)
    ↓
TornadoWeights (FloatArray[])
    ↓
GPU Kernels (parallel execution)
```

---

## GPU Acceleration

### TornadoVM Architecture

```
┌────────────────────────────────────────────────────────────┐
│ Java Application Code                                      │
│ (LlamaApp, InferenceCore, Model)                          │
└────────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────────┐
│ TornadoVM API Layer                                        │
│ (TaskGraph, FloatArray, DataTransfers)                    │
└────────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────────┐
│ TornadoVM JIT Compiler                                     │
│ (Java Bytecode → Intermediate Representation)             │
└────────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────────┐
│ Backend Code Generation                                    │
│ ├─ PTX (NVIDIA CUDA)                                       │
│ ├─ OpenCL (Cross-platform)                                │
│ └─ SPIR-V (Vulkan, future)                                │
└────────────────────────────────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────────┐
│ GPU Device Execution                                       │
│ (Parallel kernel execution on GPU cores)                  │
└────────────────────────────────────────────────────────────┘
```

### GPU Memory Management

**Weight Storage**:
```java
// Allocate on GPU (read-only)
FloatArray gpuWeights = new FloatArray(size);
gpuWeights.init(cpuWeights);

// Transfer to device (one-time)
taskGraph.transferToDevice(DataTransferMode.FIRST_EXECUTION, gpuWeights);
```

**Activation Buffers**:
```java
// Allocate on GPU (read-write)
FloatArray gpuActivations = new FloatArray(size);

// Transfer results back to host (when needed)
taskGraph.transferToHost(DataTransferMode.EVERY_EXECUTION, gpuLogits);
```

### GPU Kernel Optimizations

#### 1. RMS Normalization (Two-Phase Reduction)
```java
// Phase 1: Parallel sum of squares
parallel_for (i in 0..dim) {
    local_sum += x[i] * x[i]
}
// Phase 2: Normalize
rms = sqrt(local_sum / dim + eps)
parallel_for (i in 0..dim) {
    output[i] = x[i] / rms * weight[i]
}
```

#### 2. Matrix Multiplication (BLAS-Optimized)
```java
// Utilize GPU's native GEMM operations
output = matmul(weights, input)  // Offloaded to cuBLAS/clBLAS
```

#### 3. Attention (Fused Operations)
```java
// Fused: Q@K^T + softmax + @V
parallel_for (head in 0..n_heads) {
    scores = matmul(Q[head], K[head].T) / sqrt(head_dim)
    attention = softmax(scores)
    output[head] = matmul(attention, V[head])
}
```

---

## Build System

### Maven Configuration (pom.xml)

**Project Metadata**:
```xml
<groupId>io.github.beehive-lab</groupId>
<artifactId>gpu-llama3</artifactId>
<version>0.2.2</version>
<packaging>jar</packaging>
```

**Compiler Configuration**:
```xml
<source>21</source>
<target>21</target>
<compilerArgs>
    <arg>--enable-preview</arg>
    <arg>--add-modules</arg>
    <arg>jdk.incubator.vector</arg>
</compilerArgs>
```

**Dependencies**:
```xml
<dependency>
    <groupId>tornado</groupId>
    <artifactId>tornado-api</artifactId>
    <version>1.1.2-dev</version>
</dependency>
<dependency>
    <groupId>tornado</groupId>
    <artifactId>tornado-runtime</artifactId>
    <version>1.1.2-dev</version>
</dependency>
```

**Plugins**:
1. **maven-compiler-plugin**: Java 21 with preview features
2. **maven-shade-plugin**: Creates executable fat JAR
3. **maven-source-plugin**: Generates source JAR
4. **maven-javadoc-plugin**: Generates API documentation
5. **maven-gpg-plugin**: GPG signing for Maven Central
6. **central-publishing-maven-plugin**: Maven Central deployment

### Makefile Targets

```makefile
make all      # Clean + Package (skip tests)
make clean    # Maven clean
make package  # Maven package (skip tests)
make help     # Display available targets
```

### Build Process

```bash
# 1. Set up environment
source set_paths

# 2. Build project
make all

# 3. Output
target/gpu-llama3-0.2.2.jar     # Executable fat JAR
target/gpu-llama3-0.2.2-sources.jar
target/gpu-llama3-0.2.2-javadoc.jar
```

---

## Performance Characteristics

### Benchmark Results (FP16 Models)

| Hardware | 1B Model | 3B Model | Backend |
|----------|----------|----------|---------|
| **RTX 5090** | 96.65 tok/s | 47.68 tok/s | PTX (CUDA) |
| **RTX 4090** | 66.07 tok/s | 35.51 tok/s | PTX (CUDA) |
| **RTX 3070** | 52.00 tok/s | 22.96 tok/s | PTX (CUDA) |
| **M4 Pro** | 16.77 tok/s | 8.56 tok/s | OpenCL |

### Initialization Overhead

| Phase | Time | Description |
|-------|------|-------------|
| **Model Loading** | 500-2000ms | GGUF parsing, weight deserialization |
| **GPU Plan Creation** | 100-500ms | TornadoVM task graph construction |
| **JIT Compilation** | 500-2000ms | Java → GPU kernel compilation |
| **Weight Transfer** | 200-1000ms | Host → Device memory transfer |
| **Total Warmup** | ~1-5s | One-time initialization |

### Memory Usage

| Component | Memory |
|-----------|--------|
| **Model Weights** (1B FP16) | ~2GB |
| **Model Weights** (3B FP16) | ~6GB |
| **KV Cache** (1024 ctx) | ~100-500MB |
| **Activation Buffers** | ~50-200MB |
| **Total (1B)** | ~2.5-3GB |
| **Total (3B)** | ~7-8GB |

### Optimization Techniques

1. **KV Cache**: Eliminates recomputation of past keys/values
2. **Memory-Mapped Loading**: Efficient large file access
3. **Quantization**: FP16 reduces memory by 50% vs FP32
4. **Parallel Reduction**: Two-phase normalization
5. **Fused Kernels**: Reduce kernel launch overhead
6. **Device-to-Device**: Minimize PCI-E transfers
7. **Pre-Compilation**: Warmup JIT compilation before inference

---

## Supported Models

### Model Architectures

| Model | Sizes | Context Length | Notes |
|-------|-------|----------------|-------|
| **Llama 3** | 1B, 3B, 8B | 8192 | Primary focus |
| **Mistral** | 7B | 8192 | Llama-compatible architecture |
| **Qwen2** | 0.5B, 1.5B | 32768 | Different attention mechanism |
| **Qwen3** | 0.6B, 1.7B, 4B, 8B | 32768 | Latest Qwen architecture |
| **Phi-3** | Mini (3.8B) | 4096 | Microsoft's small model |
| **DeepSeek-R1-Distill** | 1.5B | 8192 | Reasoning-focused distilled model |

### Quantization Support

| Format | Precision | Memory Savings | Accuracy | Status |
|--------|-----------|----------------|----------|--------|
| **FP32** | 32-bit | Baseline | Highest | Supported |
| **FP16** | 16-bit | 50% | High | **Recommended** |
| **Q8_0** | 8-bit | 75% | Good | Experimental |
| **Q4_0** | 4-bit | 87.5% | Fair | Experimental |

### GGUF Format Compatibility

- **Version**: GGUF v2, v3
- **Ecosystem**: Compatible with llama.cpp, Ollama
- **Metadata**: Model name, vocab size, architecture parameters
- **Tensors**: Named tensors with type information

---

## Command-Line Interface

### Basic Usage

```bash
# Instruct mode (single prompt)
llama-tornado --model model.gguf --prompt "Explain quantum computing" --gpu

# Interactive mode (chat)
llama-tornado --model model.gguf --interactive --gpu

# CPU execution
llama-tornado --model model.gguf --prompt "Hello" --cpu
```

### Key Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model PATH` | Path to GGUF model file | Required |
| `--prompt TEXT` | Input prompt (instruct mode) | None |
| `--interactive` / `-i` | Enable chat mode | false |
| `--temperature FLOAT` | Sampling temperature (0.1-2.0) | 0.7 |
| `--top-p FLOAT` | Nucleus sampling threshold | 0.9 |
| `--max-tokens INT` | Maximum tokens to generate | 256 |
| `--gpu` / `--opencl` / `--ptx` | Enable GPU acceleration | false |
| `--gpu-memory SIZE` | GPU memory limit | 7GB |
| `--stream` | Stream tokens as generated | true |
| `--seed INT` | Random seed | Random |

### Debug Options

| Option | Description |
|--------|-------------|
| `--verbose-init` | Show initialization timing |
| `--debug` | TornadoVM debug mode |
| `--print-kernel` | Print generated GPU kernels |
| `--profiler` | Enable TornadoVM profiler |

---

## Future Roadmap

### Planned Features

1. **Metal Backend**: Native Apple Silicon support (replacing deprecated OpenCL)
2. **AMD Optimization**: Improved performance on AMD GPUs
3. **Batch Inference**: Process multiple prompts in parallel
4. **Speculative Decoding**: Draft-verify token generation
5. **KV Cache Quantization**: Reduce memory for long contexts
6. **Additional Models**: Gemma, Phi-4, Llama 4
7. **Flash Attention**: Memory-efficient attention mechanism

### Performance Targets

- **Parity with llama.cpp**: Match C++ performance in Java
- **Batch Throughput**: 10x improvement for multi-request scenarios
- **Context Extension**: Support 128K+ context lengths
- **Quantization**: Q4_0/Q4_1 as primary format (4x memory reduction)

---

## Key Design Principles

### 1. Separation of Concerns
- Model abstraction decoupled from execution (CPU/GPU)
- Weights separate from compute kernels
- State management isolated from forward pass logic
- **Allocation logic separated from state classes** (StateFieldAllocator pattern)

### 2. Extensibility
- Factory pattern for new model types
- Strategy pattern for sampling algorithms
- Adapter pattern for weight storage
- **Template Method pattern for model loaders** (AbstractModelLoader)
- **Abstract Factory for state allocation** (StateFieldAllocator)

### 3. Code Quality & Maintainability
- **DRY Principle**: Eliminated ~400 lines of duplicated code
  - ModelLoader refactoring: ~60% code reduction
  - StateFieldAllocator pattern: ~60-66% reduction per state class
- **Single Responsibility**: Each allocator/loader focuses on one model
- **Centralized Logic**: Common patterns in base classes

### 4. Performance
- Memory-mapped file loading (lazy loading)
- KV cache to eliminate recomputation
- GPU acceleration via TornadoVM
- Quantization support

### 5. Developer Experience
- Pure Java (no C/C++ build complexity)
- Maven-based build (standard tooling)
- LangChain4j integration (ecosystem compatibility)
- **Easy to add new models**: Implement 7-9 methods in allocator + loader

### 6. Portability
- Multi-backend GPU support (OpenCL, PTX, SPIR-V)
- CPU fallback when GPU unavailable
- Cross-platform (Linux, macOS, Windows)

---

## Integration Points

### LangChain4j Integration

```java
// GPULlama3.java as official LangChain4j model provider
ChatLanguageModel model = GPULlama3ChatModel.builder()
    .modelPath("/path/to/model.gguf")
    .temperature(0.7)
    .maxTokens(256)
    .useGPU(true)
    .build();

String response = model.generate("Explain AI");
```

### Docker Support

```dockerfile
FROM tornadovm/tornadovm:latest
COPY target/gpu-llama3-0.2.2.jar /app/
ENTRYPOINT ["java", "-jar", "/app/gpu-llama3-0.2.2.jar"]
```

---

## Contributing

### Adding a New Model Architecture

**Simplified Process** (with new patterns):

1. **Create model package**: `src/main/java/org/beehive/gpullama3/model/newmodel/`

2. **Core Model Classes**:
   - `NewModel.java` (extends `AbstractModel`)
   - `NewModelConfiguration.java` (implements `Configuration`)

3. **Model Loader** (extends `AbstractModelLoader`):
   - `NewModelLoader.java` - Implement 7 abstract methods:
     - `loadVocabulary(metadata)` → Load vocab
     - `createTokenizer(metadata, vocab)` → Create tokenizer
     - `createConfiguration(metadata)` → Parse config
     - `precomputeRopeFrequencies(config)` → Calculate RoPE
     - `createStandardWeights(...)` → CPU weights
     - `createTornadoVMWeights(...)` → GPU weights
     - `createModel(...)` → Instantiate model

4. **State Classes**:
   - `NewModelState.java` - Minimal class (10-20 lines)
   - `NewModelStateFieldAllocator.java` - Implement 9 abstract methods:
     - `getDimX()`, `getDimXb()`, `getDimXb2()`, etc.
     - All allocation logic centralized here

5. **Weights Classes**:
   - `NewModelStandardWeights.java` (CPU)
   - `NewModelTornadoWeights.java` (GPU)

6. **Supporting Classes**:
   - `NewModelTokenizer.java` (tokenization)
   - `NewModelChatFormat.java` (prompt formatting)
   - `NewModelTornadoVMLayerPlanner.java` (GPU execution plan)

7. **Update Factory**:
   - Add entry to `ModelType.java` enum

**Benefits of New Pattern**:
- **Less boilerplate**: State classes reduced from 70-112 lines to 20-54 lines
- **Clear structure**: Allocation logic separated from state
- **Easier debugging**: All dimensions calculated in one place
- **Type safety**: Abstract methods ensure all dimensions are defined

---

## References

### Documentation
- **Main README**: `/README.md`
- **TornadoVM Optimizations**: `/docs/TORNADOVM_TRANSFORMER_OPTIMIZATIONS.md`
- **Roadmap**: `/docs/GPULlama3_ROADMAP.md`
- **Contributing Guide**: `/CONTRIBUTING.md`

### External Links
- **Repository**: https://github.com/beehive-lab/GPULlama3.java
- **TornadoVM**: https://github.com/beehive-lab/TornadoVM
- **LangChain4j**: https://github.com/langchain4j/langchain4j
- **GGUF Format**: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

---

**End of Architecture Documentation**
