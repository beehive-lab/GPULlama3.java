# Current Architecture

**Status:** descriptive. This document records the repository as inspected on the
`refactor/framework-abstractions` branch. Where it disagrees with the code, the code
is correct and this document should be fixed.

GPULlama3.java is a working GPU-accelerated LLM engine. It runs several transformer
families on CPU and on GPU through TornadoVM, with a compile-once/execute-many GPU
path. The observations below are about *where responsibilities sit*, not about
whether the engine works.

## Build shape

- Single Maven module, `artifactId` `gpu-llama3`, packaging jar.
- Single source root: `src/main/java/org/beehive/gpullama3/`.
- Single test source file at time of writing:
  `src/test/java/org/beehive/gpullama3/model/format/ToolCallParserUtilsTest.java`.
- No ArchUnit dependency.
- TornadoVM (`tornado-api`, `tornado-runtime`) is a direct compile dependency of the
  whole module.

## Package map

```
org.beehive.gpullama3
├── LlamaApp                       CLI entry point
├── Options                        CLI options record (also sets system properties)
├── auxiliary/                     Pair, Tuple2, Timer, Parallel, RunMetrics
│   └── metrics/                   metrics renderers + RunMetricsSnapshot
├── bench/                         LlamaBench
├── inference/
│   ├── InferenceCore              CPU forward passes, per model family
│   ├── InferenceCoreWithPrefillDecode
│   ├── InferenceCoreBatchPrefillDecode
│   ├── InferenceEngine            generation loops (CPU + GPU)
│   ├── InferenceEngineWithPrefillDecode
│   ├── InferenceEngineWithBatchPrefillDecode
│   ├── operation/                 RoPE
│   ├── sampler/                   Sampler, CategoricalSampler, ToppSampler
│   ├── state/                     State + Llama/Qwen2/Qwen3/Phi3/Granite/Devstral states
│   └── weights/                   Weights; standard/ (CPU) and tornado/ (GPU) variants
├── model/
│   ├── Model, AbstractModel, Configuration, ModelType
│   ├── format/                    ChatFormat per family + tool-call parsing
│   ├── loader/                    ModelLoader, AbstractModelLoader, per-family loaders
│   └── llama|mistral|devstral|qwen2|qwen3|phi3|granite/   model + configuration
├── server/                        OpenAIServer, InferenceService, Json
├── tensor/
│   ├── GGUF, GGMLType, GGMLTensorEntry, MetadataValueType, Float16
│   ├── standard/                  FloatTensor + FP32/FP16/Q4_0/Q4_K/Q5_K/Q6_K/Q8_0
│   └── tornado/                   TornadoTensor + FP32/FP16/Q8_0 variants
├── tokenizer/                     Tokenizer, Vocabulary, per-family tokenizers
└── tornadovm/
    ├── TornadoVMMasterPlan (+ SingleToken, PrefillDecode, BatchPrefillDecode)
    ├── TensorCoreSupport, GPULLlama3TypeException
    ├── kernels/                   Java kernel methods (Transformer*, Qwen2/3, Phi3, Granite)
    ├── layers/                    TaskGraph builders (Activation, TransformerLayer, Logits)
    │   └── type/{fp16,q8_0}/{,decode,prefill}/   per-quantization layer graph builders
    ├── plan/                      ForwardPlan, ExecutionMode, ForwardPlanFactory
    │   ├── components/            per model+quant plan component providers
    │   └── layout/                task-graph index layouts
    ├── scheduling/                SchedulerDetectionService, SchedulerType, WorkerGridFactory
    └── utils/                     FloatArrayUtils
```

## Main execution flows

### Startup and load

```
LlamaApp.main
  → Options.parseOptions(args)              (also publishes llama.* system properties)
  → ModelLoader.loadModel(options)
        → GGUF.loadGGUFMetadata(path)
        → detectModelType(metadata)         (string matching on "general.name")
        → ModelType.<FAMILY>.loadModel(...) (enum-dispatched loader)
              → AbstractModelLoader template method
                    → Configuration, Vocabulary, Tokenizer, ChatFormat
                    → Weights: StandardWeights (CPU) or TornadoWeights (GPU)
  → Sampler.createSampler(model, options)
  → model.runInteractive(...) | model.runInstructOnce(...)
```

### CPU generation

```
Model.runInstructOnce (default method on the Model interface)
  → Model.generateTokens(...)               (per-model override, e.g. Llama)
        → InferenceEngine.generateTokensLlama
              → InferenceCore.forwardJava*(model, state, token, position)
              → Sampler.sampleToken(logits)
```

`Llama.generateTokens` selects between `InferenceEngine`,
`InferenceEngineWithPrefillDecode` and `InferenceEngineWithBatchPrefillDecode` by
reading the static `llama.withPrefillDecode` property and
`TornadoVMMasterPlan.PREFILL_BATCH_SIZE`.

### GPU generation

```
Model.runInstructOnce / runInteractive
  → TornadoVMMasterPlan.initializeTornadoVMPlan(state, model)     ← once per run
        → chooses SingleToken | PrefillDecode | BatchPrefillDecode
              from llama.withPrefillDecode and llama.prefillBatchSize
        → ForwardPlanFactory.create*(quantization, state, model)
              → *PlanComponents (per model + quantization)
                    → layers/*  build TaskGraph objects
              → ForwardPlan holds List<ImmutableTaskGraph> + GridScheduler
        → new TornadoExecutionPlan(graphs), withPreCompilation()
        → model.setTornadoVMPlan(plan)
  → Model.generateTokensGPU(...)
        → InferenceEngine.generateTokensGPULlama
              → InferenceCore.forwardTornadoVM(...)
                    → plan.tornadoVMForwardDecode(position)
                          → executes graph[0], graphs[1..N], graph[N+1]
                            with the stored GridScheduler
              → sampler, or state.sampledToken when device sampling is active
  → plan.freeTornadoExecutionPlan()
```

The task graphs and the execution plan are built once at initialization and executed
per token. This property is important and must be preserved.

## Current responsibilities

### Model

`model.Model` is an interface providing `configuration()`, `tokenizer()`, `weights()`,
`chatFormat()`, `getModelType()`, `createNewState()`, `forward(...)`,
`generateTokens(...)`, `generateTokensGPU(...)`, plus `tornadoVMPlan()` /
`setTornadoVMPlan(...)`.

It also carries three `default` methods that implement the *application* loop:
`runInteractive(Sampler, Options)`, `runInstructOnce(Sampler, Options)` and
`runInstructOnceLangChain4J(Sampler, Options, Consumer<String>)`. These read from
`System.in`, write to `System.out`/`System.err`, build the conversation token list,
own the GPU plan lifecycle and print metrics.

`model.AbstractModel` holds `tokenizer`, `weights`, `chatFormat` and a mutable
`TornadoVMMasterPlan plan` field.

Per-family models (`Llama`, `Mistral`, `Devstral`, `Qwen2`, `Qwen3`, `Phi3`,
`Granite`, `DeepSeekR1Qwen`) extend `AbstractModel`, hold their `Configuration`, and
delegate `forward` to the matching `InferenceCore.forwardJava*` method.

### Configuration

`model.Configuration` is an interface of transformer hyperparameters (`dim`,
`hiddenDim`, `numberOfLayers`, `numberOfHeads`, `numberOfKeyValueHeads`,
`vocabularySize`, `contextLength`, `rmsNormEps`, `ropeTheta`, `headSize`, `kvDim`,
`kvMul`, `quantization`). Per-family records implement it and add family-specific
fields. It is immutable and free of TornadoVM imports.

### Inference

`InferenceCore` holds the CPU forward passes as static methods, one per family
(`forwardJava`, `forwardJavaQwen2`, `forwardJavaQwen3`, `forwardJavaPhi3`,
`forwardJavaDevstral`, `forwardGranite`), plus `forwardTornadoVM` which delegates to
the GPU plan. `InferenceEngine` holds the generation loops. Two parallel copies exist
for the prefill/decode and batch-prefill/decode variants
(`InferenceCoreWithPrefillDecode`, `InferenceCoreBatchPrefillDecode`,
`InferenceEngineWithPrefillDecode`, `InferenceEngineWithBatchPrefillDecode`).

### State

`inference.state.State` is an abstract base holding, in one object:

- CPU activation buffers as `FloatTensor` (`x`, `xb`, `xb2`, `hb`, `hb2`, `q`, `k`,
  `v`, `att`, `logits`);
- the KV cache as `FloatTensor[] keyCache` / `valueCache`;
- GPU mirrors as TornadoVM `FloatArray` / `HalfFloatArray` / `IntArray`
  (`wrapX`, `wrapXb`, `wrapQ`, `wrapKeyCache`, `wrapValueCache`, `wrapLogits`, …);
- batch-prefill buffers (`wrapXBatch`, `qkvResultBatch`, `gateUpResultBatch`, …);
- scratch buffers (`temp`, `tempFFN`, `tempLogits`);
- `positionHolder`, `sampledToken`, `latestToken`, `localSize`, `batchsize`.

Subclasses add family-specific fields, e.g. `Qwen3State.wrapAttSplit` and
`Qwen3State.SPLIT_KV`.

### Tensors

Two parallel hierarchies:

- `tensor.standard.FloatTensor` — CPU, Vector API and `MemorySegment` based, with one
  subclass per GGML quantization (`FP16`, `FP32`, `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`,
  `Q8_0`). Documented as "over-simplified, shapeless".
- `tensor.tornado.TornadoTensor` — GPU, wrapping TornadoVM native arrays, with
  `asFloatArray()`, `asHalfFloatArray()`, `asByteArray()`, `getScales()`,
  `getQuants()` throwing `UnsupportedOperationException` when the concrete type does
  not match.

Both expose `GGMLType type()` / `getWeightType()`.

The same `tensor` package also holds the GGUF file format reader (`GGUF`,
`GGMLTensorEntry`, `MetadataValueType`) and `GGMLType`.

### Tokenizer

`tokenizer.Tokenizer` is a small interface (`encode`, `decode`, `getSpecialTokens`,
`isSpecialToken`, `shouldDisplayToken`, `regexPattern`) with per-family
implementations and a shared `Vocabulary`. It has no TornadoVM dependency and no
model dependency. This is the cleanest boundary in the codebase.

### TornadoVM layer

- `kernels/` — the actual compute, written as plain Java methods that TornadoVM
  compiles (`TransformerComputeKernels`, `TransformerComputeKernelsLayered`,
  `TransformerBatchPrefillKernels`, plus `Qwen2Kernels`, `Qwen3Kernels`,
  `Phi3Kernels`, `GraniteKernels`).
- `layers/` — builders that assemble `TaskGraph` objects and register worker grids
  (`AbstractLayer`, `ActivationTaskGraph`, `TransformerLayerTaskGraphs`,
  `AbstractLogitsTaskGraph`, and the `type/fp16` / `type/q8_0` families).
- `plan/` — topology. `ForwardPlan` stores `List<ImmutableTaskGraph>` plus a
  `GridScheduler`. `SingleTokenForwardPlan` (N+2 graphs),
  `PrefillDecodeForwardPlan`, `BatchPrefillDecodeForwardPlan` (2N+3 graphs).
- `plan/components/` — the interfaces a model+quantization pair must implement to
  supply the activation, layer and logits graphs
  (`SingleTokenForwardPlanComponents`, `PrefillDecodeForwardPlanComponents`,
  `BatchPrefillDecodeForwardPlanComponents`), with 14 concrete implementations
  (7 families × 2 quantizations).
- `plan/layout/` — index layouts naming which graph slot does what.
- `TornadoVMMasterPlan` — the runtime-facing interface: `createExecutionPlan()`,
  `forceCopyInReadOnlyData()`, `tornadoVMForwardDecode(position)`,
  `freeTornadoExecutionPlan()`, with three implementations and a static factory.
- `scheduling/` — worker-grid sizing and backend/scheduler detection.

### Server and integrations

`server.InferenceService` wraps one `Model`, one `State` and one
`TornadoVMMasterPlan`, serializing generation behind a lock because the GPU plan and
state are single-tenant. `server.OpenAIServer` exposes it over HTTP.
`Model.runInstructOnceLangChain4J` exists for LangChain4j/Quarkus integration.

## Current forward-plan hierarchy

```
TornadoVMMasterPlan                          (interface, tornadovm/)
├── TornadoVMMasterPlanSingleToken           owns TornadoExecutionPlan, executes per token
├── TornadoVMMasterPlanPrefillDecode
└── TornadoVMMasterPlanBatchPrefillDecode

    each holds a ForwardPlan:

ForwardPlan                                  (abstract, tornadovm/plan/)
│   List<ImmutableTaskGraph> + GridScheduler
├── SingleTokenForwardPlan                   [0]=activation [1..N]=layers [N+1]=logits
├── PrefillDecodeForwardPlan                 N+2 graphs shared by both phases
└── BatchPrefillDecodeForwardPlan            2N+3 graphs

    built by:

ForwardPlanFactory
    switch (GGMLType) → switch (ModelType) → switch (ExecutionMode)

    from:

SingleTokenForwardPlanComponents
└── PrefillDecodeForwardPlanComponents
    └── BatchPrefillDecodeForwardPlanComponents
        implemented by {Llama,Mistral,Devstral,Qwen2,Qwen3,Phi3,Granite}
                     × {FP16, Q8_0} PlanComponents
```

`ForwardPlan` and its subclasses are TornadoVM-specific: they are typed in terms of
`ImmutableTaskGraph` and `GridScheduler`. See
[`terminology.md`](terminology.md#forward-plan).

## Strengths to preserve

1. **Compile-once, execute-many.** Task graphs and the `TornadoExecutionPlan` are
   built during `initializeTornadoVMPlan` and reused for every token. Nothing is
   recompiled per token. Any refactoring must keep this.
2. **Kernels are Java.** The whole compute path is readable and modifiable Java.
   This is the project's distinguishing property.
3. **Plan components are already an extension point.** `*ForwardPlanComponents` is a
   real seam: adding a model family means implementing an interface, not editing the
   layer builders.
4. **Layered task-graph decomposition.** Splitting activation / N layers / logits
   into separate graphs makes prefill/decode reuse and phase skipping possible.
5. **Configuration is clean.** `Configuration` and its per-family records are
   immutable, backend-free hyperparameter carriers.
6. **Tokenizer is clean.** No dependencies on model internals or TornadoVM.
7. **Loader template method.** `AbstractModelLoader` gives every family the same
   load workflow with defined extension points.
8. **`InferenceService` already models the reuse seam** that a `Session` abstraction
   would formalize: one loaded model + one compiled plan + serialized access.

## Observed architectural pressure points

These are observations about coupling, not defects. Each is a place where the current
structure resists the framework direction described in
[`target-architecture.md`](target-architecture.md).

### P1 — `Model` owns the TornadoVM plan

`Model.tornadoVMPlan()` / `setTornadoVMPlan(...)` and the `AbstractModel.plan` field
put a mutable, backend-specific, device-memory-owning object on the model interface.
Consequences: `model` cannot be imported without TornadoVM on the classpath; one
model can hold only one plan; two concurrent sessions would contend on one field.

### P2 — `Model` also owns the application loop

`runInteractive`, `runInstructOnce` and `runInstructOnceLangChain4J` are default
methods on `Model`. They perform console I/O, take a CLI `Options` record, own plan
creation and teardown, and print metrics. Generation policy, transport and model are
in one type.

### P3 — CLI `Options` reaches into the core

`Options` lives in the root package, is consumed by `Model`, `Sampler.createSampler`
and `ModelLoader.loadModel`, and additionally publishes global state via
`System.setProperty("llama.withPrefillDecode", ...)` and `"llama.prefillBatchSize"`.

### P4 — Execution mode is selected through static system properties

`llama.withPrefillDecode`, `llama.prefillBatchSize`, `llama.deviceSample`,
`llama.cudaGraphs` and `use.tornadovm` are read into `static final` fields at class
initialization (`TornadoVMMasterPlan`, `Llama.WITH_PREFILL_DECODE`,
`LogitsFP16Layer.DEVICE_SAMPLE`). This makes execution policy process-global and
fixed at class-load time, and it is why `LlamaApp.guardDeviceSample` must clear a
property *before* the logits graph class is first touched.

### P5 — Three parallel engine and core variants

`InferenceEngine` / `WithPrefillDecode` / `WithBatchPrefillDecode` and the matching
`InferenceCore*` classes are separate copies of the generation loop selected by
static flags. Every model family that gains a new execution mode multiplies against
this.

### P6 — Central per-family switches

Adding an architecture currently requires touching at least:
`ModelType` (enum constant + loader dispatch), `ModelLoader.detectModelType` (string
matching on `general.name`), `ForwardPlanFactory` (two switch branches per
quantization), `InferenceCore` (a new `forwardJava*` method), `Sampler.createSampler`,
and the state/weights hierarchies. `ForwardPlanFactory` switches on `GGMLType`, then
`ModelType`, then `ExecutionMode`, and casts `State` to the family-specific subtype.

### P7 — `State` mixes concerns

One object holds CPU tensors, GPU mirrors, the KV cache, batch-prefill scratch and
per-invocation temporaries, with public mutable fields. Sequence-lifetime data (KV
cache, position) is not separated from invocation-lifetime data (activation and
scratch buffers), so the object cannot be split per session or per call without
splitting the type.

### P8 — Format types are runtime types

`GGMLType` — a GGUF file-format concept — is the type tag on `Weights`,
`FloatTensor` and `TornadoTensor`, and it is the first switch axis in
`ForwardPlanFactory`. `GGUF`, `GGMLTensorEntry` and the runtime tensor hierarchies
share the `tensor` package. There is no runtime-owned `DataType`.

### P9 — TornadoVM imports are spread across non-backend packages

Files importing `uk.ac.manchester.tornado.*` by package (main sources):

| Package | Files |
| --- | --- |
| `tornadovm/**` (all subpackages) | 65 |
| `model/loader` | 8 |
| `inference/state` | 7 |
| `tensor/tornado` | 4 |
| `inference` | 3 |
| `inference/sampler` | 3 |
| `tensor` | 1 |

The 26 files outside `tornadovm/**` are the concrete starting set for the dependency
rule in [`dependency-rules.md`](dependency-rules.md#rule-1--tornadovm-stays-in-the-tornado-backend).
`Sampler` in particular is typed against `FloatArray`, which couples sampling to the
backend's array type.

### P10 — No session abstraction

There is no type representing "one conversation". `InferenceService` approximates it
by holding a `State` plus a plan behind a lock, and resets the KV cache from position
0 for each request. Nothing prevents two callers from sharing one `State`, and
nothing supports two concurrent sequences against one loaded model.

### P11 — Core abstractions assume text generation

`Model` requires `tokenizer()`, `chatFormat()`, `generateTokens(...)`. `State`
always allocates a KV cache. An embedding or classification model would have to
implement or allocate all of it.

### P12 — No telemetry, despite the data being available

The project references none of TornadoVM's profiling API: zero occurrences of
`ProfilerMode`, `getProfilerResult` or `TornadoProfiler` across all main sources. Device
kernel time, host↔device transfer bytes, device memory usage and compile time are all
produced by the runtime on request and discarded.

What exists instead is `auxiliary/RunMetrics`, a static holder that prints to the console,
plus `auxiliary/metrics/` renderers. There is no interface a backend could report into, so
"where does time go on the GPU path" is currently unanswerable from inside the
application. This is the default outcome when metrics have no designed seam — see
[Rule 17](dependency-rules.md#rule-17--metrics-flow-bottom-to-top-by-design).

### P13 — Console I/O in library code

65 `System.out` / `System.err` occurrences across 20 main-source files, including the
generation loop (`Model.runInteractive`, `Model.runInstructOnce`). `pom.xml` declares no
logging dependency, so there is no facade to route through.

For a library whose stated audience is embedders (LangChain4j, Quarkus, servers), printing
to stdout corrupts structured logs and cannot be silenced. Same class of problem as P2.

### P14 — The pinned TornadoVM version predates the capabilities the design assumes

`pom.xml` pins `tornadovm.base.version` 5.0.0. Several capabilities this architecture
depends on arrived later: `FP8Array` in 5.1.0, `BFloat16Array` in 5.2.0 (which PR #120's
BF16 path needs), deterministic generated kernel source, a bytecode buffer sized to the
graph, and two large runtime improvements (53 → 103 tok/s; start-up 11.5 s → 5.2 s).

The gap is load-bearing rather than cosmetic: a position can look grounded against a
development tree and be unreachable from what the project builds. Hence the version floor
is [Phase 0](migration-roadmap.md#phase-0--tornadovm-version-floor) and the capability
ledger carries a minimum-version column.
