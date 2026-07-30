# Target Architecture

**Status: proposal.** Nothing here exists in the repository yet. Package and type
names are placeholders and will be settled by the ADRs and by the code that
implements them. The *layering and dependency direction* is the part meant to be
stable; the names are not.

Terms used here are defined in [`terminology.md`](terminology.md).

## Layering

```
                Integrations
      (CLI, OpenAI server, LangChain4j, Quarkus, user applications)
                     |
                     v
            Public API and generation
       (LocalModels, LocalModel, GenerationSession, stop
              conditions, streaming, detokenization)
                     |
                     v
                   Engine
     (LLMEngine, Scheduler, admission, KvCacheManager,
              BlockPool, PrefixCache, serving metrics)
                     |
                     v
              Models and sessions
     (loaded model: architecture + configuration + weights;
        session: sequence position + KV lease; provider SPI)
                     |
                     v
         Inference programs and operations
     (backend-neutral program description, program components,
        reusable operation vocabulary — incl. Sample/ArgMax)
                     |
                     v
            Runtime, tensors and state
     (tensor descriptors, DataType, buffer abstractions, state
      layout, memory planning, execution policy, metrics sink)
                     |
                     v
                 Backend SPI
     (Backend, DeviceSelector, CompiledProgram, Invocation,
             buffer lifetimes, capacity query)
                     |
                     v
              TornadoVM backend
     (TaskGraph, ImmutableTaskGraph, GridScheduler,
        TornadoExecutionPlan, kernel methods, worker grids)
                     |
                     v
        CUDA (PTX) / OpenCL / SPIR-V devices
```

Two edges run against the arrows, both deliberately:

- the **metrics sink** lives in the runtime layer and is written by backends, read by the
  engine and API — the one designed upward-looking seam
  ([Rule 17](dependency-rules.md#rule-17--metrics-flow-bottom-to-top-by-design));
- **KV storage** is owned by the engine's cache manager and leased downward to sessions
  ([Rule 7](dependency-rules.md#rule-7--the-kv-cache-is-never-global-model-state-storage-is-managed-and-leased)).

Dependencies point **downward only**. A layer may depend on the layer below it and on
layers further below; it must never depend on a layer above it. Sibling packages within
a layer should depend on each other only through explicit interfaces.

The enforceable form of these arrows is in
[`dependency-rules.md`](dependency-rules.md).

### Why the backend SPI sits below runtime abstractions

Tensors, state and program descriptions must be expressible without knowing which
backend will run them — that is what makes an inference program backend-neutral. The
backend SPI is the *narrow* interface through which those neutral descriptions become
executable. The TornadoVM backend implements the SPI; nothing above the SPI knows it
exists.

## Engine tier

The tier that owns work **across** sequences. Without it the architecture can only
express "one request occupies one session occupies the device", which is the throughput
ceiling the project is removing.

```java
public interface LLMEngine extends AutoCloseable {
    RequestHandle addRequest(GenerationRequest request);   // non-blocking admission
    StepResult step();                                     // one batched iteration
}
```

Components:

| Component | Owns |
| --- | --- |
| `Scheduler` | Admission, batch composition, preemption |
| `KvCacheManager` | Block storage, leases, eviction |
| `BlockPool` | The persistent pooled array backing KV blocks |
| `PrefixCache` | Prefix-keyed shared blocks, refcounting |
| serving metrics | TTFT, queue wait, occupancy, block utilization, admitted/rejected |

**Device concurrency comes from batching inside one compiled program**, not from running
several plans in parallel. Concurrent independent `TornadoExecutionPlan`s *are* supported
and tested, but device buffers are per task graph, so two plans over the same weights hold
two device copies — roughly 3.4 GB duplicated per concurrent session on a 3B-Q8 model.
Batching is therefore an economic choice, not a workaround for a missing API. See
[capability C2](tornadovm-capabilities.md#c2--device-buffers-are-per-task-graph) and
[ADR-006](decisions/ADR-006-engine-tier.md).

The seed for this tier already exists as `bench/BatchedDecodeEngine` (PR #129), which
implements continuous batching, paged KV and prefix caching and measures them.

## High-level API

Aimed at users who want to run a model. One import block, no GPU concepts.

```java
try (LocalModel model = LocalModels.load(modelPath);
     GenerationSession session = model.newSession()) {

    session.generate(GenerationRequest.builder()
            .prompt("...")
            .maxNewTokens(256)
            .onToken(System.out::print)
            .build());
}
```

Guarantees this API is meant to make:

- a loaded model is immutable and safe to share between threads;
- a session is single-sequence and not thread-safe, and says so;
- both are `AutoCloseable` because both may hold device memory;
- no TornadoVM type, GGUF type or CLI type appears in any signature.

See [`public-api.md`](public-api.md) for the sketch.

## Advanced API

Aimed at users who need control: device choice, execution policy, program reuse,
compiled-program sharing, memory behaviour, diagnostics.

```java
Backend backend = Backends.select(DeviceSelector.preferGpu());

InferenceProgram program  = architecture.createDecodeProgram(configuration);
CompiledProgram compiled  = backend.compile(program, compileOptions);

try (GenerationSession session = model.newSession(compiled)) { ... }
```

Still backend-neutral. Tornado-specific knobs (CUDA graphs, pre-compilation,
scheduler selection) belong in an explicitly optional, clearly named
Tornado-specific extension of the backend API that a user opts into — not in
`Backend` itself.

## Model providers

Adding a model architecture should mean *adding* code, not *editing* central switches
(see [`current-architecture.md`](current-architecture.md#p6--central-per-family-switches)).

Proposed shape:

```java
public interface ModelProvider {

    /** Can this provider load the given source? Inspect metadata, do not guess by filename. */
    boolean supports(ModelSource source);

    /** Load configuration, weights, tokenizer and architecture description. */
    LoadedModel load(ModelSource source, ModelOptions options, Backend backend);
}
```

Discovered through `ServiceLoader`. `LlamaModelProvider`, `Qwen3ModelProvider` and so
on register themselves. `ModelType` remains as an internal detail during migration.

Open question: whether the architecture description (which produces inference
programs) is a separate SPI from the loader, or the same one. Programs are needed by
the backend; loading is needed by the format layer. They may want to be separate.

## Backend providers

```java
public interface Backend extends AutoCloseable {

    BackendId id();
    List<Device> devices();

    CompiledProgram compile(InferenceProgram program, CompileOptions options);

    SessionState allocateState(StateLayout layout, Device device);
}
```

Also `ServiceLoader`-discovered. Expected implementations:

- **TornadoVM backend** — the primary accelerated backend. Owns all
  `uk.ac.manchester.tornado.*` usage.
- **Plain-Java CPU backend** — today's `InferenceCore` path, behind the same SPI.

PTX/CUDA, OpenCL and SPIR-V are *devices and device backends of TornadoVM*, not
separate GPULlama backends. There must not be one copy of model logic per device
backend.

## Session lifecycle

```
LocalModels.load(path, options)
    → ModelProvider.supports / load
    → weights resident (host and/or device)
    → LoadedModel (immutable, thread-safe, shared)

LoadedModel.newSession(sessionOptions)
    → resolve execution policy
    → obtain or reuse CompiledProgram for (architecture, policy, backend, device)
    → acquire a KV lease from the engine's KvCacheManager
    → allocate invocation buffers, sized by context length
    → GenerationSession (single sequence, not thread-safe)

session.generate(request)
    → prefill: ingest prompt tokens into the leased KV blocks
    → decode loop:
         bind invocation (input token, position, session state, outputs)
         invoke compiled program
         sample → emit → check stop conditions
    → GenerationResult

session.close()   → release the KV lease and invocation buffers
model.close()     → release weights and cached compiled programs
```

Key points:

- the compiled program is created **before** the decode loop and reused for every
  token — this is already true today and must stay true;
- compiled programs are keyed by (architecture, configuration shape, execution
  policy, backend, device) and can be shared between sessions of the same model;
- KV **storage** is owned by the cache manager; a session holds a **lease** — a block
  table referencing blocks it does not own
  ([ADR-005](decisions/ADR-005-kv-cache-ownership-and-leases.md));
- invocation buffers are per-session and never shared;
- closing a session must not invalidate the model or other sessions, and must not free
  blocks another lease still references.

**Concurrency — resolved.** Sessions are independent objects; several may exist against
one loaded model. Device-level concurrency is achieved by **batching many sequences into
one invocation of one compiled program**, driven by the engine tier — not by running one
plan per session. Concurrent plans are supported by TornadoVM and tested, but each task
graph owns its own device buffers, so per-session plans duplicate the weights
([capability C2](tornadovm-capabilities.md#c2--device-buffers-are-per-task-graph)).
A session used without an engine runs one sequence at a time, which is the simple path.

## Logical versus compiled programs

```
  InferenceProgram                     CompiledProgram
  (backend-neutral)                    (backend-specific)

  ordered program components    ──▶    TornadoVM backend:
  operations + weight refs      compile   ImmutableTaskGraph list
  + program signature                     + GridScheduler
                                          + TornadoExecutionPlan
  no device handles                       + device buffers
  no TornadoVM types
  no GGUF types                        CPU backend:
                                          bound method handles / direct calls
```

- One program may be compiled by several backends.
- One compiled program is invoked many times; **compilation never happens per token**.
- An invocation binds inputs, outputs and session state to a compiled program. It
  carries no compilation work.
- The existing `ForwardPlan` hierarchy is the TornadoVM backend's compiled-program
  internals, reached only through the backend.

See [ADR-002](decisions/ADR-002-program-and-compiled-program.md), including the
recommendation that the first version uses **ordered, composable components rather
than a general graph IR**.

## Reusable operations

An operation vocabulary that both the description layer and the backends agree on:

```
RmsNorm        RoPE           MatVec / MatMul     Attention
Softmax        SwiGLU / SiLU  ResidualAdd         EmbeddingLookup
Quantize / Dequantize         VocabProjection
```

Requirements:

- named and defined once, independent of model family;
- parameterized by configuration, not hard-coded per architecture;
- each backend maps operations to its own execution form (TornadoVM: kernel methods
  and task-graph entries; CPU: `InferenceCore`-style Java);
- a model architecture assembles a program out of operations; it does not write
  device code.

This is the layer that today is duplicated between `InferenceCore` (CPU) and
`tornadovm/kernels` + `tornadovm/layers` (GPU). Unifying the *vocabulary* does not
mean unifying the implementations.

**Non-goal:** this is not an operator set for arbitrary computation, and not a graph
IR to optimize over. It is the set of operations transformer inference needs.

## Future inference use cases

The layering must not assume generation. Concretely:

| Use case | What it needs | What it must not be forced to have |
| --- | --- | --- |
| Embeddings | encoder pass, pooled output | KV cache, sampler, generation loop |
| Classification | encoder pass, head, label mapping | tokenizer detok, streaming |
| Reranking | paired input encoding, score output | autoregressive decode |
| Encoder-only | bidirectional attention, no causal mask | causal masking, prefill/decode split |
| Encoder–decoder | two programs, cross-attention | single-program assumption |
| Vision transformers | image preprocessing, patch embedding | text tokenizer |
| Multimodal | multiple encoders feeding one decoder | single input modality |

Design consequence: `LoadedModel`, `InferenceProgram`, `CompiledProgram`, `Backend`
and the tensor/state abstractions must be usable without a tokenizer, without a KV
cache and without a generation loop. Those are capabilities a *text-generation* model
adds, not requirements of the core. This is
[dependency rule 14](dependency-rules.md#rule-14--core-abstractions-do-not-assume-generation).

## Likely package structure

**Proposal.** Illustrative, under the existing root package. No renames are proposed
as an early step.

```
org.beehive.gpullama3
├── api/                    LocalModels, LocalModel, GenerationSession,
│                           GenerationRequest/Result, ModelOptions, SessionOptions
├── generation/             generation loop, sampling, stop conditions, streaming
├── model/                  loaded model, architecture descriptions, configuration
│   └── provider/           ModelProvider SPI + registry
├── program/                InferenceProgram, program components, program signature
│   └── op/                 operation vocabulary
├── runtime/                session state, state layout, tensor descriptors,
│                           DataType, execution policy, memory planning
├── backend/                Backend SPI, Device, DeviceSelector,
│   │                       CompiledProgram, Invocation, CompileOptions
│   ├── cpu/                plain-Java backend (today's InferenceCore path)
│   └── tornado/            *** the only package allowed to import TornadoVM ***
│       ├── kernels/        today's tornadovm/kernels
│       ├── layers/         today's tornadovm/layers
│       └── plan/           today's tornadovm/plan (ForwardPlan et al.)
├── format/                 GGUF reader, GGML types, format→runtime mapping
│   └── gguf/
├── tokenizer/              unchanged
└── integration/            CLI, OpenAI server, LangChain4j adapter
```

Note the two moves that carry the most weight:

- `tornadovm/**` becomes `backend/tornado/**` — this is what makes the ArchUnit rule
  expressible;
- GGUF moves out of `tensor/` into `format/gguf/`, leaving runtime tensor concepts
  format-free (see [ADR-004](decisions/ADR-004-tensor-and-format-separation.md)).

Neither is proposed as a first step. See
[`migration-roadmap.md`](migration-roadmap.md).

## Likely Maven module structure

**Proposal, and explicitly a later step — not an immediate repository-wide split.**

A plausible end state:

| Module | Contents | Depends on TornadoVM? |
| --- | --- | --- |
| `gpullama3-api` | public API types, no implementation | no |
| `gpullama3-core` | model, program, runtime, backend SPI, format | no |
| `gpullama3-backend-tornado` | TornadoVM backend | **yes** |
| `gpullama3-backend-cpu` | plain-Java backend | no |
| `gpullama3-integration-*` | CLI, server, LangChain4j | no |

The value of splitting is that `gpullama3-core` *cannot* compile against TornadoVM,
making the central dependency rule structural instead of test-enforced.

The cost is real: the project is one module today with a shaded jar, native-image
considerations, release automation and a `llama-tornado` launcher that all assume the
current shape. A multi-module split should happen only after the package boundaries
hold and the ArchUnit allowlist is empty or nearly so — otherwise the split just
relocates the problem.

**Recommendation:** treat the package boundary as the near-term goal and the module
split as a later confirmation of it.

## Open questions

1. Are inference programs ordered component sequences or general graphs? (Recommended:
   ordered first — [ADR-002](decisions/ADR-002-program-and-compiled-program.md).)
2. How are compiled programs cached and keyed, and who owns their lifetime — the
   loaded model, the backend, or an explicit cache?
3. Can multiple sessions run concurrently on one device, or is invocation serialized
   per compiled program? ([ADR-001](decisions/ADR-001-model-session-separation.md))
4. Does the CPU path become a real `Backend`, or stay a separate simpler path?
5. Where does execution policy live — model options, session options, or both?
6. How much of `State` becomes backend-owned (device buffers) versus
   runtime-described (layout)?
7. Do architecture descriptions and model loaders share one SPI or two?
8. Is a shaped tensor descriptor needed, given that `FloatTensor` is deliberately
   shapeless today?
