# Terminology

**Status:** normative. Other documents in this directory use these terms with exactly
these meanings. Where a term names something that exists today, the current class is
cited. Where it names something proposed, it is marked **(proposed)**.

## Core terms

### Model architecture

The *family* of transformer: its layer structure, normalization scheme, attention
variant, positional encoding and the set of weight tensors it expects. Llama 3,
Qwen 3, Phi-3 and Granite are model architectures. An architecture is a description,
not an object holding weights.

Today the nearest equivalent is `ModelType` combined with the per-family classes under
`model/llama`, `model/qwen3`, etc.

### Model configuration

The immutable numeric and structural hyperparameters of one specific model:
`dim`, `hiddenDim`, `numberOfLayers`, `numberOfHeads`, `numberOfKeyValueHeads`,
`vocabularySize`, `contextLength`, `rmsNormEps`, `ropeTheta`, and family-specific
additions.

Today: `org.beehive.gpullama3.model.Configuration` and its per-family
implementations.

### Model weights

The immutable trained parameter tensors of one model, in the representation used for
execution (not the file representation).

Today: `inference.weights.Weights`, with `StandardWeights` (CPU) and
`TornadoWeights` (GPU) hierarchies.

### Loaded model

A configuration plus weights plus the architecture identity, resident in memory (host
and/or device), **immutable**, and shareable across many sessions. It does not own
sequence position, a KV cache, or activation buffers.

Today: `model.Model` / `AbstractModel` — but with the caveat that these also hold a
mutable `TornadoVMMasterPlan` and generation methods, so they are not yet a loaded
model in this sense. See [ADR-001](decisions/ADR-001-model-session-separation.md).

### Model provider **(proposed)**

An extension point that recognizes a model source (e.g. a GGUF file with particular
metadata) and produces a loaded model plus the architecture-specific pieces needed to
run it. Registered rather than hard-coded, so adding an architecture does not require
editing a central switch.

Today the equivalent behaviour is spread across `ModelLoader.detectModelType`,
`ModelType` and the per-family loader classes.

### Session

One in-progress sequence. Owns the mutable state associated with that sequence:
current position, KV cache, and any per-sequence bookkeeping. Multiple sessions may
share one loaded model. A session is not thread-safe; a loaded model is.

Does not exist today as a type. `server.InferenceService` is the closest approximation.

### Inference state

The mutable data a session needs across invocations. Distinct from *invocation*
bindings, which live only for one call.

Today: `inference.state.State`, which currently mixes both lifetimes plus device
mirrors.

### Transformer state

The transformer-specific part of inference state: KV cache, position, and the
activation/scratch buffers shaped by the transformer configuration. Named separately
because non-transformer or non-autoregressive use cases must not be forced to have it.

### KV cache

The cache of computed key and value projections, indexed by layer and position.

**Storage is owned by the KV cache manager, not by a session and never by a loaded
model.** A session holds a *lease*. This wording matters: blocks may outlive a sequence
and be shared between sequences, which is what makes paged attention and prefix reuse
possible.

Today: `State.keyCache` / `State.valueCache` (CPU) and `State.wrapKeyCache` /
`wrapValueCache` (device) — per-`State`, with no manager.

### KV cache manager

Engine-scoped owner of KV block storage. Allocates the pool, hands out leases, reclaims
blocks, enforces capacity, and pins blocks that are under a live lease.

### Block pool

The single persistent device array backing all KV blocks. One array, indexed in-kernel —
not a set of separately allocated buffers. This is an invariant, not an implementation
choice: `withCUDAGraph()` bakes device addresses into the captured graph, so re-pointing
a slot's buffer between replays breaks replay
([capability C1](tornadovm-capabilities.md#c1--cuda-graph-capture-fixes-device-addresses)).

### Block table

Per-sequence mapping from logical position to physical block in the pool. The attention
kernel walks it, which is why shared-then-private block layouts need no kernel change.

### Lease

A session's claim on a set of blocks it does not own. Released on session close.
Blocks under a live lease are pinned against eviction.

### Slot

One sequence's position within a batched step. B slots per invocation. Distinct from a
session: a session is the user-facing sequence, a slot is its place in the current batch.

### Prefix cache

Engine-scoped map from token-prefix identity to shared KV blocks, so a repeated system
prompt is prefilled once. Prefix identity must include model, dtype **and** position
offset. Shared blocks are refcounted.

### Admission

The scheduler's decision to accept a request into the running batch, based on free
blocks and device capacity. A request that cannot be admitted waits rather than failing.

### Engine

The tier owning work across sequences: admission, batch composition, KV management,
prefix reuse, preemption. Sits above sessions, below the public API.

**Not** the same as the existing `InferenceEngine` class, which is a generation loop.
See [terms to avoid](#terms-to-avoid-or-use-carefully).

### Scheduler

The engine component deciding which sequences run in the next step, and which are
preempted. Distinct from TornadoVM's `GridScheduler`, which sizes worker grids.

### Time to first token (TTFT)

Wall time from request admission to the first emitted token. Only reproducible if the
record states whether the compiled-kernel cache was warm or cold — cache state moves
start-up by seconds, dwarfing scheduling effects
([capability C5](tornadovm-capabilities.md#c5--performance-history-has-version-sized-discontinuities)).

### Invocation

One execution of a compiled program: a call that binds inputs, outputs and the mutable
session state, runs, and returns. Temporary buffers used only within an invocation are
invocation-scoped and must not be treated as model state.

Today there is no explicit type; the nearest thing is one call to
`TornadoVMMasterPlan.tornadoVMForwardDecode(position)`.

### Operation

A reusable, backend-neutral inference primitive with defined inputs and outputs:
RMSNorm, RoPE, matrix–vector multiply, attention, SwiGLU, softmax, residual add.
An operation describes *what* is computed, not how it is scheduled on a device.

Today the closest artefacts are the static methods in `inference.operation.RoPE`,
`InferenceCore` (CPU) and the kernel methods in `tornadovm/kernels` (GPU) — but
these are not a shared operation vocabulary; the CPU and GPU sides are separate.

### Program component **(proposed)**

A named, composable unit of an inference program — for example "embedding lookup",
"transformer layer *i*", "final norm and vocabulary projection". A program is built
from components; a backend compiles components into whatever executable form it uses.

The existing `*ForwardPlanComponents` interfaces are the TornadoVM-specific ancestor
of this idea.

### Inference program

A **backend-neutral** description of the work one forward pass performs: which
operations, over which weights, in what order, producing which outputs. It contains no
device handles, no task graphs and no TornadoVM types. It is data, not execution.

Does not exist today.

### Program signature **(proposed)**

The typed contract of an inference program: what it consumes (input bindings),
what it produces (output bindings), and what mutable state it reads and writes.
Used to validate that a compiled program is being invoked correctly and that a
session's state matches the program it was compiled for.

### Compiler

The component that turns an inference program into a compiled program for a specific
backend. **For the TornadoVM backend, the compiler is TornadoVM.** GPULlama's role is
to translate its program description into TornadoVM task graphs and hand them over;
it does not generate device code itself. See
[ADR-003](decisions/ADR-003-tornado-backend-boundary.md).

### Compiled program

A **backend-specific**, reusable, executable form of an inference program, together
with the device resources it needs. Built once. Invoked many times. Never rebuilt per
token.

Today: `TornadoVMMasterPlan` implementations together with their `ForwardPlan`
(`List<ImmutableTaskGraph>` + `GridScheduler`) and `TornadoExecutionPlan`.

### Backend

An implementation that can compile inference programs and execute compiled programs on
some class of hardware, and that owns the device memory involved. The TornadoVM
backend is the primary one. A plain-Java CPU path is a second.

Today there is no `Backend` type; the CPU path (`InferenceCore`) and the GPU path
(`tornadovm/**`) are selected by a boolean and static properties.

### Device

A concrete accelerator or processor a backend can target — a specific GPU, an
integrated GPU, a CPU. Within the TornadoVM backend, PTX/CUDA, OpenCL and SPIR-V are
**device and backend capabilities of TornadoVM**, not separate GPULlama backends. See
[ADR-003](decisions/ADR-003-tornado-backend-boundary.md).

### Execution policy **(proposed)**

An explicit, per-model-or-session choice of *how* inference is executed: single-token
decode, prefill/decode separation, batched prefill with a chunk size, device sampling
on/off. Today these choices are made by process-global system properties
(`llama.withPrefillDecode`, `llama.prefillBatchSize`, `llama.deviceSample`) read into
`static final` fields; the proposal is to make them explicit values.

Corresponds today to `tornadovm.plan.ExecutionMode` plus the `llama.*` properties.

### Prefill

The phase that ingests prompt tokens to populate the KV cache. Logits are not needed
for prompt tokens except the last one, so a prefill-specific path can skip the
vocabulary projection.

Today: `InferenceEngineWithPrefillDecode`, `PrefillDecodeForwardPlan`, and the
`layers/type/*/prefill` graph builders.

### Decode

The autoregressive phase: one token in, one logits row out, repeated until a stop
condition.

Today: the single-token path — `TornadoVMMasterPlanSingleToken`,
`SingleTokenForwardPlan`, and the `layers/type/*/decode` builders.

### Batch decode

Processing more than one token position per invocation. In this repository the
existing batched path is **batched prefill** (`BATCH_PREFILL_DECODE`,
`llama.prefillBatchSize`), which processes a chunk of prompt tokens at once and then
decodes one token at a time. Batched *decode* across independent sequences does not
exist today and would require per-sequence KV cache addressing.

Use "batch prefill" for the existing mechanism. Reserve "batch decode" for the
multi-sequence case and say which you mean.

### Generation

The loop above the model: prompt construction, invocation scheduling, sampling, stop
conditions, streaming callbacks, detokenization. **Generation is not part of the model
forward pass** and must be separable from it, because embedding, classification and
reranking use cases have no generation loop.

Today: `InferenceEngine*` plus the `runInteractive` / `runInstructOnce` default
methods on `Model`.

### Forward plan

**Existing, transitional, TornadoVM-specific.** `tornadovm.plan.ForwardPlan` and its
subclasses hold `List<ImmutableTaskGraph>` and a `GridScheduler` — they are typed in
TornadoVM terms and are therefore a *compiled program* structure for the TornadoVM
backend, not a backend-neutral concept.

**"Forward plan" must not become the generic term** for backend-neutral inference
work. Use *inference program* for the backend-neutral description and *compiled
program* for the backend-specific executable. `ForwardPlan` should be understood as
today's TornadoVM compiled-program internals, expected to end up behind the backend
boundary. See [ADR-002](decisions/ADR-002-program-and-compiled-program.md).

## Terms to avoid or use carefully

These words already carry more than one meaning in this repository or in the wider
field. Qualify them or use the precise term.

| Term | Problem | Use instead |
| --- | --- | --- |
| **engine** | Now three-way ambiguous: `InferenceEngine` (the generation loop), the new **engine tier** (scheduling across sequences), and the project as a whole. | *generation loop*, *engine tier*, *the project*. Never bare "engine" in a document. Renaming `InferenceEngine` is expected when the generation loop consolidates. |
| **batch** | Batched *prefill* (one sequence, many prompt tokens — exists today) vs batched *decode* (many sequences, one token each — PR #129) vs TornadoVM's `withBatch()` (chunking large data). | *batch prefill*, *batch decode*, *TornadoVM data batching*. |
| **slot / session** | A slot is a position in the current batch; a session is a user-facing sequence. B slots does not mean B sessions. | Say which. |
| **runtime** | Means the TornadoVM runtime, the JVM, and GPULlama's own execution layer. | *TornadoVM runtime*, *JVM*, *execution layer*. |
| **plan** | `ForwardPlan`, `TornadoVMMasterPlan`, `TornadoExecutionPlan`, and the informal "execution plan" all differ. | *compiled program*, *task graph*, *TornadoExecutionPlan* — be specific. |
| **model** | Means the file on disk, the architecture family, the loaded weights, and the `Model` interface (which also runs chat loops). | *model file*, *model architecture*, *loaded model*. |
| **state** | Means session-lifetime state, invocation-lifetime scratch, and the `State` class holding both. | *session state*, *KV cache*, *invocation buffers*. |
| **backend** | GPULlama backend vs. TornadoVM backend (PTX / OpenCL / SPIR-V) — different levels. | *GPULlama backend*, *TornadoVM device backend*. |
| **kernel** | The Java method, the compiled device code, and TornadoVM's `KernelContext` API style (as opposed to the loop-parallel style). | *kernel method*, *compiled kernel*, *KernelContext API*. |
| **tensor** | `FloatTensor` is explicitly "shapeless" — a flat float sequence, not a shaped tensor. `TornadoTensor` is a device buffer wrapper. | Say *buffer*, *weight tensor*, or *tensor descriptor* as appropriate. |
