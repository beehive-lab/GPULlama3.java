# ADR-003: TornadoVM backend boundary

## Status

**Proposed.** Not accepted. No record of maintainer approval exists in this repository.

## Context

TornadoVM is the reason this project exists in the form it does. It JIT-compiles
ordinary Java methods to PTX, OpenCL and SPIR-V, manages device memory and executes
task graphs. The kernels in `tornadovm/kernels/` —
`TransformerComputeKernels`, `TransformerComputeKernelsLayered`,
`TransformerBatchPrefillKernels`, `Qwen3Kernels` and the rest — are plain Java that
TornadoVM turns into device code.

TornadoVM is currently a dependency of the whole module, and its types reach beyond
the `tornadovm` package. 26 files outside `tornadovm/**` import
`uk.ac.manchester.tornado.*`:

| Package | Files |
| --- | --- |
| `model/loader` | 8 |
| `inference/state` | 7 |
| `tensor/tornado` | 4 |
| `inference` | 3 |
| `inference/sampler` | 3 |
| `tensor` | 1 |

Beyond raw imports, `model.Model` declares `TornadoVMMasterPlan tornadoVMPlan()` and
`setTornadoVMPlan(...)`, so the model interface itself is Tornado-shaped. And
`inference.sampler.Sampler` is typed against `FloatArray`, which means token sampling —
a pure host-side concern — cannot be used without TornadoVM on the classpath.

Two risks follow. The obvious one: the public API cannot be TornadoVM-free while the
types it must expose are TornadoVM-typed. The less obvious one, and the more serious:
without a stated boundary, the natural next step when the abstraction feels
insufficient is to build a layer that *replaces* TornadoVM's responsibilities — a
tensor IR, a scheduler, code generation. That path is expensive, and it removes the
one thing that distinguishes this project.

## Decision

**TornadoVM is the compiler and the heterogeneous runtime. GPULlama provides the
layers above it.**

Concretely:

1. **TornadoVM is the primary accelerated backend**, and the project is built around
   it rather than around an abstraction that happens to have a TornadoVM
   implementation. It is not treated as one interchangeable option among many.
2. **Raw TornadoVM types stay inside the backend implementation.** `TaskGraph`,
   `ImmutableTaskGraph`, `GridScheduler`, `TornadoExecutionPlan`, `KernelContext`,
   `WorkerGrid` and the native array types (`FloatArray`, `HalfFloatArray`,
   `IntArray`, `ByteArray`, `Int8Array`) are visible only within
   `backend.tornado.**`.
   ([Rules 1 and 11](../dependency-rules.md#rule-1--tornadovm-stays-in-the-tornado-backend).)
3. **Model and generic runtime APIs are backend-neutral.** A model architecture, an
   inference program, a tensor descriptor and a session's state layout are all
   expressible without TornadoVM.
   ([Rules 2 and 3](../dependency-rules.md#rule-2--model-architecture-packages-do-not-import-tornadovm).)
4. **PTX/CUDA, OpenCL and SPIR-V are TornadoVM device backends, not GPULlama
   backends.** There must never be one copy of model logic per device backend. Device
   capability differences (tensor cores, half2 packing, local memory limits, warp
   width) are handled as device *capabilities* queried within the Tornado backend —
   which is what `TensorCoreSupport` and `SchedulerDetectionService` already do.
5. **Tornado-specific configuration lives in an optional, explicitly-named
   backend-specific API.** Things like `withCUDAGraph()` (today `llama.cudaGraphs`),
   `withPreCompilation()`, grid scheduler tuning and device index selection are real
   and useful. They are exposed to users who opt in by depending on the Tornado
   extension, not through the generic `Backend` interface.

## Why GPULlama must not duplicate TornadoVM's compiler responsibilities

This is the load-bearing part of this ADR.

**It would remove the differentiator.** The value proposition is "write inference logic
in Java, run it on the GPU". That is TornadoVM's capability. A project that builds its
own IR and code generator is a project that happens to be written in Java, competing
with llama.cpp and ONNX Runtime on their terms, where they have years of head start.

**The cost is enormous and recurring.** A tensor IR, a loop IR, a scheduler and PTX /
OpenCL / SPIR-V code generation is a multi-year effort that then needs maintaining
against new hardware, new drivers and new instruction sets — permanently.

**It would fork the improvement path.** Improvements to TornadoVM's compiler currently
benefit this project for free. A parallel compiler stack means those improvements have
to be reimplemented, or forgone.

**The abstraction pressure is upward, not downward.** The problems in
[`current-architecture.md`](../current-architecture.md#observed-architectural-pressure-points)
are about ownership, lifetimes, dispatch and coupling — model/session separation, state
lifetime, per-family switches, format leakage. None of them is a code-generation
problem. None is solved by a compiler.

**Where a genuine gap exists, fix it in TornadoVM.** If the backend needs a capability
TornadoVM does not have, the correct response is a TornadoVM contribution, not a
workaround layer in GPULlama. The project already carries TornadoVM branch work
(packed half2 support, for example), which is evidence this path is available and
already in use.

The line, stated plainly:

| GPULlama owns | TornadoVM owns |
| --- | --- |
| Which operations run, in what order | How an operation becomes device code |
| What the weights and buffers mean | How buffers are placed and transferred |
| When to compile and what to reuse | The compilation itself |
| Session state, KV cache, lifetimes | Device memory allocation primitives |
| Kernel *methods* (Java source) | Kernel *compilation* and scheduling |
| Execution policy (prefill/decode/batch) | Task graph execution |

Note the fifth row: GPULlama writes the kernel methods, in Java. That is not compiler
work — it is the inference logic, which is exactly what the framework should own.

## Consequences

Positive:

- The public API can be TornadoVM-free, which is a requirement of
  [`public-api.md`](../public-api.md).
- Core layers become testable without a GPU or a TornadoVM installation.
- A CPU backend can implement the same SPI, making the existing `InferenceCore` path a
  first-class backend rather than a parallel universe.
- The project's effort stays on the layers that differentiate it.
- Contributions flow to TornadoVM where they belong, benefiting both projects.

Negative / costs:

- An SPI between the neutral layers and the Tornado backend is indirection that did not
  exist. It must be thin enough not to cost throughput.
- Some Tornado capabilities are genuinely useful and will be awkward to reach through a
  neutral interface; the opt-in extension API is the escape hatch, and escape hatches
  get used.
- 26 files must change, plus the whole `model`/`state` coupling — this is not a small
  boundary to establish.
- If TornadoVM lacks something the backend needs, the fix goes upstream, which is
  slower than a local workaround.

## Alternatives considered

**Leave TornadoVM types throughout (status quo).** Cheapest today. Rejected: it makes
the public API impossible to keep clean, forces TornadoVM onto every consumer of the
library, and makes the core untestable without a GPU stack.

**Abstract over TornadoVM *and* other GPU runtimes as equal peers.** Rejected as
framing: TornadoVM is the reason the project is interesting. Designing for hypothetical
peer runtimes would add generality no one has asked for and would probably produce a
lowest-common-denominator interface. The SPI exists to keep the core clean and to
accommodate a CPU backend — not to hedge against TornadoVM.

**Separate GPULlama backends per device backend (a PTX backend, an OpenCL backend, a
SPIR-V backend).** Rejected: it multiplies model logic by device count, which is
precisely what TornadoVM's multi-backend compilation exists to avoid.

**Build a tensor IR and generate device code directly.** Rejected for all the reasons
above. If this is ever revisited, it needs a new ADR with concrete evidence that
TornadoVM cannot be extended to do the job.

## Migration notes

Corresponds to [roadmap phase 9](../migration-roadmap.md#phase-9--backend-and-device-spi),
which includes the `tornadovm` → `backend.tornado` package move.

Order that keeps each step reviewable:

1. Enable [rule 11](../dependency-rules.md#rule-11--taskgraph-and-tornadoexecutionplan-live-inside-the-tornado-backend)
   first. `TaskGraph`, `ImmutableTaskGraph`, `GridScheduler` and
   `TornadoExecutionPlan` are already confined to `tornadovm/**`, so this rule is close
   to passing as soon as the package is renamed.
2. Remove the easy leaks in dependency order of difficulty:
   `Sampler` (3 files, a clear abstraction leak — sampling does not need `FloatArray`),
   then `inference` (3), then `model/loader` (8, handled by
   [ADR-004](ADR-004-tensor-and-format-separation.md)), then `inference/state` (7,
   handled by [ADR-001](ADR-001-model-session-separation.md)), then `tensor/tornado`
   (4, which becomes backend-owned storage).
3. Move `tornadovm/**` to `backend/tornado/**` and `inference/InferenceCore*` to
   `backend/cpu/**`.
4. Introduce `Backend`, `Device`, `DeviceSelector`.
5. Shrink the rule 1 allowlist to empty; delete it.

Throughout: this is a **move and re-typing** exercise, not a rewrite. Kernel bodies do
not change. Task graph structure does not change. Tokens/second should not change, and
should be measured to confirm it.

Two practical hazards worth naming:
- the shaded jar, native-image configuration and release automation reference package
  paths;
- class-initialization order matters today (`LlamaApp.guardDeviceSample` must clear a
  system property before the logits task-graph class is first loaded), and a package
  move can change initialization order.

## Open questions

1. What is the artifact/module name for the Tornado-specific extension API?
2. Does the CPU path become a real `Backend`, or stay a simpler separate path?
   (A real backend is cleaner; it is also more work for a path that exists mainly as a
   reference implementation.)
3. How are device capabilities (tensor cores, half2, local memory size, warp width)
   expressed — as a neutral capability query, or entirely inside the Tornado backend?
   `TensorCoreSupport` and `SchedulerDetectionService` currently answer this inside.
4. Does device selection need to be neutral, given that device enumeration is
   inherently backend-specific?
5. How do TornadoVM version requirements get expressed and checked? The project already
   depends on specific TornadoVM development branches for some features.
6. Where do the kernel *methods* live once the backend is separated — in
   `backend/tornado/kernels`, or in a shared place both backends can read? (They are
   TornadoVM-annotated, so probably the former.)
