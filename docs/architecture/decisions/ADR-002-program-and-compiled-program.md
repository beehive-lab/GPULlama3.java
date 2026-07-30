# ADR-002: Inference program and compiled program

## Status

**Accepted** — 2026-07-30, following the ARCH-03 and ARCH-05 review on PR #140.

Amended on acceptance:
- Sampling is admitted as an operation with a backend implementation
  ([Rule 8b](../dependency-rules.md#rule-8b--sampling-is-an-operation-and-may-execute-on-the-device)).
- Compiled-program identity depends on deterministic generated source. TornadoVM emitted
  kernel source non-deterministically across JVM runs until #999, so the identity test only
  holds from the adopted version floor onward — see
  [capability C3](../tornadovm-capabilities.md#c3--generated-kernel-source-was-non-deterministic-before-52).
- Operations are `DataType`-parameterized at the description and dispatch level, not inside
  kernel bodies: TornadoVM compiles per concrete native array type and Java has no generics
  over primitives.

## Context

The GPU path today has a structure that is close to the right one, expressed entirely
in TornadoVM types.

`tornadovm.plan.ForwardPlan` holds a `List<ImmutableTaskGraph>` and a `GridScheduler`.
Three subclasses give three topologies: `SingleTokenForwardPlan` (N+2 graphs:
activation, N layers, logits), `PrefillDecodeForwardPlan` (N+2 graphs shared between
phases, logits skipped during prefill) and `BatchPrefillDecodeForwardPlan` (2N+3
graphs). `ForwardPlanFactory` builds them by switching on `GGMLType`, then
`ModelType`, then `ExecutionMode`, producing one of 14 `*PlanComponents`
implementations (7 families × 2 quantizations).

`TornadoVMMasterPlan` implementations wrap a `ForwardPlan` in a
`TornadoExecutionPlan`, call `withPreCompilation()` once at construction, and then
execute the graphs per token in `tornadoVMForwardDecode(position)`.

The good properties already present:

- construction and compilation happen once, execution happens per token;
- the forward pass is decomposed into named, separately-executable units;
- `*ForwardPlanComponents` is a genuine per-architecture extension point;
- phases can skip units (prefill skips logits).

The limitation: all of it is typed in `ImmutableTaskGraph`, `TaskGraph` and
`GridScheduler`. There is no description of the forward pass that a non-TornadoVM
backend could consume, and the CPU path (`InferenceCore.forwardJava*`) is a completely
separate hand-written implementation with no structural relationship to it.

## Decision

Introduce two distinct concepts with distinct types:

**`InferenceProgram` — backend-neutral.**
A description of the work one forward pass performs: which operations, over which
weights, in what order, producing which outputs. It contains:

- an ordered list of program components;
- a program signature (inputs, outputs, mutable state it reads and writes);
- references to weights by role, not device handles.

It contains **no** device handles, **no** task graphs, **no** TornadoVM types and
**no** GGUF types. It is data. It can be inspected, logged and compared.

**`CompiledProgram` — backend-specific.**
The executable form of an inference program for one backend and one device, together
with the device resources it needs. For the TornadoVM backend this wraps exactly what
exists today: the `ImmutableTaskGraph` list, the `GridScheduler` and the
`TornadoExecutionPlan`.

The contract:

1. `Backend.compile(InferenceProgram, CompileOptions)` produces a `CompiledProgram`.
2. A `CompiledProgram` is **reusable**: built once, invoked many times.
3. **Compilation never happens per token.** This is the property that already holds
   and must survive
   ([rule 13](../dependency-rules.md#rule-13--no-compilation-or-task-graph-construction-per-token)).
4. An **invocation** binds inputs, outputs and mutable session state to a compiled
   program and runs it. It performs no compilation and, ideally, no allocation.
5. `ForwardPlan`, `*ForwardPlanComponents` and `TornadoVMMasterPlan` are **transitional
   TornadoVM-specific compiled-program internals**. They keep their names, stay inside
   the Tornado backend, and are not generalized upward.

The last point is deliberate. "Forward plan" must not become the generic
backend-neutral term — see
[`terminology.md`](../terminology.md#forward-plan) and
[rule 12](../dependency-rules.md#rule-12--forward-plans-are-transitional-tornado-compiled-program-structures).

## Should programs be arbitrary graphs?

**No — not in the first version.**

The argument for a general graph IR: it expresses branching, fusion opportunities and
non-linear topologies, and it would let a backend reorder work.

The arguments against, for this project specifically:

1. **Transformer inference is a sequence.** Embedding → N × (attention, FFN) → norm →
   projection. The existing plans are literally ordered lists: `[0]` activation,
   `[1..N]` layers, `[N+1]` logits. Nothing in the current code needs a graph.
2. **Reordering is TornadoVM's job.** A graph IR whose purpose is optimization is the
   beginning of a second compiler, which
   [ADR-003](ADR-003-tornado-backend-boundary.md) rules out.
3. **A graph IR needs a validator, a traversal API, a serialization format and its own
   test suite** before it computes anything. An ordered component list needs none of
   that.
4. **Branching is currently phase selection, not data flow.** Prefill vs decode is a
   choice of which components to run, expressible as two programs or one program with
   selectable components — not as a data-dependent branch.

**Recommendation:** the first version uses **ordered, composable program components**.
A component may itself be composite (a transformer layer is one component containing
attention and FFN sub-components), which gives structure without giving a general
graph.

If a future use case genuinely needs a graph — encoder–decoder cross-attention and
multimodal fusion are the plausible candidates — the ordered form should be extended
then, with that use case as the evidence. Building the graph first would be building
for a requirement that does not exist yet.

## Consequences

Positive:

- Architecture descriptions become backend-independent; one Llama description can be
  compiled by the TornadoVM backend and by the CPU backend.
- The "compile once" property becomes structural rather than conventional: you cannot
  invoke without a `CompiledProgram`, and you cannot get one without calling
  `compile`.
- Compiled programs become shareable between sessions
  ([ADR-001](ADR-001-model-session-separation.md)).
- Programs are inspectable, which makes diagnostics and memory planning tractable.
- `program/` can be tested without a GPU.

Negative / costs:

- One more layer between the architecture and the task graphs.
- The `program` → `CompiledProgram` translation for the TornadoVM backend is real
  work: it must reproduce what the 14 `*PlanComponents` implementations do today,
  producing identical task graphs.
- Risk of an invocation layer that allocates per token and erases the benefit. The
  invocation must bind existing buffers, not create them.
- Two representations of the same thing means two places to look when debugging.

## Alternatives considered

**Keep only `ForwardPlan` and make it backend-neutral.** Would mean removing
`ImmutableTaskGraph` and `GridScheduler` from its API — at which point it is a new type
with an old name, and every reference to "plan" in the codebase becomes ambiguous.
Rejected on clarity grounds; the naming confusion is already flagged in
[`terminology.md`](../terminology.md#terms-to-avoid-or-use-carefully).

**No neutral program at all; each backend has its own description.** This is the status
quo (task graphs for GPU, hand-written Java for CPU). Rejected: it means every
architecture is implemented once per backend, which is exactly the duplication the
framework is meant to remove.

**A full tensor/loop IR with its own code generation.** Rejected — see
[ADR-003](ADR-003-tornado-backend-boundary.md). It would duplicate TornadoVM and
discard the project's main advantage.

**Compile lazily on first invocation.** Tempting for API simplicity. Rejected: it hides
a multi-second cost inside what looks like a normal call, and it makes it easy to
accidentally compile inside a loop — the exact failure rule 13 exists to prevent.

## Migration notes

Corresponds to [roadmap phase 6](../migration-roadmap.md#m9--program-and-compiled-program),
after operations exist (M8) and state is separated (M6).

Suggested order:

1. Write the ArchUnit rule for `program/` **before** creating the package
   ([rule 3](../dependency-rules.md#rule-3--backend-neutral-program-interfaces-do-not-import-tornadovm)),
   so an accidental TornadoVM import fails immediately.
2. Define `InferenceProgram`, `ProgramComponent`, `ProgramSignature` as ordered
   structures.
3. Express one model family + one quantization as a program. Llama + FP16 is the
   best-covered path (it supports all three execution modes).
4. Implement the TornadoVM `compile(...)` for that program, delegating to the existing
   `*PlanComponents` machinery. Assert the resulting task graph list matches today's.
5. Verify: same number of graphs, same task names, same grid scheduler entries,
   identical output for a fixed seed, tokens/second within noise.
6. Repeat per family and quantization; delete each `*PlanComponents` only once its
   program equivalent is proven.
7. Add the compiled-program identity test: one compile, ≥ 100 tokens, identity
   unchanged.

`ExecutionMode` maps to the execution policy of
[roadmap phase 7](../migration-roadmap.md#m10--execution-policy-consolidation);
the two phases should agree on whether prefill and decode are two programs or one
program with selectable components.

## Open questions

1. Are prefill and decode two separate `InferenceProgram`s, or one program with
   phase-selectable components? Today's `PrefillDecodeForwardPlan` shares N+2 graphs
   between phases, which argues for one program.
2. What exactly is in a `ProgramSignature`, and is it validated at compile time,
   invocation time, or both?
3. Does `Invocation` allocate anything? (It should not — but the batch-prefill path
   binds different buffers per phase, so the binding itself needs a cheap
   representation.)
4. Who owns `CompiledProgram` lifetime — the loaded model's cache, the session, or the
   caller?
5. How are compiled programs keyed for reuse? (architecture, configuration shape,
   policy, backend, device) is the obvious tuple, but configuration shape is not a
   simple value today.
6. Can a program be partially compiled — e.g. layers on GPU, sampling on CPU? The
   `llama.deviceSample` flag already makes this choice, so the answer is probably yes,
   and the signature has to express it.
