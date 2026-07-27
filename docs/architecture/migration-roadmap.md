# Migration Roadmap

**Status: proposal.** Phase order is a recommendation; scope and acceptance criteria
are the parts to argue about in review.

## Principles

1. **No rewrite.** The engine works. Every phase leaves it working.
2. **No big-bang rename.** A repository-wide package rename or multi-module split as
   step one would produce one unreviewable diff and prove nothing.
3. **Rules before code.** Where a phase creates a new boundary, the ArchUnit rule for
   that boundary lands first (with an allowlist), then the code.
4. **Performance is a correctness criterion.** Every phase that touches the GPU path
   states a benchmark acceptance criterion. The compile-once/execute-many property
   ([rule 13](dependency-rules.md#rule-13--no-compilation-or-task-graph-construction-per-token))
   is non-negotiable.
5. **Additive first, removal later.** New surface is added alongside old surface; old
   surface is deprecated with a replacement before removal.

## Recommended first milestone

**Milestone 1 = Phase 1 + the additive half of Phase 2.** Low risk, no behaviour
change, no GPU path change:

- this architecture baseline (done — this directory);
- ArchUnit test module with rules 1, 2, 5, 7, 11, 13 and enumerated allowlists;
- the three trivially-final fields on `AbstractModel` made final;
- a read-only façade (`LocalModels` / `LocalModel` / `GenerationSession`) that
  delegates to existing `Model` methods without changing them.

Deliberately **not** in milestone 1: package renames, module splits, `State`
restructuring, `Model` interface changes, anything touching `tornadovm/**` internals.

---

## Phase 1 — Architecture baseline and dependency rules

**Objective.** Write the target down and make the most important boundaries
machine-checked, with the current violations enumerated rather than ignored.

**Affected.** `docs/architecture/**`; a new test source set for ArchUnit; `pom.xml`
(test-scoped ArchUnit dependency only).

**Non-goals.** No production code change. No package moves. No renames.

**Acceptance criteria.**
- Baseline documents exist and are reviewed by maintainers.
- ArchUnit tests exist for rules 1, 2, 5, 7, 11, 13 and pass with allowlists.
- Every allowlist entry is a fully-qualified class name with a phase reference; no
  wildcard entries.
- `mvn test` passes; build output is unchanged.

**Compatibility risks.** None — test-scope only.

**Performance risks.** None.

**Depends on.** Nothing.

---

## Phase 2 — Public API façade over current implementation

**Objective.** Give users the intended API shape before the internals move, so the
internals can move without breaking users again later.

**Affected.** New `api/` package. Read-only use of `model/`, `inference/`,
`tornadovm/`.

**Scope.** `LocalModels`, `LocalModel`, `GenerationSession`, `GenerationRequest`,
`GenerationResult`, `ModelOptions`, `SessionOptions` — all delegating to today's
`ModelLoader.loadModel`, `Model.createNewState`,
`TornadoVMMasterPlan.initializeTornadoVMPlan` and the `InferenceEngine*` loops.
The façade's `GenerationSession` initially wraps one `State` + one plan, i.e. what
`server.InferenceService` already does.

**Non-goals.** No change to `Model`, `State`, `InferenceEngine*` or the plan
hierarchy. No removal of `runInstructOnce` / `runInstructOnceLangChain4J`. No
`Backend` or `InferenceProgram` types yet.

**Acceptance criteria.**
- The simple example in [`public-api.md`](public-api.md) compiles and runs on both
  CPU and GPU paths.
- No TornadoVM, GGUF or `Options` type appears in any `api/` signature.
- Existing CLI, `OpenAIServer` and LangChain4j entry points behave identically.
- Generated output is token-identical to the pre-phase build for a fixed seed.

**Compatibility risks.** Low — additive. Risk is committing to names too early; hence
the API should be marked experimental in Javadoc until phase 7.

**Performance risks.** Low. Watch for the façade creating a `State` or a plan more
than once per session.

**Depends on.** Phase 1 (for the rules the façade must not violate).

---

## Phase 3 — Loaded-model and session-state separation

**Objective.** Make the loaded model immutable and move all per-sequence mutable state
behind a session. This is the phase that unblocks concurrency and most later phases.

**Affected.** `model/Model`, `model/AbstractModel`, all family model classes,
`inference/state/**`, `server/InferenceService`, `api/`.

**Scope.**
- Remove `tornadoVMPlan()` / `setTornadoVMPlan(...)` from `Model`; the compiled
  program moves to the session (or a model-owned cache keyed by policy/device).
- Make `AbstractModel` fields final.
- Split `State` into session-lifetime state (KV cache, position) and
  invocation-lifetime buffers (activations, scratch).
- Keep family-specific state (e.g. `Qwen3State.wrapAttSplit`) only where genuinely
  required.

**Non-goals.** No move of `generateTokens` off `Model` yet (phase 7). No package
renames. No new backend SPI. Not required to deliver concurrent sessions — only to
stop preventing them.

**Acceptance criteria.**
- `Model` no longer references `TornadoVMMasterPlan`; dependency rule 2 allowlist
  loses its `model/` entries.
- Two sessions can be created against one loaded model and produce correct,
  independent output (sequentially, at minimum).
- `InferenceService` is reimplemented on top of the session type without behaviour
  change.
- Deterministic output unchanged for a fixed seed.

**Compatibility risks.** **High** — `Model` is public surface used by LangChain4j and
Quarkus integrations. Removed methods must be deprecated first with a documented
replacement, and the integration repositories notified.

**Performance risks.** **Medium.** Splitting `State` changes buffer allocation and
possibly device transfer boundaries. Risks: extra allocation per invocation;
losing buffer reuse; changing which buffers are marked for device transfer. Requires a
tokens/second benchmark on both FP16 and Q8_0, single-token and prefill/decode paths,
before and after.

**Depends on.** Phase 2.

---

## Phase 4 — Generic tensor metadata and GGUF isolation

**Objective.** Introduce a runtime `DataType` and tensor descriptors, and confine
GGUF/GGML to the loading path.

**Affected.** `tensor/**`, `inference/weights/**`, `model/loader/**`, and the
`GGMLType` dispatch in `tornadovm/plan/ForwardPlanFactory`.

**Scope.**
- Add a runtime `DataType` (F32, F16, Q8_0, …) owned by the runtime layer.
- Add tensor descriptors (dtype + shape/element count + layout) separate from storage.
- Make loading an explicit mapping: `GGMLTensorEntry` → descriptor → backend storage.
  The existing `AbstractModelLoader.effectiveGpuWeightType` collapse of
  `Q4_K`/`Q5_K`/`Q6_K` → `Q8_0` becomes part of that mapping.
- Move `GGUF`, `GGMLTensorEntry`, `MetadataValueType` into a format package.

**Non-goals.** No new file format support. No shaped-tensor arithmetic or broadcasting
semantics. No change to how quantized kernels work. `FloatTensor` stays shapeless.

**Acceptance criteria.**
- `Weights`, `FloatTensor` and `TornadoTensor` expose `DataType`, not `GGMLType`.
- `ForwardPlanFactory` dispatches on `DataType`.
- No type outside the format package references `GGMLType` or `GGUF`.
- Dependency rule 4 passes with an empty or near-empty allowlist.
- Loading time and memory footprint unchanged (measure both).

**Compatibility risks.** Medium — `Weights.getWeightType()` and the tensor classes are
public. `ModelLoader.loadTensor(GGMLTensorEntry)` is public and used by loaders.

**Performance risks.** Low for execution; **watch model load time**, which is already
tracked by `RunMetrics.setLoadDuration`. An extra descriptor layer must not add a copy.

**Depends on.** Phase 1. Independent of phase 3, so the two can proceed in parallel.

---

## Phase 5 — Reusable transformer operation extraction

**Objective.** Establish one named operation vocabulary (RMSNorm, RoPE, MatVec,
Attention, SwiGLU, Softmax, ResidualAdd, EmbeddingLookup, VocabProjection) that both
the CPU path and the TornadoVM path implement.

**Affected.** New `program/op/` package; `inference/InferenceCore*`;
`tornadovm/kernels/**`; `tornadovm/layers/**`.

**Scope.** Define the operations and their parameters. Express the existing CPU
forward passes and the existing task-graph builders in terms of them. Reduce the
duplication between `InferenceCore`, `InferenceCoreWithPrefillDecode` and
`InferenceCoreBatchPrefillDecode`.

**Non-goals.** **Do not rewrite kernels.** Do not merge the CPU and GPU
implementations — only the vocabulary is shared. Do not introduce an operator
registry, a graph optimizer or fusion rules.

**Acceptance criteria.**
- Each operation is defined once, independent of model family.
- A model family's forward pass is expressible as a sequence of operations.
- Kernel method bodies in `tornadovm/kernels/**` are unchanged or provably equivalent.
- Tokens/second within noise of the previous phase on FP16 and Q8_0.

**Compatibility risks.** Low — internal.

**Performance risks.** **High if kernels are touched.** The mitigation is the
non-goal above: this phase names and organizes, it does not rewrite compute. Any
change to a kernel body in this phase should be rejected in review and deferred.

**Depends on.** Phase 4 (operations are typed in terms of `DataType`).

---

## Phase 6 — Logical program and compiled-program separation

**Objective.** Introduce `InferenceProgram` (backend-neutral) and `CompiledProgram`
(backend-specific), with the TornadoVM path as the first implementation of the latter.

**Affected.** New `program/` package; `tornadovm/plan/**`; `tornadovm/TornadoVMMasterPlan*`.

**Scope.**
- `InferenceProgram` as an ordered list of program components — **not** a graph IR
  ([ADR-002](decisions/ADR-002-program-and-compiled-program.md)).
- `CompiledProgram` implemented by wrapping today's `TornadoVMMasterPlan` +
  `ForwardPlan` + `TornadoExecutionPlan`.
- `Invocation` binding inputs, outputs and session state.
- `ForwardPlan` and `*ForwardPlanComponents` become internal to the Tornado backend.

**Non-goals.** No graph IR, no loop IR, no scheduling decisions moved out of
TornadoVM, no second compiler. No change to the number or content of task graphs.

**Acceptance criteria.**
- `program/` has zero TornadoVM dependencies (rule 3 passes with no allowlist).
- One `InferenceProgram` produces the same task graphs as today for a given
  model + quantization + policy.
- Compilation happens once per (model, policy, device); a test asserts compiled-program
  identity is stable across ≥ 100 generated tokens (rule 13).
- Tokens/second within noise.

**Compatibility risks.** Low externally; high internally — this touches the whole
`tornadovm/plan` tree.

**Performance risks.** **Medium.** The failure mode is an indirection layer that turns
one direct `execute()` into per-token allocation or lookup. Invocation must bind, not
allocate.

**Depends on.** Phases 3 and 5.

---

## Phase 7 — Consolidate engine variants behind execution policies

**Objective.** Replace the three parallel engine/core variants and the static system
properties with one generation loop parameterized by an explicit `ExecutionPolicy`.

**Affected.** `inference/InferenceEngine*`, `inference/InferenceCore*`, `Options`,
`Model` default methods, `model/llama/Llama` and siblings, `LlamaApp`.

**Scope.**
- One generation loop; prefill/decode/batch-prefill become policy, not class identity.
- `ExecutionPolicy` as a value passed at model or session creation, replacing
  `llama.withPrefillDecode`, `llama.prefillBatchSize` and `llama.deviceSample` as
  *class-initialization-time* `static final` reads.
- Move `runInteractive` / `runInstructOnce` / `runInstructOnceLangChain4J` out of
  `Model` into the CLI and integration layers.
- Retire `Options.setProperty(...)` side effects in the record constructor.

**Non-goals.** Do not remove the system properties as an *input* mechanism — the
`llama-tornado` launcher and existing scripts pass them. They should configure a policy
object rather than be read directly at class initialization.

**Acceptance criteria.**
- `InferenceEngineWithPrefillDecode` and `InferenceEngineWithBatchPrefillDecode` are
  gone or reduced to thin deprecated delegates.
- `Model` no longer performs console I/O and no longer imports `Options`.
- Dependency rule 8 passes.
- `LlamaApp.guardDeviceSample`'s class-initialization-ordering workaround is no longer
  necessary.
- All existing flag combinations produce identical output and comparable throughput.

**Compatibility risks.** **High.** `Model.runInstructOnce*` are used by external
integrations; the `llama-tornado` launcher, benchmark scripts under `scripts/` and CI
all pass `llama.*` properties. Deprecate, do not delete, and keep property parsing
working.

**Performance risks.** **Medium.** Today's static `final` flags are constant-folded by
the JIT. Moving to instance fields on a hot path could cost measurably. Mitigation:
resolve policy once per session, not per token; benchmark the decode loop specifically.

**Depends on.** Phases 3 and 6.

---

## Phase 8 — Model provider SPI

**Objective.** Adding a model architecture means adding a provider, not editing central
switches.

**Affected.** `model/ModelType`, `model/loader/**`, `tornadovm/plan/ForwardPlanFactory`,
`inference/sampler/Sampler.createSampler`, new `model/provider/`.

**Scope.** `ModelProvider` SPI with `supports(ModelSource)` / `load(...)`, discovered
via `ServiceLoader`. Per-family providers replacing the `ModelType` dispatch, the
`detectModelType` string matching, and the family branches in `ForwardPlanFactory`.

**Non-goals.** Not required to delete `ModelType` — it may remain an internal
identifier. Not required to support non-GGUF sources.

**Acceptance criteria.**
- Adding a new architecture touches only new files plus one service-registration entry.
- Dependency rule 15 passes.
- All currently supported families load and run identically.
- A deliberately-unsupported model produces a clear error, not a wrong-family load.
  (Today's `detectModelType` matches on substrings of `general.name`, so this is a
  behaviour worth testing explicitly during the migration.)

**Compatibility risks.** Medium — `ModelType` and `ModelLoader.loadModel` are public.

**Performance risks.** Low; load-time only.

**Depends on.** Phases 4 and 6.

---

## Phase 9 — Backend and device SPI

**Objective.** Make TornadoVM one backend behind an interface, with explicit device
selection.

**Affected.** New `backend/` SPI; `tornadovm/**` → `backend/tornado/**`;
`inference/InferenceCore*` → `backend/cpu/**`; `tornadovm/scheduling/**`.

**Scope.** `Backend`, `Device`, `DeviceSelector`, `CompileOptions`. TornadoVM backend
implementing them. The plain-Java CPU path implementing them. **This is the phase that
performs the `tornadovm` → `backend.tornado` package move**, which is what makes
dependency rules 1 and 11 enforceable without an allowlist.

**Non-goals.** No new hardware backend. No abstraction over TornadoVM's own PTX /
OpenCL / SPIR-V backends — those stay device-level concerns of the TornadoVM backend
([ADR-003](decisions/ADR-003-tornado-backend-boundary.md)).

**Acceptance criteria.**
- Rules 1 and 11 pass with an empty allowlist.
- Device selection is explicit and testable; the current failure mode where `--cuda`
  without `--gpu` silently runs on CPU is impossible to express.
- Both backends satisfy the same SPI test suite.
- No throughput change (this is a move, not a rewrite).

**Compatibility risks.** **High** — a large package move affecting imports repository-wide,
plus the shaded jar, native-image configuration and release automation.

**Performance risks.** Low in principle; verify anyway, because a package move can
change class-initialization order (see phase 7's note on `guardDeviceSample`).

**Depends on.** Phases 6 and 7.

---

## Phase 10 — Memory planning, diagnostics and developer experience

**Objective.** Make the framework usable in production JVM services: predictable
memory, useful errors, observable behaviour.

**Affected.** `runtime/`, `api/`, `auxiliary/metrics/`, `server/`.

**Scope.**
- Explicit memory planning: report required device memory for a
  (model, context length, policy, batch size) combination before allocating; fail with
  a clear message instead of an out-of-memory error deep in the backend.
- Diagnostics: which backend and device were chosen and why; which execution policy is
  active; where time goes (load / prefill / decode). Replaces `RunMetrics` static
  printing with values on `GenerationResult`.
- Error messages that name the actual problem — the current
  `UnsupportedOperationException("... not yet supported for MISTRAL + F16")` style in
  `ForwardPlanFactory` is a good model to generalize.
- Session pooling / reuse guidance for server use.
- Javadoc on the public API; the experimental marker from phase 2 is removed here.

**Non-goals.** No profiler. No autotuning. No scheduler of its own.

**Acceptance criteria.**
- A model that will not fit fails at load with a message stating required vs available
  memory.
- Timings are available programmatically, not only printed.
- The public API is documented and no longer marked experimental.

**Compatibility risks.** Low.

**Performance risks.** Low — diagnostics must be off or cheap on the decode path.

**Depends on.** Phase 9.

---

## Phase dependency summary

```
  1 ──┬── 2 ── 3 ──┬── 6 ── 7 ── 9 ── 10
      │            │
      └── 4 ── 5 ──┘
               └──── 8 (also needs 6)
```

Phases 3 and 4 are the two that unblock everything else, and they are independent of
each other.

## What is deliberately not scheduled

- Repository-wide package rename as an early step (only phase 9, and only for the
  backend boundary).
- Multi-module Maven split — see
  [`target-architecture.md`](target-architecture.md#likely-maven-module-structure).
  It should follow phase 9, once the package boundaries hold.
- Any non-transformer use case implementation. Phases 3, 5 and 6 must leave the door
  open ([rule 14](dependency-rules.md#rule-14--core-abstractions-do-not-assume-generation)),
  but building embeddings or vision support is separate work.
- New quantization formats, new kernels, new performance work. Those continue
  independently on their own branches; this roadmap must not block them.
