# Implementation Roadmap

**Status: agreed.** This is the plan implementation work is measured against, following
the ARCH-01..19 review on PR #140 and the 2026-08-03 hardening pass. Changing a
milestone's objective or acceptance criteria needs an ADR; re-ordering work inside a
milestone does not. Amendments marked **(ADR-007)** are binding —
[ADR-007](decisions/ADR-007-roadmap-ordering-and-transitional-contracts.md) was accepted
2026-08-03; the markers stay as provenance.

Terms: [`terminology.md`](terminology.md). Rules: [`dependency-rules.md`](dependency-rules.md).
Capabilities and version floors: [`tornadovm-capabilities.md`](tornadovm-capabilities.md).
PR-sized task breakdown: [`execution-backlog.md`](execution-backlog.md).
Open decisions and their deadlines: [`decision-gates.md`](decision-gates.md).
Gate machinery: [`verification-gates.md`](verification-gates.md).

## Principles

1. **No rewrite.** The engine works. Every milestone leaves it working.
2. **No big-bang rename.** A repository-wide package rename or module split as step one
   would produce one unreviewable diff and prove nothing.
3. **Rules before code.** Where a milestone creates a boundary, the ArchUnit rule lands
   first (with an allowlist), then the code.
4. **Performance and numerics are correctness criteria**, with named gates — see
   [Definition of done](#definition-of-done).
5. **Additive first, removal later.** New surface lands beside old surface; old surface is
   deprecated with a replacement before removal.

## Definition of done

Gates come in three classes — ordinary CI (A), accelerator qualification (B), and
release/milestone (C) — defined in [`verification-gates.md`](verification-gates.md).
`mvn test` never requires an accelerator or a model file; Class B runs under the
`accel-tests` profile on the pinned tuple, and a hardware-absent skip is never a pass.

Every milestone, without exception:

1. **Goldens bit-identical** for the pinned configuration (M1.4), run with
   `-Dtornado.recover.bailout=False` — the default swallows exactly the failure class that
   state-motion refactoring risks
   ([C4](tornadovm-capabilities.md#c4--interpreter-bytecode-buffer-overflow-was-silent)).
2. **Benchmark gate passes** on its tuple (M1.7).
3. **ArchUnit allowlists shrank or stayed equal** — never grew.
4. **Deprecations, not deletions**, wherever public surface changes.

---

## Phase 0 — TornadoVM version floor

**Objective.** Adopt TornadoVM ≥ 5.2.x. Everything else assumes capabilities that are not
reachable from the pinned version.

**Affected.** `pom.xml` (`tornadovm.base.version` 5.0.0 → ≥5.2.x), the `llama-tornado`
launcher, CI workflows, `Makefile`.

**Why first.**

| Needed | Present in 5.0.0? |
| --- | --- |
| `FP8Array` | No — 5.1.0 |
| `BFloat16Array` (PR #120's BF16 path) | No — 5.2.0 |
| Deterministic generated kernel source (compiled-program identity) | No — #999 |
| Interpreter bytecode buffer sized to the graph | No — #1004 |
| Wait-event matrix fix (53 → 103 tok/s) | No — #1002 |
| On-disk cubin cache (start-up 11.5 s → 5.2 s) | No — #1008 |

**Non-goals.** No use of the new dtypes yet. No behaviour change beyond what the bump
itself brings.

**Acceptance (ADR-007 D1).** Build and launcher work on the new version; a fresh
`perf-history.jsonl` entry recorded, since prior entries are not comparable. Goldens are
**not** part of Phase 0 — the golden infrastructure is M1.4, which depends on Phase 0 and
generates the baseline on the new floor. Goldens generated on the old floor are invalid.

**Compatibility risks.** Medium — the launcher and CI pin versions in several places.

**Performance risks.** None negative; #1002 and #1008 are large improvements. The risk is
*mistaking them for* refactor gains, which is why the gate tuple includes the version.

---

## M1 — Guardrails

**Objective.** Make the boundaries machine-checked and give every later milestone a
numerical safety net. M1.1–M1.7 touch no production file. **M1.8 is the one declared
exception (ADR-007 D2)** — a mechanical production change (three `final` keywords, no
behaviour change) that lands **after M1.4**, so the first production edit happens with
goldens already in place.

| Task | Detail | Acceptance |
| --- | --- | --- |
| M1.1 | ArchUnit test module, test-scoped dependency | `mvn test` passes; build output unchanged |
| M1.2 | Rules 1, 2, 5, 7, 11 with enumerated allowlists | Fully-qualified class names, each with a milestone reference; no wildcards |
| M1.3 | Rule 8a with allowlist; Rule 16 (console I/O) with its 20 files | New console I/O in library code fails the build |
| M1.4 | **Golden logits** — Llama-3.2-1B × {FP16, Q8_0}, fixed prompt, greedy, pinned backend, `recover.bailout=False` | Re-run asserts bit-identical; goldens committed |
| M1.5 | CPU↔GPU parity, tolerance `\|got − ref\| ≤ 1e-2·Σ\|wᵢaᵢ\| + 1e-3` | Passes on both backends |
| M1.6 | Compiled-program identity test | One compile, ≥100 tokens, identity unchanged |
| M1.7 | **Benchmark gate** — add `machine`, `gpu`, `tornadovm_version`, `cache_warm` fields to `perf-history.jsonl`; gate script | Compares against last green run of the same tuple; tolerance stated per tuple |
| M1.8 | `AbstractModel.tokenizer` / `weights` / `chatFormat` final — **after M1.4** | Compiles; goldens bit-identical |

**Non-goals.** No package renames. No `State` restructuring. No `Model` interface change.

**Bit-exactness is per pinned tuple** (device, driver, TornadoVM version, backend, build)
— not a blanket property. Cross-backend comparison uses M1.5's tolerance. Goldens are
regenerated only by an explicit reviewed commit, never silently on failure.

**Depends on.** Phase 0.

---

## M2 — Metrics seam

**Objective.** Give metrics a home before the milestones that need evidence of "no
behaviour change".

| Task | Detail | Acceptance |
| --- | --- | --- |
| M2.1 | Metrics sink interface in the runtime layer (Rule 17) | No dependency on api/generation; the permitted edge is explicit |
| M2.2 | Tornado backend reports via `withProfiler(...)` + `getProfilerResult()` | Device kernel time, transfer bytes, device memory reach the sink |
| M2.3 | Sink off by default on the decode path | Benchmarked: no tok/s change when disabled |
| M2.4 | `RunMetrics` becomes one sink implementation | CLI output unchanged |
| M2.5 | Counters: load, prefill, decode, tokens/s | Available programmatically, not only printed |
| M2.6 | Logging sink (Rule 16), no-op by default | Rule 16 allowlist begins shrinking |

**Non-goals.** No exporters. No memory planning. Those are M13.

**Grounding.** Every value comes from TornadoVM's existing profiler API, referenced
nowhere in the project today.

**Depends on.** M1.4 — the first production change happens with goldens already in place.

---

## M3 — Public API façade

**Objective.** Give users the intended surface before the internals move.

**The façade is staged (ADR-007 D3).** Version 1 exposes no type designed by a later
milestone: `ModelOptions`/`SessionOptions` v1 carry `contextLength` only; the dtype
accessors on `ModelInfo` arrive in M4.7, `executionPolicy(...)` in M10.2,
`backend(...)`/`device(...)` in M12.1. Nothing placeholder-shaped is exposed to make
code compile.

**Decision gate.** M3 is not implementable until gates
[D-01, D-03, D-04, D-06, D-07 and D-11](decision-gates.md) are closed — they define the
façade's signatures (generation as a capability, prompt/messages interaction, device
selection in v1, low-level `forward`, experimental policy, model-close semantics).

| Task | Detail | Acceptance |
| --- | --- | --- |
| M3.1 | `api/`: `LocalModels`, `LocalModel` (+ generation capability per D-01), `GenerationSession`, `GenerationRequest/Result`, `ModelOptions`, `SessionOptions` — v1 surface only | The simple example in `public-api.md` compiles and runs, CPU and GPU |
| M3.2 | Delegate to existing `ModelLoader` / `InferenceEngine*` / `TornadoVMMasterPlan` | Token-identical output for a fixed seed |
| M3.3 | Remove console I/O from library paths the façade reaches | Rule 16 allowlist shrinks |
| M3.4 | Mark experimental in Javadoc | Removed in M13 |

**Non-goals.** No change to `Model`, `State`, `InferenceEngine*` or the plan hierarchy.
No removal of `runInstructOnce` / `runInstructOnceLangChain4J`.

**Compatibility risks.** Low — additive. The risk is committing to names early, hence the
experimental marker.

**Depends on.** M1.

---

## M4 — DataType and GGUF isolation

| Task | Detail | Acceptance |
| --- | --- | --- |
| M4.1 | Runtime `DataType` alongside `GGMLType` | Additive |
| M4.2 | Explicit `GGMLType → DataType` mapping, seeded from `effectiveGpuWeightType` and `getModelQuantization` | Directly tested — first time this logic is visible |
| M4.3 | `TensorDescriptor` — exact shape (full shape vs count + layout tag, block parameters) is gate [D-08](decision-gates.md), decided before this task starts | Loaders produce descriptors, then materialize storage; load time and resident memory unchanged |
| M4.4 | `Weights` / `FloatTensor` / `TornadoTensor` expose `DataType`; deprecate `GGMLType` accessors | Rule 4 allowlist shrinks |
| M4.5 | `ForwardPlanFactory` dispatches on `DataType` | Behaviour identical |
| M4.6 | Move `GGUF`, `GGMLTensorEntry`, `MetadataValueType`, `GGMLType` to a format package | Rule 4 allowlist empty |
| M4.7 | `ModelInfo` exposes weight **and** compute dtype | Reflects the K-quant → Q8_0 collapse honestly |

**Non-goals.** No new file formats. No shaped-tensor algebra — `FloatTensor` stays
shapeless. No change to how quantized kernels work.

**Watch.** Model load time (`RunMetrics.setLoadDuration`) and resident memory, not
throughput. The descriptor layer must not add a copy.

**Depends on.** M1. Independent of M6, so the two can run in parallel.

---

## M5 — Model provider SPI, part A (detection and loading)

| Task | Detail | Acceptance |
| --- | --- | --- |
| M5.1 | `ModelProvider` SPI (`supports` / `load`), `ServiceLoader` discovery. The load target is the **transitional internal `LoadTarget` adapter** (ADR-007 D4) — never `Backend`, which does not exist until M12; removed by M12.6 | A test-only provider is discovered by `ServiceLoader` and selected by `supports` on a synthetic metadata fixture; `LoadTarget` is not public API (visibility test); runs in plain `mvn test` |
| M5.2 | Per-family providers replacing `ModelType` load dispatch | All families load identically |
| M5.3 | Replace `detectModelType` substring matching on `general.name` | An unsupported model gives a clear error, not a wrong-family load |
| M5.4 | Migrate PR #120's family onto the SPI | Adding a family touches only new files + one registration |

**Non-goals.** `ModelType` need not be deleted — it may remain an internal identifier.
The `ForwardPlanFactory` branches are **not** in scope; they need the program layer (M11).

**Depends on.** M4.

---

## M6 — Session and state split

**The highest-risk milestone.** It touches every builder in `tornadovm/layers/**`.

| Task | Detail | Acceptance |
| --- | --- | --- |
| M6.1 | Session type owning sequence position and holding a KV lease | Two sessions from one model produce correct independent output |
| M6.2 | `KvCacheManager` + `BlockPool`, single-lease, behaviour-identical (ADR-005); exposes the **internal capacity contract** — total/free blocks and the byte budget `withMemoryLimit` bounds (ADR-007 D5) — that M7.2's admission consumes until M12.2 formalizes it | Goldens bit-identical; capacity numbers assert against pool sizing in a unit test |
| M6.3 | Remove plan ownership from `Model`; deprecate `tornadoVMPlan()` / `setTornadoVMPlan` | Rule 2 allowlist loses its `model/` entries |
| M6.4 | Split `State`: KV behind the lease; activations and scratch stay per-session | Gate green on FP16 + Q8_0, single-token + prefill/decode |
| M6.5 | Reimplement `InferenceService` on the session type | Server behaviour unchanged |
| M6.6 | Keep family-specific state only where required (e.g. `Qwen3State.wrapAttSplit`) | Each surviving field justified in review |

**The pool is one persistent array with in-kernel indexing** — an invariant, not a choice.
Handing a slot a different buffer per step breaks CUDA-graph replay, and
`recover.bailout=true` turns that into wrong output rather than an error
([C1](tornadovm-capabilities.md#c1--cuda-graph-capture-fixes-device-addresses)).

**Compatibility risks.** High — `Model` is used by LangChain4j and Quarkus integrations.
Deprecate first, notify those repositories.

**Performance risks.** Medium — buffer allocation and device-transfer boundaries move.

**Depends on.** M1; **M3 as a design prerequisite** — the session type must present
through the façade, and M6.5 reimplements `InferenceService` on it (ADR-001 migration
note 2; ADR-007 D8); **PR #138 must land first** (KV layout before KV ownership).
Blocking decision gates before M6.1/M6.2: [D-10, D-11, D-12, D-14](decision-gates.md).
Ownership semantics: [`ownership-and-lifecycle.md`](ownership-and-lifecycle.md).

---

## M7 — Engine tier

| Task | Detail | Acceptance |
| --- | --- | --- |
| M7.1 | Promote #129's paged mode onto `KvCacheManager` | Paged decode runs through the manager; CUDA-graph replay survives |
| M7.2 | `Scheduler` + admission, reserving against the same budget `withMemoryLimit` bounds | Continuous batching reproduces #129's numbers |
| M7.3 | `PrefixCache` — identity includes model, dtype, position offset; refcounting; leased blocks pinned | Prefix-reuse savings reproduced; eviction under a live lease tested |
| M7.4 | `LLMEngine` (`addRequest`, `step`) + non-blocking submit | Server moves onto the engine API |
| M7.5 | Serving metrics through the M2 sink: TTFT, queue wait, occupancy, block utilization, preemptions, admitted/rejected | Per-request and aggregate; TTFT records cache warm/cold |
| M7.6 | Retire `-Dbatch.decode.*` in favour of explicit policy — **lands after M10.2**, which creates the replacement `ExecutionPolicy` value (ADR-007 D6) | Same combinations expressible; old properties warn-and-map for one release |

**Non-goals.** No second compiler. No new kernels. Promotion, not reimplementation.

**Structure (ADR-007 D6).** M7 splits into a **minimum viable engine** (M7.1, M7.2,
M7.4 + server migration) and an **extension stage** (M7.3 prefix cache, M7.5 metrics,
preemption/adaptive batching, M7.6). The core alone must reproduce #129's
continuous-batching numbers. Behavioural contract, request state machine and shutdown
semantics: [`engine-contract.md`](engine-contract.md).

**Depends on (stated by kind, ADR-007 D6).**
- *PR land prerequisite:* #129 — the design needs its consumer in tree, not in a diff.
- *Milestone prerequisites:* M6 (all of M7); **M2 for M7.5** (the metrics sink).
- *Design prerequisites:* gates [D-13…D-19, D-24](decision-gates.md); the
  `ExecutionPolicy` shape for M7.6 (decided with M10, implemented in M10.2).

---

## M8 — Operation vocabulary

| Task | Detail | Acceptance |
| --- | --- | --- |
| M8.1 | Define RmsNorm, RoPE, MatVec/MatMul, Attention, Softmax, SwiGLU, ResidualAdd, EmbeddingLookup, VocabProjection, **Sample/ArgMax** | Each defined once, family-independent |
| M8.2 | Parameterize by `DataType` at description and dispatch level | Adding a scheme adds ≤ **2** dispatch classes per operation family **and one kernel set per dtype** — enforced by a dispatch-enumeration test, not review judgement |
| M8.3 | Express the CPU forward passes in terms of operations | Kernel bodies unchanged |
| M8.4 | Express the task-graph builders in terms of operations | Same task graphs produced |

**Non-goals, enforced in review.** **No kernel rewrites.** No operator registry, graph
optimizer or fusion rules. Any change to a kernel body in this milestone is rejected and
deferred.

**Why "one kernel set per dtype" and not full collapse.** TornadoVM compiles per concrete
native array type (`FloatArray`, `HalfFloatArray`, `Int8Array`, `FP8Array`) and Java has
no generics over primitives, so one kernel body cannot serve every scheme. The
per-(model × dtype × mode × MMA) *class* explosion collapses — 36 classes under
`tornadovm/layers/type/**` today — the per-dtype *kernel* set does not.

**FP4/MXFP4 is out of scope** and depends on upstream TornadoVM work: no array type, no
`MMAShape` entry, no PTX codegen.

**Depends on.** M4.

---

## M9 — Program and compiled program

| Task | Detail | Acceptance |
| --- | --- | --- |
| M9.1 | Write ArchUnit Rule 3 **before** the package exists | Fails on an accidental TornadoVM import |
| M9.2 | `InferenceProgram`, `ProgramComponent`, `ProgramSignature` — ordered, not a graph IR | No TornadoVM types |
| M9.3 | Tornado `compile(...)` reproducing today's graphs, starting with Llama + FP16 | Same graph count, task names, grid scheduler entries |
| M9.4 | `Invocation` binds inputs/outputs/state, allocates nothing | Per-token allocation profile unchanged |
| M9.5 | Migrate remaining families; delete each `*PlanComponents` only once proven | Goldens bit-identical throughout |

**Non-goals.** No graph IR, no loop IR, no scheduling moved out of TornadoVM.

**Compiled-program identity** relies on deterministic generated kernel source, which only
holds from the Phase 0 floor
([C3](tornadovm-capabilities.md#c3--generated-kernel-source-was-non-deterministic-before-52)).

**Depends on.** M6, M8.

---

## M10 — Execution policy consolidation

| Task | Detail | Acceptance |
| --- | --- | --- |
| M10.1 | One generation loop; prefill/decode/batch become policy | `InferenceEngineWith*` reduced to deprecated delegates |
| M10.2 | `ExecutionPolicy` value replaces class-init `static final` property reads | `LlamaApp.guardDeviceSample`'s ordering workaround unnecessary |
| M10.3 | Move `runInteractive` / `runInstructOnce*` out of `Model` | Rule 8a passes; `Model` free of `Options` and console I/O |
| M10.4 | Keep `llama.*` properties as *inputs* that configure a policy | Launcher and scripts keep working |
| M10.5 | Rename `InferenceEngine` — the name now belongs to the engine tier | No bare "engine" ambiguity left in code or docs |

**Watch.** Today's `static final` flags are constant-folded by the JIT. Resolve policy once
per session, never per token, and benchmark the decode loop specifically.

**Depends on.** M6, M9.

---

## M11 — Model provider SPI, part B

Removes the `ForwardPlanFactory` family branches, which need the program layer to have
something to return. **Exit:** Rule 15 passes.

**Depends on.** M5 **and** M9 (ADR-007 D7) — the provider SPI supplies the dispatch,
the program layer supplies what a provider returns.

---

## M12 — Backend and device SPI

| Task | Detail | Acceptance |
| --- | --- | --- |
| M12.1 | `Backend`, `Device`, `DeviceSelector`, `CompileOptions` | `--cuda` without `--gpu` silently running on CPU becomes inexpressible |
| M12.2 | Buffer lifetime classes (model / engine / invocation) + capacity query — formalizes M6.2's internal capacity contract on the SPI. *Prerequisite: M7 core has landed (it migrates the engine)* | Manager and admission reimplemented on the SPI query; admission behaviour unchanged (scripted-load test) |
| M12.3 | Move `tornadovm/**` → `backend/tornado/**`, `InferenceCore*` → `backend/cpu/**` (two mechanical PRs) | Rules 1 and 11 pass with empty allowlists; goldens bit-identical |
| M12.4 | Shard-plan seam; invocation targets a device set | Design section merged into `target-architecture.md`, reviewed against the M12.1 SPI signatures with recorded maintainer sign-off; no runtime code (ArchUnit confirms no new packages) |
| M12.5 | Verify shaded jar, native-image config, release automation, class-init order | Build and launcher unchanged; release dry-run artifacts structurally identical |
| M12.6 | Remove the transitional `LoadTarget` adapter — providers take `Backend` (closes ADR-007 D4) | No `LoadTarget` references remain; provider SPI tests green |

**Non-goals.** No new hardware backend. No abstraction over TornadoVM's own PTX/OpenCL/
SPIR-V backends — those stay device-level concerns.

**Compatibility risks.** High — a repository-wide import change.

**Depends on.** M9, M10.

---

## M13 — Memory planning, diagnostics, developer experience

Expanded into concrete tasks T13.1–T13.5 in the
[execution backlog](execution-backlog.md#m13--memory-planning-diagnostics-developer-experience):
memory-planning report with a pre-flight API (predicted vs actual within 5% on the
fixture matrix; over-capacity fails fast with the number); error-message catalogue;
exporter modules over the M2 sink; session-reuse guidance and the Javadoc pass; and the
experimental-marker removal, which is the API freeze and requires all staged façade
additions (M4.7, M10.2, M12.1) to have landed.

**Depends on.** M12.

---

## Dependency summary

No edge points backward (ADR-007). M-numbers order design intent, not calendar time —
parallel branches are marked.

```
Phase 0 ── M1 ─┬─ M2 ───────────────────────────── (feeds M7.5)
               ├─ M3 ──────────────┐ (design prereq of M6, ADR-007 D8)
               ├─ M4 ─┬─ M5 ─────┐ │
               │      └─ M8 ─┐   │ │
               └─────────────┼───┼─┴─ M6 ─┬─ M9 ─┬─ M10 ── M12 ── M13
                   (#138 lands first)     │      ├─ M11 ◄┘(also needs M5)
                                          │      │
                             (#129 lands first)  │
                                          └─ M7-core ── M7-ext
                                                  (M7.6 also needs M10.2;
                                                   M12.2 also needs M7-core)
```

Critical path: **Phase 0 → M1 → M3 → M6 → M9 → M10 → M12 → M13**.
M2, M4, M5 and M8 run in parallel with the critical path. **M7-core is the product
win**; its full prerequisites are M6 and PR #129, plus M2 for the metrics task M7.5
and the M10.2 policy value for M7.6.

## PR land order

1. **Phase 0** — the version bump precedes everything, including #120, whose BF16 path
   needs `BFloat16Array` (5.2.0+).
2. **#129** — source for M7; must land before the engine tier is built. Retarget from
   `feat/mma_cuda` to `main` first (still mistargeted as of 2026-08-03).
3. **#138** — KV layout before M6 changes KV ownership.
4. **#120** — before M5, as the live example for the provider SPI.
5. **#131** — last, additive, default-off, measured as parity.
6. **#142** (ci/metal-migration) — not in this roadmap; assess against M1.7's CI
   changes before either lands.

These are **land prerequisites** (merge order); they are distinct from the milestone
and design prerequisites stated per milestone above.

**Freeze declaration.** Once M6 opens, `inference/state/**`, `tornadovm/plan/**` and
`tornadovm/layers/type/**` are in refactor; feature work in those trees rebases rather
than merges. Once M7 opens, `bench/BatchedDecode*` and `server/**` join the freeze.

## What is deliberately not scheduled

- Repository-wide package rename as an early step — only M12, and only for the backend
  boundary.
- Multi-module Maven split — after M12, once the package boundaries hold. See
  [`target-architecture.md`](target-architecture.md#likely-maven-module-structure).
- Non-transformer use cases. M6, M8 and M9 must leave the door open
  ([Rule 14](dependency-rules.md#rule-14--core-abstractions-do-not-assume-generation)),
  but building embeddings or vision support is separate work.
- FP4/MXFP4 — blocked on upstream TornadoVM capability, tracked in
  [`tornadovm-capabilities.md`](tornadovm-capabilities.md#missing--genuine-upstream-proposals).
- New kernels and performance work on their own branches. This roadmap must not block
  them; the freeze declaration is the coordination mechanism.
