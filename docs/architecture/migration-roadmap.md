# Implementation Roadmap

**Status: agreed.** This is the plan implementation work is measured against, following
the ARCH-01..19 review on PR #140. Changing a milestone's objective or acceptance criteria
needs an ADR; re-ordering work inside a milestone does not.

Terms: [`terminology.md`](terminology.md). Rules: [`dependency-rules.md`](dependency-rules.md).
Capabilities and version floors: [`tornadovm-capabilities.md`](tornadovm-capabilities.md).

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

**Acceptance.** Build and launcher work on the new version; goldens generated **once** on
the new floor and committed as the baseline; a fresh `perf-history.jsonl` entry recorded,
since prior entries are not comparable.

**Compatibility risks.** Medium — the launcher and CI pin versions in several places.

**Performance risks.** None negative; #1002 and #1008 are large improvements. The risk is
*mistaking them for* refactor gains, which is why the gate tuple includes the version.

---

## M1 — Guardrails (tests only, no production code)

**Objective.** Make the boundaries machine-checked and give every later milestone a
numerical safety net. Nothing here can break the engine — it touches no production file.

| Task | Detail | Acceptance |
| --- | --- | --- |
| M1.1 | ArchUnit test module, test-scoped dependency | `mvn test` passes; build output unchanged |
| M1.2 | Rules 1, 2, 5, 7, 11 with enumerated allowlists | Fully-qualified class names, each with a milestone reference; no wildcards |
| M1.3 | Rule 8a with allowlist; Rule 16 (console I/O) with its 20 files | New console I/O in library code fails the build |
| M1.4 | **Golden logits** — Llama-3.2-1B × {FP16, Q8_0}, fixed prompt, greedy, pinned backend, `recover.bailout=False` | Re-run asserts bit-identical; goldens committed |
| M1.5 | CPU↔GPU parity, tolerance `\|got − ref\| ≤ 1e-2·Σ\|wᵢaᵢ\| + 1e-3` | Passes on both backends |
| M1.6 | Compiled-program identity test | One compile, ≥100 tokens, identity unchanged |
| M1.7 | **Benchmark gate** — add `machine`, `gpu`, `tornadovm_version`, `cache_warm` fields to `perf-history.jsonl`; gate script | Compares against last green run of the same tuple; tolerance stated per tuple |
| M1.8 | `AbstractModel.tokenizer` / `weights` / `chatFormat` final | Compiles; no behaviour change |

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

| Task | Detail | Acceptance |
| --- | --- | --- |
| M3.1 | `api/`: `LocalModels`, `LocalModel`, `GenerationSession`, `GenerationRequest/Result`, `ModelOptions`, `SessionOptions` | The simple example in `public-api.md` compiles and runs, CPU and GPU |
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
| M4.3 | `TensorDescriptor` (dtype + element count/shape + layout) | Loaders produce descriptors, then materialize storage |
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
| M5.1 | `ModelProvider` SPI (`supports` / `load`), `ServiceLoader` discovery | — |
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
| M6.2 | `KvCacheManager` + `BlockPool`, single-lease, behaviour-identical (ADR-005) | Goldens bit-identical |
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

**Depends on.** M1; **PR #138 must land first** (KV layout before KV ownership).

---

## M7 — Engine tier

| Task | Detail | Acceptance |
| --- | --- | --- |
| M7.1 | Promote #129's paged mode onto `KvCacheManager` | Paged decode runs through the manager; CUDA-graph replay survives |
| M7.2 | `Scheduler` + admission, reserving against the same budget `withMemoryLimit` bounds | Continuous batching reproduces #129's numbers |
| M7.3 | `PrefixCache` — identity includes model, dtype, position offset; refcounting; leased blocks pinned | Prefix-reuse savings reproduced; eviction under a live lease tested |
| M7.4 | `LLMEngine` (`addRequest`, `step`) + non-blocking submit | Server moves onto the engine API |
| M7.5 | Serving metrics through the M2 sink: TTFT, queue wait, occupancy, block utilization, preemptions, admitted/rejected | Per-request and aggregate; TTFT records cache warm/cold |
| M7.6 | Retire `-Dbatch.decode.*` in favour of explicit policy | Same combinations expressible |

**Non-goals.** No second compiler. No new kernels. Promotion, not reimplementation.

**Depends on.** M6; **PR #129 must land first** — the design needs its consumer in tree,
not in a diff. M7.6 is designed together with M10, since both retire process-global
switches.

---

## M8 — Operation vocabulary

| Task | Detail | Acceptance |
| --- | --- | --- |
| M8.1 | Define RmsNorm, RoPE, MatVec/MatMul, Attention, Softmax, SwiGLU, ResidualAdd, EmbeddingLookup, VocabProjection, **Sample/ArgMax** | Each defined once, family-independent |
| M8.2 | Parameterize by `DataType` at description and dispatch level | Adding a scheme adds ≤ k dispatch classes **and one kernel set per dtype** |
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

**Depends on.** M9.

---

## M12 — Backend and device SPI

| Task | Detail | Acceptance |
| --- | --- | --- |
| M12.1 | `Backend`, `Device`, `DeviceSelector`, `CompileOptions` | `--cuda` without `--gpu` silently running on CPU becomes inexpressible |
| M12.2 | Buffer lifetime classes (model / engine / invocation) + capacity query | Engine admission consumes the capacity query |
| M12.3 | Move `tornadovm/**` → `backend/tornado/**`, `InferenceCore*` → `backend/cpu/**` | Rules 1 and 11 pass with empty allowlists |
| M12.4 | Shard-plan seam; invocation targets a device set | Design-only, no implementation |
| M12.5 | Verify shaded jar, native-image config, release automation, class-init order | Build and launcher unchanged |

**Non-goals.** No new hardware backend. No abstraction over TornadoVM's own PTX/OpenCL/
SPIR-V backends — those stay device-level concerns.

**Compatibility risks.** High — a repository-wide import change.

**Depends on.** M9, M10.

---

## M13 — Memory planning, diagnostics, developer experience

Memory planning that reports required device memory before allocating; error messages
that name the problem; exporters; session-reuse guidance; Javadoc; experimental marker
removed.

**Depends on.** M12.

---

## Dependency summary

```
Phase 0 ── M1 ─┬─ M2 ──────────────────── (feeds M7.5)
               ├─ M3
               ├─ M4 ─┬─ M5
               │      └─ M8 ─┐
               └─ M6 ─┬──────┴─ M9 ─┬─ M10 ── M12 ── M13
        (needs #138)  │             └─ M11
                      └─ M7  (needs #129)
```

Critical path: **Phase 0 → M1 → M6 → M9 → M10 → M12 → M13**.
M2, M3, M4, M5 are cheap and parallel. **M7 is the product win** and depends only on M6.

## PR land order

1. **Phase 0** — the version bump precedes everything, including #120, whose BF16 path
   needs `BFloat16Array` (5.2.0+).
2. **#129** — source for M7; must land before the engine tier is built.
3. **#138** — KV layout before M6 changes KV ownership.
4. **#120** — before M5, as the live example for the provider SPI.
5. **#131** — last, additive, default-off, measured as parity.

**Freeze declaration.** Once M6 opens, `inference/state/**`, `tornadovm/plan/**` and
`tornadovm/layers/type/**` are in refactor; feature work in those trees rebases rather
than merges.

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
