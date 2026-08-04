# Execution Backlog

**Status: normative (companion to [`migration-roadmap.md`](migration-roadmap.md);
binding since ADR-007's acceptance, 2026-08-03).** Milestones are epics; this document breaks them into
PR-sized, independently mergeable tasks. A task is self-contained: its acceptance can
pass when it merges, without waiting for a later task. Tasks whose gate is open
(**[D-nn]**, see [`decision-gates.md`](decision-gates.md)) must not start.

Field key — **Prereq** distinguishes: *land* = a PR/task must be merged first;
*design* = a decision or contract must exist first; *parallel-ok* = may proceed
alongside the named work. Gate classes A/B/C: [`verification-gates.md`](verification-gates.md).

Unless a task states otherwise: rollback = revert the PR (all tasks are additive or
mechanical until stated); compatibility = no public surface removed, deprecations only.

---

## Phase 0 — TornadoVM version floor

**T0.1 — Version bump**
- Objective: `tornadovm.base.version` 5.0.0 → ≥ 5.2.x everywhere it is pinned.
- Prereq: none. Owner is doing this himself (HANDOFF).
- Scope: `pom.xml`, `llama-tornado`, CI workflows, `Makefile`.
- Contract: new version floor; capability ledger's "reachable today" column becomes true.
- Acceptance (Class A + manual): build passes; launcher runs FP16 + Q8_0 on CUDA;
  sanity numbers move as predicted (~53→~103 tok/s, start-up ~11.5→~5.2 s) — if they
  don't, the bump didn't take.
- Perf gate: record a fresh `perf-history.jsonl` entry; prior entries are not baselines (C5).
- Non-goals: no new dtypes used; **no golden generation** (that is T1.4, per ADR-007 D1).
- Follow-up: T1.4 generates goldens on this floor.

---

## M1 — Guardrails

Tests only, except T1.8 (declared exception, ADR-007 D2).

**T1.1 — ArchUnit module**
- Objective: test-scoped ArchUnit dependency + empty rule scaffold.
- Prereq: none (parallel-ok with Phase 0).
- Scope: `pom.xml` (test scope), `src/test/java/.../arch/`.
- Acceptance (A): `mvn test` passes; build output (shaded jar) byte-comparable to before.

**T1.2 — Rules 1, 2, 5, 7, 11 with allowlists**
- Objective: the five enforceable-now structural rules, allowlisted by FQCN.
- Prereq: land T1.1.
- Acceptance (A): rules fail on a deliberately-violating test fixture; every allowlist
  entry carries a milestone reference; no wildcards; CI fails on stale entries.

**T1.3 — Rules 8a and 16**
- Objective: generation-policy and console-I/O rules with today's 20-file allowlist.
- Prereq: land T1.1.
- Acceptance (A): new console I/O in library code fails the build; device sampler is
  **not** on 8a's allowlist (Rule 8b).

**T1.4 — Golden infrastructure + goldens**
- Objective: golden capture/compare per [`verification-gates.md`](verification-gates.md#golden-logits-m14),
  goldens committed for Llama-3.2-1B × {FP16, Q8_0} on the pinned tuple.
- Prereq: land T0.1 (goldens on the old floor are invalid); land T1.1 (test module).
- Scope: `src/test/.../golden/`, `scripts/regenerate-goldens.sh`, committed golden blobs.
- Acceptance (B): re-run asserts bit-identical on the pinned tuple; NaN/Inf fails;
  absent hardware ⇒ explicit skip, not pass.

**T1.4-FP16 — FP16 logits determinism defect** *(discovered production defect)*
- Objective: find the earliest divergent operation in the FP16 GPU path and either fix it
  or record an explicit, reasoned acceptance.
- Status: **root cause found and fixed (2026-08-04).** Full record in
  [`review/fp16-determinism-investigation.md`](review/fp16-determinism-investigation.md).
- Cause: `reductionOneBlockWithLayer` combines the per-workgroup RMS partial sums inside the
  kernel with no inter-workgroup synchronization. That combine is only safe when the separate
  `reductionFinalNormalization` task follows it, which the NVIDIA path skipped — so the racy
  value was the final scale. Quantization-independent: FP16 lost the race ~11% of the time,
  Q8_0 rarely at 20GB and ~1-in-4 at 12GB.
- Fix: NVIDIA path now uses `reductionOneBlockWithLayerSingleGroup` (one workgroup) via
  `rmsReduceKernel()`/`rmsReduceWorker()`, across every FFN and logits layer class.
- Verified: 0/300 identical executions diverge for FP16 and Q8_0 at both 20GB and 12GB
  (was 33/300 for FP16); layer-0 stage buffers 0/300 (was 31/300); no throughput cost.
- Goldens regenerated 2026-08-04: both fixtures now record `bit_exact: true`, so
  `GoldenLogitsAccelTest` asserts per-row hashes and the final row bitwise — an assertion that
  had never run before. T1.5 passes as well, see [`review/cpu-gpu-parity.md`](review/cpu-gpu-parity.md).

**T1.5 — CPU↔GPU parity test**
- Prereq: land T1.4 (shares fixture plumbing).
- Acceptance (B): tolerance bound passes on both backends of the pinned tuple.
- Status: **passing (2026-08-04).** Full record in
  [`review/cpu-gpu-parity.md`](review/cpu-gpu-parity.md).
- Cause of the original failure: the GPU RoPE kernel computed its frequencies from a hardcoded
  base of 50000 while the model's `rope_theta` is 500000, so every rotation angle differed from
  the CPU's precomputed `freq_cis` tables. Fixed by switching Llama/Mistral (FP16 and Q8_0) to
  `ropeRotationWithCacheCopyPrecomputed`.
- Gate redesigned: elementwise `atol + rtol·|cpu|` with a violation budget, a hard max-error
  ceiling, whole-vector relative L2 and cosine, and decision-level argmax/top-k with the competing
  tokens' gap. Thresholds are per quantization and measured with `golden/ParityProfile`.
- Remaining: the same hardcoded RoPE base exists in the batch-prefill, Phi3 and Qwen3 kernels
  (latent, since those constants currently match their models); batch prefill is separately broken.
- Gate now runs on the pinned tuple: the accel-tests profile pins the CUDA backend, without which
  a multi-backend SDK defaults to OpenCL and the golden test degrades itself to token ids only.

**T1.6 — Compiled-program identity test**
- Prereq: land T1.4 (fixture) and T0.1 (C3 determinism).
- Acceptance (B): per [`verification-gates.md`](verification-gates.md#compiled-program-identity-m16).

**T1.7 — Benchmark gate**
- Objective: extend `perf-history.jsonl` schema (`machine`, `gpu`, `tornadovm_version`,
  `cache_warm`); gate script + per-tuple tolerances file.
- Prereq: land T0.1. Check PR #142's CI changes for conflicts before starting.
- Acceptance (A for the script's own tests; C for a real gated run): gate compares last
  green same-tuple entry; missing baseline ⇒ record-only.

**T1.8 — `AbstractModel` fields final** *(the declared production exception)*
- Objective: `tokenizer` / `weights` / `chatFormat` final.
- Prereq: land T1.4 (goldens in place before the first production change).
- Acceptance (A + B): compiles; goldens bit-identical.
- Rollback: revert; zero behavioural surface.

---

## M2 — Metrics seam

Tasks T2.1–T2.6 = roadmap M2.1–M2.6, unchanged in scope; each is one PR. Prereq for
T2.1: land T1.4. Executable acceptance already stated per task in the roadmap; the only
addition: **T2.3's "no tok/s change when disabled" uses the T1.7 gate on the pinned
tuple** (Class C), tolerance per tuple.

---

## M3 — Public API façade (staged, ADR-007 D3)

**T3.0 — Close the M3 decision gates**
- Objective: decisions for D-01, D-03, D-04, D-06, D-07, D-11 recorded (ADR amendment
  or PR-recorded maintainer decision).
- Prereq: none — can happen now.
- Acceptance: [`decision-gates.md`](decision-gates.md) rows updated to Closed with links.
- This task blocks T3.1. **M3 is not implementable while these are open.**

**T3.1 — Façade types, v1 surface only**
- Objective: `api/`: `LocalModels`, `LocalModel` (+ `TextGenerationModel` if D-01 says
  so), `GenerationSession`, `GenerationRequest/Result`, `ModelOptions`/`SessionOptions`
  (contextLength only), `ModelInfo` v1 (no dtypes — added by T4.7).
- Prereq: design T3.0; land T1.2 (rules watch the new package).
- Contract: façade v1 per [`public-api.md`](public-api.md); no type from M4/M10/M12 in
  any signature.
- Acceptance (A): types compile; ArchUnit clean; javadoc states thread-safety per type
  and matches [`ownership-and-lifecycle.md`](ownership-and-lifecycle.md).
- Non-goals: no delegation yet (T3.2); no policy/backend/dtype surface.

**T3.2 — Delegating adapters**
- Objective: façade delegates to `ModelLoader` / `InferenceEngine*` /
  `TornadoVMMasterPlan`; the session wraps today's `State` + plan (ADR-001 note 2).
- Prereq: land T3.1.
- Acceptance (B): the simple example runs CPU and GPU; token-identical output for a
  fixed seed; goldens unchanged.
- Rollback/legacy: `runInstructOnce*` untouched and remains the fallback path.
- Follow-up: T6.1 replaces the wrapper session's internals; adapter internals deleted in M10.

**T3.3 — Console I/O off façade paths**
- Acceptance (A): Rule 16 allowlist shrinks; CLI output unchanged.

**T3.4 — Experimental marker**
- Objective: `@Experimental` (or javadoc marker per D-07) on every `api/` type.
- Acceptance (A): marker present; removal tracked by T13.5.

---

## M4 — DataType and GGUF isolation

**T4.0 — Close D-08, D-09** (descriptor shape; DataType value set). Blocks T4.1/T4.3.

**T4.1 — Runtime `DataType`** — prereq: design T4.0. Acceptance (A): additive; unused.

**T4.2 — `GGMLType → DataType` mapping** — seeded from `effectiveGpuWeightType` +
`getModelQuantization`. Acceptance (A): direct unit tests incl. the K-quant → Q8_0
collapse; first time this logic is visible.

**T4.3 — `TensorDescriptor`** — prereq: design T4.0; land T4.1. Acceptance (A + B):
loaders produce descriptors then materialize; load time and resident memory unchanged
(watch metrics, not throughput); no extra copy.

**T4.4 — `DataType` accessors on `Weights`/`FloatTensor`/`TornadoTensor`; deprecate
`GGMLType` accessors** — Acceptance (A): Rule 4 allowlist shrinks; deprecated, not deleted.

**T4.5 — `ForwardPlanFactory` dispatches on `DataType`** — Acceptance (B): goldens
bit-identical; same plan classes chosen (assert by identity test).

**T4.6 — Move GGUF types to `format/`** — Acceptance (A): Rule 4 allowlist empty and
deleted; imports-only mechanical PR.

**T4.7 — `ModelInfo` dtype accessors** *(this is the M3-façade extension point)* —
prereq: land T3.1, T4.2. Acceptance (A): both dtypes exposed; a Q6_K file reports
weight=Q8_0-storage honestly per ADR-004.

---

## M5 — Model provider SPI, part A

**T5.0 — Close D-21** (one SPI or two). Blocks T5.1.

**T5.1 — `ModelProvider` SPI + discovery**
- Objective: `supports(ModelSource)` / `load(ModelSource, ModelOptions, LoadTarget)`;
  `ServiceLoader` discovery. **`LoadTarget` is the named transitional internal adapter
  (ADR-007 D4)** — package-private, wraps the use-TornadoVM boolean + device selection;
  removed by T12.6.
- Prereq: design T5.0; land T4.6 (`ModelSource` is format-layer).
- Acceptance (A — now executable, was "—" in the roadmap): a test-only provider on the
  test classpath is discovered by `ServiceLoader`; `supports` dispatch selects it for a
  synthetic metadata fixture; `LoadTarget` is not public API (ArchUnit visibility test);
  `mvn test` green with no accelerator.

**T5.2 — Per-family providers** — one PR per 2–3 families, legacy `ModelType.loadModel`
retained as fallback until the last family migrates. Acceptance (B): all families load
identically (golden load-metadata comparison); fallback still selectable by system
property for one release.

**T5.3 — Replace `detectModelType` substring matching** — Acceptance (A): unsupported
model ⇒ clear error naming the metadata seen, not a wrong-family load; test with a
doctored `general.name`.

**T5.4 — Migrate PR #120's family** — prereq: land #120. Acceptance (A): adding the
family touches only new files + one registration (assert by diff review checklist).

---

## M6 — Session and state split

Highest risk. Prereq for all: land T1.4–T1.7 (nets), land #138, land T3.2 (façade
session to migrate). Freeze active: `inference/state/**`, `tornadovm/plan/**`,
`tornadovm/layers/type/**`.

**T6.0 — Close D-10, D-11, D-12, D-14.** Blocks T6.1/T6.2.

**T6.1 — Session type with lease handle**
- Objective: real session owning position + holding a `KvLease`; lease is initially a
  façade over today's per-`State` cache (single lease).
- Acceptance (B): two sessions from one model produce correct independent output
  (serialized per D-12); goldens bit-identical.

**T6.2 — `KvCacheManager` + `BlockPool`, single-lease**
- Objective: manager owns the pool (one persistent array, C1); **exposes the internal
  capacity contract (total/free blocks, byte budget) per ADR-007 D5**; model-scoped per
  D-10 recommendation.
- Acceptance (B): behaviour-identical; goldens bit-identical; capacity numbers assert
  against pool sizing in a unit test (A).
- Follow-up: T12.2 formalizes capacity on the SPI.

**T6.3 — Plan ownership off `Model`** — deprecate `tornadoVMPlan()`/`setTornadoVMPlan`.
Acceptance (A + B): Rule 2 allowlist loses `model/` entries; LangChain4j/Quarkus
integrations notified (compatibility note in PR).

**T6.4.1 — `State` split, core + Llama {FP16, Q8_0}**
- Objective: KV behind the lease; activations/scratch stay per-session; only Llama's
  builders migrated; every other family runs the retained legacy `State` path.
- Acceptance (B): gate green on FP16 + Q8_0, single-token + prefill/decode, Llama;
  legacy families' goldens untouched.
- Rollback: legacy path is the fallback; a flag flips Llama back.

**T6.4.2 / T6.4.3 / T6.4.4 — Family migrations** — Qwen2+Qwen3; Phi3+Granite;
Mistral+Devstral. Same shape as T6.4.1; one PR each; legacy `State` deleted only in the
last one, after all families prove out.

**T6.5 — `InferenceService` on sessions** — Acceptance (B): server behaviour unchanged
(same HTTP responses for a scripted conversation, fixed seed).

**T6.6 — Family-state audit** — each surviving family-specific field justified in
review (e.g. `Qwen3State.wrapAttSplit`); unjustified copies removed. Acceptance:
review checklist + goldens.

---

## M7 — Engine tier

Prereq for all: land M6 (through T6.5); land #129 (retargeted to `main` first — it
still targets `feat/mma_cuda`, verified 2026-08-03). Contract:
[`engine-contract.md`](engine-contract.md). Freeze extends to `bench/BatchedDecode*`.

**T7.0 — Close D-13…D-19, D-24.** Blocks the rest of M7.

### M7-core (minimum viable engine)

**T7.1 — Paged mode onto the manager**
- Objective: promote #129's block table/paged decode onto `KvCacheManager`; block-table
  adaptation is *this* task, isolated from scheduling.
- Acceptance (B): paged decode through the manager; CUDA-graph replay survives —
  including the ADR-005 test: evict under a captured graph and assert a caught failure,
  not wrong output (`recover.bailout=False`).

**T7.2 — Scheduler + admission**
- Objective: FCFS, fixed B, admission reserving against T6.2's capacity contract
  (the `withMemoryLimit` budget).
- Acceptance (B + C): continuous batching reproduces #129's numbers on the pinned tuple
  (gate tolerance); queue-full backpressure behaviour per contract (A, fake backend).

**T7.3 — `LLMEngine` + handles + `step()`** *(roadmap M7.4)*
- Acceptance (A): full request state machine exercised against a fake backend — every
  allowed transition, no forbidden one (state-machine property test); (B) end-to-end on
  device.

**T7.4 — Server onto the engine API** — Acceptance (B): scripted-conversation parity;
concurrent requests no longer serialize behind one lock (throughput test, Class C).

### M7-ext

**T7.5 — `PrefixCache`** *(roadmap M7.3)* — refcounting, pinning, identity =
(model, dtype, position offset). Acceptance (B): #129's prefix savings reproduced;
eviction under live lease tested.

**T7.6 — Serving metrics through the M2 sink** — prereq: land T2.x. Acceptance:
TTFT (with cache warm/cold), queue wait, occupancy, block utilization, admitted/rejected
available programmatically; per-request subset on the result type.

**T7.7 — Retire `-Dbatch.decode.*`** *(roadmap M7.6)* — prereq: **land T10.2**
(`ExecutionPolicy` exists). Acceptance: same combinations expressible; old properties
warn-and-map for one release, then removed (follow-up T10.6).

---

## M8 — Operation vocabulary

Prereq: land M4. No kernel rewrites — enforced in review; any kernel-body diff rejects
the PR.

**T8.0 — Close D-22** (dequantization home). Blocks T8.2.

**T8.1 — Vocabulary types** — RmsNorm, RoPE, MatVec/MatMul, Attention, Softmax, SwiGLU,
ResidualAdd, EmbeddingLookup, VocabProjection, Sample/ArgMax; each defined once,
family-independent. Acceptance (A): types + javadoc; no TornadoVM imports (Rule 14 pre-check).

**T8.2 — `DataType` parameterization**
- Acceptance (A — now executable, replacing the subjective "≤ k"): **k = 2** — an
  ArchUnit/reflection test asserts that registering a new dtype for an existing
  operation adds at most 2 dispatch classes per operation family and one kernel set;
  the test enumerates dispatch registrations and fails if a family×dtype pair requires
  more.

**T8.3 — CPU forward passes as operations** — one PR per family pair (same grouping as
T6.4.x); kernel bodies unchanged (diff-scope check). Acceptance (B): goldens bit-identical.

**T8.4 — Task-graph builders as operations** — same grouping; Acceptance (B): same task
graphs produced (identity test: graph count, task names, grid entries unchanged).

---

## M9 — Program and compiled program

Prereq: land M6, M8. **T9.0 — Close D-20, D-23** first.

**T9.1 — Rule 3 before the package** — Acceptance (A): rule fails on a fixture with an
accidental TornadoVM import.

**T9.2 — `InferenceProgram` / `ProgramComponent` / `ProgramSignature`** — ordered, not
a graph. Acceptance (A): no TornadoVM types; inspectable/loggable/comparable (equals test).

**T9.3 — Tornado `compile(...)`, Llama + FP16** — Acceptance (B): same graph count,
task names, grid entries; goldens bit-identical; all three execution modes.

**T9.4 — `Invocation`** — binds, never allocates. Acceptance (B): per-token allocation
profile unchanged (allocation-tracking test on the decode loop).

**T9.5.x — Family/dtype verticals** — Llama Q8_0; Qwen2/3 both dtypes; Phi3/Granite;
Mistral/Devstral. One PR each; each deletes its `*PlanComponents` **only after** its
program equivalent is proven (goldens + identity); until then the legacy factory path
remains the fallback.

---

## M10 — Execution policy consolidation

Prereq: land M6, M9. **T10.0 — Close D-02** first.

**T10.1 — One generation loop** — `InferenceEngineWith*` become deprecated delegates.
Acceptance (B): goldens across all three modes.

**T10.2 — `ExecutionPolicy` value** — replaces class-init `static final` reads;
resolved once per session, never per token. Acceptance (B + C): `guardDeviceSample`
ordering workaround removable; decode-loop benchmark within gate tolerance (JIT
constant-fold risk is the watch item). *Unblocks T7.7 and the façade's
`executionPolicy(...)` builders (added here, per ADR-007 D3).*

**T10.3 — Loops out of `Model`** — Acceptance (A): Rule 8a passes; `Model` free of
`Options` and console I/O.

**T10.4 — `llama.*` properties as policy inputs** — launcher/scripts keep working
(script smoke test).

**T10.5 — Rename `InferenceEngine`** — Acceptance: no bare "engine" ambiguity left
(docs grep + deprecation shims).

**T10.6 — Remove `-Dbatch.decode.*` mapping shims** — follow-up removing T7.7's
transitional mapping.

---

## M11 — Model provider SPI, part B

Prereq: land M5 **and** M9 (ADR-007 D7). Removes `ForwardPlanFactory` family branches;
per-family PRs mirroring T9.5.x. Exit (A): Rule 15 passes.

---

## M12 — Backend and device SPI

Prereq: land M9, M10. T12.2 additionally prereq: land M7-core (it migrates the engine).

**T12.1 — SPI types** — `Backend`, `Device`, `DeviceSelector`, `CompileOptions`; façade
gains `backend(...)`/`device(...)` builders (ADR-007 D3). Acceptance (A + B): `--cuda`
without `--gpu` inexpressible; existing launcher flags map onto selectors.

**T12.2 — Buffer lifetimes + capacity query** — formalizes T6.2's internal contract on
the SPI; manager and admission reimplemented on it. Acceptance (A + B): admission
behaviour unchanged (same admit/reject sequence on a scripted load test).

**T12.3a — `tornadovm/**` → `backend/tornado/**`** — imports-only mechanical PR.
Acceptance (A + B): Rules 1/11 allowlists empty; goldens; shaded jar contents diff
reviewed. Rollback: revert is clean because the PR is mechanical.

**T12.3b — `InferenceCore*` → `backend/cpu/**`** — same shape, separate PR.

**T12.4 — Shard-plan seam (design-only)**
- Acceptance (now executable): a design section merged into
  [`target-architecture.md`](target-architecture.md) specifying: invocation targets a
  device *set*; shard-plan shape; KV-manager blocks-per-device implications — reviewed
  against the SPI signatures of T12.1 with a recorded maintainer sign-off. No runtime
  code; ArchUnit confirms no new packages.

**T12.5 — Build verification** — shaded jar, native-image config, release automation,
class-init order (the `guardDeviceSample` hazard is gone after T10.2; verify).
Acceptance: release dry-run artifacts identical in structure.

**T12.6 — Remove `LoadTarget`** — providers take `Backend`; transitional adapter
deleted (closes ADR-007 D4). Acceptance (A): no references remain; provider SPI tests
green.

---

## M13 — Memory planning, diagnostics, developer experience

Prereq: land M12. Expanded from one paragraph into tasks:

**T13.1 — Memory planning report** — required device memory computed from descriptors +
policy **before** allocating; `LocalModels` pre-flight API. Acceptance (A + B): predicted
vs actual within 5% for the fixture matrix; over-capacity load fails fast with the
number, not an OOM.

**T13.2 — Diagnostics and error messages** — every user-reachable failure names the
problem (device not found, model too large, unsupported quant, context overflow).
Acceptance (A): error-message catalogue test.

**T13.3 — Metrics exporters** — off-core exporter module(s) over the M2 sink.
Acceptance: sink consumers documented; core gains no dependency (Rule 17).

**T13.4 — Session-reuse guidance + Javadoc pass** — Acceptance: javadoc complete on
`api/`; ownership matrix and javadoc agree (review checklist).

**T13.5 — Experimental marker removal** — the API freeze. Prereq: D-07's policy
satisfied; all façade-staging additions (T4.7, T10.2, T12.1) landed. Acceptance:
marker gone; compatibility policy documented for 1.0.

---

## Dependency graph (no backward edges)

```
Phase 0 ─ M1 ─┬─ M2 ─────────────────────────────┐ (T7.6 metrics)
              ├─ M3* ─┐                          │
              ├─ M4 ─┬┼─ M5 ──────────────┐      │
              │      ││                   │      │
              │      │└──────┐            │      │
              │      └─ M8 ─┐│            │      │
              └───────┬─────┼┴────────────┼──────┼──────────────┐
                      ▼     ▼             │      │              │
   (#138 land) ────► M6 ─┬─ M9 ─┬─ M10 ─┬─┼──────┼─ M12 ── M13  │
                         │      └─ M11 ◄──┘      │    ▲         │
   (#129 land) ────────► M7-core ── M7-ext ◄─────┘    │         │
                         │        (T7.7 also needs T10.2)       │
                         └────────────────────────────► T12.2 ◄─┘

   M3* is also a design prerequisite of M6 (ADR-007 D8).
```

Legend: every arrow points from earlier to later. The three formerly backward edges are
resolved by: goldens moved out of Phase 0 (ADR-007 D1); `LoadTarget` transitional
adapter in M5 (D4); staged capacity contract M6.2 → M12.2 (D5); staged façade
M3 → M4.7/M10.2/M12.1 (D3); T7.7 sequenced after T10.2 (D6).

**Critical path:** Phase 0 → M1 → M3 → M6 → M9 → M10 → M12 → M13.
**Parallel-ok:** M2, M4→M5, M8 against the critical path; M7-core after M6 in parallel
with M9. **M7-core is the product win**; its full prerequisites are M6 + #129 (and M2
only for T7.6).

## PR land order (unchanged from the roadmap, confirmed by ADR-007 D9)

1. Phase 0 (version bump) — precedes everything including #120.
2. **#129** — retarget from `feat/mma_cuda` to `main` first (verified still mistargeted
   2026-08-03); source for M7.
3. **#138** — KV layout before M6 changes KV ownership.
4. **#120** — before T5.4.
5. **#131** — last, additive, default-off.
6. **#142** (ci/metal-migration) — not in the roadmap; assess against T1.7's CI changes
   before either lands.

**Freezes.** When M6 opens: `inference/state/**`, `tornadovm/plan/**`,
`tornadovm/layers/type/**`. When M7 opens: additionally `bench/BatchedDecode*` and
`server/**`. Feature work in frozen trees rebases rather than merges.
