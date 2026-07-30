# ARCH responses — positions on the baseline review

Responses to `ARCH-issues-mikepapadim.md` (review of baseline commit `7ee6f86`).

One position per issue: **Accept**, **Accept with modification**, **Reject**, **Postpone**. Positions are
proposals for the joint review, not decisions — nothing here changes `docs/architecture/` until we agree
per issue and fold the accepted ones in.

Every position is grounded in a TornadoVM capability that exists, or names the capability that does not
exist and would have to be proposed upstream. API references were checked against the local TornadoVM
tree (`~/TornadoVM`, `5.2.1-jdk21-dev`); `pom.xml` pins `tornadovm.base.version` `5.0.0`. Where the two
could differ, the position says so.

**Note on scope:** the PR body and title say 15 issues (ARCH-01..15); the file contains **19** — its own
header table lists ARCH-16..19 (metrics seam, observability ordering, logging policy, serving metrics).
The four extra are substantive and answered here. ARCH-16 is the best issue in the set.

## Summary

| # | Title | Severity (his) | Position | Severity (mine) |
|---|---|---|---|---|
| ARCH-01 | KV ownership prevents paged/shared-prefix KV | Critical | **Accept** | Critical |
| ARCH-02 | No engine layer for continuous batching | Critical | **Accept** | Critical |
| ARCH-03 | Rule 8 forbids the existing on-device sampler | High | **Accept** | High |
| ARCH-04 | No phase promotes the batched-decode engine | High | **Accept** | High |
| ARCH-05 | Quantization/dtype seam excluded from scope | High | **Accept with modification** | High |
| ARCH-06 | Phase 1 has no numerical regression net | High | **Accept with modification** | High |
| ARCH-07 | `GenerationSession` cannot express concurrent serving | High | **Accept with modification** | High |
| ARCH-08 | Prefix cache has no owner | Medium | **Accept** | High |
| ARCH-09 | Backend SPI has no device-memory seam | Medium | **Accept** | High |
| ARCH-10 | SPI does not express native library dispatch | Medium | **Accept (narrowed)** | Low |
| ARCH-11 | Rule 13 has no benchmark gate | Medium | **Accept with modification** | Medium |
| ARCH-12 | Provider SPI lands at phase 8 | Medium | **Accept with modification** | Medium |
| ARCH-13 | In-flight PRs not sequenced | Medium | **Accept** | Medium |
| ARCH-14 | Multi-device absent | Low | **Accept (seam only)** | Medium |
| ARCH-15 | API does not expose quantization | Low | **Accept** | Low |
| ARCH-16 | No metrics seam | High | **Accept** | High |
| ARCH-17 | Observability scheduled last | Medium | **Accept** | Medium |
| ARCH-18 | No logging policy | Medium | **Accept with modification** | Medium |
| ARCH-19 | Serving metrics undefined | Medium | **Accept** | Medium |

Tally: 12 Accept, 6 Accept with modification, 1 Accept-narrowed, 0 Reject. The modifications carry
partial rejections of specific sub-claims — flagged inline as **Reject the sub-claim**.

Two severity changes: ARCH-08 and ARCH-09 up to High (both are preconditions for ARCH-01/02, so they are
not optional extras); ARCH-10 down to Low (#131 measured parity for the decode path); ARCH-14 up to
Medium (TornadoVM already has the multi-device API, so the SPI shape decision is live now, not later).

---

## TornadoVM capability ledger

The review's asks divide cleanly into three groups. This is the grounding for every position below.

### Already available — GPULlama simply does not use it

| Capability | TornadoVM API | Serves | Used in `main` today? |
|---|---|---|---|
| Execution profiling | `TornadoExecutionPlan.withProfiler(ProfilerMode)`, `TornadoExecutionResult.getProfilerResult()` → `TornadoProfilerResult.getDeviceKernelTime()`, `getDeviceWriteTime()`, `getDeviceReadTime()`, `getTotalBytesCopyIn/Out()`, `getTotalDeviceMemoryUsage()`, `getCompileTime()` | ARCH-16, 17, 19, 11 | **No — 0 references** |
| Device enumeration | `TornadoDeviceMap` (`getNumBackends`, `getAllBackends`, `getDevicesByType`, `getBackendsWithDevicePredicate`) | ARCH-14, `DeviceSelector` | No |
| Device memory query | `TornadoTargetDevice.getDeviceGlobalMemorySize()`, `getDeviceMaxAllocationSize()`, `getDeviceLocalMemorySize()` | ARCH-09, memory planning | No |
| Memory limits | `withMemoryLimit(String)` / `withoutMemoryLimit()`; `memory/` — `TornadoMemoryProvider`, `XPUBuffer`, `DeviceBufferState` | ARCH-09 | No |
| Explicit device placement | `withDevice(TornadoDevice)`, **per-task** `withDevice(String taskName, TornadoDevice)`, `withConcurrentDevices()` | ARCH-14 | No |
| Native library tasks | `TaskGraph.task(...)` with a library binding factory (`CuBlas::cublasSgemv` style); modules `tornado-cublas`, `tornado-cudnn` | ARCH-10 | Only in PR #131 |
| Tensor-core MMA | `enums/MMAShape` — `M16N8K16` (fp16, bf16), `M16N8K32` (int8, FP8 E4M3/E5M2) | ARCH-05 | Yes, in the MMA layer classes |
| Narrow dtypes | `types/arrays/` — `HalfFloatArray`, `BFloat16Array`, `Int8Array`, `FP8Array`, `ByteArray` | ARCH-05, #120 BF16 | FP16/Q8_0 paths |
| CUDA graph capture | `withCUDAGraph()` | ARCH-09 (stable addresses) | Yes (`llama.cudaGraphs`) |
| Selective graph execution | `withGraph(int)`, `withAllGraphs()` | today's phase skipping | Yes |

The single most useful fact in this ledger: **every metric ARCH-16, 17 and 19 ask for is already produced
by TornadoVM and thrown away.** No upstream work is needed — only a sink interface on our side.

### Needs no TornadoVM capability at all — pure host-side Java

- Continuous batching, admission control, scheduling across sequences (ARCH-02, 04, 19).
- Paged KV and prefix sharing (ARCH-01, 08). Grounded by PR #129's paged mode: one large TornadoVM
  array as the block pool, `blockTable[b][pos/blockSize]` indexed **inside** the kernel. No sub-buffer or
  device-allocator API is required, and #129 reports the attention kernel needed **no change** to walk a
  block table. Whatever we decide about lease semantics is implementable today.
- Model provider SPI, format/dtype mapping, logging policy (ARCH-12, 18).

### Missing — would have to be proposed upstream

| Gap | Why it matters | Position |
|---|---|---|
| **Concurrent independent execution plans on one device.** `withIntraPlanConcurrency()` is *intra*-plan; there is no documented support for two independent `TornadoExecutionPlan`s sharing a device concurrently. | Decides whether multi-session concurrency comes from parallel plans or from batching inside one plan. | Design for **batching inside one plan** (ARCH-07). Raise the parallel-plan question upstream only if batching proves insufficient — do not design against a capability that does not exist. |
| **FP4 / MXFP4 / NVFP4 array type and MMA shape.** `types/arrays/` stops at `FP8Array`; `MMAShape` stops at `M16N8K32` (int8, FP8). | ARCH-05 names MXFP4/NVFP4 as the flagship performance goal. | Needs a new native array type + MMA shape + PTX codegen **in TornadoVM**. Propose upstream as a separate piece of work; keep it out of the roadmap's critical path (ARCH-05). |
| **Per-task profiler attribution** at the granularity we would want for a tok/s gate. `TornadoProfilerResult` gives plan-level device kernel time; per-kernel attribution today comes from `nsys`. | ARCH-11's gate would be sharper with per-task times. | Not blocking — plan-level time plus `perf-history.jsonl` is enough for a gate. Worth an upstream feature request, not a dependency. |

---

## ARCH-01 — KV cache ownership prevents paged and shared-prefix KV

**Position: Accept.** Critical, and settle before phase 1 lands the ArchUnit module.

He is right and my Rule 7 is wrong as written. I wrote it against a real problem — cache on the model,
ordering-dependent correctness — and then over-constrained the fix by pushing ownership *down* into the
session instead of *up* into a manager. "KV cache types are reachable only from session state" makes a
shared block pool a rule violation and shared prefix blocks unrepresentable.

**Grounding.** Lease semantics need nothing from TornadoVM: PR #129's paged mode already allocates one
pooled array and indexes `blockTable[b][pos/blockSize]` in-kernel, and reports the attention kernel
required no change to walk shared-then-private blocks. So this is a wording problem in my document, not a
capability problem.

**What changes.**
- Rule 7 → "the KV cache is never global model state; KV storage is owned by a cache manager scoped to
  the engine and leased to sessions." Keep the model prohibition; that part was correct.
- `public-api.md` `GenerationSession` javadoc: "owns KV cache" → "holds a KV lease".
- ADR-001's four lifetimes gain a fifth owner (engine-scoped storage) — cleanest as a new
  **ADR-005: KV cache ownership, block pools and leases**, superseding the relevant part of ADR-001
  rather than editing accepted text.
- Phase 3 scope changes: split lease from storage rather than move the cache wholesale into the session.

---

## ARCH-02 — No engine layer: continuous batching has no home

**Position: Accept.** Critical.

My layering has no tier owning scheduling *across* sequences, so it can only express "one request, one
session, one device" — which is the throughput ceiling the project is actively removing. #129 measures
continuous batching at 1972 gen tok/s and 56.2 req/s with 82.2% slot utilization against 41× aggregate
over single-stream. An architecture that has no place for that is describing a different project.

**Grounding.** The engine tier is pure host-side Java — admission, batch composition, block accounting.
Execution stays exactly one compiled program invoked with B slots, which is what #129 does. No TornadoVM
capability is required, and specifically **no second compiler** — this does not touch ADR-003.

**What changes.**
- Layering diagram: insert **Engine** between "Public API and generation" and "Models and sessions",
  owning `LLMEngine` (`addRequest`, `step`), `Scheduler`, `KvCacheManager`, `BlockPool`, `PrefixCache`.
- `terminology.md`: define Engine, Scheduler, KV cache manager, Block pool, Block table, Lease, Slot,
  Prefix cache, Admission. Add "engine" to the terms-to-use-carefully table — it now collides with the
  existing `InferenceEngine` class, which is a generation loop, not this.
- Dependency rules: engine may depend on models/sessions/programs; sessions must not depend on the
  engine.
- Sessions become handles the engine schedules, not the unit of execution.

---

## ARCH-03 — Rule 8 forbids the on-device sampler that already exists

**Position: Accept.** Settle before phase 1.

Verified on `main`: `LogitsFP16Layer.DEVICE_SAMPLE` runs argmax on device and writes
`state.sampledToken`; `InferenceEngine.sampleTokenGpu` reads it instead of transferring the vocab row;
`LlamaApp.guardDeviceSample` exists to police the preconditions. My Rule 8 enforceable form forbids
`..backend..` → `..generation..`, so the faster path is a violation. He is right about the consequence:
a permanent allowlist entry weakens every other rule, and rules that lose arguments with shipped code
get ignored.

**Grounding.** On-device sampling is an ordinary TornadoVM task in the logits graph — a kernel writing
one `IntArray` element. Nothing about it is a layering violation; my rule conflated *what* sampling is
with *where the generation loop lives*.

**What changes.**
- Split Rule 8 into 8a (generation **policy** — loop, stop conditions, transport, console — stays out of
  the backend; keep the current enforceable form minus sampling) and 8b (sampling is an **operation**
  that may have a backend implementation).
- `program/op` vocabulary in phase 5 gains `Sample` / `ArgMax` as first-class operations.
- Rule 14's list stays as-is: core abstractions still must not *require* a sampler.

---

## ARCH-04 — No phase promotes the batched-decode engine

**Position: Accept.** High.

My "what is deliberately not scheduled" section excluded "new performance work… must not block them",
and #129 fell through that gap. But #129 is not a performance tweak — it is continuous batching, paged
attention and prefix caching, which is the engine tier of ARCH-02 with its most demanding consumer
already written. If the refactor completes without a phase for it, phases 3 and 7 will fix session shape
and execution policy without its requirements in the room, and it gets re-implemented.

**Grounding.** #129 runs today on TornadoVM 5.x using `MMAShape` MMA tasks and `withCUDAGraph()`.
Promotion is a host-side restructuring of `bench/BatchedDecodeEngine` plus the two batch-decode layer
classes; no upstream capability is needed. Its `-Dbatch.decode.*` properties are the same
class of process-global configuration phase 7 replaces with `ExecutionPolicy`, so the two phases should
be designed together.

**What changes.** New **phase 3b — promote the batched-decode engine**, after the session/state split
and before the server/API work, mapping #129's capabilities onto named components: continuous batching →
`Scheduler`; block pool + per-slot block table → `KvCacheManager`; prefix reuse → `PrefixCache`;
on-device sampling → an operation. `bench/BatchedDecodeAttentionBench` and `BatchedProjectionBench`
stay benchmarks.

---

## ARCH-05 — Quantization/dtype seam excluded from scope

**Position: Accept with modification.** High.

Accept the distinction: **new formats** are correctly out of scope, **the dtype seam** is in scope and
load-bearing. My phase 5 said "extract reusable operations" without mentioning dtype, which would
reproduce today's multiplication inside the new vocabulary. Counted: **36 classes** under
`tornadovm/layers/type/**` (18 FP16, 18 Q8_0), and #129 adds two more. His "~32" was close.

**Reject the sub-claim** that the collapse can be "M operation templates × D schemes" all the way down.
It cannot, and the reason is TornadoVM-specific: kernels are compiled per concrete native array type
(`FloatArray`, `HalfFloatArray`, `Int8Array`, `FP8Array`), and Java has no generics over primitives, so
a single `matmul` kernel body cannot serve every scheme. The seam belongs in the **dispatch and
component layer** — one operation *description* parameterized by `DataType`, resolved to a per-dtype
kernel implementation — not inside the kernel bodies. The win is real but smaller than stated: the
per-(model × dtype × mode × MMA) *class* explosion collapses; the per-dtype *kernel* set does not.

**Grounding for the flagship goal.** MXFP4/NVFP4 has no TornadoVM support: `types/arrays/` stops at
`FP8Array`, and `MMAShape` offers `M16N8K16` (fp16/bf16) and `M16N8K32` (int8, FP8 E4M3/E5M2) only. So
FP4 needs a **new upstream capability** — native array type, MMA shape, PTX codegen. That is a TornadoVM
proposal, and it should be raised as one; it must not sit on this roadmap's critical path. Int8 and FP8
MMA, by contrast, are available now, so a Q8_0 tensor-core matmul needs no upstream work.

**What changes.** Phase 5 objective gains "operations are `DataType`-parameterized at the description and
dispatch level"; the invariant test becomes "adding a scheme adds at most k *dispatch* classes, and one
kernel set". Add a note that FP4 depends on upstream TornadoVM work.

---

## ARCH-06 — Phase 1 has no numerical regression net

**Position: Accept with modification.** High. This is the issue I most regret omitting: I wrote
"performance is a correctness criterion" and then specified no correctness check at all, in a repository
with one test file.

**Reject the sub-claim** that goldens can be "bit-identical" as a blanket property. Bit-exactness holds
only within a pinned (device, driver, TornadoVM version, backend, build) tuple — reduction order,
`withCUDAGraph()`, MMA paths and driver changes all move the last bits, and the OpenCL/PTX backends will
not agree with each other. Precedent that config-pinned bit-exactness *is* achievable: #129 verified its
batched output bit-exact against the single-stream greedy reference.

**Modified proposal.**
- **Tier 1 (bit-exact, in phase 1):** golden logits for one small model (Llama-3.2-1B) × {FP16, Q8_0},
  fixed prompt, greedy, pinned backend — asserted bit-identical. Small enough to commit; catches the
  state-motion regressions phases 3/5/6/7 risk.
- **Tier 2 (tolerance):** CPU↔GPU and cross-backend parity with his stated bound
  (`|got − ref| ≤ 1e-2·Σ|wᵢaᵢ| + 1e-3`).
- **Tier 3 (behavioural):** per-layer dumps generated on demand for bisecting, not committed.
- Goldens are regenerated only by an explicit, reviewed commit — never silently refreshed on failure.

---

## ARCH-07 — `GenerationSession` cannot express concurrent serving

**Position: Accept with modification.** High.

Accept: keep `GenerationSession` as the simple single-sequence path, add a non-blocking submission entry
point on the engine tier, and state explicitly that **the server uses the engine API, not the session
API**. My blocking `generate(...)` as the only entry point would have pushed every server into
`InferenceService`'s serialize-behind-a-lock shape.

**Reject the sub-claim** — or rather, correct the mechanism — that concurrency comes from several
in-flight requests "sharing a device" through parallel sessions. **TornadoVM does not currently expose
concurrent independent execution plans on one device.** `withIntraPlanConcurrency()` is intra-plan;
`withConcurrentDevices()` is about multiple devices. So concurrency on one device must come from
**batching inside one compiled program** — which is exactly #129's model, and is why ARCH-02's engine
tier is the answer to ARCH-07 rather than a thread pool over sessions.

This is a genuine design constraint imposed by the runtime, and the documents should say so rather than
leave "can two sessions run concurrently?" as an open question implying it is our choice. If parallel-plan
execution is ever wanted, that is an upstream TornadoVM proposal — not something to design against today.

**What changes.** `public-api.md` gains the engine-tier submit API (handle or reactive stream) and a
sentence stating that device-level concurrency is achieved by batching, not by parallel plans. ADR-001's
concurrency open question resolves to: sessions are independent objects; invocation is batched by the
engine; parallel plans are out of scope pending upstream support.

---

## ARCH-08 — Prefix cache has no owner

**Position: Accept.** Raising severity Medium → **High**, because it is a precondition for ARCH-01's
wording, not an addition to it.

**Grounding.** Already implemented and measured in #129: `-Dbatch.decode.prefixCache=true` (requires
paging), shared prefix blocks pointed at by block-table prefix rows, decode starting at
`pos = sharedPrefixLen`, **no kernel change** because attention walks the block table. Reported:
419 → 211 steps, 1307 → 2422 gen tok/s, 85.7% of prefix KV saved. So this is not a future feature to
leave room for — it exists, and my baseline has nowhere to put it.

**What changes.** `PrefixCache` named in the engine tier, keyed by token-prefix hash over blocks held by
`KvCacheManager`. Document that prefix identity must include model, dtype **and** position offset, plus
eviction and refcounting. Session `reset()` semantics in `public-api.md` must state what happens to a
held lease.

---

## ARCH-09 — Backend SPI has no device-memory allocation seam

**Position: Accept.** Raising severity Medium → **High**: an admission scheduler cannot exist without
capacity reasoning, so this blocks ARCH-02.

**Grounding — this is better supported than the issue claims.** TornadoVM already exposes everything
needed to make lifetime and capacity explicit: `TornadoTargetDevice.getDeviceGlobalMemorySize()`,
`getDeviceMaxAllocationSize()`, `getDeviceLocalMemorySize()` for capacity;
`withMemoryLimit(String)` / `withoutMemoryLimit()` for bounding a plan; the `memory/` package
(`TornadoMemoryProvider`, `XPUBuffer`, `DeviceBufferState`) for buffer state; `DataTransferMode` for
residency across executions; `withCUDAGraph()` which is precisely the case that needs addresses stable
across replays. What is missing is only *our* SPI concept, not a runtime capability.

Note for the design: TornadoVM has no user-facing sub-buffer allocator, so a block pool is one large
array with in-kernel indexing (as #129 does). The seam should therefore describe **pools and lifetimes**,
not per-block device allocations.

**What changes.** Backend SPI gains buffer lifetime classes — model lifetime (weights), engine lifetime
(KV blocks), invocation lifetime (activations) — plus a capacity query. Phases 9 and 10 both reference
it; the engine tier consumes the capacity query for admission.

---

## ARCH-10 — Backend SPI does not express native library dispatch

**Position: Accept, narrowed. Severity Medium → Low.**

Accept the one-sentence addition to ADR-003: an operation implementation may be backed either by a
compiled program or by a native library task, chosen by the backend from device capabilities. It is
cheap to state and it is true.

**Grounding.** The mechanism exists in TornadoVM — `TaskGraph.task(...)` accepts a library binding
factory (`CuBlas::cublasSgemv` style), with `tornado-cublas` and `tornado-cudnn` modules — and #131 ships
it behind `-Dllama.logitsLib={jit,gemmEx,lt}`. So this is not speculative.

**Why narrowed.** #131's own conclusion is parity, not a win: the JIT matvec at 553.8 µs/call and ~948
GB/s (94% of peak) versus cuBLAS gemvx at 550.6 µs — 0.6%, a statistical tie end-to-end, because n=1
projections are bandwidth-bound. Libraries matter in the *batched/MMA* paths (#127, #129), where the
work is compute-bound. So the SPI should admit library dispatch, and the roadmap should not schedule
work for it: the seam is one sentence in ADR-003 plus a note in phase 5's operation vocabulary.

---

## ARCH-11 — Rule 13 has no benchmark gate

**Position: Accept with modification.** Medium.

Accept binding Rule 13 to existing machinery rather than inventing any: `llama-tornado --bench` →
`bench/LlamaBench`, recording to `docs/perf-history.jsonl`, which already holds 1366 entries. Keep the
compiled-program-identity test as the structural half.

**Reject the sub-claim** that "within 3% of the previous recorded run on the same machine and model" is
expressible today. I checked the schema: entries carry `timestamp`, `commit`, `short_commit`, `branch`,
`run_id`, `run_number`, `workflow`, `backend`, `model`, `quantization`, `configuration`, `load_duration`,
`eval_rate`, `prompt_eval_rate`, `total_rate`, counts and durations — and **no machine or runner
identity**. Records come from CI workflow runs across whatever runner was assigned. A 3% gate over
heterogeneous runners is noise, and would either fire constantly or be tuned until it never fires.

**Modified proposal.**
1. Add a `machine` / `runner` identity field to the record (and GPU name — the numbers in #129/#131 are
   RTX 4090 specific).
2. Gate on the tuple (machine, model, quantization, backend, configuration); compare against the last
   green run of that tuple.
3. Tolerance stated per tuple, not globally — a 3% band is plausible on a pinned self-hosted runner and
   not on shared CI.
4. Additionally record `TornadoProfilerResult.getDeviceKernelTime()` via `withProfiler(...)`, which is a
   far less noisy signal than end-to-end tok/s and is free once ARCH-16's seam exists.

---

## ARCH-12 — Model provider SPI lands at phase 8, after the pain

**Position: Accept with modification.** Medium.

Accept the problem: model support is the most frequent change here, #120 (Gemma 4) is in flight, and
each family added before the SPI enlarges the phase-8 migration and pays the P6 switch-editing cost.

**Modification: split phase 8 rather than move it wholesale**, because his premise ("depends mainly on
the model/session split, not on the backend SPI") is only half right.

- **Phase 8a — source detection and loading provider.** Depends only on the phase 4 format/dtype work.
  Replaces `ModelLoader.detectModelType` string matching on `general.name` and `ModelType.loadModel`
  dispatch. Pure host-side, no TornadoVM involvement — genuinely movable early, and it is where the
  Gemma-4-shaped pain actually is. Land it right after phase 4.
- **Phase 8b — program/plan provider.** Removes the `ForwardPlanFactory` family branches. This one
  really does depend on phase 6, because the factory produces plan components and casts `State` to
  family-specific subtypes; without the program layer there is nothing for a provider to return.

**What changes.** Phase 8 splits; 8a moves ahead of phases 5–7 in the dependency summary. Add the
explicit statement he asks for: families landing before 8a pay the switch cost and get migrated by 8a.

---

## ARCH-13 — In-flight PRs are not sequenced against the phases

**Position: Accept.** Medium. My roadmap referenced no open PR, which is a real omission for a plan
whose first principle is "no rewrite".

Verified open: #120 (Gemma 4, CPU + GPU, BF16 and Q8_0), #129 (static batched decode), #131 (hybrid CUDA
library tasks), #138 (FP16 KV cache with packed half2 split-KV attention), plus #140 itself.

**Proposed land order, with reasons rather than preferences.**
1. **#129 first.** It is the source for the engine tier (ARCH-02/04) and the only demonstrated consumer
   of paged/prefix KV. Refactoring session and cache ownership without it in the tree means deciding
   ARCH-01 twice.
2. **#138 next.** It changes KV cache *layout*; phase 3 changes KV cache *ownership*. Layout before
   ownership is one conflict; the reverse is two.
3. **#120** any time before phase 8a — it is the motivating case for the provider SPI, and it is cheaper
   to migrate one more family than to design 8a without a live example.
4. **#131** last and independent — it is additive, default-off, and measured as parity.

**Freeze declaration:** `inference/state/**`, `tornadovm/plan/**` and `tornadovm/layers/type/**` are
"in refactor" once phase 3 opens; feature work in those trees rebases rather than merges.

---

## ARCH-14 — Multi-device execution absent from the target architecture

**Position: Accept — seams only, no implementation. Severity Low → Medium.**

I am upgrading this because the framing "record the seam now, it is cheap" undersells what I found:
**TornadoVM already has the multi-device API.** `withDevice(TornadoDevice)`, per-task
`withDevice(String taskName, TornadoDevice)`, `withConcurrentDevices()` /
`withoutConcurrentDevices()`, and `TornadoDeviceMap` enumerating devices across backends. Per-task
placement in particular means a program's tasks can already be spread across devices without any
upstream work.

So this is not a hypothetical future capability constraining a design — it is an available capability my
target architecture ignores by describing `DeviceSelector` as selecting *a* device. That is the wrong
default shape for the SPI: an invocation should be able to target a device *set*, and the KV manager
should be able to hold blocks per device.

**What changes.** `target-architecture.md` records a shard-plan seam (how a program's weights and work
map to devices) and states that invocation targets a device set. Marked **design-only** — no phase, no
implementation. `DeviceSelector` keeps a single-device convenience form.

---

## ARCH-15 — Public API does not expose quantization of a loaded model

**Position: Accept.** Low, uncontroversial, and he is right that it is the first thing every consumer
would otherwise obtain by reaching into the format layer — the exact leak ADR-004 prohibits.

**What changes.** `ModelInfo` exposes **both** weight dtype and compute dtype, as runtime `DataType`
(never `GGMLType`). Two dtypes rather than one because they already differ today:
`AbstractModelLoader.effectiveGpuWeightType` collapses `Q4_K`/`Q5_K`/`Q6_K` to `Q8_0`, and
`getModelQuantization` maps GGUF file types 14–18 to `"Q8_0"`, so a "Q6_K model" executes as Q8_0. A
single field would be a lie for exactly the models where the user most needs the truth.

---

## ARCH-16 — No metrics seam, device metrics cannot reach their consumer

**Position: Accept.** High. The best issue in the set, and the one that found a real hole in my
reasoning rather than a gap in coverage.

His structural point is correct and I missed it: metrics have the **opposite dependency direction** to
everything else in the design — produced at the bottom (backend, device), consumed at the top (API,
engine, operator). My Rule 8 forbids `..backend..` → `..api..`, so a metrics facility placed in the API
layer is unreachable from where the data originates. Two outcomes, both bad: the backend acquires the
forbidden upward dependency, or device timings are never available.

**Grounding — and the evidence that this is the default failure.** TornadoVM already produces all of it:
`withProfiler(ProfilerMode)` plus `TornadoExecutionResult.getProfilerResult()` yields
`getDeviceKernelTime()`, `getDeviceWriteTime()`, `getDeviceReadTime()`, `getDataTransfersTime()`,
`getTotalBytesCopyIn()` / `getTotalBytesCopyOut()`, `getTotalDeviceMemoryUsage()`, `getCompileTime()`,
`getTornadoCompilerTime()`, `getKernelDispatchTime()`, plus `profiler/ChromeEventTracer` for timelines.
GPULlama references **none of it** — zero hits for `ProfilerMode`, `getProfilerResult`, `TornadoProfiler`
across all main sources. So the capability has been sitting there unused, which is precisely his argument
that without an explicit seam this data never surfaces.

**What changes.** A metrics sink interface in the **runtime** layer, below the backend SPI's consumers:
backend writes, API/engine read. Sink implementations (in-memory, bench recorder, exporter) live above;
the interface does not. `auxiliary/RunMetrics` (a static holder that prints) becomes one implementation.
Add a dependency rule permitting `backend → runtime.metrics` explicitly, so it is a designed exception
rather than an allowlist entry.

**One caveat to design in:** `withProfiler(...)` is not free. The sink must be off by default on the
decode path, with profiling a policy the caller opts into — otherwise we pay for telemetry per token.

---

## ARCH-17 — Observability scheduled last, but earlier phases need it now

**Position: Accept.** Medium. Consistent with accepting ARCH-06 and ARCH-11: every phase claiming "no
behaviour change" needs evidence, and phase 10 is too late to produce it for phases 3–7.

**What changes.** Split my phase 10. Early (alongside the phase 1 guardrails): the ARCH-16 seam plus load,
prefill, decode and tokens/s counters. Phase 10 keeps memory planning, error-message work, exporters and
the Javadoc/experimental-marker removal. Cheap to do early because the seam is an interface and the data
already exists.

---

## ARCH-18 — No logging policy for an embeddable library

**Position: Accept with modification.** Medium.

Verified: **65 `System.out` / `System.err` occurrences across 20 main-source files**, including the
generation loop. He is right that this is the same class of problem as my own P2 (`Model` owns the
application loop) and that a library printing to stdout is unusable in a server — it corrupts structured
logs and cannot be silenced or routed.

**Modification: no external logging facade dependency.** He suggests "a facade (or a no-op-by-default
interface it owns)" — take the second option and commit to it. `pom.xml` declares no logging dependency
today, and the project ships a shaded jar with native-image considerations; adding SLF4J to a library
whose selling point is a self-contained JVM inference stack imports a dependency-hell surface for little
gain. A tiny project-owned sink, no-op by default, with an optional SLF4J bridge in an integration
module, gets the same result. It also composes with the ARCH-16 metrics sink — same shape, same
direction, plausibly the same lifecycle.

**What changes.** Logging policy stated in `public-api.md`; new **Rule 16** — no `System.out`/`System.err`
outside the CLI integration — with the current 20 files as the enumerated allowlist under the existing
shrink-only policy. The CLI keeps printing; that is its job.

---

## ARCH-19 — Serving-level metrics are undefined

**Position: Accept.** Medium. Follows from accepting ARCH-02 and ARCH-16: once an engine tier exists,
"where does time go" for a single sequence stops being the interesting question.

**Grounding.** #129 already reports slot utilization (82.2%), steps, gen tok/s and req/s per scheduling
mode, and its paged mode surfaces a `KV block pool exhausted` condition — i.e. these quantities are
already being measured ad hoc to evaluate the feature. Formalizing them is recording what the work
already needs, and his point that queue-wait accounting must be designed with the scheduler rather than
threaded through afterwards is correct.

**What changes.** Engine tier defines: time to first token, queue wait, batch occupancy / slot
utilization, KV block utilization, preemptions, admitted versus rejected. Emitted through the ARCH-16
sink. The per-request subset (TTFT, queue wait, tokens generated) surfaces on the result type;
`GenerationResult.timings()` in `public-api.md` extends to carry it.

---

## Consequences for the baseline if all positions are agreed

Document edits:

| Document | Change |
|---|---|
| `target-architecture.md` | Engine tier inserted; shard-plan seam recorded (design-only); metrics sink placed in runtime layer |
| `dependency-rules.md` | Rule 7 reworded (lease); Rule 8 split into 8a/8b; new Rule 16 (no console I/O); explicit `backend → runtime.metrics` permission; engine-tier direction rules |
| `public-api.md` | `GenerationSession` javadoc (lease, not ownership); engine submit API; `ModelInfo` gains weight + compute dtype; logging policy; batching-not-parallel-plans statement |
| `terminology.md` | Engine, Scheduler, KV cache manager, Block pool, Block table, Lease, Slot, Prefix cache, Admission, TTFT; "engine" added to terms-to-use-carefully |
| `migration-roadmap.md` | Goldens + metrics seam into phase 1; new phase 3b (engine promotion); phase 5 dtype-parameterized; phase 8 split into 8a/8b; benchmark gate bound to `perf-history.jsonl` with a machine field; PR land order and freeze declaration |
| `current-architecture.md` | New pressure points: P12 no metrics/telemetry despite available profiler; P13 console I/O in library code |
| ADRs | New **ADR-005** (KV ownership, block pools, leases); ADR-001 concurrency question resolved to batching; ADR-003 gains the library-task sentence; ADR-002 gains sampling-as-operation |
| New | Upstream TornadoVM proposals tracked separately: FP4/MXFP4 array type + MMA shape; concurrent independent execution plans (only if batching proves insufficient) |

Sequencing note: ARCH-01, 03 and 16 change rules that phase 1 would otherwise encode, so they should be
settled before the ArchUnit module lands. ARCH-02, 04, 08 and 09 are one connected decision — the engine
tier — and are best taken together rather than issue by issue.
