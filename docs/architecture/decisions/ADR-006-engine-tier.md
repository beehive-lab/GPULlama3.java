# ADR-006: The engine tier

## Status

**Accepted** — 2026-07-30, following the ARCH-02, ARCH-04, ARCH-07 and ARCH-19 review on
PR #140.

## Context

The original layering ran Integrations → Public API and generation → Models and sessions
→ Programs and operations → Runtime → Backend SPI → TornadoVM. No tier owned *scheduling
across sequences*: the API layer was described as sampling, stop conditions and
streaming; the model/session layer is per-sequence by definition.

The consequence is that the architecture could only express **one request occupies one
session occupies the device**. That is the throughput ceiling the project is actively
removing, and the code that removes it already exists: `bench/BatchedDecodeEngine`
(PR #129) implements continuous batching, paged KV and prefix caching, measuring
1972 gen tok/s and 56.2 req/s at 82.2% slot utilization, against 41× aggregate over
single-stream decode.

That code currently lives in a benchmark `main()`, driven by `-Dbatch.decode.*` system
properties and hard-cast to LLaMA/Qwen3 + FP16/CUDA. Left there, the refactor would
settle session shape and execution policy without its requirements in the room, and it
would be re-implemented later against a design that did not anticipate it.

The related question — how do several requests share a device — turned out to have a
different answer than first assumed. See "Concurrency" below.

## Decision

**Insert an engine tier between the public API and models/sessions.**

```
Public API and generation
        ↓
      Engine          ← LLMEngine, Scheduler, KvCacheManager,
        ↓               BlockPool, PrefixCache, serving metrics
Models and sessions
```

1. **`LLMEngine`** — `addRequest(...)` (non-blocking admission), `step()` (one batched
   iteration across admitted sequences), `metrics()`.
2. **`Scheduler`** — admission, batch composition, preemption.
3. **`KvCacheManager` / `BlockPool` / `PrefixCache`** — per
   [ADR-005](ADR-005-kv-cache-ownership-and-leases.md).
4. **Sessions become handles the engine schedules**, not the unit of execution. A session
   remains usable standalone for the simple single-sequence path.
5. **The server uses the engine API**, not the session API.
6. **Serving metrics are defined here** — TTFT, queue wait, batch occupancy, KV block
   utilization, preemptions, admitted versus rejected — emitted through the metrics sink
   ([Rule 17](../dependency-rules.md#rule-17--metrics-flow-bottom-to-top-by-design)).

This tier is **pure host-side Java**. Execution remains exactly one compiled program
invoked with B slots. It requires no TornadoVM capability and introduces **no second
compiler** — [ADR-003](ADR-003-tornado-backend-boundary.md) is untouched.

## Concurrency: batching, and why

Device-level concurrency comes from **batching many sequences into one invocation of one
compiled program**, not from running one plan per session.

The reason is not that parallel plans are unsupported. They are supported and tested:
`tornado-unittests/.../multithreaded/TestMultiThreadedExecutionPlans` constructs two
`TornadoExecutionPlan` instances in two Java threads over the same immutable graph and
runs them on the default device — 4 tests, 0 failures on CUDA/RTX 4090 against
5.2.1-jdk21-dev.

The reason is **memory**. Device buffers are per task graph: each `TaskGraph` owns its own
device buffer state, so the same Java object referenced by two graphs gets two device
buffers. (This is why TornadoVM #996 had to make cross-graph aliasing explicit for
`consumeFromDevice`.) Two sessions built as two graphs therefore hold two device copies
of the weights — roughly **3.4 GB duplicated per concurrent session on a 3B-Q8 model**.
Weight duplication becomes the binding constraint long before scheduling does.

So batching is an economic design choice, not a workaround for a missing API, and nothing
here is pending upstream. Recorded as
[capability C2](../tornadovm-capabilities.md#c2--device-buffers-are-per-task-graph).

## Consequences

Positive:

- Continuous batching, paged KV and prefix reuse have a home, and #129 is promotable
  rather than trapped.
- The OpenAI server stops serializing on one `State` behind a lock.
- Admission control gets an owner, which makes device capacity a first-class concern
  rather than an out-of-memory surprise.
- Serving metrics are designed with the scheduler instead of retrofitted — queue-wait
  accounting cannot be threaded through code that was not built to carry timestamps.

Negative / costs:

- A new tier is new surface: scheduling policy, preemption, fairness, starvation are all
  now the project's problem.
- Batched steps generate substantially more interpreter bytecode than single-stream
  execution — B slots × N layers per step. The bytecode buffer was a fixed 4096 bytes
  until TornadoVM #1004, and overflow was swallowed by `recover.bailout`, producing
  truncated bytecode and silently wrong results. This is one reason the version floor is
  a prerequisite
  ([capability C4](../tornadovm-capabilities.md#c4--interpreter-bytecode-buffer-overflow-was-silent)).
- Batched decode changes latency characteristics: better aggregate throughput, worse
  single-request latency under load. The API must not pretend otherwise.
- `-Dbatch.decode.*` is the same class of process-global configuration that
  `ExecutionPolicy` replaces, so this tier and the execution-policy work must be designed
  together or those switches get migrated twice.

## Alternatives considered

**Keep scheduling in the server.** `InferenceService` already serializes requests, so a
better scheduler could live there. Rejected: the CLI, benchmarks and embedded users all
want batching too, and a scheduler above the API cannot see KV capacity.

**Put batching in the session.** Rejected: a session is one sequence by definition;
batching across sessions cannot be a member of one of them.

**One plan per session, relying on concurrent execution plans.** The obvious design once
you learn concurrent plans work. Rejected on weight duplication — see Concurrency above.

**Leave #129 as a benchmark and design the tier from first principles.** Rejected: the
benchmark is the requirements document. Designing without it produces a tier its most
demanding consumer then has to fight.

## Migration notes

Depends on the session/state split and on [ADR-005](ADR-005-kv-cache-ownership-and-leases.md).
PR #129 should **land before** this tier is built, so the design has its consumer in
tree rather than in a diff.

1. `KvCacheManager` + `BlockPool`, single-lease, behaviour-identical (ADR-005).
2. Promote #129's paged mode onto the manager; verify CUDA-graph replay survives.
3. `Scheduler` + admission; reproduce #129's continuous-batching numbers.
4. `PrefixCache` with refcounting and pinning; reproduce the prefix-reuse savings.
5. `LLMEngine` with non-blocking submit; move the server onto it.
6. Serving metrics through the sink.
7. Retire `-Dbatch.decode.*` in favour of explicit policy, together with the
   execution-policy work.

Each step keeps the goldens bit-identical and the benchmark gate green.

## Open questions

1. Scheduling policy: FCFS, or priority/fairness? What preemption is allowed mid-sequence?
2. Does `step()` block, or is there a background thread? Who owns the thread if so?
3. How does the engine decide batch size — fixed B, or adaptive to free blocks?
4. One engine per model, or one engine over several models on one device?
5. Does the simple `GenerationSession` path run through a hidden single-slot engine, or
   bypass the engine entirely? (Bypassing is simpler; routing through gives one code path.)
6. Ragged batching: #129 notes same-length prompts for the static case. What is the
   supported shape for mixed-length sequences in a step?
