# Resource Ownership and Lifecycle

**Status: proposed-normative.** The matrix below is the single place where "who owns
what, and what happens on close" is stated. It composes ADR-001 (four lifetimes),
ADR-005 (KV leases) and ADR-006 (engine tier); rows marked **[D-nn]** depend on an open
[decision gate](decision-gates.md) and record the current recommendation, not a
decision. Once the gates close, this document is normative and API Javadoc must agree
with it.

Terms: [`terminology.md`](terminology.md).

## Ownership matrix

"Owner" = the object whose `close()` releases the resource. "Borrowers" = objects that
hold a reference but must not release it. "Parent closes with live borrowers" = the
defined behaviour when the owner is closed while borrowers still exist.

| Resource | Owner | Borrowers | Created at | Released at | Thread-safety | Permitted sharing | Parent closes with live borrowers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model weights (host + device) | `LocalModel` | compiled programs (read-only), sessions, engine | `LocalModels.load(...)` | `LocalModel.close()` | immutable after load; safe to read from any thread | any number of sessions/programs of the same model | see [Model close](#model-close--d-11) [D-11] |
| Compiled program | its cache entry — owned by the `LocalModel`'s program cache; or the caller, when compiled explicitly via `Backend.compile(...)` | sessions and the engine that invoke it | first `newSession()` needing the (architecture, config shape, policy, backend, device) key; or explicit `compile(...)` | model close (cached) or caller's `close()` (explicit) | compilation thread-safe via the cache; `invoke(...)` **not** thread-safe [D-12] | shared by all sessions of one model on one device | model close releases cached programs; sessions must already be closed, so no live borrower exists |
| Compiled-program cache | `LocalModel` | none (internal) | model load (lazily populated) | model close | internally synchronized; never a single mutable slot (ADR-001) | n/a — internal in v1 [D-05] | n/a |
| Backend handle | the process/composition root that selected it (`Backends.select(...)` caller); a model loaded with defaults owns the backend it auto-selected | models, compiled programs, engine | `Backends.select(...)` or implicit at `load(...)` | caller's `close()`; implicit backend closes with the model that created it | thread-safe | several models may share one explicit backend | closing a shared backend with live models is an error (throw); implicit backends cannot reach this state |
| Device handle | the backend | everything below the backend, descriptively above it | backend init | backend close | thread-safe (descriptive reads) | freely shared as description; never exposes a TornadoVM device | follows the backend |
| `KvCacheManager` + `BlockPool` | the engine, when an engine exists; the model-scoped single-lease manager in the engineless path [D-10] | sessions (via leases), scheduler, prefix cache | engine construction; or first engineless `newSession()` | engine close; or model close in the engineless path | manager thread-safe (admission and release race by design); pool array itself written only by kernels during a step | one pool per (engine, device) | manager close with live leases is an error (throw) — a lease is a session, and sessions must close first |
| KV lease (block table) | the session holding it | the attention kernels during an invocation | session creation (or first generate, if leases are lazy) | `session.close()` or `session.reset()` (private blocks only) | not thread-safe — owned by the session's thread | the *blocks* it references may be shared (prefix blocks, refcounted); the lease itself is never shared | n/a — lease dies with its session |
| Shared prefix blocks | `BlockPool` storage, accounting by `PrefixCache` | any lease whose block table references them | first prefill of the prefix | refcount reaches zero **and** not pinned | refcounting internally synchronized | referenced by any number of leases of the same (model, dtype, position offset) | eviction under a live lease is forbidden — leased blocks are pinned (C1) |
| Single-session invocation buffers | the session | the compiled program during `invoke(...)` | session creation, sized by context length | session close | not thread-safe | never shared between sessions (ADR-001 point 4) | n/a |
| Engine batched invocation buffers (B-slot inputs/outputs, batched scratch) | the engine | the compiled program during `step()`; sessions **read** their slot's results, never hold the buffer | engine construction (sized by max B) | engine close | written only inside `step()`; results published to handles after the step | one set per engine; slots are positions in these buffers, not per-session allocations | engine close with live requests: see [Engine shutdown](engine-contract.md#engine-shutdown) |
| Session | the user (or the engine, for engine-created internal sessions) | the engine schedules it; it never owns it when user-created | `model.newSession(...)` | `session.close()` | not thread-safe; one thread at a time | many sessions per model | model close: [D-11] |
| Request handle | the caller of `LLMEngine.addRequest(...)` | the engine (delivers into it) | `addRequest(...)` | `handle.close()` or terminal state reached | thread-safe reads (poll/await); delivery from the step thread | not shared | handle close while running ⇒ cancellation, see [engine-contract.md](engine-contract.md#cancellation-and-handle-close) |
| Engine | the user (server, CLI, embedding application) | none | engine construction over a model | `engine.close()` | `addRequest` thread-safe; `step()` single-caller [D-15] | one engine per (model, device) in v1 [D-19] | n/a — top of its subtree |
| Metrics sink | the composition root that installs it (defaults to no-op) | backends write; engine/API read (Rule 17) | process/library init | never device-owning; GC'd | must be thread-safe — written from execution paths | one sink per process is typical; per-engine sinks permitted | writers hold a reference; a replaced sink simply stops receiving |
| Logging sink | same as metrics sink (Rule 16) | all library code writes | process/library init | GC'd | must be thread-safe | as metrics sink | as metrics sink |

## CUDA-graph buffer invariants

Restated here because every row above that touches device memory is constrained by them
(capability [C1](tornadovm-capabilities.md#c1--cuda-graph-capture-fixes-device-addresses)):

1. **The block pool is one persistent device array with in-kernel indexing.** A manager
   that hands a slot a different device buffer per step breaks CUDA-graph replay, and
   `recover.bailout=true` turns the break into wrong output, not an error.
2. **Leased blocks are pinned against eviction.** Eviction that re-points a leased block
   fails at replay, not at eviction time.
3. **Engine batched invocation buffers are allocated once and never reallocated while a
   captured graph exists.** Growing B means recapture, which is a policy event, not an
   allocation.
4. **Session invocation buffers bound into a captured graph must keep their identity**
   for the life of that compiled program's capture. `Invocation` binds existing buffers;
   it allocates nothing (ADR-002, Rule 13).

## Model close — [D-11]

Two candidate semantics, decision open (ADR-001 open question 3):

- **Throw** (`IllegalStateException` naming the live sessions). Honest; surfaces
  ordering bugs. ADR-001 leans this way, and try-with-resources nesting (model outer,
  session inner) makes correct ordering the natural spelling.
- **Force-close** the sessions. Friendlier, but silently invalidates other threads'
  sessions mid-generate.

Whichever is chosen: closing a model **must not** be a silent free while kernels can
still reach the weights, and the choice is user-visible API contract, so it gates M3.1.

## The engineless session path — [D-10]

`GenerationSession` must work without an engine (Rule 18 rationale). But ADR-005 puts KV
storage in an engine-scoped manager. The recommendation on the table:

- **M6 (before any engine exists):** the model owns a **single-lease `KvCacheManager`**
  — the same manager type, engine-agnostic, one pool, one lease at a time,
  behaviour-identical to today's per-`State` cache. Sessions lease from it. Nothing in
  `model` depends on `engine` packages (Rule 18 holds).
- **M7 onward:** an engine constructed over the model brings its own manager; sessions
  created *through the engine* lease from the engine's manager. The simple path keeps
  the model-scoped manager. The alternative — routing every session through a hidden
  single-slot engine — gives one code path at the cost of making the engine a dependency
  of the simple path; ADR-006 question 5 records both.

This is a blocking decision for M6.1 and is tracked as [D-10](decision-gates.md).

## Invocation concurrency — [D-12]

A compiled program is shared by sessions, but `invoke(...)` binds mutable state.
Recommendation: `invoke(...)` is **not** thread-safe; two sessions sharing a program
serialize their invocations (matching ADR-001's "recommendation: serialized" for v1),
and the engine's `step()` is the single invoker in batched mode. True concurrent
invocation of one program is not planned — device concurrency comes from batching
(ADR-006), so there is no consumer for it.
