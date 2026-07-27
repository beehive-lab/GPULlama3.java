# ADR-001: Model and session separation

## Status

**Proposed.** Not accepted. No record of maintainer approval exists in this repository.

## Context

Today a loaded model is not a value. `model.Model` exposes
`TornadoVMMasterPlan tornadoVMPlan()` and `void setTornadoVMPlan(TornadoVMMasterPlan)`,
and `model.AbstractModel` holds a mutable `plan` field along with three other non-final
fields. The plan owns device memory and is created by
`TornadoVMMasterPlan.initializeTornadoVMPlan(state, model)`, which calls
`model.setTornadoVMPlan(plan)` as a side effect.

Mutable per-sequence data lives in `inference.state.State`, created by
`Model.createNewState()`. One `State` holds, together:

- CPU activation buffers (`x`, `xb`, `hb`, `q`, `k`, `v`, `att`, `logits`);
- the KV cache (`keyCache`, `valueCache`, and device mirrors `wrapKeyCache`,
  `wrapValueCache`);
- device mirrors of the activations (`wrapX`, `wrapQ`, …);
- batch-prefill scratch (`wrapXBatch`, `qkvResultBatch`, `gateUpResultBatch`, …);
- per-invocation temporaries (`temp`, `tempFFN`, `tempLogits`);
- `latestToken`, `positionHolder`, `sampledToken`, `localSize`.

Nearly all fields are `public`.

The consequences show up in the code that has to run more than one request:
`server.InferenceService` holds exactly one `Model`, one `State` and one plan, and
serializes every request behind a lock, resetting the KV cache from position 0 for each
request. `Model.runInteractive` creates a `State` and a plan inline and frees the plan
in a `finally` block. There is no type meaning "one conversation".

Two problems follow. First, the model cannot be shared: a second concurrent sequence
would have to share the single `plan` field and the single `State`. Second, the
lifetimes are tangled — a buffer used for one call, a cache that lives for the whole
sequence, and weights that live for the whole process are all reachable from the same
two objects.

## Decision

Split ownership into four lifetimes, as defined in
[`terminology.md`](../terminology.md):

```
Loaded model    immutable configuration, architecture identity, weights
                → shared, thread-safe, lives as long as the process needs it

Compiled program  backend-specific reusable executable resources
                → built once per (architecture, policy, backend, device),
                  shareable between sessions of the same model

Session         mutable sequence state: KV cache, position, sequence bookkeeping
                → one per conversation, NOT thread-safe

Invocation      input, output and temporary bindings for one forward call
                → no lifetime beyond the call; never stored on the model
```

Concretely:

1. **The loaded model is immutable.** All fields final. No `setTornadoVMPlan`. No
   compiled program reachable from the model as mutable state. (A model-owned
   *compiled-program cache* is acceptable, because a cache keyed by
   policy/backend/device is not per-sequence state — but it must be internally
   synchronized and must not be a single mutable slot.)
2. **The session owns the KV cache and position.** The KV cache is never model state
   ([rule 7](../dependency-rules.md#rule-7--the-kv-cache-is-never-global-model-state)).
3. **Multiple sessions may share one loaded model** and one compiled program.
4. **Invocation-scoped buffers are not stored globally.** Activation and scratch
   buffers belong to whatever the session or backend allocates for its own use, not to
   the model.
5. **Model-specific state exists only where genuinely required.** `Qwen3State`
   carries `wrapAttSplit` because the split-KV attention kernel needs a
   precisely-sized scratch buffer; that is a real requirement. A family-specific state
   subclass that exists only to hold a differently-named copy of a common buffer is
   not.

## Consequences

Positive:

- A loaded model becomes safe to share across threads by construction.
- Concurrent sequences against one model become expressible.
- Weights are loaded once and reused; the expensive part of startup is amortized.
- `InferenceService` becomes a thin wrapper over sessions rather than the only place
  that knows how to reuse a model.
- Buffer lifetimes become explicit, which is a prerequisite for the memory planning in
  [roadmap phase 10](../migration-roadmap.md#phase-10--memory-planning-diagnostics-and-developer-experience).
- `Model` stops depending on `TornadoVMMasterPlan`, clearing the `model/` entries from
  the [rule 2](../dependency-rules.md#rule-2--model-architecture-packages-do-not-import-tornadovm)
  allowlist.

Negative / costs:

- `Model` is public surface used by external integrations (LangChain4j, Quarkus).
  Removing methods from it is a breaking change requiring a deprecation cycle.
- Splitting `State` touches every task-graph builder in `tornadovm/layers/**`, because
  they read `state.wrapX`, `state.wrapKeyCache` and friends directly by field.
- More objects and more explicit wiring than today's "one `State`, one plan".
- Device memory per session goes up if each session gets its own KV cache — which is
  correct, but it means N sessions cost N KV caches. Memory planning becomes necessary
  rather than optional.

## Concurrency

This is the part that is **not** settled, and it should not be pretended otherwise.

What the decision guarantees: a loaded model is safe to read from many threads; a
session is owned by one thread at a time.

What it does not settle: whether two sessions may **invoke concurrently** on one
device. Three options:

1. **Serialized invocation** (today's behaviour, via `InferenceService`'s lock).
   Simple, correct, no device work needed. Sessions are concurrent objects but
   invocations queue. Throughput does not improve with more sessions.
2. **Per-session device buffers and queues.** Each session gets its own activation and
   KV buffers on device; invocations run on separate TornadoVM execution plans or
   streams. Real concurrency, significantly more device memory, and it requires
   understanding how TornadoVM shares or isolates device state between execution plans
   — which needs investigation, not assumption.
3. **Batched multi-sequence decode.** One invocation processes one token for each of N
   sessions. Best device utilization, but it requires per-sequence KV cache addressing
   in the attention kernels, which does not exist today.

**Recommendation for the first implementation: option 1.** It matches current
behaviour exactly, it is the smallest change, and it does not close off options 2 or 3.
Sessions being independent objects is the prerequisite for all three; how invocations
are scheduled can be decided later.

## Resource lifecycle

```
LoadedModel.close()
    → releases weights (host + device) and any cached compiled programs
    → sessions created from it must already be closed;
      closing a model with live sessions is an error, not a silent free

GenerationSession.close()
    → releases the session's KV cache and buffers
    → must NOT invalidate the loaded model or other sessions
    → must NOT free a shared compiled program

CompiledProgram.close()
    → releases the backend's device resources (for the Tornado backend, the
      TornadoExecutionPlan and its buffers)
    → owned by whoever created it: the model's cache, or the caller who compiled it
```

Both `LocalModel` and `GenerationSession` are `AutoCloseable`, because both hold
device memory. Today's cleanup — `plan.freeTornadoExecutionPlan()` in a `finally`
block inside `Model.runInstructOnce` — becomes the session's and the model's
`close()`.

Open point: whether closing a model should force-close its sessions or throw. Throwing
is more honest; force-closing is friendlier in a try-with-resources world where
ordering mistakes are easy. Leaning towards throwing with a clear message.

## Alternatives considered

**Keep `State` as-is and add a session wrapper around it.** Cheapest option; matches
`InferenceService` today. Rejected as an end state because it does not separate
sequence-lifetime from invocation-lifetime data — the wrapper would still hand out one
object containing both. Acceptable as an *intermediate* step in
[roadmap phase 2](../migration-roadmap.md#phase-2--public-api-façade-over-current-implementation).

**Put the KV cache on the model, keyed by session id.** Would avoid changing `State`.
Rejected: it makes the model mutable and shared-mutable, which is the problem being
solved, and it makes cache eviction a model concern.

**One session per model instance (load the model again per conversation).** Simple and
correct. Rejected: weights are the largest allocation and the slowest part of startup;
duplicating them per conversation is not viable for server use, which is a primary use
case.

**Immutable sessions with functional state updates.** Clean, and impossible here — the
KV cache is a large device buffer written in place by kernels.

## Migration notes

Corresponds to [roadmap phase 3](../migration-roadmap.md#phase-3--loaded-model-and-session-state-separation),
after the phase 2 façade exists.

Suggested order:

1. Make `AbstractModel.tokenizer`, `weights` and `chatFormat` final. Trivial; do it in
   milestone 1.
2. Introduce the session type in `api/`, wrapping today's `State` + plan (phase 2).
   Behaviour identical to `InferenceService`.
3. Move plan ownership from `Model` to the session; deprecate `tornadoVMPlan()` /
   `setTornadoVMPlan(...)` rather than deleting them.
4. Split `State` into session state and invocation buffers. This is the large,
   risky step — it touches every builder in `tornadovm/layers/**`. Benchmark FP16 and
   Q8_0, single-token and prefill/decode, before and after.
5. Reimplement `InferenceService` on the session type.
6. Remove the deprecated `Model` methods, after the integration repositories have
   migrated.

Deterministic output for a fixed seed must be unchanged at every step.

## Open questions

1. Serialized invocation or true concurrency for the first implementation?
   (Recommendation: serialized.)
2. Does the loaded model own a compiled-program cache, or does the caller?
3. Does closing a model with live sessions throw, or force-close them?
4. How much of the session's state is backend-owned (device buffers allocated by the
   backend) versus runtime-described (a layout the backend materializes)?
5. Does a session support `reset()` (reuse buffers, discard sequence) as well as
   `close()`? `InferenceService` effectively does this today by rewriting the KV cache
   from position 0.
6. Which family-specific state fields survive the split, and which are common buffers
   in disguise?
