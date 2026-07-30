# ADR-005: KV cache ownership, block pools and leases

## Status

**Accepted** — 2026-07-30, following the ARCH-01 and ARCH-08 review on PR #140.

Refines [ADR-001](ADR-001-model-session-separation.md), which placed the KV cache in the
session. That placement was too strong; this ADR replaces it.

## Context

ADR-001 established four lifetimes and put the KV cache under session ownership, and
[Rule 7](../dependency-rules.md#rule-7--the-kv-cache-is-never-global-model-state-storage-is-managed-and-leased)
originally read "KV cache types are reachable only from session state".

That was written against a real problem — a cache attached to the model prevents
concurrent sequences and makes correctness depend on call ordering — but the fix
over-corrected. Under "one sequence owns its cache":

- a shared block pool is a rule violation;
- prefix sharing is unrepresentable, because two sessions cannot reference the same
  cache memory;
- paged attention has nowhere to live, since paging is defined by blocks outliving and
  moving between sequences.

Meanwhile PR #129 already implements paged KV and prefix caching and measures them: a
shared pool of fixed-size blocks with a per-slot block table, prefix rows pointed at
shared blocks, decode starting at `pos = sharedPrefixLen`. Reported effect of prefix
reuse on a 512-request continuous-batching workload: 419 → 211 steps, 1307 → 2422 gen
tok/s, 85.7% of prefix KV saved. The attention kernel needed **no change** to walk a
block table.

So the architecture forbade something the code already does, and does well.

## Decision

**KV storage is owned by an engine-scoped cache manager and leased to sessions.**

1. **Never on the model.** The original prohibition stands unchanged. A loaded model is
   immutable and shared; it holds no sequence state.
2. **`KvCacheManager` owns storage.** Engine-scoped. Allocates the pool, hands out
   leases, reclaims blocks, enforces capacity.
3. **A session holds a lease**, not the storage — a block table referencing blocks it
   does not own. Releasing a lease returns private blocks; shared blocks are refcounted.
4. **The pool is one persistent device array with in-kernel indexing.** Not a set of
   separately allocated buffers. See the invariant below.
5. **Blocks under a live lease are pinned** against eviction.
6. **Prefix identity includes model, dtype and position offset.** Two prompts with the
   same tokens under different models or dtypes are different prefixes.

## The CUDA-graph invariant

This is the part that must not be optimized away later by someone who does not know why
it is written down.

`withCUDAGraph()` bakes device addresses into the captured graph. Re-pointing a captured
buffer between replays fails at replay with `CUresult=700`, and because
`tornado.recover.bailout` defaults to `TRUE` (`TornadoOptions.RECOVER_BAILOUT`), the
first symptom is **wrong output rather than an error** (TornadoVM #1006).

Therefore:

> A `KvCacheManager` that hands a slot a *different device buffer* per step cannot
> coexist with CUDA-graph capture. One persistent pooled array with in-kernel
> `blockTable` indexing is not one valid implementation among several — it is the only
> shape that keeps graph capture.

The obvious-looking optimization — allocate blocks on demand, hand each slot its own
buffer — silently breaks replay. Likewise, eviction that re-points a leased block fails
at replay rather than at eviction, which is why leased blocks are pinned.

Recorded as [capability C1](../tornadovm-capabilities.md#c1--cuda-graph-capture-fixes-device-addresses).

## Consequences

Positive:

- Paged attention and prefix sharing become expressible, and #129's implementation is
  promotable rather than a rule violation.
- Block accounting has one owner, which is what admission control needs
  ([ADR-006](ADR-006-engine-tier.md)).
- Sessions get cheaper: a session is a lease plus invocation buffers, not a full cache.
- The model prohibition — the part ADR-001 got right — is preserved exactly.

Negative / costs:

- One more owner in the lifetime model, and a refcounting scheme with the usual hazards.
- Eviction correctness now interacts with graph capture, which is a non-obvious coupling
  and must be tested rather than reasoned about.
- The block pool must be sized up front, so a bad estimate wastes device memory or
  refuses requests that would have fit.
- A session can now fail to acquire a lease. The API must express "waiting for capacity"
  as a normal state rather than an error.

## Alternatives considered

**Keep ADR-001's session ownership.** Simplest, and matches today's `State`. Rejected:
it makes the already-implemented paged and prefix-cached paths violations, and it caps
throughput at one sequence per cache.

**Cache on the model, keyed by session id.** Avoids changing `State` at all. Rejected:
makes the model shared-mutable, which is precisely the problem ADR-001 solved, and puts
eviction on the model.

**Per-session buffers allocated on demand.** The natural design if the CUDA-graph
constraint is unknown. Rejected on the invariant above — it breaks replay, and it breaks
it silently.

**Global static pool.** Avoids threading a manager through. Rejected: untestable, and it
makes two engines in one JVM interfere.

## Migration notes

Corresponds to the session/state milestone, and gates the engine milestone.

1. Introduce `KvCacheManager` + `BlockPool` with a single-lease implementation that
   behaves exactly like today's per-`State` cache. No behaviour change.
2. Split `State`: KV storage moves behind the lease; activation and scratch buffers stay
   with the session.
3. Promote #129's paged mode onto the manager. Verify graph capture still replays —
   this is the step where the invariant earns its place.
4. Add `PrefixCache` on top, with refcounting and pinning.
5. Only then allow leases to be handed to more than one concurrent session.

Test specifically: evict a block while a captured graph holds it, and assert the failure
is caught rather than silently producing wrong output. Run the golden suite with
`-Dtornado.recover.bailout=False`
([capability C4](../tornadovm-capabilities.md#c4--interpreter-bytecode-buffer-overflow-was-silent)).

## Open questions

1. Block size — fixed globally, or per model? Larger blocks waste more on short
   sequences; smaller blocks lengthen block tables.
2. Eviction policy when the pool is exhausted: refuse admission, preempt a running
   sequence, or swap blocks to host?
3. Does a lease span models? (Almost certainly not — prefix identity includes the model.)
4. How does the pool interact with `withMemoryLimit(String)`? Admission must reserve
   against the same budget the plan is limited to, or the scheduler admits requests the
   plan then refuses.
5. Is the block table a device array walked in-kernel (as #129 does) or host-side
   indirection resolved before launch?
