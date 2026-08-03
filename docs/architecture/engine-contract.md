# Engine Contract — Request Lifecycle and Scheduling

**Status: proposed-normative.** This is the behavioural contract M7 implements. Rows
marked **[D-nn]** depend on an open [decision gate](decision-gates.md) and show the
current recommendation. M7 implementation must not start while a gate its task needs is
open.

Companion documents: [ADR-006](decisions/ADR-006-engine-tier.md) (why the tier exists),
[ADR-005](decisions/ADR-005-kv-cache-ownership-and-leases.md) (KV ownership),
[`ownership-and-lifecycle.md`](ownership-and-lifecycle.md) (who owns which buffer).

## Request state machine

```
                 addRequest(request)
                        │
                        ▼
                    ┌────────┐   admission rejects permanently
                    │ QUEUED │ ──────────────────────────────► REJECTED (terminal)
                    └────────┘     (malformed, over context limit,
                        │           engine shutting down)
        admission grants blocks + slot
                        ▼
                    ┌─────────┐
                    │ RUNNING │◄─────────────┐
                    └─────────┘              │ (re-admitted after
                        │                    │  preemption — M7-ext only)
        ┌───────────────┼────────────────┐   │
        ▼               ▼                ▼   │
   COMPLETED        CANCELLED        PREEMPTED (M7-ext)
   (terminal)       (terminal)           │
        ▲                                └───► back to QUEUED
        │
   FAILED (terminal — kernel error, memory-limit breach,
           callback failure, context overflow mid-decode)
```

Allowed transitions and nothing else:
`QUEUED → RUNNING | REJECTED | CANCELLED`,
`RUNNING → COMPLETED | FAILED | CANCELLED | PREEMPTED`,
`PREEMPTED → QUEUED | CANCELLED`. Terminal states never transition.

## Admission

- `addRequest(...)` is **non-blocking** and thread-safe. It validates the request,
  assigns a handle in `QUEUED`, and returns. It never runs model work.
- **Rejection** (terminal, immediate) is only for requests that can *never* run:
  prompt exceeds the model/session context length; engine is shutting down; request is
  malformed. Capacity shortfall is **not** rejection — the request queues (ADR-005:
  "waiting for capacity is a normal state rather than an error").
- **Queueing**: FCFS order [D-16]. Admission into `RUNNING` happens at step boundaries
  when the scheduler can reserve the KV blocks the prompt needs **against the same
  budget `withMemoryLimit` bounds**, via the manager's capacity contract (ADR-007 D5).
- **Backpressure**: the queue has a configurable bound; `addRequest` beyond it returns a
  handle already in `REJECTED` with a queue-full reason. Blocking-submit convenience can
  be layered above; the engine itself never blocks in `addRequest`.

## Token and result delivery

- Each step appends newly decoded tokens to the handle. Delivery surface on the handle:
  poll (`drainTokens()`), await-completion, and an optional per-request `onToken`
  callback.
- **Callback threading [D-24]:** callbacks run on the thread executing `step()` —
  the engine does not own a dispatch thread in v1 [D-15]. Contract stated in Javadoc: a
  slow callback slows the whole batch; offload if you need isolation.
- **Callback failure [D-24]:** an exception from a callback moves *that request* to
  `FAILED` (recorded as callback failure) and releases its lease; it must not poison the
  step or other requests.

## Cancellation and handle close

- `handle.cancel()` from any thread: `QUEUED → CANCELLED` immediately;
  `RUNNING → CANCELLED` at the next step boundary. The lease's private blocks return to
  the pool; shared prefix blocks are unrefed.
- `handle.close()` on a non-terminal handle is cancellation. Closing a terminal handle
  releases delivery buffers only. Handles are cheap; leaking one leaks tokens buffered
  for delivery, never device memory (device memory follows the lease, released at the
  terminal transition).

## Failure classes

- **Context-limit at admission** → `REJECTED` (can never fit).
- **Context full mid-decode** → `COMPLETED` with `FinishReason.CONTEXT_FULL` (matches
  the existing finish-reason vocabulary; not a failure).
- **Memory-limit / pool exhaustion mid-run** (should be prevented by admission; can
  still happen with adaptive policies) → the youngest affected request(s) `FAILED` with
  a capacity reason [D-13 exhaustion policy]; the step must not silently truncate.
- **Kernel/backend error** → all requests in the failed step `FAILED` with the backend
  error attached; the engine stays usable if the backend reports the plan recoverable,
  otherwise the engine transitions to shutdown.
- Golden/parity rule: failure detection relies on `-Dtornado.recover.bailout=False` in
  test environments (C4); production behaviour with the default is documented, not
  hidden.

## step() and threading — [D-15]

- `step()` **blocks** for exactly one batched iteration: compose batch → invoke compiled
  program once → sample/deliver → release finished leases → admit from queue.
- The engine owns **no background thread** in v1. Callers (server accept loop, CLI,
  tests) drive `step()` in a loop. A `run()` convenience that loops until shutdown may
  be provided, executing on the caller's thread.
- `step()` is single-caller: concurrent `step()` calls are a caller bug and throw.
  `addRequest`/`cancel` remain safe concurrently with a running step.

## Scheduling baseline

- **v1 (minimum viable engine):** FCFS admission, no priorities, **no mid-sequence
  preemption** [D-16]. A sequence admitted runs to a terminal state.
- **Batch sizing:** fixed B chosen at engine construction [D-17], matching #129's
  static mode and the CUDA-graph buffer invariant (growing B = recapture, a policy
  event). Adaptive B is M7-ext.
- **Ragged shapes [D-18]:** v1 decodes one token per running slot per step (uniform);
  prefill of a newly admitted request occupies its slot for the prefill phase. Mixed
  prompt lengths across slots follow #129's demonstrated behaviour; anything beyond it
  is M7-ext with its own gate.
- **Fairness/starvation:** FCFS + no preemption cannot starve an admitted request; a
  queued request can wait unboundedly under load — queue wait is a first-class metric
  (M7 metrics) precisely so operators can see it.

## Engine shutdown

`engine.close()`:

1. stops admission (`addRequest` → immediate `REJECTED`, shutdown reason);
2. completes or cancels running work — default: finish the current step, then cancel
   remaining `RUNNING` requests (fast shutdown); a `drain()` variant runs queued and
   running work to completion first;
3. releases all leases, the manager, pool, batched buffers;
4. terminal-izes every outstanding handle before returning. `close()` never leaves a
   handle in a non-terminal state.

Engine close does **not** close the model — the engine borrows the model.

## Minimum viable engine vs extensions

Split per ADR-007 D6, tasks in the
[execution backlog](execution-backlog.md#m7--engine-tier):

| Stage | Contents |
| --- | --- |
| **M7-core** | paged KV on the manager (M7.1), FCFS scheduler + admission (M7.2), `LLMEngine`/handles/`step()` (M7.4), server migration |
| **M7-ext** | prefix cache (M7.3), serving metrics (M7.5), preemption, adaptive batch sizing, ragged extensions, `-Dbatch.decode.*` retirement (M7.6, after M10.2) |

M7-core alone must reproduce #129's continuous-batching numbers; that is its benchmark
gate.
