# Session handoff

**Transient working state, not architecture.** Delete when stale.

Read [`README.md`](README.md) first — it indexes the architecture documents. This file
only records what those documents do **not** say: git state, PR mechanics, decisions made
in conversation, and the immediate next actions.

## Where things stand

Architecture is **finalized**. Baseline v1.0, ADR-001..006 all `Accepted`. The 19-issue
review (PR #140) is closed out — all 19 accepted, six with modifications, none rejected.
Implementation has **not** started. No production file has been touched.

## Git state

- Branch: **`refactor/framework-abstractions`**. All work happens here.
- **5 commits ahead of `upstream/main`, none pushed.**
- Owner's decision: **not merging to `main`** for now. Do not merge or push without asking.

```
f7b5b79  separate resolved from open questions
d5e9847  fold ARCH-01..19 review decisions into the baseline   ← baseline v1.0
7069247  ARCH-16..19 issues        ┐ merged in from review/arch-issues
d98c13d  ARCH-01..15 issues        ┘ (PR #140)
7ee6f86  initial architecture baseline
```

**PR #140 still shows open on GitHub** — its branch was merged locally and never pushed.
Either push, or close it noting it is merged.

Untracked leftovers in `review/`, safe to delete, already superseded:
`ARCH-responses-pr-comments.md` (posted to the PR), `baseline-v1-candidate.md`
(dissolved into `migration-roadmap.md`).

## Pending PRs — measured, not guessed

Test-merged against `upstream/main` on 2026-07-31. The roadmap gives the *land order*;
this is the mechanical state, which it does not carry.

| PR | vs `main` | Note |
| --- | --- | --- |
| #120 Gemma 4 | **CLEAN** | GitHub reports UNKNOWN; it merges clean. Needs the bump first — BF16 requires `BFloat16Array` (5.2.0) |
| #129 batched decode | **CLEAN** | Targets `feat/mma_cuda`, which is **already contained in `main`** — retarget the PR, no rebase needed |
| #131 hybrid libs | 2 conflicts | `llama-tornado`, `LogitsFP16Layer.java` |
| #138 FP16 KV cache | **1 conflict: `pom.xml` only** | Everything else auto-merges |
| #142 ci/metal-migration | not assessed | Appeared after the review; not in the roadmap. Check whether it touches CI in ways that affect M1.7 |

**#138's only conflict is the version bump.** It already moves `tornadovm.base.version`
5.0.0 → **5.1.1** with a `-jdk21-dev` suffix. That is a partial Phase 0, and 5.1.1 is not
enough: it provides `FP8Array` but not `BFloat16Array`, so #120's BF16 path still fails on
it. Do Phase 0 to **≥5.2.x** separately, then #138 rebases conflict-free.

**#129 is the time-critical one.** Clean today, +2694 lines into exactly the files M6 and
M7 restructure, and it is the source for the entire engine tier. Landing it early is much
cheaper than rebasing it later.

## Immediate next actions

1. **Phase 0** — owner is doing the TornadoVM bump himself. Sanity check afterwards: the
   wait-event fix should show ~53 → ~103 tok/s and start-up ~11.5 s → ~5.2 s. If those
   don't move, the bump didn't take.
2. **M1 can start now.** Test-only, no PR blocks it. Generate goldens *after* the bump —
   before it they are invalid.
3. Then #138 → #129 (retarget) → #120 → #131.

## Working conventions

- **The architecture documents are human-owned.** Propose and implement; never silently
  redefine. ADR status changes are the owner's call, not yours.
- **Ground every claim in a real capability.** [`tornadovm-capabilities.md`](tornadovm-capabilities.md)
  is the ledger, with a minimum-version column because the pinned version lags the
  development tree. Verify against the local checkout at `~/TornadoVM` (currently
  5.2.1-jdk21-dev) rather than trusting recall — two claims were reversed by exactly this
  check during the review.
- **Verify assertions before writing them into a document.** Both maintainers had claims
  corrected by inspection during the review; that is the expected standard here, not an
  exception.
- Markdown links are checked before each commit — 20 files, all internal links and
  anchors must resolve.

## Two facts worth not re-deriving

Both were wrong in the first draft of the baseline and are now load-bearing:

1. **Concurrent independent `TornadoExecutionPlan`s on one device are supported and
   tested** (`TestMultiThreadedExecutionPlans`, 4/4 on CUDA). Batching is chosen because
   device buffers are per task graph — a plan per session duplicates the weights, ~3.4 GB
   on 3B-Q8. Not because an API is missing.
2. **CUDA-graph capture fixes device addresses.** Re-pointing a captured buffer fails at
   replay with `CUresult=700`, and `tornado.recover.bailout` defaults to `TRUE`, so the
   symptom is **wrong output, not an error**. This is why the KV block pool must be one
   persistent array with in-kernel indexing, and why the golden suite runs with
   `-Dtornado.recover.bailout=False`.
