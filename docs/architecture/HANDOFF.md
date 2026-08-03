# Session handoff

**Transient working state, not architecture.** Delete when stale. Nothing normative
lives here — the roadmap, backlog and gates own the plan; this file only records git
state, PR mechanics and immediate next actions. Last refreshed **2026-08-03** (roadmap-
hardening pass).

Read [`README.md`](README.md) first — it indexes the architecture documents.

## Where things stand

Architecture baseline v1.0, ADR-001..007 `Accepted` (**ADR-007 accepted 2026-08-03**,
same day the hardening pass produced it). The pass added five companion documents
([`decision-gates.md`](decision-gates.md), [`execution-backlog.md`](execution-backlog.md),
[`verification-gates.md`](verification-gates.md),
[`ownership-and-lifecycle.md`](ownership-and-lifecycle.md),
[`engine-contract.md`](engine-contract.md)), the
[hardening report](review/roadmap-hardening-2026-08-03.md) and
[`START-HERE.md`](START-HERE.md). Roadmap + backlog are now binding. The **gate
recommendations** (D-01..D-24) remain open — the M3 set is the next maintainer batch.
Implementation has not started; no production file has been touched.

## Git state (verified 2026-08-03)

- Branch: **`refactor/framework-abstractions`**, 6 commits ahead of `upstream/main`,
  **pushed** to `upstream/refactor/framework-abstractions`.
- Owner's decision: **not merging to `main`** for now. Do not merge or push without asking.
- PR #140 is **closed** on GitHub (previous handoff said open — resolved).
- `pom.xml` still pins TornadoVM **5.0.0** — Phase 0 has not happened yet.

## Pending PRs (open on GitHub, verified 2026-08-03)

Land order is normative in the
[roadmap](migration-roadmap.md#pr-land-order): Phase 0 → **#129 → #138** → #120 → #131,
with #142 assessed against M1.7 first. (An earlier handoff said #138 before #129; the
roadmap order carries the rationale and stands — ADR-007 D9.)

| PR | Note |
| --- | --- |
| #129 batched decode | **Still targets `feat/mma_cuda`** (already contained in `main`) — retarget to `main`, no rebase needed. Time-critical: +2694 lines into exactly the files M6/M7 restructure |
| #138 FP16 KV cache | Only conflict is `pom.xml` (it bumps to 5.1.1, which is a partial Phase 0 and not enough — no `BFloat16Array`). Do Phase 0 to ≥5.2.x separately, then #138 rebases clean |
| #120 Gemma 4 | Merges clean; needs the bump first (BF16 → `BFloat16Array`, 5.2.0) |
| #131 hybrid libs | 2 conflicts (`llama-tornado`, `LogitsFP16Layer.java`); last, additive, default-off |
| #142 ci/metal-migration | Not in the roadmap; check interaction with M1.7's CI changes before either lands |

## Immediate next actions

1. **Phase 0** — owner is doing the TornadoVM bump himself. Sanity check: ~53 → ~103
   tok/s and start-up ~11.5 s → ~5.2 s; if those don't move, the bump didn't take.
2. **M1 can start now** (T1.1–T1.3 need nothing; T1.4 goldens wait for the bump).
3. Then #129 (retarget) → #138 → #120 → #131 per the land order.
4. Before M3 opens: maintainer closes the M3 gate set
   (D-01, D-03, D-04, D-06, D-07, D-11) in [`decision-gates.md`](decision-gates.md).

## Working conventions

- **The architecture documents are human-owned.** Propose and implement; never silently
  redefine. ADR status changes are the owner's call, not yours.
- **Ground every claim in a real capability.** [`tornadovm-capabilities.md`](tornadovm-capabilities.md)
  is the ledger. Verify against the local checkout at `~/TornadoVM` (5.2.1-jdk21-dev)
  rather than trusting recall.
- **Verify assertions before writing them into a document** — the review standard here.
- Markdown links checked before each commit — all internal links and anchors must
  resolve (now 26 files under `docs/architecture/`).

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
