# START HERE — Maintainer Control Page

Last updated 2026-08-03. This page is self-contained: it tells you what is decided,
what needs you now, and what can wait. Details live in the linked documents.

## The goal, in five sentences

GPULlama3.java stays a working TornadoVM-based LLM engine while growing into a
structured Java inference framework. Users get a small public API (load model →
open session → generate) with no TornadoVM, GGUF or CLI types in any signature.
Internally, ownership is split into model / compiled program / session / invocation
lifetimes, KV storage moves behind an engine-scoped manager with leases, and an
engine tier adds continuous batching (promoting PR #129, up to 41× aggregate
throughput). TornadoVM remains the compiler and runtime — this project never builds
its own IR or code generator. Every step keeps goldens bit-identical and the
benchmark gate green; nothing is rewritten big-bang.

## Phase map

    DONE      Architecture baseline v1.0, ARCH-01..19 review, ADR-001..007 accepted,
              roadmap-hardening pass, Phase 0 TornadoVM 5.2.0 bump
    CURRENT   ▶ M1 guardrails
    NEXT      M2 metrics seam · M3 façade (needs its decision gates) · M4 dtypes · M5 providers
    LATER     M6 state split · M7 engine · M8 ops · M9 programs · M10 policy ·
              M11 providers B · M12 backend SPI · M13 polish

## Decision needed NOW (blocks current work)

**None.** ADR-007 (hardened roadmap structure) was accepted 2026-08-03 — the staged
façade, `LoadTarget` adapter, capacity staging, M7 split and land order #129→#138 are
binding. Phase 0 and M1 have no open decision gates; work can proceed.

Next decision batch you will face: the M3 gate set (see below), needed before the
façade starts — not before.

## Not needed yet (deferred decisions, grouped by when they bite)

| Needed before | Gates | Topic |
| --- | --- | --- |
| M3 façade | D-01, D-03, D-04, D-06, D-07, D-11 | generation capability, prompt/messages, device in façade, low-level forward, experimental policy, model-close |
| M4 / M5 | D-08, D-09, D-21 | tensor descriptor shape, DataType value set, one SPI or two |
| M6 state split | D-10, D-11, D-12, D-14 | engineless manager owner, invoke concurrency, block table |
| M7 engine | D-13, D-15…D-19, D-24 | block size, step()/threading, scheduling, batch sizing, ragged, callbacks |
| M8 / M9 / M10 | D-22, D-20, D-23, D-05, D-02 | dequant home, program phases, signature, program cache, policy level |
| M12 | module split, CPU backend shape | after package boundaries hold |

Full table with recommendations: [`decision-gates.md`](decision-gates.md).
Do not resolve these early — deciding before the gate is speculation.

## Safe to start now (prerequisites satisfied)

- **T1.1–T1.3** — ArchUnit module + rules 1/2/5/7/11/8a/16 with allowlists.
  Tests only; needs nothing.
- **T1.4–T1.7** — goldens, parity, identity test, benchmark gate. Unblocked now that
  Phase 0 has landed; generate goldens on the 5.2.0 floor only.
- **Retarget PR #129** from `feat/mma_cuda` to `main` (no rebase needed).
- **Assess PR #142** against M1.7's CI changes before either lands.

## Stop and ask the maintainer

- Changing any ADR status, or any milestone objective/acceptance (needs an ADR).
- Growing an ArchUnit allowlist (shrink-only policy).
- Regenerating goldens (explicit reviewed commit only).
- Any new public API name or signature beyond the façade v1 surface.
- Exposing TornadoVM, GGUF or CLI types in a public signature.
- Feature work in frozen trees once M6/M7 open (`inference/state/**`,
  `tornadovm/plan/**`, `tornadovm/layers/type/**`, later `bench/BatchedDecode*`,
  `server/**`).
- Merging or pushing this branch anywhere.

## Status table

| Phase | Purpose | Prereqs | Status | Exit condition |
| --- | --- | --- | --- | --- |
| Phase 0 | TornadoVM ≥ 5.2.x floor | — | **done** (pom pins 5.2.0) | build+launcher work; fresh perf entry |
| M1 | Guardrails: ArchUnit, goldens, gates | Phase 0 (for T1.4+) | **current** (T1.1–T1.7 unblocked) | rules + goldens + bench gate in CI |
| M2 | Metrics seam | M1.4 | waiting | counters programmatic; no tok/s cost |
| M3 | Public API façade v1 | M1 + gates D-01..D-11 | **blocked on gates** | simple example runs CPU+GPU, token-identical |
| M4 | DataType / GGUF isolation | M1; D-08/D-09 | waiting | Rule 4 allowlist empty |
| M5 | Provider SPI part A | M4; D-21 | waiting | families load via ServiceLoader |
| M6 | Session/state split (highest risk) | M1, M3, PR #138; D-10/12/14 | waiting | goldens green all families; Rule 2 shrinks |
| M7 | Engine tier (the product win) | M6, PR #129; D-13..D-19 | waiting | #129's numbers reproduced through the engine |
| M8 | Operation vocabulary | M4; D-22 | waiting | ops defined once; no kernel rewrites |
| M9 | Program / compiled program | M6, M8; D-20/23 | waiting | same task graphs; goldens green |
| M10 | Execution policy | M6, M9; D-02 | waiting | static-property flags replaced |
| M11 | Provider SPI part B | M5, M9 | waiting | Rule 15 passes |
| M12 | Backend SPI + package move | M9, M10 (M7 for capacity) | waiting | Rules 1/11 allowlists empty |
| M13 | Memory planning, DX, API freeze | M12 | waiting | experimental marker removed |

## Where the details live

- **Implementers start here:** [`execution-backlog.md`](execution-backlog.md) —
  PR-sized tasks with acceptance; [`migration-roadmap.md`](migration-roadmap.md) —
  milestone objectives.
- **Before touching state/engine code:** [`ownership-and-lifecycle.md`](ownership-and-lifecycle.md),
  [`engine-contract.md`](engine-contract.md).
- **Test/gate machinery:** [`verification-gates.md`](verification-gates.md).
- **Rules enforced in CI:** [`dependency-rules.md`](dependency-rules.md).
- **API shape:** [`public-api.md`](public-api.md). **Layering:**
  [`target-architecture.md`](target-architecture.md).
- **What TornadoVM can and cannot do:**
  [`tornadovm-capabilities.md`](tornadovm-capabilities.md).
- **Words:** [`terminology.md`](terminology.md). **Session state:**
  [`HANDOFF.md`](HANDOFF.md).

## Document status at a glance

- **Normative (binding):** terminology, dependency-rules, target-architecture,
  tornadovm-capabilities, migration-roadmap, execution-backlog, vision (scope),
  decision-gates (index), ADR-001..007.
- **Proposed (await your approval):** verification-gates, ownership-and-lifecycle,
  engine-contract, public-api (names not final).
- **Operational (transient):** HANDOFF.md, this page.
- **Historical (never edited):** review/ (ARCH issues, responses, hardening report),
  accepted ADR texts.
