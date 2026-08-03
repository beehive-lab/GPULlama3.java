# ADR-007: Roadmap ordering and transitional contracts

## Status

**Accepted** — 2026-08-03, maintainer approval recorded in session. Produced by the
roadmap-hardening pass ([review record](../review/roadmap-hardening-2026-08-03.md)).

The roadmap states that changing a milestone's objective or acceptance criteria needs an
ADR. The hardening pass found ordering contradictions whose resolution changes several
acceptance criteria, so those resolutions are collected here for one review. No accepted
ADR (001–006) is modified; this ADR only sequences their implementation.

## Context

The ARCH-01..19 review produced an agreed baseline, but the roadmap it settled contains
backward dependency edges — milestones whose stated acceptance cannot pass until a later
milestone lands:

1. **Phase 0 → M1.4.** Phase 0's acceptance requires goldens "generated once on the new
   floor and committed", but the golden infrastructure is created by M1.4, which depends
   on Phase 0.
2. **M1 → M1.8.** M1 is titled "tests only, no production code", but M1.8 changes
   `AbstractModel` (three fields become final) — a production change.
3. **M3 → M4/M10/M12.** The M3 façade sketch exposes `DataType` (M4), `ExecutionPolicy`
   (M10) and `Backend`/`DeviceSelector` (M12) in public signatures, while M3 depends only
   on M1.
4. **M5 → M12.** The `ModelProvider` SPI sketch takes a `Backend` parameter, but the
   backend SPI is created in M12, seven milestones later.
5. **M7 → M12.** M7.2's admission "reserves against the same budget `withMemoryLimit`
   bounds", and M12.2's acceptance says "engine admission consumes the capacity query" —
   the query M7 needs is introduced five milestones after M7.
6. **M7's dependency line understates its inputs.** M7.5 needs the M2 metrics sink; M7.6
   is explicitly "designed together with M10"; the stated dependency is "M6 only".
7. **M11 → M5.** M11 removes the `ForwardPlanFactory` family branches, which requires the
   provider SPI (M5) as well as the program layer (M9); only M9 is listed.
8. **M6 ordering vs ADR-001.** ADR-001's migration notes place M6 "after the M3 façade
   exists"; the roadmap lists only M1 and PR #138.

## Decision

### D1 — Goldens move out of Phase 0's acceptance

Phase 0's acceptance becomes: build and launcher work on the new version; a fresh
`perf-history.jsonl` entry recorded. Golden generation is M1.4's job and happens **on the
new floor** — M1.4 keeps its dependency on Phase 0, and the roadmap states explicitly
that goldens generated on the old floor are invalid.

### D2 — M1.8 is the declared exception to "tests only"

M1 is retitled "Guardrails". M1.1–M1.7 touch no production file. M1.8 is a mechanical
production change (three final keywords, no behaviour change) and is ordered **after
M1.4**, so the first production change lands with goldens already in place — the same
rule M2 already follows.

### D3 — The M3 façade is staged; version 1 exposes only what exists

M3 ships façade v1 with signatures that reference no type designed by a later milestone:

- `ModelOptions` / `SessionOptions` v1 carry `contextLength` only.
- `ModelInfo` v1 exposes `name`, `architecture`, `contextLength`; the two dtype accessors
  are **added in M4.7**.
- `executionPolicy(...)` builder methods are **added in M10.2**.
- `backend(...)` / `device(...)` builder methods are **added in M12.1**.

All façade types carry the experimental marker (M3.4), which is what makes additive
staging legitimate. Nothing placeholder-shaped is exposed to make code compile: the
options builders simply do not have those methods yet.

M3 additionally acquires a **decision gate**: the signature-defining questions in
[`decision-gates.md`](../decision-gates.md) marked "before M3.1" must be closed before
the façade types are committed to. See ADR-001/Rule 14 for the generation-capability
question.

### D4 — `ModelProvider` does not take `Backend`; a named transitional adapter carries the target

M5's SPI is `supports(ModelSource)` / `load(ModelSource, ModelOptions, LoadTarget)`,
where **`LoadTarget` is a transitional, internal (non-public) adapter** wrapping what the
loaders consume today: the use-TornadoVM boolean and the device selection the launcher
resolves. It is package-private to the provider machinery, documented as transitional,
and **removed by M12.6**, which replaces it with the real `Backend`/`Device` SPI types.
`LoadTarget` must never appear in a public signature.

### D5 — Capacity is staged: internal contract in M6.2, SPI form in M12.2

M6.2's `KvCacheManager` exposes an **internal capacity contract** from day one: total
blocks, free blocks, and the byte budget the pool was sized against (the same budget
`withMemoryLimit` bounds). M7.2's admission consumes that internal contract. M12.2
formalizes capacity as a backend-SPI query and **migrates the manager and the engine onto
it**; its acceptance becomes "the internal capacity accounting is reimplemented on the
SPI query with admission behaviour unchanged (test)".

### D6 — M7's dependencies are stated in full, by kind

- **PR land prerequisite:** #129 (and transitively #138, via M6).
- **Milestone prerequisites:** M6 (all of M7); M2 (M7.5 only).
- **Design prerequisite:** the `ExecutionPolicy` shape (a decision, not M10's
  implementation) for M7.6, which cannot retire `-Dbatch.decode.*` until the replacement
  policy value exists. M7.6 therefore lands **after M10.2** even though it remains listed
  under M7.
- M7 is additionally split into a **minimum viable engine** (M7.1, M7.2, M7.4) and an
  **extension stage** (M7.3 prefix cache, preemption, M7.5 metrics, M7.6) — see the
  [execution backlog](../execution-backlog.md#m7--engine-tier).

### D7 — M11 depends on M5 and M9

Stated in the roadmap and the dependency diagram.

### D8 — M3 is a design prerequisite of M6

M6.1's session type must be presentable through the M3 façade (ADR-001 migration note 2),
and M6.5 reimplements `InferenceService` on the session type. M6's dependency line
becomes: M1, M3 (design), PR #138 (land).

### D9 — PR land order confirmed: #129 before #138

The roadmap's order stands (Phase 0 → #129 → #138 → #120 → #131) and `HANDOFF.md` is
corrected to match. Rationale unchanged from the ARCH-13 response: #129 merges clean
today, is +2694 lines into exactly the files M6/M7 restructure, and is the requirements
document for the engine tier; #138's only conflict is the version bump, which Phase 0
resolves regardless of when #138 lands after it.

## Consequences

Positive: the dependency graph has no backward edges; every milestone's acceptance can
pass when the milestone closes; transitional code (`LoadTarget`, staged façade methods,
internal capacity contract) is named and has a recorded removal or formalization point.

Negative / costs: façade users see options builders grow across releases (mitigated by
the experimental marker); `LoadTarget` is one more thing to delete, tracked by M12.6;
M7.6 formally straddles two milestones, which the backlog has to carry explicitly.

## Alternatives considered

**Move M12 (backend SPI) before M5/M7.** Rejected: M12 is a repository-wide package move
with high compatibility risk; pulling it earlier puts the riskiest change ahead of the
guardrail and state work it depends on (M9, M10).

**Expose placeholder `Backend`/`ExecutionPolicy` types in M3 to keep the sketch whole.**
Rejected: a placeholder in a public signature becomes permanent API the moment anyone
compiles against it, experimental marker or not.

**Give M7 its own private capacity accounting, independent of the manager.** Rejected:
admission and the block pool must count the same blocks or the scheduler admits requests
the pool then refuses — ADR-005 open question 4 already records this hazard.

## Migration notes

Folded into [`migration-roadmap.md`](../migration-roadmap.md) and expanded into PR-sized
tasks in [`execution-backlog.md`](../execution-backlog.md). No production code changes in
this ADR.

## Open questions

None of its own. The decision gates this ADR references are tracked in
[`decision-gates.md`](../decision-gates.md).
