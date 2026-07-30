# Architecture Decision Records

An ADR records one architectural decision: the context that forced it, what was
decided, and what it costs. It is a historical record — once accepted, an ADR is not
rewritten. It is superseded by a later ADR.

## Format

Every ADR uses these sections, in this order:

```markdown
# ADR-NNN: Title

## Status
## Context
## Decision
## Consequences
## Alternatives considered
## Migration notes
## Open questions
```

## Status values

| Status | Meaning |
| --- | --- |
| `Proposed` | Written, under discussion. Not binding. |
| `Accepted` | Agreed by maintainers. Binding for new code. |
| `Rejected` | Considered and declined. Kept for the reasoning. |
| `Superseded by ADR-NNN` | Replaced. Kept for the history. |
| `Deprecated` | No longer relevant; the situation changed. |

ADR-001 through ADR-006 were accepted on 2026-07-30, after the ARCH-01..19 review on
PR #140 in which both maintainers recorded a position on every issue. An ADR moves to
`Accepted` only when a maintainer says so — an AI tool must not change the status.

## Current ADRs

| ADR | Title | Status |
| --- | --- | --- |
| [ADR-001](ADR-001-model-session-separation.md) | Model and session separation | Accepted 2026-07-30 |
| [ADR-002](ADR-002-program-and-compiled-program.md) | Inference program and compiled program | Accepted 2026-07-30 |
| [ADR-003](ADR-003-tornado-backend-boundary.md) | TornadoVM backend boundary | Accepted 2026-07-30 |
| [ADR-004](ADR-004-tensor-and-format-separation.md) | Tensor and format separation | Accepted 2026-07-30 |
| [ADR-005](ADR-005-kv-cache-ownership-and-leases.md) | KV cache ownership, block pools and leases | Accepted 2026-07-30 |
| [ADR-006](ADR-006-engine-tier.md) | The engine tier | Accepted 2026-07-30 |

ADR-005 refines ADR-001's KV cache placement. ADR-006 adds the tier ADR-001 assumed
did not exist.

## Adding an ADR

1. Take the next free number. Numbers are never reused.
2. Name the file `ADR-NNN-short-kebab-title.md`.
3. Start at `Proposed`.
4. Add a row to the table above.
5. Link it from the pull request that implements it.

Keep it short. An ADR that needs ten pages is usually two decisions.
