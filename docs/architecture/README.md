# GPULlama3.java — Architecture Baseline

This directory holds the **human-owned architecture baseline** for the evolution of
GPULlama3.java from a TornadoVM-based LLM engine into a structured Java inference
framework.

The framework direction, in one sentence:

> A Java-native transformer inference framework that compiles reusable Java inference
> components through TornadoVM into heterogeneous execution plans for local accelerators.

## Purpose

These documents exist so that:

- the **current** design is described accurately, from the code, in one place;
- the **proposed** design is written down before it is implemented;
- the boundaries that matter (what may depend on what) are explicit and reviewable;
- incremental refactoring work can be checked against a shared target instead of
  re-derived per pull request.

## Documents

| Document | Content | Normative? |
| --- | --- | --- |
| [`vision.md`](vision.md) | Mission, users, use cases, non-goals | Normative for scope |
| [`current-architecture.md`](current-architecture.md) | The repository as it exists today | Descriptive |
| [`target-architecture.md`](target-architecture.md) | Layering and dependency direction | **Normative** |
| [`terminology.md`](terminology.md) | Definitions used by all other documents | **Normative** |
| [`dependency-rules.md`](dependency-rules.md) | Allowed/forbidden dependency directions | **Normative** |
| [`tornadovm-capabilities.md`](tornadovm-capabilities.md) | What TornadoVM provides, minimum versions, runtime constraints | **Normative** |
| [`public-api.md`](public-api.md) | Developer-facing API surface | **Proposal** (names not final) |
| [`migration-roadmap.md`](migration-roadmap.md) | Milestones from today to target | **Normative** |
| [`decisions/`](decisions/README.md) | Architecture Decision Records | Accepted — see index |
| [`review/`](review/) | Review history: the ARCH-01..19 issues and responses | Historical record |

Read this way:

- **Normative** — treat as rules. Changing them requires an ADR or an explicit update
  to this baseline, agreed by maintainers.
- **Descriptive** — a snapshot of the code. If it disagrees with the code, the code
  wins and the document should be corrected.
- **Proposal** — a direction, not a commitment. Names, packages and signatures in
  proposal documents are placeholders until an ADR is accepted and code exists.

`terminology.md` and `dependency-rules.md` are normative today because they constrain
work that is already happening. `target-architecture.md` and `public-api.md` describe
the destination and are expected to change as ADRs are accepted.

## Proposing an architectural change

1. Open an issue or discussion describing the problem, not the solution.
2. Add an ADR under [`decisions/`](decisions/) using the format described there, with
   status `Proposed`.
3. Reference the ADR from the pull request that implements it.
4. When maintainers accept it, update the ADR status to `Accepted` and update
   `target-architecture.md`, `terminology.md` and `dependency-rules.md` in the same
   change if the decision affects them.

Small clarifications and corrections to descriptive documents do not need an ADR.

## Human ownership

**These documents are human-owned.**

AI tools (including coding agents used in this repository) may:

- propose changes to these documents through the normal review process;
- implement work that these documents describe;
- point out where the documents no longer match the code.

AI tools may **not**:

- silently redefine the architecture, terminology, layering or dependency rules;
- mark an ADR `Accepted` without evidence of maintainer approval;
- introduce new abstractions here that no one has asked for.

Every change to this directory goes through human review like any other change.

## Status of this baseline

**Baseline v1.0 — agreed 2026-07-30.**

History: the initial baseline was written by inspecting the repository, reviewed as 19
ARCH issues (PR #140), and both maintainers recorded a position on every issue. All 19
were accepted, six with modifications. The resulting decisions are folded into the
documents above; ADR-001 through ADR-006 are `Accepted`.

The review record is preserved under [`review/`](review/) — the issues, the positions and
the corrections. Two positions were reversed by evidence during review and are worth
knowing about, because the corrected reasoning is now load-bearing:

- concurrent execution plans on one device **are** supported; batching is chosen for
  device-memory reasons instead
  ([C2](tornadovm-capabilities.md#c2--device-buffers-are-per-task-graph));
- the pinned TornadoVM version predates several capabilities the design assumes, which is
  why a version floor is Phase 0.

Nothing in this directory has changed production code. Implementation starts from
[`migration-roadmap.md`](migration-roadmap.md).

Every position here is grounded in a capability listed in
[`tornadovm-capabilities.md`](tornadovm-capabilities.md). A proposal that depends on a
capability absent from that ledger is not grounded, and must either be re-grounded or
raised as an upstream TornadoVM proposal.
