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

## Documents, by audience

**Maintainers: read [`START-HERE.md`](START-HERE.md) first — nothing else required.**
It is the control page: what is decided, what needs you now, what can wait.

### Maintainer

- [`START-HERE.md`](START-HERE.md) — control page (phase map, decisions needed, status)
- [`decision-gates.md`](decision-gates.md) — open decisions, owners, decide-before points
- [`decisions/`](decisions/README.md) — ADRs (001–007 Accepted)

### Implementer

- [`execution-backlog.md`](execution-backlog.md) — PR-sized tasks with acceptance criteria
- [`migration-roadmap.md`](migration-roadmap.md) — milestone objectives and dependencies
- [`verification-gates.md`](verification-gates.md) — gate classes; golden/parity/benchmark specs
- [`ownership-and-lifecycle.md`](ownership-and-lifecycle.md) — who owns what, close semantics
- [`engine-contract.md`](engine-contract.md) — request state machine, scheduling, shutdown
- [`HANDOFF.md`](HANDOFF.md) — transient session state (git, PR mechanics)

### Architecture reference

- [`vision.md`](vision.md) — mission, users, non-goals (normative for scope)
- [`target-architecture.md`](target-architecture.md) — layering and dependency direction
- [`public-api.md`](public-api.md) — API surface proposal (names not final)
- [`dependency-rules.md`](dependency-rules.md) — allowed/forbidden dependencies
- [`terminology.md`](terminology.md) — definitions all documents use
- [`tornadovm-capabilities.md`](tornadovm-capabilities.md) — capability ledger, version floors
- [`current-architecture.md`](current-architecture.md) — the code as it is (descriptive)

### Historical record

- [`review/`](review/) — ARCH-01..19 issues and responses; 2026-08-03 hardening report

Document status classes (normative / proposed / operational / historical) are listed
per document in [`START-HERE.md`](START-HERE.md#document-status-at-a-glance). The short
rules: **normative** documents change only via ADR or agreed baseline update;
**descriptive** documents lose to the code; **proposal** documents carry placeholder
names until an ADR accepts them and code exists.

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
