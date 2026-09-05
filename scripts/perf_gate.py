#!/usr/bin/env python3
"""
Benchmark gate for the refactor (M1.7 / T1.7).

Compares a candidate measurement against the most recent gate-passing entry of the
*same tuple* in a `perf-history.jsonl`, and appends the candidate to that history.
The rules it implements are stated in docs/architecture/verification.md
("Benchmark gate (M1.7)"); this module is their executable form:

  tuple        (machine, gpu, model, quantization, backend, configuration,
                tornadovm_version) — comparisons only ever happen within one tuple
  procedure    3 warm-up generations discarded, then 5 measured runs; the metric is
                decode eval_rate (tok/s) and the aggregate is the median of the 5
  baseline     the most recent gate-passing entry of the same tuple
  tolerance    per tuple, from scripts/perf-gate-tolerances.json; default 3%
  no baseline  record-only pass; the entry becomes the baseline
  noisy        spread (max-min)/median above the configured limit reports
                "unstable environment": neither a pass nor a new baseline
  cache        throughput is compared warm-to-warm only; a cold run is record-only

Paired mode (--baseline-samples / --baseline-metrics-dir) compares two builds measured in
the *same session* instead of against history. A stored baseline ages: the same laptop
measured 172.5 tok/s and, two hours later, 167 tok/s from an unchanged build, which the
history gate reported as a 3% regression. Only a comparison whose two sides share a
thermal state says anything about the code, so a change-vs-change verdict must come from
paired mode; history comparison remains for tracking a tuple over time.

Usage:

  # from a directory of per-run metrics JSON files written by the engine
  python3 scripts/perf_gate.py --metrics-dir perf-results/run-42 \\
      --machine rtx5090-laptop --gpu "NVIDIA GeForce RTX 5090 Laptop GPU" \\
      --model Llama-3.2-1B-Instruct --quantization Q8_0 --backend cuda \\
      --configuration standard --tornadovm-version 5.2.0-jdk21 --cache-warm true \\
      --history docs/perf-history.jsonl --append

  # or from measured samples directly
  python3 scripts/perf_gate.py --samples 82.1,81.4,82.6,81.9,82.0 ...

Exit codes: 0 pass or record-only, 1 regression, 2 unstable environment, 3 usage error.
"""

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

MEASURED_RUNS = 5
WARMUP_RUNS = 3

PASS = "pass"
RECORD_ONLY = "record-only"
REGRESSION = "regression"
UNSTABLE = "unstable"

EXIT_CODES = {PASS: 0, RECORD_ONLY: 0, REGRESSION: 1, UNSTABLE: 2}

TUPLE_KEYS = ("machine", "gpu", "model", "quantization", "backend", "configuration",
              "tornadovm_version")

DEFAULT_TOLERANCE = 0.03
DEFAULT_SPREAD_LIMIT = 0.10


class GateError(Exception):
    """Usage or environment problem — never a verdict about performance."""


# ── tolerances ────────────────────────────────────────────────────────────────

def load_tolerances(path):
    if path is None:
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except OSError as e:
        raise GateError(f"tolerances file unreadable: {e}") from e
    except json.JSONDecodeError as e:
        raise GateError(f"tolerances file is not valid JSON: {e}") from e


def resolve_policy(tolerances, tup):
    """Most specific rule wins: a matching tuple rule, then the machine, then the default.

    A rule's ``match`` is a subset of the tuple; ``mode`` is "gate" or "record-only".
    """
    policy = {
        "tolerance": tolerances.get("default", {}).get("tolerance", DEFAULT_TOLERANCE),
        "mode": tolerances.get("default", {}).get("mode", "gate"),
        "spread_limit": tolerances.get("unstable_spread_limit", DEFAULT_SPREAD_LIMIT),
        "source": "default",
    }

    machine_rule = tolerances.get("machines", {}).get(tup.get("machine"))
    if machine_rule:
        policy.update({k: v for k, v in machine_rule.items() if k in ("tolerance", "mode")})
        policy["source"] = "machine:" + str(tup.get("machine"))

    best_match = None
    best_size = -1
    for rule in tolerances.get("tuples", []):
        match = rule.get("match", {})
        if not match:
            continue
        if all(tup.get(k) == v for k, v in match.items()) and len(match) > best_size:
            best_match, best_size = rule, len(match)
    if best_match:
        policy.update({k: v for k, v in best_match.items() if k in ("tolerance", "mode")})
        policy["source"] = "tuple:" + ",".join(f"{k}={v}" for k, v in sorted(best_match["match"].items()))

    return policy


# ── samples ───────────────────────────────────────────────────────────────────

def samples_from_metrics_dir(metrics_dir, pattern="*.json"):
    """eval_rate of every engine metrics file in a directory, sorted by file name.

    Files without a numeric eval_rate are skipped — a warm-up or a failed run leaves a
    file behind and must not be mistaken for a measurement.
    """
    directory = Path(metrics_dir)
    if not directory.is_dir():
        raise GateError(f"metrics directory not found: {metrics_dir}")
    rates = []
    for path in sorted(directory.glob(pattern)):
        if path.name.endswith(".meta.json"):
            continue
        try:
            with open(path) as f:
                metrics = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        rate = metrics.get("eval_rate") if isinstance(metrics, dict) else None
        if isinstance(rate, (int, float)) and rate > 0:
            rates.append(float(rate))
    if not rates:
        raise GateError(f"no usable eval_rate found under {metrics_dir}")
    return rates


def aggregate(samples):
    """Median and relative spread. The median is the metric; the spread decides stability."""
    if not samples:
        raise GateError("no samples")
    median = statistics.median(samples)
    if median <= 0:
        raise GateError(f"non-positive median eval_rate: {median}")
    spread = (max(samples) - min(samples)) / median
    return median, spread


# ── history ───────────────────────────────────────────────────────────────────

def read_history(path):
    history_path = Path(path)
    if not history_path.exists():
        return []
    rows = []
    with open(history_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def tuple_of(row):
    return {key: row.get(key) for key in TUPLE_KEYS}


def find_baseline(history, tup, cache_warm):
    """The most recent gate-passing entry of the same tuple and cache state.

    Entries written before this gate existed carry no ``gate`` object and no
    ``machine``/``gpu``/``tornadovm_version``; they cannot match a tuple and so are
    never used as a baseline. That is deliberate — see C5 in tornadovm-capabilities.md.
    """
    for row in reversed(history):
        if tuple_of(row) != tup:
            continue
        if row.get("cache_warm") != cache_warm:
            continue
        gate = row.get("gate")
        if not isinstance(gate, dict) or gate.get("status") not in (PASS, RECORD_ONLY):
            continue
        if not isinstance(row.get("eval_rate"), (int, float)):
            continue
        return row
    return None


# ── the gate ──────────────────────────────────────────────────────────────────

def evaluate(samples, tup, cache_warm, history, tolerances, expected_runs=MEASURED_RUNS):
    """Return (status, detail). Pure: no file is written and nothing is printed."""
    median, spread = aggregate(samples)
    policy = resolve_policy(tolerances, tup)
    detail = {
        "eval_rate": median,
        "samples": list(samples),
        "spread": spread,
        "policy": policy,
        "baseline": None,
        "expected_runs": expected_runs,
        "notes": [],
    }

    if len(samples) != expected_runs:
        detail["notes"].append(
            f"{len(samples)} measured runs, the procedure states {expected_runs}")

    if spread > policy["spread_limit"]:
        detail["notes"].append(
            f"spread {spread:.1%} exceeds {policy['spread_limit']:.0%} — unstable environment,"
            " no verdict and no new baseline")
        return UNSTABLE, detail

    if policy["mode"] == "record-only":
        detail["notes"].append(f"tuple is record-only by policy ({policy['source']})")
        return RECORD_ONLY, detail

    if not cache_warm:
        detail["notes"].append("cold cubin cache — throughput is compared warm-to-warm only")
        return RECORD_ONLY, detail

    baseline = find_baseline(history, tup, cache_warm)
    if baseline is None:
        detail["notes"].append("no baseline for this tuple — recording, this entry becomes it")
        return RECORD_ONLY, detail

    baseline_rate = float(baseline["eval_rate"])
    floor = baseline_rate * (1.0 - policy["tolerance"])
    detail["baseline"] = {
        "eval_rate": baseline_rate,
        "commit": baseline.get("short_commit") or baseline.get("commit"),
        "timestamp": baseline.get("timestamp"),
        "floor": floor,
        "delta": (median - baseline_rate) / baseline_rate,
    }
    if median < floor:
        return REGRESSION, detail
    return PASS, detail


def evaluate_paired(baseline_samples, candidate_samples, tup, tolerances, baseline_label=""):
    """Compare two builds measured in one session, interleaved.

    Both sides are aggregated the same way and both must be stable; the verdict is on the
    candidate's median against the baseline's median, with the same per-tuple tolerance.
    Nothing is read from, and nothing needs to exist in, the history.
    """
    baseline_median, baseline_spread = aggregate(baseline_samples)
    median, spread = aggregate(candidate_samples)
    policy = resolve_policy(tolerances, tup)

    detail = {
        "eval_rate": median,
        "samples": list(candidate_samples),
        "spread": spread,
        "policy": policy,
        "baseline": {
            "eval_rate": baseline_median,
            "commit": baseline_label,
            "timestamp": None,
            "floor": baseline_median * (1.0 - policy["tolerance"]),
            "delta": (median - baseline_median) / baseline_median,
            "samples": list(baseline_samples),
            "spread": baseline_spread,
            "paired": True,
        },
        "expected_runs": len(candidate_samples),
        "notes": ["paired A/B measured in one session against " + (baseline_label or "the baseline build")],
    }

    if len(baseline_samples) != len(candidate_samples):
        detail["notes"].append(
            f"unequal run counts ({len(baseline_samples)} vs {len(candidate_samples)}) —"
            " interleaving no longer cancels drift")

    worst_spread = max(spread, baseline_spread)
    if worst_spread > policy["spread_limit"]:
        detail["notes"].append(
            f"spread {worst_spread:.1%} exceeds {policy['spread_limit']:.0%} on one side —"
            " unstable environment, no verdict")
        return UNSTABLE, detail

    if median < detail["baseline"]["floor"]:
        return REGRESSION, detail
    return PASS, detail


def build_record(tup, cache_warm, status, detail, args):
    """The history row. Flat fields stay compatible with the existing schema."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": args.commit,
        "short_commit": (args.commit or "")[:8],
        "branch": args.branch,
        "run_id": args.run_id,
        "run_number": args.run_number,
        "run_attempt": args.run_attempt,
        "workflow": args.workflow,
    }
    record.update(tup)
    record["cache_warm"] = cache_warm
    record["eval_rate"] = detail["eval_rate"]
    record["eval_rate_samples"] = detail["samples"]
    record["eval_rate_spread"] = detail["spread"]
    record["warmup_runs"] = args.warmup_runs
    record["gate"] = {
        "status": status,
        "comparison": "paired" if (detail["baseline"] or {}).get("paired") else "history",
        "tolerance": detail["policy"]["tolerance"],
        "policy_source": detail["policy"]["source"],
        "spread_limit": detail["policy"]["spread_limit"],
        "baseline": detail["baseline"],
        "notes": detail["notes"],
    }
    return record


def append_record(history_path, record):
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def format_report(tup, status, detail):
    lines = [
        f"tuple      {' / '.join(str(tup[k]) for k in TUPLE_KEYS)}",
        f"eval_rate  {detail['eval_rate']:.2f} tok/s"
        f"  (median of {len(detail['samples'])}, spread {detail['spread']:.1%})",
    ]
    baseline = detail["baseline"]
    if baseline:
        origin = baseline["commit"] or ("the baseline build" if baseline.get("paired") else "?")
        spread = f"  (spread {baseline['spread']:.1%})" if baseline.get("paired") else ""
        lines.append(
            f"baseline   {baseline['eval_rate']:.2f} tok/s @ {origin}{spread}"
            f"  floor {baseline['floor']:.2f}  delta {baseline['delta']:+.1%}"
            f"  tolerance {detail['policy']['tolerance']:.1%}")
    for note in detail["notes"]:
        lines.append(f"note       {note}")
    lines.append(f"verdict    {status.upper()}")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_bool(value):
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in ("true", "1", "yes", "warm"):
        return True
    if str(value).strip().lower() in ("false", "0", "no", "cold"):
        return False
    raise GateError(f"expected a boolean, got {value!r}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--metrics-dir", dest="metrics_dir",
                        help="Directory of engine metrics JSON files (measured runs only)")
    source.add_argument("--samples", help="Comma-separated eval_rate values, one per measured run")

    p.add_argument("--metrics-glob", dest="metrics_glob", default="*.json",
                   help="Which files in --metrics-dir are measured runs (default: *.json)")

    baseline = p.add_mutually_exclusive_group()
    baseline.add_argument("--baseline-samples", dest="baseline_samples",
                          help="Paired mode: the baseline build's eval_rate values, measured in"
                               " the same session as the candidate")
    baseline.add_argument("--baseline-metrics-dir", dest="baseline_metrics_dir",
                          help="Paired mode: directory of the baseline build's metrics files")
    p.add_argument("--baseline-label", dest="baseline_label", default="",
                   help="What the baseline build is, e.g. a git ref (paired mode)")
    for key in TUPLE_KEYS:
        p.add_argument("--" + key.replace("_", "-"), dest=key, required=True)
    p.add_argument("--cache-warm", dest="cache_warm", required=True,
                   help="true when the on-disk cubin cache was warm for the measured runs")

    p.add_argument("--history", default="docs/perf-history.jsonl")
    p.add_argument("--tolerances", default="scripts/perf-gate-tolerances.json")
    p.add_argument("--append", action="store_true",
                   help="Append the candidate to the history (never on an unstable environment)")
    p.add_argument("--json", dest="as_json", action="store_true",
                   help="Print the record instead of the human report")

    p.add_argument("--commit", default="")
    p.add_argument("--branch", default="")
    p.add_argument("--run-id", dest="run_id", default="")
    p.add_argument("--run-number", dest="run_number", default="")
    p.add_argument("--run-attempt", dest="run_attempt", default="1")
    p.add_argument("--workflow", default="local")
    p.add_argument("--warmup-runs", dest="warmup_runs", type=int, default=WARMUP_RUNS)
    p.add_argument("--expected-runs", dest="expected_runs", type=int, default=MEASURED_RUNS)
    return p.parse_args(argv)


def main(argv=None):
    try:
        args = parse_args(argv)
        if args.samples:
            samples = [float(s) for s in args.samples.split(",") if s.strip()]
        else:
            samples = samples_from_metrics_dir(args.metrics_dir, args.metrics_glob)

        tup = {key: getattr(args, key) for key in TUPLE_KEYS}
        cache_warm = parse_bool(args.cache_warm)
        tolerances = load_tolerances(args.tolerances)

        if args.baseline_samples or args.baseline_metrics_dir:
            if args.baseline_samples:
                baseline_samples = [float(s) for s in args.baseline_samples.split(",") if s.strip()]
            else:
                baseline_samples = samples_from_metrics_dir(args.baseline_metrics_dir, args.metrics_glob)
            status, detail = evaluate_paired(baseline_samples, samples, tup, tolerances,
                                             args.baseline_label)
        else:
            history = read_history(args.history)
            status, detail = evaluate(samples, tup, cache_warm, history, tolerances,
                                      expected_runs=args.expected_runs)
        record = build_record(tup, cache_warm, status, detail, args)

        if args.append and status != UNSTABLE:
            append_record(args.history, record)

        print(json.dumps(record) if args.as_json else format_report(tup, status, detail))
        return EXIT_CODES[status]
    except GateError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
