#!/usr/bin/env python3
"""
Process benchmark JSON metrics: print a Markdown summary to stdout and, when
--history is supplied, append one JSONL row per configuration to the history file.

Workflow usage (single step, one script call):

  python3 scripts/process_metrics.py \\
    --metrics metrics-standard.json:standard \\
    --metrics metrics-prefill-decode.json:prefill-decode \\
    --backend ptx \\
    --model   Llama-3.2-1B-Instruct \\
    --quantization F16 \\
    --commit $GITHUB_SHA --branch main \\
    --run-id $GITHUB_RUN_ID --run-attempt 1 \\
    --workflow "GPULlama3 Build & Run" \\
    --history docs/perf-history.jsonl \\
    >> $GITHUB_STEP_SUMMARY
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_FIELDS = [
    "total_duration", "load_duration",
    "prompt_eval_count", "prompt_eval_duration",
    "eval_count", "eval_duration",
    "total_count", "prompt_eval_rate", "eval_rate", "total_rate",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics", action="append", required=True,
                   metavar="PATH:CONFIGURATION",
                   help="Metrics JSON file and its configuration label "
                        "(e.g. metrics-standard.json:standard). Repeatable.")

    history = p.add_argument_group("history (all required together when --history is given)")
    history.add_argument("--history",       default=None,
                         help="JSONL history file to append rows to")
    history.add_argument("--backend",       default=None)
    history.add_argument("--model",         default=None,
                         help="Model name, e.g. Llama-3.2-1B-Instruct")
    history.add_argument("--quantization",  default=None,
                         help="Quantization, e.g. F16 or Q8_0")
    history.add_argument("--commit",        default=None)
    history.add_argument("--branch",        default=None)
    history.add_argument("--run-id",        default=None, dest="run_id")
    history.add_argument("--run-number",    default=None, dest="run_number")
    history.add_argument("--run-attempt",   default=None, dest="run_attempt")
    history.add_argument("--workflow",      default=None)
    return p.parse_args()


def load_metrics(path):
    if not os.path.exists(path):
        print(f"WARNING: metrics file not found, skipping: {path}", file=sys.stderr)
        return None
    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"ERROR: failed to parse {path}: {e}", file=sys.stderr)
            sys.exit(1)
    missing = [k for k in REQUIRED_FIELDS if k not in data]
    if missing:
        print(f"ERROR: {path} missing required fields: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)
    return data


def build_summary(rows):
    lines = [
        "| configuration | eval tok/s | prompt tok/s | total tok/s"
        " | eval tokens | prompt tokens | total ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['configuration']}"
            f" | {r['eval_rate']:.2f}"
            f" | {r['prompt_eval_rate']:.2f}"
            f" | {r['total_rate']:.2f}"
            f" | {r['eval_count']}"
            f" | {r['prompt_eval_count']}"
            f" | {r['total_duration_ms']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def build_history_row(m, configuration, args):
    return {
        "timestamp":            datetime.now(timezone.utc).isoformat(),
        "commit":               args.commit,
        "short_commit":         args.commit[:8],
        "branch":               args.branch,
        "run_id":               args.run_id,
        "run_number":           args.run_number or "",
        "run_attempt":          args.run_attempt,
        "workflow":             args.workflow,
        "backend":              args.backend,
        "model":                args.model,
        "quantization":         args.quantization,
        "configuration":        configuration,
        "eval_rate":            m["eval_rate"],
        "prompt_eval_rate":     m["prompt_eval_rate"],
        "total_rate":           m["total_rate"],
        "eval_count":           m["eval_count"],
        "prompt_eval_count":    m["prompt_eval_count"],
        "total_count":          m["total_count"],
        "total_duration":       m["total_duration"],
        "load_duration":        m["load_duration"],
        "prompt_eval_duration": m["prompt_eval_duration"],
        "eval_duration":        m["eval_duration"],
    }


def main():
    args = parse_args()

    if args.history and not all([args.backend, args.model, args.quantization,
                                  args.commit, args.branch, args.run_id,
                                  args.run_attempt, args.workflow]):
        print("ERROR: --history requires --backend, --model, --quantization, "
              "--commit, --branch, --run-id, --run-attempt, --workflow", file=sys.stderr)
        sys.exit(1)

    summary_rows = []
    history_rows = []

    for spec in args.metrics:
        parts = spec.split(":", 1)
        if len(parts) != 2:
            print(f"ERROR: --metrics expects 'path:configuration', got '{spec}'", file=sys.stderr)
            sys.exit(1)
        path, configuration = parts
        m = load_metrics(path)
        if m is None:
            continue

        summary_rows.append({
            "configuration":     configuration,
            "eval_rate":         m.get("eval_rate", 0),
            "prompt_eval_rate":  m.get("prompt_eval_rate", 0),
            "total_rate":        m.get("total_rate", 0),
            "eval_count":        m.get("eval_count", 0),
            "prompt_eval_count": m.get("prompt_eval_count", 0),
            "total_duration_ms": m.get("total_duration", 0) / 1_000_000,
        })

        if args.history:
            history_rows.append(build_history_row(m, configuration, args))

    if summary_rows:
        sys.stdout.write(build_summary(summary_rows))

    if history_rows:
        history = Path(args.history)
        history.parent.mkdir(parents=True, exist_ok=True)
        with open(history, "a") as f:
            for row in history_rows:
                f.write(json.dumps(row) + "\n")
        print(f"Appended {len(history_rows)} row(s) to {history}", file=sys.stderr)


if __name__ == "__main__":
    main()
