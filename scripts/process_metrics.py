#!/usr/bin/env python3
"""
Append benchmark JSON metrics to the performance history JSONL file.

Scans --metrics-dir recursively for *.json files. Each metrics file must have a
companion *.meta.json sidecar (same stem) written by the CI step that produced it.
The sidecar schema is open-ended — this script does not assume any fixed fields.

Usage:
  python3 scripts/process_metrics.py \\
    --metrics-dir /path/to/artifacts \\
    --commit $GITHUB_SHA --branch main \\
    --run-id $GITHUB_RUN_ID --run-number $GITHUB_RUN_NUMBER \\
    --run-attempt 1 --workflow "GPULlama3 Build & Run" \\
    --history docs/perf-history.jsonl
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics-dir", required=True, dest="metrics_dir",
                   help="Directory to search recursively for *.json + *.meta.json pairs")
    p.add_argument("--history",     required=True,
                   help="JSONL history file to append rows to")
    p.add_argument("--commit",      required=True)
    p.add_argument("--branch",      required=True)
    p.add_argument("--run-id",      required=True, dest="run_id")
    p.add_argument("--run-number",  default="",    dest="run_number")
    p.add_argument("--run-attempt", required=True, dest="run_attempt")
    p.add_argument("--workflow",    required=True)
    return p.parse_args()


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"ERROR: {path}: {e}", file=sys.stderr)
        return None


def discover_pairs(metrics_dir):
    """Yield (metrics_path, meta_path) for every non-sidecar JSON found recursively."""
    for path in sorted(Path(metrics_dir).rglob("*.json")):
        if path.name.endswith(".meta.json"):
            continue
        yield path, path.with_suffix(".meta.json")


def build_row(m, meta, args):
    return {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "commit":       args.commit,
        "short_commit": args.commit[:8],
        "branch":       args.branch,
        "run_id":       args.run_id,
        "run_number":   args.run_number or "",
        "run_attempt":  args.run_attempt,
        "workflow":     args.workflow,
        # Flat compat fields — sourced from sidecar; null when absent
        "backend":              meta.get("backend"),
        "model":                meta.get("model"),
        "quantization":         meta.get("quantization"),
        "configuration":        meta.get("configuration"),
        # Key metrics promoted to top level — null when absent in the metrics file
        "eval_rate":            m.get("eval_rate"),
        "prompt_eval_rate":     m.get("prompt_eval_rate"),
        "total_rate":           m.get("total_rate"),
        "eval_count":           m.get("eval_count"),
        "prompt_eval_count":    m.get("prompt_eval_count"),
        "total_count":          m.get("total_count"),
        "total_duration":       m.get("total_duration"),
        "load_duration":        m.get("load_duration"),
        "prompt_eval_duration": m.get("prompt_eval_duration"),
        "eval_duration":        m.get("eval_duration"),
        "has_prefill_phase":    m.get("has_prefill_phase"),
        "tornadovm":            m.get("tornadovm"),
        # Nested full objects — open-ended; schema is whatever the benchmark step writes
        "benchmark": meta,
        "metrics":   m,
    }


def main():
    args = parse_args()
    rows = []

    for metrics_path, meta_path in discover_pairs(args.metrics_dir):
        m = load_json(metrics_path)
        if not isinstance(m, dict):
            print(f"WARNING: {metrics_path.name}: not a JSON object, skipping", file=sys.stderr)
            continue

        if not meta_path.exists():
            print(f"WARNING: no sidecar for {metrics_path.name}, skipping", file=sys.stderr)
            continue
        meta = load_json(meta_path)
        if not isinstance(meta, dict):
            print(f"WARNING: {meta_path.name}: not a JSON object, skipping", file=sys.stderr)
            continue

        rows.append(build_row(m, meta, args))
        label = " / ".join(filter(None, [
            meta.get("backend"),
            meta.get("model"),
            meta.get("quantization"),
            meta.get("configuration") or meta.get("task"),
        ]))
        print(f"  {label or metrics_path.name}", file=sys.stderr)

    if not rows:
        print("WARNING: no metrics loaded, nothing written", file=sys.stderr)
        return

    history = Path(args.history)
    history.parent.mkdir(parents=True, exist_ok=True)
    with open(history, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Appended {len(rows)} row(s) to {history}", file=sys.stderr)


if __name__ == "__main__":
    main()
