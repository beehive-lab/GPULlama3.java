#!/usr/bin/env python3
"""
Write a benchmark metadata sidecar JSON file from KEY=VALUE arguments.

Values are stored as strings unless they parse as a valid JSON literal
(true, false, null, or a number), making it easy to pass typed metadata
from shell without quoting gymnastics.

Usage:
  python3 scripts/write_metrics_sidecar.py --out /path/to/file.meta.json \
    backend=opencl \
    task=llama-inference \
    model_file=Llama-3.2-1B-Instruct-F16.gguf \
    configuration=standard \
    flags="" \
    prompt="Say hello"
"""

import argparse
import json
import sys
from pathlib import Path


def coerce(value):
    """Return value as a JSON-native type when unambiguous, otherwise keep as string."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="Output .meta.json path")
    p.add_argument("fields", nargs="*", metavar="KEY=VALUE")
    return p.parse_args()


def main():
    args = parse_args()
    meta = {}
    for field in args.fields:
        if "=" not in field:
            print(f"ERROR: expected KEY=VALUE, got: {field!r}", file=sys.stderr)
            sys.exit(1)
        key, _, value = field.partition("=")
        meta[key] = coerce(value)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(meta, f)
    print(f"Wrote sidecar: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
