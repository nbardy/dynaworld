"""CLI for offline candidate/probe selection reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .agreement import build_selection_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path, help="Candidate/probe matrix JSON.")
    parser.add_argument("--output", type=Path, help="Optional output JSON path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with args.matrix.open("r", encoding="utf-8") as handle:
        matrix = json.load(handle)
    report = build_selection_report(matrix)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
