"""CLI for the deterministic connection-v2 reference oracle."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any

from .oracle import run_reference_oracle


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the CPU float64 constrained-Lagrangian optical-connection "
            "oracle. This is not a training or performance benchmark."
        )
    )
    parser.add_argument("--probe-count", type=int, default=65)
    parser.add_argument("--maximum-atlas-nodes", type=int)
    parser.add_argument("--primal-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--secant-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-reference-gates",
        action="store_true",
        help="return exit status 2 when any bounded reference gate fails",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = run_reference_oracle(
        probe_count=args.probe_count,
        maximum_atlas_nodes=args.maximum_atlas_nodes,
        primal_tolerance=args.primal_tolerance,
        secant_tolerance=args.secant_tolerance,
    )
    payload = _json_ready(report)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if args.output is None:
        print(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(args.output)
    if args.require_reference_gates and not report.all_reference_correctness_gates_passed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
