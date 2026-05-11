from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv" / "bin" / "python"
VARIANTS = ROOT / "third_party" / "fast-mac-gsplat" / "variants"


@dataclass(frozen=True)
class IterationSpec:
    name: str
    kind: str
    variant: str
    description: str
    extra_args: tuple[str, ...] = ()


ITERATIONS: tuple[IterationSpec, ...] = (
    IterationSpec(
        name="v13d_v11_serial_batch",
        kind="raster",
        variant="v11_features_gradcache_zero_bg_hostmeta_fixedbin",
        description="Use existing serial batch strategy to avoid one large flattened launch.",
        extra_args=("--batch-strategy", "serial"),
    ),
    IterationSpec(
        name="v13e_v11_active_auto",
        kind="raster",
        variant="v11_features_gradcache_zero_bg_hostmeta_fixedbin",
        description="Let v11 active-tile policy choose sparse active scheduling.",
        extra_args=("--active-policy", "auto", "--batch-strategy", "flatten"),
    ),
    IterationSpec(
        name="v13f_v11_active_on",
        kind="raster",
        variant="v11_features_gradcache_zero_bg_hostmeta_fixedbin",
        description="Force active-tile scheduling even when the tile grid is dense.",
        extra_args=("--active-policy", "on", "--batch-strategy", "flatten"),
    ),
    IterationSpec(
        name="v13g_v11_frozen_features",
        kind="raster",
        variant="v11_features_gradcache_zero_bg_hostmeta_fixedbin",
        description="Skip feature/color gradient allocation to bound head-only or frozen-feature follow-ups.",
        extra_args=("--freeze-colors", "--batch-strategy", "flatten"),
    ),
    IterationSpec(
        name="v13h_v13a_recompute_state",
        kind="raster",
        variant="v13a_temporal_recompute_state",
        description="Exact recompute of saved tile state during backward.",
        extra_args=("--backward-state-strategy", "recompute", "--batch-strategy", "flatten"),
    ),
    IterationSpec(
        name="v13i_v13b_renamed_baseline",
        kind="raster",
        variant="v13b_rgb_grad_handoff",
        description="Renamed v11-compatible path that carries the RGB-gradient handoff scaffold.",
        extra_args=("--batch-strategy", "flatten"),
    ),
    IterationSpec(
        name="v13j_rgb_grad_accounting",
        kind="accounting",
        variant="v13b_rgb_grad_handoff",
        description="Dense backward-input accounting for the future RGB-gradient handoff kernel.",
    ),
)


def _json_from_stdout(stdout: str) -> dict[str, Any]:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"command did not emit a JSON object; stdout was:\n{stdout}")


def _run_command(cmd: list[str], *, env: dict[str, str]) -> tuple[dict[str, Any], str]:
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed with exit code "
            f"{completed.returncode}:\n{' '.join(cmd)}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return _json_from_stdout(completed.stdout), completed.stderr


def _base_env(args: argparse.Namespace, variant_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["GSP_FAST_CAP"] = str(args.fast_cap)
    env["GSP_FEATURE_CAP"] = str(args.feature_cap)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(variant_dir)
    return env


def run_raster_iteration(spec: IterationSpec, args: argparse.Namespace) -> dict[str, Any]:
    variant_dir = VARIANTS / spec.variant
    benchmark = variant_dir / "benchmarks" / "benchmark_mps.py"
    cmd = [
        str(PYTHON),
        str(benchmark),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--gaussians",
        str(args.gaussians),
        "--batch-size",
        str(args.batch_size),
        "--feature-dim",
        str(args.feature_dim),
        "--case",
        args.case,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--backward",
        "--alpha-loss",
        "--json",
        *spec.extra_args,
    ]
    payload, _stderr = _run_command(cmd, env=_base_env(args, variant_dir))
    payload["iteration"] = spec.name
    payload["variant"] = spec.variant
    payload["description"] = spec.description
    payload["kind"] = spec.kind
    return payload


def run_accounting_iteration(spec: IterationSpec, args: argparse.Namespace) -> dict[str, Any]:
    variant_dir = VARIANTS / spec.variant
    benchmark = variant_dir / "benchmarks" / "rgb_grad_handoff_accounting.py"
    cmd = [
        str(PYTHON),
        str(benchmark),
        "--batch",
        str(args.accounting_batch_size),
        "--height",
        str(args.accounting_height),
        "--width",
        str(args.accounting_width),
        "--feature-dim",
        str(args.feature_dim),
    ]
    payload, _stderr = _run_command(cmd, env=_base_env(args, variant_dir))
    payload["iteration"] = spec.name
    payload["variant"] = spec.variant
    payload["description"] = spec.description
    payload["kind"] = spec.kind
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run named v13d-v13j fast-mac multiframe raster iteration probes."
    )
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--gaussians", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--case", type=str, default="medium_sigma_3_8")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--fast-cap", type=int, default=4096)
    parser.add_argument("--feature-cap", type=int, default=64)
    parser.add_argument("--accounting-batch-size", type=int, default=16)
    parser.add_argument("--accounting-height", type=int, default=512)
    parser.add_argument("--accounting-width", type=int, default=512)
    parser.add_argument(
        "--versions",
        type=str,
        default=",".join(spec.name for spec in ITERATIONS),
        help="Comma-separated iteration names to run.",
    )
    parser.add_argument("--json", action="store_true", help="Print one JSON payload only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    by_name = {spec.name: spec for spec in ITERATIONS}
    requested = [name.strip() for name in args.versions.split(",") if name.strip()]
    unknown = sorted(set(requested) - set(by_name))
    if unknown:
        raise ValueError(f"Unknown iterations: {unknown}. Known: {sorted(by_name)}")

    results = []
    for name in requested:
        spec = by_name[name]
        if spec.kind == "raster":
            result = run_raster_iteration(spec, args)
        elif spec.kind == "accounting":
            result = run_accounting_iteration(spec, args)
        else:
            raise ValueError(f"unknown iteration kind: {spec.kind}")
        results.append(result)
        if not args.json:
            if spec.kind == "raster":
                print(
                    "{iteration} variant={variant} median={median_ms:.3f} "
                    "fwd={forward_ms:.3f} bwd={backward_ms:.3f}".format(**result)
                )
            else:
                print(
                    "{iteration} current={current_dense_backward_input_mib:.1f}MiB "
                    "handoff={handoff_dense_backward_input_mib:.1f}MiB "
                    "avoided={avoided_mib:.1f}MiB".format(**result)
                )
    print(json.dumps({"cases": results}, indent=2 if args.json else None, sort_keys=True))


if __name__ == "__main__":
    main()
