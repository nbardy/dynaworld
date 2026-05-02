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
VARIANTS_ROOT = ROOT / "third_party" / "fast-mac-gsplat" / "variants"


@dataclass(frozen=True)
class VariantSpec:
    name: str
    bench: str
    kind: str = "benchmark_mps"
    extra_args: tuple[str, ...] = ()


VARIANTS: dict[str, VariantSpec] = {
    "v5": VariantSpec("v5", "benchmarks/benchmark_mps.py"),
    "v5_features": VariantSpec("v5_features", "benchmarks/benchmark_mps.py", extra_args=("--feature-dim", "3")),
    "v6": VariantSpec("v6", "benchmarks/benchmark_mps.py"),
    "v6_upgrade": VariantSpec("v6_upgrade", "benchmarks/benchmark_mps.py"),
    "v6_refined": VariantSpec("v6_refined", "benchmarks/benchmark_mps.py"),
    "v8": VariantSpec("v8", "benchmarks/benchmark_mps.py"),
    "v8_hw_eval": VariantSpec("v8_hw_eval", "benchmarks/benchmark_mps.py"),
    "v8_hw_train": VariantSpec("v8_hw_train", "benchmarks/benchmark_mps.py"),
    "v8_project3d": VariantSpec("v8_project3d", "benchmarks/benchmark_mps.py"),
    "v9_project3d_train": VariantSpec("v9_project3d_train", "benchmarks/benchmark_mps.py"),
    "v9_hw_tile_exact_probe": VariantSpec(
        "v9_hw_tile_exact_probe",
        "benchmarks/benchmark_full_backward.py",
        kind="v9_full_backward",
    ),
}


def parse_csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if min(values) < 1:
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def parse_csv_strings(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    unknown = [item for item in values if item not in VARIANTS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown variants: {', '.join(unknown)}")
    return values


def json_line(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise ValueError("no JSON object found in command output")


def command_for_case(
    spec: VariantSpec,
    *,
    resolution: int,
    gaussians: int,
    batch_size: int,
    case: str,
    seed: int,
    warmup: int,
    iters: int,
) -> list[str]:
    if spec.kind == "v9_full_backward":
        return [
            sys.executable,
            spec.bench,
            "--height",
            str(resolution),
            "--width",
            str(resolution),
            "--gaussians",
            str(gaussians),
            "--batch-size",
            str(batch_size),
            "--seed",
            str(seed),
            "--warmup",
            str(warmup),
            "--iters",
            str(iters),
        ]
    return [
        sys.executable,
        spec.bench,
        "--height",
        str(resolution),
        "--width",
        str(resolution),
        "--gaussians",
        str(gaussians),
        "--batch-size",
        str(batch_size),
        "--case",
        case,
        "--seed",
        str(seed),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--backward",
        "--profile",
        "--json",
        *spec.extra_args,
    ]


def run_case(
    spec: VariantSpec,
    *,
    resolution: int,
    gaussians: int,
    batch_size: int,
    case: str,
    seed: int,
    warmup: int,
    iters: int,
    timeout: float,
) -> dict[str, Any]:
    variant_dir = VARIANTS_ROOT / spec.name
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env.setdefault("GSP_TILE_SIZE", "16")
    env.setdefault("GSP_CHUNK", "64")
    env.setdefault("GSP_FAST_CAP", "2048")
    cmd = command_for_case(
        spec,
        resolution=resolution,
        gaussians=gaussians,
        batch_size=batch_size,
        case=case,
        seed=seed,
        warmup=warmup,
        iters=iters,
    )
    base = {
        "variant": spec.name,
        "variant_kind": spec.kind,
        "resolution": int(resolution),
        "gaussians_requested": int(gaussians),
        "batch_size_requested": int(batch_size),
        "case_requested": case,
        "seed_requested": int(seed),
        "warmup_requested": int(warmup),
        "iters_requested": int(iters),
        "cmd": " ".join(cmd),
    }
    try:
        proc = subprocess.run(
            cmd,
            cwd=variant_dir,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            **base,
            "status": "timeout",
            "returncode": None,
            "stdout_tail": (exc.stdout or "")[-2000:],
            "stderr_tail": (exc.stderr or "")[-2000:],
        }
    if proc.returncode != 0:
        return {
            **base,
            "status": "error",
            "returncode": int(proc.returncode),
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
    try:
        row = json_line(proc.stdout)
    except Exception as exc:  # noqa: BLE001 - recorded as benchmark failure.
        return {
            **base,
            "status": "parse_error",
            "returncode": int(proc.returncode),
            "error": f"{type(exc).__name__}: {exc}",
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
    row.update(base)
    row["status"] = "ok"
    row["returncode"] = int(proc.returncode)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed high-res fast-mac variant timing matrix.")
    parser.add_argument(
        "--variants",
        type=parse_csv_strings,
        default=parse_csv_strings(",".join(VARIANTS)),
    )
    parser.add_argument("--resolutions", type=parse_csv_ints, default=parse_csv_ints("512,2048,4096"))
    parser.add_argument("--gaussians", type=parse_csv_ints, default=parse_csv_ints("8192,65536"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--case", type=str, default="medium_sigma_3_8")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=420.0)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for variant_name in args.variants:
            spec = VARIANTS[variant_name]
            for resolution in args.resolutions:
                for gaussians in args.gaussians:
                    row = run_case(
                        spec,
                        resolution=resolution,
                        gaussians=gaussians,
                        batch_size=args.batch_size,
                        case=args.case,
                        seed=args.seed,
                        warmup=args.warmup,
                        iters=args.iters,
                        timeout=args.timeout,
                    )
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                    if row["status"] == "ok":
                        print(
                            f"{variant_name:24s} r={resolution:4d} g={gaussians:6d} "
                            f"median={row.get('median_ms', row.get('total_median_ms', 0.0)):9.3f} "
                            f"fwd={row.get('forward_ms', row.get('forward_median_ms', 0.0)):9.3f} "
                            f"bwd={row.get('backward_ms', row.get('backward_median_ms', 0.0)):9.3f}"
                        )
                    else:
                        print(f"{variant_name:24s} r={resolution:4d} g={gaussians:6d} status={row['status']}")
    print(f"wrote {args.output_jsonl}")


if __name__ == "__main__":
    main()
