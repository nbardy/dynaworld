from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from common import resolve_dynaworld_path


RUNS = [
    {
        "name": "free_dynamic_3dgs",
        "script": "research_experiments/gauge_fields/train_splat_baseline.py",
        "config": "src/train_configs/local_mac_splat_baseline_multicam_deepview_free_dynamic_3dgs_128_16f_2048splats.jsonc",
    },
    {
        "name": "screen_disk_projected_conic",
        "script": "research_experiments/gauge_fields/train.py",
        "config": "src/train_configs/local_mac_gauge_fields_multicam_deepview_screen_disk_128_16f_2048el.jsonc",
    },
    {
        "name": "rank_adaptive_metric_projected_conic",
        "script": "research_experiments/gauge_fields/train.py",
        "config": "src/train_configs/local_mac_gauge_fields_multicam_deepview_rank_adaptive_metric_128_16f_2048el.jsonc",
    },
    {
        "name": "rank_adaptive_metric_ray_gaussian_line_mass",
        "script": "research_experiments/gauge_fields/train.py",
        "config": "src/train_configs/local_mac_gauge_fields_multicam_deepview_rank_adaptive_metric_ray_gaussian_line_mass_128_16f_2048el.jsonc",
    },
    {
        "name": "rank_adaptive_metric_ray_gaussian_line_peak",
        "script": "research_experiments/gauge_fields/train.py",
        "config": "src/train_configs/local_mac_gauge_fields_multicam_deepview_rank_adaptive_metric_ray_gaussian_line_peak_128_16f_2048el.jsonc",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DeepView incidence-mode benchmark matrix.")
    parser.add_argument(
        "--output-root",
        default="outputs/gauge_fields/multicam_deepview_incidence_matrix_80step",
    )
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--only", default=None, help="Comma-separated run names to execute.")
    return parser.parse_args()


def write_wall_clock(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    output_root = resolve_dynaworld_path(args.output_root)
    selected = set(args.only.split(",")) if args.only else None

    for spec in RUNS:
        if selected is not None and spec["name"] not in selected:
            continue

        output_dir = output_root / spec["name"]
        cmd = [
            "uv",
            "run",
            "python",
            spec["script"],
            spec["config"],
            "--device",
            args.device,
            "--steps",
            str(args.steps),
            "--output-dir",
            str(output_dir),
        ]
        if args.no_wandb:
            cmd.append("--no-wandb")

        print(f"\n==> Running {spec['name']}", flush=True)
        print(" ".join(cmd), flush=True)
        start = time.perf_counter()
        status = "completed"
        returncode = 0
        try:
            subprocess.run(cmd, cwd=resolve_dynaworld_path("."), check=True)
        except subprocess.CalledProcessError as exc:
            status = "failed"
            returncode = int(exc.returncode)
        elapsed = time.perf_counter() - start
        write_wall_clock(
            output_dir / "wall_clock.json",
            {
                "name": spec["name"],
                "status": status,
                "returncode": returncode,
                "elapsed_sec": elapsed,
                "elapsed_min": elapsed / 60.0,
                "steps": args.steps,
                "device": args.device,
                "command": cmd,
            },
        )
        print(f"<== {spec['name']} {status} in {elapsed:.2f}s", flush=True)
        if returncode != 0:
            sys.exit(returncode)


if __name__ == "__main__":
    main()
