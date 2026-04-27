from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from common import resolve_dynaworld_path


BASE_CONFIG = (
    "src/train_configs/"
    "local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_rank_adaptive_metric_128_16f_2048el.jsonc"
)

SPLAT_BASE_CONFIG = (
    "src/train_configs/"
    "local_mac_splat_baseline_multicam_deepview_3cam_train2_test1_free_dynamic_3dgs_128_16f_2048splats.jsonc"
)

STATIC_SPLAT_CONFIG = (
    "src/train_configs/"
    "local_mac_splat_baseline_multicam_deepview_3cam_train2_test1_static_3dgs_128_16f_2048splats.jsonc"
)


RUNS = [
    {
        "name": "free_dynamic_3dgs",
        "script": "research_experiments/gauge_fields/train_splat_baseline.py",
        "config": SPLAT_BASE_CONFIG,
        "extra_args": [],
    },
    {
        "name": "static_3dgs",
        "script": "research_experiments/gauge_fields/train_splat_baseline.py",
        "config": STATIC_SPLAT_CONFIG,
        "extra_args": [],
    },
    {
        "name": "screen_disk_projected_conic",
        "script": "research_experiments/gauge_fields/train.py",
        "config": BASE_CONFIG,
        "extra_args": ["--support-mode", "screen_disk"],
    },
    {
        "name": "rank_adaptive_metric_projected_conic",
        "script": "research_experiments/gauge_fields/train.py",
        "config": BASE_CONFIG,
        "extra_args": [],
    },
    {
        "name": "rank_adaptive_metric_multiview_init_pair_x",
        "script": "research_experiments/gauge_fields/train.py",
        "config": (
            "src/train_configs/"
            "local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_rank_adaptive_metric_multiview_init_pair_x_128_16f_2048el.jsonc"
        ),
        "extra_args": [],
    },
    {
        "name": "derived_support_metric_multiview_init_pair_x",
        "script": "research_experiments/gauge_fields/train.py",
        "config": (
            "src/train_configs/"
            "local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_derived_support_metric_multiview_init_pair_x_128_16f_2048el.jsonc"
        ),
        "extra_args": [],
    },
    {
        "name": "screen_disk_2048_multiview_init_pair_x",
        "script": "research_experiments/gauge_fields/train.py",
        "config": (
            "src/train_configs/"
            "local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_screen_disk_multiview_init_pair_x_128_16f_2048el.jsonc"
        ),
        "extra_args": [],
    },
    {
        "name": "transported_world_ball_2048_multiview_init_pair_x",
        "script": "research_experiments/gauge_fields/train.py",
        "config": (
            "src/train_configs/"
            "local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_transported_world_ball_multiview_init_pair_x_128_16f_2048el.jsonc"
        ),
        "extra_args": [],
    },
    {
        "name": "rank_adaptive_metric_2048_multiview_init_delayed_pair_x",
        "script": "research_experiments/gauge_fields/train.py",
        "config": (
            "src/train_configs/"
            "local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_rank_adaptive_metric_multiview_init_pair_x_delayed_128_16f_2048el.jsonc"
        ),
        "extra_args": [],
    },
    {
        "name": "screen_disk_8192_multiview_init_pair_x",
        "script": "research_experiments/gauge_fields/train.py",
        "config": (
            "src/train_configs/"
            "local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_screen_disk_multiview_init_pair_x_128_16f_8192el.jsonc"
        ),
        "extra_args": [],
    },
    {
        "name": "transported_world_ball_8192_multiview_init_pair_x",
        "script": "research_experiments/gauge_fields/train.py",
        "config": (
            "src/train_configs/"
            "local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_transported_world_ball_multiview_init_pair_x_128_16f_8192el.jsonc"
        ),
        "extra_args": [],
    },
    {
        "name": "rank_adaptive_metric_7516_multiview_init_pair_x",
        "script": "research_experiments/gauge_fields/train.py",
        "config": (
            "src/train_configs/"
            "local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_rank_adaptive_metric_multiview_init_pair_x_128_16f_7516el.jsonc"
        ),
        "extra_args": [],
    },
    {
        "name": "rank_adaptive_metric_ray_gaussian_line_mass_candidate_tiled",
        "script": "research_experiments/gauge_fields/train.py",
        "config": BASE_CONFIG,
        "extra_args": [
            "--incidence-mode",
            "ray_gaussian_line_mass",
            "--line-candidate-mode",
            "projected_bbox",
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepView train-2-cameras/test-1-camera gauge matrix.")
    parser.add_argument(
        "--output-root",
        default="outputs/gauge_fields/multicam_deepview_3cam_train2_test1_80step",
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
            *spec.get("extra_args", []),
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
