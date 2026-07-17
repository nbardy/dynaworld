from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import wandb

from config_utils import load_config_file, serialize_config_value
from powerfoam_metal_trainer import run_training as run_powerfoam_training


ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = (
    ROOT
    / "src"
    / "train_configs"
    / "local_mac_powerfoam_metal_multicam_neural3d_coffee_train2_holdout1_feature_triangulation_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux.jsonc"
)
COMPARE_SCRIPT = (
    ROOT
    / "third_party"
    / "fast-mac-gsplat"
    / "variants"
    / "star_uvt_v0"
    / "research_project"
    / "benchmarks"
    / "multicam_heldout_compare.py"
)
DEFAULT_OUT_DIR = ROOT / "outputs" / "benchmarks" / "2026-07-11_coffee_martini_matched_sweep"
DEFAULT_SEEDS = (17, 29, 43)
TRAIN_CAMERAS = ("cam04", "cam09")
HELDOUT_CAMERAS = ("cam06",)
TARGET_SIZE = 128
FRAME_COUNT = 16
STEPS = 40
PRIMITIVE_COUNT = 1024


def comparison_command(seed: int, out_dir: Path, *, python: str = sys.executable) -> list[str]:
    return [
        python,
        str(COMPARE_SCRIPT),
        "--baseline-config",
        str(BASE_CONFIG),
        "--target-size",
        str(TARGET_SIZE),
        "--max-frames",
        str(FRAME_COUNT),
        "--train-seconds",
        "600",
        "--max-steps",
        str(STEPS),
        "--device",
        "mps",
        "--seed",
        str(seed),
        "--uvt-tubes",
        str(PRIMITIVE_COUNT),
        "--uvt-render-backend",
        "metal_tile",
        "--uvt-backward-policy",
        "deterministic_quality",
        "--uvt-camera-projection",
        "dataset_lens",
        "--uvt-init-views",
        "all_train",
        "--uvt-init-sampling",
        "grid",
        "--uvt-init-frames",
        "all",
        "--uvt-train-schedule",
        "view_shuffled_cycle",
        "--splat-count",
        str(PRIMITIVE_COUNT),
        "--splat-renderer",
        "fast_mac",
        "--splat-camera-projection",
        "dataset_lens",
        "--out-dir",
        str(out_dir),
    ]


def powerfoam_config(seed: int, out_dir: Path, *, wandb_mode: str) -> dict[str, Any]:
    cfg = copy.deepcopy(load_config_file(BASE_CONFIG))
    cfg["train"]["seed"] = int(seed)
    cfg["logging"].update(
        {
            "output_dir": str(out_dir),
            "wandb_enabled": True,
            "wandb_mode": wandb_mode,
            "wandb_run_id": f"cmwf{seed:04d}",
            "wandb_project": "dynaworld",
            "wandb_run_name": f"paper-coffee-martini-worldfoam-seed{seed}",
            "wandb_tags": [
                "paper-sweep",
                "coffee_martini",
                "train2-holdout1",
                "worldfoam",
                "powerfoam-metal",
                "paper-clean-train-only-init",
                f"seed-{seed}",
            ],
        }
    )
    return cfg


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def merge_seed_runs(existing_runs: list[dict[str, Any]], new_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs_by_seed = {int(run["seed"]): run for run in [*existing_runs, *new_runs]}
    return [runs_by_seed[seed] for seed in sorted(runs_by_seed)]


def build_sweep_manifest(runs: list[dict[str, Any]], *, wandb_mode: str) -> dict[str, Any]:
    merged_runs = merge_seed_runs([], runs)
    return {
        "benchmark": "coffee_martini_matched_sweep",
        "base_config": str(BASE_CONFIG.relative_to(ROOT)),
        "train_cameras": list(TRAIN_CAMERAS),
        "heldout_cameras": list(HELDOUT_CAMERAS),
        "target_size": TARGET_SIZE,
        "frame_count": FRAME_COUNT,
        "steps": STEPS,
        "primitive_count": PRIMITIVE_COUNT,
        "seeds": [int(run["seed"]) for run in merged_runs],
        "wandb_mode": wandb_mode,
        "runs": merged_runs,
    }


def _log_comparison_lane(
    report: dict[str, Any],
    *,
    representation: str,
    seed: int,
    report_dir: Path,
    wandb_mode: str,
) -> dict[str, Any]:
    key = "star_uvt" if representation == "world_tubes" else "free_dynamic_splats"
    lane = report[key]
    metrics = lane["metrics"]
    run = wandb.init(
        project="dynaworld",
        name=f"paper-coffee-martini-{representation}-seed{seed}",
        tags=["paper-sweep", "coffee_martini", "train2-holdout1", representation, f"seed-{seed}"],
        mode=wandb_mode,
        config={
            "seed": seed,
            "train_cameras": list(TRAIN_CAMERAS),
            "heldout_cameras": list(HELDOUT_CAMERAS),
            "target_size": TARGET_SIZE,
            "frame_count": FRAME_COUNT,
            "steps": STEPS,
            "primitive_count": PRIMITIVE_COUNT,
            "pose_source": report["meta"]["pose_source"],
            "policy": report["meta"].get("uvt_backward_policy") if representation == "world_tubes" else None,
        },
        reinit="finish_previous",
    )
    media_prefix = "star_uvt" if representation == "world_tubes" else "free_dynamic_splats"
    run.log(
        {
            "train/psnr": metrics["eval_psnr"],
            "train/ssim": metrics["eval_ssim"],
            "train/l1": metrics["eval_l1"],
            "heldout/psnr": metrics["heldout_eval_psnr"],
            "heldout/ssim": metrics["heldout_eval_ssim"],
            "heldout/l1": metrics["heldout_eval_l1"],
            "runtime/train_loop_elapsed_s": lane["train_loop_elapsed_s"],
            "media/train_view": wandb.Video(str(report_dir / f"{media_prefix}_train_view0_side_by_side.mp4"), format="mp4"),
            "media/heldout_view": wandb.Video(
                str(report_dir / f"{media_prefix}_heldout_view0_side_by_side.mp4"), format="mp4"
            ),
        },
        step=STEPS,
    )
    run_id = str(run.id)
    run_dir = str(run.dir)
    run.finish()
    return {"mode": wandb_mode, "run_id": run_id, "run_dir": run_dir}


def _run_powerfoam(seed: int, out_dir: Path, *, wandb_mode: str) -> dict[str, Any]:
    run_id = f"cmwf{seed:04d}"
    run_powerfoam_training(powerfoam_config(seed, out_dir, wandb_mode=wandb_mode))
    return {"mode": wandb_mode, "run_id": run_id}


def execute_seed(seed: int, out_dir: Path, *, wandb_mode: str, reuse_existing: bool) -> dict[str, Any]:
    seed_dir = out_dir / f"seed_{seed}"
    compare_dir = seed_dir / "world_tubes_dynamic_3dgs"
    worldfoam_dir = seed_dir / "worldfoam"
    report_path = compare_dir / "comparison_report.json"
    if not (reuse_existing and report_path.exists()):
        subprocess.run(comparison_command(seed, compare_dir), cwd=ROOT, check=True)
    report = _load_json(report_path)
    if report["meta"]["train_cameras"] != list(TRAIN_CAMERAS):
        raise ValueError(f"seed {seed}: unexpected train cameras")
    if report["meta"]["heldout_cameras"] != list(HELDOUT_CAMERAS):
        raise ValueError(f"seed {seed}: unexpected heldout cameras")
    if report["meta"]["uvt_backward_policy"]["promotion_contract"] is not True:
        raise ValueError(f"seed {seed}: World Tubes policy is not promotable")
    provenance = {
        "world_tubes": _log_comparison_lane(
            report,
            representation="world_tubes",
            seed=seed,
            report_dir=compare_dir,
            wandb_mode=wandb_mode,
        ),
        "dynamic_3dgs": _log_comparison_lane(
            report,
            representation="dynamic_3dgs",
            seed=seed,
            report_dir=compare_dir,
            wandb_mode=wandb_mode,
        ),
    }
    if not (reuse_existing and (worldfoam_dir / "best_metrics.json").exists()):
        provenance["worldfoam"] = _run_powerfoam(seed, worldfoam_dir, wandb_mode=wandb_mode)
    else:
        provenance["worldfoam"] = {"mode": wandb_mode, "run_id": f"cmwf{seed:04d}", "reused": True}
    seed_summary = {
        "seed": seed,
        "comparison_report": str(report_path.relative_to(ROOT)),
        "worldfoam_dir": str(worldfoam_dir.relative_to(ROOT)),
        "wandb": provenance,
    }
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "run_summary.json").write_text(json.dumps(seed_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return seed_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="offline")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--rebuild-manifest", action="store_true")
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    manifest_path = args.out_dir / "sweep_manifest.json"
    if args.rebuild_manifest:
        runs = [_load_json(path) for path in sorted(args.out_dir.glob("seed_*/run_summary.json"))]
        if not runs:
            raise SystemExit(f"No seed run summaries found under {args.out_dir}")
        summary = build_sweep_manifest(runs, wandb_mode=args.wandb_mode)
        manifest_path.write_text(json.dumps(serialize_config_value(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise SystemExit("--seeds must not be empty")
    if not args.execute:
        print(json.dumps({"seeds": seeds, "commands": [comparison_command(seed, args.out_dir / f"seed_{seed}") for seed in seeds]}, indent=2))
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [execute_seed(seed, args.out_dir, wandb_mode=args.wandb_mode, reuse_existing=args.reuse_existing) for seed in seeds]
    existing_runs = []
    if manifest_path.exists():
        existing = _load_json(manifest_path)
        existing_runs = [run for run in existing.get("runs", []) if isinstance(run, dict)]
    merged_runs = merge_seed_runs(existing_runs, rows)
    summary = build_sweep_manifest(merged_runs, wandb_mode=args.wandb_mode)
    manifest_path.write_text(json.dumps(serialize_config_value(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
