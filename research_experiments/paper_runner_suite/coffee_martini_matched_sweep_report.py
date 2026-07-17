from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP_DIR = ROOT / "outputs" / "benchmarks" / "2026-07-11_coffee_martini_matched_sweep"
DEFAULT_OUT_DIR = DEFAULT_SWEEP_DIR / "report"
EXPECTED_SEEDS = [17, 29, 43]
EXPECTED_TRAIN_CAMERAS = ["cam04", "cam09"]
EXPECTED_HELDOUT_CAMERAS = ["cam06"]
REPRESENTATIONS = ("world_tubes", "dynamic_3dgs", "worldfoam")
METRIC_KEYS = ("train_psnr", "train_ssim", "train_l1", "heldout_psnr", "heldout_ssim", "heldout_l1")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _last_jsonl(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or not isinstance(rows[-1], dict):
        raise ValueError(f"Expected JSON objects in {path}")
    return rows[-1]


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _comparison_row(
    *,
    representation: str,
    seed: int,
    report_path: Path,
    report: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    key = "star_uvt" if representation == "world_tubes" else "free_dynamic_splats"
    prefix = "star_uvt" if representation == "world_tubes" else "free_dynamic_splats"
    lane = report[key]
    metrics = lane["metrics"]
    report_dir = report_path.parent
    media = [
        report_dir / f"{prefix}_train_view0_side_by_side.mp4",
        report_dir / f"{prefix}_heldout_view0_side_by_side.mp4",
    ]
    return {
        "representation": representation,
        "seed": seed,
        "steps": int(lane["steps"]),
        "render_size": int(report["meta"]["target_size"]),
        "frame_count": int(report["meta"]["max_frames"]),
        "primitive_count": int(lane["tube_count"] if representation == "world_tubes" else lane["splat_count"]),
        "train_psnr": float(metrics["eval_psnr"]),
        "train_ssim": float(metrics["eval_ssim"]),
        "train_l1": float(metrics["eval_l1"]),
        "heldout_psnr": float(metrics["heldout_eval_psnr"]),
        "heldout_ssim": float(metrics["heldout_eval_ssim"]),
        "heldout_l1": float(metrics["heldout_eval_l1"]),
        "train_loop_elapsed_s": float(lane["train_loop_elapsed_s"]),
        "pose_source": report["meta"]["pose_source"],
        "policy": report["meta"].get("uvt_backward_policy") if representation == "world_tubes" else None,
        "wandb": provenance,
        "artifact": _relative(report_path),
        "media": [_relative(path) for path in media],
        "media_ok": all(path.exists() for path in media),
    }


def _worldfoam_row(seed: int, worldfoam_dir: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    resolved = _load_json(worldfoam_dir / "resolved_config.json")
    final = _last_jsonl(worldfoam_dir / "eval_metrics_history.jsonl")
    train_final = _last_jsonl(worldfoam_dir / "train_metrics_history.jsonl")
    metrics = final["metrics"]
    media = [worldfoam_dir / "side_by_side_step_0040.mp4", worldfoam_dir / "heldout_side_by_side_step_0040.mp4"]
    return {
        "representation": "worldfoam",
        "seed": seed,
        "steps": int(final["step"]),
        "render_size": int(resolved["render"]["render_size"]),
        "frame_count": int(resolved["data"]["max_frames"]),
        "primitive_count": int(resolved["model"]["cells"]),
        "train_psnr": float(metrics["eval_psnr"]),
        "train_ssim": float(metrics["eval_ssim"]),
        "train_l1": float(metrics["eval_l1"]),
        "heldout_psnr": float(metrics["heldout_eval_psnr"]),
        "heldout_ssim": float(metrics["heldout_eval_ssim"]),
        "heldout_l1": float(metrics["heldout_eval_l1"]),
        "train_loop_elapsed_s": float(train_final["elapsed_s"]),
        "pose_source": "neural_3d_llff_relative_pinhole",
        "policy": None,
        "wandb": provenance,
        "artifact": _relative(worldfoam_dir / "best_metrics.json"),
        "media": [_relative(path) for path in media],
        "media_ok": all(path.exists() for path in media),
        "initialization": resolved["model"]["init_point_cloud_path"],
        "paper_clean_initialization": (
            "feature_triangulation" in str(resolved["model"]["init_point_cloud_path"])
            and "ex4dgs" not in str(resolved["model"]["init_point_cloud_path"]).lower()
        ),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"representation": rows[0]["representation"], "seed_count": len(rows)}
    for key in (*METRIC_KEYS, "train_loop_elapsed_s"):
        values = [float(row[key]) for row in rows]
        result[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "min": min(values),
            "max": max(values),
        }
    return result


def _wandb_artifact_exists(provenance: dict[str, Any]) -> bool:
    run_dir = provenance.get("run_dir")
    if isinstance(run_dir, str):
        return Path(run_dir).exists()
    run_id = provenance.get("run_id")
    return isinstance(run_id, str) and any((ROOT / "wandb").glob(f"offline-run-*-{run_id}"))


def build_report(sweep_dir: Path = DEFAULT_SWEEP_DIR) -> dict[str, Any]:
    manifest = _load_json(sweep_dir / "sweep_manifest.json")
    rows = []
    for run in manifest["runs"]:
        seed = int(run["seed"])
        comparison_path = ROOT / run["comparison_report"]
        comparison = _load_json(comparison_path)
        rows.extend(
            [
                _comparison_row(
                    representation="world_tubes",
                    seed=seed,
                    report_path=comparison_path,
                    report=comparison,
                    provenance=run["wandb"]["world_tubes"],
                ),
                _comparison_row(
                    representation="dynamic_3dgs",
                    seed=seed,
                    report_path=comparison_path,
                    report=comparison,
                    provenance=run["wandb"]["dynamic_3dgs"],
                ),
                _worldfoam_row(seed, ROOT / run["worldfoam_dir"], run["wandb"]["worldfoam"]),
            ]
        )
    grouped = {representation: [row for row in rows if row["representation"] == representation] for representation in REPRESENTATIONS}
    aggregate = [_aggregate(grouped[representation]) for representation in REPRESENTATIONS]
    best = max(aggregate, key=lambda row: row["heldout_psnr"]["mean"])["representation"]
    exact_contract = all(
        row["steps"] == 40 and row["render_size"] == 128 and row["frame_count"] == 16 and row["primitive_count"] == 1024
        for row in rows
    )
    promotion_ok = all(
        row["policy"] is not None
        and row["policy"]["deterministic"] is True
        and row["policy"]["promotion_contract"] is True
        for row in grouped["world_tubes"]
    )
    gates = {
        "exact_three_seeds_ok": manifest["seeds"] == EXPECTED_SEEDS and all(len(grouped[key]) == 3 for key in REPRESENTATIONS),
        "camera_split_ok": manifest["train_cameras"] == EXPECTED_TRAIN_CAMERAS and manifest["heldout_cameras"] == EXPECTED_HELDOUT_CAMERAS,
        "matched_budget_ok": exact_contract,
        "separate_train_heldout_metrics_ok": all(all(isinstance(row[key], float) for key in METRIC_KEYS) for row in rows),
        "world_tubes_promotion_policy_ok": promotion_ok,
        "worldfoam_paper_clean_init_ok": all(row.get("paper_clean_initialization") is True for row in grouped["worldfoam"]),
        "wandb_offline_backing_ok": all(row["wandb"].get("mode") == "offline" and _wandb_artifact_exists(row["wandb"]) for row in rows),
        "media_ok": all(row["media_ok"] for row in rows),
    }
    return {
        "benchmark": "coffee_martini_train2_holdout1_matched_3seed",
        "scope": "single_scene_single_split_128px_16f_40step_1024primitive",
        "dataset": "Neural 3D Video/coffee_martini",
        "train_cameras": EXPECTED_TRAIN_CAMERAS,
        "heldout_cameras": EXPECTED_HELDOUT_CAMERAS,
        "seeds": EXPECTED_SEEDS,
        "rows": rows,
        "aggregate": aggregate,
        "best_mean_heldout_psnr": best,
        "gates": gates,
        "status": "ok" if all(gates.values()) else "failed",
        "paper_table_ready": all(gates.values()),
        "claim_boundary": "This is a complete matched table for one coffee_martini split, not a multi-scene SOTA claim.",
    }


def verify_report(report: dict[str, Any]) -> list[str]:
    errors = []
    if report.get("status") != "ok":
        errors.append("status must be ok")
    if report.get("paper_table_ready") is not True:
        errors.append("paper_table_ready must be true")
    for key, value in report.get("gates", {}).items():
        if value is not True:
            errors.append(f"gate {key} must be true")
    if report.get("best_mean_heldout_psnr") != "world_tubes":
        errors.append("world_tubes must have the best mean heldout PSNR in the saved sweep")
    if len(report.get("rows", [])) != 9:
        errors.append("expected nine per-seed representation rows")
    return errors


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Coffee Martini matched three-seed table",
        "",
        "Train cameras: `cam04`, `cam09`. Heldout camera: `cam06`.",
        "",
        "| Representation | Heldout PSNR | Heldout SSIM | Heldout L1 | Train PSNR | Train time |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["aggregate"]:
        lines.append(
            f"| {row['representation']} | {row['heldout_psnr']['mean']:.4f} +/- {row['heldout_psnr']['std']:.4f} | "
            f"{row['heldout_ssim']['mean']:.4f} | {row['heldout_l1']['mean']:.4f} | "
            f"{row['train_psnr']['mean']:.4f} | {row['train_loop_elapsed_s']['mean']:.2f}s |"
        )
    lines.extend(["", report["claim_boundary"]])
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--verify-report", type=Path)
    args = parser.parse_args()
    if args.verify_report:
        errors = verify_report(_load_json(args.verify_report))
        if errors:
            raise SystemExit("\n".join(errors))
        print(f"Verified {args.verify_report}")
        return
    report = build_report(args.sweep_dir.resolve())
    errors = verify_report(report)
    if errors:
        raise SystemExit("\n".join(errors))
    write_report(report, args.out_dir.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
