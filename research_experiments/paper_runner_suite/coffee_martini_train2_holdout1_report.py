from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORLD_TUBES_REPORT = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-07-11_coffee_martini_train2_holdout1_matched_seed17_pilot"
    / "comparison_report.json"
)
DEFAULT_WORLDFOAM_DIR = (
    ROOT
    / "outputs"
    / "powerfoam_metal"
    / "local_mac_powerfoam_metal_multicam_neural3d_coffee_train2_holdout1_feature_triangulation_reproj8_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux"
)
DEFAULT_CAMERA_AUDIT = (
    ROOT / "outputs" / "benchmarks" / "2026-07-11_coffee_martini_train2_holdout1_camera_projection_audit.json"
)
DEFAULT_OUT_DIR = ROOT / "outputs" / "benchmarks" / "2026-07-11_coffee_martini_train2_holdout1_protocol"
EXPECTED_TRAIN_CAMERAS = ["cam04", "cam09"]
EXPECTED_HELDOUT_CAMERAS = ["cam06"]
EXPECTED_POSE_SOURCE = "neural_3d_llff_opencv_relative_pinhole_v2"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _last_jsonl(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or not isinstance(rows[-1], dict):
        raise ValueError(f"Expected at least one JSON object in {path}")
    return rows[-1]


def _metric_row(
    representation: str,
    *,
    metrics: dict[str, Any],
    steps: int,
    render_size: int,
    frame_count: int,
    elapsed_s: float | None,
    artifact: Path,
    initialization: str,
) -> dict[str, Any]:
    return {
        "representation": representation,
        "train_cameras": EXPECTED_TRAIN_CAMERAS,
        "heldout_cameras": EXPECTED_HELDOUT_CAMERAS,
        "steps": steps,
        "render_size": render_size,
        "frame_count": frame_count,
        "train_psnr": metrics.get("eval_psnr"),
        "train_l1": metrics.get("eval_l1"),
        "train_ssim": metrics.get("eval_ssim"),
        "heldout_psnr": metrics.get("heldout_eval_psnr"),
        "heldout_l1": metrics.get("heldout_eval_l1"),
        "heldout_ssim": metrics.get("heldout_eval_ssim"),
        "train_loop_elapsed_s": elapsed_s,
        "initialization": initialization,
        "artifact": str(artifact.relative_to(ROOT)),
    }


def build_report(
    world_tubes_report: Path = DEFAULT_WORLD_TUBES_REPORT,
    worldfoam_dir: Path = DEFAULT_WORLDFOAM_DIR,
    camera_audit: Path = DEFAULT_CAMERA_AUDIT,
) -> dict[str, Any]:
    comparison = _load_json(world_tubes_report)
    audit = _load_json(camera_audit)
    worldfoam_config = _load_json(worldfoam_dir / "resolved_config.json")
    worldfoam_final = _last_jsonl(worldfoam_dir / "eval_metrics_history.jsonl")
    worldfoam_train_final = _last_jsonl(worldfoam_dir / "train_metrics_history.jsonl")

    star = comparison["star_uvt"]
    splats = comparison["free_dynamic_splats"]
    worldfoam_metrics = worldfoam_final["metrics"]
    rows = [
        _metric_row(
            "world_tubes_star_uvt",
            metrics=star["metrics"],
            steps=int(star["steps"]),
            render_size=int(comparison["meta"]["target_size"]),
            frame_count=int(comparison["meta"]["max_frames"]),
            elapsed_s=float(star["train_loop_elapsed_s"]),
            artifact=world_tubes_report,
            initialization="all_train_frames_grid",
        ),
        _metric_row(
            "dynamic_3dgs_fast_mac",
            metrics=splats["metrics"],
            steps=int(splats["steps"]),
            render_size=int(comparison["meta"]["target_size"]),
            frame_count=int(comparison["meta"]["max_frames"]),
            elapsed_s=float(splats["train_loop_elapsed_s"]),
            artifact=world_tubes_report,
            initialization="all_train_frames",
        ),
        _metric_row(
            "worldfoam_powerfoam_metal",
            metrics=worldfoam_metrics,
            steps=int(worldfoam_final["step"]),
            render_size=int(worldfoam_config["render"]["render_size"]),
            frame_count=int(worldfoam_config["data"]["max_frames"]),
            elapsed_s=float(worldfoam_train_final["elapsed_s"]),
            artifact=worldfoam_dir / "best_metrics.json",
            initialization="train_camera_only_orb_triangulation_89_points",
        ),
    ]
    matched_shapes = len(
        {(row["steps"], row["render_size"], row["frame_count"]) for row in rows}
    ) == 1 and len({1024, int(star["tube_count"]), int(splats["splat_count"]), int(worldfoam_config["model"]["cells"])}) == 1
    split_ok = (
        comparison["meta"]["train_cameras"] == EXPECTED_TRAIN_CAMERAS
        and comparison["meta"]["heldout_cameras"] == EXPECTED_HELDOUT_CAMERAS
        and worldfoam_config["data"]["multicam_train_cameras"] == EXPECTED_TRAIN_CAMERAS
        and [worldfoam_config["data"]["multicam_heldout_camera"]] == EXPECTED_HELDOUT_CAMERAS
        and audit["train_cameras"] == EXPECTED_TRAIN_CAMERAS
        and audit["heldout_cameras"] == EXPECTED_HELDOUT_CAMERAS
    )
    calibration_ok = (
        comparison["meta"]["pose_source"] == EXPECTED_POSE_SOURCE
        and all(camera["lens_model"] == "pinhole" for group in audit["rows"] for camera in group["cameras"])
    )
    metrics_ok = all(
        isinstance(row[key], int | float)
        for row in rows
        for key in ("train_psnr", "train_l1", "train_ssim", "heldout_psnr", "heldout_l1", "heldout_ssim")
    )
    return {
        "benchmark": "coffee_martini_train2_holdout1_protocol",
        "dataset": "Neural 3D Video/coffee_martini",
        "scene": "coffee_martini",
        "train_cameras": EXPECTED_TRAIN_CAMERAS,
        "heldout_cameras": EXPECTED_HELDOUT_CAMERAS,
        "pose_source": comparison["meta"]["pose_source"],
        "required_pose_source": EXPECTED_POSE_SOURCE,
        "rows": rows,
        "gates": {
            "split_ok": split_ok,
            "calibration_ok": calibration_ok,
            "separate_train_and_heldout_metrics_ok": metrics_ok,
            "matched_for_ranking": matched_shapes,
            "three_seed_repeat_ok": False,
            "wandb_backing_ok": False,
            "world_tubes_promotion_policy_ok": bool(
                comparison["meta"].get("uvt_backward_policy", {}).get("promotion_contract")
            ),
        },
        "status": "ok" if split_ok and calibration_ok and metrics_ok else "failed",
        "paper_rankable": False,
        "next_protocol": {
            "render_size": 128,
            "frame_count": 16,
            "train_cameras": EXPECTED_TRAIN_CAMERAS,
            "heldout_cameras": EXPECTED_HELDOUT_CAMERAS,
            "seeds": [17, 29, 43],
            "selection_metric": "heldout_psnr",
            "required_reports": ["train_psnr", "train_ssim", "train_l1", "heldout_psnr", "heldout_ssim", "heldout_l1"],
            "fairness_rule": "No heldout-camera frames or external pretrained scene geometry may initialize any ranked lane.",
        },
    }


def verify_report(report: dict[str, Any]) -> list[str]:
    errors = []
    if report.get("status") != "ok":
        errors.append("status must be ok")
    if report.get("paper_rankable") is not False:
        errors.append("current mismatched protocol must not be paper-rankable")
    gates = report.get("gates", {})
    for key in ("split_ok", "calibration_ok", "separate_train_and_heldout_metrics_ok"):
        if gates.get(key) is not True:
            errors.append(f"gate {key} must be true")
    if gates.get("matched_for_ranking") is not True:
        errors.append("matched_for_ranking must be true for the seed-17 pilot")
    for key in ("three_seed_repeat_ok", "wandb_backing_ok", "world_tubes_promotion_policy_ok"):
        if gates.get(key) is not False:
            errors.append(f"gate {key} must remain false for the seed-17 pilot")
    if len(report.get("rows", [])) != 3:
        errors.append("expected exactly three representation rows")
    return errors


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Coffee Martini train2/holdout1 protocol",
        "",
        "Train cameras: `cam04`, `cam09`. Heldout camera: `cam06`.",
        "",
        "| Representation | Size/frames/steps | Train PSNR | Heldout PSNR | Heldout SSIM |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['representation']} | {row['render_size']} / {row['frame_count']} / {row['steps']} | "
            f"{row['train_psnr']:.4f} | {row['heldout_psnr']:.4f} | {row['heldout_ssim']:.4f} |"
        )
    lines.extend(
        [
            "",
            "This seed-17 pilot matches all three lanes at 128px/16f/40 steps/1024 primitives. "
            "It is not a final paper row until three seeds, W&B backing, and a promotable World Tubes backward policy are present.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--verify-report", type=Path)
    args = parser.parse_args()
    if args.verify_report:
        errors = verify_report(_load_json(args.verify_report))
        if errors:
            raise SystemExit("\n".join(errors))
        print(f"Verified {args.verify_report}")
        return
    report = build_report()
    errors = verify_report(report)
    if errors:
        raise SystemExit("\n".join(errors))
    write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
