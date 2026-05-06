from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MIN_CLEAN_HELDOUT_PSNR = 13.0
DEFAULT_MIN_CLEAN_HELDOUT_SSIM = 0.15

OFFICIAL_FIXTURE = (
    ROOT
    / "research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_official_camera_official_v1.json"
)

CLEAN_DEEPVIEW_CANDIDATES = (
    (
        ROOT
        / "outputs/powerfoam_metal/"
        "local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_512px_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux",
        ROOT / "research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frame0_512px_sift_wide.json",
    ),
    (
        ROOT
        / "outputs/powerfoam_metal/"
        "local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_512px_merged_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux",
        ROOT
        / "research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_512px_sift_wide_merged.json",
    ),
    (
        ROOT
        / "outputs/powerfoam_metal/"
        "local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux",
        ROOT
        / "research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_sift_wide_minucam2.json",
    ),
    (
        ROOT
        / "outputs/powerfoam_metal/"
        "local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_init_raytrace_128_16f_2524cells_40step_lowgeom_noaux",
        ROOT
        / "research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_sift_wide_minucam2.json",
    ),
    (
        ROOT
        / "outputs/powerfoam_metal/"
        "local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_top1024_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux",
        ROOT
        / "research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_sift_wide_minucam2.json",
    ),
    (
        ROOT
        / "outputs/powerfoam_metal/"
        "local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_fisheye_rays_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux",
        ROOT
        / "research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_sift_wide_minucam2.json",
    ),
    (
        ROOT
        / "outputs/powerfoam_metal/"
        "local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux",
        ROOT
        / "research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_sift_wide_minucam2.json",
    ),
    (
        ROOT
        / "outputs/powerfoam_metal/"
        "local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_128_16f_1024cells_40step_noaux",
        ROOT
        / "research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_sift_wide_minucam2.json",
    ),
    (
        ROOT
        / "outputs/powerfoam_metal/"
        "local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_40step_noaux",
        ROOT
        / "research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_sift_wide_minucam2.json",
    ),
    (
        ROOT
        / "outputs/powerfoam_metal/"
        "local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_materialonly_wandboffline_init_raytrace_128_16f_1024cells_40step_noaux",
        ROOT
        / "research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_sift_wide_minucam2.json",
    ),
    (
        ROOT
        / "outputs/powerfoam_metal/"
        "local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_first1024_appearanceonly_wandboffline_init_raytrace_128_16f_1024cells_40step_noaux",
        ROOT
        / "research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_sift_wide_minucam2.json",
    ),
    (
        ROOT
        / "outputs/powerfoam_metal/"
        "local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_init_raytrace_128_16f_2714cells_40step_lowgeom_noaux",
        ROOT
        / "research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_sift_wide_minucam2.json",
    ),
    (
        ROOT
        / "outputs/powerfoam_metal/"
        "local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_128_16f_2714cells_40step_denseeval",
        ROOT
        / "research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_sift_wide_minucam2.json",
    ),
    (
        ROOT
        / "outputs/powerfoam_metal/"
        "local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_normalmap_128_16f_2714cells_40step_denseeval",
        ROOT
        / "research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_sift_wide_minucam2.json",
    ),
    (
        ROOT
        / "outputs/powerfoam_metal/"
        "local_mac_powerfoam_metal_multicam_deepview_nearest0040_8cam_holdout0040_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_128_16f_3543cells_40step_denseeval",
        ROOT
        / "research_experiments/dynamic_foam/artifacts/deepview_03_dog_nearest0040_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_sift_wide_minucam2.json",
    ),
    (
        ROOT
        / "outputs/powerfoam_metal/"
        "local_mac_powerfoam_metal_multicam_deepview_nearest0040_8cam_holdout0040_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_evalrgbcal_128_16f_3543cells_1step_denseeval",
        ROOT
        / "research_experiments/dynamic_foam/artifacts/deepview_03_dog_nearest0040_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_sift_wide_minucam2.json",
    ),
    (
        ROOT
        / "outputs/powerfoam_metal/"
        "local_mac_powerfoam_metal_multicam_deepview_nearest0040_8cam_holdout0040_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_regular_official_objective_fastwarmup_128_16f_3543cells_9sites_cap384_40step_denseeval",
        ROOT
        / "research_experiments/dynamic_foam/artifacts/deepview_03_dog_nearest0040_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_sift_wide_minucam2.json",
    ),
    (
        ROOT
        / "outputs/powerfoam_metal/"
        "local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_multiview_plane_sweep_frame0_128px_opencv_fisheye_top1024_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux",
        ROOT
        / "research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_multiview_plane_sweep_frame0_128px_opencv_fisheye_stride4_support4_8192pts.json",
    ),
    (
        ROOT
        / "outputs/powerfoam_metal/"
        "local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_multiview_plane_sweep_frame0_128px_opencv_fisheye_inlier02_top1024_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux",
        ROOT
        / "research_experiments/dynamic_foam/artifacts/deepview_03_dog_8cam_multiview_plane_sweep_frame0_128px_opencv_fisheye_stride4_support4_inlier02_8192pts.json",
    ),
)
OPTIONAL_CLEAN_DEEPVIEW_CANDIDATES = tuple(
    (
        ROOT
        / "outputs/powerfoam_metal"
        / (
            "local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_"
            "pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_"
            f"opencv_fisheye_aliked_n16rot_{matcher}_init_raytrace_128_16f_1024cells_40step_lowgeom_noaux"
        ),
        ROOT
        / "research_experiments/dynamic_foam/artifacts"
        / (
            "deepview_03_dog_8cam_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_"
            f"opencv_fisheye_aliked_n16rot_{matcher}_minucam2.json"
        ),
    )
    for matcher in ("aliked_lightglue", "aliked_bruteforce")
)

EX4DGS_OUTPUT = (
    ROOT
    / "outputs/powerfoam_metal/"
    "local_mac_powerfoam_metal_multicam_neural3d_coffee_train2_holdout1_ex4dgs_init_raytrace_128_16f_1024cells_200step_lowgeom_noaux"
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return data


def artifact_point_cloud_path(artifact_meta: Path) -> Path | None:
    if not artifact_meta.exists():
        return None
    artifact = load_json(artifact_meta)
    output = Path(str(artifact.get("output", artifact_meta.with_suffix(".ply"))))
    return output if output.is_absolute() else ROOT / output


def candidate_ready(output_dir: Path, artifact_meta: Path, *, require_point_cloud: bool = False) -> bool:
    if not (output_dir / "best_metrics.json").exists():
        return False
    if not (output_dir / "resolved_config.json").exists():
        return False
    if not artifact_meta.exists():
        return False
    if require_point_cloud:
        point_cloud = artifact_point_cloud_path(artifact_meta)
        return bool(point_cloud is not None and point_cloud.exists())
    return True


def existing_clean_candidates(*, require_point_cloud: bool = False) -> list[tuple[Path, Path]]:
    candidates = [
        candidate
        for candidate in CLEAN_DEEPVIEW_CANDIDATES
        if candidate_ready(*candidate, require_point_cloud=require_point_cloud)
    ]
    candidates.extend(
        candidate
        for candidate in OPTIONAL_CLEAN_DEEPVIEW_CANDIDATES
        if candidate_ready(*candidate, require_point_cloud=require_point_cloud)
    )
    return candidates


def missing_optional_clean_candidates(*, require_point_cloud: bool = False) -> list[dict[str, str]]:
    return [
        {
            "output_dir": str(output_dir.relative_to(ROOT)),
            "artifact_meta": str(artifact_meta.relative_to(ROOT)),
        }
        for output_dir, artifact_meta in OPTIONAL_CLEAN_DEEPVIEW_CANDIDATES
        if not candidate_ready(output_dir, artifact_meta, require_point_cloud=require_point_cloud)
    ]


def check(condition: bool, name: str, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(condition), "evidence": evidence}


def next_blockers(checks: list[dict[str, Any]], missing_optional: list[dict[str, str]]) -> list[str]:
    blockers: list[str] = []
    failed = {str(item["name"]) for item in checks if not bool(item["passed"])}
    if "official_cuda_warp_fixture_exists" in failed:
        blockers.append("Generate the official CUDA/Warp fixture and rerun the targeted Direct/Metal parity tests.")
    if {"clean_heldout_psnr_threshold", "clean_heldout_ssim_threshold"} & failed:
        blockers.append(
            "Improve clean heldout quality beyond the current true-multiframe OPENCV_FISHEYE pycolmap init; "
            "the selected row remains below PSNR/SSIM thresholds."
        )
    if "clean_eval_color_calibration_disclosed" in failed:
        blockers.append(
            "The selected clean row uses eval color calibration but is missing its calibration artifact "
            "or uncalibrated heldout metrics."
        )
    if "clean_post_initial_paper_quality_row" in failed:
        blockers.append(
            "The selected clean row has no post-initial eval-history row with paper-quality heldout metrics, "
            "nonzero state movement, and calibration disclosure."
        )
    if {"clean_raw_heldout_psnr_threshold", "clean_raw_heldout_ssim_threshold"} & failed:
        blockers.append(
            "Improve raw uncalibrated clean heldout quality; eval color calibration can be reported, "
            "but it cannot satisfy --require-raw-quality."
        )
    if "clean_post_initial_raw_quality_row" in failed:
        blockers.append(
            "The selected clean row has no post-initial eval-history row with raw heldout metrics above threshold "
            "and nonzero state movement."
        )
    artifact_failures = {
        "clean_true_track_artifact",
        "clean_multiframe_support",
        "clean_point_count_threshold",
        "clean_long_track_threshold",
        "clean_multiview_track_support",
        "clean_temporal_track_support",
        "clean_reprojection_quality",
        "clean_verified_pair_threshold",
    }
    if artifact_failures & failed:
        blockers.append("Fix the clean train-camera-only geometry artifact before treating any run as paper-scale evidence.")
    if missing_optional:
        blockers.append(
            "Optional ALIKED/LightGlue selector rows are still missing; only generate them after a cheap "
            "ONNX/COLMAP-CLI probe shows dense enough train-camera tracks to justify the full run."
        )
    return blockers


def metric(metrics: dict[str, Any], key: str) -> float:
    return float(metrics["metrics"][key])


def optional_metric(metrics: dict[str, Any], key: str) -> float | None:
    values = metrics.get("metrics", {})
    if not isinstance(values, dict) or key not in values:
        return None
    return float(values[key])


def optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def eval_color_calibration_artifact(output_dir: Path, mode: str, step: int) -> Path | None:
    if mode == "none":
        return None
    return output_dir / f"eval_color_calibration_step_{int(step):04d}.json"


def relative_path_or_none(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_eval_history(output_dir: Path) -> list[dict[str, Any]]:
    history_path = output_dir / "eval_metrics_history.jsonl"
    if not history_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(history_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"{history_path}:{line_number} must contain a JSON object.")
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            raise TypeError(f"{history_path}:{line_number} metrics must contain a JSON object.")
        rows.append({"step": int(payload["step"]), "metrics": metrics})
    return rows


def row_uses_eval_calibration(row: dict[str, Any]) -> bool:
    return str(row.get("eval_color_calibration", "none")) != "none"


def raw_quality_source(row: dict[str, Any]) -> str:
    return "uncalibrated_heldout_eval" if row_uses_eval_calibration(row) else "heldout_eval"


def raw_quality_metrics(row: dict[str, Any]) -> dict[str, float | str | None]:
    source = raw_quality_source(row)
    if source == "heldout_eval":
        return {
            "raw_heldout_eval_psnr": optional_float(row.get("heldout_eval_psnr")),
            "raw_heldout_eval_l1": optional_float(row.get("heldout_eval_l1")),
            "raw_heldout_eval_ssim": optional_float(row.get("heldout_eval_ssim")),
            "raw_quality_source": source,
        }
    return {
        "raw_heldout_eval_psnr": optional_float(row.get("uncalibrated_heldout_eval_psnr")),
        "raw_heldout_eval_l1": optional_float(row.get("uncalibrated_heldout_eval_l1")),
        "raw_heldout_eval_ssim": optional_float(row.get("uncalibrated_heldout_eval_ssim")),
        "raw_quality_source": source,
    }


def raw_quality_ok(
    row: dict[str, Any],
    *,
    min_clean_heldout_psnr: float,
    min_clean_heldout_ssim: float,
) -> bool:
    raw_metrics = raw_quality_metrics(row)
    raw_psnr = optional_float(raw_metrics["raw_heldout_eval_psnr"])
    raw_ssim = optional_float(raw_metrics["raw_heldout_eval_ssim"])
    return (
        raw_psnr is not None
        and raw_ssim is not None
        and raw_psnr >= float(min_clean_heldout_psnr)
        and raw_ssim >= float(min_clean_heldout_ssim)
    )


def calibrated_quality_ok(
    row: dict[str, Any],
    *,
    min_clean_heldout_psnr: float,
    min_clean_heldout_ssim: float,
) -> bool:
    return (
        float(row["heldout_eval_psnr"]) >= float(min_clean_heldout_psnr)
        and float(row["heldout_eval_ssim"]) >= float(min_clean_heldout_ssim)
    )


def with_quality_flags(
    row: dict[str, Any],
    *,
    min_clean_heldout_psnr: float,
    min_clean_heldout_ssim: float,
) -> dict[str, Any]:
    flagged = dict(row)
    flagged.update(raw_quality_metrics(flagged))
    flagged["raw_quality_ok"] = raw_quality_ok(
        flagged,
        min_clean_heldout_psnr=min_clean_heldout_psnr,
        min_clean_heldout_ssim=min_clean_heldout_ssim,
    )
    flagged["calibrated_quality_ok"] = calibrated_quality_ok(
        flagged,
        min_clean_heldout_psnr=min_clean_heldout_psnr,
        min_clean_heldout_ssim=min_clean_heldout_ssim,
    )
    return flagged


def max_training_state_delta(metrics: dict[str, Any]) -> float:
    keys = (
        "state_mean_center_delta",
        "state_mean_density_delta",
        "state_mean_feature_delta",
        "state_mean_normal_delta",
        "state_mean_quaternion_delta",
        "state_mean_texel_site_delta",
        "state_mean_texel_height_delta",
        "state_mean_texel_sv_axis_delta",
        "state_mean_texel_sv_rgb_delta",
        "state_mean_xy_delta",
        "state_mean_z_delta",
        "state_max_center_delta",
    )
    values = [abs(float(metrics[key])) for key in keys if key in metrics]
    return max(values) if values else 0.0


def history_row_summary(
    output_dir: Path,
    clean_config: dict[str, Any],
    row: dict[str, Any],
    *,
    min_clean_heldout_psnr: float = DEFAULT_MIN_CLEAN_HELDOUT_PSNR,
    min_clean_heldout_ssim: float = DEFAULT_MIN_CLEAN_HELDOUT_SSIM,
) -> dict[str, Any]:
    step = int(row["step"])
    metrics = row["metrics"]
    calibration_mode = str(clean_config.get("render", {}).get("eval_color_calibration", "none"))
    calibration_artifact = eval_color_calibration_artifact(output_dir, calibration_mode, step)
    summary = {
        "output_dir": str(relative_path_or_none(output_dir)),
        "step": step,
        "heldout_eval_psnr": float(metrics["heldout_eval_psnr"]),
        "heldout_eval_l1": float(metrics["heldout_eval_l1"]),
        "heldout_eval_ssim": float(metrics["heldout_eval_ssim"]),
        "uncalibrated_heldout_eval_psnr": metrics.get("uncalibrated_heldout_eval_psnr"),
        "uncalibrated_heldout_eval_l1": metrics.get("uncalibrated_heldout_eval_l1"),
        "uncalibrated_heldout_eval_ssim": metrics.get("uncalibrated_heldout_eval_ssim"),
        "eval_color_calibration": calibration_mode,
        "eval_color_calibration_artifact": relative_path_or_none(calibration_artifact),
        "eval_color_calibration_artifact_exists": (
            calibration_artifact.exists() if calibration_artifact is not None else True
        ),
        "max_training_state_delta": max_training_state_delta(metrics),
    }
    return with_quality_flags(
        summary,
        min_clean_heldout_psnr=min_clean_heldout_psnr,
        min_clean_heldout_ssim=min_clean_heldout_ssim,
    )


def calibration_disclosed(row: dict[str, Any]) -> bool:
    if row["eval_color_calibration"] == "none":
        return True
    return (
        bool(row["eval_color_calibration_artifact_exists"])
        and row["uncalibrated_heldout_eval_psnr"] is not None
        and row["uncalibrated_heldout_eval_ssim"] is not None
    )


def post_initial_paper_quality_rows(
    output_dir: Path,
    clean_config: dict[str, Any],
    *,
    min_clean_heldout_psnr: float,
    min_clean_heldout_ssim: float,
    min_training_state_delta: float = 1.0e-8,
) -> list[dict[str, Any]]:
    rows = [
        history_row_summary(
            output_dir,
            clean_config,
            row,
            min_clean_heldout_psnr=min_clean_heldout_psnr,
            min_clean_heldout_ssim=min_clean_heldout_ssim,
        )
        for row in load_eval_history(output_dir)
        if int(row["step"]) > 0
        and "heldout_eval_psnr" in row["metrics"]
        and "heldout_eval_ssim" in row["metrics"]
        and "heldout_eval_l1" in row["metrics"]
    ]
    return [
        row
        for row in rows
        if float(row["heldout_eval_psnr"]) >= float(min_clean_heldout_psnr)
        and float(row["heldout_eval_ssim"]) >= float(min_clean_heldout_ssim)
        and float(row["max_training_state_delta"]) > float(min_training_state_delta)
        and calibration_disclosed(row)
    ]


def post_initial_raw_quality_rows(
    output_dir: Path,
    clean_config: dict[str, Any],
    *,
    min_clean_heldout_psnr: float,
    min_clean_heldout_ssim: float,
    min_training_state_delta: float = 1.0e-8,
    allow_raw_step0_acceptance: bool = False,
) -> list[dict[str, Any]]:
    rows = [
        history_row_summary(
            output_dir,
            clean_config,
            row,
            min_clean_heldout_psnr=min_clean_heldout_psnr,
            min_clean_heldout_ssim=min_clean_heldout_ssim,
        )
        for row in load_eval_history(output_dir)
        if (int(row["step"]) > 0 or (allow_raw_step0_acceptance and int(row["step"]) == 0))
        and "heldout_eval_psnr" in row["metrics"]
        and "heldout_eval_ssim" in row["metrics"]
        and "heldout_eval_l1" in row["metrics"]
    ]
    accepted = []
    for row in rows:
        step = int(row["step"])
        has_trainability = step > 0 and float(row["max_training_state_delta"]) > float(min_training_state_delta)
        if step == 0 and allow_raw_step0_acceptance:
            row = dict(row)
            row["trainability_evidence"] = "absent_step0_only"
        elif has_trainability:
            row = dict(row)
            row["trainability_evidence"] = "post_initial_state_delta"
        else:
            continue
        if bool(row["raw_quality_ok"]) and calibration_disclosed(row):
            accepted.append(row)
    return accepted


def weighted_source_mean(sources: list[dict[str, Any]], section: str, key: str) -> float:
    total = 0.0
    weight_sum = 0
    for source in sources:
        weight = int(source.get("point_count", source.get("vertex_count", 0)))
        if weight <= 0 or section not in source or key not in source[section]:
            continue
        total += float(source[section][key]) * weight
        weight_sum += weight
    if weight_sum == 0:
        raise ValueError(f"No weighted source values for {section}.{key}.")
    return total / float(weight_sum)


def max_source_value(sources: list[dict[str, Any]], section: str, key: str) -> float:
    values = [float(source[section][key]) for source in sources if section in source and key in source[section]]
    if not values:
        raise ValueError(f"No source values for {section}.{key}.")
    return max(values)


def clean_candidate_metrics(
    output_dir: Path,
    artifact_meta: Path,
    *,
    min_clean_heldout_psnr: float = DEFAULT_MIN_CLEAN_HELDOUT_PSNR,
    min_clean_heldout_ssim: float = DEFAULT_MIN_CLEAN_HELDOUT_SSIM,
) -> tuple[dict[str, Any], dict[str, Any]]:
    clean_best = load_json(output_dir / "best_metrics.json")
    clean_config = load_json(output_dir / "resolved_config.json")
    clean_artifact = load_json(artifact_meta)
    step = int(clean_best["step"])
    calibration_mode = str(clean_config.get("render", {}).get("eval_color_calibration", "none"))
    calibration_artifact = output_dir / f"eval_color_calibration_step_{step:04d}.json"

    clean_metrics = with_quality_flags(
        {
        "output_dir": str(output_dir.relative_to(ROOT)),
        "step": step,
        "heldout_eval_psnr": metric(clean_best, "heldout_eval_psnr"),
        "heldout_eval_l1": metric(clean_best, "heldout_eval_l1"),
        "heldout_eval_ssim": metric(clean_best, "heldout_eval_ssim"),
        "uncalibrated_heldout_eval_psnr": optional_metric(clean_best, "uncalibrated_heldout_eval_psnr"),
        "uncalibrated_heldout_eval_l1": optional_metric(clean_best, "uncalibrated_heldout_eval_l1"),
        "uncalibrated_heldout_eval_ssim": optional_metric(clean_best, "uncalibrated_heldout_eval_ssim"),
        "render_size": int(clean_config["render"]["render_size"]),
        "cells": int(clean_config["model"]["cells"]),
        "frames": int(clean_config["data"]["max_frames"]),
        "wandb_enabled": bool(clean_config["logging"]["wandb_enabled"]),
        "init_point_cloud_path": clean_config["model"]["init_point_cloud_path"],
        "eval_color_calibration": calibration_mode,
        "eval_color_calibration_artifact": (
            str(calibration_artifact.relative_to(ROOT)) if calibration_mode != "none" else None
        ),
        "eval_color_calibration_artifact_exists": (
            calibration_artifact.exists() if calibration_mode != "none" else True
        ),
        },
        min_clean_heldout_psnr=min_clean_heldout_psnr,
        min_clean_heldout_ssim=min_clean_heldout_ssim,
    )
    artifact_output = str(clean_artifact.get("output", artifact_meta.with_suffix(".ply")))
    if "sources" in clean_artifact:
        sources = list(clean_artifact["sources"])
        artifact_metrics = {
            "artifact_kind": "pycolmap_merged",
            "artifact": str(artifact_meta.relative_to(ROOT)),
            "output": artifact_output,
            "point_count": int(clean_artifact["point_count"]),
            "verified_pairs": max(int(source["database_num_verified_image_pairs"]) for source in sources),
            "track_mean": weighted_source_mean(sources, "filtered_track_length", "mean"),
            "track_p90": max_source_value(sources, "filtered_track_length", "p90"),
            "track_max": max_source_value(sources, "filtered_track_length", "max"),
            "reproj_median": weighted_source_mean(sources, "filtered_reproj_error", "median"),
            "reproj_p90": max_source_value(sources, "filtered_reproj_error", "p90"),
            "merge_mode": clean_artifact.get("merge_mode"),
            "source_frames": clean_artifact.get("frame_indices", []),
            "multi_frame_database": bool(clean_artifact.get("multi_frame_database", False)),
        }
    elif "database_num_verified_image_pairs" in clean_artifact:
        artifact_metrics = {
            "artifact_kind": "pycolmap_known_pose",
            "artifact": str(artifact_meta.relative_to(ROOT)),
            "output": artifact_output,
            "point_count": int(clean_artifact["point_count"]),
            "verified_pairs": int(clean_artifact["database_num_verified_image_pairs"]),
            "track_mean": float(clean_artifact["filtered_track_length"]["mean"]),
            "track_p90": float(clean_artifact["filtered_track_length"]["p90"]),
            "track_max": float(clean_artifact["filtered_track_length"]["max"]),
            "reproj_median": float(clean_artifact["filtered_reproj_error"]["median"]),
            "reproj_p90": float(clean_artifact["filtered_reproj_error"]["p90"]),
            "merge_mode": clean_artifact.get("merge_mode"),
            "source_frames": clean_artifact.get("frame_indices", []),
            "multi_frame_database": bool(clean_artifact.get("multi_frame_database", False)),
        }
        if "filtered_unique_camera_track_length" in clean_artifact:
            artifact_metrics["unique_camera_track_p90"] = float(
                clean_artifact["filtered_unique_camera_track_length"]["p90"]
            )
            artifact_metrics["unique_camera_track_mean"] = float(
                clean_artifact["filtered_unique_camera_track_length"]["mean"]
            )
        if "filtered_unique_frame_track_length" in clean_artifact:
            artifact_metrics["unique_frame_track_p90"] = float(clean_artifact["filtered_unique_frame_track_length"]["p90"])
            artifact_metrics["unique_frame_track_mean"] = float(
                clean_artifact["filtered_unique_frame_track_length"]["mean"]
            )
    elif {"support_mean", "median_error", "p90_error"}.issubset(clean_artifact):
        artifact_metrics = {
            "artifact_kind": "multiview_plane_sweep",
            "artifact": str(artifact_meta.relative_to(ROOT)),
            "output": artifact_output,
            "point_count": int(clean_artifact["point_count"]),
            "verified_pairs": 0,
            "track_mean": 0.0,
            "track_p90": 0.0,
            "track_max": 0.0,
            "reproj_median": float(clean_artifact["median_error"]),
            "reproj_p90": float(clean_artifact["p90_error"]),
            "merge_mode": None,
            "source_frames": [int(clean_artifact["frame_index"])],
            "multi_frame_database": False,
            "support_mean": float(clean_artifact["support_mean"]),
            "support_median": float(clean_artifact["support_median"]),
            "support_p90": float(clean_artifact["support_p90"]),
            "median_error": float(clean_artifact["median_error"]),
            "p90_error": float(clean_artifact["p90_error"]),
        }
    else:
        raise KeyError(f"Unsupported clean artifact schema: {artifact_meta}")
    return clean_metrics, artifact_metrics


def audit(
    *,
    min_clean_heldout_psnr: float,
    min_clean_heldout_ssim: float,
    min_track_p90: float,
    min_track_mean: float,
    min_point_count: int,
    require_raw_quality: bool = False,
    allow_raw_step0_acceptance: bool = False,
) -> dict[str, Any]:
    ex4dgs_best = load_json(EX4DGS_OUTPUT / "best_metrics.json")

    candidates = [
        clean_candidate_metrics(
            output_dir,
            artifact_meta,
            min_clean_heldout_psnr=min_clean_heldout_psnr,
            min_clean_heldout_ssim=min_clean_heldout_ssim,
        )
        for output_dir, artifact_meta in existing_clean_candidates(require_point_cloud=True)
    ]
    clean_metrics, artifact_metrics = max(candidates, key=lambda item: item[0]["heldout_eval_psnr"])
    selected_output_dir = ROOT / clean_metrics["output_dir"]
    selected_clean_config = load_json(selected_output_dir / "resolved_config.json")
    post_initial_rows = post_initial_paper_quality_rows(
        selected_output_dir,
        selected_clean_config,
        min_clean_heldout_psnr=min_clean_heldout_psnr,
        min_clean_heldout_ssim=min_clean_heldout_ssim,
    )
    post_initial_row = (
        max(post_initial_rows, key=lambda row: float(row["heldout_eval_psnr"])) if post_initial_rows else None
    )
    post_initial_raw_rows = post_initial_raw_quality_rows(
        selected_output_dir,
        selected_clean_config,
        min_clean_heldout_psnr=min_clean_heldout_psnr,
        min_clean_heldout_ssim=min_clean_heldout_ssim,
        allow_raw_step0_acceptance=allow_raw_step0_acceptance,
    )
    post_initial_raw_row = (
        max(
            post_initial_raw_rows,
            key=lambda row: (
                row.get("trainability_evidence") == "post_initial_state_delta",
                float(row["raw_heldout_eval_psnr"]),
            ),
        )
        if post_initial_raw_rows
        else None
    )
    ex4dgs_metrics = {
        "output_dir": str(EX4DGS_OUTPUT.relative_to(ROOT)),
        "step": int(ex4dgs_best["step"]),
        "heldout_eval_psnr": metric(ex4dgs_best, "heldout_eval_psnr"),
        "heldout_eval_l1": metric(ex4dgs_best, "heldout_eval_l1"),
        "heldout_eval_ssim": metric(ex4dgs_best, "heldout_eval_ssim"),
        "excluded_reason": "uses external pretrained EX4DGS point cloud, not paper-clean SfM/COLMAP init",
    }

    calibrated_checks = [
        check(
            OFFICIAL_FIXTURE.exists(),
            "official_cuda_warp_fixture_exists",
            str(OFFICIAL_FIXTURE.relative_to(ROOT)),
        ),
        check(
            clean_metrics["heldout_eval_psnr"] >= min_clean_heldout_psnr,
            "clean_heldout_psnr_threshold",
            {"actual": clean_metrics["heldout_eval_psnr"], "required": min_clean_heldout_psnr},
        ),
        check(
            clean_metrics["heldout_eval_ssim"] >= min_clean_heldout_ssim,
            "clean_heldout_ssim_threshold",
            {"actual": clean_metrics["heldout_eval_ssim"], "required": min_clean_heldout_ssim},
        ),
        check(
            clean_metrics["render_size"] >= 128 and clean_metrics["cells"] >= 1024 and clean_metrics["frames"] >= 16,
            "minimum_probe_scale",
            {
                "render_size": clean_metrics["render_size"],
                "cells": clean_metrics["cells"],
                "frames": clean_metrics["frames"],
            },
        ),
        check(
            clean_metrics["wandb_enabled"],
            "selected_clean_row_has_wandb_backing",
            {"wandb_enabled": clean_metrics["wandb_enabled"]},
        ),
        check(
            clean_metrics["init_point_cloud_path"] == artifact_metrics["output"],
            "clean_init_path_matches_artifact",
            {"config_path": clean_metrics["init_point_cloud_path"], "artifact_output": artifact_metrics["output"]},
        ),
        check(
            clean_metrics["eval_color_calibration"] == "none"
            or (
                clean_metrics["eval_color_calibration_artifact_exists"]
                and clean_metrics["uncalibrated_heldout_eval_psnr"] is not None
                and clean_metrics["uncalibrated_heldout_eval_ssim"] is not None
            ),
            "clean_eval_color_calibration_disclosed",
            {
                "mode": clean_metrics["eval_color_calibration"],
                "artifact": clean_metrics["eval_color_calibration_artifact"],
                "artifact_exists": clean_metrics["eval_color_calibration_artifact_exists"],
                "uncalibrated_heldout_eval_psnr": clean_metrics["uncalibrated_heldout_eval_psnr"],
                "uncalibrated_heldout_eval_ssim": clean_metrics["uncalibrated_heldout_eval_ssim"],
            },
        ),
        check(
            post_initial_row is not None,
            "clean_post_initial_paper_quality_row",
            {
                "selected_output_dir": clean_metrics["output_dir"],
                "required_step": "> 0",
                "required_psnr": min_clean_heldout_psnr,
                "required_ssim": min_clean_heldout_ssim,
                "row": post_initial_row,
            },
        ),
        check(
            bool(artifact_metrics.get("multi_frame_database", False))
            and artifact_metrics.get("merge_mode") != "concat_no_dedup",
            "clean_true_track_artifact",
            {
                "multi_frame_database": artifact_metrics.get("multi_frame_database"),
                "merge_mode": artifact_metrics.get("merge_mode"),
            },
        ),
        check(
            len(artifact_metrics.get("source_frames", [])) >= 4,
            "clean_multiframe_support",
            {"source_frames": artifact_metrics.get("source_frames", [])},
        ),
        check(
            artifact_metrics["point_count"] >= min_point_count,
            "clean_point_count_threshold",
            {"actual": artifact_metrics["point_count"], "required": min_point_count},
        ),
        check(
            artifact_metrics["track_mean"] >= min_track_mean and artifact_metrics["track_p90"] >= min_track_p90,
            "clean_long_track_threshold",
            {
                "actual_mean": artifact_metrics["track_mean"],
                "required_mean": min_track_mean,
                "actual_p90": artifact_metrics["track_p90"],
                "required_p90": min_track_p90,
                "actual_max": artifact_metrics["track_max"],
            },
        ),
        check(
            float(artifact_metrics.get("unique_camera_track_p90", 0.0)) >= 2.0,
            "clean_multiview_track_support",
            {
                "actual_camera_p90": artifact_metrics.get("unique_camera_track_p90"),
                "actual_camera_mean": artifact_metrics.get("unique_camera_track_mean"),
                "required_camera_p90": 2.0,
            },
        ),
        check(
            float(artifact_metrics.get("unique_frame_track_p90", 0.0)) >= 2.0,
            "clean_temporal_track_support",
            {
                "actual_frame_p90": artifact_metrics.get("unique_frame_track_p90"),
                "actual_frame_mean": artifact_metrics.get("unique_frame_track_mean"),
                "required_frame_p90": 2.0,
            },
        ),
        check(
            artifact_metrics["reproj_median"] <= 4.0 and artifact_metrics["reproj_p90"] <= 8.0,
            "clean_reprojection_quality",
            {
                "median": artifact_metrics["reproj_median"],
                "p90": artifact_metrics["reproj_p90"],
                "required_median_max": 4.0,
                "required_p90_max": 8.0,
            },
        ),
        check(
            artifact_metrics["verified_pairs"] >= 28,
            "clean_verified_pair_threshold",
            {"actual": artifact_metrics["verified_pairs"], "required": 28},
        ),
    ]
    raw_psnr = optional_float(clean_metrics["raw_heldout_eval_psnr"])
    raw_ssim = optional_float(clean_metrics["raw_heldout_eval_ssim"])
    raw_checks = [
        check(
            raw_psnr is not None and raw_psnr >= float(min_clean_heldout_psnr),
            "clean_raw_heldout_psnr_threshold",
            {
                "actual": raw_psnr,
                "required": min_clean_heldout_psnr,
                "source": clean_metrics["raw_quality_source"],
                "calibrated_heldout_eval_psnr": clean_metrics["heldout_eval_psnr"],
            },
        ),
        check(
            raw_ssim is not None and raw_ssim >= float(min_clean_heldout_ssim),
            "clean_raw_heldout_ssim_threshold",
            {
                "actual": raw_ssim,
                "required": min_clean_heldout_ssim,
                "source": clean_metrics["raw_quality_source"],
                "calibrated_heldout_eval_ssim": clean_metrics["heldout_eval_ssim"],
            },
        ),
        check(
            post_initial_raw_row is not None,
            "clean_post_initial_raw_quality_row",
            {
                "selected_output_dir": clean_metrics["output_dir"],
                "required_step": "> 0" if not allow_raw_step0_acceptance else ">= 0 explicit raw step0 acceptance",
                "required_raw_psnr": min_clean_heldout_psnr,
                "required_raw_ssim": min_clean_heldout_ssim,
                "row": post_initial_raw_row,
            },
        ),
    ]
    checks = calibrated_checks + (raw_checks if require_raw_quality else [])
    ok = all(item["passed"] for item in checks)
    missing_optional = missing_optional_clean_candidates()
    return {
        "ok": ok,
        "raw_quality_ok": bool(clean_metrics["raw_quality_ok"]),
        "calibrated_quality_ok": bool(clean_metrics["calibrated_quality_ok"]),
        "checks": checks,
        "raw_quality_checks": raw_checks,
        "require_raw_quality": bool(require_raw_quality),
        "allow_raw_step0_acceptance": bool(allow_raw_step0_acceptance),
        "clean_deepview_candidates": [
            {"run": candidate_metrics, "point_cloud": candidate_artifact}
            for candidate_metrics, candidate_artifact in candidates
        ],
        "missing_optional_clean_deepview_candidates": missing_optional,
        "selected_clean_deepview_candidate": clean_metrics,
        "selected_clean_post_initial_candidate": post_initial_row,
        "selected_clean_post_initial_raw_candidate": post_initial_raw_row,
        "clean_point_cloud": artifact_metrics,
        "external_init_reference": ex4dgs_metrics,
        "next_blockers": next_blockers(checks, missing_optional),
        "raw_quality_next_blockers": next_blockers(raw_checks, []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PowerFoam Metal paper-acceptance evidence.")
    parser.add_argument("--min-clean-heldout-psnr", type=float, default=13.0)
    parser.add_argument("--min-clean-heldout-ssim", type=float, default=0.15)
    parser.add_argument("--min-track-p90", type=float, default=3.0)
    parser.add_argument("--min-track-mean", type=float, default=2.5)
    parser.add_argument("--min-point-count", type=int, default=2000)
    parser.add_argument("--require-raw-quality", action="store_true")
    parser.add_argument("--allow-raw-step0-acceptance", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    report = audit(
        min_clean_heldout_psnr=float(args.min_clean_heldout_psnr),
        min_clean_heldout_ssim=float(args.min_clean_heldout_ssim),
        min_track_p90=float(args.min_track_p90),
        min_track_mean=float(args.min_track_mean),
        min_point_count=int(args.min_point_count),
        require_raw_quality=bool(args.require_raw_quality),
        allow_raw_step0_acceptance=bool(args.allow_raw_step0_acceptance),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"] and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
