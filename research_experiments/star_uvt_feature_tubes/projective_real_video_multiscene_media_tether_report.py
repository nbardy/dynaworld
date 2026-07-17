from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
STAR_UVT_ROOT = ROOT / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
for path in (ROOT, ROOT / "src" / "train", STAR_UVT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import star_uvt_feature_overfit_trainer as feature_overfit_trainer  # noqa: E402
from projective_interval_trainer_frame_scaling_benchmark import _row_from_payload  # noqa: E402
from projective_real_video_multiscene_trainer_matrix import (  # noqa: E402
    DEFAULT_SEGMENT_IDS,
    DEFAULT_SEGMENTS_MANIFEST,
    _apply_metal_tile_env,
    _base_config,
    _load_segments,
)
from projective_real_video_multiscene_quality_tether_report import (  # noqa: E402
    REQUIRED_GRADIENT_FLAGS,
    _max_abs_delta,
)
from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    has_projective_trace_cell_interval_backward_metal,
    has_projective_trace_cell_interval_metal,
)


DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_multiscene_media_tether"
)
CONTACT_SHEET_GUTTER_PX = 2
CONTACT_SHEET_PAYLOAD_LOSS_TOLERANCE = 2.5e-3
MEDIA_SCALAR_LOSS_TOLERANCE = 3.0e-8
MEDIA_SCALAR_PSNR_TOLERANCE = 1.0e-6


def _finite_float(value: Any, label: str, errors: list[str]) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        errors.append(f"{label} must be finite, got {value!r}")
        return 0.0
    return float(value)


def _finite_int(value: Any, label: str, errors: list[str]) -> int:
    if not isinstance(value, int):
        errors.append(f"{label} must be an integer, got {value!r}")
        return 0
    return int(value)


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rgb_pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.int16)


def _image_delta(lhs: Path, rhs: Path) -> dict[str, Any]:
    if not lhs.exists() or not rhs.exists():
        return {
            "files_exist": False,
            "shape": None,
            "pixel_count": 0,
            "max_abs_delta": math.inf,
            "mean_abs_delta": math.inf,
            "sha256_match": False,
            "lhs_sha256": _sha256_file(lhs),
            "rhs_sha256": _sha256_file(rhs),
        }
    left = _rgb_pixels(lhs)
    right = _rgb_pixels(rhs)
    if left.shape != right.shape:
        return {
            "files_exist": True,
            "shape": [list(left.shape), list(right.shape)],
            "pixel_count": 0,
            "max_abs_delta": math.inf,
            "mean_abs_delta": math.inf,
            "sha256_match": _sha256_file(lhs) == _sha256_file(rhs),
            "lhs_sha256": _sha256_file(lhs),
            "rhs_sha256": _sha256_file(rhs),
        }
    delta = np.abs(left - right)
    return {
        "files_exist": True,
        "shape": list(left.shape),
        "pixel_count": int(delta.size),
        "max_abs_delta": int(delta.max()) if delta.size else 0,
        "mean_abs_delta": float(delta.mean()) if delta.size else 0.0,
        "sha256_match": _sha256_file(lhs) == _sha256_file(rhs),
        "lhs_sha256": _sha256_file(lhs),
        "rhs_sha256": _sha256_file(rhs),
    }


def _contact_sheet_metric_row(path: Path, frame_count: int, payload_rgb_loss: Any) -> dict[str, Any]:
    if not path.exists():
        return {
            "contact_sheet_layout_valid": False,
            "contact_sheet_layout_error": "missing contact sheet",
            "contact_sheet_inferred_frame_count": int(frame_count),
            "contact_sheet_inferred_frame_height": 0,
            "contact_sheet_inferred_frame_width": 0,
            "contact_sheet_target_std": 0.0,
            "contact_sheet_pred_std": 0.0,
            "contact_sheet_target_mean": 0.0,
            "contact_sheet_pred_mean": 0.0,
            "contact_sheet_target_pred_mse": math.inf,
            "contact_sheet_target_pred_psnr": 0.0,
            "contact_sheet_payload_loss_abs_delta": math.inf,
        }
    pixels = _rgb_pixels(path).astype(np.float64) / 255.0
    sheet_height, sheet_width = int(pixels.shape[0]), int(pixels.shape[1])
    if frame_count <= 0:
        layout_error = f"invalid frame count {frame_count}"
    elif (sheet_width - (frame_count - 1) * CONTACT_SHEET_GUTTER_PX) % frame_count != 0:
        layout_error = "sheet width is not compatible with frame count and gutter"
    elif (sheet_height - CONTACT_SHEET_GUTTER_PX) % 2 != 0:
        layout_error = "sheet height is not compatible with two rows and gutter"
    else:
        layout_error = ""
    if layout_error:
        return {
            "contact_sheet_layout_valid": False,
            "contact_sheet_layout_error": layout_error,
            "contact_sheet_inferred_frame_count": int(frame_count),
            "contact_sheet_inferred_frame_height": 0,
            "contact_sheet_inferred_frame_width": 0,
            "contact_sheet_target_std": 0.0,
            "contact_sheet_pred_std": 0.0,
            "contact_sheet_target_mean": 0.0,
            "contact_sheet_pred_mean": 0.0,
            "contact_sheet_target_pred_mse": math.inf,
            "contact_sheet_target_pred_psnr": 0.0,
            "contact_sheet_payload_loss_abs_delta": math.inf,
        }
    frame_width = (sheet_width - (frame_count - 1) * CONTACT_SHEET_GUTTER_PX) // frame_count
    frame_height = (sheet_height - CONTACT_SHEET_GUTTER_PX) // 2
    target_tiles = []
    pred_tiles = []
    for frame_idx in range(frame_count):
        x0 = frame_idx * (frame_width + CONTACT_SHEET_GUTTER_PX)
        target_tiles.append(pixels[:frame_height, x0 : x0 + frame_width])
        pred_y0 = frame_height + CONTACT_SHEET_GUTTER_PX
        pred_tiles.append(pixels[pred_y0 : pred_y0 + frame_height, x0 : x0 + frame_width])
    target = np.stack(target_tiles, axis=0)
    pred = np.stack(pred_tiles, axis=0)
    mse = float(np.square(target - pred).mean())
    psnr = math.inf if mse <= 0.0 else -10.0 * math.log10(mse)
    payload_loss = _finite_float(payload_rgb_loss, "payload final_full_rgb_loss", [])
    return {
        "contact_sheet_layout_valid": True,
        "contact_sheet_layout_error": "",
        "contact_sheet_inferred_frame_count": int(frame_count),
        "contact_sheet_inferred_frame_height": int(frame_height),
        "contact_sheet_inferred_frame_width": int(frame_width),
        "contact_sheet_target_std": float(target.std()),
        "contact_sheet_pred_std": float(pred.std()),
        "contact_sheet_target_mean": float(target.mean()),
        "contact_sheet_pred_mean": float(pred.mean()),
        "contact_sheet_target_pred_mse": mse,
        "contact_sheet_target_pred_psnr": psnr,
        "contact_sheet_payload_loss_abs_delta": abs(mse - payload_loss),
    }


def _contact_sheet_path(out_dir: Path, scene_id: str, frames: int, policy: str) -> Path:
    return out_dir / "media" / f"{scene_id}_{int(frames)}f_{policy}.png"


def _run_media_case(
    *,
    scene: dict[str, Any],
    frames: int,
    policy: str,
    size: int,
    steps: int,
    refresh_every: int,
    tile_capacity: int,
    tube_count: int,
    support_guard_padding: float,
    support_guard_policy: str,
    support_guard_bisect_steps: int,
    support_stale_overshoot_epsilon: float,
    support_stale_tail_alpha_epsilon: float,
    out_json: Path,
    contact_sheet: Path,
    verbose_trainer_output: bool,
) -> dict[str, Any]:
    video_path = Path(scene["path"])
    scene_id = str(scene["segment_id"])
    cfg = _base_config(
        video_path=video_path,
        scene_id=scene_id,
        frames=frames,
        size=size,
        steps=steps,
        policy=policy,
        refresh_every=refresh_every,
        tile_capacity=tile_capacity,
        tube_count=tube_count,
        support_guard_padding=support_guard_padding,
        support_guard_policy=support_guard_policy,
        support_guard_bisect_steps=support_guard_bisect_steps,
        support_stale_overshoot_epsilon=support_stale_overshoot_epsilon,
        support_stale_tail_alpha_epsilon=support_stale_tail_alpha_epsilon,
        out_json=out_json,
    )
    cfg["output"]["contact_sheet"] = str(contact_sheet)
    cfg["output"]["contact_sheet_frames"] = int(frames)
    cfg["output"]["contact_sheet_mode"] = "linspace"
    _apply_metal_tile_env(cfg)
    started = time.perf_counter()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    if verbose_trainer_output:
        payload = feature_overfit_trainer.run_training(cfg)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            payload = feature_overfit_trainer.run_training(cfg)
    row = _row_from_payload(frames=frames, policy=policy, elapsed_sec=time.perf_counter() - started, payload=payload)
    row.update(
        {
            "scene_id": scene_id,
            "youtube_id": str(scene.get("youtube_id", "")),
            "title": str(scene.get("title", "")),
            "video_path": str(video_path),
            "source_video_exists": video_path.exists(),
            "motion_score": float(scene.get("motion_score") or 0.0),
            "scene_cut_count_in_source": int(scene.get("scene_cut_count_in_source") or 0),
            "case_json": str(out_json),
            "contact_sheet": str(contact_sheet),
            "contact_sheet_exists": contact_sheet.exists(),
            "contact_sheet_bytes": contact_sheet.stat().st_size if contact_sheet.exists() else 0,
            "contact_sheet_sha256": _sha256_file(contact_sheet),
            "contact_sheet_mode": "linspace",
            "contact_sheet_frames": int(frames),
            "media_render_ms": payload.get("media_render_ms"),
            "final_full_rgb_loss": payload.get("final_full_rgb_loss"),
            "final_full_rgb_psnr": payload.get("final_full_rgb_psnr"),
            "losses": list(payload.get("losses") or []),
            "rgb_losses": list(payload.get("rgb_losses") or []),
            "start_psnr": payload.get("start_psnr"),
            "end_psnr": payload.get("end_psnr"),
            "gradient_flags": {
                flag: payload.get(flag) is True
                for flag in REQUIRED_GRADIENT_FLAGS
            },
        }
    )
    row.update(
        _contact_sheet_metric_row(
            contact_sheet,
            int(frames),
            payload.get("final_full_rgb_loss"),
        )
    )
    return row


def _case_rows_for(report: dict[str, Any], scene_id: str, policy: str) -> list[dict[str, Any]]:
    return [
        row
        for row in report.get("case_rows", [])
        if isinstance(row, dict) and row.get("scene_id") == scene_id and row.get("policy") == policy
    ]


def _pair_row(cadence: dict[str, Any], measured: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    image = _image_delta(Path(str(cadence["contact_sheet"])), Path(str(measured["contact_sheet"])))
    max_loss_delta = _max_abs_delta(
        list(cadence.get("losses") or []),
        list(measured.get("losses") or []),
        "loss",
        errors,
    )
    max_rgb_loss_delta = _max_abs_delta(
        list(cadence.get("rgb_losses") or []),
        list(measured.get("rgb_losses") or []),
        "rgb_loss",
        errors,
    )
    cadence_full_rgb_loss = _finite_float(cadence.get("final_full_rgb_loss"), "cadence final_full_rgb_loss", errors)
    measured_full_rgb_loss = _finite_float(measured.get("final_full_rgb_loss"), "measured final_full_rgb_loss", errors)
    cadence_full_rgb_psnr = _finite_float(cadence.get("final_full_rgb_psnr"), "cadence final_full_rgb_psnr", errors)
    measured_full_rgb_psnr = _finite_float(measured.get("final_full_rgb_psnr"), "measured final_full_rgb_psnr", errors)
    measured_start_loss = _finite_float(measured.get("start_loss"), "measured start_loss", errors)
    measured_end_loss = _finite_float(measured.get("end_loss"), "measured end_loss", errors)
    measured_start_psnr = _finite_float(measured.get("start_psnr"), "measured start_psnr", errors)
    measured_end_psnr = _finite_float(measured.get("end_psnr"), "measured end_psnr", errors)
    missing_grad_flags = [
        flag
        for flag in REQUIRED_GRADIENT_FLAGS
        if cadence.get("gradient_flags", {}).get(flag) is not True
        or measured.get("gradient_flags", {}).get(flag) is not True
    ]
    cadence_sheet_mse = _finite_float(
        cadence.get("contact_sheet_target_pred_mse"),
        "cadence contact_sheet_target_pred_mse",
        errors,
    )
    measured_sheet_mse = _finite_float(
        measured.get("contact_sheet_target_pred_mse"),
        "measured contact_sheet_target_pred_mse",
        errors,
    )
    cadence_payload_delta = _finite_float(
        cadence.get("contact_sheet_payload_loss_abs_delta"),
        "cadence contact_sheet_payload_loss_abs_delta",
        errors,
    )
    measured_payload_delta = _finite_float(
        measured.get("contact_sheet_payload_loss_abs_delta"),
        "measured contact_sheet_payload_loss_abs_delta",
        errors,
    )
    return {
        "scene_id": str(cadence["scene_id"]),
        "frames": int(cadence["frames"]),
        "cadence_case_json": str(cadence["case_json"]),
        "measured_case_json": str(measured["case_json"]),
        "cadence_contact_sheet": str(cadence["contact_sheet"]),
        "measured_contact_sheet": str(measured["contact_sheet"]),
        "contact_sheet_files_exist": bool(image["files_exist"]),
        "contact_sheet_shape": image["shape"],
        "contact_sheet_pixel_count": int(image["pixel_count"]),
        "max_abs_contact_sheet_delta": image["max_abs_delta"],
        "mean_abs_contact_sheet_delta": image["mean_abs_delta"],
        "contact_sheet_sha256_match": bool(image["sha256_match"]),
        "cadence_contact_sheet_sha256": image["lhs_sha256"],
        "measured_contact_sheet_sha256": image["rhs_sha256"],
        "curve_length": len(cadence.get("losses") or []),
        "max_abs_loss_curve_delta": max_loss_delta,
        "max_abs_rgb_loss_curve_delta": max_rgb_loss_delta,
        "final_full_rgb_loss_abs_delta": abs(measured_full_rgb_loss - cadence_full_rgb_loss),
        "final_full_rgb_psnr_abs_delta": abs(measured_full_rgb_psnr - cadence_full_rgb_psnr),
        "cadence_contact_sheet_layout_valid": cadence.get("contact_sheet_layout_valid") is True,
        "measured_contact_sheet_layout_valid": measured.get("contact_sheet_layout_valid") is True,
        "cadence_contact_sheet_target_pred_mse": cadence_sheet_mse,
        "measured_contact_sheet_target_pred_mse": measured_sheet_mse,
        "contact_sheet_target_pred_mse_abs_delta": abs(measured_sheet_mse - cadence_sheet_mse),
        "cadence_contact_sheet_payload_loss_abs_delta": cadence_payload_delta,
        "measured_contact_sheet_payload_loss_abs_delta": measured_payload_delta,
        "max_contact_sheet_payload_loss_abs_delta": max(cadence_payload_delta, measured_payload_delta),
        "measured_loss_decrease": measured_start_loss - measured_end_loss,
        "measured_psnr_gain": measured_end_psnr - measured_start_psnr,
        "cadence_pass": cadence.get("pass") is True,
        "measured_pass": measured.get("pass") is True,
        "cadence_media_render_ms": cadence.get("media_render_ms"),
        "measured_media_render_ms": measured.get("media_render_ms"),
        "cadence_cache_rebuilds": cadence.get("projective_interval_cache_rebuilds"),
        "measured_cache_rebuilds": measured.get("projective_interval_cache_rebuilds"),
        "cadence_no_first_step_ms": cadence.get("no_first_step_ms"),
        "measured_no_first_step_ms": measured.get("no_first_step_ms"),
        "cadence_tile_overflow_sum": cadence.get("tile_overflow_sum"),
        "measured_tile_overflow_sum": measured.get("tile_overflow_sum"),
        "cadence_fallback_marks": cadence.get("projective_interval_cache_fallback_marks"),
        "measured_fallback_marks": measured.get("projective_interval_cache_fallback_marks"),
        "cadence_visibility_stratifications": cadence.get("projective_interval_cache_visibility_stratifications"),
        "measured_visibility_stratifications": measured.get("projective_interval_cache_visibility_stratifications"),
        "missing_gradient_flags": missing_grad_flags,
        "row_errors": errors,
    }


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    rows = report["rows"]
    case_rows = report["case_rows"]
    measured_rows = [row for row in case_rows if row["policy"] == "measured"]
    cadence_rows = [row for row in case_rows if row["policy"] == "cadence"]
    no_first_ratios = [
        float(row["measured_no_first_step_ms"]) / float(row["cadence_no_first_step_ms"])
        for row in rows
        if float(row["cadence_no_first_step_ms"]) > 0.0
    ]
    rebuild_ratios = [
        float(row["measured_cache_rebuilds"]) / float(row["cadence_cache_rebuilds"])
        for row in rows
        if int(row["cadence_cache_rebuilds"]) > 0
    ]
    return {
        "scene_count": len({row["scene_id"] for row in rows}),
        "pair_count": len(rows),
        "case_row_count": len(case_rows),
        "measured_row_count": len(measured_rows),
        "cadence_row_count": len(cadence_rows),
        "distinct_youtube_id_count": len({scene["youtube_id"] for scene in report["scenes"]}),
        "all_source_videos_exist": all(bool(scene["source_video_exists"]) for scene in report["scenes"]),
        "all_case_rows_pass": all(bool(row["pass"]) for row in case_rows),
        "all_contact_sheets_exist": all(bool(row["contact_sheet_files_exist"]) for row in rows),
        "all_contact_sheet_layouts_valid": all(bool(row["contact_sheet_layout_valid"]) for row in case_rows),
        "all_contact_sheet_pixels_match_cadence": all(int(row["max_abs_contact_sheet_delta"]) == 0 for row in rows),
        "all_contact_sheet_hashes_match_cadence": all(bool(row["contact_sheet_sha256_match"]) for row in rows),
        "all_contact_sheet_metrics_match_payload": all(
            float(row["contact_sheet_payload_loss_abs_delta"]) <= CONTACT_SHEET_PAYLOAD_LOSS_TOLERANCE
            for row in case_rows
        ),
        "all_contact_sheet_rows_nontrivial": all(
            float(row["contact_sheet_target_std"]) > 1.0e-6 and float(row["contact_sheet_pred_std"]) > 1.0e-6
            for row in case_rows
        ),
        "all_loss_curves_match_cadence": all(
            float(row["max_abs_loss_curve_delta"]) <= MEDIA_SCALAR_LOSS_TOLERANCE for row in rows
        ),
        "all_rgb_loss_curves_match_cadence": all(
            float(row["max_abs_rgb_loss_curve_delta"]) <= MEDIA_SCALAR_LOSS_TOLERANCE for row in rows
        ),
        "all_final_full_rgb_losses_match_cadence": all(
            float(row["final_full_rgb_loss_abs_delta"]) <= MEDIA_SCALAR_LOSS_TOLERANCE for row in rows
        ),
        "all_final_full_rgb_psnr_matches_cadence": all(
            float(row["final_full_rgb_psnr_abs_delta"]) <= MEDIA_SCALAR_PSNR_TOLERANCE for row in rows
        ),
        "all_gradient_flags_present": all(not row["missing_gradient_flags"] for row in rows),
        "all_measured_loss_decreases": all(float(row["measured_loss_decrease"]) > 0.0 for row in rows),
        "all_measured_psnr_improves": all(float(row["measured_psnr_gain"]) > 0.0 for row in rows),
        "all_measured_media_rendered": all(float(row["measured_media_render_ms"]) > 0.0 for row in rows),
        "all_cadence_media_rendered": all(float(row["cadence_media_render_ms"]) > 0.0 for row in rows),
        "all_rows_no_overflow": all(
            int(row["cadence_tile_overflow_sum"]) == 0 and int(row["measured_tile_overflow_sum"]) == 0
            for row in rows
        ),
        "all_rows_fallback_free": all(
            int(row["cadence_fallback_marks"]) == 0 and int(row["measured_fallback_marks"]) == 0
            for row in rows
        ),
        "all_rows_visibility_stratification_free": all(
            int(row["cadence_visibility_stratifications"]) == 0
            and int(row["measured_visibility_stratifications"]) == 0
            for row in rows
        ),
        "max_abs_contact_sheet_delta": max(int(row["max_abs_contact_sheet_delta"]) for row in rows),
        "max_mean_abs_contact_sheet_delta": max(float(row["mean_abs_contact_sheet_delta"]) for row in rows),
        "max_contact_sheet_target_pred_mse_delta": max(
            float(row["contact_sheet_target_pred_mse_abs_delta"]) for row in rows
        ),
        "max_contact_sheet_payload_loss_abs_delta": max(
            float(row["contact_sheet_payload_loss_abs_delta"]) for row in case_rows
        ),
        "max_abs_loss_curve_delta": max(float(row["max_abs_loss_curve_delta"]) for row in rows),
        "max_abs_rgb_loss_curve_delta": max(float(row["max_abs_rgb_loss_curve_delta"]) for row in rows),
        "max_final_full_rgb_loss_abs_delta": max(float(row["final_full_rgb_loss_abs_delta"]) for row in rows),
        "max_final_full_rgb_psnr_abs_delta": max(float(row["final_full_rgb_psnr_abs_delta"]) for row in rows),
        "min_measured_loss_decrease": min(float(row["measured_loss_decrease"]) for row in rows),
        "min_measured_psnr_gain": min(float(row["measured_psnr_gain"]) for row in rows),
        "min_contact_sheet_pixel_count": min(int(row["contact_sheet_pixel_count"]) for row in rows),
        "min_contact_sheet_target_std": min(float(row["contact_sheet_target_std"]) for row in case_rows),
        "min_contact_sheet_pred_std": min(float(row["contact_sheet_pred_std"]) for row in case_rows),
        "min_contact_sheet_target_pred_mse": min(float(row["contact_sheet_target_pred_mse"]) for row in case_rows),
        "max_measured_vs_cadence_no_first_step_ms_ratio": max(no_first_ratios),
        "max_measured_vs_cadence_rebuild_ratio": max(rebuild_ratios),
    }


def verify_real_video_multiscene_media_tether_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_real_video_multiscene_media_tether":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    theory_contract = report.get("theory_contract")
    if (
        not isinstance(theory_contract, str)
        or "does not prove broad real-scene quality acceptance" not in theory_contract
        or "actual contact-sheet media writer" not in theory_contract
        or "cadence full-rebuild reference" not in theory_contract
    ):
        errors.append("theory_contract must preserve the media-tether scope")

    scenes = report.get("scenes")
    rows = report.get("rows")
    case_rows = report.get("case_rows")
    if not isinstance(scenes, list) or len(scenes) < 3:
        errors.append("scenes must include at least three source-distinct real videos")
        return errors
    if len({str(scene.get("youtube_id") or "") for scene in scenes if isinstance(scene, dict)}) != len(scenes):
        errors.append("scene youtube ids must be source-distinct")
    if not isinstance(case_rows, list) or len(case_rows) != 2 * len(scenes):
        errors.append("case_rows must contain one cadence and one measured case per scene")
        return errors
    if not isinstance(rows, list) or len(rows) != len(scenes):
        errors.append("rows must contain one media pair per scene")
        return errors

    for scene in scenes:
        if not isinstance(scene, dict):
            errors.append("scene row must be an object")
            continue
        scene_id = str(scene.get("scene_id") or "")
        if scene.get("source_video_exists") is not True:
            errors.append(f"scene {scene_id} source video must exist")
        if _finite_float(scene.get("motion_score"), f"{scene_id} motion_score", errors) <= 0.0:
            errors.append(f"scene {scene_id} must retain positive motion score")
        if len(_case_rows_for(report, scene_id, "cadence")) != 1 or len(_case_rows_for(report, scene_id, "measured")) != 1:
            errors.append(f"scene {scene_id} must have one cadence and one measured case row")

    for row in case_rows:
        if not isinstance(row, dict):
            errors.append("case row must be an object")
            continue
        prefix = f"{row.get('scene_id')} {row.get('policy')}"
        if row.get("pass") is not True:
            errors.append(f"{prefix} case must pass")
        if row.get("contact_sheet_exists") is not True:
            errors.append(f"{prefix} contact sheet must exist")
        if _finite_int(row.get("contact_sheet_bytes"), f"{prefix} contact_sheet_bytes", errors) <= 0:
            errors.append(f"{prefix} contact sheet bytes must be positive")
        if _finite_float(row.get("media_render_ms"), f"{prefix} media_render_ms", errors) <= 0.0:
            errors.append(f"{prefix} media_render_ms must be positive")
        if row.get("contact_sheet_layout_valid") is not True:
            errors.append(f"{prefix} contact sheet layout must be valid: {row.get('contact_sheet_layout_error')}")
        if _finite_int(
            row.get("contact_sheet_inferred_frame_count"),
            f"{prefix} contact_sheet_inferred_frame_count",
            errors,
        ) != _finite_int(row.get("contact_sheet_frames"), f"{prefix} contact_sheet_frames", errors):
            errors.append(f"{prefix} contact sheet inferred frame count must match requested frames")
        if (
            _finite_int(
                row.get("contact_sheet_inferred_frame_height"),
                f"{prefix} contact_sheet_inferred_frame_height",
                errors,
            )
            <= 0
        ):
            errors.append(f"{prefix} contact sheet inferred frame height must be positive")
        if (
            _finite_int(
                row.get("contact_sheet_inferred_frame_width"),
                f"{prefix} contact_sheet_inferred_frame_width",
                errors,
            )
            <= 0
        ):
            errors.append(f"{prefix} contact sheet inferred frame width must be positive")
        if _finite_float(row.get("contact_sheet_target_std"), f"{prefix} target row std", errors) <= 1.0e-6:
            errors.append(f"{prefix} contact sheet target row must be nontrivial")
        if _finite_float(row.get("contact_sheet_pred_std"), f"{prefix} pred row std", errors) <= 1.0e-6:
            errors.append(f"{prefix} contact sheet pred row must be nontrivial")
        if _finite_float(row.get("contact_sheet_target_pred_mse"), f"{prefix} target/pred MSE", errors) <= 0.0:
            errors.append(f"{prefix} contact sheet target/pred MSE must be positive")
        if _finite_float(row.get("contact_sheet_target_pred_psnr"), f"{prefix} target/pred PSNR", errors) <= 0.0:
            errors.append(f"{prefix} contact sheet target/pred PSNR must be positive")
        if (
            _finite_float(
                row.get("contact_sheet_payload_loss_abs_delta"),
                f"{prefix} contact-sheet payload loss delta",
                errors,
            )
            > CONTACT_SHEET_PAYLOAD_LOSS_TOLERANCE
        ):
            errors.append(f"{prefix} contact-sheet pixel MSE must match payload final RGB loss")

    for row in rows:
        if not isinstance(row, dict):
            errors.append("pair row must be an object")
            continue
        prefix = f"{row.get('scene_id')} {row.get('frames')}f"
        if row.get("contact_sheet_files_exist") is not True:
            errors.append(f"{prefix} contact-sheet files must exist")
        if _finite_int(row.get("contact_sheet_pixel_count"), f"{prefix} pixel count", errors) <= 0:
            errors.append(f"{prefix} contact sheet must have pixels")
        if _finite_int(row.get("max_abs_contact_sheet_delta"), f"{prefix} image delta", errors) != 0:
            errors.append(f"{prefix} measured contact sheet must pixel-match cadence")
        if _finite_float(row.get("mean_abs_contact_sheet_delta"), f"{prefix} mean image delta", errors) != 0.0:
            errors.append(f"{prefix} measured contact sheet mean delta must be zero")
        if row.get("cadence_contact_sheet_layout_valid") is not True:
            errors.append(f"{prefix} cadence contact sheet layout must be valid")
        if row.get("measured_contact_sheet_layout_valid") is not True:
            errors.append(f"{prefix} measured contact sheet layout must be valid")
        if (
            _finite_float(
                row.get("contact_sheet_target_pred_mse_abs_delta"),
                f"{prefix} contact-sheet target/pred MSE delta",
                errors,
            )
            > 1.0e-12
        ):
            errors.append(f"{prefix} contact-sheet target/pred MSE must match cadence")
        if (
            _finite_float(
                row.get("max_contact_sheet_payload_loss_abs_delta"),
                f"{prefix} contact-sheet payload loss delta",
                errors,
            )
            > CONTACT_SHEET_PAYLOAD_LOSS_TOLERANCE
        ):
            errors.append(f"{prefix} contact-sheet pixel MSE must match payload final RGB loss")
        if row.get("row_errors"):
            errors.append(f"{prefix} row errors must be empty: {row.get('row_errors')}")
        if row.get("missing_gradient_flags"):
            errors.append(f"{prefix} must preserve all required gradient flags: {row.get('missing_gradient_flags')}")
        if row.get("cadence_pass") is not True or row.get("measured_pass") is not True:
            errors.append(f"{prefix} cadence and measured cases must pass")
        if _finite_int(row.get("curve_length"), f"{prefix} curve length", errors) < 2:
            errors.append(f"{prefix} loss curve must include multiple steps")
        if (
            _finite_float(row.get("max_abs_loss_curve_delta"), f"{prefix} loss curve delta", errors)
            > MEDIA_SCALAR_LOSS_TOLERANCE
        ):
            errors.append(f"{prefix} measured loss curve must match cadence")
        if (
            _finite_float(row.get("max_abs_rgb_loss_curve_delta"), f"{prefix} rgb curve delta", errors)
            > MEDIA_SCALAR_LOSS_TOLERANCE
        ):
            errors.append(f"{prefix} measured RGB loss curve must match cadence")
        if (
            _finite_float(
                row.get("final_full_rgb_loss_abs_delta"),
                f"{prefix} final_full_rgb_loss delta",
                errors,
            )
            > MEDIA_SCALAR_LOSS_TOLERANCE
        ):
            errors.append(f"{prefix} final media RGB loss must match cadence")
        if (
            _finite_float(
                row.get("final_full_rgb_psnr_abs_delta"),
                f"{prefix} final_full_rgb_psnr delta",
                errors,
            )
            > MEDIA_SCALAR_PSNR_TOLERANCE
        ):
            errors.append(f"{prefix} final media RGB PSNR must match cadence")
        if _finite_float(row.get("measured_loss_decrease"), f"{prefix} measured loss decrease", errors) <= 0.0:
            errors.append(f"{prefix} measured loss must decrease")
        if _finite_float(row.get("measured_psnr_gain"), f"{prefix} measured psnr gain", errors) <= 0.0:
            errors.append(f"{prefix} measured PSNR must improve")
        if _finite_int(row.get("measured_cache_rebuilds"), f"{prefix} measured rebuilds", errors) >= _finite_int(
            row.get("cadence_cache_rebuilds"),
            f"{prefix} cadence rebuilds",
            errors,
        ):
            errors.append(f"{prefix} measured rebuilds must be below cadence")
        for key in (
            "cadence_tile_overflow_sum",
            "measured_tile_overflow_sum",
            "cadence_fallback_marks",
            "measured_fallback_marks",
            "cadence_visibility_stratifications",
            "measured_visibility_stratifications",
        ):
            if _finite_int(row.get(key), f"{prefix} {key}", errors) != 0:
                errors.append(f"{prefix} {key} must be zero")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        return errors
    try:
        expected = summarize(report)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        errors.append(f"summary could not be recomputed: {exc}")
        return errors
    for key, expected_value in expected.items():
        actual = summary.get(key)
        if isinstance(expected_value, float):
            if not isinstance(actual, int | float) or abs(float(actual) - expected_value) > 1.0e-9:
                errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")
        elif actual != expected_value:
            errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")
    if summary.get("all_contact_sheet_pixels_match_cadence") is not True:
        errors.append("media tether must pixel-match measured and cadence contact sheets")
    if summary.get("all_contact_sheet_layouts_valid") is not True:
        errors.append("media tether must preserve valid contact-sheet row layout")
    if summary.get("all_contact_sheet_metrics_match_payload") is not True:
        errors.append("media tether must match contact-sheet pixel MSE to payload final RGB loss")
    if summary.get("all_contact_sheet_rows_nontrivial") is not True:
        errors.append("media tether must contain nontrivial target and prediction contact-sheet rows")
    if summary.get("all_loss_curves_match_cadence") is not True:
        errors.append("media tether must match measured and cadence loss curves")
    if summary.get("all_final_full_rgb_losses_match_cadence") is not True:
        errors.append("media tether must match final full RGB media losses")
    if summary.get("all_gradient_flags_present") is not True:
        errors.append("media tether must preserve all gradient flags")
    return errors


def assert_real_video_multiscene_media_tether_report(report: dict[str, Any]) -> None:
    errors = verify_real_video_multiscene_media_tether_report(report)
    if errors:
        raise AssertionError("real-video multiscene media tether failed:\n- " + "\n- ".join(errors))


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if not DEFAULT_SEGMENTS_MANIFEST.exists():
        return {"status": "skipped", "reason": f"missing segment manifest: {DEFAULT_SEGMENTS_MANIFEST}", "rows": []}
    if not torch.backends.mps.is_available():
        return {"status": "skipped", "reason": "MPS unavailable", "rows": []}
    if not has_projective_trace_cell_interval_metal() or not has_projective_trace_cell_interval_backward_metal():
        return {"status": "skipped", "reason": "projective interval Metal ops unavailable", "rows": []}

    segments = _load_segments(DEFAULT_SEGMENTS_MANIFEST)
    requested_ids = tuple(args.segment_ids or DEFAULT_SEGMENT_IDS)
    scenes = [segments[segment_id] for segment_id in requested_ids]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    case_rows: list[dict[str, Any]] = []
    for scene in scenes:
        for policy in ("cadence", "measured"):
            scene_id = str(scene["segment_id"])
            case_rows.append(
                _run_media_case(
                    scene=scene,
                    frames=int(args.frames),
                    policy=policy,
                    size=int(args.size),
                    steps=int(args.steps),
                    refresh_every=int(args.refresh_every),
                    tile_capacity=int(args.tile_capacity),
                    tube_count=int(args.tube_count),
                    support_guard_padding=float(args.support_guard_padding),
                    support_guard_policy=str(args.support_guard_policy),
                    support_guard_bisect_steps=int(args.support_guard_bisect_steps),
                    support_stale_overshoot_epsilon=float(args.support_stale_overshoot_epsilon),
                    support_stale_tail_alpha_epsilon=float(args.support_stale_tail_alpha_epsilon),
                    out_json=args.out_dir / "cases" / f"{scene_id}_{int(args.frames)}f_{policy}.json",
                    contact_sheet=_contact_sheet_path(args.out_dir, scene_id, int(args.frames), policy),
                    verbose_trainer_output=bool(args.verbose_trainer_output),
                )
            )
    rows = []
    for scene in scenes:
        scene_id = str(scene["segment_id"])
        cadence = _case_rows_for({"case_rows": case_rows}, scene_id, "cadence")[0]
        measured = _case_rows_for({"case_rows": case_rows}, scene_id, "measured")[0]
        rows.append(_pair_row(cadence, measured))
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_multiscene_media_tether",
        "base_domain": "checked-in source-distinct real-video segments through actual contact-sheet media writer",
        "theory_contract": (
            "This report does not prove broad real-scene quality acceptance. It tethers the measured live-cache "
            "projective-interval path to the cadence full-rebuild reference through the actual contact-sheet media "
            "writer, final full-RGB media loss, loss curves, PSNR, and gradient-flow flags."
        ),
        "segments_manifest": str(DEFAULT_SEGMENTS_MANIFEST),
        "frames": int(args.frames),
        "size": int(args.size),
        "steps": int(args.steps),
        "refresh_every": int(args.refresh_every),
        "tile_capacity": int(args.tile_capacity),
        "tube_count": int(args.tube_count),
        "support_guard_padding": float(args.support_guard_padding),
        "support_guard_policy": str(args.support_guard_policy),
        "support_guard_bisect_steps": int(args.support_guard_bisect_steps),
        "support_stale_overshoot_epsilon": float(args.support_stale_overshoot_epsilon),
        "support_stale_tail_alpha_epsilon": float(args.support_stale_tail_alpha_epsilon),
        "contact_sheet_mode": "linspace",
        "required_gradient_flags": list(REQUIRED_GRADIENT_FLAGS),
        "scenes": [
            {
                "scene_id": str(scene["segment_id"]),
                "youtube_id": str(scene.get("youtube_id", "")),
                "title": str(scene.get("title", "")),
                "video_path": str(scene["path"]),
                "source_video_exists": Path(scene["path"]).exists(),
                "motion_score": float(scene.get("motion_score") or 0.0),
                "scene_cut_count_in_source": int(scene.get("scene_cut_count_in_source") or 0),
            }
            for scene in scenes
        ],
        "case_rows": case_rows,
        "rows": rows,
    }
    report["summary"] = summarize(report)
    errors = verify_real_video_multiscene_media_tether_report(report)
    if errors:
        report["status"] = "failed"
        report["errors"] = errors
    return report


def write_report(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--segment-id", action="append", dest="segment_ids")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--refresh-every", type=int, default=2)
    parser.add_argument("--tile-capacity", type=int, default=128)
    parser.add_argument("--tube-count", type=int, default=128)
    parser.add_argument("--support-guard-padding", type=float, default=1.0)
    parser.add_argument("--support-guard-policy", default="slack_budgeted")
    parser.add_argument("--support-guard-bisect-steps", type=int, default=8)
    parser.add_argument("--support-stale-overshoot-epsilon", type=float, default=0.0)
    parser.add_argument("--support-stale-tail-alpha-epsilon", type=float, default=0.001)
    parser.add_argument("--verbose-trainer-output", action="store_true")
    parser.add_argument("--verify-report", type=Path)
    args = parser.parse_args()

    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_real_video_multiscene_media_tether_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_benchmark(args)
    if report.get("status") == "ok":
        assert_real_video_multiscene_media_tether_report(report)
    path = write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
