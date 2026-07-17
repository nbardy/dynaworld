from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
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

import torch  # noqa: E402

import star_uvt_feature_overfit_trainer as feature_overfit_trainer  # noqa: E402
from projective_interval_trainer_frame_scaling_benchmark import _row_from_payload  # noqa: E402
from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    has_projective_trace_cell_interval_backward_metal,
    has_projective_trace_cell_interval_metal,
)


DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix"
)
DEFAULT_SEGMENTS_MANIFEST = ROOT / "data" / "youtube_scene_distinct" / "candidates" / "segments_manifest.jsonl"
DEFAULT_SEGMENT_IDS = (
    "Bq4rmeIvJbs_seg_000",
    "Iagm3K8QtFw_seg_000",
    "KUDJ8HDFVQo_seg_000",
)


def _load_segments(path: Path = DEFAULT_SEGMENTS_MANIFEST) -> dict[str, dict[str, Any]]:
    segments: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        segment_id = str(item["segment_id"])
        segments[segment_id] = item
    return segments


def _base_config(
    *,
    video_path: Path,
    scene_id: str,
    frames: int,
    size: int,
    steps: int,
    policy: str,
    refresh_every: int,
    tile_capacity: int,
    tube_count: int,
    support_guard_padding: float,
    support_guard_policy: str,
    support_guard_bisect_steps: int,
    support_stale_overshoot_epsilon: float,
    support_stale_tail_alpha_epsilon: float,
    out_json: Path,
    trace_global_steps: tuple[int, ...] = (),
) -> dict[str, Any]:
    return {
        "data": {
            "video_path": str(video_path),
            "start_seconds": None,
            "fps": None,
            "duration_seconds": None,
            "image_crop_mode": "center",
            "target_size": int(size),
            "max_frames": int(frames),
        },
        "train": {
            "steps": int(steps),
            "lr": 0.01,
            "device": "mps",
            "seed": 29,
            "frame_chunk_size": None,
            "trace_global_steps": [int(step) for step in trace_global_steps],
            "require_loss_decrease": False,
            "require_gradient_flow": False,
            "require_no_tile_overflow": False,
        },
        "feature_uvt": {
            "tube_count": int(tube_count),
            "feature_dim": 3,
            "tile_t": 2,
            "tile_capacity": int(tile_capacity),
            "alpha_threshold": 0.01,
            "max_alpha": 0.99,
            "render_mode": "feature_direct_atomic",
            "projective_interval": {
                "enabled": True,
                "sigma_px": 2.0,
                "tile_size": 8,
                "uv_padding": 8.0,
                "support_guard_padding": float(support_guard_padding),
                "support_guard_policy": str(support_guard_policy),
                "support_guard_bisect_steps": int(support_guard_bisect_steps),
                "support_stale_overshoot_epsilon": float(support_stale_overshoot_epsilon),
                "support_stale_tail_alpha_epsilon": float(support_stale_tail_alpha_epsilon),
                "refresh_every": int(refresh_every),
                "refresh_policy": str(policy),
                "fallback_render_mode": "mixed",
            },
        },
        "colorize": {
            "hidden_dim": None,
            "activation": "sigmoid",
            "pre_norm": False,
            "weight_init": "kaiming",
            "weight_init_gain": 1.0,
        },
        "output": {
            "out_json": str(out_json),
            "contact_sheet": None,
            "contact_sheet_frames": int(frames),
            "contact_sheet_mode": "grid",
            "side_by_side_video": None,
            "side_by_side_fps": 8,
        },
        "logging": {
            "wandb_enabled": False,
            "wandb_project": "unit",
            "wandb_run_name": f"projective-real-video-multiscene-{scene_id}-{policy}",
            "wandb_tags": [],
            "wandb_mode": None,
        },
    }


def _apply_metal_tile_env(cfg: dict[str, Any]) -> None:
    backend_cfg = cfg["feature_uvt"]["projective_interval"]
    os.environ["STAR_UVT_TILE_X"] = str(int(backend_cfg["tile_size"]))
    os.environ["STAR_UVT_TILE_Y"] = str(int(backend_cfg["tile_size"]))
    os.environ["STAR_UVT_TILE_T"] = str(int(cfg["feature_uvt"]["tile_t"]))
    os.environ["STAR_UVT_TILE_CAPACITY"] = str(int(cfg["feature_uvt"]["tile_capacity"]))


def run_case(
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
    verbose_trainer_output: bool,
    trace_global_steps: tuple[int, ...] = (),
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
        trace_global_steps=trace_global_steps,
    )
    _apply_metal_tile_env(cfg)
    started = time.perf_counter()
    out_json.parent.mkdir(parents=True, exist_ok=True)
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
            "support_guard_padding": float(support_guard_padding),
            "support_guard_policy": str(support_guard_policy),
            "support_guard_bisect_steps": int(support_guard_bisect_steps),
            "support_stale_overshoot_epsilon": float(support_stale_overshoot_epsilon),
            "support_stale_tail_alpha_epsilon": float(support_stale_tail_alpha_epsilon),
            "projective_interval_effective_support_uv_padding": payload.get(
                "projective_interval_effective_support_uv_padding"
            ),
            "projective_interval_cache_last_support_missing_tile_pairs": payload.get(
                "projective_interval_cache_last_support_missing_tile_pairs"
            ),
            "projective_interval_cache_last_support_max_overshoot_px": payload.get(
                "projective_interval_cache_last_support_max_overshoot_px"
            ),
            "projective_interval_cache_max_support_max_overshoot_px": payload.get(
                "projective_interval_cache_max_support_max_overshoot_px"
            ),
            "projective_interval_cache_last_support_tail_alpha_bound": payload.get(
                "projective_interval_cache_last_support_tail_alpha_bound"
            ),
            "projective_interval_cache_max_support_tail_alpha_bound": payload.get(
                "projective_interval_cache_max_support_tail_alpha_bound"
            ),
        }
    )
    return row


def _rows_for_policy(report: dict[str, Any], scene_id: str, policy: str) -> list[dict[str, Any]]:
    return sorted(
        [
            row
            for row in report.get("rows", [])
            if isinstance(row, dict) and row.get("scene_id") == scene_id and row.get("policy") == policy
        ],
        key=lambda row: int(row.get("frames") or 0),
    )


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


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    rows = report["rows"]
    scene_ids = [scene["scene_id"] for scene in report["scenes"]]
    measured_rows = [row for row in rows if row["policy"] == "measured"]
    loss_deltas: list[float] = []
    no_first_ratios: list[float] = []
    rebuild_ratios: list[float] = []
    for scene_id in scene_ids:
        cadence = _rows_for_policy(report, scene_id, "cadence")[0]
        measured = _rows_for_policy(report, scene_id, "measured")[0]
        loss_deltas.append(abs(float(measured["end_loss"]) - float(cadence["end_loss"])))
        no_first_ratios.append(float(measured["no_first_step_ms"]) / float(cadence["no_first_step_ms"]))
        rebuild_ratios.append(
            float(measured["projective_interval_cache_rebuilds"])
            / float(cadence["projective_interval_cache_rebuilds"])
        )
    return {
        "scene_count": len(scene_ids),
        "row_count": len(rows),
        "measured_row_count": len(measured_rows),
        "distinct_youtube_id_count": len({scene["youtube_id"] for scene in report["scenes"]}),
        "all_source_videos_exist": all(bool(scene["source_video_exists"]) for scene in report["scenes"]),
        "all_rows_pass": all(bool(row["pass"]) for row in rows),
        "all_rows_loss_decreased": all(bool(row["loss_decreased"]) for row in rows),
        "all_rows_no_overflow": all(int(row["tile_overflow_sum"]) == 0 for row in rows),
        "all_rows_fallback_free": all(int(row["projective_interval_cache_fallback_marks"]) == 0 for row in rows),
        "all_rows_visibility_stratification_free": all(
            int(row["projective_interval_cache_visibility_stratifications"]) == 0 for row in rows
        ),
        "all_measured_loss_matches_cadence": max(loss_deltas) < 1.0e-5,
        "max_measured_vs_cadence_end_loss_abs_delta": max(loss_deltas),
        "measured_vs_cadence_no_first_step_ms_ratios": no_first_ratios,
        "measured_vs_cadence_rebuild_ratios": rebuild_ratios,
        "max_measured_vs_cadence_no_first_step_ms_ratio": max(no_first_ratios),
        "max_measured_vs_cadence_rebuild_ratio": max(rebuild_ratios),
        "measured_support_rebins": [int(row["projective_interval_cache_support_rebins"]) for row in measured_rows],
        "measured_stale_refreshes": [
            int(row["projective_interval_cache_stale_refreshes"]) for row in measured_rows
        ],
        "measured_live_updates": [int(row["projective_interval_cache_live_updates"]) for row in measured_rows],
        "measured_staleness_checks": [
            int(row["projective_interval_cache_staleness_checks"]) for row in measured_rows
        ],
        "measured_cache_rebuilds": [int(row["projective_interval_cache_rebuilds"]) for row in measured_rows],
        "cadence_cache_rebuilds": [
            int(_rows_for_policy(report, scene_id, "cadence")[0]["projective_interval_cache_rebuilds"])
            for scene_id in scene_ids
        ],
        "max_measured_support_rebins": max(int(row["projective_interval_cache_support_rebins"]) for row in measured_rows),
        "max_measured_stale_refreshes": max(
            int(row["projective_interval_cache_stale_refreshes"]) for row in measured_rows
        ),
        "max_measured_support_tail_alpha_bound": max(
            float(row["projective_interval_cache_max_support_tail_alpha_bound"] or 0.0) for row in measured_rows
        ),
        "max_measured_support_overshoot_px": max(
            float(row["projective_interval_cache_max_support_max_overshoot_px"] or 0.0) for row in measured_rows
        ),
        "max_tile_count": max(int(row["max_tile_count"]) for row in rows),
        "min_motion_score": min(float(scene["motion_score"]) for scene in report["scenes"]),
        "max_motion_score": max(float(scene["motion_score"]) for scene in report["scenes"]),
    }


def _assert_summary_close(actual: Any, expected: Any, key: str, errors: list[str]) -> None:
    if isinstance(expected, float):
        if not isinstance(actual, int | float) or abs(float(actual) - expected) > 1.0e-8:
            errors.append(f"summary {key} mismatch: expected {expected!r}, got {actual!r}")
    elif actual != expected:
        errors.append(f"summary {key} mismatch: expected {expected!r}, got {actual!r}")


def verify_real_video_multiscene_trainer_matrix_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_real_video_multiscene_trainer_matrix":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    if report.get("base_domain") != "checked-in source-distinct real-video segments":
        errors.append(f"base_domain must name source-distinct real-video segments, got {report.get('base_domain')!r}")
    theory_contract = report.get("theory_contract")
    if (
        not isinstance(theory_contract, str)
        or "does not prove broad real-scene quality acceptance" not in theory_contract
        or "source-distinct" not in theory_contract
        or "guarded projective-interval trainer" not in theory_contract
    ):
        errors.append("theory_contract must preserve the multiscene trainer scope")

    scenes = report.get("scenes")
    rows = report.get("rows")
    if not isinstance(scenes, list) or len(scenes) < 3:
        errors.append("scenes must include at least three source-distinct real videos")
        return errors
    if not isinstance(rows, list) or len(rows) != 2 * len(scenes):
        errors.append("rows must contain one cadence and one measured row per scene")
        return errors

    scene_ids: set[str] = set()
    youtube_ids: set[str] = set()
    for scene in scenes:
        if not isinstance(scene, dict):
            errors.append("scene row must be an object")
            continue
        scene_id = str(scene.get("scene_id") or "")
        scene_ids.add(scene_id)
        youtube_ids.add(str(scene.get("youtube_id") or ""))
        if scene.get("source_video_exists") is not True:
            errors.append(f"scene {scene_id} source video must exist")
        if _finite_float(scene.get("motion_score"), f"{scene_id} motion_score", errors) <= 0.0:
            errors.append(f"scene {scene_id} must retain positive motion score")
    if len(scene_ids) != len(scenes):
        errors.append("scene ids must be unique")
    if len(youtube_ids) != len(scenes):
        errors.append("scene youtube ids must be source-distinct")

    for scene_id in scene_ids:
        cadence_rows = _rows_for_policy(report, scene_id, "cadence")
        measured_rows = _rows_for_policy(report, scene_id, "measured")
        if len(cadence_rows) != 1 or len(measured_rows) != 1:
            errors.append(f"scene {scene_id} must have exactly one cadence and one measured row")
            continue
        cadence = cadence_rows[0]
        measured = measured_rows[0]
        for label, row in (("cadence", cadence), ("measured", measured)):
            prefix = f"{scene_id} {label}"
            if not row.get("pass"):
                errors.append(f"{prefix} row must pass")
            if row.get("loss_decreased") is not True:
                errors.append(f"{prefix} row must decrease loss")
            if _finite_int(row.get("tile_overflow_sum"), f"{prefix} tile_overflow_sum", errors) != 0:
                errors.append(f"{prefix} tile_overflow_sum must be 0")
            if _finite_int(row.get("projective_interval_cache_fallback_marks"), f"{prefix} fallback_marks", errors) != 0:
                errors.append(f"{prefix} fallback marks must be 0")
            if (
                _finite_int(
                    row.get("projective_interval_cache_visibility_stratifications"),
                    f"{prefix} visibility_stratifications",
                    errors,
                )
                != 0
            ):
                errors.append(f"{prefix} visibility stratifications must be 0")
            max_tile_count = _finite_int(row.get("max_tile_count"), f"{prefix} max_tile_count", errors)
            if max_tile_count <= 0 or max_tile_count > int(report.get("tile_capacity") or 0):
                errors.append(f"{prefix} max_tile_count must stay within tile_capacity")
            if not isinstance(row.get("no_first_step_ms"), int | float) or float(row["no_first_step_ms"]) <= 0.0:
                errors.append(f"{prefix} no_first_step_ms must be positive")
        if abs(float(measured.get("end_loss") or 0.0) - float(cadence.get("end_loss") or 0.0)) >= 1.0e-5:
            errors.append(f"{scene_id} measured loss must match cadence")
        if int(measured.get("projective_interval_cache_rebuilds") or 0) >= int(
            cadence.get("projective_interval_cache_rebuilds") or 0
        ):
            errors.append(f"{scene_id} measured rebuilds must be below cadence")
        if int(measured.get("projective_interval_cache_live_updates") or 0) <= int(
            cadence.get("projective_interval_cache_live_updates") or 0
        ):
            errors.append(f"{scene_id} measured live updates must exceed cadence")
        if int(measured.get("projective_interval_cache_staleness_checks") or 0) < int(
            measured.get("projective_interval_cache_live_updates") or 0
        ):
            errors.append(f"{scene_id} measured staleness checks must cover live updates")
        if int(measured.get("projective_interval_cache_support_rebins") or 0) != 0:
            errors.append(f"{scene_id} measured support rebins must be 0 under guard")
        if int(measured.get("projective_interval_cache_stale_refreshes") or 0) != 0:
            errors.append(f"{scene_id} measured stale refreshes must be 0 under guard")
        tail_bound = _finite_float(
            measured.get("projective_interval_cache_max_support_tail_alpha_bound"),
            f"{scene_id} measured support tail",
            errors,
        )
        if tail_bound > float(report.get("support_stale_tail_alpha_epsilon") or 0.0):
            errors.append(f"{scene_id} measured support tail bound exceeds configured epsilon")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        return errors
    try:
        expected = summarize(report)
    except (KeyError, ZeroDivisionError, TypeError, ValueError) as exc:
        errors.append(f"summary could not be recomputed: {exc}")
        return errors
    for key, expected_value in expected.items():
        _assert_summary_close(summary.get(key), expected_value, key, errors)
    if summary.get("all_source_videos_exist") is not True:
        errors.append("all source videos must exist")
    if summary.get("all_rows_pass") is not True:
        errors.append("all multiscene rows must pass")
    if summary.get("all_measured_loss_matches_cadence") is not True:
        errors.append("all measured rows must match cadence loss")
    if int(summary.get("max_measured_support_rebins") or 0) != 0:
        errors.append("multiscene guarded measured rows must have zero support rebins")
    if int(summary.get("max_measured_stale_refreshes") or 0) != 0:
        errors.append("multiscene guarded measured rows must have zero stale refreshes")
    max_rebuild_ratio = summary.get("max_measured_vs_cadence_rebuild_ratio")
    if (
        not isinstance(max_rebuild_ratio, int | float)
        or not math.isfinite(float(max_rebuild_ratio))
        or float(max_rebuild_ratio) >= 1.0
    ):
        errors.append("multiscene measured rebuild ratio must stay below cadence")
    return errors


def assert_real_video_multiscene_trainer_matrix_report(report: dict[str, Any]) -> None:
    errors = verify_real_video_multiscene_trainer_matrix_report(report)
    if errors:
        raise AssertionError("real-video multiscene trainer matrix failed:\n- " + "\n- ".join(errors))


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
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        for policy in ("cadence", "measured"):
            rows.append(
                run_case(
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
                    out_json=args.out_dir / "cases" / f"{scene['segment_id']}_{policy}.json",
                    verbose_trainer_output=bool(args.verbose_trainer_output),
                )
            )
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_multiscene_trainer_matrix",
        "base_domain": "checked-in source-distinct real-video segments",
        "theory_contract": (
            "This report does not prove broad real-scene quality acceptance. It verifies that the guarded "
            "projective-interval trainer contract holds across a small source-distinct real-video matrix, "
            "broadening evidence beyond a single high-motion clip."
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
        "rows": rows,
    }
    report["summary"] = summarize(report)
    errors = verify_real_video_multiscene_trainer_matrix_report(report)
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
        assert_real_video_multiscene_trainer_matrix_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_benchmark(args)
    if report.get("status") == "ok":
        assert_real_video_multiscene_trainer_matrix_report(report)
    path = write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
