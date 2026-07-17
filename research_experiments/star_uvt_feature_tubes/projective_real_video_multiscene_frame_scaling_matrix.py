from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
STAR_UVT_ROOT = ROOT / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
for path in (ROOT, ROOT / "src" / "train", STAR_UVT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402

from projective_real_video_multiscene_trainer_matrix import (  # noqa: E402
    DEFAULT_SEGMENT_IDS,
    DEFAULT_SEGMENTS_MANIFEST,
    _assert_summary_close,
    _finite_float,
    _finite_int,
    _load_segments,
    run_case,
)
from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    has_projective_trace_cell_interval_backward_metal,
    has_projective_trace_cell_interval_metal,
)


DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_multiscene_frame_scaling_matrix"
)
DEFAULT_FRAME_COUNTS = (4, 8, 16)


def _rows_for(report: dict[str, Any], scene_id: str, frames: int, policy: str) -> list[dict[str, Any]]:
    return [
        row
        for row in report.get("rows", [])
        if isinstance(row, dict)
        and row.get("scene_id") == scene_id
        and int(row.get("frames") or 0) == int(frames)
        and row.get("policy") == policy
    ]


def _policy_rows_for_scene(report: dict[str, Any], scene_id: str, policy: str) -> list[dict[str, Any]]:
    return sorted(
        [
            row
            for row in report.get("rows", [])
            if isinstance(row, dict) and row.get("scene_id") == scene_id and row.get("policy") == policy
        ],
        key=lambda row: int(row.get("frames") or 0),
    )


def _growth(values: list[float]) -> float:
    if len(values) < 2 or values[0] <= 0.0:
        return 0.0
    return float(values[-1]) / float(values[0])


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    rows = report["rows"]
    scenes = report["scenes"]
    frame_counts = [int(value) for value in report["frame_counts"]]
    scene_ids = [str(scene["scene_id"]) for scene in scenes]
    measured_rows = [row for row in rows if row["policy"] == "measured"]
    loss_deltas: list[float] = []
    no_first_ratios: list[float] = []
    rebuild_ratios: list[float] = []
    measured_no_first_growth_ratios: list[float] = []
    measured_cache_rebuild_growths: list[float] = []
    frame_growth = float(frame_counts[-1]) / float(frame_counts[0])
    for scene_id in scene_ids:
        measured_scene_rows = _policy_rows_for_scene(report, scene_id, "measured")
        measured_no_first_growth_ratios.append(
            _growth([float(row["no_first_step_ms"]) for row in measured_scene_rows]) / frame_growth
        )
        measured_cache_rebuild_growths.append(
            _growth([float(row["projective_interval_cache_rebuilds"]) for row in measured_scene_rows])
        )
        for frames in frame_counts:
            cadence = _rows_for(report, scene_id, frames, "cadence")[0]
            measured = _rows_for(report, scene_id, frames, "measured")[0]
            loss_deltas.append(abs(float(measured["end_loss"]) - float(cadence["end_loss"])))
            no_first_ratios.append(float(measured["no_first_step_ms"]) / float(cadence["no_first_step_ms"]))
            rebuild_ratios.append(
                float(measured["projective_interval_cache_rebuilds"])
                / float(cadence["projective_interval_cache_rebuilds"])
            )
    return {
        "scene_count": len(scene_ids),
        "frame_count_count": len(frame_counts),
        "row_count": len(rows),
        "measured_row_count": len(measured_rows),
        "distinct_youtube_id_count": len({scene["youtube_id"] for scene in scenes}),
        "frame_growth_factor": frame_growth,
        "all_source_videos_exist": all(bool(scene["source_video_exists"]) for scene in scenes),
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
        "max_measured_vs_cadence_no_first_step_ms_ratio": max(no_first_ratios),
        "measured_vs_cadence_rebuild_ratios": rebuild_ratios,
        "max_measured_vs_cadence_rebuild_ratio": max(rebuild_ratios),
        "measured_no_first_growth_vs_frame_growth_ratios": measured_no_first_growth_ratios,
        "max_measured_no_first_growth_vs_frame_growth_ratio": max(measured_no_first_growth_ratios),
        "measured_cache_rebuild_growths": measured_cache_rebuild_growths,
        "max_measured_cache_rebuild_growth": max(measured_cache_rebuild_growths),
        "measured_support_rebins": [int(row["projective_interval_cache_support_rebins"]) for row in measured_rows],
        "measured_stale_refreshes": [
            int(row["projective_interval_cache_stale_refreshes"]) for row in measured_rows
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
        "min_motion_score": min(float(scene["motion_score"]) for scene in scenes),
        "max_motion_score": max(float(scene["motion_score"]) for scene in scenes),
    }


def verify_real_video_multiscene_frame_scaling_matrix_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_real_video_multiscene_frame_scaling_matrix":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    if report.get("base_domain") != "checked-in source-distinct real-video segments over frame counts":
        errors.append(f"base_domain must name source-distinct frame scaling, got {report.get('base_domain')!r}")
    theory_contract = report.get("theory_contract")
    if (
        not isinstance(theory_contract, str)
        or "does not prove broad real-scene quality acceptance" not in theory_contract
        or "source-distinct" not in theory_contract
        or "frame growth" not in theory_contract
        or "guarded projective-interval trainer" not in theory_contract
    ):
        errors.append("theory_contract must preserve the source-distinct frame-scaling scope")

    scenes = report.get("scenes")
    rows = report.get("rows")
    frame_counts = report.get("frame_counts")
    if not isinstance(scenes, list) or len(scenes) < 3:
        errors.append("scenes must include at least three source-distinct real videos")
        return errors
    if not isinstance(frame_counts, list) or len(frame_counts) < 3:
        errors.append("frame_counts must include at least three frame counts")
        return errors
    if sorted(int(value) for value in frame_counts) != [int(value) for value in frame_counts]:
        errors.append("frame_counts must be sorted")
    if not isinstance(rows, list) or len(rows) != 2 * len(scenes) * len(frame_counts):
        errors.append("rows must contain one cadence and one measured row per scene/frame count")
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
        measured_rebuilds: list[int] = []
        for frames in [int(value) for value in frame_counts]:
            cadence_rows = _rows_for(report, scene_id, frames, "cadence")
            measured_rows = _rows_for(report, scene_id, frames, "measured")
            if len(cadence_rows) != 1 or len(measured_rows) != 1:
                errors.append(f"scene {scene_id} frames {frames} must have exactly one cadence and measured row")
                continue
            cadence = cadence_rows[0]
            measured = measured_rows[0]
            for label, row in (("cadence", cadence), ("measured", measured)):
                prefix = f"{scene_id} {frames}f {label}"
                if row.get("pass") is not True:
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
                if _finite_float(row.get("no_first_step_ms"), f"{prefix} no_first_step_ms", errors) <= 0.0:
                    errors.append(f"{prefix} no_first_step_ms must be positive")
            if abs(float(measured.get("end_loss") or 0.0) - float(cadence.get("end_loss") or 0.0)) >= 1.0e-5:
                errors.append(f"{scene_id} {frames}f measured loss must match cadence")
            if int(measured.get("projective_interval_cache_rebuilds") or 0) >= int(
                cadence.get("projective_interval_cache_rebuilds") or 0
            ):
                errors.append(f"{scene_id} {frames}f measured rebuilds must be below cadence")
            if int(measured.get("projective_interval_cache_live_updates") or 0) <= int(
                cadence.get("projective_interval_cache_live_updates") or 0
            ):
                errors.append(f"{scene_id} {frames}f measured live updates must exceed cadence")
            if int(measured.get("projective_interval_cache_staleness_checks") or 0) < int(
                measured.get("projective_interval_cache_live_updates") or 0
            ):
                errors.append(f"{scene_id} {frames}f measured staleness checks must cover live updates")
            if int(measured.get("projective_interval_cache_support_rebins") or 0) != 0:
                errors.append(f"{scene_id} {frames}f measured support rebins must be 0 under guard")
            if int(measured.get("projective_interval_cache_stale_refreshes") or 0) != 0:
                errors.append(f"{scene_id} {frames}f measured stale refreshes must be 0 under guard")
            tail_bound = _finite_float(
                measured.get("projective_interval_cache_max_support_tail_alpha_bound"),
                f"{scene_id} {frames}f measured support tail",
                errors,
            )
            if tail_bound > float(report.get("support_stale_tail_alpha_epsilon") or 0.0):
                errors.append(f"{scene_id} {frames}f measured support tail bound exceeds configured epsilon")
            measured_rebuilds.append(int(measured.get("projective_interval_cache_rebuilds") or 0))
        if len(set(measured_rebuilds)) != 1:
            errors.append(f"{scene_id} measured rebuild count must remain constant across frame growth")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        return errors
    try:
        expected = summarize(report)
    except (KeyError, IndexError, ZeroDivisionError, TypeError, ValueError) as exc:
        errors.append(f"summary could not be recomputed: {exc}")
        return errors
    for key, expected_value in expected.items():
        _assert_summary_close(summary.get(key), expected_value, key, errors)
    if summary.get("all_source_videos_exist") is not True:
        errors.append("all source videos must exist")
    if summary.get("all_rows_pass") is not True:
        errors.append("all multiscene frame-scaling rows must pass")
    if summary.get("all_measured_loss_matches_cadence") is not True:
        errors.append("all measured rows must match cadence loss")
    if int(summary.get("max_measured_support_rebins") or 0) != 0:
        errors.append("multiscene frame-scaling measured rows must have zero support rebins")
    if int(summary.get("max_measured_stale_refreshes") or 0) != 0:
        errors.append("multiscene frame-scaling measured rows must have zero stale refreshes")
    max_no_first_ratio = summary.get("max_measured_vs_cadence_no_first_step_ms_ratio")
    if (
        not isinstance(max_no_first_ratio, int | float)
        or not math.isfinite(float(max_no_first_ratio))
        or float(max_no_first_ratio) >= 1.0
    ):
        errors.append("multiscene frame-scaling measured no-first timing must beat cadence")
    max_rebuild_ratio = summary.get("max_measured_vs_cadence_rebuild_ratio")
    if (
        not isinstance(max_rebuild_ratio, int | float)
        or not math.isfinite(float(max_rebuild_ratio))
        or float(max_rebuild_ratio) >= 1.0
    ):
        errors.append("multiscene frame-scaling measured rebuild ratio must stay below cadence")
    if _finite_float(summary.get("max_measured_cache_rebuild_growth"), "max measured rebuild growth", errors) > 1.0:
        errors.append("multiscene frame-scaling measured rebuild count must not grow with frame count")
    if (
        _finite_float(
            summary.get("max_measured_no_first_growth_vs_frame_growth_ratio"),
            "max measured no-first/frame growth ratio",
            errors,
        )
        >= 1.0
    ):
        errors.append("multiscene frame-scaling no-first timing growth must stay below frame growth")
    return errors


def assert_real_video_multiscene_frame_scaling_matrix_report(report: dict[str, Any]) -> None:
    errors = verify_real_video_multiscene_frame_scaling_matrix_report(report)
    if errors:
        raise AssertionError("real-video multiscene frame-scaling matrix failed:\n- " + "\n- ".join(errors))


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
    frame_counts = tuple(int(value) for value in (args.frame_counts or DEFAULT_FRAME_COUNTS))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        for frames in frame_counts:
            for policy in ("cadence", "measured"):
                rows.append(
                    run_case(
                        scene=scene,
                        frames=frames,
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
                        out_json=args.out_dir / "cases" / f"{scene['segment_id']}_{frames}f_{policy}.json",
                        verbose_trainer_output=bool(args.verbose_trainer_output),
                    )
                )
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_multiscene_frame_scaling_matrix",
        "base_domain": "checked-in source-distinct real-video segments over frame counts",
        "theory_contract": (
            "This report does not prove broad real-scene quality acceptance. It verifies that the guarded "
            "projective-interval trainer contract holds across source-distinct real-video frame growth, "
            "broadening frame-scaling evidence beyond a single high-motion clip."
        ),
        "segments_manifest": str(DEFAULT_SEGMENTS_MANIFEST),
        "frame_counts": list(frame_counts),
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
    errors = verify_real_video_multiscene_frame_scaling_matrix_report(report)
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
    parser.add_argument("--frame-count", action="append", type=int, dest="frame_counts")
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
        assert_real_video_multiscene_frame_scaling_matrix_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_benchmark(args)
    if report.get("status") == "ok":
        assert_real_video_multiscene_frame_scaling_matrix_report(report)
    path = write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
