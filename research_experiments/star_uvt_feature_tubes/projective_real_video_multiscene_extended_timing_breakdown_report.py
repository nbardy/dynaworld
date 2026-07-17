from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_multiscene_extended_timing_breakdown"
)
DEFAULT_SOURCE_REPORT = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_multiscene_frame_scaling_matrix_extended5"
    / "summary.json"
)
EXPECTED_STRICT_TIMING_ERRORS = (
    "multiscene frame-scaling measured no-first timing must beat cadence",
    "multiscene frame-scaling no-first timing growth must stay below frame growth",
)


def _finite_float(value: Any, label: str, errors: list[str]) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        errors.append(f"{label} must be finite, got {value!r}")
        return 0.0
    return float(value)


def _finite_int(value: Any, label: str, errors: list[str]) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        errors.append(f"{label} must be an int, got {value!r}")
        return 0
    return int(value)


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return math.inf
    return float(numerator) / float(denominator)


def _rows_for(report: dict[str, Any], scene_id: str, frames: int, policy: str) -> list[dict[str, Any]]:
    return [
        row
        for row in report.get("rows", [])
        if isinstance(row, dict)
        and row.get("scene_id") == scene_id
        and int(row.get("frames") or 0) == int(frames)
        and row.get("policy") == policy
    ]


def _row_int(row: dict[str, Any], key: str) -> int:
    return int(row.get(key) or 0)


def _row_float(row: dict[str, Any], key: str) -> float:
    return float(row.get(key) or 0.0)


def _build_pair_breakdowns(source: dict[str, Any]) -> list[dict[str, Any]]:
    frame_counts = [int(value) for value in source.get("frame_counts", [])]
    pairs: list[dict[str, Any]] = []
    for scene in source.get("scenes", []):
        scene_id = str(scene["scene_id"])
        for frames in frame_counts:
            cadence = _rows_for(source, scene_id, frames, "cadence")[0]
            measured = _rows_for(source, scene_id, frames, "measured")[0]
            no_first_ratio = _ratio(
                _row_float(measured, "no_first_step_ms"),
                _row_float(cadence, "no_first_step_ms"),
            )
            rebuild_ratio = _ratio(
                _row_float(measured, "projective_interval_cache_rebuilds"),
                _row_float(cadence, "projective_interval_cache_rebuilds"),
            )
            pair = {
                "scene_id": scene_id,
                "youtube_id": str(scene.get("youtube_id") or measured.get("youtube_id") or ""),
                "title": str(scene.get("title") or measured.get("title") or ""),
                "frames": frames,
                "motion_score": float(scene.get("motion_score") or measured.get("motion_score") or 0.0),
                "cadence_no_first_step_ms": _row_float(cadence, "no_first_step_ms"),
                "measured_no_first_step_ms": _row_float(measured, "no_first_step_ms"),
                "measured_vs_cadence_no_first_step_ms_ratio": no_first_ratio,
                "measured_vs_cadence_mean_step_ms_ratio": _ratio(
                    _row_float(measured, "mean_step_ms"),
                    _row_float(cadence, "mean_step_ms"),
                ),
                "measured_vs_cadence_forward_ms_ratio": _ratio(
                    _row_float(measured, "mean_render_forward_ms"),
                    _row_float(cadence, "mean_render_forward_ms"),
                ),
                "measured_vs_cadence_backward_ms_ratio": _ratio(
                    _row_float(measured, "mean_backward_ms"),
                    _row_float(cadence, "mean_backward_ms"),
                ),
                "cadence_cache_rebuilds": _row_int(cadence, "projective_interval_cache_rebuilds"),
                "measured_cache_rebuilds": _row_int(measured, "projective_interval_cache_rebuilds"),
                "measured_vs_cadence_rebuild_ratio": rebuild_ratio,
                "measured_cache_live_updates": _row_int(measured, "projective_interval_cache_live_updates"),
                "measured_cache_staleness_checks": _row_int(
                    measured, "projective_interval_cache_staleness_checks"
                ),
                "measured_support_rebins": _row_int(measured, "projective_interval_cache_support_rebins"),
                "measured_stale_refreshes": _row_int(
                    measured, "projective_interval_cache_stale_refreshes"
                ),
                "measured_fallback_marks": _row_int(
                    measured, "projective_interval_cache_fallback_marks"
                ),
                "measured_visibility_stratifications": _row_int(
                    measured, "projective_interval_cache_visibility_stratifications"
                ),
                "measured_tile_overflow_sum": _row_int(measured, "tile_overflow_sum"),
                "measured_support_tail_alpha_bound": _row_float(
                    measured, "projective_interval_cache_max_support_tail_alpha_bound"
                ),
                "measured_support_overshoot_px": _row_float(
                    measured, "projective_interval_cache_max_support_max_overshoot_px"
                ),
                "measured_max_tile_count": _row_int(measured, "max_tile_count"),
                "cadence_max_tile_count": _row_int(cadence, "max_tile_count"),
                "end_loss_abs_delta": abs(_row_float(measured, "end_loss") - _row_float(cadence, "end_loss")),
                "no_first_timing_miss": no_first_ratio >= 1.0,
            }
            pairs.append(pair)
    return sorted(pairs, key=lambda row: (str(row["scene_id"]), int(row["frames"])))


def _growth(values: list[float]) -> float:
    if len(values) < 2 or values[0] <= 0.0:
        return 0.0
    return values[-1] / values[0]


def _build_scene_breakdowns(source: dict[str, Any], pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame_counts = [int(value) for value in source.get("frame_counts", [])]
    frame_growth = _ratio(float(frame_counts[-1]), float(frame_counts[0])) if frame_counts else 0.0
    scenes: list[dict[str, Any]] = []
    for scene in source.get("scenes", []):
        scene_id = str(scene["scene_id"])
        scene_pairs = [pair for pair in pairs if pair["scene_id"] == scene_id]
        measured_no_first = [
            float(pair["measured_no_first_step_ms"])
            for pair in sorted(scene_pairs, key=lambda pair: int(pair["frames"]))
        ]
        rebuilds = [
            float(pair["measured_cache_rebuilds"])
            for pair in sorted(scene_pairs, key=lambda pair: int(pair["frames"]))
        ]
        growth_ratio = _ratio(_growth(measured_no_first), frame_growth)
        row = {
            "scene_id": scene_id,
            "youtube_id": str(scene.get("youtube_id") or ""),
            "motion_score": float(scene.get("motion_score") or 0.0),
            "frame_counts": frame_counts,
            "max_no_first_ratio": max(
                float(pair["measured_vs_cadence_no_first_step_ms_ratio"]) for pair in scene_pairs
            ),
            "no_first_ratio_gt1_count": sum(
                1 for pair in scene_pairs if float(pair["measured_vs_cadence_no_first_step_ms_ratio"]) >= 1.0
            ),
            "measured_no_first_growth_vs_frame_growth_ratio": growth_ratio,
            "growth_timing_miss": growth_ratio >= 1.0,
            "measured_cache_rebuild_growth": _growth(rebuilds),
            "max_measured_support_rebins": max(int(pair["measured_support_rebins"]) for pair in scene_pairs),
            "max_measured_stale_refreshes": max(
                int(pair["measured_stale_refreshes"]) for pair in scene_pairs
            ),
            "max_measured_fallback_marks": max(int(pair["measured_fallback_marks"]) for pair in scene_pairs),
            "max_measured_visibility_stratifications": max(
                int(pair["measured_visibility_stratifications"]) for pair in scene_pairs
            ),
            "max_measured_tile_overflow_sum": max(
                int(pair["measured_tile_overflow_sum"]) for pair in scene_pairs
            ),
        }
        scenes.append(row)
    return sorted(scenes, key=lambda row: str(row["scene_id"]))


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    pairs = report["pair_breakdowns"]
    scenes = report["scene_breakdowns"]
    source_summary = report["source_summary"]
    failing_pairs = [pair for pair in pairs if pair["no_first_timing_miss"]]
    failing_scenes = [scene for scene in scenes if scene["growth_timing_miss"]]
    worst_pair = max(pairs, key=lambda pair: float(pair["measured_vs_cadence_no_first_step_ms_ratio"]))
    worst_scene = max(scenes, key=lambda scene: float(scene["measured_no_first_growth_vs_frame_growth_ratio"]))
    return {
        "source_status": report["source_status"],
        "source_scene_count": int(source_summary["scene_count"]),
        "source_distinct_youtube_id_count": int(source_summary["distinct_youtube_id_count"]),
        "source_row_count": int(source_summary["row_count"]),
        "pair_count": len(pairs),
        "scene_count": len(scenes),
        "frame_count_count": int(source_summary["frame_count_count"]),
        "frame_growth_factor": float(source_summary["frame_growth_factor"]),
        "strict_failure_count": len(report["source_errors"]),
        "strict_failed_only_expected_timing": tuple(report["source_errors"]) == EXPECTED_STRICT_TIMING_ERRORS,
        "no_first_ratio_gt1_count": len(failing_pairs),
        "no_first_ratio_gt1_fraction": len(failing_pairs) / max(1, len(pairs)),
        "growth_ratio_gt1_count": len(failing_scenes),
        "growth_ratio_gt1_fraction": len(failing_scenes) / max(1, len(scenes)),
        "max_measured_vs_cadence_no_first_step_ms_ratio": float(
            worst_pair["measured_vs_cadence_no_first_step_ms_ratio"]
        ),
        "max_no_first_ratio_scene_id": str(worst_pair["scene_id"]),
        "max_no_first_ratio_frames": int(worst_pair["frames"]),
        "max_no_first_ratio_overage": max(
            0.0,
            float(worst_pair["measured_vs_cadence_no_first_step_ms_ratio"]) - 1.0,
        ),
        "max_measured_no_first_growth_vs_frame_growth_ratio": float(
            worst_scene["measured_no_first_growth_vs_frame_growth_ratio"]
        ),
        "max_growth_ratio_scene_id": str(worst_scene["scene_id"]),
        "max_growth_ratio_overage": max(
            0.0,
            float(worst_scene["measured_no_first_growth_vs_frame_growth_ratio"]) - 1.0,
        ),
        "distinct_no_first_miss_scene_count": len({str(pair["scene_id"]) for pair in failing_pairs}),
        "distinct_any_timing_miss_scene_count": len(
            {str(pair["scene_id"]) for pair in failing_pairs}
            | {str(scene["scene_id"]) for scene in failing_scenes}
        ),
        "all_pair_support_clean": all(
            int(pair["measured_support_rebins"]) == 0
            and int(pair["measured_stale_refreshes"]) == 0
            and int(pair["measured_fallback_marks"]) == 0
            and int(pair["measured_visibility_stratifications"]) == 0
            and int(pair["measured_tile_overflow_sum"]) == 0
            for pair in pairs
        ),
        "all_pair_loss_matches_cadence": max(float(pair["end_loss_abs_delta"]) for pair in pairs) < 1.0e-5,
        "all_pair_rebuild_ratio_below_cadence": all(
            float(pair["measured_vs_cadence_rebuild_ratio"]) < 1.0 for pair in pairs
        ),
        "all_scene_rebuild_growth_flat": all(float(scene["measured_cache_rebuild_growth"]) <= 1.0 for scene in scenes),
        "all_failing_pairs_cache_clean": all(
            int(pair["measured_cache_rebuilds"]) < int(pair["cadence_cache_rebuilds"])
            and int(pair["measured_support_rebins"]) == 0
            and int(pair["measured_stale_refreshes"]) == 0
            and int(pair["measured_fallback_marks"]) == 0
            and int(pair["measured_visibility_stratifications"]) == 0
            and int(pair["measured_tile_overflow_sum"]) == 0
            for pair in failing_pairs
        ),
        "max_measured_vs_cadence_rebuild_ratio": max(
            float(pair["measured_vs_cadence_rebuild_ratio"]) for pair in pairs
        ),
        "max_measured_support_rebins": max(int(pair["measured_support_rebins"]) for pair in pairs),
        "max_measured_stale_refreshes": max(int(pair["measured_stale_refreshes"]) for pair in pairs),
        "max_measured_fallback_marks": max(int(pair["measured_fallback_marks"]) for pair in pairs),
        "max_measured_visibility_stratifications": max(
            int(pair["measured_visibility_stratifications"]) for pair in pairs
        ),
        "max_measured_tile_overflow_sum": max(int(pair["measured_tile_overflow_sum"]) for pair in pairs),
        "max_end_loss_abs_delta": max(float(pair["end_loss_abs_delta"]) for pair in pairs),
        "max_motion_score": max(float(scene["motion_score"]) for scene in scenes),
    }


def verify_extended_timing_breakdown_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_real_video_multiscene_extended_timing_breakdown":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    theory_contract = report.get("theory_contract")
    if (
        not isinstance(theory_contract, str)
        or "does not prove a timing win" not in theory_contract
        or "pair-level timing breakdown" not in theory_contract
        or "cache/support churn" not in theory_contract
    ):
        errors.append("theory_contract must preserve pair-level diagnostic scope")
    if tuple(report.get("source_errors") or ()) != EXPECTED_STRICT_TIMING_ERRORS:
        errors.append("source_errors must remain exactly the expected strict timing failures")
    if report.get("source_status") != "failed":
        errors.append(f"source_status must remain failed, got {report.get('source_status')!r}")
    source_summary = report.get("source_summary")
    pairs = report.get("pair_breakdowns")
    scenes = report.get("scene_breakdowns")
    if not isinstance(source_summary, dict):
        errors.append("source_summary must be an object")
        return errors
    if not isinstance(pairs, list) or not pairs:
        errors.append("pair_breakdowns must be a non-empty list")
        return errors
    if not isinstance(scenes, list) or not scenes:
        errors.append("scene_breakdowns must be a non-empty list")
        return errors
    if int(source_summary.get("scene_count") or 0) < 5:
        errors.append("timing breakdown must cover at least five source scenes")
    if int(source_summary.get("row_count") or 0) < 30:
        errors.append("timing breakdown source must cover at least 30 rows")
    if len(pairs) * 2 != int(source_summary.get("row_count") or 0):
        errors.append("pair_count must explain cadence/measured source rows")
    expected_pair_count = int(source_summary.get("scene_count") or 0) * int(
        source_summary.get("frame_count_count") or 0
    )
    if len(pairs) != expected_pair_count:
        errors.append("pair_count must equal scene_count * frame_count_count")
    seen_pairs: set[tuple[str, int]] = set()
    for pair in pairs:
        if not isinstance(pair, dict):
            errors.append("pair breakdown rows must be objects")
            continue
        key = (str(pair.get("scene_id") or ""), _finite_int(pair.get("frames"), "pair frames", errors))
        if key in seen_pairs:
            errors.append(f"duplicate pair breakdown for {key}")
        seen_pairs.add(key)
        ratio = _finite_float(
            pair.get("measured_vs_cadence_no_first_step_ms_ratio"),
            f"{key} no-first ratio",
            errors,
        )
        if (ratio >= 1.0) != bool(pair.get("no_first_timing_miss")):
            errors.append(f"{key} no_first_timing_miss must match ratio >= 1")
        if _finite_float(pair.get("end_loss_abs_delta"), f"{key} end loss delta", errors) >= 1.0e-5:
            errors.append(f"{key} end loss delta must stay below 1e-5")
        if _finite_float(pair.get("measured_vs_cadence_rebuild_ratio"), f"{key} rebuild ratio", errors) >= 1.0:
            errors.append(f"{key} measured rebuild ratio must stay below cadence")
    for scene in scenes:
        if not isinstance(scene, dict):
            errors.append("scene breakdown rows must be objects")
            continue
        growth_ratio = _finite_float(
            scene.get("measured_no_first_growth_vs_frame_growth_ratio"),
            f"{scene.get('scene_id')} growth ratio",
            errors,
        )
        if (growth_ratio >= 1.0) != bool(scene.get("growth_timing_miss")):
            errors.append(f"{scene.get('scene_id')} growth_timing_miss must match ratio >= 1")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        return errors
    try:
        expected = summarize(report)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"summary could not be recomputed: {exc}")
        return errors
    for key, expected_value in expected.items():
        actual = summary.get(key)
        if isinstance(expected_value, float):
            if not isinstance(actual, int | float) or abs(float(actual) - expected_value) > 1.0e-9:
                errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")
        elif actual != expected_value:
            errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")
    if summary.get("strict_failed_only_expected_timing") is not True:
        errors.append("timing breakdown must fail only the expected strict timing gates")
    if int(summary.get("no_first_ratio_gt1_count") or 0) <= 0:
        errors.append("timing breakdown must preserve at least one no-first timing miss")
    if int(summary.get("growth_ratio_gt1_count") or 0) <= 0:
        errors.append("timing breakdown must preserve at least one frame-growth timing miss")
    if summary.get("all_failing_pairs_cache_clean") is not True:
        errors.append("failing timing pairs must remain cache/support clean")
    if summary.get("all_pair_support_clean") is not True:
        errors.append("all measured pairs must remain support/fallback/overflow clean")
    if summary.get("all_pair_loss_matches_cadence") is not True:
        errors.append("all measured pairs must match cadence losses")
    if summary.get("all_scene_rebuild_growth_flat") is not True:
        errors.append("measured rebuild count must remain flat across frame growth")
    return errors


def assert_extended_timing_breakdown_report(report: dict[str, Any]) -> None:
    errors = verify_extended_timing_breakdown_report(report)
    if errors:
        raise AssertionError("extended timing breakdown failed:\n- " + "\n- ".join(errors))


def run_report(source_report: Path = DEFAULT_SOURCE_REPORT) -> dict[str, Any]:
    source = json.loads(source_report.read_text(encoding="utf-8"))
    pairs = _build_pair_breakdowns(source)
    scenes = _build_scene_breakdowns(source, pairs)
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_multiscene_extended_timing_breakdown",
        "base_domain": "failed strict five-source real-video frame-scaling matrix",
        "theory_contract": (
            "This report is a pair-level timing breakdown of a failed strict five-source run. "
            "It does not prove a timing win; it isolates timing misses while checking that they are not "
            "caused by cache/support churn, fallback, overflow, or cadence-loss divergence."
        ),
        "source_report": str(source_report),
        "source_status": source.get("status"),
        "source_errors": list(source.get("errors") or []),
        "source_summary": source.get("summary", {}),
        "pair_breakdowns": pairs,
        "scene_breakdowns": scenes,
    }
    report["summary"] = summarize(report)
    errors = verify_extended_timing_breakdown_report(report)
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
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--verify-report", type=Path)
    args = parser.parse_args()

    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_extended_timing_breakdown_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(source_report=args.source_report)
    if report.get("status") == "ok":
        assert_extended_timing_breakdown_report(report)
    path = write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
