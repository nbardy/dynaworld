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
    / "2026-05-25_star_uvt_projective_real_video_multiscene_extended_render_forward_residual"
)
DEFAULT_SOURCE_REPORT = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_multiscene_frame_scaling_matrix_extended5"
    / "summary.json"
)
DEFAULT_CASE_DIR = DEFAULT_SOURCE_REPORT.parent / "cases"
EXPECTED_STRICT_TIMING_ERRORS = (
    "multiscene frame-scaling measured no-first timing must beat cadence",
    "multiscene frame-scaling no-first timing growth must stay below frame growth",
)
TILE_STAT_KEYS = (
    "tile_count",
    "active_tile_count",
    "active_tile_fraction",
    "raw_tile_tube_refs",
    "clipped_tile_tube_refs",
    "mean_tile_count",
    "mean_active_tile_count",
    "p50_tile_count",
    "p95_tile_count",
    "p95_active_tile_count",
    "p99_tile_count",
    "p99_active_tile_count",
    "max_tile_count",
    "overflow_tile_count",
    "overflow_excess_tube_refs",
    "unstable_tile_count",
    "tile_capacity",
)
PHASE_KEYS = (
    "step_ms",
    "render_forward_ms",
    "backward_ms",
    "colorize_loss_ms",
    "optimizer_ms",
)


def _finite_float(value: Any, label: str, errors: list[str]) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        errors.append(f"{label} must be finite, got {value!r}")
        return 0.0
    return float(value)


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return math.inf
    return float(numerator) / float(denominator)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _rows_for(report: dict[str, Any], scene_id: str, frames: int, policy: str) -> list[dict[str, Any]]:
    return [
        row
        for row in report.get("rows", [])
        if isinstance(row, dict)
        and row.get("scene_id") == scene_id
        and int(row.get("frames") or 0) == int(frames)
        and row.get("policy") == policy
    ]


def _case_path(case_dir: Path, scene_id: str, frames: int, policy: str) -> Path:
    return case_dir / f"{scene_id}_{frames}f_{policy}.json"


def _load_case(case_dir: Path, scene_id: str, frames: int, policy: str) -> dict[str, Any]:
    return json.loads(_case_path(case_dir, scene_id, frames, policy).read_text(encoding="utf-8"))


def _phase_means(case: dict[str, Any]) -> dict[str, float]:
    rows = case.get("step_timings_ms")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("case must contain at least two step_timings_ms entries")
    no_first_rows = [row for row in rows[1:] if isinstance(row, dict)]
    if len(no_first_rows) != len(rows) - 1:
        raise ValueError("case step_timings_ms entries must be objects")
    return {
        key: _mean([float(row.get(key) or 0.0) for row in no_first_rows])
        for key in PHASE_KEYS
    }


def _numeric_tile_delta(cadence_tile: dict[str, Any], measured_tile: dict[str, Any]) -> float:
    deltas: list[float] = []
    for key in TILE_STAT_KEYS:
        cadence_value = cadence_tile.get(key)
        measured_value = measured_tile.get(key)
        if isinstance(cadence_value, int | float) and isinstance(measured_value, int | float):
            deltas.append(abs(float(measured_value) - float(cadence_value)))
        elif measured_value != cadence_value:
            deltas.append(1.0)
        else:
            deltas.append(0.0)
    return max(deltas) if deltas else 0.0


def _tile_digest(tile_stats: dict[str, Any]) -> dict[str, Any]:
    return {key: tile_stats.get(key) for key in TILE_STAT_KEYS}


def _pair_profile(source: dict[str, Any], case_dir: Path, scene_id: str, frames: int) -> dict[str, Any]:
    cadence_row = _rows_for(source, scene_id, frames, "cadence")[0]
    measured_row = _rows_for(source, scene_id, frames, "measured")[0]
    cadence_case = _load_case(case_dir, scene_id, frames, "cadence")
    measured_case = _load_case(case_dir, scene_id, frames, "measured")
    cadence_phase = _phase_means(cadence_case)
    measured_phase = _phase_means(measured_case)
    cadence_tile = cadence_case.get("tile_stats")
    measured_tile = measured_case.get("tile_stats")
    if not isinstance(cadence_tile, dict) or not isinstance(measured_tile, dict):
        raise ValueError(f"{scene_id} {frames}f cases must include tile_stats objects")
    tile_delta = _numeric_tile_delta(cadence_tile, measured_tile)
    clipped_refs = float(measured_tile.get("clipped_tile_tube_refs") or 0.0)
    active_tiles = float(measured_tile.get("active_tile_count") or 0.0)
    render_ratio = _ratio(measured_phase["render_forward_ms"], cadence_phase["render_forward_ms"])
    step_ratio = _ratio(measured_phase["step_ms"], cadence_phase["step_ms"])
    source_ratio = _ratio(
        float(measured_row["no_first_step_ms"]),
        float(cadence_row["no_first_step_ms"]),
    )
    render_delta = measured_phase["render_forward_ms"] - cadence_phase["render_forward_ms"]
    return {
        "scene_id": scene_id,
        "youtube_id": str(measured_row.get("youtube_id") or ""),
        "title": str(measured_row.get("title") or ""),
        "frames": frames,
        "motion_score": float(measured_row.get("motion_score") or 0.0),
        "source_no_first_step_ratio": source_ratio,
        "case_no_first_step_ratio": step_ratio,
        "source_case_no_first_abs_delta": abs(step_ratio - source_ratio),
        "render_forward_ratio": render_ratio,
        "render_forward_delta_ms": render_delta,
        "render_forward_ms_per_clipped_ref_ratio": _ratio(
            _ratio(measured_phase["render_forward_ms"], clipped_refs),
            _ratio(cadence_phase["render_forward_ms"], clipped_refs),
        ),
        "render_forward_ms_per_active_tile_ratio": _ratio(
            _ratio(measured_phase["render_forward_ms"], active_tiles),
            _ratio(cadence_phase["render_forward_ms"], active_tiles),
        ),
        "measured_render_forward_ms_per_clipped_ref": _ratio(
            measured_phase["render_forward_ms"],
            clipped_refs,
        ),
        "cadence_render_forward_ms_per_clipped_ref": _ratio(
            cadence_phase["render_forward_ms"],
            clipped_refs,
        ),
        "measured_render_forward_ms_per_active_tile": _ratio(
            measured_phase["render_forward_ms"],
            active_tiles,
        ),
        "cadence_render_forward_ms_per_active_tile": _ratio(
            cadence_phase["render_forward_ms"],
            active_tiles,
        ),
        "cadence_phase_no_first_ms": cadence_phase,
        "measured_phase_no_first_ms": measured_phase,
        "cadence_tile_stats": _tile_digest(cadence_tile),
        "measured_tile_stats": _tile_digest(measured_tile),
        "tile_stats_abs_delta": tile_delta,
        "tile_stats_equal": tile_delta == 0.0,
        "workload_explains_render_forward_miss": tile_delta > 0.0 and render_ratio >= 1.0,
        "no_first_timing_miss": step_ratio >= 1.0,
        "render_forward_timing_miss": render_ratio >= 1.0,
        "cadence_cache_rebuilds": int(cadence_row.get("projective_interval_cache_rebuilds") or 0),
        "measured_cache_rebuilds": int(measured_row.get("projective_interval_cache_rebuilds") or 0),
        "measured_support_rebins": int(measured_row.get("projective_interval_cache_support_rebins") or 0),
        "measured_stale_refreshes": int(measured_row.get("projective_interval_cache_stale_refreshes") or 0),
        "measured_fallback_marks": int(measured_row.get("projective_interval_cache_fallback_marks") or 0),
        "measured_visibility_stratifications": int(
            measured_row.get("projective_interval_cache_visibility_stratifications") or 0
        ),
        "measured_tile_overflow_sum": int(measured_row.get("tile_overflow_sum") or 0),
        "end_loss_abs_delta": abs(float(measured_row.get("end_loss") or 0.0) - float(cadence_row.get("end_loss") or 0.0)),
    }


def build_pair_profiles(source: dict[str, Any], case_dir: Path) -> list[dict[str, Any]]:
    frame_counts = [int(value) for value in source.get("frame_counts", [])]
    profiles: list[dict[str, Any]] = []
    for scene in source.get("scenes", []):
        scene_id = str(scene["scene_id"])
        for frames in frame_counts:
            profiles.append(_pair_profile(source, case_dir, scene_id, frames))
    return sorted(profiles, key=lambda row: (str(row["scene_id"]), int(row["frames"])))


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    profiles = report["pair_profiles"]
    source_summary = report["source_summary"]
    no_first_misses = [row for row in profiles if row["no_first_timing_miss"]]
    render_misses = [row for row in profiles if row["render_forward_timing_miss"]]
    worst_render = max(profiles, key=lambda row: float(row["render_forward_ratio"]))
    worst_step = max(profiles, key=lambda row: float(row["case_no_first_step_ratio"]))
    return {
        "source_status": report["source_status"],
        "source_scene_count": int(source_summary["scene_count"]),
        "source_row_count": int(source_summary["row_count"]),
        "strict_failure_count": len(report["source_errors"]),
        "strict_failed_only_expected_timing": tuple(report["source_errors"]) == EXPECTED_STRICT_TIMING_ERRORS,
        "pair_count": len(profiles),
        "no_first_miss_pair_count": len(no_first_misses),
        "render_forward_miss_pair_count": len(render_misses),
        "no_first_miss_render_forward_miss_pair_count": sum(
            1 for row in no_first_misses if row["render_forward_timing_miss"]
        ),
        "all_policy_tile_stats_identical": all(bool(row["tile_stats_equal"]) for row in profiles),
        "all_no_first_misses_tile_stats_identical": all(bool(row["tile_stats_equal"]) for row in no_first_misses),
        "all_render_forward_misses_tile_stats_identical": all(bool(row["tile_stats_equal"]) for row in render_misses),
        "workload_explains_render_forward_miss_count": sum(
            1 for row in render_misses if row["workload_explains_render_forward_miss"]
        ),
        "max_tile_stats_abs_delta": max(float(row["tile_stats_abs_delta"]) for row in profiles),
        "max_no_first_miss_tile_stats_abs_delta": max(float(row["tile_stats_abs_delta"]) for row in no_first_misses),
        "max_render_forward_ratio": float(worst_render["render_forward_ratio"]),
        "max_render_forward_ratio_scene_id": str(worst_render["scene_id"]),
        "max_render_forward_ratio_frames": int(worst_render["frames"]),
        "max_render_forward_ratio_tile_stats_identical": bool(worst_render["tile_stats_equal"]),
        "max_step_ratio": float(worst_step["case_no_first_step_ratio"]),
        "max_step_ratio_scene_id": str(worst_step["scene_id"]),
        "max_step_ratio_frames": int(worst_step["frames"]),
        "max_render_forward_delta_ms": max(float(row["render_forward_delta_ms"]) for row in profiles),
        "max_no_first_miss_render_forward_delta_ms": max(
            float(row["render_forward_delta_ms"]) for row in no_first_misses
        ),
        "max_render_forward_ms_per_clipped_ref_ratio": max(
            float(row["render_forward_ms_per_clipped_ref_ratio"]) for row in profiles
        ),
        "max_render_forward_ms_per_active_tile_ratio": max(
            float(row["render_forward_ms_per_active_tile_ratio"]) for row in profiles
        ),
        "max_source_case_no_first_abs_delta": max(
            float(row["source_case_no_first_abs_delta"]) for row in profiles
        ),
        "all_profile_step_no_first_matches_source": max(
            float(row["source_case_no_first_abs_delta"]) for row in profiles
        )
        < 1.0e-9,
        "all_pairs_cache_support_clean": all(
            int(row["measured_cache_rebuilds"]) < int(row["cadence_cache_rebuilds"])
            and int(row["measured_support_rebins"]) == 0
            and int(row["measured_stale_refreshes"]) == 0
            and int(row["measured_fallback_marks"]) == 0
            and int(row["measured_visibility_stratifications"]) == 0
            and int(row["measured_tile_overflow_sum"]) == 0
            for row in profiles
        ),
        "all_pairs_losses_match_cadence": max(float(row["end_loss_abs_delta"]) for row in profiles) < 1.0e-5,
        "max_loss_delta": max(float(row["end_loss_abs_delta"]) for row in profiles),
    }


def verify_extended_render_forward_residual_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_real_video_multiscene_extended_render_forward_residual":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    theory_contract = report.get("theory_contract")
    if (
        not isinstance(theory_contract, str)
        or "does not prove a timing win" not in theory_contract
        or "render-forward residual" not in theory_contract
        or "saved tile_stats" not in theory_contract
    ):
        errors.append("theory_contract must preserve render-forward residual diagnostic scope")
    if tuple(report.get("source_errors") or ()) != EXPECTED_STRICT_TIMING_ERRORS:
        errors.append("source_errors must remain exactly the expected strict timing failures")
    if report.get("source_status") != "failed":
        errors.append(f"source_status must remain failed, got {report.get('source_status')!r}")
    source_summary = report.get("source_summary")
    profiles = report.get("pair_profiles")
    if not isinstance(source_summary, dict):
        errors.append("source_summary must be an object")
        return errors
    if not isinstance(profiles, list) or not profiles:
        errors.append("pair_profiles must be a non-empty list")
        return errors
    if int(source_summary.get("scene_count") or 0) < 5:
        errors.append("render-forward residual source must cover at least five scenes")
    if int(source_summary.get("row_count") or 0) < 30:
        errors.append("render-forward residual source must cover at least 30 rows")
    for row in profiles:
        if not isinstance(row, dict):
            errors.append("pair profile rows must be objects")
            continue
        label = f"{row.get('scene_id')} {row.get('frames')}f"
        for key in (
            "source_no_first_step_ratio",
            "case_no_first_step_ratio",
            "source_case_no_first_abs_delta",
            "render_forward_ratio",
            "render_forward_delta_ms",
            "render_forward_ms_per_clipped_ref_ratio",
            "render_forward_ms_per_active_tile_ratio",
            "tile_stats_abs_delta",
            "end_loss_abs_delta",
        ):
            _finite_float(row.get(key), f"{label} {key}", errors)
        for key in ("cadence_phase_no_first_ms", "measured_phase_no_first_ms", "cadence_tile_stats", "measured_tile_stats"):
            if not isinstance(row.get(key), dict):
                errors.append(f"{label} {key} must be an object")
        if _finite_float(row.get("source_case_no_first_abs_delta"), f"{label} source/case delta", errors) >= 1.0e-9:
            errors.append(f"{label} case step no-first mean must match source row")
        if bool(row.get("no_first_timing_miss")) and _finite_float(
            row.get("case_no_first_step_ratio"), f"{label} step ratio", errors
        ) < 1.0:
            errors.append(f"{label} no-first miss flag must have step ratio >= 1")
        if bool(row.get("render_forward_timing_miss")) and _finite_float(
            row.get("render_forward_ratio"), f"{label} render ratio", errors
        ) < 1.0:
            errors.append(f"{label} render-forward miss flag must have render ratio >= 1")
        if _finite_float(row.get("end_loss_abs_delta"), f"{label} loss delta", errors) >= 1.0e-5:
            errors.append(f"{label} loss delta must stay below 1e-5")
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
        errors.append("render-forward residual must fail only the expected strict timing gates")
    if int(summary.get("no_first_miss_pair_count") or 0) <= 0:
        errors.append("render-forward residual must include no-first miss pairs")
    if int(summary.get("render_forward_miss_pair_count") or 0) <= 0:
        errors.append("render-forward residual must include render-forward miss pairs")
    if summary.get("all_policy_tile_stats_identical") is not True:
        errors.append("cadence/measured tile_stats must be identical for all saved pairs")
    if summary.get("all_no_first_misses_tile_stats_identical") is not True:
        errors.append("no-first timing misses must preserve identical tile_stats")
    if summary.get("all_render_forward_misses_tile_stats_identical") is not True:
        errors.append("render-forward timing misses must preserve identical tile_stats")
    if int(summary.get("workload_explains_render_forward_miss_count") or 0) != 0:
        errors.append("saved tile_stats must not explain render-forward misses")
    if summary.get("max_render_forward_ratio_tile_stats_identical") is not True:
        errors.append("max render-forward miss must have identical tile_stats")
    if summary.get("all_profile_step_no_first_matches_source") is not True:
        errors.append("case no-first step means must match source rows")
    if summary.get("all_pairs_cache_support_clean") is not True:
        errors.append("render-forward residual pairs must remain cache/support clean")
    if summary.get("all_pairs_losses_match_cadence") is not True:
        errors.append("render-forward residual pairs must match cadence losses")
    return errors


def assert_extended_render_forward_residual_report(report: dict[str, Any]) -> None:
    errors = verify_extended_render_forward_residual_report(report)
    if errors:
        raise AssertionError("extended render-forward residual failed:\n- " + "\n- ".join(errors))


def run_report(
    source_report: Path = DEFAULT_SOURCE_REPORT,
    case_dir: Path = DEFAULT_CASE_DIR,
) -> dict[str, Any]:
    source = json.loads(source_report.read_text(encoding="utf-8"))
    profiles = build_pair_profiles(source, case_dir)
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_multiscene_extended_render_forward_residual",
        "base_domain": "saved cases from failed strict five-source real-video frame-scaling matrix",
        "theory_contract": (
            "This render-forward residual report uses saved tile_stats and saved per-step timings from "
            "a failed strict five-source run. It does not prove a timing win; it tests whether "
            "render-forward misses are explained by candidate/support workload or by residual per-work-unit latency."
        ),
        "source_report": str(source_report),
        "case_dir": str(case_dir),
        "source_status": source.get("status"),
        "source_errors": list(source.get("errors") or []),
        "source_summary": source.get("summary", {}),
        "pair_profiles": profiles,
    }
    report["summary"] = summarize(report)
    errors = verify_extended_render_forward_residual_report(report)
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
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--verify-report", type=Path)
    args = parser.parse_args()

    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_extended_render_forward_residual_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(source_report=args.source_report, case_dir=args.case_dir)
    if report.get("status") == "ok":
        assert_extended_render_forward_residual_report(report)
    path = write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
