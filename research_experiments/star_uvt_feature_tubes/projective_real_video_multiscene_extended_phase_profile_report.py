from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
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
    / "2026-05-25_star_uvt_projective_real_video_multiscene_extended_phase_profile"
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
PROFILE_PHASES = (
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
    return values[-1] / values[0]


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _phase_means(case: dict[str, Any], phases: tuple[str, ...] = PROFILE_PHASES) -> dict[str, float]:
    steps = case.get("step_timings_ms")
    if not isinstance(steps, list) or len(steps) < 2:
        raise ValueError("case must contain at least two step_timings_ms entries")
    rows = [row for row in steps[1:] if isinstance(row, dict)]
    if len(rows) != len(steps) - 1:
        raise ValueError("case step_timings_ms entries must be objects")
    means = {"step_ms": _mean([float(row["step_ms"]) for row in rows])}
    for phase in phases:
        means[phase] = _mean([float(row.get(phase) or 0.0) for row in rows])
    return means


def _case_path(case_dir: Path, scene_id: str, frames: int, policy: str) -> Path:
    return case_dir / f"{scene_id}_{frames}f_{policy}.json"


def _case_phase_means(case_dir: Path, scene_id: str, frames: int, policy: str) -> dict[str, float]:
    path = _case_path(case_dir, scene_id, frames, policy)
    return _phase_means(json.loads(path.read_text(encoding="utf-8")))


def _profile_keys(source: dict[str, Any]) -> list[tuple[str, int, tuple[str, ...]]]:
    frame_counts = [int(value) for value in source.get("frame_counts", [])]
    frame_growth = _ratio(float(frame_counts[-1]), float(frame_counts[0])) if frame_counts else 0.0
    keys: dict[tuple[str, int], set[str]] = {}
    for scene in source.get("scenes", []):
        scene_id = str(scene["scene_id"])
        measured_rows = _policy_rows_for_scene(source, scene_id, "measured")
        measured_growth_ratio = _ratio(
            _growth([float(row["no_first_step_ms"]) for row in measured_rows]),
            frame_growth,
        )
        if measured_growth_ratio >= 1.0 and frame_counts:
            keys.setdefault((scene_id, frame_counts[0]), set()).add("growth_endpoint")
            keys.setdefault((scene_id, frame_counts[-1]), set()).add("growth_endpoint")
        for frames in frame_counts:
            cadence = _rows_for(source, scene_id, frames, "cadence")[0]
            measured = _rows_for(source, scene_id, frames, "measured")[0]
            no_first_ratio = _ratio(float(measured["no_first_step_ms"]), float(cadence["no_first_step_ms"]))
            if no_first_ratio >= 1.0:
                keys.setdefault((scene_id, frames), set()).add("no_first_miss")
    return [(scene_id, frames, tuple(sorted(reasons))) for (scene_id, frames), reasons in sorted(keys.items())]


def _phase_profile_row(
    source: dict[str, Any],
    case_dir: Path,
    scene_id: str,
    frames: int,
    reasons: tuple[str, ...],
) -> dict[str, Any]:
    cadence_row = _rows_for(source, scene_id, frames, "cadence")[0]
    measured_row = _rows_for(source, scene_id, frames, "measured")[0]
    cadence_phase = _case_phase_means(case_dir, scene_id, frames, "cadence")
    measured_phase = _case_phase_means(case_dir, scene_id, frames, "measured")
    phase_ratios = {
        key: _ratio(measured_phase[key], cadence_phase[key])
        for key in ("step_ms", *PROFILE_PHASES)
    }
    phase_deltas = {
        key: measured_phase[key] - cadence_phase[key]
        for key in ("step_ms", *PROFILE_PHASES)
    }
    dominant_candidates = {
        key: delta
        for key, delta in phase_deltas.items()
        if key != "step_ms" and delta > 0.0
    }
    if dominant_candidates:
        dominant_phase, dominant_delta = max(dominant_candidates.items(), key=lambda item: item[1])
    else:
        dominant_phase, dominant_delta = "none", 0.0
    source_ratio = _ratio(float(measured_row["no_first_step_ms"]), float(cadence_row["no_first_step_ms"]))
    case_ratio = phase_ratios["step_ms"]
    return {
        "scene_id": scene_id,
        "youtube_id": str(measured_row.get("youtube_id") or ""),
        "title": str(measured_row.get("title") or ""),
        "frames": frames,
        "reasons": list(reasons),
        "motion_score": float(measured_row.get("motion_score") or 0.0),
        "source_no_first_ratio": source_ratio,
        "case_step_no_first_ratio": case_ratio,
        "source_case_no_first_abs_delta": abs(case_ratio - source_ratio),
        "cadence_phase_no_first_ms": cadence_phase,
        "measured_phase_no_first_ms": measured_phase,
        "phase_ratios": phase_ratios,
        "phase_deltas_ms": phase_deltas,
        "dominant_positive_phase": dominant_phase,
        "dominant_positive_phase_delta_ms": dominant_delta,
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


def build_phase_profiles(source: dict[str, Any], case_dir: Path) -> list[dict[str, Any]]:
    return [
        _phase_profile_row(source, case_dir, scene_id, frames, reasons)
        for scene_id, frames, reasons in _profile_keys(source)
    ]


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    profiles = report["phase_profiles"]
    source_summary = report["source_summary"]
    no_first_profiles = [row for row in profiles if "no_first_miss" in row["reasons"]]
    growth_profiles = [row for row in profiles if "growth_endpoint" in row["reasons"]]
    worst_step = max(profiles, key=lambda row: float(row["phase_ratios"]["step_ms"]))
    worst_render = max(profiles, key=lambda row: float(row["phase_ratios"]["render_forward_ms"]))
    worst_backward = max(profiles, key=lambda row: float(row["phase_ratios"]["backward_ms"]))
    dominant_counts = Counter(
        str(row["dominant_positive_phase"])
        for row in no_first_profiles
        if str(row["dominant_positive_phase"]) != "none"
    )
    return {
        "source_status": report["source_status"],
        "source_scene_count": int(source_summary["scene_count"]),
        "source_row_count": int(source_summary["row_count"]),
        "strict_failure_count": len(report["source_errors"]),
        "strict_failed_only_expected_timing": tuple(report["source_errors"]) == EXPECTED_STRICT_TIMING_ERRORS,
        "phase_profile_count": len(profiles),
        "no_first_miss_profile_count": len(no_first_profiles),
        "growth_endpoint_profile_count": len(growth_profiles),
        "profile_scene_count": len({row["scene_id"] for row in profiles}),
        "max_source_case_no_first_abs_delta": max(float(row["source_case_no_first_abs_delta"]) for row in profiles),
        "all_profile_step_no_first_matches_source": max(
            float(row["source_case_no_first_abs_delta"]) for row in profiles
        )
        < 1.0e-9,
        "max_profile_step_ratio": float(worst_step["phase_ratios"]["step_ms"]),
        "max_profile_step_ratio_scene_id": str(worst_step["scene_id"]),
        "max_profile_step_ratio_frames": int(worst_step["frames"]),
        "max_render_forward_ratio": float(worst_render["phase_ratios"]["render_forward_ms"]),
        "max_render_forward_ratio_scene_id": str(worst_render["scene_id"]),
        "max_render_forward_ratio_frames": int(worst_render["frames"]),
        "max_backward_ratio": float(worst_backward["phase_ratios"]["backward_ms"]),
        "max_backward_ratio_scene_id": str(worst_backward["scene_id"]),
        "max_backward_ratio_frames": int(worst_backward["frames"]),
        "dominant_positive_phase_counts_for_no_first_misses": dict(sorted(dominant_counts.items())),
        "max_dominant_positive_phase_delta_ms": max(
            float(row["dominant_positive_phase_delta_ms"]) for row in profiles
        ),
        "all_profile_pairs_cache_support_clean": all(
            int(row["measured_cache_rebuilds"]) < int(row["cadence_cache_rebuilds"])
            and int(row["measured_support_rebins"]) == 0
            and int(row["measured_stale_refreshes"]) == 0
            and int(row["measured_fallback_marks"]) == 0
            and int(row["measured_visibility_stratifications"]) == 0
            and int(row["measured_tile_overflow_sum"]) == 0
            for row in profiles
        ),
        "all_profile_losses_match_cadence": max(float(row["end_loss_abs_delta"]) for row in profiles) < 1.0e-5,
        "max_profile_rebuild_ratio": max(
            _ratio(float(row["measured_cache_rebuilds"]), float(row["cadence_cache_rebuilds"]))
            for row in profiles
        ),
        "max_profile_loss_delta": max(float(row["end_loss_abs_delta"]) for row in profiles),
        "all_no_first_misses_have_positive_phase_delta": all(
            float(row["dominant_positive_phase_delta_ms"]) > 0.0 for row in no_first_profiles
        ),
    }


def verify_extended_phase_profile_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_real_video_multiscene_extended_phase_profile":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    theory_contract = report.get("theory_contract")
    if (
        not isinstance(theory_contract, str)
        or "does not prove a timing win" not in theory_contract
        or "phase-profile" not in theory_contract
        or "saved per-step timings" not in theory_contract
    ):
        errors.append("theory_contract must preserve phase-profile diagnostic scope")
    if tuple(report.get("source_errors") or ()) != EXPECTED_STRICT_TIMING_ERRORS:
        errors.append("source_errors must remain exactly the expected strict timing failures")
    if report.get("source_status") != "failed":
        errors.append(f"source_status must remain failed, got {report.get('source_status')!r}")
    source_summary = report.get("source_summary")
    profiles = report.get("phase_profiles")
    if not isinstance(source_summary, dict):
        errors.append("source_summary must be an object")
        return errors
    if not isinstance(profiles, list) or not profiles:
        errors.append("phase_profiles must be a non-empty list")
        return errors
    if int(source_summary.get("scene_count") or 0) < 5:
        errors.append("phase-profile source must cover at least five scenes")
    if int(source_summary.get("row_count") or 0) < 30:
        errors.append("phase-profile source must cover at least 30 rows")
    for row in profiles:
        if not isinstance(row, dict):
            errors.append("phase profile rows must be objects")
            continue
        label = f"{row.get('scene_id')} {row.get('frames')}f"
        reasons = row.get("reasons")
        if not isinstance(reasons, list) or not reasons:
            errors.append(f"{label} must name profile reasons")
        phase_ratios = row.get("phase_ratios")
        phase_deltas = row.get("phase_deltas_ms")
        cadence_phase = row.get("cadence_phase_no_first_ms")
        measured_phase = row.get("measured_phase_no_first_ms")
        for name, value in (
            ("phase_ratios", phase_ratios),
            ("phase_deltas_ms", phase_deltas),
            ("cadence_phase_no_first_ms", cadence_phase),
            ("measured_phase_no_first_ms", measured_phase),
        ):
            if not isinstance(value, dict):
                errors.append(f"{label} {name} must be an object")
        if not all(isinstance(value, dict) for value in (phase_ratios, phase_deltas, cadence_phase, measured_phase)):
            continue
        for key in ("step_ms", *PROFILE_PHASES):
            _finite_float(phase_ratios.get(key), f"{label} phase ratio {key}", errors)
            _finite_float(phase_deltas.get(key), f"{label} phase delta {key}", errors)
            _finite_float(cadence_phase.get(key), f"{label} cadence phase {key}", errors)
            _finite_float(measured_phase.get(key), f"{label} measured phase {key}", errors)
        if _finite_float(row.get("source_case_no_first_abs_delta"), f"{label} source/case delta", errors) >= 1.0e-9:
            errors.append(f"{label} case step no-first mean must match source row")
        if "no_first_miss" in reasons and _finite_float(phase_ratios.get("step_ms"), f"{label} step ratio", errors) < 1.0:
            errors.append(f"{label} no_first_miss reason must have step ratio >= 1")
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
        errors.append("phase profile must fail only the expected strict timing gates")
    if int(summary.get("no_first_miss_profile_count") or 0) <= 0:
        errors.append("phase profile must include no-first miss rows")
    if int(summary.get("growth_endpoint_profile_count") or 0) <= 0:
        errors.append("phase profile must include frame-growth endpoint rows")
    if summary.get("all_profile_step_no_first_matches_source") is not True:
        errors.append("phase profile no-first step means must match source rows")
    if summary.get("all_profile_pairs_cache_support_clean") is not True:
        errors.append("phase profile pairs must remain cache/support clean")
    if summary.get("all_profile_losses_match_cadence") is not True:
        errors.append("phase profile losses must match cadence")
    if summary.get("all_no_first_misses_have_positive_phase_delta") is not True:
        errors.append("no-first miss rows must expose a positive phase delta")
    return errors


def assert_extended_phase_profile_report(report: dict[str, Any]) -> None:
    errors = verify_extended_phase_profile_report(report)
    if errors:
        raise AssertionError("extended phase profile failed:\n- " + "\n- ".join(errors))


def run_report(
    source_report: Path = DEFAULT_SOURCE_REPORT,
    case_dir: Path = DEFAULT_CASE_DIR,
) -> dict[str, Any]:
    source = json.loads(source_report.read_text(encoding="utf-8"))
    profiles = build_phase_profiles(source, case_dir)
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_multiscene_extended_phase_profile",
        "base_domain": "saved cases from failed strict five-source real-video frame-scaling matrix",
        "theory_contract": (
            "This phase-profile report uses saved per-step timings from a failed strict five-source run. "
            "It does not prove a timing win; it identifies which measured/cadence phases explain the "
            "timing misses while preserving cache/support/cadence invariants."
        ),
        "source_report": str(source_report),
        "case_dir": str(case_dir),
        "source_status": source.get("status"),
        "source_errors": list(source.get("errors") or []),
        "source_summary": source.get("summary", {}),
        "phase_profiles": profiles,
    }
    report["summary"] = summarize(report)
    errors = verify_extended_phase_profile_report(report)
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
        assert_extended_phase_profile_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(source_report=args.source_report, case_dir=args.case_dir)
    if report.get("status") == "ok":
        assert_extended_phase_profile_report(report)
    path = write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
