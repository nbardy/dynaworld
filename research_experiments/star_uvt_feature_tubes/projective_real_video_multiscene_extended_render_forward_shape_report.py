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
    / "2026-05-25_star_uvt_projective_real_video_multiscene_extended_render_forward_shape"
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


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


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


def _no_first_timings(case: dict[str, Any]) -> list[dict[str, float]]:
    rows = case.get("step_timings_ms")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("case must contain at least two step_timings_ms entries")
    no_first_rows = [row for row in rows[1:] if isinstance(row, dict)]
    if len(no_first_rows) != len(rows) - 1:
        raise ValueError("case step_timings_ms entries must be objects")
    return [
        {key: float(row.get(key) or 0.0) for key in PHASE_KEYS}
        for row in no_first_rows
    ]


def _phase_values(rows: list[dict[str, float]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def _drop_largest_positive_delta_ratio(measured: list[float], cadence: list[float]) -> tuple[float, int, float]:
    deltas = [m - c for m, c in zip(measured, cadence, strict=True)]
    max_index = max(range(len(deltas)), key=lambda idx: deltas[idx])
    kept_measured = [value for idx, value in enumerate(measured) if idx != max_index]
    kept_cadence = [value for idx, value in enumerate(cadence) if idx != max_index]
    return _ratio(sum(kept_measured), sum(kept_cadence)), max_index, deltas[max_index]


def _pair_profile(source: dict[str, Any], case_dir: Path, scene_id: str, frames: int) -> dict[str, Any]:
    cadence_row = _rows_for(source, scene_id, frames, "cadence")[0]
    measured_row = _rows_for(source, scene_id, frames, "measured")[0]
    cadence_rows = _no_first_timings(_load_case(case_dir, scene_id, frames, "cadence"))
    measured_rows = _no_first_timings(_load_case(case_dir, scene_id, frames, "measured"))
    if len(cadence_rows) != len(measured_rows):
        raise ValueError(f"{scene_id} {frames}f cadence/measured timing lengths differ")
    render_cadence = _phase_values(cadence_rows, "render_forward_ms")
    render_measured = _phase_values(measured_rows, "render_forward_ms")
    step_cadence = _phase_values(cadence_rows, "step_ms")
    step_measured = _phase_values(measured_rows, "step_ms")
    render_step_ratios = [_ratio(m, c) for m, c in zip(render_measured, render_cadence, strict=True)]
    step_step_ratios = [_ratio(m, c) for m, c in zip(step_measured, step_cadence, strict=True)]
    render_no_spike_ratio, render_spike_index, render_spike_delta = _drop_largest_positive_delta_ratio(
        render_measured,
        render_cadence,
    )
    step_no_spike_ratio, step_spike_index, step_spike_delta = _drop_largest_positive_delta_ratio(
        step_measured,
        step_cadence,
    )
    render_ratio = _ratio(sum(render_measured), sum(render_cadence))
    step_ratio = _ratio(sum(step_measured), sum(step_cadence))
    source_ratio = _ratio(float(measured_row["no_first_step_ms"]), float(cadence_row["no_first_step_ms"]))
    return {
        "scene_id": scene_id,
        "youtube_id": str(measured_row.get("youtube_id") or ""),
        "title": str(measured_row.get("title") or ""),
        "frames": frames,
        "no_first_step_count": len(render_step_ratios),
        "motion_score": float(measured_row.get("motion_score") or 0.0),
        "source_no_first_step_ratio": source_ratio,
        "case_no_first_step_ratio": step_ratio,
        "source_case_no_first_abs_delta": abs(step_ratio - source_ratio),
        "render_forward_ratio": render_ratio,
        "render_forward_step_ratios": render_step_ratios,
        "step_step_ratios": step_step_ratios,
        "render_forward_step_deltas_ms": [
            m - c for m, c in zip(render_measured, render_cadence, strict=True)
        ],
        "step_deltas_ms": [m - c for m, c in zip(step_measured, step_cadence, strict=True)],
        "render_forward_max_step_ratio": max(render_step_ratios),
        "render_forward_min_step_ratio": min(render_step_ratios),
        "render_forward_median_step_ratio": _median(render_step_ratios),
        "render_forward_last_step_ratio": render_step_ratios[-1],
        "render_forward_ratio_spread": max(render_step_ratios) - min(render_step_ratios),
        "render_forward_spike_step_index": render_spike_index,
        "render_forward_spike_delta_ms": render_spike_delta,
        "render_forward_drop_largest_positive_delta_ratio": render_no_spike_ratio,
        "step_spike_step_index": step_spike_index,
        "step_spike_delta_ms": step_spike_delta,
        "step_drop_largest_positive_delta_ratio": step_no_spike_ratio,
        "render_forward_single_spike_drives_miss": render_ratio >= 1.0 and render_no_spike_ratio < 1.0,
        "step_single_spike_drives_miss": step_ratio >= 1.0 and step_no_spike_ratio < 1.0,
        "no_first_timing_miss": step_ratio >= 1.0,
        "render_forward_timing_miss": render_ratio >= 1.0,
        "chunk_traces_present": bool(
            _load_case(case_dir, scene_id, frames, "cadence").get("chunk_traces")
            or _load_case(case_dir, scene_id, frames, "measured").get("chunk_traces")
        ),
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
        "chunk_traces_present_pair_count": sum(1 for row in profiles if row["chunk_traces_present"]),
        "max_render_forward_ratio": float(worst_render["render_forward_ratio"]),
        "max_render_forward_ratio_scene_id": str(worst_render["scene_id"]),
        "max_render_forward_ratio_frames": int(worst_render["frames"]),
        "max_step_ratio": float(worst_step["case_no_first_step_ratio"]),
        "max_step_ratio_scene_id": str(worst_step["scene_id"]),
        "max_step_ratio_frames": int(worst_step["frames"]),
        "max_render_forward_step_ratio": max(float(row["render_forward_max_step_ratio"]) for row in profiles),
        "max_render_forward_ratio_spread": max(float(row["render_forward_ratio_spread"]) for row in profiles),
        "max_no_first_miss_render_forward_ratio_spread": max(
            float(row["render_forward_ratio_spread"]) for row in no_first_misses
        ),
        "max_no_first_miss_render_forward_spike_delta_ms": max(
            float(row["render_forward_spike_delta_ms"]) for row in no_first_misses
        ),
        "max_no_first_miss_render_forward_drop_spike_ratio": max(
            float(row["render_forward_drop_largest_positive_delta_ratio"]) for row in no_first_misses
        ),
        "all_no_first_misses_render_single_spike_driven": all(
            bool(row["render_forward_single_spike_drives_miss"])
            for row in no_first_misses
            if bool(row["render_forward_timing_miss"])
        ),
        "all_no_first_misses_step_single_spike_driven": all(
            bool(row["step_single_spike_drives_miss"]) for row in no_first_misses
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


def verify_extended_render_forward_shape_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_real_video_multiscene_extended_render_forward_shape":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    theory_contract = report.get("theory_contract")
    if (
        not isinstance(theory_contract, str)
        or "does not prove a timing win" not in theory_contract
        or "per-step render-forward shape" not in theory_contract
        or "single-step spike" not in theory_contract
    ):
        errors.append("theory_contract must preserve render-forward shape diagnostic scope")
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
        errors.append("render-forward shape source must cover at least five scenes")
    if int(source_summary.get("row_count") or 0) < 30:
        errors.append("render-forward shape source must cover at least 30 rows")
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
            "render_forward_max_step_ratio",
            "render_forward_min_step_ratio",
            "render_forward_median_step_ratio",
            "render_forward_last_step_ratio",
            "render_forward_ratio_spread",
            "render_forward_spike_delta_ms",
            "render_forward_drop_largest_positive_delta_ratio",
            "step_spike_delta_ms",
            "step_drop_largest_positive_delta_ratio",
            "end_loss_abs_delta",
        ):
            _finite_float(row.get(key), f"{label} {key}", errors)
        for key in ("render_forward_step_ratios", "step_step_ratios", "render_forward_step_deltas_ms", "step_deltas_ms"):
            values = row.get(key)
            if not isinstance(values, list) or len(values) != int(row.get("no_first_step_count") or 0):
                errors.append(f"{label} {key} must match no_first_step_count")
            elif not values:
                errors.append(f"{label} {key} must be non-empty")
            else:
                for idx, value in enumerate(values):
                    _finite_float(value, f"{label} {key}[{idx}]", errors)
        ratios = row.get("render_forward_step_ratios")
        if isinstance(ratios, list) and ratios:
            finite_ratios = [float(value) for value in ratios if isinstance(value, int | float)]
            if len(finite_ratios) == len(ratios):
                expected_scalars = {
                    "render_forward_max_step_ratio": max(finite_ratios),
                    "render_forward_min_step_ratio": min(finite_ratios),
                    "render_forward_median_step_ratio": _median(finite_ratios),
                    "render_forward_last_step_ratio": finite_ratios[-1],
                    "render_forward_ratio_spread": max(finite_ratios) - min(finite_ratios),
                }
                for key, expected_value in expected_scalars.items():
                    actual = row.get(key)
                    if not isinstance(actual, int | float) or abs(float(actual) - expected_value) > 1.0e-9:
                        errors.append(f"{label} {key} does not match render_forward_step_ratios")
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
        errors.append("render-forward shape must fail only the expected strict timing gates")
    if int(summary.get("no_first_miss_pair_count") or 0) <= 0:
        errors.append("render-forward shape must include no-first miss pairs")
    if int(summary.get("render_forward_miss_pair_count") or 0) <= 0:
        errors.append("render-forward shape must include render-forward miss pairs")
    if int(summary.get("chunk_traces_present_pair_count") or 0) != 0:
        errors.append("saved strict source should not already contain chunk traces")
    if summary.get("all_no_first_misses_render_single_spike_driven") is not True:
        errors.append("render-forward misses should be single-spike driven in this saved source")
    if summary.get("all_no_first_misses_step_single_spike_driven") is not True:
        errors.append("step misses should be single-spike driven in this saved source")
    if summary.get("all_profile_step_no_first_matches_source") is not True:
        errors.append("case no-first step means must match source rows")
    if summary.get("all_pairs_cache_support_clean") is not True:
        errors.append("render-forward shape pairs must remain cache/support clean")
    if summary.get("all_pairs_losses_match_cadence") is not True:
        errors.append("render-forward shape pairs must match cadence losses")
    return errors


def assert_extended_render_forward_shape_report(report: dict[str, Any]) -> None:
    errors = verify_extended_render_forward_shape_report(report)
    if errors:
        raise AssertionError("extended render-forward shape failed:\n- " + "\n- ".join(errors))


def run_report(
    source_report: Path = DEFAULT_SOURCE_REPORT,
    case_dir: Path = DEFAULT_CASE_DIR,
) -> dict[str, Any]:
    source = json.loads(source_report.read_text(encoding="utf-8"))
    profiles = build_pair_profiles(source, case_dir)
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_multiscene_extended_render_forward_shape",
        "base_domain": "saved cases from failed strict five-source real-video frame-scaling matrix",
        "theory_contract": (
            "This per-step render-forward shape report uses saved step timings from a failed strict "
            "five-source run. It does not prove a timing win; it tests whether render-forward misses "
            "are persistent slowdowns or single-step spike artifacts. Substep attribution still needs "
            "a traced rerun because these saved cases contain no chunk traces."
        ),
        "source_report": str(source_report),
        "case_dir": str(case_dir),
        "source_status": source.get("status"),
        "source_errors": list(source.get("errors") or []),
        "source_summary": source.get("summary", {}),
        "pair_profiles": profiles,
    }
    report["summary"] = summarize(report)
    errors = verify_extended_render_forward_shape_report(report)
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
        assert_extended_render_forward_shape_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(source_report=args.source_report, case_dir=args.case_dir)
    if report.get("status") == "ok":
        assert_extended_render_forward_shape_report(report)
    path = write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
