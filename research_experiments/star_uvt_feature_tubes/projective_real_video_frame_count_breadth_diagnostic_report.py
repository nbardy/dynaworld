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
    / "2026-05-25_star_uvt_projective_real_video_frame_count_breadth_diagnostic"
)
DEFAULT_SOURCE_REPORT = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_multiscene_frame_scaling_matrix_4count"
    / "summary.json"
)
EXPECTED_TIMING_ERRORS = {
    "multiscene frame-scaling measured no-first timing must beat cadence",
    "multiscene frame-scaling no-first timing growth must stay below frame growth",
}


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


def _timing_errors_only(source_errors: list[Any]) -> bool:
    return all(isinstance(error, str) and error in EXPECTED_TIMING_ERRORS for error in source_errors)


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    source_summary = report["source_summary"]
    source_errors = list(report["source_errors"])
    max_no_first = float(source_summary["max_measured_vs_cadence_no_first_step_ms_ratio"])
    max_growth = float(source_summary["max_measured_no_first_growth_vs_frame_growth_ratio"])
    return {
        "source_status": str(report["source_status"]),
        "source_scene_count": int(source_summary["scene_count"]),
        "source_distinct_youtube_id_count": int(source_summary["distinct_youtube_id_count"]),
        "source_row_count": int(source_summary["row_count"]),
        "source_measured_row_count": int(source_summary["measured_row_count"]),
        "source_frame_count_count": int(source_summary["frame_count_count"]),
        "source_frame_growth_factor": float(source_summary["frame_growth_factor"]),
        "strict_failure_count": len(source_errors),
        "strict_failed_only_expected_timing": _timing_errors_only(source_errors),
        "all_source_videos_exist": bool(source_summary["all_source_videos_exist"]),
        "all_rows_pass": bool(source_summary["all_rows_pass"]),
        "all_rows_loss_decreased": bool(source_summary["all_rows_loss_decreased"]),
        "all_rows_no_overflow": bool(source_summary["all_rows_no_overflow"]),
        "all_rows_fallback_free": bool(source_summary["all_rows_fallback_free"]),
        "all_rows_visibility_stratification_free": bool(
            source_summary["all_rows_visibility_stratification_free"]
        ),
        "all_measured_loss_matches_cadence": bool(source_summary["all_measured_loss_matches_cadence"]),
        "max_measured_vs_cadence_end_loss_abs_delta": float(
            source_summary["max_measured_vs_cadence_end_loss_abs_delta"]
        ),
        "max_measured_vs_cadence_rebuild_ratio": float(source_summary["max_measured_vs_cadence_rebuild_ratio"]),
        "max_measured_cache_rebuild_growth": float(source_summary["max_measured_cache_rebuild_growth"]),
        "max_measured_support_rebins": int(source_summary["max_measured_support_rebins"]),
        "max_measured_stale_refreshes": int(source_summary["max_measured_stale_refreshes"]),
        "max_measured_support_tail_alpha_bound": float(source_summary["max_measured_support_tail_alpha_bound"]),
        "max_measured_support_overshoot_px": float(source_summary["max_measured_support_overshoot_px"]),
        "max_motion_score": float(source_summary["max_motion_score"]),
        "max_tile_count": int(source_summary["max_tile_count"]),
        "max_measured_vs_cadence_no_first_step_ms_ratio": max_no_first,
        "max_measured_no_first_growth_vs_frame_growth_ratio": max_growth,
        "no_first_timing_win": max_no_first < 1.0,
        "no_first_growth_sublinear": max_growth < 1.0,
        "max_no_first_ratio_overage": max(0.0, max_no_first - 1.0),
        "max_growth_ratio_overage": max(0.0, max_growth - 1.0),
        "frame_count_breadth_accepted": int(source_summary["frame_count_count"]) >= 4,
    }


def verify_frame_count_breadth_diagnostic_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_real_video_frame_count_breadth_diagnostic":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    theory_contract = report.get("theory_contract")
    if (
        not isinstance(theory_contract, str)
        or "does not prove broad real-scene quality acceptance" not in theory_contract
        or "does not prove a strict timing win" not in theory_contract
        or "frame-count breadth" not in theory_contract
    ):
        errors.append("theory_contract must preserve frame-count breadth scope")

    source_errors = report.get("source_errors")
    if not isinstance(source_errors, list):
        errors.append("source_errors must be a list")
        source_errors = []
    if not _timing_errors_only(source_errors):
        errors.append(f"source_errors must contain only expected strict timing failures, got {source_errors!r}")
    source_summary = report.get("source_summary")
    if not isinstance(source_summary, dict):
        errors.append("source_summary must be an object")
        return errors
    if _finite_int(source_summary.get("scene_count"), "source scene_count", errors) < 3:
        errors.append("source diagnostic must cover at least three scenes")
    if _finite_int(source_summary.get("distinct_youtube_id_count"), "source distinct_youtube_id_count", errors) < 3:
        errors.append("source diagnostic must cover at least three distinct YouTube ids")
    if _finite_int(source_summary.get("frame_count_count"), "source frame_count_count", errors) < 4:
        errors.append("source diagnostic must cover at least four frame counts")
    if _finite_float(source_summary.get("frame_growth_factor"), "source frame growth factor", errors) < 8.0:
        errors.append("source diagnostic must cover at least 8x frame growth")
    if _finite_int(source_summary.get("row_count"), "source row_count", errors) < 24:
        errors.append("source diagnostic must cover at least 24 cadence/measured rows")
    for key in (
        "all_source_videos_exist",
        "all_rows_pass",
        "all_rows_loss_decreased",
        "all_rows_no_overflow",
        "all_rows_fallback_free",
        "all_rows_visibility_stratification_free",
        "all_measured_loss_matches_cadence",
    ):
        if source_summary.get(key) is not True:
            errors.append(f"source diagnostic {key} must be true")
    if _finite_float(source_summary.get("max_measured_vs_cadence_end_loss_abs_delta"), "source loss delta", errors) >= 1.0e-5:
        errors.append("source diagnostic measured/cadence loss delta must stay below 1e-5")
    if _finite_float(source_summary.get("max_measured_vs_cadence_rebuild_ratio"), "source rebuild ratio", errors) > 0.5:
        errors.append("source diagnostic measured rebuild ratio must stay at or below 0.5")
    if _finite_float(source_summary.get("max_measured_cache_rebuild_growth"), "source rebuild growth", errors) > 1.0:
        errors.append("source diagnostic measured rebuild count must not grow")
    if _finite_int(source_summary.get("max_measured_support_rebins"), "source support rebins", errors) != 0:
        errors.append("source diagnostic must have zero measured support rebins")
    if _finite_int(source_summary.get("max_measured_stale_refreshes"), "source stale refreshes", errors) != 0:
        errors.append("source diagnostic must have zero measured stale refreshes")
    if _finite_float(source_summary.get("max_measured_support_tail_alpha_bound"), "source support tail", errors) > 1.0e-3:
        errors.append("source diagnostic support tail bound must stay below 1e-3")
    if _finite_float(source_summary.get("max_measured_no_first_growth_vs_frame_growth_ratio"), "source timing growth ratio", errors) >= 1.0:
        errors.append("source diagnostic measured timing growth must stay below frame growth")

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
    if summary.get("frame_count_breadth_accepted") is not True:
        errors.append("frame-count breadth must be accepted")
    if summary.get("strict_failed_only_expected_timing") is not True:
        errors.append("diagnostic must fail only expected strict timing gates")
    return errors


def assert_frame_count_breadth_diagnostic_report(report: dict[str, Any]) -> None:
    errors = verify_frame_count_breadth_diagnostic_report(report)
    if errors:
        raise AssertionError("frame-count breadth diagnostic failed:\n- " + "\n- ".join(errors))


def run_report(source_report: Path = DEFAULT_SOURCE_REPORT) -> dict[str, Any]:
    source = json.loads(source_report.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_frame_count_breadth_diagnostic",
        "base_domain": "failed strict source-distinct real-video four-frame-count matrix",
        "theory_contract": (
            "This report does not prove broad real-scene quality acceptance and does not prove a strict timing win. "
            "It accepts frame-count breadth when the source frame-scaling matrix covers at least four frame counts "
            "with clean quality/cache/support/fallback invariants, while preserving strict timing failures separately."
        ),
        "source_report": str(source_report),
        "source_status": source.get("status"),
        "source_errors": list(source.get("errors") or []),
        "source_summary": source.get("summary", {}),
    }
    report["summary"] = summarize(report)
    errors = verify_frame_count_breadth_diagnostic_report(report)
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
        assert_frame_count_breadth_diagnostic_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(source_report=args.source_report)
    if report.get("status") == "ok":
        assert_frame_count_breadth_diagnostic_report(report)
    path = write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
