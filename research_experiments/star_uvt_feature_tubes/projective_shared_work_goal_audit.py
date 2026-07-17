from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_experiments.star_uvt_feature_tubes.projective_orbit_fixed_chart_scaling_benchmark import (  # noqa: E402
    verify_orbit_fixed_chart_scaling_report,
)
from research_experiments.star_uvt_feature_tubes.projective_exposure_rolling_backward_report import (  # noqa: E402
    verify_exposure_rolling_backward_report,
)
from research_experiments.star_uvt_feature_tubes.projective_exposure_rolling_quadrature_report import (  # noqa: E402
    verify_exposure_rolling_quadrature_report,
)
from research_experiments.star_uvt_feature_tubes.projective_exposure_rolling_mixed_fallback_backward_report import (  # noqa: E402
    verify_mixed_fallback_backward_report,
)
from research_experiments.star_uvt_feature_tubes.projective_trained_high_motion_trace_scaling_benchmark import (  # noqa: E402
    verify_trained_high_motion_trace_scaling_report,
)


DEFAULT_OUT_DIR = ROOT / "outputs" / "benchmarks" / "2026-05-25_star_uvt_projective_shared_work_goal_audit"
DEFAULT_ORBIT_REPORT = ROOT / "outputs" / "benchmarks" / "2026-05-24_star_uvt_revolving_orbit_fixed_chart_scaling" / "summary.json"
DEFAULT_EXPOSURE_QUADRATURE_REPORT = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_exposure_rolling_quadrature"
    / "summary.json"
)
DEFAULT_EXPOSURE_BACKWARD_REPORT = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_exposure_rolling_backward"
    / "summary.json"
)
DEFAULT_EXPOSURE_MIXED_FALLBACK_BACKWARD_REPORT = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_exposure_rolling_mixed_fallback_backward"
    / "summary.json"
)
DEFAULT_TRAINED_REPORTS = (
    ROOT / "outputs" / "benchmarks" / "2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling" / "summary.json",
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_64px_128t"
    / "summary.json",
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_96px_256t_cap256"
    / "summary.json",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _growth(values: list[float]) -> float:
    if not values or values[0] == 0.0:
        return math.inf
    return float(values[-1]) / float(values[0])


def _rows_for(report: dict[str, Any], key: str, value: str) -> list[dict[str, Any]]:
    return sorted(
        [row for row in report.get("rows", []) if row.get(key) == value],
        key=lambda row: int(row.get("frames", 0)),
    )


def _orbit_audit(path: Path) -> dict[str, Any]:
    report = _load_json(path)
    underlying_errors = verify_orbit_fixed_chart_scaling_report(report)
    fixed = _rows_for(report, "route", "fixed_chart")
    per_frame = _rows_for(report, "route", "per_frame")
    final_fixed = fixed[-1]
    final_per_frame = per_frame[-1]
    return {
        "path": str(path),
        "underlying_errors": underlying_errors,
        "frame_counts": report.get("frame_counts"),
        "fixed_payload_growth": _growth([float(row["atlas_payload_bytes"]) for row in fixed]),
        "per_frame_payload_growth": _growth([float(row["atlas_payload_bytes"]) for row in per_frame]),
        "final_payload_ratio": float(final_fixed["atlas_payload_bytes"]) / float(final_per_frame["atlas_payload_bytes"]),
        "final_trace_ratio": float(final_fixed["trace_count"]) / float(final_per_frame["trace_count"]),
        "final_segment_ratio": float(final_fixed["segment_count"]) / float(final_per_frame["segment_count"]),
        "final_forward_ms_ratio": float(final_fixed["forward_ms"]) / float(final_per_frame["forward_ms"]),
        "final_backward_ms_ratio": float(final_fixed["backward_ms"]) / float(final_per_frame["backward_ms"]),
        "final_cpu_compile_ms_ratio": float(final_fixed["cpu_compile_ms"]) / float(final_per_frame["cpu_compile_ms"]),
    }


def _trained_audit(path: Path) -> dict[str, Any]:
    report = _load_json(path)
    underlying_errors = verify_trained_high_motion_trace_scaling_report(report)
    shared = _rows_for(report, "label", "trained_checkpoint")
    per_frame = _rows_for(report, "label", "trained_checkpoint_per_frame")
    final_shared = shared[-1]
    final_per_frame = per_frame[-1]
    return {
        "path": str(path),
        "underlying_errors": underlying_errors,
        "frame_counts": report.get("frame_counts"),
        "size": report.get("size"),
        "tube_count": report.get("tube_count"),
        "tile_capacity": report.get("tile_capacity"),
        "shared_interval_entry_growth": _growth([float(row["interval_trace_entries"]) for row in shared]),
        "per_frame_interval_entry_growth": _growth([float(row["interval_trace_entries"]) for row in per_frame]),
        "final_interval_entry_ratio": float(final_shared["interval_trace_entries"])
        / float(final_per_frame["interval_trace_entries"]),
        "final_trace_count_ratio": float(final_shared["trace_count"]) / float(final_per_frame["trace_count"]),
        "final_backward_ms_ratio": float(final_shared["backward_ms"]) / float(final_per_frame["backward_ms"]),
        "final_forward_ms_ratio": float(final_shared["forward_ms"]) / float(final_per_frame["forward_ms"]),
    }


def _exposure_quadrature_audit(path: Path) -> dict[str, Any]:
    report = _load_json(path)
    underlying_errors = verify_exposure_rolling_quadrature_report(report)
    summary = report.get("summary", {})
    return {
        "path": str(path),
        "underlying_errors": underlying_errors,
        "finite_reference_lowered_max_abs_error": summary.get("finite_reference_lowered_max_abs_error"),
        "rolling_rowwise_batched_max_abs_error": summary.get("rolling_rowwise_batched_max_abs_error"),
        "rolling_unique_to_row_sample_ratio": summary.get("rolling_unique_to_row_sample_ratio"),
        "finite_fallback_fraction": summary.get("finite_fallback_fraction"),
        "rolling_fallback_fraction": summary.get("rolling_fallback_fraction"),
        "max_metal_abs_error": summary.get("max_metal_abs_error"),
        "metal_case_count": summary.get("metal_case_count"),
    }


def _exposure_backward_audit(path: Path) -> dict[str, Any]:
    report = _load_json(path)
    underlying_errors = verify_exposure_rolling_backward_report(report)
    summary = report.get("summary", {})
    return {
        "path": str(path),
        "underlying_errors": underlying_errors,
        "finite_has_metal_backward": summary.get("finite_has_metal_backward"),
        "rolling_has_metal_backward": summary.get("rolling_has_metal_backward"),
        "rolling_unique_to_row_sample_ratio": summary.get("rolling_unique_to_row_sample_ratio"),
        "max_metal_grad_abs_error": summary.get("max_metal_grad_abs_error"),
        "max_metal_grad_rel_error": summary.get("max_metal_grad_rel_error"),
        "metal_backward_case_count": summary.get("metal_backward_case_count"),
    }


def _exposure_mixed_fallback_backward_audit(path: Path) -> dict[str, Any]:
    report = _load_json(path)
    underlying_errors = verify_mixed_fallback_backward_report(report)
    summary = report.get("summary", {})
    return {
        "path": str(path),
        "underlying_errors": underlying_errors,
        "finite_has_mixed_backward": summary.get("finite_has_mixed_backward"),
        "rolling_has_mixed_backward": summary.get("rolling_has_mixed_backward"),
        "finite_fallback_fraction": summary.get("finite_fallback_fraction"),
        "rolling_fallback_fraction": summary.get("rolling_fallback_fraction"),
        "rolling_unique_to_row_sample_ratio": summary.get("rolling_unique_to_row_sample_ratio"),
        "max_mixed_output_abs_error": summary.get("max_mixed_output_abs_error"),
        "max_mixed_grad_abs_error": summary.get("max_mixed_grad_abs_error"),
        "max_mixed_grad_rel_error": summary.get("max_mixed_grad_rel_error"),
        "mixed_backward_case_count": summary.get("mixed_backward_case_count"),
    }


def summarize(
    orbit: dict[str, Any],
    trained: list[dict[str, Any]],
    exposure_quadrature: dict[str, Any],
    exposure_backward: dict[str, Any],
    exposure_mixed_fallback_backward: dict[str, Any],
) -> dict[str, Any]:
    return {
        "orbit_fixed_payload_growth": orbit["fixed_payload_growth"],
        "orbit_per_frame_payload_growth": orbit["per_frame_payload_growth"],
        "orbit_payload_growth_ratio": orbit["fixed_payload_growth"] / orbit["per_frame_payload_growth"],
        "orbit_final_payload_ratio": orbit["final_payload_ratio"],
        "orbit_final_trace_ratio": orbit["final_trace_ratio"],
        "orbit_final_segment_ratio": orbit["final_segment_ratio"],
        "orbit_final_forward_ms_ratio": orbit["final_forward_ms_ratio"],
        "orbit_final_backward_ms_ratio": orbit["final_backward_ms_ratio"],
        "orbit_final_cpu_compile_ms_ratio": orbit["final_cpu_compile_ms_ratio"],
        "trained_artifact_count": len(trained),
        "max_trained_final_interval_entry_ratio": max(row["final_interval_entry_ratio"] for row in trained),
        "max_trained_final_trace_count_ratio": max(row["final_trace_count_ratio"] for row in trained),
        "max_trained_final_forward_ms_ratio": max(row["final_forward_ms_ratio"] for row in trained),
        "max_trained_final_backward_ms_ratio": max(row["final_backward_ms_ratio"] for row in trained),
        "max_trained_shared_interval_entry_growth": max(row["shared_interval_entry_growth"] for row in trained),
        "min_trained_per_frame_interval_entry_growth": min(row["per_frame_interval_entry_growth"] for row in trained),
        "trained_shared_to_replay_interval_growth_ratio": max(
            row["shared_interval_entry_growth"] for row in trained
        )
        / min(row["per_frame_interval_entry_growth"] for row in trained),
        "exposure_forward_rolling_unique_to_row_sample_ratio": exposure_quadrature[
            "rolling_unique_to_row_sample_ratio"
        ],
        "exposure_forward_max_metal_abs_error": exposure_quadrature["max_metal_abs_error"],
        "exposure_forward_metal_case_count": exposure_quadrature["metal_case_count"],
        "exposure_backward_rolling_unique_to_row_sample_ratio": exposure_backward["rolling_unique_to_row_sample_ratio"],
        "exposure_backward_max_metal_grad_abs_error": exposure_backward["max_metal_grad_abs_error"],
        "exposure_backward_max_metal_grad_rel_error": exposure_backward["max_metal_grad_rel_error"],
        "exposure_backward_metal_case_count": exposure_backward["metal_backward_case_count"],
        "exposure_mixed_fallback_rolling_unique_to_row_sample_ratio": exposure_mixed_fallback_backward[
            "rolling_unique_to_row_sample_ratio"
        ],
        "exposure_mixed_fallback_max_output_abs_error": exposure_mixed_fallback_backward[
            "max_mixed_output_abs_error"
        ],
        "exposure_mixed_fallback_max_grad_abs_error": exposure_mixed_fallback_backward[
            "max_mixed_grad_abs_error"
        ],
        "exposure_mixed_fallback_max_grad_rel_error": exposure_mixed_fallback_backward[
            "max_mixed_grad_rel_error"
        ],
        "exposure_mixed_fallback_backward_case_count": exposure_mixed_fallback_backward[
            "mixed_backward_case_count"
        ],
    }


def _finite_float(value: Any, label: str, errors: list[str]) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        errors.append(f"{label} must be finite, got {value!r}")
        return 0.0
    return float(value)


def _finite_int(value: Any, label: str, errors: list[str]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{label} must be an integer, got {value!r}")
        return 0
    return int(value)


def _strictly_increasing_ints(values: Any, label: str, errors: list[str]) -> list[int]:
    if not isinstance(values, list) or len(values) < 2:
        errors.append(f"{label} must contain at least two frame counts")
        return []
    frame_counts = [_finite_int(value, f"{label}[{idx}]", errors) for idx, value in enumerate(values)]
    if frame_counts != sorted(frame_counts) or len(set(frame_counts)) != len(frame_counts):
        errors.append(f"{label} must be strictly increasing, got {frame_counts}")
    if any(value <= 0 for value in frame_counts):
        errors.append(f"{label} must contain positive frame counts, got {frame_counts}")
    return frame_counts


def _assert_summary_close(
    summary: dict[str, Any],
    expected: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    atol: float = 1.0e-9,
) -> None:
    actual = summary.get(key)
    expected_value = expected.get(key)
    if isinstance(expected_value, float):
        if not isinstance(actual, int | float) or abs(float(actual) - expected_value) > atol:
            errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")
    elif actual != expected_value:
        errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")


def run_report(
    *,
    orbit_report: Path = DEFAULT_ORBIT_REPORT,
    trained_reports: tuple[Path, ...] = DEFAULT_TRAINED_REPORTS,
    exposure_quadrature_report: Path = DEFAULT_EXPOSURE_QUADRATURE_REPORT,
    exposure_backward_report: Path = DEFAULT_EXPOSURE_BACKWARD_REPORT,
    exposure_mixed_fallback_backward_report: Path = DEFAULT_EXPOSURE_MIXED_FALLBACK_BACKWARD_REPORT,
) -> dict[str, Any]:
    orbit = _orbit_audit(orbit_report)
    trained = [_trained_audit(path) for path in trained_reports]
    exposure_quadrature = _exposure_quadrature_audit(exposure_quadrature_report)
    exposure_backward = _exposure_backward_audit(exposure_backward_report)
    exposure_mixed_fallback_backward = _exposure_mixed_fallback_backward_audit(exposure_mixed_fallback_backward_report)
    report = {
        "status": "ok",
        "benchmark": "star_uvt_projective_shared_work_goal_audit",
        "theory_contract": (
            "Known camera-path traces should share projection/support/binning/payload and backward work so "
            "non-pixel world-side cost grows sublinearly with frame count, including finite-exposure and "
            "rolling-shutter evaluation/backward reuse with differentiable visibility fallback."
        ),
        "orbit": orbit,
        "trained": trained,
        "exposure_quadrature": exposure_quadrature,
        "exposure_backward": exposure_backward,
        "exposure_mixed_fallback_backward": exposure_mixed_fallback_backward,
        "summary": summarize(
            orbit,
            trained,
            exposure_quadrature,
            exposure_backward,
            exposure_mixed_fallback_backward,
        ),
    }
    errors = verify_shared_work_goal_audit(report)
    if errors:
        report["status"] = "failed"
        report["errors"] = errors
    return report


def verify_shared_work_goal_audit(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_shared_work_goal_audit":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    theory_contract = report.get("theory_contract")
    if (
        not isinstance(theory_contract, str)
        or "sublinear" not in theory_contract
        or "backward" not in theory_contract
        or "fallback" not in theory_contract
    ):
        errors.append("theory_contract must state sublinear backward/shared-work objective with fallback")
    orbit = report.get("orbit")
    trained = report.get("trained")
    exposure_quadrature = report.get("exposure_quadrature")
    exposure_backward = report.get("exposure_backward")
    exposure_mixed_fallback_backward = report.get("exposure_mixed_fallback_backward")
    summary = report.get("summary")
    if not isinstance(orbit, dict):
        errors.append("orbit must be an object")
        return errors
    if not isinstance(trained, list) or len(trained) < 3:
        errors.append("trained must contain at least three artifacts")
        return errors
    if not isinstance(exposure_quadrature, dict):
        errors.append("exposure_quadrature must be an object")
        return errors
    if not isinstance(exposure_backward, dict):
        errors.append("exposure_backward must be an object")
        return errors
    if not isinstance(exposure_mixed_fallback_backward, dict):
        errors.append("exposure_mixed_fallback_backward must be an object")
        return errors
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        return errors

    if not isinstance(orbit.get("path"), str) or not orbit.get("path"):
        errors.append("orbit path must be a nonempty string")
    _strictly_increasing_ints(orbit.get("frame_counts"), "orbit frame_counts", errors)
    if orbit.get("underlying_errors"):
        errors.append(f"orbit underlying verifier failed: {orbit.get('underlying_errors')}")
    for key in (
        "fixed_payload_growth",
        "per_frame_payload_growth",
        "final_payload_ratio",
        "final_trace_ratio",
        "final_segment_ratio",
        "final_forward_ms_ratio",
        "final_backward_ms_ratio",
        "final_cpu_compile_ms_ratio",
    ):
        value = _finite_float(orbit.get(key), f"orbit {key}", errors)
        if value <= 0.0:
            errors.append(f"orbit {key} must be positive")

    seen_trained_paths: set[str] = set()
    seen_trained_sizes: set[int] = set()
    for idx, row in enumerate(trained):
        if not isinstance(row, dict):
            errors.append(f"trained row {idx} must be an object")
            continue
        path = row.get("path")
        if not isinstance(path, str) or not path:
            errors.append(f"trained artifact {idx} path must be a nonempty string")
        elif path in seen_trained_paths:
            errors.append(f"trained artifact {idx} duplicates path {path!r}")
        else:
            seen_trained_paths.add(path)
        _strictly_increasing_ints(row.get("frame_counts"), f"trained artifact {idx} frame_counts", errors)
        size = _finite_int(row.get("size"), f"trained artifact {idx} size", errors)
        tube_count = _finite_int(row.get("tube_count"), f"trained artifact {idx} tube_count", errors)
        tile_capacity = _finite_int(row.get("tile_capacity"), f"trained artifact {idx} tile_capacity", errors)
        if size <= 0:
            errors.append(f"trained artifact {idx} size must be positive")
        if size in seen_trained_sizes:
            errors.append(f"trained artifact {idx} duplicates size {size}")
        else:
            seen_trained_sizes.add(size)
        if tube_count <= 0:
            errors.append(f"trained artifact {idx} tube_count must be positive")
        if tile_capacity <= 0:
            errors.append(f"trained artifact {idx} tile_capacity must be positive")
        if row.get("underlying_errors"):
            errors.append(f"trained artifact {idx} underlying verifier failed: {row.get('underlying_errors')}")
        for key in (
            "shared_interval_entry_growth",
            "per_frame_interval_entry_growth",
            "final_interval_entry_ratio",
            "final_trace_count_ratio",
            "final_backward_ms_ratio",
            "final_forward_ms_ratio",
        ):
            value = _finite_float(row.get(key), f"trained artifact {idx} {key}", errors)
            if value <= 0.0:
                errors.append(f"trained artifact {idx} {key} must be positive")

    if not isinstance(exposure_quadrature.get("path"), str) or not exposure_quadrature.get("path"):
        errors.append("exposure_quadrature path must be a nonempty string")
    if exposure_quadrature.get("underlying_errors"):
        errors.append(f"exposure_quadrature underlying verifier failed: {exposure_quadrature.get('underlying_errors')}")
    for key in (
        "finite_reference_lowered_max_abs_error",
        "rolling_rowwise_batched_max_abs_error",
        "rolling_unique_to_row_sample_ratio",
        "finite_fallback_fraction",
        "rolling_fallback_fraction",
        "max_metal_abs_error",
    ):
        value = _finite_float(exposure_quadrature.get(key), f"exposure_quadrature {key}", errors)
        if key.endswith("error") and value < 0.0:
            errors.append(f"exposure_quadrature {key} must be nonnegative")
    exposure_metal_case_count = _finite_int(
        exposure_quadrature.get("metal_case_count"),
        "exposure_quadrature metal_case_count",
        errors,
    )

    if not isinstance(exposure_backward.get("path"), str) or not exposure_backward.get("path"):
        errors.append("exposure_backward path must be a nonempty string")
    if exposure_backward.get("underlying_errors"):
        errors.append(f"exposure_backward underlying verifier failed: {exposure_backward.get('underlying_errors')}")
    for key in ("finite_has_metal_backward", "rolling_has_metal_backward"):
        if not isinstance(exposure_backward.get(key), bool):
            errors.append(f"exposure_backward {key} must be boolean")
    for key in (
        "rolling_unique_to_row_sample_ratio",
        "max_metal_grad_abs_error",
        "max_metal_grad_rel_error",
    ):
        value = _finite_float(exposure_backward.get(key), f"exposure_backward {key}", errors)
        if key.endswith("error") and value < 0.0:
            errors.append(f"exposure_backward {key} must be nonnegative")
    exposure_backward_case_count = _finite_int(
        exposure_backward.get("metal_backward_case_count"),
        "exposure_backward metal_backward_case_count",
        errors,
    )

    if not isinstance(exposure_mixed_fallback_backward.get("path"), str) or not exposure_mixed_fallback_backward.get("path"):
        errors.append("exposure_mixed_fallback_backward path must be a nonempty string")
    if exposure_mixed_fallback_backward.get("underlying_errors"):
        errors.append(
            "exposure_mixed_fallback_backward underlying verifier failed: "
            f"{exposure_mixed_fallback_backward.get('underlying_errors')}"
        )
    for key in ("finite_has_mixed_backward", "rolling_has_mixed_backward"):
        if not isinstance(exposure_mixed_fallback_backward.get(key), bool):
            errors.append(f"exposure_mixed_fallback_backward {key} must be boolean")
    for key in (
        "finite_fallback_fraction",
        "rolling_fallback_fraction",
        "rolling_unique_to_row_sample_ratio",
        "max_mixed_output_abs_error",
        "max_mixed_grad_abs_error",
        "max_mixed_grad_rel_error",
    ):
        value = _finite_float(
            exposure_mixed_fallback_backward.get(key),
            f"exposure_mixed_fallback_backward {key}",
            errors,
        )
        if ("error" in key or key.endswith("fraction")) and value < 0.0:
            errors.append(f"exposure_mixed_fallback_backward {key} must be nonnegative")
    exposure_mixed_fallback_case_count = _finite_int(
        exposure_mixed_fallback_backward.get("mixed_backward_case_count"),
        "exposure_mixed_fallback_backward mixed_backward_case_count",
        errors,
    )

    try:
        expected_summary = summarize(
            orbit,
            [row for row in trained if isinstance(row, dict)],
            exposure_quadrature,
            exposure_backward,
            exposure_mixed_fallback_backward,
        )
        for key in expected_summary:
            _assert_summary_close(summary, expected_summary, key, errors)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"summary could not be recomputed: {exc}")

    if int(summary.get("trained_artifact_count") or -1) != len(trained):
        errors.append("summary trained_artifact_count must match trained rows")
    if float(summary.get("orbit_fixed_payload_growth") or math.inf) > 1.05:
        errors.append("orbit fixed payload growth must stay near constant")
    if float(summary.get("orbit_per_frame_payload_growth") or 0.0) < 4.0:
        errors.append("orbit per-frame payload growth must expose framewise replay cost")
    if float(summary.get("orbit_payload_growth_ratio") or math.inf) >= 0.20:
        errors.append("orbit fixed/per-frame payload growth ratio must stay below 0.20")
    if float(summary.get("orbit_final_payload_ratio") or math.inf) >= 0.25:
        errors.append("orbit final fixed/per-frame payload ratio must be below 0.25")
    if float(summary.get("orbit_final_trace_ratio") or math.inf) >= 0.25:
        errors.append("orbit final fixed/per-frame trace ratio must be below 0.25")
    if float(summary.get("orbit_final_segment_ratio") or math.inf) >= 0.25:
        errors.append("orbit final fixed/per-frame segment ratio must be below 0.25")
    if float(summary.get("orbit_final_cpu_compile_ms_ratio") or math.inf) >= 0.5:
        errors.append("orbit final fixed/per-frame CPU compile ratio must be below 0.5")
    if float(summary.get("orbit_final_forward_ms_ratio") or math.inf) >= 0.5:
        errors.append("orbit final fixed/per-frame forward ratio must be below 0.5")
    if float(summary.get("orbit_final_backward_ms_ratio") or math.inf) >= 0.5:
        errors.append("orbit final fixed/per-frame backward ratio must be below 0.5")
    if float(summary.get("max_trained_final_interval_entry_ratio") or math.inf) >= 0.20:
        errors.append("trained final shared/per-frame interval-entry ratios must stay below 0.20")
    if float(summary.get("max_trained_final_trace_count_ratio") or math.inf) >= 0.20:
        errors.append("trained final shared/per-frame trace-count ratios must stay below 0.20")
    if float(summary.get("max_trained_final_forward_ms_ratio") or math.inf) >= 0.75:
        errors.append("trained final shared/per-frame forward ratios must stay below 0.75")
    if float(summary.get("max_trained_final_backward_ms_ratio") or math.inf) >= 0.25:
        errors.append("trained final shared/per-frame backward ratios must stay below 0.25")
    if float(summary.get("max_trained_shared_interval_entry_growth") or math.inf) >= 2.0:
        errors.append("trained shared interval-entry growth must stay below 2x")
    if float(summary.get("min_trained_per_frame_interval_entry_growth") or 0.0) <= 4.0:
        errors.append("trained per-frame replay interval-entry growth must exceed 4x")
    if float(summary.get("trained_shared_to_replay_interval_growth_ratio") or math.inf) >= 0.25:
        errors.append("trained shared/replay interval-entry growth ratio must stay below 0.25")
    if _finite_float(
        exposure_quadrature.get("rolling_unique_to_row_sample_ratio"),
        "exposure_quadrature rolling_unique_to_row_sample_ratio",
        errors,
    ) >= 1.0:
        errors.append("exposure_quadrature rolling sample reuse ratio must stay below 1")
    if _finite_float(
        exposure_backward.get("rolling_unique_to_row_sample_ratio"),
        "exposure_backward rolling_unique_to_row_sample_ratio",
        errors,
    ) >= 1.0:
        errors.append("exposure_backward rolling sample reuse ratio must stay below 1")
    if _finite_float(
        exposure_mixed_fallback_backward.get("rolling_unique_to_row_sample_ratio"),
        "exposure_mixed_fallback_backward rolling_unique_to_row_sample_ratio",
        errors,
    ) >= 1.0:
        errors.append("exposure_mixed_fallback_backward rolling sample reuse ratio must stay below 1")
    if _finite_float(
        exposure_quadrature.get("finite_reference_lowered_max_abs_error"),
        "exposure_quadrature finite_reference_lowered_max_abs_error",
        errors,
    ) > 1.0e-6:
        errors.append("exposure_quadrature finite reference lowering must stay exact")
    if _finite_float(
        exposure_quadrature.get("rolling_rowwise_batched_max_abs_error"),
        "exposure_quadrature rolling_rowwise_batched_max_abs_error",
        errors,
    ) > 1.0e-6:
        errors.append("exposure_quadrature rolling batched reference must stay exact")
    for key in ("finite_fallback_fraction", "rolling_fallback_fraction"):
        fraction = _finite_float(exposure_quadrature.get(key), f"exposure_quadrature {key}", errors)
        if not 0.0 < fraction < 1.0:
            errors.append(f"exposure_quadrature {key} must keep mixed fast/fallback coverage")
    if _finite_float(
        exposure_quadrature.get("max_metal_abs_error"),
        "exposure_quadrature max_metal_abs_error",
        errors,
    ) > 3.0e-4:
        errors.append("exposure_quadrature Metal error must stay below 3e-4")
    if exposure_metal_case_count < 4:
        errors.append("exposure_quadrature must verify all four Metal forward/fallback cases")
    if exposure_backward.get("finite_has_metal_backward") is not True:
        errors.append("exposure_backward must include finite Metal backward")
    if exposure_backward.get("rolling_has_metal_backward") is not True:
        errors.append("exposure_backward must include rolling Metal backward")
    if exposure_backward_case_count < 2:
        errors.append("exposure_backward must verify both finite and rolling Metal backward cases")
    if _finite_float(
        exposure_backward.get("max_metal_grad_abs_error"),
        "exposure_backward max_metal_grad_abs_error",
        errors,
    ) > 1.0e-3:
        errors.append("exposure_backward Metal gradient abs error must stay below 1e-3")
    if _finite_float(
        exposure_backward.get("max_metal_grad_rel_error"),
        "exposure_backward max_metal_grad_rel_error",
        errors,
    ) > 5.0e-3:
        errors.append("exposure_backward Metal gradient rel error must stay below 5e-3")
    if exposure_mixed_fallback_backward.get("finite_has_mixed_backward") is not True:
        errors.append("exposure_mixed_fallback_backward must include finite mixed fallback backward")
    if exposure_mixed_fallback_backward.get("rolling_has_mixed_backward") is not True:
        errors.append("exposure_mixed_fallback_backward must include rolling mixed fallback backward")
    if exposure_mixed_fallback_case_count < 2:
        errors.append("exposure_mixed_fallback_backward must verify both finite and rolling mixed fallback cases")
    for key in ("finite_fallback_fraction", "rolling_fallback_fraction"):
        fraction = _finite_float(
            exposure_mixed_fallback_backward.get(key),
            f"exposure_mixed_fallback_backward {key}",
            errors,
        )
        if not 0.0 < fraction < 1.0:
            errors.append(f"exposure_mixed_fallback_backward {key} must keep mixed fast/fallback coverage")
    if _finite_float(
        exposure_mixed_fallback_backward.get("max_mixed_output_abs_error"),
        "exposure_mixed_fallback_backward max_mixed_output_abs_error",
        errors,
    ) > 3.0e-4:
        errors.append("exposure_mixed_fallback_backward output error must stay below 3e-4")
    if _finite_float(
        exposure_mixed_fallback_backward.get("max_mixed_grad_abs_error"),
        "exposure_mixed_fallback_backward max_mixed_grad_abs_error",
        errors,
    ) > 1.0e-3:
        errors.append("exposure_mixed_fallback_backward gradient abs error must stay below 1e-3")
    if _finite_float(
        exposure_mixed_fallback_backward.get("max_mixed_grad_rel_error"),
        "exposure_mixed_fallback_backward max_mixed_grad_rel_error",
        errors,
    ) > 5.0e-3:
        errors.append("exposure_mixed_fallback_backward gradient rel error must stay below 5e-3")
    return errors


def assert_shared_work_goal_audit(report: dict[str, Any]) -> None:
    errors = verify_shared_work_goal_audit(report)
    if errors:
        raise AssertionError("shared work goal audit failed:\n- " + "\n- ".join(errors))


def _compare_current_value(
    saved: Any,
    current: Any,
    label: str,
    errors: list[str],
    *,
    atol: float = 1.0e-9,
) -> None:
    if isinstance(current, dict):
        if not isinstance(saved, dict):
            errors.append(f"saved report differs from current inputs at {label}: expected object")
            return
        for key, current_value in current.items():
            child_label = f"{label}.{key}" if label else str(key)
            _compare_current_value(saved.get(key), current_value, child_label, errors, atol=atol)
        return
    if isinstance(current, list):
        if not isinstance(saved, list) or len(saved) != len(current):
            errors.append(
                f"saved report differs from current inputs at {label}: "
                f"expected list length {len(current)}, got {len(saved) if isinstance(saved, list) else type(saved).__name__}"
            )
            return
        for idx, (saved_value, current_value) in enumerate(zip(saved, current, strict=True)):
            _compare_current_value(saved_value, current_value, f"{label}[{idx}]", errors, atol=atol)
        return
    if isinstance(current, float):
        if not isinstance(saved, int | float) or abs(float(saved) - current) > atol:
            errors.append(f"saved report differs from current inputs at {label}: expected {current!r}, got {saved!r}")
        return
    if saved != current:
        errors.append(f"saved report differs from current inputs at {label}: expected {current!r}, got {saved!r}")


def verify_shared_work_goal_current_acceptance(
    saved_report: dict[str, Any],
    *,
    current_report: dict[str, Any] | None = None,
) -> list[str]:
    """Check that a saved aggregate report still matches current default inputs."""

    errors = [f"saved report: {error}" for error in verify_shared_work_goal_audit(saved_report)]
    current = run_report() if current_report is None else current_report
    errors.extend(f"current input report: {error}" for error in verify_shared_work_goal_audit(current))
    if errors:
        return errors
    for key in (
        "benchmark",
        "status",
        "theory_contract",
        "orbit",
        "trained",
        "exposure_quadrature",
        "exposure_backward",
        "exposure_mixed_fallback_backward",
        "summary",
    ):
        _compare_current_value(saved_report.get(key), current.get(key), key, errors)
    return errors


def assert_shared_work_goal_current_acceptance(
    saved_report: dict[str, Any],
    *,
    current_report: dict[str, Any] | None = None,
) -> None:
    errors = verify_shared_work_goal_current_acceptance(saved_report, current_report=current_report)
    if errors:
        raise AssertionError("shared work current-input acceptance failed:\n- " + "\n- ".join(errors))


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# STAR UVT Projective Shared-Work Goal Audit",
        "",
        "This report audits saved orbit and trained high-motion artifacts against the active goal:",
        "",
        "```text",
        "share projection/support/binning/payload and backward work across time",
        "```",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(report["summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## Orbit Payload",
        "",
        "```json",
        json.dumps(report["orbit"], indent=2, sort_keys=True),
        "```",
        "",
        "## Trained High-Motion Rows",
        "",
        "| size | tubes | final entry ratio | final backward ratio | shared entry growth | per-frame entry growth |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["trained"]:
        lines.append(
            "| {size} | {tube_count} | {final_interval_entry_ratio:.6g} | {final_backward_ms_ratio:.6g} | {shared_interval_entry_growth:.6g} | {per_frame_interval_entry_growth:.6g} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Exposure And Rolling",
            "",
            "```json",
            json.dumps(
                {
                    "exposure_quadrature": report["exposure_quadrature"],
                    "exposure_backward": report["exposure_backward"],
                    "exposure_mixed_fallback_backward": report["exposure_mixed_fallback_backward"],
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
        ]
    )
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--verify-report", type=Path, default=None)
    parser.add_argument(
        "--verify-current-inputs",
        action="store_true",
        help="also require the saved report to match a fresh audit of the current default input artifacts",
    )
    args = parser.parse_args()

    if args.verify_report is not None:
        report = _load_json(args.verify_report)
        if args.verify_current_inputs:
            assert_shared_work_goal_current_acceptance(report)
            print(f"verified {args.verify_report} against current inputs")
        else:
            assert_shared_work_goal_audit(report)
            print(f"verified {args.verify_report}")
        return

    if args.verify_current_inputs:
        report = _load_json(args.out_dir / "summary.json")
        assert_shared_work_goal_current_acceptance(report)
        print(f"verified {args.out_dir / 'summary.json'} against current inputs")
        return

    report = run_report()
    assert_shared_work_goal_audit(report)
    write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
