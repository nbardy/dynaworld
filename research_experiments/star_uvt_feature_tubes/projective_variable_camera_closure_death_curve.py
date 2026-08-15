from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STAR_UVT_ROOT = ROOT / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
BENCHMARK = "world_tubes_variable_camera_closure_death_curve"
SCHEMA_VERSION = 1
DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-07-28_world_tubes_variable_camera_closure_death_curve"
)
DEFAULT_HALF_SPANS_DEGREES = (
    5.0,
    15.0,
    30.0,
    45.0,
    60.0,
    75.0,
    90.0,
    105.0,
    120.0,
    135.0,
    150.0,
    165.0,
    175.0,
    178.0,
    179.0,
)
WORLD_VJP_PARAMETER_NAMES = ("point_x", "base_depth", "vertical", "opacity", "color")
COMPILED_ROW_STATUS = "compiled_quality_evaluated"
UNRESOLVED_ROW_STATUS = "compiler_unresolved_death_boundary"
COMPILED_QUALITY_AVAILABLE = "available"
COMPILED_QUALITY_UNAVAILABLE = "structurally_unavailable_compiler_unresolved"
UNAVAILABLE_COMPILED_QUALITY_FIELDS = (
    "support_event_count",
    "visibility_event_count",
    "event_count",
    "event_interval_count",
    "reference_support_policy",
    "reference_order_policy",
    "reference_fallback_reason",
    "reference_sample_semantics",
    "reference_cell_count",
    "reference_live_sorted_cell_count",
    "trace_count",
    "trace_to_replay_ratio",
    "cell_count",
    "visibility_stratum_split_cell_count",
    "interval_entry_count",
    "dense_trace_samples",
    "interval_to_dense_ratio",
    "fallback_cell_count",
    "fallback_cell_fraction",
    "fallback_trace_samples",
    "fallback_sample_fraction",
    "fallback_reasons",
    "invalid_sample_count",
    "invalid_sample_fraction",
    "post_visibility_stale",
    "post_order_mismatch_sample_count",
    "post_ambiguous_depth_sample_count",
    "image_mse",
    "image_psnr_db",
    "image_p999_abs_error",
    "image_max_abs_error",
    "world_vjp_rel_l2_by_parameter",
    "world_vjp_rel_l2_max",
    "world_vjp_parameter_names",
    "world_vjp_reference_norm_by_parameter",
    "world_vjp_compiled_norm_by_parameter",
    "world_vjp_nonzero_parameter_count",
    "vjp_topology_semantics",
)
REQUIRED_IMPLEMENTATION_SOURCE_PATHS = (
    "research_experiments/star_uvt_feature_tubes/projective_variable_camera_closure_death_curve.py",
    "third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/__init__.py",
    "third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py",
)
REQUIRED_DEATH_REASONS = (
    "unresolved_chart",
    "image_psnr_db",
    "image_p999_abs_error",
    "image_max_abs_error",
    "world_vjp_rel_l2_max",
    "fallback_cell_fraction",
    "fallback_sample_fraction",
    "invalid_sample_fraction",
    "trace_to_replay_ratio",
    "interval_to_dense_ratio",
    "post_visibility_stale",
)


class VariableCameraCurveExecutionError(RuntimeError):
    def __init__(
        self,
        *,
        failed_half_span_degrees: float,
        completed_row_count: int,
        cause: Exception,
    ) -> None:
        super().__init__(
            f"camera row {failed_half_span_degrees:g} degrees failed after "
            f"{completed_row_count} completed rows: {type(cause).__name__}: {cause}"
        )
        self.failed_half_span_degrees = float(failed_half_span_degrees)
        self.completed_row_count = int(completed_row_count)
        self.cause_type = type(cause).__name__
        self.cause_message = str(cause)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance() -> dict[str, Any]:
    def git(*args: str, cwd: Path) -> str:
        return subprocess.check_output(
            ("git", *args),
            cwd=cwd,
            text=True,
        ).strip()

    return {
        "repository_commit": git("rev-parse", "HEAD", cwd=ROOT),
        "repository_dirty": bool(git("status", "--porcelain", cwd=ROOT)),
        "star_uvt_commit": git("rev-parse", "HEAD", cwd=STAR_UVT_ROOT),
        "star_uvt_dirty": bool(
            git("status", "--porcelain", cwd=STAR_UVT_ROOT)
        ),
    }


def require_clean_source(source: dict[str, Any]) -> None:
    dirty = [
        key
        for key in ("repository_dirty", "star_uvt_dirty")
        if source.get(key) is not False
    ]
    invalid_commits = [
        key
        for key in ("repository_commit", "star_uvt_commit")
        if not isinstance(source.get(key), str)
        or len(source[key]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in source[key].lower()
        )
    ]
    if dirty or invalid_commits:
        details = []
        if dirty:
            details.append("dirty flags: " + ", ".join(dirty))
        if invalid_commits:
            details.append("invalid commits: " + ", ".join(invalid_commits))
        raise RuntimeError(
            "variable-camera paper evidence requires clean exact source; "
            + "; ".join(details)
        )


def default_world_fixture() -> dict[str, Any]:
    """Return the fixed canonical world shared by every camera-motion row."""

    return {
        "fixture_id": "bounded_yaw_orbit_three_tube_world_v1",
        "parameterization": (
            "three fixed world points with opacity/color; homogeneous pinhole "
            "projection is derived from the bounded camera program"
        ),
        "primitives": [
            {
                "point_x": 0.25,
                "base_depth": 2.5,
                "vertical": 0.10,
                "opacity": 0.55,
                "color": [0.90, 0.12, 0.05],
            },
            {
                "point_x": -0.20,
                "base_depth": 4.5,
                "vertical": -0.10,
                "opacity": 0.45,
                "color": [0.05, 0.75, 0.20],
            },
            {
                "point_x": 0.10,
                "base_depth": 6.5,
                "vertical": 0.00,
                "opacity": 0.35,
                "color": [0.08, 0.20, 0.92],
            },
        ],
    }


def default_camera_program(*, frames: int, image_size: int) -> dict[str, Any]:
    """Return the fixed camera contract; only half-span changes across rows."""

    return {
        "program_family": "bounded_yaw_orbit_tan_half_angle",
        "physical_parameter": "s",
        "physical_interval": [-1.0, 1.0],
        "sample_count": int(frames),
        "sample_schedule": "uniform_closed_interval",
        "angle_law": "theta(s)=half_span_radians*s",
        "chart_coordinate": "q=tan(theta/2)",
        "projection_model": "homogeneous_pinhole",
        "focal_px": 18.0,
        "principal_point_px": [0.5 * float(image_size), 0.5 * float(image_size)],
        "orbit_depth_offset": 0.25,
        "image_width": int(image_size),
        "image_height": int(image_size),
        "bounded_scope": "open camera path with total yaw strictly below 360 degrees",
        "excluded_claims": ["360_degree_transition", "720_degree_transition", "closed_loop_holonomy"],
    }


def default_compiler_contract(
    *,
    tile_size: int,
    sigma_px: float,
    support_padding_px: float,
    max_residual_uv: float,
    max_depth_residual: float,
    min_denominator_abs: float,
    max_windows: int,
) -> dict[str, Any]:
    return {
        "compiler_family": "projective_star_uvt_local_interval_atlas",
        "trace_family": "quadratic_homogeneous_tan_half_angle",
        "local_evaluator": "degree_1_polynomial",
        "sampled_max_residual_uv_px": float(max_residual_uv),
        "sampled_max_depth_residual": float(max_depth_residual),
        "fit_residual_semantics": "empirical_max_over_requested_samples",
        "min_denominator_abs": float(min_denominator_abs),
        "min_valid_fraction": 1.0,
        "max_windows": int(max_windows),
        "tile_size": int(tile_size),
        "sigma_px": float(sigma_px),
        "support_padding_px": float(support_padding_px),
        "event_partition": "continuous_support_plus_visibility_roots",
        "visibility_stratification": "continuous_depth_order_roots",
        "fallback_policy": "live_depth_sort_for_marked_cells",
        "reference": "exact_rational_trace_full_image_all_live_depth_sort",
        "reference_support_policy": "full_image",
        "reference_order_policy": "all_live_depth_per_sample",
        "reference_fallback_reason": "oracle_all_live_depth_sort",
        "reference_sample_semantics": "empirical_at_requested_samples",
        "vjp_scope": (
            "fixed compiled topology; world geometry, opacity, and color; "
            "no visibility-boundary derivative"
        ),
    }


def default_thresholds() -> dict[str, float]:
    """Parity thresholds are strict; compression thresholds name atlas death."""

    return {
        "min_image_psnr_db": 50.0,
        "max_image_p999_abs_error": 2.0 / 255.0,
        "max_image_abs_error": 4.0 / 255.0,
        "max_world_vjp_rel_l2": 0.02,
        "max_fallback_cell_fraction": 0.20,
        "max_fallback_sample_fraction": 0.20,
        "max_invalid_sample_fraction": 0.0,
        "max_trace_to_replay_ratio": 0.50,
        "max_interval_to_dense_ratio": 0.80,
    }


def camera_program_for_span(base: dict[str, Any], half_span_degrees: float) -> dict[str, Any]:
    program = dict(base)
    program["motion_half_span_degrees"] = float(half_span_degrees)
    program["motion_total_span_degrees"] = 2.0 * float(half_span_degrees)
    program["angular_speed_degrees_per_physical_unit"] = float(half_span_degrees)
    return program


def _row_request_payload(
    *,
    request_index: int,
    half_span_degrees: float,
    requested_sweep_sha256: str,
    world_fixture_sha256: str,
    camera_program_sha256: str,
    compiler_sha256: str,
    source_sha256: str,
) -> dict[str, Any]:
    return {
        "request_index": int(request_index),
        "motion_half_span_degrees": float(half_span_degrees),
        "motion_total_span_degrees": 2.0 * float(half_span_degrees),
        "requested_sweep_sha256": requested_sweep_sha256,
        "world_fixture_sha256": world_fixture_sha256,
        "camera_program_sha256": camera_program_sha256,
        "compiler_sha256": compiler_sha256,
        "source_sha256": source_sha256,
    }


def row_death_reasons(row: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    if row.get("row_status") == UNRESOLVED_ROW_STATUS:
        return ["unresolved_chart"] if int(row.get("unresolved_chart_count", 0)) > 0 else []
    reasons: list[str] = []
    if int(row.get("unresolved_chart_count", 0)) != 0:
        reasons.append("unresolved_chart")
    if float(row.get("image_psnr_db", -math.inf)) < float(thresholds["min_image_psnr_db"]):
        reasons.append("image_psnr_db")
    if float(row.get("image_p999_abs_error", math.inf)) > float(
        thresholds["max_image_p999_abs_error"]
    ):
        reasons.append("image_p999_abs_error")
    if float(row.get("image_max_abs_error", math.inf)) > float(
        thresholds["max_image_abs_error"]
    ):
        reasons.append("image_max_abs_error")
    if float(row.get("world_vjp_rel_l2_max", math.inf)) > float(
        thresholds["max_world_vjp_rel_l2"]
    ):
        reasons.append("world_vjp_rel_l2_max")
    if float(row.get("fallback_cell_fraction", math.inf)) > float(
        thresholds["max_fallback_cell_fraction"]
    ):
        reasons.append("fallback_cell_fraction")
    if float(row.get("fallback_sample_fraction", math.inf)) > float(
        thresholds["max_fallback_sample_fraction"]
    ):
        reasons.append("fallback_sample_fraction")
    if float(row.get("invalid_sample_fraction", math.inf)) > float(
        thresholds["max_invalid_sample_fraction"]
    ):
        reasons.append("invalid_sample_fraction")
    if float(row.get("trace_to_replay_ratio", math.inf)) >= float(
        thresholds["max_trace_to_replay_ratio"]
    ):
        reasons.append("trace_to_replay_ratio")
    if float(row.get("interval_to_dense_ratio", math.inf)) >= float(
        thresholds["max_interval_to_dense_ratio"]
    ):
        reasons.append("interval_to_dense_ratio")
    if bool(row.get("post_visibility_stale", True)):
        reasons.append("post_visibility_stale")
    return reasons


def bind_row_contract(
    row: dict[str, Any],
    *,
    world_fixture_sha256: str,
    camera_program: dict[str, Any],
    compiler_sha256: str,
    source_sha256: str,
    requested_sweep_sha256: str,
    request_index: int,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    bound = dict(row)
    bound["world_fixture_sha256"] = world_fixture_sha256
    bound["camera_program_sha256"] = _sha256_json(camera_program)
    bound["compiler_sha256"] = compiler_sha256
    bound["source_sha256"] = source_sha256
    request_payload = _row_request_payload(
        request_index=request_index,
        half_span_degrees=float(bound["motion_half_span_degrees"]),
        requested_sweep_sha256=requested_sweep_sha256,
        world_fixture_sha256=world_fixture_sha256,
        camera_program_sha256=bound["camera_program_sha256"],
        compiler_sha256=compiler_sha256,
        source_sha256=source_sha256,
    )
    bound["request_index"] = request_index
    bound["requested_sweep_sha256"] = requested_sweep_sha256
    bound["row_request_sha256"] = _sha256_json(request_payload)
    reasons = row_death_reasons(bound, thresholds)
    bound["accepted"] = not reasons
    bound["regime"] = "closure" if not reasons else "death"
    bound["death_reasons"] = reasons
    return bound


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if bool(row.get("accepted"))]
    dead = [row for row in rows if not bool(row.get("accepted"))]
    first_death = dead[0] if dead else None
    last_accepted = accepted[-1] if accepted else None
    return {
        "row_count": len(rows),
        "accepted_count": len(accepted),
        "death_count": len(dead),
        "has_closure": bool(accepted),
        "has_death": bool(dead),
        "last_accepted_half_span_degrees": (
            None if last_accepted is None else float(last_accepted["motion_half_span_degrees"])
        ),
        "first_death_half_span_degrees": (
            None if first_death is None else float(first_death["motion_half_span_degrees"])
        ),
        "max_chart_count": max((int(row.get("chart_count", 0)) for row in rows), default=0),
        "max_event_count": max((int(row.get("event_count", 0)) for row in rows), default=0),
        "compiler_unresolved_count": sum(
            row.get("row_status") == UNRESOLVED_ROW_STATUS for row in rows
        ),
        "terminal_row_status": None if not rows else rows[-1].get("row_status"),
        "max_fallback_sample_fraction": max(
            (
                float(row["fallback_sample_fraction"])
                for row in rows
                if "fallback_sample_fraction" in row
            ),
            default=0.0,
        ),
        "max_invalid_sample_fraction": max(
            (
                float(row["invalid_sample_fraction"])
                for row in rows
                if "invalid_sample_fraction" in row
            ),
            default=0.0,
        ),
        "min_image_psnr_db": min(
            (float(row["image_psnr_db"]) for row in rows if "image_psnr_db" in row),
            default=None,
        ),
        "max_world_vjp_rel_l2": max(
            (
                float(row["world_vjp_rel_l2_max"])
                for row in rows
                if "world_vjp_rel_l2_max" in row
            ),
            default=0.0,
        ),
        "regime_sequence": [str(row.get("regime")) for row in rows],
    }


def _acceptance(
    summary: dict[str, Any],
    *,
    source_eligible: bool = True,
) -> dict[str, Any]:
    reasons = []
    if not source_eligible:
        reasons.append("dirty_source_ineligible")
    if summary.get("has_closure") is not True:
        reasons.append("no_accepted_closure_row")
    if summary.get("has_death") is not True:
        reasons.append("no_observed_death_row")
    if summary.get("compiler_unresolved_count") != 1:
        reasons.append("expected_one_terminal_compiler_unresolved_row")
    if summary.get("terminal_row_status") != UNRESOLVED_ROW_STATUS:
        reasons.append("terminal_row_is_not_compiler_unresolved")
    regimes = summary.get("regime_sequence")
    if isinstance(regimes, list):
        first_death = next(
            (index for index, regime in enumerate(regimes) if regime == "death"),
            len(regimes),
        )
        if any(regime == "closure" for regime in regimes[first_death:]):
            reasons.append("nonmonotone_regime_sequence")
    else:
        reasons.append("missing_regime_sequence")
    return {
        "accepted": not reasons,
        "label": (
            "accepted_bounded_closure_death_gate"
            if not reasons
            else "incomplete_bounded_closure_death_gate"
        ),
        "reasons": reasons,
        "claim_scope": (
            "bounded open-path variable-camera projective atlas; "
            "not a 360/720-degree transition claim"
        ),
    }


def _experiment_contract_payload(
    *,
    world_fixture_sha256: str,
    camera_program_sha256: str,
    compiler_sha256: str,
    half_spans_degrees: list[float],
    thresholds_sha256: str,
) -> dict[str, Any]:
    return {
        "world_fixture_sha256": world_fixture_sha256,
        "camera_program_sha256": camera_program_sha256,
        "compiler_sha256": compiler_sha256,
        "motion_half_spans_degrees": [float(value) for value in half_spans_degrees],
        "thresholds_sha256": thresholds_sha256,
    }


def assemble_report(
    rows: list[dict[str, Any]],
    *,
    half_spans_degrees: list[float],
    world_fixture: dict[str, Any],
    camera_program: dict[str, Any],
    compiler: dict[str, Any],
    thresholds: dict[str, float],
    runtime: dict[str, Any],
    implementation: dict[str, Any],
    source: dict[str, Any],
    source_finish: dict[str, Any],
    dirty_source_allowed: bool,
) -> dict[str, Any]:
    world_hash = _sha256_json(world_fixture)
    compiler_hash = _sha256_json(compiler)
    source_hash = _sha256_json(source)
    requested_sweep_hash = _sha256_json([float(value) for value in half_spans_degrees])
    if not rows:
        raise ValueError("a completed curve must contain at least one evaluated row")
    if len(rows) > len(half_spans_degrees):
        raise ValueError("evaluated rows cannot exceed requested motion spans")
    bound_rows = [
        bind_row_contract(
            row,
            world_fixture_sha256=world_hash,
            camera_program=camera_program_for_span(camera_program, half_span),
            compiler_sha256=compiler_hash,
            source_sha256=source_hash,
            requested_sweep_sha256=requested_sweep_hash,
            request_index=request_index,
            thresholds=thresholds,
        )
        for request_index, (row, half_span) in enumerate(
            zip(rows, half_spans_degrees[: len(rows)], strict=True)
        )
    ]
    summary = summarize(bound_rows)
    source_eligible = (
        source == source_finish
        and source.get("repository_dirty") is False
        and source.get("star_uvt_dirty") is False
    )
    report: dict[str, Any] = {
        "benchmark": BENCHMARK,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "bounded_synthetic_variable_camera_closure_death_curve",
        "status": "completed_bounded_curve",
        "world_fixture": world_fixture,
        "world_fixture_sha256": world_hash,
        "camera_program": camera_program,
        "camera_program_sha256": _sha256_json(camera_program),
        "compiler": compiler,
        "compiler_sha256": compiler_hash,
        "motion_half_spans_degrees": [float(value) for value in half_spans_degrees],
        "requested_sweep_sha256": requested_sweep_hash,
        "execution": {
            "requested_row_count": len(half_spans_degrees),
            "evaluated_row_count": len(bound_rows),
            "evaluated_half_spans_degrees": [
                float(value) for value in half_spans_degrees[: len(bound_rows)]
            ],
            "stopped_after_terminal_compiler_unresolved": (
                bound_rows[-1].get("row_status") == UNRESOLVED_ROW_STATUS
            ),
        },
        "thresholds": thresholds,
        "thresholds_sha256": _sha256_json(thresholds),
        "runtime": runtime,
        "implementation": implementation,
        "source": source,
        "source_finish": source_finish,
        "source_sha256": source_hash,
        "source_policy": {
            "dirty_source_allowed": bool(dirty_source_allowed),
            "paper_evidence_eligible": source_eligible,
        },
        "rows": bound_rows,
        "summary": summary,
        "acceptance": _acceptance(
            summary,
            source_eligible=source_eligible,
        ),
    }
    report["experiment_contract_sha256"] = _sha256_json(
        _experiment_contract_payload(
            world_fixture_sha256=report["world_fixture_sha256"],
            camera_program_sha256=report["camera_program_sha256"],
            compiler_sha256=report["compiler_sha256"],
            half_spans_degrees=report["motion_half_spans_degrees"],
            thresholds_sha256=report["thresholds_sha256"],
        )
    )
    return report


def _finite_float(value: Any, label: str, errors: list[str]) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        errors.append(f"{label} must be a finite number, got {value!r}")
        return 0.0
    return float(value)


def _finite_int(value: Any, label: str, errors: list[str]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{label} must be an integer, got {value!r}")
        return 0
    return int(value)


def _close(
    actual: float,
    expected: float,
    label: str,
    errors: list[str],
    *,
    atol: float = 1.0e-8,
    rtol: float = 0.0,
) -> None:
    if not math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol):
        errors.append(f"{label} mismatch: expected {expected:.12g}, got {actual:.12g}")


def _check_fraction(value: float, label: str, errors: list[str]) -> None:
    if not 0.0 <= value <= 1.0:
        errors.append(f"{label} must be in [0,1], got {value}")


def _check_summary_value(
    summary: dict[str, Any],
    expected: dict[str, Any],
    key: str,
    errors: list[str],
) -> None:
    actual_value = summary.get(key)
    expected_value = expected.get(key)
    if isinstance(expected_value, float):
        if (
            isinstance(actual_value, bool)
            or not isinstance(actual_value, int | float)
            or not math.isfinite(float(actual_value))
            or abs(float(actual_value) - expected_value) > 1.0e-8
        ):
            errors.append(
                f"summary {key} mismatch: expected {expected_value!r}, got {actual_value!r}"
            )
    elif actual_value != expected_value:
        errors.append(
            f"summary {key} mismatch: expected {expected_value!r}, got {actual_value!r}"
        )


def _verify_implementation_manifest(implementation: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(implementation, dict):
        return ["implementation must be an object"]
    source_files = implementation.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        return ["implementation.source_files must be a nonempty list"]
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("path"), str)
        or not isinstance(item.get("sha256"), str)
        or len(item["sha256"]) != 64
        for item in source_files
    ):
        errors.append("implementation.source_files entries must bind path and SHA-256")
        return errors
    if tuple(item["path"] for item in source_files) != REQUIRED_IMPLEMENTATION_SOURCE_PATHS:
        errors.append(
            "implementation.source_files must bind the runner, bridge __init__, "
            "and projective_trace.py in canonical order"
        )
    if implementation.get("source_manifest_sha256") != _sha256_json(source_files):
        errors.append("implementation.source_manifest_sha256 mismatch")
    return errors


def _verify_source_provenance(
    report: dict[str, Any],
    *,
    require_paper_eligible: bool,
) -> tuple[list[str], bool]:
    errors: list[str] = []
    source = report.get("source")
    source_finish = report.get("source_finish")
    if not isinstance(source, dict):
        return ["source must be an object"], False
    if not isinstance(source_finish, dict):
        errors.append("source_finish must be an object")
        source_finish = {}
    if report.get("source_sha256") != _sha256_json(source):
        errors.append("source_sha256 mismatch")
    for key in ("repository_commit", "star_uvt_commit"):
        commit = source.get(key)
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(
                character not in "0123456789abcdef"
                for character in commit.lower()
            )
        ):
            errors.append(f"source.{key} must be a full commit")
    source_eligible = (
        source == source_finish
        and source.get("repository_dirty") is False
        and source.get("star_uvt_dirty") is False
    )
    policy = report.get("source_policy")
    if not isinstance(policy, dict):
        errors.append("source_policy must be an object")
    else:
        if policy.get("dirty_source_allowed") not in {True, False}:
            errors.append("source_policy.dirty_source_allowed must be boolean")
        if policy.get("paper_evidence_eligible") is not source_eligible:
            errors.append("source_policy.paper_evidence_eligible mismatch")
    if require_paper_eligible and not source_eligible:
        errors.append(
            "accepted variable-camera paper evidence requires unchanged clean "
            "superproject and STAR source"
        )
    return errors, source_eligible


def _verify_unresolved_compiler_row(
    row: dict[str, Any],
    *,
    label: str,
    frames: int,
    compiler: dict[str, Any],
    chart_count: int,
    accepted_charts: int,
    unresolved_charts: int,
    errors: list[str],
) -> None:
    if row.get("row_scope") != "death_boundary_only_not_closure_evidence":
        errors.append(f"{label}.row_scope must restrict the row to a death boundary")
    if row.get("compiled_quality_metrics_status") != COMPILED_QUALITY_UNAVAILABLE:
        errors.append(f"{label}.compiled_quality_metrics_status mismatch")
    unavailable = row.get("compiled_quality_metrics_unavailable")
    if unavailable != list(UNAVAILABLE_COMPILED_QUALITY_FIELDS):
        errors.append(f"{label}.compiled_quality_metrics_unavailable mismatch")
    present_unavailable = [key for key in UNAVAILABLE_COMPILED_QUALITY_FIELDS if key in row]
    if present_unavailable:
        errors.append(
            f"{label} fabricates unavailable compiled metrics: {present_unavailable!r}"
        )
    if row.get("compiler_failure_class") != "trace_window_certificate_unsatisfied":
        errors.append(f"{label}.compiler_failure_class mismatch")
    if unresolved_charts <= 0 or accepted_charts + unresolved_charts != chart_count:
        errors.append(f"{label} must contain at least one unresolved compiler chart")

    raw_windows = row.get("unresolved_charts")
    if not isinstance(raw_windows, list) or len(raw_windows) != unresolved_charts:
        errors.append(
            f"{label}.unresolved_charts must contain exactly unresolved_chart_count records"
        )
        raw_windows = []
    if row.get("unresolved_chart_metadata_sha256") != _sha256_json(raw_windows):
        errors.append(f"{label}.unresolved_chart_metadata_sha256 mismatch")

    known_reasons = {
        "uv_residual",
        "depth_residual",
        "denominator",
        "denominator_boundary",
        "invalid_samples",
        "max_windows",
    }
    observed_reasons: set[str] = set()
    previous_stop = -1
    uv_values: list[float] = []
    depth_values: list[float] = []
    denominator_values: list[float] = []
    valid_values: list[float] = []
    denominator_root_counts: list[int] = []
    for window_index, raw_window in enumerate(raw_windows):
        window_label = f"{label}.unresolved_charts[{window_index}]"
        if not isinstance(raw_window, dict):
            errors.append(f"{window_label} must be an object")
            continue
        start = _finite_int(raw_window.get("start"), f"{window_label}.start", errors)
        stop = _finite_int(raw_window.get("stop"), f"{window_label}.stop", errors)
        if start < 0 or stop <= start or stop > frames:
            errors.append(f"{window_label} has an invalid sample interval")
        if start < previous_stop:
            errors.append(f"{label}.unresolved_charts must be sorted and nonoverlapping")
        previous_stop = stop
        if _finite_int(
            raw_window.get("sample_count"),
            f"{window_label}.sample_count",
            errors,
        ) != stop - start:
            errors.append(f"{window_label}.sample_count mismatch")
        reason = raw_window.get("reason")
        reasons = raw_window.get("reasons")
        if not isinstance(reason, str) or not reason:
            errors.append(f"{window_label}.reason must be nonempty")
            parsed_reasons: list[str] = []
        else:
            parsed_reasons = reason.split(",")
        if reasons != parsed_reasons:
            errors.append(f"{window_label}.reasons mismatch")
        if any(value not in known_reasons for value in parsed_reasons):
            errors.append(f"{window_label} contains an unknown compiler reason")
        observed_reasons.update(parsed_reasons)

        uv = _finite_float(
            raw_window.get("sampled_max_fit_residual_uv_px"),
            f"{window_label}.sampled_max_fit_residual_uv_px",
            errors,
        )
        depth = _finite_float(
            raw_window.get("sampled_max_fit_residual_depth"),
            f"{window_label}.sampled_max_fit_residual_depth",
            errors,
        )
        denominator = _finite_float(
            raw_window.get("min_denominator_abs"),
            f"{window_label}.min_denominator_abs",
            errors,
        )
        valid_fraction = _finite_float(
            raw_window.get("min_valid_fraction"),
            f"{window_label}.min_valid_fraction",
            errors,
        )
        root_count = _finite_int(
            raw_window.get("denominator_root_count"),
            f"{window_label}.denominator_root_count",
            errors,
        )
        uv_values.append(uv)
        depth_values.append(depth)
        denominator_values.append(denominator)
        valid_values.append(valid_fraction)
        denominator_root_counts.append(root_count)
        if min(uv, depth, denominator) < 0.0:
            errors.append(f"{window_label} residual and denominator metrics must be nonnegative")
        _check_fraction(valid_fraction, f"{window_label}.min_valid_fraction", errors)
        if root_count < 0:
            errors.append(f"{window_label}.denominator_root_count must be nonnegative")

        witnessed = {
            "uv_residual": uv > float(compiler.get("sampled_max_residual_uv_px", math.inf)),
            "depth_residual": depth
            > float(compiler.get("sampled_max_depth_residual", math.inf)),
            "denominator": denominator
            < float(compiler.get("min_denominator_abs", -math.inf)),
            "denominator_boundary": root_count > 0,
            "invalid_samples": valid_fraction
            < float(compiler.get("min_valid_fraction", 1.0)),
            "max_windows": chart_count >= int(compiler.get("max_windows", chart_count + 1)),
        }
        for declared_reason in parsed_reasons:
            if not witnessed.get(declared_reason, False):
                errors.append(
                    f"{window_label}.{declared_reason} is not witnessed by stored compiler metrics"
                )

    reported_reasons = row.get("unresolved_chart_reasons")
    expected_reasons = sorted(observed_reasons)
    if reported_reasons != expected_reasons:
        errors.append(
            f"{label}.unresolved_chart_reasons mismatch: expected {expected_reasons!r}, "
            f"got {reported_reasons!r}"
        )
    aggregate_checks = (
        ("unresolved_max_fit_residual_uv_px", max(uv_values, default=0.0)),
        ("unresolved_max_fit_residual_depth", max(depth_values, default=0.0)),
        ("unresolved_min_denominator_abs", min(denominator_values, default=0.0)),
        ("unresolved_min_valid_fraction", min(valid_values, default=0.0)),
    )
    for key, expected in aggregate_checks:
        actual = _finite_float(row.get(key), f"{label}.{key}", errors)
        _close(actual, expected, f"{label}.{key}", errors, atol=1.0e-7, rtol=1.0e-6)
    root_count = _finite_int(
        row.get("unresolved_denominator_root_count"),
        f"{label}.unresolved_denominator_root_count",
        errors,
    )
    if root_count != sum(denominator_root_counts):
        errors.append(f"{label}.unresolved_denominator_root_count mismatch")


def verify_variable_camera_closure_death_curve(
    report: dict[str, Any],
    *,
    require_paper_eligible: bool = True,
) -> list[str]:
    """Validate both evidence identity and the bounded closure/death contract."""

    errors: list[str] = []
    if report.get("benchmark") != BENCHMARK:
        errors.append(f"benchmark must be {BENCHMARK}")
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if report.get("scope") != "bounded_synthetic_variable_camera_closure_death_curve":
        errors.append("scope must identify the bounded synthetic variable-camera curve")
    if report.get("status") != "completed_bounded_curve":
        errors.append("status must be completed_bounded_curve")

    world_fixture = report.get("world_fixture")
    if not isinstance(world_fixture, dict):
        errors.append("world_fixture must be an object")
        world_fixture = {}
    expected_world_hash = _sha256_json(world_fixture)
    if report.get("world_fixture_sha256") != expected_world_hash:
        errors.append("world_fixture_sha256 mismatch")
    primitives = world_fixture.get("primitives")
    if not isinstance(primitives, list) or len(primitives) < 2:
        errors.append("world_fixture.primitives must contain at least two primitives")
        primitive_count = 0
    else:
        primitive_count = len(primitives)

    camera_program = report.get("camera_program")
    if not isinstance(camera_program, dict):
        errors.append("camera_program must be an object")
        camera_program = {}
    if report.get("camera_program_sha256") != _sha256_json(camera_program):
        errors.append("camera_program_sha256 mismatch")
    if camera_program.get("program_family") != "bounded_yaw_orbit_tan_half_angle":
        errors.append("camera_program.program_family mismatch")
    physical_interval = camera_program.get("physical_interval")
    if physical_interval != [-1.0, 1.0]:
        errors.append("camera_program.physical_interval must stay fixed at [-1,1]")
    excluded = camera_program.get("excluded_claims")
    if not isinstance(excluded, list) or not {
        "360_degree_transition",
        "720_degree_transition",
        "closed_loop_holonomy",
    }.issubset(set(excluded)):
        errors.append("camera_program must explicitly exclude 360/720 and holonomy claims")
    frames = _finite_int(camera_program.get("sample_count"), "camera_program.sample_count", errors)
    image_width = _finite_int(camera_program.get("image_width"), "camera_program.image_width", errors)
    image_height = _finite_int(camera_program.get("image_height"), "camera_program.image_height", errors)
    if frames < 16:
        errors.append("camera_program.sample_count must be at least 16")
    if image_width <= 0 or image_height <= 0:
        errors.append("camera image dimensions must be positive")

    compiler = report.get("compiler")
    if not isinstance(compiler, dict):
        errors.append("compiler must be an object")
        compiler = {}
    if report.get("compiler_sha256") != _sha256_json(compiler):
        errors.append("compiler_sha256 mismatch")
    if compiler.get("compiler_family") != "projective_star_uvt_local_interval_atlas":
        errors.append("compiler.compiler_family mismatch")
    if compiler.get("event_partition") != "continuous_support_plus_visibility_roots":
        errors.append("compiler must use continuous support plus visibility event partitioning")
    if compiler.get("reference") != "exact_rational_trace_full_image_all_live_depth_sort":
        errors.append(
            "compiler reference must be exact_rational_trace_full_image_all_live_depth_sort"
        )
    expected_reference_contract = {
        "reference_support_policy": "full_image",
        "reference_order_policy": "all_live_depth_per_sample",
        "reference_fallback_reason": "oracle_all_live_depth_sort",
        "reference_sample_semantics": "empirical_at_requested_samples",
        "fit_residual_semantics": "empirical_max_over_requested_samples",
    }
    for key, expected_value in expected_reference_contract.items():
        if compiler.get(key) != expected_value:
            errors.append(f"compiler.{key} must be {expected_value}")
    if compiler.get("local_evaluator") != "degree_1_polynomial":
        errors.append("compiler.local_evaluator must be degree_1_polynomial")
    compiler_tile_size = _finite_int(
        compiler.get("tile_size"),
        "compiler.tile_size",
        errors,
    )
    if compiler_tile_size <= 0 or (
        image_width > 0 and image_width % compiler_tile_size != 0
    ):
        errors.append("compiler.tile_size must positively divide the image width")
    for key in (
        "sigma_px",
        "support_padding_px",
        "sampled_max_residual_uv_px",
        "sampled_max_depth_residual",
        "min_denominator_abs",
    ):
        if _finite_float(compiler.get(key), f"compiler.{key}", errors) <= 0.0:
            errors.append(f"compiler.{key} must be positive")
    if _finite_int(compiler.get("max_windows"), "compiler.max_windows", errors) < 4:
        errors.append("compiler.max_windows must be at least four")

    thresholds = report.get("thresholds")
    if not isinstance(thresholds, dict):
        errors.append("thresholds must be an object")
        thresholds = {}
    if report.get("thresholds_sha256") != _sha256_json(thresholds):
        errors.append("thresholds_sha256 mismatch")
    required_thresholds = set(default_thresholds())
    if not required_thresholds.issubset(thresholds):
        errors.append(f"thresholds missing keys: {sorted(required_thresholds - set(thresholds))}")
    for key in required_thresholds:
        value = _finite_float(thresholds.get(key), f"thresholds.{key}", errors)
        if key.startswith("max_") and "fraction" in key:
            _check_fraction(value, f"thresholds.{key}", errors)
        elif value <= 0.0:
            errors.append(f"thresholds.{key} must be positive")

    raw_spans = report.get("motion_half_spans_degrees")
    if not isinstance(raw_spans, list) or len(raw_spans) < 4:
        errors.append("motion_half_spans_degrees must contain at least four points")
        return errors
    spans = [
        _finite_float(value, f"motion_half_spans_degrees[{index}]", errors)
        for index, value in enumerate(raw_spans)
    ]
    if spans != sorted(spans) or len(set(spans)) != len(spans):
        errors.append("motion_half_spans_degrees must be strictly increasing")
    if any(span <= 0.0 or 2.0 * span >= 360.0 for span in spans):
        errors.append("every motion span must be positive and strictly below a 360-degree path")

    expected_contract_hash = _sha256_json(
        _experiment_contract_payload(
            world_fixture_sha256=expected_world_hash,
            camera_program_sha256=_sha256_json(camera_program),
            compiler_sha256=_sha256_json(compiler),
            half_spans_degrees=spans,
            thresholds_sha256=_sha256_json(thresholds),
        )
    )
    if report.get("experiment_contract_sha256") != expected_contract_hash:
        errors.append("experiment_contract_sha256 mismatch")

    requested_sweep_hash = _sha256_json(spans)
    if report.get("requested_sweep_sha256") != requested_sweep_hash:
        errors.append("requested_sweep_sha256 mismatch")

    raw_rows = report.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows or len(raw_rows) > len(spans):
        errors.append("rows must contain a nonempty prefix of the requested motion sweep")
        return errors
    if any(not isinstance(row, dict) for row in raw_rows):
        errors.append("all rows must be objects")
        return errors
    rows: list[dict[str, Any]] = raw_rows
    evaluated_spans = spans[: len(rows)]
    execution = report.get("execution")
    if not isinstance(execution, dict):
        errors.append("execution must be an object")
    else:
        expected_execution = {
            "requested_row_count": len(spans),
            "evaluated_row_count": len(rows),
            "evaluated_half_spans_degrees": evaluated_spans,
            "stopped_after_terminal_compiler_unresolved": (
                rows[-1].get("row_status") == UNRESOLVED_ROW_STATUS
            ),
        }
        if execution != expected_execution:
            errors.append(
                f"execution mismatch: expected {expected_execution!r}, got {execution!r}"
            )
    if len(rows) < len(spans) and rows[-1].get("row_status") != UNRESOLVED_ROW_STATUS:
        errors.append("an abbreviated sweep must stop at a terminal unresolved compiler row")

    source_value = report.get("source")
    expected_source_hash = _sha256_json(source_value if isinstance(source_value, dict) else {})
    chart_counts: list[int] = []
    regimes: list[str] = []

    for index, (row, span) in enumerate(zip(rows, evaluated_spans, strict=True)):
        label = f"rows[{index}]"
        row_span = _finite_float(
            row.get("motion_half_span_degrees"),
            f"{label}.motion_half_span_degrees",
            errors,
        )
        _close(row_span, span, f"{label}.motion_half_span_degrees", errors)
        total_span = _finite_float(
            row.get("motion_total_span_degrees"),
            f"{label}.motion_total_span_degrees",
            errors,
        )
        _close(total_span, 2.0 * span, f"{label}.motion_total_span_degrees", errors)
        expected_program_hash = _sha256_json(camera_program_for_span(camera_program, span))
        if row.get("camera_program_sha256") != expected_program_hash:
            errors.append(f"{label}.camera_program_sha256 mismatch")
        if row.get("world_fixture_sha256") != expected_world_hash:
            errors.append(f"{label}.world_fixture_sha256 mismatch")
        if row.get("physical_interval") != [-1.0, 1.0]:
            errors.append(f"{label}.physical_interval must remain [-1,1]")
        if _finite_int(row.get("sample_count"), f"{label}.sample_count", errors) != frames:
            errors.append(f"{label}.sample_count must match the camera program")
        q_min = _finite_float(row.get("q_min"), f"{label}.q_min", errors)
        q_max = _finite_float(row.get("q_max"), f"{label}.q_max", errors)
        half_angle = 0.5 * math.radians(span)
        expected_q = math.tan(half_angle)
        float32_epsilon = 2.0**-23
        # The endpoint is evaluated as tan(float32(theta)/2). Near the chart
        # pole, tan's relative condition number grows as
        # |x| (1 + tan(x)^2) / |tan(x)|. Bind the tolerance to that condition
        # number instead of using an arbitrary absolute epsilon.
        q_relative_guard = float32_epsilon * (
            4.0
            + abs(half_angle)
            * (1.0 + expected_q * expected_q)
            / max(abs(expected_q), 1.0)
        )
        q_absolute_guard = 4.0 * float32_epsilon
        _close(
            q_min,
            -expected_q,
            f"{label}.q_min",
            errors,
            atol=q_absolute_guard,
            rtol=q_relative_guard,
        )
        _close(
            q_max,
            expected_q,
            f"{label}.q_max",
            errors,
            atol=q_absolute_guard,
            rtol=q_relative_guard,
        )

        expected_row_request = _row_request_payload(
            request_index=index,
            half_span_degrees=span,
            requested_sweep_sha256=requested_sweep_hash,
            world_fixture_sha256=expected_world_hash,
            camera_program_sha256=expected_program_hash,
            compiler_sha256=_sha256_json(compiler),
            source_sha256=expected_source_hash,
        )
        if row.get("request_index") != index:
            errors.append(f"{label}.request_index mismatch")
        if row.get("requested_sweep_sha256") != requested_sweep_hash:
            errors.append(f"{label}.requested_sweep_sha256 mismatch")
        if row.get("compiler_sha256") != _sha256_json(compiler):
            errors.append(f"{label}.compiler_sha256 mismatch")
        if row.get("source_sha256") != expected_source_hash:
            errors.append(f"{label}.source_sha256 mismatch")
        if row.get("row_request_sha256") != _sha256_json(expected_row_request):
            errors.append(f"{label}.row_request_sha256 mismatch")

        chart_count = _finite_int(row.get("chart_count"), f"{label}.chart_count", errors)
        accepted_charts = _finite_int(
            row.get("accepted_chart_count"),
            f"{label}.accepted_chart_count",
            errors,
        )
        unresolved_charts = _finite_int(
            row.get("unresolved_chart_count"),
            f"{label}.unresolved_chart_count",
            errors,
        )
        accepted_fraction = _finite_float(
            row.get("accepted_chart_fraction"),
            f"{label}.accepted_chart_fraction",
            errors,
        )
        projected_samples = _finite_int(
            row.get("projected_sample_count"),
            f"{label}.projected_sample_count",
            errors,
        )
        if (
            chart_count <= 0
            or accepted_charts < 0
            or unresolved_charts < 0
            or accepted_charts + unresolved_charts != chart_count
        ):
            errors.append(f"{label} has inconsistent chart counts")
        _check_fraction(accepted_fraction, f"{label}.accepted_chart_fraction", errors)
        if chart_count > 0:
            _close(
                accepted_fraction,
                accepted_charts / float(chart_count),
                f"{label}.accepted_chart_fraction",
                errors,
            )
        if projected_samples != primitive_count * frames:
            errors.append(f"{label}.projected_sample_count must equal primitive_count*sample_count")
        if row.get("fit_residual_semantics") != "empirical_max_over_requested_samples":
            errors.append(f"{label}.fit_residual_semantics mismatch")

        row_status = row.get("row_status")
        if row_status == UNRESOLVED_ROW_STATUS:
            _verify_unresolved_compiler_row(
                row,
                label=label,
                frames=frames,
                compiler=compiler,
                chart_count=chart_count,
                accepted_charts=accepted_charts,
                unresolved_charts=unresolved_charts,
                errors=errors,
            )
            expected_reasons = row_death_reasons(row, thresholds)
            if expected_reasons != ["unresolved_chart"]:
                errors.append(f"{label} is not a witnessed unresolved-chart death")
            if row.get("death_reasons") != expected_reasons:
                errors.append(f"{label}.death_reasons mismatch")
            if row.get("accepted") is not False:
                errors.append(f"{label}.accepted must be false for an unresolved compiler row")
            if row.get("regime") != "death":
                errors.append(f"{label}.regime must be death for an unresolved compiler row")
            if index != len(rows) - 1:
                errors.append(f"{label} unresolved compiler row must be terminal")
            chart_counts.append(chart_count)
            regimes.append("death")
            continue

        if row_status != COMPILED_ROW_STATUS:
            errors.append(f"{label}.row_status is invalid")
        if row.get("row_scope") != "compiled_quality_closure_or_threshold_death":
            errors.append(f"{label}.row_scope mismatch")
        if row.get("compiled_quality_metrics_status") != COMPILED_QUALITY_AVAILABLE:
            errors.append(f"{label}.compiled_quality_metrics_status mismatch")
        if row.get("compiled_quality_metrics_unavailable") != []:
            errors.append(f"{label}.compiled_quality_metrics_unavailable must be empty")
        if unresolved_charts != 0:
            errors.append(f"{label} compiled-quality row cannot contain unresolved charts")
        if row.get("unresolved_chart_reasons") != [] or row.get("unresolved_charts") != []:
            errors.append(f"{label} compiled-quality row has stale unresolved metadata")
        if row.get("unresolved_chart_metadata_sha256") != _sha256_json([]):
            errors.append(f"{label}.unresolved_chart_metadata_sha256 mismatch")

        support_events = _finite_int(
            row.get("support_event_count"),
            f"{label}.support_event_count",
            errors,
        )
        visibility_events = _finite_int(
            row.get("visibility_event_count"),
            f"{label}.visibility_event_count",
            errors,
        )
        event_count = _finite_int(row.get("event_count"), f"{label}.event_count", errors)
        event_intervals = _finite_int(
            row.get("event_interval_count"),
            f"{label}.event_interval_count",
            errors,
        )
        trace_count = _finite_int(row.get("trace_count"), f"{label}.trace_count", errors)
        cell_count = _finite_int(row.get("cell_count"), f"{label}.cell_count", errors)
        interval_entries = _finite_int(
            row.get("interval_entry_count"),
            f"{label}.interval_entry_count",
            errors,
        )
        dense_samples = _finite_int(
            row.get("dense_trace_samples"),
            f"{label}.dense_trace_samples",
            errors,
        )
        fallback_cells = _finite_int(
            row.get("fallback_cell_count"),
            f"{label}.fallback_cell_count",
            errors,
        )
        fallback_trace_samples = _finite_int(
            row.get("fallback_trace_samples"),
            f"{label}.fallback_trace_samples",
            errors,
        )
        invalid_samples = _finite_int(
            row.get("invalid_sample_count"),
            f"{label}.invalid_sample_count",
            errors,
        )
        if min(support_events, visibility_events, event_count) < 0:
            errors.append(f"{label} event counts must be nonnegative")
        if event_count != support_events + visibility_events:
            errors.append(f"{label}.event_count must sum support and visibility events")
        if event_intervals <= 0:
            errors.append(f"{label}.event_interval_count must be positive")
        if min(trace_count, cell_count, interval_entries, dense_samples, projected_samples) <= 0:
            errors.append(f"{label} topology/sample counts must be positive")
        if interval_entries > dense_samples:
            errors.append(f"{label}.interval_entry_count cannot exceed dense_trace_samples")
        if fallback_cells < 0 or fallback_cells > cell_count:
            errors.append(f"{label}.fallback_cell_count is invalid")
        if fallback_trace_samples < 0 or fallback_trace_samples > dense_samples:
            errors.append(f"{label}.fallback_trace_samples is invalid")
        if invalid_samples < 0 or invalid_samples > projected_samples:
            errors.append(f"{label}.invalid_sample_count is invalid")
        fit_residual = _finite_float(
            row.get("sampled_max_fit_residual_uv_px"),
            f"{label}.sampled_max_fit_residual_uv_px",
            errors,
        )
        denominator_margin = _finite_float(
            row.get("min_denominator_abs"),
            f"{label}.min_denominator_abs",
            errors,
        )
        if fit_residual < 0.0 or fit_residual > float(
            compiler.get("sampled_max_residual_uv_px", 0.0)
        ):
            errors.append(
                f"{label}.sampled_max_fit_residual_uv_px violates the sampled fit contract"
            )
        if denominator_margin < float(compiler.get("min_denominator_abs", math.inf)):
            errors.append(f"{label}.min_denominator_abs violates the compiler certificate")
        if row.get("reference_support_policy") != "full_image":
            errors.append(f"{label}.reference_support_policy mismatch")
        if row.get("reference_order_policy") != "all_live_depth_per_sample":
            errors.append(f"{label}.reference_order_policy mismatch")
        if row.get("reference_fallback_reason") != "oracle_all_live_depth_sort":
            errors.append(f"{label}.reference_fallback_reason mismatch")
        if row.get("reference_sample_semantics") != "empirical_at_requested_samples":
            errors.append(f"{label}.reference_sample_semantics mismatch")
        reference_cell_count = _finite_int(
            row.get("reference_cell_count"),
            f"{label}.reference_cell_count",
            errors,
        )
        live_sorted_reference_cells = _finite_int(
            row.get("reference_live_sorted_cell_count"),
            f"{label}.reference_live_sorted_cell_count",
            errors,
        )
        if reference_cell_count <= 0 or live_sorted_reference_cells != reference_cell_count:
            errors.append(f"{label} must live-sort every nonempty reference cell")
        if not isinstance(row.get("fallback_reasons"), list):
            errors.append(f"{label}.fallback_reasons must be a list")
        if row.get("vjp_topology_semantics") != (
            "fixed_compiled_topology_away_from_event_boundaries"
        ):
            errors.append(f"{label}.vjp_topology_semantics mismatch")

        trace_ratio = _finite_float(
            row.get("trace_to_replay_ratio"),
            f"{label}.trace_to_replay_ratio",
            errors,
        )
        interval_ratio = _finite_float(
            row.get("interval_to_dense_ratio"),
            f"{label}.interval_to_dense_ratio",
            errors,
        )
        fallback_cell_fraction = _finite_float(
            row.get("fallback_cell_fraction"),
            f"{label}.fallback_cell_fraction",
            errors,
        )
        fallback_sample_fraction = _finite_float(
            row.get("fallback_sample_fraction"),
            f"{label}.fallback_sample_fraction",
            errors,
        )
        invalid_fraction = _finite_float(
            row.get("invalid_sample_fraction"),
            f"{label}.invalid_sample_fraction",
            errors,
        )
        for fraction, fraction_label in (
            (trace_ratio, "trace_to_replay_ratio"),
            (interval_ratio, "interval_to_dense_ratio"),
            (fallback_cell_fraction, "fallback_cell_fraction"),
            (fallback_sample_fraction, "fallback_sample_fraction"),
            (invalid_fraction, "invalid_sample_fraction"),
        ):
            _check_fraction(fraction, f"{label}.{fraction_label}", errors)
        _close(
            trace_ratio,
            trace_count / float(max(1, primitive_count * frames)),
            f"{label}.trace_to_replay_ratio",
            errors,
        )
        _close(
            interval_ratio,
            interval_entries / float(max(1, dense_samples)),
            f"{label}.interval_to_dense_ratio",
            errors,
        )
        _close(
            fallback_cell_fraction,
            fallback_cells / float(max(1, cell_count)),
            f"{label}.fallback_cell_fraction",
            errors,
        )
        _close(
            fallback_sample_fraction,
            fallback_trace_samples / float(max(1, dense_samples)),
            f"{label}.fallback_sample_fraction",
            errors,
        )
        _close(
            invalid_fraction,
            invalid_samples / float(max(1, projected_samples)),
            f"{label}.invalid_sample_fraction",
            errors,
        )

        image_mse = _finite_float(row.get("image_mse"), f"{label}.image_mse", errors)
        image_psnr = _finite_float(
            row.get("image_psnr_db"),
            f"{label}.image_psnr_db",
            errors,
        )
        image_p999 = _finite_float(
            row.get("image_p999_abs_error"),
            f"{label}.image_p999_abs_error",
            errors,
        )
        image_max = _finite_float(
            row.get("image_max_abs_error"),
            f"{label}.image_max_abs_error",
            errors,
        )
        if min(image_mse, image_p999, image_max) < 0.0:
            errors.append(f"{label} image errors must be nonnegative")
        expected_psnr = min(200.0, 10.0 * math.log10(1.0 / max(image_mse, 1.0e-20)))
        _close(image_psnr, expected_psnr, f"{label}.image_psnr_db", errors, atol=1.0e-5)
        if image_p999 > image_max:
            errors.append(f"{label}.image_p999_abs_error cannot exceed image_max_abs_error")

        if row.get("world_vjp_parameter_names") != list(WORLD_VJP_PARAMETER_NAMES):
            errors.append(f"{label}.world_vjp_parameter_names mismatch")
        vjp_maps: dict[str, dict[str, Any]] = {}
        for key in (
            "world_vjp_rel_l2_by_parameter",
            "world_vjp_reference_norm_by_parameter",
            "world_vjp_compiled_norm_by_parameter",
        ):
            raw_map = row.get(key)
            if not isinstance(raw_map, dict):
                errors.append(f"{label}.{key} must be an object")
                raw_map = {}
            if set(raw_map) != set(WORLD_VJP_PARAMETER_NAMES):
                errors.append(
                    f"{label}.{key} keys must be exactly {list(WORLD_VJP_PARAMETER_NAMES)!r}"
                )
            vjp_maps[key] = raw_map
        vjp_by_parameter = vjp_maps["world_vjp_rel_l2_by_parameter"]
        vjp_values = [
            _finite_float(
                vjp_by_parameter.get(name),
                f"{label}.world_vjp_rel_l2_by_parameter.{name}",
                errors,
            )
            for name in WORLD_VJP_PARAMETER_NAMES
        ]
        if any(value < 0.0 for value in vjp_values):
            errors.append(f"{label} world VJP relative errors must be nonnegative")
        reference_norms = [
            _finite_float(
                vjp_maps["world_vjp_reference_norm_by_parameter"].get(name),
                f"{label}.world_vjp_reference_norm_by_parameter.{name}",
                errors,
            )
            for name in WORLD_VJP_PARAMETER_NAMES
        ]
        compiled_norms = [
            _finite_float(
                vjp_maps["world_vjp_compiled_norm_by_parameter"].get(name),
                f"{label}.world_vjp_compiled_norm_by_parameter.{name}",
                errors,
            )
            for name in WORLD_VJP_PARAMETER_NAMES
        ]
        if any(value < 0.0 for value in (*reference_norms, *compiled_norms)):
            errors.append(f"{label} world VJP norms must be nonnegative")
        vjp_max = _finite_float(
            row.get("world_vjp_rel_l2_max"),
            f"{label}.world_vjp_rel_l2_max",
            errors,
        )
        _close(vjp_max, max(vjp_values, default=0.0), f"{label}.world_vjp_rel_l2_max", errors)
        nonzero_vjp = _finite_int(
            row.get("world_vjp_nonzero_parameter_count"),
            f"{label}.world_vjp_nonzero_parameter_count",
            errors,
        )
        expected_nonzero_vjp = sum(
            max(reference_norm, compiled_norm) > 1.0e-8
            for reference_norm, compiled_norm in zip(
                reference_norms,
                compiled_norms,
                strict=True,
            )
        )
        if nonzero_vjp != expected_nonzero_vjp or nonzero_vjp <= 0:
            errors.append(
                f"{label}.world_vjp_nonzero_parameter_count mismatch: "
                f"expected {expected_nonzero_vjp}, got {nonzero_vjp}"
            )

        expected_reasons = row_death_reasons(row, thresholds)
        if row.get("death_reasons") != expected_reasons:
            errors.append(
                f"{label}.death_reasons mismatch: expected {expected_reasons!r}, "
                f"got {row.get('death_reasons')!r}"
            )
        expected_accepted = not expected_reasons
        if row.get("accepted") is not expected_accepted:
            errors.append(f"{label}.accepted mismatch")
        expected_regime = "closure" if expected_accepted else "death"
        if row.get("regime") != expected_regime:
            errors.append(f"{label}.regime mismatch")
        if any(reason not in REQUIRED_DEATH_REASONS for reason in expected_reasons):
            errors.append(f"{label} has unknown death reason")
        chart_counts.append(chart_count)
        regimes.append(expected_regime)

    if chart_counts != sorted(chart_counts):
        errors.append(f"chart_count must be nondecreasing with motion, got {chart_counts}")
    first_death_index = next(
        (index for index, regime in enumerate(regimes) if regime == "death"),
        len(regimes),
    )
    if any(regime == "closure" for regime in regimes[first_death_index:]):
        errors.append("regime sequence must be a closure prefix followed by a death suffix")
    unresolved_indices = [
        index
        for index, row in enumerate(rows)
        if row.get("row_status") == UNRESOLVED_ROW_STATUS
    ]
    if unresolved_indices != [len(rows) - 1]:
        errors.append("the curve must end in exactly one terminal unresolved compiler row")
    if first_death_index == 0:
        errors.append("the curve must contain a nonempty accepted closure prefix")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        return errors
    expected_summary = summarize(rows)
    for key in expected_summary:
        _check_summary_value(summary, expected_summary, key, errors)

    acceptance = report.get("acceptance")
    if not isinstance(acceptance, dict):
        errors.append("acceptance must be an object")
        return errors
    source_errors, source_eligible = _verify_source_provenance(
        report,
        require_paper_eligible=require_paper_eligible,
    )
    errors.extend(source_errors)
    expected_acceptance = _acceptance(
        expected_summary,
        source_eligible=source_eligible,
    )
    if acceptance != expected_acceptance:
        errors.append(
            f"acceptance mismatch: expected {expected_acceptance!r}, got {acceptance!r}"
        )
    structural_acceptance = _acceptance(expected_summary, source_eligible=True)
    if structural_acceptance.get("accepted") is not True:
        errors.append("bounded curve must contain both a closure regime and a death boundary")
    if require_paper_eligible and acceptance.get("accepted") is not True:
        errors.append("bounded curve is not eligible paper evidence")

    errors.extend(_verify_implementation_manifest(report.get("implementation")))
    if not isinstance(report.get("runtime"), dict):
        errors.append("runtime must be an object")
    return errors


def assert_variable_camera_closure_death_curve(
    report: dict[str, Any],
    *,
    require_paper_eligible: bool = True,
) -> None:
    errors = verify_variable_camera_closure_death_curve(
        report,
        require_paper_eligible=require_paper_eligible,
    )
    if errors:
        raise AssertionError(
            "variable-camera closure/death curve failed:\n- " + "\n- ".join(errors)
        )


def verify_current_implementation(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    implementation = report.get("implementation")
    if not isinstance(implementation, dict):
        return ["implementation must be an object"]
    source_files = implementation.get("source_files")
    if not isinstance(source_files, list):
        return ["implementation.source_files must be a list"]
    for item in source_files:
        if not isinstance(item, dict):
            errors.append("implementation source entry must be an object")
            continue
        raw_path = item.get("path")
        if not isinstance(raw_path, str):
            errors.append("implementation source path must be a string")
            continue
        path = ROOT / raw_path
        if not path.is_file():
            errors.append(f"implementation source missing: {raw_path}")
            continue
        if _sha256_file(path) != item.get("sha256"):
            errors.append(f"implementation source hash mismatch: {raw_path}")
    current_source = source_provenance()
    if current_source != report.get("source"):
        errors.append("current source provenance does not match the report")
    if report.get("status") != "runtime_failure":
        try:
            require_clean_source(current_source)
        except RuntimeError as error:
            errors.append(str(error))
    return errors


def _implementation_manifest() -> dict[str, Any]:
    source_files = [
        {
            "path": relative_path,
            "sha256": _sha256_file(ROOT / relative_path),
        }
        for relative_path in REQUIRED_IMPLEMENTATION_SOURCE_PATHS
    ]
    return {
        "source_files": source_files,
        "source_manifest_sha256": _sha256_json(source_files),
    }


def _runtime_modules() -> tuple[Any, Any]:
    if str(STAR_UVT_ROOT) not in sys.path:
        sys.path.insert(0, str(STAR_UVT_ROOT))
    torch = importlib.import_module("torch")
    bridge = importlib.import_module("torch_gsplat_bridge_star_uvt")
    return torch, bridge


def _world_tensors(torch: Any, fixture: dict[str, Any], camera_program: dict[str, Any]) -> dict[str, Any]:
    primitives = fixture["primitives"]
    point_x = torch.tensor(
        [item["point_x"] for item in primitives],
        dtype=torch.float32,
        requires_grad=True,
    )
    base_depth = torch.tensor(
        [item["base_depth"] for item in primitives],
        dtype=torch.float32,
        requires_grad=True,
    )
    vertical = torch.tensor(
        [item["vertical"] for item in primitives],
        dtype=torch.float32,
        requires_grad=True,
    )
    opacity = torch.tensor(
        [item["opacity"] for item in primitives],
        dtype=torch.float32,
        requires_grad=True,
    )
    color = torch.tensor(
        [item["color"] for item in primitives],
        dtype=torch.float32,
        requires_grad=True,
    )
    focal = float(camera_program["focal_px"])
    center_u, center_v = camera_program["principal_point_px"]
    depth_offset = float(camera_program["orbit_depth_offset"])
    raw_u = torch.stack((point_x, torch.full_like(point_x, 2.0), -point_x), dim=1)
    raw_v = torch.stack((vertical, torch.zeros_like(vertical), vertical), dim=1)
    depth = torch.stack(
        (base_depth + depth_offset, 2.0 * point_x, base_depth - depth_offset),
        dim=1,
    )
    pixel_u = float(center_u) * depth + focal * raw_u
    pixel_v = float(center_v) * depth + focal * raw_v
    return {
        "coeffs": torch.cat((pixel_u, pixel_v, depth), dim=1).contiguous(),
        "opacity": opacity,
        "color": color,
        "parameters": {
            "point_x": point_x,
            "base_depth": base_depth,
            "vertical": vertical,
            "opacity": opacity,
            "color": color,
        },
    }


def _relative_vjp_errors(torch: Any, reference: tuple[Any, ...], compiled: tuple[Any, ...]) -> list[float]:
    errors = []
    for reference_grad, compiled_grad in zip(reference, compiled, strict=True):
        numerator = torch.linalg.vector_norm(compiled_grad - reference_grad)
        denominator = torch.maximum(
            torch.linalg.vector_norm(reference_grad),
            torch.linalg.vector_norm(compiled_grad),
        ).clamp_min(1.0e-8)
        errors.append(float((numerator / denominator).detach().cpu().item()))
    return errors


def _run_row(
    *,
    half_span_degrees: float,
    world_fixture: dict[str, Any],
    camera_program: dict[str, Any],
    compiler: dict[str, Any],
) -> dict[str, Any]:
    torch, bridge = _runtime_modules()
    world = _world_tensors(torch, world_fixture, camera_program)
    frames = int(camera_program["sample_count"])
    physical_times = torch.linspace(-1.0, 1.0, frames, dtype=torch.float32)
    theta = math.radians(float(half_span_degrees)) * physical_times
    times = torch.tan(0.5 * theta).contiguous()
    coeffs = world["coeffs"]
    opacity = world["opacity"]
    color = world["color"]
    windows = bridge.split_projective_trace_windows(
        coeffs,
        times,
        degree=1,
        max_residual_uv=float(compiler["sampled_max_residual_uv_px"]),
        max_depth_residual=float(compiler["sampled_max_depth_residual"]),
        min_denominator_abs=float(compiler["min_denominator_abs"]),
        min_valid_fraction=float(compiler["min_valid_fraction"]),
        max_windows=int(compiler["max_windows"]),
    )
    accepted_windows = [window for window in windows if window.accepted]
    unresolved_windows = [window for window in windows if not window.accepted]
    projected_samples = int(coeffs.shape[0]) * frames
    if unresolved_windows:
        reasons = sorted(
            {
                reason
                for window in unresolved_windows
                for reason in str(window.reason).split(",")
            }
        )
        unresolved_metadata = [
            {
                "start": int(window.start),
                "stop": int(window.stop),
                "sample_count": int(window.stop - window.start),
                "reason": str(window.reason),
                "reasons": str(window.reason).split(","),
                "sampled_max_fit_residual_uv_px": float(
                    window.fit.residual_max_uv.max().detach().cpu().item()
                ),
                "sampled_max_fit_residual_depth": float(
                    window.fit.residual_max_depth.max().detach().cpu().item()
                ),
                "min_denominator_abs": float(
                    window.fit.denominator_min_abs.min().detach().cpu().item()
                ),
                "denominator_root_count": int(
                    window.fit.denominator_has_root.sum().detach().cpu().item()
                ),
                "min_valid_fraction": float(
                    window.fit.valid_fraction.min().detach().cpu().item()
                ),
            }
            for window in unresolved_windows
        ]
        return {
            "row_status": UNRESOLVED_ROW_STATUS,
            "row_scope": "death_boundary_only_not_closure_evidence",
            "compiled_quality_metrics_status": COMPILED_QUALITY_UNAVAILABLE,
            "compiled_quality_metrics_unavailable": list(
                UNAVAILABLE_COMPILED_QUALITY_FIELDS
            ),
            "compiler_failure_class": "trace_window_certificate_unsatisfied",
            "motion_half_span_degrees": float(half_span_degrees),
            "motion_total_span_degrees": 2.0 * float(half_span_degrees),
            "physical_interval": [-1.0, 1.0],
            "sample_count": frames,
            "q_min": float(times[0].item()),
            "q_max": float(times[-1].item()),
            "chart_count": len(windows),
            "accepted_chart_count": len(accepted_windows),
            "unresolved_chart_count": len(unresolved_windows),
            "accepted_chart_fraction": len(accepted_windows)
            / float(max(1, len(windows))),
            "fit_residual_semantics": "empirical_max_over_requested_samples",
            "unresolved_chart_reasons": reasons,
            "unresolved_charts": unresolved_metadata,
            "unresolved_chart_metadata_sha256": _sha256_json(unresolved_metadata),
            "unresolved_max_fit_residual_uv_px": max(
                item["sampled_max_fit_residual_uv_px"] for item in unresolved_metadata
            ),
            "unresolved_max_fit_residual_depth": max(
                item["sampled_max_fit_residual_depth"] for item in unresolved_metadata
            ),
            "unresolved_min_denominator_abs": min(
                item["min_denominator_abs"] for item in unresolved_metadata
            ),
            "unresolved_min_valid_fraction": min(
                item["min_valid_fraction"] for item in unresolved_metadata
            ),
            "unresolved_denominator_root_count": sum(
                item["denominator_root_count"] for item in unresolved_metadata
            ),
            "projected_sample_count": projected_samples,
        }

    reference_bounds = bridge.bound_projective_trace_windows(
        windows,
        uv_padding=float(camera_program["image_width"]),
    )
    reference_cells = bridge.assemble_projective_trace_tile_time_atlas(
        bridge.bin_projective_trace_support_bounds(
            reference_bounds,
            image_width=int(camera_program["image_width"]),
            image_height=int(camera_program["image_height"]),
            tile_size=int(compiler["tile_size"]),
        )
    )
    reference_cells = [
        replace(
            cell,
            fallback=True,
            fallback_reasons=("oracle_all_live_depth_sort",),
        )
        for cell in reference_cells
    ]
    atlas = bridge.projective_trace_windows_to_cell_trace_atlas(
        windows,
        opacity=opacity,
        color=color,
        image_width=int(camera_program["image_width"]),
        image_height=int(camera_program["image_height"]),
        tile_size=int(compiler["tile_size"]),
        uv_padding=float(compiler["support_padding_px"]),
    )
    partition = bridge.projective_trace_cell_sensor_time_event_partition(
        atlas,
        times,
        image_width=int(camera_program["image_width"]),
        image_height=int(camera_program["image_height"]),
        tile_size=int(compiler["tile_size"]),
        uv_padding=float(compiler["support_padding_px"]),
        include_support=True,
        include_visibility=True,
    )
    atlas = bridge.stratify_projective_trace_cell_atlas_visibility_events(atlas, times)
    atlas = bridge.mark_projective_trace_cell_visibility_fallbacks(
        atlas,
        times,
        image_width=int(camera_program["image_width"]),
        image_height=int(camera_program["image_height"]),
        tile_size=int(compiler["tile_size"]),
    )
    reference = bridge.render_projective_trace_tile_time_atlas_reference(
        reference_cells,
        coeffs,
        times,
        color,
        opacity,
        image_width=int(camera_program["image_width"]),
        image_height=int(camera_program["image_height"]),
        tile_size=int(compiler["tile_size"]),
        sigma_px=float(compiler["sigma_px"]),
        allow_fallback_cells=True,
        fallback_sort_live_depth=True,
    )
    compiled = bridge.render_projective_trace_cell_atlas_reference(
        atlas,
        times,
        image_width=int(camera_program["image_width"]),
        image_height=int(camera_program["image_height"]),
        tile_size=int(compiler["tile_size"]),
        sigma_px=float(compiler["sigma_px"]),
        allow_fallback_cells=True,
    )
    delta = compiled - reference
    image_mse = float(delta.square().mean().detach().cpu().item())
    image_abs = delta.abs().reshape(-1)
    image_psnr = min(200.0, 10.0 * math.log10(1.0 / max(image_mse, 1.0e-20)))

    parameter_names = WORLD_VJP_PARAMETER_NAMES
    parameter_tensors = tuple(world["parameters"][name] for name in parameter_names)
    weights = torch.linspace(
        0.1,
        1.0,
        reference.numel(),
        dtype=reference.dtype,
    ).reshape_as(reference)
    reference_grads = torch.autograd.grad(
        (reference * weights).sum(),
        parameter_tensors,
        retain_graph=True,
    )
    compiled_grads = torch.autograd.grad(
        (compiled * weights).sum(),
        parameter_tensors,
    )
    vjp_errors = _relative_vjp_errors(torch, reference_grads, compiled_grads)
    reference_grad_norms = [
        float(torch.linalg.vector_norm(grad).detach().cpu().item())
        for grad in reference_grads
    ]
    compiled_grad_norms = [
        float(torch.linalg.vector_norm(grad).detach().cpu().item())
        for grad in compiled_grads
    ]

    complexity = bridge.projective_trace_cell_atlas_complexity_stats(atlas)
    fallback = bridge.projective_trace_cell_atlas_fallback_stats(atlas)
    visibility = bridge.projective_trace_cell_atlas_visibility_report(
        atlas,
        times,
        image_width=int(camera_program["image_width"]),
        image_height=int(camera_program["image_height"]),
        tile_size=int(compiler["tile_size"]),
        mark_ambiguous_stale=False,
    )
    dense_projective = bridge.eval_projective_trace_torch(coeffs, times)
    invalid_samples = int((dense_projective[:, :, 3] == 0.0).sum().item())
    fallback_cell_fraction = fallback.fallback_cells / float(max(1, fallback.total_cells))
    fallback_sample_fraction = fallback.fallback_trace_samples / float(
        max(1, fallback.total_trace_samples)
    )
    chart_count = len(windows)
    event_count = len(partition.support_events) + len(partition.visibility_events)
    trace_count = int(atlas.coeffs.shape[0])
    row = {
        "row_status": COMPILED_ROW_STATUS,
        "row_scope": "compiled_quality_closure_or_threshold_death",
        "compiled_quality_metrics_status": COMPILED_QUALITY_AVAILABLE,
        "compiled_quality_metrics_unavailable": [],
        "motion_half_span_degrees": float(half_span_degrees),
        "motion_total_span_degrees": 2.0 * float(half_span_degrees),
        "physical_interval": [-1.0, 1.0],
        "sample_count": frames,
        "q_min": float(times[0].item()),
        "q_max": float(times[-1].item()),
        "chart_count": chart_count,
        "accepted_chart_count": len(accepted_windows),
        "unresolved_chart_count": len(unresolved_windows),
        "unresolved_chart_reasons": [],
        "unresolved_charts": [],
        "unresolved_chart_metadata_sha256": _sha256_json([]),
        "accepted_chart_fraction": len(accepted_windows) / float(max(1, chart_count)),
        "sampled_max_fit_residual_uv_px": max(
            float(window.fit.residual_max_uv.max().detach().cpu().item())
            for window in windows
        ),
        "fit_residual_semantics": "empirical_max_over_requested_samples",
        "min_denominator_abs": min(
            float(window.fit.denominator_min_abs.min().detach().cpu().item())
            for window in windows
        ),
        "support_event_count": len(partition.support_events),
        "visibility_event_count": len(partition.visibility_events),
        "event_count": event_count,
        "event_interval_count": len(partition.intervals),
        "reference_support_policy": "full_image",
        "reference_order_policy": "all_live_depth_per_sample",
        "reference_fallback_reason": "oracle_all_live_depth_sort",
        "reference_sample_semantics": "empirical_at_requested_samples",
        "reference_cell_count": len(reference_cells),
        "reference_live_sorted_cell_count": sum(
            cell.fallback
            and cell.fallback_reasons == ("oracle_all_live_depth_sort",)
            for cell in reference_cells
        ),
        "trace_count": trace_count,
        "trace_to_replay_ratio": trace_count / float(max(1, projected_samples)),
        "cell_count": int(complexity.total_cells),
        "visibility_stratum_split_cell_count": int(
            complexity.visibility_stratum_split_cells
        ),
        "interval_entry_count": int(complexity.interval_trace_entries),
        "dense_trace_samples": int(complexity.dense_trace_samples),
        "interval_to_dense_ratio": float(
            complexity.interval_to_dense_trace_sample_ratio
        ),
        "fallback_cell_count": int(fallback.fallback_cells),
        "fallback_cell_fraction": fallback_cell_fraction,
        "fallback_trace_samples": int(fallback.fallback_trace_samples),
        "fallback_sample_fraction": fallback_sample_fraction,
        "fallback_reasons": list(fallback.fallback_reasons),
        "invalid_sample_count": invalid_samples,
        "projected_sample_count": projected_samples,
        "invalid_sample_fraction": invalid_samples / float(max(1, projected_samples)),
        "post_visibility_stale": bool(visibility.stale),
        "post_order_mismatch_sample_count": int(visibility.order_mismatch_samples),
        "post_ambiguous_depth_sample_count": int(visibility.ambiguous_depth_samples),
        "image_mse": image_mse,
        "image_psnr_db": image_psnr,
        "image_p999_abs_error": float(
            torch.quantile(image_abs, 0.999).detach().cpu().item()
        ),
        "image_max_abs_error": float(image_abs.max().detach().cpu().item()),
        "world_vjp_rel_l2_by_parameter": dict(zip(parameter_names, vjp_errors, strict=True)),
        "world_vjp_rel_l2_max": max(vjp_errors, default=0.0),
        "world_vjp_parameter_names": list(parameter_names),
        "world_vjp_reference_norm_by_parameter": dict(
            zip(parameter_names, reference_grad_norms, strict=True)
        ),
        "world_vjp_compiled_norm_by_parameter": dict(
            zip(parameter_names, compiled_grad_norms, strict=True)
        ),
        "world_vjp_nonzero_parameter_count": sum(
            max(reference_norm, compiled_norm) > 1.0e-8
            for reference_norm, compiled_norm in zip(
                reference_grad_norms,
                compiled_grad_norms,
                strict=True,
            )
        ),
        "vjp_topology_semantics": "fixed_compiled_topology_away_from_event_boundaries",
    }
    return row


def _protocol_components(
    *,
    frames: int,
    image_size: int,
    tile_size: int,
    sigma_px: float,
    support_padding_px: float,
    max_residual_uv: float,
    max_depth_residual: float,
    min_denominator_abs: float,
    max_windows: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, float]]:
    return (
        default_world_fixture(),
        default_camera_program(frames=frames, image_size=image_size),
        default_compiler_contract(
            tile_size=tile_size,
            sigma_px=sigma_px,
            support_padding_px=support_padding_px,
            max_residual_uv=max_residual_uv,
            max_depth_residual=max_depth_residual,
            min_denominator_abs=min_denominator_abs,
            max_windows=max_windows,
        ),
        default_thresholds(),
    )


def run_report(
    *,
    half_spans_degrees: list[float],
    frames: int,
    image_size: int,
    tile_size: int,
    sigma_px: float,
    support_padding_px: float,
    max_residual_uv: float,
    max_depth_residual: float,
    min_denominator_abs: float,
    max_windows: int,
    source_start: dict[str, Any] | None = None,
    dirty_source_allowed: bool = False,
) -> dict[str, Any]:
    source = source_provenance() if source_start is None else dict(source_start)
    if not dirty_source_allowed:
        require_clean_source(source)
    if len(half_spans_degrees) < 4:
        raise ValueError("half_spans_degrees must contain at least four points")
    if half_spans_degrees != sorted(half_spans_degrees) or len(set(half_spans_degrees)) != len(
        half_spans_degrees
    ):
        raise ValueError("half_spans_degrees must be strictly increasing")
    if any(span <= 0.0 or 2.0 * span >= 360.0 for span in half_spans_degrees):
        raise ValueError("camera motion must be positive and strictly below a 360-degree path")
    if frames < 16:
        raise ValueError("frames must be at least 16")
    if image_size <= 0 or tile_size <= 0 or image_size % tile_size != 0:
        raise ValueError("image_size must be positive and divisible by tile_size")
    if sigma_px <= 0.0 or support_padding_px <= 0.0:
        raise ValueError("sigma_px and support_padding_px must be positive")

    world_fixture, camera_program, compiler, thresholds = _protocol_components(
        frames=frames,
        image_size=image_size,
        tile_size=tile_size,
        sigma_px=sigma_px,
        support_padding_px=support_padding_px,
        max_residual_uv=max_residual_uv,
        max_depth_residual=max_depth_residual,
        min_denominator_abs=min_denominator_abs,
        max_windows=max_windows,
    )
    rows = []
    for half_span in half_spans_degrees:
        try:
            row = _run_row(
                half_span_degrees=half_span,
                world_fixture=world_fixture,
                camera_program=camera_program,
                compiler=compiler,
            )
            rows.append(row)
            if row.get("row_status") == UNRESOLVED_ROW_STATUS:
                break
        except Exception as error:
            raise VariableCameraCurveExecutionError(
                failed_half_span_degrees=half_span,
                completed_row_count=len(rows),
                cause=error,
            ) from error
    torch, _bridge = _runtime_modules()
    source_finish = source_provenance()
    if source_finish != source:
        raise RuntimeError(
            "source changed while the variable-camera paper gate executed"
        )
    return assemble_report(
        rows,
        half_spans_degrees=half_spans_degrees,
        world_fixture=world_fixture,
        camera_program=camera_program,
        compiler=compiler,
        thresholds=thresholds,
        runtime={
            "device": "cpu",
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "platform": platform.platform(),
        },
        implementation=_implementation_manifest(),
        source=source,
        source_finish=source_finish,
        dirty_source_allowed=dirty_source_allowed,
    )


def assemble_failure_report(
    *,
    error: Exception,
    half_spans_degrees: list[float],
    frames: int,
    image_size: int,
    tile_size: int,
    sigma_px: float,
    support_padding_px: float,
    max_residual_uv: float,
    max_depth_residual: float,
    min_denominator_abs: float,
    max_windows: int,
    source: dict[str, Any] | None = None,
    source_finish: dict[str, Any] | None = None,
    dirty_source_allowed: bool = False,
) -> dict[str, Any]:
    """Build a source-bound failure artifact without importing Torch."""

    world_fixture, camera_program, compiler, thresholds = _protocol_components(
        frames=frames,
        image_size=image_size,
        tile_size=tile_size,
        sigma_px=sigma_px,
        support_padding_px=support_padding_px,
        max_residual_uv=max_residual_uv,
        max_depth_residual=max_depth_residual,
        min_denominator_abs=min_denominator_abs,
        max_windows=max_windows,
    )
    world_hash = _sha256_json(world_fixture)
    camera_hash = _sha256_json(camera_program)
    compiler_hash = _sha256_json(compiler)
    thresholds_hash = _sha256_json(thresholds)
    contract_payload = _experiment_contract_payload(
        world_fixture_sha256=world_hash,
        camera_program_sha256=camera_hash,
        compiler_sha256=compiler_hash,
        half_spans_degrees=half_spans_degrees,
        thresholds_sha256=thresholds_hash,
    )
    if isinstance(error, VariableCameraCurveExecutionError):
        failure = {
            "stage": "camera_row_execution",
            "failed_half_span_degrees": error.failed_half_span_degrees,
            "completed_row_count": error.completed_row_count,
            "exception_type": error.cause_type,
            "message": error.cause_message,
        }
    else:
        failure = {
            "stage": "protocol_or_runtime_setup",
            "failed_half_span_degrees": None,
            "completed_row_count": 0,
            "exception_type": type(error).__name__,
            "message": str(error),
        }
    if source is None:
        try:
            recorded_source = source_provenance()
        except Exception as source_error:
            recorded_source = {
                "repository_commit": "0" * 40,
                "repository_dirty": True,
                "star_uvt_commit": "0" * 40,
                "star_uvt_dirty": True,
                "provenance_error": (
                    f"{type(source_error).__name__}: {source_error}"
                ),
            }
    else:
        recorded_source = dict(source)
    recorded_source_finish = (
        recorded_source
        if source_finish is None
        else dict(source_finish)
    )
    source_eligible = (
        recorded_source == recorded_source_finish
        and recorded_source.get("repository_dirty") is False
        and recorded_source.get("star_uvt_dirty") is False
    )
    report: dict[str, Any] = {
        "benchmark": BENCHMARK,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "bounded_synthetic_variable_camera_closure_death_curve",
        "status": "runtime_failure",
        "artifact_semantics": "structured_failure_not_paper_evidence",
        "motion_half_spans_degrees": [float(value) for value in half_spans_degrees],
        "world_fixture": world_fixture,
        "world_fixture_sha256": world_hash,
        "camera_program": camera_program,
        "camera_program_sha256": camera_hash,
        "compiler": compiler,
        "compiler_sha256": compiler_hash,
        "thresholds": thresholds,
        "thresholds_sha256": thresholds_hash,
        "experiment_contract_sha256": _sha256_json(contract_payload),
        "failure": failure,
        "runtime": {
            "device": "not_completed",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "implementation": _implementation_manifest(),
        "source": recorded_source,
        "source_finish": recorded_source_finish,
        "source_sha256": _sha256_json(recorded_source),
        "source_policy": {
            "dirty_source_allowed": bool(dirty_source_allowed),
            "paper_evidence_eligible": source_eligible,
        },
        "rows": [],
        "acceptance": {
            "accepted": False,
            "label": "runtime_failure",
            "reasons": ["runtime_exception"],
            "claim_scope": "no paper claim; inspect failure before rerunning",
        },
    }
    report["failure_report_sha256"] = _sha256_json(
        {key: value for key, value in report.items() if key != "failure_report_sha256"}
    )
    return report


def verify_variable_camera_failure_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("benchmark") != BENCHMARK:
        errors.append(f"benchmark must be {BENCHMARK}")
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if report.get("status") != "runtime_failure":
        errors.append("status must be runtime_failure")
    if report.get("artifact_semantics") != "structured_failure_not_paper_evidence":
        errors.append("artifact_semantics mismatch")
    spans = report.get("motion_half_spans_degrees")
    if not isinstance(spans, list):
        errors.append("motion_half_spans_degrees must be a list")
        spans = []
    parsed_spans = [
        _finite_float(
            value,
            f"motion_half_spans_degrees[{index}]",
            errors,
        )
        for index, value in enumerate(spans)
    ]

    identity_fields = (
        ("world_fixture", "world_fixture_sha256"),
        ("camera_program", "camera_program_sha256"),
        ("compiler", "compiler_sha256"),
        ("thresholds", "thresholds_sha256"),
    )
    identities: dict[str, str] = {}
    for object_key, hash_key in identity_fields:
        value = report.get(object_key)
        if not isinstance(value, dict):
            errors.append(f"{object_key} must be an object")
            value = {}
        expected_hash = _sha256_json(value)
        if report.get(hash_key) != expected_hash:
            errors.append(f"{hash_key} mismatch")
        identities[hash_key] = expected_hash
    expected_contract_hash = _sha256_json(
        _experiment_contract_payload(
            world_fixture_sha256=identities["world_fixture_sha256"],
            camera_program_sha256=identities["camera_program_sha256"],
            compiler_sha256=identities["compiler_sha256"],
            half_spans_degrees=parsed_spans,
            thresholds_sha256=identities["thresholds_sha256"],
        )
    )
    if report.get("experiment_contract_sha256") != expected_contract_hash:
        errors.append("experiment_contract_sha256 mismatch")

    failure = report.get("failure")
    if not isinstance(failure, dict):
        errors.append("failure must be an object")
    else:
        if failure.get("stage") not in {"camera_row_execution", "protocol_or_runtime_setup"}:
            errors.append("failure.stage is invalid")
        if not isinstance(failure.get("exception_type"), str) or not failure["exception_type"]:
            errors.append("failure.exception_type must be a nonempty string")
        if not isinstance(failure.get("message"), str) or not failure["message"]:
            errors.append("failure.message must be a nonempty string")
        completed = _finite_int(
            failure.get("completed_row_count"),
            "failure.completed_row_count",
            errors,
        )
        if completed < 0 or completed > len(spans):
            errors.append("failure.completed_row_count is invalid")
        failed_span = failure.get("failed_half_span_degrees")
        if failed_span is not None:
            parsed_failed_span = _finite_float(
                failed_span,
                "failure.failed_half_span_degrees",
                errors,
            )
            if parsed_failed_span not in parsed_spans:
                errors.append("failure.failed_half_span_degrees is not in the requested sweep")
    if report.get("rows") != []:
        errors.append("failure report rows must be empty because partial rows are not paper evidence")
    expected_acceptance = {
        "accepted": False,
        "label": "runtime_failure",
        "reasons": ["runtime_exception"],
        "claim_scope": "no paper claim; inspect failure before rerunning",
    }
    if report.get("acceptance") != expected_acceptance:
        errors.append("failure acceptance mismatch")
    source_errors, _source_eligible = _verify_source_provenance(
        report,
        require_paper_eligible=False,
    )
    errors.extend(source_errors)
    errors.extend(_verify_implementation_manifest(report.get("implementation")))
    expected_report_hash = _sha256_json(
        {key: value for key, value in report.items() if key != "failure_report_sha256"}
    )
    if report.get("failure_report_sha256") != expected_report_hash:
        errors.append("failure_report_sha256 mismatch")
    return errors


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    columns = (
        "motion_half_span_degrees",
        "regime",
        "row_status",
        "chart_count",
        "unresolved_chart_count",
        "unresolved_chart_reasons",
        "support_event_count",
        "visibility_event_count",
        "event_interval_count",
        "trace_count",
        "interval_entry_count",
        "interval_to_dense_ratio",
        "fallback_sample_fraction",
        "invalid_sample_fraction",
        "image_psnr_db",
        "image_p999_abs_error",
        "world_vjp_rel_l2_max",
        "compiled_quality_metrics_status",
        "death_reasons",
    )
    lines = [
        "# World Tubes bounded variable-camera closure/death curve",
        "",
        (
            "A fixed synthetic world and the fixed physical interval `s in [-1,1]` "
            "are evaluated while only bounded yaw motion increases. This is an "
            "open-path result, not a 360/720-degree transition or holonomy claim."
        ),
        "",
        "## Acceptance",
        "",
        "```json",
        json.dumps(report["acceptance"], indent=2, sort_keys=True),
        "```",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(report["summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## Rows",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in report["rows"]:
        lines.append("| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_failure_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# World Tubes variable-camera gate runtime failure",
        "",
        (
            "This is a structured execution failure, not paper evidence. "
            "Inspect and resolve the failure before rerunning the bounded sweep."
        ),
        "",
        "```json",
        json.dumps(report["failure"], indent=2, sort_keys=True),
        "```",
        "",
        f"Contract SHA-256: `{report['experiment_contract_sha256']}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_half_spans(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("motion half-span list must not be empty")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or verify the bounded World Tubes variable-camera closure/death curve."
    )
    parser.add_argument(
        "--motion-half-spans-degrees",
        default=",".join(f"{value:g}" for value in DEFAULT_HALF_SPANS_DEGREES),
    )
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--sigma-px", type=float, default=1.6)
    parser.add_argument("--support-padding-px", type=float, default=6.0)
    parser.add_argument("--max-residual-uv", type=float, default=0.25)
    parser.add_argument("--max-depth-residual", type=float, default=0.025)
    parser.add_argument("--min-denominator-abs", type=float, default=1.0e-3)
    parser.add_argument("--max-windows", type=int, default=256)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--verify-report", type=Path)
    parser.add_argument("--require-current-source", action="store_true")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--require-clean-source",
        action="store_true",
        help="Compatibility flag; paper execution requires clean source by default.",
    )
    source_group.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help=(
            "Permit a labelled mechanical run from dirty source. The emitted "
            "artifact is explicitly ineligible for paper evidence."
        ),
    )
    args = parser.parse_args()

    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text(encoding="utf-8"))
        if report.get("status") == "runtime_failure":
            errors = verify_variable_camera_failure_report(report)
        else:
            errors = verify_variable_camera_closure_death_curve(
                report,
                require_paper_eligible=not args.allow_dirty_source,
            )
        if args.require_current_source:
            errors.extend(verify_current_implementation(report))
        if errors:
            raise AssertionError(
                "variable-camera closure/death curve failed:\n- " + "\n- ".join(errors)
            )
        print(f"verified {args.verify_report}")
        return
    if not args.execute:
        parser.error("pass --execute to run the Torch CPU gate, or --verify-report to validate an artifact")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "summary.json"
    markdown_path = args.out_dir / "summary.md"
    half_spans: list[float] = []
    source_start: dict[str, Any] | None = None
    try:
        source_start = source_provenance()
        if not args.allow_dirty_source:
            require_clean_source(source_start)
        half_spans = _parse_half_spans(args.motion_half_spans_degrees)
        report = run_report(
            half_spans_degrees=half_spans,
            frames=args.frames,
            image_size=args.image_size,
            tile_size=args.tile_size,
            sigma_px=args.sigma_px,
            support_padding_px=args.support_padding_px,
            max_residual_uv=args.max_residual_uv,
            max_depth_residual=args.max_depth_residual,
            min_denominator_abs=args.min_denominator_abs,
            max_windows=args.max_windows,
            source_start=source_start,
            dirty_source_allowed=args.allow_dirty_source,
        )
    except Exception as error:
        try:
            source_finish = source_provenance()
        except Exception:
            source_finish = source_start
        failure_report = assemble_failure_report(
            error=error,
            half_spans_degrees=half_spans,
            frames=args.frames,
            image_size=args.image_size,
            tile_size=args.tile_size,
            sigma_px=args.sigma_px,
            support_padding_px=args.support_padding_px,
            max_residual_uv=args.max_residual_uv,
            max_depth_residual=args.max_depth_residual,
            min_denominator_abs=args.min_denominator_abs,
            max_windows=args.max_windows,
            source=source_start,
            source_finish=source_finish,
            dirty_source_allowed=args.allow_dirty_source,
        )
        json_path.write_text(
            json.dumps(failure_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_failure_markdown(failure_report, markdown_path)
        failure_errors = verify_variable_camera_failure_report(failure_report)
        if failure_errors:
            raise AssertionError(
                "structured variable-camera failure artifact is invalid:\n- "
                + "\n- ".join(failure_errors)
            ) from error
        print(f"wrote structured failure artifact {json_path}", file=sys.stderr)
        raise
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, markdown_path)
    assert_variable_camera_closure_death_curve(
        report,
        require_paper_eligible=not args.allow_dirty_source,
    )
    print(json.dumps(report["acceptance"], indent=2, sort_keys=True))
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
