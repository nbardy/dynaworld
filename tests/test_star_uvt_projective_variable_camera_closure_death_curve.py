from __future__ import annotations

import copy
import math
import struct

from research_experiments.star_uvt_feature_tubes.projective_variable_camera_closure_death_curve import (
    COMPILED_QUALITY_AVAILABLE,
    COMPILED_QUALITY_UNAVAILABLE,
    COMPILED_ROW_STATUS,
    REQUIRED_IMPLEMENTATION_SOURCE_PATHS,
    UNAVAILABLE_COMPILED_QUALITY_FIELDS,
    UNRESOLVED_ROW_STATUS,
    VariableCameraCurveExecutionError,
    _sha256_json,
    assemble_failure_report,
    assemble_report,
    assert_variable_camera_closure_death_curve,
    default_camera_program,
    default_compiler_contract,
    default_thresholds,
    default_world_fixture,
    summarize,
    verify_variable_camera_closure_death_curve,
    verify_variable_camera_failure_report,
)


HALF_SPANS = [15.0, 45.0, 75.0, 120.0]
FRAMES = 64
IMAGE_SIZE = 64
SOURCE = {
    "repository_commit": "a" * 40,
    "repository_dirty": False,
    "star_uvt_commit": "b" * 40,
    "star_uvt_dirty": False,
}


def _row(*, span: float, chart_count: int, dead: bool) -> dict[str, object]:
    primitive_count = 3
    projected_samples = primitive_count * FRAMES
    trace_count = primitive_count * chart_count
    cell_count = 4 * chart_count
    interval_entries = 2 * trace_count
    dense_samples = 8 * interval_entries
    image_mse = 1.0e-3 if dead else 1.0e-8
    image_psnr = 10.0 * math.log10(1.0 / image_mse)
    vjp_error = 0.05 if dead else 0.005
    q_extent = math.tan(0.5 * math.radians(span))
    return {
        "row_status": COMPILED_ROW_STATUS,
        "row_scope": "compiled_quality_closure_or_threshold_death",
        "compiled_quality_metrics_status": COMPILED_QUALITY_AVAILABLE,
        "compiled_quality_metrics_unavailable": [],
        "motion_half_span_degrees": span,
        "motion_total_span_degrees": 2.0 * span,
        "physical_interval": [-1.0, 1.0],
        "sample_count": FRAMES,
        "q_min": -q_extent,
        "q_max": q_extent,
        "chart_count": chart_count,
        "accepted_chart_count": chart_count,
        "unresolved_chart_count": 0,
        "unresolved_chart_reasons": [],
        "unresolved_charts": [],
        "unresolved_chart_metadata_sha256": _sha256_json([]),
        "accepted_chart_fraction": 1.0,
        "sampled_max_fit_residual_uv_px": 0.1,
        "fit_residual_semantics": "empirical_max_over_requested_samples",
        "min_denominator_abs": 1.0,
        "support_event_count": chart_count,
        "visibility_event_count": 0,
        "event_count": chart_count,
        "event_interval_count": chart_count + 1,
        "reference_support_policy": "full_image",
        "reference_order_policy": "all_live_depth_per_sample",
        "reference_fallback_reason": "oracle_all_live_depth_sort",
        "reference_sample_semantics": "empirical_at_requested_samples",
        "reference_cell_count": cell_count,
        "reference_live_sorted_cell_count": cell_count,
        "trace_count": trace_count,
        "trace_to_replay_ratio": trace_count / projected_samples,
        "cell_count": cell_count,
        "visibility_stratum_split_cell_count": 0,
        "interval_entry_count": interval_entries,
        "dense_trace_samples": dense_samples,
        "interval_to_dense_ratio": interval_entries / dense_samples,
        "fallback_cell_count": 0,
        "fallback_cell_fraction": 0.0,
        "fallback_trace_samples": 0,
        "fallback_sample_fraction": 0.0,
        "fallback_reasons": [],
        "invalid_sample_count": 0,
        "projected_sample_count": projected_samples,
        "invalid_sample_fraction": 0.0,
        "post_visibility_stale": False,
        "post_order_mismatch_sample_count": 0,
        "post_ambiguous_depth_sample_count": 0,
        "image_mse": image_mse,
        "image_psnr_db": image_psnr,
        "image_p999_abs_error": 0.05 if dead else 0.001,
        "image_max_abs_error": 0.08 if dead else 0.002,
        "world_vjp_rel_l2_by_parameter": {
            "point_x": vjp_error,
            "base_depth": vjp_error,
            "vertical": vjp_error,
            "opacity": vjp_error,
            "color": vjp_error,
        },
        "world_vjp_rel_l2_max": vjp_error,
        "world_vjp_parameter_names": [
            "point_x",
            "base_depth",
            "vertical",
            "opacity",
            "color",
        ],
        "world_vjp_reference_norm_by_parameter": {
            "point_x": 1.0,
            "base_depth": 1.0,
            "vertical": 1.0,
            "opacity": 1.0,
            "color": 1.0,
        },
        "world_vjp_compiled_norm_by_parameter": {
            "point_x": 1.0,
            "base_depth": 1.0,
            "vertical": 1.0,
            "opacity": 1.0,
            "color": 1.0,
        },
        "world_vjp_nonzero_parameter_count": 5,
        "vjp_topology_semantics": "fixed_compiled_topology_away_from_event_boundaries",
    }


def _unresolved_row(*, span: float, chart_count: int) -> dict[str, object]:
    q_extent = math.tan(0.5 * math.radians(span))
    unresolved = [
        {
            "start": FRAMES - 2,
            "stop": FRAMES,
            "sample_count": 2,
            "reason": "depth_residual",
            "reasons": ["depth_residual"],
            "sampled_max_fit_residual_uv_px": 0.1,
            "sampled_max_fit_residual_depth": 0.04,
            "min_denominator_abs": 1.0,
            "denominator_root_count": 0,
            "min_valid_fraction": 1.0,
        }
    ]
    return {
        "row_status": UNRESOLVED_ROW_STATUS,
        "row_scope": "death_boundary_only_not_closure_evidence",
        "compiled_quality_metrics_status": COMPILED_QUALITY_UNAVAILABLE,
        "compiled_quality_metrics_unavailable": list(UNAVAILABLE_COMPILED_QUALITY_FIELDS),
        "compiler_failure_class": "trace_window_certificate_unsatisfied",
        "motion_half_span_degrees": span,
        "motion_total_span_degrees": 2.0 * span,
        "physical_interval": [-1.0, 1.0],
        "sample_count": FRAMES,
        "q_min": -q_extent,
        "q_max": q_extent,
        "chart_count": chart_count,
        "accepted_chart_count": chart_count - 1,
        "unresolved_chart_count": 1,
        "accepted_chart_fraction": (chart_count - 1) / chart_count,
        "fit_residual_semantics": "empirical_max_over_requested_samples",
        "unresolved_chart_reasons": ["depth_residual"],
        "unresolved_charts": unresolved,
        "unresolved_chart_metadata_sha256": _sha256_json(unresolved),
        "unresolved_max_fit_residual_uv_px": 0.1,
        "unresolved_max_fit_residual_depth": 0.04,
        "unresolved_min_denominator_abs": 1.0,
        "unresolved_min_valid_fraction": 1.0,
        "unresolved_denominator_root_count": 0,
        "projected_sample_count": 3 * FRAMES,
    }


def _valid_report(*, all_closure: bool = False) -> dict[str, object]:
    if all_closure:
        rows = [
            _row(span=span, chart_count=chart_count, dead=False)
            for span, chart_count in zip(HALF_SPANS, (2, 3, 5, 8), strict=True)
        ]
    else:
        rows = [
            _row(span=HALF_SPANS[0], chart_count=2, dead=False),
            _row(span=HALF_SPANS[1], chart_count=3, dead=False),
            _row(span=HALF_SPANS[2], chart_count=5, dead=True),
            _unresolved_row(span=HALF_SPANS[3], chart_count=8),
        ]
    source_files = [
        {
            "path": path,
            "sha256": "a" * 64,
        }
        for path in REQUIRED_IMPLEMENTATION_SOURCE_PATHS
    ]
    return assemble_report(
        rows,
        half_spans_degrees=HALF_SPANS,
        world_fixture=default_world_fixture(),
        camera_program=default_camera_program(frames=FRAMES, image_size=IMAGE_SIZE),
        compiler=default_compiler_contract(
            tile_size=16,
            sigma_px=1.6,
            support_padding_px=6.0,
            max_residual_uv=0.25,
            max_depth_residual=0.025,
            min_denominator_abs=1.0e-3,
            max_windows=256,
        ),
        thresholds=default_thresholds(),
        runtime={"device": "cpu", "torch_version": "fixture"},
        implementation={
            "source_files": source_files,
            "source_manifest_sha256": _sha256_json(source_files),
        },
        source=dict(SOURCE),
        source_finish=dict(SOURCE),
        dirty_source_allowed=False,
    )


def test_variable_camera_curve_accepts_bound_closure_then_death() -> None:
    report = _valid_report()

    assert verify_variable_camera_closure_death_curve(report) == []
    assert_variable_camera_closure_death_curve(report)
    assert report["summary"]["regime_sequence"] == [
        "closure",
        "closure",
        "death",
        "death",
    ]
    assert report["acceptance"]["accepted"] is True


def test_variable_camera_curve_rejects_camera_program_identity_drift() -> None:
    report = _valid_report()
    report["camera_program"]["sample_count"] = 32

    errors = verify_variable_camera_closure_death_curve(report)

    assert any("camera_program_sha256 mismatch" in error for error in errors)
    assert any("experiment_contract_sha256 mismatch" in error for error in errors)
    assert any("sample_count must match" in error for error in errors)


def test_variable_camera_curve_rejects_stale_death_label() -> None:
    report = _valid_report()
    report["rows"][2]["accepted"] = True
    report["rows"][2]["regime"] = "closure"
    report["rows"][2]["death_reasons"] = []
    report["summary"] = summarize(report["rows"])

    errors = verify_variable_camera_closure_death_curve(report)

    assert any("death_reasons mismatch" in error for error in errors)
    assert any(".accepted mismatch" in error for error in errors)
    assert any(".regime mismatch" in error for error in errors)


def test_variable_camera_curve_rejects_curve_without_observed_death() -> None:
    report = _valid_report(all_closure=True)

    errors = verify_variable_camera_closure_death_curve(report)

    assert any("both a closure regime and a death boundary" in error for error in errors)


def test_variable_camera_curve_rejects_nonmonotone_regime_sequence() -> None:
    report = _valid_report()
    closure_row = copy.deepcopy(report["rows"][1])
    closure_row["motion_half_span_degrees"] = HALF_SPANS[3]
    closure_row["motion_total_span_degrees"] = 2.0 * HALF_SPANS[3]
    report["rows"][3] = closure_row
    report["rows"][3]["camera_program_sha256"] = report["rows"][3][
        "camera_program_sha256"
    ]
    report["summary"] = summarize(report["rows"])

    errors = verify_variable_camera_closure_death_curve(report)

    assert any("closure prefix followed by a death suffix" in error for error in errors)


def test_variable_camera_curve_rejects_inconsistent_fractions_and_nonfinite_metrics() -> None:
    report = _valid_report()
    report["rows"][0]["interval_to_dense_ratio"] = 0.75
    report["rows"][0]["image_mse"] = float("nan")

    errors = verify_variable_camera_closure_death_curve(report)

    assert any("interval_to_dense_ratio mismatch" in error for error in errors)
    assert any("image_mse must be a finite number" in error for error in errors)


def test_variable_camera_curve_rejects_stale_summary() -> None:
    report = _valid_report()
    report["summary"]["max_chart_count"] = 999

    errors = verify_variable_camera_closure_death_curve(report)

    assert any("summary max_chart_count mismatch" in error for error in errors)


def test_variable_camera_curve_requires_all_live_depth_reference_oracle() -> None:
    report = _valid_report()
    report["rows"][0]["reference_order_policy"] = "interval_midpoint"

    errors = verify_variable_camera_closure_death_curve(report)

    assert any("reference_order_policy mismatch" in error for error in errors)


def test_variable_camera_curve_requires_complete_bridge_source_manifest() -> None:
    report = _valid_report()
    report["implementation"]["source_files"].pop(1)
    report["implementation"]["source_manifest_sha256"] = _sha256_json(
        report["implementation"]["source_files"]
    )

    errors = verify_variable_camera_closure_death_curve(report)

    assert any("bridge __init__" in error for error in errors)


def test_variable_camera_curve_rejects_dirty_source_as_paper_evidence() -> None:
    report = _valid_report()
    report["source"]["repository_dirty"] = True
    report["source_finish"]["repository_dirty"] = True
    report["source_sha256"] = _sha256_json(report["source"])
    report["source_policy"] = {
        "dirty_source_allowed": True,
        "paper_evidence_eligible": False,
    }

    errors = verify_variable_camera_closure_death_curve(report)

    assert any("requires unchanged clean" in error for error in errors)
    assert any("acceptance mismatch" in error for error in errors)


def test_variable_camera_curve_rejects_incomplete_vjp_norm_keys() -> None:
    report = _valid_report()
    report["rows"][0]["world_vjp_reference_norm_by_parameter"].pop("color")

    errors = verify_variable_camera_closure_death_curve(report)

    assert any(
        "world_vjp_reference_norm_by_parameter keys must be exactly" in error
        for error in errors
    )


def test_variable_camera_curve_accepts_float32_q_endpoints_near_chart_pole() -> None:
    report = _valid_report()
    row = report["rows"][-1]
    expected_q = math.tan(0.5 * math.radians(HALF_SPANS[-1]))
    row["q_min"] = struct.unpack("f", struct.pack("f", -expected_q))[0]
    row["q_max"] = struct.unpack("f", struct.pack("f", expected_q))[0]

    assert verify_variable_camera_closure_death_curve(report) == []


def test_variable_camera_curve_rejects_material_q_endpoint_drift() -> None:
    report = _valid_report()
    report["rows"][-1]["q_max"] *= 1.0001

    errors = verify_variable_camera_closure_death_curve(report)

    assert any("q_max mismatch" in error for error in errors)


def test_variable_camera_curve_rejects_fabricated_quality_on_unresolved_row() -> None:
    report = _valid_report()
    report["rows"][-1]["image_psnr_db"] = 200.0

    errors = verify_variable_camera_closure_death_curve(report)

    assert any("fabricates unavailable compiled metrics" in error for error in errors)


def test_variable_camera_curve_rejects_stale_unresolved_chart_metadata() -> None:
    report = _valid_report()
    report["rows"][-1]["unresolved_charts"][0][
        "sampled_max_fit_residual_depth"
    ] = 0.01

    errors = verify_variable_camera_closure_death_curve(report)

    assert any("unresolved_chart_metadata_sha256 mismatch" in error for error in errors)
    assert any("depth_residual is not witnessed" in error for error in errors)


def test_variable_camera_curve_rejects_stale_death_request_identity() -> None:
    report = _valid_report()
    report["rows"][-1]["compiler_sha256"] = "0" * 64
    report["rows"][-1]["row_request_sha256"] = "1" * 64

    errors = verify_variable_camera_closure_death_curve(report)

    assert any("compiler_sha256 mismatch" in error for error in errors)
    assert any("row_request_sha256 mismatch" in error for error in errors)


def test_variable_camera_runtime_failure_artifact_is_structured_and_source_bound() -> None:
    failure = VariableCameraCurveExecutionError(
        failed_half_span_degrees=75.0,
        completed_row_count=2,
        cause=RuntimeError("unresolved sampled polynomial window"),
    )
    report = assemble_failure_report(
        error=failure,
        half_spans_degrees=HALF_SPANS,
        frames=FRAMES,
        image_size=IMAGE_SIZE,
        tile_size=16,
        sigma_px=1.6,
        support_padding_px=6.0,
        max_residual_uv=0.25,
        max_depth_residual=0.025,
        min_denominator_abs=1.0e-3,
        max_windows=256,
    )

    assert verify_variable_camera_failure_report(report) == []
    assert report["status"] == "runtime_failure"
    assert report["failure"]["failed_half_span_degrees"] == 75.0
    assert report["failure"]["completed_row_count"] == 2
    assert report["acceptance"]["accepted"] is False


def test_variable_camera_runtime_failure_rejects_stale_failure_hash() -> None:
    report = assemble_failure_report(
        error=RuntimeError("setup failed"),
        half_spans_degrees=HALF_SPANS,
        frames=FRAMES,
        image_size=IMAGE_SIZE,
        tile_size=16,
        sigma_px=1.6,
        support_padding_px=6.0,
        max_residual_uv=0.25,
        max_depth_residual=0.025,
        min_denominator_abs=1.0e-3,
        max_windows=256,
    )
    report["failure"]["message"] = "tampered"

    errors = verify_variable_camera_failure_report(report)

    assert any("failure_report_sha256 mismatch" in error for error in errors)
