from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_orbit_fixed_chart_scaling_benchmark import (
    assert_orbit_fixed_chart_scaling_report,
    summarize,
    verify_orbit_fixed_chart_scaling_report,
)


SAVED_ORBIT_FIXED_CHART_ARTIFACT = Path(
    "outputs/benchmarks/2026-05-24_star_uvt_revolving_orbit_fixed_chart_scaling/summary.json"
)


def _valid_report() -> dict[str, object]:
    frame_counts = [4, 8, 16, 32]
    fixed_interval_entries = [100, 120, 150, 170]
    fixed_dense_samples = [100, 220, 500, 900]
    per_frame_compile_ms = [30.0, 70.0, 160.0, 360.0]
    rows: list[dict[str, object]] = []
    for idx, frames in enumerate(frame_counts):
        fixed_cpu = 30.0 + 3.0 * idx
        rows.append(
            {
                "route": "fixed_chart",
                "frames": frames,
                "frames_per_segment": frames // 4,
                "temporal_chunk_count": 4,
                "segment_count": 8,
                "trace_count": 8,
                "cell_count": 64 + idx,
                "interval_trace_entries": fixed_interval_entries[idx],
                "dense_trace_samples": fixed_dense_samples[idx],
                "interval_to_dense_trace_sample_ratio": fixed_interval_entries[idx] / fixed_dense_samples[idx],
                "fallback_fraction": 0.0,
                "atlas_payload_bytes": 608,
                "project_ms": fixed_cpu * 0.25,
                "atlas_build_ms": fixed_cpu * 0.75,
                "cpu_compile_ms": fixed_cpu,
                "mps_atlas_build_ms": fixed_cpu * 0.5,
                "forward_ms": 20.0 - idx,
                "backward_ms": 25.0 - idx,
                "grad_coeff_abs_sum": 7.0 + idx,
                "grad_opacity_abs_sum": 8.0 + idx,
                "grad_color_abs_sum": 9.0 + idx,
                "grad_spatial_precision_uv_abs_sum": 10.0 + idx,
                "autograd_ma_grad_abs_sum": 1.0 + idx,
                "autograd_q_uvt_grad_abs_sum": 2.0 + idx,
                "autograd_q_uv_grad_abs_sum": 3.0 + idx,
                "autograd_q_temporal_grad_abs_sum": 4.0 + idx,
                "autograd_opacity_grad_abs_sum": 5.0 + idx,
                "autograd_color_grad_abs_sum": 6.0 + idx,
            }
        )
        payload_scale = frames // 4
        rows.append(
            {
                "route": "per_frame",
                "frames": frames,
                "frames_per_segment": 1,
                "temporal_chunk_count": frames,
                "segment_count": 2 * frames,
                "trace_count": 2 * frames,
                "cell_count": 64 * payload_scale,
                "interval_trace_entries": fixed_dense_samples[idx],
                "dense_trace_samples": fixed_dense_samples[idx],
                "interval_to_dense_trace_sample_ratio": 1.0,
                "fallback_fraction": 0.0,
                "atlas_payload_bytes": 608 * payload_scale,
                "project_ms": per_frame_compile_ms[idx] * 0.25,
                "atlas_build_ms": per_frame_compile_ms[idx] * 0.75,
                "cpu_compile_ms": per_frame_compile_ms[idx],
                "mps_atlas_build_ms": per_frame_compile_ms[idx] * 0.5,
                "forward_ms": 25.0 * payload_scale,
                "backward_ms": 30.0 * payload_scale,
                "grad_coeff_abs_sum": 17.0 + idx,
                "grad_opacity_abs_sum": 18.0 + idx,
                "grad_color_abs_sum": 19.0 + idx,
                "grad_spatial_precision_uv_abs_sum": 20.0 + idx,
            }
        )
    return {
        "benchmark": "star_uvt_revolving_orbit_fixed_chart_scaling",
        "frame_counts": frame_counts,
        "fixed_temporal_chunks": 4,
        "image_size": 32,
        "tile_size": 8,
        "tile_t": 2,
        "tile_capacity": 128,
        "iterations": 3,
        "warmup": 1,
        "rows": rows,
        "summary": summarize(rows),
    }


def _row(report: dict[str, object], route: str, frames: int) -> dict[str, object]:
    rows = report["rows"]
    assert isinstance(rows, list)
    for raw_row in rows:
        assert isinstance(raw_row, dict)
        if raw_row["route"] == route and raw_row["frames"] == frames:
            return raw_row
    raise AssertionError(f"missing {route} row for frames={frames}")


def test_orbit_fixed_chart_report_verifier_accepts_valid_payload() -> None:
    report = _valid_report()

    assert verify_orbit_fixed_chart_scaling_report(report) == []
    assert_orbit_fixed_chart_scaling_report(report)


def test_orbit_fixed_chart_report_verifier_rejects_growing_fixed_trace_count() -> None:
    report = _valid_report()
    _row(report, "fixed_chart", 32)["trace_count"] = 16
    report["summary"] = summarize(report["rows"])

    errors = verify_orbit_fixed_chart_scaling_report(report)

    assert any("fixed_chart trace counts must stay constant" in error for error in errors)


def test_orbit_fixed_chart_report_verifier_rejects_fallback() -> None:
    report = _valid_report()
    _row(report, "fixed_chart", 16)["fallback_fraction"] = 0.125
    report["summary"] = summarize(report["rows"])

    errors = verify_orbit_fixed_chart_scaling_report(report)

    assert any("must be fallback-free" in error for error in errors)


def test_orbit_fixed_chart_report_verifier_rejects_missing_orbit_metric_gradient() -> None:
    report = _valid_report()
    _row(report, "fixed_chart", 32)["autograd_q_uv_grad_abs_sum"] = 0.0
    report["summary"] = summarize(report["rows"])

    errors = verify_orbit_fixed_chart_scaling_report(report)

    assert any("autograd_q_uv_grad_abs_sum" in error for error in errors)


def test_orbit_fixed_chart_report_verifier_rejects_missing_direct_metal_gradient() -> None:
    report = _valid_report()
    _row(report, "per_frame", 16)["grad_opacity_abs_sum"] = 0.0
    report["summary"] = summarize(report["rows"])

    errors = verify_orbit_fixed_chart_scaling_report(report)

    assert any("grad_opacity_abs_sum" in error for error in errors)


def test_orbit_fixed_chart_report_verifier_rejects_inconsistent_interval_ratio() -> None:
    report = _valid_report()
    _row(report, "fixed_chart", 16)["interval_to_dense_trace_sample_ratio"] = 0.99
    report["summary"] = summarize(report["rows"])

    errors = verify_orbit_fixed_chart_scaling_report(report)

    assert any("interval ratio" in error for error in errors)


def test_orbit_fixed_chart_report_verifier_rejects_inconsistent_cpu_compile_sum() -> None:
    report = _valid_report()
    _row(report, "fixed_chart", 16)["cpu_compile_ms"] = 999.0
    report["summary"] = summarize(report["rows"])

    errors = verify_orbit_fixed_chart_scaling_report(report)

    assert any("cpu_compile_ms" in error for error in errors)


def test_orbit_fixed_chart_report_verifier_rejects_stale_summary() -> None:
    report = _valid_report()
    summary = report["summary"]
    assert isinstance(summary, dict)
    summary["last_fixed_vs_per_frame_forward_ms_ratio"] = 0.0

    errors = verify_orbit_fixed_chart_scaling_report(report)

    assert any("summary last_fixed_vs_per_frame_forward_ms_ratio mismatch" in error for error in errors)


def test_orbit_fixed_chart_report_verifier_rejects_lost_gpu_timing_win() -> None:
    report = _valid_report()
    _row(report, "fixed_chart", 32)["forward_ms"] = 300.0
    report["summary"] = summarize(report["rows"])

    errors = verify_orbit_fixed_chart_scaling_report(report)

    assert any("forward timing ratio" in error for error in errors)


def test_saved_orbit_fixed_chart_artifact_satisfies_contract() -> None:
    if not SAVED_ORBIT_FIXED_CHART_ARTIFACT.exists():
        pytest.skip(f"missing optional saved artifact: {SAVED_ORBIT_FIXED_CHART_ARTIFACT}")

    report = json.loads(SAVED_ORBIT_FIXED_CHART_ARTIFACT.read_text(encoding="utf-8"))

    assert_orbit_fixed_chart_scaling_report(report)
