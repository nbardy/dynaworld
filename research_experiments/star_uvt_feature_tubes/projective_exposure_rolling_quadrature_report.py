from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
STAR_UVT_ROOT = ROOT / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(STAR_UVT_ROOT) not in sys.path:
    sys.path.insert(0, str(STAR_UVT_ROOT))

from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    ProjectiveTraceCellTraceAtlas,
    UVTRenderConfig,
    has_projective_trace_cell_interval_metal,
    has_projective_trace_cell_interval_rows_metal,
    lower_projective_trace_cell_atlas_quadrature,
    lower_projective_trace_cell_atlas_rolling_quadrature,
    mark_projective_trace_cell_visibility_fallbacks,
    projective_trace_cell_atlas_complexity_stats,
    projective_trace_cell_atlas_fallback_stats,
    projective_trace_cell_sensor_time_event_partition,
    projective_trace_cell_sensor_time_partition_quadrature,
    projective_trace_cell_sensor_time_partition_rolling_quadrature,
    render_projective_trace_cell_atlas_quadrature_interval_metal,
    render_projective_trace_cell_atlas_quadrature_interval_mixed_metal,
    render_projective_trace_cell_atlas_quadrature_reference,
    render_projective_trace_cell_atlas_reference,
    render_projective_trace_cell_atlas_rolling_quadrature_batched_reference,
    render_projective_trace_cell_atlas_rolling_quadrature_interval_metal,
    render_projective_trace_cell_atlas_rolling_quadrature_interval_mixed_metal,
    render_projective_trace_cell_atlas_rolling_quadrature_reference,
)


DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_exposure_rolling_quadrature"
)


def _direct_continuous_cell_atlas(
    *,
    coeffs: torch.Tensor | None = None,
    colors: torch.Tensor | None = None,
    opacities: torch.Tensor | None = None,
    depth_affine_uv: torch.Tensor | None = None,
) -> ProjectiveTraceCellTraceAtlas:
    if coeffs is None:
        coeffs = torch.tensor(
            [
                [3.5, 0.25, 0.0, 3.5, 0.08, 0.0, 1.0, 0.10, 0.0],
                [4.6, -0.18, 0.0, 3.2, 0.12, 0.0, 1.8, -0.06, 0.0],
            ],
            dtype=torch.float32,
        ).contiguous()
    if colors is None:
        colors = torch.tensor([[1.0, 0.1, 0.05], [0.05, 0.25, 1.0]], dtype=torch.float32)
    if opacities is None:
        opacities = torch.tensor([0.65, 0.45], dtype=torch.float32)
    trace_count = int(coeffs.shape[0])
    return ProjectiveTraceCellTraceAtlas(
        coeffs=coeffs,
        opacity=opacities,
        color=colors,
        cells=[],
        source_window_indices=tuple(0 for _ in range(trace_count)),
        source_primitive_ids=tuple(range(trace_count)),
        active_start=tuple(0 for _ in range(trace_count)),
        active_stop=tuple(1 for _ in range(trace_count)),
        depth_affine_uv=depth_affine_uv,
    )


def _mixed_fallback_cell_atlas() -> ProjectiveTraceCellTraceAtlas:
    coeffs = torch.tensor(
        [
            [3.4, 0.0, 0.0, 3.5, 0.0, 0.0, 1.0, 0.0, 0.0],
            [4.4, 0.0, 0.0, 3.5, 0.0, 0.0, 1.0, 0.0, 0.0],
            [12.0, 0.0, 0.0, 3.5, 0.0, 0.0, 2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).contiguous()
    colors = torch.tensor(
        [[1.0, 0.1, 0.05], [0.05, 0.2, 1.0], [0.1, 1.0, 0.2]],
        dtype=torch.float32,
    )
    opacities = torch.tensor([0.65, 0.45, 0.55], dtype=torch.float32)
    trace_count = int(coeffs.shape[0])
    return ProjectiveTraceCellTraceAtlas(
        coeffs=coeffs,
        opacity=opacities,
        color=colors,
        cells=[],
        source_window_indices=tuple(0 for _ in range(trace_count)),
        source_primitive_ids=tuple(range(trace_count)),
        active_start=tuple(0 for _ in range(trace_count)),
        active_stop=tuple(1 for _ in range(trace_count)),
    )


def _config(*, width: int, height: int, frames: int = 1, tile_size: int = 8) -> UVTRenderConfig:
    return UVTRenderConfig(
        height=height,
        width=width,
        frames=frames,
        tile_x=tile_size,
        tile_y=tile_size,
        tile_t=1,
        tile_capacity=128,
        alpha_threshold=1.0e-6,
        transmittance_threshold=0.0,
        max_alpha=1.0,
    )


def _atlas_to_mps(atlas: ProjectiveTraceCellTraceAtlas) -> ProjectiveTraceCellTraceAtlas:
    return ProjectiveTraceCellTraceAtlas(
        coeffs=atlas.coeffs.to("mps"),
        opacity=atlas.opacity.to("mps"),
        color=atlas.color.to("mps"),
        cells=atlas.cells,
        source_window_indices=atlas.source_window_indices,
        source_primitive_ids=atlas.source_primitive_ids,
        active_start=atlas.active_start,
        active_stop=atlas.active_stop,
        depth_affine_uv=None if atlas.depth_affine_uv is None else atlas.depth_affine_uv.to("mps"),
        opacity_time_coeffs=None if atlas.opacity_time_coeffs is None else atlas.opacity_time_coeffs.to("mps"),
        spatial_precision_uv=None if atlas.spatial_precision_uv is None else atlas.spatial_precision_uv.to("mps"),
    )


def _complexity_row(atlas: ProjectiveTraceCellTraceAtlas) -> dict[str, Any]:
    stats = projective_trace_cell_atlas_complexity_stats(atlas)
    return {
        "total_cells": int(stats.total_cells),
        "tile_active_set_groups": int(stats.tile_active_set_groups),
        "visibility_stratum_split_cells": int(stats.visibility_stratum_split_cells),
        "max_cells_per_active_set_group": int(stats.max_cells_per_active_set_group),
        "interval_trace_entries": int(stats.interval_trace_entries),
        "dense_trace_samples": int(stats.dense_trace_samples),
        "interval_to_dense_trace_sample_ratio": float(stats.interval_to_dense_trace_sample_ratio),
        "fallback_cells": int(stats.fallback_cells),
        "fallback_fraction": float(stats.fallback_fraction),
    }


def _fallback_row(atlas: ProjectiveTraceCellTraceAtlas) -> dict[str, Any]:
    stats = projective_trace_cell_atlas_fallback_stats(atlas)
    return {
        "total_cells": int(stats.total_cells),
        "fallback_cells": int(stats.fallback_cells),
        "total_tile_samples": int(stats.total_tile_samples),
        "fallback_tile_samples": int(stats.fallback_tile_samples),
        "total_trace_samples": int(stats.total_trace_samples),
        "fallback_trace_samples": int(stats.fallback_trace_samples),
        "fallback_fraction": float(stats.fallback_fraction),
        "fallback_reasons": list(stats.fallback_reasons),
    }


def _finite_exposure_case(*, run_metal: bool) -> dict[str, Any]:
    atlas = _direct_continuous_cell_atlas()
    width = 8
    height = 8
    tile_size = 8
    sigma_px = 1.7
    domain_times = torch.arange(4, dtype=torch.float32).contiguous()
    partition = projective_trace_cell_sensor_time_event_partition(
        atlas,
        domain_times,
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        include_support=False,
        include_visibility=False,
        extra_split_times=[1.0, 2.0],
    )
    quadrature = projective_trace_cell_sensor_time_partition_quadrature(
        partition,
        exposure_start=0.25,
        exposure_stop=2.75,
        samples_per_interval=2,
    )
    lowering = lower_projective_trace_cell_atlas_quadrature(
        atlas,
        quadrature,
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        uv_padding=4.0,
    )
    reference = render_projective_trace_cell_atlas_quadrature_reference(
        atlas,
        quadrature,
        image_width=width,
        image_height=height,
        sigma_px=sigma_px,
    )
    interval_samples = render_projective_trace_cell_atlas_reference(
        lowering.atlas,
        lowering.times,
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        sigma_px=sigma_px,
    )
    lowered = (interval_samples * lowering.weights.reshape(-1, 1, 1, 1)).sum(dim=0)
    metal_max_abs_error: float | None = None
    if run_metal and torch.backends.mps.is_available() and has_projective_trace_cell_interval_metal():
        metal = render_projective_trace_cell_atlas_quadrature_interval_metal(
            _atlas_to_mps(atlas),
            quadrature,
            _config(width=width, height=height, tile_size=tile_size),
            sigma_px=sigma_px,
            uv_padding=4.0,
        )
        metal_max_abs_error = float((metal.cpu() - reference).abs().max().item())
    return {
        "quadrature_sample_count": len(quadrature.samples),
        "partition_interval_count": len(partition.intervals),
        "lowered_unique_sample_count": int(lowering.times.numel()),
        "source_trace_count": int(atlas.coeffs.shape[0]),
        "lowered_trace_count": int(lowering.atlas.coeffs.shape[0]),
        "source_trace_indices": list(lowering.source_trace_indices),
        "weight_sum": float(lowering.weights.sum().item()),
        "reference_lowered_max_abs_error": float((lowered - reference).abs().max().item()),
        "metal_max_abs_error": metal_max_abs_error,
        "complexity": _complexity_row(lowering.atlas),
    }


def _rolling_shutter_case(*, run_metal: bool) -> dict[str, Any]:
    atlas = _direct_continuous_cell_atlas()
    width = 8
    height = 3
    tile_size = 8
    sigma_px = 1.7
    domain_times = torch.arange(4, dtype=torch.float32).contiguous()
    partition = projective_trace_cell_sensor_time_event_partition(
        atlas,
        domain_times,
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        include_support=False,
        include_visibility=False,
        extra_split_times=[1.0, 2.0],
    )
    row_quadrature = projective_trace_cell_sensor_time_partition_rolling_quadrature(
        partition,
        row_count=height,
        frame_time=0.0,
        exposure_duration=0.75,
        readout_duration=1.0,
        samples_per_interval=2,
    )
    lowering = lower_projective_trace_cell_atlas_rolling_quadrature(
        atlas,
        row_quadrature,
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        uv_padding=4.0,
    )
    rowwise = render_projective_trace_cell_atlas_rolling_quadrature_reference(
        atlas,
        row_quadrature,
        image_width=width,
        image_height=height,
        sigma_px=sigma_px,
    )
    batched = render_projective_trace_cell_atlas_rolling_quadrature_batched_reference(
        atlas,
        row_quadrature,
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        sigma_px=sigma_px,
        uv_padding=4.0,
    )
    total_row_samples = sum(len(quadrature.samples) for quadrature in row_quadrature)
    metal_max_abs_error: float | None = None
    if run_metal and torch.backends.mps.is_available() and has_projective_trace_cell_interval_rows_metal():
        metal = render_projective_trace_cell_atlas_rolling_quadrature_interval_metal(
            _atlas_to_mps(atlas),
            row_quadrature,
            _config(width=width, height=height, tile_size=tile_size),
            sigma_px=sigma_px,
            uv_padding=4.0,
        )
        metal_max_abs_error = float((metal.cpu() - rowwise).abs().max().item())
    return {
        "row_count": height,
        "total_row_sample_count": int(total_row_samples),
        "lowered_unique_sample_count": int(lowering.times.numel()),
        "unique_to_row_sample_ratio": float(lowering.times.numel()) / float(max(total_row_samples, 1)),
        "row_weight_sums": [float(value) for value in lowering.row_weights.sum(dim=0).detach().cpu().tolist()],
        "rowwise_batched_max_abs_error": float((batched - rowwise).abs().max().item()),
        "metal_max_abs_error": metal_max_abs_error,
        "complexity": _complexity_row(lowering.atlas),
    }


def _finite_fallback_case(*, run_metal: bool) -> dict[str, Any]:
    atlas = _mixed_fallback_cell_atlas()
    width = 16
    height = 8
    tile_size = 8
    sigma_px = 1.7
    domain_times = torch.arange(4, dtype=torch.float32).contiguous()
    partition = projective_trace_cell_sensor_time_event_partition(
        atlas,
        domain_times,
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        include_support=False,
        include_visibility=False,
        extra_split_times=[1.0, 2.0],
    )
    quadrature = projective_trace_cell_sensor_time_partition_quadrature(
        partition,
        exposure_start=0.25,
        exposure_stop=2.75,
        samples_per_interval=2,
    )
    lowering = lower_projective_trace_cell_atlas_quadrature(
        atlas,
        quadrature,
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        uv_padding=2.0,
    )
    lowering = replace(
        lowering,
        atlas=mark_projective_trace_cell_visibility_fallbacks(
            lowering.atlas,
            lowering.times,
            depth_epsilon=1.0e-6,
        ),
    )
    interval_ref = render_projective_trace_cell_atlas_reference(
        lowering.atlas,
        lowering.times,
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        sigma_px=sigma_px,
        alpha_cutoff=1.0e-6,
        transmittance_cutoff=0.0,
        allow_fallback_cells=True,
    )
    expected = (interval_ref * lowering.weights.reshape(-1, 1, 1, 1)).sum(dim=0)
    metal_max_abs_error: float | None = None
    if run_metal and torch.backends.mps.is_available() and has_projective_trace_cell_interval_metal():
        mixed = render_projective_trace_cell_atlas_quadrature_interval_mixed_metal(
            _atlas_to_mps(atlas),
            quadrature,
            _config(width=width, height=height, tile_size=tile_size),
            sigma_px=sigma_px,
            uv_padding=2.0,
            depth_epsilon=1.0e-6,
        )
        metal_max_abs_error = float((mixed.cpu() - expected).abs().max().item())
    return {
        "lowered_unique_sample_count": int(lowering.times.numel()),
        "weight_sum": float(lowering.weights.sum().item()),
        "expected_l1": float(expected.abs().sum().item()),
        "metal_max_abs_error": metal_max_abs_error,
        "fallback": _fallback_row(lowering.atlas),
        "complexity": _complexity_row(lowering.atlas),
    }


def _rolling_fallback_case(*, run_metal: bool) -> dict[str, Any]:
    atlas = _mixed_fallback_cell_atlas()
    width = 16
    height = 4
    tile_size = 8
    sigma_px = 1.7
    domain_times = torch.arange(4, dtype=torch.float32).contiguous()
    partition = projective_trace_cell_sensor_time_event_partition(
        atlas,
        domain_times,
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        include_support=False,
        include_visibility=False,
        extra_split_times=[1.0, 2.0],
    )
    row_quadrature = projective_trace_cell_sensor_time_partition_rolling_quadrature(
        partition,
        row_count=height,
        frame_time=0.0,
        exposure_duration=0.75,
        readout_duration=1.0,
        samples_per_interval=2,
    )
    lowering = lower_projective_trace_cell_atlas_rolling_quadrature(
        atlas,
        row_quadrature,
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        uv_padding=2.0,
    )
    lowering = replace(
        lowering,
        atlas=mark_projective_trace_cell_visibility_fallbacks(
            lowering.atlas,
            lowering.times,
            depth_epsilon=1.0e-6,
        ),
    )
    interval_ref = render_projective_trace_cell_atlas_reference(
        lowering.atlas,
        lowering.times,
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        sigma_px=sigma_px,
        alpha_cutoff=1.0e-6,
        transmittance_cutoff=0.0,
        allow_fallback_cells=True,
    )
    expected = (interval_ref * lowering.row_weights.reshape(-1, height, 1, 1)).sum(dim=0)
    metal_max_abs_error: float | None = None
    if run_metal and torch.backends.mps.is_available() and has_projective_trace_cell_interval_metal():
        mixed = render_projective_trace_cell_atlas_rolling_quadrature_interval_mixed_metal(
            _atlas_to_mps(atlas),
            row_quadrature,
            _config(width=width, height=height, tile_size=tile_size),
            sigma_px=sigma_px,
            uv_padding=2.0,
            depth_epsilon=1.0e-6,
        )
        metal_max_abs_error = float((mixed.cpu() - expected).abs().max().item())
    return {
        "lowered_unique_sample_count": int(lowering.times.numel()),
        "row_weight_sums": [float(value) for value in lowering.row_weights.sum(dim=0).detach().cpu().tolist()],
        "expected_l1": float(expected.abs().sum().item()),
        "metal_max_abs_error": metal_max_abs_error,
        "fallback": _fallback_row(lowering.atlas),
        "complexity": _complexity_row(lowering.atlas),
    }


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    finite = report["finite_exposure"]
    rolling = report["rolling_shutter"]
    finite_fallback = report["finite_fallback"]
    rolling_fallback = report["rolling_fallback"]
    metal_errors = [
        value
        for value in (
            finite.get("metal_max_abs_error"),
            rolling.get("metal_max_abs_error"),
            finite_fallback.get("metal_max_abs_error"),
            rolling_fallback.get("metal_max_abs_error"),
        )
        if value is not None
    ]
    return {
        "finite_reference_lowered_max_abs_error": finite["reference_lowered_max_abs_error"],
        "rolling_rowwise_batched_max_abs_error": rolling["rowwise_batched_max_abs_error"],
        "rolling_unique_to_row_sample_ratio": rolling["unique_to_row_sample_ratio"],
        "finite_fallback_fraction": finite_fallback["fallback"]["fallback_fraction"],
        "rolling_fallback_fraction": rolling_fallback["fallback"]["fallback_fraction"],
        "max_metal_abs_error": max(metal_errors) if metal_errors else None,
        "metal_case_count": len(metal_errors),
    }


def run_report(*, run_metal: bool | None = None) -> dict[str, Any]:
    should_run_metal = bool(torch.backends.mps.is_available()) if run_metal is None else bool(run_metal)
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_exposure_rolling_quadrature",
        "theory_contract": (
            "Finite exposure and rolling shutter integrate the rendered sensor-time field I(u,v,tau). "
            "Quadrature samples lower into one shared interval atlas; rolling shutter stores row weights over "
            "deduplicated sample times; visibility-ambiguous tile samples patch through live-depth fallback."
        ),
        "device": {
            "mps_available": bool(torch.backends.mps.is_available()),
            "interval_metal_available": bool(has_projective_trace_cell_interval_metal()),
            "row_interval_metal_available": bool(has_projective_trace_cell_interval_rows_metal()),
            "requested_metal": should_run_metal,
        },
        "finite_exposure": _finite_exposure_case(run_metal=should_run_metal),
        "rolling_shutter": _rolling_shutter_case(run_metal=should_run_metal),
        "finite_fallback": _finite_fallback_case(run_metal=should_run_metal),
        "rolling_fallback": _rolling_fallback_case(run_metal=should_run_metal),
    }
    report["summary"] = summarize(report)
    errors = verify_exposure_rolling_quadrature_report(report)
    if errors:
        report["status"] = "failed"
        report["errors"] = errors
    return report


def _finite_float(value: Any, label: str, errors: list[str]) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        errors.append(f"{label} must be finite, got {value!r}")
        return 0.0
    return float(value)


def _finite_int(value: Any, label: str, errors: list[str]) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(f"{label} must be a nonnegative integer, got {value!r}")
        return 0
    return int(value)


def _require_close(value: Any, expected: float, label: str, errors: list[str], *, atol: float) -> None:
    actual = _finite_float(value, label, errors)
    if abs(actual - float(expected)) > float(atol):
        errors.append(f"{label} must be within {atol:g} of {expected:g}, got {actual:g}")


def _verify_complexity(row: Any, label: str, errors: list[str]) -> None:
    if not isinstance(row, dict):
        errors.append(f"{label} complexity must be an object")
        return
    total_cells = _finite_int(row.get("total_cells"), f"{label} total_cells", errors)
    _finite_int(row.get("tile_active_set_groups"), f"{label} tile_active_set_groups", errors)
    _finite_int(row.get("visibility_stratum_split_cells"), f"{label} visibility_stratum_split_cells", errors)
    _finite_int(row.get("max_cells_per_active_set_group"), f"{label} max_cells_per_active_set_group", errors)
    entries = _finite_int(row.get("interval_trace_entries"), f"{label} interval_trace_entries", errors)
    dense = _finite_int(row.get("dense_trace_samples"), f"{label} dense_trace_samples", errors)
    fallback_cells = _finite_int(row.get("fallback_cells"), f"{label} fallback_cells", errors)
    fallback_fraction = _finite_float(row.get("fallback_fraction"), f"{label} fallback_fraction", errors)
    ratio = _finite_float(row.get("interval_to_dense_trace_sample_ratio"), f"{label} interval ratio", errors)
    if total_cells <= 0:
        errors.append(f"{label} must contain compiled cells")
    if entries <= 0 or dense <= 0:
        errors.append(f"{label} must contain interval and dense trace samples")
    if entries > dense:
        errors.append(f"{label} interval_trace_entries must not exceed dense_trace_samples")
    if ratio < 0.0 or ratio > 1.0:
        errors.append(f"{label} interval_to_dense_trace_sample_ratio must be in [0,1], got {ratio}")
    elif dense > 0:
        _require_close(
            ratio,
            float(entries) / float(dense),
            f"{label} interval_to_dense_trace_sample_ratio",
            errors,
            atol=1.0e-12,
        )
    if fallback_cells > total_cells:
        errors.append(f"{label} fallback_cells must not exceed total_cells")
    elif total_cells > 0:
        _require_close(
            fallback_fraction,
            float(fallback_cells) / float(total_cells),
            f"{label} fallback_fraction",
            errors,
            atol=1.0e-12,
        )


def _verify_fallback(row: Any, label: str, errors: list[str]) -> None:
    if not isinstance(row, dict):
        errors.append(f"{label} fallback must be an object")
        return
    total_cells = _finite_int(row.get("total_cells"), f"{label} total_cells", errors)
    fallback_cells = _finite_int(row.get("fallback_cells"), f"{label} fallback_cells", errors)
    total_tile_samples = _finite_int(row.get("total_tile_samples"), f"{label} total_tile_samples", errors)
    fallback_tile_samples = _finite_int(row.get("fallback_tile_samples"), f"{label} fallback_tile_samples", errors)
    total_trace_samples = _finite_int(row.get("total_trace_samples"), f"{label} total_trace_samples", errors)
    fallback_trace_samples = _finite_int(row.get("fallback_trace_samples"), f"{label} fallback_trace_samples", errors)
    fraction = _finite_float(row.get("fallback_fraction"), f"{label} fallback_fraction", errors)
    if fallback_cells <= 0:
        errors.append(f"{label} must contain fallback cells")
    if total_cells <= fallback_cells:
        errors.append(f"{label} must leave some cells on the fast path")
    if not 0.0 < fraction < 1.0:
        errors.append(f"{label} fallback_fraction must be strictly between 0 and 1, got {fraction}")
    elif total_cells > 0:
        _require_close(
            fraction,
            float(fallback_cells) / float(total_cells),
            f"{label} fallback_fraction",
            errors,
            atol=1.0e-12,
        )
    if not 0 < fallback_tile_samples < total_tile_samples:
        errors.append(f"{label} fallback_tile_samples must be a strict subset of total_tile_samples")
    if not 0 < fallback_trace_samples < total_trace_samples:
        errors.append(f"{label} fallback_trace_samples must be a strict subset of total_trace_samples")
    if fallback_trace_samples < fallback_tile_samples:
        errors.append(f"{label} fallback_trace_samples must cover fallback_tile_samples")
    if total_trace_samples < total_tile_samples:
        errors.append(f"{label} total_trace_samples must cover total_tile_samples")
    reasons = row.get("fallback_reasons")
    if "visibility_ambiguous_depth" not in reasons if isinstance(reasons, list) else True:
        errors.append(f"{label} must include visibility_ambiguous_depth fallback reason")
    if isinstance(reasons, list) and not all(isinstance(reason, str) and reason for reason in reasons):
        errors.append(f"{label} fallback_reasons must be non-empty strings")


def _verify_optional_metal_error(row: dict[str, Any], label: str, errors: list[str]) -> None:
    value = row.get("metal_max_abs_error")
    if value is None:
        return
    error = _finite_float(value, f"{label} metal_max_abs_error", errors)
    if error > 3.0e-4:
        errors.append(f"{label} metal_max_abs_error must be <= 3e-4, got {error}")


def _require_zero_fallback_complexity(row: dict[str, Any], label: str, errors: list[str]) -> None:
    if row.get("fallback_cells") != 0:
        errors.append(f"{label} complexity must have zero fallback cells")
    if row.get("fallback_fraction") != 0.0:
        errors.append(f"{label} complexity must have zero fallback fraction")


def _verify_fallback_complexity_match(case: dict[str, Any], label: str, errors: list[str]) -> None:
    complexity = case.get("complexity")
    fallback = case.get("fallback")
    if not isinstance(complexity, dict) or not isinstance(fallback, dict):
        return
    for key in ("total_cells", "fallback_cells", "fallback_fraction"):
        if complexity.get(key) != fallback.get(key):
            errors.append(f"{label} complexity {key} must match fallback stats")


def _verify_summary(report: dict[str, Any], summary: dict[str, Any], errors: list[str]) -> None:
    expected = summarize(report)
    for key, expected_value in expected.items():
        actual_value = summary.get(key)
        if expected_value is None:
            if actual_value is not None:
                errors.append(f"summary {key} must be None, got {actual_value!r}")
            continue
        _require_close(actual_value, float(expected_value), f"summary {key}", errors, atol=1.0e-12)


def verify_exposure_rolling_quadrature_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_exposure_rolling_quadrature":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    if not isinstance(report.get("theory_contract"), str) or "Finite exposure" not in report.get("theory_contract", ""):
        errors.append("theory_contract must state the finite-exposure sensor-time contract")
    if not isinstance(report.get("device"), dict):
        errors.append("device must be an object")
    for key in ("finite_exposure", "rolling_shutter", "finite_fallback", "rolling_fallback", "summary"):
        if not isinstance(report.get(key), dict):
            errors.append(f"{key} must be an object")
    if errors:
        return errors

    finite = report["finite_exposure"]
    rolling = report["rolling_shutter"]
    finite_fallback = report["finite_fallback"]
    rolling_fallback = report["rolling_fallback"]
    summary = report["summary"]
    device = report["device"]

    for key in ("mps_available", "interval_metal_available", "row_interval_metal_available", "requested_metal"):
        if not isinstance(device.get(key), bool):
            errors.append(f"device {key} must be boolean")

    finite_samples = _finite_int(finite.get("quadrature_sample_count"), "finite quadrature_sample_count", errors)
    finite_unique_samples = _finite_int(
        finite.get("lowered_unique_sample_count"),
        "finite lowered_unique_sample_count",
        errors,
    )
    finite_source_trace_count = _finite_int(finite.get("source_trace_count"), "finite source_trace_count", errors)
    finite_lowered_trace_count = _finite_int(finite.get("lowered_trace_count"), "finite lowered_trace_count", errors)
    if finite_samples <= 0:
        errors.append("finite quadrature_sample_count must be positive")
    if finite_samples != finite_unique_samples:
        errors.append("finite lowering must preserve one interval sample per exposure quadrature sample")
    if finite_source_trace_count != finite_lowered_trace_count:
        errors.append("finite lowering must preserve active source trace count")
    source_trace_indices = finite.get("source_trace_indices")
    if not isinstance(source_trace_indices, list) or source_trace_indices != list(range(finite_source_trace_count)):
        errors.append("finite source_trace_indices must preserve source trace order")
    _require_close(finite.get("weight_sum"), 1.0, "finite weight_sum", errors, atol=1.0e-6)
    finite_error = _finite_float(
        finite.get("reference_lowered_max_abs_error"),
        "finite reference_lowered_max_abs_error",
        errors,
    )
    if finite_error > 1.0e-6:
        errors.append(f"finite reference_lowered_max_abs_error must be <= 1e-6, got {finite_error}")
    _verify_complexity(finite.get("complexity"), "finite", errors)
    if isinstance(finite.get("complexity"), dict):
        _require_zero_fallback_complexity(finite["complexity"], "finite", errors)
    _verify_optional_metal_error(finite, "finite", errors)

    rolling_row_count = _finite_int(rolling.get("row_count"), "rolling row_count", errors)
    total_row_samples = _finite_int(rolling.get("total_row_sample_count"), "rolling total_row_sample_count", errors)
    unique_row_samples = _finite_int(
        rolling.get("lowered_unique_sample_count"), "rolling lowered_unique_sample_count", errors
    )
    if rolling_row_count <= 0:
        errors.append("rolling row_count must be positive")
    if not 0 < unique_row_samples < total_row_samples:
        errors.append("rolling lowering must deduplicate row quadrature samples")
    ratio = _finite_float(rolling.get("unique_to_row_sample_ratio"), "rolling unique_to_row_sample_ratio", errors)
    if not 0.0 < ratio < 1.0:
        errors.append(f"rolling unique_to_row_sample_ratio must be in (0,1), got {ratio}")
    elif total_row_samples > 0:
        _require_close(
            ratio,
            float(unique_row_samples) / float(total_row_samples),
            "rolling unique_to_row_sample_ratio",
            errors,
            atol=1.0e-12,
        )
    row_weight_sums = rolling.get("row_weight_sums")
    if not isinstance(row_weight_sums, list) or len(row_weight_sums) != rolling_row_count:
        errors.append("rolling row_weight_sums must contain one entry per row")
    else:
        for row_index, value in enumerate(row_weight_sums):
            _require_close(value, 1.0, f"rolling row {row_index} weight sum", errors, atol=1.0e-6)
    rolling_error = _finite_float(
        rolling.get("rowwise_batched_max_abs_error"),
        "rolling rowwise_batched_max_abs_error",
        errors,
    )
    if rolling_error > 1.0e-6:
        errors.append(f"rolling rowwise_batched_max_abs_error must be <= 1e-6, got {rolling_error}")
    _verify_complexity(rolling.get("complexity"), "rolling", errors)
    if isinstance(rolling.get("complexity"), dict):
        _require_zero_fallback_complexity(rolling["complexity"], "rolling", errors)
    _verify_optional_metal_error(rolling, "rolling", errors)

    if _finite_int(finite_fallback.get("lowered_unique_sample_count"), "finite fallback lowered_unique_sample_count", errors) <= 0:
        errors.append("finite fallback lowered_unique_sample_count must be positive")
    if _finite_float(finite_fallback.get("expected_l1"), "finite fallback expected_l1", errors) <= 0.0:
        errors.append("finite fallback expected_l1 must be positive")
    _require_close(finite_fallback.get("weight_sum"), 1.0, "finite fallback weight_sum", errors, atol=1.0e-6)
    _verify_fallback(finite_fallback.get("fallback"), "finite fallback", errors)
    _verify_complexity(finite_fallback.get("complexity"), "finite fallback", errors)
    _verify_fallback_complexity_match(finite_fallback, "finite fallback", errors)
    _verify_optional_metal_error(finite_fallback, "finite fallback", errors)
    if _finite_int(
        rolling_fallback.get("lowered_unique_sample_count"),
        "rolling fallback lowered_unique_sample_count",
        errors,
    ) <= 0:
        errors.append("rolling fallback lowered_unique_sample_count must be positive")
    if _finite_float(rolling_fallback.get("expected_l1"), "rolling fallback expected_l1", errors) <= 0.0:
        errors.append("rolling fallback expected_l1 must be positive")
    rolling_fallback_row_sums = rolling_fallback.get("row_weight_sums")
    if not isinstance(rolling_fallback_row_sums, list) or not rolling_fallback_row_sums:
        errors.append("rolling fallback row_weight_sums must be non-empty")
    else:
        for row_index, value in enumerate(rolling_fallback_row_sums):
            _require_close(value, 1.0, f"rolling fallback row {row_index} weight sum", errors, atol=1.0e-6)
    _verify_fallback(rolling_fallback.get("fallback"), "rolling fallback", errors)
    _verify_complexity(rolling_fallback.get("complexity"), "rolling fallback", errors)
    _verify_fallback_complexity_match(rolling_fallback, "rolling fallback", errors)
    _verify_optional_metal_error(rolling_fallback, "rolling fallback", errors)

    _verify_summary(report, summary, errors)
    return errors


def assert_exposure_rolling_quadrature_report(report: dict[str, Any]) -> None:
    errors = verify_exposure_rolling_quadrature_report(report)
    if errors:
        raise AssertionError("exposure/rolling quadrature report failed:\n- " + "\n- ".join(errors))


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# STAR UVT Projective Exposure And Rolling Quadrature",
        "",
        "This report pins the exposure/rolling contract for the sensor-time trace atlas:",
        "",
        "```text",
        "frame = integral_tau Composite(TraceAtlas(u,v,tau)) d tau",
        "```",
        "",
        "The report checks finite-exposure sample lowering, rolling-row time deduplication, and mixed Metal fallback patching when available.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(report["summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## Device",
        "",
        "```json",
        json.dumps(report["device"], indent=2, sort_keys=True),
        "```",
        "",
        "## Cases",
        "",
        "```json",
        json.dumps(
            {
                "finite_exposure": report["finite_exposure"],
                "rolling_shutter": report["rolling_shutter"],
                "finite_fallback": report["finite_fallback"],
                "rolling_fallback": report["rolling_fallback"],
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--verify-report", type=Path, default=None)
    parser.add_argument("--skip-metal", action="store_true")
    args = parser.parse_args()

    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_exposure_rolling_quadrature_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(run_metal=not args.skip_metal)
    assert_exposure_rolling_quadrature_report(report)
    write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
