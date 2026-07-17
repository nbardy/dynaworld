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
    has_projective_trace_cell_interval_backward_metal,
    lower_projective_trace_cell_atlas_quadrature,
    lower_projective_trace_cell_atlas_rolling_quadrature,
    mark_projective_trace_cell_visibility_fallbacks,
    projective_trace_cell_atlas_fallback_stats,
    projective_trace_cell_atlas_fallback_tile_sample_mask,
    projective_trace_cell_sensor_time_event_partition,
    projective_trace_cell_sensor_time_partition_quadrature,
    projective_trace_cell_sensor_time_partition_rolling_quadrature,
    render_projective_trace_cell_atlas_reference,
    split_projective_trace_cell_atlas_fallback_cells,
)
from research_project.trainer_harness import (  # noqa: E402
    render_projective_cell_interval_atlas_metal_backward,
)
from research_experiments.star_uvt_feature_tubes.projective_exposure_rolling_quadrature_report import (  # noqa: E402
    _config,
)


DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_exposure_rolling_mixed_fallback_backward"
)


def _mixed_atlas(*, device: torch.device | str = "cpu", requires_grad: bool = False) -> ProjectiveTraceCellTraceAtlas:
    coeffs = torch.tensor(
        [
            [3.4, 0.0, 0.0, 3.5, 0.0, 0.0, 1.0, 0.0, 0.0],
            [4.4, 0.0, 0.0, 3.5, 0.0, 0.0, 1.0, 0.0, 0.0],
            [12.0, 0.0, 0.0, 3.5, 0.0, 0.0, 2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    ).contiguous()
    opacity = torch.tensor([0.65, 0.45, 0.55], dtype=torch.float32, device=device)
    color = torch.tensor(
        [[1.0, 0.1, 0.05], [0.05, 0.2, 1.0], [0.1, 1.0, 0.2]],
        dtype=torch.float32,
        device=device,
    )
    if requires_grad:
        coeffs.requires_grad_(True)
        opacity.requires_grad_(True)
        color.requires_grad_(True)
    trace_count = int(coeffs.shape[0])
    return ProjectiveTraceCellTraceAtlas(
        coeffs=coeffs,
        opacity=opacity,
        color=color,
        cells=[],
        source_window_indices=tuple(0 for _ in range(trace_count)),
        source_primitive_ids=tuple(range(trace_count)),
        active_start=tuple(0 for _ in range(trace_count)),
        active_stop=tuple(1 for _ in range(trace_count)),
    )


def _grad_image(*, height: int, width: int) -> torch.Tensor:
    values = torch.linspace(-0.35, 0.42, steps=height * width * 3, dtype=torch.float32)
    return values.reshape(height, width, 3).contiguous()


def _compare_tensor(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    abs_error = float((actual - expected).abs().max().item())
    denom = expected.abs().max().clamp_min(1.0e-8)
    rel_error = float(((actual - expected).abs().max() / denom).item())
    return {
        "max_abs_error": abs_error,
        "max_rel_error": rel_error,
        "reference_norm": float(expected.norm().item()),
    }


def _compare_grads(actual_atlas: ProjectiveTraceCellTraceAtlas, expected_atlas: ProjectiveTraceCellTraceAtlas) -> dict[str, Any]:
    if actual_atlas.coeffs.grad is None or actual_atlas.opacity.grad is None or actual_atlas.color.grad is None:
        raise RuntimeError("actual gradients are missing")
    if expected_atlas.coeffs.grad is None or expected_atlas.opacity.grad is None or expected_atlas.color.grad is None:
        raise RuntimeError("reference gradients are missing")
    rows = {
        "coeffs": _compare_tensor(actual_atlas.coeffs.grad.detach().cpu(), expected_atlas.coeffs.grad.detach().cpu()),
        "opacity": _compare_tensor(actual_atlas.opacity.grad.detach().cpu(), expected_atlas.opacity.grad.detach().cpu()),
        "color": _compare_tensor(actual_atlas.color.grad.detach().cpu(), expected_atlas.color.grad.detach().cpu()),
    }
    return {
        **rows,
        "max_abs_error": max(row["max_abs_error"] for row in rows.values()),
        "max_rel_error": max(row["max_rel_error"] for row in rows.values()),
    }


def _patch_fallback_samples(
    fast_samples: torch.Tensor,
    fallback_samples: torch.Tensor,
    fallback_mask: torch.Tensor,
    *,
    tile_size: int,
) -> torch.Tensor:
    if fast_samples.shape != fallback_samples.shape:
        raise ValueError("fast_samples and fallback_samples must have the same shape")
    if fallback_mask.shape[0] != fast_samples.shape[0]:
        raise ValueError("fallback_mask sample count must match fast_samples")
    if not bool(torch.any(fallback_mask).item()):
        return fast_samples
    patched = fast_samples.clone()
    height = int(fast_samples.shape[1])
    width = int(fast_samples.shape[2])
    for sample_index, tile_v, tile_u in fallback_mask.detach().cpu().nonzero(as_tuple=False).tolist():
        v0 = int(tile_v) * int(tile_size)
        u0 = int(tile_u) * int(tile_size)
        v1 = min(height, v0 + int(tile_size))
        u1 = min(width, u0 + int(tile_size))
        patched[int(sample_index), v0:v1, u0:u1, :] = fallback_samples[int(sample_index), v0:v1, u0:u1, :]
    return patched


def _render_mixed_interval_samples_with_backward(
    atlas: ProjectiveTraceCellTraceAtlas,
    times: torch.Tensor,
    *,
    width: int,
    height: int,
    tile_size: int,
    sigma_px: float,
):
    config = _config(width=width, height=height, tile_size=tile_size)
    sample_config = replace(config, frames=int(times.numel()))
    fast_atlas, _fallback_atlas = split_projective_trace_cell_atlas_fallback_cells(atlas)
    if fast_atlas.cells:
        fast_samples = render_projective_cell_interval_atlas_metal_backward(
            fast_atlas,
            times,
            sample_config,
            sigma_px=sigma_px,
        )
    else:
        fast_samples = torch.zeros(
            (int(times.numel()), height, width, int(atlas.color.shape[1])),
            dtype=atlas.color.dtype,
            device=atlas.color.device,
        )
    reference_samples = render_projective_trace_cell_atlas_reference(
        atlas,
        times,
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        sigma_px=sigma_px,
        alpha_cutoff=float(config.alpha_threshold),
        transmittance_cutoff=float(config.transmittance_threshold),
        allow_fallback_cells=True,
        fallback_sort_live_depth=True,
    )
    fallback_mask = projective_trace_cell_atlas_fallback_tile_sample_mask(
        atlas,
        frames=int(times.numel()),
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        device=fast_samples.device,
    )
    return _patch_fallback_samples(
        fast_samples,
        reference_samples.to(device=fast_samples.device),
        fallback_mask,
        tile_size=tile_size,
    )


def _fallback_row(atlas: ProjectiveTraceCellTraceAtlas) -> dict[str, Any]:
    stats = projective_trace_cell_atlas_fallback_stats(atlas)
    return {
        "total_cells": int(stats.total_cells),
        "fallback_cells": int(stats.fallback_cells),
        "fallback_fraction": float(stats.fallback_fraction),
        "fallback_tile_samples": int(stats.fallback_tile_samples),
        "total_tile_samples": int(stats.total_tile_samples),
        "fallback_reasons": list(stats.fallback_reasons),
    }


def _finite_reference(
    atlas: ProjectiveTraceCellTraceAtlas,
    *,
    width: int,
    height: int,
    tile_size: int,
    sigma_px: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    partition = projective_trace_cell_sensor_time_event_partition(
        atlas,
        torch.arange(4, dtype=torch.float32, device=atlas.coeffs.device).contiguous(),
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
    samples = render_projective_trace_cell_atlas_reference(
        lowering.atlas,
        lowering.times,
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        sigma_px=sigma_px,
        alpha_cutoff=1.0e-6,
        transmittance_cutoff=0.0,
        allow_fallback_cells=True,
        fallback_sort_live_depth=True,
    )
    return (samples * lowering.weights.reshape(-1, 1, 1, 1)).sum(dim=0), {
        "fallback": _fallback_row(lowering.atlas),
        "lowered_unique_sample_count": int(lowering.times.numel()),
        "weight_sum": float(lowering.weights.sum().item()),
    }


def _rolling_reference(
    atlas: ProjectiveTraceCellTraceAtlas,
    *,
    width: int,
    height: int,
    tile_size: int,
    sigma_px: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    partition = projective_trace_cell_sensor_time_event_partition(
        atlas,
        torch.arange(4, dtype=torch.float32, device=atlas.coeffs.device).contiguous(),
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
        readout_duration=0.5,
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
    samples = render_projective_trace_cell_atlas_reference(
        lowering.atlas,
        lowering.times,
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        sigma_px=sigma_px,
        alpha_cutoff=1.0e-6,
        transmittance_cutoff=0.0,
        allow_fallback_cells=True,
        fallback_sort_live_depth=True,
    )
    total_row_samples = sum(len(quadrature.samples) for quadrature in row_quadrature)
    return (samples * lowering.row_weights.reshape(-1, height, 1, 1)).sum(dim=0), {
        "fallback": _fallback_row(lowering.atlas),
        "total_row_sample_count": int(total_row_samples),
        "lowered_unique_sample_count": int(lowering.times.numel()),
        "unique_to_row_sample_ratio": float(lowering.times.numel()) / float(max(total_row_samples, 1)),
        "row_weight_sums": [float(value) for value in lowering.row_weights.sum(dim=0).detach().cpu().tolist()],
    }


def _finite_case(*, run_metal: bool) -> dict[str, Any]:
    width = 16
    height = 8
    tile_size = 8
    sigma_px = 1.7
    reference_atlas = _mixed_atlas(requires_grad=True)
    reference, metadata = _finite_reference(
        reference_atlas,
        width=width,
        height=height,
        tile_size=tile_size,
        sigma_px=sigma_px,
    )
    final_adjoint = _grad_image(height=height, width=width)
    (reference * final_adjoint).sum().backward()
    row: dict[str, Any] = {
        **metadata,
        "reference_l1": float(reference.abs().sum().item()),
        "reference_grad_norms": {
            "coeffs": float(reference_atlas.coeffs.grad.norm().item()) if reference_atlas.coeffs.grad is not None else 0.0,
            "opacity": float(reference_atlas.opacity.grad.norm().item()) if reference_atlas.opacity.grad is not None else 0.0,
            "color": float(reference_atlas.color.grad.norm().item()) if reference_atlas.color.grad is not None else 0.0,
        },
        "mixed_compare": None,
    }
    if run_metal and torch.backends.mps.is_available() and has_projective_trace_cell_interval_backward_metal():
        mixed_atlas = _mixed_atlas(device="mps", requires_grad=True)
        partition = projective_trace_cell_sensor_time_event_partition(
            mixed_atlas,
            torch.arange(4, dtype=torch.float32, device="mps").contiguous(),
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
            mixed_atlas,
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
        mixed_samples = _render_mixed_interval_samples_with_backward(
            lowering.atlas,
            lowering.times,
            width=width,
            height=height,
            tile_size=tile_size,
            sigma_px=sigma_px,
        )
        mixed = (mixed_samples * lowering.weights.reshape(-1, 1, 1, 1)).sum(dim=0)
        (mixed * final_adjoint.to("mps")).sum().backward()
        row["mixed_output_max_abs_error"] = float((mixed.detach().cpu() - reference.detach()).abs().max().item())
        row["mixed_compare"] = _compare_grads(mixed_atlas, reference_atlas)
    return row


def _rolling_case(*, run_metal: bool) -> dict[str, Any]:
    width = 16
    height = 4
    tile_size = 8
    sigma_px = 1.7
    reference_atlas = _mixed_atlas(requires_grad=True)
    reference, metadata = _rolling_reference(
        reference_atlas,
        width=width,
        height=height,
        tile_size=tile_size,
        sigma_px=sigma_px,
    )
    final_adjoint = _grad_image(height=height, width=width)
    (reference * final_adjoint).sum().backward()
    row: dict[str, Any] = {
        **metadata,
        "reference_l1": float(reference.abs().sum().item()),
        "reference_grad_norms": {
            "coeffs": float(reference_atlas.coeffs.grad.norm().item()) if reference_atlas.coeffs.grad is not None else 0.0,
            "opacity": float(reference_atlas.opacity.grad.norm().item()) if reference_atlas.opacity.grad is not None else 0.0,
            "color": float(reference_atlas.color.grad.norm().item()) if reference_atlas.color.grad is not None else 0.0,
        },
        "mixed_compare": None,
    }
    if run_metal and torch.backends.mps.is_available() and has_projective_trace_cell_interval_backward_metal():
        mixed_atlas = _mixed_atlas(device="mps", requires_grad=True)
        partition = projective_trace_cell_sensor_time_event_partition(
            mixed_atlas,
            torch.arange(4, dtype=torch.float32, device="mps").contiguous(),
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
            readout_duration=0.5,
            samples_per_interval=2,
        )
        lowering = lower_projective_trace_cell_atlas_rolling_quadrature(
            mixed_atlas,
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
        mixed_samples = _render_mixed_interval_samples_with_backward(
            lowering.atlas,
            lowering.times,
            width=width,
            height=height,
            tile_size=tile_size,
            sigma_px=sigma_px,
        )
        mixed = (mixed_samples * lowering.row_weights.reshape(-1, height, 1, 1)).sum(dim=0)
        (mixed * final_adjoint.to("mps")).sum().backward()
        row["mixed_output_max_abs_error"] = float((mixed.detach().cpu() - reference.detach()).abs().max().item())
        row["mixed_compare"] = _compare_grads(mixed_atlas, reference_atlas)
    return row


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    compares = [
        row["mixed_compare"]
        for row in (report["finite_mixed_fallback_backward"], report["rolling_mixed_fallback_backward"])
        if row.get("mixed_compare") is not None
    ]
    return {
        "finite_has_mixed_backward": report["finite_mixed_fallback_backward"].get("mixed_compare") is not None,
        "rolling_has_mixed_backward": report["rolling_mixed_fallback_backward"].get("mixed_compare") is not None,
        "finite_fallback_fraction": report["finite_mixed_fallback_backward"]["fallback"]["fallback_fraction"],
        "rolling_fallback_fraction": report["rolling_mixed_fallback_backward"]["fallback"]["fallback_fraction"],
        "rolling_unique_to_row_sample_ratio": report["rolling_mixed_fallback_backward"]["unique_to_row_sample_ratio"],
        "max_mixed_output_abs_error": max(
            (
                float(row.get("mixed_output_max_abs_error", 0.0))
                for row in (report["finite_mixed_fallback_backward"], report["rolling_mixed_fallback_backward"])
                if row.get("mixed_compare") is not None
            ),
            default=None,
        ),
        "max_mixed_grad_abs_error": max((row["max_abs_error"] for row in compares), default=None),
        "max_mixed_grad_rel_error": max((row["max_rel_error"] for row in compares), default=None),
        "mixed_backward_case_count": len(compares),
    }


def run_report(*, run_metal: bool | None = None) -> dict[str, Any]:
    should_run_metal = bool(torch.backends.mps.is_available()) if run_metal is None else bool(run_metal)
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_exposure_rolling_mixed_fallback_backward",
        "theory_contract": (
            "Visibility-ambiguous finite/rolling fallback tile-samples must be differentiable: "
            "fast regions use interval Metal autograd, fallback regions use live-depth Torch reference gradients, "
            "and exposure/row weights accumulate the patched sample adjoints."
        ),
        "device": {
            "mps_available": bool(torch.backends.mps.is_available()),
            "interval_backward_metal_available": bool(has_projective_trace_cell_interval_backward_metal()),
            "requested_metal": should_run_metal,
        },
        "finite_mixed_fallback_backward": _finite_case(run_metal=should_run_metal),
        "rolling_mixed_fallback_backward": _rolling_case(run_metal=should_run_metal),
    }
    report["summary"] = summarize(report)
    errors = verify_mixed_fallback_backward_report(report)
    if errors:
        report["status"] = "failed"
        report["errors"] = errors
    return report


def _finite_float(value: Any, label: str, errors: list[str]) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        errors.append(f"{label} must be finite, got {value!r}")
        return 0.0
    return float(value)


def _verify_fallback(row: Any, label: str, errors: list[str]) -> None:
    if not isinstance(row, dict):
        errors.append(f"{label} fallback must be an object")
        return
    fallback_cells = int(_finite_float(row.get("fallback_cells"), f"{label} fallback_cells", errors))
    fraction = _finite_float(row.get("fallback_fraction"), f"{label} fallback_fraction", errors)
    if fallback_cells <= 0:
        errors.append(f"{label} fallback_cells must be positive")
    if not 0.0 < fraction < 1.0:
        errors.append(f"{label} fallback_fraction must be in (0,1), got {fraction}")
    reasons = row.get("fallback_reasons")
    if "visibility_ambiguous_depth" not in reasons if isinstance(reasons, list) else True:
        errors.append(f"{label} must include visibility_ambiguous_depth reason")


def _verify_grad_norms(row: Any, label: str, errors: list[str]) -> None:
    if not isinstance(row, dict):
        errors.append(f"{label} reference_grad_norms must be an object")
        return
    for key in ("coeffs", "opacity", "color"):
        value = _finite_float(row.get(key), f"{label} {key} reference grad norm", errors)
        if value <= 0.0:
            errors.append(f"{label} {key} reference grad norm must be positive")


def _verify_compare(row: Any, label: str, errors: list[str]) -> None:
    if row is None:
        return
    if not isinstance(row, dict):
        errors.append(f"{label} mixed_compare must be an object or null")
        return
    max_abs = _finite_float(row.get("max_abs_error"), f"{label} max_abs_error", errors)
    max_rel = _finite_float(row.get("max_rel_error"), f"{label} max_rel_error", errors)
    if max_abs > 2.0e-3:
        errors.append(f"{label} max_abs_error must be <= 2e-3, got {max_abs}")
    if max_rel > 1.0e-2:
        errors.append(f"{label} max_rel_error must be <= 1e-2, got {max_rel}")
    for key in ("coeffs", "opacity", "color"):
        subrow = row.get(key)
        if not isinstance(subrow, dict):
            errors.append(f"{label} {key} compare must be an object")
            continue
        norm = _finite_float(subrow.get("reference_norm"), f"{label} {key} reference_norm", errors)
        if norm <= 0.0:
            errors.append(f"{label} {key} reference_norm must be positive")


def verify_mixed_fallback_backward_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_exposure_rolling_mixed_fallback_backward":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    for key in ("finite_mixed_fallback_backward", "rolling_mixed_fallback_backward", "summary"):
        if not isinstance(report.get(key), dict):
            errors.append(f"{key} must be an object")
    if errors:
        return errors
    finite = report["finite_mixed_fallback_backward"]
    rolling = report["rolling_mixed_fallback_backward"]
    summary = report["summary"]
    _verify_fallback(finite.get("fallback"), "finite", errors)
    _verify_fallback(rolling.get("fallback"), "rolling", errors)
    _verify_grad_norms(finite.get("reference_grad_norms"), "finite", errors)
    _verify_grad_norms(rolling.get("reference_grad_norms"), "rolling", errors)
    weight_sum = _finite_float(finite.get("weight_sum"), "finite weight_sum", errors)
    if abs(weight_sum - 1.0) > 1.0e-6:
        errors.append(f"finite weight_sum must equal 1, got {weight_sum}")
    total_row_samples = int(_finite_float(rolling.get("total_row_sample_count"), "rolling total_row_sample_count", errors))
    unique_row_samples = int(_finite_float(rolling.get("lowered_unique_sample_count"), "rolling lowered_unique_sample_count", errors))
    if not 0 < unique_row_samples < total_row_samples:
        errors.append("rolling mixed fallback must reuse deduplicated row sample times")
    row_weight_sums = rolling.get("row_weight_sums")
    if not isinstance(row_weight_sums, list) or len(row_weight_sums) != 4:
        errors.append("rolling row_weight_sums must contain four rows")
    else:
        for row_index, value in enumerate(row_weight_sums):
            row_sum = _finite_float(value, f"rolling row {row_index} weight sum", errors)
            if abs(row_sum - 1.0) > 1.0e-6:
                errors.append(f"rolling row {row_index} weight sum must equal 1, got {row_sum}")
    _verify_compare(finite.get("mixed_compare"), "finite", errors)
    _verify_compare(rolling.get("mixed_compare"), "rolling", errors)
    for label, row in (("finite", finite), ("rolling", rolling)):
        if row.get("mixed_compare") is not None:
            output_error = _finite_float(row.get("mixed_output_max_abs_error"), f"{label} output error", errors)
            if output_error > 3.0e-4:
                errors.append(f"{label} mixed output error must be <= 3e-4, got {output_error}")
    finite_has = finite.get("mixed_compare") is not None
    rolling_has = rolling.get("mixed_compare") is not None
    if summary.get("finite_has_mixed_backward") is not finite_has:
        errors.append("summary finite_has_mixed_backward must match finite row")
    if summary.get("rolling_has_mixed_backward") is not rolling_has:
        errors.append("summary rolling_has_mixed_backward must match rolling row")
    if summary.get("finite_fallback_fraction") != finite.get("fallback", {}).get("fallback_fraction"):
        errors.append("summary finite_fallback_fraction must match finite row")
    if summary.get("rolling_fallback_fraction") != rolling.get("fallback", {}).get("fallback_fraction"):
        errors.append("summary rolling_fallback_fraction must match rolling row")
    if summary.get("rolling_unique_to_row_sample_ratio") != rolling.get("unique_to_row_sample_ratio"):
        errors.append("summary rolling_unique_to_row_sample_ratio must match rolling row")
    compare_rows = [row for row in (finite.get("mixed_compare"), rolling.get("mixed_compare")) if row is not None]
    if summary.get("mixed_backward_case_count") != len(compare_rows):
        errors.append("summary mixed_backward_case_count must match compare rows")
    if compare_rows:
        expected_abs = max(row["max_abs_error"] for row in compare_rows)
        expected_rel = max(row["max_rel_error"] for row in compare_rows)
        if summary.get("max_mixed_grad_abs_error") != expected_abs:
            errors.append("summary max_mixed_grad_abs_error must match compare rows")
        if summary.get("max_mixed_grad_rel_error") != expected_rel:
            errors.append("summary max_mixed_grad_rel_error must match compare rows")
    return errors


def assert_mixed_fallback_backward_report(report: dict[str, Any]) -> None:
    errors = verify_mixed_fallback_backward_report(report)
    if errors:
        raise AssertionError("mixed fallback backward report failed:\n- " + "\n- ".join(errors))


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# STAR UVT Projective Mixed Fallback Backward",
        "",
        "This report checks differentiable fallback patches for finite exposure and rolling shutter.",
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
                "finite_mixed_fallback_backward": report["finite_mixed_fallback_backward"],
                "rolling_mixed_fallback_backward": report["rolling_mixed_fallback_backward"],
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
        assert_mixed_fallback_backward_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(run_metal=not args.skip_metal)
    assert_mixed_fallback_backward_report(report)
    write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
