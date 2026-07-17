from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
STAR_UVT_ROOT = ROOT / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
STAR_UVT_HARNESS_ROOT = STAR_UVT_ROOT / "research_project" / "trainer_harness"
for path in (ROOT, ROOT / "src" / "train", STAR_UVT_ROOT, STAR_UVT_HARNESS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    UVTRenderConfig,
    direct_backward_projective_trace_cell_interval_atlas_metal,
    has_projective_trace_cell_interval_backward_metal,
    has_projective_trace_cell_interval_metal,
    projective_trace_cell_atlas_complexity_stats,
    projective_trace_cell_atlas_fallback_stats,
    render_projective_trace_cell_interval_atlas_metal,
    uvt_tubes_to_projective_trace_cell_atlas,
)
from tile_metal_autograd import render_projective_cell_interval_atlas_metal_backward  # noqa: E402
from variable_camera_segments import project_piecewise_camera_time_segments  # noqa: E402
from world_tube import WorldTubeBatch  # noqa: E402


DEFAULT_OUT_DIR = ROOT / "outputs" / "benchmarks" / "2026-05-24_star_uvt_revolving_orbit_fixed_chart_scaling"


def _look_at_w2c(eye: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    forward = target - eye
    forward = forward / forward.norm().clamp_min(1.0e-8)
    up_hint = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    right = torch.cross(up_hint, forward, dim=0)
    right = right / right.norm().clamp_min(1.0e-8)
    up = torch.cross(forward, right, dim=0)
    up = up / up.norm().clamp_min(1.0e-8)
    rotation = torch.stack((right, up, forward), dim=0)
    w2c = torch.eye(4, dtype=torch.float32)
    w2c[:3, :3] = rotation
    w2c[:3, 3] = -(rotation @ eye)
    return w2c


def _elevated_orbit_camera_sequence(frames: int, *, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    k_seq = torch.eye(3, dtype=torch.float32).repeat(frames, 1, 1)
    k_seq[:, 0, 0] = 1.875 * float(image_size)
    k_seq[:, 1, 1] = 1.8125 * float(image_size)
    k_seq[:, 0, 2] = 0.5 * float(image_size)
    k_seq[:, 1, 2] = 0.5 * float(image_size)
    target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    w2c = []
    for theta in torch.linspace(-math.radians(60.0), math.radians(60.0), frames):
        eye = torch.tensor(
            [
                2.5 * math.sin(float(theta)),
                0.7,
                -2.5 * math.cos(float(theta)),
            ],
            dtype=torch.float32,
        )
        w2c.append(_look_at_w2c(eye, target))
    return k_seq.contiguous(), torch.stack(w2c, dim=0).contiguous()


def _orbit_world_tube_batch() -> WorldTubeBatch:
    return WorldTubeBatch(
        x0=torch.tensor([[0.15, 0.08, 0.0], [-0.10, 0.05, 0.04]], dtype=torch.float32),
        velocity=torch.tensor([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0]], dtype=torch.float32),
        t0=torch.zeros(2, dtype=torch.float32),
        precision_xy=torch.tensor([[40.0, 160.0], [120.0, 50.0]], dtype=torch.float32),
        lambda_t=torch.tensor([0.2, 0.3], dtype=torch.float32),
        opacity=torch.tensor([0.5, 0.45], dtype=torch.float32),
        color=torch.tensor([[0.8, 0.2, 0.1], [0.1, 0.6, 0.9]], dtype=torch.float32),
    )


def _orbit_times(frames: int, *, device: torch.device | str = "cpu") -> torch.Tensor:
    return (torch.arange(frames, dtype=torch.float32, device=device) - 0.5 * float(frames - 1)).contiguous()


def _orbit_config(frames: int, *, image_size: int, tile_size: int, tile_t: int, tile_capacity: int) -> UVTRenderConfig:
    return UVTRenderConfig(
        height=image_size,
        width=image_size,
        frames=frames,
        tile_x=tile_size,
        tile_y=tile_size,
        tile_t=tile_t,
        tile_capacity=tile_capacity,
        alpha_threshold=0.01,
        max_alpha=1.0,
    )


def _apply_metal_tile_env(config: UVTRenderConfig) -> None:
    os.environ["STAR_UVT_TILE_X"] = str(config.tile_x)
    os.environ["STAR_UVT_TILE_Y"] = str(config.tile_y)
    os.environ["STAR_UVT_TILE_T"] = str(config.tile_t)
    os.environ["STAR_UVT_TILE_CAPACITY"] = str(config.tile_capacity)


def _compile_orbit_interval_atlas(projected: object, times: torch.Tensor, *, image_size: int, tile_size: int):
    return uvt_tubes_to_projective_trace_cell_atlas(
        projected.ma.to(device=times.device),
        projected.q_uvt.to(device=times.device),
        projected.depth0.to(device=times.device),
        projected.depth_beta.to(device=times.device),
        projected.opacity.to(device=times.device),
        projected.color.to(device=times.device),
        times,
        sigma_px=2.0,
        image_width=image_size,
        image_height=image_size,
        tile_size=tile_size,
        uv_padding=0.0,
        alpha_threshold=0.01,
        require_isotropic_spatial=False,
        auto_support_padding_from_alpha=True,
    )


def _tensor_bytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def _atlas_payload_bytes(atlas: object) -> int:
    tensors = (
        atlas.coeffs,
        atlas.opacity,
        atlas.color,
        atlas.opacity_time_coeffs,
        atlas.spatial_precision_uv,
        atlas.depth_affine_uv,
    )
    return sum(_tensor_bytes(tensor) for tensor in tensors)


def _time_forward(atlas: object, times: torch.Tensor, config: UVTRenderConfig, *, iterations: int, warmup: int) -> dict[str, Any]:
    _apply_metal_tile_env(config)
    with torch.no_grad():
        for _ in range(warmup):
            render_projective_trace_cell_interval_atlas_metal(atlas, times, config, sigma_px=2.0)
        torch.mps.synchronize()
        started = time.perf_counter()
        image = None
        for _ in range(iterations):
            image = render_projective_trace_cell_interval_atlas_metal(atlas, times, config, sigma_px=2.0)
        torch.mps.synchronize()
    if image is None:
        raise AssertionError("interval Metal forward did not run")
    return {
        "forward_ms": (time.perf_counter() - started) * 1000.0 / float(iterations),
        "image_sum": float(image.sum().detach().cpu().item()),
    }


def _time_backward(atlas: object, times: torch.Tensor, config: UVTRenderConfig, *, iterations: int, warmup: int) -> dict[str, Any]:
    _apply_metal_tile_env(config)
    sample_count = int(config.frames) * int(config.height) * int(config.width) * 3
    grad_image = torch.linspace(-0.25, 0.35, steps=sample_count, dtype=torch.float32, device=times.device)
    grad_image = grad_image.reshape(int(config.frames), int(config.height), int(config.width), 3).contiguous()
    with torch.no_grad():
        for _ in range(warmup):
            direct_backward_projective_trace_cell_interval_atlas_metal(atlas, times, grad_image, config, sigma_px=2.0)
        torch.mps.synchronize()
        started = time.perf_counter()
        grads = None
        for _ in range(iterations):
            grads = direct_backward_projective_trace_cell_interval_atlas_metal(
                atlas,
                times,
                grad_image,
                config,
                sigma_px=2.0,
            )
        torch.mps.synchronize()
    if grads is None:
        raise AssertionError("interval Metal backward did not run")
    return {
        "backward_ms": (time.perf_counter() - started) * 1000.0 / float(iterations),
        "grad_coeff_abs_sum": float(grads.grad_coeffs.abs().sum().detach().cpu().item()),
        "grad_opacity_abs_sum": float(grads.grad_opacity.abs().sum().detach().cpu().item()),
        "grad_color_abs_sum": float(grads.grad_color.abs().sum().detach().cpu().item()),
        "grad_spatial_precision_uv_abs_sum": (
            None
            if grads.grad_spatial_precision_uv is None
            else float(grads.grad_spatial_precision_uv.abs().sum().detach().cpu().item())
        ),
    }


def _autograd_topology_check(
    projected: object,
    *,
    frames: int,
    image_size: int,
    tile_size: int,
    tile_t: int,
    tile_capacity: int,
) -> dict[str, Any]:
    times = _orbit_times(frames, device="mps")
    ma = projected.ma.to("mps").detach().requires_grad_(True)
    q_uvt = projected.q_uvt.to("mps").detach().requires_grad_(True)
    opacity = projected.opacity.to("mps").detach().requires_grad_(True)
    color = projected.color.to("mps").detach().requires_grad_(True)
    atlas = uvt_tubes_to_projective_trace_cell_atlas(
        ma,
        q_uvt,
        projected.depth0.to("mps").detach(),
        projected.depth_beta.to("mps").detach(),
        opacity,
        color,
        times,
        sigma_px=2.0,
        image_width=image_size,
        image_height=image_size,
        tile_size=tile_size,
        uv_padding=0.0,
        alpha_threshold=0.01,
        require_isotropic_spatial=False,
        auto_support_padding_from_alpha=True,
    )
    image = render_projective_cell_interval_atlas_metal_backward(
        atlas,
        times,
        _orbit_config(frames, image_size=image_size, tile_size=tile_size, tile_t=tile_t, tile_capacity=tile_capacity),
        sigma_px=2.0,
    )
    x_weight = torch.linspace(-1.0, 1.0, image_size, device="mps").view(1, 1, image_size, 1)
    y_weight = torch.linspace(-0.7, 1.3, image_size, device="mps").view(1, image_size, 1, 1)
    t_weight = torch.linspace(0.8, 1.2, frames, device="mps").view(frames, 1, 1, 1)
    loss = (image * x_weight * y_weight * t_weight).sum() + 0.1 * image.square().mean()
    loss.backward()
    return {
        "autograd_ma_grad_abs_sum": float(ma.grad.detach().abs().sum().cpu().item()) if ma.grad is not None else 0.0,
        "autograd_q_uvt_grad_abs_sum": (
            float(q_uvt.grad.detach().abs().sum().cpu().item()) if q_uvt.grad is not None else 0.0
        ),
        "autograd_q_uv_grad_abs_sum": (
            float(q_uvt.grad[:, 1].detach().abs().sum().cpu().item()) if q_uvt.grad is not None else 0.0
        ),
        "autograd_q_temporal_grad_abs_sum": (
            float(q_uvt.grad[:, [2, 4, 5]].detach().abs().sum().cpu().item()) if q_uvt.grad is not None else 0.0
        ),
        "autograd_opacity_grad_abs_sum": (
            float(opacity.grad.detach().abs().sum().cpu().item()) if opacity.grad is not None else 0.0
        ),
        "autograd_color_grad_abs_sum": (
            float(color.grad.detach().abs().sum().cpu().item()) if color.grad is not None else 0.0
        ),
    }


def run_route(
    *,
    route: str,
    frames: int,
    fixed_temporal_chunks: int,
    image_size: int,
    tile_size: int,
    tile_t: int,
    tile_capacity: int,
    iterations: int,
    warmup: int,
    run_metal: bool,
    run_autograd_topology: bool,
) -> dict[str, Any]:
    if route == "per_frame":
        frames_per_segment = 1
    elif route == "fixed_chart":
        if frames % fixed_temporal_chunks != 0:
            raise ValueError(f"frames={frames} must be divisible by fixed_temporal_chunks={fixed_temporal_chunks}")
        frames_per_segment = frames // fixed_temporal_chunks
    else:
        raise ValueError(f"unknown route: {route}")

    batch = _orbit_world_tube_batch()
    k_seq, w2c_seq = _elevated_orbit_camera_sequence(frames, image_size=image_size)
    config = _orbit_config(frames, image_size=image_size, tile_size=tile_size, tile_t=tile_t, tile_capacity=tile_capacity)
    project_started = time.perf_counter()
    projected = project_piecewise_camera_time_segments(
        batch,
        k_seq,
        w2c_seq,
        config,
        full_frames=frames,
        frames_per_segment=frames_per_segment,
    )
    project_ms = (time.perf_counter() - project_started) * 1000.0
    times_cpu = _orbit_times(frames)
    atlas_started = time.perf_counter()
    atlas_cpu = _compile_orbit_interval_atlas(projected, times_cpu, image_size=image_size, tile_size=tile_size)
    atlas_build_ms = (time.perf_counter() - atlas_started) * 1000.0
    stats = projective_trace_cell_atlas_complexity_stats(atlas_cpu)
    fallback = projective_trace_cell_atlas_fallback_stats(atlas_cpu)
    row: dict[str, Any] = {
        "route": route,
        "frames": frames,
        "frames_per_segment": frames_per_segment,
        "temporal_chunk_count": int(projected.diagnostics.temporal_chunk_count),
        "segment_count": int(projected.diagnostics.segment_count),
        "trace_count": int(atlas_cpu.coeffs.shape[0]),
        "cell_count": int(len(atlas_cpu.cells)),
        "interval_trace_entries": int(stats.interval_trace_entries),
        "dense_trace_samples": int(stats.dense_trace_samples),
        "interval_to_dense_trace_sample_ratio": float(stats.interval_to_dense_trace_sample_ratio),
        "fallback_fraction": float(fallback.fallback_fraction),
        "atlas_payload_bytes": _atlas_payload_bytes(atlas_cpu),
        "project_ms": project_ms,
        "atlas_build_ms": atlas_build_ms,
        "cpu_compile_ms": project_ms + atlas_build_ms,
    }
    if run_metal:
        if not torch.backends.mps.is_available():
            row["metal_skipped"] = "MPS unavailable"
        elif not has_projective_trace_cell_interval_metal():
            row["metal_skipped"] = "projective interval cell Metal op unavailable"
        elif not has_projective_trace_cell_interval_backward_metal():
            row["metal_skipped"] = "projective interval cell Metal backward op unavailable"
        else:
            times_mps = _orbit_times(frames, device="mps")
            mps_atlas_started = time.perf_counter()
            atlas_mps = _compile_orbit_interval_atlas(projected, times_mps, image_size=image_size, tile_size=tile_size)
            row["mps_atlas_build_ms"] = (time.perf_counter() - mps_atlas_started) * 1000.0
            row.update(_time_forward(atlas_mps, times_mps, config, iterations=iterations, warmup=warmup))
            row.update(_time_backward(atlas_mps, times_mps, config, iterations=iterations, warmup=warmup))
            if run_autograd_topology and route == "fixed_chart":
                row.update(
                    _autograd_topology_check(
                        projected,
                        frames=frames,
                        image_size=image_size,
                        tile_size=tile_size,
                        tile_t=tile_t,
                        tile_capacity=tile_capacity,
                    )
                )
    return row


def _growth(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows]
    if not values or values[0] in (None, 0) or values[-1] is None:
        return None
    return float(values[-1]) / float(values[0])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fixed = [row for row in rows if row["route"] == "fixed_chart"]
    per_frame = [row for row in rows if row["route"] == "per_frame"]
    summary: dict[str, Any] = {
        "fixed_chart_segment_counts": [row["segment_count"] for row in fixed],
        "per_frame_segment_counts": [row["segment_count"] for row in per_frame],
        "fixed_chart_trace_counts": [row["trace_count"] for row in fixed],
        "per_frame_trace_counts": [row["trace_count"] for row in per_frame],
        "fixed_chart_interval_ratio_start": fixed[0]["interval_to_dense_trace_sample_ratio"] if fixed else None,
        "fixed_chart_interval_ratio_end": fixed[-1]["interval_to_dense_trace_sample_ratio"] if fixed else None,
        "fixed_chart_interval_entry_growth": _growth(fixed, "interval_trace_entries"),
        "fixed_chart_dense_trace_sample_growth": _growth(fixed, "dense_trace_samples"),
        "fixed_chart_payload_byte_growth": _growth(fixed, "atlas_payload_bytes"),
        "per_frame_payload_byte_growth": _growth(per_frame, "atlas_payload_bytes"),
        "fixed_chart_project_ms_growth": _growth(fixed, "project_ms"),
        "fixed_chart_atlas_build_ms_growth": _growth(fixed, "atlas_build_ms"),
        "fixed_chart_cpu_compile_ms_growth": _growth(fixed, "cpu_compile_ms"),
        "per_frame_project_ms_growth": _growth(per_frame, "project_ms"),
        "per_frame_atlas_build_ms_growth": _growth(per_frame, "atlas_build_ms"),
        "per_frame_cpu_compile_ms_growth": _growth(per_frame, "cpu_compile_ms"),
        "fixed_chart_forward_ms_growth": _growth(fixed, "forward_ms"),
        "fixed_chart_backward_ms_growth": _growth(fixed, "backward_ms"),
        "per_frame_forward_ms_growth": _growth(per_frame, "forward_ms"),
        "per_frame_backward_ms_growth": _growth(per_frame, "backward_ms"),
        "all_fixed_chart_fallback_zero": all(row["fallback_fraction"] == 0.0 for row in fixed),
        "all_fixed_chart_autograd_q_uv_nonzero": all(
            row.get("autograd_q_uv_grad_abs_sum", 1.0) > 0.0 for row in fixed
        ),
    }
    if fixed and per_frame:
        summary["last_fixed_vs_per_frame_segment_ratio"] = fixed[-1]["segment_count"] / float(per_frame[-1]["segment_count"])
        summary["last_fixed_vs_per_frame_trace_ratio"] = fixed[-1]["trace_count"] / float(per_frame[-1]["trace_count"])
        if fixed[-1].get("forward_ms") is not None and per_frame[-1].get("forward_ms") is not None:
            summary["last_fixed_vs_per_frame_forward_ms_ratio"] = fixed[-1]["forward_ms"] / float(per_frame[-1]["forward_ms"])
        if fixed[-1].get("backward_ms") is not None and per_frame[-1].get("backward_ms") is not None:
            summary["last_fixed_vs_per_frame_backward_ms_ratio"] = fixed[-1]["backward_ms"] / float(per_frame[-1]["backward_ms"])
        if fixed[-1].get("cpu_compile_ms") is not None and per_frame[-1].get("cpu_compile_ms") is not None:
            summary["last_fixed_vs_per_frame_cpu_compile_ms_ratio"] = fixed[-1]["cpu_compile_ms"] / float(
                per_frame[-1]["cpu_compile_ms"]
            )
    return summary


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


def _assert_summary_close(
    summary: dict[str, Any],
    expected: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    atol: float = 1.0e-9,
) -> None:
    actual_value = summary.get(key)
    expected_value = expected.get(key)
    if isinstance(expected_value, float):
        if not isinstance(actual_value, int | float) or abs(float(actual_value) - expected_value) > atol:
            errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual_value!r}")
    elif actual_value != expected_value:
        errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual_value!r}")


def _assert_close(actual: float, expected: float, label: str, errors: list[str], *, atol: float = 1.0e-5) -> None:
    if abs(actual - expected) > atol:
        errors.append(f"{label} mismatch: expected {expected:.9g}, got {actual:.9g}")


def verify_orbit_fixed_chart_scaling_report(report: dict[str, object]) -> list[str]:
    """Return contract failures for a saved revolving-camera fixed-chart scaling report."""

    errors: list[str] = []
    if report.get("benchmark") != "star_uvt_revolving_orbit_fixed_chart_scaling":
        errors.append("benchmark must be star_uvt_revolving_orbit_fixed_chart_scaling")

    raw_frame_counts = report.get("frame_counts")
    if not isinstance(raw_frame_counts, list) or len(raw_frame_counts) < 3:
        errors.append("frame_counts must contain at least three frame counts")
        return errors
    frame_counts = [_finite_int(value, f"frame_counts[{idx}]", errors) for idx, value in enumerate(raw_frame_counts)]
    if frame_counts != sorted(frame_counts) or len(set(frame_counts)) != len(frame_counts):
        errors.append(f"frame_counts must be strictly increasing, got {frame_counts}")
    if any(frames <= 0 for frames in frame_counts):
        errors.append(f"frame_counts must be positive, got {frame_counts}")

    fixed_temporal_chunks = _finite_int(report.get("fixed_temporal_chunks"), "fixed_temporal_chunks", errors)
    for key in ("image_size", "tile_size", "tile_t", "tile_capacity"):
        if _finite_int(report.get(key), key, errors) <= 0:
            errors.append(f"{key} must be positive")
    if _finite_int(report.get("iterations"), "iterations", errors) <= 0:
        errors.append("iterations must be positive")
    if _finite_int(report.get("warmup"), "warmup", errors) < 0:
        errors.append("warmup must be nonnegative")
    if fixed_temporal_chunks <= 0:
        errors.append(f"fixed_temporal_chunks must be positive, got {fixed_temporal_chunks}")

    raw_rows = report.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != 2 * len(frame_counts):
        errors.append("rows must contain one fixed_chart and one per_frame row per frame count")
        return errors
    rows = [row for row in raw_rows if isinstance(row, dict)]
    if len(rows) != len(raw_rows):
        errors.append("all rows must be objects")
    by_route: dict[str, list[dict[str, Any]]] = {"fixed_chart": [], "per_frame": []}
    for row in rows:
        route = str(row.get("route"))
        if route not in by_route:
            errors.append(f"unknown route {route!r}")
            continue
        by_route[route].append(row)
    fixed = sorted(by_route["fixed_chart"], key=lambda row: int(row.get("frames", -1)))
    per_frame = sorted(by_route["per_frame"], key=lambda row: int(row.get("frames", -1)))
    if [row.get("frames") for row in fixed] != frame_counts:
        errors.append("fixed_chart rows must cover frame_counts exactly")
    if [row.get("frames") for row in per_frame] != frame_counts:
        errors.append("per_frame rows must cover frame_counts exactly")

    for route, route_rows in by_route.items():
        seen_frames = set()
        for row in route_rows:
            frames = _finite_int(row.get("frames"), f"{route} frames", errors)
            seen_frames.add(frames)
            frames_per_segment = _finite_int(row.get("frames_per_segment"), f"{route} frames_per_segment", errors)
            temporal_chunk_count = _finite_int(row.get("temporal_chunk_count"), f"{route} temporal_chunk_count", errors)
            segment_count = _finite_int(row.get("segment_count"), f"{route} segment_count", errors)
            trace_count = _finite_int(row.get("trace_count"), f"{route} trace_count", errors)
            cell_count = _finite_int(row.get("cell_count"), f"{route} cell_count", errors)
            interval_entries = _finite_int(row.get("interval_trace_entries"), f"{route} interval_trace_entries", errors)
            dense_samples = _finite_int(row.get("dense_trace_samples"), f"{route} dense_trace_samples", errors)
            payload_bytes = _finite_int(row.get("atlas_payload_bytes"), f"{route} atlas_payload_bytes", errors)
            project_ms = _finite_float(row.get("project_ms"), f"{route} project_ms", errors)
            atlas_build_ms = _finite_float(row.get("atlas_build_ms"), f"{route} atlas_build_ms", errors)
            cpu_compile_ms = _finite_float(row.get("cpu_compile_ms"), f"{route} cpu_compile_ms", errors)
            interval_ratio = _finite_float(
                row.get("interval_to_dense_trace_sample_ratio"),
                f"{route} interval_to_dense_trace_sample_ratio",
                errors,
            )
            fallback_fraction = _finite_float(row.get("fallback_fraction"), f"{route} fallback_fraction", errors)
            if frames <= 0 or frames not in frame_counts:
                errors.append(f"{route} row has invalid frames {frames}")
            if route == "fixed_chart":
                if frames % fixed_temporal_chunks != 0:
                    errors.append(f"fixed_chart frames {frames} must divide by fixed_temporal_chunks")
                if frames_per_segment != frames // max(fixed_temporal_chunks, 1):
                    errors.append(f"fixed_chart frames_per_segment wrong for frames={frames}")
                if temporal_chunk_count != fixed_temporal_chunks:
                    errors.append(f"fixed_chart temporal_chunk_count must stay fixed at {fixed_temporal_chunks}")
            else:
                if frames_per_segment != 1:
                    errors.append("per_frame frames_per_segment must be 1")
                if temporal_chunk_count != frames:
                    errors.append(f"per_frame temporal_chunk_count must equal frames={frames}")
            if segment_count <= 0 or trace_count <= 0 or cell_count <= 0:
                errors.append(f"{route} row for frames={frames} must have positive topology counts")
            if interval_entries <= 0 or dense_samples <= 0 or interval_entries > dense_samples:
                errors.append(f"{route} row for frames={frames} has invalid interval/dense counts")
            if not 0.0 < interval_ratio <= 1.0:
                errors.append(f"{route} row for frames={frames} has invalid interval ratio {interval_ratio}")
            if dense_samples > 0:
                _assert_close(
                    interval_ratio,
                    interval_entries / float(dense_samples),
                    f"{route} interval ratio for frames={frames}",
                    errors,
                )
            if route == "per_frame" and interval_entries != dense_samples:
                errors.append(f"per_frame row for frames={frames} must replay every dense trace sample")
            if fallback_fraction != 0.0:
                errors.append(f"{route} row for frames={frames} must be fallback-free")
            if payload_bytes <= 0:
                errors.append(f"{route} row for frames={frames} must carry a positive atlas payload")
            if project_ms <= 0.0:
                errors.append(f"{route} row for frames={frames} must have positive project_ms")
            if atlas_build_ms <= 0.0:
                errors.append(f"{route} row for frames={frames} must have positive atlas_build_ms")
            if cpu_compile_ms <= 0.0:
                errors.append(f"{route} row for frames={frames} must have positive cpu_compile_ms")
            _assert_close(
                cpu_compile_ms,
                project_ms + atlas_build_ms,
                f"{route} cpu_compile_ms for frames={frames}",
                errors,
            )
            if "metal_skipped" not in row:
                if _finite_float(row.get("mps_atlas_build_ms"), f"{route} mps_atlas_build_ms", errors) <= 0.0:
                    errors.append(f"{route} row for frames={frames} must have positive mps_atlas_build_ms")
                if _finite_float(row.get("forward_ms"), f"{route} forward_ms", errors) <= 0.0:
                    errors.append(f"{route} row for frames={frames} must have positive forward_ms")
                if _finite_float(row.get("backward_ms"), f"{route} backward_ms", errors) <= 0.0:
                    errors.append(f"{route} row for frames={frames} must have positive backward_ms")
                for grad_key in (
                    "grad_coeff_abs_sum",
                    "grad_opacity_abs_sum",
                    "grad_color_abs_sum",
                    "grad_spatial_precision_uv_abs_sum",
                ):
                    if _finite_float(row.get(grad_key), f"{route} {grad_key}", errors) <= 0.0:
                        errors.append(f"{route} row for frames={frames} must have positive {grad_key}")
        if seen_frames != set(frame_counts):
            errors.append(f"{route} rows have duplicate or missing frame counts")

    if not fixed or not per_frame:
        return errors

    fixed_segment_counts = [int(row["segment_count"]) for row in fixed]
    fixed_trace_counts = [int(row["trace_count"]) for row in fixed]
    fixed_payloads = [int(row["atlas_payload_bytes"]) for row in fixed]
    per_frame_segment_counts = [int(row["segment_count"]) for row in per_frame]
    per_frame_trace_counts = [int(row["trace_count"]) for row in per_frame]
    per_frame_payloads = [int(row["atlas_payload_bytes"]) for row in per_frame]
    fixed_ratios = [float(row["interval_to_dense_trace_sample_ratio"]) for row in fixed]

    if len(set(fixed_segment_counts)) != 1:
        errors.append(f"fixed_chart segment counts must stay constant, got {fixed_segment_counts}")
    if len(set(fixed_trace_counts)) != 1:
        errors.append(f"fixed_chart trace counts must stay constant, got {fixed_trace_counts}")
    if len(set(fixed_payloads)) != 1:
        errors.append(f"fixed_chart payload bytes must stay constant, got {fixed_payloads}")
    if per_frame_segment_counts != sorted(per_frame_segment_counts) or len(set(per_frame_segment_counts)) != len(per_frame_segment_counts):
        errors.append(f"per_frame segment counts must strictly grow, got {per_frame_segment_counts}")
    if per_frame_trace_counts != sorted(per_frame_trace_counts) or len(set(per_frame_trace_counts)) != len(per_frame_trace_counts):
        errors.append(f"per_frame trace counts must strictly grow, got {per_frame_trace_counts}")
    if per_frame_payloads != sorted(per_frame_payloads) or per_frame_payloads[-1] <= per_frame_payloads[0]:
        errors.append(f"per_frame payload bytes must grow, got {per_frame_payloads}")
    if fixed_ratios != sorted(fixed_ratios, reverse=True):
        errors.append(f"fixed_chart interval ratios must be non-increasing, got {fixed_ratios}")
    if fixed_ratios[-1] >= 0.35 * fixed_ratios[0]:
        errors.append("fixed_chart final interval ratio must fall by at least 65%")

    fixed_interval_growth = _growth(fixed, "interval_trace_entries") or 0.0
    fixed_dense_growth = _growth(fixed, "dense_trace_samples") or 0.0
    per_frame_compile_growth = _growth(per_frame, "cpu_compile_ms") or 0.0
    fixed_compile_growth = _growth(fixed, "cpu_compile_ms") or 0.0
    if fixed_dense_growth <= 4.0:
        errors.append(f"fixed_chart dense sample growth must prove frame densification, got {fixed_dense_growth}")
    if fixed_interval_growth >= fixed_dense_growth:
        errors.append("fixed_chart interval entries must grow slower than dense samples")
    if fixed_interval_growth >= 2.0:
        errors.append(f"fixed_chart interval entry growth should stay below 2x, got {fixed_interval_growth}")
    if fixed_compile_growth >= per_frame_compile_growth:
        errors.append("fixed_chart CPU compile growth must stay below per-frame growth")

    last_fixed = fixed[-1]
    last_per_frame = per_frame[-1]
    if int(last_fixed["segment_count"]) / float(last_per_frame["segment_count"]) >= 0.25:
        errors.append("last fixed/per-frame segment ratio must show world-side reuse")
    if int(last_fixed["trace_count"]) / float(last_per_frame["trace_count"]) >= 0.25:
        errors.append("last fixed/per-frame trace ratio must show payload reuse")
    if float(last_fixed["cpu_compile_ms"]) / float(last_per_frame["cpu_compile_ms"]) >= 0.5:
        errors.append("last fixed/per-frame CPU compile ratio must be below 0.5")
    if "metal_skipped" not in last_fixed and "metal_skipped" not in last_per_frame:
        if float(last_fixed["forward_ms"]) / float(last_per_frame["forward_ms"]) >= 0.5:
            errors.append("last fixed/per-frame forward timing ratio must be below 0.5")
        if float(last_fixed["backward_ms"]) / float(last_per_frame["backward_ms"]) >= 0.5:
            errors.append("last fixed/per-frame backward timing ratio must be below 0.5")

    for row in fixed:
        frames = int(row["frames"])
        for key in (
            "autograd_ma_grad_abs_sum",
            "autograd_q_uvt_grad_abs_sum",
            "autograd_q_uv_grad_abs_sum",
            "autograd_q_temporal_grad_abs_sum",
            "autograd_opacity_grad_abs_sum",
            "autograd_color_grad_abs_sum",
        ):
            if _finite_float(row.get(key), f"fixed_chart {key}", errors) <= 0.0:
                errors.append(f"fixed_chart row for frames={frames} must have positive {key}")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be present")
        return errors
    expected_summary = summarize([*fixed, *per_frame])
    for key in expected_summary:
        _assert_summary_close(summary, expected_summary, key, errors)
    if summary.get("all_fixed_chart_fallback_zero") is not True:
        errors.append("summary must report all_fixed_chart_fallback_zero true")
    if summary.get("all_fixed_chart_autograd_q_uv_nonzero") is not True:
        errors.append("summary must report all_fixed_chart_autograd_q_uv_nonzero true")

    return errors


def assert_orbit_fixed_chart_scaling_report(report: dict[str, object]) -> None:
    errors = verify_orbit_fixed_chart_scaling_report(report)
    if errors:
        raise AssertionError("orbit fixed-chart scaling report failed:\n- " + "\n- ".join(errors))


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    columns = (
        "route",
        "frames",
        "frames_per_segment",
        "segment_count",
        "trace_count",
        "cell_count",
        "interval_trace_entries",
        "dense_trace_samples",
        "interval_to_dense_trace_sample_ratio",
        "fallback_fraction",
        "atlas_payload_bytes",
        "project_ms",
        "atlas_build_ms",
        "cpu_compile_ms",
        "forward_ms",
        "backward_ms",
        "autograd_q_uv_grad_abs_sum",
        "autograd_q_temporal_grad_abs_sum",
    )
    lines = [
        "# STAR UVT Revolving Orbit Fixed-Chart Scaling",
        "",
        "This artifact measures a synthetic revolving-camera UVT trace atlas.",
        "It separates unavoidable pixel/sample growth from reusable world-side chart and interval-atlas work.",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame-counts",
        default="8,16,32,64",
        help=(
            "Comma-separated frame counts. The default starts at 8 frames so "
            "MPS launch/packing overhead does not dominate the last-frame "
            "fixed-chart/per-frame timing ratio."
        ),
    )
    parser.add_argument("--fixed-temporal-chunks", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--tile-t", type=int, default=2)
    parser.add_argument("--tile-capacity", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--skip-metal", action="store_true")
    parser.add_argument("--skip-autograd-topology", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--verify-report", type=Path, help="validate an existing summary.json without rerunning")
    args = parser.parse_args()

    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_orbit_fixed_chart_scaling_report(report)
        print(f"verified {args.verify_report}")
        return

    frame_counts = [int(part.strip()) for part in args.frame_counts.split(",") if part.strip()]
    if not args.skip_metal and frame_counts:
        run_route(
            route="fixed_chart",
            frames=frame_counts[0],
            fixed_temporal_chunks=args.fixed_temporal_chunks,
            image_size=args.image_size,
            tile_size=args.tile_size,
            tile_t=args.tile_t,
            tile_capacity=args.tile_capacity,
            iterations=1,
            warmup=max(1, args.warmup),
            run_metal=True,
            run_autograd_topology=False,
        )
    rows = []
    for frames in frame_counts:
        for route in ("fixed_chart", "per_frame"):
            rows.append(
                run_route(
                    route=route,
                    frames=frames,
                    fixed_temporal_chunks=args.fixed_temporal_chunks,
                    image_size=args.image_size,
                    tile_size=args.tile_size,
                    tile_t=args.tile_t,
                    tile_capacity=args.tile_capacity,
                    iterations=args.iterations,
                    warmup=args.warmup,
                    run_metal=not args.skip_metal,
                    run_autograd_topology=not args.skip_autograd_topology,
                )
            )
    report = {
        "benchmark": "star_uvt_revolving_orbit_fixed_chart_scaling",
        "frame_counts": frame_counts,
        "fixed_temporal_chunks": int(args.fixed_temporal_chunks),
        "image_size": int(args.image_size),
        "tile_size": int(args.tile_size),
        "tile_t": int(args.tile_t),
        "tile_capacity": int(args.tile_capacity),
        "iterations": int(args.iterations),
        "warmup": int(args.warmup),
        "summary": summarize(rows),
        "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "summary.json"
    md_path = args.out_dir / "summary.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
