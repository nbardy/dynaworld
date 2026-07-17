from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
STAR_UVT_ROOT = ROOT / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
for path in (ROOT, ROOT / "src" / "train", STAR_UVT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import star_uvt_feature_overfit_trainer as feature_overfit_trainer  # noqa: E402
from star_uvt_checkpoints import load_star_model_from_training_checkpoint  # noqa: E402
from star_uvt_feature_config import resolve_config  # noqa: E402
from star_uvt_models import build_feature_tube_model  # noqa: E402
from star_uvt_render_configs import feature_tube_render_config_from_cfg  # noqa: E402
from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    UVTRenderConfig,
    count_projective_trace_dense_per_frame_tile_pairs,
    direct_backward_projective_trace_cell_interval_atlas_metal,
    has_projective_trace_cell_interval_backward_metal,
    has_projective_trace_cell_interval_metal,
    projective_trace_cell_atlas_complexity_stats,
    projective_trace_cell_atlas_fallback_stats,
    render_projective_trace_cell_interval_atlas_metal,
    uvt_tubes_to_projective_trace_cell_atlas,
)


HIGH_MOTION_VIDEO = ROOT / "data/youtube_curated_spans/high_motion_smokes/hlaZbH_OFBU_seg_003_4fps_16f.mp4"
DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling"
)


def _base_config(
    *,
    frames: int,
    size: int,
    steps: int,
    tube_count: int,
    tile_capacity: int,
    out_json: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    return {
        "data": {
            "video_path": str(HIGH_MOTION_VIDEO),
            "start_seconds": None,
            "fps": None,
            "duration_seconds": None,
            "image_crop_mode": "center",
            "target_size": int(size),
            "max_frames": int(frames),
        },
        "train": {
            "steps": int(steps),
            "lr": 0.01,
            "device": "mps",
            "seed": 13,
            "frame_chunk_size": None,
            "require_loss_decrease": False,
            "require_gradient_flow": False,
            "require_no_tile_overflow": False,
        },
        "feature_uvt": {
            "tube_count": int(tube_count),
            "feature_dim": 3,
            "tile_t": 2,
            "tile_capacity": int(tile_capacity),
            "alpha_threshold": 0.01,
            "max_alpha": 0.99,
            "render_mode": "feature_direct_atomic",
            "projective_interval": {
                "enabled": True,
                "sigma_px": 2.0,
                "tile_size": 8,
                "uv_padding": 8.0,
                "refresh_every": max(1, int(steps) + 1),
                "refresh_policy": "measured",
                "fallback_render_mode": "mixed",
            },
        },
        "colorize": {
            "hidden_dim": None,
            "activation": "sigmoid",
            "pre_norm": False,
            "weight_init": "kaiming",
            "weight_init_gain": 1.0,
        },
        "output": {
            "out_json": str(out_json),
            "checkpoint": str(checkpoint),
            "contact_sheet": None,
            "contact_sheet_frames": int(frames),
            "contact_sheet_mode": "grid",
            "side_by_side_video": None,
            "side_by_side_fps": 8,
        },
        "logging": {
            "wandb_enabled": False,
            "wandb_project": "unit",
            "wandb_run_name": "projective-trained-high-motion-trace-scaling",
            "wandb_tags": [],
            "wandb_mode": None,
        },
    }


def _projective_times(*, frame_count: int, trained_frames: int, device: torch.device) -> torch.Tensor:
    return (
        torch.arange(int(frame_count), dtype=torch.float32, device=device)
        - 0.5 * float(int(trained_frames) - 1)
    ).contiguous()


def _apply_metal_tile_env(render_cfg: UVTRenderConfig) -> None:
    import os

    os.environ["STAR_UVT_TILE_X"] = str(render_cfg.tile_x)
    os.environ["STAR_UVT_TILE_Y"] = str(render_cfg.tile_y)
    os.environ["STAR_UVT_TILE_T"] = str(render_cfg.tile_t)
    os.environ["STAR_UVT_TILE_CAPACITY"] = str(render_cfg.tile_capacity)


def _time_interval_metal(
    atlas: Any,
    times: torch.Tensor,
    render_cfg: UVTRenderConfig,
    *,
    sigma_px: float,
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    _apply_metal_tile_env(render_cfg)
    with torch.no_grad():
        for _ in range(int(warmup)):
            render_projective_trace_cell_interval_atlas_metal(atlas, times, render_cfg, sigma_px=float(sigma_px))
        torch.mps.synchronize()
        started = time.perf_counter()
        image = None
        for _ in range(int(iterations)):
            image = render_projective_trace_cell_interval_atlas_metal(
                atlas,
                times,
                render_cfg,
                sigma_px=float(sigma_px),
            )
        torch.mps.synchronize()
        forward_ms = (time.perf_counter() - started) * 1000.0 / float(max(1, int(iterations)))
        sample_count = int(render_cfg.frames) * int(render_cfg.height) * int(render_cfg.width) * 3
        grad_image = torch.linspace(-0.25, 0.35, steps=sample_count, dtype=torch.float32, device=times.device)
        grad_image = grad_image.reshape(int(render_cfg.frames), int(render_cfg.height), int(render_cfg.width), 3)
        for _ in range(int(warmup)):
            direct_backward_projective_trace_cell_interval_atlas_metal(
                atlas,
                times,
                grad_image.contiguous(),
                render_cfg,
                sigma_px=float(sigma_px),
            )
        torch.mps.synchronize()
        started = time.perf_counter()
        grads = None
        for _ in range(int(iterations)):
            grads = direct_backward_projective_trace_cell_interval_atlas_metal(
                atlas,
                times,
                grad_image.contiguous(),
                render_cfg,
                sigma_px=float(sigma_px),
            )
        torch.mps.synchronize()
        backward_ms = (time.perf_counter() - started) * 1000.0 / float(max(1, int(iterations)))
    if image is None or grads is None:
        raise AssertionError("interval Metal timing did not run")
    return {
        "forward_ms": float(forward_ms),
        "backward_ms": float(backward_ms),
        "image_sum": float(image.sum().detach().cpu().item()),
        "grad_coeff_abs_sum": float(grads.grad_coeffs.abs().sum().detach().cpu().item()),
        "grad_opacity_abs_sum": float(grads.grad_opacity.abs().sum().detach().cpu().item()),
        "grad_color_abs_sum": float(grads.grad_color.abs().sum().detach().cpu().item()),
    }


def _make_render_config(
    *,
    cfg: dict[str, Any],
    feature_config: Any,
    frame_count: int,
) -> UVTRenderConfig:
    backend_cfg = cfg["feature_uvt"]["projective_interval"]
    return UVTRenderConfig(
        height=int(feature_config.height),
        width=int(feature_config.width),
        frames=int(frame_count),
        tile_x=int(backend_cfg["tile_size"]),
        tile_y=int(backend_cfg["tile_size"]),
        tile_t=int(cfg["feature_uvt"]["tile_t"]),
        tile_capacity=int(cfg["feature_uvt"]["tile_capacity"]),
        alpha_threshold=float(feature_config.alpha_threshold),
        max_alpha=float(feature_config.max_alpha),
    )


def _compile_projective_atlas(
    *,
    ma: torch.Tensor,
    q_uvt: torch.Tensor,
    depth0: torch.Tensor,
    depth_beta: torch.Tensor,
    opacity: torch.Tensor,
    feature: torch.Tensor,
    times: torch.Tensor,
    render_cfg: UVTRenderConfig,
    backend_cfg: dict[str, Any],
) -> Any:
    return uvt_tubes_to_projective_trace_cell_atlas(
        ma.detach(),
        q_uvt.detach(),
        depth0.detach(),
        depth_beta.detach(),
        opacity.detach(),
        feature.detach(),
        times,
        sigma_px=float(backend_cfg["sigma_px"]),
        image_width=int(render_cfg.width),
        image_height=int(render_cfg.height),
        tile_size=int(backend_cfg["tile_size"]),
        uv_padding=float(backend_cfg["uv_padding"]),
        alpha_threshold=float(render_cfg.alpha_threshold),
        temporal_mode="trace",
        stratify_visibility=True,
        mark_visibility_fallback=True,
    )


def _compile_geometry_row(
    *,
    label: str,
    model: torch.nn.Module,
    cfg: dict[str, Any],
    frame_count: int,
    trained_frames: int,
    run_metal_timing: bool,
    timing_iterations: int,
    timing_warmup: int,
) -> dict[str, Any]:
    feature_config = feature_tube_render_config_from_cfg(cfg)
    backend_cfg = cfg["feature_uvt"]["projective_interval"]
    render_cfg = _make_render_config(cfg=cfg, feature_config=feature_config, frame_count=frame_count)
    with torch.no_grad():
        ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()
        times = _projective_times(frame_count=frame_count, trained_frames=trained_frames, device=ma.device)
        atlas = _compile_projective_atlas(
            ma=ma,
            q_uvt=q_uvt,
            depth0=depth0,
            depth_beta=depth_beta,
            opacity=opacity,
            feature=feature,
            times=times,
            render_cfg=render_cfg,
            backend_cfg=backend_cfg,
        )
        complexity = projective_trace_cell_atlas_complexity_stats(atlas)
        fallback = projective_trace_cell_atlas_fallback_stats(atlas)
        dense_tile_pairs = count_projective_trace_dense_per_frame_tile_pairs(
            atlas.coeffs,
            times,
            image_width=int(render_cfg.width),
            image_height=int(render_cfg.height),
            tile_size=int(backend_cfg["tile_size"]),
            uv_padding=float(backend_cfg["uv_padding"]),
        )
        velocity_norm = model.velocity_uv.detach().norm(dim=1)
        interval_entries = int(complexity.interval_trace_entries)
        row = {
            "label": str(label),
            "frames": int(frame_count),
            "trace_count": int(atlas.coeffs.shape[0]),
            "cell_count": int(complexity.total_cells),
            "tile_active_set_groups": int(complexity.tile_active_set_groups),
            "max_cells_per_active_set_group": int(complexity.max_cells_per_active_set_group),
            "interval_trace_entries": interval_entries,
            "dense_trace_samples": int(complexity.dense_trace_samples),
            "interval_to_dense_trace_sample_ratio": float(complexity.interval_to_dense_trace_sample_ratio),
            "dense_per_frame_tile_pairs": int(dense_tile_pairs),
            "interval_to_dense_tile_pair_ratio": float(interval_entries) / float(max(1, dense_tile_pairs)),
            "fallback_cells": int(fallback.fallback_cells),
            "fallback_fraction": float(fallback.fallback_fraction),
            "fallback_reasons": [str(reason) for reason in fallback.fallback_reasons],
            "velocity_nonzero_count": int((velocity_norm > 0.0).sum().cpu().item()),
            "velocity_mean_px_per_frame": float(velocity_norm.mean().cpu().item()),
            "velocity_max_px_per_frame": float(velocity_norm.max().cpu().item()),
            "opacity_min": float(opacity.detach().min().cpu().item()),
            "opacity_max": float(opacity.detach().max().cpu().item()),
        }
        if run_metal_timing:
            row.update(
                _time_interval_metal(
                    atlas,
                    times,
                    render_cfg,
                    sigma_px=float(backend_cfg["sigma_px"]),
                    iterations=int(timing_iterations),
                    warmup=int(timing_warmup),
                )
            )
        return row


def _time_per_frame_metal(
    atlases: list[Any],
    times: list[torch.Tensor],
    render_cfg: UVTRenderConfig,
    *,
    sigma_px: float,
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    _apply_metal_tile_env(render_cfg)
    sample_count = int(render_cfg.height) * int(render_cfg.width) * 3
    grad_image = torch.linspace(-0.25, 0.35, steps=sample_count, dtype=torch.float32, device=times[0].device)
    grad_image = grad_image.reshape(1, int(render_cfg.height), int(render_cfg.width), 3).contiguous()
    image_sum = 0.0
    grad_coeff_abs_sum = 0.0
    grad_opacity_abs_sum = 0.0
    grad_color_abs_sum = 0.0
    with torch.no_grad():
        for _ in range(int(warmup)):
            for atlas, frame_time in zip(atlases, times, strict=True):
                render_projective_trace_cell_interval_atlas_metal(atlas, frame_time, render_cfg, sigma_px=float(sigma_px))
        torch.mps.synchronize()
        started = time.perf_counter()
        for _ in range(int(iterations)):
            for atlas, frame_time in zip(atlases, times, strict=True):
                image = render_projective_trace_cell_interval_atlas_metal(
                    atlas,
                    frame_time,
                    render_cfg,
                    sigma_px=float(sigma_px),
                )
                image_sum += float(image.sum().detach().cpu().item())
        torch.mps.synchronize()
        forward_ms = (time.perf_counter() - started) * 1000.0 / float(max(1, int(iterations)))
        for _ in range(int(warmup)):
            for atlas, frame_time in zip(atlases, times, strict=True):
                direct_backward_projective_trace_cell_interval_atlas_metal(
                    atlas,
                    frame_time,
                    grad_image,
                    render_cfg,
                    sigma_px=float(sigma_px),
                )
        torch.mps.synchronize()
        started = time.perf_counter()
        for _ in range(int(iterations)):
            for atlas, frame_time in zip(atlases, times, strict=True):
                grads = direct_backward_projective_trace_cell_interval_atlas_metal(
                    atlas,
                    frame_time,
                    grad_image,
                    render_cfg,
                    sigma_px=float(sigma_px),
                )
                grad_coeff_abs_sum += float(grads.grad_coeffs.abs().sum().detach().cpu().item())
                grad_opacity_abs_sum += float(grads.grad_opacity.abs().sum().detach().cpu().item())
                grad_color_abs_sum += float(grads.grad_color.abs().sum().detach().cpu().item())
        torch.mps.synchronize()
        backward_ms = (time.perf_counter() - started) * 1000.0 / float(max(1, int(iterations)))
    return {
        "forward_ms": float(forward_ms),
        "backward_ms": float(backward_ms),
        "image_sum": float(image_sum / float(max(1, int(iterations)))),
        "grad_coeff_abs_sum": float(grad_coeff_abs_sum / float(max(1, int(iterations)))),
        "grad_opacity_abs_sum": float(grad_opacity_abs_sum / float(max(1, int(iterations)))),
        "grad_color_abs_sum": float(grad_color_abs_sum / float(max(1, int(iterations)))),
    }


def _compile_per_frame_baseline_row(
    *,
    label: str,
    model: torch.nn.Module,
    cfg: dict[str, Any],
    frame_count: int,
    trained_frames: int,
    run_metal_timing: bool,
    timing_iterations: int,
    timing_warmup: int,
) -> dict[str, Any]:
    feature_config = feature_tube_render_config_from_cfg(cfg)
    backend_cfg = cfg["feature_uvt"]["projective_interval"]
    single_render_cfg = _make_render_config(cfg=cfg, feature_config=feature_config, frame_count=1)
    with torch.no_grad():
        ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()
        prefix_times = _projective_times(frame_count=frame_count, trained_frames=trained_frames, device=ma.device)
        atlases = []
        frame_times = []
        interval_trace_entries = 0
        dense_trace_samples = 0
        dense_tile_pairs = 0
        cell_count = 0
        fallback_cells = 0
        fallback_reasons: set[str] = set()
        for frame_time in prefix_times:
            time_1 = frame_time.reshape(1).contiguous()
            atlas = _compile_projective_atlas(
                ma=ma,
                q_uvt=q_uvt,
                depth0=depth0,
                depth_beta=depth_beta,
                opacity=opacity,
                feature=feature,
                times=time_1,
                render_cfg=single_render_cfg,
                backend_cfg=backend_cfg,
            )
            complexity = projective_trace_cell_atlas_complexity_stats(atlas)
            fallback = projective_trace_cell_atlas_fallback_stats(atlas)
            interval_trace_entries += int(complexity.interval_trace_entries)
            dense_trace_samples += int(complexity.dense_trace_samples)
            cell_count += int(complexity.total_cells)
            fallback_cells += int(fallback.fallback_cells)
            fallback_reasons.update(str(reason) for reason in fallback.fallback_reasons)
            dense_tile_pairs += count_projective_trace_dense_per_frame_tile_pairs(
                atlas.coeffs,
                time_1,
                image_width=int(single_render_cfg.width),
                image_height=int(single_render_cfg.height),
                tile_size=int(backend_cfg["tile_size"]),
                uv_padding=float(backend_cfg["uv_padding"]),
            )
            atlases.append(atlas)
            frame_times.append(time_1)
        velocity_norm = model.velocity_uv.detach().norm(dim=1)
        row = {
            "label": str(label),
            "frames": int(frame_count),
            "trace_count": int(sum(int(atlas.coeffs.shape[0]) for atlas in atlases)),
            "cell_count": int(cell_count),
            "tile_active_set_groups": None,
            "max_cells_per_active_set_group": None,
            "interval_trace_entries": int(interval_trace_entries),
            "dense_trace_samples": int(dense_trace_samples),
            "interval_to_dense_trace_sample_ratio": float(interval_trace_entries) / float(max(1, dense_trace_samples)),
            "dense_per_frame_tile_pairs": int(dense_tile_pairs),
            "interval_to_dense_tile_pair_ratio": float(interval_trace_entries) / float(max(1, dense_tile_pairs)),
            "fallback_cells": int(fallback_cells),
            "fallback_fraction": float(fallback_cells) / float(max(1, cell_count)),
            "fallback_reasons": sorted(fallback_reasons),
            "velocity_nonzero_count": int((velocity_norm > 0.0).sum().cpu().item()),
            "velocity_mean_px_per_frame": float(velocity_norm.mean().cpu().item()),
            "velocity_max_px_per_frame": float(velocity_norm.max().cpu().item()),
            "opacity_min": float(opacity.detach().min().cpu().item()),
            "opacity_max": float(opacity.detach().max().cpu().item()),
        }
        if run_metal_timing:
            row.update(
                _time_per_frame_metal(
                    atlases,
                    frame_times,
                    single_render_cfg,
                    sigma_px=float(backend_cfg["sigma_px"]),
                    iterations=int(timing_iterations),
                    warmup=int(timing_warmup),
                )
            )
        return row


def _load_trained_model(*, cfg: dict[str, Any], checkpoint: Path, device: torch.device) -> torch.nn.Module:
    feature_config = feature_tube_render_config_from_cfg(cfg)
    model = build_feature_tube_model(cfg, feature_config, device=device)
    load_star_model_from_training_checkpoint(checkpoint, model=model, device=device, freeze_model=True)
    model.eval()
    return model


def _build_init_model(*, cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    feature_config = feature_tube_render_config_from_cfg(cfg)
    model = build_feature_tube_model(cfg, feature_config, device=device)
    model.eval()
    return model


def _growth(values: list[float | int]) -> float | None:
    if not values or float(values[0]) == 0.0:
        return None
    return float(values[-1]) / float(values[0])


def _rows_for_label(report: dict[str, Any], label: str) -> list[dict[str, Any]]:
    rows = [row for row in report.get("rows", []) if row.get("label") == label]
    return sorted(rows, key=lambda row: int(row.get("frames", 0)))


def _finite_positive(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value)) and float(value) > 0.0


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


def _assert_close(actual: float, expected: float, label: str, errors: list[str], *, atol: float = 1.0e-6) -> None:
    if abs(actual - expected) > atol:
        errors.append(f"{label} mismatch: expected {expected:.9g}, got {actual:.9g}")


def _assert_summary_close(
    summary: dict[str, Any],
    expected: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    atol: float = 1.0e-6,
) -> None:
    actual = summary.get(key)
    expected_value = expected.get(key)
    if isinstance(expected_value, float):
        if not isinstance(actual, int | float) or abs(float(actual) - expected_value) > atol:
            errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")
    elif actual != expected_value:
        errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")


def verify_trained_high_motion_trace_scaling_report(report: dict[str, Any]) -> list[str]:
    """Return human-readable contract failures for the saved scaling artifact."""

    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_trained_high_motion_trace_scaling":
        errors.append(f"unexpected benchmark name {report.get('benchmark')!r}")
    if not report.get("source_video_exists"):
        errors.append("source high-motion video must exist")
    raw_frame_counts = report.get("frame_counts")
    if not isinstance(raw_frame_counts, list) or len(raw_frame_counts) < 2:
        errors.append("frame_counts must contain at least two frame counts")
        return errors
    frame_counts = [_finite_int(value, f"frame_counts[{idx}]", errors) for idx, value in enumerate(raw_frame_counts)]
    if frame_counts != sorted(frame_counts) or len(set(frame_counts)) != len(frame_counts):
        errors.append(f"frame_counts must be strictly increasing, got {frame_counts}")
    if any(value <= 0 for value in frame_counts):
        errors.append(f"frame_counts must be positive, got {frame_counts}")
    trained_frames = _finite_int(report.get("trained_frames"), "trained_frames", errors)
    if trained_frames != max(frame_counts):
        errors.append(f"trained_frames must equal max(frame_counts), got {trained_frames}")
    for key in ("size", "steps", "tube_count", "tile_capacity"):
        if _finite_int(report.get(key), key, errors) <= 0:
            errors.append(f"{key} must be positive")

    train = report.get("train") if isinstance(report.get("train"), dict) else {}
    if not train.get("pass"):
        errors.append("training row must pass")
    if not train.get("loss_decreased"):
        errors.append("training loss must decrease")
    start_loss = _finite_float(train.get("start_loss"), "train start_loss", errors)
    end_loss = _finite_float(train.get("end_loss"), "train end_loss", errors)
    if not end_loss < start_loss:
        errors.append("training end_loss must be lower than start_loss")
    if int(train.get("tile_overflow_sum") or 0) != 0:
        errors.append(f"training tile_overflow_sum must be 0, got {train.get('tile_overflow_sum')!r}")

    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("rows must be a nonempty list")
        return errors
    if any(not isinstance(row, dict) for row in rows):
        errors.append("all rows must be objects")
        return errors
    allowed_labels = {"init", "trained_checkpoint", "trained_checkpoint_per_frame"}
    for idx, row in enumerate(rows):
        label = str(row.get("label"))
        if label not in allowed_labels:
            errors.append(f"row {idx} has unknown label {label!r}")
            continue
        frames = _finite_int(row.get("frames"), f"row {idx} frames", errors)
        if frames not in frame_counts:
            errors.append(f"row {idx} frames must be one of frame_counts, got {frames}")
        trace_count = _finite_int(row.get("trace_count"), f"row {idx} trace_count", errors)
        cell_count = _finite_int(row.get("cell_count"), f"row {idx} cell_count", errors)
        interval_entries = _finite_int(row.get("interval_trace_entries"), f"row {idx} interval_trace_entries", errors)
        dense_samples = _finite_int(row.get("dense_trace_samples"), f"row {idx} dense_trace_samples", errors)
        dense_tile_pairs = _finite_int(row.get("dense_per_frame_tile_pairs"), f"row {idx} dense_per_frame_tile_pairs", errors)
        fallback_cells = _finite_int(row.get("fallback_cells"), f"row {idx} fallback_cells", errors)
        interval_sample_ratio = _finite_float(
            row.get("interval_to_dense_trace_sample_ratio"),
            f"row {idx} interval_to_dense_trace_sample_ratio",
            errors,
        )
        interval_tile_ratio = _finite_float(
            row.get("interval_to_dense_tile_pair_ratio"),
            f"row {idx} interval_to_dense_tile_pair_ratio",
            errors,
        )
        fallback_fraction = _finite_float(row.get("fallback_fraction"), f"row {idx} fallback_fraction", errors)
        if trace_count <= 0 or cell_count <= 0 or interval_entries <= 0 or dense_samples <= 0 or dense_tile_pairs <= 0:
            errors.append(f"row {idx} topology/work counts must be positive")
        if interval_entries > dense_samples:
            errors.append(f"row {idx} interval_trace_entries must not exceed dense_trace_samples")
        if dense_samples > 0:
            _assert_close(
                interval_sample_ratio,
                interval_entries / float(dense_samples),
                f"row {idx} interval_to_dense_trace_sample_ratio",
                errors,
            )
        if dense_tile_pairs > 0:
            _assert_close(
                interval_tile_ratio,
                interval_entries / float(dense_tile_pairs),
                f"row {idx} interval_to_dense_tile_pair_ratio",
                errors,
            )
        if label == "trained_checkpoint_per_frame" and interval_entries != dense_samples:
            errors.append(f"row {idx} per-frame baseline must replay every dense trace sample")
        if fallback_cells != 0 or fallback_fraction != 0.0 or row.get("fallback_reasons"):
            errors.append(f"row {idx} must be fallback-free")
        velocity_nonzero_count = _finite_int(row.get("velocity_nonzero_count"), f"row {idx} velocity_nonzero_count", errors)
        velocity_mean = _finite_float(row.get("velocity_mean_px_per_frame"), f"row {idx} velocity_mean_px_per_frame", errors)
        velocity_max = _finite_float(row.get("velocity_max_px_per_frame"), f"row {idx} velocity_max_px_per_frame", errors)
        if velocity_nonzero_count <= 0 or velocity_mean <= 0.0 or velocity_max <= 0.0:
            errors.append(f"row {idx} must carry nonzero learned/projected velocity")
        if velocity_max < velocity_mean:
            errors.append(f"row {idx} velocity_max_px_per_frame must be >= velocity_mean_px_per_frame")
        opacity_min = _finite_float(row.get("opacity_min"), f"row {idx} opacity_min", errors)
        opacity_max = _finite_float(row.get("opacity_max"), f"row {idx} opacity_max", errors)
        if not 0.0 <= opacity_min <= opacity_max <= 1.0:
            errors.append(f"row {idx} opacity range must stay in [0, 1], got {opacity_min}..{opacity_max}")
        if row.get("forward_ms") is not None or row.get("backward_ms") is not None:
            if _finite_float(row.get("forward_ms"), f"row {idx} forward_ms", errors) <= 0.0:
                errors.append(f"row {idx} forward_ms must be positive when timing is present")
            if _finite_float(row.get("backward_ms"), f"row {idx} backward_ms", errors) <= 0.0:
                errors.append(f"row {idx} backward_ms must be positive when timing is present")
            for grad_key in ("grad_coeff_abs_sum", "grad_opacity_abs_sum", "grad_color_abs_sum"):
                if _finite_float(row.get(grad_key), f"row {idx} {grad_key}", errors) <= 0.0:
                    errors.append(f"row {idx} {grad_key} must be positive when timing is present")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
    else:
        try:
            expected_summary = _summarize(rows, train)
            for key in expected_summary:
                _assert_summary_close(summary, expected_summary, key, errors)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"summary could not be recomputed: {exc}")

    for label in ("trained_checkpoint", "trained_checkpoint_per_frame"):
        label_frames = [int(row.get("frames", 0)) for row in rows if row.get("label") == label]
        if sorted(label_frames) != frame_counts:
            errors.append(f"{label} rows must cover frame_counts exactly, got {label_frames}")

    trained_rows = _rows_for_label(report, "trained_checkpoint")
    if len(trained_rows) < 2:
        errors.append("need at least two trained_checkpoint rows to measure growth")
        return errors

    trained_frames = [int(row.get("frames", 0)) for row in trained_rows]
    if trained_frames != sorted(trained_frames) or len(set(trained_frames)) != len(trained_frames):
        errors.append(f"trained_checkpoint frames must be unique and sorted, got {trained_frames}")

    fallback_rows = [row for row in trained_rows if float(row.get("fallback_fraction") or 0.0) != 0.0]
    if fallback_rows:
        errors.append("trained_checkpoint rows must be fallback-free")

    trace_counts = [int(row.get("trace_count", -1)) for row in trained_rows]
    if len(set(trace_counts)) != 1 or trace_counts[0] <= 0:
        errors.append(f"trained_checkpoint trace_count must stay constant and positive, got {trace_counts}")

    interval_entries = [int(row.get("interval_trace_entries", 0)) for row in trained_rows]
    dense_tile_pairs = [int(row.get("dense_per_frame_tile_pairs", 0)) for row in trained_rows]
    if any(value <= 0 for value in interval_entries + dense_tile_pairs):
        errors.append("trained_checkpoint interval entries and dense tile pairs must be positive")
    else:
        interval_growth = _growth(interval_entries)
        dense_growth = _growth(dense_tile_pairs)
        if interval_growth is None or dense_growth is None or not interval_growth < dense_growth:
            errors.append(
                "trained_checkpoint interval entries must grow slower than dense tile-pair work "
                f"(interval={interval_growth}, dense={dense_growth})"
            )
        final_ratio = float(trained_rows[-1].get("interval_to_dense_tile_pair_ratio") or 0.0)
        if not 0.0 < final_ratio < 1.0:
            errors.append(f"final trained interval/dense tile ratio must be in (0,1), got {final_ratio}")
        first_ratio = float(trained_rows[0].get("interval_to_dense_tile_pair_ratio") or 0.0)
        if final_ratio >= first_ratio:
            errors.append(f"trained interval/dense tile ratio must decrease, got {first_ratio} -> {final_ratio}")

    per_frame_rows = _rows_for_label(report, "trained_checkpoint_per_frame")
    if per_frame_rows:
        per_frame_by_frame = {int(row.get("frames", 0)): row for row in per_frame_rows}
        trained_by_frame = {int(row.get("frames", 0)): row for row in trained_rows}
        common_frames = sorted(set(trained_by_frame) & set(per_frame_by_frame))
        for frame_count in common_frames:
            trained_row = trained_by_frame[frame_count]
            per_frame_row = per_frame_by_frame[frame_count]
            if frame_count != common_frames[0]:
                if int(trained_row.get("interval_trace_entries", 0)) >= int(
                    per_frame_row.get("interval_trace_entries", 0)
                ):
                    errors.append(
                        f"trained interval entries must beat per-frame replay entries for frames={frame_count}"
                    )
                if int(trained_row.get("trace_count", 0)) >= int(per_frame_row.get("trace_count", 0)):
                    errors.append(f"trained trace count must stay below per-frame replay trace count for frames={frame_count}")
            for key in ("forward_ms", "backward_ms"):
                trained_value = trained_row.get(key)
                per_frame_value = per_frame_row.get(key)
                if trained_value is None or per_frame_value is None:
                    continue
                if not (_finite_positive(trained_value) and _finite_positive(per_frame_value)):
                    errors.append(f"{key} values must be finite and positive when timing is present")
        final_frame = trained_frames[-1]
        if final_frame not in per_frame_by_frame:
            errors.append(f"per-frame baseline missing final frame {final_frame}")
        else:
            final_trained = trained_rows[-1]
            final_per_frame = per_frame_by_frame[final_frame]
            if int(final_trained.get("interval_trace_entries", 0)) >= int(
                final_per_frame.get("interval_trace_entries", 0)
            ):
                errors.append("final trained interval entries must beat per-frame replay entries")
            if int(final_trained.get("trace_count", 0)) >= int(final_per_frame.get("trace_count", 0)):
                errors.append("final trained trace count must stay below per-frame replay trace count")
            for key in ("forward_ms", "backward_ms"):
                trained_value = final_trained.get(key)
                per_frame_value = final_per_frame.get(key)
                if trained_value is None or per_frame_value is None:
                    continue
                if not (_finite_positive(trained_value) and _finite_positive(per_frame_value)):
                    errors.append(f"{key} values must be finite and positive when timing is present")
                elif float(trained_value) >= float(per_frame_value):
                    errors.append(f"final trained {key} must beat per-frame replay timing")

    return errors


def assert_trained_high_motion_trace_scaling_report(report: dict[str, Any]) -> None:
    errors = verify_trained_high_motion_trace_scaling_report(report)
    if errors:
        raise AssertionError("trained high-motion trace scaling report failed:\n- " + "\n- ".join(errors))


def _summarize(rows: list[dict[str, Any]], train_row: dict[str, Any]) -> dict[str, Any]:
    by_label = {
        label: [row for row in rows if row["label"] == label]
        for label in sorted({str(row["label"]) for row in rows})
    }
    summary: dict[str, Any] = {
        "train_loss_decreased": bool(train_row.get("loss_decreased")),
        "train_start_loss": train_row.get("start_loss"),
        "train_end_loss": train_row.get("end_loss"),
        "train_tile_overflow_sum": train_row.get("tile_overflow_sum"),
        "max_fallback_fraction": max(float(row["fallback_fraction"]) for row in rows) if rows else None,
        "max_interval_to_dense_tile_pair_ratio": max(
            float(row["interval_to_dense_tile_pair_ratio"]) for row in rows
        )
        if rows
        else None,
    }
    for label, label_rows in by_label.items():
        summary[f"{label}_interval_entries"] = [row["interval_trace_entries"] for row in label_rows]
        summary[f"{label}_dense_tile_pairs"] = [row["dense_per_frame_tile_pairs"] for row in label_rows]
        summary[f"{label}_tile_pair_ratio_growth"] = _growth(
            [row["interval_to_dense_tile_pair_ratio"] for row in label_rows]
        )
        summary[f"{label}_interval_entry_growth"] = _growth([row["interval_trace_entries"] for row in label_rows])
        if all(row.get("forward_ms") is not None for row in label_rows):
            summary[f"{label}_forward_ms_growth"] = _growth([row["forward_ms"] for row in label_rows])
        if all(row.get("backward_ms") is not None for row in label_rows):
            summary[f"{label}_backward_ms_growth"] = _growth([row["backward_ms"] for row in label_rows])
    interval_rows = [row for row in rows if row["label"] == "trained_checkpoint"]
    per_frame_rows = [row for row in rows if row["label"] == "trained_checkpoint_per_frame"]
    if interval_rows and per_frame_rows:
        frames = sorted({int(row["frames"]) for row in interval_rows} & {int(row["frames"]) for row in per_frame_rows})
        forward_ratios = []
        backward_ratios = []
        interval_entry_ratios = []
        for frame_count in frames:
            interval = next(row for row in interval_rows if int(row["frames"]) == frame_count)
            per_frame = next(row for row in per_frame_rows if int(row["frames"]) == frame_count)
            if per_frame.get("forward_ms") not in (None, 0) and interval.get("forward_ms") is not None:
                forward_ratios.append(float(interval["forward_ms"]) / float(per_frame["forward_ms"]))
            if per_frame.get("backward_ms") not in (None, 0) and interval.get("backward_ms") is not None:
                backward_ratios.append(float(interval["backward_ms"]) / float(per_frame["backward_ms"]))
            if int(per_frame["interval_trace_entries"]) > 0:
                interval_entry_ratios.append(
                    float(interval["interval_trace_entries"]) / float(per_frame["interval_trace_entries"])
                )
        summary["trained_interval_vs_per_frame_forward_ms_ratios"] = forward_ratios
        summary["trained_interval_vs_per_frame_backward_ms_ratios"] = backward_ratios
        summary["trained_interval_vs_per_frame_interval_entry_ratios"] = interval_entry_ratios
    return summary


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    columns = (
        "label",
        "frames",
        "trace_count",
        "cell_count",
        "interval_trace_entries",
        "dense_trace_samples",
        "interval_to_dense_trace_sample_ratio",
        "dense_per_frame_tile_pairs",
        "interval_to_dense_tile_pair_ratio",
        "forward_ms",
        "backward_ms",
        "fallback_fraction",
        "velocity_mean_px_per_frame",
        "velocity_max_px_per_frame",
    )
    lines = [
        "# STAR UVT Trained High-Motion Trace Scaling",
        "",
        "This benchmark trains a tiny projective-interval STAR UVT feature model on",
        "the checked-in high-motion smoke video, reloads the saved checkpoint, and",
        "compiles its learned UVT tensors into projective trace-cell atlases.",
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


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    frame_counts = [int(part.strip()) for part in str(args.frame_counts).split(",") if part.strip()]
    if not frame_counts:
        raise ValueError("--frame-counts must include at least one integer")
    trained_frames = max(frame_counts)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not HIGH_MOTION_VIDEO.exists():
        return {"status": "skipped", "reason": f"missing high-motion video: {HIGH_MOTION_VIDEO}", "rows": []}
    if not torch.backends.mps.is_available():
        return {"status": "skipped", "reason": "MPS unavailable", "rows": []}
    if not has_projective_trace_cell_interval_metal() or not has_projective_trace_cell_interval_backward_metal():
        return {"status": "skipped", "reason": "projective interval Metal ops unavailable", "rows": []}

    checkpoint = args.out_dir / "trained_high_motion_checkpoint.pt"
    train_json = args.out_dir / "trained_high_motion_train.json"
    cfg = resolve_config(
        _base_config(
            frames=trained_frames,
            size=int(args.size),
            steps=int(args.steps),
            tube_count=int(args.tube_count),
            tile_capacity=int(args.tile_capacity),
            out_json=train_json,
            checkpoint=checkpoint,
        )
    )
    train_feature_config = feature_tube_render_config_from_cfg(cfg)
    train_render_cfg = _make_render_config(cfg=cfg, feature_config=train_feature_config, frame_count=trained_frames)
    _apply_metal_tile_env(train_render_cfg)
    started = time.perf_counter()
    if bool(args.verbose_trainer_output):
        train_row = feature_overfit_trainer.run_training(cfg)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            train_row = feature_overfit_trainer.run_training(cfg)
    train_elapsed_sec = time.perf_counter() - started

    run_metal_timing = bool(args.run_metal_timing)
    row_device = torch.device("mps" if run_metal_timing else "cpu")
    init_model = _build_init_model(cfg=cfg, device=row_device)
    trained_model = _load_trained_model(cfg=cfg, checkpoint=checkpoint, device=row_device)
    rows: list[dict[str, Any]] = []
    for frame_count in frame_counts:
        rows.append(
            _compile_geometry_row(
                label="init",
                model=init_model,
                cfg=cfg,
                frame_count=frame_count,
                trained_frames=trained_frames,
                run_metal_timing=run_metal_timing,
                timing_iterations=int(args.timing_iterations),
                timing_warmup=int(args.timing_warmup),
            )
        )
        rows.append(
            _compile_geometry_row(
                label="trained_checkpoint",
                model=trained_model,
                cfg=cfg,
                frame_count=frame_count,
                trained_frames=trained_frames,
                run_metal_timing=run_metal_timing,
                timing_iterations=int(args.timing_iterations),
                timing_warmup=int(args.timing_warmup),
            )
        )
        if bool(args.include_per_frame_baseline):
            rows.append(
                _compile_per_frame_baseline_row(
                    label="trained_checkpoint_per_frame",
                    model=trained_model,
                    cfg=cfg,
                    frame_count=frame_count,
                    trained_frames=trained_frames,
                    run_metal_timing=run_metal_timing,
                    timing_iterations=int(args.timing_iterations),
                    timing_warmup=int(args.timing_warmup),
                )
            )

    return {
        "status": "ok",
        "benchmark": "star_uvt_projective_trained_high_motion_trace_scaling",
        "source_video": str(HIGH_MOTION_VIDEO),
        "source_video_exists": bool(HIGH_MOTION_VIDEO.exists()),
        "trained_checkpoint": str(checkpoint),
        "train_json": str(train_json),
        "train_elapsed_sec": float(train_elapsed_sec),
        "frame_counts": frame_counts,
        "trained_frames": int(trained_frames),
        "size": int(args.size),
        "steps": int(args.steps),
        "tube_count": int(args.tube_count),
        "tile_capacity": int(args.tile_capacity),
        "run_metal_timing": run_metal_timing,
        "include_per_frame_baseline": bool(args.include_per_frame_baseline),
        "timing_iterations": int(args.timing_iterations),
        "timing_warmup": int(args.timing_warmup),
        "train": {
            "pass": bool(train_row.get("pass")),
            "loss_decreased": bool(train_row.get("loss_decreased")),
            "start_loss": train_row.get("start_loss"),
            "end_loss": train_row.get("end_loss"),
            "tile_overflow_sum": train_row.get("tile_overflow_sum"),
            "projective_interval_cache_rebuilds": train_row.get("projective_interval_cache_rebuilds"),
            "projective_interval_cache_live_updates": train_row.get("projective_interval_cache_live_updates"),
            "projective_interval_cache_staleness_checks": train_row.get(
                "projective_interval_cache_staleness_checks"
            ),
        },
        "summary": _summarize(rows, train_row),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-counts", default="4,8,16")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--tube-count", type=int, default=64)
    parser.add_argument("--tile-capacity", type=int, default=128)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-metal-timing", action="store_true")
    parser.add_argument("--include-per-frame-baseline", action="store_true")
    parser.add_argument("--timing-iterations", type=int, default=3)
    parser.add_argument("--timing-warmup", type=int, default=2)
    parser.add_argument("--verbose-trainer-output", action="store_true")
    parser.add_argument("--verify-report", type=Path, help="validate an existing summary.json without rerunning")
    args = parser.parse_args()

    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_trained_high_motion_trace_scaling_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_benchmark(args)
    if report.get("status") == "ok":
        assert_trained_high_motion_trace_scaling_report(report)
    json_path = args.out_dir / "summary.json"
    md_path = args.out_dir / "summary.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report, md_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
