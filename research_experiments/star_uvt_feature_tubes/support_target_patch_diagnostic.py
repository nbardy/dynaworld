from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from checkpoint_utils import load_checkpoint_mapping
from config_utils import load_config_file, path_or_none
try:
    from .report_artifacts import write_report_json, write_report_text
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import write_report_json, write_report_text
from star_uvt_checkpoints import load_star_training_checkpoint
from star_uvt_colorizers import build_feature_colorizer
from star_uvt_common import load_colorizer_init_checkpoint, load_training_sequence
from star_uvt_feature_config import resolve_config
from star_uvt_models import build_feature_tube_model
from star_uvt_render_configs import star_uvt_render_configs_from_cfg
from star_uvt_runtime import resolve_device as _resolve_device, sync_device as _sync_device
from star_uvt_sparse_visual_losses import _compose_sparse_visual_rgb, _gather_sparse_visual_rgb_values
from star_uvt_visibility_support import (
    _support_birth_split_sample_grid,
    _support_birth_split_sampled_tile_load,
    _support_birth_split_target_patch_pixel_ids_for_chunk,
    _support_birth_split_target_points,
)
from torch_gsplat_bridge_star_uvt.feature_rasterize import (
    chunked_uvt_config,
    render_uvt_feature_sparse_pixels_with_bins,
    shift_ma_for_frame_chunk,
)


ALPHA_THRESHOLDS = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75)


def _psnr(mse: float) -> float:
    if mse <= 0.0:
        return float("inf")
    return -10.0 * math.log10(float(mse))


def _load_checkpoint(path: Path, *, model: torch.nn.Module, colorizer: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    payload = load_checkpoint_mapping(path, map_location=device)
    model_state = payload.get("model")
    colorizer_state = payload.get("colorizer")
    if not isinstance(model_state, dict) or not isinstance(colorizer_state, dict):
        raise ValueError(f"Checkpoint {path} must contain model and colorizer states")
    model.load_state_dict(model_state)
    colorizer.load_state_dict(colorizer_state)
    row = payload.get("row")
    return row if isinstance(row, dict) else {}


def _load_final_case(config_path: Path) -> dict[str, Any]:
    cfg = resolve_config(load_config_file(config_path))
    device = _resolve_device(str(cfg["train"]["device"]))
    if device.type != "mps":
        raise RuntimeError("STAR UVT support-target patch diagnostic currently requires MPS")
    torch.manual_seed(int(cfg["train"]["seed"]))
    feature_config, uvt_config = star_uvt_render_configs_from_cfg(cfg)
    target_rgb = load_training_sequence(cfg, device).frames.contiguous()
    model = build_feature_tube_model(cfg, feature_config, device=device)
    colorizer = build_feature_colorizer(cfg["colorize"], feature_dim=feature_config.feature_dim, device=device)
    checkpoint = Path(cfg["output"]["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Expected checkpoint from config output.checkpoint: {checkpoint}")
    row = _load_checkpoint(checkpoint, model=model, colorizer=colorizer, device=device)
    model.eval()
    colorizer.eval()
    return {
        "cfg": cfg,
        "config_path": str(config_path),
        "checkpoint": str(checkpoint),
        "row": row,
        "device": device,
        "feature_config": feature_config,
        "uvt_config": uvt_config,
        "target_rgb": target_rgb,
        "model": model,
        "colorizer": colorizer,
    }


def _load_selection_model(case: dict[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    cfg = case["cfg"]
    feature_config = case["feature_config"]
    device = case["device"]
    model = build_feature_tube_model(cfg, feature_config, device=device)
    colorizer = build_feature_colorizer(cfg["colorize"], feature_dim=feature_config.feature_dim, device=device)
    optimizer = torch.optim.Adam([*model.parameters(), *colorizer.parameters()], lr=float(cfg["train"]["lr"]))
    resume_checkpoint = path_or_none(cfg["train"].get("resume_checkpoint"))
    resume_state: dict[str, Any] = {"path": None, "loaded": False}
    if resume_checkpoint is not None:
        resume_state = load_star_training_checkpoint(
            resume_checkpoint,
            model=model,
            colorizer=colorizer,
            optimizer=optimizer,
            device=device,
            resume_optimizer=False,
            resume_colorizer=bool(cfg["train"].get("resume_colorizer", True)),
        )
    init_checkpoint = path_or_none(cfg["colorize"].get("init_checkpoint"))
    colorizer_init_state = {"path": None, "loaded": False}
    if init_checkpoint is not None:
        colorizer_init_state = load_colorizer_init_checkpoint(init_checkpoint, colorizer=colorizer, device=device)
    model.eval()
    colorizer.eval()
    return model, colorizer, {"resume": resume_state, "colorizer_init": colorizer_init_state}


def _recompute_support_birth_target_points(case: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    cfg = case["cfg"]
    support_cfg = cfg["support_birth_split"]
    if not bool(support_cfg.get("enabled", False)):
        raise ValueError("support_birth_split.enabled must be true for support-target patch diagnostic")
    model, colorizer, selection_state = _load_selection_model(case)
    target_rgb = case["target_rgb"]
    device = case["device"]
    feature_config = case["feature_config"]
    uvt_config = case["uvt_config"]
    sampled_alpha: torch.Tensor | None = None
    sampled_residual: torch.Tensor | None = None
    sampled_tile_load: torch.Tensor | None = None
    alpha_sample_ms = 0.0
    source = str(support_cfg["target_point_source"])
    if source != "top_brightness":
        _sync_device(device)
        started = time.perf_counter()
        _frame_ids, y_ids, x_ids, pixel_ids = _support_birth_split_sample_grid(
            frames=feature_config.frames,
            height=feature_config.height,
            width=feature_config.width,
            frame_stride=int(support_cfg["frame_stride"]),
            grid_stride=int(support_cfg["grid_stride"]),
            device=device,
        )
        ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()
        needs_residual_target = "residual" in source
        if needs_residual_target and getattr(colorizer, "view_condition", "none") != "none":
            raise RuntimeError("support_birth_split residual target sources require colorize.view_condition='none'")
        feature_for_sample = feature if needs_residual_target else torch.zeros(
            (int(ma.shape[0]), 1),
            dtype=torch.float32,
            device=device,
        )
        alpha_render = render_uvt_feature_sparse_pixels_with_bins(
            ma,
            q_uvt,
            depth0.detach(),
            depth_beta.detach(),
            opacity,
            feature_for_sample,
            pixel_ids,
            uvt_config,
        )
        sampled_alpha = alpha_render.alpha_values.reshape(-1, int(y_ids.numel()), int(x_ids.numel())).contiguous()
        if needs_residual_target:
            with torch.no_grad():
                target_values = _gather_sparse_visual_rgb_values(target_rgb, pixel_ids)
                pred_values = _compose_sparse_visual_rgb(
                    alpha_render.feature_values.detach(),
                    alpha_render.alpha_values.detach(),
                    colorizer,
                    composition="black",
                )
                sampled_residual = (pred_values - target_values).abs().mean(dim=1).reshape_as(sampled_alpha)
        sampled_tile_load = _support_birth_split_sampled_tile_load(
            alpha_render.tile_counts,
            frames=feature_config.frames,
            height=feature_config.height,
            width=feature_config.width,
            frame_stride=int(support_cfg["frame_stride"]),
            grid_stride=int(support_cfg["grid_stride"]),
            tile_x=int(uvt_config.tile_x),
            tile_y=int(uvt_config.tile_y),
            tile_t=int(uvt_config.tile_t),
        )
        _sync_device(device)
        alpha_sample_ms = (time.perf_counter() - started) * 1000.0

    target_points, meta = _support_birth_split_target_points(
        target_rgb,
        target_point_source=source,
        target_top_fraction=float(support_cfg["target_top_fraction"]),
        max_points=int(support_cfg["max_points"]),
        grid_stride=int(support_cfg["grid_stride"]),
        frame_stride=int(support_cfg["frame_stride"]),
        device=device,
        sampled_alpha=sampled_alpha,
        sampled_residual=sampled_residual,
        sampled_tile_load=sampled_tile_load,
        tile_capacity=int(cfg["feature_uvt"]["tile_capacity"]),
        footprint_radius_px=float(support_cfg["support_radius_px"]),
    )
    meta["recomputed_alpha_sample_ms"] = alpha_sample_ms
    meta["selection_state"] = selection_state
    return target_points, meta


def _limit_points(points: torch.Tensor, max_points: int) -> torch.Tensor:
    if int(max_points) <= 0 or int(points.shape[0]) <= int(max_points):
        return points
    select = (
        torch.linspace(0, int(points.shape[0]) - 1, int(max_points), device=points.device)
        .round()
        .to(torch.int64)
    )
    return points.index_select(0, select).contiguous()


def _selected_tube_ids(case: dict[str, Any]) -> torch.Tensor:
    support = case["row"].get("support_birth_split", {})
    ids = support.get("selected_tube_ids", []) if isinstance(support, dict) else []
    return torch.tensor([int(item) for item in ids], dtype=torch.int64, device=case["device"])


def _quadratic(q_uvt: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return (
        q_uvt[..., 0] * delta[..., 0] * delta[..., 0]
        + 2.0 * q_uvt[..., 1] * delta[..., 0] * delta[..., 1]
        + 2.0 * q_uvt[..., 2] * delta[..., 0] * delta[..., 2]
        + q_uvt[..., 3] * delta[..., 1] * delta[..., 1]
        + 2.0 * q_uvt[..., 4] * delta[..., 1] * delta[..., 2]
        + q_uvt[..., 5] * delta[..., 2] * delta[..., 2]
    )


def _selected_tube_point_support(case: dict[str, Any], target_points: torch.Tensor, selected_ids: torch.Tensor) -> dict[str, Any]:
    if int(selected_ids.numel()) == 0 or int(target_points.shape[0]) == 0:
        return {
            "selected_tube_count": int(selected_ids.numel()),
            "point_count": int(target_points.shape[0]),
            "max_alpha_mean": 0.0,
            "max_alpha_max": 0.0,
            "min_qv_mean": None,
            "min_qv_min": None,
            "fraction_over_alpha_threshold": 0.0,
            "fraction_over_0.01": 0.0,
            "fraction_over_0.05": 0.0,
            "fraction_over_0.1": 0.0,
        }
    ma, q_uvt, _depth0, _depth_beta, opacity, _feature = case["model"].tensors()
    selected_ids = selected_ids.to(device=ma.device, dtype=torch.int64)
    points = target_points.to(device=ma.device, dtype=ma.dtype)
    selected_ma = ma.index_select(0, selected_ids)
    selected_q = q_uvt.index_select(0, selected_ids)
    selected_opacity = opacity.index_select(0, selected_ids)
    delta = points[:, None, :] - selected_ma[None, :, :]
    qv = _quadratic(selected_q[None, :, :], delta)
    alpha = selected_opacity[None, :] * torch.exp(torch.clamp(-0.5 * qv, min=-80.0, max=0.0))
    max_alpha, _max_idx = alpha.max(dim=1)
    min_qv, _min_idx = qv.min(dim=1)
    threshold = float(case["cfg"]["feature_uvt"]["alpha_threshold"])
    return {
        "selected_tube_count": int(selected_ids.numel()),
        "point_count": int(target_points.shape[0]),
        "max_alpha_mean": float(max_alpha.mean().detach().cpu().item()),
        "max_alpha_max": float(max_alpha.max().detach().cpu().item()),
        "min_qv_mean": float(min_qv.mean().detach().cpu().item()),
        "min_qv_min": float(min_qv.min().detach().cpu().item()),
        "fraction_over_alpha_threshold": float((max_alpha >= threshold).float().mean().detach().cpu().item()),
        "fraction_over_0.01": float((max_alpha >= 0.01).float().mean().detach().cpu().item()),
        "fraction_over_0.05": float((max_alpha >= 0.05).float().mean().detach().cpu().item()),
        "fraction_over_0.1": float((max_alpha >= 0.1).float().mean().detach().cpu().item()),
    }


def _chunk_render_inputs(
    case: dict[str, Any],
    frame_start: int,
    chunk_frames: int,
    *,
    opacity_mode: str,
    selected_ids: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], Any]:
    model = case["model"]
    uvt_config = case["uvt_config"]
    ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()
    if opacity_mode not in {"normal", "hide_selected", "selected_only"}:
        raise ValueError(f"unknown opacity mode {opacity_mode!r}")
    if opacity_mode != "normal":
        adjusted = torch.zeros_like(opacity) if opacity_mode == "selected_only" else opacity.clone()
        if int(selected_ids.numel()) > 0:
            if opacity_mode == "hide_selected":
                adjusted.index_fill_(0, selected_ids.to(device=opacity.device), 0.0)
            else:
                adjusted.index_copy_(0, selected_ids.to(device=opacity.device), opacity.index_select(0, selected_ids))
        opacity = adjusted
    if chunk_frames == int(uvt_config.frames):
        return (ma, q_uvt, depth0.detach(), depth_beta.detach(), opacity, feature), uvt_config
    ma_chunk = shift_ma_for_frame_chunk(
        ma,
        global_frames=int(uvt_config.frames),
        frame_start=frame_start,
        chunk_frames=chunk_frames,
    )
    return (
        ma_chunk,
        q_uvt,
        depth0.detach(),
        depth_beta.detach(),
        opacity,
        feature,
    ), chunked_uvt_config(uvt_config, chunk_frames=chunk_frames)


def _new_accumulator() -> dict[str, Any]:
    return {
        "pixel_sse": {"black": 0.0, "forced": 0.0, "target_background": 0.0},
        "patch_sse": {"black": 0.0, "forced": 0.0, "target_background": 0.0},
        "pixel_elems": 0,
        "patch_elems": 0,
        "alpha_sum": 0.0,
        "alpha_sq_sum": 0.0,
        "alpha_max": 0.0,
        "alpha_count": 0,
        "patch_alpha_sum": 0.0,
        "patch_alpha_sq_sum": 0.0,
        "patch_alpha_count": 0,
        "alpha_threshold_counts": {threshold: 0 for threshold in ALPHA_THRESHOLDS},
        "patch_alpha_threshold_counts": {threshold: 0 for threshold in ALPHA_THRESHOLDS},
        "cell_count": 0,
        "pixel_count": 0,
        "render_ms": 0.0,
    }


def _colorize_sparse(feature_values: torch.Tensor, colorizer: torch.nn.Module) -> torch.Tensor:
    splat_rgb = colorizer(feature_values.transpose(0, 1).unsqueeze(0).unsqueeze(-1))
    return splat_rgb.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()


def _accumulate_patch_metrics(
    acc: dict[str, Any],
    *,
    splat_rgb: torch.Tensor,
    alpha: torch.Tensor,
    target_values: torch.Tensor,
    cell_count: int,
    patch_shape: tuple[int, int],
    render_ms: float,
) -> None:
    if int(cell_count) <= 0:
        return
    patch_area = int(patch_shape[0]) * int(patch_shape[1])
    if int(splat_rgb.shape[0]) != int(cell_count) * patch_area:
        raise ValueError("sparse patch values do not match cell_count and patch_shape")
    alpha_col = alpha.to(dtype=splat_rgb.dtype).unsqueeze(1)
    pixel_preds = {
        "black": alpha_col * splat_rgb,
        "forced": splat_rgb,
        "target_background": target_values + alpha_col * (splat_rgb - target_values),
    }
    target_patch = target_values.reshape(int(cell_count), patch_area, 3).mean(dim=1)
    alpha_patch = alpha.reshape(int(cell_count), patch_area).mean(dim=1)
    for key, pred in pixel_preds.items():
        acc["pixel_sse"][key] += float((pred - target_values).square().sum().detach().cpu().item())
        pred_patch = pred.reshape(int(cell_count), patch_area, 3).mean(dim=1)
        acc["patch_sse"][key] += float((pred_patch - target_patch).square().sum().detach().cpu().item())
    acc["pixel_elems"] += int(target_values.numel())
    acc["patch_elems"] += int(cell_count) * 3
    acc["alpha_sum"] += float(alpha.sum().detach().cpu().item())
    acc["alpha_sq_sum"] += float(alpha.square().sum().detach().cpu().item())
    acc["alpha_max"] = max(float(acc["alpha_max"]), float(alpha.max().detach().cpu().item()))
    acc["alpha_count"] += int(alpha.numel())
    acc["patch_alpha_sum"] += float(alpha_patch.sum().detach().cpu().item())
    acc["patch_alpha_sq_sum"] += float(alpha_patch.square().sum().detach().cpu().item())
    acc["patch_alpha_count"] += int(alpha_patch.numel())
    for threshold in ALPHA_THRESHOLDS:
        acc["alpha_threshold_counts"][threshold] += int((alpha > threshold).sum().detach().cpu().item())
        acc["patch_alpha_threshold_counts"][threshold] += int((alpha_patch > threshold).sum().detach().cpu().item())
    acc["cell_count"] += int(cell_count)
    acc["pixel_count"] += int(alpha.numel())
    acc["render_ms"] += float(render_ms)


def _finalize_accumulator(acc: dict[str, Any]) -> dict[str, Any]:
    if int(acc["pixel_elems"]) <= 0 or int(acc["patch_elems"]) <= 0:
        raise RuntimeError("diagnostic collected no support-target patch samples")
    alpha_count = max(int(acc["alpha_count"]), 1)
    patch_alpha_count = max(int(acc["patch_alpha_count"]), 1)
    alpha_mean = float(acc["alpha_sum"]) / float(alpha_count)
    patch_alpha_mean = float(acc["patch_alpha_sum"]) / float(patch_alpha_count)
    alpha_var = max(float(acc["alpha_sq_sum"]) / float(alpha_count) - alpha_mean * alpha_mean, 0.0)
    patch_alpha_var = max(
        float(acc["patch_alpha_sq_sum"]) / float(patch_alpha_count) - patch_alpha_mean * patch_alpha_mean,
        0.0,
    )
    return {
        "pixel_psnr": {
            key: _psnr(float(value) / float(acc["pixel_elems"]))
            for key, value in acc["pixel_sse"].items()
        },
        "patch_psnr": {
            key: _psnr(float(value) / float(acc["patch_elems"]))
            for key, value in acc["patch_sse"].items()
        },
        "pixel_mse": {
            key: float(value) / float(acc["pixel_elems"])
            for key, value in acc["pixel_sse"].items()
        },
        "patch_mse": {
            key: float(value) / float(acc["patch_elems"])
            for key, value in acc["patch_sse"].items()
        },
        "alpha_mean": alpha_mean,
        "alpha_std": math.sqrt(alpha_var),
        "alpha_max": float(acc["alpha_max"]),
        "patch_alpha_mean": patch_alpha_mean,
        "patch_alpha_std": math.sqrt(patch_alpha_var),
        "cell_count": int(acc["cell_count"]),
        "pixel_count": int(acc["pixel_count"]),
        "render_ms": float(acc["render_ms"]),
        "alpha_thresholds": {
            str(threshold): {
                "pixel_fraction": int(acc["alpha_threshold_counts"][threshold]) / float(alpha_count),
                "patch_fraction": int(acc["patch_alpha_threshold_counts"][threshold]) / float(patch_alpha_count),
            }
            for threshold in ALPHA_THRESHOLDS
        },
    }


def _analyze_case(
    label: str,
    config_path: Path,
    *,
    max_points: int | None,
    patch_shape: tuple[int, int] | None,
) -> dict[str, Any]:
    case = _load_final_case(config_path)
    cfg = case["cfg"]
    feature_config = case["feature_config"]
    device = case["device"]
    support_cfg = cfg["support_birth_split"]
    selected_patch_shape = patch_shape
    if selected_patch_shape is None:
        selected_patch_shape = tuple(int(item) for item in support_cfg.get("target_area_patch_shape", (2, 2)))
    selected_max_points = (
        int(max_points)
        if max_points is not None
        else int(support_cfg.get("target_area_max_points", support_cfg.get("max_points", 1024)))
    )
    target_points, target_meta = _recompute_support_birth_target_points(case)
    selected_points = _limit_points(target_points, selected_max_points)
    selected_ids = _selected_tube_ids(case)
    selected_support = _selected_tube_point_support(case, selected_points, selected_ids)
    chunk_size_cfg = cfg["train"]["frame_chunk_size"]
    chunk_size = int(feature_config.frames) if chunk_size_cfg is None else min(int(chunk_size_cfg), int(feature_config.frames))
    modes = {
        "normal": _new_accumulator(),
        "hide_selected": _new_accumulator(),
        "selected_only": _new_accumulator(),
    }
    _sync_device(device)
    with torch.no_grad():
        for frame_start in range(0, int(feature_config.frames), chunk_size):
            chunk_frames = min(chunk_size, int(feature_config.frames) - frame_start)
            pixel_ids, cell_count = _support_birth_split_target_patch_pixel_ids_for_chunk(
                selected_points,
                frames=int(feature_config.frames),
                height=int(feature_config.height),
                width=int(feature_config.width),
                frame_start=frame_start,
                chunk_frames=chunk_frames,
                patch_shape=selected_patch_shape,
                device=device,
            )
            if int(pixel_ids.numel()) == 0:
                continue
            target_chunk = case["target_rgb"][frame_start : frame_start + chunk_frames]
            target_values = _gather_sparse_visual_rgb_values(target_chunk, pixel_ids)
            for mode, acc in modes.items():
                render_inputs, render_config = _chunk_render_inputs(
                    case,
                    frame_start,
                    chunk_frames,
                    opacity_mode=mode,
                    selected_ids=selected_ids,
                )
                _sync_device(device)
                started = time.perf_counter()
                render = render_uvt_feature_sparse_pixels_with_bins(*render_inputs, pixel_ids, render_config)
                _sync_device(device)
                render_ms = (time.perf_counter() - started) * 1000.0
                _accumulate_patch_metrics(
                    acc,
                    splat_rgb=_colorize_sparse(render.feature_values, case["colorizer"]),
                    alpha=render.alpha_values,
                    target_values=target_values,
                    cell_count=cell_count,
                    patch_shape=selected_patch_shape,
                    render_ms=render_ms,
                )
    finalized_modes = {mode: _finalize_accumulator(acc) for mode, acc in modes.items()}
    support_row = case["row"].get("support_birth_split", {})
    stored_target_meta = support_row.get("target_point_meta", {}) if isinstance(support_row, dict) else {}
    return {
        "label": label,
        "config_path": case["config_path"],
        "checkpoint": case["checkpoint"],
        "frames": int(feature_config.frames),
        "size": int(feature_config.height),
        "chunk_size": int(chunk_size),
        "patch_shape": [int(selected_patch_shape[0]), int(selected_patch_shape[1])],
        "target_point_count": int(target_points.shape[0]),
        "selected_target_point_count": int(selected_points.shape[0]),
        "selected_tube_count": int(selected_ids.numel()),
        "selected_tube_ids": [int(item) for item in selected_ids.detach().cpu().tolist()],
        "selected_tube_point_support": selected_support,
        "target_point_meta": target_meta,
        "stored_target_point_meta": stored_target_meta,
        "modes": finalized_modes,
        "support_birth_split": support_row,
    }


def _parse_case(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        return label.strip(), Path(path)
    path = Path(raw)
    return path.stem, path


def _make_read(cases: list[dict[str, Any]]) -> str:
    reads: list[str] = []
    for case in cases:
        normal = case["modes"]["normal"]
        hidden = case["modes"]["hide_selected"]
        selected_only = case["modes"]["selected_only"]
        black = normal["patch_psnr"]["black"]
        forced = normal["patch_psnr"]["forced"]
        oracle = normal["patch_psnr"]["target_background"]
        alpha = normal["patch_alpha_mean"]
        hidden_black = hidden["patch_psnr"]["black"]
        selected_alpha = selected_only["patch_alpha_mean"]
        selected_support = case["selected_tube_point_support"]
        if forced > black + 3.0 and alpha < 0.75:
            reads.append(
                f"`{case['label']}` selected patches are still alpha/composition limited: "
                f"patch normal/forced/oracle PSNR is {black:.3f}/{forced:.3f}/{oracle:.3f}, "
                f"patch alpha mean is {alpha:.3f}."
            )
        else:
            reads.append(
                f"`{case['label']}` selected patches are not mainly rescued by forcing alpha: "
                f"patch normal/forced/oracle PSNR is {black:.3f}/{forced:.3f}/{oracle:.3f}."
            )
        if abs(black - hidden_black) < 0.25 and selected_alpha < 0.1:
            reads.append(
                f"Hiding selected birth tubes barely changes `{case['label']}` "
                f"({hidden_black:.3f} patch PSNR), and selected-only patch alpha is {selected_alpha:.3f}; "
                "born tubes are not carrying much selected-patch mass. "
                f"Analytic selected-tube max alpha is mean/max "
                f"{selected_support['max_alpha_mean']:.4f}/{selected_support['max_alpha_max']:.4f}, "
                f"with {selected_support['fraction_over_alpha_threshold']:.1%} above the renderer threshold."
            )
        else:
            reads.append(
                f"Selected birth tubes affect `{case['label']}` locally: hide-selected patch PSNR is "
                f"{hidden_black:.3f}, selected-only patch alpha is {selected_alpha:.3f}."
            )
    if len(cases) > 1:
        normal_values = [case["modes"]["normal"]["patch_psnr"]["black"] for case in cases]
        spread = max(normal_values) - min(normal_values)
        if spread < 0.5:
            reads.append(
                f"Across cases, selected-patch normal PSNR spread is only {spread:.3f}dB; "
                "the target-init/alpha/area variants are not separating strongly on their own selected patches."
            )
    return " ".join(reads)


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    rows = [
        [
            "label",
            "mode",
            "patch normal",
            "patch forced",
            "patch target-bg",
            "patch alpha",
            "patch alpha>0.1",
            "sel alpha mean/max",
            "sel above thresh",
            "pixel normal",
            "cells",
        ]
    ]
    for case in result["cases"]:
        for mode in ("normal", "hide_selected", "selected_only"):
            data = case["modes"][mode]
            support = case["selected_tube_point_support"]
            rows.append(
                [
                    case["label"],
                    mode,
                    f"{data['patch_psnr']['black']:.3f}",
                    f"{data['patch_psnr']['forced']:.3f}",
                    f"{data['patch_psnr']['target_background']:.3f}",
                    f"{data['patch_alpha_mean']:.4f}",
                    f"{data['alpha_thresholds']['0.1']['patch_fraction']:.3f}",
                    (
                        ""
                        if mode != "normal"
                        else f"{support['max_alpha_mean']:.4f}/{support['max_alpha_max']:.4f}"
                    ),
                    "" if mode != "normal" else f"{support['fraction_over_alpha_threshold']:.3f}",
                    f"{data['pixel_psnr']['black']:.3f}",
                    str(data["cell_count"]),
                ]
            )
    widths = [max(len(row[idx]) for row in rows) for idx in range(len(rows[0]))]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) + " |"

    lines = [
        "# STAR UVT Support-Target Patch Diagnostic",
        "",
        f"Date: {result['date']}",
        "",
        "## Purpose",
        "",
        "Measure the exact support-birth target patches before the next shader change.",
        "`normal` is the final checkpoint's black-background composite on those",
        "patches. `forced` ignores alpha. `target-bg` composites over the target",
        "background and is therefore an oracle for black holes. `hide_selected`",
        "zeros the support-birth tube IDs recorded in the checkpoint row.",
        "`selected_only` zeros all other tubes.",
        "",
        "## Results",
        "",
        fmt(rows[0]),
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(fmt(row) for row in rows[1:])
    lines.extend(["", "## Read", "", result["read"], "", "## Inputs", ""])
    for case in result["cases"]:
        lines.extend(
            [
                f"- `{case['label']}` config: `{case['config_path']}`",
                f"- `{case['label']}` checkpoint: `{case['checkpoint']}`",
                f"- `{case['label']}` selected target points: `{case['selected_target_point_count']}`",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_report_text(path, "\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True, help="label=config.jsonc or config.jsonc")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--date", default="2026-05-26")
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--patch-shape", default=None, help="Optional H,W override, for example 2,2")
    args = parser.parse_args()

    patch_shape = None
    if args.patch_shape:
        parts = [int(part) for part in str(args.patch_shape).split(",")]
        if len(parts) != 2:
            raise ValueError("--patch-shape must contain exactly two comma-separated integers")
        patch_shape = (parts[0], parts[1])
    cases = [
        _analyze_case(label, path, max_points=args.max_points, patch_shape=patch_shape)
        for label, path in (_parse_case(raw) for raw in args.case)
    ]
    result = {
        "date": args.date,
        "max_points": args.max_points,
        "patch_shape_override": None if patch_shape is None else list(patch_shape),
        "cases": cases,
    }
    result["read"] = _make_read(cases)
    write_report_json(Path(args.out_json), result)
    _write_markdown(Path(args.out_md), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
