from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


try:
    from .report_artifacts import ROOT, summary_stats, write_report_json, write_report_text
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import ROOT, summary_stats, write_report_json, write_report_text
from research_experiments.star_uvt_feature_tubes.star_uvt_feature1_wholegraph_profile import (
    _chunk_render_inputs,
    _load_case,
)
from torch_gsplat_bridge_star_uvt.feature_rasterize import (
    direct_atomic_feature_backward,
    direct_atomic_feature_sparse_pixels_backward_cached_bins,
    render_uvt_feature_tubes,
    render_uvt_feature_tubes_autograd,
)
from star_uvt_common import target_grid_slice_for_render_chunk as _target_grid_slice_for_render_chunk
from star_uvt_feature_losses import _feature_target_loss, _pack_sparse_image_vjp
from star_uvt_feature_targets import _adapt_render_to_feature_target
from star_uvt_runtime import sync_device as _sync_device
from star_uvt_render_modes import backward_mode_for_feature_render_mode
from star_uvt_schedules import (
    _feature_target_weight_schedule,
    _feature_target_weights_for_step,
)
from star_uvt_sparse_grid import _pack_sparse_target_grid_vjp
from star_uvt_tile_stats import _tile_load_stats


DEFAULT_CONFIG = (
    ROOT
    / "src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_checkpoint_media.jsonc"
)
DEFAULT_OUT_BASE = ROOT / "outputs/benchmarks/2026-05-19_star_uvt_targetgrid_vjp_bridge_profile"
IMAGE_VJP_MODES = {"autograd", "analytic", "analytic_sparse_pixels", "analytic_sparse_grid"}


def _zero_grads(module: nn.Module) -> None:
    for param in module.parameters():
        param.grad = None


def _collect_grads(prefix: str, module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}.{name}": param.grad.detach().cpu().clone()
        for name, param in module.named_parameters()
        if param.grad is not None
    }


def _grad_comparison(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, Any]:
    names = sorted(set(a) | set(b))
    rows: list[dict[str, Any]] = []
    max_abs = 0.0
    max_rel = 0.0
    missing: list[str] = []
    for name in names:
        if name not in a or name not in b:
            missing.append(name)
            continue
        diff = (a[name] - b[name]).abs()
        abs_err = float(diff.max().item()) if diff.numel() else 0.0
        denom = float(torch.maximum(a[name].abs(), b[name].abs()).max().item()) if diff.numel() else 0.0
        rel_err = 0.0 if denom <= 1.0e-12 else abs_err / denom
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, rel_err)
        rows.append(
            {
                "name": name,
                "max_abs_error": abs_err,
                "max_rel_error": rel_err,
                "baseline_norm": float(a[name].norm().item()),
                "bridge_norm": float(b[name].norm().item()),
            }
        )
    return {
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "missing": missing,
        "rows": rows,
    }


def _target_loss_context(case: dict[str, Any], selected_global_step: int) -> dict[str, Any]:
    cfg = case["cfg"]
    target_feature = case["target_feature"]
    rgb_probe = case["rgb_probe"]
    rgb_probe_target = case["rgb_probe_target"]
    stage = _feature_target_weights_for_step(_feature_target_weight_schedule(cfg), selected_global_step)
    if stage.rgb_probe_loss_weight > 0.0 and (rgb_probe is None or rgb_probe_target is None):
        raise ValueError("rgb_probe_loss_weight > 0 requires a loaded rgb_probe and target")
    return {
        "feature_loss_weight": float(stage.loss_weight),
        "rgb_probe_loss_weight": float(stage.rgb_probe_loss_weight),
        "feature_loss_type": str(cfg["feature_target"]["loss_type"]),
        "total_feature_loss_elems": int(target_feature.numel),
        "total_rgb_probe_loss_elems": 0 if rgb_probe_target is None else int(rgb_probe_target.numel()),
    }


def _loss_from_feature_image(
    *,
    feature_image: torch.Tensor,
    case: dict[str, Any],
    loss_context: dict[str, Any],
    frame_start: int,
    chunk_frames: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_feature = case["target_feature"]
    rgb_probe = case["rgb_probe"]
    rgb_probe_target = case["rgb_probe_target"]
    feature_config = case["feature_config"]
    loss = feature_image.new_zeros(())
    timings = {
        "target_grid_prep_ms": 0.0,
        "feature_loss_forward_ms": 0.0,
        "rgb_probe_loss_forward_ms": 0.0,
    }

    _sync_device(case["device"])
    t0 = time.perf_counter()
    target_feature_chunk = target_feature.chunk(frame_start, chunk_frames)
    rendered_target_grid = _adapt_render_to_feature_target(
        feature_image,
        target_shape=tuple(int(item) for item in target_feature_chunk.shape),
        mode=target_feature.grid_mode,
    )
    _sync_device(case["device"])
    t1 = time.perf_counter()
    timings["target_grid_prep_ms"] = (t1 - t0) * 1000.0

    if loss_context["feature_loss_weight"] > 0.0:
        feature_loss_t0 = time.perf_counter()
        feature_loss = _feature_target_loss(
            rendered_target_grid,
            target_feature_chunk,
            str(loss_context["feature_loss_type"]),
        ) / float(loss_context["total_feature_loss_elems"])
        loss = loss + float(loss_context["feature_loss_weight"]) * feature_loss
        _sync_device(case["device"])
        feature_loss_t1 = time.perf_counter()
        timings["feature_loss_forward_ms"] = (feature_loss_t1 - feature_loss_t0) * 1000.0

    if rgb_probe is not None and loss_context["rgb_probe_loss_weight"] > 0.0:
        if rgb_probe_target is None:
            raise RuntimeError("RGB probe target is missing")
        probe_t0 = time.perf_counter()
        target_start, target_frames = _target_grid_slice_for_render_chunk(
            target_frames=int(rgb_probe_target.shape[0]),
            render_frames=int(feature_config.frames),
            frame_start=frame_start,
            chunk_frames=chunk_frames,
        )
        target_rgb_probe_chunk = rgb_probe_target[target_start : target_start + target_frames]
        rgb_probe_pred = rgb_probe(rendered_target_grid)
        rgb_probe_loss = (
            (rgb_probe_pred - target_rgb_probe_chunk).square().sum()
            / float(loss_context["total_rgb_probe_loss_elems"])
        )
        loss = loss + float(loss_context["rgb_probe_loss_weight"]) * rgb_probe_loss
        _sync_device(case["device"])
        probe_t1 = time.perf_counter()
        timings["rgb_probe_loss_forward_ms"] = (probe_t1 - probe_t0) * 1000.0
    return loss, timings


def _gelu_derivative_exact(x: torch.Tensor) -> torch.Tensor:
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    inv_sqrt2pi = 1.0 / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(x * inv_sqrt2))
    pdf = torch.exp(-0.5 * x.square()) * inv_sqrt2pi
    return cdf + x * pdf


def _manual_probe_loss_and_grid_grad(
    rgb_probe: nn.Module,
    rendered_target_grid: torch.Tensor,
    target_rgb_probe_chunk: torch.Tensor,
    *,
    total_rgb_probe_loss_elems: int,
    loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if getattr(rgb_probe, "pre_norm", None) is not None:
        raise ValueError("analytic probe VJP currently requires rgb_probe pre_norm=false")
    if getattr(rgb_probe, "activation", None) != "sigmoid":
        raise ValueError("analytic probe VJP currently requires rgb_probe activation=sigmoid")
    if getattr(rgb_probe, "view_condition", "none") != "none":
        raise ValueError("analytic probe VJP currently requires rgb_probe view_condition=none")
    net = getattr(rgb_probe, "net", None)
    if not isinstance(net, nn.Sequential) or len(net) != 3:
        raise ValueError("analytic probe VJP currently requires hidden Conv2d -> GELU -> Conv2d probe")
    conv1, activation, conv2 = net
    if not isinstance(conv1, nn.Conv2d) or not isinstance(activation, nn.GELU) or not isinstance(conv2, nn.Conv2d):
        raise ValueError("analytic probe VJP currently requires hidden Conv2d -> GELU -> Conv2d probe")
    if conv1.kernel_size != (1, 1) or conv2.kernel_size != (1, 1):
        raise ValueError("analytic probe VJP currently requires 1x1 convolutions")
    hidden_pre = F.conv2d(rendered_target_grid, conv1.weight, conv1.bias)
    hidden = F.gelu(hidden_pre)
    logits = F.conv2d(hidden, conv2.weight, conv2.bias)
    rgb = torch.sigmoid(logits)
    diff = rgb - target_rgb_probe_chunk
    loss = diff.square().sum() / float(total_rgb_probe_loss_elems)
    grad_rgb = (2.0 * float(loss_weight) / float(total_rgb_probe_loss_elems)) * diff
    grad_logits = grad_rgb * rgb * (1.0 - rgb)
    grad_hidden = F.conv_transpose2d(grad_logits, conv2.weight)
    grad_hidden_pre = grad_hidden * _gelu_derivative_exact(hidden_pre)
    grad_grid = F.conv_transpose2d(grad_hidden_pre, conv1.weight).contiguous()
    return loss, grad_grid


def _render_grid_vjp_to_feature_image(
    grad_target_grid: torch.Tensor,
    *,
    input_shape: tuple[int, int, int, int],
    mode: str,
) -> torch.Tensor:
    chunk_frames, feature_dim, height, width = input_shape
    output_size = (
        int(grad_target_grid.shape[0]),
        int(grad_target_grid.shape[2]),
        int(grad_target_grid.shape[3]),
    )
    grad_out = grad_target_grid.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    input_size = (1, feature_dim, chunk_frames, height, width)
    if mode == "trilinear":
        grad_in = torch.ops.aten.upsample_trilinear3d_backward(
            grad_out,
            output_size,
            input_size,
            False,
            None,
            None,
            None,
        )
    elif mode == "nearest":
        grad_in = torch.ops.aten.upsample_nearest3d_backward(
            grad_out,
            output_size,
            input_size,
            None,
            None,
            None,
        )
    else:
        raise ValueError(f"Unsupported feature target temporal_spatial_adapter={mode!r}.")
    return grad_in[0].permute(1, 0, 2, 3).contiguous()


def _manual_loss_and_feature_vjp(
    *,
    feature_image: torch.Tensor,
    case: dict[str, Any],
    loss_context: dict[str, Any],
    frame_start: int,
    chunk_frames: int,
    sparse_grid: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, Any, dict[str, float]]:
    target_feature = case["target_feature"]
    rgb_probe = case["rgb_probe"]
    rgb_probe_target = case["rgb_probe_target"]
    feature_config = case["feature_config"]
    timings = {
        "target_grid_prep_ms": 0.0,
        "feature_loss_forward_ms": 0.0,
        "rgb_probe_loss_forward_ms": 0.0,
        "image_vjp_backward_ms": 0.0,
    }
    _sync_device(case["device"])
    t0 = time.perf_counter()
    target_feature_chunk = target_feature.chunk(frame_start, chunk_frames)
    rendered_target_grid = _adapt_render_to_feature_target(
        feature_image,
        target_shape=tuple(int(item) for item in target_feature_chunk.shape),
        mode=target_feature.grid_mode,
    )
    _sync_device(case["device"])
    t1 = time.perf_counter()
    timings["target_grid_prep_ms"] = (t1 - t0) * 1000.0
    loss = feature_image.new_zeros(())
    grad_target_grid = torch.zeros_like(rendered_target_grid)
    if loss_context["feature_loss_weight"] > 0.0:
        feature_t0 = time.perf_counter()
        diff = rendered_target_grid - target_feature_chunk
        feature_loss = diff.square().sum() / float(loss_context["total_feature_loss_elems"])
        loss = loss + float(loss_context["feature_loss_weight"]) * feature_loss
        grad_target_grid = grad_target_grid + (
            2.0
            * float(loss_context["feature_loss_weight"])
            / float(loss_context["total_feature_loss_elems"])
        ) * diff
        _sync_device(case["device"])
        feature_t1 = time.perf_counter()
        timings["feature_loss_forward_ms"] = (feature_t1 - feature_t0) * 1000.0
    if rgb_probe is not None and loss_context["rgb_probe_loss_weight"] > 0.0:
        if rgb_probe_target is None:
            raise RuntimeError("RGB probe target is missing")
        probe_t0 = time.perf_counter()
        target_start, target_frames = _target_grid_slice_for_render_chunk(
            target_frames=int(rgb_probe_target.shape[0]),
            render_frames=int(feature_config.frames),
            frame_start=frame_start,
            chunk_frames=chunk_frames,
        )
        target_rgb_probe_chunk = rgb_probe_target[target_start : target_start + target_frames]
        probe_loss, probe_grad_grid = _manual_probe_loss_and_grid_grad(
            rgb_probe,
            rendered_target_grid,
            target_rgb_probe_chunk,
            total_rgb_probe_loss_elems=int(loss_context["total_rgb_probe_loss_elems"]),
            loss_weight=float(loss_context["rgb_probe_loss_weight"]),
        )
        loss = loss + float(loss_context["rgb_probe_loss_weight"]) * probe_loss
        grad_target_grid = grad_target_grid + probe_grad_grid
        _sync_device(case["device"])
        probe_t1 = time.perf_counter()
        timings["rgb_probe_loss_forward_ms"] = (probe_t1 - probe_t0) * 1000.0
    vjp_t0 = time.perf_counter()
    sparse_pack = None
    if sparse_grid:
        grad_feature_image = None
        sparse_pack = _pack_sparse_target_grid_vjp(
            grad_target_grid,
            input_shape=tuple(int(item) for item in feature_image.shape),
            mode=target_feature.grid_mode,
        )
    else:
        grad_feature_image = _render_grid_vjp_to_feature_image(
            grad_target_grid,
            input_shape=tuple(int(item) for item in feature_image.shape),
            mode=target_feature.grid_mode,
        )
    _sync_device(case["device"])
    vjp_t1 = time.perf_counter()
    timings["image_vjp_backward_ms"] = (vjp_t1 - vjp_t0) * 1000.0
    return loss, grad_feature_image, sparse_pack, timings


def _run_autograd(case: dict[str, Any], *, chunk_size: int, backward_mode: str, selected_global_step: int) -> dict[str, Any]:
    model = case["model"]
    feature_config = case["feature_config"]
    uvt_config = case["uvt_config"]
    device = case["device"]
    loss_context = _target_loss_context(case, selected_global_step)
    _zero_grads(model)
    phase = {
        "render_forward_ms": 0.0,
        "target_grid_prep_ms": 0.0,
        "feature_loss_forward_ms": 0.0,
        "rgb_probe_loss_forward_ms": 0.0,
        "backward_ms": 0.0,
    }
    loss_value = 0.0
    for frame_start in range(0, feature_config.frames, chunk_size):
        chunk_frames = min(chunk_size, feature_config.frames - frame_start)
        render_inputs, chunk_config = _chunk_render_inputs(model, uvt_config, frame_start, chunk_frames)
        _sync_device(device)
        t0 = time.perf_counter()
        render = render_uvt_feature_tubes_autograd(
            *render_inputs,
            chunk_config,
            backward_mode=backward_mode,
        )
        _sync_device(device)
        t1 = time.perf_counter()
        loss, timings = _loss_from_feature_image(
            feature_image=render.feature_image,
            case=case,
            loss_context=loss_context,
            frame_start=frame_start,
            chunk_frames=chunk_frames,
        )
        loss_value += float(loss.detach().cpu().item())
        _sync_device(device)
        t2 = time.perf_counter()
        loss.backward()
        _sync_device(device)
        t3 = time.perf_counter()
        phase["render_forward_ms"] += (t1 - t0) * 1000.0
        phase["target_grid_prep_ms"] += timings["target_grid_prep_ms"]
        phase["feature_loss_forward_ms"] += timings["feature_loss_forward_ms"]
        phase["rgb_probe_loss_forward_ms"] += timings["rgb_probe_loss_forward_ms"]
        phase["backward_ms"] += (t3 - t2) * 1000.0
    return {
        "loss": loss_value,
        "timing_ms": {**phase, "total_ms": sum(phase.values())},
        "grads": _collect_grads("model", model),
    }


def _run_bridge(
    case: dict[str, Any],
    *,
    chunk_size: int,
    backward_mode: str,
    selected_global_step: int,
    image_vjp_mode: str,
) -> dict[str, Any]:
    model = case["model"]
    feature_config = case["feature_config"]
    uvt_config = case["uvt_config"]
    device = case["device"]
    cfg = case["cfg"]
    loss_context = _target_loss_context(case, selected_global_step)
    _zero_grads(model)
    phase = {
        "render_forward_ms": 0.0,
        "target_grid_prep_ms": 0.0,
        "feature_loss_forward_ms": 0.0,
        "rgb_probe_loss_forward_ms": 0.0,
        "image_vjp_backward_ms": 0.0,
        "sparse_pack_ms": 0.0,
        "renderer_backward_ms": 0.0,
        "param_backward_ms": 0.0,
    }
    loss_value = 0.0
    finite = True
    alpha_grad_missing_count = 0
    sparse_pixel_count = 0
    sparse_total_pixels = 0
    tile_counts: list[torch.Tensor] = []
    tile_overflow: list[torch.Tensor] = []
    tile_unstable: list[torch.Tensor] = []
    for frame_start in range(0, feature_config.frames, chunk_size):
        chunk_frames = min(chunk_size, feature_config.frames - frame_start)
        render_inputs, chunk_config = _chunk_render_inputs(model, uvt_config, frame_start, chunk_frames)
        ma, q_uvt, _depth0, _depth_beta, opacity, feature = render_inputs
        _sync_device(device)
        t0 = time.perf_counter()
        sparse_mode = image_vjp_mode in {"analytic_sparse_pixels", "analytic_sparse_grid"}
        render = render_uvt_feature_tubes(*render_inputs, chunk_config, return_bins=sparse_mode)
        _sync_device(device)
        t1 = time.perf_counter()

        if image_vjp_mode == "autograd":
            feature_probe = render.feature_image.detach().requires_grad_(True)
            alpha_probe = render.alpha.detach().requires_grad_(True)
            loss, timings = _loss_from_feature_image(
                feature_image=feature_probe,
                case=case,
                loss_context=loss_context,
                frame_start=frame_start,
                chunk_frames=chunk_frames,
            )
            loss_value += float(loss.detach().cpu().item())
            _sync_device(device)
            t2 = time.perf_counter()
            loss.backward()
            _sync_device(device)
            t3 = time.perf_counter()
            grad_feature_image = feature_probe.grad
            if grad_feature_image is None:
                raise RuntimeError("target-grid/probe loss did not produce feature-image gradients")
            grad_alpha = alpha_probe.grad
            if grad_alpha is None:
                alpha_grad_missing_count += 1
                grad_alpha = torch.zeros_like(alpha_probe)
            image_vjp_ms = (t3 - t2) * 1000.0
        elif image_vjp_mode in {"analytic", "analytic_sparse_pixels", "analytic_sparse_grid"}:
            loss, grad_feature_image, sparse_pack, timings = _manual_loss_and_feature_vjp(
                feature_image=render.feature_image,
                case=case,
                loss_context=loss_context,
                frame_start=frame_start,
                chunk_frames=chunk_frames,
                sparse_grid=image_vjp_mode == "analytic_sparse_grid",
            )
            loss_value += float(loss.detach().cpu().item())
            grad_alpha = torch.zeros_like(render.alpha)
            alpha_grad_missing_count += 1
            _sync_device(device)
            t3 = time.perf_counter()
            image_vjp_ms = timings["image_vjp_backward_ms"]
        else:
            raise ValueError(f"image_vjp_mode must be one of {sorted(IMAGE_VJP_MODES)}, got {image_vjp_mode!r}")
        renderer_t0 = t3
        if sparse_mode:
            if render.tile_tube_ids is None or render.tile_depths is None:
                raise RuntimeError(f"{image_vjp_mode} requires render bins")
            if image_vjp_mode == "analytic_sparse_grid":
                if sparse_pack is None:
                    raise RuntimeError("analytic_sparse_grid did not produce sparse image gradients")
                pack_ms = 0.0
            else:
                if grad_feature_image is None:
                    raise RuntimeError("analytic_sparse_pixels did not produce feature-image gradients")
                _sync_device(device)
                pack_t0 = time.perf_counter()
                sparse_pack = _pack_sparse_image_vjp(grad_feature_image.contiguous(), grad_alpha.contiguous())
                _sync_device(device)
                pack_ms = (time.perf_counter() - pack_t0) * 1000.0
            sparse_pixel_count += sparse_pack.pixel_count
            sparse_total_pixels += sparse_pack.total_pixels
            renderer_t0 = time.perf_counter()
            grads = direct_atomic_feature_sparse_pixels_backward_cached_bins(
                *render_inputs,
                sparse_pack.pixel_ids,
                sparse_pack.grad_feature_values,
                sparse_pack.grad_alpha_values,
                render.tile_counts,
                render.tile_tube_ids,
                render.tile_depths,
                render.tile_unstable,
                chunk_config,
            )
        else:
            if grad_feature_image is None:
                raise RuntimeError(f"{image_vjp_mode} did not produce feature-image gradients")
            pack_ms = 0.0
            grads = direct_atomic_feature_backward(
                *render_inputs,
                grad_feature_image.contiguous(),
                grad_alpha.contiguous(),
                chunk_config,
                backward_mode=backward_mode,
            )
        _sync_device(device)
        t4 = time.perf_counter()
        torch.autograd.backward(
            (ma, q_uvt, opacity, feature),
            (grads[0], grads[1], grads[2], grads[3]),
        )
        _sync_device(device)
        t5 = time.perf_counter()

        phase["render_forward_ms"] += (t1 - t0) * 1000.0
        phase["target_grid_prep_ms"] += timings["target_grid_prep_ms"]
        phase["feature_loss_forward_ms"] += timings["feature_loss_forward_ms"]
        phase["rgb_probe_loss_forward_ms"] += timings["rgb_probe_loss_forward_ms"]
        phase["image_vjp_backward_ms"] += image_vjp_ms
        phase["sparse_pack_ms"] += pack_ms
        phase["renderer_backward_ms"] += (t4 - renderer_t0) * 1000.0
        phase["param_backward_ms"] += (t5 - t4) * 1000.0
        tile_counts.append(render.tile_counts)
        tile_overflow.append(render.tile_overflow)
        tile_unstable.append(grads[-1])
        grad_image_finite = (
            bool(torch.isfinite(sparse_pack.grad_feature_values).all().cpu())
            if sparse_mode
            else bool(torch.isfinite(grad_feature_image).all().cpu())
        )
        finite = (
            finite
            and bool(torch.isfinite(render.feature_image).all().cpu())
            and grad_image_finite
            and bool(torch.isfinite(grad_alpha).all().cpu())
            and all(bool(torch.isfinite(grad).all().cpu()) for grad in grads[:4])
        )
    return {
        "loss": loss_value,
        "timing_ms": {**phase, "total_ms": sum(phase.values())},
        "grads": _collect_grads("model", model),
        "tile_stats": _tile_load_stats(
            tile_counts=tile_counts,
            tile_overflow=tile_overflow,
            tile_unstable=tile_unstable,
            tile_capacity=int(cfg["feature_uvt"]["tile_capacity"]),
        ),
        "alpha_grad_missing_count": alpha_grad_missing_count,
        "sparse_pixel_count": sparse_pixel_count,
        "sparse_pixel_fraction": 0.0 if sparse_total_pixels <= 0 else sparse_pixel_count / float(sparse_total_pixels),
        "finite": finite,
    }


def profile(
    config_path: Path,
    *,
    warmup: int,
    repeat: int,
    global_step: int | None,
    image_vjp_mode: str,
) -> dict[str, Any]:
    if image_vjp_mode not in IMAGE_VJP_MODES:
        raise ValueError(f"image_vjp_mode must be one of {sorted(IMAGE_VJP_MODES)}, got {image_vjp_mode!r}")
    case = _load_case(config_path)
    cfg = case["cfg"]
    feature_config = case["feature_config"]
    render_mode = str(cfg["feature_uvt"]["render_mode"])
    backward_mode = backward_mode_for_feature_render_mode(render_mode, int(feature_config.feature_dim))
    chunk_size_cfg = cfg["train"]["frame_chunk_size"]
    chunk_size = int(feature_config.frames) if chunk_size_cfg is None else min(int(chunk_size_cfg), int(feature_config.frames))
    if chunk_size <= 0:
        raise ValueError("train.frame_chunk_size must be null or positive")
    selected_global_step = int(cfg["train"]["global_step_offset"] if global_step is None else global_step)
    loss_context = _target_loss_context(case, selected_global_step)

    autograd_samples = {
        "render_forward_ms": [],
        "target_grid_prep_ms": [],
        "feature_loss_forward_ms": [],
        "rgb_probe_loss_forward_ms": [],
        "backward_ms": [],
        "total_ms": [],
    }
    bridge_samples = {
        "render_forward_ms": [],
        "target_grid_prep_ms": [],
        "feature_loss_forward_ms": [],
        "rgb_probe_loss_forward_ms": [],
        "image_vjp_backward_ms": [],
        "sparse_pack_ms": [],
        "renderer_backward_ms": [],
        "param_backward_ms": [],
        "total_ms": [],
    }
    comparison: dict[str, Any] | None = None
    last_bridge: dict[str, Any] | None = None
    loss_abs_error = 0.0
    for index in range(warmup + repeat):
        autograd = _run_autograd(
            case,
            chunk_size=chunk_size,
            backward_mode=backward_mode,
            selected_global_step=selected_global_step,
        )
        bridge = _run_bridge(
            case,
            chunk_size=chunk_size,
            backward_mode=backward_mode,
            selected_global_step=selected_global_step,
            image_vjp_mode=image_vjp_mode,
        )
        if index >= warmup:
            for key, value in autograd["timing_ms"].items():
                autograd_samples[key].append(float(value))
            for key, value in bridge["timing_ms"].items():
                bridge_samples[key].append(float(value))
            loss_abs_error = max(loss_abs_error, abs(float(autograd["loss"]) - float(bridge["loss"])))
            if comparison is None:
                comparison = _grad_comparison(autograd["grads"], bridge["grads"])
            last_bridge = bridge

    if comparison is None or last_bridge is None:
        raise RuntimeError("repeat must be positive")
    autograd_stats = {key: summary_stats(values) for key, values in autograd_samples.items()}
    bridge_stats = {key: summary_stats(values) for key, values in bridge_samples.items()}
    autograd_total = autograd_stats["total_ms"]["mean"]
    bridge_total = bridge_stats["total_ms"]["mean"]
    tile = last_bridge["tile_stats"]
    pass_flag = (
        bool(last_bridge["finite"])
        and int(tile["overflow_tile_count"]) == 0
        and int(tile["unstable_tile_count"]) == 0
        and comparison["max_abs_error"] <= 2.0e-4
        and loss_abs_error <= 1.0e-5
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "gate": "star_uvt_targetgrid_vjp_bridge_profile",
        "config": str(config_path),
        "global_step": selected_global_step,
        "frames": int(feature_config.frames),
        "size": int(feature_config.height),
        "tubes": int(cfg["feature_uvt"]["tube_count"]),
        "feature_dim": int(feature_config.feature_dim),
        "frame_chunk_size": chunk_size,
        "tile_t": int(cfg["feature_uvt"]["tile_t"]),
        "tile_capacity": int(cfg["feature_uvt"]["tile_capacity"]),
        "render_mode": render_mode,
        "backward_mode": backward_mode,
        "image_vjp_mode": image_vjp_mode,
        "feature_loss_weight": float(loss_context["feature_loss_weight"]),
        "rgb_probe_loss_weight": float(loss_context["rgb_probe_loss_weight"]),
        "resume_checkpoint": case["resume_state"]["path"],
        "resume_loaded": bool(case["resume_state"]["loaded"]),
        "resume_checkpoint_steps": case["resume_state"]["steps"],
        "scope": (
            "Current target-grid V-JEPA feature loss plus frozen RGB-probe VJP. "
            "The bridge path computes image-space gradients explicitly and calls the STAR UVT Metal feature backward manually. "
            f"image_vjp_mode={image_vjp_mode}."
        ),
        "warmup": warmup,
        "repeat": repeat,
        "autograd_timing_ms": autograd_stats,
        "bridge_timing_ms": bridge_stats,
        "speedup_vs_autograd_total": 0.0 if bridge_total <= 0.0 else autograd_total / bridge_total,
        "loss_max_abs_error": loss_abs_error,
        "grad_comparison": comparison,
        "tile_stats": tile,
        "alpha_grad_missing_count": int(last_bridge["alpha_grad_missing_count"]),
        "sparse_pixel_count": int(last_bridge["sparse_pixel_count"]),
        "sparse_pixel_fraction": float(last_bridge["sparse_pixel_fraction"]),
        "finite": bool(last_bridge["finite"]),
        "pass": pass_flag,
    }


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    auto = result["autograd_timing_ms"]
    bridge = result["bridge_timing_ms"]
    comp = result["grad_comparison"]
    tile = result["tile_stats"]
    auto_loss_forward = (
        auto["target_grid_prep_ms"]["mean"]
        + auto["feature_loss_forward_ms"]["mean"]
        + auto["rgb_probe_loss_forward_ms"]["mean"]
    )
    bridge_loss_forward = (
        bridge["target_grid_prep_ms"]["mean"]
        + bridge["feature_loss_forward_ms"]["mean"]
        + bridge["rgb_probe_loss_forward_ms"]["mean"]
    )
    bridge_backward = (
        bridge["image_vjp_backward_ms"]["mean"]
        + bridge["sparse_pack_ms"]["mean"]
        + bridge["renderer_backward_ms"]["mean"]
        + bridge["param_backward_ms"]["mean"]
    )
    if (
        result["image_vjp_mode"] in {"analytic", "analytic_sparse_pixels", "analytic_sparse_grid"}
        and result["speedup_vs_autograd_total"] > 1.02
    ):
        decision_lines = [
            "- Passing parity proves the analytic target-grid/probe image VJP matches normal autograd on the current objective.",
            "- The repeat timing is a real boundary win, so this is now evidence for a native/fused target-grid/probe VJP path rather than another renderer-only change.",
            "- This profile is only the bridge benchmark; trainer acceptance is tracked by the matching overfit JSON and progress note.",
        ]
    else:
        decision_lines = [
            "- Passing parity proves the manual image-space VJP bridge matches normal autograd on the current target-grid/frozen-probe objective.",
            "- This is a correctness bridge and timing breakdown, not yet a faster native loss: the bridge still computes the image-space VJP in Torch and still calls the same Metal feature backward.",
            "- If this row is near parity but not much faster, the next speed implementation should fuse or simplify the target-grid/probe VJP or improve scalar fixedbin/tile-slot renderer backward, instead of only rewiring Python.",
        ]
    lines = [
        "# STAR UVT Target-Grid VJP Bridge Profile",
        "",
        f"Generated: {result['generated_at']}",
        "",
        "## Scope",
        "",
        result["scope"],
        "This is the real target-grid/frozen-probe counterpart to the narrower linear RGB logit-handoff profile.",
        "",
        "## Timing",
        "",
        "| path | total | render fwd | loss fwd | image VJP bwd | sparse pack | renderer bwd | param bwd | total bwd |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| autograd | "
        + " | ".join(
            [
                _fmt(auto["total_ms"]["mean"]),
                _fmt(auto["render_forward_ms"]["mean"]),
                _fmt(auto_loss_forward),
                "",
                "",
                "",
                "",
                _fmt(auto["backward_ms"]["mean"]),
            ]
        )
        + " |",
        f"| image_vjp_bridge/{result['image_vjp_mode']} | "
        + " | ".join(
            [
                _fmt(bridge["total_ms"]["mean"]),
                _fmt(bridge["render_forward_ms"]["mean"]),
                _fmt(bridge_loss_forward),
                _fmt(bridge["image_vjp_backward_ms"]["mean"]),
                _fmt(bridge["sparse_pack_ms"]["mean"]),
                _fmt(bridge["renderer_backward_ms"]["mean"]),
                _fmt(bridge["param_backward_ms"]["mean"]),
                _fmt(bridge_backward),
            ]
        )
        + " |",
        "",
        f"Total speedup versus autograd: `{result['speedup_vs_autograd_total']:.3f}x`.",
        "",
        "## Gradient Parity",
        "",
        f"- loss max abs error: `{result['loss_max_abs_error']:.3e}`",
        f"- grad max abs error: `{comp['max_abs_error']:.3e}`",
        f"- grad max rel error: `{comp['max_rel_error']:.3e}`",
        f"- missing grad names: `{comp['missing']}`",
        "",
        "## Tile State",
        "",
        f"- overflow tiles: `{tile['overflow_tile_count']}`",
        f"- unstable tiles: `{tile['unstable_tile_count']}`",
        f"- max/p95/cap: `{tile['max_tile_count']}/{tile['p95_tile_count']}/{tile['tile_capacity']}`",
        f"- alpha grad missing chunks: `{result['alpha_grad_missing_count']}`",
        f"- sparse pixels: `{result['sparse_pixel_count']}` (`{result['sparse_pixel_fraction']:.6f}` of dense pixels)",
        "",
        "## Decision",
        "",
        *decision_lines,
        "",
        f"Pass: `{result['pass']}`",
        "",
    ]
    write_report_text(path, "\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--global-step", type=int, default=None)
    parser.add_argument("--image-vjp-mode", choices=sorted(IMAGE_VJP_MODES), default="autograd")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    out_base = args.out_base if args.out_base.is_absolute() else ROOT / args.out_base
    result = profile(
        config,
        warmup=int(args.warmup),
        repeat=int(args.repeat),
        global_step=args.global_step,
        image_vjp_mode=str(args.image_vjp_mode),
    )
    write_report_json(out_base.with_suffix(".json"), result)
    _write_markdown(out_base.with_suffix(".md"), result)
    print(json.dumps({"out_base": str(out_base), "pass": result["pass"]}, sort_keys=True))


if __name__ == "__main__":
    main()
