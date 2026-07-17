from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

try:
    from .report_artifacts import split_csv_ints, write_report_json
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import split_csv_ints, write_report_json
from train_devices import sync_torch_device
from torch_gsplat_bridge_star_uvt import UVTRenderConfig
from torch_gsplat_bridge_star_uvt.feature_rasterize import (
    bin_uvt_feature_tubes,
    brute_force_render_uvt_feature_tubes,
    direct_atomic_feature_sparse_pixels_backward_cached_bins,
    direct_hidden_sigmoid_target_area_backward_cached_bins,
    render_uvt_feature_sparse_pixels_with_bins,
    sparse_hidden_sigmoid_target_area_forward_sums_cached_bins,
)

TARGET_AREA_BACKWARD_MODES = (
    "target_area_star_only",
    "target_area_skip_feature_grad",
    "target_area_feature_grad_only",
    "target_area_recompute_only",
    "target_area_traversal_only",
    "target_area_hidden_forward_only",
    "target_area_hidden_preact_only",
    "target_area_star_only_rowmajor_wt",
    "target_area_recompute_only_rowmajor_wt",
    "target_area_star_only_vec4_wt",
    "target_area_recompute_only_vec4_wt",
    "target_area_colorizer_grad_only",
    "target_area_colorizer_vec4_wt",
    "target_area_colorizer_simdreduce_grad_only",
    "target_area_colorizer_simdreduce_vec4_wt",
)


def _sync() -> None:
    sync_torch_device(torch.device("mps"))


def _max_err(got: Tensor, expected: Tensor) -> float:
    return float((got.detach().cpu() - expected.detach().cpu()).abs().max().item())


def _tiny_scene(feature_dim: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, UVTRenderConfig]:
    tile_capacity = int(os.environ.get("STAR_UVT_TILE_CAPACITY", "128"))
    config = UVTRenderConfig(
        height=8,
        width=8,
        frames=2,
        tile_t=2,
        tile_capacity=tile_capacity,
        alpha_threshold=1.0 / 255.0,
    )
    ma = torch.tensor([[3.5, 3.5, -0.2], [4.5, 3.8, 0.1]], dtype=torch.float32)
    q_uvt = torch.tensor(
        [[0.30, 0.0, 0.0, 0.30, 0.0, 0.40], [0.24, 0.0, 0.03, 0.28, -0.02, 0.35]],
        dtype=torch.float32,
    )
    depth0 = torch.tensor([0.8, 1.2], dtype=torch.float32)
    depth_beta = torch.zeros((2, 3), dtype=torch.float32)
    opacity = torch.tensor([0.55, 0.45], dtype=torch.float32)
    feature = torch.linspace(-0.4, 0.5, 2 * feature_dim, dtype=torch.float32).view(2, feature_dim)
    return ma, q_uvt, depth0, depth_beta, opacity, feature, config


def _random_timing_scene(
    *,
    frames: int,
    size: int,
    tubes: int,
    feature_dim: int,
    tile_capacity: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, UVTRenderConfig]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    center_u = torch.rand((tubes,), generator=generator) * float(size)
    center_v = torch.rand((tubes,), generator=generator) * float(size)
    center_t = (torch.rand((tubes,), generator=generator) - 0.5) * float(frames)
    ma = torch.stack((center_u, center_v, center_t), dim=-1).to(dtype=torch.float32)
    precision_uv = torch.full((tubes,), 0.18, dtype=torch.float32)
    precision_t = torch.full((tubes,), 0.12, dtype=torch.float32)
    q_uvt = torch.stack(
        (
            precision_uv,
            torch.zeros_like(precision_uv),
            torch.zeros_like(precision_uv),
            precision_uv,
            torch.zeros_like(precision_uv),
            precision_t,
        ),
        dim=-1,
    )
    depth0 = torch.linspace(0.5, 1.5, tubes, dtype=torch.float32)
    depth_beta = torch.zeros((tubes, 3), dtype=torch.float32)
    opacity = torch.full((tubes,), 0.35, dtype=torch.float32)
    feature = torch.randn((tubes, feature_dim), generator=generator, dtype=torch.float32) * 0.1
    config = UVTRenderConfig(height=size, width=size, frames=frames, tile_t=2, tile_capacity=tile_capacity)
    return ma, q_uvt, depth0, depth_beta, opacity, feature, config


def _hidden_params(feature_dim: int, hidden_dim: int, *, seed: int = 7001) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden_weight = torch.randn((hidden_dim, feature_dim), generator=generator, dtype=torch.float32) * 0.1
    hidden_bias = torch.randn((hidden_dim,), generator=generator, dtype=torch.float32) * 0.05
    output_weight = torch.randn((3, hidden_dim), generator=generator, dtype=torch.float32) * 0.1
    output_bias = torch.randn((3,), generator=generator, dtype=torch.float32) * 0.05
    return hidden_weight, hidden_bias, output_weight, output_bias


def _cell_grid_pixel_ids(
    *,
    frames: int,
    size: int,
    grid_side: int,
    patch_size: int,
) -> tuple[Tensor, Tensor, int, int]:
    if grid_side <= 0 or patch_size <= 0:
        raise ValueError("grid_side and patch_size must be positive")
    if grid_side * patch_size > size:
        raise ValueError("grid_side * patch_size must fit inside size")
    gy = torch.arange(grid_side, dtype=torch.long)[:, None, None, None]
    gx = torch.arange(grid_side, dtype=torch.long)[None, :, None, None]
    py = torch.arange(patch_size, dtype=torch.long)[None, None, :, None]
    px = torch.arange(patch_size, dtype=torch.long)[None, None, None, :]
    local_pixels = ((gy * patch_size + py) * size + (gx * patch_size + px)).reshape(-1)
    local_cells = (gy * grid_side + gx).expand(grid_side, grid_side, patch_size, patch_size).reshape(-1)
    frame_offsets = torch.arange(frames, dtype=torch.long)[:, None] * (size * size)
    cell_offsets = torch.arange(frames, dtype=torch.long)[:, None] * (grid_side * grid_side)
    pixel_ids = (frame_offsets + local_pixels[None, :]).reshape(-1)
    cell_ids = (cell_offsets + local_cells[None, :]).reshape(-1)
    return (
        pixel_ids.to(dtype=torch.int32),
        cell_ids.to(dtype=torch.int32),
        frames * grid_side * grid_side,
        patch_size * patch_size,
    )


def _gather_sparse(feature_tfhw: Tensor, alpha_thw: Tensor, pixel_ids: Tensor) -> tuple[Tensor, Tensor]:
    feature_flat = feature_tfhw.permute(0, 2, 3, 1).contiguous().view(-1, feature_tfhw.shape[1])
    alpha_flat = alpha_thw.contiguous().view(-1)
    return feature_flat.index_select(0, pixel_ids.to(torch.long)), alpha_flat.index_select(0, pixel_ids.to(torch.long))


def _exact_gelu_grad(x: Tensor) -> Tensor:
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    inv_sqrt2pi = 1.0 / math.sqrt(2.0 * math.pi)
    return 0.5 * (1.0 + torch.erf(x * inv_sqrt2)) + x * inv_sqrt2pi * torch.exp(-0.5 * x * x)


def _hidden_target_area_loss_and_grads(
    feature_values: Tensor,
    alpha_values: Tensor,
    cell_ids: Tensor,
    target_cells: Tensor,
    patch_area: int,
    hidden_weight: Tensor,
    hidden_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    hidden_pre = feature_values @ hidden_weight.T + hidden_bias
    hidden = torch.nn.functional.gelu(hidden_pre)
    logits = hidden @ output_weight.T + output_bias
    splat_rgb = torch.sigmoid(logits)
    rgb = alpha_values[:, None] * splat_rgb
    pred_sums = torch.zeros_like(target_cells)
    pred_sums.index_add_(0, cell_ids.to(torch.long), rgb)
    diff = pred_sums / float(patch_area) - target_cells
    loss = diff.square().sum() / float(target_cells.numel())
    cell_grad_rgb = (2.0 / float(target_cells.numel())) * diff / float(patch_area)
    grad_rgb = cell_grad_rgb.index_select(0, cell_ids.to(torch.long))
    grad_logits = grad_rgb * alpha_values[:, None] * splat_rgb * (1.0 - splat_rgb)
    grad_alpha_values = (grad_rgb * splat_rgb).sum(dim=1).contiguous()
    grad_hidden = grad_logits @ output_weight
    grad_hidden_pre = grad_hidden * _exact_gelu_grad(hidden_pre)
    grad_feature_values = (grad_hidden_pre @ hidden_weight).contiguous()
    grad_hidden_weight = grad_hidden_pre.T @ feature_values
    grad_hidden_bias = grad_hidden_pre.sum(dim=0)
    grad_output_weight = grad_logits.T @ hidden
    grad_output_bias = grad_logits.sum(dim=0)
    return (
        loss,
        grad_feature_values,
        grad_alpha_values,
        cell_grad_rgb.contiguous(),
        grad_hidden_weight.contiguous(),
        grad_hidden_bias.contiguous(),
        grad_output_weight.contiguous(),
        grad_output_bias.contiguous(),
    )


def _hidden_target_area_loss_and_colorizer_grads(
    feature_values: Tensor,
    alpha_values: Tensor,
    cell_ids: Tensor,
    target_cells: Tensor,
    patch_area: int,
    hidden_weight: Tensor,
    hidden_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    hidden_pre = feature_values @ hidden_weight.T + hidden_bias
    hidden = torch.nn.functional.gelu(hidden_pre)
    logits = hidden @ output_weight.T + output_bias
    splat_rgb = torch.sigmoid(logits)
    rgb = alpha_values[:, None] * splat_rgb
    pred_sums = torch.zeros_like(target_cells)
    pred_sums.index_add_(0, cell_ids.to(torch.long), rgb)
    diff = pred_sums / float(patch_area) - target_cells
    loss = diff.square().sum() / float(target_cells.numel())
    cell_grad_rgb = (2.0 / float(target_cells.numel())) * diff / float(patch_area)
    grad_rgb = cell_grad_rgb.index_select(0, cell_ids.to(torch.long))
    grad_logits = grad_rgb * alpha_values[:, None] * splat_rgb * (1.0 - splat_rgb)
    grad_hidden = grad_logits @ output_weight
    grad_hidden_pre = grad_hidden * _exact_gelu_grad(hidden_pre)
    grad_hidden_weight = grad_hidden_pre.T @ feature_values
    grad_hidden_bias = grad_hidden_pre.sum(dim=0)
    grad_output_weight = grad_logits.T @ hidden
    grad_output_bias = grad_logits.sum(dim=0)
    return (
        loss,
        cell_grad_rgb.contiguous(),
        grad_hidden_weight.contiguous(),
        grad_hidden_bias.contiguous(),
        grad_output_weight.contiguous(),
        grad_output_bias.contiguous(),
    )


def _checked_tiny_error_names(backward_mode: str) -> tuple[str, ...]:
    if backward_mode == "target_area_skip_feature_grad":
        return ("loss", "ma", "q", "opacity")
    if backward_mode == "target_area_feature_grad_only":
        return ("loss", "feature")
    if backward_mode in {
        "target_area_recompute_only",
        "target_area_traversal_only",
        "target_area_hidden_forward_only",
        "target_area_hidden_preact_only",
        "target_area_recompute_only_rowmajor_wt",
        "target_area_recompute_only_vec4_wt",
    }:
        return ("loss",)
    if backward_mode in {"target_area_colorizer_vec4_wt", "target_area_colorizer_simdreduce_vec4_wt"}:
        return ("loss", "ma", "q", "opacity", "feature", "hidden_weight", "hidden_bias", "output_weight", "output_bias")
    if backward_mode in {"target_area_colorizer_grad_only", "target_area_colorizer_simdreduce_grad_only"}:
        return ("loss", "hidden_weight", "hidden_bias", "output_weight", "output_bias")
    return ("loss", "ma", "q", "opacity", "feature")


def run_tiny_parity(feature_dim: int, hidden_dim: int, backward_mode: str) -> dict[str, Any]:
    ma, q_uvt, depth0, depth_beta, opacity, feature, config = _tiny_scene(feature_dim)
    pixel_ids, cell_ids, cell_count, patch_area = _cell_grid_pixel_ids(frames=2, size=8, grid_side=2, patch_size=4)
    hidden_weight, hidden_bias, output_weight, output_bias = _hidden_params(feature_dim, hidden_dim)
    generator = torch.Generator(device="cpu").manual_seed(9001 + feature_dim + hidden_dim)
    target_cells = torch.rand((cell_count, 3), generator=generator, dtype=torch.float32)

    ma_ref = ma.clone().requires_grad_(True)
    q_ref = q_uvt.clone().requires_grad_(True)
    opacity_ref = opacity.clone().requires_grad_(True)
    feature_ref = feature.clone().requires_grad_(True)
    ref_feature, ref_alpha = brute_force_render_uvt_feature_tubes(
        ma_ref, q_ref, depth0, depth_beta, opacity_ref, feature_ref, config
    )
    sparse_feature_ref, sparse_alpha_ref = _gather_sparse(ref_feature, ref_alpha, pixel_ids)
    (
        loss_ref,
        _,
        _,
        _,
        ref_grad_hidden_weight,
        ref_grad_hidden_bias,
        ref_grad_output_weight,
        ref_grad_output_bias,
    ) = _hidden_target_area_loss_and_grads(
        sparse_feature_ref,
        sparse_alpha_ref,
        cell_ids,
        target_cells,
        patch_area,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
    )
    loss_ref.backward()

    mps_inputs = [tensor.to("mps").contiguous() for tensor in (ma, q_uvt, depth0, depth_beta, opacity, feature)]
    pixel_ids_mps = pixel_ids.to("mps").contiguous()
    cell_ids_mps = cell_ids.to("mps").contiguous()
    target_cells_mps = target_cells.to("mps").contiguous()
    hidden_weight_mps, hidden_bias_mps, output_weight_mps, output_bias_mps = [
        tensor.to("mps").contiguous() for tensor in (hidden_weight, hidden_bias, output_weight, output_bias)
    ]
    bins = bin_uvt_feature_tubes(*mps_inputs[:5], config, feature_dim=feature_dim)
    forward = sparse_hidden_sigmoid_target_area_forward_sums_cached_bins(
        *mps_inputs,
        pixel_ids_mps,
        cell_ids_mps,
        hidden_weight_mps,
        hidden_bias_mps,
        output_weight_mps,
        output_bias_mps,
        bins.tile_counts,
        bins.tile_tube_ids,
        bins.tile_depths,
        bins.tile_unstable,
        config,
        cell_count=cell_count,
    )
    diff = forward.pred_sums / float(patch_area) - target_cells_mps
    loss_native = diff.square().sum() / float(target_cells.numel())
    cell_grad_rgb = ((2.0 / float(target_cells.numel())) * diff / float(patch_area)).contiguous()
    backward = direct_hidden_sigmoid_target_area_backward_cached_bins(
        *mps_inputs,
        pixel_ids_mps,
        cell_ids_mps,
        cell_grad_rgb,
        hidden_weight_mps,
        hidden_bias_mps,
        output_weight_mps,
        output_bias_mps,
        bins.tile_counts,
        bins.tile_tube_ids,
        bins.tile_depths,
        forward.tile_unstable,
        config,
        backward_mode=backward_mode,
    )
    errors = {
        "loss": _max_err(loss_native.reshape(()), loss_ref.detach()),
        "ma": _max_err(backward.grad_ma, ma_ref.grad),
        "q": _max_err(backward.grad_q_uvt, q_ref.grad),
        "opacity": _max_err(backward.grad_opacity, opacity_ref.grad),
        "hidden_weight": _max_err(backward.grad_hidden_weight, ref_grad_hidden_weight)
        if backward.grad_hidden_weight is not None
        else None,
        "hidden_bias": _max_err(backward.grad_hidden_bias, ref_grad_hidden_bias)
        if backward.grad_hidden_bias is not None
        else None,
        "output_weight": _max_err(backward.grad_output_weight, ref_grad_output_weight)
        if backward.grad_output_weight is not None
        else None,
        "output_bias": _max_err(backward.grad_output_bias, ref_grad_output_bias)
        if backward.grad_output_bias is not None
        else None,
    }
    feature_error = _max_err(backward.grad_feature, feature_ref.grad)
    if backward_mode in {
        "target_area_skip_feature_grad",
        "target_area_recompute_only",
        "target_area_traversal_only",
        "target_area_hidden_forward_only",
        "target_area_hidden_preact_only",
        "target_area_recompute_only_rowmajor_wt",
        "target_area_recompute_only_vec4_wt",
    }:
        errors["feature"] = None
    else:
        errors["feature"] = feature_error
    if backward_mode in {
        "target_area_feature_grad_only",
        "target_area_recompute_only",
        "target_area_traversal_only",
        "target_area_hidden_forward_only",
        "target_area_hidden_preact_only",
        "target_area_recompute_only_rowmajor_wt",
        "target_area_recompute_only_vec4_wt",
    }:
        errors["ma"] = None
        errors["q"] = None
        errors["opacity"] = None
    else:
        errors.setdefault("feature", feature_error)
    checked_names = _checked_tiny_error_names(backward_mode)
    checked_errors = [errors[name] for name in checked_names if errors[name] is not None]
    ignored_errors = {name: value for name, value in errors.items() if name not in checked_names}
    return {
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "backward_mode": backward_mode,
        "cell_count": cell_count,
        "patch_area": patch_area,
        "sparse_count": int(pixel_ids.numel()),
        "loss_ref": float(loss_ref.detach().item()),
        "loss_native": float(loss_native.detach().cpu().item()),
        "max_abs_errors": errors,
        "checked_error_names": list(checked_names),
        "ignored_max_abs_errors": ignored_errors,
        "feature_grad_error_ignored": None
        if backward_mode
        not in {
            "target_area_skip_feature_grad",
            "target_area_recompute_only",
            "target_area_traversal_only",
            "target_area_hidden_forward_only",
            "target_area_hidden_preact_only",
            "target_area_recompute_only_rowmajor_wt",
            "target_area_recompute_only_vec4_wt",
        }
        else feature_error,
        "ma_grad_error_ignored": None
        if backward_mode
        not in {
            "target_area_feature_grad_only",
            "target_area_recompute_only",
            "target_area_traversal_only",
            "target_area_hidden_forward_only",
            "target_area_hidden_preact_only",
            "target_area_recompute_only_rowmajor_wt",
            "target_area_recompute_only_vec4_wt",
        }
        else _max_err(backward.grad_ma, ma_ref.grad),
        "q_grad_error_ignored": None
        if backward_mode
        not in {
            "target_area_feature_grad_only",
            "target_area_recompute_only",
            "target_area_traversal_only",
            "target_area_hidden_forward_only",
            "target_area_hidden_preact_only",
            "target_area_recompute_only_rowmajor_wt",
            "target_area_recompute_only_vec4_wt",
        }
        else _max_err(backward.grad_q_uvt, q_ref.grad),
        "opacity_grad_error_ignored": None
        if backward_mode
        not in {
            "target_area_feature_grad_only",
            "target_area_recompute_only",
            "target_area_traversal_only",
            "target_area_hidden_forward_only",
            "target_area_hidden_preact_only",
            "target_area_recompute_only_rowmajor_wt",
            "target_area_recompute_only_vec4_wt",
        }
        else _max_err(backward.grad_opacity, opacity_ref.grad),
        "ma_grad_norm": float(backward.grad_ma.detach().norm().cpu().item()),
        "q_grad_norm": float(backward.grad_q_uvt.detach().norm().cpu().item()),
        "opacity_grad_norm": float(backward.grad_opacity.detach().norm().cpu().item()),
        "feature_grad_norm": float(backward.grad_feature.detach().norm().cpu().item()),
        "feature_ref_grad_norm": float(feature_ref.grad.detach().norm().cpu().item()),
        "tile_overflow_sum": int(bins.tile_overflow.sum().cpu().item()),
        "tile_unstable_sum": int(backward.tile_unstable.sum().cpu().item()),
        "pass": max(checked_errors) <= 5.0e-4 and int(bins.tile_overflow.sum().cpu().item()) == 0,
    }


def _mean(samples: list[float]) -> float:
    return sum(samples) / float(len(samples))


def _finite_tensors(tensors: tuple[Tensor, ...]) -> bool:
    return all(bool(torch.isfinite(tensor).all().cpu()) for tensor in tensors)


def run_timing_case(
    *,
    frames: int,
    size: int,
    tubes: int,
    feature_dim: int,
    hidden_dim: int,
    grid_side: int,
    patch_size: int,
    tile_capacity: int,
    seed: int,
    warmup: int,
    repeat: int,
    include_baseline: bool,
    backward_mode: str,
    include_torch_reducer_prototype: bool,
) -> dict[str, Any]:
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    ma, q_uvt, depth0, depth_beta, opacity, feature, config = _random_timing_scene(
        frames=frames,
        size=size,
        tubes=tubes,
        feature_dim=feature_dim,
        tile_capacity=tile_capacity,
        seed=seed,
    )
    pixel_ids, cell_ids, cell_count, patch_area = _cell_grid_pixel_ids(
        frames=frames,
        size=size,
        grid_side=grid_side,
        patch_size=patch_size,
    )
    mps_inputs = [tensor.to("mps").contiguous() for tensor in (ma, q_uvt, depth0, depth_beta, opacity, feature)]
    pixel_ids_mps = pixel_ids.to("mps").contiguous()
    cell_ids_mps = cell_ids.to("mps").contiguous()
    hidden_weight_mps, hidden_bias_mps, output_weight_mps, output_bias_mps = [
        tensor.to("mps").contiguous() for tensor in _hidden_params(feature_dim, hidden_dim, seed=seed + 2003)
    ]
    target_generator = torch.Generator(device="cpu").manual_seed(seed + 3001)
    target_cells = torch.rand((cell_count, 3), generator=target_generator, dtype=torch.float32).to("mps").contiguous()

    samples: dict[str, list[float]] = {
        "native_bin_ms": [],
        "native_forward_loss_ms": [],
        "native_backward_ms": [],
        "baseline_render_ms": [],
        "baseline_loss_ms": [],
        "baseline_backward_ms": [],
        "loss_error": [],
        "prototype_sparse_forward_ms": [],
        "prototype_colorizer_reduce_ms": [],
        "prototype_native_star_backward_ms": [],
        "prototype_colorizer_max_error": [],
        "prototype_star_max_error": [],
    }
    tile_overflow_sum = 0
    tile_unstable_sum = 0
    finite = True

    for iteration in range(warmup + repeat):
        baseline_loss: Tensor | None = None
        if include_baseline:
            _sync()
            t0 = time.perf_counter()
            sparse_render = render_uvt_feature_sparse_pixels_with_bins(*mps_inputs, pixel_ids_mps, config)
            _sync()
            baseline_render_ms = (time.perf_counter() - t0) * 1000.0

            _sync()
            t1 = time.perf_counter()
            baseline_loss, grad_feature_values, grad_alpha_values, _, *_ = _hidden_target_area_loss_and_grads(
                sparse_render.feature_values,
                sparse_render.alpha_values,
                cell_ids_mps,
                target_cells,
                patch_area,
                hidden_weight_mps,
                hidden_bias_mps,
                output_weight_mps,
                output_bias_mps,
            )
            _sync()
            baseline_loss_ms = (time.perf_counter() - t1) * 1000.0

            _sync()
            t2 = time.perf_counter()
            baseline_grads = direct_atomic_feature_sparse_pixels_backward_cached_bins(
                *mps_inputs,
                pixel_ids_mps,
                grad_feature_values,
                grad_alpha_values,
                sparse_render.tile_counts,
                sparse_render.tile_tube_ids,
                sparse_render.tile_depths,
                sparse_render.tile_unstable,
                config,
            )
            _sync()
            baseline_backward_ms = (time.perf_counter() - t2) * 1000.0
            finite = finite and _finite_tensors((*baseline_grads[:4], baseline_loss))
        else:
            baseline_render_ms = 0.0
            baseline_loss_ms = 0.0
            baseline_backward_ms = 0.0

        _sync()
        t3 = time.perf_counter()
        bins = bin_uvt_feature_tubes(*mps_inputs[:5], config, feature_dim=feature_dim)
        _sync()
        native_bin_ms = (time.perf_counter() - t3) * 1000.0

        _sync()
        t4 = time.perf_counter()
        forward = sparse_hidden_sigmoid_target_area_forward_sums_cached_bins(
            *mps_inputs,
            pixel_ids_mps,
            cell_ids_mps,
            hidden_weight_mps,
            hidden_bias_mps,
            output_weight_mps,
            output_bias_mps,
            bins.tile_counts,
            bins.tile_tube_ids,
            bins.tile_depths,
            bins.tile_unstable,
            config,
            cell_count=cell_count,
        )
        diff = forward.pred_sums / float(patch_area) - target_cells
        native_loss = diff.square().sum() / float(target_cells.numel())
        cell_grad_rgb = ((2.0 / float(target_cells.numel())) * diff / float(patch_area)).contiguous()
        _sync()
        native_forward_loss_ms = (time.perf_counter() - t4) * 1000.0

        _sync()
        t5 = time.perf_counter()
        native_grads = direct_hidden_sigmoid_target_area_backward_cached_bins(
            *mps_inputs,
            pixel_ids_mps,
            cell_ids_mps,
            cell_grad_rgb,
            hidden_weight_mps,
            hidden_bias_mps,
            output_weight_mps,
            output_bias_mps,
            bins.tile_counts,
            bins.tile_tube_ids,
            bins.tile_depths,
            forward.tile_unstable,
            config,
            backward_mode=backward_mode,
        )
        _sync()
        native_backward_ms = (time.perf_counter() - t5) * 1000.0
        if include_torch_reducer_prototype:
            _sync()
            t6 = time.perf_counter()
            sparse_render = render_uvt_feature_sparse_pixels_with_bins(*mps_inputs, pixel_ids_mps, config)
            _sync()
            prototype_sparse_forward_ms = (time.perf_counter() - t6) * 1000.0

            _sync()
            t7 = time.perf_counter()
            (
                prototype_loss,
                prototype_cell_grad_rgb,
                prototype_grad_hidden_weight,
                prototype_grad_hidden_bias,
                prototype_grad_output_weight,
                prototype_grad_output_bias,
            ) = _hidden_target_area_loss_and_colorizer_grads(
                sparse_render.feature_values,
                sparse_render.alpha_values,
                cell_ids_mps,
                target_cells,
                patch_area,
                hidden_weight_mps,
                hidden_bias_mps,
                output_weight_mps,
                output_bias_mps,
            )
            _sync()
            prototype_colorizer_reduce_ms = (time.perf_counter() - t7) * 1000.0

            _sync()
            t8 = time.perf_counter()
            prototype_star_grads = direct_hidden_sigmoid_target_area_backward_cached_bins(
                *mps_inputs,
                pixel_ids_mps,
                cell_ids_mps,
                prototype_cell_grad_rgb,
                hidden_weight_mps,
                hidden_bias_mps,
                output_weight_mps,
                output_bias_mps,
                sparse_render.tile_counts,
                sparse_render.tile_tube_ids,
                sparse_render.tile_depths,
                sparse_render.tile_unstable,
                config,
                backward_mode="target_area_star_only_vec4_wt",
            )
            _sync()
            prototype_native_star_backward_ms = (time.perf_counter() - t8) * 1000.0

            prototype_colorizer_errors = []
            for got, expected in (
                (prototype_grad_hidden_weight, native_grads.grad_hidden_weight),
                (prototype_grad_hidden_bias, native_grads.grad_hidden_bias),
                (prototype_grad_output_weight, native_grads.grad_output_weight),
                (prototype_grad_output_bias, native_grads.grad_output_bias),
            ):
                if expected is not None:
                    prototype_colorizer_errors.append(_max_err(got, expected))
            prototype_star_errors = [
                _max_err(prototype_star_grads.grad_ma, native_grads.grad_ma),
                _max_err(prototype_star_grads.grad_q_uvt, native_grads.grad_q_uvt),
                _max_err(prototype_star_grads.grad_opacity, native_grads.grad_opacity),
                _max_err(prototype_star_grads.grad_feature, native_grads.grad_feature),
            ]
            finite = finite and _finite_tensors(
                (
                    sparse_render.feature_values,
                    sparse_render.alpha_values,
                    prototype_loss,
                    prototype_cell_grad_rgb,
                    prototype_grad_hidden_weight,
                    prototype_grad_hidden_bias,
                    prototype_grad_output_weight,
                    prototype_grad_output_bias,
                    prototype_star_grads.grad_ma,
                    prototype_star_grads.grad_q_uvt,
                    prototype_star_grads.grad_opacity,
                    prototype_star_grads.grad_feature,
                )
            )
        else:
            prototype_sparse_forward_ms = 0.0
            prototype_colorizer_reduce_ms = 0.0
            prototype_native_star_backward_ms = 0.0
            prototype_colorizer_errors = []
            prototype_star_errors = []

        if iteration >= warmup:
            samples["baseline_render_ms"].append(baseline_render_ms)
            samples["baseline_loss_ms"].append(baseline_loss_ms)
            samples["baseline_backward_ms"].append(baseline_backward_ms)
            samples["native_bin_ms"].append(native_bin_ms)
            samples["native_forward_loss_ms"].append(native_forward_loss_ms)
            samples["native_backward_ms"].append(native_backward_ms)
            if include_torch_reducer_prototype:
                samples["prototype_sparse_forward_ms"].append(prototype_sparse_forward_ms)
                samples["prototype_colorizer_reduce_ms"].append(prototype_colorizer_reduce_ms)
                samples["prototype_native_star_backward_ms"].append(prototype_native_star_backward_ms)
                samples["prototype_colorizer_max_error"].append(max(prototype_colorizer_errors) if prototype_colorizer_errors else 0.0)
                samples["prototype_star_max_error"].append(max(prototype_star_errors) if prototype_star_errors else 0.0)
            if baseline_loss is not None:
                samples["loss_error"].append(_max_err(native_loss.reshape(()), baseline_loss.detach()))
            tile_overflow_sum = int(bins.tile_overflow.sum().cpu().item())
            tile_unstable_sum = int(native_grads.tile_unstable.sum().cpu().item())
            finite = finite and _finite_tensors(
                (
                    forward.pred_sums,
                    native_loss,
                    native_grads.grad_ma,
                    native_grads.grad_q_uvt,
                    native_grads.grad_opacity,
                    native_grads.grad_feature,
                )
            )

    native_bin_ms = _mean(samples["native_bin_ms"])
    native_forward_loss_ms = _mean(samples["native_forward_loss_ms"])
    native_backward_ms = _mean(samples["native_backward_ms"])
    baseline_render_ms = _mean(samples["baseline_render_ms"]) if include_baseline else None
    baseline_loss_ms = _mean(samples["baseline_loss_ms"]) if include_baseline else None
    baseline_backward_ms = _mean(samples["baseline_backward_ms"]) if include_baseline else None
    native_total_ms = native_bin_ms + native_forward_loss_ms + native_backward_ms
    baseline_total_ms = None
    if include_baseline and baseline_render_ms is not None and baseline_loss_ms is not None and baseline_backward_ms is not None:
        baseline_total_ms = baseline_render_ms + baseline_loss_ms + baseline_backward_ms
    prototype_sparse_forward_ms = _mean(samples["prototype_sparse_forward_ms"]) if include_torch_reducer_prototype else None
    prototype_colorizer_reduce_ms = _mean(samples["prototype_colorizer_reduce_ms"]) if include_torch_reducer_prototype else None
    prototype_native_star_backward_ms = (
        _mean(samples["prototype_native_star_backward_ms"]) if include_torch_reducer_prototype else None
    )
    prototype_total_ms = None
    if (
        prototype_sparse_forward_ms is not None
        and prototype_colorizer_reduce_ms is not None
        and prototype_native_star_backward_ms is not None
    ):
        prototype_total_ms = (
            prototype_sparse_forward_ms + prototype_colorizer_reduce_ms + prototype_native_star_backward_ms
        )
    prototype_ok = (
        not include_torch_reducer_prototype
        or (
            samples["prototype_colorizer_max_error"]
            and samples["prototype_star_max_error"]
            and max(samples["prototype_colorizer_max_error"]) <= 5.0e-4
            and max(samples["prototype_star_max_error"]) <= 5.0e-4
        )
    )
    return {
        "frames": frames,
        "size": size,
        "tubes": tubes,
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "backward_mode": backward_mode,
        "grid_side": grid_side,
        "patch_size": patch_size,
        "tile_capacity": tile_capacity,
        "cell_count": cell_count,
        "patch_area": patch_area,
        "sparse_count": int(pixel_ids.numel()),
        "include_baseline": include_baseline,
        "warmup": warmup,
        "repeat": repeat,
        "samples": samples,
        "native_bin_ms": native_bin_ms,
        "native_forward_loss_ms": native_forward_loss_ms,
        "native_backward_ms": native_backward_ms,
        "native_total_ms": native_total_ms,
        "baseline_render_ms": baseline_render_ms,
        "baseline_loss_ms": baseline_loss_ms,
        "baseline_backward_ms": baseline_backward_ms,
        "baseline_total_ms": baseline_total_ms,
        "native_vs_baseline_speedup": None if baseline_total_ms is None else baseline_total_ms / max(native_total_ms, 1.0e-9),
        "prototype_sparse_forward_ms": prototype_sparse_forward_ms,
        "prototype_colorizer_reduce_ms": prototype_colorizer_reduce_ms,
        "prototype_native_star_backward_ms": prototype_native_star_backward_ms,
        "prototype_total_ms": prototype_total_ms,
        "prototype_vs_native_speedup": None if prototype_total_ms is None else native_total_ms / max(prototype_total_ms, 1.0e-9),
        "prototype_colorizer_max_error": None
        if not samples["prototype_colorizer_max_error"]
        else max(samples["prototype_colorizer_max_error"]),
        "prototype_star_max_error": None
        if not samples["prototype_star_max_error"]
        else max(samples["prototype_star_max_error"]),
        "max_loss_error": None if not samples["loss_error"] else max(samples["loss_error"]),
        "tile_overflow_sum": tile_overflow_sum,
        "tile_unstable_sum": tile_unstable_sum,
        "finite": finite,
        "pass": finite
        and tile_overflow_sum == 0
        and (not samples["loss_error"] or max(samples["loss_error"]) <= 5.0e-4)
        and prototype_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dims", default="4,32")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--timing-frames", type=int, default=64)
    parser.add_argument("--timing-size", type=int, default=128)
    parser.add_argument("--timing-tubes", type=int, default=8192)
    parser.add_argument("--timing-feature-dim", type=int, default=32)
    parser.add_argument("--tile-capacity", type=int, default=128)
    parser.add_argument("--grid-side", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--timing-warmup", type=int, default=1)
    parser.add_argument("--timing-repeat", type=int, default=3)
    parser.add_argument(
        "--backward-mode",
        choices=TARGET_AREA_BACKWARD_MODES,
        default="target_area_star_only",
    )
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--include-torch-reducer-prototype", action="store_true")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("sparse hidden target-area benchmark requires MPS")

    feature_dims = split_csv_ints(args.feature_dims)
    timing_patch_size = args.patch_size if args.patch_size > 0 else args.timing_size // args.grid_side
    result: dict[str, Any] = {
        "gate": "star_uvt_sparse_hidden_target_area_native",
        "backward_mode": args.backward_mode,
        "tiny_parity": [run_tiny_parity(feature_dim, args.hidden_dim, args.backward_mode) for feature_dim in feature_dims],
    }
    if not args.skip_timing:
        result["timing"] = run_timing_case(
            frames=args.timing_frames,
            size=args.timing_size,
            tubes=args.timing_tubes,
            feature_dim=args.timing_feature_dim,
            hidden_dim=args.hidden_dim,
            grid_side=args.grid_side,
            patch_size=timing_patch_size,
            tile_capacity=args.tile_capacity,
            seed=args.seed,
            warmup=args.timing_warmup,
            repeat=args.timing_repeat,
            include_baseline=not args.skip_baseline,
            backward_mode=args.backward_mode,
            include_torch_reducer_prototype=bool(args.include_torch_reducer_prototype),
        )
    result["pass"] = all(row["pass"] for row in result["tiny_parity"]) and result.get("timing", {}).get("pass", True)

    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out_json is not None:
        write_report_json(args.out_json, result)
    print(payload)


if __name__ == "__main__":
    main()
