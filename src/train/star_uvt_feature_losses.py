from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from colorize import FeatureToColor
from star_uvt_common import target_grid_slice_for_render_chunk as _target_grid_slice_for_render_chunk
from star_uvt_feature_targets import FeatureTargetTensor, _adapt_render_to_feature_target
from star_uvt_runtime import sync_device as _sync_device
from star_uvt_sparse_grid import (
    SparseImageVjpPack,
    _batched_pack_sparse_target_grid_vjp,
    _batched_sparse_feature_values_to_target_grid,
    _pack_sparse_target_grid_vjp,
    _sparse_feature_values_to_target_grid,
)
from star_uvt_sparse_visual_losses import _add_param_grad


@dataclass(frozen=True)
class ManualImageVjpResult:
    loss: torch.Tensor
    grad_feature_image: torch.Tensor | None
    sparse_pack: SparseImageVjpPack | None
    feature_target_loss: float
    rgb_grid_loss: float
    rgb_probe_loss: float
    target_start: int | None
    target_frames: int | None
    rgb_grid_target_start: int | None
    rgb_grid_target_frames: int | None
    probe_target_start: int | None
    probe_target_frames: int | None
    feature_target_ms: float
    rgb_grid_loss_ms: float
    rgb_probe_loss_ms: float
    image_vjp_ms: float


@dataclass(frozen=True)
class BatchedSparseTargetGridVjpResult:
    loss: torch.Tensor
    sparse_packs: tuple[SparseImageVjpPack, ...]
    feature_target_loss: float
    rgb_grid_loss: float
    rgb_probe_loss: float
    feature_target_ms: float
    rgb_grid_loss_ms: float
    rgb_probe_loss_ms: float
    image_vjp_ms: float


def _feature_target_loss(rendered: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "mse":
        return (rendered - target).square().sum()
    if loss_type == "l1":
        return (rendered - target).abs().sum()
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(rendered, target, reduction="sum")
    raise ValueError(f"Unsupported feature target loss_type={loss_type!r}.")


def _gelu_derivative_exact(x: torch.Tensor) -> torch.Tensor:
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    inv_sqrt2pi = 1.0 / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(x * inv_sqrt2))
    pdf = torch.exp(-0.5 * x.square()) * inv_sqrt2pi
    return cdf + x * pdf


def _manual_rgb_probe_loss_and_grid_grad(
    rgb_probe: FeatureToColor,
    rendered_target_grid: torch.Tensor,
    target_rgb_probe_chunk: torch.Tensor,
    *,
    total_rgb_probe_loss_elems: int,
    loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rgb_probe.pre_norm is not None:
        raise ValueError("feature_target.image_vjp_mode=analytic requires RGB probe pre_norm=false")
    if rgb_probe.activation != "sigmoid":
        raise ValueError("feature_target.image_vjp_mode=analytic requires RGB probe activation=sigmoid")
    if rgb_probe.view_condition != "none":
        raise ValueError("feature_target.image_vjp_mode=analytic requires RGB probe view_condition=none")
    net = rgb_probe.net
    if not isinstance(net, nn.Sequential) or len(net) != 3:
        raise ValueError("feature_target.image_vjp_mode=analytic requires Conv2d -> GELU -> Conv2d RGB probe")
    conv1, activation, conv2 = net
    if not isinstance(conv1, nn.Conv2d) or not isinstance(activation, nn.GELU) or not isinstance(conv2, nn.Conv2d):
        raise ValueError("feature_target.image_vjp_mode=analytic requires Conv2d -> GELU -> Conv2d RGB probe")
    if conv1.kernel_size != (1, 1) or conv2.kernel_size != (1, 1):
        raise ValueError("feature_target.image_vjp_mode=analytic requires 1x1 RGB-probe convolutions")
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


def _trainable_colorizer_grid_loss_and_grid_grad(
    colorizer: FeatureToColor,
    rendered_target_grid: torch.Tensor,
    target_rgb_grid_chunk: torch.Tensor,
    *,
    total_rgb_grid_loss_elems: int,
    loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_grid = rendered_target_grid.detach().requires_grad_(True)
    rgb = colorizer(local_grid)
    diff = rgb - target_rgb_grid_chunk
    loss = diff.square().sum() / float(total_rgb_grid_loss_elems)
    params = tuple(colorizer.parameters())
    grads = torch.autograd.grad(float(loss_weight) * loss, (local_grid, *params), allow_unused=True)
    grad_grid = grads[0]
    if grad_grid is None:
        raise RuntimeError("trainable colorizer grid loss did not produce a grid gradient")
    for param, grad in zip(params, grads[1:], strict=True):
        if grad is not None:
            _add_param_grad(param, grad)
    return loss.detach(), grad_grid.contiguous()


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


def _pack_sparse_image_vjp(grad_feature_image: torch.Tensor, grad_alpha: torch.Tensor) -> SparseImageVjpPack:
    if grad_feature_image.ndim != 4:
        raise ValueError("grad_feature_image must have shape [frames,feature_dim,height,width]")
    if grad_alpha.shape != (grad_feature_image.shape[0], grad_feature_image.shape[2], grad_feature_image.shape[3]):
        raise ValueError("grad_alpha must have shape [frames,height,width]")
    pixel_mask = grad_feature_image.abs().amax(dim=1) > 0.0
    pixel_mask = pixel_mask | (grad_alpha.abs() > 0.0)
    pixel_ids_long = torch.where(pixel_mask.reshape(-1))[0]
    feature_dim = int(grad_feature_image.shape[1])
    grad_feature_values = (
        grad_feature_image.permute(0, 2, 3, 1).reshape(-1, feature_dim).index_select(0, pixel_ids_long).contiguous()
    )
    grad_alpha_values = grad_alpha.reshape(-1).index_select(0, pixel_ids_long).contiguous()
    return SparseImageVjpPack(
        pixel_ids=pixel_ids_long.to(torch.int32).contiguous(),
        grad_feature_values=grad_feature_values,
        grad_alpha_values=grad_alpha_values,
        pixel_count=int(pixel_ids_long.numel()),
        total_pixels=int(pixel_mask.numel()),
    )


def _manual_target_grid_loss_and_vjp(
    rendered_feature_image: torch.Tensor,
    *,
    target_feature: FeatureTargetTensor,
    colorizer: FeatureToColor,
    rgb_grid_target: torch.Tensor | None,
    rgb_probe: FeatureToColor | None,
    rgb_probe_target: torch.Tensor | None,
    feature_config: Any,
    frame_start: int,
    chunk_frames: int,
    feature_loss_type: str,
    feature_loss_weight: float,
    rgb_grid_loss_weight: float,
    rgb_probe_loss_weight: float,
    total_feature_loss_elems: int,
    total_rgb_grid_loss_elems: int,
    total_rgb_probe_loss_elems: int,
    device: torch.device,
    image_vjp_mode: str,
) -> ManualImageVjpResult:
    if target_feature.materialization != "target_grid":
        raise ValueError("feature_target.image_vjp_mode=analytic requires materialization=target_grid")
    if target_feature.source is None:
        raise RuntimeError("feature_target.image_vjp_mode=analytic requires target-grid source tensor")
    if feature_loss_type != "mse":
        raise ValueError("feature_target.image_vjp_mode=analytic currently requires loss_type=mse")
    _sync_device(device)
    target_t0 = time.perf_counter()
    target_start, target_frames = _target_grid_slice_for_render_chunk(
        target_frames=int(target_feature.source.shape[0]),
        render_frames=int(feature_config.frames),
        frame_start=frame_start,
        chunk_frames=chunk_frames,
    )
    target_feature_chunk = target_feature.chunk(frame_start, chunk_frames)
    rendered_target_grid = _adapt_render_to_feature_target(
        rendered_feature_image,
        target_shape=tuple(int(item) for item in target_feature_chunk.shape),
        mode=target_feature.grid_mode,
    )
    _sync_device(device)
    target_t1 = time.perf_counter()
    feature_target_ms = (target_t1 - target_t0) * 1000.0
    loss = rendered_feature_image.new_zeros(())
    grad_target_grid = torch.zeros_like(rendered_target_grid)
    feature_target_loss_value = 0.0
    rgb_grid_loss_value = 0.0
    rgb_grid_loss_ms = 0.0
    rgb_probe_loss_value = 0.0
    rgb_probe_loss_ms = 0.0
    if feature_loss_weight > 0.0:
        diff = rendered_target_grid - target_feature_chunk
        feature_target_loss = diff.square().sum() / float(total_feature_loss_elems)
        loss = loss + float(feature_loss_weight) * feature_target_loss
        grad_target_grid = grad_target_grid + (
            2.0 * float(feature_loss_weight) / float(total_feature_loss_elems)
        ) * diff
        feature_target_loss_value = float(feature_target_loss.detach().cpu().item())
    rgb_grid_target_start: int | None = None
    rgb_grid_target_frames: int | None = None
    if rgb_grid_loss_weight > 0.0:
        if rgb_grid_target is None:
            raise RuntimeError("RGB-grid colorizer loss missing target")
        _sync_device(device)
        rgb_grid_t0 = time.perf_counter()
        rgb_grid_target_start, rgb_grid_target_frames = _target_grid_slice_for_render_chunk(
            target_frames=int(rgb_grid_target.shape[0]),
            render_frames=int(feature_config.frames),
            frame_start=frame_start,
            chunk_frames=chunk_frames,
        )
        target_rgb_grid_chunk = rgb_grid_target[
            rgb_grid_target_start : rgb_grid_target_start + rgb_grid_target_frames
        ]
        rgb_grid_loss, grid_colorizer_grad = _trainable_colorizer_grid_loss_and_grid_grad(
            colorizer,
            rendered_target_grid,
            target_rgb_grid_chunk,
            total_rgb_grid_loss_elems=total_rgb_grid_loss_elems,
            loss_weight=rgb_grid_loss_weight,
        )
        loss = loss + float(rgb_grid_loss_weight) * rgb_grid_loss
        grad_target_grid = grad_target_grid + grid_colorizer_grad
        rgb_grid_loss_value = float(rgb_grid_loss.detach().cpu().item())
        _sync_device(device)
        rgb_grid_loss_ms = (time.perf_counter() - rgb_grid_t0) * 1000.0
    probe_target_start: int | None = None
    probe_target_frames: int | None = None
    if rgb_probe is not None and rgb_probe_loss_weight > 0.0:
        if rgb_probe_target is None:
            raise RuntimeError("RGB probe loss missing target")
        _sync_device(device)
        probe_t0 = time.perf_counter()
        probe_target_start, probe_target_frames = _target_grid_slice_for_render_chunk(
            target_frames=int(rgb_probe_target.shape[0]),
            render_frames=int(feature_config.frames),
            frame_start=frame_start,
            chunk_frames=chunk_frames,
        )
        target_rgb_probe_chunk = rgb_probe_target[probe_target_start : probe_target_start + probe_target_frames]
        rgb_probe_loss, probe_grad_grid = _manual_rgb_probe_loss_and_grid_grad(
            rgb_probe,
            rendered_target_grid,
            target_rgb_probe_chunk,
            total_rgb_probe_loss_elems=total_rgb_probe_loss_elems,
            loss_weight=rgb_probe_loss_weight,
        )
        loss = loss + float(rgb_probe_loss_weight) * rgb_probe_loss
        grad_target_grid = grad_target_grid + probe_grad_grid
        rgb_probe_loss_value = float(rgb_probe_loss.detach().cpu().item())
        _sync_device(device)
        rgb_probe_loss_ms = (time.perf_counter() - probe_t0) * 1000.0
    _sync_device(device)
    vjp_t0 = time.perf_counter()
    grad_feature_image: torch.Tensor | None
    sparse_pack: SparseImageVjpPack | None
    if image_vjp_mode == "analytic_sparse_grid":
        grad_feature_image = None
        sparse_pack = _pack_sparse_target_grid_vjp(
            grad_target_grid,
            input_shape=tuple(int(item) for item in rendered_feature_image.shape),
            mode=target_feature.grid_mode,
        )
    else:
        grad_feature_image = _render_grid_vjp_to_feature_image(
            grad_target_grid,
            input_shape=tuple(int(item) for item in rendered_feature_image.shape),
            mode=target_feature.grid_mode,
        )
        sparse_pack = None
    _sync_device(device)
    image_vjp_ms = (time.perf_counter() - vjp_t0) * 1000.0
    return ManualImageVjpResult(
        loss=loss,
        grad_feature_image=grad_feature_image,
        sparse_pack=sparse_pack,
        feature_target_loss=feature_target_loss_value,
        rgb_grid_loss=rgb_grid_loss_value,
        rgb_probe_loss=rgb_probe_loss_value,
        target_start=target_start,
        target_frames=target_frames,
        rgb_grid_target_start=rgb_grid_target_start,
        rgb_grid_target_frames=rgb_grid_target_frames,
        probe_target_start=probe_target_start,
        probe_target_frames=probe_target_frames,
        feature_target_ms=feature_target_ms,
        rgb_grid_loss_ms=rgb_grid_loss_ms,
        rgb_probe_loss_ms=rgb_probe_loss_ms,
        image_vjp_ms=image_vjp_ms,
    )


def _manual_sparse_target_grid_loss_and_vjp(
    sparse_feature_values: torch.Tensor,
    *,
    target_feature: FeatureTargetTensor,
    colorizer: FeatureToColor,
    rgb_grid_target: torch.Tensor | None,
    rgb_probe: FeatureToColor | None,
    rgb_probe_target: torch.Tensor | None,
    feature_config: Any,
    frame_start: int,
    chunk_frames: int,
    feature_loss_type: str,
    feature_loss_weight: float,
    rgb_grid_loss_weight: float,
    rgb_probe_loss_weight: float,
    total_feature_loss_elems: int,
    total_rgb_grid_loss_elems: int,
    total_rgb_probe_loss_elems: int,
    device: torch.device,
) -> ManualImageVjpResult:
    if target_feature.materialization != "target_grid":
        raise ValueError("feature_target.image_vjp_mode=analytic_sparse_grid_forward requires materialization=target_grid")
    if target_feature.source is None:
        raise RuntimeError("feature_target.image_vjp_mode=analytic_sparse_grid_forward requires target-grid source tensor")
    if feature_loss_type != "mse":
        raise ValueError("feature_target.image_vjp_mode=analytic_sparse_grid_forward currently requires loss_type=mse")
    input_shape = (
        int(chunk_frames),
        int(feature_config.feature_dim),
        int(feature_config.height),
        int(feature_config.width),
    )
    _sync_device(device)
    target_t0 = time.perf_counter()
    target_start, target_frames = _target_grid_slice_for_render_chunk(
        target_frames=int(target_feature.source.shape[0]),
        render_frames=int(feature_config.frames),
        frame_start=frame_start,
        chunk_frames=chunk_frames,
    )
    target_feature_chunk = target_feature.chunk(frame_start, chunk_frames)
    rendered_target_grid = _sparse_feature_values_to_target_grid(
        sparse_feature_values,
        input_shape=input_shape,
        target_shape=tuple(int(item) for item in target_feature_chunk.shape),
        mode=target_feature.grid_mode,
    )
    _sync_device(device)
    target_t1 = time.perf_counter()
    feature_target_ms = (target_t1 - target_t0) * 1000.0
    loss = sparse_feature_values.new_zeros(())
    grad_target_grid = torch.zeros_like(rendered_target_grid)
    feature_target_loss_value = 0.0
    rgb_grid_loss_value = 0.0
    rgb_grid_loss_ms = 0.0
    rgb_probe_loss_value = 0.0
    rgb_probe_loss_ms = 0.0
    if feature_loss_weight > 0.0:
        diff = rendered_target_grid - target_feature_chunk
        feature_target_loss = diff.square().sum() / float(total_feature_loss_elems)
        loss = loss + float(feature_loss_weight) * feature_target_loss
        grad_target_grid = grad_target_grid + (
            2.0 * float(feature_loss_weight) / float(total_feature_loss_elems)
        ) * diff
        feature_target_loss_value = float(feature_target_loss.detach().cpu().item())
    rgb_grid_target_start: int | None = None
    rgb_grid_target_frames: int | None = None
    if rgb_grid_loss_weight > 0.0:
        if rgb_grid_target is None:
            raise RuntimeError("RGB-grid colorizer loss missing target")
        _sync_device(device)
        rgb_grid_t0 = time.perf_counter()
        rgb_grid_target_start, rgb_grid_target_frames = _target_grid_slice_for_render_chunk(
            target_frames=int(rgb_grid_target.shape[0]),
            render_frames=int(feature_config.frames),
            frame_start=frame_start,
            chunk_frames=chunk_frames,
        )
        target_rgb_grid_chunk = rgb_grid_target[
            rgb_grid_target_start : rgb_grid_target_start + rgb_grid_target_frames
        ]
        rgb_grid_loss, grid_colorizer_grad = _trainable_colorizer_grid_loss_and_grid_grad(
            colorizer,
            rendered_target_grid,
            target_rgb_grid_chunk,
            total_rgb_grid_loss_elems=total_rgb_grid_loss_elems,
            loss_weight=rgb_grid_loss_weight,
        )
        loss = loss + float(rgb_grid_loss_weight) * rgb_grid_loss
        grad_target_grid = grad_target_grid + grid_colorizer_grad
        rgb_grid_loss_value = float(rgb_grid_loss.detach().cpu().item())
        _sync_device(device)
        rgb_grid_loss_ms = (time.perf_counter() - rgb_grid_t0) * 1000.0
    probe_target_start: int | None = None
    probe_target_frames: int | None = None
    if rgb_probe is not None and rgb_probe_loss_weight > 0.0:
        if rgb_probe_target is None:
            raise RuntimeError("RGB probe loss missing target")
        _sync_device(device)
        probe_t0 = time.perf_counter()
        probe_target_start, probe_target_frames = _target_grid_slice_for_render_chunk(
            target_frames=int(rgb_probe_target.shape[0]),
            render_frames=int(feature_config.frames),
            frame_start=frame_start,
            chunk_frames=chunk_frames,
        )
        target_rgb_probe_chunk = rgb_probe_target[probe_target_start : probe_target_start + probe_target_frames]
        rgb_probe_loss, probe_grad_grid = _manual_rgb_probe_loss_and_grid_grad(
            rgb_probe,
            rendered_target_grid,
            target_rgb_probe_chunk,
            total_rgb_probe_loss_elems=total_rgb_probe_loss_elems,
            loss_weight=rgb_probe_loss_weight,
        )
        loss = loss + float(rgb_probe_loss_weight) * rgb_probe_loss
        grad_target_grid = grad_target_grid + probe_grad_grid
        rgb_probe_loss_value = float(rgb_probe_loss.detach().cpu().item())
        _sync_device(device)
        rgb_probe_loss_ms = (time.perf_counter() - probe_t0) * 1000.0
    _sync_device(device)
    vjp_t0 = time.perf_counter()
    sparse_pack = _pack_sparse_target_grid_vjp(
        grad_target_grid,
        input_shape=input_shape,
        mode=target_feature.grid_mode,
    )
    _sync_device(device)
    image_vjp_ms = (time.perf_counter() - vjp_t0) * 1000.0
    return ManualImageVjpResult(
        loss=loss,
        grad_feature_image=None,
        sparse_pack=sparse_pack,
        feature_target_loss=feature_target_loss_value,
        rgb_grid_loss=rgb_grid_loss_value,
        rgb_probe_loss=rgb_probe_loss_value,
        target_start=target_start,
        target_frames=target_frames,
        rgb_grid_target_start=rgb_grid_target_start,
        rgb_grid_target_frames=rgb_grid_target_frames,
        probe_target_start=probe_target_start,
        probe_target_frames=probe_target_frames,
        feature_target_ms=feature_target_ms,
        rgb_grid_loss_ms=rgb_grid_loss_ms,
        rgb_probe_loss_ms=rgb_probe_loss_ms,
        image_vjp_ms=image_vjp_ms,
    )


def _manual_batched_sparse_target_grid_loss_and_vjp(
    sparse_feature_values: list[torch.Tensor],
    *,
    target_feature: FeatureTargetTensor,
    colorizer: FeatureToColor,
    rgb_grid_target: torch.Tensor | None,
    rgb_probe: FeatureToColor | None,
    rgb_probe_target: torch.Tensor | None,
    feature_config: Any,
    frame_starts: list[int],
    chunk_frames: list[int],
    feature_loss_type: str,
    feature_loss_weight: float,
    rgb_grid_loss_weight: float,
    rgb_probe_loss_weight: float,
    total_feature_loss_elems: int,
    total_rgb_grid_loss_elems: int,
    total_rgb_probe_loss_elems: int,
    device: torch.device,
) -> BatchedSparseTargetGridVjpResult:
    if not sparse_feature_values:
        raise ValueError("sparse_feature_values must be non-empty")
    if target_feature.materialization != "target_grid":
        raise ValueError(
            "feature_target.image_vjp_mode=analytic_sparse_grid_forward_batched requires materialization=target_grid"
        )
    if target_feature.source is None:
        raise RuntimeError(
            "feature_target.image_vjp_mode=analytic_sparse_grid_forward_batched requires target-grid source tensor"
        )
    if feature_loss_type != "mse":
        raise ValueError("feature_target.image_vjp_mode=analytic_sparse_grid_forward_batched currently requires loss_type=mse")
    if len(sparse_feature_values) != len(frame_starts) or len(frame_starts) != len(chunk_frames):
        raise ValueError("sparse values, frame starts, and chunk frame counts must have matching lengths")
    input_shapes: list[tuple[int, int, int, int]] = []
    target_shapes: list[tuple[int, int, int, int]] = []
    target_chunks: list[torch.Tensor] = []
    rgb_grid_chunks: list[torch.Tensor] = []
    probe_chunks: list[torch.Tensor] = []
    for frame_start, frames in zip(frame_starts, chunk_frames, strict=True):
        input_shapes.append(
            (
                int(frames),
                int(feature_config.feature_dim),
                int(feature_config.height),
                int(feature_config.width),
            )
        )
        target_chunk = target_feature.chunk(frame_start, frames)
        target_shapes.append(tuple(int(item) for item in target_chunk.shape))
        target_chunks.append(target_chunk)
        if rgb_grid_target is not None:
            rgb_grid_target_start, rgb_grid_target_frames = _target_grid_slice_for_render_chunk(
                target_frames=int(rgb_grid_target.shape[0]),
                render_frames=int(feature_config.frames),
                frame_start=frame_start,
                chunk_frames=frames,
            )
            rgb_grid_chunks.append(
                rgb_grid_target[rgb_grid_target_start : rgb_grid_target_start + rgb_grid_target_frames]
            )
        if rgb_probe_target is not None:
            probe_target_start, probe_target_frames = _target_grid_slice_for_render_chunk(
                target_frames=int(rgb_probe_target.shape[0]),
                render_frames=int(feature_config.frames),
                frame_start=frame_start,
                chunk_frames=frames,
            )
            probe_chunks.append(rgb_probe_target[probe_target_start : probe_target_start + probe_target_frames])
    if any(shape != input_shapes[0] for shape in input_shapes) or any(shape != target_shapes[0] for shape in target_shapes):
        raise ValueError("batched sparse target-grid VJP requires equal render chunk and target chunk shapes")

    _sync_device(device)
    target_t0 = time.perf_counter()
    batched_values = torch.stack(sparse_feature_values, dim=0).contiguous()
    target_feature_chunk = torch.cat(target_chunks, dim=0).contiguous()
    rendered_target_grid = _batched_sparse_feature_values_to_target_grid(
        batched_values,
        input_shape=input_shapes[0],
        target_shape=target_shapes[0],
        mode=target_feature.grid_mode,
    )
    _sync_device(device)
    target_t1 = time.perf_counter()
    feature_target_ms = (target_t1 - target_t0) * 1000.0
    loss = batched_values.new_zeros(())
    grad_target_grid = torch.zeros_like(rendered_target_grid)
    feature_target_loss_value = 0.0
    rgb_grid_loss_value = 0.0
    rgb_probe_loss_value = 0.0
    if feature_loss_weight > 0.0:
        _sync_device(device)
        feature_t0 = time.perf_counter()
        diff = rendered_target_grid - target_feature_chunk
        feature_target_loss = diff.square().sum() / float(total_feature_loss_elems)
        loss = loss + float(feature_loss_weight) * feature_target_loss
        grad_target_grid = grad_target_grid + (
            2.0 * float(feature_loss_weight) / float(total_feature_loss_elems)
        ) * diff
        feature_target_loss_value = float(feature_target_loss.detach().cpu().item())
        _sync_device(device)
        feature_target_ms += (time.perf_counter() - feature_t0) * 1000.0

    rgb_grid_loss_ms = 0.0
    if rgb_grid_loss_weight > 0.0:
        if rgb_grid_target is None:
            raise RuntimeError("RGB-grid colorizer loss missing target")
        _sync_device(device)
        rgb_grid_t0 = time.perf_counter()
        target_rgb_grid_chunk = torch.cat(rgb_grid_chunks, dim=0).contiguous()
        rgb_grid_loss, grid_colorizer_grad = _trainable_colorizer_grid_loss_and_grid_grad(
            colorizer,
            rendered_target_grid,
            target_rgb_grid_chunk,
            total_rgb_grid_loss_elems=total_rgb_grid_loss_elems,
            loss_weight=rgb_grid_loss_weight,
        )
        loss = loss + float(rgb_grid_loss_weight) * rgb_grid_loss
        grad_target_grid = grad_target_grid + grid_colorizer_grad
        rgb_grid_loss_value = float(rgb_grid_loss.detach().cpu().item())
        _sync_device(device)
        rgb_grid_loss_ms = (time.perf_counter() - rgb_grid_t0) * 1000.0

    rgb_probe_loss_ms = 0.0
    if rgb_probe is not None and rgb_probe_loss_weight > 0.0:
        if rgb_probe_target is None:
            raise RuntimeError("RGB probe loss missing target")
        _sync_device(device)
        probe_t0 = time.perf_counter()
        target_rgb_probe_chunk = torch.cat(probe_chunks, dim=0).contiguous()
        rgb_probe_loss, probe_grad_grid = _manual_rgb_probe_loss_and_grid_grad(
            rgb_probe,
            rendered_target_grid,
            target_rgb_probe_chunk,
            total_rgb_probe_loss_elems=total_rgb_probe_loss_elems,
            loss_weight=rgb_probe_loss_weight,
        )
        loss = loss + float(rgb_probe_loss_weight) * rgb_probe_loss
        grad_target_grid = grad_target_grid + probe_grad_grid
        rgb_probe_loss_value = float(rgb_probe_loss.detach().cpu().item())
        _sync_device(device)
        rgb_probe_loss_ms = (time.perf_counter() - probe_t0) * 1000.0

    _sync_device(device)
    vjp_t0 = time.perf_counter()
    sparse_packs = _batched_pack_sparse_target_grid_vjp(
        grad_target_grid,
        input_shape=input_shapes[0],
        target_shape=target_shapes[0],
        mode=target_feature.grid_mode,
    )
    _sync_device(device)
    image_vjp_ms = (time.perf_counter() - vjp_t0) * 1000.0
    return BatchedSparseTargetGridVjpResult(
        loss=loss,
        sparse_packs=sparse_packs,
        feature_target_loss=feature_target_loss_value,
        rgb_grid_loss=rgb_grid_loss_value,
        rgb_probe_loss=rgb_probe_loss_value,
        feature_target_ms=feature_target_ms,
        rgb_grid_loss_ms=rgb_grid_loss_ms,
        rgb_probe_loss_ms=rgb_probe_loss_ms,
        image_vjp_ms=image_vjp_ms,
    )
