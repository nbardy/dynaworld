from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch


@dataclass(frozen=True)
class SparseImageVjpPack:
    pixel_ids: torch.Tensor
    grad_feature_values: torch.Tensor
    grad_alpha_values: torch.Tensor
    pixel_count: int
    total_pixels: int


@dataclass(frozen=True)
class SparseTargetGridVjpDevicePlan:
    source_pixel_ids: torch.Tensor
    target_flat_ids: torch.Tensor
    weights: torch.Tensor
    unique_pixel_ids: torch.Tensor
    inverse: torch.Tensor
    has_duplicate_pixels: bool
    total_pixels: int


_TARGET_GRID_SPARSE_VJP_DEVICE_PLAN_CACHE: dict[
    tuple[tuple[int, int, int, int], tuple[int, int, int, int], str, str, str],
    SparseTargetGridVjpDevicePlan,
] = {}


def _linear_interp_plan_1d_cpu(input_size: int, output_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if input_size <= 0 or output_size <= 0:
        raise ValueError("input_size and output_size must be positive")
    out = torch.arange(output_size, dtype=torch.float32)
    real = (out + 0.5) * (float(input_size) / float(output_size)) - 0.5
    real = real.clamp(0.0, float(input_size - 1))
    idx0 = torch.floor(real).to(torch.int64)
    idx1 = (idx0 + 1).clamp(max=input_size - 1)
    w1 = real - idx0.to(real.dtype)
    w0 = 1.0 - w1
    return torch.stack((idx0, idx1), dim=1), torch.stack((w0, w1), dim=1)


@lru_cache(maxsize=128)
def _target_grid_sparse_vjp_plan_cpu(
    input_shape: tuple[int, int, int, int],
    target_shape: tuple[int, int, int, int],
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    if mode != "trilinear":
        raise ValueError("feature_target.image_vjp_mode=analytic_sparse_grid currently requires trilinear adapter")
    chunk_frames, feature_dim, height, width = (int(item) for item in input_shape)
    target_frames, target_feature_dim, target_height, target_width = (int(item) for item in target_shape)
    if feature_dim != target_feature_dim:
        raise ValueError(f"feature dimension mismatch: rendered={feature_dim}, target={target_feature_dim}")
    t_idx, t_weight = _linear_interp_plan_1d_cpu(chunk_frames, target_frames)
    y_idx, y_weight = _linear_interp_plan_1d_cpu(height, target_height)
    x_idx, x_weight = _linear_interp_plan_1d_cpu(width, target_width)
    target_ids = torch.arange(target_frames * target_height * target_width, dtype=torch.int64).reshape(
        target_frames,
        target_height,
        target_width,
    )
    pixel_id_parts: list[torch.Tensor] = []
    target_id_parts: list[torch.Tensor] = []
    weight_parts: list[torch.Tensor] = []
    for t_corner in range(2):
        for y_corner in range(2):
            for x_corner in range(2):
                src_t = t_idx[:, t_corner].view(target_frames, 1, 1)
                src_y = y_idx[:, y_corner].view(1, target_height, 1)
                src_x = x_idx[:, x_corner].view(1, 1, target_width)
                pixel_ids = ((src_t * height + src_y) * width + src_x).reshape(-1)
                weights = (
                    t_weight[:, t_corner].view(target_frames, 1, 1)
                    * y_weight[:, y_corner].view(1, target_height, 1)
                    * x_weight[:, x_corner].view(1, 1, target_width)
                ).reshape(-1)
                keep = weights != 0.0
                pixel_id_parts.append(pixel_ids[keep])
                target_id_parts.append(target_ids.reshape(-1)[keep])
                weight_parts.append(weights[keep])
    source_pixel_ids = torch.cat(pixel_id_parts).to(torch.int64)
    target_flat_ids = torch.cat(target_id_parts).to(torch.int64)
    weights = torch.cat(weight_parts).to(torch.float32)
    unique_pixel_ids, inverse = torch.unique(source_pixel_ids, sorted=True, return_inverse=True)
    has_duplicate_pixels = int(unique_pixel_ids.numel()) != int(source_pixel_ids.numel())
    return source_pixel_ids, target_flat_ids, weights, unique_pixel_ids, inverse.to(torch.int64), has_duplicate_pixels


def _target_grid_sparse_vjp_plan_device(
    input_shape: tuple[int, int, int, int],
    target_shape: tuple[int, int, int, int],
    mode: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> SparseTargetGridVjpDevicePlan:
    key = (
        tuple(int(item) for item in input_shape),
        tuple(int(item) for item in target_shape),
        str(mode),
        str(device),
        str(dtype),
    )
    cached = _TARGET_GRID_SPARSE_VJP_DEVICE_PLAN_CACHE.get(key)
    if cached is not None:
        return cached
    source_ids_cpu, target_ids_cpu, weights_cpu, unique_cpu, inverse_cpu, has_duplicate_pixels = (
        _target_grid_sparse_vjp_plan_cpu(input_shape, target_shape, mode)
    )
    plan = SparseTargetGridVjpDevicePlan(
        source_pixel_ids=source_ids_cpu.to(device=device, dtype=torch.int32).contiguous(),
        target_flat_ids=target_ids_cpu.to(device=device, dtype=torch.int64).contiguous(),
        weights=weights_cpu.to(device=device, dtype=dtype).contiguous(),
        unique_pixel_ids=unique_cpu.to(device=device, dtype=torch.int32).contiguous(),
        inverse=inverse_cpu.to(device=device, dtype=torch.int64).contiguous(),
        has_duplicate_pixels=bool(has_duplicate_pixels),
        total_pixels=int(int(input_shape[0]) * int(input_shape[2]) * int(input_shape[3])),
    )
    _TARGET_GRID_SPARSE_VJP_DEVICE_PLAN_CACHE[key] = plan
    return plan


def _pack_sparse_target_grid_vjp(
    grad_target_grid: torch.Tensor,
    *,
    input_shape: tuple[int, int, int, int],
    mode: str,
) -> SparseImageVjpPack:
    if grad_target_grid.ndim != 4:
        raise ValueError("grad_target_grid must have shape [frames,feature_dim,height,width]")
    chunk_frames, feature_dim, height, width = (int(item) for item in input_shape)
    target_shape = tuple(int(item) for item in grad_target_grid.shape)
    plan = _target_grid_sparse_vjp_plan_device(
        (chunk_frames, feature_dim, height, width),
        target_shape,
        mode,
        device=grad_target_grid.device,
        dtype=grad_target_grid.dtype,
    )
    target_values = grad_target_grid.permute(0, 2, 3, 1).reshape(-1, feature_dim)
    weighted_values = target_values.index_select(0, plan.target_flat_ids) * plan.weights[:, None]
    if plan.has_duplicate_pixels:
        grad_feature_values = torch.zeros(
            (int(plan.unique_pixel_ids.numel()), feature_dim),
            device=grad_target_grid.device,
            dtype=grad_target_grid.dtype,
        )
        grad_feature_values.index_add_(0, plan.inverse, weighted_values)
        pixel_ids = plan.unique_pixel_ids
        nonzero = grad_feature_values.abs().amax(dim=1) > 0.0
        pixel_ids = pixel_ids[nonzero]
        grad_feature_values = grad_feature_values[nonzero].contiguous()
    else:
        pixel_ids = plan.source_pixel_ids
        grad_feature_values = weighted_values.contiguous()
    grad_alpha_values = torch.zeros(
        (int(pixel_ids.numel()),),
        device=grad_target_grid.device,
        dtype=grad_target_grid.dtype,
    )
    return SparseImageVjpPack(
        pixel_ids=pixel_ids.contiguous(),
        grad_feature_values=grad_feature_values,
        grad_alpha_values=grad_alpha_values.contiguous(),
        pixel_count=int(pixel_ids.numel()),
        total_pixels=plan.total_pixels,
    )


def _sparse_target_grid_pixel_ids(
    *,
    input_shape: tuple[int, int, int, int],
    target_shape: tuple[int, int, int, int],
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    plan = _target_grid_sparse_vjp_plan_device(
        input_shape,
        target_shape,
        mode,
        device=device,
        dtype=torch.float32,
    )
    return plan.unique_pixel_ids


def _sparse_feature_values_to_target_grid(
    feature_values: torch.Tensor,
    *,
    input_shape: tuple[int, int, int, int],
    target_shape: tuple[int, int, int, int],
    mode: str,
) -> torch.Tensor:
    if feature_values.ndim != 2:
        raise ValueError("feature_values must have shape [M,feature_dim]")
    plan = _target_grid_sparse_vjp_plan_device(
        input_shape,
        target_shape,
        mode,
        device=feature_values.device,
        dtype=feature_values.dtype,
    )
    feature_dim = int(feature_values.shape[1])
    if feature_dim != int(input_shape[1]) or feature_dim != int(target_shape[1]):
        raise ValueError("feature dimension mismatch in sparse target-grid forward")
    if int(feature_values.shape[0]) != int(plan.unique_pixel_ids.numel()):
        raise ValueError("feature_values row count must match sparse target-grid pixel ids")
    target_flat = torch.zeros(
        (int(target_shape[0]) * int(target_shape[2]) * int(target_shape[3]), feature_dim),
        device=feature_values.device,
        dtype=feature_values.dtype,
    )
    source_values = feature_values.index_select(0, plan.inverse)
    weighted = source_values * plan.weights.unsqueeze(1)
    target_flat.index_add_(0, plan.target_flat_ids, weighted)
    return target_flat.reshape(int(target_shape[0]), int(target_shape[2]), int(target_shape[3]), feature_dim).permute(
        0, 3, 1, 2
    ).contiguous()


def _batched_sparse_feature_values_to_target_grid(
    feature_values: torch.Tensor,
    *,
    input_shape: tuple[int, int, int, int],
    target_shape: tuple[int, int, int, int],
    mode: str,
) -> torch.Tensor:
    if feature_values.ndim != 3:
        raise ValueError("feature_values must have shape [chunks,sparse_pixels,feature_dim]")
    batch = int(feature_values.shape[0])
    feature_dim = int(feature_values.shape[2])
    plan = _target_grid_sparse_vjp_plan_device(
        input_shape,
        target_shape,
        mode,
        device=feature_values.device,
        dtype=feature_values.dtype,
    )
    source_values = feature_values.index_select(1, plan.inverse)
    weighted = source_values * plan.weights.view(1, -1, 1)
    target_cells = int(target_shape[0]) * int(target_shape[2]) * int(target_shape[3])
    target_flat = torch.zeros((batch, target_cells, feature_dim), device=feature_values.device, dtype=feature_values.dtype)
    scatter_ids = plan.target_flat_ids.view(1, -1, 1).expand(batch, -1, feature_dim)
    target_flat.scatter_add_(1, scatter_ids, weighted)
    return (
        target_flat.reshape(batch * int(target_shape[0]), int(target_shape[2]), int(target_shape[3]), feature_dim)
        .permute(0, 3, 1, 2)
        .contiguous()
    )


def _batched_pack_sparse_target_grid_vjp(
    grad_target_grid: torch.Tensor,
    *,
    input_shape: tuple[int, int, int, int],
    target_shape: tuple[int, int, int, int],
    mode: str,
) -> tuple[SparseImageVjpPack, ...]:
    target_frames = int(target_shape[0])
    if int(grad_target_grid.shape[0]) % target_frames != 0:
        raise ValueError("batched grad_target_grid frame count must be divisible by target_shape[0]")
    batch = int(grad_target_grid.shape[0]) // target_frames
    feature_dim = int(grad_target_grid.shape[1])
    plan = _target_grid_sparse_vjp_plan_device(
        input_shape,
        target_shape,
        mode,
        device=grad_target_grid.device,
        dtype=grad_target_grid.dtype,
    )
    target_values = (
        grad_target_grid.reshape(batch, target_frames, feature_dim, int(target_shape[2]), int(target_shape[3]))
        .permute(0, 1, 3, 4, 2)
        .reshape(batch, -1, feature_dim)
    )
    weighted = target_values.index_select(1, plan.target_flat_ids) * plan.weights.view(1, -1, 1)
    if plan.has_duplicate_pixels:
        grad_values = torch.zeros(
            (batch, int(plan.unique_pixel_ids.numel()), feature_dim),
            device=grad_target_grid.device,
            dtype=grad_target_grid.dtype,
        )
        scatter_ids = plan.inverse.view(1, -1, 1).expand(batch, -1, feature_dim)
        grad_values.scatter_add_(1, scatter_ids, weighted)
        pixel_ids = plan.unique_pixel_ids
    else:
        grad_values = weighted.contiguous()
        pixel_ids = plan.source_pixel_ids
    zero_alpha = torch.zeros((int(pixel_ids.numel()),), device=grad_target_grid.device, dtype=grad_target_grid.dtype)
    return tuple(
        SparseImageVjpPack(
            pixel_ids=pixel_ids.contiguous(),
            grad_feature_values=grad_values[index].contiguous(),
            grad_alpha_values=zero_alpha.contiguous(),
            pixel_count=int(pixel_ids.numel()),
            total_pixels=plan.total_pixels,
        )
        for index in range(batch)
    )


__all__ = [
    "SparseImageVjpPack",
    "_batched_pack_sparse_target_grid_vjp",
    "_batched_sparse_feature_values_to_target_grid",
    "_pack_sparse_target_grid_vjp",
    "_sparse_feature_values_to_target_grid",
    "_sparse_target_grid_pixel_ids",
]
