from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

try:
    from . import _C  # noqa: F401
except Exception:
    _C = None

RAYTRACE_MAX_BACKWARD_EVENTS = 64
RAYTRACE_PER_RAY_START_PIXEL_LIMIT = 65_536


@dataclass(frozen=True)
class FoamRasterConfig:
    near_plane: float = 1.0e-4
    alpha_threshold: float = 0.0
    transmittance_threshold: float = 1.0e-4
    max_alpha: float = 0.999
    eps: float = 1.0e-8
    texel_temperature: float = 10.0
    use_tiled: bool = False
    tiled_builder: str = "auto"


@dataclass(frozen=True)
class FoamAuxOutputs:
    normal_distance: Tensor
    normal: Tensor
    median_depth: Tensor
    contrib: Tensor
    point_error: Tensor
    visible_mask: Tensor
    depth_quantile_depths: Tensor | None = None
    depth_quantile_values: Tensor | None = None


def _normalize_rays(rays: Tensor) -> tuple[Tensor, bool]:
    if rays.ndim == 3:
        if rays.shape[-1] != 6:
            raise ValueError("rays must have shape [H,W,6] or [B,H,W,6]")
        return rays.unsqueeze(0), False
    if rays.ndim == 4 and rays.shape[-1] == 6:
        return rays, True
    raise ValueError("rays must have shape [H,W,6] or [B,H,W,6]")


def _check_float_mps(name: str, tensor: Tensor, ndim: int | None = None) -> None:
    if tensor.device.type != "mps":
        raise ValueError(f"{name} must be on MPS")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must be float32")
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"{name} must be rank {ndim}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_int_mps(name: str, tensor: Tensor, ndim: int | None = None) -> None:
    if tensor.device.type != "mps":
        raise ValueError(f"{name} must be on MPS")
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must be int32")
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"{name} must be rank {ndim}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _make_meta(
    rays_b: Tensor,
    points: Tensor,
    features: Tensor,
    config: FoamRasterConfig,
    *,
    output_dim: int | None = None,
    feature_mode: int = 0,
    sv_dof: int = 0,
    depth_quantile_count: int = 1,
    start_mode: int = 0,
) -> tuple[Tensor, Tensor]:
    batch_size, height, width = rays_b.shape[:3]
    cell_count = points.shape[0]
    feature_dim = features.shape[1]
    if output_dim is None:
        output_dim = feature_dim
    meta_i32 = torch.tensor(
        [
            batch_size,
            height,
            width,
            cell_count,
            feature_dim,
            int(output_dim),
            int(feature_mode),
            int(sv_dof),
            int(depth_quantile_count),
            int(start_mode),
        ],
        device=rays_b.device,
        dtype=torch.int32,
    )
    meta_f32 = torch.tensor(
        [
            float(config.near_plane),
            float(config.alpha_threshold),
            float(config.transmittance_threshold),
            float(config.max_alpha),
            float(config.eps),
            float(config.texel_temperature),
        ],
        device=rays_b.device,
        dtype=torch.float32,
    )
    return meta_i32, meta_f32


def _normalize_depth_quantiles(
    depth_quantiles: Tensor | Sequence[float] | None,
    device: torch.device,
) -> Tensor:
    if depth_quantiles is None:
        out = torch.tensor([0.5], device=device, dtype=torch.float32)
    elif isinstance(depth_quantiles, Tensor):
        out = depth_quantiles.to(device=device, dtype=torch.float32).flatten()
    else:
        out = torch.tensor(list(depth_quantiles), device=device, dtype=torch.float32).flatten()
    if out.numel() < 1:
        raise ValueError("depth_quantiles must contain at least one value")
    if not bool(torch.isfinite(out).all().item()):
        raise ValueError("depth_quantiles must be finite")
    if bool(((out < 0.0) | (out > 1.0)).any().item()):
        raise ValueError("depth_quantiles must be in [0, 1]")
    return out.contiguous()


def _full_screen_bounds(rays_b: Tensor, points: Tensor) -> Tensor:
    batch_size, height, width = rays_b.shape[:3]
    cell_count = points.shape[0]
    bounds = torch.tensor([0, 0, width - 1, height - 1], device=rays_b.device, dtype=torch.int32)
    return bounds.view(1, 1, 4).expand(batch_size, cell_count, 4).contiguous()


def _projected_screen_bounds(rays_b: Tensor, points: Tensor, radii: Tensor, config: FoamRasterConfig) -> Tensor:
    batch_size, height, width = rays_b.shape[:3]
    cell_count = points.shape[0]
    if height < 2 or width < 2:
        return _full_screen_bounds(rays_b, points)

    dirs = rays_b[..., 3:]
    z = dirs[..., 2].clamp_min(float(config.eps))
    x_slopes = dirs[:, 0, :, 0] / z[:, 0, :]
    y_slopes = dirs[:, :, 0, 1] / z[:, :, 0]
    x0_s = x_slopes[:, 0]
    x1_s = x_slopes[:, -1]
    y0_s = y_slopes[:, 0]
    y1_s = y_slopes[:, -1]
    x_span = (x1_s - x0_s).abs().clamp_min(float(config.eps))
    y_span = (y1_s - y0_s).abs().clamp_min(float(config.eps))

    origins = rays_b[:, 0, 0, :3]
    rel = points.unsqueeze(0) - origins.unsqueeze(1)
    depth = rel[..., 2]
    radius = radii.view(1, cell_count)
    safe_depth = depth.clamp_min(float(config.eps))
    sx = rel[..., 0] / safe_depth
    sy = rel[..., 1] / safe_depth
    # Use the near side of the sphere and a small pixel pad. This is deliberately
    # conservative because missing a candidate cell changes the rendered image.
    slope_radius = radius / (depth - radius).clamp_min(float(config.eps))
    x_den = x1_s - x0_s
    y_den = y1_s - y0_s
    x_den = torch.where(x_den.abs() > float(config.eps), x_den, torch.full_like(x_den, float(config.eps)))
    y_den = torch.where(y_den.abs() > float(config.eps), y_den, torch.full_like(y_den, float(config.eps)))
    px = (sx - x0_s[:, None]) / x_den[:, None] * float(width - 1)
    py = (sy - y0_s[:, None]) / y_den[:, None] * float(height - 1)
    pad_x = slope_radius / x_span[:, None] * float(width - 1) + 3.0
    pad_y = slope_radius / y_span[:, None] * float(height - 1) + 3.0

    raw_x0 = torch.floor(px - pad_x)
    raw_y0 = torch.floor(py - pad_y)
    raw_x1 = torch.ceil(px + pad_x)
    raw_y1 = torch.ceil(py + pad_y)
    full = depth <= (radius + float(config.near_plane) + float(config.eps))
    visible = (raw_x1 >= 0.0) & (raw_x0 <= float(width - 1)) & (raw_y1 >= 0.0) & (raw_y0 <= float(height - 1))
    visible = visible | full

    bx0 = raw_x0.clamp(0.0, float(width - 1))
    by0 = raw_y0.clamp(0.0, float(height - 1))
    bx1 = raw_x1.clamp(0.0, float(width - 1))
    by1 = raw_y1.clamp(0.0, float(height - 1))
    bx0 = torch.where(full, torch.zeros_like(bx0), bx0)
    by0 = torch.where(full, torch.zeros_like(by0), by0)
    bx1 = torch.where(full, torch.full_like(bx1, float(width - 1)), bx1)
    by1 = torch.where(full, torch.full_like(by1, float(height - 1)), by1)
    bx0 = torch.where(visible, bx0, torch.ones_like(bx0))
    by0 = torch.where(visible, by0, torch.ones_like(by0))
    bx1 = torch.where(visible, bx1, torch.zeros_like(bx1))
    by1 = torch.where(visible, by1, torch.zeros_like(by1))
    return torch.stack([bx0, by0, bx1, by1], dim=-1).to(torch.int32).contiguous()


def _default_sorted_ids(points: Tensor, radii: Tensor, rays_b: Tensor) -> Tensor:
    origins = rays_b[:, 0, 0, :3]
    power = (points.unsqueeze(0) - origins.unsqueeze(1)).square().sum(dim=-1) - radii.unsqueeze(0).square()
    return torch.argsort(power.detach(), dim=1, stable=True).to(torch.int32).contiguous()


def _sampled_ray_support_counts(
    points: Tensor,
    radii: Tensor,
    rays_b: Tensor,
    config: FoamRasterConfig,
    *,
    samples_per_axis: int = 9,
) -> Tensor:
    batch_size, height, width = rays_b.shape[:3]
    sample_y = min(max(int(samples_per_axis), 1), int(height))
    sample_x = min(max(int(samples_per_axis), 1), int(width))
    ys = torch.linspace(0, height - 1, sample_y, device=rays_b.device, dtype=torch.float32)
    xs = torch.linspace(0, width - 1, sample_x, device=rays_b.device, dtype=torch.float32)
    ys = ys.round().to(torch.long)
    xs = xs.round().to(torch.long)
    rays_s = rays_b.index_select(1, ys).index_select(2, xs).reshape(batch_size, -1, 6)
    origins = rays_s[..., :3]
    dirs = F.normalize(rays_s[..., 3:], dim=-1, eps=float(config.eps))
    rel = points.view(1, 1, -1, 3) - origins.unsqueeze(2)
    t = (rel * dirs.unsqueeze(2)).sum(dim=-1).clamp_min(float(config.near_plane))
    closest = rel - t.unsqueeze(-1) * dirs.unsqueeze(2)
    power = closest.square().sum(dim=-1) - radii.view(1, 1, -1).square()
    min_power, nearest_ids = power.detach().min(dim=-1)
    hit = min_power <= 0.0
    counts = torch.zeros((batch_size, points.shape[0]), device=rays_b.device, dtype=torch.float32)
    counts.scatter_add_(1, nearest_ids, hit.to(dtype=torch.float32))
    return counts


def _default_start_ids(points: Tensor, radii: Tensor, rays_b: Tensor, config: FoamRasterConfig) -> Tensor:
    if int(rays_b.shape[1]) * int(rays_b.shape[2]) <= RAYTRACE_PER_RAY_START_PIXEL_LIMIT:
        return _default_per_ray_start_ids(points, radii, rays_b, config)
    origins = rays_b[:, 0, 0, :3]
    power = (points.unsqueeze(0) - origins.unsqueeze(1)).square().sum(dim=-1) - radii.unsqueeze(0).square()
    origin_ids = torch.argmin(power.detach(), dim=1)
    counts = _sampled_ray_support_counts(points, radii, rays_b, config)
    support_ids = counts.argmax(dim=1)
    origin_support = counts.gather(1, origin_ids.view(-1, 1)).squeeze(1)
    support_count = counts.gather(1, support_ids.view(-1, 1)).squeeze(1)
    start_ids = torch.where(support_count > origin_support, support_ids, origin_ids)
    return start_ids.to(torch.int32).contiguous()


def _default_per_ray_start_ids(
    points: Tensor,
    radii: Tensor,
    rays_b: Tensor,
    config: FoamRasterConfig,
    *,
    chunk_size: int = 4096,
) -> Tensor:
    flat = rays_b.reshape(-1, 6)
    out = torch.empty((flat.shape[0],), device=rays_b.device, dtype=torch.int32)
    radius2 = radii.square().view(1, -1)
    for start in range(0, int(flat.shape[0]), int(chunk_size)):
        rays_c = flat[start : start + int(chunk_size)]
        origins = rays_c[:, :3]
        dirs = F.normalize(rays_c[:, 3:], dim=-1, eps=float(config.eps))
        if float(config.near_plane) > 0.0:
            query = origins + dirs * float(config.near_plane)
            power = (points.view(1, -1, 3) - query[:, None, :]).square().sum(dim=-1) - radius2
        else:
            rel = points.view(1, -1, 3) - origins[:, None, :]
            t = (rel * dirs[:, None, :]).sum(dim=-1).clamp_min(float(config.near_plane))
            closest = rel - t[..., None] * dirs[:, None, :]
            power = closest.square().sum(dim=-1) - radius2
        out[start : start + int(rays_c.shape[0])] = power.detach().argmin(dim=1).to(torch.int32)
    return out.contiguous()


def _raytrace_start_mode(start_ids: Tensor, rays_b: Tensor) -> int:
    batch_size, height, width = rays_b.shape[:3]
    if start_ids.ndim == 1 and int(start_ids.shape[0]) == int(batch_size):
        return 0
    if start_ids.numel() == int(batch_size) * int(height) * int(width):
        return 1
    raise ValueError("start_ids must have shape [B], [B*H*W], or [B,H,W]")


def _make_adjacency_diff(points: Tensor, radii: Tensor, adjacency: Tensor, offsets: Tensor) -> Tensor:
    """Pack official-style power-face diffs for the Metal tiled kernels."""
    with torch.no_grad():
        if adjacency.numel() == 0:
            return torch.empty((0, 4), device=points.device, dtype=torch.float32)
        counts = (offsets[1:].detach().cpu().to(torch.long) - offsets[:-1].detach().cpu().to(torch.long)).clamp_min(0)
        owners_cpu = torch.repeat_interleave(torch.arange(points.shape[0], dtype=torch.long), counts)
        if owners_cpu.numel() != adjacency.numel():
            raise ValueError("offsets/adjacency edge count mismatch")
        owners = owners_cpu.to(device=points.device)
        adj = adjacency.detach().to(device=points.device, dtype=torch.long)
        points_d = points.detach()
        radii_d = radii.detach()
        owner_points = points_d[owners]
        adj_points = points_d[adj]
        owner_radii = radii_d[owners]
        adj_radii = radii_d[adj]
        diff = adj_points - owner_points
        pm_diff = 0.5 * (
            adj_points.square().sum(dim=-1)
            - owner_points.square().sum(dim=-1)
            + owner_radii.square()
            - adj_radii.square()
        )
        return torch.cat([diff, pm_diff[:, None]], dim=-1).to(dtype=torch.float32).contiguous()


def _check_inputs(
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    features: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    rays_b: Tensor,
    sorted_ids: Tensor,
    output_dim: int,
    feature_mode: int,
    sv_dof: int = 0,
) -> None:
    _check_float_mps("points", points, 2)
    _check_float_mps("radii", radii, 1)
    _check_float_mps("densities", densities, 1)
    _check_float_mps("features", features, 2)
    _check_float_mps("rays", rays_b, 4)
    _check_int_mps("adjacency", adjacency, 1)
    _check_int_mps("offsets", offsets, 1)
    _check_int_mps("sorted_ids", sorted_ids, 2)

    cell_count = points.shape[0]
    if points.shape[1] != 3:
        raise ValueError("points must have shape [N,3]")
    if radii.shape[0] != cell_count or densities.shape[0] != cell_count or features.shape[0] != cell_count:
        raise ValueError("points/radii/densities/features must agree on N")
    if features.shape[1] <= 0:
        raise ValueError("features must have a positive feature dimension")
    if output_dim <= 0:
        raise ValueError("output_dim must be positive")
    if feature_mode == 0:
        if features.shape[1] != output_dim:
            raise ValueError("constant feature mode requires features.shape[1] == output_dim")
    elif feature_mode in {1, 2}:
        if features.shape[1] != output_dim * 4:
            raise ValueError("linear feature mode requires flattened features.shape[1] == output_dim * 4")
    elif feature_mode == 3:
        if features.shape[1] != output_dim * 4 + 3:
            raise ValueError("oriented surface-linear feature mode requires flattened features.shape[1] == output_dim * 4 + 3")
    elif feature_mode == 4:
        stride = output_dim + 2
        if features.shape[1] <= 9 or (features.shape[1] - 9) % stride != 0:
            raise ValueError("oriented texel-surface feature mode requires features.shape[1] == S * (output_dim + 2) + 9")
    elif feature_mode == 5:
        stride = output_dim + 3
        if features.shape[1] <= 9 or (features.shape[1] - 9) % stride != 0:
            raise ValueError(
                "oriented height texel-surface feature mode requires features.shape[1] == S * (output_dim + 3) + 9"
            )
    elif feature_mode == 6:
        if output_dim != 3:
            raise ValueError("height SV texel-surface feature mode requires output_dim == 3")
        if sv_dof <= 0:
            raise ValueError("height SV texel-surface feature mode requires sv_dof > 0")
        stride = 3 + 6 * sv_dof
        if features.shape[1] <= 9 or (features.shape[1] - 9) % stride != 0:
            raise ValueError(
                "height SV texel-surface feature mode requires features.shape[1] == S * (3 + 6 * sv_dof) + 9"
            )
    else:
        raise ValueError("feature_mode must be 0, 1, 2, 3, 4, 5, or 6")
    if offsets.shape[0] != cell_count + 1:
        raise ValueError("offsets must have shape [N+1]")
    if rays_b.shape[-1] != 6:
        raise ValueError("rays must have trailing dimension 6")
    if sorted_ids.shape != (rays_b.shape[0], cell_count):
        raise ValueError("sorted_ids must have shape [B,N]")


class _RasterizePowerFoamFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        points: Tensor,
        radii: Tensor,
        densities: Tensor,
        features: Tensor,
        adjacency: Tensor,
        offsets: Tensor,
        sorted_ids: Tensor,
        screen_bounds: Tensor,
        rays_b: Tensor,
        meta_i32: Tensor,
        meta_f32: Tensor,
    ) -> tuple[Tensor, Tensor]:
        ctx.output_dim = int(meta_i32.detach().cpu()[5])
        out, alpha, log_t, pixel_stop = torch.ops.powerfoam_metal.rasterize_train_forward(
            points,
            radii,
            densities,
            features,
            adjacency,
            offsets,
            sorted_ids,
            screen_bounds,
            rays_b,
            meta_i32,
            meta_f32,
        )
        ctx.save_for_backward(
            points,
            radii,
            densities,
            features,
            adjacency,
            offsets,
            sorted_ids,
            screen_bounds,
            rays_b,
            log_t,
            pixel_stop,
            meta_i32,
            meta_f32,
        )
        return out, alpha

    @staticmethod
    def backward(ctx, grad_out: Tensor | None, grad_alpha: Tensor | None):
        (
            points,
            radii,
            densities,
            features,
            adjacency,
            offsets,
            sorted_ids,
            screen_bounds,
            rays_b,
            log_t,
            pixel_stop,
            meta_i32,
            meta_f32,
        ) = ctx.saved_tensors
        if grad_out is None:
            grad_out = torch.zeros(
                (*rays_b.shape[:3], ctx.output_dim),
                device=points.device,
                dtype=torch.float32,
            )
        if grad_alpha is None:
            grad_alpha = torch.zeros(rays_b.shape[:3], device=points.device, dtype=torch.float32)
        grad_points, grad_radii, grad_densities, grad_features = torch.ops.powerfoam_metal.rasterize_train_backward(
            points,
            radii,
            densities,
            features,
            adjacency,
            offsets,
            sorted_ids,
            screen_bounds,
            rays_b,
            log_t,
            pixel_stop,
            grad_out.contiguous(),
            grad_alpha.contiguous(),
            meta_i32,
            meta_f32,
        )
        return (
            grad_points,
            grad_radii,
            grad_densities,
            grad_features,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _RasterizePowerFoamTiledFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        builder: str,
        points: Tensor,
        radii: Tensor,
        densities: Tensor,
        features: Tensor,
        adjacency: Tensor,
        offsets: Tensor,
        adjacency_diff: Tensor,
        sorted_ids: Tensor,
        screen_bounds: Tensor,
        rays_b: Tensor,
        meta_i32: Tensor,
        meta_f32: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        ctx.output_dim = int(meta_i32.detach().cpu()[5])
        tile_offsets, tile_cell_ids = _build_tiled_candidates(builder, points, screen_bounds, sorted_ids, meta_i32, meta_f32)
        out, alpha, normal_distance, log_t, tile_stop = torch.ops.powerfoam_metal.rasterize_tiled_train_forward(
            points,
            radii,
            densities,
            features,
            adjacency,
            offsets,
            adjacency_diff,
            sorted_ids,
            screen_bounds,
            tile_offsets,
            tile_cell_ids,
            rays_b,
            meta_i32,
            meta_f32,
        )
        ctx.save_for_backward(
            points,
            radii,
            densities,
            features,
            adjacency,
            offsets,
            adjacency_diff,
            sorted_ids,
            screen_bounds,
            tile_offsets,
            tile_cell_ids,
            tile_stop,
            rays_b,
            log_t,
            meta_i32,
            meta_f32,
        )
        return out, alpha, normal_distance

    @staticmethod
    def backward(ctx, grad_out: Tensor | None, grad_alpha: Tensor | None, grad_normal_distance: Tensor | None):
        (
            points,
            radii,
            densities,
            features,
            adjacency,
            offsets,
            adjacency_diff,
            sorted_ids,
            screen_bounds,
            tile_offsets,
            tile_cell_ids,
            tile_stop,
            rays_b,
            log_t,
            meta_i32,
            meta_f32,
        ) = ctx.saved_tensors
        if grad_out is None:
            grad_out = torch.zeros(
                (*rays_b.shape[:3], ctx.output_dim),
                device=points.device,
                dtype=torch.float32,
            )
        if grad_alpha is None:
            grad_alpha = torch.zeros(rays_b.shape[:3], device=points.device, dtype=torch.float32)
        if grad_normal_distance is None:
            grad_normal_distance_arg = torch.empty((0,), device=points.device, dtype=torch.float32)
        else:
            grad_normal_distance_arg = grad_normal_distance.contiguous()
        grad_points, grad_radii, grad_densities, grad_features = torch.ops.powerfoam_metal.rasterize_tiled_train_backward(
            points,
            radii,
            densities,
            features,
            adjacency,
            offsets,
            adjacency_diff,
            sorted_ids,
            screen_bounds,
            tile_offsets,
            tile_cell_ids,
            tile_stop,
            rays_b,
            log_t,
            grad_out.contiguous(),
            grad_alpha.contiguous(),
            grad_normal_distance_arg,
            meta_i32,
            meta_f32,
        )
        return (
            None,
            grad_points,
            grad_radii,
            grad_densities,
            grad_features,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _build_tiled_candidates(
    builder: str,
    points: Tensor,
    screen_bounds: Tensor,
    sorted_ids: Tensor,
    meta_i32: Tensor,
    meta_f32: Tensor,
) -> tuple[Tensor, Tensor]:
    if builder == "sorted_scan":
        tile_counts = torch.ops.powerfoam_metal.rasterize_tiled_count(
            screen_bounds,
            sorted_ids,
            meta_i32,
            meta_f32,
        )
        tile_offsets_cpu = torch.cumsum(tile_counts.detach().cpu(), dim=0, dtype=torch.int32)
        tile_offsets = tile_offsets_cpu.to(device=points.device, dtype=torch.int32).contiguous()
        tile_cell_ids = torch.ops.powerfoam_metal.rasterize_tiled_write(
            screen_bounds,
            sorted_ids,
            tile_offsets,
            meta_i32,
            meta_f32,
        )
    elif builder == "emit_sort":
        tile_counts = torch.ops.powerfoam_metal.rasterize_tiled_emit_count(
            screen_bounds,
            sorted_ids,
            meta_i32,
            meta_f32,
        )
        tile_offsets_cpu = torch.cumsum(tile_counts.detach().cpu(), dim=0, dtype=torch.int32)
        tile_offsets = tile_offsets_cpu.to(device=points.device, dtype=torch.int32).contiguous()
        sort_keys, unsorted_tile_cell_ids = torch.ops.powerfoam_metal.rasterize_tiled_emit_write(
            screen_bounds,
            sorted_ids,
            tile_offsets,
            meta_i32,
            meta_f32,
        )
        sort_order = torch.argsort(sort_keys, stable=True)
        tile_cell_ids = unsorted_tile_cell_ids[sort_order].contiguous()
    else:
        raise ValueError("tiled_builder must be 'auto', 'sorted_scan', or 'emit_sort'")
    return tile_offsets, tile_cell_ids


def _rasterize_apply(
    config: FoamRasterConfig,
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    features: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    sorted_ids: Tensor,
    screen_bounds: Tensor,
    rays_b: Tensor,
    meta_i32: Tensor,
    meta_f32: Tensor,
) -> tuple[Tensor, Tensor]:
    if not bool(config.use_tiled):
        return _RasterizePowerFoamFunction.apply(
            points,
            radii,
            densities,
            features,
            adjacency,
            offsets,
            sorted_ids,
            screen_bounds,
            rays_b,
            meta_i32,
            meta_f32,
        )
    builder = str(config.tiled_builder)
    if builder == "auto":
        builder = "sorted_scan" if int(points.shape[0]) <= 4096 else "emit_sort"
    adjacency_diff = _make_adjacency_diff(points, radii, adjacency, offsets)
    out, alpha, _normal_distance = _RasterizePowerFoamTiledFunction.apply(
        builder,
        points,
        radii,
        densities,
        features,
        adjacency,
        offsets,
        adjacency_diff,
        sorted_ids,
        screen_bounds,
        rays_b,
        meta_i32,
        meta_f32,
    )
    return out, alpha


def _rasterize_apply_with_normal_distance(
    config: FoamRasterConfig,
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    features: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    sorted_ids: Tensor,
    screen_bounds: Tensor,
    rays_b: Tensor,
    meta_i32: Tensor,
    meta_f32: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if not bool(config.use_tiled):
        raise ValueError("normal_distance autograd output is currently implemented for the tiled Metal path")
    builder = str(config.tiled_builder)
    if builder == "auto":
        builder = "sorted_scan" if int(points.shape[0]) <= 4096 else "emit_sort"
    adjacency_diff = _make_adjacency_diff(points, radii, adjacency, offsets)
    return _RasterizePowerFoamTiledFunction.apply(
        builder,
        points,
        radii,
        densities,
        features,
        adjacency,
        offsets,
        adjacency_diff,
        sorted_ids,
        screen_bounds,
        rays_b,
        meta_i32,
        meta_f32,
    )


def _normalize_target_features(target: Tensor | None, rays_b: Tensor, output_dim: int) -> Tensor:
    if target is None:
        return torch.zeros((*rays_b.shape[:3], output_dim), device=rays_b.device, dtype=torch.float32)
    target = target.to(device=rays_b.device, dtype=torch.float32)
    if target.ndim == 3 and target.shape[-1] == output_dim:
        target = target.unsqueeze(0)
    elif target.ndim == 3 and target.shape[0] == output_dim:
        target = target.permute(1, 2, 0).unsqueeze(0)
    elif target.ndim == 4 and target.shape[-1] == output_dim:
        pass
    elif target.ndim == 4 and target.shape[1] == output_dim:
        target = target.permute(0, 2, 3, 1)
    else:
        raise ValueError("target must be [H,W,C], [C,H,W], [B,H,W,C], or [B,C,H,W]")
    if target.shape != (*rays_b.shape[:3], output_dim):
        raise ValueError("target shape must match rays and output_dim")
    return target.contiguous()


def rasterize_power_foam_aux(
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    features: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    rays: Tensor,
    target: Tensor | None = None,
    config: FoamRasterConfig | None = None,
    sorted_ids: Tensor | None = None,
    *,
    output_dim: int | None = None,
    feature_mode: int = 0,
    sv_dof: int = 0,
    depth_quantiles: Tensor | Sequence[float] | None = None,
) -> FoamAuxOutputs:
    """Compute non-gradient auxiliary outputs with the tiled Metal path.

    This low-level entry point expects the same flattened `features` layout used
    internally by the rasterizer wrappers. `target` is used only for per-cell
    point-error accumulation and may be omitted to compare against zeros.
    """
    if not hasattr(torch.ops, "powerfoam_metal"):
        raise RuntimeError("powerfoam_metal custom ops not found. Build the extension first.")
    if config is None:
        config = FoamRasterConfig(use_tiled=True)
    rays_b, keep_batch = _normalize_rays(rays.contiguous())
    points = points.contiguous()
    radii = radii.contiguous()
    densities = densities.contiguous()
    features = features.contiguous()
    adjacency = adjacency.contiguous()
    offsets = offsets.contiguous()
    if output_dim is None:
        output_dim = int(features.shape[1])
    if sorted_ids is None:
        sorted_ids = _default_sorted_ids(points, radii, rays_b)
    else:
        sorted_ids = sorted_ids.contiguous()
    _check_inputs(
        points,
        radii,
        densities,
        features,
        adjacency,
        offsets,
        rays_b,
        sorted_ids,
        output_dim=int(output_dim),
        feature_mode=int(feature_mode),
        sv_dof=int(sv_dof),
    )
    target_b = _normalize_target_features(target, rays_b, int(output_dim))
    depth_quantiles_t = _normalize_depth_quantiles(depth_quantiles, rays_b.device)
    meta_i32, meta_f32 = _make_meta(
        rays_b,
        points,
        features,
        config,
        output_dim=int(output_dim),
        feature_mode=int(feature_mode),
        sv_dof=int(sv_dof),
        depth_quantile_count=int(depth_quantiles_t.numel()),
    )
    screen_bounds = _projected_screen_bounds(rays_b, points, radii, config)
    builder = str(config.tiled_builder)
    if builder == "auto":
        builder = "sorted_scan" if int(points.shape[0]) <= 4096 else "emit_sort"
    adjacency_diff = _make_adjacency_diff(points, radii, adjacency, offsets)
    tile_offsets, tile_cell_ids = _build_tiled_candidates(builder, points, screen_bounds, sorted_ids, meta_i32, meta_f32)
    with torch.no_grad():
        (
            normal_distance,
            normal_hwc,
            depth_quantile_depths,
            contrib,
            point_error,
            visible_i32,
        ) = torch.ops.powerfoam_metal.rasterize_tiled_aux_forward(
            points,
            radii,
            densities,
            features,
            adjacency,
            offsets,
            adjacency_diff,
            sorted_ids,
            screen_bounds,
            tile_offsets,
            tile_cell_ids,
            rays_b,
            target_b,
            depth_quantiles_t,
            meta_i32,
            meta_f32,
        )
    normal = normal_hwc.permute(0, 3, 1, 2).contiguous()
    visible_mask = visible_i32.to(torch.bool)
    median_idx = int(torch.argmin((depth_quantiles_t - 0.5).abs()).item())
    median_depth = depth_quantile_depths[..., median_idx].contiguous()
    if keep_batch:
        return FoamAuxOutputs(
            normal_distance,
            normal,
            median_depth,
            contrib,
            point_error,
            visible_mask,
            depth_quantile_depths,
            depth_quantiles_t,
        )
    return FoamAuxOutputs(
        normal_distance[0],
        normal[0],
        median_depth[0],
        contrib[0],
        point_error[0],
        visible_mask[0],
        depth_quantile_depths[0],
        depth_quantiles_t,
    )


def raytrace_power_foam_flat(
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    features: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    rays: Tensor,
    config: FoamRasterConfig | None = None,
    start_ids: Tensor | None = None,
    *,
    output_dim: int | None = None,
    feature_mode: int = 0,
    sv_dof: int = 0,
    return_steps: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    """Forward-only PowerFoam ray-walk probe.

    This is intentionally not an autograd path yet. It exercises the
    official-style start-cell + adjacency walk on Metal so traversal speed and
    parity can be measured before designing replay backward state.
    """
    if not hasattr(torch.ops, "powerfoam_metal"):
        raise RuntimeError("powerfoam_metal custom ops not found. Build the extension first.")
    if config is None:
        config = FoamRasterConfig()
    rays_b, keep_batch = _normalize_rays(rays.contiguous())
    points = points.contiguous()
    radii = radii.contiguous()
    densities = densities.contiguous()
    features = features.contiguous()
    adjacency = adjacency.contiguous()
    offsets = offsets.contiguous()
    if output_dim is None:
        output_dim = int(features.shape[1])
    sorted_ids = _default_sorted_ids(points, radii, rays_b)
    _check_inputs(
        points,
        radii,
        densities,
        features,
        adjacency,
        offsets,
        rays_b,
        sorted_ids,
        output_dim=int(output_dim),
        feature_mode=int(feature_mode),
        sv_dof=int(sv_dof),
    )
    if start_ids is None:
        start_ids = _default_start_ids(points, radii, rays_b, config)
    else:
        start_ids = start_ids.to(device=points.device, dtype=torch.int32).contiguous()
    start_mode = _raytrace_start_mode(start_ids, rays_b)
    adjacency_diff = _make_adjacency_diff(points, radii, adjacency, offsets)
    meta_i32, meta_f32 = _make_meta(
        rays_b,
        points,
        features,
        config,
        output_dim=int(output_dim),
        feature_mode=int(feature_mode),
        sv_dof=int(sv_dof),
        start_mode=start_mode,
    )
    with torch.no_grad():
        out, alpha, _normal_distance, _normal, steps = torch.ops.powerfoam_metal.raytrace_forward(
            points,
            radii,
            densities,
            features,
            adjacency,
            offsets,
            adjacency_diff,
            start_ids,
            rays_b,
            meta_i32,
            meta_f32,
        )
    if not keep_batch:
        out = out[0]
        alpha = alpha[0]
        steps = steps[0]
    if return_steps:
        return out, alpha, steps
    return out, alpha


def raytrace_power_foam(
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    features: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    rays: Tensor,
    config: FoamRasterConfig | None = None,
    start_ids: Tensor | None = None,
    *,
    return_steps: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    return raytrace_power_foam_flat(
        points,
        radii,
        densities,
        features,
        adjacency,
        offsets,
        rays,
        config,
        start_ids,
        output_dim=int(features.shape[1]),
        feature_mode=0,
        return_steps=return_steps,
    )


class _RaytracePowerFoamHeightSVFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        points: Tensor,
        radii: Tensor,
        densities: Tensor,
        features: Tensor,
        adjacency: Tensor,
        offsets: Tensor,
        adjacency_diff: Tensor,
        start_ids: Tensor,
        rays: Tensor,
        meta_i32: Tensor,
        meta_f32: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        out, alpha, normal_distance, normal, steps = torch.ops.powerfoam_metal.raytrace_forward(
            points,
            radii,
            densities,
            features,
            adjacency,
            offsets,
            adjacency_diff,
            start_ids,
            rays,
            meta_i32,
            meta_f32,
        )
        if any(ctx.needs_input_grad[:4]):
            max_steps = int(steps.max().detach().cpu().item())
            if max_steps > RAYTRACE_MAX_BACKWARD_EVENTS:
                raise RuntimeError(
                    "raytrace height+SV backward replay cap exceeded: "
                    f"max_steps={max_steps}, cap={RAYTRACE_MAX_BACKWARD_EVENTS}. "
                    "Increase FOAM_RAYTRACE_MAX_EVENTS / RAYTRACE_MAX_BACKWARD_EVENTS or use tiled rendering."
                )
        ctx.save_for_backward(
            points,
            radii,
            densities,
            features,
            adjacency,
            offsets,
            adjacency_diff,
            start_ids,
            rays,
            meta_i32,
            meta_f32,
        )
        return out, alpha, normal_distance, normal, steps

    @staticmethod
    def backward(
        ctx,
        grad_out: Tensor | None,
        grad_alpha: Tensor | None,
        grad_normal_distance: Tensor | None,
        grad_normal: Tensor | None,
        _grad_steps: Tensor | None,
    ):
        (
            points,
            radii,
            densities,
            features,
            adjacency,
            offsets,
            adjacency_diff,
            start_ids,
            rays,
            meta_i32,
            meta_f32,
        ) = ctx.saved_tensors
        if grad_out is None:
            grad_out_arg = torch.zeros((*rays.shape[:3], 3), device=points.device, dtype=torch.float32)
        else:
            grad_out_arg = grad_out.contiguous()
        if grad_alpha is None:
            grad_alpha_arg = torch.zeros(rays.shape[:3], device=points.device, dtype=torch.float32)
        else:
            grad_alpha_arg = grad_alpha.contiguous()
        if grad_normal_distance is None:
            grad_normal_distance_arg = torch.empty((0,), device=points.device, dtype=torch.float32)
        else:
            grad_normal_distance_arg = grad_normal_distance.contiguous()
        if grad_normal is None:
            grad_normal_arg = torch.empty((0,), device=points.device, dtype=torch.float32)
        else:
            grad_normal_arg = grad_normal.contiguous()
        grad_points, grad_radii, grad_densities, grad_features = torch.ops.powerfoam_metal.raytrace_height_sv_backward(
            points,
            radii,
            densities,
            features,
            adjacency,
            offsets,
            adjacency_diff,
            start_ids,
            rays,
            grad_out_arg,
            grad_alpha_arg,
            grad_normal_distance_arg,
            grad_normal_arg,
            meta_i32,
            meta_f32,
        )
        return (
            grad_points,
            grad_radii,
            grad_densities,
            grad_features,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _raytrace_height_sv_apply(
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    features: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    rays: Tensor,
    config: FoamRasterConfig | None,
    start_ids: Tensor | None,
    *,
    sv_dof: int,
    return_normal_distance: bool,
    return_normal: bool,
    return_steps: bool,
) -> tuple[Tensor, ...]:
    if not hasattr(torch.ops, "powerfoam_metal"):
        raise RuntimeError("powerfoam_metal custom ops not found. Build the extension first.")
    if config is None:
        config = FoamRasterConfig()
    rays_b, keep_batch = _normalize_rays(rays.contiguous())
    points = points.contiguous()
    radii = radii.contiguous()
    densities = densities.contiguous()
    features = features.contiguous()
    adjacency = adjacency.contiguous()
    offsets = offsets.contiguous()
    sorted_ids = _default_sorted_ids(points, radii, rays_b)
    _check_inputs(
        points,
        radii,
        densities,
        features,
        adjacency,
        offsets,
        rays_b,
        sorted_ids,
        output_dim=3,
        feature_mode=6,
        sv_dof=int(sv_dof),
    )
    if start_ids is None:
        start_ids = _default_start_ids(points, radii, rays_b, config)
    else:
        start_ids = start_ids.to(device=points.device, dtype=torch.int32).contiguous()
    start_mode = _raytrace_start_mode(start_ids, rays_b)
    adjacency_diff = _make_adjacency_diff(points, radii, adjacency, offsets)
    meta_i32, meta_f32 = _make_meta(
        rays_b,
        points,
        features,
        config,
        output_dim=3,
        feature_mode=6,
        sv_dof=int(sv_dof),
        start_mode=start_mode,
    )
    out, alpha, normal_distance, normal, steps = _RaytracePowerFoamHeightSVFunction.apply(
        points,
        radii,
        densities,
        features,
        adjacency,
        offsets,
        adjacency_diff,
        start_ids,
        rays_b,
        meta_i32,
        meta_f32,
    )
    if not keep_batch:
        out = out[0]
        alpha = alpha[0]
        normal_distance = normal_distance[0]
        normal = normal[0]
        steps = steps[0]
    result = (out, alpha)
    if return_normal_distance:
        result = (*result, normal_distance)
    if return_normal:
        result = (*result, normal)
    if return_steps:
        result = (*result, steps)
    return result


def raytrace_power_foam_oriented_height_sv_texel_surface(
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    texel_sites: Tensor,
    texel_heights: Tensor,
    texel_sv_axis: Tensor,
    texel_sv_rgb: Tensor,
    normals: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    rays: Tensor,
    config: FoamRasterConfig | None = None,
    start_ids: Tensor | None = None,
    *,
    tangents: Tensor | None = None,
    bitangents: Tensor | None = None,
    return_normal_distance: bool = False,
    return_normal: bool = False,
    return_steps: bool = False,
) -> tuple[Tensor, ...]:
    """Trainable ray-walk path for the full height+SV primitive."""
    if texel_sites.ndim != 3 or texel_sites.shape[2] != 2:
        raise ValueError("texel_sites must have shape [N,S,2]")
    if texel_heights.ndim != 2 or texel_heights.shape != texel_sites.shape[:2]:
        raise ValueError("texel_heights must have shape [N,S]")
    if texel_sv_axis.ndim != 4 or texel_sv_axis.shape[-1] != 3:
        raise ValueError("texel_sv_axis must have shape [N,S,D,3]")
    if texel_sv_rgb.ndim != 4 or texel_sv_rgb.shape != texel_sv_axis.shape:
        raise ValueError("texel_sv_rgb must have shape [N,S,D,3] matching texel_sv_axis")
    if texel_sv_axis.shape[:2] != texel_sites.shape[:2]:
        raise ValueError("texel_sites and SV tensors must agree on [N,S]")
    if normals.ndim != 2 or normals.shape != (texel_sites.shape[0], 3):
        raise ValueError("normals must have shape [N,3]")
    if (tangents is None) != (bitangents is None):
        raise ValueError("tangents and bitangents must be provided together")
    if tangents is None or bitangents is None:
        tangents, bitangents = _frame_from_normals(normals)
    if tangents.ndim != 2 or tangents.shape != (texel_sites.shape[0], 3):
        raise ValueError("tangents must have shape [N,3]")
    if bitangents.ndim != 2 or bitangents.shape != (texel_sites.shape[0], 3):
        raise ValueError("bitangents must have shape [N,3]")

    sv_dof = int(texel_sv_axis.shape[2])
    texel_flat = torch.cat(
        [
            texel_sites.contiguous(),
            texel_heights.contiguous().unsqueeze(-1),
            texel_sv_axis.contiguous().reshape(texel_sites.shape[0], texel_sites.shape[1], sv_dof * 3),
            texel_sv_rgb.contiguous().reshape(texel_sites.shape[0], texel_sites.shape[1], sv_dof * 3),
        ],
        dim=-1,
    )
    features_flat = torch.cat(
        [
            texel_flat.reshape(texel_sites.shape[0], -1),
            normals.contiguous(),
            tangents.contiguous(),
            bitangents.contiguous(),
        ],
        dim=1,
    ).contiguous()
    return _raytrace_height_sv_apply(
        points,
        radii,
        densities,
        features_flat,
        adjacency,
        offsets,
        rays,
        config,
        start_ids,
        sv_dof=sv_dof,
        return_normal_distance=return_normal_distance,
        return_normal=return_normal,
        return_steps=return_steps,
    )


def rasterize_power_foam(
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    features: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    rays: Tensor,
    config: FoamRasterConfig | None = None,
    sorted_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Rasterize bounded power cells along rays.

    Each cell is clipped by its bounding sphere and by radical planes against
    the supplied neighbor graph, then alpha-composited with constant per-cell
    features. The Metal path has a custom replay backward for points, radii,
    densities, and features.
    """
    if not hasattr(torch.ops, "powerfoam_metal"):
        raise RuntimeError("powerfoam_metal custom ops not found. Build the extension first.")
    if config is None:
        config = FoamRasterConfig()

    rays_b, keep_batch = _normalize_rays(rays.contiguous())
    points = points.contiguous()
    radii = radii.contiguous()
    densities = densities.contiguous()
    features = features.contiguous()
    adjacency = adjacency.contiguous()
    offsets = offsets.contiguous()
    if sorted_ids is None:
        sorted_ids = _default_sorted_ids(points, radii, rays_b)
    else:
        sorted_ids = sorted_ids.contiguous()

    _check_inputs(
        points,
        radii,
        densities,
        features,
        adjacency,
        offsets,
        rays_b,
        sorted_ids,
        output_dim=features.shape[1],
        feature_mode=0,
    )
    meta_i32, meta_f32 = _make_meta(rays_b, points, features, config)
    screen_bounds = _projected_screen_bounds(rays_b, points, radii, config)
    out, alpha = _rasterize_apply(
        config,
        points,
        radii,
        densities,
        features,
        adjacency,
        offsets,
        sorted_ids,
        screen_bounds,
        rays_b,
        meta_i32,
        meta_f32,
    )
    if keep_batch:
        return out, alpha
    return out[0], alpha[0]


def _frame_from_normals(normals: Tensor) -> tuple[Tensor, Tensor]:
    normals = F.normalize(normals, dim=-1, eps=1.0e-6)
    z_axis = normals.new_tensor([0.0, 0.0, 1.0]).expand_as(normals)
    y_axis = normals.new_tensor([0.0, 1.0, 0.0]).expand_as(normals)
    helper = torch.where(normals[..., 2:3].abs() < 0.9, z_axis, y_axis)
    tangents = F.normalize(torch.cross(helper, normals, dim=-1), dim=-1, eps=1.0e-6)
    bitangents = F.normalize(torch.cross(normals, tangents, dim=-1), dim=-1, eps=1.0e-6)
    return tangents, bitangents


def quaternion_frames(quaternions: Tensor, eps: float = 1.0e-6) -> tuple[Tensor, Tensor, Tensor]:
    """Derive the official PowerFoam normal/tangent frame from quaternions."""
    if quaternions.ndim < 1 or quaternions.shape[-1] != 4:
        raise ValueError("quaternions must have shape [...,4]")
    q = quaternions / quaternions.norm(dim=-1, keepdim=True).clamp_min(float(eps))
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]
    normals = torch.stack(
        [
            1.0 - 2.0 * (y.square() + z.square()),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        dim=-1,
    )
    tangents = torch.stack(
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x.square() + z.square()),
            2.0 * (y * z - x * w),
        ],
        dim=-1,
    )
    bitangents = torch.stack(
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x.square() + y.square()),
        ],
        dim=-1,
    )
    return (
        F.normalize(normals, dim=-1, eps=float(eps)),
        F.normalize(tangents, dim=-1, eps=float(eps)),
        F.normalize(bitangents, dim=-1, eps=float(eps)),
    )


def rasterize_power_foam_linear(
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    features: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    rays: Tensor,
    config: FoamRasterConfig | None = None,
    sorted_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Rasterize bounded power cells with local linear per-cell features.

    `features` is `[N, C, 4]`: base feature plus x/y/z coefficients evaluated
    at the ray midpoint in radius-normalized local cell coordinates.
    """
    if not hasattr(torch.ops, "powerfoam_metal"):
        raise RuntimeError("powerfoam_metal custom ops not found. Build the extension first.")
    if config is None:
        config = FoamRasterConfig()
    if features.ndim != 3 or features.shape[2] != 4:
        raise ValueError("linear features must have shape [N,C,4]")

    rays_b, keep_batch = _normalize_rays(rays.contiguous())
    points = points.contiguous()
    radii = radii.contiguous()
    densities = densities.contiguous()
    output_dim = int(features.shape[1])
    features_flat = features.permute(0, 2, 1).contiguous().reshape(features.shape[0], output_dim * 4)
    adjacency = adjacency.contiguous()
    offsets = offsets.contiguous()
    if sorted_ids is None:
        sorted_ids = _default_sorted_ids(points, radii, rays_b)
    else:
        sorted_ids = sorted_ids.contiguous()

    _check_inputs(
        points,
        radii,
        densities,
        features_flat,
        adjacency,
        offsets,
        rays_b,
        sorted_ids,
        output_dim=output_dim,
        feature_mode=1,
    )
    meta_i32, meta_f32 = _make_meta(
        rays_b,
        points,
        features_flat,
        config,
        output_dim=output_dim,
        feature_mode=1,
    )
    screen_bounds = _projected_screen_bounds(rays_b, points, radii, config)
    out, alpha = _rasterize_apply(
        config,
        points,
        radii,
        densities,
        features_flat,
        adjacency,
        offsets,
        sorted_ids,
        screen_bounds,
        rays_b,
        meta_i32,
        meta_f32,
    )
    if keep_batch:
        return out, alpha
    return out[0], alpha[0]


def rasterize_power_foam_surface_linear(
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    features: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    rays: Tensor,
    config: FoamRasterConfig | None = None,
    sorted_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Rasterize bounded power cells with a fixed camera-facing surface plane.

    This uses the same `[N,C,4]` feature layout as `rasterize_power_foam_linear`,
    but clips each cell by a camera-facing `-z` plane through the cell center and
    evaluates the local-linear feature at that surface intersection.
    """
    if not hasattr(torch.ops, "powerfoam_metal"):
        raise RuntimeError("powerfoam_metal custom ops not found. Build the extension first.")
    if config is None:
        config = FoamRasterConfig()
    if features.ndim != 3 or features.shape[2] != 4:
        raise ValueError("surface-linear features must have shape [N,C,4]")

    rays_b, keep_batch = _normalize_rays(rays.contiguous())
    points = points.contiguous()
    radii = radii.contiguous()
    densities = densities.contiguous()
    output_dim = int(features.shape[1])
    features_flat = features.permute(0, 2, 1).contiguous().reshape(features.shape[0], output_dim * 4)
    adjacency = adjacency.contiguous()
    offsets = offsets.contiguous()
    if sorted_ids is None:
        sorted_ids = _default_sorted_ids(points, radii, rays_b)
    else:
        sorted_ids = sorted_ids.contiguous()

    _check_inputs(
        points,
        radii,
        densities,
        features_flat,
        adjacency,
        offsets,
        rays_b,
        sorted_ids,
        output_dim=output_dim,
        feature_mode=2,
    )
    meta_i32, meta_f32 = _make_meta(
        rays_b,
        points,
        features_flat,
        config,
        output_dim=output_dim,
        feature_mode=2,
    )
    screen_bounds = _projected_screen_bounds(rays_b, points, radii, config)
    out, alpha = _rasterize_apply(
        config,
        points,
        radii,
        densities,
        features_flat,
        adjacency,
        offsets,
        sorted_ids,
        screen_bounds,
        rays_b,
        meta_i32,
        meta_f32,
    )
    if keep_batch:
        return out, alpha
    return out[0], alpha[0]


def rasterize_power_foam_oriented_surface_linear(
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    features: Tensor,
    normals: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    rays: Tensor,
    config: FoamRasterConfig | None = None,
    sorted_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Rasterize bounded power cells with learned per-cell surface normals.

    `features` is `[N,C,4]`. `normals` is `[N,3]` and is expected to already be
    normalized by the caller so autograd can own the normalization gradient.
    """
    if not hasattr(torch.ops, "powerfoam_metal"):
        raise RuntimeError("powerfoam_metal custom ops not found. Build the extension first.")
    if config is None:
        config = FoamRasterConfig()
    if features.ndim != 3 or features.shape[2] != 4:
        raise ValueError("oriented surface-linear features must have shape [N,C,4]")
    if normals.ndim != 2 or normals.shape != (features.shape[0], 3):
        raise ValueError("normals must have shape [N,3]")

    rays_b, keep_batch = _normalize_rays(rays.contiguous())
    points = points.contiguous()
    radii = radii.contiguous()
    densities = densities.contiguous()
    output_dim = int(features.shape[1])
    features_flat = features.permute(0, 2, 1).contiguous().reshape(features.shape[0], output_dim * 4)
    features_flat = torch.cat([features_flat, normals.contiguous()], dim=1).contiguous()
    adjacency = adjacency.contiguous()
    offsets = offsets.contiguous()
    if sorted_ids is None:
        sorted_ids = _default_sorted_ids(points, radii, rays_b)
    else:
        sorted_ids = sorted_ids.contiguous()

    _check_inputs(
        points,
        radii,
        densities,
        features_flat,
        adjacency,
        offsets,
        rays_b,
        sorted_ids,
        output_dim=output_dim,
        feature_mode=3,
    )
    meta_i32, meta_f32 = _make_meta(
        rays_b,
        points,
        features_flat,
        config,
        output_dim=output_dim,
        feature_mode=3,
    )
    screen_bounds = _projected_screen_bounds(rays_b, points, radii, config)
    out, alpha = _rasterize_apply(
        config,
        points,
        radii,
        densities,
        features_flat,
        adjacency,
        offsets,
        sorted_ids,
        screen_bounds,
        rays_b,
        meta_i32,
        meta_f32,
    )
    if keep_batch:
        return out, alpha
    return out[0], alpha[0]


def rasterize_power_foam_quaternion_texel_surface(
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    texel_sites: Tensor,
    texel_features: Tensor,
    quaternions: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    rays: Tensor,
    config: FoamRasterConfig | None = None,
    sorted_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Rasterize oriented texel-surface cells with official quaternion frames.

    This is the same Metal primitive as `rasterize_power_foam_oriented_texel_surface`,
    but the material frame comes from PowerFoam quaternions rather than learned
    normal/tangent vectors. Gradients from the Metal replay backward flow through
    the derived frame and back into `quaternions`.
    """
    if config is None:
        config = FoamRasterConfig()
    normals, tangents, bitangents = quaternion_frames(quaternions, eps=config.eps)
    return rasterize_power_foam_oriented_texel_surface(
        points,
        radii,
        densities,
        texel_sites,
        texel_features,
        normals,
        adjacency,
        offsets,
        rays,
        config,
        sorted_ids=sorted_ids,
        tangents=tangents,
        bitangents=bitangents,
    )


def rasterize_power_foam_quaternion_height_texel_surface(
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    texel_sites: Tensor,
    texel_heights: Tensor,
    texel_features: Tensor,
    quaternions: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    rays: Tensor,
    config: FoamRasterConfig | None = None,
    sorted_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Rasterize height-displaced texel-surface cells with PowerFoam quaternions."""
    if config is None:
        config = FoamRasterConfig()
    normals, tangents, bitangents = quaternion_frames(quaternions, eps=config.eps)
    return rasterize_power_foam_oriented_height_texel_surface(
        points,
        radii,
        densities,
        texel_sites,
        texel_heights,
        texel_features,
        normals,
        adjacency,
        offsets,
        rays,
        config,
        sorted_ids=sorted_ids,
        tangents=tangents,
        bitangents=bitangents,
    )


def rasterize_power_foam_quaternion_height_sv_texel_surface(
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    texel_sites: Tensor,
    texel_heights: Tensor,
    texel_sv_axis: Tensor,
    texel_sv_rgb: Tensor,
    quaternions: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    rays: Tensor,
    config: FoamRasterConfig | None = None,
    sorted_ids: Tensor | None = None,
    return_normal_distance: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    """Rasterize height-displaced texel surfaces with quaternion frames and SV color."""
    if config is None:
        config = FoamRasterConfig()
    normals, tangents, bitangents = quaternion_frames(quaternions, eps=config.eps)
    return rasterize_power_foam_oriented_height_sv_texel_surface(
        points,
        radii,
        densities,
        texel_sites,
        texel_heights,
        texel_sv_axis,
        texel_sv_rgb,
        normals,
        adjacency,
        offsets,
        rays,
        config,
        sorted_ids=sorted_ids,
        tangents=tangents,
        bitangents=bitangents,
        return_normal_distance=return_normal_distance,
    )


def rasterize_power_foam_oriented_texel_surface(
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    texel_sites: Tensor,
    texel_features: Tensor,
    normals: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    rays: Tensor,
    config: FoamRasterConfig | None = None,
    sorted_ids: Tensor | None = None,
    tangents: Tensor | None = None,
    bitangents: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Rasterize oriented surface cells with learned local detail sites.

    `texel_sites` is `[N,S,2]` in radius-normalized local surface coordinates,
    `texel_features` is `[N,S,C]`, and `normals` is `[N,3]`. Optional
    `tangents`/`bitangents` define the material-frame axes used to turn the 3D
    surface hit into texel coordinates; when omitted they are derived from the
    normals with a stable but roll-free frame.
    """
    if not hasattr(torch.ops, "powerfoam_metal"):
        raise RuntimeError("powerfoam_metal custom ops not found. Build the extension first.")
    if config is None:
        config = FoamRasterConfig()
    if texel_sites.ndim != 3 or texel_sites.shape[2] != 2:
        raise ValueError("texel_sites must have shape [N,S,2]")
    if texel_features.ndim != 3:
        raise ValueError("texel_features must have shape [N,S,C]")
    if texel_sites.shape[:2] != texel_features.shape[:2]:
        raise ValueError("texel_sites and texel_features must agree on [N,S]")
    if normals.ndim != 2 or normals.shape != (texel_sites.shape[0], 3):
        raise ValueError("normals must have shape [N,3]")
    if (tangents is None) != (bitangents is None):
        raise ValueError("tangents and bitangents must be provided together")
    if tangents is None or bitangents is None:
        tangents, bitangents = _frame_from_normals(normals)
    if tangents.ndim != 2 or tangents.shape != (texel_sites.shape[0], 3):
        raise ValueError("tangents must have shape [N,3]")
    if bitangents.ndim != 2 or bitangents.shape != (texel_sites.shape[0], 3):
        raise ValueError("bitangents must have shape [N,3]")

    rays_b, keep_batch = _normalize_rays(rays.contiguous())
    points = points.contiguous()
    radii = radii.contiguous()
    densities = densities.contiguous()
    output_dim = int(texel_features.shape[2])
    texel_flat = torch.cat([texel_sites.contiguous(), texel_features.contiguous()], dim=-1)
    features_flat = torch.cat(
        [
            texel_flat.reshape(texel_sites.shape[0], -1),
            normals.contiguous(),
            tangents.contiguous(),
            bitangents.contiguous(),
        ],
        dim=1,
    ).contiguous()
    adjacency = adjacency.contiguous()
    offsets = offsets.contiguous()
    if sorted_ids is None:
        sorted_ids = _default_sorted_ids(points, radii, rays_b)
    else:
        sorted_ids = sorted_ids.contiguous()

    _check_inputs(
        points,
        radii,
        densities,
        features_flat,
        adjacency,
        offsets,
        rays_b,
        sorted_ids,
        output_dim=output_dim,
        feature_mode=4,
    )
    meta_i32, meta_f32 = _make_meta(
        rays_b,
        points,
        features_flat,
        config,
        output_dim=output_dim,
        feature_mode=4,
    )
    screen_bounds = _projected_screen_bounds(rays_b, points, radii, config)
    out, alpha = _rasterize_apply(
        config,
        points,
        radii,
        densities,
        features_flat,
        adjacency,
        offsets,
        sorted_ids,
        screen_bounds,
        rays_b,
        meta_i32,
        meta_f32,
    )
    if keep_batch:
        return out, alpha
    return out[0], alpha[0]


def rasterize_power_foam_oriented_height_texel_surface(
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    texel_sites: Tensor,
    texel_heights: Tensor,
    texel_features: Tensor,
    normals: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    rays: Tensor,
    config: FoamRasterConfig | None = None,
    sorted_ids: Tensor | None = None,
    tangents: Tensor | None = None,
    bitangents: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Rasterize oriented surface cells with per-site height displacement.

    `texel_heights` is `[N,S]` in world units along the cell normal. The Metal
    layout stores each texel as `[u, v, height, feature...]`, followed by the
    normal/tangent/bitangent frame footer.
    """
    if not hasattr(torch.ops, "powerfoam_metal"):
        raise RuntimeError("powerfoam_metal custom ops not found. Build the extension first.")
    if config is None:
        config = FoamRasterConfig()
    if texel_sites.ndim != 3 or texel_sites.shape[2] != 2:
        raise ValueError("texel_sites must have shape [N,S,2]")
    if texel_heights.ndim != 2 or texel_heights.shape != texel_sites.shape[:2]:
        raise ValueError("texel_heights must have shape [N,S]")
    if texel_features.ndim != 3:
        raise ValueError("texel_features must have shape [N,S,C]")
    if texel_sites.shape[:2] != texel_features.shape[:2]:
        raise ValueError("texel_sites and texel_features must agree on [N,S]")
    if normals.ndim != 2 or normals.shape != (texel_sites.shape[0], 3):
        raise ValueError("normals must have shape [N,3]")
    if (tangents is None) != (bitangents is None):
        raise ValueError("tangents and bitangents must be provided together")
    if tangents is None or bitangents is None:
        tangents, bitangents = _frame_from_normals(normals)
    if tangents.ndim != 2 or tangents.shape != (texel_sites.shape[0], 3):
        raise ValueError("tangents must have shape [N,3]")
    if bitangents.ndim != 2 or bitangents.shape != (texel_sites.shape[0], 3):
        raise ValueError("bitangents must have shape [N,3]")

    rays_b, keep_batch = _normalize_rays(rays.contiguous())
    points = points.contiguous()
    radii = radii.contiguous()
    densities = densities.contiguous()
    output_dim = int(texel_features.shape[2])
    texel_flat = torch.cat(
        [
            texel_sites.contiguous(),
            texel_heights.contiguous().unsqueeze(-1),
            texel_features.contiguous(),
        ],
        dim=-1,
    )
    features_flat = torch.cat(
        [
            texel_flat.reshape(texel_sites.shape[0], -1),
            normals.contiguous(),
            tangents.contiguous(),
            bitangents.contiguous(),
        ],
        dim=1,
    ).contiguous()
    adjacency = adjacency.contiguous()
    offsets = offsets.contiguous()
    if sorted_ids is None:
        sorted_ids = _default_sorted_ids(points, radii, rays_b)
    else:
        sorted_ids = sorted_ids.contiguous()

    _check_inputs(
        points,
        radii,
        densities,
        features_flat,
        adjacency,
        offsets,
        rays_b,
        sorted_ids,
        output_dim=output_dim,
        feature_mode=5,
    )
    meta_i32, meta_f32 = _make_meta(
        rays_b,
        points,
        features_flat,
        config,
        output_dim=output_dim,
        feature_mode=5,
    )
    screen_bounds = _projected_screen_bounds(rays_b, points, radii, config)
    out, alpha = _rasterize_apply(
        config,
        points,
        radii,
        densities,
        features_flat,
        adjacency,
        offsets,
        sorted_ids,
        screen_bounds,
        rays_b,
        meta_i32,
        meta_f32,
    )
    if keep_batch:
        return out, alpha
    return out[0], alpha[0]


def rasterize_power_foam_oriented_height_sv_texel_surface(
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    texel_sites: Tensor,
    texel_heights: Tensor,
    texel_sv_axis: Tensor,
    texel_sv_rgb: Tensor,
    normals: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    rays: Tensor,
    config: FoamRasterConfig | None = None,
    sorted_ids: Tensor | None = None,
    tangents: Tensor | None = None,
    bitangents: Tensor | None = None,
    return_normal_distance: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    """Rasterize height-displaced texel surfaces with spherical-Voronoi color.

    `texel_sv_axis` and `texel_sv_rgb` have shape `[N,S,D,3]`. The SV view
    direction uses each texel's world-space site relative to the ray origin;
    the Metal backward matches the official detach semantics by not routing SV
    color-query gradients back into geometry.
    """
    if not hasattr(torch.ops, "powerfoam_metal"):
        raise RuntimeError("powerfoam_metal custom ops not found. Build the extension first.")
    if config is None:
        config = FoamRasterConfig()
    if texel_sites.ndim != 3 or texel_sites.shape[2] != 2:
        raise ValueError("texel_sites must have shape [N,S,2]")
    if texel_heights.ndim != 2 or texel_heights.shape != texel_sites.shape[:2]:
        raise ValueError("texel_heights must have shape [N,S]")
    if texel_sv_axis.ndim != 4 or texel_sv_axis.shape[-1] != 3:
        raise ValueError("texel_sv_axis must have shape [N,S,D,3]")
    if texel_sv_rgb.ndim != 4 or texel_sv_rgb.shape != texel_sv_axis.shape:
        raise ValueError("texel_sv_rgb must have shape [N,S,D,3] matching texel_sv_axis")
    if texel_sv_axis.shape[:2] != texel_sites.shape[:2]:
        raise ValueError("texel_sites and SV tensors must agree on [N,S]")
    if normals.ndim != 2 or normals.shape != (texel_sites.shape[0], 3):
        raise ValueError("normals must have shape [N,3]")
    if (tangents is None) != (bitangents is None):
        raise ValueError("tangents and bitangents must be provided together")
    if tangents is None or bitangents is None:
        tangents, bitangents = _frame_from_normals(normals)
    if tangents.ndim != 2 or tangents.shape != (texel_sites.shape[0], 3):
        raise ValueError("tangents must have shape [N,3]")
    if bitangents.ndim != 2 or bitangents.shape != (texel_sites.shape[0], 3):
        raise ValueError("bitangents must have shape [N,3]")

    rays_b, keep_batch = _normalize_rays(rays.contiguous())
    points = points.contiguous()
    radii = radii.contiguous()
    densities = densities.contiguous()
    sv_dof = int(texel_sv_axis.shape[2])
    texel_flat = torch.cat(
        [
            texel_sites.contiguous(),
            texel_heights.contiguous().unsqueeze(-1),
            texel_sv_axis.contiguous().reshape(texel_sites.shape[0], texel_sites.shape[1], sv_dof * 3),
            texel_sv_rgb.contiguous().reshape(texel_sites.shape[0], texel_sites.shape[1], sv_dof * 3),
        ],
        dim=-1,
    )
    features_flat = torch.cat(
        [
            texel_flat.reshape(texel_sites.shape[0], -1),
            normals.contiguous(),
            tangents.contiguous(),
            bitangents.contiguous(),
        ],
        dim=1,
    ).contiguous()
    adjacency = adjacency.contiguous()
    offsets = offsets.contiguous()
    if sorted_ids is None:
        sorted_ids = _default_sorted_ids(points, radii, rays_b)
    else:
        sorted_ids = sorted_ids.contiguous()

    _check_inputs(
        points,
        radii,
        densities,
        features_flat,
        adjacency,
        offsets,
        rays_b,
        sorted_ids,
        output_dim=3,
        feature_mode=6,
        sv_dof=sv_dof,
    )
    meta_i32, meta_f32 = _make_meta(
        rays_b,
        points,
        features_flat,
        config,
        output_dim=3,
        feature_mode=6,
        sv_dof=sv_dof,
    )
    screen_bounds = _projected_screen_bounds(rays_b, points, radii, config)
    if return_normal_distance:
        out, alpha, normal_distance = _rasterize_apply_with_normal_distance(
            config,
            points,
            radii,
            densities,
            features_flat,
            adjacency,
            offsets,
            sorted_ids,
            screen_bounds,
            rays_b,
            meta_i32,
            meta_f32,
        )
        if keep_batch:
            return out, alpha, normal_distance
        return out[0], alpha[0], normal_distance[0]
    out, alpha = _rasterize_apply(
        config,
        points,
        radii,
        densities,
        features_flat,
        adjacency,
        offsets,
        sorted_ids,
        screen_bounds,
        rays_b,
        meta_i32,
        meta_f32,
    )
    if keep_batch:
        return out, alpha
    return out[0], alpha[0]


def rasterize_power_foam_oriented_height_sv_texel_surface_aux(
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    texel_sites: Tensor,
    texel_heights: Tensor,
    texel_sv_axis: Tensor,
    texel_sv_rgb: Tensor,
    normals: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    rays: Tensor,
    target: Tensor | None = None,
    config: FoamRasterConfig | None = None,
    sorted_ids: Tensor | None = None,
    tangents: Tensor | None = None,
    bitangents: Tensor | None = None,
    depth_quantiles: Tensor | Sequence[float] | None = None,
) -> FoamAuxOutputs:
    """Compute tiled Metal aux outputs for the full height+SV primitive."""
    if config is None:
        config = FoamRasterConfig(use_tiled=True)
    if texel_sites.ndim != 3 or texel_sites.shape[2] != 2:
        raise ValueError("texel_sites must have shape [N,S,2]")
    if texel_heights.ndim != 2 or texel_heights.shape != texel_sites.shape[:2]:
        raise ValueError("texel_heights must have shape [N,S]")
    if texel_sv_axis.ndim != 4 or texel_sv_axis.shape[-1] != 3:
        raise ValueError("texel_sv_axis must have shape [N,S,D,3]")
    if texel_sv_rgb.ndim != 4 or texel_sv_rgb.shape != texel_sv_axis.shape:
        raise ValueError("texel_sv_rgb must have shape [N,S,D,3] matching texel_sv_axis")
    if texel_sv_axis.shape[:2] != texel_sites.shape[:2]:
        raise ValueError("texel_sites and SV tensors must agree on [N,S]")
    if normals.ndim != 2 or normals.shape != (texel_sites.shape[0], 3):
        raise ValueError("normals must have shape [N,3]")
    if (tangents is None) != (bitangents is None):
        raise ValueError("tangents and bitangents must be provided together")
    if tangents is None or bitangents is None:
        tangents, bitangents = _frame_from_normals(normals)

    sv_dof = int(texel_sv_axis.shape[2])
    texel_flat = torch.cat(
        [
            texel_sites.contiguous(),
            texel_heights.contiguous().unsqueeze(-1),
            texel_sv_axis.contiguous().reshape(texel_sites.shape[0], texel_sites.shape[1], sv_dof * 3),
            texel_sv_rgb.contiguous().reshape(texel_sites.shape[0], texel_sites.shape[1], sv_dof * 3),
        ],
        dim=-1,
    )
    features_flat = torch.cat(
        [
            texel_flat.reshape(texel_sites.shape[0], -1),
            normals.contiguous(),
            tangents.contiguous(),
            bitangents.contiguous(),
        ],
        dim=1,
    ).contiguous()
    return rasterize_power_foam_aux(
        points,
        radii,
        densities,
        features_flat,
        adjacency,
        offsets,
        rays,
        target,
        config,
        sorted_ids=sorted_ids,
        output_dim=3,
        feature_mode=6,
        sv_dof=sv_dof,
        depth_quantiles=depth_quantiles,
    )


def rasterize_power_foam_quaternion_height_sv_texel_surface_aux(
    points: Tensor,
    radii: Tensor,
    densities: Tensor,
    texel_sites: Tensor,
    texel_heights: Tensor,
    texel_sv_axis: Tensor,
    texel_sv_rgb: Tensor,
    quaternions: Tensor,
    adjacency: Tensor,
    offsets: Tensor,
    rays: Tensor,
    target: Tensor | None = None,
    config: FoamRasterConfig | None = None,
    sorted_ids: Tensor | None = None,
    depth_quantiles: Tensor | Sequence[float] | None = None,
) -> FoamAuxOutputs:
    """Compute tiled Metal aux outputs for quaternion height+SV PowerFoam."""
    if config is None:
        config = FoamRasterConfig(use_tiled=True)
    normals, tangents, bitangents = quaternion_frames(quaternions, eps=config.eps)
    return rasterize_power_foam_oriented_height_sv_texel_surface_aux(
        points,
        radii,
        densities,
        texel_sites,
        texel_heights,
        texel_sv_axis,
        texel_sv_rgb,
        normals,
        adjacency,
        offsets,
        rays,
        target,
        config,
        sorted_ids=sorted_ids,
        tangents=tangents,
        bitangents=bitangents,
        depth_quantiles=depth_quantiles,
    )
