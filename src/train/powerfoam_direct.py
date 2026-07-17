from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


POWERFOAM_SOFTPLUS_BETA = 100.0


@dataclass(frozen=True)
class PowerFoamRenderOptions:
    near_plane: float = 0.05
    alpha_threshold: float = 0.0
    transmittance_threshold: float = 1.0e-4
    max_alpha: float = 0.99
    eps: float = 1.0e-6
    texel_temperature: float = 10.0
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)


def direct_powerfoam_render_options(render_cfg: dict[str, Any]) -> PowerFoamRenderOptions:
    return PowerFoamRenderOptions(
        near_plane=float(render_cfg["near_plane"]),
        alpha_threshold=float(render_cfg["alpha_threshold"]),
        transmittance_threshold=float(render_cfg["transmittance_threshold"]),
        max_alpha=float(render_cfg["max_alpha"]),
        eps=float(render_cfg["eps"]),
        texel_temperature=float(render_cfg["texel_temperature"]),
        background=tuple(float(v) for v in render_cfg["background"]),
    )


@dataclass(frozen=True)
class PowerFoamInitialization:
    points: torch.Tensor
    radii: torch.Tensor
    quaternions: torch.Tensor
    texel_sites: torch.Tensor
    texel_sv_axis: torch.Tensor
    texel_sv_rgb: torch.Tensor
    texel_height: torch.Tensor


@dataclass(frozen=True)
class PowerFoamRenderResult:
    rendered: torch.Tensor
    alpha: torch.Tensor
    transmittance: torch.Tensor
    stop_order: torch.Tensor
    normal_distance: torch.Tensor
    normal: torch.Tensor
    contrib: torch.Tensor
    point_error: torch.Tensor
    visible_mask: torch.Tensor

    def __iter__(self):
        yield self.rendered
        yield self.alpha
        yield self.transmittance
        yield self.stop_order


def inverse_softplus(value: torch.Tensor, *, beta: float = 1.0) -> torch.Tensor:
    value = value.clamp_min(1.0e-8)
    beta_value = float(beta) * value
    one_minus_exp = (-torch.expm1(-beta_value)).clamp_min(1.0e-30)
    return value + torch.log(one_minus_exp) / float(beta)


def logit_clamped(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(1.0e-5, 1.0 - 1.0e-5)
    return torch.log(value / (1.0 - value))


def atanh_clamped(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(-1.0 + 1.0e-5, 1.0 - 1.0e-5)
    return 0.5 * (torch.log1p(value) - torch.log1p(-value))


def make_pinhole_ray_directions(height: int, width: int, fov_degrees: float, device: torch.device) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    half_fov = math.radians(float(fov_degrees)) * 0.5
    focal = 0.5 * float(width) / math.tan(half_fov)
    x = (xs + 0.5 - float(width) * 0.5) / focal
    y = -(ys + 0.5 - float(height) * 0.5) / focal
    dirs = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    return F.normalize(dirs, dim=-1)


def camera_facing_quaternion(frame_count: int, cell_count: int) -> torch.Tensor:
    # Official PowerFoam stores orientation as quaternions and derives the
    # normal/tangent frame from the first/second/third rotation-matrix columns.
    # This rotates the official +x normal onto camera-facing -z.
    q = torch.zeros(frame_count, cell_count, 4, dtype=torch.float32)
    q[..., 0] = math.sqrt(0.5)
    q[..., 2] = -math.sqrt(0.5)
    return q


def quaternion_frames(quaternions: torch.Tensor, eps: float = 1.0e-6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = quaternions / quaternions.norm(dim=-1, keepdim=True).clamp_min(eps)
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
        F.normalize(normals, dim=-1, eps=eps),
        F.normalize(tangents, dim=-1, eps=eps),
        F.normalize(bitangents, dim=-1, eps=eps),
    )


def build_knn_adjacency(points: torch.Tensor, neighbor_count: int) -> torch.Tensor:
    if points.dim() != 3:
        raise ValueError("points must be [T,N,3]")
    frame_count, cell_count, _ = points.shape
    if cell_count < 2:
        return torch.empty(frame_count, cell_count, 0, dtype=torch.long)

    k = min(max(int(neighbor_count), 1), cell_count - 1)
    distances = torch.cdist(points, points)
    eye = torch.eye(cell_count, dtype=torch.bool, device=points.device).unsqueeze(0)
    distances = distances.masked_fill(eye, torch.inf)
    return torch.topk(distances, k=k, largest=False, dim=-1).indices.to(torch.long)


def build_power_adjacency(
    points: torch.Tensor,
    radii: torch.Tensor,
    neighbor_count: int,
    *,
    mode: str = "overlap",
) -> torch.Tensor:
    if mode == "knn":
        return build_knn_adjacency(points, neighbor_count)
    if points.dim() != 3 or radii.dim() != 2:
        raise ValueError("points must be [T,N,3] and radii must be [T,N]")
    frame_count, cell_count, _ = points.shape
    if cell_count < 2:
        return torch.empty(frame_count, cell_count, 0, dtype=torch.long, device=points.device)

    k = min(max(int(neighbor_count), 1), cell_count - 1)
    distances = torch.cdist(points, points)
    eye = torch.eye(cell_count, dtype=torch.bool, device=points.device).unsqueeze(0)
    distances = distances.masked_fill(eye, torch.inf)
    if mode == "cech_aabb":
        overlap = distances <= (radii[:, :, None] + radii[:, None, :])
        max_degree = int(overlap.sum(dim=-1).max().item())
        if max_degree == 0:
            return torch.empty(frame_count, cell_count, 0, dtype=torch.long, device=points.device)
        adjacency = torch.full((frame_count, cell_count, max_degree), -1, dtype=torch.long, device=points.device)
        for frame in range(frame_count):
            for cell in range(cell_count):
                ids = torch.nonzero(overlap[frame, cell], as_tuple=False).flatten()
                if ids.numel() > 0:
                    order = torch.argsort(distances[frame, cell, ids], stable=True)
                    adjacency[frame, cell, : ids.numel()] = ids[order].to(torch.long)
        return adjacency
    if mode == "overlap":
        overlap = distances <= (radii[:, :, None] + radii[:, None, :])
        score = torch.where(overlap, distances, torch.full_like(distances, torch.inf))
        # Conservative fallback: if a cell has fewer overlaps than the cap, fill
        # the remaining slots with nearest neighbors rather than dropping faces.
        overlap_count = torch.isfinite(score).sum(dim=-1, keepdim=True)
        score = torch.where(overlap_count > 0, score, distances)
        top = torch.topk(score, k=k, largest=False, dim=-1).indices
        fallback = torch.topk(distances, k=k, largest=False, dim=-1).indices
        finite = torch.gather(score, dim=-1, index=top).isfinite()
        return torch.where(finite, top, fallback).to(torch.long)
    raise ValueError(f"Unknown power adjacency mode {mode!r}")


def estimate_knn_radii(points: torch.Tensor, *, radius_scale: float, radius_min: float) -> torch.Tensor:
    if points.size(1) < 2:
        return torch.full(points.shape[:2], float(radius_min), dtype=points.dtype, device=points.device)
    distances = torch.cdist(points, points)
    eye = torch.eye(points.size(1), dtype=torch.bool, device=points.device).unsqueeze(0)
    distances = distances.masked_fill(eye, torch.inf)
    k = min(6, points.size(1) - 1)
    radii = torch.topk(distances, k=k, largest=False, dim=-1).values.mean(dim=-1)
    return (radii * float(radius_scale)).clamp_min(float(radius_min))


def make_image_init_uv(
    cell_count: int,
    *,
    jitter_fraction: float,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    cols = math.ceil(math.sqrt(float(cell_count)))
    rows = math.ceil(float(cell_count) / float(cols))
    cell_ids = torch.arange(cell_count, dtype=torch.float32)
    x_cell = torch.remainder(cell_ids, cols) + 0.5
    y_cell = torch.floor(cell_ids / float(cols)) + 0.5
    jitter = float(jitter_fraction)
    if jitter > 0.0:
        noise = (torch.rand(cell_count, 2, generator=generator) - 0.5) * jitter
        x_cell = x_cell + noise[:, 0]
        y_cell = y_cell + noise[:, 1]
    x01 = (x_cell / float(cols)).clamp(0.5 / float(cols), 1.0 - 0.5 / float(cols))
    y01 = (y_cell / float(rows)).clamp(0.5 / float(rows), 1.0 - 0.5 / float(rows))
    return x01, y01, rows, cols


def initialize_powerfoam_from_video(
    init_frames: torch.Tensor,
    *,
    cell_count: int,
    xy_extent: float,
    z_min: float,
    z_max: float,
    fov_degrees: float,
    image_init_depth: float | None,
    image_init_jitter: float = 0.0,
    generator: torch.Generator | None = None,
    image_init_uv: tuple[torch.Tensor, torch.Tensor, int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if init_frames.dim() != 4 or init_frames.size(1) != 3:
        raise ValueError("init_frames must be [T,3,H,W]")
    frame_count, _, height, width = init_frames.shape
    if frame_count < 1:
        raise ValueError("init_frames must contain at least one frame")

    if image_init_uv is None:
        x01, y01, rows, cols = make_image_init_uv(
            cell_count,
            jitter_fraction=image_init_jitter,
            generator=generator,
        )
    else:
        x01, y01, rows, cols = image_init_uv

    depth = 0.5 * (float(z_min) + float(z_max)) if image_init_depth is None else float(image_init_depth)
    depth = min(max(depth, float(z_min) + 1.0e-4), float(z_max) - 1.0e-4)
    tan_half_fov = math.tan(math.radians(float(fov_degrees)) * 0.5)
    x_ray = (2.0 * x01 - 1.0) * tan_half_fov
    y_ray = -(2.0 * y01 - 1.0) * tan_half_fov * (float(height) / float(width))
    xy = torch.stack([x_ray, y_ray], dim=-1) * depth
    xy = xy.clamp(-0.95 * float(xy_extent), 0.95 * float(xy_extent))
    z = torch.full((cell_count, 1), depth, dtype=torch.float32)
    points = torch.cat([xy, z], dim=-1).unsqueeze(0).repeat(frame_count, 1, 1)

    sample_grid = torch.stack([2.0 * x01 - 1.0, 2.0 * y01 - 1.0], dim=-1).view(1, cell_count, 1, 2)
    sample_grid = sample_grid.repeat(frame_count, 1, 1, 1)
    colors = F.grid_sample(
        init_frames.detach().cpu().float().clamp(0.0, 1.0),
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    colors = colors.squeeze(-1).permute(0, 2, 1).contiguous()
    return points, colors


def initialize_full_powerfoam_from_video(
    init_frames: torch.Tensor,
    *,
    cell_count: int,
    xy_extent: float,
    z_min: float,
    z_max: float,
    fov_degrees: float,
    image_init_depth: float | None,
    radius_min: float,
    radius_scale: float,
    num_texel_sites: int,
    sv_dof: int,
    sv_axis_init: float,
    image_init_jitter: float = 0.0,
    generator: torch.Generator | None = None,
) -> PowerFoamInitialization:
    if init_frames.dim() != 4 or init_frames.size(1) != 3:
        raise ValueError("init_frames must be [T,3,H,W]")
    frame_count, _, height, width = init_frames.shape
    center_x01, center_y01, rows, cols = make_image_init_uv(
        cell_count,
        jitter_fraction=image_init_jitter,
        generator=generator,
    )
    points, _ = initialize_powerfoam_from_video(
        init_frames,
        cell_count=cell_count,
        xy_extent=xy_extent,
        z_min=z_min,
        z_max=z_max,
        fov_degrees=fov_degrees,
        image_init_depth=image_init_depth,
        image_init_uv=(center_x01, center_y01, rows, cols),
    )
    radii = estimate_knn_radii(points, radius_scale=radius_scale, radius_min=radius_min)
    quaternions = camera_facing_quaternion(frame_count, cell_count)
    normals, tangents, bitangents = quaternion_frames(quaternions)

    site_cols = math.ceil(math.sqrt(float(num_texel_sites)))
    site_rows = math.ceil(float(num_texel_sites) / float(site_cols))
    site_x = (torch.arange(site_cols, dtype=torch.float32) + 0.5) / float(site_cols) - 0.5
    site_y = (torch.arange(site_rows, dtype=torch.float32) + 0.5) / float(site_rows) - 0.5
    sy, sx = torch.meshgrid(site_y, site_x, indexing="ij")
    site_offsets_01 = torch.stack([sx.reshape(-1), sy.reshape(-1)], dim=-1)[:num_texel_sites]
    sample_x01 = (center_x01[:, None] + 0.85 * site_offsets_01[None, :, 0] / float(cols)).clamp(0.0, 1.0)
    sample_y01 = (center_y01[:, None] + 0.85 * site_offsets_01[None, :, 1] / float(rows)).clamp(0.0, 1.0)
    sample_grid = torch.stack([2.0 * sample_x01 - 1.0, 2.0 * sample_y01 - 1.0], dim=-1)
    sample_grid = sample_grid.unsqueeze(0).repeat(frame_count, 1, 1, 1)
    texel_colors = F.grid_sample(
        init_frames.detach().cpu().float().clamp(0.0, 1.0),
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).permute(0, 2, 3, 1).contiguous()

    depth = points[:, :, None, 2:3]
    tan_half_fov = math.tan(math.radians(float(fov_degrees)) * 0.5)
    x_ray = (2.0 * sample_x01 - 1.0) * tan_half_fov
    y_ray = -(2.0 * sample_y01 - 1.0) * tan_half_fov * (float(height) / float(width))
    texel_world_xy = torch.stack([x_ray, y_ray], dim=-1).unsqueeze(0).repeat(frame_count, 1, 1, 1) * depth
    texel_world = torch.cat([texel_world_xy, depth.expand(-1, -1, num_texel_sites, -1)], dim=-1)
    deltas = texel_world - points[:, :, None, :]
    texel_site_offsets = torch.stack(
        [
            (deltas * tangents[:, :, None, :]).sum(dim=-1),
            (deltas * bitangents[:, :, None, :]).sum(dim=-1),
        ],
        dim=-1,
    )
    texel_sites = texel_site_offsets / radii[:, :, None, None].clamp_min(1.0e-6)

    view_dirs = F.normalize(texel_world, dim=-1).unsqueeze(3).repeat(1, 1, 1, int(sv_dof), 1)
    texel_sv_axis = view_dirs * float(sv_axis_init)
    texel_sv_rgb = (texel_colors - 0.5).unsqueeze(3).repeat(1, 1, 1, int(sv_dof), 1)
    texel_height = torch.zeros(frame_count, cell_count, num_texel_sites, dtype=torch.float32)
    _ = normals  # keep the init contract explicit; normals are derived from quaternions at decode time.
    return PowerFoamInitialization(
        points=points,
        radii=radii,
        quaternions=quaternions,
        texel_sites=texel_sites,
        texel_sv_axis=texel_sv_axis,
        texel_sv_rgb=texel_sv_rgb,
        texel_height=texel_height,
    )


def initialize_random_full_powerfoam(
    *,
    frame_count: int,
    cell_count: int,
    xy_extent: float,
    z_min: float,
    z_max: float,
    radius_init: float,
    radius_min: float,
    num_texel_sites: int,
    sv_dof: int,
    sv_axis_init: float,
    generator: torch.Generator,
) -> PowerFoamInitialization:
    xy = (torch.rand(frame_count, cell_count, 2, generator=generator) * 2.0 - 1.0) * float(xy_extent)
    z = torch.rand(frame_count, cell_count, 1, generator=generator) * (float(z_max) - float(z_min)) + float(z_min)
    points = torch.cat([xy, z], dim=-1)
    radii = torch.full((frame_count, cell_count), max(float(radius_init), float(radius_min)), dtype=torch.float32)
    quaternions = F.normalize(torch.randn(frame_count, cell_count, 4, generator=generator), dim=-1)
    texel_sites = 0.1 * torch.randn(frame_count, cell_count, num_texel_sites, 2, generator=generator)
    texel_sv_axis = float(sv_axis_init) * F.normalize(
        torch.randn(frame_count, cell_count, num_texel_sites, sv_dof, 3, generator=generator), dim=-1
    )
    texel_sv_rgb = torch.rand(frame_count, cell_count, num_texel_sites, sv_dof, 3, generator=generator) - 0.5
    texel_height = torch.zeros(frame_count, cell_count, num_texel_sites, dtype=torch.float32)
    return PowerFoamInitialization(
        points=points,
        radii=radii,
        quaternions=quaternions,
        texel_sites=texel_sites,
        texel_sv_axis=texel_sv_axis,
        texel_sv_rgb=texel_sv_rgb,
        texel_height=texel_height,
    )


def spherical_voronoi_texel_color(
    texel_sites_world: torch.Tensor,
    texel_sv_axis: torch.Tensor,
    texel_sv_rgb: torch.Tensor,
    *,
    view_origin: torch.Tensor | None = None,
    eps: float,
) -> torch.Tensor:
    if view_origin is None:
        view_origin = torch.zeros(3, dtype=texel_sites_world.dtype, device=texel_sites_world.device)
    view_dirs = F.normalize(texel_sites_world - view_origin, dim=-1, eps=eps).unsqueeze(-2)
    temps = texel_sv_axis.norm(dim=-1).clamp_min(eps)
    axes = texel_sv_axis / temps[..., None]
    dist = (view_dirs - axes).norm(dim=-1)
    weights = torch.exp(-temps * dist)
    weight_sum = weights.sum(dim=-1, keepdim=True).clamp_min(eps)
    color = (weights[..., None] * texel_sv_rgb).sum(dim=-2) / weight_sum + 0.5
    return color.clamp_min(0.0)


def query_powerfoam_texels(
    origins: torch.Tensor,
    dirs: torch.Tensor,
    t_near: torch.Tensor,
    center: torch.Tensor,
    radius: torch.Tensor,
    normal: torch.Tensor,
    texel_sites: torch.Tensor,
    texel_rgb: torch.Tensor | None,
    texel_height: torch.Tensor,
    options: PowerFoamRenderOptions,
    *,
    texel_sv_axis: torch.Tensor | None = None,
    texel_sv_rgb: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    denom = torch.einsum("bpc,bc->bp", dirs, normal)
    safe_denom = torch.where(denom.abs() > options.eps, denom, denom.sign().clamp(min=0.0) * 2.0 - 1.0)
    safe_denom = torch.where(safe_denom.abs() > options.eps, safe_denom, torch.full_like(safe_denom, options.eps))
    plane_offset = (center * normal).sum(dim=-1)[:, None]
    origin_plane = torch.einsum("bpc,bc->bp", origins, normal)
    t_surf0 = (plane_offset - origin_plane) / safe_denom
    t_query0 = torch.where(denom >= 0.0, t_near, torch.maximum(t_near, t_surf0))
    point0 = origins + t_query0[:, :, None] * dirs
    dist_sq0 = (point0[:, :, None, :] - texel_sites[:, None, :, :]).square().sum(dim=-1)
    weights0 = torch.exp(-float(options.texel_temperature) * dist_sq0 / radius[:, None, None].square().clamp_min(options.eps))
    weight_sum0 = weights0.sum(dim=-1).clamp_min(options.eps)
    height = (weights0 * texel_height[:, None, :]).sum(dim=-1) / weight_sum0

    t_surf = (plane_offset - origin_plane + height) / safe_denom
    t_query = torch.where(denom >= 0.0, t_near, torch.maximum(t_near, t_surf))
    point = origins + t_query[:, :, None] * dirs
    dist_sq = (point[:, :, None, :] - texel_sites[:, None, :, :]).square().sum(dim=-1)
    weights = torch.exp(-float(options.texel_temperature) * dist_sq / radius[:, None, None].square().clamp_min(options.eps))
    weight_sum = weights.sum(dim=-1, keepdim=True).clamp_min(options.eps)
    if texel_sv_axis is not None or texel_sv_rgb is not None:
        if texel_sv_axis is None or texel_sv_rgb is None:
            raise ValueError("texel_sv_axis and texel_sv_rgb must be provided together.")
        view_dirs = F.normalize(texel_sites[:, None, :, None, :] - origins[:, :, None, None, :], dim=-1, eps=options.eps)
        temps = texel_sv_axis.norm(dim=-1).clamp_min(options.eps)
        axes = texel_sv_axis / temps[..., None]
        sv_dist = (view_dirs - axes[:, None]).norm(dim=-1)
        sv_weights = torch.exp(-temps[:, None] * sv_dist)
        sv_weight_sum = sv_weights.sum(dim=-1, keepdim=True).clamp_min(options.eps)
        texel_color = (sv_weights[..., None] * texel_sv_rgb[:, None]).sum(dim=-2) / sv_weight_sum + 0.5
        texel_color = texel_color.clamp_min(0.0)
    else:
        if texel_rgb is None:
            raise ValueError("texel_rgb is required unless SV color parameters are provided.")
        texel_color = texel_rgb[:, None, :, :]
    color = (weights[..., None] * texel_color).sum(dim=-2) / weight_sum
    return t_surf, denom, color


def prepare_powerfoam_rays(
    rays: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if rays.ndim not in {3, 4} or rays.shape[-1] not in {3, 6}:
        raise ValueError("rays must have shape [H,W,3], [H,W,6], [B,H,W,3], or [B,H,W,6].")
    if rays.ndim == 3:
        rays = rays.unsqueeze(0)
    if int(rays.shape[0]) not in {1, int(batch_size)}:
        raise ValueError(f"rays batch must be 1 or {batch_size}, got {rays.shape[0]}.")
    rays = rays.to(device=device, dtype=dtype)
    if int(rays.shape[0]) == 1 and int(batch_size) != 1:
        rays = rays.expand(int(batch_size), -1, -1, -1)
    height, width = int(rays.shape[1]), int(rays.shape[2])
    flat = rays.reshape(int(batch_size), height * width, int(rays.shape[-1]))
    if int(rays.shape[-1]) == 6:
        origins = flat[..., :3]
        dirs = flat[..., 3:]
    else:
        origins = torch.zeros(int(batch_size), height * width, 3, dtype=dtype, device=device)
        dirs = flat
    dirs = F.normalize(dirs, dim=-1, eps=1.0e-6)
    return origins, dirs, height, width


def render_powerfoam_torch(
    points: torch.Tensor,
    radii: torch.Tensor,
    densities: torch.Tensor,
    normals: torch.Tensor,
    texel_sites: torch.Tensor,
    texel_rgb: torch.Tensor | None,
    texel_height: torch.Tensor,
    adjacency: torch.Tensor,
    rays: torch.Tensor,
    options: PowerFoamRenderOptions,
    target_rgb: torch.Tensor | None = None,
    texel_sv_axis: torch.Tensor | None = None,
    texel_sv_rgb: torch.Tensor | None = None,
) -> PowerFoamRenderResult:
    if points.dim() != 3 or points.size(-1) != 3:
        raise ValueError("points must be [B,N,3]")
    if texel_sites.dim() != 4 or texel_sites.size(-1) != 3:
        raise ValueError("texel_sites must be [B,N,S,3]")
    if texel_rgb is not None and texel_rgb.shape != texel_sites.shape:
        raise ValueError("texel_rgb must match texel_sites shape")
    if texel_sv_axis is not None and (texel_sv_axis.dim() != 5 or texel_sv_axis.shape[:3] != texel_sites.shape[:3]):
        raise ValueError("texel_sv_axis must have shape [B,N,S,D,3] matching texel_sites.")
    if (texel_sv_axis is None) != (texel_sv_rgb is None):
        raise ValueError("texel_sv_axis and texel_sv_rgb must be provided together.")
    if texel_sv_rgb is not None and texel_sv_rgb.shape != texel_sv_axis.shape:
        raise ValueError("texel_sv_rgb must match texel_sv_axis.")
    if adjacency.dim() != 3:
        raise ValueError("adjacency must be [B,N,K]")

    batch_size, cell_count, _ = points.shape
    device = points.device
    dtype = points.dtype
    origins, dirs, height, width = prepare_powerfoam_rays(rays, batch_size=batch_size, device=device, dtype=dtype)
    pixel_count = height * width
    camera_origin = origins[:, 0, :]
    batch_ids = torch.arange(batch_size, device=device)
    if target_rgb is not None:
        if target_rgb.shape != (batch_size, 3, height, width):
            raise ValueError("target_rgb must be [B,3,H,W] matching ray_dirs")
        target_flat = target_rgb.to(device=device, dtype=dtype).permute(0, 2, 3, 1).reshape(batch_size, pixel_count, 3)
    else:
        target_flat = None

    out = torch.zeros(batch_size, pixel_count, 3, dtype=dtype, device=device)
    transmittance = torch.ones(batch_size, pixel_count, dtype=dtype, device=device)
    stop_order = torch.full((batch_size, pixel_count), cell_count, dtype=torch.int32, device=device)
    stopped = torch.zeros(batch_size, pixel_count, dtype=torch.bool, device=device)
    normal_distance = torch.zeros(batch_size, pixel_count, dtype=dtype, device=device)
    normal_out = torch.zeros(batch_size, pixel_count, 3, dtype=dtype, device=device)
    contrib = torch.zeros(batch_size, cell_count, dtype=dtype, device=device)
    point_error = torch.zeros(batch_size, cell_count, dtype=dtype, device=device)
    visible_mask = torch.zeros(batch_size, cell_count, dtype=torch.bool, device=device)

    sort_key = (points - camera_origin[:, None, :]).square().sum(dim=-1) - radii.square()
    sorted_ids = torch.argsort(sort_key.detach(), dim=1)

    for order in range(cell_count):
        newly_stopped = (~stopped) & (transmittance <= options.transmittance_threshold)
        stop_order = torch.where(newly_stopped, torch.full_like(stop_order, order), stop_order)
        stopped = stopped | newly_stopped

        cell_ids = sorted_ids[:, order]
        center = points[batch_ids, cell_ids]
        radius = radii[batch_ids, cell_ids].clamp_min(options.eps)
        sigma = densities[batch_ids, cell_ids].clamp_min(0.0)
        normal = normals[batch_ids, cell_ids]
        cell_texel_sites = texel_sites[batch_ids, cell_ids]
        cell_texel_rgb = None if texel_rgb is None else texel_rgb[batch_ids, cell_ids]
        cell_texel_height = texel_height[batch_ids, cell_ids]
        cell_texel_sv_axis = None if texel_sv_axis is None else texel_sv_axis[batch_ids, cell_ids]
        cell_texel_sv_rgb = None if texel_sv_rgb is None else texel_sv_rgb[batch_ids, cell_ids]

        oc = origins - center[:, None, :]
        qb = 2.0 * (dirs * oc).sum(dim=-1)
        qc = oc.square().sum(dim=-1) - radius[:, None].square()
        discriminant = qb.square() - 4.0 * qc
        root = torch.sqrt(discriminant.clamp_min(options.eps))
        t_near = (-qb - root) * 0.5
        t_far = (-qb + root) * 0.5
        hit = (discriminant >= 0.0) & (t_far > options.near_plane)
        t_near = t_near.clamp_min(options.near_plane)

        neighbors = adjacency[batch_ids, cell_ids]
        if neighbors.size(1) > 0:
            valid_neighbor = (neighbors >= 0) & (neighbors < cell_count) & (neighbors != cell_ids[:, None])
            safe_neighbor_ids = neighbors.clamp(0, max(cell_count - 1, 0))
            neighbor_center = points[batch_ids[:, None], safe_neighbor_ids]
            neighbor_radius = radii[batch_ids[:, None], safe_neighbor_ids].clamp_min(options.eps)

            face_normal = neighbor_center - center[:, None, :]
            offset = 0.5 * (
                neighbor_center.square().sum(dim=-1)
                - center.square().sum(dim=-1)[:, None]
                + radius.square()[:, None]
                - neighbor_radius.square()
            )
            denom = torch.einsum("bpc,bkc->bkp", dirs, face_normal)
            numer = offset[:, :, None] - torch.einsum("bpc,bkc->bkp", origins, face_normal)
            valid = valid_neighbor[:, :, None]
            parallel = denom.abs() <= options.eps
            hit = hit & (~(parallel & (numer < -options.eps) & valid).any(dim=1))
            t_face = numer / torch.where(parallel, torch.ones_like(denom), denom)

            far_candidates = torch.where((denom > options.eps) & valid, t_face, torch.full_like(t_face, torch.inf))
            near_candidates = torch.where((denom < -options.eps) & valid, t_face, torch.full_like(t_face, -torch.inf))
            t_far = torch.minimum(t_far, far_candidates.amin(dim=1))
            t_near = torch.maximum(t_near, near_candidates.amax(dim=1))
            hit = hit & (t_far > t_near)

        t_surf, dp_surf, color = query_powerfoam_texels(
            origins,
            dirs,
            t_near,
            center,
            radius,
            normal,
            cell_texel_sites,
            cell_texel_rgb,
            cell_texel_height,
            options,
            texel_sv_axis=cell_texel_sv_axis,
            texel_sv_rgb=cell_texel_sv_rgb,
        )
        t_far = torch.minimum(t_far, torch.where(dp_surf >= 0.0, t_surf, t_far))
        t_near = torch.maximum(t_near, torch.where(dp_surf < 0.0, t_surf, t_near))
        hit = hit & (t_far > t_near)

        segment = (t_far - t_near).clamp_min(0.0)
        active = hit & (~stopped) & (sigma[:, None] > 0.0) & (segment > 0.0)
        delta = -sigma[:, None] * segment
        alpha = (1.0 - torch.exp(delta)).clamp(0.0, options.max_alpha)
        if options.alpha_threshold > 0.0:
            alpha = torch.where(alpha >= options.alpha_threshold, alpha, torch.zeros_like(alpha))
        alpha = torch.where(active, alpha, torch.zeros_like(alpha))

        weight = transmittance * alpha
        out = out + weight[:, :, None] * color
        ndv = torch.einsum("bc,bpc->bp", normal, dirs)
        normal_distance = normal_distance + torch.where(ndv > 0.0, ndv.square() * weight, torch.zeros_like(weight))
        normal_out = normal_out + weight[:, :, None] * normal[:, None, :]
        cell_contrib = weight.sum(dim=-1) / float(pixel_count)
        contrib = contrib.scatter_add(1, cell_ids[:, None], cell_contrib[:, None])
        if target_flat is not None:
            cell_point_error = (weight * (color - target_flat).abs().sum(dim=-1)).sum(dim=-1) / float(pixel_count)
            point_error = point_error.scatter_add(1, cell_ids[:, None], cell_point_error[:, None])
        visible_mask[batch_ids, cell_ids] = visible_mask[batch_ids, cell_ids] | active.detach().any(dim=1)
        transmittance = transmittance * (1.0 - alpha)

    background = torch.tensor(options.background, dtype=dtype, device=device)
    out = out + transmittance[:, :, None] * background
    alpha = 1.0 - transmittance
    rendered = out.reshape(batch_size, height, width, 3).permute(0, 3, 1, 2).contiguous()
    return PowerFoamRenderResult(
        rendered=rendered,
        alpha=alpha.reshape(batch_size, height, width),
        transmittance=transmittance.reshape(batch_size, height, width),
        stop_order=stop_order,
        normal_distance=normal_distance.reshape(batch_size, height, width),
        normal=normal_out.reshape(batch_size, height, width, 3).permute(0, 3, 1, 2).contiguous(),
        contrib=contrib,
        point_error=point_error,
        visible_mask=visible_mask,
    )


class DirectPowerFoamVideo(nn.Module):
    def __init__(
        self,
        *,
        frame_count: int,
        cell_count: int,
        render_size: int,
        fov_degrees: float,
        neighbor_count: int,
        xy_extent: float,
        z_min: float,
        z_max: float,
        radius_init: float,
        radius_min: float,
        density_init: float,
        radius_scale: float = 0.75,
        seed: int,
        render_options: PowerFoamRenderOptions,
        init_frames: torch.Tensor | None = None,
        image_init_depth: float | None = None,
        image_init_jitter: float = 0.0,
        num_texel_sites: int = 8,
        sv_dof: int = 8,
        sv_axis_init: float = 8.0,
        adjacency_mode: str = "cech_aabb",
    ) -> None:
        super().__init__()
        if frame_count < 1:
            raise ValueError("frame_count must be positive")
        if cell_count < 1:
            raise ValueError("cell_count must be positive")
        if z_max <= z_min:
            raise ValueError("z_max must be larger than z_min")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))

        if init_frames is None:
            init = initialize_random_full_powerfoam(
                frame_count=frame_count,
                cell_count=cell_count,
                xy_extent=xy_extent,
                z_min=z_min,
                z_max=z_max,
                radius_init=radius_init,
                radius_min=radius_min,
                num_texel_sites=int(num_texel_sites),
                sv_dof=int(sv_dof),
                sv_axis_init=float(sv_axis_init),
                generator=generator,
            )
        else:
            init = initialize_full_powerfoam_from_video(
                init_frames,
                cell_count=cell_count,
                xy_extent=xy_extent,
                z_min=z_min,
                z_max=z_max,
                fov_degrees=fov_degrees,
                image_init_depth=image_init_depth,
                radius_min=radius_min,
                radius_scale=radius_scale,
                num_texel_sites=int(num_texel_sites),
                sv_dof=int(sv_dof),
                sv_axis_init=float(sv_axis_init),
                image_init_jitter=float(image_init_jitter),
                generator=generator,
            )
            if init.points.shape[0] != frame_count:
                raise ValueError("init_frames frame count must match frame_count")
        init_points = init.points
        raw_xy = atanh_clamped(init_points[..., :2] / float(xy_extent))
        raw_z = logit_clamped((init_points[..., 2:] - float(z_min)) / (float(z_max) - float(z_min)))
        self.raw_xy = nn.Parameter(raw_xy)
        self.raw_z = nn.Parameter(raw_z)

        init_density = torch.full((frame_count, cell_count), max(float(density_init), 1.0e-4))
        self.raw_radii = nn.Parameter(
            inverse_softplus(
                (init.radii - float(radius_min)).clamp_min(1.0e-4),
                beta=POWERFOAM_SOFTPLUS_BETA,
            )
        )
        self.raw_densities = nn.Parameter(inverse_softplus(init_density, beta=POWERFOAM_SOFTPLUS_BETA))
        self.raw_quaternions = nn.Parameter(init.quaternions)
        self.raw_texel_sites = nn.Parameter(init.texel_sites)
        self.raw_texel_sv_axis = nn.Parameter(init.texel_sv_axis)
        self.raw_texel_sv_rgb = nn.Parameter(init.texel_sv_rgb)
        self.raw_texel_height = nn.Parameter(init.texel_height)

        self.xy_extent = float(xy_extent)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.radius_min = float(radius_min)
        self.render_options = render_options
        self.neighbor_count = int(neighbor_count)
        self.adjacency_mode = str(adjacency_mode)

        adjacency = build_power_adjacency(init_points, init.radii, neighbor_count=neighbor_count, mode=self.adjacency_mode)
        self.register_buffer("adjacency", adjacency, persistent=True)
        self.register_buffer(
            "ray_dirs",
            make_pinhole_ray_directions(render_size, render_size, fov_degrees, device=torch.device("cpu")),
            persistent=False,
        )

    def decoded_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        xy = torch.tanh(self.raw_xy) * self.xy_extent
        z01 = torch.sigmoid(self.raw_z)
        z = self.z_min + z01 * (self.z_max - self.z_min)
        points = torch.cat([xy, z], dim=-1)
        radii = F.softplus(self.raw_radii, beta=POWERFOAM_SOFTPLUS_BETA) + self.radius_min
        densities = F.softplus(self.raw_densities, beta=POWERFOAM_SOFTPLUS_BETA)
        colors = (self.raw_texel_sv_rgb.mean(dim=(-3, -2)) + 0.5).clamp(0.0, 1.0)
        return points, radii, densities, colors

    def decoded_powerfoam_parameters(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        points, radii, densities, _ = self.decoded_parameters()
        normals, tangents, bitangents = quaternion_frames(self.raw_quaternions, eps=self.render_options.eps)
        texel_offsets = self.raw_texel_sites * radii[:, :, None, None]
        texel_sites = (
            points[:, :, None, :]
            + texel_offsets[..., 0:1] * tangents[:, :, None, :]
            + texel_offsets[..., 1:2] * bitangents[:, :, None, :]
        )
        texel_rgb = spherical_voronoi_texel_color(
            texel_sites.detach(),
            self.raw_texel_sv_axis,
            self.raw_texel_sv_rgb,
            eps=self.render_options.eps,
        )
        texel_height = self.raw_texel_height * radii[:, :, None]
        return points, radii, densities, normals, texel_sites, texel_rgb, texel_height

    def interpenetration(self) -> torch.Tensor:
        points, radii, _, _ = self.decoded_parameters()
        frame_count, cell_count, _ = points.shape
        if self.adjacency.size(-1) == 0:
            return torch.zeros(frame_count, cell_count, dtype=points.dtype, device=points.device)

        frame_ids = torch.arange(frame_count, device=points.device)[:, None, None].expand_as(self.adjacency)
        safe_neighbor_ids = self.adjacency.clamp(0, max(cell_count - 1, 0))
        valid = (self.adjacency >= 0) & (self.adjacency < cell_count)
        neighbor_points = points[frame_ids, safe_neighbor_ids]
        neighbor_radii = radii[frame_ids, safe_neighbor_ids]
        distances = (points[:, :, None, :] - neighbor_points).norm(dim=-1)
        penetration = (radii[:, :, None] + neighbor_radii - distances).clamp_min(0.0)
        return (penetration.square() * valid.to(dtype=points.dtype)).sum(dim=-1)

    @torch.no_grad()
    def rebuild_adjacency(self) -> None:
        points, radii, _, _ = self.decoded_parameters()
        self.adjacency = build_power_adjacency(
            points.detach(),
            radii.detach(),
            neighbor_count=self.neighbor_count,
            mode=self.adjacency_mode,
        ).to(device=self.raw_xy.device)

    def optimizer_param_groups(self, base_lr: float) -> list[dict[str, object]]:
        lr = float(base_lr)
        return [
            {"params": [self.raw_xy, self.raw_z], "lr": 0.05 * lr, "name": "points"},
            {"params": [self.raw_radii], "lr": 0.02 * lr, "name": "radii"},
            {"params": [self.raw_densities], "lr": 0.05 * lr, "name": "density"},
            {"params": [self.raw_quaternions], "lr": 0.05 * lr, "name": "quaternions"},
            {"params": [self.raw_texel_sites], "lr": 0.25 * lr, "name": "texel_sites"},
            {"params": [self.raw_texel_sv_axis], "lr": 0.10 * lr, "name": "texel_sv_axis"},
            {"params": [self.raw_texel_sv_rgb], "lr": lr, "name": "texel_sv_rgb"},
            {"params": [self.raw_texel_height], "lr": 0.25 * lr, "name": "texel_height"},
        ]

    def forward(
        self,
        frame_indices: torch.Tensor,
        target_rgb: torch.Tensor | None = None,
        rays: torch.Tensor | None = None,
    ) -> PowerFoamRenderResult:
        frame_indices = frame_indices.to(device=self.raw_xy.device, dtype=torch.long)
        points, radii, densities, normals, texel_sites, texel_rgb, texel_height = self.decoded_powerfoam_parameters()
        ray_data = self.ray_dirs if rays is None else rays
        return render_powerfoam_torch(
            points[frame_indices],
            radii[frame_indices],
            densities[frame_indices],
            normals[frame_indices],
            texel_sites[frame_indices],
            None,
            texel_height[frame_indices],
            self.adjacency[frame_indices],
            ray_data.to(device=self.raw_xy.device, dtype=points.dtype),
            self.render_options,
            target_rgb=target_rgb,
            texel_sv_axis=self.raw_texel_sv_axis[frame_indices],
            texel_sv_rgb=self.raw_texel_sv_rgb[frame_indices],
        )
