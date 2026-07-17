from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from powerfoam_direct import (
    PowerFoamInitialization,
    camera_facing_quaternion,
    estimate_knn_radii,
    logit_clamped,
    make_image_init_uv,
)
from powerfoam_implicit_camera import PowerFoamImplicitCameraDecoder


def transform_powerfoam_frame_to_camera(
    points: torch.Tensor,
    normals: torch.Tensor,
    tangents: torch.Tensor,
    bitangents: torch.Tensor,
    camera_to_world: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    world_to_camera = torch.linalg.inv(camera_to_world.to(device=points.device, dtype=points.dtype))
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    points_camera = points @ rotation.T + translation
    normals_camera = F.normalize(normals @ rotation.T, dim=-1, eps=1.0e-6)
    tangents_camera = F.normalize(tangents @ rotation.T, dim=-1, eps=1.0e-6)
    bitangents_camera = F.normalize(bitangents @ rotation.T, dim=-1, eps=1.0e-6)
    return points_camera, normals_camera, tangents_camera, bitangents_camera


def transform_points_camera_to_world(points_camera: torch.Tensor, camera_to_world: torch.Tensor) -> torch.Tensor:
    rotation = camera_to_world[:, :3, :3]
    translation = camera_to_world[:, :3, 3]
    return torch.bmm(rotation, points_camera.unsqueeze(-1)).squeeze(-1) + translation


def make_texel_feature_init(
    texel_rgb: torch.Tensor,
    *,
    feature_dim: int,
    rgb_init: str,
    noise_std: float,
    generator: torch.Generator,
) -> torch.Tensor:
    features = float(noise_std) * torch.randn(
        (*texel_rgb.shape[:-1], int(feature_dim)),
        generator=generator,
        dtype=texel_rgb.dtype,
    )
    if rgb_init == "logit":
        features[..., :3] = logit_clamped(texel_rgb.clamp(0.0, 1.0))
    elif rgb_init == "rgb":
        features[..., :3] = texel_rgb.clamp(0.0, 1.0)
    return features.contiguous()


def initialize_powerfoam_normals(
    *,
    frame_count: int,
    cell_count: int,
    dtype: torch.dtype,
    normal_init_jitter: float,
    video_init_mode: str,
    camera_decoder: PowerFoamImplicitCameraDecoder | None,
    generator: torch.Generator,
) -> torch.Tensor:
    if str(video_init_mode) == "orbit_camera":
        if camera_decoder is None:
            raise ValueError("model.video_init_mode='orbit_camera' requires camera.enabled=true")
        source_frame = torch.remainder(torch.arange(int(cell_count), dtype=torch.long), int(frame_count))
        c2w = camera_decoder.base_camera_to_world_matrices(
            torch.arange(int(frame_count)),
            device=torch.device("cpu"),
            dtype=dtype,
        )
        camera_facing = torch.tensor([0.0, 0.0, -1.0], dtype=dtype)
        init_normals = torch.matmul(c2w[source_frame, :3, :3], camera_facing)
        init_normals = init_normals.unsqueeze(0).repeat(int(frame_count), 1, 1).contiguous()
    else:
        init_normals = torch.zeros(int(frame_count), int(cell_count), 3, dtype=dtype)
        init_normals[..., 2] = -1.0
    if float(normal_init_jitter) != 0.0:
        init_normals = init_normals + float(normal_init_jitter) * torch.randn(
            int(frame_count),
            int(cell_count),
            3,
            generator=generator,
            dtype=dtype,
        )
    return F.normalize(init_normals, dim=-1, eps=1.0e-6)


def initialize_full_powerfoam_from_orbit_video(
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
    image_init_jitter: float,
    texel_site_scale: float,
    camera_decoder: PowerFoamImplicitCameraDecoder,
    generator: torch.Generator,
) -> PowerFoamInitialization:
    if init_frames.dim() != 4 or init_frames.size(1) != 3:
        raise ValueError("init_frames must be [T,3,H,W]")
    frame_count, _channels, height, width = init_frames.shape
    x01, y01, _rows, _cols = make_image_init_uv(
        cell_count,
        jitter_fraction=float(image_init_jitter),
        generator=generator,
    )
    source_frame = torch.remainder(torch.arange(cell_count, dtype=torch.long), int(frame_count))
    depth = float(camera_decoder.base_radius) if image_init_depth is None else float(image_init_depth)
    tan_half_fov = math.tan(math.radians(float(fov_degrees)) * 0.5)
    dirs = torch.stack(
        [
            (2.0 * x01 - 1.0) * tan_half_fov,
            -(2.0 * y01 - 1.0) * tan_half_fov * (float(height) / float(width)),
            torch.ones_like(x01),
        ],
        dim=-1,
    )
    dirs = F.normalize(dirs, dim=-1)
    points_camera = dirs * depth
    c2w = camera_decoder.base_camera_to_world_matrices(
        torch.arange(int(frame_count)),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    points = transform_points_camera_to_world(points_camera, c2w[source_frame])
    points = points.clone()
    points[:, :2] = points[:, :2].clamp(-0.95 * float(xy_extent), 0.95 * float(xy_extent))
    points[:, 2] = points[:, 2].clamp(float(z_min) + 1.0e-4, float(z_max) - 1.0e-4)
    points = points.unsqueeze(0).repeat(int(frame_count), 1, 1).contiguous()

    sample_grid = torch.stack([2.0 * x01 - 1.0, 2.0 * y01 - 1.0], dim=-1).view(1, cell_count, 1, 2)
    sampled_colors = F.grid_sample(
        init_frames.detach().cpu().float().clamp(0.0, 1.0),
        sample_grid.repeat(int(frame_count), 1, 1, 1),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).squeeze(-1).permute(0, 2, 1)
    cell_colors = sampled_colors[source_frame, torch.arange(cell_count)]

    radii = estimate_knn_radii(points, radius_scale=float(radius_scale), radius_min=float(radius_min))
    quaternions = camera_facing_quaternion(int(frame_count), int(cell_count))
    site_cols = math.ceil(math.sqrt(float(num_texel_sites)))
    site_rows = math.ceil(float(num_texel_sites) / float(site_cols))
    site_x = (torch.arange(site_cols, dtype=torch.float32) + 0.5) / float(site_cols) - 0.5
    site_y = (torch.arange(site_rows, dtype=torch.float32) + 0.5) / float(site_rows) - 0.5
    sy, sx = torch.meshgrid(site_y, site_x, indexing="ij")
    site_offsets = torch.stack([sx.reshape(-1), sy.reshape(-1)], dim=-1)[: int(num_texel_sites)]
    texel_sites = (0.5 * float(texel_site_scale) * site_offsets).view(1, 1, int(num_texel_sites), 2)
    texel_sites = texel_sites.repeat(int(frame_count), int(cell_count), 1, 1).contiguous()
    texel_sv_axis = torch.zeros(int(frame_count), int(cell_count), int(num_texel_sites), 1, 3, dtype=torch.float32)
    texel_sv_axis[..., 2] = 1.0
    texel_sv_rgb = (cell_colors.view(1, cell_count, 1, 1, 3).repeat(int(frame_count), 1, int(num_texel_sites), 1, 1) - 0.5)
    texel_height = torch.zeros(int(frame_count), int(cell_count), int(num_texel_sites), dtype=torch.float32)
    return PowerFoamInitialization(
        points=points,
        radii=radii,
        quaternions=quaternions,
        texel_sites=texel_sites,
        texel_sv_axis=texel_sv_axis,
        texel_sv_rgb=texel_sv_rgb.contiguous(),
        texel_height=texel_height,
    )
