from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from camera import CameraSpec, build_camera_rays


def make_pinhole_rays(height: int, width: int, fov_degrees: float, device: torch.device) -> torch.Tensor:
    half_y = math.tan(math.radians(float(fov_degrees)) * 0.5)
    half_x = half_y * (float(width) / float(height))
    ys = torch.linspace(half_y, -half_y, height, device=device, dtype=torch.float32)
    xs = torch.linspace(-half_x, half_x, width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dirs = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
    dirs = F.normalize(dirs, dim=-1)
    origins = torch.zeros_like(dirs)
    return torch.cat([origins, dirs], dim=-1).unsqueeze(0).contiguous()


def powerfoam_rays_from_camera(
    camera: CameraSpec,
    *,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    origins, directions = build_camera_rays(camera, height, width, device=device, dtype=dtype)
    return torch.cat([origins, directions], dim=-1).unsqueeze(0).contiguous()


def powerfoam_rays_from_camera_grid(
    cameras: tuple[tuple[CameraSpec, ...], ...],
    *,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not cameras:
        raise ValueError("Expected at least one camera view.")
    per_view = []
    for view_cameras in cameras:
        if not view_cameras:
            raise ValueError("Expected at least one frame camera per view.")
        per_view.append(
            torch.cat(
                [
                    powerfoam_rays_from_camera(
                        camera,
                        height=height,
                        width=width,
                        device=device,
                        dtype=dtype,
                    )
                    for camera in view_cameras
                ],
                dim=0,
            )
        )
    return torch.stack(per_view, dim=0).contiguous()


def stable_tangent_from_normals(normals: torch.Tensor) -> torch.Tensor:
    z_axis = normals.new_tensor([0.0, 0.0, 1.0]).expand_as(normals)
    y_axis = normals.new_tensor([0.0, 1.0, 0.0]).expand_as(normals)
    helper = torch.where(normals[..., 2:3].abs() < 0.9, z_axis, y_axis)
    return F.normalize(torch.cross(helper, normals, dim=-1), dim=-1, eps=1.0e-6)


def orthonormal_surface_frame(normals: torch.Tensor, raw_tangents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tangents = raw_tangents - (raw_tangents * normals).sum(dim=-1, keepdim=True) * normals
    fallback = stable_tangent_from_normals(normals)
    tangent_norm = tangents.norm(dim=-1, keepdim=True)
    tangents = torch.where(tangent_norm > 1.0e-6, tangents / tangent_norm.clamp_min(1.0e-6), fallback)
    bitangents = F.normalize(torch.cross(normals, tangents, dim=-1), dim=-1, eps=1.0e-6)
    return tangents, bitangents


__all__ = [
    "make_pinhole_rays",
    "orthonormal_surface_frame",
    "powerfoam_rays_from_camera",
    "powerfoam_rays_from_camera_grid",
    "stable_tangent_from_normals",
]
