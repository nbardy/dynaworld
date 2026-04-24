from __future__ import annotations

from typing import Sequence

import torch

from camera import CameraSpec
from .common import (
    MIN_RENDER_DEPTH,
    _validate_near_plane,
    project_gaussians_2d,
    project_gaussians_2d_batch,
    transform_world_to_camera,
)


def _camera_scalar_tensor(value, device, dtype):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(float(value), device=device, dtype=dtype)


def _camera_scalar_vector(cameras: Sequence[CameraSpec], field_name: str, device, dtype) -> torch.Tensor:
    values = [getattr(camera, field_name) for camera in cameras]
    if any(torch.is_tensor(value) for value in values):
        return torch.stack(
            [
                value.to(device=device, dtype=dtype)
                if torch.is_tensor(value)
                else torch.tensor(float(value), device=device, dtype=dtype)
                for value in values
            ],
            dim=0,
        ).reshape(-1)
    return torch.tensor([float(value) for value in values], device=device, dtype=dtype)


def _distortion_tensor(camera: CameraSpec, size: int, device, dtype) -> torch.Tensor:
    if camera.distortion is None:
        return torch.zeros(size, device=device, dtype=dtype)
    distortion = torch.as_tensor(camera.distortion, device=device, dtype=dtype).flatten()
    if distortion.numel() > size:
        raise ValueError(
            f"{camera.lens_model} expects at most {size} distortion coefficients, got {distortion.numel()}."
        )
    if distortion.numel() == size:
        return distortion
    padded = torch.zeros(size, device=device, dtype=dtype)
    padded[: distortion.numel()] = distortion
    return padded


def _covariance_from_scales_quats(means3d: torch.Tensor, scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    qr, qi, qj, qk = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    rotation = means3d.new_zeros((means3d.shape[0], 3, 3))
    rotation[:, 0, 0] = 1.0 - 2 * (qj**2 + qk**2)
    rotation[:, 0, 1] = 2 * (qi * qj - qr * qk)
    rotation[:, 0, 2] = 2 * (qi * qk + qr * qj)
    rotation[:, 1, 0] = 2 * (qi * qj + qr * qk)
    rotation[:, 1, 1] = 1.0 - 2 * (qi**2 + qk**2)
    rotation[:, 1, 2] = 2 * (qj * qk - qr * qi)
    rotation[:, 2, 0] = 2 * (qi * qk - qr * qj)
    rotation[:, 2, 1] = 2 * (qj * qk + qr * qi)
    rotation[:, 2, 2] = 1.0 - 2 * (qi**2 + qj**2)

    scale_matrix = means3d.new_zeros((means3d.shape[0], 3, 3))
    scale_matrix[:, 0, 0] = scales[:, 0]
    scale_matrix[:, 1, 1] = scales[:, 1]
    scale_matrix[:, 2, 2] = scales[:, 2]

    gaussian_basis = rotation @ scale_matrix
    return gaussian_basis @ gaussian_basis.transpose(1, 2)


def _normalized_xy_and_jacobian(points_camera: torch.Tensor, near_plane: float):
    x, y, z = points_camera[:, 0], points_camera[:, 1], points_camera[:, 2]
    front_mask_bool = z > near_plane
    z_safe = torch.where(front_mask_bool, torch.clamp(z, min=near_plane), torch.ones_like(z))
    x_project = torch.where(front_mask_bool, x, torch.zeros_like(x))
    y_project = torch.where(front_mask_bool, y, torch.zeros_like(y))
    x_norm = x_project / z_safe
    y_norm = y_project / z_safe

    norm_jacobian = points_camera.new_zeros((points_camera.shape[0], 2, 3))
    norm_jacobian[:, 0, 0] = 1.0 / z_safe
    norm_jacobian[:, 0, 2] = -x_project / (z_safe**2)
    norm_jacobian[:, 1, 1] = 1.0 / z_safe
    norm_jacobian[:, 1, 2] = -y_project / (z_safe**2)
    norm_jacobian = norm_jacobian * front_mask_bool.to(points_camera.dtype).view(-1, 1, 1)
    return x_norm, y_norm, norm_jacobian, front_mask_bool, z


def _radial_tangential_project_normalized(
    x: torch.Tensor,
    y: torch.Tensor,
    coeffs: torch.Tensor,
):
    k1, k2, p1, p2, k3 = coeffs
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

    x_distorted = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_distorted = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    radial_derivative_r2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4
    radial_dx = 2.0 * x * radial_derivative_r2
    radial_dy = 2.0 * y * radial_derivative_r2

    lens_jacobian = x.new_zeros((x.shape[0], 2, 2))
    lens_jacobian[:, 0, 0] = radial + x * radial_dx + 2.0 * p1 * y + 6.0 * p2 * x
    lens_jacobian[:, 0, 1] = x * radial_dy + 2.0 * p1 * x + 2.0 * p2 * y
    lens_jacobian[:, 1, 0] = y * radial_dx + 2.0 * p1 * x + 2.0 * p2 * y
    lens_jacobian[:, 1, 1] = radial + y * radial_dy + 6.0 * p1 * y + 2.0 * p2 * x
    return x_distorted, y_distorted, lens_jacobian


def _opencv_fisheye_project_normalized(
    x: torch.Tensor,
    y: torch.Tensor,
    coeffs: torch.Tensor,
):
    k1, k2, k3, k4 = coeffs
    eps = 1.0e-8
    r2 = x * x + y * y
    radius = torch.sqrt(torch.clamp(r2, min=eps * eps))
    theta = torch.atan(radius)
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4
    theta_distorted = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)

    valid_radius = r2 > eps * eps
    scale_raw = theta_distorted / radius
    scale = torch.where(valid_radius, scale_raw, torch.ones_like(scale_raw))
    x_distorted = scale * x
    y_distorted = scale * y

    dtheta_distorted_dtheta = (
        1.0
        + 3.0 * k1 * theta2
        + 5.0 * k2 * theta4
        + 7.0 * k3 * theta6
        + 9.0 * k4 * theta8
    )
    dtheta_dr = 1.0 / (1.0 + radius * radius)
    dscale_dr_raw = (dtheta_distorted_dtheta * dtheta_dr * radius - theta_distorted) / (radius * radius)
    dscale_dr = torch.where(valid_radius, dscale_dr_raw, torch.zeros_like(dscale_dr_raw))

    inv_radius = torch.where(valid_radius, 1.0 / radius, torch.zeros_like(radius))
    ds_dx = dscale_dr * x * inv_radius
    ds_dy = dscale_dr * y * inv_radius

    lens_jacobian = x.new_zeros((x.shape[0], 2, 2))
    lens_jacobian[:, 0, 0] = scale + x * ds_dx
    lens_jacobian[:, 0, 1] = x * ds_dy
    lens_jacobian[:, 1, 0] = y * ds_dx
    lens_jacobian[:, 1, 1] = scale + y * ds_dy
    return x_distorted, y_distorted, lens_jacobian


def project_points_camera(
    points_camera: torch.Tensor,
    camera: CameraSpec,
    *,
    near_plane: float = MIN_RENDER_DEPTH,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project camera-frame points through a central lens model.

    Returns:
        pixels: [N, 2]
        depths: [N]
        pixel_jacobian: [N, 2, 3], d(pixel)/d(camera_xyz)
        front_mask: [N] boolean
    """
    near_plane = _validate_near_plane(near_plane)
    device = points_camera.device
    dtype = points_camera.dtype
    fx = _camera_scalar_tensor(camera.fx, device=device, dtype=dtype)
    fy = _camera_scalar_tensor(camera.fy, device=device, dtype=dtype)
    cx = _camera_scalar_tensor(camera.cx, device=device, dtype=dtype)
    cy = _camera_scalar_tensor(camera.cy, device=device, dtype=dtype)

    x_norm, y_norm, norm_jacobian, front_mask_bool, depths = _normalized_xy_and_jacobian(
        points_camera,
        near_plane=near_plane,
    )

    if camera.lens_model == "radial_tangential":
        coeffs = _distortion_tensor(camera, 5, device=device, dtype=dtype)
        x_distorted, y_distorted, lens_jacobian = _radial_tangential_project_normalized(x_norm, y_norm, coeffs)
    elif camera.lens_model == "opencv_fisheye":
        coeffs = _distortion_tensor(camera, 4, device=device, dtype=dtype)
        x_distorted, y_distorted, lens_jacobian = _opencv_fisheye_project_normalized(x_norm, y_norm, coeffs)
    elif camera.lens_model == "pinhole":
        x_distorted = x_norm
        y_distorted = y_norm
        lens_jacobian = points_camera.new_zeros((points_camera.shape[0], 2, 2))
        lens_jacobian[:, 0, 0] = 1.0
        lens_jacobian[:, 1, 1] = 1.0
    else:
        raise ValueError(f"Unknown lens_model: {camera.lens_model}")

    intrinsics_jacobian = points_camera.new_zeros((points_camera.shape[0], 2, 2))
    intrinsics_jacobian[:, 0, 0] = fx
    intrinsics_jacobian[:, 1, 1] = fy
    pixel_jacobian = intrinsics_jacobian @ lens_jacobian @ norm_jacobian

    pixels = points_camera.new_zeros((points_camera.shape[0], 2))
    pixels[:, 0] = fx * x_distorted + cx
    pixels[:, 1] = fy * y_distorted + cy
    return pixels, depths, pixel_jacobian, front_mask_bool


def _invert_cov2d(cov2d: torch.Tensor) -> torch.Tensor:
    det = torch.clamp(cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0], min=1e-6)
    inv_cov2d = torch.zeros_like(cov2d)
    inv_cov2d[:, 0, 0] = cov2d[:, 1, 1] / det
    inv_cov2d[:, 1, 1] = cov2d[:, 0, 0] / det
    inv_cov2d[:, 0, 1] = -cov2d[:, 0, 1] / det
    inv_cov2d[:, 1, 0] = -cov2d[:, 1, 0] / det
    return inv_cov2d


def project_gaussians_2d_camera(
    means3d: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    rgbs: torch.Tensor,
    camera: CameraSpec,
    *,
    near_plane: float = MIN_RENDER_DEPTH,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project 3D Gaussians with the lens model stored in ``CameraSpec``."""
    if camera.lens_model == "pinhole":
        return project_gaussians_2d(
            means3d,
            scales,
            quats,
            opacities,
            rgbs,
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            camera_to_world=camera.camera_to_world,
            near_plane=near_plane,
        )

    near_plane = _validate_near_plane(near_plane)
    cov3d = _covariance_from_scales_quats(means3d, scales, quats)
    means_camera, cov_camera = transform_world_to_camera(means3d, cov3d, camera.camera_to_world)

    z = means_camera[:, 2]
    sorted_idx = torch.argsort(z, descending=False, stable=True)
    means_camera = means_camera[sorted_idx]
    cov_camera = cov_camera[sorted_idx]
    opacities = opacities[sorted_idx]
    rgbs = rgbs[sorted_idx]

    means2d, _depths, jacobian, front_mask_bool = project_points_camera(
        means_camera,
        camera,
        near_plane=near_plane,
    )
    front_mask = front_mask_bool.to(opacities.dtype).unsqueeze(-1)
    opacities = opacities * front_mask

    cov2d = jacobian @ cov_camera @ jacobian.transpose(1, 2)
    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3
    inv_cov2d = _invert_cov2d(cov2d)
    return means2d, inv_cov2d, cov2d, opacities, rgbs


def _all_pinhole(cameras: Sequence[CameraSpec]) -> bool:
    return all(camera.lens_model == "pinhole" for camera in cameras)


def project_gaussians_2d_camera_batch(
    means3d: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    rgbs: torch.Tensor,
    cameras: Sequence[CameraSpec],
    *,
    near_plane: float = MIN_RENDER_DEPTH,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch projection through CameraSpec lenses.

    The all-pinhole path delegates to the legacy vectorized implementation for
    parity. Mixed or non-pinhole batches use the single-camera path per frame.
    """
    if means3d.shape[0] != len(cameras):
        raise ValueError(f"Expected {means3d.shape[0]} cameras, got {len(cameras)}.")

    if _all_pinhole(cameras):
        device = means3d.device
        dtype = means3d.dtype
        return project_gaussians_2d_batch(
            means3d,
            scales,
            quats,
            opacities,
            rgbs,
            _camera_scalar_vector(cameras, "fx", device, dtype),
            _camera_scalar_vector(cameras, "fy", device, dtype),
            _camera_scalar_vector(cameras, "cx", device, dtype),
            _camera_scalar_vector(cameras, "cy", device, dtype),
            camera_to_world=torch.stack(
                [camera.camera_to_world.to(device=device, dtype=dtype) for camera in cameras],
                dim=0,
            ),
            near_plane=near_plane,
        )

    projected = [
        project_gaussians_2d_camera(
            means3d[index],
            scales[index],
            quats[index],
            opacities[index],
            rgbs[index],
            cameras[index],
            near_plane=near_plane,
        )
        for index in range(means3d.shape[0])
    ]
    return tuple(torch.stack(items, dim=0) for items in zip(*projected))


__all__ = [
    "project_gaussians_2d_camera",
    "project_gaussians_2d_camera_batch",
    "project_points_camera",
]
