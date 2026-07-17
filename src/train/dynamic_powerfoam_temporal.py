from __future__ import annotations

import math

import torch


def make_gaussian_time_basis(frame_count: int, basis_count: int, sigma_scale: float) -> torch.Tensor:
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    times = torch.linspace(0.0, 1.0, frame_count, dtype=torch.float32)
    centers = torch.linspace(0.0, 1.0, basis_count, dtype=torch.float32)
    spacing = 1.0 / float(max(basis_count - 1, 1))
    sigma = max(spacing * float(sigma_scale), 1.0e-4)
    basis = torch.exp(-0.5 * ((times[:, None] - centers[None, :]) / sigma).square())
    return basis / basis.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)


def fit_temporal_basis(values: torch.Tensor, basis: torch.Tensor, *, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    base = values.mean(dim=0)
    coeff = torch.zeros((basis.shape[1], *values.shape[1:]), dtype=values.dtype)
    if mode == "fit" and values.shape[0] > 1:
        residual = (values - base).reshape(values.shape[0], -1)
        solution = torch.linalg.pinv(basis).to(residual.dtype) @ residual
        coeff = solution.reshape(basis.shape[1], *values.shape[1:]).contiguous()
    return base.contiguous(), coeff.contiguous()


def temporal_accel(values: torch.Tensor) -> torch.Tensor:
    if values.shape[0] < 3:
        return values.new_zeros(())
    return (values[2:] - 2.0 * values[1:-1] + values[:-2]).square().mean()


def atanh_clamped(values: torch.Tensor) -> torch.Tensor:
    return torch.atanh(values.clamp(-0.9999, 0.9999))


def temporal_motion_metrics(
    points: torch.Tensor,
    features: torch.Tensor,
    *,
    render_size: int,
    fov_degrees: float,
    camera_to_world: torch.Tensor | None = None,
) -> dict[str, float]:
    if points.shape[0] < 2:
        return {
            "state_mean_temporal_xy_delta": 0.0,
            "state_p95_temporal_xy_delta": 0.0,
            "state_mean_temporal_z_delta": 0.0,
            "state_mean_temporal_screen_delta_px": 0.0,
            "state_p95_temporal_screen_delta_px": 0.0,
            "state_temporal_screen_valid_fraction": 0.0,
            "state_mean_temporal_feature_abs_delta": 0.0,
        }
    dxy = torch.linalg.vector_norm(points[1:, :, :2] - points[:-1, :, :2], dim=-1)
    dz = (points[1:, :, 2] - points[:-1, :, 2]).abs()
    screen_points = points
    if camera_to_world is not None:
        world_to_camera = torch.linalg.inv(camera_to_world.to(device=points.device, dtype=points.dtype))
        screen_points = torch.bmm(points, world_to_camera[:, :3, :3].transpose(1, 2)) + world_to_camera[:, None, :3, 3]
    tan_half_fov = math.tan(math.radians(float(fov_degrees)) * 0.5)
    z = screen_points[..., 2]
    screen = torch.stack(
        [
            0.5 * (screen_points[..., 0] / (z.clamp_min(1.0e-6) * tan_half_fov) + 1.0) * float(int(render_size) - 1),
            0.5 * (-screen_points[..., 1] / (z.clamp_min(1.0e-6) * tan_half_fov) + 1.0) * float(int(render_size) - 1),
        ],
        dim=-1,
    )
    dscreen = torch.linalg.vector_norm(screen[1:] - screen[:-1], dim=-1)
    valid_screen = (z[1:] > 1.0e-4) & (z[:-1] > 1.0e-4)
    valid_dscreen = dscreen[valid_screen]
    if valid_dscreen.numel() == 0:
        mean_screen_delta = points.new_zeros(())
        p95_screen_delta = points.new_zeros(())
    else:
        mean_screen_delta = valid_dscreen.mean()
        p95_screen_delta = valid_dscreen.flatten().quantile(0.95)
    feature_delta = (features[1:] - features[:-1]).abs()
    return {
        "state_mean_temporal_xy_delta": float(dxy.mean().cpu()),
        "state_p95_temporal_xy_delta": float(dxy.flatten().quantile(0.95).cpu()),
        "state_mean_temporal_z_delta": float(dz.mean().cpu()),
        "state_mean_temporal_screen_delta_px": float(mean_screen_delta.cpu()),
        "state_p95_temporal_screen_delta_px": float(p95_screen_delta.cpu()),
        "state_temporal_screen_valid_fraction": float(valid_screen.float().mean().cpu()),
        "state_mean_temporal_feature_abs_delta": float(feature_delta.mean().cpu()),
    }
