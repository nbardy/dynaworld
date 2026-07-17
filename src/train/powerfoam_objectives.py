from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from losses import ssim_per_image
from powerfoam_training import exp_scheduled_weight


def powerfoam_ssim_loss(rendered: torch.Tensor, target: torch.Tensor, loss_cfg: dict[str, Any]) -> torch.Tensor:
    window_size = min(int(loss_cfg["ssim_window_size"]), int(rendered.shape[-1]), int(rendered.shape[-2]))
    if window_size % 2 == 0:
        window_size -= 1
    window_size = max(window_size, 1)
    return 1.0 - ssim_per_image(
        rendered,
        target,
        window_size=window_size,
        c1=float(loss_cfg["ssim_c1"]),
        c2=float(loss_cfg["ssim_c2"]),
    ).mean()


def scheduled_loss_weights(loss_cfg: dict[str, Any], step: int, total_steps: int) -> dict[str, float]:
    def aux_weight(name: str) -> float:
        if int(step) < int(loss_cfg[f"{name}_weight_start_step"]):
            return 0.0
        return exp_scheduled_weight(
            float(loss_cfg[f"{name}_weight"]),
            float(loss_cfg[f"{name}_weight_final_multiplier"]),
            int(step) - int(loss_cfg[f"{name}_weight_start_step"]),
            max(int(total_steps) - int(loss_cfg[f"{name}_weight_start_step"]), 1),
        )

    weights = {
        "l1_weight": float(loss_cfg["l1_weight"]),
        "mse_weight": float(loss_cfg["mse_weight"]),
        "ssim_weight": float(loss_cfg["ssim_weight"]),
        "radius_l2_weight": float(loss_cfg["radius_l2_weight"]),
        "density_l2_weight": float(loss_cfg["density_l2_weight"]),
        "normal_weight": aux_weight("normal"),
        "normal_map_weight": aux_weight("normal_map"),
        "contribution_weight": aux_weight("contribution"),
        "interpenetration_weight": aux_weight("interpenetration"),
    }
    if "rgb_mse_sum_weight" in loss_cfg:
        weights["rgb_mse_sum_weight"] = float(loss_cfg["rgb_mse_sum_weight"])
    return weights


def direct_powerfoam_loss(
    model: Any,
    rendered: torch.Tensor,
    target: torch.Tensor,
    render_result: Any,
    loss_cfg: dict[str, Any],
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    l1 = F.l1_loss(rendered, target)
    mse = F.mse_loss(rendered, target)
    rgb_mse_sum = (rendered - target).square().sum(dim=1).mean()
    ssim_loss = 1.0 - ssim_per_image(
        rendered,
        target,
        window_size=int(loss_cfg["ssim_window_size"]),
        c1=float(loss_cfg["ssim_c1"]),
        c2=float(loss_cfg["ssim_c2"]),
    ).mean()
    normal_loss = render_result.normal_distance.mean()
    contribution_loss = render_result.contrib.sum(dim=1).mean()
    interpenetration_loss = model.interpenetration().sum(dim=1).mean()
    _, radii, densities, _ = model.decoded_parameters()
    radius_l2 = radii.square().mean()
    density_l2 = densities.square().mean()

    terms = {
        "l1": l1,
        "mse": mse,
        "rgb_mse_sum": rgb_mse_sum,
        "ssim": ssim_loss,
        "normal": normal_loss,
        "contribution": contribution_loss,
        "interpenetration": interpenetration_loss,
        "radius_l2": radius_l2,
        "density_l2": density_l2,
    }
    loss = (
        weights["l1_weight"] * l1
        + weights["mse_weight"] * mse
        + weights["rgb_mse_sum_weight"] * rgb_mse_sum
        + weights["ssim_weight"] * ssim_loss
        + weights["normal_weight"] * normal_loss
        + weights["contribution_weight"] * contribution_loss
        + weights["interpenetration_weight"] * interpenetration_loss
        + weights["radius_l2_weight"] * radius_l2
        + weights["density_l2_weight"] * density_l2
    )
    return loss, terms


def powerfoam_contribution_loss(alpha: torch.Tensor) -> torch.Tensor:
    if alpha.ndim != 3:
        raise ValueError(f"alpha must have shape [B,H,W], got {tuple(alpha.shape)}.")
    return alpha.mean()


def powerfoam_normal_distance_loss(normal_distance: torch.Tensor) -> torch.Tensor:
    if normal_distance.ndim != 3:
        raise ValueError(f"normal_distance must have shape [B,H,W], got {tuple(normal_distance.shape)}.")
    return normal_distance.mean()


def expand_powerfoam_rays_to_batch(rays: torch.Tensor, batch_size: int) -> torch.Tensor:
    if rays.ndim != 4 or rays.shape[-1] != 6:
        raise ValueError(f"rays must have shape [B,H,W,6], got {tuple(rays.shape)}.")
    if rays.shape[0] == int(batch_size):
        return rays.contiguous()
    if rays.shape[0] == 1:
        return rays.expand(int(batch_size), -1, -1, -1).contiguous()
    raise ValueError(f"Expected {batch_size} ray batches or one shared ray batch, got {rays.shape[0]}.")


def normals_from_ray_depth(depth: torch.Tensor, rays: torch.Tensor, *, eps: float = 1.0e-6) -> tuple[torch.Tensor, torch.Tensor]:
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"depth must have shape [B,H,W], got {tuple(depth.shape)}.")
    rays = expand_powerfoam_rays_to_batch(rays.to(device=depth.device, dtype=depth.dtype), int(depth.shape[0]))
    if tuple(rays.shape[:3]) != tuple(depth.shape):
        raise ValueError(f"ray/depth shape mismatch: {tuple(rays.shape[:3])} vs {tuple(depth.shape)}.")
    origins = rays[..., :3]
    raw_dirs = rays[..., 3:]
    dir_norm = torch.linalg.vector_norm(raw_dirs, dim=-1, keepdim=True)
    dirs = raw_dirs / dir_norm.clamp_min(float(eps))
    valid = (
        torch.isfinite(depth)
        & (depth > 0.0)
        & torch.isfinite(origins).all(dim=-1)
        & torch.isfinite(raw_dirs).all(dim=-1)
        & (dir_norm[..., 0] > float(eps))
    )
    points = origins + depth[..., None] * dirs
    normals = torch.zeros_like(points)
    mask = torch.zeros_like(depth, dtype=torch.bool)
    if depth.shape[1] < 3 or depth.shape[2] < 3:
        return normals, mask
    dx = points[:, 1:-1, 2:, :] - points[:, 1:-1, :-2, :]
    dy = points[:, 2:, 1:-1, :] - points[:, :-2, 1:-1, :]
    inner = torch.cross(dy, dx, dim=-1)
    inner_norm = torch.linalg.vector_norm(inner, dim=-1, keepdim=True)
    inner_normal = inner / inner_norm.clamp_min(float(eps))
    inner_dirs = dirs[:, 1:-1, 1:-1, :]
    inner_normal = torch.where(
        (inner_normal * inner_dirs).sum(dim=-1, keepdim=True) > 0.0,
        -inner_normal,
        inner_normal,
    )
    inner_mask = (
        valid[:, 1:-1, 1:-1]
        & valid[:, 1:-1, 2:]
        & valid[:, 1:-1, :-2]
        & valid[:, 2:, 1:-1]
        & valid[:, :-2, 1:-1]
        & (inner_norm[..., 0] > float(eps))
        & torch.isfinite(inner_normal).all(dim=-1)
    )
    normals[:, 1:-1, 1:-1, :] = torch.where(inner_mask[..., None], inner_normal, 0.0)
    mask[:, 1:-1, 1:-1] = inner_mask
    return normals, mask


def powerfoam_normal_map_loss(
    rendered_normal: torch.Tensor,
    target_normal: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if rendered_normal.ndim != 4 or rendered_normal.shape[-1] != 3:
        raise ValueError(f"rendered_normal must have shape [B,H,W,3], got {tuple(rendered_normal.shape)}.")
    if target_normal.shape != rendered_normal.shape:
        raise ValueError(f"target_normal shape mismatch: {tuple(target_normal.shape)} vs {tuple(rendered_normal.shape)}.")
    if valid_mask.shape != rendered_normal.shape[:3]:
        raise ValueError(f"valid_mask shape mismatch: {tuple(valid_mask.shape)} vs {tuple(rendered_normal.shape[:3])}.")
    mask = valid_mask.to(device=rendered_normal.device, dtype=rendered_normal.dtype)
    per_pixel = (
        rendered_normal - target_normal.to(device=rendered_normal.device, dtype=rendered_normal.dtype)
    ).square().sum(dim=-1)
    return (per_pixel * mask).sum() / mask.sum().clamp_min(1.0)


def fixed_background_tensor(rendered: torch.Tensor, render_cfg: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(
        render_cfg["background"],
        device=rendered.device,
        dtype=rendered.dtype,
    ).view(1, 3, 1, 1)


def training_background_tensor(rendered: torch.Tensor, render_cfg: dict[str, Any]) -> torch.Tensor:
    if str(render_cfg["background_mode"]) == "random":
        return torch.rand((rendered.shape[0], 3, 1, 1), device=rendered.device, dtype=rendered.dtype)
    return fixed_background_tensor(rendered, render_cfg)


def composite_powerfoam_background(
    rendered: torch.Tensor,
    alpha: torch.Tensor,
    background: torch.Tensor,
) -> torch.Tensor:
    if alpha.ndim != 3:
        raise ValueError(f"alpha must have shape [B,H,W], got {tuple(alpha.shape)}.")
    if rendered.ndim != 4 or rendered.shape[1] != 3:
        raise ValueError(f"rendered must have shape [B,3,H,W], got {tuple(rendered.shape)}.")
    return rendered + (1.0 - alpha).unsqueeze(1) * background


def composite_fixed_background(rendered: torch.Tensor, alpha: torch.Tensor, render_cfg: dict[str, Any]) -> torch.Tensor:
    return composite_powerfoam_background(rendered, alpha, fixed_background_tensor(rendered, render_cfg))


__all__ = [
    "composite_fixed_background",
    "composite_powerfoam_background",
    "direct_powerfoam_loss",
    "expand_powerfoam_rays_to_batch",
    "fixed_background_tensor",
    "normals_from_ray_depth",
    "powerfoam_contribution_loss",
    "powerfoam_normal_distance_loss",
    "powerfoam_normal_map_loss",
    "powerfoam_ssim_loss",
    "scheduled_loss_weights",
    "training_background_tensor",
]
