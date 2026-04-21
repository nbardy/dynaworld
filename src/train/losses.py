from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _mean_per_image(values: torch.Tensor) -> torch.Tensor:
    return values.flatten(1).mean(dim=1)


def _local_mean(images: torch.Tensor, window_size: int) -> torch.Tensor:
    if window_size <= 1:
        return images
    pad = window_size // 2
    padded = F.pad(images, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=window_size, stride=1)


def ssim_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int,
    c1: float,
    c2: float,
) -> torch.Tensor:
    prediction = prediction.float()
    target = target.float()
    mu_x = _local_mean(prediction, window_size)
    mu_y = _local_mean(target, window_size)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y

    sigma_x_sq = _local_mean(prediction.square(), window_size) - mu_x_sq
    sigma_y_sq = _local_mean(target.square(), window_size) - mu_y_sq
    sigma_xy = _local_mean(prediction * target, window_size) - mu_xy

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / denominator.clamp_min(1.0e-12)
    return _mean_per_image(ssim_map).clamp(-1.0, 1.0)


def dssim_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int,
    c1: float,
    c2: float,
) -> torch.Tensor:
    return (1.0 - ssim_per_image(prediction, target, window_size=window_size, c1=c1, c2=c2)) * 0.5


def reconstruction_loss_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> torch.Tensor:
    loss_type = loss_cfg["type"]
    delta = prediction - target

    if loss_type == "mse":
        return _mean_per_image(delta.square())
    if loss_type == "l1":
        return _mean_per_image(delta.abs())
    if loss_type == "l1_mse":
        l1 = _mean_per_image(delta.abs())
        mse = _mean_per_image(delta.square())
        return float(loss_cfg["l1_weight"]) * l1 + float(loss_cfg["mse_weight"]) * mse
    if loss_type == "standard_gs":
        l1 = _mean_per_image(delta.abs())
        dssim = dssim_per_image(
            prediction,
            target,
            window_size=int(loss_cfg["ssim_window_size"]),
            c1=float(loss_cfg["ssim_c1"]),
            c2=float(loss_cfg["ssim_c2"]),
        )
        return float(loss_cfg["l1_weight"]) * l1 + float(loss_cfg["dssim_weight"]) * dssim

    raise ValueError(f"Unknown reconstruction loss type: {loss_type!r}")


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> torch.Tensor:
    return reconstruction_loss_per_image(prediction, target, loss_cfg).mean()
