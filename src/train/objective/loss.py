from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from .types import ReconstructionLossSpec, RenderedView, TargetView, ViewLoss
from .background import background_spec_from_mapping
from .types import ObjectiveSpec


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


def reconstruction_loss_spec_from_mapping(values: Mapping[str, Any]) -> ReconstructionLossSpec:
    """Boundary helper for legacy loss dictionaries.

    Hot-path objective code consumes `ReconstructionLossSpec`; raw dict reads
    stay at this adapter boundary.
    """

    kind = str(values["type"])
    allowed = {"mse", "l1", "l1_mse", "standard_gs"}
    if kind not in allowed:
        raise ValueError(f"Unknown reconstruction loss type: {kind!r}")
    return ReconstructionLossSpec(
        kind=kind,  # type: ignore[arg-type]
        l1_weight=float(values["l1_weight"]) if "l1_weight" in values else 0.8,
        dssim_weight=float(values["dssim_weight"]) if "dssim_weight" in values else 0.2,
        mse_weight=float(values["mse_weight"]) if "mse_weight" in values else 0.0,
        ssim_window_size=int(values["ssim_window_size"]) if "ssim_window_size" in values else 11,
        ssim_c1=float(values["ssim_c1"]) if "ssim_c1" in values else 0.01**2,
        ssim_c2=float(values["ssim_c2"]) if "ssim_c2" in values else 0.03**2,
    )


def objective_spec_from_loss_config(
    values: Mapping[str, Any],
    *,
    version: str = "rgb_recon_v1",
) -> ObjectiveSpec:
    return ObjectiveSpec(
        version=version,
        reconstruction=reconstruction_loss_spec_from_mapping(values),
        background=background_spec_from_mapping(values["background"]),
    )


def reconstruction_loss_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    spec: ReconstructionLossSpec,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")
    if prediction.dim() != 4 or prediction.shape[1] != 3:
        raise ValueError(f"RGB reconstruction expects [K, 3, H, W], got {tuple(prediction.shape)}")

    delta = prediction - target
    if spec.kind == "mse":
        return _mean_per_image(delta.square())
    if spec.kind == "l1":
        return _mean_per_image(delta.abs())
    if spec.kind == "l1_mse":
        l1 = _mean_per_image(delta.abs())
        mse = _mean_per_image(delta.square())
        return spec.l1_weight * l1 + spec.mse_weight * mse
    if spec.kind == "standard_gs":
        l1 = _mean_per_image(delta.abs())
        dssim = dssim_per_image(
            prediction,
            target,
            window_size=spec.ssim_window_size,
            c1=spec.ssim_c1,
            c2=spec.ssim_c2,
        )
        return spec.l1_weight * l1 + spec.dssim_weight * dssim
    raise ValueError(f"Unknown reconstruction loss type: {spec.kind!r}")


def resize_target_for_render(target: TargetView, *, render_size: int) -> torch.Tensor:
    frames = target.frames
    if frames.dim() != 4 or frames.shape[1] != 3:
        raise ValueError(f"TargetView.frames must have shape [K, 3, H, W], got {tuple(frames.shape)}")
    if int(frames.shape[-2]) == render_size and int(frames.shape[-1]) == render_size:
        return frames
    return F.interpolate(frames, size=(render_size, render_size), mode="bilinear", align_corners=False)


def reconstruction_loss_for_rendered_view(
    rendered: RenderedView,
    loss_spec: ReconstructionLossSpec,
    *,
    weight: float | None = None,
) -> ViewLoss:
    if rendered.target_rgb is None:
        raise ValueError("RenderedView.target_rgb is required for reconstruction loss.")
    per_image = reconstruction_loss_per_image(rendered.rgb, rendered.target_rgb, loss_spec)
    view_weight = rendered.view.loss_weight if weight is None else float(weight)
    return ViewLoss(
        view_id=rendered.view.view_id,
        role=rendered.view.role,
        total=per_image.mean(),
        per_image=per_image,
        weight=view_weight,
    )
