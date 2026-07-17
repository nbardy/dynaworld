from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from colorize import FeatureToColor
from objective.objective import colorize_and_compose_feature_rgb
from objective.types import BackgroundSample
from star_uvt_colorizers import build_default_feature_colorizer


@dataclass(frozen=True)
class FeatureTubeRenderConfig:
    frames: int
    height: int
    width: int
    feature_dim: int = 32
    alpha_threshold: float = 1.0 / 255.0
    max_alpha: float = 0.99
    feature_background: float = 0.0


def _logit(value: Tensor) -> Tensor:
    clamped = value.clamp(1.0e-5, 1.0 - 1.0e-5)
    return torch.log(clamped) - torch.log1p(-clamped)


def _inv_softplus(value: Tensor) -> Tensor:
    clamped = value.clamp_min(1.0e-8)
    return clamped + torch.log(-torch.expm1(-clamped))


def _quadratic(q_uvt: Tensor, delta: Tensor) -> Tensor:
    return (
        q_uvt[..., 0] * delta[..., 0] * delta[..., 0]
        + 2.0 * q_uvt[..., 1] * delta[..., 0] * delta[..., 1]
        + 2.0 * q_uvt[..., 2] * delta[..., 0] * delta[..., 2]
        + q_uvt[..., 3] * delta[..., 1] * delta[..., 1]
        + 2.0 * q_uvt[..., 4] * delta[..., 1] * delta[..., 2]
        + q_uvt[..., 5] * delta[..., 2] * delta[..., 2]
    )


def make_uvt_grid(
    config: FeatureTubeRenderConfig,
    device: torch.device | str,
    frame_indices: Tensor | None = None,
) -> Tensor:
    dev = torch.device(device)
    if frame_indices is None:
        frames = torch.arange(config.frames, dtype=torch.float32, device=dev)
    else:
        frames = frame_indices.to(device=dev, dtype=torch.float32)
    t = frames - 0.5 * float(config.frames - 1)
    y = torch.arange(config.height, dtype=torch.float32, device=dev) + 0.5
    x = torch.arange(config.width, dtype=torch.float32, device=dev) + 0.5
    tt, yy, xx = torch.meshgrid(t, y, x, indexing="ij")
    return torch.stack((xx, yy, tt), dim=-1).contiguous()


class FeatureScreenTimeTubeModel(nn.Module):
    """Screen-space UVT tubes whose appearance is an unconstrained F-vector."""

    def __init__(
        self,
        tube_count: int,
        config: FeatureTubeRenderConfig,
        *,
        seed: int = 0,
        device: torch.device | str = "cpu",
        min_precision: float = 1.0e-4,
        max_spatial_correlation: float = 0.95,
    ) -> None:
        super().__init__()
        if tube_count <= 0:
            raise ValueError("tube_count must be positive")
        if config.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.config = config
        self.tube_count = int(tube_count)
        self.min_precision = float(min_precision)
        self.max_spatial_correlation = float(max_spatial_correlation)
        if not 0.0 <= self.max_spatial_correlation < 1.0:
            raise ValueError("max_spatial_correlation must be in [0, 1)")

        generator = torch.Generator(device="cpu").manual_seed(seed)
        center_u = torch.rand((tube_count,), generator=generator) * float(config.width)
        center_v = torch.rand((tube_count,), generator=generator) * float(config.height)
        center_uv = torch.stack((center_u, center_v), dim=-1)
        center_t = torch.zeros((tube_count, 1), dtype=torch.float32)
        velocity_uv = torch.randn((tube_count, 2), generator=generator) * 0.25
        precision = torch.full((tube_count, 3), 0.25, dtype=torch.float32)
        opacity = torch.full((tube_count,), 0.35, dtype=torch.float32)
        feature = torch.randn((tube_count, config.feature_dim), generator=generator) * 0.10
        depth0 = torch.linspace(0.8, 1.2, tube_count, dtype=torch.float32)

        dev = torch.device(device)
        self.center_uv = nn.Parameter(center_uv.to(dev))
        self.center_t = nn.Parameter(center_t.to(dev))
        self.velocity_uv = nn.Parameter(velocity_uv.to(dev))
        self.raw_precision = nn.Parameter(_inv_softplus(precision - self.min_precision).to(dev))
        self.raw_spatial_correlation = nn.Parameter(torch.zeros((tube_count,), dtype=torch.float32, device=dev))
        self.raw_opacity = nn.Parameter(_logit(opacity / 0.99).to(dev))
        self.raw_feature = nn.Parameter(feature.to(dev))
        self.depth0 = nn.Parameter(depth0.to(dev))

    def tensors(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        precision = F.softplus(self.raw_precision) + self.min_precision
        lambda_u = precision[:, 0]
        lambda_v = precision[:, 1]
        lambda_t = precision[:, 2]
        spatial_correlation = self.max_spatial_correlation * torch.tanh(self.raw_spatial_correlation)
        lambda_uv = spatial_correlation * torch.sqrt((lambda_u * lambda_v).clamp_min(1.0e-12))
        velocity_u = self.velocity_uv[:, 0]
        velocity_v = self.velocity_uv[:, 1]
        q_uvt = torch.stack(
            (
                lambda_u,
                lambda_uv,
                -(lambda_u * velocity_u + lambda_uv * velocity_v),
                lambda_v,
                -(lambda_uv * velocity_u + lambda_v * velocity_v),
                lambda_t
                + lambda_u * velocity_u.square()
                + 2.0 * lambda_uv * velocity_u * velocity_v
                + lambda_v * velocity_v.square(),
            ),
            dim=-1,
        )
        ma = torch.cat((self.center_uv, self.center_t), dim=-1)
        depth_beta = torch.zeros((self.tube_count, 3), dtype=torch.float32, device=ma.device)
        opacity = torch.sigmoid(self.raw_opacity) * 0.99
        return ma, q_uvt, self.depth0, depth_beta, opacity, self.raw_feature


def dense_render_feature_tubes(
    ma: Tensor,
    q_uvt: Tensor,
    depth0: Tensor,
    depth_beta: Tensor,
    opacity: Tensor,
    feature: Tensor,
    config: FeatureTubeRenderConfig,
    *,
    frame_indices: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Render tiny feature tubes as `[T,F,H,W]` plus alpha `[T,H,W]`."""

    del depth_beta
    if feature.shape != (ma.shape[0], config.feature_dim):
        raise ValueError(f"feature must have shape [N,{config.feature_dim}], got {tuple(feature.shape)}")

    grid = make_uvt_grid(config, ma.device, frame_indices=frame_indices)
    frame_count = int(grid.shape[0])
    delta = grid.unsqueeze(3) - ma.view(1, 1, 1, -1, 3)
    qv = _quadratic(q_uvt.view(1, 1, 1, -1, 6), delta)
    alpha = torch.clamp(opacity.view(1, 1, 1, -1) * torch.exp(-0.5 * qv), max=config.max_alpha)
    if config.alpha_threshold > 0.0:
        alpha = torch.where(alpha >= float(config.alpha_threshold), alpha, torch.zeros_like(alpha))

    order = torch.argsort(depth0.detach(), stable=True).detach().cpu().tolist()
    accum = torch.zeros(
        (frame_count, config.height, config.width, config.feature_dim),
        dtype=torch.float32,
        device=ma.device,
    )
    transmittance = torch.ones((frame_count, config.height, config.width, 1), dtype=torch.float32, device=ma.device)
    background = torch.full((1, 1, 1, config.feature_dim), float(config.feature_background), device=ma.device)
    for tube_id in order:
        alpha_i = alpha[..., tube_id].unsqueeze(-1)
        accum = accum + transmittance * alpha_i * feature[tube_id].view(1, 1, 1, config.feature_dim)
        transmittance = transmittance * (1.0 - alpha_i)

    feature_image = accum + transmittance * background
    alpha_image = 1.0 - transmittance.squeeze(-1)
    return feature_image.permute(0, 3, 1, 2).contiguous(), alpha_image.contiguous()


def render_model_features(
    model: FeatureScreenTimeTubeModel,
    *,
    frame_indices: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()
    return dense_render_feature_tubes(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        feature,
        model.config,
        frame_indices=frame_indices,
    )


def make_default_colorizer(feature_dim: int) -> FeatureToColor:
    return build_default_feature_colorizer(feature_dim=feature_dim, device=torch.device("cpu"))


def colorize_and_compose(
    feature_image: Tensor,
    alpha: Tensor,
    colorizer: FeatureToColor,
    *,
    background_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tensor:
    bg = torch.tensor(background_rgb, dtype=feature_image.dtype, device=feature_image.device).view(1, 3, 1, 1)
    return colorize_and_compose_feature_rgb(
        feature_image,
        alpha,
        colorizer,
        BackgroundSample(rgb=bg, mode="fixed_rgb", phase="train"),
    )


__all__ = [
    "FeatureScreenTimeTubeModel",
    "FeatureTubeRenderConfig",
    "_inv_softplus",
    "_logit",
    "colorize_and_compose",
    "dense_render_feature_tubes",
    "make_default_colorizer",
    "make_uvt_grid",
    "render_model_features",
]
