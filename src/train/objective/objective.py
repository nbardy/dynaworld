from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Protocol

import torch

try:
    from runtime_types import GaussianSequence
except ImportError:  # pragma: no cover - package-style imports in tests.
    from ..runtime_types import GaussianSequence

from .background import BackgroundPolicy
from .loss import (
    reconstruction_loss_for_rendered_view,
    reconstruction_loss_per_image,
    resize_target_for_render,
)
from .types import (
    BackgroundSample,
    ColorizedView,
    ObjectiveSpec,
    RasterizedView,
    RenderedView,
    RunPhase,
    TargetView,
    ViewLoss,
)


class ColorizerProtocol(Protocol):
    def __call__(
        self,
        features: torch.Tensor,
        view_dirs: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class RasterizerProtocol(Protocol):
    def rasterize(
        self,
        decoded: GaussianSequence,
        target: TargetView,
    ) -> RasterizedView: ...


def _validate_rasterized_shape(rasterized: RasterizedView) -> None:
    features = rasterized.features
    if features.dim() != 4:
        raise ValueError(f"RasterizedView.features must have shape [K, F, H, W], got {tuple(features.shape)}")
    if features.shape[0] != rasterized.view.frame_count:
        raise ValueError(
            "RasterizedView frame count must match TargetView frame count: "
            f"{features.shape[0]} vs {rasterized.view.frame_count}"
        )
    if rasterized.alpha is not None:
        expected_alpha = (features.shape[0], features.shape[-2], features.shape[-1])
        if tuple(rasterized.alpha.shape) != expected_alpha:
            raise ValueError(f"alpha must have shape {expected_alpha}, got {tuple(rasterized.alpha.shape)}")


def _validate_rgb_shape(name: str, value: torch.Tensor, *, frame_count: int, height: int, width: int) -> None:
    expected = (frame_count, 3, height, width)
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")


def _validate_alpha_shape(alpha: torch.Tensor, *, frame_count: int, height: int, width: int) -> None:
    expected = (frame_count, height, width)
    if tuple(alpha.shape) != expected:
        raise ValueError(f"alpha must have shape {expected}, got {tuple(alpha.shape)}")


def compose_rgb_background_tensor(
    splat_rgb: torch.Tensor,
    alpha: torch.Tensor | None,
    background: BackgroundSample,
) -> torch.Tensor:
    if splat_rgb.dim() != 4:
        raise ValueError(f"splat_rgb must have shape [K,3,H,W], got {tuple(splat_rgb.shape)}")
    frame_count, _channels, height, width = splat_rgb.shape
    _validate_rgb_shape("splat_rgb", splat_rgb, frame_count=frame_count, height=height, width=width)
    if background.feature is not None and background.rgb is not None:
        raise ValueError("Feature-space and RGB-space backgrounds are mutually exclusive for RGB composition.")
    if alpha is None or background.rgb is None:
        return splat_rgb
    _validate_alpha_shape(alpha, frame_count=frame_count, height=height, width=width)
    alpha_rgb = alpha.unsqueeze(1).to(device=splat_rgb.device, dtype=splat_rgb.dtype)
    bg = background.rgb.to(device=splat_rgb.device, dtype=splat_rgb.dtype)
    return alpha_rgb * splat_rgb + (1.0 - alpha_rgb) * bg


def compose_feature_background_tensor(
    features: torch.Tensor,
    alpha: torch.Tensor | None,
    background: BackgroundSample,
) -> torch.Tensor:
    if features.dim() != 4:
        raise ValueError(f"features must have shape [K,F,H,W], got {tuple(features.shape)}")
    if background.feature is None:
        return features
    if alpha is None:
        raise ValueError("Feature-space background composition requires alpha.")
    _validate_alpha_shape(
        alpha,
        frame_count=int(features.shape[0]),
        height=int(features.shape[-2]),
        width=int(features.shape[-1]),
    )
    alpha_feature = alpha.unsqueeze(1).to(device=features.device, dtype=features.dtype)
    feature_bg = background.feature.to(device=features.device, dtype=features.dtype)
    return features + (1.0 - alpha_feature) * feature_bg


def colorize_and_compose_feature_rgb(
    features: torch.Tensor,
    alpha: torch.Tensor | None,
    colorizer: ColorizerProtocol,
    background: BackgroundSample,
    *,
    view_dirs: torch.Tensor | None = None,
) -> torch.Tensor:
    feature_input = compose_feature_background_tensor(features, alpha, background)
    if view_dirs is None:
        splat_rgb = colorizer(feature_input)
    else:
        splat_rgb = colorizer(feature_input, view_dirs=view_dirs)
    return compose_rgb_background_tensor(splat_rgb, alpha, background)


def compose_rgb(
    *,
    rasterized: RasterizedView,
    colorized: ColorizedView | None,
    background: BackgroundSample,
) -> torch.Tensor:
    _validate_rasterized_shape(rasterized)
    features = rasterized.features
    frame_count, feature_dim, height, width = features.shape

    if colorized is None:
        if feature_dim != 3:
            raise ValueError(
                f"F={feature_dim} feature splatting requires a colorizer before RGB reconstruction."
            )
        splat_rgb = features
    else:
        splat_rgb = colorized.splat_rgb
        _validate_rgb_shape("ColorizedView.splat_rgb", splat_rgb, frame_count=frame_count, height=height, width=width)

    return compose_rgb_background_tensor(splat_rgb, rasterized.alpha, background)


def compose_feature_background(
    rasterized: RasterizedView,
    background: BackgroundSample,
) -> torch.Tensor:
    return compose_feature_background_tensor(rasterized.features, rasterized.alpha, background)


def validate_rendered_rgb_shape(rendered: RenderedView) -> None:
    frame_count, _feature_dim, height, width = rendered.rasterized.features.shape
    _validate_rgb_shape("RenderedView.rgb", rendered.rgb, frame_count=frame_count, height=height, width=width)
    if rendered.target_rgb is not None:
        _validate_rgb_shape("RenderedView.target_rgb", rendered.target_rgb, frame_count=frame_count, height=height, width=width)


class RGBReconObjective:
    def __init__(
        self,
        objective_spec: ObjectiveSpec,
        *,
        colorizer: ColorizerProtocol | None = None,
        background_policy: Any | None = None,
        rasterizer: RasterizerProtocol | None = None,
    ) -> None:
        self.objective_spec = objective_spec
        self.colorizer = colorizer
        self.background_policy = background_policy or BackgroundPolicy(objective_spec.background)
        self.rasterizer = rasterizer

    def profile_section(self, name: str):
        if self.rasterizer is not None and hasattr(self.rasterizer, "profile_section"):
            return self.rasterizer.profile_section(name)
        return nullcontext()

    def rasterize_view(
        self,
        decoded: GaussianSequence,
        target: TargetView,
    ) -> RasterizedView:
        if self.rasterizer is None:
            raise ValueError("RGBReconObjective.rasterize_view requires a rasterizer.")
        with self.profile_section("render/rasterize"):
            return self.rasterizer.rasterize(decoded, target)

    def colorize_view(
        self,
        rasterized: RasterizedView,
        background: BackgroundSample | None = None,
    ) -> ColorizedView | None:
        with self.profile_section("render/colorize"):
            feature_input = (
                rasterized.features if background is None else compose_feature_background(rasterized, background)
            )
            if self.colorizer is None:
                if feature_input.shape[1] == 3:
                    return ColorizedView(splat_rgb=feature_input, view_dirs=rasterized.view_dirs)
                return None
            if hasattr(self.colorizer, "forward_with_logits"):
                rgb, logits = self.colorizer.forward_with_logits(feature_input, view_dirs=rasterized.view_dirs)
                return ColorizedView(splat_rgb=rgb, logits=logits, view_dirs=rasterized.view_dirs)
            return ColorizedView(
                splat_rgb=self.colorizer(feature_input, view_dirs=rasterized.view_dirs),
                view_dirs=rasterized.view_dirs,
            )

    def background_for_view(
        self,
        rasterized: RasterizedView,
        *,
        phase: RunPhase,
        generator: torch.Generator | None = None,
        step: int | None = None,
    ) -> BackgroundSample:
        return self.background_policy.sample(
            phase=phase,
            like=rasterized.features,
            frame_count=rasterized.frame_count,
            generator=generator,
            step=step,
        )

    def sample_background(
        self,
        *,
        phase: RunPhase,
        like: torch.Tensor,
        frame_count: int | None = None,
        generator: torch.Generator | None = None,
        step: int | None = None,
    ) -> BackgroundSample:
        if frame_count is None:
            frame_count = int(like.shape[1] if like.dim() == 5 else like.shape[0])
        return self.background_policy.sample(
            phase=phase,
            like=like,
            frame_count=frame_count,
            generator=generator,
            step=step,
        )

    def compose_view(
        self,
        rasterized: RasterizedView,
        colorized: ColorizedView | None,
        *,
        target_rgb: torch.Tensor | None,
        background: BackgroundSample,
        phase: RunPhase,
    ) -> RenderedView:
        if (
            rasterized.alpha is None
            and background.rgb is not None
            and self.objective_spec.background.apply_when_alpha_missing
        ):
            raise ValueError("background.apply_when_alpha_missing=True requires RasterizedView.alpha.")
        with self.profile_section("render/compose"):
            rgb = compose_rgb(rasterized=rasterized, colorized=colorized, background=background)
        rendered = RenderedView(
            view=rasterized.view,
            rgb=rgb,
            target_rgb=target_rgb,
            rasterized=rasterized,
            colorized=colorized,
            background=background,
            phase=phase,
            metrics_prefix=rasterized.view.metrics_prefix,
        )
        validate_rendered_rgb_shape(rendered)
        return rendered

    def compose_rasterized(
        self,
        rasterized: RasterizedView,
        *,
        phase: RunPhase,
        background: BackgroundSample | None = None,
        generator: torch.Generator | None = None,
        step: int | None = None,
        retain_target: bool = True,
    ) -> RenderedView:
        bg = background or self.background_for_view(
            rasterized,
            phase=phase,
            generator=generator,
            step=step,
        )
        colorized = self.colorize_view(rasterized, bg)
        target_rgb = None
        if retain_target:
            target_rgb = resize_target_for_render(rasterized.view, render_size=rasterized.render_size)
        return self.compose_view(
            rasterized,
            colorized,
            target_rgb=target_rgb,
            background=bg,
            phase=phase,
        )

    def loss_for_view(self, rendered: RenderedView, *, weight: float | None = None) -> ViewLoss:
        return reconstruction_loss_for_rendered_view(
            rendered,
            self.objective_spec.reconstruction,
            weight=weight,
        )

    def reconstruction_loss_per_image(
        self,
        rendered: RenderedView,
        target_rgb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with self.profile_section("loss/reconstruction"):
            target = rendered.target_rgb if target_rgb is None else target_rgb
            if target is None:
                raise ValueError("target_rgb is required for reconstruction_loss_per_image.")
            return reconstruction_loss_per_image(rendered.rgb, target, self.objective_spec.reconstruction)

    def reconstruction_loss(
        self,
        rendered: RenderedView,
        target_rgb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.reconstruction_loss_per_image(rendered, target_rgb).mean()

    def require_alpha_for_feature_background(
        self,
        rendered: RenderedView,
        *,
        context: str = "training",
    ) -> None:
        if rendered.phase != "train":
            return
        if rendered.features.shape[1] == 3:
            return
        if (
            rendered.background.rgb is None
            and rendered.background.feature is None
        ) or rendered.alpha is not None:
            return
        raise ValueError(
            f"F-channel {context} requires alpha-aware render output so background composition is active. "
            "Got alpha=None; check renderer='fast_mac' and v5_features build."
        )

    def loss_for_rasterized(
        self,
        rasterized: RasterizedView,
        *,
        phase: RunPhase,
        background: BackgroundSample | None = None,
        generator: torch.Generator | None = None,
        step: int | None = None,
    ) -> tuple[RenderedView, ViewLoss]:
        rendered = self.compose_rasterized(
            rasterized,
            phase=phase,
            background=background,
            generator=generator,
            step=step,
            retain_target=True,
        )
        return rendered, self.loss_for_view(rendered)

    def render_view(
        self,
        decoded: GaussianSequence,
        target: TargetView,
        *,
        phase: RunPhase,
        background: BackgroundSample | None = None,
        generator: torch.Generator | None = None,
        step: int | None = None,
        retain_target: bool = True,
    ) -> RenderedView:
        rasterized = self.rasterize_view(decoded, target)
        return self.compose_rasterized(
            rasterized,
            phase=phase,
            background=background,
            generator=generator,
            step=step,
            retain_target=retain_target,
        )
