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

    if rasterized.alpha is None or background.rgb is None:
        return splat_rgb

    alpha = rasterized.alpha.unsqueeze(1).to(device=splat_rgb.device, dtype=splat_rgb.dtype)
    bg = background.rgb.to(device=splat_rgb.device, dtype=splat_rgb.dtype)
    return alpha * splat_rgb + (1.0 - alpha) * bg


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

    def colorize_view(self, rasterized: RasterizedView) -> ColorizedView | None:
        with self.profile_section("render/colorize"):
            if self.colorizer is None:
                if rasterized.feature_dim == 3:
                    return ColorizedView(splat_rgb=rasterized.features, view_dirs=rasterized.view_dirs)
                return None
            if hasattr(self.colorizer, "forward_with_logits"):
                rgb, logits = self.colorizer.forward_with_logits(rasterized.features, view_dirs=rasterized.view_dirs)
                return ColorizedView(splat_rgb=rgb, logits=logits, view_dirs=rasterized.view_dirs)
            return ColorizedView(
                splat_rgb=self.colorizer(rasterized.features, view_dirs=rasterized.view_dirs),
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
        colorized = self.colorize_view(rasterized)
        bg = background or self.background_for_view(
            rasterized,
            phase=phase,
            generator=generator,
            step=step,
        )
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
