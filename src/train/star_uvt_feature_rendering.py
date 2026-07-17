from __future__ import annotations

import time
from typing import Any

import torch

from colorize import FeatureToColor
from objective.background import BACKGROUND_SAMPLE_SCOPES, BackgroundPolicy
from objective.objective import colorize_and_compose_feature_rgb
from objective.types import BackgroundSample, BackgroundSpec
from star_uvt_runtime import sync_device as _sync_device


ALPHA_BACKGROUND_STRATEGIES = {
    "fixed_black_after_colorizer",
    "random_rgb_after_colorizer",
    "random_feature_before_colorizer",
    "fixed_zero_feature_before_colorizer",
}
ALPHA_BACKGROUND_SAMPLE_SCOPES = set(BACKGROUND_SAMPLE_SCOPES)


def _alpha_background_spec(strategy: str, sample_scope: str) -> BackgroundSpec:
    if strategy not in ALPHA_BACKGROUND_STRATEGIES:
        expected = ", ".join(sorted(ALPHA_BACKGROUND_STRATEGIES))
        raise ValueError(f"alpha_background strategy must be one of: {expected}")
    if sample_scope not in ALPHA_BACKGROUND_SAMPLE_SCOPES:
        expected = ", ".join(sorted(ALPHA_BACKGROUND_SAMPLE_SCOPES))
        raise ValueError(f"alpha_background.sample_scope must be one of: {expected}")
    if strategy == "fixed_black_after_colorizer":
        return BackgroundSpec(
            train_mode="black",
            eval_mode="black",
            fixed_rgb=(0.0, 0.0, 0.0),
            sample_scope=sample_scope,
        )
    if strategy == "random_rgb_after_colorizer":
        return BackgroundSpec(
            train_mode="random_rgb",
            eval_mode="black",
            fixed_rgb=(0.0, 0.0, 0.0),
            sample_scope=sample_scope,
        )
    if strategy == "random_feature_before_colorizer":
        return BackgroundSpec(
            train_mode="none",
            eval_mode="none",
            feature_train_mode="random_feature",
            feature_eval_mode="fixed_zero",
            feature_sample_scope=sample_scope,
        )
    if strategy == "fixed_zero_feature_before_colorizer":
        return BackgroundSpec(
            train_mode="none",
            eval_mode="none",
            feature_train_mode="fixed_zero",
            feature_eval_mode="fixed_zero",
            feature_sample_scope=sample_scope,
        )
    raise AssertionError(f"unhandled alpha background strategy: {strategy}")


def _sample_alpha_background(
    feature_image: torch.Tensor,
    *,
    strategy: str,
    sample_scope: str,
) -> BackgroundSample:
    return BackgroundPolicy(_alpha_background_spec(strategy, sample_scope)).sample(
        phase="train",
        like=feature_image,
        frame_count=int(feature_image.shape[0]),
    )


def _compose_alpha_background_rgb(
    feature_image: torch.Tensor,
    alpha: torch.Tensor,
    colorizer: FeatureToColor,
    *,
    strategy: str,
    sample_scope: str,
) -> torch.Tensor:
    if alpha.shape != (feature_image.shape[0], feature_image.shape[-2], feature_image.shape[-1]):
        raise ValueError(
            "alpha must have shape [T,H,W] matching feature_image, "
            f"got alpha={tuple(alpha.shape)} feature_image={tuple(feature_image.shape)}"
        )
    return colorize_and_compose_feature_rgb(
        feature_image,
        alpha,
        colorizer,
        _sample_alpha_background(feature_image, strategy=strategy, sample_scope=sample_scope),
    )


def _render_rgb_chunks(
    *,
    model: Any,
    colorizer: FeatureToColor,
    render_uvt_feature_tubes: Any,
    shift_ma_for_frame_chunk: Any,
    chunked_uvt_config: Any,
    uvt_config: Any,
    frames: int,
    chunk_size: int,
    device: torch.device,
    alpha_background_strategy: str,
    alpha_background_sample_scope: str,
) -> tuple[torch.Tensor, float]:
    outputs: list[torch.Tensor] = []
    _sync_device(device)
    started = time.perf_counter()
    for frame_start in range(0, frames, chunk_size):
        chunk_frames = min(chunk_size, frames - frame_start)
        ma, q_uvt, depth0, depth_beta, opacity, feature = model.tensors()
        if chunk_frames == frames:
            render = render_uvt_feature_tubes(
                ma,
                q_uvt,
                depth0.detach(),
                depth_beta.detach(),
                opacity,
                feature,
                uvt_config,
            )
        else:
            ma_chunk = shift_ma_for_frame_chunk(
                ma,
                global_frames=uvt_config.frames,
                frame_start=frame_start,
                chunk_frames=chunk_frames,
            )
            render = render_uvt_feature_tubes(
                ma_chunk,
                q_uvt,
                depth0.detach(),
                depth_beta.detach(),
                opacity,
                feature,
                chunked_uvt_config(uvt_config, chunk_frames=chunk_frames),
            )
        rgb = _compose_alpha_background_rgb(
            render.feature_image,
            render.alpha,
            colorizer,
            strategy=alpha_background_strategy,
            sample_scope=alpha_background_sample_scope,
        )
        outputs.append(rgb.permute(0, 2, 3, 1).detach().cpu())
    _sync_device(device)
    return torch.cat(outputs, dim=0).contiguous(), (time.perf_counter() - started) * 1000.0


__all__ = [
    "ALPHA_BACKGROUND_SAMPLE_SCOPES",
    "ALPHA_BACKGROUND_STRATEGIES",
    "_compose_alpha_background_rgb",
    "_render_rgb_chunks",
]
