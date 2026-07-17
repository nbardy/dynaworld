from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .choices import checked_choice
from .types import (
    BackgroundMode,
    BackgroundSample,
    BackgroundSampleScope,
    BackgroundSpec,
    FeatureBackgroundMode,
    RunPhase,
)

BACKGROUND_MODES: frozenset[BackgroundMode] = frozenset(("white", "black", "fixed_rgb", "random_rgb", "none"))
FEATURE_BACKGROUND_MODES: frozenset[FeatureBackgroundMode] = frozenset(("none", "fixed_zero", "random_feature"))
BACKGROUND_SAMPLE_SCOPES: frozenset[BackgroundSampleScope] = frozenset(("step", "view", "frame", "pixel"))


def background_mode_for_phase(spec: BackgroundSpec, phase: RunPhase) -> BackgroundMode:
    if phase == "train":
        return spec.train_mode
    if phase == "eval":
        return spec.eval_mode
    if phase == "preview":
        return spec.preview_mode or spec.eval_mode
    if phase == "export":
        return spec.export_mode or spec.eval_mode
    raise ValueError(f"Unknown run phase: {phase!r}")


def feature_background_mode_for_phase(spec: BackgroundSpec, phase: RunPhase) -> FeatureBackgroundMode:
    if phase == "train":
        return spec.feature_train_mode
    if phase == "eval":
        return spec.feature_eval_mode
    if phase == "preview":
        return spec.feature_preview_mode or spec.feature_eval_mode
    if phase == "export":
        return spec.feature_export_mode or spec.feature_eval_mode
    raise ValueError(f"Unknown run phase: {phase!r}")


def _validate_background_mode(value: Any, *, key: str) -> BackgroundMode:
    return checked_choice(value, allowed=BACKGROUND_MODES, label=f"background {key}")


def _validate_feature_background_mode(value: Any, *, key: str) -> FeatureBackgroundMode:
    return checked_choice(value, allowed=FEATURE_BACKGROUND_MODES, label=f"feature background {key}")


def _validate_sample_scope(value: Any) -> BackgroundSampleScope:
    return checked_choice(value, allowed=BACKGROUND_SAMPLE_SCOPES, label="background sample_scope")


def background_spec_from_mapping(values: Mapping[str, Any]) -> BackgroundSpec:
    """Convert a normalized config boundary dict into a typed background spec."""

    train_mode = values["train_mode"] if "train_mode" in values else values["mode"]
    eval_mode = values["eval_mode"]
    fixed_rgb = values["fixed_rgb"]
    if len(fixed_rgb) != 3:
        raise ValueError(f"background.fixed_rgb must have 3 values, got {fixed_rgb!r}.")
    spec = BackgroundSpec(
        train_mode=_validate_background_mode(train_mode, key="train_mode"),
        eval_mode=_validate_background_mode(eval_mode, key="eval_mode"),
        preview_mode=(
            None
            if "preview_mode" not in values or values["preview_mode"] is None
            else _validate_background_mode(values["preview_mode"], key="preview_mode")
        ),
        export_mode=(
            None
            if "export_mode" not in values or values["export_mode"] is None
            else _validate_background_mode(values["export_mode"], key="export_mode")
        ),
        feature_train_mode=_validate_feature_background_mode(
            values.get("feature_train_mode", "none"), key="feature_train_mode"
        ),
        feature_eval_mode=_validate_feature_background_mode(
            values.get("feature_eval_mode", "none"), key="feature_eval_mode"
        ),
        feature_preview_mode=(
            None
            if "feature_preview_mode" not in values or values["feature_preview_mode"] is None
            else _validate_feature_background_mode(values["feature_preview_mode"], key="feature_preview_mode")
        ),
        feature_export_mode=(
            None
            if "feature_export_mode" not in values or values["feature_export_mode"] is None
            else _validate_feature_background_mode(values["feature_export_mode"], key="feature_export_mode")
        ),
        fixed_rgb=(float(fixed_rgb[0]), float(fixed_rgb[1]), float(fixed_rgb[2])),
        sample_scope=_validate_sample_scope(values["sample_scope"]) if "sample_scope" in values else "step",
        feature_sample_scope=(
            _validate_sample_scope(values["feature_sample_scope"]) if "feature_sample_scope" in values else "step"
        ),
        apply_when_alpha_missing=bool(values["apply_when_alpha_missing"])
        if "apply_when_alpha_missing" in values
        else False,
    )
    if spec.eval_mode == "random_rgb":
        raise ValueError("background.eval_mode='random_rgb' is intentionally unsupported for comparable eval.")
    if spec.feature_eval_mode == "random_feature":
        raise ValueError("background.feature_eval_mode='random_feature' is intentionally unsupported for comparable eval.")
    return spec


def _sample_shape(
    scope: BackgroundSampleScope,
    *,
    frame_count: int,
    channel_count: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    if scope in {"step", "view"}:
        return (1, channel_count, 1, 1)
    if scope == "frame":
        return (frame_count, channel_count, 1, 1)
    if scope == "pixel":
        return (frame_count, channel_count, height, width)
    raise ValueError(f"Unknown background sample scope: {scope!r}")


def _fixed_rgb_tensor(
    rgb: tuple[float, float, float],
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    return like.new_tensor(rgb).view(1, 3, 1, 1)


def sample_background_rgb(
    spec: BackgroundSpec,
    mode: BackgroundMode,
    *,
    like: torch.Tensor,
    frame_count: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode == "white":
        return _fixed_rgb_tensor((1.0, 1.0, 1.0), like=like)
    if mode == "black":
        return _fixed_rgb_tensor((0.0, 0.0, 0.0), like=like)
    if mode == "fixed_rgb":
        return _fixed_rgb_tensor(spec.fixed_rgb, like=like)
    if mode == "random_rgb":
        shape = _sample_shape(
            spec.sample_scope,
            frame_count=frame_count,
            channel_count=3,
            height=int(like.shape[-2]),
            width=int(like.shape[-1]),
        )
        kwargs = {"device": like.device, "dtype": like.dtype}
        if generator is not None:
            kwargs["generator"] = generator
        return torch.rand(shape, **kwargs)
    raise ValueError(f"Unknown background mode: {mode!r}")


def sample_background_feature(
    spec: BackgroundSpec,
    mode: FeatureBackgroundMode,
    *,
    like: torch.Tensor,
    frame_count: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    feature_dim = int(like.shape[-3])
    if mode == "fixed_zero":
        return like.new_zeros((1, feature_dim, 1, 1))
    if mode == "random_feature":
        shape = _sample_shape(
            spec.feature_sample_scope,
            frame_count=frame_count,
            channel_count=feature_dim,
            height=int(like.shape[-2]),
            width=int(like.shape[-1]),
        )
        kwargs = {"device": like.device, "dtype": like.dtype}
        if generator is not None:
            kwargs["generator"] = generator
        return torch.rand(shape, **kwargs)
    raise ValueError(f"Unknown feature background mode: {mode!r}")


class BackgroundPolicy:
    def __init__(self, spec: BackgroundSpec) -> None:
        self.spec = spec

    def sample(
        self,
        *,
        phase: RunPhase,
        like: torch.Tensor,
        frame_count: int,
        generator: torch.Generator | None = None,
        step: int | None = None,
    ) -> BackgroundSample:
        mode = background_mode_for_phase(self.spec, phase)
        feature_mode = feature_background_mode_for_phase(self.spec, phase)
        return BackgroundSample(
            rgb=sample_background_rgb(
                self.spec, mode, like=like, frame_count=frame_count, generator=generator
            ),
            feature=sample_background_feature(
                self.spec,
                feature_mode,
                like=like,
                frame_count=frame_count,
                generator=generator,
            ),
            mode=mode,
            phase=phase,
            step=step,
            feature_mode=feature_mode,
        )
