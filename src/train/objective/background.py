from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .types import BackgroundMode, BackgroundSample, BackgroundSampleScope, BackgroundSpec, RunPhase


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


def _validate_background_mode(value: Any, *, key: str) -> BackgroundMode:
    mode = str(value).lower()
    if mode not in {"white", "black", "fixed_rgb", "random_rgb", "none"}:
        raise ValueError(
            f"Unknown background {key}={mode!r}; expected white, black, fixed_rgb, random_rgb, or none."
        )
    return mode  # type: ignore[return-value]


def _validate_sample_scope(value: Any) -> BackgroundSampleScope:
    scope = str(value).lower()
    if scope not in {"step", "view", "frame", "pixel"}:
        raise ValueError(f"Unknown background sample_scope={scope!r}; expected step, view, frame, or pixel.")
    return scope  # type: ignore[return-value]


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
        fixed_rgb=(float(fixed_rgb[0]), float(fixed_rgb[1]), float(fixed_rgb[2])),
        sample_scope=_validate_sample_scope(values["sample_scope"]) if "sample_scope" in values else "step",
        apply_when_alpha_missing=bool(values["apply_when_alpha_missing"])
        if "apply_when_alpha_missing" in values
        else False,
    )
    if spec.eval_mode == "random_rgb":
        raise ValueError("background.eval_mode='random_rgb' is intentionally unsupported for comparable eval.")
    return spec


def _sample_shape(
    scope: BackgroundSampleScope,
    *,
    frame_count: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    if scope in {"step", "view"}:
        return (1, 3, 1, 1)
    if scope == "frame":
        return (frame_count, 3, 1, 1)
    if scope == "pixel":
        return (frame_count, 3, height, width)
    raise ValueError(f"Unknown background sample scope: {scope!r}")


def _fixed_rgb_tensor(
    rgb: tuple[float, float, float],
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    return like.new_tensor(rgb).view(1, 3, 1, 1)


def sample_background_rgb(
    mode: BackgroundMode,
    *,
    fixed_rgb: tuple[float, float, float],
    scope: BackgroundSampleScope,
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
        return _fixed_rgb_tensor(fixed_rgb, like=like)
    if mode == "random_rgb":
        height = int(like.shape[-2])
        width = int(like.shape[-1])
        shape = _sample_shape(scope, frame_count=frame_count, height=height, width=width)
        kwargs = {"device": like.device, "dtype": like.dtype}
        if generator is not None:
            kwargs["generator"] = generator
        return torch.rand(shape, **kwargs)
    raise ValueError(f"Unknown background mode: {mode!r}")


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
        rgb = sample_background_rgb(
            mode,
            fixed_rgb=self.spec.fixed_rgb,
            scope=self.spec.sample_scope,
            like=like,
            frame_count=frame_count,
            generator=generator,
        )
        return BackgroundSample(
            rgb=rgb,
            mode=mode,
            phase=phase,
            scope=self.spec.sample_scope,
            step=step,
        )
