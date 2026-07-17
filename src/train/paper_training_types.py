from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import torch


PaperRepresentation = Literal["world_tubes", "worldfoam", "dynamic_3dgs"]


@dataclass(frozen=True, order=True)
class ImageSize:
    height: int
    width: int

    def __post_init__(self) -> None:
        if self.height < 1 or self.width < 1:
            raise ValueError(f"image size must be positive, got {self.width}x{self.height}")

    @property
    def pixels(self) -> int:
        return self.height * self.width

    def as_list(self) -> list[int]:
        return [self.height, self.width]


@dataclass(frozen=True)
class PaperStage:
    label: str
    start_step: int
    end_step: int
    image_size: ImageSize
    primitive_count: int
    frames_per_step: int
    lr_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("paper stage label must not be empty")
        if self.start_step < 0 or self.end_step <= self.start_step:
            raise ValueError("paper stage steps must satisfy 0 <= start_step < end_step")
        if self.primitive_count < 1:
            raise ValueError("paper stage primitive_count must be positive")
        if self.frames_per_step < 1:
            raise ValueError("paper stage frames_per_step must be positive")
        if self.lr_multiplier <= 0.0:
            raise ValueError("paper stage lr_multiplier must be positive")

    def contains(self, step: int) -> bool:
        return self.start_step <= int(step) < self.end_step

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "height": self.image_size.height,
            "width": self.image_size.width,
            "primitive_count": self.primitive_count,
            "frames_per_step": self.frames_per_step,
            "lr_multiplier": self.lr_multiplier,
        }


@dataclass(frozen=True, order=True)
class SpacetimeSample:
    view_index: int
    frame_index: int

    def flat_index(self, frame_count: int) -> int:
        return self.view_index * int(frame_count) + self.frame_index


@dataclass(frozen=True)
class SpacetimeBatch:
    samples: tuple[SpacetimeSample, ...]
    epoch: int
    batch_index: int
    completes_epoch: bool

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("spacetime batch must contain at least one sample")
        if len(set(self.samples)) != len(self.samples):
            raise ValueError("spacetime batch samples must be unique")

    def flat_indices(self, frame_count: int, *, device: torch.device | str) -> torch.Tensor:
        return torch.tensor(
            [sample.flat_index(frame_count) for sample in self.samples],
            dtype=torch.long,
            device=device,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "batch_index": self.batch_index,
            "completes_epoch": self.completes_epoch,
            "samples": [
                {"view_index": sample.view_index, "frame_index": sample.frame_index}
                for sample in self.samples
            ],
        }


@dataclass(frozen=True)
class MetalKernelSpec:
    representation: PaperRepresentation
    family: str
    forward: str
    backward: str
    deterministic: bool
    implementation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "representation": self.representation,
            "family": self.family,
            "forward": self.forward,
            "backward": self.backward,
            "deterministic": self.deterministic,
            "implementation": self.implementation,
        }


@dataclass(frozen=True)
class PaperCostSnapshot:
    optimizer_steps: int
    target_frames: int
    rasterized_frames: int
    target_pixels: int
    rasterized_pixels: int
    parameter_count: int
    trainable_parameter_count: int
    parameter_bytes: int
    optimizer_state_bytes: int
    elapsed_s: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "optimizer_steps": self.optimizer_steps,
            "target_frames": self.target_frames,
            "rasterized_frames": self.rasterized_frames,
            "target_pixels": self.target_pixels,
            "rasterized_pixels": self.rasterized_pixels,
            "parameter_count": self.parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "parameter_bytes": self.parameter_bytes,
            "optimizer_state_bytes": self.optimizer_state_bytes,
            "elapsed_s": self.elapsed_s,
        }


@runtime_checkable
class PaperTrainerAdapter(Protocol):
    representation: PaperRepresentation
    kernel: MetalKernelSpec

    def run(self, config: dict[str, Any]) -> dict[str, Any]: ...


__all__ = [
    "ImageSize",
    "MetalKernelSpec",
    "PaperCostSnapshot",
    "PaperRepresentation",
    "PaperStage",
    "PaperTrainerAdapter",
    "SpacetimeBatch",
    "SpacetimeSample",
]
