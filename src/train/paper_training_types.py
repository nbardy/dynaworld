from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import torch


PaperRepresentation = Literal["world_tubes", "worldfoam", "dynamic_3dgs"]


@dataclass(frozen=True)
class PaperDatasetContract:
    manifest: str
    sample_id: str
    train_cameras: tuple[str, ...]
    heldout_cameras: tuple[str, ...]
    frame_count: int
    fps: float

    def __post_init__(self) -> None:
        if not self.manifest or not self.sample_id:
            raise ValueError("paper dataset manifest and sample_id must not be empty")
        if not self.train_cameras or not self.heldout_cameras:
            raise ValueError("paper dataset must declare train and heldout cameras")
        if set(self.train_cameras) & set(self.heldout_cameras):
            raise ValueError("paper train and heldout cameras must be disjoint")
        if self.frame_count < 1 or self.fps <= 0.0:
            raise ValueError("paper dataset frame_count and fps must be positive")

    @property
    def samples_per_epoch(self) -> int:
        return len(self.train_cameras) * self.frame_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest,
            "sample_id": self.sample_id,
            "train_cameras": list(self.train_cameras),
            "heldout_cameras": list(self.heldout_cameras),
            "frame_count": self.frame_count,
            "fps": self.fps,
        }


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


@dataclass(frozen=True)
class PaperTrainingProtocol:
    name: str
    dataset: PaperDatasetContract
    steps: int
    max_train_seconds: float
    same_time_count: int
    local_time_count: int
    local_time_radius: int
    sampler_seed_offset: int
    stages: tuple[PaperStage, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("paper protocol name must not be empty")
        if self.steps < 1 or self.max_train_seconds <= 0.0:
            raise ValueError("paper protocol steps and max_train_seconds must be positive")
        if not self.stages or self.stages[-1].end_step != self.steps:
            raise ValueError("paper protocol stages must terminate at steps")
        if self.same_time_count < 1 or self.local_time_count < 0 or self.local_time_radius < 0:
            raise ValueError("paper protocol grouping counts and radius are invalid")
        if any(
            self.same_time_count + self.local_time_count > stage.frames_per_step
            for stage in self.stages
        ):
            raise ValueError("same_time_count + local_time_count must fit every paper stage batch")

    @property
    def final_stage(self) -> PaperStage:
        return self.stages[-1]

    @property
    def target_frame_budget(self) -> int:
        return sum(
            (stage.end_step - stage.start_step) * stage.frames_per_step
            for stage in self.stages
        )

    @property
    def target_pixel_budget(self) -> int:
        return sum(
            (stage.end_step - stage.start_step) * stage.frames_per_step * stage.image_size.pixels
            for stage in self.stages
        )

    @property
    def nominal_epoch_coverage(self) -> float:
        return float(self.target_frame_budget) / float(self.dataset.samples_per_epoch)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset": self.dataset.as_dict(),
            "steps": self.steps,
            "max_train_seconds": self.max_train_seconds,
            "same_time_count": self.same_time_count,
            "local_time_count": self.local_time_count,
            "local_time_radius": self.local_time_radius,
            "sampler_seed_offset": self.sampler_seed_offset,
            "stages": [stage.as_dict() for stage in self.stages],
            "target_frame_budget": self.target_frame_budget,
            "target_pixel_budget": self.target_pixel_budget,
            "samples_per_epoch": self.dataset.samples_per_epoch,
            "nominal_epoch_coverage": self.nominal_epoch_coverage,
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
    serialized_checkpoint_bytes: int
    sampled_peak_current_allocated_bytes: int
    sampled_peak_driver_allocated_bytes: int
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
            "serialized_checkpoint_bytes": self.serialized_checkpoint_bytes,
            "sampled_peak_current_allocated_bytes": self.sampled_peak_current_allocated_bytes,
            "sampled_peak_driver_allocated_bytes": self.sampled_peak_driver_allocated_bytes,
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
    "PaperDatasetContract",
    "PaperRepresentation",
    "PaperStage",
    "PaperTrainingProtocol",
    "PaperTrainerAdapter",
    "SpacetimeBatch",
    "SpacetimeSample",
]
