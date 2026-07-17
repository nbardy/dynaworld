from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.nn import functional as F

from paper_training_types import ImageSize, PaperCostSnapshot, PaperStage, SpacetimeBatch, SpacetimeSample


def normalize_image_size(value: Any, *, name: str = "image size") -> ImageSize:
    if isinstance(value, ImageSize):
        return value
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, [height, width], or object")
    if isinstance(value, int):
        return ImageSize(value, value)
    if isinstance(value, Mapping):
        return ImageSize(int(value["height"]), int(value["width"]))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        return ImageSize(int(value[0]), int(value[1]))
    raise ValueError(f"{name} must be an integer, [height, width], or object")


def normalize_paper_stages(
    raw_stages: Any,
    *,
    total_steps: int,
    default_image_size: ImageSize,
    default_primitive_count: int,
    default_frames_per_step: int,
) -> tuple[PaperStage, ...]:
    if int(total_steps) < 1:
        raise ValueError("total_steps must be positive")
    if raw_stages is None:
        return (
            PaperStage(
                label="fixed",
                start_step=0,
                end_step=int(total_steps),
                image_size=default_image_size,
                primitive_count=int(default_primitive_count),
                frames_per_step=int(default_frames_per_step),
            ),
        )
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("paper_protocol.stages must be a non-empty list or null")

    stages: list[PaperStage] = []
    start_step = 0
    for index, raw in enumerate(raw_stages):
        if not isinstance(raw, Mapping):
            raise ValueError("paper_protocol.stages entries must be objects")
        end_step = int(raw["until_step"])
        image_size = normalize_image_size(
            raw.get("image_size", {"height": raw.get("height"), "width": raw.get("width")}),
            name=f"paper_protocol.stages[{index}].image_size",
        )
        stages.append(
            PaperStage(
                label=str(raw.get("label", f"stage_{index}")),
                start_step=start_step,
                end_step=end_step,
                image_size=image_size,
                primitive_count=int(raw.get("primitive_count", default_primitive_count)),
                frames_per_step=int(raw.get("frames_per_step", default_frames_per_step)),
                lr_multiplier=float(raw.get("lr_multiplier", 1.0)),
            )
        )
        start_step = end_step

    if stages[-1].end_step != int(total_steps):
        raise ValueError("the final paper stage until_step must equal the training step count")
    for previous, current in zip(stages, stages[1:]):
        if current.start_step != previous.end_step:
            raise ValueError("paper stages must be contiguous")
        if current.image_size.height < previous.image_size.height or current.image_size.width < previous.image_size.width:
            raise ValueError("paper stage image sizes must be non-decreasing")
        if current.primitive_count < previous.primitive_count:
            raise ValueError("paper stage primitive counts must be non-decreasing")
    return tuple(stages)


def paper_stage_for_step(stages: tuple[PaperStage, ...], step: int) -> PaperStage:
    for stage in stages:
        if stage.contains(step):
            return stage
    raise IndexError(f"step {step} is outside the paper stage schedule")


class SpacetimeEpochSampler:
    """Coverage-exact shuffled epochs with best-effort spatial/temporal grouping."""

    def __init__(
        self,
        *,
        view_count: int,
        frame_indices: Sequence[int],
        batch_size: int,
        same_time_count: int,
        local_time_count: int,
        local_time_radius: int,
        seed: int,
    ) -> None:
        if int(view_count) < 1:
            raise ValueError("view_count must be positive")
        frames = tuple(int(frame) for frame in frame_indices)
        if not frames or len(set(frames)) != len(frames):
            raise ValueError("frame_indices must be non-empty and unique")
        if min(frames) < 0:
            raise ValueError("frame_indices must be non-negative")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        if int(same_time_count) < 1:
            raise ValueError("same_time_count must be at least one because it includes the anchor")
        if int(local_time_count) < 0 or int(local_time_radius) < 0:
            raise ValueError("local_time_count and local_time_radius must be non-negative")
        if int(same_time_count) + int(local_time_count) > int(batch_size):
            raise ValueError("same_time_count + local_time_count must not exceed batch_size")
        self.view_count = int(view_count)
        self.frame_indices = frames
        self.batch_size = int(batch_size)
        self.same_time_count = int(same_time_count)
        self.local_time_count = int(local_time_count)
        self.local_time_radius = int(local_time_radius)
        self.seed = int(seed)
        self.epoch = -1
        self.batch_index = 0
        self._remaining: list[SpacetimeSample] = []
        self._start_epoch()

    @property
    def samples_per_epoch(self) -> int:
        return self.view_count * len(self.frame_indices)

    def _start_epoch(self) -> None:
        self.epoch += 1
        self.batch_index = 0
        self._remaining = [
            SpacetimeSample(view_index=view, frame_index=frame)
            for view in range(self.view_count)
            for frame in self.frame_indices
        ]
        random.Random(self.seed + self.epoch).shuffle(self._remaining)

    def _take_matching(self, selected: list[SpacetimeSample], predicate: Any, count: int) -> None:
        if count <= 0:
            return
        matches = [sample for sample in self._remaining if predicate(sample)][:count]
        for sample in matches:
            self._remaining.remove(sample)
            selected.append(sample)

    def next_batch(self, batch_size: int | None = None) -> SpacetimeBatch:
        resolved_batch_size = self.batch_size if batch_size is None else int(batch_size)
        if resolved_batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not self._remaining:
            self._start_epoch()
        epoch = self.epoch
        batch_index = self.batch_index
        anchor = self._remaining.pop(0)
        selected = [anchor]
        self._take_matching(
            selected,
            lambda sample: sample.frame_index == anchor.frame_index and sample.view_index != anchor.view_index,
            min(self.same_time_count - 1, resolved_batch_size - len(selected)),
        )
        self._take_matching(
            selected,
            lambda sample: (
                sample.view_index == anchor.view_index
                and 0 < abs(sample.frame_index - anchor.frame_index) <= self.local_time_radius
            ),
            min(self.local_time_count, resolved_batch_size - len(selected)),
        )
        fill_count = min(resolved_batch_size - len(selected), len(self._remaining))
        selected.extend(self._remaining[:fill_count])
        del self._remaining[:fill_count]
        self.batch_index += 1
        return SpacetimeBatch(
            samples=tuple(selected),
            epoch=epoch,
            batch_index=batch_index,
            completes_epoch=not self._remaining,
        )


def resize_video_frames(frames: torch.Tensor, image_size: ImageSize) -> torch.Tensor:
    if frames.ndim not in {4, 5}:
        raise ValueError(f"expected frames [T,C,H,W] or [V,T,C,H,W], got {tuple(frames.shape)}")
    if tuple(frames.shape[-2:]) == (image_size.height, image_size.width):
        return frames
    leading = frames.shape[:-3]
    flattened = frames.reshape(-1, *frames.shape[-3:])
    resized = F.interpolate(
        flattened,
        size=(image_size.height, image_size.width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return resized.reshape(*leading, *resized.shape[-3:]).contiguous()


def resize_ray_grids(rays: torch.Tensor, image_size: ImageSize) -> torch.Tensor:
    if rays.ndim != 4 or rays.shape[-1] != 6:
        raise ValueError(f"expected rays [B,H,W,6], got {tuple(rays.shape)}")
    if tuple(rays.shape[1:3]) == (image_size.height, image_size.width):
        return rays
    channels = rays.permute(0, 3, 1, 2)
    resized = F.interpolate(
        channels,
        size=(image_size.height, image_size.width),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1).contiguous()
    origins = resized[..., :3]
    directions = F.normalize(resized[..., 3:], dim=-1)
    return torch.cat((origins, directions), dim=-1).contiguous()


def scale_intrinsics(K: torch.Tensor, *, source: ImageSize, target: ImageSize) -> torch.Tensor:
    if K.shape[-2:] != (3, 3):
        raise ValueError(f"expected intrinsics [...,3,3], got {tuple(K.shape)}")
    if source == target:
        return K
    scaled = K.clone()
    sx = float(target.width) / float(source.width)
    sy = float(target.height) / float(source.height)
    scaled[..., 0, 0] *= sx
    scaled[..., 0, 2] *= sx
    scaled[..., 1, 1] *= sy
    scaled[..., 1, 2] *= sy
    return scaled


def tensor_bytes(value: torch.Tensor) -> int:
    return int(value.numel() * value.element_size())


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(
        tensor_bytes(value)
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )


class PaperCostTracker:
    def __init__(self) -> None:
        self.optimizer_steps = 0
        self.target_frames = 0
        self.rasterized_frames = 0
        self.target_pixels = 0
        self.rasterized_pixels = 0

    def record(self, *, stage: PaperStage, target_frames: int, rasterized_frames: int) -> None:
        self.optimizer_steps += 1
        self.target_frames += int(target_frames)
        self.rasterized_frames += int(rasterized_frames)
        self.target_pixels += int(target_frames) * stage.image_size.pixels
        self.rasterized_pixels += int(rasterized_frames) * stage.image_size.pixels

    def snapshot(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        elapsed_s: float,
    ) -> PaperCostSnapshot:
        parameters = tuple(model.parameters())
        return PaperCostSnapshot(
            optimizer_steps=self.optimizer_steps,
            target_frames=self.target_frames,
            rasterized_frames=self.rasterized_frames,
            target_pixels=self.target_pixels,
            rasterized_pixels=self.rasterized_pixels,
            parameter_count=sum(parameter.numel() for parameter in parameters),
            trainable_parameter_count=sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
            parameter_bytes=sum(tensor_bytes(parameter) for parameter in parameters),
            optimizer_state_bytes=optimizer_state_bytes(optimizer),
            elapsed_s=float(elapsed_s),
        )


__all__ = [
    "PaperCostTracker",
    "SpacetimeEpochSampler",
    "normalize_image_size",
    "normalize_paper_stages",
    "paper_stage_for_step",
    "resize_ray_grids",
    "resize_video_frames",
    "scale_intrinsics",
]
