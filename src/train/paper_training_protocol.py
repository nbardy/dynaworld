from __future__ import annotations

import random
import io
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from torch.nn import functional as F

from paper_training_types import (
    ImageSize,
    PaperCostSnapshot,
    PaperDatasetContract,
    PaperStage,
    PaperTrainingProtocol,
    SpacetimeBatch,
    SpacetimeSample,
)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def serialized_state_dict_bytes(model: torch.nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return int(buffer.tell())


class PaperPhaseTimer:
    """Device-synchronized cold-forward, steady-forward, backward, and optimizer timing."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.totals = {"forward": 0.0, "backward": 0.0, "optimizer": 0.0}
        self.counts = {"forward": 0, "backward": 0, "optimizer": 0}
        self.cold_compile_forward_s: float | None = None

    @contextmanager
    def measure(self, phase: str):
        started_at = self.start(phase)
        yield
        self.stop(phase, started_at)

    def start(self, phase: str) -> float:
        if phase not in self.totals:
            raise ValueError(f"unsupported paper timing phase: {phase}")
        synchronize_device(self.device)
        return time.perf_counter()

    def stop(self, phase: str, started_at: float) -> float:
        synchronize_device(self.device)
        elapsed = time.perf_counter() - started_at
        if phase == "forward" and self.cold_compile_forward_s is None:
            self.cold_compile_forward_s = elapsed
        else:
            self.totals[phase] += elapsed
            self.counts[phase] += 1
        return elapsed

    def snapshot(self, *, train_wall_s: float) -> dict[str, Any]:
        cold = 0.0 if self.cold_compile_forward_s is None else self.cold_compile_forward_s
        return {
            "definition": "device-synchronized; cold_compile_forward is the first forward including lazy kernel compilation",
            "cold_compile_forward_s": cold,
            "steady_forward_s": self.totals["forward"],
            "steady_forward_calls": self.counts["forward"],
            "backward_s": self.totals["backward"],
            "backward_calls": self.counts["backward"],
            "optimizer_s": self.totals["optimizer"],
            "optimizer_calls": self.counts["optimizer"],
            "train_wall_s": float(train_wall_s),
            "steady_forward_mean_s": (
                self.totals["forward"] / self.counts["forward"] if self.counts["forward"] else 0.0
            ),
            "backward_mean_s": (
                self.totals["backward"] / self.counts["backward"] if self.counts["backward"] else 0.0
            ),
            "optimizer_mean_s": (
                self.totals["optimizer"] / self.counts["optimizer"] if self.counts["optimizer"] else 0.0
            ),
        }


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
        raw_image_size = raw.get("image_size")
        if raw_image_size is None:
            if raw.get("height") is None and raw.get("width") is None:
                raw_image_size = default_image_size
            elif raw.get("height") is not None and raw.get("width") is not None:
                raw_image_size = {"height": raw["height"], "width": raw["width"]}
            else:
                raise ValueError(
                    f"paper_protocol.stages[{index}] must provide both height and width"
                )
        image_size = normalize_image_size(
            raw_image_size,
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


def resolve_paper_training_protocol(raw: Mapping[str, Any]) -> PaperTrainingProtocol:
    if not bool(raw.get("enabled", False)):
        raise ValueError("paper protocol requires enabled=true")
    dataset_raw = raw.get("dataset")
    if not isinstance(dataset_raw, Mapping):
        raise ValueError("paper protocol dataset must be an object")
    raw_stages = raw.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("paper protocol stages must be a non-empty list")
    final_raw = raw_stages[-1]
    if not isinstance(final_raw, Mapping):
        raise ValueError("paper protocol stage entries must be objects")
    final_image_size = normalize_image_size(final_raw.get("image_size"), name="final paper image size")
    final_primitive_count = int(final_raw["primitive_count"])
    default_frames_per_step = int(raw.get("frames_per_step", final_raw.get("frames_per_step", 1)))
    steps = int(raw["steps"])
    stages = normalize_paper_stages(
        raw_stages,
        total_steps=steps,
        default_image_size=final_image_size,
        default_primitive_count=final_primitive_count,
        default_frames_per_step=default_frames_per_step,
    )
    return PaperTrainingProtocol(
        name=str(raw["name"]),
        dataset=PaperDatasetContract(
            manifest=str(dataset_raw["manifest"]),
            sample_id=str(dataset_raw["sample_id"]),
            train_cameras=tuple(str(value) for value in dataset_raw["train_cameras"]),
            heldout_cameras=tuple(str(value) for value in dataset_raw["heldout_cameras"]),
            frame_count=int(dataset_raw["frame_count"]),
            fps=float(dataset_raw["fps"]),
        ),
        steps=steps,
        max_train_seconds=float(raw["max_train_seconds"]),
        same_time_count=int(raw.get("same_time_count", 1)),
        local_time_count=int(raw.get("local_time_count", 0)),
        local_time_radius=int(raw.get("local_time_radius", 0)),
        sampler_seed_offset=int(raw.get("sampler_seed_offset", 7001)),
        stages=stages,
    )


def apply_paper_dataset_contract(
    data_cfg: Mapping[str, Any],
    protocol: Mapping[str, Any] | PaperTrainingProtocol | None,
) -> dict[str, Any]:
    resolved = dict(data_cfg)
    if protocol is None:
        return resolved
    paper = protocol if isinstance(protocol, PaperTrainingProtocol) else resolve_paper_training_protocol(protocol)
    if len(paper.dataset.heldout_cameras) != 1:
        raise ValueError("the current multicam loader requires exactly one heldout camera")
    resolved.update(
        {
            "frame_source": "multicam_val",
            "max_frames": paper.dataset.frame_count,
            "multicam_manifest": paper.dataset.manifest,
            "multicam_sample_id": paper.dataset.sample_id,
            "multicam_train_cameras": list(paper.dataset.train_cameras),
            "multicam_heldout_camera": paper.dataset.heldout_cameras[0],
            "multicam_anchor_camera": paper.dataset.train_cameras[0],
        }
    )
    return resolved


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
        memory: Mapping[str, int] | None = None,
        serialized_checkpoint_bytes: int | None = None,
    ) -> PaperCostSnapshot:
        parameters = tuple(model.parameters())
        memory_values = memory or {}
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
            serialized_checkpoint_bytes=(
                serialized_state_dict_bytes(model)
                if serialized_checkpoint_bytes is None
                else int(serialized_checkpoint_bytes)
            ),
            sampled_peak_current_allocated_bytes=int(
                memory_values.get("sampled_peak_current_allocated_bytes", 0)
            ),
            sampled_peak_driver_allocated_bytes=int(
                memory_values.get("sampled_peak_driver_allocated_bytes", 0)
            ),
            elapsed_s=float(elapsed_s),
        )


__all__ = [
    "apply_paper_dataset_contract",
    "PaperCostTracker",
    "PaperPhaseTimer",
    "SpacetimeEpochSampler",
    "normalize_image_size",
    "normalize_paper_stages",
    "paper_stage_for_step",
    "resolve_paper_training_protocol",
    "resize_ray_grids",
    "resize_video_frames",
    "scale_intrinsics",
    "serialized_state_dict_bytes",
    "synchronize_device",
]
