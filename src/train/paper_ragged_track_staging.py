"""Bounded data-side bridge from paper batches to view-local track staging.

The paper sampler selects arbitrary ``(view, frame)`` observations.  Native
WorldFoam track programs are view-local because every pixel track owns one ray
program.  This module groups a :class:`SpacetimeBatch` by view without taking a
view/time Cartesian product, preserves each observation's original batch slot,
and carries one explicit ``pixels * observations * RGB`` denominator across all
groups.

This remains deliberately only a data adapter.  The corresponding outer
reduction is implemented in :mod:`paper_ragged_material_bar_coordinator`: it
uses ``P`` global tracks and ``B`` global logical observations, gives each
view-local world a disjoint logical subrange of length ``K_v``, scatters all
compact material bars into one global bar, and authorizes one optimizer update
only after exact coverage.  Connecting that generic compact-bar contract to a
rebuilt native kinetic runtime remains a separate backend integration step.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from paper_training_types import SpacetimeBatch
from powerfoam_track_staging import (
    PowerFoamTrackLossNormalization,
    PowerFoamTrackStagingPlan,
    PowerFoamTrackTargetStageBlock,
)
from powerfoam_training_data import PowerFoamRayProvider, PowerFoamTargetProvider


@dataclass(frozen=True)
class PaperRaggedTrackTargetStageBlock:
    """One view-local ``B_p x K_v`` target block with global normalization."""

    view_index: int
    batch_positions: torch.Tensor
    logical_sample_start: int
    logical_sample_end: int
    staged: PowerFoamTrackTargetStageBlock
    normalization: PowerFoamTrackLossNormalization

    def __post_init__(self) -> None:
        positions = torch.as_tensor(self.batch_positions)
        if positions.ndim != 1 or int(positions.numel()) != int(self.staged.sample_indices.numel()):
            raise ValueError("ragged staged batch positions must match its local sample partition")
        if positions.dtype != torch.long or positions.device.type != "cpu":
            raise ValueError("ragged staged batch positions must be CPU int64")
        if positions.numel() and int(torch.unique(positions).numel()) != int(positions.numel()):
            raise ValueError("ragged staged batch positions must be unique")
        if (
            self.logical_sample_start < 0
            or self.logical_sample_end <= self.logical_sample_start
            or self.logical_sample_end > self.normalization.global_sample_count
            or self.logical_sample_end - self.logical_sample_start != int(positions.numel())
        ):
            raise ValueError("ragged staged logical sample range is invalid")
        if not bool(torch.all(self.staged.view_indices == int(self.view_index)).item()):
            raise ValueError("ragged target block crossed a view boundary")
        if self.normalization.block_track_count != int(self.staged.pixel_indices.numel()):
            raise ValueError("ragged global normalization changed the staged track partition")
        if self.normalization.block_sample_count != int(self.staged.sample_indices.numel()):
            raise ValueError("ragged global normalization changed the staged sample partition")
        if self.normalization.global_rgb_element_count < self.normalization.block_rgb_element_count:
            raise ValueError("ragged target block exceeds its global RGB denominator")

    @property
    def targets(self) -> torch.Tensor:
        return self.staged.targets

    @property
    def pixel_indices(self) -> torch.Tensor:
        return self.staged.pixel_indices

    @property
    def sample_indices(self) -> torch.Tensor:
        return self.staged.sample_indices

    @property
    def frame_indices(self) -> torch.Tensor:
        return self.staged.frame_indices

    @property
    def sample_times(self) -> torch.Tensor:
        return self.staged.sample_times

    @property
    def accounting(self) -> dict[str, Any]:
        return {
            **self.staged.accounting,
            "layout": "paper_ragged_view_pixel_tracks",
            "view_index": self.view_index,
            "batch_positions": self.batch_positions.tolist(),
            "logical_sample_range": [self.logical_sample_start, self.logical_sample_end],
            "local_rgb_element_count": self.normalization.block_rgb_element_count,
            "global_rgb_element_count": self.normalization.global_rgb_element_count,
            "global_denominator_preserved": True,
        }


@dataclass(frozen=True)
class PaperRaggedViewTrackGroup:
    """One sampler batch's observations for a single camera view."""

    view_index: int
    batch_positions: torch.Tensor
    staging_plan: PowerFoamTrackStagingPlan
    global_observation_count: int
    logical_sample_start: int

    def __post_init__(self) -> None:
        if self.global_observation_count < 1:
            raise ValueError("ragged paper batches require a positive global observation count")
        positions = torch.as_tensor(self.batch_positions)
        if positions.ndim != 1 or int(positions.numel()) != self.staging_plan.sample_count:
            raise ValueError("ragged view positions must match the view-local sample count")
        if positions.dtype != torch.long or positions.device.type != "cpu":
            raise ValueError("ragged view positions must be CPU int64")
        if positions.numel() and (int(positions.min()) < 0 or int(positions.max()) >= self.global_observation_count):
            raise IndexError("ragged view position leaves the logical paper batch")
        if int(torch.unique(positions).numel()) != int(positions.numel()):
            raise ValueError("ragged view positions must be unique")
        if self.logical_sample_start < 0 or self.logical_sample_end > self.global_observation_count:
            raise ValueError("ragged view logical sample range leaves the paper batch")
        views = torch.div(
            self.staging_plan.sample_indices,
            self.staging_plan.target_provider.frame_count,
            rounding_mode="floor",
        )
        if not bool(torch.all(views == int(self.view_index)).item()):
            raise ValueError("ragged view group contains samples from another view")

    @property
    def observation_count(self) -> int:
        return self.staging_plan.sample_count

    @property
    def pixel_count(self) -> int:
        return self.staging_plan.track_count

    @property
    def local_rgb_element_count(self) -> int:
        return self.pixel_count * self.observation_count * 3

    @property
    def global_rgb_element_count(self) -> int:
        return self.pixel_count * self.global_observation_count * 3

    @property
    def logical_sample_end(self) -> int:
        return self.logical_sample_start + self.observation_count

    @property
    def logical_step_weight(self) -> float:
        """Weight for combining this group's locally averaged loss and bars."""

        return self.local_rgb_element_count / self.global_rgb_element_count

    def native_sample_state_coordinates(
        self,
        *,
        local_sample_start: int = 0,
        local_sample_end: int | None = None,
    ) -> dict[str, int]:
        """Map a local chart partition to the existing native token scalars."""

        resolved_end = self.observation_count if local_sample_end is None else int(local_sample_end)
        resolved_start = int(local_sample_start)
        if not 0 <= resolved_start < resolved_end <= self.observation_count:
            raise ValueError("native local sample range must be nonempty and stay inside its view group")
        return {
            "global_track_count": self.pixel_count,
            "global_sample_count": self.global_observation_count,
            "global_sample_start": self.logical_sample_start + resolved_start,
            "global_sample_end": self.logical_sample_start + resolved_end,
            "global_loss_element_count": self.global_rgb_element_count,
        }

    def stage_targets(
        self,
        *,
        track_start: int = 0,
        track_end: int | None = None,
        sample_start: int = 0,
        sample_end: int | None = None,
    ) -> PaperRaggedTrackTargetStageBlock:
        """Stage one local block while retaining the whole paper-step denominator."""

        staged = self.staging_plan.stage_targets(
            track_start=track_start,
            track_end=track_end,
            sample_start=sample_start,
            sample_end=sample_end,
        )
        resolved_sample_end = self.observation_count if sample_end is None else int(sample_end)
        positions = self.batch_positions[int(sample_start) : resolved_sample_end]
        normalization = PowerFoamTrackLossNormalization(
            global_track_count=self.pixel_count,
            global_sample_count=self.global_observation_count,
            block_track_count=int(staged.pixel_indices.numel()),
            block_sample_count=int(staged.sample_indices.numel()),
        )
        return PaperRaggedTrackTargetStageBlock(
            view_index=self.view_index,
            batch_positions=positions,
            logical_sample_start=self.logical_sample_start + int(sample_start),
            logical_sample_end=self.logical_sample_start + resolved_sample_end,
            staged=staged,
            normalization=normalization,
        )


@dataclass(frozen=True)
class PaperRaggedTrackBatch:
    """A complete paper batch represented as disjoint view-local track groups."""

    batch: SpacetimeBatch
    pixel_indices: torch.Tensor
    groups: tuple[PaperRaggedViewTrackGroup, ...]
    loss_normalization_id: str

    def __post_init__(self) -> None:
        if not self.groups:
            raise ValueError("ragged track batch requires at least one view group")
        if not self.loss_normalization_id.strip():
            raise ValueError("ragged track batch loss normalization id must be nonempty")
        pixels = torch.as_tensor(self.pixel_indices)
        if pixels.ndim != 1 or pixels.numel() < 1 or pixels.dtype != torch.long or pixels.device.type != "cpu":
            raise ValueError("ragged track pixels must be a nonempty CPU int64 vector")
        expected_positions = list(range(len(self.batch.samples)))
        positions = sorted(position for group in self.groups for position in group.batch_positions.tolist())
        if positions != expected_positions:
            raise ValueError("ragged view groups must cover every paper batch position exactly once")
        if tuple(group.view_index for group in self.groups) != tuple(sorted(group.view_index for group in self.groups)):
            raise ValueError("ragged view groups must use canonical ascending view order")
        if len({group.view_index for group in self.groups}) != len(self.groups):
            raise ValueError("ragged view groups must contain each active view exactly once")
        next_logical_sample = 0
        for group in self.groups:
            if group.logical_sample_start != next_logical_sample:
                raise ValueError("ragged view logical sample ranges must form one canonical tiling")
            next_logical_sample = group.logical_sample_end
        if next_logical_sample != self.observation_count:
            raise ValueError("ragged view logical sample ranges must cover the paper batch")
        target_provider = self.groups[0].staging_plan.target_provider
        ray_provider = self.groups[0].staging_plan.ray_provider
        image_size = (
            self.groups[0].staging_plan.height,
            self.groups[0].staging_plan.width,
        )
        for group in self.groups:
            if group.global_observation_count != self.observation_count:
                raise ValueError("ragged view groups disagree on global observation count")
            if (
                group.staging_plan.target_provider is not target_provider
                or group.staging_plan.ray_provider is not ray_provider
            ):
                raise ValueError("ragged view groups must share exact target and ray providers")
            if (group.staging_plan.height, group.staging_plan.width) != image_size:
                raise ValueError("ragged view groups must share one target image size")
            if not torch.equal(group.staging_plan.pixel_indices, pixels):
                raise ValueError("ragged view groups disagree on the canonical pixel tracks")
            for position, sample_index in zip(
                group.batch_positions.tolist(),
                group.staging_plan.sample_indices.tolist(),
                strict=True,
            ):
                sample = self.batch.samples[int(position)]
                expected = sample.flat_index(group.staging_plan.target_provider.frame_count)
                if int(sample_index) != expected or sample.view_index != group.view_index:
                    raise ValueError("ragged view grouping changed paper sampler identity")
        if sum(group.local_rgb_element_count for group in self.groups) != self.global_rgb_element_count:
            raise ValueError("ragged view groups changed the global RGB denominator")

    @property
    def observation_count(self) -> int:
        return len(self.batch.samples)

    @property
    def pixel_count(self) -> int:
        return int(self.pixel_indices.numel())

    @property
    def global_rgb_element_count(self) -> int:
        return self.pixel_count * self.observation_count * 3

    @property
    def active_view_count(self) -> int:
        return len(self.groups)

    def accounting(self) -> dict[str, Any]:
        """Report adapter-owned state separately from provider-owned residency."""

        tensors = (
            self.pixel_indices,
            *(
                tensor
                for group in self.groups
                for tensor in (
                    group.batch_positions,
                    group.staging_plan.pixel_indices,
                    group.staging_plan.sample_indices,
                    group.staging_plan.sample_times,
                )
            ),
        )
        provider = self.groups[0].staging_plan.target_provider
        ray_provider = self.groups[0].staging_plan.ray_provider
        return {
            "layout": "paper_ragged_view_groups",
            "global_pixel_count": self.pixel_count,
            "global_observation_count": self.observation_count,
            "global_rgb_element_count": self.global_rgb_element_count,
            "active_view_count": self.active_view_count,
            "view_observation_counts": {str(group.view_index): group.observation_count for group in self.groups},
            "view_logical_sample_ranges": {
                str(group.view_index): [group.logical_sample_start, group.logical_sample_end] for group in self.groups
            },
            "view_logical_step_weights": {str(group.view_index): group.logical_step_weight for group in self.groups},
            "adapter_owned_tensor_storage_bytes": _unique_tensor_storage_bytes(tensors),
            "adapter_owned_dense_pixel_observation_tensors": 0,
            "adapter_owned_explicit_ray_tensors": 0,
            "target_provider_residency": provider.residency(),
            "ray_provider_camera_record_count": ray_provider.view_count * ray_provider.frame_count,
            "bounded_adapter_state": True,
        }


def adapt_paper_spacetime_batch_to_track_groups(
    batch: SpacetimeBatch,
    *,
    target_provider: PowerFoamTargetProvider,
    ray_provider: PowerFoamRayProvider,
    frame_times: torch.Tensor,
    height: int | None = None,
    width: int | None = None,
    device: torch.device | str | None = None,
    loss_normalization_id: str | None = None,
) -> PaperRaggedTrackBatch:
    """Group a paper batch by view using only ``O(P + B)`` adapter tensors."""

    if not isinstance(batch, SpacetimeBatch):
        raise TypeError("paper ragged track staging requires a SpacetimeBatch")
    if (target_provider.view_count, target_provider.frame_count) != (
        ray_provider.view_count,
        ray_provider.frame_count,
    ):
        raise ValueError("paper ragged target and ray providers must share one view/frame grid")
    if (target_provider.height, target_provider.width) != (ray_provider.height, ray_provider.width):
        raise ValueError("paper ragged target and ray providers must share source dimensions")
    target_height = target_provider.height if height is None else int(height)
    target_width = target_provider.width if width is None else int(width)
    if target_height < 1 or target_width < 1:
        raise ValueError("paper ragged track staging requires positive target dimensions")

    times = torch.as_tensor(frame_times)
    if times.ndim == 2 and tuple(times.shape) == (target_provider.frame_count, 1):
        times = times[:, 0]
    if times.ndim != 1 or int(times.numel()) != target_provider.frame_count:
        raise ValueError("frame_times must contain exactly one value per provider frame")
    times = times.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(times).all().item()):
        raise ValueError("paper ragged frame_times must be finite")

    grouped_positions: dict[int, list[int]] = {}
    grouped_frames: dict[int, list[int]] = {}
    for position, sample in enumerate(batch.samples):
        if sample.view_index < 0 or sample.view_index >= target_provider.view_count:
            raise IndexError(f"paper sample view {sample.view_index} is outside [0, {target_provider.view_count})")
        if sample.frame_index < 0 or sample.frame_index >= target_provider.frame_count:
            raise IndexError(f"paper sample frame {sample.frame_index} is outside [0, {target_provider.frame_count})")
        grouped_positions.setdefault(sample.view_index, []).append(position)
        grouped_frames.setdefault(sample.view_index, []).append(sample.frame_index)

    pixels = torch.arange(target_height * target_width, dtype=torch.long, device="cpu")
    groups = []
    next_logical_sample = 0
    for view_index in sorted(grouped_positions):
        positions = torch.tensor(grouped_positions[view_index], dtype=torch.long, device="cpu")
        frames = torch.tensor(grouped_frames[view_index], dtype=torch.long, device="cpu")
        sample_indices = frames + int(view_index) * target_provider.frame_count
        groups.append(
            PaperRaggedViewTrackGroup(
                view_index=view_index,
                batch_positions=positions,
                staging_plan=PowerFoamTrackStagingPlan(
                    target_provider=target_provider,
                    ray_provider=ray_provider,
                    pixel_indices=pixels,
                    sample_indices=sample_indices,
                    height=target_height,
                    width=target_width,
                    sample_times=times.index_select(0, frames),
                    device=device,
                ),
                global_observation_count=len(batch.samples),
                logical_sample_start=next_logical_sample,
            )
        )
        next_logical_sample += int(positions.numel())

    normalization_id = loss_normalization_id
    if normalization_id is None:
        digest = hashlib.sha256()
        digest.update(int(batch.epoch).to_bytes(8, "little", signed=True))
        digest.update(int(batch.batch_index).to_bytes(8, "little", signed=True))
        digest.update(bytes((int(batch.completes_epoch),)))
        digest.update(int(target_height).to_bytes(8, "little", signed=False))
        digest.update(int(target_width).to_bytes(8, "little", signed=False))
        for sample in batch.samples:
            digest.update(int(sample.view_index).to_bytes(8, "little", signed=True))
            digest.update(int(sample.frame_index).to_bytes(8, "little", signed=True))
            digest.update(times[sample.frame_index].numpy().tobytes())
        normalization_id = f"paper-ragged-track-batch:{digest.hexdigest()}"
    return PaperRaggedTrackBatch(
        batch=batch,
        pixel_indices=pixels,
        groups=tuple(groups),
        loss_normalization_id=normalization_id,
    )


def _unique_tensor_storage_bytes(tensors: tuple[torch.Tensor, ...]) -> int:
    storages: dict[tuple[str, int, int], int] = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storage_bytes = int(storage.nbytes())
        storages.setdefault((str(tensor.device), int(storage.data_ptr()), storage_bytes), storage_bytes)
    return sum(storages.values())


__all__ = [
    "PaperRaggedTrackBatch",
    "PaperRaggedTrackTargetStageBlock",
    "PaperRaggedViewTrackGroup",
    "adapt_paper_spacetime_batch_to_track_groups",
]
