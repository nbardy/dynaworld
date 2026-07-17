from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch

from clip_sampling import sample_clip_batch
from multicam_video_data import MulticamVideoBundle
from runtime_types import ClipBatch, SequenceData

LossKind = Literal["same_view", "novel_view"]
ScheduleMode = Literal["both", "alternate"]


@dataclass(frozen=True)
class SameViewBatch:
    """One same-camera reconstruction sample from the broad scale path."""

    sequence: SequenceData
    clip: ClipBatch
    weight: float = 1.0
    loss_kind: Literal["same_view"] = "same_view"
    loss_name: str = "same_view_recon"


@dataclass(frozen=True)
class NovelViewBatch:
    """One calibrated multicam heldout-view reconstruction sample."""

    bundle: MulticamVideoBundle
    condition_sequence: SequenceData
    clip: ClipBatch
    train_views: tuple[int, ...]
    heldout_views: tuple[int, ...]
    weight: float = 1.0
    loss_kind: Literal["novel_view"] = "novel_view"
    loss_name: str = "heldout_view_recon"


@dataclass(frozen=True)
class MixedStepBatch:
    """The mixed scheduler output for one trainer step.

    `same_view` and `novel_view` stay separate so trainer logs cannot collapse
    the two contracts into a vague aggregate reconstruction loss.
    """

    same_view: SameViewBatch | None = None
    novel_view: NovelViewBatch | None = None

    def batches(self) -> tuple[SameViewBatch | NovelViewBatch, ...]:
        items: list[SameViewBatch | NovelViewBatch] = []
        if self.same_view is not None:
            items.append(self.same_view)
        if self.novel_view is not None:
            items.append(self.novel_view)
        return tuple(items)

    def loss_kinds(self) -> tuple[LossKind, ...]:
        return tuple(batch.loss_kind for batch in self.batches())

    def loss_names(self) -> tuple[str, ...]:
        return tuple(batch.loss_name for batch in self.batches())


def sample_view_indices(
    view_count: int,
    views_per_step: int,
    *,
    device: torch.device | str,
) -> tuple[int, ...]:
    """Sample multicam view ids with the repo's existing 0/all semantics."""

    count = int(view_count)
    if count < 1:
        raise ValueError(f"view_count must be >= 1, got {view_count}.")
    requested = int(views_per_step)
    if requested <= 0 or requested >= count:
        return tuple(range(count))
    return tuple(torch.randperm(count, device=device)[:requested].detach().cpu().tolist())


def scheduled_loss_kinds(
    step: int,
    *,
    mode: ScheduleMode,
    include_same_view: bool = True,
    include_novel_view: bool = True,
) -> tuple[LossKind, ...]:
    """Return which loss families should run for a mixed-training step."""

    enabled: tuple[LossKind, ...] = tuple(
        kind
        for kind, include in (("same_view", include_same_view), ("novel_view", include_novel_view))
        if include
    )
    if not enabled:
        raise ValueError("At least one mixed scheduler loss family must be enabled.")
    if mode == "both":
        return enabled
    if mode == "alternate":
        return (enabled[(max(int(step), 1) - 1) % len(enabled)],)
    raise ValueError(f"Unsupported mixed data schedule mode={mode!r}. Expected one of: both, alternate.")


def sample_same_view_batch(
    sequence: SequenceData,
    *,
    train_frame_count: int,
    frame_sampling: Mapping[str, Any],
    device: torch.device | str,
    weight: float = 1.0,
) -> SameViewBatch:
    return SameViewBatch(
        sequence=sequence,
        clip=sample_clip_batch(
            sequence,
            train_frame_count=train_frame_count,
            frame_sampling=frame_sampling,
            device=device,
        ),
        weight=float(weight),
    )


def sample_novel_view_batch(
    bundle: MulticamVideoBundle,
    *,
    train_frame_count: int,
    frame_sampling: Mapping[str, Any],
    device: torch.device | str,
    train_views_per_step: int = 0,
    heldout_views_per_step: int = 1,
    weight: float = 1.0,
) -> NovelViewBatch:
    if bundle.heldout_view_count < 1:
        raise ValueError("Novel-view mixed training requires at least one heldout view.")
    return NovelViewBatch(
        bundle=bundle,
        condition_sequence=bundle.condition_sequence,
        clip=sample_clip_batch(
            bundle.condition_sequence,
            train_frame_count=train_frame_count,
            frame_sampling=frame_sampling,
            device=device,
        ),
        train_views=sample_view_indices(bundle.train_view_count, train_views_per_step, device=device),
        heldout_views=sample_view_indices(bundle.heldout_view_count, heldout_views_per_step, device=device),
        weight=float(weight),
    )


def sample_mixed_step_batch(
    *,
    step: int,
    schedule_mode: ScheduleMode,
    same_view_sequence: SequenceData | Callable[[], SequenceData] | None,
    multicam_bundle: MulticamVideoBundle | None,
    train_frame_count: int,
    frame_sampling: Mapping[str, Any],
    device: torch.device | str,
    same_view_weight: float = 1.0,
    novel_view_weight: float = 1.0,
    train_views_per_step: int = 0,
    heldout_views_per_step: int = 1,
) -> MixedStepBatch:
    kinds = scheduled_loss_kinds(
        step,
        mode=schedule_mode,
        include_same_view=same_view_sequence is not None,
        include_novel_view=multicam_bundle is not None,
    )
    same_view = None
    if "same_view" in kinds:
        if same_view_sequence is None:
            raise ValueError("scheduled same_view loss without a same_view_sequence.")
        resolved_same_view_sequence = (
            same_view_sequence() if callable(same_view_sequence) else same_view_sequence
        )
        same_view = sample_same_view_batch(
            resolved_same_view_sequence,
            train_frame_count=train_frame_count,
            frame_sampling=frame_sampling,
            device=device,
            weight=same_view_weight,
        )
    novel_view = None
    if "novel_view" in kinds:
        if multicam_bundle is None:
            raise ValueError("scheduled novel_view loss without a multicam_bundle.")
        novel_view = sample_novel_view_batch(
            multicam_bundle,
            train_frame_count=train_frame_count,
            frame_sampling=frame_sampling,
            device=device,
            train_views_per_step=train_views_per_step,
            heldout_views_per_step=heldout_views_per_step,
            weight=novel_view_weight,
        )
    return MixedStepBatch(same_view=same_view, novel_view=novel_view)


__all__ = [
    "LossKind",
    "MixedStepBatch",
    "NovelViewBatch",
    "SameViewBatch",
    "ScheduleMode",
    "sample_mixed_step_batch",
    "sample_novel_view_batch",
    "sample_same_view_batch",
    "sample_view_indices",
    "scheduled_loss_kinds",
]
