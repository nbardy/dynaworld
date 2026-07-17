from __future__ import annotations

import pytest
import torch

from mixed_data_scheduler import (
    sample_mixed_step_batch,
    sample_novel_view_batch,
    sample_view_indices,
    scheduled_loss_kinds,
)
from multicam_video_data import MulticamVideoBundle
from runtime_types import SequenceData


def test_scheduled_loss_kinds_keeps_contract_names_separate() -> None:
    assert scheduled_loss_kinds(1, mode="both") == ("same_view", "novel_view")
    assert scheduled_loss_kinds(1, mode="alternate") == ("same_view",)
    assert scheduled_loss_kinds(2, mode="alternate") == ("novel_view",)
    assert scheduled_loss_kinds(3, mode="alternate") == ("same_view",)

    with pytest.raises(ValueError, match="At least one"):
        scheduled_loss_kinds(1, mode="both", include_same_view=False, include_novel_view=False)


def test_sample_view_indices_uses_zero_as_all_views() -> None:
    assert sample_view_indices(3, 0, device=torch.device("cpu")) == (0, 1, 2)
    assert sample_view_indices(3, 5, device=torch.device("cpu")) == (0, 1, 2)


def test_sample_novel_view_batch_uses_condition_clip_and_heldout_targets() -> None:
    torch.manual_seed(4)
    bundle = _bundle(train_views=3, heldout_views=2, frame_count=6)

    batch = sample_novel_view_batch(
        bundle,
        train_frame_count=4,
        frame_sampling={"mode": "contiguous"},
        train_views_per_step=2,
        heldout_views_per_step=1,
        device=torch.device("cpu"),
        weight=0.25,
    )

    assert batch.loss_kind == "novel_view"
    assert batch.loss_name == "heldout_view_recon"
    assert batch.weight == 0.25
    assert batch.condition_sequence is bundle.condition_sequence
    assert batch.clip.frame_count == 4
    assert len(batch.train_views) == 2
    assert len(batch.heldout_views) == 1
    assert set(batch.train_views).issubset({0, 1, 2})
    assert set(batch.heldout_views).issubset({0, 1})


def test_sample_mixed_step_batch_returns_explicit_same_and_novel_batches() -> None:
    same_sequence = _sequence(5, value=10.0)
    bundle = _bundle(train_views=2, heldout_views=1, frame_count=5)

    both = sample_mixed_step_batch(
        step=1,
        schedule_mode="both",
        same_view_sequence=same_sequence,
        multicam_bundle=bundle,
        train_frame_count=3,
        frame_sampling={"mode": "contiguous"},
        device=torch.device("cpu"),
        same_view_weight=0.7,
        novel_view_weight=0.3,
    )

    assert both.loss_kinds() == ("same_view", "novel_view")
    assert both.loss_names() == ("same_view_recon", "heldout_view_recon")
    assert both.same_view is not None
    assert both.same_view.weight == 0.7
    assert both.same_view.clip.frame_count == 3
    assert both.novel_view is not None
    assert both.novel_view.weight == 0.3
    assert both.novel_view.heldout_views == (0,)

    alternate = sample_mixed_step_batch(
        step=2,
        schedule_mode="alternate",
        same_view_sequence=same_sequence,
        multicam_bundle=bundle,
        train_frame_count=3,
        frame_sampling={"mode": "contiguous"},
        device=torch.device("cpu"),
    )
    assert alternate.loss_kinds() == ("novel_view",)
    assert alternate.same_view is None
    assert alternate.novel_view is not None


def test_sample_mixed_step_batch_defers_same_view_callable_until_needed() -> None:
    bundle = _bundle(train_views=2, heldout_views=1, frame_count=5)
    calls = 0

    def same_view_sequence() -> SequenceData:
        nonlocal calls
        calls += 1
        return _sequence(5, value=12.0)

    novel_only = sample_mixed_step_batch(
        step=2,
        schedule_mode="alternate",
        same_view_sequence=same_view_sequence,
        multicam_bundle=bundle,
        train_frame_count=3,
        frame_sampling={"mode": "contiguous"},
        device=torch.device("cpu"),
    )
    assert novel_only.loss_kinds() == ("novel_view",)
    assert calls == 0

    same_only = sample_mixed_step_batch(
        step=3,
        schedule_mode="alternate",
        same_view_sequence=same_view_sequence,
        multicam_bundle=bundle,
        train_frame_count=3,
        frame_sampling={"mode": "contiguous"},
        device=torch.device("cpu"),
    )
    assert same_only.loss_kinds() == ("same_view",)
    assert calls == 1


def test_sample_novel_view_batch_requires_heldout_views() -> None:
    bundle = _bundle(train_views=1, heldout_views=0, frame_count=4)

    with pytest.raises(ValueError, match="requires at least one heldout view"):
        sample_novel_view_batch(
            bundle,
            train_frame_count=2,
            frame_sampling={"mode": "contiguous"},
            device=torch.device("cpu"),
        )


def _sequence(frame_count: int, *, value: float = 0.0) -> SequenceData:
    return SequenceData(
        frames=torch.full((frame_count, 3, 1, 1), value, dtype=torch.float32),
        frame_times=torch.linspace(0.0, 1.0, frame_count).view(frame_count, 1),
        video_fps=4.0,
        frame_source="explicit_video",
        selected_frame_count=frame_count,
        all_frame_count=frame_count,
    )


def _bundle(*, train_views: int, heldout_views: int, frame_count: int) -> MulticamVideoBundle:
    train_sequences = tuple(_sequence(frame_count, value=float(view)) for view in range(train_views))
    heldout_sequences = tuple(
        _sequence(frame_count, value=float(100 + view)) for view in range(heldout_views)
    )
    train_frames = torch.stack([sequence.frames for sequence in train_sequences], dim=0)
    if heldout_sequences:
        heldout_frames = torch.stack([sequence.frames for sequence in heldout_sequences], dim=0)
        heldout_K = torch.eye(3).repeat(heldout_views, 1, 1)
        heldout_w2c = torch.eye(4).repeat(heldout_views, frame_count, 1, 1)
        heldout_names = [f"heldout_{view}" for view in range(heldout_views)]
    else:
        heldout_frames = None
        heldout_K = None
        heldout_w2c = None
        heldout_names = []
    return MulticamVideoBundle(
        condition_sequence=train_sequences[0],
        train_sequences=train_sequences,
        train_frames=train_frames,
        train_K=torch.eye(3).repeat(train_views, 1, 1),
        train_w2c=torch.eye(4).repeat(train_views, frame_count, 1, 1),
        train_camera_names=[f"train_{view}" for view in range(train_views)],
        heldout_sequences=heldout_sequences,
        heldout_frames=heldout_frames,
        heldout_K=heldout_K,
        heldout_w2c=heldout_w2c,
        heldout_camera_names=heldout_names,
    )
