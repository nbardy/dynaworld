from __future__ import annotations

import pytest
import torch

from fixed_render_graph import background_for_chunk, clone_sequence_for_fixed_render, detach_sequence_for_fixed_render
from objective.types import BackgroundSample
from runtime_types import GaussianSequence


def _sequence() -> GaussianSequence:
    return GaussianSequence(
        xyz=torch.randn(2, 3, 3, requires_grad=True),
        scales=torch.randn(2, 3, 3, requires_grad=True),
        quats=torch.randn(2, 3, 4, requires_grad=True),
        opacities=torch.randn(2, 3, 1, requires_grad=True),
        rgbs=torch.randn(2, 3, 5, requires_grad=True),
    )


def test_background_for_chunk_slices_time_varying_backgrounds() -> None:
    sample = BackgroundSample(
        rgb=torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3),
        feature=torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2),
        mode="random_rgb",
        phase="train",
        feature_mode="random_feature",
        step=7,
    )

    chunk = background_for_chunk(sample, chunk_start=1, chunk_end=3)
    assert torch.equal(chunk.rgb, sample.rgb[1:3])
    assert torch.equal(chunk.feature, sample.feature[1:3])
    assert chunk.mode == "random_rgb"
    assert chunk.phase == "train"
    assert chunk.feature_mode == "random_feature"
    assert chunk.step == 7


def test_background_for_chunk_preserves_singleton_broadcast_backgrounds() -> None:
    sample = BackgroundSample(
        rgb=torch.ones(1, 3),
        feature=torch.zeros(1, 4),
        mode="fixed_rgb",
        phase="train",
        feature_mode="fixed_zero",
    )
    chunk = background_for_chunk(sample, chunk_start=2, chunk_end=5)
    assert chunk.rgb is sample.rgb
    assert chunk.feature is sample.feature


def test_background_for_chunk_rejects_short_nonbroadcast_backgrounds() -> None:
    sample = BackgroundSample(rgb=torch.ones(2, 3), feature=None, mode="random_rgb", phase="train")
    with pytest.raises(ValueError, match="Cannot slice RGB background"):
        background_for_chunk(sample, chunk_start=1, chunk_end=4)


def test_fixed_render_sequence_helpers_detach_or_clone_leaves() -> None:
    sequence = _sequence()
    detached = detach_sequence_for_fixed_render(sequence)
    assert not detached.xyz.requires_grad
    assert not detached.rgbs.requires_grad

    cloned = clone_sequence_for_fixed_render(sequence, freeze_colors=True)
    assert cloned.xyz.requires_grad
    assert not cloned.rgbs.requires_grad
    assert cloned.xyz.data_ptr() != sequence.xyz.data_ptr()

    color_trainable = clone_sequence_for_fixed_render(sequence, freeze_colors=False)
    assert color_trainable.rgbs.requires_grad
