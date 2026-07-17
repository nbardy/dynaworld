from __future__ import annotations

import pytest
import torch
from torch import nn

from star_uvt_common import grad_norms, target_grid_slice_for_render_chunk


def test_target_grid_slice_for_render_chunk_maps_aligned_windows() -> None:
    assert target_grid_slice_for_render_chunk(
        target_frames=8,
        render_frames=16,
        frame_start=4,
        chunk_frames=4,
    ) == (2, 2)


def test_target_grid_slice_for_render_chunk_rejects_unaligned_windows() -> None:
    with pytest.raises(ValueError, match="frame_chunk_size to align"):
        target_grid_slice_for_render_chunk(
            target_frames=5,
            render_frames=16,
            frame_start=1,
            chunk_frames=4,
        )


def test_grad_norms_reports_model_and_colorizer_gradients() -> None:
    model = nn.Linear(2, 1, bias=False)
    colorizer = nn.Linear(1, 1, bias=False)

    loss = colorizer(model(torch.ones(1, 2))).sum()
    loss.backward()

    norms = grad_norms(model, colorizer)

    assert norms["model.weight"] > 0.0
    assert norms["colorizer.weight"] > 0.0
