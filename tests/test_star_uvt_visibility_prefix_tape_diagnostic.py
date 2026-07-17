from __future__ import annotations

from types import SimpleNamespace

import torch

from research_experiments.star_uvt_feature_tubes.visibility_prefix_tape_diagnostic import (
    _compute_prefix_tape_tensors,
)


def test_prefix_tape_reports_selected_tube_hidden_by_front_mass() -> None:
    ma = torch.tensor(
        [
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=torch.float32,
    )
    q_uvt = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    depth0 = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
    depth_beta = torch.zeros((3, 3), dtype=torch.float32)
    opacity = torch.tensor([0.8, 0.5, 0.2], dtype=torch.float32)
    points = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float32)
    config = SimpleNamespace(alpha_threshold=1.0e-4, transmittance_threshold=0.0, max_alpha=0.99)

    tape = _compute_prefix_tape_tensors(
        ma=ma,
        q_uvt=q_uvt,
        depth0=depth0,
        depth_beta=depth_beta,
        opacity=opacity,
        points=points,
        config=config,
        selected_ids=torch.tensor([1], dtype=torch.int64),
    )

    assert tape["order"].tolist() == [[0, 1, 2]]
    assert torch.allclose(tape["prefix"], torch.tensor([[1.0, 0.2, 0.1]], dtype=torch.float32))
    assert torch.allclose(tape["weight"], torch.tensor([[0.8, 0.1, 0.02]], dtype=torch.float32))
    assert torch.allclose(tape["final_alpha"], torch.tensor([0.92], dtype=torch.float32))
    assert torch.allclose(tape["selected_alpha_max"], torch.tensor([0.5], dtype=torch.float32))
    assert torch.allclose(tape["selected_prefix_at_alpha_max"], torch.tensor([0.2], dtype=torch.float32))
    assert torch.allclose(tape["selected_weight_sum"], torch.tensor([0.1], dtype=torch.float32))
    assert tape["top_tube_id"].tolist() == [0]
    assert tape["top_is_selected"].tolist() == [False]
