from __future__ import annotations

import pytest
import torch

from camera_swap_sampling import (
    build_heldout_camera_swap_pairs,
    build_train_camera_swap_pairs,
    camera_swap_pair_counts,
    sample_train_camera_swap_pairs,
)


def _pair_ids(pairs):
    return {
        (pair.source_set, pair.source_view, pair.query_set, pair.query_view, pair.target_set, pair.target_view)
        for pair in pairs
    }


def test_build_train_camera_swap_pairs_includes_self_and_cross_items() -> None:
    pairs = build_train_camera_swap_pairs(2, train_camera_names=["cam_a", "cam_b"])

    assert _pair_ids(pairs) == {
        ("train", 0, "train", 0, "train", 0),
        ("train", 0, "train", 1, "train", 1),
        ("train", 1, "train", 1, "train", 1),
        ("train", 1, "train", 0, "train", 0),
    }
    assert camera_swap_pair_counts(pairs) == {
        "total": 4,
        "self": 2,
        "train_cross": 2,
        "heldout": 0,
    }
    assert pairs[0].source_name == "cam_a"
    assert pairs[1].query_name == "cam_b"


def test_sample_train_camera_swap_pairs_can_force_self_or_cross_class() -> None:
    generator = torch.Generator().manual_seed(7)
    self_pairs = sample_train_camera_swap_pairs(
        3,
        pairs_per_step=12,
        self_pair_probability=1.0,
        generator=generator,
    )

    assert len(self_pairs) == 12
    assert all(pair.is_self_reconstruction for pair in self_pairs)

    generator = torch.Generator().manual_seed(7)
    cross_pairs = sample_train_camera_swap_pairs(
        3,
        pairs_per_step=12,
        self_pair_probability=0.0,
        generator=generator,
    )

    assert len(cross_pairs) == 12
    assert all(pair.is_train_cross_view for pair in cross_pairs)


def test_sample_train_camera_swap_pairs_uses_all_pairs_when_count_is_zero() -> None:
    pairs = sample_train_camera_swap_pairs(
        2,
        pairs_per_step=0,
        generator=torch.Generator().manual_seed(3),
    )

    assert len(pairs) == 4
    assert camera_swap_pair_counts(pairs) == {
        "total": 4,
        "self": 2,
        "train_cross": 2,
        "heldout": 0,
    }


def test_build_heldout_camera_swap_pairs_queries_heldout_from_each_train_world() -> None:
    pairs = build_heldout_camera_swap_pairs(
        2,
        1,
        train_camera_names=["cam_a", "cam_b"],
        heldout_camera_names=["cam_h"],
    )

    assert _pair_ids(pairs) == {
        ("train", 0, "heldout", 0, "heldout", 0),
        ("train", 1, "heldout", 0, "heldout", 0),
    }
    assert camera_swap_pair_counts(pairs) == {
        "total": 2,
        "self": 0,
        "train_cross": 0,
        "heldout": 2,
    }
    assert pairs[0].source_name == "cam_a"
    assert pairs[0].query_name == "cam_h"


def test_camera_swap_pair_builders_validate_inputs() -> None:
    with pytest.raises(ValueError, match="train_view_count"):
        build_train_camera_swap_pairs(0)
    with pytest.raises(ValueError, match="At least one"):
        build_train_camera_swap_pairs(2, include_self=False, include_cross=False)
    with pytest.raises(ValueError, match="train_camera_names"):
        build_train_camera_swap_pairs(2, train_camera_names=["cam_a"])
    with pytest.raises(ValueError, match="self_pair_probability"):
        sample_train_camera_swap_pairs(2, pairs_per_step=1, self_pair_probability=1.5)
