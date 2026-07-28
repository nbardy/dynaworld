from types import SimpleNamespace

import pytest
import torch

from research_experiments.dynamic_foam.build_pycolmap_known_pose_point_cloud import (
    known_pose_neighbor_pairs,
    seed_provenance_for_bundle,
)


def test_known_pose_builder_declares_train_only_seed_provenance() -> None:
    bundle = SimpleNamespace(
        train_camera_names=["cam04", "cam09"],
        heldout_camera_names=["cam06"],
    )

    assert seed_provenance_for_bundle(bundle) == {
        "method": "known_pose_colmap_triangulation",
        "input_cameras": ["cam04", "cam09"],
        "train_only_verified": True,
        "coordinate_frame": "model",
    }


def test_known_pose_builder_rejects_train_heldout_overlap() -> None:
    bundle = SimpleNamespace(
        train_camera_names=["cam04", "cam06"],
        heldout_camera_names=["cam06"],
    )

    with pytest.raises(ValueError, match=r"overlap heldout cameras.*cam06"):
        seed_provenance_for_bundle(bundle)


def test_known_pose_neighbor_pairs_use_extrinsics_without_crossing_time() -> None:
    train_w2c = torch.eye(4).repeat(4, 2, 1, 1)
    for view_index, camera_center in enumerate((0.0, 1.0, 3.0, 10.0)):
        train_w2c[view_index, :, 0, 3] = -camera_center
    bundle = SimpleNamespace(train_w2c=train_w2c)
    image_records = {
        f"cam{view_index:02d}_frame{frame_index:04d}.png": (view_index, frame_index)
        for frame_index in range(2)
        for view_index in range(4)
    }

    pairs = known_pose_neighbor_pairs(bundle, image_records, neighbor_count=1)

    assert len(pairs) == 6
    assert {
        (image_records[left][0], image_records[right][0])
        for left, right in pairs
        if image_records[left][1] == 0
    } == {(0, 1), (1, 2), (2, 3)}
    assert all(image_records[left][1] == image_records[right][1] for left, right in pairs)
