from __future__ import annotations

import torch

from multicam_video_data import source_relative_cameras_from_K_w2c, validate_multicam_camera_split


def _translate(x: float, y: float, z: float) -> torch.Tensor:
    transform = torch.eye(4)
    transform[:3, 3] = torch.tensor([x, y, z])
    return transform


def test_source_relative_cameras_express_target_pose_in_source_frame() -> None:
    source_c2w = _translate(1.0, 0.0, 0.0)
    target_c2w = _translate(1.0, 2.0, 3.0)
    source_w2c_anchor = torch.linalg.inv(source_c2w).unsqueeze(0)
    target_w2c_anchor = torch.linalg.inv(target_c2w).unsqueeze(0)
    K = torch.tensor(
        [
            [10.0, 0.0, 4.0],
            [0.0, 11.0, 5.0],
            [0.0, 0.0, 1.0],
        ]
    )

    cameras = source_relative_cameras_from_K_w2c(
        source_w2c=source_w2c_anchor,
        target_K=K,
        target_w2c=target_w2c_anchor,
        frame_indices=torch.tensor([0]),
    )

    assert len(cameras) == 1
    assert torch.allclose(cameras[0].camera_to_world, torch.linalg.inv(source_c2w) @ target_c2w)
    assert torch.equal(cameras[0].fx, K[0, 0])
    assert torch.equal(cameras[0].fy, K[1, 1])


def test_source_relative_cameras_self_query_is_identity() -> None:
    source_c2w = _translate(1.0, 2.0, 3.0)
    source_w2c_anchor = torch.linalg.inv(source_c2w).unsqueeze(0)
    K = torch.eye(3)

    cameras = source_relative_cameras_from_K_w2c(
        source_w2c=source_w2c_anchor,
        target_K=K,
        target_w2c=source_w2c_anchor,
        frame_indices=torch.tensor([0]),
    )

    assert torch.allclose(cameras[0].camera_to_world, torch.eye(4))


def test_multicam_camera_split_rejects_train_heldout_overlap() -> None:
    try:
        validate_multicam_camera_split(
            train_cameras=["camera_0001", "camera_0015"],
            heldout_cameras=["camera_0015"],
            anchor_camera="camera_0001",
            condition_camera="camera_0001",
        )
    except ValueError as exc:
        assert "overlaps" in str(exc)
    else:
        raise AssertionError("Expected overlapping train/heldout cameras to fail.")
