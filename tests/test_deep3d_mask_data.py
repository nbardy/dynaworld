from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dataset_pipeline.deep3d_mask import _frame_directories, _validated_poses
from multicam_video_data import make_llff_video_multiview_cameras


def _write_frame(root: Path, index: int, poses: np.ndarray) -> None:
    images = root / "frames" / f"{index:05d}" / "images"
    images.mkdir(parents=True)
    for camera in range(10):
        (images / f"{camera:03d}.jpg").touch()
    np.save(images.parent / "poses_bounds.npy", poses)


def test_eval_preparation_requires_contiguous_static_rig(tmp_path: Path) -> None:
    poses = np.zeros((10, 17), dtype=np.float64)
    _write_frame(tmp_path, 670, poses)
    _write_frame(tmp_path, 671, poses)
    frames = _frame_directories(tmp_path)
    assert [path.name for path in frames] == ["00670", "00671"]
    np.testing.assert_array_equal(_validated_poses(frames), poses)

    changed = poses.copy()
    changed[0, 0] = 1.0
    np.save(frames[1] / "poses_bounds.npy", changed)
    with pytest.raises(ValueError, match="Static-rig poses changed"):
        _validated_poses(frames)


def test_deep3d_llff_camera_identity_is_preserved(tmp_path: Path) -> None:
    poses = np.zeros((10, 17), dtype=np.float64)
    for camera in range(10):
        pose = poses[camera, :15].reshape(3, 5)
        pose[:, :3] = np.eye(3)
        pose[0, 3] = float(camera)
        pose[:, 4] = [1080.0, 1920.0, 1000.0]
        (tmp_path / f"cam{camera:02d}.mp4").touch()
    np.save(tmp_path / "poses_bounds.npy", poses)
    record = {"dataset": "deep3d_mask", "dataset_scene_dir": str(tmp_path), "sample_id": "fixture"}

    train_K, train_w2c, heldout_K, heldout_w2c, source = make_llff_video_multiview_cameras(
        record,
        train_cameras=["cam00", "cam01"],
        heldout_cameras=["cam05"],
        anchor_camera="cam00",
        T=3,
        H=72,
        W=128,
        device=torch.device("cpu"),
    )

    assert train_K.shape == (2, 3, 3)
    assert train_w2c.shape == (2, 3, 4, 4)
    assert heldout_K.shape == (1, 3, 3)
    assert heldout_w2c.shape == (1, 3, 4, 4)
    assert torch.allclose(train_w2c[0], torch.eye(4).expand(3, 4, 4))
    assert source == "deep3d_mask_llff_opencv_relative_pinhole_v2"
