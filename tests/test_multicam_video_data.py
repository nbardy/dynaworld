from __future__ import annotations

from pathlib import Path

import pytest
import torch

import multicam_video_data
from multicam_video_data import camera_start_seconds, cameras_from_K_w2c, load_multicam_video_bundle


def test_camera_start_seconds_uses_target_offset_for_pair_target() -> None:
    record = {
        "dataset": "vivo",
        "sample_id": "vivo_pair",
        "source_camera": "source",
        "target_camera": "target",
        "source_start_seconds": 1.0,
        "target_start_seconds": 7.5,
    }

    assert camera_start_seconds(record, "source") == 1.0
    assert camera_start_seconds(record, "target") == 7.5


def test_camera_start_seconds_rejects_unindexed_vivo_camera() -> None:
    record = {
        "dataset": "vivo",
        "sample_id": "vivo_pair",
        "source_camera": "source",
        "target_camera": "target",
        "source_start_seconds": 1.0,
        "target_start_seconds": 7.5,
    }

    with pytest.raises(ValueError, match="does not carry its capture-timestamp offset"):
        camera_start_seconds(record, "heldout")


def test_camera_start_seconds_allows_synchronized_extra_camera_sources() -> None:
    record = {
        "dataset": "aist_dance_db",
        "sample_id": "aist_pair",
        "source_camera": "c01",
        "target_camera": "c05",
        "source_start_seconds": 2.0,
        "target_start_seconds": 2.0,
    }

    assert camera_start_seconds(record, "c09") == 2.0


def test_load_multicam_video_bundle_preserves_train_heldout_and_condition_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    record = {
        "dataset": "synthetic",
        "sample_id": "synthetic_multicam",
        "source_camera": "cam_a",
        "target_camera": "cam_c",
        "source_video_path": str(tmp_path / "source.mp4"),
        "target_video_path": str(tmp_path / "target.mp4"),
        "fps": 4.0,
        "frame_count": 4,
    }
    camera_values = {"cam_a": 1.0, "cam_b": 2.0, "cam_c": 3.0}
    requested_counts = []

    monkeypatch.setattr(multicam_video_data, "select_multicam_record", lambda _data_cfg: record)

    def fake_load_camera_video(
        _record: dict,
        camera_name: str,
        *,
        target_size: int,
        device: torch.device,
        frame_count: int | None = None,
    ) -> torch.Tensor:
        requested_counts.append(frame_count)
        count = int(frame_count or _record["frame_count"])
        frames = torch.full((count, 3, target_size, target_size), camera_values[camera_name], device=device)
        frames[:, 0, 0, 0] += torch.arange(count, dtype=frames.dtype, device=device)
        return frames

    monkeypatch.setattr(multicam_video_data, "load_camera_video", fake_load_camera_video)

    bundle = load_multicam_video_bundle(
        data_cfg={
            "multicam_train_cameras": ["cam_a", "cam_b"],
            "multicam_heldout_cameras": ["cam_c"],
            "multicam_anchor_camera": "cam_a",
            "multicam_condition_camera": "cam_b",
            "frame_indices": [0, 2],
            "max_frames": 3,
        },
        camera_cfg={"rig_init": "orthogonal_origin", "base_radius": 2.0, "base_fov_degrees": 60.0},
        target_size=2,
        device=torch.device("cpu"),
    )

    assert bundle.train_camera_names == ["cam_a", "cam_b"]
    assert bundle.heldout_camera_names == ["cam_c"]
    assert requested_counts == [3, 3, 3]
    assert bundle.train_frames.shape == (2, 2, 3, 2, 2)
    assert bundle.heldout_frames is not None and bundle.heldout_frames.shape == (1, 2, 3, 2, 2)
    assert bundle.train_K.shape == (2, 3, 3)
    assert bundle.train_w2c.shape == (2, 2, 4, 4)
    assert bundle.heldout_K is not None and bundle.heldout_K.shape == (1, 3, 3)
    assert bundle.heldout_w2c is not None and bundle.heldout_w2c.shape == (1, 2, 4, 4)
    assert torch.all(bundle.condition_sequence.frames[:, 1:] == 2.0)
    assert torch.allclose(bundle.condition_sequence.frames[:, 0, 0, 0], torch.tensor([2.0, 4.0]))

    camera_grid = cameras_from_K_w2c(bundle.train_K, bundle.train_w2c)
    assert len(camera_grid) == 2
    assert len(camera_grid[0]) == 2
    assert camera_grid[0][0].camera_to_world.shape == (4, 4)


def test_load_multicam_video_bundle_preserves_deepview_lens_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models_path = tmp_path / "models.json"
    models_path.write_text(
        """
[
  {
    "name": "cam_a",
    "projection_type": "fisheye",
    "radial_distortion": [0.1, -0.02],
    "width": 100,
    "height": 100,
    "focal_length": 50.0,
    "pixel_aspect_ratio": 1.0,
    "principal_point": [50.0, 50.0],
    "orientation": [0.0, 0.0, 0.0],
    "position": [0.0, 0.0, 0.0]
  },
  {
    "name": "cam_b",
    "projection_type": "fisheye",
    "radial_distortion": [0.2, -0.03],
    "width": 100,
    "height": 100,
    "focal_length": 50.0,
    "pixel_aspect_ratio": 1.0,
    "principal_point": [50.0, 50.0],
    "orientation": [0.0, 0.1, 0.0],
    "position": [1.0, 0.0, 0.0]
  },
  {
    "name": "cam_c",
    "projection_type": "fisheye",
    "radial_distortion": [0.3, -0.04],
    "width": 100,
    "height": 100,
    "focal_length": 50.0,
    "pixel_aspect_ratio": 1.0,
    "principal_point": [50.0, 50.0],
    "orientation": [0.0, -0.1, 0.0],
    "position": [0.0, 1.0, 0.0]
  }
]
""",
        encoding="utf-8",
    )
    record = {
        "dataset": "deepview_video",
        "sample_id": "deepview_fisheye",
        "source_camera": "cam_a",
        "target_camera": "cam_c",
        "source_video_path": str(tmp_path / "cam_a.mp4"),
        "target_video_path": str(tmp_path / "cam_c.mp4"),
        "models_path": str(models_path),
        "fps": 4.0,
        "frame_count": 2,
    }

    monkeypatch.setattr(multicam_video_data, "select_multicam_record", lambda _data_cfg: record)

    def fake_load_camera_video(
        _record: dict,
        camera_name: str,
        *,
        target_size: int,
        device: torch.device,
        frame_count: int | None = None,
    ) -> torch.Tensor:
        value = {"cam_a": 1.0, "cam_b": 2.0, "cam_c": 3.0}[camera_name]
        count = int(frame_count or _record["frame_count"])
        return torch.full((count, 3, target_size, target_size), value, device=device)

    monkeypatch.setattr(multicam_video_data, "load_camera_video", fake_load_camera_video)

    bundle = load_multicam_video_bundle(
        data_cfg={
            "multicam_train_cameras": ["cam_a", "cam_b"],
            "multicam_heldout_camera": "cam_c",
            "multicam_anchor_camera": "cam_a",
            "max_frames": 2,
        },
        camera_cfg={"rig_init": "deepview"},
        target_size=4,
        device=torch.device("cpu"),
    )

    assert bundle.pose_source == "deepview_models_relative_opencv_fisheye"
    assert bundle.train_lens_models == ["opencv_fisheye", "opencv_fisheye"]
    assert bundle.heldout_lens_models == ["opencv_fisheye"]
    assert bundle.train_distortions is not None
    assert bundle.heldout_distortions is not None
    assert torch.allclose(bundle.train_distortions[1], torch.tensor([0.2, -0.03, 0.0, 0.0]))
    assert torch.allclose(bundle.heldout_distortions[0], torch.tensor([0.3, -0.04, 0.0, 0.0]))


def test_cameras_from_K_w2c_preserves_lens_metadata() -> None:
    K = torch.eye(3).unsqueeze(0)
    w2c = torch.eye(4).view(1, 1, 4, 4)
    distortion = torch.tensor([[0.1, -0.02, 0.0, 0.0]])

    camera_grid = cameras_from_K_w2c(
        K,
        w2c,
        lens_models=["opencv_fisheye"],
        distortions=distortion,
    )

    assert camera_grid[0][0].lens_model == "opencv_fisheye"
    assert torch.equal(camera_grid[0][0].distortion, distortion[0])


def test_deepview_lens_metadata_maps_fisheye_radial_distortion(tmp_path: Path) -> None:
    models_path = tmp_path / "models.json"
    models_path.write_text(
        """
[
  {
    "name": "camera_0001",
    "projection_type": "fisheye",
    "radial_distortion": [0.1, -0.02, 0.003],
    "width": 2560,
    "height": 1920,
    "focal_length": 1100.0,
    "pixel_aspect_ratio": 1.0,
    "principal_point": [1280.0, 960.0],
    "orientation": [0.0, 0.0, 0.0],
    "position": [0.0, 0.0, 0.0]
  }
]
""",
        encoding="utf-8",
    )

    lens_models, distortions = multicam_video_data.deepview_lens_metadata(
        {"models_path": str(models_path)},
        ["camera_0001"],
        device=torch.device("cpu"),
    )

    assert lens_models == ["opencv_fisheye"]
    assert distortions is not None
    assert torch.allclose(distortions[0], torch.tensor([0.1, -0.02, 0.003, 0.0]))
