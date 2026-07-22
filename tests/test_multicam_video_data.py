from __future__ import annotations

import json
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


def test_load_multicam_video_bundle_uses_official_dnerf_matched_trajectories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from PIL import Image

    scene_dir = tmp_path / "bouncingballs"
    (scene_dir / "train").mkdir(parents=True)
    (scene_dir / "test").mkdir()

    def transform(*, x: float = 0.0, y: float = 0.0) -> list[list[float]]:
        matrix = torch.eye(4)
        matrix[0, 3] = x
        matrix[1, 3] = y
        return matrix.tolist()

    train_frames = [
        {"file_path": "./train/r_000", "time": 0.0, "transform_matrix": transform()},
        {"file_path": "./train/r_001", "time": 1.0, "transform_matrix": transform(x=1.0)},
    ]
    test_frames = [
        {"file_path": "./test/r_000", "time": 0.0, "transform_matrix": transform(y=2.0)},
        {"file_path": "./test/r_001", "time": 1.0, "transform_matrix": transform(x=1.0, y=2.0)},
    ]
    for split, frames in (("train", train_frames), ("test", test_frames)):
        (scene_dir / f"transforms_{split}.json").write_text(
            json.dumps({"camera_angle_x": 0.7, "frames": frames}),
            encoding="utf-8",
        )
        for index in range(2):
            image = Image.new("RGBA", (2, 2), (255, 0, 0, 255))
            image.putpixel((0, 0), (255, 255, 255, 0))
            image.save(scene_dir / split / f"r_{index:03d}.png")

    record = {
        "dataset": "dnerf",
        "sample_id": "dnerf_fixture",
        "dataset_scene_dir": str(scene_dir),
        "source_camera": "train_trajectory",
        "target_camera": "test_trajectory",
        "train_cameras": ["train_trajectory"],
        "heldout_cameras": ["test_trajectory"],
        "anchor_camera": "train_trajectory",
        "condition_camera": "train_trajectory",
        "frame_count": 2,
        "fps": 2.0,
        "dnerf_camera_splits": {"train_trajectory": "train", "test_trajectory": "test"},
        "dnerf_frame_indices": {"train_trajectory": [0, 1], "test_trajectory": [0, 1]},
        "dnerf_times": [0.0, 1.0],
        "dnerf_background": [0.0, 0.0, 0.0],
        "sample_layout": "matched_posed_trajectories",
    }
    monkeypatch.setattr(multicam_video_data, "select_multicam_record", lambda _data_cfg: record)

    bundle = load_multicam_video_bundle(
        data_cfg={"max_frames": 2},
        camera_cfg={"rig_init": "dnerf"},
        target_size=(2, 2),
        device=torch.device("cpu"),
    )

    assert bundle.train_frames.shape == (1, 2, 3, 2, 2)
    assert bundle.heldout_frames is not None and bundle.heldout_frames.shape == (1, 2, 3, 2, 2)
    assert torch.equal(bundle.train_frames[0, 0, :, 0, 0], torch.zeros(3))
    assert torch.equal(bundle.condition_sequence.frame_times[:, 0], torch.tensor([0.0, 1.0]))
    assert not torch.allclose(bundle.train_w2c[0, 0], bundle.train_w2c[0, 1])
    assert bundle.heldout_w2c is not None
    assert not torch.allclose(bundle.train_w2c[0, 0], bundle.heldout_w2c[0, 0])
    assert bundle.pose_source == "dnerf_matched_time_blender_to_opencv_relative_pinhole"
    assert bundle.metadata["sample_layout"] == "matched_posed_trajectories"

    record["dnerf_times"] = [0.0, 0.5]
    with pytest.raises(ValueError, match="do not match the declared paired times"):
        multicam_video_data.dnerf_camera_frames(record, "test_trajectory")


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


def test_camxtime_extra_camera_path_and_start_time(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene_1"
    scene_dir.mkdir()
    (scene_dir / "camera_020.mp4").touch()
    record = {
        "dataset": "camxtime_full_grid",
        "sample_id": "camxtime_row1",
        "source_camera": "camera_000",
        "target_camera": "camera_040",
        "source_video_path": str(scene_dir / "camera_000.mp4"),
        "target_video_path": str(scene_dir / "camera_040.mp4"),
        "camxtime_scene_dir": str(scene_dir),
        "source_start_seconds": 1.25,
        "target_start_seconds": 9.0,
    }

    assert camera_start_seconds(record, "camera_020") == 1.25
    assert multicam_video_data.video_path_for_camera(record, "20") == scene_dir / "camera_020.mp4"


def test_camxtime_defaults_to_opengl_camera_axes() -> None:
    camera_data = {
        "cameras": {
            "0": {"c2w": torch.eye(4).tolist()},
        },
    }

    c2w = multicam_video_data.camxtime_c2w_for_index(
        camera_data,
        0,
        device=torch.device("cpu"),
    )

    assert torch.allclose(c2w, torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0])))
    assert torch.allclose(
        multicam_video_data.camxtime_c2w_for_index(
            camera_data,
            0,
            device=torch.device("cpu"),
            camera_convention="opencv",
        ),
        torch.eye(4),
    )


def test_load_multicam_video_bundle_supports_camxtime_record_level_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scene_dir = tmp_path / "scene_1"
    scene_dir.mkdir()
    for camera_name in ("camera_000", "camera_020", "camera_040"):
        (scene_dir / f"{camera_name}.mp4").touch()
    c2w0 = torch.eye(4).tolist()
    c2w20 = torch.eye(4)
    c2w20[0, 3] = 2.0
    c2w40 = torch.eye(4)
    c2w40[0, 3] = 4.0
    camera_data_path = scene_dir / "camera_data.json"
    camera_data_path.write_text(
        """
{
  "intrinsics": {
    "K": [[100.0, 0.0, 50.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]]
  },
  "n_cameras": 120,
  "cameras": {
    "0": {"c2w": %s},
    "20": {"c2w": %s},
    "40": {"c2w": %s}
  }
}
"""
        % (c2w0, c2w20.tolist(), c2w40.tolist()),
        encoding="utf-8",
    )
    record = {
        "dataset": "camxtime_full_grid",
        "sample_id": "camxtime_row1",
        "source_camera": "camera_000",
        "target_camera": "camera_040",
        "source_video_path": str(scene_dir / "camera_000.mp4"),
        "target_video_path": str(scene_dir / "camera_040.mp4"),
        "camxtime_scene_dir": str(scene_dir),
        "camxtime_camera_data_path": str(camera_data_path),
        "camxtime_source_width": 100,
        "camxtime_source_height": 120,
        "camxtime_camera_convention": "opencv",
        "fps": 4.0,
        "frame_count": 4,
        "train_cameras": ["camera_000", "camera_020"],
        "heldout_cameras": ["camera_040"],
        "anchor_camera": "camera_000",
        "condition_camera": "camera_020",
    }
    camera_values = {"camera_000": 1.0, "camera_020": 2.0, "camera_040": 3.0}

    monkeypatch.setattr(multicam_video_data, "select_multicam_record", lambda _data_cfg: record)

    def fake_load_camera_video(
        _record: dict,
        camera_name: str,
        *,
        target_size: int,
        device: torch.device,
        frame_count: int | None = None,
    ) -> torch.Tensor:
        count = int(frame_count or _record["frame_count"])
        frames = torch.full((count, 3, target_size, target_size), camera_values[camera_name], device=device)
        frames[:, 0, 0, 0] += torch.arange(count, dtype=frames.dtype, device=device)
        return frames

    monkeypatch.setattr(multicam_video_data, "load_camera_video", fake_load_camera_video)

    bundle = load_multicam_video_bundle(
        data_cfg={"max_frames": 3, "frame_indices": [0, 2]},
        camera_cfg={"rig_init": "camxtime", "base_radius": 2.0},
        target_size=4,
        device=torch.device("cpu"),
    )

    assert bundle.pose_source == "camxtime_full_grid_opencv_to_opencv_relative_pinhole"
    assert bundle.train_camera_names == ["camera_000", "camera_020"]
    assert bundle.heldout_camera_names == ["camera_040"]
    assert bundle.train_frames.shape == (2, 2, 3, 4, 4)
    assert bundle.heldout_frames is not None and bundle.heldout_frames.shape == (1, 2, 3, 4, 4)
    assert torch.allclose(bundle.train_K[0], torch.tensor([[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]))
    assert torch.allclose(bundle.train_w2c[0, 0], torch.eye(4))
    assert torch.allclose(bundle.train_w2c[1, 0, :3, 3], torch.tensor([-2.0, 0.0, 0.0]))
    assert bundle.anchor_c2w is not None and torch.allclose(bundle.anchor_c2w, torch.eye(4))
    assert torch.all(bundle.condition_sequence.frames[:, 1:] == 2.0)
    assert torch.allclose(bundle.condition_sequence.frames[:, 0, 0, 0], torch.tensor([2.0, 4.0]))


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
