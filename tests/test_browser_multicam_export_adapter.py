from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from export_dynaworld_browser_bundle import (
    _browser_camera_filename_component,
    _browser_camera_rows,
    _farthest_point_subset,
    _write_browser_frame_atlases,
)
from multicam_video_data import select_multicam_record, validate_multicam_camera_split


TRAIN17_MANIFEST = Path(
    "src/dataset_configs/neural3d_coffee_martini_train17_holdout1_full_300f_manifest.jsonl"
)
TRAIN17_BROWSER_BUNDLE = Path("web/dynaworld_browser_trainer/coffee_martini_train17_holdout1.json")


def test_browser_multicam_adapter_preserves_split_and_writes_exact_frame_atlases(tmp_path: Path) -> None:
    train_frames = torch.zeros(2, 3, 3, 4, 6)
    heldout_frames = torch.zeros(1, 3, 3, 4, 6)
    train_frames[0, 1, 0] = 0.5
    train_frames[1, 2, 1] = 1.0
    heldout_frames[0, 0, 2] = 0.75
    K = torch.tensor([[[6.0, 0.0, 3.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]])
    bundle = SimpleNamespace(
        train_camera_names=["cam04", "cam09"],
        heldout_camera_names=["cam06"],
        train_sequences=(SimpleNamespace(source_path=Path("data/cam04.mp4")), SimpleNamespace(source_path=Path("data/cam09.mp4"))),
        heldout_sequences=(SimpleNamespace(source_path=Path("data/cam06.mp4")),),
        train_frames=train_frames,
        heldout_frames=heldout_frames,
        train_K=K.repeat(2, 1, 1),
        heldout_K=K,
        train_w2c=torch.eye(4).repeat(2, 3, 1, 1),
        heldout_w2c=torch.eye(4).repeat(1, 3, 1, 1),
    )

    atlas_urls = _write_browser_frame_atlases(bundle, tmp_path / "bundle.json")
    rows = _browser_camera_rows(bundle, width=6, height=4, atlas_urls=atlas_urls)

    assert [(row["name"], row["role"]) for row in rows] == [
        ("cam04", "train"),
        ("cam09", "train"),
        ("cam06", "heldout"),
    ]
    assert rows[0]["intrinsics"] == [1.0, 1.0, 0.5, 0.5]
    assert rows[2]["frame_atlas_url"] == "./bundle_cam06.png"
    assert "video_url" not in rows[0]
    with Image.open(tmp_path / "bundle_cam04.png") as atlas:
        assert atlas.size == (18, 4)


def test_browser_multicam_adapter_rejects_moving_camera_v1_payload() -> None:
    moving_w2c = torch.eye(4).repeat(1, 2, 1, 1)
    moving_w2c[0, 1, 0, 3] = 0.1
    bundle = SimpleNamespace(
        train_camera_names=["cam04"],
        heldout_camera_names=[],
        train_sequences=(SimpleNamespace(source_path=Path("data/cam04.mp4")),),
        heldout_sequences=(),
        train_K=torch.eye(3).repeat(1, 1, 1),
        heldout_K=None,
        train_w2c=moving_w2c,
        heldout_w2c=None,
    )

    with pytest.raises(ValueError, match="static camera rigs only"):
        _browser_camera_rows(
            bundle,
            width=6,
            height=4,
            atlas_urls={"cam04": "./bundle_cam04.png"},
        )


def test_browser_camera_filename_components_are_portable() -> None:
    assert _browser_camera_filename_component("rig/cam 04") == "rig_cam_04"
    with pytest.raises(ValueError, match="cannot be represented"):
        _browser_camera_filename_component("../")


def test_browser_seed_subset_is_deterministic_and_spatially_spread() -> None:
    points = torch.tensor([[float(index), 0.0, 0.0] for index in range(10)])

    first = _farthest_point_subset(points, 3)
    second = _farthest_point_subset(points, 3)

    assert torch.equal(first, second)
    assert set(first.tolist()) == {4, 0, 9}


def test_coffee_martini_train17_manifest_preserves_canonical_split() -> None:
    record = select_multicam_record(
        {
            "multicam_manifest": str(TRAIN17_MANIFEST),
            "multicam_split": "train17_holdout1",
            "multicam_sample_id": "neural3d_coffee_martini_train17_holdout_cam06_full_300f",
        }
    )

    assert record["frame_count"] == 300
    assert record["fps"] == 30.0
    assert len(record["train_cameras"]) == 17
    assert record["heldout_cameras"] == ["cam06"]
    assert set(record["train_cameras"]) | set(record["heldout_cameras"]) == {
        "cam00", "cam01", "cam02", "cam04", "cam05", "cam06", "cam07", "cam08", "cam09",
        "cam10", "cam11", "cam12", "cam13", "cam14", "cam16", "cam18", "cam19", "cam20",
    }
    assert record["browser_demo_tier"] == {
        "sampled_frame_count": 16,
        "decode_size": [96, 72],
        "heldout_usage": "validation_only",
    }
    validate_multicam_camera_split(
        train_cameras=record["train_cameras"],
        heldout_cameras=record["heldout_cameras"],
        anchor_camera=record["anchor_camera"],
        condition_camera=record["condition_camera"],
    )


def test_coffee_martini_train17_browser_bundle_is_portable_and_validation_only() -> None:
    payload = json.loads(TRAIN17_BROWSER_BUNDLE.read_text(encoding="utf-8"))
    train_cameras = [camera for camera in payload["cameras"] if camera["role"] == "train"]
    heldout_cameras = [camera for camera in payload["cameras"] if camera["role"] == "heldout"]

    assert payload["dataset_contract"]["split"] == "train17_holdout1"
    assert payload["dataset_contract"]["heldout_usage"] == "validation_only"
    assert payload["dataset_contract"]["camera_motion"] == "static_rig"
    assert payload["decode_size"] == [96, 72]
    assert payload["frame_count"] == 16
    assert payload["frame_indices"] == [
        0, 20, 40, 60, 80, 100, 120, 140, 159, 179, 199, 219, 239, 259, 279, 299,
    ]
    assert len(train_cameras) == 17
    assert [camera["name"] for camera in heldout_cameras] == ["cam06"]
    assert len(payload["seed_points_xyzrgb"]) == 4096

    for camera in payload["cameras"]:
        assert "video_url" not in camera
        atlas_path = TRAIN17_BROWSER_BUNDLE.parent / camera["frame_atlas_url"].removeprefix("./")
        with Image.open(atlas_path) as atlas:
            assert atlas.size == (96 * 16, 72)
