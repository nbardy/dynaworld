from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCENES = {
    "cook_spinach": (20, "train20_holdout1"),
    "cut_roasted_beef": (19, "train19_holdout1"),
    "flame_steak": (20, "train20_holdout1"),
}


@pytest.mark.parametrize(("scene", "expected_train_count", "split"), [
    (scene, count, split) for scene, (count, split) in SCENES.items()
])
def test_dense_browser_scene_catalog_preserves_camera_split(
    scene: str, expected_train_count: int, split: str
) -> None:
    manifest_path = ROOT / "src" / "dataset_configs" / f"neural3d_{scene}_{split}_full_300f_manifest.jsonl"
    record = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert record["split"] == split
    assert len(record["train_cameras"]) == expected_train_count
    assert record["heldout_cameras"] == ["cam16"]
    assert "cam16" not in record["train_cameras"]
    assert record["frame_count"] == 300

    for suffix, size in (("", [96, 72]), ("_384", [384, 288])):
        bundle_path = ROOT / "web" / "dynaworld_browser_trainer" / f"{scene}_{split}{suffix}.json"
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
        train_rows = [camera for camera in payload["cameras"] if camera["role"] == "train"]
        heldout_rows = [camera for camera in payload["cameras"] if camera["role"] == "heldout"]

        assert payload["decode_size"] == size
        assert payload["dataset_contract"]["split"] == split
        assert len(train_rows) == expected_train_count
        assert [camera["name"] for camera in heldout_rows] == ["cam16"]
        assert all(camera["name"] != "cam16" for camera in train_rows)
        for camera in payload["cameras"]:
            assert (bundle_path.parent / camera["frame_atlas_url"].removeprefix("./")).is_file()
            assert (bundle_path.parent / camera["video_url"].removeprefix("./")).is_file()


def test_deep3d_jump_browser_scene_preserves_native_timeline_and_holdout() -> None:
    manifest_path = ROOT / "src/dataset_configs/deep3d_mask_eval_jump_train9_holdout1_60f_manifest.jsonl"
    record = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert record["dataset"] == "deep3d_mask"
    assert record["fps"] == 120.0
    assert record["frame_count"] == 60
    assert len(record["train_cameras"]) == 9
    assert record["heldout_cameras"] == ["cam05"]
    assert "cam05" not in record["train_cameras"]

    for suffix, size in (("", [96, 72]), ("_384", [384, 288])):
        bundle_path = ROOT / "web/dynaworld_browser_trainer" / f"deep3d_mask_eval_jump_train9_holdout1{suffix}.json"
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
        train_rows = [camera for camera in payload["cameras"] if camera["role"] == "train"]
        heldout_rows = [camera for camera in payload["cameras"] if camera["role"] == "heldout"]

        assert payload["decode_size"] == size
        assert payload["dataset"] == "deep3d_mask"
        assert payload["dataset_contract"]["pose_source"] == "deep3d_mask_llff_opencv_relative_pinhole_v2"
        assert payload["temporal_stream"]["native_frame_count"] == 60
        assert payload["temporal_stream"]["fps"] == 120.0
        assert len(train_rows) == 9
        assert [camera["name"] for camera in heldout_rows] == ["cam05"]
        for camera in payload["cameras"]:
            assert (bundle_path.parent / camera["frame_atlas_url"].removeprefix("./")).is_file()
            assert (bundle_path.parent / camera["video_url"].removeprefix("./")).is_file()
