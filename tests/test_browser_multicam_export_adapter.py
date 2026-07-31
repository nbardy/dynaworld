from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import export_dynaworld_browser_bundle as browser_export
import pytest
import torch
from export_dynaworld_browser_bundle import (
    _browser_camera_filename_component,
    _browser_camera_rows,
    _browser_sparse_frame_record,
    _farthest_point_subset,
    _load_browser_multicam_sparse_frames,
    _resolve_seed_provenance,
    _seed_points_in_anchor_frame,
    _write_browser_frame_atlases,
)
from multicam_video_data import select_multicam_record, validate_multicam_camera_split
from PIL import Image

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


def test_browser_sparse_frame_record_preserves_contract_and_offsets_exact_time() -> None:
    record = {
        "frame_count": 300,
        "fps": 30.0,
        "source_start_seconds": 1.0,
        "target_start_seconds": 1.5,
        "train_cameras": ["cam04", "cam09"],
        "heldout_cameras": ["cam06"],
    }

    sampled = _browser_sparse_frame_record(record, 159)

    assert sampled["frame_count"] == 1
    assert sampled["duration_seconds"] == pytest.approx(1.0 / 30.0)
    assert sampled["source_start_seconds"] == pytest.approx(1.0 + 159.0 / 30.0)
    assert sampled["target_start_seconds"] == pytest.approx(1.5 + 159.0 / 30.0)
    assert sampled["train_cameras"] == record["train_cameras"]
    assert record["frame_count"] == 300
    with pytest.raises(ValueError, match="outside source frame_count"):
        _browser_sparse_frame_record(record, 300)


def test_browser_sparse_frame_decode_loads_only_requested_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "neural_3d_video",
                "sample_id": "sample",
                "split": "train17_holdout1",
                "frame_count": 300,
                "fps": 30.0,
                "duration_seconds": 10.0,
                "source_start_seconds": 1.0,
                "target_start_seconds": 1.5,
                "train_cameras": ["cam04", "cam09"],
                "heldout_cameras": ["cam06"],
                "anchor_camera": "cam04",
                "condition_camera": "cam04",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sampled_records = []

    def fake_load_multicam_video_bundle(**kwargs):
        sampled_record = json.loads(
            Path(kwargs["data_cfg"]["multicam_manifest"]).read_text(encoding="utf-8")
        )
        sampled_records.append(sampled_record)
        marker = round((float(sampled_record["source_start_seconds"]) - 1.0) * 30.0)
        train_w2c = torch.eye(4).repeat(2, 1, 1, 1)
        heldout_w2c = torch.eye(4).repeat(1, 1, 1, 1)
        return SimpleNamespace(
            frame_count=1,
            condition_sequence=None,
            train_sequences=(),
            train_frames=torch.full((2, 1, 3, 2, 3), float(marker)),
            train_K=torch.eye(3).repeat(2, 1, 1),
            train_w2c=train_w2c,
            train_camera_names=["cam04", "cam09"],
            train_lens_models=None,
            train_distortions=None,
            heldout_sequences=(),
            heldout_frames=torch.full((1, 1, 3, 2, 3), float(marker)),
            heldout_K=torch.eye(3).repeat(1, 1, 1),
            heldout_w2c=heldout_w2c,
            heldout_camera_names=["cam06"],
            heldout_lens_models=None,
            heldout_distortions=None,
            pose_source="neural_3d_llff_relative_pinhole",
            anchor_c2w=torch.eye(4),
            metadata=sampled_record,
        )

    monkeypatch.setattr(
        browser_export,
        "load_multicam_video_bundle",
        fake_load_multicam_video_bundle,
    )

    bundle = _load_browser_multicam_sparse_frames(
        manifest_path=manifest_path,
        sample_id="sample",
        split="train17_holdout1",
        target_size=(2, 3),
        frame_indices=[0, 159, 299],
    )

    assert len(sampled_records) == 3
    assert all(record["frame_count"] == 1 for record in sampled_records)
    assert [record["source_start_seconds"] for record in sampled_records] == pytest.approx(
        [1.0, 1.0 + 159.0 / 30.0, 1.0 + 299.0 / 30.0]
    )
    assert bundle.train_frames.shape == (2, 3, 3, 2, 3)
    assert bundle.train_frames[0, :, 0, 0, 0].tolist() == [0.0, 159.0, 299.0]
    assert bundle.heldout_frames is not None
    assert bundle.heldout_frames[0, :, 0, 0, 0].tolist() == [0.0, 159.0, 299.0]
    assert bundle.metadata["frame_count"] == 300


def test_browser_sparse_frame_decode_rejects_camera_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "neural_3d_video",
                "sample_id": "sample",
                "split": "train17_holdout1",
                "frame_count": 2,
                "fps": 30.0,
                "duration_seconds": 2.0 / 30.0,
                "source_start_seconds": 0.0,
                "target_start_seconds": 0.0,
                "train_cameras": ["cam04"],
                "heldout_cameras": ["cam06"],
                "anchor_camera": "cam04",
                "condition_camera": "cam04",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    call_count = 0

    def fake_load_multicam_video_bundle(**_kwargs):
        nonlocal call_count
        K = torch.eye(3).unsqueeze(0)
        K[0, 0, 0] += call_count
        call_count += 1
        return SimpleNamespace(
            frame_count=1,
            condition_sequence=None,
            train_sequences=(),
            train_frames=torch.zeros((1, 1, 3, 2, 2)),
            train_K=K,
            train_w2c=torch.eye(4).repeat(1, 1, 1, 1),
            train_camera_names=["cam04"],
            train_lens_models=None,
            train_distortions=None,
            heldout_sequences=(),
            heldout_frames=torch.zeros((1, 1, 3, 2, 2)),
            heldout_K=torch.eye(3).unsqueeze(0),
            heldout_w2c=torch.eye(4).repeat(1, 1, 1, 1),
            heldout_camera_names=["cam06"],
            heldout_lens_models=None,
            heldout_distortions=None,
            pose_source="neural_3d_llff_relative_pinhole",
            anchor_c2w=torch.eye(4),
            metadata={},
        )

    monkeypatch.setattr(
        browser_export,
        "load_multicam_video_bundle",
        fake_load_multicam_video_bundle,
    )

    with pytest.raises(RuntimeError, match="train intrinsics changed"):
        _load_browser_multicam_sparse_frames(
            manifest_path=manifest_path,
            sample_id="sample",
            split="train17_holdout1",
            target_size=(2, 2),
            frame_indices=[0, 1],
        )


def test_browser_seed_subset_is_deterministic_and_spatially_spread() -> None:
    points = torch.tensor([[float(index), 0.0, 0.0] for index in range(10)])

    first = _farthest_point_subset(points, 3)
    second = _farthest_point_subset(points, 3)

    assert torch.equal(first, second)
    assert set(first.tolist()) == {4, 0, 9}


def test_browser_export_serializes_verified_train_only_seed_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = tmp_path / "colmap_seed_report.json"
    report_path.write_text(
        json.dumps(
            {
                "method": "colmap_sfm",
                "input_cameras": ["cam04", "cam09"],
                "train_only_verified": True,
                "coordinate_frame": "model",
            }
        ),
        encoding="utf-8",
    )
    bundle = SimpleNamespace(
        train_camera_names=["cam04", "cam09"],
        heldout_camera_names=["cam06"],
        anchor_c2w=torch.eye(4),
        train_view_count=2,
        train_w2c=torch.eye(4).repeat(2, 1, 1, 1),
        train_K=torch.tensor(
            [
                [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]],
                [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]],
            ]
        ),
        metadata={
            "fps": 30.0,
            "anchor_camera": "cam04",
            "dataset": "Neural3D",
            "scene": "coffee_martini",
        },
        pose_source="neural_3d_video",
    )
    seed_path = tmp_path / "train_only_sparse.ply"
    output_path = tmp_path / "bundle.json"
    monkeypatch.setattr(browser_export, "load_multicam_video_bundle", lambda **_kwargs: bundle)
    monkeypatch.setattr(
        browser_export,
        "load_point_cloud_xyz_rgb",
        lambda _path: (
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([[0.25, 0.5, 0.75]]),
        ),
    )
    monkeypatch.setattr(
        browser_export,
        "_write_browser_frame_atlases",
        lambda _bundle, _path: {"cam04": "./cam04.png", "cam09": "./cam09.png", "cam06": "./cam06.png"},
    )
    monkeypatch.setattr(browser_export, "_browser_camera_rows", lambda *_args, **_kwargs: [])

    browser_export.export_browser_multicam_dataset_bundle(
        manifest_path=tmp_path / "manifest.jsonl",
        sample_id="sample",
        split="train2_holdout1",
        seed_point_cloud_path=seed_path,
        output_path=output_path,
        target_size=(2, 2),
        frame_indices=[0],
        seed_count=1,
        seed_provenance_report_path=report_path,
    )

    assert json.loads(output_path.read_text(encoding="utf-8"))["seed_provenance"] == {
        "method": "colmap_sfm",
        "source_report": str(report_path),
        "source_path": str(seed_path),
        "input_cameras": ["cam04", "cam09"],
        "train_only_verified": True,
        "coordinate_frame": "model",
    }


def test_browser_export_rejects_raw_sparse_seed_before_decoding_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        browser_export,
        "load_point_cloud_xyz_rgb",
        lambda _path: (torch.zeros((1, 3)), torch.zeros((1, 3))),
    )
    monkeypatch.setattr(
        browser_export,
        "load_multicam_video_bundle",
        lambda **_kwargs: pytest.fail("video bundle should not load for an impossible raw seed count"),
    )

    with pytest.raises(ValueError, match=r"only 1 points before visibility filtering.*requested 2"):
        browser_export.export_browser_multicam_dataset_bundle(
            manifest_path=tmp_path / "manifest.jsonl",
            sample_id="sample",
            split="train2_holdout1",
            seed_point_cloud_path=tmp_path / "sparse.ply",
            output_path=tmp_path / "bundle.json",
            target_size=(2, 2),
            frame_indices=[0],
            seed_count=2,
            allow_unverified_seed_provenance=True,
        )


def test_browser_seed_provenance_rejects_heldout_camera_input(tmp_path: Path) -> None:
    report_path = tmp_path / "leaky_seed_report.json"
    report_path.write_text(
        json.dumps(
            {
                "method": "colmap_sfm",
                "input_cameras": ["cam04", "cam06"],
                "train_only_verified": True,
                "coordinate_frame": "model",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"overlap canonical heldout cameras.*cam06"):
        _resolve_seed_provenance(
            report_path=report_path,
            seed_point_cloud_path=tmp_path / "sparse.ply",
            train_cameras=["cam04", "cam09"],
            heldout_cameras=["cam06"],
            allow_unverified=False,
        )


def test_browser_seed_provenance_allows_explicit_unverified_external_without_train_only_claim(
    tmp_path: Path,
) -> None:
    seed_path = tmp_path / "external_ex4dgs" / "input.ply"

    with pytest.raises(ValueError, match="requires --seed-provenance-report"):
        _resolve_seed_provenance(
            report_path=None,
            seed_point_cloud_path=seed_path,
            train_cameras=["cam04", "cam09"],
            heldout_cameras=["cam06"],
            allow_unverified=False,
        )

    assert _resolve_seed_provenance(
        report_path=None,
        seed_point_cloud_path=seed_path,
        train_cameras=["cam04", "cam09"],
        heldout_cameras=["cam06"],
        allow_unverified=True,
    ) == {
        "method": "external_unverified",
        "source_report": None,
        "source_path": str(seed_path),
        "input_cameras": [],
        "train_only_verified": False,
        "coordinate_frame": "world",
    }


def test_browser_seed_coordinates_distinguish_world_from_known_pose_model_frame() -> None:
    anchor_c2w = torch.eye(4)
    anchor_c2w[:3, 3] = torch.tensor([10.0, 20.0, 30.0])
    bundle = SimpleNamespace(
        metadata={"anchor_camera": "cam04"},
        anchor_c2w=anchor_c2w,
    )
    points = torch.tensor([[11.0, 22.0, 33.0]])

    world_points = _seed_points_in_anchor_frame(
        points,
        bundle=bundle,
        seed_provenance={"coordinate_frame": "world"},
    )
    model_points = _seed_points_in_anchor_frame(
        points,
        bundle=bundle,
        seed_provenance={"coordinate_frame": "model"},
    )

    torch.testing.assert_close(world_points, torch.tensor([[1.0, 2.0, 3.0]]))
    torch.testing.assert_close(model_points, points)


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
    assert payload["dataset_contract"]["pose_source"] == "neural_3d_llff_opencv_relative_pinhole_v2"
    assert payload["decode_size"] == [96, 72]
    assert payload["frame_count"] == 16
    assert payload["frame_indices"] == [
        0, 20, 40, 60, 80, 100, 120, 140, 159, 179, 199, 219, 239, 259, 279, 299,
    ]
    assert len(train_cameras) == 17
    assert [camera["name"] for camera in heldout_cameras] == ["cam06"]
    assert len(payload["seed_points_xyzrgb"]) == 4096
    assert payload["seed_coordinate_frame"] == "cam04_opencv"
    assert payload["seed_provenance"] == {
        "method": "external_unverified",
        "source_report": None,
        "source_path": "data/external/ex4dgs_pretrained/extracted/coffee_martini/input.ply",
        "input_cameras": [],
        "train_only_verified": False,
        "coordinate_frame": "world",
    }

    for camera in payload["cameras"]:
        assert "video_url" not in camera
        atlas_path = TRAIN17_BROWSER_BUNDLE.parent / camera["frame_atlas_url"].removeprefix("./")
        with Image.open(atlas_path) as atlas:
            assert atlas.size == (96 * 16, 72)
