from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from config_utils import load_config_file
from PIL import Image

TRAIN_SCRIPTS = Path("src/train_scripts").resolve()
if str(TRAIN_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TRAIN_SCRIPTS))

import export_browser_coffee_martini_384x288 as export_lane  # noqa: E402

EXPORT_CONFIG = Path(
    "src/train_configs/browser_coffee_martini_train17_holdout1_384x288_export.jsonc"
)


def _recipe(tmp_path: Path) -> dict:
    return {
        "target_size": (288, 384),
        "frame_indices": export_lane.EXPECTED_FRAME_INDICES,
        "train_cameras": [
            "cam00",
            "cam01",
            "cam02",
            "cam04",
            "cam05",
            "cam07",
            "cam08",
            "cam09",
            "cam10",
            "cam11",
            "cam12",
            "cam13",
            "cam14",
            "cam16",
            "cam18",
            "cam19",
            "cam20",
        ],
        "heldout_cameras": ["cam06"],
        "output_directory": tmp_path / "bundle",
        "config_path": EXPORT_CONFIG.resolve(),
        "manifest_path": Path("manifest.jsonl"),
        "manifest_sha256": "manifest",
        "seed_point_cloud_path": Path("seed.ply"),
        "point_cloud_sha256": "seed",
        "seed_report_path": Path("seed.json"),
        "thresholds": {
            "minimum_available_memory_bytes": 4 * export_lane.GIB,
            "minimum_free_memory_fraction": 0.25,
            "maximum_swap_used_fraction": 0.75,
            "maximum_load_5m_per_logical_cpu": 0.75,
            "minimum_free_disk_bytes": 2 * export_lane.GIB,
            "working_set_headroom_multiplier": 2.0,
        },
    }


def _safe_host() -> dict:
    return {
        "physical_memory_bytes": 24 * export_lane.GIB,
        "free_memory_fraction": 0.5,
        "available_memory_bytes": 12 * export_lane.GIB,
        "swap_total_bytes": 8 * export_lane.GIB,
        "swap_used_bytes": 2 * export_lane.GIB,
        "swap_used_fraction": 0.25,
        "free_disk_bytes": 20 * export_lane.GIB,
        "logical_cpu_count": 10,
        "load_1m_per_logical_cpu": 0.2,
        "load_5m_per_logical_cpu": 0.2,
        "load_15m_per_logical_cpu": 0.2,
    }


def test_checked_in_384_export_recipe_is_exact_and_separate() -> None:
    config = load_config_file(EXPORT_CONFIG)

    assert config["version"] == export_lane.RECIPE_VERSION
    assert config["dataset"]["target_size"] == [288, 384]
    assert len(config["dataset"]["expected_manifest_sha256"]) == 64
    assert config["dataset"]["frame_indices"] == export_lane.EXPECTED_FRAME_INDICES
    assert config["dataset"]["expected_source_frame_count"] == 300
    assert config["dataset"]["expected_train_camera_count"] == 17
    assert config["dataset"]["expected_heldout_cameras"] == ["cam06"]
    assert config["dataset"]["expected_anchor_camera"] == "cam04"
    assert config["dataset"]["sparse_frame_decode"] is True
    assert config["seed"]["count"] == 768
    assert len(config["seed"]["expected_point_cloud_sha256"]) == 64
    assert "384x288_verified_sparse" in config["output"]["directory"]
    assert "384x288_verified_sparse" in config["output"]["bundle_filename"]


def test_export_recipe_validation_fails_closed_on_manifest_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_video = tmp_path / "camera.mp4"
    source_video.write_bytes(b"fixture")
    seed_cloud = tmp_path / "seed.ply"
    seed_report = tmp_path / "seed.json"
    seed_cloud.write_bytes(b"fixture")
    seed_report.write_text("{}", encoding="utf-8")
    config = load_config_file(EXPORT_CONFIG)
    config["dataset"]["manifest"] = str(tmp_path / "manifest.jsonl")
    config["seed"]["point_cloud"] = str(seed_cloud)
    config["seed"]["provenance_report"] = str(seed_report)
    (tmp_path / "manifest.jsonl").write_text("{}\n", encoding="utf-8")
    config_path = tmp_path / "recipe.json"
    config_path.write_text(__import__("json").dumps(config), encoding="utf-8")

    monkeypatch.setattr(
        export_lane,
        "select_multicam_record",
        lambda _cfg: {
            "frame_count": 300,
            "train_cameras": ["cam04"] * 17,
            "heldout_cameras": ["cam07"],
            "anchor_camera": "cam04",
        },
    )
    monkeypatch.setattr(
        export_lane,
        "_sha256",
        lambda path: (
            config["dataset"]["expected_manifest_sha256"]
            if path.name == "manifest.jsonl"
            else config["seed"]["expected_point_cloud_sha256"]
        ),
    )
    monkeypatch.setattr(export_lane, "video_path_for_camera", lambda *_args: source_video)
    monkeypatch.setattr(
        export_lane,
        "_resolve_seed_provenance",
        lambda **_kwargs: {"train_only_verified": True},
    )
    monkeypatch.setattr(
        export_lane,
        "load_point_cloud_xyz_rgb",
        lambda _path: (torch.zeros((815, 3)), torch.zeros((815, 3))),
    )

    with pytest.raises(ValueError, match="manifest drifted"):
        export_lane.load_export_recipe(config_path)


def test_384_export_preflight_blocks_pressure_disk_and_existing_output(tmp_path: Path) -> None:
    recipe = _recipe(tmp_path)
    safe = export_lane.evaluate_preflight(recipe, _safe_host(), overwrite=False)
    assert safe["status"] == "safe"
    assert safe["failures"] == []
    assert safe["estimates"]["decoded_rgb_f32_bytes"] == 18 * 16 * 288 * 384 * 3 * 4
    assert safe["estimates"]["legacy_eager_rgb_f32_bytes"] == 18 * 300 * 288 * 384 * 3 * 4

    contended_host = _safe_host()
    contended_host["swap_used_fraction"] = 0.9
    contended = export_lane.evaluate_preflight(recipe, contended_host, overwrite=False)
    assert contended["status"] == "blocked"
    assert contended["failures"] == ["swap_used_fraction"]

    low_disk_host = _safe_host()
    low_disk_host["free_disk_bytes"] = 1
    low_disk = export_lane.evaluate_preflight(recipe, low_disk_host, overwrite=False)
    assert low_disk["status"] == "blocked"
    assert low_disk["failures"] == ["free_disk_bytes"]

    loaded_host = _safe_host()
    loaded_host["load_5m_per_logical_cpu"] = 0.9
    loaded = export_lane.evaluate_preflight(recipe, loaded_host, overwrite=False)
    assert loaded["status"] == "blocked"
    assert loaded["failures"] == ["load_5m_per_logical_cpu"]

    recipe["output_directory"].mkdir()
    (recipe["output_directory"] / "partial.json").write_text("{}", encoding="utf-8")
    collision = export_lane.evaluate_preflight(recipe, _safe_host(), overwrite=False)
    assert collision["status"] == "blocked"
    assert collision["failures"] == ["output_directory_not_empty"]
    overwrite = export_lane.evaluate_preflight(recipe, _safe_host(), overwrite=True)
    assert overwrite["status"] == "safe"

    file_recipe = _recipe(tmp_path / "file-collision")
    file_recipe["output_directory"].parent.mkdir(parents=True)
    file_recipe["output_directory"].write_text("occupied", encoding="utf-8")
    file_collision = export_lane.evaluate_preflight(
        file_recipe,
        _safe_host(),
        overwrite=False,
    )
    assert file_collision["status"] == "blocked"
    assert file_collision["failures"] == ["output_directory_not_empty"]


def test_capture_host_resources_fails_closed_when_pressure_is_unparseable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter(["25769803776\n", "unexpected output\n"])
    monkeypatch.setattr(export_lane, "_command_output", lambda _command: next(outputs))

    with pytest.raises(RuntimeError, match="free-memory percentage"):
        export_lane.capture_host_resources(tmp_path)


def test_generated_bundle_verifier_checks_real_intrinsics_and_atlas_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = _recipe(tmp_path)
    recipe.update(
        {
            "record": {},
            "seed_count": 1,
        }
    )
    cameras = [
        {
            "role": role,
            "name": camera,
            "frame_atlas_url": f"./bundle_{camera}.png",
            "intrinsics": [0.5, 2.0 / 3.0, 0.5, 0.5],
        }
        for role, camera in [
            *(("train", camera) for camera in recipe["train_cameras"]),
            ("heldout", "cam06"),
        ]
    ]
    for camera in cameras:
        Image.new("RGB", (384 * 16, 288)).save(
            tmp_path / camera["frame_atlas_url"].removeprefix("./")
        )
    payload = {
        "version": export_lane.BROWSER_MULTICAM_BUNDLE_VERSION,
        "decode_size": [384, 288],
        "frame_indices": export_lane.EXPECTED_FRAME_INDICES,
        "dataset_contract": {"frame_decode": "sparse_exact"},
        "seed_points_xyzrgb": [[0.0] * 6],
        "seed_provenance": {
            "train_only_verified": True,
            "input_cameras": ["cam04"],
        },
        "cameras": cameras,
    }
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        export_lane,
        "neural_3d_camera_from_poses_bounds",
        lambda *_args, **_kwargs: (
            torch.tensor(
                [[192.0, 0.0, 192.0], [0.0, 192.0, 144.0], [0.0, 0.0, 1.0]]
            ),
            torch.eye(4),
        ),
    )

    result = export_lane.verify_generated_bundle(recipe, bundle_path)

    assert result["status"] == "verified"
    assert result["decode_size"] == [384, 288]
    assert result["train_camera_count"] == 17
