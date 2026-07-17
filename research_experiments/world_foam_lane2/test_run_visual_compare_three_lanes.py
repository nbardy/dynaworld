from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import run_visual_compare_three_lanes as visual_compare


def _dry_run_args(tmp_path: Path, lanes: list[str] | None = None, *, tier: str = "tiny") -> argparse.Namespace:
    return argparse.Namespace(
        dry_run=True,
        lane=lanes,
        tier=tier,
        logs_dir=str(tmp_path / "logs"),
        timeout_s=1.0,
        continue_on_failure=True,
        no_media_deps=False,
    )


def test_dry_run_default_lanes_plans_all_three_representations(tmp_path: Path) -> None:
    summary = visual_compare.build_summary(_dry_run_args(tmp_path))

    assert summary["status"] == "planned"
    assert [lane["name"] for lane in summary["lanes"]] == [
        "worldfoam_dynamic_powerfoam_metal",
        "worldtubes_star_uvt_metal",
        "dynamic_gsplat_fast_mac_metal",
    ]
    assert all(lane["command"][1:] for lane in summary["lanes"])
    assert summary["lanes"][0]["command"][:2] == ["uv", "run"]
    assert summary["tier"] == "tiny"


def test_medium_tier_uses_128px_configs_and_stable_lane_names(tmp_path: Path) -> None:
    summary = visual_compare.build_summary(_dry_run_args(tmp_path, tier="medium"))

    assert summary["status"] == "planned"
    assert summary["tier"] == "medium"
    lanes = {lane["name"]: lane for lane in summary["lanes"]}
    assert set(lanes) == {
        "worldfoam_dynamic_powerfoam_metal",
        "worldtubes_star_uvt_metal",
        "dynamic_gsplat_fast_mac_metal",
    }
    assert lanes["worldfoam_dynamic_powerfoam_metal"]["backend"]["render_size"] == 128
    assert lanes["worldtubes_star_uvt_metal"]["backend"]["render_size"] == 128
    assert lanes["dynamic_gsplat_fast_mac_metal"]["backend"]["render_size"] == 128
    assert lanes["worldfoam_dynamic_powerfoam_metal"]["config"].endswith(
        "visual_compare_worldfoam_dynamic_powerfoam_metal_128_16f_40step.jsonc"
    )


def test_capacity_tier_uses_scaled_128px_configs(tmp_path: Path) -> None:
    summary = visual_compare.build_summary(_dry_run_args(tmp_path, tier="capacity"))

    assert summary["status"] == "planned"
    assert summary["tier"] == "capacity"
    lanes = {lane["name"]: lane for lane in summary["lanes"]}
    assert lanes["worldfoam_dynamic_powerfoam_metal"]["backend"]["render_size"] == 128
    assert lanes["worldfoam_dynamic_powerfoam_metal"]["backend"]["cell_count"] == 2048
    assert lanes["worldfoam_dynamic_powerfoam_metal"]["backend"]["steps"] == 80
    assert lanes["worldtubes_star_uvt_metal"]["backend"]["tube_count"] == 2048
    assert lanes["worldtubes_star_uvt_metal"]["backend"]["steps"] == 60
    assert lanes["dynamic_gsplat_fast_mac_metal"]["backend"]["render_size"] == 128
    assert lanes["dynamic_gsplat_fast_mac_metal"]["backend"]["steps"] == 60
    assert lanes["dynamic_gsplat_fast_mac_metal"]["config"].endswith(
        "visual_compare_dynamic_gsplat_fast_mac_metal_128_16f_60step_4096gs.jsonc"
    )


def test_backend_summary_identifies_metal_routes(tmp_path: Path) -> None:
    summary = visual_compare.build_summary(_dry_run_args(tmp_path))
    lanes = {lane["name"]: lane for lane in summary["lanes"]}

    assert lanes["worldfoam_dynamic_powerfoam_metal"]["backend"]["metal_backend"] == "torch_dynamic_powerfoam_metal"
    assert lanes["worldtubes_star_uvt_metal"]["backend"]["metal_backend"] == "metal_tile"
    assert lanes["worldtubes_star_uvt_metal"]["backend"]["sample_emission_mode"] == "direct_atomic"
    assert lanes["dynamic_gsplat_fast_mac_metal"]["backend"]["renderer"] == "fast_mac"
    assert lanes["dynamic_gsplat_fast_mac_metal"]["backend"]["fast_mac_rgb_variant"] == "v6_refined"


def test_declared_artifacts_include_star_uvt_media_paths(tmp_path: Path) -> None:
    summary = visual_compare.build_summary(_dry_run_args(tmp_path, ["worldtubes_star_uvt_metal"]))
    artifacts = {artifact["label"]: artifact["path"] for artifact in summary["lanes"][0]["declared_artifacts"]}

    assert artifacts["out_json"].endswith("star_uvt_worldtubes_metal_64_16f_20step.json")
    assert artifacts["contact_sheet"].endswith("star_uvt_worldtubes_metal_64_16f_20step_contact.jpg")
    assert artifacts["side_by_side_video"].endswith("star_uvt_worldtubes_metal_64_16f_20step_sbs.mp4")


def test_declared_artifacts_include_dynamic_gsplat_output_dir_media(tmp_path: Path) -> None:
    summary = visual_compare.build_summary(_dry_run_args(tmp_path, ["dynamic_gsplat_fast_mac_metal"]))
    artifacts = {artifact["label"]: artifact["path"] for artifact in summary["lanes"][0]["declared_artifacts"]}

    output_dir = "outputs/visual_comparisons/dynamic_gsplat_fast_mac_metal_64_16f_20step"
    assert artifacts["preview_image"].endswith(f"{output_dir}/preview_step_0020.png")
    assert artifacts["render_video"].endswith(f"{output_dir}/render_step_0020.mp4")
    assert artifacts["side_by_side_video"].endswith(f"{output_dir}/side_by_side_step_0020.mp4")


def test_medium_tier_declares_128px_disk_media(tmp_path: Path) -> None:
    summary = visual_compare.build_summary(_dry_run_args(tmp_path, ["dynamic_gsplat_fast_mac_metal"], tier="medium"))
    artifacts = {artifact["label"]: artifact["path"] for artifact in summary["lanes"][0]["declared_artifacts"]}

    output_dir = "outputs/visual_comparisons/dynamic_gsplat_fast_mac_metal_128_16f_20step"
    assert artifacts["preview_image"].endswith(f"{output_dir}/preview_step_0020.png")
    assert artifacts["render_video"].endswith(f"{output_dir}/render_step_0020.mp4")
    assert artifacts["side_by_side_video"].endswith(f"{output_dir}/side_by_side_step_0020.mp4")


def test_capacity_tier_declares_scaled_disk_media(tmp_path: Path) -> None:
    summary = visual_compare.build_summary(_dry_run_args(tmp_path, ["dynamic_gsplat_fast_mac_metal"], tier="capacity"))
    artifacts = {artifact["label"]: artifact["path"] for artifact in summary["lanes"][0]["declared_artifacts"]}

    output_dir = "outputs/visual_comparisons/dynamic_gsplat_fast_mac_metal_128_16f_60step_4096gs"
    assert artifacts["preview_image"].endswith(f"{output_dir}/preview_step_0060.png")
    assert artifacts["render_video"].endswith(f"{output_dir}/render_step_0060.mp4")
    assert artifacts["side_by_side_video"].endswith(f"{output_dir}/side_by_side_step_0060.mp4")


def test_declared_artifacts_include_powerfoam_output_dir_media() -> None:
    cfg = {
        "arch": "dynamic_powerfoam_metal",
        "train": {"steps": 7},
        "logging": {
            "output_dir": "outputs/example_powerfoam",
            "always_log_last_step": True,
        },
    }

    artifacts = {
        artifact["label"]: artifact["path"]
        for artifact in visual_compare.collect_declared_artifacts(cfg)
    }

    assert artifacts["preview_image"].endswith("outputs/example_powerfoam/preview_step_0007.png")
    assert artifacts["render_video"].endswith("outputs/example_powerfoam/render_step_0007.mp4")
    assert artifacts["side_by_side_video"].endswith("outputs/example_powerfoam/side_by_side_step_0007.mp4")


def test_recent_wandb_media_collects_only_media_after_start(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(visual_compare, "PROJECT_ROOT", tmp_path)
    media_dir = tmp_path / "wandb" / "offline-run-test" / "files" / "media" / "videos"
    media_dir.mkdir(parents=True)
    keep = media_dir / "render.mp4"
    ignore = media_dir / "notes.txt"
    keep.write_bytes(b"video")
    ignore.write_text("not media", encoding="utf-8")

    rows = visual_compare.collect_recent_wandb_media(0.0)

    assert [row["path"] for row in rows] == ["wandb/offline-run-test/files/media/videos/render.mp4"]
