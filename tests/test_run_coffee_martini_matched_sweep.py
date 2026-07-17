from __future__ import annotations

from pathlib import Path

from research_experiments.paper_runner_suite.run_coffee_martini_matched_sweep import (
    BASE_CONFIG,
    FRAME_COUNT,
    HELDOUT_CAMERAS,
    PRIMITIVE_COUNT,
    STEPS,
    TARGET_SIZE,
    TRAIN_CAMERAS,
    comparison_command,
    build_sweep_manifest,
    merge_seed_runs,
    powerfoam_config,
)


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_comparison_command_pins_matched_promotable_protocol(tmp_path: Path) -> None:
    command = comparison_command(29, tmp_path / "compare", python="python")

    assert _value_after(command, "--baseline-config") == str(BASE_CONFIG)
    assert _value_after(command, "--target-size") == str(TARGET_SIZE)
    assert _value_after(command, "--max-frames") == str(FRAME_COUNT)
    assert _value_after(command, "--max-steps") == str(STEPS)
    assert _value_after(command, "--uvt-tubes") == str(PRIMITIVE_COUNT)
    assert _value_after(command, "--splat-count") == str(PRIMITIVE_COUNT)
    assert _value_after(command, "--uvt-backward-policy") == "deterministic_quality"
    assert _value_after(command, "--uvt-camera-projection") == "dataset_lens"
    assert _value_after(command, "--splat-camera-projection") == "dataset_lens"
    assert _value_after(command, "--seed") == "29"
    assert _value_after(command, "--out-dir") == str(tmp_path / "compare")


def test_powerfoam_config_pins_same_split_budget_seed_and_offline_wandb(tmp_path: Path) -> None:
    cfg = powerfoam_config(43, tmp_path / "worldfoam", wandb_mode="offline")

    assert cfg["data"]["multicam_train_cameras"] == list(TRAIN_CAMERAS)
    assert [cfg["data"]["multicam_heldout_camera"]] == list(HELDOUT_CAMERAS)
    assert cfg["render"]["render_size"] == TARGET_SIZE
    assert cfg["data"]["max_frames"] == FRAME_COUNT
    assert cfg["train"]["steps"] == STEPS
    assert cfg["model"]["cells"] == PRIMITIVE_COUNT
    assert cfg["train"]["seed"] == 43
    assert cfg["logging"]["wandb_enabled"] is True
    assert cfg["logging"]["wandb_mode"] == "offline"
    assert cfg["logging"]["wandb_run_id"] == "cmwf0043"
    assert cfg["logging"]["output_dir"] == str(tmp_path / "worldfoam")
    assert "feature_triangulation" in cfg["model"]["init_point_cloud_path"]


def test_partial_sweep_manifest_merges_by_seed_and_new_run_wins() -> None:
    merged = merge_seed_runs(
        [{"seed": 17, "value": "old"}, {"seed": 29, "value": "keep"}],
        [{"seed": 17, "value": "new"}, {"seed": 43, "value": "add"}],
    )

    assert merged == [
        {"seed": 17, "value": "new"},
        {"seed": 29, "value": "keep"},
        {"seed": 43, "value": "add"},
    ]


def test_sweep_manifest_sorts_and_declares_shared_contract() -> None:
    manifest = build_sweep_manifest([{"seed": 43}, {"seed": 17}, {"seed": 29}], wandb_mode="offline")

    assert manifest["seeds"] == [17, 29, 43]
    assert [run["seed"] for run in manifest["runs"]] == [17, 29, 43]
    assert manifest["train_cameras"] == list(TRAIN_CAMERAS)
    assert manifest["heldout_cameras"] == list(HELDOUT_CAMERAS)
    assert manifest["target_size"] == TARGET_SIZE
    assert manifest["frame_count"] == FRAME_COUNT
    assert manifest["steps"] == STEPS
    assert manifest["primitive_count"] == PRIMITIVE_COUNT
