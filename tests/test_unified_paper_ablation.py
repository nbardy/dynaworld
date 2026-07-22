from __future__ import annotations

from pathlib import Path

import pytest

from config_utils import load_config_file
from paper_training_protocol import resolve_paper_training_protocol
from research_experiments.paper_runner_suite.run_unified_paper_ablation import (
    DEFAULT_PROTOCOL,
    build_lane_evidence,
    build_dry_run_manifest,
    comparison_command,
    kernel_specs,
    load_final_powerfoam_metrics,
    paper_scene_tag,
    powerfoam_config,
    require_clean_provenance,
    source_provenance,
    validate_lane_cost,
    validate_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
SMOKE_PROTOCOL = (
    ROOT
    / "src"
    / "train_configs"
    / "paper_protocols"
    / "coffee_martini_protocol_smoke_2step.jsonc"
)


def _protocol(path: Path = SMOKE_PROTOCOL):
    raw = load_config_file(path)
    return raw, resolve_paper_training_protocol(raw)


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_unified_command_selects_the_practical_metal_lanes(tmp_path: Path) -> None:
    _, protocol = _protocol()
    command = comparison_command(
        SMOKE_PROTOCOL,
        protocol,
        29,
        tmp_path / "compare",
        backward_policy="fast_exploration",
        device="mps",
        python="python",
    )

    assert _value_after(command, "--paper-protocol") == str(SMOKE_PROTOCOL)
    assert _value_after(command, "--uvt-loss-scope") == "paper_batch"
    assert _value_after(command, "--uvt-backward-policy") == "fast_exploration"
    assert _value_after(command, "--uvt-render-backend") == "metal_tile"
    assert _value_after(command, "--splat-renderer") == "fast_mac"
    assert _value_after(command, "--max-frames") == "4"
    assert _value_after(command, "--max-steps") == "2"
    assert _value_after(command, "--uvt-tubes") == "256"
    assert _value_after(command, "--splat-count") == "256"


def test_unified_powerfoam_config_uses_the_same_protocol(tmp_path: Path) -> None:
    raw, protocol = _protocol()
    cfg = powerfoam_config(raw, protocol, 43, tmp_path / "worldfoam", wandb_mode="offline")

    assert cfg["data"]["max_frames"] == 4
    assert cfg["data"]["multicam_train_cameras"] == ["cam04", "cam09"]
    assert cfg["data"]["multicam_heldout_camera"] == "cam06"
    assert cfg["render"]["image_size"] == [96, 128]
    assert cfg["model"]["cells"] == 256
    assert cfg["train"]["steps"] == 2
    assert cfg["logging"]["image_log_every"] == 2
    assert cfg["logging"]["video_log_every"] == 2
    assert cfg["paper_protocol"] == raw
    assert cfg["logging"]["wandb_enabled"] is True
    assert cfg["logging"]["wandb_mode"] == "offline"
    assert cfg["logging"]["wandb_resume"] == "allow"
    assert "scene-neural3d_coffee_martini" in cfg["logging"]["wandb_tags"]


def test_paper_scene_tag_is_derived_from_each_protocol() -> None:
    _, coffee = _protocol()
    raw = load_config_file(
        ROOT / "src" / "train_configs" / "paper_protocols" / "cook_spinach_full_300f_progressive_512_v1.jsonc"
    )
    spinach = resolve_paper_training_protocol(raw)

    assert paper_scene_tag(coffee) == "scene-neural3d_coffee_martini"
    assert paper_scene_tag(spinach) == "scene-neural3d_cook_spinach"


def test_worldfoam_paper_metrics_are_from_the_final_checkpoint(tmp_path: Path) -> None:
    history = tmp_path / "eval_metrics_history.jsonl"
    history.write_text(
        '{"step":0,"metrics":{"heldout_eval_psnr":9.0}}\n'
        '{"step":600,"metrics":{"heldout_eval_psnr":8.0,"heldout_eval_lpips":0.3}}\n',
        encoding="utf-8",
    )

    metrics = load_final_powerfoam_metrics(history, expected_step=600)

    assert metrics == {"heldout_eval_psnr": 8.0, "heldout_eval_lpips": 0.3}
    with pytest.raises(ValueError, match="no evaluation at final step"):
        load_final_powerfoam_metrics(history, expected_step=601)


def test_worldfoam_initializer_cannot_leak_coffee_geometry_into_breadth_rows(tmp_path: Path) -> None:
    raw, protocol = _protocol()
    cfg = powerfoam_config(
        raw,
        protocol,
        17,
        tmp_path / "worldfoam",
        wandb_mode="offline",
        worldfoam_initializer="video",
    )

    assert cfg["model"]["init_from_video"] is True
    assert cfg["model"]["init_point_cloud_path"] is None
    with pytest.raises(FileNotFoundError, match="initializer does not exist"):
        powerfoam_config(
            raw,
            protocol,
            17,
            tmp_path / "missing",
            wandb_mode="offline",
            worldfoam_initializer=str(tmp_path / "missing.ply"),
        )


def test_submission_source_gate_records_both_repository_revisions() -> None:
    provenance = source_provenance()

    assert len(provenance["repository_commit"]) == 40
    assert len(provenance["star_uvt_commit"]) == 40
    with pytest.raises(RuntimeError, match="repository_dirty"):
        require_clean_provenance(
            {
                "repository_dirty": True,
                "star_uvt_dirty": False,
            }
        )


def test_checked_in_full_protocol_manifest_is_all_300_frames() -> None:
    raw = load_config_file(DEFAULT_PROTOCOL)
    protocol = resolve_paper_training_protocol(raw)
    validation = validate_manifest(protocol)

    assert protocol.dataset.frame_count == 300
    assert protocol.dataset.samples_per_epoch == 600
    assert protocol.nominal_epoch_coverage == 4.0
    assert all(validation["checks"].values())
    assert validation["duration_seconds"] == 10.0
    assert validation["source_image_size"] == [2028, 2704]


def test_dry_run_declares_costs_kernels_and_artifacts(tmp_path: Path) -> None:
    raw, protocol = _protocol()
    manifest = build_dry_run_manifest(
        SMOKE_PROTOCOL,
        raw,
        protocol,
        seed=17,
        out_dir=tmp_path,
        backward_policy="fast_exploration",
        device="mps",
        wandb_mode="offline",
    )

    assert manifest["status"] == "dry_run"
    assert manifest["protocol"]["target_frame_budget"] == 4
    assert manifest["protocol"]["target_pixel_budget"] == 30_720
    assert manifest["kernels"]["world_tubes"]["forward"] == "metal_tile_selected_time"
    assert manifest["kernels"]["world_tubes"]["backward"] == "direct_atomic+index_add"
    assert manifest["kernels"]["worldfoam"]["forward"] == "raytrace"
    assert manifest["kernels"]["dynamic_3dgs"]["forward"] == "fast_mac"
    assert manifest["expected_artifacts"]["run_summary"].endswith("run_summary.json")


def test_cost_validator_keeps_target_budget_separate_from_extra_rasterization() -> None:
    _, protocol = _protocol()
    lane = {
        "steps": 2,
        "paper_protocol": {
            "enabled": True,
            "cost": {
                "optimizer_steps": 2,
                "target_frames": 4,
                "target_pixels": 30_720,
                "rasterized_frames": 16,
            },
        },
    }

    validate_lane_cost("world_tubes", lane, protocol)
    assert lane["paper_protocol"]["cost"]["rasterized_frames"] > protocol.target_frame_budget
    lane["paper_protocol"]["cost"]["target_pixels"] = 1
    with pytest.raises(ValueError, match="target-pixel"):
        validate_lane_cost("world_tubes", lane, protocol)


def test_kernel_registry_separates_fast_and_deterministic_world_tubes() -> None:
    fast = kernel_specs("fast_exploration")["world_tubes"]
    reference = kernel_specs("deterministic_quality")["world_tubes"]

    assert fast.deterministic is False
    assert reference.deterministic is True
    assert fast.backward != reference.backward


def test_paper_evidence_is_fail_closed_and_keeps_trace_diagnostics() -> None:
    lane = {
        "tube_count": 256,
        "metrics": {
            "eval_psnr": 20.0,
            "eval_ssim": 0.8,
            "eval_l1": 0.1,
            "heldout_eval_psnr": 18.0,
            "heldout_eval_ssim": 0.7,
            "heldout_eval_l1": 0.15,
            "heldout_eval_lpips": 0.25,
        },
        "paper_protocol": {
            "cost": {
                "optimizer_steps": 2,
                "target_frames": 4,
                "rasterized_frames": 4,
                "target_pixels": 30_720,
                "rasterized_pixels": 30_720,
                "parameter_count": 100,
                "trainable_parameter_count": 100,
                "parameter_bytes": 400,
                "optimizer_state_bytes": 800,
                "serialized_checkpoint_bytes": 1_024,
                "sampled_peak_current_allocated_bytes": 2_048,
                "sampled_peak_driver_allocated_bytes": 4_096,
                "elapsed_s": 1.0,
            },
            "timing": {
                "definition": "test",
                "cold_compile_forward_s": 0.2,
                "steady_forward_s": 0.3,
                "steady_forward_calls": 1,
                "backward_s": 0.4,
                "backward_calls": 2,
                "optimizer_s": 0.1,
                "optimizer_calls": 2,
                "train_wall_s": 1.0,
            },
        },
        "metal_stats": {
            "rows": [
                {
                    "stats": {
                        "uvt_tile_tube_pairs": 20,
                        "summed_per_frame_tile_splat_pairs": 40,
                        "effective_pair_ratio_after_unstable_fallback": 0.5,
                        "unstable_tile_fraction": 0.1,
                        "overflow_tile_count": 0,
                        "metal_buffer_memory": 8192,
                    }
                }
            ]
        },
    }

    evidence = build_lane_evidence("world_tubes", lane, frame_count=4)
    assert evidence["quality"]["heldout_eval_lpips"] == 0.25
    assert evidence["cost"]["serialized_checkpoint_bytes"] == 1_024
    assert evidence["diagnostics"]["active_trace_count"] == 256

    del lane["metrics"]["heldout_eval_lpips"]
    with pytest.raises(ValueError, match="heldout_eval_lpips"):
        build_lane_evidence("world_tubes", lane, frame_count=4)
