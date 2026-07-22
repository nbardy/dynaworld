from __future__ import annotations

import json
from pathlib import Path

import pytest

from config_utils import load_config_file
from paper_training_protocol import resolve_paper_training_protocol
from research_experiments.paper_runner_suite.run_unified_paper_ablation import (
    DEFAULT_PROTOCOL,
    build_lane_evidence,
    build_dry_run_manifest,
    comparison_command,
    comparison_lane_commands,
    kernel_specs,
    load_final_powerfoam_metrics,
    local_mps_safety_estimate,
    materialize_isolated_comparison_report,
    merge_comparison_lane_reports,
    paper_camera_rig_init,
    paper_world_tubes_camera_policy,
    paper_scene_tag,
    powerfoam_config,
    require_execution_safety_acknowledgement,
    require_clean_provenance,
    source_provenance,
    validate_lane_cost,
    validate_manifest,
    worldfoam_lane_command,
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


def _isolated_reports() -> dict:
    meta = {
        "baseline_config": "baseline.jsonc",
        "target_size": [96, 128],
        "image_size": [96, 128],
        "max_frames": 4,
        "frame_count": 4,
        "train_seconds": 1.0,
        "device": "mps",
        "seed": 17,
        "train_cameras": ["cam04", "cam09"],
        "heldout_cameras": ["cam06"],
        "pose_source": "dataset",
        "uvt_camera_projection": "dataset_lens",
        "uvt_camera_sequence_mode": "static_view",
        "uvt_segment_frames": 4,
        "uvt_backward_policy": {"name": "fast_exploration"},
        "splat_camera_projection": "dataset_lens",
        "eval_chunk_frames": 2,
        "eval_media_max_frames": 32,
    }
    return {
        "world_tubes": {
            "meta": {**meta, "only_lane": "world_tubes"},
            "star_uvt": {"lane": "world_tubes"},
            "star_uvt_selected": {"checkpoint": "final"},
            "free_dynamic_splats": None,
        },
        "dynamic_3dgs": {
            "meta": {**meta, "only_lane": "dynamic_3dgs"},
            "star_uvt": None,
            "star_uvt_selected": None,
            "free_dynamic_splats": {"lane": "dynamic_3dgs"},
        },
    }


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
    assert _value_after(command, "--eval-chunk-frames") == "2"
    assert _value_after(command, "--eval-media-max-frames") == "32"
    assert _value_after(command, "--only-lane") == "combined"
    assert "--allow-paper-local-mps-execution" not in command


def test_unified_commands_isolate_each_allocator_and_worldfoam_inherits_device(tmp_path: Path) -> None:
    _, protocol = _protocol()
    compare_dir = tmp_path / "compare"
    commands = comparison_lane_commands(
        SMOKE_PROTOCOL,
        protocol,
        29,
        compare_dir,
        backward_policy="fast_exploration",
        device="cpu",
        python="python",
    )

    assert set(commands) == {"world_tubes", "dynamic_3dgs"}
    for lane_name, command in commands.items():
        assert _value_after(command, "--only-lane") == lane_name
        assert _value_after(command, "--out-dir") == str(compare_dir / lane_name)
        assert _value_after(command, "--device") == "cpu"

    worldfoam = worldfoam_lane_command(
        SMOKE_PROTOCOL,
        29,
        tmp_path / "worldfoam",
        device="cpu",
        wandb_mode="offline",
        python="python",
    )
    assert "--execute" in worldfoam
    assert _value_after(worldfoam, "--device") == "cpu"
    assert "--allow-local-mps-execution" not in worldfoam

    approved = comparison_command(
        SMOKE_PROTOCOL,
        protocol,
        29,
        compare_dir,
        backward_policy="fast_exploration",
        device="mps",
        only_lane="world_tubes",
        allow_local_mps_execution=True,
        python="python",
    )
    assert "--allow-paper-local-mps-execution" in approved


def test_isolated_comparison_reports_merge_only_when_metadata_matches() -> None:
    reports = _isolated_reports()

    merged = merge_comparison_lane_reports(reports)
    assert merged["star_uvt"]["lane"] == "world_tubes"
    assert merged["free_dynamic_splats"]["lane"] == "dynamic_3dgs"
    assert merged["meta"]["execution_model"] == "one_child_process_per_representation"

    reports["dynamic_3dgs"]["meta"]["seed"] = 29
    with pytest.raises(ValueError, match="metadata drifted: seed"):
        merge_comparison_lane_reports(reports)


def test_isolated_comparison_resume_reuses_completed_lane_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, protocol = _protocol()
    comparison_dir = tmp_path / "compare"
    for lane_name, report in _isolated_reports().items():
        lane_dir = comparison_dir / lane_name
        lane_dir.mkdir(parents=True)
        (lane_dir / "comparison_report.json").write_text(json.dumps(report), encoding="utf-8")

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("completed isolated lane must not be relaunched")

    monkeypatch.setattr("subprocess.run", unexpected_run)
    report_path = materialize_isolated_comparison_report(
        SMOKE_PROTOCOL,
        protocol,
        17,
        comparison_dir,
        backward_policy="fast_exploration",
        device="mps",
        reuse_existing=True,
        python="python",
    )

    merged = json.loads(report_path.read_text(encoding="utf-8"))
    assert merged["meta"]["only_lane"] == "isolated_merged"
    assert merged["star_uvt"]["lane"] == "world_tubes"
    assert merged["free_dynamic_splats"]["lane"] == "dynamic_3dgs"


def test_unified_powerfoam_config_uses_the_same_protocol(tmp_path: Path) -> None:
    raw, protocol = _protocol()
    cfg = powerfoam_config(raw, protocol, 43, tmp_path / "worldfoam", wandb_mode="offline")

    assert cfg["data"]["max_frames"] == 4
    assert cfg["data"]["multicam_train_cameras"] == ["cam04", "cam09"]
    assert cfg["data"]["multicam_heldout_camera"] == "cam06"
    assert cfg["render"]["image_size"] == [96, 128]
    assert cfg["model"]["cells"] == 256
    assert cfg["train"]["steps"] == 2
    assert cfg["train"]["device"] == "mps"
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


def test_dnerf_protocol_routes_both_trainers_through_the_posed_trajectory_adapter(tmp_path: Path) -> None:
    path = (
        ROOT
        / "src"
        / "train_configs"
        / "paper_protocols"
        / "dnerf_bouncingballs_matched_20f_progressive_512_v1.jsonc"
    )
    raw = load_config_file(path)
    protocol = resolve_paper_training_protocol(raw)
    command = comparison_command(
        path,
        protocol,
        17,
        tmp_path / "compare",
        backward_policy="fast_exploration",
        device="mps",
        python="python",
    )
    cfg = powerfoam_config(
        raw,
        protocol,
        17,
        tmp_path / "worldfoam",
        wandb_mode="offline",
        worldfoam_initializer="video",
    )

    assert paper_camera_rig_init(protocol) == "dnerf"
    assert _value_after(command, "--camera-rig-init") == "dnerf"
    assert _value_after(command, "--uvt-camera-projection") == "legacy_pinhole"
    assert paper_world_tubes_camera_policy(protocol) == ("legacy_pinhole", "segmented", 1)
    assert _value_after(command, "--uvt-camera-sequence-mode") == "segmented"
    assert _value_after(command, "--uvt-segment-frames") == "1"
    assert cfg["camera"]["rig_init"] == "dnerf"
    assert cfg["data"]["multicam_train_cameras"] == ["train_trajectory"]
    assert cfg["data"]["multicam_heldout_camera"] == "test_trajectory"


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


def test_local_mps_execution_is_fail_closed_and_full_protocol_needs_second_acknowledgement() -> None:
    _, smoke = _protocol()
    full_raw = load_config_file(DEFAULT_PROTOCOL)
    full = resolve_paper_training_protocol(full_raw)

    assert local_mps_safety_estimate(smoke)["high_risk"] is False
    assert local_mps_safety_estimate(full)["high_risk"] is True
    with pytest.raises(RuntimeError, match="allow-local-mps-execution"):
        require_execution_safety_acknowledgement(
            smoke,
            device="mps",
            allow_local_mps_execution=False,
            allow_high_risk_local_mps=False,
        )
    with pytest.raises(RuntimeError, match="allow-high-risk-local-mps"):
        require_execution_safety_acknowledgement(
            full,
            device="mps",
            allow_local_mps_execution=True,
            allow_high_risk_local_mps=False,
        )
    estimate = require_execution_safety_acknowledgement(
        full,
        device="cpu",
        allow_local_mps_execution=False,
        allow_high_risk_local_mps=False,
    )
    assert estimate["estimated_peak_bytes"] > estimate["safety_limit_bytes"]


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
    assert set(manifest["comparison_lane_commands"]) == {"world_tubes", "dynamic_3dgs"}
    assert "--execute" in manifest["worldfoam_lane_command"]
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
                        "projected_trace_count": 256,
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
    assert evidence["diagnostics"]["compiled_trace_count_mean"] == 256

    del lane["metrics"]["heldout_eval_lpips"]
    with pytest.raises(ValueError, match="heldout_eval_lpips"):
        build_lane_evidence("world_tubes", lane, frame_count=4)
