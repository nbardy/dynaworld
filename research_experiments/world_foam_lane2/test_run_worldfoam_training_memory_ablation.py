from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import run_worldfoam_memory_scaling_acceptance as shared
import run_worldfoam_training_memory_ablation as producer
import verify_worldfoam_training_memory_ablation as verifier


ROOT = Path(__file__).resolve().parents[2]


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        execute=False,
        backend="mps",
        config=producer.DEFAULT_CONFIG,
        contract=producer.DEFAULT_CONTRACT,
        output=tmp_path / "must-not-exist.json",
        driver_module=producer.DEFAULT_DRIVER_MODULE,
        minimum_free_disk_bytes=shared.DEFAULT_MINIMUM_FREE_DISK_BYTES,
        minimum_available_memory_bytes=shared.DEFAULT_MINIMUM_AVAILABLE_MEMORY_BYTES,
        maximum_swap_used_bytes=shared.DEFAULT_MAXIMUM_SWAP_USED_BYTES,
        maximum_load_average=shared.DEFAULT_MAXIMUM_LOAD_AVERAGE,
    )


def test_paper_training_config_is_distinct_exact_and_deterministic() -> None:
    config = verifier.load_json_object(producer.DEFAULT_CONFIG)
    contract = verifier.load_json_object(producer.DEFAULT_CONTRACT)
    verifier.validate_config(config)
    verifier.validate_contract(contract)

    assert config["procedural_world"]["rows"] == 32
    assert config["procedural_world"]["columns"] == 32
    assert config["procedural_world"]["site_count"] == 1024
    assert config["image"] == {"height": 384, "width": 512}
    assert config["temporal_grid"]["dataset_frame_count"] == 300
    assert config["temporal_grid"]["requested_frame_counts"] == [8, 64, 300]
    assert config["track_manifest"]["track_count"] == 512
    assert config["track_manifest"]["identical_tracks_across_requested_frame_counts"]
    assert config["target_source"]["direct_selected_pixel_only"] is True
    assert config["target_source"]["full_frame_materialization"] is False
    assert config["compiler"]["certification_mode"] == "all_competitor_active_owner"
    assert config["compiler"]["maximum_sites_per_track_compile"] == 1024
    assert config["compiler"]["heuristic_spatial_culling_allowed"] is False
    assert config["spatial_streaming"]["maximum_active_sites_per_device_block"] == 64
    assert "maximum_sites_per_source_block" not in config["spatial_streaming"]
    assert config["optimizer"]["gradient_clipping"] == "none"
    assert "gradient_clip_norm" not in config["optimizer"]
    assert config["state_accounting"]["combined_live_state_bytes_per_site"] == 112
    assert config["state_accounting"]["combined_checkpoint_bytes_per_site"] == 80
    assert config["state_accounting"]["live_state_plus_checkpoint_bytes_per_site"] == 192
    assert (
        config["state_accounting"][
            "live_state_plus_checkpoint_payload_clone_peak_bytes_per_site"
        ]
        == 272
    )
    assert config["ablation"]["fused_frame_counts"] == [8, 64, 300]
    assert config["ablation"]["staged_frame_counts"] == [8]
    assert len(producer.planned_row_keys(config)) == 12
    assert config["ablation"]["control_frame_counts"] == [8, 64, 300]
    assert config["ablation"]["control_mode"] == "per_frame_replay_sequential"
    assert config["ablation"]["control_measured_required_frame_counts"] == [8, 64, 300]
    assert config["ablation"]["control_censorable_frame_counts"] == []
    assert len(producer.planned_control_row_keys(config)) == 9

    mechanical = verifier.load_json_object(
        Path(__file__).with_name("worldfoam_memory_scaling_mps_trial_v1.json")
    )
    mechanical_contract = verifier.load_json_object(
        Path(__file__).with_name("worldfoam_memory_scaling_acceptance_v3.json")
    )
    assert len(mechanical["scene"]["positions0"]) == 2
    assert mechanical_contract["contract_id"] == "worldfoam-fixed-site-material-memory-scaling-v3"
    assert config["config_id"] != mechanical["dataset_generation_id"]


def test_plan_reuses_incident_calibrated_guards_and_is_explicitly_blocked(
    tmp_path: Path,
) -> None:
    assert producer.HOST_GUARD is shared._guard_host
    assert producer.RESOURCE_POLICY_VALIDATOR is shared._validate_resource_policy
    assert producer.MPS_MEMORY_SAMPLER is shared._MpsMemorySampler
    assert producer.PARENT_WATCHDOG is shared._run_guarded_worker

    plan = producer.make_plan(_args(tmp_path))
    assert plan["status"] == "blocked"
    assert plan["execution_ready"] is False
    assert plan["artifact_written"] is False
    assert plan["evidence_rows_emitted"] == 0
    assert plan["control_evidence_rows_emitted"] == 0
    assert plan["paper_claim_permitted"] is False
    assert plan["mechanical_two_site_fixture_reused"] is False
    assert plan["planned_row_count"] == 12
    assert plan["planned_control_row_count"] == 9
    assert plan["planned_total_process_count"] == 24
    assert plan["fixed_track_count"] == 512
    assert plan["all_competitor_active_owner_certification_required"] is True
    assert plan["heuristic_spatial_culling_permitted"] is False
    assert plan["full_frame_target_materialization_permitted"] is False
    assert set(plan["blocking_reasons"]) == {
        "native_extension_older_than_bound_native_sources",
    }
    assert not (tmp_path / "must-not-exist.json").exists()

    manifest, digest, native_digest = producer.build_source_manifest()
    labels = {record["path"] for record in manifest}
    assert verifier.REQUIRED_MANIFEST_PATHS <= labels
    assert "research_experiments/world_foam_lane2/worldfoam_memory_scaling_mps_trial_v1.json" not in labels
    assert digest == verifier.source_manifest_sha256(manifest)
    assert len(native_digest) == 64


def test_fused_compiled_framewise_f8_parity_binds_both_measured_rows() -> None:
    payload = {
        "loss": 0.25,
        "material_gradient": (1.0, 2.0),
        "geometry_gradient": (3.0, 4.0, 5.0),
        "parameters_after_step": (6.0, 7.0),
    }
    record = producer._fused_compiled_framewise_parity_record(
        repeat_index=2,
        fused_receipt={
            "parity_payload": payload,
            "row": {"evidence_sha256": "f" * 64},
        },
        control_receipt={
            "parity_payload": dict(payload),
            "row": {"evidence_sha256": "c" * 64},
        },
    )
    assert record == {
        "repeat_index": 2,
        "fused_row_evidence_sha256": "f" * 64,
        "compiled_framewise_control_row_evidence_sha256": "c" * 64,
        "loss_absolute_error": 0.0,
        "material_gradient_relative_l2": 0.0,
        "geometry_gradient_relative_l2": 0.0,
        "parameter_relative_l2": 0.0,
    }


def test_lifecycle_attachment_binds_primary_step_but_keeps_scaling_measurement_clean() -> None:
    delta_keys = (
        "raw_color_parameter_delta_l2_norm",
        "raw_density_parameter_delta_l2_norm",
        "positions0_parameter_delta_l2_norm",
        "velocities_parameter_delta_l2_norm",
        "weight_coefficients_parameter_delta_l2_norm",
    )
    deltas = {key: float(index + 1) for index, key in enumerate(delta_keys)}
    primary_parity = {
        "loss": 0.5,
        "material_gradient": [1.0, 2.0],
        "geometry_gradient": [3.0, 4.0, 5.0],
        "parameters_after_step": [6.0, 7.0, 8.0],
    }
    gradient_digest = producer._sha256(
        {
            key: primary_parity[key]
            for key in ("loss", "material_gradient", "geometry_gradient")
        }
    )
    parameter_digest = producer._sha256(primary_parity["parameters_after_step"])
    step_1 = {
        "loss_pre_update": primary_parity["loss"],
        "gradient_sha256": gradient_digest,
        "parameters_after_step_sha256": parameter_digest,
        "state_sha256": "s" * 64,
        "update_receipt_sha256": "a" * 64,
        "parameter_delta_l2": deltas,
    }
    step_1["update_content_sha256"] = producer._sha256(
        {
            key: step_1[key]
            for key in (
                "loss_pre_update",
                "gradient_sha256",
                "parameters_after_step_sha256",
                "parameter_delta_l2",
            )
        }
    )
    step_2 = {
        **step_1,
        "loss_pre_update": 0.4,
        "gradient_sha256": "g" * 64,
        "parameters_after_step_sha256": "p" * 64,
        "state_sha256": "t" * 64,
        "update_receipt_sha256": "u" * 64,
    }
    step_2["update_content_sha256"] = producer._sha256(
        {
            key: step_2[key]
            for key in (
                "loss_pre_update",
                "gradient_sha256",
                "parameters_after_step_sha256",
                "parameter_delta_l2",
            )
        }
    )
    row = {
        "execution": {
            "gradient_update": deltas,
            "combined_update_receipt_generation_digest": "z" * 64,
        },
        "lifecycle": None,
    }
    producer._attach_lifecycle(
        row,
        {"parity_payload": primary_parity},
        {
            "process_generation_id": "process",
            "command_sha256": "command",
            "bindings": {
                "hardware_fingerprint_sha256": "h" * 64,
                "source_manifest_sha256": "m" * 64,
                "native_source_sha256": "n" * 64,
                "native_extension_sha256": "e" * 64,
            },
            "parent_watchdog": {"watchdog_completed": True},
            "parent_watchdog_evidence_sha256": "w" * 64,
            "restart_result": {
                "auxiliary_step_1": step_1,
                "uninterrupted_step_2": step_2,
                "restored_step_2": {
                    **step_2,
                    # Raw receipt lineage may differ after restore; portable
                    # update content and the complete state must still match.
                    "update_receipt_sha256": "r" * 64,
                },
                "restore_receipt": {"restored": True},
                "checkpoint_sha256": "c" * 64,
                "auxiliary_optimizer_mutation_count": 3,
                "uninterrupted_process_optimizer_mutation_count": 2,
                "fresh_restart_optimizer_mutation_count": 1,
                "maximum_simultaneously_retained_world_count": 1,
                "uninterrupted_world_released_before_restore": True,
            },
        },
        verifier.load_json_object(producer.DEFAULT_CONTRACT),
    )
    lifecycle = row["lifecycle"]
    assert lifecycle["primary_scaling_worker_step_count"] == 1
    assert lifecycle["primary_scaling_worker_checkpoint_count"] == 0
    assert lifecycle[
        "primary_scaling_worker_measurement_excludes_auxiliary_lifecycle"
    ] is True
    assert lifecycle["auxiliary_step_1_matches_primary_scaling_row"] is True
    assert lifecycle[
        "measurement_includes_checkpoint_and_uninterrupted_second_step"
    ] is False
    assert lifecycle["step_2_update_content_match"] is True


def test_cli_dry_run_emits_plan_only_and_execute_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "no-artifact.json"
    command = [
        sys.executable,
        str(Path(producer.__file__).resolve()),
        "--output",
        str(output),
    ]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    dry = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert dry.returncode == 0, dry.stderr
    plan = json.loads(dry.stdout)
    assert plan["status"] == "blocked"
    assert plan["evidence_rows_emitted"] == 0
    assert not output.exists()

    execute = subprocess.run(
        [*command, "--execute"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert execute.returncode != 0
    assert "execution is fail-closed" in execute.stderr
    assert not output.exists()
