from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import verify_worldfoam_training_memory_ablation as verifier


ROOT = Path(__file__).resolve().parents[2]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _watchdog_measurement(process_generation_id: str):
    command_sha256 = _sha(f"command-{process_generation_id}")
    watchdog = {
        "returncode": 0,
        "elapsed_seconds": 0.5,
        "rss_measurement_kind": "parent-ps-sampled-high-water",
        "rss_sampling_interval_seconds": 0.25,
        "sampled_process_group_rss_high_water_bytes": 300_000_000,
        "sample_count": 3,
        "worker_timeout_seconds": 1800.0,
        "worker_process_group_rss_limit_bytes": 4_294_967_296,
        "watchdog_completed": True,
        "process_group_empty_after_exit": True,
        "worker_terminated_by_watchdog": False,
    }
    return {
        "worker_command_sha256": command_sha256,
        "parent_watchdog": watchdog,
        "parent_watchdog_evidence_sha256": verifier.canonical_sha256(
            {
                "parent_watchdog": watchdog,
                "process_generation_id": process_generation_id,
                "worker_command_sha256": command_sha256,
            }
        ),
    }


def _manifest():
    return [
        {
            "path": path,
            "size_bytes": (ROOT / path).stat().st_size,
            "sha256": verifier.file_sha256(ROOT / path),
        }
        for path in sorted(verifier.REQUIRED_MANIFEST_PATHS)
    ]


def _lifecycle(repeat_index: int, bindings, primary_update_receipt: str):
    state = _sha(f"state-{repeat_index}")
    step_1_gradient = _sha(f"step-1-gradient-{repeat_index}")
    step_2_gradient = _sha(f"step-2-gradient-{repeat_index}")
    step_1_parameters = _sha(f"step-1-parameters-{repeat_index}")
    step_2_parameters = _sha(f"step-2-parameters-{repeat_index}")
    delta_keys = (
        "raw_color_parameter_delta_l2_norm",
        "raw_density_parameter_delta_l2_norm",
        "positions0_parameter_delta_l2_norm",
        "velocities_parameter_delta_l2_norm",
        "weight_coefficients_parameter_delta_l2_norm",
    )
    step_1_delta = {key: 0.01 for key in delta_keys}
    step_2_delta = {key: 0.005 for key in delta_keys}
    step_1_update = verifier.canonical_sha256(
        {
            "loss_pre_update": 1.0,
            "gradient_sha256": step_1_gradient,
            "parameters_after_step_sha256": step_1_parameters,
            "parameter_delta_l2": step_1_delta,
        }
    )
    step_2_update = verifier.canonical_sha256(
        {
            "loss_pre_update": 0.8,
            "gradient_sha256": step_2_gradient,
            "parameters_after_step_sha256": step_2_parameters,
            "parameter_delta_l2": step_2_delta,
        }
    )
    restart_process = f"restart-process-{repeat_index}"
    watchdog = _watchdog_measurement(restart_process)
    restore_receipt = {
        key: _sha(f"restore-{key}-{repeat_index}")
        for key in (
            "checkpoint_generation_digest",
            "policy_generation_digest",
            "dataset_generation_digest",
            "target_residency_digest",
            "camera_grid_digest",
            "factory_generation_digest",
            "provider_generation_digest",
            "world_generation_digest",
            "sites_content_digest",
            "initializer_generation_digest",
            "geometry_generation_id",
            "material_generation_id",
            "manifest_generation_digest",
            "checkpoint_cold_recompile_seal_generation_digest",
            "restored_cold_recompile_seal_generation_digest",
            "generation_digest",
        )
    }
    restore_receipt.update(
        checkpoint_tensor_bytes=81920,
        live_state_logical_tensor_bytes=114688,
        state_checkpoint_logical_tensor_bytes=196608,
        state_checkpoint_payload_peak_logical_tensor_bytes=278528,
        persistent_tensor_bytes=0,
        compiled_tensor_bytes_retained=0,
        allocator_peak_measured=False,
    )
    return {
        "performed": True,
        "step_count": 2,
        "loss_step_1_pre_update": 1.0,
        "loss_step_1_pre_update_auxiliary": 1.0,
        "loss_step_2_uninterrupted_pre_update": 0.8,
        "loss_step_2_restored_pre_update": 0.8,
        "step_1_to_step_2_pre_update_loss_decrease": 0.2,
        "checkpoint_created_after_step": 1,
        "checkpoint_restore_used": True,
        "checkpoint_sha256": _sha(f"checkpoint-{repeat_index}"),
        "combined_checkpoint_payload_bytes": 81920,
        "live_state_logical_tensor_bytes_at_checkpoint": 114688,
        "live_state_plus_checkpoint_bytes": 196608,
        "live_state_plus_checkpoint_payload_clone_peak_bytes": 278528,
        "optimizer_history_tensor_bytes": 0,
        "restart_fresh_process": True,
        "restart_process_generation_id": restart_process,
        "restart_hardware_fingerprint_sha256": bindings[
            "hardware_fingerprint_sha256"
        ],
        "restart_source_manifest_sha256": bindings["source_manifest_sha256"],
        "restart_native_source_sha256": bindings["native_source_sha256"],
        "restart_native_extension_sha256": bindings[
            "native_extension_sha256"
        ],
        "restart_worker_command_sha256": watchdog["worker_command_sha256"],
        "restart_parent_watchdog": watchdog["parent_watchdog"],
        "restart_parent_watchdog_evidence_sha256": watchdog[
            "parent_watchdog_evidence_sha256"
        ],
        "primary_scaling_worker_step_count": 1,
        "primary_scaling_worker_checkpoint_count": 0,
        "primary_scaling_worker_measurement_excludes_auxiliary_lifecycle": True,
        "auxiliary_lifecycle_worker": True,
        "auxiliary_step_1_matches_primary_scaling_row": True,
        "step_1_gradient_sha256_primary": step_1_gradient,
        "step_1_gradient_sha256_auxiliary": step_1_gradient,
        "step_1_parameters_after_step_sha256_primary": step_1_parameters,
        "step_1_parameters_after_step_sha256_auxiliary": step_1_parameters,
        "step_1_parameter_delta_l2_primary": step_1_delta,
        "step_1_parameter_delta_l2_auxiliary": dict(step_1_delta),
        "step_1_update_content_sha256_primary": step_1_update,
        "step_1_update_content_sha256_auxiliary": step_1_update,
        "step_1_update_receipt_generation_sha256_primary": (
            primary_update_receipt
        ),
        "step_1_update_receipt_generation_sha256_auxiliary": _sha(
            f"auxiliary-step-1-receipt-{repeat_index}"
        ),
        "step_2_gradient_sha256_uninterrupted": step_2_gradient,
        "step_2_gradient_sha256_restored": step_2_gradient,
        "step_2_gradient_content_match": True,
        "step_2_parameters_after_step_sha256_uninterrupted": step_2_parameters,
        "step_2_parameters_after_step_sha256_restored": step_2_parameters,
        "step_2_parameter_delta_l2_uninterrupted": step_2_delta,
        "step_2_parameter_delta_l2_restored": dict(step_2_delta),
        "step_2_state_sha256_uninterrupted": state,
        "step_2_state_sha256_restored": state,
        "step_2_state_content_match": True,
        "step_2_update_content_sha256_uninterrupted": step_2_update,
        "step_2_update_content_sha256_restored": step_2_update,
        "step_2_update_content_match": True,
        "uninterrupted_process_optimizer_mutation_count": 2,
        "fresh_restart_optimizer_mutation_count": 1,
        "auxiliary_optimizer_mutation_count": 3,
        "post_step_1_loss_measured_by_step_2_pre_update": True,
        "measurement_includes_checkpoint_and_uninterrupted_second_step": False,
        "maximum_simultaneously_retained_world_count": 1,
        "uninterrupted_world_released_before_restore": True,
        "restore_receipt": restore_receipt,
        "restart_loss_absolute_error": 0.0,
    }


def _row(mode: str, frame_count: int, repeat_index: int, bindings):
    primary_update_receipt = _sha(
        f"combined-update-{mode}-{frame_count}-{repeat_index}"
    )
    lifecycle = (
        _lifecycle(repeat_index, bindings, primary_update_receipt)
        if mode == "fused_union_v2" and frame_count == 8
        else None
    )
    mutations = 1
    process_generation_id = f"{mode}-f{frame_count}-r{repeat_index}"
    watchdog = _watchdog_measurement(process_generation_id)
    sample_count = 512 * frame_count
    bundle_count = {8: 4, 64: 8, 300: 40}[frame_count]
    active_block_count = 4 * bundle_count
    fused = mode == "fused_union_v2"
    core_work = {
        "compile_track_count": 512,
        "compiler_work_receipt_count": 512,
        "compiler_work_receipt_bundle_count": bundle_count,
        "compiler_work_receipt_chain_link_count": 512,
        "root_complement_witness_count": 1024,
        "candidate_source_attempt_count": 2_619_392,
        "all_site_witness_check_count": 2_619_392,
        "unique_pair_difference_count": 4096,
        "per_witness_candidate_bound_verified": True,
        "exhaustive_triple_enumeration_used": False,
        "requested_frame_sampling_used": False,
        "active_compiler_accounting_complete": True,
        "all_track_receipt_digests_verified": True,
        "compiler_work_receipt_provenance": "active-kinetic-compiler-work-v1",
        "compiler_work_receipt_chain_digest": _sha("compiler-chain"),
        "retained_compiled_program_count": 0,
        "retained_compiler_receipt_entry_count": 0,
        "retained_compiler_tensor_bytes": 0,
        "compiler_receipt_state_scaling": (
            "O(1) rolling digest plus scalar totals; no retained programs"
        ),
        "spatial_bundle_count": bundle_count,
        "eligible_native_block_count": active_block_count,
        "active_native_block_count": active_block_count,
        "native_node_forward_launch_count": active_block_count,
        "native_material_word_vjp_launch_count": 0 if fused else active_block_count,
        "native_full_geometry_vjp_launch_count": 0 if fused else active_block_count,
        "native_fused_union_v2_vjp_launch_count": active_block_count if fused else 0,
        "native_fused_union_v2_transaction_count": bundle_count if fused else 0,
        "ordered_word_node_interactions": 131_072,
        "streamed_sample_count": sample_count,
        "sample_to_node_linear_interactions": 4 * sample_count,
        "sample_to_node_dense_fallback_interactions": 0,
        "selected_pixel_read_call_count": bundle_count * frame_count,
        "direct_selected_pixel_observation_count": sample_count,
        "camera_ray_slice_work_count": sample_count,
        "camera_ray_slice_scalar_count": 6 * sample_count,
    }
    row = {
        "mode": mode,
        "requested_frame_count": frame_count,
        "repeat_index": repeat_index,
        "execution": {
            "adapter_provenance": (
                "worldfoam-training-memory-ablation-adapter-v1"
            ),
            "real_native_spatial_block_coordinator": True,
            "fake_native_backend": False,
            "native_runtime_executed": True,
            "core_native_runtime_verified": False,
            "full_geometry_trainable": True,
            "material_trainable": True,
            "fixed_camera": True,
            "all_competitor_active_owner_certified": True,
            "heuristic_spatial_culling_used": False,
            "post_certification_compact_device_lowering": True,
            "direct_selected_pixel_target_stream": True,
            "full_frame_target_materialization_used": False,
            "repeat_changes_world_or_data": False,
            "cold_cpu_compile_measured": True,
            "cold_cpu_compile_measurement_count": mutations,
            "cold_cpu_compile_wall_time_seconds": 0.25,
            "dataset_is_procedural_synthetic": True,
            "public_quality_evidence": False,
            "cpu_optimizer_mutation_count": mutations,
            "cpu_optimizer_parameter_delta_l2": 5 ** 0.5 * 0.01,
            "step_wall_time_seconds": 0.5,
            "core_forward_backward_wall_time_seconds": 0.2,
            "autograd_graph_retained": False,
            "geometry_mutation_count": mutations,
            "material_device_gradient_receipt_count": mutations,
            "geometry_d2h_receipt_count": mutations,
            "combined_optimizer_authorization_count": mutations,
            "gradient_update": {
                key: 0.01
                for key in (
                    "grad_site_rgba_l2_norm",
                    "grad_site_rgba_max_abs",
                    "grad_positions0_l2_norm",
                    "grad_positions0_max_abs",
                    "grad_velocities_l2_norm",
                    "grad_velocities_max_abs",
                    "grad_weight_coefficients_l2_norm",
                    "grad_weight_coefficients_max_abs",
                    "raw_color_parameter_delta_l2_norm",
                    "raw_color_parameter_delta_max_abs",
                    "raw_density_parameter_delta_l2_norm",
                    "raw_density_parameter_delta_max_abs",
                    "positions0_parameter_delta_l2_norm",
                    "positions0_parameter_delta_max_abs",
                    "velocities_parameter_delta_l2_norm",
                    "velocities_parameter_delta_max_abs",
                    "weight_coefficients_parameter_delta_l2_norm",
                    "weight_coefficients_parameter_delta_max_abs",
                )
            },
            "bridge_receipt_generation_digest": _sha(
                f"material-bridge-{mode}-{frame_count}-{repeat_index}"
            ),
            "geometry_d2h_receipt_generation_digests": [
                _sha(f"geometry-bridge-{mode}-{frame_count}-{repeat_index}")
            ],
            "authorization_generation_digest": _sha(
                f"geometry-auth-{mode}-{frame_count}-{repeat_index}"
            ),
            "combined_update_receipt_generation_digest": primary_update_receipt,
            "stale_provider_store_retirement_count": mutations,
            "provider_store_retirement_receipt_chain_sha256": _sha(
                f"provider-retire-{mode}-{frame_count}-{repeat_index}"
            ),
            "fresh_selected_track_recompile_count": mutations,
            "fresh_selected_track_recompile_receipt_sha256": _sha(
                f"fresh-recompile-{mode}-{frame_count}-{repeat_index}"
            ),
            "fresh_selected_track_recompile_request_count": 4 * mutations,
            "worker_measurement_scope": "single_optimizer_step_scaling_row_v2",
            "worker_measurement_covers_checkpoint_and_uninterrupted_step_2": False,
            "parity_payload_scope": (
                "single_step_pre_update_gradient_and_post_update_parameters"
            ),
        },
        "structure": {
            "image_height": 384,
            "image_width": 512,
            "dataset_frame_count": 300,
            "world_site_count": 1024,
            "weight_coefficient_count": 2,
            "fixed_track_count": 512,
            "selected_frame_count": frame_count,
            "node_count": 4,
            "compile_certification_mode": "all_competitor_active_owner",
            "maximum_sites_per_track_compile": 1024,
            "track_manifest_sha256": (
                "d643b6d2fff6cb25acbbe457ca424384c430ac050cd91b54086a19cd07ee915f"
            ),
            "camera_program_sha256": (
                "95b2f7cd2b22a21eb0f42197f9e6010889ff1ffbb4d8a6e6a715448564c3b9d2"
            ),
            "target_teacher_sha256": (
                "b2b83969b14f316433d5e8fb6a2b930c725ff56910059c4e43d19415470e5196"
            ),
            "loss": "global_rgb_mean",
            "expected_observation_count": frame_count * 512,
            "loss_element_count": frame_count * 512 * 3,
            "global_loss_denominator": frame_count * 512 * 3,
            "compiled_world_sha256": _sha("compiled-world"),
            "physical_grid_sha256": _sha("physical-grid"),
            "camera_grid_sha256": _sha("camera-grid"),
            "spatial_block_manifest_sha256": _sha("spatial-blocks"),
            "target_stream_manifest_sha256": _sha(
                f"target-stream-{frame_count}"
            ),
            "observation_manifest_sha256": _sha(
                f"observation-stream-{frame_count}"
            ),
            "provider_generation_before_sha256": _sha(
                f"provider-before-{frame_count}"
            ),
            "provider_generation_after_sha256": _sha(
                f"provider-after-{frame_count}-{repeat_index}"
            ),
            "factory_generation_sha256": _sha("factory-generation"),
            "selected_track_recompile_manifest_sha256": _sha(
                "selected-track-recompile"
            ),
            "optimizer_policy_sha256": _sha("optimizer-policy"),
        },
        "work": {
            "core_accounting": dict(core_work),
            **core_work,
            "fresh_selected_track_recompile_request_count": 4,
            "fresh_selected_track_recompile_track_count": 512,
        },
        "memory": {
            "process_rss_baseline_bytes": 100_000_000,
            "process_rss_peak_bytes": 200_000_000,
            "sampled_mps_current_baseline_bytes": 10_000_000,
            "sampled_mps_current_peak_bytes": 20_000_000,
            "sampled_mps_driver_baseline_bytes": 30_000_000,
            "sampled_mps_driver_peak_bytes": 50_000_000,
            "parent_process_group_rss_sampled_peak_bytes": 300_000_000,
            "retained_driver_input_tensor_bytes": 0,
            "combined_live_state_logical_tensor_bytes": 114688,
            "material_state_logical_tensor_bytes": 49152,
            "trainable_geometry_state_logical_tensor_bytes": 65536,
            "global_material_bar_logical_tensor_bytes": 16384,
            "global_geometry_bar_logical_tensor_bytes": 65536,
            "peak_lane_resident_logical_tensor_bytes": 250_000,
            "peak_active_node_state_logical_tensor_bytes": 100_000,
            "peak_sample_launch_logical_tensor_bytes": 200_000,
            "peak_coordinator_visible_logical_tensor_upper_bound_bytes": 1_000_000,
            "maximum_geometry_bridge_visible_logical_tensor_bytes": 2_000_000,
            "combined_update_authorization_logical_tensor_bytes": 212_992,
            "combined_update_transaction_tracked_logical_and_store_accounted_upper_bound_bytes": 4_000_000,
            "persistent_frame_tensor_bytes": 0,
            "persistent_sample_tensor_bytes": 0,
            "persistent_target_tensor_bytes": 0,
            "persistent_prediction_tensor_bytes": 0,
            "optimizer_history_tensor_bytes": 0,
            "measured_peak_fields_producer_owned": True,
            "logical_bounds_are_measured_peaks": False,
        },
        "measurement": {
            "fresh_process": True,
            "process_generation_id": process_generation_id,
            "measurement_kind": "fresh-process-mps-and-rss-sampled-high-water-v1",
            "completion_fenced_before_final_measurement": True,
            "allocator_exact_peak_claimed": False,
            "mps_memory_sample_count": 3,
            "mps_memory_limit_bytes": 2147483648,
            "process_group_rss_limit_bytes": 4294967296,
            "bindings": {key: bindings[key] for key in verifier.ROW_BINDING_SHA_KEYS},
            **watchdog,
        },
        "quality": {
            "finite": True,
            "loss_before_update": 1.0,
            "post_update_loss_measured": False,
            "all_material_and_geometry_gradient_l2_norms_nonzero": True,
            "all_material_and_geometry_parameter_delta_l2_norms_nonzero": True,
        },
        "lifecycle": lifecycle,
        "evidence_sha256": "",
    }
    row["evidence_sha256"] = verifier.row_evidence_sha256(row)
    return row


def _control_row(frame_count: int, repeat_index: int, bindings, contract):
    process_generation_id = f"compiled-framewise-f{frame_count}-r{repeat_index}"
    watchdog = _watchdog_measurement(process_generation_id)
    sample_count = 512 * frame_count
    request_count = 4
    active_block_count = 8 * frame_count
    store_bytes = 4_000_000
    maximum_frame_local = 2_281_924
    expensive_peak = 6_478_536
    precompile = {
        "provider_generation_digest": _sha(f"control-provider-{frame_count}"),
        "selected_track_manifest_digest": _sha(
            f"control-selected-tracks-{frame_count}"
        ),
        "request_count": request_count,
        "track_count": 512,
        "artifact_generation_digests": [
            _sha(f"control-artifact-{frame_count}-{index}")
            for index in range(request_count)
        ],
        "compile_receipt_generation_digests": [
            _sha(f"control-compile-{frame_count}-{index}")
            for index in range(request_count)
        ],
        "compiler_work_receipt_chain_digest": _sha(
            f"control-compiler-chain-{frame_count}"
        ),
        "compiler_work_receipt_count": 512,
        "root_complement_witness_count": 1024,
        "candidate_source_attempt_count": 2_619_392,
        "all_site_witness_check_count": 2_619_392,
        "unique_pair_difference_count": 4096,
        "store_current_resident_accounted_bytes": store_bytes,
        "store_peak_resident_accounted_bytes": store_bytes,
        "store_maximum_resident_accounted_bytes": 536_870_912,
        "generation_digest": _sha(
            f"control-precompile-generation-{frame_count}"
        ),
        "provenance": (
            "paper-kinetic-compiled-framewise-selected-track-precompile-v1"
        ),
        "compile_pass_count": 1,
        "requested_frame_sampling_used": False,
        "retained_observation_count": 0,
        "retained_target_tensor_bytes": 0,
        "retained_frame_tensor_bytes": 0,
        "allocator_peak_measured": False,
    }
    update = {
        "loss": 1.0,
        "material_gradient_l2_norm": 0.1,
        "position_gradient_l2_norm": 0.1,
        "velocity_gradient_l2_norm": 0.1,
        "weight_gradient_l2_norm": 0.1,
        "raw_color_parameter_delta_l2_norm": 0.01,
        "raw_density_parameter_delta_l2_norm": 0.01,
        "positions0_parameter_delta_l2_norm": 0.01,
        "velocities_parameter_delta_l2_norm": 0.01,
        "weight_coefficients_parameter_delta_l2_norm": 0.01,
        "parameters_before_digest": _sha("control-parameters-before"),
        "parameters_after_digest": _sha(
            f"control-parameters-after-{frame_count}-{repeat_index}"
        ),
        "gradient_digest": _sha(
            f"control-gradient-{frame_count}-{repeat_index}"
        ),
        "update_authorization_digest": _sha(
            f"control-authorization-{frame_count}-{repeat_index}"
        ),
        "generation_digest": _sha(
            f"control-update-{frame_count}-{repeat_index}"
        ),
        "provenance": (
            "paper-kinetic-compiled-framewise-terminal-manual-sgd-v1"
        ),
        "cpu_optimizer_mutation_count": 1,
        "geometry_mutation_count": 1,
        "stale_provider_store_retirement_count": 0,
        "fresh_selected_track_recompile_count": 0,
        "optimizer_history_tensor_bytes": 0,
        "terminal_control_generation": True,
    }
    execution = {
        "adapter_provenance": "worldfoam-training-memory-ablation-adapter-v1",
        "control_result_receipt": {
            "provenance": (
                "paper-kinetic-compiled-framewise-full-geometry-control-v1"
            ),
            "runtime_status": "source_integrated/native_runtime_unverified",
            "native_runtime_verified": False,
            "allocator_peak_measured": False,
            "generation_digest": _sha(
                f"control-result-{frame_count}-{repeat_index}"
            ),
        },
        "update_receipt": update,
        "adapter_measurements": {
            "continuous_precompile_measurement_count": 1,
            "continuous_precompile_wall_time_seconds": 0.25,
            "control_transaction_measurement_count": 1,
            "control_transaction_wall_time_seconds": 0.02 * frame_count,
        },
    }
    accounting = {
        "control_mode": "per_frame_replay_sequential",
        "same_continuous_compiled_representation": True,
        "reverse_mode": "staged_sparse",
        "selected_frame_count": frame_count,
        "selected_track_count": 512,
        "per_frame_replay_count": frame_count,
        "per_frame_replay_wall_time_seconds": [0.01] * frame_count,
        "step_wall_time_seconds": 0.01 * frame_count,
        "one_time_continuous_compile_pass_count": 1,
        "one_time_continuous_compile_request_count": request_count,
        "one_time_continuous_compile_track_count": 512,
        "per_frame_continuous_recompile_count": 0,
        "fresh_selected_track_recompile_count": 0,
        "compiled_artifact_warm_acquisition_count": request_count * frame_count,
        "compiled_artifact_warm_hit_count": request_count * frame_count,
        "frame_result_generation_digest_chain": _sha(
            f"control-frame-result-chain-{frame_count}-{repeat_index}"
        ),
        "frame_readback_receipt_chain_digest": _sha(
            f"control-readback-chain-{frame_count}-{repeat_index}"
        ),
        "frame_readback_receipt_count": frame_count,
        "frame_release_fence_call_count": frame_count,
        "frame_release_fence_provenance": "torch.mps.synchronize/v1",
        "maximum_simultaneously_live_frame_count": 1,
        "maximum_in_flight_frame_target_count": 1,
        "maximum_in_flight_frame_prediction_count": 1,
        "maximum_in_flight_frame_reverse_count": 1,
        "frame_result_capability_retained_after_release_count": 0,
        "frame_target_released_before_next_frame": True,
        "frame_prediction_released_before_next_frame": True,
        "frame_reverse_scratch_released_before_next_frame": True,
        "cpu_optimizer_mutation_count": 1,
        "geometry_mutation_count": 1,
        "combined_optimizer_authorization_count": 1,
        "stale_provider_store_retirement_count": 0,
        "terminal_control_generation_invalidated_after_mutation": True,
        "compiler_work_receipt_chain_digest": _sha(
            f"control-compiler-chain-{frame_count}"
        ),
        "compiler_work_receipt_count": 512,
        "root_complement_witness_count": 1024,
        "candidate_source_attempt_count": 2_619_392,
        "all_site_witness_check_count": 2_619_392,
        "unique_pair_difference_count": 4096,
        "eligible_native_block_count": active_block_count,
        "active_native_block_count": active_block_count,
        "native_node_forward_launch_count": active_block_count,
        "native_sample_prepare_count": request_count * frame_count,
        "native_sample_launch_count": request_count * frame_count,
        "native_sample_completion_fence_count": request_count * frame_count,
        "native_full_geometry_vjp_launch_count": active_block_count,
        "native_fused_union_v2_transaction_count": 0,
        "geometry_d2h_completion_fence_count": active_block_count,
        "streamed_sample_count": sample_count,
        "ordered_word_node_interactions": 4096 * frame_count,
        "sample_to_node_linear_interactions": 4 * sample_count,
        "sample_to_node_dense_fallback_interactions": 0,
        "selected_pixel_read_call_count": request_count * frame_count,
        "direct_selected_pixel_observation_count": sample_count,
        "full_frame_target_materialization_count": 0,
        "camera_ray_slice_work_count": sample_count,
        "camera_ray_slice_scalar_count": 6 * sample_count,
        "peak_lane_resident_logical_tensor_bytes": 100_000,
        "peak_active_node_state_tensor_bytes": 120_000,
        "peak_sample_launch_tensor_bytes": 140_000,
        "peak_decoded_frame_scratch_upper_bound_bytes": 160_000,
        "peak_selected_frame_target_tensor_upper_bound_bytes": 6_144,
        "peak_coordinator_visible_live_tensor_upper_bound_bytes": 200_000,
        "maximum_geometry_bridge_visible_peak_logical_tensor_bytes": 2_000_000,
        "selected_pixel_read_modes": ["direct_pixels"],
        "selected_pixel_source_manifest_digests": [
            _sha("control-target-source")
        ],
        "direct_selected_pixel_target_stream": True,
        "global_material_bar_logical_tensor_bytes": 16_384,
        "global_geometry_bar_logical_tensor_bytes": 65_536,
        "global_bar_and_loss_logical_tensor_bytes": 81_924,
        "frame_material_bar_logical_tensor_bytes": 16_384,
        "frame_geometry_bar_logical_tensor_bytes": 65_536,
        "frame_material_readback_and_loss_logical_tensor_bytes": 16_388,
        "frame_coordinator_visible_logical_tensor_bytes_upper_bound": 200_000,
        "frame_geometry_bridge_visible_logical_tensor_bytes_upper_bound": 2_000_000,
        "frame_material_bar_included_in_coordinator_bound": True,
        "frame_geometry_bridge_may_overlap_coordinator": True,
        "frame_readback_cumulative_tensor_bytes": 16_388 * frame_count,
        "compiled_program_store_resident_accounted_bytes": store_bytes,
        "compiled_program_store_peak_resident_accounted_bytes": store_bytes,
        "combined_live_state_logical_tensor_bytes": 114_688,
        "maximum_frame_local_logical_tensor_bytes_upper_bound": maximum_frame_local,
        "expensive_live_logical_and_accounted_peak_upper_bound_bytes": expensive_peak,
        "expensive_peak_is_frame_count_invariant": True,
        "persistent_frame_tensor_bytes": 0,
        "persistent_sample_tensor_bytes": 0,
        "persistent_target_tensor_bytes": 0,
        "persistent_prediction_tensor_bytes": 0,
        "optimizer_history_tensor_bytes": 0,
        "camera_time_scalar_count": 300,
        "camera_time_slice_is_allowed_linear_metadata": True,
        "full_frame_target_tensor_materialized": False,
        "full_image_compile_used": False,
        "scalar_fixed_time_topology_discovery_used": False,
        "allocator_peak_measured": False,
        "rss_peak_measured": False,
        "native_runtime_verified": False,
        "frame_result_generation_digests_retained": [
            _sha(f"control-frame-result-{frame_index}")
            for frame_index in range(frame_count)
        ],
    }
    logical = {
        key: accounting[key] for key in contract["control_logical_accounting_keys"]
    }
    memory = {
        "logical_accounting": logical,
        "measured_peak_fields_producer_owned": True,
        "logical_bounds_are_measured_peaks": False,
        "process_rss_baseline_bytes": 100_000_000,
        "process_rss_peak_bytes": 250_000_000,
        "sampled_mps_current_baseline_bytes": 10_000_000,
        "sampled_mps_current_peak_bytes": 40_000_000,
        "sampled_mps_driver_baseline_bytes": 30_000_000,
        "sampled_mps_driver_peak_bytes": 70_000_000,
        "parent_process_group_rss_sampled_peak_bytes": 300_000_000,
    }
    preflight_model = {
        "kind": "same-representation-sequential-per-frame-always-launch-v1",
        "requested_frame_count": frame_count,
        "same_representation_and_native_kernels_required": True,
        "sequential_frame_release_required": True,
        "fixed_combined_live_state_bytes": 114688,
        "logical_peak_lower_bound_bytes": 114688,
        "censorship_permitted": False,
        "working_set_limit_bytes": 2147483648,
    }
    row = {
        "mode": "per_frame_replay_sequential",
        "requested_frame_count": frame_count,
        "repeat_index": repeat_index,
        "status": "measured",
        "execution": execution,
        "structure": {
            "image_height": 384,
            "image_width": 512,
            "world_site_count": 1024,
            "weight_coefficient_count": 2,
            "fixed_track_count": 512,
            "dataset_frame_count": 300,
            "selected_frame_count": frame_count,
            "expected_observation_count": 512 * frame_count,
            "loss_element_count": 512 * frame_count * 3,
            "node_count": 4,
            "compile_certification_mode": "all_competitor_active_owner",
            "maximum_sites_per_track_compile": 1024,
            "track_manifest_sha256": (
                "d643b6d2fff6cb25acbbe457ca424384c430ac050cd91b54086a19cd07ee915f"
            ),
            "camera_program_sha256": (
                "95b2f7cd2b22a21eb0f42197f9e6010889ff1ffbb4d8a6e6a715448564c3b9d2"
            ),
            "target_teacher_sha256": (
                "b2b83969b14f316433d5e8fb6a2b930c725ff56910059c4e43d19415470e5196"
            ),
            "target_stream_manifest_sha256": _sha(
                f"control-target-stream-{frame_count}"
            ),
            "observation_manifest_sha256": _sha(
                f"control-observations-{frame_count}"
            ),
            "compiled_world_sha256": _sha("compiled-world"),
            "physical_grid_sha256": _sha("physical-grid"),
            "camera_grid_sha256": _sha("camera-grid"),
            "spatial_block_manifest_sha256": _sha("spatial-blocks"),
            "provider_generation_sha256": _sha(
                f"control-provider-{frame_count}"
            ),
            "factory_generation_sha256": _sha("factory-generation"),
            "optimizer_policy_sha256": _sha("optimizer-policy"),
            "loss": "global_rgb_mean",
        },
        "work": {
            "precompile_receipt": precompile,
            "accounting": accounting,
        },
        "memory": memory,
        "measurement": {
            "fresh_process": True,
            "process_generation_id": process_generation_id,
            "measurement_kind": (
                "fresh-process-mps-and-rss-sampled-high-water-v1"
            ),
            "completion_fenced_before_final_measurement": True,
            "allocator_exact_peak_claimed": False,
            "mps_memory_sample_count": 3,
            "mps_memory_limit_bytes": 2147483648,
            "process_group_rss_limit_bytes": 4294967296,
            "bindings": {
                key: bindings[key] for key in verifier.ROW_BINDING_SHA_KEYS
            },
            **watchdog,
        },
        "preflight": {
            "performed": True,
            "policy": "no_censorship_parent_guard_failure_is_failed_row_v1",
            "model_sha256": verifier.canonical_sha256(preflight_model),
            "model": preflight_model,
            "projected_peak_bytes": 114688,
            "working_set_limit_bytes": 2147483648,
            "decision": "launch",
            "censor_reason": None,
        },
        "quality": {
            "finite": True,
            "loss": 1.0,
            "post_update_loss_measured": False,
        },
        "lifecycle": None,
        "evidence_sha256": "",
    }
    row["evidence_sha256"] = verifier.row_evidence_sha256(row)
    return row


def _artifact():
    config = verifier.load_json_object(verifier.DEFAULT_CONFIG)
    contract = verifier.load_json_object(verifier.DEFAULT_CONTRACT)
    manifest = _manifest()
    hardware = {
        "backend": "mps",
        "platform": "test-platform",
        "machine": "arm64",
        "processor": "test-processor",
        "python": "3.11.0",
        "torch": "test-torch",
        "device": "Apple MPS",
        "mps_memory_limit": {
            "requested_fraction": 0.5,
            "effective_fraction": 0.25,
            "recommended_max_memory_bytes": 8_589_934_592,
            "absolute_working_set_limit_bytes": 2_147_483_648,
            "effective_working_set_limit_bytes": 2_147_483_648,
        },
    }
    bindings = {
        "config_sha256": verifier.file_sha256(verifier.DEFAULT_CONFIG),
        "contract_sha256": verifier.file_sha256(verifier.DEFAULT_CONTRACT),
        "source_manifest_sha256": verifier.source_manifest_sha256(manifest),
        "native_source_sha256": _sha("native-source"),
        "native_extension_sha256": _sha("native-extension"),
        "hardware_fingerprint_sha256": verifier.canonical_sha256(hardware),
        "producer_sha256": _sha("producer"),
        "driver_sha256": _sha("driver"),
    }
    rows = [
        _row(row["mode"], row["requested_frame_count"], repeat_index, bindings)
        for row in contract["required_rows"]
        for repeat_index in range(contract["required_repeat_count"])
    ]
    control_rows = [
        _control_row(
            row["requested_frame_count"], repeat_index, bindings, contract
        )
        for row in contract["required_control_rows"]
        for repeat_index in range(contract["required_repeat_count"])
    ]
    row_by_key = {
        (row["mode"], row["requested_frame_count"], row["repeat_index"]): row
        for row in rows
    }
    control_by_key = {
        (row["mode"], row["requested_frame_count"], row["repeat_index"]): row
        for row in control_rows
    }
    parity = [
        {
            "repeat_index": repeat_index,
            "staged_row_evidence_sha256": row_by_key[
                ("staged_sparse", 8, repeat_index)
            ]["evidence_sha256"],
            "fused_row_evidence_sha256": row_by_key[
                ("fused_union_v2", 8, repeat_index)
            ]["evidence_sha256"],
            "loss_absolute_error": 0.0,
            "material_gradient_relative_l2": 0.0,
            "geometry_gradient_relative_l2": 0.0,
            "parameter_relative_l2": 0.0,
        }
        for repeat_index in range(3)
    ]
    fused_control_parity = [
        {
            "repeat_index": repeat_index,
            "fused_row_evidence_sha256": row_by_key[
                ("fused_union_v2", 8, repeat_index)
            ]["evidence_sha256"],
            "compiled_framewise_control_row_evidence_sha256": control_by_key[
                ("per_frame_replay_sequential", 8, repeat_index)
            ]["evidence_sha256"],
            "loss_absolute_error": 0.0,
            "material_gradient_relative_l2": 0.0,
            "geometry_gradient_relative_l2": 0.0,
            "parameter_relative_l2": 0.0,
        }
        for repeat_index in range(3)
    ]
    artifact = {
        "schema_version": 1,
        "artifact_kind": verifier.ARTIFACT_KIND,
        "benchmark": verifier.BENCHMARK,
        "config_id": config["config_id"],
        "backend": config["backend"],
        "execution_scope": config["execution_scope"],
        "status": "measured",
        "evidence_origin": "fresh_process_production_ablation_v1",
        "proxy_or_test_artifact": False,
        "dataset_is_procedural_synthetic": True,
        "native_execution_measured": True,
        "measurement_is_simulated": False,
        "public_quality_evidence": False,
        "claim_scope": "synthetic_systems_memory_trainability_only",
        "producer_execution_implemented": True,
        "all_rows_fresh_process_completed": True,
        "bindings": bindings,
        "hardware": hardware,
        "source_manifest": manifest,
        "rows": rows,
        "control_rows": control_rows,
        "staged_fused_f8_parity": parity,
        "fused_compiled_framewise_f8_parity": fused_control_parity,
    }
    return artifact, config, contract


def _verify(artifact, config, contract):
    return verifier.verify_artifact_payload(
        artifact,
        config,
        contract,
        config_sha256=verifier.file_sha256(verifier.DEFAULT_CONFIG),
        contract_sha256=verifier.file_sha256(verifier.DEFAULT_CONTRACT),
    )


def test_complete_measured_matrix_passes_strict_contract() -> None:
    artifact, config, contract = _artifact()
    report = _verify(artifact, config, contract)
    assert report["status"] == "passed", report["failures"]
    assert report["accepted"] is True
    assert report["required_row_count"] == report["observed_row_count"] == 12
    assert (
        report["required_control_row_count"]
        == report["observed_control_row_count"]
        == 9
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda artifact: artifact["rows"].pop(), "required rows are missing"),
        (
            lambda artifact: artifact["control_rows"].pop(),
            "required control rows are missing",
        ),
        (
            lambda artifact: artifact.pop(
                "fused_compiled_framewise_f8_parity"
            ),
            "fused/compiled-framewise F=8 parity records are missing",
        ),
        (
            lambda artifact: artifact[
                "fused_compiled_framewise_f8_parity"
            ][0].__setitem__("loss_absolute_error", 1.0),
            "fused/compiled-framewise parity repeat 0: loss_absolute_error exceeds contract",
        ),
            (
                lambda artifact: artifact["rows"][0]["memory"].__setitem__(
                    "persistent_frame_tensor_bytes", 4
                ),
                "persistent_frame_tensor_bytes must be exactly zero",
        ),
        (
            lambda artifact: artifact["rows"][0]["execution"].update(
                real_native_spatial_block_coordinator=False,
                fake_native_backend=True,
            ),
            "real_native_spatial_block_coordinator must be True",
        ),
        (
                lambda artifact: artifact["rows"][0]["execution"].update(
                    cpu_optimizer_mutation_count=0,
                    cpu_optimizer_parameter_delta_l2=0.0,
                ),
                "cpu_optimizer_mutation_count must be 1",
        ),
        (
            lambda artifact: artifact.update(proxy_or_test_artifact=True),
            "artifact proxy_or_test_artifact changed",
        ),
        (
            lambda artifact: artifact["rows"][0]["execution"].update(
                heuristic_spatial_culling_used=True
            ),
            "heuristic_spatial_culling_used must be False",
        ),
            (
                lambda artifact: artifact["rows"][0]["memory"].__setitem__(
                    "trainable_geometry_state_logical_tensor_bytes", 0
                ),
                "trainable_geometry_state_logical_tensor_bytes changed",
            ),
            (
                lambda artifact: artifact["rows"][0]["execution"].update(
                    geometry_d2h_receipt_count=0
                ),
                "geometry D2H receipt coverage is incomplete",
        ),
        (
            lambda artifact: next(
                row
                for row in artifact["rows"]
                if row["mode"] == "fused_union_v2"
                and row["requested_frame_count"] == 64
            )["work"].__setitem__("streamed_sample_count", 17),
            "not an integer per-frame multiple",
        ),
        (
            lambda artifact: next(
                row
                for row in artifact["rows"]
                if row["mode"] == "fused_union_v2"
                and row["requested_frame_count"] == 8
            )["lifecycle"].__setitem__("loss_step_2_restored_pre_update", 1.1),
            "restored loss differs from uninterrupted loss",
        ),
        (
            lambda artifact: next(
                row
                for row in artifact["rows"]
                if row["mode"] == "fused_union_v2"
                and row["requested_frame_count"] == 8
            )["execution"].__setitem__("cpu_optimizer_mutation_count", 2),
            "cpu_optimizer_mutation_count must be 1",
        ),
        (
            lambda artifact: next(
                row
                for row in artifact["rows"]
                if row["mode"] == "fused_union_v2"
                and row["requested_frame_count"] == 8
            )["execution"].__setitem__(
                "worker_measurement_covers_checkpoint_and_uninterrupted_step_2",
                True,
            ),
            "worker_measurement_covers_checkpoint_and_uninterrupted_step_2 must be False",
        ),
        (
            lambda artifact: next(
                row
                for row in artifact["rows"]
                if row["mode"] == "fused_union_v2"
                and row["requested_frame_count"] == 8
            )["lifecycle"].__setitem__(
                "measurement_includes_checkpoint_and_uninterrupted_second_step",
                True,
            ),
            "measurement_includes_checkpoint_and_uninterrupted_second_step changed",
        ),
        (
            lambda artifact: next(
                row
                for row in artifact["rows"]
                if row["mode"] == "fused_union_v2"
                and row["requested_frame_count"] == 8
            )["lifecycle"].__setitem__(
                "step_1_parameters_after_step_sha256_auxiliary",
                _sha("foreign-step-1-parameters"),
            ),
            "auxiliary step-1 parameters differ from primary",
        ),
        (
            lambda artifact: next(
                row
                for row in artifact["rows"]
                if row["mode"] == "fused_union_v2"
                and row["requested_frame_count"] == 8
            )["lifecycle"].__setitem__("legacy_receipt_match", True),
            "lifecycle receipt keys are noncanonical",
        ),
        (
            lambda artifact: artifact["rows"][0]["measurement"]["bindings"].__setitem__(
                "native_extension_sha256", _sha("foreign-native")
            ),
            "measurement binding native_extension_sha256 drifted",
        ),
    ),
)
def test_verifier_fails_closed_on_missing_or_invalid_evidence(mutate, message) -> None:
    artifact, config, contract = _artifact()
    mutate(artifact)
    for row in artifact["rows"]:
        row["evidence_sha256"] = verifier.row_evidence_sha256(row)
    report = _verify(artifact, config, contract)
    assert report["status"] == "failed"
    assert report["accepted"] is False
    assert message in "\n".join(report["failures"])


@pytest.mark.parametrize(
    ("mutate", "message"),
        (
            (
                lambda row: row.__setitem__("status", "censored_preflight_memory"),
                "control row status must be measured",
            ),
            (
                lambda row: row["execution"]["control_result_receipt"].__setitem__(
                    "native_runtime_verified", True
                ),
                "control result receipt.native_runtime_verified changed",
            ),
            (
                lambda row: row["work"]["accounting"].__setitem__(
                    "per_frame_continuous_recompile_count", 1
                ),
                "per_frame_continuous_recompile_count must be exactly zero",
            ),
            (
                lambda row: row["work"]["accounting"].__setitem__(
                    "frame_release_fence_call_count", 7
                ),
                "control accounting.frame_release_fence_call_count changed",
            ),
            (
                lambda row: row["execution"]["update_receipt"].__setitem__(
                    "material_gradient_l2_norm", 0.0
                ),
                "control update receipt.material_gradient_l2_norm must be finite and nonzero",
            ),
            (
                lambda row: row["execution"]["update_receipt"].__setitem__(
                    "raw_density_parameter_delta_l2_norm", 0.0
                ),
                "control update receipt.raw_density_parameter_delta_l2_norm must be finite and nonzero",
            ),
            (
                lambda row: row["work"]["accounting"].__setitem__(
                    "one_time_continuous_compile_pass_count", 2
                ),
                "control accounting.one_time_continuous_compile_pass_count changed",
            ),
            (
                lambda row: row["memory"]["logical_accounting"].__setitem__(
                    "persistent_sample_tensor_bytes", 4
                ),
                "logical accounting.persistent_sample_tensor_bytes differs from raw accounting",
            ),
            (
                lambda row: row["work"]["accounting"].__setitem__(
                    "full_image_compile_used", True
                ),
                "control accounting.full_image_compile_used changed",
            ),
        (
            lambda row: row["quality"].__setitem__("loss_after", 0.5),
            "measured control quality keys are noncanonical",
        ),
    ),
)
def test_control_verifier_requires_exact_compiled_framewise_receipts(
    mutate, message
) -> None:
    artifact, config, contract = _artifact()
    mutate(artifact["control_rows"][0])
    for row in artifact["control_rows"]:
        row["evidence_sha256"] = verifier.row_evidence_sha256(row)
    report = _verify(artifact, config, contract)
    assert report["status"] == "failed"
    assert report["accepted"] is False
    assert message in "\n".join(report["failures"])


def test_config_and_contract_tampering_fail_before_artifact_review() -> None:
    artifact, config, contract = _artifact()
    broken_config = copy.deepcopy(config)
    broken_config["procedural_world"]["site_count"] = 2
    with pytest.raises(ValueError, match="site_count"):
        _verify(artifact, broken_config, contract)

    broken_contract = copy.deepcopy(contract)
    broken_contract["required_rows"].pop()
    with pytest.raises(ValueError, match="row matrix"):
        _verify(artifact, config, broken_contract)
