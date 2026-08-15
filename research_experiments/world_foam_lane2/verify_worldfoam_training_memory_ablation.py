#!/usr/bin/env python3
"""Fail-closed verifier for the paper-scale WorldFoam training-memory ablation.

This contract is deliberately distinct from the two-site material-only v3
mechanical fixture.  It accepts only real-native, full-geometry, 1024-site
training rows with an actual CPU optimizer mutation issued through the
production CPU/device bridge.  Logical byte formulas never substitute for
fresh-process RSS and sampled public MPS allocator evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
CONTRACT_SCHEMA_VERSION = 1
BENCHMARK = "worldfoam_training_memory_ablation"
ARTIFACT_KIND = "worldfoam-training-memory-ablation-evidence-v1"
DEFAULT_CONFIG = Path(__file__).with_name("worldfoam_training_memory_ablation_v1.json")
DEFAULT_CONTRACT = Path(__file__).with_name(
    "worldfoam_training_memory_ablation_acceptance_v1.json"
)
ROOT = Path(__file__).resolve().parents[2]

REQUIRED_BINDING_SHA_KEYS = (
    "config_sha256",
    "contract_sha256",
    "source_manifest_sha256",
    "native_source_sha256",
    "native_extension_sha256",
    "hardware_fingerprint_sha256",
    "producer_sha256",
    "driver_sha256",
)
ROW_BINDING_SHA_KEYS = (
    "config_sha256",
    "source_manifest_sha256",
    "native_source_sha256",
    "native_extension_sha256",
    "hardware_fingerprint_sha256",
)
REQUIRED_MANIFEST_PATHS = frozenset(
    {
        "research_experiments/world_foam_lane2/run_worldfoam_training_memory_ablation.py",
        "research_experiments/world_foam_lane2/worldfoam_training_memory_spatial_native_driver.py",
        "research_experiments/world_foam_lane2/verify_worldfoam_training_memory_ablation.py",
        "research_experiments/world_foam_lane2/worldfoam_training_memory_ablation_v1.json",
        "research_experiments/world_foam_lane2/worldfoam_training_memory_ablation_acceptance_v1.json",
        "research_experiments/world_foam_lane2/kinetic_dense_cached_native_material_request.py",
        "research_experiments/world_foam_lane2/kinetic_active_owner_chart_compiler.py",
        "research_experiments/world_foam_lane2/kinetic_compiled_cpu_artifact_store.py",
        "research_experiments/world_foam_lane2/kinetic_power_word_compiler.py",
        "research_experiments/world_foam_lane2/kinetic_native_equal_rank_sparse_geometry_reduction.py",
        "src/train/worldfoam_training_memory_ablation_adapter.py",
        "src/train/paper_kinetic_active_track_program_factory.py",
        "src/train/paper_kinetic_compiled_framewise_full_geometry_control.py",
        "src/train/paper_kinetic_fixed_camera_combined_state.py",
        "src/train/paper_kinetic_fixed_camera_full_geometry_step.py",
        "src/train/paper_kinetic_fixed_site_material_device_bridge.py",
        "src/train/paper_kinetic_fixed_site_material_state.py",
        "src/train/paper_kinetic_lazy_full_geometry_device_bridge.py",
        "src/train/paper_kinetic_lazy_full_geometry_step.py",
        "src/train/paper_kinetic_lazy_program_bundles.py",
        "src/train/paper_kinetic_union_local_bar_assembly.py",
        "src/train/paper_kinetic_world_initializer.py",
    }
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def source_manifest_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    frozen = tuple(
        {
            "path": record["path"],
            "size_bytes": record["size_bytes"],
            "sha256": record["sha256"],
        }
        for record in records
    )
    return canonical_sha256(frozen)


def row_evidence_sha256(row: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: value for key, value in row.items() if key != "evidence_sha256"}
    )


def _fixed_track_manifest() -> tuple[dict[str, int], ...]:
    return tuple(
        {
            "track_id": 32 * row_index + column_index,
            "row": 6 + 24 * row_index,
            "column": 8 + 16 * column_index,
            "pixel_id": (6 + 24 * row_index) * 512 + (8 + 16 * column_index),
        }
        for row_index in range(16)
        for column_index in range(32)
    )


def _at(value: Mapping[str, Any], *path: str) -> Any:
    current: Any = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"missing configuration field: {'.'.join(path)}")
        current = current[part]
    return current


def _require_exact(value: Mapping[str, Any], path: tuple[str, ...], expected: Any) -> None:
    actual = _at(value, *path)
    if actual != expected:
        raise ValueError(
            f"{'.'.join(path)} is {actual!r}, expected {expected!r}"
        )


def validate_config(config: Mapping[str, Any]) -> None:
    exact = {
        ("schema_version",): 1,
        ("config_id",): "worldfoam-training-memory-ablation-1024site-v1",
        ("benchmark",): BENCHMARK,
        ("backend",): "mps",
        ("execution_scope",): "production_spatial_native_full_geometry_manual_sgd",
        ("image", "height"): 384,
        ("image", "width"): 512,
        ("camera", "program", "kind"): "fixed_pinhole_world_to_camera_v1",
        ("camera", "program", "fx"): 460.8,
        ("camera", "program", "fy"): 460.8,
        ("camera", "program", "cx"): 255.5,
        ("camera", "program", "cy"): 191.5,
        ("camera", "program_sha256"): (
            "95b2f7cd2b22a21eb0f42197f9e6010889ff1ffbb4d8a6e6a715448564c3b9d2"
        ),
        ("track_manifest", "kind"): "fixed_regular_selected_pixel_grid_v1",
        ("track_manifest", "track_count"): 512,
        ("track_manifest", "grid_rows"): 16,
        ("track_manifest", "grid_columns"): 32,
        ("track_manifest", "row_formula"): "row=6+24*r for r in [0,15]",
        ("track_manifest", "column_formula"): "column=8+16*c for c in [0,31]",
        ("track_manifest", "track_id_formula"): "track_id=32*r+c",
        ("track_manifest", "pixel_id_formula"): (
            "pixel_id=(6+24*r)*512+(8+16*c)"
        ),
        ("track_manifest", "ordered_manifest_digest_algorithm"): (
            "sha256-canonical-json-track_id-row-column-pixel_id-v1"
        ),
        ("track_manifest", "ordered_manifest_sha256"): (
            "d643b6d2fff6cb25acbbe457ca424384c430ac050cd91b54086a19cd07ee915f"
        ),
        ("track_manifest", "identical_tracks_across_requested_frame_counts"): True,
        ("target_source", "kind"): (
            "procedural_teacher_direct_selected_pixel_stream_v1"
        ),
        ("target_source", "teacher_generation_sha256"): (
            "b2b83969b14f316433d5e8fb6a2b930c725ff56910059c4e43d19415470e5196"
        ),
        ("target_source", "direct_selected_pixel_only"): True,
        ("target_source", "full_frame_materialization"): False,
        ("target_source", "maximum_resident_observations"): 4096,
        ("target_source", "maximum_resident_target_tensor_bytes"): 49152,
        ("target_source", "loss"): "global_rgb_mean",
        ("target_source", "loss_denominator_formula"): (
            "512*requested_frame_count*3"
        ),
        ("temporal_grid", "dataset_frame_count"): 300,
        ("temporal_grid", "physical_t_min"): -1.0,
        ("temporal_grid", "physical_t_max"): 1.0,
        ("temporal_grid", "requested_frame_counts"): [8, 64, 300],
        ("temporal_grid", "requested_subset_formula"): "endpoint_including_even_index_v1",
        ("procedural_world", "kind"): "rectangular_kinetic_power_sites_v1",
        ("procedural_world", "rows"): 32,
        ("procedural_world", "columns"): 32,
        ("procedural_world", "site_count"): 1024,
        ("procedural_world", "generation_seed"): 17029,
        ("material", "model"): "physical_p0_beer_lambert_v1",
        ("material", "initializer"): "external_physical_p0_procedural_v1",
        ("compiler", "node_count"): 4,
        ("compiler", "certification_mode"): "all_competitor_active_owner",
        ("compiler", "maximum_sites_per_track_compile"): 1024,
        ("compiler", "active_owner_source_policy"): (
            "witnessed_active_owner_words_only"
        ),
        ("compiler", "lazy_competitor_pair_cache"): True,
        ("compiler", "heuristic_spatial_culling_allowed"): False,
        ("compiler", "cold_cpu_compile_allowed_and_measured"): True,
        ("spatial_streaming", "partition_semantics"): (
            "post_certification_active_owner_device_lowering_v1"
        ),
        ("spatial_streaming", "maximum_active_sites_per_device_block"): 64,
        ("spatial_streaming", "maximum_device_blocks_per_request"): 16,
        ("spatial_streaming", "maximum_active_union_sites_per_request"): 1024,
        ("spatial_streaming", "maximum_tracks_per_request"): 128,
        ("spatial_streaming", "artifact_store_maximum_entries"): 1,
        ("spatial_streaming", "native_token_cache_maximum_entries"): 1,
        ("optimizer", "kind"): "manual_sgd_cpu_v1",
        ("optimizer", "learning_rate"): 0.001,
        ("optimizer", "momentum"): 0.0,
        ("optimizer", "weight_decay"): 0.0,
        ("optimizer", "gradient_clipping"): "none",
        ("optimizer", "mutation_steps_per_row"): 1,
        ("optimizer", "lifecycle_frame_count"): 8,
        ("optimizer", "lifecycle_step_count"): 2,
        ("optimizer", "checkpoint_after_step"): 1,
        ("optimizer", "restart_replays_step"): 2,
        ("ablation", "repeat_count"): 3,
        ("ablation", "staged_mode"): "staged_sparse",
        ("ablation", "staged_frame_counts"): [8],
        ("ablation", "fused_mode"): "fused_union_v2",
        ("ablation", "fused_frame_counts"): [8, 64, 300],
        ("ablation", "control_mode"): "per_frame_replay_sequential",
        ("ablation", "control_frame_counts"): [8, 64, 300],
        ("ablation", "control_same_representation"): True,
        ("ablation", "control_releases_frame_tape_before_next_frame"): True,
        ("ablation", "control_expected_peak_scaling_in_frame_count"): "O(1)",
        ("ablation", "control_expected_world_work_scaling_in_frame_count"): "O(F)",
        ("ablation", "control_measured_required_frame_counts"): [8, 64, 300],
        ("ablation", "control_censorable_frame_counts"): [],
        ("ablation", "control_preflight_required_before_launch"): False,
        ("ablation", "control_memory_censor_policy"): (
            "no_censorship_parent_guard_failure_is_failed_row_v1"
        ),
        ("ablation", "matched_repeat_indices"): [0, 1, 2],
        ("ablation", "fixed_camera"): True,
        ("ablation", "train_geometry"): True,
        ("ablation", "train_material"): True,
        ("state_accounting", "weight_coefficient_count"): 2,
        ("state_accounting", "material_parameter_bytes_per_site"): 16,
        ("state_accounting", "material_physical_snapshot_bytes_per_site"): 16,
        ("state_accounting", "material_gradient_buffer_bytes_per_site"): 16,
        ("state_accounting", "material_live_state_bytes_per_site"): 48,
        ("state_accounting", "trainable_live_geometry_state_bytes_per_site"): 64,
        ("state_accounting", "combined_live_state_bytes_per_site"): 112,
        ("state_accounting", "device_material_snapshot_bytes_per_site"): 16,
        ("state_accounting", "global_material_bar_bytes_per_site"): 16,
        ("state_accounting", "transient_global_geometry_bar_bytes_per_site"): 64,
        ("state_accounting", "optimizer_history_bytes_per_site"): 0,
        ("state_accounting", "combined_checkpoint_bytes_per_site"): 80,
        ("state_accounting", "live_state_plus_checkpoint_bytes_per_site"): 192,
        (
            "state_accounting",
            "live_state_plus_checkpoint_payload_clone_peak_bytes_per_site",
        ): 272,
        ("memory_limits_bytes", "maximum_mps_working_set"): 2 * 1024**3,
        ("memory_limits_bytes", "maximum_worker_process_group_rss"): 4 * 1024**3,
    }
    for path, expected in exact.items():
        _require_exact(config, path, expected)
    if "gradient_clip_norm" in _at(config, "optimizer"):
        raise ValueError("optimizer.gradient_clip_norm is not part of canonical manual SGD")
    if canonical_sha256(_at(config, "camera", "program")) != _at(
        config, "camera", "program_sha256"
    ):
        raise ValueError("camera program digest changed")
    if canonical_sha256(_fixed_track_manifest()) != _at(
        config, "track_manifest", "ordered_manifest_sha256"
    ):
        raise ValueError("fixed track manifest digest changed")
    target_source = dict(_at(config, "target_source"))
    target_digest = target_source.pop("teacher_generation_sha256", None)
    if canonical_sha256(target_source) != target_digest:
        raise ValueError("target teacher generation digest changed")
    world = _at(config, "procedural_world")
    if world["rows"] * world["columns"] != world["site_count"]:
        raise ValueError("procedural site grid does not equal site_count")
    for key in ("position_formula", "velocity_formula", "weight_formula"):
        if not isinstance(world.get(key), str) or not world[key].strip():
            raise ValueError(f"procedural_world.{key} must be nonempty")
    material = _at(config, "material")
    if not 0.0 < material["minimum_density"] <= material["maximum_density"]:
        raise ValueError("physical P0 density bounds are invalid")


def validate_contract(contract: Mapping[str, Any]) -> None:
    exact = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_id": "worldfoam-training-memory-ablation-acceptance-v1",
        "benchmark": BENCHMARK,
        "required_config_id": "worldfoam-training-memory-ablation-1024site-v1",
        "required_backend": "mps",
        "required_execution_scope": (
            "production_spatial_native_full_geometry_manual_sgd"
        ),
        "required_evidence_origin": "fresh_process_production_ablation_v1",
        "required_image": [384, 512],
        "required_dataset_frame_count": 300,
        "required_world_site_count": 1024,
        "required_site_grid": [32, 32],
        "required_weight_coefficient_count": 2,
        "required_fixed_track_count": 512,
        "required_track_grid": [16, 32],
        "required_track_manifest_sha256": (
            "d643b6d2fff6cb25acbbe457ca424384c430ac050cd91b54086a19cd07ee915f"
        ),
        "required_camera_program_sha256": (
            "95b2f7cd2b22a21eb0f42197f9e6010889ff1ffbb4d8a6e6a715448564c3b9d2"
        ),
        "required_target_teacher_sha256": (
            "b2b83969b14f316433d5e8fb6a2b930c725ff56910059c4e43d19415470e5196"
        ),
        "required_loss": "global_rgb_mean",
        "required_compile_certification_mode": "all_competitor_active_owner",
        "required_maximum_sites_per_track_compile": 1024,
        "required_repeat_count": 3,
        "required_measured_control_frame_counts": [8, 64, 300],
        "censorable_control_frame_counts": [],
        "required_control_row_status": "measured",
        "required_control_preflight_policy": (
            "no_censorship_parent_guard_failure_is_failed_row_v1"
        ),
        "require_control_same_representation": True,
        "require_control_releases_frame_tape_before_next_frame": True,
        "required_control_peak_scaling_in_frame_count": "O(1)",
        "required_control_world_work_scaling_in_frame_count": "O(F)",
        "required_control_adapter_provenance": (
            "worldfoam-training-memory-ablation-adapter-v1"
        ),
        "required_control_step_provenance": (
            "paper-kinetic-compiled-framewise-full-geometry-control-v1"
        ),
        "required_control_runtime_status": (
            "source_integrated/native_runtime_unverified"
        ),
        "required_control_precompile_provenance": (
            "paper-kinetic-compiled-framewise-selected-track-precompile-v1"
        ),
        "required_control_update_provenance": (
            "paper-kinetic-compiled-framewise-terminal-manual-sgd-v1"
        ),
        "required_control_frame_release_fence_provenance": (
            "torch.mps.synchronize/v1"
        ),
        "required_control_precompile_request_count": 4,
        "maximum_control_compiled_program_store_bytes": 536870912,
        "required_lifecycle_mode": "fused_union_v2",
        "required_lifecycle_frame_count": 8,
        "required_primary_scaling_optimizer_mutations_per_row": 1,
        "required_primary_scaling_checkpoint_count_per_row": 0,
        "required_auxiliary_lifecycle_optimizer_mutations": 3,
        "required_auxiliary_uninterrupted_process_optimizer_mutations": 2,
        "required_auxiliary_fresh_restart_optimizer_mutations": 1,
        "global_material_bar_shape": [1024, 4],
        "global_material_bar_bytes": 16384,
        "transient_global_geometry_bar_bytes": 65536,
        "cpu_material_parameter_state_bytes": 16384,
        "cpu_material_physical_snapshot_bytes": 16384,
        "cpu_material_gradient_buffer_bytes": 16384,
        "cpu_material_live_state_bytes": 49152,
        "trainable_live_geometry_state_bytes": 65536,
        "combined_live_state_bytes": 114688,
        "device_material_snapshot_bytes": 16384,
        "live_state_and_transient_training_bars_logical_bytes": 212992,
        "optimizer_history_tensor_bytes": 0,
        "combined_checkpoint_payload_bytes": 81920,
        "live_state_plus_checkpoint_bytes": 196608,
        "live_state_plus_checkpoint_payload_clone_peak_bytes": 278528,
        "require_real_native_spatial_block_coordinator": True,
        "require_fake_native_backend_false": True,
        "require_full_geometry_trainable": True,
        "require_all_competitor_active_owner_certification": True,
        "forbid_heuristic_spatial_culling": True,
        "require_post_certification_compact_device_lowering": True,
        "require_cold_cpu_compile_measurement": True,
        "require_production_material_device_gradient_receipt": True,
        "require_production_geometry_device_to_host_reduction_receipt": True,
        "require_geometry_optimizer_authorization_receipt": True,
        "require_cpu_optimizer_mutation": True,
        "require_direct_selected_pixel_target_stream": True,
        "forbid_full_frame_target_materialization": True,
        "require_measurement_repeat_only": True,
        "require_fresh_process_per_row": True,
        "require_completion_fence": True,
        "require_hash_bound_config_source_native_hardware": True,
        "require_staged_fused_f8_parity": True,
        "require_fused_compiled_framewise_f8_parity": True,
        "require_f8_two_step_loss_decrease": True,
        "require_f8_checkpoint_restart_parity": True,
        "reject_proxy_or_test_artifacts": True,
        "require_dataset_is_procedural_synthetic": True,
        "require_native_execution_measured": True,
        "require_measurement_is_simulated_false": True,
        "required_claim_scope": "synthetic_systems_memory_trainability_only",
        "require_public_quality_evidence_false": True,
        "reject_dry_run_rows": True,
        "required_parent_watchdog_rss_measurement_kind": (
            "parent-ps-sampled-high-water"
        ),
        "required_worker_watchdog_poll_interval_seconds": 0.25,
        "maximum_worker_timeout_seconds": 1800.0,
        "minimum_parent_watchdog_sample_count": 2,
        "maximum_control_process_rss_growth_bytes": 67108864,
        "maximum_control_mps_current_growth_bytes": 33554432,
        "maximum_control_mps_driver_growth_bytes": 67108864,
        "maximum_control_memory_scale": 1.25,
        "maximum_fused_compiled_framewise_loss_absolute_error": 0.00001,
        "maximum_fused_compiled_framewise_material_gradient_relative_l2": 0.0001,
        "maximum_fused_compiled_framewise_geometry_gradient_relative_l2": 0.0001,
        "maximum_fused_compiled_framewise_parameter_relative_l2": 0.0001,
    }
    for key, expected in exact.items():
        if contract.get(key) != expected:
            raise ValueError(f"contract {key} changed")
    rows = contract.get("required_rows")
    if not isinstance(rows, list):
        raise ValueError("contract required_rows must be a list")
    row_keys = {
        (row.get("mode"), row.get("requested_frame_count"))
        for row in rows
        if isinstance(row, Mapping)
    }
    if row_keys != {
        ("staged_sparse", 8),
        ("fused_union_v2", 8),
        ("fused_union_v2", 64),
        ("fused_union_v2", 300),
    } or len(rows) != 4:
        raise ValueError("contract required row matrix changed")
    control_rows = contract.get("required_control_rows")
    if not isinstance(control_rows, list):
        raise ValueError("contract required_control_rows must be a list")
    control_keys = {
        (row.get("mode"), row.get("requested_frame_count"))
        for row in control_rows
        if isinstance(row, Mapping)
    }
    if control_keys != {
        ("per_frame_replay_sequential", 8),
        ("per_frame_replay_sequential", 64),
        ("per_frame_replay_sequential", 300),
    } or len(control_rows) != 3:
        raise ValueError("contract required control row matrix changed")
    exact_receipt_lists = {
        "zero_retained_frame_state_keys": [
            "persistent_frame_tensor_bytes",
            "persistent_sample_tensor_bytes",
            "persistent_target_tensor_bytes",
            "persistent_prediction_tensor_bytes",
        ],
        "fused_frame_invariant_work_keys": [
            "compile_track_count",
            "compiler_work_receipt_count",
            "compiler_work_receipt_chain_link_count",
            "root_complement_witness_count",
            "candidate_source_attempt_count",
            "all_site_witness_check_count",
            "unique_pair_difference_count",
            "ordered_word_node_interactions",
        ],
        "fused_fixed_schedule_linear_work_keys": [
            "streamed_sample_count",
            "camera_ray_slice_work_count",
            "camera_ray_slice_scalar_count",
            "direct_selected_pixel_observation_count",
            "sample_to_node_linear_interactions",
        ],
        "fused_allowed_streamed_work_keys": [
            "spatial_bundle_count",
            "compiler_work_receipt_bundle_count",
            "eligible_native_block_count",
            "active_native_block_count",
            "native_node_forward_launch_count",
            "native_fused_union_v2_transaction_count",
            "selected_pixel_read_call_count",
        ],
        "control_update_gradient_l2_keys": [
            "material_gradient_l2_norm",
            "position_gradient_l2_norm",
            "velocity_gradient_l2_norm",
            "weight_gradient_l2_norm",
        ],
        "control_parameter_delta_l2_norm_keys": [
            "raw_color_parameter_delta_l2_norm",
            "raw_density_parameter_delta_l2_norm",
            "positions0_parameter_delta_l2_norm",
            "velocities_parameter_delta_l2_norm",
            "weight_coefficients_parameter_delta_l2_norm",
        ],
        "control_update_sha256_keys": [
            "parameters_before_digest",
            "parameters_after_digest",
            "gradient_digest",
            "update_authorization_digest",
            "generation_digest",
        ],
        "control_precompile_sha256_keys": [
            "provider_generation_digest",
            "selected_track_manifest_digest",
            "compiler_work_receipt_chain_digest",
            "generation_digest",
        ],
        "control_precompile_sha256_sequence_keys": [
            "artifact_generation_digests",
            "compile_receipt_generation_digests",
        ],
        "control_accounting_sha256_keys": [
            "frame_result_generation_digest_chain",
            "frame_readback_receipt_chain_digest",
            "compiler_work_receipt_chain_digest",
        ],
        "control_accounting_frame_sha256_sequence_keys": [
            "frame_result_generation_digests_retained",
        ],
        "control_accounting_nonempty_sha256_sequence_keys": [
            "selected_pixel_source_manifest_digests",
        ],
        "control_zero_accounting_keys": [
            "per_frame_continuous_recompile_count",
            "fresh_selected_track_recompile_count",
            "native_fused_union_v2_transaction_count",
            "sample_to_node_dense_fallback_interactions",
            "full_frame_target_materialization_count",
            "frame_result_capability_retained_after_release_count",
            "stale_provider_store_retirement_count",
            "persistent_frame_tensor_bytes",
            "persistent_sample_tensor_bytes",
            "persistent_target_tensor_bytes",
            "persistent_prediction_tensor_bytes",
            "optimizer_history_tensor_bytes",
        ],
        "control_positive_logical_accounting_keys": [
            "global_material_bar_logical_tensor_bytes",
            "global_geometry_bar_logical_tensor_bytes",
            "global_bar_and_loss_logical_tensor_bytes",
            "frame_material_bar_logical_tensor_bytes",
            "frame_geometry_bar_logical_tensor_bytes",
            "frame_material_readback_and_loss_logical_tensor_bytes",
            "frame_coordinator_visible_logical_tensor_bytes_upper_bound",
            "frame_geometry_bridge_visible_logical_tensor_bytes_upper_bound",
            "compiled_program_store_resident_accounted_bytes",
            "compiled_program_store_peak_resident_accounted_bytes",
            "combined_live_state_logical_tensor_bytes",
            "maximum_frame_local_logical_tensor_bytes_upper_bound",
            "expensive_live_logical_and_accounted_peak_upper_bound_bytes",
            "peak_lane_resident_logical_tensor_bytes",
            "peak_active_node_state_tensor_bytes",
            "peak_sample_launch_tensor_bytes",
            "peak_decoded_frame_scratch_upper_bound_bytes",
            "peak_selected_frame_target_tensor_upper_bound_bytes",
            "peak_coordinator_visible_live_tensor_upper_bound_bytes",
            "maximum_geometry_bridge_visible_peak_logical_tensor_bytes",
        ],
        "control_logical_accounting_keys": [
            "global_material_bar_logical_tensor_bytes",
            "global_geometry_bar_logical_tensor_bytes",
            "global_bar_and_loss_logical_tensor_bytes",
            "frame_material_bar_logical_tensor_bytes",
            "frame_geometry_bar_logical_tensor_bytes",
            "frame_material_readback_and_loss_logical_tensor_bytes",
            "frame_coordinator_visible_logical_tensor_bytes_upper_bound",
            "frame_geometry_bridge_visible_logical_tensor_bytes_upper_bound",
            "compiled_program_store_resident_accounted_bytes",
            "compiled_program_store_peak_resident_accounted_bytes",
            "combined_live_state_logical_tensor_bytes",
            "maximum_frame_local_logical_tensor_bytes_upper_bound",
            "expensive_live_logical_and_accounted_peak_upper_bound_bytes",
            "peak_lane_resident_logical_tensor_bytes",
            "peak_active_node_state_tensor_bytes",
            "peak_sample_launch_tensor_bytes",
            "peak_decoded_frame_scratch_upper_bound_bytes",
            "peak_selected_frame_target_tensor_upper_bound_bytes",
            "peak_coordinator_visible_live_tensor_upper_bound_bytes",
            "maximum_geometry_bridge_visible_peak_logical_tensor_bytes",
            "persistent_frame_tensor_bytes",
            "persistent_sample_tensor_bytes",
            "persistent_target_tensor_bytes",
            "persistent_prediction_tensor_bytes",
            "optimizer_history_tensor_bytes",
        ],
        "control_compile_invariant_work_keys": [
            "compiler_work_receipt_count",
            "root_complement_witness_count",
            "candidate_source_attempt_count",
            "all_site_witness_check_count",
            "unique_pair_difference_count",
        ],
        "control_exact_frame_linear_work_keys": [
            "per_frame_replay_count",
            "compiled_artifact_warm_acquisition_count",
            "compiled_artifact_warm_hit_count",
            "frame_readback_receipt_count",
            "frame_release_fence_call_count",
            "streamed_sample_count",
            "sample_to_node_linear_interactions",
            "selected_pixel_read_call_count",
            "direct_selected_pixel_observation_count",
            "camera_ray_slice_work_count",
            "camera_ray_slice_scalar_count",
            "frame_readback_cumulative_tensor_bytes",
        ],
        "control_allowed_frame_dependent_work_keys": [
            "eligible_native_block_count",
            "active_native_block_count",
            "native_node_forward_launch_count",
            "native_sample_prepare_count",
            "native_sample_launch_count",
            "native_sample_completion_fence_count",
            "native_full_geometry_vjp_launch_count",
            "geometry_d2h_completion_fence_count",
            "ordered_word_node_interactions",
        ],
    }
    for key, expected in exact_receipt_lists.items():
        if contract.get(key) != expected:
            raise ValueError(f"contract {key} changed")
    for list_key in (
        "zero_retained_frame_state_keys",
        "fused_frame_invariant_work_keys",
        "fused_fixed_schedule_linear_work_keys",
        "fused_allowed_streamed_work_keys",
        "required_memory_peak_keys",
        "control_update_gradient_l2_keys",
        "control_parameter_delta_l2_norm_keys",
        "control_update_sha256_keys",
        "control_precompile_sha256_keys",
        "control_precompile_sha256_sequence_keys",
        "control_accounting_sha256_keys",
        "control_accounting_frame_sha256_sequence_keys",
        "control_accounting_nonempty_sha256_sequence_keys",
        "control_zero_accounting_keys",
        "control_positive_logical_accounting_keys",
        "control_logical_accounting_keys",
        "control_compile_invariant_work_keys",
        "control_exact_frame_linear_work_keys",
        "control_allowed_frame_dependent_work_keys",
    ):
        values = contract.get(list_key)
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f"contract {list_key} must be a unique nonempty list")
    work_lists = (
        set(contract["fused_frame_invariant_work_keys"]),
        set(contract["fused_fixed_schedule_linear_work_keys"]),
        set(contract["fused_allowed_streamed_work_keys"]),
    )
    if any(left & right for index, left in enumerate(work_lists) for right in work_lists[index + 1 :]):
        raise ValueError("contract work classifications must be disjoint")
    site_count = int(contract["required_world_site_count"])
    weight_count = int(contract["required_weight_coefficient_count"])
    expected_bytes = {
        "cpu_material_parameter_state_bytes": 16 * site_count,
        "cpu_material_physical_snapshot_bytes": 16 * site_count,
        "cpu_material_gradient_buffer_bytes": 16 * site_count,
        "cpu_material_live_state_bytes": 48 * site_count,
        "trainable_live_geometry_state_bytes": 8
        * site_count
        * (6 + weight_count),
        "combined_live_state_bytes": (48 + 8 * (6 + weight_count))
        * site_count,
        "device_material_snapshot_bytes": 16 * site_count,
        "live_state_and_transient_training_bars_logical_bytes": (
            48 + 8 * (6 + weight_count) + 16 + 16 + 8 * (6 + weight_count)
        )
        * site_count,
        "global_material_bar_bytes": 16 * site_count,
        "transient_global_geometry_bar_bytes": 8
        * site_count
        * (6 + weight_count),
        "optimizer_history_tensor_bytes": 0,
        "combined_checkpoint_payload_bytes": (16 + 8 * (6 + weight_count))
        * site_count,
        "live_state_plus_checkpoint_bytes": (64 + 16 * (6 + weight_count))
        * site_count,
        "live_state_plus_checkpoint_payload_clone_peak_bytes": (
            80 + 24 * (6 + weight_count)
        )
        * site_count,
    }
    for key, expected in expected_bytes.items():
        if contract.get(key) != expected:
            raise ValueError(f"contract {key} violates combined-state accounting")


def _sha_is_valid(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _int(value: Any, *, minimum: int = 0) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= minimum


def _number(value: Any, *, minimum: float = 0.0) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= minimum
    )


def _required_row_keys(contract: Mapping[str, Any]) -> set[tuple[str, int, int]]:
    return {
        (row["mode"], int(row["requested_frame_count"]), repeat_index)
        for row in contract["required_rows"]
        for repeat_index in range(int(contract["required_repeat_count"]))
    }


def _required_control_row_keys(
    contract: Mapping[str, Any],
) -> set[tuple[str, int, int]]:
    return {
        (row["mode"], int(row["requested_frame_count"]), repeat_index)
        for row in contract["required_control_rows"]
        for repeat_index in range(int(contract["required_repeat_count"]))
    }


def _memory_delta(memory: Mapping[str, Any], prefix: str) -> int:
    return int(memory[f"{prefix}_peak_bytes"]) - int(
        memory[f"{prefix}_baseline_bytes"]
    )


def _append_parent_watchdog_failures(
    failures: list[str],
    measurement: Mapping[str, Any],
    *,
    label: str,
    contract: Mapping[str, Any],
) -> None:
    watchdog = measurement.get("parent_watchdog")
    if not isinstance(watchdog, Mapping):
        failures.append(f"{label}: parent watchdog receipt is missing")
        return
    required_keys = {
        "returncode",
        "elapsed_seconds",
        "rss_measurement_kind",
        "rss_sampling_interval_seconds",
        "sampled_process_group_rss_high_water_bytes",
        "sample_count",
        "worker_timeout_seconds",
        "worker_process_group_rss_limit_bytes",
        "watchdog_completed",
        "process_group_empty_after_exit",
        "worker_terminated_by_watchdog",
    }
    if set(watchdog) != required_keys:
        failures.append(f"{label}: parent watchdog receipt keys are noncanonical")
        return
    command_sha256 = measurement.get("worker_command_sha256")
    process_generation_id = measurement.get("process_generation_id")
    evidence_sha256 = measurement.get("parent_watchdog_evidence_sha256")
    if not _sha_is_valid(command_sha256):
        failures.append(f"{label}: worker command is not hash-bound")
    if not isinstance(process_generation_id, str) or not process_generation_id.strip():
        failures.append(f"{label}: watchdog process generation is missing")
    expected_evidence_sha256 = canonical_sha256(
        {
            "parent_watchdog": dict(watchdog),
            "process_generation_id": process_generation_id,
            "worker_command_sha256": command_sha256,
        }
    )
    if evidence_sha256 != expected_evidence_sha256:
        failures.append(f"{label}: parent watchdog evidence digest changed")
    elapsed = watchdog.get("elapsed_seconds")
    interval = watchdog.get("rss_sampling_interval_seconds")
    sampled_rss = watchdog.get("sampled_process_group_rss_high_water_bytes")
    sample_count = watchdog.get("sample_count")
    timeout = watchdog.get("worker_timeout_seconds")
    rss_limit = watchdog.get("worker_process_group_rss_limit_bytes")
    if not _number(elapsed, minimum=0.0) or float(elapsed) <= 0.0:
        failures.append(f"{label}: parent watchdog elapsed time is invalid")
    if interval != contract["required_worker_watchdog_poll_interval_seconds"]:
        failures.append(f"{label}: parent watchdog sampling interval changed")
    if (
        not _int(sampled_rss, minimum=1)
        or sampled_rss > contract["maximum_worker_process_group_rss_bytes"]
    ):
        failures.append(f"{label}: process-group RSS did not remain under the bound")
    if not _int(
        sample_count,
        minimum=contract["minimum_parent_watchdog_sample_count"],
    ):
        failures.append(f"{label}: parent watchdog has too few RSS samples")
    if timeout != contract["maximum_worker_timeout_seconds"]:
        failures.append(f"{label}: parent watchdog timeout changed")
    if rss_limit != contract["maximum_worker_process_group_rss_bytes"]:
        failures.append(f"{label}: parent watchdog RSS limit changed")
    if watchdog.get("rss_measurement_kind") != contract[
        "required_parent_watchdog_rss_measurement_kind"
    ]:
        failures.append(f"{label}: parent watchdog measurement kind changed")
    if (
        watchdog.get("returncode") != 0
        or watchdog.get("watchdog_completed") is not True
        or watchdog.get("process_group_empty_after_exit") is not True
        or watchdog.get("worker_terminated_by_watchdog") is not False
    ):
        failures.append(f"{label}: parent watchdog did not complete cleanly")


def _append_row_failures(
    failures: list[str],
    row: Mapping[str, Any],
    *,
    label: str,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    bindings: Mapping[str, Any],
) -> None:
    execution = row.get("execution")
    structure = row.get("structure")
    work = row.get("work")
    memory = row.get("memory")
    measurement = row.get("measurement")
    quality = row.get("quality")
    if not all(
        isinstance(value, Mapping)
        for value in (execution, structure, work, memory, measurement, quality)
    ):
        failures.append(f"{label}: row sections are missing")
        return
    expected_execution = {
        "adapter_provenance": "worldfoam-training-memory-ablation-adapter-v1",
        "real_native_spatial_block_coordinator": True,
        "fake_native_backend": False,
        "native_runtime_executed": True,
        # Native execution is attested by the producer and extension binding.
        # The source receipt deliberately does not duplicate that claim.
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
        "dataset_is_procedural_synthetic": True,
        "public_quality_evidence": False,
        "autograd_graph_retained": False,
        "worker_measurement_scope": "single_optimizer_step_scaling_row_v2",
        "worker_measurement_covers_checkpoint_and_uninterrupted_step_2": False,
        "parity_payload_scope": (
            "single_step_pre_update_gradient_and_post_update_parameters"
        ),
    }
    for key, expected in expected_execution.items():
        if execution.get(key) != expected:
            failures.append(f"{label}: execution.{key} must be {expected!r}")
    requested_frame_count = row.get("requested_frame_count")
    lifecycle_required = (
        row.get("mode") == contract["required_lifecycle_mode"]
        and requested_frame_count == contract["required_lifecycle_frame_count"]
    )
    # The paper scaling row is always a single optimizer transaction.  The
    # F=8 restart proof is attached from a separate auxiliary fresh process;
    # counting its checkpoint or second-step work here would confound the
    # memory and timing curve.
    expected_mutation_count = contract[
        "required_primary_scaling_optimizer_mutations_per_row"
    ]
    mutation_count = execution.get("cpu_optimizer_mutation_count")
    if mutation_count != expected_mutation_count:
        failures.append(
            f"{label}: execution.cpu_optimizer_mutation_count must be "
            f"{expected_mutation_count}"
        )
    for key in (
        "geometry_mutation_count",
        "material_device_gradient_receipt_count",
        "combined_optimizer_authorization_count",
        "stale_provider_store_retirement_count",
        "fresh_selected_track_recompile_count",
    ):
        if execution.get(key) != expected_mutation_count:
            failures.append(
                f"{label}: execution.{key} must match measured optimizer mutations"
            )
    geometry_d2h_count = execution.get("geometry_d2h_receipt_count")
    if not _int(geometry_d2h_count, minimum=expected_mutation_count):
        failures.append(f"{label}: geometry D2H receipt coverage is incomplete")
    if execution.get("cold_cpu_compile_measurement_count") != expected_mutation_count:
        failures.append(f"{label}: cold CPU compile measurement count changed")
    for key in (
        "step_wall_time_seconds",
        "core_forward_backward_wall_time_seconds",
        "cold_cpu_compile_wall_time_seconds",
    ):
        value = execution.get(key)
        if not _number(value, minimum=0.0) or float(value) <= 0.0:
            failures.append(f"{label}: execution.{key} is not a measured duration")
    total_delta = execution.get("cpu_optimizer_parameter_delta_l2")
    if not _number(total_delta, minimum=0.0) or float(total_delta) <= 0.0:
        failures.append(f"{label}: CPU optimizer did not change parameters")

    gradient_update = execution.get("gradient_update")
    gradient_keys = (
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
    if not isinstance(gradient_update, Mapping):
        failures.append(f"{label}: execution.gradient_update is missing")
    else:
        for key in gradient_keys:
            value = gradient_update.get(key)
            if not _number(value, minimum=0.0) or float(value) <= 0.0:
                failures.append(
                    f"{label}: execution.gradient_update.{key} must be finite and nonzero"
                )
        delta_keys = (
            "raw_color_parameter_delta_l2_norm",
            "raw_density_parameter_delta_l2_norm",
            "positions0_parameter_delta_l2_norm",
            "velocities_parameter_delta_l2_norm",
            "weight_coefficients_parameter_delta_l2_norm",
        )
        if all(_number(gradient_update.get(key), minimum=0.0) for key in delta_keys):
            reconstructed_delta = math.sqrt(
                sum(float(gradient_update[key]) ** 2 for key in delta_keys)
            )
            if not _number(total_delta, minimum=0.0) or not math.isclose(
                float(total_delta),
                reconstructed_delta,
                rel_tol=1.0e-6,
                abs_tol=1.0e-12,
            ):
                failures.append(f"{label}: combined optimizer delta norm changed")

    for key in (
        "bridge_receipt_generation_digest",
        "authorization_generation_digest",
        "combined_update_receipt_generation_digest",
        "provider_store_retirement_receipt_chain_sha256",
        "fresh_selected_track_recompile_receipt_sha256",
    ):
        if not _sha_is_valid(execution.get(key)):
            failures.append(f"{label}: execution.{key} is not hash-bound")
    d2h_digests = execution.get("geometry_d2h_receipt_generation_digests")
    if (
        not isinstance(d2h_digests, Sequence)
        or isinstance(d2h_digests, (str, bytes))
        or not d2h_digests
        or not all(_sha_is_valid(value) for value in d2h_digests)
    ):
        failures.append(f"{label}: geometry D2H receipt digests are invalid")
    elif len(d2h_digests) != geometry_d2h_count:
        failures.append(f"{label}: geometry D2H receipt count/digests disagree")
    if not _int(
        execution.get("fresh_selected_track_recompile_request_count"), minimum=1
    ):
        failures.append(f"{label}: selected-track cold recompile did not run")
    for forbidden in (
        "combined_update_receipt_chain_sha256",
        "geometry_d2h_receipt_chain_sha256",
        "fresh_selected_track_recompile_receipt_chain_sha256",
    ):
        if forbidden in execution:
            failures.append(
                f"{label}: primary scaling execution contains auxiliary "
                f"lifecycle field {forbidden}"
            )

    expected_structure = {
        "image_height": 384,
        "image_width": 512,
        "dataset_frame_count": 300,
        "world_site_count": 1024,
        "weight_coefficient_count": 2,
        "fixed_track_count": 512,
        "selected_frame_count": requested_frame_count,
        "node_count": 4,
        "compile_certification_mode": "all_competitor_active_owner",
        "maximum_sites_per_track_compile": 1024,
        "track_manifest_sha256": contract["required_track_manifest_sha256"],
        "camera_program_sha256": contract["required_camera_program_sha256"],
        "target_teacher_sha256": contract["required_target_teacher_sha256"],
        "loss": contract["required_loss"],
    }
    for key, expected in expected_structure.items():
        if structure.get(key) != expected:
            failures.append(f"{label}: structure.{key} changed")
    for key in (
        "compiled_world_sha256",
        "physical_grid_sha256",
        "camera_grid_sha256",
        "spatial_block_manifest_sha256",
        "target_stream_manifest_sha256",
        "observation_manifest_sha256",
        "provider_generation_before_sha256",
        "provider_generation_after_sha256",
        "factory_generation_sha256",
        "selected_track_recompile_manifest_sha256",
        "optimizer_policy_sha256",
    ):
        if not _sha_is_valid(structure.get(key)):
            failures.append(f"{label}: structure.{key} is not hash-bound")

    if _int(requested_frame_count, minimum=1):
        observation_count = 512 * requested_frame_count
        if structure.get("expected_observation_count") != observation_count:
            failures.append(f"{label}: structure.expected_observation_count changed")
        if structure.get("loss_element_count") != 3 * observation_count:
            failures.append(f"{label}: structure.loss_element_count changed")
        if structure.get("global_loss_denominator") != 3 * observation_count:
            failures.append(f"{label}: structure.global_loss_denominator changed")
    if structure.get("provider_generation_before_sha256") == structure.get(
        "provider_generation_after_sha256"
    ):
        failures.append(f"{label}: provider generation did not advance after SGD")

    core = work.get("core_accounting")
    if not isinstance(core, Mapping):
        failures.append(f"{label}: work.core_accounting is missing")
        core = {}
    if work.get("compile_track_count") != 512:
        failures.append(f"{label}: compiler track coverage changed")
    if work.get("compiler_work_receipt_count") != 512:
        failures.append(f"{label}: compiler receipt coverage changed")
    if work.get("compiler_work_receipt_chain_link_count") != 512:
        failures.append(f"{label}: compiler receipt chain coverage changed")
    if work.get("compiler_work_receipt_bundle_count") != work.get(
        "spatial_bundle_count"
    ):
        failures.append(f"{label}: compiler receipt bundle count changed")
    # These are sums of the compiler's own ActiveKineticCompilerWork receipts.
    # Candidate-source construction is indexed by unique witnessed owner words,
    # while all-site checks are indexed by root-complement witnesses.  Their
    # run-count populations differ, so no aggregate scalar identity between
    # them is valid.  Preserve the exact counters and the compiler's own
    # per-witness-bound attestation instead of inventing a proxy equation.
    for key in (
        "root_complement_witness_count",
        "candidate_source_attempt_count",
        "all_site_witness_check_count",
        "unique_pair_difference_count",
    ):
        if not _int(work.get(key), minimum=1):
            failures.append(f"{label}: compiler receipt work.{key} is invalid")
    for key, expected in {
        "per_witness_candidate_bound_verified": True,
        "exhaustive_triple_enumeration_used": False,
        "requested_frame_sampling_used": False,
        "active_compiler_accounting_complete": True,
        "all_track_receipt_digests_verified": True,
    }.items():
        if work.get(key) is not expected:
            failures.append(f"{label}: compiler receipt work.{key} changed")
    if not isinstance(work.get("compiler_work_receipt_provenance"), str) or not work[
        "compiler_work_receipt_provenance"
    ]:
        failures.append(f"{label}: compiler receipt provenance is missing")
    if not _sha_is_valid(work.get("compiler_work_receipt_chain_digest")):
        failures.append(f"{label}: compiler receipt chain is not hash-bound")
    for key in (
        "retained_compiled_program_count",
        "retained_compiler_receipt_entry_count",
        "retained_compiler_tensor_bytes",
        "sample_to_node_dense_fallback_interactions",
    ):
        if work.get(key) != 0:
            failures.append(f"{label}: work.{key} must be exactly zero")
    if not isinstance(work.get("compiler_receipt_state_scaling"), str) or not work[
        "compiler_receipt_state_scaling"
    ]:
        failures.append(f"{label}: compiler receipt state scaling is missing")

    projected_work_keys = (
        "compile_track_count",
        "compiler_work_receipt_count",
        "compiler_work_receipt_bundle_count",
        "compiler_work_receipt_chain_link_count",
        "root_complement_witness_count",
        "candidate_source_attempt_count",
        "all_site_witness_check_count",
        "unique_pair_difference_count",
        "per_witness_candidate_bound_verified",
        "exhaustive_triple_enumeration_used",
        "requested_frame_sampling_used",
        "active_compiler_accounting_complete",
        "all_track_receipt_digests_verified",
        "compiler_work_receipt_provenance",
        "compiler_work_receipt_chain_digest",
        "retained_compiled_program_count",
        "retained_compiler_receipt_entry_count",
        "retained_compiler_tensor_bytes",
        "compiler_receipt_state_scaling",
        "spatial_bundle_count",
        "eligible_native_block_count",
        "active_native_block_count",
        "native_node_forward_launch_count",
        "native_material_word_vjp_launch_count",
        "native_full_geometry_vjp_launch_count",
        "native_fused_union_v2_transaction_count",
        "ordered_word_node_interactions",
        "streamed_sample_count",
        "sample_to_node_linear_interactions",
        "sample_to_node_dense_fallback_interactions",
        "selected_pixel_read_call_count",
        "direct_selected_pixel_observation_count",
        "camera_ray_slice_work_count",
        "camera_ray_slice_scalar_count",
    )
    for key in projected_work_keys:
        if work.get(key) != core.get(key):
            failures.append(f"{label}: work.{key} is not bound to core accounting")
    for key in (
        "spatial_bundle_count",
        "eligible_native_block_count",
        "active_native_block_count",
        "native_node_forward_launch_count",
        "ordered_word_node_interactions",
        "selected_pixel_read_call_count",
    ):
        if not _int(work.get(key), minimum=1):
            failures.append(f"{label}: work.{key} must be positive")
    if work.get("active_native_block_count", 0) > work.get(
        "eligible_native_block_count", 0
    ):
        failures.append(f"{label}: active native blocks exceed eligible blocks")
    if work.get("native_node_forward_launch_count") != work.get(
        "active_native_block_count"
    ):
        failures.append(f"{label}: forward launch coverage changed")
    mode = row.get("mode")
    if mode == "staged_sparse":
        if work.get("native_material_word_vjp_launch_count") != work.get(
            "active_native_block_count"
        ):
            failures.append(f"{label}: staged material VJP coverage changed")
        if work.get("native_full_geometry_vjp_launch_count") != work.get(
            "active_native_block_count"
        ):
            failures.append(f"{label}: staged geometry VJP coverage changed")
        if work.get("native_fused_union_v2_transaction_count") != 0 or core.get(
            "native_fused_union_v2_vjp_launch_count"
        ) != 0:
            failures.append(f"{label}: staged row executed fused transactions")
    elif mode == "fused_union_v2":
        for key in (
            "native_material_word_vjp_launch_count",
            "native_full_geometry_vjp_launch_count",
        ):
            if work.get(key) != 0:
                failures.append(f"{label}: fused work.{key} must be exactly zero")
        if core.get("native_fused_union_v2_vjp_launch_count") != work.get(
            "active_native_block_count"
        ):
            failures.append(f"{label}: fused VJP coverage changed")
        if work.get("native_fused_union_v2_transaction_count") != work.get(
            "spatial_bundle_count"
        ):
            failures.append(f"{label}: fused transaction coverage changed")
    if _int(requested_frame_count, minimum=1):
        expected_samples = 512 * requested_frame_count
        exact_sample_work = {
            "streamed_sample_count": expected_samples,
            "direct_selected_pixel_observation_count": expected_samples,
            "camera_ray_slice_work_count": expected_samples,
            "camera_ray_slice_scalar_count": 6 * expected_samples,
            "sample_to_node_linear_interactions": 4 * expected_samples,
        }
        for key, expected in exact_sample_work.items():
            if work.get(key) != expected:
                failures.append(f"{label}: work.{key} changed")
        expected_bundle_count = {8: 4, 64: 8, 300: 40}.get(
            requested_frame_count
        )
        if work.get("spatial_bundle_count") != expected_bundle_count:
            failures.append(f"{label}: bounded spatial bundle schedule changed")
    if work.get("fresh_selected_track_recompile_track_count") != 512:
        failures.append(f"{label}: selected-track recompile coverage changed")
    first_step_recompile_requests = work.get(
        "fresh_selected_track_recompile_request_count"
    )
    if not _int(first_step_recompile_requests, minimum=1):
        failures.append(f"{label}: selected-track recompile requests are missing")
    elif execution.get("fresh_selected_track_recompile_request_count") != (
        expected_mutation_count * first_step_recompile_requests
    ):
        failures.append(f"{label}: recompile request count/lifecycle coverage changed")

    for key in contract["required_memory_peak_keys"]:
        if not _int(memory.get(key), minimum=0):
            failures.append(f"{label}: memory.{key} is missing")
    for key in contract["zero_retained_frame_state_keys"]:
        if memory.get(key) != 0:
            failures.append(f"{label}: memory.{key} must be exactly zero")
    fixed_memory = {
        "retained_driver_input_tensor_bytes": 0,
        "combined_live_state_logical_tensor_bytes": contract[
            "combined_live_state_bytes"
        ],
        "material_state_logical_tensor_bytes": contract[
            "cpu_material_live_state_bytes"
        ],
        "trainable_geometry_state_logical_tensor_bytes": contract[
            "trainable_live_geometry_state_bytes"
        ],
        "global_material_bar_logical_tensor_bytes": contract[
            "global_material_bar_bytes"
        ],
        "global_geometry_bar_logical_tensor_bytes": contract[
            "transient_global_geometry_bar_bytes"
        ],
        "optimizer_history_tensor_bytes": contract[
            "optimizer_history_tensor_bytes"
        ],
        "measured_peak_fields_producer_owned": True,
        "logical_bounds_are_measured_peaks": False,
    }
    for key, expected in fixed_memory.items():
        if memory.get(key) != expected:
            failures.append(f"{label}: memory.{key} changed")
    for key in (
        "peak_lane_resident_logical_tensor_bytes",
        "peak_active_node_state_logical_tensor_bytes",
        "peak_sample_launch_logical_tensor_bytes",
        "peak_coordinator_visible_logical_tensor_upper_bound_bytes",
        "maximum_geometry_bridge_visible_logical_tensor_bytes",
        "combined_update_authorization_logical_tensor_bytes",
        "combined_update_transaction_tracked_logical_and_store_accounted_upper_bound_bytes",
    ):
        if not _int(memory.get(key), minimum=1):
            failures.append(f"{label}: memory.{key} must be a positive logical bound")
    for key in (
        "peak_coordinator_visible_logical_tensor_upper_bound_bytes",
        "maximum_geometry_bridge_visible_logical_tensor_bytes",
    ):
        if _int(memory.get(key), minimum=1) and memory[key] > contract[
            "maximum_bridge_visible_peak_bytes"
        ]:
            failures.append(f"{label}: memory.{key} exceeds the logical bridge bound")
    transaction_bound = memory.get(
        "combined_update_transaction_tracked_logical_and_store_accounted_upper_bound_bytes"
    )
    if _int(transaction_bound, minimum=1) and transaction_bound > contract[
        "maximum_worker_process_group_rss_bytes"
    ]:
        failures.append(f"{label}: combined update logical bound exceeds worker RSS limit")
    if all(
        _int(memory.get(key), minimum=0)
        for key in (
            "process_rss_baseline_bytes",
            "process_rss_peak_bytes",
            "sampled_mps_current_baseline_bytes",
            "sampled_mps_current_peak_bytes",
            "sampled_mps_driver_baseline_bytes",
            "sampled_mps_driver_peak_bytes",
        )
    ):
        if memory["process_rss_peak_bytes"] < memory["process_rss_baseline_bytes"]:
            failures.append(f"{label}: process RSS peak precedes its baseline")
        if memory["sampled_mps_current_peak_bytes"] < memory["sampled_mps_current_baseline_bytes"]:
            failures.append(f"{label}: sampled MPS current peak precedes baseline")
        if memory["sampled_mps_driver_peak_bytes"] < memory["sampled_mps_driver_baseline_bytes"]:
            failures.append(f"{label}: sampled MPS driver peak precedes baseline")
        if _memory_delta(memory, "process_rss") > contract["maximum_process_rss_peak_delta_bytes"]:
            failures.append(f"{label}: process RSS delta exceeds contract")
        if _memory_delta(memory, "sampled_mps_current") > contract["maximum_sampled_mps_current_delta_bytes"]:
            failures.append(f"{label}: sampled MPS current delta exceeds contract")
        if _memory_delta(memory, "sampled_mps_driver") > contract["maximum_sampled_mps_driver_delta_bytes"]:
            failures.append(f"{label}: sampled MPS driver delta exceeds contract")
        if memory["sampled_mps_current_peak_bytes"] > contract[
            "maximum_mps_working_set_bytes"
        ]:
            failures.append(f"{label}: sampled MPS tensor peak exceeds the working-set bound")
        if memory["sampled_mps_driver_peak_bytes"] > contract[
            "maximum_mps_working_set_bytes"
        ]:
            failures.append(f"{label}: sampled MPS driver peak exceeds the working-set bound")
        if memory.get("parent_process_group_rss_sampled_peak_bytes", 0) > contract[
            "maximum_worker_process_group_rss_bytes"
        ]:
            failures.append(f"{label}: sampled process-group RSS exceeds the watchdog bound")

    expected_measurement = {
        "fresh_process": True,
        "measurement_kind": "fresh-process-mps-and-rss-sampled-high-water-v1",
        "completion_fenced_before_final_measurement": True,
        "allocator_exact_peak_claimed": False,
        "mps_memory_limit_bytes": contract["maximum_mps_working_set_bytes"],
        "process_group_rss_limit_bytes": contract["maximum_worker_process_group_rss_bytes"],
    }
    for key, expected in expected_measurement.items():
        if measurement.get(key) != expected:
            failures.append(f"{label}: measurement.{key} changed")
    _append_parent_watchdog_failures(
        failures,
        measurement,
        label=label,
        contract=contract,
    )
    if isinstance(measurement.get("parent_watchdog"), Mapping) and (
        measurement["parent_watchdog"].get(
            "sampled_process_group_rss_high_water_bytes"
        )
        != memory.get("parent_process_group_rss_sampled_peak_bytes")
    ):
        failures.append(f"{label}: parent watchdog RSS peak is not bound to row memory")
    if not _int(
        measurement.get("mps_memory_sample_count"),
        minimum=contract["minimum_fresh_process_mps_sample_count"],
    ):
        failures.append(f"{label}: insufficient MPS high-water samples")
    if not isinstance(measurement.get("process_generation_id"), str) or not measurement[
        "process_generation_id"
    ].strip():
        failures.append(f"{label}: process generation is missing")
    row_bindings = measurement.get("bindings")
    if not isinstance(row_bindings, Mapping):
        failures.append(f"{label}: measurement bindings are missing")
    else:
        for key in ROW_BINDING_SHA_KEYS:
            if row_bindings.get(key) != bindings.get(key):
                failures.append(f"{label}: measurement binding {key} drifted")

    expected_quality_keys = {
        "finite",
        "loss_before_update",
        "post_update_loss_measured",
        "all_material_and_geometry_gradient_l2_norms_nonzero",
        "all_material_and_geometry_parameter_delta_l2_norms_nonzero",
    }
    if set(quality) != expected_quality_keys:
        failures.append(f"{label}: quality receipt keys are noncanonical")
    if quality.get("finite") is not True:
        failures.append(f"{label}: quality values are not certified finite")
    if not _number(quality.get("loss_before_update"), minimum=0.0):
        failures.append(f"{label}: quality.loss_before_update is invalid")
    if quality.get("post_update_loss_measured") is not False:
        failures.append(f"{label}: one-step row must not invent a post-update loss")
    if "loss_after" in quality:
        failures.append(f"{label}: unmeasured quality.loss_after is forbidden")
    for key in (
        "all_material_and_geometry_gradient_l2_norms_nonzero",
        "all_material_and_geometry_parameter_delta_l2_norms_nonzero",
    ):
        if quality.get(key) is not True:
            failures.append(f"{label}: quality.{key} must be True")
    lifecycle = row.get("lifecycle")
    if lifecycle_required:
        if not isinstance(lifecycle, Mapping):
            failures.append(f"{label}: F=8 fused lifecycle evidence is missing")
        else:
            _append_lifecycle_failures(
                failures,
                lifecycle,
                label=label,
                row_process_generation_id=measurement.get("process_generation_id"),
                bindings=bindings,
                execution=execution,
                quality=quality,
                contract=contract,
            )
    elif lifecycle is not None:
        failures.append(f"{label}: lifecycle evidence is only valid for fused F=8")
    if row.get("evidence_sha256") != row_evidence_sha256(row):
        failures.append(f"{label}: row evidence digest changed")


_LIFECYCLE_DELTA_KEYS = (
    "raw_color_parameter_delta_l2_norm",
    "raw_density_parameter_delta_l2_norm",
    "positions0_parameter_delta_l2_norm",
    "velocities_parameter_delta_l2_norm",
    "weight_coefficients_parameter_delta_l2_norm",
)

_LIFECYCLE_KEYS = frozenset(
    {
        "performed",
        "step_count",
        "checkpoint_created_after_step",
        "checkpoint_restore_used",
        "restart_fresh_process",
        "restart_process_generation_id",
        "restart_hardware_fingerprint_sha256",
        "restart_source_manifest_sha256",
        "restart_native_source_sha256",
        "restart_native_extension_sha256",
        "restart_worker_command_sha256",
        "restart_parent_watchdog",
        "restart_parent_watchdog_evidence_sha256",
        "primary_scaling_worker_step_count",
        "primary_scaling_worker_checkpoint_count",
        "primary_scaling_worker_measurement_excludes_auxiliary_lifecycle",
        "auxiliary_lifecycle_worker",
        "auxiliary_step_1_matches_primary_scaling_row",
        "loss_step_1_pre_update",
        "loss_step_1_pre_update_auxiliary",
        "loss_step_2_uninterrupted_pre_update",
        "loss_step_2_restored_pre_update",
        "step_1_to_step_2_pre_update_loss_decrease",
        "restart_loss_absolute_error",
        "step_1_gradient_sha256_primary",
        "step_1_gradient_sha256_auxiliary",
        "step_1_parameters_after_step_sha256_primary",
        "step_1_parameters_after_step_sha256_auxiliary",
        "step_1_parameter_delta_l2_primary",
        "step_1_parameter_delta_l2_auxiliary",
        "step_1_update_content_sha256_primary",
        "step_1_update_content_sha256_auxiliary",
        "step_1_update_receipt_generation_sha256_primary",
        "step_1_update_receipt_generation_sha256_auxiliary",
        "step_2_gradient_sha256_uninterrupted",
        "step_2_gradient_sha256_restored",
        "step_2_gradient_content_match",
        "step_2_parameters_after_step_sha256_uninterrupted",
        "step_2_parameters_after_step_sha256_restored",
        "step_2_parameter_delta_l2_uninterrupted",
        "step_2_parameter_delta_l2_restored",
        "step_2_state_sha256_uninterrupted",
        "step_2_state_sha256_restored",
        "step_2_state_content_match",
        "step_2_update_content_sha256_uninterrupted",
        "step_2_update_content_sha256_restored",
        "step_2_update_content_match",
        "uninterrupted_process_optimizer_mutation_count",
        "fresh_restart_optimizer_mutation_count",
        "auxiliary_optimizer_mutation_count",
        "post_step_1_loss_measured_by_step_2_pre_update",
        "measurement_includes_checkpoint_and_uninterrupted_second_step",
        "maximum_simultaneously_retained_world_count",
        "uninterrupted_world_released_before_restore",
        "restore_receipt",
        "checkpoint_sha256",
        "combined_checkpoint_payload_bytes",
        "live_state_logical_tensor_bytes_at_checkpoint",
        "live_state_plus_checkpoint_bytes",
        "live_state_plus_checkpoint_payload_clone_peak_bytes",
        "optimizer_history_tensor_bytes",
    }
)


def _lifecycle_update_content_sha256(
    *,
    loss: Any,
    gradient_sha256: Any,
    parameters_after_step_sha256: Any,
    parameter_delta_l2: Any,
) -> str | None:
    if (
        not _number(loss, minimum=0.0)
        or not _sha_is_valid(gradient_sha256)
        or not _sha_is_valid(parameters_after_step_sha256)
        or not isinstance(parameter_delta_l2, Mapping)
        or set(parameter_delta_l2) != set(_LIFECYCLE_DELTA_KEYS)
        or any(
            not _number(parameter_delta_l2.get(key), minimum=0.0)
            or float(parameter_delta_l2[key]) <= 0.0
            for key in _LIFECYCLE_DELTA_KEYS
        )
    ):
        return None
    return canonical_sha256(
        {
            "loss_pre_update": loss,
            "gradient_sha256": gradient_sha256,
            "parameters_after_step_sha256": parameters_after_step_sha256,
            "parameter_delta_l2": dict(parameter_delta_l2),
        }
    )


def _append_lifecycle_failures(
    failures: list[str],
    lifecycle: Mapping[str, Any],
    *,
    label: str,
    row_process_generation_id: Any,
    bindings: Mapping[str, Any],
    execution: Mapping[str, Any],
    quality: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    if set(lifecycle) != _LIFECYCLE_KEYS:
        failures.append(f"{label}: lifecycle receipt keys are noncanonical")
    expected = {
        "performed": True,
        "step_count": 2,
        "checkpoint_created_after_step": 1,
        "checkpoint_restore_used": True,
        "restart_fresh_process": True,
        "restart_hardware_fingerprint_sha256": bindings[
            "hardware_fingerprint_sha256"
        ],
        "restart_source_manifest_sha256": bindings["source_manifest_sha256"],
        "restart_native_source_sha256": bindings["native_source_sha256"],
        "restart_native_extension_sha256": bindings[
            "native_extension_sha256"
        ],
        "combined_checkpoint_payload_bytes": contract[
            "combined_checkpoint_payload_bytes"
        ],
        "live_state_logical_tensor_bytes_at_checkpoint": contract[
            "combined_live_state_bytes"
        ],
        "live_state_plus_checkpoint_bytes": contract[
            "live_state_plus_checkpoint_bytes"
        ],
        "live_state_plus_checkpoint_payload_clone_peak_bytes": contract[
            "live_state_plus_checkpoint_payload_clone_peak_bytes"
        ],
        "optimizer_history_tensor_bytes": 0,
        "primary_scaling_worker_step_count": 1,
        "primary_scaling_worker_checkpoint_count": contract[
            "required_primary_scaling_checkpoint_count_per_row"
        ],
        "primary_scaling_worker_measurement_excludes_auxiliary_lifecycle": True,
        "auxiliary_lifecycle_worker": True,
        "auxiliary_step_1_matches_primary_scaling_row": True,
        "uninterrupted_process_optimizer_mutation_count": contract[
            "required_auxiliary_uninterrupted_process_optimizer_mutations"
        ],
        "fresh_restart_optimizer_mutation_count": contract[
            "required_auxiliary_fresh_restart_optimizer_mutations"
        ],
        "auxiliary_optimizer_mutation_count": contract[
            "required_auxiliary_lifecycle_optimizer_mutations"
        ],
        "post_step_1_loss_measured_by_step_2_pre_update": True,
        "measurement_includes_checkpoint_and_uninterrupted_second_step": False,
        "maximum_simultaneously_retained_world_count": 1,
        "uninterrupted_world_released_before_restore": True,
        "step_2_gradient_content_match": True,
        "step_2_state_content_match": True,
        "step_2_update_content_match": True,
    }
    for key, value in expected.items():
        if lifecycle.get(key) != value:
            failures.append(f"{label}: lifecycle.{key} changed")
    losses = tuple(
        lifecycle.get(key)
        for key in (
            "loss_step_1_pre_update",
            "loss_step_1_pre_update_auxiliary",
            "loss_step_2_uninterrupted_pre_update",
            "loss_step_2_restored_pre_update",
        )
    )
    if not all(_number(value, minimum=0.0) for value in losses):
        failures.append(f"{label}: lifecycle losses are invalid")
    else:
        first, auxiliary_first, uninterrupted, restored = map(float, losses)
        if auxiliary_first != first:
            failures.append(f"{label}: auxiliary step-1 loss differs from primary")
        if not uninterrupted < first:
            failures.append(
                f"{label}: the first update did not lower the step-2 pre-update loss"
            )
        observed_decrease = lifecycle.get(
            "step_1_to_step_2_pre_update_loss_decrease"
        )
        if (
            not _number(observed_decrease, minimum=0.0)
            or float(observed_decrease) <= 0.0
            or not math.isclose(
                float(observed_decrease),
                first - uninterrupted,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            failures.append(f"{label}: lifecycle loss-decrease receipt changed")
        if abs(uninterrupted - restored) > contract["maximum_restart_loss_absolute_error"]:
            failures.append(f"{label}: restored loss differs from uninterrupted loss")
    restart_loss_error = lifecycle.get("restart_loss_absolute_error")
    if (
        not _number(restart_loss_error, minimum=0.0)
        or float(restart_loss_error)
        > contract["maximum_restart_loss_absolute_error"]
    ):
        failures.append(f"{label}: lifecycle.restart_loss_absolute_error exceeds contract")
    elif all(_number(value, minimum=0.0) for value in losses) and not math.isclose(
        float(restart_loss_error),
        abs(float(losses[2]) - float(losses[3])),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        failures.append(f"{label}: lifecycle restart loss error is not receipt-bound")
    if not _sha_is_valid(lifecycle.get("checkpoint_sha256")):
        failures.append(f"{label}: checkpoint is not hash-bound")
    for kind in ("gradient", "state"):
        uninterrupted_key = f"step_2_{kind}_sha256_uninterrupted"
        restored_key = f"step_2_{kind}_sha256_restored"
        if (
            not _sha_is_valid(lifecycle.get(uninterrupted_key))
            or lifecycle.get(uninterrupted_key) != lifecycle.get(restored_key)
        ):
            failures.append(f"{label}: restart {kind} content parity failed")
    step_1_gradient_primary = lifecycle.get("step_1_gradient_sha256_primary")
    step_1_gradient_auxiliary = lifecycle.get("step_1_gradient_sha256_auxiliary")
    if (
        not _sha_is_valid(step_1_gradient_primary)
        or step_1_gradient_primary != step_1_gradient_auxiliary
    ):
        failures.append(f"{label}: auxiliary step-1 gradient differs from primary")
    step_1_parameters_primary = lifecycle.get(
        "step_1_parameters_after_step_sha256_primary"
    )
    step_1_parameters_auxiliary = lifecycle.get(
        "step_1_parameters_after_step_sha256_auxiliary"
    )
    if (
        not _sha_is_valid(step_1_parameters_primary)
        or step_1_parameters_primary != step_1_parameters_auxiliary
    ):
        failures.append(f"{label}: auxiliary step-1 parameters differ from primary")
    step_1_delta_primary = lifecycle.get("step_1_parameter_delta_l2_primary")
    step_1_delta_auxiliary = lifecycle.get("step_1_parameter_delta_l2_auxiliary")
    if step_1_delta_primary != step_1_delta_auxiliary:
        failures.append(f"{label}: auxiliary step-1 deltas differ from primary")
    gradient_update = execution.get("gradient_update")
    expected_primary_delta = (
        {key: gradient_update.get(key) for key in _LIFECYCLE_DELTA_KEYS}
        if isinstance(gradient_update, Mapping)
        else None
    )
    if step_1_delta_primary != expected_primary_delta:
        failures.append(f"{label}: lifecycle step-1 deltas are not row-bound")
    step_1_update_primary = lifecycle.get("step_1_update_content_sha256_primary")
    step_1_update_auxiliary = lifecycle.get(
        "step_1_update_content_sha256_auxiliary"
    )
    recomputed_step_1_primary = _lifecycle_update_content_sha256(
        loss=lifecycle.get("loss_step_1_pre_update"),
        gradient_sha256=step_1_gradient_primary,
        parameters_after_step_sha256=step_1_parameters_primary,
        parameter_delta_l2=step_1_delta_primary,
    )
    recomputed_step_1_auxiliary = _lifecycle_update_content_sha256(
        loss=lifecycle.get("loss_step_1_pre_update_auxiliary"),
        gradient_sha256=step_1_gradient_auxiliary,
        parameters_after_step_sha256=step_1_parameters_auxiliary,
        parameter_delta_l2=step_1_delta_auxiliary,
    )
    if (
        recomputed_step_1_primary is None
        or recomputed_step_1_auxiliary is None
        or step_1_update_primary != recomputed_step_1_primary
        or step_1_update_auxiliary != recomputed_step_1_auxiliary
        or step_1_update_primary != step_1_update_auxiliary
    ):
        failures.append(f"{label}: auxiliary step-1 update differs from primary")
    if lifecycle.get("loss_step_1_pre_update") != quality.get("loss_before_update"):
        failures.append(f"{label}: lifecycle step-1 loss is not bound to primary row")
    if (
        lifecycle.get("step_1_update_receipt_generation_sha256_primary")
        != execution.get("combined_update_receipt_generation_digest")
    ):
        failures.append(f"{label}: lifecycle primary step-1 receipt is not row-bound")
    for key in (
        "step_1_update_receipt_generation_sha256_primary",
        "step_1_update_receipt_generation_sha256_auxiliary",
    ):
        if not _sha_is_valid(lifecycle.get(key)):
            failures.append(f"{label}: lifecycle.{key} is not hash-bound")

    step_2_parameters_uninterrupted = lifecycle.get(
        "step_2_parameters_after_step_sha256_uninterrupted"
    )
    step_2_parameters_restored = lifecycle.get(
        "step_2_parameters_after_step_sha256_restored"
    )
    if (
        not _sha_is_valid(step_2_parameters_uninterrupted)
        or step_2_parameters_uninterrupted != step_2_parameters_restored
    ):
        failures.append(f"{label}: restart parameter content parity failed")
    step_2_delta_uninterrupted = lifecycle.get(
        "step_2_parameter_delta_l2_uninterrupted"
    )
    step_2_delta_restored = lifecycle.get("step_2_parameter_delta_l2_restored")
    if step_2_delta_uninterrupted != step_2_delta_restored:
        failures.append(f"{label}: restart parameter deltas differ")
    uninterrupted_update = lifecycle.get(
        "step_2_update_content_sha256_uninterrupted"
    )
    restored_update = lifecycle.get("step_2_update_content_sha256_restored")
    recomputed_uninterrupted_update = _lifecycle_update_content_sha256(
        loss=lifecycle.get("loss_step_2_uninterrupted_pre_update"),
        gradient_sha256=lifecycle.get("step_2_gradient_sha256_uninterrupted"),
        parameters_after_step_sha256=step_2_parameters_uninterrupted,
        parameter_delta_l2=step_2_delta_uninterrupted,
    )
    recomputed_restored_update = _lifecycle_update_content_sha256(
        loss=lifecycle.get("loss_step_2_restored_pre_update"),
        gradient_sha256=lifecycle.get("step_2_gradient_sha256_restored"),
        parameters_after_step_sha256=step_2_parameters_restored,
        parameter_delta_l2=step_2_delta_restored,
    )
    if (
        recomputed_uninterrupted_update is None
        or recomputed_restored_update is None
        or uninterrupted_update != recomputed_uninterrupted_update
        or restored_update != recomputed_restored_update
        or uninterrupted_update != restored_update
    ):
        failures.append(
            f"{label}: fresh-process update content parity failed"
        )
    restore_receipt = lifecycle.get("restore_receipt")
    if not isinstance(restore_receipt, Mapping):
        failures.append(f"{label}: checkpoint restore receipt is missing")
    else:
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
        ):
            if not _sha_is_valid(restore_receipt.get(key)):
                failures.append(f"{label}: restore_receipt.{key} is not hash-bound")
        if (
            restore_receipt.get("checkpoint_tensor_bytes")
            != contract["combined_checkpoint_payload_bytes"]
            or restore_receipt.get("live_state_logical_tensor_bytes")
            != contract["combined_live_state_bytes"]
            or restore_receipt.get("state_checkpoint_logical_tensor_bytes")
            != contract["live_state_plus_checkpoint_bytes"]
            or restore_receipt.get(
                "state_checkpoint_payload_peak_logical_tensor_bytes"
            )
            != contract["live_state_plus_checkpoint_payload_clone_peak_bytes"]
            or restore_receipt.get("persistent_tensor_bytes") != 0
            or restore_receipt.get("compiled_tensor_bytes_retained") != 0
            or restore_receipt.get("allocator_peak_measured") is not False
        ):
            failures.append(f"{label}: checkpoint restore memory receipt changed")
    restart_process = lifecycle.get("restart_process_generation_id")
    if (
        not isinstance(restart_process, str)
        or not restart_process.strip()
        or restart_process == row_process_generation_id
    ):
        failures.append(f"{label}: checkpoint restart did not use a fresh process")
    _append_parent_watchdog_failures(
        failures,
        {
            "process_generation_id": restart_process,
            "worker_command_sha256": lifecycle.get(
                "restart_worker_command_sha256"
            ),
            "parent_watchdog": lifecycle.get("restart_parent_watchdog"),
            "parent_watchdog_evidence_sha256": lifecycle.get(
                "restart_parent_watchdog_evidence_sha256"
            ),
        },
        label=f"{label} restart",
        contract=contract,
    )


_CONTROL_RESULT_RECEIPT_KEYS = frozenset(
    {
        "provenance",
        "runtime_status",
        "native_runtime_verified",
        "allocator_peak_measured",
        "generation_digest",
    }
)
_CONTROL_UPDATE_RECEIPT_KEYS = frozenset(
    {
        "loss",
        "material_gradient_l2_norm",
        "position_gradient_l2_norm",
        "velocity_gradient_l2_norm",
        "weight_gradient_l2_norm",
        "raw_color_parameter_delta_l2_norm",
        "raw_density_parameter_delta_l2_norm",
        "positions0_parameter_delta_l2_norm",
        "velocities_parameter_delta_l2_norm",
        "weight_coefficients_parameter_delta_l2_norm",
        "parameters_before_digest",
        "parameters_after_digest",
        "gradient_digest",
        "update_authorization_digest",
        "generation_digest",
        "provenance",
        "cpu_optimizer_mutation_count",
        "geometry_mutation_count",
        "stale_provider_store_retirement_count",
        "fresh_selected_track_recompile_count",
        "optimizer_history_tensor_bytes",
        "terminal_control_generation",
    }
)
_CONTROL_PRECOMPILE_RECEIPT_KEYS = frozenset(
    {
        "provider_generation_digest",
        "selected_track_manifest_digest",
        "request_count",
        "track_count",
        "artifact_generation_digests",
        "compile_receipt_generation_digests",
        "compiler_work_receipt_chain_digest",
        "compiler_work_receipt_count",
        "root_complement_witness_count",
        "candidate_source_attempt_count",
        "all_site_witness_check_count",
        "unique_pair_difference_count",
        "store_current_resident_accounted_bytes",
        "store_peak_resident_accounted_bytes",
        "store_maximum_resident_accounted_bytes",
        "generation_digest",
        "provenance",
        "compile_pass_count",
        "requested_frame_sampling_used",
        "retained_observation_count",
        "retained_target_tensor_bytes",
        "retained_frame_tensor_bytes",
        "allocator_peak_measured",
    }
)
_CONTROL_ACCOUNTING_KEYS = frozenset(
    {
        "control_mode",
        "same_continuous_compiled_representation",
        "reverse_mode",
        "selected_frame_count",
        "selected_track_count",
        "per_frame_replay_count",
        "per_frame_replay_wall_time_seconds",
        "step_wall_time_seconds",
        "one_time_continuous_compile_pass_count",
        "one_time_continuous_compile_request_count",
        "one_time_continuous_compile_track_count",
        "per_frame_continuous_recompile_count",
        "fresh_selected_track_recompile_count",
        "compiled_artifact_warm_acquisition_count",
        "compiled_artifact_warm_hit_count",
        "frame_result_generation_digest_chain",
        "frame_readback_receipt_chain_digest",
        "frame_readback_receipt_count",
        "frame_release_fence_call_count",
        "frame_release_fence_provenance",
        "maximum_simultaneously_live_frame_count",
        "maximum_in_flight_frame_target_count",
        "maximum_in_flight_frame_prediction_count",
        "maximum_in_flight_frame_reverse_count",
        "frame_result_capability_retained_after_release_count",
        "frame_target_released_before_next_frame",
        "frame_prediction_released_before_next_frame",
        "frame_reverse_scratch_released_before_next_frame",
        "cpu_optimizer_mutation_count",
        "geometry_mutation_count",
        "combined_optimizer_authorization_count",
        "stale_provider_store_retirement_count",
        "terminal_control_generation_invalidated_after_mutation",
        "compiler_work_receipt_chain_digest",
        "compiler_work_receipt_count",
        "root_complement_witness_count",
        "candidate_source_attempt_count",
        "all_site_witness_check_count",
        "unique_pair_difference_count",
        "eligible_native_block_count",
        "active_native_block_count",
        "native_node_forward_launch_count",
        "native_sample_prepare_count",
        "native_sample_launch_count",
        "native_sample_completion_fence_count",
        "native_full_geometry_vjp_launch_count",
        "native_fused_union_v2_transaction_count",
        "geometry_d2h_completion_fence_count",
        "streamed_sample_count",
        "ordered_word_node_interactions",
        "sample_to_node_linear_interactions",
        "sample_to_node_dense_fallback_interactions",
        "selected_pixel_read_call_count",
        "direct_selected_pixel_observation_count",
        "full_frame_target_materialization_count",
        "camera_ray_slice_work_count",
        "camera_ray_slice_scalar_count",
        "peak_lane_resident_logical_tensor_bytes",
        "peak_active_node_state_tensor_bytes",
        "peak_sample_launch_tensor_bytes",
        "peak_decoded_frame_scratch_upper_bound_bytes",
        "peak_selected_frame_target_tensor_upper_bound_bytes",
        "peak_coordinator_visible_live_tensor_upper_bound_bytes",
        "maximum_geometry_bridge_visible_peak_logical_tensor_bytes",
        "selected_pixel_read_modes",
        "selected_pixel_source_manifest_digests",
        "direct_selected_pixel_target_stream",
        "global_material_bar_logical_tensor_bytes",
        "global_geometry_bar_logical_tensor_bytes",
        "global_bar_and_loss_logical_tensor_bytes",
        "frame_material_bar_logical_tensor_bytes",
        "frame_geometry_bar_logical_tensor_bytes",
        "frame_material_readback_and_loss_logical_tensor_bytes",
        "frame_coordinator_visible_logical_tensor_bytes_upper_bound",
        "frame_geometry_bridge_visible_logical_tensor_bytes_upper_bound",
        "frame_material_bar_included_in_coordinator_bound",
        "frame_geometry_bridge_may_overlap_coordinator",
        "frame_readback_cumulative_tensor_bytes",
        "compiled_program_store_resident_accounted_bytes",
        "compiled_program_store_peak_resident_accounted_bytes",
        "combined_live_state_logical_tensor_bytes",
        "maximum_frame_local_logical_tensor_bytes_upper_bound",
        "expensive_live_logical_and_accounted_peak_upper_bound_bytes",
        "expensive_peak_is_frame_count_invariant",
        "persistent_frame_tensor_bytes",
        "persistent_sample_tensor_bytes",
        "persistent_target_tensor_bytes",
        "persistent_prediction_tensor_bytes",
        "optimizer_history_tensor_bytes",
        "camera_time_scalar_count",
        "camera_time_slice_is_allowed_linear_metadata",
        "full_frame_target_tensor_materialized",
        "full_image_compile_used",
        "scalar_fixed_time_topology_discovery_used",
        "allocator_peak_measured",
        "rss_peak_measured",
        "native_runtime_verified",
        "frame_result_generation_digests_retained",
    }
)


def _append_control_row_failures(
    failures: list[str],
    row: Mapping[str, Any],
    *,
    label: str,
    contract: Mapping[str, Any],
    bindings: Mapping[str, Any],
) -> None:
    """Validate the raw compile-once/framewise-replay ablation receipts."""

    frame_count = row.get("requested_frame_count")
    if not _int(frame_count, minimum=1):
        failures.append(f"{label}: requested frame count is invalid")
        return
    expected_top_keys = {
        "mode",
        "requested_frame_count",
        "repeat_index",
        "status",
        "execution",
        "structure",
        "work",
        "memory",
        "quality",
        "preflight",
        "measurement",
        "lifecycle",
        "evidence_sha256",
    }
    if set(row) != expected_top_keys:
        failures.append(f"{label}: control row keys are noncanonical")
    if row.get("status") != contract["required_control_row_status"]:
        failures.append(f"{label}: control row status must be measured")
    if row.get("lifecycle") is not None:
        failures.append(f"{label}: compiled-framewise control cannot claim lifecycle")

    execution = row.get("execution")
    structure = row.get("structure")
    work = row.get("work")
    memory = row.get("memory")
    measurement = row.get("measurement")
    preflight = row.get("preflight")
    quality = row.get("quality")
    if not all(
        isinstance(value, Mapping)
        for value in (
            execution,
            structure,
            work,
            memory,
            measurement,
            preflight,
            quality,
        )
    ):
        failures.append(f"{label}: control row sections are missing")
        return

    if set(execution) != {
        "adapter_provenance",
        "control_result_receipt",
        "update_receipt",
        "adapter_measurements",
    }:
        failures.append(f"{label}: control execution keys are noncanonical")
    if execution.get("adapter_provenance") != contract[
        "required_control_adapter_provenance"
    ]:
        failures.append(f"{label}: execution.adapter_provenance changed")
    control_result = execution.get("control_result_receipt")
    update = execution.get("update_receipt")
    adapter_measurements = execution.get("adapter_measurements")
    if not all(
        isinstance(value, Mapping)
        for value in (control_result, update, adapter_measurements)
    ):
        failures.append(f"{label}: raw control execution receipts are missing")
        return
    if set(control_result) != _CONTROL_RESULT_RECEIPT_KEYS:
        failures.append(f"{label}: control result receipt keys are noncanonical")
    expected_result = {
        "provenance": contract["required_control_step_provenance"],
        "runtime_status": contract["required_control_runtime_status"],
        "native_runtime_verified": False,
        "allocator_peak_measured": False,
    }
    for key, expected in expected_result.items():
        if control_result.get(key) != expected:
            failures.append(f"{label}: control result receipt.{key} changed")
    if not _sha_is_valid(control_result.get("generation_digest")):
        failures.append(f"{label}: control result receipt is not hash-bound")

    if set(update) != _CONTROL_UPDATE_RECEIPT_KEYS:
        failures.append(f"{label}: control update receipt keys are noncanonical")
    expected_update = {
        "provenance": contract["required_control_update_provenance"],
        "cpu_optimizer_mutation_count": 1,
        "geometry_mutation_count": 1,
        "stale_provider_store_retirement_count": 0,
        "fresh_selected_track_recompile_count": 0,
        "optimizer_history_tensor_bytes": 0,
        "terminal_control_generation": True,
    }
    for key, expected in expected_update.items():
        if update.get(key) != expected:
            failures.append(f"{label}: control update receipt.{key} changed")
    if not _number(update.get("loss"), minimum=0.0):
        failures.append(f"{label}: control update loss is invalid")
    for key in (
        *contract["control_update_gradient_l2_keys"],
        *contract["control_parameter_delta_l2_norm_keys"],
    ):
        value = update.get(key)
        if not _number(value, minimum=0.0) or float(value) <= 0.0:
            failures.append(
                f"{label}: control update receipt.{key} must be finite and nonzero"
            )
    for key in contract["control_update_sha256_keys"]:
        if not _sha_is_valid(update.get(key)):
            failures.append(
                f"{label}: control update receipt.{key} is not hash-bound"
            )
    if update.get("parameters_before_digest") == update.get(
        "parameters_after_digest"
    ):
        failures.append(f"{label}: control update did not change parameters")
    if set(adapter_measurements) != {
        "continuous_precompile_measurement_count",
        "continuous_precompile_wall_time_seconds",
        "control_transaction_measurement_count",
        "control_transaction_wall_time_seconds",
    }:
        failures.append(f"{label}: control adapter measurements are noncanonical")
    for key in (
        "continuous_precompile_wall_time_seconds",
        "control_transaction_wall_time_seconds",
    ):
        if not _number(adapter_measurements.get(key), minimum=0.0) or float(
            adapter_measurements.get(key, 0.0)
        ) <= 0.0:
            failures.append(f"{label}: adapter measurement {key} is invalid")
    if (
        adapter_measurements.get("continuous_precompile_measurement_count") != 1
        or adapter_measurements.get("control_transaction_measurement_count") != 1
    ):
        failures.append(f"{label}: adapter measurement counts changed")

    expected_samples = 512 * frame_count
    expected_structure = {
        "image_height": 384,
        "image_width": 512,
        "world_site_count": 1024,
        "weight_coefficient_count": 2,
        "fixed_track_count": 512,
        "dataset_frame_count": 300,
        "selected_frame_count": frame_count,
        "node_count": 4,
        "compile_certification_mode": contract[
            "required_compile_certification_mode"
        ],
        "maximum_sites_per_track_compile": contract[
            "required_maximum_sites_per_track_compile"
        ],
        "track_manifest_sha256": contract["required_track_manifest_sha256"],
        "camera_program_sha256": contract["required_camera_program_sha256"],
        "target_teacher_sha256": contract["required_target_teacher_sha256"],
        "expected_observation_count": expected_samples,
        "loss_element_count": 3 * expected_samples,
        "loss": contract["required_loss"],
    }
    expected_structure_keys = set(expected_structure) | {
        "target_stream_manifest_sha256",
        "observation_manifest_sha256",
        "compiled_world_sha256",
        "physical_grid_sha256",
        "camera_grid_sha256",
        "spatial_block_manifest_sha256",
        "provider_generation_sha256",
        "factory_generation_sha256",
        "optimizer_policy_sha256",
    }
    if set(structure) != expected_structure_keys:
        failures.append(f"{label}: control structure keys are noncanonical")
    for key, expected in expected_structure.items():
        if structure.get(key) != expected:
            failures.append(f"{label}: structure.{key} changed")
    for key in (
        "target_stream_manifest_sha256",
        "observation_manifest_sha256",
        "compiled_world_sha256",
        "physical_grid_sha256",
        "camera_grid_sha256",
        "spatial_block_manifest_sha256",
        "provider_generation_sha256",
        "factory_generation_sha256",
        "optimizer_policy_sha256",
    ):
        if not _sha_is_valid(structure.get(key)):
            failures.append(f"{label}: structure.{key} is not hash-bound")

    if set(work) != {"precompile_receipt", "accounting"}:
        failures.append(f"{label}: control work keys are noncanonical")
    precompile = work.get("precompile_receipt")
    accounting = work.get("accounting")
    if not isinstance(precompile, Mapping) or not isinstance(accounting, Mapping):
        failures.append(f"{label}: raw precompile/accounting receipts are missing")
        return
    if set(precompile) != _CONTROL_PRECOMPILE_RECEIPT_KEYS:
        failures.append(f"{label}: precompile receipt keys are noncanonical")
    expected_precompile = {
        "provenance": contract["required_control_precompile_provenance"],
        "compile_pass_count": 1,
        "request_count": contract["required_control_precompile_request_count"],
        "track_count": contract["required_fixed_track_count"],
        "compiler_work_receipt_count": contract["required_fixed_track_count"],
        "requested_frame_sampling_used": False,
        "retained_observation_count": 0,
        "retained_target_tensor_bytes": 0,
        "retained_frame_tensor_bytes": 0,
        "allocator_peak_measured": False,
    }
    for key, expected in expected_precompile.items():
        if precompile.get(key) != expected:
            failures.append(f"{label}: precompile receipt.{key} changed")
    for key in contract["control_precompile_sha256_keys"]:
        if not _sha_is_valid(precompile.get(key)):
            failures.append(f"{label}: precompile receipt.{key} is not hash-bound")
    if precompile.get("provider_generation_digest") != structure.get(
        "provider_generation_sha256"
    ):
        failures.append(f"{label}: precompile receipt belongs to another provider")
    for key in contract["control_precompile_sha256_sequence_keys"]:
        values = precompile.get(key)
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or len(values) != contract["required_control_precompile_request_count"]
            or not all(_sha_is_valid(value) for value in values)
        ):
            failures.append(f"{label}: precompile receipt.{key} is invalid")
    for key in (
        "root_complement_witness_count",
        "candidate_source_attempt_count",
        "all_site_witness_check_count",
        "unique_pair_difference_count",
    ):
        if not _int(precompile.get(key), minimum=1):
            failures.append(f"{label}: precompile receipt.{key} must be positive")
    if precompile.get("candidate_source_attempt_count") != precompile.get(
        "all_site_witness_check_count"
    ):
        failures.append(
            f"{label}: candidate-source and all-site witness totals disagree"
        )
    store_current = precompile.get("store_current_resident_accounted_bytes")
    store_peak = precompile.get("store_peak_resident_accounted_bytes")
    store_maximum = precompile.get("store_maximum_resident_accounted_bytes")
    if not (
        _int(store_current, minimum=1)
        and _int(store_peak, minimum=1)
        and _int(store_maximum, minimum=1)
        and store_current <= store_peak <= store_maximum
        and store_maximum
        == contract["maximum_control_compiled_program_store_bytes"]
    ):
        failures.append(f"{label}: bounded precompile store accounting changed")

    if set(accounting) != _CONTROL_ACCOUNTING_KEYS:
        failures.append(f"{label}: compiled-framewise accounting keys are noncanonical")
    exact_accounting = {
        "control_mode": "per_frame_replay_sequential",
        "same_continuous_compiled_representation": True,
        "reverse_mode": "staged_sparse",
        "selected_frame_count": frame_count,
        "selected_track_count": contract["required_fixed_track_count"],
        "per_frame_replay_count": frame_count,
        "one_time_continuous_compile_pass_count": 1,
        "one_time_continuous_compile_request_count": contract[
            "required_control_precompile_request_count"
        ],
        "one_time_continuous_compile_track_count": contract[
            "required_fixed_track_count"
        ],
        "compiled_artifact_warm_acquisition_count": (
            frame_count * contract["required_control_precompile_request_count"]
        ),
        "compiled_artifact_warm_hit_count": (
            frame_count * contract["required_control_precompile_request_count"]
        ),
        "frame_readback_receipt_count": frame_count,
        "frame_release_fence_call_count": frame_count,
        "frame_release_fence_provenance": contract[
            "required_control_frame_release_fence_provenance"
        ],
        "maximum_simultaneously_live_frame_count": 1,
        "maximum_in_flight_frame_target_count": 1,
        "maximum_in_flight_frame_prediction_count": 1,
        "maximum_in_flight_frame_reverse_count": 1,
        "frame_target_released_before_next_frame": True,
        "frame_prediction_released_before_next_frame": True,
        "frame_reverse_scratch_released_before_next_frame": True,
        "cpu_optimizer_mutation_count": 1,
        "geometry_mutation_count": 1,
        "combined_optimizer_authorization_count": 1,
        "terminal_control_generation_invalidated_after_mutation": True,
        "streamed_sample_count": expected_samples,
        "sample_to_node_linear_interactions": 4 * expected_samples,
        "selected_pixel_read_call_count": (
            frame_count * contract["required_control_precompile_request_count"]
        ),
        "direct_selected_pixel_observation_count": expected_samples,
        "camera_ray_slice_work_count": expected_samples,
        "camera_ray_slice_scalar_count": 6 * expected_samples,
        "selected_pixel_read_modes": ["direct_pixels"],
        "direct_selected_pixel_target_stream": True,
        "global_material_bar_logical_tensor_bytes": contract[
            "global_material_bar_bytes"
        ],
        "global_geometry_bar_logical_tensor_bytes": contract[
            "transient_global_geometry_bar_bytes"
        ],
        "global_bar_and_loss_logical_tensor_bytes": (
            contract["global_material_bar_bytes"]
            + contract["transient_global_geometry_bar_bytes"]
            + 4
        ),
        "frame_material_bar_logical_tensor_bytes": contract[
            "global_material_bar_bytes"
        ],
        "frame_geometry_bar_logical_tensor_bytes": contract[
            "transient_global_geometry_bar_bytes"
        ],
        "frame_material_readback_and_loss_logical_tensor_bytes": (
            contract["global_material_bar_bytes"] + 4
        ),
        "frame_material_bar_included_in_coordinator_bound": True,
        "frame_geometry_bridge_may_overlap_coordinator": True,
        "frame_readback_cumulative_tensor_bytes": (
            frame_count * (contract["global_material_bar_bytes"] + 4)
        ),
        "combined_live_state_logical_tensor_bytes": contract[
            "combined_live_state_bytes"
        ],
        "expensive_peak_is_frame_count_invariant": True,
        "camera_time_scalar_count": contract["required_dataset_frame_count"],
        "camera_time_slice_is_allowed_linear_metadata": True,
        "full_frame_target_tensor_materialized": False,
        "full_image_compile_used": False,
        "scalar_fixed_time_topology_discovery_used": False,
        "allocator_peak_measured": False,
        "rss_peak_measured": False,
        "native_runtime_verified": False,
    }
    for key, expected in exact_accounting.items():
        if accounting.get(key) != expected:
            failures.append(f"{label}: control accounting.{key} changed")
    for key in contract["control_zero_accounting_keys"]:
        if accounting.get(key) != 0:
            failures.append(f"{label}: control accounting.{key} must be exactly zero")
    for key in contract["control_accounting_sha256_keys"]:
        if not _sha_is_valid(accounting.get(key)):
            failures.append(f"{label}: control accounting.{key} is not hash-bound")
    for key in contract["control_accounting_frame_sha256_sequence_keys"]:
        values = accounting.get(key)
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or len(values) != frame_count
            or not all(_sha_is_valid(value) for value in values)
        ):
            failures.append(f"{label}: control accounting.{key} is invalid")
    for key in contract["control_accounting_nonempty_sha256_sequence_keys"]:
        values = accounting.get(key)
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or not values
            or len(values) > frame_count
            or not all(_sha_is_valid(value) for value in values)
            or list(values) != sorted(set(values))
        ):
            failures.append(f"{label}: control accounting.{key} is invalid")
    frame_times = accounting.get("per_frame_replay_wall_time_seconds")
    if (
        not isinstance(frame_times, Sequence)
        or isinstance(frame_times, (str, bytes))
        or len(frame_times) != frame_count
        or not all(_number(value, minimum=0.0) and float(value) > 0.0 for value in frame_times)
    ):
        failures.append(f"{label}: per-frame replay timings are invalid")
    elif not math.isclose(
        float(accounting.get("step_wall_time_seconds", -1.0)),
        sum(float(value) for value in frame_times),
        rel_tol=1.0e-9,
        abs_tol=1.0e-12,
    ):
        failures.append(f"{label}: control step timing is not the per-frame sum")
    if _number(accounting.get("step_wall_time_seconds"), minimum=0.0) and (
        float(adapter_measurements.get("control_transaction_wall_time_seconds", 0.0))
        < float(accounting["step_wall_time_seconds"])
    ):
        failures.append(f"{label}: control transaction timing excludes replay work")
    if precompile.get("compiler_work_receipt_chain_digest") != accounting.get(
        "compiler_work_receipt_chain_digest"
    ):
        failures.append(f"{label}: compiler receipt chain changed after precompile")
    for key in contract["control_compile_invariant_work_keys"]:
        if precompile.get(key) != accounting.get(key):
            failures.append(f"{label}: precompile/accounting {key} disagree")
    if (
        accounting.get("compiled_program_store_resident_accounted_bytes")
        != store_current
        or accounting.get("compiled_program_store_peak_resident_accounted_bytes")
        != store_peak
    ):
        failures.append(f"{label}: replay changed compiled-store accounting")
    active_blocks = accounting.get("active_native_block_count")
    eligible_blocks = accounting.get("eligible_native_block_count")
    if not (
        _int(active_blocks, minimum=1)
        and _int(eligible_blocks, minimum=1)
        and active_blocks <= eligible_blocks
        and accounting.get("native_node_forward_launch_count") == active_blocks
        and accounting.get("native_full_geometry_vjp_launch_count") == active_blocks
    ):
        failures.append(f"{label}: staged native block launch accounting changed")
    if not _int(accounting.get("ordered_word_node_interactions"), minimum=1):
        failures.append(f"{label}: ordered-word replay work is missing")
    for key in contract["control_positive_logical_accounting_keys"]:
        if not _int(accounting.get(key), minimum=1):
            failures.append(f"{label}: control accounting.{key} must be positive")
    maximum_frame = accounting.get(
        "maximum_frame_local_logical_tensor_bytes_upper_bound"
    )
    coordinator_visible = accounting.get(
        "frame_coordinator_visible_logical_tensor_bytes_upper_bound"
    )
    geometry_bridge_visible = accounting.get(
        "frame_geometry_bridge_visible_logical_tensor_bytes_upper_bound"
    )
    if (
        coordinator_visible
        != accounting.get("peak_coordinator_visible_live_tensor_upper_bound_bytes")
        or geometry_bridge_visible
        != accounting.get(
            "maximum_geometry_bridge_visible_peak_logical_tensor_bytes"
        )
    ):
        failures.append(f"{label}: frame-local component aliases changed")
    if not _int(coordinator_visible, minimum=contract["global_material_bar_bytes"]):
        failures.append(f"{label}: coordinator bound omits its frame material bar")
    expected_maximum_frame = (
        int(coordinator_visible or 0)
        + int(geometry_bridge_visible or 0)
        + contract["transient_global_geometry_bar_bytes"]
        + contract["global_material_bar_bytes"]
        + 4
    )
    if maximum_frame != expected_maximum_frame:
        failures.append(
            f"{label}: frame-local logical peak omits material/geometry bars"
        )
    expensive_peak = accounting.get(
        "expensive_live_logical_and_accounted_peak_upper_bound_bytes"
    )
    expected_expensive_peak = (
        contract["combined_live_state_bytes"]
        + int(store_current or 0)
        + contract["global_material_bar_bytes"]
        + contract["transient_global_geometry_bar_bytes"]
        + 4
        + int(maximum_frame or 0)
    )
    if expensive_peak != expected_expensive_peak:
        failures.append(f"{label}: expensive live logical peak equation changed")

    expected_preflight_model = {
        "kind": "same-representation-sequential-per-frame-always-launch-v1",
        "requested_frame_count": frame_count,
        "same_representation_and_native_kernels_required": True,
        "sequential_frame_release_required": True,
        "fixed_combined_live_state_bytes": contract["combined_live_state_bytes"],
        "logical_peak_lower_bound_bytes": contract["combined_live_state_bytes"],
        "censorship_permitted": False,
        "working_set_limit_bytes": contract["maximum_mps_working_set_bytes"],
    }
    expected_preflight = {
        "performed": True,
        "policy": contract["required_control_preflight_policy"],
        "model": expected_preflight_model,
        "model_sha256": canonical_sha256(expected_preflight_model),
        "projected_peak_bytes": contract["combined_live_state_bytes"],
        "working_set_limit_bytes": contract["maximum_mps_working_set_bytes"],
        "decision": "launch",
        "censor_reason": None,
    }
    for key, expected in expected_preflight.items():
        if preflight.get(key) != expected:
            failures.append(f"{label}: control preflight.{key} changed")

    expected_memory_keys = {
        "logical_accounting",
        "measured_peak_fields_producer_owned",
        "logical_bounds_are_measured_peaks",
        *contract["required_memory_peak_keys"],
    }
    if set(memory) != expected_memory_keys:
        failures.append(f"{label}: control memory keys are noncanonical")
    logical = memory.get("logical_accounting")
    if not isinstance(logical, Mapping):
        failures.append(f"{label}: control logical accounting is missing")
    else:
        if set(logical) != set(contract["control_logical_accounting_keys"]):
            failures.append(f"{label}: logical accounting keys are noncanonical")
        for key in contract["control_logical_accounting_keys"]:
            if logical.get(key) != accounting.get(key):
                failures.append(
                    f"{label}: logical accounting.{key} differs from raw accounting"
                )
    if (
        memory.get("measured_peak_fields_producer_owned") is not True
        or memory.get("logical_bounds_are_measured_peaks") is not False
    ):
        failures.append(f"{label}: logical and measured memory were conflated")

    expected_measurement = {
        "fresh_process": True,
        "measurement_kind": "fresh-process-mps-and-rss-sampled-high-water-v1",
        "completion_fenced_before_final_measurement": True,
        "allocator_exact_peak_claimed": False,
        "mps_memory_limit_bytes": contract["maximum_mps_working_set_bytes"],
        "process_group_rss_limit_bytes": contract[
            "maximum_worker_process_group_rss_bytes"
        ],
    }
    if set(measurement) != {
        *expected_measurement,
        "mps_memory_sample_count",
        "process_generation_id",
        "bindings",
        "worker_command_sha256",
        "parent_watchdog",
        "parent_watchdog_evidence_sha256",
    }:
        failures.append(f"{label}: control measurement keys are noncanonical")
    for key, expected in expected_measurement.items():
        if measurement.get(key) != expected:
            failures.append(f"{label}: control measurement.{key} changed")
    generation_id = measurement.get("process_generation_id")
    if not isinstance(generation_id, str) or not generation_id.strip():
        failures.append(f"{label}: control process generation is missing")
    row_bindings = measurement.get("bindings")
    if not isinstance(row_bindings, Mapping):
        failures.append(f"{label}: control measurement bindings are missing")
    else:
        for key in ROW_BINDING_SHA_KEYS:
            if row_bindings.get(key) != bindings.get(key):
                failures.append(f"{label}: control measurement binding {key} drifted")
    _append_parent_watchdog_failures(
        failures,
        measurement,
        label=label,
        contract=contract,
    )
    if isinstance(measurement.get("parent_watchdog"), Mapping) and (
        measurement["parent_watchdog"].get(
            "sampled_process_group_rss_high_water_bytes"
        )
        != memory.get("parent_process_group_rss_sampled_peak_bytes")
    ):
        failures.append(f"{label}: control watchdog RSS peak is not bound to row memory")
    if not _int(
        measurement.get("mps_memory_sample_count"),
        minimum=contract["minimum_fresh_process_mps_sample_count"],
    ):
        failures.append(f"{label}: insufficient control MPS high-water samples")

    for key in contract["required_memory_peak_keys"]:
        if not _int(memory.get(key), minimum=0):
            failures.append(f"{label}: measured control memory.{key} is missing")
    for prefix in ("process_rss", "sampled_mps_current", "sampled_mps_driver"):
        baseline = memory.get(f"{prefix}_baseline_bytes")
        peak = memory.get(f"{prefix}_peak_bytes")
        if _int(baseline, minimum=0) and _int(peak, minimum=0) and peak < baseline:
            failures.append(f"{label}: control {prefix} peak precedes its baseline")
    if all(
        _int(memory.get(key), minimum=0)
        for key in (
            "process_rss_baseline_bytes",
            "process_rss_peak_bytes",
            "sampled_mps_current_baseline_bytes",
            "sampled_mps_current_peak_bytes",
            "sampled_mps_driver_baseline_bytes",
            "sampled_mps_driver_peak_bytes",
        )
    ):
        for prefix, maximum_key in (
            ("process_rss", "maximum_process_rss_peak_delta_bytes"),
            ("sampled_mps_current", "maximum_sampled_mps_current_delta_bytes"),
            ("sampled_mps_driver", "maximum_sampled_mps_driver_delta_bytes"),
        ):
            if _memory_delta(memory, prefix) > contract[maximum_key]:
                failures.append(f"{label}: control {prefix} delta exceeds contract")
    if memory.get("sampled_mps_current_peak_bytes", 0) > contract[
        "maximum_mps_working_set_bytes"
    ]:
        failures.append(f"{label}: control MPS tensor peak exceeds the working-set bound")
    if memory.get("sampled_mps_driver_peak_bytes", 0) > contract[
        "maximum_mps_working_set_bytes"
    ]:
        failures.append(f"{label}: control MPS driver peak exceeds the working-set bound")
    if memory.get("parent_process_group_rss_sampled_peak_bytes", 0) > contract[
        "maximum_worker_process_group_rss_bytes"
    ]:
        failures.append(f"{label}: control process-group RSS exceeds the watchdog bound")

    if set(quality) != {
        "finite",
        "loss",
        "post_update_loss_measured",
    }:
        failures.append(f"{label}: measured control quality keys are noncanonical")
    if (
        quality.get("finite") is not True
        or not _number(quality.get("loss"), minimum=0.0)
        or quality.get("loss") != update.get("loss")
        or quality.get("post_update_loss_measured") is not False
    ):
        failures.append(f"{label}: measured control loss evidence is not honest")
    if row.get("evidence_sha256") != row_evidence_sha256(row):
        failures.append(f"{label}: control row evidence digest changed")


def verify_artifact_payload(
    artifact: Mapping[str, Any],
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    config_sha256: str,
    contract_sha256: str,
) -> dict[str, Any]:
    validate_config(config)
    validate_contract(contract)
    failures: list[str] = []
    expected_top = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "benchmark": BENCHMARK,
        "config_id": config["config_id"],
        "backend": config["backend"],
        "execution_scope": config["execution_scope"],
        "status": "measured",
        "evidence_origin": contract["required_evidence_origin"],
        "proxy_or_test_artifact": False,
        "dataset_is_procedural_synthetic": True,
        "native_execution_measured": True,
        "measurement_is_simulated": False,
        "public_quality_evidence": False,
        "claim_scope": "synthetic_systems_memory_trainability_only",
        "producer_execution_implemented": True,
        "all_rows_fresh_process_completed": True,
    }
    for key, expected in expected_top.items():
        if artifact.get(key) != expected:
            failures.append(f"artifact {key} changed")
    bindings = artifact.get("bindings")
    if not isinstance(bindings, Mapping):
        failures.append("artifact bindings are missing")
        bindings = {}
    for key in REQUIRED_BINDING_SHA_KEYS:
        if not _sha_is_valid(bindings.get(key)):
            failures.append(f"artifact binding {key} is invalid")
    if bindings.get("config_sha256") != config_sha256:
        failures.append("artifact config hash does not match checked-in config")
    if bindings.get("contract_sha256") != contract_sha256:
        failures.append("artifact contract hash does not match checked-in contract")
    hardware = artifact.get("hardware")
    if not isinstance(hardware, Mapping):
        failures.append("artifact hardware receipt is missing")
    else:
        if canonical_sha256(hardware) != bindings.get("hardware_fingerprint_sha256"):
            failures.append("artifact hardware receipt digest changed")
        memory_limit = hardware.get("mps_memory_limit")
        if (
            hardware.get("backend") != "mps"
            or hardware.get("device") != "Apple MPS"
            or not isinstance(memory_limit, Mapping)
        ):
            failures.append("artifact hardware is not the bound MPS execution host")
        else:
            requested_fraction = memory_limit.get("requested_fraction")
            effective_fraction = memory_limit.get("effective_fraction")
            recommended_bytes = memory_limit.get("recommended_max_memory_bytes")
            absolute_limit = memory_limit.get("absolute_working_set_limit_bytes")
            effective_limit = memory_limit.get("effective_working_set_limit_bytes")
            if (
                not _number(requested_fraction, minimum=0.0)
                or float(requested_fraction) <= 0.0
                or not _number(effective_fraction, minimum=0.0)
                or float(effective_fraction) <= 0.0
                or float(effective_fraction) > float(requested_fraction)
                or not _int(recommended_bytes, minimum=1)
                or absolute_limit != contract["maximum_mps_working_set_bytes"]
                or not _int(effective_limit, minimum=1)
                or effective_limit > contract["maximum_mps_working_set_bytes"]
            ):
                failures.append("artifact MPS memory-limit receipt changed or was relaxed")
    manifest = artifact.get("source_manifest")
    if not isinstance(manifest, list) or not manifest:
        failures.append("source manifest is missing")
        manifest = []
    else:
        paths: list[str] = []
        for index, record in enumerate(manifest):
            if not isinstance(record, Mapping):
                failures.append(f"source manifest record {index} is not an object")
                continue
            if (
                not isinstance(record.get("path"), str)
                or not _int(record.get("size_bytes"), minimum=0)
                or not _sha_is_valid(record.get("sha256"))
            ):
                failures.append(f"source manifest record {index} is invalid")
                continue
            paths.append(record["path"])
            relative = Path(record["path"])
            candidate = (ROOT / relative).resolve()
            try:
                candidate.relative_to(ROOT)
            except ValueError:
                failures.append(
                    f"source manifest record {index} escapes the repository"
                )
                continue
            if (
                relative.is_absolute()
                or record["path"].startswith("external/")
                or not candidate.is_file()
            ):
                failures.append(
                    f"source manifest record {index} is not a checked-in source file"
                )
                continue
            if (
                candidate.stat().st_size != record["size_bytes"]
                or file_sha256(candidate) != record["sha256"]
            ):
                failures.append(
                    f"source manifest record {index} does not match this checkout"
                )
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            failures.append("source manifest paths must be sorted and unique")
        missing_paths = sorted(REQUIRED_MANIFEST_PATHS - set(paths))
        if missing_paths:
            failures.append("source manifest omits required paths: " + ", ".join(missing_paths))
        if bindings.get("source_manifest_sha256") != source_manifest_sha256(manifest):
            failures.append("source manifest digest changed")

    rows = artifact.get("rows")
    if not isinstance(rows, list):
        failures.append("artifact rows are missing")
        rows = []
    expected_keys = _required_row_keys(contract)
    row_by_key: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    process_generations: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            failures.append(f"row {index} is not an object")
            continue
        key = (
            row.get("mode"),
            row.get("requested_frame_count"),
            row.get("repeat_index"),
        )
        label = f"row {key!r}"
        if key in row_by_key:
            failures.append(f"{label}: duplicate row")
            continue
        row_by_key[key] = row
        if key not in expected_keys:
            failures.append(f"{label}: unexpected row")
        _append_row_failures(
            failures,
            row,
            label=label,
            config=config,
            contract=contract,
            bindings=bindings,
        )
        measurement = row.get("measurement")
        if isinstance(measurement, Mapping) and isinstance(
            measurement.get("process_generation_id"), str
        ):
            process_generations.append(measurement["process_generation_id"])
        lifecycle = row.get("lifecycle")
        if isinstance(lifecycle, Mapping) and isinstance(
            lifecycle.get("restart_process_generation_id"), str
        ):
            process_generations.append(lifecycle["restart_process_generation_id"])
    missing = sorted(expected_keys - set(row_by_key))
    if missing:
        failures.append(f"required rows are missing: {missing!r}")
    control_rows = artifact.get("control_rows")
    if not isinstance(control_rows, list):
        failures.append("artifact control rows are missing")
        control_rows = []
    expected_control_keys = _required_control_row_keys(contract)
    control_by_key: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for index, row in enumerate(control_rows):
        if not isinstance(row, Mapping):
            failures.append(f"control row {index} is not an object")
            continue
        key = (
            row.get("mode"),
            row.get("requested_frame_count"),
            row.get("repeat_index"),
        )
        label = f"control row {key!r}"
        if key in control_by_key:
            failures.append(f"{label}: duplicate row")
            continue
        control_by_key[key] = row
        if key not in expected_control_keys:
            failures.append(f"{label}: unexpected row")
        _append_control_row_failures(
            failures,
            row,
            label=label,
            contract=contract,
            bindings=bindings,
        )
        measurement = row.get("measurement")
        if isinstance(measurement, Mapping) and isinstance(
            measurement.get("process_generation_id"), str
        ):
            process_generations.append(measurement["process_generation_id"])
    missing_controls = sorted(expected_control_keys - set(control_by_key))
    if missing_controls:
        failures.append(f"required control rows are missing: {missing_controls!r}")
    if len(process_generations) != len(set(process_generations)):
        failures.append("fresh-process generation ids are not unique")
    if set(row_by_key) == expected_keys:
        _append_cross_row_failures(
            failures,
            row_by_key,
            artifact=artifact,
            contract=contract,
        )
    if set(control_by_key) == expected_control_keys and row_by_key:
        _append_cross_control_failures(
            failures,
            control_by_key,
            primary_rows=row_by_key,
            artifact=artifact,
            contract=contract,
        )
        digest_keys = (
            "compiled_world_sha256",
            "physical_grid_sha256",
            "camera_grid_sha256",
            "spatial_block_manifest_sha256",
            "optimizer_policy_sha256",
            "track_manifest_sha256",
            "camera_program_sha256",
            "target_teacher_sha256",
        )
        for digest_key in digest_keys:
            values = {
                row["structure"][digest_key]
                for row in (*row_by_key.values(), *control_by_key.values())
            }
            if len(values) != 1:
                failures.append(
                    f"primary/control structure.{digest_key} changed across rows"
                )
    report = {
        "status": "passed" if not failures else "failed",
        "accepted": not failures,
        "benchmark": BENCHMARK,
        "required_row_count": len(expected_keys),
        "observed_row_count": len(rows),
        "required_control_row_count": len(expected_control_keys),
        "observed_control_row_count": len(control_rows),
        "failures": failures,
        "claim_scope": "synthetic_systems_memory_trainability_only",
        "allocator_boundary": (
            "sampled public MPS high water plus hard working-set ceiling; no exact allocator peak claim"
        ),
    }
    return report


def _append_cross_row_failures(
    failures: list[str],
    rows: Mapping[tuple[str, int, int], Mapping[str, Any]],
    *,
    artifact: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    fused = [row for key, row in rows.items() if key[0] == "fused_union_v2"]
    for key in contract["fused_frame_invariant_work_keys"]:
        values = [row.get("work", {}).get(key) for row in rows.values()]
        if any(value is None for value in values) or len(set(values)) != 1:
            failures.append(f"all-row compiler work.{key} changed")
    for key in contract["fused_fixed_schedule_linear_work_keys"]:
        per_frame = set()
        for row in fused:
            count = row.get("work", {}).get(key)
            frame_count = row["requested_frame_count"]
            if not _int(count, minimum=1) or count % frame_count:
                failures.append(f"fused work.{key} is not an integer per-frame multiple")
            else:
                per_frame.add(count // frame_count)
        if len(per_frame) != 1:
            failures.append(f"fused work.{key} is not linear in requested F")
    for key in contract["fused_allowed_streamed_work_keys"]:
        medians = {
            frame_count: statistics.median(
                row.get("work", {}).get(key, -1)
                for row in fused
                if row["requested_frame_count"] == frame_count
            )
            for frame_count in (8, 64, 300)
        }
        if not (medians[8] <= medians[64] <= medians[300]):
            failures.append(f"fused allowed streamed work.{key} is not monotone in F")
    for key in (
        "compiled_world_sha256",
        "physical_grid_sha256",
        "camera_grid_sha256",
        "spatial_block_manifest_sha256",
        "factory_generation_sha256",
        "selected_track_recompile_manifest_sha256",
        "optimizer_policy_sha256",
        "track_manifest_sha256",
        "camera_program_sha256",
        "target_teacher_sha256",
    ):
        values = [row.get("structure", {}).get(key) for row in rows.values()]
        if any(value is None for value in values) or len(set(values)) != 1:
            failures.append(f"structure.{key} changed across rows")
    grouped: dict[int, list[Mapping[str, Any]]] = {8: [], 64: [], 300: []}
    for row in fused:
        grouped[row["requested_frame_count"]].append(row)
    memory_metrics = (
        ("process_rss", contract["maximum_fused_process_rss_growth_bytes"]),
        ("sampled_mps_current", contract["maximum_fused_mps_current_growth_bytes"]),
        ("sampled_mps_driver", contract["maximum_fused_mps_driver_growth_bytes"]),
    )
    for prefix, maximum_growth in memory_metrics:
        medians = {
            frame_count: statistics.median(
                _memory_delta(row["memory"], prefix)
                for row in grouped[frame_count]
            )
            for frame_count in grouped
        }
        if medians[300] - medians[8] > maximum_growth:
            failures.append(f"fused {prefix} median growth exceeds contract")
        denominator = max(1.0, float(medians[8]))
        if medians[300] / denominator > contract["maximum_fused_memory_scale"]:
            failures.append(f"fused {prefix} median scale exceeds contract")

    parity = artifact.get("staged_fused_f8_parity")
    if not isinstance(parity, list):
        failures.append("staged/fused F=8 parity records are missing")
        return
    parity_by_repeat: dict[int, Mapping[str, Any]] = {}
    for record in parity:
        if not isinstance(record, Mapping) or not _int(record.get("repeat_index"), minimum=0):
            failures.append("staged/fused parity record is invalid")
            continue
        parity_by_repeat[record["repeat_index"]] = record
    if set(parity_by_repeat) != set(range(contract["required_repeat_count"])):
        failures.append("staged/fused F=8 parity repeat coverage is incomplete")
        return
    limits = (
        ("loss_absolute_error", "maximum_staged_fused_loss_absolute_error"),
        (
            "material_gradient_relative_l2",
            "maximum_staged_fused_material_gradient_relative_l2",
        ),
        (
            "geometry_gradient_relative_l2",
            "maximum_staged_fused_geometry_gradient_relative_l2",
        ),
        ("parameter_relative_l2", "maximum_staged_fused_parameter_relative_l2"),
    )
    for repeat_index, record in parity_by_repeat.items():
        staged = rows[("staged_sparse", 8, repeat_index)]
        fused_row = rows[("fused_union_v2", 8, repeat_index)]
        if record.get("staged_row_evidence_sha256") != staged["evidence_sha256"]:
            failures.append(f"parity repeat {repeat_index}: staged row digest changed")
        if record.get("fused_row_evidence_sha256") != fused_row["evidence_sha256"]:
            failures.append(f"parity repeat {repeat_index}: fused row digest changed")
        for key, contract_key in limits:
            if not _number(record.get(key), minimum=0.0) or record[key] > contract[contract_key]:
                failures.append(f"parity repeat {repeat_index}: {key} exceeds contract")


def _append_cross_control_failures(
    failures: list[str],
    rows: Mapping[tuple[str, int, int], Mapping[str, Any]],
    *,
    primary_rows: Mapping[tuple[str, int, int], Mapping[str, Any]],
    artifact: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    """Validate the measured O(F)-work/O(1)-expensive-state control sweep."""

    controls = list(rows.values())
    grouped: dict[int, list[Mapping[str, Any]]] = {8: [], 64: [], 300: []}
    for row in controls:
        grouped[int(row["requested_frame_count"])].append(row)
    precompile_identity_fields = (
        "provider_generation_digest",
        "selected_track_manifest_digest",
        "artifact_generation_digests",
        "compile_receipt_generation_digests",
        "compiler_work_receipt_chain_digest",
        "generation_digest",
    )
    for frame_count, frame_rows in grouped.items():
        for key in precompile_identity_fields:
            values = {
                canonical_sha256(row["work"]["precompile_receipt"].get(key))
                for row in frame_rows
            }
            if len(values) != 1:
                failures.append(
                    f"control F={frame_count} precompile receipt.{key} changed across repeats"
                )
    precompile_numeric_invariants = (
        "request_count",
        "track_count",
        "compiler_work_receipt_count",
        "root_complement_witness_count",
        "candidate_source_attempt_count",
        "all_site_witness_check_count",
        "unique_pair_difference_count",
        "store_current_resident_accounted_bytes",
        "store_peak_resident_accounted_bytes",
        "store_maximum_resident_accounted_bytes",
    )
    for key in precompile_numeric_invariants:
        values = {
            canonical_sha256(row["work"]["precompile_receipt"].get(key))
            for row in controls
        }
        if len(values) != 1:
            failures.append(
                f"control precompile numeric receipt.{key} changed across F/repeats"
            )
    for key in contract["control_compile_invariant_work_keys"]:
        values = {row["work"]["accounting"].get(key) for row in controls}
        if len(values) != 1:
            failures.append(f"control compiler accounting.{key} changed across F/repeats")
    for key in contract["control_exact_frame_linear_work_keys"]:
        per_frame: set[int] = set()
        for row in controls:
            frame_count = int(row["requested_frame_count"])
            value = row["work"]["accounting"].get(key)
            if not _int(value, minimum=1) or value % frame_count:
                failures.append(
                    f"control accounting.{key} is not an integer per-frame multiple"
                )
            else:
                per_frame.add(value // frame_count)
        if len(per_frame) != 1:
            failures.append(f"control accounting.{key} is not exactly linear in F")
    for key in contract["control_allowed_frame_dependent_work_keys"]:
        medians = {
            frame_count: statistics.median(
                row["work"]["accounting"].get(key, -1)
                for row in grouped[frame_count]
            )
            for frame_count in grouped
        }
        if any(value < 0 for value in medians.values()) or not (
            medians[8] <= medians[64] <= medians[300]
        ):
            failures.append(
                f"control allowed frame-dependent accounting.{key} is not monotone"
            )
    for key in contract["control_logical_accounting_keys"]:
        values = {
            row["memory"]["logical_accounting"].get(key) for row in controls
        }
        if len(values) != 1:
            failures.append(
                f"control expensive logical accounting.{key} changed across F/repeats"
            )
    for key in (
        "compiled_world_sha256",
        "physical_grid_sha256",
        "camera_grid_sha256",
        "spatial_block_manifest_sha256",
        "factory_generation_sha256",
        "optimizer_policy_sha256",
        "track_manifest_sha256",
        "camera_program_sha256",
        "target_teacher_sha256",
    ):
        values = {row["structure"].get(key) for row in controls}
        if len(values) != 1:
            failures.append(f"control structure.{key} changed across F/repeats")
    for frame_count, frame_rows in grouped.items():
        for key in (
            "target_stream_manifest_sha256",
            "observation_manifest_sha256",
            "provider_generation_sha256",
        ):
            if len({row["structure"].get(key) for row in frame_rows}) != 1:
                failures.append(
                    f"control F={frame_count} structure.{key} changed across repeats"
                )

    memory_metrics = (
        ("process_rss", contract["maximum_control_process_rss_growth_bytes"]),
        (
            "sampled_mps_current",
            contract["maximum_control_mps_current_growth_bytes"],
        ),
        (
            "sampled_mps_driver",
            contract["maximum_control_mps_driver_growth_bytes"],
        ),
    )
    for prefix, maximum_growth in memory_metrics:
        medians = {
            frame_count: statistics.median(
                _memory_delta(row["memory"], prefix)
                for row in grouped[frame_count]
            )
            for frame_count in grouped
        }
        if medians[300] - medians[8] > maximum_growth:
            failures.append(f"control {prefix} median growth exceeds contract")
        denominator = max(1.0, float(medians[8]))
        if medians[300] / denominator > contract["maximum_control_memory_scale"]:
            failures.append(f"control {prefix} median scale exceeds contract")

    parity = artifact.get("fused_compiled_framewise_f8_parity")
    if not isinstance(parity, list):
        failures.append("fused/compiled-framewise F=8 parity records are missing")
        return
    parity_by_repeat: dict[int, Mapping[str, Any]] = {}
    expected_record_keys = {
        "repeat_index",
        "fused_row_evidence_sha256",
        "compiled_framewise_control_row_evidence_sha256",
        "loss_absolute_error",
        "material_gradient_relative_l2",
        "geometry_gradient_relative_l2",
        "parameter_relative_l2",
    }
    for record in parity:
        if not isinstance(record, Mapping) or set(record) != expected_record_keys:
            failures.append("fused/compiled-framewise parity record is noncanonical")
            continue
        repeat_index = record.get("repeat_index")
        if not _int(repeat_index, minimum=0) or repeat_index in parity_by_repeat:
            failures.append("fused/compiled-framewise parity repeat is invalid")
            continue
        parity_by_repeat[repeat_index] = record
    expected_repeats = set(range(contract["required_repeat_count"]))
    if set(parity_by_repeat) != expected_repeats:
        failures.append("fused/compiled-framewise F=8 parity repeat coverage is incomplete")
        return
    limits = (
        (
            "loss_absolute_error",
            "maximum_fused_compiled_framewise_loss_absolute_error",
        ),
        (
            "material_gradient_relative_l2",
            "maximum_fused_compiled_framewise_material_gradient_relative_l2",
        ),
        (
            "geometry_gradient_relative_l2",
            "maximum_fused_compiled_framewise_geometry_gradient_relative_l2",
        ),
        (
            "parameter_relative_l2",
            "maximum_fused_compiled_framewise_parameter_relative_l2",
        ),
    )
    for repeat_index, record in parity_by_repeat.items():
        fused = primary_rows.get(("fused_union_v2", 8, repeat_index))
        control = rows.get(("per_frame_replay_sequential", 8, repeat_index))
        if fused is None or control is None:
            failures.append(
                f"fused/compiled-framewise parity repeat {repeat_index}: row is missing"
            )
            continue
        if record.get("fused_row_evidence_sha256") != fused.get("evidence_sha256"):
            failures.append(
                f"fused/compiled-framewise parity repeat {repeat_index}: fused row digest changed"
            )
        if record.get(
            "compiled_framewise_control_row_evidence_sha256"
        ) != control.get("evidence_sha256"):
            failures.append(
                "fused/compiled-framewise parity repeat "
                f"{repeat_index}: control row digest changed"
            )
        for key, contract_key in limits:
            value = record.get(key)
            if not _number(value, minimum=0.0) or value > contract[contract_key]:
                failures.append(
                    "fused/compiled-framewise parity repeat "
                    f"{repeat_index}: {key} exceeds contract"
                )


def verify_artifact_file(
    artifact_path: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
    contract_path: Path = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    return verify_artifact_payload(
        load_json_object(artifact_path),
        load_json_object(config_path),
        load_json_object(contract_path),
        config_sha256=file_sha256(config_path),
        contract_sha256=file_sha256(contract_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify real-native WorldFoam training-memory ablation evidence."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    report = verify_artifact_file(
        args.artifact,
        config_path=args.config,
        contract_path=args.contract,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["accepted"] else 1)


if __name__ == "__main__":
    main()


__all__ = [
    "ARTIFACT_KIND",
    "BENCHMARK",
    "DEFAULT_CONFIG",
    "DEFAULT_CONTRACT",
    "canonical_sha256",
    "file_sha256",
    "load_json_object",
    "row_evidence_sha256",
    "source_manifest_sha256",
    "validate_config",
    "validate_contract",
    "verify_artifact_file",
    "verify_artifact_payload",
]
