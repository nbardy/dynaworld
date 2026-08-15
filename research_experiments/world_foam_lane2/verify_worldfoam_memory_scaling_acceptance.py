#!/usr/bin/env python3
"""Fail-closed memory/scaling acceptance for the kinetic WorldFoam end state.

The gate deliberately does not infer a measured allocator peak from logical
tensor formulas.  A publishable row must report both:

* deterministic logical categories from the fixed-site coordinator; and
* fresh-process process-RSS high water and sampled public MPS counters;
* an absolute MPS allocation ceiling and parent process-group RSS watchdog;
* selected-pixel target receipts with zero full-frame materialization; and
* resource attestation for the exact three custom Metal kernels.

MPS exposes current tensor/driver allocation counters, not a resettable exact
peak or register/private/spill-byte counter.  This gate preserves that evidence
boundary: sampled maxima are labelled lower bounds, the 2-GiB allocation limit
is the hard upper bound, and kernel-private bytes remain unobservable.

Rows vary only frame count.  World size, active spatial blocks, selected node
rank, image shape, and sample-block dimensions must be identical.  The only
admitted frame-dependent live payload is small identity/time metadata.  The
dense observation set is replayed exhaustively through bounded direct-pixel
reads; any hidden ``F*H*W`` target tensor is rejected by source receipts and
still contributes to the measured process/MPS high water.

Structural-signature equality is scoped to denser requested samples over one
fixed physical interval.  It does not claim that event/chart counts remain
fixed when the represented physical duration itself grows.
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


SCHEMA_VERSION = 3
CONTRACT_SCHEMA_VERSION = 3
BENCHMARK = "worldfoam_fixed_site_material_memory_scaling"
PRODUCER_NAME = "run_worldfoam_memory_scaling_acceptance"
PRODUCER_SCHEMA_VERSION = 3
ACTIVE_MATERIAL_MODEL_FORMULA = (
    "sum_q(16*S_q+32*Q_q*J_q)+16*U+16*max(S_q)+4*n_q+16"
)
ALLOCATOR_PEAK_SCOPE = (
    "fresh_process_baseline_through_cold_compile_dense_step_and_release"
)
MPS_SAMPLED_MEASUREMENT_PROVENANCE = (
    "producer-thread-sampled-high-water-lower-bound-v1"
)
AUTOGRAD_SAVED_TENSOR_MEASUREMENT_PROVENANCE = (
    "producer-saved-tensors-hooks-cumulative-logical-bytes-v1"
)
COMPLETION_FENCE_PROVENANCE = "producer-torch.mps.synchronize-v1"
PARENT_WATCHDOG_RSS_MEASUREMENT_KIND = "parent-ps-sampled-high-water"
SCOPE_LIMIT = "does_not_certify_full_geometry_or_geometry_optimizer"
DEFAULT_CONTRACT = Path(__file__).with_name(
    "worldfoam_memory_scaling_acceptance_v3.json"
)

ROW_FIXED_DIMENSION_KEYS = (
    "image_height",
    "image_width",
    "view_count",
    "dataset_frame_count",
    "world_site_count",
    "artifact_working_set_count",
    "active_native_block_count",
    "maximum_tracks_per_request",
    "maximum_target_observations_per_chunk",
    "maximum_samples_per_launch",
    "maximum_node_count",
)

FRAME_INVARIANT_LOGICAL_KEYS = (
    "persistent_world_geometry_tensor_bytes",
    "persistent_material_state_tensor_bytes",
    "persistent_optimizer_state_tensor_bytes",
    "serialized_checkpoint_payload_bytes",
    "artifact_store_peak_resident_accounted_bytes",
    "active_request_logical_tensor_bytes_upper_bound",
    "step_accumulator_logical_tensor_bytes",
    "peak_target_payload_logical_tensor_bytes",
    "peak_ray_payload_logical_tensor_bytes",
    "peak_sample_launch_logical_tensor_bytes",
    "peak_sample_materialization_logical_tensor_bytes_upper_bound",
    "peak_interpolation_evaluator_scratch_logical_tensor_bytes_upper_bound",
    "peak_public_sample_launch_logical_tensor_bytes",
    "peak_native_preparation_scratch_logical_tensor_bytes",
    "peak_target_decode_bridge_logical_tensor_bytes",
    "peak_source_visible_target_read_logical_tensor_bytes_upper_bound",
)

FRAME_INVARIANT_LIVE_LOGICAL_KEYS = tuple(
    key
    for key in FRAME_INVARIANT_LOGICAL_KEYS
    if key != "serialized_checkpoint_payload_bytes"
)

ZERO_RETENTION_LOGICAL_KEYS = (
    "persistent_frame_tensor_bytes",
    "persistent_sample_tensor_bytes",
    "persistent_target_tensor_bytes",
    "persistent_prediction_tensor_bytes",
    "target_source_resident_tensor_bytes",
)

REQUIRED_WORK_KEYS = (
    "streamed_observation_count",
    "sample_launch_count",
    "sample_node_interaction_count",
    "transferred_target_payload_bytes",
    "native_material_vjp_launch_count",
    "structural_compile_track_count",
    "cold_artifact_compile_count",
    "direct_selected_pixel_observation_count",
    "bounded_region_selected_pixel_observation_count",
    "full_frame_fallback_observation_count",
    "full_frame_target_materialization_count",
    "bounded_region_target_materialization_count",
)

REQUIRED_STRUCTURE_STRING_KEYS = (
    "compiler_provenance",
    "active_material_model_formula",
    "world_generation_digest",
    "camera_generation_digest",
    "physical_interval_digest",
    "tolerance_policy_digest",
    "structural_signature_sha256",
)

REQUIRED_STRUCTURE_INT_KEYS = (
    "event_count",
    "track_chart_row_count",
    "word_entry_count",
    "fallback_count",
    "node_forward_launch_count",
    "node_forward_thread_count",
    "node_forward_interaction_count",
    "material_word_vjp_interaction_count",
    "active_material_exact_model_bytes",
)

REQUIRED_RETENTION_FLAGS = {
    "full_dense_observation_replay": True,
    "sample_and_target_payloads_streamed": True,
    "step_accumulator_retains_frame_axis": False,
    "full_video_target_tensor_retained": False,
    "autograd_graph_retained": False,
    "geometry_trainable": False,
    "fake_native_backend": False,
    "selected_pixel_read_acceptance_capable": True,
    "target_source_decode_budget_enforced_before_allocation": True,
}

REQUIRED_MEASUREMENT_BOOL_KEYS = (
    "fresh_process",
    "completion_fenced_before_measurement",
    "autograd_saved_tensor_hooks_enabled",
    "cold_compile_included",
    "native_runtime_verified",
    "production_coordinator_integrated",
    "mps_sampled_high_water_measured",
    "mps_completion_fenced_before_final_sample",
)

# Metal exposes selected-kernel execution width, maximum threads per
# threadgroup, and static threadgroup-memory length. It does not expose a
# measured register/spill/private-byte peak. Schema v3 therefore requires the
# observable resource attestation and explicitly forbids using it as a private
# scratch measurement.

REQUIRED_MEASUREMENT_BYTE_KEYS = (
    "process_rss_baseline_bytes",
    "process_rss_peak_bytes",
    "mps_current_allocated_baseline_bytes",
    "mps_current_allocated_sampled_maximum_bytes",
    "mps_driver_allocated_baseline_bytes",
    "mps_driver_allocated_sampled_maximum_bytes",
    "autograd_saved_tensor_peak_bytes",
)

REQUIRED_MEASUREMENT_STRING_KEYS = (
    "process_generation_id",
    "measurement_scope",
    "mps_sampled_measurement_provenance",
    "autograd_saved_tensor_measurement_provenance",
    "completion_fence_provenance",
    "source_manifest_sha256",
    "trial_driver_sha256",
    "trial_config_sha256",
    "hardware_fingerprint_sha256",
    "native_extension_sha256",
    "trial_command_sha256",
    "trial_execution_evidence_sha256",
    "mps_memory_limit_sha256",
    "parent_watchdog_evidence_sha256",
    "selected_kernel_resource_attestation_sha256",
)

PRODUCER_INVARIANT_MEASUREMENT_KEYS = (
    "source_manifest_sha256",
    "trial_driver_sha256",
    "trial_config_sha256",
    "hardware_fingerprint_sha256",
    "native_extension_sha256",
    "mps_memory_limit_sha256",
)

PRODUCER_UNIQUE_MEASUREMENT_KEYS = (
    "trial_command_sha256",
    "trial_execution_evidence_sha256",
    "parent_watchdog_evidence_sha256",
)

PRODUCER_BINDING_SHA_KEYS = (
    "source_manifest_sha256",
    "trial_driver_sha256",
    "trial_config_sha256",
    "hardware_fingerprint_sha256",
    "native_extension_sha256",
    "producer_source_sha256",
    "mps_memory_limit_sha256",
)

PRODUCER_BINDING_STRING_KEYS = (
    "trial_driver_path",
    "trial_config_path",
    "native_extension_path",
    "python_executable",
    "hardware_summary",
    "command_protocol",
    "worker_watchdog_rss_measurement_kind",
)

_COORDINATOR_LOGICAL_FIELDS = (
    "persistent_world_geometry_tensor_bytes",
    "serialized_checkpoint_payload_bytes",
    "artifact_store_peak_resident_accounted_bytes",
    "active_request_logical_tensor_bytes_upper_bound",
    "step_accumulator_logical_tensor_bytes",
    "peak_target_payload_logical_tensor_bytes",
    "peak_ray_payload_logical_tensor_bytes",
    "peak_sample_launch_logical_tensor_bytes",
    "peak_sample_materialization_logical_tensor_bytes_upper_bound",
    "peak_interpolation_evaluator_scratch_logical_tensor_bytes_upper_bound",
    "peak_public_sample_launch_logical_tensor_bytes",
    "peak_native_preparation_scratch_logical_tensor_bytes",
    "peak_target_decode_bridge_logical_tensor_bytes",
    "peak_source_visible_target_read_logical_tensor_bytes_upper_bound",
    *ZERO_RETENTION_LOGICAL_KEYS,
    "frame_metadata_logical_bytes",
)
_MATERIAL_STATE_LOGICAL_FIELDS = (
    "persistent_material_state_tensor_bytes",
    "persistent_optimizer_state_tensor_bytes",
)
_COORDINATOR_RETENTION_FIELDS = (
    "full_dense_observation_replay",
    "sample_and_target_payloads_streamed",
    "step_accumulator_retains_frame_axis",
    "full_video_target_tensor_retained",
    "autograd_graph_retained",
    "maximum_simultaneously_decoded_target_frame_count",
    "maximum_in_flight_sample_observation_count",
    "maximum_in_flight_target_observation_count",
    "peak_sample_launch_node_count",
    "selected_pixel_read_mode",
    "selected_pixel_read_acceptance_capable",
    "target_source_decode_budget_enforced_before_allocation",
)
_EXTERNAL_MEASUREMENT_FIELDS = (
    *REQUIRED_MEASUREMENT_BOOL_KEYS,
    *REQUIRED_MEASUREMENT_BYTE_KEYS,
    *REQUIRED_MEASUREMENT_STRING_KEYS,
    "autograd_saved_tensor_count",
    "mps_memory_sample_count",
    "mps_memory_sampling_interval_ms",
    "mps_exact_peak_claimed",
    "mps_memory_limit",
    "parent_watchdog",
    "selected_kernel_resource_attestation",
)

# Executable ownership map: the adapter reads source-owned facts only from the
# sealed coordinator/material reports.  Deployment code supplies only native
# identity and actual measurements; it cannot shadow work or structure keys.
ACCEPTANCE_FIELD_PRODUCERS = {
    **{
        f"logical.{key}": "paper_kinetic_fixed_site_material_step.accounting"
        for key in _COORDINATOR_LOGICAL_FIELDS
    },
    **{
        f"logical.{key}": "paper_kinetic_fixed_site_material_state.accounting"
        for key in _MATERIAL_STATE_LOGICAL_FIELDS
    },
    **{
        f"work.{key}": "paper_kinetic_fixed_site_material_step.accounting"
        for key in REQUIRED_WORK_KEYS
    },
    **{
        f"structure.{key}": "paper_kinetic_fixed_site_material_step.structural_receipt_fold"
        for key in (
            *REQUIRED_STRUCTURE_STRING_KEYS,
            *REQUIRED_STRUCTURE_INT_KEYS,
            "chart_node_ranks",
        )
    },
    **{
        f"retention.{key}": "paper_kinetic_fixed_site_material_step.accounting"
        for key in _COORDINATOR_RETENTION_FIELDS
    },
    "retention.geometry_trainable": (
        "paper_kinetic_fixed_site_material_state.accounting"
    ),
    "retention.fake_native_backend": "fresh_process_native_abi_attestation",
    **{
        f"measurement.{key}": "fresh_process_external_phase_sampler_or_native_abi"
        for key in _EXTERNAL_MEASUREMENT_FIELDS
    },
}


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_payload_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _endpoint_including_frame_indices(
    *,
    dataset_frame_count: int,
    requested_frame_count: int,
) -> tuple[int, ...]:
    if requested_frame_count > dataset_frame_count:
        raise ValueError("requested frame count exceeds the fixed dataset grid")
    if requested_frame_count == 1:
        return (0,)
    result = tuple(
        index * (dataset_frame_count - 1) // (requested_frame_count - 1)
        for index in range(requested_frame_count)
    )
    if len(set(result)) != requested_frame_count:
        raise ValueError("fixed-grid requested frame subset contains duplicates")
    return result


def validate_contract(contract: Mapping[str, Any]) -> None:
    if _integer(contract.get("schema_version")) != CONTRACT_SCHEMA_VERSION:
        raise ValueError("memory-scaling contract schema is missing or stale")
    if contract.get("benchmark") != BENCHMARK:
        raise ValueError("memory-scaling contract benchmark changed")
    if not _nonempty_string(contract.get("contract_id")):
        raise ValueError("memory-scaling contract_id must be nonempty")
    frames = _positive_int_sequence(
        contract.get("required_frame_counts"), name="required_frame_counts"
    )
    if len(frames) < 3 or tuple(sorted(set(frames))) != frames:
        raise ValueError(
            "required_frame_counts must be at least three unique ascending integers"
        )
    minimum_frame_scale = _positive_real(
        contract.get("minimum_frame_scale"), name="minimum_frame_scale"
    )
    if frames[-1] / frames[0] < minimum_frame_scale:
        raise ValueError("required frame matrix is too narrow for the declared scale")
    dataset_frames = _positive_int(
        contract.get("required_dataset_frame_count"),
        "required_dataset_frame_count",
    )
    if frames[-1] > dataset_frames:
        raise ValueError("required frame matrix exceeds the fixed dataset grid")
    if contract.get("required_requested_frame_subset_kind") != (
        "endpoint_including_even_index_v1"
    ):
        raise ValueError("required requested-frame subset rule changed")
    if contract.get("require_fixed_dataset_grid_across_rows") is not True:
        raise ValueError("canonical contract must fix one dataset grid across rows")
    if _positive_int(contract.get("minimum_repeat_count"), "minimum_repeat_count") < 2:
        raise ValueError("minimum_repeat_count must be at least two")
    _positive_int(contract.get("minimum_image_pixel_count"), "minimum_image_pixel_count")
    allowed_backends = contract.get("allowed_backends")
    if (
        not isinstance(allowed_backends, list)
        or not allowed_backends
        or any(not _nonempty_string(value) for value in allowed_backends)
        or len(set(allowed_backends)) != len(allowed_backends)
    ):
        raise ValueError("allowed_backends must contain unique nonempty strings")
    if not _nonempty_string(contract.get("required_execution_mode")):
        raise ValueError("required_execution_mode must be nonempty")
    if contract.get("claim_scope") != (
        "denser_requested_sampling_over_one_fixed_physical_interval_and_fixed_compiled_world"
    ):
        raise ValueError("memory-scaling claim_scope changed or broadened")
    for key in (
        "material_state_bytes_per_site",
        "checkpoint_bytes_per_site",
        "maximum_frame_metadata_base_bytes",
        "maximum_frame_metadata_bytes_per_view_frame",
        "maximum_frame_invariant_logical_live_bytes",
        "maximum_artifact_store_peak_resident_bytes",
        "maximum_source_visible_target_read_peak_bytes",
        "maximum_process_rss_peak_delta_bytes",
        "maximum_sampled_mps_current_delta_bytes",
        "maximum_sampled_mps_driver_delta_bytes",
        "maximum_process_rss_growth_bytes",
        "maximum_sampled_mps_current_growth_bytes",
        "maximum_sampled_mps_driver_growth_bytes",
        "maximum_mps_working_set_bytes",
        "maximum_worker_process_group_rss_bytes",
        "maximum_selected_kernel_count",
        "maximum_selected_kernel_threads_per_threadgroup",
        "maximum_selected_kernel_static_threadgroup_memory_bytes",
        "logical_frame_invariant_absolute_slack_bytes",
    ):
        _nonnegative_int(contract.get(key), key)
    for key in (
        "maximum_sampled_memory_scale",
        "maximum_dense_video_growth_fraction",
        "maximum_worker_timeout_seconds",
        "required_worker_watchdog_poll_interval_seconds",
        "required_mps_memory_sampling_interval_ms",
    ):
        _positive_real(contract.get(key), name=key)
    for key in (
        "maximum_mps_working_set_bytes",
        "maximum_worker_process_group_rss_bytes",
    ):
        _positive_int(contract.get(key), key)
    for key in (
        "require_fresh_process_per_trial",
        "require_cold_compile_measurement",
        "require_completion_fence_before_measurement",
        "require_process_rss_high_water",
        "require_sampled_mps_high_water",
        "require_autograd_saved_tensor_measurement",
        "require_selected_pixel_target_access",
        "require_selected_kernel_resource_attestation",
    ):
        if contract.get(key) is not True:
            raise ValueError(f"canonical contract must set {key}=true")
    for key in (
        "require_exact_allocator_peak_measurement",
        "require_native_private_scratch_measurement",
    ):
        if contract.get(key) is not False:
            raise ValueError(f"canonical observable contract must set {key}=false")


def build_trial_from_fixed_site_accounting(
    *,
    frame_count: int,
    repeat_index: int,
    maximum_node_count: int,
    persistent_world_geometry_tensor_bytes: int,
    step_accounting: Mapping[str, Any],
    material_state_accounting: Mapping[str, Any],
    runtime_measurements: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize actual coordinator/state/measurement reports into one trial.

    This is intentionally a strict adapter, not a source-only proof.  Missing
    runtime measurements stay missing and fail here instead of being filled by
    formulas or policy limits.
    """

    frame_count = _positive_int(frame_count, "frame_count")
    repeat_index = _nonnegative_int(repeat_index, "repeat_index")
    maximum_node_count = _positive_int(maximum_node_count, "maximum_node_count")
    persistent_world_geometry_tensor_bytes = _positive_int(
        persistent_world_geometry_tensor_bytes,
        "persistent_world_geometry_tensor_bytes",
    )
    _require_mapping(step_accounting, "step_accounting")
    _require_mapping(material_state_accounting, "material_state_accounting")
    _require_mapping(runtime_measurements, "runtime_measurements")
    coordinator_world_geometry_bytes = _positive_int(
        step_accounting.get("persistent_world_geometry_tensor_bytes"),
        "coordinator persistent_world_geometry_tensor_bytes",
    )
    if persistent_world_geometry_tensor_bytes != coordinator_world_geometry_bytes:
        raise ValueError(
            "driver world geometry bytes do not match coordinator-owned geometry"
        )
    if _integer(material_state_accounting.get("requested_frame_count")) != frame_count:
        raise ValueError("material state accounting frame count does not match the row")
    dataset_frame_count = _positive_int(
        material_state_accounting.get("dataset_frame_count"),
        "dataset_frame_count",
    )
    requested_frame_subset_kind = material_state_accounting.get(
        "requested_frame_subset_kind"
    )
    if requested_frame_subset_kind != "endpoint_including_even_index_v1":
        raise ValueError("material state accounting requested-frame subset rule changed")
    expected_requested_indices = _endpoint_including_frame_indices(
        dataset_frame_count=dataset_frame_count,
        requested_frame_count=frame_count,
    )
    requested_frame_indices_sha256 = _require_sha256(
        material_state_accounting.get("requested_frame_indices_sha256"),
        "requested_frame_indices_sha256",
    )
    if requested_frame_indices_sha256 != _canonical_payload_sha256(
        expected_requested_indices
    ):
        raise ValueError("material state accounting requested-frame subset changed")
    view_count = _positive_int(
        step_accounting.get("selected_view_count"), "selected_view_count"
    )
    # The coordinator's historical key counts selected camera-frame samples,
    # not unique temporal frames.  A dense V-view row therefore reports V*F.
    selected_camera_frame_count = _positive_int(
        step_accounting.get("selected_frame_count"), "selected_frame_count"
    )
    if selected_camera_frame_count != view_count * frame_count:
        raise ValueError(
            "coordinator selected_frame_count must equal dense view_count*frame_count"
        )
    image_pixel_count = _positive_int(
        step_accounting.get("image_pixel_count"), "image_pixel_count"
    )
    world_site_count = _positive_int(
        step_accounting.get("world_site_count"), "world_site_count"
    )
    exact_observation_count = _positive_int(
        step_accounting.get("exact_observation_count"), "exact_observation_count"
    )
    if exact_observation_count != view_count * frame_count * image_pixel_count:
        raise ValueError("coordinator did not replay every dense observation")
    image_height = _positive_int(
        step_accounting.get("image_height"), "image_height"
    )
    image_width = _positive_int(
        step_accounting.get("image_width"), "image_width"
    )
    if image_height * image_width != image_pixel_count:
        raise ValueError("runtime image shape does not match coordinator pixels")

    target_payload = max(
        _nonnegative_int(
            step_accounting.get("peak_cpu_chunk_target_tensor_bytes"),
            "peak_cpu_chunk_target_tensor_bytes",
        ),
        _nonnegative_int(
            step_accounting.get("peak_device_chunk_target_tensor_bytes"),
            "peak_device_chunk_target_tensor_bytes",
        ),
    )
    if target_payload < 12 or target_payload % 12 != 0:
        raise ValueError("peak target payload must contain whole float32 RGB rows")
    maximum_in_flight_target_observation_count = target_payload // 12
    native_preparation_scratch = _nonnegative_int(
        step_accounting.get("peak_native_prepared_sample_scratch_tensor_bytes"),
        "peak_native_prepared_sample_scratch_tensor_bytes",
    )
    if native_preparation_scratch < 24 or (native_preparation_scratch - 20) % 4:
        raise ValueError("native preparation scratch must encode 4*N_launch+20")
    maximum_in_flight_sample_observation_count = (
        native_preparation_scratch - 20
    ) // 4
    logical = {
        "persistent_world_geometry_tensor_bytes": (
            persistent_world_geometry_tensor_bytes
        ),
        "persistent_material_state_tensor_bytes": _nonnegative_int(
            material_state_accounting.get("total_persistent_tensor_bytes"),
            "total_persistent_tensor_bytes",
        ),
        "persistent_optimizer_state_tensor_bytes": _nonnegative_int(
            material_state_accounting.get("optimizer_history_tensor_bytes"),
            "optimizer_history_tensor_bytes",
        ),
        "serialized_checkpoint_payload_bytes": _nonnegative_int(
            step_accounting.get("material_checkpoint_logical_tensor_bytes"),
            "material_checkpoint_logical_tensor_bytes",
        ),
        "artifact_store_peak_resident_accounted_bytes": _nonnegative_int(
            step_accounting.get("artifact_store_peak_resident_accounted_bytes"),
            "artifact_store_peak_resident_accounted_bytes",
        ),
        "active_request_logical_tensor_bytes_upper_bound": _nonnegative_int(
            step_accounting.get("active_request_logical_tensor_bytes_upper_bound"),
            "active_request_logical_tensor_bytes_upper_bound",
        ),
        "step_accumulator_logical_tensor_bytes": _nonnegative_int(
            step_accounting.get("step_accumulator_logical_tensor_bytes"),
            "step_accumulator_logical_tensor_bytes",
        ),
        "peak_target_payload_logical_tensor_bytes": target_payload,
        "peak_ray_payload_logical_tensor_bytes": _nonnegative_int(
            step_accounting.get("peak_ray_payload_logical_tensor_bytes"),
            "peak_ray_payload_logical_tensor_bytes",
        ),
        "peak_sample_launch_logical_tensor_bytes": _nonnegative_int(
            step_accounting.get("peak_sample_launch_tensor_bytes"),
            "peak_sample_launch_tensor_bytes",
        ),
        "peak_sample_materialization_logical_tensor_bytes_upper_bound": (
            _nonnegative_int(
                step_accounting.get(
                    "peak_sample_materialization_logical_tensor_bytes_upper_bound"
                ),
                "peak_sample_materialization_logical_tensor_bytes_upper_bound",
            )
        ),
        "peak_interpolation_evaluator_scratch_logical_tensor_bytes_upper_bound": (
            _nonnegative_int(
                step_accounting.get(
                    "peak_interpolation_evaluator_scratch_logical_tensor_bytes_upper_bound"
                ),
                "peak_interpolation_evaluator_scratch_logical_tensor_bytes_upper_bound",
            )
        ),
        "peak_public_sample_launch_logical_tensor_bytes": _nonnegative_int(
            step_accounting.get("peak_public_sample_launch_logical_tensor_bytes"),
            "peak_public_sample_launch_logical_tensor_bytes",
        ),
        "peak_native_preparation_scratch_logical_tensor_bytes": (
            native_preparation_scratch
        ),
        "peak_target_decode_bridge_logical_tensor_bytes": _nonnegative_int(
            step_accounting.get("peak_target_decode_bridge_logical_tensor_bytes"),
            "peak_target_decode_bridge_logical_tensor_bytes",
        ),
        "peak_source_visible_target_read_logical_tensor_bytes_upper_bound": (
            _nonnegative_int(
                step_accounting.get(
                    "peak_source_visible_target_read_logical_tensor_bytes_upper_bound"
                ),
                "peak_source_visible_target_read_logical_tensor_bytes_upper_bound",
            )
        ),
        "persistent_frame_tensor_bytes": _nonnegative_int(
            step_accounting.get("persistent_frame_tensor_bytes"),
            "persistent_frame_tensor_bytes",
        ),
        "persistent_sample_tensor_bytes": _nonnegative_int(
            step_accounting.get("persistent_sample_tensor_bytes"),
            "persistent_sample_tensor_bytes",
        ),
        "persistent_target_tensor_bytes": _nonnegative_int(
            step_accounting.get("persistent_target_tensor_bytes"),
            "persistent_target_tensor_bytes",
        ),
        "persistent_prediction_tensor_bytes": _nonnegative_int(
            step_accounting.get("persistent_prediction_tensor_bytes"),
            "persistent_prediction_tensor_bytes",
        ),
        "target_source_resident_tensor_bytes": _nonnegative_int(
            step_accounting.get("target_source_resident_tensor_bytes"),
            "target_source_resident_tensor_bytes",
        ),
        "frame_metadata_logical_bytes": _nonnegative_int(
            step_accounting.get("source_retained_frame_metadata_logical_bytes"),
            "source_retained_frame_metadata_logical_bytes",
        ),
    }
    work = {
        "streamed_observation_count": _positive_int(
            step_accounting.get("streamed_observation_count"),
            "streamed_observation_count",
        ),
        "sample_launch_count": _positive_int(
            step_accounting.get("sample_launch_count"), "sample_launch_count"
        ),
        "sample_node_interaction_count": _positive_int(
            step_accounting.get("sample_node_interaction_count"),
            "sample_node_interaction_count",
        ),
        "transferred_target_payload_bytes": _positive_int(
            step_accounting.get("transferred_target_payload_bytes"),
            "transferred_target_payload_bytes",
        ),
        "native_material_vjp_launch_count": _positive_int(
            step_accounting.get("native_material_vjp_launch_count"),
            "native_material_vjp_launch_count",
        ),
        "structural_compile_track_count": _positive_int(
            step_accounting.get("artifact_store_cold_compiled_track_count"),
            "artifact_store_cold_compiled_track_count",
        ),
        "cold_artifact_compile_count": _positive_int(
            step_accounting.get("cold_artifact_acquisition_count"),
            "cold_artifact_acquisition_count",
        ),
        "direct_selected_pixel_observation_count": _nonnegative_int(
            step_accounting.get("direct_selected_pixel_observation_count"),
            "direct_selected_pixel_observation_count",
        ),
        "bounded_region_selected_pixel_observation_count": _nonnegative_int(
            step_accounting.get("bounded_region_selected_pixel_observation_count"),
            "bounded_region_selected_pixel_observation_count",
        ),
        "full_frame_fallback_observation_count": _nonnegative_int(
            step_accounting.get("full_frame_fallback_observation_count"),
            "full_frame_fallback_observation_count",
        ),
        "full_frame_target_materialization_count": _nonnegative_int(
            step_accounting.get("full_frame_target_materialization_count"),
            "full_frame_target_materialization_count",
        ),
        "bounded_region_target_materialization_count": _nonnegative_int(
            step_accounting.get("bounded_region_target_materialization_count"),
            "bounded_region_target_materialization_count",
        ),
    }
    retention = {
        "full_dense_observation_replay": _boolean(
            step_accounting.get("full_dense_observation_replay"),
            "full_dense_observation_replay",
        ),
        "sample_and_target_payloads_streamed": _boolean(
            step_accounting.get("sample_and_target_payloads_streamed"),
            "sample_and_target_payloads_streamed",
        ),
        "step_accumulator_retains_frame_axis": _boolean(
            step_accounting.get("step_accumulator_retains_frame_axis"),
            "step_accumulator_retains_frame_axis",
        ),
        "full_video_target_tensor_retained": _boolean(
            step_accounting.get("full_video_target_tensor_retained"),
            "full_video_target_tensor_retained",
        ),
        "autograd_graph_retained": _boolean(
            step_accounting.get("autograd_graph_retained"),
            "autograd_graph_retained",
        ),
        "geometry_trainable": _boolean(
            material_state_accounting.get("geometry_trainable"),
            "geometry_trainable",
        ),
        "fake_native_backend": _boolean(
            runtime_measurements.get("fake_native_backend"),
            "fake_native_backend",
        ),
        "maximum_simultaneously_decoded_target_frame_count": _nonnegative_int(
            step_accounting.get(
                "maximum_simultaneously_decoded_target_frame_count"
            ),
            "maximum_simultaneously_decoded_target_frame_count",
        ),
        "maximum_in_flight_sample_observation_count": _positive_int(
            maximum_in_flight_sample_observation_count,
            "derived maximum_in_flight_sample_observation_count",
        ),
        "maximum_in_flight_target_observation_count": _positive_int(
            maximum_in_flight_target_observation_count,
            "derived maximum_in_flight_target_observation_count",
        ),
        "peak_sample_launch_node_count": _positive_int(
            step_accounting.get("peak_sample_launch_node_count"),
            "peak_sample_launch_node_count",
        ),
        "selected_pixel_read_mode": step_accounting.get(
            "selected_pixel_read_mode"
        ),
        "selected_pixel_read_acceptance_capable": _boolean(
            step_accounting.get("selected_pixel_read_acceptance_capable"),
            "selected_pixel_read_acceptance_capable",
        ),
        "target_source_decode_budget_enforced_before_allocation": _boolean(
            step_accounting.get(
                "target_source_decode_budget_enforced_before_allocation"
            ),
            "target_source_decode_budget_enforced_before_allocation",
        ),
    }
    raw_structure = step_accounting
    structure: dict[str, Any] = {
        key: raw_structure.get(key) for key in REQUIRED_STRUCTURE_STRING_KEYS
    }
    structure.update(
        {
            key: _nonnegative_int(raw_structure.get(key), key)
            for key in REQUIRED_STRUCTURE_INT_KEYS
        }
    )
    structure["chart_node_ranks"] = list(
        _positive_int_sequence(
            raw_structure.get("chart_node_ranks"), name="chart_node_ranks"
        )
    )
    raw_mps_sampled = runtime_measurements.get("mps_sampled_memory")
    mps_sampled = (
        raw_mps_sampled if isinstance(raw_mps_sampled, Mapping) else {}
    )
    raw_mps_memory_limit = runtime_measurements.get("mps_memory_limit")
    mps_memory_limit = (
        dict(raw_mps_memory_limit)
        if isinstance(raw_mps_memory_limit, Mapping)
        else None
    )
    raw_parent_watchdog = runtime_measurements.get("parent_watchdog")
    parent_watchdog = (
        dict(raw_parent_watchdog)
        if isinstance(raw_parent_watchdog, Mapping)
        else None
    )
    raw_kernel_attestation = runtime_measurements.get(
        "selected_kernel_resource_attestation"
    )
    kernel_attestation = (
        dict(raw_kernel_attestation)
        if isinstance(raw_kernel_attestation, Mapping)
        else None
    )
    measurement = {
        key: runtime_measurements.get(key)
        for key in (
            *REQUIRED_MEASUREMENT_BOOL_KEYS,
            *REQUIRED_MEASUREMENT_BYTE_KEYS,
            *REQUIRED_MEASUREMENT_STRING_KEYS,
            "autograd_saved_tensor_count",
        )
    }
    measurement.update(
        {
            "measurement_scope": ALLOCATOR_PEAK_SCOPE,
            "mps_sampled_high_water_measured": bool(mps_sampled),
            "mps_completion_fenced_before_final_sample": mps_sampled.get(
                "completion_fenced_before_final_sample"
            ),
            "mps_sampled_measurement_provenance": mps_sampled.get(
                "measurement_kind"
            ),
            "mps_memory_sample_count": mps_sampled.get("sample_count"),
            "mps_memory_sampling_interval_ms": mps_sampled.get(
                "sampling_interval_ms"
            ),
            "mps_exact_peak_claimed": mps_sampled.get("exact_peak_claimed"),
            "mps_current_allocated_baseline_bytes": mps_sampled.get(
                "baseline_current_allocated_bytes"
            ),
            "mps_current_allocated_sampled_maximum_bytes": mps_sampled.get(
                "maximum_current_allocated_bytes"
            ),
            "mps_driver_allocated_baseline_bytes": mps_sampled.get(
                "baseline_driver_allocated_bytes"
            ),
            "mps_driver_allocated_sampled_maximum_bytes": mps_sampled.get(
                "maximum_driver_allocated_bytes"
            ),
            "mps_memory_limit": mps_memory_limit,
            "mps_memory_limit_sha256": (
                None
                if mps_memory_limit is None
                else _canonical_payload_sha256(mps_memory_limit)
            ),
            "parent_watchdog": parent_watchdog,
            "selected_kernel_resource_attestation": kernel_attestation,
            "selected_kernel_resource_attestation_sha256": (
                None
                if kernel_attestation is None
                else _canonical_payload_sha256(kernel_attestation)
            ),
        }
    )
    return {
        "repeat_index": repeat_index,
        "status": "ok",
        "logical": logical,
        "work": work,
        "structure": structure,
        "retention": retention,
        "measurement": measurement,
        "normalization": {
            "frame_count": frame_count,
            "dataset_frame_count": dataset_frame_count,
            "requested_frame_subset_kind": requested_frame_subset_kind,
            "requested_frame_indices_sha256": requested_frame_indices_sha256,
            "image_height": image_height,
            "image_width": image_width,
            "view_count": view_count,
            "world_site_count": world_site_count,
            "artifact_working_set_count": _positive_int(
                step_accounting.get("canonical_artifact_working_set_count"),
                "canonical_artifact_working_set_count",
            ),
            "active_native_block_count": _positive_int(
                step_accounting.get("active_native_block_count"),
                "active_native_block_count",
            ),
            "maximum_tracks_per_request": _positive_int(
                step_accounting.get("maximum_tracks_per_request"),
                "maximum_tracks_per_request",
            ),
            "maximum_target_observations_per_chunk": _positive_int(
                step_accounting.get("maximum_target_observations_per_chunk"),
                "maximum_target_observations_per_chunk",
            ),
            "maximum_samples_per_launch": _positive_int(
                step_accounting.get("maximum_samples_per_launch"),
                "maximum_samples_per_launch",
            ),
            "maximum_node_count": maximum_node_count,
            "exact_observation_count": exact_observation_count,
        },
    }


def build_artifact(
    *,
    backend: str,
    source_tree_sha256: str,
    source_manifest: Sequence[Mapping[str, Any]],
    contract_path: Path,
    producer_binding: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the canonical top-level envelope around measured row records."""

    contract = load_json_object(contract_path)
    validate_contract(contract)
    if backend not in contract["allowed_backends"]:
        raise ValueError(f"backend {backend!r} is not admitted by the contract")
    _require_sha256(source_tree_sha256, "source_tree_sha256")
    _validate_producer_binding(
        producer_binding,
        source_tree_sha256=source_tree_sha256,
    )
    frozen_manifest = [dict(record) for record in source_manifest]
    _validate_source_manifest(
        frozen_manifest,
        expected_sha256=source_tree_sha256,
    )
    if producer_binding.get("source_manifest_file_count") != len(frozen_manifest):
        raise ValueError("producer source_manifest_file_count does not match manifest")
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK,
        "status": "ok",
        "execution_mode": contract["required_execution_mode"],
        "scope_limit": SCOPE_LIMIT,
        "full_geometry_certified": False,
        "backend": backend,
        "source_tree_sha256": source_tree_sha256,
        "source_manifest": frozen_manifest,
        "producer_binding": dict(producer_binding),
        "contract_binding": {
            "contract_id": contract["contract_id"],
            "contract_sha256": file_sha256(contract_path),
        },
        "rows": [dict(row) for row in rows],
    }


def build_row_from_normalized_trials(
    trials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine fresh-process normalized trials without hiding identity drift."""

    if not trials:
        raise ValueError("a memory-scaling row requires at least one trial")
    first_normalization = _require_mapping(
        trials[0].get("normalization"), "trial normalization"
    )
    for index, trial in enumerate(trials):
        normalization = _require_mapping(
            trial.get("normalization"), f"trial {index} normalization"
        )
        if dict(normalization) != dict(first_normalization):
            raise ValueError("normalized trial dimensions changed within one row")
        if _integer(trial.get("repeat_index")) != index:
            raise ValueError("normalized trial repeat_index values must be contiguous")
    row = {
        "frame_count": first_normalization["frame_count"],
        "status": "ok",
        **{
            key: first_normalization[key]
            for key in (
                *ROW_FIXED_DIMENSION_KEYS,
                "requested_frame_subset_kind",
                "requested_frame_indices_sha256",
                "exact_observation_count",
            )
        },
        "trials": [
            {key: value for key, value in trial.items() if key != "normalization"}
            for trial in trials
        ],
    }
    return row


def verify_artifact_payload(
    artifact: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    contract_sha256: str,
) -> dict[str, Any]:
    """Return a machine-readable acceptance report without raising on rows."""

    failures: list[str] = []
    try:
        validate_contract(contract)
        _require_sha256(contract_sha256, "contract_sha256")
    except (TypeError, ValueError) as exc:
        return {"status": "failed", "failures": [f"invalid contract: {exc}"]}

    if _integer(artifact.get("schema_version")) != SCHEMA_VERSION:
        failures.append("artifact schema_version is missing or stale")
    if artifact.get("benchmark") != BENCHMARK:
        failures.append("artifact benchmark is missing or wrong")
    if artifact.get("status") != "ok":
        failures.append(f"artifact status is {artifact.get('status')!r}, expected 'ok'")
    if artifact.get("execution_mode") != contract["required_execution_mode"]:
        failures.append("artifact execution_mode does not match the contract")
    if artifact.get("scope_limit") != SCOPE_LIMIT:
        failures.append("artifact must disclose the fixed-site material-only scope limit")
    if artifact.get("full_geometry_certified") is not False:
        failures.append("material-only memory evidence cannot certify full geometry")
    backend = artifact.get("backend")
    if backend not in contract["allowed_backends"]:
        failures.append(f"artifact backend {backend!r} is not admitted")
    try:
        _require_sha256(artifact.get("source_tree_sha256"), "source_tree_sha256")
    except (TypeError, ValueError) as exc:
        failures.append(str(exc))
    source_manifest = artifact.get("source_manifest")
    if not isinstance(source_manifest, list) or not source_manifest:
        failures.append("artifact source_manifest must be a nonempty list")
        source_manifest = []
    else:
        try:
            _validate_source_manifest(
                source_manifest,
                expected_sha256=str(artifact.get("source_tree_sha256")),
            )
        except (TypeError, ValueError) as exc:
            failures.append(f"invalid source_manifest: {exc}")
    producer_binding = artifact.get("producer_binding")
    if not isinstance(producer_binding, Mapping):
        failures.append("artifact producer_binding is missing")
        producer_binding = {}
    else:
        try:
            _validate_producer_binding(
                producer_binding,
                source_tree_sha256=str(artifact.get("source_tree_sha256")),
            )
        except (TypeError, ValueError) as exc:
            failures.append(f"invalid producer_binding: {exc}")
        if producer_binding.get("source_manifest_file_count") != len(source_manifest):
            failures.append("producer source_manifest_file_count does not match artifact")
        if producer_binding.get("maximum_mps_working_set_bytes") != contract.get(
            "maximum_mps_working_set_bytes"
        ):
            failures.append("producer MPS hard limit does not match the contract")
        for binding_key, contract_key, label in (
            (
                "worker_process_group_rss_limit_bytes",
                "maximum_worker_process_group_rss_bytes",
                "worker process-group sampled-watchdog limit",
            ),
            (
                "worker_timeout_seconds",
                "maximum_worker_timeout_seconds",
                "worker timeout",
            ),
            (
                "worker_watchdog_poll_interval_seconds",
                "required_worker_watchdog_poll_interval_seconds",
                "worker watchdog cadence",
            ),
            (
                "mps_memory_sample_interval_ms",
                "required_mps_memory_sampling_interval_ms",
                "MPS sampling cadence",
            ),
        ):
            if producer_binding.get(binding_key) != contract.get(contract_key):
                failures.append(f"producer {label} does not match the contract")
    binding = artifact.get("contract_binding")
    if not isinstance(binding, Mapping):
        failures.append("artifact contract_binding is missing")
    else:
        if binding.get("contract_id") != contract["contract_id"]:
            failures.append("artifact contract_id does not match")
        if binding.get("contract_sha256") != contract_sha256:
            failures.append("artifact contract_sha256 does not match")

    expected_frames = tuple(contract["required_frame_counts"])
    raw_rows = artifact.get("rows")
    if not isinstance(raw_rows, list):
        failures.append("artifact rows must be a list")
        raw_rows = []
    rows_by_frame: dict[int, dict[str, Any]] = {}
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            failures.append("artifact rows must contain only objects")
            continue
        frame_count = _integer(raw_row.get("frame_count"))
        if frame_count is None or frame_count < 1:
            failures.append(f"row has invalid frame_count {raw_row.get('frame_count')!r}")
            continue
        if frame_count in rows_by_frame:
            failures.append(f"duplicate frame_count {frame_count}")
            continue
        rows_by_frame[frame_count] = raw_row
    found_frames = tuple(sorted(rows_by_frame))
    if found_frames != expected_frames:
        failures.append(
            f"frame counts {found_frames} do not match required {expected_frames}"
        )

    parsed_rows: list[dict[str, Any]] = []
    for frame_count in expected_frames:
        row = rows_by_frame.get(frame_count)
        if row is None:
            continue
        parsed = _validate_row(
            row,
            frame_count,
            contract,
            failures,
            backend=str(backend),
            producer_binding=producer_binding,
        )
        if parsed is not None:
            parsed_rows.append(parsed)
    if len(parsed_rows) == len(expected_frames):
        _validate_cross_frame_scaling(parsed_rows, contract, failures)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK,
        "status": "passed" if not failures else "failed",
        "backend": backend,
        "contract_id": contract["contract_id"],
        "contract_sha256": contract_sha256,
        "scope_limit": SCOPE_LIMIT,
        "full_geometry_certified": False,
        "frame_counts": list(expected_frames),
        "failures": failures,
    }
    if len(parsed_rows) == len(expected_frames):
        report["scaling"] = _scaling_summary(parsed_rows)
    return report


def verify_artifact(path: Path, contract_path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    try:
        artifact = load_json_object(path)
        contract = load_json_object(contract_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "failed", "failures": [f"could not load input: {exc}"]}
    return verify_artifact_payload(
        artifact,
        contract,
        contract_sha256=file_sha256(contract_path),
    )


def _validate_row(
    row: Mapping[str, Any],
    frame_count: int,
    contract: Mapping[str, Any],
    failures: list[str],
    *,
    backend: str,
    producer_binding: Mapping[str, Any],
) -> dict[str, Any] | None:
    prefix = f"{frame_count}f"
    if row.get("status") != "ok":
        failures.append(f"{prefix}: row status is {row.get('status')!r}")
    dimensions: dict[str, int] = {}
    for key in ROW_FIXED_DIMENSION_KEYS:
        value = _integer(row.get(key))
        if value is None or value < 1:
            failures.append(f"{prefix}: {key} must be a positive integer")
        else:
            dimensions[key] = value
    if len(dimensions) != len(ROW_FIXED_DIMENSION_KEYS):
        return None
    dataset_frame_count = dimensions["dataset_frame_count"]
    if dataset_frame_count != int(contract["required_dataset_frame_count"]):
        failures.append(
            f"{prefix}: dataset_frame_count must remain the contract-fixed "
            f"{contract['required_dataset_frame_count']}"
        )
    subset_kind = row.get("requested_frame_subset_kind")
    if subset_kind != contract["required_requested_frame_subset_kind"]:
        failures.append(f"{prefix}: requested-frame subset rule changed")
    if frame_count > dataset_frame_count:
        failures.append(f"{prefix}: requested frames exceed the row dataset grid")
    else:
        expected_indices = _endpoint_including_frame_indices(
            dataset_frame_count=dataset_frame_count,
            requested_frame_count=frame_count,
        )
        expected_indices_sha256 = _canonical_payload_sha256(expected_indices)
        if row.get("requested_frame_indices_sha256") != expected_indices_sha256:
            failures.append(
                f"{prefix}: requested-frame subset does not match the fixed grid"
            )
    image_pixels = dimensions["image_height"] * dimensions["image_width"]
    if image_pixels < int(contract["minimum_image_pixel_count"]):
        failures.append(
            f"{prefix}: image pixel count {image_pixels} is below the contract minimum"
        )
    exact_observations = _integer(row.get("exact_observation_count"))
    expected_observations = frame_count * dimensions["view_count"] * image_pixels
    if exact_observations != expected_observations:
        failures.append(
            f"{prefix}: exact_observation_count must equal dense V*F*H*W "
            f"({expected_observations})"
        )
    expected_artifact_working_set_count = dimensions["view_count"] * math.ceil(
        image_pixels / dimensions["maximum_tracks_per_request"]
    )
    if (
        dimensions["artifact_working_set_count"]
        != expected_artifact_working_set_count
    ):
        failures.append(
            f"{prefix}: artifact_working_set_count must equal "
            f"V*ceil(H*W/maximum_tracks_per_request) "
            f"({expected_artifact_working_set_count})"
        )
    if (
        dimensions["active_native_block_count"]
        < dimensions["artifact_working_set_count"]
    ):
        failures.append(
            f"{prefix}: every nonempty dense artifact requires an active native block"
        )

    raw_trials = row.get("trials")
    if not isinstance(raw_trials, list):
        failures.append(f"{prefix}: trials must be a list")
        return None
    minimum_repeats = int(contract["minimum_repeat_count"])
    if len(raw_trials) < minimum_repeats:
        failures.append(
            f"{prefix}: trial count {len(raw_trials)} is below {minimum_repeats}"
        )
    parsed_trials: list[dict[str, Any]] = []
    repeat_indices: list[int] = []
    for raw_trial in raw_trials:
        if not isinstance(raw_trial, Mapping):
            failures.append(f"{prefix}: trials must contain only objects")
            continue
        parsed = _validate_trial(
            raw_trial,
            frame_count=frame_count,
            expected_observations=expected_observations,
            dimensions=dimensions,
            contract=contract,
            failures=failures,
            backend=backend,
            producer_binding=producer_binding,
        )
        if parsed is not None:
            parsed_trials.append(parsed)
            repeat_indices.append(parsed["repeat_index"])
    if repeat_indices != list(range(len(raw_trials))):
        failures.append(
            f"{prefix}: repeat_index values must be contiguous in trial order"
        )
    if not parsed_trials:
        return None
    return {
        "frame_count": frame_count,
        "dimensions": dimensions,
        "trials": parsed_trials,
    }


def _validate_selected_kernel_resource_attestation(
    raw: Any,
    *,
    contract: Mapping[str, Any],
    label: str,
    failures: list[str],
) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        failures.append(f"{label}: selected-kernel resource attestation is missing")
        return None
    required_top = {
        "kernels",
        "abi_namespace",
        "compiled_operator_name",
        "selected_execution_path",
        "queried_properties",
        "compiled_abi_schema_verified",
        "compiled_source_mtime_gate_passed",
        "optional_full_geometry_vjp_included",
        "kernel_execution_verified_by_this_query",
        "native_private_or_spill_bytes_measured",
    }
    if set(raw) != required_top:
        failures.append(f"{label}: selected-kernel attestation keys are noncanonical")
        return None
    if (
        raw.get("abi_namespace") != "world_foam_lane2_fused_slab_v0"
        or raw.get("compiled_operator_name")
        != "kinetic_memory_light_selected_kernel_resource_attestation"
        or raw.get("selected_execution_path") != "kinetic_material_only"
        or raw.get("compiled_abi_schema_verified") is not True
        or raw.get("compiled_source_mtime_gate_passed") is not True
        or raw.get("optional_full_geometry_vjp_included") is not False
        or raw.get("kernel_execution_verified_by_this_query") is not False
        or raw.get("native_private_or_spill_bytes_measured") is not False
    ):
        failures.append(f"{label}: selected-kernel attestation scope/ABI flags changed")
    queried = raw.get("queried_properties")
    expected_properties = [
        "MetalKernelFunction::getMaxThreadsPerThreadgroup()",
        "MetalKernelFunction::getThreadExecutionWidth()",
        "MetalKernelFunction::getStaticThreadGroupMemoryLength()",
    ]
    normalized_queried = (
        list(queried) if isinstance(queried, (list, tuple)) else None
    )
    if normalized_queried != expected_properties:
        failures.append(f"{label}: selected-kernel queried properties changed")
    kernels = raw.get("kernels")
    if not isinstance(kernels, (list, tuple)):
        failures.append(f"{label}: selected-kernel attestation kernels are missing")
        return None
    expected_kernels = (
        (
            "kinetic_precompiled_length_p0_lie_node_forward_launch_only",
            "wf2_kinetic_precompiled_length_p0_lie_node_forward_tensor",
        ),
        (
            "kinetic_ragged_p0_lie_sample_accumulate_loss_only_launch_only",
            "wf2_kinetic_ragged_p0_lie_sample_mse_vjp_accumulate_only_tensor",
        ),
        (
            "kinetic_precompiled_length_p0_lie_material_node_vjp_accumulate_launch_only",
            "wf2_kinetic_precompiled_length_p0_lie_material_node_vjp_tensor",
        ),
    )
    if len(kernels) != len(expected_kernels) or len(kernels) > int(
        contract["maximum_selected_kernel_count"]
    ):
        failures.append(f"{label}: selected material path must attest exactly three kernels")
        return None
    required_kernel = {
        "operator_name",
        "metal_function_name",
        "max_threads_per_threadgroup",
        "thread_execution_width",
        "static_threadgroup_memory_length_bytes",
        "queried_from_compiled_metal_kernel_function",
        "static_threadgroup_memory_length_observable",
        "register_bytes_per_thread",
        "register_bytes_per_thread_observable",
        "private_memory_bytes_per_thread",
        "private_memory_bytes_per_thread_observable",
        "compiler_spill_bytes",
        "compiler_spill_bytes_observable",
    }
    normalized_kernels: list[dict[str, Any]] = []
    for index, (
        (expected_operator, expected_metal_function),
        raw_kernel,
    ) in enumerate(zip(expected_kernels, kernels, strict=True)):
        if not isinstance(raw_kernel, Mapping) or set(raw_kernel) != required_kernel:
            failures.append(f"{label}: selected kernel {index} has noncanonical fields")
            return None
        maximum = _integer(raw_kernel.get("max_threads_per_threadgroup"))
        width = _integer(raw_kernel.get("thread_execution_width"))
        static_bytes = _integer(
            raw_kernel.get("static_threadgroup_memory_length_bytes")
        )
        if (
            raw_kernel.get("operator_name") != expected_operator
            or raw_kernel.get("metal_function_name") != expected_metal_function
            or maximum is None
            or not 1
            <= maximum
            <= int(contract["maximum_selected_kernel_threads_per_threadgroup"])
            or width is None
            or not 1 <= width <= maximum
            or static_bytes is None
            or not 0
            <= static_bytes
            <= int(
                contract["maximum_selected_kernel_static_threadgroup_memory_bytes"]
            )
            or raw_kernel.get("queried_from_compiled_metal_kernel_function") is not True
            or raw_kernel.get("static_threadgroup_memory_length_observable") is not True
            or raw_kernel.get("register_bytes_per_thread") is not None
            or raw_kernel.get("register_bytes_per_thread_observable") is not False
            or raw_kernel.get("private_memory_bytes_per_thread") is not None
            or raw_kernel.get("private_memory_bytes_per_thread_observable") is not False
            or raw_kernel.get("compiler_spill_bytes") is not None
            or raw_kernel.get("compiler_spill_bytes_observable") is not False
        ):
            failures.append(f"{label}: selected kernel {index} resource record is invalid")
        normalized_kernels.append(dict(raw_kernel))
    return {
        "kernels": normalized_kernels,
        "abi_namespace": raw.get("abi_namespace"),
        "compiled_operator_name": raw.get("compiled_operator_name"),
        "selected_execution_path": raw.get("selected_execution_path"),
        "queried_properties": expected_properties,
        "compiled_abi_schema_verified": raw.get("compiled_abi_schema_verified"),
        "compiled_source_mtime_gate_passed": raw.get(
            "compiled_source_mtime_gate_passed"
        ),
        "optional_full_geometry_vjp_included": raw.get(
            "optional_full_geometry_vjp_included"
        ),
        "kernel_execution_verified_by_this_query": raw.get(
            "kernel_execution_verified_by_this_query"
        ),
        "native_private_or_spill_bytes_measured": raw.get(
            "native_private_or_spill_bytes_measured"
        ),
    }


def _validate_mps_memory_limit_receipt(
    raw: Any,
    *,
    contract: Mapping[str, Any],
    producer_binding: Mapping[str, Any],
    label: str,
    failures: list[str],
) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        failures.append(f"{label}: MPS memory-limit receipt is missing")
        return None
    required = {
        "requested_fraction",
        "effective_fraction",
        "recommended_max_memory_bytes",
        "absolute_working_set_limit_bytes",
        "effective_working_set_limit_bytes",
    }
    if set(raw) != required:
        failures.append(f"{label}: MPS memory-limit receipt keys are noncanonical")
        return None
    try:
        requested_fraction = _positive_real(
            raw.get("requested_fraction"), name="requested_fraction"
        )
        effective_fraction = _positive_real(
            raw.get("effective_fraction"), name="effective_fraction"
        )
        recommended_bytes = _positive_int(
            raw.get("recommended_max_memory_bytes"),
            "recommended_max_memory_bytes",
        )
        absolute_limit = _positive_int(
            raw.get("absolute_working_set_limit_bytes"),
            "absolute_working_set_limit_bytes",
        )
        effective_limit = _positive_int(
            raw.get("effective_working_set_limit_bytes"),
            "effective_working_set_limit_bytes",
        )
    except (TypeError, ValueError) as exc:
        failures.append(f"{label}: invalid MPS memory-limit receipt: {exc}")
        return None
    expected_fraction = min(
        requested_fraction,
        float(absolute_limit) / float(recommended_bytes),
    )
    expected_limit = min(
        absolute_limit,
        int(requested_fraction * recommended_bytes),
    )
    if (
        requested_fraction > 0.5
        or not math.isclose(
            effective_fraction,
            expected_fraction,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or effective_limit != expected_limit
        or absolute_limit != int(contract["maximum_mps_working_set_bytes"])
        or not isinstance(producer_binding.get("mps_memory_limit"), Mapping)
        or dict(raw) != dict(producer_binding["mps_memory_limit"])
    ):
        failures.append(f"{label}: MPS memory-limit receipt changed or was relaxed")
    return dict(raw)


def _validate_parent_watchdog_receipt(
    raw: Any,
    *,
    contract: Mapping[str, Any],
    producer_binding: Mapping[str, Any],
    label: str,
    failures: list[str],
) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        failures.append(f"{label}: parent watchdog receipt is missing")
        return None
    required = {
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
    if set(raw) != required:
        failures.append(f"{label}: parent watchdog receipt keys are noncanonical")
        return None
    try:
        returncode = _nonnegative_int(raw.get("returncode"), "returncode")
        elapsed = _positive_real(raw.get("elapsed_seconds"), name="elapsed_seconds")
        interval = _positive_real(
            raw.get("rss_sampling_interval_seconds"),
            name="rss_sampling_interval_seconds",
        )
        sampled_rss = _positive_int(
            raw.get("sampled_process_group_rss_high_water_bytes"),
            "sampled_process_group_rss_high_water_bytes",
        )
        sample_count = _positive_int(raw.get("sample_count"), "sample_count")
        timeout = _positive_real(
            raw.get("worker_timeout_seconds"), name="worker_timeout_seconds"
        )
        rss_limit = _positive_int(
            raw.get("worker_process_group_rss_limit_bytes"),
            "worker_process_group_rss_limit_bytes",
        )
    except (TypeError, ValueError) as exc:
        failures.append(f"{label}: invalid parent watchdog receipt: {exc}")
        return None
    if (
        returncode != 0
        or elapsed > timeout
        or sampled_rss > rss_limit
        or sample_count < 2
        or (sample_count - 1) * interval > elapsed + 1.0e-9
        or raw.get("rss_measurement_kind")
        != producer_binding.get("worker_watchdog_rss_measurement_kind")
        or interval != float(contract["required_worker_watchdog_poll_interval_seconds"])
        or interval
        != producer_binding.get("worker_watchdog_poll_interval_seconds")
        or timeout != float(contract["maximum_worker_timeout_seconds"])
        or timeout != producer_binding.get("worker_timeout_seconds")
        or rss_limit != int(contract["maximum_worker_process_group_rss_bytes"])
        or rss_limit != producer_binding.get("worker_process_group_rss_limit_bytes")
        or raw.get("watchdog_completed") is not True
        or raw.get("process_group_empty_after_exit") is not True
        or raw.get("worker_terminated_by_watchdog") is not False
    ):
        failures.append(f"{label}: parent watchdog did not complete under bound limits")
    return dict(raw)


def _validate_trial(
    trial: Mapping[str, Any],
    *,
    frame_count: int,
    expected_observations: int,
    dimensions: Mapping[str, int],
    contract: Mapping[str, Any],
    failures: list[str],
    backend: str,
    producer_binding: Mapping[str, Any],
) -> dict[str, Any] | None:
    repeat_index = _integer(trial.get("repeat_index"))
    label = f"{frame_count}f trial {trial.get('repeat_index')!r}"
    if repeat_index is None or repeat_index < 0:
        failures.append(f"{label}: repeat_index must be nonnegative integer")
        return None
    if trial.get("status") != "ok":
        failures.append(f"{label}: status is {trial.get('status')!r}")

    logical = trial.get("logical")
    work = trial.get("work")
    structure = trial.get("structure")
    retention = trial.get("retention")
    measurement = trial.get("measurement")
    if not all(
        isinstance(value, Mapping)
        for value in (logical, work, structure, retention, measurement)
    ):
        failures.append(
            f"{label}: logical/work/structure/retention/measurement objects are required"
        )
        return None
    assert isinstance(logical, Mapping)
    assert isinstance(work, Mapping)
    assert isinstance(structure, Mapping)
    assert isinstance(retention, Mapping)
    assert isinstance(measurement, Mapping)

    parsed_logical: dict[str, int] = {}
    for key in (
        *FRAME_INVARIANT_LOGICAL_KEYS,
        *ZERO_RETENTION_LOGICAL_KEYS,
        "frame_metadata_logical_bytes",
    ):
        value = _integer(logical.get(key))
        if value is None or value < 0:
            failures.append(f"{label}: logical.{key} must be a nonnegative integer")
        else:
            parsed_logical[key] = value
    if len(parsed_logical) != (
        len(FRAME_INVARIANT_LOGICAL_KEYS)
        + len(ZERO_RETENTION_LOGICAL_KEYS)
        + 1
    ):
        return None
    for key in ZERO_RETENTION_LOGICAL_KEYS:
        if parsed_logical[key] != 0:
            failures.append(f"{label}: logical.{key} must be zero")
    expected_material = (
        dimensions["world_site_count"] * int(contract["material_state_bytes_per_site"])
    )
    if parsed_logical["persistent_material_state_tensor_bytes"] != expected_material:
        failures.append(
            f"{label}: persistent material state must be {expected_material} bytes"
        )
    expected_checkpoint = (
        dimensions["world_site_count"] * int(contract["checkpoint_bytes_per_site"])
    )
    if parsed_logical["serialized_checkpoint_payload_bytes"] != expected_checkpoint:
        failures.append(
            f"{label}: checkpoint payload must be {expected_checkpoint} bytes"
        )
    if parsed_logical["persistent_optimizer_state_tensor_bytes"] != 0:
        failures.append(f"{label}: fixed-site manual SGD optimizer state must be zero")
    if parsed_logical["persistent_world_geometry_tensor_bytes"] <= 0:
        failures.append(f"{label}: persistent world geometry bytes must be positive")
    if (
        parsed_logical["artifact_store_peak_resident_accounted_bytes"]
        > int(contract["maximum_artifact_store_peak_resident_bytes"])
    ):
        failures.append(f"{label}: artifact store peak exceeds the contract")
    logical_live = sum(
        parsed_logical[key] for key in FRAME_INVARIANT_LIVE_LOGICAL_KEYS
    )
    if logical_live > int(contract["maximum_frame_invariant_logical_live_bytes"]):
        failures.append(f"{label}: frame-invariant logical live bound exceeds the contract")
    metadata_limit = int(contract["maximum_frame_metadata_base_bytes"]) + (
        int(contract["maximum_frame_metadata_bytes_per_view_frame"])
        * dimensions["view_count"]
        * frame_count
    )
    if parsed_logical["frame_metadata_logical_bytes"] > metadata_limit:
        failures.append(
            f"{label}: frame metadata exceeds the admitted cheap O(F) bound"
        )
    source_visible_target_read_peak = parsed_logical[
        "peak_source_visible_target_read_logical_tensor_bytes_upper_bound"
    ]
    if source_visible_target_read_peak > int(
        contract["maximum_source_visible_target_read_peak_bytes"]
    ):
        failures.append(f"{label}: selected target-read peak exceeds the contract")

    parsed_work: dict[str, int] = {}
    zero_allowed_work = {
        "direct_selected_pixel_observation_count",
        "bounded_region_selected_pixel_observation_count",
        "full_frame_fallback_observation_count",
        "full_frame_target_materialization_count",
        "bounded_region_target_materialization_count",
    }
    for key in REQUIRED_WORK_KEYS:
        value = _integer(work.get(key))
        minimum = 0 if key in zero_allowed_work else 1
        if value is None or value < minimum:
            failures.append(
                f"{label}: work.{key} must be an integer >= {minimum}"
            )
        else:
            parsed_work[key] = value
    if len(parsed_work) != len(REQUIRED_WORK_KEYS):
        return None
    if parsed_work["streamed_observation_count"] != expected_observations:
        failures.append(f"{label}: streamed observations do not cover dense V*F*H*W")
    if parsed_work["transferred_target_payload_bytes"] != expected_observations * 12:
        failures.append(
            f"{label}: transferred target bytes must equal dense RGB float32 payload"
        )
    selected_observations = (
        parsed_work["direct_selected_pixel_observation_count"]
        + parsed_work["bounded_region_selected_pixel_observation_count"]
    )
    if selected_observations != expected_observations:
        failures.append(
            f"{label}: selected-pixel observations do not cover dense replay exactly"
        )
    if parsed_work["full_frame_fallback_observation_count"] != 0:
        failures.append(f"{label}: full-frame target fallback is not acceptance-capable")
    if parsed_work["full_frame_target_materialization_count"] != 0:
        failures.append(f"{label}: full-frame target materialization must be zero")

    minimum_sample_launch_count = math.ceil(
        expected_observations / dimensions["maximum_samples_per_launch"]
    )
    if not (
        minimum_sample_launch_count
        <= parsed_work["sample_launch_count"]
        <= expected_observations
    ):
        failures.append(
            f"{label}: sample launches cannot cover dense observations under the "
            "declared maximum_samples_per_launch"
        )
    if not (
        expected_observations
        <= parsed_work["sample_node_interaction_count"]
        <= expected_observations * dimensions["maximum_node_count"]
    ):
        failures.append(
            f"{label}: sample-node interactions must cover every observation with "
            "a rank in [1,J_max]"
        )
    expected_compiled_tracks = (
        dimensions["view_count"]
        * dimensions["image_height"]
        * dimensions["image_width"]
    )
    if parsed_work["structural_compile_track_count"] != expected_compiled_tracks:
        failures.append(
            f"{label}: a fresh cold trial must compile every dense view/pixel track"
        )
    if (
        parsed_work["cold_artifact_compile_count"]
        != dimensions["artifact_working_set_count"]
    ):
        failures.append(
            f"{label}: a fresh cold trial must compile the complete artifact working set"
        )

    parsed_structure: dict[str, Any] = {}
    for key in REQUIRED_STRUCTURE_STRING_KEYS:
        value = structure.get(key)
        if not _nonempty_string(value):
            failures.append(f"{label}: structure.{key} must be nonempty")
        else:
            parsed_structure[key] = value
    for key in (
        "world_generation_digest",
        "camera_generation_digest",
        "physical_interval_digest",
        "tolerance_policy_digest",
        "structural_signature_sha256",
    ):
        try:
            _require_sha256(structure.get(key), f"structure.{key}")
        except (TypeError, ValueError) as exc:
            failures.append(f"{label}: {exc}")
    if structure.get("active_material_model_formula") != ACTIVE_MATERIAL_MODEL_FORMULA:
        failures.append(f"{label}: active material memory formula changed")
    for key in REQUIRED_STRUCTURE_INT_KEYS:
        value = _integer(structure.get(key))
        minimum = 0 if key == "fallback_count" else 1
        if value is None or value < minimum:
            failures.append(
                f"{label}: structure.{key} must be an integer >= {minimum}"
            )
        else:
            parsed_structure[key] = value
    ranks = structure.get("chart_node_ranks")
    if (
        not isinstance(ranks, list)
        or not ranks
        or any(_integer(value) is None or int(value) < 1 for value in ranks)
    ):
        failures.append(
            f"{label}: structure.chart_node_ranks must contain positive integers"
        )
    else:
        parsed_structure["chart_node_ranks"] = tuple(int(value) for value in ranks)
    if len(parsed_structure) != (
        len(REQUIRED_STRUCTURE_STRING_KEYS)
        + len(REQUIRED_STRUCTURE_INT_KEYS)
        + 1
    ):
        return None

    for key, expected in REQUIRED_RETENTION_FLAGS.items():
        if retention.get(key) is not expected:
            failures.append(f"{label}: retention.{key} must be {expected}")
    selected_pixel_mode = retention.get("selected_pixel_read_mode")
    if selected_pixel_mode not in {"direct_pixels", "certified_bounded_region"}:
        failures.append(
            f"{label}: target access must be direct pixels or a certified bounded region"
        )
    if retention.get("maximum_simultaneously_decoded_target_frame_count") != 0:
        failures.append(f"{label}: memory-light target access cannot decode a full frame")
    if selected_pixel_mode == "direct_pixels":
        if parsed_work["direct_selected_pixel_observation_count"] != expected_observations:
            failures.append(f"{label}: direct-pixel mode did not serve every observation")
        if (
            parsed_work["bounded_region_selected_pixel_observation_count"] != 0
            or parsed_work["bounded_region_target_materialization_count"] != 0
        ):
            failures.append(f"{label}: direct-pixel mode reported bounded-region work")
    elif selected_pixel_mode == "certified_bounded_region":
        if (
            parsed_work["bounded_region_selected_pixel_observation_count"]
            != expected_observations
            or parsed_work["direct_selected_pixel_observation_count"] != 0
            or parsed_work["bounded_region_target_materialization_count"] < 1
        ):
            failures.append(f"{label}: bounded-region mode reported inconsistent work")
    maximum_in_flight_sample = _integer(
        retention.get("maximum_in_flight_sample_observation_count")
    )
    if (
        maximum_in_flight_sample is None
        or not 1
        <= maximum_in_flight_sample
        <= dimensions["maximum_samples_per_launch"]
    ):
        failures.append(
            f"{label}: in-flight native sample observations exceed K_launch"
        )
    maximum_in_flight_target = _integer(
        retention.get("maximum_in_flight_target_observation_count")
    )
    if (
        maximum_in_flight_target is None
        or not 1
        <= maximum_in_flight_target
        <= dimensions["maximum_target_observations_per_chunk"]
    ):
        failures.append(
            f"{label}: in-flight target observations exceed K_target"
        )
    if (
        maximum_in_flight_sample is not None
        and maximum_in_flight_target is not None
        and maximum_in_flight_sample > maximum_in_flight_target
    ):
        failures.append(f"{label}: a native launch cannot exceed its target chunk")
    if maximum_in_flight_sample is not None and maximum_in_flight_target is not None:
        peak_sample_node_count = _integer(
            retention.get("peak_sample_launch_node_count")
        )
        if (
            peak_sample_node_count is None
            or not 1
            <= peak_sample_node_count
            <= dimensions["maximum_node_count"]
        ):
            failures.append(
                f"{label}: peak sample launch node count exceeds J_max"
            )
            peak_sample_node_count = dimensions["maximum_node_count"]
        sample_launch_unit_bytes = 4 * peak_sample_node_count + 24
        peak_sample_launch_bytes = parsed_logical[
            "peak_sample_launch_logical_tensor_bytes"
        ]
        if (
            peak_sample_launch_bytes < sample_launch_unit_bytes
            or peak_sample_launch_bytes % sample_launch_unit_bytes != 0
            or peak_sample_launch_bytes // sample_launch_unit_bytes
            > maximum_in_flight_sample
        ):
            failures.append(
                f"{label}: sample launch payload must encode N_sample*(4*J_peak+24) "
                "with N_sample<=K_launch"
            )
        expected_native_prepare = 4 * maximum_in_flight_sample + 20
        expected_target_payload = 12 * maximum_in_flight_target
        target_resident_bytes = expected_target_payload * (
            2 if backend != "cpu" else 1
        )
        if (
            parsed_logical[
                "peak_native_preparation_scratch_logical_tensor_bytes"
            ]
            != expected_native_prepare
        ):
            failures.append(f"{label}: native preparation scratch must equal 4*N+20")
        if (
            parsed_logical["peak_target_payload_logical_tensor_bytes"]
            != expected_target_payload
        ):
            failures.append(f"{label}: material target payload must equal 12*N")
        if source_visible_target_read_peak < expected_target_payload:
            failures.append(
                f"{label}: source-visible target-read peak omits the returned RGB chunk"
            )
        if (
            parsed_logical["peak_public_sample_launch_logical_tensor_bytes"]
            < max(
                target_resident_bytes,
                peak_sample_launch_bytes,
                expected_native_prepare,
            )
            or parsed_logical["peak_public_sample_launch_logical_tensor_bytes"]
            > (
                target_resident_bytes
                + peak_sample_launch_bytes
                + expected_native_prepare
            )
        ):
            failures.append(
                f"{label}: public sample-launch peak leaves the independently "
                "bounded target/sample/native envelope"
            )
        materialization_upper_bound = parsed_logical[
            "peak_sample_materialization_logical_tensor_bytes_upper_bound"
        ]
        interpolation_upper_bound = parsed_logical[
            "peak_interpolation_evaluator_scratch_logical_tensor_bytes_upper_bound"
        ]
        if interpolation_upper_bound < 1:
            failures.append(
                f"{label}: interpolation evaluator logical bound must be positive"
            )
        if materialization_upper_bound < interpolation_upper_bound:
            failures.append(
                f"{label}: materialization bound does not cover interpolation scratch"
            )
    if max(parsed_structure["chart_node_ranks"]) != dimensions["maximum_node_count"]:
        failures.append(f"{label}: maximum_node_count does not match chart ranks")
    if (
        parsed_structure["active_material_exact_model_bytes"]
        > parsed_logical["active_request_logical_tensor_bytes_upper_bound"]
    ):
        failures.append(
            f"{label}: active request bound does not cover the exact material model"
        )
    if (
        parsed_structure["node_forward_launch_count"]
        != dimensions["active_native_block_count"]
    ):
        failures.append(f"{label}: node forward count must equal active blocks")
    if (
        parsed_structure["node_forward_thread_count"]
        > parsed_structure["node_forward_interaction_count"]
    ):
        failures.append(
            f"{label}: ordered-run forward interactions cannot be smaller than "
            "the dispatched row-node thread count"
        )
    if (
        parsed_structure["node_forward_interaction_count"]
        != parsed_structure["material_word_vjp_interaction_count"]
    ):
        failures.append(
            f"{label}: one fixed-program material step must scan the same "
            "ordered run-node incidences in forward and reverse"
        )
    if (
        parsed_work["native_material_vjp_launch_count"]
        != dimensions["active_native_block_count"]
    ):
        failures.append(f"{label}: one material word VJP is required per active block")

    for key in REQUIRED_MEASUREMENT_BOOL_KEYS:
        if measurement.get(key) is not True:
            failures.append(f"{label}: measurement.{key} must be true")
    if measurement.get("measurement_scope") != ALLOCATOR_PEAK_SCOPE:
        failures.append(f"{label}: measurement does not cover the whole cold step")
    if (
        measurement.get("mps_sampled_measurement_provenance")
        != MPS_SAMPLED_MEASUREMENT_PROVENANCE
    ):
        failures.append(f"{label}: sampled MPS provenance changed")
    if (
        measurement.get("autograd_saved_tensor_measurement_provenance")
        != AUTOGRAD_SAVED_TENSOR_MEASUREMENT_PROVENANCE
    ):
        failures.append(f"{label}: saved-tensor measurement provenance changed")
    if (
        measurement.get("completion_fence_provenance")
        != COMPLETION_FENCE_PROVENANCE
    ):
        failures.append(f"{label}: producer completion-fence provenance changed")
    if measurement.get("mps_exact_peak_claimed") is not False:
        failures.append(f"{label}: sampled MPS counters cannot claim an exact peak")
    parsed_measurement: dict[str, Any] = {}
    for key in REQUIRED_MEASUREMENT_STRING_KEYS:
        value = measurement.get(key)
        if not _nonempty_string(value):
            failures.append(f"{label}: measurement.{key} must be nonempty")
        else:
            parsed_measurement[key] = value
    for measurement_key, producer_key in (
        ("source_manifest_sha256", "source_manifest_sha256"),
        ("trial_driver_sha256", "trial_driver_sha256"),
        ("trial_config_sha256", "trial_config_sha256"),
        ("hardware_fingerprint_sha256", "hardware_fingerprint_sha256"),
        ("native_extension_sha256", "native_extension_sha256"),
        ("mps_memory_limit_sha256", "mps_memory_limit_sha256"),
    ):
        if measurement.get(measurement_key) != producer_binding.get(producer_key):
            failures.append(
                f"{label}: measurement.{measurement_key} does not match producer_binding"
            )
    for key in (
        *PRODUCER_INVARIANT_MEASUREMENT_KEYS,
        *PRODUCER_UNIQUE_MEASUREMENT_KEYS,
        "selected_kernel_resource_attestation_sha256",
    ):
        try:
            _require_sha256(measurement.get(key), f"measurement.{key}")
        except (TypeError, ValueError) as exc:
            failures.append(f"{label}: {exc}")
    for key in REQUIRED_MEASUREMENT_BYTE_KEYS:
        value = _integer(measurement.get(key))
        if value is None or value < 0:
            failures.append(f"{label}: measurement.{key} must be nonnegative integer")
        else:
            parsed_measurement[key] = value
    sample_count = _integer(measurement.get("mps_memory_sample_count"))
    if sample_count is None or sample_count < 2:
        failures.append(f"{label}: MPS sampler requires at least two samples")
    else:
        parsed_measurement["mps_memory_sample_count"] = sample_count
    try:
        sample_interval_ms = _positive_real(
            measurement.get("mps_memory_sampling_interval_ms"),
            name="mps_memory_sampling_interval_ms",
        )
    except (TypeError, ValueError) as exc:
        failures.append(f"{label}: {exc}")
    else:
        parsed_measurement["mps_memory_sampling_interval_ms"] = sample_interval_ms
        if (
            sample_interval_ms
            != float(contract["required_mps_memory_sampling_interval_ms"])
            or sample_interval_ms
            != float(producer_binding.get("mps_memory_sample_interval_ms", -1.0))
        ):
            failures.append(
                f"{label}: MPS sampling cadence differs from the bound producer"
            )
    saved_count = _integer(measurement.get("autograd_saved_tensor_count"))
    if saved_count is None or saved_count < 0:
        failures.append(
            f"{label}: measurement.autograd_saved_tensor_count must be nonnegative integer"
        )
    else:
        parsed_measurement["autograd_saved_tensor_count"] = saved_count
        if saved_count != 0:
            failures.append(f"{label}: manual VJP retained saved autograd tensors")
    if any(key not in parsed_measurement for key in REQUIRED_MEASUREMENT_BYTE_KEYS):
        return None
    if any(key not in parsed_measurement for key in REQUIRED_MEASUREMENT_STRING_KEYS):
        return None
    if "autograd_saved_tensor_count" not in parsed_measurement:
        return None
    if int(parsed_measurement["autograd_saved_tensor_peak_bytes"]) != 0:
        failures.append(f"{label}: saved autograd tensor bytes must be zero")
    parsed_mps_limit = _validate_mps_memory_limit_receipt(
        measurement.get("mps_memory_limit"),
        contract=contract,
        producer_binding=producer_binding,
        label=label,
        failures=failures,
    )
    if parsed_mps_limit is None:
        return None
    if _canonical_payload_sha256(parsed_mps_limit) != measurement.get(
        "mps_memory_limit_sha256"
    ):
        failures.append(f"{label}: MPS memory-limit receipt digest changed")
    parsed_watchdog = _validate_parent_watchdog_receipt(
        measurement.get("parent_watchdog"),
        contract=contract,
        producer_binding=producer_binding,
        label=label,
        failures=failures,
    )
    if parsed_watchdog is None:
        return None
    expected_watchdog_digest = _canonical_payload_sha256(
        {
            "parent_watchdog": parsed_watchdog,
            "trial_execution_evidence_sha256": measurement.get(
                "trial_execution_evidence_sha256"
            ),
        }
    )
    if expected_watchdog_digest != measurement.get(
        "parent_watchdog_evidence_sha256"
    ):
        failures.append(f"{label}: parent watchdog evidence digest changed")
    process_rss_delta = int(parsed_measurement["process_rss_peak_bytes"]) - int(
        parsed_measurement["process_rss_baseline_bytes"]
    )
    mps_current_delta = int(
        parsed_measurement["mps_current_allocated_sampled_maximum_bytes"]
    ) - int(parsed_measurement["mps_current_allocated_baseline_bytes"])
    mps_driver_delta = int(
        parsed_measurement["mps_driver_allocated_sampled_maximum_bytes"]
    ) - int(parsed_measurement["mps_driver_allocated_baseline_bytes"])
    if backend != "mps":
        failures.append(f"{label}: schema-v3 sampled-counter gate requires MPS")
    if process_rss_delta <= 0:
        failures.append(f"{label}: measured process RSS high-water delta must be positive")
    if mps_current_delta <= 0:
        failures.append(f"{label}: sampled MPS tensor-memory delta must be positive")
    if mps_driver_delta <= 0:
        failures.append(f"{label}: sampled MPS driver-memory delta must be positive")
    if process_rss_delta > int(contract["maximum_process_rss_peak_delta_bytes"]):
        failures.append(f"{label}: process RSS high-water delta exceeds the contract")
    if mps_current_delta > int(contract["maximum_sampled_mps_current_delta_bytes"]):
        failures.append(f"{label}: sampled MPS tensor-memory delta exceeds the contract")
    if mps_driver_delta > int(contract["maximum_sampled_mps_driver_delta_bytes"]):
        failures.append(f"{label}: sampled MPS driver-memory delta exceeds the contract")
    effective_mps_limit = int(
        parsed_mps_limit["effective_working_set_limit_bytes"]
    )
    if int(
        parsed_measurement["mps_current_allocated_sampled_maximum_bytes"]
    ) > effective_mps_limit:
        failures.append(f"{label}: sampled MPS tensor maximum exceeds the applied limit")
    if int(
        parsed_measurement["mps_driver_allocated_sampled_maximum_bytes"]
    ) > effective_mps_limit:
        failures.append(f"{label}: sampled MPS driver maximum exceeds the applied limit")

    kernel_attestation = measurement.get("selected_kernel_resource_attestation")
    parsed_attestation = _validate_selected_kernel_resource_attestation(
        kernel_attestation,
        contract=contract,
        label=label,
        failures=failures,
    )
    if parsed_attestation is None:
        return None
    if _canonical_payload_sha256(parsed_attestation) != measurement.get(
        "selected_kernel_resource_attestation_sha256"
    ):
        failures.append(f"{label}: selected-kernel attestation digest changed")

    parsed_measurement.update(
        {
            "process_rss_peak_delta_bytes": process_rss_delta,
            "mps_current_allocated_sampled_delta_bytes": mps_current_delta,
            "mps_driver_allocated_sampled_delta_bytes": mps_driver_delta,
            "mps_memory_limit": parsed_mps_limit,
            "parent_watchdog": parsed_watchdog,
            "selected_kernel_resource_attestation": parsed_attestation,
        }
    )
    return {
        "repeat_index": repeat_index,
        "logical": parsed_logical,
        "logical_frame_invariant_live_upper_bound_bytes": logical_live,
        "work": parsed_work,
        "structure": parsed_structure,
        "measurement": parsed_measurement,
    }


def _validate_cross_frame_scaling(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    failures: list[str],
) -> None:
    first_dimensions = rows[0]["dimensions"]
    for row in rows[1:]:
        for key in ROW_FIXED_DIMENSION_KEYS:
            if row["dimensions"][key] != first_dimensions[key]:
                failures.append(
                    f"{row['frame_count']}f: {key} changed across the frame matrix"
                )

    slack = int(contract["logical_frame_invariant_absolute_slack_bytes"])
    for key in FRAME_INVARIANT_LOGICAL_KEYS:
        values = [
            int(trial["logical"][key])
            for row in rows
            for trial in row["trials"]
        ]
        if max(values) - min(values) > slack:
            failures.append(
                f"logical.{key} changed with frame count/repeat by "
                f"{max(values) - min(values)} bytes"
            )

    for key in (
        "native_material_vjp_launch_count",
        "structural_compile_track_count",
        "cold_artifact_compile_count",
    ):
        values = [
            int(trial["work"][key])
            for row in rows
            for trial in row["trials"]
        ]
        if len(set(values)) != 1:
            failures.append(f"work.{key} changed with frame count/repeat")
    for key in (
        *REQUIRED_STRUCTURE_STRING_KEYS,
        *REQUIRED_STRUCTURE_INT_KEYS,
        "chart_node_ranks",
    ):
        values = [
            trial["structure"][key]
            for row in rows
            for trial in row["trials"]
        ]
        if len(set(values)) != 1:
            failures.append(f"structure.{key} changed with frame count/repeat")
    process_ids = [
        trial["measurement"]["process_generation_id"]
        for row in rows
        for trial in row["trials"]
    ]
    if len(set(process_ids)) != len(process_ids):
        failures.append("fresh-process trials reused a process_generation_id")
    for key in REQUIRED_MEASUREMENT_STRING_KEYS:
        if key in {"process_generation_id", *PRODUCER_UNIQUE_MEASUREMENT_KEYS}:
            continue
        values = [
            trial["measurement"][key]
            for row in rows
            for trial in row["trials"]
        ]
        if len(set(values)) != 1:
            failures.append(f"measurement.{key} changed across the matrix")
    for key in PRODUCER_UNIQUE_MEASUREMENT_KEYS:
        values = [
            trial["measurement"][key]
            for row in rows
            for trial in row["trials"]
        ]
        if len(set(values)) != len(values):
            failures.append(f"measurement.{key} was reused across fresh trials")
    first_sample_launches = statistics.median(
        trial["work"]["sample_launch_count"] for trial in rows[0]["trials"]
    )
    last_sample_launches = statistics.median(
        trial["work"]["sample_launch_count"] for trial in rows[-1]["trials"]
    )
    if last_sample_launches <= first_sample_launches:
        failures.append(
            "sample_launch_count did not grow while dense observations increased; "
            "the artifact does not demonstrate streamed temporal work"
        )
    first_sample_interactions = statistics.median(
        trial["work"]["sample_node_interaction_count"]
        for trial in rows[0]["trials"]
    )
    last_sample_interactions = statistics.median(
        trial["work"]["sample_node_interaction_count"]
        for trial in rows[-1]["trials"]
    )
    if last_sample_interactions <= first_sample_interactions:
        failures.append(
            "sample_node_interaction_count did not grow with dense temporal replay"
        )

    summary = _scaling_summary(rows)
    max_scale = float(contract["maximum_sampled_memory_scale"])
    if summary["process_rss_peak_delta_scale"] > max_scale:
        failures.append("process RSS peak delta scale exceeds the contract")
    if summary["mps_current_allocated_sampled_delta_scale"] > max_scale:
        failures.append("sampled MPS tensor-memory delta scale exceeds the contract")
    if summary["mps_driver_allocated_sampled_delta_scale"] > max_scale:
        failures.append("sampled MPS driver-memory delta scale exceeds the contract")

    process_rss_growth = summary["process_rss_peak_delta_growth_bytes"]
    mps_current_growth = summary[
        "mps_current_allocated_sampled_delta_growth_bytes"
    ]
    mps_driver_growth = summary[
        "mps_driver_allocated_sampled_delta_growth_bytes"
    ]
    if process_rss_growth > int(contract["maximum_process_rss_growth_bytes"]):
        failures.append("process RSS peak growth exceeds the absolute contract")
    if mps_current_growth > int(
        contract["maximum_sampled_mps_current_growth_bytes"]
    ):
        failures.append("sampled MPS tensor-memory growth exceeds the absolute contract")
    if mps_driver_growth > int(
        contract["maximum_sampled_mps_driver_growth_bytes"]
    ):
        failures.append("sampled MPS driver-memory growth exceeds the absolute contract")

    dense_growth = (
        (rows[-1]["frame_count"] - rows[0]["frame_count"])
        * first_dimensions["view_count"]
        * first_dimensions["image_height"]
        * first_dimensions["image_width"]
        * 3
        * 4
    )
    dense_fraction_limit = (
        float(contract["maximum_dense_video_growth_fraction"]) * dense_growth
    )
    if process_rss_growth > dense_fraction_limit:
        failures.append(
            "process RSS peak growth is consistent with hidden O(F*pixels) retention"
        )
    if mps_current_growth > dense_fraction_limit:
        failures.append(
            "sampled MPS tensor-memory growth is consistent with hidden O(F*pixels) retention"
        )
    if mps_driver_growth > dense_fraction_limit:
        failures.append(
            "sampled MPS driver-memory growth is consistent with hidden O(F*pixels) retention"
        )


def _scaling_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "frame_scale": rows[-1]["frame_count"] / rows[0]["frame_count"]
    }
    for label, measurement_key in (
        ("process_rss_peak_delta", "process_rss_peak_delta_bytes"),
        (
            "mps_current_allocated_sampled_delta",
            "mps_current_allocated_sampled_delta_bytes",
        ),
        (
            "mps_driver_allocated_sampled_delta",
            "mps_driver_allocated_sampled_delta_bytes",
        ),
    ):
        medians = [
            statistics.median(
                int(trial["measurement"][measurement_key])
                for trial in row["trials"]
            )
            for row in rows
        ]
        maxima = [
            max(
                int(trial["measurement"][measurement_key])
                for trial in row["trials"]
            )
            for row in rows
        ]
        result[f"{label}_median_bytes_by_frame"] = medians
        result[f"{label}_max_bytes_by_frame"] = maxima
        result[f"{label}_growth_bytes"] = max(maxima) - min(maxima)
        result[f"{label}_scale"] = _growth_scale(min(maxima), max(maxima))
    return result


def _growth_scale(first: float, last: float) -> float:
    if first == 0.0:
        return 1.0 if last == 0.0 else float("inf")
    return last / first


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _positive_int(value: Any, name: str) -> int:
    normalized = _integer(value)
    if normalized is None or normalized < 1:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _nonnegative_int(value: Any, name: str) -> int:
    normalized = _integer(value)
    if normalized is None or normalized < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return normalized


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _positive_real(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be positive finite")
    return float(value)


def _positive_int_sequence(value: Any, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")
    return tuple(_positive_int(item, f"{name} item") for item in value)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _validate_producer_binding(
    binding: Mapping[str, Any],
    *,
    source_tree_sha256: str,
) -> None:
    _require_mapping(binding, "producer_binding")
    if binding.get("producer_name") != PRODUCER_NAME:
        raise ValueError("producer_name is missing or wrong")
    if _integer(binding.get("schema_version")) != PRODUCER_SCHEMA_VERSION:
        raise ValueError("producer binding schema is missing or stale")
    if binding.get("fresh_process_per_trial") is not True:
        raise ValueError("producer must launch every trial in a fresh process")
    if binding.get("material_only_scope") is not True:
        raise ValueError("producer must preserve the material-only scope")
    if binding.get("real_native_required") is not True:
        raise ValueError("producer must require the compiled native backend")
    file_count = _integer(binding.get("source_manifest_file_count"))
    if file_count is None or file_count < 1:
        raise ValueError("source_manifest_file_count must be positive")
    for key in PRODUCER_BINDING_STRING_KEYS:
        if not _nonempty_string(binding.get(key)):
            raise ValueError(f"producer_binding.{key} must be nonempty")
    if (
        binding.get("worker_watchdog_rss_measurement_kind")
        != PARENT_WATCHDOG_RSS_MEASUREMENT_KIND
    ):
        raise ValueError("producer parent-watchdog measurement kind changed")
    for key in PRODUCER_BINDING_SHA_KEYS:
        _require_sha256(binding.get(key), f"producer_binding.{key}")
    _positive_int(
        binding.get("maximum_mps_working_set_bytes"),
        "producer_binding.maximum_mps_working_set_bytes",
    )
    _positive_int(
        binding.get("worker_process_group_rss_limit_bytes"),
        "producer_binding.worker_process_group_rss_limit_bytes",
    )
    _positive_real(
        binding.get("worker_timeout_seconds"),
        name="producer_binding.worker_timeout_seconds",
    )
    _positive_real(
        binding.get("worker_watchdog_poll_interval_seconds"),
        name="producer_binding.worker_watchdog_poll_interval_seconds",
    )
    _positive_real(
        binding.get("mps_memory_sample_interval_ms"),
        name="producer_binding.mps_memory_sample_interval_ms",
    )
    mps_memory_limit = _require_mapping(
        binding.get("mps_memory_limit"),
        "producer_binding.mps_memory_limit",
    )
    if _canonical_payload_sha256(mps_memory_limit) != binding.get(
        "mps_memory_limit_sha256"
    ):
        raise ValueError("producer MPS memory-limit receipt hash changed")
    if binding.get("source_manifest_sha256") != source_tree_sha256:
        raise ValueError("producer source manifest does not bind source_tree_sha256")


def _validate_source_manifest(
    manifest: list[Any],
    *,
    expected_sha256: str,
) -> None:
    _require_sha256(expected_sha256, "source_manifest expected_sha256")
    paths: list[str] = []
    for index, raw_record in enumerate(manifest):
        record = _require_mapping(raw_record, f"source_manifest[{index}]")
        if set(record) != {"path", "size_bytes", "sha256"}:
            raise ValueError(f"source_manifest[{index}] has noncanonical keys")
        path = record.get("path")
        size_bytes = _integer(record.get("size_bytes"))
        if not _nonempty_string(path) or str(path).startswith("/") or ".." in str(path).split("/"):
            raise ValueError(f"source_manifest[{index}].path is not canonical")
        if size_bytes is None or size_bytes < 0:
            raise ValueError(f"source_manifest[{index}].size_bytes is invalid")
        _require_sha256(record.get("sha256"), f"source_manifest[{index}].sha256")
        paths.append(str(path))
    if paths != sorted(set(paths)):
        raise ValueError("source_manifest paths must be sorted and unique")
    rendered = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    if hashlib.sha256(rendered).hexdigest() != expected_sha256:
        raise ValueError("source_manifest digest does not match source_tree_sha256")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify measured WorldFoam memory is frame-sublinear."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify_artifact(args.artifact, args.contract)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()


__all__ = [
    "BENCHMARK",
    "ACTIVE_MATERIAL_MODEL_FORMULA",
    "ALLOCATOR_PEAK_SCOPE",
    "DEFAULT_CONTRACT",
    "PRODUCER_NAME",
    "PRODUCER_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SCOPE_LIMIT",
    "build_artifact",
    "build_row_from_normalized_trials",
    "build_trial_from_fixed_site_accounting",
    "file_sha256",
    "load_json_object",
    "validate_contract",
    "verify_artifact",
    "verify_artifact_payload",
]
