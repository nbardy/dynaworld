from __future__ import annotations

import copy
import hashlib
import json
import math

import pytest

import verify_worldfoam_memory_scaling_acceptance as verifier


def _contract() -> dict[str, object]:
    return verifier.load_json_object(verifier.DEFAULT_CONTRACT)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_manifest() -> list[dict[str, object]]:
    return [
        {
            "path": "research_experiments/world_foam_lane2/test_driver.py",
            "size_bytes": 17,
            "sha256": "0" * 64,
        }
    ]


def _source_manifest_sha256() -> str:
    return hashlib.sha256(
        json.dumps(
            _source_manifest(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _mps_memory_limit() -> dict[str, object]:
    recommended = 4 * 1024**3
    requested_fraction = 0.35
    absolute_limit = 2 * 1024**3
    return {
        "requested_fraction": requested_fraction,
        "effective_fraction": requested_fraction,
        "recommended_max_memory_bytes": recommended,
        "absolute_working_set_limit_bytes": absolute_limit,
        "effective_working_set_limit_bytes": int(requested_fraction * recommended),
    }


def _parent_watchdog(frame_count: int, repeat_index: int) -> dict[str, object]:
    return {
        "returncode": 0,
        "elapsed_seconds": 10.0 + frame_count / 100.0 + repeat_index,
        "rss_measurement_kind": "parent-ps-sampled-high-water",
        "rss_sampling_interval_seconds": 0.25,
        "sampled_process_group_rss_high_water_bytes": 800_000_000,
        "sample_count": 40,
        "worker_timeout_seconds": 1800.0,
        "worker_process_group_rss_limit_bytes": 4 * 1024**3,
        "watchdog_completed": True,
        "process_group_empty_after_exit": True,
        "worker_terminated_by_watchdog": False,
    }


def _producer_binding() -> dict[str, object]:
    mps_memory_limit = _mps_memory_limit()
    return {
        "producer_name": verifier.PRODUCER_NAME,
        "schema_version": verifier.PRODUCER_SCHEMA_VERSION,
        "fresh_process_per_trial": True,
        "material_only_scope": True,
        "real_native_required": True,
        "source_manifest_sha256": _source_manifest_sha256(),
        "source_manifest_file_count": 1,
        "trial_driver_path": "research_experiments/world_foam_lane2/real_driver.py",
        "trial_driver_sha256": "b" * 64,
        "trial_config_path": "configs/worldfoam_memory.json",
        "trial_config_sha256": "c" * 64,
        "hardware_fingerprint_sha256": "d" * 64,
        "hardware_summary": "test mps",
        "native_extension_path": "native/_C.test.so",
        "native_extension_sha256": "e" * 64,
        "producer_source_sha256": "f" * 64,
        "python_executable": "/test/python",
        "command_protocol": "argv-no-shell+nonce-bound-receipt-v1",
        "worker_timeout_seconds": 1800.0,
        "worker_process_group_rss_limit_bytes": 4 * 1024**3,
        "worker_watchdog_rss_measurement_kind": (
            "parent-ps-sampled-high-water"
        ),
        "worker_watchdog_poll_interval_seconds": 0.25,
        "maximum_mps_working_set_bytes": 2 * 1024**3,
        "mps_memory_limit": mps_memory_limit,
        "mps_memory_limit_sha256": verifier._canonical_payload_sha256(
            mps_memory_limit
        ),
        "mps_memory_sample_interval_ms": 5.0,
    }


def _selected_kernel_resource_attestation() -> dict[str, object]:
    operator_names = (
        "kinetic_precompiled_length_p0_lie_node_forward_launch_only",
        "kinetic_ragged_p0_lie_sample_accumulate_loss_only_launch_only",
        "kinetic_precompiled_length_p0_lie_material_node_vjp_accumulate_launch_only",
    )
    metal_function_names = (
        "wf2_kinetic_precompiled_length_p0_lie_node_forward_tensor",
        "wf2_kinetic_ragged_p0_lie_sample_mse_vjp_accumulate_only_tensor",
        "wf2_kinetic_precompiled_length_p0_lie_material_node_vjp_tensor",
    )
    return {
        "kernels": [
            {
                "operator_name": operator_name,
                "metal_function_name": metal_function_name,
                "max_threads_per_threadgroup": 256,
                "thread_execution_width": 32,
                "static_threadgroup_memory_length_bytes": 0,
                "queried_from_compiled_metal_kernel_function": True,
                "static_threadgroup_memory_length_observable": True,
                "register_bytes_per_thread": None,
                "register_bytes_per_thread_observable": False,
                "private_memory_bytes_per_thread": None,
                "private_memory_bytes_per_thread_observable": False,
                "compiler_spill_bytes": None,
                "compiler_spill_bytes_observable": False,
            }
            for operator_name, metal_function_name in zip(
                operator_names,
                metal_function_names,
                strict=True,
            )
        ],
        "abi_namespace": "world_foam_lane2_fused_slab_v0",
        "compiled_operator_name": (
            "kinetic_memory_light_selected_kernel_resource_attestation"
        ),
        "selected_execution_path": "kinetic_material_only",
        "queried_properties": [
            "MetalKernelFunction::getMaxThreadsPerThreadgroup()",
            "MetalKernelFunction::getThreadExecutionWidth()",
            "MetalKernelFunction::getStaticThreadGroupMemoryLength()",
        ],
        "compiled_abi_schema_verified": True,
        "compiled_source_mtime_gate_passed": True,
        "optional_full_geometry_vjp_included": False,
        "kernel_execution_verified_by_this_query": False,
        "native_private_or_spill_bytes_measured": False,
    }


def _structure() -> dict[str, object]:
    return {
        "compiler_provenance": "kinetic-owner-chart-compiler-v1",
        "active_material_model_formula": (
            "sum_q(16*S_q+32*Q_q*J_q)+16*U+16*max(S_q)+4*n_q+16"
        ),
        "world_generation_digest": "1" * 64,
        "camera_generation_digest": "2" * 64,
        "physical_interval_digest": "3" * 64,
        "tolerance_policy_digest": "4" * 64,
        "structural_signature_sha256": "5" * 64,
        "event_count": 48,
        "track_chart_row_count": 4096,
        "word_entry_count": 8192,
        "fallback_count": 0,
        "node_forward_launch_count": 576,
        "node_forward_thread_count": 4096 * 16,
        "node_forward_interaction_count": 8192 * 16,
        "material_word_vjp_interaction_count": 8192 * 16,
        "active_material_exact_model_bytes": 4_000_000,
        "chart_node_ranks": [16] * 576,
    }


def _trial(frame_count: int, repeat_index: int) -> dict[str, object]:
    height = 288
    width = 512
    observations = frame_count * height * width
    maximum_target_in_flight = 256 * 32
    maximum_sample_in_flight = 32
    measured_delta = 400_000_000 + frame_count * 1_000 + repeat_index * 4_096
    kernel_attestation = _selected_kernel_resource_attestation()
    execution_evidence_sha256 = _sha256(
        f"evidence-{frame_count}-{repeat_index}"
    )
    mps_memory_limit = _mps_memory_limit()
    parent_watchdog = _parent_watchdog(frame_count, repeat_index)
    parent_watchdog_evidence_sha256 = verifier._canonical_payload_sha256(
        {
            "parent_watchdog": parent_watchdog,
            "trial_execution_evidence_sha256": execution_evidence_sha256,
        }
    )
    return {
        "repeat_index": repeat_index,
        "status": "ok",
        "logical": {
            "persistent_world_geometry_tensor_bytes": 1_000_000,
            "persistent_material_state_tensor_bytes": 48_000,
            "persistent_optimizer_state_tensor_bytes": 0,
            "serialized_checkpoint_payload_bytes": 16_000,
            "artifact_store_peak_resident_accounted_bytes": 20_000_000,
            "active_request_logical_tensor_bytes_upper_bound": 5_000_000,
            "step_accumulator_logical_tensor_bytes": 16_004,
            "peak_target_payload_logical_tensor_bytes": maximum_target_in_flight * 12,
            "peak_ray_payload_logical_tensor_bytes": 0,
            "peak_sample_launch_logical_tensor_bytes": (
                4 * maximum_sample_in_flight * 16 + 24 * maximum_sample_in_flight
            ),
            "peak_sample_materialization_logical_tensor_bytes_upper_bound": (
                8_000_000
            ),
            "peak_interpolation_evaluator_scratch_logical_tensor_bytes_upper_bound": (
                4_000_000
            ),
            "peak_public_sample_launch_logical_tensor_bytes": (
                24 * maximum_target_in_flight
                + 4 * maximum_sample_in_flight * 16
                + 24 * maximum_sample_in_flight
                + 4 * maximum_sample_in_flight
                + 20
            ),
            "peak_native_preparation_scratch_logical_tensor_bytes": (
                4 * maximum_sample_in_flight + 20
            ),
            "peak_target_decode_bridge_logical_tensor_bytes": (
                maximum_target_in_flight * 24
            ),
            "peak_source_visible_target_read_logical_tensor_bytes_upper_bound": (
                maximum_target_in_flight * 12
            ),
            "persistent_frame_tensor_bytes": 0,
            "persistent_sample_tensor_bytes": 0,
            "persistent_target_tensor_bytes": 0,
            "persistent_prediction_tensor_bytes": 0,
            "target_source_resident_tensor_bytes": 0,
            "frame_metadata_logical_bytes": 24 * (frame_count + 1),
        },
        "work": {
            "streamed_observation_count": observations,
            "sample_launch_count": math.ceil(observations / 32),
            "sample_node_interaction_count": observations * 16,
            "transferred_target_payload_bytes": observations * 12,
            "native_material_vjp_launch_count": 576,
            "structural_compile_track_count": height * width,
            "cold_artifact_compile_count": 576,
            "direct_selected_pixel_observation_count": observations,
            "bounded_region_selected_pixel_observation_count": 0,
            "full_frame_fallback_observation_count": 0,
            "full_frame_target_materialization_count": 0,
            "bounded_region_target_materialization_count": 0,
        },
        "structure": _structure(),
        "retention": {
            "full_dense_observation_replay": True,
            "sample_and_target_payloads_streamed": True,
            "step_accumulator_retains_frame_axis": False,
            "full_video_target_tensor_retained": False,
            "autograd_graph_retained": False,
            "geometry_trainable": False,
            "fake_native_backend": False,
            "selected_pixel_read_acceptance_capable": True,
            "target_source_decode_budget_enforced_before_allocation": True,
            "selected_pixel_read_mode": "direct_pixels",
            "maximum_simultaneously_decoded_target_frame_count": 0,
            "maximum_in_flight_sample_observation_count": maximum_sample_in_flight,
            "maximum_in_flight_target_observation_count": maximum_target_in_flight,
            "peak_sample_launch_node_count": 16,
        },
        "measurement": {
            "fresh_process": True,
            "completion_fenced_before_measurement": True,
            "autograd_saved_tensor_hooks_enabled": True,
            "cold_compile_included": True,
            "native_runtime_verified": True,
            "production_coordinator_integrated": True,
            "mps_sampled_high_water_measured": True,
            "mps_completion_fenced_before_final_sample": True,
            "process_generation_id": f"fresh-{frame_count}-{repeat_index}",
            "measurement_scope": verifier.ALLOCATOR_PEAK_SCOPE,
            "mps_sampled_measurement_provenance": (
                verifier.MPS_SAMPLED_MEASUREMENT_PROVENANCE
            ),
            "autograd_saved_tensor_measurement_provenance": (
                verifier.AUTOGRAD_SAVED_TENSOR_MEASUREMENT_PROVENANCE
            ),
            "completion_fence_provenance": verifier.COMPLETION_FENCE_PROVENANCE,
            "source_manifest_sha256": _source_manifest_sha256(),
            "trial_driver_sha256": "b" * 64,
            "trial_config_sha256": "c" * 64,
            "hardware_fingerprint_sha256": "d" * 64,
            "native_extension_sha256": "e" * 64,
            "trial_command_sha256": _sha256(
                f"command-{frame_count}-{repeat_index}"
            ),
            "trial_execution_evidence_sha256": execution_evidence_sha256,
            "mps_memory_limit": mps_memory_limit,
            "mps_memory_limit_sha256": verifier._canonical_payload_sha256(
                mps_memory_limit
            ),
            "parent_watchdog": parent_watchdog,
            "parent_watchdog_evidence_sha256": (
                parent_watchdog_evidence_sha256
            ),
            "process_rss_baseline_bytes": 100_000_000,
            "process_rss_peak_bytes": 100_000_000 + measured_delta,
            "mps_current_allocated_baseline_bytes": 50_000_000,
            "mps_current_allocated_sampled_maximum_bytes": (
                50_000_000 + measured_delta
            ),
            "mps_driver_allocated_baseline_bytes": 150_000_000,
            "mps_driver_allocated_sampled_maximum_bytes": (
                150_000_000 + measured_delta
            ),
            "mps_memory_sample_count": 100,
            "mps_memory_sampling_interval_ms": 5.0,
            "mps_exact_peak_claimed": False,
            "autograd_saved_tensor_peak_bytes": 0,
            "autograd_saved_tensor_count": 0,
            "selected_kernel_resource_attestation": kernel_attestation,
            "selected_kernel_resource_attestation_sha256": (
                verifier._canonical_payload_sha256(kernel_attestation)
            ),
        },
    }


def _payload() -> dict[str, object]:
    contract = _contract()
    rows = []
    for frame_count in contract["required_frame_counts"]:  # type: ignore[index]
        assert isinstance(frame_count, int)
        rows.append(
            {
                "frame_count": frame_count,
                "status": "ok",
                "image_height": 288,
                "image_width": 512,
                "view_count": 1,
                "dataset_frame_count": 300,
                "requested_frame_subset_kind": "endpoint_including_even_index_v1",
                "requested_frame_indices_sha256": verifier._canonical_payload_sha256(
                    verifier._endpoint_including_frame_indices(
                        dataset_frame_count=300,
                        requested_frame_count=frame_count,
                    )
                ),
                "world_site_count": 1_000,
                "artifact_working_set_count": 576,
                "active_native_block_count": 576,
                "maximum_tracks_per_request": 256,
                "maximum_target_observations_per_chunk": 256 * 32,
                "maximum_samples_per_launch": 32,
                "maximum_node_count": 16,
                "exact_observation_count": frame_count * 288 * 512,
                "trials": [_trial(frame_count, repeat) for repeat in range(3)],
            }
        )
    payload = verifier.build_artifact(
        backend="mps",
        source_tree_sha256=_source_manifest_sha256(),
        source_manifest=_source_manifest(),
        contract_path=verifier.DEFAULT_CONTRACT,
        producer_binding=_producer_binding(),
        rows=rows,
    )
    return payload


def _verify(payload: dict[str, object]) -> dict[str, object]:
    return verifier.verify_artifact_payload(
        payload,
        _contract(),
        contract_sha256=verifier.file_sha256(verifier.DEFAULT_CONTRACT),
    )


def test_acceptance_field_producer_map_covers_every_required_input() -> None:
    expected = {
        *(f"logical.{key}" for key in verifier.FRAME_INVARIANT_LOGICAL_KEYS),
        *(f"logical.{key}" for key in verifier.ZERO_RETENTION_LOGICAL_KEYS),
        "logical.frame_metadata_logical_bytes",
        *(f"work.{key}" for key in verifier.REQUIRED_WORK_KEYS),
        *(
            f"structure.{key}"
            for key in (
                *verifier.REQUIRED_STRUCTURE_STRING_KEYS,
                *verifier.REQUIRED_STRUCTURE_INT_KEYS,
                "chart_node_ranks",
            )
        ),
        *(f"retention.{key}" for key in verifier.REQUIRED_RETENTION_FLAGS),
        "retention.maximum_simultaneously_decoded_target_frame_count",
        "retention.maximum_in_flight_sample_observation_count",
        "retention.maximum_in_flight_target_observation_count",
        "retention.peak_sample_launch_node_count",
        "retention.selected_pixel_read_mode",
        *(
            f"measurement.{key}"
            for key in (
                *verifier.REQUIRED_MEASUREMENT_BOOL_KEYS,
                *verifier.REQUIRED_MEASUREMENT_BYTE_KEYS,
                *verifier.REQUIRED_MEASUREMENT_STRING_KEYS,
                "autograd_saved_tensor_count",
                "mps_memory_sample_count",
                "mps_memory_sampling_interval_ms",
                "mps_exact_peak_claimed",
                "mps_memory_limit",
                "parent_watchdog",
                "selected_kernel_resource_attestation",
            )
        ),
    }

    assert set(verifier.ACCEPTANCE_FIELD_PRODUCERS) == expected
    assert all(
        producer == "fresh_process_external_phase_sampler_or_native_abi"
        for field, producer in verifier.ACCEPTANCE_FIELD_PRODUCERS.items()
        if field.startswith("measurement.")
    )


def test_complete_measured_frame_matrix_passes() -> None:
    report = _verify(_payload())

    assert report["status"] == "passed"
    assert report["failures"] == []
    assert report["scaling"]["frame_scale"] == 37.5  # type: ignore[index]


def test_frame_matrix_must_reuse_one_fixed_dataset_grid() -> None:
    payload = _payload()
    payload["rows"][0]["dataset_frame_count"] = 8  # type: ignore[index]

    report = _verify(payload)

    assert report["status"] == "failed"
    assert any("dataset_frame_count" in item for item in report["failures"])


def test_requested_frame_subset_must_match_endpoint_including_fixed_grid() -> None:
    payload = _payload()
    payload["rows"][1]["requested_frame_indices_sha256"] = "0" * 64  # type: ignore[index]

    report = _verify(payload)

    assert report["status"] == "failed"
    assert any("requested-frame subset" in item for item in report["failures"])


def test_missing_or_tampered_source_manifest_fails() -> None:
    missing = _payload()
    del missing["source_manifest"]
    missing_report = _verify(missing)

    tampered = _payload()
    tampered["source_manifest"][0]["size_bytes"] = 18  # type: ignore[index]
    tampered_report = _verify(tampered)

    assert missing_report["status"] == "failed"
    assert any("source_manifest" in item for item in missing_report["failures"])
    assert tampered_report["status"] == "failed"
    assert any("digest" in item for item in tampered_report["failures"])


def test_hidden_dense_video_retention_fails_logical_and_measured_checks() -> None:
    payload = _payload()
    last = payload["rows"][-1]["trials"]  # type: ignore[index]
    dense_video_bytes = 300 * 288 * 512 * 3 * 4
    for trial in last:
        trial["logical"]["target_source_resident_tensor_bytes"] = dense_video_bytes
        trial["retention"]["full_video_target_tensor_retained"] = True
        trial["measurement"]["process_rss_peak_bytes"] += dense_video_bytes
        trial["measurement"][
            "mps_current_allocated_sampled_maximum_bytes"
        ] += dense_video_bytes
        trial["measurement"][
            "mps_driver_allocated_sampled_maximum_bytes"
        ] += dense_video_bytes

    report = _verify(payload)

    assert report["status"] == "failed"
    assert any("target_source_resident_tensor_bytes must be zero" in item for item in report["failures"])
    assert any("hidden O(F*pixels) retention" in item for item in report["failures"])


def test_missing_sampled_mps_and_autograd_measurement_fails_closed() -> None:
    payload = _payload()
    trial = payload["rows"][0]["trials"][0]  # type: ignore[index]
    for key in (
        "mps_sampled_high_water_measured",
        "mps_completion_fenced_before_final_sample",
        "autograd_saved_tensor_hooks_enabled",
    ):
        trial["measurement"][key] = False
    trial["measurement"]["autograd_saved_tensor_count"] = 2
    trial["measurement"]["autograd_saved_tensor_peak_bytes"] = 4096

    report = _verify(payload)

    assert report["status"] == "failed"
    assert any(
        "measurement.mps_sampled_high_water_measured must be true" in item
        for item in report["failures"]
    )
    assert any("manual VJP retained saved autograd tensors" in item for item in report["failures"])


def test_full_frame_target_fallback_cannot_satisfy_selected_pixel_gate() -> None:
    payload = _payload()
    trial = payload["rows"][0]["trials"][0]  # type: ignore[index]
    observations = trial["work"]["streamed_observation_count"]
    trial["work"]["direct_selected_pixel_observation_count"] = 0
    trial["work"]["full_frame_fallback_observation_count"] = observations
    trial["work"]["full_frame_target_materialization_count"] = 1
    trial["retention"]["selected_pixel_read_mode"] = "full_frame_fallback"
    trial["retention"]["selected_pixel_read_acceptance_capable"] = False
    trial["retention"]["maximum_simultaneously_decoded_target_frame_count"] = 1

    report = _verify(payload)

    assert report["status"] == "failed"
    assert any(
        "full-frame target fallback is not acceptance-capable" in item
        for item in report["failures"]
    )


def test_selected_pixel_source_must_enforce_budget_before_allocation() -> None:
    payload = _payload()
    trial = payload["rows"][0]["trials"][0]  # type: ignore[index]
    trial["retention"][
        "target_source_decode_budget_enforced_before_allocation"
    ] = False

    report = _verify(payload)

    assert report["status"] == "failed"
    assert any(
        "target_source_decode_budget_enforced_before_allocation must be True"
        in item
        for item in report["failures"]
    )


def test_kernel_attestation_cannot_invent_private_memory_measurement() -> None:
    payload = _payload()
    trial = payload["rows"][0]["trials"][0]  # type: ignore[index]
    attestation = trial["measurement"]["selected_kernel_resource_attestation"]
    attestation["kernels"][0]["private_memory_bytes_per_thread"] = 64
    attestation["kernels"][0]["private_memory_bytes_per_thread_observable"] = True
    trial["measurement"]["selected_kernel_resource_attestation_sha256"] = (
        verifier._canonical_payload_sha256(attestation)
    )

    report = _verify(payload)

    assert report["status"] == "failed"
    assert any(
        "selected kernel 0 resource record is invalid" in item
        for item in report["failures"]
    )


def test_kernel_attestation_seals_the_exact_metal_function() -> None:
    payload = _payload()
    trial = payload["rows"][0]["trials"][0]  # type: ignore[index]
    attestation = trial["measurement"]["selected_kernel_resource_attestation"]
    attestation["kernels"][1]["metal_function_name"] = "sibling_kernel"
    trial["measurement"]["selected_kernel_resource_attestation_sha256"] = (
        verifier._canonical_payload_sha256(attestation)
    )

    report = _verify(payload)

    assert report["status"] == "failed"
    assert any(
        "selected kernel 1 resource record is invalid" in item
        for item in report["failures"]
    )


def test_mps_sampling_cadence_is_exactly_bound() -> None:
    payload = _payload()
    trial = payload["rows"][0]["trials"][0]  # type: ignore[index]
    trial["measurement"]["mps_memory_sampling_interval_ms"] = 6.0

    report = _verify(payload)

    assert report["status"] == "failed"
    assert any(
        "MPS sampling cadence differs from the bound producer" in item
        for item in report["failures"]
    )


def test_parent_watchdog_receipt_is_required_and_cannot_relax_rss_limit() -> None:
    missing = _payload()
    missing_trial = missing["rows"][0]["trials"][0]  # type: ignore[index]
    del missing_trial["measurement"]["parent_watchdog"]

    missing_report = _verify(missing)

    assert any(
        "parent watchdog receipt is missing" in item
        for item in missing_report["failures"]
    )

    relaxed = _payload()
    relaxed_trial = relaxed["rows"][0]["trials"][0]  # type: ignore[index]
    watchdog = relaxed_trial["measurement"]["parent_watchdog"]
    watchdog["worker_process_group_rss_limit_bytes"] = 8 * 1024**3
    relaxed_trial["measurement"]["parent_watchdog_evidence_sha256"] = (
        verifier._canonical_payload_sha256(
            {
                "parent_watchdog": watchdog,
                "trial_execution_evidence_sha256": relaxed_trial["measurement"][
                    "trial_execution_evidence_sha256"
                ],
            }
        )
    )

    relaxed_report = _verify(relaxed)

    assert any(
        "parent watchdog did not complete under bound limits" in item
        for item in relaxed_report["failures"]
    )

    unsampled = _payload()
    unsampled_trial = unsampled["rows"][0]["trials"][0]  # type: ignore[index]
    unsampled_watchdog = unsampled_trial["measurement"]["parent_watchdog"]
    unsampled_watchdog["sampled_process_group_rss_high_water_bytes"] = 0
    unsampled_trial["measurement"]["parent_watchdog_evidence_sha256"] = (
        verifier._canonical_payload_sha256(
            {
                "parent_watchdog": unsampled_watchdog,
                "trial_execution_evidence_sha256": unsampled_trial["measurement"][
                    "trial_execution_evidence_sha256"
                ],
            }
        )
    )

    unsampled_report = _verify(unsampled)

    assert any(
        "sampled_process_group_rss_high_water_bytes must be a positive integer"
        in item
        for item in unsampled_report["failures"]
    )

    impossible_cadence = _payload()
    cadence_trial = impossible_cadence["rows"][0]["trials"][0]  # type: ignore[index]
    cadence_watchdog = cadence_trial["measurement"]["parent_watchdog"]
    cadence_watchdog["sample_count"] = 100
    cadence_trial["measurement"]["parent_watchdog_evidence_sha256"] = (
        verifier._canonical_payload_sha256(
            {
                "parent_watchdog": cadence_watchdog,
                "trial_execution_evidence_sha256": cadence_trial["measurement"][
                    "trial_execution_evidence_sha256"
                ],
            }
        )
    )

    cadence_report = _verify(impossible_cadence)

    assert any(
        "parent watchdog did not complete under bound limits" in item
        for item in cadence_report["failures"]
    )


def test_applied_mps_limit_receipt_and_sampled_absolute_maximum_are_bound() -> None:
    relaxed = _payload()
    relaxed_trial = relaxed["rows"][0]["trials"][0]  # type: ignore[index]
    limit = relaxed_trial["measurement"]["mps_memory_limit"]
    limit["absolute_working_set_limit_bytes"] = 4 * 1024**3
    relaxed_trial["measurement"]["mps_memory_limit_sha256"] = (
        verifier._canonical_payload_sha256(limit)
    )

    relaxed_report = _verify(relaxed)

    assert any(
        "MPS memory-limit receipt changed or was relaxed" in item
        for item in relaxed_report["failures"]
    )

    exceeded = _payload()
    exceeded_trial = exceeded["rows"][0]["trials"][0]  # type: ignore[index]
    exceeded_trial["measurement"][
        "mps_driver_allocated_sampled_maximum_bytes"
    ] = 1_600_000_000

    exceeded_report = _verify(exceeded)

    assert any(
        "sampled MPS driver maximum exceeds the applied limit" in item
        for item in exceeded_report["failures"]
    )


def test_logical_materialization_bound_must_cover_interpolation_scratch() -> None:
    payload = _payload()
    trial = payload["rows"][0]["trials"][0]  # type: ignore[index]
    trial["logical"][
        "peak_sample_materialization_logical_tensor_bytes_upper_bound"
    ] = 1

    report = _verify(payload)

    assert any(
        "materialization bound does not cover interpolation scratch" in item
        for item in report["failures"]
    )


def test_material_gate_rejects_a_full_geometry_certification_claim() -> None:
    payload = _payload()
    payload["full_geometry_certified"] = True
    payload["scope_limit"] = "full_geometry"

    report = _verify(payload)

    assert report["status"] == "failed"
    assert "material-only memory evidence cannot certify full geometry" in report["failures"]


def test_reverse_structure_must_not_scale_with_frame_density() -> None:
    payload = _payload()
    last = payload["rows"][-1]["trials"]  # type: ignore[index]
    for trial in last:
        trial["work"]["native_material_vjp_launch_count"] = 300
        trial["structure"]["material_word_vjp_interaction_count"] *= 300
        trial["structure"]["structural_signature_sha256"] = "f" * 64

    report = _verify(payload)

    assert report["status"] == "failed"
    assert "work.native_material_vjp_launch_count changed with frame count/repeat" in report["failures"]
    assert "structure.structural_signature_sha256 changed with frame count/repeat" in report["failures"]


def test_fresh_cold_trial_must_compile_every_track_and_artifact() -> None:
    payload = _payload()
    trial = payload["rows"][0]["trials"][0]  # type: ignore[index]
    trial["work"]["structural_compile_track_count"] -= 1
    trial["work"]["cold_artifact_compile_count"] -= 1

    report = _verify(payload)

    assert any(
        "fresh cold trial must compile every dense view/pixel track" in item
        for item in report["failures"]
    )
    assert any(
        "fresh cold trial must compile the complete artifact working set" in item
        for item in report["failures"]
    )


def test_world_block_node_and_batch_dimensions_must_remain_comparable() -> None:
    payload = _payload()
    payload["rows"][-1]["world_site_count"] = 2_000  # type: ignore[index]
    payload["rows"][-1]["maximum_samples_per_launch"] = 64  # type: ignore[index]

    report = _verify(payload)

    assert report["status"] == "failed"
    assert any("world_site_count changed across the frame matrix" in item for item in report["failures"])
    assert any("maximum_samples_per_launch changed across the frame matrix" in item for item in report["failures"])


def test_impossible_partial_request_and_sample_work_fails() -> None:
    payload = _payload()
    trial = payload["rows"][0]["trials"][0]  # type: ignore[index]
    payload["rows"][0]["artifact_working_set_count"] = 4  # type: ignore[index]
    trial["work"]["sample_launch_count"] = 1
    trial["work"]["sample_node_interaction_count"] = 1

    report = _verify(payload)

    assert report["status"] == "failed"
    assert any("artifact_working_set_count must equal" in item for item in report["failures"])
    assert any("sample launches cannot cover" in item for item in report["failures"])
    assert any("sample-node interactions must cover" in item for item in report["failures"])


def test_intermediate_frame_peak_cannot_hide_between_endpoints() -> None:
    payload = _payload()
    middle_trials = payload["rows"][1]["trials"]  # type: ignore[index]
    for trial in middle_trials:
        trial["measurement"][
            "mps_current_allocated_sampled_maximum_bytes"
        ] += 64_000_000

    report = _verify(payload)

    assert report["status"] == "failed"
    assert (
        "sampled MPS tensor-memory growth exceeds the absolute contract"
        in report["failures"]
    )


def test_fixed_site_accounting_adapter_preserves_separate_memory_categories() -> None:
    frame_count = 8
    observations = frame_count * 288 * 512
    step = {
        "selected_frame_count": frame_count,
        "selected_view_count": 1,
        "image_height": 288,
        "image_width": 512,
        "image_pixel_count": 288 * 512,
        "world_site_count": 1_000,
        "persistent_world_geometry_tensor_bytes": 1_000_000,
        "target_source_resident_tensor_bytes": 0,
        "exact_observation_count": observations,
        "streamed_observation_count": observations,
        "sample_launch_count": math.ceil(observations / 32),
        "sample_node_interaction_count": observations * 16,
        "transferred_target_payload_bytes": observations * 12,
        "native_material_vjp_launch_count": 576,
        "artifact_store_cold_compiled_track_count": 288 * 512,
        "cold_artifact_acquisition_count": 576,
        "material_checkpoint_logical_tensor_bytes": 16_000,
        "artifact_store_peak_resident_accounted_bytes": 20_000_000,
        "active_request_logical_tensor_bytes_upper_bound": 5_000_000,
        "step_accumulator_logical_tensor_bytes": 16_004,
        "peak_cpu_chunk_target_tensor_bytes": 256 * 32 * 12,
        "peak_device_chunk_target_tensor_bytes": 256 * 32 * 12,
        "peak_ray_payload_logical_tensor_bytes": 0,
        "peak_sample_launch_tensor_bytes": (
            4 * 32 * 16 + 24 * 32
        ),
        "peak_sample_materialization_logical_tensor_bytes_upper_bound": 8_000_000,
        "peak_interpolation_evaluator_scratch_logical_tensor_bytes_upper_bound": 4_000_000,
        "peak_public_sample_launch_logical_tensor_bytes": (
            24 * 256 * 32
            + 4 * 32 * 16
            + 24 * 32
            + 4 * 32
            + 20
        ),
        "peak_native_prepared_sample_scratch_tensor_bytes": 4 * 32 + 20,
        "peak_target_decode_bridge_logical_tensor_bytes": 256 * 32 * 24,
        "peak_source_visible_target_read_logical_tensor_bytes_upper_bound": (
            256 * 32 * 12
        ),
        "persistent_frame_tensor_bytes": 0,
        "persistent_sample_tensor_bytes": 0,
        "persistent_target_tensor_bytes": 0,
        "persistent_prediction_tensor_bytes": 0,
        "source_retained_frame_metadata_logical_bytes": 24 * (frame_count + 1),
        "maximum_simultaneously_decoded_target_frame_count": 0,
        "canonical_artifact_working_set_count": 576,
        "maximum_tracks_per_request": 256,
        "maximum_target_observations_per_chunk": 256 * 32,
        "maximum_samples_per_launch": 32,
        "full_dense_observation_replay": True,
        "sample_and_target_payloads_streamed": True,
        "step_accumulator_retains_frame_axis": False,
        "full_video_target_tensor_retained": False,
        "autograd_graph_retained": False,
        "selected_pixel_read_mode": "direct_pixels",
        "selected_pixel_read_acceptance_capable": True,
        "target_source_decode_budget_enforced_before_allocation": True,
        "direct_selected_pixel_observation_count": observations,
        "bounded_region_selected_pixel_observation_count": 0,
        "full_frame_fallback_observation_count": 0,
        "full_frame_target_materialization_count": 0,
        "bounded_region_target_materialization_count": 0,
        "peak_sample_launch_node_count": 16,
        "active_native_block_count": 576,
        **_structure(),
    }
    material_state = {
        "requested_frame_count": frame_count,
        "dataset_frame_count": 300,
        "requested_frame_subset_kind": "endpoint_including_even_index_v1",
        "requested_frame_indices_sha256": verifier._canonical_payload_sha256(
            verifier._endpoint_including_frame_indices(
                dataset_frame_count=300,
                requested_frame_count=frame_count,
            )
        ),
        "total_persistent_tensor_bytes": 48_000,
        "optimizer_history_tensor_bytes": 0,
        "geometry_trainable": False,
    }
    runtime = copy.deepcopy(_trial(frame_count, 0)["measurement"])
    runtime.update(
        {
            "fake_native_backend": False,
            "mps_sampled_memory": {
                "measurement_kind": runtime[
                    "mps_sampled_measurement_provenance"
                ],
                "sampling_interval_ms": runtime[
                    "mps_memory_sampling_interval_ms"
                ],
                "sample_count": runtime["mps_memory_sample_count"],
                "baseline_current_allocated_bytes": runtime[
                    "mps_current_allocated_baseline_bytes"
                ],
                "maximum_current_allocated_bytes": runtime[
                    "mps_current_allocated_sampled_maximum_bytes"
                ],
                "baseline_driver_allocated_bytes": runtime[
                    "mps_driver_allocated_baseline_bytes"
                ],
                "maximum_driver_allocated_bytes": runtime[
                    "mps_driver_allocated_sampled_maximum_bytes"
                ],
                "exact_peak_claimed": False,
                "completion_fenced_before_final_sample": True,
            },
        }
    )
    # Deployment measurement code cannot forge coordinator-owned work or
    # structure by shadowing those names in runtime_measurements.
    runtime["sample_node_interaction_count"] = 1
    runtime["transferred_target_payload_bytes"] = 12
    runtime["structure"] = {"structural_signature_sha256": "f" * 64}

    normalized = verifier.build_trial_from_fixed_site_accounting(
        frame_count=frame_count,
        repeat_index=0,
        maximum_node_count=16,
        persistent_world_geometry_tensor_bytes=1_000_000,
        step_accounting=step,
        material_state_accounting=material_state,
        runtime_measurements=runtime,
    )

    assert normalized["logical"]["persistent_material_state_tensor_bytes"] == 48_000
    assert normalized["logical"]["peak_ray_payload_logical_tensor_bytes"] == 0
    assert normalized["logical"][
        "peak_sample_materialization_logical_tensor_bytes_upper_bound"
    ] == 8_000_000
    assert normalized["logical"][
        "peak_interpolation_evaluator_scratch_logical_tensor_bytes_upper_bound"
    ] == 4_000_000
    assert normalized["logical"]["artifact_store_peak_resident_accounted_bytes"] == 20_000_000
    assert normalized["work"]["sample_node_interaction_count"] == observations * 16
    assert normalized["work"]["transferred_target_payload_bytes"] == observations * 12
    assert normalized["work"]["direct_selected_pixel_observation_count"] == observations
    assert normalized["structure"]["structural_signature_sha256"] == "5" * 64
    assert normalized["measurement"]["mps_sampled_high_water_measured"] is True
    assert normalized["measurement"]["selected_kernel_resource_attestation"] == (
        _selected_kernel_resource_attestation()
    )
    step.pop("full_video_target_tensor_retained")

    with pytest.raises(TypeError, match="full_video_target_tensor_retained"):
        verifier.build_trial_from_fixed_site_accounting(
            frame_count=frame_count,
            repeat_index=0,
            maximum_node_count=16,
            persistent_world_geometry_tensor_bytes=1_000_000,
            step_accounting=step,
            material_state_accounting=material_state,
            runtime_measurements=runtime,
        )
