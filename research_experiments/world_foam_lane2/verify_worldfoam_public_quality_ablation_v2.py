#!/usr/bin/env python3
"""Independently collect and verify the matched selected-ray G4-v2 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "src" / "train"
for import_root in (TRAIN, Path(__file__).resolve().parent):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from config_utils import serialize_config_value  # noqa: E402
from verify_worldfoam_public_quality_ablation import (  # noqa: E402
    REQUIRED_COST,
    REQUIRED_METRICS,
    ROW_KEYS as V1_ROW_KEYS,
    compute_acceptance,
    file_sha256,
    validate_contract as validate_v1_contract,
)
from worldfoam_g4_selected_ray_contract import (  # noqa: E402
    CONTRACT_KIND,
    DEFAULT_CONFIG,
    REQUIRED_ROUTES,
    REQUIRED_SCENES,
    REQUIRED_SEEDS,
    build_matrix_workload_receipts,
    canonical_sha256,
)
from worldfoam_g4_v2_capability import required_source_capability  # noqa: E402


SCHEMA_VERSION = 2
ROW_KIND = "worldfoam-native4d-public-quality-selected-ray-row-v2"
ARTIFACT_KIND = CONTRACT_KIND
ARTIFACT_FILENAME = "worldfoam_public_quality_selected_ray_ablation.json"
ROW_WATCHDOG_KIND = "worldfoam-g4-v2-row-process-group-watchdog-v1"
ROW_WATCHDOG_FILENAME = "process_group_watchdog.json"
ROW_WATCHDOG_MEASUREMENT_KIND = (
    "parent-ps-sampled-process-group-high-water-v1"
)
V2_EXTRA_ROW_KEYS = frozenset(
    {
        "v2_config_path",
        "v2_config_sha256",
        "base_g4_v1_sha256",
        "training_sampling_kind",
        "training_loss_contract",
        "training_loss_contract_sha256",
        "selected_pixels_per_spacetime_sample",
        "selected_loss_scalar_count",
        "route_schedule_sha256",
        "workload_receipt",
        "workload_receipt_generation_digest",
        "full_heldout_target_pixels",
        "training_rasterized_work_claimed_equal",
        "target_source_read_receipt",
        "heldout_execution_receipt",
        "mps_working_set_limit_receipt",
        "parent_rusage_memory_scope",
        "heldout_wall_time_cross_route_comparable",
    }
)
ROW_KEYS = frozenset(V1_ROW_KEYS | V2_EXTRA_ROW_KEYS)


def _repo_path(value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} path is missing")
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"{name} leaves the repository") from error
    return path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _file_identity_errors(value: Any, *, name: str) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{name} identity is missing"]
    try:
        path = _repo_path(value.get("path"), name=name)
    except (TypeError, ValueError) as error:
        return [str(error)]
    errors: list[str] = []
    if not path.is_file():
        return [f"{name} file is missing: {path}"]
    if value.get("bytes") != path.stat().st_size or path.stat().st_size < 1:
        errors.append(f"{name} byte count changed or is empty")
    if value.get("sha256") != file_sha256(path):
        errors.append(f"{name} sha256 changed")
    return errors


def _finite_number(value: Any, *, nonnegative: bool = True) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and (not nonnegative or float(value) >= 0.0)
    )


def _validate_identity_against_path(
    value: Any,
    *,
    path: Path,
    name: str,
) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{name} identity is missing"]
    errors: list[str] = []
    expected_path = str(path.resolve().relative_to(ROOT.resolve()))
    if value.get("path") != expected_path:
        errors.append(f"{name} path changed")
    if not path.is_file():
        return [*errors, f"{name} file is missing"]
    if value.get("bytes") != path.stat().st_size:
        errors.append(f"{name} byte count changed")
    if value.get("sha256") != file_sha256(path):
        errors.append(f"{name} sha256 changed")
    return errors


def validate_row_watchdog(
    value: Any,
    *,
    row: Mapping[str, Any],
    row_path: Path,
    config: Mapping[str, Any],
    config_path: Path,
    source_capability: Mapping[str, Any],
    verify_referenced_files: bool,
) -> list[str]:
    """Validate parent-sampled total host RSS and its raw-row/log bindings."""

    label = str(row.get("row_id", "<unknown>"))
    if not isinstance(value, Mapping):
        return [f"{label}: process-group watchdog receipt is missing"]
    receipt = dict(value)
    required = {
        "schema_version",
        "kind",
        "row_id",
        "worker_argv",
        "worker_command_sha256",
        "v2_config_sha256",
        "source_capability_sha256",
        "row_file",
        "stdout_log",
        "stderr_log",
        "measurement",
        "pre_worker_host_resource_guard",
        "parent_only_rusage_is_not_total_host_memory",
        "cross_route_host_memory_field",
        "generation_digest",
    }
    errors: list[str] = []
    if set(receipt) != required:
        errors.append(f"{label}: process-group watchdog keys changed")
        return errors
    without_digest = {
        key: item for key, item in receipt.items() if key != "generation_digest"
    }
    if (
        receipt["schema_version"] != 1
        or receipt["kind"] != ROW_WATCHDOG_KIND
        or receipt["row_id"] != label
        or receipt["generation_digest"] != canonical_sha256(without_digest)
        or receipt["v2_config_sha256"] != file_sha256(config_path)
        or receipt["source_capability_sha256"]
        != source_capability.get("capability_sha256")
        or receipt["parent_only_rusage_is_not_total_host_memory"] is not True
        or receipt["cross_route_host_memory_field"]
        != "measurement.sampled_process_group_rss_high_water_bytes"
    ):
        errors.append(f"{label}: process-group watchdog binding changed")
    argv = receipt.get("worker_argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        errors.append(f"{label}: worker argv is invalid")
    else:
        expected_suffix = [
            str((ROOT / "src/train/train_worldfoam_native4d_public_quality_row_v2.py").resolve()),
            "--execute",
            "--g4-v2-config",
            str(config_path.resolve()),
            "--protocol",
            str((ROOT / str(row["protocol_path"])).resolve()),
            "--scene",
            str(row["scene"]),
            "--seed",
            str(row["seed"]),
            "--route",
            str(row["route"]),
            "--output",
            str(row_path.resolve()),
            "--maximum-mps-working-set-bytes",
            str(config["execution"]["maximum_mps_working_set_bytes_per_worker"]),
            "--allow-local-mps-execution",
        ]
        if (
            len(argv) != len(expected_suffix) + 1
            or argv[0] != str(ROOT / ".venv" / "bin" / "python")
            or argv[1:] != expected_suffix
            or receipt["worker_command_sha256"] != canonical_sha256(argv)
        ):
            errors.append(f"{label}: worker command changed")
    measurement = receipt.get("measurement")
    required_measurement = {
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
    execution = config["execution"]
    if not isinstance(measurement, Mapping) or set(measurement) != required_measurement:
        errors.append(f"{label}: process-group measurement keys changed")
    elif (
        measurement["returncode"] != 0
        or not _finite_number(measurement["elapsed_seconds"])
        or float(measurement["elapsed_seconds"]) <= 0.0
        or float(measurement["elapsed_seconds"])
        > float(execution["worker_timeout_seconds"])
        or measurement["rss_measurement_kind"] != ROW_WATCHDOG_MEASUREMENT_KIND
        or measurement["rss_sampling_interval_seconds"]
        != execution["worker_watchdog_poll_interval_seconds"]
        or isinstance(measurement["sampled_process_group_rss_high_water_bytes"], bool)
        or not isinstance(measurement["sampled_process_group_rss_high_water_bytes"], int)
        or measurement["sampled_process_group_rss_high_water_bytes"] <= 0
        or measurement["sampled_process_group_rss_high_water_bytes"]
        > execution["maximum_worker_process_group_rss_bytes"]
        or isinstance(measurement["sample_count"], bool)
        or not isinstance(measurement["sample_count"], int)
        or measurement["sample_count"] < 2
        or measurement["worker_timeout_seconds"] != execution["worker_timeout_seconds"]
        or measurement["worker_process_group_rss_limit_bytes"]
        != execution["maximum_worker_process_group_rss_bytes"]
        or measurement["watchdog_completed"] is not True
        or measurement["process_group_empty_after_exit"] is not True
        or measurement["worker_terminated_by_watchdog"] is not False
    ):
        errors.append(f"{label}: process-group RSS evidence failed")
    host_guard = receipt.get("pre_worker_host_resource_guard")
    expected_host_policy = {
        "required": execution["pre_matrix_host_resource_guard_required"],
        "minimum_free_disk_bytes": execution[
            "pre_matrix_minimum_free_disk_bytes"
        ],
        "minimum_available_memory_bytes": execution[
            "pre_matrix_minimum_available_memory_bytes"
        ],
        "maximum_swap_used_bytes": execution[
            "pre_matrix_maximum_swap_used_bytes"
        ],
        "maximum_load_average": execution["pre_matrix_maximum_load_average"],
        "default_dry_plan_samples_host_resources": False,
        "rechecked_immediately_before_every_row": True,
    }
    if not isinstance(host_guard, Mapping):
        errors.append(f"{label}: pre-worker host-resource guard is missing")
    else:
        without_digest = {
            key: item
            for key, item in host_guard.items()
            if key != "generation_digest"
        }
        snapshot = host_guard.get("snapshot")
        if (
            set(host_guard)
            != {
                "schema_version",
                "kind",
                "policy",
                "snapshot",
                "failures",
                "passed_immediately_before_worker",
                "generation_digest",
            }
            or host_guard.get("schema_version") != 1
            or host_guard.get("kind")
            != "worldfoam-g4-v2-pre-worker-host-resource-guard-v1"
            or host_guard.get("policy") != expected_host_policy
            or host_guard.get("failures") != []
            or host_guard.get("passed_immediately_before_worker") is not True
            or not isinstance(snapshot, Mapping)
            or snapshot.get("platform") != "darwin"
            or not isinstance(snapshot.get("free_disk_bytes"), int)
            or int(snapshot.get("free_disk_bytes", 0))
            < execution["pre_matrix_minimum_free_disk_bytes"]
            or not isinstance(snapshot.get("available_memory_bytes"), int)
            or int(snapshot.get("available_memory_bytes", 0))
            < execution["pre_matrix_minimum_available_memory_bytes"]
            or not isinstance(snapshot.get("swap_used_bytes"), int)
            or int(snapshot.get("swap_used_bytes", 0))
            > execution["pre_matrix_maximum_swap_used_bytes"]
            or not _finite_number(snapshot.get("load_average_1m"))
            or float(snapshot.get("load_average_1m", float("inf")))
            > float(execution["pre_matrix_maximum_load_average"])
            or host_guard.get("generation_digest")
            != canonical_sha256(without_digest)
        ):
            errors.append(f"{label}: pre-worker host-resource guard failed")
    if verify_referenced_files:
        errors.extend(
            _validate_identity_against_path(
                receipt.get("row_file"), path=row_path, name=f"{label} watchdog row"
            )
        )
        for key, filename in (
            ("stdout_log", "row_worker.stdout.log"),
            ("stderr_log", "row_worker.stderr.log"),
        ):
            errors.extend(
                _validate_identity_against_path(
                    receipt.get(key),
                    path=row_path.parent / filename,
                    name=f"{label} {key}",
                )
            )
    return errors


def _route_specs(base: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["route"]): row for row in base["routes"]}


def _scene_receipts(base: Mapping[str, Any], base_path: Path) -> Mapping[str, Any]:
    return validate_v1_contract(base, config_path=base_path)["scenes"]


def validate_raw_row(
    row: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    base: Mapping[str, Any],
    base_path: Path,
    config_path: Path = DEFAULT_CONFIG,
    workload: Any,
    source_commit: str,
    verify_referenced_files: bool = True,
) -> list[str]:
    """Validate one raw v2 row without trusting the row worker."""

    label = str(row.get("row_id", "<unknown>"))
    errors: list[str] = []
    if set(row) != set(ROW_KEYS):
        errors.append(f"{label}: row keys changed")
    scene = str(row.get("scene", ""))
    route = str(row.get("route", ""))
    seed = row.get("seed")
    route_spec = _route_specs(base).get(route, {})
    scene_receipt = _scene_receipts(base, base_path).get(scene, {})
    exact = {
        "schema_version": SCHEMA_VERSION,
        "row_kind": ROW_KIND,
        "row_id": f"{scene}/seed_{seed}/{route}",
        "lane": route_spec.get("lane"),
        "execution_mode": route_spec.get("execution_mode"),
        "backend": route_spec.get("backend"),
        "protocol_path": scene_receipt.get("protocol_path"),
        "protocol_sha256": scene_receipt.get("protocol_sha256"),
        "dataset_manifest_path": scene_receipt.get("manifest_path"),
        "dataset_manifest_sha256": scene_receipt.get("manifest_sha256"),
        "sample_id": scene_receipt.get("sample_id"),
        "train_cameras": scene_receipt.get("train_cameras"),
        "heldout_cameras": scene_receipt.get("heldout_cameras"),
        "frame_count": 300,
        "image_size": [384, 512],
        "optimizer_steps": 300,
        "frames_per_step": 4,
        "target_pixel_budget": workload.selected_target_pixels,
        "sample_schedule_sha256": workload.sample_schedule_sha256,
        "source_commit": source_commit,
        "source_dirty": False,
        "public_quality_evidence": True,
        "paper_evidence_eligible": True,
        "proxy_or_test_artifact": False,
        "measurement_is_simulated": False,
        "smoke": False,
        "dataset_is_public": True,
        "calibrated_multiview": True,
        "final_checkpoint_evaluation": True,
        "full_temporal_heldout_evaluation": True,
        "v2_config_path": str(Path(config_path).resolve().relative_to(ROOT)),
        "v2_config_sha256": workload.v2_config_sha256,
        "base_g4_v1_sha256": workload.base_g4_v1_sha256,
        "training_sampling_kind": config["training_sampling"]["kind"],
        "training_loss_contract": config["training_loss"],
        "training_loss_contract_sha256": workload.training_loss_contract_sha256,
        "selected_pixels_per_spacetime_sample": (
            workload.selected_pixels_per_spacetime_sample
        ),
        "selected_loss_scalar_count": workload.selected_loss_scalar_count,
        "route_schedule_sha256": workload.route_schedule_sha256,
        "workload_receipt_generation_digest": workload.generation_digest,
        "full_heldout_target_pixels": workload.heldout_target_pixels,
        "training_rasterized_work_claimed_equal": False,
        "parent_rusage_memory_scope": (
            "worker_parent_only_excludes_children_use_process_group_watchdog"
        ),
        "heldout_wall_time_cross_route_comparable": False,
    }
    for key, expected in exact.items():
        if row.get(key) != expected:
            errors.append(f"{label}: {key} changed")
    if scene not in REQUIRED_SCENES or seed not in REQUIRED_SEEDS or route not in REQUIRED_ROUTES:
        errors.append(f"{label}: row key left the frozen matrix")
    expected_scope = "per_frame" if route == "dynamic_3dgs" else "shared_across_time"
    if row.get("primitive_state_temporal_scope") != expected_scope:
        errors.append(f"{label}: primitive temporal scope changed")
    if row.get("workload_receipt") != workload.as_dict():
        errors.append(f"{label}: workload receipt differs from the independent scheduler")
    source_reads = row.get("target_source_read_receipt")
    expected_ownership = (
        "executor_internal_single_read"
        if route.startswith("worldfoam_")
        else "row_worker_external_single_read"
    )
    if (
        not isinstance(source_reads, Mapping)
        or source_reads.get("selected_pixel_read_observation_count")
        != workload.selected_target_pixels
        or source_reads.get("full_frame_target_materialization_count") != 0
        or source_reads.get("request_schedule_sha256")
        != workload.sample_schedule_sha256
        or source_reads.get("ownership") != expected_ownership
        or source_reads.get("external_row_worker_target_read_call_count")
        != (0 if route.startswith("worldfoam_") else workload.selected_pixel_chunk_count)
        or isinstance(source_reads.get("selected_pixel_read_call_count"), bool)
        or not isinstance(source_reads.get("selected_pixel_read_call_count"), int)
        or source_reads.get("selected_pixel_read_call_count")
        < workload.selected_pixel_chunk_count
        or (
            not route.startswith("worldfoam_")
            and source_reads.get("selected_pixel_read_call_count")
            != workload.selected_pixel_chunk_count
        )
        or source_reads.get("generation_digest")
        != canonical_sha256(
            {
                key: value
                for key, value in source_reads.items()
                if key != "generation_digest"
            }
        )
    ):
        errors.append(f"{label}: selected target source-read receipt changed")
    heldout_execution = row.get("heldout_execution_receipt")
    if not isinstance(heldout_execution, Mapping):
        errors.append(f"{label}: heldout execution receipt is missing")
    elif heldout_execution.get("generation_digest") != canonical_sha256(
        {
            key: value
            for key, value in heldout_execution.items()
            if key != "generation_digest"
        }
    ):
        errors.append(f"{label}: heldout execution receipt digest changed")
    elif route.startswith("worldfoam_"):
        try:
            from worldfoam_spatial_major_heldout_evaluator import (
                validate_spatial_replay_receipt,
            )

            replay = validate_spatial_replay_receipt(heldout_execution)
        except Exception as error:
            replay = {}
            errors.append(
                f"{label}: WorldFoam spatial-major receipt is invalid: "
                f"{type(error).__name__}:{error}"
            )
        session_replay = replay.get("session_receipt", {})
        if (
            replay.get("kind")
            != "worldfoam-spatial-major-heldout-evaluation-v1"
            or replay.get("target_pixel_count") != workload.heldout_target_pixels
            or replay.get("spatial_track_count")
            != workload.heldout_cold_track_compile_count
            or replay.get("spatial_track_block_limit")
            != workload.heldout_cross_time_track_block_size
            or replay.get("spatial_track_block_count")
            != workload.heldout_spatial_major_render_call_count
            or replay.get("metric_pixel_chunk_limit") != 32_768
            or replay.get("metric_pixel_chunk_count")
            != workload.heldout_frame_count * math.ceil((384 * 512) / 32_768)
            or replay.get("lpips_evaluation_count")
            != workload.heldout_frame_count
            or replay.get("media_frame_count") != workload.heldout_frame_count
            or replay.get("metric_and_media_order")
            != "camera_major_then_frame_then_ascending_pixel_chunks"
            or replay.get("prediction_spool_bytes")
            != workload.heldout_target_pixels * 3 * 4
            or replay.get("target_spool_bytes")
            != workload.heldout_target_pixels * 3
            or replay.get("total_spool_bytes")
            != workload.heldout_target_pixels * 15
            or replay.get("native_prediction_target_source_observation_read_count")
            != workload.heldout_target_pixels
            or replay.get("target_spool_source_observation_read_count")
            != workload.heldout_target_pixels
            or replay.get("total_target_source_observation_read_count")
            != 2 * workload.heldout_target_pixels
            or replay.get("total_target_observation_traversal_count")
            != 3 * workload.heldout_target_pixels
            or replay.get("heldout_wall_time_target_io_matched_across_routes")
            is not False
            or not isinstance(session_replay, Mapping)
            or session_replay.get("cold_track_compile_count")
            != workload.heldout_cold_track_compile_count
            or session_replay.get("complete_camera_record_validation_count")
            != workload.heldout_complete_camera_record_validation_count
            or session_replay.get("native_bundle_count")
            != workload.heldout_native_bundle_count
            or session_replay.get("expected_native_bundle_count")
            != workload.heldout_native_bundle_count
            or session_replay.get("native_sample_count")
            != workload.heldout_target_pixels
            or session_replay.get("render_call_count")
            != workload.heldout_spatial_major_render_call_count
            or session_replay.get("frame_major_recompile_per_time_used") is not False
            or session_replay.get("full_pixel_full_temporal") is not True
        ):
            errors.append(f"{label}: WorldFoam heldout route was not spatial-major")
    elif (
        heldout_execution.get("kind")
        != "gaussian-frame-major-full-image-heldout-v1"
        or heldout_execution.get("full_pixel_full_temporal") is not True
        or heldout_execution.get("target_pixel_count") != workload.heldout_target_pixels
    ):
        errors.append(f"{label}: Gaussian heldout execution receipt changed")
    for key in ("evaluator_sha256", "representation_sha256"):
        value = row.get(key)
        if not isinstance(value, str) or len(value) != 64:
            errors.append(f"{label}: {key} is invalid")

    attestation = row.get("route_attestation")
    if not isinstance(attestation, Mapping):
        errors.append(f"{label}: route attestation is missing")
    else:
        common = {
            "real_native": True,
            "native_extension_attested": False,
            "fake_native": False,
            "source_only": False,
            "procedural_target": False,
            "public_target_provider": True,
            "heldout_evaluator": True,
            "full_geometry_trainable": True,
        }
        for key, expected in common.items():
            if attestation.get(key) is not expected:
                errors.append(f"{label}: route attestation {key} changed")
        if attestation.get("compiled_shared_adjoint") is not (
            route == "worldfoam_native4d"
        ):
            errors.append(f"{label}: compiled-adjoint attestation changed")
        if attestation.get("same_representation_framewise_replay") is not (
            route == "worldfoam_framewise_replay"
        ):
            errors.append(f"{label}: framewise-replay attestation changed")

    if verify_referenced_files:
        for key in ("checkpoint", "heldout_media", "wandb_run_file"):
            errors.extend(_file_identity_errors(row.get(key), name=f"{label} {key}"))
    checkpoint = row.get("checkpoint")
    if isinstance(checkpoint, Mapping):
        if checkpoint.get("step") != 300:
            errors.append(f"{label}: checkpoint is not final")
        checkpoint_exact = {
            "training_loss_contract_sha256": workload.training_loss_contract_sha256,
            "sample_schedule_sha256": workload.sample_schedule_sha256,
            "v2_config_sha256": workload.v2_config_sha256,
            "workload_receipt_generation_digest": workload.generation_digest,
            "route_schedule_sha256": workload.route_schedule_sha256,
        }
        if any(checkpoint.get(key) != expected for key, expected in checkpoint_exact.items()):
            errors.append(f"{label}: checkpoint v2 workload binding changed")
    media = row.get("heldout_media")
    if isinstance(media, Mapping) and (
        media.get("camera_ids") != row.get("heldout_cameras")
        or media.get("frame_count") != 300
    ):
        errors.append(f"{label}: heldout media coverage changed")
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != set(REQUIRED_METRICS):
        errors.append(f"{label}: metric keys changed")
    elif any(not _finite_number(metrics[key]) for key in REQUIRED_METRICS):
        errors.append(f"{label}: metrics are invalid")
    cost = row.get("cost")
    if not isinstance(cost, Mapping) or set(cost) != set(REQUIRED_COST):
        errors.append(f"{label}: cost keys changed")
    else:
        if any(not _finite_number(cost[key]) for key in REQUIRED_COST):
            errors.append(f"{label}: cost values are invalid")
        if cost.get("optimizer_steps") != 300:
            errors.append(f"{label}: optimizer-step count changed")
        if cost.get("target_pixels") != workload.selected_target_pixels:
            errors.append(f"{label}: selected target-pixel count changed")
        if _finite_number(cost.get("rasterized_pixels")) and cost["rasterized_pixels"] < cost["target_pixels"]:
            errors.append(f"{label}: rasterized pixels are below loss pixels")
        expected_rasterized = (
            workload.selected_target_pixels
            if route.startswith("worldfoam_")
            else 300 * 4 * 384 * 512
        )
        if cost.get("rasterized_pixels") != expected_rasterized:
            errors.append(f"{label}: route-specific rasterized-pixel count changed")
        if cost.get("final_active_primitive_count_per_render") != 1024:
            errors.append(f"{label}: active primitive count changed")
        expected_stored = 307200 if route == "dynamic_3dgs" else 1024
        if cost.get("stored_primitive_state_count") != expected_stored:
            errors.append(f"{label}: stored primitive state count changed")
        mps_cap = int(
            config["execution"]["maximum_mps_working_set_bytes_per_worker"]
        )
        for key in (
            "sampled_peak_mps_driver_during_training_and_checkpoint_bytes",
            "sampled_peak_mps_driver_through_heldout_evaluation_bytes",
        ):
            if _finite_number(cost.get(key)) and int(cost[key]) > mps_cap:
                errors.append(f"{label}: {key} exceeded the hard MPS cap")
    mps_limit = row.get("mps_working_set_limit_receipt")
    if not isinstance(mps_limit, Mapping):
        errors.append(f"{label}: MPS working-set limit receipt is missing")
    else:
        without_digest = {
            key: item
            for key, item in mps_limit.items()
            if key != "generation_digest"
        }
        if (
            set(mps_limit)
            != {
                "schema_version",
                "kind",
                "requested_working_set_limit_bytes",
                "recommended_max_memory_bytes",
                "effective_fraction",
                "effective_working_set_limit_bytes",
                "installed_before_dataset_executor_native_or_tensor_allocation",
                "generation_digest",
            }
            or mps_limit.get("schema_version") != 1
            or mps_limit.get("kind")
            != "worldfoam-g4-v2-row-mps-working-set-limit-v1"
            or mps_limit.get("requested_working_set_limit_bytes")
            != config["execution"]["maximum_mps_working_set_bytes_per_worker"]
            or not isinstance(mps_limit.get("recommended_max_memory_bytes"), int)
            or int(mps_limit.get("recommended_max_memory_bytes", 0)) < 1
            or not _finite_number(mps_limit.get("effective_fraction"))
            or not 0.0 < float(mps_limit.get("effective_fraction", 0.0)) <= 1.0
            or not isinstance(
                mps_limit.get("effective_working_set_limit_bytes"), int
            )
            or not 0
            < int(mps_limit.get("effective_working_set_limit_bytes", 0))
            <= config["execution"]["maximum_mps_working_set_bytes_per_worker"]
            or mps_limit.get(
                "installed_before_dataset_executor_native_or_tensor_allocation"
            )
            is not True
            or mps_limit.get("generation_digest")
            != canonical_sha256(without_digest)
        ):
            errors.append(f"{label}: MPS working-set limit receipt changed")
    return errors


def artifact_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )


def verify_artifact_file(
    path: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Revalidate one collected artifact without rescanning or rewriting rows."""

    artifact_path = Path(path).resolve()
    config_path = Path(config_path).resolve()
    failures: list[str] = []
    try:
        artifact = _load_json(artifact_path)
    except Exception as error:
        artifact = {}
        failures.append(f"artifact is unreadable: {type(error).__name__}: {error}")
    try:
        config, base, base_path, workloads = build_matrix_workload_receipts(
            config_path
        )
        source_capability = required_source_capability(config_path)
    except Exception as error:
        report_payload = {
            "schema_version": 1,
            "kind": "worldfoam-g4-v2-collected-artifact-verification-v1",
            "accepted": False,
            "public_quality_evidence": False,
            "artifact_path": str(artifact_path),
            "artifact_sha256": None,
            "row_count": 0,
            "failures": [
                *failures,
                f"verification contract is invalid: {type(error).__name__}: {error}",
            ],
        }
        return {
            **report_payload,
            "generation_digest": canonical_sha256(report_payload),
        }

    expected_artifact_keys = {
        "schema_version",
        "artifact_kind",
        "status",
        "public_quality_evidence",
        "proxy_or_test_artifact",
        "measurement_is_simulated",
        "matrix_config",
        "matrix_config_sha256",
        "base_g4_v1_config",
        "base_g4_v1_sha256",
        "source_commit",
        "cross_route_host_memory_source",
        "raw_row_rusage_scope",
        "workload_receipts",
        "workload_receipts_sha256",
        "rows",
        "acceptance",
        "failures",
        "artifact_sha256",
    }
    if set(artifact) != expected_artifact_keys:
        failures.append("collected artifact keys changed")
    if (
        artifact.get("schema_version") != SCHEMA_VERSION
        or artifact.get("artifact_kind") != ARTIFACT_KIND
        or artifact.get("status") != "measured"
        or artifact.get("public_quality_evidence") is not True
        or artifact.get("proxy_or_test_artifact") is not False
        or artifact.get("measurement_is_simulated") is not False
        or artifact.get("matrix_config")
        != str(config_path.relative_to(ROOT))
        or artifact.get("matrix_config_sha256") != file_sha256(config_path)
        or artifact.get("base_g4_v1_config") != str(base_path.relative_to(ROOT))
        or artifact.get("base_g4_v1_sha256") != file_sha256(base_path)
        or artifact.get("cross_route_host_memory_source")
        != (
            "rows[].execution_watchdog.measurement."
            "sampled_process_group_rss_high_water_bytes"
        )
        or artifact.get("raw_row_rusage_scope")
        != "worker_parent_only_excludes_child_processes"
    ):
        failures.append("collected artifact evidence binding changed")
    if artifact.get("artifact_sha256") != artifact_sha256(artifact):
        failures.append("collected artifact digest changed")
    source_commit = artifact.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
    ):
        failures.append("collected artifact source commit is invalid")
    workload_payload = {
        f"{scene}/seed_{seed}": receipt.as_dict()
        for (scene, seed), receipt in workloads.items()
    }
    if (
        artifact.get("workload_receipts") != workload_payload
        or artifact.get("workload_receipts_sha256")
        != canonical_sha256(workload_payload)
    ):
        failures.append("collected artifact workload receipts changed")
    embedded_rows = artifact.get("rows")
    raw_rows: list[dict[str, Any]] = []
    claimed: list[tuple[str, int, str]] = []
    expected_grid = {
        (scene, seed, route)
        for scene in REQUIRED_SCENES
        for seed in REQUIRED_SEEDS
        for route in REQUIRED_ROUTES
    }
    if not isinstance(embedded_rows, list) or len(embedded_rows) != 36:
        failures.append("collected artifact does not contain exactly 36 rows")
        embedded_rows = []
    for embedded in embedded_rows:
        if not isinstance(embedded, Mapping):
            failures.append("collected artifact row is not a mapping")
            continue
        if set(embedded) != set(ROW_KEYS | {"receipt", "execution_watchdog"}):
            failures.append("collected artifact embedded-row keys changed")
            continue
        raw = {key: embedded[key] for key in ROW_KEYS}
        key = (str(raw["scene"]), int(raw["seed"]), str(raw["route"]))
        claimed.append(key)
        workload = workloads.get((key[0], key[1]))
        if workload is None:
            failures.append(f"embedded row left the workload grid: {key}")
            continue
        failures.extend(
            validate_raw_row(
                raw,
                config=config,
                base=base,
                base_path=base_path,
                config_path=config_path,
                workload=workload,
                source_commit=str(source_commit),
                verify_referenced_files=False,
            )
        )
        raw_receipt = embedded["receipt"]
        if (
            not isinstance(raw_receipt, Mapping)
            or set(raw_receipt) != {"path", "bytes", "sha256"}
            or not isinstance(raw_receipt.get("bytes"), int)
            or raw_receipt.get("bytes", 0) < 1
            or not isinstance(raw_receipt.get("sha256"), str)
            or len(raw_receipt.get("sha256", "")) != 64
        ):
            failures.append(f"{raw['row_id']}: embedded raw-row identity is invalid")
        raw_row_reference = (
            ROOT / str(raw_receipt.get("path", ""))
            if isinstance(raw_receipt, Mapping)
            else ROOT / ".invalid-embedded-row-reference"
        )
        watchdog_record = embedded["execution_watchdog"]
        if not isinstance(watchdog_record, Mapping):
            failures.append(f"{raw['row_id']}: embedded watchdog is invalid")
        else:
            watchdog = {
                key_name: item
                for key_name, item in watchdog_record.items()
                if key_name != "receipt_file"
            }
            if isinstance(raw_receipt, Mapping) and watchdog.get("row_file") != raw_receipt:
                failures.append(
                    f"{raw['row_id']}: embedded watchdog row identity differs "
                    "from the embedded raw-row identity"
                )
            failures.extend(
                validate_row_watchdog(
                    watchdog,
                    row=raw,
                    row_path=raw_row_reference,
                    config=config,
                    config_path=config_path,
                    source_capability=source_capability,
                    verify_referenced_files=False,
                )
            )
            receipt_file = watchdog_record.get("receipt_file")
            if (
                not isinstance(receipt_file, Mapping)
                or set(receipt_file) != {"path", "bytes", "sha256"}
                or not isinstance(receipt_file.get("bytes"), int)
                or receipt_file.get("bytes", 0) < 1
            ):
                failures.append(
                    f"{raw['row_id']}: embedded watchdog-file identity is invalid"
                )
        raw_rows.append(raw)
    if len(set(claimed)) != len(claimed) or set(claimed) != expected_grid:
        failures.append("collected artifact embedded row grid changed")
    try:
        recomputed_acceptance = compute_acceptance(raw_rows, base)
    except Exception as error:
        recomputed_acceptance = {"accepted": False, "failures": [str(error)]}
    if (
        recomputed_acceptance.get("accepted") is not True
        or artifact.get("acceptance") != recomputed_acceptance
    ):
        failures.append("collected artifact acceptance changed")
    if artifact.get("failures") != []:
        failures.append("collected artifact records failures")
    unique_failures = sorted(set(failures))
    report_payload = {
        "schema_version": 1,
        "kind": "worldfoam-g4-v2-collected-artifact-verification-v1",
        "accepted": not unique_failures,
        "public_quality_evidence": not unique_failures,
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact.get("artifact_sha256"),
        "row_count": len(raw_rows),
        "failures": unique_failures,
    }
    return {
        **report_payload,
        "generation_digest": canonical_sha256(report_payload),
    }


def collect_and_verify(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_path: Path | None = None,
) -> dict[str, Any]:
    config, base, base_path, workloads = build_matrix_workload_receipts(config_path)
    config_path = Path(config_path).resolve()
    source_capability = required_source_capability(config_path)
    output_root = (ROOT / str(config["output_root"])).resolve()
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    source_commits: set[str] = set()
    claimed_keys: list[tuple[str, int, str]] = []
    for scene in REQUIRED_SCENES:
        for seed in REQUIRED_SEEDS:
            for route in REQUIRED_ROUTES:
                path = output_root / scene / f"seed_{seed}" / route / "g4_v2_row.json"
                if not path.is_file():
                    failures.append(f"missing raw row: {path.relative_to(ROOT)}")
                    continue
                row = _load_json(path)
                source = str(row.get("source_commit", ""))
                source_commits.add(source)
                claimed_keys.append(
                    (
                        str(row.get("scene", "")),
                        int(row.get("seed", -1)),
                        str(row.get("route", "")),
                    )
                )
                failures.extend(
                    validate_raw_row(
                        row,
                        config=config,
                        base=base,
                        base_path=base_path,
                        config_path=config_path,
                        workload=workloads[(scene, seed)],
                        source_commit=source,
                    )
                )
                watchdog_path = path.parent / ROW_WATCHDOG_FILENAME
                if watchdog_path.is_file():
                    watchdog = _load_json(watchdog_path)
                    failures.extend(
                        validate_row_watchdog(
                            watchdog,
                            row=row,
                            row_path=path,
                            config=config,
                            config_path=config_path,
                            source_capability=source_capability,
                            verify_referenced_files=True,
                        )
                    )
                    watchdog_record: Mapping[str, Any] = {
                        **watchdog,
                        "receipt_file": {
                            "path": str(watchdog_path.relative_to(ROOT)),
                            "bytes": watchdog_path.stat().st_size,
                            "sha256": file_sha256(watchdog_path),
                        },
                    }
                else:
                    failures.append(
                        f"{row.get('row_id', path)}: process-group watchdog receipt is missing"
                    )
                    watchdog_record = {}
                rows.append(
                    {
                        **row,
                        "receipt": {
                            "path": str(path.relative_to(ROOT)),
                            "bytes": path.stat().st_size,
                            "sha256": file_sha256(path),
                        },
                        "execution_watchdog": watchdog_record,
                    }
                )
    expected_grid = {
        (scene, seed, route)
        for scene in REQUIRED_SCENES
        for seed in REQUIRED_SEEDS
        for route in REQUIRED_ROUTES
    }
    if len(rows) != 36:
        failures.append(f"expected 36 raw rows, observed {len(rows)}")
    if len(set(claimed_keys)) != len(claimed_keys):
        failures.append("raw matrix contains duplicate claimed row keys")
    if set(claimed_keys) != expected_grid:
        failures.append("raw matrix claimed keys differ from the exact 36-row grid")
    if (
        len(source_commits) != 1
        or any(
            len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
            for value in source_commits
        )
    ):
        failures.append("rows do not share one source commit")
    evaluator_groups: dict[tuple[str, int], set[str]] = {}
    schedule_groups: dict[tuple[str, int], set[str]] = {}
    route_schedule_groups: dict[tuple[str, int], set[str]] = {}
    worldfoam_representation_groups: dict[tuple[str, int], set[str]] = {}
    for row in rows:
        group = (str(row["scene"]), int(row["seed"]))
        evaluator_groups.setdefault(group, set()).add(str(row["evaluator_sha256"]))
        schedule_groups.setdefault(group, set()).add(str(row["sample_schedule_sha256"]))
        route_schedule_groups.setdefault(group, set()).add(str(row["route_schedule_sha256"]))
        if row["route"] in {"worldfoam_native4d", "worldfoam_framewise_replay"}:
            worldfoam_representation_groups.setdefault(group, set()).add(
                str(row["representation_sha256"])
            )
    if any(len(values) != 1 for values in evaluator_groups.values()):
        failures.append("paired routes do not share the common metric/evaluation contract")
    if any(len(values) != 1 for values in schedule_groups.values()):
        failures.append("paired routes do not share the selected sample schedule")
    if any(len(values) != 1 for values in route_schedule_groups.values()):
        failures.append("paired routes do not share the selected route schedule")
    if any(len(values) != 1 for values in worldfoam_representation_groups.values()):
        failures.append("compiled and replay WorldFoam final representations differ")
    try:
        acceptance = compute_acceptance(rows, base)
    except Exception as error:
        acceptance = {"accepted": False, "failures": [str(error)]}
    if acceptance.get("accepted") is not True:
        failures.extend(str(value) for value in acceptance.get("failures", ()))
    workload_payload = {
        f"{scene}/seed_{seed}": receipt.as_dict()
        for (scene, seed), receipt in workloads.items()
    }
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "status": "measured" if not failures else "rejected",
        "public_quality_evidence": not failures,
        "proxy_or_test_artifact": False,
        "measurement_is_simulated": False,
        "matrix_config": str(Path(config_path).resolve().relative_to(ROOT)),
        "matrix_config_sha256": file_sha256(Path(config_path)),
        "base_g4_v1_config": str(base_path.relative_to(ROOT)),
        "base_g4_v1_sha256": file_sha256(base_path),
        "source_commit": next(iter(source_commits)) if len(source_commits) == 1 else None,
        "cross_route_host_memory_source": (
            "rows[].execution_watchdog.measurement."
            "sampled_process_group_rss_high_water_bytes"
        ),
        "raw_row_rusage_scope": "worker_parent_only_excludes_child_processes",
        "workload_receipts": workload_payload,
        "workload_receipts_sha256": canonical_sha256(workload_payload),
        "rows": rows,
        "acceptance": acceptance,
        "failures": sorted(set(failures)),
        "artifact_sha256": "",
    }
    artifact["artifact_sha256"] = artifact_sha256(artifact)
    destination = output_path or (output_root / ARTIFACT_FILENAME)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(serialize_config_value(artifact), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    artifact = collect_and_verify(config_path=args.config, output_path=args.output)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["status"] == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "ARTIFACT_FILENAME",
    "ARTIFACT_KIND",
    "ROW_KEYS",
    "ROW_KIND",
    "SCHEMA_VERSION",
    "artifact_sha256",
    "collect_and_verify",
    "validate_raw_row",
    "validate_row_watchdog",
    "verify_artifact_file",
)
