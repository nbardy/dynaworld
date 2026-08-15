#!/usr/bin/env python3
"""Independently verify the real-native WorldFoam G4-v2 timing pilot.

The pilot is a tractability gate, never a public-quality row.  It must prove
that both WorldFoam routes execute one exact selected-ray optimizer step and a
bounded 300-frame spatial-major replay, while retaining a bitwise comparison
against the old frame-major path.  This verifier intentionally reads the raw
worker logs and all bound source/native files; a producer-authored summary is
not accepted on trust.

Importing this module is allocation-free: it imports neither Torch nor the
native WorldFoam extension.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "src" / "train"
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

from worldfoam_g4_selected_ray_contract import (  # noqa: E402
    DEFAULT_CONFIG,
    canonical_sha256,
    file_sha256,
    load_selected_ray_contract,
)
from worldfoam_g4_v2_capability import required_source_capability  # noqa: E402


PILOT_SCHEMA_VERSION = 1
PILOT_KIND = "worldfoam-g4-v2-selected-ray-real-native-pilot-v1"
ROUTE_KIND = "worldfoam-g4-v2-real-native-pilot-route-v1"
RAW_LOG_KIND = "worldfoam-g4-v2-real-native-pilot-raw-log-v1"
REQUIRED_ROUTES = ("worldfoam_native4d", "worldfoam_framewise_replay")
PILOT_SCENE = "coffee_martini"
PILOT_SEED = 17
PILOT_OPTIMIZER_STEPS = 1
PILOT_TARGET_PIXELS = 4096
PILOT_RGB_SCALARS = PILOT_TARGET_PIXELS * 3
PILOT_FRAME_COUNT = 300
PILOT_SPATIAL_TRACK_COUNT = 128
FULL_HELDOUT_SPATIAL_TRACK_COUNT = 384 * 512
PROJECTION_SAFETY_MULTIPLIER = 1.25
WORKER_RESULT_PREFIX = "G4_V2_PILOT_RESULT="

_TOP_KEYS = {
    "schema_version",
    "kind",
    "status",
    "scene",
    "seed",
    "v2_config_path",
    "v2_config_sha256",
    "source_capability_path",
    "source_capability_sha256",
    "training_loss_contract",
    "training_loss_contract_sha256",
    "public_quality_evidence",
    "pilot_only",
    "spatial_major_full_temporal_heldout_exercised",
    "host_guard",
    "source_binding",
    "native_binding",
    "raw_logs",
    "routes",
    "generation_digest",
}
_ROUTE_KEYS = {
    "schema_version",
    "kind",
    "route",
    "real_native",
    "backend",
    "device",
    "selected_training_optimizer_steps",
    "selected_training_target_pixels",
    "selected_training_rgb_scalar_count",
    "selected_training_spacetime_sample_count",
    "training_loss_identifier",
    "training_loss_contract_sha256",
    "heldout_frame_count_exercised",
    "heldout_spatial_track_count_exercised",
    "heldout_prediction_observation_count",
    "frame_major_parity_track_count",
    "frame_major_parity_observation_count",
    "frame_major_cross_time_bitwise_equal",
    "stage_timings_s",
    "native_counts",
    "compiler_counts",
    "projection",
    "runtime_measurements",
    "native_library_sha256",
    "source_capability_sha256",
    "worker_source_sha256",
    "pilot_transition_receipt",
    "spatial_replay_receipt",
    "generation_digest",
}
_STAGE_TIMING_KEYS = {
    "dataset_and_session_setup",
    "selected_training_step",
    "pilot_heldout_prepare",
    "spatial_major_track_block",
    "frame_major_parity",
    "device_completion_fence",
    "route_total",
}
_PROJECTION_KEYS = {
    "training_step_multiplier",
    "full_heldout_spatial_track_count",
    "measured_spatial_track_block_count",
    "projected_heldout_block_count",
    "safety_multiplier",
    "projected_training_seconds",
    "projected_heldout_seconds",
    "projected_fixed_seconds",
    "projected_total_seconds",
    "projected_full_row_hours",
}
_RUNTIME_KEYS = {
    "worker_process_peak_rss_bytes",
    "worker_mps_baseline_driver_bytes",
    "worker_mps_peak_driver_bytes",
    "worker_mps_effective_limit_bytes",
    "parent_observed_process_group_peak_rss_bytes",
    "parent_watchdog_sample_count",
    "parent_watchdog_timeout_seconds",
    "parent_watchdog_rss_limit_bytes",
    "completion_fenced",
}
_RAW_LOG_KEYS = {
    "path",
    "bytes",
    "sha256",
    "worker_returncode",
    "worker_report_generation_digest",
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _finite_nonnegative(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _repo_file(value: Any, *, label: str) -> Path:
    path = Path(str(value))
    resolved = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"{label} left the repository") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _generation_valid(value: Mapping[str, Any]) -> bool:
    digest = value.get("generation_digest")
    return _is_sha256(digest) and digest == canonical_sha256(
        {key: item for key, item in value.items() if key != "generation_digest"}
    )


def _parse_worker_report(raw_log: Mapping[str, Any]) -> Mapping[str, Any]:
    if set(raw_log) != {
        "schema_version",
        "kind",
        "route",
        "command",
        "returncode",
        "stdout",
        "stderr",
        "generation_digest",
    }:
        raise ValueError("pilot raw-log key set changed")
    if (
        raw_log.get("schema_version") != 1
        or raw_log.get("kind") != RAW_LOG_KIND
        or raw_log.get("returncode") != 0
        or not isinstance(raw_log.get("stdout"), str)
        or not isinstance(raw_log.get("stderr"), str)
        or not isinstance(raw_log.get("command"), list)
        or not _generation_valid(raw_log)
    ):
        raise ValueError("pilot raw-log contract changed")
    matches = [
        line[len(WORKER_RESULT_PREFIX) :]
        for line in raw_log["stdout"].splitlines()
        if line.startswith(WORKER_RESULT_PREFIX)
    ]
    if len(matches) != 1:
        raise ValueError("pilot raw log must contain exactly one worker report")
    report = json.loads(matches[0])
    if not isinstance(report, Mapping):
        raise TypeError("pilot worker report is not a mapping")
    return report


def _validate_source_binding(
    value: Any,
    *,
    verify_files: bool,
    failures: list[str],
) -> None:
    if not isinstance(value, Mapping):
        failures.append("source_binding_missing")
        return
    if set(value) != {
        "repository_commit",
        "repository_dirty",
        "source_manifest",
        "source_manifest_sha256",
        "parent_process_peak_rss_bytes",
    }:
        failures.append("source_binding_keys_changed")
        return
    commit = value.get("repository_commit")
    manifest = value.get("source_manifest")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit.lower())
        or not isinstance(value.get("repository_dirty"), bool)
        or not isinstance(manifest, Mapping)
        or not _positive_int(value.get("parent_process_peak_rss_bytes"))
        or value.get("source_manifest_sha256") != canonical_sha256(manifest)
    ):
        failures.append("source_binding_contract_changed")
        return
    for relative, identity in manifest.items():
        if not isinstance(identity, Mapping) or set(identity) != {"bytes", "sha256"}:
            failures.append(f"source_identity_invalid:{relative}")
            continue
        try:
            path = _repo_file(relative, label="pilot source")
        except (OSError, ValueError) as error:
            failures.append(f"source_file_invalid:{relative}:{type(error).__name__}")
            continue
        if (
            not _positive_int(identity.get("bytes"))
            or not _is_sha256(identity.get("sha256"))
        ):
            failures.append(f"source_identity_invalid:{relative}")
        elif verify_files and (
            path.stat().st_size != identity["bytes"]
            or file_sha256(path) != identity["sha256"]
        ):
            failures.append(f"source_file_drifted:{relative}")


def _validate_native_binding(
    value: Any,
    *,
    verify_files: bool,
    failures: list[str],
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "module",
        "library_path",
        "library_bytes",
        "library_sha256",
        "same_library_both_routes",
    }:
        failures.append("native_binding_contract_changed")
        return
    try:
        path = _repo_file(value.get("library_path"), label="pilot native library")
    except (OSError, ValueError) as error:
        failures.append(f"native_library_invalid:{type(error).__name__}")
        return
    if (
        value.get("module") != "torch_world_foam_lane2_fused_slab.ops"
        or not _positive_int(value.get("library_bytes"))
        or not _is_sha256(value.get("library_sha256"))
        or value.get("same_library_both_routes") is not True
    ):
        failures.append("native_binding_contract_changed")
    elif verify_files and (
        path.stat().st_size != value["library_bytes"]
        or file_sha256(path) != value["library_sha256"]
    ):
        failures.append("native_library_drifted")


def _validate_host_guard(value: Any, failures: list[str]) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "policy",
        "snapshot",
        "failures",
        "passed_before_workers",
    }:
        failures.append("host_guard_contract_changed")
        return
    policy = value.get("policy")
    snapshot = value.get("snapshot")
    if (
        value.get("passed_before_workers") is not True
        or value.get("failures") != []
        or not isinstance(policy, Mapping)
        or not isinstance(snapshot, Mapping)
    ):
        failures.append("host_guard_did_not_pass")
        return
    numeric_valid = True
    for key in (
        "minimum_free_disk_bytes",
        "minimum_available_memory_bytes",
        "maximum_swap_used_bytes",
    ):
        if not _finite_nonnegative(policy.get(key)):
            failures.append(f"host_guard_policy_invalid:{key}")
            numeric_valid = False
    if not _finite_nonnegative(policy.get("maximum_load_average")):
        failures.append("host_guard_policy_invalid:maximum_load_average")
        numeric_valid = False
    for key in ("free_disk_bytes", "available_memory_bytes", "swap_used_bytes"):
        if not _finite_nonnegative(snapshot.get(key)):
            failures.append(f"host_guard_snapshot_invalid:{key}")
            numeric_valid = False
    if not _finite_nonnegative(snapshot.get("load_average_1m")):
        failures.append("host_guard_snapshot_invalid:load_average_1m")
        numeric_valid = False
    if not numeric_valid:
        return
    if (
        snapshot.get("platform") != "darwin"
        or float(policy.get("minimum_free_disk_bytes", -1)) < 8 * 1024**3
        or float(policy.get("minimum_available_memory_bytes", -1)) < 8 * 1024**3
        or float(policy.get("maximum_swap_used_bytes", float("inf")))
        > 2 * 1024**3
        or float(policy.get("maximum_load_average", float("inf"))) > 8.0
        or float(snapshot.get("free_disk_bytes", -1))
        < float(policy.get("minimum_free_disk_bytes", float("inf")))
        or float(snapshot.get("available_memory_bytes", -1))
        < float(policy.get("minimum_available_memory_bytes", float("inf")))
        or float(snapshot.get("swap_used_bytes", float("inf")))
        > float(policy.get("maximum_swap_used_bytes", -1))
        or float(snapshot.get("load_average_1m", float("inf")))
        > float(policy.get("maximum_load_average", -1))
    ):
        failures.append("host_guard_thresholds_not_satisfied")


def _validate_route(
    route: str,
    value: Any,
    *,
    capability_sha256: str,
    training_loss_sha256: str,
    maximum_projected_hours: float,
    failures: list[str],
) -> None:
    if not isinstance(value, Mapping) or set(value) != _ROUTE_KEYS:
        failures.append(f"route_contract_changed:{route}")
        return
    if (
        value.get("schema_version") != 1
        or value.get("kind") != ROUTE_KIND
        or value.get("route") != route
        or value.get("real_native") is not True
        or value.get("backend") != "metal"
        or value.get("device") != "mps"
        or value.get("selected_training_optimizer_steps") != PILOT_OPTIMIZER_STEPS
        or value.get("selected_training_target_pixels") != PILOT_TARGET_PIXELS
        or value.get("selected_training_rgb_scalar_count") != PILOT_RGB_SCALARS
        or value.get("selected_training_spacetime_sample_count") != 4
        or value.get("training_loss_identifier") != "rgb_mse_mean_v1"
        or value.get("training_loss_contract_sha256") != training_loss_sha256
        or value.get("heldout_frame_count_exercised") != PILOT_FRAME_COUNT
        or value.get("heldout_spatial_track_count_exercised")
        != PILOT_SPATIAL_TRACK_COUNT
        or value.get("heldout_prediction_observation_count")
        != PILOT_FRAME_COUNT * value.get("heldout_spatial_track_count_exercised", 0)
        or value.get("frame_major_parity_track_count") != 1
        or value.get("frame_major_parity_observation_count") != PILOT_FRAME_COUNT
        or value.get("frame_major_cross_time_bitwise_equal") is not True
        or value.get("source_capability_sha256") != capability_sha256
        or not _is_sha256(value.get("native_library_sha256"))
        or not _is_sha256(value.get("worker_source_sha256"))
        or not _generation_valid(value)
    ):
        failures.append(f"route_semantics_changed:{route}")

    timings = value.get("stage_timings_s")
    if (
        not isinstance(timings, Mapping)
        or set(timings) != _STAGE_TIMING_KEYS
        or any(not _finite_nonnegative(item) for item in timings.values())
        or not _finite_nonnegative(timings.get("route_total"))
        or float(timings.get("route_total", -1.0))
        < sum(float(timings.get(key, 0.0)) for key in _STAGE_TIMING_KEYS if key != "route_total")
    ):
        failures.append(f"route_timings_invalid:{route}")

    native_counts = value.get("native_counts")
    compiler_counts = value.get("compiler_counts")
    required_native_counts = {
        "training_native_call_count",
        "training_native_sample_count",
        "spatial_major_native_bundle_count",
        "spatial_major_native_sample_count",
        "frame_major_parity_native_call_count",
        "frame_major_parity_native_sample_count",
    }
    required_compiler_counts = {
        "training_cold_track_compile_count",
        "training_complete_camera_record_validation_count",
        "spatial_major_cold_track_compile_count",
        "spatial_major_complete_camera_record_validation_count",
        "frame_major_parity_cold_track_compile_count",
        "frame_major_parity_complete_camera_record_validation_count",
    }
    for label, counts, required in (
        ("native", native_counts, required_native_counts),
        ("compiler", compiler_counts, required_compiler_counts),
    ):
        if (
            not isinstance(counts, Mapping)
            or not required.issubset(counts)
            or any(
                isinstance(counts[key], bool)
                or not isinstance(counts[key], int)
                or counts[key] < 0
                for key in required
            )
        ):
            failures.append(f"route_{label}_counts_invalid:{route}")
    if (
        isinstance(native_counts, Mapping)
        and required_native_counts.issubset(native_counts)
        and all(
            not isinstance(native_counts[key], bool)
            and isinstance(native_counts[key], int)
            for key in required_native_counts
        )
    ):
        if (
            native_counts["training_native_call_count"] < 1
            or native_counts["training_native_sample_count"] != PILOT_TARGET_PIXELS
            or native_counts["spatial_major_native_bundle_count"] < 1
            or native_counts["spatial_major_native_sample_count"]
            != PILOT_FRAME_COUNT * PILOT_SPATIAL_TRACK_COUNT
            or native_counts["frame_major_parity_native_call_count"]
            != PILOT_FRAME_COUNT
            or native_counts["frame_major_parity_native_sample_count"]
            != PILOT_FRAME_COUNT
        ):
            failures.append(f"route_native_counts_changed:{route}")
    if (
        isinstance(compiler_counts, Mapping)
        and required_compiler_counts.issubset(compiler_counts)
        and all(
            not isinstance(compiler_counts[key], bool)
            and isinstance(compiler_counts[key], int)
            for key in required_compiler_counts
        )
    ):
        if (
            compiler_counts["training_cold_track_compile_count"] < 1
            or compiler_counts["training_complete_camera_record_validation_count"]
            != compiler_counts["training_cold_track_compile_count"]
            * PILOT_FRAME_COUNT
            or compiler_counts["spatial_major_cold_track_compile_count"]
            != PILOT_SPATIAL_TRACK_COUNT
            or compiler_counts[
                "spatial_major_complete_camera_record_validation_count"
            ]
            != PILOT_SPATIAL_TRACK_COUNT * PILOT_FRAME_COUNT
            or compiler_counts["frame_major_parity_cold_track_compile_count"]
            != PILOT_FRAME_COUNT
            or compiler_counts[
                "frame_major_parity_complete_camera_record_validation_count"
            ]
            != PILOT_FRAME_COUNT * PILOT_FRAME_COUNT
        ):
            failures.append(f"route_compiler_counts_changed:{route}")

    projection = value.get("projection")
    raw_track_count = value.get("heldout_spatial_track_count_exercised")
    track_count = int(raw_track_count) if _positive_int(raw_track_count) else 0
    expected_blocks = (
        math.ceil(FULL_HELDOUT_SPATIAL_TRACK_COUNT / track_count)
        if track_count > 0
        else 0
    )
    if (
        not isinstance(projection, Mapping)
        or set(projection) != _PROJECTION_KEYS
        or projection.get("training_step_multiplier") != 300
        or projection.get("full_heldout_spatial_track_count")
        != FULL_HELDOUT_SPATIAL_TRACK_COUNT
        or projection.get("measured_spatial_track_block_count") != track_count
        or projection.get("projected_heldout_block_count") != expected_blocks
        or projection.get("safety_multiplier") != PROJECTION_SAFETY_MULTIPLIER
        or any(
            not _finite_nonnegative(projection.get(key))
            for key in (
                "projected_training_seconds",
                "projected_heldout_seconds",
                "projected_fixed_seconds",
                "projected_total_seconds",
                "projected_full_row_hours",
            )
        )
        or not math.isclose(
            float(projection.get("projected_full_row_hours", -1.0)),
            float(projection.get("projected_total_seconds", -1.0)) / 3600.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or float(projection.get("projected_full_row_hours", float("inf"))) <= 0.0
        or float(projection.get("projected_full_row_hours", float("inf")))
        > maximum_projected_hours
    ):
        failures.append(f"route_projection_invalid:{route}")

    runtime = value.get("runtime_measurements")
    if (
        not isinstance(runtime, Mapping)
        or set(runtime) != _RUNTIME_KEYS
        or any(
            isinstance(runtime.get(key), bool)
            or not isinstance(runtime.get(key), int)
            or int(runtime.get(key)) < 0
            for key in _RUNTIME_KEYS - {"completion_fenced"}
        )
        or runtime.get("completion_fenced") is not True
        or int(runtime.get("worker_process_peak_rss_bytes", 0)) < 1
        or int(runtime.get("parent_watchdog_sample_count", 0)) < 1
        or int(runtime.get("parent_watchdog_timeout_seconds", -1)) != 7200
        or int(runtime.get("parent_watchdog_rss_limit_bytes", -1))
        != 4 * 1024**3
        or not 0
        < int(runtime.get("worker_mps_effective_limit_bytes", 0))
        <= 2 * 1024**3
        or int(runtime.get("worker_mps_peak_driver_bytes", -1))
        < int(runtime.get("worker_mps_baseline_driver_bytes", 0))
        or int(runtime.get("parent_observed_process_group_peak_rss_bytes", 0))
        > int(runtime.get("parent_watchdog_rss_limit_bytes", -1))
        or int(runtime.get("worker_process_peak_rss_bytes", 0))
        > int(runtime.get("parent_watchdog_rss_limit_bytes", -1))
        or int(runtime.get("worker_mps_peak_driver_bytes", 0))
        > int(runtime.get("worker_mps_effective_limit_bytes", -1))
    ):
        failures.append(f"route_runtime_measurements_invalid:{route}")

    for receipt_name in ("pilot_transition_receipt", "spatial_replay_receipt"):
        receipt = value.get(receipt_name)
        if not isinstance(receipt, Mapping) or not _generation_valid(receipt):
            failures.append(f"route_receipt_invalid:{route}:{receipt_name}")
    transition = value.get("pilot_transition_receipt")
    if isinstance(transition, Mapping) and (
        transition.get("kind") != "worldfoam-g4-v2-heldout-pilot-binding-v1"
        or transition.get("route") != route
        or transition.get("optimizer_step") != 1
        or transition.get("training_finalized") is not False
        or transition.get("pilot_only") is not True
        or transition.get("frame_count") != PILOT_FRAME_COUNT
        or transition.get("image_height") != 384
        or transition.get("image_width") != 512
        or transition.get("native_library_sha256")
        != value.get("native_library_sha256")
    ):
        failures.append(f"route_transition_receipt_changed:{route}")
    spatial = value.get("spatial_replay_receipt")
    if isinstance(spatial, Mapping) and (
        spatial.get("kind")
        != "worldfoam-spatial-major-heldout-partial-pilot-v1"
        or spatial.get("pilot_only") is not True
        or spatial.get("paper_evidence") is not False
        or spatial.get("full_coverage") is not False
        or spatial.get("optimizer_step") != 1
        or spatial.get("frame_count") != PILOT_FRAME_COUNT
        or spatial.get("cross_time_render_call_count") != 1
        or spatial.get("cross_time_cold_track_compile_count")
        != PILOT_SPATIAL_TRACK_COUNT
        or spatial.get("cross_time_complete_camera_record_validation_count")
        != PILOT_SPATIAL_TRACK_COUNT * PILOT_FRAME_COUNT
        or spatial.get("cross_time_native_sample_count")
        != PILOT_SPATIAL_TRACK_COUNT * PILOT_FRAME_COUNT
        or spatial.get("cross_time_prediction_target_observation_read_count")
        != PILOT_SPATIAL_TRACK_COUNT * PILOT_FRAME_COUNT
        or spatial.get("cross_time_target_staging_call_count") != 1
        or spatial.get("cross_time_target_staging_observation_count")
        != PILOT_SPATIAL_TRACK_COUNT * PILOT_FRAME_COUNT
        or spatial.get("old_frame_major_render_call_count") != PILOT_FRAME_COUNT
        or spatial.get("old_frame_major_observation_count") != PILOT_FRAME_COUNT
    ):
        failures.append(f"route_spatial_receipt_changed:{route}")


def validate_pilot_receipt(
    payload: Any,
    *,
    config_path: Path = DEFAULT_CONFIG,
    artifact_path: Path | None = None,
    verify_files: bool = True,
) -> list[str]:
    """Return every fail-closed pilot validation failure."""

    failures: list[str] = []
    if not isinstance(payload, Mapping):
        return ["pilot_payload_not_mapping"]
    if set(payload) != _TOP_KEYS:
        return ["pilot_top_level_keys_changed"]
    try:
        config, _base, _base_path = load_selected_ray_contract(config_path)
    except Exception as error:
        return [f"v2_config_invalid:{type(error).__name__}:{error}"]
    config_path = Path(config_path).resolve()
    source_capability_path = (ROOT / str(config["execution"]["source_capability"])).resolve()
    expected_output = (ROOT / str(config["execution"]["pilot_receipt"])).resolve()
    if artifact_path is not None and Path(artifact_path).resolve() != expected_output:
        failures.append("pilot_artifact_path_changed")
    training_loss = dict(config["training_loss"])
    training_loss_sha256 = canonical_sha256(training_loss)
    try:
        capability = _load_json(source_capability_path)
    except Exception as error:
        capability = {}
        failures.append(f"source_capability_invalid:{type(error).__name__}:{error}")
    else:
        try:
            expected_capability = required_source_capability(config_path)
        except Exception as error:
            failures.append(
                f"source_capability_recompute_failed:{type(error).__name__}:{error}"
            )
        else:
            if capability != expected_capability:
                failures.append("source_capability_stale")
    capability_sha256 = capability.get("capability_sha256")
    if (
        payload.get("schema_version") != PILOT_SCHEMA_VERSION
        or payload.get("kind") != PILOT_KIND
        or payload.get("status") != "pass"
        or payload.get("scene") != config["execution"]["pilot_scene"]
        or payload.get("scene") != PILOT_SCENE
        or payload.get("seed") != config["execution"]["pilot_seed"]
        or payload.get("seed") != PILOT_SEED
        or payload.get("v2_config_path") != str(config_path.relative_to(ROOT))
        or payload.get("v2_config_sha256") != file_sha256(config_path)
        or payload.get("source_capability_path")
        != str(source_capability_path.relative_to(ROOT))
        or payload.get("source_capability_sha256") != capability_sha256
        or payload.get("training_loss_contract") != training_loss
        or payload.get("training_loss_contract_sha256") != training_loss_sha256
        or payload.get("public_quality_evidence") is not False
        or payload.get("pilot_only") is not True
        or payload.get("spatial_major_full_temporal_heldout_exercised") is not True
        or not _generation_valid(payload)
    ):
        failures.append("pilot_semantic_contract_changed")

    _validate_host_guard(payload.get("host_guard"), failures)
    _validate_source_binding(
        payload.get("source_binding"),
        verify_files=verify_files,
        failures=failures,
    )
    _validate_native_binding(
        payload.get("native_binding"),
        verify_files=verify_files,
        failures=failures,
    )

    routes = payload.get("routes")
    raw_logs = payload.get("raw_logs")
    if not isinstance(routes, Mapping) or tuple(routes) != REQUIRED_ROUTES:
        failures.append("pilot_route_order_changed")
        routes = {}
    if not isinstance(raw_logs, Mapping) or tuple(raw_logs) != REQUIRED_ROUTES:
        failures.append("pilot_raw_log_order_changed")
        raw_logs = {}
    maximum_hours = float(config["execution"]["maximum_projected_worldfoam_row_hours"])
    for route in REQUIRED_ROUTES:
        report = routes.get(route)
        _validate_route(
            route,
            report,
            capability_sha256=str(capability_sha256),
            training_loss_sha256=training_loss_sha256,
            maximum_projected_hours=maximum_hours,
            failures=failures,
        )
        identity = raw_logs.get(route)
        if not isinstance(identity, Mapping) or set(identity) != _RAW_LOG_KEYS:
            failures.append(f"raw_log_identity_invalid:{route}")
            continue
        try:
            log_path = _repo_file(identity.get("path"), label="pilot raw log")
            raw_log = _load_json(log_path)
            parsed = _parse_worker_report(raw_log)
        except Exception as error:
            failures.append(f"raw_log_invalid:{route}:{type(error).__name__}:{error}")
            continue
        if (
            raw_log.get("route") != route
            or identity.get("worker_returncode") != 0
            or identity.get("bytes") != log_path.stat().st_size
            or identity.get("sha256") != file_sha256(log_path)
            or identity.get("worker_report_generation_digest")
            != parsed.get("generation_digest")
            or parsed != report
        ):
            failures.append(f"raw_log_binding_changed:{route}")
    native = payload.get("native_binding")
    if isinstance(native, Mapping) and isinstance(routes, Mapping):
        digests = [
            routes.get(route, {}).get("native_library_sha256")
            if isinstance(routes.get(route), Mapping)
            else None
            for route in REQUIRED_ROUTES
        ]
        if digests != [native.get("library_sha256")] * len(REQUIRED_ROUTES):
            failures.append("route_native_library_binding_changed")
    return sorted(set(failures))


def verify_pilot_file(
    path: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    path = Path(path).resolve()
    try:
        payload = _load_json(path)
        failures = validate_pilot_receipt(
            payload,
            config_path=config_path,
            artifact_path=path,
            verify_files=True,
        )
    except Exception as error:
        payload = {}
        failures = [f"pilot_read_failed:{type(error).__name__}:{error}"]
    return {
        "schema_version": 1,
        "kind": "worldfoam-g4-v2-selected-ray-real-native-pilot-verification-v1",
        "status": "pass" if not failures else "fail",
        "pilot": str(path),
        "pilot_sha256": file_sha256(path) if path.is_file() else None,
        "failure_count": len(failures),
        "failures": failures,
        "public_quality_evidence": False,
        "pilot_only": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    report = verify_pilot_file(args.pilot, config_path=args.config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "PILOT_KIND",
    "PILOT_SCHEMA_VERSION",
    "RAW_LOG_KIND",
    "REQUIRED_ROUTES",
    "ROUTE_KIND",
    "WORKER_RESULT_PREFIX",
    "main",
    "validate_pilot_receipt",
    "verify_pilot_file",
)
