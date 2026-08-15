#!/usr/bin/env python3
"""Deterministic driver for the paper-scale WorldFoam memory ablation.

This module is the narrow deployment boundary between the fresh-process
producer and the production lazy full-geometry coordinator.  It owns the
fixed S=1024/P=512 procedural input program and the naive-replay preflight;
the producer owns process/MPS measurements, native-extension attestation,
watchdogs, and evidence binding.

No target video is materialized.  Targets are yielded in bounded selected-
pixel chunks, while the physical world, 300-time grid, camera, and track
manifest stay identical for every requested F and repeat.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DRIVER_PROTOCOL = "worldfoam-training-memory-spatial-native-driver-v1"
DRIVER_FUNCTION = "run_worldfoam_training_memory_worker"

# This must remain an AST literal: the parent validates it without importing
# torch or any production coordinator module.
WORLDFOAM_TRAINING_MEMORY_DRIVER_CAPABILITIES = {
    "schema_version": 1,
    "driver_protocol": "worldfoam-training-memory-spatial-native-driver-v1",
    "driver_function": "run_worldfoam_training_memory_worker",
    "supported_backends": ["mps"],
    "supported_worker_kinds": ["primary", "control", "restart"],
    "required_runtime_capabilities": [
        "all_competitor_active_owner_certification",
        "checkpoint_restart_lifecycle",
        "cpu_manual_sgd_mutation",
        "direct_selected_pixel_target_stream",
        "fresh_process_measurements",
        "full_geometry_trainable",
        "fused_union_v2_mode",
        "geometry_optimizer_authorization_receipt",
        "post_certification_compact_device_lowering",
        "production_device_material_gradient_receipt",
        "production_geometry_device_to_host_reduction_receipt",
        "real_native_spatial_block_coordinator",
        "staged_sparse_mode",
        "zero_full_frame_target_materialization",
    ],
    "production_core_module": "paper_kinetic_lazy_full_geometry_step",
    "production_core_callable": "run_paper_kinetic_lazy_native_full_geometry_step",
    "production_adapter_status": "source_complete",
    "sequential_control_adapter_status": "source_complete",
    "compiled_framewise_control_provenance": "paper-kinetic-compiled-framewise-full-geometry-control-v1",
    "required_core_capability_seal": "paper-kinetic-lazy-full-geometry-step-capability-seal-v1",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _install_runtime_paths() -> None:
    import sys

    for path in (
        ROOT / "src" / "train",
        ROOT / "research_experiments" / "world_foam_lane2",
    ):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def endpoint_including_frame_indices(
    *, dataset_frame_count: int, requested_frame_count: int
) -> tuple[int, ...]:
    """Choose an integer, endpoint-including subset without float rounding."""

    dataset_frame_count = _positive_int(
        dataset_frame_count, name="dataset_frame_count"
    )
    requested_frame_count = _positive_int(
        requested_frame_count, name="requested_frame_count"
    )
    if requested_frame_count > dataset_frame_count:
        raise ValueError("requested frames exceed the fixed dataset grid")
    if requested_frame_count == 1:
        return (0,)
    denominator = 2 * (requested_frame_count - 1)
    indices = tuple(
        (2 * index * (dataset_frame_count - 1) + requested_frame_count - 1)
        // denominator
        for index in range(requested_frame_count)
    )
    if (
        indices[0] != 0
        or indices[-1] != dataset_frame_count - 1
        or any(right <= left for left, right in zip(indices, indices[1:]))
    ):
        raise ArithmeticError("endpoint-including frame subset is not strict")
    return indices


def build_fixed_track_manifest(config: Mapping[str, Any]) -> tuple[dict[str, int], ...]:
    track = _mapping(config.get("track_manifest"), name="track_manifest")
    rows = _positive_int(track.get("grid_rows"), name="track_manifest.grid_rows")
    columns = _positive_int(
        track.get("grid_columns"), name="track_manifest.grid_columns"
    )
    image = _mapping(config.get("image"), name="image")
    width = _positive_int(image.get("width"), name="image.width")
    result = tuple(
        {
            "track_id": columns * row_index + column_index,
            "row": 6 + 24 * row_index,
            "column": 8 + 16 * column_index,
            "pixel_id": (6 + 24 * row_index) * width + (8 + 16 * column_index),
        }
        for row_index in range(rows)
        for column_index in range(columns)
    )
    if len(result) != track.get("track_count"):
        raise ValueError("fixed track grid changed its declared count")
    if _sha256(result) != track.get("ordered_manifest_sha256"):
        raise ValueError("fixed track manifest changed its checked-in digest")
    return result


def build_full_physical_time_grid(config: Mapping[str, Any]) -> tuple[float, ...]:
    temporal = _mapping(config.get("temporal_grid"), name="temporal_grid")
    frame_count = _positive_int(
        temporal.get("dataset_frame_count"), name="temporal_grid.dataset_frame_count"
    )
    lower = float(temporal.get("physical_t_min"))
    upper = float(temporal.get("physical_t_max"))
    if not math.isfinite(lower) or not math.isfinite(upper) or not lower < upper:
        raise ValueError("physical time interval is invalid")
    return tuple(
        lower + (upper - lower) * index / (frame_count - 1)
        for index in range(frame_count)
    )


def build_procedural_world_rows(
    config: Mapping[str, Any],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    world = _mapping(config.get("procedural_world"), name="procedural_world")
    material = _mapping(config.get("material"), name="material")
    rows = _positive_int(world.get("rows"), name="procedural_world.rows")
    columns = _positive_int(world.get("columns"), name="procedural_world.columns")
    positions: list[tuple[float, float, float]] = []
    velocities: list[tuple[float, float, float]] = []
    weights: list[tuple[float, float]] = []
    rgba: list[tuple[float, float, float, float]] = []
    for row_index in range(rows):
        for column_index in range(columns):
            row_phase = 2.0 * math.pi * row_index / 31.0
            column_phase = 2.0 * math.pi * column_index / 31.0
            positions.append(
                (
                    -1.55 + 3.10 * column_index / 31.0,
                    -1.55 + 3.10 * row_index / 31.0,
                    2.4 + 0.12 * math.sin(row_phase) * math.cos(column_phase),
                )
            )
            velocities.append(
                (
                    0.025 * math.sin(row_phase),
                    0.025 * math.cos(column_phase),
                    0.01 * math.sin(2.0 * math.pi * (row_index + column_index) / 62.0),
                )
            )
            weights.append(
                (
                    -0.15
                    + 0.30 * ((17 * row_index + 29 * column_index) % 31) / 30.0,
                    0.04
                    * math.sin(2.0 * math.pi * (row_index - column_index) / 31.0),
                )
            )
            rgba.append(
                (
                    0.15 + 0.70 * column_index / 31.0,
                    0.15 + 0.70 * row_index / 31.0,
                    0.20
                    + 0.60 * ((row_index + column_index) % 32) / 31.0,
                    0.04
                    + 0.08 * ((13 * row_index + 7 * column_index) % 32) / 31.0,
                )
            )
    site_count = _positive_int(world.get("site_count"), name="procedural_world.site_count")
    if not all(len(value) == site_count for value in (positions, velocities, weights, rgba)):
        raise ArithmeticError("procedural world changed its site count")
    densities = tuple(row[3] for row in rgba)
    if min(densities) < float(material["minimum_density"]) or max(densities) > float(
        material["maximum_density"]
    ):
        raise ArithmeticError("procedural P0 density escaped its checked-in bounds")
    return tuple(positions), tuple(velocities), tuple(weights), tuple(rgba)


def teacher_rgb(*, row: int, column: int, physical_time: float) -> tuple[float, float, float]:
    u = (column + 0.5) / 512.0
    v = (row + 0.5) / 384.0
    return (
        0.5 + 0.25 * math.sin(2.0 * math.pi * (u + 0.15 * physical_time)),
        0.5
        + 0.25
        * math.sin(2.0 * math.pi * (v - 0.10 * physical_time) + 2.0 * math.pi / 3.0),
        0.5
        + 0.25
        * math.sin(
            2.0 * math.pi * (u + v + 0.05 * physical_time) + 4.0 * math.pi / 3.0
        ),
    )


def iter_direct_selected_pixel_target_chunks(
    config: Mapping[str, Any],
    *,
    requested_frame_count: int,
) -> Iterator[dict[str, Any]]:
    """Yield at most 4096 selected-pixel observations per Python object."""

    temporal = _mapping(config.get("temporal_grid"), name="temporal_grid")
    target = _mapping(config.get("target_source"), name="target_source")
    maximum = _positive_int(
        target.get("maximum_resident_observations"),
        name="target_source.maximum_resident_observations",
    )
    full_times = build_full_physical_time_grid(config)
    selected = endpoint_including_frame_indices(
        dataset_frame_count=_positive_int(
            temporal.get("dataset_frame_count"),
            name="temporal_grid.dataset_frame_count",
        ),
        requested_frame_count=requested_frame_count,
    )
    tracks = build_fixed_track_manifest(config)
    chunk: list[dict[str, Any]] = []
    for frame_index in selected:
        physical_time = full_times[frame_index]
        for record in tracks:
            chunk.append(
                {
                    "dataset_frame_index": frame_index,
                    "physical_time": physical_time,
                    **record,
                    "rgb": teacher_rgb(
                        row=record["row"],
                        column=record["column"],
                        physical_time=physical_time,
                    ),
                }
            )
            if len(chunk) == maximum:
                yield {
                    "observation_count": len(chunk),
                    "records": tuple(chunk),
                    "logical_target_tensor_bytes": len(chunk) * 3 * 4,
                    "full_frame_materialized": False,
                }
                chunk = []
    if chunk:
        yield {
            "observation_count": len(chunk),
            "records": tuple(chunk),
            "logical_target_tensor_bytes": len(chunk) * 3 * 4,
            "full_frame_materialized": False,
        }


def build_training_inputs(
    config: Mapping[str, Any], *, requested_frame_count: int
) -> dict[str, Any]:
    """Build O(S+P+F) CPU tensors and a bounded target-chunk factory."""

    _install_runtime_paths()
    torch = importlib.import_module("torch")
    positions, velocities, weights, rgba = build_procedural_world_rows(config)
    tracks = build_fixed_track_manifest(config)
    full_times = build_full_physical_time_grid(config)
    selected = endpoint_including_frame_indices(
        dataset_frame_count=len(full_times),
        requested_frame_count=requested_frame_count,
    )
    camera = _mapping(config.get("camera"), name="camera")
    material = _mapping(config.get("material"), name="material")
    spatial = _mapping(config.get("spatial_streaming"), name="spatial_streaming")
    track_blocks = tuple(
        (start, min(start + int(spatial["maximum_tracks_per_request"]), len(tracks)))
        for start in range(0, len(tracks), int(spatial["maximum_tracks_per_request"]))
    )
    structure_identity = {
        "world": {
            "positions0": positions,
            "velocities": velocities,
            "weight_coefficients": weights,
            "site_rgba": rgba,
            "generation_seed": config["procedural_world"]["generation_seed"],
        },
        "physical_time_grid": full_times,
        "camera": camera["program"],
        "tracks": tracks,
        "track_blocks": track_blocks,
    }
    return {
        "positions0_f64_cpu": torch.tensor(positions, dtype=torch.float64),
        "velocities_f64_cpu": torch.tensor(velocities, dtype=torch.float64),
        "weight_coefficients_f64_cpu": torch.tensor(weights, dtype=torch.float64),
        "site_rgba_f32_cpu": torch.tensor(rgba, dtype=torch.float32),
        "background_rgb_f32_cpu": torch.tensor(
            material["background_rgb"], dtype=torch.float32
        ),
        "full_physical_time_grid_f64_cpu": torch.tensor(full_times, dtype=torch.float64),
        "selected_frame_indices_i64_cpu": torch.tensor(selected, dtype=torch.int64),
        "selected_physical_times_f64_cpu": torch.tensor(
            tuple(full_times[index] for index in selected), dtype=torch.float64
        ),
        "track_ids_i64_cpu": torch.tensor(
            tuple(record["track_id"] for record in tracks), dtype=torch.int64
        ),
        "pixel_ids_i64_cpu": torch.tensor(
            tuple(record["pixel_id"] for record in tracks), dtype=torch.int64
        ),
        "rows_i64_cpu": torch.tensor(
            tuple(record["row"] for record in tracks), dtype=torch.int64
        ),
        "columns_i64_cpu": torch.tensor(
            tuple(record["column"] for record in tracks), dtype=torch.int64
        ),
        "track_blocks": track_blocks,
        "target_chunk_factory": lambda: iter_direct_selected_pixel_target_chunks(
            config, requested_frame_count=requested_frame_count
        ),
        "target_generation_id": config["target_source"]["teacher_generation_sha256"],
        "track_manifest_sha256": config["track_manifest"]["ordered_manifest_sha256"],
        "camera_program_sha256": camera["program_sha256"],
        "compiled_world_sha256": _sha256(structure_identity["world"]),
        "physical_grid_sha256": _sha256(full_times),
        "camera_grid_sha256": _sha256(
            {"camera": camera["program"], "physical_time_grid": full_times}
        ),
        "spatial_block_manifest_sha256": _sha256(track_blocks),
        "input_program_sha256": _sha256(structure_identity),
        "expected_observation_count": len(tracks) * requested_frame_count,
        "loss_element_count": len(tracks) * requested_frame_count * 3,
        "full_frame_target_materialization_used": False,
        "dataset_is_synthetic": True,
    }


def build_training_structure_receipt(
    config: Mapping[str, Any], *, requested_frame_count: int
) -> dict[str, Any]:
    """Return the JSON-only identity shared by measured and censored rows."""

    positions, velocities, weights, rgba = build_procedural_world_rows(config)
    tracks = build_fixed_track_manifest(config)
    full_times = build_full_physical_time_grid(config)
    camera = _mapping(config.get("camera"), name="camera")
    spatial = _mapping(config.get("spatial_streaming"), name="spatial_streaming")
    maximum_tracks = int(spatial["maximum_tracks_per_request"])
    track_blocks = tuple(
        (start, min(start + maximum_tracks, len(tracks)))
        for start in range(0, len(tracks), maximum_tracks)
    )
    world_identity = {
        "positions0": positions,
        "velocities": velocities,
        "weight_coefficients": weights,
        "site_rgba": rgba,
        "generation_seed": config["procedural_world"]["generation_seed"],
    }
    return {
        "compiled_world_sha256": _sha256(world_identity),
        "physical_grid_sha256": _sha256(full_times),
        "camera_grid_sha256": _sha256(
            {"camera": camera["program"], "physical_time_grid": full_times}
        ),
        "spatial_block_manifest_sha256": _sha256(track_blocks),
        "track_manifest_sha256": config["track_manifest"]["ordered_manifest_sha256"],
        "camera_program_sha256": camera["program_sha256"],
        "target_teacher_sha256": config["target_source"]["teacher_generation_sha256"],
        "expected_observation_count": len(tracks) * requested_frame_count,
        "loss_element_count": len(tracks) * requested_frame_count * 3,
        "dataset_is_procedural_synthetic": True,
        "full_frame_target_materialization_used": False,
    }


def sequential_control_launch_policy(
    config: Mapping[str, Any],
    *,
    requested_frame_count: int,
    f8_calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Declare the non-censoring policy for same-representation replay."""

    working_set_limit = int(config["memory_limits_bytes"]["maximum_mps_working_set"])
    site_count = int(config["procedural_world"]["site_count"])
    fixed_bytes = (
        int(config["state_accounting"]["combined_live_state_bytes_per_site"])
        * site_count
    )
    if f8_calibration is not None:
        raise ValueError("sequential replay does not accept censorship calibration")
    projected = fixed_bytes
    model = {
        "kind": "same-representation-sequential-per-frame-always-launch-v1",
        "requested_frame_count": requested_frame_count,
        "same_representation_and_native_kernels_required": True,
        "sequential_frame_release_required": True,
        "fixed_combined_live_state_bytes": fixed_bytes,
        "logical_peak_lower_bound_bytes": projected,
        "censorship_permitted": False,
        "working_set_limit_bytes": working_set_limit,
    }
    return {
        "performed": True,
        "policy": config["ablation"]["control_memory_censor_policy"],
        "model_sha256": _sha256(model),
        "model": model,
        "projected_peak_bytes": projected,
        "working_set_limit_bytes": working_set_limit,
        "decision": "launch",
        "censor_reason": None,
    }


def _load_production_adapter() -> Any:
    """Load the high-level adapter only after worker memory baselines exist."""

    _install_runtime_paths()
    core = importlib.import_module(
        WORLDFOAM_TRAINING_MEMORY_DRIVER_CAPABILITIES["production_core_module"]
    )
    core_callable = getattr(
        core,
        WORLDFOAM_TRAINING_MEMORY_DRIVER_CAPABILITIES["production_core_callable"],
        None,
    )
    if not callable(core_callable):
        raise RuntimeError(
            "production lazy full-geometry callable is not source-complete"
        )
    adapter = getattr(core, "run_worldfoam_training_memory_ablation_adapter", None)
    if not callable(adapter):
        raise RuntimeError(
            "production lazy full-geometry core exists, but the ablation adapter "
            "and combined-state bridge are not source-complete"
        )
    return adapter


def run_worldfoam_training_memory_worker(context: Mapping[str, Any]) -> Mapping[str, Any]:
    """Run one primary/control/restart transaction through production code."""

    context = _mapping(context, name="worker context")
    if context.get("backend") != "mps" or context.get("require_real_native") is not True:
        raise ValueError("training-memory driver requires producer-attested real MPS native ops")
    config = _mapping(context.get("config"), name="worker config")
    frame_count = _positive_int(context.get("frame_count"), name="frame_count")
    repeat_index = _nonnegative_int(context.get("repeat_index"), name="repeat_index")
    worker_kind = str(context.get("worker_kind", ""))
    if worker_kind not in {"primary", "control", "restart"}:
        raise ValueError("worker_kind must be primary, control, or restart")
    mode = str(context.get("mode", ""))
    allowed_modes = {
        config["ablation"]["staged_mode"],
        config["ablation"]["fused_mode"],
        config["ablation"]["control_mode"],
    }
    if mode not in allowed_modes:
        raise ValueError("worker mode is outside the checked-in ablation")

    preflight = None
    if worker_kind == "control":
        if mode != config["ablation"]["control_mode"]:
            raise ValueError("control worker requires the naive replay mode")
        preflight = sequential_control_launch_policy(
            config,
            requested_frame_count=frame_count,
            f8_calibration=context.get("control_f8_calibration"),
        )

    inputs = build_training_inputs(config, requested_frame_count=frame_count)
    adapter = _load_production_adapter()
    result = adapter(
        {
            **dict(context),
            "repeat_index": repeat_index,
            "preflight": preflight,
            "inputs": inputs,
            "fresh_inputs_factory": (
                lambda: build_training_inputs(
                    config, requested_frame_count=frame_count
                )
            ),
        }
    )
    result = _mapping(result, name="production full-geometry adapter result")
    mutable_result = dict(result)
    if mutable_result.pop("native_ops_used", None) is not context.get("native_ops"):
        raise ValueError("production adapter did not preserve native_ops identity")
    mutable_result["native_ops_identity_verified"] = True
    return mutable_result


__all__ = [
    "DRIVER_FUNCTION",
    "DRIVER_PROTOCOL",
    "WORLDFOAM_TRAINING_MEMORY_DRIVER_CAPABILITIES",
    "build_fixed_track_manifest",
    "build_full_physical_time_grid",
    "build_procedural_world_rows",
    "build_training_inputs",
    "build_training_structure_receipt",
    "endpoint_including_frame_indices",
    "iter_direct_selected_pixel_target_chunks",
    "sequential_control_launch_policy",
    "run_worldfoam_training_memory_worker",
    "teacher_rgb",
]
