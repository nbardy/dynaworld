from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from config_utils import load_config_file
from paper_training_protocol import (
    PAPER_DATASET_BUNDLE_SCHEMA_VERSION,
    PAPER_RUNTIME_SCHEMA_VERSION,
    paper_evaluator_contract,
    resolve_paper_training_protocol,
)
from research_experiments.paper_runner_suite.run_frozen_world_replay_compiled import (
    CANONICAL_FRAME_COUNTS,
    DEFAULT_FRAME_COUNTS,
    DEFAULT_TIMING_REPEATS,
    DEFAULT_TIMING_WARMUPS,
    LIVE_RESOURCE_THRESHOLDS,
    MAX_FRAME_COUNT_REQUESTS,
    MAX_TIMING_REPEATS,
    MAX_TIMING_WARMUPS,
    TIMING_METRIC_KEYS,
    build_command,
    failure_identity,
    full_interval_frame_indices,
    parse_frame_counts,
    require_live_resources,
    resolve_frame_counts,
    sequence_sha256,
    sweep_publication_eligible,
    timing_summary,
    validate_timing_controls,
    validate_execution_identity,
    validate_report_identity,
)
from research_experiments.paper_runner_suite.frozen_atlas_storage import (
    LOGICAL_PAYLOAD_DEFINITION,
    REPLAY_STORAGE_REASON,
    RETAINED_STORAGE_DEFINITION,
    ROUTE_MEMORY_DEFINITION,
    ROUTE_MEMORY_MEASUREMENT_SOURCE,
    write_retained_storage_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
SMOKE_PROTOCOL = (
    ROOT
    / "src"
    / "train_configs"
    / "paper_protocols"
    / "coffee_martini_protocol_smoke_2step.jsonc"
)


def _protocol():
    return resolve_paper_training_protocol(load_config_file(SMOKE_PROTOCOL))


def _write_checkpoint(
    path: Path,
    protocol,
    *,
    value: float = 1.0,
) -> str:
    tube_count = protocol.final_stage.primitive_count
    parameter_names = [
        "x0",
        "velocity",
        "raw_precision_xy",
        "raw_lambda_t",
        "raw_opacity",
        "raw_color",
        "t0",
    ]
    metadata = {
        "representation": "legacy_tube",
        "frame_count": protocol.dataset.frame_count,
        "active_tube_count": tube_count,
        "tube_count": tube_count,
        "alpha_mode": "peak_splat",
        "amplitude_convention": "fiber_integrated",
        "min_precision_xy": 1.0e-4,
        "min_lambda_t": 1.0e-4,
        "parameter_names": parameter_names,
    }
    state = {
        "x0": torch.full((tube_count, 3), value, dtype=torch.float32),
        "velocity": torch.zeros((tube_count, 3), dtype=torch.float32),
        "raw_precision_xy": torch.ones(
            (tube_count, 2),
            dtype=torch.float32,
        ),
        "raw_lambda_t": torch.ones((tube_count,), dtype=torch.float32),
        "raw_opacity": torch.zeros((tube_count,), dtype=torch.float32),
        "raw_color": torch.zeros((tube_count, 3), dtype=torch.float32),
        "t0": torch.zeros((tube_count,), dtype=torch.float32),
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for name in sorted(state):
        tensor = state[name].contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(json.dumps(list(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes(order="C"))
    world_state_sha = digest.hexdigest()
    torch.save(
        {
            "schema_version": 1,
            **metadata,
            "world_state_sha256": world_state_sha,
            "state_dict": state,
        },
        path,
    )
    return world_state_sha


def _native_identity(path: Path) -> dict:
    source_payload = {
        "schema_version": 1,
        "root": str(path.parent.resolve()),
        "file_count": 1,
        "files": [
            {
                "path": str(path.resolve()),
                "relative_path": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ],
    }
    return {
        "module": "test.star_uvt._C",
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "source_tree_sha256": "d" * 64,
        "source_file_count": 1,
        "runtime_source_tree": {
            **source_payload,
            "sha256": hashlib.sha256(
                json.dumps(
                    source_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        },
    }


def _hashed_contract(schema_version: int, **values) -> dict:
    payload = {"schema_version": schema_version, **values}
    return {
        **payload,
        "sha256": hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def _timing_benchmark(
    frame_count: int,
    *,
    warmups: int = DEFAULT_TIMING_WARMUPS,
    repeats: int = DEFAULT_TIMING_REPEATS,
) -> dict:
    samples = {key: [] for key in TIMING_METRIC_KEYS}
    for repeat in range(repeats):
        if warmups == 0 and repeats == 1:
            replay_forward = 1.0
            replay_backward = 1.0
            compiled_compile = 0.1
            compiled_forward = 0.2
            compiled_backward = 0.3
        else:
            replay_forward = 1.0 + 0.1 * repeat
            replay_backward = 2.0 + 0.1 * repeat
            compiled_compile = 0.5 + 0.01 * repeat
            compiled_forward = 0.2 + 0.01 * repeat
            compiled_backward = 0.3 + 0.01 * repeat
        values = {
            "replay_total_forward": replay_forward,
            "replay_total_backward": replay_backward,
            "replay_total_forward_backward": (
                replay_forward + replay_backward
            ),
            "replay_per_frame_forward": replay_forward / frame_count,
            "replay_per_frame_backward": replay_backward / frame_count,
            "compiled_atlas_compile": compiled_compile,
            "compiled_total_forward": compiled_forward,
            "compiled_total_backward": compiled_backward,
            "compiled_total_forward_backward": (
                compiled_forward + compiled_backward
            ),
            "compiled_compile_plus_forward_backward": (
                compiled_compile + compiled_forward + compiled_backward
            ),
            "compiled_per_frame_forward": compiled_forward / frame_count,
            "compiled_per_frame_backward": compiled_backward / frame_count,
        }
        for key, value in values.items():
            samples[key].append(value)
    publication_ready = warmups >= 1 and repeats >= 3
    single_shot = warmups == 0 and repeats == 1
    return {
        "schema_version": 1,
        "status": "complete",
        "label": (
            "single_shot_correctness_timing"
            if single_shot
            else (
                "warmed_repeated_wall_timing_v1"
                if publication_ready
                else "diagnostic_repeated_wall_timing_v1"
            )
        ),
        "publication_ready": publication_ready,
        "warmups": warmups,
        "repeats": repeats,
        "measurement_source": (
            "backward_compatible_correctness_pass"
            if single_shot
            else "independent_alternating_paired_trials"
        ),
        "timing_definition": (
            "device-synchronized perf_counter segments; forward includes "
            "target transfer; compile includes world projection; summed totals "
            "exclude inter-segment cleanup and optimizer work"
        ),
        "route_order": (
            "correctness_pass_replay_then_compiled"
            if single_shot
            else "alternating_paired_replay_compiled_v1"
        ),
        "device_synchronized_at_boundaries": True,
        "compiled_evaluator_uses_chunk_slices": True,
        "forward_includes_cpu_target_to_device_transfer": True,
        "compiled_atlas_compile_includes_world_projection": True,
        "backward_excludes_optimizer": True,
        "resident_chunk_frames": min(2, frame_count),
        "correctness_and_slice_parity_time_excluded": not single_shot,
        "samples_s": samples,
        "summary_s": {
            key: timing_summary(values) for key, values in samples.items()
        },
    }


def _fixture_storage_artifact(
    checkpoint: Path,
    *,
    frame_count: int,
) -> dict[str, object]:
    trace_count = 2
    cell_count = 2
    return write_retained_storage_artifact(
        checkpoint.parent
        / f"{checkpoint.stem}_frame_{frame_count:04d}.world_tubes_atlas",
        frame_count=frame_count,
        trace_count=trace_count,
        cell_count=cell_count,
        tensors={
            "coeffs": ("float32", (trace_count, 3, 2), bytes(48)),
            "opacity": ("float32", (trace_count,), bytes(8)),
            "color": ("float32", (trace_count, 3), bytes(24)),
            "opacity_time_coeffs": None,
            "spatial_precision_uv": None,
            "depth_affine_uv": None,
        },
        topology={
            "source_window_indices": [0, 1],
            "source_primitive_ids": [0, 1],
            "active_start": [0, 0],
            "active_stop": [frame_count, frame_count],
            "cells": [
                {
                    "tile_u": index,
                    "tile_v": 0,
                    "start": 0,
                    "stop": frame_count,
                    "primitive_ids": [index],
                    "ordered_primitive_ids": [index],
                    "depth_intervals": [[1.0 + index, 2.0 + index]],
                    "fallback": False,
                    "fallback_reasons": [],
                }
                for index in range(cell_count)
            ],
        },
    )


def _fixture_route_memory(
    *,
    frame_count: int,
    resident_chunk_frames: int,
) -> dict[str, object]:
    def route_record(route: str, names: list[str]) -> dict[str, object]:
        phases = [
            {
                "name": name,
                "sampled_peak_current_allocated_bytes": 1_000 + index,
                "sampled_peak_driver_allocated_bytes": 2_000 + index,
                "memory_sample_count": 2,
            }
            for index, name in enumerate(names)
        ]
        return {
            "schema_version": 1,
            "route": route,
            "device_type": "mps",
            "route_scoped": True,
            "baseline_current_allocated_bytes": 100,
            "baseline_driver_allocated_bytes": 200,
            "sampled_peak_current_allocated_bytes": max(
                phase["sampled_peak_current_allocated_bytes"]
                for phase in phases
            ),
            "sampled_peak_driver_allocated_bytes": max(
                phase["sampled_peak_driver_allocated_bytes"]
                for phase in phases
            ),
            "peak_increment_current_allocated_bytes": max(
                phase["sampled_peak_current_allocated_bytes"]
                for phase in phases
            )
            - 100,
            "peak_increment_driver_allocated_bytes": max(
                phase["sampled_peak_driver_allocated_bytes"]
                for phase in phases
            )
            - 200,
            "memory_sample_count": sum(
                phase["memory_sample_count"] for phase in phases
            ),
            "phase_count": len(phases),
            "phases": phases,
            "measurement_claim_eligible": True,
        }

    compiled_names = ["atlas_compile"]
    for start in range(0, frame_count, resident_chunk_frames):
        stop = min(frame_count, start + resident_chunk_frames)
        compiled_names.extend(
            (
                f"chunk_{start:04d}_{stop:04d}_forward",
                f"chunk_{start:04d}_{stop:04d}_backward",
            )
        )
    return {
        "schema_version": 1,
        "definition": ROUTE_MEMORY_DEFINITION,
        "measurement_source": ROUTE_MEMORY_MEASUREMENT_SOURCE,
        "sampler_interval_ms": 5.0,
        "compiled_parity_replay_excluded": True,
        "replay": route_record("replay", ["correctness_forward_backward"]),
        "compiled": route_record("compiled", compiled_names),
        "publication_claim_eligible": True,
    }


def _frozen_report(
    checkpoint: Path,
    protocol,
    *,
    frame_count: int = 4,
    world_state_sha: str | None = None,
    slice_parity: bool = False,
    timing_warmups: int = DEFAULT_TIMING_WARMUPS,
    timing_repeats: int = DEFAULT_TIMING_REPEATS,
) -> dict:
    if world_state_sha is None:
        world_state_sha = str(
            torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=True,
            )["world_state_sha256"]
        )
    primitive_count = protocol.final_stage.primitive_count
    frame_indices = full_interval_frame_indices(
        protocol.dataset.frame_count,
        frame_count,
    )
    centered_frame_times = tuple(
        float(frame) - 0.5 * float(protocol.dataset.frame_count - 1)
        for frame in frame_indices
    )
    contract_hashes = {
        "target_frames_sha256": "a" * 64,
        "camera_program_sha256": "b" * 64,
        "frame_indices_sha256": sequence_sha256(frame_indices),
        "centered_frame_times_sha256": sequence_sha256(
            centered_frame_times
        ),
        "evaluation_contract_sha256": "e" * 64,
    }
    if slice_parity:
        parity_acceptance = {
            "image_max_abs_error": 1.0e-5,
            "loss_absolute_delta": 1.0e-5,
            "gradient_global_normalized_l2_error": 1.0e-5,
            "gradient_max_parameter_normalized_l2_error": 1.0e-5,
            "min_world_vjp_l2_norm": 1.0e-12,
        }
        selected_time_slice_parity = {
            "schema_version": 1,
            "status": "complete",
            "accepted": True,
            "timing_claim_eligible": False,
            "frame_count": frame_count,
            "full_dataset_frame_count": protocol.dataset.frame_count,
            "frame_indices": list(frame_indices),
            "centered_frame_times": list(centered_frame_times),
            "time_steps": [
                centered_frame_times[index + 1]
                - centered_frame_times[index]
                for index in range(frame_count - 1)
            ],
            "slice_chunk_frames": 1,
            "slice_count": frame_count,
            "parent_atlas_trace_count": 2,
            "parent_atlas_cell_count": 2,
            "cumulative_sliced_trace_count": 4,
            "cumulative_sliced_cell_count": 4,
            "contract_hashes": contract_hashes,
            "world_state": {
                "before_sha256": world_state_sha,
                "after_full_atlas_sha256": world_state_sha,
                "after_sliced_atlas_sha256": world_state_sha,
                "unchanged": True,
            },
            "loss": {
                "full_atlas": 0.1,
                "chunk_sliced": 0.1,
                "absolute_delta": 0.0,
            },
            "image": {"max_abs_error": 0.0, "mean_abs_error": 0.0},
            "gradient": {
                "global_normalized_l2_error": 0.0,
                "cosine_similarity": 1.0,
                "full_atlas_l2_norm": 1.0,
                "chunk_sliced_l2_norm": 1.0,
                "parameter_tensor_count": 7,
                "full_atlas_gradient_tensor_count": 7,
                "chunk_sliced_gradient_tensor_count": 7,
                "gradient_coverage_matches": True,
                "max_parameter_normalized_l2_error": 0.0,
                "per_parameter_normalized_l2_error": {
                    name: 0.0
                    for name in (
                        "x0",
                        "velocity",
                        "raw_precision_xy",
                        "raw_lambda_t",
                        "raw_opacity",
                        "raw_color",
                        "t0",
                    )
                },
                "full_atlas_gradient_parameters": [
                    "x0",
                    "velocity",
                    "raw_precision_xy",
                    "raw_lambda_t",
                    "raw_opacity",
                    "raw_color",
                    "t0",
                ],
                "chunk_sliced_gradient_parameters": [
                    "x0",
                    "velocity",
                    "raw_precision_xy",
                    "raw_lambda_t",
                    "raw_opacity",
                    "raw_color",
                    "t0",
                ],
            },
            "acceptance": parity_acceptance,
            "checks": {
                "non_unit_selected_times": True,
                "same_parent_atlas": True,
                "world_state_unchanged": True,
                "image_matches": True,
                "loss_matches": True,
                "world_vjp_matches": True,
                "world_vjp_per_parameter_matches": True,
                "world_vjp_nonzero": True,
                "world_vjp_coverage_matches": True,
            },
        }
    else:
        selected_time_slice_parity = {
            "schema_version": 1,
            "status": "not_run",
            "accepted": False,
            "reason": "bounded proof runs on one sweep row",
            "timing_claim_eligible": False,
        }
    storage_artifact = _fixture_storage_artifact(
        checkpoint,
        frame_count=frame_count,
    )
    retained_storage = {
        "schema_version": 1,
        "definition": RETAINED_STORAGE_DEFINITION,
        "shared_checkpoint_bytes": checkpoint.stat().st_size,
        "shared_checkpoint_excluded_from_route_totals": True,
        "replay": {
            "route": "replay",
            "serialized_retained_evaluator_bytes": 0,
            "topology_applicable": False,
            "storage_claim_eligible": True,
            "reason": REPLAY_STORAGE_REASON,
        },
        "compiled": {
            "route": "compiled",
            "serialized_retained_evaluator_bytes": storage_artifact["bytes"],
            "tensor_payload_bytes": storage_artifact["tensor_payload_bytes"],
            "topology_and_container_bytes": storage_artifact[
                "topology_and_container_bytes"
            ],
            "topology_bytes_included": True,
            "artifact": storage_artifact,
            "storage_claim_eligible": True,
        },
        "topology_bytes_included": True,
        "storage_claim_eligible": True,
        "publication_claim_eligible": True,
    }
    resident_chunk_frames = min(2, frame_count)
    return {
        "schema_version": 2,
        "status": "complete",
        "accepted": True,
        "frame_count": frame_count,
        "full_dataset_frame_count": protocol.dataset.frame_count,
        "frame_indices": list(frame_indices),
        "centered_frame_times": list(centered_frame_times),
        "temporal_sampling": "ordered_full_interval_integer_lattice_v1",
        "heldout_camera": "cam06",
        "image_size": [
            protocol.final_stage.image_size.height,
            protocol.final_stage.image_size.width,
        ],
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "bytes": checkpoint.stat().st_size,
            "parameter_tensor_count": 7,
            "parameter_names": [
                "x0",
                "velocity",
                "raw_precision_xy",
                "raw_lambda_t",
                "raw_opacity",
                "raw_color",
                "t0",
            ],
            "world_state_sha256": world_state_sha,
            "representation": "legacy_tube",
            "frame_count": protocol.dataset.frame_count,
            "active_tube_count": primitive_count,
            "tube_count": primitive_count,
            "alpha_mode": "peak_splat",
            "amplitude_convention": "fiber_integrated",
            "min_precision_xy": 1.0e-4,
            "min_lambda_t": 1.0e-4,
        },
        "world_state": {
            "checkpoint_sha256": world_state_sha,
            "before_routes_sha256": world_state_sha,
            "after_replay_sha256": world_state_sha,
            "after_compiled_sha256": world_state_sha,
            "matches_checkpoint": True,
        },
        "loss": {"replay": 0.1, "compiled": 0.1, "absolute_delta": 0.0},
        "image": {"max_abs_error": 0.0, "mean_abs_error": 0.0},
        "gradient": {
            "global_normalized_l2_error": 0.0,
            "cosine_similarity": 1.0,
            "replay_l2_norm": 1.0,
            "compiled_l2_norm": 1.0,
            "max_parameter_normalized_l2_error": 0.0,
            "parameter_tensor_count": 7,
            "replay_gradient_tensor_count": 7,
            "compiled_gradient_tensor_count": 7,
            "gradient_coverage_matches": True,
            "replay_gradient_parameters": [
                "x0",
                "velocity",
                "raw_precision_xy",
                "raw_lambda_t",
                "raw_opacity",
                "raw_color",
                "t0",
            ],
            "compiled_gradient_parameters": [
                "x0",
                "velocity",
                "raw_precision_xy",
                "raw_lambda_t",
                "raw_opacity",
                "raw_color",
                "t0",
            ],
        },
        "timing_s": {
            "replay_total_forward": 1.0,
            "replay_total_backward": 1.0,
            "replay_per_frame_forward": 1.0 / frame_count,
            "replay_per_frame_backward": 1.0 / frame_count,
            "compiled_atlas_compile": 0.1,
            "compiled_total_forward": 0.2,
            "compiled_total_backward": 0.3,
            "compiled_per_frame_forward": 0.2 / frame_count,
            "compiled_per_frame_backward": 0.3 / frame_count,
            "parity_replay_total_forward": 0.4,
        },
        "timing_benchmark": _timing_benchmark(
            frame_count,
            warmups=timing_warmups,
            repeats=timing_repeats,
        ),
        "selected_time_slice_parity": selected_time_slice_parity,
        "payload_bytes": {
            "schema_version": 1,
            "metric_kind": "logical_work_volume_proxy",
            "definition": LOGICAL_PAYLOAD_DEFINITION,
            "topology_bytes_included": False,
            "storage_claim_eligible": False,
            "publication_claim_eligible": False,
            "replay_cumulative_logical_tensor_bytes": 400,
            "compiled_trace_table_logical_tensor_bytes": 100,
            "compiled_to_replay_logical_volume_ratio": 0.25,
        },
        "retained_storage_bytes": retained_storage,
        "route_memory": _fixture_route_memory(
            frame_count=frame_count,
            resident_chunk_frames=resident_chunk_frames,
        ),
        "atlas": {
            "trace_count": 2,
            "cell_count": 2,
            "interval_trace_entries": 4,
            "dense_trace_samples": 8,
            "fallback_cells": 0,
            "total_tile_samples": 4,
            "fallback_tile_samples": 0,
            "fallback_fraction": 0.0,
        },
        "contract": {
            "same_checkpoint": True,
            "same_heldout_camera": True,
            "same_target_frames": True,
            "same_loss": True,
            "same_precision": True,
            "same_alpha_mode": True,
            "bounded_device_frame_residency": True,
            "host_target_storage": "eager_cpu_selected_frames",
            "resident_chunk_frames": resident_chunk_frames,
            "timing_excludes_parity_replay": True,
        },
        "contract_hashes": contract_hashes,
        "acceptance": {
            "image_max_abs_error": 1.0e-5,
            "loss_absolute_delta": 1.0e-5,
            "gradient_global_normalized_l2_error": 1.0e-5,
            "gradient_max_parameter_normalized_l2_error": 1.0e-5,
            "min_world_vjp_l2_norm": 1.0e-12,
            "fallback_fraction": 0.2,
        },
        "checks": {
            "checkpoint_matches": True,
            "image_matches": True,
            "loss_matches": True,
            "world_vjp_matches": True,
            "world_vjp_per_parameter_matches": True,
            "world_vjp_nonzero": True,
            "world_vjp_coverage_matches": True,
            "fallback_within_budget": True,
        },
    }


def _frozen_sweep(
    checkpoint: Path,
    protocol,
    *,
    max_frames: int = 4,
    frame_counts: tuple[int, ...] | None = None,
    row_checkpoints: dict[int, tuple[Path, str]] | None = None,
    timing_warmups: int = DEFAULT_TIMING_WARMUPS,
    timing_repeats: int = DEFAULT_TIMING_REPEATS,
) -> tuple[dict, dict]:
    requested = tuple(frame_counts or ())
    resolved = []
    for count in (max_frames, *requested):
        frame_count = (
            protocol.dataset.frame_count
            if count == 0
            else min(count, protocol.dataset.frame_count)
        )
        if frame_count not in resolved:
            resolved.append(frame_count)
    resolved.sort()
    slice_parity_frame_count = next(
        (
            frame_count
            for frame_count in resolved
            if frame_count < protocol.dataset.frame_count
            and any(
                right - left != 1
                for left, right in zip(
                    full_interval_frame_indices(
                        protocol.dataset.frame_count,
                        frame_count,
                    ),
                    full_interval_frame_indices(
                        protocol.dataset.frame_count,
                        frame_count,
                    )[1:],
                )
            )
        ),
        None,
    )
    rows = []
    for frame_count in resolved:
        row_override = (row_checkpoints or {}).get(frame_count)
        if row_override is None:
            row_checkpoint = checkpoint
            world_state_sha = None
        else:
            row_checkpoint, world_state_sha = row_override
        rows.append(
            _frozen_report(
                row_checkpoint,
                protocol,
                frame_count=frame_count,
                world_state_sha=world_state_sha,
                slice_parity=frame_count == slice_parity_frame_count,
                timing_warmups=timing_warmups,
                timing_repeats=timing_repeats,
            )
        )
    primary_frames = (
        protocol.dataset.frame_count
        if max_frames == 0
        else min(max_frames, protocol.dataset.frame_count)
    )
    primary = next(row for row in rows if row["frame_count"] == primary_frames)
    shared_checkpoint = rows[0]["checkpoint"]
    all_rows_accepted = all(row["accepted"] is True for row in rows)
    all_rows_timing_publication_ready = all(
        row["timing_benchmark"]["publication_ready"] is True
        for row in rows
    )
    all_rows_storage_publication_ready = all(
        row["retained_storage_bytes"]["publication_claim_eligible"] is True
        for row in rows
    )
    all_rows_route_memory_publication_ready = all(
        row["route_memory"]["publication_claim_eligible"] is True
        for row in rows
    )
    slice_parity_accepted = any(
        row["selected_time_slice_parity"]["accepted"] is True
        for row in rows
    )
    publication_eligible = (
        all_rows_accepted
        and protocol.dataset.frame_count >= 128
        and (0 in requested or protocol.dataset.frame_count in requested)
        and {4, 8, 16, 32, 64, 128}.issubset(requested)
        and all_rows_timing_publication_ready
        and all_rows_storage_publication_ready
        and all_rows_route_memory_publication_ready
        and slice_parity_accepted
    )
    return primary, {
        "schema_version": 1,
        "status": "complete",
        "timing_label": "single_shot_correctness_timing",
        "timing_repeats": 1,
        "timing_warmups": 0,
        "timing_benchmark_label": (
            "warmed_repeated_wall_timing_v1"
            if all_rows_timing_publication_ready
            else rows[0]["timing_benchmark"]["label"]
        ),
        "timing_benchmark_repeats": timing_repeats,
        "timing_benchmark_warmups": timing_warmups,
        "all_rows_timing_publication_ready": (
            all_rows_timing_publication_ready
        ),
        "all_rows_storage_publication_ready": (
            all_rows_storage_publication_ready
        ),
        "all_rows_route_memory_publication_ready": (
            all_rows_route_memory_publication_ready
        ),
        "temporal_sampling": "ordered_full_interval_integer_lattice_v1",
        "requested_frame_counts": list(requested),
        "primary_requested_frame_count": max_frames,
        "primary_resolved_frame_count": primary_frames,
        "resolved_frame_counts": resolved,
        "full_dataset_frame_count": protocol.dataset.frame_count,
        "shared_checkpoint": shared_checkpoint,
        "shared_checkpoint_file_sha256": shared_checkpoint["sha256"],
        "shared_world_state_sha256": shared_checkpoint["world_state_sha256"],
        "checkpoint_shared_across_rows": True,
        "world_state_shared_across_rows": True,
        "selected_time_slice_parity_frame_count": (
            slice_parity_frame_count
        ),
        "selected_time_slice_parity_accepted": slice_parity_accepted,
        "all_rows_accepted": all_rows_accepted,
        "publication_eligible": publication_eligible,
        "rows": rows,
    }


def _world_tubes_lane(
    checkpoint: Path,
    protocol,
    *,
    max_frames: int = 4,
    frame_counts: tuple[int, ...] | None = None,
    row_checkpoints: dict[int, tuple[Path, str]] | None = None,
    timing_warmups: int = DEFAULT_TIMING_WARMUPS,
    timing_repeats: int = DEFAULT_TIMING_REPEATS,
) -> dict:
    primary, sweep = _frozen_sweep(
        checkpoint,
        protocol,
        max_frames=max_frames,
        frame_counts=frame_counts,
        row_checkpoints=row_checkpoints,
        timing_warmups=timing_warmups,
        timing_repeats=timing_repeats,
    )
    return {
        "tube_count": protocol.final_stage.primitive_count,
        "steps": protocol.steps,
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
            "enabled": True,
            "sampling": {
                "mode": "spacetime_epoch",
                "same_time_count": protocol.same_time_count,
                "local_time_count": protocol.local_time_count,
                "local_time_radius": protocol.local_time_radius,
            },
            "sample_schedule": {
                "schema_version": 1,
                "algorithm": "spacetime_epoch_v1",
                "sampler_seed": 17 + protocol.sampler_seed_offset,
                "record_count": protocol.steps,
                "sha256": "a" * 64,
            },
            "stages": [stage.as_dict() for stage in protocol.stages],
            "cost": {
                "optimizer_steps": protocol.steps,
                "target_frames": protocol.target_frame_budget,
                "rasterized_frames": protocol.target_frame_budget,
                "target_pixels": protocol.target_pixel_budget,
                "rasterized_pixels": protocol.target_pixel_budget,
                "parameter_count": 100,
                "trainable_parameter_count": 100,
                "parameter_bytes": 400,
                "optimizer_state_bytes": 800,
                "serialized_checkpoint_bytes": checkpoint.stat().st_size,
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
                "backward_calls": protocol.steps,
                "optimizer_s": 0.1,
                "optimizer_calls": protocol.steps,
                "train_wall_s": 1.0,
            },
        },
        "metal_stats": {
            "rows": [
                {
                    "stats": {
                        "projected_trace_count": protocol.final_stage.primitive_count,
                        "uvt_tile_tube_pairs": 20,
                        "summed_per_frame_tile_splat_pairs": 40,
                        "effective_pair_ratio_after_unstable_fallback": 0.5,
                        "unstable_tile_fraction": 0.0,
                        "overflow_tile_count": 0,
                        "metal_buffer_memory": 8_192,
                    }
                }
            ]
        },
        "frozen_world_replay_compiled": primary,
        "frozen_world_replay_compiled_sweep": sweep,
    }


def _comparison_report(
    checkpoint: Path,
    protocol,
    native: dict,
    *,
    max_frames: int = 4,
    frame_counts: tuple[int, ...] | None = None,
    row_checkpoints: dict[int, tuple[Path, str]] | None = None,
    timing_warmups: int = DEFAULT_TIMING_WARMUPS,
    timing_repeats: int = DEFAULT_TIMING_REPEATS,
) -> dict:
    return {
        "meta": {
            "seed": 17,
            "frame_count": protocol.dataset.frame_count,
            "train_cameras": ["cam04", "cam09"],
            "heldout_cameras": ["cam06"],
            "only_lane": "world_tubes",
            "frozen_world_replay_compiled": True,
            "frozen_world_max_frames": max_frames,
            "frozen_world_temporal_sampling": (
                "ordered_full_interval_integer_lattice_v1"
            ),
            "frozen_world_frame_counts": (
                None if frame_counts is None else list(frame_counts)
            ),
            "frozen_world_timing_warmups": timing_warmups,
            "frozen_world_timing_repeats": timing_repeats,
            "uvt_world_representation": "legacy_tube",
            "uvt_alpha_mode": "peak_splat",
            "uvt_render_backend": "metal_tile",
            "uvt_camera_projection": "dataset",
            "uvt_camera_sequence_mode": "static_view",
            "paper_dataset_bundle": _hashed_contract(
                PAPER_DATASET_BUNDLE_SCHEMA_VERSION,
                dataset="fixture",
            ),
            "paper_evaluator": paper_evaluator_contract(),
            "paper_runtime": _hashed_contract(
                PAPER_RUNTIME_SCHEMA_VERSION,
                runtime="fixture",
            ),
            "route_native_extension": native,
            "star_uvt_native_extension": native,
        },
        "star_uvt": _world_tubes_lane(
            checkpoint,
            protocol,
            max_frames=max_frames,
            frame_counts=frame_counts,
            row_checkpoints=row_checkpoints,
            timing_warmups=timing_warmups,
            timing_repeats=timing_repeats,
        ),
    }


def test_frozen_world_command_runs_only_the_world_tubes_lane(
    tmp_path: Path,
) -> None:
    protocol = _protocol()

    command = build_command(
        SMOKE_PROTOCOL,
        protocol,
        seed=17,
        out_dir=tmp_path,
        device="mps",
        max_frames=4,
        allow_local_mps_execution=False,
        frame_counts=(0, 2, 4),
    )

    assert command[command.index("--only-lane") + 1] == "world_tubes"
    assert "--frozen-world-replay-compiled" in command
    assert command[command.index("--frozen-world-max-frames") + 1] == "4"
    assert command[command.index("--frozen-world-frame-counts") + 1] == "0,2,4"
    assert command[
        command.index("--frozen-world-timing-warmups") + 1
    ] == str(DEFAULT_TIMING_WARMUPS)
    assert command[
        command.index("--frozen-world-timing-repeats") + 1
    ] == str(DEFAULT_TIMING_REPEATS)
    assert "--allow-paper-local-mps-execution" not in command


def test_frozen_world_publication_defaults_and_cli_limits_fail_closed() -> None:
    assert DEFAULT_FRAME_COUNTS == (0, 4, 8, 16, 32, 64, 128)
    assert parse_frame_counts(
        ",".join(str(value) for value in DEFAULT_FRAME_COUNTS)
    ) == DEFAULT_FRAME_COUNTS
    with pytest.raises(ValueError, match="base-10 integers"):
        parse_frame_counts("0,4,nope")
    with pytest.raises(ValueError, match="too many entries"):
        parse_frame_counts(
            ",".join("2" for _ in range(MAX_FRAME_COUNT_REQUESTS + 1))
        )
    validate_timing_controls(
        warmups=MAX_TIMING_WARMUPS,
        repeats=MAX_TIMING_REPEATS,
    )
    with pytest.raises(ValueError, match="warmups"):
        validate_timing_controls(
            warmups=MAX_TIMING_WARMUPS + 1,
            repeats=1,
        )
    with pytest.raises(ValueError, match="repeats"):
        validate_timing_controls(
            warmups=0,
            repeats=MAX_TIMING_REPEATS + 1,
        )
    assert failure_identity(RuntimeError("fixture")) == {
        "type": "RuntimeError",
        "message": "fixture",
    }


def test_frozen_world_execution_identity_binds_current_source_and_native_binary(
    tmp_path: Path,
) -> None:
    protocol_path = tmp_path / "protocol.jsonc"
    protocol_path.write_text("{}", encoding="utf-8")
    report_path = tmp_path / "comparison_report.json"
    report_path.write_text("{}", encoding="utf-8")
    native_dir = tmp_path / "native"
    native_dir.mkdir()
    native_path = native_dir / "_C.so"
    native_path.write_bytes(b"native-extension")
    native = _native_identity(native_path)
    source = {
        "repository_commit": "a" * 40,
        "repository_dirty": False,
        "star_uvt_commit": "b" * 40,
        "star_uvt_dirty": False,
    }
    command = ["python", "compare.py"]
    dataset_input_identity = {"schema_version": 1, "sha256": "d" * 64}
    identity = {
        "schema_version": 1,
        "protocol_path": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "command": command,
        "comparison_report_sha256": hashlib.sha256(
            report_path.read_bytes()
        ).hexdigest(),
        "source_start": source,
        "source_finish": source,
        "star_uvt_native_extension": native,
        "dataset_input_identity": dataset_input_identity,
    }

    validate_execution_identity(
        identity,
        protocol_path=protocol_path,
        command=command,
        report_path=report_path,
        expected_source=source,
        expected_native_extension=native,
        expected_dataset_input_identity=dataset_input_identity,
    )

    changed_source = {**source, "repository_commit": "c" * 40}
    with pytest.raises(ValueError, match="does not match current source"):
        validate_execution_identity(
            identity,
            protocol_path=protocol_path,
            command=command,
            report_path=report_path,
            expected_source=changed_source,
            expected_native_extension=native,
            expected_dataset_input_identity=dataset_input_identity,
        )


def test_frozen_world_live_resource_gate_rejects_incident_pressure() -> None:
    healthy = {
        "platform": "darwin",
        "available_memory_bytes": LIVE_RESOURCE_THRESHOLDS[
            "available_memory_bytes"
        ],
        "swap_used_bytes": LIVE_RESOURCE_THRESHOLDS["maximum_swap_used_bytes"],
        "disk_free_bytes": LIVE_RESOURCE_THRESHOLDS["disk_free_bytes"],
        "load_1m_per_logical_cpu": LIVE_RESOURCE_THRESHOLDS[
            "maximum_load_1m_per_logical_cpu"
        ],
    }
    require_live_resources(healthy)

    pressured = {
        **healthy,
        "swap_used_bytes": LIVE_RESOURCE_THRESHOLDS[
            "maximum_swap_used_bytes"
        ]
        + 1,
    }
    with pytest.raises(RuntimeError, match="swap_used_bytes"):
        require_live_resources(pressured)


def test_frozen_world_sweep_binds_one_checkpoint_and_rejects_row_drift(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    checkpoint = tmp_path / "world.pt"
    _write_checkpoint(checkpoint, protocol)
    native_dir = tmp_path / "native"
    native_dir.mkdir()
    native_extension = native_dir / "_C.so"
    native_extension.write_bytes(b"native-extension")
    native = _native_identity(native_extension)
    frame_counts = (0, 2, 4)
    report = _comparison_report(
        checkpoint,
        protocol,
        native,
        frame_counts=frame_counts,
    )

    validate_report_identity(
        report,
        protocol=protocol,
        seed=17,
        max_frames=4,
        frame_counts=frame_counts,
    )
    sweep = report["star_uvt"]["frozen_world_replay_compiled_sweep"]
    assert sweep["resolved_frame_counts"] == [2, 4]
    assert sweep["rows"][0]["frame_indices"] == [0, 3]
    assert sweep["rows"][0]["centered_frame_times"] == [-1.5, 1.5]
    assert {
        row["checkpoint"]["sha256"] for row in sweep["rows"]
    } == {sweep["shared_checkpoint_file_sha256"]}
    assert sweep["selected_time_slice_parity_frame_count"] == 2
    assert sweep["selected_time_slice_parity_accepted"] is True
    assert sweep["all_rows_timing_publication_ready"] is True
    assert sweep["rows"][0]["selected_time_slice_parity"][
        "accepted"
    ] is True
    assert sweep["rows"][1]["selected_time_slice_parity"][
        "status"
    ] == "not_run"

    drifted_checkpoint = tmp_path / "world_drifted.pt"
    drifted_world_state_sha = _write_checkpoint(
        drifted_checkpoint,
        protocol,
        value=2.0,
    )
    drifted = _comparison_report(
        checkpoint,
        protocol,
        native,
        frame_counts=frame_counts,
        row_checkpoints={2: (drifted_checkpoint, drifted_world_state_sha)},
    )
    with pytest.raises(ValueError, match="checkpoint identity drifted"):
        validate_report_identity(
            drifted,
            protocol=protocol,
            seed=17,
            max_frames=4,
            frame_counts=frame_counts,
        )


def test_frozen_world_publication_gate_requires_full_canonical_sweep() -> None:
    prefix_only = {
        "all_rows_accepted": True,
        "requested_frame_counts": [4, 8, 16],
        "selected_time_slice_parity_accepted": True,
        "all_rows_timing_publication_ready": True,
        "all_rows_storage_publication_ready": True,
        "all_rows_route_memory_publication_ready": True,
        "timing_benchmark_warmups": 1,
        "timing_benchmark_repeats": 5,
    }
    assert not sweep_publication_eligible(prefix_only, full_frames=300)
    complete = {
        "all_rows_accepted": True,
        "requested_frame_counts": [0, *CANONICAL_FRAME_COUNTS],
        "selected_time_slice_parity_accepted": True,
        "all_rows_timing_publication_ready": True,
        "all_rows_storage_publication_ready": True,
        "all_rows_route_memory_publication_ready": True,
        "timing_benchmark_warmups": 1,
        "timing_benchmark_repeats": 5,
    }
    assert sweep_publication_eligible(complete, full_frames=300)
    without_parity = {**complete, "selected_time_slice_parity_accepted": False}
    assert not sweep_publication_eligible(without_parity, full_frames=300)
    without_storage = {**complete, "all_rows_storage_publication_ready": False}
    assert not sweep_publication_eligible(without_storage, full_frames=300)
    without_memory = {
        **complete,
        "all_rows_route_memory_publication_ready": False,
    }
    assert not sweep_publication_eligible(without_memory, full_frames=300)
    single_shot = {
        **complete,
        "all_rows_timing_publication_ready": False,
        "timing_benchmark_warmups": 0,
        "timing_benchmark_repeats": 1,
    }
    assert not sweep_publication_eligible(single_shot, full_frames=300)


def test_frozen_world_retained_storage_binds_serialized_topology(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    checkpoint = tmp_path / "world.pt"
    _write_checkpoint(checkpoint, protocol)
    native_path = tmp_path / "native" / "_C.so"
    native_path.parent.mkdir()
    native_path.write_bytes(b"native-extension")
    report = _comparison_report(
        checkpoint,
        protocol,
        _native_identity(native_path),
        frame_counts=(2, 4),
    )
    validate_report_identity(
        report,
        protocol=protocol,
        seed=17,
        max_frames=4,
        frame_counts=(2, 4),
    )

    artifact_path = Path(
        report["star_uvt"]["frozen_world_replay_compiled_sweep"]["rows"][0]
        ["retained_storage_bytes"]["compiled"]["artifact"]["path"]
    )
    artifact = bytearray(artifact_path.read_bytes())
    artifact[-1] ^= 1
    artifact_path.write_bytes(artifact)
    with pytest.raises(ValueError, match="retained atlas .* drifted"):
        validate_report_identity(
            report,
            protocol=protocol,
            seed=17,
            max_frames=4,
            frame_counts=(2, 4),
        )


def test_frozen_world_route_memory_rejects_peak_algebra_drift(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    checkpoint = tmp_path / "world.pt"
    _write_checkpoint(checkpoint, protocol)
    native_path = tmp_path / "native" / "_C.so"
    native_path.parent.mkdir()
    native_path.write_bytes(b"native-extension")
    report = _comparison_report(
        checkpoint,
        protocol,
        _native_identity(native_path),
        frame_counts=(2, 4),
    )
    compiled_memory = report["star_uvt"][
        "frozen_world_replay_compiled_sweep"
    ]["rows"][0]["route_memory"]["compiled"]
    compiled_memory["peak_increment_current_allocated_bytes"] += 1

    with pytest.raises(ValueError, match="compiled memory algebra drifted"):
        validate_report_identity(
            report,
            protocol=protocol,
            seed=17,
            max_frames=4,
            frame_counts=(2, 4),
        )


def test_frozen_world_logical_volume_cannot_be_promoted_to_storage(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    checkpoint = tmp_path / "world.pt"
    _write_checkpoint(checkpoint, protocol)
    native_path = tmp_path / "native" / "_C.so"
    native_path.parent.mkdir()
    native_path.write_bytes(b"native-extension")
    report = _comparison_report(
        checkpoint,
        protocol,
        _native_identity(native_path),
        frame_counts=(2, 4),
    )
    payload = report["star_uvt"]["frozen_world_replay_compiled_sweep"][
        "rows"
    ][0]["payload_bytes"]
    payload["storage_claim_eligible"] = True

    with pytest.raises(
        ValueError,
        match="excluded from storage claims|logical payload proxy",
    ):
        validate_report_identity(
            report,
            protocol=protocol,
            seed=17,
            max_frames=4,
            frame_counts=(2, 4),
        )


def test_frozen_world_single_shot_timing_fields_remain_diagnostic(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    checkpoint = tmp_path / "world.pt"
    _write_checkpoint(checkpoint, protocol)
    native_dir = tmp_path / "native"
    native_dir.mkdir()
    native_extension = native_dir / "_C.so"
    native_extension.write_bytes(b"native-extension")
    report = _comparison_report(
        checkpoint,
        protocol,
        _native_identity(native_extension),
        frame_counts=(2, 4),
        timing_warmups=0,
        timing_repeats=1,
    )

    validate_report_identity(
        report,
        protocol=protocol,
        seed=17,
        max_frames=4,
        frame_counts=(2, 4),
        timing_warmups=0,
        timing_repeats=1,
    )

    sweep = report["star_uvt"]["frozen_world_replay_compiled_sweep"]
    assert sweep["timing_label"] == "single_shot_correctness_timing"
    assert sweep["timing_repeats"] == 1
    assert sweep["timing_warmups"] == 0
    assert sweep["all_rows_timing_publication_ready"] is False
    assert sweep["rows"][0]["timing_benchmark"][
        "measurement_source"
    ] == "backward_compatible_correctness_pass"


def test_frozen_world_time_grid_spans_one_fixed_full_interval() -> None:
    assert full_interval_frame_indices(300, 4) == (0, 100, 199, 299)
    assert full_interval_frame_indices(300, 8)[0::7] == (0, 299)
    with pytest.raises(ValueError, match="at least two frames"):
        resolve_frame_counts(
            full_frames=300,
            max_frames=1,
            frame_counts=None,
        )


def test_frozen_world_report_validator_checks_identity_and_checkpoint(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    checkpoint = tmp_path / "world.pt"
    _write_checkpoint(checkpoint, protocol)
    native_dir = tmp_path / "native"
    native_dir.mkdir()
    native_extension = native_dir / "_C.so"
    native_extension.write_bytes(b"native-extension")
    native = _native_identity(native_extension)
    report = _comparison_report(checkpoint, protocol, native)

    frozen = validate_report_identity(
        report,
        protocol=protocol,
        seed=17,
        max_frames=4,
    )

    assert frozen["accepted"] is True
    broken = copy.deepcopy(report)
    broken["star_uvt"]["frozen_world_replay_compiled"]["frame_count"] = 3
    with pytest.raises(ValueError, match="frame count drifted"):
        validate_report_identity(
            broken,
            protocol=protocol,
            seed=17,
            max_frames=4,
        )

    broken = copy.deepcopy(report)
    broken["star_uvt"]["frozen_world_replay_compiled"]["accepted"] = False
    with pytest.raises(ValueError, match="accepted status"):
        validate_report_identity(
            broken,
            protocol=protocol,
            seed=17,
            max_frames=4,
        )

    broken = copy.deepcopy(report)
    broken["star_uvt"]["frozen_world_replay_compiled"]["loss"]["compiled"] = 0.2
    with pytest.raises(ValueError, match="loss delta is inconsistent"):
        validate_report_identity(
            broken,
            protocol=protocol,
            seed=17,
            max_frames=4,
        )

    broken = copy.deepcopy(report)
    broken["star_uvt"]["frozen_world_replay_compiled"]["gradient"][
        "replay_l2_norm"
    ] = 0.0
    with pytest.raises(ValueError, match="acceptance checks"):
        validate_report_identity(
            broken,
            protocol=protocol,
            seed=17,
            max_frames=4,
        )

    broken = copy.deepcopy(report)
    broken["star_uvt"]["frozen_world_replay_compiled"]["acceptance"][
        "gradient_global_normalized_l2_error"
    ] = 1.0
    with pytest.raises(ValueError, match="thresholds drifted"):
        validate_report_identity(
            broken,
            protocol=protocol,
            seed=17,
            max_frames=4,
        )

    broken = copy.deepcopy(report)
    broken["star_uvt"]["frozen_world_replay_compiled"]["timing_s"][
        "compiled_per_frame_forward"
    ] = 0.25
    with pytest.raises(ValueError, match="timing compiled_per_frame_forward"):
        validate_report_identity(
            broken,
            protocol=protocol,
            seed=17,
            max_frames=4,
        )

    broken = copy.deepcopy(report)
    broken["star_uvt"]["frozen_world_replay_compiled"]["atlas"][
        "fallback_tile_samples"
    ] = 1
    with pytest.raises(ValueError, match="fallback fraction is inconsistent"):
        validate_report_identity(
            broken,
            protocol=protocol,
            seed=17,
            max_frames=4,
        )

    broken = copy.deepcopy(report)
    broken["star_uvt"]["frozen_world_replay_compiled"]["gradient"][
        "replay_gradient_parameters"
    ] = ["color"]
    with pytest.raises(ValueError, match="gradient coverage"):
        validate_report_identity(
            broken,
            protocol=protocol,
            seed=17,
            max_frames=4,
        )

    broken = copy.deepcopy(report)
    broken["star_uvt"]["frozen_world_replay_compiled"][
        "timing_benchmark"
    ]["summary_s"]["compiled_total_forward"]["median"] += 1.0
    with pytest.raises(ValueError, match="robust timing"):
        validate_report_identity(
            broken,
            protocol=protocol,
            seed=17,
            max_frames=4,
        )

    parity_report = _comparison_report(
        checkpoint,
        protocol,
        native,
        frame_counts=(2, 4),
    )
    validate_report_identity(
        parity_report,
        protocol=protocol,
        seed=17,
        max_frames=4,
        frame_counts=(2, 4),
    )
    broken = copy.deepcopy(parity_report)
    parity = broken["star_uvt"][
        "frozen_world_replay_compiled_sweep"
    ]["rows"][0]["selected_time_slice_parity"]
    parity["image"]["max_abs_error"] = 1.0
    with pytest.raises(
        ValueError,
        match="selected-time slice parity checks",
    ):
        validate_report_identity(
            broken,
            protocol=protocol,
            seed=17,
            max_frames=4,
            frame_counts=(2, 4),
        )

    checkpoint.write_bytes(b"checkpoinx")
    with pytest.raises(ValueError, match="SHA-256 does not match"):
        validate_report_identity(
            report,
            protocol=protocol,
            seed=17,
            max_frames=4,
        )
