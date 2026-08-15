from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import load_config_file, serialize_config_value
from paper_training_protocol import resolve_paper_training_protocol
from research_experiments.paper_runner_suite import (
    run_unified_paper_ablation as single,
)
from research_experiments.paper_runner_suite.frozen_atlas_storage import (
    LOGICAL_PAYLOAD_DEFINITION,
    REPLAY_STORAGE_REASON,
    RETAINED_STORAGE_DEFINITION,
    ROUTE_MEMORY_DEFINITION,
    ROUTE_MEMORY_MEASUREMENT_SOURCE,
    verify_retained_storage_artifact,
)


DEFAULT_PROTOCOL = (
    ROOT
    / "src"
    / "train_configs"
    / "paper_protocols"
    / "coffee_martini_full_300f_progressive_512_v1.jsonc"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "world_tubes_frozen_world_replay_compiled_v1"
)
LIVE_RESOURCE_THRESHOLDS = single.LIVE_RESOURCE_THRESHOLDS
live_resource_snapshot = single.live_resource_snapshot
require_live_resources = single.require_live_resources
CANONICAL_FRAME_COUNTS = (4, 8, 16, 32, 64, 128)
DEFAULT_FRAME_COUNTS = (0, *CANONICAL_FRAME_COUNTS)
DEFAULT_TIMING_WARMUPS = 1
DEFAULT_TIMING_REPEATS = 5
MIN_PUBLICATION_TIMING_WARMUPS = 1
MIN_PUBLICATION_TIMING_REPEATS = 3
MAX_FRAME_COUNT_REQUESTS = 16
MAX_TIMING_WARMUPS = 10
MAX_TIMING_REPEATS = 20
TIMING_METRIC_KEYS = (
    "replay_total_forward",
    "replay_total_backward",
    "replay_total_forward_backward",
    "replay_per_frame_forward",
    "replay_per_frame_backward",
    "compiled_atlas_compile",
    "compiled_total_forward",
    "compiled_total_backward",
    "compiled_total_forward_backward",
    "compiled_compile_plus_forward_backward",
    "compiled_per_frame_forward",
    "compiled_per_frame_backward",
)


def parse_frame_counts(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    tokens = value.split(",")
    if not tokens or any(not token.strip() for token in tokens):
        raise ValueError(
            "--frame-counts must be a comma-separated list of nonnegative integers"
        )
    try:
        counts = tuple(int(token.strip()) for token in tokens)
    except ValueError as error:
        raise ValueError(
            "--frame-counts must contain only base-10 integers"
        ) from error
    if any(count < 0 for count in counts):
        raise ValueError("--frame-counts must be nonnegative")
    if len(counts) > MAX_FRAME_COUNT_REQUESTS:
        raise ValueError(
            "--frame-counts has too many entries; "
            f"maximum is {MAX_FRAME_COUNT_REQUESTS}"
        )
    return counts


def validate_timing_controls(*, warmups: int, repeats: int) -> None:
    if warmups < 0 or warmups > MAX_TIMING_WARMUPS:
        raise ValueError(
            "frozen-world timing warmups must be in "
            f"[0, {MAX_TIMING_WARMUPS}]"
        )
    if repeats < 1 or repeats > MAX_TIMING_REPEATS:
        raise ValueError(
            "frozen-world timing repeats must be in "
            f"[1, {MAX_TIMING_REPEATS}]"
        )


def resolve_frame_counts(
    *,
    full_frames: int,
    max_frames: int,
    frame_counts: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if full_frames < 1:
        raise ValueError("full_frames must be positive")
    if len(frame_counts or ()) > MAX_FRAME_COUNT_REQUESTS:
        raise ValueError("frozen-world frame-count request is too large")
    if max_frames < 0 or any(count < 0 for count in frame_counts or ()):
        raise ValueError("frozen-world frame counts must be nonnegative")
    resolved: list[int] = []
    for requested in (int(max_frames), *(frame_counts or ())):
        frame_count = full_frames if requested == 0 else min(requested, full_frames)
        if frame_count not in resolved:
            resolved.append(frame_count)
    if full_frames > 1 and 1 in resolved:
        raise ValueError(
            "full-interval frozen-world sampling requires at least two frames"
        )
    return tuple(sorted(resolved))


def full_interval_frame_indices(
    full_frames: int,
    frame_count: int,
) -> tuple[int, ...]:
    if full_frames < 1 or frame_count < 1 or frame_count > full_frames:
        raise ValueError("sampled frame count must be in [1, full_frames]")
    if frame_count == 1:
        if full_frames > 1:
            raise ValueError(
                "full-interval frozen-world sampling requires at least two frames"
            )
        return (full_frames // 2,)
    denominator = frame_count - 1
    indices = tuple(
        (
            sample * (full_frames - 1) + denominator // 2
        )
        // denominator
        for sample in range(frame_count)
    )
    if len(set(indices)) != frame_count:
        raise ValueError("full-interval frame indices are not unique")
    return indices


def sequence_sha256(values: tuple[int | float, ...]) -> str:
    return hashlib.sha256(
        json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sweep_publication_eligible(
    sweep: Mapping[str, Any],
    *,
    full_frames: int,
) -> bool:
    requested = set(sweep.get("requested_frame_counts", ()))
    return (
        sweep.get("all_rows_accepted") is True
        and full_frames >= max(CANONICAL_FRAME_COUNTS)
        and (0 in requested or full_frames in requested)
        and set(CANONICAL_FRAME_COUNTS).issubset(requested)
        and sweep.get("selected_time_slice_parity_accepted") is True
        and sweep.get("all_rows_timing_publication_ready") is True
        and sweep.get("all_rows_storage_publication_ready") is True
        and sweep.get("all_rows_route_memory_publication_ready") is True
        and int(sweep.get("timing_benchmark_warmups", -1))
        >= MIN_PUBLICATION_TIMING_WARMUPS
        and int(sweep.get("timing_benchmark_repeats", 0))
        >= MIN_PUBLICATION_TIMING_REPEATS
    )


def _timing_quantile(samples: list[float], probability: float) -> float:
    position = float(len(samples) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return samples[lower]
    fraction = position - float(lower)
    return samples[lower] * (1.0 - fraction) + samples[upper] * fraction


def timing_summary(samples: list[float]) -> dict[str, float | int]:
    if not samples or any(not _finite_nonnegative(value) for value in samples):
        raise ValueError("timing samples must be finite and nonnegative")
    ordered = sorted(float(value) for value in samples)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": _timing_quantile(ordered, 0.25),
        "median": statistics.median(ordered),
        "p75": _timing_quantile(ordered, 0.75),
        "max": ordered[-1],
        "mean": math.fsum(ordered) / float(len(ordered)),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_attempt_status(
    path: Path,
    base: Mapping[str, Any],
    *,
    status: str,
    phase: str,
    **values: Any,
) -> None:
    single.write_json(
        path,
        {
            **dict(base),
            "status": status,
            "phase": phase,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **values,
        },
    )


def failure_identity(error: BaseException) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message": str(error),
    }


def validate_native_extension_identity(
    native: Mapping[str, Any],
) -> None:
    single.validate_native_extension_identity(native)


def validate_execution_identity(
    identity: Mapping[str, Any],
    *,
    protocol_path: Path,
    command: list[str],
    report_path: Path,
    expected_source: Mapping[str, Any],
    expected_native_extension: Mapping[str, Any],
    expected_dataset_input_identity: Mapping[str, Any],
    expected_protocol: Mapping[str, Any] | None = None,
) -> None:
    if int(identity.get("schema_version", -1)) != 1:
        raise ValueError("frozen-world execution identity schema is invalid")
    if identity.get("protocol_path") != single.display_path(protocol_path):
        raise ValueError("frozen-world execution protocol identity drifted")
    if expected_protocol is not None and identity.get("protocol") != dict(
        expected_protocol
    ):
        raise ValueError("frozen-world execution protocol contract drifted")
    if identity.get("protocol_sha256") != file_sha256(protocol_path):
        raise ValueError("frozen-world execution protocol hash drifted")
    if list(identity.get("command", ())) != command:
        raise ValueError("frozen-world execution command drifted")
    if (
        not report_path.is_file()
        or identity.get("comparison_report_sha256") != file_sha256(report_path)
    ):
        raise ValueError("frozen-world comparison report hash drifted")
    start = identity.get("source_start")
    finish = identity.get("source_finish")
    if not isinstance(start, Mapping) or not isinstance(finish, Mapping):
        raise ValueError("frozen-world execution source provenance is missing")
    if dict(start) != dict(finish):
        raise ValueError("frozen-world source changed during execution")
    if dict(start) != dict(expected_source):
        raise ValueError("frozen-world reused source does not match current source")
    single.require_clean_provenance(start)
    if identity.get("star_uvt_native_extension") != dict(
        expected_native_extension
    ):
        raise ValueError("frozen-world native extension identity drifted")
    if identity.get("dataset_input_identity") != dict(
        expected_dataset_input_identity
    ):
        raise ValueError("frozen-world raw dataset identity drifted")
    validate_native_extension_identity(expected_native_extension)


def _finite_nonnegative(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def validate_timing_benchmark(
    timing: Mapping[str, Any],
    *,
    frame_count: int,
    resident_chunk_frames: int,
    legacy_timing: Mapping[str, Any],
    expected_warmups: int,
    expected_repeats: int,
) -> None:
    publication_ready = (
        expected_warmups >= MIN_PUBLICATION_TIMING_WARMUPS
        and expected_repeats >= MIN_PUBLICATION_TIMING_REPEATS
    )
    single_shot = expected_warmups == 0 and expected_repeats == 1
    expected_label = (
        "single_shot_correctness_timing"
        if single_shot
        else (
            "warmed_repeated_wall_timing_v1"
            if publication_ready
            else "diagnostic_repeated_wall_timing_v1"
        )
    )
    expected_source = (
        "backward_compatible_correctness_pass"
        if single_shot
        else "independent_alternating_paired_trials"
    )
    if (
        int(timing.get("schema_version", -1)) != 1
        or timing.get("status") != "complete"
        or timing.get("label") != expected_label
        or timing.get("publication_ready") is not publication_ready
        or int(timing.get("warmups", -1)) != expected_warmups
        or int(timing.get("repeats", 0)) != expected_repeats
        or timing.get("measurement_source") != expected_source
        or timing.get("timing_definition")
        != (
            "device-synchronized perf_counter segments; forward includes "
            "target transfer; compile includes world projection; summed totals "
            "exclude inter-segment cleanup and optimizer work"
        )
        or timing.get("device_synchronized_at_boundaries") is not True
        or timing.get("compiled_evaluator_uses_chunk_slices") is not True
        or timing.get("forward_includes_cpu_target_to_device_transfer")
        is not True
        or timing.get("compiled_atlas_compile_includes_world_projection")
        is not True
        or timing.get("backward_excludes_optimizer") is not True
        or int(timing.get("resident_chunk_frames", 0))
        != resident_chunk_frames
        or timing.get("correctness_and_slice_parity_time_excluded")
        is not (not single_shot)
    ):
        raise ValueError("frozen-world robust timing contract drifted")
    expected_order = (
        "correctness_pass_replay_then_compiled"
        if single_shot
        else "alternating_paired_replay_compiled_v1"
    )
    if timing.get("route_order") != expected_order:
        raise ValueError("frozen-world robust timing route order drifted")
    samples = timing.get("samples_s")
    summaries = timing.get("summary_s")
    if (
        not isinstance(samples, Mapping)
        or not isinstance(summaries, Mapping)
        or set(samples) != set(TIMING_METRIC_KEYS)
        or set(summaries) != set(TIMING_METRIC_KEYS)
    ):
        raise ValueError("frozen-world robust timing metrics are incomplete")
    for key in TIMING_METRIC_KEYS:
        values = samples[key]
        summary = summaries[key]
        if (
            not isinstance(values, list)
            or len(values) != expected_repeats
            or any(not _finite_nonnegative(value) for value in values)
            or not isinstance(summary, Mapping)
        ):
            raise ValueError(f"frozen-world robust timing {key} is invalid")
        expected_summary = timing_summary(
            [float(value) for value in values]
        )
        if set(summary) != set(expected_summary):
            raise ValueError(
                f"frozen-world robust timing {key} summary is incomplete"
            )
        for statistic, expected_value in expected_summary.items():
            actual_value = summary[statistic]
            if statistic == "count":
                matches = (
                    not isinstance(actual_value, bool)
                    and isinstance(actual_value, int)
                    and actual_value == expected_value
                )
            else:
                matches = _finite_nonnegative(actual_value) and math.isclose(
                    float(actual_value),
                    float(expected_value),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
            if not matches:
                raise ValueError(
                    "frozen-world robust timing "
                    f"{key}.{statistic} is inconsistent"
                )
    for sample_index in range(expected_repeats):
        replay_forward = float(samples["replay_total_forward"][sample_index])
        replay_backward = float(samples["replay_total_backward"][sample_index])
        compiled_compile = float(
            samples["compiled_atlas_compile"][sample_index]
        )
        compiled_forward = float(
            samples["compiled_total_forward"][sample_index]
        )
        compiled_backward = float(
            samples["compiled_total_backward"][sample_index]
        )
        expected_derived = {
            "replay_total_forward_backward": (
                replay_forward + replay_backward
            ),
            "replay_per_frame_forward": replay_forward / float(frame_count),
            "replay_per_frame_backward": replay_backward / float(frame_count),
            "compiled_total_forward_backward": (
                compiled_forward + compiled_backward
            ),
            "compiled_compile_plus_forward_backward": (
                compiled_compile + compiled_forward + compiled_backward
            ),
            "compiled_per_frame_forward": (
                compiled_forward / float(frame_count)
            ),
            "compiled_per_frame_backward": (
                compiled_backward / float(frame_count)
            ),
        }
        for key, expected_value in expected_derived.items():
            if not math.isclose(
                float(samples[key][sample_index]),
                expected_value,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    f"frozen-world robust timing {key} is inconsistent"
                )
    if single_shot:
        legacy_keys = {
            "replay_total_forward": "replay_total_forward",
            "replay_total_backward": "replay_total_backward",
            "compiled_atlas_compile": "compiled_atlas_compile",
            "compiled_total_forward": "compiled_total_forward",
            "compiled_total_backward": "compiled_total_backward",
        }
        for sample_key, legacy_key in legacy_keys.items():
            if not math.isclose(
                float(samples[sample_key][0]),
                float(legacy_timing[legacy_key]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "frozen-world single-shot timing no longer matches "
                    f"timing_s.{legacy_key}"
                )


def _nonnegative_integer(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and value >= 0
    )


def validate_logical_payload(payload: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "metric_kind",
        "definition",
        "topology_bytes_included",
        "storage_claim_eligible",
        "publication_claim_eligible",
        "replay_cumulative_logical_tensor_bytes",
        "compiled_trace_table_logical_tensor_bytes",
        "compiled_to_replay_logical_volume_ratio",
    }
    replay_bytes = payload.get("replay_cumulative_logical_tensor_bytes")
    compiled_bytes = payload.get("compiled_trace_table_logical_tensor_bytes")
    ratio = payload.get("compiled_to_replay_logical_volume_ratio")
    if (
        set(payload) != expected_keys
        or int(payload.get("schema_version", -1)) != 1
        or payload.get("metric_kind") != "logical_work_volume_proxy"
        or payload.get("definition") != LOGICAL_PAYLOAD_DEFINITION
        or payload.get("topology_bytes_included") is not False
        or payload.get("storage_claim_eligible") is not False
        or payload.get("publication_claim_eligible") is not False
        or not _nonnegative_integer(replay_bytes)
        or int(replay_bytes) <= 0
        or not _nonnegative_integer(compiled_bytes)
        or int(compiled_bytes) <= 0
        or not _finite_nonnegative(ratio)
        or not math.isclose(
            float(ratio),
            float(compiled_bytes) / float(replay_bytes),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(
            "frozen-world logical payload proxy contract drifted"
        )


def validate_retained_storage(
    storage: Mapping[str, Any],
    *,
    checkpoint_bytes: int,
    frame_count: int,
    trace_count: int,
    cell_count: int,
) -> None:
    expected_keys = {
        "schema_version",
        "definition",
        "shared_checkpoint_bytes",
        "shared_checkpoint_excluded_from_route_totals",
        "replay",
        "compiled",
        "topology_bytes_included",
        "storage_claim_eligible",
        "publication_claim_eligible",
    }
    replay = storage.get("replay")
    compiled = storage.get("compiled")
    if (
        set(storage) != expected_keys
        or int(storage.get("schema_version", -1)) != 1
        or storage.get("definition") != RETAINED_STORAGE_DEFINITION
        or int(storage.get("shared_checkpoint_bytes", -1))
        != checkpoint_bytes
        or storage.get("shared_checkpoint_excluded_from_route_totals")
        is not True
        or storage.get("topology_bytes_included") is not True
        or storage.get("storage_claim_eligible") is not True
        or storage.get("publication_claim_eligible") is not True
        or not isinstance(replay, Mapping)
        or not isinstance(compiled, Mapping)
    ):
        raise ValueError("frozen-world retained storage contract drifted")
    if (
        set(replay)
        != {
            "route",
            "serialized_retained_evaluator_bytes",
            "topology_applicable",
            "storage_claim_eligible",
            "reason",
        }
        or replay.get("route") != "replay"
        or replay.get("serialized_retained_evaluator_bytes") != 0
        or replay.get("topology_applicable") is not False
        or replay.get("storage_claim_eligible") is not True
        or replay.get("reason") != REPLAY_STORAGE_REASON
    ):
        raise ValueError("frozen-world replay retained storage drifted")
    artifact = compiled.get("artifact")
    if (
        set(compiled)
        != {
            "route",
            "serialized_retained_evaluator_bytes",
            "tensor_payload_bytes",
            "topology_and_container_bytes",
            "topology_bytes_included",
            "artifact",
            "storage_claim_eligible",
        }
        or compiled.get("route") != "compiled"
        or compiled.get("topology_bytes_included") is not True
        or compiled.get("storage_claim_eligible") is not True
        or not isinstance(artifact, Mapping)
    ):
        raise ValueError("frozen-world compiled retained storage drifted")
    verify_retained_storage_artifact(
        artifact,
        expected_frame_count=frame_count,
        expected_trace_count=trace_count,
        expected_cell_count=cell_count,
    )
    serialized_bytes = compiled.get("serialized_retained_evaluator_bytes")
    tensor_bytes = compiled.get("tensor_payload_bytes")
    topology_bytes = compiled.get("topology_and_container_bytes")
    if (
        not _nonnegative_integer(serialized_bytes)
        or int(serialized_bytes) <= 0
        or not _nonnegative_integer(tensor_bytes)
        or int(tensor_bytes) <= 0
        or not _nonnegative_integer(topology_bytes)
        or int(topology_bytes) <= 0
        or int(serialized_bytes) != int(tensor_bytes) + int(topology_bytes)
        or int(serialized_bytes) != int(artifact.get("bytes", -1))
        or int(tensor_bytes) != int(artifact.get("tensor_payload_bytes", -1))
        or int(topology_bytes)
        != int(artifact.get("topology_and_container_bytes", -1))
    ):
        raise ValueError("frozen-world retained storage byte algebra drifted")


def _validate_route_memory_record(
    record: Mapping[str, Any],
    *,
    route: str,
    expected_phase_names: list[str],
) -> None:
    expected_keys = {
        "schema_version",
        "route",
        "device_type",
        "route_scoped",
        "baseline_current_allocated_bytes",
        "baseline_driver_allocated_bytes",
        "sampled_peak_current_allocated_bytes",
        "sampled_peak_driver_allocated_bytes",
        "peak_increment_current_allocated_bytes",
        "peak_increment_driver_allocated_bytes",
        "memory_sample_count",
        "phase_count",
        "phases",
        "measurement_claim_eligible",
    }
    phases = record.get("phases")
    if (
        set(record) != expected_keys
        or int(record.get("schema_version", -1)) != 1
        or record.get("route") != route
        or record.get("device_type") not in {"mps", "cuda"}
        or record.get("route_scoped") is not True
        or record.get("measurement_claim_eligible") is not True
        or not isinstance(phases, list)
        or int(record.get("phase_count", -1)) != len(expected_phase_names)
        or len(phases) != len(expected_phase_names)
    ):
        raise ValueError(f"frozen-world {route} route memory contract drifted")
    phase_keys = {
        "name",
        "sampled_peak_current_allocated_bytes",
        "sampled_peak_driver_allocated_bytes",
        "memory_sample_count",
    }
    for phase, expected_name in zip(phases, expected_phase_names, strict=True):
        if (
            not isinstance(phase, Mapping)
            or set(phase) != phase_keys
            or phase.get("name") != expected_name
            or any(
                not _nonnegative_integer(phase.get(key))
                for key in (
                    "sampled_peak_current_allocated_bytes",
                    "sampled_peak_driver_allocated_bytes",
                    "memory_sample_count",
                )
            )
            or int(phase["memory_sample_count"]) <= 0
        ):
            raise ValueError(
                f"frozen-world {route} memory phase contract drifted"
            )
    integer_keys = (
        "baseline_current_allocated_bytes",
        "baseline_driver_allocated_bytes",
        "sampled_peak_current_allocated_bytes",
        "sampled_peak_driver_allocated_bytes",
        "peak_increment_current_allocated_bytes",
        "peak_increment_driver_allocated_bytes",
        "memory_sample_count",
        "phase_count",
    )
    if any(not _nonnegative_integer(record.get(key)) for key in integer_keys):
        raise ValueError(f"frozen-world {route} memory values are invalid")
    baseline_current = int(record["baseline_current_allocated_bytes"])
    baseline_driver = int(record["baseline_driver_allocated_bytes"])
    peak_current = max(
        [baseline_current]
        + [int(phase["sampled_peak_current_allocated_bytes"]) for phase in phases]
    )
    peak_driver = max(
        [baseline_driver]
        + [int(phase["sampled_peak_driver_allocated_bytes"]) for phase in phases]
    )
    if (
        int(record["sampled_peak_current_allocated_bytes"]) != peak_current
        or int(record["sampled_peak_driver_allocated_bytes"]) != peak_driver
        or int(record["peak_increment_current_allocated_bytes"])
        != peak_current - baseline_current
        or int(record["peak_increment_driver_allocated_bytes"])
        != peak_driver - baseline_driver
        or int(record["memory_sample_count"])
        != sum(int(phase["memory_sample_count"]) for phase in phases)
    ):
        raise ValueError(f"frozen-world {route} memory algebra drifted")


def validate_route_memory(
    memory: Mapping[str, Any],
    *,
    frame_count: int,
    resident_chunk_frames: int,
) -> None:
    expected_keys = {
        "schema_version",
        "definition",
        "measurement_source",
        "sampler_interval_ms",
        "compiled_parity_replay_excluded",
        "replay",
        "compiled",
        "publication_claim_eligible",
    }
    replay = memory.get("replay")
    compiled = memory.get("compiled")
    if (
        set(memory) != expected_keys
        or int(memory.get("schema_version", -1)) != 1
        or memory.get("definition") != ROUTE_MEMORY_DEFINITION
        or memory.get("measurement_source")
        != ROUTE_MEMORY_MEASUREMENT_SOURCE
        or memory.get("sampler_interval_ms") != 5.0
        or memory.get("compiled_parity_replay_excluded") is not True
        or memory.get("publication_claim_eligible") is not True
        or not isinstance(replay, Mapping)
        or not isinstance(compiled, Mapping)
    ):
        raise ValueError("frozen-world route memory contract drifted")
    chunk_ranges = [
        (start, min(frame_count, start + resident_chunk_frames))
        for start in range(0, frame_count, resident_chunk_frames)
    ]
    compiled_phase_names = ["atlas_compile"]
    for start, stop in chunk_ranges:
        compiled_phase_names.extend(
            (
                f"chunk_{start:04d}_{stop:04d}_forward",
                f"chunk_{start:04d}_{stop:04d}_backward",
            )
        )
    _validate_route_memory_record(
        replay,
        route="replay",
        expected_phase_names=["correctness_forward_backward"],
    )
    _validate_route_memory_record(
        compiled,
        route="compiled",
        expected_phase_names=compiled_phase_names,
    )
    if replay["device_type"] != compiled["device_type"]:
        raise ValueError("frozen-world route memory device identity drifted")


def validate_selected_time_slice_parity(
    parity: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
) -> None:
    frame_count = int(row["frame_count"])
    centered_times = tuple(
        float(value) for value in row["centered_frame_times"]
    )
    time_steps = tuple(
        centered_times[index + 1] - centered_times[index]
        for index in range(frame_count - 1)
    )
    if (
        int(parity.get("schema_version", -1)) != 1
        or parity.get("status") != "complete"
        or parity.get("timing_claim_eligible") is not False
        or int(parity.get("frame_count", 0)) != frame_count
        or int(parity.get("full_dataset_frame_count", 0))
        != int(row["full_dataset_frame_count"])
        or tuple(parity.get("frame_indices", ()))
        != tuple(row["frame_indices"])
        or tuple(parity.get("centered_frame_times", ()))
        != centered_times
        or tuple(parity.get("time_steps", ())) != time_steps
        or int(parity.get("slice_chunk_frames", 0)) != 1
        or int(parity.get("slice_count", 0)) != frame_count
        or parity.get("contract_hashes") != row["contract_hashes"]
    ):
        raise ValueError("frozen-world selected-time slice parity drifted")
    non_unit = any(
        not math.isclose(abs(step), 1.0, rel_tol=0.0, abs_tol=1.0e-7)
        for step in time_steps
    )
    for key in (
        "parent_atlas_trace_count",
        "parent_atlas_cell_count",
        "cumulative_sliced_trace_count",
        "cumulative_sliced_cell_count",
    ):
        value = parity.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            raise ValueError(
                f"frozen-world selected-time slice parity {key} is invalid"
            )
    world_state = parity.get("world_state")
    checkpoint_sha = row["checkpoint"]["world_state_sha256"]
    if (
        not isinstance(world_state, Mapping)
        or {
            world_state.get("before_sha256"),
            world_state.get("after_full_atlas_sha256"),
            world_state.get("after_sliced_atlas_sha256"),
        }
        != {checkpoint_sha}
        or world_state.get("unchanged") is not True
    ):
        raise ValueError(
            "frozen-world selected-time slice parity changed the world"
        )
    loss = parity.get("loss")
    image = parity.get("image")
    gradient = parity.get("gradient")
    acceptance = parity.get("acceptance")
    checks = parity.get("checks")
    if any(
        not isinstance(section, Mapping)
        for section in (loss, image, gradient, acceptance, checks)
    ):
        raise ValueError(
            "frozen-world selected-time slice parity sections are missing"
        )
    scalar_paths = (
        (loss, "full_atlas"),
        (loss, "chunk_sliced"),
        (loss, "absolute_delta"),
        (image, "max_abs_error"),
        (image, "mean_abs_error"),
        (gradient, "global_normalized_l2_error"),
        (gradient, "full_atlas_l2_norm"),
        (gradient, "chunk_sliced_l2_norm"),
        (gradient, "max_parameter_normalized_l2_error"),
    )
    if any(not _finite_nonnegative(section.get(key)) for section, key in scalar_paths):
        raise ValueError(
            "frozen-world selected-time slice parity scalars are invalid"
        )
    cosine_similarity = gradient.get("cosine_similarity")
    if (
        isinstance(cosine_similarity, bool)
        or not isinstance(cosine_similarity, (int, float))
        or not math.isfinite(float(cosine_similarity))
        or not -1.0 <= float(cosine_similarity) <= 1.0
    ):
        raise ValueError(
            "frozen-world selected-time slice parity cosine is invalid"
        )
    if not math.isclose(
        float(loss["absolute_delta"]),
        abs(float(loss["full_atlas"]) - float(loss["chunk_sliced"])),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "frozen-world selected-time slice parity loss is inconsistent"
        )
    expected_acceptance = {
        key: single.FROZEN_WORLD_ACCEPTANCE[key]
        for key in (
            "image_max_abs_error",
            "loss_absolute_delta",
            "gradient_global_normalized_l2_error",
            "gradient_max_parameter_normalized_l2_error",
            "min_world_vjp_l2_norm",
        )
    }
    parameter_names = row["checkpoint"]["parameter_names"]
    per_parameter = gradient.get("per_parameter_normalized_l2_error")
    if (
        dict(acceptance) != expected_acceptance
        or gradient.get("gradient_coverage_matches") is not True
        or int(gradient.get("parameter_tensor_count", 0))
        != len(parameter_names)
        or int(gradient.get("full_atlas_gradient_tensor_count", 0))
        != len(parameter_names)
        or int(gradient.get("chunk_sliced_gradient_tensor_count", 0))
        != len(parameter_names)
        or gradient.get("full_atlas_gradient_parameters")
        != parameter_names
        or gradient.get("chunk_sliced_gradient_parameters")
        != parameter_names
        or not isinstance(per_parameter, Mapping)
        or set(per_parameter) != set(parameter_names)
        or any(
            not _finite_nonnegative(value)
            for value in per_parameter.values()
        )
        or not math.isclose(
            float(gradient["max_parameter_normalized_l2_error"]),
            max(float(value) for value in per_parameter.values()),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(
            "frozen-world selected-time slice parity gradient contract drifted"
        )
    expected_checks = {
        "non_unit_selected_times": non_unit,
        "same_parent_atlas": True,
        "world_state_unchanged": True,
        "image_matches": float(image["max_abs_error"])
        <= float(acceptance["image_max_abs_error"]),
        "loss_matches": float(loss["absolute_delta"])
        <= float(acceptance["loss_absolute_delta"]),
        "world_vjp_matches": float(
            gradient["global_normalized_l2_error"]
        )
        <= float(acceptance["gradient_global_normalized_l2_error"]),
        "world_vjp_per_parameter_matches": float(
            gradient["max_parameter_normalized_l2_error"]
        )
        <= float(
            acceptance["gradient_max_parameter_normalized_l2_error"]
        ),
        "world_vjp_nonzero": min(
            float(gradient["full_atlas_l2_norm"]),
            float(gradient["chunk_sliced_l2_norm"]),
        )
        > float(acceptance["min_world_vjp_l2_norm"]),
        "world_vjp_coverage_matches": True,
    }
    if dict(checks) != expected_checks:
        raise ValueError(
            "frozen-world selected-time slice parity checks are inconsistent"
        )
    if parity.get("accepted") is not all(expected_checks.values()):
        raise ValueError(
            "frozen-world selected-time slice parity status is inconsistent"
        )


def validate_report_identity(
    report: Mapping[str, Any],
    *,
    protocol,
    seed: int,
    max_frames: int,
    frame_counts: tuple[int, ...] | None = None,
    timing_warmups: int = DEFAULT_TIMING_WARMUPS,
    timing_repeats: int = DEFAULT_TIMING_REPEATS,
) -> Mapping[str, Any]:
    validate_timing_controls(
        warmups=timing_warmups,
        repeats=timing_repeats,
    )
    meta = report.get("meta")
    lane = report.get("star_uvt")
    if not isinstance(meta, Mapping) or not isinstance(lane, Mapping):
        raise ValueError("frozen-world run is missing World Tubes report data")
    if int(meta.get("seed", -1)) != int(seed):
        raise ValueError("frozen-world report seed drifted")
    if int(meta.get("frame_count", 0)) != protocol.dataset.frame_count:
        raise ValueError("frozen-world report dataset frame count drifted")
    if tuple(meta.get("train_cameras", ())) != protocol.dataset.train_cameras:
        raise ValueError("frozen-world report train cameras drifted")
    if tuple(meta.get("heldout_cameras", ())) != protocol.dataset.heldout_cameras:
        raise ValueError("frozen-world report heldout cameras drifted")
    if meta.get("only_lane") != "world_tubes":
        raise ValueError("frozen-world report was not lane-isolated")
    if meta.get("frozen_world_replay_compiled") is not True:
        raise ValueError("frozen-world report mode is disabled")
    if int(meta.get("frozen_world_max_frames", 0)) != int(max_frames):
        raise ValueError("frozen-world report frame limit drifted")
    if (
        meta.get("frozen_world_temporal_sampling")
        != "ordered_full_interval_integer_lattice_v1"
    ):
        raise ValueError("frozen-world report temporal sampling drifted")
    expected_meta_frame_counts = (
        None if frame_counts is None else list(frame_counts)
    )
    if meta.get("frozen_world_frame_counts") != expected_meta_frame_counts:
        raise ValueError("frozen-world report requested frame counts drifted")
    if (
        int(meta.get("frozen_world_timing_warmups", -1))
        != timing_warmups
        or int(meta.get("frozen_world_timing_repeats", 0))
        != timing_repeats
    ):
        raise ValueError("frozen-world report timing controls drifted")
    if meta.get("uvt_world_representation") != "legacy_tube":
        raise ValueError("frozen-world report representation drifted")
    if meta.get("uvt_alpha_mode") != "peak_splat":
        raise ValueError("frozen-world report alpha mode drifted")
    if meta.get("uvt_render_backend") != "metal_tile":
        raise ValueError("frozen-world report renderer drifted")
    if meta.get("uvt_camera_projection") != "dataset":
        raise ValueError("frozen-world camera projection drifted")
    if meta.get("uvt_camera_sequence_mode") != "static_view":
        raise ValueError("frozen-world camera sequence mode drifted")
    native_extension = meta.get("star_uvt_native_extension")
    if not isinstance(native_extension, Mapping):
        raise ValueError("frozen-world report native extension identity is missing")
    validate_native_extension_identity(native_extension)
    route_native_extension = meta.get("route_native_extension")
    single.validate_route_native_extension_identity(
        "world_tubes",
        route_native_extension,
    )
    if route_native_extension != native_extension:
        raise ValueError("frozen-world route-native identity drifted")
    for name, schema_version in (
        (
            "paper_dataset_bundle",
            single.PAPER_DATASET_BUNDLE_SCHEMA_VERSION,
        ),
        ("paper_evaluator", single.PAPER_EVALUATOR_SCHEMA_VERSION),
        ("paper_runtime", single.PAPER_RUNTIME_SCHEMA_VERSION),
    ):
        single.validate_hashed_contract(
            f"frozen-world {name}",
            meta.get(name),
            schema_version=schema_version,
        )
    if meta["paper_evaluator"] != single.paper_evaluator_contract():
        raise ValueError("frozen-world evaluator is not canonical")
    single.validate_lane_cost("world_tubes", lane, protocol, seed=seed)
    single.build_lane_evidence(
        "world_tubes",
        lane,
        frame_count=protocol.dataset.frame_count,
    )
    frozen = lane.get("frozen_world_replay_compiled")
    if not isinstance(frozen, Mapping):
        raise ValueError("frozen-world evidence is missing")
    expected_frames = (
        protocol.dataset.frame_count
        if max_frames <= 0
        else min(max_frames, protocol.dataset.frame_count)
    )
    single.validate_frozen_world_evidence(
        frozen,
        expected_frames=expected_frames,
        expected_full_frames=protocol.dataset.frame_count,
        expected_image_size=(
            protocol.final_stage.image_size.height,
            protocol.final_stage.image_size.width,
        ),
        expected_heldout_camera=protocol.dataset.heldout_cameras[0],
        expected_active_tubes=protocol.final_stage.primitive_count,
    )
    sweep = lane.get("frozen_world_replay_compiled_sweep")
    if not isinstance(sweep, Mapping):
        raise ValueError("frozen-world sweep evidence is missing")
    if int(sweep.get("schema_version", -1)) != 1:
        raise ValueError("frozen-world sweep schema is missing or stale")
    if sweep.get("status") != "complete":
        raise ValueError("frozen-world sweep is incomplete")
    if (
        sweep.get("timing_label") != "single_shot_correctness_timing"
        or int(sweep.get("timing_repeats", 0)) != 1
        or int(sweep.get("timing_warmups", -1)) != 0
    ):
        raise ValueError("frozen-world sweep timing contract drifted")
    expected_timing_publication_ready = (
        timing_warmups >= MIN_PUBLICATION_TIMING_WARMUPS
        and timing_repeats >= MIN_PUBLICATION_TIMING_REPEATS
    )
    expected_timing_label = (
        "single_shot_correctness_timing"
        if timing_warmups == 0 and timing_repeats == 1
        else (
            "warmed_repeated_wall_timing_v1"
            if expected_timing_publication_ready
            else "diagnostic_repeated_wall_timing_v1"
        )
    )
    if (
        sweep.get("timing_benchmark_label") != expected_timing_label
        or int(sweep.get("timing_benchmark_warmups", -1))
        != timing_warmups
        or int(sweep.get("timing_benchmark_repeats", 0))
        != timing_repeats
    ):
        raise ValueError("frozen-world robust timing sweep contract drifted")
    if (
        sweep.get("temporal_sampling")
        != "ordered_full_interval_integer_lattice_v1"
    ):
        raise ValueError("frozen-world sweep temporal sampling drifted")
    if list(sweep.get("requested_frame_counts", ())) != list(
        frame_counts or ()
    ):
        raise ValueError("frozen-world sweep requested frame counts drifted")
    if int(sweep.get("primary_requested_frame_count", -1)) != int(max_frames):
        raise ValueError("frozen-world sweep primary request drifted")
    expected_resolved = resolve_frame_counts(
        full_frames=protocol.dataset.frame_count,
        max_frames=max_frames,
        frame_counts=frame_counts,
    )
    if list(sweep.get("resolved_frame_counts", ())) != list(expected_resolved):
        raise ValueError("frozen-world sweep resolved frame counts drifted")
    expected_slice_parity_frame_count = next(
        (
            frame_count
            for frame_count in expected_resolved
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
    if sweep.get("selected_time_slice_parity_frame_count") != (
        expected_slice_parity_frame_count
    ):
        raise ValueError(
            "frozen-world selected-time slice parity row drifted"
        )
    if (
        int(sweep.get("primary_resolved_frame_count", 0)) != expected_frames
        or int(sweep.get("full_dataset_frame_count", 0))
        != protocol.dataset.frame_count
    ):
        raise ValueError("frozen-world sweep frame contract drifted")
    rows = sweep.get("rows")
    if not isinstance(rows, list) or len(rows) != len(expected_resolved):
        raise ValueError("frozen-world sweep rows are missing")
    shared_checkpoint = sweep.get("shared_checkpoint")
    shared_world_state_sha = sweep.get("shared_world_state_sha256")
    if not isinstance(shared_checkpoint, Mapping):
        raise ValueError("frozen-world sweep shared checkpoint is missing")
    selected_time_slice_parity_accepted = False
    for row, frame_count in zip(rows, expected_resolved, strict=True):
        if not isinstance(row, Mapping):
            raise ValueError("frozen-world sweep row is invalid")
        single.validate_frozen_world_evidence(
            row,
            expected_frames=frame_count,
            expected_full_frames=protocol.dataset.frame_count,
            expected_image_size=(
                protocol.final_stage.image_size.height,
                protocol.final_stage.image_size.width,
            ),
            expected_heldout_camera=protocol.dataset.heldout_cameras[0],
            expected_active_tubes=protocol.final_stage.primitive_count,
        )
        payload = row.get("payload_bytes")
        retained_storage = row.get("retained_storage_bytes")
        route_memory = row.get("route_memory")
        if (
            not isinstance(payload, Mapping)
            or not isinstance(retained_storage, Mapping)
            or not isinstance(route_memory, Mapping)
        ):
            raise ValueError(
                "frozen-world storage or route memory evidence is missing"
            )
        validate_logical_payload(payload)
        validate_retained_storage(
            retained_storage,
            checkpoint_bytes=int(row["checkpoint"]["bytes"]),
            frame_count=frame_count,
            trace_count=int(row["atlas"]["trace_count"]),
            cell_count=int(row["atlas"]["cell_count"]),
        )
        validate_route_memory(
            route_memory,
            frame_count=frame_count,
            resident_chunk_frames=int(
                row["contract"]["resident_chunk_frames"]
            ),
        )
        validate_timing_benchmark(
            row.get("timing_benchmark", {}),
            frame_count=frame_count,
            resident_chunk_frames=int(
                row["contract"]["resident_chunk_frames"]
            ),
            legacy_timing=row["timing_s"],
            expected_warmups=timing_warmups,
            expected_repeats=timing_repeats,
        )
        expected_indices = full_interval_frame_indices(
            protocol.dataset.frame_count,
            frame_count,
        )
        expected_times = tuple(
            float(frame) - 0.5 * float(protocol.dataset.frame_count - 1)
            for frame in expected_indices
        )
        if (
            row.get("temporal_sampling")
            != "ordered_full_interval_integer_lattice_v1"
            or tuple(row.get("frame_indices", ())) != expected_indices
            or tuple(row.get("centered_frame_times", ())) != expected_times
            or row["contract_hashes"].get("frame_indices_sha256")
            != sequence_sha256(expected_indices)
            or row["contract_hashes"].get(
                "centered_frame_times_sha256"
            )
            != sequence_sha256(expected_times)
        ):
            raise ValueError(
                "frozen-world sweep fixed-program time grid drifted"
            )
        if dict(row["checkpoint"]) != dict(shared_checkpoint):
            raise ValueError(
                "frozen-world sweep checkpoint identity drifted across rows"
            )
        if any(
            row["world_state"].get(key) != shared_world_state_sha
            for key in (
                "checkpoint_sha256",
                "before_routes_sha256",
                "after_replay_sha256",
                "after_compiled_sha256",
            )
        ):
            raise ValueError(
                "frozen-world sweep world-state identity drifted across rows"
            )
        parity = row.get("selected_time_slice_parity")
        if not isinstance(parity, Mapping):
            raise ValueError(
                "frozen-world selected-time slice parity evidence is missing"
            )
        if frame_count == expected_slice_parity_frame_count:
            validate_selected_time_slice_parity(parity, row=row)
            selected_time_slice_parity_accepted = (
                parity.get("accepted") is True
            )
        elif (
            int(parity.get("schema_version", -1)) != 1
            or parity.get("status") != "not_run"
            or parity.get("accepted") is not False
            or parity.get("timing_claim_eligible") is not False
        ):
            raise ValueError(
                "frozen-world selected-time slice parity ran on the wrong row"
            )
    if (
        shared_checkpoint.get("sha256")
        != sweep.get("shared_checkpoint_file_sha256")
        or shared_checkpoint.get("world_state_sha256")
        != shared_world_state_sha
        or sweep.get("checkpoint_shared_across_rows") is not True
        or sweep.get("world_state_shared_across_rows") is not True
    ):
        raise ValueError("frozen-world sweep shared identities are inconsistent")
    all_rows_accepted = all(row.get("accepted") is True for row in rows)
    if sweep.get("all_rows_accepted") is not all_rows_accepted:
        raise ValueError("frozen-world sweep accepted status is inconsistent")
    all_rows_timing_publication_ready = all(
        row["timing_benchmark"]["publication_ready"] is True
        for row in rows
    )
    if (
        sweep.get("all_rows_timing_publication_ready")
        is not all_rows_timing_publication_ready
        or all_rows_timing_publication_ready
        is not expected_timing_publication_ready
    ):
        raise ValueError(
            "frozen-world robust timing publication status is inconsistent"
        )
    all_rows_storage_publication_ready = all(
        row["retained_storage_bytes"]["publication_claim_eligible"] is True
        for row in rows
    )
    all_rows_route_memory_publication_ready = all(
        row["route_memory"]["publication_claim_eligible"] is True
        for row in rows
    )
    if (
        sweep.get("all_rows_storage_publication_ready")
        is not all_rows_storage_publication_ready
        or sweep.get("all_rows_route_memory_publication_ready")
        is not all_rows_route_memory_publication_ready
    ):
        raise ValueError(
            "frozen-world storage or memory publication status is inconsistent"
        )
    if (
        sweep.get("selected_time_slice_parity_accepted")
        is not selected_time_slice_parity_accepted
    ):
        raise ValueError(
            "frozen-world selected-time slice parity status is inconsistent"
        )
    publication_eligible = sweep_publication_eligible(
        sweep,
        full_frames=protocol.dataset.frame_count,
    )
    if sweep.get("publication_eligible") is not publication_eligible:
        raise ValueError("frozen-world sweep publication status is inconsistent")
    primary_row = next(
        row for row in rows if int(row["frame_count"]) == expected_frames
    )
    if dict(primary_row) != dict(frozen):
        raise ValueError("frozen-world primary row drifted from the sweep")
    return frozen


def build_command(
    protocol_path: Path,
    protocol,
    *,
    seed: int,
    out_dir: Path,
    device: str,
    max_frames: int,
    allow_local_mps_execution: bool,
    frame_counts: tuple[int, ...] | None = None,
    timing_warmups: int = DEFAULT_TIMING_WARMUPS,
    timing_repeats: int = DEFAULT_TIMING_REPEATS,
) -> list[str]:
    validate_timing_controls(
        warmups=timing_warmups,
        repeats=timing_repeats,
    )
    if len(frame_counts or ()) > MAX_FRAME_COUNT_REQUESTS:
        raise ValueError("frozen-world frame-count request is too large")
    command = single.comparison_command(
        protocol_path,
        protocol,
        seed,
        out_dir,
        backward_policy="fast_exploration",
        device=device,
        frozen_world_replay_compiled=True,
        frozen_world_max_frames=max_frames,
        only_lane="world_tubes",
        allow_local_mps_execution=allow_local_mps_execution,
    )
    if frame_counts is not None:
        command.extend(
            (
                "--frozen-world-frame-counts",
                ",".join(str(count) for count in frame_counts),
            )
        )
    command.extend(
        (
            "--frozen-world-timing-warmups",
            str(timing_warmups),
            "--frozen-world-timing-repeats",
            str(timing_repeats),
        )
    )
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help=(
            "Zero evaluates every protocol frame; a positive value samples "
            "that many ordered frames over the full temporal interval."
        ),
    )
    parser.add_argument(
        "--frame-counts",
        default=",".join(str(value) for value in DEFAULT_FRAME_COUNTS),
        help=(
            "Comma-separated same-checkpoint sweep. Zero denotes the full "
            "dataset; the default is the publication sweep and every count "
            "spans the same full temporal interval."
        ),
    )
    parser.add_argument(
        "--timing-warmups",
        type=int,
        default=DEFAULT_TIMING_WARMUPS,
        help="Unreported synchronized timing pairs per frame-count row.",
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=DEFAULT_TIMING_REPEATS,
        help=(
            "Reported synchronized timing pairs per frame-count row; the "
            "publication runner defaults to five."
        ),
    )
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--require-clean-source",
        action="store_true",
        help=(
            "Compatibility/explicit-assertion flag; every executed frozen-world "
            "paper run requires clean main and STAR source regardless."
        ),
    )
    parser.add_argument("--allow-local-mps-execution", action="store_true")
    parser.add_argument("--allow-high-risk-local-mps", action="store_true")
    args = parser.parse_args()

    if args.max_frames < 0:
        raise ValueError("--max-frames must be nonnegative")
    validate_timing_controls(
        warmups=args.timing_warmups,
        repeats=args.timing_repeats,
    )
    frame_counts = parse_frame_counts(args.frame_counts)
    protocol_path = single.resolve_root_path(args.protocol)
    protocol = resolve_paper_training_protocol(load_config_file(protocol_path))
    resolved_frame_counts = resolve_frame_counts(
        full_frames=protocol.dataset.frame_count,
        max_frames=args.max_frames,
        frame_counts=frame_counts,
    )
    out_root = single.resolve_root_path(args.out_dir)
    run_dir = out_root / protocol.name / f"seed_{args.seed}"
    report_path = run_dir / "comparison_report.json"
    summary_path = run_dir / "summary.json"
    command = build_command(
        protocol_path,
        protocol,
        seed=args.seed,
        out_dir=run_dir,
        device=args.device,
        max_frames=args.max_frames,
        allow_local_mps_execution=args.allow_local_mps_execution,
        frame_counts=frame_counts,
        timing_warmups=args.timing_warmups,
        timing_repeats=args.timing_repeats,
    )
    dry_run = {
        "status": "dry_run",
        "protocol": protocol.as_dict(),
        "seed": args.seed,
        "max_frames": args.max_frames,
        "frame_counts": None if frame_counts is None else list(frame_counts),
        "resolved_frame_counts": list(resolved_frame_counts),
        "timing_warmups": args.timing_warmups,
        "timing_repeats": args.timing_repeats,
        "timing_route_pairs_per_row": (
            args.timing_warmups + args.timing_repeats
        ),
        "total_timing_route_pairs": len(resolved_frame_counts)
        * (args.timing_warmups + args.timing_repeats),
        "execution_safety": single.local_mps_safety_estimate(protocol),
        "live_resources": live_resource_snapshot(),
        "live_resource_thresholds": LIVE_RESOURCE_THRESHOLDS,
        "clean_source_policy": "always_required_for_execute",
        "require_clean_source_flag": args.require_clean_source,
        "command": command,
        "expected_report": single.display_path(report_path),
        "expected_summary": single.display_path(summary_path),
    }
    if not args.execute:
        print(json.dumps(serialize_config_value(dry_run), indent=2, sort_keys=True))
        return

    provenance = single.source_provenance()
    single.require_clean_provenance(provenance)
    manifest_validation = single.validate_manifest(protocol)

    report: Mapping[str, Any] | None = None
    execution_identity_path = run_dir / "execution_identity.json"
    execution_identity: Mapping[str, Any] | None = None
    reused_existing = False
    reuse_rejection: dict[str, str] | None = None
    if (
        args.reuse_existing
        and report_path.exists()
        and execution_identity_path.exists()
    ):
        try:
            candidate = single.load_json(report_path)
            candidate_identity = single.load_json(execution_identity_path)
            validate_report_identity(
                candidate,
                protocol=protocol,
                seed=args.seed,
                max_frames=args.max_frames,
                frame_counts=frame_counts,
                timing_warmups=args.timing_warmups,
                timing_repeats=args.timing_repeats,
            )
            validate_execution_identity(
                candidate_identity,
                protocol_path=protocol_path,
                command=command,
                report_path=report_path,
                expected_source=provenance,
                expected_native_extension=candidate["meta"][
                    "star_uvt_native_extension"
                ],
                expected_dataset_input_identity=manifest_validation[
                    "input_identity"
                ],
                expected_protocol=protocol.as_dict(),
            )
        except (KeyError, OSError, TypeError, ValueError) as error:
            report = None
            reuse_rejection = failure_identity(error)
        else:
            report = candidate
            execution_identity = candidate_identity
            reused_existing = True
    elif args.reuse_existing:
        reuse_rejection = {
            "type": "MissingReuseArtifacts",
            "message": (
                "both comparison_report.json and execution_identity.json "
                "are required for validated reuse"
            ),
        }

    attempt_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S.%fZ"
    )
    attempt_status_path = run_dir / "frozen_world_attempt.json"
    attempt_base = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "protocol": protocol.name,
        "protocol_path": single.display_path(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "seed": args.seed,
        "command": command,
        "requested_frame_counts": list(frame_counts or ()),
        "resolved_frame_counts": list(resolved_frame_counts),
        "timing_warmups": args.timing_warmups,
        "timing_repeats": args.timing_repeats,
        "clean_source_policy": "always_required",
        "source": provenance,
        "dataset_input_identity": manifest_validation["input_identity"],
        "comparison_report": single.display_path(report_path),
        "execution_identity": single.display_path(execution_identity_path),
        "summary": single.display_path(summary_path),
        "preexisting_artifact_sha256": {
            "comparison_report": (
                file_sha256(report_path) if report_path.is_file() else None
            ),
            "execution_identity": (
                file_sha256(execution_identity_path)
                if execution_identity_path.is_file()
                else None
            ),
            "summary": (
                file_sha256(summary_path) if summary_path.is_file() else None
            ),
        },
        "authoritative_completion_contract": (
            "frozen_world_attempt.json must have status=complete and its "
            "attempt_id must equal summary.json:attempt_id"
        ),
        "require_clean_source_flag": args.require_clean_source,
        "reuse_requested": args.reuse_existing,
        "reuse_rejection": reuse_rejection,
    }

    if report is None:
        write_attempt_status(
            attempt_status_path,
            attempt_base,
            status="running",
            phase="execution_safety_preflight",
            reused_existing=False,
        )
        try:
            execution_safety = (
                single.require_execution_safety_acknowledgement(
                    protocol,
                    device=args.device,
                    allow_local_mps_execution=(
                        args.allow_local_mps_execution
                    ),
                    allow_high_risk_local_mps=(
                        args.allow_high_risk_local_mps
                    ),
                )
            )
        except BaseException as error:
            write_attempt_status(
                attempt_status_path,
                attempt_base,
                status="failed",
                phase="execution_safety_preflight",
                reused_existing=False,
                failure=failure_identity(error),
            )
            raise
        live_resources = execution_safety["live_resources"]
        write_attempt_status(
            attempt_status_path,
            attempt_base,
            status="running",
            phase="child_process",
            reused_existing=False,
            live_resources_at_launch=live_resources,
        )
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except BaseException as error:
            write_attempt_status(
                attempt_status_path,
                attempt_base,
                status="failed",
                phase="child_process",
                reused_existing=False,
                live_resources_at_launch=live_resources,
                failure=failure_identity(error),
                child_progress=single.display_path(
                    run_dir / "frozen_world_sweep_progress.json"
                ),
            )
            raise
        try:
            report = single.load_json(report_path)
            validate_report_identity(
                report,
                protocol=protocol,
                seed=args.seed,
                max_frames=args.max_frames,
                frame_counts=frame_counts,
                timing_warmups=args.timing_warmups,
                timing_repeats=args.timing_repeats,
            )
            source_finish = single.source_provenance()
            single.require_clean_provenance(source_finish)
            if provenance != source_finish:
                raise RuntimeError(
                    "source changed while the frozen-world run was executing"
                )
            execution_identity = {
                "schema_version": 1,
                "protocol": protocol.as_dict(),
                "protocol_path": single.display_path(protocol_path),
                "protocol_sha256": file_sha256(protocol_path),
                "command": command,
                "source_start": provenance,
                "source_finish": source_finish,
                "comparison_report": single.display_path(report_path),
                "comparison_report_sha256": file_sha256(report_path),
                "star_uvt_native_extension": report["meta"][
                    "star_uvt_native_extension"
                ],
                "dataset_input_identity": manifest_validation[
                    "input_identity"
                ],
                "live_resources_at_launch": live_resources,
                "attempt_id": attempt_id,
            }
            single.write_json(execution_identity_path, execution_identity)
        except BaseException as error:
            write_attempt_status(
                attempt_status_path,
                attempt_base,
                status="failed",
                phase="report_validation",
                reused_existing=False,
                live_resources_at_launch=live_resources,
                failure=failure_identity(error),
            )
            raise
    else:
        write_attempt_status(
            attempt_status_path,
            attempt_base,
            status="running",
            phase="validated_reuse",
            reused_existing=True,
            original_live_resources_at_launch=execution_identity.get(
                "live_resources_at_launch"
            ),
        )

    write_attempt_status(
        attempt_status_path,
        attempt_base,
        status="running",
        phase="final_report_validation",
        reused_existing=reused_existing,
    )
    try:
        frozen = validate_report_identity(
            report,
            protocol=protocol,
            seed=args.seed,
            max_frames=args.max_frames,
            frame_counts=frame_counts,
            timing_warmups=args.timing_warmups,
            timing_repeats=args.timing_repeats,
        )
    except BaseException as error:
        write_attempt_status(
            attempt_status_path,
            attempt_base,
            status="failed",
            phase="final_report_validation",
            reused_existing=reused_existing,
            failure=failure_identity(error),
        )
        raise
    if execution_identity is None:
        error = RuntimeError(
            "frozen-world execution identity was not materialized"
        )
        write_attempt_status(
            attempt_status_path,
            attempt_base,
            status="failed",
            phase="execution_identity",
            reused_existing=reused_existing,
            failure=failure_identity(error),
        )
        raise error
    write_attempt_status(
        attempt_status_path,
        attempt_base,
        status="running",
        phase="wandb_logging",
        reused_existing=reused_existing,
    )
    try:
        wandb = single._comparison_wandb_log(
            report,
            protocol,
            lane_name="world_tubes",
            seed=args.seed,
            report_dir=run_dir,
            wandb_mode=args.wandb_mode,
            execution_source=execution_identity["source_start"],
        )
    except BaseException as error:
        write_attempt_status(
            attempt_status_path,
            attempt_base,
            status="failed",
            phase="wandb_logging",
            reused_existing=reused_existing,
            failure=failure_identity(error),
        )
        raise
    sweep = report["star_uvt"]["frozen_world_replay_compiled_sweep"]
    publication_eligible = sweep_publication_eligible(
        sweep,
        full_frames=protocol.dataset.frame_count,
    )
    all_rows_accepted = sweep["all_rows_accepted"] is True
    summary = {
        "schema_version": 1,
        "status": (
            "accepted"
            if all_rows_accepted and publication_eligible
            else (
                "complete_diagnostic"
                if all_rows_accepted
                else "complete_negative"
            )
        ),
        "publication_eligible": publication_eligible,
        "protocol": protocol.as_dict(),
        "protocol_path": single.display_path(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "seed": args.seed,
        "timing_warmups": args.timing_warmups,
        "timing_repeats": args.timing_repeats,
        "resolved_frame_counts": list(resolved_frame_counts),
        "execution_mode": (
            "validated_reuse" if reused_existing else "fresh_execution"
        ),
        "attempt_id": attempt_id,
        "attempt_status": single.display_path(attempt_status_path),
        "manifest_validation": manifest_validation,
        "common_evidence_contract": {
            "schema_version": 1,
            "dataset_input_identity": manifest_validation["input_identity"],
            "decoded_dataset_bundle": report["meta"][
                "paper_dataset_bundle"
            ],
            "evaluator": report["meta"]["paper_evaluator"],
            "runtime": report["meta"]["paper_runtime"],
        },
        "source": execution_identity["source_start"],
        "source_finish": execution_identity["source_finish"],
        "execution_identity": single.display_path(execution_identity_path),
        "execution_identity_sha256": file_sha256(execution_identity_path),
        "comparison_report": single.display_path(report_path),
        "comparison_report_sha256": file_sha256(report_path),
        "wandb": wandb,
        "frozen_world_replay_compiled": frozen,
        "frozen_world_replay_compiled_sweep": sweep,
    }
    write_attempt_status(
        attempt_status_path,
        attempt_base,
        status="running",
        phase="summary_write",
        reused_existing=reused_existing,
        publication_eligible=publication_eligible,
    )
    try:
        single.write_json(summary_path, summary)
    except BaseException as error:
        write_attempt_status(
            attempt_status_path,
            attempt_base,
            status="failed",
            phase="summary_write",
            reused_existing=reused_existing,
            publication_eligible=publication_eligible,
            failure=failure_identity(error),
        )
        raise
    write_attempt_status(
        attempt_status_path,
        attempt_base,
        status="complete",
        phase="summary_written",
        reused_existing=reused_existing,
        publication_eligible=publication_eligible,
        summary=single.display_path(summary_path),
        summary_sha256=file_sha256(summary_path),
    )
    print(json.dumps(serialize_config_value(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
