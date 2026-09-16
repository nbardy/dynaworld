from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import load_config_file, serialize_config_value
from paper_training_protocol import resolve_paper_training_protocol
from research_experiments.paper_runner_suite import (
    run_frozen_world_replay_compiled as static_frozen,
)
from research_experiments.paper_runner_suite import (
    run_unified_paper_ablation as single,
)


SCHEMA_VERSION = 1
DEFAULT_PROTOCOL = (
    ROOT
    / "src"
    / "train_configs"
    / "paper_protocols"
    / "coffee_martini_full_300f_progressive_512_v1.jsonc"
)
DEFAULT_STATIC_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "world_tubes_frozen_world_replay_compiled_v1"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "world_tubes_frozen_world_moving_camera_v1"
)
FRAME_COUNTS = (8, 16, 32, 64)
IMAGE_SIZE = (256, 256)
CAMERA_PROGRAM_MODE = "bounded_yaw_projective_first_order_v1"
COMPILER_CHART_POLICY = "single_midpoint_first_order"
MULTI_CHART_GAUGE_COMPILER = False
YAW_TOTAL_DEGREES = 45.0
TIMING_WARMUPS = 1
TIMING_REPEATS = 5
GIB = 1024**3
TARGET_SEMANTICS = (
    "deterministic_residual_target_not_ground_truth_moving_camera_quality"
)
PARITY_METRIC_SEMANTICS = "replay_vs_compiled_same_world_same_program"
TARGET_RESIZE_MODE = "direct_decode_256"

# These are the predeclared hard-kill thresholds in the July 28 meta-review.
# They are deliberately not CLI knobs: a failed row is retained as a negative
# result rather than made positive by changing its acceptance contract.
PUBLICATION_THRESHOLDS = {
    "max_heavy_structural_ratio_t32_t8": 1.10,
    "max_interaction_memory_ratio_t32_t8": 1.10,
    "min_f32_forward_speedup": 2.0,
    "min_f32_reverse_speedup": 2.0,
    "min_f32_total_speedup": 1.7,
    "max_inference_break_even_frame_count": 8,
    "max_training_break_even_frame_count": 16,
    "min_image_psnr_db": 50.0,
    "max_lpips_delta": 0.001,
    "max_image_p999_abs_error": 2.0 / 255.0,
    "max_loss_absolute_delta": 1.0e-5,
    "max_world_vjp_global_normalized_l2_error": 1.0e-5,
    "max_world_vjp_max_parameter_normalized_l2_error": 1.0e-5,
    "min_certified_stable_or_event_aligned_fraction": 0.98,
    "max_expensive_unresolved_fallback_fraction_exclusive": 0.02,
    "max_f32_continuous_to_sliced_reference_ratio": 0.40,
}
HEAVY_STRUCTURAL_KEYS = (
    "chart_count",
    "structural_atlas_record_count",
    "continuous_candidate_reference_count",
    "event_count",
    "coefficient_count",
)
PUBLICATION_METRIC_KEYS = (
    "image_psnr_db",
    "lpips_delta",
    "image_p999_abs_error",
    "loss_absolute_delta",
    "world_vjp_global_normalized_l2_error",
    "world_vjp_max_parameter_normalized_l2_error",
    *HEAVY_STRUCTURAL_KEYS,
    "summed_sliced_candidate_reference_count",
    "interaction_memory_bytes_excluding_outputs_residuals",
    "certified_stable_or_event_aligned_fraction",
    "expensive_unresolved_fallback_fraction",
)
REQUIRED_MECHANICAL_ROW_CHECKS = (
    "checkpoint_matches",
    "world_vjp_nonzero",
    "world_vjp_coverage_matches",
)


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_nonnegative(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def camera_program_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": CAMERA_PROGRAM_MODE,
        "compiler_chart_policy": COMPILER_CHART_POLICY,
        "multi_chart_gauge_compiler": MULTI_CHART_GAUGE_COMPILER,
        "path_scope": "bounded_open_path",
        "yaw_total_degrees": YAW_TOTAL_DEGREES,
        "yaw_start_degrees": -0.5 * YAW_TOTAL_DEGREES,
        "yaw_end_degrees": 0.5 * YAW_TOTAL_DEGREES,
        "sampling": "uniform_closed_interval",
        "frame_counts": list(FRAME_COUNTS),
        "image_size": list(IMAGE_SIZE),
    }


def camera_program_sha256() -> str:
    return canonical_json_sha256(camera_program_contract())


def _protocol_name(summary: Mapping[str, Any]) -> str | None:
    protocol = summary.get("protocol")
    return (
        str(protocol.get("name"))
        if isinstance(protocol, Mapping) and protocol.get("name") is not None
        else None
    )


def default_static_summary_path(protocol_name: str, seed: int) -> Path:
    return DEFAULT_STATIC_OUT_DIR / protocol_name / f"seed_{seed}" / "summary.json"


def validate_static_checkpoint_input(
    summary_path: Path,
    *,
    protocol_name: str,
    seed: int,
    expected_dataset_input_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the accepted static sweep and return its immutable input identity."""

    resolved_summary = summary_path.resolve()
    summary = single.load_json(resolved_summary)
    if (
        int(summary.get("schema_version", -1)) != 1
        or summary.get("status") != "accepted"
        or summary.get("publication_eligible") is not True
        or int(summary.get("seed", -1)) != seed
        or _protocol_name(summary) != protocol_name
    ):
        raise ValueError(
            "moving-camera input must be an accepted publication-eligible "
            "static frozen-world summary for the same protocol and seed"
        )
    common = summary.get("common_evidence_contract")
    if (
        not isinstance(common, Mapping)
        or common.get("dataset_input_identity")
        != dict(expected_dataset_input_identity)
    ):
        raise ValueError(
            "static frozen checkpoint raw dataset identity does not match "
            "the current Coffee Martini manifest inputs"
        )
    sweep = summary.get("frozen_world_replay_compiled_sweep")
    if not isinstance(sweep, Mapping):
        raise ValueError("static frozen summary is missing its canonical sweep")
    checkpoint = sweep.get("shared_checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("static frozen summary has no shared checkpoint")
    checkpoint_path = Path(str(checkpoint.get("path", ""))).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (ROOT / checkpoint_path).resolve()
    else:
        checkpoint_path = checkpoint_path.resolve()
    expected_checkpoint_path = (
        resolved_summary.parent / "world_tubes_frozen_final_state.pt"
    ).resolve()
    checkpoint_sha256 = checkpoint.get("sha256")
    world_state_sha256 = checkpoint.get("world_state_sha256")
    if (
        checkpoint_path != expected_checkpoint_path
        or checkpoint_path.name != "world_tubes_frozen_final_state.pt"
        or not checkpoint_path.is_file()
        or not _is_sha256(checkpoint_sha256)
        or checkpoint_sha256 != file_sha256(checkpoint_path)
        or int(checkpoint.get("bytes", -1)) != checkpoint_path.stat().st_size
        or not _is_sha256(world_state_sha256)
        or sweep.get("shared_checkpoint_file_sha256") != checkpoint_sha256
        or sweep.get("shared_world_state_sha256") != world_state_sha256
        or sweep.get("checkpoint_shared_across_rows") is not True
        or sweep.get("world_state_shared_across_rows") is not True
        or sweep.get("all_rows_accepted") is not True
    ):
        raise ValueError(
            "static frozen checkpoint file/world-state identity is invalid"
        )
    rows = sweep.get("rows")
    if (
        not isinstance(rows, list)
        or not rows
        or any(
            not isinstance(row, Mapping)
            or row.get("checkpoint") != checkpoint
            or row.get("world_state", {}).get("checkpoint_sha256")
            != world_state_sha256
            for row in rows
        )
    ):
        raise ValueError(
            "static frozen sweep did not use one identical checkpoint/world state"
        )
    source = summary.get("source")
    source_finish = summary.get("source_finish")
    if (
        not isinstance(source, Mapping)
        or dict(source) != dict(source_finish or {})
    ):
        raise ValueError("static frozen source provenance is missing or changed")
    single.require_clean_provenance(source)
    return {
        "schema_version": 1,
        "summary": single.display_path(resolved_summary),
        "summary_sha256": file_sha256(resolved_summary),
        "static_source": dict(source),
        "dataset_input_identity": dict(expected_dataset_input_identity),
        "checkpoint": {
            **dict(checkpoint),
            "path": str(checkpoint_path),
        },
    }


def _replace_command_value(command: list[str], flag: str, value: str) -> None:
    try:
        index = command.index(flag)
    except ValueError as error:
        raise ValueError(f"base paper command is missing {flag}") from error
    if index + 1 >= len(command):
        raise ValueError(f"base paper command has no value for {flag}")
    command[index + 1] = value


def build_command(
    protocol_path: Path,
    protocol,
    *,
    seed: int,
    out_dir: Path,
    checkpoint: Mapping[str, Any],
    device: str,
    allow_local_mps_execution: bool,
) -> list[str]:
    """Build the isolated checkpoint-only moving-camera child command."""

    command = static_frozen.build_command(
        protocol_path,
        protocol,
        seed=seed,
        out_dir=out_dir,
        device=device,
        max_frames=max(FRAME_COUNTS),
        allow_local_mps_execution=allow_local_mps_execution,
        frame_counts=FRAME_COUNTS,
        timing_warmups=TIMING_WARMUPS,
        timing_repeats=TIMING_REPEATS,
    )
    _replace_command_value(command, "--target-size", str(IMAGE_SIZE[0]))
    _replace_command_value(command, "--uvt-camera-projection", "legacy_pinhole")
    _replace_command_value(
        command,
        "--uvt-camera-sequence-mode",
        "projective_first_order",
    )
    command.extend(
        (
            "--frozen-world-camera-program-mode",
            CAMERA_PROGRAM_MODE,
            "--frozen-world-yaw-total-degrees",
            str(YAW_TOTAL_DEGREES),
            "--checkpoint",
            str(checkpoint["path"]),
            "--expected-checkpoint-sha256",
            str(checkpoint["sha256"]),
            "--expected-world-state-sha256",
            str(checkpoint["world_state_sha256"]),
        )
    )
    return command


def moving_camera_resource_estimate(
    *,
    checkpoint_bytes: int,
    host_physical_memory_bytes: int | None = None,
) -> dict[str, Any]:
    """Conservative checkpoint-only bound; it never relaxes the live gate."""

    host_bytes = (
        single.host_physical_memory_bytes()
        if host_physical_memory_bytes is None
        else int(host_physical_memory_bytes)
    )
    output_bytes = (
        max(FRAME_COUNTS)
        * IMAGE_SIZE[0]
        * IMAGE_SIZE[1]
        * 4
        * 4
    )
    # Reserve 4 GiB for Python/Metal/runtime state, sixteen checkpoint-sized
    # working copies for optimizer-free world/VJP temporaries, and twelve
    # output-sized arrays for paired routes, gradients, residuals, and staging.
    estimated_peak_bytes = (
        4 * GIB + 16 * int(checkpoint_bytes) + 12 * output_bytes
    )
    safety_limit_bytes = math.floor(0.60 * host_bytes)
    return {
        "schema_version": 1,
        "definition": (
            "checkpoint-only moving-camera conservative bound: 4 GiB runtime "
            "+ 16x checkpoint + 12x max-frame RGBA float output"
        ),
        "frame_counts": list(FRAME_COUNTS),
        "image_size": list(IMAGE_SIZE),
        "checkpoint_bytes": int(checkpoint_bytes),
        "output_bytes": output_bytes,
        "estimated_peak_bytes": estimated_peak_bytes,
        "host_physical_memory_bytes": host_bytes,
        "safety_limit_bytes": safety_limit_bytes,
        "high_risk": estimated_peak_bytes > safety_limit_bytes,
    }


def require_execution_resources(
    *,
    device: str,
    allow_local_mps_execution: bool,
    checkpoint_bytes: int,
) -> dict[str, Any]:
    if device.lower() != "mps":
        raise RuntimeError(
            "moving-camera World Tubes paper evidence requires the native MPS path"
        )
    if not allow_local_mps_execution:
        raise RuntimeError(
            "moving-camera MPS execution is fail-closed after the resource "
            "incident; explicit --allow-local-mps-execution is required"
        )
    estimate = moving_camera_resource_estimate(
        checkpoint_bytes=checkpoint_bytes
    )
    if estimate["high_risk"] is True:
        raise RuntimeError(
            "moving-camera checkpoint-only workload exceeds the 60% physical "
            "memory safety limit; use a larger quiet Apple-Silicon host"
        )
    live = single.live_resource_snapshot()
    single.require_live_resources(live)
    return {
        "estimate": estimate,
        "live_resources": live,
        "live_resource_thresholds": single.LIVE_RESOURCE_THRESHOLDS,
    }


def _timing_median(row: Mapping[str, Any], key: str) -> float:
    timing = row.get("timing_benchmark")
    if not isinstance(timing, Mapping):
        raise ValueError("moving-camera row timing benchmark is missing")
    summary = timing.get("summary_s")
    if not isinstance(summary, Mapping):
        raise ValueError("moving-camera row timing summaries are missing")
    metric = summary.get(key)
    if not isinstance(metric, Mapping) or not _finite_nonnegative(
        metric.get("median")
    ):
        raise ValueError(f"moving-camera timing median {key} is invalid")
    return float(metric["median"])


def validate_loaded_checkpoint_identity(
    loaded: Any,
    *,
    expected: Mapping[str, Any],
) -> None:
    """Bind the reconstructed child model to the accepted static checkpoint."""

    if not isinstance(loaded, Mapping):
        raise ValueError("moving-camera loaded checkpoint identity is missing")
    identity_keys = (
        "path",
        "sha256",
        "bytes",
        "world_state_sha256",
        "representation",
        "frame_count",
        "active_tube_count",
        "tube_count",
        "alpha_mode",
        "amplitude_convention",
        "min_precision_xy",
        "min_lambda_t",
        "parameter_names",
    )
    if (
        any(loaded.get(key) != expected.get(key) for key in identity_keys)
        or loaded.get("loaded_from_input_checkpoint") is not True
    ):
        raise ValueError(
            "moving-camera child did not reconstruct the exact accepted "
            "static checkpoint"
        )
    path = Path(str(loaded["path"])).resolve()
    if (
        not path.is_file()
        or file_sha256(path) != expected["sha256"]
        or path.stat().st_size != int(expected["bytes"])
    ):
        raise ValueError("moving-camera loaded checkpoint file drifted")


def _positive_ratio(numerator: float, denominator: float, label: str) -> float:
    if denominator <= 0.0:
        raise ValueError(f"{label} denominator must be positive")
    value = numerator / denominator
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} is invalid")
    return value


def _structural_growth_ratio(
    numerator: float,
    denominator: float,
) -> tuple[float | None, bool]:
    """Return an invariance ratio without rejecting a legitimate zero baseline."""

    if denominator == 0.0:
        return (1.0, False) if numerator == 0.0 else (None, True)
    return numerator / denominator, False


def derive_publication_gate(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_frame = {int(row["frame_count"]): row for row in rows}
    if set(by_frame) != set(FRAME_COUNTS) or len(by_frame) != len(rows):
        raise ValueError("moving-camera publication rows must be F=8,16,32,64")
    metrics: dict[int, Mapping[str, Any]] = {}
    for frame_count, row in by_frame.items():
        raw = row.get("publication_metrics")
        if (
            not isinstance(raw, Mapping)
            or set(PUBLICATION_METRIC_KEYS) - set(raw)
            or any(not _finite_nonnegative(raw[key]) for key in PUBLICATION_METRIC_KEYS)
        ):
            raise ValueError(
                f"moving-camera F={frame_count} publication metrics are incomplete"
            )
        metrics[frame_count] = raw

    structural_ratios: dict[str, float | None] = {}
    structural_components_grew_from_zero = []
    for key in HEAVY_STRUCTURAL_KEYS:
        ratio, grew_from_zero = _structural_growth_ratio(
            float(metrics[32][key]),
            float(metrics[8][key]),
        )
        structural_ratios[key] = ratio
        if grew_from_zero:
            structural_components_grew_from_zero.append(key)
    max_structural_ratio = (
        None
        if structural_components_grew_from_zero
        else max(
            value
            for value in structural_ratios.values()
            if value is not None
        )
    )
    interaction_memory_ratio = _positive_ratio(
        float(
            metrics[32][
                "interaction_memory_bytes_excluding_outputs_residuals"
            ]
        ),
        float(
            metrics[8][
                "interaction_memory_bytes_excluding_outputs_residuals"
            ]
        ),
        "F32/F8 interaction memory",
    )
    row32 = by_frame[32]
    replay_forward = _timing_median(row32, "replay_total_forward")
    replay_reverse = _timing_median(row32, "replay_total_backward")
    compiled_compile = _timing_median(row32, "compiled_atlas_compile")
    compiled_forward = _timing_median(row32, "compiled_total_forward")
    compiled_reverse = _timing_median(row32, "compiled_total_backward")
    speedups = {
        "forward": _positive_ratio(
            replay_forward,
            compiled_forward,
            "F32 forward speedup",
        ),
        "reverse": _positive_ratio(
            replay_reverse,
            compiled_reverse,
            "F32 reverse speedup",
        ),
        "total": _positive_ratio(
            replay_forward + replay_reverse,
            compiled_compile + compiled_forward + compiled_reverse,
            "F32 total speedup",
        ),
    }

    inference_amortized_ratios = {}
    training_amortized_ratios = {}
    for frame_count, row in by_frame.items():
        compile_time = _timing_median(row, "compiled_atlas_compile")
        replay_f = _timing_median(row, "replay_total_forward")
        replay_b = _timing_median(row, "replay_total_backward")
        compiled_f = _timing_median(row, "compiled_total_forward")
        compiled_b = _timing_median(row, "compiled_total_backward")
        inference_amortized_ratios[frame_count] = _positive_ratio(
            compile_time + compiled_f,
            replay_f,
            f"F={frame_count} inference amortized ratio",
        )
        training_amortized_ratios[frame_count] = _positive_ratio(
            compile_time + compiled_f + compiled_b,
            replay_f + replay_b,
            f"F={frame_count} training amortized ratio",
        )
    inference_break_even = next(
        (
            frame_count
            for frame_count in FRAME_COUNTS
            if inference_amortized_ratios[frame_count] <= 1.0
        ),
        None,
    )
    training_break_even = next(
        (
            frame_count
            for frame_count in FRAME_COUNTS
            if training_amortized_ratios[frame_count] <= 1.0
        ),
        None,
    )
    continuous_to_sliced_f32 = _positive_ratio(
        float(metrics[32]["continuous_candidate_reference_count"]),
        float(metrics[32]["summed_sliced_candidate_reference_count"]),
        "F32 continuous/sliced references",
    )

    parity_checks = {
        str(frame_count): {
            "psnr": float(raw["image_psnr_db"])
            >= PUBLICATION_THRESHOLDS["min_image_psnr_db"],
            "lpips": float(raw["lpips_delta"])
            <= PUBLICATION_THRESHOLDS["max_lpips_delta"],
            "p999": float(raw["image_p999_abs_error"])
            <= PUBLICATION_THRESHOLDS["max_image_p999_abs_error"],
        }
        for frame_count, raw in metrics.items()
    }
    optimization_parity_checks = {
        str(frame_count): {
            "loss": float(raw["loss_absolute_delta"])
            <= PUBLICATION_THRESHOLDS["max_loss_absolute_delta"],
            "world_vjp_global": float(
                raw["world_vjp_global_normalized_l2_error"]
            )
            <= PUBLICATION_THRESHOLDS[
                "max_world_vjp_global_normalized_l2_error"
            ],
            "world_vjp_per_parameter": float(
                raw["world_vjp_max_parameter_normalized_l2_error"]
            )
            <= PUBLICATION_THRESHOLDS[
                "max_world_vjp_max_parameter_normalized_l2_error"
            ],
        }
        for frame_count, raw in metrics.items()
    }
    coverage_checks = {
        str(frame_count): {
            "stable_or_event_aligned": float(
                raw["certified_stable_or_event_aligned_fraction"]
            )
            >= PUBLICATION_THRESHOLDS[
                "min_certified_stable_or_event_aligned_fraction"
            ],
            "expensive_fallback": float(
                raw["expensive_unresolved_fallback_fraction"]
            )
            < PUBLICATION_THRESHOLDS[
                "max_expensive_unresolved_fallback_fraction_exclusive"
            ],
        }
        for frame_count, raw in metrics.items()
    }
    checks = {
        "all_rows_timing_publication_ready": all(
            row["timing_benchmark"].get("publication_ready") is True
            for row in rows
        ),
        "heavy_structural_work_t32_t8": (
            max_structural_ratio is not None
            and max_structural_ratio
            <= PUBLICATION_THRESHOLDS[
                "max_heavy_structural_ratio_t32_t8"
            ]
        ),
        "interaction_memory_t32_t8": interaction_memory_ratio
        <= PUBLICATION_THRESHOLDS[
            "max_interaction_memory_ratio_t32_t8"
        ],
        "f32_forward_speedup": speedups["forward"]
        >= PUBLICATION_THRESHOLDS["min_f32_forward_speedup"],
        "f32_reverse_speedup": speedups["reverse"]
        >= PUBLICATION_THRESHOLDS["min_f32_reverse_speedup"],
        "f32_total_speedup": speedups["total"]
        >= PUBLICATION_THRESHOLDS["min_f32_total_speedup"],
        "inference_break_even_by_f8": inference_break_even is not None
        and inference_break_even
        <= PUBLICATION_THRESHOLDS["max_inference_break_even_frame_count"],
        "training_break_even_by_f16": training_break_even is not None
        and training_break_even
        <= PUBLICATION_THRESHOLDS["max_training_break_even_frame_count"],
        "all_rows_variable_camera_parity": all(
            all(row_checks.values()) for row_checks in parity_checks.values()
        ),
        "all_rows_loss_parity": all(
            row_checks["loss"]
            for row_checks in optimization_parity_checks.values()
        ),
        "all_rows_world_vjp_parity": all(
            row_checks["world_vjp_global"]
            and row_checks["world_vjp_per_parameter"]
            for row_checks in optimization_parity_checks.values()
        ),
        "all_rows_stable_coverage": all(
            all(row_checks.values()) for row_checks in coverage_checks.values()
        ),
        "f32_continuous_to_sliced_references": continuous_to_sliced_f32
        <= PUBLICATION_THRESHOLDS[
            "max_f32_continuous_to_sliced_reference_ratio"
        ],
    }
    return {
        "schema_version": 1,
        "thresholds": dict(PUBLICATION_THRESHOLDS),
        "measurements": {
            "heavy_structural_component_ratios_t32_t8": structural_ratios,
            "heavy_structural_work_ratio_t32_t8": max_structural_ratio,
            "heavy_structural_components_grew_from_zero": (
                structural_components_grew_from_zero
            ),
            "heavy_structural_work_grew_from_zero": bool(
                structural_components_grew_from_zero
            ),
            "interaction_memory_ratio_t32_t8": interaction_memory_ratio,
            "f32_speedups": speedups,
            "inference_amortized_ratio_by_frame_count": {
                str(key): value
                for key, value in inference_amortized_ratios.items()
            },
            "training_amortized_ratio_by_frame_count": {
                str(key): value
                for key, value in training_amortized_ratios.items()
            },
            "inference_break_even_frame_count": inference_break_even,
            "training_break_even_frame_count": training_break_even,
            "f32_continuous_to_sliced_reference_ratio": (
                continuous_to_sliced_f32
            ),
            "parity_checks_by_frame_count": parity_checks,
            "optimization_parity_checks_by_frame_count": (
                optimization_parity_checks
            ),
            "coverage_checks_by_frame_count": coverage_checks,
        },
        "checks": checks,
        "accepted": all(checks.values()),
        "failure_reasons": [
            name for name, accepted in checks.items() if not accepted
        ],
    }


def validate_child_progress(
    progress_path: Path,
    *,
    checkpoint: Mapping[str, Any],
) -> Mapping[str, Any]:
    progress = single.load_json(progress_path)
    validate_loaded_checkpoint_identity(
        progress.get("checkpoint"),
        expected=checkpoint,
    )
    row_artifacts = progress.get("row_artifacts")
    row_progress = progress.get("rows")
    if (
        int(progress.get("schema_version", -1)) != 1
        or progress.get("status") != "complete"
        or progress.get("camera_program_sha256")
        != camera_program_sha256()
        or progress.get("world_state_sha256")
        != checkpoint["world_state_sha256"]
        or list(progress.get("requested_frame_counts", ()))
        != list(FRAME_COUNTS)
        or list(progress.get("resolved_frame_counts", ()))
        != list(FRAME_COUNTS)
        or list(progress.get("completed_frame_counts", ()))
        != list(FRAME_COUNTS)
        or list(progress.get("decoded_image_size", ()))
        != list(IMAGE_SIZE)
        or list(progress.get("render_image_size", ()))
        != list(IMAGE_SIZE)
        or progress.get("target_resize_mode") != TARGET_RESIZE_MODE
        or progress.get("compiler_chart_policy")
        != COMPILER_CHART_POLICY
        or progress.get("multi_chart_gauge_compiler")
        is not MULTI_CHART_GAUGE_COMPILER
        or not _is_sha256(progress.get("camera_sequence_sha256"))
        or progress.get(
            "all_rows_selected_time_slice_parity_accepted"
        )
        is not True
        or progress.get("all_rows_mechanically_valid") is not True
        or not isinstance(row_artifacts, list)
        or len(row_artifacts) != len(FRAME_COUNTS)
        or not isinstance(row_progress, list)
        or len(row_progress) != len(FRAME_COUNTS)
    ):
        raise ValueError(
            "moving-camera child progress is incomplete or identity-drifted"
        )
    for frame_count, artifact, row in zip(
        FRAME_COUNTS,
        row_artifacts,
        row_progress,
        strict=True,
    ):
        if not isinstance(artifact, Mapping) or not isinstance(row, Mapping):
            raise ValueError("moving-camera per-row progress is invalid")
        path = Path(str(artifact.get("path", ""))).resolve()
        if (
            int(artifact.get("frame_count", 0)) != frame_count
            or not path.is_file()
            or artifact.get("sha256") != file_sha256(path)
            or int(row.get("frame_count", 0)) != frame_count
            or row.get("mechanically_valid") is not True
            or row.get("camera_program_sha256")
            != camera_program_sha256()
            or row.get("compiler_chart_policy")
            != COMPILER_CHART_POLICY
            or row.get("multi_chart_gauge_compiler")
            is not MULTI_CHART_GAUGE_COMPILER
            or row.get("camera_sequence_sha256")
            != progress["camera_sequence_sha256"]
            or row.get("world_state_sha256")
            != checkpoint["world_state_sha256"]
            or row.get("artifact_sha256") != artifact["sha256"]
        ):
            raise ValueError(
                f"moving-camera durable F={frame_count} row artifact drifted"
            )
    return progress


def validate_report(
    report: Mapping[str, Any],
    *,
    protocol,
    seed: int,
    checkpoint: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, Any], Mapping[str, Any]]:
    meta = report.get("meta")
    lane = report.get("star_uvt")
    if not isinstance(meta, Mapping) or not isinstance(lane, Mapping):
        raise ValueError("moving-camera report is missing World Tubes data")
    if (
        int(meta.get("seed", -1)) != seed
        or int(meta.get("frame_count", 0)) != protocol.dataset.frame_count
        or tuple(meta.get("train_cameras", ()))
        != protocol.dataset.train_cameras
        or tuple(meta.get("heldout_cameras", ()))
        != protocol.dataset.heldout_cameras
        or meta.get("only_lane") != "world_tubes"
        or meta.get("frozen_world_replay_compiled") is not True
        or int(meta.get("frozen_world_max_frames", 0))
        != max(FRAME_COUNTS)
        or meta.get("frozen_world_frame_counts") != list(FRAME_COUNTS)
        or int(meta.get("frozen_world_timing_warmups", -1))
        != TIMING_WARMUPS
        or int(meta.get("frozen_world_timing_repeats", 0))
        != TIMING_REPEATS
        or meta.get("uvt_world_representation") != "legacy_tube"
        or meta.get("uvt_alpha_mode") != "peak_splat"
        or meta.get("uvt_render_backend") != "metal_tile"
        or meta.get("uvt_camera_projection") != "legacy_pinhole"
        or meta.get("uvt_camera_sequence_mode")
        != "projective_first_order"
        or meta.get("frozen_world_camera_program_mode")
        != CAMERA_PROGRAM_MODE
        or meta.get("frozen_world_compiler_chart_policy")
        != COMPILER_CHART_POLICY
        or meta.get("frozen_world_multi_chart_gauge_compiler")
        is not MULTI_CHART_GAUGE_COMPILER
        or float(meta.get("frozen_world_yaw_total_degrees", math.nan))
        != YAW_TOTAL_DEGREES
        or meta.get("frozen_world_checkpoint_only") is not True
        or Path(str(meta.get("frozen_world_input_checkpoint", ""))).resolve()
        != Path(str(checkpoint["path"])).resolve()
        or meta.get("frozen_world_expected_checkpoint_sha256")
        != checkpoint["sha256"]
        or meta.get("frozen_world_expected_world_state_sha256")
        != checkpoint["world_state_sha256"]
        or meta.get("frozen_world_target_resize_mode")
        != TARGET_RESIZE_MODE
    ):
        raise ValueError("moving-camera outer report contract drifted")
    native = meta.get("star_uvt_native_extension")
    if not isinstance(native, Mapping):
        raise ValueError("moving-camera native extension identity is missing")
    single.validate_native_extension_identity(native)
    route_native = meta.get("route_native_extension")
    single.validate_route_native_extension_identity(
        "world_tubes",
        route_native,
    )
    if route_native != native:
        raise ValueError("moving-camera route-native identity drifted")
    for name, schema_version in (
        (
            "paper_dataset_bundle",
            single.PAPER_DATASET_BUNDLE_SCHEMA_VERSION,
        ),
        ("paper_evaluator", single.PAPER_EVALUATOR_SCHEMA_VERSION),
        ("paper_runtime", single.PAPER_RUNTIME_SCHEMA_VERSION),
    ):
        single.validate_hashed_contract(
            f"moving-camera {name}",
            meta.get(name),
            schema_version=schema_version,
        )
    decoded_bundle = meta["paper_dataset_bundle"]
    if (
        decoded_bundle.get("sample_id") != protocol.dataset.sample_id
        or decoded_bundle.get("image_size") != list(IMAGE_SIZE)
        or decoded_bundle.get("frame_count")
        != protocol.dataset.frame_count
        or decoded_bundle.get("train_camera_names")
        != list(protocol.dataset.train_cameras)
        or decoded_bundle.get("heldout_camera_names")
        != list(protocol.dataset.heldout_cameras)
    ):
        raise ValueError(
            "moving-camera decoded Coffee Martini bundle identity drifted"
        )
    if meta["paper_evaluator"] != single.paper_evaluator_contract():
        raise ValueError("moving-camera evaluator is not canonical")

    sweep = lane.get("frozen_world_replay_compiled_sweep")
    if not isinstance(sweep, Mapping):
        raise ValueError("moving-camera sweep is missing")
    loaded_checkpoint = sweep.get("shared_checkpoint")
    validate_loaded_checkpoint_identity(
        loaded_checkpoint,
        expected=checkpoint,
    )
    program = camera_program_contract()
    if (
        int(sweep.get("schema_version", -1)) != 1
        or sweep.get("status") != "complete"
        or sweep.get("camera_program") != program
        or sweep.get("camera_program_sha256")
        != canonical_json_sha256(program)
        or not _is_sha256(sweep.get("camera_sequence_sha256"))
        or list(sweep.get("requested_frame_counts", ()))
        != list(FRAME_COUNTS)
        or list(sweep.get("resolved_frame_counts", ()))
        != list(FRAME_COUNTS)
        or list(sweep.get("image_size", ())) != list(IMAGE_SIZE)
        or list(sweep.get("decoded_image_size", ())) != list(IMAGE_SIZE)
        or list(sweep.get("render_image_size", ())) != list(IMAGE_SIZE)
        or sweep.get("target_resize_mode") != TARGET_RESIZE_MODE
        or sweep.get("target_semantics") != TARGET_SEMANTICS
        or sweep.get("parity_metric_semantics")
        != PARITY_METRIC_SEMANTICS
        or sweep.get("compiler_chart_policy")
        != COMPILER_CHART_POLICY
        or sweep.get("multi_chart_gauge_compiler")
        is not MULTI_CHART_GAUGE_COMPILER
        or int(sweep.get("timing_benchmark_warmups", -1))
        != TIMING_WARMUPS
        or int(sweep.get("timing_benchmark_repeats", 0))
        != TIMING_REPEATS
        or sweep.get("shared_checkpoint_file_sha256")
        != checkpoint["sha256"]
        or sweep.get("shared_world_state_sha256")
        != checkpoint["world_state_sha256"]
        or sweep.get("checkpoint_shared_across_rows") is not True
        or sweep.get("world_state_shared_across_rows") is not True
        or sweep.get("checkpoint_loaded_not_trained") is not True
        or sweep.get("all_rows_mechanically_valid") is not True
        or sweep.get("all_rows_timing_publication_ready") is not True
        or sweep.get("evidence_complete") is not True
        or int(sweep.get("full_dataset_frame_count", 0))
        != protocol.dataset.frame_count
        or int(sweep.get("primary_requested_frame_count", -1))
        != max(FRAME_COUNTS)
        or int(sweep.get("primary_resolved_frame_count", 0))
        != max(FRAME_COUNTS)
        or sweep.get("temporal_sampling")
        != "ordered_full_interval_integer_lattice_v1"
        or sweep.get(
            "all_rows_selected_time_slice_parity_accepted"
        )
        is not True
    ):
        raise ValueError("moving-camera sweep identity or fixed contract drifted")
    rows = sweep.get("rows")
    if not isinstance(rows, list) or len(rows) != len(FRAME_COUNTS):
        raise ValueError("moving-camera sweep rows are missing")
    for row, frame_count in zip(rows, FRAME_COUNTS, strict=True):
        if (
            not isinstance(row, Mapping)
            or int(row.get("schema_version", -1)) != 2
            or row.get("status") != "complete"
            or row.get("mechanically_valid") is not True
        ):
            raise ValueError(
                f"moving-camera F={frame_count} mechanical validity failed"
            )
        if (
            int(row.get("frame_count", 0)) != frame_count
            or list(row.get("image_size", ())) != list(IMAGE_SIZE)
            or list(row.get("decoded_image_size", ())) != list(IMAGE_SIZE)
            or list(row.get("render_image_size", ())) != list(IMAGE_SIZE)
            or row.get("target_resize_mode") != sweep["target_resize_mode"]
            or row.get("target_semantics") != TARGET_SEMANTICS
            or row.get("parity_metric_semantics")
            != PARITY_METRIC_SEMANTICS
            or row.get("compiler_chart_policy")
            != COMPILER_CHART_POLICY
            or row.get("multi_chart_gauge_compiler")
            is not MULTI_CHART_GAUGE_COMPILER
            or row.get("camera_program_mode") != CAMERA_PROGRAM_MODE
            or row.get("camera_program_sha256")
            != sweep["camera_program_sha256"]
            or row.get("camera_sequence_sha256")
            != sweep["camera_sequence_sha256"]
            or row.get("checkpoint") != loaded_checkpoint
        ):
            raise ValueError(
                f"moving-camera F={frame_count} identity or resolution drifted"
            )
        checks = row.get("checks")
        mechanical_checks = row.get("mechanical_checks")
        if (
            not isinstance(checks, Mapping)
            or any(
                checks.get(key) is not True
                for key in REQUIRED_MECHANICAL_ROW_CHECKS
            )
            or not isinstance(mechanical_checks, Mapping)
            or not mechanical_checks
            or any(value is not True for value in mechanical_checks.values())
        ):
            raise ValueError(
                f"moving-camera F={frame_count} world-state/gradient "
                "mechanical contract failed"
            )
        world_state = row.get("world_state")
        if (
            not isinstance(world_state, Mapping)
            or world_state.get("checkpoint_sha256")
            != checkpoint["world_state_sha256"]
            or world_state.get("before_routes_sha256")
            != checkpoint["world_state_sha256"]
            or world_state.get("after_replay_sha256")
            != checkpoint["world_state_sha256"]
            or world_state.get("after_compiled_sha256")
            != checkpoint["world_state_sha256"]
            or world_state.get("matches_checkpoint") is not True
        ):
            raise ValueError(
                f"moving-camera F={frame_count} changed the frozen world"
            )
        single.validate_frozen_world_evidence(
            row,
            expected_frames=frame_count,
            expected_full_frames=protocol.dataset.frame_count,
            expected_image_size=IMAGE_SIZE,
            expected_heldout_camera=protocol.dataset.heldout_cameras[0],
            expected_active_tubes=protocol.final_stage.primitive_count,
        )
        selected_time_slice_parity = row.get(
            "selected_time_slice_parity"
        )
        if (
            not isinstance(selected_time_slice_parity, Mapping)
            or selected_time_slice_parity.get("status") != "complete"
            or selected_time_slice_parity.get("accepted") is not True
        ):
            raise ValueError(
                f"moving-camera F={frame_count} selected-time slice parity failed"
            )
        static_frozen.validate_selected_time_slice_parity(
            selected_time_slice_parity,
            row=row,
        )
        static_frozen.validate_timing_benchmark(
            row.get("timing_benchmark", {}),
            frame_count=frame_count,
            resident_chunk_frames=int(
                row.get("contract", {}).get("resident_chunk_frames", 0)
            ),
            legacy_timing=row.get("timing_s", {}),
            expected_warmups=TIMING_WARMUPS,
            expected_repeats=TIMING_REPEATS,
        )
    gate = derive_publication_gate(rows)
    return sweep, gate, native


def validate_execution_identity(
    identity: Mapping[str, Any],
    *,
    protocol_path: Path,
    command: list[str],
    report_path: Path,
    progress_path: Path,
    expected_source: Mapping[str, Any],
    expected_dataset_input_identity: Mapping[str, Any],
    expected_static_input: Mapping[str, Any],
    expected_native_extension: Mapping[str, Any],
) -> None:
    if (
        int(identity.get("schema_version", -1)) != 1
        or identity.get("protocol_sha256") != file_sha256(protocol_path)
        or list(identity.get("command", ())) != command
        or not report_path.is_file()
        or identity.get("comparison_report_sha256")
        != file_sha256(report_path)
        or not progress_path.is_file()
        or identity.get("progress_sha256") != file_sha256(progress_path)
        or identity.get("camera_program") != camera_program_contract()
        or identity.get("camera_program_sha256")
        != camera_program_sha256()
        or identity.get("compiler_chart_policy")
        != COMPILER_CHART_POLICY
        or identity.get("multi_chart_gauge_compiler")
        is not MULTI_CHART_GAUGE_COMPILER
        or identity.get("dataset_input_identity")
        != dict(expected_dataset_input_identity)
        or identity.get("static_checkpoint_input")
        != dict(expected_static_input)
        or identity.get("star_uvt_native_extension")
        != dict(expected_native_extension)
    ):
        raise ValueError("moving-camera execution identity drifted")
    start = identity.get("source_start")
    finish = identity.get("source_finish")
    if (
        not isinstance(start, Mapping)
        or dict(start) != dict(finish or {})
        or dict(start) != dict(expected_source)
    ):
        raise ValueError("moving-camera source identity drifted")
    single.require_clean_provenance(start)
    single.validate_native_extension_identity(expected_native_extension)
    resources = identity.get("resources_at_launch")
    estimate = resources.get("estimate") if isinstance(resources, Mapping) else None
    if not isinstance(estimate, Mapping):
        raise ValueError("moving-camera execution safety receipt is missing")
    single.validate_process_memory(
        identity.get("process_memory"),
        expected_rss_limit_bytes=int(estimate.get("safety_limit_bytes", 0)),
    )


def _failure(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(
        json.dumps(
            serialize_config_value(dict(payload)),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_attempt(
    path: Path,
    base: Mapping[str, Any],
    *,
    status: str,
    phase: str,
    **values: Any,
) -> None:
    write_json_atomic(
        path,
        {
            **dict(base),
            "status": status,
            "phase": phase,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **values,
        },
    )


def moving_camera_wandb_contract(
    report: Mapping[str, Any],
    *,
    protocol,
    seed: int,
    execution_source: Mapping[str, Any],
    gate: Mapping[str, Any],
    static_input: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the immutable identity inputs for the route-parity W&B run."""

    source_digest = canonical_json_sha256(dict(execution_source))
    report_digest = canonical_json_sha256(dict(report))
    config = {
        "protocol": protocol.as_dict(),
        "seed": seed,
        "camera_program": camera_program_contract(),
        "camera_program_sha256": camera_program_sha256(),
        "compiler_chart_policy": COMPILER_CHART_POLICY,
        "multi_chart_gauge_compiler": MULTI_CHART_GAUGE_COMPILER,
        "target_semantics": TARGET_SEMANTICS,
        "parity_metric_semantics": PARITY_METRIC_SEMANTICS,
        "publication_thresholds": PUBLICATION_THRESHOLDS,
        "publication_gate": dict(gate),
        "source": dict(execution_source),
        "source_digest": source_digest,
        "comparison_report_sha256": report_digest,
        "static_checkpoint_input": dict(static_input),
        "paper_dataset_bundle": report["meta"]["paper_dataset_bundle"],
        "paper_evaluator": report["meta"]["paper_evaluator"],
        "paper_runtime": report["meta"]["paper_runtime"],
        "star_uvt_native_extension": report["meta"][
            "star_uvt_native_extension"
        ],
    }
    config_digest = canonical_json_sha256(config)
    run_id = "pm" + hashlib.sha1(
        (
            f"{protocol.name}:{seed}:{camera_program_sha256()}:"
            f"{source_digest}:{report_digest}"
        ).encode("utf-8")
    ).hexdigest()[:12]
    return {
        "config": config,
        "config_digest": config_digest,
        "source_digest": source_digest,
        "report_digest": report_digest,
        "run_id": run_id,
    }


def moving_camera_wandb_log(
    report: Mapping[str, Any],
    *,
    protocol,
    seed: int,
    report_dir: Path,
    wandb_mode: str,
    execution_source: Mapping[str, Any],
    gate: Mapping[str, Any],
    static_input: Mapping[str, Any],
) -> dict[str, Any]:
    """Log only same-world route-parity evidence, never scene-quality claims."""

    import wandb

    contract = moving_camera_wandb_contract(
        report,
        protocol=protocol,
        seed=seed,
        execution_source=execution_source,
        gate=gate,
        static_input=static_input,
    )
    config = contract["config"]
    config_digest = contract["config_digest"]
    source_digest = contract["source_digest"]
    report_digest = contract["report_digest"]
    run_id = contract["run_id"]
    identity_path = report_dir / "wandb_identity.json"
    if identity_path.is_file():
        identity = single.load_json(identity_path)
        single.validate_wandb_identity(
            identity,
            run_id=run_id,
            mode=wandb_mode,
            source_digest=source_digest,
            report_digest=report_digest,
            config_digest=config_digest,
        )
        return identity
    run = wandb.init(
        project="dynaworld",
        name=(
            f"paper-{protocol.name}-world-tubes-moving-camera-seed{seed}"
        ),
        tags=[
            "paper-ablation-v2",
            "world-tubes-moving-camera",
            "same-world-route-parity",
            single.paper_scene_tag(protocol),
            protocol.name,
            f"seed-{seed}",
        ],
        mode=wandb_mode,
        id=run_id,
        resume="never",
        config=config,
        settings=wandb.Settings(disable_git=True, disable_code=True),
        reinit="finish_previous",
    )
    measurements = gate["measurements"]
    payload: dict[str, int | float] = {
        "publication/accepted": int(bool(gate["accepted"])),
        **{
            f"publication/check_{name}": int(bool(value))
            for name, value in gate["checks"].items()
        },
        "publication/heavy_structural_work_grew_from_zero": int(
            bool(measurements["heavy_structural_work_grew_from_zero"])
        ),
        "publication/interaction_memory_ratio_t32_t8": measurements[
            "interaction_memory_ratio_t32_t8"
        ],
        "publication/f32_forward_speedup": measurements["f32_speedups"][
            "forward"
        ],
        "publication/f32_reverse_speedup": measurements["f32_speedups"][
            "reverse"
        ],
        "publication/f32_total_speedup": measurements["f32_speedups"][
            "total"
        ],
        "publication/f32_continuous_to_sliced_reference_ratio": measurements[
            "f32_continuous_to_sliced_reference_ratio"
        ],
    }
    if measurements["heavy_structural_work_ratio_t32_t8"] is not None:
        payload[
            "publication/heavy_structural_work_ratio_t32_t8"
        ] = float(measurements["heavy_structural_work_ratio_t32_t8"])
    for row in report["star_uvt"]["frozen_world_replay_compiled_sweep"][
        "rows"
    ]:
        frame_count = int(row["frame_count"])
        payload.update(
            {
                f"f{frame_count}/parity_{key}": float(value)
                for key, value in row["publication_metrics"].items()
                if isinstance(value, (int, float))
            }
        )
        payload.update(
            {
                f"f{frame_count}/timing_median_{key}": float(
                    value["median"]
                )
                for key, value in row["timing_benchmark"][
                    "summary_s"
                ].items()
            }
        )
    run.log(payload, step=0)
    run_dir = str(run.dir)
    actual_run_id = str(run.id)
    remote_identity = {
        "entity": None if run.entity is None else str(run.entity),
        "project": None if run.project is None else str(run.project),
        "url": None if run.url is None else str(run.url),
        "finish_called": False,
    }
    run.finish()
    remote_identity["finish_called"] = True
    identity = {
        "schema_version": 1,
        "project": "dynaworld",
        "name": (
            f"paper-{protocol.name}-world-tubes-moving-camera-seed{seed}"
        ),
        "mode": wandb_mode,
        "run_id": actual_run_id,
        "run_dir": run_dir,
        "source_digest": source_digest,
        "comparison_report_sha256": report_digest,
        "config_sha256": config_digest,
        "remote_identity": remote_identity,
        "run_file": single.wandb_file_identity(run_dir, actual_run_id),
    }
    single.validate_wandb_remote_identity(identity, mode=wandb_mode)
    write_json_atomic(identity_path, identity)
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the isolated frozen learned-world bounded moving-camera "
            "World Tubes publication gate."
        )
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--static-summary", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline"),
        default="online",
    )
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--require-clean-source", action="store_true")
    parser.add_argument("--allow-local-mps-execution", action="store_true")
    args = parser.parse_args()

    if args.execute and args.wandb_mode != "online":
        raise ValueError(
            "moving-camera publication execution requires --wandb-mode online"
        )
    if args.seed != 17:
        raise ValueError(
            "the moving-camera tenth paper job is frozen to Coffee Martini seed 17"
        )
    protocol_path = single.resolve_root_path(args.protocol)
    protocol = resolve_paper_training_protocol(load_config_file(protocol_path))
    if (
        protocol.name != "coffee_martini_full_300f_progressive_512_v1"
        or protocol.dataset.sample_id
        != "neural3d_coffee_martini_train_cam04_cam09_holdout_cam06_full_300f"
    ):
        raise ValueError(
            "the moving-camera tenth paper job is frozen to the accepted "
            "Coffee Martini full-300-frame protocol"
        )
    static_summary_path = (
        single.resolve_root_path(args.static_summary)
        if args.static_summary is not None
        else default_static_summary_path(protocol.name, args.seed)
    )
    out_root = single.resolve_root_path(args.out_dir)
    run_dir = out_root / protocol.name / f"seed_{args.seed}"
    report_path = run_dir / "comparison_report.json"
    progress_path = run_dir / "frozen_world_moving_camera_progress.json"
    identity_path = run_dir / "execution_identity.json"
    summary_path = run_dir / "summary.json"
    attempt_path = run_dir / "moving_camera_attempt.json"

    dry_run = {
        "schema_version": SCHEMA_VERSION,
        "status": "dry_run",
        "protocol": protocol.as_dict(),
        "seed": args.seed,
        "frame_counts": list(FRAME_COUNTS),
        "image_size": list(IMAGE_SIZE),
        "target_semantics": TARGET_SEMANTICS,
        "parity_metric_semantics": PARITY_METRIC_SEMANTICS,
        "compiler_chart_policy": COMPILER_CHART_POLICY,
        "multi_chart_gauge_compiler": MULTI_CHART_GAUGE_COMPILER,
        "timing_warmups": TIMING_WARMUPS,
        "timing_repeats": TIMING_REPEATS,
        "camera_program": camera_program_contract(),
        "camera_program_sha256": camera_program_sha256(),
        "publication_thresholds": PUBLICATION_THRESHOLDS,
        "static_summary": single.display_path(static_summary_path),
        "out_dir": single.display_path(run_dir),
        "clean_source_policy": "always_required_for_execute",
        "resource_policy": (
            "explicit MPS opt-in; conservative 60% estimate; live memory, "
            "swap, disk, and load gates; no high-risk override"
        ),
    }
    if not args.execute:
        print(
            json.dumps(
                serialize_config_value(dry_run),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return

    source_start = single.source_provenance()
    single.require_clean_provenance(source_start)
    manifest_validation = single.validate_manifest(protocol)
    static_input = validate_static_checkpoint_input(
        static_summary_path,
        protocol_name=protocol.name,
        seed=args.seed,
        expected_dataset_input_identity=manifest_validation["input_identity"],
    )
    checkpoint = static_input["checkpoint"]
    command = build_command(
        protocol_path,
        protocol,
        seed=args.seed,
        out_dir=run_dir,
        checkpoint=checkpoint,
        device=args.device,
        allow_local_mps_execution=args.allow_local_mps_execution,
    )
    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    attempt_base = {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "protocol": protocol.name,
        "protocol_sha256": file_sha256(protocol_path),
        "seed": args.seed,
        "command": command,
        "camera_program": camera_program_contract(),
        "camera_program_sha256": camera_program_sha256(),
        "compiler_chart_policy": COMPILER_CHART_POLICY,
        "multi_chart_gauge_compiler": MULTI_CHART_GAUGE_COMPILER,
        "static_checkpoint_input": static_input,
        "dataset_input_identity": manifest_validation["input_identity"],
        "source": source_start,
        "comparison_report": single.display_path(report_path),
        "child_progress": single.display_path(progress_path),
        "execution_identity": single.display_path(identity_path),
        "summary": single.display_path(summary_path),
        "reuse_requested": args.reuse_existing,
        "require_clean_source_flag": args.require_clean_source,
        "authoritative_completion_contract": (
            "moving_camera_attempt.json status=complete and attempt_id "
            "equals summary.json attempt_id"
        ),
    }

    report: Mapping[str, Any] | None = None
    execution_identity: Mapping[str, Any] | None = None
    sweep: Mapping[str, Any] | None = None
    gate: dict[str, Any] | None = None
    native: Mapping[str, Any] | None = None
    reused = False
    reuse_rejection: dict[str, str] | None = None
    if (
        args.reuse_existing
        and report_path.is_file()
        and progress_path.is_file()
        and identity_path.is_file()
    ):
        try:
            candidate = single.load_json(report_path)
            candidate_identity = single.load_json(identity_path)
            candidate_sweep, candidate_gate, candidate_native = validate_report(
                candidate,
                protocol=protocol,
                seed=args.seed,
                checkpoint=checkpoint,
            )
            validate_child_progress(progress_path, checkpoint=checkpoint)
            validate_execution_identity(
                candidate_identity,
                protocol_path=protocol_path,
                command=command,
                report_path=report_path,
                progress_path=progress_path,
                expected_source=source_start,
                expected_dataset_input_identity=manifest_validation[
                    "input_identity"
                ],
                expected_static_input=static_input,
                expected_native_extension=candidate_native,
            )
        except (KeyError, OSError, TypeError, ValueError) as error:
            reuse_rejection = _failure(error)
        else:
            report = candidate
            execution_identity = candidate_identity
            sweep = candidate_sweep
            gate = candidate_gate
            native = candidate_native
            reused = True
    elif args.reuse_existing:
        reuse_rejection = {
            "type": "MissingReuseArtifacts",
            "message": (
                "validated reuse requires comparison_report.json, "
                "frozen_world_moving_camera_progress.json, and "
                "execution_identity.json"
            ),
        }
    attempt_base["reuse_rejection"] = reuse_rejection

    if report is None:
        write_attempt(
            attempt_path,
            attempt_base,
            status="running",
            phase="resource_preflight",
            reused_existing=False,
        )
        try:
            resources = require_execution_resources(
                device=args.device,
                allow_local_mps_execution=args.allow_local_mps_execution,
                checkpoint_bytes=int(checkpoint["bytes"]),
            )
        except BaseException as error:
            write_attempt(
                attempt_path,
                attempt_base,
                status="failed",
                phase="resource_preflight",
                reused_existing=False,
                failure=_failure(error),
            )
            raise
        write_attempt(
            attempt_path,
            attempt_base,
            status="running",
            phase="child_process",
            reused_existing=False,
            resources=resources,
        )
        try:
            child_process_memory = single.run_checked_with_peak_rss(
                command,
                cwd=ROOT,
                rss_limit_bytes=int(resources["estimate"]["safety_limit_bytes"]),
            )
        except BaseException as error:
            write_attempt(
                attempt_path,
                attempt_base,
                status="failed",
                phase="child_process",
                reused_existing=False,
                resources=resources,
                failure=_failure(error),
                child_progress=single.display_path(progress_path),
            )
            raise
        try:
            report = single.load_json(report_path)
            sweep, gate, native = validate_report(
                report,
                protocol=protocol,
                seed=args.seed,
                checkpoint=checkpoint,
            )
            validate_child_progress(progress_path, checkpoint=checkpoint)
            source_finish = single.source_provenance()
            single.require_clean_provenance(source_finish)
            if source_finish != source_start:
                raise RuntimeError(
                    "source changed while moving-camera job executed"
                )
            manifest_finish = single.validate_manifest(protocol)
            if (
                manifest_finish["input_identity"]
                != manifest_validation["input_identity"]
            ):
                raise RuntimeError(
                    "Coffee Martini raw inputs changed during moving-camera job"
                )
            static_finish = validate_static_checkpoint_input(
                static_summary_path,
                protocol_name=protocol.name,
                seed=args.seed,
                expected_dataset_input_identity=manifest_finish[
                    "input_identity"
                ],
            )
            if static_finish != static_input:
                raise RuntimeError(
                    "static checkpoint identity changed during moving-camera job"
                )
            execution_identity = {
                "schema_version": SCHEMA_VERSION,
                "attempt_id": attempt_id,
                "protocol_sha256": file_sha256(protocol_path),
                "command": command,
                "source_start": source_start,
                "source_finish": source_finish,
                "dataset_input_identity": manifest_validation[
                    "input_identity"
                ],
                "static_checkpoint_input": static_input,
                "camera_program": camera_program_contract(),
                "camera_program_sha256": camera_program_sha256(),
                "compiler_chart_policy": COMPILER_CHART_POLICY,
                "multi_chart_gauge_compiler": MULTI_CHART_GAUGE_COMPILER,
                "comparison_report": single.display_path(report_path),
                "comparison_report_sha256": file_sha256(report_path),
                "progress": single.display_path(progress_path),
                "progress_sha256": file_sha256(progress_path),
                "star_uvt_native_extension": native,
                "resources_at_launch": resources,
                "process_memory": child_process_memory,
            }
            write_json_atomic(identity_path, execution_identity)
        except BaseException as error:
            write_attempt(
                attempt_path,
                attempt_base,
                status="failed",
                phase="report_validation",
                reused_existing=False,
                failure=_failure(error),
            )
            raise
    else:
        write_attempt(
            attempt_path,
            attempt_base,
            status="running",
            phase="validated_reuse",
            reused_existing=True,
        )

    if any(value is None for value in (report, execution_identity, sweep, gate, native)):
        raise RuntimeError("moving-camera execution artifacts were not materialized")
    write_attempt(
        attempt_path,
        attempt_base,
        status="running",
        phase="wandb_logging",
        reused_existing=reused,
        publication_eligible=gate["accepted"],
    )
    try:
        wandb = moving_camera_wandb_log(
            report,
            protocol=protocol,
            seed=args.seed,
            report_dir=run_dir,
            wandb_mode=args.wandb_mode,
            execution_source=execution_identity["source_start"],
            gate=gate,
            static_input=static_input,
        )
    except BaseException as error:
        write_attempt(
            attempt_path,
            attempt_base,
            status="failed",
            phase="wandb_logging",
            reused_existing=reused,
            publication_eligible=gate["accepted"],
            failure=_failure(error),
        )
        raise
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "accepted" if gate["accepted"] else "complete_negative",
        "publication_eligible": gate["accepted"],
        "attempt_id": attempt_id,
        "protocol": protocol.as_dict(),
        "seed": args.seed,
        "frame_counts": list(FRAME_COUNTS),
        "image_size": list(IMAGE_SIZE),
        "decoded_image_size": list(sweep["decoded_image_size"]),
        "render_image_size": list(sweep["render_image_size"]),
        "target_resize_mode": sweep["target_resize_mode"],
        "target_semantics": TARGET_SEMANTICS,
        "parity_metric_semantics": PARITY_METRIC_SEMANTICS,
        "camera_program": camera_program_contract(),
        "camera_program_sha256": camera_program_sha256(),
        "compiler_chart_policy": COMPILER_CHART_POLICY,
        "multi_chart_gauge_compiler": MULTI_CHART_GAUGE_COMPILER,
        "publication_gate": gate,
        "execution_mode": (
            "validated_reuse" if reused else "fresh_execution"
        ),
        "source": execution_identity["source_start"],
        "source_finish": execution_identity["source_finish"],
        "dataset_input_identity": manifest_validation["input_identity"],
        "decoded_dataset_bundle": report["meta"]["paper_dataset_bundle"],
        "evaluator": report["meta"]["paper_evaluator"],
        "runtime": report["meta"]["paper_runtime"],
        "static_checkpoint_input": static_input,
        "star_uvt_native_extension": native,
        "comparison_report": single.display_path(report_path),
        "child_progress": single.display_path(progress_path),
        "execution_identity": single.display_path(identity_path),
        "moving_camera_sweep": sweep,
        "wandb": wandb,
    }
    write_json_atomic(summary_path, summary)
    write_attempt(
        attempt_path,
        attempt_base,
        status="complete",
        phase="summary_written",
        reused_existing=reused,
        publication_eligible=gate["accepted"],
        result_status=summary["status"],
        summary=single.display_path(summary_path),
        summary_sha256=file_sha256(summary_path),
    )
    print(
        json.dumps(
            serialize_config_value(summary),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
