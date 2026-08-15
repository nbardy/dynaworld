"""CPU fixed-duration frame-density gate for the compiled WorldFoam adjoint.

This report isolates the strong systems claim.  One adaptive affine-Lie atlas
is compiled over a fixed physical interval, then evaluated at increasingly
dense requested time samples.  Expensive ordered-word compile/reverse work and
bounded reverse scratch must remain invariant; only output/sample-basis work is
allowed to grow with frame density.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from compiled_lie_world_adjoint import (
    AdaptiveLieWorldCompilePolicy,
    compile_adaptive_lie_world_atlas,
)
from compiled_transfer_adjoint import make_stable_cell_word, streamed_word_mse_vjp
from prepared_track_block import prepare_worldfoam_track_block
from staged_compiled_lie_adjoint import (
    accumulate_staged_piecewise_lie_mse,
    begin_staged_piecewise_lie_mse,
    finalize_staged_piecewise_lie_world_vjp,
    refresh_staged_lie_world_snapshot,
)

DTYPE = torch.float64
DEFAULT_FRAME_COUNTS = (16, 64, 256, 1024)
DEFAULT_FRAME_BLOCK_SIZE = 4
SCHEMA_VERSION = 3


def build_compiled_lie_frame_density_report(
    *,
    frame_counts: Sequence[int] = DEFAULT_FRAME_COUNTS,
    frame_block_size: int = DEFAULT_FRAME_BLOCK_SIZE,
) -> dict[str, Any]:
    """Run exact-versus-compiled CPU rows over one fixed physical interval."""

    schedule = tuple(int(value) for value in frame_counts)
    if not schedule or any(value < 2 for value in schedule):
        raise ValueError("frame_counts must contain values >= 2")
    if any(right <= left for left, right in zip(schedule, schedule[1:], strict=False)):
        raise ValueError("frame_counts must be strictly increasing")
    if frame_block_size < 1:
        raise ValueError("frame_block_size must be positive")
    fixture = _hard_dormant_fixture()
    policy = _scaling_policy()
    atlas = compile_adaptive_lie_world_atlas(
        **_compile_inputs(fixture),
        policy=policy,
        track_block_size=1,
        frame_block_size=frame_block_size,
    )
    prepared = prepare_worldfoam_track_block(
        fixture["words"],
        torch.tensor([[0, 1]], dtype=torch.int64),
        site_count=2,
        track_start=0,
        track_end=1,
    )
    world_snapshot = refresh_staged_lie_world_snapshot(
        atlas,
        assume_fixed_topology=True,
        boundary=fixture["boundary"],
        ray_coefficients=fixture["ray_coefficients"],
        site_density=fixture["site_density"],
        site_color=fixture["site_color"],
    )
    atlas = world_snapshot.atlas
    run_count = sum(int(word.owners.numel()) for word in fixture["words"])
    track_count = len(fixture["words"])
    atlas_structural_bytes = sum(chart.structural_bytes for chart in atlas.charts)
    coefficient_fit_interactions = track_count * sum(
        chart.node_count * chart.node_count for chart in atlas.charts
    )
    rows = []
    for frame_count in schedule:
        times = torch.linspace(
            fixture["t_min"],
            fixture["t_max"],
            frame_count,
            dtype=DTYPE,
        )
        targets = _targets(times).unsqueeze(0)
        chart_sample_counts = _chart_sample_counts(atlas, times)
        accumulator = begin_staged_piecewise_lie_mse(
            world_snapshot,
            background=fixture["background"],
            total_frame_count=frame_count,
            frame_block_size=frame_block_size,
            track_block_size=1,
        )
        prediction_blocks = []
        for frame_start in range(0, frame_count, frame_block_size):
            frame_end = min(frame_start + frame_block_size, frame_count)
            prediction_blocks.append(
                accumulate_staged_piecewise_lie_mse(
                    accumulator,
                    times=times[frame_start:frame_end],
                    targets=targets[:, frame_start:frame_end],
                    return_predictions=True,
                )
            )
        compiled = finalize_staged_piecewise_lie_world_vjp(accumulator)
        compiled_predictions = torch.cat(prediction_blocks, dim=1)
        exact = streamed_word_mse_vjp(
            **_exact_inputs(fixture),
            times=times,
            targets=targets,
            frame_block_size=frame_block_size,
            return_predictions=True,
            compute_ray_grad=False,
        )
        error = {
            "loss_max_abs": _max_abs(compiled.loss - exact.loss),
            "prediction_max_abs": _max_abs(compiled_predictions - exact.predictions),
            "site_density_grad_max_abs": _max_abs(
                compiled.grad_site_density - exact.grad_site_density
            ),
            "site_color_grad_max_abs": _max_abs(
                compiled.grad_site_color - exact.grad_site_color
            ),
            "depth_coefficient_grad_max_abs": _max_abs(
                compiled.grad_depth_coefficients - exact.grad_depth_coefficients
            ),
            "boundary_grad_max_abs": _max_abs(
                compiled.grad_boundary - exact.grad_boundary
            ),
        }
        exact_forward = frame_count * run_count
        exact_reverse = 2 * exact_forward
        sample_weight_linear = compiled.accounting["sample_weight_linear_interactions"]
        sample_weight_dense_fallback = compiled.accounting[
            "sample_weight_dense_fallback_interactions"
        ]
        compiled_total_proxy = (
            compiled.accounting["refresh_world_forward_run_interactions"]
            + compiled.accounting["step_world_reverse_run_interactions"]
            + compiled.accounting["sample_basis_interactions"]
            + coefficient_fit_interactions
            + sample_weight_linear
            + sample_weight_dense_fallback
        )
        exact_total_proxy = exact_forward + exact_reverse
        output_scalars = track_count * frame_count * 3
        rows.append(
            {
                "frame_count": frame_count,
                "track_count": track_count,
                "selection_signature": _selection_signature(atlas.selection_signature),
                "chart_count": atlas.chart_count,
                "chart_sample_counts": chart_sample_counts,
                "total_node_count": atlas.total_node_count,
                "run_count": run_count,
                "refresh_world_forward_run_interactions": compiled.accounting[
                    "refresh_world_forward_run_interactions"
                ],
                "step_world_reverse_run_interactions": compiled.accounting[
                    "step_world_reverse_run_interactions"
                ],
                "sample_basis_interactions": compiled.accounting[
                    "sample_basis_interactions"
                ],
                "coefficient_fit_interactions": coefficient_fit_interactions,
                "sample_weight_linear_interactions": sample_weight_linear,
                "sample_weight_dense_fallback_interactions": (
                    sample_weight_dense_fallback
                ),
                "sample_weight_dense_fallback_rows": compiled.accounting[
                    "sample_weight_dense_fallback_rows"
                ],
                "sample_weight_interactions": (
                    sample_weight_linear + sample_weight_dense_fallback
                ),
                "exact_replay_forward_run_interactions": exact_forward,
                "exact_replay_reverse_run_interactions": exact_reverse,
                "compiled_to_exact_reverse_interaction_ratio": (
                    compiled.accounting["step_world_reverse_run_interactions"]
                    / exact_reverse
                ),
                "compiled_total_interaction_proxy": compiled_total_proxy,
                "exact_total_interaction_proxy": exact_total_proxy,
                "compiled_to_exact_total_interaction_proxy_ratio": (
                    compiled_total_proxy / exact_total_proxy
                ),
                "logical_selected_reverse_state_bytes_excluding_targets_and_predictions": compiled.accounting[
                    "logical_selected_reverse_state_bytes_excluding_targets_and_predictions"
                ],
                "target_bytes": output_scalars * torch.tensor([], dtype=DTYPE).element_size(),
                "prediction_bytes": output_scalars * torch.tensor([], dtype=DTYPE).element_size(),
                "frame_run_reverse_state_elements": compiled.accounting[
                    "frame_run_reverse_state_elements"
                ],
                "per_sample_run_tape_bytes": compiled.accounting["per_sample_run_tape_bytes"],
                "sampled_validation_count_in_warm_step": compiled.accounting[
                    "sampled_validation_count"
                ],
                "sample_block_count": compiled.accounting["sample_block_count"],
                "world_finalize_calls": compiled.accounting["world_finalize_calls"],
                "boundary_finalize_calls": compiled.accounting["boundary_finalize_calls"],
                "retained_target_bytes": compiled.accounting["retained_target_bytes"],
                "retained_prediction_bytes": compiled.accounting["retained_prediction_bytes"],
                "accumulator_bytes_excluding_atlas": compiled.accounting[
                    "accumulator_bytes_excluding_atlas"
                ],
                "error": error,
            }
        )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "fixture": "hard_dormant_colored_tail_fixed_topology",
        "physical_interval": [fixture["t_min"], fixture["t_max"]],
        "frame_counts": list(schedule),
        "frame_block_size": frame_block_size,
        "compiler_policy_has_no_frame_count": (
            "frame_count" not in inspect.signature(compile_adaptive_lie_world_atlas).parameters
        ),
        "selection_signature": _selection_signature(atlas.selection_signature),
        "selected_rank_independent_of_frame_count": all(
            row["selection_signature"] == _selection_signature(atlas.selection_signature)
            for row in rows
        ),
        "atlas_structural_bytes": atlas_structural_bytes,
        "prepared_track_block_bytes": prepared.resident_bytes,
        "policy": {
            "node_count_schedule": list(policy.node_count_schedule),
            "probe_validation_count": policy.probe_validation_count,
            "heldout_validation_count": policy.heldout_validation_count,
            "probe_direction_count": policy.probe_direction_count,
            "heldout_direction_count": policy.heldout_direction_count,
            "forward_absolute_tolerance": policy.forward_absolute_tolerance,
            "forward_relative_tolerance": policy.forward_relative_tolerance,
            "tangent_absolute_tolerance": policy.tangent_absolute_tolerance,
            "tangent_relative_tolerance": policy.tangent_relative_tolerance,
            "max_split_depth": policy.max_split_depth,
            "max_chart_count": policy.max_chart_count,
        },
        "acceptance": {
            "reverse_state_frame_scale_max": 1.10,
            "loss_max_abs": 1.0e-12,
            "prediction_max_abs": 1.0e-12,
            "site_density_grad_max_abs": 1.0e-6,
            "site_color_grad_max_abs": 1.0e-12,
            "depth_coefficient_grad_max_abs": 1.0e-10,
            "boundary_grad_max_abs": 1.0e-10,
        },
        "rows": rows,
        "scope": {
            "cpu_only": True,
            "fixed_topology": True,
            "sampled_directional_rank_gate": True,
            "continuous_jacobian_certificate": False,
            "atlas_world_snapshot_bound": True,
            "staged_sample_accumulation": True,
            "compact_spatial_world_gradients": False,
            "native_runtime_parity": False,
            "measured_allocator_peak": False,
        },
    }
    failures = verify_compiled_lie_frame_density_report(report)
    report["failures"] = failures
    report["verified"] = not failures
    return report


def verify_compiled_lie_frame_density_report(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        failures.append("unsupported schema_version")
    if report.get("verified") is not True and "verified" in report:
        failures.append("stored report is not marked verified")
    if report.get("failures") not in (None, []):
        failures.append("stored report contains failures")
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) < 2:
        return [*failures, "rows must contain at least two frame-density measurements"]
    if not report.get("compiler_policy_has_no_frame_count"):
        failures.append("adaptive compiler unexpectedly accepts frame_count")
    if not report.get("selected_rank_independent_of_frame_count"):
        failures.append("rank/chart selection changed with frame density")
    signature = report.get("selection_signature")
    parsed_signature = _validated_report_signature(signature, failures)
    expected_chart_count = len(parsed_signature)
    expected_node_count = sum(rank for _, _, rank in parsed_signature)
    physical_interval = report.get("physical_interval")
    if (
        not isinstance(physical_interval, list)
        or len(physical_interval) != 2
        or not all(_is_finite_number(value) for value in physical_interval)
        or float(physical_interval[1]) <= float(physical_interval[0])
    ):
        failures.append("physical_interval must be a finite increasing pair")
    elif parsed_signature and (
        parsed_signature[0][0] != float(physical_interval[0])
        or parsed_signature[-1][1] != float(physical_interval[1])
    ):
        failures.append("selection signature does not cover the physical interval")
    frame_counts = [int(row.get("frame_count", -1)) for row in rows]
    if report.get("frame_counts") != frame_counts:
        failures.append("top-level frame_counts do not match rows")
    if any(right <= left for left, right in zip(frame_counts, frame_counts[1:], strict=False)):
        failures.append("frame_count rows must be strictly increasing")
    report_frame_block_size = report.get("frame_block_size")
    if (
        isinstance(report_frame_block_size, bool)
        or not isinstance(report_frame_block_size, int)
        or report_frame_block_size < 1
    ):
        failures.append("frame_block_size must be a positive integer")
        report_frame_block_size = 1
    if any(row.get("selection_signature") != signature for row in rows):
        failures.append("a row does not use the frozen selection signature")
    for key in (
        "refresh_world_forward_run_interactions",
        "step_world_reverse_run_interactions",
        "coefficient_fit_interactions",
        "logical_selected_reverse_state_bytes_excluding_targets_and_predictions",
    ):
        values = [int(row.get(key, -1)) for row in rows]
        if min(values) < 0 or len(set(values)) != 1:
            failures.append(f"{key} must be nonnegative and invariant in frame density")
    if int(report.get("atlas_structural_bytes", 0)) <= 0:
        failures.append("atlas_structural_bytes must be positive")
    if int(report.get("prepared_track_block_bytes", 0)) <= 0:
        failures.append("prepared_track_block_bytes must be positive")
    expected_scope = {
        "cpu_only": True,
        "fixed_topology": True,
        "sampled_directional_rank_gate": True,
        "continuous_jacobian_certificate": False,
        "atlas_world_snapshot_bound": True,
        "staged_sample_accumulation": True,
        "compact_spatial_world_gradients": False,
        "native_runtime_parity": False,
        "measured_allocator_peak": False,
    }
    if report.get("scope") != expected_scope:
        failures.append("scope flags do not match this CPU fixed-topology artifact")
    acceptance = report.get("acceptance")
    if not isinstance(acceptance, dict):
        return [*failures, "acceptance must be an object"]
    if any(
        not _is_finite_number(value) or float(value) <= 0.0
        for value in acceptance.values()
    ):
        failures.append("acceptance tolerances must be finite and positive")
    reverse_bytes = [
        int(
            row.get(
                "logical_selected_reverse_state_bytes_excluding_targets_and_predictions",
                -1,
            )
        )
        for row in rows
    ]
    if min(reverse_bytes) <= 0:
        failures.append("reverse-state byte accounting must be positive")
    elif max(reverse_bytes) / min(reverse_bytes) > float(
        acceptance["reverse_state_frame_scale_max"]
    ):
        failures.append("bounded reverse-state bytes grew beyond the frame-density contract")
    exact_reverse = [int(row.get("exact_replay_reverse_run_interactions", -1)) for row in rows]
    if any(
        current * previous_frames != previous * current_frames
        for previous, current, previous_frames, current_frames in zip(
            exact_reverse[:-1],
            exact_reverse[1:],
            frame_counts[:-1],
            frame_counts[1:],
            strict=True,
        )
    ):
        failures.append("exact replay interactions are not linear in frame count")
    ratios = [float(row.get("compiled_to_exact_reverse_interaction_ratio", float("inf"))) for row in rows]
    if not all(math.isfinite(value) and value > 0.0 for value in ratios):
        failures.append("compiled/exact reverse ratios must be finite and positive")
    elif any(current >= previous for previous, current in zip(ratios, ratios[1:], strict=False)):
        failures.append("compiled/exact reverse interaction ratio did not decrease with frame density")
    for row in rows:
        frame_count = int(row.get("frame_count", -1))
        track_count = int(row.get("track_count", -1))
        run_count = int(row.get("run_count", -1))
        node_count = int(row.get("total_node_count", -1))
        expected_refresh = node_count * run_count
        expected_reverse = expected_refresh
        expected_exact_forward = frame_count * run_count
        expected_exact_reverse = 2 * expected_exact_forward
        if expected_chart_count < 1 or int(row.get("chart_count", -1)) != expected_chart_count:
            failures.append("row chart_count is inconsistent with selection_signature")
        if expected_node_count < 1 or node_count != expected_node_count:
            failures.append("row total_node_count is inconsistent with selection_signature")
        if track_count < 1 or run_count < 1:
            failures.append("row track_count and run_count must be positive")
        chart_sample_counts = row.get("chart_sample_counts")
        if (
            not isinstance(chart_sample_counts, list)
            or len(chart_sample_counts) != expected_chart_count
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in chart_sample_counts
            )
            or sum(chart_sample_counts) != frame_count
        ):
            failures.append("row chart_sample_counts must partition the requested frames")
        else:
            expected_sample_basis = track_count * sum(
                count * rank
                for count, (_, _, rank) in zip(
                    chart_sample_counts,
                    parsed_signature,
                    strict=True,
                )
            )
            if int(row.get("sample_basis_interactions", -1)) != expected_sample_basis:
                failures.append("sample-basis interaction accounting is inconsistent")
            expected_sample_weight_linear = sum(
                count * rank
                for count, (_, _, rank) in zip(
                    chart_sample_counts,
                    parsed_signature,
                    strict=True,
                )
            )
            if (
                int(row.get("sample_weight_linear_interactions", -1))
                != expected_sample_weight_linear
            ):
                failures.append("linear sample-weight interaction accounting is inconsistent")
        expected_coefficient_fit = track_count * sum(
            rank * rank for _, _, rank in parsed_signature
        )
        if int(row.get("coefficient_fit_interactions", -1)) != expected_coefficient_fit:
            failures.append("coefficient-fit interaction accounting is inconsistent")
        dense_fallback_rows = int(row.get("sample_weight_dense_fallback_rows", -1))
        dense_fallback_interactions = int(
            row.get("sample_weight_dense_fallback_interactions", -1)
        )
        if not 0 <= dense_fallback_rows <= frame_count:
            failures.append("dense sample-weight fallback rows are out of range")
        if dense_fallback_rows == 0 and dense_fallback_interactions != 0:
            failures.append("dense sample-weight fallback work exists without fallback rows")
        if dense_fallback_rows > 0 and not parsed_signature:
            failures.append("dense sample-weight fallback rows require a valid chart signature")
        elif dense_fallback_rows > 0:
            minimum_rank = min(rank for _, _, rank in parsed_signature)
            maximum_rank = max(rank for _, _, rank in parsed_signature)
            if not (
                dense_fallback_rows * minimum_rank * minimum_rank
                <= dense_fallback_interactions
                <= dense_fallback_rows * maximum_rank * maximum_rank
            ):
                failures.append("dense sample-weight fallback interaction accounting is inconsistent")
        expected_sample_weight_total = (
            int(row.get("sample_weight_linear_interactions", -1))
            + dense_fallback_interactions
        )
        if int(row.get("sample_weight_interactions", -1)) != expected_sample_weight_total:
            failures.append("total sample-weight interaction accounting is inconsistent")
        if int(row.get("refresh_world_forward_run_interactions", -1)) != expected_refresh:
            failures.append("compiled refresh interaction accounting is inconsistent")
        if int(row.get("step_world_reverse_run_interactions", -1)) != expected_reverse:
            failures.append("compiled reverse interaction accounting is inconsistent")
        if int(row.get("exact_replay_forward_run_interactions", -1)) != expected_exact_forward:
            failures.append("exact forward interaction accounting is inconsistent")
        if int(row.get("exact_replay_reverse_run_interactions", -1)) != expected_exact_reverse:
            failures.append("exact reverse interaction accounting is inconsistent")
        expected_compiled_total = (
            expected_refresh
            + expected_reverse
            + int(row.get("sample_basis_interactions", -1))
            + expected_coefficient_fit
            + expected_sample_weight_total
        )
        expected_exact_total = expected_exact_forward + expected_exact_reverse
        if int(row.get("compiled_total_interaction_proxy", -1)) != expected_compiled_total:
            failures.append("compiled total interaction proxy is inconsistent")
        if int(row.get("exact_total_interaction_proxy", -1)) != expected_exact_total:
            failures.append("exact total interaction proxy is inconsistent")
        total_ratio = float(
            row.get("compiled_to_exact_total_interaction_proxy_ratio", float("inf"))
        )
        if (
            not math.isfinite(total_ratio)
            or abs(total_ratio - expected_compiled_total / expected_exact_total) > 1.0e-15
        ):
            failures.append("compiled/exact total interaction proxy ratio is inconsistent")
        observed_ratio = float(row.get("compiled_to_exact_reverse_interaction_ratio", float("inf")))
        if abs(observed_ratio - expected_reverse / expected_exact_reverse) > 1.0e-15:
            failures.append("compiled/exact reverse ratio is inconsistent")
        if int(row.get("frame_run_reverse_state_elements", -1)) != 0:
            failures.append("warm step retained frame-by-run reverse state")
        if int(row.get("per_sample_run_tape_bytes", -1)) != 0:
            failures.append("warm step retained a per-sample run tape")
        if int(row.get("sampled_validation_count_in_warm_step", -1)) != 0:
            failures.append("warm step hid exact rank validation work")
        expected_sample_blocks = math.ceil(frame_count / report_frame_block_size)
        if int(row.get("sample_block_count", -1)) != expected_sample_blocks:
            failures.append("staged sample-block accounting is inconsistent")
        if int(row.get("world_finalize_calls", -1)) != 1:
            failures.append("streamed blocks did not share one world finalize")
        if int(row.get("boundary_finalize_calls", -1)) != 1:
            failures.append("streamed charts did not share one boundary finalize")
        if int(row.get("retained_target_bytes", -1)) != 0:
            failures.append("staged accumulator retained target storage")
        if int(row.get("retained_prediction_bytes", -1)) != 0:
            failures.append("staged accumulator retained prediction storage")
        for key, tolerance in acceptance.items():
            if key == "reverse_state_frame_scale_max":
                continue
            value = row.get("error", {}).get(key, float("inf"))
            if not _is_finite_number(value):
                failures.append(f"F={row.get('frame_count')} {key} is not finite")
            elif float(value) > float(tolerance):
                failures.append(
                    f"F={row.get('frame_count')} {key}={float(value):.3e} exceeds {float(tolerance):.3e}"
                )
    for key in ("target_bytes", "prediction_bytes"):
        bytes_per_frame = []
        for row in rows:
            frame_count = int(row.get("frame_count", 0))
            if frame_count <= 0:
                failures.append(f"{key} row has an invalid frame_count")
                continue
            bytes_per_frame.append(int(row.get(key, -1)) / frame_count)
        if not bytes_per_frame or min(bytes_per_frame) <= 0 or len(set(bytes_per_frame)) != 1:
            failures.append(f"{key} must expose the unavoidable linear output/sample axis")
    return failures


def _validated_report_signature(
    value: Any,
    failures: list[str],
) -> list[tuple[float, float, int]]:
    if not isinstance(value, list) or not value:
        failures.append("selection_signature must be a non-empty list")
        return []
    parsed: list[tuple[float, float, int]] = []
    for row in value:
        if (
            not isinstance(row, list)
            or len(row) != 3
            or not _is_finite_number(row[0])
            or not _is_finite_number(row[1])
            or isinstance(row[2], bool)
            or not isinstance(row[2], int)
        ):
            failures.append("selection_signature contains an invalid chart row")
            return []
        t_min, t_max, rank = float(row[0]), float(row[1]), int(row[2])
        if t_max <= t_min or rank < 2:
            failures.append("selection_signature charts need increasing intervals and rank >= 2")
        parsed.append((t_min, t_max, rank))
    for left, right in zip(parsed, parsed[1:], strict=False):
        if left[1] != right[0]:
            failures.append("selection_signature charts must be ordered and contiguous")
    return parsed


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def assert_compiled_lie_frame_density_report(report: dict[str, Any]) -> None:
    failures = verify_compiled_lie_frame_density_report(report)
    if failures:
        raise ValueError("compiled Lie frame-density report failed: " + "; ".join(failures))


def _hard_dormant_fixture() -> dict[str, Any]:
    return {
        "boundary": torch.tensor([[0.0, 0.0, 1.0, -0.9, -1.0]], dtype=DTYPE),
        "ray_coefficients": torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            dtype=DTYPE,
        ),
        "words": (make_stable_cell_word([0, 1], [-1, 0], [0, -2]),),
        "site_density": torch.tensor([50.0, 0.0], dtype=DTYPE),
        "site_color": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=DTYPE),
        "background": torch.tensor([0.07, 0.03, 0.11], dtype=DTYPE),
        "t_min": -1.0,
        "t_max": 1.0,
        "near": 0.05,
        "far": 2.0,
    }


def _scaling_policy() -> AdaptiveLieWorldCompilePolicy:
    return AdaptiveLieWorldCompilePolicy(
        node_count_schedule=(2, 4, 8, 16),
        probe_validation_count=17,
        heldout_validation_count=16,
        probe_direction_count=2,
        heldout_direction_count=2,
        forward_absolute_tolerance=1.0e-10,
        forward_relative_tolerance=1.0e-6,
        tangent_absolute_tolerance=1.0e-10,
        tangent_relative_tolerance=1.0e-3,
        max_split_depth=3,
        max_chart_count=8,
    )


def _compile_inputs(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        key: fixture[key]
        for key in (
            "boundary",
            "ray_coefficients",
            "words",
            "site_density",
            "site_color",
            "t_min",
            "t_max",
            "near",
            "far",
        )
    }


def _exact_inputs(fixture: dict[str, Any]) -> dict[str, Any]:
    return {**_compile_inputs(fixture), "background": fixture["background"]}


def _targets(times: torch.Tensor) -> torch.Tensor:
    phase = 1.7 * (times + 1.0)
    return torch.stack(
        (
            0.2 + 0.1 * torch.sin(phase),
            0.3 + 0.05 * torch.cos(phase),
            0.1 + 0.03 * torch.sin(2.0 * phase),
        ),
        dim=1,
    )


def _chart_sample_counts(
    atlas: Any,
    times: torch.Tensor,
) -> list[int]:
    counts = []
    for chart_id, chart in enumerate(atlas.charts):
        is_last = chart_id == len(atlas.charts) - 1
        mask = times >= chart.transfer_atlas.t_min
        mask &= (
            times <= chart.transfer_atlas.t_max
            if is_last
            else times < chart.transfer_atlas.t_max
        )
        counts.append(int(mask.sum().item()))
    return counts


def _selection_signature(signature: tuple[tuple[float, float, int], ...]) -> list[list[float | int]]:
    return [[float(t_min), float(t_max), int(rank)] for t_min, t_max, rank in signature]


def _max_abs(value: torch.Tensor | None) -> float:
    if value is None or value.numel() == 0:
        return 0.0
    return float(value.detach().abs().max().item())


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("expected a JSON object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify-report", type=Path)
    args = parser.parse_args()
    if args.verify_report is not None:
        report = _load_json(args.verify_report)
        assert_compiled_lie_frame_density_report(report)
        print(json.dumps({"status": "ok", "report": str(args.verify_report)}, indent=2))
        return
    report = build_compiled_lie_frame_density_report()
    assert_compiled_lie_frame_density_report(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
