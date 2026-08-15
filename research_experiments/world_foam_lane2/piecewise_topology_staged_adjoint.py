"""CPU orchestration for WorldFoam time charts with different owner words.

The fixed-topology staged adjoint deliberately requires every chart inside one
``PreparedStagedLieWorld`` to share its word and sparse-incidence CSR.  A real
power diagram can change that topology over time.  This module composes
multiple independently prepared snapshots without weakening the lower-level
invariant:

* time charts form one ordered half-open partition (the final chart is closed
  on the right);
* each topology chart streams only its selected samples through its own staged
  accumulator, but every accumulator uses the same ``P * F * 3`` loss
  denominator;
* each local boundary adjoint is lowered through that chart's active power
  faces, then compact site rows are index-added into caller-owned global bars;
* no target, prediction, or ``F x R`` ordered-run tape is retained by the
  returned step result.

At a chart seam the sample is assigned to the right chart.  The reported
parameter VJP is therefore the right one-sided, frozen-topology derivative.
The derivative of event time and the algebraic derivative of discrete topology
dispatch are intentionally unresolved; the result exposes that fact as
machine-readable metadata instead of claiming a smooth gradient at the seam.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from compiled_transfer_adjoint import power_boundary_parameters, power_boundary_parameters_vjp
from prepared_track_block import accumulate_prepared_rows_
from staged_compiled_lie_adjoint import (
    CompactSpatialGradientBuffers,
    PreparedStagedLieWorld,
    _tensor_signature,
    _validate_piecewise_atlas,
    accumulate_staged_piecewise_lie_mse,
    begin_staged_piecewise_lie_mse,
    finalize_staged_piecewise_lie_world_vjp,
)
from transfer_lie_chart import DTYPE


@dataclass(frozen=True)
class PreparedTopologyLieChart:
    """One certified fixed-topology snapshot over a global time subinterval.

    ``source_site_ids`` maps the snapshot's compact site rows into one common
    caller-owned site table. ``boundary_site_pairs`` uses snapshot-local site
    ids, so separate charts may have different active faces and CSR layouts.
    """

    chart_id: str
    t_min: float
    t_max: float
    world_snapshot: PreparedStagedLieWorld
    source_site_ids: torch.Tensor
    boundary_site_pairs: torch.Tensor


@dataclass(frozen=True)
class TopologyEventGradientMetadata:
    """Explicit derivative convention at one topology-chart seam."""

    time: float
    left_chart_id: str
    right_chart_id: str
    seam_sample_assignment: Literal["right_chart"] = "right_chart"
    frozen_topology_parameter_vjp: Literal["right_one_sided"] = "right_one_sided"
    event_time_vjp: Literal["not_implemented"] = "not_implemented"
    algebraic_event_dispatch_vjp: Literal["unresolved"] = "unresolved"
    differentiability: Literal["nondifferentiable_or_stratified"] = (
        "nondifferentiable_or_stratified"
    )


@dataclass(frozen=True)
class PiecewiseTopologyStagedResult:
    """Global loss and views of the caller-owned common-site gradient bars."""

    loss: torch.Tensor
    gradients: CompactSpatialGradientBuffers
    predictions: torch.Tensor | None
    event_gradients: tuple[TopologyEventGradientMetadata, ...]
    accounting: dict[str, object]


def piecewise_topology_staged_lie_mse_vjp(
    charts: Sequence[PreparedTopologyLieChart],
    *,
    site_geometry: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    gradients: CompactSpatialGradientBuffers,
    times: torch.Tensor,
    targets: torch.Tensor,
    background: torch.Tensor | tuple[float, float, float] | list[float],
    frame_block_size: int,
    track_block_size: int = 64,
    loss_normalization_id: str = "piecewise-topology-step",
    return_predictions: bool = False,
) -> PiecewiseTopologyStagedResult:
    """Stream and reduce independently compiled topology charts into one VJP.

    This is a CPU reference/lifecycle contract, not event differentiation.  It
    assumes that a separate compiler has already certified each supplied owner
    word on its half-open interval and has selected the topology seams.
    """

    charts_tuple = tuple(charts)
    geometry = torch.as_tensor(site_geometry)
    density = torch.as_tensor(site_density)
    color = torch.as_tensor(site_color)
    times_f64 = torch.as_tensor(times, dtype=DTYPE).reshape(-1).detach()
    targets_f64 = torch.as_tensor(targets, dtype=DTYPE).detach()
    _validate_global_inputs(
        charts_tuple,
        site_geometry=geometry,
        site_density=density,
        site_color=color,
        gradients=gradients,
        times=times_f64,
        targets=targets_f64,
        frame_block_size=frame_block_size,
        track_block_size=track_block_size,
        loss_normalization_id=loss_normalization_id,
    )
    global_frame_count = int(times_f64.numel())
    global_track_count = int(targets_f64.shape[0])
    source_tensors = (geometry, density, color)
    source_signatures = tuple(_tensor_signature(tensor) for tensor in source_tensors)
    for tensor in gradients.tensors:
        tensor.zero_()
    gradient_pointers = tuple(tensor.untyped_storage().data_ptr() for tensor in gradients.tensors)

    sample_ids_by_chart = _partition_sample_ids(charts_tuple, times_f64)
    predictions = torch.empty_like(targets_f64) if return_predictions else None
    loss = torch.zeros((), dtype=DTYPE)
    total_refresh_interactions = 0
    total_reverse_interactions = 0
    total_basis_interactions = 0
    total_boundary_finalize_calls = 0
    total_world_finalize_calls = 0
    peak_local_accumulator_bytes = 0
    maximum_selected_block_elements = 0
    chart_accounting: list[dict[str, object]] = []

    for chart, sample_ids in zip(charts_tuple, sample_ids_by_chart, strict=True):
        source_site_ids = torch.as_tensor(chart.source_site_ids)
        boundary_site_pairs = torch.as_tensor(chart.boundary_site_pairs)
        selected_count = int(sample_ids.numel())
        if selected_count == 0:
            chart_accounting.append(
                {
                    "chart_id": chart.chart_id,
                    "sample_count": 0,
                    "world_vjp_finalized": False,
                    "reason": "no_selected_samples",
                }
            )
            continue
        accumulator = begin_staged_piecewise_lie_mse(
            chart.world_snapshot,
            background=background,
            total_frame_count=selected_count,
            global_frame_count=global_frame_count,
            global_track_count=global_track_count,
            loss_normalization_id=loss_normalization_id,
            frame_block_size=frame_block_size,
            track_block_size=track_block_size,
        )
        peak_local_accumulator_bytes = max(
            peak_local_accumulator_bytes,
            accumulator.resident_bytes_excluding_atlas,
        )
        for local_start in range(0, selected_count, frame_block_size):
            local_end = min(local_start + frame_block_size, selected_count)
            block_ids = sample_ids[local_start:local_end]
            block_predictions = accumulate_staged_piecewise_lie_mse(
                accumulator,
                times=times_f64.index_select(0, block_ids),
                targets=targets_f64.index_select(1, block_ids),
                global_frame_start=local_start,
                return_predictions=return_predictions,
            )
            maximum_selected_block_elements = max(
                maximum_selected_block_elements,
                global_track_count * int(block_ids.numel()) * 3,
            )
            if predictions is not None and block_predictions is not None:
                predictions.index_copy_(1, block_ids, block_predictions)

        local_result = finalize_staged_piecewise_lie_world_vjp(accumulator)
        local_sites = geometry.index_select(0, source_site_ids)
        local_site_geometry_bar = power_boundary_parameters_vjp(
            local_sites,
            boundary_site_pairs,
            local_result.grad_boundary,
        )
        accumulate_prepared_rows_(
            gradients.grad_site_geometry,
            local_site_geometry_bar[:, :4],
            source_site_ids,
        )
        accumulate_prepared_rows_(
            gradients.grad_site_weight,
            local_site_geometry_bar[:, 4],
            source_site_ids,
        )
        accumulate_prepared_rows_(
            gradients.grad_site_density,
            local_result.grad_site_density,
            source_site_ids,
        )
        accumulate_prepared_rows_(
            gradients.grad_site_color,
            local_result.grad_site_color,
            source_site_ids,
        )
        loss.add_(local_result.loss)
        total_refresh_interactions += int(
            local_result.accounting["refresh_world_forward_run_interactions"]
        )
        total_reverse_interactions += int(
            local_result.accounting["step_world_reverse_run_interactions"]
        )
        total_basis_interactions += int(local_result.accounting["sample_basis_interactions"])
        total_boundary_finalize_calls += int(local_result.accounting["boundary_finalize_calls"])
        total_world_finalize_calls += int(local_result.accounting["world_finalize_calls"])
        chart_accounting.append(
            {
                "chart_id": chart.chart_id,
                "sample_count": selected_count,
                "world_vjp_finalized": True,
                "local_chart_count": chart.world_snapshot.atlas.chart_count,
                "local_total_node_count": chart.world_snapshot.atlas.total_node_count,
                "local_boundary_count": int(boundary_site_pairs.shape[0]),
                "local_site_count": int(source_site_ids.numel()),
            }
        )

    if tuple(_tensor_signature(tensor) for tensor in source_tensors) != source_signatures:
        raise ValueError("common site tensors changed during the piecewise-topology step")
    if tuple(tensor.untyped_storage().data_ptr() for tensor in gradients.tensors) != gradient_pointers:
        raise ValueError("caller-owned global gradient storage changed during accumulation")
    if predictions is not None and not bool(torch.isfinite(predictions).all().item()):
        raise ValueError("piecewise topology evaluation left predictions unassigned or non-finite")

    event_gradients = tuple(
        TopologyEventGradientMetadata(
            time=left.t_max,
            left_chart_id=left.chart_id,
            right_chart_id=right.chart_id,
        )
        for left, right in zip(charts_tuple[:-1], charts_tuple[1:], strict=True)
    )
    gradient_bytes = sum(tensor.numel() * tensor.element_size() for tensor in gradients.tensors)
    return PiecewiseTopologyStagedResult(
        loss=loss,
        gradients=gradients,
        predictions=predictions,
        event_gradients=event_gradients,
        accounting={
            "global_track_count": global_track_count,
            "global_frame_count": global_frame_count,
            "global_loss_element_count": global_track_count * global_frame_count * 3,
            "loss_normalization_id": loss_normalization_id,
            "topology_chart_count": len(charts_tuple),
            "selected_topology_chart_count": sum(
                int(sample_ids.numel() > 0) for sample_ids in sample_ids_by_chart
            ),
            "chart_sample_counts": tuple(int(sample_ids.numel()) for sample_ids in sample_ids_by_chart),
            "chart_accounting": tuple(chart_accounting),
            "topology_event_count": len(event_gradients),
            "event_parameter_vjp_convention": "right_one_sided_frozen_topology",
            "algebraic_event_dispatch_vjp": "unresolved",
            "refresh_world_forward_run_interactions": total_refresh_interactions,
            "step_world_reverse_run_interactions": total_reverse_interactions,
            "sample_basis_interactions": total_basis_interactions,
            "world_finalize_calls": total_world_finalize_calls,
            "boundary_finalize_calls": total_boundary_finalize_calls,
            "frame_run_reverse_state_elements": 0,
            "per_sample_run_tape_bytes": 0,
            "retained_target_bytes": 0,
            "retained_prediction_bytes": 0 if predictions is None else predictions.numel() * predictions.element_size(),
            "maximum_selected_block_elements": maximum_selected_block_elements,
            "peak_local_accumulator_bytes_excluding_atlas": peak_local_accumulator_bytes,
            "global_gradient_buffer_allocations": 4,
            "global_gradient_buffer_bytes": gradient_bytes,
        },
    )


def _validate_global_inputs(
    charts: tuple[PreparedTopologyLieChart, ...],
    *,
    site_geometry: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    gradients: CompactSpatialGradientBuffers,
    times: torch.Tensor,
    targets: torch.Tensor,
    frame_block_size: int,
    track_block_size: int,
    loss_normalization_id: str,
) -> None:
    if not charts:
        raise ValueError("piecewise topology charts must be nonempty")
    if frame_block_size < 1 or track_block_size < 1:
        raise ValueError("frame_block_size and track_block_size must be positive")
    if not loss_normalization_id.strip():
        raise ValueError("loss_normalization_id must be nonempty")
    if times.numel() < 1 or not bool(torch.isfinite(times).all().item()):
        raise ValueError("times must be nonempty and finite")
    if times.device.type != "cpu":
        raise ValueError("piecewise topology orchestration is CPU-only")
    if site_geometry.ndim != 2 or site_geometry.shape[1] != 5 or site_geometry.shape[0] < 1:
        raise ValueError("site_geometry must have shape [S,5] with S > 0")
    site_count = int(site_geometry.shape[0])
    if tuple(site_density.shape) != (site_count,) or tuple(site_color.shape) != (site_count, 3):
        raise ValueError("site_density and site_color must match the common site table")
    source_tensors = (site_geometry, site_density, site_color)
    if any(tensor.dtype != DTYPE or tensor.device.type != "cpu" for tensor in source_tensors):
        raise ValueError("common site tensors must be CPU float64")
    if any(not tensor.is_contiguous() for tensor in source_tensors):
        raise ValueError("common site tensors must be contiguous")
    if any(not bool(torch.isfinite(tensor).all().item()) for tensor in source_tensors):
        raise ValueError("common site tensors must be finite")
    if targets.ndim != 3 or targets.shape[1] != times.numel() or targets.shape[2] != 3:
        raise ValueError("targets must have shape [P,F,3]")
    if targets.shape[0] < 1 or not bool(torch.isfinite(targets).all().item()):
        raise ValueError("targets must be nonempty and finite")
    global_track_count = int(targets.shape[0])
    if targets.device.type != "cpu" or targets.dtype != DTYPE:
        raise ValueError("targets must be CPU float64")
    expected_gradient_shapes = (
        (site_count, 4),
        (site_count,),
        (site_count,),
        (site_count, 3),
    )
    source_pointers = {tensor.untyped_storage().data_ptr() for tensor in source_tensors}
    gradient_pointers = []
    for tensor, shape in zip(gradients.tensors, expected_gradient_shapes, strict=True):
        if tuple(tensor.shape) != shape or tensor.dtype != DTYPE or tensor.device.type != "cpu":
            raise ValueError("caller-owned global gradient buffers have incompatible shape/dtype/device")
        if tensor.requires_grad or not tensor.is_contiguous():
            raise ValueError("caller-owned global gradient buffers must be contiguous non-autograd tensors")
        gradient_pointers.append(tensor.untyped_storage().data_ptr())
    if len(set(gradient_pointers)) != 4 or any(pointer in source_pointers for pointer in gradient_pointers):
        raise ValueError("global gradient buffers must own distinct storage")

    chart_ids = tuple(chart.chart_id for chart in charts)
    if any(not chart_id.strip() for chart_id in chart_ids) or len(set(chart_ids)) != len(chart_ids):
        raise ValueError("topology chart ids must be nonempty and unique")
    if len({id(chart.world_snapshot) for chart in charts}) != len(charts):
        raise ValueError("each topology chart must own a separate prepared world snapshot")
    for chart_id, chart in enumerate(charts):
        if not math.isfinite(chart.t_min) or not math.isfinite(chart.t_max) or chart.t_max <= chart.t_min:
            raise ValueError("topology chart bounds must be finite and strictly increasing")
        if chart_id and charts[chart_id - 1].t_max != chart.t_min:
            raise ValueError("topology charts must form one ordered contiguous half-open partition")
        chart.world_snapshot.assert_current()
        _validate_piecewise_atlas(chart.world_snapshot.atlas)
        atlas = chart.world_snapshot.atlas
        if atlas.track_count != global_track_count:
            raise ValueError("every topology chart must contain all global tracks")
        if atlas.charts[0].transfer_atlas.t_min != chart.t_min or atlas.charts[-1].transfer_atlas.t_max != chart.t_max:
            raise ValueError("prepared snapshot interval must match its topology chart bounds")
        source_site_ids = torch.as_tensor(chart.source_site_ids)
        pairs = torch.as_tensor(chart.boundary_site_pairs)
        local_site_count = int(chart.world_snapshot.site_density.numel())
        if (
            source_site_ids.dtype != torch.int64
            or source_site_ids.device.type != "cpu"
            or source_site_ids.ndim != 1
            or source_site_ids.numel() != local_site_count
        ):
            raise ValueError("source_site_ids must be CPU int64 with one row per snapshot site")
        if source_site_ids.numel() and (
            int(source_site_ids.min().item()) < 0 or int(source_site_ids.max().item()) >= site_count
        ):
            raise ValueError("source_site_ids leave the common site table")
        if int(torch.unique(source_site_ids).numel()) != local_site_count:
            raise ValueError("source_site_ids must not contain duplicates")
        if pairs.dtype != torch.int64 or pairs.device.type != "cpu" or tuple(pairs.shape) != (
            int(chart.world_snapshot.boundary.shape[0]),
            2,
        ):
            raise ValueError("boundary_site_pairs must be CPU int64 with one row per local boundary")
        if pairs.numel() and (int(pairs.min().item()) < 0 or int(pairs.max().item()) >= local_site_count):
            raise ValueError("boundary_site_pairs leave the snapshot-local site table")
        local_geometry = site_geometry.index_select(0, source_site_ids)
        local_density = site_density.index_select(0, source_site_ids)
        local_color = site_color.index_select(0, source_site_ids)
        if not torch.equal(local_density, chart.world_snapshot.site_density) or not torch.equal(
            local_color,
            chart.world_snapshot.site_color,
        ):
            raise ValueError("prepared snapshot material rows do not match the common site table")
        expected_boundary = power_boundary_parameters(local_geometry, pairs)
        if not torch.equal(expected_boundary, chart.world_snapshot.boundary):
            raise ValueError("prepared snapshot boundaries do not derive from the common site table")


def _partition_sample_ids(
    charts: tuple[PreparedTopologyLieChart, ...],
    times: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    assigned = torch.zeros(times.numel(), dtype=torch.bool)
    sample_ids_by_chart = []
    for chart_id, chart in enumerate(charts):
        mask = times >= chart.t_min
        mask &= times <= chart.t_max if chart_id == len(charts) - 1 else times < chart.t_max
        sample_ids = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if sample_ids.numel() and bool(assigned.index_select(0, sample_ids).any().item()):
            raise ValueError("topology chart partition assigned one sample more than once")
        assigned[sample_ids] = True
        sample_ids_by_chart.append(sample_ids)
    if not bool(assigned.all().item()):
        raise ValueError("requested times leave the topology chart partition")
    return tuple(sample_ids_by_chart)


__all__ = [
    "PiecewiseTopologyStagedResult",
    "PreparedTopologyLieChart",
    "TopologyEventGradientMetadata",
    "piecewise_topology_staged_lie_mse_vjp",
]
