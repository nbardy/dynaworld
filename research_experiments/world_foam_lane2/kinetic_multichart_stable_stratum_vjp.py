"""End-to-end frozen-stratum VJP for compact kinetic WorldFoam charts.

The expensive reverse path is split at the compact transfer nodes. Requested
samples are streamed in bounded blocks and reduced to ``O(sum_c J_c)`` node
transfer cotangents. Each certified owner chart is then replayed exactly once
at its ``J_c`` compiler nodes through :func:`kinetic_p0_compiler_node_vjp`.
Consequently the geometry/material reverse is independent of requested frame
density; only the cheap sample-to-node reduction remains ``O(F J)``.

This is a stable-stratum derivative, not a derivative of the compiler. Owner
events, chart dispatch, chart endpoints, compiler-node times, and rank
selection are frozen. The exact owner-program digest supplies continuous
*topology* provenance, but no continuous geometry-Jacobian or compact-gradient
approximation bound is claimed here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from kinetic_multichart_transfer_program import (
    KineticMultiChartP0Program,
    KineticMultiChartP0Transfer,
    reduce_kinetic_multichart_mse_to_node_transfers,
)
from kinetic_stable_stratum_vjp import (
    DERIVATIVE_SCOPE as NODE_DERIVATIVE_SCOPE,
)
from kinetic_stable_stratum_vjp import (
    ObservedStableStratumMargins,
    StableStratumThresholds,
    kinetic_p0_compiler_node_vjp,
    make_frozen_kinetic_owner_word,
)

DTYPE = torch.float64
DERIVATIVE_SCOPE = "frozen_exact_owner_charts_fixed_dispatch_fixed_endpoints_fixed_rank_fixed_node_times"


@dataclass(frozen=True)
class BoundKineticMultiChartStableStratumVJP:
    """Immutable digest binding for one frozen multi-chart derivative."""

    source_content_digest: str
    owner_program_semantic_digest: str
    transfer_program_generation_digest: str
    compiler_provenance: str
    chart_topology_certificate_ids: tuple[str, ...]
    binding_digest: str
    derivative_scope: str = DERIVATIVE_SCOPE
    continuous_geometry_approximation_certificate_id: None = None
    requested_sample_sampling_used: bool = False
    event_time_derivatives_included: bool = False
    chart_endpoint_derivatives_included: bool = False
    sample_dispatch_derivatives_included: bool = False
    node_time_or_rank_derivatives_included: bool = False
    continuous_geometry_jacobian_certified: bool = False

    def assert_current(self, program: KineticMultiChartP0Program) -> None:
        """Reject a stale source, owner program, chart program, or certificate."""

        if not isinstance(program, KineticMultiChartP0Program):
            raise TypeError("program must be KineticMultiChartP0Program")
        program.assert_current()
        expected = _provenance_parts(program)
        observed = (
            self.source_content_digest,
            self.owner_program_semantic_digest,
            self.transfer_program_generation_digest,
            self.compiler_provenance,
            self.chart_topology_certificate_ids,
        )
        if observed != expected:
            raise ValueError("stable-stratum VJP provenance does not match the current program")
        if self.binding_digest != _binding_digest(*expected):
            raise ValueError("stable-stratum VJP binding digest mismatch")
        if (
            self.derivative_scope != DERIVATIVE_SCOPE
            or self.continuous_geometry_approximation_certificate_id is not None
            or self.requested_sample_sampling_used
            or self.event_time_derivatives_included
            or self.chart_endpoint_derivatives_included
            or self.sample_dispatch_derivatives_included
            or self.node_time_or_rank_derivatives_included
            or self.continuous_geometry_jacobian_certified
        ):
            raise ValueError("stable-stratum VJP derivative contract changed")


@dataclass(frozen=True)
class KineticMultiChartStableStratumMSEVJP:
    """Compact MSE and source gradients after a streamed node reduction."""

    loss: torch.Tensor
    predictions: torch.Tensor | None
    grad_positions0: torch.Tensor
    grad_velocities: torch.Tensor
    grad_weight_coefficients: torch.Tensor
    grad_ray_coefficients: torch.Tensor
    grad_site_density: torch.Tensor
    grad_site_color: torch.Tensor
    chart_margins: tuple[ObservedStableStratumMargins, ...]
    provenance: BoundKineticMultiChartStableStratumVJP
    accounting: dict[str, int | str | bool]
    derivative_scope: str = DERIVATIVE_SCOPE
    geometry_vjp_implemented: bool = True
    material_vjp_implemented: bool = True
    event_time_derivatives_included: bool = False
    chart_endpoint_derivatives_included: bool = False
    sample_dispatch_derivatives_included: bool = False
    node_time_or_rank_derivatives_included: bool = False
    continuous_geometry_jacobian_certified: bool = False
    continuous_geometry_approximation_certified: bool = False


def bind_kinetic_multichart_stable_stratum_vjp(
    program: KineticMultiChartP0Program,
) -> BoundKineticMultiChartStableStratumVJP:
    """Bind a derivative contract to immutable source/program certificates."""

    if not isinstance(program, KineticMultiChartP0Program):
        raise TypeError("program must be KineticMultiChartP0Program")
    program.assert_current()
    parts = _provenance_parts(program)
    result = BoundKineticMultiChartStableStratumVJP(
        source_content_digest=parts[0],
        owner_program_semantic_digest=parts[1],
        transfer_program_generation_digest=parts[2],
        compiler_provenance=parts[3],
        chart_topology_certificate_ids=parts[4],
        binding_digest=_binding_digest(*parts),
    )
    result.assert_current(program)
    return result


@torch.no_grad()
def kinetic_multichart_stable_stratum_mse_vjp(
    transfer: KineticMultiChartP0Transfer,
    provenance: BoundKineticMultiChartStableStratumVJP,
    times: torch.Tensor,
    targets: torch.Tensor,
    *,
    background: torch.Tensor,
    sample_block_size: int = 16,
    return_predictions: bool = False,
    thresholds: StableStratumThresholds = StableStratumThresholds(),
) -> KineticMultiChartStableStratumMSEVJP:
    """Stream ``F`` samples to nodes, then reverse each frozen chart once.

    The returned parameter gradients have no requested-sample axis. The
    optional predictions are an output, never a reverse tape. Geometry
    gradients are with respect to the fixed chart partition and fixed
    interpolation schedules named by ``provenance``.
    """

    if not isinstance(transfer, KineticMultiChartP0Transfer):
        raise TypeError("transfer must be KineticMultiChartP0Transfer")
    if not isinstance(provenance, BoundKineticMultiChartStableStratumVJP):
        raise TypeError("provenance must be BoundKineticMultiChartStableStratumVJP")
    if not isinstance(thresholds, StableStratumThresholds):
        raise TypeError("thresholds must be StableStratumThresholds")
    provenance.assert_current(transfer.program)

    reduced = reduce_kinetic_multichart_mse_to_node_transfers(
        transfer,
        times,
        targets,
        background=background,
        sample_block_size=sample_block_size,
        return_predictions=return_predictions,
    )
    if len(reduced.grad_chart_node_transfers) != transfer.program.chart_count:
        raise ArithmeticError("sample reduction did not cover every kinetic chart")

    sites = transfer.program.binding.sites
    ray = transfer.program.binding.ray_coefficients.reshape(1, 12)
    grad_positions0 = torch.zeros_like(sites.positions0)
    grad_velocities = torch.zeros_like(sites.velocities)
    grad_weights = torch.zeros_like(sites.weight_coefficients)
    grad_ray = torch.zeros_like(ray)
    grad_density = torch.zeros_like(transfer.site_density)
    grad_color = torch.zeros_like(transfer.site_color)
    margins = []
    run_interactions = 0
    cut_interactions = 0
    owner_margin_evaluations = 0
    peak_chart_node_count = 0
    peak_chart_run_count = 0

    owner_program = transfer.program.binding.program
    for chart_id, (chart, grad_node_transfer) in enumerate(
        zip(
            transfer.program.charts,
            reduced.grad_chart_node_transfers,
            strict=True,
        )
    ):
        node_result = kinetic_p0_compiler_node_vjp(
            sites,
            ray,
            chart.schedule.node_times,
            (make_frozen_kinetic_owner_word(chart.owners),),
            transfer.site_density,
            transfer.site_color,
            grad_node_transfer.unsqueeze(0),
            near=float(owner_program.near),
            far=float(owner_program.far),
            continuous_topology_certificate_id=(provenance.chart_topology_certificate_ids[chart_id]),
            thresholds=thresholds,
        )
        _require_same_node_transfer(
            node_result.node_transfers[0],
            transfer.chart_node_transfers[chart_id],
            chart_id=chart_id,
        )
        if node_result.derivative_scope != NODE_DERIVATIVE_SCOPE:
            raise ValueError("node VJP derivative scope changed")
        if node_result.continuous_topology_certificate_id != (provenance.chart_topology_certificate_ids[chart_id]):
            raise ValueError("node VJP returned stale topology-certificate provenance")

        grad_positions0 += node_result.grad_positions0
        grad_velocities += node_result.grad_velocities
        grad_weights += node_result.grad_weight_coefficients
        grad_ray += node_result.grad_ray_coefficients
        grad_density += node_result.grad_site_density
        grad_color += node_result.grad_site_color
        margins.append(node_result.margins)
        run_interactions += int(node_result.accounting["active_run_node_interactions"])
        cut_interactions += int(node_result.accounting["active_cut_node_interactions"])
        owner_margin_evaluations += int(node_result.accounting["owner_margin_evaluations"])
        peak_chart_node_count = max(peak_chart_node_count, chart.node_count)
        peak_chart_run_count = max(peak_chart_run_count, chart.run_count)

    parameter_gradient_bytes = _tensor_bytes(
        (
            grad_positions0,
            grad_velocities,
            grad_weights,
            grad_ray,
            grad_density,
            grad_color,
        )
    )
    node_cotangent_bytes = int(reduced.accounting["returned_node_transfer_cotangent_bytes"])
    accounting: dict[str, int | str | bool] = {
        "requested_sample_count": int(reduced.accounting["requested_sample_count"]),
        "sample_block_size": sample_block_size,
        "chart_count": transfer.program.chart_count,
        "compile_node_count": transfer.program.total_node_count,
        "peak_chart_node_count": peak_chart_node_count,
        "peak_chart_run_count": peak_chart_run_count,
        "sample_to_node_linear_interactions": int(reduced.accounting["sample_to_node_linear_interactions"]),
        "sample_to_node_dense_fallback_interactions": int(
            reduced.accounting["sample_to_node_dense_fallback_interactions"]
        ),
        "active_run_node_interactions": run_interactions,
        "active_cut_node_interactions": cut_interactions,
        "owner_margin_evaluations": owner_margin_evaluations,
        "world_geometry_material_reverse_node_count": transfer.program.total_node_count,
        "material_prefix_reverse_node_count": transfer.program.total_node_count,
        "node_transfer_cotangent_bytes": node_cotangent_bytes,
        "parameter_gradient_bytes": parameter_gradient_bytes,
        "peak_sample_block_bytes": int(reduced.accounting["peak_sample_block_bytes"]),
        "returned_prediction_bytes": (
            0 if reduced.predictions is None else reduced.predictions.numel() * reduced.predictions.element_size()
        ),
        "frame_dependent_reverse_tape_bytes": 0,
        "retained_requested_sample_bytes": 0,
        "dense_sample_by_chart_state_bytes": 0,
        "requested_frame_sampling_used_for_compile": False,
        "world_reverse_independent_of_requested_frame_count": True,
        "sample_reduction_scaling": "O(F * J_active)",
        "world_reverse_scaling": "O(sum_c J_c * R_c)",
        "owner_validation_scaling": "O(sum_c J_c * S * R_c)",
        "geometry_gradients_emitted": True,
        "material_gradients_emitted": True,
        "event_time_gradients_emitted": False,
        "chart_endpoint_gradients_emitted": False,
        "sample_dispatch_gradients_emitted": False,
        "node_time_or_rank_gradients_emitted": False,
        "continuous_geometry_jacobian_certified": False,
        "continuous_geometry_approximation_certified": False,
    }
    return KineticMultiChartStableStratumMSEVJP(
        loss=reduced.loss,
        predictions=reduced.predictions,
        grad_positions0=grad_positions0,
        grad_velocities=grad_velocities,
        grad_weight_coefficients=grad_weights,
        grad_ray_coefficients=grad_ray.reshape(12),
        grad_site_density=grad_density,
        grad_site_color=grad_color,
        chart_margins=tuple(margins),
        provenance=provenance,
        accounting=accounting,
    )


def _provenance_parts(
    program: KineticMultiChartP0Program,
) -> tuple[str, str, str, str, tuple[str, ...]]:
    binding = program.binding
    chart_certificate_ids = tuple(
        _chart_topology_certificate_id(program, chart_id) for chart_id in range(program.chart_count)
    )
    return (
        binding.source_content_digest,
        binding.program_semantic_digest,
        program.generation_digest,
        binding.compiler_provenance,
        chart_certificate_ids,
    )


def _chart_topology_certificate_id(
    program: KineticMultiChartP0Program,
    chart_id: int,
) -> str:
    chart = program.charts[chart_id]
    return _digest_parts(
        "kinetic-continuous-owner-topology-certificate-v1",
        program.binding.compiler_provenance,
        program.binding.source_content_digest,
        program.binding.program_semantic_digest,
        program.generation_digest,
        chart.chart_id,
        chart.owner_word,
        chart.schedule.t_min,
        chart.schedule.t_max,
        chart.right_closed,
    )


def _binding_digest(
    source_content_digest: str,
    owner_program_semantic_digest: str,
    transfer_program_generation_digest: str,
    compiler_provenance: str,
    chart_topology_certificate_ids: tuple[str, ...],
) -> str:
    return _digest_parts(
        "kinetic-multichart-stable-stratum-vjp-binding-v1",
        source_content_digest,
        owner_program_semantic_digest,
        transfer_program_generation_digest,
        compiler_provenance,
        chart_topology_certificate_ids,
        DERIVATIVE_SCOPE,
    )


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _require_same_node_transfer(
    replayed: torch.Tensor,
    compiled: torch.Tensor,
    *,
    chart_id: int,
) -> None:
    if tuple(replayed.shape) != tuple(compiled.shape) or not torch.allclose(
        replayed,
        compiled,
        rtol=3.0e-13,
        atol=3.0e-13,
    ):
        raise ValueError(f"chart {chart_id} stable-stratum node replay disagrees with the bound transfer")


def _tensor_bytes(tensors: tuple[torch.Tensor, ...]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


__all__ = [
    "BoundKineticMultiChartStableStratumVJP",
    "DERIVATIVE_SCOPE",
    "KineticMultiChartStableStratumMSEVJP",
    "bind_kinetic_multichart_stable_stratum_vjp",
    "kinetic_multichart_stable_stratum_mse_vjp",
]
