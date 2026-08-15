"""Bounded continuous acceptance for a prepared adaptive affine-Lie atlas.

This wrapper is the provenance and aggregation seam missing from the isolated
single-chart certificate.  It accepts only a :class:`PreparedStagedLieWorld`,
replays the deterministic refresh to prove that every stored chart still
belongs to the bound world snapshot, and then runs the continuous interval
jet certificate on every selected chart.  It performs no dense validation
sampling and never changes ranks or chart boundaries.

When active power sites ``[S,5]`` and boundary pairs ``[B,2]`` are supplied,
the wrapper verifies that they regenerate the prepared boundary tensor
bit-for-bit.  If ``E_b`` bounds every entry of the transfer-Jacobian error with
respect to boundary parameters and ``B(s)`` is the active power-boundary map,
then

``max |Delta J_site| <= E_b * max_j sum_k |d B_k / d s_j|``.

The column one-norm is evaluated exactly from the binary64 site snapshot and
rounded upward.  This covers all four site coordinates and the power weight;
it is conservative because the single boundary-block maximum is shared across
all active faces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import torch
from compiled_lie_world_adjoint import refresh_fixed_topology_lie_world_atlas
from compiled_transfer_adjoint import power_boundary_parameters
from continuous_lie_jet_certificate import (
    ContinuousCertificateError,
    ContinuousLieJetCertificate,
    _float_up,
    certify_fixed_topology_lie_jet,
    certify_fixed_topology_lie_jet_track_local,
)
from continuous_owner_identity_certificate import (
    ContinuousOwnerIdentityCertificate,
    ContinuousOwnerIdentityError,
    certify_fixed_word_owner_identity,
)
from staged_compiled_lie_adjoint import PreparedStagedLieWorld, _validate_piecewise_atlas


@dataclass(frozen=True)
class ContinuousAdaptiveLieCertificationPolicy:
    """Explicit tolerances and finite work budget for continuous acceptance."""

    transfer_tolerance: float
    world_jacobian_tolerance: float
    site_geometry_jacobian_tolerance: float | None = None
    max_split_depth: int = 8
    max_leaves_per_chart: int = 1024
    max_interval_jet_work_units_per_chart: int = 100_000
    arithmetic_fraction_bits: int = 96
    require_compiled_lie_cone: bool = True
    compute_point_witnesses: bool = False
    owner_identity_tolerance: float | None = None
    owner_max_split_depth: int = 14
    owner_max_leaves_per_chart: int = 4096
    owner_max_work_units_per_chart: int = 2_000_000
    certificate_mode: Literal["dense_oracle", "track_local_sparse"] = "dense_oracle"
    max_local_dual_dimension: int = 512


@dataclass(frozen=True)
class ContinuousAdaptiveLieChartAcceptance:
    """Continuous result for one frozen selected chart."""

    chart_index: int
    t_min: float
    t_max: float
    node_count: int
    passed: bool
    certificate: ContinuousLieJetCertificate | None
    owner_identity_certificate: ContinuousOwnerIdentityCertificate | None
    site_geometry_jacobian_error_upper_bound: float | None
    estimated_interval_jet_work_units: int
    failure_reasons: tuple[str, ...]
    rank_or_certificate_limit: bool


@dataclass(frozen=True)
class ContinuousAdaptiveLieAcceptance:
    """Aggregate fail-closed result for one prepared piecewise atlas."""

    passed: bool
    policy: ContinuousAdaptiveLieCertificationPolicy
    charts: tuple[ContinuousAdaptiveLieChartAcceptance, ...]
    maximum_transfer_error_upper_bound: float | None
    maximum_world_jacobian_error_upper_bound: float | None
    maximum_site_geometry_jacobian_error_upper_bound: float | None
    maximum_owner_difference_upper_bound: float | None
    minimum_cut_denominator_absolute_lower_bound: float | None
    minimum_fiber_speed_lower_bound: float | None
    minimum_coordinate_segment_length_lower_bound: float | None
    minimum_physical_segment_length_lower_bound: float | None
    total_certificate_leaves: int
    total_owner_certificate_leaves: int
    maximum_estimated_interval_jet_work_units: int
    boundary_to_site_maximum_column_l1_norm: float | None
    failure_reasons: tuple[str, ...]
    rank_or_certificate_limit: bool
    continuous_time_coverage: bool
    atlas_world_provenance_certified: bool
    optimizer_site_geometry_covered: bool
    optimizer_site_geometry_accepted: bool
    continuous_acceptance_used_sampling: bool = False
    atlas_selection_was_not_reperformed: bool = True
    owner_identity_certified: bool = False
    runtime_floating_point_roundoff_certified: bool = False


def certify_prepared_adaptive_lie_world(
    world_snapshot: PreparedStagedLieWorld,
    *,
    policy: ContinuousAdaptiveLieCertificationPolicy,
    sites: torch.Tensor | None = None,
    boundary_pairs: torch.Tensor | None = None,
) -> ContinuousAdaptiveLieAcceptance:
    """Continuously certify every selected chart in a prepared world snapshot."""

    _validate_policy(policy)
    _assert_prepared_atlas_provenance(world_snapshot)
    geometry = _prepare_power_geometry(
        world_snapshot,
        sites=sites,
        boundary_pairs=boundary_pairs,
        site_tolerance=policy.site_geometry_jacobian_tolerance,
    )
    geometry_norm = None if geometry is None else geometry[2]
    chart_results = []
    for chart_index, chart in enumerate(world_snapshot.atlas.charts):
        world_snapshot.assert_current()
        estimated_work = _estimated_interval_jet_work_units(
            chart,
            world_snapshot,
            maximum_leaf_count=policy.max_leaves_per_chart,
            certificate_mode=policy.certificate_mode,
        )
        if estimated_work > policy.max_interval_jet_work_units_per_chart:
            chart_results.append(
                ContinuousAdaptiveLieChartAcceptance(
                    chart_index=chart_index,
                    t_min=chart.transfer_atlas.t_min,
                    t_max=chart.transfer_atlas.t_max,
                    node_count=chart.node_count,
                    passed=False,
                    certificate=None,
                    owner_identity_certificate=None,
                    site_geometry_jacobian_error_upper_bound=None,
                    estimated_interval_jet_work_units=estimated_work,
                    failure_reasons=(
                        "continuous_certificate_work_budget_exceeded: "
                        f"estimated={estimated_work}, "
                        f"limit={policy.max_interval_jet_work_units_per_chart}",
                    ),
                    rank_or_certificate_limit=True,
                )
            )
            continue
        try:
            arguments = {
                "boundary": world_snapshot.boundary,
                "ray_coefficients": world_snapshot.ray_coefficients,
                "site_density": world_snapshot.site_density,
                "site_color": world_snapshot.site_color,
                "transfer_tolerance": policy.transfer_tolerance,
                "world_jacobian_tolerance": policy.world_jacobian_tolerance,
                "max_split_depth": policy.max_split_depth,
                "max_leaf_count": policy.max_leaves_per_chart,
                "arithmetic_fraction_bits": policy.arithmetic_fraction_bits,
                "compute_point_witnesses": policy.compute_point_witnesses,
            }
            if policy.certificate_mode == "dense_oracle":
                certificate = certify_fixed_topology_lie_jet(chart, **arguments)
            else:
                certificate = certify_fixed_topology_lie_jet_track_local(
                    chart,
                    max_local_dual_dimension=policy.max_local_dual_dimension,
                    **arguments,
                )
        except ContinuousCertificateError as error:
            reason = _certificate_error_reason(error)
            chart_results.append(
                ContinuousAdaptiveLieChartAcceptance(
                    chart_index=chart_index,
                    t_min=chart.transfer_atlas.t_min,
                    t_max=chart.transfer_atlas.t_max,
                    node_count=chart.node_count,
                    passed=False,
                    certificate=None,
                    owner_identity_certificate=None,
                    site_geometry_jacobian_error_upper_bound=None,
                    estimated_interval_jet_work_units=estimated_work,
                    failure_reasons=(reason,),
                    rank_or_certificate_limit=True,
                )
            )
            continue

        owner_certificate = None
        owner_reasons: list[str] = []
        owner_limit = False
        if policy.owner_identity_tolerance is not None:
            if geometry is None:
                owner_reasons.append("continuous_owner_identity_requires_power_sites")
            else:
                try:
                    owner_certificate = certify_fixed_word_owner_identity(
                        sites=geometry[0],
                        boundary=world_snapshot.boundary,
                        ray_coefficients=world_snapshot.ray_coefficients,
                        words=chart.words,
                        t_min=chart.transfer_atlas.t_min,
                        t_max=chart.transfer_atlas.t_max,
                        near=chart.near,
                        far=chart.far,
                        ownership_tolerance=policy.owner_identity_tolerance,
                        max_split_depth=policy.owner_max_split_depth,
                        max_leaf_count=policy.owner_max_leaves_per_chart,
                        max_work_units=policy.owner_max_work_units_per_chart,
                        arithmetic_fraction_bits=policy.arithmetic_fraction_bits,
                    )
                except ContinuousOwnerIdentityError as error:
                    owner_reasons.append(f"continuous_owner_identity_unproved: {error}")
                    owner_limit = "budget" in str(error) or "maximum split depth" in str(error)

        site_upper = (
            None
            if geometry_norm is None
            else _multiply_upper(
                certificate.world_jacobian_error_upper_bound_by_block["boundary"],
                geometry_norm,
            )
        )
        reasons = list(owner_reasons)
        if certificate.transfer_error_upper_bound > policy.transfer_tolerance:
            reasons.append("continuous_transfer_tolerance_exceeded")
        if certificate.world_jacobian_error_upper_bound > policy.world_jacobian_tolerance:
            reasons.append("continuous_world_jacobian_tolerance_exceeded")
        if policy.require_compiled_lie_cone and not certificate.compiled_lie_cone_certified:
            reasons.append("continuous_compiled_lie_cone_unproved")
        if (
            site_upper is not None
            and policy.site_geometry_jacobian_tolerance is not None
            and site_upper > policy.site_geometry_jacobian_tolerance
        ):
            reasons.append("continuous_site_geometry_jacobian_tolerance_exceeded")
        limit = bool(reasons) and certificate.deepest_split >= policy.max_split_depth
        chart_results.append(
            ContinuousAdaptiveLieChartAcceptance(
                chart_index=chart_index,
                t_min=chart.transfer_atlas.t_min,
                t_max=chart.transfer_atlas.t_max,
                node_count=chart.node_count,
                passed=not reasons,
                certificate=certificate,
                owner_identity_certificate=owner_certificate,
                site_geometry_jacobian_error_upper_bound=site_upper,
                estimated_interval_jet_work_units=estimated_work,
                failure_reasons=tuple(reasons),
                rank_or_certificate_limit=limit or owner_limit,
            )
        )
    world_snapshot.assert_current()

    results = tuple(chart_results)
    certificates = tuple(result.certificate for result in results if result.certificate is not None)
    owner_certificates = tuple(
        result.owner_identity_certificate
        for result in results
        if result.owner_identity_certificate is not None
    )
    transfer_bounds = tuple(certificate.transfer_error_upper_bound for certificate in certificates)
    jacobian_bounds = tuple(certificate.world_jacobian_error_upper_bound for certificate in certificates)
    site_bounds = tuple(
        result.site_geometry_jacobian_error_upper_bound
        for result in results
        if result.site_geometry_jacobian_error_upper_bound is not None
    )
    denominator_margins = tuple(
        certificate.minimum_cut_denominator_absolute_lower_bound
        for certificate in certificates
        if certificate.minimum_cut_denominator_absolute_lower_bound is not None
    )
    failures = tuple(
        f"chart[{result.chart_index}]: {reason}" for result in results for reason in result.failure_reasons
    )
    geometry_covered = geometry is not None and len(site_bounds) == len(results)
    geometry_accepted = (
        geometry_covered
        and policy.site_geometry_jacobian_tolerance is not None
        and all(bound <= policy.site_geometry_jacobian_tolerance for bound in site_bounds)
    )
    return ContinuousAdaptiveLieAcceptance(
        passed=bool(results) and all(result.passed for result in results),
        policy=policy,
        charts=results,
        maximum_transfer_error_upper_bound=max(transfer_bounds, default=None),
        maximum_world_jacobian_error_upper_bound=max(jacobian_bounds, default=None),
        maximum_site_geometry_jacobian_error_upper_bound=max(site_bounds, default=None),
        maximum_owner_difference_upper_bound=max(
            (
                certificate.maximum_owner_difference_upper_bound
                for certificate in owner_certificates
            ),
            default=None,
        ),
        minimum_cut_denominator_absolute_lower_bound=min(denominator_margins, default=None),
        minimum_fiber_speed_lower_bound=_minimum_certificate_field(
            certificates,
            "minimum_fiber_speed_lower_bound",
        ),
        minimum_coordinate_segment_length_lower_bound=_minimum_certificate_field(
            certificates,
            "minimum_coordinate_segment_length_lower_bound",
        ),
        minimum_physical_segment_length_lower_bound=_minimum_certificate_field(
            certificates,
            "minimum_physical_segment_length_lower_bound",
        ),
        total_certificate_leaves=sum(certificate.leaf_count for certificate in certificates),
        total_owner_certificate_leaves=sum(
            certificate.leaf_count for certificate in owner_certificates
        ),
        maximum_estimated_interval_jet_work_units=max(
            (result.estimated_interval_jet_work_units for result in results),
            default=0,
        ),
        boundary_to_site_maximum_column_l1_norm=geometry_norm,
        failure_reasons=failures,
        rank_or_certificate_limit=any(result.rank_or_certificate_limit for result in results),
        continuous_time_coverage=(
            len(certificates) == len(results)
            and (
                policy.owner_identity_tolerance is None
                or len(owner_certificates) == len(results)
            )
        ),
        atlas_world_provenance_certified=True,
        optimizer_site_geometry_covered=geometry_covered,
        optimizer_site_geometry_accepted=geometry_accepted,
        owner_identity_certified=(
            policy.owner_identity_tolerance is not None
            and len(owner_certificates) == len(results)
            and all(certificate.passed for certificate in owner_certificates)
        ),
    )


def power_boundary_to_site_maximum_column_l1_norm(
    sites: torch.Tensor,
    boundary_pairs: torch.Tensor,
) -> float:
    """Return an outward upper bound on ``max_j sum_k |dB_k/ds_j|``."""

    sites_f64, pairs_i64 = _validate_power_inputs(sites, boundary_pairs)
    column_sums = [[Fraction(0) for _ in range(5)] for _ in range(int(sites_f64.shape[0]))]
    for left, right in pairs_i64.tolist():
        for site_id in (int(left), int(right)):
            for coordinate in range(4):
                value = Fraction.from_float(float(sites_f64[site_id, coordinate].item()))
                column_sums[site_id][coordinate] += 2 + 2 * abs(value)
            column_sums[site_id][4] += 1
    exact_norm = max((value for row in column_sums for value in row), default=Fraction(0))
    return _float_up(exact_norm)


def _assert_prepared_atlas_provenance(world_snapshot: PreparedStagedLieWorld) -> None:
    world_snapshot.assert_current()
    _validate_piecewise_atlas(world_snapshot.atlas)
    tensors = (
        world_snapshot.boundary,
        world_snapshot.ray_coefficients,
        world_snapshot.site_density,
        world_snapshot.site_color,
    )
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise ValueError("continuous adaptive certification is CPU-only")
    refreshed = refresh_fixed_topology_lie_world_atlas(
        world_snapshot.atlas,
        assume_fixed_topology=True,
        boundary=world_snapshot.boundary,
        ray_coefficients=world_snapshot.ray_coefficients,
        site_density=world_snapshot.site_density,
        site_color=world_snapshot.site_color,
    )
    if len(refreshed.charts) != len(world_snapshot.atlas.charts):
        raise ValueError("prepared atlas chart count changed during provenance replay")
    for chart_id, (stored, expected) in enumerate(zip(world_snapshot.atlas.charts, refreshed.charts, strict=True)):
        tensor_pairs = (
            (stored.transfer_atlas.node_times, expected.transfer_atlas.node_times),
            (stored.transfer_atlas.fit_matrix, expected.transfer_atlas.fit_matrix),
            (stored.transfer_atlas.coefficients, expected.transfer_atlas.coefficients),
            (stored.node_chart, expected.node_chart),
            (stored.depth_coefficient_incidence, expected.depth_coefficient_incidence),
            (stored.sparse_depth_coefficients, expected.sparse_depth_coefficients),
        )
        if any(not torch.equal(left, right) for left, right in tensor_pairs):
            raise ValueError(f"prepared atlas chart {chart_id} no longer matches its bound world snapshot")
    world_snapshot.assert_current()


def _prepare_power_geometry(
    world_snapshot: PreparedStagedLieWorld,
    *,
    sites: torch.Tensor | None,
    boundary_pairs: torch.Tensor | None,
    site_tolerance: float | None,
) -> tuple[torch.Tensor, torch.Tensor, float] | None:
    if (sites is None) != (boundary_pairs is None):
        raise ValueError("sites and boundary_pairs must be supplied together")
    if sites is None:
        if site_tolerance is not None:
            raise ValueError("site geometry tolerance requires active sites and boundary_pairs")
        return None
    if site_tolerance is None:
        raise ValueError("active sites require an explicit site geometry Jacobian tolerance")
    sites_f64, pairs_i64 = _validate_power_inputs(sites, boundary_pairs)
    if int(sites_f64.shape[0]) != int(world_snapshot.site_density.numel()):
        raise ValueError("active power sites must match the prepared site table")
    if int(pairs_i64.shape[0]) != int(world_snapshot.boundary.shape[0]):
        raise ValueError("boundary_pairs must contain one row per prepared boundary")
    derived = power_boundary_parameters(sites_f64, pairs_i64)
    if not torch.equal(derived, world_snapshot.boundary.detach().cpu()):
        raise ValueError("active sites/boundary_pairs do not reproduce the prepared boundary snapshot")
    return (
        sites_f64,
        pairs_i64,
        power_boundary_to_site_maximum_column_l1_norm(sites_f64, pairs_i64),
    )


def _validate_power_inputs(
    sites: torch.Tensor,
    boundary_pairs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sites_f64 = torch.as_tensor(sites, dtype=torch.float64).detach().cpu()
    pairs_i64 = torch.as_tensor(boundary_pairs, dtype=torch.int64).detach().cpu()
    if sites_f64.ndim != 2 or sites_f64.shape[1] != 5:
        raise ValueError("sites must have shape [S,5]")
    if not bool(torch.isfinite(sites_f64).all().item()):
        raise ValueError("sites must be finite")
    if pairs_i64.ndim != 2 or pairs_i64.shape[1] != 2:
        raise ValueError("boundary_pairs must have shape [B,2]")
    if pairs_i64.numel():
        if int(pairs_i64.min().item()) < 0 or int(pairs_i64.max().item()) >= int(sites_f64.shape[0]):
            raise ValueError("boundary_pairs contain a site id outside sites")
        if bool((pairs_i64[:, 0] == pairs_i64[:, 1]).any().item()):
            raise ValueError("a power boundary pair must contain two distinct sites")
        rows = tuple(tuple(int(value) for value in row) for row in pairs_i64.tolist())
        if len(set(rows)) != len(rows):
            raise ValueError("boundary_pairs rows must be unique")
    return sites_f64, pairs_i64


def _validate_policy(policy: ContinuousAdaptiveLieCertificationPolicy) -> None:
    tolerances = (policy.transfer_tolerance, policy.world_jacobian_tolerance)
    if any(not math.isfinite(value) or value < 0 for value in tolerances):
        raise ValueError("transfer and world-Jacobian tolerances must be finite and non-negative")
    if policy.site_geometry_jacobian_tolerance is not None and (
        not math.isfinite(policy.site_geometry_jacobian_tolerance) or policy.site_geometry_jacobian_tolerance < 0
    ):
        raise ValueError("site geometry Jacobian tolerance must be finite and non-negative")
    if policy.max_split_depth < 0 or policy.max_leaves_per_chart < 1:
        raise ValueError("certificate split depth and leaf budget must be non-negative/positive")
    if policy.max_interval_jet_work_units_per_chart < 1:
        raise ValueError("certificate work-unit budget must be positive")
    if policy.arithmetic_fraction_bits < 64:
        raise ValueError("arithmetic_fraction_bits must be at least 64")
    if policy.certificate_mode not in {"dense_oracle", "track_local_sparse"}:
        raise ValueError("certificate_mode must be dense_oracle or track_local_sparse")
    if policy.max_local_dual_dimension < 1:
        raise ValueError("max_local_dual_dimension must be positive")
    if policy.owner_identity_tolerance is not None and (
        not math.isfinite(policy.owner_identity_tolerance)
        or policy.owner_identity_tolerance < 0
    ):
        raise ValueError("owner identity tolerance must be finite and non-negative")
    if (
        policy.owner_max_split_depth < 0
        or policy.owner_max_leaves_per_chart < 1
        or policy.owner_max_work_units_per_chart < 1
    ):
        raise ValueError("owner certificate split, leaf, and work budgets must be positive")


def _multiply_upper(left: float, right: float) -> float:
    return _float_up(Fraction.from_float(left) * Fraction.from_float(right))


def _estimated_interval_jet_work_units(
    chart: object,
    world_snapshot: PreparedStagedLieWorld,
    *,
    maximum_leaf_count: int,
    certificate_mode: Literal["dense_oracle", "track_local_sparse"],
) -> int:
    if certificate_mode == "track_local_sparse":
        rank = int(chart.node_count)
        total = 0
        for word in chart.words:
            cuts = torch.cat((word.left_cut_ids, word.right_cut_ids))
            boundary_count = len({int(value) for value in cuts.tolist() if int(value) >= 0})
            site_count = len({int(value) for value in word.owners.tolist()})
            local_parameter_count = 12 + 9 * boundary_count + 4 * site_count
            run_count = int(word.owners.numel())
            total += local_parameter_count * local_parameter_count + 4 * local_parameter_count * (
                rank * rank + maximum_leaf_count * run_count
            )
        return total
    boundary_parameters = int(world_snapshot.boundary.numel())
    ray_parameters = int(world_snapshot.ray_coefficients.numel())
    mobius_parameters = int(chart.depth_coefficient_incidence.numel()) * 2
    material_parameters = int(world_snapshot.site_density.numel()) + int(world_snapshot.site_color.numel())
    parameter_count = boundary_parameters + ray_parameters + mobius_parameters + material_parameters
    run_count = sum(int(word.owners.numel()) for word in chart.words)
    rank = int(chart.node_count)
    # The exact-rational implementation forms four chart components.  Node
    # linearization is quadratic in rank because it applies the dense fit
    # matrix; each leaf replays every run and parameter once to enclose the
    # primal, Jacobian, and mixed time/world jet.
    return 4 * parameter_count * (rank * rank + maximum_leaf_count * run_count)


def _minimum_certificate_field(
    certificates: tuple[ContinuousLieJetCertificate, ...],
    field: str,
) -> float | None:
    values = tuple(float(getattr(certificate, field)) for certificate in certificates)
    return min(values, default=None)


def _certificate_error_reason(error: ContinuousCertificateError) -> str:
    message = str(error)
    if "max_leaf_count" in message:
        return f"continuous_certificate_leaf_budget_exhausted: {message}"
    if "max depth" in message:
        return f"continuous_precondition_unproved_at_max_depth: {message}"
    return f"continuous_precondition_unproved: {message}"


__all__ = [
    "ContinuousAdaptiveLieAcceptance",
    "ContinuousAdaptiveLieCertificationPolicy",
    "ContinuousAdaptiveLieChartAcceptance",
    "certify_prepared_adaptive_lie_world",
    "power_boundary_to_site_maximum_column_l1_norm",
]
