"""Frozen-program node VJP for Euclidean affine kinetic WorldFoam cells.

This module is deliberately narrower than a kinetic topology compiler.  It
accepts a caller-certified owner word that is fixed across ``J`` compiler
nodes, evaluates exact P0 ordered transfer at those nodes, and accumulates a
manual sparse VJP into

``positions0, velocities, quadratic weight coefficients, affine rays``.

The reverse never sees requested frame samples.  Its differentiable work is
``O(J * sum_p R_p)`` for word runs and ``O(J * sum_p (R_p - 1))`` for active
cuts.  Sample-to-node reduction belongs upstream.  The deliberately strict
all-competitor trust audit is separate ``O(J * S * sum_p R_p)`` validation
work; it allocates no frame-by-run reverse state.

No derivative is taken through owner discovery, event times, chart endpoints,
node times, rank selection, or certificate decisions.  Node-local denominator,
positive-length, ray-speed, active-tie, and all-competitor owner margins fail
closed.  A separate continuous owner/event certificate is still required
between compiler nodes; its identity is an explicit input so callers cannot
accidentally present this as a full topology derivative.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from kinetic_power_word_compiler import AffineKineticPowerSites

DTYPE = torch.float64
DERIVATIVE_SCOPE = "frozen_owner_topology_chart_rank_node_times"
FIXED_TIME_DERIVATIVE_SCOPE = "frozen_exact_fixed_time_owner_word"


class StableStratumError(ValueError):
    """A supplied owner word is not safely differentiable at a compiler node."""


@dataclass(frozen=True)
class FrozenKineticOwnerWord:
    """One front-to-back owner sequence fixed by an external chart certificate."""

    owners: torch.Tensor

    def __post_init__(self) -> None:
        owners = torch.as_tensor(self.owners, dtype=torch.int64, device="cpu").reshape(-1).clone()
        if owners.numel() < 1:
            raise ValueError("a frozen kinetic owner word must contain at least one run")
        owner_ids = tuple(int(value) for value in owners.tolist())
        if len(set(owner_ids)) != len(owner_ids):
            raise ValueError("an unbounded power-cell line envelope cannot repeat an owner in one word")
        object.__setattr__(self, "owners", owners.contiguous())

    @property
    def run_count(self) -> int:
        return int(self.owners.numel())


@dataclass(frozen=True)
class StableStratumThresholds:
    """Raw and normalized margins required by the frozen-program VJP."""

    minimum_absolute_cut_denominator: float = 1.0e-10
    minimum_cut_cosine: float = 1.0e-8
    minimum_coordinate_length: float = 1.0e-8
    minimum_physical_length: float = 1.0e-8
    minimum_ray_speed: float = 1.0e-8
    minimum_owner_gap: float = 1.0e-9
    active_tie_tolerance: float = 1.0e-9

    def __post_init__(self) -> None:
        values = (
            self.minimum_absolute_cut_denominator,
            self.minimum_cut_cosine,
            self.minimum_coordinate_length,
            self.minimum_physical_length,
            self.minimum_ray_speed,
            self.minimum_owner_gap,
            self.active_tie_tolerance,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("stable-stratum thresholds must be finite and nonnegative")
        if self.minimum_cut_cosine > 1.0:
            raise ValueError("minimum_cut_cosine cannot exceed one")


@dataclass(frozen=True)
class ObservedStableStratumMargins:
    """Smallest accepted node-local margins across the whole sparse program."""

    minimum_absolute_cut_denominator: float
    minimum_cut_cosine: float
    minimum_coordinate_length: float
    minimum_physical_length: float
    minimum_ray_speed: float
    minimum_owner_gap: float
    maximum_active_tie_residual: float


@dataclass(frozen=True)
class KineticP0CompilerNodeVJP:
    """Node transfers and accumulated frozen-program parameter cotangents."""

    node_transfers: torch.Tensor
    grad_positions0: torch.Tensor
    grad_velocities: torch.Tensor
    grad_weight_coefficients: torch.Tensor
    grad_ray_coefficients: torch.Tensor
    grad_site_density: torch.Tensor
    grad_site_color: torch.Tensor
    margins: ObservedStableStratumMargins
    continuous_topology_certificate_id: str
    derivative_scope: str
    event_time_derivatives_included: bool
    chart_endpoint_derivatives_included: bool
    node_time_or_rank_derivatives_included: bool
    accounting: dict[str, int | str | bool]


@dataclass(frozen=True)
class KineticP0NodePhysicalLengthGeometryVJP:
    """Geometry-only reverse of one native ``[J,R]`` node-length chart.

    The node lengths are recomputed from the certified frozen word instead of
    trusted as caller input.  Only their cotangents cross this seam; material,
    sample, event, endpoint, node-time, rank, and compiler-choice derivatives
    are intentionally absent.
    """

    node_physical_lengths: torch.Tensor
    grad_positions0: torch.Tensor
    grad_velocities: torch.Tensor
    grad_weight_coefficients: torch.Tensor
    grad_ray_coefficients: torch.Tensor
    margins: ObservedStableStratumMargins
    continuous_topology_certificate_id: str
    node_physical_length_cotangent_provenance_id: str
    accounting: dict[str, int | str | bool]
    derivative_scope: str = DERIVATIVE_SCOPE
    geometry_vjp_implemented: bool = True
    material_gradients_included: bool = False
    event_time_derivatives_included: bool = False
    chart_endpoint_derivatives_included: bool = False
    node_time_or_rank_derivatives_included: bool = False
    compiler_choice_derivatives_included: bool = False


@dataclass(frozen=True)
class KineticP0FixedTimePhysicalLengthGeometryVJP:
    """Node-local geometry reverse for one exact fixed-time owner word.

    This result is deliberately not a continuous-chart certificate.  Its
    owner word must come from the exact fixed-time lower-envelope discovery
    and is differentiated only on that point's stable stratum.
    """

    physical_lengths: torch.Tensor
    grad_positions0: torch.Tensor
    grad_velocities: torch.Tensor
    grad_weight_coefficients: torch.Tensor
    grad_ray_coefficients: torch.Tensor
    margins: ObservedStableStratumMargins
    fixed_time_owner_discovery_receipt_id: str
    accounting: dict[str, int | str | bool]
    derivative_scope: str = FIXED_TIME_DERIVATIVE_SCOPE
    geometry_vjp_implemented: bool = True
    continuous_topology_certified: bool = False
    fixed_time_lower_envelope_discovery_required: bool = True
    event_time_derivatives_included: bool = False
    compiler_choice_derivatives_included: bool = False


@dataclass(frozen=True)
class _ActiveCut:
    left_owner: int
    right_owner: int
    depth: torch.Tensor
    denominator: torch.Tensor


@dataclass(frozen=True)
class _NodeWordGeometry:
    positions: torch.Tensor
    weights: torch.Tensor
    origin: torch.Tensor
    direction: torch.Tensor
    speed: torch.Tensor
    cuts: torch.Tensor
    active_cuts: tuple[_ActiveCut, ...]
    coordinate_lengths: torch.Tensor
    physical_lengths: torch.Tensor


@dataclass(frozen=True)
class _NodeWordGeometryBars:
    positions: torch.Tensor
    weights: torch.Tensor
    origin: torch.Tensor
    direction: torch.Tensor


@dataclass(frozen=True)
class _KineticGeometryGradients:
    positions0: torch.Tensor
    velocities: torch.Tensor
    weight_coefficients: torch.Tensor
    ray_coefficients: torch.Tensor


@dataclass
class _MarginAccumulator:
    minimum_absolute_cut_denominator: float = math.inf
    minimum_cut_cosine: float = math.inf
    minimum_coordinate_length: float = math.inf
    minimum_physical_length: float = math.inf
    minimum_ray_speed: float = math.inf
    minimum_owner_gap: float = math.inf
    maximum_active_tie_residual: float = 0.0

    def freeze(self) -> ObservedStableStratumMargins:
        return ObservedStableStratumMargins(
            minimum_absolute_cut_denominator=self.minimum_absolute_cut_denominator,
            minimum_cut_cosine=self.minimum_cut_cosine,
            minimum_coordinate_length=self.minimum_coordinate_length,
            minimum_physical_length=self.minimum_physical_length,
            minimum_ray_speed=self.minimum_ray_speed,
            minimum_owner_gap=self.minimum_owner_gap,
            maximum_active_tie_residual=self.maximum_active_tie_residual,
        )


def make_frozen_kinetic_owner_word(
    owners: Sequence[int] | torch.Tensor,
) -> FrozenKineticOwnerWord:
    """Normalize one externally certified owner sequence."""

    return FrozenKineticOwnerWord(torch.as_tensor(owners, dtype=torch.int64))


@torch.no_grad()
def kinetic_p0_node_physical_length_geometry_vjp(
    sites: AffineKineticPowerSites,
    ray_coefficients: torch.Tensor,
    node_times: torch.Tensor | Sequence[float],
    words: Sequence[FrozenKineticOwnerWord],
    grad_node_physical_lengths: torch.Tensor,
    *,
    near: float,
    far: float,
    continuous_topology_certificate_id: str,
    node_physical_length_cotangent_provenance_id: str,
    thresholds: StableStratumThresholds = StableStratumThresholds(),
) -> KineticP0NodePhysicalLengthGeometryVJP:
    """Reverse native node-length bars through one frozen kinetic chart.

    This is the geometry half of the current native precompiled-length seam:
    exactly one ``[12]`` affine ray, one fixed owner word with ``R`` runs, and
    one ``[J,R]`` cotangent tensor.  The function recomputes and validates each
    certified compiler-node geometry once, then differentiates only the
    physical lengths.  It accepts no requested sample or frame axis.
    """

    _require_provenance_id(
        continuous_topology_certificate_id,
        name="continuous_topology_certificate_id",
    )
    _require_provenance_id(
        node_physical_length_cotangent_provenance_id,
        name="node_physical_length_cotangent_provenance_id",
    )
    if not isinstance(sites, AffineKineticPowerSites):
        raise TypeError("sites must be AffineKineticPowerSites")
    if not isinstance(thresholds, StableStratumThresholds):
        raise TypeError("thresholds must be StableStratumThresholds")
    if not math.isfinite(near) or not math.isfinite(far) or far <= near:
        raise ValueError("near/far must be finite with near < far")

    ray = _finite_f64_cpu(ray_coefficients, name="ray_coefficients")
    times = _finite_f64_cpu(node_times, name="node_times").reshape(-1)
    if tuple(ray.shape) != (12,):
        raise ValueError("the native node-length geometry bridge requires ray_coefficients with shape [12]")
    if times.numel() < 1:
        raise ValueError("node_times must contain at least one compiler node")
    if len(words) != 1 or not isinstance(words[0], FrozenKineticOwnerWord):
        raise ValueError("the native node-length geometry bridge requires exactly one FrozenKineticOwnerWord")
    word = words[0]
    if int(word.owners.min().item()) < 0 or int(word.owners.max().item()) >= sites.site_count:
        raise ValueError("the frozen owner word leaves the kinetic site table")
    grad_lengths = _finite_f64_cpu(
        grad_node_physical_lengths,
        name="grad_node_physical_lengths",
    )
    expected_grad_shape = (int(times.numel()), word.run_count)
    if tuple(grad_lengths.shape) != expected_grad_shape:
        raise ValueError(f"grad_node_physical_lengths must have shape {expected_grad_shape}")

    rays = ray.unsqueeze(0)
    gradients = _zero_kinetic_geometry_gradients(sites, rays)
    node_lengths = torch.empty(expected_grad_shape, dtype=DTYPE)
    margins = _MarginAccumulator()
    for node_id, time in enumerate(times):
        positions, weights, time_powers = _kinetic_site_state_at_time(sites, time)
        origin = rays[0, :3] + time * rays[0, 3:6]
        direction = rays[0, 6:9] + time * rays[0, 9:12]
        geometry = _prepare_node_word_geometry(
            positions,
            weights,
            origin,
            direction,
            word,
            near=near,
            far=far,
            thresholds=thresholds,
            margins=margins,
            track_id=0,
            node_id=node_id,
        )
        node_lengths[node_id].copy_(geometry.physical_lengths)
        _accumulate_node_geometry_bars_(
            gradients,
            _node_word_physical_length_vjp(
                geometry,
                grad_lengths[node_id],
            ),
            time=time,
            time_powers=time_powers,
            track_id=0,
        )

    node_count = int(times.numel())
    owner_margin_evaluations = node_count * 3 * word.run_count * (sites.site_count - 1)
    accounting: dict[str, int | str | bool] = {
        "track_count": 1,
        "compiler_node_count": node_count,
        "ordered_run_count": word.run_count,
        "node_geometry_recompute_count": node_count,
        "node_geometry_recomputed_once_per_node": True,
        "physical_length_reverse_interactions": node_count * word.run_count,
        "active_cut_node_interactions": node_count * (word.run_count - 1),
        "owner_margin_evaluations": owner_margin_evaluations,
        "requested_sample_count_used": 0,
        "requested_frame_count_used": 0,
        "frame_by_run_reverse_state_allocated": False,
        "reverse_interaction_scaling": "O(J * R)",
        "validation_interaction_scaling": "O(J * S * R)",
        "material_gradient_tensors_emitted": 0,
        "event_time_derivatives_included": False,
        "chart_endpoint_derivatives_included": False,
        "node_time_or_rank_derivatives_included": False,
        "compiler_choice_derivatives_included": False,
    }
    return KineticP0NodePhysicalLengthGeometryVJP(
        node_physical_lengths=node_lengths,
        grad_positions0=gradients.positions0,
        grad_velocities=gradients.velocities,
        grad_weight_coefficients=gradients.weight_coefficients,
        grad_ray_coefficients=gradients.ray_coefficients[0],
        margins=margins.freeze(),
        continuous_topology_certificate_id=continuous_topology_certificate_id,
        node_physical_length_cotangent_provenance_id=(node_physical_length_cotangent_provenance_id),
        accounting=accounting,
    )


@torch.no_grad()
def kinetic_p0_fixed_time_physical_length_geometry_vjp(
    sites: AffineKineticPowerSites,
    ray_coefficients: torch.Tensor,
    *,
    time: float,
    owners: Sequence[int] | torch.Tensor,
    grad_physical_lengths: torch.Tensor,
    near: float,
    far: float,
    fixed_time_owner_discovery_receipt_id: str,
    thresholds: StableStratumThresholds = StableStratumThresholds(),
) -> KineticP0FixedTimePhysicalLengthGeometryVJP:
    """Reverse one exact fixed-time word without claiming chart continuity."""

    _require_provenance_id(
        fixed_time_owner_discovery_receipt_id,
        name="fixed_time_owner_discovery_receipt_id",
    )
    if not isinstance(sites, AffineKineticPowerSites):
        raise TypeError("sites must be AffineKineticPowerSites")
    if not isinstance(thresholds, StableStratumThresholds):
        raise TypeError("thresholds must be StableStratumThresholds")
    if not math.isfinite(time):
        raise ValueError("time must be finite")
    if not math.isfinite(near) or not math.isfinite(far) or far <= near:
        raise ValueError("near/far must be finite with near < far")

    ray = _finite_f64_cpu(ray_coefficients, name="ray_coefficients")
    if tuple(ray.shape) != (12,):
        raise ValueError("fixed-time geometry requires ray_coefficients with shape [12]")
    word = make_frozen_kinetic_owner_word(owners)
    if int(word.owners.min().item()) < 0 or int(word.owners.max().item()) >= sites.site_count:
        raise ValueError("the fixed-time owner word leaves the kinetic site table")
    grad_lengths = _finite_f64_cpu(
        grad_physical_lengths,
        name="grad_physical_lengths",
    ).reshape(-1)
    if tuple(grad_lengths.shape) != (word.run_count,):
        raise ValueError(
            f"grad_physical_lengths must have shape {(word.run_count,)}"
        )

    time_tensor = torch.tensor(float(time), dtype=DTYPE)
    positions, weights, time_powers = _kinetic_site_state_at_time(
        sites,
        time_tensor,
    )
    origin = ray[:3] + time_tensor * ray[3:6]
    direction = ray[6:9] + time_tensor * ray[9:12]
    margins = _MarginAccumulator()
    geometry = _prepare_node_word_geometry(
        positions,
        weights,
        origin,
        direction,
        word,
        near=near,
        far=far,
        thresholds=thresholds,
        margins=margins,
        track_id=0,
        node_id=0,
    )
    gradients = _zero_kinetic_geometry_gradients(sites, ray.unsqueeze(0))
    _accumulate_node_geometry_bars_(
        gradients,
        _node_word_physical_length_vjp(geometry, grad_lengths),
        time=time_tensor,
        time_powers=time_powers,
        track_id=0,
    )
    accounting: dict[str, int | str | bool] = {
        "fixed_time_lower_envelope_word_count": 1,
        "continuous_topology_certificate_count": 0,
        "ordered_run_count": word.run_count,
        "physical_length_reverse_interactions": word.run_count,
        "active_cut_reverse_interactions": word.run_count - 1,
        "owner_margin_evaluations": 3 * word.run_count * (sites.site_count - 1),
        "requested_frame_count_used": 1,
        "frame_by_run_reverse_state_allocated": False,
        "event_time_derivatives_included": False,
        "compiler_choice_derivatives_included": False,
    }
    return KineticP0FixedTimePhysicalLengthGeometryVJP(
        physical_lengths=geometry.physical_lengths,
        grad_positions0=gradients.positions0,
        grad_velocities=gradients.velocities,
        grad_weight_coefficients=gradients.weight_coefficients,
        grad_ray_coefficients=gradients.ray_coefficients[0],
        margins=margins.freeze(),
        fixed_time_owner_discovery_receipt_id=(
            fixed_time_owner_discovery_receipt_id
        ),
        accounting=accounting,
    )


@torch.no_grad()
def kinetic_p0_compiler_node_vjp(
    sites: AffineKineticPowerSites,
    ray_coefficients: torch.Tensor,
    node_times: torch.Tensor | Sequence[float],
    words: Sequence[FrozenKineticOwnerWord],
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    grad_node_transfer: torch.Tensor,
    *,
    near: float,
    far: float,
    continuous_topology_certificate_id: str,
    thresholds: StableStratumThresholds = StableStratumThresholds(),
) -> KineticP0CompilerNodeVJP:
    """Accumulate a sparse P0 VJP at fixed compiler nodes.

    ``grad_node_transfer[p,j]`` follows ``[beta,m_r,m_g,m_b]``.  The caller
    must have reduced all requested-sample residuals to these node cotangents
    already.  Consequently no requested-frame count or frame-by-run state is
    accepted by this API.

    The certificate id is provenance, not proof performed by this function.
    This function independently checks the supplied word at every node, but a
    continuous certificate must exclude missed events between nodes.
    """

    _require_provenance_id(
        continuous_topology_certificate_id,
        name="continuous_topology_certificate_id",
    )
    if not isinstance(sites, AffineKineticPowerSites):
        raise TypeError("sites must be AffineKineticPowerSites")
    if not isinstance(thresholds, StableStratumThresholds):
        raise TypeError("thresholds must be StableStratumThresholds")
    if not math.isfinite(near) or not math.isfinite(far) or far <= near:
        raise ValueError("near/far must be finite with near < far")

    rays = _finite_f64_cpu(ray_coefficients, name="ray_coefficients")
    times = _finite_f64_cpu(node_times, name="node_times").reshape(-1)
    density = _finite_f64_cpu(site_density, name="site_density").reshape(-1)
    color = _finite_f64_cpu(site_color, name="site_color")
    grad_transfer = _finite_f64_cpu(grad_node_transfer, name="grad_node_transfer")
    if rays.ndim != 2 or rays.shape[1] != 12 or rays.shape[0] < 1:
        raise ValueError("ray_coefficients must have shape [P,12] with P >= 1")
    if times.numel() < 1:
        raise ValueError("node_times must contain at least one compiler node")
    if density.shape != (sites.site_count,):
        raise ValueError("site_density must have shape [S]")
    if color.shape != (sites.site_count, 3):
        raise ValueError("site_color must have shape [S,3]")
    if bool((density < 0.0).any().item()):
        raise ValueError("P0 site density must be nonnegative")
    if len(words) != int(rays.shape[0]) or any(not isinstance(word, FrozenKineticOwnerWord) for word in words):
        raise ValueError("words must contain one FrozenKineticOwnerWord per track")
    for track_id, word in enumerate(words):
        if int(word.owners.min().item()) < 0 or int(word.owners.max().item()) >= sites.site_count:
            raise ValueError(f"track {track_id} owner id leaves the kinetic site table")
    expected_grad_shape = (int(rays.shape[0]), int(times.numel()), 4)
    if tuple(grad_transfer.shape) != expected_grad_shape:
        raise ValueError(f"grad_node_transfer must have shape {expected_grad_shape}")

    track_count = int(rays.shape[0])
    node_count = int(times.numel())
    node_transfers = torch.empty((track_count, node_count, 4), dtype=DTYPE)
    geometry_gradients = _zero_kinetic_geometry_gradients(sites, rays)
    grad_density = torch.zeros_like(density)
    grad_color = torch.zeros_like(color)
    margins = _MarginAccumulator()

    for node_id, time in enumerate(times):
        position, weight, time_powers = _kinetic_site_state_at_time(sites, time)
        for track_id, word in enumerate(words):
            origin = rays[track_id, :3] + time * rays[track_id, 3:6]
            direction = rays[track_id, 6:9] + time * rays[track_id, 9:12]
            geometry = _prepare_node_word_geometry(
                position,
                weight,
                origin,
                direction,
                word,
                near=near,
                far=far,
                thresholds=thresholds,
                margins=margins,
                track_id=track_id,
                node_id=node_id,
            )
            transfer, segment_beta, segment_alpha = _forward_word_transfer(
                word,
                geometry.physical_lengths,
                density,
                color,
            )
            node_transfers[track_id, node_id] = transfer

            total_beta = transfer[0]
            total_m = transfer[1:]
            beta_bar = grad_transfer[track_id, node_id, 0]
            m_bar = grad_transfer[track_id, node_id, 1:]
            prefix_beta = torch.ones((), dtype=DTYPE)
            prefix_m = torch.zeros(3, dtype=DTYPE)
            grad_physical_lengths = torch.empty(word.run_count, dtype=DTYPE)

            for run_id, owner_raw in enumerate(word.owners.tolist()):
                owner = int(owner_raw)
                beta = segment_beta[run_id]
                alpha = segment_alpha[run_id]
                tau_bar = (
                    torch.dot(
                        m_bar,
                        prefix_m + prefix_beta * color[owner] - total_m,
                    )
                    - total_beta * beta_bar
                )
                grad_physical_lengths[run_id] = density[owner] * tau_bar
                grad_density[owner] += geometry.physical_lengths[run_id] * tau_bar
                grad_color[owner] += prefix_beta * alpha * m_bar
                prefix_m = prefix_m + prefix_beta * alpha * color[owner]
                prefix_beta = prefix_beta * beta

            _accumulate_node_geometry_bars_(
                geometry_gradients,
                _node_word_physical_length_vjp(
                    geometry,
                    grad_physical_lengths,
                ),
                time=time,
                time_powers=time_powers,
                track_id=track_id,
            )

    run_interactions = node_count * sum(word.run_count for word in words)
    cut_interactions = node_count * sum(word.run_count - 1 for word in words)
    owner_margin_evaluations = node_count * sum(3 * word.run_count * (sites.site_count - 1) for word in words)
    accounting: dict[str, int | str | bool] = {
        "track_count": track_count,
        "compiler_node_count": node_count,
        "node_transfer_count": track_count * node_count,
        "active_run_node_interactions": run_interactions,
        "active_cut_node_interactions": cut_interactions,
        "owner_margin_evaluations": owner_margin_evaluations,
        "requested_frame_count_used": 0,
        "frame_by_run_reverse_state_allocated": False,
        "reverse_interaction_scaling": "O(J * sum_p R_p)",
        "validation_interaction_scaling": "O(J * S * sum_p R_p)",
        "continuous_topology_checked_between_nodes": False,
    }
    return KineticP0CompilerNodeVJP(
        node_transfers=node_transfers,
        grad_positions0=geometry_gradients.positions0,
        grad_velocities=geometry_gradients.velocities,
        grad_weight_coefficients=geometry_gradients.weight_coefficients,
        grad_ray_coefficients=geometry_gradients.ray_coefficients,
        grad_site_density=grad_density,
        grad_site_color=grad_color,
        margins=margins.freeze(),
        continuous_topology_certificate_id=continuous_topology_certificate_id,
        derivative_scope=DERIVATIVE_SCOPE,
        event_time_derivatives_included=False,
        chart_endpoint_derivatives_included=False,
        node_time_or_rank_derivatives_included=False,
        accounting=accounting,
    )


def _require_provenance_id(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"a nonempty {name} is required")


def _zero_kinetic_geometry_gradients(
    sites: AffineKineticPowerSites,
    rays: torch.Tensor,
) -> _KineticGeometryGradients:
    return _KineticGeometryGradients(
        positions0=torch.zeros_like(sites.positions0),
        velocities=torch.zeros_like(sites.velocities),
        weight_coefficients=torch.zeros_like(sites.weight_coefficients),
        ray_coefficients=torch.zeros_like(rays),
    )


def _kinetic_site_state_at_time(
    sites: AffineKineticPowerSites,
    time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    time_powers = torch.stack((torch.ones_like(time), time, time.square()))[: int(sites.weight_coefficients.shape[1])]
    return (
        sites.positions0 + time * sites.velocities,
        sites.weight_coefficients @ time_powers,
        time_powers,
    )


def _node_word_physical_length_vjp(
    geometry: _NodeWordGeometry,
    grad_physical_lengths: torch.Tensor,
) -> _NodeWordGeometryBars:
    """Reverse one validated node's physical run lengths into local bars."""

    grad_lengths = torch.as_tensor(
        grad_physical_lengths,
        dtype=DTYPE,
        device="cpu",
    ).reshape(-1)
    if tuple(grad_lengths.shape) != tuple(geometry.physical_lengths.shape) or not bool(
        torch.isfinite(grad_lengths).all().item()
    ):
        raise ValueError("physical-length cotangents must be finite and match the validated owner word")

    coordinate_length_bars = geometry.speed * grad_lengths
    cut_bars = coordinate_length_bars[:-1] - coordinate_length_bars[1:]
    speed_bar = torch.dot(geometry.coordinate_lengths, grad_lengths)
    positions_bar = torch.zeros_like(geometry.positions)
    weights_bar = torch.zeros_like(geometry.weights)
    origin_bar = torch.zeros_like(geometry.origin)
    direction_bar = speed_bar * geometry.direction / geometry.speed
    for cut, cut_bar in zip(geometry.active_cuts, cut_bars, strict=True):
        left = cut.left_owner
        right = cut.right_owner
        point = geometry.origin + cut.depth * geometry.direction
        normal = 2.0 * (geometry.positions[right] - geometry.positions[left])
        implicit_bar = -cut_bar / cut.denominator
        positions_bar[left] += implicit_bar * 2.0 * (geometry.positions[left] - point)
        positions_bar[right] += implicit_bar * 2.0 * (point - geometry.positions[right])
        weights_bar[left] -= implicit_bar
        weights_bar[right] += implicit_bar
        origin_bar += implicit_bar * normal
        direction_bar += implicit_bar * cut.depth * normal
    return _NodeWordGeometryBars(
        positions=positions_bar,
        weights=weights_bar,
        origin=origin_bar,
        direction=direction_bar,
    )


def _accumulate_node_geometry_bars_(
    gradients: _KineticGeometryGradients,
    bars: _NodeWordGeometryBars,
    *,
    time: torch.Tensor,
    time_powers: torch.Tensor,
    track_id: int,
) -> None:
    gradients.positions0.add_(bars.positions)
    gradients.velocities.add_(time * bars.positions)
    gradients.weight_coefficients.add_(bars.weights[:, None] * time_powers[None, :])
    gradients.ray_coefficients[track_id, :3].add_(bars.origin)
    gradients.ray_coefficients[track_id, 3:6].add_(time * bars.origin)
    gradients.ray_coefficients[track_id, 6:9].add_(bars.direction)
    gradients.ray_coefficients[track_id, 9:12].add_(time * bars.direction)


def _prepare_node_word_geometry(
    positions: torch.Tensor,
    weights: torch.Tensor,
    origin: torch.Tensor,
    direction: torch.Tensor,
    word: FrozenKineticOwnerWord,
    *,
    near: float,
    far: float,
    thresholds: StableStratumThresholds,
    margins: _MarginAccumulator,
    track_id: int,
    node_id: int,
) -> _NodeWordGeometry:
    speed = torch.linalg.vector_norm(direction)
    speed_value = float(speed.item())
    margins.minimum_ray_speed = min(margins.minimum_ray_speed, speed_value)
    if speed_value <= thresholds.minimum_ray_speed:
        _fail(track_id, node_id, "ray-speed margin failed")

    active_cuts: list[_ActiveCut] = []
    cut_depths = [torch.as_tensor(near, dtype=DTYPE)]
    owners = tuple(int(value) for value in word.owners.tolist())
    for cut_id, (left, right) in enumerate(zip(owners[:-1], owners[1:], strict=True)):
        separation = positions[right] - positions[left]
        normal = 2.0 * separation
        denominator = torch.dot(normal, direction)
        denominator_value = abs(float(denominator.item()))
        denominator_scale = float(torch.linalg.vector_norm(normal).item()) * speed_value
        cosine = denominator_value / denominator_scale if denominator_scale > 0.0 else 0.0
        margins.minimum_absolute_cut_denominator = min(
            margins.minimum_absolute_cut_denominator,
            denominator_value,
        )
        margins.minimum_cut_cosine = min(margins.minimum_cut_cosine, cosine)
        if denominator_value <= thresholds.minimum_absolute_cut_denominator:
            _fail(track_id, node_id, f"cut {cut_id} absolute denominator margin failed")
        if cosine <= thresholds.minimum_cut_cosine:
            _fail(track_id, node_id, f"cut {cut_id} cosine denominator margin failed")
        intercept = (
            torch.dot(normal, origin)
            + torch.dot(positions[left], positions[left])
            - torch.dot(positions[right], positions[right])
            - weights[left]
            + weights[right]
        )
        depth = -intercept / denominator
        if not bool(torch.isfinite(depth).item()):
            _fail(track_id, node_id, f"cut {cut_id} depth is nonfinite")
        active_cuts.append(
            _ActiveCut(
                left_owner=left,
                right_owner=right,
                depth=depth,
                denominator=denominator,
            )
        )
        cut_depths.append(depth)
    cut_depths.append(torch.as_tensor(far, dtype=DTYPE))
    cuts = torch.stack(cut_depths)
    coordinate_lengths = cuts[1:] - cuts[:-1]
    physical_lengths = speed * coordinate_lengths
    for run_id, (coordinate_length, physical_length) in enumerate(
        zip(coordinate_lengths.tolist(), physical_lengths.tolist(), strict=True)
    ):
        margins.minimum_coordinate_length = min(
            margins.minimum_coordinate_length,
            float(coordinate_length),
        )
        margins.minimum_physical_length = min(
            margins.minimum_physical_length,
            float(physical_length),
        )
        if coordinate_length <= thresholds.minimum_coordinate_length:
            _fail(track_id, node_id, f"run {run_id} coordinate-length margin failed")
        if physical_length <= thresholds.minimum_physical_length:
            _fail(track_id, node_id, f"run {run_id} physical-length margin failed")

    _check_owner_margins(
        positions,
        weights,
        origin,
        direction,
        owners,
        cuts,
        thresholds=thresholds,
        margins=margins,
        track_id=track_id,
        node_id=node_id,
    )
    return _NodeWordGeometry(
        positions=positions,
        weights=weights,
        origin=origin,
        direction=direction,
        speed=speed,
        cuts=cuts,
        active_cuts=tuple(active_cuts),
        coordinate_lengths=coordinate_lengths,
        physical_lengths=physical_lengths,
    )


def _check_owner_margins(
    positions: torch.Tensor,
    weights: torch.Tensor,
    origin: torch.Tensor,
    direction: torch.Tensor,
    owners: tuple[int, ...],
    cuts: torch.Tensor,
    *,
    thresholds: StableStratumThresholds,
    margins: _MarginAccumulator,
    track_id: int,
    node_id: int,
) -> None:
    for run_id, owner in enumerate(owners):
        midpoint = 0.5 * (cuts[run_id] + cuts[run_id + 1])
        for label, depth, allowed_tie in (
            ("left", cuts[run_id], owners[run_id - 1] if run_id > 0 else None),
            (
                "right",
                cuts[run_id + 1],
                owners[run_id + 1] if run_id + 1 < len(owners) else None,
            ),
            ("midpoint", midpoint, None),
        ):
            point = origin + depth * direction
            powers = (point[None, :] - positions).square().sum(dim=1) - weights
            owner_power = powers[owner]
            for competitor in range(int(positions.shape[0])):
                if competitor == owner:
                    continue
                gap = float((powers[competitor] - owner_power).item())
                if competitor == allowed_tie:
                    tie_scale = max(
                        1.0,
                        abs(float(powers[competitor].item())),
                        abs(float(owner_power.item())),
                    )
                    residual = abs(gap)
                    margins.maximum_active_tie_residual = max(
                        margins.maximum_active_tie_residual,
                        residual,
                    )
                    if residual > thresholds.active_tie_tolerance * tie_scale:
                        _fail(
                            track_id,
                            node_id,
                            f"run {run_id} {label} active-tie residual failed",
                        )
                    continue
                margins.minimum_owner_gap = min(margins.minimum_owner_gap, gap)
                if gap <= thresholds.minimum_owner_gap:
                    _fail(
                        track_id,
                        node_id,
                        f"run {run_id} {label} owner/topology margin failed against site {competitor}",
                    )


def _forward_word_transfer(
    word: FrozenKineticOwnerWord,
    physical_lengths: torch.Tensor,
    density: torch.Tensor,
    color: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    beta_total = torch.ones((), dtype=DTYPE)
    m_total = torch.zeros(3, dtype=DTYPE)
    segment_beta = []
    segment_alpha = []
    for run_id, owner_raw in enumerate(word.owners.tolist()):
        owner = int(owner_raw)
        optical_depth = density[owner] * physical_lengths[run_id]
        beta = torch.exp(-optical_depth)
        alpha = -torch.expm1(-optical_depth)
        segment_beta.append(beta)
        segment_alpha.append(alpha)
        m_total = m_total + beta_total * alpha * color[owner]
        beta_total = beta_total * beta
    return (
        torch.cat((beta_total.reshape(1), m_total)),
        torch.stack(segment_beta),
        torch.stack(segment_alpha),
    )


def _finite_f64_cpu(value: Any, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=DTYPE, device="cpu").detach().clone()
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor.contiguous()


def _fail(track_id: int, node_id: int, reason: str) -> None:
    raise StableStratumError(f"track {track_id}, compiler node {node_id}: {reason}")


__all__ = [
    "DERIVATIVE_SCOPE",
    "FIXED_TIME_DERIVATIVE_SCOPE",
    "FrozenKineticOwnerWord",
    "KineticP0CompilerNodeVJP",
    "KineticP0FixedTimePhysicalLengthGeometryVJP",
    "KineticP0NodePhysicalLengthGeometryVJP",
    "ObservedStableStratumMargins",
    "StableStratumThresholds",
    "StableStratumError",
    "kinetic_p0_compiler_node_vjp",
    "kinetic_p0_fixed_time_physical_length_geometry_vjp",
    "kinetic_p0_node_physical_length_geometry_vjp",
    "make_frozen_kinetic_owner_word",
]
