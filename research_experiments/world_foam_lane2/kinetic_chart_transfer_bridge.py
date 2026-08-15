"""CPU bridge from exact kinetic owner charts to compact P0 transfer.

The kinetic owner compiler proves which site owns each positive-length depth
run on an open time chart.  The existing compact Lie schedule knows how to
interpolate a fixed number of total-transfer nodes and reduce arbitrarily many
sample cotangents back to those nodes.  This module joins those two contracts
without pretending that the legacy Mobius boundary ABI can represent a
general kinetic face.

Compilation performs exact rational owner/cut discovery at the schedule's
``J`` time nodes.  Physical lengths and Beer--Lambert transfer are then
evaluated in CPU float64 (the norm and exponential are not rational
operations).  Requested sample count never enters the geometry or transfer
program.  A blocked material-only VJP uses the existing verified barycentric
sample-to-node schedule and the existing affine-Lie encode/decode VJPs before
one prefix-only reverse sweep over the ``J`` ordered words.

This is deliberately not a geometry adjoint.  In particular it emits no
gradient for kinetic positions, velocities, weights, ray coefficients, or
algebraic event times.  Non-rational event boundaries are replaced by a
certified interior float interval; the tiny isolator neighborhoods are not
claimed as covered.  Multi-chart CPU seam dispatch lives in
:mod:`kinetic_multichart_transfer_program`; native lowering remains open.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction

import torch
from compact_lie_schedule import (
    CompactLieChartSchedule,
    CompactLieChartSpec,
    compact_lie_world_schedule_from_specs,
)
from kinetic_active_owner_chart_compiler import (
    ActiveKineticEventGuard,
    ActiveKineticOwnerChartProgram,
    ActiveKineticTimeBoundary,
    compile_active_kinetic_owner_charts,
)
from kinetic_owner_chart_compiler import (
    KineticAlgebraicEventGuard,
    KineticOwnerChartProgram,
    KineticTimeBoundary,
    compile_exact_kinetic_owner_charts,
)
from kinetic_power_word_compiler import (
    AffineKineticPowerSites,
    discover_kinetic_power_word_at_time,
)
from transfer_lie_chart import (
    check_lie_chart_cone,
    check_transfer_cone,
    transfer_lie_decode,
    transfer_lie_decode_vjp,
    transfer_lie_encode,
    transfer_lie_encode_vjp,
)

DTYPE = torch.float64
KineticOwnerProgramLike = KineticOwnerChartProgram | ActiveKineticOwnerChartProgram
KineticEventGuardLike = KineticAlgebraicEventGuard | ActiveKineticEventGuard
KineticTimeBoundaryLike = KineticTimeBoundary | ActiveKineticTimeBoundary


@dataclass(frozen=True)
class BoundKineticOwnerProgram:
    """Immutable-by-digest owner program and its exact binary64 source."""

    program: KineticOwnerProgramLike
    sites: AffineKineticPowerSites
    ray_coefficients: torch.Tensor
    source_content_digest: str
    program_semantic_digest: str
    compiler_provenance: str

    def assert_current(self) -> None:
        if (
            not self.program.passed
            or not self.program.continuous_time_coverage
            or not self.program.owner_identity_certified
            or self.program.unresolved_degeneracies
        ):
            raise ValueError("bound kinetic owner program is no longer a passed exact program")
        expected_provenance = (
            "active_kinetic_owner_chart_compiler_v1"
            if isinstance(self.program, ActiveKineticOwnerChartProgram)
            else "exhaustive_kinetic_owner_chart_oracle_v1"
        )
        if self.compiler_provenance != expected_provenance:
            raise ValueError("kinetic owner compiler provenance mismatch")
        if (
            _source_content_digest(
                self.sites,
                self.ray_coefficients,
                t_min=self.program.t_min,
                t_max=self.program.t_max,
                near=self.program.near,
                far=self.program.far,
            )
            != self.source_content_digest
        ):
            raise ValueError("kinetic source content digest mismatch")
        if _program_semantic_digest(self.program) != self.program_semantic_digest:
            raise ValueError("kinetic owner-program semantic digest mismatch")


@dataclass(frozen=True)
class KineticChartP0Geometry:
    """Frame-density-independent geometry sampled only at compile nodes."""

    chart_id: int
    owner_word: tuple[int, ...]
    owners: torch.Tensor
    schedule: CompactLieChartSchedule
    node_physical_lengths: torch.Tensor
    exact_node_times: tuple[Fraction, ...]
    exact_node_transition_depths: tuple[tuple[Fraction, ...], ...]
    binding_digest: str
    left_boundary_uncertainty: Fraction
    right_boundary_uncertainty: Fraction
    right_closed: bool
    site_count: int
    exact_owner_and_cut_discovery_at_nodes: bool = True
    float64_metric_and_transfer_evaluation: bool = True
    safe_interval_is_certified_inside_owner_chart: bool = True
    full_algebraic_boundary_coverage: bool = False
    seam_dispatch_implemented: bool = False
    geometry_vjp_implemented: bool = False
    event_time_vjp_implemented: bool = False
    requested_frame_sampling_used: bool = False

    @property
    def node_count(self) -> int:
        return self.schedule.node_count

    @property
    def run_count(self) -> int:
        return len(self.owner_word)

    @property
    def structural_tensor_bytes(self) -> int:
        return self.schedule.resident_bytes + _tensor_bytes((self.owners, self.node_physical_lengths))

    @property
    def exact_certificate_scalar_count(self) -> int:
        return len(self.exact_node_times) + sum(len(depths) for depths in self.exact_node_transition_depths)


@dataclass(frozen=True)
class KineticChartP0Transfer:
    """One material snapshot over a fixed kinetic chart geometry."""

    geometry: KineticChartP0Geometry
    site_density: torch.Tensor
    site_color: torch.Tensor
    node_transfers: torch.Tensor
    node_transfer_cone_passed: bool
    node_lie_cone_passed: bool
    requested_frame_sampling_used: bool = False

    @property
    def resident_tensor_bytes(self) -> int:
        return self.geometry.structural_tensor_bytes + _tensor_bytes(
            (self.site_density, self.site_color, self.node_transfers)
        )


@dataclass(frozen=True)
class KineticChartMaterialVJP:
    """Blocked sample loss and material gradients for one kinetic chart."""

    loss: torch.Tensor
    predictions: torch.Tensor | None
    grad_site_density: torch.Tensor
    grad_site_color: torch.Tensor
    accounting: dict[str, int | bool]
    geometry_gradients: None = None
    event_time_gradients: None = None
    geometry_vjp_implemented: bool = False
    event_time_vjp_implemented: bool = False


def compile_kinetic_chart_p0_geometry(
    program: KineticOwnerProgramLike,
    sites: AffineKineticPowerSites,
    ray_coefficients: torch.Tensor,
    *,
    chart_id: int,
    node_count: int,
) -> KineticChartP0Geometry:
    """Bind one passed owner chart to exact compile-node run geometry.

    The current owner-chart result does not carry a source tensor digest.  To
    avoid trusting an unbound ``program`` with different sites or ray data,
    this proof bridge recompiles the exact chart arrangement once and compares
    its semantic seam signature.  A production ABI should carry an immutable
    compiler-input digest and remove this duplicate ``O(S^3)`` proof work.
    """

    binding = bind_kinetic_owner_program(program, sites, ray_coefficients)
    return compile_bound_kinetic_chart_p0_geometry(
        binding,
        chart_id=chart_id,
        node_count=node_count,
    )


def bind_kinetic_owner_program(
    program: KineticOwnerProgramLike,
    sites: AffineKineticPowerSites,
    ray_coefficients: torch.Tensor,
) -> BoundKineticOwnerProgram:
    """Recompile once, then seal exact source and program provenance."""

    _require_bindable_owner_program(program)
    ray = _finite_cpu_f64_vector(ray_coefficients, name="ray_coefficients", size=12)
    sites_copy = AffineKineticPowerSites(
        positions0=sites.positions0.detach().clone(),
        velocities=sites.velocities.detach().clone(),
        weight_coefficients=sites.weight_coefficients.detach().clone(),
    )
    if isinstance(program, ActiveKineticOwnerChartProgram):
        rebound: KineticOwnerProgramLike = compile_active_kinetic_owner_charts(
            sites_copy,
            ray,
            t_min=program.t_min,
            t_max=program.t_max,
            near=program.near,
            far=program.far,
        )
        compiler_provenance = "active_kinetic_owner_chart_compiler_v1"
    elif isinstance(program, KineticOwnerChartProgram):
        rebound = compile_exact_kinetic_owner_charts(
            sites_copy,
            ray,
            t_min=program.t_min,
            t_max=program.t_max,
            near=program.near,
            far=program.far,
        )
        compiler_provenance = "exhaustive_kinetic_owner_chart_oracle_v1"
    else:
        raise TypeError("unsupported kinetic owner-program type")
    if not rebound.passed:
        raise ValueError("sites/ray do not reproduce a passed kinetic owner program")
    _require_same_program_semantics(program, rebound)
    return _seal_bound_kinetic_owner_program(
        rebound,
        sites_copy,
        ray,
        compiler_provenance=compiler_provenance,
    )


def compile_and_bind_active_kinetic_owner_program(
    sites: AffineKineticPowerSites,
    ray_coefficients: torch.Tensor,
    *,
    t_min: Fraction | float | int,
    t_max: Fraction | float | int,
    near: Fraction | float | int,
    far: Fraction | float | int,
) -> BoundKineticOwnerProgram:
    """Compile active closure once and bind the exact inputs used for it.

    This is the production construction path.  Unlike
    :func:`bind_kinetic_owner_program`, it does not accept a detached program
    from an arbitrary caller and therefore needs no second proof compilation
    to establish source identity.
    """

    ray = _finite_cpu_f64_vector(
        ray_coefficients,
        name="ray_coefficients",
        size=12,
    )
    sites_copy = AffineKineticPowerSites(
        positions0=sites.positions0.detach().clone(),
        velocities=sites.velocities.detach().clone(),
        weight_coefficients=sites.weight_coefficients.detach().clone(),
    )
    program = compile_active_kinetic_owner_charts(
        sites_copy,
        ray,
        t_min=t_min,
        t_max=t_max,
        near=near,
        far=far,
    )
    _require_bindable_owner_program(program)
    return _seal_bound_kinetic_owner_program(
        program,
        sites_copy,
        ray,
        compiler_provenance="active_kinetic_owner_chart_compiler_v1",
    )


def _require_bindable_owner_program(program: KineticOwnerProgramLike) -> None:
    if (
        not program.passed
        or not program.continuous_time_coverage
        or not program.owner_identity_certified
    ):
        raise ValueError(
            "kinetic transfer requires a passed, continuously certified owner program"
        )
    if program.unresolved_degeneracies:
        reasons = tuple(value.kind for value in program.unresolved_degeneracies)
        raise ValueError(
            f"kinetic transfer cannot bind unresolved degeneracies: {reasons!r}"
        )


def _seal_bound_kinetic_owner_program(
    program: KineticOwnerProgramLike,
    sites: AffineKineticPowerSites,
    ray: torch.Tensor,
    *,
    compiler_provenance: str,
) -> BoundKineticOwnerProgram:
    binding = BoundKineticOwnerProgram(
        program=program,
        sites=sites,
        ray_coefficients=ray,
        source_content_digest=_source_content_digest(
            sites,
            ray,
            t_min=program.t_min,
            t_max=program.t_max,
            near=program.near,
            far=program.far,
        ),
        program_semantic_digest=_program_semantic_digest(program),
        compiler_provenance=compiler_provenance,
    )
    binding.assert_current()
    return binding


def compile_bound_kinetic_chart_p0_geometry(
    binding: BoundKineticOwnerProgram,
    *,
    chart_id: int,
    node_count: int,
) -> KineticChartP0Geometry:
    """Compile one chart without repeating exact source/program binding."""

    binding.assert_current()
    program = binding.program
    sites = binding.sites
    ray = binding.ray_coefficients
    if isinstance(chart_id, bool) or not isinstance(chart_id, int):
        raise TypeError("chart_id must be an integer")
    if not 0 <= chart_id < len(program.charts):
        raise ValueError("chart_id is outside the certified owner program")
    if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count < 2:
        raise ValueError("node_count must be an integer at least two")
    chart = program.charts[chart_id]

    left_exact, left_uncertainty = _safe_boundary(chart.left_boundary, side="left")
    right_exact, right_uncertainty = _safe_boundary(chart.right_boundary, side="right")
    safe_t_min = _fraction_to_safe_float(left_exact, lower=True)
    safe_t_max = _fraction_to_safe_float(right_exact, lower=False)
    if not math.isfinite(safe_t_min) or not math.isfinite(safe_t_max) or safe_t_max <= safe_t_min:
        raise ValueError("algebraic guard isolation leaves no representable float64 chart interior")
    left_uncertainty += Fraction.from_float(safe_t_min) - left_exact
    right_uncertainty += right_exact - Fraction.from_float(safe_t_max)

    compact = compact_lie_world_schedule_from_specs(
        (
            CompactLieChartSpec(
                t_min=safe_t_min,
                t_max=safe_t_max,
                near=float(program.near),
                far=float(program.far),
                node_count=node_count,
                chart="lie",
            ),
        ),
        global_track_count=1,
        selection_provenance="exact_kinetic_owner_chart_cpu_bridge_v1",
    )
    schedule = compact.charts[0]
    owners = torch.tensor(chart.owner_word, dtype=torch.int64)
    node_times_exact: list[Fraction] = []
    node_depths: list[tuple[Fraction, ...]] = []
    physical_lengths: list[torch.Tensor] = []
    for node_time in schedule.node_times:
        time_float = float(node_time.item())
        time_exact = Fraction.from_float(time_float)
        fixed_word = discover_kinetic_power_word_at_time(
            sites,
            ray,
            time=time_exact,
            near=program.near,
            far=program.far,
        )
        fixed_owners = tuple(int(owner) for owner in fixed_word.word.owners.tolist())
        if fixed_owners != chart.owner_word:
            raise ValueError("compile node does not reproduce the certified chart owner word")
        cuts = (program.near, *fixed_word.transition_depths, program.far)
        coordinate_lengths = tuple(right - left for left, right in zip(cuts, cuts[1:], strict=False))
        if len(coordinate_lengths) != len(chart.owner_word) or any(length <= 0 for length in coordinate_lengths):
            raise ArithmeticError("exact kinetic node word contains a nonpositive run")
        direction = ray[6:9] + node_time * ray[9:12]
        fiber_speed = torch.linalg.vector_norm(direction)
        if not bool(torch.isfinite(fiber_speed).item()) or float(fiber_speed.item()) <= 0.0:
            raise ValueError("kinetic compile node has zero or nonfinite fiber speed")
        node_times_exact.append(time_exact)
        node_depths.append(fixed_word.transition_depths)
        physical_lengths.append(
            fiber_speed * torch.tensor([float(length) for length in coordinate_lengths], dtype=DTYPE)
        )

    node_physical_lengths = torch.stack(physical_lengths, dim=0).contiguous()
    if not bool(torch.isfinite(node_physical_lengths).all().item()) or bool(
        torch.any(node_physical_lengths <= 0.0).item()
    ):
        raise ValueError("kinetic compile-node physical lengths must be finite and positive")
    return KineticChartP0Geometry(
        chart_id=chart.chart_id,
        owner_word=chart.owner_word,
        owners=owners,
        schedule=schedule,
        node_physical_lengths=node_physical_lengths,
        exact_node_times=tuple(node_times_exact),
        exact_node_transition_depths=tuple(node_depths),
        binding_digest=binding.source_content_digest,
        left_boundary_uncertainty=left_uncertainty,
        right_boundary_uncertainty=right_uncertainty,
        right_closed=chart.right_closed,
        site_count=sites.site_count,
        full_algebraic_boundary_coverage=left_uncertainty == 0 and right_uncertainty == 0,
    )


def refresh_kinetic_chart_p0_transfer(
    geometry: KineticChartP0Geometry,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
) -> KineticChartP0Transfer:
    """Refresh exact-node P0 transfer after a material parameter update."""

    density = _finite_cpu_f64_vector(
        site_density,
        name="site_density",
        size=geometry.site_count,
    )
    color = torch.as_tensor(site_color, dtype=DTYPE, device="cpu").detach().clone().contiguous()
    if tuple(color.shape) != (geometry.site_count, 3) or not bool(torch.isfinite(color).all().item()):
        raise ValueError(f"site_color must be finite CPU-compatible data with shape [{geometry.site_count},3]")
    if bool(torch.any(density < 0.0).item()):
        raise ValueError("site_density must be nonnegative")
    if bool(torch.any(color < 0.0).item()) or bool(torch.any(color > 1.0).item()):
        raise ValueError("site_color must stay in [0,1] for the certified transfer cone")

    node_transfers = torch.stack(
        [
            ordered_p0_transfer(
                geometry.owners,
                lengths,
                density,
                color,
            )
            for lengths in geometry.node_physical_lengths
        ],
        dim=0,
    ).contiguous()
    transfer_cone = check_transfer_cone(node_transfers)
    if not transfer_cone.passed:
        raise ValueError("exact-node kinetic P0 transfer left the physical cone")
    node_lie = transfer_lie_encode(node_transfers)
    lie_cone = check_lie_chart_cone(node_lie)
    if not lie_cone.passed:
        raise ValueError("exact-node kinetic P0 transfer left the affine-Lie cone")
    return KineticChartP0Transfer(
        geometry=geometry,
        site_density=density,
        site_color=color,
        node_transfers=node_transfers,
        node_transfer_cone_passed=True,
        node_lie_cone_passed=True,
    )


def evaluate_kinetic_chart_p0_transfer(
    transfer: KineticChartP0Transfer,
    times: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the compact approximate temporal closure as ``[beta,m]``."""

    times_f64 = _validate_sample_times(transfer.geometry, times)
    weights = transfer.geometry.schedule.sample_to_node_weights(times_f64).weights
    sample_chart = weights @ transfer_lie_encode(transfer.node_transfers)
    cone = check_lie_chart_cone(sample_chart)
    if not cone.passed:
        raise ValueError("interpolated kinetic affine-Lie chart left the physical cone")
    return transfer_lie_decode(sample_chart)


def kinetic_chart_material_mse_vjp(
    transfer: KineticChartP0Transfer,
    times: torch.Tensor,
    targets: torch.Tensor,
    *,
    background: torch.Tensor,
    frame_block_size: int = 16,
    return_predictions: bool = False,
) -> KineticChartMaterialVJP:
    """Use bounded sample blocks, then one compact node/material reverse."""

    if isinstance(frame_block_size, bool) or not isinstance(frame_block_size, int) or frame_block_size < 1:
        raise ValueError("frame_block_size must be a positive integer")
    times_f64 = _validate_sample_times(transfer.geometry, times)
    target = torch.as_tensor(targets, dtype=DTYPE, device="cpu").detach()
    if tuple(target.shape) != (int(times_f64.numel()), 3) or not bool(torch.isfinite(target).all().item()):
        raise ValueError("targets must be finite and have shape [F,3]")
    background_f64 = torch.as_tensor(background, dtype=DTYPE, device="cpu").detach().reshape(-1)
    if tuple(background_f64.shape) != (3,) or not bool(torch.isfinite(background_f64).all().item()):
        raise ValueError("background must be a finite RGB vector")

    node_chart = transfer_lie_encode(transfer.node_transfers)
    grad_node_chart = torch.zeros_like(node_chart)
    predictions = torch.empty_like(target) if return_predictions else None
    loss = torch.zeros((), dtype=DTYPE)
    inv_element_count = 1.0 / float(target.numel())
    peak_block = min(frame_block_size, int(times_f64.numel()))
    linear_interactions = 0
    dense_fallback_interactions = 0
    for start in range(0, int(times_f64.numel()), frame_block_size):
        stop = min(start + frame_block_size, int(times_f64.numel()))
        weight_result = transfer.geometry.schedule.sample_to_node_weights(times_f64[start:stop])
        linear_interactions += weight_result.linear_weight_interactions
        dense_fallback_interactions += weight_result.dense_fallback_interactions
        sample_chart = weight_result.weights @ node_chart
        cone = check_lie_chart_cone(sample_chart)
        if not cone.passed:
            raise ValueError("interpolated kinetic affine-Lie chart left the physical cone")
        sample_transfer = transfer_lie_decode(sample_chart)
        prediction = sample_transfer[:, 1:] + sample_transfer[:, :1] * background_f64
        if predictions is not None:
            predictions[start:stop] = prediction
        residual = prediction - target[start:stop]
        loss += residual.square().sum() * inv_element_count
        grad_prediction = 2.0 * residual * inv_element_count
        grad_sample_transfer = torch.cat(
            (
                (grad_prediction * background_f64).sum(dim=1, keepdim=True),
                grad_prediction,
            ),
            dim=1,
        )
        grad_sample_chart = transfer_lie_decode_vjp(sample_chart, grad_sample_transfer)
        grad_node_chart += weight_result.weights.T @ grad_sample_chart

    grad_node_transfer = transfer_lie_encode_vjp(
        transfer.node_transfers,
        grad_node_chart,
    )
    grad_density, grad_color = kinetic_chart_p0_node_material_vjp(
        transfer,
        grad_node_transfer,
    )

    rank = transfer.geometry.node_count
    accounting: dict[str, int | bool] = {
        "requested_sample_count": int(times_f64.numel()),
        "compile_node_count": rank,
        "ordered_run_count": transfer.geometry.run_count,
        "world_node_replay_count": rank,
        "sample_to_node_linear_interactions": linear_interactions,
        "sample_to_node_dense_fallback_interactions": dense_fallback_interactions,
        "structural_tensor_bytes": transfer.resident_tensor_bytes,
        "reverse_node_tensor_bytes": _tensor_bytes((grad_node_chart, grad_node_transfer)),
        "peak_sample_block_bytes": peak_block * (rank + 4 + 4 + 3 + 3) * 8,
        "frame_dependent_structural_bytes": 0,
        "requested_frame_sampling_used_for_compile": False,
        "geometry_gradients_emitted": False,
        "event_time_gradients_emitted": False,
    }
    return KineticChartMaterialVJP(
        loss=loss,
        predictions=predictions,
        grad_site_density=grad_density,
        grad_site_color=grad_color,
        accounting=accounting,
    )


def kinetic_chart_p0_node_material_vjp(
    transfer: KineticChartP0Transfer,
    grad_node_transfer: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce accumulated node cotangents with one prefix reverse per node."""

    grad = torch.as_tensor(grad_node_transfer, dtype=DTYPE, device="cpu").detach()
    if tuple(grad.shape) != tuple(transfer.node_transfers.shape) or not bool(torch.isfinite(grad).all().item()):
        raise ValueError("grad_node_transfer must be finite and match node_transfers")
    grad_density = torch.zeros_like(transfer.site_density)
    grad_color = torch.zeros_like(transfer.site_color)
    for node_id in range(transfer.geometry.node_count):
        density_node_grad, color_node_grad = _ordered_p0_material_vjp(
            transfer.geometry.owners,
            transfer.geometry.node_physical_lengths[node_id],
            transfer.site_density,
            transfer.site_color,
            transfer.node_transfers[node_id],
            grad[node_id],
        )
        grad_density += density_node_grad
        grad_color += color_node_grad
    return grad_density, grad_color


def ordered_p0_transfer(
    owners: torch.Tensor,
    physical_lengths: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
) -> torch.Tensor:
    beta_total = torch.ones((), dtype=DTYPE)
    moment_total = torch.zeros(3, dtype=DTYPE)
    for owner_raw, length in zip(owners.tolist(), physical_lengths, strict=True):
        owner = int(owner_raw)
        optical_depth = site_density[owner] * length
        beta = torch.exp(-optical_depth)
        alpha = -torch.expm1(-optical_depth)
        moment_total = moment_total + beta_total * alpha * site_color[owner]
        beta_total = beta_total * beta
    return torch.cat((beta_total.reshape(1), moment_total))


def _ordered_p0_material_vjp(
    owners: torch.Tensor,
    physical_lengths: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    total_transfer: torch.Tensor,
    grad_transfer: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_density = torch.zeros_like(site_density)
    grad_color = torch.zeros_like(site_color)
    total_beta = total_transfer[0]
    total_moment = total_transfer[1:]
    grad_beta = grad_transfer[0]
    grad_moment = grad_transfer[1:]
    prefix_beta = torch.ones((), dtype=DTYPE)
    prefix_moment = torch.zeros(3, dtype=DTYPE)
    for owner_raw, length in zip(owners.tolist(), physical_lengths, strict=True):
        owner = int(owner_raw)
        optical_depth = site_density[owner] * length
        beta = torch.exp(-optical_depth)
        alpha = -torch.expm1(-optical_depth)
        optical_depth_bar = (
            torch.dot(
                grad_moment,
                prefix_moment + prefix_beta * site_color[owner] - total_moment,
            )
            - total_beta * grad_beta
        )
        grad_density[owner] += length * optical_depth_bar
        grad_color[owner] += prefix_beta * alpha * grad_moment
        prefix_moment = prefix_moment + prefix_beta * alpha * site_color[owner]
        prefix_beta = prefix_beta * beta
    return grad_density, grad_color


def _safe_boundary(
    boundary: KineticTimeBoundaryLike,
    *,
    side: str,
) -> tuple[Fraction, Fraction]:
    if boundary.kind in {"domain_start", "domain_end"}:
        if boundary.rational_value is None or boundary.event_guard is not None:
            raise ValueError("kinetic domain boundary is malformed")
        return boundary.rational_value, Fraction(0)
    if boundary.kind != "event" or boundary.event_guard is None:
        raise ValueError("kinetic event boundary is malformed")
    guard = boundary.event_guard
    if guard.exact:
        if guard.lower_bound != guard.upper_bound:
            raise ValueError("exact kinetic guard has a nonzero isolating width")
        return guard.lower_bound, Fraction(0)
    if side == "left":
        return guard.upper_bound, guard.upper_bound - guard.lower_bound
    if side == "right":
        return guard.lower_bound, guard.upper_bound - guard.lower_bound
    raise ValueError("boundary side must be left or right")


def _fraction_to_safe_float(value: Fraction, *, lower: bool) -> float:
    result = float(value)
    exact_float = Fraction.from_float(result)
    if lower and exact_float < value:
        result = math.nextafter(result, math.inf)
    elif not lower and exact_float > value:
        result = math.nextafter(result, -math.inf)
    return result


def _validate_sample_times(
    geometry: KineticChartP0Geometry,
    times: torch.Tensor,
) -> torch.Tensor:
    values = torch.as_tensor(times, dtype=DTYPE, device="cpu").detach().reshape(-1)
    if values.numel() < 1 or not bool(torch.isfinite(values).all().item()):
        raise ValueError("times must be nonempty and finite")
    if bool(torch.any(values < geometry.schedule.t_min).item()):
        raise ValueError("sample time precedes the certified safe chart interval")
    outside_right = values > geometry.schedule.t_max if geometry.right_closed else values >= geometry.schedule.t_max
    if bool(torch.any(outside_right).item()):
        interval_kind = "closed" if geometry.right_closed else "right-open"
        raise ValueError(f"sample time leaves this chart's {interval_kind} safe interval")
    return values.contiguous()


def _require_same_program_semantics(
    supplied: KineticOwnerProgramLike,
    rebound: KineticOwnerProgramLike,
) -> None:
    if type(supplied) is not type(rebound):
        raise ValueError("kinetic owner program changed compiler family during provenance replay")
    if (
        supplied.t_min,
        supplied.t_max,
        supplied.near,
        supplied.far,
        tuple(chart.owner_word for chart in supplied.charts),
    ) != (
        rebound.t_min,
        rebound.t_max,
        rebound.near,
        rebound.far,
        tuple(chart.owner_word for chart in rebound.charts),
    ):
        raise ValueError("supplied kinetic owner program is not bound to these sites/ray")
    supplied_guards = supplied.active_event_guards
    rebound_guards = rebound.active_event_guards
    if len(supplied_guards) != len(rebound_guards):
        raise ValueError("supplied kinetic owner program has a different active seam count")
    for left, right in zip(supplied_guards, rebound_guards, strict=True):
        if _guard_source_signature(left) != _guard_source_signature(right):
            raise ValueError("supplied kinetic owner program has a different active seam predicate")
        if left.upper_bound < right.lower_bound or right.upper_bound < left.lower_bound:
            raise ValueError("supplied and rebound kinetic seam isolators do not overlap")


def _guard_source_signature(guard: KineticEventGuardLike) -> tuple[object, ...]:
    return tuple(
        (
            source.kind,
            source.site_ids,
            source.polynomial.coefficients,
            source.analytic_guard_only,
            multiplicity,
        )
        for source, multiplicity in zip(
            guard.sources,
            guard.source_multiplicities,
            strict=True,
        )
    )


def _source_content_digest(
    sites: AffineKineticPowerSites,
    ray: torch.Tensor,
    *,
    t_min: Fraction,
    t_max: Fraction,
    near: Fraction,
    far: Fraction,
) -> str:
    return _digest_exact_parts(
        "kinetic-source-content-v1",
        _tensor_content_signature(sites.positions0),
        _tensor_content_signature(sites.velocities),
        _tensor_content_signature(sites.weight_coefficients),
        _tensor_content_signature(ray),
        _fraction_signature(t_min),
        _fraction_signature(t_max),
        _fraction_signature(near),
        _fraction_signature(far),
    )


def _program_semantic_digest(program: KineticOwnerProgramLike) -> str:
    chart_signature = tuple(
        (
            chart.chart_id,
            chart.owner_word,
            chart.left_closed,
            chart.right_closed,
        )
        for chart in program.charts
    )
    guard_signature = tuple(
        (
            _fraction_signature(guard.lower_bound),
            _fraction_signature(guard.upper_bound),
            guard.exact,
            _guard_source_signature(guard),
        )
        for guard in program.active_event_guards
    )
    return _digest_exact_parts(
        "kinetic-owner-program-semantics-v1",
        type(program).__name__,
        chart_signature,
        guard_signature,
        program.seam_policy_id,
    )


def _tensor_content_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    value = torch.as_tensor(tensor, dtype=DTYPE, device="cpu").detach().contiguous()
    digest = hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()
    return (tuple(value.shape), str(value.dtype), digest)


def _fraction_signature(value: Fraction) -> tuple[int, int]:
    return value.numerator, value.denominator


def _digest_exact_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _finite_cpu_f64_vector(
    value: torch.Tensor,
    *,
    name: str,
    size: int,
) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=DTYPE, device="cpu").detach().clone().reshape(-1).contiguous()
    if tuple(result.shape) != (size,) or not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"{name} must be finite CPU-compatible data with shape [{size}]")
    return result


def _tensor_bytes(tensors: tuple[torch.Tensor, ...]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


__all__ = [
    "BoundKineticOwnerProgram",
    "KineticChartMaterialVJP",
    "KineticChartP0Geometry",
    "KineticChartP0Transfer",
    "KineticEventGuardLike",
    "KineticOwnerProgramLike",
    "bind_kinetic_owner_program",
    "compile_and_bind_active_kinetic_owner_program",
    "compile_bound_kinetic_chart_p0_geometry",
    "compile_kinetic_chart_p0_geometry",
    "evaluate_kinetic_chart_p0_transfer",
    "kinetic_chart_material_mse_vjp",
    "kinetic_chart_p0_node_material_vjp",
    "ordered_p0_transfer",
    "refresh_kinetic_chart_p0_transfer",
]
