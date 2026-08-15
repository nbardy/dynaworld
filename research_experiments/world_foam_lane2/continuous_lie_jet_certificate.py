"""Sound CPU interval certificate for one fixed-word affine-Lie chart.

This module certifies a narrow but useful statement.  For P0 WorldFoam,
affine rays, ordinary depth, a supplied fixed owner word, and one finite time
interval, it encloses continuously in time both

``exact ordered transfer - decoded compiled Lie transfer``

and its full first derivative with respect to the canonical float64 snapshot
``(boundary planes, affine rays, site densities, site RGB)``.  It additionally
seeds independent perturbations of each referenced lowered Möbius coefficient,
covering the internal sparse-depth VJP seam before canonical scattering.

The implementation is deliberately independent of the sampled adaptive gate
in :mod:`compiled_lie_world_adjoint`.  Every binary64 input is treated as an
exact rational.  Basic operations are rounded outward to a dyadic grid;
``sqrt`` is bounded with integer square roots; and ``exp`` is bounded by an
exact-rational Taylor remainder after range reduction.  Forward-mode interval
automatic differentiation then encloses every entry of the world Jacobian.
Adaptive bisection only tightens these enclosures.  It is not the source of
the proof: the union of accepted leaves covers the full closed interval.

The compiled primal uses the *stored* atlas coefficients.  Its derivative
uses the stored float64 fit matrix applied to exact node-chart derivatives,
which is the real-arithmetic linearization of the current compiler/VJP seam.
Hardware floating-point evaluation error is outside this certificate.

The original public certifier below intentionally remains a dense, global-dual
oracle for tiny fixtures.  :func:`certify_fixed_topology_lie_jet_track_local`
is the production-scale variant: it streams independent track-local dual
problems, caps their dimension before quadratic state is created, and retains
only aggregate bounds plus a canonical global-id scope digest.

Claim boundary
--------------

* certified: continuous transfer and canonical-world Jacobian approximation;
  nonzero cut denominators; positive fiber speed; positive supplied segments;
* assumed: the supplied owner identities are the correct cells, and the
  stored atlas is the refresh produced from the supplied world snapshot;
* excluded: topology discovery, optimizer trust regions, higher-order material
  bases, projective depth, exposure/rolling-shutter integration, and runtime
  binary64/Metal roundoff.

The public result distinguishes a continuous upper bound from optional
point-witness lower bounds.  A witness may disprove a tolerance, but is never
called a continuous certificate.  The Jacobian bound is the entrywise
``sup_t max_(output,parameter)`` error.  Consequently a transfer cotangent
``lambda`` has per-parameter VJP error at most
``||lambda||_1 * world_jacobian_error_upper_bound``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import torch
from compiled_lie_world_adjoint import FAR_CUT_ID, NEAR_CUT_ID, CompiledLieWorldAtlas
from compiled_transfer_adjoint import StableCellWord
from transfer_lie_chart import TemporalTransferAtlas

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class ContinuousLieJetCertificate:
    """Continuous enclosure for a frozen compiled chart and world snapshot."""

    passed: bool
    transfer_tolerance: float
    world_jacobian_tolerance: float
    transfer_error_upper_bound: float
    world_jacobian_error_upper_bound: float
    transfer_point_witness_lower_bound: float | None
    world_jacobian_point_witness_lower_bound: float | None
    world_jacobian_error_upper_bound_by_block: dict[str, float]
    parameter_labels: tuple[str, ...]
    leaf_count: int
    deepest_split: int
    arithmetic_fraction_bits: int
    minimum_cut_denominator_absolute_lower_bound: float | None
    minimum_fiber_speed_lower_bound: float
    minimum_coordinate_segment_length_lower_bound: float
    minimum_physical_segment_length_lower_bound: float
    minimum_exact_total_optical_depth_lower_bound: float
    minimum_compiled_kappa_lower_bound: float
    compiled_lie_cone_certified: bool
    continuous_time_coverage: bool = True
    owner_identity_certified: bool = False
    atlas_snapshot_provenance_certified: bool = False
    runtime_floating_point_roundoff_certified: bool = False
    semantics: str = (
        "real arithmetic over exact binary64 inputs and stored atlas tensors; runtime floating-point rounding excluded"
    )
    certification_mode: str = "dense_global_dual_oracle"
    certified_track_count: int = 0
    maximum_dual_dimension: int = 0
    global_parameter_count: int = 0
    total_seeded_parameter_occurrences: int = 0
    parameter_scope_digest: str | None = None
    parameter_labels_complete: bool = True
    local_dual_dimension_limit: int | None = None


class ContinuousCertificateError(ValueError):
    """The requested fixed-word interval could not be certified fail-closed."""


@dataclass(frozen=True)
class _Interval:
    lo: Fraction
    hi: Fraction

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError("interval lower endpoint exceeds upper endpoint")


@dataclass(frozen=True)
class _Dual:
    value: _Interval
    tangent: tuple[_Interval, ...]
    time_tangent: _Interval
    mixed_time_tangent: tuple[_Interval, ...]


@dataclass(frozen=True)
class _Layout:
    labels: tuple[str, ...]
    boundary: tuple[tuple[int, ...], ...]
    ray: tuple[tuple[int, ...], ...]
    mobius: tuple[tuple[int, ...], ...]
    density: tuple[int, ...]
    color: tuple[tuple[int, ...], ...]
    blocks: tuple[tuple[str, tuple[int, ...]], ...]

    @property
    def size(self) -> int:
        return len(self.labels)


@dataclass(frozen=True)
class _Margins:
    denominator: Fraction | None
    fiber_speed: Fraction
    coordinate_segment_length: Fraction
    physical_segment_length: Fraction
    exact_kappa: Fraction
    compiled_kappa: Fraction
    compiled_cone: Fraction


@dataclass(frozen=True)
class _LeafBounds:
    transfer_upper: Fraction
    jacobian_upper: Fraction
    jacobian_upper_by_block: tuple[Fraction, ...]
    margins: _Margins


class _NeedsSplitError(Exception):
    pass


class _Arithmetic:
    """Exact-rational interval operations with dyadic outward rounding."""

    def __init__(self, fraction_bits: int) -> None:
        if fraction_bits < 64:
            raise ValueError("fraction_bits must be at least 64")
        self.bits = int(fraction_bits)
        self.scale = 1 << self.bits
        self.zero = _Interval(Fraction(0), Fraction(0))
        self.one = _Interval(Fraction(1), Fraction(1))

    def down(self, value: Fraction) -> Fraction:
        scaled = value * self.scale
        return Fraction(scaled.numerator // scaled.denominator, self.scale)

    def up(self, value: Fraction) -> Fraction:
        scaled = value * self.scale
        return Fraction(-((-scaled.numerator) // scaled.denominator), self.scale)

    def point(self, value: float | int | Fraction) -> _Interval:
        exact = (
            value
            if isinstance(value, Fraction)
            else Fraction.from_float(value)
            if isinstance(value, float)
            else Fraction(value)
        )
        return _Interval(self.down(exact), self.up(exact))

    def interval(self, lo: float | Fraction, hi: float | Fraction) -> _Interval:
        lo_q = lo if isinstance(lo, Fraction) else Fraction.from_float(lo)
        hi_q = hi if isinstance(hi, Fraction) else Fraction.from_float(hi)
        return _Interval(self.down(lo_q), self.up(hi_q))

    def add(self, left: _Interval, right: _Interval) -> _Interval:
        return _Interval(self.down(left.lo + right.lo), self.up(left.hi + right.hi))

    def neg(self, value: _Interval) -> _Interval:
        return _Interval(self.down(-value.hi), self.up(-value.lo))

    def sub(self, left: _Interval, right: _Interval) -> _Interval:
        return self.add(left, self.neg(right))

    def mul(self, left: _Interval, right: _Interval) -> _Interval:
        products = (
            left.lo * right.lo,
            left.lo * right.hi,
            left.hi * right.lo,
            left.hi * right.hi,
        )
        return _Interval(self.down(min(products)), self.up(max(products)))

    def reciprocal(self, value: _Interval) -> _Interval:
        if value.lo <= 0 <= value.hi:
            raise _NeedsSplitError("division interval contains zero")
        return _Interval(self.down(1 / value.hi), self.up(1 / value.lo))

    def div(self, numerator: _Interval, denominator: _Interval) -> _Interval:
        return self.mul(numerator, self.reciprocal(denominator))

    def square(self, value: _Interval) -> _Interval:
        lower = Fraction(0) if value.lo <= 0 <= value.hi else min(value.lo * value.lo, value.hi * value.hi)
        upper = max(value.lo * value.lo, value.hi * value.hi)
        return _Interval(self.down(lower), self.up(upper))

    def sqrt(self, value: _Interval) -> _Interval:
        if value.lo < 0:
            raise _NeedsSplitError("square-root interval has a negative lower bound")
        lower = self._sqrt_point(value.lo)[0]
        upper = self._sqrt_point(value.hi)[1]
        return _Interval(self.down(lower), self.up(upper))

    def exp(self, value: _Interval) -> _Interval:
        lower = self._exp_point(value.lo)[0]
        upper = self._exp_point(value.hi)[1]
        return _Interval(self.down(lower), self.up(upper))

    def _sqrt_point(self, value: Fraction) -> tuple[Fraction, Fraction]:
        if value < 0:
            raise ValueError("sqrt requires a non-negative argument")
        if value == 0:
            return Fraction(0), Fraction(0)
        scaled_numerator = value.numerator * self.scale * self.scale
        floor_argument = scaled_numerator // value.denominator
        root = math.isqrt(floor_argument)
        lower = Fraction(root, self.scale)
        if root * root * value.denominator == scaled_numerator:
            return lower, lower
        return lower, Fraction(root + 1, self.scale)

    def _exp_point(self, value: Fraction) -> tuple[Fraction, Fraction]:
        if value == 0:
            return Fraction(1), Fraction(1)
        if value < 0:
            positive_lo, positive_hi = self._exp_positive(-value)
            return self.down(1 / positive_hi), self.up(1 / positive_lo)
        return self._exp_positive(value)

    def _exp_positive(self, value: Fraction) -> tuple[Fraction, Fraction]:
        # exp(x) = exp(x / 2**s)**(2**s).  The reduced positive argument is
        # at most 1/8, so the omitted Taylor terms are bounded by a geometric
        # tail whose ratio is itself an exact rational upper bound.
        reduced = value
        squarings = 0
        while reduced > Fraction(1, 8):
            reduced /= 2
            squarings += 1
        total = Fraction(1)
        term = Fraction(1)
        target = Fraction(1, 1 << (self.bits + 16))
        degree = 0
        while True:
            degree += 1
            term = term * reduced / degree
            total += term
            next_term = term * reduced / (degree + 1)
            ratio = reduced / (degree + 2)
            remainder = next_term / (1 - ratio)
            if remainder <= target:
                break
            if degree > 10000:
                raise ArithmeticError("exp Taylor bound failed to converge")
        result = _Interval(self.down(total), self.up(total + remainder))
        for _ in range(squarings):
            result = self.square(result)
        return result.lo, result.hi

    def dual_constant(self, value: _Interval, size: int) -> _Dual:
        return _Dual(value, (self.zero,) * size, self.zero, (self.zero,) * size)

    def dual_variable(self, value: _Interval, size: int, index: int) -> _Dual:
        tangent = [self.zero] * size
        tangent[index] = self.one
        return _Dual(value, tuple(tangent), self.zero, (self.zero,) * size)

    def dual_time(self, value: _Interval, size: int) -> _Dual:
        return _Dual(value, (self.zero,) * size, self.one, (self.zero,) * size)

    def dual_add(self, left: _Dual, right: _Dual) -> _Dual:
        return _Dual(
            self.add(left.value, right.value),
            tuple(self.add(a, b) for a, b in zip(left.tangent, right.tangent, strict=True)),
            self.add(left.time_tangent, right.time_tangent),
            tuple(self.add(a, b) for a, b in zip(left.mixed_time_tangent, right.mixed_time_tangent, strict=True)),
        )

    def dual_neg(self, value: _Dual) -> _Dual:
        return _Dual(
            self.neg(value.value),
            tuple(self.neg(item) for item in value.tangent),
            self.neg(value.time_tangent),
            tuple(self.neg(item) for item in value.mixed_time_tangent),
        )

    def dual_sub(self, left: _Dual, right: _Dual) -> _Dual:
        return self.dual_add(left, self.dual_neg(right))

    def dual_mul(self, left: _Dual, right: _Dual) -> _Dual:
        return _Dual(
            self.mul(left.value, right.value),
            tuple(
                self.add(self.mul(a, right.value), self.mul(left.value, b))
                for a, b in zip(left.tangent, right.tangent, strict=True)
            ),
            self.add(
                self.mul(left.time_tangent, right.value),
                self.mul(left.value, right.time_tangent),
            ),
            tuple(
                self.add(
                    self.add(
                        self.mul(left_mixed, right.value),
                        self.mul(left_tangent, right.time_tangent),
                    ),
                    self.add(
                        self.mul(left.time_tangent, right_tangent),
                        self.mul(left.value, right_mixed),
                    ),
                )
                for left_mixed, left_tangent, right_tangent, right_mixed in zip(
                    left.mixed_time_tangent,
                    left.tangent,
                    right.tangent,
                    right.mixed_time_tangent,
                    strict=True,
                )
            ),
        )

    def dual_reciprocal(self, value: _Dual) -> _Dual:
        inverse = self.reciprocal(value.value)
        first_derivative = self.neg(self.square(inverse))
        second_derivative = self.mul(self.point(2), self.mul(self.square(inverse), inverse))
        return _Dual(
            inverse,
            tuple(self.mul(first_derivative, item) for item in value.tangent),
            self.mul(first_derivative, value.time_tangent),
            tuple(
                self.add(
                    self.mul(second_derivative, self.mul(value.time_tangent, tangent)),
                    self.mul(first_derivative, mixed),
                )
                for tangent, mixed in zip(value.tangent, value.mixed_time_tangent, strict=True)
            ),
        )

    def dual_div(self, numerator: _Dual, denominator: _Dual) -> _Dual:
        return self.dual_mul(numerator, self.dual_reciprocal(denominator))

    def dual_exp(self, value: _Dual) -> _Dual:
        exponential = self.exp(value.value)
        return _Dual(
            exponential,
            tuple(self.mul(exponential, item) for item in value.tangent),
            self.mul(exponential, value.time_tangent),
            tuple(
                self.mul(
                    exponential,
                    self.add(self.mul(value.time_tangent, tangent), mixed),
                )
                for tangent, mixed in zip(value.tangent, value.mixed_time_tangent, strict=True)
            ),
        )

    def dual_sqrt(self, value: _Dual) -> _Dual:
        root = self.sqrt(value.value)
        if root.lo <= 0:
            raise _NeedsSplitError("fiber-speed derivative has no positive lower bound")
        first_derivative = self.reciprocal(self.mul(self.point(2), root))
        second_derivative = self.neg(self.reciprocal(self.mul(self.point(4), self.mul(self.square(root), root))))
        return _Dual(
            root,
            tuple(self.mul(first_derivative, item) for item in value.tangent),
            self.mul(first_derivative, value.time_tangent),
            tuple(
                self.add(
                    self.mul(second_derivative, self.mul(value.time_tangent, tangent)),
                    self.mul(first_derivative, mixed),
                )
                for tangent, mixed in zip(value.tangent, value.mixed_time_tangent, strict=True)
            ),
        )


@dataclass(frozen=True)
class _SnapshotTensors:
    boundary: torch.Tensor
    rays: torch.Tensor
    density: torch.Tensor
    color: torch.Tensor
    incidence: torch.Tensor


@dataclass(frozen=True)
class _WorldSnapshot(_SnapshotTensors):
    layout: _Layout


@dataclass(frozen=True)
class _CompiledLinearization:
    # [track][coefficient][component][parameter]
    coefficient_tangents: tuple[tuple[tuple[tuple[_Interval, ...], ...], ...], ...]


@dataclass(frozen=True)
class _TrackLocalProblem:
    atlas: CompiledLieWorldAtlas
    boundary: torch.Tensor
    rays: torch.Tensor
    density: torch.Tensor
    color: torch.Tensor
    global_boundary_ids: tuple[int, ...]
    global_incidence_ids: tuple[int, ...]
    global_site_ids: tuple[int, ...]
    dual_dimension: int


def certify_fixed_topology_lie_jet(
    atlas: CompiledLieWorldAtlas,
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    transfer_tolerance: float,
    world_jacobian_tolerance: float,
    max_split_depth: int = 10,
    max_leaf_count: int = 4096,
    arithmetic_fraction_bits: int = 112,
    compute_point_witnesses: bool = True,
) -> ContinuousLieJetCertificate:
    """Certify transfer and full canonical-world Jacobian error on a chart.

    The returned upper bounds hold for every real time in the closed atlas
    interval under the module claim boundary.  ``passed`` is true only when
    both requested absolute tolerances are proved by those continuous bounds.
    """

    if not math.isfinite(transfer_tolerance) or transfer_tolerance < 0:
        raise ValueError("transfer_tolerance must be finite and non-negative")
    if not math.isfinite(world_jacobian_tolerance) or world_jacobian_tolerance < 0:
        raise ValueError("world_jacobian_tolerance must be finite and non-negative")
    if max_split_depth < 0 or max_leaf_count < 1:
        raise ValueError("split depth must be non-negative and max_leaf_count positive")
    _validate_atlas_structure(atlas)
    arithmetic = _Arithmetic(arithmetic_fraction_bits)
    world = _validate_snapshot(
        atlas,
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        site_density=site_density,
        site_color=site_color,
    )
    try:
        linearization = _compile_node_linearization(arithmetic, atlas, world)
    except _NeedsSplitError as error:
        raise ContinuousCertificateError(
            f"the frozen compiler nodes violate a fixed-word precondition: {error}"
        ) from error
    t_min = Fraction.from_float(float(atlas.transfer_atlas.t_min))
    t_max = Fraction.from_float(float(atlas.transfer_atlas.t_max))
    if t_max <= t_min:
        raise ValueError("atlas interval must be finite with t_min < t_max")

    queue: list[tuple[Fraction, Fraction, int]] = [(t_min, t_max, 0)]
    leaves: list[_LeafBounds] = []
    deepest = 0
    while queue:
        lo, hi, depth = queue.pop()
        deepest = max(deepest, depth)
        try:
            leaf = _evaluate_leaf(arithmetic, atlas, world, linearization, lo, hi)
        except _NeedsSplitError as error:
            if depth >= max_split_depth or lo == hi:
                raise ContinuousCertificateError(
                    f"continuous fixed-word precondition remains unproved at max depth on "
                    f"[{float(lo):.17g},{float(hi):.17g}]: {error}"
                ) from error
            leaf = None
        should_split = leaf is None or (
            (
                _fraction_exceeds_float(leaf.transfer_upper, transfer_tolerance)
                or _fraction_exceeds_float(leaf.jacobian_upper, world_jacobian_tolerance)
            )
            and depth < max_split_depth
        )
        if should_split:
            midpoint = (lo + hi) / 2
            if not lo < midpoint < hi:
                raise ContinuousCertificateError("time interval can no longer be bisected")
            if len(leaves) + len(queue) + 2 > max_leaf_count:
                raise ContinuousCertificateError("continuous certificate exceeded max_leaf_count")
            queue.append((midpoint, hi, depth + 1))
            queue.append((lo, midpoint, depth + 1))
        else:
            assert leaf is not None
            leaves.append(leaf)

    transfer_upper = max(leaf.transfer_upper for leaf in leaves)
    jacobian_upper = max(leaf.jacobian_upper for leaf in leaves)
    block_upper = tuple(
        max(leaf.jacobian_upper_by_block[block_id] for leaf in leaves) for block_id in range(len(world.layout.blocks))
    )
    denominator_values = [leaf.margins.denominator for leaf in leaves if leaf.margins.denominator is not None]
    minimum_denominator = min(denominator_values) if denominator_values else None
    minimum_speed = min(leaf.margins.fiber_speed for leaf in leaves)
    minimum_coordinate_length = min(leaf.margins.coordinate_segment_length for leaf in leaves)
    minimum_physical_length = min(leaf.margins.physical_segment_length for leaf in leaves)
    minimum_exact_kappa = min(leaf.margins.exact_kappa for leaf in leaves)
    minimum_compiled_kappa = min(leaf.margins.compiled_kappa for leaf in leaves)
    minimum_cone = min(leaf.margins.compiled_cone for leaf in leaves)
    witness = _point_witness_lower_bounds(arithmetic, atlas, world, linearization) if compute_point_witnesses else None
    transfer_upper_float = _float_up(transfer_upper)
    jacobian_upper_float = _float_up(jacobian_upper)
    parameter_scope_digest = hashlib.sha256(
        json.dumps(
            {
                "schema": "worldfoam-continuous-lie-jet-parameter-scope-v1",
                "mode": "dense_global_dual_oracle",
                "labels": world.layout.labels,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return ContinuousLieJetCertificate(
        passed=(transfer_upper_float <= transfer_tolerance and jacobian_upper_float <= world_jacobian_tolerance),
        transfer_tolerance=float(transfer_tolerance),
        world_jacobian_tolerance=float(world_jacobian_tolerance),
        transfer_error_upper_bound=transfer_upper_float,
        world_jacobian_error_upper_bound=jacobian_upper_float,
        transfer_point_witness_lower_bound=(None if witness is None else _float_down(witness[0])),
        world_jacobian_point_witness_lower_bound=(None if witness is None else _float_down(witness[1])),
        world_jacobian_error_upper_bound_by_block={
            name: _float_up(value) for (name, _), value in zip(world.layout.blocks, block_upper, strict=True)
        },
        parameter_labels=world.layout.labels,
        leaf_count=len(leaves),
        deepest_split=deepest,
        arithmetic_fraction_bits=arithmetic.bits,
        minimum_cut_denominator_absolute_lower_bound=(
            None if minimum_denominator is None else _float_down(minimum_denominator)
        ),
        minimum_fiber_speed_lower_bound=_float_down(minimum_speed),
        minimum_coordinate_segment_length_lower_bound=_float_down(minimum_coordinate_length),
        minimum_physical_segment_length_lower_bound=_float_down(minimum_physical_length),
        minimum_exact_total_optical_depth_lower_bound=_float_down(minimum_exact_kappa),
        minimum_compiled_kappa_lower_bound=_float_down(minimum_compiled_kappa),
        compiled_lie_cone_certified=minimum_cone >= 0,
        certification_mode="dense_global_dual_oracle",
        certified_track_count=atlas.track_count,
        maximum_dual_dimension=world.layout.size,
        global_parameter_count=world.layout.size,
        total_seeded_parameter_occurrences=world.layout.size,
        parameter_scope_digest=parameter_scope_digest,
        parameter_labels_complete=True,
    )


def certify_fixed_topology_lie_jet_track_local(
    atlas: CompiledLieWorldAtlas,
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    transfer_tolerance: float,
    world_jacobian_tolerance: float,
    max_split_depth: int = 10,
    max_leaf_count: int = 4096,
    arithmetic_fraction_bits: int = 112,
    compute_point_witnesses: bool = True,
    shared_parameter_reduction: str = "entrywise_max",
    max_local_dual_dimension: int = 512,
) -> ContinuousLieJetCertificate:
    """Certify a chart with a bounded dual layout streamed one track at a time.

    The dense oracle seeds every canonical world scalar at once.  That makes
    each of its ``D`` dual variables carry a tangent tuple of length ``D`` and
    retains coefficient tangents with a global ``P x J x 4 x D`` dimension.
    This production path instead constructs a one-track problem containing
    only that track's 12 ray coefficients and the boundaries, independent
    Mobius incidences, and sites referenced by its fixed word.  The dense
    certifier remains the arithmetic kernel/oracle for that compact problem;
    only worst-case bounds and a canonical mapping digest survive each track.

    Shared canonical boundaries and sites are sound under the certificate's
    documented *entrywise* Jacobian-error semantics: a track output has zero
    derivative with respect to unrelated parameters, while equal global ids
    in different local scopes refer back to the same canonical scalar.  A
    caller asking for a summed/global VJP norm needs a separate reduction
    proof and therefore fails closed here.

    ``max_leaf_count`` is a per-track work bound.  The returned ``leaf_count``
    is the sum over all certified tracks.  ``max_local_dual_dimension`` is
    checked before constructing any quadratic local dual state, making a
    pathological single word a deterministic failure rather than an OOM.
    """

    if shared_parameter_reduction != "entrywise_max":
        raise ContinuousCertificateError(
            "track-local certification supports only entrywise_max shared-parameter reduction; "
            "summed/global VJP reductions require a separate coupling certificate"
        )
    if not math.isfinite(transfer_tolerance) or transfer_tolerance < 0:
        raise ValueError("transfer_tolerance must be finite and non-negative")
    if not math.isfinite(world_jacobian_tolerance) or world_jacobian_tolerance < 0:
        raise ValueError("world_jacobian_tolerance must be finite and non-negative")
    if max_split_depth < 0 or max_leaf_count < 1:
        raise ValueError("split depth must be non-negative and max_leaf_count positive")
    if max_local_dual_dimension < 1:
        raise ValueError("max_local_dual_dimension must be positive")
    _assert_track_local_cpu_inputs(
        atlas,
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        site_density=site_density,
        site_color=site_color,
    )
    _validate_atlas_structure(atlas)
    tensors = _validate_snapshot_tensors(
        atlas,
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        site_density=site_density,
        site_color=site_color,
    )

    scope_hasher = hashlib.sha256()
    scope_hasher.update(b"worldfoam-track-local-lie-jet-parameter-scope-v1\n")
    block_upper = {
        "boundary": 0.0,
        "ray": 0.0,
        "mobius_depth_coefficient": 0.0,
        "site_density": 0.0,
        "site_color": 0.0,
    }
    transfer_upper = 0.0
    jacobian_upper = 0.0
    transfer_witness: float | None = None
    jacobian_witness: float | None = None
    minimum_denominator: float | None = None
    minimum_speed: float | None = None
    minimum_coordinate_length: float | None = None
    minimum_physical_length: float | None = None
    minimum_exact_kappa: float | None = None
    minimum_compiled_kappa: float | None = None
    total_leaves = 0
    deepest_split = 0
    maximum_dual_dimension = 0
    total_seeded_occurrences = 0
    passed = True
    compiled_cone = True

    incidence_ids_by_track: list[list[int]] = [[] for _ in range(atlas.track_count)]
    for incidence_id, (incidence_track, _) in enumerate(tensors.incidence.tolist()):
        incidence_ids_by_track[int(incidence_track)].append(incidence_id)

    for track_id in range(atlas.track_count):
        local = _make_track_local_problem(
            atlas,
            tensors,
            track_id,
            global_incidence_ids=tuple(incidence_ids_by_track[track_id]),
        )
        if local.dual_dimension > max_local_dual_dimension:
            raise ContinuousCertificateError(
                f"track[{track_id}] local dual dimension {local.dual_dimension} exceeds "
                f"max_local_dual_dimension={max_local_dual_dimension} before interval-jet construction"
            )
        scope_hasher.update(
            json.dumps(
                {
                    "track_id": track_id,
                    "global_boundary_ids": local.global_boundary_ids,
                    "global_incidence_ids": local.global_incidence_ids,
                    "global_site_ids": local.global_site_ids,
                    "dual_dimension": local.dual_dimension,
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        scope_hasher.update(b"\n")
        try:
            report = certify_fixed_topology_lie_jet(
                local.atlas,
                boundary=local.boundary,
                ray_coefficients=local.rays,
                site_density=local.density,
                site_color=local.color,
                transfer_tolerance=transfer_tolerance,
                world_jacobian_tolerance=world_jacobian_tolerance,
                max_split_depth=max_split_depth,
                max_leaf_count=max_leaf_count,
                arithmetic_fraction_bits=arithmetic_fraction_bits,
                compute_point_witnesses=compute_point_witnesses,
            )
        except ContinuousCertificateError as error:
            raise ContinuousCertificateError(f"track[{track_id}] failed local certification: {error}") from error

        passed = passed and report.passed
        compiled_cone = compiled_cone and report.compiled_lie_cone_certified
        transfer_upper = max(transfer_upper, report.transfer_error_upper_bound)
        jacobian_upper = max(jacobian_upper, report.world_jacobian_error_upper_bound)
        for name in block_upper:
            block_upper[name] = max(block_upper[name], report.world_jacobian_error_upper_bound_by_block[name])
        transfer_witness = _optional_max(transfer_witness, report.transfer_point_witness_lower_bound)
        jacobian_witness = _optional_max(jacobian_witness, report.world_jacobian_point_witness_lower_bound)
        minimum_denominator = _optional_min(
            minimum_denominator,
            report.minimum_cut_denominator_absolute_lower_bound,
        )
        minimum_speed = _optional_min(minimum_speed, report.minimum_fiber_speed_lower_bound)
        minimum_coordinate_length = _optional_min(
            minimum_coordinate_length,
            report.minimum_coordinate_segment_length_lower_bound,
        )
        minimum_physical_length = _optional_min(
            minimum_physical_length,
            report.minimum_physical_segment_length_lower_bound,
        )
        minimum_exact_kappa = _optional_min(
            minimum_exact_kappa,
            report.minimum_exact_total_optical_depth_lower_bound,
        )
        minimum_compiled_kappa = _optional_min(
            minimum_compiled_kappa,
            report.minimum_compiled_kappa_lower_bound,
        )
        total_leaves += report.leaf_count
        deepest_split = max(deepest_split, report.deepest_split)
        maximum_dual_dimension = max(maximum_dual_dimension, local.dual_dimension)
        total_seeded_occurrences += local.dual_dimension

    if atlas.track_count < 1:
        raise ContinuousCertificateError("track-local certification requires at least one track")
    if any(
        value is None
        for value in (
            minimum_speed,
            minimum_coordinate_length,
            minimum_physical_length,
            minimum_exact_kappa,
            minimum_compiled_kappa,
        )
    ):
        raise AssertionError("track-local certificate aggregation lost a required margin")
    global_parameter_count = (
        int(tensors.boundary.numel())
        + int(tensors.rays.numel())
        + 4 * int(tensors.incidence.shape[0])
        + int(tensors.density.numel())
        + int(tensors.color.numel())
    )
    return ContinuousLieJetCertificate(
        passed=(
            passed
            and transfer_upper <= transfer_tolerance
            and jacobian_upper <= world_jacobian_tolerance
        ),
        transfer_tolerance=float(transfer_tolerance),
        world_jacobian_tolerance=float(world_jacobian_tolerance),
        transfer_error_upper_bound=transfer_upper,
        world_jacobian_error_upper_bound=jacobian_upper,
        transfer_point_witness_lower_bound=transfer_witness,
        world_jacobian_point_witness_lower_bound=jacobian_witness,
        world_jacobian_error_upper_bound_by_block=block_upper,
        parameter_labels=(),
        leaf_count=total_leaves,
        deepest_split=deepest_split,
        arithmetic_fraction_bits=arithmetic_fraction_bits,
        minimum_cut_denominator_absolute_lower_bound=minimum_denominator,
        minimum_fiber_speed_lower_bound=float(minimum_speed),
        minimum_coordinate_segment_length_lower_bound=float(minimum_coordinate_length),
        minimum_physical_segment_length_lower_bound=float(minimum_physical_length),
        minimum_exact_total_optical_depth_lower_bound=float(minimum_exact_kappa),
        minimum_compiled_kappa_lower_bound=float(minimum_compiled_kappa),
        compiled_lie_cone_certified=compiled_cone,
        certification_mode="track_local_sparse",
        certified_track_count=atlas.track_count,
        maximum_dual_dimension=maximum_dual_dimension,
        global_parameter_count=global_parameter_count,
        total_seeded_parameter_occurrences=total_seeded_occurrences,
        parameter_scope_digest=scope_hasher.hexdigest(),
        parameter_labels_complete=False,
        local_dual_dimension_limit=max_local_dual_dimension,
    )


def _make_track_local_problem(
    atlas: CompiledLieWorldAtlas,
    tensors: _SnapshotTensors,
    track_id: int,
    *,
    global_incidence_ids: tuple[int, ...],
) -> _TrackLocalProblem:
    """Return a canonical direct slice; no shared/global tangent is invented."""

    if track_id < 0 or track_id >= atlas.track_count:
        raise ValueError("track_id leaves the atlas")
    word = atlas.words[track_id]
    cut_values = tuple(
        int(value)
        for value in torch.cat((word.left_cut_ids, word.right_cut_ids)).tolist()
    )
    invalid_negative = tuple(value for value in cut_values if value < 0 and value not in {NEAR_CUT_ID, FAR_CUT_ID})
    if invalid_negative:
        raise ContinuousCertificateError(
            f"track[{track_id}] uses unsupported global/synthetic cut ids {invalid_negative}"
        )
    boundary_ids = tuple(sorted({value for value in cut_values if value >= 0}))
    site_ids = tuple(sorted({int(value) for value in word.owners.tolist()}))
    boundary_map = {global_id: local_id for local_id, global_id in enumerate(boundary_ids)}
    site_map = {global_id: local_id for local_id, global_id in enumerate(site_ids)}

    incidence_cuts = tuple(int(tensors.incidence[index, 1].item()) for index in global_incidence_ids)
    if len(incidence_cuts) != len(set(incidence_cuts)) or set(incidence_cuts) != set(boundary_ids):
        raise ContinuousCertificateError(
            f"track[{track_id}] incidence is not a direct one-row-per-referenced-boundary scope"
        )
    local_incidence = torch.tensor(
        [[0, boundary_map[cut_id]] for cut_id in incidence_cuts],
        dtype=torch.int64,
    ).reshape(-1, 2)
    incidence_index = torch.tensor(global_incidence_ids, dtype=torch.int64)
    boundary_index = torch.tensor(boundary_ids, dtype=torch.int64)
    site_index = torch.tensor(site_ids, dtype=torch.int64)

    def remap_cut(cut_id: int) -> int:
        return cut_id if cut_id < 0 else boundary_map[cut_id]

    local_word = StableCellWord(
        owners=torch.tensor([site_map[int(value)] for value in word.owners.tolist()], dtype=torch.int64),
        left_cut_ids=torch.tensor([remap_cut(int(value)) for value in word.left_cut_ids.tolist()], dtype=torch.int64),
        right_cut_ids=torch.tensor(
            [remap_cut(int(value)) for value in word.right_cut_ids.tolist()],
            dtype=torch.int64,
        ),
    )
    transfer = atlas.transfer_atlas
    local_atlas = CompiledLieWorldAtlas(
        transfer_atlas=TemporalTransferAtlas(
            t_min=transfer.t_min,
            t_max=transfer.t_max,
            node_times=transfer.node_times,
            fit_matrix=transfer.fit_matrix,
            coefficients=transfer.coefficients[track_id : track_id + 1],
            chart=transfer.chart,
        ),
        node_chart=atlas.node_chart[track_id : track_id + 1],
        near=atlas.near,
        far=atlas.far,
        words=(local_word,),
        depth_coefficient_incidence=local_incidence,
        sparse_depth_coefficients=atlas.sparse_depth_coefficients.index_select(0, incidence_index),
        supplied_word_ordering_check=atlas.supplied_word_ordering_check,
    )
    local_boundary = tensors.boundary.index_select(0, boundary_index)
    local_density = tensors.density.index_select(0, site_index)
    local_color = tensors.color.index_select(0, site_index)
    dual_dimension = (
        5 * len(boundary_ids)
        + 12
        + 4 * len(global_incidence_ids)
        + 4 * len(site_ids)
    )
    return _TrackLocalProblem(
        atlas=local_atlas,
        boundary=local_boundary,
        rays=tensors.rays[track_id : track_id + 1],
        density=local_density,
        color=local_color,
        global_boundary_ids=boundary_ids,
        global_incidence_ids=global_incidence_ids,
        global_site_ids=site_ids,
        dual_dimension=dual_dimension,
    )


def _assert_track_local_cpu_inputs(
    atlas: CompiledLieWorldAtlas,
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
) -> None:
    """Reject hidden full-snapshot device/dtype copies on the bounded path."""

    world_tensors = {
        "boundary": boundary,
        "ray_coefficients": ray_coefficients,
        "site_density": site_density,
        "site_color": site_color,
    }
    atlas_tensors = {
        "node_times": atlas.transfer_atlas.node_times,
        "fit_matrix": atlas.transfer_atlas.fit_matrix,
        "coefficients": atlas.transfer_atlas.coefficients,
        "node_chart": atlas.node_chart,
        "depth_coefficient_incidence": atlas.depth_coefficient_incidence,
        "sparse_depth_coefficients": atlas.sparse_depth_coefficients,
    }
    for name, tensor in (*world_tensors.items(), *atlas_tensors.items()):
        if not isinstance(tensor, torch.Tensor):
            raise ContinuousCertificateError(
                f"track-local strict certification requires tensor {name}; hidden materialization is forbidden"
            )
        if tensor.device.type != "cpu":
            raise ContinuousCertificateError(
                f"track-local strict certification requires CPU-resident {name}; "
                "hidden full-snapshot copies are forbidden"
            )
    for name, tensor in world_tensors.items():
        if tensor.dtype != torch.float64:
            raise ContinuousCertificateError(
                f"track-local strict certification requires float64 {name}; "
                "hidden full-snapshot casts are forbidden"
            )


def _optional_max(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _optional_min(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _validate_atlas_structure(atlas: CompiledLieWorldAtlas) -> None:
    transfer_atlas = atlas.transfer_atlas
    if transfer_atlas.chart != "lie":
        raise ValueError("continuous Lie certificate requires a Lie-chart atlas")
    if not all(
        math.isfinite(value)
        for value in (
            transfer_atlas.t_min,
            transfer_atlas.t_max,
            atlas.near,
            atlas.far,
        )
    ):
        raise ValueError("atlas interval and near/far cuts must be finite")
    if transfer_atlas.t_max <= transfer_atlas.t_min or atlas.far <= atlas.near:
        raise ValueError("atlas requires t_min < t_max and near < far")
    node_count = atlas.node_count
    if node_count < 2:
        raise ValueError("continuous Lie certificate requires at least two nodes")
    expected_shapes = {
        "node_times": (node_count,),
        "fit_matrix": (node_count, node_count),
        "coefficients": (atlas.track_count, node_count, 4),
    }
    for name, expected_shape in expected_shapes.items():
        tensor = getattr(transfer_atlas, name).detach().cpu()
        if tuple(tensor.shape) != expected_shape or not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"atlas {name} must be finite with shape {expected_shape}")
    node_chart = atlas.node_chart.detach().cpu()
    if tuple(node_chart.shape) != (atlas.track_count, node_count, 4) or not bool(
        torch.isfinite(node_chart).all().item()
    ):
        raise ValueError(
            f"atlas node_chart must be finite with shape {(atlas.track_count, node_count, 4)}"
        )
    if len(atlas.words) != atlas.track_count:
        raise ValueError("atlas must contain one fixed word per track")
    incidence = atlas.depth_coefficient_incidence.detach().cpu()
    if incidence.ndim != 2 or incidence.shape[1] != 2:
        raise ValueError("atlas depth-coefficient incidence must have shape [I,2]")
    sparse_depth = atlas.sparse_depth_coefficients.detach().cpu()
    if tuple(sparse_depth.shape) != (int(incidence.shape[0]), 4) or not bool(
        torch.isfinite(sparse_depth).all().item()
    ):
        raise ValueError(
            "atlas sparse_depth_coefficients must be finite with shape [I,4]"
        )
    observed = [tuple(int(value) for value in row) for row in incidence.tolist()]
    if len(set(observed)) != len(observed):
        raise ValueError("atlas depth-coefficient incidence rows must be unique")
    expected = {
        (track_id, int(cut_id))
        for track_id, word in enumerate(atlas.words)
        for cut_id in torch.cat((word.left_cut_ids, word.right_cut_ids)).tolist()
        if int(cut_id) >= 0
    }
    if set(observed) != expected:
        raise ValueError("atlas incidence rows must exactly cover referenced track-boundary cuts")


def _validate_snapshot(
    atlas: CompiledLieWorldAtlas,
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
) -> _WorldSnapshot:
    tensors = _validate_snapshot_tensors(
        atlas,
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        site_density=site_density,
        site_color=site_color,
    )
    layout = _make_layout(
        int(tensors.boundary.shape[0]),
        atlas.track_count,
        int(tensors.incidence.shape[0]),
        int(tensors.density.numel()),
    )
    return _WorldSnapshot(
        boundary=tensors.boundary,
        rays=tensors.rays,
        density=tensors.density,
        color=tensors.color,
        incidence=tensors.incidence,
        layout=layout,
    )


def _validate_snapshot_tensors(
    atlas: CompiledLieWorldAtlas,
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
) -> _SnapshotTensors:
    """Validate a canonical snapshot without allocating a global dual layout."""

    tensors = {
        "boundary": torch.as_tensor(boundary, dtype=torch.float64).detach().cpu(),
        "ray_coefficients": torch.as_tensor(ray_coefficients, dtype=torch.float64).detach().cpu(),
        "site_density": torch.as_tensor(site_density, dtype=torch.float64).reshape(-1).detach().cpu(),
        "site_color": torch.as_tensor(site_color, dtype=torch.float64).detach().cpu(),
    }
    if tensors["boundary"].ndim != 2 or tensors["boundary"].shape[1] != 5:
        raise ValueError("boundary must have shape [B,5]")
    if tensors["ray_coefficients"].ndim != 2 or tensors["ray_coefficients"].shape[1] != 12:
        raise ValueError("ray_coefficients must have shape [P,12]")
    if int(tensors["ray_coefficients"].shape[0]) != atlas.track_count:
        raise ValueError("ray_coefficients must have one row per atlas track")
    site_count = int(tensors["site_density"].numel())
    if tuple(tensors["site_color"].shape) != (site_count, 3):
        raise ValueError("site_color must have shape [S,3]")
    if any(not bool(torch.isfinite(value).all().item()) for value in tensors.values()):
        raise ValueError("world snapshot tensors must be finite")
    if bool((tensors["site_density"] < 0).any().item()):
        raise ValueError("P0 site densities must be non-negative")
    if bool(((tensors["site_color"] < 0) | (tensors["site_color"] > 1)).any().item()):
        raise ValueError("physical P0 site colors must lie in [0,1]")
    boundary_count = int(tensors["boundary"].shape[0])
    for word in atlas.words:
        if (
            word.owners.ndim != 1
            or word.left_cut_ids.shape != word.owners.shape
            or word.right_cut_ids.shape != word.owners.shape
        ):
            raise ValueError("each word must contain aligned one-dimensional owner/cut tensors")
        if word.owners.numel() == 0:
            raise ValueError("each fixed owner word must be non-empty")
        if int(word.owners.min().item()) < 0 or int(word.owners.max().item()) >= site_count:
            raise ValueError("word owner leaves the site table")
        cuts = torch.cat((word.left_cut_ids, word.right_cut_ids))
        ordinary = cuts[cuts >= 0]
        if ordinary.numel() and int(ordinary.max().item()) >= boundary_count:
            raise ValueError("word cut leaves the boundary table")
    incidence = atlas.depth_coefficient_incidence.detach().cpu()
    if incidence.ndim != 2 or incidence.shape[1] != 2:
        raise ValueError("atlas depth-coefficient incidence must have shape [I,2]")
    return _SnapshotTensors(
        boundary=tensors["boundary"],
        rays=tensors["ray_coefficients"],
        density=tensors["site_density"],
        color=tensors["site_color"],
        incidence=incidence,
    )


def _make_layout(
    boundary_count: int,
    track_count: int,
    incidence_count: int,
    site_count: int,
) -> _Layout:
    labels: list[str] = []

    def rows(prefix: str, count: int, columns: int) -> tuple[tuple[int, ...], ...]:
        result = []
        for row in range(count):
            indices = []
            for column in range(columns):
                indices.append(len(labels))
                labels.append(f"{prefix}[{row},{column}]")
            result.append(tuple(indices))
        return tuple(result)

    boundary = rows("boundary", boundary_count, 5)
    ray = rows("ray", track_count, 12)
    mobius = rows("mobius_delta", incidence_count, 4)
    density_rows = rows("density", site_count, 1)
    color = rows("color", site_count, 3)
    density = tuple(row[0] for row in density_rows)
    block_members = (
        ("boundary", tuple(index for row in boundary for index in row)),
        ("ray", tuple(index for row in ray for index in row)),
        ("mobius_depth_coefficient", tuple(index for row in mobius for index in row)),
        ("site_density", density),
        ("site_color", tuple(index for row in color for index in row)),
    )
    return _Layout(tuple(labels), boundary, ray, mobius, density, color, block_members)


def _dual_world(
    arithmetic: _Arithmetic,
    world: _WorldSnapshot,
) -> tuple[list[list[_Dual]], list[list[_Dual]], list[list[_Dual]], list[_Dual], list[list[_Dual]]]:
    size = world.layout.size

    def variables(tensor: torch.Tensor, indices: Sequence[Sequence[int]]) -> list[list[_Dual]]:
        return [
            [
                arithmetic.dual_variable(arithmetic.point(float(tensor[row, column].item())), size, index)
                for column, index in enumerate(row_indices)
            ]
            for row, row_indices in enumerate(indices)
        ]

    boundary = variables(world.boundary, world.layout.boundary)
    rays = variables(world.rays, world.layout.ray)
    mobius = variables(
        torch.zeros((len(world.layout.mobius), 4), dtype=torch.float64),
        world.layout.mobius,
    )
    density = [
        arithmetic.dual_variable(arithmetic.point(float(world.density[row].item())), size, index)
        for row, index in enumerate(world.layout.density)
    ]
    color = variables(world.color, world.layout.color)
    return boundary, rays, mobius, density, color


def _compile_node_linearization(
    arithmetic: _Arithmetic,
    atlas: CompiledLieWorldAtlas,
    world: _WorldSnapshot,
) -> _CompiledLinearization:
    node_jacobians: list[list[tuple[tuple[_Interval, ...], ...]]] = []
    for track_id, word in enumerate(atlas.words):
        track_nodes = []
        for time in atlas.transfer_atlas.node_times.tolist():
            _, chart, _ = _exact_track_jet(
                arithmetic,
                world,
                word=word,
                track_id=track_id,
                time=arithmetic.point(float(time)),
                near=atlas.near,
                far=atlas.far,
                need_chart=True,
            )
            assert chart is not None
            track_nodes.append(tuple(component.tangent for component in chart))
        node_jacobians.append(track_nodes)

    fit = atlas.transfer_atlas.fit_matrix.detach().cpu()
    track_coefficients = []
    for track_id in range(atlas.track_count):
        coefficients = []
        for coefficient_id in range(atlas.node_count):
            components = []
            for component in range(4):
                tangents = []
                for parameter in range(world.layout.size):
                    total = arithmetic.zero
                    for node_id in range(atlas.node_count):
                        total = arithmetic.add(
                            total,
                            arithmetic.mul(
                                arithmetic.point(float(fit[coefficient_id, node_id].item())),
                                node_jacobians[track_id][node_id][component][parameter],
                            ),
                        )
                    tangents.append(total)
                components.append(tuple(tangents))
            coefficients.append(tuple(components))
        track_coefficients.append(tuple(coefficients))
    return _CompiledLinearization(tuple(track_coefficients))


def _evaluate_leaf(
    arithmetic: _Arithmetic,
    atlas: CompiledLieWorldAtlas,
    world: _WorldSnapshot,
    linearization: _CompiledLinearization,
    lo: Fraction,
    hi: Fraction,
) -> _LeafBounds:
    time = _Interval(arithmetic.down(lo), arithmetic.up(hi))
    time_jet = arithmetic.dual_time(time, world.layout.size)
    basis = _chebyshev_basis_jet(
        arithmetic,
        time_jet,
        t_min=Fraction.from_float(atlas.transfer_atlas.t_min),
        t_max=Fraction.from_float(atlas.transfer_atlas.t_max),
        rank=atlas.node_count,
    )
    transfer_upper = Fraction(0)
    jacobian_upper = Fraction(0)
    block_upper = [Fraction(0)] * len(world.layout.blocks)
    denominators: list[Fraction] = []
    speeds: list[Fraction] = []
    lengths: list[Fraction] = []
    physical_lengths: list[Fraction] = []
    exact_kappas: list[Fraction] = []
    compiled_kappas: list[Fraction] = []
    cone_margins: list[Fraction] = []
    midpoint = (lo + hi) / 2
    radius = (hi - lo) / 2
    midpoint_interval = _Interval(midpoint, midpoint)
    midpoint_basis = _chebyshev_basis_jet(
        arithmetic,
        arithmetic.dual_time(midpoint_interval, world.layout.size),
        t_min=Fraction.from_float(atlas.transfer_atlas.t_min),
        t_max=Fraction.from_float(atlas.transfer_atlas.t_max),
        rank=atlas.node_count,
    )
    for track_id, word in enumerate(atlas.words):
        exact, _, exact_margins = _exact_track_jet(
            arithmetic,
            world,
            word=word,
            track_id=track_id,
            time=time,
            near=atlas.near,
            far=atlas.far,
            need_chart=False,
        )
        compiled_chart = _compiled_chart_jet(arithmetic, atlas, linearization, track_id, basis)
        compiled = _decode_lie_chart(arithmetic, compiled_chart)
        exact_midpoint, _, _ = _exact_track_jet(
            arithmetic,
            world,
            word=word,
            track_id=track_id,
            time=midpoint_interval,
            near=atlas.near,
            far=atlas.far,
            need_chart=False,
        )
        compiled_midpoint = _decode_lie_chart(
            arithmetic,
            _compiled_chart_jet(
                arithmetic,
                atlas,
                linearization,
                track_id,
                midpoint_basis,
            ),
        )
        compiled_kappa = compiled_chart[0].value
        if _is_exact_zero(compiled_kappa):
            if any(not _is_exact_zero(component.value) for component in compiled_chart[1:]):
                raise _NeedsSplitError("compiled zero-kappa chart has a nonzero color velocity")
        elif compiled_kappa.lo <= 0:
            raise _NeedsSplitError("compiled kappa has no positive lower bound")
        compiled_kappas.append(compiled_kappa.lo)
        cone_margins.extend(
            [compiled_kappa.lo]
            + [component.value.lo for component in compiled_chart[1:]]
            + [arithmetic.sub(compiled_kappa, component.value).lo for component in compiled_chart[1:]]
        )
        denominators.extend(exact_margins[0])
        speeds.append(exact_margins[1])
        lengths.append(exact_margins[2])
        physical_lengths.append(exact_margins[3])
        exact_kappas.append(exact_margins[4])
        for exact_component, compiled_component, exact_mid, compiled_mid in zip(
            exact,
            compiled,
            exact_midpoint,
            compiled_midpoint,
            strict=True,
        ):
            difference = arithmetic.dual_sub(exact_component, compiled_component)
            midpoint_difference = arithmetic.dual_sub(exact_mid, compiled_mid)
            natural_transfer_bound = _maximum_absolute(difference.value)
            centered_transfer_bound = _maximum_absolute(midpoint_difference.value) + radius * _maximum_absolute(
                difference.time_tangent
            )
            transfer_upper = max(
                transfer_upper,
                min(natural_transfer_bound, centered_transfer_bound),
            )
            for parameter, tangent in enumerate(difference.tangent):
                natural_jacobian_bound = _maximum_absolute(tangent)
                centered_jacobian_bound = _maximum_absolute(
                    midpoint_difference.tangent[parameter]
                ) + radius * _maximum_absolute(difference.mixed_time_tangent[parameter])
                error = min(natural_jacobian_bound, centered_jacobian_bound)
                jacobian_upper = max(jacobian_upper, error)
                for block_id, (_, members) in enumerate(world.layout.blocks):
                    if parameter in members:
                        block_upper[block_id] = max(block_upper[block_id], error)
                        break
    return _LeafBounds(
        transfer_upper=transfer_upper,
        jacobian_upper=jacobian_upper,
        jacobian_upper_by_block=tuple(block_upper),
        margins=_Margins(
            denominator=min(denominators) if denominators else None,
            fiber_speed=min(speeds),
            coordinate_segment_length=min(lengths),
            physical_segment_length=min(physical_lengths),
            exact_kappa=min(exact_kappas),
            compiled_kappa=min(compiled_kappas),
            compiled_cone=min(cone_margins),
        ),
    )


def _exact_track_jet(
    arithmetic: _Arithmetic,
    world: _WorldSnapshot,
    *,
    word: StableCellWord,
    track_id: int,
    time: _Interval,
    near: float,
    far: float,
    need_chart: bool,
) -> tuple[
    tuple[_Dual, ...],
    tuple[_Dual, ...] | None,
    tuple[list[Fraction], Fraction, Fraction, Fraction, Fraction],
]:
    boundary, rays, mobius_delta, density, color = _dual_world(arithmetic, world)
    track_cut_to_incidence = {
        (int(incidence_track), int(cut_id)): incidence_id
        for incidence_id, (incidence_track, cut_id) in enumerate(world.incidence.tolist())
    }
    time_dual = arithmetic.dual_time(time, world.layout.size)
    direction = [
        arithmetic.dual_add(rays[track_id][6 + axis], arithmetic.dual_mul(time_dual, rays[track_id][9 + axis]))
        for axis in range(3)
    ]
    speed_squared = arithmetic.dual_constant(arithmetic.zero, world.layout.size)
    for component in direction:
        speed_squared = arithmetic.dual_add(speed_squared, arithmetic.dual_mul(component, component))
    speed = arithmetic.dual_sqrt(speed_squared)
    if speed.value.lo <= 0:
        raise _NeedsSplitError("ray fiber speed has no positive lower bound")
    kappa = arithmetic.dual_constant(arithmetic.zero, world.layout.size)
    prefix_beta = arithmetic.dual_constant(arithmetic.one, world.layout.size)
    moment = [arithmetic.dual_constant(arithmetic.zero, world.layout.size) for _ in range(3)]
    denominator_margins: list[Fraction] = []
    segment_lower = None
    physical_segment_lower = None
    for owner_raw, left_raw, right_raw in zip(
        word.owners.tolist(), word.left_cut_ids.tolist(), word.right_cut_ids.tolist(), strict=True
    ):
        owner = int(owner_raw)
        left, left_denominator = _cut_depth(
            arithmetic,
            boundary,
            rays[track_id],
            time_dual,
            int(left_raw),
            coefficient_delta=(
                None if int(left_raw) < 0 else mobius_delta[track_cut_to_incidence[(track_id, int(left_raw))]]
            ),
            near=near,
            far=far,
        )
        right, right_denominator = _cut_depth(
            arithmetic,
            boundary,
            rays[track_id],
            time_dual,
            int(right_raw),
            coefficient_delta=(
                None if int(right_raw) < 0 else mobius_delta[track_cut_to_incidence[(track_id, int(right_raw))]]
            ),
            near=near,
            far=far,
        )
        if left_denominator is not None:
            denominator_margins.append(left_denominator)
        if right_denominator is not None:
            denominator_margins.append(right_denominator)
        coordinate_length = arithmetic.dual_sub(right, left)
        if coordinate_length.value.lo <= 0:
            raise _NeedsSplitError("supplied word segment has no positive length lower bound")
        segment_lower = (
            coordinate_length.value.lo if segment_lower is None else min(segment_lower, coordinate_length.value.lo)
        )
        physical_length = arithmetic.dual_mul(speed, coordinate_length)
        physical_segment_lower = (
            physical_length.value.lo
            if physical_segment_lower is None
            else min(physical_segment_lower, physical_length.value.lo)
        )
        optical_depth = arithmetic.dual_mul(density[owner], physical_length)
        if optical_depth.value.lo < 0:
            raise _NeedsSplitError("optical-depth interval became negative")
        beta = arithmetic.dual_exp(arithmetic.dual_neg(optical_depth))
        alpha = arithmetic.dual_sub(arithmetic.dual_constant(arithmetic.one, world.layout.size), beta)
        for channel in range(3):
            moment[channel] = arithmetic.dual_add(
                moment[channel],
                arithmetic.dual_mul(arithmetic.dual_mul(prefix_beta, alpha), color[owner][channel]),
            )
        prefix_beta = arithmetic.dual_mul(prefix_beta, beta)
        kappa = arithmetic.dual_add(kappa, optical_depth)
    if segment_lower is None or physical_segment_lower is None:
        raise ValueError("fixed word must be non-empty")
    transfer = (prefix_beta, *moment)
    chart = None
    if need_chart:
        if _is_exact_zero(kappa.value):
            inverse_phi = _dual_unary_at_exact_zero(
                arithmetic,
                kappa,
                value=Fraction(1),
                first_derivative=Fraction(1, 2),
                second_derivative=Fraction(1, 6),
            )
        elif kappa.value.lo <= 0:
            raise ContinuousCertificateError(
                "Lie node derivative requires a strictly positive total optical-depth margin"
            )
        else:
            one_minus_beta = arithmetic.dual_sub(
                arithmetic.dual_constant(arithmetic.one, world.layout.size),
                prefix_beta,
            )
            inverse_phi = arithmetic.dual_div(kappa, one_minus_beta)
        chart = (kappa, *(arithmetic.dual_mul(inverse_phi, component) for component in moment))
    return (
        transfer,
        chart,
        (
            denominator_margins,
            speed.value.lo,
            segment_lower,
            physical_segment_lower,
            kappa.value.lo,
        ),
    )


def _cut_depth(
    arithmetic: _Arithmetic,
    boundary: list[list[_Dual]],
    ray: list[_Dual],
    time: _Dual,
    cut_id: int,
    *,
    coefficient_delta: list[_Dual] | None,
    near: float,
    far: float,
) -> tuple[_Dual, Fraction | None]:
    size = len(time.tangent)
    if cut_id == NEAR_CUT_ID:
        return arithmetic.dual_constant(arithmetic.point(near), size), None
    if cut_id == FAR_CUT_ID:
        return arithmetic.dual_constant(arithmetic.point(far), size), None
    if cut_id < 0 or cut_id >= len(boundary):
        raise ValueError(f"invalid ordinary cut id {cut_id}")
    plane = boundary[cut_id]
    dot_o0 = arithmetic.dual_constant(arithmetic.zero, size)
    dot_o1 = arithmetic.dual_constant(arithmetic.zero, size)
    dot_d0 = arithmetic.dual_constant(arithmetic.zero, size)
    dot_d1 = arithmetic.dual_constant(arithmetic.zero, size)
    for axis in range(3):
        dot_o0 = arithmetic.dual_add(dot_o0, arithmetic.dual_mul(ray[axis], plane[axis]))
        dot_o1 = arithmetic.dual_add(dot_o1, arithmetic.dual_mul(ray[3 + axis], plane[axis]))
        dot_d0 = arithmetic.dual_add(dot_d0, arithmetic.dual_mul(ray[6 + axis], plane[axis]))
        dot_d1 = arithmetic.dual_add(dot_d1, arithmetic.dual_mul(ray[9 + axis], plane[axis]))
    coefficient_a = arithmetic.dual_neg(arithmetic.dual_add(dot_o0, plane[4]))
    coefficient_b = arithmetic.dual_neg(arithmetic.dual_add(dot_o1, plane[3]))
    if coefficient_delta is None or len(coefficient_delta) != 4:
        raise ValueError("ordinary cuts require four independent Möbius perturbations")
    coefficient_a = arithmetic.dual_add(coefficient_a, coefficient_delta[0])
    coefficient_b = arithmetic.dual_add(coefficient_b, coefficient_delta[1])
    coefficient_c = arithmetic.dual_add(dot_d0, coefficient_delta[2])
    coefficient_d = arithmetic.dual_add(dot_d1, coefficient_delta[3])
    numerator = arithmetic.dual_add(coefficient_a, arithmetic.dual_mul(time, coefficient_b))
    denominator = arithmetic.dual_add(coefficient_c, arithmetic.dual_mul(time, coefficient_d))
    if denominator.value.lo <= 0 <= denominator.value.hi:
        raise _NeedsSplitError("Möbius cut denominator contains zero")
    denominator_margin = min(abs(denominator.value.lo), abs(denominator.value.hi))
    return arithmetic.dual_div(numerator, denominator), denominator_margin


def _compiled_chart_jet(
    arithmetic: _Arithmetic,
    atlas: CompiledLieWorldAtlas,
    linearization: _CompiledLinearization,
    track_id: int,
    basis: tuple[_Dual, ...],
) -> tuple[_Dual, ...]:
    components = []
    coefficients = atlas.transfer_atlas.coefficients.detach().cpu()
    for component in range(4):
        size = len(linearization.coefficient_tangents[track_id][0][component])
        total = arithmetic.dual_constant(arithmetic.zero, size)
        for coefficient_id, basis_value in enumerate(basis):
            coefficient = _Dual(
                arithmetic.point(float(coefficients[track_id, coefficient_id, component].item())),
                linearization.coefficient_tangents[track_id][coefficient_id][component],
                arithmetic.zero,
                (arithmetic.zero,) * size,
            )
            total = arithmetic.dual_add(total, arithmetic.dual_mul(basis_value, coefficient))
        components.append(total)
    return tuple(components)


def _decode_lie_chart(arithmetic: _Arithmetic, chart: tuple[_Dual, ...]) -> tuple[_Dual, ...]:
    kappa = chart[0]
    beta = arithmetic.dual_exp(arithmetic.dual_neg(kappa))
    if _is_exact_zero(kappa.value):
        phi = _dual_unary_at_exact_zero(
            arithmetic,
            kappa,
            value=Fraction(1),
            first_derivative=Fraction(-1, 2),
            second_derivative=Fraction(1, 3),
        )
    elif kappa.value.lo <= 0:
        raise _NeedsSplitError("compiled Lie kappa interval reaches its removable singularity")
    else:
        phi = arithmetic.dual_div(
            arithmetic.dual_sub(arithmetic.dual_constant(arithmetic.one, len(kappa.tangent)), beta),
            kappa,
        )
    return (beta, *(arithmetic.dual_mul(phi, component) for component in chart[1:]))


def _is_exact_zero(value: _Interval) -> bool:
    return value.lo == 0 and value.hi == 0


def _dual_unary_at_exact_zero(
    arithmetic: _Arithmetic,
    argument: _Dual,
    *,
    value: Fraction,
    first_derivative: Fraction,
    second_derivative: Fraction,
) -> _Dual:
    """Apply a scalar removable-singularity jet at the exact point zero."""

    if not _is_exact_zero(argument.value):
        raise ValueError("exact-zero unary branch requires a point-zero primal")
    result = arithmetic.point(value)
    first = arithmetic.point(first_derivative)
    second = arithmetic.point(second_derivative)
    return _Dual(
        result,
        tuple(arithmetic.mul(first, tangent) for tangent in argument.tangent),
        arithmetic.mul(first, argument.time_tangent),
        tuple(
            arithmetic.add(
                arithmetic.mul(second, arithmetic.mul(argument.time_tangent, tangent)),
                arithmetic.mul(first, mixed),
            )
            for tangent, mixed in zip(
                argument.tangent,
                argument.mixed_time_tangent,
                strict=True,
            )
        ),
    )


def _chebyshev_basis_jet(
    arithmetic: _Arithmetic,
    time: _Dual,
    *,
    t_min: Fraction,
    t_max: Fraction,
    rank: int,
) -> tuple[_Dual, ...]:
    size = len(time.tangent)
    # ``chebyshev_basis`` forms these two scalar constants in Python before
    # entering Torch.  Treat those binary64 results—not an unrounded rational
    # re-evaluation of the additions—as the stored evaluator constants.
    endpoint_sum = Fraction.from_float(float(float(t_max) + float(t_min)))
    endpoint_difference = Fraction.from_float(float(float(t_max) - float(t_min)))
    normalized = arithmetic.dual_div(
        arithmetic.dual_sub(
            arithmetic.dual_mul(arithmetic.dual_constant(arithmetic.point(2), size), time),
            arithmetic.dual_constant(arithmetic.point(endpoint_sum), size),
        ),
        arithmetic.dual_constant(arithmetic.point(endpoint_difference), size),
    )
    basis = [arithmetic.dual_constant(arithmetic.one, size)]
    if rank > 1:
        basis.append(normalized)
    for _ in range(2, rank):
        basis.append(
            arithmetic.dual_sub(
                arithmetic.dual_mul(
                    arithmetic.dual_mul(arithmetic.dual_constant(arithmetic.point(2), size), normalized),
                    basis[-1],
                ),
                basis[-2],
            )
        )
    return tuple(basis)


def _point_witness_lower_bounds(
    arithmetic: _Arithmetic,
    atlas: CompiledLieWorldAtlas,
    world: _WorldSnapshot,
    linearization: _CompiledLinearization,
) -> tuple[Fraction, Fraction]:
    points = {
        Fraction.from_float(atlas.transfer_atlas.t_min),
        Fraction.from_float(atlas.transfer_atlas.t_max),
        (Fraction.from_float(atlas.transfer_atlas.t_min) + Fraction.from_float(atlas.transfer_atlas.t_max)) / 2,
        *(Fraction.from_float(float(value)) for value in atlas.transfer_atlas.node_times.tolist()),
    }
    transfer_lower = Fraction(0)
    jacobian_lower = Fraction(0)
    for point in points:
        leaf = _evaluate_point_errors(arithmetic, atlas, world, linearization, point)
        transfer_lower = max(transfer_lower, leaf[0])
        jacobian_lower = max(jacobian_lower, leaf[1])
    return transfer_lower, jacobian_lower


def _evaluate_point_errors(
    arithmetic: _Arithmetic,
    atlas: CompiledLieWorldAtlas,
    world: _WorldSnapshot,
    linearization: _CompiledLinearization,
    point: Fraction,
) -> tuple[Fraction, Fraction]:
    time = _Interval(point, point)
    basis = _chebyshev_basis_jet(
        arithmetic,
        arithmetic.dual_time(time, world.layout.size),
        t_min=Fraction.from_float(atlas.transfer_atlas.t_min),
        t_max=Fraction.from_float(atlas.transfer_atlas.t_max),
        rank=atlas.node_count,
    )
    transfer_lower = Fraction(0)
    jacobian_lower = Fraction(0)
    for track_id, word in enumerate(atlas.words):
        exact, _, _ = _exact_track_jet(
            arithmetic,
            world,
            word=word,
            track_id=track_id,
            time=time,
            near=atlas.near,
            far=atlas.far,
            need_chart=False,
        )
        compiled = _decode_lie_chart(
            arithmetic,
            _compiled_chart_jet(arithmetic, atlas, linearization, track_id, basis),
        )
        for exact_component, compiled_component in zip(exact, compiled, strict=True):
            difference = arithmetic.dual_sub(exact_component, compiled_component)
            transfer_lower = max(transfer_lower, _minimum_absolute(difference.value))
            jacobian_lower = max(
                jacobian_lower,
                *(_minimum_absolute(value) for value in difference.tangent),
            )
    return transfer_lower, jacobian_lower


def _maximum_absolute(value: _Interval) -> Fraction:
    return max(abs(value.lo), abs(value.hi))


def _minimum_absolute(value: _Interval) -> Fraction:
    if value.lo <= 0 <= value.hi:
        return Fraction(0)
    return min(abs(value.lo), abs(value.hi))


def _fraction_exceeds_float(value: Fraction, threshold: float) -> bool:
    return value > Fraction.from_float(threshold)


def _float_up(value: Fraction) -> float:
    result = float(value)
    if result == -math.inf:
        return math.nextafter(-math.inf, math.inf)
    if result == math.inf:
        return math.inf
    if Fraction.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def _float_down(value: Fraction) -> float:
    result = float(value)
    if result == math.inf:
        return math.nextafter(math.inf, -math.inf)
    if result == -math.inf:
        return -math.inf
    if Fraction.from_float(result) > value:
        result = math.nextafter(result, -math.inf)
    return result
