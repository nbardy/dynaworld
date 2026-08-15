"""Continuous third-cell owner certificate for fixed-word WorldFoam charts.

The transfer/Jacobian certificate proves that a *supplied* ordered word is
evaluated accurately.  That is insufficient for a power diagram: a third site
that is not named by either adjacent cut can still have lower power distance
inside a claimed segment.  This module closes that logical gap for P0 power
cells, affine ray tracks, ordinary depth, and one fixed time chart.

For owner ``i`` and competitor ``j`` the power-distance difference is affine
in the spacetime point::

    Delta_ij(x,t) = d_i(x,t) - d_j(x,t)
                  = n_ij . x + n_t,ij t + b_ij.

Along an affine ray ``x(t,z)=o0+t o1+z(d0+t d1)`` it is affine in ``z`` for
every fixed ``t``.  Therefore an owner wins throughout a word segment iff it
wins at both segment endpoints.  Finite endpoints are the same Mobius cuts
used by the renderer.  We bound both endpoint inequalities continuously in
time with the exact-rational, outward-rounded interval arithmetic used by the
independent Lie-jet certificate.  Every owner is compared against *every*
site, so a passed result excludes an unlisted third-cell undercut.

The certificate is deliberately compile-time work.  Its cost depends on
tracks, word runs, sites, and adaptive time leaves, never on requested frame
sampling density.  It does not discover a better word after failure, certify
runtime floating-point roundoff, or differentiate through topology changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import torch
from compiled_transfer_adjoint import (
    FAR_CUT_ID,
    NEAR_CUT_ID,
    StableCellWord,
    check_supplied_word_ordering,
)
from continuous_lie_jet_certificate import (
    _Arithmetic,
    _float_down,
    _float_up,
    _Interval,
    _NeedsSplitError,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class ContinuousOwnerIdentityCertificate:
    """Sound fixed-chart exclusion of every third-site power undercut."""

    passed: bool
    ownership_tolerance: float
    track_count: int
    run_count: int
    site_count: int
    leaf_count: int
    deepest_split: int
    checked_endpoint_inequality_count: int
    maximum_owner_difference_upper_bound: float
    minimum_certified_owner_margin: float
    arithmetic_fraction_bits: int
    continuous_time_coverage: bool = True
    owner_identity_certified: bool = True
    all_competitor_sites_checked: bool = True
    runtime_floating_point_roundoff_certified: bool = False
    semantics: str = (
        "real arithmetic over exact binary64 site, boundary, and affine-ray inputs; "
        "owner ties up to ownership_tolerance are accepted"
    )


class ContinuousOwnerIdentityError(ValueError):
    """The fixed word is wrong or could not be certified within its budget."""


class _OwnerNeedsSplitError(Exception):
    def __init__(self, message: str, upper_bound: Fraction) -> None:
        super().__init__(message)
        self.upper_bound = upper_bound


class _OwnerViolationError(Exception):
    pass


@dataclass(frozen=True)
class _OwnerWorld:
    sites: torch.Tensor
    boundary: torch.Tensor
    rays: torch.Tensor
    words: tuple[StableCellWord, ...]


def certify_fixed_word_owner_identity(
    *,
    sites: torch.Tensor,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    words: Sequence[StableCellWord],
    t_min: float,
    t_max: float,
    near: float,
    far: float,
    ownership_tolerance: float = 1.0e-9,
    denominator_epsilon: float = 1.0e-9,
    segment_length_epsilon: float = 1.0e-8,
    max_split_depth: int = 14,
    max_leaf_count: int = 4096,
    max_work_units: int = 2_000_000,
    arithmetic_fraction_bits: int = 112,
) -> ContinuousOwnerIdentityCertificate:
    """Certify every supplied owner against every power site for all times.

    ``ownership_tolerance`` is a power-distance tolerance, not a spatial
    distance.  Exact transition ties need tolerance zero in real arithmetic,
    but a small explicit tolerance is normally required when site-derived
    planes and stored binary64 boundaries have undergone independent rounding.
    """

    scalars = (t_min, t_max, near, far, ownership_tolerance, denominator_epsilon)
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("owner certificate bounds and tolerances must be finite")
    if t_max <= t_min or far <= near:
        raise ValueError("owner certificate requires t_min < t_max and near < far")
    if ownership_tolerance < 0.0 or denominator_epsilon <= 0.0 or segment_length_epsilon <= 0.0:
        raise ValueError("owner tolerance must be nonnegative and safety epsilons positive")
    if max_split_depth < 0 or max_leaf_count < 1 or max_work_units < 1:
        raise ValueError("owner certificate split, leaf, and work budgets must be positive")
    if arithmetic_fraction_bits < 64:
        raise ValueError("arithmetic_fraction_bits must be at least 64")

    world = _validate_world(sites, boundary, ray_coefficients, words)
    try:
        ordering = check_supplied_word_ordering(
            boundary=world.boundary,
            ray_coefficients=world.rays,
            words=world.words,
            site_count=int(world.sites.shape[0]),
            t_min=t_min,
            t_max=t_max,
            near=near,
            far=far,
            denominator_epsilon=denominator_epsilon,
            length_epsilon=segment_length_epsilon,
        )
    except ValueError as error:
        raise ContinuousOwnerIdentityError(
            f"supplied-word ordering prerequisite failed: {error}"
        ) from error
    if not bool(ordering["passed"]):
        raise ContinuousOwnerIdentityError("supplied-word ordering prerequisite failed")

    arithmetic = _Arithmetic(arithmetic_fraction_bits)
    tolerance = Fraction.from_float(ownership_tolerance)
    t_lo = Fraction.from_float(t_min)
    t_hi = Fraction.from_float(t_max)
    queue: list[tuple[Fraction, Fraction, int]] = [(t_lo, t_hi, 0)]
    accepted_upper_bounds: list[Fraction] = []
    deepest_split = 0
    checked = 0
    checks_per_interval = 2 * sum(
        int(word.owners.numel()) * (int(world.sites.shape[0]) - 1)
        for word in world.words
    )

    while queue:
        lo, hi, depth = queue.pop()
        deepest_split = max(deepest_split, depth)
        checked += checks_per_interval
        if checked > max_work_units:
            raise ContinuousOwnerIdentityError(
                f"owner certificate work budget exceeded: {checked} > {max_work_units}"
            )
        try:
            upper = _certify_time_interval(
                arithmetic,
                world,
                lo=lo,
                hi=hi,
                near=near,
                far=far,
                tolerance=tolerance,
            )
            accepted_upper_bounds.append(upper)
        except _OwnerViolationError as error:
            raise ContinuousOwnerIdentityError(str(error)) from error
        except (_OwnerNeedsSplitError, _NeedsSplitError) as error:
            if depth >= max_split_depth or lo == hi:
                raise ContinuousOwnerIdentityError(
                    "continuous owner identity remains unproved at maximum split depth on "
                    f"[{float(lo):.17g},{float(hi):.17g}]: {error}"
                ) from error
            if len(queue) + len(accepted_upper_bounds) + 2 > max_leaf_count:
                raise ContinuousOwnerIdentityError(
                    f"owner certificate leaf budget exceeded: limit={max_leaf_count}"
                ) from error
            midpoint = (lo + hi) / 2
            queue.append((midpoint, hi, depth + 1))
            queue.append((lo, midpoint, depth + 1))

    maximum_upper = max(accepted_upper_bounds, default=Fraction(0))
    return ContinuousOwnerIdentityCertificate(
        passed=True,
        ownership_tolerance=ownership_tolerance,
        track_count=int(world.rays.shape[0]),
        run_count=sum(int(word.owners.numel()) for word in world.words),
        site_count=int(world.sites.shape[0]),
        leaf_count=len(accepted_upper_bounds),
        deepest_split=deepest_split,
        checked_endpoint_inequality_count=checked,
        maximum_owner_difference_upper_bound=_float_up(maximum_upper),
        minimum_certified_owner_margin=_float_down(-maximum_upper),
        arithmetic_fraction_bits=arithmetic_fraction_bits,
    )


def _validate_world(
    sites: torch.Tensor,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    words: Sequence[StableCellWord],
) -> _OwnerWorld:
    sites_f64 = torch.as_tensor(sites, dtype=torch.float64).detach().cpu()
    boundary_f64 = torch.as_tensor(boundary, dtype=torch.float64).detach().cpu()
    rays_f64 = torch.as_tensor(ray_coefficients, dtype=torch.float64).detach().cpu()
    words_tuple = tuple(words)
    if sites_f64.ndim != 2 or sites_f64.shape[1] != 5 or int(sites_f64.shape[0]) < 1:
        raise ValueError("sites must have shape [S,5] with S >= 1")
    if boundary_f64.ndim != 2 or boundary_f64.shape[1] != 5:
        raise ValueError("boundary must have shape [B,5]")
    if rays_f64.ndim != 2 or rays_f64.shape[1] != 12:
        raise ValueError("ray_coefficients must have shape [P,12]")
    if len(words_tuple) != int(rays_f64.shape[0]) or not words_tuple:
        raise ValueError("words must contain one nonempty fixed word per ray track")
    if not all(bool(torch.isfinite(value).all().item()) for value in (sites_f64, boundary_f64, rays_f64)):
        raise ValueError("owner certificate tensors must be finite")
    boundary_count = int(boundary_f64.shape[0])
    site_count = int(sites_f64.shape[0])
    for word in words_tuple:
        if word.owners.ndim != 1 or int(word.owners.numel()) < 1:
            raise ValueError("each owner word must be a nonempty vector")
        if word.left_cut_ids.shape != word.owners.shape or word.right_cut_ids.shape != word.owners.shape:
            raise ValueError("word owner and cut vectors must have identical shapes")
        if int(word.owners.min()) < 0 or int(word.owners.max()) >= site_count:
            raise ValueError("word owner lies outside the site table")
        cuts = torch.cat((word.left_cut_ids, word.right_cut_ids))
        finite = cuts[cuts >= 0]
        if finite.numel() and int(finite.max()) >= boundary_count:
            raise ValueError("word cut lies outside the boundary table")
    return _OwnerWorld(sites_f64, boundary_f64, rays_f64, words_tuple)


def _certify_time_interval(
    arithmetic: _Arithmetic,
    world: _OwnerWorld,
    *,
    lo: Fraction,
    hi: Fraction,
    near: float,
    far: float,
    tolerance: Fraction,
) -> Fraction:
    time = arithmetic.interval(lo, hi)
    witness_time = arithmetic.point((lo + hi) / 2)
    maximum_upper: Fraction | None = None
    for track_id, word in enumerate(world.words):
        ray = tuple(arithmetic.point(float(value)) for value in world.rays[track_id].tolist())
        for run_id, (owner_raw, left_raw, right_raw) in enumerate(
            zip(
                word.owners.tolist(),
                word.left_cut_ids.tolist(),
                word.right_cut_ids.tolist(),
                strict=True,
            )
        ):
            owner = int(owner_raw)
            endpoint_ids = (int(left_raw), int(right_raw))
            endpoint_depths = tuple(
                _cut_depth_interval(
                    arithmetic,
                    world.boundary,
                    ray,
                    cut_id=cut_id,
                    time=time,
                    near=near,
                    far=far,
                )
                for cut_id in endpoint_ids
            )
            witness_depths = tuple(
                _cut_depth_interval(
                    arithmetic,
                    world.boundary,
                    ray,
                    cut_id=cut_id,
                    time=witness_time,
                    near=near,
                    far=far,
                )
                for cut_id in endpoint_ids
            )
            for competitor in range(int(world.sites.shape[0])):
                if competitor == owner:
                    continue
                plane = _power_difference_plane(arithmetic, world.sites, owner, competitor)
                for endpoint_id, depth, witness_depth in zip(
                    endpoint_ids,
                    endpoint_depths,
                    witness_depths,
                    strict=True,
                ):
                    endpoint_plane = _endpoint_owner_plane(
                        arithmetic,
                        world,
                        plane=plane,
                        cut_id=endpoint_id,
                        owner=owner,
                        competitor=competitor,
                    )
                    witness = _owner_difference(
                        arithmetic,
                        endpoint_plane,
                        ray,
                        witness_time,
                        witness_depth,
                    )
                    if witness.lo > tolerance:
                        raise _OwnerViolationError(
                            "third-cell undercut witness: "
                            f"track={track_id}, run={run_id}, owner={owner}, competitor={competitor}, "
                            f"cut={endpoint_id}, t={float((lo + hi) / 2):.17g}, "
                            f"power_difference_lower={float(witness.lo):.9g}, "
                            f"tolerance={float(tolerance):.9g}"
                        )
                    value = _owner_difference(arithmetic, endpoint_plane, ray, time, depth)
                    maximum_upper = value.hi if maximum_upper is None else max(maximum_upper, value.hi)
                    if value.hi > tolerance:
                        raise _OwnerNeedsSplitError(
                            "owner inequality enclosure crosses tolerance: "
                            f"track={track_id}, run={run_id}, owner={owner}, competitor={competitor}, "
                            f"cut={endpoint_id}, upper={float(value.hi):.9g}, "
                            f"tolerance={float(tolerance):.9g}",
                            value.hi,
                        )
    return Fraction(0) if maximum_upper is None else maximum_upper


def _endpoint_owner_plane(
    arithmetic: _Arithmetic,
    world: _OwnerWorld,
    *,
    plane: tuple[_Interval, ...],
    cut_id: int,
    owner: int,
    competitor: int,
) -> tuple[_Interval, ...]:
    """Remove an exact multiple of a finite endpoint's stored zero plane.

    At a finite word endpoint the stored cut plane is exactly zero in the
    real-arithmetic model. Subtracting any scalar multiple therefore leaves
    the owner inequality unchanged at that endpoint. Choosing the largest
    stored coefficient as a pivot cancels the dominant dependency and makes
    site-derived planes that differ by one rounding bit certifiable without
    pretending they are bit-identical.
    """

    if cut_id < 0:
        return plane

    stored = tuple(Fraction.from_float(float(value)) for value in world.boundary[cut_id].tolist())
    left = tuple(Fraction.from_float(float(value)) for value in world.sites[owner].tolist())
    right = tuple(Fraction.from_float(float(value)) for value in world.sites[competitor].tolist())
    derived = [2 * (right[axis] - left[axis]) for axis in range(4)]
    bias = sum((left[axis] * left[axis] - right[axis] * right[axis] for axis in range(4)), Fraction(0))
    derived.append(bias - left[4] + right[4])
    pivot = max(range(5), key=lambda index: abs(stored[index]))
    if stored[pivot] == 0:
        raise ValueError("a finite word cut references a zero plane")
    scale = derived[pivot] / stored[pivot]
    return tuple(
        arithmetic.point(derived_value - scale * stored_value)
        for stored_value, derived_value in zip(stored, derived, strict=True)
    )


def _cut_depth_interval(
    arithmetic: _Arithmetic,
    boundary: torch.Tensor,
    ray: tuple[_Interval, ...],
    *,
    cut_id: int,
    time: _Interval,
    near: float,
    far: float,
) -> _Interval:
    if cut_id == NEAR_CUT_ID:
        return arithmetic.point(near)
    if cut_id == FAR_CUT_ID:
        return arithmetic.point(far)
    if cut_id < 0 or cut_id >= int(boundary.shape[0]):
        raise ValueError(f"unsupported word cut id {cut_id}")
    plane = tuple(arithmetic.point(float(value)) for value in boundary[cut_id].tolist())
    origin = tuple(
        arithmetic.add(ray[axis], arithmetic.mul(time, ray[3 + axis]))
        for axis in range(3)
    )
    direction = tuple(
        arithmetic.add(ray[6 + axis], arithmetic.mul(time, ray[9 + axis]))
        for axis in range(3)
    )
    numerator = arithmetic.add(plane[4], arithmetic.mul(plane[3], time))
    denominator = arithmetic.zero
    for axis in range(3):
        numerator = arithmetic.add(numerator, arithmetic.mul(plane[axis], origin[axis]))
        denominator = arithmetic.add(denominator, arithmetic.mul(plane[axis], direction[axis]))
    return arithmetic.div(arithmetic.neg(numerator), denominator)


def _power_difference_plane(
    arithmetic: _Arithmetic,
    sites: torch.Tensor,
    owner: int,
    competitor: int,
) -> tuple[_Interval, ...]:
    left = tuple(arithmetic.point(float(value)) for value in sites[owner].tolist())
    right = tuple(arithmetic.point(float(value)) for value in sites[competitor].tolist())
    two = arithmetic.point(2)
    normal = tuple(
        arithmetic.mul(two, arithmetic.sub(right[axis], left[axis]))
        for axis in range(4)
    )
    bias = arithmetic.zero
    for axis in range(4):
        bias = arithmetic.add(bias, arithmetic.square(left[axis]))
        bias = arithmetic.sub(bias, arithmetic.square(right[axis]))
    bias = arithmetic.sub(bias, left[4])
    bias = arithmetic.add(bias, right[4])
    return (*normal, bias)


def _owner_difference(
    arithmetic: _Arithmetic,
    plane: tuple[_Interval, ...],
    ray: tuple[_Interval, ...],
    time: _Interval,
    depth: _Interval,
) -> _Interval:
    origin = tuple(
        arithmetic.add(ray[axis], arithmetic.mul(time, ray[3 + axis]))
        for axis in range(3)
    )
    direction = tuple(
        arithmetic.add(ray[6 + axis], arithmetic.mul(time, ray[9 + axis]))
        for axis in range(3)
    )
    value = arithmetic.add(plane[4], arithmetic.mul(plane[3], time))
    for axis in range(3):
        point = arithmetic.add(origin[axis], arithmetic.mul(depth, direction[axis]))
        value = arithmetic.add(value, arithmetic.mul(plane[axis], point))
    return value
