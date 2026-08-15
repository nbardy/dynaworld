"""Sparse fixed-time power-cell word discovery by a lower line envelope.

For one ray at one camera time, the 4D power distance to site ``i`` is

``||d||^2 z^2 + a_i z + b_i``.

The quadratic coefficient is common to every site, so the owner word is the
lower envelope of ``S`` lines.  Sorting slopes and constructing their monotone
lower hull discovers the exact binary64-real word in ``O(S log S)`` work and
``O(S)`` scratch.  Only adjacent owners emit power-boundary pairs; no ``S^2``
boundary table is materialized.

This is a fixed-time discovery primitive.  A continuous camera chart must
still certify the resulting word over its interval and split on topology-death
events.  :mod:`continuous_owner_identity_certificate` supplies that fail-
closed all-competitor check.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import torch
from compiled_transfer_adjoint import (
    FAR_CUT_ID,
    NEAR_CUT_ID,
    StableCellWord,
    power_boundary_parameters,
)
from continuous_owner_identity_certificate import (
    ContinuousOwnerIdentityCertificate,
    ContinuousOwnerIdentityError,
    certify_fixed_word_owner_identity,
)


@dataclass(frozen=True)
class SparsePowerRayWord:
    """One fixed-time lower envelope with only active adjacent pairs."""

    word: StableCellWord
    boundary_site_pairs: torch.Tensor
    transition_depths: tuple[Fraction, ...]
    site_count: int

    @property
    def run_count(self) -> int:
        return int(self.word.owners.numel())

    @property
    def active_boundary_count(self) -> int:
        return int(self.boundary_site_pairs.shape[0])


@dataclass(frozen=True)
class SparsePowerWordProgram:
    """Several ray words sharing one compact active-boundary table."""

    words: tuple[StableCellWord, ...]
    boundary_site_pairs: torch.Tensor
    discovery_time: float
    site_count: int
    candidate_line_count: int
    active_run_count: int

    @property
    def track_count(self) -> int:
        return len(self.words)


@dataclass(frozen=True)
class CertifiedSparseOwnerChart:
    """One continuously valid sparse owner program over a closed interval."""

    t_min: float
    t_max: float
    program: SparsePowerWordProgram
    boundary: torch.Tensor
    certificate: ContinuousOwnerIdentityCertificate


@dataclass(frozen=True)
class UnresolvedSparseOwnerInterval:
    """A topology-death or certificate interval that failed closed."""

    t_min: float
    t_max: float
    depth: int
    reason: str


@dataclass(frozen=True)
class CertifiedSparseOwnerProgram:
    """Adaptive, frame-density-independent owner-chart compilation result."""

    passed: bool
    charts: tuple[CertifiedSparseOwnerChart, ...]
    unresolved_intervals: tuple[UnresolvedSparseOwnerInterval, ...]
    leaf_count: int
    deepest_split: int
    site_count: int
    track_count: int
    candidate_line_evaluations: int
    active_boundary_rows: int
    continuous_time_coverage: bool
    owner_identity_certified: bool
    frame_sampling_used: bool = False


@dataclass(frozen=True)
class _Line:
    site_id: int
    slope: Fraction
    intercept: Fraction


@dataclass(frozen=True)
class _HullLine:
    line: _Line
    start: Fraction | None


def discover_sparse_power_word_at_time(
    sites: torch.Tensor,
    ray_coefficients: torch.Tensor,
    *,
    time: float,
    near: float,
    far: float,
) -> SparsePowerRayWord:
    """Discover the exact fixed-time owner word for one affine ray track."""

    sites_f64 = _validate_sites(sites)
    ray_f64 = torch.as_tensor(ray_coefficients, dtype=torch.float64).detach().cpu()
    if ray_f64.shape != (12,) or not bool(torch.isfinite(ray_f64).all().item()):
        raise ValueError("ray_coefficients must be a finite vector with 12 entries")
    _validate_bounds(time, near, far)
    lines = _power_lines(sites_f64, ray_f64, time=time)
    return _discover_sparse_power_word_from_lines(
        lines,
        near=Fraction.from_float(near),
        far=Fraction.from_float(far),
    )


def discover_sparse_line_envelope_word(
    slopes: Sequence[Fraction | float | int],
    intercepts: Sequence[Fraction | float | int],
    *,
    near: Fraction | float | int,
    far: Fraction | float | int,
) -> SparsePowerRayWord:
    """Discover one exact owner word from site-indexed line coefficients.

    Every candidate has cost ``common_quadratic(z) + slope[i] * z +
    intercept[i]``.  The common term does not affect ownership.  This public
    compiler primitive lets other site parameterizations reuse the same exact
    lower-envelope implementation without imitating the legacy 4D site table.
    Site ids are the coefficient row indices.
    """

    slope_values = tuple(slopes)
    intercept_values = tuple(intercepts)
    if not slope_values or len(slope_values) != len(intercept_values):
        raise ValueError("slopes and intercepts must have the same positive length")
    near_q = _as_finite_fraction(near, name="near")
    far_q = _as_finite_fraction(far, name="far")
    if far_q <= near_q:
        raise ValueError("line-envelope discovery requires near < far")
    lines = tuple(
        _Line(
            site_id,
            _as_finite_fraction(slope, name=f"slope[{site_id}]"),
            _as_finite_fraction(intercept, name=f"intercept[{site_id}]"),
        )
        for site_id, (slope, intercept) in enumerate(
            zip(slope_values, intercept_values, strict=True)
        )
    )
    return _discover_sparse_power_word_from_lines(lines, near=near_q, far=far_q)


def _discover_sparse_power_word_from_lines(
    lines: Sequence[_Line],
    *,
    near: Fraction,
    far: Fraction,
) -> SparsePowerRayWord:
    hull = _lower_envelope(lines)
    active: list[_HullLine] = []
    for index, entry in enumerate(hull):
        start = near if entry.start is None else max(near, entry.start)
        next_start = far if index + 1 == len(hull) else min(far, hull[index + 1].start)
        if start < next_start:
            active.append(_HullLine(entry.line, start))
    if not active:
        raise ValueError("lower power envelope has no positive-length segment in [near,far]")

    owners = [entry.line.site_id for entry in active]
    transitions = tuple(entry.start for entry in active[1:] if entry.start is not None)
    pairs = torch.tensor(
        [[left, right] for left, right in zip(owners, owners[1:], strict=False)],
        dtype=torch.int64,
    ).reshape(-1, 2)
    run_count = len(owners)
    left_cuts = [NEAR_CUT_ID, *range(run_count - 1)]
    right_cuts = [*range(run_count - 1), FAR_CUT_ID]
    return SparsePowerRayWord(
        word=StableCellWord(
            owners=torch.tensor(owners, dtype=torch.int64),
            left_cut_ids=torch.tensor(left_cuts, dtype=torch.int64),
            right_cut_ids=torch.tensor(right_cuts, dtype=torch.int64),
        ),
        boundary_site_pairs=pairs,
        transition_depths=transitions,
        site_count=len(lines),
    )


def discover_sparse_power_words_at_time(
    sites: torch.Tensor,
    ray_coefficients: torch.Tensor,
    *,
    time: float,
    near: float,
    far: float,
) -> SparsePowerWordProgram:
    """Discover several words and merge only their active adjacent faces."""

    sites_f64 = _validate_sites(sites)
    rays_f64 = torch.as_tensor(ray_coefficients, dtype=torch.float64).detach().cpu()
    if rays_f64.ndim != 2 or rays_f64.shape[1] != 12 or int(rays_f64.shape[0]) < 1:
        raise ValueError("ray_coefficients must have shape [P,12] with P >= 1")
    if not bool(torch.isfinite(rays_f64).all().item()):
        raise ValueError("ray_coefficients must be finite")
    _validate_bounds(time, near, far)
    discovered = tuple(
        discover_sparse_power_word_at_time(
            sites_f64,
            ray,
            time=time,
            near=near,
            far=far,
        )
        for ray in rays_f64
    )
    canonical_pairs = sorted(
        {
            tuple(sorted((int(left), int(right))))
            for result in discovered
            for left, right in result.boundary_site_pairs.tolist()
        }
    )
    pair_to_global = {pair: index for index, pair in enumerate(canonical_pairs)}
    words = []
    for result in discovered:
        local_to_global = {
            local_id: pair_to_global[tuple(sorted((int(left), int(right))))]
            for local_id, (left, right) in enumerate(result.boundary_site_pairs.tolist())
        }
        words.append(
            StableCellWord(
                owners=result.word.owners.clone(),
                left_cut_ids=_remap_cuts(result.word.left_cut_ids, local_to_global),
                right_cut_ids=_remap_cuts(result.word.right_cut_ids, local_to_global),
            )
        )
    return SparsePowerWordProgram(
        words=tuple(words),
        boundary_site_pairs=torch.tensor(canonical_pairs, dtype=torch.int64).reshape(-1, 2),
        discovery_time=float(time),
        site_count=int(sites_f64.shape[0]),
        candidate_line_count=int(rays_f64.shape[0] * sites_f64.shape[0]),
        active_run_count=sum(int(word.owners.numel()) for word in words),
    )


def compile_certified_sparse_owner_program(
    sites: torch.Tensor,
    ray_coefficients: torch.Tensor,
    *,
    t_min: float,
    t_max: float,
    near: float,
    far: float,
    ownership_tolerance: float = 1.0e-9,
    max_split_depth: int = 12,
    max_leaf_count: int = 4096,
    max_owner_work_units_per_leaf: int = 2_000_000,
    arithmetic_fraction_bits: int = 112,
) -> CertifiedSparseOwnerProgram:
    """Discover midpoint words, certify them continuously, and split failures.

    A passed result covers the full closed input interval with independently
    certified sparse words. An unresolved leaf is surfaced rather than filled
    by sampled guesses. Segment birth/death events whose zero-length stratum
    cannot satisfy the strict fixed-word prerequisite therefore remain explicit
    compiler failures until an event-stratum representation is supplied.
    """

    sites_f64 = _validate_sites(sites)
    rays_f64 = torch.as_tensor(ray_coefficients, dtype=torch.float64).detach().cpu()
    if rays_f64.ndim != 2 or rays_f64.shape[1] != 12 or int(rays_f64.shape[0]) < 1:
        raise ValueError("ray_coefficients must have shape [P,12] with P >= 1")
    if not bool(torch.isfinite(rays_f64).all().item()):
        raise ValueError("ray_coefficients must be finite")
    _validate_bounds(t_min, near, far)
    if not math.isfinite(t_max) or t_max <= t_min:
        raise ValueError("certified owner compilation requires finite t_min < t_max")
    if ownership_tolerance < 0.0 or not math.isfinite(ownership_tolerance):
        raise ValueError("ownership_tolerance must be finite and nonnegative")
    if max_split_depth < 0 or max_leaf_count < 1 or max_owner_work_units_per_leaf < 1:
        raise ValueError("owner chart split, leaf, and work budgets must be positive")

    queue: list[tuple[Fraction, Fraction, int]] = [
        (Fraction.from_float(t_min), Fraction.from_float(t_max), 0)
    ]
    charts: list[CertifiedSparseOwnerChart] = []
    unresolved: list[UnresolvedSparseOwnerInterval] = []
    deepest = 0
    candidate_lines = 0
    active_boundaries = 0
    while queue:
        lo, hi, depth = queue.pop()
        deepest = max(deepest, depth)
        midpoint = (lo + hi) / 2
        program = discover_sparse_power_words_at_time(
            sites_f64,
            rays_f64,
            time=float(midpoint),
            near=near,
            far=far,
        )
        candidate_lines += program.candidate_line_count
        boundary = power_boundary_parameters(sites_f64, program.boundary_site_pairs)
        try:
            certificate = certify_fixed_word_owner_identity(
                sites=sites_f64,
                boundary=boundary,
                ray_coefficients=rays_f64,
                words=program.words,
                t_min=float(lo),
                t_max=float(hi),
                near=near,
                far=far,
                ownership_tolerance=ownership_tolerance,
                max_split_depth=max_split_depth,
                max_leaf_count=max_leaf_count,
                max_work_units=max_owner_work_units_per_leaf,
                arithmetic_fraction_bits=arithmetic_fraction_bits,
            )
        except ContinuousOwnerIdentityError as error:
            if depth < max_split_depth and len(queue) + len(charts) + len(unresolved) + 2 <= max_leaf_count:
                queue.append((midpoint, hi, depth + 1))
                queue.append((lo, midpoint, depth + 1))
                continue
            unresolved.append(
                UnresolvedSparseOwnerInterval(
                    t_min=float(lo),
                    t_max=float(hi),
                    depth=depth,
                    reason=str(error),
                )
            )
            continue
        active_boundaries += int(program.boundary_site_pairs.shape[0])
        charts.append(
            CertifiedSparseOwnerChart(
                t_min=float(lo),
                t_max=float(hi),
                program=program,
                boundary=boundary,
                certificate=certificate,
            )
        )

    ordered_charts = tuple(sorted(charts, key=lambda chart: (chart.t_min, chart.t_max)))
    for left, right in zip(ordered_charts, ordered_charts[1:], strict=False):
        if left.t_max != right.t_min:
            unresolved.append(
                UnresolvedSparseOwnerInterval(
                    t_min=left.t_max,
                    t_max=right.t_min,
                    depth=max(left.certificate.deepest_split, right.certificate.deepest_split),
                    reason="certified owner charts do not meet at one exact seam",
                )
            )
            continue
        seam = discover_sparse_power_words_at_time(
            sites_f64,
            rays_f64,
            time=left.t_max,
            near=near,
            far=far,
        )
        candidate_lines += seam.candidate_line_count
        if not _programs_have_same_owner_words(seam, right.program):
            unresolved.append(
                UnresolvedSparseOwnerInterval(
                    t_min=left.t_max,
                    t_max=left.t_max,
                    depth=max(left.certificate.deepest_split, right.certificate.deepest_split),
                    reason=(
                        "right-continuous chart dispatch disagrees with exact fixed-time tie ownership; "
                        "an event seam policy is required"
                    ),
                )
            )
    ordered_unresolved = tuple(sorted(unresolved, key=lambda interval: (interval.t_min, interval.t_max)))
    passed = bool(ordered_charts) and not ordered_unresolved
    return CertifiedSparseOwnerProgram(
        passed=passed,
        charts=ordered_charts,
        unresolved_intervals=ordered_unresolved,
        leaf_count=len(ordered_charts) + len(ordered_unresolved),
        deepest_split=deepest,
        site_count=int(sites_f64.shape[0]),
        track_count=int(rays_f64.shape[0]),
        candidate_line_evaluations=candidate_lines,
        active_boundary_rows=active_boundaries,
        continuous_time_coverage=passed,
        owner_identity_certified=passed and all(chart.certificate.passed for chart in ordered_charts),
    )


def _validate_sites(sites: torch.Tensor) -> torch.Tensor:
    sites_f64 = torch.as_tensor(sites, dtype=torch.float64).detach().cpu()
    if sites_f64.ndim != 2 or sites_f64.shape[1] != 5 or int(sites_f64.shape[0]) < 1:
        raise ValueError("sites must have shape [S,5] with S >= 1")
    if not bool(torch.isfinite(sites_f64).all().item()):
        raise ValueError("sites must be finite")
    return sites_f64


def _validate_bounds(time: float, near: float, far: float) -> None:
    if not all(math.isfinite(value) for value in (time, near, far)) or far <= near:
        raise ValueError("discovery requires finite time and near < far")


def _as_finite_fraction(
    value: Fraction | float | int,
    *,
    name: str,
) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if not isinstance(value, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite rational, integer, or float")
    return Fraction.from_float(value)


def _power_lines(
    sites: torch.Tensor,
    ray: torch.Tensor,
    *,
    time: float,
) -> tuple[_Line, ...]:
    values = tuple(Fraction.from_float(float(value)) for value in ray.tolist())
    time_q = Fraction.from_float(time)
    origin = tuple(values[axis] + time_q * values[3 + axis] for axis in range(3))
    direction = tuple(values[6 + axis] + time_q * values[9 + axis] for axis in range(3))
    if sum(component * component for component in direction) == 0:
        raise ValueError("ray direction must be nonzero")
    lines = []
    for site_id, row in enumerate(sites.tolist()):
        site = tuple(Fraction.from_float(float(value)) for value in row)
        delta = tuple(origin[axis] - site[axis] for axis in range(3))
        slope = 2 * sum((direction[axis] * delta[axis] for axis in range(3)), Fraction(0))
        intercept = sum((value * value for value in delta), Fraction(0))
        intercept += (time_q - site[3]) * (time_q - site[3]) - site[4]
        lines.append(_Line(site_id, slope, intercept))
    return tuple(lines)


def _lower_envelope(lines: Sequence[_Line]) -> tuple[_HullLine, ...]:
    # As z increases, lower-envelope slopes decrease. For equal slopes retain
    # only the smallest intercept, breaking exact ties by site id.
    ordered = sorted(lines, key=lambda line: (-line.slope, line.intercept, line.site_id))
    unique: list[_Line] = []
    for line in ordered:
        if unique and line.slope == unique[-1].slope:
            continue
        unique.append(line)
    hull: list[_HullLine] = []
    for line in unique:
        start = None
        while hull:
            previous = hull[-1]
            denominator = previous.line.slope - line.slope
            if denominator <= 0:
                raise ArithmeticError("lower-envelope slopes are not strictly decreasing")
            start = (line.intercept - previous.line.intercept) / denominator
            if previous.start is None or start > previous.start:
                break
            hull.pop()
        hull.append(_HullLine(line, start))
    return tuple(hull)


def _remap_cuts(cuts: torch.Tensor, local_to_global: dict[int, int]) -> torch.Tensor:
    return torch.tensor(
        [local_to_global[int(value)] if int(value) >= 0 else int(value) for value in cuts.tolist()],
        dtype=torch.int64,
    )


def _programs_have_same_owner_words(
    left: SparsePowerWordProgram,
    right: SparsePowerWordProgram,
) -> bool:
    return len(left.words) == len(right.words) and all(
        torch.equal(left_word.owners, right_word.owners)
        for left_word, right_word in zip(left.words, right.words, strict=True)
    )
