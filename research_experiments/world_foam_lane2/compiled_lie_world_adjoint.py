"""End-to-end CPU reference for a compiled affine-Lie WorldFoam adjoint.

The expensive ordered word is scanned only at ``J`` Chebyshev nodes.  Total
transfer is stored in the affine-group logarithm ``[kappa,v_rgb]``.  Requested
samples stream through ``K``-frame blocks and reduce immediately to ``J`` node
cotangents; the reverse then replays the exact word at those nodes with only a
constant prefix state.

Idealized work is ``O(P J R + P F J)`` for ``P`` ray tracks, ``R`` ordered
runs, ``J`` chart nodes and ``F`` requested samples.  Reverse state contains
``O(P J + I + S + B + P K)`` scalars for ``I`` referenced track-boundary
incidences, ``S`` sites and ``B`` shared boundaries.  In particular, there is
no ``F x R`` run tape.  This is an approximate temporal closure: rank ``J``
must track physical chart complexity and a caller still needs a split/rank
error gate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from compiled_transfer_adjoint import StableCellWord, check_supplied_word_ordering
from transfer_lie_chart import (
    DTYPE,
    TemporalTransferAtlas,
    chebyshev_basis,
    chebyshev_nodes,
    check_lie_chart_cone,
    evaluate_transfer_atlas_chart,
    lie_chart_word_cotangents,
    transfer_lie_decode,
    transfer_lie_decode_vjp,
)

NEAR_CUT_ID = -1
FAR_CUT_ID = -2


@dataclass(frozen=True)
class CompiledLieWorldAtlas:
    """Fixed-word geometry plus total-transfer Lie coefficients."""

    transfer_atlas: TemporalTransferAtlas
    node_chart: torch.Tensor
    near: float
    far: float
    words: tuple[StableCellWord, ...]
    depth_coefficient_incidence: torch.Tensor
    sparse_depth_coefficients: torch.Tensor
    supplied_word_ordering_check: dict[str, float | int | bool]

    @property
    def track_count(self) -> int:
        return int(self.transfer_atlas.coefficients.shape[0])

    @property
    def node_count(self) -> int:
        return self.transfer_atlas.rank

    @property
    def structural_bytes(self) -> int:
        tensors = (
            self.transfer_atlas.node_times,
            self.transfer_atlas.fit_matrix,
            self.transfer_atlas.coefficients,
            self.node_chart,
            self.depth_coefficient_incidence,
            self.sparse_depth_coefficients,
        )
        word_tensors = tuple(
            tensor for word in self.words for tensor in (word.owners, word.left_cut_ids, word.right_cut_ids)
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors + word_tensors)


@dataclass(frozen=True)
class CompiledLieWorldVJP:
    """Loss, optional predictions, and world gradients from the compiled path."""

    loss: torch.Tensor
    predictions: torch.Tensor | None
    atlas: CompiledLieWorldAtlas
    grad_site_density: torch.Tensor
    grad_site_color: torch.Tensor
    grad_depth_coefficients: torch.Tensor
    grad_boundary: torch.Tensor
    sampled_validation_error: float | None
    sampled_tangent_validation: TangentValidationReport | None
    accounting: dict[str, int]


@dataclass(frozen=True)
class TangentValidationReport:
    """Sampled primal-tangent comparison; not a continuous certificate."""

    maximum_world_gradient_error: float
    grad_boundary_error: float
    grad_site_density_error: float
    grad_site_color_error: float
    grad_depth_coefficient_error: float
    grad_boundary_scale: float
    grad_site_density_scale: float
    grad_site_color_scale: float
    grad_depth_coefficient_scale: float
    validation_count: int
    maximum_normalized_world_gradient_error: float
    grad_boundary_normalized_error: float
    grad_site_density_normalized_error: float
    grad_site_color_normalized_error: float
    grad_depth_coefficient_normalized_error: float
    direction_count: int
    directions: tuple[TangentDirectionValidation, ...]


@dataclass(frozen=True)
class TangentDirectionValidation:
    """One deterministic output-cotangent probe lowered to world blocks."""

    split: Literal["probe", "heldout"]
    direction_id: int
    grad_boundary_error: float
    grad_site_density_error: float
    grad_site_color_error: float
    grad_depth_coefficient_error: float
    grad_boundary_scale: float
    grad_site_density_scale: float
    grad_site_color_scale: float
    grad_depth_coefficient_scale: float
    grad_boundary_normalized_error: float
    grad_site_density_normalized_error: float
    grad_site_color_normalized_error: float
    grad_depth_coefficient_normalized_error: float

    @property
    def maximum_world_gradient_error(self) -> float:
        return max(
            self.grad_boundary_error,
            self.grad_site_density_error,
            self.grad_site_color_error,
            self.grad_depth_coefficient_error,
        )

    @property
    def maximum_normalized_world_gradient_error(self) -> float:
        return max(
            self.grad_boundary_normalized_error,
            self.grad_site_density_normalized_error,
            self.grad_site_color_normalized_error,
            self.grad_depth_coefficient_normalized_error,
        )


@dataclass(frozen=True)
class AdaptiveLieWorldCompilePolicy:
    """Compile/refresh-only policy for rank selection and time-chart splits.

    The policy deliberately has no requested-frame count.  Probe directions
    choose rank and splits.  Disjoint held-out directions certify the frozen
    choice and fail closed; they are never fed back into selection.
    """

    node_count_schedule: tuple[int, ...] = (2, 4, 8, 16, 32)
    probe_validation_count: int = 65
    heldout_validation_count: int = 64
    probe_direction_count: int = 3
    heldout_direction_count: int = 3
    forward_absolute_tolerance: float = 1.0e-10
    forward_relative_tolerance: float = 1.0e-6
    tangent_absolute_tolerance: float = 1.0e-10
    tangent_relative_tolerance: float = 1.0e-5
    max_split_depth: int = 4
    max_chart_count: int = 16


@dataclass(frozen=True)
class AdaptiveValidationReport:
    """Joint primal and multi-direction tangent validation for one chart."""

    split: Literal["probe", "heldout"]
    forward_maximum_error: float
    forward_normalized_error: float
    tangent: TangentValidationReport
    passed: bool


@dataclass(frozen=True)
class AdaptiveLieChartSelection:
    """Frozen rank decision for one fixed-topology time interval."""

    t_min: float
    t_max: float
    node_count: int
    split_depth: int
    probe_validation: AdaptiveValidationReport
    heldout_validation: AdaptiveValidationReport


@dataclass(frozen=True)
class AdaptiveCompiledLieWorldAtlas:
    """Piecewise affine-Lie atlas selected independently of requested frames."""

    charts: tuple[CompiledLieWorldAtlas, ...]
    selections: tuple[AdaptiveLieChartSelection, ...]
    policy: AdaptiveLieWorldCompilePolicy
    supplied_word_ordering_check: dict[str, float | int | bool]

    @property
    def track_count(self) -> int:
        return self.charts[0].track_count

    @property
    def chart_count(self) -> int:
        return len(self.charts)

    @property
    def total_node_count(self) -> int:
        return sum(chart.node_count for chart in self.charts)

    @property
    def selection_signature(self) -> tuple[tuple[float, float, int], ...]:
        return tuple(
            (
                chart.transfer_atlas.t_min,
                chart.transfer_atlas.t_max,
                chart.node_count,
            )
            for chart in self.charts
        )


@dataclass(frozen=True)
class PiecewiseCompiledLieWorldVJP:
    """Per-step result from an already compiled adaptive atlas."""

    loss: torch.Tensor
    predictions: torch.Tensor | None
    atlas: AdaptiveCompiledLieWorldAtlas
    grad_site_density: torch.Tensor
    grad_site_color: torch.Tensor
    grad_depth_coefficients: torch.Tensor
    grad_boundary: torch.Tensor
    accounting: dict[str, int]


def compile_lie_world_atlas(
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    words: Sequence[StableCellWord],
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    t_min: float,
    t_max: float,
    near: float,
    far: float,
    node_count: int,
) -> CompiledLieWorldAtlas:
    """Exact ``J``-node word scan followed by a Lie-coordinate fit."""

    boundary_f64, rays_f64, density_f64, color_f64, words_tuple = _validate_world_inputs(
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        words=words,
        site_density=site_density,
        site_color=site_color,
    )
    boundary_f64 = boundary_f64.detach()
    rays_f64 = rays_f64.detach()
    density_f64 = density_f64.detach()
    color_f64 = color_f64.detach()
    ordering_check = check_supplied_word_ordering(
        boundary=boundary_f64,
        ray_coefficients=rays_f64,
        words=words_tuple,
        site_count=int(density_f64.numel()),
        t_min=t_min,
        t_max=t_max,
        near=near,
        far=far,
    )
    return _compile_lie_world_atlas_from_validated(
        boundary=boundary_f64,
        ray_coefficients=rays_f64,
        words=words_tuple,
        site_density=density_f64,
        site_color=color_f64,
        t_min=t_min,
        t_max=t_max,
        near=near,
        far=far,
        node_count=node_count,
        ordering_check=ordering_check,
    )


def _compile_lie_world_atlas_from_validated(
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    words: tuple[StableCellWord, ...],
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    t_min: float,
    t_max: float,
    near: float,
    far: float,
    node_count: int,
    ordering_check: dict[str, float | int | bool],
) -> CompiledLieWorldAtlas:
    """Compile node values after an explicit fixed-topology preflight."""

    nodes = chebyshev_nodes(node_count, t_min=t_min, t_max=t_max)
    node_basis = chebyshev_basis(nodes, t_min=t_min, t_max=t_max, rank=node_count)
    fit_matrix = torch.linalg.inv(node_basis)
    incidence = referenced_depth_coefficient_incidence(words)
    sparse_coefficients = sparse_factorized_depth_coefficients(
        boundary,
        ray_coefficients,
        incidence,
    )
    incidence_maps = _track_cut_incidence_maps(
        incidence,
        track_count=int(ray_coefficients.shape[0]),
    )
    node_chart = torch.stack(
        [
            torch.stack(
                [
                    _scan_word_lie_chart(
                        word=words[track_id],
                        cut_incidence=incidence_maps[track_id],
                        sparse_depth_coefficients=sparse_coefficients,
                        ray_coefficients=ray_coefficients[track_id],
                        time=time,
                        site_density=site_density,
                        site_color=site_color,
                        near=near,
                        far=far,
                    )
                    for time in nodes
                ],
                dim=0,
            )
            for track_id in range(int(ray_coefficients.shape[0]))
        ],
        dim=0,
    )
    cone = check_lie_chart_cone(node_chart)
    if not cone.passed:
        raise ValueError(
            f"compiled node transfers left the physical Lie cone; maximum violation={cone.maximum_violation:.3e}"
        )
    coefficients = torch.einsum("kn,pnc->pkc", fit_matrix, node_chart)
    return CompiledLieWorldAtlas(
        transfer_atlas=TemporalTransferAtlas(
            t_min=float(t_min),
            t_max=float(t_max),
            node_times=nodes,
            fit_matrix=fit_matrix,
            coefficients=coefficients,
            chart="lie",
        ),
        node_chart=node_chart,
        near=float(near),
        far=float(far),
        words=words,
        depth_coefficient_incidence=incidence,
        sparse_depth_coefficients=sparse_coefficients,
        supplied_word_ordering_check=ordering_check,
    )


def compile_adaptive_lie_world_atlas(
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    words: Sequence[StableCellWord],
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    t_min: float,
    t_max: float,
    near: float,
    far: float,
    policy: AdaptiveLieWorldCompilePolicy = AdaptiveLieWorldCompilePolicy(),
    track_block_size: int = 64,
    frame_block_size: int = 32,
) -> AdaptiveCompiledLieWorldAtlas:
    """Select rank and fixed-topology chart splits during compile/refresh.

    This is intentionally a separate operation from sample evaluation and the
    training-step VJP.  It performs exact word replays for deterministic probe
    directions, freezes the selected piecewise atlas, then runs a disjoint
    held-out audit.  A held-out failure raises instead of silently becoming a
    new rank-selection observation.
    """

    _validate_adaptive_policy(policy)
    if track_block_size < 1 or frame_block_size < 1:
        raise ValueError("track_block_size and frame_block_size must be positive")
    boundary_f64, rays_f64, density_f64, color_f64, words_tuple = _validate_world_inputs(
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        words=words,
        site_density=site_density,
        site_color=site_color,
    )
    boundary_f64 = boundary_f64.detach()
    rays_f64 = rays_f64.detach()
    density_f64 = density_f64.detach()
    color_f64 = color_f64.detach()

    # Splitting must not be allowed to conceal an event or ownership change.
    # The supplied word is first certified over the entire requested interval.
    global_ordering_check = check_supplied_word_ordering(
        boundary=boundary_f64,
        ray_coefficients=rays_f64,
        words=words_tuple,
        site_count=int(density_f64.numel()),
        t_min=t_min,
        t_max=t_max,
        near=near,
        far=far,
    )

    selected: list[tuple[CompiledLieWorldAtlas, int, AdaptiveValidationReport]] = []

    def select_interval(interval_min: float, interval_max: float, split_depth: int) -> None:
        failures: list[str] = []
        for node_count in policy.node_count_schedule:
            candidate = compile_lie_world_atlas(
                boundary=boundary_f64,
                ray_coefficients=rays_f64,
                words=words_tuple,
                site_density=density_f64,
                site_color=color_f64,
                t_min=interval_min,
                t_max=interval_max,
                near=near,
                far=far,
                node_count=node_count,
            )
            try:
                probe_report = _adaptive_validation_report(
                    candidate,
                    boundary=boundary_f64,
                    ray_coefficients=rays_f64,
                    site_density=density_f64,
                    site_color=color_f64,
                    split="probe",
                    validation_count=policy.probe_validation_count,
                    direction_count=policy.probe_direction_count,
                    policy=policy,
                    track_block_size=track_block_size,
                    frame_block_size=frame_block_size,
                )
            except ValueError as error:
                if "physical cone between nodes" not in str(error):
                    raise
                failures.append(f"J={node_count}: {error}")
                continue
            if probe_report.passed:
                selected.append((candidate, split_depth, probe_report))
                return
            failures.append(
                f"J={node_count}: forward={probe_report.forward_normalized_error:.3e}, "
                "tangent="
                f"{probe_report.tangent.maximum_normalized_world_gradient_error:.3e}"
            )

        if split_depth >= policy.max_split_depth:
            raise ValueError(
                "adaptive affine-Lie compile exhausted the rank schedule and split depth; "
                f"interval=[{interval_min:.17g},{interval_max:.17g}], failures={' | '.join(failures)}"
            )
        if len(selected) + 2 > policy.max_chart_count:
            raise ValueError(
                "adaptive affine-Lie compile would exceed max_chart_count after the maximum-rank failure"
            )
        midpoint = 0.5 * (interval_min + interval_max)
        if not interval_min < midpoint < interval_max:
            raise ValueError("adaptive affine-Lie compile cannot split the interval further")
        select_interval(interval_min, midpoint, split_depth + 1)
        select_interval(midpoint, interval_max, split_depth + 1)

    select_interval(float(t_min), float(t_max), 0)
    if len(selected) > policy.max_chart_count:
        raise ValueError("adaptive affine-Lie compile exceeded max_chart_count")

    selections: list[AdaptiveLieChartSelection] = []
    charts: list[CompiledLieWorldAtlas] = []
    for chart, split_depth, probe_report in selected:
        heldout_report = _adaptive_validation_report(
            chart,
            boundary=boundary_f64,
            ray_coefficients=rays_f64,
            site_density=density_f64,
            site_color=color_f64,
            split="heldout",
            validation_count=policy.heldout_validation_count,
            direction_count=policy.heldout_direction_count,
            policy=policy,
            track_block_size=track_block_size,
            frame_block_size=frame_block_size,
        )
        if not heldout_report.passed:
            raise ValueError(
                "adaptive affine-Lie held-out audit failed after rank/chart selection; "
                f"interval=[{chart.transfer_atlas.t_min:.17g},{chart.transfer_atlas.t_max:.17g}], "
                f"J={chart.node_count}, forward={heldout_report.forward_normalized_error:.3e}, "
                "tangent="
                f"{heldout_report.tangent.maximum_normalized_world_gradient_error:.3e}; "
                "refresh with a predeclared stronger policy instead of tuning on held-out directions"
            )
        charts.append(chart)
        selections.append(
            AdaptiveLieChartSelection(
                t_min=chart.transfer_atlas.t_min,
                t_max=chart.transfer_atlas.t_max,
                node_count=chart.node_count,
                split_depth=split_depth,
                probe_validation=probe_report,
                heldout_validation=heldout_report,
            )
        )
    return AdaptiveCompiledLieWorldAtlas(
        charts=tuple(charts),
        selections=tuple(selections),
        policy=policy,
        supplied_word_ordering_check=global_ordering_check,
    )


def refresh_fixed_topology_lie_world_atlas(
    template: AdaptiveCompiledLieWorldAtlas,
    *,
    assume_fixed_topology: Literal[True],
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
) -> AdaptiveCompiledLieWorldAtlas:
    """Refresh selected node values without validation or rank reselection.

    This is the intended per-step forward compile.  It assumes the caller's
    event/topology gate has established that the stored words remain valid.
    Node scans still reject non-positive sampled segments and unphysical node
    values, but this function does not pretend that those samples certify the
    continuous interval.  Call :func:`compile_adaptive_lie_world_atlas` again
    when topology or the validation policy must be refreshed.
    """

    if assume_fixed_topology is not True:
        raise ValueError("refresh requires assume_fixed_topology=True; otherwise recompile and validate")
    if not template.charts:
        raise ValueError("adaptive atlas must contain at least one chart")
    first = template.charts[0]
    boundary_f64, rays_f64, density_f64, color_f64, words_tuple = _validate_world_inputs(
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        words=first.words,
        site_density=site_density,
        site_color=site_color,
    )
    boundary_f64 = boundary_f64.detach()
    rays_f64 = rays_f64.detach()
    density_f64 = density_f64.detach()
    color_f64 = color_f64.detach()
    charts = tuple(
        _compile_lie_world_atlas_from_validated(
            boundary=boundary_f64,
            ray_coefficients=rays_f64,
            words=words_tuple,
            site_density=density_f64,
            site_color=color_f64,
            t_min=chart.transfer_atlas.t_min,
            t_max=chart.transfer_atlas.t_max,
            near=chart.near,
            far=chart.far,
            node_count=chart.node_count,
            ordering_check=template.supplied_word_ordering_check,
        )
        for chart in template.charts
    )
    return AdaptiveCompiledLieWorldAtlas(
        charts=charts,
        selections=template.selections,
        policy=template.policy,
        supplied_word_ordering_check=template.supplied_word_ordering_check,
    )


def piecewise_compiled_lie_world_mse_vjp(
    atlas: AdaptiveCompiledLieWorldAtlas,
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    times: torch.Tensor,
    targets: torch.Tensor,
    background: torch.Tensor | Sequence[float],
    frame_block_size: int,
    track_block_size: int = 64,
    return_predictions: bool = False,
) -> PiecewiseCompiledLieWorldVJP:
    """Evaluate a frozen adaptive atlas without hidden validation or refresh.

    The caller is responsible for refreshing ``atlas`` when its world snapshot
    changes.  This warm path performs only cheap sample reduction and the
    selected node-word adjoints; it never runs exact validation probes or
    changes ranks/chart boundaries.
    """

    if not atlas.charts:
        raise ValueError("adaptive atlas must contain at least one chart")
    if frame_block_size < 1 or track_block_size < 1:
        raise ValueError("frame_block_size and track_block_size must be positive")
    first = atlas.charts[0]
    boundary_f64, rays_f64, density_f64, color_f64, words_tuple = _validate_world_inputs(
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        words=first.words,
        site_density=site_density,
        site_color=site_color,
    )
    boundary_f64 = boundary_f64.detach()
    rays_f64 = rays_f64.detach()
    density_f64 = density_f64.detach()
    color_f64 = color_f64.detach()
    times_f64 = torch.as_tensor(times, dtype=DTYPE).reshape(-1).detach()
    if times_f64.numel() == 0 or not bool(torch.isfinite(times_f64).all().item()):
        raise ValueError("times must be non-empty and finite")
    targets_f64 = torch.as_tensor(targets, dtype=DTYPE).detach()
    expected_target_shape = (first.track_count, int(times_f64.numel()), 3)
    if tuple(targets_f64.shape) != expected_target_shape:
        raise ValueError(f"targets must have shape {expected_target_shape}")
    if not bool(torch.isfinite(targets_f64).all().item()):
        raise ValueError("targets must be finite")
    background_f64 = torch.as_tensor(background, dtype=DTYPE).reshape(3).detach()
    if not bool(torch.isfinite(background_f64).all().item()):
        raise ValueError("background must be finite")
    if (
        float(times_f64.min().item()) < first.transfer_atlas.t_min
        or float(times_f64.max().item()) > atlas.charts[-1].transfer_atlas.t_max
    ):
        raise ValueError("requested times leave the adaptive atlas interval")

    for previous, chart in zip(atlas.charts[:-1], atlas.charts[1:], strict=True):
        if previous.transfer_atlas.t_max != chart.transfer_atlas.t_min:
            raise ValueError("adaptive atlas charts must be ordered and exactly contiguous")
        if not _words_have_same_topology(chart.words, words_tuple):
            raise ValueError("adaptive atlas charts must share one fixed cell word per track")
        if not torch.equal(chart.depth_coefficient_incidence, first.depth_coefficient_incidence):
            raise ValueError("adaptive atlas charts must share sparse incidence ordering")

    loss = torch.zeros((), dtype=DTYPE)
    predictions = torch.empty_like(targets_f64) if return_predictions else None
    grad_density = torch.zeros_like(density_f64)
    grad_color = torch.zeros_like(color_f64)
    grad_depth = torch.zeros_like(first.sparse_depth_coefficients)
    grad_boundary = torch.zeros_like(boundary_f64)
    assigned = torch.zeros(int(times_f64.numel()), dtype=torch.bool)
    sample_basis_interactions = 0
    peak_reverse_state_bytes = 0
    normalization = float(targets_f64.numel())
    for chart_id, chart in enumerate(atlas.charts):
        is_last = chart_id == len(atlas.charts) - 1
        mask = times_f64 >= chart.transfer_atlas.t_min
        mask &= (
            times_f64 <= chart.transfer_atlas.t_max
            if is_last
            else times_f64 < chart.transfer_atlas.t_max
        )
        indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if indices.numel() == 0:
            continue
        if bool(assigned[indices].any().item()):
            raise ValueError("adaptive atlas assigned a requested sample to multiple charts")
        assigned[indices] = True
        chart_result = _compiled_lie_world_atlas_mse_vjp_core(
            atlas=chart,
            boundary=boundary_f64,
            ray_coefficients=rays_f64,
            site_density=density_f64,
            site_color=color_f64,
            times=times_f64[indices],
            targets=targets_f64[:, indices],
            background=background_f64,
            frame_block_size=frame_block_size,
            track_block_size=track_block_size,
            normalization=normalization,
            return_predictions=return_predictions,
        )
        chart_loss, chart_predictions, chart_density, chart_color, chart_depth, chart_boundary, chart_peak = chart_result
        loss += chart_loss
        grad_density += chart_density
        grad_color += chart_color
        grad_depth += chart_depth
        grad_boundary += chart_boundary
        if predictions is not None and chart_predictions is not None:
            predictions[:, indices] = chart_predictions
        sample_basis_interactions += first.track_count * int(indices.numel()) * chart.node_count
        peak_reverse_state_bytes = max(peak_reverse_state_bytes, chart_peak)
    if not bool(assigned.all().item()):
        raise ValueError("adaptive atlas did not cover every requested sample")

    run_count = sum(int(word.owners.numel()) for word in words_tuple)
    accounting = {
        "track_count": first.track_count,
        "frame_count": int(times_f64.numel()),
        "chart_count": atlas.chart_count,
        "total_node_count": atlas.total_node_count,
        "run_count": run_count,
        "referenced_track_boundaries": int(first.depth_coefficient_incidence.shape[0]),
        "refresh_world_forward_run_interactions": atlas.total_node_count * run_count,
        "step_world_reverse_run_interactions": atlas.total_node_count * run_count,
        "sample_basis_interactions": sample_basis_interactions,
        "frame_run_reverse_state_elements": 0,
        "per_sample_run_tape_bytes": 0,
        "peak_reverse_state_bytes_excluding_targets_and_predictions": peak_reverse_state_bytes,
        "sampled_validation_count": 0,
        "validation_exact_run_interactions": 0,
    }
    return PiecewiseCompiledLieWorldVJP(
        loss=loss,
        predictions=predictions,
        atlas=atlas,
        grad_site_density=grad_density,
        grad_site_color=grad_color,
        grad_depth_coefficients=grad_depth,
        grad_boundary=grad_boundary,
        accounting=accounting,
    )


def compiled_lie_world_mse_vjp(
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    words: Sequence[StableCellWord],
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    times: torch.Tensor,
    targets: torch.Tensor,
    background: torch.Tensor | Sequence[float],
    t_min: float,
    t_max: float,
    near: float,
    far: float,
    node_count: int,
    frame_block_size: int,
    track_block_size: int = 64,
    validation_count: int = 0,
    sampled_forward_tolerance: float | None = None,
    sampled_tangent_tolerance: float | None = None,
    return_predictions: bool = False,
) -> CompiledLieWorldVJP:
    """Run the compiled forward and its streamed, prefix-only world VJP."""

    if frame_block_size < 1 or track_block_size < 1:
        raise ValueError("frame_block_size and track_block_size must be positive")
    boundary_f64, rays_f64, density_f64, color_f64, words_tuple = _validate_world_inputs(
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        words=words,
        site_density=site_density,
        site_color=site_color,
    )
    boundary_f64 = boundary_f64.detach()
    rays_f64 = rays_f64.detach()
    density_f64 = density_f64.detach()
    color_f64 = color_f64.detach()
    times_f64 = torch.as_tensor(times, dtype=DTYPE).reshape(-1).detach()
    if times_f64.numel() == 0 or not bool(torch.isfinite(times_f64).all().item()):
        raise ValueError("times must be non-empty and finite")
    if float(times_f64.min().item()) < t_min or float(times_f64.max().item()) > t_max:
        raise ValueError("requested times leave the compiled chart interval")
    targets_f64 = torch.as_tensor(targets, dtype=DTYPE).detach()
    expected_target_shape = (int(rays_f64.shape[0]), int(times_f64.numel()), 3)
    if tuple(targets_f64.shape) != expected_target_shape:
        raise ValueError(f"targets must have shape {expected_target_shape}")
    if not bool(torch.isfinite(targets_f64).all().item()):
        raise ValueError("targets must be finite")
    background_f64 = torch.as_tensor(background, dtype=DTYPE).reshape(3).detach()
    if not bool(torch.isfinite(background_f64).all().item()):
        raise ValueError("background must be finite")

    atlas = compile_lie_world_atlas(
        boundary=boundary_f64,
        ray_coefficients=rays_f64,
        words=words_tuple,
        site_density=density_f64,
        site_color=color_f64,
        t_min=t_min,
        t_max=t_max,
        near=near,
        far=far,
        node_count=node_count,
    )
    if validation_count < 0 or validation_count == 1:
        raise ValueError("validation_count must be zero (disabled) or at least 2")
    if validation_count == 0 and (sampled_forward_tolerance is not None or sampled_tangent_tolerance is not None):
        raise ValueError("sampled rank tolerances require validation_count >= 2")
    validation_error = (
        sampled_lie_world_transfer_error(
            atlas,
            boundary=boundary_f64,
            ray_coefficients=rays_f64,
            site_density=density_f64,
            site_color=color_f64,
            validation_count=validation_count,
            track_block_size=track_block_size,
            frame_block_size=frame_block_size,
        )
        if validation_count >= 2
        else None
    )
    if (
        sampled_forward_tolerance is not None
        and validation_error is not None
        and validation_error > sampled_forward_tolerance
    ):
        raise ValueError(
            "sampled (non-continuous) forward rank gate failed: "
            f"error={validation_error:.3e} tolerance={sampled_forward_tolerance:.3e}; "
            "split the time chart or raise node_count"
        )
    predictions = torch.empty_like(targets_f64) if return_predictions else None
    grad_node_chart = torch.zeros(
        (atlas.track_count, node_count, 4),
        dtype=DTYPE,
    )
    loss = torch.zeros((), dtype=DTYPE)
    normalization = float(targets_f64.numel())
    for track_start in range(0, atlas.track_count, track_block_size):
        track_end = min(track_start + track_block_size, atlas.track_count)
        coefficients = atlas.transfer_atlas.coefficients[track_start:track_end]
        for frame_start in range(0, int(times_f64.numel()), frame_block_size):
            frame_end = min(frame_start + frame_block_size, int(times_f64.numel()))
            time_block = times_f64[frame_start:frame_end]
            basis = chebyshev_basis(
                time_block,
                t_min=t_min,
                t_max=t_max,
                rank=node_count,
            )
            chart_block = torch.einsum("fk,pkc->pfc", basis, coefficients)
            _require_interpolated_chart_cone(chart_block)
            transfer_block = transfer_lie_decode(chart_block)
            prediction_block = transfer_block[..., 1:] + transfer_block[..., :1] * background_f64
            residual = prediction_block - targets_f64[track_start:track_end, frame_start:frame_end]
            loss += residual.square().sum() / normalization
            if predictions is not None:
                predictions[track_start:track_end, frame_start:frame_end] = prediction_block
            grad_prediction = 2.0 * residual / normalization
            grad_transfer = torch.cat(
                (
                    (grad_prediction * background_f64).sum(dim=-1, keepdim=True),
                    grad_prediction,
                ),
                dim=-1,
            )
            grad_chart = transfer_lie_decode_vjp(chart_block, grad_transfer)
            node_interpolation = basis @ atlas.transfer_atlas.fit_matrix
            grad_node_chart[track_start:track_end] += torch.einsum(
                "fn,pfc->pnc",
                node_interpolation,
                grad_chart,
            )

    grad_density, grad_color, grad_sparse_depth, grad_boundary = _world_vjp_from_node_chart(
        atlas=atlas,
        boundary=boundary_f64,
        ray_coefficients=rays_f64,
        site_density=density_f64,
        site_color=color_f64,
        grad_node_chart=grad_node_chart,
    )
    tangent_validation = (
        sampled_lie_world_tangent_error(
            atlas,
            boundary=boundary_f64,
            ray_coefficients=rays_f64,
            site_density=density_f64,
            site_color=color_f64,
            validation_count=validation_count,
            track_block_size=track_block_size,
            frame_block_size=frame_block_size,
        )
        if validation_count >= 2
        else None
    )
    if (
        sampled_tangent_tolerance is not None
        and tangent_validation is not None
        and tangent_validation.maximum_world_gradient_error > sampled_tangent_tolerance
    ):
        raise ValueError(
            "sampled (non-continuous) tangent/VJP rank gate failed: "
            f"error={tangent_validation.maximum_world_gradient_error:.3e} "
            f"tolerance={sampled_tangent_tolerance:.3e}; split the time chart or raise node_count"
        )
    run_count = sum(int(word.owners.numel()) for word in words_tuple)
    scalar_bytes = torch.tensor([], dtype=DTYPE).element_size()
    block_tracks = min(track_block_size, atlas.track_count)
    block_frames = min(frame_block_size, int(times_f64.numel()))
    reverse_state_scalars = (
        grad_node_chart.numel()
        + grad_sparse_depth.numel()
        + grad_density.numel()
        + grad_color.numel()
        + grad_boundary.numel()
        + block_tracks * block_frames * (4 + 4 + 3 + 3 + 4 + 4)
        + 2 * block_frames * node_count
    )
    accounting = {
        "track_count": atlas.track_count,
        "frame_count": int(times_f64.numel()),
        "node_count": node_count,
        "run_count": run_count,
        "referenced_track_boundaries": int(atlas.depth_coefficient_incidence.shape[0]),
        "world_forward_run_interactions": node_count * run_count,
        "world_reverse_run_interactions": node_count * run_count,
        "sample_basis_interactions": atlas.track_count * int(times_f64.numel()) * node_count,
        "frame_run_reverse_state_elements": 0,
        "per_sample_run_tape_bytes": 0,
        "reverse_state_bytes_excluding_targets_and_predictions": reverse_state_scalars * scalar_bytes,
        "atlas_structural_bytes": atlas.structural_bytes,
        "sampled_validation_count": validation_count,
        "validation_exact_forward_run_interactions": validation_count * run_count,
        "validation_exact_tangent_run_interactions": 3 * validation_count * run_count,
        "validation_compiled_tangent_reverse_run_interactions": (
            node_count * run_count if validation_count >= 2 else 0
        ),
    }
    return CompiledLieWorldVJP(
        loss=loss,
        predictions=predictions,
        atlas=atlas,
        grad_site_density=grad_density,
        grad_site_color=grad_color,
        grad_depth_coefficients=grad_sparse_depth,
        grad_boundary=grad_boundary,
        sampled_validation_error=validation_error,
        sampled_tangent_validation=tangent_validation,
        accounting=accounting,
    )


def sampled_lie_world_transfer_error(
    atlas: CompiledLieWorldAtlas,
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    validation_count: int = 257,
    track_block_size: int = 64,
    frame_block_size: int = 32,
) -> float:
    """Blockwise sampled max transfer error; not a continuous certificate."""

    if validation_count < 2 or track_block_size < 1 or frame_block_size < 1:
        raise ValueError("validation_count must be >=2 and block sizes must be positive")
    times = _deterministic_validation_times(
        atlas.transfer_atlas.t_min,
        atlas.transfer_atlas.t_max,
        validation_count,
        split="probe",
    )
    maximum_error, _, _ = _sampled_lie_world_transfer_report(
        atlas,
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        site_density=site_density,
        site_color=site_color,
        validation_times=times,
        absolute_tolerance=0.0,
        relative_tolerance=1.0,
        track_block_size=track_block_size,
        frame_block_size=frame_block_size,
    )
    return maximum_error


def _sampled_lie_world_transfer_report(
    atlas: CompiledLieWorldAtlas,
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    validation_times: torch.Tensor,
    absolute_tolerance: float,
    relative_tolerance: float,
    track_block_size: int,
    frame_block_size: int,
) -> tuple[float, float, float]:
    """Return raw error, scale-normalized gate ratio, and reference scale."""

    if track_block_size < 1 or frame_block_size < 1:
        raise ValueError("block sizes must be positive")
    _validate_error_tolerances(absolute_tolerance, relative_tolerance)
    boundary_f64, rays_f64, density_f64, color_f64, _ = _validate_world_inputs(
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        words=atlas.words,
        site_density=site_density,
        site_color=site_color,
    )
    boundary_f64 = boundary_f64.detach()
    rays_f64 = rays_f64.detach()
    density_f64 = density_f64.detach()
    color_f64 = color_f64.detach()
    times = torch.as_tensor(validation_times, dtype=DTYPE).reshape(-1).detach()
    if times.numel() < 2 or not bool(torch.isfinite(times).all().item()):
        raise ValueError("validation_times must contain at least two finite values")
    if (
        float(times.min().item()) < atlas.transfer_atlas.t_min
        or float(times.max().item()) > atlas.transfer_atlas.t_max
    ):
        raise ValueError("validation_times leave the atlas interval")
    incidence_maps = _track_cut_incidence_maps(
        atlas.depth_coefficient_incidence,
        track_count=atlas.track_count,
    )
    validation_sparse_depth = sparse_factorized_depth_coefficients(
        boundary_f64,
        rays_f64,
        atlas.depth_coefficient_incidence,
    )
    max_error = 0.0
    reference_scale = 0.0
    compiled_scale = 0.0
    for track_start in range(0, atlas.track_count, track_block_size):
        track_end = min(track_start + track_block_size, atlas.track_count)
        for frame_start in range(0, int(times.numel()), frame_block_size):
            frame_end = min(frame_start + frame_block_size, int(times.numel()))
            time_block = times[frame_start:frame_end]
            compiled = _evaluate_transfer_block(
                atlas.transfer_atlas,
                time_block,
                track_start=track_start,
                track_end=track_end,
            )
            exact_rows = []
            for track_id in range(track_start, track_end):
                exact_rows.append(
                    torch.stack(
                        [
                            _scan_word_transfer(
                                word=atlas.words[track_id],
                                cut_incidence=incidence_maps[track_id],
                                sparse_depth_coefficients=validation_sparse_depth,
                                ray_coefficients=rays_f64[track_id],
                                time=time,
                                site_density=density_f64,
                                site_color=color_f64,
                                near=atlas.near,
                                far=atlas.far,
                            )
                            for time in time_block
                        ],
                        dim=0,
                    )
                )
            exact = torch.stack(exact_rows, dim=0)
            max_error = max(max_error, float((compiled - exact).abs().max().item()))
            reference_scale = max(reference_scale, _max_abs(exact))
            compiled_scale = max(compiled_scale, _max_abs(compiled))
    scale = max(reference_scale, compiled_scale)
    normalized_error = _normalized_block_error(
        max_error,
        scale,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    return max_error, normalized_error, scale


def sampled_lie_world_tangent_error(
    atlas: CompiledLieWorldAtlas,
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    validation_count: int = 257,
    track_block_size: int = 64,
    frame_block_size: int = 32,
    direction_ids: Sequence[int] = (0,),
    direction_split: Literal["probe", "heldout"] = "probe",
    validation_times: torch.Tensor | None = None,
    absolute_tolerance: float = 1.0e-12,
    relative_tolerance: float = 1.0,
) -> TangentValidationReport:
    """Compare deterministic sampled transfer VJPs to exact word replay.

    This catches tangent-rank failures that a primal transfer error cannot see.
    Errors are reported both absolutely and per world-parameter block after
    scaling by ``atol + rtol * max(||exact||_inf, ||compiled||_inf)``.  The
    result remains a finite directional probe, not a continuous Jacobian bound.
    """

    if validation_count < 2 or track_block_size < 1 or frame_block_size < 1:
        raise ValueError("validation_count must be >=2 and block sizes must be positive")
    if direction_split not in ("probe", "heldout"):
        raise ValueError("direction_split must be 'probe' or 'heldout'")
    direction_ids_tuple = tuple(int(direction_id) for direction_id in direction_ids)
    if not direction_ids_tuple or len(set(direction_ids_tuple)) != len(direction_ids_tuple):
        raise ValueError("direction_ids must be non-empty and unique")
    if any(direction_id < 0 for direction_id in direction_ids_tuple):
        raise ValueError("direction_ids must be non-negative")
    _validate_error_tolerances(absolute_tolerance, relative_tolerance)
    boundary_f64, rays_f64, density_f64, color_f64, _ = _validate_world_inputs(
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        words=atlas.words,
        site_density=site_density,
        site_color=site_color,
    )
    boundary_f64 = boundary_f64.detach()
    rays_f64 = rays_f64.detach()
    density_f64 = density_f64.detach()
    color_f64 = color_f64.detach()
    times = (
        _deterministic_validation_times(
            atlas.transfer_atlas.t_min,
            atlas.transfer_atlas.t_max,
            validation_count,
            split=direction_split,
        )
        if validation_times is None
        else torch.as_tensor(validation_times, dtype=DTYPE).reshape(-1).detach()
    )
    if int(times.numel()) != validation_count:
        raise ValueError("validation_times length must equal validation_count")
    if not bool(torch.isfinite(times).all().item()):
        raise ValueError("validation_times must be finite")
    if (
        float(times.min().item()) < atlas.transfer_atlas.t_min
        or float(times.max().item()) > atlas.transfer_atlas.t_max
    ):
        raise ValueError("validation_times leave the atlas interval")
    incidence_maps = _track_cut_incidence_maps(
        atlas.depth_coefficient_incidence,
        track_count=atlas.track_count,
    )
    validation_sparse_depth = sparse_factorized_depth_coefficients(
        boundary_f64,
        rays_f64,
        atlas.depth_coefficient_incidence,
    )
    direction_reports: list[TangentDirectionValidation] = []
    for direction_id in direction_ids_tuple:
        grad_node_chart = torch.zeros((atlas.track_count, atlas.node_count, 4), dtype=DTYPE)
        exact_density = torch.zeros_like(density_f64)
        exact_color = torch.zeros_like(color_f64)
        exact_depth = torch.zeros_like(atlas.sparse_depth_coefficients)
        normalization = float(atlas.track_count * validation_count)
        for track_start in range(0, atlas.track_count, track_block_size):
            track_end = min(track_start + track_block_size, atlas.track_count)
            coefficients = atlas.transfer_atlas.coefficients[track_start:track_end]
            track_ids = torch.arange(track_start, track_end, dtype=DTYPE).reshape(-1, 1)
            for frame_start in range(0, validation_count, frame_block_size):
                frame_end = min(frame_start + frame_block_size, validation_count)
                time_block = times[frame_start:frame_end]
                basis = chebyshev_basis(
                    time_block,
                    t_min=atlas.transfer_atlas.t_min,
                    t_max=atlas.transfer_atlas.t_max,
                    rank=atlas.node_count,
                )
                compiled_chart = torch.einsum("fk,pkc->pfc", basis, coefficients)
                _require_interpolated_chart_cone(compiled_chart)
                cotangent = (
                    _tangent_validation_cotangent(
                        track_ids,
                        time_block.reshape(1, -1),
                        direction_id=direction_id,
                        split=direction_split,
                    )
                    / normalization
                )
                compiled_chart_grad = transfer_lie_decode_vjp(compiled_chart, cotangent)
                node_interpolation = basis @ atlas.transfer_atlas.fit_matrix
                grad_node_chart[track_start:track_end] += torch.einsum(
                    "fn,pfc->pnc",
                    node_interpolation,
                    compiled_chart_grad,
                )
                for local_track_id, track_id in enumerate(range(track_start, track_end)):
                    for local_frame_id, time in enumerate(time_block):
                        exact_chart = _scan_word_lie_chart(
                            word=atlas.words[track_id],
                            cut_incidence=incidence_maps[track_id],
                            sparse_depth_coefficients=validation_sparse_depth,
                            ray_coefficients=rays_f64[track_id],
                            time=time,
                            site_density=density_f64,
                            site_color=color_f64,
                            near=atlas.near,
                            far=atlas.far,
                        )
                        exact_chart_grad = transfer_lie_decode_vjp(
                            exact_chart,
                            cotangent[local_track_id, local_frame_id],
                        )
                        density_grad, color_grad, depth_grad = _word_lie_chart_vjp(
                            word=atlas.words[track_id],
                            cut_incidence=incidence_maps[track_id],
                            sparse_depth_coefficients=validation_sparse_depth,
                            ray_coefficients=rays_f64[track_id],
                            time=time,
                            site_density=density_f64,
                            site_color=color_f64,
                            total_chart=exact_chart,
                            grad_chart=exact_chart_grad,
                            near=atlas.near,
                            far=atlas.far,
                        )
                        exact_density += density_grad
                        exact_color += color_grad
                        exact_depth += depth_grad
        compiled_density, compiled_color, compiled_depth, compiled_boundary = _world_vjp_from_node_chart(
            atlas=atlas,
            boundary=boundary_f64,
            ray_coefficients=rays_f64,
            site_density=density_f64,
            site_color=color_f64,
            grad_node_chart=grad_node_chart,
        )
        exact_boundary = sparse_factorized_depth_coefficients_boundary_vjp(
            boundary_f64,
            rays_f64,
            atlas.depth_coefficient_incidence,
            exact_depth,
        )
        compiled_blocks = (compiled_boundary, compiled_density, compiled_color, compiled_depth)
        exact_blocks = (exact_boundary, exact_density, exact_color, exact_depth)
        errors = tuple(
            _max_abs(compiled - exact)
            for compiled, exact in zip(compiled_blocks, exact_blocks, strict=True)
        )
        block_scales = tuple(
            max(_max_abs(compiled), _max_abs(exact))
            for compiled, exact in zip(compiled_blocks, exact_blocks, strict=True)
        )
        normalized_errors = tuple(
            _normalized_block_error(
                error,
                scale,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
            for error, scale in zip(errors, block_scales, strict=True)
        )
        direction_reports.append(
            TangentDirectionValidation(
                split=direction_split,
                direction_id=direction_id,
                grad_boundary_error=errors[0],
                grad_site_density_error=errors[1],
                grad_site_color_error=errors[2],
                grad_depth_coefficient_error=errors[3],
                grad_boundary_scale=block_scales[0],
                grad_site_density_scale=block_scales[1],
                grad_site_color_scale=block_scales[2],
                grad_depth_coefficient_scale=block_scales[3],
                grad_boundary_normalized_error=normalized_errors[0],
                grad_site_density_normalized_error=normalized_errors[1],
                grad_site_color_normalized_error=normalized_errors[2],
                grad_depth_coefficient_normalized_error=normalized_errors[3],
            )
        )
    raw_by_block = tuple(
        max(getattr(direction, field) for direction in direction_reports)
        for field in (
            "grad_boundary_error",
            "grad_site_density_error",
            "grad_site_color_error",
            "grad_depth_coefficient_error",
        )
    )
    normalized_by_block = tuple(
        max(getattr(direction, field) for direction in direction_reports)
        for field in (
            "grad_boundary_normalized_error",
            "grad_site_density_normalized_error",
            "grad_site_color_normalized_error",
            "grad_depth_coefficient_normalized_error",
        )
    )
    scales_by_block = tuple(
        max(getattr(direction, field) for direction in direction_reports)
        for field in (
            "grad_boundary_scale",
            "grad_site_density_scale",
            "grad_site_color_scale",
            "grad_depth_coefficient_scale",
        )
    )
    return TangentValidationReport(
        maximum_world_gradient_error=max(raw_by_block),
        grad_boundary_error=raw_by_block[0],
        grad_site_density_error=raw_by_block[1],
        grad_site_color_error=raw_by_block[2],
        grad_depth_coefficient_error=raw_by_block[3],
        grad_boundary_scale=scales_by_block[0],
        grad_site_density_scale=scales_by_block[1],
        grad_site_color_scale=scales_by_block[2],
        grad_depth_coefficient_scale=scales_by_block[3],
        validation_count=validation_count,
        maximum_normalized_world_gradient_error=max(normalized_by_block),
        grad_boundary_normalized_error=normalized_by_block[0],
        grad_site_density_normalized_error=normalized_by_block[1],
        grad_site_color_normalized_error=normalized_by_block[2],
        grad_depth_coefficient_normalized_error=normalized_by_block[3],
        direction_count=len(direction_reports),
        directions=tuple(direction_reports),
    )


def referenced_depth_coefficient_incidence(words: Sequence[StableCellWord]) -> torch.Tensor:
    """Return unique sorted ``[track_id,boundary_id]`` rows actually used."""

    rows: list[tuple[int, int]] = []
    for track_id, word in enumerate(words):
        cut_ids = sorted(
            {int(cut_id) for cut_id in torch.cat((word.left_cut_ids, word.right_cut_ids)).tolist() if int(cut_id) >= 0}
        )
        rows.extend((track_id, cut_id) for cut_id in cut_ids)
    if not rows:
        return torch.empty((0, 2), dtype=torch.int64)
    return torch.tensor(rows, dtype=torch.int64)


def sparse_factorized_depth_coefficients(
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    incidence: torch.Tensor,
) -> torch.Tensor:
    """Lower only referenced track-boundary pairs to Mobius ``[A,B,C,D]``."""

    boundary_f64 = _require_matrix("boundary", boundary, columns=5)
    rays_f64 = _require_matrix("ray_coefficients", ray_coefficients, columns=12)
    incidence_i64 = _require_incidence(
        incidence,
        track_count=int(rays_f64.shape[0]),
        boundary_count=int(boundary_f64.shape[0]),
    )
    if incidence_i64.shape[0] == 0:
        return torch.empty((0, 4), dtype=DTYPE)
    track_ids = incidence_i64[:, 0]
    boundary_ids = incidence_i64[:, 1]
    planes = boundary_f64[boundary_ids]
    rays = rays_f64[track_ids]
    normal = planes[:, :3]
    return torch.stack(
        (
            -(rays[:, 0:3] * normal).sum(dim=1) - planes[:, 4],
            -(rays[:, 3:6] * normal).sum(dim=1) - planes[:, 3],
            (rays[:, 6:9] * normal).sum(dim=1),
            (rays[:, 9:12] * normal).sum(dim=1),
        ),
        dim=1,
    )


def sparse_factorized_depth_coefficients_boundary_vjp(
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    incidence: torch.Tensor,
    grad_depth_coefficients: torch.Tensor,
) -> torch.Tensor:
    """Scatter each sparse incidence adjoint once into shared boundaries."""

    boundary_f64 = _require_matrix("boundary", boundary, columns=5)
    rays_f64 = _require_matrix("ray_coefficients", ray_coefficients, columns=12)
    incidence_i64 = _require_incidence(
        incidence,
        track_count=int(rays_f64.shape[0]),
        boundary_count=int(boundary_f64.shape[0]),
    )
    grad = _require_matrix("grad_depth_coefficients", grad_depth_coefficients, columns=4)
    if grad.shape[0] != incidence_i64.shape[0]:
        raise ValueError("grad_depth_coefficients must have one row per incidence")
    if incidence_i64.shape[0] == 0:
        return torch.zeros_like(boundary_f64)
    track_ids = incidence_i64[:, 0]
    boundary_ids = incidence_i64[:, 1]
    rays = rays_f64[track_ids]
    grad_a, grad_b, grad_c, grad_d = grad.unbind(dim=1)
    grad_normal = (
        -grad_a.unsqueeze(1) * rays[:, 0:3]
        - grad_b.unsqueeze(1) * rays[:, 3:6]
        + grad_c.unsqueeze(1) * rays[:, 6:9]
        + grad_d.unsqueeze(1) * rays[:, 9:12]
    )
    incidence_grad = torch.cat(
        (
            grad_normal,
            -grad_b.unsqueeze(1),
            -grad_a.unsqueeze(1),
        ),
        dim=1,
    )
    grad_boundary = torch.zeros_like(boundary_f64)
    grad_boundary.index_add_(0, boundary_ids, incidence_grad)
    return grad_boundary


def _compiled_lie_world_atlas_mse_vjp_core(
    *,
    atlas: CompiledLieWorldAtlas,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    times: torch.Tensor,
    targets: torch.Tensor,
    background: torch.Tensor,
    frame_block_size: int,
    track_block_size: int,
    normalization: float,
    return_predictions: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    """Warm single-chart sample reduction and node replay, without checks."""

    predictions = torch.empty_like(targets) if return_predictions else None
    grad_node_chart = torch.zeros(
        (atlas.track_count, atlas.node_count, 4),
        dtype=DTYPE,
    )
    loss = torch.zeros((), dtype=DTYPE)
    for track_start in range(0, atlas.track_count, track_block_size):
        track_end = min(track_start + track_block_size, atlas.track_count)
        coefficients = atlas.transfer_atlas.coefficients[track_start:track_end]
        for frame_start in range(0, int(times.numel()), frame_block_size):
            frame_end = min(frame_start + frame_block_size, int(times.numel()))
            time_block = times[frame_start:frame_end]
            basis = chebyshev_basis(
                time_block,
                t_min=atlas.transfer_atlas.t_min,
                t_max=atlas.transfer_atlas.t_max,
                rank=atlas.node_count,
            )
            chart_block = torch.einsum("fk,pkc->pfc", basis, coefficients)
            _require_interpolated_chart_cone(chart_block)
            transfer_block = transfer_lie_decode(chart_block)
            prediction_block = transfer_block[..., 1:] + transfer_block[..., :1] * background
            residual = prediction_block - targets[track_start:track_end, frame_start:frame_end]
            loss += residual.square().sum() / normalization
            if predictions is not None:
                predictions[track_start:track_end, frame_start:frame_end] = prediction_block
            grad_prediction = 2.0 * residual / normalization
            grad_transfer = torch.cat(
                (
                    (grad_prediction * background).sum(dim=-1, keepdim=True),
                    grad_prediction,
                ),
                dim=-1,
            )
            grad_chart = transfer_lie_decode_vjp(chart_block, grad_transfer)
            node_interpolation = basis @ atlas.transfer_atlas.fit_matrix
            grad_node_chart[track_start:track_end] += torch.einsum(
                "fn,pfc->pnc",
                node_interpolation,
                grad_chart,
            )
    grad_density, grad_color, grad_depth, grad_boundary = _world_vjp_from_node_chart(
        atlas=atlas,
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        site_density=site_density,
        site_color=site_color,
        grad_node_chart=grad_node_chart,
    )
    scalar_bytes = torch.tensor([], dtype=DTYPE).element_size()
    block_tracks = min(track_block_size, atlas.track_count)
    block_frames = min(frame_block_size, int(times.numel()))
    reverse_state_scalars = (
        grad_node_chart.numel()
        + grad_depth.numel()
        + grad_density.numel()
        + grad_color.numel()
        + grad_boundary.numel()
        + block_tracks * block_frames * (4 + 4 + 3 + 3 + 4 + 4)
        + 2 * block_frames * atlas.node_count
    )
    return (
        loss,
        predictions,
        grad_density,
        grad_color,
        grad_depth,
        grad_boundary,
        reverse_state_scalars * scalar_bytes,
    )


def _world_vjp_from_node_chart(
    *,
    atlas: CompiledLieWorldAtlas,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    grad_node_chart: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    incidence_maps = _track_cut_incidence_maps(
        atlas.depth_coefficient_incidence,
        track_count=atlas.track_count,
    )
    grad_density = torch.zeros_like(site_density)
    grad_color = torch.zeros_like(site_color)
    grad_sparse_depth = torch.zeros_like(atlas.sparse_depth_coefficients)
    for track_id, word in enumerate(atlas.words):
        for node_id, time in enumerate(atlas.transfer_atlas.node_times):
            density_grad, color_grad, depth_grad = _word_lie_chart_vjp(
                word=word,
                cut_incidence=incidence_maps[track_id],
                sparse_depth_coefficients=atlas.sparse_depth_coefficients,
                ray_coefficients=ray_coefficients[track_id],
                time=time,
                site_density=site_density,
                site_color=site_color,
                total_chart=atlas.node_chart[track_id, node_id],
                grad_chart=grad_node_chart[track_id, node_id],
                near=atlas.near,
                far=atlas.far,
            )
            grad_density += density_grad
            grad_color += color_grad
            grad_sparse_depth += depth_grad
    grad_boundary = sparse_factorized_depth_coefficients_boundary_vjp(
        boundary,
        ray_coefficients,
        atlas.depth_coefficient_incidence,
        grad_sparse_depth,
    )
    return grad_density, grad_color, grad_sparse_depth, grad_boundary


def _adaptive_validation_report(
    atlas: CompiledLieWorldAtlas,
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    split: Literal["probe", "heldout"],
    validation_count: int,
    direction_count: int,
    policy: AdaptiveLieWorldCompilePolicy,
    track_block_size: int,
    frame_block_size: int,
) -> AdaptiveValidationReport:
    times = _deterministic_validation_times(
        atlas.transfer_atlas.t_min,
        atlas.transfer_atlas.t_max,
        validation_count,
        split=split,
    )
    forward_error, forward_normalized, _ = _sampled_lie_world_transfer_report(
        atlas,
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        site_density=site_density,
        site_color=site_color,
        validation_times=times,
        absolute_tolerance=policy.forward_absolute_tolerance,
        relative_tolerance=policy.forward_relative_tolerance,
        track_block_size=track_block_size,
        frame_block_size=frame_block_size,
    )
    tangent = sampled_lie_world_tangent_error(
        atlas,
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        site_density=site_density,
        site_color=site_color,
        validation_count=validation_count,
        track_block_size=track_block_size,
        frame_block_size=frame_block_size,
        direction_ids=tuple(range(direction_count)),
        direction_split=split,
        validation_times=times,
        absolute_tolerance=policy.tangent_absolute_tolerance,
        relative_tolerance=policy.tangent_relative_tolerance,
    )
    return AdaptiveValidationReport(
        split=split,
        forward_maximum_error=forward_error,
        forward_normalized_error=forward_normalized,
        tangent=tangent,
        passed=(
            forward_normalized <= 1.0
            and tangent.maximum_normalized_world_gradient_error <= 1.0
        ),
    )


def _validate_adaptive_policy(policy: AdaptiveLieWorldCompilePolicy) -> None:
    if any(not isinstance(node_count, int) or isinstance(node_count, bool) for node_count in policy.node_count_schedule):
        raise ValueError("node_count_schedule must contain integer ranks")
    schedule = tuple(policy.node_count_schedule)
    if not schedule or any(node_count < 2 for node_count in schedule):
        raise ValueError("node_count_schedule must contain ranks >= 2")
    if tuple(sorted(set(schedule))) != schedule:
        raise ValueError("node_count_schedule must be strictly increasing and unique")
    integer_fields = (
        policy.probe_validation_count,
        policy.heldout_validation_count,
        policy.probe_direction_count,
        policy.heldout_direction_count,
        policy.max_split_depth,
        policy.max_chart_count,
    )
    if any(not isinstance(value, int) or isinstance(value, bool) for value in integer_fields):
        raise ValueError("validation counts, direction counts, and split limits must be integers")
    if policy.probe_validation_count < 2 or policy.heldout_validation_count < 2:
        raise ValueError("probe and heldout validation counts must be at least 2")
    if policy.heldout_validation_count != policy.probe_validation_count - 1:
        raise ValueError(
            "heldout_validation_count must equal probe_validation_count - 1 so "
            "endpoint-bearing probe samples and midpoint heldout samples are disjoint"
        )
    if policy.probe_direction_count < 2 or policy.heldout_direction_count < 2:
        raise ValueError("probe and heldout direction counts must each be at least 2")
    _validate_error_tolerances(
        policy.forward_absolute_tolerance,
        policy.forward_relative_tolerance,
    )
    _validate_error_tolerances(
        policy.tangent_absolute_tolerance,
        policy.tangent_relative_tolerance,
    )
    if policy.max_split_depth < 0 or policy.max_chart_count < 1:
        raise ValueError("max_split_depth must be non-negative and max_chart_count positive")


def _validate_error_tolerances(absolute_tolerance: float, relative_tolerance: float) -> None:
    if not math.isfinite(absolute_tolerance) or not math.isfinite(relative_tolerance):
        raise ValueError("error tolerances must be finite")
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("error tolerances must be non-negative")
    if absolute_tolerance == 0.0 and relative_tolerance == 0.0:
        raise ValueError("at least one error tolerance must be positive")


def _normalized_block_error(
    error: float,
    scale: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> float:
    denominator = absolute_tolerance + relative_tolerance * scale
    if denominator == 0.0:
        return 0.0 if error == 0.0 else float("inf")
    return error / denominator


def _deterministic_validation_times(
    t_min: float,
    t_max: float,
    count: int,
    *,
    split: Literal["probe", "heldout"],
) -> torch.Tensor:
    if count < 2:
        raise ValueError("validation count must be at least 2")
    if split == "probe":
        return torch.linspace(t_min, t_max, count, dtype=DTYPE)
    if split == "heldout":
        # Cell midpoints are disjoint from the endpoint-bearing probe grid for
        # ordinary counts and exercise interpolation rather than fit nodes.
        unit = (torch.arange(count, dtype=DTYPE) + 0.5) / float(count)
        return t_min + (t_max - t_min) * unit
    raise ValueError("split must be 'probe' or 'heldout'")


def _tangent_validation_cotangent(
    track_ids: torch.Tensor,
    times: torch.Tensor,
    *,
    direction_id: int = 0,
    split: Literal["probe", "heldout"] = "probe",
) -> torch.Tensor:
    if direction_id < 0:
        raise ValueError("direction_id must be non-negative")
    split_phase = 0.0 if split == "probe" else 0.413
    direction_phase = 0.619 * float(direction_id)
    phase = times + 0.37 * track_ids + split_phase + direction_phase
    frequency_shift = 0.173 * float(direction_id)
    return torch.stack(
        (
            0.17 + 0.03 * torch.cos((1.0 + frequency_shift) * phase),
            -0.11 + 0.05 * torch.sin((1.3 + frequency_shift) * phase),
            0.07 + 0.04 * torch.cos((0.7 + 0.5 * frequency_shift) * phase),
            -0.09 + 0.02 * torch.sin((1.9 + 0.25 * frequency_shift) * phase),
        ),
        dim=-1,
    )


def _require_interpolated_chart_cone(chart: torch.Tensor) -> None:
    cone = check_lie_chart_cone(chart)
    if not cone.passed:
        raise ValueError(
            "interpolated Lie chart left the physical cone between nodes; "
            f"maximum violation={cone.maximum_violation:.3e}; split the time chart or raise node_count"
        )


def _scan_word_lie_chart(
    *,
    word: StableCellWord,
    cut_incidence: dict[int, int],
    sparse_depth_coefficients: torch.Tensor,
    ray_coefficients: torch.Tensor,
    time: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    near: float,
    far: float,
) -> torch.Tensor:
    kappa_total, total_moment = _scan_word_state(
        word=word,
        cut_incidence=cut_incidence,
        sparse_depth_coefficients=sparse_depth_coefficients,
        ray_coefficients=ray_coefficients,
        time=time,
        site_density=site_density,
        site_color=site_color,
        near=near,
        far=far,
    )
    inverse_phi = _stable_inverse_phi(kappa_total)
    return torch.cat((kappa_total.reshape(1), inverse_phi * total_moment))


def _scan_word_transfer(**kwargs: object) -> torch.Tensor:
    kappa_total, total_moment = _scan_word_state(**kwargs)
    return torch.cat((torch.exp(-kappa_total).reshape(1), total_moment))


def _scan_word_state(
    *,
    word: StableCellWord,
    cut_incidence: dict[int, int],
    sparse_depth_coefficients: torch.Tensor,
    ray_coefficients: torch.Tensor,
    time: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    near: float,
    far: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    fiber_speed = _fiber_speed(ray_coefficients, time)
    kappa_total = torch.zeros((), dtype=DTYPE)
    prefix_beta = torch.ones((), dtype=DTYPE)
    total_moment = torch.zeros(3, dtype=DTYPE)
    for owner_raw, left_raw, right_raw in zip(
        word.owners.tolist(),
        word.left_cut_ids.tolist(),
        word.right_cut_ids.tolist(),
        strict=True,
    ):
        owner = int(owner_raw)
        left_depth, _ = _cut_depth_and_jacobian(
            cut_incidence,
            sparse_depth_coefficients,
            int(left_raw),
            time,
            near=near,
            far=far,
        )
        right_depth, _ = _cut_depth_and_jacobian(
            cut_incidence,
            sparse_depth_coefficients,
            int(right_raw),
            time,
            near=near,
            far=far,
        )
        coordinate_length = right_depth - left_depth
        if float(coordinate_length.item()) <= 0.0:
            raise ValueError("stable word produced a non-positive segment length")
        optical_depth = site_density[owner] * fiber_speed * coordinate_length
        if float(optical_depth.item()) < 0.0:
            raise ValueError("site density must not produce negative optical depth")
        beta = torch.exp(-optical_depth)
        alpha = -torch.expm1(-optical_depth)
        total_moment = total_moment + prefix_beta * alpha * site_color[owner]
        prefix_beta = prefix_beta * beta
        kappa_total = kappa_total + optical_depth
    return kappa_total, total_moment


def _word_lie_chart_vjp(
    *,
    word: StableCellWord,
    cut_incidence: dict[int, int],
    sparse_depth_coefficients: torch.Tensor,
    ray_coefficients: torch.Tensor,
    time: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    total_chart: torch.Tensor,
    grad_chart: torch.Tensor,
    near: float,
    far: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact node reverse with a constant prefix and no run tape."""

    total_chart_f64 = torch.as_tensor(total_chart, dtype=DTYPE).reshape(4)
    kappa_total = total_chart_f64[0]
    total_moment = transfer_lie_decode(total_chart_f64)[1:]
    grad_moment, grad_kappa_word = lie_chart_word_cotangents(
        kappa_total,
        total_moment,
        grad_chart,
    )
    grad_density = torch.zeros_like(site_density)
    grad_color = torch.zeros_like(site_color)
    grad_sparse_depth = torch.zeros_like(sparse_depth_coefficients)
    fiber_speed = _fiber_speed(ray_coefficients, time)
    prefix_beta = torch.ones((), dtype=DTYPE)
    prefix_moment = torch.zeros(3, dtype=DTYPE)
    for owner_raw, left_raw, right_raw in zip(
        word.owners.tolist(),
        word.left_cut_ids.tolist(),
        word.right_cut_ids.tolist(),
        strict=True,
    ):
        owner = int(owner_raw)
        left_id = int(left_raw)
        right_id = int(right_raw)
        left_depth, left_jacobian = _cut_depth_and_jacobian(
            cut_incidence,
            sparse_depth_coefficients,
            left_id,
            time,
            near=near,
            far=far,
        )
        right_depth, right_jacobian = _cut_depth_and_jacobian(
            cut_incidence,
            sparse_depth_coefficients,
            right_id,
            time,
            near=near,
            far=far,
        )
        coordinate_length = right_depth - left_depth
        physical_length = fiber_speed * coordinate_length
        optical_depth = site_density[owner] * physical_length
        beta = torch.exp(-optical_depth)
        alpha = -torch.expm1(-optical_depth)
        tau_bar = (
            torch.dot(
                grad_moment,
                prefix_moment + prefix_beta * site_color[owner] - total_moment,
            )
            + grad_kappa_word
        )
        grad_density[owner] += physical_length * tau_bar
        grad_color[owner] += prefix_beta * alpha * grad_moment
        coordinate_length_bar = fiber_speed * site_density[owner] * tau_bar
        if left_id >= 0:
            grad_sparse_depth[cut_incidence[left_id]] -= coordinate_length_bar * left_jacobian
        if right_id >= 0:
            grad_sparse_depth[cut_incidence[right_id]] += coordinate_length_bar * right_jacobian
        prefix_moment = prefix_moment + prefix_beta * alpha * site_color[owner]
        prefix_beta = prefix_beta * beta
    return grad_density, grad_color, grad_sparse_depth


def _cut_depth_and_jacobian(
    cut_incidence: dict[int, int],
    sparse_depth_coefficients: torch.Tensor,
    cut_id: int,
    time: torch.Tensor,
    *,
    near: float,
    far: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cut_id == NEAR_CUT_ID:
        return torch.tensor(near, dtype=DTYPE), torch.zeros(4, dtype=DTYPE)
    if cut_id == FAR_CUT_ID:
        return torch.tensor(far, dtype=DTYPE), torch.zeros(4, dtype=DTYPE)
    if cut_id not in cut_incidence:
        raise ValueError(f"word references missing track-boundary incidence {cut_id}")
    coefficient = sparse_depth_coefficients[cut_incidence[cut_id]]
    numerator = coefficient[0] + time * coefficient[1]
    denominator = coefficient[2] + time * coefficient[3]
    scale = max(1.0, float(coefficient[2].abs().item()) + float(coefficient[3].abs().item()))
    if float(denominator.abs().item()) <= 1.0e-9 * scale:
        raise ValueError("boundary/ray denominator is unsafe")
    depth = numerator / denominator
    jacobian = torch.stack(
        (
            1.0 / denominator,
            time / denominator,
            -depth / denominator,
            -time * depth / denominator,
        )
    )
    return depth, jacobian


def _fiber_speed(ray_coefficients: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    direction = ray_coefficients[6:9] + time * ray_coefficients[9:12]
    speed = torch.linalg.vector_norm(direction)
    if float(speed.item()) <= 0.0:
        raise ValueError("ray direction has zero fiber speed")
    return speed


def _stable_inverse_phi(kappa: torch.Tensor) -> torch.Tensor:
    if float(kappa.abs().item()) < 1.0e-4:
        k2 = kappa * kappa
        k4 = k2 * k2
        k6 = k4 * k2
        return 1.0 + kappa / 2.0 + k2 / 12.0 - k4 / 720.0 + k6 / 30240.0
    return kappa / (-torch.expm1(-kappa))


def _evaluate_transfer_block(
    atlas: TemporalTransferAtlas,
    times: torch.Tensor,
    *,
    track_start: int,
    track_end: int,
) -> torch.Tensor:
    subset = TemporalTransferAtlas(
        t_min=atlas.t_min,
        t_max=atlas.t_max,
        node_times=atlas.node_times,
        fit_matrix=atlas.fit_matrix,
        coefficients=atlas.coefficients[track_start:track_end],
        chart=atlas.chart,
    )
    chart = evaluate_transfer_atlas_chart(subset, times)
    _require_interpolated_chart_cone(chart)
    return transfer_lie_decode(chart)


def _track_cut_incidence_maps(
    incidence: torch.Tensor,
    *,
    track_count: int,
) -> tuple[dict[int, int], ...]:
    maps = [dict() for _ in range(track_count)]
    for incidence_id, (track_id, cut_id) in enumerate(incidence.tolist()):
        maps[int(track_id)][int(cut_id)] = incidence_id
    return tuple(maps)


def _words_have_same_topology(
    left: Sequence[StableCellWord],
    right: Sequence[StableCellWord],
) -> bool:
    if len(left) != len(right):
        return False
    return all(
        torch.equal(left_word.owners, right_word.owners)
        and torch.equal(left_word.left_cut_ids, right_word.left_cut_ids)
        and torch.equal(left_word.right_cut_ids, right_word.right_cut_ids)
        for left_word, right_word in zip(left, right, strict=True)
    )


def _validate_world_inputs(
    *,
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    words: Sequence[StableCellWord],
    site_density: torch.Tensor,
    site_color: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[StableCellWord, ...]]:
    boundary_f64 = _require_matrix("boundary", boundary, columns=5)
    rays_f64 = _require_matrix("ray_coefficients", ray_coefficients, columns=12)
    density_f64 = torch.as_tensor(site_density, dtype=DTYPE).reshape(-1)
    color_f64 = _require_matrix("site_color", site_color, columns=3)
    words_tuple = tuple(words)
    if len(words_tuple) != int(rays_f64.shape[0]):
        raise ValueError("words must contain one word per ray track")
    if color_f64.shape[0] != density_f64.shape[0]:
        raise ValueError("site_density and site_color must have the same site count")
    if not bool(torch.isfinite(density_f64).all().item()) or bool(torch.any(density_f64 < 0.0).item()):
        raise ValueError("site_density must be finite and non-negative")
    if not bool(torch.isfinite(color_f64).all().item()):
        raise ValueError("site_color must be finite")
    for word in words_tuple:
        if (
            word.owners.ndim != 1
            or word.left_cut_ids.shape != word.owners.shape
            or word.right_cut_ids.shape != word.owners.shape
        ):
            raise ValueError("each word must have matching one-dimensional owner/cut arrays")
        if word.owners.numel() == 0:
            raise ValueError("words must be non-empty")
        if int(word.owners.min()) < 0 or int(word.owners.max()) >= int(density_f64.numel()):
            raise ValueError("word owner is outside the site arrays")
        referenced = torch.cat((word.left_cut_ids, word.right_cut_ids))
        positive = referenced[referenced >= 0]
        if positive.numel() and int(positive.max()) >= int(boundary_f64.shape[0]):
            raise ValueError("word boundary id is outside boundary")
    return boundary_f64, rays_f64, density_f64, color_f64, words_tuple


def _require_matrix(name: str, value: torch.Tensor, *, columns: int) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=DTYPE)
    if tensor.ndim != 2 or tensor.shape[1] != columns:
        raise ValueError(f"{name} must have shape [N,{columns}]")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite")
    return tensor


def _require_incidence(
    incidence: torch.Tensor,
    *,
    track_count: int,
    boundary_count: int,
) -> torch.Tensor:
    incidence_i64 = torch.as_tensor(incidence, dtype=torch.int64)
    if incidence_i64.ndim != 2 or incidence_i64.shape[1] != 2:
        raise ValueError("incidence must have shape [I,2]")
    if incidence_i64.numel():
        if int(incidence_i64[:, 0].min()) < 0 or int(incidence_i64[:, 0].max()) >= track_count:
            raise ValueError("incidence track id is out of range")
        if int(incidence_i64[:, 1].min()) < 0 or int(incidence_i64[:, 1].max()) >= boundary_count:
            raise ValueError("incidence boundary id is out of range")
    return incidence_i64


def _max_abs(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(value.detach().abs().max().item())
