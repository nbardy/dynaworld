"""Staged CPU contract for streamed affine-Lie WorldFoam training.

The production schedule is deliberately split into three operations:

1. refresh the fixed-topology node atlas once when the world changes;
2. accumulate any number of target/time blocks into persistent node bars using
   one global loss denominator;
3. replay the ordered words and finalize boundary gradients exactly once.

This module makes that schedule executable without retaining targets,
predictions, residuals, or an ``F x R`` reverse tape in the accumulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from compact_lie_schedule import (
    CompactLieWorldSchedule,
    compact_lie_world_schedule_from_atlas,
)
from compiled_lie_world_adjoint import (
    AdaptiveCompiledLieWorldAtlas,
    CompiledLieWorldAtlas,
    PiecewiseCompiledLieWorldVJP,
    _require_interpolated_chart_cone,
    _track_cut_incidence_maps,
    _validate_world_inputs,
    _word_lie_chart_vjp,
    _words_have_same_topology,
    compile_lie_world_atlas,
    refresh_fixed_topology_lie_world_atlas,
    sparse_factorized_depth_coefficients_boundary_vjp,
)
from compiled_transfer_adjoint import (
    StableCellWord,
    check_power_word_adjacency,
    check_supplied_word_ordering,
    make_stable_cell_word,
    power_boundary_parameters,
    power_boundary_parameters_vjp,
)
from prepared_track_block import (
    PreparedWorldFoamTrackBlock,
    accumulate_prepared_rows_,
    gather_prepared_rows,
)
from transfer_lie_chart import (
    DTYPE,
    TemporalTransferAtlas,
    chebyshev_basis,
    transfer_lie_decode,
    transfer_lie_decode_vjp,
)


@dataclass(frozen=True)
class PreparedStagedLieWorld:
    """One refreshed atlas bound to the exact detached world used to build it."""

    atlas: AdaptiveCompiledLieWorldAtlas
    boundary: torch.Tensor
    ray_coefficients: torch.Tensor
    site_density: torch.Tensor
    site_color: torch.Tensor
    tensor_signatures: tuple[tuple[object, ...], ...]

    def assert_current(self) -> None:
        tensors = (
            self.boundary,
            self.ray_coefficients,
            self.site_density,
            self.site_color,
        )
        if tuple(_tensor_signature(tensor) for tensor in tensors) != self.tensor_signatures:
            raise ValueError("prepared staged world tensors changed after atlas refresh")


@dataclass(frozen=True)
class PreparedCompactStagedLieWorld:
    """A spatially compact step snapshot whose boundaries derive from its sites."""

    template: AdaptiveCompiledLieWorldAtlas | None
    template_tensor_signatures: tuple[tuple[object, ...], ...]
    schedule: CompactLieWorldSchedule
    world_snapshot: PreparedStagedLieWorld
    topology: PreparedWorldFoamTrackBlock
    site_geometry: torch.Tensor
    site_geometry_signature: tuple[object, ...]
    topology_tensor_signatures: tuple[tuple[object, ...], ...]
    source_tensors: tuple[torch.Tensor, ...]
    source_tensor_signatures: tuple[tuple[object, ...], ...]

    def assert_current(self) -> None:
        self.schedule.assert_current()
        if self.template is None:
            if self.template_tensor_signatures:
                raise ValueError("template-free compact preparation retained template signatures")
        elif tuple(_tensor_signature(tensor) for tensor in _atlas_tensors(self.template)) != (
            self.template_tensor_signatures
        ):
            raise ValueError("global atlas template changed after compact preparation")
        self.world_snapshot.assert_current()
        if _tensor_signature(self.site_geometry) != self.site_geometry_signature:
            raise ValueError("prepared compact site geometry changed after boundary derivation")
        if tuple(_tensor_signature(tensor) for tensor in _topology_tensors(self.topology)) != (
            self.topology_tensor_signatures
        ):
            raise ValueError("compact topology tensors changed after atlas refresh")
        if tuple(_tensor_signature(tensor) for tensor in self.source_tensors) != (self.source_tensor_signatures):
            raise ValueError("source world tensors changed after compact atlas refresh")


@dataclass(frozen=True)
class CompactStagedLieWorldVJP:
    prepared: PreparedCompactStagedLieWorld
    accumulator: StagedPiecewiseLieMSEAccumulator
    transfer: PiecewiseCompiledLieWorldVJP
    grad_site_geometry: torch.Tensor


@dataclass(frozen=True)
class CompactSpatialGradientBuffers:
    """Caller-owned global parameter bars reused by every spatial block."""

    grad_site_geometry: torch.Tensor
    grad_site_weight: torch.Tensor
    grad_site_density: torch.Tensor
    grad_site_color: torch.Tensor

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.grad_site_geometry,
            self.grad_site_weight,
            self.grad_site_density,
            self.grad_site_color,
        )

    @property
    def resident_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.tensors)


@dataclass(frozen=True)
class CompactSpatialStepResult:
    """Final views of the same caller-owned buffers allocated before the step."""

    loss: torch.Tensor
    gradients: CompactSpatialGradientBuffers
    accounting: dict[str, int | str]


@dataclass
class CompactSpatialStepLedger:
    """One global normalization plus caller-owned bars for all ``B_p`` blocks."""

    template: AdaptiveCompiledLieWorldAtlas | None
    template_tensor_signatures: tuple[tuple[object, ...], ...]
    schedule: CompactLieWorldSchedule
    source_tensors: tuple[torch.Tensor, ...]
    source_tensor_signatures: tuple[tuple[object, ...], ...]
    global_track_count: int
    global_frame_count: int
    global_site_count: int
    loss_normalization_id: str
    expected_blocks: tuple[tuple[str, int, int], ...]
    expected_block_schedule_generations: tuple[tuple[str, str], ...]
    gradients: CompactSpatialGradientBuffers
    loss: torch.Tensor
    state_tensor_signatures: tuple[tuple[object, ...], ...]
    consumed_block_ids: set[str]
    compact_site_rows_accumulated: int = 0
    finalized: bool = False

    @property
    def global_loss_element_count(self) -> int:
        return self.global_track_count * self.global_frame_count * 3

    @property
    def resident_tensor_bytes(self) -> int:
        return self.gradients.resident_bytes + self.loss.numel() * self.loss.element_size()

    @property
    def resident_schedule_bytes(self) -> int:
        return self.schedule.resident_bytes


@dataclass
class StagedPiecewiseLieMSEAccumulator:
    """Persistent sample-reduction state with no target or run-tape axis."""

    world_snapshot: PreparedStagedLieWorld
    interpolation_schedule: CompactLieWorldSchedule
    background: torch.Tensor
    total_frame_count: int
    global_frame_count: int
    global_track_count: int
    loss_normalization_id: str
    frame_block_size: int
    track_block_size: int
    grad_node_charts: tuple[torch.Tensor, ...]
    loss: torch.Tensor
    accumulated_frame_count: int = 0
    sample_basis_interactions: int = 0
    sample_weight_linear_interactions: int = 0
    sample_weight_dense_fallback_interactions: int = 0
    sample_weight_exact_node_rows: int = 0
    sample_weight_dense_fallback_rows: int = 0
    sample_weight_evaluations: set[str] = field(default_factory=set)
    sample_block_count: int = 0
    next_global_frame_start: int = 0
    finalized: bool = False

    @property
    def atlas(self) -> AdaptiveCompiledLieWorldAtlas:
        return self.world_snapshot.atlas

    @property
    def normalization(self) -> float:
        return float(self.global_track_count * self.global_frame_count * 3)

    @property
    def resident_bytes_excluding_atlas(self) -> int:
        tensors = (self.background, self.loss, *self.grad_node_charts)
        return (
            sum(tensor.numel() * tensor.element_size() for tensor in tensors)
            + self.interpolation_schedule.resident_bytes
        )


def refresh_staged_lie_world_snapshot(
    template: AdaptiveCompiledLieWorldAtlas,
    *,
    assume_fixed_topology: Literal[True],
    boundary: torch.Tensor,
    ray_coefficients: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
) -> PreparedStagedLieWorld:
    """Refresh node values and bind the exact world snapshot used by the VJP."""

    atlas = refresh_fixed_topology_lie_world_atlas(
        template,
        assume_fixed_topology=assume_fixed_topology,
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        site_density=site_density,
        site_color=site_color,
    )
    first = atlas.charts[0]
    boundary_f64, rays_f64, density_f64, color_f64, _ = _validate_world_inputs(
        boundary=boundary,
        ray_coefficients=ray_coefficients,
        words=first.words,
        site_density=site_density,
        site_color=site_color,
    )
    tensors = tuple(tensor.detach() for tensor in (boundary_f64, rays_f64, density_f64, color_f64))
    return PreparedStagedLieWorld(
        atlas=atlas,
        boundary=tensors[0],
        ray_coefficients=tensors[1],
        site_density=tensors[2],
        site_color=tensors[3],
        tensor_signatures=tuple(_tensor_signature(tensor) for tensor in tensors),
    )


def prepare_compact_staged_lie_world_snapshot(
    template: AdaptiveCompiledLieWorldAtlas,
    topology: PreparedWorldFoamTrackBlock,
    *,
    site_geometry: torch.Tensor,
    ray_coefficients: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
) -> PreparedCompactStagedLieWorld:
    """Legacy full-template wrapper for the compact reference path.

    New memory-light callers should extract one
    :class:`CompactLieWorldSchedule`, release the full ``P``-track atlas, and
    call :func:`prepare_compact_staged_lie_world_snapshot_v2` for each block.
    Keeping this wrapper preserves the reference path's template mutation and
    object-identity checks.
    """

    _validate_piecewise_atlas(template)
    return _prepare_compact_staged_lie_world_snapshot(
        compact_lie_world_schedule_from_atlas(template),
        topology,
        template=template,
        site_geometry=site_geometry,
        ray_coefficients=ray_coefficients,
        site_density=site_density,
        site_color=site_color,
    )


def prepare_compact_staged_lie_world_snapshot_v2(
    schedule: CompactLieWorldSchedule,
    topology: PreparedWorldFoamTrackBlock,
    *,
    site_geometry: torch.Tensor,
    ray_coefficients: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
) -> PreparedCompactStagedLieWorld:
    """Prepare one ``B_p`` block without retaining a full global atlas.

    The caller supplies global tensors.  The returned token owns a compact,
    detached step snapshot and also retains version signatures for the global
    source tensors, so an optimizer mutation before finalization fails closed.
    No independent boundary tensor is accepted at this seam.
    """

    return _prepare_compact_staged_lie_world_snapshot(
        schedule,
        topology,
        template=None,
        site_geometry=site_geometry,
        ray_coefficients=ray_coefficients,
        site_density=site_density,
        site_color=site_color,
    )


def _prepare_compact_staged_lie_world_snapshot(
    schedule: CompactLieWorldSchedule,
    topology: PreparedWorldFoamTrackBlock,
    *,
    template: AdaptiveCompiledLieWorldAtlas | None,
    site_geometry: torch.Tensor,
    ray_coefficients: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
) -> PreparedCompactStagedLieWorld:
    schedule.assert_current()
    source_tensors = tuple(
        torch.as_tensor(tensor) for tensor in (site_geometry, ray_coefficients, site_density, site_color)
    )
    source_signatures = tuple(_tensor_signature(tensor) for tensor in source_tensors)
    track_ids = topology.source_track_ids
    site_ids = topology.source_site_ids
    if topology.track_count < 1 or topology.site_count < 1:
        raise ValueError("compact topology must contain tracks and sites")
    if int(track_ids.min().item()) < 0 or int(track_ids.max().item()) >= schedule.global_track_count:
        raise ValueError("compact topology references a track outside the global schedule")

    compact_sites = gather_prepared_rows(source_tensors[0], site_ids).to(dtype=DTYPE).detach()
    compact_rays = gather_prepared_rows(source_tensors[1], track_ids).to(dtype=DTYPE).detach()
    compact_density = gather_prepared_rows(source_tensors[2], site_ids).to(dtype=DTYPE).detach()
    compact_color = gather_prepared_rows(source_tensors[3], site_ids).to(dtype=DTYPE).detach()
    compact_pairs = topology.boundary_site_pairs_i32.to(dtype=torch.int64)
    compact_boundary = power_boundary_parameters(compact_sites, compact_pairs).detach()
    compact_words = _prepared_block_words(topology)

    first = schedule.charts[0]
    final = schedule.charts[-1]
    check_power_word_adjacency(
        sites=compact_sites,
        boundary_pairs=compact_pairs,
        ray_coefficients=compact_rays,
        words=compact_words,
        t_min=first.t_min,
        t_max=final.t_max,
    )
    ordering_check = check_supplied_word_ordering(
        boundary=compact_boundary,
        ray_coefficients=compact_rays,
        words=compact_words,
        site_count=topology.site_count,
        t_min=first.t_min,
        t_max=final.t_max,
        near=first.near,
        far=first.far,
    )
    charts = tuple(
        compile_lie_world_atlas(
            boundary=compact_boundary,
            ray_coefficients=compact_rays,
            words=compact_words,
            site_density=compact_density,
            site_color=compact_color,
            t_min=chart.t_min,
            t_max=chart.t_max,
            near=chart.near,
            far=chart.far,
            node_count=chart.node_count,
        )
        for chart in schedule.charts
    )
    for actual, expected in zip(charts, schedule.charts, strict=True):
        if (
            actual.transfer_atlas.chart != expected.chart
            or not torch.equal(
                actual.transfer_atlas.node_times,
                expected.node_times,
            )
            or not torch.equal(actual.transfer_atlas.fit_matrix, expected.fit_matrix)
        ):
            raise ValueError("compact block compiler changed the global chart schedule")
    atlas = AdaptiveCompiledLieWorldAtlas(
        charts=charts,
        selections=schedule.selections,
        policy=schedule.policy,
        supplied_word_ordering_check=ordering_check,
    )
    snapshot_tensors = (compact_boundary, compact_rays, compact_density, compact_color)
    world_snapshot = PreparedStagedLieWorld(
        atlas=atlas,
        boundary=snapshot_tensors[0],
        ray_coefficients=snapshot_tensors[1],
        site_density=snapshot_tensors[2],
        site_color=snapshot_tensors[3],
        tensor_signatures=tuple(_tensor_signature(tensor) for tensor in snapshot_tensors),
    )
    if tuple(_tensor_signature(tensor) for tensor in source_tensors) != source_signatures:
        raise ValueError("source world tensors changed while preparing the compact atlas")
    return PreparedCompactStagedLieWorld(
        template=template,
        template_tensor_signatures=(
            tuple(_tensor_signature(tensor) for tensor in _atlas_tensors(template)) if template is not None else ()
        ),
        schedule=schedule,
        world_snapshot=world_snapshot,
        topology=topology,
        site_geometry=compact_sites,
        site_geometry_signature=_tensor_signature(compact_sites),
        topology_tensor_signatures=tuple(_tensor_signature(tensor) for tensor in _topology_tensors(topology)),
        source_tensors=source_tensors,
        source_tensor_signatures=source_signatures,
    )


def finalize_compact_staged_lie_world_vjp(
    accumulator: StagedPiecewiseLieMSEAccumulator,
    prepared: PreparedCompactStagedLieWorld,
) -> CompactStagedLieWorldVJP:
    """Finalize one block and lower active-face bars to site/weight bars once."""

    prepared.assert_current()
    if accumulator.world_snapshot is not prepared.world_snapshot:
        raise ValueError("accumulator and compact prepared token belong to different world snapshots")
    transfer = finalize_staged_piecewise_lie_world_vjp(accumulator)
    grad_site_geometry = power_boundary_parameters_vjp(
        prepared.site_geometry,
        prepared.topology.boundary_site_pairs_i32,
        transfer.grad_boundary,
    )
    prepared.assert_current()
    return CompactStagedLieWorldVJP(
        prepared=prepared,
        accumulator=accumulator,
        transfer=transfer,
        grad_site_geometry=grad_site_geometry,
    )


def allocate_compact_spatial_gradient_buffers(
    *,
    site_geometry: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
) -> CompactSpatialGradientBuffers:
    """Allocate the four global parameter-gradient buffers exactly once."""

    geometry = torch.as_tensor(site_geometry)
    density = torch.as_tensor(site_density)
    color = torch.as_tensor(site_color)
    _validate_global_site_parameters(geometry, density, color)
    return CompactSpatialGradientBuffers(
        grad_site_geometry=torch.zeros_like(geometry[:, :4]),
        grad_site_weight=torch.zeros_like(geometry[:, 4]),
        grad_site_density=torch.zeros_like(density),
        grad_site_color=torch.zeros_like(color),
    )


def begin_compact_spatial_step(
    *,
    template: AdaptiveCompiledLieWorldAtlas,
    site_geometry: torch.Tensor,
    ray_coefficients: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    gradients: CompactSpatialGradientBuffers,
    global_track_count: int,
    global_frame_count: int,
    loss_normalization_id: str,
    expected_blocks: tuple[tuple[str, int, int], ...],
) -> CompactSpatialStepLedger:
    """Legacy full-template wrapper for the global reference ledger."""

    _validate_piecewise_atlas(template)
    return _begin_compact_spatial_step(
        compact_lie_world_schedule_from_atlas(template),
        template=template,
        site_geometry=site_geometry,
        ray_coefficients=ray_coefficients,
        site_density=site_density,
        site_color=site_color,
        gradients=gradients,
        global_track_count=global_track_count,
        global_frame_count=global_frame_count,
        loss_normalization_id=loss_normalization_id,
        expected_blocks=expected_blocks,
    )


def begin_compact_spatial_step_v2(
    *,
    schedule: CompactLieWorldSchedule,
    site_geometry: torch.Tensor,
    ray_coefficients: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    gradients: CompactSpatialGradientBuffers,
    global_track_count: int,
    global_frame_count: int,
    loss_normalization_id: str,
    expected_blocks: tuple[tuple[str, int, int], ...],
    expected_block_schedule_generations: tuple[tuple[str, str], ...] | None = None,
) -> CompactSpatialStepLedger:
    """Bind a logical step while retaining only a compact chart schedule.

    By default every block uses ``schedule``.  A caller may register an exact
    per-block generation map when independently certified blocks require
    different ranks or chart splits; the global ledger itself only owns loss
    normalization and parameter bars.
    """

    return _begin_compact_spatial_step(
        schedule,
        template=None,
        site_geometry=site_geometry,
        ray_coefficients=ray_coefficients,
        site_density=site_density,
        site_color=site_color,
        gradients=gradients,
        global_track_count=global_track_count,
        global_frame_count=global_frame_count,
        loss_normalization_id=loss_normalization_id,
        expected_blocks=expected_blocks,
        expected_block_schedule_generations=expected_block_schedule_generations,
    )


def _begin_compact_spatial_step(
    schedule: CompactLieWorldSchedule,
    *,
    template: AdaptiveCompiledLieWorldAtlas | None,
    site_geometry: torch.Tensor,
    ray_coefficients: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
    gradients: CompactSpatialGradientBuffers,
    global_track_count: int,
    global_frame_count: int,
    loss_normalization_id: str,
    expected_blocks: tuple[tuple[str, int, int], ...],
    expected_block_schedule_generations: tuple[tuple[str, str], ...] | None = None,
) -> CompactSpatialStepLedger:
    schedule.assert_current()
    if global_track_count < 1 or global_frame_count < 1:
        raise ValueError("global track and frame counts must be positive")
    if schedule.global_track_count != global_track_count:
        raise ValueError("global_track_count must match the compact chart schedule")
    if not loss_normalization_id.strip():
        raise ValueError("loss_normalization_id must be nonempty")
    if not expected_blocks:
        raise ValueError("expected_blocks must be nonempty")
    block_ids = tuple(block_id for block_id, _, _ in expected_blocks)
    if any(not block_id.strip() for block_id in block_ids) or len(set(block_ids)) != len(block_ids):
        raise ValueError("spatial block ids must be nonempty and unique")
    next_track = 0
    for _, track_start, track_end in expected_blocks:
        if track_start != next_track or track_end <= track_start:
            raise ValueError("spatial blocks must form one ordered half-open tiling")
        next_track = track_end
    if next_track != global_track_count:
        raise ValueError("spatial blocks must cover every global track exactly once")
    if expected_block_schedule_generations is None:
        schedule_generations = tuple((block_id, schedule.generation_digest) for block_id in block_ids)
    else:
        schedule_generations = tuple(expected_block_schedule_generations)
        generation_block_ids = tuple(block_id for block_id, _ in schedule_generations)
        if generation_block_ids != block_ids:
            raise ValueError("block schedule generations must follow the registered block order exactly")
        if any(
            len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
            for _, digest in schedule_generations
        ):
            raise ValueError("block schedule generations must be lowercase SHA-256 digests")
    source_tensors = tuple(
        torch.as_tensor(tensor) for tensor in (site_geometry, ray_coefficients, site_density, site_color)
    )
    _validate_compact_spatial_sources(
        source_tensors,
        global_track_count=global_track_count,
    )
    _validate_global_gradient_buffers(gradients, source_tensors)
    source_ptrs = {tensor.untyped_storage().data_ptr() for tensor in source_tensors}
    gradient_ptrs = [tensor.untyped_storage().data_ptr() for tensor in gradients.tensors]
    if len(set(gradient_ptrs)) != len(gradient_ptrs) or any(pointer in source_ptrs for pointer in gradient_ptrs):
        raise ValueError("global gradient buffers must own distinct storage from sources and each other")
    for tensor in gradients.tensors:
        tensor.zero_()
    loss = torch.zeros((), dtype=source_tensors[2].dtype, device=source_tensors[2].device)
    state_tensors = (*gradients.tensors, loss)
    return CompactSpatialStepLedger(
        template=template,
        template_tensor_signatures=(
            tuple(_tensor_signature(tensor) for tensor in _atlas_tensors(template)) if template is not None else ()
        ),
        schedule=schedule,
        source_tensors=source_tensors,
        source_tensor_signatures=tuple(_tensor_signature(tensor) for tensor in source_tensors),
        global_track_count=global_track_count,
        global_frame_count=global_frame_count,
        global_site_count=int(source_tensors[0].shape[0]),
        loss_normalization_id=loss_normalization_id,
        expected_blocks=expected_blocks,
        expected_block_schedule_generations=schedule_generations,
        gradients=gradients,
        loss=loss,
        state_tensor_signatures=tuple(_tensor_signature(tensor) for tensor in state_tensors),
        consumed_block_ids=set(),
    )


def consume_compact_spatial_block_result(
    ledger: CompactSpatialStepLedger,
    *,
    block_id: str,
    prepared: PreparedCompactStagedLieWorld,
    accumulator: StagedPiecewiseLieMSEAccumulator,
    result: CompactStagedLieWorldVJP,
) -> None:
    """Index-add one finalized block into the step's existing global bars."""

    _assert_compact_spatial_step_current(ledger)
    if ledger.finalized:
        raise ValueError("compact spatial step was already finalized")
    if block_id in ledger.consumed_block_ids:
        raise ValueError("compact spatial block was already consumed")
    matches = tuple(row for row in ledger.expected_blocks if row[0] == block_id)
    if len(matches) != 1:
        raise ValueError("compact spatial block id is not registered")
    _, track_start, track_end = matches[0]
    prepared.assert_current()
    if (
        result.prepared is not prepared
        or result.accumulator is not accumulator
        or accumulator.world_snapshot is not prepared.world_snapshot
    ):
        raise ValueError("compact result, accumulator, and prepared token do not match")
    _assert_prepared_schedule_matches_ledger(
        ledger,
        prepared,
        block_id=block_id,
        legacy_error="compact spatial block uses a different global atlas template",
    )
    if len(prepared.source_tensors) != len(ledger.source_tensors) or any(
        actual is not expected for actual, expected in zip(prepared.source_tensors, ledger.source_tensors, strict=True)
    ):
        raise ValueError("compact spatial blocks do not share the step's source world tensors")
    if not accumulator.finalized:
        raise ValueError("compact spatial block must finish its world VJP before consumption")
    if (
        accumulator.global_track_count != ledger.global_track_count
        or accumulator.global_frame_count != ledger.global_frame_count
        or accumulator.loss_normalization_id != ledger.loss_normalization_id
    ):
        raise ValueError("compact spatial block uses a different global loss normalization")
    expected_track_ids = torch.arange(
        track_start,
        track_end,
        dtype=prepared.topology.source_track_ids.dtype,
        device=prepared.topology.source_track_ids.device,
    )
    if not torch.equal(prepared.topology.source_track_ids, expected_track_ids):
        raise ValueError("compact topology does not match its registered global track range")
    _validate_compact_result_for_global_accumulation(
        ledger,
        prepared,
        result,
        block_id=block_id,
    )
    source_site_ids = prepared.topology.source_site_ids
    accumulate_prepared_rows_(
        ledger.gradients.grad_site_geometry,
        result.grad_site_geometry[:, :4],
        source_site_ids,
    )
    accumulate_prepared_rows_(
        ledger.gradients.grad_site_weight,
        result.grad_site_geometry[:, 4],
        source_site_ids,
    )
    accumulate_prepared_rows_(
        ledger.gradients.grad_site_density,
        result.transfer.grad_site_density,
        source_site_ids,
    )
    accumulate_prepared_rows_(
        ledger.gradients.grad_site_color,
        result.transfer.grad_site_color,
        source_site_ids,
    )
    ledger.loss.add_(result.transfer.loss)
    ledger.compact_site_rows_accumulated += int(source_site_ids.numel())
    ledger.consumed_block_ids.add(block_id)
    ledger.state_tensor_signatures = tuple(
        _tensor_signature(tensor) for tensor in (*ledger.gradients.tensors, ledger.loss)
    )
    _assert_compact_spatial_step_current(ledger)


def finalize_compact_spatial_step(
    ledger: CompactSpatialStepLedger,
) -> CompactSpatialStepResult:
    """Return views of the preallocated bars after an exact spatial tiling."""

    _assert_compact_spatial_step_current(ledger)
    if ledger.finalized:
        raise ValueError("compact spatial step was already finalized")
    expected_ids = {block_id for block_id, _, _ in ledger.expected_blocks}
    if ledger.consumed_block_ids != expected_ids:
        raise ValueError("compact spatial step cannot finalize with missing track blocks")
    ledger.finalized = True
    return CompactSpatialStepResult(
        loss=ledger.loss,
        gradients=ledger.gradients,
        accounting={
            "global_track_count": ledger.global_track_count,
            "global_frame_count": ledger.global_frame_count,
            "global_site_count": ledger.global_site_count,
            "global_loss_element_count": ledger.global_loss_element_count,
            "loss_normalization_id": ledger.loss_normalization_id,
            "global_gradient_buffer_allocations": 4,
            "global_gradient_buffer_bytes": ledger.gradients.resident_bytes,
            "step_state_tensor_bytes": ledger.resident_tensor_bytes,
            "chart_schedule_bytes": ledger.resident_schedule_bytes,
            "chart_schedule_generation": ledger.schedule.generation_digest,
            "full_global_atlas_retained": int(ledger.template is not None),
            "distinct_expected_chart_schedule_count": len(
                {digest for _, digest in ledger.expected_block_schedule_generations}
            ),
            "expected_spatial_block_count": len(ledger.expected_blocks),
            "consumed_spatial_block_count": len(ledger.consumed_block_ids),
            "compact_site_rows_accumulated": ledger.compact_site_rows_accumulated,
        },
    )


def begin_staged_piecewise_lie_mse(
    world_snapshot: PreparedStagedLieWorld,
    *,
    background: torch.Tensor | tuple[float, float, float] | list[float],
    total_frame_count: int,
    global_frame_count: int | None = None,
    global_track_count: int | None = None,
    loss_normalization_id: str = "logical-step",
    frame_block_size: int,
    track_block_size: int = 64,
) -> StagedPiecewiseLieMSEAccumulator:
    """Allocate only loss and node-cotangent state for one logical step."""

    world_snapshot.assert_current()
    atlas = world_snapshot.atlas
    _validate_piecewise_atlas(atlas)
    if total_frame_count < 1:
        raise ValueError("total_frame_count must be positive")
    normalized_global_frame_count = int(total_frame_count) if global_frame_count is None else int(global_frame_count)
    if normalized_global_frame_count < total_frame_count:
        raise ValueError("global_frame_count cannot be smaller than the local sample count")
    normalized_global_track_count = atlas.track_count if global_track_count is None else int(global_track_count)
    if normalized_global_track_count < atlas.track_count:
        raise ValueError("global_track_count cannot be smaller than the selected atlas block")
    if not loss_normalization_id.strip():
        raise ValueError("loss_normalization_id must be nonempty")
    if frame_block_size < 1 or track_block_size < 1:
        raise ValueError("frame_block_size and track_block_size must be positive")
    background_f64 = torch.as_tensor(background, dtype=DTYPE).reshape(3).detach()
    if not bool(torch.isfinite(background_f64).all().item()):
        raise ValueError("background must be finite")
    return StagedPiecewiseLieMSEAccumulator(
        world_snapshot=world_snapshot,
        interpolation_schedule=compact_lie_world_schedule_from_atlas(atlas),
        background=background_f64,
        total_frame_count=int(total_frame_count),
        global_frame_count=normalized_global_frame_count,
        global_track_count=normalized_global_track_count,
        loss_normalization_id=loss_normalization_id,
        frame_block_size=int(frame_block_size),
        track_block_size=int(track_block_size),
        grad_node_charts=tuple(
            torch.zeros((chart.track_count, chart.node_count, 4), dtype=DTYPE) for chart in atlas.charts
        ),
        loss=torch.zeros((), dtype=DTYPE),
    )


def accumulate_staged_piecewise_lie_mse(
    accumulator: StagedPiecewiseLieMSEAccumulator,
    *,
    times: torch.Tensor,
    targets: torch.Tensor,
    global_frame_start: int | None = None,
    return_predictions: bool = False,
) -> torch.Tensor | None:
    """Reduce one selected target block; never scan an ordered world word."""

    if accumulator.finalized:
        raise ValueError("cannot accumulate after the staged world VJP was finalized")
    accumulator.world_snapshot.assert_current()
    times_f64 = torch.as_tensor(times, dtype=DTYPE).reshape(-1).detach()
    if times_f64.numel() < 1 or not bool(torch.isfinite(times_f64).all().item()):
        raise ValueError("times must be non-empty and finite")
    targets_f64 = torch.as_tensor(targets, dtype=DTYPE).detach()
    expected_shape = (accumulator.atlas.track_count, int(times_f64.numel()), 3)
    if tuple(targets_f64.shape) != expected_shape:
        raise ValueError(f"targets must have shape {expected_shape}")
    if not bool(torch.isfinite(targets_f64).all().item()):
        raise ValueError("targets must be finite")
    frame_count = int(times_f64.numel())
    interval_start = accumulator.next_global_frame_start if global_frame_start is None else int(global_frame_start)
    interval_end = interval_start + frame_count
    if interval_start < 0 or interval_end > accumulator.total_frame_count:
        raise ValueError("sample block exceeds the declared global frame interval")
    if interval_start != accumulator.next_global_frame_start:
        raise ValueError("sample block does not start at the next global frame slot")
    next_count = accumulator.accumulated_frame_count + frame_count
    if next_count > accumulator.total_frame_count:
        raise ValueError("sample blocks exceed the declared global frame count")

    predictions = torch.empty_like(targets_f64) if return_predictions else None
    assigned = torch.zeros(int(times_f64.numel()), dtype=torch.bool)
    if accumulator.interpolation_schedule.selection_signature != tuple(
        (chart.transfer_atlas.t_min, chart.transfer_atlas.t_max, chart.node_count)
        for chart in accumulator.atlas.charts
    ):
        raise ValueError("sample-weight schedule no longer matches the staged atlas")
    for chart_id, (chart, weight_schedule) in enumerate(
        zip(accumulator.atlas.charts, accumulator.interpolation_schedule.charts, strict=True)
    ):
        is_last = chart_id == accumulator.atlas.chart_count - 1
        mask = times_f64 >= chart.transfer_atlas.t_min
        mask &= times_f64 <= chart.transfer_atlas.t_max if is_last else times_f64 < chart.transfer_atlas.t_max
        sample_ids = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if sample_ids.numel() == 0:
            continue
        if bool(assigned[sample_ids].any().item()):
            raise ValueError("adaptive atlas assigned a sample to multiple charts")
        assigned[sample_ids] = True
        chart_times = times_f64[sample_ids]
        for frame_start in range(0, int(sample_ids.numel()), accumulator.frame_block_size):
            frame_end = min(frame_start + accumulator.frame_block_size, int(sample_ids.numel()))
            frame_times = chart_times[frame_start:frame_end]
            local_ids = sample_ids[frame_start:frame_end]
            basis = chebyshev_basis(
                frame_times,
                t_min=chart.transfer_atlas.t_min,
                t_max=chart.transfer_atlas.t_max,
                rank=chart.node_count,
            )
            interpolation = weight_schedule.sample_to_node_weights(frame_times)
            accumulator.sample_weight_evaluations.add(interpolation.evaluation)
            accumulator.sample_weight_linear_interactions += interpolation.linear_weight_interactions
            accumulator.sample_weight_dense_fallback_interactions += (
                interpolation.dense_fallback_interactions
            )
            accumulator.sample_weight_exact_node_rows += interpolation.exact_node_row_count
            accumulator.sample_weight_dense_fallback_rows += interpolation.dense_fallback_row_count
            node_interpolation = interpolation.weights
            for track_start in range(0, chart.track_count, accumulator.track_block_size):
                track_end = min(track_start + accumulator.track_block_size, chart.track_count)
                coefficients = chart.transfer_atlas.coefficients[track_start:track_end]
                chart_block = torch.einsum("fk,pkc->pfc", basis, coefficients)
                _require_interpolated_chart_cone(chart_block)
                transfer = transfer_lie_decode(chart_block)
                prediction = transfer[..., 1:] + transfer[..., :1] * accumulator.background
                residual = prediction - targets_f64[track_start:track_end, local_ids]
                accumulator.loss += residual.square().sum() / accumulator.normalization
                if predictions is not None:
                    predictions[track_start:track_end, local_ids] = prediction
                grad_prediction = 2.0 * residual / accumulator.normalization
                grad_transfer = torch.cat(
                    (
                        (grad_prediction * accumulator.background).sum(dim=-1, keepdim=True),
                        grad_prediction,
                    ),
                    dim=-1,
                )
                grad_chart = transfer_lie_decode_vjp(chart_block, grad_transfer)
                accumulator.grad_node_charts[chart_id][track_start:track_end] += torch.einsum(
                    "fn,pfc->pnc",
                    node_interpolation,
                    grad_chart,
                )
        accumulator.sample_basis_interactions += chart.track_count * int(sample_ids.numel()) * chart.node_count
    if not bool(assigned.all().item()):
        raise ValueError("adaptive atlas did not cover every requested sample")
    accumulator.accumulated_frame_count = next_count
    accumulator.sample_block_count += 1
    accumulator.next_global_frame_start = interval_end
    return predictions


def finalize_staged_piecewise_lie_world_vjp(
    accumulator: StagedPiecewiseLieMSEAccumulator,
) -> PiecewiseCompiledLieWorldVJP:
    """Run all node-word reverses, then finalize shared boundary bars once."""

    if accumulator.finalized:
        raise ValueError("staged world VJP was already finalized")
    accumulator.world_snapshot.assert_current()
    if accumulator.accumulated_frame_count != accumulator.total_frame_count:
        raise ValueError(
            "cannot finalize before all declared frames were accumulated: "
            f"{accumulator.accumulated_frame_count}/{accumulator.total_frame_count}"
        )
    if accumulator.next_global_frame_start != accumulator.total_frame_count:
        raise ValueError("sample blocks do not cover the declared global frame interval")
    first = accumulator.atlas.charts[0]
    boundary_f64 = accumulator.world_snapshot.boundary
    rays_f64 = accumulator.world_snapshot.ray_coefficients
    density_f64 = accumulator.world_snapshot.site_density
    color_f64 = accumulator.world_snapshot.site_color
    words = first.words
    incidence_maps = _track_cut_incidence_maps(
        first.depth_coefficient_incidence,
        track_count=first.track_count,
    )
    grad_density = torch.zeros_like(density_f64)
    grad_color = torch.zeros_like(color_f64)
    grad_depth = torch.zeros_like(first.sparse_depth_coefficients)
    for chart, grad_node_chart in zip(
        accumulator.atlas.charts,
        accumulator.grad_node_charts,
        strict=True,
    ):
        for track_id, word in enumerate(words):
            for node_id, time in enumerate(chart.transfer_atlas.node_times):
                density_bar, color_bar, depth_bar = _word_lie_chart_vjp(
                    word=word,
                    cut_incidence=incidence_maps[track_id],
                    sparse_depth_coefficients=chart.sparse_depth_coefficients,
                    ray_coefficients=rays_f64[track_id],
                    time=time,
                    site_density=density_f64,
                    site_color=color_f64,
                    total_chart=chart.node_chart[track_id, node_id],
                    grad_chart=grad_node_chart[track_id, node_id],
                    near=chart.near,
                    far=chart.far,
                )
                grad_density += density_bar
                grad_color += color_bar
                grad_depth += depth_bar
    grad_boundary = sparse_factorized_depth_coefficients_boundary_vjp(
        boundary_f64,
        rays_f64,
        first.depth_coefficient_incidence,
        grad_depth,
    )
    accumulator.finalized = True
    run_count = sum(int(word.owners.numel()) for word in words)
    scalar_bytes = torch.tensor([], dtype=DTYPE).element_size()
    block_tracks = min(accumulator.track_block_size, first.track_count)
    block_frames = min(accumulator.frame_block_size, accumulator.total_frame_count)
    max_node_count = max(chart.node_count for chart in accumulator.atlas.charts)
    sample_scratch_scalars = block_tracks * block_frames * (4 + 4 + 3 + 3 + 4 + 4) + 2 * block_frames * max_node_count
    peak_reverse_state_bytes = (
        accumulator.resident_bytes_excluding_atlas
        + sum(
            tensor.numel() * tensor.element_size() for tensor in (grad_density, grad_color, grad_depth, grad_boundary)
        )
        + sample_scratch_scalars * scalar_bytes
    )
    accounting = {
        "track_count": first.track_count,
        "global_track_count": accumulator.global_track_count,
        "frame_count": accumulator.total_frame_count,
        "global_frame_count": accumulator.global_frame_count,
        "chart_count": accumulator.atlas.chart_count,
        "total_node_count": accumulator.atlas.total_node_count,
        "run_count": run_count,
        "referenced_track_boundaries": int(first.depth_coefficient_incidence.shape[0]),
        "refresh_world_forward_run_interactions": accumulator.atlas.total_node_count * run_count,
        "step_world_reverse_run_interactions": accumulator.atlas.total_node_count * run_count,
        "sample_basis_interactions": accumulator.sample_basis_interactions,
        "sample_weight_evaluation": "+".join(sorted(accumulator.sample_weight_evaluations)),
        "sample_weight_common_path_complexity": "O(FJ)",
        "sample_weight_dense_fallback_complexity": "O(F_fallback*J^2)",
        "sample_weight_linear_interactions": accumulator.sample_weight_linear_interactions,
        "sample_weight_dense_fallback_interactions": (
            accumulator.sample_weight_dense_fallback_interactions
        ),
        "sample_weight_exact_node_rows": accumulator.sample_weight_exact_node_rows,
        "sample_weight_dense_fallback_rows": accumulator.sample_weight_dense_fallback_rows,
        "sample_weight_schedule_bytes": accumulator.interpolation_schedule.resident_bytes,
        "sample_block_count": accumulator.sample_block_count,
        "world_finalize_calls": 1,
        "boundary_finalize_calls": 1,
        "frame_run_reverse_state_elements": 0,
        "per_sample_run_tape_bytes": 0,
        "retained_target_bytes": 0,
        "retained_prediction_bytes": 0,
        "accumulator_bytes_excluding_atlas": accumulator.resident_bytes_excluding_atlas,
        "logical_selected_reverse_state_bytes_excluding_targets_and_predictions": (peak_reverse_state_bytes),
        "sampled_validation_count": 0,
        "validation_exact_run_interactions": 0,
    }
    return PiecewiseCompiledLieWorldVJP(
        loss=accumulator.loss,
        predictions=None,
        atlas=accumulator.atlas,
        grad_site_density=grad_density,
        grad_site_color=grad_color,
        grad_depth_coefficients=grad_depth,
        grad_boundary=grad_boundary,
        accounting=accounting,
    )


def slice_adaptive_lie_world_atlas_tracks(
    atlas: AdaptiveCompiledLieWorldAtlas,
    *,
    track_start: int,
    track_end: int,
) -> AdaptiveCompiledLieWorldAtlas:
    """Create a compact coefficient/incidence view for one spatial track block."""

    _validate_piecewise_atlas(atlas)
    if track_start < 0 or track_end <= track_start or track_end > atlas.track_count:
        raise ValueError("expected 0 <= track_start < track_end <= atlas.track_count")
    charts = []
    for chart in atlas.charts:
        incidence_mask = (chart.depth_coefficient_incidence[:, 0] >= track_start) & (
            chart.depth_coefficient_incidence[:, 0] < track_end
        )
        local_incidence = chart.depth_coefficient_incidence[incidence_mask].clone()
        if local_incidence.numel():
            local_incidence[:, 0] -= track_start
        charts.append(
            CompiledLieWorldAtlas(
                transfer_atlas=TemporalTransferAtlas(
                    t_min=chart.transfer_atlas.t_min,
                    t_max=chart.transfer_atlas.t_max,
                    node_times=chart.transfer_atlas.node_times,
                    fit_matrix=chart.transfer_atlas.fit_matrix,
                    coefficients=chart.transfer_atlas.coefficients[track_start:track_end].contiguous(),
                    chart=chart.transfer_atlas.chart,
                ),
                node_chart=chart.node_chart[track_start:track_end].contiguous(),
                near=chart.near,
                far=chart.far,
                words=chart.words[track_start:track_end],
                depth_coefficient_incidence=local_incidence.contiguous(),
                sparse_depth_coefficients=chart.sparse_depth_coefficients[incidence_mask].contiguous(),
                supplied_word_ordering_check=chart.supplied_word_ordering_check,
            )
        )
    sliced = AdaptiveCompiledLieWorldAtlas(
        charts=tuple(charts),
        selections=atlas.selections,
        policy=atlas.policy,
        supplied_word_ordering_check=atlas.supplied_word_ordering_check,
    )
    _validate_piecewise_atlas(sliced)
    return sliced


def _validate_global_site_parameters(
    site_geometry: torch.Tensor,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
) -> None:
    site_count = int(site_geometry.shape[0]) if site_geometry.ndim == 2 else -1
    if site_count < 1 or tuple(site_geometry.shape) != (site_count, 5):
        raise ValueError("site_geometry must have shape [S,5] with S > 0")
    if tuple(site_density.shape) != (site_count,):
        raise ValueError("site_density must have shape [S]")
    if tuple(site_color.shape) != (site_count, 3):
        raise ValueError("site_color must have shape [S,3]")
    tensors = (site_geometry, site_density, site_color)
    if any(not tensor.dtype.is_floating_point for tensor in tensors):
        raise ValueError("global site parameters must use floating-point tensors")
    if len({tensor.dtype for tensor in tensors}) != 1 or len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("global site parameters must share one dtype and device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("global site parameters must be contiguous")
    if any(not bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
        raise ValueError("global site parameters must be finite")


def _validate_compact_spatial_sources(
    source_tensors: tuple[torch.Tensor, ...],
    *,
    global_track_count: int,
) -> None:
    if len(source_tensors) != 4:
        raise ValueError("compact spatial step requires geometry, rays, density, and color")
    geometry, rays, density, color = source_tensors
    _validate_global_site_parameters(geometry, density, color)
    if tuple(rays.shape) != (global_track_count, 12):
        raise ValueError("ray_coefficients must have shape [global_track_count,12]")
    if not rays.dtype.is_floating_point or rays.dtype != geometry.dtype or rays.device != geometry.device:
        raise ValueError("ray coefficients must match global site parameter dtype and device")
    if not rays.is_contiguous() or not bool(torch.isfinite(rays).all().item()):
        raise ValueError("ray coefficients must be contiguous and finite")


def _validate_global_gradient_buffers(
    gradients: CompactSpatialGradientBuffers,
    source_tensors: tuple[torch.Tensor, ...],
) -> None:
    geometry, _, density, color = source_tensors
    expected = (
        (gradients.grad_site_geometry, geometry[:, :4].shape),
        (gradients.grad_site_weight, geometry[:, 4].shape),
        (gradients.grad_site_density, density.shape),
        (gradients.grad_site_color, color.shape),
    )
    for tensor, shape in expected:
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError("global gradient buffer shape does not match its parameter")
        if tensor.dtype != geometry.dtype or tensor.device != geometry.device:
            raise ValueError("global gradient buffers must match parameter dtype and device")
        if not tensor.is_contiguous() or tensor.requires_grad:
            raise ValueError("global gradient buffers must be contiguous manual-adjoint tensors")


def _assert_compact_spatial_step_current(ledger: CompactSpatialStepLedger) -> None:
    ledger.schedule.assert_current()
    if ledger.template is None:
        if ledger.template_tensor_signatures:
            raise ValueError("template-free compact spatial step retained template signatures")
    elif tuple(_tensor_signature(tensor) for tensor in _atlas_tensors(ledger.template)) != (
        ledger.template_tensor_signatures
    ):
        raise ValueError("compact spatial step atlas template changed")
    if tuple(_tensor_signature(tensor) for tensor in ledger.source_tensors) != (ledger.source_tensor_signatures):
        raise ValueError("compact spatial step source world tensors changed")
    if (
        tuple(_tensor_signature(tensor) for tensor in (*ledger.gradients.tensors, ledger.loss))
        != ledger.state_tensor_signatures
    ):
        raise ValueError("compact spatial step gradient state changed outside its accumulator")


def _validate_compact_result_for_global_accumulation(
    ledger: CompactSpatialStepLedger,
    prepared: PreparedCompactStagedLieWorld,
    result: CompactStagedLieWorldVJP,
    *,
    block_id: str,
) -> None:
    _assert_prepared_schedule_matches_ledger(
        ledger,
        prepared,
        block_id=block_id,
        legacy_error="compact spatial block uses a different atlas generation",
    )
    if prepared.source_tensor_signatures != ledger.source_tensor_signatures:
        raise ValueError("compact spatial block uses a different source world generation")
    if result.transfer.atlas is not prepared.world_snapshot.atlas:
        raise ValueError("compact transfer result belongs to a different prepared world atlas")
    site_count = prepared.topology.site_count
    tensors_and_shapes = (
        (result.grad_site_geometry, (site_count, 5), ledger.gradients.grad_site_geometry),
        (result.transfer.grad_site_density, (site_count,), ledger.gradients.grad_site_density),
        (result.transfer.grad_site_color, (site_count, 3), ledger.gradients.grad_site_color),
    )
    for tensor, shape, destination in tensors_and_shapes:
        if tuple(tensor.shape) != shape:
            raise ValueError("compact result gradient shape does not match its prepared site table")
        if tensor.dtype != destination.dtype or tensor.device != destination.device:
            raise ValueError("compact result gradients must match global gradient dtype and device")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("compact result gradients must be finite")
    if result.transfer.loss.ndim != 0:
        raise ValueError("compact result loss must be scalar")
    if (
        result.transfer.loss.dtype != ledger.loss.dtype
        or result.transfer.loss.device != ledger.loss.device
        or not bool(torch.isfinite(result.transfer.loss).item())
    ):
        raise ValueError("compact result loss must be finite and match global state")


def _assert_prepared_schedule_matches_ledger(
    ledger: CompactSpatialStepLedger,
    prepared: object,
    *,
    block_id: str,
    legacy_error: str,
    validate_prepared_current: bool = True,
) -> None:
    """Bind blocks by legacy identity or by the template-free schedule digest."""

    if validate_prepared_current:
        prepared.assert_current()
    else:
        # Material-only native training validates topology, geometry, rays,
        # and this schedule through its sealed training capability.  Calling
        # PreparedCompactStagedLieWorld.assert_current here would also compare
        # density/color source versions and incorrectly reject the next
        # optimizer step.  The live ledger owns those refreshed materials.
        prepared.schedule.assert_current()
    ledger.schedule.assert_current()
    if prepared.schedule.global_track_count != ledger.global_track_count:
        raise ValueError("compact spatial block schedule changed the global track count")
    prepared_template = getattr(prepared, "template", None)
    if (prepared_template is None) != (ledger.template is None):
        raise ValueError("compact spatial block mixes legacy and template-free schedule modes")
    if prepared_template is not None:
        if prepared_template is not ledger.template:
            raise ValueError(legacy_error)
        return
    expected = tuple(
        digest
        for expected_block_id, digest in ledger.expected_block_schedule_generations
        if expected_block_id == block_id
    )
    if len(expected) != 1:
        raise ValueError("compact spatial block has no unique registered schedule generation")
    if prepared.schedule.generation_digest != expected[0]:
        raise ValueError("compact spatial block uses a different chart schedule generation")


def _atlas_tensors(atlas: AdaptiveCompiledLieWorldAtlas) -> tuple[torch.Tensor, ...]:
    tensors: list[torch.Tensor] = []
    for chart in atlas.charts:
        tensors.extend(
            (
                chart.transfer_atlas.node_times,
                chart.transfer_atlas.fit_matrix,
                chart.transfer_atlas.coefficients,
                chart.node_chart,
                chart.depth_coefficient_incidence,
                chart.sparse_depth_coefficients,
            )
        )
        for word in chart.words:
            tensors.extend((word.owners, word.left_cut_ids, word.right_cut_ids))
    return tuple(tensors)


def _prepared_block_words(
    topology: PreparedWorldFoamTrackBlock,
) -> tuple[StableCellWord, ...]:
    """Decode row-local CSR incidences into compact boundary ids."""

    words = []
    for track_id in range(topology.track_count):
        word_start = int(topology.word_offsets_i32[track_id].item())
        word_end = int(topology.word_offsets_i32[track_id + 1].item())
        incidence_start = int(topology.track_incidence_offsets_i32[track_id].item())
        incidence_end = int(topology.track_incidence_offsets_i32[track_id + 1].item())
        row_boundaries = topology.incidence_boundary_i32[incidence_start:incidence_end]

        def decode(cut_id: int, boundaries: torch.Tensor = row_boundaries) -> int:
            if cut_id < 0:
                return cut_id
            if cut_id >= int(boundaries.numel()):
                raise ValueError("prepared word cut escaped its row incidence table")
            return int(boundaries[cut_id].item())

        words.append(
            make_stable_cell_word(
                topology.word_owner_i32[word_start:word_end].to(dtype=torch.int64),
                [decode(int(value)) for value in topology.word_left_incidence_i32[word_start:word_end].tolist()],
                [decode(int(value)) for value in topology.word_right_incidence_i32[word_start:word_end].tolist()],
            )
        )
    return tuple(words)


def _topology_tensors(
    topology: PreparedWorldFoamTrackBlock,
) -> tuple[torch.Tensor, ...]:
    return (
        topology.source_track_ids,
        topology.source_boundary_ids,
        topology.source_site_ids,
        topology.word_offsets_i32,
        topology.word_owner_i32,
        topology.word_left_incidence_i32,
        topology.word_right_incidence_i32,
        topology.track_incidence_offsets_i32,
        topology.incidence_boundary_i32,
        topology.boundary_site_pairs_i32,
    )


def _validate_piecewise_atlas(atlas: AdaptiveCompiledLieWorldAtlas) -> None:
    if not atlas.charts:
        raise ValueError("adaptive atlas must contain at least one chart")
    first = atlas.charts[0]
    for previous, chart in zip(atlas.charts[:-1], atlas.charts[1:], strict=True):
        if previous.transfer_atlas.t_max != chart.transfer_atlas.t_min:
            raise ValueError("adaptive atlas charts must be ordered and exactly contiguous")
    for chart in atlas.charts:
        if not _words_have_same_topology(chart.words, first.words):
            raise ValueError("adaptive atlas charts must share one fixed cell word per track")
        if not torch.equal(chart.depth_coefficient_incidence, first.depth_coefficient_incidence):
            raise ValueError("adaptive atlas charts must share sparse incidence ordering")
        if not torch.equal(chart.sparse_depth_coefficients, first.sparse_depth_coefficients):
            raise ValueError("adaptive atlas charts must share one refreshed sparse world state")


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        tensor.untyped_storage().data_ptr(),
        tensor._version,
        tuple(tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
    )


__all__ = [
    "CompactSpatialGradientBuffers",
    "CompactSpatialStepLedger",
    "CompactSpatialStepResult",
    "CompactStagedLieWorldVJP",
    "PreparedCompactStagedLieWorld",
    "PreparedStagedLieWorld",
    "StagedPiecewiseLieMSEAccumulator",
    "accumulate_staged_piecewise_lie_mse",
    "allocate_compact_spatial_gradient_buffers",
    "begin_compact_spatial_step",
    "begin_compact_spatial_step_v2",
    "begin_staged_piecewise_lie_mse",
    "consume_compact_spatial_block_result",
    "finalize_compact_staged_lie_world_vjp",
    "finalize_compact_spatial_step",
    "finalize_staged_piecewise_lie_world_vjp",
    "prepare_compact_staged_lie_world_snapshot",
    "prepare_compact_staged_lie_world_snapshot_v2",
    "refresh_staged_lie_world_snapshot",
    "slice_adaptive_lie_world_atlas_tracks",
]
