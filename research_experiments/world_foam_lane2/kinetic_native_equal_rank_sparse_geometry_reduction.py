"""Certificate-backed sparse geometry reduction for kinetic native blocks.

The legacy geometry bridge is intentionally retained as a dense correctness
oracle.  It re-validates every owner against every site and emits one global
``[S,...]`` result per CSR row.  This production-shaped bridge instead trusts
the immutable continuous owner-chart certificate produced by the kinetic
compiler, differentiates only adjacent certified cuts, accumulates into the
block's compact owner union, and leaves one explicit compact-to-global scatter
to its caller.

Only one native ``[J,W]`` length bar exists at a time.  The CPU fallback copies
one ``[J,R_row]`` slice at a time, so it never creates a second full-block
``[J,W]`` bar.  This is an oracle/bridge, not the desired final ABI: the native
node-word reverse should fuse ``bar_ell`` production with adjacent-cut kinetic
scatter, carrying only the previous/current run bars and eliminating the
``[J,W]`` allocation entirely.  That kernel can follow the existing sparse
incidence -> boundary -> site-finalize pattern; the missing stage is the
kinetic affine-position/velocity/polynomial-weight frontend.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import torch
from kinetic_native_equal_rank_geometry_reduction import (
    kinetic_native_equal_rank_vjp_provenance_id,
)
from kinetic_native_equal_rank_lowering import (
    KineticNativeEqualRankBlockSpec,
    KineticNativeEqualRankChartSource,
    KineticNativeEqualRankRowSpec,
)
from kinetic_native_equal_rank_runtime_adapter import (
    KineticNativeEqualRankVJPResult,
)
from kinetic_stable_stratum_vjp import StableStratumThresholds
from paper_kinetic_ragged_sample_plan import (
    PaperKineticRowBinding,
    PaperKineticRowRaggedSampler,
)

DTYPE = torch.float64
SPARSE_GEOMETRY_REDUCTION_PROVENANCE = (
    "kinetic-native-equal-rank-certified-sparse-geometry-reduction-v1"
)
SPARSE_GEOMETRY_DERIVATIVE_SCOPE = (
    "certified_owner_topology_chart_rank_node_times_adjacent_cuts_only"
)

_RESULT_SEAL = object()


@dataclass(frozen=True)
class KineticNativeSparseGeometryMargins:
    """Node-local numerical margins; owner identity comes from the certificate."""

    minimum_absolute_cut_denominator: float
    minimum_cut_cosine: float
    minimum_coordinate_length: float
    minimum_physical_length: float
    minimum_ray_speed: float
    maximum_active_tie_residual: float
    maximum_exact_transition_depth_residual: float


@dataclass
class _MarginAccumulator:
    minimum_absolute_cut_denominator: float = math.inf
    minimum_cut_cosine: float = math.inf
    minimum_coordinate_length: float = math.inf
    minimum_physical_length: float = math.inf
    minimum_ray_speed: float = math.inf
    maximum_active_tie_residual: float = 0.0
    maximum_exact_transition_depth_residual: float = 0.0

    def freeze(self) -> KineticNativeSparseGeometryMargins:
        return KineticNativeSparseGeometryMargins(
            minimum_absolute_cut_denominator=self.minimum_absolute_cut_denominator,
            minimum_cut_cosine=self.minimum_cut_cosine,
            minimum_coordinate_length=self.minimum_coordinate_length,
            minimum_physical_length=self.minimum_physical_length,
            minimum_ray_speed=self.minimum_ray_speed,
            maximum_active_tie_residual=self.maximum_active_tie_residual,
            maximum_exact_transition_depth_residual=(
                self.maximum_exact_transition_depth_residual
            ),
        )


@dataclass(frozen=True)
class KineticNativeSparseGeometryReductionMemory:
    """Selected logical bytes; this is not an allocator-peak measurement."""

    requested_frame_count: int
    native_full_length_bar_tensor_bytes: int
    cpu_full_length_bar_copy_tensor_bytes: int
    maximum_cpu_row_length_bar_tensor_bytes: int
    compact_source_site_id_tensor_bytes: int
    grad_compact_positions0_tensor_bytes: int
    grad_compact_velocities_tensor_bytes: int
    grad_compact_weight_coefficients_tensor_bytes: int
    grad_track_ray_coefficients_tensor_bytes: int
    output_compact_parameter_bar_tensor_bytes: int
    maximum_row_source_tensor_bytes: int
    maximum_row_parameter_bar_tensor_bytes: int
    maximum_node_scratch_tensor_bytes: int
    maximum_validation_scratch_tensor_bytes: int
    bridge_visible_peak_logical_tensor_bytes: int
    maximum_simultaneous_jw_length_bar_tensors: int
    persistent_frame_tensor_bytes: int
    persistent_sample_tensor_bytes: int
    persistent_target_tensor_bytes: int
    persistent_prediction_tensor_bytes: int
    persistent_material_tensor_bytes: int
    dense_global_geometry_bar_tensor_bytes: int
    allocator_storage_bytes_measured: bool
    allocator_peak_measured: bool
    python_object_bytes_measured: bool


@dataclass(frozen=True)
class KineticNativeEqualRankSparseGeometryReduction:
    """Compact CPU geometry bars from one fenced native equal-rank block."""

    sampler_generation_digest: str
    native_block_generation_digest: str
    native_world_generation_id: str
    native_vjp_provenance_id: str
    device_completion_fence_provenance: str
    view_index: int
    track_ids: tuple[int, ...]
    ray_gradients_included: bool
    ray_bar_keys: tuple[tuple[int, int], ...]
    continuous_topology_certificate_digests: tuple[str, ...]
    source_site_ids_i64: torch.Tensor = field(repr=False)
    grad_compact_positions0_f64: torch.Tensor = field(repr=False)
    grad_compact_velocities_f64: torch.Tensor = field(repr=False)
    grad_compact_weight_coefficients_f64: torch.Tensor = field(repr=False)
    grad_track_ray_coefficients_f64: torch.Tensor = field(repr=False)
    row_count: int
    node_count: int
    word_count: int
    global_site_count: int
    compact_site_count: int
    weight_coefficient_count: int
    row_geometry_vjp_call_count: int
    device_completion_fence_call_count: int
    native_length_bar_row_copy_count: int
    differentiable_word_reverse_interactions: int
    active_cut_node_interactions: int
    adjacency_cut_validation_evaluations: int
    compact_owner_scatter_rows: int
    dense_global_site_accumulation_elements: int
    all_site_owner_validation_evaluations: int
    maximum_simultaneous_jw_length_bar_tensors: int
    margins: KineticNativeSparseGeometryMargins
    memory: KineticNativeSparseGeometryReductionMemory
    accounting: MappingProxyType
    generation_digest: str
    tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    tensor_content_digests: tuple[str, ...] = field(repr=False)
    provenance: str = SPARSE_GEOMETRY_REDUCTION_PROVENANCE
    derivative_scope: str = SPARSE_GEOMETRY_DERIVATIVE_SCOPE
    geometry_reduction_mode: str = "certified_sparse_compact"
    requested_frame_sampling_used: bool = False
    frame_or_sample_axis_retained: bool = False
    target_prediction_or_material_retained: bool = False
    native_result_retained: bool = False
    native_world_or_runtime_retained: bool = False
    row_scratch_retained: bool = False
    geometry_vjp_implemented: bool = True
    continuous_owner_topology_certificate_consumed: bool = True
    all_site_owner_revalidation_performed: bool = False
    event_time_derivatives_included: bool = False
    chart_endpoint_derivatives_included: bool = False
    node_time_or_rank_derivatives_included: bool = False
    compiler_choice_derivatives_included: bool = False
    native_kinetic_sparse_finalize_implemented: bool = False
    allocator_peak_measured: bool = False
    _seal: object = field(default=None, repr=False)

    @property
    def track_count(self) -> int:
        return len(self.track_ids)

    @property
    def output_compact_parameter_bar_tensor_bytes(self) -> int:
        return _tensor_bytes(self._tensors())

    def _tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.source_site_ids_i64,
            self.grad_compact_positions0_f64,
            self.grad_compact_velocities_f64,
            self.grad_compact_weight_coefficients_f64,
            self.grad_track_ray_coefficients_f64,
        )

    def memory_report(
        self,
        requested_frame_count: int,
    ) -> KineticNativeSparseGeometryReductionMemory:
        _require_positive_int(requested_frame_count, name="requested_frame_count")
        return KineticNativeSparseGeometryReductionMemory(
            **{
                **self.memory.__dict__,
                "requested_frame_count": requested_frame_count,
            }
        )

    def assert_current(self) -> None:
        expected_ray_keys = (
            tuple((self.view_index, track_id) for track_id in self.track_ids)
            if self.ray_gradients_included
            else ()
        )
        expected_active_cuts = self.node_count * (self.word_count - self.row_count)
        if (
            self._seal is not _RESULT_SEAL
            or self.provenance != SPARSE_GEOMETRY_REDUCTION_PROVENANCE
            or self.derivative_scope != SPARSE_GEOMETRY_DERIVATIVE_SCOPE
            or self.geometry_reduction_mode != "certified_sparse_compact"
            or self.requested_frame_sampling_used
            or self.frame_or_sample_axis_retained
            or self.target_prediction_or_material_retained
            or self.native_result_retained
            or self.native_world_or_runtime_retained
            or self.row_scratch_retained
            or not self.geometry_vjp_implemented
            or not self.continuous_owner_topology_certificate_consumed
            or self.all_site_owner_revalidation_performed
            or self.event_time_derivatives_included
            or self.chart_endpoint_derivatives_included
            or self.node_time_or_rank_derivatives_included
            or self.compiler_choice_derivatives_included
            or self.native_kinetic_sparse_finalize_implemented
            or self.allocator_peak_measured
            or self.device_completion_fence_call_count != 1
            or self.row_geometry_vjp_call_count != self.row_count
            or self.native_length_bar_row_copy_count != self.row_count
            or self.row_count < 1
            or self.node_count < 2
            or self.word_count < self.row_count
            or self.global_site_count < self.compact_site_count
            or self.compact_site_count < 1
            or self.weight_coefficient_count < 1
            or self.view_index < 0
            or not self.track_ids
            or tuple(sorted(set(self.track_ids))) != self.track_ids
            or self.ray_bar_keys != expected_ray_keys
            or len(self.continuous_topology_certificate_digests) != self.row_count
            or self.differentiable_word_reverse_interactions
            != self.node_count * self.word_count
            or self.active_cut_node_interactions != expected_active_cuts
            or self.adjacency_cut_validation_evaluations != expected_active_cuts
            or self.compact_owner_scatter_rows != self.word_count
            or self.dense_global_site_accumulation_elements != 0
            or self.all_site_owner_validation_evaluations != 0
            or self.maximum_simultaneous_jw_length_bar_tensors != 1
            or self.memory.maximum_simultaneous_jw_length_bar_tensors != 1
            or self.memory.cpu_full_length_bar_copy_tensor_bytes != 0
            or self.memory.dense_global_geometry_bar_tensor_bytes != 0
            or self.memory.maximum_validation_scratch_tensor_bytes < 1
        ):
            raise ValueError("certified sparse geometry reduction contract changed")
        for name, value in (
            ("sampler_generation_digest", self.sampler_generation_digest),
            ("native_block_generation_digest", self.native_block_generation_digest),
            ("native_vjp_provenance_id", self.native_vjp_provenance_id),
        ):
            _require_sha256(value, name=name)
        for digest in self.continuous_topology_certificate_digests:
            _require_sha256(digest, name="continuous_topology_certificate_digest")
        _require_provenance(
            self.native_world_generation_id,
            name="native_world_generation_id",
        )
        _require_provenance(
            self.device_completion_fence_provenance,
            name="device_completion_fence_provenance",
        )
        tensors = self._tensors()
        if tuple(_tensor_signature(tensor) for tensor in tensors) != self.tensor_signatures:
            raise ValueError("certified sparse geometry result tensor identity/layout changed")
        _require_cpu_tensor(
            self.source_site_ids_i64,
            dtype=torch.int64,
            shape=(self.compact_site_count,),
        )
        expected_shapes = (
            (self.compact_site_count, 3),
            (self.compact_site_count, 3),
            (self.compact_site_count, self.weight_coefficient_count),
            (self.track_count if self.ray_gradients_included else 0, 12),
        )
        for tensor, shape in zip(tensors[1:], expected_shapes, strict=True):
            _require_cpu_tensor(tensor, dtype=torch.float64, shape=shape)
        source_ids = tuple(int(value) for value in self.source_site_ids_i64.tolist())
        if (
            source_ids != tuple(sorted(set(source_ids)))
            or source_ids[0] < 0
            or source_ids[-1] >= self.global_site_count
        ):
            raise ValueError("certified sparse geometry compact source ids changed")
        if tuple(_tensor_digest(tensor) for tensor in tensors) != self.tensor_content_digests:
            raise ValueError("certified sparse geometry result tensor content changed")
        if (
            self.output_compact_parameter_bar_tensor_bytes
            != self.memory.output_compact_parameter_bar_tensor_bytes
        ):
            raise ValueError("certified sparse geometry output-byte accounting changed")
        if dict(self.accounting) != _result_accounting(self):
            raise ValueError("certified sparse geometry accounting changed")
        if self.generation_digest != _result_digest(self):
            raise ValueError("certified sparse geometry generation changed")


@torch.no_grad()
def reduce_kinetic_native_equal_rank_sparse_geometry_vjp(
    native_vjp: KineticNativeEqualRankVJPResult,
    sampler: PaperKineticRowRaggedSampler,
    *,
    expected_native_vjp_provenance_id: str,
    device_completion_fence: Callable[[], None],
    device_completion_fence_provenance: str,
    maximum_bridge_visible_peak_logical_tensor_bytes: int,
    include_ray_gradients: bool = True,
    thresholds: StableStratumThresholds = StableStratumThresholds(),
) -> KineticNativeEqualRankSparseGeometryReduction:
    """Reduce one native length bar through certified adjacent cuts only."""

    if not isinstance(native_vjp, KineticNativeEqualRankVJPResult):
        raise TypeError("native_vjp must be KineticNativeEqualRankVJPResult")
    if not isinstance(sampler, PaperKineticRowRaggedSampler):
        raise TypeError("sampler must be PaperKineticRowRaggedSampler")
    if not isinstance(thresholds, StableStratumThresholds):
        raise TypeError("thresholds must be StableStratumThresholds")
    if not isinstance(include_ray_gradients, bool):
        raise TypeError("include_ray_gradients must be bool")
    if not callable(device_completion_fence):
        raise TypeError("device_completion_fence must be callable")
    _require_sha256(
        expected_native_vjp_provenance_id,
        name="expected_native_vjp_provenance_id",
    )
    _require_provenance(
        device_completion_fence_provenance,
        name="device_completion_fence_provenance",
    )
    _require_positive_int(
        maximum_bridge_visible_peak_logical_tensor_bytes,
        name="maximum_bridge_visible_peak_logical_tensor_bytes",
    )

    # Warm checks inspect only immutable metadata and tensor identity/layout.
    native_vjp.assert_warm_layout()
    actual_native_provenance = kinetic_native_equal_rank_vjp_provenance_id(native_vjp)
    if actual_native_provenance != expected_native_vjp_provenance_id:
        raise ValueError("native equal-rank VJP provenance changed")
    row_bindings, row_specs, block = _block_rows(
        sampler,
        native_vjp.world.runtime.payload.block.generation_digest,
    )
    # Recompute the narrow continuous certificate before the encompassing
    # lowering descriptor seal.  This preserves the actionable certificate
    # failure (and still happens before any completion fence) instead of
    # collapsing a certificate-only tamper into a generic descriptor error.
    certificate_digests = tuple(
        _validate_continuous_owner_certificate(binding, row)
        for binding, row in zip(row_bindings, row_specs, strict=True)
    )
    sampler.assert_warm_layout()
    _validate_block_mapping(
        native_vjp,
        sampler,
        block=block,
        row_bindings=row_bindings,
        row_specs=row_specs,
    )
    sites = row_bindings[0].program.binding.sites
    track_ids = tuple(sorted({binding.track_id for binding in row_bindings}))
    weight_count = int(sites.weight_coefficients.shape[1])
    memory = preflight_kinetic_native_equal_rank_sparse_geometry_reduction_memory(
        sampler,
        block_generation_digest=block.generation_digest,
        include_ray_gradients=include_ray_gradients,
    )
    if (
        memory.bridge_visible_peak_logical_tensor_bytes
        > maximum_bridge_visible_peak_logical_tensor_bytes
    ):
        raise MemoryError("certified sparse geometry reduction exceeds its preflight byte budget")

    fence_result = device_completion_fence()
    if fence_result is not None:
        raise TypeError("device_completion_fence must return None")
    native_vjp.assert_warm_layout()
    sampler.assert_warm_layout()
    if kinetic_native_equal_rank_vjp_provenance_id(native_vjp) != actual_native_provenance:
        raise ValueError("native equal-rank VJP changed across the completion fence")

    source_site_ids = torch.tensor(block.source_site_ids, dtype=torch.int64)
    source_to_compact = {
        site_id: compact_id for compact_id, site_id in enumerate(block.source_site_ids)
    }
    compact_positions0 = torch.zeros(
        (len(block.source_site_ids), 3),
        dtype=DTYPE,
    )
    compact_velocities = torch.zeros_like(compact_positions0)
    compact_weights = torch.zeros(
        (len(block.source_site_ids), weight_count),
        dtype=DTYPE,
    )
    ray_bars = torch.zeros(
        (len(track_ids) if include_ray_gradients else 0, 12),
        dtype=DTYPE,
    )
    track_to_compact = (
        {track_id: index for index, track_id in enumerate(track_ids)}
        if include_ray_gradients
        else {}
    )
    margins = _MarginAccumulator()

    word_start = 0
    for binding, row in zip(row_bindings, row_specs, strict=True):
        word_end = word_start + row.word_count
        # This is deliberately a row copy, never a second [J,W_block] copy.
        grad_row_lengths = (
            native_vjp.grad_node_physical_length_f32[:, word_start:word_end]
            .detach()
            .to(device="cpu", dtype=DTYPE)
            .contiguous()
        )
        if not bool(torch.isfinite(grad_row_lengths).all().item()):
            raise ValueError("native sparse geometry row length bar is nonfinite")
        (
            row_positions0,
            row_velocities,
            row_weights,
            row_ray,
        ) = _reduce_certified_row(
            binding,
            row,
            grad_row_lengths,
            thresholds=thresholds,
            margins=margins,
        )
        row_compact_ids = torch.tensor(
            [source_to_compact[site_id] for site_id in row.owner_word],
            dtype=torch.int64,
        )
        compact_positions0.index_add_(0, row_compact_ids, row_positions0)
        compact_velocities.index_add_(0, row_compact_ids, row_velocities)
        compact_weights.index_add_(0, row_compact_ids, row_weights)
        if include_ray_gradients:
            ray_bars[track_to_compact[binding.track_id]].add_(row_ray)
        word_start = word_end
        del grad_row_lengths, row_positions0, row_velocities, row_weights, row_ray
    if word_start != block.word_count:
        raise ValueError("certified sparse reduction did not consume the native CSR word bar")

    tensors = (
        source_site_ids,
        compact_positions0,
        compact_velocities,
        compact_weights,
        ray_bars,
    )
    if any(not bool(torch.isfinite(tensor).all().item()) for tensor in tensors[1:]):
        raise ValueError("certified sparse geometry reduction produced nonfinite bars")
    provisional = KineticNativeEqualRankSparseGeometryReduction(
        sampler_generation_digest=sampler.generation_digest,
        native_block_generation_digest=block.generation_digest,
        native_world_generation_id=native_vjp.world.generation_id,
        native_vjp_provenance_id=expected_native_vjp_provenance_id,
        device_completion_fence_provenance=device_completion_fence_provenance,
        view_index=sampler.view_index,
        track_ids=track_ids,
        ray_gradients_included=include_ray_gradients,
        ray_bar_keys=(
            tuple((sampler.view_index, track_id) for track_id in track_ids)
            if include_ray_gradients
            else ()
        ),
        continuous_topology_certificate_digests=certificate_digests,
        source_site_ids_i64=source_site_ids,
        grad_compact_positions0_f64=compact_positions0,
        grad_compact_velocities_f64=compact_velocities,
        grad_compact_weight_coefficients_f64=compact_weights,
        grad_track_ray_coefficients_f64=ray_bars,
        row_count=len(row_bindings),
        node_count=block.node_count,
        word_count=block.word_count,
        global_site_count=sites.site_count,
        compact_site_count=len(block.source_site_ids),
        weight_coefficient_count=weight_count,
        row_geometry_vjp_call_count=len(row_bindings),
        device_completion_fence_call_count=1,
        native_length_bar_row_copy_count=len(row_bindings),
        differentiable_word_reverse_interactions=block.node_count * block.word_count,
        active_cut_node_interactions=(
            block.node_count * (block.word_count - block.row_count)
        ),
        adjacency_cut_validation_evaluations=(
            block.node_count * (block.word_count - block.row_count)
        ),
        compact_owner_scatter_rows=block.word_count,
        dense_global_site_accumulation_elements=0,
        all_site_owner_validation_evaluations=0,
        maximum_simultaneous_jw_length_bar_tensors=1,
        margins=margins.freeze(),
        memory=memory,
        accounting=MappingProxyType({}),
        generation_digest="",
        tensor_signatures=tuple(_tensor_signature(tensor) for tensor in tensors),
        tensor_content_digests=tuple(_tensor_digest(tensor) for tensor in tensors),
        _seal=_RESULT_SEAL,
    )
    with_accounting = replace(
        provisional,
        accounting=MappingProxyType(_result_accounting(provisional)),
    )
    result = replace(
        with_accounting,
        generation_digest=_result_digest(with_accounting),
    )
    result.assert_current()
    return result


def preflight_kinetic_native_equal_rank_sparse_geometry_reduction_memory(
    sampler: PaperKineticRowRaggedSampler,
    *,
    block_generation_digest: str,
    include_ray_gradients: bool = True,
) -> KineticNativeSparseGeometryReductionMemory:
    """Return the reducer's deterministic logical-tensor upper bound.

    This is the admission-time counterpart of ``result.memory``.  It exposes
    the source-visible bridge calculation before a native length VJP is
    launched and allocates no tensor. A higher-level request can therefore
    compose the reduction phase with the state that remains live around it.
    Conservative scratch terms are included; allocator and Python storage are
    not measured.
    """

    if not isinstance(sampler, PaperKineticRowRaggedSampler):
        raise TypeError("sampler must be PaperKineticRowRaggedSampler")
    if not isinstance(include_ray_gradients, bool):
        raise TypeError("include_ray_gradients must be bool")
    _require_sha256(
        block_generation_digest,
        name="block_generation_digest",
    )
    sampler.assert_warm_layout()
    row_bindings, row_specs, block = _block_rows(
        sampler,
        block_generation_digest,
    )
    sites = row_bindings[0].program.binding.sites
    if any(binding.program.binding.sites is not sites for binding in row_bindings):
        raise ValueError("one sparse native block cannot mix kinetic site tables")
    return _preflight_memory(
        block,
        compact_site_count=len(block.source_site_ids),
        weight_coefficient_count=int(sites.weight_coefficients.shape[1]),
        track_count=len({binding.track_id for binding in row_bindings}),
        include_ray_gradients=include_ray_gradients,
        maximum_row_word_count=max(row.word_count for row in row_specs),
    )


def _reduce_certified_row(
    binding: PaperKineticRowBinding,
    row: KineticNativeEqualRankRowSpec,
    grad_lengths: torch.Tensor,
    *,
    thresholds: StableStratumThresholds,
    margins: _MarginAccumulator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate one word using only its adjacent certified incidences."""

    program = binding.program
    chart = program.charts[binding.chart_index]
    sites = program.binding.sites
    owner_ids = torch.tensor(row.owner_word, dtype=torch.int64)
    positions0 = sites.positions0.index_select(0, owner_ids)
    velocities = sites.velocities.index_select(0, owner_ids)
    weight_coefficients = sites.weight_coefficients.index_select(0, owner_ids)
    ray = program.binding.ray_coefficients
    if tuple(ray.shape) != (12,) or tuple(grad_lengths.shape) != (
        row.node_count,
        row.word_count,
    ):
        raise ValueError("certified sparse row has incompatible ray/length-bar shapes")
    if len(chart.exact_node_transition_depths) != row.node_count:
        raise ValueError("certified sparse row lost exact node transition depths")

    grad_positions0 = torch.zeros_like(positions0)
    grad_velocities = torch.zeros_like(velocities)
    grad_weights = torch.zeros_like(weight_coefficients)
    grad_ray = torch.zeros_like(ray)
    coefficient_count = int(weight_coefficients.shape[1])

    for node_id, time in enumerate(chart.schedule.node_times):
        time_powers = torch.stack((torch.ones_like(time), time, time.square()))[
            :coefficient_count
        ]
        positions = positions0 + time * velocities
        weights = weight_coefficients @ time_powers
        origin = ray[:3] + time * ray[3:6]
        direction = ray[6:9] + time * ray[9:12]
        speed = torch.linalg.vector_norm(direction)
        speed_value = float(speed.item())
        margins.minimum_ray_speed = min(margins.minimum_ray_speed, speed_value)
        if speed_value <= thresholds.minimum_ray_speed:
            raise ValueError("certified sparse geometry ray-speed margin failed")

        cut_depths = [torch.as_tensor(row.near, dtype=DTYPE)]
        denominators: list[torch.Tensor] = []
        normals: list[torch.Tensor] = []
        exact_depths = chart.exact_node_transition_depths[node_id]
        if len(exact_depths) != row.word_count - 1:
            raise ValueError("certified sparse row transition rank changed")
        for cut_id in range(row.word_count - 1):
            left = cut_id
            right = cut_id + 1
            normal = 2.0 * (positions[right] - positions[left])
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
                raise ValueError("certified sparse geometry cut denominator margin failed")
            if cosine <= thresholds.minimum_cut_cosine:
                raise ValueError("certified sparse geometry cut cosine margin failed")
            intercept = (
                torch.dot(normal, origin)
                + torch.dot(positions[left], positions[left])
                - torch.dot(positions[right], positions[right])
                - weights[left]
                + weights[right]
            )
            depth = -intercept / denominator
            if not bool(torch.isfinite(depth).item()):
                raise ValueError("certified sparse geometry cut depth is nonfinite")
            exact_depth = float(exact_depths[cut_id])
            exact_residual = abs(float(depth.item()) - exact_depth)
            margins.maximum_exact_transition_depth_residual = max(
                margins.maximum_exact_transition_depth_residual,
                exact_residual,
            )
            if exact_residual > 2.0e-10 * max(1.0, abs(exact_depth)):
                raise ValueError("certified sparse geometry disagrees with exact transition depth")
            tie_residual = abs(float((intercept + depth * denominator).item()))
            tie_scale = max(1.0, abs(float(intercept.item())))
            margins.maximum_active_tie_residual = max(
                margins.maximum_active_tie_residual,
                tie_residual,
            )
            if tie_residual > thresholds.active_tie_tolerance * tie_scale:
                raise ValueError("certified sparse geometry active-cut tie residual failed")
            cut_depths.append(depth)
            denominators.append(denominator)
            normals.append(normal)
        cut_depths.append(torch.as_tensor(row.far, dtype=DTYPE))
        cuts = torch.stack(cut_depths)
        coordinate_lengths = cuts[1:] - cuts[:-1]
        physical_lengths = speed * coordinate_lengths
        if not torch.allclose(
            physical_lengths,
            chart.node_physical_lengths[node_id],
            rtol=2.0e-6,
            atol=2.0e-7,
        ):
            raise ValueError(
                "certified sparse row recompute disagrees with compiled native lengths"
            )
        minimum_coordinate = float(coordinate_lengths.min().item())
        minimum_physical = float(physical_lengths.min().item())
        margins.minimum_coordinate_length = min(
            margins.minimum_coordinate_length,
            minimum_coordinate,
        )
        margins.minimum_physical_length = min(
            margins.minimum_physical_length,
            minimum_physical,
        )
        if minimum_coordinate <= thresholds.minimum_coordinate_length:
            raise ValueError("certified sparse geometry coordinate-length margin failed")
        if minimum_physical <= thresholds.minimum_physical_length:
            raise ValueError("certified sparse geometry physical-length margin failed")

        coordinate_length_bars = speed * grad_lengths[node_id]
        cut_bars = coordinate_length_bars[:-1] - coordinate_length_bars[1:]
        speed_bar = torch.dot(coordinate_lengths, grad_lengths[node_id])
        position_bars = torch.zeros_like(positions)
        weight_bars = torch.zeros_like(weights)
        origin_bar = torch.zeros_like(origin)
        direction_bar = speed_bar * direction / speed
        for cut_id, cut_bar in enumerate(cut_bars):
            left = cut_id
            right = cut_id + 1
            depth = cuts[cut_id + 1]
            point = origin + depth * direction
            implicit_bar = -cut_bar / denominators[cut_id]
            position_bars[left] += implicit_bar * 2.0 * (positions[left] - point)
            position_bars[right] += implicit_bar * 2.0 * (point - positions[right])
            weight_bars[left] -= implicit_bar
            weight_bars[right] += implicit_bar
            origin_bar += implicit_bar * normals[cut_id]
            direction_bar += implicit_bar * depth * normals[cut_id]
        grad_positions0.add_(position_bars)
        grad_velocities.add_(time * position_bars)
        grad_weights.add_(weight_bars[:, None] * time_powers[None, :])
        grad_ray[:3].add_(origin_bar)
        grad_ray[3:6].add_(time * origin_bar)
        grad_ray[6:9].add_(direction_bar)
        grad_ray[9:12].add_(time * direction_bar)

    return grad_positions0, grad_velocities, grad_weights, grad_ray


def _validate_continuous_owner_certificate(
    binding: PaperKineticRowBinding,
    row: KineticNativeEqualRankRowSpec,
) -> str:
    return validate_kinetic_native_equal_rank_continuous_owner_certificate(
        binding.source,
        row,
    )


def validate_kinetic_native_equal_rank_continuous_owner_certificate(
    source: KineticNativeEqualRankChartSource,
    row: KineticNativeEqualRankRowSpec,
) -> str:
    """Revalidate one compiler-bound owner-topology certificate digest.

    This public cold validator is shared by the staged sparse oracle and the
    unpromoted fused fixed-camera adapter.  It accepts the sealed chart source,
    not a caller-supplied digest, so admission is bound to live program,
    topology, schedule, and row identities.  It recomputes the digest from
    sealed live metadata; it does not rerun the active all-site compiler.
    """

    if not isinstance(row, KineticNativeEqualRankRowSpec):
        raise TypeError("row must be KineticNativeEqualRankRowSpec")
    if not isinstance(source, KineticNativeEqualRankChartSource):
        raise TypeError("source must be KineticNativeEqualRankChartSource")
    program = source.program
    lowering = source.lowering
    chart_index = source.chart_index
    if (
        source.row_identity != row.row_identity
        or chart_index != row.chart_index
        or not 0 <= chart_index < program.chart_count
        or not 0 <= chart_index < lowering.chart_count
    ):
        raise ValueError("continuous owner-topology source/row identity changed")
    owner_program = program.binding.program
    chart = program.charts[chart_index]
    topology = lowering.charts[chart_index]
    expected = _digest_parts(
        "kinetic-native-owner-topology-certificate-v1",
        program.binding.source_content_digest,
        program.binding.program_semantic_digest,
        program.generation_digest,
        chart.chart_id,
        chart.owner_word,
        chart.schedule.t_min,
        chart.schedule.t_max,
        chart.right_closed,
    )
    if topology.owner_topology_certificate_digest != expected:
        raise ValueError("continuous owner-topology certificate digest mismatch")
    source.assert_current()
    if (
        not owner_program.passed
        or not owner_program.continuous_time_coverage
        or not owner_program.owner_identity_certified
        or owner_program.unresolved_degeneracies
        or not chart.exact_owner_and_cut_discovery_at_nodes
        or not chart.safe_interval_is_certified_inside_owner_chart
        or tuple(chart.owner_word) != row.owner_word
        or topology.owner_word != row.owner_word
        or topology.node_physical_lengths_digest != row.node_physical_lengths_digest
        or topology.payload_digest != row.chart_payload_digest
    ):
        raise ValueError("continuous owner-topology certificate contract changed")
    _require_sha256(expected, name="continuous_topology_certificate_digest")
    return expected


def _block_rows(
    sampler: PaperKineticRowRaggedSampler,
    block_generation_digest: str,
) -> tuple[
    tuple[PaperKineticRowBinding, ...],
    tuple[KineticNativeEqualRankRowSpec, ...],
    KineticNativeEqualRankBlockSpec,
]:
    bindings = tuple(
        sorted(
            (
                row
                for row in sampler.rows
                if row.native_block_generation_digest == block_generation_digest
            ),
            key=lambda row: row.native_local_row_index,
        )
    )
    blocks = tuple(
        block
        for bucket in sampler.lowering.buckets
        for block in bucket.blocks
        if block.generation_digest == block_generation_digest
    )
    if len(blocks) != 1 or not bindings:
        raise ValueError("native VJP block is foreign to the sparse ragged sampler")
    block = blocks[0]
    rows_by_index = {row.global_row_index: row for row in sampler.lowering.rows}
    rows = tuple(rows_by_index[index] for index in block.global_row_indices)
    return bindings, rows, block


def _validate_block_mapping(
    native_vjp: KineticNativeEqualRankVJPResult,
    sampler: PaperKineticRowRaggedSampler,
    *,
    block: KineticNativeEqualRankBlockSpec,
    row_bindings: tuple[PaperKineticRowBinding, ...],
    row_specs: tuple[KineticNativeEqualRankRowSpec, ...],
) -> None:
    runtime = native_vjp.world.runtime
    native_block = runtime.payload.block
    if (
        native_block.generation_digest != block.generation_digest
        or native_block.source_site_ids != block.source_site_ids
        or len(row_bindings) != block.row_count
        or len(row_specs) != block.row_count
        or tuple(binding.native_local_row_index for binding in row_bindings)
        != tuple(range(block.row_count))
        or tuple(binding.global_row_index for binding in row_bindings)
        != block.global_row_indices
        or tuple(binding.row_identity_digest for binding in row_bindings)
        != block.row_identity_digests
        or tuple(row.global_row_index for row in row_specs)
        != block.global_row_indices
        or sum(row.word_count for row in row_specs) != block.word_count
        or sampler.lowering.global_site_count != runtime.global_site_count
    ):
        raise ValueError("native VJP CSR rows do not match the sparse ragged sampler")
    for binding, row in zip(row_bindings, row_specs, strict=True):
        source = binding.source
        if (
            binding.track_id != row.track_id
            or binding.chart_index != row.chart_index
            or binding.node_count != row.node_count
            or row.node_count != block.node_count
            or source.row_identity != row.row_identity
            or source.program is not binding.program
            or source.program.generation_digest != row.kinetic_program_generation_digest
            or source.lowering.generation_digest
            != row.topology_lowering_generation_digest
        ):
            raise ValueError("sparse native VJP row source/program provenance changed")


def _preflight_memory(
    block: KineticNativeEqualRankBlockSpec,
    *,
    compact_site_count: int,
    weight_coefficient_count: int,
    track_count: int,
    include_ray_gradients: bool,
    maximum_row_word_count: int,
) -> KineticNativeSparseGeometryReductionMemory:
    input_bytes = block.node_count * block.word_count * 4
    maximum_row_copy_bytes = block.node_count * maximum_row_word_count * 8
    source_id_bytes = compact_site_count * 8
    positions_bytes = compact_site_count * 3 * 8
    velocities_bytes = positions_bytes
    weights_bytes = compact_site_count * weight_coefficient_count * 8
    rays_bytes = track_count * 12 * 8 if include_ray_gradients else 0
    output_bytes = source_id_bytes + positions_bytes + velocities_bytes + weights_bytes + rays_bytes
    row_parameter_bytes = (
        maximum_row_word_count * (6 + weight_coefficient_count) * 8
        + (12 * 8 if include_ray_gradients else 0)
    )
    # Conservative bridge-visible source payload: owner ids, kinetic site
    # parameters, affine ray, row-local node times, and the compiled node
    # lengths used by the parity check. Exact transition-depth tuples and
    # Python container overhead remain explicitly outside tensor accounting.
    row_source_bytes = (
        maximum_row_word_count * 8
        + maximum_row_word_count * (6 + weight_coefficient_count) * 8
        + 12 * 8
        + block.node_count * 8
        + block.node_count * maximum_row_word_count * 8
    )
    # Overbound the simultaneously live tensor temporaries in one node: local
    # site state, cut depths/denominators/normals, physical/coordinate lengths,
    # their cotangents, and local output bars. This is intentionally wider than
    # the algebraic minimum; allocator/Python-object peaks are still unmeasured.
    node_scratch_bytes = (24 * maximum_row_word_count + 64) * 8
    # ``torch.isfinite(...).all()`` materializes one byte-per-element bool mask
    # plus a scalar. Cover both the largest row-length validation and the final
    # compact-output validation while the input and output bars remain live.
    validation_scratch_bytes = 1 + max(
        block.node_count * maximum_row_word_count,
        compact_site_count * 3,
        compact_site_count * weight_coefficient_count,
        track_count * 12 if include_ray_gradients else 0,
    )
    peak = (
        input_bytes
        + maximum_row_copy_bytes
        + output_bytes
        + row_source_bytes
        + row_parameter_bytes
        + node_scratch_bytes
        + validation_scratch_bytes
    )
    return KineticNativeSparseGeometryReductionMemory(
        requested_frame_count=1,
        native_full_length_bar_tensor_bytes=input_bytes,
        cpu_full_length_bar_copy_tensor_bytes=0,
        maximum_cpu_row_length_bar_tensor_bytes=maximum_row_copy_bytes,
        compact_source_site_id_tensor_bytes=source_id_bytes,
        grad_compact_positions0_tensor_bytes=positions_bytes,
        grad_compact_velocities_tensor_bytes=velocities_bytes,
        grad_compact_weight_coefficients_tensor_bytes=weights_bytes,
        grad_track_ray_coefficients_tensor_bytes=rays_bytes,
        output_compact_parameter_bar_tensor_bytes=output_bytes,
        maximum_row_source_tensor_bytes=row_source_bytes,
        maximum_row_parameter_bar_tensor_bytes=row_parameter_bytes,
        maximum_node_scratch_tensor_bytes=node_scratch_bytes,
        maximum_validation_scratch_tensor_bytes=validation_scratch_bytes,
        bridge_visible_peak_logical_tensor_bytes=peak,
        maximum_simultaneous_jw_length_bar_tensors=1,
        persistent_frame_tensor_bytes=0,
        persistent_sample_tensor_bytes=0,
        persistent_target_tensor_bytes=0,
        persistent_prediction_tensor_bytes=0,
        persistent_material_tensor_bytes=0,
        dense_global_geometry_bar_tensor_bytes=0,
        allocator_storage_bytes_measured=False,
        allocator_peak_measured=False,
        python_object_bytes_measured=False,
    )


def _result_accounting(
    result: KineticNativeEqualRankSparseGeometryReduction,
) -> dict[str, int | str | bool]:
    return {
        "provenance": result.provenance,
        "derivative_scope": result.derivative_scope,
        "geometry_reduction_mode": result.geometry_reduction_mode,
        "view_index": result.view_index,
        "row_count": result.row_count,
        "track_count": result.track_count,
        "node_count": result.node_count,
        "word_count": result.word_count,
        "global_site_count": result.global_site_count,
        "compact_site_count": result.compact_site_count,
        "row_geometry_vjp_call_count": result.row_geometry_vjp_call_count,
        "device_completion_fence_call_count": result.device_completion_fence_call_count,
        "native_length_bar_row_copy_count": result.native_length_bar_row_copy_count,
        "maximum_simultaneous_jw_length_bar_tensors": (
            result.maximum_simultaneous_jw_length_bar_tensors
        ),
        "cpu_full_jw_length_bar_copy_count": 0,
        "differentiable_word_reverse_interactions": (
            result.differentiable_word_reverse_interactions
        ),
        "active_cut_node_interactions": result.active_cut_node_interactions,
        "adjacency_cut_validation_evaluations": (
            result.adjacency_cut_validation_evaluations
        ),
        "compact_owner_scatter_rows": result.compact_owner_scatter_rows,
        "dense_global_site_accumulation_elements": (
            result.dense_global_site_accumulation_elements
        ),
        "all_site_owner_validation_evaluations": (
            result.all_site_owner_validation_evaluations
        ),
        "all_site_owner_revalidation_performed": False,
        "continuous_owner_topology_certificate_consumed": True,
        "continuous_owner_topology_certificate_count": (
            len(result.continuous_topology_certificate_digests)
        ),
        "reverse_scaling": "O(J * W_block + compact_block_parameters)",
        "validation_scaling": (
            "O(J * W_block + S_compact * parameter_width + track_count); "
            "no all-site owner scan"
        ),
        "compact_to_global_scatter_scaling": "O(S_compact * parameter_width)",
        "ray_gradients_included": result.ray_gradients_included,
        "ray_bar_key_kind": (
            "(view_index, track_id)"
            if result.ray_gradients_included
            else "disabled/fixed_camera"
        ),
        "native_result_retained": False,
        "native_world_or_runtime_retained": False,
        "row_scratch_retained": False,
        "requested_frame_count_used": 0,
        "requested_sample_count_used": 0,
        "persistent_frame_tensor_bytes": 0,
        "persistent_sample_tensor_bytes": 0,
        "persistent_target_tensor_bytes": 0,
        "persistent_prediction_tensor_bytes": 0,
        "persistent_material_tensor_bytes": 0,
        "dense_global_geometry_bar_tensor_bytes": 0,
        "output_compact_parameter_bar_tensor_bytes": (
            result.output_compact_parameter_bar_tensor_bytes
        ),
        "bridge_visible_peak_logical_tensor_bytes": (
            result.memory.bridge_visible_peak_logical_tensor_bytes
        ),
        "maximum_validation_scratch_tensor_bytes": (
            result.memory.maximum_validation_scratch_tensor_bytes
        ),
        "future_native_reuse": (
            "sparse_mobius_boundary_finalize_then_sparse_power_boundary_vjp_to_sites"
        ),
        "future_native_row_node_time_shape": "[Q_block,J]",
        "future_native_row_domain_shape": "row_near_far[Q_block,2]",
        "future_native_row_ray_shape": "row_ray_coefficients[Q_block,12]",
        "future_native_owner_index_space": "compact word_owner_i32",
        "future_native_global_scatter_ids": "source_site_ids_i64",
        "equal_rank_implies_equal_node_times": False,
        "preferred_native_endpoint": (
            "fused_node_word_bar_ell_to_adjacent_cut_kinetic_scatter"
        ),
        "preferred_native_full_jw_length_bar_allocated": False,
        "preferred_native_cut_bar_recurrence": (
            "bar_z=speed*(previous_bar_ell-current_bar_ell)"
        ),
        "preferred_native_ray_speed_accumulation": (
            "sum((physical_length/speed)*bar_ell)"
        ),
        "missing_native_stage": (
            "kinetic_affine_position_velocity_polynomial_weight_frontend"
        ),
        "native_kinetic_sparse_finalize_implemented": False,
        "allocator_peak_measured": False,
    }


def _result_digest(result: KineticNativeEqualRankSparseGeometryReduction) -> str:
    return _digest_parts(
        SPARSE_GEOMETRY_REDUCTION_PROVENANCE,
        result.sampler_generation_digest,
        result.native_block_generation_digest,
        result.native_world_generation_id,
        result.native_vjp_provenance_id,
        result.device_completion_fence_provenance,
        result.view_index,
        result.track_ids,
        result.ray_gradients_included,
        result.ray_bar_keys,
        result.continuous_topology_certificate_digests,
        result.row_count,
        result.node_count,
        result.word_count,
        result.global_site_count,
        result.compact_site_count,
        result.weight_coefficient_count,
        result.row_geometry_vjp_call_count,
        result.device_completion_fence_call_count,
        result.native_length_bar_row_copy_count,
        result.differentiable_word_reverse_interactions,
        result.active_cut_node_interactions,
        result.adjacency_cut_validation_evaluations,
        result.compact_owner_scatter_rows,
        result.dense_global_site_accumulation_elements,
        result.all_site_owner_validation_evaluations,
        result.maximum_simultaneous_jw_length_bar_tensors,
        result.margins,
        result.memory,
        tuple(sorted(dict(result.accounting).items())),
        result.tensor_content_digests,
    )


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        id(tensor),
        int(getattr(tensor, "_version", 0)),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        tensor.layout,
        tensor.requires_grad,
        tensor.is_contiguous(),
    )


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(repr(tuple(value.shape)).encode("utf-8"))
    if value.numel():
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def _tensor_bytes(tensors: tuple[torch.Tensor, ...]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _require_cpu_tensor(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device.type != "cpu"
        or tensor.dtype != dtype
        or tensor.layout != torch.strided
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
        or tensor.requires_grad
    ):
        raise ValueError("certified sparse geometry result tensor has an invalid layout")


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _require_sha256(value: str, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")


def _require_provenance(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")


def _require_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


__all__ = [
    "KineticNativeEqualRankSparseGeometryReduction",
    "KineticNativeSparseGeometryMargins",
    "KineticNativeSparseGeometryReductionMemory",
    "SPARSE_GEOMETRY_DERIVATIVE_SCOPE",
    "SPARSE_GEOMETRY_REDUCTION_PROVENANCE",
    "preflight_kinetic_native_equal_rank_sparse_geometry_reduction_memory",
    "reduce_kinetic_native_equal_rank_sparse_geometry_vjp",
    "validate_kinetic_native_equal_rank_continuous_owner_certificate",
]
