"""Row-ragged paper observations for native kinetic WorldFoam reduction.

Kinetic WorldFoam compiles one temporal owner program per pixel track.  The
native material path evaluates equal-rank ``(track_id, chart_index)`` rows,
whereas the paper sampler emits arbitrary view/frame observations and bounded
pixel blocks.  This module is the missing data-side join:

* a cold-prepared sampler binds the canonical rows of one view-local
  :class:`KineticNativeEqualRankLowering`;
* exact right-continuous dispatch selects one chart for every
  ``(pixel, observation time)`` pair, including the terminal closed endpoint;
* selected samples are streamed by native equal-rank block with row-local
  interpolation weights and the one global paper-step MSE scale;
* no global temporal refinement or ``row x global_time`` table is built.

The sampler retains no sample, target, frame, or weight tensor.  A yielded
launch block is ephemeral and bounded by ``maximum_samples_per_launch`` (and
therefore by the caller's ``B_p * K`` target rectangle).  Content checks and
exact dispatch are cold work.  ``assert_warm_layout`` methods inspect only
object identity and tensor identity/layout/mutation version; they perform no
device-to-host copy, scalar extraction, content hash, or tensor allocation.

This is a CPU/source bridge.  It deliberately does not build or launch Metal.
Its tensors match the existing
``prepare_kinetic_ragged_p0_lie_sample_block`` inputs: CPU ``sample_row_i32``
plus device-local float32 ``sample_to_node_f32`` and ``target_rgb_f32``.
Material bars from several equal-rank native blocks are joined by
``paper_kinetic_union_local_bar_assembly``.  Its cold-sealed compact-to-union
maps and one caller-owned union scratch avoid a request-sized global site bar.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import TYPE_CHECKING

import torch
from paper_kinetic_runtime_paths import ensure_worldfoam_lane2_research_path

ensure_worldfoam_lane2_research_path()

from kinetic_multichart_transfer_program import (  # noqa: E402
    KineticMultiChartP0Program,
    dispatch_prevalidated_kinetic_chart_index,
)
from kinetic_native_equal_rank_lowering import (  # noqa: E402
    KineticNativeEqualRankChartSource,
    KineticNativeEqualRankLowering,
)
from paper_ragged_track_staging import PaperRaggedTrackTargetStageBlock  # noqa: E402

if TYPE_CHECKING:
    from paper_ragged_material_bar_coordinator import PaperRaggedMaterialBarRequest


SAMPLER_PROVENANCE = "paper-kinetic-row-ragged-sample-plan-v1"
WARM_VALIDATION_KIND = "identity_shape_stride_dtype_device_version_only"

_SAMPLER_SEAL = object()
_BLOCK_SEAL = object()


@dataclass(frozen=True)
class PaperKineticRowBinding:
    """One canonical global row and its local native-block row."""

    global_row_index: int
    track_id: int
    chart_index: int
    node_count: int
    native_block_generation_digest: str
    native_bucket_block_index: int
    native_local_row_index: int
    row_identity_digest: str
    program_generation_digest: str
    right_closed: bool
    program: KineticMultiChartP0Program = field(repr=False)
    source: KineticNativeEqualRankChartSource = field(repr=False)
    _program_identity: int = field(repr=False)
    _source_identity: int = field(repr=False)

    @property
    def row_identity(self) -> tuple[int, int]:
        return (self.track_id, self.chart_index)

    def assert_warm_identity(self) -> None:
        """Check immutable Python identity only; source tensors are checked once globally."""

        if id(self.program) != self._program_identity or id(self.source) != self._source_identity:
            raise ValueError("kinetic ragged row source identity changed")
        if self.source.program is not self.program:
            raise ValueError("kinetic ragged row no longer references its sealed program")
        if self.program.generation_digest != self.program_generation_digest:
            raise ValueError("kinetic ragged row program generation changed")
        if self.source.row_identity != self.row_identity:
            raise ValueError("kinetic ragged row identity changed")


@dataclass(frozen=True)
class PaperKineticRowRaggedMemoryReport:
    """Persistent sampler accounting, explicitly excluding ephemeral requests."""

    requested_frame_count: int
    descriptor_tensor_bytes: int
    descriptor_canonical_metadata_bytes: int
    source_kinetic_program_tensor_bytes: int
    persistent_sample_tensor_bytes: int
    persistent_target_tensor_bytes: int
    persistent_interpolation_weight_tensor_bytes: int
    dense_row_by_global_time_tensor_bytes: int
    global_common_temporal_refinement_used: bool
    requested_frame_sampling_used_for_compile: bool
    allocator_peak_measured: bool
    descriptor_python_allocator_bytes_measured: bool


@dataclass(frozen=True)
class PaperKineticRowRaggedSampler:
    """Frame-free, cold-provenance-sealed row catalog for one camera view."""

    view_index: int
    lowering: KineticNativeEqualRankLowering = field(repr=False)
    sources: tuple[KineticNativeEqualRankChartSource, ...] = field(repr=False)
    rows: tuple[PaperKineticRowBinding, ...]
    track_ids: tuple[int, ...]
    descriptor_canonical_metadata_bytes: int
    generation_digest: str
    cold_source_tensor_digests: tuple[str, ...] = field(repr=False)
    warm_source_tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    _lowering_identity: int = field(repr=False)
    _source_identities: tuple[int, ...] = field(repr=False)
    provenance: str = SAMPLER_PROVENANCE
    descriptor_tensor_bytes: int = 0
    persistent_sample_tensor_bytes: int = 0
    persistent_target_tensor_bytes: int = 0
    persistent_interpolation_weight_tensor_bytes: int = 0
    dense_row_by_global_time_tensor_bytes: int = 0
    global_common_temporal_refinement_used: bool = False
    requested_frame_sampling_used_for_compile: bool = False
    warm_validation_kind: str = WARM_VALIDATION_KIND
    warm_validation_device_to_host_syncs: int = 0
    warm_validation_tensor_allocations: int = 0
    descriptor_python_allocator_bytes_measured: bool = False
    _seal: object = field(default=None, repr=False)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def memory_report(self, requested_frame_count: int) -> PaperKineticRowRaggedMemoryReport:
        _require_positive_int(requested_frame_count, name="requested_frame_count")
        return PaperKineticRowRaggedMemoryReport(
            requested_frame_count=requested_frame_count,
            descriptor_tensor_bytes=self.descriptor_tensor_bytes,
            descriptor_canonical_metadata_bytes=self.descriptor_canonical_metadata_bytes,
            source_kinetic_program_tensor_bytes=self.lowering.source_kinetic_program_tensor_bytes,
            persistent_sample_tensor_bytes=self.persistent_sample_tensor_bytes,
            persistent_target_tensor_bytes=self.persistent_target_tensor_bytes,
            persistent_interpolation_weight_tensor_bytes=(self.persistent_interpolation_weight_tensor_bytes),
            dense_row_by_global_time_tensor_bytes=self.dense_row_by_global_time_tensor_bytes,
            global_common_temporal_refinement_used=self.global_common_temporal_refinement_used,
            requested_frame_sampling_used_for_compile=self.requested_frame_sampling_used_for_compile,
            allocator_peak_measured=False,
            descriptor_python_allocator_bytes_measured=False,
        )

    def assert_warm_layout(self) -> None:
        """Validate the warm seal without source content inspection or allocation."""

        if self._seal is not _SAMPLER_SEAL:
            raise ValueError("kinetic ragged sampler was not sealed by its preparer")
        if (
            self.provenance != SAMPLER_PROVENANCE
            or self.view_index < 0
            or self.descriptor_tensor_bytes != 0
            or self.persistent_sample_tensor_bytes != 0
            or self.persistent_target_tensor_bytes != 0
            or self.persistent_interpolation_weight_tensor_bytes != 0
            or self.dense_row_by_global_time_tensor_bytes != 0
            or self.global_common_temporal_refinement_used
            or self.requested_frame_sampling_used_for_compile
            or self.warm_validation_kind != WARM_VALIDATION_KIND
            or self.warm_validation_device_to_host_syncs != 0
            or self.warm_validation_tensor_allocations != 0
            or self.descriptor_python_allocator_bytes_measured
        ):
            raise ValueError("kinetic ragged sampler execution/memory contract changed")
        if id(self.lowering) != self._lowering_identity:
            raise ValueError("kinetic ragged sampler lowering identity changed")
        if tuple(id(source) for source in self.sources) != self._source_identities:
            raise ValueError("kinetic ragged sampler chart source identity changed")
        for row in self.rows:
            row.assert_warm_identity()
        tensors = _source_tensors(self.sources)
        if len(tensors) != len(self.warm_source_tensor_signatures) or any(
            _warm_tensor_signature(tensor) != signature
            for tensor, signature in zip(tensors, self.warm_source_tensor_signatures, strict=True)
        ):
            raise ValueError("kinetic ragged sampler source tensor identity/layout/version changed")

    def assert_cold_current(self) -> None:
        """Re-run complete content/provenance validation at a cold boundary."""

        self.lowering.assert_current(self.sources)
        self.assert_warm_layout()
        tensors = _source_tensors(self.sources)
        if tuple(_tensor_digest(tensor) for tensor in tensors) != self.cold_source_tensor_digests:
            raise ValueError("kinetic ragged sampler source tensor content changed")
        expected_metadata = _sampler_metadata_encoding(
            view_index=self.view_index,
            lowering=self.lowering,
            rows=self.rows,
            track_ids=self.track_ids,
        )
        if len(expected_metadata) != self.descriptor_canonical_metadata_bytes:
            raise ValueError("kinetic ragged sampler metadata accounting changed")
        expected_digest = _sampler_digest(
            view_index=self.view_index,
            lowering=self.lowering,
            rows=self.rows,
            track_ids=self.track_ids,
            metadata_bytes=self.descriptor_canonical_metadata_bytes,
            source_tensor_digests=self.cold_source_tensor_digests,
        )
        if self.generation_digest != expected_digest:
            raise ValueError("kinetic ragged sampler generation digest changed")


@dataclass(frozen=True)
class PaperKineticRowRaggedSampleBlock:
    """One ephemeral source-ABI-shaped sample block for one native row block."""

    sampler_generation_digest: str
    native_block_generation_digest: str
    view_index: int
    node_count: int
    row_count: int
    sample_count: int
    global_loss_element_count: int
    loss_scale: float
    loss_normalization_id: str
    sample_row_i32: torch.Tensor = field(repr=False)
    sample_to_node_f32: torch.Tensor = field(repr=False)
    target_rgb_f32: torch.Tensor = field(repr=False)
    flat_sample_index_i64: torch.Tensor = field(repr=False)
    exact_node_row_count: int
    dense_fallback_row_count: int
    linear_weight_interactions: int
    dense_fallback_interactions: int
    dispatch_generation_digest: str
    warm_tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    generation_digest: str
    provenance: str = SAMPLER_PROVENANCE
    global_common_temporal_refinement_used: bool = False
    dense_row_by_global_time_tensor_bytes: int = 0
    persistent_after_launch_tensor_bytes: int = 0
    warm_validation_kind: str = WARM_VALIDATION_KIND
    warm_validation_device_to_host_syncs: int = 0
    warm_validation_tensor_allocations: int = 0
    _seal: object = field(default=None, repr=False)

    @property
    def device(self) -> torch.device:
        """Device shared by the launch tensors in this sealed sample block."""

        return self.sample_to_node_f32.device

    @property
    def retained_tensor_bytes(self) -> int:
        return _tensor_bytes(self._tensors())

    @property
    def native_sample_tensor_bytes(self) -> int:
        return _tensor_bytes((self.sample_row_i32, self.sample_to_node_f32, self.target_rgb_f32))

    @property
    def accounting(self) -> dict[str, int | float | str | bool]:
        return {
            "view_index": self.view_index,
            "native_block_generation_digest": self.native_block_generation_digest,
            "row_count": self.row_count,
            "node_count": self.node_count,
            "sample_count": self.sample_count,
            "global_loss_element_count": self.global_loss_element_count,
            "loss_scale": self.loss_scale,
            "retained_tensor_bytes": self.retained_tensor_bytes,
            "native_sample_tensor_bytes": self.native_sample_tensor_bytes,
            "flat_sample_provenance_tensor_bytes": _tensor_bytes((self.flat_sample_index_i64,)),
            "persistent_after_launch_tensor_bytes": self.persistent_after_launch_tensor_bytes,
            "dense_row_by_global_time_tensor_bytes": self.dense_row_by_global_time_tensor_bytes,
            "global_common_temporal_refinement_used": self.global_common_temporal_refinement_used,
            "global_denominator_preserved": True,
            "coordinator_compact_bar_assembly_implemented": True,
            "per_request_global_site_bar_allocated": False,
            "sample_to_node_linear_interactions": self.linear_weight_interactions,
            "sample_to_node_dense_fallback_interactions": self.dense_fallback_interactions,
            "exact_node_row_count": self.exact_node_row_count,
            "dense_fallback_row_count": self.dense_fallback_row_count,
            "allocator_peak_measured": False,
        }

    def _tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.sample_row_i32,
            self.sample_to_node_f32,
            self.target_rgb_f32,
            self.flat_sample_index_i64,
        )

    def assert_warm_layout(self) -> None:
        """Check the launch seal without copying, scalar reads, hashes, or allocations."""

        if self._seal is not _BLOCK_SEAL:
            raise ValueError("kinetic ragged sample block was not sealed by its materializer")
        if (
            self.provenance != SAMPLER_PROVENANCE
            or self.view_index < 0
            or self.node_count < 2
            or self.row_count < 1
            or self.sample_count < 1
            or self.global_loss_element_count < self.sample_count * 3
            or not math.isfinite(self.loss_scale)
            or self.loss_scale <= 0.0
            or self.loss_scale != 1.0 / float(self.global_loss_element_count)
            or not self.loss_normalization_id.strip()
            or self.global_common_temporal_refinement_used
            or self.dense_row_by_global_time_tensor_bytes != 0
            or self.persistent_after_launch_tensor_bytes != 0
            or self.warm_validation_kind != WARM_VALIDATION_KIND
            or self.warm_validation_device_to_host_syncs != 0
            or self.warm_validation_tensor_allocations != 0
        ):
            raise ValueError("kinetic ragged sample block execution/memory contract changed")
        tensors = self._tensors()
        if len(tensors) != len(self.warm_tensor_signatures) or any(
            _warm_tensor_signature(tensor) != signature
            for tensor, signature in zip(tensors, self.warm_tensor_signatures, strict=True)
        ):
            raise ValueError("kinetic ragged sample block tensor identity/layout/version changed")
        _require_warm_tensor(
            self.sample_row_i32,
            dtype=torch.int32,
            shape=(self.sample_count,),
            device=torch.device("cpu"),
        )
        launch_device = self.sample_to_node_f32.device
        _require_warm_tensor(
            self.sample_to_node_f32,
            dtype=torch.float32,
            shape=(self.sample_count, self.node_count),
            device=launch_device,
        )
        _require_warm_tensor(
            self.target_rgb_f32,
            dtype=torch.float32,
            shape=(self.sample_count, 3),
            device=launch_device,
        )
        _require_warm_tensor(
            self.flat_sample_index_i64,
            dtype=torch.int64,
            shape=(self.sample_count,),
            device=torch.device("cpu"),
        )

    def assert_cold_current(self, sampler: PaperKineticRowRaggedSampler) -> None:
        """Validate structural provenance without recertifying world contents.

        Full source-content recertification is the explicit
        :meth:`PaperKineticRowRaggedSampler.assert_cold_current` operation.  It
        must not be hidden in every streamed target request.
        """

        sampler.assert_warm_layout()
        self.assert_warm_layout()
        if self.sampler_generation_digest != sampler.generation_digest:
            raise ValueError("kinetic ragged sample block belongs to a stale sampler")
        native_blocks = tuple(
            block
            for bucket in sampler.lowering.buckets
            for block in bucket.blocks
            if block.generation_digest == self.native_block_generation_digest
        )
        if len(native_blocks) != 1:
            raise ValueError("kinetic ragged sample block has no unique native block")
        native_block = native_blocks[0]
        if native_block.node_count != self.node_count or native_block.row_count != self.row_count:
            raise ValueError("kinetic ragged sample/native row shape changed")
        if self.generation_digest != _sample_block_digest(self):
            raise ValueError("kinetic ragged sample block generation digest changed")

    def diagnose_cpu_content(self) -> None:
        """Opt-in CPU-only numeric diagnostic, never part of production launch."""

        self.assert_warm_layout()
        if self.sample_to_node_f32.device.type != "cpu" or self.target_rgb_f32.device.type != "cpu":
            raise ValueError("kinetic ragged content diagnostics require CPU launch tensors")
        _diagnose_cpu_sample_block_values(self)


@dataclass(frozen=True)
class PaperKineticCoordinatorBarAssemblyDiagnostic:
    """Explicit proof boundary between sample reduction and one compact bar."""

    native_block_generation_digests: tuple[str, ...]
    native_block_count: int
    global_site_count: int
    union_source_site_count: int
    summed_native_compact_site_count: int
    cross_native_block_merge_required: bool
    bounded_union_local_mapping_implemented: bool = True
    exactly_one_coordinator_compact_bar_proven: bool = True
    per_request_global_site_bar_allocated: bool = False
    missing_seam: str = ""
    proof_boundary: str = (
        "union-local coverage/provenance is implemented; injected sample reduction "
        "and native VJP numerical correctness remain separate backend obligations"
    )

    def require_ready(self) -> None:
        """Fail closed if the bounded union-local implementation is unavailable."""

        if not self.bounded_union_local_mapping_implemented or not self.exactly_one_coordinator_compact_bar_proven:
            raise NotImplementedError(self.missing_seam or "union-local material-bar assembly is unavailable")


def prepare_paper_kinetic_row_ragged_sampler(
    *,
    view_index: int,
    lowering: KineticNativeEqualRankLowering,
    sources: Sequence[KineticNativeEqualRankChartSource],
) -> PaperKineticRowRaggedSampler:
    """Cold-bind a view's canonical equal-rank rows without sample tensors."""

    _require_nonnegative_int(view_index, name="view_index")
    normalized_sources = tuple(sorted(tuple(sources), key=lambda source: source.row_identity))
    lowering.assert_current(normalized_sources)
    source_by_identity = {source.row_identity: source for source in normalized_sources}
    native_location: dict[int, tuple[str, int, int]] = {}
    for bucket in lowering.buckets:
        for block in bucket.blocks:
            for local_row, global_row in enumerate(block.global_row_indices):
                if global_row in native_location:
                    raise ValueError("equal-rank lowering maps one row into multiple native blocks")
                native_location[global_row] = (
                    block.generation_digest,
                    block.bucket_block_index,
                    local_row,
                )

    rows = []
    for row in lowering.rows:
        try:
            source = source_by_identity[row.row_identity]
            block_digest, block_index, local_row = native_location[row.global_row_index]
        except KeyError as error:
            raise ValueError("equal-rank sampler row source/native location is incomplete") from error
        chart = source.program.charts[source.chart_index]
        if (
            row.node_count != chart.node_count
            or row.t_min != chart.schedule.t_min
            or row.t_max != chart.schedule.t_max
            or row.right_closed != chart.right_closed
        ):
            raise ValueError("equal-rank sampler row and source schedule disagree")
        rows.append(
            PaperKineticRowBinding(
                global_row_index=row.global_row_index,
                track_id=row.track_id,
                chart_index=row.chart_index,
                node_count=row.node_count,
                native_block_generation_digest=block_digest,
                native_bucket_block_index=block_index,
                native_local_row_index=local_row,
                row_identity_digest=row.row_identity_digest,
                program_generation_digest=source.program.generation_digest,
                right_closed=row.right_closed,
                program=source.program,
                source=source,
                _program_identity=id(source.program),
                _source_identity=id(source),
            )
        )
    row_tuple = tuple(rows)
    track_ids = tuple(sorted({row.track_id for row in row_tuple}))
    _validate_complete_track_chart_cover(row_tuple, track_ids)
    tensors = _source_tensors(normalized_sources)
    content_digests = tuple(_tensor_digest(tensor) for tensor in tensors)
    metadata = _sampler_metadata_encoding(
        view_index=view_index,
        lowering=lowering,
        rows=row_tuple,
        track_ids=track_ids,
    )
    result = PaperKineticRowRaggedSampler(
        view_index=view_index,
        lowering=lowering,
        sources=normalized_sources,
        rows=row_tuple,
        track_ids=track_ids,
        descriptor_canonical_metadata_bytes=len(metadata),
        generation_digest=_sampler_digest(
            view_index=view_index,
            lowering=lowering,
            rows=row_tuple,
            track_ids=track_ids,
            metadata_bytes=len(metadata),
            source_tensor_digests=content_digests,
        ),
        cold_source_tensor_digests=content_digests,
        warm_source_tensor_signatures=tuple(_warm_tensor_signature(tensor) for tensor in tensors),
        _lowering_identity=id(lowering),
        _source_identities=tuple(id(source) for source in normalized_sources),
        _seal=_SAMPLER_SEAL,
    )
    result.assert_warm_layout()
    return result


def iter_paper_kinetic_row_ragged_sample_blocks(
    sampler: PaperKineticRowRaggedSampler,
    staged: PaperRaggedTrackTargetStageBlock,
    *,
    loss_normalization_id: str,
    maximum_samples_per_launch: int,
) -> Iterator[PaperKineticRowRaggedSampleBlock]:
    """Cold-dispatch one ``B_p x K`` target block into bounded native rows."""

    _require_positive_int(maximum_samples_per_launch, name="maximum_samples_per_launch")
    if not loss_normalization_id.strip():
        raise ValueError("loss_normalization_id must be nonempty")
    # Full source content was certified when the sampler was prepared.  A
    # streamed request performs only the sealed identity/layout/version check.
    sampler.assert_warm_layout()
    _validate_staged_observation_block(sampler, staged)

    track_ids = tuple(int(value) for value in staged.pixel_indices.tolist())
    times = tuple(float(value) for value in staged.sample_times.tolist())
    row_by_identity = {row.row_identity: row for row in sampler.rows}
    first_row_by_track = {
        track_id: next(row for row in sampler.rows if row.track_id == track_id) for track_id in track_ids
    }
    entries_by_native_block: dict[str, list[tuple[int, int, PaperKineticRowBinding]]] = {}
    observation_count = len(times)
    for track_position, track_id in enumerate(track_ids):
        track_binding = first_row_by_track[track_id]
        program = track_binding.program
        for observation_position, time in enumerate(times):
            chart_index = dispatch_prevalidated_kinetic_chart_index(
                program,
                Fraction.from_float(time),
                expected_generation_digest=track_binding.program_generation_digest,
            )
            try:
                row = row_by_identity[(track_id, chart_index)]
            except KeyError as error:
                raise ValueError("dispatched kinetic chart has no canonical equal-rank row") from error
            _validate_dispatched_row_time(row, time)
            flat_index = track_position * observation_count + observation_position
            entries_by_native_block.setdefault(row.native_block_generation_digest, []).append(
                (flat_index, observation_position, row)
            )

    expected_sample_count = len(track_ids) * observation_count
    if sum(len(entries) for entries in entries_by_native_block.values()) != expected_sample_count:
        raise ArithmeticError("kinetic ragged dispatch did not cover every pixel/observation pair")
    target_flat = staged.targets.reshape(expected_sample_count, 3)
    for bucket in sampler.lowering.buckets:
        for native_block in bucket.blocks:
            entries = entries_by_native_block.get(native_block.generation_digest, ())
            for start in range(0, len(entries), maximum_samples_per_launch):
                selected = tuple(entries[start : start + maximum_samples_per_launch])
                if selected:
                    yield _materialize_sample_block(
                        sampler,
                        native_block_generation_digest=native_block.generation_digest,
                        native_row_count=native_block.row_count,
                        node_count=native_block.node_count,
                        entries=selected,
                        sample_times=times,
                        target_flat=target_flat,
                        global_loss_element_count=staged.normalization.global_rgb_element_count,
                        loss_normalization_id=loss_normalization_id,
                    )


def iter_paper_kinetic_row_ragged_request_blocks(
    sampler: PaperKineticRowRaggedSampler,
    request: PaperRaggedMaterialBarRequest,
    *,
    maximum_samples_per_launch: int,
) -> Iterator[PaperKineticRowRaggedSampleBlock]:
    """Bind the generic bridge directly to a coordinator request token."""

    from paper_ragged_material_bar_coordinator import PaperRaggedMaterialBarRequest

    if not isinstance(request, PaperRaggedMaterialBarRequest):
        raise TypeError("kinetic ragged request bridge requires PaperRaggedMaterialBarRequest")
    request.assert_current()
    if request.view_index != sampler.view_index:
        raise ValueError("kinetic ragged sampler and coordinator request belong to different views")
    expected_scale = 1.0 / float(request.global_loss_element_count)
    if request.global_loss_scale != expected_scale:
        raise ValueError("coordinator request changed the one global paper-step loss scale")
    return iter_paper_kinetic_row_ragged_sample_blocks(
        sampler,
        request.staged,
        loss_normalization_id=request.loss_normalization_id,
        maximum_samples_per_launch=maximum_samples_per_launch,
    )


def diagnose_paper_kinetic_coordinator_bar_assembly(
    sampler: PaperKineticRowRaggedSampler,
    sample_blocks: Sequence[PaperKineticRowRaggedSampleBlock],
) -> PaperKineticCoordinatorBarAssemblyDiagnostic:
    """Report the implemented bounded multi-block assembly contract."""

    sampler.assert_warm_layout()
    blocks = tuple(sample_blocks)
    if not blocks:
        raise ValueError("coordinator bar diagnostic requires at least one sample block")
    for block in blocks:
        block.assert_cold_current(sampler)
    unique_digests = tuple(dict.fromkeys(block.native_block_generation_digest for block in blocks))
    native_by_digest = {
        block.generation_digest: block for bucket in sampler.lowering.buckets for block in bucket.blocks
    }
    selected = tuple(native_by_digest[digest] for digest in unique_digests)
    source_union = {site_id for block in selected for site_id in block.source_site_ids}
    return PaperKineticCoordinatorBarAssemblyDiagnostic(
        native_block_generation_digests=unique_digests,
        native_block_count=len(unique_digests),
        global_site_count=sampler.lowering.global_site_count,
        union_source_site_count=len(source_union),
        summed_native_compact_site_count=sum(len(block.source_site_ids) for block in selected),
        cross_native_block_merge_required=len(unique_digests) > 1,
    )


def _materialize_sample_block(
    sampler: PaperKineticRowRaggedSampler,
    *,
    native_block_generation_digest: str,
    native_row_count: int,
    node_count: int,
    entries: tuple[tuple[int, int, PaperKineticRowBinding], ...],
    sample_times: tuple[float, ...],
    target_flat: torch.Tensor,
    global_loss_element_count: int,
    loss_normalization_id: str,
) -> PaperKineticRowRaggedSampleBlock:
    sample_count = len(entries)
    weights_f64 = torch.empty((sample_count, node_count), dtype=torch.float64, device="cpu")
    sample_rows = torch.empty((sample_count,), dtype=torch.int32, device="cpu")
    flat_indices = torch.empty((sample_count,), dtype=torch.int64, device="cpu")
    exact_node_rows = 0
    dense_fallback_rows = 0
    linear_interactions = 0
    dense_fallback_interactions = 0

    entries_by_row: dict[tuple[int, int], list[int]] = {}
    for sample_index, (flat_index, _observation_position, row) in enumerate(entries):
        entries_by_row.setdefault(row.row_identity, []).append(sample_index)
        sample_rows[sample_index] = row.native_local_row_index
        flat_indices[sample_index] = flat_index
    for row_identity, sample_indices in entries_by_row.items():
        row = next(row for row in sampler.rows if row.row_identity == row_identity)
        row_times = torch.tensor(
            [sample_times[entries[index][1]] for index in sample_indices],
            dtype=torch.float64,
            device="cpu",
        )
        result = row.program.charts[row.chart_index].schedule.sample_to_node_weights(row_times)
        index = torch.tensor(sample_indices, dtype=torch.int64, device="cpu")
        weights_f64.index_copy_(0, index, result.weights)
        exact_node_rows += result.exact_node_row_count
        dense_fallback_rows += result.dense_fallback_row_count
        linear_interactions += result.linear_weight_interactions
        dense_fallback_interactions += result.dense_fallback_interactions

    launch_device = target_flat.device
    launch_indices = flat_indices.to(device=launch_device)
    weights = weights_f64.to(device=launch_device, dtype=torch.float32).contiguous()
    target = target_flat.index_select(0, launch_indices).to(dtype=torch.float32).contiguous()
    tensors = (sample_rows, weights, target, flat_indices)
    dispatch_digest = _digest_parts(
        "paper-kinetic-ragged-dispatch-v1",
        tuple(
            (
                flat_index,
                observation_position,
                row.global_row_index,
                row.native_local_row_index,
            )
            for flat_index, observation_position, row in entries
        ),
    )
    provisional = PaperKineticRowRaggedSampleBlock(
        sampler_generation_digest=sampler.generation_digest,
        native_block_generation_digest=native_block_generation_digest,
        view_index=sampler.view_index,
        node_count=node_count,
        row_count=native_row_count,
        sample_count=sample_count,
        global_loss_element_count=global_loss_element_count,
        loss_scale=1.0 / float(global_loss_element_count),
        loss_normalization_id=loss_normalization_id,
        sample_row_i32=sample_rows,
        sample_to_node_f32=weights,
        target_rgb_f32=target,
        flat_sample_index_i64=flat_indices,
        exact_node_row_count=exact_node_rows,
        dense_fallback_row_count=dense_fallback_rows,
        linear_weight_interactions=linear_interactions,
        dense_fallback_interactions=dense_fallback_interactions,
        dispatch_generation_digest=dispatch_digest,
        warm_tensor_signatures=tuple(_warm_tensor_signature(tensor) for tensor in tensors),
        generation_digest="",
        _seal=_BLOCK_SEAL,
    )
    result = replace(
        provisional,
        generation_digest=_sample_block_digest(provisional),
    )
    result.assert_warm_layout()
    return result


def seal_paper_kinetic_row_ragged_sample_block(
    sampler: PaperKineticRowRaggedSampler,
    *,
    native_block_generation_digest: str,
    sample_row_i32: torch.Tensor,
    sample_to_node_f32: torch.Tensor,
    target_rgb_f32: torch.Tensor,
    flat_sample_index_i64: torch.Tensor,
    global_loss_element_count: int,
    loss_normalization_id: str,
    exact_node_row_count: int,
    dense_fallback_row_count: int,
    linear_weight_interactions: int,
    dense_fallback_interactions: int,
    dispatch_generation_digest: str,
) -> PaperKineticRowRaggedSampleBlock:
    """Seal externally planned sparse rows into the native sample-block ABI.

    This is the narrow public constructor for dataset adapters whose logical
    observations are not a ``tracks x times`` Cartesian rectangle.  It does
    not dispatch charts or read targets; callers must provide those bounded
    tensors and a provenance digest.  The same warm/cold invariants used by
    the rectangular paper planner are enforced before return.
    """

    if not isinstance(sampler, PaperKineticRowRaggedSampler):
        raise TypeError("kinetic ragged sample sealing requires a sampler")
    sampler.assert_warm_layout()
    if not native_block_generation_digest.strip() or not dispatch_generation_digest.strip():
        raise ValueError("kinetic ragged sparse block provenance must be nonempty")
    if not loss_normalization_id.strip():
        raise ValueError("loss_normalization_id must be nonempty")
    _require_positive_int(global_loss_element_count, name="global_loss_element_count")
    matches = tuple(
        block
        for bucket in sampler.lowering.buckets
        for block in bucket.blocks
        if block.generation_digest == native_block_generation_digest
    )
    if len(matches) != 1:
        raise ValueError("kinetic ragged sparse block has no unique native block")
    native_block = matches[0]
    sample_count = int(torch.as_tensor(sample_row_i32).numel())
    if sample_count < 1 or global_loss_element_count < sample_count * 3:
        raise ValueError("kinetic ragged sparse block changed the global RGB denominator")
    tensors = (
        sample_row_i32,
        sample_to_node_f32,
        target_rgb_f32,
        flat_sample_index_i64,
    )
    provisional = PaperKineticRowRaggedSampleBlock(
        sampler_generation_digest=sampler.generation_digest,
        native_block_generation_digest=native_block_generation_digest,
        view_index=sampler.view_index,
        node_count=native_block.node_count,
        row_count=native_block.row_count,
        sample_count=sample_count,
        global_loss_element_count=global_loss_element_count,
        loss_scale=1.0 / float(global_loss_element_count),
        loss_normalization_id=loss_normalization_id,
        sample_row_i32=sample_row_i32,
        sample_to_node_f32=sample_to_node_f32,
        target_rgb_f32=target_rgb_f32,
        flat_sample_index_i64=flat_sample_index_i64,
        exact_node_row_count=exact_node_row_count,
        dense_fallback_row_count=dense_fallback_row_count,
        linear_weight_interactions=linear_weight_interactions,
        dense_fallback_interactions=dense_fallback_interactions,
        dispatch_generation_digest=dispatch_generation_digest,
        warm_tensor_signatures=tuple(_warm_tensor_signature(tensor) for tensor in tensors),
        generation_digest="",
        _seal=_BLOCK_SEAL,
    )
    result = replace(provisional, generation_digest=_sample_block_digest(provisional))
    result.assert_cold_current(sampler)
    if result.sample_to_node_f32.device.type == "cpu":
        result.diagnose_cpu_content()
    return result


def _validate_staged_observation_block(
    sampler: PaperKineticRowRaggedSampler,
    staged: PaperRaggedTrackTargetStageBlock,
) -> None:
    if not isinstance(staged, PaperRaggedTrackTargetStageBlock):
        raise TypeError("kinetic ragged sample planning requires PaperRaggedTrackTargetStageBlock")
    if staged.view_index != sampler.view_index:
        raise ValueError("kinetic ragged sampler and staged observations belong to different views")
    pixels = tuple(int(value) for value in staged.pixel_indices.tolist())
    if not pixels or len(set(pixels)) != len(pixels):
        raise ValueError("kinetic ragged staged pixel tracks must be unique and nonempty")
    if any(pixel not in sampler.track_ids for pixel in pixels):
        raise ValueError("kinetic ragged staged pixel has no compiled track program")
    if staged.sample_times.device.type != "cpu" or staged.sample_times.ndim != 1:
        raise ValueError("kinetic ragged sample times must be a CPU vector at the cold boundary")
    if staged.sample_times.numel() < 1 or not bool(torch.isfinite(staged.sample_times).all().item()):
        raise ValueError("kinetic ragged sample times must be nonempty and finite")
    normalization = staged.normalization
    if (
        normalization.block_track_count != len(pixels)
        or normalization.block_sample_count != int(staged.sample_times.numel())
        or normalization.global_rgb_element_count < normalization.block_rgb_element_count
        or normalization.rgb_channel_count != 3
    ):
        raise ValueError("kinetic ragged staged observations changed global loss normalization")


def _validate_dispatched_row_time(row: PaperKineticRowBinding, time: float) -> None:
    schedule = row.program.charts[row.chart_index].schedule
    if time < schedule.t_min or time > schedule.t_max:
        raise ValueError("exact kinetic dispatch selected a chart outside its float64 schedule")
    if time == schedule.t_max and not row.right_closed:
        raise ValueError("right-open kinetic chart received its excluded terminal sample")


def _validate_complete_track_chart_cover(
    rows: tuple[PaperKineticRowBinding, ...],
    track_ids: tuple[int, ...],
) -> None:
    if not rows or not track_ids:
        raise ValueError("kinetic ragged sampler requires at least one row and track")
    for expected_global_row, row in enumerate(rows):
        if row.global_row_index != expected_global_row:
            raise ValueError("kinetic ragged sampler rows are not canonically indexed")
    for track_id in track_ids:
        selected = tuple(row for row in rows if row.track_id == track_id)
        program_ids = {id(row.program) for row in selected}
        if len(program_ids) != 1:
            raise ValueError("one kinetic ragged track mixes multiple programs")
        expected = tuple(range(selected[0].program.chart_count))
        if tuple(row.chart_index for row in selected) != expected:
            raise ValueError("kinetic ragged track does not cover every chart exactly once")


def _diagnose_cpu_sample_block_values(block: PaperKineticRowRaggedSampleBlock) -> None:
    if block.sample_row_i32.numel() and (
        int(block.sample_row_i32.min().item()) < 0 or int(block.sample_row_i32.max().item()) >= block.row_count
    ):
        raise IndexError("kinetic ragged sample row leaves its native equal-rank block")
    if (
        block.flat_sample_index_i64.numel()
        and int(torch.unique(block.flat_sample_index_i64).numel()) != block.sample_count
    ):
        raise ValueError("kinetic ragged sample block duplicates a flat pixel/observation pair")
    if not bool(torch.isfinite(block.sample_to_node_f32).all().item()) or not bool(
        torch.isfinite(block.target_rgb_f32).all().item()
    ):
        raise ValueError("kinetic ragged sample weights and targets must be finite")
    tolerance = 128.0 * torch.finfo(torch.float32).eps * max(1, block.node_count)
    if bool(torch.any((block.sample_to_node_f32.sum(dim=1) - 1.0).abs() > tolerance).item()):
        raise ValueError("kinetic ragged sample weights violate partition of unity")
    if block.linear_weight_interactions != block.sample_count * block.node_count:
        raise ValueError("kinetic ragged linear interpolation accounting changed")
    if block.exact_node_row_count + block.dense_fallback_row_count > block.sample_count:
        raise ValueError("kinetic ragged interpolation row accounting is impossible")


def _source_tensors(
    sources: tuple[KineticNativeEqualRankChartSource, ...],
) -> tuple[torch.Tensor, ...]:
    tensors: list[torch.Tensor] = []
    seen_tracks: set[int] = set()
    for source in sources:
        if source.track_id in seen_tracks:
            continue
        seen_tracks.add(source.track_id)
        program = source.program
        binding = program.binding
        tensors.extend(
            (
                binding.sites.positions0,
                binding.sites.velocities,
                binding.sites.weight_coefficients,
                binding.ray_coefficients,
            )
        )
        for owner_chart in binding.program.charts:
            representative = owner_chart.representative_word
            tensors.extend(
                (
                    representative.word.owners,
                    representative.word.left_cut_ids,
                    representative.word.right_cut_ids,
                    representative.boundary_site_pairs,
                )
            )
        for chart in program.charts:
            tensors.extend(
                (
                    chart.schedule.node_times,
                    chart.schedule.fit_matrix,
                    chart.schedule.barycentric_weights,
                    chart.owners,
                    chart.node_physical_lengths,
                )
            )
    return tuple(tensors)


def _sampler_metadata_encoding(
    *,
    view_index: int,
    lowering: KineticNativeEqualRankLowering,
    rows: tuple[PaperKineticRowBinding, ...],
    track_ids: tuple[int, ...],
) -> bytes:
    return repr(
        (
            SAMPLER_PROVENANCE,
            view_index,
            lowering.generation_digest,
            track_ids,
            tuple(
                (
                    row.global_row_index,
                    row.track_id,
                    row.chart_index,
                    row.node_count,
                    row.native_block_generation_digest,
                    row.native_bucket_block_index,
                    row.native_local_row_index,
                    row.row_identity_digest,
                    row.program_generation_digest,
                    row.right_closed,
                )
                for row in rows
            ),
        )
    ).encode("utf-8")


def _sampler_digest(
    *,
    view_index: int,
    lowering: KineticNativeEqualRankLowering,
    rows: tuple[PaperKineticRowBinding, ...],
    track_ids: tuple[int, ...],
    metadata_bytes: int,
    source_tensor_digests: tuple[str, ...],
) -> str:
    return _digest_parts(
        SAMPLER_PROVENANCE,
        view_index,
        lowering.generation_digest,
        tuple(row.row_identity_digest for row in rows),
        track_ids,
        metadata_bytes,
        source_tensor_digests,
        False,
        0,
        0,
        0,
    )


def _sample_block_digest(
    block: PaperKineticRowRaggedSampleBlock,
) -> str:
    return _digest_parts(
        SAMPLER_PROVENANCE,
        block.sampler_generation_digest,
        block.native_block_generation_digest,
        block.view_index,
        block.node_count,
        block.row_count,
        block.sample_count,
        block.global_loss_element_count,
        block.loss_scale,
        block.loss_normalization_id,
        block.exact_node_row_count,
        block.dense_fallback_row_count,
        block.linear_weight_interactions,
        block.dense_fallback_interactions,
        block.dispatch_generation_digest,
        False,
        0,
    )


def _warm_tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        tensor.untyped_storage().data_ptr(),
        tensor._version,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.dtype,
        tensor.device,
    )


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    return _digest_parts(
        "paper-kinetic-ragged-tensor-v1",
        tuple(value.shape),
        str(value.dtype),
        value.numpy().tobytes(order="C"),
    )


def _tensor_bytes(tensors: tuple[torch.Tensor, ...]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _require_warm_tensor(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.dtype != dtype
        or tensor.device != device
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
    ):
        raise ValueError("kinetic ragged sample tensor warm layout changed")


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _require_nonnegative_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _require_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


__all__ = [
    "PaperKineticCoordinatorBarAssemblyDiagnostic",
    "PaperKineticRowBinding",
    "PaperKineticRowRaggedMemoryReport",
    "PaperKineticRowRaggedSampleBlock",
    "PaperKineticRowRaggedSampler",
    "SAMPLER_PROVENANCE",
    "WARM_VALIDATION_KIND",
    "diagnose_paper_kinetic_coordinator_bar_assembly",
    "iter_paper_kinetic_row_ragged_request_blocks",
    "iter_paper_kinetic_row_ragged_sample_blocks",
    "prepare_paper_kinetic_row_ragged_sampler",
    "seal_paper_kinetic_row_ragged_sample_block",
]
