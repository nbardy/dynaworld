"""Streamed equal-rank lowering for many kinetic WorldFoam ray charts.

The single-ray proof compiler emits one provenance-sealed payload per owner
chart.  The native precompiled-length kernel, however, accepts many CSR words
that share one temporal node rank ``J``.  This module is the tensor-free bridge
between those contracts:

* every row is exactly one ``(track_id, chart_index)``;
* rows are bucketed by their actual ``J`` without a global time refinement or
  padding to ``J_max``;
* each bounded block owns one union-compacted global site table, CSR owner
  word, and ``[J, W_total]`` physical-length table;
* chart domains and right closure remain row-local, while interpolation
  schedules stay in the already-counted source program rather than being
  duplicated as ``O(P J^2)`` native bucket state;
* no requested frame, sample time, target, prediction, or sample weight is
  retained by either the descriptor or a materialized block.

Content/provenance validation is deliberately cold.  ``assert_warm_layout``
checks only sealed tensor identity, shape, stride, dtype, device, and mutation
version; it performs no device-to-host copy, scalar extraction, content hash,
or tensor allocation.  This distinction is required before the payload can be
placed behind a production warm native launch.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import torch
from kinetic_multichart_transfer_program import KineticMultiChartP0Program
from kinetic_native_topology_lowering import (
    KineticNativeTopologyLowering,
    lower_kinetic_multichart_to_native_topology,
    materialize_kinetic_native_topology_chart,
)

LOWERING_PROVENANCE = "kinetic-native-equal-rank-batched-lowering-v1"
WARM_VALIDATION_KIND = "identity_shape_stride_dtype_device_version_only"

_DESCRIPTOR_SEAL = object()
_PAYLOAD_SEAL = object()


@dataclass(frozen=True)
class KineticNativeEqualRankChartSource:
    """One explicitly selected single-ray chart and its sealed source."""

    track_id: int
    lowering: KineticNativeTopologyLowering = field(repr=False)
    program: KineticMultiChartP0Program = field(repr=False)
    chart_index: int

    @property
    def row_identity(self) -> tuple[int, int]:
        return (self.track_id, self.chart_index)

    def assert_current(self) -> None:
        _validate_source_identity(self)
        self.lowering.assert_current(self.program)
        if not 0 <= self.chart_index < self.lowering.chart_count:
            raise ValueError("chart_index leaves its kinetic native lowering")


@dataclass(frozen=True)
class KineticNativeEqualRankRowSpec:
    """Tensor-free canonical identity and provenance for one native row."""

    global_row_index: int
    track_id: int
    chart_index: int
    row_identity_digest: str
    t_min: float
    t_max: float
    near: float
    far: float
    right_closed: bool
    node_count: int
    word_count: int
    owner_word: tuple[int, ...]
    source_site_ids: tuple[int, ...]
    source_content_digest: str
    kinetic_program_generation_digest: str
    topology_lowering_generation_digest: str
    chart_payload_digest: str
    schedule_generation_digest: str
    node_physical_lengths_digest: str

    @property
    def row_identity(self) -> tuple[int, int]:
        return (self.track_id, self.chart_index)

    def assert_self_consistent(self) -> None:
        _require_nonnegative_int(self.global_row_index, name="global_row_index")
        _require_nonnegative_int(self.track_id, name="track_id")
        _require_nonnegative_int(self.chart_index, name="chart_index")
        if (
            not math.isfinite(self.t_min)
            or not math.isfinite(self.t_max)
            or self.t_max <= self.t_min
            or not math.isfinite(self.near)
            or not math.isfinite(self.far)
            or self.far <= self.near
        ):
            raise ValueError("equal-rank row has an invalid time/depth domain")
        if not isinstance(self.right_closed, bool):
            raise TypeError("equal-rank row closure must be boolean")
        if self.node_count < 2 or self.word_count < 1 or self.word_count != len(self.owner_word):
            raise ValueError("equal-rank row rank/word metadata is invalid")
        if tuple(sorted(set(self.owner_word))) != self.source_site_ids:
            raise ValueError("equal-rank row compact source ids do not match its owner word")
        if any(site_id < 0 for site_id in self.source_site_ids):
            raise ValueError("equal-rank row source ids must be nonnegative")
        for digest in (
            self.row_identity_digest,
            self.source_content_digest,
            self.kinetic_program_generation_digest,
            self.topology_lowering_generation_digest,
            self.chart_payload_digest,
            self.schedule_generation_digest,
            self.node_physical_lengths_digest,
        ):
            _require_sha256(digest)


@dataclass(frozen=True)
class KineticNativeEqualRankBlockSpec:
    """One bounded block inside a rank bucket."""

    node_count: int
    bucket_block_index: int
    global_row_indices: tuple[int, ...]
    row_identity_digests: tuple[str, ...]
    source_site_ids: tuple[int, ...]
    word_count: int
    expected_retained_tensor_bytes: int
    generation_digest: str

    @property
    def row_count(self) -> int:
        return len(self.global_row_indices)

    def assert_self_consistent(
        self,
        rows_by_index: dict[int, KineticNativeEqualRankRowSpec],
        *,
        maximum_rows_per_block: int,
    ) -> None:
        if self.node_count < 2 or self.bucket_block_index < 0:
            raise ValueError("equal-rank block rank/index is invalid")
        if not self.global_row_indices or self.row_count > maximum_rows_per_block:
            raise ValueError("equal-rank block violates its bounded row count")
        if tuple(sorted(self.global_row_indices)) != self.global_row_indices:
            raise ValueError("equal-rank block rows are not in canonical order")
        try:
            rows = tuple(rows_by_index[index] for index in self.global_row_indices)
        except KeyError as error:
            raise ValueError("equal-rank block references an unknown row") from error
        if any(row.node_count != self.node_count for row in rows):
            raise ValueError("equal-rank block mixes temporal node ranks")
        if self.row_identity_digests != tuple(row.row_identity_digest for row in rows):
            raise ValueError("equal-rank block row identity provenance changed")
        expected_sites = tuple(sorted({site_id for row in rows for site_id in row.source_site_ids}))
        if self.source_site_ids != expected_sites:
            raise ValueError("equal-rank block compact site union changed")
        expected_words = sum(row.word_count for row in rows)
        if self.word_count != expected_words:
            raise ValueError("equal-rank block word accounting changed")
        expected_bytes = _expected_block_tensor_bytes(
            row_count=self.row_count,
            node_count=self.node_count,
            word_count=self.word_count,
            compact_site_count=len(self.source_site_ids),
        )
        if self.expected_retained_tensor_bytes != expected_bytes:
            raise ValueError("equal-rank block byte accounting changed")
        if self.generation_digest != _block_digest(
            node_count=self.node_count,
            bucket_block_index=self.bucket_block_index,
            rows=rows,
            source_site_ids=self.source_site_ids,
            word_count=self.word_count,
            expected_retained_tensor_bytes=expected_bytes,
        ):
            raise ValueError("equal-rank block generation digest mismatch")


@dataclass(frozen=True)
class KineticNativeEqualRankBucketSpec:
    """All canonical row blocks sharing one actual temporal node rank."""

    node_count: int
    global_row_indices: tuple[int, ...]
    blocks: tuple[KineticNativeEqualRankBlockSpec, ...]

    @property
    def row_count(self) -> int:
        return len(self.global_row_indices)


@dataclass(frozen=True)
class KineticNativeEqualRankMemoryReport:
    """Logical retained-tensor accounting, explicitly not allocator peak."""

    requested_frame_count: int
    descriptor_tensor_bytes: int
    descriptor_canonical_metadata_bytes: int
    descriptor_row_count: int
    descriptor_rank_bucket_count: int
    descriptor_block_count: int
    source_kinetic_program_tensor_bytes: int
    total_materialized_block_tensor_bytes: int
    maximum_materialized_block_tensor_bytes: int
    source_plus_maximum_block_tensor_bytes: int
    persistent_frame_tensor_bytes: int
    persistent_sample_tensor_bytes: int
    persistent_target_tensor_bytes: int
    persistent_prediction_tensor_bytes: int
    dense_row_by_global_time_tensor_bytes: int
    allocator_peak_measured: bool
    descriptor_python_allocator_bytes_measured: bool


@dataclass(frozen=True)
class KineticNativeEqualRankLowering:
    """Tensor-free descriptor for bounded equal-rank bucket streaming."""

    global_site_count: int
    site_namespace_digest: str
    maximum_rows_per_block: int
    rows: tuple[KineticNativeEqualRankRowSpec, ...]
    buckets: tuple[KineticNativeEqualRankBucketSpec, ...]
    source_kinetic_program_tensor_bytes: int
    descriptor_canonical_metadata_bytes: int
    generation_digest: str
    lowering_provenance: str = LOWERING_PROVENANCE
    descriptor_tensor_bytes: int = 0
    descriptor_python_allocator_bytes_measured: bool = False
    global_common_temporal_refinement_used: bool = False
    jmax_padding_used: bool = False
    requested_frame_sampling_used: bool = False
    persistent_frame_tensor_bytes: int = 0
    persistent_sample_tensor_bytes: int = 0
    persistent_target_tensor_bytes: int = 0
    persistent_prediction_tensor_bytes: int = 0
    dense_row_by_global_time_tensor_bytes: int = 0
    warm_validation_kind: str = WARM_VALIDATION_KIND
    warm_validation_device_to_host_syncs: int = 0
    warm_validation_tensor_allocations: int = 0
    _seal: object = field(default=None, repr=False)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def rank_bucket_count(self) -> int:
        return len(self.buckets)

    @property
    def block_count(self) -> int:
        return sum(len(bucket.blocks) for bucket in self.buckets)

    @property
    def maximum_materialized_block_tensor_bytes(self) -> int:
        return max(
            block.expected_retained_tensor_bytes
            for bucket in self.buckets
            for block in bucket.blocks
        )

    @property
    def total_materialized_block_tensor_bytes(self) -> int:
        """Logical sum if a caller defeats streaming and retains every block."""

        return sum(
            block.expected_retained_tensor_bytes
            for bucket in self.buckets
            for block in bucket.blocks
        )

    def memory_report(self, requested_frame_count: int) -> KineticNativeEqualRankMemoryReport:
        _require_positive_int(requested_frame_count, name="requested_frame_count")
        return KineticNativeEqualRankMemoryReport(
            requested_frame_count=requested_frame_count,
            descriptor_tensor_bytes=self.descriptor_tensor_bytes,
            descriptor_canonical_metadata_bytes=self.descriptor_canonical_metadata_bytes,
            descriptor_row_count=self.row_count,
            descriptor_rank_bucket_count=self.rank_bucket_count,
            descriptor_block_count=self.block_count,
            source_kinetic_program_tensor_bytes=self.source_kinetic_program_tensor_bytes,
            total_materialized_block_tensor_bytes=self.total_materialized_block_tensor_bytes,
            maximum_materialized_block_tensor_bytes=self.maximum_materialized_block_tensor_bytes,
            source_plus_maximum_block_tensor_bytes=(
                self.source_kinetic_program_tensor_bytes + self.maximum_materialized_block_tensor_bytes
            ),
            persistent_frame_tensor_bytes=self.persistent_frame_tensor_bytes,
            persistent_sample_tensor_bytes=self.persistent_sample_tensor_bytes,
            persistent_target_tensor_bytes=self.persistent_target_tensor_bytes,
            persistent_prediction_tensor_bytes=self.persistent_prediction_tensor_bytes,
            dense_row_by_global_time_tensor_bytes=self.dense_row_by_global_time_tensor_bytes,
            allocator_peak_measured=False,
            descriptor_python_allocator_bytes_measured=False,
        )

    def assert_current(self, sources: Sequence[KineticNativeEqualRankChartSource]) -> None:
        if self._seal is not _DESCRIPTOR_SEAL:
            raise ValueError("equal-rank lowering was not sealed by its compiler")
        if (
            self.lowering_provenance != LOWERING_PROVENANCE
            or self.descriptor_tensor_bytes != 0
            or self.descriptor_python_allocator_bytes_measured
            or self.global_common_temporal_refinement_used
            or self.jmax_padding_used
            or self.requested_frame_sampling_used
            or self.persistent_frame_tensor_bytes != 0
            or self.persistent_sample_tensor_bytes != 0
            or self.persistent_target_tensor_bytes != 0
            or self.persistent_prediction_tensor_bytes != 0
            or self.dense_row_by_global_time_tensor_bytes != 0
            or self.warm_validation_kind != WARM_VALIDATION_KIND
            or self.warm_validation_device_to_host_syncs != 0
            or self.warm_validation_tensor_allocations != 0
        ):
            raise ValueError("equal-rank lowering memory/execution contract changed")
        components = _descriptor_components(
            sources,
            maximum_rows_per_block=self.maximum_rows_per_block,
        )
        observed = (
            self.global_site_count,
            self.site_namespace_digest,
            self.rows,
            self.buckets,
            self.source_kinetic_program_tensor_bytes,
        )
        if observed != components:
            raise ValueError("equal-rank lowering source/provenance changed")
        expected_metadata_bytes = len(
            _descriptor_metadata_encoding(
                global_site_count=self.global_site_count,
                site_namespace_digest=self.site_namespace_digest,
                maximum_rows_per_block=self.maximum_rows_per_block,
                rows=self.rows,
                buckets=self.buckets,
                source_kinetic_program_tensor_bytes=self.source_kinetic_program_tensor_bytes,
            )
        )
        if self.descriptor_canonical_metadata_bytes != expected_metadata_bytes:
            raise ValueError("equal-rank descriptor canonical metadata accounting changed")
        rows_by_index = {row.global_row_index: row for row in self.rows}
        if len(rows_by_index) != self.row_count:
            raise ValueError("equal-rank lowering contains duplicate global row indices")
        expected_indices = tuple(range(self.row_count))
        if tuple(rows_by_index) != expected_indices:
            raise ValueError("equal-rank lowering rows are not canonically indexed")
        expected_bucket_ranks = tuple(sorted({row.node_count for row in self.rows}))
        if tuple(bucket.node_count for bucket in self.buckets) != expected_bucket_ranks:
            raise ValueError("equal-rank lowering buckets are not canonically ranked")
        seen_rows: list[int] = []
        for bucket in self.buckets:
            if bucket.global_row_indices != tuple(
                row.global_row_index for row in self.rows if row.node_count == bucket.node_count
            ):
                raise ValueError("equal-rank bucket rows changed")
            for block_index, block in enumerate(bucket.blocks):
                if block.bucket_block_index != block_index:
                    raise ValueError("equal-rank bucket block indices are not canonical")
                block.assert_self_consistent(
                    rows_by_index,
                    maximum_rows_per_block=self.maximum_rows_per_block,
                )
                seen_rows.extend(block.global_row_indices)
        if tuple(sorted(seen_rows)) != expected_indices or len(seen_rows) != self.row_count:
            raise ValueError("equal-rank bucket partition does not cover rows exactly once")
        if self.generation_digest != _descriptor_digest(
            global_site_count=self.global_site_count,
            site_namespace_digest=self.site_namespace_digest,
            maximum_rows_per_block=self.maximum_rows_per_block,
            rows=self.rows,
            buckets=self.buckets,
            source_kinetic_program_tensor_bytes=self.source_kinetic_program_tensor_bytes,
            descriptor_canonical_metadata_bytes=self.descriptor_canonical_metadata_bytes,
        ):
            raise ValueError("equal-rank lowering generation digest mismatch")


@dataclass(frozen=True)
class KineticNativeEqualRankBlockPayload:
    """One launch-shaped, frame-free equal-rank CSR/length block."""

    block: KineticNativeEqualRankBlockSpec
    source_site_ids_i64: torch.Tensor = field(repr=False)
    row_global_index_i64: torch.Tensor = field(repr=False)
    row_track_id_i64: torch.Tensor = field(repr=False)
    row_chart_index_i32: torch.Tensor = field(repr=False)
    row_right_closed_bool: torch.Tensor = field(repr=False)
    row_t_min_f64: torch.Tensor = field(repr=False)
    row_t_max_f64: torch.Tensor = field(repr=False)
    row_near_f64: torch.Tensor = field(repr=False)
    row_far_f64: torch.Tensor = field(repr=False)
    word_offsets_i32: torch.Tensor = field(repr=False)
    word_owner_i32: torch.Tensor = field(repr=False)
    node_physical_length_f32: torch.Tensor = field(repr=False)
    config_i32: torch.Tensor = field(repr=False)
    cold_content_digests: tuple[str, ...] = field(repr=False)
    warm_tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    generation_digest: str
    lowering_provenance: str = LOWERING_PROVENANCE
    requested_frame_sampling_used: bool = False
    frame_or_sample_axis_retained: bool = False
    target_or_prediction_retained: bool = False
    warm_validation_kind: str = WARM_VALIDATION_KIND
    warm_validation_device_to_host_syncs: int = 0
    warm_validation_tensor_allocations: int = 0
    _seal: object = field(default=None, repr=False)

    @property
    def row_count(self) -> int:
        return self.block.row_count

    @property
    def node_count(self) -> int:
        return self.block.node_count

    @property
    def word_count(self) -> int:
        return self.block.word_count

    @property
    def compact_site_count(self) -> int:
        return len(self.block.source_site_ids)

    @property
    def retained_tensor_bytes(self) -> int:
        return _tensor_bytes(self._tensors())

    @property
    def byte_accounting(self) -> dict[str, int | bool]:
        identity_domain = _tensor_bytes(
            (
                self.row_global_index_i64,
                self.row_track_id_i64,
                self.row_chart_index_i32,
                self.row_right_closed_bool,
                self.row_t_min_f64,
                self.row_t_max_f64,
                self.row_near_f64,
                self.row_far_f64,
            )
        )
        topology = _tensor_bytes(
            (
                self.source_site_ids_i64,
                self.word_offsets_i32,
                self.word_owner_i32,
                self.config_i32,
            )
        )
        return {
            "identity_domain_tensor_bytes": identity_domain,
            "topology_tensor_bytes": topology,
            "node_physical_length_tensor_bytes": _tensor_bytes((self.node_physical_length_f32,)),
            "duplicated_schedule_tensor_bytes": 0,
            "retained_tensor_bytes": self.retained_tensor_bytes,
            "persistent_frame_tensor_bytes": 0,
            "persistent_sample_tensor_bytes": 0,
            "persistent_target_tensor_bytes": 0,
            "persistent_prediction_tensor_bytes": 0,
            "dense_row_by_global_time_tensor_bytes": 0,
            "allocator_peak_measured": False,
        }

    def _tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.source_site_ids_i64,
            self.row_global_index_i64,
            self.row_track_id_i64,
            self.row_chart_index_i32,
            self.row_right_closed_bool,
            self.row_t_min_f64,
            self.row_t_max_f64,
            self.row_near_f64,
            self.row_far_f64,
            self.word_offsets_i32,
            self.word_owner_i32,
            self.node_physical_length_f32,
            self.config_i32,
        )

    def assert_warm_layout(self) -> None:
        """Validate a sealed warm payload without synchronization or allocation.

        The method intentionally does not validate tensor values.  Cold
        provenance validation did that once, and mutation versions make later
        in-place writes fail closed before launch.
        """

        if self._seal is not _PAYLOAD_SEAL:
            raise ValueError("equal-rank payload was not sealed by its materializer")
        if (
            self.lowering_provenance != LOWERING_PROVENANCE
            or self.requested_frame_sampling_used
            or self.frame_or_sample_axis_retained
            or self.target_or_prediction_retained
            or self.warm_validation_kind != WARM_VALIDATION_KIND
            or self.warm_validation_device_to_host_syncs != 0
            or self.warm_validation_tensor_allocations != 0
        ):
            raise ValueError("equal-rank payload warm/memory contract changed")
        tensors = self._tensors()
        if len(tensors) != len(self.warm_tensor_signatures) or any(
            _warm_tensor_signature(tensor) != signature
            for tensor, signature in zip(tensors, self.warm_tensor_signatures, strict=True)
        ):
            raise ValueError("equal-rank payload tensor identity/layout/version changed")
        _require_warm_tensor(
            self.source_site_ids_i64,
            dtype=torch.int64,
            shape=(self.compact_site_count,),
        )
        _require_warm_tensor(self.row_global_index_i64, dtype=torch.int64, shape=(self.row_count,))
        _require_warm_tensor(self.row_track_id_i64, dtype=torch.int64, shape=(self.row_count,))
        _require_warm_tensor(self.row_chart_index_i32, dtype=torch.int32, shape=(self.row_count,))
        _require_warm_tensor(self.row_right_closed_bool, dtype=torch.bool, shape=(self.row_count,))
        for tensor in (self.row_t_min_f64, self.row_t_max_f64, self.row_near_f64, self.row_far_f64):
            _require_warm_tensor(tensor, dtype=torch.float64, shape=(self.row_count,))
        _require_warm_tensor(self.word_offsets_i32, dtype=torch.int32, shape=(self.row_count + 1,))
        _require_warm_tensor(self.word_owner_i32, dtype=torch.int32, shape=(self.word_count,))
        _require_warm_tensor(
            self.node_physical_length_f32,
            dtype=torch.float32,
            shape=(self.node_count, self.word_count),
        )
        _require_warm_tensor(self.config_i32, dtype=torch.int32, shape=(4,))
        if self.retained_tensor_bytes != self.block.expected_retained_tensor_bytes:
            raise ValueError("equal-rank payload retained-byte accounting changed")

    def assert_cold_current(
        self,
        lowering: KineticNativeEqualRankLowering,
        sources: Sequence[KineticNativeEqualRankChartSource],
    ) -> None:
        """Perform complete host content/provenance validation before reuse."""

        lowering.assert_current(sources)
        matching = [
            block
            for bucket in lowering.buckets
            for block in bucket.blocks
            if block.generation_digest == self.block.generation_digest
        ]
        if len(matching) != 1 or matching[0] != self.block:
            raise ValueError("equal-rank payload block is not in its lowering")
        self.assert_warm_layout()
        tensors = self._tensors()
        current_digests = tuple(_tensor_digest(tensor) for tensor in tensors)
        if current_digests != self.cold_content_digests:
            raise ValueError("equal-rank payload tensor content changed")
        if self.generation_digest != _payload_digest(self.block, current_digests):
            raise ValueError("equal-rank payload generation digest mismatch")
        _validate_cold_payload_values(self)


def kinetic_native_equal_rank_chart_sources_for_track(
    track_id: int,
    program: KineticMultiChartP0Program,
    *,
    lowering: KineticNativeTopologyLowering | None = None,
) -> tuple[KineticNativeEqualRankChartSource, ...]:
    """Select every chart of one track without materializing chart tensors."""

    _require_nonnegative_int(track_id, name="track_id")
    selected_lowering = lowering or lower_kinetic_multichart_to_native_topology(program)
    selected_lowering.assert_current(program)
    return tuple(
        KineticNativeEqualRankChartSource(
            track_id=track_id,
            lowering=selected_lowering,
            program=program,
            chart_index=chart_index,
        )
        for chart_index in range(program.chart_count)
    )


def lower_kinetic_native_equal_rank_buckets(
    sources: Sequence[KineticNativeEqualRankChartSource],
    *,
    maximum_rows_per_block: int,
) -> KineticNativeEqualRankLowering:
    """Compile a tensor-free, bounded bucket plan from selected chart rows."""

    _require_positive_int(maximum_rows_per_block, name="maximum_rows_per_block")
    components = _descriptor_components(
        sources,
        maximum_rows_per_block=maximum_rows_per_block,
    )
    global_site_count, site_namespace_digest, rows, buckets, source_bytes = components
    metadata_bytes = len(
        _descriptor_metadata_encoding(
            global_site_count=global_site_count,
            site_namespace_digest=site_namespace_digest,
            maximum_rows_per_block=maximum_rows_per_block,
            rows=rows,
            buckets=buckets,
            source_kinetic_program_tensor_bytes=source_bytes,
        )
    )
    result = KineticNativeEqualRankLowering(
        global_site_count=global_site_count,
        site_namespace_digest=site_namespace_digest,
        maximum_rows_per_block=maximum_rows_per_block,
        rows=rows,
        buckets=buckets,
        source_kinetic_program_tensor_bytes=source_bytes,
        descriptor_canonical_metadata_bytes=metadata_bytes,
        generation_digest=_descriptor_digest(
            global_site_count=global_site_count,
            site_namespace_digest=site_namespace_digest,
            maximum_rows_per_block=maximum_rows_per_block,
            rows=rows,
            buckets=buckets,
            source_kinetic_program_tensor_bytes=source_bytes,
            descriptor_canonical_metadata_bytes=metadata_bytes,
        ),
        _seal=_DESCRIPTOR_SEAL,
    )
    result.assert_current(sources)
    return result


def iter_materialize_kinetic_native_equal_rank_blocks(
    lowering: KineticNativeEqualRankLowering,
    sources: Sequence[KineticNativeEqualRankChartSource],
) -> Iterator[KineticNativeEqualRankBlockPayload]:
    """Yield one bounded rank block at a time; callers need retain only one."""

    lowering.assert_current(sources)
    sources_by_identity = {source.row_identity: source for source in _normalize_sources(sources)}
    rows_by_index = {row.global_row_index: row for row in lowering.rows}
    for bucket in lowering.buckets:
        for block in bucket.blocks:
            rows = tuple(rows_by_index[index] for index in block.global_row_indices)
            yield _materialize_block(block, rows, sources_by_identity)


def pack_equal_rank_compact_site_rgba(
    payload: KineticNativeEqualRankBlockPayload,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
) -> torch.Tensor:
    """Cold-pack global material rows as native ``[RGB,density]`` float32."""

    payload.assert_warm_layout()
    density = torch.as_tensor(site_density, dtype=torch.float32, device="cpu").reshape(-1).detach()
    color = torch.as_tensor(site_color, dtype=torch.float32, device="cpu").detach()
    if color.ndim != 2 or color.shape != (density.numel(), 3):
        raise ValueError("site_color must have shape [global_site_count,3]")
    if not bool(torch.isfinite(density).all().item()) or not bool(torch.isfinite(color).all().item()):
        raise ValueError("global material tensors must be finite")
    if bool(torch.any(density < 0.0).item()):
        raise ValueError("site density must be nonnegative")
    ids = payload.source_site_ids_i64
    if ids.numel() and int(ids[-1].item()) >= density.numel():
        raise ValueError("payload compact source ids leave the global material table")
    compact_density = density.index_select(0, ids)
    compact_color = color.index_select(0, ids)
    return torch.cat((compact_color, compact_density[:, None]), dim=1).contiguous()


def _descriptor_components(
    sources: Sequence[KineticNativeEqualRankChartSource],
    *,
    maximum_rows_per_block: int,
) -> tuple[
    int,
    str,
    tuple[KineticNativeEqualRankRowSpec, ...],
    tuple[KineticNativeEqualRankBucketSpec, ...],
    int,
]:
    _require_positive_int(maximum_rows_per_block, name="maximum_rows_per_block")
    normalized = _normalize_sources(sources)
    namespace_digests = {_site_namespace_digest(source.program) for source in normalized}
    site_counts = {source.program.binding.sites.site_count for source in normalized}
    if len(namespace_digests) != 1 or len(site_counts) != 1:
        raise ValueError("equal-rank rows must share one global kinetic site namespace")
    site_namespace_digest = next(iter(namespace_digests))
    global_site_count = next(iter(site_counts))

    rows = tuple(
        _describe_row(source, global_row_index=global_row_index)
        for global_row_index, source in enumerate(normalized)
    )
    grouped: dict[int, list[KineticNativeEqualRankRowSpec]] = defaultdict(list)
    for row in rows:
        grouped[row.node_count].append(row)
    buckets = tuple(
        _make_bucket(
            node_count,
            tuple(grouped[node_count]),
            maximum_rows_per_block=maximum_rows_per_block,
        )
        for node_count in sorted(grouped)
    )

    source_bytes = 0
    seen_tracks: set[int] = set()
    for source in normalized:
        if source.track_id not in seen_tracks:
            source_bytes += source.lowering.source_kinetic_program_tensor_bytes
            seen_tracks.add(source.track_id)
    return global_site_count, site_namespace_digest, rows, buckets, source_bytes


def _normalize_sources(
    sources: Sequence[KineticNativeEqualRankChartSource],
) -> tuple[KineticNativeEqualRankChartSource, ...]:
    normalized = tuple(sources)
    if not normalized:
        raise ValueError("equal-rank lowering requires at least one chart source")
    for source in normalized:
        if not isinstance(source, KineticNativeEqualRankChartSource):
            raise TypeError("sources must contain KineticNativeEqualRankChartSource values")
        _validate_source_identity(source)
    result = tuple(sorted(normalized, key=lambda source: source.row_identity))
    identities = tuple(source.row_identity for source in result)
    if len(set(identities)) != len(identities):
        raise ValueError("equal-rank lowering received a duplicate (track_id, chart_index) identity")

    track_provenance: dict[int, tuple[str, str, str]] = {}
    first_source_by_track: dict[int, KineticNativeEqualRankChartSource] = {}
    for source in result:
        provenance = (
            source.program.binding.source_content_digest,
            source.program.generation_digest,
            source.lowering.generation_digest,
        )
        previous = track_provenance.setdefault(source.track_id, provenance)
        if previous != provenance:
            raise ValueError("one track_id cannot mix different kinetic programs/lowerings")
        first_source_by_track.setdefault(source.track_id, source)
    for source in first_source_by_track.values():
        source.lowering.assert_current(source.program)
    for source in result:
        if not 0 <= source.chart_index < source.lowering.chart_count:
            raise ValueError("chart_index leaves its kinetic native lowering")
    return result


def _describe_row(
    source: KineticNativeEqualRankChartSource,
    *,
    global_row_index: int,
) -> KineticNativeEqualRankRowSpec:
    chart = source.program.charts[source.chart_index]
    lowered = source.lowering.charts[source.chart_index]
    owner_word = tuple(int(site_id) for site_id in chart.owner_word)
    source_site_ids = tuple(sorted(set(owner_word)))
    identity_digest = _digest_parts(
        "kinetic-native-equal-rank-row-identity-v1",
        source.track_id,
        source.chart_index,
        source.program.binding.source_content_digest,
        source.program.generation_digest,
        source.lowering.generation_digest,
        lowered.payload_digest,
        chart.right_closed,
    )
    result = KineticNativeEqualRankRowSpec(
        global_row_index=global_row_index,
        track_id=source.track_id,
        chart_index=source.chart_index,
        row_identity_digest=identity_digest,
        t_min=chart.schedule.t_min,
        t_max=chart.schedule.t_max,
        near=chart.schedule.near,
        far=chart.schedule.far,
        right_closed=chart.right_closed,
        node_count=chart.node_count,
        word_count=chart.run_count,
        owner_word=owner_word,
        source_site_ids=source_site_ids,
        source_content_digest=source.program.binding.source_content_digest,
        kinetic_program_generation_digest=source.program.generation_digest,
        topology_lowering_generation_digest=source.lowering.generation_digest,
        chart_payload_digest=lowered.payload_digest,
        schedule_generation_digest=lowered.schedule_generation_digest,
        node_physical_lengths_digest=lowered.node_physical_lengths_digest,
    )
    result.assert_self_consistent()
    return result


def _make_bucket(
    node_count: int,
    rows: tuple[KineticNativeEqualRankRowSpec, ...],
    *,
    maximum_rows_per_block: int,
) -> KineticNativeEqualRankBucketSpec:
    blocks = []
    for block_index, start in enumerate(range(0, len(rows), maximum_rows_per_block)):
        selected = rows[start : start + maximum_rows_per_block]
        source_site_ids = tuple(sorted({site_id for row in selected for site_id in row.source_site_ids}))
        word_count = sum(row.word_count for row in selected)
        expected_bytes = _expected_block_tensor_bytes(
            row_count=len(selected),
            node_count=node_count,
            word_count=word_count,
            compact_site_count=len(source_site_ids),
        )
        blocks.append(
            KineticNativeEqualRankBlockSpec(
                node_count=node_count,
                bucket_block_index=block_index,
                global_row_indices=tuple(row.global_row_index for row in selected),
                row_identity_digests=tuple(row.row_identity_digest for row in selected),
                source_site_ids=source_site_ids,
                word_count=word_count,
                expected_retained_tensor_bytes=expected_bytes,
                generation_digest=_block_digest(
                    node_count=node_count,
                    bucket_block_index=block_index,
                    rows=selected,
                    source_site_ids=source_site_ids,
                    word_count=word_count,
                    expected_retained_tensor_bytes=expected_bytes,
                ),
            )
        )
    return KineticNativeEqualRankBucketSpec(
        node_count=node_count,
        global_row_indices=tuple(row.global_row_index for row in rows),
        blocks=tuple(blocks),
    )


def _materialize_block(
    block: KineticNativeEqualRankBlockSpec,
    rows: tuple[KineticNativeEqualRankRowSpec, ...],
    sources_by_identity: dict[tuple[int, int], KineticNativeEqualRankChartSource],
) -> KineticNativeEqualRankBlockPayload:
    compact_site = {site_id: local_id for local_id, site_id in enumerate(block.source_site_ids)}
    word_offsets = [0]
    compact_owners: list[int] = []
    lengths: list[torch.Tensor] = []
    for row in rows:
        try:
            source = sources_by_identity[row.row_identity]
        except KeyError as error:
            raise ValueError("equal-rank block source identity disappeared") from error
        chart_payload = materialize_kinetic_native_topology_chart(
            source.lowering,
            source.program,
            source.chart_index,
        )
        source_owner = chart_payload.topology.source_site_ids.index_select(
            0,
            chart_payload.topology.word_owner_i32.to(dtype=torch.int64),
        )
        if tuple(int(value) for value in source_owner.tolist()) != row.owner_word:
            raise ValueError("single-chart lowering owner word changed before batching")
        compact_owners.extend(compact_site[site_id] for site_id in row.owner_word)
        word_offsets.append(len(compact_owners))
        length_f32 = chart_payload.node_physical_lengths.to(dtype=torch.float32).contiguous()
        if not bool(torch.isfinite(length_f32).all().item()) or bool(torch.any(length_f32 <= 0.0).item()):
            raise ValueError("physical lengths must remain finite and positive after native float32 conversion")
        lengths.append(length_f32)

    tensors = (
        torch.tensor(block.source_site_ids, dtype=torch.int64),
        torch.tensor(block.global_row_indices, dtype=torch.int64),
        torch.tensor([row.track_id for row in rows], dtype=torch.int64),
        torch.tensor([row.chart_index for row in rows], dtype=torch.int32),
        torch.tensor([row.right_closed for row in rows], dtype=torch.bool),
        torch.tensor([row.t_min for row in rows], dtype=torch.float64),
        torch.tensor([row.t_max for row in rows], dtype=torch.float64),
        torch.tensor([row.near for row in rows], dtype=torch.float64),
        torch.tensor([row.far for row in rows], dtype=torch.float64),
        torch.tensor(word_offsets, dtype=torch.int32),
        torch.tensor(compact_owners, dtype=torch.int32),
        torch.cat(lengths, dim=1).contiguous(),
        torch.tensor(
            [len(rows), block.node_count, len(block.source_site_ids), block.word_count],
            dtype=torch.int32,
        ),
    )
    content_digests = tuple(_tensor_digest(tensor) for tensor in tensors)
    result = KineticNativeEqualRankBlockPayload(
        block=block,
        source_site_ids_i64=tensors[0],
        row_global_index_i64=tensors[1],
        row_track_id_i64=tensors[2],
        row_chart_index_i32=tensors[3],
        row_right_closed_bool=tensors[4],
        row_t_min_f64=tensors[5],
        row_t_max_f64=tensors[6],
        row_near_f64=tensors[7],
        row_far_f64=tensors[8],
        word_offsets_i32=tensors[9],
        word_owner_i32=tensors[10],
        node_physical_length_f32=tensors[11],
        config_i32=tensors[12],
        cold_content_digests=content_digests,
        warm_tensor_signatures=tuple(_warm_tensor_signature(tensor) for tensor in tensors),
        generation_digest=_payload_digest(block, content_digests),
        _seal=_PAYLOAD_SEAL,
    )
    result.assert_warm_layout()
    _validate_cold_payload_values(result)
    return result


def _validate_cold_payload_values(payload: KineticNativeEqualRankBlockPayload) -> None:
    if tuple(payload.source_site_ids_i64.tolist()) != payload.block.source_site_ids:
        raise ValueError("equal-rank payload compact source ids changed")
    if tuple(payload.row_global_index_i64.tolist()) != payload.block.global_row_indices:
        raise ValueError("equal-rank payload global row ids changed")
    if int(payload.word_offsets_i32[0].item()) != 0 or int(payload.word_offsets_i32[-1].item()) != payload.word_count:
        raise ValueError("equal-rank payload word offsets are invalid")
    if bool(torch.any(payload.word_offsets_i32[1:] < payload.word_offsets_i32[:-1]).item()):
        raise ValueError("equal-rank payload word offsets must be monotone")
    if payload.word_owner_i32.numel() and (
        int(payload.word_owner_i32.min().item()) < 0
        or int(payload.word_owner_i32.max().item()) >= payload.compact_site_count
    ):
        raise ValueError("equal-rank payload compact owner ids are out of range")
    if not bool(torch.isfinite(payload.node_physical_length_f32).all().item()) or bool(
        torch.any(payload.node_physical_length_f32 <= 0.0).item()
    ):
        raise ValueError("equal-rank payload physical lengths must be finite and positive")
    finite_tensors = (
        payload.row_t_min_f64,
        payload.row_t_max_f64,
        payload.row_near_f64,
        payload.row_far_f64,
    )
    if not all(bool(torch.isfinite(tensor).all().item()) for tensor in finite_tensors):
        raise ValueError("equal-rank payload domain/schedule tensors must be finite")
    if not bool(torch.all(payload.row_t_max_f64 > payload.row_t_min_f64).item()) or not bool(
        torch.all(payload.row_far_f64 > payload.row_near_f64).item()
    ):
        raise ValueError("equal-rank payload domains must be increasing")
    expected_config = torch.tensor(
        [payload.row_count, payload.node_count, payload.compact_site_count, payload.word_count],
        dtype=torch.int32,
    )
    if not torch.equal(payload.config_i32, expected_config):
        raise ValueError("equal-rank payload native config changed")


def _expected_block_tensor_bytes(
    *,
    row_count: int,
    node_count: int,
    word_count: int,
    compact_site_count: int,
) -> int:
    # int64: compact sites + global rows + track ids
    int64_bytes = 8 * (compact_site_count + 2 * row_count)
    # int32: chart ids + offsets + owners + native config
    int32_bytes = 4 * (row_count + (row_count + 1) + word_count + 4)
    bool_bytes = row_count
    # float64: four row-local domain boundaries.  The source program already
    # owns the interpolation schedules, so the native block does not duplicate
    # their O(P J^2) fit matrices.
    float64_bytes = 8 * 4 * row_count
    # float32: physical lengths [J,W_total]
    float32_bytes = 4 * node_count * word_count
    return int64_bytes + int32_bytes + bool_bytes + float64_bytes + float32_bytes


def _block_digest(
    *,
    node_count: int,
    bucket_block_index: int,
    rows: tuple[KineticNativeEqualRankRowSpec, ...],
    source_site_ids: tuple[int, ...],
    word_count: int,
    expected_retained_tensor_bytes: int,
) -> str:
    return _digest_parts(
        "kinetic-native-equal-rank-block-v1",
        node_count,
        bucket_block_index,
        tuple(row.row_identity_digest for row in rows),
        source_site_ids,
        word_count,
        expected_retained_tensor_bytes,
        False,
        False,
    )


def _descriptor_digest(
    *,
    global_site_count: int,
    site_namespace_digest: str,
    maximum_rows_per_block: int,
    rows: tuple[KineticNativeEqualRankRowSpec, ...],
    buckets: tuple[KineticNativeEqualRankBucketSpec, ...],
    source_kinetic_program_tensor_bytes: int,
    descriptor_canonical_metadata_bytes: int,
) -> str:
    return _digest_parts(
        "kinetic-native-equal-rank-lowering-v1",
        global_site_count,
        site_namespace_digest,
        maximum_rows_per_block,
        tuple(row.row_identity_digest for row in rows),
        tuple(
            (bucket.node_count, tuple(block.generation_digest for block in bucket.blocks))
            for bucket in buckets
        ),
        source_kinetic_program_tensor_bytes,
        descriptor_canonical_metadata_bytes,
        LOWERING_PROVENANCE,
        False,
        False,
        0,
        0,
        0,
        0,
    )


def _descriptor_metadata_encoding(
    *,
    global_site_count: int,
    site_namespace_digest: str,
    maximum_rows_per_block: int,
    rows: tuple[KineticNativeEqualRankRowSpec, ...],
    buckets: tuple[KineticNativeEqualRankBucketSpec, ...],
    source_kinetic_program_tensor_bytes: int,
) -> bytes:
    """Canonical logical metadata encoding; not a Python heap-size estimate."""

    value = (
        "kinetic-native-equal-rank-descriptor-metadata-v1",
        global_site_count,
        site_namespace_digest,
        maximum_rows_per_block,
        tuple(
            (
                row.global_row_index,
                row.track_id,
                row.chart_index,
                row.row_identity_digest,
                row.t_min,
                row.t_max,
                row.near,
                row.far,
                row.right_closed,
                row.node_count,
                row.word_count,
                row.owner_word,
                row.source_site_ids,
                row.source_content_digest,
                row.kinetic_program_generation_digest,
                row.topology_lowering_generation_digest,
                row.chart_payload_digest,
                row.schedule_generation_digest,
                row.node_physical_lengths_digest,
            )
            for row in rows
        ),
        tuple(
            (
                bucket.node_count,
                bucket.global_row_indices,
                tuple(
                    (
                        block.bucket_block_index,
                        block.global_row_indices,
                        block.row_identity_digests,
                        block.source_site_ids,
                        block.word_count,
                        block.expected_retained_tensor_bytes,
                        block.generation_digest,
                    )
                    for block in bucket.blocks
                ),
            )
            for bucket in buckets
        ),
        source_kinetic_program_tensor_bytes,
    )
    return repr(value).encode("utf-8")


def _payload_digest(block: KineticNativeEqualRankBlockSpec, content_digests: tuple[str, ...]) -> str:
    return _digest_parts(
        "kinetic-native-equal-rank-payload-v1",
        block.generation_digest,
        content_digests,
        LOWERING_PROVENANCE,
        WARM_VALIDATION_KIND,
    )


def _site_namespace_digest(program: KineticMultiChartP0Program) -> str:
    sites = program.binding.sites
    return _digest_parts(
        "kinetic-native-global-site-namespace-v1",
        sites.site_count,
        _tensor_digest(sites.positions0),
        _tensor_digest(sites.velocities),
        _tensor_digest(sites.weight_coefficients),
    )


def _tensor_bytes(tensors: Sequence[torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _warm_tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    """No-copy/no-scalar-extraction identity and layout seal."""

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


def _require_warm_tensor(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> None:
    if (
        tensor.dtype != dtype
        or tensor.device.type != "cpu"
        or tensor.layout != torch.strided
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
        or tensor.requires_grad
    ):
        raise ValueError("equal-rank payload tensor has an invalid warm layout")


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    return _digest_parts(
        "tensor-v1",
        tuple(value.shape),
        str(value.dtype),
        value.numpy().tobytes(order="C"),
    )


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _require_sha256(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("equal-rank lowering digest must be lowercase SHA-256")


def _require_nonnegative_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _require_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_source_identity(source: KineticNativeEqualRankChartSource) -> None:
    _require_nonnegative_int(source.track_id, name="track_id")
    if isinstance(source.chart_index, bool) or not isinstance(source.chart_index, int):
        raise TypeError("chart_index must be an integer")


__all__ = [
    "KineticNativeEqualRankBlockPayload",
    "KineticNativeEqualRankBlockSpec",
    "KineticNativeEqualRankBucketSpec",
    "KineticNativeEqualRankChartSource",
    "KineticNativeEqualRankLowering",
    "KineticNativeEqualRankMemoryReport",
    "KineticNativeEqualRankRowSpec",
    "LOWERING_PROVENANCE",
    "WARM_VALIDATION_KIND",
    "iter_materialize_kinetic_native_equal_rank_blocks",
    "kinetic_native_equal_rank_chart_sources_for_track",
    "lower_kinetic_native_equal_rank_buckets",
    "pack_equal_rank_compact_site_rgba",
]
