"""Bounded dense-observation replay through one cached kinetic native lane.

This module is the current candidate for a narrow CPU/source integration seam
between three existing
pieces of the memory-light WorldFoam path:

* :class:`PaperKineticCompiledCpuArtifact` retains a frame-free structural
  sampler in a byte-bounded store;
* :class:`PaperKineticReplayableDenseObservationSource` emits only one bounded
  observation-identity chunk at a time and owns the exact coverage cursor;
* :class:`KineticNativeMaterialStepExecutor` accumulates many sample launches
  into one node cotangent and launches exactly one selected reverse per active
  native block: material-only or material plus physical-length geometry.

The request accepts only a sealed target-loader capability bound to the exact
source, request, device, and memory policy. The loader publishes one bounded
transfer lifetime before enqueue, so a post-enqueue exception cannot hide its
CPU/device roots from the request's restart-required quarantine.

One request builds exactly one ephemeral device/native lane from the cached
sampler.  Every dense replay chunk reuses that lane.  Targets are decoded only
for the current chunk, no ray tensor is built, and each bounded sample launch
is followed by the caller-supplied completion fence before its K-local inputs
are released.  Caller-visible loss/material bars are committed only after the
request generator has exhausted, its cursor
has advanced exactly once, every active block has received exactly one
selected reverse, the executor has sealed, and a caller-supplied device fence has
completed. In full-geometry mode, each bounded ``[J,W_b]`` length bar is
fenced and reduced immediately into request-local site/trajectory/weight/ray
bars. Camera-ray bars are opt-in: the fixed-camera default discards each
bounded row-local ray derivative and never constructs a global
``[view*pixel,12]`` tensor or Python key table. The request returns one sealed
combined delta. A zero-owned, world-bound
step accumulator accepts at most one pending delta in replay order, and exposes
material and geometry bars only after the replay session seals the exact full
manifest.

The default full-geometry reverse remains that staged sparse oracle.  The
explicit ``fused_direct_v1`` selector is fixed-camera-only: after exact replay
coverage, the executor session derives the authoritative active manifest and
runs one all-block validate/accumulate/finalize transaction. It allocates no
``[J,W]`` cotangent. Accepted per-block compact material bars are scattered
into the same union-local delta, while accepted global float32 geometry bars
are bridged once to the existing CPU-float64 public delta ABI. Separate caps
cover every prepared block, all output scratch, and the bridge overlap.

The expensive structural/node/word work is therefore independent of requested
frame density for a fixed spatial artifact. Sample dispatch, target reads, and
sample reduction remain linear in requested observations and linear in the
fixed node rank (with a quadratic-in-rank fallback), rather than multiplying
the ordered-word reverse by frame count. The source now provides the
executor-bound full-geometry route, but does not
yet provide a production trainer route, native backend verification,
allocator-peak measurement, or a device-lane cache.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import wraps
from typing import Any

import torch
from kinetic_compiled_cpu_artifact_store import PaperKineticCompiledCpuArtifact
from kinetic_multichart_transfer_program import (
    dispatch_prevalidated_kinetic_chart_index,
)
from kinetic_native_equal_rank_lowering import (
    iter_materialize_kinetic_native_equal_rank_blocks,
)
from kinetic_native_equal_rank_geometry_reduction import (
    kinetic_native_equal_rank_vjp_provenance_id,
)
from kinetic_native_equal_rank_sparse_geometry_reduction import (
    preflight_kinetic_native_equal_rank_sparse_geometry_reduction_memory,
    reduce_kinetic_native_equal_rank_sparse_geometry_vjp,
)
from kinetic_native_equal_rank_runtime_adapter import (
    KineticNativeEqualRankRuntimeConstructionLifetime,
    KineticNativeEqualRankRuntimeBlock,
    prepare_kinetic_native_equal_rank_fused_direct_full_vjp_v1,
    prepare_kinetic_native_equal_rank_runtime_construction_lifetime,
    prepare_kinetic_native_equal_rank_runtime_block,
)
from kinetic_native_material_step_executor import (
    KineticNativeMaterialStepExecutor,
    KineticNativeMaterialStepTelemetry,
    KineticNativeMaterialStepWorldToken,
    KineticNativeNodeForwardIntoLifetime,
    KineticNativeSampleLaunchLifetime,
    prepare_kinetic_native_node_forward_into_lifetime,
    prepare_kinetic_native_material_step_executor,
)
from paper_kinetic_lazy_program_bundles import (
    PaperKineticLazyProgramBundleProvider,
    PaperKineticObservation,
)
from paper_kinetic_ragged_sample_plan import (
    PaperKineticRowBinding,
    PaperKineticRowRaggedSampleBlock,
    PaperKineticRowRaggedSampler,
    seal_paper_kinetic_row_ragged_sample_block,
)
from paper_kinetic_replayable_observations import (
    PaperKineticDenseObservationChunk,
    PaperKineticDenseObservationReplayReceipt,
    PaperKineticDenseObservationReplaySession,
    PaperKineticDenseObservationTrackRequest,
    PaperKineticReplayableDenseObservationSource,
)
from paper_kinetic_union_local_bar_assembly import (
    PaperKineticUnionLocalSpatialBundle,
    PaperKineticUnionLocalSpatialBundleConstructionLifetime,
    materialize_paper_kinetic_union_local_spatial_bundle,
    prepare_paper_kinetic_union_local_spatial_bundle_construction_lifetime,
)

REQUEST_PROVENANCE = "paper-kinetic-dense-cached-native-request-v6"
REQUEST_STATUS = "cpu_fake_native_candidate/full_geometry_source_runtime_unverified"
TARGET_PROVENANCE = "paper-kinetic-dense-cached-chunk-targets-v4"
TARGET_LOADER_PROVENANCE = "paper-kinetic-dense-sealed-target-loader-v1"
LANE_PROVENANCE = "paper-kinetic-dense-cached-native-lane-v2"
STEP_ACCUMULATOR_PROVENANCE = "paper-kinetic-dense-step-gradient-accumulator-v1"
REQUEST_DELTA_PROVENANCE = "paper-kinetic-dense-request-gradient-delta-v1"
REQUEST_DELTA_COMMIT_PROVENANCE = (
    "paper-kinetic-dense-request-gradient-delta-commit-v1"
)
OPTIMIZER_AUTHORIZATION_PROVENANCE = (
    "paper-kinetic-dense-optimizer-authorization-v1"
)
MPS_DEVICE_COMPLETION_FENCE_PROVENANCE = "torch.mps.synchronize/v1"
STAGED_SPARSE_FULL_GEOMETRY_REVERSE = "staged_sparse"
FUSED_DIRECT_V1_FULL_GEOMETRY_REVERSE = "fused_direct_v1"
_FULL_GEOMETRY_REVERSE_MODES = frozenset(
    {
        STAGED_SPARSE_FULL_GEOMETRY_REVERSE,
        FUSED_DIRECT_V1_FULL_GEOMETRY_REVERSE,
    }
)

_TARGET_SEAL = object()
_RECEIPT_SEAL = object()
_RESULT_SEAL = object()
_STEP_ACCUMULATOR_SEAL = object()
_REQUEST_DELTA_SEAL = object()
_REQUEST_DELTA_COMMIT_SEAL = object()
_OPTIMIZER_AUTHORIZATION_SEAL = object()
_TARGET_DECODE_OWNERSHIP_SEAL = object()
_TARGET_LOAD_LIFETIME_SEAL = object()
_TARGET_LOADER_SEAL = object()
_TARGET_LOADER_TEST_FAULT_SEAL = object()
_SAMPLE_MATERIALIZATION_LEASE_SEAL = object()
_COMPACT_GATHER_LIFETIME_SEAL = object()
_DENSE_LANE_CONSTRUCTION_LIFETIME_SEAL = object()


def synchronize_mps_device_completion_fence() -> None:
    """Canonical production fence for the conservative one-launch queue."""

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS completion fence requires an available MPS backend")
    torch.mps.synchronize()


@dataclass(frozen=True)
class PaperKineticDenseCachedNativeMemoryPolicy:
    """Fail-before-work logical tensor budgets for one spatial request."""

    maximum_lane_resident_logical_tensor_bytes: int
    # Historical field name: staged admission now composes the complete
    # one-block sparse reduction phase with all still-live active/request/step
    # tensors under this cap. It remains logical tensor accounting, not an
    # allocator or process peak.
    maximum_active_node_and_union_bar_tensor_bytes: int
    maximum_decoded_frame_scratch_tensor_bytes: int
    maximum_chunk_target_tensor_bytes: int
    maximum_target_decode_bridge_peak_logical_tensor_bytes: int
    maximum_sample_materialization_logical_tensor_bytes: int
    maximum_sample_launch_tensor_bytes: int
    maximum_request_geometry_bar_tensor_bytes: int
    maximum_geometry_bridge_visible_peak_logical_tensor_bytes: int
    # Zero keeps the unpromoted fused lane unavailable to existing callers.
    # An opt-in fused request must set all three bounds explicitly.
    maximum_fused_prepared_owned_logical_tensor_bytes: int = 0
    maximum_fused_output_scratch_logical_tensor_bytes: int = 0
    maximum_fused_geometry_bridge_visible_peak_logical_tensor_bytes: int = 0

    def assert_valid(self) -> None:
        for name, value in self.__dict__.items():
            if name.startswith("maximum_fused_"):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{name} must be a nonnegative int")
                continue
            _require_positive_int(value, name=name)


@dataclass(frozen=True)
class _DenseSampleMaterializationMemoryPlan:
    """Conservative public-tensor bound for one CPU-f64 weight materialization.

    This is deliberately not an allocator/RSS claim.  It sums a conservative
    envelope for every source-visible tensor family in the second-form
    barycentric evaluator, including its dense exceptional-row fallback, then
    adds the caller-owned output and CPU-to-device transfer overlap.  Internal
    PyTorch/allocator/Python-object storage remains unmeasured.
    """

    sample_count: int
    node_count: int
    maximum_logical_tensor_bytes: int
    interpolation_rows_per_subchunk: int
    interpolation_evaluator_scratch_logical_tensor_bytes_upper_bound: int
    materialization_peak_logical_tensor_bytes_upper_bound: int


@dataclass
class _DenseSampleMaterializationLease:
    """One bounded launch plus predecessor-command roots until its fence."""

    sample_block: PaperKineticRowRaggedSampleBlock | None = field(repr=False)
    weights_source_f64: torch.Tensor | None = field(repr=False)
    positions_i64: torch.Tensor | None = field(repr=False)
    chunk_target_rgb_f32: torch.Tensor | None = field(repr=False)
    sample_block_identity: int
    weights_source_signature: tuple[object, ...] = field(repr=False)
    positions_signature: tuple[object, ...] = field(repr=False)
    chunk_target_signature: tuple[object, ...] = field(repr=False)
    released_after_completion_fence: bool = False
    _seal: object = field(default=None, repr=False)

    def assert_retained(self) -> None:
        if (
            self._seal is not _SAMPLE_MATERIALIZATION_LEASE_SEAL
            or self.released_after_completion_fence
            or not isinstance(self.sample_block, PaperKineticRowRaggedSampleBlock)
            or id(self.sample_block) != self.sample_block_identity
            or not isinstance(self.weights_source_f64, torch.Tensor)
            or _tensor_signature(self.weights_source_f64)
            != self.weights_source_signature
            or not isinstance(self.positions_i64, torch.Tensor)
            or _tensor_signature(self.positions_i64) != self.positions_signature
            or not isinstance(self.chunk_target_rgb_f32, torch.Tensor)
            or _tensor_signature(self.chunk_target_rgb_f32)
            != self.chunk_target_signature
        ):
            raise ValueError("dense sample materialization lease changed")
        self.sample_block.assert_warm_layout()

    def release_after_fence(self) -> None:
        try:
            self.assert_retained()
        finally:
            self.sample_block = None
            self.weights_source_f64 = None
            self.positions_i64 = None
            self.chunk_target_rgb_f32 = None
            self.released_after_completion_fence = True


_INTERPOLATION_EVALUATOR_FIXED_LOGICAL_BYTES = 4096
_INTERPOLATION_EVALUATOR_PER_NODE_LOGICAL_BYTES = 512
_INTERPOLATION_EVALUATOR_PER_NODE_SQUARED_LOGICAL_BYTES = 8
_INTERPOLATION_EVALUATOR_PER_ROW_LOGICAL_BYTES = 1024
_INTERPOLATION_EVALUATOR_PER_ROW_NODE_LOGICAL_BYTES = 512


@dataclass(frozen=True)
class _DenseTargetDecodeOwnership:
    """Unforgeable proof of one bounded CPU pixel read before transfer."""

    source_generation_digest: str
    request_generation_digest: str
    chunk_generation_digest: str
    selected_pixel_read_mode: str
    selected_pixel_read_source_provenance: str
    selected_pixel_read_call_count: int
    selected_pixel_read_acceptance_capable: bool
    direct_selected_pixel_observation_count: int
    bounded_region_selected_pixel_observation_count: int
    full_frame_fallback_observation_count: int
    decoded_frame_count: int
    maximum_cpu_decoded_frame_tensor_bytes: int
    bounded_region_materialization_count: int
    maximum_bounded_region_materialization_tensor_bytes: int
    source_visible_target_read_peak_logical_tensor_bytes_upper_bound: int
    transient_mapped_address_space_bytes: int
    maximum_requested_unique_mapped_page_count: int
    total_requested_unique_mapped_page_count: int
    mapped_page_size_bytes: int
    maximum_requested_mapped_page_bytes_upper_bound: int
    total_requested_mapped_page_bytes_upper_bound: int
    mapping_closed_before_return: bool
    cpu_chunk_target_tensor_bytes: int
    device_chunk_target_tensor_bytes: int
    target_decode_bridge_peak_logical_tensor_bytes: int
    target_tensor_identity: int
    target_tensor_signature: tuple[object, ...] = field(repr=False)
    cpu_transfer_source_identity: int
    cpu_transfer_source_signature: tuple[object, ...] = field(repr=False)
    cpu_transfer_source_ref: torch.Tensor = field(repr=False)
    decoded_frame_device_type: str = "cpu"
    decoded_frame_mps_completion_fence_call_count: int = 0
    cpu_to_device_chunk_transfer_requested_non_blocking: bool = False
    single_bounded_chunk_transfer: bool = True
    real_device_transfer_completion_verified: bool = False
    _seal: object = field(default=None, repr=False)

    def assert_current(
        self,
        source: PaperKineticReplayableDenseObservationSource,
        request: PaperKineticDenseObservationTrackRequest,
        chunk: PaperKineticDenseObservationChunk,
        target_rgb_f32: torch.Tensor,
    ) -> None:
        expected_frame_count, _ = _chunk_frame_decode_cardinality(chunk)
        expected_frame_bytes = 3 * source.provider.height * source.provider.width * 4
        expected_chunk_bytes = chunk.observation_count * 3 * 4
        expected_peak = _target_pixel_read_bridge_peak_logical_tensor_bytes(
            source_visible_read_peak_logical_tensor_bytes=(
                self.source_visible_target_read_peak_logical_tensor_bytes_upper_bound
            ),
            chunk_target_tensor_bytes=expected_chunk_bytes,
            target_device=target_rgb_f32.device,
        )
        mode_counts_are_valid = (
            (
                self.selected_pixel_read_mode == "direct_pixels"
                and self.direct_selected_pixel_observation_count
                == chunk.observation_count
                and self.bounded_region_selected_pixel_observation_count == 0
                and self.full_frame_fallback_observation_count == 0
                and self.decoded_frame_count == 0
                and self.maximum_cpu_decoded_frame_tensor_bytes == 0
                and self.bounded_region_materialization_count == 0
                and self.maximum_bounded_region_materialization_tensor_bytes == 0
            )
            or (
                self.selected_pixel_read_mode == "certified_bounded_region"
                and self.direct_selected_pixel_observation_count == 0
                and self.bounded_region_selected_pixel_observation_count
                == chunk.observation_count
                and self.full_frame_fallback_observation_count == 0
                and self.decoded_frame_count == 0
                and self.maximum_cpu_decoded_frame_tensor_bytes == 0
                and self.bounded_region_materialization_count >= 1
                and 0
                < self.maximum_bounded_region_materialization_tensor_bytes
                < expected_frame_bytes
            )
            or (
                self.selected_pixel_read_mode == "full_frame_fallback"
                and self.direct_selected_pixel_observation_count == 0
                and self.bounded_region_selected_pixel_observation_count == 0
                and self.full_frame_fallback_observation_count
                == chunk.observation_count
                and self.decoded_frame_count == expected_frame_count
                and self.maximum_cpu_decoded_frame_tensor_bytes
                == expected_frame_bytes
                and self.bounded_region_materialization_count == 0
                and self.maximum_bounded_region_materialization_tensor_bytes == 0
            )
        )
        mapped_receipt_is_valid = (
            (
                self.transient_mapped_address_space_bytes > 0
                and self.maximum_requested_unique_mapped_page_count > 0
                and self.total_requested_unique_mapped_page_count
                >= self.maximum_requested_unique_mapped_page_count
                and self.mapped_page_size_bytes > 0
                and self.maximum_requested_mapped_page_bytes_upper_bound
                == self.maximum_requested_unique_mapped_page_count
                * self.mapped_page_size_bytes
                and self.total_requested_mapped_page_bytes_upper_bound
                == self.total_requested_unique_mapped_page_count
                * self.mapped_page_size_bytes
                and self.mapping_closed_before_return
            )
            or (
                self.transient_mapped_address_space_bytes == 0
                and self.maximum_requested_unique_mapped_page_count == 0
                and self.total_requested_unique_mapped_page_count == 0
                and self.mapped_page_size_bytes == 0
                and self.maximum_requested_mapped_page_bytes_upper_bound == 0
                and self.total_requested_mapped_page_bytes_upper_bound == 0
                and self.mapping_closed_before_return
            )
        )
        if (
            self._seal is not _TARGET_DECODE_OWNERSHIP_SEAL
            or self.source_generation_digest != source.generation_digest
            or self.request_generation_digest != request.generation_digest
            or self.chunk_generation_digest != chunk.generation_digest
            or not self.selected_pixel_read_source_provenance.strip()
            or self.selected_pixel_read_call_count != 1
            or self.selected_pixel_read_acceptance_capable
            != (
                self.selected_pixel_read_mode
                in {"direct_pixels", "certified_bounded_region"}
            )
            or not mode_counts_are_valid
            or not mapped_receipt_is_valid
            or self.source_visible_target_read_peak_logical_tensor_bytes_upper_bound
            < expected_chunk_bytes
            or self.cpu_chunk_target_tensor_bytes != expected_chunk_bytes
            or self.device_chunk_target_tensor_bytes != expected_chunk_bytes
            or self.target_decode_bridge_peak_logical_tensor_bytes != expected_peak
            or id(target_rgb_f32) != self.target_tensor_identity
            or _tensor_signature(target_rgb_f32) != self.target_tensor_signature
            or id(self.cpu_transfer_source_ref)
            != self.cpu_transfer_source_identity
            or _tensor_signature(self.cpu_transfer_source_ref)
            != self.cpu_transfer_source_signature
            or self.cpu_transfer_source_ref.device.type != "cpu"
            or self.cpu_transfer_source_ref.dtype != torch.float32
            or tuple(self.cpu_transfer_source_ref.shape)
            != (chunk.observation_count, 3)
            or self.decoded_frame_device_type != "cpu"
            or self.decoded_frame_mps_completion_fence_call_count != 0
            or self.cpu_to_device_chunk_transfer_requested_non_blocking
            or not self.single_bounded_chunk_transfer
            or self.real_device_transfer_completion_verified
        ):
            raise ValueError("dense target decode ownership changed or is foreign")


@dataclass(frozen=True)
class PaperKineticDenseChunkTargetLoaderTestFault:
    """Sealed source-test fault; it cannot run arbitrary callback code."""

    stage: str
    message: str
    fail_on_load_number: int
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        if (
            self._seal is not _TARGET_LOADER_TEST_FAULT_SEAL
            or self.stage != "after_transfer_before_target_seal"
            or not self.message.strip()
            or isinstance(self.fail_on_load_number, bool)
            or not isinstance(self.fail_on_load_number, int)
            or self.fail_on_load_number < 1
        ):
            raise ValueError("dense target-loader test fault changed")


@dataclass
class PaperKineticDenseChunkTargetLoadLifetime:
    """One target read/transfer rooted across return, failure, and fencing.

    The carrier is installed on the sealed loader before target-provider work.
    Immediately before the CPU-to-device transfer it retains the CPU source and
    conservatively marks device work as possibly in flight.  Every device
    tensor returned by a transfer/materialization command is then added to the
    same bounded carrier.  A post-enqueue exception therefore cannot hide the
    transfer roots from request-level quarantine.
    """

    source_generation_digest: str
    request_generation_digest: str
    chunk_generation_digest: str
    source_identity: int
    request_identity: int
    chunk_identity: int
    chunk_ref: PaperKineticDenseObservationChunk = field(repr=False)
    phase: str = "reading"
    device_work_may_be_in_flight: bool = False
    failure_after_enqueue: bool = False
    completion_fence_proven: bool = False
    released: bool = False
    selected_read_ref: Any = field(default=None, repr=False)
    cpu_transfer_source_ref: torch.Tensor | None = field(default=None, repr=False)
    cpu_transfer_source_signature: tuple[object, ...] | None = field(
        default=None,
        repr=False,
    )
    device_tensor_refs: tuple[torch.Tensor, ...] = field(default=(), repr=False)
    device_tensor_signatures: tuple[tuple[object, ...], ...] = field(
        default=(),
        repr=False,
    )
    returned_targets_ref: PaperKineticDenseChunkTargets | None = field(
        default=None,
        repr=False,
    )
    failure_ref: BaseException | None = field(default=None, repr=False)
    _seal: object = field(default=None, repr=False)

    def assert_for(
        self,
        source: PaperKineticReplayableDenseObservationSource,
        request: PaperKineticDenseObservationTrackRequest,
        chunk: PaperKineticDenseObservationChunk,
    ) -> None:
        if (
            self._seal is not _TARGET_LOAD_LIFETIME_SEAL
            or self.source_generation_digest != source.generation_digest
            or self.request_generation_digest != request.generation_digest
            or self.chunk_generation_digest != chunk.generation_digest
            or self.source_identity != id(source)
            or self.request_identity != id(request)
            or self.chunk_identity != id(chunk)
            or self.chunk_ref is not chunk
            or self.released
            or self.phase not in {
                "reading",
                "transfer_pending",
                "returned",
                "failed_after_enqueue",
            }
            or self.completion_fence_proven
        ):
            raise ValueError("dense target-load lifetime changed or is foreign")
        if self.cpu_transfer_source_ref is not None:
            if (
                self.cpu_transfer_source_signature
                != _tensor_signature(self.cpu_transfer_source_ref)
                or self.cpu_transfer_source_ref.device.type != "cpu"
            ):
                raise ValueError("dense target-load CPU transfer root changed")
        if (
            len(self.device_tensor_refs) != len(self.device_tensor_signatures)
            or tuple(_tensor_signature(tensor) for tensor in self.device_tensor_refs)
            != self.device_tensor_signatures
        ):
            raise ValueError("dense target-load device roots changed")
        if self.phase == "returned":
            if (
                not isinstance(
                    self.returned_targets_ref,
                    PaperKineticDenseChunkTargets,
                )
                or not self.device_work_may_be_in_flight
                or self.failure_after_enqueue
                or self.failure_ref is not None
            ):
                raise ValueError("dense returned target-load lifetime changed")
        if self.phase == "failed_after_enqueue" and (
            not self.device_work_may_be_in_flight
            or not self.failure_after_enqueue
            or self.failure_ref is None
        ):
            raise ValueError("dense failed target-load lifetime changed")

    def retain_transfer_source(
        self,
        selected_read: Any,
        cpu_targets: torch.Tensor,
    ) -> None:
        if self.phase != "reading" or self.released:
            raise ValueError("dense target-load transfer started twice")
        self.selected_read_ref = selected_read
        self.cpu_transfer_source_ref = cpu_targets
        self.cpu_transfer_source_signature = _tensor_signature(cpu_targets)
        # Mark before invoking ``Tensor.to``: an exception from the backend is
        # not proof that it queued no work.
        self.device_work_may_be_in_flight = True
        self.phase = "transfer_pending"

    def retain_device_tensor(self, tensor: torch.Tensor) -> None:
        if self.phase != "transfer_pending" or self.released:
            raise ValueError("dense target-load device root arrived out of order")
        self.device_tensor_refs = (*self.device_tensor_refs, tensor)
        self.device_tensor_signatures = (
            *self.device_tensor_signatures,
            _tensor_signature(tensor),
        )

    def mark_returned(self, targets: PaperKineticDenseChunkTargets) -> None:
        if self.phase != "transfer_pending" or self.released:
            raise ValueError("dense target-load return arrived out of order")
        self.returned_targets_ref = targets
        self.phase = "returned"

    def mark_failed(self, error: BaseException) -> None:
        if not self.device_work_may_be_in_flight:
            raise ValueError("pre-enqueue target-load failure needs no quarantine")
        self.failure_ref = error
        self.failure_after_enqueue = True
        self.phase = "failed_after_enqueue"

    def release_after_completion_fence(self) -> None:
        if self.phase not in {"returned", "failed_after_enqueue"} or self.released:
            raise ValueError("dense target-load lifetime cannot be released")
        self.completion_fence_proven = True
        self.selected_read_ref = None
        self.cpu_transfer_source_ref = None
        self.cpu_transfer_source_signature = None
        self.device_tensor_refs = ()
        self.device_tensor_signatures = ()
        self.returned_targets_ref = None
        self.failure_ref = None
        self.released = True
        self.phase = "released"


@dataclass
class PaperKineticDenseChunkTargetLoader:
    """Exact bounded target-loader capability accepted by dense replay."""

    source_generation_digest: str
    request_generation_digest: str
    target_generation_id: str
    device: torch.device
    maximum_decoded_frame_scratch_tensor_bytes: int
    maximum_chunk_target_tensor_bytes: int
    maximum_target_decode_bridge_peak_logical_tensor_bytes: int
    generation_digest: str
    _source_ref: PaperKineticReplayableDenseObservationSource = field(repr=False)
    _request_ref: PaperKineticDenseObservationTrackRequest = field(repr=False)
    _source_identity: int = field(repr=False)
    _request_identity: int = field(repr=False)
    _test_fault: PaperKineticDenseChunkTargetLoaderTestFault | None = field(
        default=None,
        repr=False,
    )
    _active_lifetime: PaperKineticDenseChunkTargetLoadLifetime | None = field(
        default=None,
        repr=False,
    )
    completed_load_count: int = 0
    failed_after_enqueue_count: int = 0
    provenance: str = TARGET_LOADER_PROVENANCE
    _seal: object = field(default=None, repr=False)

    def assert_current(
        self,
        source: PaperKineticReplayableDenseObservationSource,
        request: PaperKineticDenseObservationTrackRequest,
        *,
        device: torch.device,
    ) -> None:
        if self._test_fault is not None:
            self._test_fault.assert_current()
        if (
            self._seal is not _TARGET_LOADER_SEAL
            or self.provenance != TARGET_LOADER_PROVENANCE
            or self._source_ref is not source
            or self._request_ref is not request
            or self._source_identity != id(source)
            or self._request_identity != id(request)
            or self.source_generation_digest != source.generation_digest
            or self.request_generation_digest != request.generation_digest
            or self.device != device
            or not self.target_generation_id.strip()
            or self.completed_load_count < 0
            or self.failed_after_enqueue_count < 0
            or self.generation_digest != _target_loader_digest(self)
        ):
            raise ValueError("dense sealed target loader changed or is foreign")
        if self._active_lifetime is not None:
            self._active_lifetime.assert_for(source, request, self._active_chunk())

    def _active_chunk(self) -> PaperKineticDenseObservationChunk:
        lifetime = self._active_lifetime
        if lifetime is None:
            raise ValueError("dense target loader has no active chunk")
        chunk = lifetime.chunk_ref
        if not isinstance(chunk, PaperKineticDenseObservationChunk):
            raise ValueError("dense target loader lost its active chunk")
        return chunk

    def load(
        self,
        chunk: PaperKineticDenseObservationChunk,
    ) -> PaperKineticDenseChunkTargets:
        self.assert_current(self._source_ref, self._request_ref, device=self.device)
        if self._active_lifetime is not None:
            raise ValueError("dense target loader permits one outstanding load")
        lifetime = PaperKineticDenseChunkTargetLoadLifetime(
            source_generation_digest=self.source_generation_digest,
            request_generation_digest=self.request_generation_digest,
            chunk_generation_digest=chunk.generation_digest,
            source_identity=id(self._source_ref),
            request_identity=id(self._request_ref),
            chunk_identity=id(chunk),
            chunk_ref=chunk,
            _seal=_TARGET_LOAD_LIFETIME_SEAL,
        )
        self._active_lifetime = lifetime
        try:
            targets = decode_paper_kinetic_dense_chunk_targets(
                self._source_ref,
                self._request_ref,
                chunk,
                device=self.device,
                target_generation_id=self.target_generation_id,
                maximum_decoded_frame_scratch_tensor_bytes=(
                    self.maximum_decoded_frame_scratch_tensor_bytes
                ),
                maximum_chunk_target_tensor_bytes=(
                    self.maximum_chunk_target_tensor_bytes
                ),
                maximum_target_decode_bridge_peak_logical_tensor_bytes=(
                    self.maximum_target_decode_bridge_peak_logical_tensor_bytes
                ),
                _load_lifetime=lifetime,
                _test_fault=(
                    self._test_fault
                    if self._test_fault is not None
                    and self.completed_load_count + 1
                    == self._test_fault.fail_on_load_number
                    else None
                ),
            )
            lifetime.mark_returned(targets)
        except BaseException as error:
            if lifetime.device_work_may_be_in_flight:
                lifetime.mark_failed(error)
                self.failed_after_enqueue_count += 1
            else:
                lifetime.released = True
                lifetime.phase = "released"
                self._active_lifetime = None
            raise
        lifetime.assert_for(self._source_ref, self._request_ref, chunk)
        return targets

    def release_returned_after_completion_fence(
        self,
        targets: PaperKineticDenseChunkTargets,
    ) -> None:
        lifetime = self._active_lifetime
        if (
            lifetime is None
            or lifetime.phase != "returned"
            or lifetime.returned_targets_ref is not targets
        ):
            raise ValueError("dense target loader cannot release foreign targets")
        lifetime.assert_for(self._source_ref, self._request_ref, self._active_chunk())
        lifetime.release_after_completion_fence()
        self._active_lifetime = None
        self.completed_load_count += 1

    def release_failed_after_completion_fence(self) -> None:
        lifetime = self._active_lifetime
        if lifetime is None or lifetime.phase != "failed_after_enqueue":
            raise ValueError("dense target loader has no failed transfer lifetime")
        lifetime.assert_for(self._source_ref, self._request_ref, self._active_chunk())
        lifetime.release_after_completion_fence()
        self._active_lifetime = None

    def release_active_after_completion_fence(self) -> None:
        """Release either returned or failed roots after a proven fence."""

        lifetime = self._active_lifetime
        if lifetime is None:
            return
        if lifetime.phase not in {"returned", "failed_after_enqueue"}:
            raise ValueError("dense target loader has no releasable lifetime")
        lifetime.assert_for(self._source_ref, self._request_ref, self._active_chunk())
        returned = lifetime.phase == "returned"
        lifetime.release_after_completion_fence()
        self._active_lifetime = None
        if returned:
            self.completed_load_count += 1


@dataclass(frozen=True)
class PaperKineticDenseChunkTargets:
    """One sealed, bounded target payload corresponding exactly to one chunk."""

    source_generation_digest: str
    request_generation_digest: str
    chunk_generation_digest: str
    target_generation_id: str
    target_rgb_f32: torch.Tensor = field(repr=False)
    logical_tensor_bytes: int
    selected_pixel_read_mode: str
    selected_pixel_read_source_provenance: str
    selected_pixel_read_call_count: int
    selected_pixel_read_acceptance_capable: bool
    direct_selected_pixel_observation_count: int
    bounded_region_selected_pixel_observation_count: int
    full_frame_fallback_observation_count: int
    decoded_frame_count: int
    maximum_cpu_decoded_frame_tensor_bytes: int
    bounded_region_materialization_count: int
    maximum_bounded_region_materialization_tensor_bytes: int
    source_visible_target_read_peak_logical_tensor_bytes_upper_bound: int
    transient_mapped_address_space_bytes: int
    maximum_requested_unique_mapped_page_count: int
    total_requested_unique_mapped_page_count: int
    mapped_page_size_bytes: int
    maximum_requested_mapped_page_bytes_upper_bound: int
    total_requested_mapped_page_bytes_upper_bound: int
    mapping_closed_before_return: bool
    cpu_chunk_target_tensor_bytes: int
    device_chunk_target_tensor_bytes: int
    target_decode_bridge_peak_logical_tensor_bytes: int
    decoded_frame_device_type: str
    decoded_frame_mps_completion_fence_call_count: int
    cpu_to_device_chunk_transfer_requested_non_blocking: bool
    single_bounded_chunk_transfer: bool
    real_device_transfer_completion_verified: bool
    warm_tensor_signature: tuple[object, ...] = field(repr=False)
    generation_digest: str
    _cpu_transfer_source_ref: torch.Tensor = field(repr=False)
    _cpu_transfer_source_identity: int = field(repr=False)
    _cpu_transfer_source_signature: tuple[object, ...] = field(repr=False)
    provenance: str = TARGET_PROVENANCE
    persistent_after_chunk_tensor_bytes: int = 0
    _seal: object = field(default=None, repr=False)

    @property
    def observation_count(self) -> int:
        return int(self.target_rgb_f32.shape[0])

    def assert_transfer_retained(self) -> None:
        if (
            id(self._cpu_transfer_source_ref)
            != self._cpu_transfer_source_identity
            or _tensor_signature(self._cpu_transfer_source_ref)
            != self._cpu_transfer_source_signature
            or self._cpu_transfer_source_ref.device.type != "cpu"
            or self._cpu_transfer_source_ref.dtype != torch.float32
            or tuple(self._cpu_transfer_source_ref.shape)
            != tuple(self.target_rgb_f32.shape)
        ):
            raise ValueError("dense chunk CPU transfer source is not retained")

    def assert_current(
        self,
        source: PaperKineticReplayableDenseObservationSource,
        request: PaperKineticDenseObservationTrackRequest,
        chunk: PaperKineticDenseObservationChunk,
        *,
        device: torch.device,
    ) -> None:
        source.assert_warm_current()
        request.assert_current(source)
        chunk.assert_self_consistent(source, request)
        expected_frame_count, _ = _chunk_frame_decode_cardinality(chunk)
        expected_frame_bytes = 3 * source.provider.height * source.provider.width * 4
        expected_bridge_peak = _target_pixel_read_bridge_peak_logical_tensor_bytes(
            source_visible_read_peak_logical_tensor_bytes=(
                self.source_visible_target_read_peak_logical_tensor_bytes_upper_bound
            ),
            chunk_target_tensor_bytes=self.logical_tensor_bytes,
            target_device=device,
        )
        mode_counts_are_valid = (
            (
                self.selected_pixel_read_mode == "direct_pixels"
                and self.direct_selected_pixel_observation_count
                == chunk.observation_count
                and self.bounded_region_selected_pixel_observation_count == 0
                and self.full_frame_fallback_observation_count == 0
                and self.decoded_frame_count == 0
                and self.maximum_cpu_decoded_frame_tensor_bytes == 0
                and self.bounded_region_materialization_count == 0
                and self.maximum_bounded_region_materialization_tensor_bytes == 0
            )
            or (
                self.selected_pixel_read_mode == "certified_bounded_region"
                and self.direct_selected_pixel_observation_count == 0
                and self.bounded_region_selected_pixel_observation_count
                == chunk.observation_count
                and self.full_frame_fallback_observation_count == 0
                and self.decoded_frame_count == 0
                and self.maximum_cpu_decoded_frame_tensor_bytes == 0
                and self.bounded_region_materialization_count >= 1
                and 0
                < self.maximum_bounded_region_materialization_tensor_bytes
                < expected_frame_bytes
            )
            or (
                self.selected_pixel_read_mode == "full_frame_fallback"
                and self.direct_selected_pixel_observation_count == 0
                and self.bounded_region_selected_pixel_observation_count == 0
                and self.full_frame_fallback_observation_count
                == chunk.observation_count
                and self.decoded_frame_count == expected_frame_count
                and self.maximum_cpu_decoded_frame_tensor_bytes
                == expected_frame_bytes
                and self.bounded_region_materialization_count == 0
                and self.maximum_bounded_region_materialization_tensor_bytes == 0
            )
        )
        mapped_receipt_is_valid = (
            (
                self.transient_mapped_address_space_bytes > 0
                and self.maximum_requested_unique_mapped_page_count > 0
                and self.total_requested_unique_mapped_page_count
                >= self.maximum_requested_unique_mapped_page_count
                and self.mapped_page_size_bytes > 0
                and self.maximum_requested_mapped_page_bytes_upper_bound
                == self.maximum_requested_unique_mapped_page_count
                * self.mapped_page_size_bytes
                and self.total_requested_mapped_page_bytes_upper_bound
                == self.total_requested_unique_mapped_page_count
                * self.mapped_page_size_bytes
                and self.mapping_closed_before_return
            )
            or (
                self.transient_mapped_address_space_bytes == 0
                and self.maximum_requested_unique_mapped_page_count == 0
                and self.total_requested_unique_mapped_page_count == 0
                and self.mapped_page_size_bytes == 0
                and self.maximum_requested_mapped_page_bytes_upper_bound == 0
                and self.total_requested_mapped_page_bytes_upper_bound == 0
                and self.mapping_closed_before_return
            )
        )
        if (
            self._seal is not _TARGET_SEAL
            or self.provenance != TARGET_PROVENANCE
            or self.source_generation_digest != source.generation_digest
            or self.request_generation_digest != request.generation_digest
            or self.chunk_generation_digest != chunk.generation_digest
            or not self.target_generation_id.strip()
            or self.persistent_after_chunk_tensor_bytes != 0
            or self.observation_count != chunk.observation_count
            or self.logical_tensor_bytes != chunk.observation_count * 3 * 4
            or not self.selected_pixel_read_source_provenance.strip()
            or self.selected_pixel_read_call_count != 1
            or self.selected_pixel_read_acceptance_capable
            != (
                self.selected_pixel_read_mode
                in {"direct_pixels", "certified_bounded_region"}
            )
            or not mode_counts_are_valid
            or not mapped_receipt_is_valid
            or self.source_visible_target_read_peak_logical_tensor_bytes_upper_bound
            < self.logical_tensor_bytes
            or self.cpu_chunk_target_tensor_bytes != self.logical_tensor_bytes
            or self.device_chunk_target_tensor_bytes != self.logical_tensor_bytes
            or self.target_decode_bridge_peak_logical_tensor_bytes
            != expected_bridge_peak
            or self.decoded_frame_device_type != "cpu"
            or self.decoded_frame_mps_completion_fence_call_count != 0
            or self.cpu_to_device_chunk_transfer_requested_non_blocking
            or not self.single_bounded_chunk_transfer
            or self.real_device_transfer_completion_verified
            or self.warm_tensor_signature != _tensor_signature(self.target_rgb_f32)
            or id(self._cpu_transfer_source_ref)
            != self._cpu_transfer_source_identity
            or _tensor_signature(self._cpu_transfer_source_ref)
            != self._cpu_transfer_source_signature
            or self._cpu_transfer_source_ref.device.type != "cpu"
            or self._cpu_transfer_source_ref.dtype != torch.float32
            or tuple(self._cpu_transfer_source_ref.shape)
            != (chunk.observation_count, 3)
            or self.generation_digest != _target_digest(self)
        ):
            raise ValueError("dense cached chunk targets changed or are foreign")
        _require_tensor(
            self.target_rgb_f32,
            name="target_rgb_f32",
            device=device,
            dtype=torch.float32,
            shape=(chunk.observation_count, 3),
        )


@dataclass
class _DenseCachedNativeLaneConstructionLifetime:
    """Caller-owned roots for partial union/runtime device construction."""

    artifact: PaperKineticCompiledCpuArtifact = field(repr=False)
    provider: PaperKineticLazyProgramBundleProvider = field(repr=False)
    request: PaperKineticDenseObservationTrackRequest = field(repr=False)
    native_ops: Any = field(repr=False)
    device: torch.device
    backend_provenance: str
    resident_logical_tensor_bytes_upper_bound: int
    expected_runtime_block_digests: tuple[str, ...]
    spatial_construction_lifetime: (
        PaperKineticUnionLocalSpatialBundleConstructionLifetime | None
    ) = field(default=None, repr=False)
    spatial_bundle: PaperKineticUnionLocalSpatialBundle | None = field(
        default=None,
        repr=False,
    )
    payloads: tuple[Any, ...] = field(default=(), repr=False)
    runtime_lifetimes: list[
        KineticNativeEqualRankRuntimeConstructionLifetime
    ] = field(default_factory=list, repr=False)
    runtimes: list[KineticNativeEqualRankRuntimeBlock] = field(
        default_factory=list,
        repr=False,
    )
    executor: KineticNativeMaterialStepExecutor | None = field(
        default=None,
        repr=False,
    )
    current_payload: Any = field(default=None, repr=False)
    current_runtime_lifetime: (
        KineticNativeEqualRankRuntimeConstructionLifetime | None
    ) = field(default=None, repr=False)
    phase: str = "installed"
    _artifact_identity: int = field(default=0, repr=False)
    _provider_identity: int = field(default=0, repr=False)
    _request_identity: int = field(default=0, repr=False)
    _native_ops_identity: int = field(default=0, repr=False)
    _seal: object = field(default=None, repr=False)

    def assert_retained(self) -> None:
        if (
            self._seal is not _DENSE_LANE_CONSTRUCTION_LIFETIME_SEAL
            or self.phase not in {"installed", "materializing", "transferred"}
            or id(self.artifact) != self._artifact_identity
            or id(self.provider) != self._provider_identity
            or id(self.request) != self._request_identity
            or id(self.native_ops) != self._native_ops_identity
            or not self.backend_provenance.strip()
            or self.resident_logical_tensor_bytes_upper_bound < 1
            or not self.expected_runtime_block_digests
            or len(set(self.expected_runtime_block_digests))
            != len(self.expected_runtime_block_digests)
            or len(self.runtimes) > len(self.runtime_lifetimes)
            or len(self.runtime_lifetimes) > len(self.payloads)
            or len(self.payloads) > len(self.expected_runtime_block_digests)
            or tuple(
                payload.block.generation_digest for payload in self.payloads
            )
            != self.expected_runtime_block_digests[: len(self.payloads)]
            or (
                self.spatial_construction_lifetime is not None
                and not isinstance(
                    self.spatial_construction_lifetime,
                    PaperKineticUnionLocalSpatialBundleConstructionLifetime,
                )
            )
            or (
                self.spatial_bundle is not None
                and not isinstance(
                    self.spatial_bundle,
                    PaperKineticUnionLocalSpatialBundle,
                )
            )
            or (self.phase == "transferred")
            != (
                isinstance(
                    self.spatial_bundle,
                    PaperKineticUnionLocalSpatialBundle,
                )
                and isinstance(self.executor, KineticNativeMaterialStepExecutor)
                and len(self.runtimes) == len(self.payloads)
                and len(self.runtime_lifetimes) == len(self.payloads)
                and len(self.payloads)
                == len(self.expected_runtime_block_digests)
            )
        ):
            raise ValueError("dense lane construction lifetime changed")
        if self.spatial_construction_lifetime is not None:
            self.spatial_construction_lifetime.assert_retained()
        for index, lifetime in enumerate(self.runtime_lifetimes):
            lifetime.assert_retained()
            if lifetime.payload is not self.payloads[index]:
                raise ValueError("dense runtime construction payload changed")
        for index, runtime in enumerate(self.runtimes):
            runtime.assert_warm_layout()
            if runtime.payload is not self.payloads[index]:
                raise ValueError("dense runtime construction result changed")
        if self.current_payload is not None and all(
            self.current_payload is not payload for payload in self.payloads
        ):
            raise ValueError("dense current runtime payload is foreign")
        if self.current_runtime_lifetime is not None:
            self.current_runtime_lifetime.assert_retained()
            if self.current_payload is not self.current_runtime_lifetime.payload:
                raise ValueError("dense current runtime predecessor changed")
        if self.executor is not None:
            self.executor.assert_current()


@dataclass(frozen=True)
class _DenseCachedNativeLane:
    """Internal carrier; existing sampler/runtime/executor types own validation."""

    artifact: PaperKineticCompiledCpuArtifact = field(repr=False)
    spatial_bundle: PaperKineticUnionLocalSpatialBundle = field(repr=False)
    runtimes: tuple[KineticNativeEqualRankRuntimeBlock, ...] = field(repr=False)
    executor: KineticNativeMaterialStepExecutor = field(repr=False)
    construction_lifetime: _DenseCachedNativeLaneConstructionLifetime = field(
        repr=False
    )
    resident_logical_tensor_bytes_upper_bound: int
    generation_digest: str

    def runtime_for_digest(self, digest: str) -> KineticNativeEqualRankRuntimeBlock:
        self.construction_lifetime.assert_retained()
        selected = tuple(runtime for runtime in self.runtimes if runtime.payload.block.generation_digest == digest)
        if len(selected) != 1:
            raise ValueError("dense cached lane has no unique runtime for block")
        return selected[0]


@dataclass(frozen=True)
class _DenseAsyncFailureQuarantine:
    """Accumulator-owned fail-stop roots for work lacking a completion proof."""

    stage: str
    original_error: BaseException = field(repr=False)
    original_traceback: Any = field(repr=False)
    cleanup_fence_error: BaseException = field(repr=False)
    retained_reference_roles: tuple[str, ...]
    retained_references: tuple[Any, ...] = field(repr=False)
    device_completion_fence_provenance: str
    generation_digest: str
    restart_required: bool = True

    def assert_current(self) -> None:
        retained_by_role = dict(
            zip(
                self.retained_reference_roles,
                self.retained_references,
                strict=True,
            )
        )
        if (
            not self.stage.strip()
            or self.original_traceback is None
            or len(self.retained_reference_roles) != len(self.retained_references)
            or any(not role.strip() for role in self.retained_reference_roles)
            or not self.device_completion_fence_provenance.strip()
            or not self.restart_required
            or self.generation_digest != _async_failure_quarantine_digest(self)
        ):
            raise ValueError("dense async failure quarantine changed")
        targets = retained_by_role.get("current_chunk_targets")
        if isinstance(targets, PaperKineticDenseChunkTargets):
            targets.assert_transfer_retained()
        target_loader = retained_by_role.get("target_loader")
        target_load_lifetime = retained_by_role.get(
            "target_loader_active_lifetime"
        )
        if isinstance(target_loader, PaperKineticDenseChunkTargetLoader):
            target_loader.assert_current(
                target_loader._source_ref,
                target_loader._request_ref,
                device=target_loader.device,
            )
            if target_loader._active_lifetime is not target_load_lifetime:
                raise ValueError("dense target-load lifetime escaped quarantine")
        materialization = retained_by_role.get("current_sample_materialization")
        if (
            isinstance(materialization, _DenseSampleMaterializationLease)
            and not materialization.released_after_completion_fence
        ):
            materialization.assert_retained()
        native_session = retained_by_role.get("native_session")
        lifetime = retained_by_role.get("session_outstanding_sample_lifetime")
        if lifetime is None:
            lifetime = retained_by_role.get("current_sample_lifetime")
        if (
            isinstance(lifetime, KineticNativeSampleLaunchLifetime)
            and native_session is not None
            and not lifetime.consumed
        ):
            lifetime.assert_retained(native_session)
        lane_construction_lifetime = retained_by_role.get(
            "dense_lane_construction_lifetime"
        )
        if isinstance(
            lane_construction_lifetime,
            _DenseCachedNativeLaneConstructionLifetime,
        ):
            lane_construction_lifetime.assert_retained()
        active_blocks = retained_by_role.get("active_blocks")
        if isinstance(active_blocks, dict):
            for block_state in active_blocks.values():
                if isinstance(block_state, _ActiveBlock):
                    block_state.compact_gather_lifetime.assert_retained()
                    block_state.forward_into_lifetime.assert_retained(native_session)
        compact_gather_lifetime = retained_by_role.get(
            "current_compact_gather_lifetime"
        )
        if isinstance(
            compact_gather_lifetime,
            _DenseCompactMaterialGatherLifetime,
        ):
            compact_gather_lifetime.assert_retained()
        forward_into_lifetime = retained_by_role.get(
            "current_forward_into_lifetime"
        )
        if isinstance(
            forward_into_lifetime,
            KineticNativeNodeForwardIntoLifetime,
        ):
            forward_into_lifetime.assert_retained(native_session)
        reverse_block = retained_by_role.get("current_reverse_block_state")
        if (
            reverse_block is not None
            and (
                not isinstance(active_blocks, dict)
                or all(value is not reverse_block for value in active_blocks.values())
            )
        ):
            raise ValueError("dense reverse block escaped its retained active lane")


@dataclass
class PaperKineticDenseStepGradientAccumulator:
    """Zero-owned whole-step bars; only a sealed manifest can authorize use."""

    source_generation_digest: str
    compact_manifest_digest: str
    step_generation_id: str
    loss_normalization_id: str
    global_loss_element_count: int
    loss_scale: float
    material_generation_id: str
    background_generation_id: str
    material_tensor_identity: int
    material_tensor_signature: tuple[object, ...] = field(repr=False)
    background_tensor_identity: int
    background_tensor_signature: tuple[object, ...] = field(repr=False)
    world_generation_digest: str
    world_sites_content_digest: str
    site_table_identity: int
    ray_bar_keys: tuple[tuple[int, int], ...]
    ray_bar_keys_generation_digest: str
    full_geometry: bool
    optimize_camera_rays: bool
    grad_site_rgba_f32: torch.Tensor = field(repr=False)
    loss_f32: torch.Tensor = field(repr=False)
    grad_positions0_f64: torch.Tensor | None = field(repr=False)
    grad_velocities_f64: torch.Tensor | None = field(repr=False)
    grad_weight_coefficients_f64: torch.Tensor | None = field(repr=False)
    grad_track_ray_coefficients_f64: torch.Tensor | None = field(repr=False)
    tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    consumed_request_count: int
    consumed_observation_count: int
    fenced_request_commit_count: int
    request_commit_fence_provenance: str
    request_commit_chain_digest: str
    pending_request_generation_digest: str
    pending_delta_generation_digest: str
    poisoned: bool
    sealed: bool
    optimizer_authorized: bool
    generation_digest: str
    _source_identity: int = field(repr=False)
    _session_identity: int = field(repr=False)
    _ray_bar_keys_identity: int = field(repr=False)
    _material_tensor_ref: torch.Tensor = field(repr=False)
    _background_tensor_ref: torch.Tensor = field(repr=False)
    _request_commit_fence_identity: int = field(repr=False)
    _async_failure_quarantine: _DenseAsyncFailureQuarantine | None = field(
        repr=False
    )
    provenance: str = STEP_ACCUMULATOR_PROVENANCE
    _seal: object = field(default=None, repr=False)

    def _tensors(self) -> tuple[torch.Tensor, ...]:
        geometry = tuple(
            tensor
            for tensor in (
                self.grad_positions0_f64,
                self.grad_velocities_f64,
                self.grad_weight_coefficients_f64,
                self.grad_track_ray_coefficients_f64,
            )
            if tensor is not None
        )
        return (self.grad_site_rgba_f32, self.loss_f32, *geometry)

    @property
    def logical_tensor_bytes(self) -> int:
        return _tensor_bytes(*self._tensors())

    def assert_current(
        self,
        source: PaperKineticReplayableDenseObservationSource,
        session: PaperKineticDenseObservationReplaySession,
    ) -> None:
        source.assert_warm_current()
        session.assert_current()
        if self._async_failure_quarantine is not None:
            self._async_failure_quarantine.assert_current()
        sites = source.provider.world.sites
        geometry = self._tensors()[2:]
        if (
            self._seal is not _STEP_ACCUMULATOR_SEAL
            or self.provenance != STEP_ACCUMULATOR_PROVENANCE
            or id(source) != self._source_identity
            or id(session) != self._session_identity
            or session.source is not source
            or self.source_generation_digest != source.generation_digest
            or self.compact_manifest_digest != source.compact_manifest_digest
            or self.world_generation_digest
            != source.provider.world.generation_digest
            or self.world_sites_content_digest
            != source.provider.world.sites_content_digest
            or self.site_table_identity != id(sites)
            or id(self.ray_bar_keys) != self._ray_bar_keys_identity
            or not self.ray_bar_keys_generation_digest.strip()
            or not isinstance(self.full_geometry, bool)
            or not isinstance(self.optimize_camera_rays, bool)
            or (self.optimize_camera_rays and not self.full_geometry)
            or self.optimize_camera_rays != bool(self.ray_bar_keys)
            or not self.step_generation_id.strip()
            or not self.loss_normalization_id.strip()
            or self.global_loss_element_count <= 0
            or self.global_loss_element_count != source.observation_count * 3
            or self.loss_scale != 1.0 / float(self.global_loss_element_count)
            or not self.material_generation_id.strip()
            or not self.background_generation_id.strip()
            or self.material_tensor_identity <= 0
            or not self.material_tensor_signature
            or self.background_tensor_identity <= 0
            or not self.background_tensor_signature
            or id(self._material_tensor_ref) != self.material_tensor_identity
            or _tensor_signature(self._material_tensor_ref)
            != self.material_tensor_signature
            or id(self._background_tensor_ref) != self.background_tensor_identity
            or _tensor_signature(self._background_tensor_ref)
            != self.background_tensor_signature
            or self.consumed_request_count < 0
            or self.consumed_observation_count < 0
            or self.fenced_request_commit_count < 0
            or self.fenced_request_commit_count != self.consumed_request_count
            or bool(self.request_commit_fence_provenance)
            != bool(self.fenced_request_commit_count)
            or bool(self.request_commit_chain_digest)
            != bool(self.fenced_request_commit_count)
            or bool(self._request_commit_fence_identity)
            != bool(self.fenced_request_commit_count)
            or (
                self._async_failure_quarantine is not None
                and (
                    not self.poisoned
                    or self.sealed
                    or self.optimizer_authorized
                )
            )
            or self.consumed_request_count > session.request_count
            or self.consumed_observation_count > session.emitted_observation_count
            or bool(self.pending_request_generation_digest)
            != bool(self.pending_delta_generation_digest)
            or self.sealed != self.optimizer_authorized
            or self.sealed and not session.sealed
            or self.sealed and self.poisoned
            or tuple(_tensor_signature(tensor) for tensor in self._tensors())
            != self.tensor_signatures
            or self.generation_digest != _step_accumulator_digest(self)
        ):
            raise ValueError("dense step accumulator changed or is foreign")
        _require_tensor(
            self.grad_site_rgba_f32,
            name="step grad_site_rgba_f32",
            device=self.grad_site_rgba_f32.device,
            dtype=torch.float32,
            shape=(source.provider.world.site_count, 4),
        )
        _require_tensor(
            self.loss_f32,
            name="step loss_f32",
            device=self.grad_site_rgba_f32.device,
            dtype=torch.float32,
            shape=(1,),
        )
        if (
            self.grad_site_rgba_f32.requires_grad
            or self.loss_f32.requires_grad
            or not bool(torch.isfinite(self.grad_site_rgba_f32).all().item())
            or not bool(torch.isfinite(self.loss_f32).all().item())
        ):
            raise ValueError("dense step material/loss bars must be finite explicit bars")
        if self.full_geometry:
            expected_geometry_tensor_count = 4 if self.optimize_camera_rays else 3
            expected_ray_key_count = (
                source.selected_view_count * source.image_pixel_count
                if self.optimize_camera_rays
                else 0
            )
            if (
                len(geometry) != expected_geometry_tensor_count
                or len(self.ray_bar_keys) != expected_ray_key_count
            ):
                raise ValueError(
                    "full-geometry step accumulator lost its selected geometry bars"
                )
            _require_cpu_f64_tensor(
                self.grad_positions0_f64,
                name="step grad_positions0_f64",
                shape=tuple(sites.positions0.shape),
            )
            _require_cpu_f64_tensor(
                self.grad_velocities_f64,
                name="step grad_velocities_f64",
                shape=tuple(sites.velocities.shape),
            )
            _require_cpu_f64_tensor(
                self.grad_weight_coefficients_f64,
                name="step grad_weight_coefficients_f64",
                shape=tuple(sites.weight_coefficients.shape),
            )
            if self.optimize_camera_rays:
                _require_cpu_f64_tensor(
                    self.grad_track_ray_coefficients_f64,
                    name="step grad_track_ray_coefficients_f64",
                    shape=(len(self.ray_bar_keys), 12),
                )
            elif self.grad_track_ray_coefficients_f64 is not None:
                raise ValueError("fixed-camera geometry step retained a ray-gradient tensor")
        elif geometry or self.ray_bar_keys or self.optimize_camera_rays:
            raise ValueError("material-only step retained geometry state")
        _require_distinct_storage(*self._tensors())


@dataclass
class PaperKineticDenseRequestGradientDelta:
    """One request-local combined bar, single-use and free of any frame axis."""

    source_generation_digest: str
    request_generation_digest: str
    artifact_generation_digest: str
    step_generation_id: str
    receipt: PaperKineticDenseCachedRequestReceipt
    telemetry: KineticNativeMaterialStepTelemetry
    full_geometry: bool
    optimize_camera_rays: bool
    ray_bar_keys: tuple[tuple[int, int], ...]
    source_site_ids_i64: torch.Tensor | None = field(repr=False)
    grad_union_site_rgba_f32: torch.Tensor | None = field(repr=False)
    loss_f32: torch.Tensor | None = field(repr=False)
    grad_positions0_f64: torch.Tensor | None = field(repr=False)
    grad_velocities_f64: torch.Tensor | None = field(repr=False)
    grad_weight_coefficients_f64: torch.Tensor | None = field(repr=False)
    grad_track_ray_coefficients_f64: torch.Tensor | None = field(repr=False)
    sealed_tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    generation_digest: str
    consumed: bool
    consumed_by_accumulator_generation_digest: str
    _accumulator_identity: int = field(repr=False)
    provenance: str = REQUEST_DELTA_PROVENANCE
    persistent_frame_tensor_bytes: int = 0
    persistent_sample_tensor_bytes: int = 0
    persistent_target_tensor_bytes: int = 0
    persistent_prediction_tensor_bytes: int = 0
    _seal: object = field(default=None, repr=False)

    def _live_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            tensor
            for tensor in (
                self.source_site_ids_i64,
                self.grad_union_site_rgba_f32,
                self.loss_f32,
                self.grad_positions0_f64,
                self.grad_velocities_f64,
                self.grad_weight_coefficients_f64,
                self.grad_track_ray_coefficients_f64,
            )
            if tensor is not None
        )

    @property
    def logical_tensor_bytes(self) -> int:
        return _tensor_bytes(*self._live_tensors())

    def assert_current(
        self,
        accumulator: PaperKineticDenseStepGradientAccumulator,
        source: PaperKineticReplayableDenseObservationSource,
        request: PaperKineticDenseObservationTrackRequest,
        artifact: PaperKineticCompiledCpuArtifact,
        session: PaperKineticDenseObservationReplaySession,
    ) -> None:
        self.receipt.assert_current(source, request, artifact, session)
        self.telemetry.assert_current()
        expected_ray_bar_keys = (
            tuple((request.view_index, track_id) for track_id in request.track_ids)
            if self.full_geometry and self.optimize_camera_rays
            else ()
        )
        if (
            self._seal is not _REQUEST_DELTA_SEAL
            or self.provenance != REQUEST_DELTA_PROVENANCE
            or id(accumulator) != self._accumulator_identity
            or self.source_generation_digest != source.generation_digest
            or self.request_generation_digest != request.generation_digest
            or self.artifact_generation_digest != artifact.generation_digest
            or self.step_generation_id != accumulator.step_generation_id
            or self.full_geometry != accumulator.full_geometry
            or self.optimize_camera_rays != accumulator.optimize_camera_rays
            or (self.optimize_camera_rays and not self.full_geometry)
            or self.optimize_camera_rays != bool(self.ray_bar_keys)
            or self.full_geometry
            != (
                self.telemetry.reverse_mode
                in {"full_geometry", "fused_full_geometry"}
            )
            or self.ray_bar_keys != expected_ray_bar_keys
            or self.telemetry.loss_normalization_id
            != accumulator.loss_normalization_id
            or self.telemetry.global_loss_element_count
            != accumulator.global_loss_element_count
            or self.telemetry.loss_scale != accumulator.loss_scale
            or self.persistent_frame_tensor_bytes != 0
            or self.persistent_sample_tensor_bytes != 0
            or self.persistent_target_tensor_bytes != 0
            or self.persistent_prediction_tensor_bytes != 0
            or self.generation_digest != _request_delta_digest(self)
        ):
            raise ValueError("dense request gradient delta changed or is foreign")
        if self.consumed:
            if self._live_tensors() or not self.consumed_by_accumulator_generation_digest:
                raise ValueError("consumed dense request delta retained request tensors")
            return
        if self.consumed_by_accumulator_generation_digest:
            raise ValueError("unconsumed dense request delta has a consumer")
        material_tensors = (
            self.source_site_ids_i64,
            self.grad_union_site_rgba_f32,
            self.loss_f32,
        )
        geometry_tensors = (
            self.grad_positions0_f64,
            self.grad_velocities_f64,
            self.grad_weight_coefficients_f64,
        )
        if (
            any(tensor is None for tensor in material_tensors)
            or (
                self.full_geometry
                and any(tensor is None for tensor in geometry_tensors)
            )
            or (
                not self.full_geometry
                and any(tensor is not None for tensor in geometry_tensors)
            )
            or self.optimize_camera_rays
            != (self.grad_track_ray_coefficients_f64 is not None)
        ):
            raise ValueError("dense request gradient delta mode/tensors disagree")
        tensors = self._live_tensors()
        if tuple(_tensor_signature(tensor) for tensor in tensors) != self.sealed_tensor_signatures:
            raise ValueError("dense request gradient delta tensor state changed")
        expected_tensor_count = (
            7
            if self.full_geometry and self.optimize_camera_rays
            else 6
            if self.full_geometry
            else 3
        )
        if len(tensors) != expected_tensor_count:
            raise ValueError("dense request gradient delta mode/tensors disagree")
        _require_distinct_storage(*tensors)


@dataclass(frozen=True)
class PaperKineticDenseRequestDeltaCommitReceipt:
    """Tensor-free proof that one whole-step delta commit completed on device."""

    source_generation_digest: str
    request_generation_digest: str
    artifact_generation_digest: str
    step_generation_id: str
    delta_generation_digest: str
    accumulator_generation_digest_after_commit: str
    request_commit_chain_digest_after_commit: str
    consumed_request_count: int
    consumed_observation_count: int
    device_completion_fence_provenance: str
    device_completion_fence_call_count: int
    generation_digest: str
    _accumulator_identity: int = field(repr=False)
    provenance: str = REQUEST_DELTA_COMMIT_PROVENANCE
    persistent_tensor_bytes: int = 0
    delta_tensors_released_after_fence: bool = True
    _seal: object = field(default=None, repr=False)

    def assert_current(
        self,
        accumulator: PaperKineticDenseStepGradientAccumulator,
        source: PaperKineticReplayableDenseObservationSource,
        session: PaperKineticDenseObservationReplaySession,
        request: PaperKineticDenseObservationTrackRequest,
        artifact: PaperKineticCompiledCpuArtifact,
        delta: PaperKineticDenseRequestGradientDelta,
    ) -> None:
        accumulator.assert_current(source, session)
        delta.assert_current(accumulator, source, request, artifact, session)
        if (
            self._seal is not _REQUEST_DELTA_COMMIT_SEAL
            or self.provenance != REQUEST_DELTA_COMMIT_PROVENANCE
            or id(accumulator) != self._accumulator_identity
            or self.source_generation_digest != source.generation_digest
            or self.request_generation_digest != request.generation_digest
            or self.artifact_generation_digest != artifact.generation_digest
            or self.step_generation_id != accumulator.step_generation_id
            or self.delta_generation_digest != delta.generation_digest
            or not delta.consumed
            or delta.consumed_by_accumulator_generation_digest
            != self.accumulator_generation_digest_after_commit
            or self.consumed_request_count
            != delta.receipt.session_request_count_after
            or self.consumed_observation_count
            != delta.receipt.session_emitted_observation_count_after
            or accumulator.consumed_request_count < self.consumed_request_count
            or accumulator.consumed_observation_count
            < self.consumed_observation_count
            or accumulator.fenced_request_commit_count < self.consumed_request_count
            or not self.device_completion_fence_provenance.strip()
            or self.device_completion_fence_call_count != 1
            or self.persistent_tensor_bytes != 0
            or not self.delta_tensors_released_after_fence
            or self.generation_digest != _request_delta_commit_receipt_digest(self)
        ):
            raise ValueError("dense request-delta commit receipt changed or is foreign")


@dataclass(frozen=True)
class PaperKineticDenseOptimizerAuthorization:
    """Point-in-time permission; revalidation never mutates stale step bars."""

    source_generation_digest: str
    compact_manifest_digest: str
    step_generation_id: str
    replay_receipt_generation_digest: str
    accumulator_generation_digest: str
    request_count: int
    observation_count: int
    full_geometry: bool
    optimize_camera_rays: bool
    grad_site_rgba_f32: torch.Tensor = field(repr=False)
    loss_f32: torch.Tensor = field(repr=False)
    grad_positions0_f64: torch.Tensor | None = field(repr=False)
    grad_velocities_f64: torch.Tensor | None = field(repr=False)
    grad_weight_coefficients_f64: torch.Tensor | None = field(repr=False)
    ray_bar_keys: tuple[tuple[int, int], ...]
    grad_track_ray_coefficients_f64: torch.Tensor | None = field(repr=False)
    tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    generation_digest: str
    _accumulator_identity: int = field(repr=False)
    provenance: str = OPTIMIZER_AUTHORIZATION_PROVENANCE
    _seal: object = field(default=None, repr=False)

    def _tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            tensor
            for tensor in (
                self.grad_site_rgba_f32,
                self.loss_f32,
                self.grad_positions0_f64,
                self.grad_velocities_f64,
                self.grad_weight_coefficients_f64,
                self.grad_track_ray_coefficients_f64,
            )
            if tensor is not None
        )

    def assert_current(
        self,
        accumulator: PaperKineticDenseStepGradientAccumulator,
        replay_receipt: PaperKineticDenseObservationReplayReceipt,
    ) -> None:
        if (
            self._seal is not _OPTIMIZER_AUTHORIZATION_SEAL
            or self.provenance != OPTIMIZER_AUTHORIZATION_PROVENANCE
            or id(accumulator) != self._accumulator_identity
            or not accumulator.sealed
            or not accumulator.optimizer_authorized
            or accumulator.poisoned
            or id(accumulator._material_tensor_ref)
            != accumulator.material_tensor_identity
            or _tensor_signature(accumulator._material_tensor_ref)
            != accumulator.material_tensor_signature
            or id(accumulator._background_tensor_ref)
            != accumulator.background_tensor_identity
            or _tensor_signature(accumulator._background_tensor_ref)
            != accumulator.background_tensor_signature
            or self.source_generation_digest != accumulator.source_generation_digest
            or self.compact_manifest_digest != accumulator.compact_manifest_digest
            or self.step_generation_id != accumulator.step_generation_id
            or self.replay_receipt_generation_digest != replay_receipt.generation_digest
            or self.accumulator_generation_digest != accumulator.generation_digest
            or self.request_count != replay_receipt.request_count
            or self.observation_count != replay_receipt.observation_count
            or self.request_count != accumulator.consumed_request_count
            or self.observation_count != accumulator.consumed_observation_count
            or self.full_geometry != accumulator.full_geometry
            or self.optimize_camera_rays != accumulator.optimize_camera_rays
            or self.ray_bar_keys != accumulator.ray_bar_keys
            or self.optimize_camera_rays
            != bool(self.ray_bar_keys)
            or self.optimize_camera_rays
            != (self.grad_track_ray_coefficients_f64 is not None)
            or (self.optimize_camera_rays and not self.full_geometry)
            or self.tensor_signatures != accumulator.tensor_signatures
            or tuple(_tensor_signature(tensor) for tensor in self._tensors())
            != self.tensor_signatures
            or self.generation_digest != _optimizer_authorization_digest(self)
        ):
            raise ValueError("dense optimizer authorization changed or is foreign")
        _require_distinct_storage(*self._tensors())


@dataclass(frozen=True)
class PaperKineticDenseCachedRequestReceipt:
    """Proof that one canonical track request fully advanced the replay cursor."""

    source_generation_digest: str
    request_generation_digest: str
    artifact_generation_digest: str
    session_identity: int
    session_request_count_before: int
    session_request_count_after: int
    session_emitted_observation_count_before: int
    session_emitted_observation_count_after: int
    expected_observation_count: int
    replay_chunk_count: int
    replay_chunk_manifest_digest: str
    generation_digest: str
    provenance: str = REQUEST_PROVENANCE
    _seal: object = field(default=None, repr=False)

    def assert_current(
        self,
        source: PaperKineticReplayableDenseObservationSource,
        request: PaperKineticDenseObservationTrackRequest,
        artifact: PaperKineticCompiledCpuArtifact,
        session: PaperKineticDenseObservationReplaySession,
    ) -> None:
        source.assert_warm_current()
        request.assert_current(source)
        artifact.assert_warm_reusable_with_provider(source.provider)
        session.assert_current()
        if (
            self._seal is not _RECEIPT_SEAL
            or self.provenance != REQUEST_PROVENANCE
            or self.source_generation_digest != source.generation_digest
            or self.request_generation_digest != request.generation_digest
            or self.artifact_generation_digest != artifact.generation_digest
            or self.session_identity != id(session)
            or self.session_request_count_after != self.session_request_count_before + 1
            or session.request_count < self.session_request_count_after
            or self.session_emitted_observation_count_after - self.session_emitted_observation_count_before
            != self.expected_observation_count
            or session.emitted_observation_count < self.session_emitted_observation_count_after
            or self.expected_observation_count < 1
            or self.replay_chunk_count < 1
            or not self.replay_chunk_manifest_digest.strip()
            or self.generation_digest != _receipt_digest(self)
        ):
            raise ValueError("dense cached request receipt changed or is foreign")


@dataclass(frozen=True)
class PaperKineticDenseCachedNativeRequestResult:
    """Bounded material-only or full-geometry request result."""

    source_generation_digest: str
    request_generation_digest: str
    artifact_generation_digest: str
    lane_generation_digest: str
    receipt: PaperKineticDenseCachedRequestReceipt
    telemetry: KineticNativeMaterialStepTelemetry
    delta: PaperKineticDenseRequestGradientDelta = field(repr=False)
    loss_delta_f32: float
    accounting: Mapping[str, Any]
    generation_digest: str
    full_geometry_reverse_mode: str = STAGED_SPARSE_FULL_GEOMETRY_REVERSE
    provenance: str = REQUEST_PROVENANCE
    runtime_status: str = REQUEST_STATUS
    production_trainer_integrated: bool = False
    full_geometry_vjp_integrated: bool = False
    post_dedup_runtime_verified: bool = False
    production_promotion_allowed: bool = False
    native_runtime_verified: bool = False
    allocator_peak_measured: bool = False
    target_loader_is_arbitrary_callable: bool = False
    target_loader_partial_failure_lifetime_certified: bool = True
    decoder_allocator_peak_measured: bool = False
    sample_materialization_float64_scratch_measured: bool = False
    whole_step_python_object_peak_measured: bool = False
    _seal: object = field(default=None, repr=False)

    def assert_current(
        self,
        source: PaperKineticReplayableDenseObservationSource,
        request: PaperKineticDenseObservationTrackRequest,
        artifact: PaperKineticCompiledCpuArtifact,
        session: PaperKineticDenseObservationReplaySession,
        accumulator: PaperKineticDenseStepGradientAccumulator,
    ) -> None:
        self.receipt.assert_current(source, request, artifact, session)
        self.telemetry.assert_current()
        self.delta.assert_current(
            accumulator,
            source,
            request,
            artifact,
            session,
        )
        if (
            self._seal is not _RESULT_SEAL
            or self.provenance != REQUEST_PROVENANCE
            or self.runtime_status != REQUEST_STATUS
            or self.source_generation_digest != source.generation_digest
            or self.request_generation_digest != request.generation_digest
            or self.artifact_generation_digest != artifact.generation_digest
            or not self.lane_generation_digest.strip()
            or not math.isfinite(self.loss_delta_f32)
            or self.loss_delta_f32 < 0.0
            or self.full_geometry_reverse_mode not in _FULL_GEOMETRY_REVERSE_MODES
            or (
                self.telemetry.reverse_mode == "fused_full_geometry"
            )
            != (
                self.full_geometry_reverse_mode
                == FUSED_DIRECT_V1_FULL_GEOMETRY_REVERSE
            )
            or self.production_trainer_integrated
            or self.full_geometry_vjp_integrated
            != (
                self.telemetry.reverse_mode
                in {"full_geometry", "fused_full_geometry"}
            )
            or self.delta.receipt is not self.receipt
            or self.delta.telemetry is not self.telemetry
            or bool(self.accounting.get("full_geometry_vjp_integrated"))
            != self.full_geometry_vjp_integrated
            or self.accounting.get("full_geometry_reverse_mode")
            != self.full_geometry_reverse_mode
            or bool(self.accounting.get("camera_ray_gradients_enabled"))
            != accumulator.optimize_camera_rays
            or bool(self.accounting.get("fixed_camera_avoids_global_ray_bar"))
            != (accumulator.full_geometry and not accumulator.optimize_camera_rays)
            or int(
                self.accounting.get(
                    "native_full_geometry_vjp_launch_count",
                    -1,
                )
            )
            != self.telemetry.native_full_geometry_vjp_launch_count
            or int(
                self.accounting.get(
                    "native_fused_full_geometry_vjp_launch_count",
                    -1,
                )
            )
            != self.telemetry.native_fused_full_geometry_vjp_launch_count
            or int(
                self.accounting.get(
                    "native_fused_full_geometry_transaction_count",
                    -1,
                )
            )
            != self.telemetry.native_fused_full_geometry_transaction_count
            or int(
                self.accounting.get(
                    "native_material_word_vjp_launch_count",
                    -1,
                )
            )
            != self.telemetry.native_material_word_vjp_launch_count
            or self.accounting.get("node_forward_abi")
            != "caller_preallocated_into_v1"
            or int(
                self.accounting.get(
                    "return_allocating_node_forward_launch_count",
                    -1,
                )
            )
            != 0
            or int(
                self.accounting.get(
                    "caller_preallocated_node_forward_launch_count",
                    -1,
                )
            )
            != self.telemetry.native_node_forward_launch_count
            or int(
                self.accounting.get("forward_into_lifetime_install_count", -1)
            )
            != self.telemetry.active_native_block_count
            or int(
                self.accounting.get("forward_into_lifetime_retire_count", -1)
            )
            != self.telemetry.active_native_block_count
            or int(
                self.accounting.get("compact_gather_lifetime_install_count", -1)
            )
            != self.telemetry.active_native_block_count
            or int(
                self.accounting.get("compact_gather_lifetime_retire_count", -1)
            )
            != self.telemetry.active_native_block_count
            or int(
                self.accounting.get(
                    "retained_forward_into_lifetime_count_after_request",
                    -1,
                )
            )
            != 0
            or int(
                self.accounting.get(
                    "retained_compact_gather_lifetime_count_after_request",
                    -1,
                )
            )
            != 0
            or int(
                self.accounting.get(
                    "forward_into_lifetime_additional_logical_tensor_bytes",
                    -1,
                )
            )
            != 0
            or int(
                self.accounting.get(
                    "compact_gather_lifetime_additional_logical_tensor_bytes",
                    -1,
                )
            )
            != 0
            or self.accounting.get(
                "forward_predecessor_and_output_roots_released_only_after_reverse_or_abort_fence"
            )
            is not True
            or self.accounting.get("native_lane_two_phase_construction")
            is not True
            or self.accounting.get(
                "union_and_runtime_construction_lifetimes_retained_through_lane_fence"
            )
            is not True
            or self.accounting.get("accelerator_release_capability_integrated")
            is not False
            or int(
                self.accounting.get(
                    "lane_two_phase_construction_predecessor_logical_tensor_bytes_upper_bound",
                    -1,
                )
            )
            != _lane_two_phase_construction_predecessor_upper_bound_bytes(
                artifact
            )
            or self.accounting.get(
                "lane_two_phase_construction_predecessors_overlap_active_request"
            )
            is not True
            or self.post_dedup_runtime_verified
            or self.production_promotion_allowed
            or self.native_runtime_verified
            or self.allocator_peak_measured
            or self.target_loader_is_arbitrary_callable
            or not self.target_loader_partial_failure_lifetime_certified
            or self.decoder_allocator_peak_measured
            or self.sample_materialization_float64_scratch_measured
            or self.whole_step_python_object_peak_measured
            or bool(self.accounting.get("target_loader_is_arbitrary_callable"))
            != self.target_loader_is_arbitrary_callable
            or self.accounting.get(
                "target_loader_partial_failure_lifetime_certified"
            )
            is not True
            or self.accounting.get("target_loader_provenance")
            != TARGET_LOADER_PROVENANCE
            or int(
                self.accounting.get("target_loader_completed_load_count", -1)
            )
            != self.receipt.replay_chunk_count
            or int(
                self.accounting.get(
                    "target_loader_failed_after_enqueue_count",
                    -1,
                )
            )
            != 0
            or int(
                self.accounting.get(
                    "target_loader_maximum_outstanding_lifetime_count",
                    -1,
                )
            )
            != 1
            or int(
                self.accounting.get(
                    "target_loader_retained_lifetime_count_after_request",
                    -1,
                )
            )
            != 0
            or self.accounting.get(
                "target_loader_transfer_roots_released_only_after_completion_fence"
            )
            is not True
            or int(
                self.accounting.get(
                    "target_loader_lifetime_additional_logical_tensor_bytes",
                    -1,
                )
            )
            != 0
            or self.accounting.get(
                "target_loader_lifetime_python_heap_bytes_measured"
            )
            is not False
            or self.accounting.get("target_loader_test_fault_enabled") is not False
            or self.accounting.get(
                "target_loader_retained_closure_state_measured"
            )
            is not True
            or int(
                self.accounting.get(
                    "native_sample_completion_fence_count",
                    -1,
                )
            )
            != self.telemetry.native_sample_completion_fence_count
            or int(
                self.accounting.get(
                    "maximum_in_flight_sample_lifetime_token_count",
                    -1,
                )
            )
            != self.telemetry.maximum_simultaneous_sample_lifetime_count
            or int(
                self.accounting.get(
                    "retained_sample_lifetime_token_count_after_seal",
                    -1,
                )
            )
            != 0
            or self.accounting.get(
                "sample_lifetime_token_history_retained"
            )
            is not False
            or int(
                self.accounting.get(
                    "sample_lifetime_additional_logical_tensor_bytes",
                    -1,
                )
            )
            != 0
            or self.accounting.get(
                "sample_lifetime_python_heap_bytes_measured"
            )
            is not False
            or self.accounting.get(
                "sample_lifetime_roots_released_only_after_completion_fence"
            )
            is not True
            or self.accounting.get(
                "sample_materialization_predecessor_roots_leased_until_fence"
            )
            is not True
            or self.accounting.get(
                "chunk_cpu_transfer_source_retained_through_sample_fences"
            )
            is not True
            or bool(self.accounting.get("decoder_allocator_peak_measured"))
            != self.decoder_allocator_peak_measured
            or bool(
                self.accounting.get(
                    "sample_materialization_float64_scratch_measured"
                )
            )
            != self.sample_materialization_float64_scratch_measured
            or not bool(
                self.accounting.get(
                    "sample_materialization_source_visible_logical_tensors_accounted"
                )
            )
            or self.accounting.get(
                "target_source_decode_budget_enforced_before_allocation"
            )
            is not True
            or int(self.accounting.get("selected_pixel_read_call_count", -1))
            != self.receipt.replay_chunk_count
            or sum(
                int(self.accounting.get(key, -1))
                for key in (
                    "direct_selected_pixel_observation_count",
                    "bounded_region_selected_pixel_observation_count",
                    "full_frame_fallback_observation_count",
                )
            )
            != self.receipt.expected_observation_count
            or int(
                self.accounting.get(
                    "full_frame_target_materialization_count",
                    -1,
                )
            )
            != int(self.accounting.get("decoded_frame_count", -2))
            or bool(
                self.accounting.get("selected_pixel_read_acceptance_capable")
            )
            != (
                self.accounting.get("selected_pixel_read_mode")
                in {"direct_pixels", "certified_bounded_region"}
                and int(
                    self.accounting.get(
                        "full_frame_target_materialization_count",
                        -1,
                    )
                )
                == 0
                and int(
                    self.accounting.get(
                        "full_frame_fallback_observation_count",
                        -1,
                    )
                )
                == 0
            )
            or bool(self.accounting.get("whole_step_python_object_peak_measured"))
            != self.whole_step_python_object_peak_measured
            or self.generation_digest != _result_digest(self)
        ):
            raise ValueError("dense cached native request result changed")


@dataclass
class _DenseCompactMaterialGatherLifetime:
    """Retain gather predecessors/results until a proven reverse/abort fence."""

    global_site_rgba_f32: torch.Tensor | None = field(repr=False)
    source_site_ids_i64: torch.Tensor | None = field(repr=False)
    index_select_result_f32: torch.Tensor | None = field(default=None, repr=False)
    compact_site_rgba_f32: torch.Tensor | None = field(default=None, repr=False)
    phase: str = "installed"
    completion_fenced: bool = False
    _global_identity: int = field(default=0, repr=False)
    _indices_identity: int = field(default=0, repr=False)
    _gathered_identity: int | None = field(default=None, repr=False)
    _compact_identity: int | None = field(default=None, repr=False)
    _seal: object = field(default=None, repr=False)

    def assert_retained(self) -> None:
        if self._seal is not _COMPACT_GATHER_LIFETIME_SEAL:
            raise ValueError("dense compact material gather lifetime was not sealed")
        if self.phase == "released":
            if (
                not self.completion_fenced
                or self.global_site_rgba_f32 is not None
                or self.source_site_ids_i64 is not None
                or self.index_select_result_f32 is not None
                or self.compact_site_rgba_f32 is not None
            ):
                raise ValueError("released dense compact gather retained roots")
            return
        if self.phase not in {"installed", "gathered", "materialized"}:
            raise ValueError("dense compact material gather phase changed")
        if (
            self.completion_fenced
            or not isinstance(self.global_site_rgba_f32, torch.Tensor)
            or id(self.global_site_rgba_f32) != self._global_identity
            or not isinstance(self.source_site_ids_i64, torch.Tensor)
            or id(self.source_site_ids_i64) != self._indices_identity
            or (self.phase in {"gathered", "materialized"})
            != isinstance(self.index_select_result_f32, torch.Tensor)
            or (self.phase == "materialized")
            != isinstance(self.compact_site_rgba_f32, torch.Tensor)
        ):
            raise ValueError("dense compact material gather roots changed")
        if (
            isinstance(self.index_select_result_f32, torch.Tensor)
            and id(self.index_select_result_f32) != self._gathered_identity
        ):
            raise ValueError("dense compact material gathered result changed")
        if (
            isinstance(self.compact_site_rgba_f32, torch.Tensor)
            and id(self.compact_site_rgba_f32) != self._compact_identity
        ):
            raise ValueError("dense compact material result changed")

    def publish_gathered(self, gathered_f32: torch.Tensor) -> None:
        self.assert_retained()
        if self.phase != "installed" or not isinstance(gathered_f32, torch.Tensor):
            raise ValueError("dense compact gather publication is not current")
        self.index_select_result_f32 = gathered_f32
        self._gathered_identity = id(gathered_f32)
        self.phase = "gathered"
        self.assert_retained()

    def publish_materialized(self, compact_f32: torch.Tensor) -> None:
        self.assert_retained()
        if self.phase != "gathered" or not isinstance(compact_f32, torch.Tensor):
            raise ValueError("dense compact material publication is not current")
        self.compact_site_rgba_f32 = compact_f32
        self._compact_identity = id(compact_f32)
        self.phase = "materialized"
        self.assert_retained()

    def retire_after_completion_fence(self) -> None:
        self.assert_retained()
        self.completion_fenced = True
        self.global_site_rgba_f32 = None
        self.source_site_ids_i64 = None
        self.index_select_result_f32 = None
        self.compact_site_rgba_f32 = None
        self.phase = "released"
        self.assert_retained()


@dataclass
class _ActiveBlock:
    token: KineticNativeMaterialStepWorldToken
    grad_node_chart_f32: torch.Tensor
    loss_f32: torch.Tensor
    compact_gather_lifetime: _DenseCompactMaterialGatherLifetime
    forward_into_lifetime: KineticNativeNodeForwardIntoLifetime


def prepare_paper_kinetic_dense_step_gradient_accumulator(
    source: PaperKineticReplayableDenseObservationSource,
    session: PaperKineticDenseObservationReplaySession,
    *,
    step_generation_id: str,
    loss_normalization_id: str,
    material_generation_id: str,
    background_generation_id: str,
    global_site_rgba_f32: torch.Tensor,
    background_rgb_f32: torch.Tensor,
    device: torch.device | str,
    full_geometry: bool,
    optimize_camera_rays: bool = False,
) -> PaperKineticDenseStepGradientAccumulator:
    """Create fresh zero-owned step bars before the first replay request."""

    if not isinstance(source, PaperKineticReplayableDenseObservationSource):
        raise TypeError("dense step accumulator requires a replayable source")
    if not isinstance(session, PaperKineticDenseObservationReplaySession):
        raise TypeError("dense step accumulator requires a replay session")
    source.assert_current()
    session.assert_current()
    if session.source is not source or session.request_count or session.emitted_observation_count:
        raise ValueError("dense step accumulator must own a fresh replay session")
    if session.poisoned or session.sealed or session._active_request:
        raise ValueError("dense step accumulator requires an open replay session")
    for name, value in (
        ("step_generation_id", step_generation_id),
        ("loss_normalization_id", loss_normalization_id),
        ("material_generation_id", material_generation_id),
        ("background_generation_id", background_generation_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be nonempty")
    if not isinstance(full_geometry, bool):
        raise TypeError("full_geometry must be bool")
    if not isinstance(optimize_camera_rays, bool):
        raise TypeError("optimize_camera_rays must be bool")
    if optimize_camera_rays and not full_geometry:
        raise ValueError("camera-ray gradients require full_geometry=True")
    resolved_device = torch.device(device)
    sites = source.provider.world.sites
    _require_tensor(
        global_site_rgba_f32,
        name="global_site_rgba_f32",
        device=resolved_device,
        dtype=torch.float32,
        shape=(source.provider.world.site_count, 4),
    )
    _require_tensor(
        background_rgb_f32,
        name="background_rgb_f32",
        device=resolved_device,
        dtype=torch.float32,
        shape=(3,),
    )
    if any(
        tensor.requires_grad
        for tensor in (global_site_rgba_f32, background_rgb_f32)
    ):
        raise ValueError("dense step owns explicit bars and forbids autograd tensors")
    if any(
        not bool(torch.isfinite(tensor).all().item())
        for tensor in (global_site_rgba_f32, background_rgb_f32)
    ):
        raise ValueError("dense step material/background snapshot must be finite")
    _require_distinct_storage(global_site_rgba_f32, background_rgb_f32)
    grad_site = torch.zeros(
        (source.provider.world.site_count, 4),
        dtype=torch.float32,
        device=resolved_device,
    )
    loss = torch.zeros((1,), dtype=torch.float32, device=resolved_device)
    ray_keys = (
        _expected_step_ray_bar_keys(source)
        if full_geometry and optimize_camera_rays
        else ()
    )
    if full_geometry:
        grad_positions = torch.zeros_like(
            sites.positions0,
            device="cpu",
            dtype=torch.float64,
        )
        grad_velocities = torch.zeros_like(
            sites.velocities,
            device="cpu",
            dtype=torch.float64,
        )
        grad_weights = torch.zeros_like(
            sites.weight_coefficients,
            device="cpu",
            dtype=torch.float64,
        )
        grad_rays = (
            torch.zeros(
                (len(ray_keys), 12),
                dtype=torch.float64,
                device="cpu",
            )
            if optimize_camera_rays
            else None
        )
    else:
        grad_positions = None
        grad_velocities = None
        grad_weights = None
        grad_rays = None
    provisional = PaperKineticDenseStepGradientAccumulator(
        source_generation_digest=source.generation_digest,
        compact_manifest_digest=source.compact_manifest_digest,
        step_generation_id=step_generation_id,
        loss_normalization_id=loss_normalization_id,
        global_loss_element_count=source.observation_count * 3,
        loss_scale=1.0 / float(source.observation_count * 3),
        material_generation_id=material_generation_id,
        background_generation_id=background_generation_id,
        material_tensor_identity=id(global_site_rgba_f32),
        material_tensor_signature=_tensor_signature(global_site_rgba_f32),
        background_tensor_identity=id(background_rgb_f32),
        background_tensor_signature=_tensor_signature(background_rgb_f32),
        world_generation_digest=source.provider.world.generation_digest,
        world_sites_content_digest=source.provider.world.sites_content_digest,
        site_table_identity=id(sites),
        ray_bar_keys=ray_keys,
        ray_bar_keys_generation_digest=_digest_parts(
            STEP_ACCUMULATOR_PROVENANCE,
            "ray-bar-keys",
            ray_keys,
        ),
        full_geometry=full_geometry,
        optimize_camera_rays=optimize_camera_rays,
        grad_site_rgba_f32=grad_site,
        loss_f32=loss,
        grad_positions0_f64=grad_positions,
        grad_velocities_f64=grad_velocities,
        grad_weight_coefficients_f64=grad_weights,
        grad_track_ray_coefficients_f64=grad_rays,
        tensor_signatures=(),
        consumed_request_count=0,
        consumed_observation_count=0,
        fenced_request_commit_count=0,
        request_commit_fence_provenance="",
        request_commit_chain_digest="",
        pending_request_generation_digest="",
        pending_delta_generation_digest="",
        poisoned=False,
        sealed=False,
        optimizer_authorized=False,
        generation_digest="",
        _source_identity=id(source),
        _session_identity=id(session),
        _ray_bar_keys_identity=id(ray_keys),
        _material_tensor_ref=global_site_rgba_f32,
        _background_tensor_ref=background_rgb_f32,
        _request_commit_fence_identity=0,
        _async_failure_quarantine=None,
        _seal=_STEP_ACCUMULATOR_SEAL,
    )
    provisional.tensor_signatures = tuple(
        _tensor_signature(tensor) for tensor in provisional._tensors()
    )
    provisional.generation_digest = _step_accumulator_digest(provisional)
    provisional.assert_current(source, session)
    return provisional


@torch.no_grad()
def consume_paper_kinetic_dense_request_delta(
    accumulator: PaperKineticDenseStepGradientAccumulator,
    source: PaperKineticReplayableDenseObservationSource,
    session: PaperKineticDenseObservationReplaySession,
    request: PaperKineticDenseObservationTrackRequest,
    artifact: PaperKineticCompiledCpuArtifact,
    delta: PaperKineticDenseRequestGradientDelta,
    *,
    device_completion_fence: Callable[[], None],
    device_completion_fence_provenance: str,
) -> PaperKineticDenseRequestDeltaCommitReceipt:
    """Consume exactly the pending request; failure permanently poisons the step."""

    accumulator.assert_current(source, session)
    if accumulator.poisoned or accumulator.sealed:
        raise ValueError("dense step accumulator is not open")
    if not isinstance(delta, PaperKineticDenseRequestGradientDelta):
        raise TypeError("dense step consumer requires a sealed request delta")
    if not callable(device_completion_fence):
        raise TypeError("dense step consumer requires a device completion fence")
    if (
        not isinstance(device_completion_fence_provenance, str)
        or not device_completion_fence_provenance.strip()
    ):
        raise ValueError("device_completion_fence_provenance must be nonempty")
    if accumulator.grad_site_rgba_f32.device.type == "mps" and (
        device_completion_fence is not synchronize_mps_device_completion_fence
        or device_completion_fence_provenance
        != MPS_DEVICE_COMPLETION_FENCE_PROVENANCE
    ):
        raise ValueError("MPS delta commit requires the canonical completion fence")
    if accumulator.fenced_request_commit_count and (
        accumulator._request_commit_fence_identity != id(device_completion_fence)
        or accumulator.request_commit_fence_provenance
        != device_completion_fence_provenance
    ):
        raise ValueError("dense step delta commit changed its completion fence")
    delta.assert_current(accumulator, source, request, artifact, session)
    if delta.consumed:
        raise ValueError("dense request delta was already consumed")
    if (
        accumulator.pending_request_generation_digest
        != delta.request_generation_digest
        or accumulator.pending_delta_generation_digest != delta.generation_digest
        or delta.receipt.session_request_count_before
        != accumulator.consumed_request_count
        or delta.receipt.session_emitted_observation_count_before
        != accumulator.consumed_observation_count
    ):
        raise ValueError("dense request delta is duplicate, out of order, or foreign")
    commit_fence_completed = False
    try:
        source_ids = delta.source_site_ids_i64
        union_bar = delta.grad_union_site_rgba_f32
        request_loss = delta.loss_f32
        if source_ids is None or union_bar is None or request_loss is None:
            raise ValueError("dense request delta lost its material bars")
        _require_tensor(
            source_ids,
            name="request source_site_ids_i64",
            device=accumulator.grad_site_rgba_f32.device,
            dtype=torch.int64,
            shape=(int(union_bar.shape[0]),),
        )
        _require_tensor(
            union_bar,
            name="request grad_union_site_rgba_f32",
            device=accumulator.grad_site_rgba_f32.device,
            dtype=torch.float32,
            shape=(int(source_ids.numel()), 4),
        )
        _require_tensor(
            request_loss,
            name="request loss_f32",
            device=accumulator.loss_f32.device,
            dtype=torch.float32,
            shape=(1,),
        )
        if (
            union_bar.requires_grad
            or request_loss.requires_grad
            or not bool(torch.isfinite(union_bar).all().item())
            or not bool(torch.isfinite(request_loss).all().item())
            or bool(((source_ids < 0) | (source_ids >= accumulator.grad_site_rgba_f32.shape[0])).any().item())
        ):
            raise ValueError("dense request material delta is nonfinite or out of bounds")
        if delta.full_geometry:
            sites = source.provider.world.sites
            site_geometry = (
                delta.grad_positions0_f64,
                delta.grad_velocities_f64,
                delta.grad_weight_coefficients_f64,
            )
            if any(tensor is None for tensor in site_geometry):
                raise ValueError("dense request delta lost geometry bars")
            for tensor, name, shape in (
                (site_geometry[0], "request grad_positions0_f64", tuple(sites.positions0.shape)),
                (site_geometry[1], "request grad_velocities_f64", tuple(sites.velocities.shape)),
                (site_geometry[2], "request grad_weight_coefficients_f64", tuple(sites.weight_coefficients.shape)),
            ):
                _require_cpu_f64_tensor(tensor, name=name, shape=shape)
            if delta.optimize_camera_rays:
                _require_cpu_f64_tensor(
                    delta.grad_track_ray_coefficients_f64,
                    name="request grad_track_ray_coefficients_f64",
                    shape=(len(delta.ray_bar_keys), 12),
                )
                if not set(delta.ray_bar_keys).issubset(accumulator.ray_bar_keys):
                    raise ValueError("dense request delta has foreign ray bars")
            elif (
                delta.ray_bar_keys
                or delta.grad_track_ray_coefficients_f64 is not None
            ):
                raise ValueError("fixed-camera request retained ray bars")
        accumulator.grad_site_rgba_f32.index_add_(0, source_ids, union_bar)
        accumulator.loss_f32.add_(request_loss)
        if delta.full_geometry:
            accumulator.grad_positions0_f64.add_(delta.grad_positions0_f64)
            accumulator.grad_velocities_f64.add_(delta.grad_velocities_f64)
            accumulator.grad_weight_coefficients_f64.add_(
                delta.grad_weight_coefficients_f64
            )
            if delta.optimize_camera_rays:
                ray_position = {
                    key: index
                    for index, key in enumerate(accumulator.ray_bar_keys)
                }
                for request_index, key in enumerate(delta.ray_bar_keys):
                    accumulator.grad_track_ray_coefficients_f64[
                        ray_position[key]
                    ].add_(
                        delta.grad_track_ray_coefficients_f64[request_index]
                    )
        returned = device_completion_fence()
        if returned is not None:
            raise TypeError("dense request-delta commit fence must return None")
        commit_fence_completed = True
        accumulator.consumed_request_count = delta.receipt.session_request_count_after
        accumulator.consumed_observation_count = (
            delta.receipt.session_emitted_observation_count_after
        )
        accumulator.pending_request_generation_digest = ""
        accumulator.pending_delta_generation_digest = ""
        accumulator.fenced_request_commit_count += 1
        accumulator.request_commit_fence_provenance = (
            device_completion_fence_provenance
        )
        accumulator._request_commit_fence_identity = id(device_completion_fence)
        accumulator.request_commit_chain_digest = _digest_parts(
            REQUEST_DELTA_COMMIT_PROVENANCE,
            accumulator.request_commit_chain_digest,
            delta.generation_digest,
            accumulator.fenced_request_commit_count,
            accumulator.consumed_request_count,
            accumulator.consumed_observation_count,
            device_completion_fence_provenance,
        )
        accumulator.tensor_signatures = tuple(
            _tensor_signature(tensor) for tensor in accumulator._tensors()
        )
        accumulator.generation_digest = _step_accumulator_digest(accumulator)
        receipt_provisional = PaperKineticDenseRequestDeltaCommitReceipt(
            source_generation_digest=source.generation_digest,
            request_generation_digest=request.generation_digest,
            artifact_generation_digest=artifact.generation_digest,
            step_generation_id=accumulator.step_generation_id,
            delta_generation_digest=delta.generation_digest,
            accumulator_generation_digest_after_commit=(
                accumulator.generation_digest
            ),
            request_commit_chain_digest_after_commit=(
                accumulator.request_commit_chain_digest
            ),
            consumed_request_count=accumulator.consumed_request_count,
            consumed_observation_count=accumulator.consumed_observation_count,
            device_completion_fence_provenance=(
                device_completion_fence_provenance
            ),
            device_completion_fence_call_count=1,
            generation_digest="",
            _accumulator_identity=id(accumulator),
            _seal=_REQUEST_DELTA_COMMIT_SEAL,
        )
        receipt = replace(
            receipt_provisional,
            generation_digest=_request_delta_commit_receipt_digest(
                receipt_provisional
            ),
        )
        _release_consumed_request_delta(delta, accumulator.generation_digest)
        accumulator.assert_current(source, session)
        receipt.assert_current(
            accumulator,
            source,
            session,
            request,
            artifact,
            delta,
        )
    except BaseException:
        _invalidate_dense_step_accumulator_without_tensor_mutation(accumulator)
        # Never destroy request-local tensors unless their device commit fence
        # completed. A failed fence leaves one retained delta on a poisoned
        # step, preventing both cross-request queue growth and unsafe release.
        if commit_fence_completed and not delta.consumed:
            try:
                _release_consumed_request_delta(delta, accumulator.generation_digest)
            except BaseException:
                pass
        raise
    return receipt


@torch.no_grad()
def authorize_paper_kinetic_dense_optimizer_step(
    accumulator: PaperKineticDenseStepGradientAccumulator,
    source: PaperKineticReplayableDenseObservationSource,
    session: PaperKineticDenseObservationReplaySession,
    replay_receipt: PaperKineticDenseObservationReplayReceipt,
) -> PaperKineticDenseOptimizerAuthorization:
    """Expose step bars only after the replay session proves the full manifest."""

    if not isinstance(replay_receipt, PaperKineticDenseObservationReplayReceipt):
        raise TypeError("optimizer authorization requires a dense replay receipt")
    try:
        replay_receipt.assert_current(session)
        accumulator.assert_current(source, session)
    except BaseException:
        _poison_dense_step_accumulator(accumulator)
        raise
    if (
        accumulator.poisoned
        or accumulator.sealed
        or accumulator.pending_delta_generation_digest
        or accumulator.fenced_request_commit_count
        != accumulator.consumed_request_count
        or not accumulator.request_commit_chain_digest
        or accumulator.consumed_request_count != replay_receipt.request_count
        or accumulator.consumed_observation_count != replay_receipt.observation_count
        or replay_receipt.observation_count != source.observation_count
    ):
        _poison_dense_step_accumulator(accumulator)
        raise ValueError("dense step cannot authorize incomplete or foreign coverage")
    accumulator.sealed = True
    accumulator.optimizer_authorized = True
    accumulator.tensor_signatures = tuple(
        _tensor_signature(tensor) for tensor in accumulator._tensors()
    )
    accumulator.generation_digest = _step_accumulator_digest(accumulator)
    provisional = PaperKineticDenseOptimizerAuthorization(
        source_generation_digest=source.generation_digest,
        compact_manifest_digest=source.compact_manifest_digest,
        step_generation_id=accumulator.step_generation_id,
        replay_receipt_generation_digest=replay_receipt.generation_digest,
        accumulator_generation_digest=accumulator.generation_digest,
        request_count=replay_receipt.request_count,
        observation_count=replay_receipt.observation_count,
        full_geometry=accumulator.full_geometry,
        optimize_camera_rays=accumulator.optimize_camera_rays,
        grad_site_rgba_f32=accumulator.grad_site_rgba_f32,
        loss_f32=accumulator.loss_f32,
        grad_positions0_f64=accumulator.grad_positions0_f64,
        grad_velocities_f64=accumulator.grad_velocities_f64,
        grad_weight_coefficients_f64=accumulator.grad_weight_coefficients_f64,
        ray_bar_keys=accumulator.ray_bar_keys,
        grad_track_ray_coefficients_f64=(
            accumulator.grad_track_ray_coefficients_f64
        ),
        tensor_signatures=tuple(
            _tensor_signature(tensor) for tensor in accumulator._tensors()
        ),
        generation_digest="",
        _accumulator_identity=id(accumulator),
        _seal=_OPTIMIZER_AUTHORIZATION_SEAL,
    )
    result = replace(
        provisional,
        generation_digest=_optimizer_authorization_digest(provisional),
    )
    accumulator.assert_current(source, session)
    result.assert_current(accumulator, replay_receipt)
    return result


@torch.no_grad()
def fail_stop_paper_kinetic_dense_step(
    accumulator: PaperKineticDenseStepGradientAccumulator,
    source: PaperKineticReplayableDenseObservationSource,
    session: PaperKineticDenseObservationReplaySession,
) -> None:
    """Irreversibly invalidate one incomplete dense step without unsafe reuse.

    A request-level failure normally performs its own fenced cleanup.  The
    whole-step coordinator still needs a public way to invalidate already
    committed request bars when a later artifact compile, request, replay
    seal, or authorization fails.  If lower-level cleanup already poisoned the
    accumulator, this function deliberately performs no additional tensor
    mutation: an abort-fence failure may have left device work in flight and
    the accumulator's quarantine must remain the sole lifetime root.
    """

    if not isinstance(accumulator, PaperKineticDenseStepGradientAccumulator):
        raise TypeError("dense step fail-stop requires its gradient accumulator")
    if not isinstance(source, PaperKineticReplayableDenseObservationSource):
        raise TypeError("dense step fail-stop requires its replay source")
    if not isinstance(session, PaperKineticDenseObservationReplaySession):
        raise TypeError("dense step fail-stop requires its replay session")
    if (
        session.source is not source
        or accumulator._source_identity != id(source)
        or accumulator._session_identity != id(session)
    ):
        raise ValueError("dense step fail-stop received a foreign replay session")
    if accumulator.poisoned:
        # Do not inspect, reduce, zero, or otherwise touch device tensors on an
        # already-poisoned step.  This includes both a sealed native-lifetime
        # quarantine and an unfenced request-delta commit; the outer
        # coordinator durably retains the accumulator (and pending delta) until
        # process restart.
        if accumulator._async_failure_quarantine is not None:
            accumulator._async_failure_quarantine.assert_current()
        if (
            accumulator.sealed
            or accumulator.optimizer_authorized
            or accumulator.generation_digest
            != _step_accumulator_digest(accumulator)
        ):
            raise ValueError("dense step poisoned fail-stop metadata changed")
        if not session.sealed:
            session.poisoned = True
        session.assert_current()
        return
    accumulator.assert_current(source, session)
    if not accumulator.poisoned:
        _poison_dense_step_accumulator(accumulator)
    if not session.sealed:
        session.poisoned = True
    accumulator.assert_current(source, session)
    session.assert_current()


def seal_paper_kinetic_dense_chunk_targets(
    source: PaperKineticReplayableDenseObservationSource,
    request: PaperKineticDenseObservationTrackRequest,
    chunk: PaperKineticDenseObservationChunk,
    target_rgb_f32: torch.Tensor,
    *,
    target_generation_id: str,
    decode_ownership: _DenseTargetDecodeOwnership,
) -> PaperKineticDenseChunkTargets:
    """Seal one adapter-produced target payload against exact chunk identity."""

    source.assert_warm_current()
    request.assert_current(source)
    chunk.assert_self_consistent(source, request)
    if not target_generation_id.strip():
        raise ValueError("target_generation_id must be nonempty")
    tensor = torch.as_tensor(target_rgb_f32)
    if not isinstance(decode_ownership, _DenseTargetDecodeOwnership):
        raise TypeError("dense chunk targets require sealed CPU decode ownership")
    decode_ownership.assert_current(source, request, chunk, tensor)
    provisional = PaperKineticDenseChunkTargets(
        source_generation_digest=source.generation_digest,
        request_generation_digest=request.generation_digest,
        chunk_generation_digest=chunk.generation_digest,
        target_generation_id=target_generation_id,
        target_rgb_f32=tensor,
        logical_tensor_bytes=tensor.numel() * tensor.element_size(),
        selected_pixel_read_mode=decode_ownership.selected_pixel_read_mode,
        selected_pixel_read_source_provenance=(
            decode_ownership.selected_pixel_read_source_provenance
        ),
        selected_pixel_read_call_count=(
            decode_ownership.selected_pixel_read_call_count
        ),
        selected_pixel_read_acceptance_capable=(
            decode_ownership.selected_pixel_read_acceptance_capable
        ),
        direct_selected_pixel_observation_count=(
            decode_ownership.direct_selected_pixel_observation_count
        ),
        bounded_region_selected_pixel_observation_count=(
            decode_ownership.bounded_region_selected_pixel_observation_count
        ),
        full_frame_fallback_observation_count=(
            decode_ownership.full_frame_fallback_observation_count
        ),
        decoded_frame_count=decode_ownership.decoded_frame_count,
        maximum_cpu_decoded_frame_tensor_bytes=(
            decode_ownership.maximum_cpu_decoded_frame_tensor_bytes
        ),
        bounded_region_materialization_count=(
            decode_ownership.bounded_region_materialization_count
        ),
        maximum_bounded_region_materialization_tensor_bytes=(
            decode_ownership.maximum_bounded_region_materialization_tensor_bytes
        ),
        source_visible_target_read_peak_logical_tensor_bytes_upper_bound=(
            decode_ownership.source_visible_target_read_peak_logical_tensor_bytes_upper_bound
        ),
        transient_mapped_address_space_bytes=(
            decode_ownership.transient_mapped_address_space_bytes
        ),
        maximum_requested_unique_mapped_page_count=(
            decode_ownership.maximum_requested_unique_mapped_page_count
        ),
        total_requested_unique_mapped_page_count=(
            decode_ownership.total_requested_unique_mapped_page_count
        ),
        mapped_page_size_bytes=decode_ownership.mapped_page_size_bytes,
        maximum_requested_mapped_page_bytes_upper_bound=(
            decode_ownership.maximum_requested_mapped_page_bytes_upper_bound
        ),
        total_requested_mapped_page_bytes_upper_bound=(
            decode_ownership.total_requested_mapped_page_bytes_upper_bound
        ),
        mapping_closed_before_return=(
            decode_ownership.mapping_closed_before_return
        ),
        cpu_chunk_target_tensor_bytes=(
            decode_ownership.cpu_chunk_target_tensor_bytes
        ),
        device_chunk_target_tensor_bytes=(
            decode_ownership.device_chunk_target_tensor_bytes
        ),
        target_decode_bridge_peak_logical_tensor_bytes=(
            decode_ownership.target_decode_bridge_peak_logical_tensor_bytes
        ),
        decoded_frame_device_type=decode_ownership.decoded_frame_device_type,
        decoded_frame_mps_completion_fence_call_count=(
            decode_ownership.decoded_frame_mps_completion_fence_call_count
        ),
        cpu_to_device_chunk_transfer_requested_non_blocking=(
            decode_ownership.cpu_to_device_chunk_transfer_requested_non_blocking
        ),
        single_bounded_chunk_transfer=decode_ownership.single_bounded_chunk_transfer,
        real_device_transfer_completion_verified=(
            decode_ownership.real_device_transfer_completion_verified
        ),
        warm_tensor_signature=_tensor_signature(tensor),
        generation_digest="",
        _cpu_transfer_source_ref=decode_ownership.cpu_transfer_source_ref,
        _cpu_transfer_source_identity=(
            decode_ownership.cpu_transfer_source_identity
        ),
        _cpu_transfer_source_signature=(
            decode_ownership.cpu_transfer_source_signature
        ),
        _seal=_TARGET_SEAL,
    )
    result = replace(provisional, generation_digest=_target_digest(provisional))
    result.assert_current(
        source,
        request,
        chunk,
        device=tensor.device,
    )
    return result


@torch.no_grad()
def decode_paper_kinetic_dense_chunk_targets(
    source: PaperKineticReplayableDenseObservationSource,
    request: PaperKineticDenseObservationTrackRequest,
    chunk: PaperKineticDenseObservationChunk,
    *,
    device: torch.device | str,
    target_generation_id: str,
    maximum_decoded_frame_scratch_tensor_bytes: int,
    maximum_chunk_target_tensor_bytes: int,
    maximum_target_decode_bridge_peak_logical_tensor_bytes: int,
    _load_lifetime: PaperKineticDenseChunkTargetLoadLifetime | None = None,
    _test_fault: PaperKineticDenseChunkTargetLoaderTestFault | None = None,
) -> PaperKineticDenseChunkTargets:
    """Gather on CPU, then transfer exactly one bounded ``[N,3]`` chunk."""

    source.assert_warm_current()
    request.assert_current(source)
    chunk.assert_self_consistent(source, request)
    if _load_lifetime is not None:
        _load_lifetime.assert_for(source, request, chunk)
    if _test_fault is not None:
        if _load_lifetime is None:
            raise ValueError("target-loader test fault requires a sealed lifetime")
        _test_fault.assert_current()
    _require_positive_int(
        maximum_decoded_frame_scratch_tensor_bytes,
        name="maximum_decoded_frame_scratch_tensor_bytes",
    )
    _require_positive_int(
        maximum_chunk_target_tensor_bytes,
        name="maximum_chunk_target_tensor_bytes",
    )
    _require_positive_int(
        maximum_target_decode_bridge_peak_logical_tensor_bytes,
        name="maximum_target_decode_bridge_peak_logical_tensor_bytes",
    )
    expected_target_bytes = chunk.observation_count * 3 * 4
    full_frame_tensor_bytes = 3 * source.provider.height * source.provider.width * 4
    fallback_frame_count, maximum_frame_observation_count = (
        _chunk_frame_decode_cardinality(chunk)
    )
    resolved_device = torch.device(device)
    if resolved_device.type != "cpu" and _load_lifetime is None:
        raise ValueError(
            "accelerator target decode requires the sealed target-loader lifetime"
        )
    if expected_target_bytes > maximum_chunk_target_tensor_bytes:
        raise MemoryError("dense chunk target budget fails before target decode")
    target_provider = source.provider.target_provider
    if not target_provider.native_selected_pixel_method_available:
        fallback_peak = _legacy_full_frame_pixel_read_peak_logical_tensor_bytes(
            chunk_observation_count=chunk.observation_count,
            maximum_frame_observation_count=maximum_frame_observation_count,
            full_frame_tensor_bytes=full_frame_tensor_bytes,
        )
        if full_frame_tensor_bytes > maximum_decoded_frame_scratch_tensor_bytes:
            raise MemoryError("dense chunk decoded-frame budget fails before fallback decode")
        if (
            _target_pixel_read_bridge_peak_logical_tensor_bytes(
                source_visible_read_peak_logical_tensor_bytes=fallback_peak,
                chunk_target_tensor_bytes=expected_target_bytes,
                target_device=resolved_device,
            )
            > maximum_target_decode_bridge_peak_logical_tensor_bytes
        ):
            raise MemoryError(
                "dense chunk fallback target-read bridge budget fails before decode"
            )

    selected_read = target_provider.select_view_frame_pixels_cpu(
        tuple(observation.view_index for observation in chunk.observations),
        tuple(observation.frame_index for observation in chunk.observations),
        tuple(observation.pixel_index for observation in chunk.observations),
        maximum_source_decode_tensor_bytes=(
            maximum_target_decode_bridge_peak_logical_tensor_bytes
        ),
    )
    selected_read.assert_valid(
        expected_observation_count=chunk.observation_count,
        full_frame_tensor_bytes=full_frame_tensor_bytes,
    )
    if (
        selected_read.maximum_full_frame_materialization_tensor_bytes
        > maximum_decoded_frame_scratch_tensor_bytes
    ):
        raise MemoryError("selected-pixel source exceeded the decoded-frame budget")
    decode_bridge_peak = _target_pixel_read_bridge_peak_logical_tensor_bytes(
        source_visible_read_peak_logical_tensor_bytes=(
            selected_read.source_visible_peak_logical_tensor_bytes_upper_bound
        ),
        chunk_target_tensor_bytes=expected_target_bytes,
        target_device=resolved_device,
    )
    if decode_bridge_peak > maximum_target_decode_bridge_peak_logical_tensor_bytes:
        raise MemoryError("selected-pixel source exceeded the target-read bridge budget")
    cpu_targets = selected_read.rgb_f32_cpu
    selected_pixel_read_mode = selected_read.selection_mode
    selected_pixel_read_source_provenance = selected_read.source_provenance
    selected_pixel_read_acceptance_capable = selected_read.acceptance_capable
    direct_selected_pixel_observation_count = (
        chunk.observation_count
        if selected_pixel_read_mode == "direct_pixels"
        else 0
    )
    bounded_region_selected_pixel_observation_count = (
        chunk.observation_count
        if selected_pixel_read_mode == "certified_bounded_region"
        else 0
    )
    full_frame_fallback_observation_count = (
        chunk.observation_count
        if selected_pixel_read_mode == "full_frame_fallback"
        else 0
    )
    decoded_frame_count = selected_read.full_frame_materialization_count
    maximum_cpu_decoded_frame_tensor_bytes = (
        selected_read.maximum_full_frame_materialization_tensor_bytes
    )
    bounded_region_materialization_count = (
        selected_read.bounded_region_materialization_count
    )
    maximum_bounded_region_materialization_tensor_bytes = (
        selected_read.maximum_bounded_region_materialization_tensor_bytes
    )
    source_visible_target_read_peak = (
        selected_read.source_visible_peak_logical_tensor_bytes_upper_bound
    )
    transient_mapped_address_space_bytes = (
        selected_read.transient_mapped_address_space_bytes
    )
    maximum_requested_unique_mapped_page_count = (
        selected_read.maximum_requested_unique_mapped_page_count
    )
    total_requested_unique_mapped_page_count = (
        selected_read.total_requested_unique_mapped_page_count
    )
    mapped_page_size_bytes = selected_read.mapped_page_size_bytes
    maximum_requested_mapped_page_bytes_upper_bound = (
        selected_read.maximum_requested_mapped_page_bytes_upper_bound
    )
    total_requested_mapped_page_bytes_upper_bound = (
        selected_read.total_requested_mapped_page_bytes_upper_bound
    )
    mapping_closed_before_return = selected_read.mapping_closed_before_return
    if (
        selected_pixel_read_mode == "full_frame_fallback"
        and decoded_frame_count != fallback_frame_count
    ):
        raise ArithmeticError("full-frame fallback changed frame coverage")
    if _load_lifetime is not None:
        _load_lifetime.retain_transfer_source(selected_read, cpu_targets)
    transferred_targets = cpu_targets.to(
        device=resolved_device,
        dtype=torch.float32,
        non_blocking=False,
    )
    if _load_lifetime is not None:
        _load_lifetime.retain_device_tensor(transferred_targets)
    targets = transferred_targets.contiguous()
    if _load_lifetime is not None and targets is not transferred_targets:
        _load_lifetime.retain_device_tensor(targets)
    if _test_fault is not None:
        raise RuntimeError(_test_fault.message)
    # ``non_blocking=False`` is only the requested transfer mode, not a proven
    # MPS completion event.  Seal the CPU source into the returned chunk lease;
    # the request retains it through all sample fences for this bounded chunk.
    del selected_read
    ownership = _DenseTargetDecodeOwnership(
        source_generation_digest=source.generation_digest,
        request_generation_digest=request.generation_digest,
        chunk_generation_digest=chunk.generation_digest,
        selected_pixel_read_mode=selected_pixel_read_mode,
        selected_pixel_read_source_provenance=(
            selected_pixel_read_source_provenance
        ),
        selected_pixel_read_call_count=1,
        selected_pixel_read_acceptance_capable=(
            selected_pixel_read_acceptance_capable
        ),
        direct_selected_pixel_observation_count=(
            direct_selected_pixel_observation_count
        ),
        bounded_region_selected_pixel_observation_count=(
            bounded_region_selected_pixel_observation_count
        ),
        full_frame_fallback_observation_count=(
            full_frame_fallback_observation_count
        ),
        decoded_frame_count=decoded_frame_count,
        maximum_cpu_decoded_frame_tensor_bytes=(
            maximum_cpu_decoded_frame_tensor_bytes
        ),
        bounded_region_materialization_count=(
            bounded_region_materialization_count
        ),
        maximum_bounded_region_materialization_tensor_bytes=(
            maximum_bounded_region_materialization_tensor_bytes
        ),
        source_visible_target_read_peak_logical_tensor_bytes_upper_bound=(
            source_visible_target_read_peak
        ),
        transient_mapped_address_space_bytes=(
            transient_mapped_address_space_bytes
        ),
        maximum_requested_unique_mapped_page_count=(
            maximum_requested_unique_mapped_page_count
        ),
        total_requested_unique_mapped_page_count=(
            total_requested_unique_mapped_page_count
        ),
        mapped_page_size_bytes=mapped_page_size_bytes,
        maximum_requested_mapped_page_bytes_upper_bound=(
            maximum_requested_mapped_page_bytes_upper_bound
        ),
        total_requested_mapped_page_bytes_upper_bound=(
            total_requested_mapped_page_bytes_upper_bound
        ),
        mapping_closed_before_return=mapping_closed_before_return,
        cpu_chunk_target_tensor_bytes=expected_target_bytes,
        device_chunk_target_tensor_bytes=expected_target_bytes,
        target_decode_bridge_peak_logical_tensor_bytes=decode_bridge_peak,
        target_tensor_identity=id(targets),
        target_tensor_signature=_tensor_signature(targets),
        cpu_transfer_source_identity=id(cpu_targets),
        cpu_transfer_source_signature=_tensor_signature(cpu_targets),
        cpu_transfer_source_ref=cpu_targets,
        _seal=_TARGET_DECODE_OWNERSHIP_SEAL,
    )
    result = seal_paper_kinetic_dense_chunk_targets(
        source,
        request,
        chunk,
        targets,
        target_generation_id=target_generation_id,
        decode_ownership=ownership,
    )
    if result.logical_tensor_bytes > maximum_chunk_target_tensor_bytes:
        raise ArithmeticError("dense chunk target tensor exceeded its preflight")
    if (
        result.target_decode_bridge_peak_logical_tensor_bytes
        > maximum_target_decode_bridge_peak_logical_tensor_bytes
    ):
        raise ArithmeticError("dense chunk target-decode bridge exceeded its preflight")
    return result


def prepare_paper_kinetic_dense_chunk_target_loader(
    source: PaperKineticReplayableDenseObservationSource,
    request: PaperKineticDenseObservationTrackRequest,
    *,
    device: torch.device | str,
    target_generation_id: str,
    maximum_decoded_frame_scratch_tensor_bytes: int,
    maximum_chunk_target_tensor_bytes: int,
    maximum_target_decode_bridge_peak_logical_tensor_bytes: int,
    source_test_fault: PaperKineticDenseChunkTargetLoaderTestFault | None = None,
) -> PaperKineticDenseChunkTargetLoader:
    """Prepare the only target-loader capability accepted by dense replay."""

    source.assert_warm_current()
    request.assert_current(source)
    resolved_device = torch.device(device)
    if not target_generation_id.strip():
        raise ValueError("target_generation_id must be nonempty")
    for name, value in (
        (
            "maximum_decoded_frame_scratch_tensor_bytes",
            maximum_decoded_frame_scratch_tensor_bytes,
        ),
        ("maximum_chunk_target_tensor_bytes", maximum_chunk_target_tensor_bytes),
        (
            "maximum_target_decode_bridge_peak_logical_tensor_bytes",
            maximum_target_decode_bridge_peak_logical_tensor_bytes,
        ),
    ):
        _require_positive_int(value, name=name)
    if source_test_fault is not None:
        source_test_fault.assert_current()
    provisional = PaperKineticDenseChunkTargetLoader(
        source_generation_digest=source.generation_digest,
        request_generation_digest=request.generation_digest,
        target_generation_id=target_generation_id,
        device=resolved_device,
        maximum_decoded_frame_scratch_tensor_bytes=(
            maximum_decoded_frame_scratch_tensor_bytes
        ),
        maximum_chunk_target_tensor_bytes=maximum_chunk_target_tensor_bytes,
        maximum_target_decode_bridge_peak_logical_tensor_bytes=(
            maximum_target_decode_bridge_peak_logical_tensor_bytes
        ),
        generation_digest="",
        _source_ref=source,
        _request_ref=request,
        _source_identity=id(source),
        _request_identity=id(request),
        _test_fault=source_test_fault,
        _seal=_TARGET_LOADER_SEAL,
    )
    provisional.generation_digest = _target_loader_digest(provisional)
    provisional.assert_current(source, request, device=resolved_device)
    return provisional


def prepare_paper_kinetic_dense_chunk_target_loader_test_fault(
    *,
    message: str,
    fail_on_load_number: int = 1,
) -> PaperKineticDenseChunkTargetLoaderTestFault:
    """Create the single exact post-transfer fault supported by source tests."""

    fault = PaperKineticDenseChunkTargetLoaderTestFault(
        stage="after_transfer_before_target_seal",
        message=message,
        fail_on_load_number=fail_on_load_number,
        _seal=_TARGET_LOADER_TEST_FAULT_SEAL,
    )
    fault.assert_current()
    return fault


def _prepare_dense_cached_native_lane(
    artifact: PaperKineticCompiledCpuArtifact,
    provider: PaperKineticLazyProgramBundleProvider,
    request: PaperKineticDenseObservationTrackRequest,
    accumulator: PaperKineticDenseStepGradientAccumulator,
    native_ops: Any,
    *,
    device: torch.device | str,
    backend_provenance: str,
    maximum_resident_logical_tensor_bytes: int,
    device_completion_fence: Callable[[], None],
    device_completion_fence_provenance: str,
) -> _DenseCachedNativeLane:
    """Build one frame-free lane directly from a bounded cached artifact."""

    if not isinstance(artifact, PaperKineticCompiledCpuArtifact):
        raise TypeError("dense cached lane requires a compiled CPU artifact")
    if not isinstance(provider, PaperKineticLazyProgramBundleProvider):
        raise TypeError("dense cached lane requires a lazy program provider")
    if not backend_provenance.strip():
        raise ValueError("backend_provenance must be nonempty")
    _require_positive_int(
        maximum_resident_logical_tensor_bytes,
        name="maximum_resident_logical_tensor_bytes",
    )
    artifact.assert_warm_reusable_with_provider(provider)
    if request.view_index != artifact.key.view_index or request.track_ids != artifact.track_ids:
        raise ValueError("dense request and cached artifact describe different tracks")
    resolved_device = torch.device(device)
    preflight = _lane_resident_upper_bound_bytes(artifact)
    if preflight > maximum_resident_logical_tensor_bytes:
        raise MemoryError("dense cached lane exceeds its budget before device materialization")

    sampler = artifact.sampler
    construction_lifetime = _DenseCachedNativeLaneConstructionLifetime(
        artifact=artifact,
        provider=provider,
        request=request,
        native_ops=native_ops,
        device=resolved_device,
        backend_provenance=backend_provenance,
        resident_logical_tensor_bytes_upper_bound=preflight,
        expected_runtime_block_digests=tuple(
            block.generation_digest
            for bucket in sampler.lowering.buckets
            for block in bucket.blocks
        ),
        _artifact_identity=id(artifact),
        _provider_identity=id(provider),
        _request_identity=id(request),
        _native_ops_identity=id(native_ops),
        _seal=_DENSE_LANE_CONSTRUCTION_LIFETIME_SEAL,
    )
    construction_lifetime.assert_retained()
    try:
        construction_lifetime.phase = "materializing"
        spatial_construction_lifetime = (
            prepare_paper_kinetic_union_local_spatial_bundle_construction_lifetime(
                sampler,
                track_ids=request.track_ids,
                device=resolved_device,
            )
        )
        construction_lifetime.spatial_construction_lifetime = (
            spatial_construction_lifetime
        )
        construction_lifetime.assert_retained()
        spatial_bundle = materialize_paper_kinetic_union_local_spatial_bundle(
            spatial_construction_lifetime
        )
        construction_lifetime.spatial_bundle = spatial_bundle
        construction_lifetime.assert_retained()
        construction_lifetime.payloads = tuple(
            iter_materialize_kinetic_native_equal_rank_blocks(
                sampler.lowering,
                sampler.sources,
            )
        )
        construction_lifetime.assert_retained()
        for payload in construction_lifetime.payloads:
            construction_lifetime.current_payload = payload
            runtime_lifetime = (
                prepare_kinetic_native_equal_rank_runtime_construction_lifetime(
                    payload,
                    lowering=sampler.lowering,
                    sources=sampler.sources,
                    native_ops=native_ops,
                    device=resolved_device,
                )
            )
            construction_lifetime.current_runtime_lifetime = runtime_lifetime
            construction_lifetime.runtime_lifetimes.append(runtime_lifetime)
            construction_lifetime.assert_retained()
            runtime = prepare_kinetic_native_equal_rank_runtime_block(
                payload,
                lowering=sampler.lowering,
                sources=sampler.sources,
                native_ops=native_ops,
                device=resolved_device,
                construction_lifetime=runtime_lifetime,
            )
            construction_lifetime.runtimes.append(runtime)
            construction_lifetime.current_runtime_lifetime = None
            construction_lifetime.current_payload = None
            construction_lifetime.assert_retained()
        runtimes = tuple(construction_lifetime.runtimes)
        executor = prepare_kinetic_native_material_step_executor(
            native_ops,
            tuple((runtime, sampler) for runtime in runtimes),
            backend_provenance=backend_provenance,
        )
        construction_lifetime.executor = executor
        construction_lifetime.phase = "transferred"
        construction_lifetime.assert_retained()
        for runtime in runtimes:
            runtime.assert_warm_layout()
        executor.assert_current()
        spatial_bundle.assert_warm_layout()
        generation_digest = _digest_parts(
            LANE_PROVENANCE,
            artifact.generation_digest,
            request.generation_digest,
            spatial_bundle.generation_digest,
            tuple(runtime.generation_id for runtime in runtimes),
            executor.generation_id,
            resolved_device,
            backend_provenance,
            preflight,
        )
        return _DenseCachedNativeLane(
            artifact=artifact,
            spatial_bundle=spatial_bundle,
            runtimes=runtimes,
            executor=executor,
            construction_lifetime=construction_lifetime,
            resident_logical_tensor_bytes_upper_bound=preflight,
            generation_digest=generation_digest,
        )
    except BaseException as error:
        original_traceback = error.__traceback__
        try:
            returned = device_completion_fence()
            if returned is not None:
                raise TypeError(
                    "partial dense-lane completion fence must return None"
                )
        except BaseException as fence_error:
            _quarantine_dense_async_failure(
                accumulator,
                stage="partial-lane-construction",
                original_error=error,
                original_traceback=original_traceback,
                cleanup_fence_error=fence_error,
                retained_references=(
                    (
                        "dense_lane_construction_lifetime",
                        construction_lifetime,
                    ),
                    (
                        "spatial_bundle_construction_lifetime",
                        construction_lifetime.spatial_construction_lifetime,
                    ),
                    ("spatial_bundle", construction_lifetime.spatial_bundle),
                    ("payloads", construction_lifetime.payloads),
                    (
                        "runtime_construction_lifetimes",
                        tuple(construction_lifetime.runtime_lifetimes),
                    ),
                    ("runtimes", tuple(construction_lifetime.runtimes)),
                    ("executor", construction_lifetime.executor),
                    ("native_ops", native_ops),
                ),
                device_completion_fence_provenance=(
                    device_completion_fence_provenance
                ),
            )
            error.add_note(
                "partial dense-lane completion fence failed; constructed "
                "references are quarantined and the process must restart"
            )
            raise error.with_traceback(original_traceback) from fence_error
        raise


def iter_paper_kinetic_dense_chunk_sample_blocks(
    sampler: PaperKineticRowRaggedSampler,
    provider: PaperKineticLazyProgramBundleProvider,
    source: PaperKineticReplayableDenseObservationSource,
    request: PaperKineticDenseObservationTrackRequest,
    chunk: PaperKineticDenseObservationChunk,
    targets: PaperKineticDenseChunkTargets,
    *,
    loss_normalization_id: str,
    maximum_samples_per_launch: int,
    maximum_sample_materialization_logical_tensor_bytes: int,
) -> Iterator[_DenseSampleMaterializationLease]:
    """Dispatch one non-rectangular replay chunk into bounded native blocks."""

    sampler.assert_warm_layout()
    provider.assert_warm_current()
    targets.assert_current(source, request, chunk, device=targets.target_rgb_f32.device)
    if sampler.view_index != request.view_index or sampler.track_ids != request.track_ids:
        raise ValueError("dense chunk sampler and track request differ")
    if not loss_normalization_id.strip():
        raise ValueError("loss_normalization_id must be nonempty")
    _require_positive_int(maximum_samples_per_launch, name="maximum_samples_per_launch")
    _require_positive_int(
        maximum_sample_materialization_logical_tensor_bytes,
        name="maximum_sample_materialization_logical_tensor_bytes",
    )

    rows = {(row.track_id, row.chart_index): row for row in sampler.rows}
    first_row = {
        track_id: next(row for row in sampler.rows if row.track_id == track_id) for track_id in request.track_ids
    }
    grouped: dict[str, list[tuple[int, PaperKineticObservation, PaperKineticRowBinding, float]]] = {}
    for position, observation in enumerate(chunk.observations):
        binding = first_row[observation.pixel_index]
        time = provider.frame_times[observation.frame_index]
        chart_index = dispatch_prevalidated_kinetic_chart_index(
            binding.program,
            Fraction.from_float(time),
            expected_generation_digest=binding.program_generation_digest,
        )
        row = rows[(observation.pixel_index, chart_index)]
        _validate_row_time(row, time)
        grouped.setdefault(row.native_block_generation_digest, []).append((position, observation, row, time))

    if sum(len(values) for values in grouped.values()) != chunk.observation_count:
        raise ArithmeticError("dense chunk dispatch did not cover every observation")
    for native_block in (block for bucket in sampler.lowering.buckets for block in bucket.blocks):
        entries = grouped.get(native_block.generation_digest, ())
        for start in range(0, len(entries), maximum_samples_per_launch):
            selected = tuple(entries[start : start + maximum_samples_per_launch])
            if selected:
                yield _materialize_dense_sample_block(
                    sampler,
                    native_block_generation_digest=native_block.generation_digest,
                    entries=selected,
                    target_rgb_f32=targets.target_rgb_f32,
                    global_loss_element_count=source.observation_count * 3,
                    loss_normalization_id=loss_normalization_id,
                    maximum_materialization_logical_tensor_bytes=(
                        maximum_sample_materialization_logical_tensor_bytes
                    ),
                )


def _poison_step_on_new_replay_progress(function: Callable[..., Any]):
    """Fail closed if result sealing breaks after this call advances replay."""

    @wraps(function)
    def guarded(
        source: PaperKineticReplayableDenseObservationSource,
        session: PaperKineticDenseObservationReplaySession,
        request: PaperKineticDenseObservationTrackRequest,
        artifact: PaperKineticCompiledCpuArtifact,
        accumulator: PaperKineticDenseStepGradientAccumulator,
        *args: Any,
        **kwargs: Any,
    ):
        request_count_before = getattr(session, "request_count", None)
        emitted_before = getattr(session, "emitted_observation_count", None)
        try:
            return function(
                source,
                session,
                request,
                artifact,
                accumulator,
                *args,
                **kwargs,
            )
        except BaseException:
            if (
                isinstance(accumulator, PaperKineticDenseStepGradientAccumulator)
                and accumulator._session_identity == id(session)
                and accumulator._source_identity == id(source)
                and isinstance(request_count_before, int)
                and isinstance(emitted_before, int)
                and (
                    session.request_count > request_count_before
                    or session.emitted_observation_count > emitted_before
                )
            ):
                session.poisoned = True
                # A failed cleanup fence leaves native work potentially in
                # flight.  ``_quarantine_dense_async_failure`` already sealed
                # the accumulator as poisoned without touching its tensors and
                # retained every unsafe lifetime root.  Enqueueing ``zero_``
                # here would violate that quarantine after the very fence that
                # should make mutation safe has failed.
                if accumulator._async_failure_quarantine is None:
                    _poison_dense_step_accumulator(accumulator)
            raise

    return guarded


@torch.no_grad()
@_poison_step_on_new_replay_progress
def run_paper_kinetic_dense_cached_native_request(
    source: PaperKineticReplayableDenseObservationSource,
    session: PaperKineticDenseObservationReplaySession,
    request: PaperKineticDenseObservationTrackRequest,
    artifact: PaperKineticCompiledCpuArtifact,
    accumulator: PaperKineticDenseStepGradientAccumulator,
    *,
    step_generation_id: str,
    loss_normalization_id: str,
    material_generation_id: str,
    background_generation_id: str,
    global_site_rgba_f32: torch.Tensor,
    background_rgb_f32: torch.Tensor,
    native_ops: Any,
    backend_provenance: str,
    maximum_samples_per_launch: int,
    memory_policy: PaperKineticDenseCachedNativeMemoryPolicy,
    load_chunk_targets: PaperKineticDenseChunkTargetLoader,
    device_completion_fence: Callable[[], None],
    device_completion_fence_provenance: str,
    full_geometry_reverse_mode: str = STAGED_SPARSE_FULL_GEOMETRY_REVERSE,
    cone_tolerance: float = 1.0e-5,
) -> PaperKineticDenseCachedNativeRequestResult:
    """Replay one request and return one uncommitted combined gradient delta."""

    _validate_request_inputs(
        source,
        session,
        request,
        artifact,
        accumulator,
        step_generation_id=step_generation_id,
        loss_normalization_id=loss_normalization_id,
        material_generation_id=material_generation_id,
        background_generation_id=background_generation_id,
        global_site_rgba_f32=global_site_rgba_f32,
        background_rgb_f32=background_rgb_f32,
        backend_provenance=backend_provenance,
        maximum_samples_per_launch=maximum_samples_per_launch,
        memory_policy=memory_policy,
        load_chunk_targets=load_chunk_targets,
        device_completion_fence=device_completion_fence,
        device_completion_fence_provenance=device_completion_fence_provenance,
        full_geometry_reverse_mode=full_geometry_reverse_mode,
        cone_tolerance=cone_tolerance,
    )
    provider = source.provider
    device = accumulator.grad_site_rgba_f32.device
    if device.type == "mps" and (
        device_completion_fence is not synchronize_mps_device_completion_fence
        or device_completion_fence_provenance
        != MPS_DEVICE_COMPLETION_FENCE_PROVENANCE
    ):
        raise ValueError(
            "MPS dense replay requires the canonical torch.mps.synchronize fence"
        )
    expected_observation_count = _request_observation_count(source, request)
    full_geometry = accumulator.full_geometry
    optimize_camera_rays = accumulator.optimize_camera_rays
    staged_full_geometry = (
        full_geometry
        and full_geometry_reverse_mode == STAGED_SPARSE_FULL_GEOMETRY_REVERSE
    )
    fused_full_geometry = (
        full_geometry
        and full_geometry_reverse_mode == FUSED_DIRECT_V1_FULL_GEOMETRY_REVERSE
    )
    lane_resident_upper_bound_preflight = _lane_resident_upper_bound_bytes(
        artifact
    )
    if (
        lane_resident_upper_bound_preflight
        > memory_policy.maximum_lane_resident_logical_tensor_bytes
    ):
        _poison_dense_step_accumulator(accumulator)
        raise MemoryError(
            "dense cached lane budget fails before native lane build"
        )
    request_geometry_bar_bytes = (
        _request_geometry_bar_bytes(
            artifact.sampler,
            request,
            include_ray_gradients=optimize_camera_rays,
        )
        if full_geometry
        else 0
    )
    if (
        request_geometry_bar_bytes
        > memory_policy.maximum_request_geometry_bar_tensor_bytes
    ):
        _poison_dense_step_accumulator(accumulator)
        raise MemoryError(
            "dense cached request geometry-bar budget fails before native lane build"
        )
    fused_prepared_owned_upper_bound = 0
    fused_compact_output_upper_bound = 0
    fused_global_output_upper_bound = 0
    fused_output_upper_bound = 0
    fused_geometry_bridge_upper_bound = 0
    staged_sparse_geometry_bridge_upper_bound = 0
    if staged_full_geometry:
        staged_sparse_geometry_bridge_upper_bound = (
            _staged_sparse_geometry_bridge_visible_peak_logical_tensor_bytes_upper_bound(
                artifact,
                include_ray_gradients=optimize_camera_rays,
            )
        )
        if (
            staged_sparse_geometry_bridge_upper_bound
            > memory_policy.maximum_geometry_bridge_visible_peak_logical_tensor_bytes
        ):
            _poison_dense_step_accumulator(accumulator)
            raise MemoryError(
                "dense cached staged geometry bridge budget fails before native lane build"
            )
    if fused_full_geometry:
        fused_prepared_owned_upper_bound = (
            _fused_prepared_owned_logical_tensor_bytes_upper_bound(artifact)
        )
        (
            fused_compact_output_upper_bound,
            fused_global_output_upper_bound,
            fused_output_upper_bound,
        ) = _fused_output_scratch_logical_tensor_bytes_upper_bound(artifact)
        fused_geometry_bridge_upper_bound = (
            _fused_geometry_bridge_visible_peak_logical_tensor_bytes(
                artifact.sampler
            )
        )
        if (
            fused_geometry_bridge_upper_bound
            != fused_global_output_upper_bound + request_geometry_bar_bytes
        ):
            _poison_dense_step_accumulator(accumulator)
            raise ArithmeticError(
                "dense fused bridge and active-peak byte proofs disagree"
            )
        for name, required, allowed in (
            (
                "fused prepared payload",
                fused_prepared_owned_upper_bound,
                memory_policy.maximum_fused_prepared_owned_logical_tensor_bytes,
            ),
            (
                "fused output scratch",
                fused_output_upper_bound,
                memory_policy.maximum_fused_output_scratch_logical_tensor_bytes,
            ),
            (
                "fused geometry bridge",
                fused_geometry_bridge_upper_bound,
                memory_policy.maximum_fused_geometry_bridge_visible_peak_logical_tensor_bytes,
            ),
        ):
            if allowed < 1 or required > allowed:
                _poison_dense_step_accumulator(accumulator)
                raise MemoryError(
                    f"dense cached {name} budget fails before native lane build"
                )
    active_upper_bound = _active_state_upper_bound_bytes(
        artifact,
        include_full_geometry=False,
    ) + request_geometry_bar_bytes + accumulator.logical_tensor_bytes
    if staged_full_geometry:
        # The sparse reducer preflight already includes the native [J,W]
        # cotangent, its row bridge, compact geometry outputs, and node scratch.
        # Compose that complete phase with the still-live request state rather
        # than adding the [J,W] tensor a second time.
        active_upper_bound += staged_sparse_geometry_bridge_upper_bound
    if fused_full_geometry:
        active_upper_bound += (
            fused_prepared_owned_upper_bound + fused_output_upper_bound
        )
    if active_upper_bound > memory_policy.maximum_active_node_and_union_bar_tensor_bytes:
        _poison_dense_step_accumulator(accumulator)
        raise MemoryError("dense cached active-state budget fails before native lane build")
    reverse_lane_plus_active_upper_bound = (
        lane_resident_upper_bound_preflight + active_upper_bound
    )
    max_node_count = max(
        block.node_count
        for bucket in artifact.sampler.lowering.buckets
        for block in bucket.blocks
    )
    maximum_materialized_samples = _maximum_materialized_samples_for_budget(
        node_count=max_node_count,
        maximum_logical_tensor_bytes=(
            memory_policy.maximum_sample_materialization_logical_tensor_bytes
        ),
    )
    if maximum_materialized_samples < 1:
        _poison_dense_step_accumulator(accumulator)
        raise MemoryError(
            "dense sample materialization budget cannot fit one row before native lane build"
        )
    effective_maximum_samples_per_launch = min(
        maximum_samples_per_launch,
        maximum_materialized_samples,
    )

    lane: _DenseCachedNativeLane | None = None
    native_session = None
    active: dict[str, _ActiveBlock] = {}
    local_loss = torch.zeros((1,), dtype=torch.float32, device=device)
    local_union_bar: torch.Tensor | None = None
    local_positions0_f64: torch.Tensor | None = None
    local_velocities_f64: torch.Tensor | None = None
    local_weight_coefficients_f64: torch.Tensor | None = None
    request_ray_bar_keys: tuple[tuple[int, int], ...] = ()
    local_track_ray_coefficients_f64: torch.Tensor | None = None
    cone = torch.zeros((3,), dtype=torch.int32, device=device)
    request_count_before = session.request_count
    emitted_before = session.emitted_observation_count
    chunk_count = 0
    streamed_observation_count = 0
    sample_launch_count = 0
    sample_node_interaction_count = 0
    transferred_target_payload_bytes = 0
    peak_chunk_target_bytes = 0
    selected_pixel_read_call_count = 0
    direct_selected_pixel_observation_count = 0
    bounded_region_selected_pixel_observation_count = 0
    full_frame_fallback_observation_count = 0
    bounded_region_materialization_count = 0
    decoded_frame_count = 0
    decoded_frame_mps_completion_fence_call_count = 0
    peak_cpu_decoded_frame_bytes = 0
    peak_bounded_region_materialization_bytes = 0
    peak_source_visible_target_read_bytes = 0
    mapped_selected_pixel_read_call_count = 0
    mapping_closed_before_return_count = 0
    peak_transient_mapped_address_space_bytes = 0
    peak_requested_unique_mapped_page_count = 0
    peak_mapped_page_size_bytes = 0
    peak_requested_mapped_page_bytes_upper_bound = 0
    cumulative_requested_mapped_page_count = 0
    cumulative_requested_mapped_page_bytes_upper_bound = 0
    peak_cpu_chunk_target_bytes = 0
    peak_device_chunk_target_bytes = 0
    peak_target_decode_bridge_bytes = 0
    peak_sample_launch_bytes = 0
    peak_sample_launch_node_count = 0
    peak_sample_materialization_bytes = 0
    peak_interpolation_evaluator_scratch_bytes = 0
    maximum_interpolation_rows_per_subchunk = 0
    peak_native_prepared_sample_scratch_bytes = 0
    peak_public_sample_launch_logical_bytes = 0
    peak_chunk_dispatch_identity_bytes = 0
    selected_pixel_read_modes: set[str] = set()
    selected_pixel_read_source_provenances: set[str] = set()
    chunk_manifest = hashlib.sha256()
    chunk_manifest.update(REQUEST_PROVENANCE.encode("utf-8"))
    body_error: BaseException | None = None
    fence_call_count = 0
    geometry_reduction_fence_call_count = 0
    geometry_completion_receipt_count = 0
    sample_backpressure_fence_call_count = 0
    active_block_commit_fence_call_count = 0
    compact_gather_lifetime_install_count = 0
    compact_gather_lifetime_retire_count = 0
    forward_into_lifetime_install_count = 0
    forward_into_lifetime_retire_count = 0
    geometry_row_vjp_call_count = 0
    geometry_dense_global_site_accumulation_elements = 0
    geometry_all_site_owner_validation_evaluations = 0
    geometry_compact_to_global_scatter_elements = 0
    maximum_simultaneous_geometry_jw_length_bar_tensors = 0
    maximum_native_length_bar_tensor_bytes = 0
    maximum_geometry_bridge_visible_tensor_bytes = 0
    maximum_active_block_commit_scratch_tensor_bytes = 0
    fused_prepared_blocks: tuple[Any, ...] = ()
    fused_execution_receipt: Any = None
    fused_transaction_result: Any = None
    fused_prepared_owned_logical_tensor_bytes = 0
    fused_output_scratch_logical_tensor_bytes = 0
    fused_compact_output_scratch_logical_tensor_bytes = 0
    fused_global_output_scratch_logical_tensor_bytes = 0
    fused_geometry_bridge_visible_tensor_bytes = 0
    fused_transaction_fence_call_count = 0
    fused_post_accept_commit_fence_call_count = 0
    fused_compact_material_scatter_elements = 0
    fused_active_manifest_certified = False
    telemetry: KineticNativeMaterialStepTelemetry | None = None
    lane_resident_bytes_upper_bound = 0
    lane_generation_digest = ""
    commit_source_site_ids_i64: torch.Tensor | None = None
    lifetime_release_fence_completed = False
    current_chunk_targets: PaperKineticDenseChunkTargets | None = None
    current_sample_materialization: _DenseSampleMaterializationLease | None = None
    current_sample_block: PaperKineticRowRaggedSampleBlock | None = None
    current_sample_lifetime: KineticNativeSampleLaunchLifetime | None = None
    current_forward_runtime: KineticNativeEqualRankRuntimeBlock | None = None
    current_compact_gather_lifetime: (
        _DenseCompactMaterialGatherLifetime | None
    ) = None
    current_forward_compact_material: torch.Tensor | None = None
    current_forward_into_lifetime: (
        KineticNativeNodeForwardIntoLifetime | None
    ) = None
    current_forward_node_chart_out: torch.Tensor | None = None
    current_forward_token: KineticNativeMaterialStepWorldToken | None = None
    current_forward_grad_node_chart: torch.Tensor | None = None
    current_forward_loss: torch.Tensor | None = None
    current_reverse_block_state: _ActiveBlock | None = None
    current_reverse_compact_bar: torch.Tensor | None = None
    current_material_execution: Any = None
    current_full_geometry_execution: Any = None
    current_native_vjp: Any = None
    current_geometry_reduction: Any = None
    current_geometry_completion: Any = None

    def invoke_device_completion_fence() -> None:
        nonlocal fence_call_count
        returned = device_completion_fence()
        fence_call_count += 1
        if returned is not None:
            _fail_type("device completion fence must return None")

    def retire_block_forward_lifetimes_after_proven_completion(
        block_state: _ActiveBlock,
    ) -> None:
        nonlocal compact_gather_lifetime_retire_count
        nonlocal forward_into_lifetime_retire_count
        if block_state.forward_into_lifetime.phase != "released":
            block_state.forward_into_lifetime.retire_after_completion_fence()
            forward_into_lifetime_retire_count += 1
        if block_state.compact_gather_lifetime.phase != "released":
            block_state.compact_gather_lifetime.retire_after_completion_fence()
            compact_gather_lifetime_retire_count += 1

    def retire_all_forward_lifetimes_after_proven_completion() -> None:
        nonlocal compact_gather_lifetime_retire_count
        nonlocal forward_into_lifetime_retire_count
        forward_lifetimes = [
            block_state.forward_into_lifetime for block_state in active.values()
        ]
        gather_lifetimes = [
            block_state.compact_gather_lifetime for block_state in active.values()
        ]
        if isinstance(
            current_forward_into_lifetime,
            KineticNativeNodeForwardIntoLifetime,
        ):
            forward_lifetimes.append(current_forward_into_lifetime)
        if isinstance(
            current_compact_gather_lifetime,
            _DenseCompactMaterialGatherLifetime,
        ):
            gather_lifetimes.append(current_compact_gather_lifetime)
        seen: set[int] = set()
        for lifetime in forward_lifetimes:
            if id(lifetime) in seen:
                continue
            seen.add(id(lifetime))
            if lifetime.phase != "released":
                lifetime.retire_after_completion_fence()
                forward_into_lifetime_retire_count += 1
        seen.clear()
        for lifetime in gather_lifetimes:
            if id(lifetime) in seen:
                continue
            seen.add(id(lifetime))
            if lifetime.phase != "released":
                lifetime.retire_after_completion_fence()
                compact_gather_lifetime_retire_count += 1

    def request_lifetime_references() -> tuple[tuple[str, Any], ...]:
        return (
            ("lane", lane),
            ("native_session", native_session),
            ("active_blocks", active),
            ("local_loss", local_loss),
            ("local_union_bar", local_union_bar),
            ("local_positions0", local_positions0_f64),
            ("local_velocities", local_velocities_f64),
            ("local_weight_coefficients", local_weight_coefficients_f64),
            ("local_track_ray_coefficients", local_track_ray_coefficients_f64),
            ("cone", cone),
            ("commit_source_site_ids", commit_source_site_ids_i64),
            ("target_loader", load_chunk_targets),
            (
                "target_loader_active_lifetime",
                load_chunk_targets._active_lifetime,
            ),
            ("current_chunk_targets", current_chunk_targets),
            (
                "current_chunk_cpu_transfer_source",
                None
                if not isinstance(
                    current_chunk_targets,
                    PaperKineticDenseChunkTargets,
                )
                else current_chunk_targets._cpu_transfer_source_ref,
            ),
            ("current_sample_materialization", current_sample_materialization),
            ("current_sample_block", current_sample_block),
            ("current_sample_lifetime", current_sample_lifetime),
            ("current_forward_runtime", current_forward_runtime),
            (
                "current_compact_gather_lifetime",
                current_compact_gather_lifetime,
            ),
            (
                "current_forward_compact_material",
                current_forward_compact_material,
            ),
            ("current_forward_into_lifetime", current_forward_into_lifetime),
            (
                "current_forward_node_chart_out",
                current_forward_node_chart_out,
            ),
            ("current_forward_token", current_forward_token),
            (
                "current_forward_grad_node_chart",
                current_forward_grad_node_chart,
            ),
            ("current_forward_loss", current_forward_loss),
            (
                "session_outstanding_sample_lifetime",
                None
                if native_session is None
                else native_session._outstanding_sample_lifetime,
            ),
            ("current_reverse_block_state", current_reverse_block_state),
            ("current_reverse_compact_bar", current_reverse_compact_bar),
            ("current_material_execution", current_material_execution),
            ("current_full_geometry_execution", current_full_geometry_execution),
            ("current_native_vjp", current_native_vjp),
            ("current_geometry_reduction", current_geometry_reduction),
            ("current_geometry_completion", current_geometry_completion),
            ("fused_prepared_blocks", fused_prepared_blocks),
            ("fused_execution_receipt", fused_execution_receipt),
            ("fused_transaction_result", fused_transaction_result),
        )
    try:
        lane = _prepare_dense_cached_native_lane(
            artifact,
            provider,
            request,
            accumulator,
            native_ops,
            device=device,
            backend_provenance=backend_provenance,
            maximum_resident_logical_tensor_bytes=(memory_policy.maximum_lane_resident_logical_tensor_bytes),
            device_completion_fence=invoke_device_completion_fence,
            device_completion_fence_provenance=(
                device_completion_fence_provenance
            ),
        )
        lane_generation_digest = lane.generation_digest
        lane_resident_bytes_upper_bound = lane.resident_logical_tensor_bytes_upper_bound
        if lane_resident_bytes_upper_bound != lane_resident_upper_bound_preflight:
            _fail_arithmetic("dense cached lane changed its admitted byte proof")
        commit_source_site_ids_i64 = lane.spatial_bundle.source_site_ids_i64
        local_union_bar = torch.zeros(
            (lane.spatial_bundle.union_site_count, 4),
            dtype=torch.float32,
            device=device,
        )
        if staged_full_geometry:
            sites = _shared_kinetic_sites(artifact.sampler)
            request_ray_bar_keys = (
                tuple(
                    (request.view_index, track_id)
                    for track_id in request.track_ids
                )
                if optimize_camera_rays
                else ()
            )
            local_positions0_f64 = torch.zeros_like(
                sites.positions0,
                device="cpu",
                dtype=torch.float64,
            )
            local_velocities_f64 = torch.zeros_like(
                sites.velocities,
                device="cpu",
                dtype=torch.float64,
            )
            local_weight_coefficients_f64 = torch.zeros_like(
                sites.weight_coefficients,
                device="cpu",
                dtype=torch.float64,
            )
            local_track_ray_coefficients_f64 = (
                torch.zeros(
                    (len(request_ray_bar_keys), 12),
                    device="cpu",
                    dtype=torch.float64,
                )
                if optimize_camera_rays
                else None
            )
        elif fused_full_geometry:
            # The accepted native result is the sole source of these public
            # CPU-f64 request bars. Avoid allocating an unused zero clone.
            request_ray_bar_keys = ()
        native_session = lane.executor.begin_step(
            step_generation_id=(f"{step_generation_id}:dense-request:{request.generation_digest}"),
            requested_observation_count=expected_observation_count,
        )
        replay_iterator = session.iter_request_chunks(request)
        with closing(replay_iterator) as chunks:
            for chunk in chunks:
                chunk.assert_self_consistent(source, request)
                target_bytes = chunk.observation_count * 3 * 4
                minimum_target_bridge_bytes = target_bytes * (
                    2 if device.type != "cpu" else 1
                )
                maximum_launch_sample_count = min(
                    chunk.observation_count,
                    effective_maximum_samples_per_launch,
                )
                materialization_upper_bound = _sample_materialization_memory_plan(
                    sample_count=maximum_launch_sample_count,
                    node_count=max_node_count,
                    maximum_logical_tensor_bytes=(
                        memory_policy.maximum_sample_materialization_logical_tensor_bytes
                    ),
                )
                native_prepared_sample_scratch_upper_bound = (
                    4 * maximum_launch_sample_count + 20
                )
                transfer_resident_target_bytes = target_bytes * (
                    2 if device.type != "cpu" else 1
                )
                sample_upper_bound = (
                    transfer_resident_target_bytes
                    + materialization_upper_bound.materialization_peak_logical_tensor_bytes_upper_bound
                    + native_prepared_sample_scratch_upper_bound
                )
                if target_bytes > memory_policy.maximum_chunk_target_tensor_bytes:
                    _fail_memory("dense chunk target budget fails before target load")
                if (
                    minimum_target_bridge_bytes
                    > memory_policy.maximum_target_decode_bridge_peak_logical_tensor_bytes
                ):
                    _fail_memory(
                        "dense chunk minimum target-transfer bridge budget fails "
                        "before target load"
                    )
                if sample_upper_bound > memory_policy.maximum_sample_launch_tensor_bytes:
                    _fail_memory("dense chunk sample-launch budget fails before target load")

                targets = load_chunk_targets.load(chunk)
                current_chunk_targets = targets
                if not isinstance(targets, PaperKineticDenseChunkTargets):
                    _fail_type("dense target loader returned the wrong sealed type")
                targets.assert_current(source, request, chunk, device=device)
                peak_chunk_target_bytes = max(
                    peak_chunk_target_bytes,
                    targets.logical_tensor_bytes,
                )
                selected_pixel_read_modes.add(targets.selected_pixel_read_mode)
                selected_pixel_read_source_provenances.add(
                    targets.selected_pixel_read_source_provenance
                )
                selected_pixel_read_call_count += (
                    targets.selected_pixel_read_call_count
                )
                direct_selected_pixel_observation_count += (
                    targets.direct_selected_pixel_observation_count
                )
                bounded_region_selected_pixel_observation_count += (
                    targets.bounded_region_selected_pixel_observation_count
                )
                full_frame_fallback_observation_count += (
                    targets.full_frame_fallback_observation_count
                )
                bounded_region_materialization_count += (
                    targets.bounded_region_materialization_count
                )
                decoded_frame_count += targets.decoded_frame_count
                decoded_frame_mps_completion_fence_call_count += (
                    targets.decoded_frame_mps_completion_fence_call_count
                )
                peak_cpu_decoded_frame_bytes = max(
                    peak_cpu_decoded_frame_bytes,
                    targets.maximum_cpu_decoded_frame_tensor_bytes,
                )
                peak_bounded_region_materialization_bytes = max(
                    peak_bounded_region_materialization_bytes,
                    targets.maximum_bounded_region_materialization_tensor_bytes,
                )
                peak_source_visible_target_read_bytes = max(
                    peak_source_visible_target_read_bytes,
                    targets.source_visible_target_read_peak_logical_tensor_bytes_upper_bound,
                )
                if targets.transient_mapped_address_space_bytes > 0:
                    mapped_selected_pixel_read_call_count += 1
                    mapping_closed_before_return_count += int(
                        targets.mapping_closed_before_return
                    )
                peak_transient_mapped_address_space_bytes = max(
                    peak_transient_mapped_address_space_bytes,
                    targets.transient_mapped_address_space_bytes,
                )
                peak_requested_unique_mapped_page_count = max(
                    peak_requested_unique_mapped_page_count,
                    targets.maximum_requested_unique_mapped_page_count,
                )
                peak_mapped_page_size_bytes = max(
                    peak_mapped_page_size_bytes,
                    targets.mapped_page_size_bytes,
                )
                peak_requested_mapped_page_bytes_upper_bound = max(
                    peak_requested_mapped_page_bytes_upper_bound,
                    targets.maximum_requested_mapped_page_bytes_upper_bound,
                )
                cumulative_requested_mapped_page_count += (
                    targets.total_requested_unique_mapped_page_count
                )
                cumulative_requested_mapped_page_bytes_upper_bound += (
                    targets.total_requested_mapped_page_bytes_upper_bound
                )
                peak_cpu_chunk_target_bytes = max(
                    peak_cpu_chunk_target_bytes,
                    targets.cpu_chunk_target_tensor_bytes,
                )
                peak_device_chunk_target_bytes = max(
                    peak_device_chunk_target_bytes,
                    targets.device_chunk_target_tensor_bytes,
                )
                transferred_target_payload_bytes += (
                    targets.device_chunk_target_tensor_bytes
                )
                peak_target_decode_bridge_bytes = max(
                    peak_target_decode_bridge_bytes,
                    targets.target_decode_bridge_peak_logical_tensor_bytes,
                )
                sample_iterator = iter_paper_kinetic_dense_chunk_sample_blocks(
                    artifact.sampler,
                    provider,
                    source,
                    request,
                    chunk,
                    targets,
                    loss_normalization_id=loss_normalization_id,
                    maximum_samples_per_launch=effective_maximum_samples_per_launch,
                    maximum_sample_materialization_logical_tensor_bytes=(
                        memory_policy.maximum_sample_materialization_logical_tensor_bytes
                    ),
                )
                chunk_streamed = 0
                launched_observation_ids: list[int] = []
                with closing(sample_iterator) as sample_blocks:
                    for materialization in sample_blocks:
                        current_sample_materialization = materialization
                        materialization.assert_retained()
                        sample_block = materialization.sample_block
                        if not isinstance(
                            sample_block,
                            PaperKineticRowRaggedSampleBlock,
                        ):
                            _fail_arithmetic(
                                "dense sample materialization lost its sample block"
                            )
                        current_sample_block = sample_block
                        materialization_plan = _sample_materialization_memory_plan(
                            sample_count=sample_block.sample_count,
                            node_count=int(sample_block.sample_to_node_f32.shape[1]),
                            maximum_logical_tensor_bytes=(
                                memory_policy.maximum_sample_materialization_logical_tensor_bytes
                            ),
                        )
                        peak_sample_materialization_bytes = max(
                            peak_sample_materialization_bytes,
                            materialization_plan.materialization_peak_logical_tensor_bytes_upper_bound,
                        )
                        peak_interpolation_evaluator_scratch_bytes = max(
                            peak_interpolation_evaluator_scratch_bytes,
                            materialization_plan.interpolation_evaluator_scratch_logical_tensor_bytes_upper_bound,
                        )
                        maximum_interpolation_rows_per_subchunk = max(
                            maximum_interpolation_rows_per_subchunk,
                            materialization_plan.interpolation_rows_per_subchunk,
                        )
                        digest = sample_block.native_block_generation_digest
                        block_state = active.get(digest)
                        if block_state is None:
                            runtime = lane.runtime_for_digest(digest)
                            current_forward_runtime = runtime
                            compact_gather_lifetime = (
                                _DenseCompactMaterialGatherLifetime(
                                    global_site_rgba_f32=global_site_rgba_f32,
                                    source_site_ids_i64=runtime.source_site_ids_i64,
                                    _global_identity=id(global_site_rgba_f32),
                                    _indices_identity=id(
                                        runtime.source_site_ids_i64
                                    ),
                                    _seal=_COMPACT_GATHER_LIFETIME_SEAL,
                                )
                            )
                            compact_gather_lifetime_install_count += 1
                            compact_gather_lifetime.assert_retained()
                            current_compact_gather_lifetime = (
                                compact_gather_lifetime
                            )
                            gathered_material = global_site_rgba_f32.index_select(
                                0,
                                runtime.source_site_ids_i64,
                            )
                            compact_gather_lifetime.publish_gathered(
                                gathered_material
                            )
                            compact_material = gathered_material.contiguous()
                            compact_gather_lifetime.publish_materialized(
                                compact_material
                            )
                            current_forward_compact_material = compact_material
                            forward_into_lifetime = (
                                prepare_kinetic_native_node_forward_into_lifetime(
                                    native_session,
                                    runtime,
                                    compact_material,
                                )
                            )
                            forward_into_lifetime_install_count += 1
                            current_forward_into_lifetime = forward_into_lifetime
                            node_chart_out = torch.empty(
                                (runtime.row_count, runtime.node_count, 4),
                                dtype=torch.float32,
                                device=device,
                            )
                            current_forward_node_chart_out = node_chart_out
                            forward_into_lifetime.publish_output(node_chart_out)
                            token = native_session.launch_node_forward_into(
                                forward_into_lifetime,
                            )
                            current_forward_token = token
                            current_forward_grad_node_chart = torch.zeros_like(
                                token.world.node_chart_f32
                            )
                            current_forward_loss = torch.zeros(
                                (1,),
                                dtype=torch.float32,
                                device=device,
                            )
                            block_state = _ActiveBlock(
                                token=token,
                                grad_node_chart_f32=(
                                    current_forward_grad_node_chart
                                ),
                                loss_f32=current_forward_loss,
                                compact_gather_lifetime=(
                                    compact_gather_lifetime
                                ),
                                forward_into_lifetime=forward_into_lifetime,
                            )
                            active[digest] = block_state
                            current_forward_loss = None
                            current_forward_grad_node_chart = None
                            current_forward_token = None
                            current_forward_node_chart_out = None
                            current_forward_into_lifetime = None
                            current_forward_compact_material = None
                            current_compact_gather_lifetime = None
                            current_forward_runtime = None
                            del gathered_material, compact_material
                            del (
                                compact_gather_lifetime,
                                forward_into_lifetime,
                                node_chart_out,
                                token,
                                runtime,
                            )
                        current_sample_lifetime = (
                            native_session.launch_sample_accumulate(
                                block_state.token,
                                sample_block,
                                sampler=artifact.sampler,
                                background_rgb_f32=background_rgb_f32,
                                loss_f32=block_state.loss_f32,
                                grad_node_chart_f32=(
                                    block_state.grad_node_chart_f32
                                ),
                                cone_diagnostic_i32=cone,
                                cone_tolerance=cone_tolerance,
                            )
                        )
                        sample_completion = native_session.settle_sample_accumulate(
                            current_sample_lifetime,
                            device_completion_fence=(
                                invoke_device_completion_fence
                            ),
                            device_completion_fence_provenance=(
                                device_completion_fence_provenance
                            ),
                        )
                        sample_completion.assert_current()
                        sample_backpressure_fence_call_count += (
                            sample_completion.device_completion_fence_call_count
                        )
                        # Settlement proves both the native prepared payload
                        # and materialization predecessor commands complete.
                        materialization.release_after_fence()
                        current_sample_lifetime = None
                        current_sample_block = None
                        current_sample_materialization = None
                        native_prepared_sample_scratch_bytes = (
                            4 * sample_block.sample_count + 20
                        )
                        peak_sample_launch_bytes = max(
                            peak_sample_launch_bytes,
                            sample_block.retained_tensor_bytes,
                        )
                        peak_sample_launch_node_count = max(
                            peak_sample_launch_node_count,
                            sample_block.node_count,
                        )
                        sample_node_interaction_count += (
                            sample_block.linear_weight_interactions
                        )
                        peak_native_prepared_sample_scratch_bytes = max(
                            peak_native_prepared_sample_scratch_bytes,
                            native_prepared_sample_scratch_bytes,
                        )
                        peak_public_sample_launch_logical_bytes = max(
                            peak_public_sample_launch_logical_bytes,
                            (
                                targets.cpu_chunk_target_tensor_bytes
                                * (2 if device.type != "cpu" else 1)
                                + sample_block.retained_tensor_bytes
                                + native_prepared_sample_scratch_bytes
                            ),
                        )
                        chunk_streamed += sample_block.sample_count
                        launched_observation_ids.extend(
                            int(value)
                            for value in sample_block.flat_sample_index_i64.tolist()
                        )
                        sample_launch_count += 1
                        del sample_completion, sample_block, materialization
                if chunk_streamed != chunk.observation_count:
                    _fail_arithmetic("dense chunk sample dispatch changed coverage")
                expected_observation_ids = tuple(
                    sorted(
                        observation.observation_id
                        for observation in chunk.observations
                    )
                )
                if tuple(sorted(launched_observation_ids)) != expected_observation_ids:
                    _fail_arithmetic(
                        "dense chunk sample dispatch changed observation identities"
                    )
                # Every command which can consume the target transfer has now
                # crossed a successful sample-completion fence.  Retire the
                # exact loader lifetime before advancing to the next chunk.
                load_chunk_targets.release_returned_after_completion_fence(
                    targets
                )
                peak_chunk_dispatch_identity_bytes = max(
                    peak_chunk_dispatch_identity_bytes,
                    8 * len(launched_observation_ids),
                )
                chunk_count += 1
                streamed_observation_count += chunk_streamed
                _update_digest(chunk_manifest, chunk.generation_digest)
                current_chunk_targets = None
                del targets, chunk

        if (
            session.request_count != request_count_before + 1
            or session.emitted_observation_count - emitted_before != expected_observation_count
            or streamed_observation_count != expected_observation_count
        ):
            _fail_arithmetic("dense replay request did not advance exact coverage")

        if fused_full_geometry:
            prepared_blocks = []
            for runtime in lane.runtimes:
                digest = runtime.payload.block.generation_digest
                block_state = active.get(digest)
                if block_state is None:
                    continue
                if not isinstance(
                    block_state.token,
                    KineticNativeMaterialStepWorldToken,
                ):
                    _fail_arithmetic(
                        "dense fused active block lost its exact world token"
                    )
                block_state.token.assert_current(native_session)
                prepared_blocks.append(
                    prepare_kinetic_native_equal_rank_fused_direct_full_vjp_v1(
                        block_state.token.world,
                        lowering=artifact.sampler.lowering,
                        sources=artifact.sampler.sources,
                    )
                )
            fused_prepared_blocks = tuple(prepared_blocks)
            del prepared_blocks
            if not fused_prepared_blocks:
                _fail_arithmetic(
                    "dense fused request produced no prepared active blocks"
                )
            fused_prepared_owned_logical_tensor_bytes = sum(
                block.memory.owned_persistent_tensor_bytes
                for block in fused_prepared_blocks
            )
            if (
                fused_prepared_owned_logical_tensor_bytes
                > fused_prepared_owned_upper_bound
                or fused_prepared_owned_logical_tensor_bytes
                > memory_policy.maximum_fused_prepared_owned_logical_tensor_bytes
            ):
                _fail_arithmetic(
                    "dense fused prepared payload exceeded its pre-allocation proof"
                )
            fused_execution_receipt = (
                native_session.execute_fused_full_geometry_vjp_transaction(
                    fused_prepared_blocks,
                    max_output_scratch_tensor_bytes=(
                        memory_policy.maximum_fused_output_scratch_logical_tensor_bytes
                    ),
                    device_completion_fence=invoke_device_completion_fence,
                    device_completion_fence_provenance=(
                        device_completion_fence_provenance
                    ),
                )
            )
            fused_execution_receipt.assert_current(native_session)
            fused_transaction_result = (
                fused_execution_receipt.transaction_result
            )
            fused_transaction_result.assert_current()
            if (
                not fused_execution_receipt.active_manifest_coverage_certified
                or fused_execution_receipt.length_cotangent_allocated
                or fused_execution_receipt.optimizer_commit_performed
                or fused_execution_receipt.active_block_generation_ids
                != tuple(
                    runtime.payload.block.generation_digest
                    for runtime in lane.runtimes
                    if runtime.payload.block.generation_digest in active
                )
                or fused_transaction_result.active_block_generation_ids
                != fused_execution_receipt.active_block_generation_ids
            ):
                _fail_arithmetic(
                    "dense fused receipt lost exact active-manifest provenance"
                )
            fused_output_scratch_logical_tensor_bytes = (
                fused_transaction_result.retained_output_tensor_bytes
            )
            fused_compact_output_scratch_logical_tensor_bytes = _tensor_bytes(
                *fused_transaction_result.grad_compact_site_rgba_f32_by_block
            )
            fused_global_output_scratch_logical_tensor_bytes = _tensor_bytes(
                fused_transaction_result.grad_global_positions0_f32,
                fused_transaction_result.grad_global_velocities_f32,
                fused_transaction_result.grad_global_weight_coefficients_f32,
            )
            if (
                fused_output_scratch_logical_tensor_bytes
                != fused_compact_output_scratch_logical_tensor_bytes
                + fused_global_output_scratch_logical_tensor_bytes
                or fused_output_scratch_logical_tensor_bytes
                > fused_output_upper_bound
                or fused_output_scratch_logical_tensor_bytes
                > memory_policy.maximum_fused_output_scratch_logical_tensor_bytes
            ):
                _fail_arithmetic(
                    "dense fused output scratch exceeded its pre-allocation proof"
                )
            fused_compact_material_scatter_elements = (
                _accumulate_fused_compact_material_and_loss(
                    lane.spatial_bundle,
                    active,
                    active_block_generation_ids=(
                        fused_execution_receipt.active_block_generation_ids
                    ),
                    grad_compact_site_rgba_f32_by_block=(
                        fused_transaction_result.grad_compact_site_rgba_f32_by_block
                    ),
                    local_union_bar=local_union_bar,
                    local_loss=local_loss,
                )
            )
            (
                bridged_geometry,
                fused_geometry_bridge_visible_tensor_bytes,
            ) = _bridge_fused_global_geometry_bars_to_cpu_f64(
                fused_transaction_result,
                artifact.sampler,
                maximum_bridge_visible_peak_logical_tensor_bytes=(
                    memory_policy.maximum_fused_geometry_bridge_visible_peak_logical_tensor_bytes
                ),
            )
            (
                local_positions0_f64,
                local_velocities_f64,
                local_weight_coefficients_f64,
            ) = bridged_geometry
            del bridged_geometry
            invoke_device_completion_fence()
            fused_post_accept_commit_fence_call_count = 1
            active_block_commit_fence_call_count = 1
            lifetime_release_fence_completed = True
            retire_all_forward_lifetimes_after_proven_completion()
            fused_transaction_fence_call_count = (
                fused_execution_receipt.device_completion_fence_call_count
            )
            fused_active_manifest_certified = (
                fused_execution_receipt.active_manifest_coverage_certified
            )
            maximum_active_block_commit_scratch_tensor_bytes = (
                fused_compact_output_scratch_logical_tensor_bytes
            )
            maximum_geometry_bridge_visible_tensor_bytes = (
                fused_geometry_bridge_visible_tensor_bytes
            )
            geometry_row_vjp_call_count = sum(
                block.world.row_count for block in fused_prepared_blocks
            )
            weight_count = int(
                _shared_kinetic_sites(artifact.sampler)
                .weight_coefficients.shape[1]
            )
            geometry_compact_to_global_scatter_elements = sum(
                block.world.compact_site_count * (6 + weight_count)
                for block in fused_prepared_blocks
            )
            active.clear()

        for runtime in lane.runtimes:
            digest = runtime.payload.block.generation_digest
            block_state = active.get(digest)
            if block_state is None:
                continue
            current_reverse_block_state = block_state
            compact_bar = torch.empty(
                (runtime.compact_site_count, 4),
                dtype=torch.float32,
                device=device,
            )
            current_reverse_compact_bar = compact_bar
            maximum_active_block_commit_scratch_tensor_bytes = max(
                maximum_active_block_commit_scratch_tensor_bytes,
                compact_bar.numel() * compact_bar.element_size(),
            )
            material_execution = None
            if staged_full_geometry:
                if any(
                    value is None
                    for value in (
                        local_positions0_f64,
                        local_velocities_f64,
                        local_weight_coefficients_f64,
                    )
                ):
                    _fail_arithmetic("dense full-geometry request lost its local bars")
                execution = native_session.launch_full_geometry_vjp(
                    block_state.token,
                    block_state.grad_node_chart_f32,
                    compact_grad_site_rgba_f32=compact_bar,
                )
                current_full_geometry_execution = execution
                execution.assert_current(
                    native_session,
                    loss_f32=block_state.loss_f32,
                )
                native_vjp = execution.native_vjp_result
                current_native_vjp = native_vjp
                native_provenance = kinetic_native_equal_rank_vjp_provenance_id(
                    native_vjp
                )
                geometry = reduce_kinetic_native_equal_rank_sparse_geometry_vjp(
                    native_vjp,
                    artifact.sampler,
                    expected_native_vjp_provenance_id=native_provenance,
                    device_completion_fence=invoke_device_completion_fence,
                    device_completion_fence_provenance=(
                        device_completion_fence_provenance
                    ),
                    maximum_bridge_visible_peak_logical_tensor_bytes=(
                        memory_policy.maximum_geometry_bridge_visible_peak_logical_tensor_bytes
                    ),
                    include_ray_gradients=optimize_camera_rays,
                )
                current_geometry_reduction = geometry
                if (
                    geometry.native_block_generation_digest != digest
                    or geometry.ray_gradients_included != optimize_camera_rays
                    or (
                        not optimize_camera_rays
                        and (
                            geometry.ray_bar_keys
                            or geometry.grad_track_ray_coefficients_f64.numel()
                        )
                    )
                ):
                    _fail_arithmetic(
                        "dense geometry reduction changed block/ray provenance"
                    )
                local_positions0_f64.index_add_(
                    0,
                    geometry.source_site_ids_i64,
                    geometry.grad_compact_positions0_f64,
                )
                local_velocities_f64.index_add_(
                    0,
                    geometry.source_site_ids_i64,
                    geometry.grad_compact_velocities_f64,
                )
                local_weight_coefficients_f64.index_add_(
                    0,
                    geometry.source_site_ids_i64,
                    geometry.grad_compact_weight_coefficients_f64,
                )
                if optimize_camera_rays:
                    if (
                        local_track_ray_coefficients_f64 is None
                        or not set(geometry.ray_bar_keys).issubset(
                            request_ray_bar_keys
                        )
                    ):
                        _fail_arithmetic(
                            "dense geometry reduction changed trainable ray provenance"
                        )
                    request_ray_position = {
                        key: index
                        for index, key in enumerate(request_ray_bar_keys)
                    }
                    for compact_index, key in enumerate(geometry.ray_bar_keys):
                        local_track_ray_coefficients_f64[
                            request_ray_position[key]
                        ].add_(
                            geometry.grad_track_ray_coefficients_f64[
                                compact_index
                            ]
                        )
                geometry_reduction_fence_call_count += (
                    geometry.device_completion_fence_call_count
                )
                geometry_row_vjp_call_count += geometry.row_geometry_vjp_call_count
                geometry_dense_global_site_accumulation_elements += (
                    geometry.dense_global_site_accumulation_elements
                )
                geometry_all_site_owner_validation_evaluations += (
                    geometry.all_site_owner_validation_evaluations
                )
                geometry_compact_to_global_scatter_elements += (
                    geometry.compact_site_count
                    * (6 + geometry.weight_coefficient_count)
                )
                maximum_simultaneous_geometry_jw_length_bar_tensors = max(
                    maximum_simultaneous_geometry_jw_length_bar_tensors,
                    geometry.maximum_simultaneous_jw_length_bar_tensors,
                )
                maximum_native_length_bar_tensor_bytes = max(
                    maximum_native_length_bar_tensor_bytes,
                    execution.native_length_bar_tensor_bytes,
                )
                maximum_geometry_bridge_visible_tensor_bytes = max(
                    maximum_geometry_bridge_visible_tensor_bytes,
                    geometry.memory.bridge_visible_peak_logical_tensor_bytes,
                )
                completion = native_session.consume_full_geometry_vjp_execution(
                    execution,
                    geometry_reduction=geometry,
                    expected_device_completion_fence_provenance=(
                        device_completion_fence_provenance
                    ),
                )
                current_geometry_completion = completion
                completion.assert_current()
                geometry_completion_receipt_count += 1
            elif not full_geometry:
                material_execution = native_session.launch_material_vjp(
                    block_state.token,
                    block_state.grad_node_chart_f32,
                    compact_grad_site_rgba_f32=compact_bar,
                )
                current_material_execution = material_execution
            else:
                _fail_arithmetic(
                    "dense fused request retained an active block after its transaction"
                )
            binding = lane.spatial_bundle.binding_for_digest(digest)
            local_union_bar.index_add_(
                0,
                binding.compact_to_union_i64,
                compact_bar,
            )
            local_loss.add_(block_state.loss_f32)
            # Commit commands and compact reverse scratch must complete before
            # the next block starts, otherwise asynchronous queues can grow
            # with active-block count despite bounded Python references.
            invoke_device_completion_fence()
            active_block_commit_fence_call_count += 1
            retire_block_forward_lifetimes_after_proven_completion(
                block_state
            )
            del active[digest]
            current_reverse_block_state = None
            current_reverse_compact_bar = None
            current_material_execution = None
            current_full_geometry_execution = None
            current_native_vjp = None
            current_geometry_reduction = None
            current_geometry_completion = None
            if material_execution is not None:
                del material_execution
            if staged_full_geometry:
                del completion, geometry, native_vjp, execution
            del compact_bar, block_state
        if active:
            _fail_arithmetic("dense cached request retained an unreversed block")
        telemetry = native_session.seal()
        # The accepted outputs and prepared launch payloads were fenced above,
        # and ``seal`` has consumed their executor/session provenance. Keep
        # only scalar accounting before constructing the public delta.
        fused_prepared_blocks = ()
        fused_execution_receipt = None
        fused_transaction_result = None
        if (
            telemetry.streamed_sample_count != expected_observation_count
            or telemetry.native_sample_completion_fence_count
            != sample_launch_count
            or transferred_target_payload_bytes != expected_observation_count * 12
            or not (
                expected_observation_count
                <= sample_node_interaction_count
                <= expected_observation_count * max_node_count
            )
            or not 1 <= peak_sample_launch_node_count <= max_node_count
        ):
            _fail_arithmetic("dense native telemetry changed request coverage")
    except BaseException as error:
        body_error = error
        original_traceback = error.__traceback__
        if native_session is not None and not native_session._sealed:
            try:
                native_session.abort(
                    device_completion_fence=invoke_device_completion_fence,
                    device_completion_fence_provenance=(
                        device_completion_fence_provenance
                    ),
                )
                load_chunk_targets.release_active_after_completion_fence()
                retire_all_forward_lifetimes_after_proven_completion()
                lifetime_release_fence_completed = True
            except BaseException as abort_error:
                _quarantine_dense_async_failure(
                    accumulator,
                    stage="native-session-abort",
                    original_error=error,
                    original_traceback=original_traceback,
                    cleanup_fence_error=abort_error,
                    retained_references=request_lifetime_references(),
                    device_completion_fence_provenance=(
                        device_completion_fence_provenance
                    ),
                )
                error.add_note(
                    "dense native-session abort fence failed; active/native/"
                    "lane references are quarantined and the process must restart"
                )
                raise error.with_traceback(original_traceback) from abort_error
        if accumulator._async_failure_quarantine is None:
            _poison_dense_step_accumulator(accumulator)
        raise
    finally:
        if (
            lane is not None
            and accumulator._async_failure_quarantine is None
        ):
            try:
                if not lifetime_release_fence_completed:
                    invoke_device_completion_fence()
            except BaseException as fence_error:
                original_error = body_error if body_error is not None else fence_error
                _quarantine_dense_async_failure(
                    accumulator,
                    stage="outer-lane-release",
                    original_error=original_error,
                    original_traceback=original_error.__traceback__,
                    cleanup_fence_error=fence_error,
                    retained_references=request_lifetime_references(),
                    device_completion_fence_provenance=(
                        device_completion_fence_provenance
                    ),
                )
                if body_error is None:
                    raise
                body_error.add_note(
                    "dense lane-release fence also failed; active/native/lane "
                    "references are quarantined and the process must restart: "
                    f"{type(fence_error).__name__}: {fence_error}"
                )
            else:
                retire_all_forward_lifetimes_after_proven_completion()
                active.clear()
                native_session = None
                lane = None

    if telemetry is None or local_union_bar is None or commit_source_site_ids_i64 is None:
        raise ArithmeticError("dense cached request produced no sealed outcome")
    expected_fence_count = (
        sample_launch_count + 2
        if fused_full_geometry
        else (
            sample_launch_count
            + 1
            + telemetry.active_native_block_count
            + (
                telemetry.active_native_block_count
                if staged_full_geometry
                else 0
            )
        )
    )
    if fence_call_count != expected_fence_count:
        raise ArithmeticError("dense cached request completion-fence count changed")
    if sample_backpressure_fence_call_count != sample_launch_count:
        raise ArithmeticError("dense cached request sample backpressure changed")
    if (
        telemetry.native_sample_prepare_count
        != telemetry.native_sample_launch_count
        or telemetry.native_sample_launch_count
        != telemetry.native_sample_completion_fence_count
        or telemetry.native_sample_completion_fence_count
        != sample_backpressure_fence_call_count
    ):
        raise ArithmeticError("dense cached request sample lifetime proof changed")
    if active_block_commit_fence_call_count != (
        1 if fused_full_geometry else telemetry.active_native_block_count
    ):
        raise ArithmeticError("dense cached request active-block commit fencing changed")
    if geometry_reduction_fence_call_count != (
        telemetry.active_native_block_count if staged_full_geometry else 0
    ):
        raise ArithmeticError("dense cached geometry fence coverage changed")
    if geometry_completion_receipt_count != (
        telemetry.active_native_block_count if staged_full_geometry else 0
    ):
        raise ArithmeticError("dense cached geometry completion proof changed")
    if staged_full_geometry and (
        geometry_dense_global_site_accumulation_elements != 0
        or geometry_all_site_owner_validation_evaluations != 0
        or maximum_simultaneous_geometry_jw_length_bar_tensors != 1
    ):
        raise ArithmeticError("dense cached request lost its sparse geometry contract")
    if fused_full_geometry and (
        telemetry.reverse_mode != "fused_full_geometry"
        or telemetry.native_fused_full_geometry_vjp_launch_count
        != telemetry.active_native_block_count
        or telemetry.native_fused_full_geometry_transaction_count != 1
        or telemetry.native_fused_full_geometry_completion_fence_count != 1
        or telemetry.native_length_bar_tensor_bytes != 0
        or maximum_native_length_bar_tensor_bytes != 0
        or maximum_simultaneous_geometry_jw_length_bar_tensors != 0
        or geometry_reduction_fence_call_count != 0
        or geometry_completion_receipt_count != 0
        or fused_transaction_fence_call_count != 1
        or fused_post_accept_commit_fence_call_count != 1
        or not fused_active_manifest_certified
    ):
        raise ArithmeticError("dense cached request lost its fused geometry contract")
    if selected_pixel_read_call_count != chunk_count:
        raise ArithmeticError("dense cached target reader changed one-read-per-chunk coverage")
    if (
        load_chunk_targets._active_lifetime is not None
        or load_chunk_targets.completed_load_count != chunk_count
        or load_chunk_targets.failed_after_enqueue_count != 0
    ):
        raise ArithmeticError("dense sealed target-loader lifetime coverage changed")
    if (
        direct_selected_pixel_observation_count
        + bounded_region_selected_pixel_observation_count
        + full_frame_fallback_observation_count
        != streamed_observation_count
    ):
        raise ArithmeticError("dense cached target reader changed observation coverage")
    if not selected_pixel_read_modes:
        raise ArithmeticError("dense cached request recorded no target read mode")
    if decoded_frame_mps_completion_fence_call_count != 0:
        raise ArithmeticError("dense cached CPU target decode unexpectedly fenced MPS")
    if not bool(torch.isfinite(local_loss).all().item()) or not bool(torch.isfinite(local_union_bar).all().item()):
        raise FloatingPointError("dense cached request produced non-finite loss/material bars")

    if full_geometry and any(
        value is None
        for value in (
            local_positions0_f64,
            local_velocities_f64,
            local_weight_coefficients_f64,
        )
    ):
        raise ArithmeticError("dense full-geometry request produced no local geometry bars")
    if optimize_camera_rays and local_track_ray_coefficients_f64 is None:
        raise ArithmeticError("trainable-camera request produced no ray bars")
    receipt_provisional = PaperKineticDenseCachedRequestReceipt(
        source_generation_digest=source.generation_digest,
        request_generation_digest=request.generation_digest,
        artifact_generation_digest=artifact.generation_digest,
        session_identity=id(session),
        session_request_count_before=request_count_before,
        session_request_count_after=session.request_count,
        session_emitted_observation_count_before=emitted_before,
        session_emitted_observation_count_after=session.emitted_observation_count,
        expected_observation_count=expected_observation_count,
        replay_chunk_count=chunk_count,
        replay_chunk_manifest_digest=chunk_manifest.hexdigest(),
        generation_digest="",
        _seal=_RECEIPT_SEAL,
    )
    receipt = replace(
        receipt_provisional,
        generation_digest=_receipt_digest(receipt_provisional),
    )
    geometry_delta_tensors = (
        (
            local_positions0_f64,
            local_velocities_f64,
            local_weight_coefficients_f64,
        )
        if full_geometry
        else ()
    )
    ray_delta_tensors = (
        (local_track_ray_coefficients_f64,) if optimize_camera_rays else ()
    )
    delta_tensors = (
        commit_source_site_ids_i64,
        local_union_bar,
        local_loss,
        *geometry_delta_tensors,
        *ray_delta_tensors,
    )
    delta_provisional = PaperKineticDenseRequestGradientDelta(
        source_generation_digest=source.generation_digest,
        request_generation_digest=request.generation_digest,
        artifact_generation_digest=artifact.generation_digest,
        step_generation_id=accumulator.step_generation_id,
        receipt=receipt,
        telemetry=telemetry,
        full_geometry=full_geometry,
        optimize_camera_rays=accumulator.optimize_camera_rays,
        ray_bar_keys=request_ray_bar_keys,
        source_site_ids_i64=commit_source_site_ids_i64,
        grad_union_site_rgba_f32=local_union_bar,
        loss_f32=local_loss,
        grad_positions0_f64=local_positions0_f64,
        grad_velocities_f64=local_velocities_f64,
        grad_weight_coefficients_f64=local_weight_coefficients_f64,
        grad_track_ray_coefficients_f64=local_track_ray_coefficients_f64,
        sealed_tensor_signatures=tuple(
            _tensor_signature(tensor) for tensor in delta_tensors
        ),
        generation_digest="",
        consumed=False,
        consumed_by_accumulator_generation_digest="",
        _accumulator_identity=id(accumulator),
        _seal=_REQUEST_DELTA_SEAL,
    )
    delta = replace(
        delta_provisional,
        generation_digest=_request_delta_digest(delta_provisional),
    )
    structural_accounting = _request_structural_accounting(
        artifact,
        telemetry,
    )
    selected_pixel_read_mode = (
        next(iter(selected_pixel_read_modes))
        if len(selected_pixel_read_modes) == 1
        else "mixed"
    )
    selected_pixel_read_acceptance_capable = (
        selected_pixel_read_modes
        <= {"direct_pixels", "certified_bounded_region"}
        and decoded_frame_count == 0
        and full_frame_fallback_observation_count == 0
    )
    accounting: dict[str, Any] = {
        "provenance": REQUEST_PROVENANCE,
        "runtime_status": REQUEST_STATUS,
        "expected_observation_count": expected_observation_count,
        "streamed_observation_count": streamed_observation_count,
        "replay_chunk_count": chunk_count,
        "sample_launch_count": sample_launch_count,
        "sample_node_interaction_count": sample_node_interaction_count,
        "transferred_target_payload_bytes": transferred_target_payload_bytes,
        "native_lane_prepare_count": 1,
        "native_lane_two_phase_construction": True,
        "union_and_runtime_construction_lifetimes_retained_through_lane_fence": True,
        "accelerator_release_capability_integrated": False,
        "native_lane_fence_count": fence_call_count,
        "sample_completion_fence_call_count": (
            sample_backpressure_fence_call_count
        ),
        "native_sample_lifetime_token_count": (
            telemetry.native_sample_launch_count
        ),
        "native_sample_lifetime_settle_count": (
            telemetry.native_sample_completion_fence_count
        ),
        "native_sample_completion_fence_count": (
            telemetry.native_sample_completion_fence_count
        ),
        "sample_completion_fence_provenance": (
            telemetry.sample_completion_fence_provenance
        ),
        "maximum_in_flight_sample_lifetime_token_count": (
            telemetry.maximum_simultaneous_sample_lifetime_count
        ),
        "retained_sample_lifetime_token_count_after_seal": (
            telemetry.outstanding_sample_lifetime_count_at_seal
        ),
        "sample_lifetime_token_history_retained": (
            telemetry.sample_lifetime_history_retained
        ),
        "sample_lifetime_additional_logical_tensor_bytes": (
            telemetry.sample_lifetime_additional_logical_tensor_bytes
        ),
        "sample_lifetime_python_heap_bytes_measured": (
            telemetry.sample_lifetime_python_heap_bytes_measured
        ),
        "sample_lifetime_roots_released_only_after_completion_fence": True,
        "sample_materialization_predecessor_roots_leased_until_fence": True,
        "chunk_cpu_transfer_source_retained_through_sample_fences": True,
        "active_block_commit_fence_call_count": (
            active_block_commit_fence_call_count
        ),
        "active_block_commit_fenced_before_scratch_release": True,
        "maximum_active_block_commit_scratch_tensor_bytes": (
            maximum_active_block_commit_scratch_tensor_bytes
        ),
        "maximum_in_flight_active_block_commit_scratch_count": (
            telemetry.active_native_block_count if fused_full_geometry else 1
        ),
        "sample_launch_fence_requested_after_every_launch": True,
        "maximum_requested_in_flight_sample_launches": 1,
        "real_device_fence_semantics_verified": False,
        "lane_reused_across_all_chunks": True,
        "node_forward_abi": "caller_preallocated_into_v1",
        "return_allocating_node_forward_launch_count": 0,
        "caller_preallocated_node_forward_launch_count": (
            telemetry.native_node_forward_launch_count
        ),
        "forward_into_lifetime_install_count": (
            forward_into_lifetime_install_count
        ),
        "forward_into_lifetime_retire_count": (
            forward_into_lifetime_retire_count
        ),
        "compact_gather_lifetime_install_count": (
            compact_gather_lifetime_install_count
        ),
        "compact_gather_lifetime_retire_count": (
            compact_gather_lifetime_retire_count
        ),
        "retained_forward_into_lifetime_count_after_request": 0,
        "retained_compact_gather_lifetime_count_after_request": 0,
        "forward_into_lifetime_additional_logical_tensor_bytes": 0,
        "compact_gather_lifetime_additional_logical_tensor_bytes": 0,
        "forward_predecessor_and_output_roots_released_only_after_reverse_or_abort_fence": True,
        "native_node_forward_launch_count": telemetry.native_node_forward_launch_count,
        "native_material_word_vjp_launch_count": telemetry.native_material_word_vjp_launch_count,
        "native_full_geometry_vjp_launch_count": (
            telemetry.native_full_geometry_vjp_launch_count
        ),
        "native_fused_full_geometry_vjp_launch_count": (
            telemetry.native_fused_full_geometry_vjp_launch_count
        ),
        "native_fused_full_geometry_transaction_count": (
            telemetry.native_fused_full_geometry_transaction_count
        ),
        "native_fused_full_geometry_completion_fence_count": (
            telemetry.native_fused_full_geometry_completion_fence_count
        ),
        "exactly_one_material_vjp_per_active_block": not full_geometry,
        "exactly_one_full_geometry_vjp_per_active_block": staged_full_geometry,
        "exactly_one_fused_full_geometry_vjp_per_active_block": (
            fused_full_geometry
        ),
        "reverse_started_after_request_coverage": True,
        "geometry_reduction_fence_call_count": (
            geometry_reduction_fence_call_count
        ),
        "geometry_completion_receipt_count": geometry_completion_receipt_count,
        "geometry_completion_receipt_retains_native_tensors": False,
        "geometry_row_vjp_call_count": geometry_row_vjp_call_count,
        "geometry_reduction_mode": (
            "fused_direct_v1"
            if fused_full_geometry
            else "certified_sparse_compact"
            if staged_full_geometry
            else "material_only"
        ),
        "geometry_dense_global_site_accumulation_elements": (
            geometry_dense_global_site_accumulation_elements
        ),
        "geometry_all_site_owner_validation_evaluations": (
            geometry_all_site_owner_validation_evaluations
        ),
        "geometry_compact_to_global_scatter_elements": (
            geometry_compact_to_global_scatter_elements
        ),
        "maximum_simultaneous_geometry_jw_length_bar_tensors": (
            maximum_simultaneous_geometry_jw_length_bar_tensors
        ),
        "maximum_native_length_bar_tensor_bytes": (
            maximum_native_length_bar_tensor_bytes
        ),
        "maximum_geometry_bridge_visible_tensor_bytes": (
            maximum_geometry_bridge_visible_tensor_bytes
        ),
        "staged_sparse_geometry_bridge_visible_peak_logical_tensor_bytes_upper_bound": (
            staged_sparse_geometry_bridge_upper_bound
        ),
        "staged_sparse_geometry_bridge_included_in_main_active_peak": (
            staged_full_geometry
            and staged_sparse_geometry_bridge_upper_bound > 0
        ),
        "staged_main_active_peak_formula": (
            "base_active_node_union+request_cpu_f64_geometry+step_accumulator+"
            "complete_one_block_sparse_geometry_bridge"
            if staged_full_geometry
            else "not_applicable"
        ),
        "fused_prepared_owned_logical_tensor_bytes": (
            fused_prepared_owned_logical_tensor_bytes
        ),
        "fused_prepared_owned_logical_tensor_bytes_upper_bound": (
            fused_prepared_owned_upper_bound
        ),
        "maximum_fused_prepared_owned_logical_tensor_bytes": (
            memory_policy.maximum_fused_prepared_owned_logical_tensor_bytes
        ),
        "fused_output_scratch_logical_tensor_bytes": (
            fused_output_scratch_logical_tensor_bytes
        ),
        "fused_compact_output_scratch_logical_tensor_bytes": (
            fused_compact_output_scratch_logical_tensor_bytes
        ),
        "fused_global_output_scratch_logical_tensor_bytes": (
            fused_global_output_scratch_logical_tensor_bytes
        ),
        "fused_output_scratch_logical_tensor_bytes_upper_bound": (
            fused_output_upper_bound
        ),
        "maximum_fused_output_scratch_logical_tensor_bytes": (
            memory_policy.maximum_fused_output_scratch_logical_tensor_bytes
        ),
        "fused_compact_output_scratch_logical_tensor_bytes_upper_bound": (
            fused_compact_output_upper_bound
        ),
        "fused_global_output_scratch_logical_tensor_bytes_upper_bound": (
            fused_global_output_upper_bound
        ),
        "fused_geometry_bridge_visible_tensor_bytes": (
            fused_geometry_bridge_visible_tensor_bytes
        ),
        "fused_geometry_bridge_visible_tensor_bytes_upper_bound": (
            fused_geometry_bridge_upper_bound
        ),
        "maximum_fused_geometry_bridge_visible_peak_logical_tensor_bytes": (
            memory_policy.maximum_fused_geometry_bridge_visible_peak_logical_tensor_bytes
        ),
        "fused_validation_status_tensor_bytes": (
            4 if fused_full_geometry else 0
        ),
        "fused_transaction_fence_call_count": (
            fused_transaction_fence_call_count
        ),
        "fused_post_accept_commit_fence_call_count": (
            fused_post_accept_commit_fence_call_count
        ),
        "fused_active_manifest_coverage_certified": (
            fused_active_manifest_certified
        ),
        "fused_length_cotangent_allocated": False,
        "fused_optimizer_commit_performed": False,
        "fused_compact_material_scatter_elements": (
            fused_compact_material_scatter_elements
        ),
        "fused_main_active_peak_formula": (
            "base_active_node_union+request_cpu_f64_geometry+"
            "step_accumulator+prepared_owned+compact_and_global_f32_output"
        ),
        "fused_geometry_bridge_peak_formula": "S*(6+C)*(4+8)",
        "fused_bridge_overlap_included_in_main_active_peak": (
            fused_geometry_bridge_upper_bound
            == fused_global_output_upper_bound + request_geometry_bar_bytes
        ),
        "fused_prepared_allocation_transient_peak_measured": False,
        "fused_output_allocator_peak_measured": False,
        "fused_bridge_allocator_peak_measured": False,
        "fused_global_geometry_finalization_scaling": (
            "O(global_site_count_per_request)"
        ),
        "fused_reverse_requested_frame_scaling": (
            "independent_of_requested_frame_count_for_fixed_compiled_artifact"
        ),
        "request_geometry_bar_tensor_bytes": request_geometry_bar_bytes,
        "step_accumulator_logical_tensor_bytes": accumulator.logical_tensor_bytes,
        "step_ray_bar_key_logical_bytes": 16 * len(accumulator.ray_bar_keys),
        "camera_ray_gradients_enabled": full_geometry and optimize_camera_rays,
        "fixed_camera_avoids_global_ray_bar": (
            full_geometry and not optimize_camera_rays
        ),
        "request_delta_logical_tensor_bytes": delta.logical_tensor_bytes,
        "request_delta_ray_bar_key_logical_bytes": 16 * len(delta.ray_bar_keys),
        "caller_bars_mutated_by_request": False,
        "request_returns_one_combined_uncommitted_delta": True,
        "step_accumulator_world_bound_not_sampler_bound": True,
        "step_accumulator_retains_frame_axis": False,
        "optimizer_authorization_requires_full_manifest_seal": True,
        "optimizer_authorization_is_point_in_time": True,
        "post_authorization_snapshot_mutation_zeroes_bars": False,
        "target_loader_is_arbitrary_callable": False,
        "target_loader_partial_failure_lifetime_certified": True,
        "target_loader_provenance": load_chunk_targets.provenance,
        "target_loader_generation_digest": load_chunk_targets.generation_digest,
        "target_loader_completed_load_count": (
            load_chunk_targets.completed_load_count
        ),
        "target_loader_failed_after_enqueue_count": (
            load_chunk_targets.failed_after_enqueue_count
        ),
        "target_loader_maximum_outstanding_lifetime_count": 1,
        "target_loader_retained_lifetime_count_after_request": 0,
        "target_loader_transfer_roots_released_only_after_completion_fence": True,
        "target_loader_lifetime_additional_logical_tensor_bytes": 0,
        "target_loader_lifetime_python_heap_bytes_measured": False,
        "target_loader_test_fault_enabled": (
            load_chunk_targets._test_fault is not None
        ),
        "target_loader_retained_closure_state_measured": True,
        "whole_pipeline_target_loader_memory_proven": False,
        "decoder_allocator_peak_measured": False,
        "sample_materialization_float64_scratch_measured": False,
        "sample_materialization_source_visible_logical_tensors_accounted": True,
        "requested_maximum_samples_per_launch": maximum_samples_per_launch,
        "effective_maximum_samples_per_launch": (
            effective_maximum_samples_per_launch
        ),
        "maximum_sample_materialization_logical_tensor_bytes": (
            memory_policy.maximum_sample_materialization_logical_tensor_bytes
        ),
        "peak_sample_materialization_logical_tensor_bytes_upper_bound": (
            peak_sample_materialization_bytes
        ),
        "peak_interpolation_evaluator_scratch_logical_tensor_bytes_upper_bound": (
            peak_interpolation_evaluator_scratch_bytes
        ),
        "maximum_interpolation_rows_per_subchunk": (
            maximum_interpolation_rows_per_subchunk
        ),
        "interpolation_evaluator_scratch_formula": (
            "4096+512*J+8*J^2+K_sub*(1024+512*J)"
        ),
        "sample_materialization_peak_formula": (
            "max(N*(8*J+12)+interpolation_scratch+16*K_sub,N*(16*J+32))"
        ),
        "whole_step_python_object_peak_measured": False,
        "geometry_committed_after_executor_seal": False,
        "structural_compile_track_count_during_request": 0,
        "cached_artifact_track_count": artifact.track_count,
        "peak_chunk_target_tensor_bytes": peak_chunk_target_bytes,
        "selected_pixel_read_mode": selected_pixel_read_mode,
        "selected_pixel_read_source_provenances": tuple(
            sorted(selected_pixel_read_source_provenances)
        ),
        "selected_pixel_read_call_count": selected_pixel_read_call_count,
        "mapped_selected_pixel_read_call_count": (
            mapped_selected_pixel_read_call_count
        ),
        "mapping_closed_before_return_count": (
            mapping_closed_before_return_count
        ),
        "all_selected_pixel_mappings_closed_before_return": (
            mapping_closed_before_return_count
            == mapped_selected_pixel_read_call_count
        ),
        "selected_pixel_read_acceptance_capable": (
            selected_pixel_read_acceptance_capable
        ),
        "direct_selected_pixel_observation_count": (
            direct_selected_pixel_observation_count
        ),
        "bounded_region_selected_pixel_observation_count": (
            bounded_region_selected_pixel_observation_count
        ),
        "full_frame_fallback_observation_count": (
            full_frame_fallback_observation_count
        ),
        "full_frame_target_materialization_count": decoded_frame_count,
        "bounded_region_target_materialization_count": (
            bounded_region_materialization_count
        ),
        "decoded_frame_count": decoded_frame_count,
        "decoded_frame_device_type": "cpu",
        "decoded_frame_mps_completion_fence_call_count": (
            decoded_frame_mps_completion_fence_call_count
        ),
        "peak_cpu_decoded_frame_tensor_bytes": peak_cpu_decoded_frame_bytes,
        "peak_bounded_region_materialization_tensor_bytes": (
            peak_bounded_region_materialization_bytes
        ),
        "peak_source_visible_target_read_logical_tensor_bytes_upper_bound": (
            peak_source_visible_target_read_bytes
        ),
        "peak_transient_mapped_address_space_bytes": (
            peak_transient_mapped_address_space_bytes
        ),
        "peak_requested_unique_mapped_page_count": (
            peak_requested_unique_mapped_page_count
        ),
        "peak_mapped_page_size_bytes": peak_mapped_page_size_bytes,
        "peak_requested_mapped_page_bytes_upper_bound": (
            peak_requested_mapped_page_bytes_upper_bound
        ),
        "cumulative_requested_mapped_page_count": (
            cumulative_requested_mapped_page_count
        ),
        "cumulative_requested_mapped_page_bytes_upper_bound": (
            cumulative_requested_mapped_page_bytes_upper_bound
        ),
        "peak_cpu_chunk_target_tensor_bytes": peak_cpu_chunk_target_bytes,
        "peak_device_chunk_target_tensor_bytes": peak_device_chunk_target_bytes,
        "peak_target_decode_bridge_logical_tensor_bytes": (
            peak_target_decode_bridge_bytes
        ),
        "target_frame_access_mode": (
            f"{selected_pixel_read_mode}_then_single_chunk_transfer"
        ),
        "target_source_decode_budget_enforced_before_allocation": True,
        "single_bounded_chunk_transfer_per_replay_chunk": True,
        "cpu_to_device_chunk_transfer_requested_non_blocking": False,
        "real_device_transfer_completion_verified": False,
        "maximum_simultaneously_decoded_target_frame_count": (
            1 if decoded_frame_count > 0 else 0
        ),
        "unique_frames_batched_for_target_decode": False,
        "peak_sample_launch_tensor_bytes": peak_sample_launch_bytes,
        "peak_sample_launch_node_count": peak_sample_launch_node_count,
        "peak_native_prepared_sample_scratch_tensor_bytes": (
            peak_native_prepared_sample_scratch_bytes
        ),
        "peak_public_sample_launch_logical_tensor_bytes": (
            peak_public_sample_launch_logical_bytes
        ),
        "native_prepared_sample_scratch_formula": "4*N+20",
        "native_prepared_sample_scratch_contract": "public_production_abi",
        "native_prepared_sample_public_tensor_scratch_accounted": True,
        "sample_preflight_includes_retained_cpu_transfer_source": True,
        "sample_preflight_includes_public_native_prepare_scratch": True,
        "native_driver_allocator_scratch_measured": False,
        "peak_chunk_dispatch_identity_logical_bytes": (
            peak_chunk_dispatch_identity_bytes
        ),
        "lane_resident_logical_tensor_bytes_upper_bound": lane_resident_bytes_upper_bound,
        "lane_two_phase_construction_predecessor_logical_tensor_bytes_upper_bound": (
            _lane_two_phase_construction_predecessor_upper_bound_bytes(artifact)
        ),
        "lane_two_phase_construction_predecessors_overlap_active_request": True,
        "active_request_logical_tensor_bytes_upper_bound": active_upper_bound,
        "reverse_lane_plus_active_logical_tensor_bytes_upper_bound": (
            reverse_lane_plus_active_upper_bound
        ),
        "reverse_lane_plus_active_policy_cap_sum": (
            memory_policy.maximum_lane_resident_logical_tensor_bytes
            + memory_policy.maximum_active_node_and_union_bar_tensor_bytes
        ),
        "reverse_lane_plus_active_is_allocator_peak": False,
        "persistent_frame_tensor_bytes": 0,
        "persistent_sample_tensor_bytes": 0,
        "persistent_target_tensor_bytes": 0,
        "persistent_prediction_tensor_bytes": 0,
        "retained_observation_count_after_request": 0,
        "total_pf_sample_work_is_linear": (
            selected_pixel_read_acceptance_capable
        ),
        "structural_node_word_work_invariance_requires_cross_row_verification": True,
        "full_geometry_vjp_integrated": full_geometry,
        "full_geometry_reverse_mode": full_geometry_reverse_mode,
        "production_trainer_integrated": False,
        "post_dedup_runtime_verified": False,
        "production_promotion_allowed": False,
        "native_runtime_verified": False,
        "allocator_peak_measured": False,
        **structural_accounting,
    }
    result_provisional = PaperKineticDenseCachedNativeRequestResult(
        source_generation_digest=source.generation_digest,
        request_generation_digest=request.generation_digest,
        artifact_generation_digest=artifact.generation_digest,
        lane_generation_digest=lane_generation_digest,
        receipt=receipt,
        telemetry=telemetry,
        delta=delta,
        loss_delta_f32=float(local_loss.detach().cpu().item()),
        accounting=accounting,
        generation_digest="",
        full_geometry_reverse_mode=full_geometry_reverse_mode,
        full_geometry_vjp_integrated=full_geometry,
        _seal=_RESULT_SEAL,
    )
    result = replace(
        result_provisional,
        generation_digest=_result_digest(result_provisional),
    )
    result.assert_current(source, request, artifact, session, accumulator)
    _register_pending_request_delta(accumulator, source, session, result.delta)
    return result


def _materialize_dense_sample_block(
    sampler: PaperKineticRowRaggedSampler,
    *,
    native_block_generation_digest: str,
    entries: tuple[tuple[int, PaperKineticObservation, PaperKineticRowBinding, float], ...],
    target_rgb_f32: torch.Tensor,
    global_loss_element_count: int,
    loss_normalization_id: str,
    maximum_materialization_logical_tensor_bytes: int,
) -> _DenseSampleMaterializationLease:
    native_block = next(
        block
        for bucket in sampler.lowering.buckets
        for block in bucket.blocks
        if block.generation_digest == native_block_generation_digest
    )
    sample_count = len(entries)
    materialization_plan = _sample_materialization_memory_plan(
        sample_count=sample_count,
        node_count=native_block.node_count,
        maximum_logical_tensor_bytes=maximum_materialization_logical_tensor_bytes,
    )
    sample_rows = torch.empty((sample_count,), dtype=torch.int32, device="cpu")
    weights_f64 = torch.empty(
        (sample_count, native_block.node_count),
        dtype=torch.float64,
        device="cpu",
    )
    flat_indices = torch.empty((sample_count,), dtype=torch.int64, device="cpu")
    exact_node_rows = 0
    dense_fallback_rows = 0
    linear_interactions = 0
    dense_fallback_interactions = 0
    by_row: dict[tuple[int, int], list[int]] = {}
    for sample_index, (_position, observation, row, _time) in enumerate(entries):
        sample_rows[sample_index] = row.native_local_row_index
        flat_indices[sample_index] = observation.observation_id
        by_row.setdefault(row.row_identity, []).append(sample_index)
    row_by_identity = {row.row_identity: row for row in sampler.rows}
    for identity, sample_indices in by_row.items():
        row = row_by_identity[identity]
        for start in range(
            0,
            len(sample_indices),
            materialization_plan.interpolation_rows_per_subchunk,
        ):
            subchunk_indices = sample_indices[
                start : start
                + materialization_plan.interpolation_rows_per_subchunk
            ]
            times = torch.tensor(
                [entries[index][3] for index in subchunk_indices],
                dtype=torch.float64,
                device="cpu",
            )
            sampled = row.program.charts[
                row.chart_index
            ].schedule.sample_to_node_weights(times)
            destination = torch.tensor(
                subchunk_indices,
                dtype=torch.int64,
                device="cpu",
            )
            weights_f64.index_copy_(0, destination, sampled.weights)
            exact_node_rows += sampled.exact_node_row_count
            dense_fallback_rows += sampled.dense_fallback_row_count
            linear_interactions += sampled.linear_weight_interactions
            dense_fallback_interactions += sampled.dense_fallback_interactions
            # The whole point of this inner bound is to release every
            # float64 evaluator tensor before constructing the next row block.
            del sampled, times, destination, subchunk_indices

    positions = torch.tensor(
        [entry[0] for entry in entries],
        dtype=torch.int64,
        device=target_rgb_f32.device,
    )
    weights = weights_f64.to(
        device=target_rgb_f32.device,
        dtype=torch.float32,
    ).contiguous()
    targets = target_rgb_f32.index_select(0, positions).contiguous()
    dispatch_digest = _digest_parts(
        REQUEST_PROVENANCE,
        "chunk-dispatch",
        tuple(
            (
                position,
                observation.sample_identity,
                row.global_row_index,
                row.native_local_row_index,
            )
            for position, observation, row, _time in entries
        ),
    )
    sample_block = seal_paper_kinetic_row_ragged_sample_block(
        sampler,
        native_block_generation_digest=native_block_generation_digest,
        sample_row_i32=sample_rows,
        sample_to_node_f32=weights,
        target_rgb_f32=targets,
        flat_sample_index_i64=flat_indices,
        global_loss_element_count=global_loss_element_count,
        loss_normalization_id=loss_normalization_id,
        exact_node_row_count=exact_node_rows,
        dense_fallback_row_count=dense_fallback_rows,
        linear_weight_interactions=linear_interactions,
        dense_fallback_interactions=dense_fallback_interactions,
        dispatch_generation_digest=dispatch_digest,
    )
    lease = _DenseSampleMaterializationLease(
        sample_block=sample_block,
        weights_source_f64=weights_f64,
        positions_i64=positions,
        chunk_target_rgb_f32=target_rgb_f32,
        sample_block_identity=id(sample_block),
        weights_source_signature=_tensor_signature(weights_f64),
        positions_signature=_tensor_signature(positions),
        chunk_target_signature=_tensor_signature(target_rgb_f32),
        _seal=_SAMPLE_MATERIALIZATION_LEASE_SEAL,
    )
    lease.assert_retained()
    return lease


def _request_observation_count(
    source: PaperKineticReplayableDenseObservationSource,
    request: PaperKineticDenseObservationTrackRequest,
) -> int:
    selected_frame_count = sum(sample.view_index == request.view_index for sample in source.batch.samples)
    count = selected_frame_count * len(request.track_ids)
    if count < 1:
        raise ValueError("dense cached request has no selected observations")
    return count


def _chunk_frame_decode_cardinality(
    chunk: PaperKineticDenseObservationChunk,
) -> tuple[int, int]:
    counts: dict[tuple[int, int], int] = {}
    for observation in chunk.observations:
        key = (observation.view_index, observation.frame_index)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        raise ValueError("dense target chunk contains no decoded frame")
    return len(counts), max(counts.values())


def _legacy_full_frame_pixel_read_peak_logical_tensor_bytes(
    *,
    chunk_observation_count: int,
    maximum_frame_observation_count: int,
    full_frame_tensor_bytes: int,
) -> int:
    """Public CPU-tensor peak for the compatibility full-frame fallback."""
    cpu_chunk_bytes = chunk_observation_count * 3 * 4
    per_frame_index_and_gather_bytes = maximum_frame_observation_count * (
        4 * 8 + 3 * 4
    )
    return (
        cpu_chunk_bytes
        + full_frame_tensor_bytes
        + per_frame_index_and_gather_bytes
    )


def _target_pixel_read_bridge_peak_logical_tensor_bytes(
    *,
    source_visible_read_peak_logical_tensor_bytes: int,
    chunk_target_tensor_bytes: int,
    target_device: torch.device,
) -> int:
    """Bound a sealed CPU pixel read overlapping one blocking device copy."""

    transfer_peak = chunk_target_tensor_bytes * (
        2 if target_device.type != "cpu" else 1
    )
    return max(source_visible_read_peak_logical_tensor_bytes, transfer_peak)


def _interpolation_evaluator_scratch_logical_tensor_bytes_upper_bound(
    *,
    row_count: int,
    node_count: int,
) -> int:
    """Conservative source-visible bound for one float64 weight subchunk.

    The rank-squared coefficient covers the boolean validation temporary for
    the stored ``[J,J]`` fit matrix.  The row-node coefficient covers 64
    float64-equivalent ``[K,J]`` tensors.  The current implementation uses
    fewer even on the dense fallback, but this intentionally leaves room for
    expression temporaries while source and allocator peaks remain distinct
    claims.
    """

    _require_positive_int(row_count, name="interpolation row_count")
    _require_positive_int(node_count, name="interpolation node_count")
    return (
        _INTERPOLATION_EVALUATOR_FIXED_LOGICAL_BYTES
        + _INTERPOLATION_EVALUATOR_PER_NODE_LOGICAL_BYTES * node_count
        + _INTERPOLATION_EVALUATOR_PER_NODE_SQUARED_LOGICAL_BYTES
        * node_count
        * node_count
        + row_count
        * (
            _INTERPOLATION_EVALUATOR_PER_ROW_LOGICAL_BYTES
            + _INTERPOLATION_EVALUATOR_PER_ROW_NODE_LOGICAL_BYTES * node_count
        )
    )


def _maximum_materialized_samples_for_budget(
    *,
    node_count: int,
    maximum_logical_tensor_bytes: int,
) -> int:
    """Largest sample block for which at least one interpolation row fits."""

    _require_positive_int(node_count, name="materialization node_count")
    _require_positive_int(
        maximum_logical_tensor_bytes,
        name="maximum_sample_materialization_logical_tensor_bytes",
    )
    one_row_scratch = (
        _interpolation_evaluator_scratch_logical_tensor_bytes_upper_bound(
            row_count=1,
            node_count=node_count,
        )
        + 16
    )
    # During interpolation the complete CPU float64 destination and CPU row /
    # identity vectors remain live: N*(8*J + 12).  During transfer we
    # conservatively allow CPU f64 weights, two device f32 weight buffers,
    # device positions/targets, and the returned row/identity vectors:
    # N*(16*J + 32).  Both bounds exclude the separately-accounted target
    # chunk and the subsequent public native-prepare scratch.
    interpolation_capacity = (
        maximum_logical_tensor_bytes - one_row_scratch
    ) // (8 * node_count + 12)
    transfer_capacity = maximum_logical_tensor_bytes // (16 * node_count + 32)
    return max(0, min(interpolation_capacity, transfer_capacity))


def _sample_materialization_memory_plan(
    *,
    sample_count: int,
    node_count: int,
    maximum_logical_tensor_bytes: int,
) -> _DenseSampleMaterializationMemoryPlan:
    """Choose a bounded row subchunk and prove its public logical-tensor peak."""

    _require_positive_int(sample_count, name="materialization sample_count")
    maximum_samples = _maximum_materialized_samples_for_budget(
        node_count=node_count,
        maximum_logical_tensor_bytes=maximum_logical_tensor_bytes,
    )
    if sample_count > maximum_samples:
        raise MemoryError(
            "dense sample materialization output exceeds its explicit logical-tensor budget"
        )
    persistent_cpu_bytes = sample_count * (8 * node_count + 12)
    per_subchunk_row_bytes = (
        _INTERPOLATION_EVALUATOR_PER_ROW_LOGICAL_BYTES
        + _INTERPOLATION_EVALUATOR_PER_ROW_NODE_LOGICAL_BYTES * node_count
        + 16
    )
    fixed_subchunk_bytes = (
        _INTERPOLATION_EVALUATOR_FIXED_LOGICAL_BYTES
        + _INTERPOLATION_EVALUATOR_PER_NODE_LOGICAL_BYTES * node_count
        + _INTERPOLATION_EVALUATOR_PER_NODE_SQUARED_LOGICAL_BYTES
        * node_count
        * node_count
    )
    available_for_subchunk = (
        maximum_logical_tensor_bytes
        - persistent_cpu_bytes
        - fixed_subchunk_bytes
    )
    interpolation_rows = min(
        sample_count,
        max(0, available_for_subchunk // per_subchunk_row_bytes),
    )
    if interpolation_rows < 1:
        raise MemoryError(
            "dense sample interpolation cannot fit one row in its explicit logical-tensor budget"
        )
    evaluator_scratch = (
        _interpolation_evaluator_scratch_logical_tensor_bytes_upper_bound(
            row_count=interpolation_rows,
            node_count=node_count,
        )
    )
    interpolation_peak = (
        persistent_cpu_bytes
        + evaluator_scratch
        + 16 * interpolation_rows
    )
    transfer_peak = sample_count * (16 * node_count + 32)
    peak = max(interpolation_peak, transfer_peak)
    if peak > maximum_logical_tensor_bytes:
        raise ArithmeticError("dense sample materialization plan exceeded its own budget")
    return _DenseSampleMaterializationMemoryPlan(
        sample_count=sample_count,
        node_count=node_count,
        maximum_logical_tensor_bytes=maximum_logical_tensor_bytes,
        interpolation_rows_per_subchunk=interpolation_rows,
        interpolation_evaluator_scratch_logical_tensor_bytes_upper_bound=(
            evaluator_scratch
        ),
        materialization_peak_logical_tensor_bytes_upper_bound=peak,
    )


def _lane_resident_upper_bound_bytes(
    artifact: PaperKineticCompiledCpuArtifact,
) -> int:
    blocks = tuple(block for bucket in artifact.sampler.lowering.buckets for block in bucket.blocks)
    union_count = len({source_id for block in blocks for source_id in block.source_site_ids})
    mapping_tensor_bytes = 8 * union_count + 8 * sum(
        len(block.source_site_ids) for block in blocks
    )
    construction_predecessor_bytes = mapping_tensor_bytes + 4 * len(blocks)
    runtime_copy_upper_bound = sum(
        8 * len(block.source_site_ids)
        + 4 * (block.row_count + 1)
        + 4 * block.word_count
        + 4 * block.node_count * block.word_count
        + 4 * 4
        + 4
        for block in blocks
    )
    return (
        artifact.accounted_resident_bytes
        # Two-phase construction retains the fresh CPU union/map predecessors,
        # one CPU epsilon per runtime, and the transferred destination tensors
        # together until the lane release fence. Count both mapping domains
        # even when CPU execution aliases them.
        + construction_predecessor_bytes
        + mapping_tensor_bytes
        + artifact.sampler.lowering.total_materialized_block_tensor_bytes
        + runtime_copy_upper_bound
    )


def _lane_two_phase_construction_predecessor_upper_bound_bytes(
    artifact: PaperKineticCompiledCpuArtifact,
) -> int:
    blocks = tuple(
        block
        for bucket in artifact.sampler.lowering.buckets
        for block in bucket.blocks
    )
    union_count = len(
        {source_id for block in blocks for source_id in block.source_site_ids}
    )
    return (
        8 * union_count
        + 8 * sum(len(block.source_site_ids) for block in blocks)
        + 4 * len(blocks)
    )


def _active_state_upper_bound_bytes(
    artifact: PaperKineticCompiledCpuArtifact,
    *,
    include_full_geometry: bool,
) -> int:
    blocks = tuple(block for bucket in artifact.sampler.lowering.buckets for block in bucket.blocks)
    union_count = len({source_id for block in blocks for source_id in block.source_site_ids})
    base = (
        sum(32 * block.row_count * block.node_count + 16 * len(block.source_site_ids) for block in blocks)
        + 16 * union_count
        + 16 * max(len(block.source_site_ids) for block in blocks)
        + 4 * len(blocks)
        + 4
        + 12
    )
    if not include_full_geometry:
        return base
    maximum_length_bar_bytes = max(
        4 * block.node_count * block.word_count for block in blocks
    )
    return base + maximum_length_bar_bytes


def _staged_sparse_geometry_bridge_visible_peak_logical_tensor_bytes_upper_bound(
    artifact: PaperKineticCompiledCpuArtifact,
    *,
    include_ray_gradients: bool,
) -> int:
    """Maximum complete one-block staged reduction phase before native work."""

    blocks = tuple(
        block
        for bucket in artifact.sampler.lowering.buckets
        for block in bucket.blocks
    )
    if not blocks:
        raise ValueError("dense cached artifact has no staged geometry block")
    return max(
        preflight_kinetic_native_equal_rank_sparse_geometry_reduction_memory(
            artifact.sampler,
            block_generation_digest=block.generation_digest,
            include_ray_gradients=include_ray_gradients,
        ).bridge_visible_peak_logical_tensor_bytes
        for block in blocks
    )


def _fused_prepared_owned_logical_tensor_bytes_upper_bound(
    artifact: PaperKineticCompiledCpuArtifact,
) -> int:
    """Exact source-visible owned payload if every eligible block is active."""

    blocks = tuple(
        block
        for bucket in artifact.sampler.lowering.buckets
        for block in bucket.blocks
    )
    weight_count = int(
        _shared_kinetic_sites(artifact.sampler).weight_coefficients.shape[1]
    )
    # Per block: row node times [R,J], near/far [R,2], affine ray [R,12],
    # compact position/velocity/weight [S_b,6+C], and 6 int32 + 7 float32
    # configuration scalars. Runtime topology/world tensors are aliases.
    return sum(
        4
        * (
            block.row_count * (block.node_count + 14)
            + len(block.source_site_ids) * (6 + weight_count)
            + 13
        )
        for block in blocks
    )


def _fused_output_scratch_logical_tensor_bytes_upper_bound(
    artifact: PaperKineticCompiledCpuArtifact,
) -> tuple[int, int, int]:
    """Compact, global, and total float32 scratch for all eligible blocks."""

    blocks = tuple(
        block
        for bucket in artifact.sampler.lowering.buckets
        for block in bucket.blocks
    )
    sites = _shared_kinetic_sites(artifact.sampler)
    weight_count = int(sites.weight_coefficients.shape[1])
    compact = 16 * sum(len(block.source_site_ids) for block in blocks)
    global_geometry = 4 * sites.site_count * (6 + weight_count)
    return compact, global_geometry, compact + global_geometry


def _fused_geometry_bridge_visible_peak_logical_tensor_bytes(
    sampler: PaperKineticRowRaggedSampler,
) -> int:
    """Bound simultaneous device-f32 sources and CPU-f64 destinations."""

    sites = _shared_kinetic_sites(sampler)
    element_count = (
        sites.positions0.numel()
        + sites.velocities.numel()
        + sites.weight_coefficients.numel()
    )
    return element_count * (4 + 8)


def _bridge_fused_global_geometry_bars_to_cpu_f64(
    transaction_result: Any,
    sampler: PaperKineticRowRaggedSampler,
    *,
    maximum_bridge_visible_peak_logical_tensor_bytes: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], int]:
    """Convert one accepted fixed-camera fused result to the public CPU ABI."""

    _require_positive_int(
        maximum_bridge_visible_peak_logical_tensor_bytes,
        name="maximum_fused_geometry_bridge_visible_peak_logical_tensor_bytes",
    )
    sites = _shared_kinetic_sites(sampler)
    source_bars = (
        transaction_result.grad_global_positions0_f32,
        transaction_result.grad_global_velocities_f32,
        transaction_result.grad_global_weight_coefficients_f32,
    )
    device = source_bars[0].device
    for tensor, name, shape in zip(
        source_bars,
        (
            "fused grad_global_positions0_f32",
            "fused grad_global_velocities_f32",
            "fused grad_global_weight_coefficients_f32",
        ),
        (
            tuple(sites.positions0.shape),
            tuple(sites.velocities.shape),
            tuple(sites.weight_coefficients.shape),
        ),
        strict=True,
    ):
        _require_tensor(
            tensor,
            name=name,
            device=device,
            dtype=torch.float32,
            shape=shape,
        )
        if tensor.requires_grad:
            raise ValueError("fused global geometry bars must be explicit bars")
    _require_distinct_storage(*source_bars)
    bridge_bytes = _fused_geometry_bridge_visible_peak_logical_tensor_bytes(
        sampler
    )
    if bridge_bytes > maximum_bridge_visible_peak_logical_tensor_bytes:
        raise MemoryError(
            "fused global geometry bridge exceeds its pre-allocation byte cap"
        )
    cpu_bars = tuple(
        tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()
        for tensor in source_bars
    )
    for tensor, name, shape in zip(
        cpu_bars,
        (
            "fused bridged grad_positions0_f64",
            "fused bridged grad_velocities_f64",
            "fused bridged grad_weight_coefficients_f64",
        ),
        (
            tuple(sites.positions0.shape),
            tuple(sites.velocities.shape),
            tuple(sites.weight_coefficients.shape),
        ),
        strict=True,
    ):
        _require_cpu_f64_tensor(tensor, name=name, shape=shape)
    _require_distinct_storage(*source_bars, *cpu_bars)
    return cpu_bars, bridge_bytes


def _accumulate_fused_compact_material_and_loss(
    spatial_bundle: PaperKineticUnionLocalSpatialBundle,
    active: Mapping[str, _ActiveBlock],
    *,
    active_block_generation_ids: Sequence[str],
    grad_compact_site_rgba_f32_by_block: Sequence[torch.Tensor],
    local_union_bar: torch.Tensor,
    local_loss: torch.Tensor,
) -> int:
    """Scatter every accepted compact bar and sum each exact block loss once."""

    block_ids = tuple(active_block_generation_ids)
    compact_bars = tuple(grad_compact_site_rgba_f32_by_block)
    if (
        not block_ids
        or len(block_ids) != len(compact_bars)
        or len(set(block_ids)) != len(block_ids)
        or set(block_ids) != set(active)
    ):
        raise ValueError(
            "fused compact outputs do not cover the exact request active manifest"
        )
    _require_tensor(
        local_union_bar,
        name="fused request local_union_bar",
        device=local_union_bar.device,
        dtype=torch.float32,
        shape=(spatial_bundle.union_site_count, 4),
    )
    _require_tensor(
        local_loss,
        name="fused request local_loss",
        device=local_union_bar.device,
        dtype=torch.float32,
        shape=(1,),
    )
    scatter_elements = 0
    for block_id, compact_bar in zip(block_ids, compact_bars, strict=True):
        binding = spatial_bundle.binding_for_digest(block_id)
        block_state = active[block_id]
        _require_tensor(
            compact_bar,
            name="fused compact material bar",
            device=local_union_bar.device,
            dtype=torch.float32,
            shape=(binding.compact_site_count, 4),
        )
        _require_tensor(
            block_state.loss_f32,
            name="fused block loss",
            device=local_loss.device,
            dtype=torch.float32,
            shape=(1,),
        )
        local_union_bar.index_add_(
            0,
            binding.compact_to_union_i64,
            compact_bar,
        )
        local_loss.add_(block_state.loss_f32)
        scatter_elements += int(compact_bar.numel())
    return scatter_elements


def _request_structural_accounting(
    artifact: PaperKineticCompiledCpuArtifact,
    telemetry: KineticNativeMaterialStepTelemetry,
) -> dict[str, Any]:
    """Report runtime-used and compiler-owned structural work without F proxies.

    The structural signature includes view/pixel identities, compiled affine
    ray coefficients, program semantic/geometry digests, equal-rank lowering,
    the fixed physical interval, actual active blocks, ranks, and ordered-word
    work.  It deliberately excludes artifact/provider generation ids, camera
    record counts, frame-time sample lists, and requested ``F``.  Equality
    therefore tests denser sampling of one fixed interval; it says nothing
    about the extra events/charts that a longer physical duration may require.
    """

    artifact.assert_warm_current()
    telemetry.assert_current()
    programs: dict[int, Any] = {}
    for row in artifact.sampler.rows:
        previous = programs.setdefault(row.track_id, row.program)
        if previous is not row.program:
            raise ValueError("one structural track retained multiple programs")
    if tuple(sorted(programs)) != artifact.track_ids:
        raise ArithmeticError("structural accounting lost artifact track coverage")
    compiler_provenances = {
        program.binding.compiler_provenance for program in programs.values()
    }
    if len(compiler_provenances) != 1:
        raise ValueError("one artifact mixed structural compiler provenances")
    physical_intervals = {
        (
            repr(program.binding.program.t_min),
            repr(program.binding.program.t_max),
            repr(program.binding.program.near),
            repr(program.binding.program.far),
        )
        for program in programs.values()
    }
    if len(physical_intervals) != 1:
        raise ValueError("one artifact mixed physical ray-time intervals")
    compiled_camera_path_signature = _digest_parts(
        "paper-kinetic-compiled-affine-camera-path-v1",
        artifact.key.view_index,
        tuple(
            (
                track_id,
                _cpu_tensor_content_digest(
                    programs[track_id].binding.ray_coefficients
                ),
            )
            for track_id in artifact.track_ids
        ),
    )
    blocks = tuple(
        block
        for bucket in artifact.sampler.lowering.buckets
        for block in bucket.blocks
    )
    block_by_digest = {block.generation_digest: block for block in blocks}
    if len(block_by_digest) != len(blocks):
        raise ValueError("artifact structural blocks have duplicate identities")
    try:
        active_blocks = tuple(
            block_by_digest[block.native_block_generation_digest]
            for block in telemetry.blocks
        )
    except KeyError as error:
        raise ValueError("native telemetry references a foreign structural block") from error
    if len(active_blocks) != telemetry.active_native_block_count:
        raise ArithmeticError("native telemetry changed active block coverage")
    event_count = sum(
        len(program.binding.program.active_event_guards)
        for program in programs.values()
    )
    fallback_count = sum(
        program.unresolved_algebraic_endpoint_count
        for program in programs.values()
    )
    chart_node_ranks = tuple(sorted({block.node_count for block in active_blocks}))
    if not chart_node_ranks:
        raise ArithmeticError("dense request executed no structural node rank")
    report: dict[str, Any] = {
        "compiler_provenance": next(iter(compiler_provenances)),
        "physical_interval_digest": _digest_parts(
            "paper-kinetic-physical-ray-time-interval-v1",
            next(iter(physical_intervals)),
        ),
        "compiled_camera_path_signature_sha256": (
            compiled_camera_path_signature
        ),
        "event_count": event_count,
        "track_chart_row_count": artifact.sampler.row_count,
        "word_entry_count": sum(block.word_count for block in blocks),
        "fallback_count": fallback_count,
        "active_native_block_count": telemetry.active_native_block_count,
        "node_forward_launch_count": telemetry.native_node_forward_launch_count,
        # Dispatch geometry is one thread per (row,node), but each thread scans
        # the full CSR owner word for that row.  Report both quantities so
        # forward world work is not understated by the mean word length.
        "node_forward_thread_count": sum(
            block.row_count * block.node_count for block in active_blocks
        ),
        "node_forward_interaction_count": sum(
            block.word_count * block.node_count for block in active_blocks
        ),
        "caller_preallocated_node_forward_output_bytes": 16
        * sum(block.row_count * block.node_count for block in active_blocks),
        "material_word_vjp_interaction_count": sum(
            block.word_count * block.node_count for block in active_blocks
        ),
        "active_material_exact_model_bytes": _active_state_upper_bound_bytes(
            artifact,
            include_full_geometry=False,
        ),
        "chart_node_ranks": chart_node_ranks,
    }
    report["artifact_structural_signature_sha256"] = _digest_parts(
        "paper-kinetic-request-structural-accounting-v1",
        artifact.key.view_index,
        artifact.track_ids,
        tuple(
            programs[track_id].generation_digest
            for track_id in artifact.track_ids
        ),
        artifact.sampler.lowering.generation_digest,
        tuple(sorted(report.items())),
    )
    return report


def _cpu_tensor_content_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(repr((tuple(value.shape), str(value.dtype))).encode("utf-8"))
    digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def _request_geometry_bar_bytes(
    sampler: PaperKineticRowRaggedSampler,
    request: PaperKineticDenseObservationTrackRequest,
    *,
    include_ray_gradients: bool,
) -> int:
    if not isinstance(include_ray_gradients, bool):
        raise TypeError("include_ray_gradients must be bool")
    sites = _shared_kinetic_sites(sampler)
    return 8 * (
        sites.positions0.numel()
        + sites.velocities.numel()
        + sites.weight_coefficients.numel()
        + (12 * len(request.track_ids) if include_ray_gradients else 0)
    )


def _validate_request_inputs(
    source: PaperKineticReplayableDenseObservationSource,
    session: PaperKineticDenseObservationReplaySession,
    request: PaperKineticDenseObservationTrackRequest,
    artifact: PaperKineticCompiledCpuArtifact,
    accumulator: PaperKineticDenseStepGradientAccumulator,
    *,
    step_generation_id: str,
    loss_normalization_id: str,
    material_generation_id: str,
    background_generation_id: str,
    global_site_rgba_f32: torch.Tensor,
    background_rgb_f32: torch.Tensor,
    backend_provenance: str,
    maximum_samples_per_launch: int,
    memory_policy: PaperKineticDenseCachedNativeMemoryPolicy,
    load_chunk_targets: PaperKineticDenseChunkTargetLoader,
    device_completion_fence: Callable[[], None],
    device_completion_fence_provenance: str,
    full_geometry_reverse_mode: str,
    cone_tolerance: float,
) -> None:
    if not isinstance(source, PaperKineticReplayableDenseObservationSource):
        raise TypeError("source must be a replayable dense observation source")
    if not isinstance(session, PaperKineticDenseObservationReplaySession):
        raise TypeError("session must be a dense observation replay session")
    source.assert_warm_current()
    session.assert_current()
    request.assert_current(source)
    artifact.assert_warm_reusable_with_provider(source.provider)
    if not isinstance(accumulator, PaperKineticDenseStepGradientAccumulator):
        raise TypeError("dense cached request requires a step accumulator")
    accumulator.assert_current(source, session)
    if session.source is not source:
        raise ValueError("dense cached request session belongs to another source")
    if session.poisoned or session.sealed or session._active_request:
        raise ValueError("dense cached request requires an open replay session")
    if artifact.key.view_index != request.view_index or artifact.track_ids != request.track_ids:
        raise ValueError("dense cached artifact and request differ")
    if accumulator.poisoned or accumulator.sealed:
        raise ValueError("dense cached request step accumulator is not open")
    if accumulator.pending_delta_generation_digest:
        raise ValueError("dense step must consume its pending request before replaying another")
    if (
        accumulator.consumed_request_count != session.request_count
        or accumulator.consumed_observation_count != session.emitted_observation_count
        or accumulator.step_generation_id != step_generation_id
        or accumulator.loss_normalization_id != loss_normalization_id
        or accumulator.material_generation_id != material_generation_id
        or accumulator.background_generation_id != background_generation_id
        or _shared_kinetic_sites(artifact.sampler) is not source.provider.world.sites
    ):
        raise ValueError("dense request and step accumulator provenance/cursor differ")
    for name, value in (
        ("step_generation_id", step_generation_id),
        ("loss_normalization_id", loss_normalization_id),
        ("material_generation_id", material_generation_id),
        ("background_generation_id", background_generation_id),
        ("backend_provenance", backend_provenance),
        ("device_completion_fence_provenance", device_completion_fence_provenance),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be nonempty")
    _require_positive_int(maximum_samples_per_launch, name="maximum_samples_per_launch")
    if not isinstance(memory_policy, PaperKineticDenseCachedNativeMemoryPolicy):
        raise TypeError("dense cached request requires its memory policy")
    memory_policy.assert_valid()
    if full_geometry_reverse_mode not in _FULL_GEOMETRY_REVERSE_MODES:
        raise ValueError(
            "full_geometry_reverse_mode must be staged_sparse or fused_direct_v1"
        )
    if (
        full_geometry_reverse_mode == FUSED_DIRECT_V1_FULL_GEOMETRY_REVERSE
        and (not accumulator.full_geometry or accumulator.optimize_camera_rays)
    ):
        raise ValueError(
            "fused_direct_v1 requires fixed-camera full geometry with no ray bars"
        )
    if not isinstance(
        load_chunk_targets,
        PaperKineticDenseChunkTargetLoader,
    ) or not callable(device_completion_fence):
        raise TypeError(
            "dense cached replay requires its sealed target loader and a callable fence"
        )
    if not math.isfinite(cone_tolerance) or cone_tolerance <= 0.0:
        raise ValueError("cone_tolerance must be finite and positive")
    device = accumulator.grad_site_rgba_f32.device
    load_chunk_targets.assert_current(source, request, device=device)
    if (
        load_chunk_targets._active_lifetime is not None
        or load_chunk_targets.completed_load_count != 0
        or load_chunk_targets.failed_after_enqueue_count != 0
    ):
        raise ValueError("dense cached request requires a fresh sealed target loader")
    if (
        load_chunk_targets.maximum_decoded_frame_scratch_tensor_bytes
        != memory_policy.maximum_decoded_frame_scratch_tensor_bytes
        or load_chunk_targets.maximum_chunk_target_tensor_bytes
        != memory_policy.maximum_chunk_target_tensor_bytes
        or load_chunk_targets.maximum_target_decode_bridge_peak_logical_tensor_bytes
        != memory_policy.maximum_target_decode_bridge_peak_logical_tensor_bytes
    ):
        raise ValueError("dense sealed target loader changed the request memory policy")
    _require_tensor(
        global_site_rgba_f32,
        name="global_site_rgba_f32",
        device=device,
        dtype=torch.float32,
        shape=(source.provider.world.site_count, 4),
    )
    _require_tensor(
        background_rgb_f32,
        name="background_rgb_f32",
        device=device,
        dtype=torch.float32,
        shape=(3,),
    )
    if any(
        tensor.requires_grad
        for tensor in (global_site_rgba_f32, background_rgb_f32)
    ):
        raise ValueError("dense cached request owns explicit bars and forbids autograd tensors")
    if (
        accumulator.material_tensor_identity != id(global_site_rgba_f32)
        or accumulator.material_tensor_signature
        != _tensor_signature(global_site_rgba_f32)
        or accumulator.background_tensor_identity != id(background_rgb_f32)
        or accumulator.background_tensor_signature
        != _tensor_signature(background_rgb_f32)
    ):
        raise ValueError(
            "dense cached request changed the step material/background snapshot"
        )
    if (
        len({id(tensor) for tensor in (global_site_rgba_f32, background_rgb_f32)})
        != 2
    ):
        raise ValueError("dense cached request mutable/read-only tensors must not alias")


def _register_pending_request_delta(
    accumulator: PaperKineticDenseStepGradientAccumulator,
    source: PaperKineticReplayableDenseObservationSource,
    session: PaperKineticDenseObservationReplaySession,
    delta: PaperKineticDenseRequestGradientDelta,
) -> None:
    accumulator.assert_current(source, session)
    if (
        accumulator.poisoned
        or accumulator.sealed
        or accumulator.pending_delta_generation_digest
        or delta.consumed
        or delta.receipt.session_request_count_before
        != accumulator.consumed_request_count
        or delta.receipt.session_emitted_observation_count_before
        != accumulator.consumed_observation_count
    ):
        _poison_dense_step_accumulator(accumulator)
        raise ValueError("dense step cannot register an out-of-order request delta")
    accumulator.pending_request_generation_digest = delta.request_generation_digest
    accumulator.pending_delta_generation_digest = delta.generation_digest
    accumulator.tensor_signatures = tuple(
        _tensor_signature(tensor) for tensor in accumulator._tensors()
    )
    accumulator.generation_digest = _step_accumulator_digest(accumulator)
    accumulator.assert_current(source, session)


def _release_consumed_request_delta(
    delta: PaperKineticDenseRequestGradientDelta,
    accumulator_generation_digest: str,
) -> None:
    if delta.consumed or not accumulator_generation_digest.strip():
        raise ValueError("dense request delta release changed")
    delta.source_site_ids_i64 = None
    delta.grad_union_site_rgba_f32 = None
    delta.loss_f32 = None
    delta.grad_positions0_f64 = None
    delta.grad_velocities_f64 = None
    delta.grad_weight_coefficients_f64 = None
    delta.grad_track_ray_coefficients_f64 = None
    delta.consumed = True
    delta.consumed_by_accumulator_generation_digest = (
        accumulator_generation_digest
    )


def _invalidate_dense_step_accumulator_without_tensor_mutation(
    accumulator: PaperKineticDenseStepGradientAccumulator,
) -> None:
    """Poison an unfenced device commit without enqueueing more device work."""

    if not isinstance(accumulator, PaperKineticDenseStepGradientAccumulator):
        return
    accumulator.poisoned = True
    accumulator.sealed = False
    accumulator.optimizer_authorized = False
    accumulator.tensor_signatures = tuple(
        _tensor_signature(tensor) for tensor in accumulator._tensors()
    )
    accumulator.generation_digest = _step_accumulator_digest(accumulator)


def _quarantine_dense_async_failure(
    accumulator: PaperKineticDenseStepGradientAccumulator,
    *,
    stage: str,
    original_error: BaseException,
    original_traceback: Any,
    cleanup_fence_error: BaseException,
    retained_references: tuple[tuple[str, Any], ...],
    device_completion_fence_provenance: str,
) -> None:
    """Retain unsafe lifetime roots on the poisoned world-bound accumulator."""

    if accumulator._async_failure_quarantine is not None:
        return
    retained = tuple(
        (role, reference)
        for role, reference in retained_references
        if reference is not None
    )
    provisional = _DenseAsyncFailureQuarantine(
        stage=stage,
        original_error=original_error,
        original_traceback=original_traceback,
        cleanup_fence_error=cleanup_fence_error,
        retained_reference_roles=tuple(role for role, _ in retained),
        retained_references=tuple(reference for _, reference in retained),
        device_completion_fence_provenance=(
            device_completion_fence_provenance
        ),
        generation_digest="",
    )
    quarantine = replace(
        provisional,
        generation_digest=_async_failure_quarantine_digest(provisional),
    )
    accumulator._async_failure_quarantine = quarantine
    _invalidate_dense_step_accumulator_without_tensor_mutation(accumulator)
    quarantine.assert_current()


@torch.no_grad()
def _poison_dense_step_accumulator(
    accumulator: PaperKineticDenseStepGradientAccumulator,
) -> None:
    if not isinstance(accumulator, PaperKineticDenseStepGradientAccumulator):
        return
    for tensor in accumulator._tensors():
        tensor.zero_()
    accumulator.pending_request_generation_digest = ""
    accumulator.pending_delta_generation_digest = ""
    accumulator.poisoned = True
    accumulator.sealed = False
    accumulator.optimizer_authorized = False
    accumulator.tensor_signatures = tuple(
        _tensor_signature(tensor) for tensor in accumulator._tensors()
    )
    accumulator.generation_digest = _step_accumulator_digest(accumulator)


def _shared_kinetic_sites(sampler: PaperKineticRowRaggedSampler):
    if not sampler.rows:
        raise ValueError("kinetic sampler has no rows")
    sites = sampler.rows[0].program.binding.sites
    if any(row.program.binding.sites is not sites for row in sampler.rows):
        raise ValueError("one dense geometry request cannot mix kinetic site tables")
    return sites


def _expected_step_ray_bar_keys(
    source: PaperKineticReplayableDenseObservationSource,
) -> tuple[tuple[int, int], ...]:
    views = tuple(span[0] for span in source._view_position_spans)
    return tuple(
        (view_index, track_id)
        for view_index in views
        for track_id in range(source.image_pixel_count)
    )


def _async_failure_quarantine_digest(
    quarantine: _DenseAsyncFailureQuarantine,
) -> str:
    return _digest_parts(
        STEP_ACCUMULATOR_PROVENANCE,
        "async-failure-quarantine-v1",
        quarantine.stage,
        type(quarantine.original_error).__qualname__,
        str(quarantine.original_error),
        id(quarantine.original_traceback),
        type(quarantine.cleanup_fence_error).__qualname__,
        str(quarantine.cleanup_fence_error),
        quarantine.retained_reference_roles,
        tuple(id(reference) for reference in quarantine.retained_references),
        quarantine.device_completion_fence_provenance,
        quarantine.restart_required,
    )


def _step_accumulator_digest(
    accumulator: PaperKineticDenseStepGradientAccumulator,
) -> str:
    return _digest_parts(
        STEP_ACCUMULATOR_PROVENANCE,
        accumulator.source_generation_digest,
        accumulator.compact_manifest_digest,
        accumulator.step_generation_id,
        accumulator.loss_normalization_id,
        accumulator.global_loss_element_count,
        accumulator.loss_scale,
        accumulator.material_generation_id,
        accumulator.background_generation_id,
        accumulator.material_tensor_identity,
        accumulator.material_tensor_signature,
        accumulator.background_tensor_identity,
        accumulator.background_tensor_signature,
        accumulator.world_generation_digest,
        accumulator.world_sites_content_digest,
        accumulator.site_table_identity,
        accumulator.ray_bar_keys_generation_digest,
        accumulator.full_geometry,
        accumulator.optimize_camera_rays,
        accumulator.tensor_signatures,
        accumulator.consumed_request_count,
        accumulator.consumed_observation_count,
        accumulator.fenced_request_commit_count,
        accumulator.request_commit_fence_provenance,
        accumulator.request_commit_chain_digest,
        accumulator._request_commit_fence_identity,
        (
            accumulator._async_failure_quarantine.generation_digest
            if accumulator._async_failure_quarantine is not None
            else ""
        ),
        accumulator.pending_request_generation_digest,
        accumulator.pending_delta_generation_digest,
        accumulator.poisoned,
        accumulator.sealed,
        accumulator.optimizer_authorized,
    )


def _request_delta_digest(delta: PaperKineticDenseRequestGradientDelta) -> str:
    return _digest_parts(
        REQUEST_DELTA_PROVENANCE,
        delta.source_generation_digest,
        delta.request_generation_digest,
        delta.artifact_generation_digest,
        delta.step_generation_id,
        delta.receipt.generation_digest,
        delta.telemetry.generation_digest,
        delta.full_geometry,
        delta.optimize_camera_rays,
        delta.ray_bar_keys,
        delta.sealed_tensor_signatures,
        delta.persistent_frame_tensor_bytes,
        delta.persistent_sample_tensor_bytes,
        delta.persistent_target_tensor_bytes,
        delta.persistent_prediction_tensor_bytes,
    )


def _request_delta_commit_receipt_digest(
    receipt: PaperKineticDenseRequestDeltaCommitReceipt,
) -> str:
    return _digest_parts(
        REQUEST_DELTA_COMMIT_PROVENANCE,
        receipt.source_generation_digest,
        receipt.request_generation_digest,
        receipt.artifact_generation_digest,
        receipt.step_generation_id,
        receipt.delta_generation_digest,
        receipt.accumulator_generation_digest_after_commit,
        receipt.request_commit_chain_digest_after_commit,
        receipt.consumed_request_count,
        receipt.consumed_observation_count,
        receipt.device_completion_fence_provenance,
        receipt.device_completion_fence_call_count,
        receipt.persistent_tensor_bytes,
        receipt.delta_tensors_released_after_fence,
    )


def _optimizer_authorization_digest(
    authorization: PaperKineticDenseOptimizerAuthorization,
) -> str:
    return _digest_parts(
        OPTIMIZER_AUTHORIZATION_PROVENANCE,
        authorization.source_generation_digest,
        authorization.compact_manifest_digest,
        authorization.step_generation_id,
        authorization.replay_receipt_generation_digest,
        authorization.accumulator_generation_digest,
        authorization.request_count,
        authorization.observation_count,
        authorization.full_geometry,
        authorization.optimize_camera_rays,
        authorization.ray_bar_keys,
        authorization.tensor_signatures,
    )


def _validate_row_time(row: PaperKineticRowBinding, time: float) -> None:
    schedule = row.program.charts[row.chart_index].schedule
    if time < schedule.t_min or time > schedule.t_max:
        raise ValueError("dense chunk dispatched outside its kinetic chart")
    if time == schedule.t_max and not row.right_closed:
        raise ValueError("dense chunk dispatched to an excluded chart endpoint")


def _target_digest(targets: PaperKineticDenseChunkTargets) -> str:
    return _digest_parts(
        TARGET_PROVENANCE,
        targets.source_generation_digest,
        targets.request_generation_digest,
        targets.chunk_generation_digest,
        targets.target_generation_id,
        targets.logical_tensor_bytes,
        targets.selected_pixel_read_mode,
        targets.selected_pixel_read_source_provenance,
        targets.selected_pixel_read_call_count,
        targets.selected_pixel_read_acceptance_capable,
        targets.direct_selected_pixel_observation_count,
        targets.bounded_region_selected_pixel_observation_count,
        targets.full_frame_fallback_observation_count,
        targets.decoded_frame_count,
        targets.maximum_cpu_decoded_frame_tensor_bytes,
        targets.bounded_region_materialization_count,
        targets.maximum_bounded_region_materialization_tensor_bytes,
        targets.source_visible_target_read_peak_logical_tensor_bytes_upper_bound,
        targets.transient_mapped_address_space_bytes,
        targets.maximum_requested_unique_mapped_page_count,
        targets.total_requested_unique_mapped_page_count,
        targets.mapped_page_size_bytes,
        targets.maximum_requested_mapped_page_bytes_upper_bound,
        targets.total_requested_mapped_page_bytes_upper_bound,
        targets.mapping_closed_before_return,
        targets.cpu_chunk_target_tensor_bytes,
        targets.device_chunk_target_tensor_bytes,
        targets.target_decode_bridge_peak_logical_tensor_bytes,
        targets.decoded_frame_device_type,
        targets.decoded_frame_mps_completion_fence_call_count,
        targets.cpu_to_device_chunk_transfer_requested_non_blocking,
        targets.single_bounded_chunk_transfer,
        targets.real_device_transfer_completion_verified,
        targets.warm_tensor_signature,
        targets._cpu_transfer_source_identity,
        targets._cpu_transfer_source_signature,
    )


def _target_loader_digest(loader: PaperKineticDenseChunkTargetLoader) -> str:
    # Bind immutable authority/configuration only. Completion/failure counters
    # and the one active lifetime are intentionally mutable state validated by
    # ``assert_current``; including either would require resealing after every
    # chunk and would turn the digest itself into replay history.
    return _digest_parts(
        TARGET_LOADER_PROVENANCE,
        loader.source_generation_digest,
        loader.request_generation_digest,
        loader.target_generation_id,
        loader.device,
        loader.maximum_decoded_frame_scratch_tensor_bytes,
        loader.maximum_chunk_target_tensor_bytes,
        loader.maximum_target_decode_bridge_peak_logical_tensor_bytes,
        loader._source_identity,
        loader._request_identity,
        None if loader._test_fault is None else loader._test_fault.stage,
        None if loader._test_fault is None else loader._test_fault.message,
        (
            None
            if loader._test_fault is None
            else loader._test_fault.fail_on_load_number
        ),
    )


def _receipt_digest(receipt: PaperKineticDenseCachedRequestReceipt) -> str:
    return _digest_parts(
        REQUEST_PROVENANCE,
        "receipt",
        receipt.source_generation_digest,
        receipt.request_generation_digest,
        receipt.artifact_generation_digest,
        receipt.session_identity,
        receipt.session_request_count_before,
        receipt.session_request_count_after,
        receipt.session_emitted_observation_count_before,
        receipt.session_emitted_observation_count_after,
        receipt.expected_observation_count,
        receipt.replay_chunk_count,
        receipt.replay_chunk_manifest_digest,
    )


def _result_digest(result: PaperKineticDenseCachedNativeRequestResult) -> str:
    return _digest_parts(
        REQUEST_PROVENANCE,
        "result",
        result.source_generation_digest,
        result.request_generation_digest,
        result.artifact_generation_digest,
        result.lane_generation_digest,
        result.receipt.generation_digest,
        result.telemetry.generation_digest,
        result.delta.generation_digest,
        result.loss_delta_f32,
        tuple(result.accounting.items()),
        result.full_geometry_vjp_integrated,
        result.full_geometry_reverse_mode,
        result.runtime_status,
        result.target_loader_is_arbitrary_callable,
        result.target_loader_partial_failure_lifetime_certified,
        result.decoder_allocator_peak_measured,
        result.sample_materialization_float64_scratch_measured,
        result.whole_step_python_object_peak_measured,
    )


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        id(tensor),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        bool(tensor.requires_grad),
        int(tensor._version),
    )


def _require_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tensor.device != device or tensor.dtype != dtype or tuple(tensor.shape) != shape:
        raise ValueError(
            f"{name} must be {dtype} {shape} on {device}, got {tensor.dtype} {tuple(tensor.shape)} on {tensor.device}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _require_cpu_f64_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
) -> None:
    _require_tensor(
        tensor,
        name=name,
        device=torch.device("cpu"),
        dtype=torch.float64,
        shape=shape,
    )
    if tensor.requires_grad or not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite and explicit-bar only")


def _require_distinct_storage(*tensors: torch.Tensor) -> None:
    identities = {
        (str(tensor.device), tensor.untyped_storage().data_ptr())
        for tensor in tensors
    }
    if len(identities) != len(tensors):
        raise ValueError("dense explicit tensors must own distinct storage")


def _tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        _update_digest(digest, part)
    return digest.hexdigest()


def _update_digest(digest: Any, part: object) -> None:
    encoded = repr(part).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
    digest.update(encoded)


def _require_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _fail_memory(message: str) -> None:
    raise MemoryError(message)


def _fail_arithmetic(message: str) -> None:
    raise ArithmeticError(message)


def _fail_type(message: str) -> None:
    raise TypeError(message)


# Compatibility surface for the earlier material-only one-request seam. It
# fails closed unless that one request is the source's complete manifest.
PaperKineticDenseCachedNativeMaterialRequestResult = (
    PaperKineticDenseCachedNativeRequestResult
)


@torch.no_grad()
def run_paper_kinetic_dense_cached_native_material_request(
    source: PaperKineticReplayableDenseObservationSource,
    session: PaperKineticDenseObservationReplaySession,
    request: PaperKineticDenseObservationTrackRequest,
    artifact: PaperKineticCompiledCpuArtifact,
    *,
    step_generation_id: str,
    loss_normalization_id: str,
    material_generation_id: str,
    background_generation_id: str,
    global_site_rgba_f32: torch.Tensor,
    global_grad_site_rgba_f32: torch.Tensor,
    loss_f32: torch.Tensor,
    background_rgb_f32: torch.Tensor,
    native_ops: Any,
    backend_provenance: str,
    maximum_samples_per_launch: int,
    memory_policy: PaperKineticDenseCachedNativeMemoryPolicy,
    load_chunk_targets: PaperKineticDenseChunkTargetLoader,
    device_completion_fence: Callable[[], None],
    device_completion_fence_provenance: str,
    cone_tolerance: float = 1.0e-5,
) -> PaperKineticDenseCachedNativeMaterialRequestResult:
    """Non-production compatibility commit with a fenced fail-stop boundary."""

    device = global_site_rgba_f32.device
    _require_tensor(
        global_grad_site_rgba_f32,
        name="legacy global_grad_site_rgba_f32",
        device=device,
        dtype=torch.float32,
        shape=(source.provider.world.site_count, 4),
    )
    _require_tensor(
        loss_f32,
        name="legacy loss_f32",
        device=device,
        dtype=torch.float32,
        shape=(1,),
    )
    if global_grad_site_rgba_f32.requires_grad or loss_f32.requires_grad:
        raise ValueError("legacy explicit destination bars must not require grad")
    accumulator = prepare_paper_kinetic_dense_step_gradient_accumulator(
        source,
        session,
        step_generation_id=step_generation_id,
        loss_normalization_id=loss_normalization_id,
        material_generation_id=material_generation_id,
        background_generation_id=background_generation_id,
        global_site_rgba_f32=global_site_rgba_f32,
        background_rgb_f32=background_rgb_f32,
        device=device,
        full_geometry=False,
    )
    _require_distinct_storage(
        global_site_rgba_f32,
        global_grad_site_rgba_f32,
        loss_f32,
        background_rgb_f32,
    )
    result = run_paper_kinetic_dense_cached_native_request(
        source,
        session,
        request,
        artifact,
        accumulator,
        step_generation_id=step_generation_id,
        loss_normalization_id=loss_normalization_id,
        material_generation_id=material_generation_id,
        background_generation_id=background_generation_id,
        global_site_rgba_f32=global_site_rgba_f32,
        background_rgb_f32=background_rgb_f32,
        native_ops=native_ops,
        backend_provenance=backend_provenance,
        maximum_samples_per_launch=maximum_samples_per_launch,
        memory_policy=memory_policy,
        load_chunk_targets=load_chunk_targets,
        device_completion_fence=device_completion_fence,
        device_completion_fence_provenance=device_completion_fence_provenance,
        cone_tolerance=cone_tolerance,
    )
    consume_paper_kinetic_dense_request_delta(
        accumulator,
        source,
        session,
        request,
        artifact,
        result.delta,
        device_completion_fence=device_completion_fence,
        device_completion_fence_provenance=device_completion_fence_provenance,
    )
    replay_receipt = session.seal()
    authorization = authorize_paper_kinetic_dense_optimizer_step(
        accumulator,
        source,
        session,
        replay_receipt,
    )
    try:
        global_grad_site_rgba_f32.add_(authorization.grad_site_rgba_f32)
        loss_f32.add_(authorization.loss_f32)
        returned = device_completion_fence()
        if returned is not None:
            raise TypeError("legacy caller-visible commit fence must return None")
    except BaseException:
        _poison_dense_step_accumulator(accumulator)
        # This compatibility path cannot atomically roll back two external
        # tensors. Mark both unusable and attempt one final flush before the
        # triggering exception escapes; production code must use authorization.
        try:
            global_grad_site_rgba_f32.fill_(float("nan"))
            loss_f32.fill_(float("nan"))
            device_completion_fence()
        except BaseException:
            pass
        raise
    legacy_accounting = dict(result.accounting)
    legacy_accounting.update(
        {
            "legacy_material_compatibility_wrapper": True,
            "legacy_production_promotion_allowed": False,
            "request_delta_commit_fence_call_count": 1,
            "legacy_caller_visible_commit_fence_call_count": 1,
            "total_completion_fence_call_count_including_legacy_commit": (
                int(result.accounting["native_lane_fence_count"]) + 2
            ),
            "legacy_commit_failure_policy": "poison_destinations_and_raise",
        }
    )
    updated = replace(result, accounting=legacy_accounting, generation_digest="")
    return replace(updated, generation_digest=_result_digest(updated))


__all__ = [
    "FUSED_DIRECT_V1_FULL_GEOMETRY_REVERSE",
    "LANE_PROVENANCE",
    "MPS_DEVICE_COMPLETION_FENCE_PROVENANCE",
    "REQUEST_PROVENANCE",
    "REQUEST_STATUS",
    "TARGET_PROVENANCE",
    "TARGET_LOADER_PROVENANCE",
    "STEP_ACCUMULATOR_PROVENANCE",
    "STAGED_SPARSE_FULL_GEOMETRY_REVERSE",
    "REQUEST_DELTA_PROVENANCE",
    "REQUEST_DELTA_COMMIT_PROVENANCE",
    "OPTIMIZER_AUTHORIZATION_PROVENANCE",
    "PaperKineticDenseCachedNativeRequestResult",
    "PaperKineticDenseCachedNativeMaterialRequestResult",
    "PaperKineticDenseCachedNativeMemoryPolicy",
    "PaperKineticDenseCachedRequestReceipt",
    "PaperKineticDenseChunkTargetLoader",
    "PaperKineticDenseChunkTargetLoaderTestFault",
    "PaperKineticDenseChunkTargetLoadLifetime",
    "PaperKineticDenseChunkTargets",
    "PaperKineticDenseOptimizerAuthorization",
    "PaperKineticDenseRequestDeltaCommitReceipt",
    "PaperKineticDenseRequestGradientDelta",
    "PaperKineticDenseStepGradientAccumulator",
    "authorize_paper_kinetic_dense_optimizer_step",
    "consume_paper_kinetic_dense_request_delta",
    "decode_paper_kinetic_dense_chunk_targets",
    "fail_stop_paper_kinetic_dense_step",
    "iter_paper_kinetic_dense_chunk_sample_blocks",
    "prepare_paper_kinetic_dense_step_gradient_accumulator",
    "prepare_paper_kinetic_dense_chunk_target_loader",
    "prepare_paper_kinetic_dense_chunk_target_loader_test_fault",
    "run_paper_kinetic_dense_cached_native_request",
    "run_paper_kinetic_dense_cached_native_material_request",
    "seal_paper_kinetic_dense_chunk_targets",
    "synchronize_mps_device_completion_fence",
]
