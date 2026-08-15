"""Lazy dataset-bound kinetic programs for sparse paper observations.

The paper sampler names image observations, while the kinetic compiler owns
one continuous-time program per ``(view, pixel)`` track.  This module seals the
join without introducing a ``view x frame x pixel`` tensor:

* callers provide only the exact ``(view, frame, pixel)`` observations used by
  a paper step;
* calibrated rays are generated for those observations only;
* programs are compiled lazily in bounded, view-local spatial bundles;
* every yielded bundle contains exactly its requested sparse observations --
  never the Cartesian product of its tracks and frames;
* targets are never decoded and dense ray grids are never materialized;
* camera, dataset, initializer, factory, world, program, and coverage
  provenance fail closed when stale.

The retained ``PowerFoamRayProvider`` camera records and one tuple of frame
times are the allowed cheap ``O(VF)`` camera slice.  Kinetic program payloads,
native lowerings, and union maps have no dense frame axis.  A yielded bundle
does retain one Python ray record per *selected* observation and therefore is
deliberately limited to bounded sparse sampling.  Dense-``F`` training still
requires pairing frame-free structural program artifacts with the bounded
source in ``paper_kinetic_replayable_observations``.  That source now exists,
but this legacy bundle still embeds selected records and does not consume it.
This is a CPU/source integration seam; it launches no native kernel and
performs no optimizer update.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

import torch
from camera import CameraSpec, build_camera_rays_at_pixels
from paper_kinetic_runtime_paths import ensure_worldfoam_lane2_research_path

ensure_worldfoam_lane2_research_path()

from kinetic_multichart_transfer_program import KineticMultiChartP0Program  # noqa: E402
from kinetic_native_equal_rank_lowering import (  # noqa: E402
    KineticNativeEqualRankChartSource,
    kinetic_native_equal_rank_chart_sources_for_track,
    lower_kinetic_native_equal_rank_buckets,
)
from kinetic_native_topology_lowering import (  # noqa: E402
    lower_kinetic_multichart_to_native_topology,
)
from kinetic_power_word_compiler import AffineKineticPowerSites  # noqa: E402
from paper_kinetic_ragged_sample_plan import (  # noqa: E402
    PaperKineticRowRaggedSampler,
    prepare_paper_kinetic_row_ragged_sampler,
)
from paper_kinetic_union_local_bar_assembly import (  # noqa: E402
    PaperKineticUnionLocalSpatialBundle,
    PaperKineticUnionLocalSpatialBundleConstructionLifetime,
    certify_paper_kinetic_union_local_spatial_bundle_cold_current,
    prepare_paper_kinetic_union_local_spatial_bundle,
    prepare_paper_kinetic_union_local_spatial_bundle_construction_lifetime,
)
from paper_training_types import SpacetimeBatch  # noqa: E402
from powerfoam_training_data import PowerFoamRayProvider, PowerFoamTargetProvider  # noqa: E402

PROVIDER_PROVENANCE = "paper-kinetic-lazy-program-bundle-provider-v1"
RAY_RECORD_PROVENANCE = "paper-kinetic-selected-calibrated-ray-v1"
WORLD_SNAPSHOT_PROVENANCE = "paper-kinetic-world-snapshot-v1"
PROGRAM_REQUEST_PROVENANCE = "paper-kinetic-track-program-request-v1"
TRACK_REPLAY_VALIDATION_PROVENANCE = "paper-kinetic-track-replay-validation-v1"
BOUNDED_OBSERVATION_SCOPE = "bounded_sparse_sampled_observations_only"
COMPILE_RECEIPT_PROVENANCE = (
    "paper-kinetic-lazy-bundle-compiler-receipt-v1"
)

_PROVIDER_SEAL = object()
_BUNDLE_SEAL = object()
_BUNDLE_CONSTRUCTION_SLOT_SEAL = object()


@dataclass
class PaperKineticLazyBundleConstructionLifetimeSlot:
    """Caller-owned slot retaining at most one partial spatial construction."""

    active_lifetime: (
        PaperKineticUnionLocalSpatialBundleConstructionLifetime | None
    ) = field(default=None, repr=False)
    install_count: int = 0
    completion_count: int = 0
    release_after_completion_fence_count: int = 0
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        outstanding_count = (
            self.install_count
            - self.completion_count
            - self.release_after_completion_fence_count
        )
        if (
            self._seal is not _BUNDLE_CONSTRUCTION_SLOT_SEAL
            or type(self) is not PaperKineticLazyBundleConstructionLifetimeSlot
            or type(self.install_count) is not int
            or type(self.completion_count) is not int
            or type(self.release_after_completion_fence_count) is not int
            or outstanding_count not in {0, 1}
            or (self.active_lifetime is None) != (outstanding_count == 0)
        ):
            raise ValueError("lazy bundle construction lifetime slot changed")
        if self.active_lifetime is not None:
            self.active_lifetime.assert_retained()

    def install(
        self,
        lifetime: PaperKineticUnionLocalSpatialBundleConstructionLifetime,
    ) -> None:
        self.assert_current()
        if self.active_lifetime is not None:
            raise RuntimeError("lazy bundle construction lifetime is already active")
        lifetime.assert_retained()
        if lifetime.phase != "installed":
            raise ValueError("lazy bundle construction lifetime was already used")
        self.active_lifetime = lifetime
        self.install_count += 1
        self.assert_current()

    def complete(
        self,
        lifetime: PaperKineticUnionLocalSpatialBundleConstructionLifetime,
    ) -> None:
        self.assert_current()
        if (
            self.active_lifetime is not lifetime
            or lifetime.phase not in {"materialized", "settled", "retired"}
        ):
            raise ValueError("lazy bundle construction completion is foreign")
        self.active_lifetime = None
        self.completion_count += 1
        self.assert_current()

    def release_active_after_completion_fence(self) -> None:
        self.assert_current()
        if self.active_lifetime is None:
            return
        if self.active_lifetime.device.type != "cpu":
            raise RuntimeError(
                "authority-free partial-bundle release is CPU-only; "
                "accelerator release requires an exact consumed subject receipt"
            )
        self._commit_active_release_after_consumed_receipt()
        self.assert_current()

    def assert_active_releasable_after_consumed_receipt(self) -> None:
        """Validate the partial construction before receipt consumption."""

        self.assert_current()
        if self.active_lifetime is None:
            raise RuntimeError("lazy bundle construction has no active lifetime")

    def _commit_active_release_after_consumed_receipt(self) -> None:
        """Assignment-only slot release after exact authority is consumed."""

        self.active_lifetime = None
        self.release_after_completion_fence_count += 1


def prepare_paper_kinetic_lazy_bundle_construction_lifetime_slot(
) -> PaperKineticLazyBundleConstructionLifetimeSlot:
    slot = PaperKineticLazyBundleConstructionLifetimeSlot(
        _seal=_BUNDLE_CONSTRUCTION_SLOT_SEAL,
    )
    slot.assert_current()
    return slot


@dataclass(frozen=True, order=True)
class PaperKineticObservation:
    """One logical paper loss element before RGB channel expansion."""

    observation_id: int
    view_index: int
    frame_index: int
    pixel_index: int

    def __post_init__(self) -> None:
        for name, value in (
            ("observation_id", self.observation_id),
            ("view_index", self.view_index),
            ("frame_index", self.frame_index),
            ("pixel_index", self.pixel_index),
        ):
            _require_nonnegative_int(value, name=name)

    @property
    def track_identity(self) -> tuple[int, int]:
        return (self.view_index, self.pixel_index)

    @property
    def sample_identity(self) -> tuple[int, int, int, int]:
        return (
            self.observation_id,
            self.view_index,
            self.frame_index,
            self.pixel_index,
        )


@dataclass(frozen=True)
class PaperKineticObservationRayRecord:
    """A selected calibrated ray represented as six Python float64 scalars."""

    observation: PaperKineticObservation
    sample_time: float
    ray_origin_direction: tuple[float, float, float, float, float, float]
    camera_record_digest: str
    generation_digest: str
    provenance: str = RAY_RECORD_PROVENANCE

    def assert_self_consistent(self) -> None:
        if (
            self.provenance != RAY_RECORD_PROVENANCE
            or not math.isfinite(self.sample_time)
            or len(self.ray_origin_direction) != 6
            or not all(math.isfinite(value) for value in self.ray_origin_direction)
        ):
            raise ValueError("paper kinetic observation ray record is invalid")
        _require_sha256(self.camera_record_digest, name="camera_record_digest")
        if self.generation_digest != _ray_record_digest(
            self.observation,
            sample_time=self.sample_time,
            ray=self.ray_origin_direction,
            camera_record_digest=self.camera_record_digest,
        ):
            raise ValueError("paper kinetic observation ray provenance changed")


@dataclass(frozen=True)
class PaperKineticWorldInitializationRequest:
    """Frame-payload-free metadata offered to the world initializer."""

    dataset_generation_digest: str
    camera_grid_digest: str
    view_count: int
    frame_count: int
    height: int
    width: int
    initializer_generation_digest: str
    generation_digest: str

    def assert_self_consistent(self) -> None:
        for name, digest in (
            ("dataset_generation_digest", self.dataset_generation_digest),
            ("camera_grid_digest", self.camera_grid_digest),
            ("initializer_generation_digest", self.initializer_generation_digest),
        ):
            _require_sha256(digest, name=name)
        for name, value in (
            ("view_count", self.view_count),
            ("frame_count", self.frame_count),
            ("height", self.height),
            ("width", self.width),
        ):
            _require_positive_int(value, name=name)
        if self.generation_digest != _digest_parts(
            PROVIDER_PROVENANCE,
            "world-init-request",
            self.dataset_generation_digest,
            self.camera_grid_digest,
            self.view_count,
            self.frame_count,
            self.height,
            self.width,
            self.initializer_generation_digest,
        ):
            raise ValueError("paper kinetic world initialization request changed")


@dataclass(frozen=True)
class PaperKineticWorldSnapshot:
    """Cold-sealed affine kinetic geometry shared by every lazy track."""

    sites: AffineKineticPowerSites = field(repr=False)
    dataset_generation_digest: str
    initializer_generation_digest: str
    sites_content_digest: str
    generation_digest: str
    _site_tensor_identities: tuple[int, int, int] = field(repr=False)
    _site_tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    provenance: str = WORLD_SNAPSHOT_PROVENANCE

    @property
    def site_count(self) -> int:
        return self.sites.site_count

    def assert_warm_current(self) -> None:
        """O(1)-tensor-count identity/version check for warm bundle paths.

        Full tensor contents are certified by :meth:`assert_current` at a cold
        step/provider boundary.  Warm checks use tensor identity, layout, and
        PyTorch mutation versions, so they do not rehash every world scalar for
        every spatial bundle.
        """

        if self.provenance != WORLD_SNAPSHOT_PROVENANCE:
            raise ValueError("paper kinetic world provenance changed")
        _require_sha256(
            self.dataset_generation_digest,
            name="world.dataset_generation_digest",
        )
        _require_sha256(
            self.initializer_generation_digest,
            name="world.initializer_generation_digest",
        )
        tensors = _site_tensors(self.sites)
        if tuple(id(tensor) for tensor in tensors) != self._site_tensor_identities:
            raise ValueError("paper kinetic world tensor identity changed")
        if tuple(_tensor_signature(tensor) for tensor in tensors) != self._site_tensor_signatures:
            raise ValueError("paper kinetic world tensor content changed after cold certification")
        if self.generation_digest != _digest_parts(
            WORLD_SNAPSHOT_PROVENANCE,
            self.dataset_generation_digest,
            self.initializer_generation_digest,
            self.sites_content_digest,
            self.site_count,
        ):
            raise ValueError("paper kinetic world snapshot generation changed")

    def assert_current(self) -> None:
        """Cold full-content certification; call once before warm replay."""

        self.assert_warm_current()
        current_content = _site_content_digest(self.sites)
        if current_content != self.sites_content_digest:
            raise ValueError("paper kinetic world tensor content changed")


@dataclass(frozen=True)
class PaperKineticTrackProgramRequest:
    """Structural compile request with bounded, sample-independent witnesses.

    ``observations`` is retained as a compatibility name for factories already
    using one calibrated ray to construct an affine track.  The provider now
    supplies fixed endpoint witnesses, never the logical step's sampled
    observations, and the structural generation digest deliberately excludes
    them.  Selected-observation replay validation is performed separately
    after compilation, so changing sampled frames cannot invalidate a compiled
    track key.
    """

    world: PaperKineticWorldSnapshot = field(repr=False)
    dataset_generation_digest: str
    factory_generation_digest: str
    view_index: int
    pixel_index: int
    height: int
    width: int
    frame_times: tuple[float, ...]
    cameras: tuple[CameraSpec, ...] = field(repr=False)
    camera_path_digest: str
    static_camera_path_certified: bool
    observations: tuple[PaperKineticObservationRayRecord, ...]
    generation_digest: str
    provenance: str = PROGRAM_REQUEST_PROVENANCE

    def assert_self_consistent(self) -> None:
        self.world.assert_warm_current()
        if (
            self.provenance != PROGRAM_REQUEST_PROVENANCE
            or self.dataset_generation_digest != self.world.dataset_generation_digest
            or len(self.frame_times) != len(self.cameras)
            or len(self.frame_times) < 1
            or not isinstance(self.static_camera_path_certified, bool)
            or not self.observations
            or any(
                record.observation.track_identity != (self.view_index, self.pixel_index) for record in self.observations
            )
            or tuple(record.observation.frame_index for record in self.observations)
            != tuple(dict.fromkeys((0, len(self.frame_times) - 1)))
        ):
            raise ValueError("paper kinetic track program request is inconsistent")
        _require_sha256(self.factory_generation_digest, name="factory_generation_digest")
        _require_sha256(self.camera_path_digest, name="camera_path_digest")
        if self.height < 1 or self.width < 1:
            raise ValueError("paper kinetic track program request has invalid dimensions")
        if not all(math.isfinite(time) for time in self.frame_times):
            raise ValueError("paper kinetic track program frame times must be finite")
        for record in self.observations:
            record.assert_self_consistent()
        if self.generation_digest != _track_request_digest(self):
            raise ValueError("paper kinetic track program request generation changed")


@dataclass(frozen=True)
class PaperKineticTrackReplayValidation:
    """Bounded sampled-ray check kept outside structural compile identity."""

    structural_request_generation_digest: str
    view_index: int
    pixel_index: int
    observations: tuple[PaperKineticObservationRayRecord, ...]
    generation_digest: str
    provenance: str = TRACK_REPLAY_VALIDATION_PROVENANCE

    def assert_self_consistent(
        self,
        request: PaperKineticTrackProgramRequest,
    ) -> None:
        if (
            self.provenance != TRACK_REPLAY_VALIDATION_PROVENANCE
            or self.structural_request_generation_digest != request.generation_digest
            or (self.view_index, self.pixel_index) != (request.view_index, request.pixel_index)
            or not self.observations
            or any(
                record.observation.track_identity != (self.view_index, self.pixel_index) for record in self.observations
            )
        ):
            raise ValueError("paper kinetic sampled-ray replay validation is inconsistent")
        for record in self.observations:
            record.assert_self_consistent()
        if self.generation_digest != _track_replay_validation_digest(self):
            raise ValueError("paper kinetic sampled-ray replay validation changed")


@runtime_checkable
class PaperKineticWorldInitializer(Protocol):
    provenance: str
    generation_digest: str

    def initialize_world(
        self,
        request: PaperKineticWorldInitializationRequest,
    ) -> AffineKineticPowerSites: ...


@runtime_checkable
class PaperKineticTrackProgramFactory(Protocol):
    provenance: str
    generation_digest: str

    def compile_track(
        self,
        request: PaperKineticTrackProgramRequest,
    ) -> KineticMultiChartP0Program: ...


@dataclass(frozen=True)
class PaperKineticLazyProgramBundleMemoryReport:
    requested_frame_count: int
    observation_count: int
    track_count: int
    camera_record_count: int
    persistent_frame_tensor_bytes: int
    persistent_target_tensor_bytes: int
    persistent_dense_ray_tensor_bytes: int
    persistent_observation_ray_tensor_bytes: int
    dense_track_frame_tensor_bytes: int
    full_target_video_resident: bool
    full_ray_video_resident: bool
    selected_ray_scalar_count: int
    cartesian_padding_observation_count: int
    provider_owned_retained_bundle_count: int
    one_live_bundle_enforced_by_provider: bool
    consumer_must_release_bundle_before_next: bool
    observation_residency_scope: str
    dense_frame_observation_streaming_implemented: bool
    python_object_bytes_measured: bool
    allocator_peak_measured: bool


@dataclass(frozen=True)
class PaperKineticLazyProgramCompileReceipt:
    """Tensor-free aggregate captured before compiled programs are released."""

    provider_generation_digest: str
    view_index: int
    compile_track_count: int
    compiler_work_receipt_count: int
    compiler_work_receipt_chain_link_count: int
    root_complement_witness_count: int
    candidate_source_attempt_count: int
    all_site_witness_check_count: int
    unique_pair_difference_count: int
    per_witness_candidate_bound_verified: bool
    exhaustive_triple_enumeration_used: bool
    requested_frame_sampling_used: bool
    compiler_accounting_complete: bool
    all_track_receipt_digests_verified: bool
    compiler_work_receipt_provenance: str
    compiler_work_receipt_chain_digest: str
    generation_digest: str
    retained_compiled_program_count: int = 0
    retained_compiler_receipt_entry_count: int = 0
    retained_compiler_tensor_bytes: int = 0
    provenance: str = COMPILE_RECEIPT_PROVENANCE

    def assert_current(
        self,
        *,
        track_ids: tuple[int, ...],
        program_generation_digests: tuple[str, ...],
        request_generation_digests: tuple[str, ...],
    ) -> None:
        counts = (
            self.compile_track_count,
            self.compiler_work_receipt_count,
            self.compiler_work_receipt_chain_link_count,
            self.root_complement_witness_count,
            self.candidate_source_attempt_count,
            self.all_site_witness_check_count,
            self.unique_pair_difference_count,
            self.retained_compiled_program_count,
            self.retained_compiler_receipt_entry_count,
            self.retained_compiler_tensor_bytes,
        )
        if (
            self.provenance != COMPILE_RECEIPT_PROVENANCE
            or any(type(value) is not int or value < 0 for value in counts)
            or any(
                type(value) is not bool
                for value in (
                    self.per_witness_candidate_bound_verified,
                    self.exhaustive_triple_enumeration_used,
                    self.requested_frame_sampling_used,
                    self.compiler_accounting_complete,
                    self.all_track_receipt_digests_verified,
                )
            )
            or self.compile_track_count != len(track_ids)
            or self.compile_track_count != len(program_generation_digests)
            or self.compile_track_count != len(request_generation_digests)
            or self.compiler_work_receipt_chain_link_count
            != self.compile_track_count
            or self.compiler_work_receipt_count > self.compile_track_count
            or self.compiler_accounting_complete
            != (
                self.compile_track_count > 0
                and self.compiler_work_receipt_count == self.compile_track_count
                and self.compiler_work_receipt_provenance
                not in {"unavailable", "mixed"}
            )
            or self.all_track_receipt_digests_verified
            != (
                self.compile_track_count > 0
                and self.compiler_work_receipt_count == self.compile_track_count
            )
            or self.per_witness_candidate_bound_verified
            and self.compiler_work_receipt_count == 0
            or not self.compiler_work_receipt_provenance.strip()
            or self.retained_compiled_program_count != 0
            or self.retained_compiler_receipt_entry_count != 0
            or self.retained_compiler_tensor_bytes != 0
        ):
            raise ValueError("paper kinetic lazy compiler receipt changed")
        _require_sha256(
            self.provider_generation_digest,
            name="compile_receipt.provider_generation_digest",
        )
        _require_nonnegative_int(self.view_index, name="compile_receipt.view_index")
        _require_sha256(
            self.compiler_work_receipt_chain_digest,
            name="compiler_work_receipt_chain_digest",
        )
        if self.generation_digest != _compile_receipt_digest(
            self,
            track_ids=track_ids,
            program_generation_digests=program_generation_digests,
            request_generation_digests=request_generation_digests,
        ):
            raise ValueError("paper kinetic lazy compiler receipt generation changed")


@dataclass(frozen=True)
class PaperKineticLazyProgramBundle:
    """One ephemeral view-local program bundle for exact sparse observations."""

    provider_generation_digest: str
    bundle_index: int
    view_index: int
    track_ids: tuple[int, ...]
    observations: tuple[PaperKineticObservationRayRecord, ...]
    sampler: PaperKineticRowRaggedSampler = field(repr=False)
    spatial_bundle: PaperKineticUnionLocalSpatialBundle = field(repr=False)
    program_generation_digests: tuple[str, ...]
    factory_request_generation_digests: tuple[str, ...]
    compile_receipt: PaperKineticLazyProgramCompileReceipt
    generation_digest: str
    _provider_identity: int = field(repr=False)
    _sampler_identity: int = field(repr=False)
    _spatial_bundle_identity: int = field(repr=False)
    provenance: str = PROVIDER_PROVENANCE
    persistent_frame_tensor_bytes: int = 0
    persistent_target_tensor_bytes: int = 0
    persistent_dense_ray_tensor_bytes: int = 0
    dense_track_frame_tensor_bytes: int = 0
    cartesian_padding_observation_count: int = 0
    _seal: object = field(default=None, repr=False)

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    @property
    def track_count(self) -> int:
        return len(self.track_ids)

    @property
    def observation_identities(self) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(record.observation.sample_identity for record in self.observations)

    @property
    def selected_ray_scalar_count(self) -> int:
        return self.observation_count * 6

    def memory_report(self, requested_frame_count: int) -> PaperKineticLazyProgramBundleMemoryReport:
        _require_positive_int(requested_frame_count, name="requested_frame_count")
        return PaperKineticLazyProgramBundleMemoryReport(
            requested_frame_count=requested_frame_count,
            observation_count=self.observation_count,
            track_count=self.track_count,
            camera_record_count=requested_frame_count,
            persistent_frame_tensor_bytes=0,
            persistent_target_tensor_bytes=0,
            persistent_dense_ray_tensor_bytes=0,
            persistent_observation_ray_tensor_bytes=0,
            dense_track_frame_tensor_bytes=0,
            full_target_video_resident=False,
            full_ray_video_resident=False,
            selected_ray_scalar_count=self.selected_ray_scalar_count,
            cartesian_padding_observation_count=0,
            provider_owned_retained_bundle_count=0,
            one_live_bundle_enforced_by_provider=False,
            consumer_must_release_bundle_before_next=True,
            observation_residency_scope=BOUNDED_OBSERVATION_SCOPE,
            dense_frame_observation_streaming_implemented=False,
            python_object_bytes_measured=False,
            allocator_peak_measured=False,
        )

    def assert_exact_observation_coverage(
        self,
        observations: Sequence[PaperKineticObservation],
    ) -> None:
        """Reject missing, extra, duplicated, or substituted observations."""

        requested = tuple(observation.sample_identity for observation in observations)
        if len(set(requested)) != len(requested):
            raise ValueError("paper kinetic coverage request contains duplicate observations")
        expected = set(self.observation_identities)
        observed = set(requested)
        if observed != expected:
            missing = tuple(sorted(expected - observed))
            extra = tuple(sorted(observed - expected))
            raise ValueError(f"paper kinetic bundle observation coverage mismatch: missing={missing}, extra={extra}")

    def assert_cold_current(
        self,
        provider: PaperKineticLazyProgramBundleProvider,
    ) -> None:
        self._assert_metadata_current(provider)
        if self.spatial_bundle.device.type == "cpu":
            certify_paper_kinetic_union_local_spatial_bundle_cold_current(
                self.spatial_bundle
            )
        else:
            self.spatial_bundle.assert_accelerator_cold_current_after_settlement()
        self._assert_records_and_digest_current()

    def assert_accelerator_transfer_pending(
        self,
        provider: PaperKineticLazyProgramBundleProvider,
    ) -> None:
        """Validate a yielded pre-fence bundle without a device readback."""

        self._assert_metadata_current(provider)
        if self.spatial_bundle.device.type == "cpu":
            raise ValueError("pending accelerator bundle cannot be CPU resident")
        self.spatial_bundle.assert_accelerator_transfer_pending()
        self._assert_records_and_digest_current()

    def _assert_metadata_current(
        self,
        provider: PaperKineticLazyProgramBundleProvider,
    ) -> None:
        if (
            self._seal is not _BUNDLE_SEAL
            or self.provenance != PROVIDER_PROVENANCE
            or id(provider) != self._provider_identity
            or self.provider_generation_digest != provider.generation_digest
            or id(self.sampler) != self._sampler_identity
            or id(self.spatial_bundle) != self._spatial_bundle_identity
            or self.spatial_bundle.sampler is not self.sampler
            or self.view_index != self.sampler.view_index
            or self.track_ids != self.sampler.track_ids
            or self.track_ids != self.spatial_bundle.track_ids
            or not self.observations
            or any(record.observation.view_index != self.view_index for record in self.observations)
            or tuple(sorted({record.observation.pixel_index for record in self.observations})) != self.track_ids
            or len(self.program_generation_digests) != self.track_count
            or len(self.factory_request_generation_digests) != self.track_count
            or self.persistent_frame_tensor_bytes != 0
            or self.persistent_target_tensor_bytes != 0
            or self.persistent_dense_ray_tensor_bytes != 0
            or self.dense_track_frame_tensor_bytes != 0
            or self.cartesian_padding_observation_count != 0
        ):
            raise ValueError("paper kinetic lazy bundle metadata/memory contract changed")
        provider.assert_warm_current()

    def _assert_records_and_digest_current(self) -> None:
        for record in self.observations:
            record.assert_self_consistent()
        self.compile_receipt.assert_current(
            track_ids=self.track_ids,
            program_generation_digests=self.program_generation_digests,
            request_generation_digests=self.factory_request_generation_digests,
        )
        if self.generation_digest != _bundle_digest(self):
            raise ValueError("paper kinetic lazy bundle generation changed")


@dataclass(frozen=True)
class PaperKineticLazyProgramBundleProvider:
    """Dataset/world-bound compiler for bounded sparse observations only.

    The cold initializer is deliberately not retained.  Its provenance and
    generation digest form a tensor-free receipt, while ``world`` owns the
    cloned live geometry.  This keeps point-cloud template geometry and its
    RGBA seed out of the steady-state training object graph.
    """

    dataset_generation_digest: str
    target_provider: PowerFoamTargetProvider = field(repr=False)
    ray_provider: PowerFoamRayProvider = field(repr=False)
    frame_times: tuple[float, ...]
    height: int
    width: int
    maximum_tracks_per_bundle: int
    maximum_observations_per_bundle: int
    maximum_rows_per_native_block: int
    program_factory: PaperKineticTrackProgramFactory = field(repr=False)
    world: PaperKineticWorldSnapshot = field(repr=False)
    target_residency_digest: str
    camera_grid_digest: str
    view_camera_path_digests: tuple[str, ...]
    view_static_camera_path_certified: tuple[bool, ...]
    initializer_provenance: str
    initializer_generation_digest: str
    factory_provenance: str
    factory_generation_digest: str
    generation_digest: str
    _target_provider_identity: int = field(repr=False)
    _target_source_identity: int = field(repr=False)
    _ray_provider_identity: int = field(repr=False)
    _factory_identity: int = field(repr=False)
    _camera_grid_identity: int = field(repr=False)
    provenance: str = PROVIDER_PROVENANCE
    persistent_target_tensor_bytes: int = 0
    persistent_dense_ray_tensor_bytes: int = 0
    dense_track_frame_tensor_bytes: int = 0
    observation_residency_scope: str = BOUNDED_OBSERVATION_SCOPE
    dense_frame_observation_streaming_implemented: bool = False
    _seal: object = field(default=None, repr=False)

    @property
    def view_count(self) -> int:
        return self.target_provider.view_count

    @property
    def frame_count(self) -> int:
        return self.target_provider.frame_count

    def accounting(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance,
            "view_count": self.view_count,
            "frame_count": self.frame_count,
            "height": self.height,
            "width": self.width,
            "maximum_tracks_per_bundle": self.maximum_tracks_per_bundle,
            "maximum_observations_per_bundle": self.maximum_observations_per_bundle,
            "maximum_observations_per_track": self.maximum_observations_per_bundle,
            "maximum_rows_per_native_block": self.maximum_rows_per_native_block,
            "camera_record_count": self.view_count * self.frame_count,
            "camera_metadata_is_allowed_linear_slice": True,
            "static_camera_path_certified_view_count": sum(
                self.view_static_camera_path_certified
            ),
            "persistent_frame_tensor_bytes": 0,
            "persistent_target_tensor_bytes": 0,
            "persistent_dense_ray_tensor_bytes": 0,
            "dense_track_frame_tensor_bytes": 0,
            "full_target_video_resident": False,
            "full_ray_video_resident": False,
            "target_residency": self.target_provider.residency(),
            "programs_compiled_lazily": True,
            "provider_retains_yielded_bundles": False,
            "provider_owned_retained_bundle_count": 0,
            "caller_visible_bundle_construction_lifetime_slot_available": True,
            "accelerator_bundle_construction_requires_lifetime_slot": True,
            "one_live_bundle_enforced_by_provider": False,
            "consumer_must_release_bundle_before_next": True,
            "program_payload_peak_scaling": (
                "O(max_spatial_bundle) under a consumer that releases before next; "
                "the provider itself retains zero yielded bundles"
            ),
            "observation_residency_scope": self.observation_residency_scope,
            "bounded_sparse_sampled_observations_only": True,
            "dense_frame_observation_streaming_implemented": False,
            "dense_F_observation_residency_closed": False,
            "dense_F_requires_replayable_observation_source": True,
            "dense_F_replay_source_module_available": True,
            "dense_F_replay_source_integrated_with_legacy_bundle": False,
            "structural_track_request_sample_independent": True,
            "structural_track_request_fixed_endpoint_witness_count": min(
                2,
                self.frame_count,
            ),
            "cold_camera_content_certification_owned_by_outer_step": True,
            "warm_bundle_checks_rehash_full_camera_grid": False,
            "selected_observation_python_metadata_scaling": "O(selected observations in bounded bundle)",
            "frame_scaling": (
                "legacy bundle: O(VF) camera metadata + O(F) scalar times + "
                "bounded O(K_selected) observation/ray metadata; separate "
                "dense replay source exists but is not integrated here"
            ),
            "world_site_storage_shared_across_bundle_tracks": True,
            "provider_retains_world_initializer": False,
            "initializer_contract_receipt_only": True,
            "initializer_template_tensor_bytes_retained_by_provider": 0,
            "equal_rank_source_byte_report_is_conservative_per_track_accounting": True,
        }

    def assert_warm_current(self) -> None:
        """Check sealed identities/generations without an ``O(VF)`` rehash.

        The full camera-path and target-residency contents are certified by
        :meth:`assert_current` once at the outer replay/step boundary.  Between
        cold certifications the caller must not mutate the camera grid.  The
        warm path still fails closed on provider/source/factory identity,
        component generation, grid layout, and world tensor identity/version.
        """

        if (
            self._seal is not _PROVIDER_SEAL
            or self.provenance != PROVIDER_PROVENANCE
            or id(self.target_provider) != self._target_provider_identity
            or id(self.target_provider.source) != self._target_source_identity
            or id(self.ray_provider) != self._ray_provider_identity
            or id(self.program_factory) != self._factory_identity
            or id(self.ray_provider.cameras) != self._camera_grid_identity
            or self.ray_provider.view_count != self.view_count
            or self.ray_provider.frame_count != self.frame_count
            or len(self.view_static_camera_path_certified) != self.view_count
            or any(
                not isinstance(value, bool)
                for value in self.view_static_camera_path_certified
            )
            or self.persistent_target_tensor_bytes != 0
            or self.persistent_dense_ray_tensor_bytes != 0
            or self.dense_track_frame_tensor_bytes != 0
            or self.observation_residency_scope != BOUNDED_OBSERVATION_SCOPE
            or self.dense_frame_observation_streaming_implemented
        ):
            raise ValueError("paper kinetic lazy provider identity/memory contract changed")
        if self.world.initializer_generation_digest != self.initializer_generation_digest:
            raise ValueError("paper kinetic world initializer receipt changed")
        if _component_contract(self.program_factory, name="program_factory") != (
            self.factory_provenance,
            self.factory_generation_digest,
        ):
            raise ValueError("paper kinetic program factory provenance changed")
        self.world.assert_warm_current()
        if self.world.dataset_generation_digest != self.dataset_generation_digest:
            raise ValueError("paper kinetic world belongs to a different dataset")
        if self.generation_digest != _provider_digest(self):
            raise ValueError("paper kinetic lazy provider generation changed")

    def assert_current(self) -> None:
        """Cold full-content certification before a replay/step begins."""

        self.assert_warm_current()
        if _target_residency_digest(self.target_provider) != self.target_residency_digest:
            raise ValueError("paper kinetic target source residency/provenance changed")
        camera_grid_digest, path_digests, static_certificates = _camera_grid_digests(
            self.ray_provider,
            height=self.height,
            width=self.width,
            frame_times=self.frame_times,
        )
        if (
            camera_grid_digest != self.camera_grid_digest
            or path_digests != self.view_camera_path_digests
            or static_certificates != self.view_static_camera_path_certified
        ):
            raise ValueError("paper kinetic calibrated camera records changed")
        self.world.assert_current()

    def iter_spatial_bundles(
        self,
        observations: Sequence[PaperKineticObservation],
        *,
        device: torch.device | str,
        construction_lifetime_slot: (
            PaperKineticLazyBundleConstructionLifetimeSlot | None
        ) = None,
    ) -> Iterator[PaperKineticLazyProgramBundle]:
        """Sort a materialized convenience request, then stream bounded bundles.

        Publication training should prefer
        :meth:`iter_canonical_spatial_bundles` so a full ``P x K`` Python
        observation list is never materialized merely to sort it.
        """

        self.assert_warm_current()
        canonical = _validate_and_canonicalize_observations(self, observations)
        yield from self.iter_canonical_spatial_bundles(
            canonical,
            device=device,
            construction_lifetime_slot=construction_lifetime_slot,
        )

    def iter_canonical_spatial_bundles(
        self,
        observations: Iterable[PaperKineticObservation],
        *,
        device: torch.device | str,
        construction_lifetime_slot: (
            PaperKineticLazyBundleConstructionLifetimeSlot | None
        ) = None,
    ) -> Iterator[PaperKineticLazyProgramBundle]:
        """Stream bounded bundles without retaining them inside the provider.

        A caller can still retain yielded bundles.  The one-live-bundle peak is
        therefore enforced by the outer step coordinator, not by this Python
        generator alone.  The caller must perform one ``assert_current()``
        cold certification at its outer replay/step boundary; this iterator
        uses only warm identity/generation checks so each bundle does not
        repeat an ``O(VF)`` camera-path hash.
        """

        self.assert_warm_current()
        resolved_device = torch.device(device)
        if construction_lifetime_slot is not None:
            if not isinstance(
                construction_lifetime_slot,
                PaperKineticLazyBundleConstructionLifetimeSlot,
            ):
                raise TypeError("construction_lifetime_slot has the wrong type")
            construction_lifetime_slot.assert_current()
        elif resolved_device.type != "cpu":
            raise RuntimeError(
                "accelerator lazy bundle production requires a caller-visible "
                "construction lifetime slot"
            )
        partitions = _iter_canonical_observation_partitions(self, observations)
        for bundle_index, partition in enumerate(partitions):
            self.assert_warm_current()
            bundle = _materialize_lazy_bundle(
                self,
                bundle_index=bundle_index,
                observations=partition,
                device=resolved_device,
                construction_lifetime_slot=construction_lifetime_slot,
            )
            yield bundle
            del bundle


def observations_from_spacetime_batch(
    batch: SpacetimeBatch,
    *,
    pixel_indices_by_batch_position: Sequence[Sequence[int]],
) -> tuple[PaperKineticObservation, ...]:
    """Expand only explicitly selected pixels for each paper observation."""

    if not isinstance(batch, SpacetimeBatch):
        raise TypeError("paper kinetic observations require a SpacetimeBatch")
    selections = tuple(tuple(int(pixel) for pixel in pixels) for pixels in pixel_indices_by_batch_position)
    if len(selections) != len(batch.samples):
        raise ValueError("paper kinetic pixel selections must cover every batch position")
    observations: list[PaperKineticObservation] = []
    next_id = 0
    for sample, pixels in zip(batch.samples, selections, strict=True):
        if not pixels:
            raise ValueError("paper kinetic pixel selections must be nonempty")
        if len(set(pixels)) != len(pixels):
            raise ValueError("paper kinetic pixels must be unique within each observation")
        for pixel in pixels:
            observations.append(
                PaperKineticObservation(
                    observation_id=next_id,
                    view_index=sample.view_index,
                    frame_index=sample.frame_index,
                    pixel_index=pixel,
                )
            )
            next_id += 1
    return tuple(observations)


def iter_canonical_observations_from_spacetime_batch(
    batch: SpacetimeBatch,
    *,
    pixel_indices_by_batch_position: Sequence[Iterable[int]],
    image_pixel_count: int,
) -> Iterator[PaperKineticObservation]:
    """K-way merge selected pixels into canonical order using ``O(K)`` state.

    ``range(image_pixel_count)`` is the intended full-image selection; callers
    need not construct one Python integer per pixel.  Every input pixel stream
    must itself be strictly increasing.  The stable observation id is
    ``batch_position * image_pixel_count + pixel``.
    """

    if not isinstance(batch, SpacetimeBatch):
        raise TypeError("paper kinetic observations require a SpacetimeBatch")
    _require_positive_int(image_pixel_count, name="image_pixel_count")
    selections = tuple(pixel_indices_by_batch_position)
    if len(selections) != len(batch.samples):
        raise ValueError("paper kinetic pixel selections must cover every batch position")

    positions_by_view: dict[int, list[int]] = {}
    for position, sample in enumerate(batch.samples):
        positions_by_view.setdefault(sample.view_index, []).append(position)
    for view_index in sorted(positions_by_view):
        heap: list[tuple[int, int, Iterator[int]]] = []
        for position in positions_by_view[view_index]:
            iterator = _validated_pixel_iterator(
                selections[position],
                image_pixel_count=image_pixel_count,
                batch_position=position,
            )
            try:
                pixel = next(iterator)
            except StopIteration as error:
                raise ValueError("paper kinetic pixel selections must be nonempty") from error
            heapq.heappush(heap, (pixel, position, iterator))

        while heap:
            pixel = heap[0][0]
            same_pixel: list[tuple[int, Iterator[int]]] = []
            while heap and heap[0][0] == pixel:
                _pixel, position, iterator = heapq.heappop(heap)
                same_pixel.append((position, iterator))
            for position, _iterator in sorted(
                same_pixel,
                key=lambda item: (
                    batch.samples[item[0]].frame_index,
                    item[0],
                ),
            ):
                sample = batch.samples[position]
                yield PaperKineticObservation(
                    observation_id=position * image_pixel_count + pixel,
                    view_index=view_index,
                    frame_index=sample.frame_index,
                    pixel_index=pixel,
                )
            for position, iterator in same_pixel:
                try:
                    next_pixel = next(iterator)
                except StopIteration:
                    continue
                heapq.heappush(heap, (next_pixel, position, iterator))


def prepare_paper_kinetic_lazy_program_bundle_provider(
    *,
    dataset_generation_digest: str,
    target_provider: PowerFoamTargetProvider,
    ray_provider: PowerFoamRayProvider,
    frame_times: Sequence[float] | torch.Tensor,
    height: int,
    width: int,
    maximum_tracks_per_bundle: int,
    maximum_observations_per_bundle: int,
    maximum_rows_per_native_block: int,
    world_initializer: PaperKineticWorldInitializer,
    program_factory: PaperKineticTrackProgramFactory,
) -> PaperKineticLazyProgramBundleProvider:
    """Cold-seal dataset/world metadata; drop the initializer after return."""

    _require_sha256(dataset_generation_digest, name="dataset_generation_digest")
    if not isinstance(target_provider, PowerFoamTargetProvider):
        raise TypeError("paper kinetic provider requires PowerFoamTargetProvider")
    if not isinstance(ray_provider, PowerFoamRayProvider):
        raise TypeError("paper kinetic provider requires PowerFoamRayProvider")
    _validate_provider_grid(target_provider, ray_provider, height=height, width=width)
    for name, value in (
        ("maximum_tracks_per_bundle", maximum_tracks_per_bundle),
        ("maximum_observations_per_bundle", maximum_observations_per_bundle),
        ("maximum_rows_per_native_block", maximum_rows_per_native_block),
    ):
        _require_positive_int(value, name=name)
    times = _frame_time_tuple(frame_times, frame_count=target_provider.frame_count)
    target_digest = _target_residency_digest(target_provider)
    camera_grid_digest, path_digests, static_certificates = _camera_grid_digests(
        ray_provider,
        height=height,
        width=width,
        frame_times=times,
    )
    initializer_contract = _component_contract(world_initializer, name="world_initializer")
    factory_contract = _component_contract(program_factory, name="program_factory")
    init_request = PaperKineticWorldInitializationRequest(
        dataset_generation_digest=dataset_generation_digest,
        camera_grid_digest=camera_grid_digest,
        view_count=target_provider.view_count,
        frame_count=target_provider.frame_count,
        height=int(height),
        width=int(width),
        initializer_generation_digest=initializer_contract[1],
        generation_digest=_digest_parts(
            PROVIDER_PROVENANCE,
            "world-init-request",
            dataset_generation_digest,
            camera_grid_digest,
            target_provider.view_count,
            target_provider.frame_count,
            int(height),
            int(width),
            initializer_contract[1],
        ),
    )
    init_request.assert_self_consistent()
    sites = world_initializer.initialize_world(init_request)
    if not isinstance(sites, AffineKineticPowerSites):
        raise TypeError("paper kinetic world initializer must return AffineKineticPowerSites")
    if _component_contract(world_initializer, name="world_initializer") != initializer_contract:
        raise ValueError("paper kinetic world initializer changed during initialization")
    site_digest = _site_content_digest(sites)
    world = PaperKineticWorldSnapshot(
        sites=sites,
        dataset_generation_digest=dataset_generation_digest,
        initializer_generation_digest=initializer_contract[1],
        sites_content_digest=site_digest,
        generation_digest=_digest_parts(
            WORLD_SNAPSHOT_PROVENANCE,
            dataset_generation_digest,
            initializer_contract[1],
            site_digest,
            sites.site_count,
        ),
        _site_tensor_identities=tuple(id(tensor) for tensor in _site_tensors(sites)),
        _site_tensor_signatures=tuple(_tensor_signature(tensor) for tensor in _site_tensors(sites)),
    )
    world.assert_current()
    provisional = PaperKineticLazyProgramBundleProvider(
        dataset_generation_digest=dataset_generation_digest,
        target_provider=target_provider,
        ray_provider=ray_provider,
        frame_times=times,
        height=int(height),
        width=int(width),
        maximum_tracks_per_bundle=int(maximum_tracks_per_bundle),
        maximum_observations_per_bundle=int(maximum_observations_per_bundle),
        maximum_rows_per_native_block=int(maximum_rows_per_native_block),
        program_factory=program_factory,
        world=world,
        target_residency_digest=target_digest,
        camera_grid_digest=camera_grid_digest,
        view_camera_path_digests=path_digests,
        view_static_camera_path_certified=static_certificates,
        initializer_provenance=initializer_contract[0],
        initializer_generation_digest=initializer_contract[1],
        factory_provenance=factory_contract[0],
        factory_generation_digest=factory_contract[1],
        generation_digest="",
        _target_provider_identity=id(target_provider),
        _target_source_identity=id(target_provider.source),
        _ray_provider_identity=id(ray_provider),
        _factory_identity=id(program_factory),
        _camera_grid_identity=id(ray_provider.cameras),
        _seal=_PROVIDER_SEAL,
    )
    result = _replace_provider_digest(provisional, _provider_digest(provisional))
    result.assert_current()
    return result


def compile_paper_kinetic_cpu_ragged_sampler(
    provider: PaperKineticLazyProgramBundleProvider,
    *,
    view_index: int,
    track_ids: Sequence[int],
) -> PaperKineticRowRaggedSampler:
    """Compile one bounded contiguous spatial request without a device map.

    This is the frame-free structural seam for dense fixed-site training.  It
    compiles each ``(view, pixel)`` program from the provider's fixed endpoint
    witnesses, validates the returned program against those calibrated rays,
    lowers every chart on CPU, and returns the cold-sealed ragged sampler.
    Full-camera-path replay remains the responsibility of compiled-artifact
    cold admission, where the sampler is bound to an observation-invariant
    cache key.

    Unlike the legacy sparse-bundle path, this function constructs no selected
    observation records beyond the fixed endpoint witnesses and never creates
    a union-local device mapping.
    """

    selected_tracks = _bounded_contiguous_cpu_sampler_track_ids(
        provider,
        view_index=view_index,
        track_ids=track_ids,
    )
    provider.assert_current()
    sampler, _program_digests, _request_digests, _compile_receipt = (
        _compile_cpu_ragged_sampler(
            provider,
            view_index=view_index,
            track_ids=selected_tracks,
            replay_records_by_track=None,
        )
    )
    sampler.assert_cold_current()
    return sampler


def _materialize_lazy_bundle(
    provider: PaperKineticLazyProgramBundleProvider,
    *,
    bundle_index: int,
    observations: tuple[PaperKineticObservation, ...],
    device: torch.device,
    construction_lifetime_slot: (
        PaperKineticLazyBundleConstructionLifetimeSlot | None
    ),
) -> PaperKineticLazyProgramBundle:
    view_index = observations[0].view_index
    track_ids = tuple(sorted({observation.pixel_index for observation in observations}))
    records = tuple(_materialize_ray_record(provider, observation) for observation in observations)
    records_by_track = {
        track_id: tuple(record for record in records if record.observation.pixel_index == track_id)
        for track_id in track_ids
    }
    sampler, program_digests, request_digests, compile_receipt = (
        _compile_cpu_ragged_sampler(
            provider,
            view_index=view_index,
            track_ids=track_ids,
            replay_records_by_track=records_by_track,
        )
    )
    construction_lifetime = (
        prepare_paper_kinetic_union_local_spatial_bundle_construction_lifetime(
            sampler,
            track_ids=track_ids,
            device=device,
        )
    )
    if construction_lifetime_slot is not None:
        construction_lifetime_slot.install(construction_lifetime)
    elif device.type != "cpu":
        raise RuntimeError(
            "accelerator lazy bundle construction lifetime is not caller-visible"
        )
    spatial = prepare_paper_kinetic_union_local_spatial_bundle(
        sampler,
        track_ids=track_ids,
        device=device,
        construction_lifetime=construction_lifetime,
    )
    provisional = PaperKineticLazyProgramBundle(
        provider_generation_digest=provider.generation_digest,
        bundle_index=bundle_index,
        view_index=view_index,
        track_ids=track_ids,
        observations=records,
        sampler=sampler,
        spatial_bundle=spatial,
        program_generation_digests=program_digests,
        factory_request_generation_digests=request_digests,
        compile_receipt=compile_receipt,
        generation_digest="",
        _provider_identity=id(provider),
        _sampler_identity=id(sampler),
        _spatial_bundle_identity=id(spatial),
        _seal=_BUNDLE_SEAL,
    )
    result = _replace_bundle_digest(provisional, _bundle_digest(provisional))
    if device.type == "cpu":
        result.assert_cold_current(provider)
        if construction_lifetime_slot is not None:
            construction_lifetime_slot.complete(construction_lifetime)
    else:
        # The outer coordinator owns the already-registered completion epoch.
        # Leave the slot active and both copy endpoints rooted until it consumes
        # that receipt, commits predecessor release, cold-admits the bundle, and
        # completes this slot.
        result.assert_accelerator_transfer_pending(provider)
    return result


def _compile_cpu_ragged_sampler(
    provider: PaperKineticLazyProgramBundleProvider,
    *,
    view_index: int,
    track_ids: tuple[int, ...],
    replay_records_by_track: Mapping[
        int,
        tuple[PaperKineticObservationRayRecord, ...],
    ]
    | None,
) -> tuple[
    PaperKineticRowRaggedSampler,
    tuple[str, ...],
    tuple[str, ...],
    PaperKineticLazyProgramCompileReceipt,
]:
    """Compile the CPU sampler shared by direct and legacy bundle paths."""

    if not track_ids:
        raise ValueError("paper kinetic CPU sampler requires at least one track")
    if replay_records_by_track is not None and tuple(sorted(replay_records_by_track)) != track_ids:
        raise ValueError("paper kinetic replay records do not cover the CPU sampler tracks")
    cameras = provider.ray_provider.cameras[view_index]
    sources: list[KineticNativeEqualRankChartSource] = []
    program_digests: list[str] = []
    request_digests: list[str] = []
    compile_receipts = _LazyCompileReceiptAccumulator(
        provider_generation_digest=provider.generation_digest,
        view_index=view_index,
        track_ids=track_ids,
    )
    for track_id in track_ids:
        structural_witnesses = _structural_ray_witnesses(
            provider,
            view_index=view_index,
            pixel_index=track_id,
        )
        request = PaperKineticTrackProgramRequest(
            world=provider.world,
            dataset_generation_digest=provider.dataset_generation_digest,
            factory_generation_digest=provider.factory_generation_digest,
            view_index=view_index,
            pixel_index=track_id,
            height=provider.height,
            width=provider.width,
            frame_times=provider.frame_times,
            cameras=cameras,
            camera_path_digest=provider.view_camera_path_digests[view_index],
            static_camera_path_certified=(
                provider.view_static_camera_path_certified[view_index]
            ),
            observations=structural_witnesses,
            generation_digest="",
        )
        request = _replace_track_request_digest(request, _track_request_digest(request))
        request.assert_self_consistent()
        program = provider.program_factory.compile_track(request)
        if _component_contract(provider.program_factory, name="program_factory") != (
            provider.factory_provenance,
            provider.factory_generation_digest,
        ):
            raise ValueError("paper kinetic program factory changed during compilation")
        replay_validation = _prepare_track_replay_validation(
            request,
            (
                structural_witnesses
                if replay_records_by_track is None
                else replay_records_by_track[track_id]
            ),
        )
        program = _validate_compiled_track_program(
            provider,
            request,
            replay_validation,
            program,
        )
        compile_receipts.add(
            track_id=track_id,
            request=request,
            program=program,
            program_factory=provider.program_factory,
        )
        if _component_contract(provider.program_factory, name="program_factory") != (
            provider.factory_provenance,
            provider.factory_generation_digest,
        ):
            raise ValueError(
                "paper kinetic program factory changed during compile accounting"
            )
        topology = lower_kinetic_multichart_to_native_topology(program)
        sources.extend(
            kinetic_native_equal_rank_chart_sources_for_track(
                track_id,
                program,
                lowering=topology,
            )
        )
        program_digests.append(program.generation_digest)
        request_digests.append(request.generation_digest)
    source_tuple = tuple(sources)
    lowering = lower_kinetic_native_equal_rank_buckets(
        source_tuple,
        maximum_rows_per_block=provider.maximum_rows_per_native_block,
    )
    sampler = prepare_paper_kinetic_row_ragged_sampler(
        view_index=view_index,
        lowering=lowering,
        sources=source_tuple,
    )
    sampler.assert_cold_current()
    provider.assert_warm_current()
    program_digest_tuple = tuple(program_digests)
    request_digest_tuple = tuple(request_digests)
    compile_receipt = compile_receipts.finish(
        program_generation_digests=program_digest_tuple,
        request_generation_digests=request_digest_tuple,
    )
    return sampler, program_digest_tuple, request_digest_tuple, compile_receipt


class _LazyCompileReceiptAccumulator:
    """Fold live per-track compiler receipts into constant-memory scalars."""

    _INTEGER_FIELDS = (
        "root_complement_witness_count",
        "candidate_source_attempt_count",
        "all_site_witness_check_count",
        "unique_pair_difference_count",
    )
    _BOOLEAN_FIELDS = (
        "per_witness_candidate_bound_verified",
        "exhaustive_triple_enumeration_used",
        "requested_frame_sampling_used",
    )

    def __init__(
        self,
        *,
        provider_generation_digest: str,
        view_index: int,
        track_ids: tuple[int, ...],
    ) -> None:
        _require_sha256(
            provider_generation_digest,
            name="compile_receipt.provider_generation_digest",
        )
        _require_nonnegative_int(view_index, name="compile_receipt.view_index")
        if not track_ids:
            raise ValueError("lazy compiler receipt requires at least one track")
        self._provider_generation_digest = provider_generation_digest
        self._view_index = view_index
        self._track_ids = track_ids
        self._compile_track_count = 0
        self._compiler_work_receipt_count = 0
        self._root_complement_witness_count = 0
        self._candidate_source_attempt_count = 0
        self._all_site_witness_check_count = 0
        self._unique_pair_difference_count = 0
        self._per_witness_candidate_bound_verified = True
        self._exhaustive_triple_enumeration_used = False
        self._requested_frame_sampling_used = False
        self._receipt_provenance: str | None = None
        self._receipt_provenance_mixed = False
        self._finished = False
        self._chain_digest = _digest_parts(
            COMPILE_RECEIPT_PROVENANCE,
            "chain-root",
            provider_generation_digest,
            view_index,
            track_ids,
        )

    def add(
        self,
        *,
        track_id: int,
        request: PaperKineticTrackProgramRequest,
        program: KineticMultiChartP0Program,
        program_factory: Any,
    ) -> None:
        if self._finished:
            raise ValueError("lazy compiler receipt accumulator is already sealed")
        ordinal = self._compile_track_count
        if ordinal >= len(self._track_ids) or track_id != self._track_ids[ordinal]:
            raise ValueError("lazy compiler receipt track order changed")
        request.assert_self_consistent()
        program.assert_current()
        extractor = getattr(program_factory, "compile_accounting", None)
        if extractor is None:
            receipt_payload: object = "unavailable"
        else:
            if not callable(extractor):
                raise TypeError("program factory compile_accounting must be callable")
            receipt = extractor(program)
            receipt_payload = self._consume_receipt(receipt, program=program)
        self._chain_digest = _digest_parts(
            COMPILE_RECEIPT_PROVENANCE,
            "chain-link",
            self._chain_digest,
            ordinal,
            track_id,
            request.generation_digest,
            program.generation_digest,
            receipt_payload,
        )
        self._compile_track_count += 1

    def _consume_receipt(
        self,
        receipt: object,
        *,
        program: KineticMultiChartP0Program,
    ) -> tuple[tuple[str, object], ...]:
        if not isinstance(receipt, Mapping):
            raise TypeError("program factory compile_accounting must return a mapping")
        payload = tuple(
            (key, value)
            for key, value in receipt.items()
            if key
            not in {
                "compiler_work_receipt_digest",
                "compiler_work_receipt_provenance",
            }
        )
        if any(
            not isinstance(key, str)
            or isinstance(value, torch.Tensor)
            or not isinstance(value, (int, bool, str))
            for key, value in receipt.items()
        ):
            raise TypeError("compiler accounting receipt must be tensor-free scalars")
        if receipt.get("compile_track_count") != 1:
            raise ValueError("compiler accounting receipt must describe one track")
        if (
            receipt.get("compiler_program_generation_digest")
            != program.generation_digest
        ):
            raise ValueError("compiler accounting receipt belongs to another program")
        for name in self._INTEGER_FIELDS:
            value = receipt.get(name)
            if type(value) is not int or value < 0:
                raise ValueError(f"compiler accounting {name} must be nonnegative")
        for name in self._BOOLEAN_FIELDS:
            if type(receipt.get(name)) is not bool:
                raise ValueError(f"compiler accounting {name} must be boolean")
        provenance = receipt.get("compiler_work_receipt_provenance")
        digest = receipt.get("compiler_work_receipt_digest")
        if not isinstance(provenance, str) or not provenance.strip():
            raise ValueError("compiler accounting receipt provenance is missing")
        if not isinstance(digest, str):
            raise ValueError("compiler accounting receipt digest is missing")
        _require_sha256(digest, name="compiler_work_receipt_digest")
        if digest != _digest_parts(provenance, payload):
            raise ValueError("compiler accounting receipt digest changed")

        if self._receipt_provenance is None:
            self._receipt_provenance = provenance
        elif self._receipt_provenance != provenance:
            self._receipt_provenance_mixed = True
        self._compiler_work_receipt_count += 1
        self._root_complement_witness_count += int(
            receipt["root_complement_witness_count"]
        )
        self._candidate_source_attempt_count += int(
            receipt["candidate_source_attempt_count"]
        )
        self._all_site_witness_check_count += int(
            receipt["all_site_witness_check_count"]
        )
        self._unique_pair_difference_count += int(
            receipt["unique_pair_difference_count"]
        )
        self._per_witness_candidate_bound_verified = (
            self._per_witness_candidate_bound_verified
            and bool(receipt["per_witness_candidate_bound_verified"])
        )
        self._exhaustive_triple_enumeration_used = (
            self._exhaustive_triple_enumeration_used
            or bool(receipt["exhaustive_triple_enumeration_used"])
        )
        self._requested_frame_sampling_used = (
            self._requested_frame_sampling_used
            or bool(receipt["requested_frame_sampling_used"])
        )
        return tuple(receipt.items())

    def finish(
        self,
        *,
        program_generation_digests: tuple[str, ...],
        request_generation_digests: tuple[str, ...],
    ) -> PaperKineticLazyProgramCompileReceipt:
        if self._finished:
            raise ValueError("lazy compiler receipt accumulator is already sealed")
        if (
            self._compile_track_count != len(self._track_ids)
            or self._compile_track_count != len(program_generation_digests)
            or self._compile_track_count != len(request_generation_digests)
        ):
            raise ArithmeticError("lazy compiler receipt lost a compiled track")
        self._chain_digest = _digest_parts(
            COMPILE_RECEIPT_PROVENANCE,
            "chain-seal",
            self._chain_digest,
            self._compile_track_count,
            program_generation_digests,
            request_generation_digests,
        )
        self._finished = True
        provenance = (
            "mixed"
            if self._receipt_provenance_mixed
            else self._receipt_provenance or "unavailable"
        )
        complete = (
            self._compile_track_count > 0
            and self._compiler_work_receipt_count == self._compile_track_count
            and provenance not in {"unavailable", "mixed"}
        )
        provisional = PaperKineticLazyProgramCompileReceipt(
            provider_generation_digest=self._provider_generation_digest,
            view_index=self._view_index,
            compile_track_count=self._compile_track_count,
            compiler_work_receipt_count=self._compiler_work_receipt_count,
            compiler_work_receipt_chain_link_count=self._compile_track_count,
            root_complement_witness_count=(
                self._root_complement_witness_count
            ),
            candidate_source_attempt_count=(
                self._candidate_source_attempt_count
            ),
            all_site_witness_check_count=self._all_site_witness_check_count,
            unique_pair_difference_count=self._unique_pair_difference_count,
            per_witness_candidate_bound_verified=(
                self._compiler_work_receipt_count > 0
                and self._per_witness_candidate_bound_verified
            ),
            exhaustive_triple_enumeration_used=(
                self._exhaustive_triple_enumeration_used
            ),
            requested_frame_sampling_used=self._requested_frame_sampling_used,
            compiler_accounting_complete=complete,
            all_track_receipt_digests_verified=(
                self._compile_track_count > 0
                and self._compiler_work_receipt_count == self._compile_track_count
            ),
            compiler_work_receipt_provenance=provenance,
            compiler_work_receipt_chain_digest=self._chain_digest,
            generation_digest="",
        )
        result = replace(
            provisional,
            generation_digest=_compile_receipt_digest(
                provisional,
                track_ids=self._track_ids,
                program_generation_digests=program_generation_digests,
                request_generation_digests=request_generation_digests,
            ),
        )
        result.assert_current(
            track_ids=self._track_ids,
            program_generation_digests=program_generation_digests,
            request_generation_digests=request_generation_digests,
        )
        return result


def _bounded_contiguous_cpu_sampler_track_ids(
    provider: PaperKineticLazyProgramBundleProvider,
    *,
    view_index: int,
    track_ids: Sequence[int],
) -> tuple[int, ...]:
    if not isinstance(provider, PaperKineticLazyProgramBundleProvider):
        raise TypeError("paper kinetic CPU sampler requires a lazy program provider")
    _require_nonnegative_int(view_index, name="view_index")
    if view_index >= provider.view_count:
        raise IndexError("paper kinetic CPU sampler view leaves the provider")
    if not isinstance(track_ids, Sequence):
        raise TypeError("paper kinetic CPU sampler track_ids must be a bounded sequence")
    track_count = len(track_ids)
    if track_count < 1 or track_count > provider.maximum_tracks_per_bundle:
        raise MemoryError(
            "paper kinetic CPU sampler track request exceeds the provider's bounded spatial capacity"
        )
    selected = tuple(track_ids)
    for track_id in selected:
        _require_nonnegative_int(track_id, name="track_id")
    if selected[-1] >= provider.height * provider.width:
        raise IndexError("paper kinetic CPU sampler track leaves the stage image")
    if any(
        right != left + 1
        for left, right in zip(selected, selected[1:], strict=False)
    ):
        raise ValueError("paper kinetic CPU sampler tracks must be contiguous and increasing")
    return selected


def _validate_compiled_track_program(
    provider: PaperKineticLazyProgramBundleProvider,
    request: PaperKineticTrackProgramRequest,
    replay_validation: PaperKineticTrackReplayValidation,
    program: KineticMultiChartP0Program,
) -> KineticMultiChartP0Program:
    if not isinstance(program, KineticMultiChartP0Program):
        raise TypeError("paper kinetic program factory returned the wrong type")
    program.assert_current()
    provider.world.assert_warm_current()
    replay_validation.assert_self_consistent(request)
    if _site_content_digest(program.binding.sites) != provider.world.sites_content_digest:
        raise ValueError("paper kinetic program was compiled from a different world")
    if program.binding.sites is not provider.world.sites:
        program = replace(
            program,
            binding=replace(program.binding, sites=provider.world.sites),
        )
        program.assert_current()
    domain_min = float(program.binding.program.t_min)
    domain_max = float(program.binding.program.t_max)
    if domain_min > provider.frame_times[0] or domain_max < provider.frame_times[-1]:
        raise ValueError("paper kinetic track program does not cover the dataset time domain")
    coefficients = program.binding.ray_coefficients
    for record in (*request.observations, *replay_validation.observations):
        time = torch.tensor(record.sample_time, dtype=torch.float64)
        replay = torch.cat(
            (
                coefficients[0:3] + time * coefficients[3:6],
                coefficients[6:9] + time * coefficients[9:12],
            )
        )
        expected = torch.tensor(record.ray_origin_direction, dtype=torch.float64)
        if not torch.allclose(replay, expected, rtol=0.0, atol=1.0e-10):
            raise ValueError("paper kinetic affine camera program does not reproduce a selected calibrated ray")
    return program


def _structural_ray_witnesses(
    provider: PaperKineticLazyProgramBundleProvider,
    *,
    view_index: int,
    pixel_index: int,
) -> tuple[PaperKineticObservationRayRecord, ...]:
    """Fixed endpoint witnesses independent of logical sampled observations."""

    return tuple(
        _materialize_ray_record(
            provider,
            PaperKineticObservation(
                observation_id=(
                    (view_index * provider.height * provider.width + pixel_index) * provider.frame_count + frame_index
                ),
                view_index=view_index,
                frame_index=frame_index,
                pixel_index=pixel_index,
            ),
        )
        for frame_index in dict.fromkeys((0, provider.frame_count - 1))
    )


def _prepare_track_replay_validation(
    request: PaperKineticTrackProgramRequest,
    observations: tuple[PaperKineticObservationRayRecord, ...],
) -> PaperKineticTrackReplayValidation:
    provisional = PaperKineticTrackReplayValidation(
        structural_request_generation_digest=request.generation_digest,
        view_index=request.view_index,
        pixel_index=request.pixel_index,
        observations=observations,
        generation_digest="",
    )
    result = replace(
        provisional,
        generation_digest=_track_replay_validation_digest(provisional),
    )
    result.assert_self_consistent(request)
    return result


def _materialize_ray_record(
    provider: PaperKineticLazyProgramBundleProvider,
    observation: PaperKineticObservation,
) -> PaperKineticObservationRayRecord:
    camera = _scaled_cpu_camera(
        provider.ray_provider.cameras[observation.view_index][observation.frame_index],
        source_height=provider.ray_provider.height,
        source_width=provider.ray_provider.width,
        target_height=provider.height,
        target_width=provider.width,
    )
    pixel = torch.tensor([observation.pixel_index], dtype=torch.int64, device="cpu")
    origins, directions = build_camera_rays_at_pixels(
        camera,
        pixel,
        height=provider.height,
        width=provider.width,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    ray = tuple(float(value) for value in torch.cat((origins[0], directions[0])).tolist())
    camera_digest = _camera_digest(camera)
    sample_time = provider.frame_times[observation.frame_index]
    result = PaperKineticObservationRayRecord(
        observation=observation,
        sample_time=sample_time,
        ray_origin_direction=ray,
        camera_record_digest=camera_digest,
        generation_digest=_ray_record_digest(
            observation,
            sample_time=sample_time,
            ray=ray,
            camera_record_digest=camera_digest,
        ),
    )
    result.assert_self_consistent()
    return result


def _validate_and_canonicalize_observations(
    provider: PaperKineticLazyProgramBundleProvider,
    observations: Sequence[PaperKineticObservation],
) -> tuple[PaperKineticObservation, ...]:
    selected = tuple(observations)
    if not selected or any(not isinstance(value, PaperKineticObservation) for value in selected):
        raise ValueError("paper kinetic bundle request requires observations")
    identities = tuple(observation.sample_identity for observation in selected)
    if len(set(identities)) != len(identities):
        raise ValueError("paper kinetic bundle request contains duplicate observation identities")
    for observation in selected:
        _validate_observation(provider, observation)
    return tuple(
        sorted(
            selected,
            key=lambda value: (
                value.view_index,
                value.pixel_index,
                value.frame_index,
                value.observation_id,
            ),
        )
    )


def _validate_observation(
    provider: PaperKineticLazyProgramBundleProvider,
    observation: PaperKineticObservation,
) -> None:
    if not isinstance(observation, PaperKineticObservation):
        raise TypeError("paper kinetic observation stream contains the wrong type")
    if observation.view_index >= provider.view_count:
        raise IndexError("paper kinetic observation view leaves the dataset")
    if observation.frame_index >= provider.frame_count:
        raise IndexError("paper kinetic observation frame leaves the dataset")
    if observation.pixel_index >= provider.height * provider.width:
        raise IndexError("paper kinetic observation pixel leaves the stage image")


def _iter_canonical_observation_partitions(
    provider: PaperKineticLazyProgramBundleProvider,
    observations: Iterable[PaperKineticObservation],
) -> Iterator[tuple[PaperKineticObservation, ...]]:
    """Greedy one-pass partition with one track of lookahead."""

    current_bundle: list[PaperKineticObservation] = []
    current_track: list[PaperKineticObservation] = []
    current_track_identity: tuple[int, int] | None = None
    current_bundle_track_count = 0
    previous_key: tuple[int, int, int, int] | None = None
    saw_observation = False

    def flush_track() -> tuple[PaperKineticObservation, ...] | None:
        nonlocal current_bundle, current_track, current_bundle_track_count
        if not current_track:
            return None
        if len(current_track) > provider.maximum_observations_per_bundle:
            raise ValueError("one paper kinetic track exceeds maximum_observations_per_bundle")
        track_view = current_track[0].view_index
        bundle_view = current_bundle[0].view_index if current_bundle else track_view
        would_overflow = bool(current_bundle) and (
            track_view != bundle_view
            or current_bundle_track_count + 1 > provider.maximum_tracks_per_bundle
            or len(current_bundle) + len(current_track) > provider.maximum_observations_per_bundle
        )
        completed = tuple(current_bundle) if would_overflow else None
        if would_overflow:
            current_bundle = []
            current_bundle_track_count = 0
        current_bundle.extend(current_track)
        current_bundle_track_count += 1
        current_track = []
        if completed is None and (
            current_bundle_track_count == provider.maximum_tracks_per_bundle
            or len(current_bundle) == provider.maximum_observations_per_bundle
        ):
            completed = tuple(current_bundle)
            current_bundle = []
            current_bundle_track_count = 0
        return completed

    for observation in observations:
        saw_observation = True
        _validate_observation(provider, observation)
        key = (
            observation.view_index,
            observation.pixel_index,
            observation.frame_index,
            observation.observation_id,
        )
        if previous_key is not None and key <= previous_key:
            raise ValueError(
                "paper kinetic streaming observations must be strictly canonical by (view,pixel,frame,observation_id)"
            )
        previous_key = key
        if current_track_identity is not None and observation.track_identity != current_track_identity:
            completed = flush_track()
            if completed is not None:
                yield completed
        current_track_identity = observation.track_identity
        current_track.append(observation)
    completed = flush_track()
    if completed is not None:
        yield completed
    if current_bundle:
        yield tuple(current_bundle)
    if not saw_observation:
        raise ValueError("paper kinetic bundle request requires observations")


def _validated_pixel_iterator(
    pixels: Iterable[int],
    *,
    image_pixel_count: int,
    batch_position: int,
) -> Iterator[int]:
    previous: int | None = None
    for raw_pixel in pixels:
        pixel = int(raw_pixel)
        if isinstance(raw_pixel, bool) or pixel != raw_pixel:
            raise ValueError("paper kinetic pixel selections must contain integer ids")
        if pixel < 0 or pixel >= image_pixel_count:
            raise IndexError(f"paper kinetic pixel {pixel} at batch position {batch_position} leaves the stage image")
        if previous is not None and pixel <= previous:
            raise ValueError("paper kinetic streaming pixel selections must be strictly increasing")
        previous = pixel
        yield pixel


def _validate_provider_grid(
    target_provider: PowerFoamTargetProvider,
    ray_provider: PowerFoamRayProvider,
    *,
    height: int,
    width: int,
) -> None:
    if (
        target_provider.view_count,
        target_provider.frame_count,
        target_provider.height,
        target_provider.width,
    ) != (
        ray_provider.view_count,
        ray_provider.frame_count,
        ray_provider.height,
        ray_provider.width,
    ):
        raise ValueError("paper kinetic target and camera providers must share one grid")
    if height < 1 or width < 1:
        raise ValueError("paper kinetic stage dimensions must be positive")
    if not ray_provider.cameras or any(len(cameras) != ray_provider.frame_count for cameras in ray_provider.cameras):
        raise ValueError("paper kinetic calibrated cameras must form a rectangular grid")
    residency = target_provider.residency()
    if bool(residency.get("full_source_resident")) or int(residency.get("resident_bytes", -1)) != 0:
        raise ValueError("paper kinetic lazy programs require a nonresident target source")


def _frame_time_tuple(
    frame_times: Sequence[float] | torch.Tensor,
    *,
    frame_count: int,
) -> tuple[float, ...]:
    values = torch.as_tensor(frame_times, dtype=torch.float64, device="cpu").reshape(-1)
    result = tuple(float(value) for value in values.tolist())
    if len(result) != frame_count:
        raise ValueError("paper kinetic frame times must match the dataset frame count")
    if not all(math.isfinite(value) for value in result):
        raise ValueError("paper kinetic frame times must be finite")
    if any(right <= left for left, right in zip(result, result[1:], strict=False)):
        raise ValueError("paper kinetic frame times must be strictly increasing")
    return result


def _component_contract(component: Any, *, name: str) -> tuple[str, str]:
    provenance = str(getattr(component, "provenance", ""))
    generation_digest = str(getattr(component, "generation_digest", ""))
    if not provenance.strip():
        raise ValueError(f"paper kinetic {name} provenance must be nonempty")
    _require_sha256(generation_digest, name=f"{name}.generation_digest")
    return provenance, generation_digest


def _target_residency_digest(provider: PowerFoamTargetProvider) -> str:
    residency = provider.residency()
    if bool(residency.get("full_source_resident")) or int(residency.get("resident_bytes", -1)) != 0:
        raise ValueError("paper kinetic target source became resident")
    return _digest_parts(
        PROVIDER_PROVENANCE,
        "target-residency",
        _canonical_json(residency),
        provider.view_count,
        provider.frame_count,
        provider.height,
        provider.width,
    )


def _camera_grid_digests(
    provider: PowerFoamRayProvider,
    *,
    height: int,
    width: int,
    frame_times: tuple[float, ...],
) -> tuple[str, tuple[str, ...], tuple[bool, ...]]:
    if len(frame_times) != provider.frame_count:
        raise ValueError("paper kinetic camera times do not cover the camera grid")
    camera_digests = tuple(
        tuple(
            _camera_digest(
                _scaled_cpu_camera(
                    camera,
                    source_height=provider.height,
                    source_width=provider.width,
                    target_height=height,
                    target_width=width,
                )
            )
            for camera in cameras
        )
        for cameras in provider.cameras
    )
    paths = tuple(
        _digest_parts(
            PROVIDER_PROVENANCE,
            "camera-path",
            view_index,
            frame_times,
            digests,
        )
        for view_index, digests in enumerate(camera_digests)
    )
    static_certificates = tuple(
        bool(digests) and all(digest == digests[0] for digest in digests[1:])
        for digests in camera_digests
    )
    return (
        _digest_parts(
            PROVIDER_PROVENANCE,
            "camera-grid",
            provider.view_count,
            provider.frame_count,
            height,
            width,
            paths,
        ),
        paths,
        static_certificates,
    )


def _scaled_cpu_camera(
    camera: CameraSpec,
    *,
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
) -> CameraSpec:
    scale_x = float(target_width) / float(source_width)
    scale_y = float(target_height) / float(source_height)
    return CameraSpec(
        fx=float(torch.as_tensor(camera.fx).detach().cpu().item()) * scale_x,
        fy=float(torch.as_tensor(camera.fy).detach().cpu().item()) * scale_y,
        cx=float(torch.as_tensor(camera.cx).detach().cpu().item()) * scale_x,
        cy=float(torch.as_tensor(camera.cy).detach().cpu().item()) * scale_y,
        camera_to_world=torch.as_tensor(camera.camera_to_world)
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous(),
        lens_model=camera.lens_model,
        distortion=(
            None
            if camera.distortion is None
            else torch.as_tensor(camera.distortion).detach().to(device="cpu", dtype=torch.float64).contiguous()
        ),
    )


def _camera_digest(camera: CameraSpec) -> str:
    return _digest_parts(
        PROVIDER_PROVENANCE,
        "camera",
        float(torch.as_tensor(camera.fx).item()),
        float(torch.as_tensor(camera.fy).item()),
        float(torch.as_tensor(camera.cx).item()),
        float(torch.as_tensor(camera.cy).item()),
        camera.lens_model,
        _tensor_content_digest(torch.as_tensor(camera.camera_to_world)),
        None if camera.distortion is None else _tensor_content_digest(torch.as_tensor(camera.distortion)),
    )


def _site_tensors(sites: AffineKineticPowerSites) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (sites.positions0, sites.velocities, sites.weight_coefficients)


def _site_content_digest(sites: AffineKineticPowerSites) -> str:
    return _digest_parts(
        WORLD_SNAPSHOT_PROVENANCE,
        sites.site_count,
        tuple(_tensor_content_digest(tensor) for tensor in _site_tensors(sites)),
    )


def _tensor_content_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(repr(tuple(value.shape)).encode("utf-8"))
    digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    """Warm identity/layout/version seal; deliberately does not hash content."""

    storage = tensor.untyped_storage()
    return (
        id(tensor),
        int(getattr(tensor, "_version", 0)),
        int(storage.data_ptr()),
        int(storage.nbytes()),
        int(tensor.storage_offset()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        tensor.layout,
        bool(tensor.requires_grad),
    )


def _ray_record_digest(
    observation: PaperKineticObservation,
    *,
    sample_time: float,
    ray: tuple[float, float, float, float, float, float],
    camera_record_digest: str,
) -> str:
    return _digest_parts(
        RAY_RECORD_PROVENANCE,
        observation.sample_identity,
        sample_time,
        ray,
        camera_record_digest,
    )


def _track_request_digest(request: PaperKineticTrackProgramRequest) -> str:
    # Deliberately excludes bounded ray witnesses and sampled observations.
    # ``camera_path_digest`` already seals every calibrated camera and frame
    # time once at provider preparation, so structural compile identity is
    # stable across sampler choices without losing camera-path provenance.
    return _digest_parts(
        PROGRAM_REQUEST_PROVENANCE,
        request.world.generation_digest,
        request.dataset_generation_digest,
        request.factory_generation_digest,
        request.view_index,
        request.pixel_index,
        request.height,
        request.width,
        len(request.frame_times),
        request.frame_times[0],
        request.frame_times[-1],
        request.camera_path_digest,
        request.static_camera_path_certified,
    )


def _track_replay_validation_digest(
    validation: PaperKineticTrackReplayValidation,
) -> str:
    return _digest_parts(
        TRACK_REPLAY_VALIDATION_PROVENANCE,
        validation.structural_request_generation_digest,
        validation.view_index,
        validation.pixel_index,
        tuple(record.generation_digest for record in validation.observations),
    )


def _provider_digest(provider: PaperKineticLazyProgramBundleProvider) -> str:
    # The full immutable frame-time tuple and every camera record are already
    # cold-bound by camera_grid_digest/view_camera_path_digests. Rehashing the
    # tuple here would make every warm assertion O(F_dataset), despite the
    # documented outer-boundary cold-certification contract. Keep only
    # constant-size shape/domain facts in the warm generation recomputation.
    return _digest_parts(
        PROVIDER_PROVENANCE,
        provider.dataset_generation_digest,
        provider.target_residency_digest,
        provider.camera_grid_digest,
        provider.view_camera_path_digests,
        provider.view_static_camera_path_certified,
        len(provider.frame_times),
        provider.frame_times[0],
        provider.frame_times[-1],
        provider.height,
        provider.width,
        provider.maximum_tracks_per_bundle,
        provider.maximum_observations_per_bundle,
        provider.maximum_rows_per_native_block,
        provider.initializer_provenance,
        provider.initializer_generation_digest,
        provider.factory_provenance,
        provider.factory_generation_digest,
        provider.world.generation_digest,
        provider.observation_residency_scope,
        provider.dense_frame_observation_streaming_implemented,
    )


def _bundle_digest(bundle: PaperKineticLazyProgramBundle) -> str:
    return _digest_parts(
        PROVIDER_PROVENANCE,
        "bundle",
        bundle.provider_generation_digest,
        bundle.bundle_index,
        bundle.view_index,
        bundle.track_ids,
        tuple(record.generation_digest for record in bundle.observations),
        bundle.sampler.generation_digest,
        bundle.spatial_bundle.generation_digest,
        bundle.program_generation_digests,
        bundle.factory_request_generation_digests,
        bundle.compile_receipt.generation_digest,
        0,
        0,
        0,
        0,
    )


def _compile_receipt_digest(
    receipt: PaperKineticLazyProgramCompileReceipt,
    *,
    track_ids: tuple[int, ...],
    program_generation_digests: tuple[str, ...],
    request_generation_digests: tuple[str, ...],
) -> str:
    return _digest_parts(
        COMPILE_RECEIPT_PROVENANCE,
        receipt.provider_generation_digest,
        receipt.view_index,
        track_ids,
        program_generation_digests,
        request_generation_digests,
        receipt.compile_track_count,
        receipt.compiler_work_receipt_count,
        receipt.compiler_work_receipt_chain_link_count,
        receipt.root_complement_witness_count,
        receipt.candidate_source_attempt_count,
        receipt.all_site_witness_check_count,
        receipt.unique_pair_difference_count,
        receipt.per_witness_candidate_bound_verified,
        receipt.exhaustive_triple_enumeration_used,
        receipt.requested_frame_sampling_used,
        receipt.compiler_accounting_complete,
        receipt.all_track_receipt_digests_verified,
        receipt.compiler_work_receipt_provenance,
        receipt.compiler_work_receipt_chain_digest,
        receipt.retained_compiled_program_count,
        receipt.retained_compiler_receipt_entry_count,
        receipt.retained_compiler_tensor_bytes,
    )


def _replace_provider_digest(
    value: PaperKineticLazyProgramBundleProvider,
    digest: str,
) -> PaperKineticLazyProgramBundleProvider:
    return replace(value, generation_digest=digest)


def _replace_track_request_digest(
    value: PaperKineticTrackProgramRequest,
    digest: str,
) -> PaperKineticTrackProgramRequest:
    return replace(value, generation_digest=digest)


def _replace_bundle_digest(
    value: PaperKineticLazyProgramBundle,
    digest: str,
) -> PaperKineticLazyProgramBundle:
    return replace(value, generation_digest=digest)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _require_sha256(value: str, *, name: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    try:
        parsed = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest") from error
    if len(parsed) != 32 or value != value.lower():
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_nonnegative_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _require_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


__all__ = [
    "BOUNDED_OBSERVATION_SCOPE",
    "COMPILE_RECEIPT_PROVENANCE",
    "PROGRAM_REQUEST_PROVENANCE",
    "PROVIDER_PROVENANCE",
    "TRACK_REPLAY_VALIDATION_PROVENANCE",
    "PaperKineticLazyBundleConstructionLifetimeSlot",
    "PaperKineticLazyProgramBundle",
    "PaperKineticLazyProgramCompileReceipt",
    "PaperKineticLazyProgramBundleMemoryReport",
    "PaperKineticLazyProgramBundleProvider",
    "PaperKineticObservation",
    "PaperKineticObservationRayRecord",
    "PaperKineticTrackProgramFactory",
    "PaperKineticTrackProgramRequest",
    "PaperKineticTrackReplayValidation",
    "PaperKineticWorldInitializationRequest",
    "PaperKineticWorldInitializer",
    "PaperKineticWorldSnapshot",
    "compile_paper_kinetic_cpu_ragged_sampler",
    "iter_canonical_observations_from_spacetime_batch",
    "observations_from_spacetime_batch",
    "prepare_paper_kinetic_lazy_bundle_construction_lifetime_slot",
    "prepare_paper_kinetic_lazy_program_bundle_provider",
]
