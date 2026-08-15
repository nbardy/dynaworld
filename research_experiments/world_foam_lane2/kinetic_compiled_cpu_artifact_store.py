"""Bounded CPU structural-program reuse for kinetic WorldFoam.

The lazy paper path must not compile the same camera/world track programs on
every optimizer step.  It also must not solve that cost problem by retaining
targets, sampled observations, device payloads, or an unbounded Python cache.
This module places the cache boundary at the last frame-free CPU object:
``PaperKineticRowRaggedSampler``.

An artifact therefore retains the already-compiled kinetic programs, their
tensor-free native-topology/equal-rank descriptors, and the ragged row catalog.
It deliberately does *not* retain a lazy bundle, observation ray records,
dataset frame-time tuples, union-local device mappings, materialized native
blocks, native runtimes, targets, samples, predictions, or gradients.

The cache key is observation invariant.  It binds the provider, world, full
camera path, program factory, lowering policy, view, and pixel tracks; sampled
frame ids and observation ids never enter the key.  Correctness does not rest
on that omission: cold admission streams the full calibrated camera path
through the cached affine-ray program.  The outer request cold-certifies the
complete camera grid once. Warm hits consume only the sealed provider/view
certificate and program tensor versions; they neither retain nor rebuild one
camera signature per dataset frame. A factory whose
output depended incorrectly on the sampled observation subset consequently
fails closed instead of poisoning cross-step reuse.

Accounting is a conservative logical upper bound: reachable CPU tensor bytes
plus canonical descriptor/key bytes are charged independently per artifact,
even when two artifacts share a tensor object.  Python allocator overhead and
allocator storage/peak are explicitly unmeasured.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import torch
from camera import CameraSpec, build_camera_rays_at_pixels
from paper_kinetic_lazy_program_bundles import (
    PaperKineticLazyProgramBundle,
    PaperKineticLazyProgramBundleProvider,
    PaperKineticObservation,
    compile_paper_kinetic_cpu_ragged_sampler,
)
from paper_kinetic_ragged_sample_plan import PaperKineticRowRaggedSampler

STORE_PROVENANCE = "paper-kinetic-bounded-compiled-cpu-artifact-store-v1"
ARTIFACT_SCOPE = "frame-free-cpu-program-topology-equal-rank-sampler"
ACCOUNTING_SCOPE = "logical-tensor-plus-canonical-metadata-upper-bound"

_ARTIFACT_SEAL = object()


@dataclass(frozen=True)
class PaperKineticCompiledCpuArtifactKey:
    """Observation-invariant identity of one view-local spatial program set."""

    provider_generation_digest: str
    dataset_generation_digest: str
    world_generation_digest: str
    world_sites_content_digest: str
    camera_grid_digest: str
    camera_path_digest: str
    factory_provenance: str
    factory_generation_digest: str
    maximum_rows_per_native_block: int
    frame_count: int
    height: int
    width: int
    view_index: int
    track_ids: tuple[int, ...]
    generation_digest: str
    provenance: str = STORE_PROVENANCE

    @property
    def track_count(self) -> int:
        return len(self.track_ids)

    def assert_self_consistent(self) -> None:
        if self.provenance != STORE_PROVENANCE:
            raise ValueError("compiled CPU artifact key provenance changed")
        for name, digest in (
            ("provider_generation_digest", self.provider_generation_digest),
            ("dataset_generation_digest", self.dataset_generation_digest),
            ("world_generation_digest", self.world_generation_digest),
            ("world_sites_content_digest", self.world_sites_content_digest),
            ("camera_grid_digest", self.camera_grid_digest),
            ("camera_path_digest", self.camera_path_digest),
            ("factory_generation_digest", self.factory_generation_digest),
        ):
            _require_sha256(digest, name=name)
        if not self.factory_provenance.strip():
            raise ValueError("compiled CPU artifact factory provenance is empty")
        for name, value in (
            ("maximum_rows_per_native_block", self.maximum_rows_per_native_block),
            ("frame_count", self.frame_count),
            ("height", self.height),
            ("width", self.width),
        ):
            _require_positive_int(value, name=name)
        _require_nonnegative_int(self.view_index, name="view_index")
        if (
            not self.track_ids
            or tuple(sorted(set(self.track_ids))) != self.track_ids
            or self.track_ids[-1] >= self.height * self.width
        ):
            raise ValueError("compiled CPU artifact tracks are not canonical pixels")
        if self.generation_digest != _key_digest(self):
            raise ValueError("compiled CPU artifact key generation changed")


@dataclass(frozen=True)
class PaperKineticCompiledCpuArtifact:
    """Sealed frame-free CPU compiler output admitted to the bounded store."""

    key: PaperKineticCompiledCpuArtifactKey
    sampler: PaperKineticRowRaggedSampler
    program_generation_digests: tuple[str, ...]
    logical_tensor_bytes: int
    canonical_metadata_bytes: int
    accounted_resident_bytes: int
    generation_digest: str
    structural_compile_track_count: int
    provenance: str = STORE_PROVENANCE
    artifact_scope: str = ARTIFACT_SCOPE
    accounting_scope: str = ACCOUNTING_SCOPE
    retained_observation_count: int = 0
    retained_target_tensor_bytes: int = 0
    retained_dataset_frame_time_count: int = 0
    retained_sample_tensor_bytes: int = 0
    retained_prediction_tensor_bytes: int = 0
    retained_gradient_tensor_bytes: int = 0
    retained_union_device_mapping_tensor_bytes: int = 0
    retained_native_payload_count: int = 0
    retained_native_runtime_count: int = 0
    allocator_storage_bytes_measured: bool = False
    allocator_peak_measured: bool = False
    python_allocator_bytes_measured: bool = False
    _seal: object = None

    @property
    def track_ids(self) -> tuple[int, ...]:
        return self.key.track_ids

    @property
    def track_count(self) -> int:
        return self.key.track_count

    def accounting(self) -> dict[str, int | str | bool]:
        return {
            "provenance": self.provenance,
            "artifact_scope": self.artifact_scope,
            "accounting_scope": self.accounting_scope,
            "track_count": self.track_count,
            "logical_tensor_bytes": self.logical_tensor_bytes,
            "canonical_metadata_bytes": self.canonical_metadata_bytes,
            "accounted_resident_bytes": self.accounted_resident_bytes,
            "retained_observation_count": 0,
            "retained_target_tensor_bytes": 0,
            "retained_dataset_frame_time_count": 0,
            "retained_sample_tensor_bytes": 0,
            "retained_prediction_tensor_bytes": 0,
            "retained_gradient_tensor_bytes": 0,
            "retained_union_device_mapping_tensor_bytes": 0,
            "retained_native_payload_count": 0,
            "retained_native_runtime_count": 0,
            "all_reachable_tensors_are_cpu": True,
            "observation_invariant_cache_key": True,
            "full_camera_path_validated_at_cold_admission": True,
            "warm_validation_uses_provider_camera_certificate_only": True,
            "allocator_storage_bytes_measured": False,
            "allocator_peak_measured": False,
            "python_allocator_bytes_measured": False,
        }

    def assert_warm_current(self) -> None:
        """Validate seals and tensor versions without content hashes or rays."""

        if (
            self._seal is not _ARTIFACT_SEAL
            or self.provenance != STORE_PROVENANCE
            or self.artifact_scope != ARTIFACT_SCOPE
            or self.accounting_scope != ACCOUNTING_SCOPE
            or self.structural_compile_track_count != self.track_count
            or self.retained_observation_count != 0
            or self.retained_target_tensor_bytes != 0
            or self.retained_dataset_frame_time_count != 0
            or self.retained_sample_tensor_bytes != 0
            or self.retained_prediction_tensor_bytes != 0
            or self.retained_gradient_tensor_bytes != 0
            or self.retained_union_device_mapping_tensor_bytes != 0
            or self.retained_native_payload_count != 0
            or self.retained_native_runtime_count != 0
            or self.allocator_storage_bytes_measured
            or self.allocator_peak_measured
            or self.python_allocator_bytes_measured
        ):
            raise ValueError("compiled CPU artifact execution/memory contract changed")
        self.key.assert_self_consistent()
        if not isinstance(self.sampler, PaperKineticRowRaggedSampler):
            raise TypeError("compiled CPU artifact must retain a ragged sampler")
        self.sampler.assert_warm_layout()
        if self.sampler.view_index != self.key.view_index or self.sampler.track_ids != self.track_ids:
            raise ValueError("compiled CPU artifact sampler leaves its structural key")
        programs = _programs_by_track(self.sampler)
        current_program_digests = tuple(programs[track].generation_digest for track in self.track_ids)
        if self.program_generation_digests != current_program_digests:
            raise ValueError("compiled CPU artifact program generation changed")
        tensors = _reachable_tensors(self.sampler)
        _assert_cpu_structural_tensors(tensors)
        current_tensor_bytes = _logical_tensor_bytes(tensors)
        current_metadata_bytes = _artifact_metadata_bytes(
            self.key,
            self.sampler,
            self.program_generation_digests,
        )
        if (
            self.logical_tensor_bytes != current_tensor_bytes
            or self.canonical_metadata_bytes != current_metadata_bytes
            or self.accounted_resident_bytes != current_tensor_bytes + current_metadata_bytes
        ):
            raise ValueError("compiled CPU artifact resident-byte accounting changed")
        if self.generation_digest != _artifact_digest(self):
            raise ValueError("compiled CPU artifact generation changed")

    def assert_current(self) -> None:
        """Cold full-content validation of the cached structural artifact."""

        self.assert_warm_current()
        self.sampler.assert_cold_current()

    def assert_warm_reusable_with_provider(
        self,
        provider: PaperKineticLazyProgramBundleProvider,
    ) -> None:
        """Warm identity/layout/version validation with no frame-ray replay."""

        self.assert_warm_current()
        expected = _prepare_artifact_key(
            provider,
            view_index=self.key.view_index,
            track_ids=self.track_ids,
            cold=False,
        )
        if expected != self.key:
            raise ValueError("compiled CPU artifact is stale for this provider/camera/world")
        # The outer request owns the O(V F_dataset) cold camera certification.
        # Rebuilding an F_dataset-long signature tuple for every bounded warm
        # acquisition would be hidden frame-dependent work and metadata. The
        # warm key instead binds the sealed provider generation and its
        # precomputed per-view path digest.
        for program in _programs_by_track(self.sampler).values():
            if program.binding.sites is not provider.world.sites:
                raise ValueError("compiled CPU artifact retained a foreign world snapshot")

    def assert_cold_admissible_with_provider(
        self,
        provider: PaperKineticLazyProgramBundleProvider,
    ) -> None:
        """Cold-validate provenance against the certified calibrated path."""

        self.assert_current()
        expected = _prepare_artifact_key(
            provider,
            view_index=self.key.view_index,
            track_ids=self.track_ids,
            cold=False,
        )
        if expected != self.key:
            raise ValueError("compiled CPU artifact is stale for this provider/camera/world")
        programs = _programs_by_track(self.sampler)
        frame_indices = (
            ((0,) if provider.frame_count == 1 else (0, provider.frame_count - 1))
            if provider.view_static_camera_path_certified[self.key.view_index]
            else range(provider.frame_count)
        )
        for track_id, program in programs.items():
            if program.binding.sites is not provider.world.sites:
                raise ValueError("compiled CPU artifact retained a foreign world snapshot")
            domain_min = float(program.binding.program.t_min)
            domain_max = float(program.binding.program.t_max)
            if domain_min > provider.frame_times[0] or domain_max < provider.frame_times[-1]:
                raise ValueError("compiled CPU artifact does not cover the full dataset time domain")
            for frame_index in frame_indices:
                sample_time = provider.frame_times[frame_index]
                expected_ray = _calibrated_ray(
                    provider,
                    view_index=self.key.view_index,
                    frame_index=frame_index,
                    pixel_index=track_id,
                )
                _assert_program_replays_ray(program, sample_time, expected_ray)

    def assert_reusable_with_provider(
        self,
        provider: PaperKineticLazyProgramBundleProvider,
    ) -> None:
        """Compatibility alias for explicit cold admission validation."""

        self.assert_cold_admissible_with_provider(provider)

    def assert_replays_selected_observations(
        self,
        provider: PaperKineticLazyProgramBundleProvider,
        observations: Sequence[PaperKineticObservation],
    ) -> None:
        """Validate an ephemeral sampled subset without retaining it."""

        self.assert_warm_reusable_with_provider(provider)
        selected = tuple(observations)
        if not selected or len({item.sample_identity for item in selected}) != len(selected):
            raise ValueError("compiled CPU artifact observations must be unique and nonempty")
        programs = _programs_by_track(self.sampler)
        for observation in selected:
            if not isinstance(observation, PaperKineticObservation):
                raise TypeError("compiled CPU artifact observations have the wrong type")
            if (
                observation.view_index != self.key.view_index
                or observation.pixel_index not in programs
                or observation.frame_index >= provider.frame_count
            ):
                raise ValueError("sampled observation leaves the compiled CPU artifact")
            expected_ray = _calibrated_ray(
                provider,
                view_index=observation.view_index,
                frame_index=observation.frame_index,
                pixel_index=observation.pixel_index,
            )
            _assert_program_replays_ray(
                programs[observation.pixel_index],
                provider.frame_times[observation.frame_index],
                expected_ray,
            )


@dataclass(frozen=True)
class PaperKineticCompiledCpuArtifactStorePolicy:
    maximum_entries: int
    maximum_resident_accounted_bytes: int

    def __post_init__(self) -> None:
        _require_positive_int(self.maximum_entries, name="maximum_entries")
        _require_positive_int(
            self.maximum_resident_accounted_bytes,
            name="maximum_resident_accounted_bytes",
        )


@dataclass(frozen=True)
class PaperKineticCompiledCpuArtifactAcquisition:
    artifact: PaperKineticCompiledCpuArtifact
    cache_status: str
    evicted_entry_count: int
    evicted_accounted_bytes: int
    cold_compiled_track_count: int
    avoided_compile_track_count: int

    @property
    def warm_hit(self) -> bool:
        return self.cache_status == "warm_hit"


@dataclass(frozen=True)
class PaperKineticCompiledCpuArtifactStoreReport:
    maximum_entries: int
    maximum_resident_accounted_bytes: int
    current_entry_count: int
    current_resident_accounted_bytes: int
    peak_resident_accounted_bytes: int
    lookup_count: int
    hit_count: int
    miss_count: int
    compile_attempt_count: int
    cold_compile_count: int
    compile_failure_count: int
    eviction_count: int
    evicted_accounted_bytes: int
    stale_rejection_count: int
    cold_compiled_track_count: int
    avoided_compile_track_count: int
    cold_full_camera_path_validation_count: int
    warm_identity_version_validation_count: int
    warm_full_camera_path_validation_count: int
    retained_observation_count: int
    retained_target_tensor_bytes: int
    retained_dataset_frame_time_count: int
    retained_native_payload_count: int
    retained_native_runtime_count: int
    observation_invariant_cache_key: bool
    full_camera_path_validated_at_cold_admission: bool
    warm_validation_uses_provider_camera_certificate_only: bool
    device_runtime_cache_enabled: bool
    cold_compile_scratch_budget_enforced_by_store: bool
    cold_compile_scratch_peak_measured: bool
    allocator_peak_measured: bool
    python_allocator_bytes_measured: bool


class PaperKineticCompiledCpuArtifactStore:
    """Explicitly byte- and entry-bounded LRU for structural CPU artifacts."""

    def __init__(self, policy: PaperKineticCompiledCpuArtifactStorePolicy) -> None:
        if not isinstance(policy, PaperKineticCompiledCpuArtifactStorePolicy):
            raise TypeError("compiled CPU artifact store requires an explicit policy")
        self._policy = policy
        self._entries: OrderedDict[
            PaperKineticCompiledCpuArtifactKey,
            PaperKineticCompiledCpuArtifact,
        ] = OrderedDict()
        self._resident_accounted_bytes = 0
        self._peak_resident_accounted_bytes = 0
        self._lookup_count = 0
        self._hit_count = 0
        self._miss_count = 0
        self._compile_attempt_count = 0
        self._cold_compile_count = 0
        self._compile_failure_count = 0
        self._eviction_count = 0
        self._evicted_accounted_bytes = 0
        self._stale_rejection_count = 0
        self._cold_compiled_track_count = 0
        self._avoided_compile_track_count = 0
        self._cold_full_camera_path_validation_count = 0
        self._warm_identity_version_validation_count = 0
        self._lock = threading.RLock()
        self._acquire_active = False
        self._closed = False

    @property
    def policy(self) -> PaperKineticCompiledCpuArtifactStorePolicy:
        return self._policy

    def acquire(
        self,
        provider: PaperKineticLazyProgramBundleProvider,
        *,
        view_index: int,
        track_ids: Sequence[int],
        maximum_artifact_accounted_bytes: int,
        compile_artifact: Callable[
            [PaperKineticCompiledCpuArtifactKey],
            PaperKineticCompiledCpuArtifact,
        ],
    ) -> PaperKineticCompiledCpuArtifactAcquisition:
        """Acquire a validated artifact, compiling only after budget preflight.

        ``maximum_artifact_accounted_bytes`` is a caller-owned hard upper bound
        on the cold callback's retained result.  If it cannot fit, the callback
        is never invoked, and LRU eviction occurs before invocation.  Temporary
        compiler scratch is outside this store and must be bounded by the outer
        compiler/coordinator memory policy; it is not measured here.
        """

        if not callable(compile_artifact):
            raise TypeError("compile_artifact must be callable")
        _require_positive_int(
            maximum_artifact_accounted_bytes,
            name="maximum_artifact_accounted_bytes",
        )
        key = _prepare_artifact_key(
            provider,
            view_index=view_index,
            track_ids=track_ids,
            cold=False,
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("compiled CPU artifact store is closed")
            if self._acquire_active:
                raise RuntimeError("compiled CPU artifact store acquisition is not reentrant")
            self._acquire_active = True
            try:
                self._lookup_count += 1
                cached = self._entries.get(key)
                if cached is not None:
                    try:
                        cached.assert_warm_reusable_with_provider(provider)
                    except BaseException:
                        self._remove_entry(key)
                        self._stale_rejection_count += 1
                        raise
                    self._entries.move_to_end(key)
                    self._hit_count += 1
                    self._warm_identity_version_validation_count += 1
                    self._avoided_compile_track_count += cached.track_count
                    return PaperKineticCompiledCpuArtifactAcquisition(
                        artifact=cached,
                        cache_status="warm_hit",
                        evicted_entry_count=0,
                        evicted_accounted_bytes=0,
                        cold_compiled_track_count=0,
                        avoided_compile_track_count=cached.track_count,
                    )

                self._miss_count += 1
                if maximum_artifact_accounted_bytes > self._policy.maximum_resident_accounted_bytes:
                    raise MemoryError("compiled CPU artifact upper bound exceeds the store byte policy")
                evicted_count = 0
                evicted_bytes = 0
                while self._entries and (
                    len(self._entries) + 1 > self._policy.maximum_entries
                    or self._resident_accounted_bytes + maximum_artifact_accounted_bytes
                    > self._policy.maximum_resident_accounted_bytes
                ):
                    old_key = next(iter(self._entries))
                    removed = self._remove_entry(old_key)
                    evicted_count += 1
                    evicted_bytes += removed.accounted_resident_bytes
                if (
                    len(self._entries) + 1 > self._policy.maximum_entries
                    or self._resident_accounted_bytes + maximum_artifact_accounted_bytes
                    > self._policy.maximum_resident_accounted_bytes
                ):
                    raise MemoryError("compiled CPU artifact cannot fit after bounded LRU eviction")

                self._compile_attempt_count += 1
                try:
                    artifact = compile_artifact(key)
                    _validate_compiled_callback_result(
                        artifact,
                        provider=provider,
                        expected_key=key,
                        maximum_artifact_accounted_bytes=(maximum_artifact_accounted_bytes),
                    )
                except BaseException:
                    self._compile_failure_count += 1
                    raise
                self._entries[key] = artifact
                self._resident_accounted_bytes += artifact.accounted_resident_bytes
                self._peak_resident_accounted_bytes = max(
                    self._peak_resident_accounted_bytes,
                    self._resident_accounted_bytes,
                )
                self._cold_compile_count += 1
                self._cold_compiled_track_count += artifact.track_count
                self._cold_full_camera_path_validation_count += 1
                return PaperKineticCompiledCpuArtifactAcquisition(
                    artifact=artifact,
                    cache_status="cold_compiled",
                    evicted_entry_count=evicted_count,
                    evicted_accounted_bytes=evicted_bytes,
                    cold_compiled_track_count=artifact.track_count,
                    avoided_compile_track_count=0,
                )
            finally:
                self._acquire_active = False

    def report(self) -> PaperKineticCompiledCpuArtifactStoreReport:
        with self._lock:
            return PaperKineticCompiledCpuArtifactStoreReport(
                maximum_entries=self._policy.maximum_entries,
                maximum_resident_accounted_bytes=(self._policy.maximum_resident_accounted_bytes),
                current_entry_count=len(self._entries),
                current_resident_accounted_bytes=self._resident_accounted_bytes,
                peak_resident_accounted_bytes=self._peak_resident_accounted_bytes,
                lookup_count=self._lookup_count,
                hit_count=self._hit_count,
                miss_count=self._miss_count,
                compile_attempt_count=self._compile_attempt_count,
                cold_compile_count=self._cold_compile_count,
                compile_failure_count=self._compile_failure_count,
                eviction_count=self._eviction_count,
                evicted_accounted_bytes=self._evicted_accounted_bytes,
                stale_rejection_count=self._stale_rejection_count,
                cold_compiled_track_count=self._cold_compiled_track_count,
                avoided_compile_track_count=self._avoided_compile_track_count,
                cold_full_camera_path_validation_count=(self._cold_full_camera_path_validation_count),
                warm_identity_version_validation_count=(self._warm_identity_version_validation_count),
                warm_full_camera_path_validation_count=0,
                retained_observation_count=0,
                retained_target_tensor_bytes=0,
                retained_dataset_frame_time_count=0,
                retained_native_payload_count=0,
                retained_native_runtime_count=0,
                observation_invariant_cache_key=True,
                full_camera_path_validated_at_cold_admission=True,
                warm_validation_uses_provider_camera_certificate_only=True,
                device_runtime_cache_enabled=False,
                cold_compile_scratch_budget_enforced_by_store=False,
                cold_compile_scratch_peak_measured=False,
                allocator_peak_measured=False,
                python_allocator_bytes_measured=False,
            )

    def clear(self) -> None:
        with self._lock:
            if self._acquire_active:
                raise RuntimeError("cannot clear compiled CPU artifacts during acquisition")
            while self._entries:
                self._remove_entry(next(iter(self._entries)))

    def close(self) -> None:
        with self._lock:
            if self._acquire_active:
                raise RuntimeError("cannot close compiled CPU artifacts during acquisition")
            self.clear()
            self._closed = True

    def _remove_entry(
        self,
        key: PaperKineticCompiledCpuArtifactKey,
    ) -> PaperKineticCompiledCpuArtifact:
        artifact = self._entries.pop(key)
        self._resident_accounted_bytes -= artifact.accounted_resident_bytes
        if self._resident_accounted_bytes < 0:
            raise ArithmeticError("compiled CPU artifact byte accounting underflow")
        self._eviction_count += 1
        self._evicted_accounted_bytes += artifact.accounted_resident_bytes
        return artifact


def prepare_paper_kinetic_compiled_cpu_artifact_key(
    provider: PaperKineticLazyProgramBundleProvider,
    *,
    view_index: int,
    track_ids: Sequence[int],
) -> PaperKineticCompiledCpuArtifactKey:
    """Build a structural lookup key with no sampled-observation component."""

    return _prepare_artifact_key(
        provider,
        view_index=view_index,
        track_ids=track_ids,
        cold=True,
    )


def compile_paper_kinetic_compiled_cpu_artifact(
    provider: PaperKineticLazyProgramBundleProvider,
    key: PaperKineticCompiledCpuArtifactKey,
) -> PaperKineticCompiledCpuArtifact:
    """Compile a bounded dense-track artifact without a transient device map.

    The key must name one contiguous view-local pixel interval within the
    provider's spatial compile bound.  Track programs and their ragged native
    descriptors are compiled directly on CPU.  The resulting artifact is then
    cold-admitted against the complete calibrated camera path, exactly like a
    transitional bundle-derived artifact, while retaining no observations,
    device mapping, native runtime, or frame-time tuple.

    This function is shaped for direct use as a bounded-store callback:
    ``lambda key: compile_paper_kinetic_compiled_cpu_artifact(provider, key)``.
    Compiler scratch remains outside the artifact-store byte budget.
    """

    if not isinstance(key, PaperKineticCompiledCpuArtifactKey):
        raise TypeError("direct compiled CPU artifact requires its structural key")
    key.assert_self_consistent()
    # Pay the complete provider/camera certification once for this cold
    # callback. Nested key/seal/result validation below is warm and consumes
    # the resulting scalar certificates instead of repeating O(VF_dataset).
    provider.assert_current()
    expected_key = _prepare_artifact_key(
        provider,
        view_index=key.view_index,
        track_ids=key.track_ids,
        cold=False,
    )
    if key != expected_key:
        raise ValueError("direct compiled CPU artifact key is foreign to the provider")
    sampler = compile_paper_kinetic_cpu_ragged_sampler(
        provider,
        view_index=key.view_index,
        track_ids=key.track_ids,
    )
    return _seal_paper_kinetic_compiled_cpu_artifact_from_sampler(
        sampler,
        provider,
        key=key,
    )


def _prepare_artifact_key(
    provider: PaperKineticLazyProgramBundleProvider,
    *,
    view_index: int,
    track_ids: Sequence[int],
    cold: bool,
) -> PaperKineticCompiledCpuArtifactKey:
    """Build the same key after either cold or warm provider validation."""

    if not isinstance(provider, PaperKineticLazyProgramBundleProvider):
        raise TypeError("compiled CPU artifact key requires a lazy program provider")
    if cold:
        provider.assert_current()
    else:
        provider.assert_warm_current()
    _require_nonnegative_int(view_index, name="view_index")
    if view_index >= provider.view_count:
        raise ValueError("compiled CPU artifact view leaves the provider")
    normalized_tracks = tuple(sorted(int(track_id) for track_id in track_ids))
    if (
        not normalized_tracks
        or len(set(normalized_tracks)) != len(normalized_tracks)
        or normalized_tracks[-1] >= provider.height * provider.width
        or normalized_tracks[0] < 0
    ):
        raise ValueError("compiled CPU artifact tracks must be unique valid pixels")
    provisional = PaperKineticCompiledCpuArtifactKey(
        provider_generation_digest=provider.generation_digest,
        dataset_generation_digest=provider.dataset_generation_digest,
        world_generation_digest=provider.world.generation_digest,
        world_sites_content_digest=provider.world.sites_content_digest,
        camera_grid_digest=provider.camera_grid_digest,
        camera_path_digest=provider.view_camera_path_digests[view_index],
        factory_provenance=provider.factory_provenance,
        factory_generation_digest=provider.factory_generation_digest,
        maximum_rows_per_native_block=provider.maximum_rows_per_native_block,
        frame_count=provider.frame_count,
        height=provider.height,
        width=provider.width,
        view_index=view_index,
        track_ids=normalized_tracks,
        generation_digest="",
    )
    result = PaperKineticCompiledCpuArtifactKey(
        **{
            **provisional.__dict__,
            "generation_digest": _key_digest(provisional),
        }
    )
    result.assert_self_consistent()
    return result


def seal_paper_kinetic_compiled_cpu_artifact_from_bundle(
    bundle: PaperKineticLazyProgramBundle,
    provider: PaperKineticLazyProgramBundleProvider,
    *,
    key: PaperKineticCompiledCpuArtifactKey | None = None,
) -> PaperKineticCompiledCpuArtifact:
    """Extract only the reusable CPU sampler from one cold compiled bundle.

    This compatibility adapter remains for sparse callers that already own a
    bundle. Dense fixed-site training should use
    :func:`compile_paper_kinetic_compiled_cpu_artifact`, which compiles the CPU
    sampler before any union-local device mapping exists. Extracting from a
    bundle remains safe because neither the bundle nor its observations or
    spatial mapping are stored.
    """

    if not isinstance(bundle, PaperKineticLazyProgramBundle):
        raise TypeError("compiled CPU artifact sealing requires a lazy bundle")
    bundle.assert_cold_current(provider)
    provider.assert_current()
    return _seal_paper_kinetic_compiled_cpu_artifact_from_sampler(
        bundle.sampler,
        provider,
        key=key,
    )


def _seal_paper_kinetic_compiled_cpu_artifact_from_sampler(
    sampler: PaperKineticRowRaggedSampler,
    provider: PaperKineticLazyProgramBundleProvider,
    *,
    key: PaperKineticCompiledCpuArtifactKey | None,
) -> PaperKineticCompiledCpuArtifact:
    """Bind one cold CPU sampler to its structural key and full camera path."""

    if not isinstance(sampler, PaperKineticRowRaggedSampler):
        raise TypeError("compiled CPU artifact sealing requires a ragged sampler")
    sampler.assert_cold_current()
    expected_key = _prepare_artifact_key(
        provider,
        view_index=sampler.view_index,
        track_ids=sampler.track_ids,
        cold=False,
    )
    selected_key = expected_key if key is None else key
    selected_key.assert_self_consistent()
    if selected_key != expected_key:
        raise ValueError("compiled CPU artifact key does not describe the sampler")
    programs = _programs_by_track(sampler)
    program_digests = tuple(programs[track_id].generation_digest for track_id in selected_key.track_ids)
    tensors = _reachable_tensors(sampler)
    _assert_cpu_structural_tensors(tensors)
    tensor_bytes = _logical_tensor_bytes(tensors)
    metadata_bytes = _artifact_metadata_bytes(
        selected_key,
        sampler,
        program_digests,
    )
    provisional = PaperKineticCompiledCpuArtifact(
        key=selected_key,
        sampler=sampler,
        program_generation_digests=program_digests,
        logical_tensor_bytes=tensor_bytes,
        canonical_metadata_bytes=metadata_bytes,
        accounted_resident_bytes=tensor_bytes + metadata_bytes,
        generation_digest="",
        structural_compile_track_count=selected_key.track_count,
        _seal=_ARTIFACT_SEAL,
    )
    result = PaperKineticCompiledCpuArtifact(
        **{
            **provisional.__dict__,
            "generation_digest": _artifact_digest(provisional),
        }
    )
    result.assert_cold_admissible_with_provider(provider)
    return result


def _programs_by_track(sampler: PaperKineticRowRaggedSampler) -> dict[int, Any]:
    programs: dict[int, Any] = {}
    for row in sampler.rows:
        previous = programs.setdefault(row.track_id, row.program)
        if previous is not row.program:
            raise ValueError("compiled CPU sampler mixes programs for one track")
    if tuple(sorted(programs)) != sampler.track_ids:
        raise ValueError("compiled CPU sampler track/program coverage changed")
    return programs


def _validate_compiled_callback_result(
    artifact: object,
    *,
    provider: PaperKineticLazyProgramBundleProvider,
    expected_key: PaperKineticCompiledCpuArtifactKey,
    maximum_artifact_accounted_bytes: int,
) -> None:
    if not isinstance(artifact, PaperKineticCompiledCpuArtifact):
        raise TypeError("compile_artifact returned the wrong artifact type")
    artifact.assert_cold_admissible_with_provider(provider)
    if artifact.key != expected_key:
        raise ValueError("compiled CPU artifact callback returned a foreign key")
    if artifact.accounted_resident_bytes > maximum_artifact_accounted_bytes:
        raise MemoryError("compiled CPU artifact exceeded its preflight upper bound")


def _calibrated_ray(
    provider: PaperKineticLazyProgramBundleProvider,
    *,
    view_index: int,
    frame_index: int,
    pixel_index: int,
) -> torch.Tensor:
    camera = _scaled_cpu_camera(
        provider.ray_provider.cameras[view_index][frame_index],
        source_height=provider.ray_provider.height,
        source_width=provider.ray_provider.width,
        target_height=provider.height,
        target_width=provider.width,
    )
    pixel = torch.tensor((pixel_index,), dtype=torch.int64, device="cpu")
    origins, directions = build_camera_rays_at_pixels(
        camera,
        pixel,
        height=provider.height,
        width=provider.width,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    return torch.cat((origins[0], directions[0])).contiguous()


def _assert_program_replays_ray(
    program: Any,
    sample_time: float,
    expected_ray: torch.Tensor,
) -> None:
    coefficients = program.binding.ray_coefficients
    if coefficients.dtype != torch.float64 or tuple(coefficients.shape) != (12,):
        raise ValueError("compiled CPU affine camera coefficients changed ABI")
    time = torch.tensor(sample_time, dtype=torch.float64, device=coefficients.device)
    replay = torch.cat(
        (
            coefficients[0:3] + time * coefficients[3:6],
            coefficients[6:9] + time * coefficients[9:12],
        )
    ).to(device="cpu")
    if not torch.allclose(replay, expected_ray, rtol=0.0, atol=1.0e-10):
        raise ValueError("compiled CPU artifact does not reproduce the full calibrated camera path")


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


def _reachable_tensors(root: Any) -> tuple[torch.Tensor, ...]:
    tensors: list[torch.Tensor] = []
    seen_objects: set[int] = set()
    seen_tensors: set[int] = set()

    def visit(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            if id(value) not in seen_tensors:
                seen_tensors.add(id(value))
                tensors.append(value)
            return
        if value is None or isinstance(value, (str, bytes, int, float, bool, complex)):
            return
        identity = id(value)
        if identity in seen_objects:
            return
        seen_objects.add(identity)
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(key)
                visit(item)
            return
        if isinstance(value, (tuple, list, set, frozenset)):
            for item in value:
                visit(item)
            return
        if is_dataclass(value) and not isinstance(value, type):
            for item in fields(value):
                if item.name == "frame_times":
                    raise ValueError("compiled CPU artifact retained dataset frame times")
                visit(getattr(value, item.name))
            return
        namespace = getattr(value, "__dict__", None)
        if isinstance(namespace, dict):
            if "frame_times" in namespace:
                raise ValueError("compiled CPU artifact retained dataset frame times")
            for item in namespace.values():
                visit(item)

    visit(root)
    return tuple(tensors)


def _assert_cpu_structural_tensors(tensors: Sequence[torch.Tensor]) -> None:
    if not tensors:
        raise ValueError("compiled CPU artifact unexpectedly has no structural tensors")
    for tensor in tensors:
        if tensor.device.type != "cpu":
            raise ValueError("compiled CPU artifact retained a device tensor")
        if tensor.requires_grad:
            raise ValueError("compiled CPU artifact retained an autograd tensor")


def _logical_tensor_bytes(tensors: Sequence[torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _artifact_metadata_bytes(
    key: PaperKineticCompiledCpuArtifactKey,
    sampler: PaperKineticRowRaggedSampler,
    program_generation_digests: tuple[str, ...],
) -> int:
    encoded = repr(
        (
            key,
            sampler.generation_digest,
            sampler.descriptor_canonical_metadata_bytes,
            sampler.lowering.generation_digest,
            sampler.lowering.descriptor_canonical_metadata_bytes,
            program_generation_digests,
        )
    ).encode("utf-8")
    return (
        len(encoded)
        + sampler.descriptor_canonical_metadata_bytes
        + sampler.lowering.descriptor_canonical_metadata_bytes
    )


def _key_digest(key: PaperKineticCompiledCpuArtifactKey) -> str:
    return _digest_parts(
        STORE_PROVENANCE,
        "key",
        key.provider_generation_digest,
        key.dataset_generation_digest,
        key.world_generation_digest,
        key.world_sites_content_digest,
        key.camera_grid_digest,
        key.camera_path_digest,
        key.factory_provenance,
        key.factory_generation_digest,
        key.maximum_rows_per_native_block,
        key.frame_count,
        key.height,
        key.width,
        key.view_index,
        key.track_ids,
    )


def _artifact_digest(artifact: PaperKineticCompiledCpuArtifact) -> str:
    return _digest_parts(
        STORE_PROVENANCE,
        "artifact",
        artifact.key.generation_digest,
        artifact.sampler.generation_digest,
        artifact.program_generation_digests,
        artifact.logical_tensor_bytes,
        artifact.canonical_metadata_bytes,
        artifact.accounted_resident_bytes,
        artifact.structural_compile_track_count,
        artifact.artifact_scope,
        artifact.accounting_scope,
    )


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
    "ACCOUNTING_SCOPE",
    "ARTIFACT_SCOPE",
    "STORE_PROVENANCE",
    "PaperKineticCompiledCpuArtifact",
    "PaperKineticCompiledCpuArtifactAcquisition",
    "PaperKineticCompiledCpuArtifactKey",
    "PaperKineticCompiledCpuArtifactStore",
    "PaperKineticCompiledCpuArtifactStorePolicy",
    "PaperKineticCompiledCpuArtifactStoreReport",
    "compile_paper_kinetic_compiled_cpu_artifact",
    "prepare_paper_kinetic_compiled_cpu_artifact_key",
    "seal_paper_kinetic_compiled_cpu_artifact_from_bundle",
]
