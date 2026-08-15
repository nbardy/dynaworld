"""Same-representation framewise control for the WorldFoam memory ablation.

The primary WorldFoam paper path compiles continuous owner programs and shares
their reverse across all selected times.  A fair scaling control must not fall
back to scalar fixed-time topology discovery or compile those same continuous
programs once per frame.  This module therefore keeps the representation fixed
and changes only the reverse schedule:

* compile the selected-track continuous programs once into the existing
  bounded CPU artifact store;
* replay exactly one selected frame through the native P0 forward/full-
  geometry reverse at a time;
* read back, accumulate, fence, and release that frame before opening the next;
* apply the identical raw sigmoid/softplus material chain rule and stateless
  geometry SGD exactly once after the complete frame manifest is sealed.

The retained compiler store and global gradient bars are independent of the
requested frame count.  Per-frame scalar timings are intentionally O(F), while
targets, predictions, native lanes, reverse scratch, and result capabilities
are frame-local.  This is an ablation coordinator, not a reusable trainer
generation: geometry mutation invalidates the compiled programs, so the
updated state is terminal and no retirement/recompile lifecycle is charged.

Logical tensor/accounted-byte receipts are not allocator or RSS measurements.
Fresh-process RSS/allocator sampling remains the producer's responsibility.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import torch
from paper_kinetic_runtime_paths import ensure_worldfoam_lane2_research_path

ensure_worldfoam_lane2_research_path()

from kinetic_compiled_cpu_artifact_store import (  # noqa: E402
    PaperKineticCompiledCpuArtifact,
    PaperKineticCompiledCpuArtifactKey,
    PaperKineticCompiledCpuArtifactStore,
    PaperKineticCompiledCpuArtifactStoreReport,
    _seal_paper_kinetic_compiled_cpu_artifact_from_sampler,
)
from kinetic_lazy_native_material_step import (  # noqa: E402
    PaperKineticLazyNativeMemoryPolicy,
    paper_kinetic_observation_manifest_digest,
    prepare_paper_kinetic_lazy_native_trainer_state,
)
from paper_kinetic_fixed_camera_combined_state import (  # noqa: E402
    PaperKineticFixedCameraCombinedSGDPolicy,
    PaperKineticFixedCameraCombinedState,
    _build_update_candidates,
)
from paper_kinetic_lazy_full_geometry_step import (  # noqa: E402
    STAGED_SPARSE,
    PaperKineticLazyFullGeometryMemoryPolicy,
    PaperKineticLazyNativeFullGeometryStepResult,
    run_paper_kinetic_lazy_native_full_geometry_step,
)
from paper_kinetic_lazy_program_bundles import (  # noqa: E402
    PaperKineticLazyBundleConstructionLifetimeSlot,
    PaperKineticLazyProgramBundle,
    PaperKineticLazyProgramBundleProvider,
    PaperKineticLazyProgramCompileReceipt,
    PaperKineticObservation,
    _BUNDLE_SEAL,
    _bundle_digest,
    _compile_cpu_ragged_sampler,
    _iter_canonical_observation_partitions,
    _materialize_ray_record,
    prepare_paper_kinetic_lazy_bundle_construction_lifetime_slot,
)
from paper_kinetic_union_local_bar_assembly import (  # noqa: E402
    prepare_paper_kinetic_union_local_spatial_bundle,
    prepare_paper_kinetic_union_local_spatial_bundle_construction_lifetime,
)


CONTROL_PROVENANCE = (
    "paper-kinetic-compiled-framewise-full-geometry-control-v1"
)
PRECOMPILE_PROVENANCE = (
    "paper-kinetic-compiled-framewise-selected-track-precompile-v1"
)
FRAME_READBACK_PROVENANCE = (
    "paper-kinetic-compiled-framewise-frame-readback-v1"
)
UPDATE_PROVENANCE = (
    "paper-kinetic-compiled-framewise-terminal-manual-sgd-v1"
)
CONTROL_STATUS = "source_integrated/native_runtime_unverified"

_PRECOMPILE_SEAL = object()
_UPDATE_SEAL = object()
_RESULT_SEAL = object()


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(int(tensor.numel()) * int(tensor.element_size()) for tensor in tensors)


def _tensor_content_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(repr(tuple(int(item) for item in value.shape)).encode("utf-8"))
    digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def _require_sha256(value: str, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class PaperKineticCompiledFramewiseArtifactMetadata:
    """Tensor-free one-time compiler metadata for one selected-track request."""

    view_index: int
    track_ids: tuple[int, ...]
    artifact_generation_digest: str
    sampler_generation_digest: str
    program_generation_digests: tuple[str, ...]
    request_generation_digests: tuple[str, ...]
    compile_receipt: PaperKineticLazyProgramCompileReceipt

    @property
    def track_count(self) -> int:
        return len(self.track_ids)

    def assert_current(self, artifact: PaperKineticCompiledCpuArtifact) -> None:
        artifact.assert_warm_current()
        self.compile_receipt.assert_current(
            track_ids=self.track_ids,
            program_generation_digests=self.program_generation_digests,
            request_generation_digests=self.request_generation_digests,
        )
        if (
            self.view_index != artifact.key.view_index
            or self.track_ids != artifact.track_ids
            or self.artifact_generation_digest != artifact.generation_digest
            or self.sampler_generation_digest != artifact.sampler.generation_digest
            or self.program_generation_digests
            != artifact.program_generation_digests
        ):
            raise ValueError("framewise artifact compiler metadata changed")


@dataclass(frozen=True)
class PaperKineticCompiledFramewisePrecompileReceipt:
    """Exact scalar totals for the sole continuous selected-track compile."""

    provider_generation_digest: str
    selected_track_manifest_digest: str
    request_count: int
    track_count: int
    artifact_generation_digests: tuple[str, ...]
    compile_receipt_generation_digests: tuple[str, ...]
    compiler_work_receipt_chain_digest: str
    compiler_work_receipt_count: int
    root_complement_witness_count: int
    candidate_source_attempt_count: int
    all_site_witness_check_count: int
    unique_pair_difference_count: int
    store_current_resident_accounted_bytes: int
    store_peak_resident_accounted_bytes: int
    store_maximum_resident_accounted_bytes: int
    generation_digest: str
    provenance: str = PRECOMPILE_PROVENANCE
    compile_pass_count: int = 1
    requested_frame_sampling_used: bool = False
    retained_observation_count: int = 0
    retained_target_tensor_bytes: int = 0
    retained_frame_tensor_bytes: int = 0
    allocator_peak_measured: bool = False
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        for name, value in (
            ("provider_generation_digest", self.provider_generation_digest),
            ("selected_track_manifest_digest", self.selected_track_manifest_digest),
            ("compiler_work_receipt_chain_digest", self.compiler_work_receipt_chain_digest),
            ("generation_digest", self.generation_digest),
        ):
            _require_sha256(value, name=name)
        if (
            self._seal is not _PRECOMPILE_SEAL
            or self.provenance != PRECOMPILE_PROVENANCE
            or self.compile_pass_count != 1
            or self.request_count < 1
            or self.track_count < 1
            or len(self.artifact_generation_digests) != self.request_count
            or len(self.compile_receipt_generation_digests) != self.request_count
            or self.compiler_work_receipt_count != self.track_count
            or self.requested_frame_sampling_used
            or self.retained_observation_count
            or self.retained_target_tensor_bytes
            or self.retained_frame_tensor_bytes
            or self.allocator_peak_measured
            or self.store_current_resident_accounted_bytes < 1
            or self.store_peak_resident_accounted_bytes
            < self.store_current_resident_accounted_bytes
            or self.store_peak_resident_accounted_bytes
            > self.store_maximum_resident_accounted_bytes
            or self.generation_digest != _precompile_digest(self)
        ):
            raise ValueError("framewise selected-track precompile receipt changed")
        for digest in (
            *self.artifact_generation_digests,
            *self.compile_receipt_generation_digests,
        ):
            _require_sha256(digest, name="precompile child digest")


class _PaperKineticCompiledFramewiseController:
    """Mutable ledger hidden behind an immutable provider subclass."""

    def __init__(
        self,
        artifact_store: PaperKineticCompiledCpuArtifactStore,
        *,
        view_index: int,
        selected_track_ids: tuple[int, ...],
        maximum_artifact_accounted_bytes_per_entry: int,
    ) -> None:
        if not isinstance(artifact_store, PaperKineticCompiledCpuArtifactStore):
            raise TypeError("framewise controller requires a bounded artifact store")
        _require_positive_int(
            maximum_artifact_accounted_bytes_per_entry,
            name="maximum_artifact_accounted_bytes_per_entry",
        )
        if not selected_track_ids or tuple(sorted(set(selected_track_ids))) != selected_track_ids:
            raise ValueError("framewise selected tracks must be canonical and nonempty")
        if isinstance(view_index, bool) or not isinstance(view_index, int) or view_index < 0:
            raise ValueError("framewise view index must be a nonnegative integer")
        self.artifact_store = artifact_store
        self.view_index = view_index
        self.selected_track_ids = selected_track_ids
        self.maximum_artifact_accounted_bytes_per_entry = (
            maximum_artifact_accounted_bytes_per_entry
        )
        self.metadata_by_track_ids: dict[
            tuple[int, ...], PaperKineticCompiledFramewiseArtifactMetadata
        ] = {}
        self.precompile_receipt: PaperKineticCompiledFramewisePrecompileReceipt | None = None
        self.frame_bundle_acquisition_count = 0
        self.frame_bundle_warm_hit_count = 0
        self.frame_bundle_cold_compile_count = 0
        self.active_iterator_count = 0

    @property
    def request_count(self) -> int:
        return len(self.metadata_by_track_ids)

    def precompile(
        self,
        provider: PaperKineticLazyProgramBundleProvider,
    ) -> PaperKineticCompiledFramewisePrecompileReceipt:
        if self.precompile_receipt is not None or self.metadata_by_track_ids:
            raise RuntimeError("framewise selected tracks were already precompiled")
        before = self.artifact_store.report()
        if before.current_entry_count or before.lookup_count or before.cold_compile_count:
            raise ValueError("framewise precompile requires an empty fresh artifact store")
        partitions = tuple(
            self.selected_track_ids[start : start + provider.maximum_tracks_per_bundle]
            for start in range(0, len(self.selected_track_ids), provider.maximum_tracks_per_bundle)
        )
        if len(partitions) > self.artifact_store.policy.maximum_entries:
            raise MemoryError(
                "framewise selected-track working set exceeds the artifact entry bound"
            )
        if (
            len(partitions) * self.maximum_artifact_accounted_bytes_per_entry
            > self.artifact_store.policy.maximum_resident_accounted_bytes
        ):
            raise MemoryError(
                "framewise per-entry artifact bounds exceed the resident store bound"
            )
        chain = _digest_parts(
            PRECOMPILE_PROVENANCE,
            "compiler-chain-root",
            provider.generation_digest,
            self.view_index,
            self.selected_track_ids,
        )
        totals = {
            "compiler_work_receipt_count": 0,
            "root_complement_witness_count": 0,
            "candidate_source_attempt_count": 0,
            "all_site_witness_check_count": 0,
            "unique_pair_difference_count": 0,
        }
        for ordinal, track_ids in enumerate(partitions):
            captured: dict[str, Any] = {}

            def compile_artifact(
                key: PaperKineticCompiledCpuArtifactKey,
            ) -> PaperKineticCompiledCpuArtifact:
                sampler, program_digests, request_digests, compile_receipt = (
                    _compile_cpu_ragged_sampler(
                        provider,
                        view_index=key.view_index,
                        track_ids=key.track_ids,
                        replay_records_by_track=None,
                    )
                )
                artifact = _seal_paper_kinetic_compiled_cpu_artifact_from_sampler(
                    sampler,
                    provider,
                    key=key,
                )
                captured.update(
                    artifact=artifact,
                    program_digests=program_digests,
                    request_digests=request_digests,
                    compile_receipt=compile_receipt,
                )
                return artifact

            acquisition = self.artifact_store.acquire(
                provider,
                view_index=self.view_index,
                track_ids=track_ids,
                maximum_artifact_accounted_bytes=(
                    self.maximum_artifact_accounted_bytes_per_entry
                ),
                compile_artifact=compile_artifact,
            )
            if acquisition.cache_status != "cold_compiled" or acquisition.evicted_entry_count:
                raise ArithmeticError(
                    "framewise one-time precompile did not cold-admit exactly once"
                )
            artifact = acquisition.artifact
            if captured.get("artifact") is not artifact:
                raise ArithmeticError("framewise compile callback lost its admitted artifact")
            metadata = PaperKineticCompiledFramewiseArtifactMetadata(
                view_index=self.view_index,
                track_ids=track_ids,
                artifact_generation_digest=artifact.generation_digest,
                sampler_generation_digest=artifact.sampler.generation_digest,
                program_generation_digests=tuple(captured["program_digests"]),
                request_generation_digests=tuple(captured["request_digests"]),
                compile_receipt=captured["compile_receipt"],
            )
            metadata.assert_current(artifact)
            self.metadata_by_track_ids[track_ids] = metadata
            receipt = metadata.compile_receipt
            for key in totals:
                receipt_key = key
                totals[key] += int(getattr(receipt, receipt_key))
            chain = _digest_parts(
                PRECOMPILE_PROVENANCE,
                "compiler-chain-link",
                chain,
                ordinal,
                track_ids,
                artifact.generation_digest,
                receipt.generation_digest,
                receipt.compiler_work_receipt_chain_digest,
            )
        after = self.artifact_store.report()
        if (
            after.current_entry_count != len(partitions)
            or after.cold_compile_count != len(partitions)
            or after.cold_compiled_track_count != len(self.selected_track_ids)
            or after.eviction_count
            or after.compile_failure_count
        ):
            raise ArithmeticError("framewise selected-track store lost exact precompile coverage")
        manifest_digest = _digest_parts(
            PRECOMPILE_PROVENANCE,
            "selected-track-manifest",
            provider.generation_digest,
            self.view_index,
            self.selected_track_ids,
            partitions,
        )
        provisional = PaperKineticCompiledFramewisePrecompileReceipt(
            provider_generation_digest=provider.generation_digest,
            selected_track_manifest_digest=manifest_digest,
            request_count=len(partitions),
            track_count=len(self.selected_track_ids),
            artifact_generation_digests=tuple(
                metadata.artifact_generation_digest
                for metadata in self.metadata_by_track_ids.values()
            ),
            compile_receipt_generation_digests=tuple(
                metadata.compile_receipt.generation_digest
                for metadata in self.metadata_by_track_ids.values()
            ),
            compiler_work_receipt_chain_digest=chain,
            compiler_work_receipt_count=totals["compiler_work_receipt_count"],
            root_complement_witness_count=totals[
                "root_complement_witness_count"
            ],
            candidate_source_attempt_count=totals[
                "candidate_source_attempt_count"
            ],
            all_site_witness_check_count=totals[
                "all_site_witness_check_count"
            ],
            unique_pair_difference_count=totals[
                "unique_pair_difference_count"
            ],
            store_current_resident_accounted_bytes=(
                after.current_resident_accounted_bytes
            ),
            store_peak_resident_accounted_bytes=(
                after.peak_resident_accounted_bytes
            ),
            store_maximum_resident_accounted_bytes=(
                after.maximum_resident_accounted_bytes
            ),
            generation_digest="",
            _seal=_PRECOMPILE_SEAL,
        )
        result = replace(
            provisional,
            generation_digest=_precompile_digest(provisional),
        )
        result.assert_current()
        self.precompile_receipt = result
        return result

    def iter_bundles(
        self,
        provider: PaperKineticLazyProgramBundleProvider,
        observations: Iterable[PaperKineticObservation],
        *,
        device: torch.device | str,
        construction_lifetime_slot: (
            PaperKineticLazyBundleConstructionLifetimeSlot | None
        ),
    ) -> Iterator[PaperKineticLazyProgramBundle]:
        if self.precompile_receipt is None:
            raise RuntimeError("framewise provider was not precompiled")
        if self.active_iterator_count:
            raise RuntimeError("framewise provider only permits one active frame replay")
        self.active_iterator_count = 1
        resolved_device = torch.device(device)
        try:
            for bundle_index, partition in enumerate(
                _iter_canonical_observation_partitions(provider, observations)
            ):
                track_ids = tuple(
                    sorted({observation.pixel_index for observation in partition})
                )
                metadata = self.metadata_by_track_ids.get(track_ids)
                if metadata is None:
                    raise ValueError(
                        "framewise replay requested tracks outside the one-time compile manifest"
                    )
                if partition[0].view_index != self.view_index:
                    raise ValueError(
                        "framewise replay requested another calibrated view"
                    )
                acquisition = self.artifact_store.acquire(
                    provider,
                    view_index=partition[0].view_index,
                    track_ids=track_ids,
                    maximum_artifact_accounted_bytes=(
                        self.maximum_artifact_accounted_bytes_per_entry
                    ),
                    compile_artifact=lambda _key: (_raise_cold_recompile()),
                )
                self.frame_bundle_acquisition_count += 1
                if acquisition.warm_hit:
                    self.frame_bundle_warm_hit_count += 1
                else:
                    self.frame_bundle_cold_compile_count += 1
                    raise RuntimeError(
                        "framewise replay attempted a forbidden continuous recompile"
                    )
                artifact = acquisition.artifact
                metadata.assert_current(artifact)
                artifact.assert_replays_selected_observations(provider, partition)
                records = tuple(
                    _materialize_ray_record(provider, observation)
                    for observation in partition
                )
                construction_lifetime = (
                    prepare_paper_kinetic_union_local_spatial_bundle_construction_lifetime(
                        artifact.sampler,
                        track_ids=track_ids,
                        device=resolved_device,
                    )
                )
                if construction_lifetime_slot is not None:
                    construction_lifetime_slot.install(construction_lifetime)
                elif resolved_device.type != "cpu":
                    raise RuntimeError(
                        "accelerator framewise bundle construction requires its lifetime slot"
                    )
                spatial = prepare_paper_kinetic_union_local_spatial_bundle(
                    artifact.sampler,
                    track_ids=track_ids,
                    device=resolved_device,
                    construction_lifetime=construction_lifetime,
                )
                provisional = PaperKineticLazyProgramBundle(
                    provider_generation_digest=provider.generation_digest,
                    bundle_index=bundle_index,
                    view_index=partition[0].view_index,
                    track_ids=track_ids,
                    observations=records,
                    sampler=artifact.sampler,
                    spatial_bundle=spatial,
                    program_generation_digests=(
                        metadata.program_generation_digests
                    ),
                    factory_request_generation_digests=(
                        metadata.request_generation_digests
                    ),
                    compile_receipt=metadata.compile_receipt,
                    generation_digest="",
                    _provider_identity=id(provider),
                    _sampler_identity=id(artifact.sampler),
                    _spatial_bundle_identity=id(spatial),
                    _seal=_BUNDLE_SEAL,
                )
                bundle = replace(
                    provisional,
                    generation_digest=_bundle_digest(provisional),
                )
                if resolved_device.type == "cpu":
                    bundle.assert_cold_current(provider)
                    if construction_lifetime_slot is not None:
                        construction_lifetime_slot.complete(construction_lifetime)
                else:
                    bundle.assert_accelerator_transfer_pending(provider)
                yield bundle
                del bundle, spatial, construction_lifetime, records, artifact, acquisition
        finally:
            self.active_iterator_count = 0


@dataclass(frozen=True)
class PaperKineticCompiledFramewiseProgramProvider(
    PaperKineticLazyProgramBundleProvider
):
    """Provider subtype replacing compilation with warm artifact replay."""

    framewise_controller: _PaperKineticCompiledFramewiseController | None = field(
        default=None,
        repr=False,
        compare=False,
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
        controller = self.framewise_controller
        if not isinstance(controller, _PaperKineticCompiledFramewiseController):
            raise ValueError("compiled framewise provider lost its controller")
        yield from controller.iter_bundles(
            self,
            observations,
            device=device,
            construction_lifetime_slot=construction_lifetime_slot,
        )


def prepare_paper_kinetic_compiled_framewise_program_provider(
    provider: PaperKineticLazyProgramBundleProvider,
    artifact_store: PaperKineticCompiledCpuArtifactStore,
    *,
    view_index: int = 0,
    selected_track_ids: Sequence[int],
    maximum_artifact_accounted_bytes_per_entry: int,
) -> PaperKineticCompiledFramewiseProgramProvider:
    """Clone provider metadata, then cold-compile the exact selected working set."""

    if type(provider) is not PaperKineticLazyProgramBundleProvider:
        raise TypeError("framewise wrapping requires one fresh base kinetic provider")
    provider.assert_current()
    if (
        isinstance(view_index, bool)
        or not isinstance(view_index, int)
        or view_index < 0
        or view_index >= provider.view_count
    ):
        raise ValueError("framewise view index leaves the calibrated provider")
    selected = tuple(int(value) for value in selected_track_ids)
    if (
        not selected
        or tuple(sorted(set(selected))) != selected
        or selected[0] < 0
        or selected[-1] >= provider.height * provider.width
    ):
        raise ValueError("framewise selected tracks leave the calibrated image")
    controller = _PaperKineticCompiledFramewiseController(
        artifact_store,
        view_index=view_index,
        selected_track_ids=selected,
        maximum_artifact_accounted_bytes_per_entry=(
            maximum_artifact_accounted_bytes_per_entry
        ),
    )
    result = PaperKineticCompiledFramewiseProgramProvider(
        **provider.__dict__,
        framewise_controller=controller,
    )
    result.assert_current()
    controller.precompile(result)
    result.assert_current()
    return result


@dataclass(frozen=True)
class PaperKineticCompiledFramewiseManualSGDReceipt:
    loss: float
    material_gradient_l2_norm: float
    position_gradient_l2_norm: float
    velocity_gradient_l2_norm: float
    weight_gradient_l2_norm: float
    raw_color_parameter_delta_l2_norm: float
    raw_density_parameter_delta_l2_norm: float
    positions0_parameter_delta_l2_norm: float
    velocities_parameter_delta_l2_norm: float
    weight_coefficients_parameter_delta_l2_norm: float
    parameters_before_digest: str
    parameters_after_digest: str
    gradient_digest: str
    update_authorization_digest: str
    generation_digest: str
    provenance: str = UPDATE_PROVENANCE
    cpu_optimizer_mutation_count: int = 1
    geometry_mutation_count: int = 1
    stale_provider_store_retirement_count: int = 0
    fresh_selected_track_recompile_count: int = 0
    optimizer_history_tensor_bytes: int = 0
    terminal_control_generation: bool = True
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        for name, value in (
            ("parameters_before_digest", self.parameters_before_digest),
            ("parameters_after_digest", self.parameters_after_digest),
            ("gradient_digest", self.gradient_digest),
            ("update_authorization_digest", self.update_authorization_digest),
            ("generation_digest", self.generation_digest),
        ):
            _require_sha256(value, name=name)
        scalars = (
            self.loss,
            self.material_gradient_l2_norm,
            self.position_gradient_l2_norm,
            self.velocity_gradient_l2_norm,
            self.weight_gradient_l2_norm,
            self.raw_color_parameter_delta_l2_norm,
            self.raw_density_parameter_delta_l2_norm,
            self.positions0_parameter_delta_l2_norm,
            self.velocities_parameter_delta_l2_norm,
            self.weight_coefficients_parameter_delta_l2_norm,
        )
        if (
            self._seal is not _UPDATE_SEAL
            or self.provenance != UPDATE_PROVENANCE
            or not all(math.isfinite(value) and value >= 0.0 for value in scalars)
            or self.cpu_optimizer_mutation_count != 1
            or self.geometry_mutation_count != 1
            or self.stale_provider_store_retirement_count
            or self.fresh_selected_track_recompile_count
            or self.optimizer_history_tensor_bytes
            or not self.terminal_control_generation
            or self.generation_digest != _update_digest(self)
        ):
            raise ValueError("framewise terminal manual-SGD receipt changed")


@dataclass(frozen=True)
class PaperKineticCompiledFramewiseFullGeometryControlResult:
    precompile_receipt: PaperKineticCompiledFramewisePrecompileReceipt
    update_receipt: PaperKineticCompiledFramewiseManualSGDReceipt
    accounting: Mapping[str, Any]
    parity_payload: Mapping[str, Any] | None
    generation_digest: str
    provenance: str = CONTROL_PROVENANCE
    runtime_status: str = CONTROL_STATUS
    native_runtime_verified: bool = False
    allocator_peak_measured: bool = False
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        self.precompile_receipt.assert_current()
        self.update_receipt.assert_current()
        if (
            self._seal is not _RESULT_SEAL
            or self.provenance != CONTROL_PROVENANCE
            or self.runtime_status != CONTROL_STATUS
            or self.native_runtime_verified
            or self.allocator_peak_measured
            or not isinstance(self.accounting, MappingProxyType)
            or self.accounting.get("per_frame_replay_count", 0) < 1
            or self.accounting.get("maximum_simultaneously_live_frame_count") != 1
            or self.accounting.get("cpu_optimizer_mutation_count") != 1
            or self.accounting.get("fresh_selected_track_recompile_count") != 0
            or self.accounting.get("persistent_frame_tensor_bytes") != 0
            or self.accounting.get("persistent_sample_tensor_bytes") != 0
            or self.accounting.get("persistent_target_tensor_bytes") != 0
            or self.accounting.get("persistent_prediction_tensor_bytes") != 0
            or self.generation_digest != _result_digest(self)
        ):
            raise ValueError("compiled framewise control result changed")


@dataclass(frozen=True)
class _AggregateAuthorization:
    grad_site_rgba_f32: torch.Tensor = field(repr=False)
    grad_positions0_f64: torch.Tensor = field(repr=False)
    grad_velocities_f64: torch.Tensor = field(repr=False)
    grad_weight_coefficients_f64: torch.Tensor = field(repr=False)

    def _tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.grad_site_rgba_f32,
            self.grad_positions0_f64,
            self.grad_velocities_f64,
            self.grad_weight_coefficients_f64,
        )


@dataclass(frozen=True)
class _SingleFrameObservations:
    frame_index: int
    pixel_ids: tuple[int, ...]
    dataset_frame_count: int

    @property
    def expected_observation_count(self) -> int:
        return len(self.pixel_ids)

    def __iter__(self) -> Iterator[PaperKineticObservation]:
        for track_position, pixel_id in enumerate(self.pixel_ids):
            yield PaperKineticObservation(
                observation_id=(
                    track_position * self.dataset_frame_count + self.frame_index
                ),
                view_index=0,
                frame_index=self.frame_index,
                pixel_index=pixel_id,
            )


@torch.no_grad()
def run_paper_kinetic_compiled_framewise_full_geometry_control(
    state: PaperKineticFixedCameraCombinedState,
    provider: PaperKineticCompiledFramewiseProgramProvider,
    artifact_store: PaperKineticCompiledCpuArtifactStore,
    *,
    selected_frame_indices: Sequence[int],
    selected_track_ids: Sequence[int],
    global_site_rgba_f32: torch.Tensor,
    background_rgb_f32: torch.Tensor,
    background_generation_id: str,
    native_ops: Any,
    maximum_samples_per_launch: int,
    cone_tolerance: float,
    memory_policy: PaperKineticLazyNativeMemoryPolicy,
    full_geometry_memory_policy: PaperKineticLazyFullGeometryMemoryPolicy,
    combined_sgd_policy: PaperKineticFixedCameraCombinedSGDPolicy,
    device_completion_fence: Callable[[], None],
    device_completion_fence_provenance: str,
    emit_parity_payload: bool = False,
) -> PaperKineticCompiledFramewiseFullGeometryControlResult:
    """Run the publishable O(1)-frame-scratch same-representation control."""

    if not isinstance(provider, PaperKineticCompiledFramewiseProgramProvider):
        raise TypeError("framewise control requires its compiled provider subtype")
    if not isinstance(state, PaperKineticFixedCameraCombinedState):
        raise TypeError("framewise control requires the combined trainable state")
    if not isinstance(artifact_store, PaperKineticCompiledCpuArtifactStore):
        raise TypeError("framewise control requires its bounded artifact store")
    if not callable(device_completion_fence):
        raise TypeError("framewise control requires a device completion fence")
    if not isinstance(device_completion_fence_provenance, str) or not device_completion_fence_provenance.strip():
        raise ValueError("device completion fence provenance must be nonempty")
    if not isinstance(emit_parity_payload, bool):
        raise TypeError("emit_parity_payload must be bool")
    _require_positive_int(maximum_samples_per_launch, name="maximum_samples_per_launch")
    if not math.isfinite(float(cone_tolerance)) or float(cone_tolerance) <= 0.0:
        raise ValueError("cone_tolerance must be finite and positive")
    _require_sha256(background_generation_id, name="background_generation_id")
    combined_sgd_policy.assert_valid()
    state.assert_current(provider, artifact_store)
    provider.assert_current()
    controller = provider.framewise_controller
    if not isinstance(controller, _PaperKineticCompiledFramewiseController):
        raise ValueError("framewise control provider lost its compiler controller")
    if controller.artifact_store is not artifact_store:
        raise ValueError("framewise control provider/store identity changed")
    precompile = controller.precompile_receipt
    if not isinstance(precompile, PaperKineticCompiledFramewisePrecompileReceipt):
        raise ValueError("framewise control has no one-time compile receipt")
    precompile.assert_current()
    frames = tuple(int(value) for value in selected_frame_indices)
    tracks = tuple(int(value) for value in selected_track_ids)
    if (
        not frames
        or tuple(sorted(set(frames))) != frames
        or frames[0] < 0
        or frames[-1] >= provider.frame_count
        or tracks != controller.selected_track_ids
    ):
        raise ValueError("framewise control frame/track manifest changed")
    device = global_site_rgba_f32.device
    if (
        global_site_rgba_f32.dtype != torch.float32
        or tuple(global_site_rgba_f32.shape) != (state.site_count, 4)
        or background_rgb_f32.device != device
        or background_rgb_f32.dtype != torch.float32
        or tuple(background_rgb_f32.shape) != (3,)
        or global_site_rgba_f32.requires_grad
        or background_rgb_f32.requires_grad
    ):
        raise ValueError("framewise control device material/background changed")
    if not all(
        bool(torch.isfinite(tensor).all().item())
        for tensor in (global_site_rgba_f32, background_rgb_f32)
    ):
        raise FloatingPointError("framewise control device snapshot is nonfinite")

    trainer_state = prepare_paper_kinetic_lazy_native_trainer_state(
        provider,
        device=device,
        initial_step_index=0,
    )
    global_material_bar = torch.zeros(
        (state.site_count, 4), dtype=torch.float32, device="cpu"
    )
    global_position_bar = torch.zeros_like(state.positions0_f64, device="cpu")
    global_velocity_bar = torch.zeros_like(state.velocities_f64, device="cpu")
    global_weight_bar = torch.zeros_like(
        state.weight_coefficients_f64, device="cpu"
    )
    global_loss = torch.zeros((1,), dtype=torch.float32, device="cpu")
    frame_count = len(frames)
    scale = 1.0 / float(frame_count)
    live_frame_count = 0
    maximum_live_frame_count = 0
    release_fence_count = 0
    readback_chain = _digest_parts(
        FRAME_READBACK_PROVENANCE,
        "chain-root",
        provider.generation_digest,
        frames,
        tracks,
    )
    result_chain = _digest_parts(
        CONTROL_PROVENANCE,
        "frame-result-chain-root",
        precompile.generation_digest,
    )
    frame_wall_times: list[float] = []
    sum_keys = (
        "eligible_native_block_count",
        "active_native_block_count",
        "native_node_forward_launch_count",
        "native_sample_prepare_count",
        "native_sample_launch_count",
        "native_sample_completion_fence_count",
        "native_full_geometry_vjp_launch_count",
        "native_fused_union_v2_transaction_count",
        "geometry_d2h_completion_fence_count",
        "streamed_sample_count",
        "ordered_word_node_interactions",
        "sample_to_node_linear_interactions",
        "sample_to_node_dense_fallback_interactions",
        "selected_pixel_read_call_count",
        "direct_selected_pixel_observation_count",
        "full_frame_target_materialization_count",
        "camera_ray_slice_work_count",
        "camera_ray_slice_scalar_count",
    )
    max_keys = (
        "peak_lane_resident_logical_tensor_bytes",
        "peak_active_node_state_tensor_bytes",
        "peak_sample_launch_tensor_bytes",
        "peak_decoded_frame_scratch_upper_bound_bytes",
        "peak_selected_frame_target_tensor_upper_bound_bytes",
        "peak_coordinator_visible_live_tensor_upper_bound_bytes",
        "maximum_geometry_bridge_visible_peak_logical_tensor_bytes",
    )
    totals = {key: 0 for key in sum_keys}
    peaks = {key: 0 for key in max_keys}
    selected_pixel_modes: set[str] = set()
    selected_pixel_source_digests: set[str] = set()
    frame_result_generation_digests: list[str] = []
    d2h_tensor_bytes = 0
    try:
        for frame_position, frame_index in enumerate(frames):
            if live_frame_count:
                raise RuntimeError("framewise control retained a previous frame")
            live_frame_count = 1
            maximum_live_frame_count = max(maximum_live_frame_count, live_frame_count)
            observations = _SingleFrameObservations(
                frame_index=frame_index,
                pixel_ids=tracks,
                dataset_frame_count=provider.frame_count,
            )
            manifest_digest = paper_kinetic_observation_manifest_digest(observations)
            frame_material_bar = torch.empty_like(global_site_rgba_f32)
            frame_position_bar = torch.empty_like(state.positions0_f64, device="cpu")
            frame_velocity_bar = torch.empty_like(state.velocities_f64, device="cpu")
            frame_weight_bar = torch.empty_like(
                state.weight_coefficients_f64, device="cpu"
            )
            captures: list[PaperKineticLazyNativeFullGeometryStepResult] = []
            started = time.perf_counter()
            result = run_paper_kinetic_lazy_native_full_geometry_step(
                trainer_state,
                provider,
                observations,
                step_index=frame_position,
                expected_observation_count=observations.expected_observation_count,
                expected_observation_manifest_digest=manifest_digest,
                loss_normalization_id=_digest_parts(
                    CONTROL_PROVENANCE,
                    "per-frame-rgb-mean",
                    observations.expected_observation_count * 3,
                ),
                material_generation_id=state.material_state.material_generation_id,
                geometry_generation_id=state.geometry_generation_id,
                background_generation_id=background_generation_id,
                global_site_rgba_f32=global_site_rgba_f32,
                global_grad_site_rgba_f32=frame_material_bar,
                grad_positions0_f64_cpu=frame_position_bar,
                grad_velocities_f64_cpu=frame_velocity_bar,
                grad_weight_coefficients_f64_cpu=frame_weight_bar,
                background_rgb_f32=background_rgb_f32,
                native_ops=native_ops,
                maximum_samples_per_launch=maximum_samples_per_launch,
                memory_policy=memory_policy,
                full_geometry_memory_policy=full_geometry_memory_policy,
                reverse_mode=STAGED_SPARSE,
                optimizer_update=captures.append,
                cone_tolerance=float(cone_tolerance),
            )
            frame_wall_times.append(time.perf_counter() - started)
            if captures != [result]:
                raise ArithmeticError("framewise scheduler callback coverage changed")
            result.assert_current()
            frame_material_cpu = result.grad_global_site_rgba_f32.detach().to(
                device="cpu"
            )
            frame_loss_cpu = result.loss_f32.detach().to(device="cpu")
            returned = device_completion_fence()
            if returned is not None:
                raise TypeError("framewise release fence must return None")
            release_fence_count += 1
            if not all(
                bool(torch.isfinite(tensor).all().item())
                for tensor in (
                    frame_material_cpu,
                    frame_loss_cpu,
                    result.grad_positions0_f64_cpu,
                    result.grad_velocities_f64_cpu,
                    result.grad_weight_coefficients_f64_cpu,
                )
            ):
                raise FloatingPointError("framewise native replay produced nonfinite bars")
            global_material_bar.add_(frame_material_cpu, alpha=scale)
            global_position_bar.add_(result.grad_positions0_f64_cpu, alpha=scale)
            global_velocity_bar.add_(result.grad_velocities_f64_cpu, alpha=scale)
            global_weight_bar.add_(
                result.grad_weight_coefficients_f64_cpu, alpha=scale
            )
            global_loss.add_(frame_loss_cpu, alpha=scale)
            d2h_tensor_bytes += _tensor_bytes(frame_material_cpu, frame_loss_cpu)
            readback_digest = _digest_parts(
                FRAME_READBACK_PROVENANCE,
                frame_position,
                frame_index,
                result.generation_digest,
                _tensor_content_digest(frame_material_cpu),
                _tensor_content_digest(frame_loss_cpu),
                device_completion_fence_provenance,
            )
            readback_chain = _digest_parts(
                FRAME_READBACK_PROVENANCE,
                "chain-link",
                readback_chain,
                readback_digest,
            )
            result_chain = _digest_parts(
                CONTROL_PROVENANCE,
                "frame-result-chain-link",
                result_chain,
                frame_position,
                frame_index,
                result.generation_digest,
                tuple(
                    receipt.generation_digest
                    for receipt in result.geometry_d2h_receipts
                ),
            )
            frame_result_generation_digests.append(result.generation_digest)
            for key in sum_keys:
                totals[key] += int(result.accounting[key])
            for key in max_keys:
                peaks[key] = max(peaks[key], int(result.accounting[key]))
            selected_pixel_modes.add(str(result.accounting["selected_pixel_read_mode"]))
            selected_pixel_source_digests.add(
                str(
                    result.accounting[
                        "selected_pixel_read_source_provenance_manifest_digest"
                    ]
                )
            )
            if any(
                int(result.accounting[key]) != 0
                for key in (
                    "persistent_frame_tensor_bytes",
                    "persistent_sample_tensor_bytes",
                    "persistent_target_tensor_bytes",
                    "persistent_prediction_tensor_bytes",
                )
            ):
                raise MemoryError("framewise native result retained frame-axis state")
            del (
                result,
                captures,
                frame_material_cpu,
                frame_loss_cpu,
                frame_material_bar,
                frame_position_bar,
                frame_velocity_bar,
                frame_weight_bar,
                observations,
            )
            live_frame_count = 0
    except BaseException:
        state.poisoned = True
        state.material_state.poisoned = True
        raise
    if live_frame_count or maximum_live_frame_count != 1:
        raise ArithmeticError("framewise live-frame ledger changed")
    expected_bundle_replays = frame_count * precompile.request_count
    store_after_replay = artifact_store.report()
    if (
        controller.frame_bundle_acquisition_count != expected_bundle_replays
        or controller.frame_bundle_warm_hit_count != expected_bundle_replays
        or controller.frame_bundle_cold_compile_count
        or store_after_replay.cold_compile_count != precompile.request_count
        or store_after_replay.cold_compiled_track_count != precompile.track_count
        or store_after_replay.hit_count != expected_bundle_replays
        or store_after_replay.eviction_count
    ):
        raise ArithmeticError("framewise replay changed its compile-once contract")

    gradient_digest = _digest_parts(
        CONTROL_PROVENANCE,
        "four-global-bars",
        _tensor_content_digest(global_material_bar),
        _tensor_content_digest(global_position_bar),
        _tensor_content_digest(global_velocity_bar),
        _tensor_content_digest(global_weight_bar),
        _tensor_content_digest(global_loss),
    )
    update_authorization_digest = _digest_parts(
        UPDATE_PROVENANCE,
        state.generation_digest,
        provider.generation_digest,
        precompile.generation_digest,
        result_chain,
        readback_chain,
        gradient_digest,
        combined_sgd_policy.generation_digest,
    )
    authorization = _AggregateAuthorization(
        grad_site_rgba_f32=global_material_bar,
        grad_positions0_f64=global_position_bar,
        grad_velocities_f64=global_velocity_bar,
        grad_weight_coefficients_f64=global_weight_bar,
    )
    parameters_before_digest = _parameter_digest(state)
    candidates = _build_update_candidates(
        state,
        authorization,  # type: ignore[arg-type]
        policy=combined_sgd_policy,
    )
    parity_payload: Mapping[str, Any] | None = None
    if emit_parity_payload:
        parity_payload = MappingProxyType(
            {
                "loss": float(global_loss.item()),
                "material_gradient": _flat_finite_values(global_material_bar),
                "geometry_gradient": _flat_finite_values(
                    global_position_bar,
                    global_velocity_bar,
                    global_weight_bar,
                ),
                "parameters_after_step": _flat_finite_values(
                    candidates.raw_color_f32,
                    candidates.raw_density_f32,
                    candidates.positions0_f64,
                    candidates.velocities_f64,
                    candidates.weight_coefficients_f64,
                ),
            }
        )
    mutation_started = False
    try:
        mutation_started = True
        state.material_state.raw_color_f32.copy_(candidates.raw_color_f32)
        state.material_state.raw_density_f32.copy_(candidates.raw_density_f32)
        state.material_state.site_rgba_f32.copy_(candidates.site_rgba_f32)
        state.positions0_f64.copy_(candidates.positions0_f64)
        state.velocities_f64.copy_(candidates.velocities_f64)
        state.weight_coefficients_f64.copy_(candidates.weight_coefficients_f64)
        parameters_after_digest = _parameter_digest(state)
        state.active = False
        state.poisoned = True
        state.material_state.poisoned = True
    except BaseException:
        if mutation_started:
            state.poisoned = True
            state.material_state.poisoned = True
        raise
    provisional_update = PaperKineticCompiledFramewiseManualSGDReceipt(
        loss=float(global_loss.item()),
        material_gradient_l2_norm=float(
            torch.linalg.vector_norm(global_material_bar).item()
        ),
        position_gradient_l2_norm=candidates.position_gradient_norm,
        velocity_gradient_l2_norm=candidates.velocity_gradient_norm,
        weight_gradient_l2_norm=candidates.weight_gradient_norm,
        raw_color_parameter_delta_l2_norm=float(
            torch.linalg.vector_norm(
                candidates.raw_color_f32 - state.material_state.raw_color_f32
            ).item()
        ),
        raw_density_parameter_delta_l2_norm=float(
            torch.linalg.vector_norm(
                candidates.raw_density_f32 - state.material_state.raw_density_f32
            ).item()
        ),
        positions0_parameter_delta_l2_norm=float(
            torch.linalg.vector_norm(
                candidates.positions0_f64 - state.positions0_f64
            ).item()
        ),
        velocities_parameter_delta_l2_norm=float(
            torch.linalg.vector_norm(
                candidates.velocities_f64 - state.velocities_f64
            ).item()
        ),
        weight_coefficients_parameter_delta_l2_norm=float(
            torch.linalg.vector_norm(
                candidates.weight_coefficients_f64
                - state.weight_coefficients_f64
            ).item()
        ),
        parameters_before_digest=parameters_before_digest,
        parameters_after_digest=parameters_after_digest,
        gradient_digest=gradient_digest,
        update_authorization_digest=update_authorization_digest,
        generation_digest="",
        _seal=_UPDATE_SEAL,
    )
    # Candidate-minus-live is zero after the in-place terminal copy.  Replace
    # those five values with the norms computed by the shared candidate builder
    # and explicit before/after deltas retained as scalar receipts.
    delta_norms = _candidate_delta_norms_from_gradient_receipt(
        state,
        candidates,
        combined_sgd_policy=combined_sgd_policy,
    )
    provisional_update = replace(
        provisional_update,
        raw_color_parameter_delta_l2_norm=delta_norms[0],
        raw_density_parameter_delta_l2_norm=delta_norms[1],
        positions0_parameter_delta_l2_norm=delta_norms[2],
        velocities_parameter_delta_l2_norm=delta_norms[3],
        weight_coefficients_parameter_delta_l2_norm=delta_norms[4],
    )
    update = replace(
        provisional_update,
        generation_digest=_update_digest(provisional_update),
    )
    update.assert_current()
    global_bar_bytes = _tensor_bytes(
        global_material_bar,
        global_position_bar,
        global_velocity_bar,
        global_weight_bar,
        global_loss,
    )
    frame_material_bar_bytes = _tensor_bytes(global_site_rgba_f32)
    frame_geometry_bar_bytes = state.geometry_tensor_bytes
    frame_material_readback_and_loss_bytes = frame_material_bar_bytes + 4
    coordinator_visible_bytes = peaks[
        "peak_coordinator_visible_live_tensor_upper_bound_bytes"
    ]
    geometry_bridge_visible_bytes = peaks[
        "maximum_geometry_bridge_visible_peak_logical_tensor_bytes"
    ]
    if coordinator_visible_bytes < frame_material_bar_bytes:
        raise ArithmeticError(
            "framewise coordinator bound omitted its live material bar"
        )
    # The coordinator bound already charges ``frame_material_bar`` through its
    # fixed-tensor set.  The geometry bridge executes while that coordinator
    # lifetime and the caller-owned CPU geometry bars remain live, so it must be
    # added rather than selected with ``max``.  After the scheduler returns, the
    # CPU material/loss readback overlaps the returned result and those same
    # frame-local bars.  Summing all four independently-owned components is a
    # conservative union bound for both phases; it intentionally does not
    # assume allocator reuse between them.
    maximum_frame_scratch = (
        coordinator_visible_bytes
        + geometry_bridge_visible_bytes
        + frame_geometry_bar_bytes
        + frame_material_readback_and_loss_bytes
    )
    accounting = MappingProxyType(
        {
            "control_mode": "per_frame_replay_sequential",
            "same_continuous_compiled_representation": True,
            "reverse_mode": STAGED_SPARSE,
            "selected_frame_count": frame_count,
            "selected_track_count": len(tracks),
            "per_frame_replay_count": frame_count,
            "per_frame_replay_wall_time_seconds": tuple(frame_wall_times),
            "step_wall_time_seconds": float(sum(frame_wall_times)),
            "one_time_continuous_compile_pass_count": 1,
            "one_time_continuous_compile_request_count": precompile.request_count,
            "one_time_continuous_compile_track_count": precompile.track_count,
            "per_frame_continuous_recompile_count": 0,
            "fresh_selected_track_recompile_count": 0,
            "compiled_artifact_warm_acquisition_count": expected_bundle_replays,
            "compiled_artifact_warm_hit_count": expected_bundle_replays,
            "frame_result_generation_digest_chain": result_chain,
            "frame_readback_receipt_chain_digest": readback_chain,
            "frame_readback_receipt_count": frame_count,
            "frame_release_fence_call_count": release_fence_count,
            "frame_release_fence_provenance": device_completion_fence_provenance,
            "maximum_simultaneously_live_frame_count": maximum_live_frame_count,
            "maximum_in_flight_frame_target_count": 1,
            "maximum_in_flight_frame_prediction_count": 1,
            "maximum_in_flight_frame_reverse_count": 1,
            "frame_result_capability_retained_after_release_count": 0,
            "frame_target_released_before_next_frame": True,
            "frame_prediction_released_before_next_frame": True,
            "frame_reverse_scratch_released_before_next_frame": True,
            "cpu_optimizer_mutation_count": 1,
            "geometry_mutation_count": 1,
            "combined_optimizer_authorization_count": 1,
            "stale_provider_store_retirement_count": 0,
            "terminal_control_generation_invalidated_after_mutation": True,
            "compiler_work_receipt_chain_digest": (
                precompile.compiler_work_receipt_chain_digest
            ),
            "compiler_work_receipt_count": precompile.compiler_work_receipt_count,
            "root_complement_witness_count": precompile.root_complement_witness_count,
            "candidate_source_attempt_count": precompile.candidate_source_attempt_count,
            "all_site_witness_check_count": precompile.all_site_witness_check_count,
            "unique_pair_difference_count": precompile.unique_pair_difference_count,
            **totals,
            **peaks,
            "selected_pixel_read_modes": tuple(sorted(selected_pixel_modes)),
            "selected_pixel_source_manifest_digests": tuple(
                sorted(selected_pixel_source_digests)
            ),
            "direct_selected_pixel_target_stream": (
                totals["full_frame_target_materialization_count"] == 0
                and totals["direct_selected_pixel_observation_count"]
                == frame_count * len(tracks)
            ),
            "global_material_bar_logical_tensor_bytes": _tensor_bytes(
                global_material_bar
            ),
            "global_geometry_bar_logical_tensor_bytes": _tensor_bytes(
                global_position_bar,
                global_velocity_bar,
                global_weight_bar,
            ),
            "global_bar_and_loss_logical_tensor_bytes": global_bar_bytes,
            "frame_material_bar_logical_tensor_bytes": frame_material_bar_bytes,
            "frame_geometry_bar_logical_tensor_bytes": frame_geometry_bar_bytes,
            "frame_material_readback_and_loss_logical_tensor_bytes": (
                frame_material_readback_and_loss_bytes
            ),
            "frame_coordinator_visible_logical_tensor_bytes_upper_bound": (
                coordinator_visible_bytes
            ),
            "frame_geometry_bridge_visible_logical_tensor_bytes_upper_bound": (
                geometry_bridge_visible_bytes
            ),
            "frame_material_bar_included_in_coordinator_bound": True,
            "frame_geometry_bridge_may_overlap_coordinator": True,
            "frame_readback_cumulative_tensor_bytes": d2h_tensor_bytes,
            "compiled_program_store_resident_accounted_bytes": (
                store_after_replay.current_resident_accounted_bytes
            ),
            "compiled_program_store_peak_resident_accounted_bytes": (
                store_after_replay.peak_resident_accounted_bytes
            ),
            "combined_live_state_logical_tensor_bytes": (
                state.total_persistent_tensor_bytes
            ),
            "maximum_frame_local_logical_tensor_bytes_upper_bound": (
                maximum_frame_scratch
            ),
            "expensive_live_logical_and_accounted_peak_upper_bound_bytes": (
                state.total_persistent_tensor_bytes
                + store_after_replay.current_resident_accounted_bytes
                + global_bar_bytes
                + maximum_frame_scratch
            ),
            "expensive_peak_is_frame_count_invariant": True,
            "persistent_frame_tensor_bytes": 0,
            "persistent_sample_tensor_bytes": 0,
            "persistent_target_tensor_bytes": 0,
            "persistent_prediction_tensor_bytes": 0,
            "optimizer_history_tensor_bytes": 0,
            "camera_time_scalar_count": len(provider.frame_times),
            "camera_time_slice_is_allowed_linear_metadata": True,
            "full_frame_target_tensor_materialized": False,
            "full_image_compile_used": False,
            "scalar_fixed_time_topology_discovery_used": False,
            "allocator_peak_measured": False,
            "rss_peak_measured": False,
            "native_runtime_verified": False,
            "frame_result_generation_digests_retained": tuple(
                frame_result_generation_digests
            ),
        }
    )
    provisional_result = PaperKineticCompiledFramewiseFullGeometryControlResult(
        precompile_receipt=precompile,
        update_receipt=update,
        accounting=accounting,
        parity_payload=parity_payload,
        generation_digest="",
        _seal=_RESULT_SEAL,
    )
    result = replace(
        provisional_result,
        generation_digest=_result_digest(provisional_result),
    )
    result.assert_current()
    return result


def _raise_cold_recompile() -> PaperKineticCompiledCpuArtifact:
    raise RuntimeError("continuous selected-track programs may compile only once")


def _parameter_digest(state: PaperKineticFixedCameraCombinedState) -> str:
    return _digest_parts(
        CONTROL_PROVENANCE,
        "parameter-state",
        _tensor_content_digest(state.material_state.raw_color_f32),
        _tensor_content_digest(state.material_state.raw_density_f32),
        _tensor_content_digest(state.positions0_f64),
        _tensor_content_digest(state.velocities_f64),
        _tensor_content_digest(state.weight_coefficients_f64),
    )


def _flat_finite_values(*tensors: torch.Tensor) -> tuple[float, ...]:
    values: list[float] = []
    for tensor in tensors:
        value = tensor.detach().to(device="cpu").reshape(-1)
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError("framewise parity tensor is nonfinite")
        values.extend(float(item) for item in value.tolist())
    return tuple(values)


def _candidate_delta_norms_from_gradient_receipt(
    state: PaperKineticFixedCameraCombinedState,
    candidates: Any,
    *,
    combined_sgd_policy: PaperKineticFixedCameraCombinedSGDPolicy,
) -> tuple[float, float, float, float, float]:
    """Recover exact stateless-SGD deltas after the terminal in-place copy."""

    material_policy = state.material_state.optimizer_policy
    return (
        float(material_policy.color_learning_rate * candidates.raw_color_gradient_norm),
        float(material_policy.density_learning_rate * candidates.raw_density_gradient_norm),
        float(combined_sgd_policy.position_learning_rate * candidates.position_gradient_norm),
        float(combined_sgd_policy.velocity_learning_rate * candidates.velocity_gradient_norm),
        float(combined_sgd_policy.weight_learning_rate * candidates.weight_gradient_norm),
    )


def _precompile_digest(
    receipt: PaperKineticCompiledFramewisePrecompileReceipt,
) -> str:
    return _digest_parts(
        receipt.provenance,
        receipt.provider_generation_digest,
        receipt.selected_track_manifest_digest,
        receipt.request_count,
        receipt.track_count,
        receipt.artifact_generation_digests,
        receipt.compile_receipt_generation_digests,
        receipt.compiler_work_receipt_chain_digest,
        receipt.compiler_work_receipt_count,
        receipt.root_complement_witness_count,
        receipt.candidate_source_attempt_count,
        receipt.all_site_witness_check_count,
        receipt.unique_pair_difference_count,
        receipt.store_current_resident_accounted_bytes,
        receipt.store_peak_resident_accounted_bytes,
        receipt.store_maximum_resident_accounted_bytes,
        receipt.compile_pass_count,
        receipt.requested_frame_sampling_used,
    )


def _update_digest(receipt: PaperKineticCompiledFramewiseManualSGDReceipt) -> str:
    return _digest_parts(
        receipt.provenance,
        receipt.loss,
        receipt.material_gradient_l2_norm,
        receipt.position_gradient_l2_norm,
        receipt.velocity_gradient_l2_norm,
        receipt.weight_gradient_l2_norm,
        receipt.raw_color_parameter_delta_l2_norm,
        receipt.raw_density_parameter_delta_l2_norm,
        receipt.positions0_parameter_delta_l2_norm,
        receipt.velocities_parameter_delta_l2_norm,
        receipt.weight_coefficients_parameter_delta_l2_norm,
        receipt.parameters_before_digest,
        receipt.parameters_after_digest,
        receipt.gradient_digest,
        receipt.update_authorization_digest,
        receipt.cpu_optimizer_mutation_count,
        receipt.geometry_mutation_count,
        receipt.stale_provider_store_retirement_count,
        receipt.fresh_selected_track_recompile_count,
        receipt.optimizer_history_tensor_bytes,
        receipt.terminal_control_generation,
    )


def _result_digest(
    result: PaperKineticCompiledFramewiseFullGeometryControlResult,
) -> str:
    return _digest_parts(
        result.provenance,
        result.runtime_status,
        result.precompile_receipt.generation_digest,
        result.update_receipt.generation_digest,
        tuple(sorted(result.accounting.items())),
        None
        if result.parity_payload is None
        else tuple(sorted(result.parity_payload.items())),
        result.native_runtime_verified,
        result.allocator_peak_measured,
    )


__all__ = (
    "CONTROL_PROVENANCE",
    "PaperKineticCompiledFramewiseFullGeometryControlResult",
    "PaperKineticCompiledFramewiseManualSGDReceipt",
    "PaperKineticCompiledFramewisePrecompileReceipt",
    "PaperKineticCompiledFramewiseProgramProvider",
    "prepare_paper_kinetic_compiled_framewise_program_provider",
    "run_paper_kinetic_compiled_framewise_full_geometry_control",
)
