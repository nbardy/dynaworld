"""Bounded lazy-recompile promotion for streamed WorldFoam training.

The older combined-state transaction eagerly recompiles an explicitly named
working set after every geometry update.  That is useful for restart fixtures,
but it is the wrong ownership model for a public all-pixel trainer: a
``384 x 512`` camera contains 196,608 tracks and the next stochastic step may
use another camera.  Retaining or eagerly compiling that camera would erase
the memory advantage that the lazy spatial-bundle renderer is meant to test.

This module consumes one complete CPU gradient authorization, applies the
existing material chain rule and geometry SGD, retires the stale provider and
artifact store, and publishes a fresh provider with an empty bounded store.
The fresh generation is sealed for *lazy on-demand recompilation*: its first
render request compiles only the bounded spatial bundle it actually consumes.
No target, ray, frame, prediction, compiled program, or optimizer-history
tensor is retained by the promoted state.

This is a production source boundary, not evidence that the native route fits
in memory.  Allocator/RSS evidence belongs to the executor that runs it.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import torch
from paper_kinetic_runtime_paths import ensure_worldfoam_lane2_research_path


ensure_worldfoam_lane2_research_path()

from kinetic_compiled_cpu_artifact_store import (  # noqa: E402
    PaperKineticCompiledCpuArtifactStore,
    PaperKineticCompiledCpuArtifactStorePolicy,
)
from kinetic_power_word_compiler import AffineKineticPowerSites  # noqa: E402
import paper_kinetic_fixed_camera_combined_state as _combined  # noqa: E402
from paper_kinetic_fixed_camera_combined_state import (  # noqa: E402
    PaperKineticFixedCameraCombinedSGDPolicy,
    PaperKineticFixedCameraCombinedState,
)
from paper_kinetic_lazy_program_bundles import (  # noqa: E402
    PaperKineticLazyProgramBundleProvider,
    prepare_paper_kinetic_lazy_program_bundle_provider,
)


AUTHORIZATION_PROVENANCE = (
    "paper-kinetic-streaming-combined-gradient-authorization-v1"
)
PROMOTION_PROVENANCE = "paper-kinetic-streaming-combined-promotion-v1"
LAZY_RECOMPILE_PROVENANCE = (
    "paper-kinetic-lazy-on-demand-recompile-generation-seal-v1"
)

_AUTHORIZATION_SEAL = object()
_READY_SEAL = object()


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(int(tensor.numel()) * int(tensor.element_size()) for tensor in tensors)


def _content_digest(tensor: torch.Tensor) -> str:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("streaming authorization digests require contiguous CPU tensors")
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(repr(tuple(int(value) for value in tensor.shape)).encode("ascii"))
    digest.update(memoryview(tensor.detach().numpy()).cast("B"))
    return digest.hexdigest()


@dataclass
class PaperKineticStreamingCombinedGradientAuthorization:
    """One-shot, CPU-owned global RGB-mean gradient for one optimizer step."""

    source_state_identity: int
    source_state_generation_digest: str
    source_provider_identity: int
    source_provider_generation_digest: str
    source_artifact_store_identity: int
    step_index: int
    step_generation_id: str
    observation_count: int
    loss_f32: torch.Tensor = field(repr=False)
    grad_site_rgba_f32: torch.Tensor = field(repr=False)
    grad_positions0_f64: torch.Tensor = field(repr=False)
    grad_velocities_f64: torch.Tensor = field(repr=False)
    grad_weight_coefficients_f64: torch.Tensor = field(repr=False)
    tensor_content_digests: tuple[str, ...]
    generation_digest: str
    consumed: bool = False
    provenance: str = AUTHORIZATION_PROVENANCE
    _seal: object = field(default=None, repr=False)

    def _tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.grad_site_rgba_f32,
            self.loss_f32,
            self.grad_positions0_f64,
            self.grad_velocities_f64,
            self.grad_weight_coefficients_f64,
        )

    @property
    def logical_tensor_bytes(self) -> int:
        return _tensor_bytes(*self._tensors())

    def assert_current(
        self,
        state: PaperKineticFixedCameraCombinedState,
        provider: PaperKineticLazyProgramBundleProvider,
        artifact_store: PaperKineticCompiledCpuArtifactStore,
    ) -> None:
        state.assert_current(provider, artifact_store)
        tensors = self._tensors()
        expected_shapes = (
            (state.site_count, 4),
            (1,),
            tuple(state.positions0_f64.shape),
            tuple(state.velocities_f64.shape),
            tuple(state.weight_coefficients_f64.shape),
        )
        expected_dtypes = (
            torch.float32,
            torch.float32,
            torch.float64,
            torch.float64,
            torch.float64,
        )
        if (
            self._seal is not _AUTHORIZATION_SEAL
            or self.provenance != AUTHORIZATION_PROVENANCE
            or self.consumed
            or self.source_state_identity != id(state)
            or self.source_state_generation_digest != state.generation_digest
            or self.source_provider_identity != id(provider)
            or self.source_provider_generation_digest != provider.generation_digest
            or self.source_artifact_store_identity != id(artifact_store)
            or self.step_index != state.geometry_update_count
            or not _is_sha256(self.step_generation_id)
            or self.observation_count < 1
            or len({tensor.untyped_storage().data_ptr() for tensor in tensors})
            != len(tensors)
        ):
            raise ValueError("streaming combined authorization is stale or foreign")
        for tensor, shape, dtype in zip(
            tensors,
            expected_shapes,
            expected_dtypes,
            strict=True,
        ):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tensor.dtype != dtype
                or tensor.layout != torch.strided
                or tuple(tensor.shape) != shape
                or not tensor.is_contiguous()
                or tensor.requires_grad
                or not bool(torch.isfinite(tensor).all().item())
            ):
                raise ValueError("streaming combined authorization tensor changed")
        current_content = tuple(_content_digest(tensor) for tensor in tensors)
        if (
            current_content != self.tensor_content_digests
            or self.generation_digest != _authorization_digest(self)
        ):
            raise ValueError("streaming combined authorization contents changed")


@dataclass(frozen=True)
class PaperKineticStreamingCombinedPromotionReceipt:
    step_index_before: int
    step_index_after: int
    step_generation_id: str
    authorization_generation_digest: str
    source_state_generation_digest: str
    promoted_state_generation_digest: str
    source_provider_generation_digest: str
    promoted_provider_generation_digest: str
    source_world_generation_digest: str
    promoted_world_generation_digest: str
    material_generation_id_before: str
    material_generation_id_after: str
    geometry_generation_id_before: str
    geometry_generation_id_after: str
    lazy_recompile_generation_digest: str
    observation_count: int
    loss: float
    learning_rate_multiplier: float
    authorization_logical_tensor_bytes: int
    source_state_logical_tensor_bytes: int
    candidate_logical_tensor_bytes: int
    candidate_world_geometry_logical_tensor_bytes: int
    transaction_logical_and_accounted_peak_upper_bound_bytes: int
    old_artifact_store_resident_accounted_bytes: int
    fresh_artifact_store_resident_accounted_bytes: int
    persistent_frame_tensor_bytes: int
    persistent_sample_tensor_bytes: int
    persistent_target_tensor_bytes: int
    persistent_prediction_tensor_bytes: int
    optimizer_history_tensor_bytes: int
    generation_digest: str
    provenance: str = PROMOTION_PROVENANCE

    def assert_self_consistent(self) -> None:
        digest_fields = (
            self.step_generation_id,
            self.authorization_generation_digest,
            self.source_state_generation_digest,
            self.promoted_state_generation_digest,
            self.source_provider_generation_digest,
            self.promoted_provider_generation_digest,
            self.source_world_generation_digest,
            self.promoted_world_generation_digest,
            self.material_generation_id_before,
            self.material_generation_id_after,
            self.geometry_generation_id_before,
            self.geometry_generation_id_after,
            self.lazy_recompile_generation_digest,
            self.generation_digest,
        )
        byte_fields = (
            self.authorization_logical_tensor_bytes,
            self.source_state_logical_tensor_bytes,
            self.candidate_logical_tensor_bytes,
            self.candidate_world_geometry_logical_tensor_bytes,
            self.transaction_logical_and_accounted_peak_upper_bound_bytes,
            self.old_artifact_store_resident_accounted_bytes,
            self.fresh_artifact_store_resident_accounted_bytes,
            self.persistent_frame_tensor_bytes,
            self.persistent_sample_tensor_bytes,
            self.persistent_target_tensor_bytes,
            self.persistent_prediction_tensor_bytes,
            self.optimizer_history_tensor_bytes,
        )
        if (
            self.provenance != PROMOTION_PROVENANCE
            or not all(_is_sha256(value) for value in digest_fields)
            or self.step_index_after != self.step_index_before + 1
            or self.observation_count < 1
            or not math.isfinite(self.loss)
            or not math.isfinite(self.learning_rate_multiplier)
            or self.learning_rate_multiplier <= 0.0
            or any(value < 0 for value in byte_fields)
            or any(
                value != 0
                for value in (
                    self.fresh_artifact_store_resident_accounted_bytes,
                    self.persistent_frame_tensor_bytes,
                    self.persistent_sample_tensor_bytes,
                    self.persistent_target_tensor_bytes,
                    self.persistent_prediction_tensor_bytes,
                    self.optimizer_history_tensor_bytes,
                )
            )
            or self.generation_digest != _promotion_digest(self)
        ):
            raise ValueError("streaming combined promotion receipt changed")

    def accounting(self) -> Mapping[str, int | float | str]:
        self.assert_self_consistent()
        return MappingProxyType(
            {
                key: value
                for key, value in self.__dict__.items()
                if key != "provenance"
            }
        )


@dataclass
class PaperKineticStreamingCombinedReadyGeneration:
    state: PaperKineticFixedCameraCombinedState = field(repr=False)
    provider: PaperKineticLazyProgramBundleProvider = field(repr=False)
    artifact_store: PaperKineticCompiledCpuArtifactStore = field(repr=False)
    receipt: PaperKineticStreamingCombinedPromotionReceipt
    generation_digest: str
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        self.state.assert_current(self.provider, self.artifact_store)
        self.receipt.assert_self_consistent()
        if (
            self._seal is not _READY_SEAL
            or self.state.generation_digest
            != self.receipt.promoted_state_generation_digest
            or self.provider.generation_digest
            != self.receipt.promoted_provider_generation_digest
            or self.state.cold_recompile_seal_generation_digest
            != self.receipt.lazy_recompile_generation_digest
            or self.generation_digest != _ready_digest(self)
        ):
            raise ValueError("streaming combined ready generation changed")


@torch.no_grad()
def seal_paper_kinetic_streaming_combined_gradient(
    state: PaperKineticFixedCameraCombinedState,
    provider: PaperKineticLazyProgramBundleProvider,
    artifact_store: PaperKineticCompiledCpuArtifactStore,
    *,
    step_generation_id: str,
    observation_count: int,
    loss_f32_cpu: torch.Tensor,
    grad_site_rgba_f32_cpu: torch.Tensor,
    grad_positions0_f64_cpu: torch.Tensor,
    grad_velocities_f64_cpu: torch.Tensor,
    grad_weight_coefficients_f64_cpu: torch.Tensor,
) -> PaperKineticStreamingCombinedGradientAuthorization:
    """Take ownership of already-fenced global CPU bars without cloning them."""

    state.assert_current(provider, artifact_store)
    provisional = PaperKineticStreamingCombinedGradientAuthorization(
        source_state_identity=id(state),
        source_state_generation_digest=state.generation_digest,
        source_provider_identity=id(provider),
        source_provider_generation_digest=provider.generation_digest,
        source_artifact_store_identity=id(artifact_store),
        step_index=state.geometry_update_count,
        step_generation_id=str(step_generation_id),
        observation_count=int(observation_count),
        loss_f32=loss_f32_cpu,
        grad_site_rgba_f32=grad_site_rgba_f32_cpu,
        grad_positions0_f64=grad_positions0_f64_cpu,
        grad_velocities_f64=grad_velocities_f64_cpu,
        grad_weight_coefficients_f64=grad_weight_coefficients_f64_cpu,
        tensor_content_digests=tuple(
            _content_digest(tensor)
            for tensor in (
                grad_site_rgba_f32_cpu,
                loss_f32_cpu,
                grad_positions0_f64_cpu,
                grad_velocities_f64_cpu,
                grad_weight_coefficients_f64_cpu,
            )
        ),
        generation_digest="",
        _seal=_AUTHORIZATION_SEAL,
    )
    provisional.generation_digest = _authorization_digest(provisional)
    provisional.assert_current(state, provider, artifact_store)
    return provisional


@torch.no_grad()
def apply_paper_kinetic_streaming_combined_sgd(
    state: PaperKineticFixedCameraCombinedState,
    provider: PaperKineticLazyProgramBundleProvider,
    artifact_store: PaperKineticCompiledCpuArtifactStore,
    authorization: PaperKineticStreamingCombinedGradientAuthorization,
    *,
    policy: PaperKineticFixedCameraCombinedSGDPolicy,
    fresh_store_policy: PaperKineticCompiledCpuArtifactStorePolicy,
    learning_rate_multiplier: float = 1.0,
) -> PaperKineticStreamingCombinedReadyGeneration:
    """Promote one update and leave the next compiler working set empty."""

    if not isinstance(authorization, PaperKineticStreamingCombinedGradientAuthorization):
        raise TypeError("streaming promotion requires its exact authorization")
    if not isinstance(fresh_store_policy, PaperKineticCompiledCpuArtifactStorePolicy):
        raise TypeError("streaming promotion requires an explicit fresh store policy")
    policy.assert_valid()
    if (
        not math.isfinite(learning_rate_multiplier)
        or learning_rate_multiplier <= 0.0
    ):
        raise ValueError("learning_rate_multiplier must be finite and positive")
    authorization.assert_current(state, provider, artifact_store)
    old_store_report = artifact_store.report()
    source_state_bytes = state.total_persistent_tensor_bytes
    authorization_bytes = authorization.logical_tensor_bytes
    candidates = _combined._build_update_candidates(
        state,
        authorization,  # structural optimizer-authorization protocol
        policy=policy,
        learning_rate_multiplier=learning_rate_multiplier,
    )
    geometry_bytes = _tensor_bytes(
        candidates.positions0_f64,
        candidates.velocities_f64,
        candidates.weight_coefficients_f64,
    )
    transaction_peak = (
        source_state_bytes
        + authorization_bytes
        + candidates.logical_tensor_bytes
        + geometry_bytes
        + old_store_report.current_resident_accounted_bytes
    )
    bounds = (
        (
            source_state_bytes,
            policy.maximum_combined_state_logical_tensor_bytes,
            "source combined state",
        ),
        (
            candidates.logical_tensor_bytes,
            policy.maximum_update_candidate_logical_tensor_bytes,
            "update candidate",
        ),
        (
            geometry_bytes,
            policy.maximum_candidate_world_geometry_clone_logical_tensor_bytes,
            "candidate world geometry",
        ),
        (
            source_state_bytes + authorization_bytes + candidates.logical_tensor_bytes,
            policy.maximum_old_candidate_authorization_logical_tensor_bytes,
            "old/candidate/authorization overlap",
        ),
        (
            transaction_peak,
            policy.maximum_transaction_tracked_logical_and_store_accounted_bytes,
            "streaming promotion transaction",
        ),
    )
    for observed, maximum, name in bounds:
        if observed > maximum:
            raise MemoryError(f"{name} exceeds its explicit policy bound")

    initializer_digest = _digest_parts(
        PROMOTION_PROVENANCE,
        "owned-candidate-world",
        state.geometry_generation_id,
        authorization.generation_digest,
        authorization.step_generation_id,
        _content_digest(candidates.positions0_f64),
        _content_digest(candidates.velocities_f64),
        _content_digest(candidates.weight_coefficients_f64),
    )
    initializer = _combined._OwnedCandidateWorldInitializer(
        sites=AffineKineticPowerSites(
            positions0=candidates.positions0_f64,
            velocities=candidates.velocities_f64,
            weight_coefficients=candidates.weight_coefficients_f64,
        ),
        generation_digest=initializer_digest,
    )
    fresh_provider = prepare_paper_kinetic_lazy_program_bundle_provider(
        dataset_generation_digest=provider.dataset_generation_digest,
        target_provider=provider.target_provider,
        ray_provider=provider.ray_provider,
        frame_times=provider.frame_times,
        height=provider.height,
        width=provider.width,
        maximum_tracks_per_bundle=provider.maximum_tracks_per_bundle,
        maximum_observations_per_bundle=provider.maximum_observations_per_bundle,
        maximum_rows_per_native_block=provider.maximum_rows_per_native_block,
        world_initializer=initializer,
        program_factory=provider.program_factory,
    )
    if not initializer.consumed:
        raise ArithmeticError("candidate world initializer was not consumed")
    fresh_material = _combined._build_fresh_material_state(
        state.material_state,
        fresh_provider,
        authorization,  # structural optimizer-authorization protocol
        candidates,
    )
    fresh_store = PaperKineticCompiledCpuArtifactStore(fresh_store_policy)
    lazy_recompile_digest = _digest_parts(
        LAZY_RECOMPILE_PROVENANCE,
        fresh_provider.generation_digest,
        fresh_provider.world.generation_digest,
        fresh_provider.maximum_tracks_per_bundle,
        fresh_provider.maximum_observations_per_bundle,
        fresh_store_policy.maximum_entries,
        fresh_store_policy.maximum_resident_accounted_bytes,
        authorization.generation_digest,
        authorization.step_generation_id,
        "empty-store;compile-first-consumed-spatial-bundle-on-demand",
    )
    old_state_digest = state.generation_digest
    old_provider_digest = provider.generation_digest
    old_world_digest = provider.world.generation_digest
    old_material_digest = state.material_state.material_generation_id
    old_geometry_digest = state.geometry_generation_id
    new_state: PaperKineticFixedCameraCombinedState | None = None
    try:
        _combined._retire_combined_generation(state, provider, artifact_store)
        authorization.consumed = True
        new_geometry_digest = (
            _combined.paper_kinetic_fixed_camera_provider_geometry_generation_id(
                fresh_provider
            )
        )
        new_state = PaperKineticFixedCameraCombinedState(
            material_state=fresh_material,
            positions0_f64=fresh_provider.world.sites.positions0,
            velocities_f64=fresh_provider.world.sites.velocities,
            weight_coefficients_f64=fresh_provider.world.sites.weight_coefficients,
            provider_generation_digest=fresh_provider.generation_digest,
            world_generation_digest=fresh_provider.world.generation_digest,
            sites_content_digest=fresh_provider.world.sites_content_digest,
            geometry_generation_parent_digest=old_geometry_digest,
            geometry_generation_id=new_geometry_digest,
            last_authorization_generation_digest=authorization.generation_digest,
            last_step_generation_id=authorization.step_generation_id,
            last_update_policy_generation_digest=policy.generation_digest,
            geometry_update_count=state.geometry_update_count + 1,
            cold_recompile_seal_generation_digest=lazy_recompile_digest,
            tensor_signatures=tuple(
                _combined._tensor_signature(tensor)
                for tensor in (
                    fresh_provider.world.sites.positions0,
                    fresh_provider.world.sites.velocities,
                    fresh_provider.world.sites.weight_coefficients,
                )
            ),
            generation_digest="",
            active=True,
            retired=False,
            poisoned=False,
            _provider_identity=id(fresh_provider),
            _artifact_store_identity=id(fresh_store),
            _seal=_combined._STATE_SEAL,
        )
        new_state.generation_digest = _combined._combined_state_digest(new_state)
        new_state.assert_current(fresh_provider, fresh_store)
        fresh_report = fresh_store.report()
        if fresh_report.current_entry_count or fresh_report.current_resident_accounted_bytes:
            raise ArithmeticError("lazy-recompile promotion retained compiled artifacts")
        provisional_receipt = PaperKineticStreamingCombinedPromotionReceipt(
            step_index_before=authorization.step_index,
            step_index_after=new_state.geometry_update_count,
            step_generation_id=authorization.step_generation_id,
            authorization_generation_digest=authorization.generation_digest,
            source_state_generation_digest=old_state_digest,
            promoted_state_generation_digest=new_state.generation_digest,
            source_provider_generation_digest=old_provider_digest,
            promoted_provider_generation_digest=fresh_provider.generation_digest,
            source_world_generation_digest=old_world_digest,
            promoted_world_generation_digest=fresh_provider.world.generation_digest,
            material_generation_id_before=old_material_digest,
            material_generation_id_after=fresh_material.material_generation_id,
            geometry_generation_id_before=old_geometry_digest,
            geometry_generation_id_after=new_geometry_digest,
            lazy_recompile_generation_digest=lazy_recompile_digest,
            observation_count=authorization.observation_count,
            loss=float(authorization.loss_f32.item()),
            learning_rate_multiplier=float(learning_rate_multiplier),
            authorization_logical_tensor_bytes=authorization_bytes,
            source_state_logical_tensor_bytes=source_state_bytes,
            candidate_logical_tensor_bytes=candidates.logical_tensor_bytes,
            candidate_world_geometry_logical_tensor_bytes=geometry_bytes,
            transaction_logical_and_accounted_peak_upper_bound_bytes=(
                transaction_peak
            ),
            old_artifact_store_resident_accounted_bytes=(
                old_store_report.current_resident_accounted_bytes
            ),
            fresh_artifact_store_resident_accounted_bytes=0,
            persistent_frame_tensor_bytes=0,
            persistent_sample_tensor_bytes=0,
            persistent_target_tensor_bytes=0,
            persistent_prediction_tensor_bytes=0,
            optimizer_history_tensor_bytes=0,
            generation_digest="",
        )
        receipt = PaperKineticStreamingCombinedPromotionReceipt(
            **{
                **provisional_receipt.__dict__,
                "generation_digest": _promotion_digest(provisional_receipt),
            }
        )
        receipt.assert_self_consistent()
        ready = PaperKineticStreamingCombinedReadyGeneration(
            state=new_state,
            provider=fresh_provider,
            artifact_store=fresh_store,
            receipt=receipt,
            generation_digest="",
            _seal=_READY_SEAL,
        )
        ready.generation_digest = _ready_digest(ready)
        ready.assert_current()
        return ready
    except BaseException as error:
        cleanup = _combined._invalidate_candidate_generation(
            fresh_material,
            fresh_provider,
            fresh_store,
            combined_state=new_state,
        )
        for note in cleanup:
            error.add_note(note)
        raise


def _authorization_digest(
    value: PaperKineticStreamingCombinedGradientAuthorization,
) -> str:
    return _digest_parts(
        value.provenance,
        value.source_state_identity,
        value.source_state_generation_digest,
        value.source_provider_identity,
        value.source_provider_generation_digest,
        value.source_artifact_store_identity,
        value.step_index,
        value.step_generation_id,
        value.observation_count,
        value.tensor_content_digests,
    )


def _promotion_digest(value: PaperKineticStreamingCombinedPromotionReceipt) -> str:
    return _digest_parts(
        value.provenance,
        *(
            item
            for name, item in value.__dict__.items()
            if name not in {"generation_digest", "provenance"}
        ),
    )


def _ready_digest(value: PaperKineticStreamingCombinedReadyGeneration) -> str:
    return _digest_parts(
        PROMOTION_PROVENANCE,
        "ready",
        value.state.generation_digest,
        value.provider.generation_digest,
        value.receipt.generation_digest,
    )


__all__ = (
    "AUTHORIZATION_PROVENANCE",
    "LAZY_RECOMPILE_PROVENANCE",
    "PROMOTION_PROVENANCE",
    "PaperKineticStreamingCombinedGradientAuthorization",
    "PaperKineticStreamingCombinedPromotionReceipt",
    "PaperKineticStreamingCombinedReadyGeneration",
    "apply_paper_kinetic_streaming_combined_sgd",
    "seal_paper_kinetic_streaming_combined_gradient",
)
