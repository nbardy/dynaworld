"""One exact-coverage fixed-camera full-geometry authorization step.

This is the production coordinator seam for the already-implemented dense
request-level full-geometry reverse.  It intentionally remains a sibling of
``paper_kinetic_fixed_site_material_step``: material-only result contracts stay
material-only, while this module requires one combined material/geometry VJP
per active native block and exposes the resulting world bars only after the
complete replay manifest is sealed.

Cameras are fixed in this first seam.  No ray-gradient key or tensor is
allocated, accepted, or returned.  This module also stops before parameter
mutation, geometry checkpointing, and the mandatory fresh-world recompile;
those absent lifecycle stages are named explicitly in the accounting receipt.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
import paper_kinetic_fixed_site_material_step as _material_step
from kinetic_compiled_cpu_artifact_store import (
    PaperKineticCompiledCpuArtifactStore,
    compile_paper_kinetic_compiled_cpu_artifact,
)
from kinetic_dense_cached_native_material_request import (
    FUSED_DIRECT_V1_FULL_GEOMETRY_REVERSE,
    STAGED_SPARSE_FULL_GEOMETRY_REVERSE,
    PaperKineticDenseOptimizerAuthorization,
    PaperKineticDenseStepGradientAccumulator,
    authorize_paper_kinetic_dense_optimizer_step,
    consume_paper_kinetic_dense_request_delta,
    prepare_paper_kinetic_dense_chunk_target_loader,
    prepare_paper_kinetic_dense_step_gradient_accumulator,
    run_paper_kinetic_dense_cached_native_request,
)
from paper_kinetic_fixed_site_material_state import (
    PaperKineticFixedSiteMaterialState,
)
from paper_kinetic_lazy_program_bundles import (
    PaperKineticLazyProgramBundleProvider,
)
from paper_kinetic_replayable_observations import (
    PaperKineticDenseObservationReplayReceipt,
    PaperKineticDenseObservationReplaySession,
    PaperKineticReplayableDenseObservationSource,
    prepare_paper_kinetic_replayable_dense_observation_source,
)
from paper_training_types import SpacetimeBatch


STEP_PROVENANCE = "paper-kinetic-fixed-camera-full-geometry-step-v1"
STEP_STATUS = "source_integrated/native_runtime_unverified"
GLOBAL_RGB_MEAN_LOSS_NORMALIZATION_ID = (
    _material_step.GLOBAL_RGB_MEAN_LOSS_NORMALIZATION_ID
)

_RESULT_SEAL = object()


@dataclass(frozen=True)
class PaperKineticFixedCameraFullGeometryStepPolicy(
    _material_step.PaperKineticFixedSiteMaterialOnlyStepPolicy
):
    """Material-step bounds plus the exact fixed-camera world-bar bound."""

    maximum_geometry_bar_logical_tensor_bytes: int
    full_geometry_reverse_mode: str = STAGED_SPARSE_FULL_GEOMETRY_REVERSE

    def assert_valid(self) -> None:
        super().assert_valid()
        _material_step._require_positive_int(
            self.maximum_geometry_bar_logical_tensor_bytes,
            name="maximum_geometry_bar_logical_tensor_bytes",
        )
        if (
            self.maximum_geometry_bar_logical_tensor_bytes
            > self.request_memory_policy.maximum_request_geometry_bar_tensor_bytes
        ):
            raise ValueError(
                "full-geometry step bar bound exceeds the request geometry bound"
            )
        if self.full_geometry_reverse_mode not in {
            STAGED_SPARSE_FULL_GEOMETRY_REVERSE,
            FUSED_DIRECT_V1_FULL_GEOMETRY_REVERSE,
        }:
            raise ValueError(
                "full_geometry_reverse_mode must be staged_sparse or fused_direct_v1"
            )
        if self.full_geometry_reverse_mode == FUSED_DIRECT_V1_FULL_GEOMETRY_REVERSE:
            fused_caps = (
                self.request_memory_policy.maximum_fused_prepared_owned_logical_tensor_bytes,
                self.request_memory_policy.maximum_fused_output_scratch_logical_tensor_bytes,
                self.request_memory_policy.maximum_fused_geometry_bridge_visible_peak_logical_tensor_bytes,
            )
            if any(cap < 1 for cap in fused_caps):
                raise ValueError(
                    "fused_direct_v1 requires explicit positive prepared, output, and bridge caps"
                )

    @property
    def generation_digest(self) -> str:
        self.assert_valid()
        return _material_step._digest_parts(
            STEP_PROVENANCE,
            "step-policy",
            super().generation_digest,
            self.maximum_geometry_bar_logical_tensor_bytes,
            self.full_geometry_reverse_mode,
            "fixed-camera/no-ray-bars",
        )


@dataclass(frozen=True)
class PaperKineticFixedCameraFullGeometryGenerationPolicy(
    _material_step.PaperKineticFixedSiteMaterialOnlyGenerationPolicy
):
    """Logical step identity including the immutable geometry generation."""

    geometry_generation_id: str

    def assert_valid(self) -> None:
        super().assert_valid()
        if (
            not isinstance(self.geometry_generation_id, str)
            or len(self.geometry_generation_id) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.geometry_generation_id
            )
        ):
            raise ValueError("geometry_generation_id must be a SHA-256 digest")

    @property
    def step_generation_id(self) -> str:
        self.assert_valid()
        return _material_step._digest_parts(
            STEP_PROVENANCE,
            "logical-step",
            self.step_index,
            self.material_generation_id,
            self.geometry_generation_id,
            self.background_generation_id,
            self.target_generation_id,
            "fixed-camera/no-ray-bars",
        )

    @property
    def generation_digest(self) -> str:
        self.assert_valid()
        return _material_step._digest_parts(
            STEP_PROVENANCE,
            "generation-policy",
            self.step_generation_id,
            self.geometry_generation_id,
        )


def paper_kinetic_fixed_camera_provider_geometry_generation_id(
    provider: PaperKineticLazyProgramBundleProvider,
) -> str:
    """Derive the only accepted geometry generation from the live world."""

    if not isinstance(provider, PaperKineticLazyProgramBundleProvider):
        raise TypeError("geometry generation requires a kinetic provider")
    provider.assert_current()
    return paper_kinetic_fixed_camera_geometry_generation_id(
        world_generation_digest=provider.world.generation_digest,
        world_sites_content_digest=provider.world.sites_content_digest,
        world_site_count=provider.world.site_count,
    )


# The lifetime root is mode-agnostic: it owns the immutable provider/store and
# quarantines source/session/accumulator objects after partial device progress.
# Keep one implementation until the geometry updater introduces a fresh-world
# generation state with mandatory recompilation.
PaperKineticFixedCameraFullGeometryStepState = (
    _material_step.PaperKineticFixedSiteMaterialStepState
)


class PaperKineticFixedCameraFullGeometryStepPartialFailure(RuntimeError):
    """Partial/device progress failed and is durably rooted on ``state``."""

    def __init__(
        self,
        state: PaperKineticFixedCameraFullGeometryStepState,
        cause: BaseException,
    ) -> None:
        super().__init__(
            "fixed-camera full-geometry step failed after partial/device "
            "progress; the state is poisoned and process restart is required: "
            f"{type(cause).__qualname__}: {cause}"
        )
        self.state = state


@dataclass(frozen=True)
class PaperKineticFixedCameraFullGeometryStepResult:
    """Sealed material and fixed-camera world bars for an external updater."""

    authorization: PaperKineticDenseOptimizerAuthorization = field(repr=False)
    accumulator: PaperKineticDenseStepGradientAccumulator = field(repr=False)
    replay_receipt: PaperKineticDenseObservationReplayReceipt = field(repr=False)
    loss_rgb_mean: float
    accounting: Mapping[str, Any]
    generation_digest: str
    _authorization_identity: int = field(repr=False)
    _accumulator_identity: int = field(repr=False)
    _replay_receipt_identity: int = field(repr=False)
    provenance: str = STEP_PROVENANCE
    runtime_status: str = STEP_STATUS
    parameter_mutation_count: int = 0
    retained_authorization_capability_object_count: int = 3
    retained_source_count: int = 0
    retained_session_count: int = 0
    retained_request_count: int = 0
    retained_artifact_count: int = 0
    retained_target_count: int = 0
    retained_native_lane_count: int = 0
    native_runtime_verified: bool = False
    allocator_peak_measured: bool = False
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        if not isinstance(self.authorization, PaperKineticDenseOptimizerAuthorization):
            raise TypeError("full-geometry result lost its optimizer authorization")
        if not isinstance(self.accumulator, PaperKineticDenseStepGradientAccumulator):
            raise TypeError("full-geometry result lost its gradient accumulator")
        if not isinstance(
            self.replay_receipt,
            PaperKineticDenseObservationReplayReceipt,
        ):
            raise TypeError("full-geometry result lost its replay receipt")
        self.authorization.assert_current(self.accumulator, self.replay_receipt)
        geometry_bars = (
            self.authorization.grad_positions0_f64,
            self.authorization.grad_velocities_f64,
            self.authorization.grad_weight_coefficients_f64,
        )
        if (
            self.authorization.optimize_camera_rays
            or self.accumulator.optimize_camera_rays
            or self.authorization.ray_bar_keys
            or self.accumulator.ray_bar_keys
            or self.authorization.grad_track_ray_coefficients_f64 is not None
            or self.accumulator.grad_track_ray_coefficients_f64 is not None
            or self.accounting.get("camera_ray_gradients_enabled") is not False
            or self.accounting.get("fixed_camera_avoids_global_ray_bar") is not True
            or int(self.accounting.get("step_ray_bar_key_logical_bytes", -1)) != 0
            or int(
                self.accounting.get(
                    "maximum_request_delta_ray_bar_key_logical_bytes",
                    -1,
                )
            )
            != 0
            or int(self.accounting.get("peak_ray_payload_logical_tensor_bytes", -1))
            != 0
        ):
            raise ValueError("fixed-camera full-geometry result retained ray bars")
        _assert_result_reverse_accounting(self.accounting)
        provider_generation_digest = self.accounting.get(
            "provider_generation_digest"
        )
        provider_identity = self.accounting.get("provider_identity")
        world_generation_digest = self.accounting.get("world_generation_digest")
        world_sites_content_digest = self.accounting.get(
            "world_sites_content_digest"
        )
        world_site_count = self.accounting.get("world_site_count")
        site_table_identity = self.accounting.get("site_table_identity")
        geometry_generation_id = self.accounting.get("geometry_generation_id")
        geometry_bar_bytes = _geometry_bar_tensor_bytes(self.accumulator)
        if (
            self._seal is not _RESULT_SEAL
            or self.provenance != STEP_PROVENANCE
            or self.runtime_status != STEP_STATUS
            or id(self.authorization) != self._authorization_identity
            or id(self.accumulator) != self._accumulator_identity
            or id(self.replay_receipt) != self._replay_receipt_identity
            or not self.authorization.full_geometry
            or not self.accumulator.full_geometry
            or any(value is None for value in geometry_bars)
            or not math.isfinite(self.loss_rgb_mean)
            or self.loss_rgb_mean < 0.0
            or self.parameter_mutation_count != 0
            or self.retained_authorization_capability_object_count != 3
            or any(
                value != 0
                for value in (
                    self.retained_source_count,
                    self.retained_session_count,
                    self.retained_request_count,
                    self.retained_artifact_count,
                    self.retained_target_count,
                    self.retained_native_lane_count,
                )
            )
            or self.native_runtime_verified
            or self.allocator_peak_measured
            or not isinstance(self.accounting, MappingProxyType)
            or not isinstance(provider_generation_digest, str)
            or len(provider_generation_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in provider_generation_digest
            )
            or isinstance(provider_identity, bool)
            or not isinstance(provider_identity, int)
            or provider_identity < 1
            or not isinstance(world_generation_digest, str)
            or len(world_generation_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in world_generation_digest
            )
            or not isinstance(world_sites_content_digest, str)
            or len(world_sites_content_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in world_sites_content_digest
            )
            or isinstance(world_site_count, bool)
            or not isinstance(world_site_count, int)
            or world_site_count < 1
            or isinstance(site_table_identity, bool)
            or not isinstance(site_table_identity, int)
            or site_table_identity < 1
            or not isinstance(geometry_generation_id, str)
            or geometry_generation_id
            != paper_kinetic_fixed_camera_geometry_generation_id(
                world_generation_digest=world_generation_digest,
                world_sites_content_digest=world_sites_content_digest,
                world_site_count=world_site_count,
            )
            or world_generation_digest
            != self.accumulator.world_generation_digest
            or world_sites_content_digest
            != self.accumulator.world_sites_content_digest
            or site_table_identity != self.accumulator.site_table_identity
            or world_site_count != int(self.accumulator.grad_site_rgba_f32.shape[0])
            or self.accounting.get("full_geometry") is not True
            or self.accounting.get("full_geometry_vjp_integrated") is not True
            or self.accounting.get(
                "fixed_camera_full_geometry_step_coordinator_integrated"
            )
            is not True
            or self.accounting.get("production_trainer_integrated") is not False
            or int(self.accounting.get("native_material_vjp_launch_count", -1))
            != 0
            or self.accounting.get(
                "geometry_completion_receipt_retains_native_tensors"
            )
            is not False
            or int(self.accounting.get("geometry_row_vjp_call_count", 0)) < 1
            or int(
                self.accounting.get(
                    "maximum_geometry_bridge_visible_tensor_bytes",
                    0,
                )
            )
            < 1
            or int(self.accounting.get("maximum_request_geometry_bar_tensor_bytes", 0))
            < 1
            or self.accounting.get("geometry_bar_tensor_bytes")
            != geometry_bar_bytes
            or self.accounting.get("geometry_bar_memory_receipt_kind")
            != "logical_tensor_bytes"
            or self.accounting.get("geometry_bar_allocator_peak_measured")
            is not False
            or self.accounting.get("step_accumulator_logical_tensor_bytes")
            != self.accumulator.logical_tensor_bytes
            or self.accounting.get("loss_normalization_id")
            != GLOBAL_RGB_MEAN_LOSS_NORMALIZATION_ID
            or self.accounting.get("global_rgb_mean_application_count") != 1
            or self.accounting.get("accumulator_initialization_fence_call_count")
            != 1
            or self.accounting.get("parameter_mutation_count") != 0
            or self.accounting.get("optimizer_step_executed") is not False
            or self.accounting.get(
                "optimizer_authorization_requires_full_manifest_seal"
            )
            is not True
            or self.accounting.get("geometry_update_executed") is not False
            or self.accounting.get("fresh_world_recompile_executed") is not False
            or self.accounting.get("stale_structure_reuse_prevention_integrated")
            is not False
            or any(
                self.accounting.get(key) != 0
                for key in (
                    "persistent_frame_tensor_bytes",
                    "persistent_sample_tensor_bytes",
                    "persistent_target_tensor_bytes",
                    "persistent_prediction_tensor_bytes",
                    "reachable_autograd_tensor_count",
                )
            )
            or self.accounting.get("step_accumulator_retains_frame_axis") is not False
            or self.accounting.get("autograd_graph_retained") is not False
            or self.accounting.get("allocator_peak_measured") is not False
            or self.generation_digest != _result_digest(self)
        ):
            raise ValueError("fixed-camera full-geometry step result changed")


def prepare_paper_kinetic_fixed_camera_full_geometry_step_state(
    provider: PaperKineticLazyProgramBundleProvider,
    artifact_store: PaperKineticCompiledCpuArtifactStore,
    *,
    device: torch.device | str,
    resume_material_state: PaperKineticFixedSiteMaterialState | None = None,
) -> PaperKineticFixedCameraFullGeometryStepState:
    """Bind the existing fail-stop lifetime root to the fixed-camera seam."""

    return _material_step.prepare_paper_kinetic_fixed_site_material_step_state(
        provider,
        artifact_store,
        device=device,
        resume_material_state=resume_material_state,
    )


@torch.no_grad()
def run_paper_kinetic_fixed_camera_full_geometry_step(
    state: PaperKineticFixedCameraFullGeometryStepState,
    provider: PaperKineticLazyProgramBundleProvider,
    batch: SpacetimeBatch,
    *,
    policy: PaperKineticFixedCameraFullGeometryStepPolicy,
    generation_policy: PaperKineticFixedCameraFullGeometryGenerationPolicy,
    global_site_rgba_f32: torch.Tensor,
    background_rgb_f32: torch.Tensor,
    native_ops: Any,
    backend_provenance: str,
    device_completion_fence: Callable[[], None],
    device_completion_fence_provenance: str,
) -> PaperKineticFixedCameraFullGeometryStepResult:
    """Authorize one exact fixed-camera material-and-world reverse."""

    if not isinstance(
        state,
        _material_step.PaperKineticFixedSiteMaterialStepState,
    ):
        raise TypeError("full-geometry step requires its caller-owned state")
    acquired_lock = state._execution_lock.acquire(blocking=False)
    if not acquired_lock:
        raise RuntimeError("fixed-camera full-geometry step state is already active")
    source: PaperKineticReplayableDenseObservationSource | None = None
    session: PaperKineticDenseObservationReplaySession | None = None
    accumulator: PaperKineticDenseStepGradientAccumulator | None = None
    active_request: Any = None
    active_artifact: Any = None
    active_request_result: Any = None
    execution_started = False
    unsafe_device_fence_failure = False
    try:
        state.assert_current(provider)
        if state.poisoned:
            raise RuntimeError(
                "fixed-camera full-geometry step state requires process restart"
            )
        _validate_step_policy(
            state,
            provider,
            batch,
            policy=policy,
            generation_policy=generation_policy,
            global_site_rgba_f32=global_site_rgba_f32,
            background_rgb_f32=background_rgb_f32,
            backend_provenance=backend_provenance,
            device_completion_fence=device_completion_fence,
            device_completion_fence_provenance=(
                device_completion_fence_provenance
            ),
        )
        state.active_step_generation_id = generation_policy.step_generation_id
        material_signature = _material_step._tensor_signature(global_site_rgba_f32)
        background_signature = _material_step._tensor_signature(background_rgb_f32)
        world_geometry_signatures = tuple(
            _material_step._tensor_signature(tensor)
            for tensor in _world_geometry_tensors(provider)
        )
        state.artifact_store.report()
        source = prepare_paper_kinetic_replayable_dense_observation_source(
            provider,
            batch,
            memory_policy=policy.observation_memory_policy,
        )
        session = source.open_session()
        accumulator = prepare_paper_kinetic_dense_step_gradient_accumulator(
            source,
            session,
            step_generation_id=generation_policy.step_generation_id,
            loss_normalization_id=GLOBAL_RGB_MEAN_LOSS_NORMALIZATION_ID,
            material_generation_id=generation_policy.material_generation_id,
            background_generation_id=generation_policy.background_generation_id,
            global_site_rgba_f32=global_site_rgba_f32,
            background_rgb_f32=background_rgb_f32,
            device=state.device,
            full_geometry=True,
            optimize_camera_rays=False,
        )
        execution_started = True
        try:
            returned = device_completion_fence()
        except BaseException:
            unsafe_device_fence_failure = True
            raise
        if returned is not None:
            raise TypeError("accumulator initialization fence must return None")
        if accumulator.logical_tensor_bytes != _step_accumulator_tensor_bytes(provider):
            raise ArithmeticError(
                "fixed-camera full-geometry accumulator changed its exact layout"
            )
        if (
            accumulator.logical_tensor_bytes
            > policy.maximum_step_accumulator_logical_tensor_bytes
        ):
            raise ArithmeticError("full-geometry accumulator exceeded its preflight")

        expected_requests_per_view = (
            source.image_pixel_count + policy.maximum_tracks_per_request - 1
        ) // policy.maximum_tracks_per_request
        expected_request_count = (
            source.selected_view_count * expected_requests_per_view
        )
        counters = _zero_counters()
        counters["accumulator_initialization_fence_call_count"] = 1
        common_peaks = {key: 0 for key in _COMMON_PEAK_KEYS}
        geometry_peaks = {key: 0 for key in _GEOMETRY_PEAK_KEYS}
        structural_accounting = _material_step._StepStructuralAccounting()
        for view_index in source.canonical_view_indices:
            for track_start in range(
                0,
                source.image_pixel_count,
                policy.maximum_tracks_per_request,
            ):
                track_end = min(
                    track_start + policy.maximum_tracks_per_request,
                    source.image_pixel_count,
                )
                request = source.prepare_track_request(
                    view_index=view_index,
                    track_ids=tuple(range(track_start, track_end)),
                )
                active_request = request
                acquisition = state.artifact_store.acquire(
                    provider,
                    view_index=view_index,
                    track_ids=request.track_ids,
                    maximum_artifact_accounted_bytes=(
                        policy.maximum_artifact_accounted_bytes
                    ),
                    compile_artifact=lambda key: (
                        compile_paper_kinetic_compiled_cpu_artifact(provider, key)
                    ),
                )
                artifact = acquisition.artifact
                active_artifact = artifact

                built_in_target_loader = (
                    prepare_paper_kinetic_dense_chunk_target_loader(
                        source,
                        request,
                        device=state.device,
                        target_generation_id=(
                            f"{generation_policy.target_generation_id}:"
                            f"{request.generation_digest}"
                        ),
                        maximum_decoded_frame_scratch_tensor_bytes=(
                            policy.request_memory_policy.maximum_decoded_frame_scratch_tensor_bytes
                        ),
                        maximum_chunk_target_tensor_bytes=(
                            policy.request_memory_policy.maximum_chunk_target_tensor_bytes
                        ),
                        maximum_target_decode_bridge_peak_logical_tensor_bytes=(
                            policy.request_memory_policy.maximum_target_decode_bridge_peak_logical_tensor_bytes
                        ),
                    )
                )

                request_result = run_paper_kinetic_dense_cached_native_request(
                    source,
                    session,
                    request,
                    artifact,
                    accumulator,
                    step_generation_id=generation_policy.step_generation_id,
                    loss_normalization_id=GLOBAL_RGB_MEAN_LOSS_NORMALIZATION_ID,
                    material_generation_id=generation_policy.material_generation_id,
                    background_generation_id=(
                        generation_policy.background_generation_id
                    ),
                    global_site_rgba_f32=global_site_rgba_f32,
                    background_rgb_f32=background_rgb_f32,
                    native_ops=native_ops,
                    backend_provenance=backend_provenance,
                    maximum_samples_per_launch=policy.maximum_samples_per_launch,
                    memory_policy=policy.request_memory_policy,
                    load_chunk_targets=built_in_target_loader,
                    device_completion_fence=device_completion_fence,
                    device_completion_fence_provenance=(
                        device_completion_fence_provenance
                    ),
                    full_geometry_reverse_mode=policy.full_geometry_reverse_mode,
                    cone_tolerance=policy.cone_tolerance,
                )
                active_request_result = request_result
                request_result.assert_current(
                    source,
                    request,
                    artifact,
                    session,
                    accumulator,
                )
                commit_receipt = consume_paper_kinetic_dense_request_delta(
                    accumulator,
                    source,
                    session,
                    request,
                    artifact,
                    request_result.delta,
                    device_completion_fence=device_completion_fence,
                    device_completion_fence_provenance=(
                        device_completion_fence_provenance
                    ),
                )
                _accumulate_request_accounting(
                    counters,
                    common_peaks,
                    geometry_peaks,
                    request_result.accounting,
                    telemetry=request_result.telemetry,
                    acquisition=acquisition,
                    structural_accounting=structural_accounting,
                    expected_full_geometry_reverse_mode=(
                        policy.full_geometry_reverse_mode
                    ),
                    commit_fence_call_count=(
                        commit_receipt.device_completion_fence_call_count
                    ),
                    expected_global_loss_element_count=(
                        source.observation_count * 3
                    ),
                )
                active_request = None
                active_artifact = None
                active_request_result = None
                del commit_receipt, request_result, artifact, acquisition, request

        if (
            counters["request_count"] != expected_request_count
            or counters["streamed_observation_count"] != source.observation_count
        ):
            raise ArithmeticError(
                "full-geometry step request partition changed coverage"
            )
        replay_receipt = session.seal()
        if (
            replay_receipt.request_count != expected_request_count
            or replay_receipt.observation_count != source.observation_count
        ):
            raise ArithmeticError(
                "full-geometry step replay seal changed exact coverage"
            )
        authorization = authorize_paper_kinetic_dense_optimizer_step(
            accumulator,
            source,
            session,
            replay_receipt,
        )
        authorization.assert_current(accumulator, replay_receipt)
        if (
            not authorization.full_geometry
            or authorization.optimize_camera_rays
            or authorization.ray_bar_keys
            or authorization.grad_track_ray_coefficients_f64 is not None
        ):
            raise ValueError(
                "fixed-camera full-geometry authorization retained ray bars"
            )
        if _material_step._tensor_signature(global_site_rgba_f32) != material_signature:
            raise ValueError("full-geometry step mutated the material snapshot")
        if _material_step._tensor_signature(background_rgb_f32) != background_signature:
            raise ValueError("full-geometry step mutated the background snapshot")
        if tuple(
            _material_step._tensor_signature(tensor)
            for tensor in _world_geometry_tensors(provider)
        ) != world_geometry_signatures:
            raise ValueError("full-geometry step mutated its immutable world snapshot")
        loss_rgb_mean = float(authorization.loss_f32.detach().cpu().item())
        if not math.isfinite(loss_rgb_mean) or loss_rgb_mean < 0.0:
            raise FloatingPointError(
                "full-geometry step produced a non-finite RGB mean"
            )
        accounting = MappingProxyType(
            _step_accounting(
                source,
                accumulator,
                replay_receipt,
                policy=policy,
                generation_policy=generation_policy,
                counters=counters,
                common_peaks=common_peaks,
                geometry_peaks=geometry_peaks,
                structural_accounting=structural_accounting,
                store_after=state.artifact_store.report(),
                loss_rgb_mean=loss_rgb_mean,
                backend_provenance=backend_provenance,
                device_completion_fence_provenance=(
                    device_completion_fence_provenance
                ),
            )
        )
        provisional = PaperKineticFixedCameraFullGeometryStepResult(
            authorization=authorization,
            accumulator=accumulator,
            replay_receipt=replay_receipt,
            loss_rgb_mean=loss_rgb_mean,
            accounting=accounting,
            generation_digest="",
            _authorization_identity=id(authorization),
            _accumulator_identity=id(accumulator),
            _replay_receipt_identity=id(replay_receipt),
            _seal=_RESULT_SEAL,
        )
        result = PaperKineticFixedCameraFullGeometryStepResult(
            **{
                **provisional.__dict__,
                "generation_digest": _result_digest(provisional),
            }
        )
        result.assert_current()
        state.authorized_step_count += 1
        state.last_step_generation_id = generation_policy.step_generation_id
        state.last_authorized_material_generation_id = (
            generation_policy.material_generation_id
        )
        state.active_step_generation_id = ""
        state.assert_current(provider)
        return result
    except BaseException as error:
        partial_progress = bool(
            execution_started
            or session is not None
            and (session.emitted_observation_count or session.sealed)
            or accumulator is not None
            and (
                accumulator.consumed_request_count
                or accumulator.pending_delta_generation_digest
                or accumulator.optimizer_authorized
                or accumulator.poisoned
            )
        )
        if (
            partial_progress
            and source is not None
            and session is not None
            and accumulator is not None
        ):
            _material_step._retain_failed_step(
                state,
                source,
                session,
                accumulator,
                error,
                lifetime_roots=(
                    ("request", active_request),
                    ("artifact", active_artifact),
                    ("request_result_and_delta", active_request_result),
                ),
                attempt_fail_stop=not unsafe_device_fence_failure,
            )
            raise PaperKineticFixedCameraFullGeometryStepPartialFailure(
                state,
                error,
            ) from error
        state.active_step_generation_id = ""
        raise
    finally:
        state._execution_lock.release()


def _validate_step_policy(
    state: PaperKineticFixedCameraFullGeometryStepState,
    provider: PaperKineticLazyProgramBundleProvider,
    batch: SpacetimeBatch,
    *,
    policy: PaperKineticFixedCameraFullGeometryStepPolicy,
    generation_policy: PaperKineticFixedCameraFullGeometryGenerationPolicy,
    global_site_rgba_f32: torch.Tensor,
    background_rgb_f32: torch.Tensor,
    backend_provenance: str,
    device_completion_fence: Callable[[], None],
    device_completion_fence_provenance: str,
) -> None:
    if not isinstance(policy, PaperKineticFixedCameraFullGeometryStepPolicy):
        raise TypeError("full-geometry step requires its explicit step policy")
    if not isinstance(
        generation_policy,
        PaperKineticFixedCameraFullGeometryGenerationPolicy,
    ):
        raise TypeError(
            "full-geometry step requires its explicit generation policy"
        )
    _material_step._validate_step_policy(
        state,
        provider,
        batch,
        policy=policy,
        generation_policy=generation_policy,
        global_site_rgba_f32=global_site_rgba_f32,
        background_rgb_f32=background_rgb_f32,
        backend_provenance=backend_provenance,
        device_completion_fence=device_completion_fence,
        device_completion_fence_provenance=(
            device_completion_fence_provenance
        ),
    )
    if generation_policy.geometry_generation_id != (
        paper_kinetic_fixed_camera_provider_geometry_generation_id(provider)
    ):
        raise ValueError(
            "geometry_generation_id is foreign to the live provider world"
        )
    geometry_bar_bytes = _geometry_bar_tensor_bytes_from_provider(provider)
    if geometry_bar_bytes > policy.maximum_geometry_bar_logical_tensor_bytes:
        raise MemoryError(
            "fixed-camera geometry bars exceed their explicit whole-step bound"
        )
    if (
        geometry_bar_bytes
        > policy.request_memory_policy.maximum_request_geometry_bar_tensor_bytes
    ):
        raise MemoryError(
            "fixed-camera geometry bars exceed the request-local geometry bound"
        )
    if (
        _step_accumulator_tensor_bytes(provider)
        > policy.maximum_step_accumulator_logical_tensor_bytes
    ):
        raise MemoryError(
            "fixed-camera full-geometry accumulator exceeds its explicit bound"
        )


def _accumulate_request_accounting(
    counters: dict[str, int],
    common_peaks: dict[str, int],
    geometry_peaks: dict[str, int],
    accounting: Mapping[str, Any],
    *,
    telemetry: Any,
    acquisition: Any,
    structural_accounting: Any,
    commit_fence_call_count: int,
    expected_global_loss_element_count: int,
    expected_full_geometry_reverse_mode: str,
) -> None:
    telemetry.assert_current()
    active_block_count = telemetry.active_native_block_count
    staged = (
        expected_full_geometry_reverse_mode
        == STAGED_SPARSE_FULL_GEOMETRY_REVERSE
    )
    fused = (
        expected_full_geometry_reverse_mode
        == FUSED_DIRECT_V1_FULL_GEOMETRY_REVERSE
    )
    staged_vjp_count = int(
        accounting.get("native_full_geometry_vjp_launch_count", -1)
    )
    fused_vjp_count = int(
        accounting.get("native_fused_full_geometry_vjp_launch_count", -1)
    )
    fused_transaction_count = int(
        accounting.get("native_fused_full_geometry_transaction_count", -1)
    )
    fused_completion_count = int(
        accounting.get(
            "native_fused_full_geometry_completion_fence_count",
            -1,
        )
    )
    reduction_fence_count = int(
        accounting.get("geometry_reduction_fence_call_count", -1)
    )
    completion_count = int(accounting.get("geometry_completion_receipt_count", -1))
    staged_contract = staged and (
        telemetry.reverse_mode == "full_geometry"
        and accounting.get("geometry_reduction_mode")
        == "certified_sparse_compact"
        and accounting.get("exactly_one_full_geometry_vjp_per_active_block")
        is True
        and accounting.get(
            "exactly_one_fused_full_geometry_vjp_per_active_block"
        )
        is False
        and staged_vjp_count == active_block_count
        and fused_vjp_count == 0
        and telemetry.native_full_geometry_vjp_launch_count == active_block_count
        and telemetry.native_fused_full_geometry_vjp_launch_count == 0
        and telemetry.native_full_geometry_fenced_reduction_count
        == active_block_count
        and fused_transaction_count == 0
        and fused_completion_count == 0
        and reduction_fence_count == active_block_count
        and completion_count == active_block_count
        and int(accounting.get("maximum_native_length_bar_tensor_bytes", 0)) > 0
        and int(
            accounting.get(
                "maximum_simultaneous_geometry_jw_length_bar_tensors",
                0,
            )
        )
        == 1
        and accounting.get("fused_active_manifest_coverage_certified") is False
        and int(accounting.get("fused_validation_status_tensor_bytes", -1)) == 0
        and int(accounting.get("fused_compact_material_scatter_elements", -1))
        == 0
        and int(
            accounting.get(
                "staged_sparse_geometry_bridge_visible_peak_logical_tensor_bytes_upper_bound",
                0,
            )
        )
        >= int(accounting.get("maximum_geometry_bridge_visible_tensor_bytes", 0))
        and accounting.get(
            "staged_sparse_geometry_bridge_included_in_main_active_peak"
        )
        is True
    )
    fused_contract = fused and (
        telemetry.reverse_mode == "fused_full_geometry"
        and accounting.get("geometry_reduction_mode") == "fused_direct_v1"
        and accounting.get("exactly_one_full_geometry_vjp_per_active_block")
        is False
        and accounting.get(
            "exactly_one_fused_full_geometry_vjp_per_active_block"
        )
        is True
        and staged_vjp_count == 0
        and fused_vjp_count == active_block_count
        and telemetry.native_full_geometry_vjp_launch_count == 0
        and telemetry.native_fused_full_geometry_vjp_launch_count
        == active_block_count
        and telemetry.native_full_geometry_fenced_reduction_count == 0
        and fused_transaction_count == 1
        and fused_completion_count == 1
        and reduction_fence_count == 0
        and completion_count == 0
        and int(accounting.get("maximum_native_length_bar_tensor_bytes", -1)) == 0
        and int(
            accounting.get(
                "maximum_simultaneous_geometry_jw_length_bar_tensors",
                -1,
            )
        )
        == 0
        and accounting.get("fused_length_cotangent_allocated") is False
        and int(accounting.get("fused_transaction_fence_call_count", -1)) == 1
        and int(
            accounting.get("fused_post_accept_commit_fence_call_count", -1)
        )
        == 1
        and accounting.get("fused_active_manifest_coverage_certified") is True
        and accounting.get("fused_optimizer_commit_performed") is False
        and int(accounting.get("fused_validation_status_tensor_bytes", -1)) == 4
        and int(accounting.get("fused_compact_material_scatter_elements", 0))
        > 0
        and int(accounting.get("fused_prepared_owned_logical_tensor_bytes", 0))
        > 0
        and int(accounting.get("fused_output_scratch_logical_tensor_bytes", 0))
        > 0
        and int(accounting.get("fused_geometry_bridge_visible_tensor_bytes", 0))
        > 0
    )
    if (
        accounting.get("full_geometry_vjp_integrated") is not True
        or accounting.get("full_geometry_reverse_mode")
        != expected_full_geometry_reverse_mode
        or not (staged_contract or fused_contract)
        or accounting.get("fused_length_cotangent_allocated") is not False
        or accounting.get("fused_optimizer_commit_performed") is not False
        or int(accounting.get("native_material_word_vjp_launch_count", -1)) != 0
        or telemetry.native_material_word_vjp_launch_count != 0
        or accounting.get("geometry_completion_receipt_retains_native_tensors")
        is not False
        or int(accounting.get("geometry_row_vjp_call_count", 0)) < 1
        or int(
            accounting.get("maximum_geometry_bridge_visible_tensor_bytes", 0)
        )
        < 1
        or int(accounting.get("request_geometry_bar_tensor_bytes", 0)) < 1
        or accounting.get("camera_ray_gradients_enabled") is not False
        or accounting.get("fixed_camera_avoids_global_ray_bar") is not True
        or int(accounting.get("step_ray_bar_key_logical_bytes", -1)) != 0
        or int(accounting.get("request_delta_ray_bar_key_logical_bytes", -1)) != 0
        or accounting.get("caller_bars_mutated_by_request") is not False
        or accounting.get(
            "optimizer_authorization_requires_full_manifest_seal"
        )
        is not True
        or accounting.get(
            "sample_materialization_source_visible_logical_tensors_accounted"
        )
        is not True
        or accounting.get(
            "target_source_decode_budget_enforced_before_allocation"
        )
        is not True
    ):
        raise ValueError(
            "fixed-camera coordinator received an invalid full-geometry request"
        )
    if (
        telemetry.loss_normalization_id
        != GLOBAL_RGB_MEAN_LOSS_NORMALIZATION_ID
        or telemetry.global_loss_element_count != expected_global_loss_element_count
        or telemetry.loss_scale != 1.0 / float(expected_global_loss_element_count)
    ):
        raise ValueError("full-geometry request changed the one global RGB mean")
    if accounting.get("expected_observation_count", 0) < 1:
        raise ArithmeticError("full-geometry request reported empty coverage")

    counters["request_count"] += 1
    if acquisition.cache_status == "cold_compiled":
        counters["cold_artifact_count"] += 1
    elif acquisition.cache_status == "warm_hit":
        counters["warm_artifact_count"] += 1
    else:
        raise ValueError("full-geometry step received unknown artifact cache status")
    counters["artifact_store_eviction_count"] += int(
        acquisition.evicted_entry_count
    )
    counters["artifact_store_evicted_accounted_bytes"] += int(
        acquisition.evicted_accounted_bytes
    )
    counters["artifact_store_cold_compiled_track_count"] += int(
        acquisition.cold_compiled_track_count
    )
    counters["artifact_store_avoided_compile_track_count"] += int(
        acquisition.avoided_compile_track_count
    )
    for destination, source_key in _COMMON_SUM_KEYS:
        counters[destination] += int(accounting[source_key])
    counters["native_full_geometry_vjp_launch_count"] += staged_vjp_count
    counters["native_fused_full_geometry_vjp_launch_count"] += fused_vjp_count
    counters["native_fused_full_geometry_transaction_count"] += (
        fused_transaction_count
    )
    counters["native_fused_full_geometry_completion_fence_count"] += (
        fused_completion_count
    )
    counters["native_full_geometry_fenced_reduction_count"] += int(
        telemetry.native_full_geometry_fenced_reduction_count
    )
    counters["geometry_reduction_fence_call_count"] += reduction_fence_count
    counters["geometry_completion_receipt_count"] += completion_count
    counters["geometry_row_vjp_call_count"] += int(
        accounting["geometry_row_vjp_call_count"]
    )
    counters["fused_transaction_fence_call_count"] += int(
        accounting["fused_transaction_fence_call_count"]
    )
    counters["fused_post_accept_commit_fence_call_count"] += int(
        accounting["fused_post_accept_commit_fence_call_count"]
    )
    counters["fused_active_manifest_certified_request_count"] += int(
        accounting["fused_active_manifest_coverage_certified"]
    )
    counters["fused_compact_material_scatter_elements"] += int(
        accounting["fused_compact_material_scatter_elements"]
    )
    counters["geometry_compact_to_global_scatter_elements"] += int(
        accounting["geometry_compact_to_global_scatter_elements"]
    )
    counters["request_commit_fence_call_count"] += commit_fence_call_count
    structural_accounting.add(accounting)
    for key in common_peaks:
        common_peaks[key] = max(common_peaks[key], int(accounting[key]))
    for key in geometry_peaks:
        geometry_peaks[key] = max(geometry_peaks[key], int(accounting[key]))


def _step_accounting(
    source: PaperKineticReplayableDenseObservationSource,
    accumulator: PaperKineticDenseStepGradientAccumulator,
    replay_receipt: PaperKineticDenseObservationReplayReceipt,
    *,
    policy: PaperKineticFixedCameraFullGeometryStepPolicy,
    generation_policy: PaperKineticFixedCameraFullGeometryGenerationPolicy,
    counters: Mapping[str, int],
    common_peaks: Mapping[str, int],
    geometry_peaks: Mapping[str, int],
    structural_accounting: Any,
    store_after: Any,
    loss_rgb_mean: float,
    backend_provenance: str,
    device_completion_fence_provenance: str,
) -> dict[str, Any]:
    # Reuse the material coordinator's target/store/coverage receipt builder
    # only after proving the full reverse counts independently.  The temporary
    # counter substitution satisfies that helper's material-only structural
    # equality; all reverse-mode fields are then replaced below.
    structure_report = structural_accounting.report(
        source,
        expected_artifact_count=counters["request_count"],
    )
    active_block_count = int(structure_report["active_native_block_count"])
    staged = policy.full_geometry_reverse_mode == STAGED_SPARSE_FULL_GEOMETRY_REVERSE
    fused = policy.full_geometry_reverse_mode == FUSED_DIRECT_V1_FULL_GEOMETRY_REVERSE
    staged_vjp_count = counters["native_full_geometry_vjp_launch_count"]
    fused_vjp_count = counters["native_fused_full_geometry_vjp_launch_count"]
    reverse_vjp_count = staged_vjp_count + fused_vjp_count
    staged_contract = staged and (
        staged_vjp_count == active_block_count
        and fused_vjp_count == 0
        and counters["native_full_geometry_fenced_reduction_count"]
        == active_block_count
        and counters["geometry_reduction_fence_call_count"] == active_block_count
        and counters["geometry_completion_receipt_count"] == active_block_count
        and counters["native_fused_full_geometry_transaction_count"] == 0
        and counters["native_fused_full_geometry_completion_fence_count"] == 0
        and counters["fused_transaction_fence_call_count"] == 0
        and counters["fused_post_accept_commit_fence_call_count"] == 0
        and counters["fused_active_manifest_certified_request_count"] == 0
        and counters["fused_compact_material_scatter_elements"] == 0
        and geometry_peaks["maximum_native_length_bar_tensor_bytes"] > 0
        and geometry_peaks[
            "maximum_simultaneous_geometry_jw_length_bar_tensors"
        ]
        == 1
    )
    fused_contract = fused and (
        staged_vjp_count == 0
        and fused_vjp_count == active_block_count
        and counters["native_full_geometry_fenced_reduction_count"] == 0
        and counters["geometry_reduction_fence_call_count"] == 0
        and counters["geometry_completion_receipt_count"] == 0
        and counters["native_fused_full_geometry_transaction_count"]
        == counters["request_count"]
        and counters["native_fused_full_geometry_completion_fence_count"]
        == counters["request_count"]
        and counters["fused_transaction_fence_call_count"]
        == counters["request_count"]
        and counters["fused_post_accept_commit_fence_call_count"]
        == counters["request_count"]
        and counters["fused_active_manifest_certified_request_count"]
        == counters["request_count"]
        and counters["fused_compact_material_scatter_elements"] > 0
        and geometry_peaks["maximum_native_length_bar_tensor_bytes"] == 0
        and geometry_peaks[
            "maximum_simultaneous_geometry_jw_length_bar_tensors"
        ]
        == 0
        and geometry_peaks["fused_prepared_owned_logical_tensor_bytes"] > 0
        and geometry_peaks["fused_output_scratch_logical_tensor_bytes"] > 0
        and geometry_peaks[
            "fused_geometry_bridge_visible_tensor_bytes"
        ]
        > 0
    )
    if (
        reverse_vjp_count != active_block_count
        or not (staged_contract or fused_contract)
        or counters["native_material_vjp_launch_count"] != 0
        or counters["geometry_row_vjp_call_count"] < 1
    ):
        raise ArithmeticError(
            "full-geometry step structural/reverse/reduction counts disagree"
        )
    common_counters = dict(counters)
    common_counters["native_material_vjp_launch_count"] = reverse_vjp_count
    result = _material_step._step_accounting(
        source,
        accumulator,
        replay_receipt,
        policy=policy,
        generation_policy=generation_policy,
        counters=common_counters,
        peaks=common_peaks,
        structural_accounting=structural_accounting,
        store_after=store_after,
        loss_rgb_mean=loss_rgb_mean,
        backend_provenance=backend_provenance,
        device_completion_fence_provenance=device_completion_fence_provenance,
    )
    result.pop("material_step_accumulator_preflight_logical_tensor_bytes", None)
    result.update(
        {
            "provenance": STEP_PROVENANCE,
            "runtime_status": STEP_STATUS,
            "provider_generation_digest": source.provider_generation_digest,
            "provider_identity": id(source.provider),
            "world_generation_digest": source.provider.world.generation_digest,
            "world_sites_content_digest": (
                source.provider.world.sites_content_digest
            ),
            "world_site_count": source.provider.world.site_count,
            "site_table_identity": id(source.provider.world.sites),
            "geometry_generation_id": (
                paper_kinetic_fixed_camera_provider_geometry_generation_id(
                    source.provider
                )
            ),
            "native_material_vjp_launch_count": 0,
            "full_geometry_reverse_mode": policy.full_geometry_reverse_mode,
            "geometry_reduction_mode": (
                "fused_direct_v1" if fused else "certified_sparse_compact"
            ),
            "native_full_geometry_vjp_launch_count": staged_vjp_count,
            "native_fused_full_geometry_vjp_launch_count": fused_vjp_count,
            "native_fused_full_geometry_transaction_count": counters[
                "native_fused_full_geometry_transaction_count"
            ],
            "native_fused_full_geometry_completion_fence_count": counters[
                "native_fused_full_geometry_completion_fence_count"
            ],
            "native_full_geometry_fenced_reduction_count": counters[
                "native_full_geometry_fenced_reduction_count"
            ],
            "exactly_one_full_geometry_vjp_per_active_block": staged,
            "exactly_one_fused_full_geometry_vjp_per_active_block": fused,
            "exactly_one_selected_full_geometry_vjp_per_active_block": True,
            "geometry_reduction_fence_call_count": counters[
                "geometry_reduction_fence_call_count"
            ],
            "geometry_completion_receipt_count": counters[
                "geometry_completion_receipt_count"
            ],
            "geometry_completion_receipt_retains_native_tensors": False,
            "native_length_bar_released_after_fenced_reduction": staged,
            "fused_length_cotangent_allocated": False,
            "fused_optimizer_commit_performed": False,
            "fused_transaction_fence_call_count": counters[
                "fused_transaction_fence_call_count"
            ],
            "fused_post_accept_commit_fence_call_count": counters[
                "fused_post_accept_commit_fence_call_count"
            ],
            "fused_active_manifest_coverage_certified": (
                counters["fused_active_manifest_certified_request_count"]
                == counters["request_count"]
                if fused
                else False
            ),
            "fused_compact_material_scatter_elements": counters[
                "fused_compact_material_scatter_elements"
            ],
            "geometry_compact_to_global_scatter_elements": counters[
                "geometry_compact_to_global_scatter_elements"
            ],
            "geometry_row_vjp_call_count": counters["geometry_row_vjp_call_count"],
            "maximum_native_length_bar_tensor_bytes": geometry_peaks[
                "maximum_native_length_bar_tensor_bytes"
            ],
            "maximum_simultaneous_geometry_jw_length_bar_tensors": (
                geometry_peaks[
                    "maximum_simultaneous_geometry_jw_length_bar_tensors"
                ]
            ),
            "maximum_geometry_bridge_visible_tensor_bytes": geometry_peaks[
                "maximum_geometry_bridge_visible_tensor_bytes"
            ],
            "staged_sparse_geometry_bridge_visible_peak_logical_tensor_bytes_upper_bound": (
                geometry_peaks[
                    "staged_sparse_geometry_bridge_visible_peak_logical_tensor_bytes_upper_bound"
                ]
            ),
            "staged_sparse_geometry_bridge_included_in_main_active_peak": (
                staged
                and geometry_peaks[
                    "staged_sparse_geometry_bridge_visible_peak_logical_tensor_bytes_upper_bound"
                ]
                > 0
            ),
            "maximum_request_fused_prepared_owned_logical_tensor_bytes": geometry_peaks[
                "fused_prepared_owned_logical_tensor_bytes"
            ],
            "maximum_request_fused_output_scratch_logical_tensor_bytes": geometry_peaks[
                "fused_output_scratch_logical_tensor_bytes"
            ],
            "maximum_request_fused_compact_output_scratch_logical_tensor_bytes": (
                geometry_peaks[
                    "fused_compact_output_scratch_logical_tensor_bytes"
                ]
            ),
            "maximum_request_fused_global_output_scratch_logical_tensor_bytes": (
                geometry_peaks[
                    "fused_global_output_scratch_logical_tensor_bytes"
                ]
            ),
            "maximum_request_fused_geometry_bridge_visible_tensor_bytes": geometry_peaks[
                "fused_geometry_bridge_visible_tensor_bytes"
            ],
            "maximum_fused_prepared_owned_logical_tensor_bytes": (
                policy.request_memory_policy.maximum_fused_prepared_owned_logical_tensor_bytes
            ),
            "maximum_fused_output_scratch_logical_tensor_bytes": (
                policy.request_memory_policy.maximum_fused_output_scratch_logical_tensor_bytes
            ),
            "maximum_fused_geometry_bridge_visible_peak_logical_tensor_bytes": (
                policy.request_memory_policy.maximum_fused_geometry_bridge_visible_peak_logical_tensor_bytes
            ),
            "maximum_fused_validation_status_tensor_bytes": geometry_peaks[
                "fused_validation_status_tensor_bytes"
            ],
            "maximum_in_flight_active_block_commit_scratch_count": common_peaks[
                "maximum_in_flight_active_block_commit_scratch_count"
            ],
            "maximum_request_geometry_bar_tensor_bytes": geometry_peaks[
                "request_geometry_bar_tensor_bytes"
            ],
            "maximum_request_delta_ray_bar_key_logical_bytes": geometry_peaks[
                "request_delta_ray_bar_key_logical_bytes"
            ],
            "geometry_bar_tensor_bytes": _geometry_bar_tensor_bytes(accumulator),
            "fixed_camera_full_geometry_step_accumulator_preflight_logical_tensor_bytes": (
                _step_accumulator_tensor_bytes(source.provider)
            ),
            "maximum_geometry_bar_logical_tensor_bytes": (
                policy.maximum_geometry_bar_logical_tensor_bytes
            ),
            "geometry_bar_memory_receipt_kind": "logical_tensor_bytes",
            "geometry_bar_allocator_peak_measured": False,
            "full_geometry": True,
            "full_geometry_vjp_integrated": True,
            "camera_ray_gradients_enabled": False,
            "fixed_camera_avoids_global_ray_bar": True,
            "step_ray_bar_key_logical_bytes": 0,
            "peak_ray_payload_logical_tensor_bytes": 0,
            "optimizer_authorization_requires_full_manifest_seal": True,
            "fixed_camera_full_geometry_step_coordinator_integrated": True,
            "production_trainer_integrated": False,
            "geometry_update_executed": False,
            "fresh_world_recompile_executed": False,
            "stale_structure_reuse_prevention_integrated": False,
            "coordinator_completion_semantics": (
                "authorization_only_external_geometry_update_and_recompile_required"
            ),
        }
    )
    return result


def _assert_result_reverse_accounting(accounting: Mapping[str, Any]) -> None:
    """Validate one staged or fused reverse without conflating their ledgers."""

    mode = accounting.get("full_geometry_reverse_mode")
    staged = mode == STAGED_SPARSE_FULL_GEOMETRY_REVERSE
    fused = mode == FUSED_DIRECT_V1_FULL_GEOMETRY_REVERSE
    active_block_count = int(accounting.get("active_native_block_count", -1))
    request_count = int(accounting.get("exact_request_count", -1))
    staged_vjp_count = int(
        accounting.get("native_full_geometry_vjp_launch_count", -1)
    )
    fused_vjp_count = int(
        accounting.get("native_fused_full_geometry_vjp_launch_count", -1)
    )
    common = (
        active_block_count > 0
        and request_count > 0
        and staged_vjp_count + fused_vjp_count == active_block_count
        and accounting.get(
            "exactly_one_selected_full_geometry_vjp_per_active_block"
        )
        is True
        and accounting.get("fused_length_cotangent_allocated") is False
    )
    staged_contract = staged and (
        accounting.get("geometry_reduction_mode")
        == "certified_sparse_compact"
        and staged_vjp_count == active_block_count
        and fused_vjp_count == 0
        and int(
            accounting.get("native_full_geometry_fenced_reduction_count", -1)
        )
        == active_block_count
        and int(accounting.get("geometry_reduction_fence_call_count", -1))
        == active_block_count
        and int(accounting.get("geometry_completion_receipt_count", -1))
        == active_block_count
        and int(
            accounting.get("native_fused_full_geometry_transaction_count", -1)
        )
        == 0
        and int(
            accounting.get(
                "native_fused_full_geometry_completion_fence_count",
                -1,
            )
        )
        == 0
        and accounting.get("exactly_one_full_geometry_vjp_per_active_block")
        is True
        and accounting.get(
            "exactly_one_fused_full_geometry_vjp_per_active_block"
        )
        is False
        and accounting.get("native_length_bar_released_after_fenced_reduction")
        is True
        and int(accounting.get("maximum_native_length_bar_tensor_bytes", 0)) > 0
        and int(
            accounting.get(
                "maximum_simultaneous_geometry_jw_length_bar_tensors",
                0,
            )
        )
        == 1
        and int(accounting.get("fused_transaction_fence_call_count", -1)) == 0
        and int(
            accounting.get("fused_post_accept_commit_fence_call_count", -1)
        )
        == 0
        and accounting.get("fused_active_manifest_coverage_certified") is False
        and accounting.get("fused_optimizer_commit_performed") is False
        and int(accounting.get("maximum_fused_validation_status_tensor_bytes", -1))
        == 0
        and int(accounting.get("fused_compact_material_scatter_elements", -1))
        == 0
        and int(
            accounting.get(
                "maximum_request_fused_prepared_owned_logical_tensor_bytes",
                -1,
            )
        )
        == 0
        and int(
            accounting.get(
                "maximum_request_fused_output_scratch_logical_tensor_bytes",
                -1,
            )
        )
        == 0
        and int(
            accounting.get(
                "maximum_request_fused_compact_output_scratch_logical_tensor_bytes",
                -1,
            )
        )
        == 0
        and int(
            accounting.get(
                "maximum_request_fused_global_output_scratch_logical_tensor_bytes",
                -1,
            )
        )
        == 0
        and int(
            accounting.get(
                "maximum_request_fused_geometry_bridge_visible_tensor_bytes",
                -1,
            )
        )
        == 0
        and int(
            accounting.get(
                "maximum_in_flight_active_block_commit_scratch_count",
                -1,
            )
        )
        == 1
        and int(
            accounting.get(
                "staged_sparse_geometry_bridge_visible_peak_logical_tensor_bytes_upper_bound",
                0,
            )
        )
        >= int(accounting.get("maximum_geometry_bridge_visible_tensor_bytes", 0))
        and accounting.get(
            "staged_sparse_geometry_bridge_included_in_main_active_peak"
        )
        is True
    )
    fused_contract = fused and (
        accounting.get("geometry_reduction_mode") == "fused_direct_v1"
        and staged_vjp_count == 0
        and fused_vjp_count == active_block_count
        and int(
            accounting.get("native_full_geometry_fenced_reduction_count", -1)
        )
        == 0
        and int(accounting.get("geometry_reduction_fence_call_count", -1)) == 0
        and int(accounting.get("geometry_completion_receipt_count", -1)) == 0
        and int(
            accounting.get("native_fused_full_geometry_transaction_count", -1)
        )
        == request_count
        and int(
            accounting.get(
                "native_fused_full_geometry_completion_fence_count",
                -1,
            )
        )
        == request_count
        and accounting.get("exactly_one_full_geometry_vjp_per_active_block")
        is False
        and accounting.get(
            "exactly_one_fused_full_geometry_vjp_per_active_block"
        )
        is True
        and accounting.get("native_length_bar_released_after_fenced_reduction")
        is False
        and int(accounting.get("maximum_native_length_bar_tensor_bytes", -1)) == 0
        and int(
            accounting.get(
                "maximum_simultaneous_geometry_jw_length_bar_tensors",
                -1,
            )
        )
        == 0
        and int(accounting.get("fused_transaction_fence_call_count", -1))
        == request_count
        and int(
            accounting.get("fused_post_accept_commit_fence_call_count", -1)
        )
        == request_count
        and accounting.get("fused_active_manifest_coverage_certified") is True
        and accounting.get("fused_optimizer_commit_performed") is False
        and int(accounting.get("maximum_fused_validation_status_tensor_bytes", -1))
        == 4
        and int(accounting.get("fused_compact_material_scatter_elements", 0))
        > 0
        and int(
            accounting.get(
                "maximum_request_fused_prepared_owned_logical_tensor_bytes",
                0,
            )
        )
        > 0
        and int(
            accounting.get(
                "maximum_request_fused_output_scratch_logical_tensor_bytes",
                0,
            )
        )
        > 0
        and int(
            accounting.get(
                "maximum_request_fused_geometry_bridge_visible_tensor_bytes",
                0,
            )
        )
        > 0
        and int(
            accounting[
                "maximum_request_fused_output_scratch_logical_tensor_bytes"
            ]
        )
        == int(
            accounting[
                "maximum_request_fused_compact_output_scratch_logical_tensor_bytes"
            ]
        )
        + int(
            accounting[
                "maximum_request_fused_global_output_scratch_logical_tensor_bytes"
            ]
        )
        and int(
            accounting[
                "maximum_request_fused_geometry_bridge_visible_tensor_bytes"
            ]
        )
        == int(
            accounting[
                "maximum_request_fused_global_output_scratch_logical_tensor_bytes"
            ]
        )
        + int(accounting["maximum_request_geometry_bar_tensor_bytes"])
        and int(
            accounting[
                "maximum_request_fused_prepared_owned_logical_tensor_bytes"
            ]
        )
        <= int(accounting["maximum_fused_prepared_owned_logical_tensor_bytes"])
        and int(
            accounting[
                "maximum_request_fused_output_scratch_logical_tensor_bytes"
            ]
        )
        <= int(accounting["maximum_fused_output_scratch_logical_tensor_bytes"])
        and int(
            accounting[
                "maximum_request_fused_geometry_bridge_visible_tensor_bytes"
            ]
        )
        <= int(
            accounting[
                "maximum_fused_geometry_bridge_visible_peak_logical_tensor_bytes"
            ]
        )
        and int(
            accounting.get(
                "maximum_in_flight_active_block_commit_scratch_count",
                0,
            )
        )
        > 0
        and int(
            accounting.get(
                "staged_sparse_geometry_bridge_visible_peak_logical_tensor_bytes_upper_bound",
                -1,
            )
        )
        == 0
        and accounting.get(
            "staged_sparse_geometry_bridge_included_in_main_active_peak"
        )
        is False
    )
    if not common or not (staged_contract or fused_contract):
        raise ValueError("fixed-camera full-geometry reverse accounting changed")


def _world_geometry_tensors(
    provider: PaperKineticLazyProgramBundleProvider,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sites = provider.world.sites
    return sites.positions0, sites.velocities, sites.weight_coefficients


def _geometry_bar_tensor_bytes_from_provider(
    provider: PaperKineticLazyProgramBundleProvider,
) -> int:
    return sum(tensor.numel() * 8 for tensor in _world_geometry_tensors(provider))


def _geometry_bar_tensor_bytes(
    accumulator: PaperKineticDenseStepGradientAccumulator,
) -> int:
    tensors = (
        accumulator.grad_positions0_f64,
        accumulator.grad_velocities_f64,
        accumulator.grad_weight_coefficients_f64,
    )
    if any(tensor is None for tensor in tensors):
        return 0
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _step_accumulator_tensor_bytes(
    provider: PaperKineticLazyProgramBundleProvider,
) -> int:
    material_and_loss_bytes = (4 * provider.world.site_count + 1) * 4
    return material_and_loss_bytes + _geometry_bar_tensor_bytes_from_provider(provider)


def paper_kinetic_fixed_camera_geometry_generation_id(
    *,
    world_generation_digest: str,
    world_sites_content_digest: str,
    world_site_count: int,
) -> str:
    """Canonical logical geometry ID for one already-certified world snapshot."""

    return _material_step._digest_parts(
        STEP_PROVENANCE,
        "provider-bound-geometry-generation-v1",
        world_generation_digest,
        world_sites_content_digest,
        world_site_count,
        "fixed-camera/no-ray-bars",
    )


def _result_digest(
    result: PaperKineticFixedCameraFullGeometryStepResult,
) -> str:
    return _material_step._digest_parts(
        STEP_PROVENANCE,
        result.runtime_status,
        result.authorization.generation_digest,
        result.accumulator.generation_digest,
        result.replay_receipt.generation_digest,
        result.loss_rgb_mean,
        tuple(result.accounting.items()),
        result.parameter_mutation_count,
        result.retained_authorization_capability_object_count,
        result.retained_source_count,
        result.retained_session_count,
        result.retained_request_count,
        result.retained_artifact_count,
        result.retained_target_count,
        result.retained_native_lane_count,
        result.native_runtime_verified,
        result.allocator_peak_measured,
    )


def _zero_counters() -> dict[str, int]:
    return {
        "request_count": 0,
        "cold_artifact_count": 0,
        "warm_artifact_count": 0,
        "artifact_store_eviction_count": 0,
        "artifact_store_evicted_accounted_bytes": 0,
        "artifact_store_cold_compiled_track_count": 0,
        "artifact_store_avoided_compile_track_count": 0,
        "streamed_observation_count": 0,
        "replay_chunk_count": 0,
        "sample_launch_count": 0,
        "sample_node_interaction_count": 0,
        "transferred_target_payload_bytes": 0,
        "native_material_vjp_launch_count": 0,
        "native_full_geometry_vjp_launch_count": 0,
        "native_fused_full_geometry_vjp_launch_count": 0,
        "native_fused_full_geometry_transaction_count": 0,
        "native_fused_full_geometry_completion_fence_count": 0,
        "native_full_geometry_fenced_reduction_count": 0,
        "geometry_reduction_fence_call_count": 0,
        "geometry_completion_receipt_count": 0,
        "geometry_row_vjp_call_count": 0,
        "fused_transaction_fence_call_count": 0,
        "fused_post_accept_commit_fence_call_count": 0,
        "fused_active_manifest_certified_request_count": 0,
        "fused_compact_material_scatter_elements": 0,
        "geometry_compact_to_global_scatter_elements": 0,
        "request_commit_fence_call_count": 0,
        "native_lane_fence_call_count": 0,
        "selected_pixel_read_call_count": 0,
        "mapped_selected_pixel_read_call_count": 0,
        "mapping_closed_before_return_count": 0,
        "cumulative_requested_mapped_page_count": 0,
        "cumulative_requested_mapped_page_bytes_upper_bound": 0,
        "direct_selected_pixel_observation_count": 0,
        "bounded_region_selected_pixel_observation_count": 0,
        "full_frame_fallback_observation_count": 0,
        "full_frame_target_materialization_count": 0,
        "bounded_region_target_materialization_count": 0,
        "decoded_frame_count": 0,
        "accumulator_initialization_fence_call_count": 0,
    }


_COMMON_SUM_KEYS = (
    ("streamed_observation_count", "streamed_observation_count"),
    ("replay_chunk_count", "replay_chunk_count"),
    ("sample_launch_count", "sample_launch_count"),
    ("sample_node_interaction_count", "sample_node_interaction_count"),
    ("transferred_target_payload_bytes", "transferred_target_payload_bytes"),
    ("native_material_vjp_launch_count", "native_material_word_vjp_launch_count"),
    ("native_lane_fence_call_count", "native_lane_fence_count"),
    ("selected_pixel_read_call_count", "selected_pixel_read_call_count"),
    (
        "mapped_selected_pixel_read_call_count",
        "mapped_selected_pixel_read_call_count",
    ),
    ("mapping_closed_before_return_count", "mapping_closed_before_return_count"),
    (
        "cumulative_requested_mapped_page_count",
        "cumulative_requested_mapped_page_count",
    ),
    (
        "cumulative_requested_mapped_page_bytes_upper_bound",
        "cumulative_requested_mapped_page_bytes_upper_bound",
    ),
    (
        "direct_selected_pixel_observation_count",
        "direct_selected_pixel_observation_count",
    ),
    (
        "bounded_region_selected_pixel_observation_count",
        "bounded_region_selected_pixel_observation_count",
    ),
    (
        "full_frame_fallback_observation_count",
        "full_frame_fallback_observation_count",
    ),
    (
        "full_frame_target_materialization_count",
        "full_frame_target_materialization_count",
    ),
    (
        "bounded_region_target_materialization_count",
        "bounded_region_target_materialization_count",
    ),
    ("decoded_frame_count", "decoded_frame_count"),
)


_COMMON_PEAK_KEYS = (
    "lane_resident_logical_tensor_bytes_upper_bound",
    "active_request_logical_tensor_bytes_upper_bound",
    "reverse_lane_plus_active_logical_tensor_bytes_upper_bound",
    "peak_target_decode_bridge_logical_tensor_bytes",
    "peak_sample_launch_tensor_bytes",
    "peak_sample_launch_node_count",
    "peak_cpu_decoded_frame_tensor_bytes",
    "peak_bounded_region_materialization_tensor_bytes",
    "peak_source_visible_target_read_logical_tensor_bytes_upper_bound",
    "peak_transient_mapped_address_space_bytes",
    "peak_requested_unique_mapped_page_count",
    "peak_mapped_page_size_bytes",
    "peak_requested_mapped_page_bytes_upper_bound",
    "peak_cpu_chunk_target_tensor_bytes",
    "peak_device_chunk_target_tensor_bytes",
    "peak_sample_materialization_logical_tensor_bytes_upper_bound",
    "peak_interpolation_evaluator_scratch_logical_tensor_bytes_upper_bound",
    "maximum_interpolation_rows_per_subchunk",
    "effective_maximum_samples_per_launch",
    "peak_native_prepared_sample_scratch_tensor_bytes",
    "peak_public_sample_launch_logical_tensor_bytes",
    "peak_chunk_dispatch_identity_logical_bytes",
    "maximum_active_block_commit_scratch_tensor_bytes",
    "maximum_in_flight_active_block_commit_scratch_count",
    "request_delta_logical_tensor_bytes",
)


_GEOMETRY_PEAK_KEYS = (
    "maximum_native_length_bar_tensor_bytes",
    "maximum_simultaneous_geometry_jw_length_bar_tensors",
    "maximum_geometry_bridge_visible_tensor_bytes",
    "staged_sparse_geometry_bridge_visible_peak_logical_tensor_bytes_upper_bound",
    "fused_prepared_owned_logical_tensor_bytes",
    "fused_output_scratch_logical_tensor_bytes",
    "fused_compact_output_scratch_logical_tensor_bytes",
    "fused_global_output_scratch_logical_tensor_bytes",
    "fused_geometry_bridge_visible_tensor_bytes",
    "fused_validation_status_tensor_bytes",
    "request_geometry_bar_tensor_bytes",
    "request_delta_ray_bar_key_logical_bytes",
)


__all__ = [
    "GLOBAL_RGB_MEAN_LOSS_NORMALIZATION_ID",
    "STEP_PROVENANCE",
    "STEP_STATUS",
    "PaperKineticFixedCameraFullGeometryGenerationPolicy",
    "PaperKineticFixedCameraFullGeometryStepPartialFailure",
    "PaperKineticFixedCameraFullGeometryStepPolicy",
    "PaperKineticFixedCameraFullGeometryStepResult",
    "PaperKineticFixedCameraFullGeometryStepState",
    "prepare_paper_kinetic_fixed_camera_full_geometry_step_state",
    "run_paper_kinetic_fixed_camera_full_geometry_step",
]
