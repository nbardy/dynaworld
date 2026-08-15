"""Bounded CPU material/geometry update with mandatory cold recompilation.

The fixed-camera full-geometry step exposes a point-in-time authorization.  This
module consumes that capability exactly once and performs the next production
lifecycle boundary:

1. build finite material and affine-kinetic geometry candidates out of place;
2. construct a fresh immutable world/provider generation;
3. close the old artifact store, revoke the old provider seal, and poison the
   old material state;
4. stream a cold compile of every request in an explicit next-step manifest
   through a fresh bounded LRU and seal a tensor-free full-chain receipt;
5. only then expose a valid combined state/provider/store generation.

There is no rollback after retirement.  A compile/promotion failure therefore
fails closed: both the old and candidate generations are unusable and process
restart from the last durable checkpoint is required.  Checkpoint creation is
an explicit post-promotion operation with separate state+checkpoint and
payload-clone peak bounds; the live ready generation never retains that clone.
Successful promotion revokes the consumed full-geometry result and drops every
tensor reference held by its authorization/accumulator before cold compilation.
Camera-ray updates are not represented by the policy, state, checkpoint, or
receipt.

The implementation is CPU/source complete but has not been runtime-verified.
Tracked state/candidate/authorization tensors, the unavoidable candidate-world
geometry clone, update-validation scratch, and both old/fresh store-owned
accounted bytes are explicit.  Caller-retained artifacts, checkpoint payloads,
or retired-generation objects are outside that ownership boundary, so the
next-step handoff requires an explicit zero-retained-byte attestation.  Provider
recertification,
cold-compiler transient scratch, allocator, and Python-object peaks are not
measured here.
Strict policy-bounded combined-payload parsing and live restart are
source-written.  Restart reconstructs a fresh provider, material state, bounded
artifact store, and cold-compiled working set; it refuses any runtime input
whose dataset/camera/factory/provider/world generation differs from the
checkpoint.  Production-trainer routing remains a separate seam.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from paper_kinetic_runtime_paths import ensure_worldfoam_lane2_research_path

ensure_worldfoam_lane2_research_path()

from kinetic_compiled_cpu_artifact_store import (  # noqa: E402
    PaperKineticCompiledCpuArtifactStore,
    PaperKineticCompiledCpuArtifactStorePolicy,
    compile_paper_kinetic_compiled_cpu_artifact,
)
from kinetic_dense_cached_native_material_request import (  # noqa: E402
    FUSED_DIRECT_V1_FULL_GEOMETRY_REVERSE,
    STAGED_SPARSE_FULL_GEOMETRY_REVERSE,
    PaperKineticDenseOptimizerAuthorization,
    PaperKineticDenseStepGradientAccumulator,
)
from kinetic_power_word_compiler import AffineKineticPowerSites  # noqa: E402
from paper_kinetic_fixed_camera_full_geometry_step import (  # noqa: E402
    PaperKineticFixedCameraFullGeometryStepState,
    PaperKineticFixedCameraFullGeometryStepResult,
    paper_kinetic_fixed_camera_geometry_generation_id,
    paper_kinetic_fixed_camera_provider_geometry_generation_id,
    prepare_paper_kinetic_fixed_camera_full_geometry_step_state,
)
import paper_kinetic_fixed_site_material_state as _material_state  # noqa: E402
from paper_kinetic_fixed_site_material_state import (  # noqa: E402
    PaperKineticFixedSiteMaterialCheckpoint,
    PaperKineticFixedSiteMaterialState,
    checkpoint_paper_kinetic_fixed_site_material_state,
    paper_kinetic_fixed_site_material_checkpoint_from_payload,
    restore_paper_kinetic_fixed_site_material_state,
)
from paper_kinetic_lazy_program_bundles import (  # noqa: E402
    PaperKineticLazyProgramBundleProvider,
    PaperKineticTrackProgramFactory,
    PaperKineticWorldInitializationRequest,
    prepare_paper_kinetic_lazy_program_bundle_provider,
)
from paper_kinetic_replayable_observations import (  # noqa: E402
    PaperKineticDenseObservationReplayReceipt,
)
from powerfoam_training_data import (  # noqa: E402
    PowerFoamRayProvider,
    PowerFoamTargetProvider,
)


STATE_PROVENANCE = "paper-kinetic-fixed-camera-combined-state-v1"
UPDATE_PROVENANCE = "paper-kinetic-fixed-camera-combined-sgd-transaction-v1"
RECOMPILE_PROVENANCE = "paper-kinetic-fixed-camera-cold-recompile-seal-v1"
CHECKPOINT_PROVENANCE = "paper-kinetic-fixed-camera-combined-checkpoint-v1"
CHECKPOINT_SCHEMA = "paper_kinetic_fixed_camera_combined_checkpoint_v1"
READY_PROVENANCE = "paper-kinetic-fixed-camera-ready-generation-v1"
RESTORE_PROVENANCE = "paper-kinetic-fixed-camera-combined-restore-v1"
RESTORED_READY_PROVENANCE = (
    "paper-kinetic-fixed-camera-restored-ready-generation-v1"
)
RUNTIME_STATUS = "source_integrated/runtime_unverified"

_STATE_SEAL = object()
_UPDATE_SEAL = object()
_RECOMPILE_SEAL = object()
_CHECKPOINT_SEAL = object()
_READY_SEAL = object()
_RESTORE_SEAL = object()
_RESTORED_READY_SEAL = object()
_LOCK_TYPE = type(threading.Lock())


@dataclass(frozen=True)
class PaperKineticFixedCameraCombinedSGDPolicy:
    """Stateless SGD and explicit transaction-owned logical/accounted bounds."""

    position_learning_rate: float
    velocity_learning_rate: float
    weight_learning_rate: float
    maximum_absolute_position_update: float
    maximum_absolute_velocity_update: float
    maximum_absolute_weight_update: float
    maximum_absolute_position_value: float
    maximum_absolute_velocity_value: float
    maximum_absolute_weight_value: float
    maximum_combined_state_logical_tensor_bytes: int
    maximum_update_candidate_logical_tensor_bytes: int
    maximum_candidate_world_geometry_clone_logical_tensor_bytes: int
    maximum_update_validation_scratch_logical_tensor_bytes: int
    maximum_old_candidate_authorization_logical_tensor_bytes: int
    maximum_checkpoint_logical_tensor_bytes: int
    maximum_state_checkpoint_logical_tensor_bytes: int
    maximum_state_checkpoint_payload_logical_tensor_bytes: int
    maximum_transaction_tracked_logical_and_store_accounted_bytes: int
    maximum_recompile_request_count: int
    maximum_recompile_track_id_logical_bytes: int
    maximum_artifact_accounted_bytes: int

    def assert_valid(self) -> None:
        for name, value in (
            ("position_learning_rate", self.position_learning_rate),
            ("velocity_learning_rate", self.velocity_learning_rate),
            ("weight_learning_rate", self.weight_learning_rate),
            (
                "maximum_absolute_position_update",
                self.maximum_absolute_position_update,
            ),
            (
                "maximum_absolute_velocity_update",
                self.maximum_absolute_velocity_update,
            ),
            ("maximum_absolute_weight_update", self.maximum_absolute_weight_update),
            ("maximum_absolute_position_value", self.maximum_absolute_position_value),
            ("maximum_absolute_velocity_value", self.maximum_absolute_velocity_value),
            ("maximum_absolute_weight_value", self.maximum_absolute_weight_value),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            (
                "maximum_combined_state_logical_tensor_bytes",
                self.maximum_combined_state_logical_tensor_bytes,
            ),
            (
                "maximum_update_candidate_logical_tensor_bytes",
                self.maximum_update_candidate_logical_tensor_bytes,
            ),
            (
                "maximum_candidate_world_geometry_clone_logical_tensor_bytes",
                self.maximum_candidate_world_geometry_clone_logical_tensor_bytes,
            ),
            (
                "maximum_update_validation_scratch_logical_tensor_bytes",
                self.maximum_update_validation_scratch_logical_tensor_bytes,
            ),
            (
                "maximum_old_candidate_authorization_logical_tensor_bytes",
                self.maximum_old_candidate_authorization_logical_tensor_bytes,
            ),
            (
                "maximum_checkpoint_logical_tensor_bytes",
                self.maximum_checkpoint_logical_tensor_bytes,
            ),
            (
                "maximum_state_checkpoint_logical_tensor_bytes",
                self.maximum_state_checkpoint_logical_tensor_bytes,
            ),
            (
                "maximum_state_checkpoint_payload_logical_tensor_bytes",
                self.maximum_state_checkpoint_payload_logical_tensor_bytes,
            ),
            (
                "maximum_transaction_tracked_logical_and_store_accounted_bytes",
                self.maximum_transaction_tracked_logical_and_store_accounted_bytes,
            ),
            ("maximum_recompile_request_count", self.maximum_recompile_request_count),
            (
                "maximum_recompile_track_id_logical_bytes",
                self.maximum_recompile_track_id_logical_bytes,
            ),
            (
                "maximum_artifact_accounted_bytes",
                self.maximum_artifact_accounted_bytes,
            ),
        ):
            _require_positive_int(value, name=name)

    @property
    def generation_digest(self) -> str:
        self.assert_valid()
        return _digest_parts(
            UPDATE_PROVENANCE,
            tuple(self.__dict__.items()),
            "manual_sgd/no_momentum/no_weight_decay",
            "fixed_camera/no_ray_updates",
        )

    def payload(self) -> dict[str, float | int | str | bool]:
        self.assert_valid()
        return {
            **self.__dict__,
            "optimizer": "manual_sgd",
            "momentum": 0.0,
            "weight_decay": 0.0,
            "fixed_camera": True,
            "ray_updates_enabled": False,
            "generation_digest": self.generation_digest,
        }


@dataclass(frozen=True)
class PaperKineticFixedCameraColdRecompileRequest:
    view_index: int
    track_start: int
    track_stop: int

    @property
    def track_count(self) -> int:
        return self.track_stop - self.track_start

    @property
    def track_ids(self) -> tuple[int, ...]:
        """Materialize only the one bounded request currently being compiled."""

        return tuple(range(self.track_start, self.track_stop))

    def assert_compatible(
        self,
        provider: PaperKineticLazyProgramBundleProvider,
    ) -> None:
        if (
            isinstance(self.view_index, bool)
            or not isinstance(self.view_index, int)
            or not 0 <= self.view_index < provider.view_count
            or isinstance(self.track_start, bool)
            or not isinstance(self.track_start, int)
            or isinstance(self.track_stop, bool)
            or not isinstance(self.track_stop, int)
            or self.track_start < 0
            or self.track_stop <= self.track_start
            or self.track_stop > provider.height * provider.width
            or self.track_count > provider.maximum_tracks_per_bundle
        ):
            raise ValueError("cold-recompile request is incompatible with provider")


@dataclass(frozen=True)
class PaperKineticFixedCameraColdRecompileManifest:
    dataset_generation_digest: str
    camera_grid_digest: str
    factory_generation_digest: str
    height: int
    width: int
    maximum_tracks_per_bundle: int
    requests: tuple[PaperKineticFixedCameraColdRecompileRequest, ...]
    generation_digest: str

    @property
    def request_count(self) -> int:
        return len(self.requests)

    @property
    def track_count(self) -> int:
        return sum(request.track_count for request in self.requests)

    @property
    def track_id_logical_bytes(self) -> int:
        # Persistent partition metadata is three int64-equivalent scalars per
        # interval. Concrete track-id tuples are request-local and bounded by
        # ``maximum_tracks_per_bundle``.
        return self.request_count * 3 * 8

    def assert_self_consistent(self) -> None:
        for name, value in (
            ("height", self.height),
            ("width", self.width),
            ("maximum_tracks_per_bundle", self.maximum_tracks_per_bundle),
        ):
            _require_positive_int(value, name=name)
        if (
            not isinstance(self.requests, tuple)
            or not self.requests
            or any(
                not isinstance(request, PaperKineticFixedCameraColdRecompileRequest)
                for request in self.requests
            )
        ):
            raise ValueError("cold-recompile manifest requests changed")
        canonical = tuple(
            sorted(
                self.requests,
                key=lambda request: (
                    request.view_index,
                    request.track_start,
                ),
            )
        )
        image_pixel_count = self.height * self.width
        if (
            canonical != self.requests
            or len(set(self.requests)) != len(self.requests)
            or any(
                isinstance(request.view_index, bool)
                or not isinstance(request.view_index, int)
                or request.view_index < 0
                or isinstance(request.track_start, bool)
                or not isinstance(request.track_start, int)
                or isinstance(request.track_stop, bool)
                or not isinstance(request.track_stop, int)
                or request.track_start < 0
                or request.track_stop <= request.track_start
                or request.track_stop > image_pixel_count
                or request.track_count > self.maximum_tracks_per_bundle
                for request in self.requests
            )
            or any(
                left.view_index == right.view_index
                and right.track_start < left.track_stop
                for left, right in zip(self.requests, self.requests[1:])
            )
        ):
            raise ValueError("cold-recompile manifest is not self-consistent")
        for name, digest in (
            ("dataset_generation_digest", self.dataset_generation_digest),
            ("camera_grid_digest", self.camera_grid_digest),
            ("factory_generation_digest", self.factory_generation_digest),
        ):
            _require_sha256(digest, name=name)
        if self.generation_digest != _manifest_digest(self):
            raise ValueError("cold-recompile manifest generation changed")

    def assert_compatible(
        self,
        provider: PaperKineticLazyProgramBundleProvider,
    ) -> None:
        provider.assert_current()
        self.assert_self_consistent()
        if (
            self.dataset_generation_digest != provider.dataset_generation_digest
            or self.camera_grid_digest != provider.camera_grid_digest
            or self.factory_generation_digest != provider.factory_generation_digest
            or self.height != provider.height
            or self.width != provider.width
            or self.maximum_tracks_per_bundle
            != provider.maximum_tracks_per_bundle
        ):
            raise ValueError("cold-recompile manifest changed or is foreign")
        for request in self.requests:
            request.assert_compatible(provider)

    def payload(self) -> dict[str, Any]:
        self.assert_self_consistent()
        return {
            "dataset_generation_digest": self.dataset_generation_digest,
            "camera_grid_digest": self.camera_grid_digest,
            "factory_generation_digest": self.factory_generation_digest,
            "height": self.height,
            "width": self.width,
            "maximum_tracks_per_bundle": self.maximum_tracks_per_bundle,
            "requests": tuple(
                (request.view_index, request.track_start, request.track_stop)
                for request in self.requests
            ),
            "request_count": self.request_count,
            "track_count": self.track_count,
            "persistent_partition_logical_bytes": self.track_id_logical_bytes,
            "generation_digest": self.generation_digest,
        }


def prepare_paper_kinetic_fixed_camera_cold_recompile_manifest(
    provider: PaperKineticLazyProgramBundleProvider,
    *,
    view_indices: Sequence[int],
    maximum_tracks_per_request: int,
) -> PaperKineticFixedCameraColdRecompileManifest:
    """Partition an exact view/pixel working set independently of world geometry."""

    provider.assert_current()
    _require_positive_int(
        maximum_tracks_per_request,
        name="maximum_tracks_per_request",
    )
    views = tuple(view_indices)
    if (
        not views
        or any(
            isinstance(view_index, bool) or not isinstance(view_index, int)
            for view_index in views
        )
        or tuple(sorted(set(views))) != views
        or views[0] < 0
        or views[-1] >= provider.view_count
        or maximum_tracks_per_request > provider.maximum_tracks_per_bundle
    ):
        raise ValueError("cold-recompile view/track partition is invalid")
    image_pixel_count = provider.height * provider.width
    requests = tuple(
        PaperKineticFixedCameraColdRecompileRequest(
            view_index=view_index,
            track_start=track_start,
            track_stop=min(
                track_start + maximum_tracks_per_request,
                image_pixel_count,
            ),
        )
        for view_index in views
        for track_start in range(0, image_pixel_count, maximum_tracks_per_request)
    )
    provisional = PaperKineticFixedCameraColdRecompileManifest(
        dataset_generation_digest=provider.dataset_generation_digest,
        camera_grid_digest=provider.camera_grid_digest,
        factory_generation_digest=provider.factory_generation_digest,
        height=provider.height,
        width=provider.width,
        maximum_tracks_per_bundle=provider.maximum_tracks_per_bundle,
        requests=requests,
        generation_digest="",
    )
    result = replace(provisional, generation_digest=_manifest_digest(provisional))
    result.assert_compatible(provider)
    return result


def prepare_paper_kinetic_fixed_camera_selected_tracks_cold_recompile_manifest(
    provider: PaperKineticLazyProgramBundleProvider,
    *,
    selected_track_ids_by_view: (
        Mapping[int, Sequence[int]]
        | Sequence[tuple[int, Sequence[int]]]
    ),
    maximum_tracks_per_request: int,
    maximum_request_count: int,
    maximum_track_id_logical_bytes: int,
) -> PaperKineticFixedCameraColdRecompileManifest:
    """Seal only an explicitly selected spatial working set.

    The full-view constructor above is useful for tiny fixtures but expands
    every ``H*W`` pixel.  Publication training samples a sparse, canonical
    sensor-time slice, so restart/recompile must carry exactly those track ids.
    This constructor accepts one strictly increasing pixel sequence per view,
    coalesces adjacent pixels into non-overlapping interval requests, and
    splits each interval at the bounded request size.  It never enumerates an
    unselected pixel and rejects a selection whose explicit ids or interval
    descriptors exceed the caller-owned restart policy.
    """

    provider.assert_current()
    for name, value in (
        ("maximum_tracks_per_request", maximum_tracks_per_request),
        ("maximum_request_count", maximum_request_count),
        ("maximum_track_id_logical_bytes", maximum_track_id_logical_bytes),
    ):
        _require_positive_int(value, name=name)
    if maximum_tracks_per_request > provider.maximum_tracks_per_bundle:
        raise ValueError(
            "selected-track recompile request exceeds the provider bundle bound"
        )
    if isinstance(selected_track_ids_by_view, Mapping):
        raw_items = tuple(selected_track_ids_by_view.items())
    elif isinstance(selected_track_ids_by_view, Sequence) and not isinstance(
        selected_track_ids_by_view,
        (str, bytes),
    ):
        raw_items = tuple(selected_track_ids_by_view)
    else:
        raise TypeError(
            "selected tracks must be a mapping or canonical (view, pixels) sequence"
        )
    if not raw_items:
        raise ValueError("selected-track recompile manifest cannot be empty")

    image_pixel_count = provider.height * provider.width
    canonical_items: list[tuple[int, tuple[int, ...]]] = []
    explicit_track_count = 0
    previous_view = -1
    for raw_item in raw_items:
        if not isinstance(raw_item, tuple) or len(raw_item) != 2:
            raise ValueError("selected-track view entry changed")
        view_index, raw_pixels = raw_item
        if (
            isinstance(view_index, bool)
            or not isinstance(view_index, int)
            or view_index <= previous_view
            or not 0 <= view_index < provider.view_count
        ):
            raise ValueError(
                "selected-track views must be unique, strictly increasing, and in range"
            )
        if not isinstance(raw_pixels, Sequence) or isinstance(
            raw_pixels,
            (str, bytes),
        ):
            raise TypeError("selected-track pixels must be a bounded sequence")
        if len(raw_pixels) < 1:
            raise ValueError("each selected-track view must contain at least one pixel")
        if 8 * (explicit_track_count + len(raw_pixels)) > maximum_track_id_logical_bytes:
            raise MemoryError(
                "selected-track ids exceed the restart track-id policy"
            )
        pixels = tuple(raw_pixels)
        if any(
            isinstance(pixel, bool)
            or not isinstance(pixel, int)
            or not 0 <= pixel < image_pixel_count
            for pixel in pixels
        ) or any(right <= left for left, right in zip(pixels, pixels[1:])):
            raise ValueError(
                "selected-track pixels must be unique, strictly increasing, and in range"
            )
        canonical_items.append((view_index, pixels))
        explicit_track_count += len(pixels)
        previous_view = view_index

    requests: list[PaperKineticFixedCameraColdRecompileRequest] = []
    for view_index, pixels in canonical_items:
        run_start = pixels[0]
        run_stop = run_start + 1
        for pixel in (*pixels[1:], None):
            if pixel is not None and pixel == run_stop:
                run_stop += 1
                continue
            for track_start in range(
                run_start,
                run_stop,
                maximum_tracks_per_request,
            ):
                requests.append(
                    PaperKineticFixedCameraColdRecompileRequest(
                        view_index=view_index,
                        track_start=track_start,
                        track_stop=min(
                            track_start + maximum_tracks_per_request,
                            run_stop,
                        ),
                    )
                )
                if len(requests) > maximum_request_count:
                    raise MemoryError(
                        "selected-track intervals exceed the restart request policy"
                    )
            if pixel is not None:
                run_start = pixel
                run_stop = pixel + 1

    provisional = PaperKineticFixedCameraColdRecompileManifest(
        dataset_generation_digest=provider.dataset_generation_digest,
        camera_grid_digest=provider.camera_grid_digest,
        factory_generation_digest=provider.factory_generation_digest,
        height=provider.height,
        width=provider.width,
        maximum_tracks_per_bundle=provider.maximum_tracks_per_bundle,
        requests=tuple(requests),
        generation_digest="",
    )
    result = replace(provisional, generation_digest=_manifest_digest(provisional))
    result.assert_compatible(provider)
    if (
        result.track_count != explicit_track_count
        or result.request_count > maximum_request_count
        or result.track_id_logical_bytes > maximum_track_id_logical_bytes
    ):
        raise ArithmeticError("selected-track manifest accounting changed")
    return result


def paper_kinetic_fixed_camera_cold_recompile_manifest_from_payload(
    payload: Mapping[str, Any],
    *,
    maximum_request_count: int,
    maximum_track_id_logical_bytes: int,
) -> PaperKineticFixedCameraColdRecompileManifest:
    """Rebuild one tensor-free manifest under caller-owned restart caps."""

    _require_positive_int(maximum_request_count, name="maximum_request_count")
    _require_positive_int(
        maximum_track_id_logical_bytes,
        name="maximum_track_id_logical_bytes",
    )
    required = {
        "dataset_generation_digest",
        "camera_grid_digest",
        "factory_generation_digest",
        "height",
        "width",
        "maximum_tracks_per_bundle",
        "requests",
        "request_count",
        "track_count",
        "persistent_partition_logical_bytes",
        "generation_digest",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("cold-recompile manifest payload keys changed")
    for name in (
        "height",
        "width",
        "maximum_tracks_per_bundle",
        "request_count",
        "track_count",
        "persistent_partition_logical_bytes",
    ):
        _require_positive_int(payload[name], name=name)
    for name in (
        "dataset_generation_digest",
        "camera_grid_digest",
        "factory_generation_digest",
        "generation_digest",
    ):
        _require_sha256(payload[name], name=name)
    raw_requests = payload["requests"]
    if (
        not isinstance(raw_requests, tuple)
        or not raw_requests
        or len(raw_requests) != payload["request_count"]
        or len(raw_requests) > maximum_request_count
        or payload["persistent_partition_logical_bytes"]
        != len(raw_requests) * 3 * 8
        or payload["persistent_partition_logical_bytes"]
        > maximum_track_id_logical_bytes
    ):
        raise MemoryError(
            "cold-recompile manifest exceeds its restart request/partition bound"
        )
    requests: list[PaperKineticFixedCameraColdRecompileRequest] = []
    for raw_request in raw_requests:
        if not isinstance(raw_request, tuple) or len(raw_request) != 3:
            raise ValueError("cold-recompile manifest request payload changed")
        for name, value in zip(
            ("view_index", "track_start", "track_stop"),
            raw_request,
            strict=True,
        ):
            _require_nonnegative_int(value, name=name)
        requests.append(
            PaperKineticFixedCameraColdRecompileRequest(
                view_index=raw_request[0],
                track_start=raw_request[1],
                track_stop=raw_request[2],
            )
        )
    maximum_request_track_id_logical_bytes = 8 * max(
        request.track_count for request in requests
    )
    declared_maximum_track_id_logical_bytes = (
        8 * payload["maximum_tracks_per_bundle"]
    )
    if (
        8 * payload["track_count"] > maximum_track_id_logical_bytes
        or maximum_request_track_id_logical_bytes
        > maximum_track_id_logical_bytes
        or declared_maximum_track_id_logical_bytes
        > maximum_track_id_logical_bytes
    ):
        raise MemoryError(
            "cold-recompile manifest exceeds its request-local track-id bound"
        )
    result = PaperKineticFixedCameraColdRecompileManifest(
        dataset_generation_digest=payload["dataset_generation_digest"],
        camera_grid_digest=payload["camera_grid_digest"],
        factory_generation_digest=payload["factory_generation_digest"],
        height=payload["height"],
        width=payload["width"],
        maximum_tracks_per_bundle=payload["maximum_tracks_per_bundle"],
        requests=tuple(requests),
        generation_digest=payload["generation_digest"],
    )
    result.assert_self_consistent()
    if (
        result.request_count != payload["request_count"]
        or result.track_count != payload["track_count"]
        or result.track_id_logical_bytes
        != payload["persistent_partition_logical_bytes"]
    ):
        raise ValueError("cold-recompile manifest derived accounting changed")
    return result


@dataclass
class PaperKineticFixedCameraCombinedState:
    """O(S) material/geometry state with no artifact or program reference."""

    material_state: PaperKineticFixedSiteMaterialState = field(repr=False)
    positions0_f64: torch.Tensor = field(repr=False)
    velocities_f64: torch.Tensor = field(repr=False)
    weight_coefficients_f64: torch.Tensor = field(repr=False)
    provider_generation_digest: str
    world_generation_digest: str
    sites_content_digest: str
    geometry_generation_parent_digest: str
    geometry_generation_id: str
    last_authorization_generation_digest: str
    last_step_generation_id: str
    last_update_policy_generation_digest: str
    geometry_update_count: int
    cold_recompile_seal_generation_digest: str
    tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    generation_digest: str
    active: bool
    retired: bool
    poisoned: bool
    _provider_identity: int = field(repr=False)
    _artifact_store_identity: int = field(repr=False)
    provenance: str = STATE_PROVENANCE
    runtime_status: str = RUNTIME_STATUS
    persistent_frame_tensor_bytes: int = 0
    persistent_sample_tensor_bytes: int = 0
    persistent_target_tensor_bytes: int = 0
    persistent_prediction_tensor_bytes: int = 0
    optimizer_history_tensor_bytes: int = 0
    camera_ray_parameter_tensor_bytes: int = 0
    allocator_peak_measured: bool = False
    _seal: object = field(default=None, repr=False)

    @property
    def site_count(self) -> int:
        return self.material_state.site_count

    @property
    def geometry_tensor_bytes(self) -> int:
        return _tensor_bytes(
            self.positions0_f64,
            self.velocities_f64,
            self.weight_coefficients_f64,
        )

    @property
    def total_persistent_tensor_bytes(self) -> int:
        return self.material_state.total_persistent_tensor_bytes + self.geometry_tensor_bytes

    def _geometry_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.positions0_f64, self.velocities_f64, self.weight_coefficients_f64

    def assert_current(
        self,
        provider: PaperKineticLazyProgramBundleProvider,
        artifact_store: PaperKineticCompiledCpuArtifactStore,
    ) -> None:
        if (
            self._seal is not _STATE_SEAL
            or self.provenance != STATE_PROVENANCE
            or self.runtime_status != RUNTIME_STATUS
            or not self.active
            or self.retired
            or self.poisoned
            or id(provider) != self._provider_identity
            or id(artifact_store) != self._artifact_store_identity
            or self.provider_generation_digest != provider.generation_digest
            or self.world_generation_digest != provider.world.generation_digest
            or self.sites_content_digest != provider.world.sites_content_digest
            or self.geometry_update_count < 0
            or bool(self.geometry_update_count)
            != bool(self.geometry_generation_parent_digest)
            or bool(self.geometry_update_count)
            != bool(self.last_authorization_generation_digest)
            or bool(self.geometry_update_count) != bool(self.last_step_generation_id)
            or bool(self.geometry_update_count)
            != bool(self.last_update_policy_generation_digest)
            or bool(self.geometry_update_count)
            != bool(self.cold_recompile_seal_generation_digest)
            or self.material_state.step_index != self.geometry_update_count
            or self.material_state.world_generation_digest
            != self.world_generation_digest
            or self.material_state.sites_content_digest != self.sites_content_digest
            or self.material_state.last_authorization_generation_digest
            != self.last_authorization_generation_digest
            or self.material_state.last_step_generation_id
            != self.last_step_generation_id
            or self.persistent_frame_tensor_bytes
            or self.persistent_sample_tensor_bytes
            or self.persistent_target_tensor_bytes
            or self.persistent_prediction_tensor_bytes
            or self.optimizer_history_tensor_bytes
            or self.camera_ray_parameter_tensor_bytes
            or self.allocator_peak_measured
        ):
            raise ValueError("combined fixed-camera state changed or is not active")
        provider.assert_current()
        if self.geometry_generation_id != (
            paper_kinetic_fixed_camera_provider_geometry_generation_id(provider)
        ):
            raise ValueError(
                "combined state geometry generation is foreign to the provider world"
            )
        artifact_store.report()
        self.material_state.assert_current()
        provider_geometry = (
            provider.world.sites.positions0,
            provider.world.sites.velocities,
            provider.world.sites.weight_coefficients,
        )
        if any(
            state_tensor is not provider_tensor
            for state_tensor, provider_tensor in zip(
                self._geometry_tensors(),
                provider_geometry,
                strict=True,
            )
        ):
            raise ValueError("combined state does not own the provider world tensors")
        _validate_geometry_tensors(self._geometry_tensors(), site_count=self.site_count)
        if tuple(_tensor_signature(tensor) for tensor in self._geometry_tensors()) != self.tensor_signatures:
            raise ValueError("combined geometry tensor identity/version changed")
        _require_sha256(self.geometry_generation_id, name="geometry_generation_id")
        if self.geometry_generation_parent_digest:
            _require_sha256(
                self.geometry_generation_parent_digest,
                name="geometry_generation_parent_digest",
            )
        if self.cold_recompile_seal_generation_digest:
            _require_sha256(
                self.cold_recompile_seal_generation_digest,
                name="cold_recompile_seal_generation_digest",
            )
        if self.last_update_policy_generation_digest:
            _require_sha256(
                self.last_update_policy_generation_digest,
                name="last_update_policy_generation_digest",
            )
        if self.generation_digest != _combined_state_digest(self):
            raise ValueError("combined fixed-camera state generation changed")

    def assert_retired(self) -> None:
        if (
            self._seal is not _STATE_SEAL
            or self.provenance != STATE_PROVENANCE
            or self.active
            or not self.retired
            or not self.poisoned
            or not self.material_state.poisoned
            or self.geometry_generation_id
            != paper_kinetic_fixed_camera_geometry_generation_id(
                world_generation_digest=self.world_generation_digest,
                world_sites_content_digest=self.sites_content_digest,
                world_site_count=self.site_count,
            )
            or self.generation_digest != _combined_state_digest(self)
        ):
            raise ValueError("retired combined state changed")

    def accounting(self, *, requested_frame_count: int) -> dict[str, Any]:
        _require_positive_int(requested_frame_count, name="requested_frame_count")
        return {
            "provenance": self.provenance,
            "runtime_status": self.runtime_status,
            "site_count": self.site_count,
            "geometry_update_count": self.geometry_update_count,
            "last_update_policy_generation_digest": (
                self.last_update_policy_generation_digest
            ),
            "material_state_tensor_bytes": self.material_state.total_persistent_tensor_bytes,
            "geometry_tensor_bytes": self.geometry_tensor_bytes,
            "total_persistent_tensor_bytes": self.total_persistent_tensor_bytes,
            "frame_dependent_parameter_bytes": 0,
            "requested_frame_count": requested_frame_count,
            "persistent_frame_tensor_bytes": 0,
            "persistent_sample_tensor_bytes": 0,
            "persistent_target_tensor_bytes": 0,
            "persistent_prediction_tensor_bytes": 0,
            "optimizer_history_tensor_bytes": 0,
            "camera_ray_parameter_tensor_bytes": 0,
            "compiled_tensor_bytes_retained_by_state": 0,
            "allocator_peak_measured": False,
        }


def prepare_paper_kinetic_fixed_camera_combined_state(
    material_state: PaperKineticFixedSiteMaterialState,
    provider: PaperKineticLazyProgramBundleProvider,
    artifact_store: PaperKineticCompiledCpuArtifactStore,
    *,
    maximum_combined_state_logical_tensor_bytes: int,
) -> PaperKineticFixedCameraCombinedState:
    """Bind an initial material state to the exact immutable provider geometry."""

    if not isinstance(material_state, PaperKineticFixedSiteMaterialState):
        raise TypeError("combined state requires a fixed-site material state")
    if not isinstance(provider, PaperKineticLazyProgramBundleProvider):
        raise TypeError("combined state requires a kinetic provider")
    if not isinstance(artifact_store, PaperKineticCompiledCpuArtifactStore):
        raise TypeError("combined state requires a bounded artifact store")
    _require_positive_int(
        maximum_combined_state_logical_tensor_bytes,
        name="maximum_combined_state_logical_tensor_bytes",
    )
    provider.assert_current()
    artifact_store.report()
    material_state.assert_current()
    if (
        material_state.world_generation_digest != provider.world.generation_digest
        or material_state.sites_content_digest != provider.world.sites_content_digest
        or material_state.site_count != provider.world.site_count
    ):
        raise ValueError("material state is foreign to the provider world")
    geometry = (
        provider.world.sites.positions0,
        provider.world.sites.velocities,
        provider.world.sites.weight_coefficients,
    )
    required = material_state.total_persistent_tensor_bytes + _tensor_bytes(*geometry)
    if required > maximum_combined_state_logical_tensor_bytes:
        raise MemoryError("combined material/geometry state exceeds its explicit bound")
    if material_state.step_index:
        raise ValueError(
            "initial combined binding requires step-zero material state; restored "
            "combined generations must use the combined checkpoint path"
        )
    provisional = PaperKineticFixedCameraCombinedState(
        material_state=material_state,
        positions0_f64=geometry[0],
        velocities_f64=geometry[1],
        weight_coefficients_f64=geometry[2],
        provider_generation_digest=provider.generation_digest,
        world_generation_digest=provider.world.generation_digest,
        sites_content_digest=provider.world.sites_content_digest,
        geometry_generation_parent_digest="",
        geometry_generation_id=(
            paper_kinetic_fixed_camera_provider_geometry_generation_id(provider)
        ),
        last_authorization_generation_digest="",
        last_step_generation_id="",
        last_update_policy_generation_digest="",
        geometry_update_count=0,
        cold_recompile_seal_generation_digest="",
        tensor_signatures=tuple(_tensor_signature(tensor) for tensor in geometry),
        generation_digest="",
        active=True,
        retired=False,
        poisoned=False,
        _provider_identity=id(provider),
        _artifact_store_identity=id(artifact_store),
        _seal=_STATE_SEAL,
    )
    provisional.generation_digest = _combined_state_digest(provisional)
    provisional.assert_current(provider, artifact_store)
    return provisional


@dataclass(frozen=True)
class PaperKineticFixedCameraCombinedUpdateReceipt:
    step_index: int
    step_generation_id: str
    authorization_generation_digest: str
    full_geometry_step_result_generation_digest: str
    full_geometry_reverse_mode: str
    policy_generation_digest: str
    material_generation_id_before: str
    material_generation_id_after: str
    geometry_generation_id_before: str
    geometry_generation_id_after: str
    old_provider_generation_digest: str
    new_provider_generation_digest: str
    old_world_generation_digest: str
    new_world_generation_digest: str
    loss: float
    raw_color_gradient_norm: float
    raw_density_gradient_norm: float
    position_gradient_norm: float
    velocity_gradient_norm: float
    weight_gradient_norm: float
    maximum_absolute_position_update: float
    maximum_absolute_velocity_update: float
    maximum_absolute_weight_update: float
    combined_state_logical_tensor_bytes: int
    update_candidate_logical_tensor_bytes: int
    authorization_logical_tensor_bytes: int
    released_authorization_logical_tensor_bytes: int
    candidate_world_geometry_clone_logical_tensor_bytes: int
    update_validation_scratch_logical_tensor_bytes_upper_bound: int
    old_candidate_authorization_logical_tensor_bytes: int
    old_store_resident_accounted_bytes_before_retirement: int
    fresh_store_resident_accounted_bytes_upper_bound: int
    transaction_tracked_logical_and_store_accounted_bytes_upper_bound: int
    transaction_tracked_policy_bound: int
    transaction_accounting_scope: str
    old_store_closed_and_emptied: bool
    old_provider_seal_revoked: bool
    old_material_state_poisoned: bool
    authorization_capability_revoked: bool
    authorization_accumulator_revoked: bool
    authorization_tensor_references_released: bool
    full_geometry_step_result_revoked: bool
    caller_retained_untracked_bytes_included: bool
    ray_updates_enabled: bool
    generation_digest: str
    provenance: str = UPDATE_PROVENANCE
    runtime_status: str = RUNTIME_STATUS
    persistent_tensor_bytes: int = 0
    compiled_tensor_bytes_retained: int = 0
    allocator_peak_measured: bool = False
    cold_compile_scratch_peak_measured: bool = False
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        if (
            self._seal is not _UPDATE_SEAL
            or self.provenance != UPDATE_PROVENANCE
            or self.runtime_status != RUNTIME_STATUS
            or self.step_index < 1
            or not self.step_generation_id.strip()
            or self.full_geometry_reverse_mode
            not in {
                STAGED_SPARSE_FULL_GEOMETRY_REVERSE,
                FUSED_DIRECT_V1_FULL_GEOMETRY_REVERSE,
            }
            or len(self.policy_generation_digest) != 64
            or self.material_generation_id_before
            == self.material_generation_id_after
            or self.geometry_generation_id_before
            == self.geometry_generation_id_after
            or self.old_provider_generation_digest
            == self.new_provider_generation_digest
            or self.old_world_generation_digest == self.new_world_generation_digest
            or not all(
                math.isfinite(value)
                for value in (
                    self.loss,
                    self.raw_color_gradient_norm,
                    self.raw_density_gradient_norm,
                    self.position_gradient_norm,
                    self.velocity_gradient_norm,
                    self.weight_gradient_norm,
                    self.maximum_absolute_position_update,
                    self.maximum_absolute_velocity_update,
                    self.maximum_absolute_weight_update,
                )
            )
            or min(
                self.loss,
                self.raw_color_gradient_norm,
                self.raw_density_gradient_norm,
                self.position_gradient_norm,
                self.velocity_gradient_norm,
                self.weight_gradient_norm,
                self.maximum_absolute_position_update,
                self.maximum_absolute_velocity_update,
                self.maximum_absolute_weight_update,
            )
            < 0.0
            or min(
                self.combined_state_logical_tensor_bytes,
                self.update_candidate_logical_tensor_bytes,
                self.authorization_logical_tensor_bytes,
                self.released_authorization_logical_tensor_bytes,
                self.candidate_world_geometry_clone_logical_tensor_bytes,
                self.update_validation_scratch_logical_tensor_bytes_upper_bound,
                self.old_candidate_authorization_logical_tensor_bytes,
            )
            < 1
            or self.update_candidate_logical_tensor_bytes
            != self.combined_state_logical_tensor_bytes
            or self.old_candidate_authorization_logical_tensor_bytes
            != self.combined_state_logical_tensor_bytes
            + self.update_candidate_logical_tensor_bytes
            + self.authorization_logical_tensor_bytes
            or self.released_authorization_logical_tensor_bytes
            != self.authorization_logical_tensor_bytes
            or self.old_store_resident_accounted_bytes_before_retirement < 0
            or self.fresh_store_resident_accounted_bytes_upper_bound < 1
            or self.transaction_tracked_policy_bound < 1
            or self.transaction_tracked_logical_and_store_accounted_bytes_upper_bound
            != self.old_candidate_authorization_logical_tensor_bytes
            + self.candidate_world_geometry_clone_logical_tensor_bytes
            + self.update_validation_scratch_logical_tensor_bytes_upper_bound
            + self.old_store_resident_accounted_bytes_before_retirement
            + self.fresh_store_resident_accounted_bytes_upper_bound
            or self.transaction_tracked_logical_and_store_accounted_bytes_upper_bound
            > self.transaction_tracked_policy_bound
            or self.transaction_accounting_scope
            != (
                "transaction-owned-state-candidate-authorization-geometry-"
                "clone-validation-scratch-plus-"
                "store-owned-accounted-entries"
            )
            or not self.old_store_closed_and_emptied
            or not self.old_provider_seal_revoked
            or not self.old_material_state_poisoned
            or not self.authorization_capability_revoked
            or not self.authorization_accumulator_revoked
            or not self.authorization_tensor_references_released
            or not self.full_geometry_step_result_revoked
            or self.caller_retained_untracked_bytes_included
            or self.ray_updates_enabled
            or self.persistent_tensor_bytes
            or self.compiled_tensor_bytes_retained
            or self.allocator_peak_measured
            or self.cold_compile_scratch_peak_measured
            or self.generation_digest != _update_receipt_digest(self)
        ):
            raise ValueError("combined update receipt changed")
        _require_sha256(
            self.policy_generation_digest,
            name="policy_generation_digest",
        )
        _require_sha256(
            self.authorization_generation_digest,
            name="authorization_generation_digest",
        )
        _require_sha256(
            self.full_geometry_step_result_generation_digest,
            name="full_geometry_step_result_generation_digest",
        )


@dataclass(frozen=True)
class PaperKineticFixedCameraColdRecompileReceipt:
    manifest_generation_digest: str
    provider_generation_digest: str
    world_generation_digest: str
    world_sites_content_digest: str
    artifact_key_chain_digest: str
    request_count: int
    track_count: int
    store_maximum_entries: int
    store_maximum_resident_accounted_bytes: int
    store_current_entry_count: int
    store_current_resident_accounted_bytes: int
    cold_compile_count: int
    cold_compiled_track_count: int
    warm_hit_count: int
    eviction_count: int
    evicted_accounted_bytes: int
    generation_digest: str
    _provider_identity: int = field(repr=False)
    _artifact_store_identity: int = field(repr=False)
    provenance: str = RECOMPILE_PROVENANCE
    runtime_status: str = RUNTIME_STATUS
    all_acquisitions_cold: bool = True
    all_artifacts_bind_fresh_world: bool = True
    full_manifest_digest_chain_bound: bool = True
    final_store_is_bounded_lru_working_set: bool = True
    compiled_tensors_retained_by_receipt: int = 0
    allocator_peak_measured: bool = False
    _seal: object = field(default=None, repr=False)

    def assert_current(
        self,
        provider: PaperKineticLazyProgramBundleProvider,
        artifact_store: PaperKineticCompiledCpuArtifactStore,
        manifest: PaperKineticFixedCameraColdRecompileManifest,
    ) -> None:
        provider.assert_current()
        manifest.assert_compatible(provider)
        report = artifact_store.report()
        if (
            self._seal is not _RECOMPILE_SEAL
            or self.provenance != RECOMPILE_PROVENANCE
            or self.runtime_status != RUNTIME_STATUS
            or id(provider) != self._provider_identity
            or id(artifact_store) != self._artifact_store_identity
            or self.manifest_generation_digest != manifest.generation_digest
            or self.provider_generation_digest != provider.generation_digest
            or self.world_generation_digest != provider.world.generation_digest
            or self.world_sites_content_digest != provider.world.sites_content_digest
            or len(self.artifact_key_chain_digest) != 64
            or self.request_count != manifest.request_count
            or self.track_count != manifest.track_count
            or self.store_maximum_entries != report.maximum_entries
            or self.store_maximum_resident_accounted_bytes
            != report.maximum_resident_accounted_bytes
            or not 1 <= self.store_current_entry_count <= self.request_count
            or self.store_current_entry_count > self.store_maximum_entries
            or self.store_current_entry_count != report.current_entry_count
            or not 0 < self.store_current_resident_accounted_bytes
            or self.store_current_resident_accounted_bytes
            > self.store_maximum_resident_accounted_bytes
            or self.store_current_resident_accounted_bytes
            != report.current_resident_accounted_bytes
            or self.cold_compile_count != manifest.request_count
            or self.cold_compile_count != report.cold_compile_count
            or self.cold_compiled_track_count != manifest.track_count
            or self.cold_compiled_track_count != report.cold_compiled_track_count
            or self.warm_hit_count
            or report.hit_count
            or report.lookup_count != manifest.request_count
            or report.miss_count != manifest.request_count
            or report.compile_attempt_count != manifest.request_count
            or report.compile_failure_count
            or report.stale_rejection_count
            or self.eviction_count != report.eviction_count
            or self.eviction_count
            != self.cold_compile_count - self.store_current_entry_count
            or self.evicted_accounted_bytes != report.evicted_accounted_bytes
            or not self.all_acquisitions_cold
            or not self.all_artifacts_bind_fresh_world
            or not self.full_manifest_digest_chain_bound
            or not self.final_store_is_bounded_lru_working_set
            or self.compiled_tensors_retained_by_receipt
            or self.allocator_peak_measured
            or self.generation_digest != _recompile_receipt_digest(self)
        ):
            raise ValueError("cold-recompile receipt changed or was already used")


@dataclass(frozen=True)
class PaperKineticFixedCameraCombinedCheckpoint:
    material_checkpoint: PaperKineticFixedSiteMaterialCheckpoint = field(repr=False)
    combined_sgd_policy: PaperKineticFixedCameraCombinedSGDPolicy
    positions0_f64_cpu: torch.Tensor = field(repr=False)
    velocities_f64_cpu: torch.Tensor = field(repr=False)
    weight_coefficients_f64_cpu: torch.Tensor = field(repr=False)
    provider_generation_digest: str
    world_generation_digest: str
    sites_content_digest: str
    initializer_generation_digest: str
    geometry_generation_parent_digest: str
    geometry_generation_id: str
    last_authorization_generation_digest: str
    last_step_generation_id: str
    last_update_policy_generation_digest: str
    geometry_update_count: int
    cold_recompile_manifest: PaperKineticFixedCameraColdRecompileManifest
    cold_recompile_seal_generation_digest: str
    positions0_content_digest: str
    velocities_content_digest: str
    weight_coefficients_content_digest: str
    live_state_logical_tensor_bytes_at_checkpoint: int
    state_checkpoint_logical_tensor_bytes: int
    state_checkpoint_payload_peak_logical_tensor_bytes: int
    generation_digest: str
    provenance: str = CHECKPOINT_PROVENANCE
    schema: str = CHECKPOINT_SCHEMA
    runtime_status: str = RUNTIME_STATUS
    persistent_frame_tensor_bytes: int = 0
    persistent_sample_tensor_bytes: int = 0
    persistent_target_tensor_bytes: int = 0
    persistent_prediction_tensor_bytes: int = 0
    optimizer_history_tensor_bytes: int = 0
    camera_ray_parameter_tensor_bytes: int = 0
    compiled_tensor_bytes: int = 0
    combined_checkpoint_restore_integrated: bool = False
    production_trainer_integrated: bool = False
    allocator_peak_measured: bool = False
    _seal: object = field(default=None, repr=False)

    @property
    def site_count(self) -> int:
        return int(self.positions0_f64_cpu.shape[0])

    @property
    def checkpoint_tensor_bytes(self) -> int:
        return self.material_checkpoint.checkpoint_tensor_bytes + _tensor_bytes(
            self.positions0_f64_cpu,
            self.velocities_f64_cpu,
            self.weight_coefficients_f64_cpu,
        )

    def assert_current(self) -> None:
        self.material_checkpoint.assert_current()
        self.combined_sgd_policy.assert_valid()
        self.cold_recompile_manifest.assert_self_consistent()
        geometry = (
            self.positions0_f64_cpu,
            self.velocities_f64_cpu,
            self.weight_coefficients_f64_cpu,
        )
        _validate_geometry_tensors(geometry, site_count=self.material_checkpoint.site_count)
        if (
            self._seal is not _CHECKPOINT_SEAL
            or self.provenance != CHECKPOINT_PROVENANCE
            or self.schema != CHECKPOINT_SCHEMA
            or self.runtime_status != RUNTIME_STATUS
            or self.provider_generation_digest == ""
            or self.world_generation_digest
            != self.material_checkpoint.world_generation_digest
            or self.sites_content_digest != self.material_checkpoint.sites_content_digest
            or self.last_authorization_generation_digest
            != self.material_checkpoint.last_authorization_generation_digest
            or self.last_step_generation_id
            != self.material_checkpoint.last_step_generation_id
            or self.last_update_policy_generation_digest
            != self.combined_sgd_policy.generation_digest
            or self.geometry_update_count != self.material_checkpoint.step_index
            or self.geometry_update_count < 1
            or not self.geometry_generation_parent_digest
            or not self.cold_recompile_seal_generation_digest
            or self.positions0_content_digest != _tensor_content_digest(geometry[0])
            or self.velocities_content_digest != _tensor_content_digest(geometry[1])
            or self.weight_coefficients_content_digest
            != _tensor_content_digest(geometry[2])
            or self.checkpoint_tensor_bytes
            > self.combined_sgd_policy.maximum_checkpoint_logical_tensor_bytes
            or self.live_state_logical_tensor_bytes_at_checkpoint < 1
            or self.state_checkpoint_logical_tensor_bytes
            != self.live_state_logical_tensor_bytes_at_checkpoint
            + self.checkpoint_tensor_bytes
            or self.state_checkpoint_logical_tensor_bytes
            > self.combined_sgd_policy.maximum_state_checkpoint_logical_tensor_bytes
            or self.state_checkpoint_payload_peak_logical_tensor_bytes
            != self.live_state_logical_tensor_bytes_at_checkpoint
            + 2 * self.checkpoint_tensor_bytes
            or self.state_checkpoint_payload_peak_logical_tensor_bytes
            > self.combined_sgd_policy.maximum_state_checkpoint_payload_logical_tensor_bytes
            or self.persistent_frame_tensor_bytes
            or self.persistent_sample_tensor_bytes
            or self.persistent_target_tensor_bytes
            or self.persistent_prediction_tensor_bytes
            or self.optimizer_history_tensor_bytes
            or self.camera_ray_parameter_tensor_bytes
            or self.compiled_tensor_bytes
            or self.combined_checkpoint_restore_integrated
            or self.production_trainer_integrated
            or self.allocator_peak_measured
            or self.generation_digest != _combined_checkpoint_digest(self)
        ):
            raise ValueError("combined fixed-camera checkpoint changed")
        for name, digest in (
            ("provider_generation_digest", self.provider_generation_digest),
            ("world_generation_digest", self.world_generation_digest),
            ("sites_content_digest", self.sites_content_digest),
            ("initializer_generation_digest", self.initializer_generation_digest),
            (
                "geometry_generation_parent_digest",
                self.geometry_generation_parent_digest,
            ),
            ("geometry_generation_id", self.geometry_generation_id),
            (
                "last_authorization_generation_digest",
                self.last_authorization_generation_digest,
            ),
            (
                "last_update_policy_generation_digest",
                self.last_update_policy_generation_digest,
            ),
            (
                "cold_recompile_seal_generation_digest",
                self.cold_recompile_seal_generation_digest,
            ),
        ):
            _require_sha256(digest, name=name)
        if self.geometry_generation_id != (
            paper_kinetic_fixed_camera_geometry_generation_id(
                world_generation_digest=self.world_generation_digest,
                world_sites_content_digest=self.sites_content_digest,
                world_site_count=self.site_count,
            )
        ):
            raise ValueError(
                "combined checkpoint geometry generation is foreign to its world"
            )

    def payload(self) -> dict[str, Any]:
        self.assert_current()
        return {
            "schema": self.schema,
            "provenance": self.provenance,
            "runtime_status": self.runtime_status,
            "material_checkpoint": self.material_checkpoint.payload(),
            "combined_sgd_policy": self.combined_sgd_policy.payload(),
            "positions0_f64_cpu": self.positions0_f64_cpu.clone(),
            "velocities_f64_cpu": self.velocities_f64_cpu.clone(),
            "weight_coefficients_f64_cpu": self.weight_coefficients_f64_cpu.clone(),
            "provider_generation_digest": self.provider_generation_digest,
            "world_generation_digest": self.world_generation_digest,
            "sites_content_digest": self.sites_content_digest,
            "initializer_generation_digest": self.initializer_generation_digest,
            "geometry_generation_parent_digest": self.geometry_generation_parent_digest,
            "geometry_generation_id": self.geometry_generation_id,
            "last_authorization_generation_digest": (
                self.last_authorization_generation_digest
            ),
            "last_step_generation_id": self.last_step_generation_id,
            "last_update_policy_generation_digest": (
                self.last_update_policy_generation_digest
            ),
            "geometry_update_count": self.geometry_update_count,
            "cold_recompile_manifest": self.cold_recompile_manifest.payload(),
            "cold_recompile_seal_generation_digest": (
                self.cold_recompile_seal_generation_digest
            ),
            "positions0_content_digest": self.positions0_content_digest,
            "velocities_content_digest": self.velocities_content_digest,
            "weight_coefficients_content_digest": (
                self.weight_coefficients_content_digest
            ),
            "live_state_logical_tensor_bytes_at_checkpoint": (
                self.live_state_logical_tensor_bytes_at_checkpoint
            ),
            "state_checkpoint_logical_tensor_bytes": (
                self.state_checkpoint_logical_tensor_bytes
            ),
            "state_checkpoint_payload_peak_logical_tensor_bytes": (
                self.state_checkpoint_payload_peak_logical_tensor_bytes
            ),
            "generation_digest": self.generation_digest,
            "persistent_frame_tensor_bytes": 0,
            "persistent_sample_tensor_bytes": 0,
            "persistent_target_tensor_bytes": 0,
            "persistent_prediction_tensor_bytes": 0,
            "optimizer_history_tensor_bytes": 0,
            "camera_ray_parameter_tensor_bytes": 0,
            "compiled_tensor_bytes": 0,
            "combined_checkpoint_restore_integrated": False,
            "production_trainer_integrated": False,
            "allocator_peak_measured": False,
        }


@dataclass(frozen=True)
class PaperKineticFixedCameraCombinedRestoreReceipt:
    """Tensor-free proof that a checkpoint became one fresh live generation."""

    checkpoint_generation_digest: str
    policy_generation_digest: str
    dataset_generation_digest: str
    target_residency_digest: str
    camera_grid_digest: str
    factory_generation_digest: str
    provider_generation_digest: str
    world_generation_digest: str
    sites_content_digest: str
    initializer_generation_digest: str
    geometry_generation_id: str
    material_generation_id: str
    manifest_generation_digest: str
    checkpoint_cold_recompile_seal_generation_digest: str
    restored_cold_recompile_seal_generation_digest: str
    source_payload_tensor_bytes: int
    checkpoint_tensor_bytes: int
    live_state_logical_tensor_bytes: int
    state_checkpoint_logical_tensor_bytes: int
    state_checkpoint_payload_peak_logical_tensor_bytes: int
    state_checkpoint_payload_policy_bound: int
    fresh_store_maximum_resident_accounted_bytes: int
    restore_tracked_logical_and_store_accounted_bytes_upper_bound: int
    restore_tracked_policy_bound: int
    generation_digest: str
    provenance: str = RESTORE_PROVENANCE
    runtime_status: str = RUNTIME_STATUS
    persistent_tensor_bytes: int = 0
    compiled_tensor_bytes_retained: int = 0
    allocator_peak_measured: bool = False
    _seal: object = field(default=None, repr=False)

    def assert_current(
        self,
        state: PaperKineticFixedCameraCombinedState,
        provider: PaperKineticLazyProgramBundleProvider,
        artifact_store: PaperKineticCompiledCpuArtifactStore,
        manifest: PaperKineticFixedCameraColdRecompileManifest,
        recompile_receipt: PaperKineticFixedCameraColdRecompileReceipt,
    ) -> None:
        state.assert_current(provider, artifact_store)
        recompile_receipt.assert_current(provider, artifact_store, manifest)
        report = artifact_store.report()
        for name, digest in (
            ("checkpoint_generation_digest", self.checkpoint_generation_digest),
            ("policy_generation_digest", self.policy_generation_digest),
            ("dataset_generation_digest", self.dataset_generation_digest),
            ("target_residency_digest", self.target_residency_digest),
            ("camera_grid_digest", self.camera_grid_digest),
            ("factory_generation_digest", self.factory_generation_digest),
            ("provider_generation_digest", self.provider_generation_digest),
            ("world_generation_digest", self.world_generation_digest),
            ("sites_content_digest", self.sites_content_digest),
            ("initializer_generation_digest", self.initializer_generation_digest),
            ("geometry_generation_id", self.geometry_generation_id),
            ("material_generation_id", self.material_generation_id),
            ("manifest_generation_digest", self.manifest_generation_digest),
            (
                "checkpoint_cold_recompile_seal_generation_digest",
                self.checkpoint_cold_recompile_seal_generation_digest,
            ),
            (
                "restored_cold_recompile_seal_generation_digest",
                self.restored_cold_recompile_seal_generation_digest,
            ),
        ):
            _require_sha256(digest, name=name)
        if (
            self._seal is not _RESTORE_SEAL
            or self.provenance != RESTORE_PROVENANCE
            or self.runtime_status != RUNTIME_STATUS
            or self.dataset_generation_digest
            != provider.dataset_generation_digest
            or self.dataset_generation_digest
            != manifest.dataset_generation_digest
            or self.target_residency_digest != provider.target_residency_digest
            or self.camera_grid_digest != provider.camera_grid_digest
            or self.camera_grid_digest != manifest.camera_grid_digest
            or self.factory_generation_digest
            != provider.factory_generation_digest
            or self.factory_generation_digest
            != manifest.factory_generation_digest
            or self.provider_generation_digest != provider.generation_digest
            or self.world_generation_digest != provider.world.generation_digest
            or self.sites_content_digest != provider.world.sites_content_digest
            or self.initializer_generation_digest
            != provider.initializer_generation_digest
            or self.geometry_generation_id != state.geometry_generation_id
            or self.material_generation_id
            != state.material_state.material_generation_id
            or self.manifest_generation_digest != manifest.generation_digest
            or self.checkpoint_cold_recompile_seal_generation_digest
            != self.restored_cold_recompile_seal_generation_digest
            or self.restored_cold_recompile_seal_generation_digest
            != recompile_receipt.generation_digest
            or self.source_payload_tensor_bytes < 0
            or min(
                self.checkpoint_tensor_bytes,
                self.live_state_logical_tensor_bytes,
                self.state_checkpoint_logical_tensor_bytes,
                self.state_checkpoint_payload_peak_logical_tensor_bytes,
                self.state_checkpoint_payload_policy_bound,
                self.fresh_store_maximum_resident_accounted_bytes,
                self.restore_tracked_logical_and_store_accounted_bytes_upper_bound,
                self.restore_tracked_policy_bound,
            )
            < 1
            or self.live_state_logical_tensor_bytes
            != state.total_persistent_tensor_bytes
            or self.state_checkpoint_logical_tensor_bytes
            != self.live_state_logical_tensor_bytes + self.checkpoint_tensor_bytes
            or self.state_checkpoint_payload_peak_logical_tensor_bytes
            != self.live_state_logical_tensor_bytes
            + 2 * self.checkpoint_tensor_bytes
            or self.state_checkpoint_payload_peak_logical_tensor_bytes
            > self.state_checkpoint_payload_policy_bound
            or self.source_payload_tensor_bytes
            + self.state_checkpoint_logical_tensor_bytes
            > self.state_checkpoint_payload_policy_bound
            or self.fresh_store_maximum_resident_accounted_bytes
            != report.maximum_resident_accounted_bytes
            or self.restore_tracked_logical_and_store_accounted_bytes_upper_bound
            != self.source_payload_tensor_bytes
            + self.state_checkpoint_logical_tensor_bytes
            + self.fresh_store_maximum_resident_accounted_bytes
            or self.restore_tracked_logical_and_store_accounted_bytes_upper_bound
            > self.restore_tracked_policy_bound
            or self.persistent_tensor_bytes
            or self.compiled_tensor_bytes_retained
            or self.allocator_peak_measured
            or self.generation_digest != _restore_receipt_digest(self)
        ):
            raise ValueError("combined checkpoint restore receipt changed")


@dataclass
class PaperKineticFixedCameraRestoredReadyGeneration:
    """Fresh-process ready generation reconstructed from a sealed checkpoint."""

    state: PaperKineticFixedCameraCombinedState = field(repr=False)
    provider: PaperKineticLazyProgramBundleProvider = field(repr=False)
    artifact_store: PaperKineticCompiledCpuArtifactStore = field(repr=False)
    restore_receipt: PaperKineticFixedCameraCombinedRestoreReceipt
    recompile_receipt: PaperKineticFixedCameraColdRecompileReceipt
    manifest: PaperKineticFixedCameraColdRecompileManifest
    generation_digest: str
    next_step_claimed: bool
    provenance: str = RESTORED_READY_PROVENANCE
    runtime_status: str = RUNTIME_STATUS
    _claim_lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
    )
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        if (
            self._seal is not _RESTORED_READY_SEAL
            or self.provenance != RESTORED_READY_PROVENANCE
            or self.runtime_status != RUNTIME_STATUS
            or self.next_step_claimed
            or not isinstance(self._claim_lock, _LOCK_TYPE)
        ):
            raise ValueError(
                "restored ready generation was changed or already claimed"
            )
        self.restore_receipt.assert_current(
            self.state,
            self.provider,
            self.artifact_store,
            self.manifest,
            self.recompile_receipt,
        )
        if (
            self.state.cold_recompile_seal_generation_digest
            != self.recompile_receipt.generation_digest
            or self.generation_digest != _restored_ready_generation_digest(self)
        ):
            raise ValueError("restored ready generation components disagree")


@dataclass
class PaperKineticFixedCameraReadyGeneration:
    """Fresh live generation; checkpoints are created explicitly after promotion."""

    state: PaperKineticFixedCameraCombinedState = field(repr=False)
    provider: PaperKineticLazyProgramBundleProvider = field(repr=False)
    artifact_store: PaperKineticCompiledCpuArtifactStore = field(repr=False)
    update_receipt: PaperKineticFixedCameraCombinedUpdateReceipt
    recompile_receipt: PaperKineticFixedCameraColdRecompileReceipt
    manifest: PaperKineticFixedCameraColdRecompileManifest
    generation_digest: str
    next_step_claimed: bool
    provenance: str = READY_PROVENANCE
    runtime_status: str = RUNTIME_STATUS
    _claim_lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
    )
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        if (
            self._seal is not _READY_SEAL
            or self.provenance != READY_PROVENANCE
            or self.runtime_status != RUNTIME_STATUS
            or self.next_step_claimed
            or not isinstance(self._claim_lock, _LOCK_TYPE)
        ):
            raise ValueError("ready generation was changed or already claimed")
        self.state.assert_current(self.provider, self.artifact_store)
        self.update_receipt.assert_current()
        self.recompile_receipt.assert_current(
            self.provider,
            self.artifact_store,
            self.manifest,
        )
        if (
            self.state.cold_recompile_seal_generation_digest
            != self.recompile_receipt.generation_digest
            or self.update_receipt.authorization_generation_digest
            != self.state.last_authorization_generation_digest
            or self.update_receipt.step_generation_id
            != self.state.last_step_generation_id
            or self.update_receipt.policy_generation_digest
            != self.state.last_update_policy_generation_digest
            or self.update_receipt.material_generation_id_after
            != self.state.material_state.material_generation_id
            or self.update_receipt.geometry_generation_id_after
            != self.state.geometry_generation_id
            or self.update_receipt.new_provider_generation_digest
            != self.state.provider_generation_digest
            or self.update_receipt.new_world_generation_digest
            != self.state.world_generation_digest
            or self.generation_digest != _ready_generation_digest(self)
        ):
            raise ValueError("ready generation components disagree")


class PaperKineticFixedCameraCombinedTransactionFailure(RuntimeError):
    """The old generation was retired and candidate promotion failed closed."""

    def __init__(self, stage: str, cause: BaseException) -> None:
        super().__init__(
            "combined geometry transaction failed after old-generation "
            f"retirement at {stage}; restart is required: "
            f"{type(cause).__qualname__}: {cause}"
        )
        self.stage = stage
        self.restart_required = True
        self.old_generation_unusable = True
        self.candidate_generation_unusable = True


@dataclass(frozen=True)
class _AuthorizationTransactionSnapshot:
    generation_digest: str
    full_geometry_step_result_generation_digest: str
    full_geometry_reverse_mode: str
    step_generation_id: str
    logical_tensor_bytes: int
    loss: float


@dataclass(frozen=True)
class _TransactionMemoryPreflight:
    combined_state_logical_tensor_bytes: int
    update_candidate_logical_tensor_bytes: int
    authorization_logical_tensor_bytes: int
    old_candidate_authorization_logical_tensor_bytes: int
    candidate_world_geometry_clone_logical_tensor_bytes: int
    update_validation_scratch_logical_tensor_bytes_upper_bound: int
    old_store_resident_accounted_bytes: int
    fresh_store_resident_accounted_bytes_upper_bound: int
    transaction_tracked_logical_and_store_accounted_bytes_upper_bound: int


@torch.no_grad()
def apply_paper_kinetic_fixed_camera_combined_sgd_transaction(
    state: PaperKineticFixedCameraCombinedState,
    current_provider: PaperKineticLazyProgramBundleProvider,
    current_artifact_store: PaperKineticCompiledCpuArtifactStore,
    step_result: PaperKineticFixedCameraFullGeometryStepResult,
    *,
    policy: PaperKineticFixedCameraCombinedSGDPolicy,
    cold_recompile_manifest: PaperKineticFixedCameraColdRecompileManifest,
    fresh_store_policy: PaperKineticCompiledCpuArtifactStorePolicy | None = None,
) -> PaperKineticFixedCameraReadyGeneration:
    """Consume one full-geometry authorization and promote only after recompile."""

    if not isinstance(state, PaperKineticFixedCameraCombinedState):
        raise TypeError("combined update requires its combined state")
    if not isinstance(step_result, PaperKineticFixedCameraFullGeometryStepResult):
        raise TypeError("combined update requires a fixed-camera full-geometry result")
    if not isinstance(
        cold_recompile_manifest,
        PaperKineticFixedCameraColdRecompileManifest,
    ):
        raise TypeError("combined update requires a cold-recompile manifest")
    if not isinstance(
        current_artifact_store,
        PaperKineticCompiledCpuArtifactStore,
    ):
        raise TypeError("combined update requires its bounded artifact store")
    policy.assert_valid()
    authorization = step_result.authorization
    accumulator = step_result.accumulator
    replay_receipt = step_result.replay_receipt
    if not isinstance(accumulator, PaperKineticDenseStepGradientAccumulator):
        raise TypeError("combined update requires a full-geometry accumulator")
    resolved_store_policy = (
        current_artifact_store.policy
        if fresh_store_policy is None
        else fresh_store_policy
    )
    if not isinstance(
        resolved_store_policy,
        PaperKineticCompiledCpuArtifactStorePolicy,
    ):
        raise TypeError("combined update requires an explicit fresh-store policy")
    old_store_resident_accounted_bytes = (
        current_artifact_store.report().current_resident_accounted_bytes
    )
    memory_preflight = _preflight_transaction(
        state,
        accumulator,
        policy=policy,
        manifest=cold_recompile_manifest,
        store_policy=resolved_store_policy,
        old_store_resident_accounted_bytes=(
            old_store_resident_accounted_bytes
        ),
    )
    state.assert_current(current_provider, current_artifact_store)
    if (
        state.geometry_update_count
        and state.last_update_policy_generation_digest != policy.generation_digest
    ):
        raise ValueError(
            "combined update policy changed; start a deliberate fresh generation"
        )
    step_result.assert_current()
    cold_recompile_manifest.assert_compatible(current_provider)
    _validate_authorization(
        state,
        current_provider,
        authorization,
        accumulator,
        replay_receipt,
        step_result=step_result,
    )
    authorization_snapshot = _AuthorizationTransactionSnapshot(
        generation_digest=authorization.generation_digest,
        full_geometry_step_result_generation_digest=step_result.generation_digest,
        full_geometry_reverse_mode=str(
            step_result.accounting["full_geometry_reverse_mode"]
        ),
        step_generation_id=authorization.step_generation_id,
        logical_tensor_bytes=accumulator.logical_tensor_bytes,
        loss=float(authorization.loss_f32.item()),
    )
    if (
        authorization_snapshot.logical_tensor_bytes
        != memory_preflight.authorization_logical_tensor_bytes
    ):
        raise ArithmeticError("authorization changed its preflighted tensor layout")

    old_material_generation = state.material_state.material_generation_id
    old_geometry_generation = state.geometry_generation_id
    old_provider_generation = current_provider.generation_digest
    old_world_generation = current_provider.world.generation_digest
    candidates = _build_update_candidates(state, authorization, policy=policy)
    if (
        candidates.logical_tensor_bytes
        != memory_preflight.update_candidate_logical_tensor_bytes
    ):
        raise ArithmeticError(
            "combined update candidate changed its preflighted layout"
        )
    initializer_generation_digest = _digest_parts(
        UPDATE_PROVENANCE,
        "owned-candidate-world-initializer",
        state.geometry_generation_id,
        authorization_snapshot.generation_digest,
        authorization_snapshot.step_generation_id,
        _tensor_content_digest(candidates.positions0_f64),
        _tensor_content_digest(candidates.velocities_f64),
        _tensor_content_digest(candidates.weight_coefficients_f64),
    )
    initializer = _OwnedCandidateWorldInitializer(
        sites=AffineKineticPowerSites(
            positions0=candidates.positions0_f64,
            velocities=candidates.velocities_f64,
            weight_coefficients=candidates.weight_coefficients_f64,
        ),
        generation_digest=initializer_generation_digest,
    )
    fresh_provider = prepare_paper_kinetic_lazy_program_bundle_provider(
        dataset_generation_digest=current_provider.dataset_generation_digest,
        target_provider=current_provider.target_provider,
        ray_provider=current_provider.ray_provider,
        frame_times=current_provider.frame_times,
        height=current_provider.height,
        width=current_provider.width,
        maximum_tracks_per_bundle=current_provider.maximum_tracks_per_bundle,
        maximum_observations_per_bundle=(
            current_provider.maximum_observations_per_bundle
        ),
        maximum_rows_per_native_block=current_provider.maximum_rows_per_native_block,
        world_initializer=initializer,
        program_factory=current_provider.program_factory,
    )
    if not initializer.consumed:
        raise ArithmeticError("fresh provider did not consume candidate geometry")
    if (
        _tensor_bytes(
            fresh_provider.world.sites.positions0,
            fresh_provider.world.sites.velocities,
            fresh_provider.world.sites.weight_coefficients,
        )
        != memory_preflight.candidate_world_geometry_clone_logical_tensor_bytes
    ):
        raise ArithmeticError("candidate world clone changed its preflighted layout")
    cold_recompile_manifest.assert_compatible(fresh_provider)
    geometry_generation_after = (
        paper_kinetic_fixed_camera_provider_geometry_generation_id(fresh_provider)
    )
    fresh_material_state = _build_fresh_material_state(
        state.material_state,
        fresh_provider,
        authorization,
        candidates,
    )
    fresh_store = PaperKineticCompiledCpuArtifactStore(resolved_store_policy)

    # Irreversible commit point.  The outer fail-closed guard starts before the
    # first retirement mutation, so even a close/revocation failure poisons both
    # generations and yields one explicit restart-required exception.
    stage = "retire_old_generation"
    new_state: PaperKineticFixedCameraCombinedState | None = None
    try:
        _retire_combined_generation(
            state,
            current_provider,
            current_artifact_store,
        )
        stage = "consume_full_geometry_authorization"
        released_authorization_logical_tensor_bytes = (
            _consume_full_geometry_step_result(step_result)
        )
        if (
            released_authorization_logical_tensor_bytes
            != authorization_snapshot.logical_tensor_bytes
        ):
            raise ArithmeticError(
                "consumed authorization changed its preflighted tensor layout"
            )
        stage = "cold_recompile"
        recompile_receipt = _cold_recompile_and_seal(
            fresh_provider,
            fresh_store,
            cold_recompile_manifest,
            maximum_artifact_accounted_bytes=(
                policy.maximum_artifact_accounted_bytes
            ),
        )
        new_state = PaperKineticFixedCameraCombinedState(
            material_state=fresh_material_state,
            positions0_f64=fresh_provider.world.sites.positions0,
            velocities_f64=fresh_provider.world.sites.velocities,
            weight_coefficients_f64=(
                fresh_provider.world.sites.weight_coefficients
            ),
            provider_generation_digest=fresh_provider.generation_digest,
            world_generation_digest=fresh_provider.world.generation_digest,
            sites_content_digest=fresh_provider.world.sites_content_digest,
            geometry_generation_parent_digest=old_geometry_generation,
            geometry_generation_id=geometry_generation_after,
            last_authorization_generation_digest=(
                authorization_snapshot.generation_digest
            ),
            last_step_generation_id=authorization_snapshot.step_generation_id,
            last_update_policy_generation_digest=policy.generation_digest,
            geometry_update_count=state.geometry_update_count + 1,
            cold_recompile_seal_generation_digest=(
                recompile_receipt.generation_digest
            ),
            tensor_signatures=tuple(
                _tensor_signature(tensor)
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
            _seal=_STATE_SEAL,
        )
        new_state.generation_digest = _combined_state_digest(new_state)
        new_state.assert_current(fresh_provider, fresh_store)
        stage = "seal_update_receipt"
        update_receipt = _seal_update_receipt(
            state,
            new_state,
            authorization_snapshot,
            candidates,
            policy=policy,
            memory_preflight=memory_preflight,
            released_authorization_logical_tensor_bytes=(
                released_authorization_logical_tensor_bytes
            ),
            old_material_generation=old_material_generation,
            old_geometry_generation=old_geometry_generation,
            old_provider_generation=old_provider_generation,
            old_world_generation=old_world_generation,
        )
        stage = "seal_ready_generation"
        provisional = PaperKineticFixedCameraReadyGeneration(
            state=new_state,
            provider=fresh_provider,
            artifact_store=fresh_store,
            update_receipt=update_receipt,
            recompile_receipt=recompile_receipt,
            manifest=cold_recompile_manifest,
            generation_digest="",
            next_step_claimed=False,
            _seal=_READY_SEAL,
        )
        provisional.generation_digest = _ready_generation_digest(provisional)
        provisional.assert_current()
        return provisional
    except BaseException as error:
        cleanup_errors = _invalidate_candidate_generation(
            fresh_material_state,
            fresh_provider,
            fresh_store,
            combined_state=new_state,
        )
        failure = PaperKineticFixedCameraCombinedTransactionFailure(stage, error)
        for cleanup_error in cleanup_errors:
            failure.add_note(cleanup_error)
        raise failure from error


def claim_paper_kinetic_fixed_camera_ready_generation_for_next_step(
    ready: PaperKineticFixedCameraReadyGeneration,
    *,
    caller_retained_untracked_logical_and_accounted_bytes: int,
) -> PaperKineticFixedCameraFullGeometryStepState:
    """Consume one ready seal after attesting no untracked caller roots remain."""

    if not isinstance(ready, PaperKineticFixedCameraReadyGeneration):
        raise TypeError("next-step claim requires a ready combined generation")
    _require_nonnegative_int(
        caller_retained_untracked_logical_and_accounted_bytes,
        name="caller_retained_untracked_logical_and_accounted_bytes",
    )
    acquired = ready._claim_lock.acquire(blocking=False)
    if not acquired:
        raise RuntimeError("ready generation next-step claim is already active")
    try:
        ready.assert_current()
        if caller_retained_untracked_logical_and_accounted_bytes:
            raise MemoryError(
                "next-step claim requires zero caller-retained artifact/checkpoint/"
                "retired-generation logical/accounted bytes"
            )
        coordinator_state = (
            prepare_paper_kinetic_fixed_camera_full_geometry_step_state(
                ready.provider,
                ready.artifact_store,
                device=ready.state.material_state.device,
                resume_material_state=ready.state.material_state,
            )
        )
        ready.next_step_claimed = True
        return coordinator_state
    finally:
        ready._claim_lock.release()


@torch.no_grad()
def checkpoint_paper_kinetic_fixed_camera_combined_state(
    state: PaperKineticFixedCameraCombinedState,
    provider: PaperKineticLazyProgramBundleProvider,
    artifact_store: PaperKineticCompiledCpuArtifactStore,
    *,
    manifest: PaperKineticFixedCameraColdRecompileManifest,
    recompile_receipt: PaperKineticFixedCameraColdRecompileReceipt,
    policy: PaperKineticFixedCameraCombinedSGDPolicy,
    initializer_generation_digest: str,
) -> PaperKineticFixedCameraCombinedCheckpoint:
    """Create a raw-only checkpoint; no sampler/program/store object is retained."""

    state.assert_current(provider, artifact_store)
    recompile_receipt.assert_current(provider, artifact_store, manifest)
    policy.assert_valid()
    if state.last_update_policy_generation_digest != policy.generation_digest:
        raise ValueError("combined checkpoint policy is foreign to the live state")
    _require_sha256(
        initializer_generation_digest,
        name="initializer_generation_digest",
    )
    if initializer_generation_digest != provider.initializer_generation_digest:
        raise ValueError("combined checkpoint initializer is foreign to its provider")
    required = _combined_checkpoint_tensor_bytes(state)
    state_and_checkpoint = state.total_persistent_tensor_bytes + required
    payload_peak = state.total_persistent_tensor_bytes + 2 * required
    if required > policy.maximum_checkpoint_logical_tensor_bytes:
        raise MemoryError("combined checkpoint exceeds its explicit tensor-byte bound")
    if state_and_checkpoint > policy.maximum_state_checkpoint_logical_tensor_bytes:
        raise MemoryError("live state plus checkpoint exceeds its explicit bound")
    if payload_peak > policy.maximum_state_checkpoint_payload_logical_tensor_bytes:
        raise MemoryError(
            "live state plus checkpoint payload clone exceeds its explicit bound"
        )
    material_checkpoint = checkpoint_paper_kinetic_fixed_site_material_state(
        state.material_state
    )
    positions0 = state.positions0_f64.detach().clone().contiguous()
    velocities = state.velocities_f64.detach().clone().contiguous()
    weights = state.weight_coefficients_f64.detach().clone().contiguous()
    provisional = PaperKineticFixedCameraCombinedCheckpoint(
        material_checkpoint=material_checkpoint,
        combined_sgd_policy=policy,
        positions0_f64_cpu=positions0,
        velocities_f64_cpu=velocities,
        weight_coefficients_f64_cpu=weights,
        provider_generation_digest=provider.generation_digest,
        world_generation_digest=provider.world.generation_digest,
        sites_content_digest=provider.world.sites_content_digest,
        initializer_generation_digest=initializer_generation_digest,
        geometry_generation_parent_digest=state.geometry_generation_parent_digest,
        geometry_generation_id=state.geometry_generation_id,
        last_authorization_generation_digest=(
            state.last_authorization_generation_digest
        ),
        last_step_generation_id=state.last_step_generation_id,
        last_update_policy_generation_digest=(
            state.last_update_policy_generation_digest
        ),
        geometry_update_count=state.geometry_update_count,
        cold_recompile_manifest=manifest,
        cold_recompile_seal_generation_digest=recompile_receipt.generation_digest,
        positions0_content_digest=_tensor_content_digest(positions0),
        velocities_content_digest=_tensor_content_digest(velocities),
        weight_coefficients_content_digest=_tensor_content_digest(weights),
        live_state_logical_tensor_bytes_at_checkpoint=(
            state.total_persistent_tensor_bytes
        ),
        state_checkpoint_logical_tensor_bytes=state_and_checkpoint,
        state_checkpoint_payload_peak_logical_tensor_bytes=payload_peak,
        generation_digest="",
        _seal=_CHECKPOINT_SEAL,
    )
    result = replace(
        provisional,
        generation_digest=_combined_checkpoint_digest(provisional),
    )
    result.assert_current()
    if result.checkpoint_tensor_bytes != required:
        raise ArithmeticError("combined checkpoint changed its preflighted layout")
    return result


def paper_kinetic_fixed_camera_combined_checkpoint_from_payload(
    payload: Mapping[str, Any],
    *,
    expected_world_site_count: int,
    expected_combined_sgd_policy: PaperKineticFixedCameraCombinedSGDPolicy,
) -> PaperKineticFixedCameraCombinedCheckpoint:
    """Parse one raw-only restart payload under a caller-owned policy bound.

    The serialized policy is provenance, not authority: restart accepts it only
    when it exactly matches the policy supplied by the current configuration.
    This prevents a payload from raising its own tensor or recompile caps.
    """

    _require_positive_int(
        expected_world_site_count,
        name="expected_world_site_count",
    )
    if not isinstance(
        expected_combined_sgd_policy,
        PaperKineticFixedCameraCombinedSGDPolicy,
    ):
        raise TypeError("combined checkpoint restart requires an expected SGD policy")
    expected_combined_sgd_policy.assert_valid()
    required = {
        "schema",
        "provenance",
        "runtime_status",
        "material_checkpoint",
        "combined_sgd_policy",
        "positions0_f64_cpu",
        "velocities_f64_cpu",
        "weight_coefficients_f64_cpu",
        "provider_generation_digest",
        "world_generation_digest",
        "sites_content_digest",
        "initializer_generation_digest",
        "geometry_generation_parent_digest",
        "geometry_generation_id",
        "last_authorization_generation_digest",
        "last_step_generation_id",
        "last_update_policy_generation_digest",
        "geometry_update_count",
        "cold_recompile_manifest",
        "cold_recompile_seal_generation_digest",
        "positions0_content_digest",
        "velocities_content_digest",
        "weight_coefficients_content_digest",
        "live_state_logical_tensor_bytes_at_checkpoint",
        "state_checkpoint_logical_tensor_bytes",
        "state_checkpoint_payload_peak_logical_tensor_bytes",
        "generation_digest",
        "persistent_frame_tensor_bytes",
        "persistent_sample_tensor_bytes",
        "persistent_target_tensor_bytes",
        "persistent_prediction_tensor_bytes",
        "optimizer_history_tensor_bytes",
        "camera_ray_parameter_tensor_bytes",
        "compiled_tensor_bytes",
        "combined_checkpoint_restore_integrated",
        "production_trainer_integrated",
        "allocator_peak_measured",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("combined fixed-camera checkpoint payload keys changed")
    if (
        payload["schema"] != CHECKPOINT_SCHEMA
        or payload["provenance"] != CHECKPOINT_PROVENANCE
        or payload["runtime_status"] != RUNTIME_STATUS
    ):
        raise ValueError("combined checkpoint schema/provenance changed")
    for name in (
        "provider_generation_digest",
        "world_generation_digest",
        "sites_content_digest",
        "initializer_generation_digest",
        "geometry_generation_parent_digest",
        "geometry_generation_id",
        "last_authorization_generation_digest",
        "last_update_policy_generation_digest",
        "cold_recompile_seal_generation_digest",
        "positions0_content_digest",
        "velocities_content_digest",
        "weight_coefficients_content_digest",
        "generation_digest",
    ):
        _require_sha256(payload[name], name=name)
    if (
        not isinstance(payload["last_step_generation_id"], str)
        or not payload["last_step_generation_id"].strip()
    ):
        raise ValueError("combined checkpoint last_step_generation_id changed")
    parsed_policy = _combined_sgd_policy_from_payload(payload["combined_sgd_policy"])
    if (
        parsed_policy != expected_combined_sgd_policy
        or parsed_policy.generation_digest
        != expected_combined_sgd_policy.generation_digest
    ):
        raise ValueError("combined checkpoint policy differs from the restart policy")
    for name in (
        "geometry_update_count",
        "live_state_logical_tensor_bytes_at_checkpoint",
        "state_checkpoint_logical_tensor_bytes",
        "state_checkpoint_payload_peak_logical_tensor_bytes",
    ):
        _require_positive_int(payload[name], name=name)
    for name in (
        "persistent_frame_tensor_bytes",
        "persistent_sample_tensor_bytes",
        "persistent_target_tensor_bytes",
        "persistent_prediction_tensor_bytes",
        "optimizer_history_tensor_bytes",
        "camera_ray_parameter_tensor_bytes",
        "compiled_tensor_bytes",
    ):
        _require_nonnegative_int(payload[name], name=name)
        if payload[name] != 0:
            raise ValueError("combined checkpoint retained forbidden runtime tensors")
    for name in (
        "combined_checkpoint_restore_integrated",
        "production_trainer_integrated",
        "allocator_peak_measured",
    ):
        if payload[name] is not False:
            raise ValueError(f"combined checkpoint {name} must be exactly false")

    geometry_payload = (
        payload["positions0_f64_cpu"],
        payload["velocities_f64_cpu"],
        payload["weight_coefficients_f64_cpu"],
    )
    _validate_geometry_tensor_metadata(
        geometry_payload,
        site_count=expected_world_site_count,
    )
    material_payload = payload["material_checkpoint"]
    if not isinstance(material_payload, Mapping):
        raise ValueError("combined checkpoint material payload changed")
    material_payload_tensors = (
        material_payload.get("raw_color_f32_cpu"),
        material_payload.get("raw_density_f32_cpu"),
    )
    _validate_checkpoint_material_tensor_metadata(
        material_payload_tensors,
        site_count=expected_world_site_count,
    )
    weight_coefficient_count = int(geometry_payload[2].shape[1])
    geometry_checkpoint_bytes = (
        8 * expected_world_site_count * (6 + weight_coefficient_count)
    )
    checkpoint_bytes = 16 * expected_world_site_count + geometry_checkpoint_bytes
    live_state_bytes = 48 * expected_world_site_count + geometry_checkpoint_bytes
    state_checkpoint_bytes = live_state_bytes + checkpoint_bytes
    payload_peak_bytes = live_state_bytes + 2 * checkpoint_bytes
    if (
        payload["live_state_logical_tensor_bytes_at_checkpoint"]
        != live_state_bytes
        or payload["state_checkpoint_logical_tensor_bytes"]
        != state_checkpoint_bytes
        or payload["state_checkpoint_payload_peak_logical_tensor_bytes"]
        != payload_peak_bytes
    ):
        raise ValueError("combined checkpoint serialized byte accounting changed")
    if (
        checkpoint_bytes
        > expected_combined_sgd_policy.maximum_checkpoint_logical_tensor_bytes
        or state_checkpoint_bytes
        > expected_combined_sgd_policy.maximum_state_checkpoint_logical_tensor_bytes
        or payload_peak_bytes
        > expected_combined_sgd_policy.maximum_state_checkpoint_payload_logical_tensor_bytes
    ):
        raise MemoryError("combined checkpoint payload exceeds the restart policy")

    # Parse the tensor-free bounded work manifest before any full-tensor scan
    # or clone. Its per-request track tuple is capped independently of the
    # persistent interval descriptors.
    manifest = paper_kinetic_fixed_camera_cold_recompile_manifest_from_payload(
        payload["cold_recompile_manifest"],
        maximum_request_count=(
            expected_combined_sgd_policy.maximum_recompile_request_count
        ),
        maximum_track_id_logical_bytes=(
            expected_combined_sgd_policy.maximum_recompile_track_id_logical_bytes
        ),
    )

    # Include source backing-storage size in the pre-clone gate. A contiguous
    # view over oversized storage cannot evade either the checkpoint cap or the
    # payload-plus-owned-clone coexistence cap.
    source_checkpoint_storage_bytes = _tensor_bytes(
        *geometry_payload,
        *material_payload_tensors,
    )
    if (
        source_checkpoint_storage_bytes
        > expected_combined_sgd_policy.maximum_checkpoint_logical_tensor_bytes
        or source_checkpoint_storage_bytes + checkpoint_bytes
        > expected_combined_sgd_policy.maximum_state_checkpoint_payload_logical_tensor_bytes
    ):
        raise MemoryError(
            "combined checkpoint source storage exceeds the restart coexistence policy"
        )

    _validate_geometry_tensors(
        geometry_payload,
        site_count=expected_world_site_count,
    )
    material_checkpoint = paper_kinetic_fixed_site_material_checkpoint_from_payload(
        material_payload,
        expected_world_site_count=expected_world_site_count,
        maximum_checkpoint_logical_tensor_bytes=(
            expected_combined_sgd_policy.maximum_checkpoint_logical_tensor_bytes
        ),
    )
    result = PaperKineticFixedCameraCombinedCheckpoint(
        material_checkpoint=material_checkpoint,
        combined_sgd_policy=parsed_policy,
        positions0_f64_cpu=geometry_payload[0].detach().clone().contiguous(),
        velocities_f64_cpu=geometry_payload[1].detach().clone().contiguous(),
        weight_coefficients_f64_cpu=geometry_payload[2].detach().clone().contiguous(),
        provider_generation_digest=payload["provider_generation_digest"],
        world_generation_digest=payload["world_generation_digest"],
        sites_content_digest=payload["sites_content_digest"],
        initializer_generation_digest=payload["initializer_generation_digest"],
        geometry_generation_parent_digest=payload[
            "geometry_generation_parent_digest"
        ],
        geometry_generation_id=payload["geometry_generation_id"],
        last_authorization_generation_digest=payload[
            "last_authorization_generation_digest"
        ],
        last_step_generation_id=payload["last_step_generation_id"],
        last_update_policy_generation_digest=payload[
            "last_update_policy_generation_digest"
        ],
        geometry_update_count=payload["geometry_update_count"],
        cold_recompile_manifest=manifest,
        cold_recompile_seal_generation_digest=payload[
            "cold_recompile_seal_generation_digest"
        ],
        positions0_content_digest=payload["positions0_content_digest"],
        velocities_content_digest=payload["velocities_content_digest"],
        weight_coefficients_content_digest=payload[
            "weight_coefficients_content_digest"
        ],
        live_state_logical_tensor_bytes_at_checkpoint=payload[
            "live_state_logical_tensor_bytes_at_checkpoint"
        ],
        state_checkpoint_logical_tensor_bytes=payload[
            "state_checkpoint_logical_tensor_bytes"
        ],
        state_checkpoint_payload_peak_logical_tensor_bytes=payload[
            "state_checkpoint_payload_peak_logical_tensor_bytes"
        ],
        generation_digest=payload["generation_digest"],
        provenance=payload["provenance"],
        schema=payload["schema"],
        runtime_status=payload["runtime_status"],
        persistent_frame_tensor_bytes=payload["persistent_frame_tensor_bytes"],
        persistent_sample_tensor_bytes=payload["persistent_sample_tensor_bytes"],
        persistent_target_tensor_bytes=payload["persistent_target_tensor_bytes"],
        persistent_prediction_tensor_bytes=payload[
            "persistent_prediction_tensor_bytes"
        ],
        optimizer_history_tensor_bytes=payload["optimizer_history_tensor_bytes"],
        camera_ray_parameter_tensor_bytes=payload["camera_ray_parameter_tensor_bytes"],
        compiled_tensor_bytes=payload["compiled_tensor_bytes"],
        combined_checkpoint_restore_integrated=payload[
            "combined_checkpoint_restore_integrated"
        ],
        production_trainer_integrated=payload["production_trainer_integrated"],
        allocator_peak_measured=payload["allocator_peak_measured"],
        _seal=_CHECKPOINT_SEAL,
    )
    result.assert_current()
    return result


@torch.no_grad()
def restore_paper_kinetic_fixed_camera_combined_generation(
    checkpoint: PaperKineticFixedCameraCombinedCheckpoint,
    *,
    target_provider: PowerFoamTargetProvider,
    ray_provider: PowerFoamRayProvider,
    frame_times: Sequence[float] | torch.Tensor,
    maximum_observations_per_bundle: int,
    maximum_rows_per_native_block: int,
    program_factory: PaperKineticTrackProgramFactory,
    fresh_store_policy: PaperKineticCompiledCpuArtifactStorePolicy,
    device: torch.device | str = "cpu",
) -> PaperKineticFixedCameraRestoredReadyGeneration:
    """Restore one sealed checkpoint after its serialized payload is released."""

    return _restore_paper_kinetic_fixed_camera_combined_generation(
        checkpoint,
        target_provider=target_provider,
        ray_provider=ray_provider,
        frame_times=frame_times,
        maximum_observations_per_bundle=maximum_observations_per_bundle,
        maximum_rows_per_native_block=maximum_rows_per_native_block,
        program_factory=program_factory,
        fresh_store_policy=fresh_store_policy,
        device=device,
        source_payload_tensor_bytes=0,
    )


@torch.no_grad()
def restore_paper_kinetic_fixed_camera_combined_generation_from_payload(
    payload: Mapping[str, Any],
    *,
    expected_world_site_count: int,
    expected_combined_sgd_policy: PaperKineticFixedCameraCombinedSGDPolicy,
    target_provider: PowerFoamTargetProvider,
    ray_provider: PowerFoamRayProvider,
    frame_times: Sequence[float] | torch.Tensor,
    maximum_observations_per_bundle: int,
    maximum_rows_per_native_block: int,
    program_factory: PaperKineticTrackProgramFactory,
    fresh_store_policy: PaperKineticCompiledCpuArtifactStorePolicy,
    device: torch.device | str = "cpu",
) -> PaperKineticFixedCameraRestoredReadyGeneration:
    """Validate a raw payload, clone it once, and restore under the same caps.

    The receipt counts the caller-owned source payload together with the parsed
    checkpoint, reconstructed live state, and fresh bounded store.  The caller
    must still release its payload before claiming the returned generation for
    the next optimizer step.
    """

    checkpoint = paper_kinetic_fixed_camera_combined_checkpoint_from_payload(
        payload,
        expected_world_site_count=expected_world_site_count,
        expected_combined_sgd_policy=expected_combined_sgd_policy,
    )
    material_payload = payload["material_checkpoint"]
    source_payload_tensor_bytes = _tensor_bytes(
        payload["positions0_f64_cpu"],
        payload["velocities_f64_cpu"],
        payload["weight_coefficients_f64_cpu"],
        material_payload["raw_color_f32_cpu"],
        material_payload["raw_density_f32_cpu"],
    )
    return _restore_paper_kinetic_fixed_camera_combined_generation(
        checkpoint,
        target_provider=target_provider,
        ray_provider=ray_provider,
        frame_times=frame_times,
        maximum_observations_per_bundle=maximum_observations_per_bundle,
        maximum_rows_per_native_block=maximum_rows_per_native_block,
        program_factory=program_factory,
        fresh_store_policy=fresh_store_policy,
        device=device,
        source_payload_tensor_bytes=source_payload_tensor_bytes,
    )


@torch.no_grad()
def _restore_paper_kinetic_fixed_camera_combined_generation(
    checkpoint: PaperKineticFixedCameraCombinedCheckpoint,
    *,
    target_provider: PowerFoamTargetProvider,
    ray_provider: PowerFoamRayProvider,
    frame_times: Sequence[float] | torch.Tensor,
    maximum_observations_per_bundle: int,
    maximum_rows_per_native_block: int,
    program_factory: PaperKineticTrackProgramFactory,
    fresh_store_policy: PaperKineticCompiledCpuArtifactStorePolicy,
    device: torch.device | str,
    source_payload_tensor_bytes: int,
) -> PaperKineticFixedCameraRestoredReadyGeneration:
    if not isinstance(checkpoint, PaperKineticFixedCameraCombinedCheckpoint):
        raise TypeError("combined restart requires a sealed combined checkpoint")
    if not isinstance(
        fresh_store_policy,
        PaperKineticCompiledCpuArtifactStorePolicy,
    ):
        raise TypeError("combined restart requires a bounded artifact-store policy")
    _require_nonnegative_int(
        source_payload_tensor_bytes,
        name="source_payload_tensor_bytes",
    )
    checkpoint.assert_current()
    policy = checkpoint.combined_sgd_policy
    manifest = checkpoint.cold_recompile_manifest
    if (
        manifest.request_count > policy.maximum_recompile_request_count
        or manifest.track_id_logical_bytes
        > policy.maximum_recompile_track_id_logical_bytes
        or 8 * manifest.track_count
        > policy.maximum_recompile_track_id_logical_bytes
    ):
        raise MemoryError("combined restart manifest exceeds its policy")
    if (
        checkpoint.live_state_logical_tensor_bytes_at_checkpoint
        > policy.maximum_combined_state_logical_tensor_bytes
    ):
        raise MemoryError("combined restart live state exceeds its policy")
    payload_state_coexistence = (
        source_payload_tensor_bytes
        + checkpoint.state_checkpoint_logical_tensor_bytes
    )
    if (
        payload_state_coexistence
        > policy.maximum_state_checkpoint_payload_logical_tensor_bytes
    ):
        raise MemoryError(
            "combined restart payload/checkpoint/live-state coexistence exceeds policy"
        )
    tracked_upper_bound = (
        payload_state_coexistence
        + fresh_store_policy.maximum_resident_accounted_bytes
    )
    if tracked_upper_bound > (
        policy.maximum_transaction_tracked_logical_and_store_accounted_bytes
    ):
        raise MemoryError(
            "combined restart tracked state/store peak exceeds its policy"
        )

    geometry = tuple(
        tensor.detach().clone().contiguous()
        for tensor in (
            checkpoint.positions0_f64_cpu,
            checkpoint.velocities_f64_cpu,
            checkpoint.weight_coefficients_f64_cpu,
        )
    )
    _validate_geometry_tensors(geometry, site_count=checkpoint.site_count)
    initializer = _OwnedCandidateWorldInitializer(
        sites=AffineKineticPowerSites(
            positions0=geometry[0],
            velocities=geometry[1],
            weight_coefficients=geometry[2],
        ),
        generation_digest=checkpoint.initializer_generation_digest,
    )
    fresh_provider: PaperKineticLazyProgramBundleProvider | None = None
    fresh_material_state: PaperKineticFixedSiteMaterialState | None = None
    fresh_store: PaperKineticCompiledCpuArtifactStore | None = None
    restored_state: PaperKineticFixedCameraCombinedState | None = None
    try:
        fresh_provider = prepare_paper_kinetic_lazy_program_bundle_provider(
            dataset_generation_digest=manifest.dataset_generation_digest,
            target_provider=target_provider,
            ray_provider=ray_provider,
            frame_times=frame_times,
            height=manifest.height,
            width=manifest.width,
            maximum_tracks_per_bundle=manifest.maximum_tracks_per_bundle,
            maximum_observations_per_bundle=maximum_observations_per_bundle,
            maximum_rows_per_native_block=maximum_rows_per_native_block,
            world_initializer=initializer,
            program_factory=program_factory,
        )
        if not initializer.consumed:
            raise ArithmeticError("combined restart initializer was not consumed")
        manifest.assert_compatible(fresh_provider)
        if (
            fresh_provider.generation_digest
            != checkpoint.provider_generation_digest
            or fresh_provider.world.generation_digest
            != checkpoint.world_generation_digest
            or fresh_provider.world.sites_content_digest
            != checkpoint.sites_content_digest
            or fresh_provider.initializer_generation_digest
            != checkpoint.initializer_generation_digest
            or paper_kinetic_fixed_camera_provider_geometry_generation_id(
                fresh_provider
            )
            != checkpoint.geometry_generation_id
        ):
            raise ValueError(
                "combined restart runtime inputs do not reconstruct the checkpoint generation"
            )
        fresh_material_state = restore_paper_kinetic_fixed_site_material_state(
            checkpoint.material_checkpoint,
            world=fresh_provider.world,
            device=device,
            maximum_material_state_logical_tensor_bytes=(
                policy.maximum_combined_state_logical_tensor_bytes
            ),
        )
        fresh_store = PaperKineticCompiledCpuArtifactStore(fresh_store_policy)
        recompile_receipt = _cold_recompile_and_seal(
            fresh_provider,
            fresh_store,
            manifest,
            maximum_artifact_accounted_bytes=(
                policy.maximum_artifact_accounted_bytes
            ),
        )
        if (
            recompile_receipt.generation_digest
            != checkpoint.cold_recompile_seal_generation_digest
        ):
            raise ValueError(
                "combined restart cold-recompile seal differs from the checkpoint"
            )
        restored_state = PaperKineticFixedCameraCombinedState(
            material_state=fresh_material_state,
            positions0_f64=fresh_provider.world.sites.positions0,
            velocities_f64=fresh_provider.world.sites.velocities,
            weight_coefficients_f64=(
                fresh_provider.world.sites.weight_coefficients
            ),
            provider_generation_digest=fresh_provider.generation_digest,
            world_generation_digest=fresh_provider.world.generation_digest,
            sites_content_digest=fresh_provider.world.sites_content_digest,
            geometry_generation_parent_digest=(
                checkpoint.geometry_generation_parent_digest
            ),
            geometry_generation_id=checkpoint.geometry_generation_id,
            last_authorization_generation_digest=(
                checkpoint.last_authorization_generation_digest
            ),
            last_step_generation_id=checkpoint.last_step_generation_id,
            last_update_policy_generation_digest=(
                checkpoint.last_update_policy_generation_digest
            ),
            geometry_update_count=checkpoint.geometry_update_count,
            cold_recompile_seal_generation_digest=(
                recompile_receipt.generation_digest
            ),
            tensor_signatures=tuple(
                _tensor_signature(tensor)
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
            _seal=_STATE_SEAL,
        )
        restored_state.generation_digest = _combined_state_digest(restored_state)
        restored_state.assert_current(fresh_provider, fresh_store)
        if (
            restored_state.total_persistent_tensor_bytes
            != checkpoint.live_state_logical_tensor_bytes_at_checkpoint
        ):
            raise ArithmeticError(
                "combined restart changed the checkpointed live-state layout"
            )
        provisional_receipt = PaperKineticFixedCameraCombinedRestoreReceipt(
            checkpoint_generation_digest=checkpoint.generation_digest,
            policy_generation_digest=policy.generation_digest,
            dataset_generation_digest=fresh_provider.dataset_generation_digest,
            target_residency_digest=fresh_provider.target_residency_digest,
            camera_grid_digest=fresh_provider.camera_grid_digest,
            factory_generation_digest=fresh_provider.factory_generation_digest,
            provider_generation_digest=fresh_provider.generation_digest,
            world_generation_digest=fresh_provider.world.generation_digest,
            sites_content_digest=fresh_provider.world.sites_content_digest,
            initializer_generation_digest=(
                fresh_provider.initializer_generation_digest
            ),
            geometry_generation_id=restored_state.geometry_generation_id,
            material_generation_id=(
                restored_state.material_state.material_generation_id
            ),
            manifest_generation_digest=manifest.generation_digest,
            checkpoint_cold_recompile_seal_generation_digest=(
                checkpoint.cold_recompile_seal_generation_digest
            ),
            restored_cold_recompile_seal_generation_digest=(
                recompile_receipt.generation_digest
            ),
            source_payload_tensor_bytes=source_payload_tensor_bytes,
            checkpoint_tensor_bytes=checkpoint.checkpoint_tensor_bytes,
            live_state_logical_tensor_bytes=(
                restored_state.total_persistent_tensor_bytes
            ),
            state_checkpoint_logical_tensor_bytes=(
                checkpoint.state_checkpoint_logical_tensor_bytes
            ),
            state_checkpoint_payload_peak_logical_tensor_bytes=(
                checkpoint.state_checkpoint_payload_peak_logical_tensor_bytes
            ),
            state_checkpoint_payload_policy_bound=(
                policy.maximum_state_checkpoint_payload_logical_tensor_bytes
            ),
            fresh_store_maximum_resident_accounted_bytes=(
                fresh_store_policy.maximum_resident_accounted_bytes
            ),
            restore_tracked_logical_and_store_accounted_bytes_upper_bound=(
                tracked_upper_bound
            ),
            restore_tracked_policy_bound=(
                policy.maximum_transaction_tracked_logical_and_store_accounted_bytes
            ),
            generation_digest="",
            _seal=_RESTORE_SEAL,
        )
        restore_receipt = replace(
            provisional_receipt,
            generation_digest=_restore_receipt_digest(provisional_receipt),
        )
        restore_receipt.assert_current(
            restored_state,
            fresh_provider,
            fresh_store,
            manifest,
            recompile_receipt,
        )
        restored_ready = PaperKineticFixedCameraRestoredReadyGeneration(
            state=restored_state,
            provider=fresh_provider,
            artifact_store=fresh_store,
            restore_receipt=restore_receipt,
            recompile_receipt=recompile_receipt,
            manifest=manifest,
            generation_digest="",
            next_step_claimed=False,
            _seal=_RESTORED_READY_SEAL,
        )
        restored_ready.generation_digest = _restored_ready_generation_digest(
            restored_ready
        )
        restored_ready.assert_current()
        return restored_ready
    except BaseException as error:
        cleanup_notes: tuple[str, ...] = ()
        if (
            fresh_provider is not None
            and fresh_material_state is not None
            and fresh_store is not None
        ):
            cleanup_notes = _invalidate_candidate_generation(
                fresh_material_state,
                fresh_provider,
                fresh_store,
                combined_state=restored_state,
            )
        else:
            failures: list[str] = []
            if fresh_store is not None:
                try:
                    fresh_store.close()
                except BaseException as cleanup_error:
                    failures.append(
                        "restore artifact-store close failed: "
                        f"{type(cleanup_error).__qualname__}: {cleanup_error}"
                    )
            if fresh_provider is not None:
                try:
                    object.__setattr__(fresh_provider, "_seal", None)
                except BaseException as cleanup_error:
                    failures.append(
                        "restore provider revocation failed: "
                        f"{type(cleanup_error).__qualname__}: {cleanup_error}"
                    )
            if fresh_material_state is not None:
                fresh_material_state.poisoned = True
            cleanup_notes = tuple(failures)
        for note in cleanup_notes:
            error.add_note(note)
        raise


def claim_paper_kinetic_fixed_camera_restored_ready_generation_for_next_step(
    ready: PaperKineticFixedCameraRestoredReadyGeneration,
    *,
    caller_retained_untracked_logical_and_accounted_bytes: int,
) -> PaperKineticFixedCameraFullGeometryStepState:
    """Consume a restored ready seal after checkpoint/payload roots are gone."""

    if not isinstance(ready, PaperKineticFixedCameraRestoredReadyGeneration):
        raise TypeError("restored next-step claim requires a restored ready generation")
    _require_nonnegative_int(
        caller_retained_untracked_logical_and_accounted_bytes,
        name="caller_retained_untracked_logical_and_accounted_bytes",
    )
    acquired = ready._claim_lock.acquire(blocking=False)
    if not acquired:
        raise RuntimeError("restored ready generation next-step claim is already active")
    try:
        ready.assert_current()
        if caller_retained_untracked_logical_and_accounted_bytes:
            raise MemoryError(
                "restored next-step claim requires zero caller-retained payload/"
                "checkpoint/artifact logical/accounted bytes"
            )
        coordinator_state = (
            prepare_paper_kinetic_fixed_camera_full_geometry_step_state(
                ready.provider,
                ready.artifact_store,
                device=ready.state.material_state.device,
                resume_material_state=ready.state.material_state,
            )
        )
        ready.next_step_claimed = True
        return coordinator_state
    finally:
        ready._claim_lock.release()


def claim_paper_kinetic_fixed_camera_restored_ready_generation_for_lazy_native_next_step(
    ready: PaperKineticFixedCameraRestoredReadyGeneration,
    *,
    caller_retained_untracked_logical_and_accounted_bytes: int,
    device: torch.device | str,
) -> Any:
    """Consume a restored generation into the lazy-native trainer state."""

    if not isinstance(ready, PaperKineticFixedCameraRestoredReadyGeneration):
        raise TypeError(
            "lazy restart claim requires a restored ready combined generation"
        )
    _require_nonnegative_int(
        caller_retained_untracked_logical_and_accounted_bytes,
        name="caller_retained_untracked_logical_and_accounted_bytes",
    )
    acquired = ready._claim_lock.acquire(blocking=False)
    if not acquired:
        raise RuntimeError("restored generation lazy next-step claim is already active")
    try:
        ready.assert_current()
        if caller_retained_untracked_logical_and_accounted_bytes:
            raise MemoryError(
                "restored lazy next-step claim requires zero caller-retained "
                "payload/checkpoint/artifact logical or accounted bytes"
            )
        from kinetic_lazy_native_material_step import (
            prepare_paper_kinetic_lazy_native_trainer_state,
        )

        trainer_state = prepare_paper_kinetic_lazy_native_trainer_state(
            ready.provider,
            device=device,
            initial_step_index=ready.state.geometry_update_count,
        )
        ready.next_step_claimed = True
        return trainer_state
    finally:
        ready._claim_lock.release()


@dataclass
class _UpdateCandidates:
    raw_color_f32: torch.Tensor
    raw_density_f32: torch.Tensor
    site_rgba_f32: torch.Tensor
    raw_color_grad_f32: torch.Tensor
    raw_density_grad_f32: torch.Tensor
    positions0_f64: torch.Tensor
    velocities_f64: torch.Tensor
    weight_coefficients_f64: torch.Tensor
    raw_color_gradient_norm: float
    raw_density_gradient_norm: float
    position_gradient_norm: float
    velocity_gradient_norm: float
    weight_gradient_norm: float
    maximum_absolute_position_update: float
    maximum_absolute_velocity_update: float
    maximum_absolute_weight_update: float

    @property
    def logical_tensor_bytes(self) -> int:
        return _tensor_bytes(
            self.raw_color_f32,
            self.raw_density_f32,
            self.site_rgba_f32,
            self.raw_color_grad_f32,
            self.raw_density_grad_f32,
            self.positions0_f64,
            self.velocities_f64,
            self.weight_coefficients_f64,
        )


@torch.no_grad()
def _build_update_candidates(
    state: PaperKineticFixedCameraCombinedState,
    authorization: PaperKineticDenseOptimizerAuthorization,
    *,
    policy: PaperKineticFixedCameraCombinedSGDPolicy,
    learning_rate_multiplier: float = 1.0,
) -> _UpdateCandidates:
    if (
        not math.isfinite(learning_rate_multiplier)
        or learning_rate_multiplier <= 0.0
    ):
        raise ValueError("learning_rate_multiplier must be finite and positive")
    material = state.material_state
    runtime = material.parameterization.runtime_parameterization
    color_grad = torch.empty_like(material.raw_color_f32)
    density_grad = torch.empty_like(material.raw_density_f32)
    runtime.color_vjp_(
        color_grad,
        material.site_rgba_f32[:, :3],
        authorization.grad_site_rgba_f32[:, :3],
    )
    runtime.density_vjp_(
        density_grad,
        material.raw_density_f32,
        authorization.grad_site_rgba_f32[:, 3],
    )
    raw_color_gradient_norm = float(torch.linalg.vector_norm(color_grad).item())
    raw_density_gradient_norm = float(torch.linalg.vector_norm(density_grad).item())
    raw_color = material.raw_color_f32.clone().add_(
        color_grad,
        alpha=(
            -material.optimizer_policy.color_learning_rate
            * learning_rate_multiplier
        ),
    )
    raw_density = material.raw_density_f32.clone().add_(
        density_grad,
        alpha=(
            -material.optimizer_policy.density_learning_rate
            * learning_rate_multiplier
        ),
    )
    site_rgba = torch.empty_like(material.site_rgba_f32)
    _material_state._decode_physical_(
        site_rgba,
        raw_color,
        raw_density,
        parameterization=material.parameterization,
    )
    positions0 = torch.add(
        state.positions0_f64,
        authorization.grad_positions0_f64,
        alpha=-policy.position_learning_rate * learning_rate_multiplier,
    )
    velocities = torch.add(
        state.velocities_f64,
        authorization.grad_velocities_f64,
        alpha=-policy.velocity_learning_rate * learning_rate_multiplier,
    )
    weights = torch.add(
        state.weight_coefficients_f64,
        authorization.grad_weight_coefficients_f64,
        alpha=-policy.weight_learning_rate * learning_rate_multiplier,
    )
    maxima = tuple(
        learning_rate
        * float(torch.linalg.vector_norm(bar.reshape(-1), ord=float("inf")).item())
        for bar, learning_rate in zip(
            (
                authorization.grad_positions0_f64,
                authorization.grad_velocities_f64,
                authorization.grad_weight_coefficients_f64,
            ),
            (
                policy.position_learning_rate * learning_rate_multiplier,
                policy.velocity_learning_rate * learning_rate_multiplier,
                policy.weight_learning_rate * learning_rate_multiplier,
            ),
            strict=True,
        )
    )
    if (
        maxima[0] > policy.maximum_absolute_position_update
        or maxima[1] > policy.maximum_absolute_velocity_update
        or maxima[2] > policy.maximum_absolute_weight_update
    ):
        raise ValueError("combined geometry candidate exceeds its update bound")
    if (
        float(positions0.abs().max().item())
        > policy.maximum_absolute_position_value
        or float(velocities.abs().max().item())
        > policy.maximum_absolute_velocity_value
        or float(weights.abs().max().item()) > policy.maximum_absolute_weight_value
    ):
        raise ValueError("combined geometry candidate exceeds its value bound")
    if (
        float(raw_color.abs().max().item())
        > material.optimizer_policy.maximum_absolute_raw_color_value
        or float(raw_density.abs().max().item())
        > material.optimizer_policy.maximum_absolute_raw_density_value
    ):
        raise ValueError("combined material candidate exceeds its raw-value bound")
    tensors = (
        color_grad,
        density_grad,
        raw_color,
        raw_density,
        site_rgba,
        positions0,
        velocities,
        weights,
    )
    if not all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
        raise FloatingPointError("combined material/geometry candidate is non-finite")
    _require_distinct_storage(
        *state._geometry_tensors(),
        *state.material_state._tensors(),
        *authorization._tensors(),
        *tensors,
    )
    position_gradient_norm = float(
        torch.linalg.vector_norm(authorization.grad_positions0_f64).item()
    )
    velocity_gradient_norm = float(
        torch.linalg.vector_norm(authorization.grad_velocities_f64).item()
    )
    weight_gradient_norm = float(
        torch.linalg.vector_norm(
            authorization.grad_weight_coefficients_f64
        ).item()
    )
    color_grad.zero_()
    density_grad.zero_()
    return _UpdateCandidates(
        raw_color_f32=raw_color,
        raw_density_f32=raw_density,
        site_rgba_f32=site_rgba,
        raw_color_grad_f32=color_grad,
        raw_density_grad_f32=density_grad,
        positions0_f64=positions0.contiguous(),
        velocities_f64=velocities.contiguous(),
        weight_coefficients_f64=weights.contiguous(),
        raw_color_gradient_norm=raw_color_gradient_norm,
        raw_density_gradient_norm=raw_density_gradient_norm,
        position_gradient_norm=position_gradient_norm,
        velocity_gradient_norm=velocity_gradient_norm,
        weight_gradient_norm=weight_gradient_norm,
        maximum_absolute_position_update=maxima[0],
        maximum_absolute_velocity_update=maxima[1],
        maximum_absolute_weight_update=maxima[2],
    )


def _build_fresh_material_state(
    previous: PaperKineticFixedSiteMaterialState,
    provider: PaperKineticLazyProgramBundleProvider,
    authorization: PaperKineticDenseOptimizerAuthorization,
    candidates: _UpdateCandidates,
) -> PaperKineticFixedSiteMaterialState:
    provisional = PaperKineticFixedSiteMaterialState(
        world_generation_digest=provider.world.generation_digest,
        sites_content_digest=provider.world.sites_content_digest,
        p0_material_seed_generation_digest=(
            previous.p0_material_seed_generation_digest
        ),
        parameterization=previous.parameterization,
        optimizer_policy=previous.optimizer_policy,
        raw_color_f32=candidates.raw_color_f32,
        raw_density_f32=candidates.raw_density_f32,
        site_rgba_f32=candidates.site_rgba_f32,
        raw_color_grad_f32=candidates.raw_color_grad_f32,
        raw_density_grad_f32=candidates.raw_density_grad_f32,
        initialization_content_digest=previous.initialization_content_digest,
        generation_parent_digest=previous.material_generation_id,
        last_authorization_generation_digest=authorization.generation_digest,
        last_step_generation_id=authorization.step_generation_id,
        step_index=previous.step_index + 1,
        material_generation_id="",
        restart_checkpoint_generation_digest="",
        tensor_signatures=(),
        poisoned=False,
        _seal=_material_state._STATE_SEAL,
    )
    provisional.material_generation_id = _material_state._state_generation_digest(
        provisional
    )
    provisional.tensor_signatures = tuple(
        _material_state._tensor_signature(tensor)
        for tensor in provisional._tensors()
    )
    provisional.assert_current()
    return provisional


def _validate_authorization(
    state: PaperKineticFixedCameraCombinedState,
    provider: PaperKineticLazyProgramBundleProvider,
    authorization: PaperKineticDenseOptimizerAuthorization,
    accumulator: PaperKineticDenseStepGradientAccumulator,
    replay_receipt: PaperKineticDenseObservationReplayReceipt,
    *,
    step_result: PaperKineticFixedCameraFullGeometryStepResult,
) -> None:
    authorization.assert_current(accumulator, replay_receipt)
    if (
        not authorization.full_geometry
        or not accumulator.full_geometry
        or authorization.optimize_camera_rays
        or accumulator.optimize_camera_rays
        or authorization.ray_bar_keys
        or accumulator.ray_bar_keys
        or authorization.grad_track_ray_coefficients_f64 is not None
        or accumulator.grad_track_ray_coefficients_f64 is not None
        or accumulator._material_tensor_ref is not state.material_state.site_rgba_f32
        or accumulator.material_generation_id
        != state.material_state.material_generation_id
        or accumulator.world_generation_digest != provider.world.generation_digest
        or accumulator.world_sites_content_digest
        != provider.world.sites_content_digest
        or accumulator.site_table_identity != id(provider.world.sites)
        or step_result.accounting.get("provider_generation_digest")
        != provider.generation_digest
        or step_result.accounting.get("provider_identity") != id(provider)
        or step_result.accounting.get("geometry_generation_id")
        != state.geometry_generation_id
        or authorization.generation_digest
        == state.last_authorization_generation_digest
        or authorization.step_generation_id == state.last_step_generation_id
    ):
        raise ValueError("combined updater received a stale/foreign/ray authorization")
    _material_state._validate_authorized_material_bars(
        state.material_state,
        authorization,
    )
    geometry_bars = (
        authorization.grad_positions0_f64,
        authorization.grad_velocities_f64,
        authorization.grad_weight_coefficients_f64,
    )
    if any(bar is None for bar in geometry_bars):
        raise ValueError("combined updater lost fixed-camera geometry bars")
    for name, bar, expected in zip(
        ("positions0", "velocities", "weight_coefficients"),
        geometry_bars,
        state._geometry_tensors(),
        strict=True,
    ):
        if (
            bar.dtype != torch.float64
            or bar.device.type != "cpu"
            or tuple(bar.shape) != tuple(expected.shape)
            or not bar.is_contiguous()
            or bar.requires_grad
            or not bool(torch.isfinite(bar).all().item())
        ):
            raise ValueError(f"authorized {name} bar is invalid")
    _require_distinct_storage(
        *state._geometry_tensors(),
        *geometry_bars,
        *state.material_state._tensors(),
        authorization.grad_site_rgba_f32,
        authorization.loss_f32,
    )


def _preflight_transaction(
    state: PaperKineticFixedCameraCombinedState,
    accumulator: PaperKineticDenseStepGradientAccumulator,
    *,
    policy: PaperKineticFixedCameraCombinedSGDPolicy,
    manifest: PaperKineticFixedCameraColdRecompileManifest,
    store_policy: PaperKineticCompiledCpuArtifactStorePolicy,
    old_store_resident_accounted_bytes: int,
) -> _TransactionMemoryPreflight:
    tracked_tensors = (
        *state._geometry_tensors(),
        *state.material_state._tensors(),
        *accumulator._tensors(),
    )
    if not tracked_tensors or any(
        not isinstance(tensor, torch.Tensor) for tensor in tracked_tensors
    ):
        raise ValueError("transaction preflight lost a tracked tensor")
    combined = state.total_persistent_tensor_bytes
    candidate = combined
    authorization = accumulator.logical_tensor_bytes
    overlap = combined + candidate + authorization
    geometry_clone = state.geometry_tensor_bytes
    validation_scratch = (
        _update_validation_scratch_logical_tensor_bytes_upper_bound(
            state,
            accumulator,
        )
    )
    checkpoint = _combined_checkpoint_tensor_bytes(state)
    state_checkpoint = combined + checkpoint
    state_checkpoint_payload = combined + 2 * checkpoint
    transaction_tracked = (
        overlap
        + geometry_clone
        + validation_scratch
        + old_store_resident_accounted_bytes
        + store_policy.maximum_resident_accounted_bytes
    )
    if old_store_resident_accounted_bytes < 0:
        raise ArithmeticError("old artifact-store resident accounting is negative")
    if combined > policy.maximum_combined_state_logical_tensor_bytes:
        raise MemoryError("combined persistent state exceeds its explicit bound")
    if candidate > policy.maximum_update_candidate_logical_tensor_bytes:
        raise MemoryError("combined update candidate exceeds its explicit bound")
    if (
        geometry_clone
        > policy.maximum_candidate_world_geometry_clone_logical_tensor_bytes
    ):
        raise MemoryError(
            "candidate world geometry clone exceeds its explicit bound"
        )
    if (
        validation_scratch
        > policy.maximum_update_validation_scratch_logical_tensor_bytes
    ):
        raise MemoryError("update validation scratch exceeds its explicit bound")
    if overlap > policy.maximum_old_candidate_authorization_logical_tensor_bytes:
        raise MemoryError("combined update overlap exceeds its explicit bound")
    if checkpoint > policy.maximum_checkpoint_logical_tensor_bytes:
        raise MemoryError("combined checkpoint exceeds its explicit bound")
    if state_checkpoint > policy.maximum_state_checkpoint_logical_tensor_bytes:
        raise MemoryError("live state plus checkpoint exceeds its explicit bound")
    if (
        state_checkpoint_payload
        > policy.maximum_state_checkpoint_payload_logical_tensor_bytes
    ):
        raise MemoryError(
            "live state plus checkpoint payload clone exceeds its explicit bound"
        )
    if (
        transaction_tracked
        > policy.maximum_transaction_tracked_logical_and_store_accounted_bytes
    ):
        raise MemoryError(
            "transaction tracked state/store peak exceeds its explicit bound"
        )
    if manifest.request_count > policy.maximum_recompile_request_count:
        raise MemoryError("cold-recompile manifest exceeds its request-count bound")
    if (
        manifest.track_id_logical_bytes
        > policy.maximum_recompile_track_id_logical_bytes
        or 8 * manifest.track_count
        > policy.maximum_recompile_track_id_logical_bytes
    ):
        raise MemoryError("cold-recompile manifest exceeds its track-id byte bound")
    if (
        policy.maximum_artifact_accounted_bytes
        > store_policy.maximum_resident_accounted_bytes
    ):
        raise MemoryError(
            "fresh store cannot admit one bounded cold-recompile artifact"
        )
    return _TransactionMemoryPreflight(
        combined_state_logical_tensor_bytes=combined,
        update_candidate_logical_tensor_bytes=candidate,
        authorization_logical_tensor_bytes=authorization,
        old_candidate_authorization_logical_tensor_bytes=overlap,
        candidate_world_geometry_clone_logical_tensor_bytes=geometry_clone,
        update_validation_scratch_logical_tensor_bytes_upper_bound=(
            validation_scratch
        ),
        old_store_resident_accounted_bytes=(
            old_store_resident_accounted_bytes
        ),
        fresh_store_resident_accounted_bytes_upper_bound=(
            store_policy.maximum_resident_accounted_bytes
        ),
        transaction_tracked_logical_and_store_accounted_bytes_upper_bound=(
            transaction_tracked
        ),
    )


def _update_validation_scratch_logical_tensor_bytes_upper_bound(
    state: PaperKineticFixedCameraCombinedState,
    accumulator: PaperKineticDenseStepGradientAccumulator,
) -> int:
    """Bound one largest tensor-sized validation reduction temporary."""

    tensors = (
        *state._geometry_tensors(),
        *state.material_state._tensors(),
        *accumulator._tensors(),
    )
    return max(
        int(tensor.numel()) * int(tensor.element_size()) for tensor in tensors
    )


def _cold_recompile_and_seal(
    provider: PaperKineticLazyProgramBundleProvider,
    artifact_store: PaperKineticCompiledCpuArtifactStore,
    manifest: PaperKineticFixedCameraColdRecompileManifest,
    *,
    maximum_artifact_accounted_bytes: int,
) -> PaperKineticFixedCameraColdRecompileReceipt:
    manifest.assert_compatible(provider)
    initial = artifact_store.report()
    if any(
        value
        for value in (
            initial.current_entry_count,
            initial.lookup_count,
            initial.hit_count,
            initial.miss_count,
            initial.compile_attempt_count,
            initial.cold_compile_count,
        )
    ):
        raise ValueError("cold-recompile requires a fresh empty artifact store")
    chain = _digest_parts(
        RECOMPILE_PROVENANCE,
        "full-cold-manifest-chain",
        manifest.generation_digest,
        manifest.request_count,
        manifest.track_count,
    )
    for request_index, request in enumerate(manifest.requests):
        acquisition = artifact_store.acquire(
            provider,
            view_index=request.view_index,
            track_ids=request.track_ids,
            maximum_artifact_accounted_bytes=maximum_artifact_accounted_bytes,
            compile_artifact=lambda key: (
                compile_paper_kinetic_compiled_cpu_artifact(provider, key)
            ),
        )
        if (
            acquisition.cache_status != "cold_compiled"
            or acquisition.warm_hit
            or acquisition.avoided_compile_track_count
            or acquisition.artifact.key.provider_generation_digest
            != provider.generation_digest
            or acquisition.artifact.key.world_generation_digest
            != provider.world.generation_digest
            or acquisition.artifact.key.world_sites_content_digest
            != provider.world.sites_content_digest
        ):
            raise ValueError("fresh recompile admitted a warm/stale artifact")
        acquisition.artifact.assert_cold_admissible_with_provider(provider)
        chain = _digest_parts(
            RECOMPILE_PROVENANCE,
            request_index,
            chain,
            request.view_index,
            request.track_start,
            request.track_stop,
            acquisition.artifact.key.generation_digest,
            acquisition.artifact.generation_digest,
            acquisition.evicted_entry_count,
            acquisition.evicted_accounted_bytes,
        )
        del acquisition
    report = artifact_store.report()
    provisional = PaperKineticFixedCameraColdRecompileReceipt(
        manifest_generation_digest=manifest.generation_digest,
        provider_generation_digest=provider.generation_digest,
        world_generation_digest=provider.world.generation_digest,
        world_sites_content_digest=provider.world.sites_content_digest,
        artifact_key_chain_digest=chain,
        request_count=manifest.request_count,
        track_count=manifest.track_count,
        store_maximum_entries=report.maximum_entries,
        store_maximum_resident_accounted_bytes=(
            report.maximum_resident_accounted_bytes
        ),
        store_current_entry_count=report.current_entry_count,
        store_current_resident_accounted_bytes=(
            report.current_resident_accounted_bytes
        ),
        cold_compile_count=report.cold_compile_count,
        cold_compiled_track_count=report.cold_compiled_track_count,
        warm_hit_count=report.hit_count,
        eviction_count=report.eviction_count,
        evicted_accounted_bytes=report.evicted_accounted_bytes,
        generation_digest="",
        _provider_identity=id(provider),
        _artifact_store_identity=id(artifact_store),
        _seal=_RECOMPILE_SEAL,
    )
    result = replace(
        provisional,
        generation_digest=_recompile_receipt_digest(provisional),
    )
    result.assert_current(provider, artifact_store, manifest)
    return result


def _seal_update_receipt(
    retired_state: PaperKineticFixedCameraCombinedState,
    new_state: PaperKineticFixedCameraCombinedState,
    authorization_snapshot: _AuthorizationTransactionSnapshot,
    candidates: _UpdateCandidates,
    *,
    policy: PaperKineticFixedCameraCombinedSGDPolicy,
    memory_preflight: _TransactionMemoryPreflight,
    released_authorization_logical_tensor_bytes: int,
    old_material_generation: str,
    old_geometry_generation: str,
    old_provider_generation: str,
    old_world_generation: str,
) -> PaperKineticFixedCameraCombinedUpdateReceipt:
    if (
        retired_state.total_persistent_tensor_bytes
        != memory_preflight.combined_state_logical_tensor_bytes
        or new_state.total_persistent_tensor_bytes
        != memory_preflight.combined_state_logical_tensor_bytes
        or candidates.logical_tensor_bytes
        != memory_preflight.update_candidate_logical_tensor_bytes
    ):
        raise ArithmeticError("combined receipt inputs changed after preflight")
    provisional = PaperKineticFixedCameraCombinedUpdateReceipt(
        step_index=new_state.geometry_update_count,
        step_generation_id=authorization_snapshot.step_generation_id,
        authorization_generation_digest=authorization_snapshot.generation_digest,
        full_geometry_step_result_generation_digest=(
            authorization_snapshot.full_geometry_step_result_generation_digest
        ),
        full_geometry_reverse_mode=(
            authorization_snapshot.full_geometry_reverse_mode
        ),
        policy_generation_digest=policy.generation_digest,
        material_generation_id_before=old_material_generation,
        material_generation_id_after=new_state.material_state.material_generation_id,
        geometry_generation_id_before=old_geometry_generation,
        geometry_generation_id_after=new_state.geometry_generation_id,
        old_provider_generation_digest=old_provider_generation,
        new_provider_generation_digest=new_state.provider_generation_digest,
        old_world_generation_digest=old_world_generation,
        new_world_generation_digest=new_state.world_generation_digest,
        loss=authorization_snapshot.loss,
        raw_color_gradient_norm=candidates.raw_color_gradient_norm,
        raw_density_gradient_norm=candidates.raw_density_gradient_norm,
        position_gradient_norm=candidates.position_gradient_norm,
        velocity_gradient_norm=candidates.velocity_gradient_norm,
        weight_gradient_norm=candidates.weight_gradient_norm,
        maximum_absolute_position_update=(
            candidates.maximum_absolute_position_update
        ),
        maximum_absolute_velocity_update=(
            candidates.maximum_absolute_velocity_update
        ),
        maximum_absolute_weight_update=candidates.maximum_absolute_weight_update,
        combined_state_logical_tensor_bytes=(
            memory_preflight.combined_state_logical_tensor_bytes
        ),
        update_candidate_logical_tensor_bytes=(
            memory_preflight.update_candidate_logical_tensor_bytes
        ),
        authorization_logical_tensor_bytes=(
            memory_preflight.authorization_logical_tensor_bytes
        ),
        released_authorization_logical_tensor_bytes=(
            released_authorization_logical_tensor_bytes
        ),
        candidate_world_geometry_clone_logical_tensor_bytes=(
            memory_preflight.candidate_world_geometry_clone_logical_tensor_bytes
        ),
        update_validation_scratch_logical_tensor_bytes_upper_bound=(
            memory_preflight.update_validation_scratch_logical_tensor_bytes_upper_bound
        ),
        old_candidate_authorization_logical_tensor_bytes=(
            memory_preflight.old_candidate_authorization_logical_tensor_bytes
        ),
        old_store_resident_accounted_bytes_before_retirement=(
            memory_preflight.old_store_resident_accounted_bytes
        ),
        fresh_store_resident_accounted_bytes_upper_bound=(
            memory_preflight.fresh_store_resident_accounted_bytes_upper_bound
        ),
        transaction_tracked_logical_and_store_accounted_bytes_upper_bound=(
            memory_preflight
            .transaction_tracked_logical_and_store_accounted_bytes_upper_bound
        ),
        transaction_tracked_policy_bound=(
            policy.maximum_transaction_tracked_logical_and_store_accounted_bytes
        ),
        transaction_accounting_scope=(
            "transaction-owned-state-candidate-authorization-geometry-"
            "clone-validation-scratch-plus-"
            "store-owned-accounted-entries"
        ),
        old_store_closed_and_emptied=True,
        old_provider_seal_revoked=True,
        old_material_state_poisoned=retired_state.material_state.poisoned,
        authorization_capability_revoked=True,
        authorization_accumulator_revoked=True,
        authorization_tensor_references_released=True,
        full_geometry_step_result_revoked=True,
        caller_retained_untracked_bytes_included=False,
        ray_updates_enabled=False,
        generation_digest="",
        _seal=_UPDATE_SEAL,
    )
    result = replace(
        provisional,
        generation_digest=_update_receipt_digest(provisional),
    )
    result.assert_current()
    return result


def _consume_full_geometry_step_result(
    step_result: PaperKineticFixedCameraFullGeometryStepResult,
) -> int:
    """Revoke a successful step capability and release all retained tensor roots."""

    step_result.assert_current()
    authorization = step_result.authorization
    accumulator = step_result.accumulator
    released = accumulator.logical_tensor_bytes
    if released < 1:
        raise ArithmeticError("full-geometry authorization retained no tensors")

    object.__setattr__(step_result, "_seal", None)
    object.__setattr__(authorization, "_seal", None)
    object.__setattr__(authorization, "tensor_signatures", ())
    for name in (
        "grad_site_rgba_f32",
        "loss_f32",
        "grad_positions0_f64",
        "grad_velocities_f64",
        "grad_weight_coefficients_f64",
        "grad_track_ray_coefficients_f64",
    ):
        object.__setattr__(authorization, name, None)

    accumulator._seal = None
    accumulator.optimizer_authorized = False
    accumulator.sealed = False
    accumulator.poisoned = True
    accumulator.tensor_signatures = ()
    accumulator.ray_bar_keys = ()
    accumulator._ray_bar_keys_identity = id(accumulator.ray_bar_keys)
    for name in (
        "grad_site_rgba_f32",
        "loss_f32",
        "grad_positions0_f64",
        "grad_velocities_f64",
        "grad_weight_coefficients_f64",
        "grad_track_ray_coefficients_f64",
        "_material_tensor_ref",
        "_background_tensor_ref",
    ):
        setattr(accumulator, name, None)
    accumulator.material_tensor_identity = 0
    accumulator.material_tensor_signature = ()
    accumulator.background_tensor_identity = 0
    accumulator.background_tensor_signature = ()
    accumulator.site_table_identity = 0
    released_tensor_fields = (
        authorization.grad_site_rgba_f32,
        authorization.loss_f32,
        authorization.grad_positions0_f64,
        authorization.grad_velocities_f64,
        authorization.grad_weight_coefficients_f64,
        authorization.grad_track_ray_coefficients_f64,
        accumulator.grad_site_rgba_f32,
        accumulator.loss_f32,
        accumulator.grad_positions0_f64,
        accumulator.grad_velocities_f64,
        accumulator.grad_weight_coefficients_f64,
        accumulator.grad_track_ray_coefficients_f64,
        accumulator._material_tensor_ref,
        accumulator._background_tensor_ref,
    )
    if (
        step_result._seal is not None
        or authorization._seal is not None
        or accumulator._seal is not None
        or accumulator.optimizer_authorized
        or accumulator.sealed
        or not accumulator.poisoned
        or any(value is not None for value in released_tensor_fields)
    ):
        raise ArithmeticError("full-geometry authorization revocation was incomplete")
    return released


@dataclass
class _OwnedCandidateWorldInitializer:
    sites: AffineKineticPowerSites | None = field(repr=False)
    generation_digest: str
    provenance: str = "paper-kinetic-owned-updated-world-initializer-v1"
    consumed: bool = False

    def initialize_world(
        self,
        request: PaperKineticWorldInitializationRequest,
    ) -> AffineKineticPowerSites:
        request.assert_self_consistent()
        if self.consumed or self.sites is None:
            raise RuntimeError("updated world initializer is single-use")
        result = self.sites
        self.sites = None
        self.consumed = True
        return result


def _retire_combined_generation(
    state: PaperKineticFixedCameraCombinedState,
    provider: PaperKineticLazyProgramBundleProvider,
    artifact_store: PaperKineticCompiledCpuArtifactStore,
) -> None:
    """Retire all old-generation entry points, then report any close defect."""

    failures: list[str] = []
    try:
        artifact_store.close()
    except BaseException as error:
        failures.append(
            "old artifact-store close failed: "
            f"{type(error).__qualname__}: {error}"
        )
    try:
        object.__setattr__(provider, "_seal", None)
    except BaseException as error:
        failures.append(
            "old provider seal revocation failed: "
            f"{type(error).__qualname__}: {error}"
        )
    state.material_state.poisoned = True
    state.active = False
    state.retired = True
    state.poisoned = True
    state.generation_digest = _combined_state_digest(state)
    try:
        provider.assert_warm_current()
    except BaseException:
        pass
    else:
        failures.append("old provider remained valid after seal revocation")
    try:
        report = artifact_store.report()
        if report.current_entry_count or report.current_resident_accounted_bytes:
            failures.append("old artifact store retained entries after retirement")
    except BaseException as error:
        failures.append(
            "old artifact-store retirement report failed: "
            f"{type(error).__qualname__}: {error}"
        )
    try:
        state.assert_retired()
    except BaseException as error:
        failures.append(
            "old combined-state poisoning failed: "
            f"{type(error).__qualname__}: {error}"
        )
    if failures:
        failure = RuntimeError("; ".join(failures))
        failure.add_note(
            "The old provider and material state were invalidated; process restart "
            "is still required even if the artifact store could not be closed."
        )
        raise failure


def _invalidate_candidate_generation(
    material_state: PaperKineticFixedSiteMaterialState,
    provider: PaperKineticLazyProgramBundleProvider,
    artifact_store: PaperKineticCompiledCpuArtifactStore,
    *,
    combined_state: PaperKineticFixedCameraCombinedState | None,
) -> tuple[str, ...]:
    """Best-effort cleanup whose provider/state invalidation is unconditional."""

    failures: list[str] = []
    try:
        artifact_store.close()
    except BaseException as error:
        failures.append(
            "candidate artifact-store close failed: "
            f"{type(error).__qualname__}: {error}"
        )
    try:
        object.__setattr__(provider, "_seal", None)
    except BaseException as error:
        failures.append(
            "candidate provider seal revocation failed: "
            f"{type(error).__qualname__}: {error}"
        )
    material_state.poisoned = True
    if combined_state is not None:
        combined_state.active = False
        combined_state.retired = True
        combined_state.poisoned = True
        combined_state.generation_digest = _combined_state_digest(combined_state)
    try:
        provider.assert_warm_current()
    except BaseException:
        pass
    else:
        failures.append("candidate provider remained valid after seal revocation")
    return tuple(failures)


def _combined_sgd_policy_from_payload(
    payload: Mapping[str, Any],
) -> PaperKineticFixedCameraCombinedSGDPolicy:
    float_fields = (
        "position_learning_rate",
        "velocity_learning_rate",
        "weight_learning_rate",
        "maximum_absolute_position_update",
        "maximum_absolute_velocity_update",
        "maximum_absolute_weight_update",
        "maximum_absolute_position_value",
        "maximum_absolute_velocity_value",
        "maximum_absolute_weight_value",
    )
    integer_fields = (
        "maximum_combined_state_logical_tensor_bytes",
        "maximum_update_candidate_logical_tensor_bytes",
        "maximum_candidate_world_geometry_clone_logical_tensor_bytes",
        "maximum_update_validation_scratch_logical_tensor_bytes",
        "maximum_old_candidate_authorization_logical_tensor_bytes",
        "maximum_checkpoint_logical_tensor_bytes",
        "maximum_state_checkpoint_logical_tensor_bytes",
        "maximum_state_checkpoint_payload_logical_tensor_bytes",
        "maximum_transaction_tracked_logical_and_store_accounted_bytes",
        "maximum_recompile_request_count",
        "maximum_recompile_track_id_logical_bytes",
        "maximum_artifact_accounted_bytes",
    )
    required = {
        *float_fields,
        *integer_fields,
        "optimizer",
        "momentum",
        "weight_decay",
        "fixed_camera",
        "ray_updates_enabled",
        "generation_digest",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("combined SGD policy payload keys changed")
    for name in float_fields:
        value = payload[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(f"combined SGD policy {name} must be finite and positive")
    for name in integer_fields:
        _require_positive_int(payload[name], name=name)
    if (
        payload["optimizer"] != "manual_sgd"
        or isinstance(payload["momentum"], bool)
        or not isinstance(payload["momentum"], (int, float))
        or float(payload["momentum"]) != 0.0
        or isinstance(payload["weight_decay"], bool)
        or not isinstance(payload["weight_decay"], (int, float))
        or float(payload["weight_decay"]) != 0.0
        or payload["fixed_camera"] is not True
        or payload["ray_updates_enabled"] is not False
    ):
        raise ValueError("combined SGD policy restart semantics changed")
    result = PaperKineticFixedCameraCombinedSGDPolicy(
        **{name: payload[name] for name in float_fields},
        **{name: payload[name] for name in integer_fields},
    )
    result.assert_valid()
    _require_sha256(payload["generation_digest"], name="policy generation_digest")
    if payload["generation_digest"] != result.generation_digest:
        raise ValueError("combined SGD policy generation changed")
    return result


def _validate_geometry_tensors(
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    site_count: int,
) -> None:
    _validate_geometry_tensor_metadata(tensors, site_count=site_count)
    if any(not bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
        raise ValueError("combined geometry tensor is nonfinite")


def _validate_geometry_tensor_metadata(
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    site_count: int,
) -> None:
    """Validate geometry layout without scanning tensor contents."""

    expected_shapes = ((site_count, 3), (site_count, 3), None)
    for name, tensor, expected_shape in zip(
        ("positions0", "velocities", "weight_coefficients"),
        tensors,
        expected_shapes,
        strict=True,
    ):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.float64
            or tensor.device.type != "cpu"
            or tensor.ndim != 2
            or not tensor.is_contiguous()
            or tensor.requires_grad
            or expected_shape is not None
            and tuple(tensor.shape) != expected_shape
        ):
            raise ValueError(f"combined {name} tensor is invalid")
    if (
        tensors[2].shape[0] != site_count
        or not 1 <= int(tensors[2].shape[1]) <= 3
    ):
        raise ValueError("combined weight coefficient tensor is invalid")
    _require_distinct_storage(*tensors)


def _validate_checkpoint_material_tensor_metadata(
    tensors: tuple[Any, Any],
    *,
    site_count: int,
) -> None:
    """Validate raw material payload layout without scanning or cloning it."""

    for name, tensor, shape in zip(
        ("raw_color_f32_cpu", "raw_density_f32_cpu"),
        tensors,
        ((site_count, 3), (site_count,)),
        strict=True,
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"combined checkpoint {name} must be a tensor")
        if (
            tensor.device.type != "cpu"
            or tensor.dtype != torch.float32
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
            or tensor.requires_grad
        ):
            raise ValueError(
                f"combined checkpoint {name} must be contiguous CPU float32 {shape}"
            )


def _combined_checkpoint_tensor_bytes(
    state: PaperKineticFixedCameraCombinedState,
) -> int:
    # Raw RGB+density plus the three float64 geometry tensors.
    return state.site_count * 4 * 4 + state.geometry_tensor_bytes


def _manifest_digest(
    manifest: PaperKineticFixedCameraColdRecompileManifest,
) -> str:
    return _digest_parts(
        RECOMPILE_PROVENANCE,
        "manifest",
        manifest.dataset_generation_digest,
        manifest.camera_grid_digest,
        manifest.factory_generation_digest,
        manifest.height,
        manifest.width,
        manifest.maximum_tracks_per_bundle,
        tuple(
            (request.view_index, request.track_start, request.track_stop)
            for request in manifest.requests
        ),
    )


def _combined_state_digest(state: PaperKineticFixedCameraCombinedState) -> str:
    return _digest_parts(
        STATE_PROVENANCE,
        state.material_state.material_generation_id,
        state.provider_generation_digest,
        state.world_generation_digest,
        state.sites_content_digest,
        state.geometry_generation_parent_digest,
        state.geometry_generation_id,
        state.last_authorization_generation_digest,
        state.last_step_generation_id,
        state.last_update_policy_generation_digest,
        state.geometry_update_count,
        state.cold_recompile_seal_generation_digest,
        state.active,
        state.retired,
        state.poisoned,
    )


def _update_receipt_digest(
    receipt: PaperKineticFixedCameraCombinedUpdateReceipt,
) -> str:
    return _digest_parts(
        UPDATE_PROVENANCE,
        *(
            value
            for name, value in receipt.__dict__.items()
            if name not in {"generation_digest", "_seal"}
        ),
    )


def _recompile_receipt_digest(
    receipt: PaperKineticFixedCameraColdRecompileReceipt,
) -> str:
    return _digest_parts(
        RECOMPILE_PROVENANCE,
        *(
            value
            for name, value in receipt.__dict__.items()
            if name
            not in {
                "generation_digest",
                "_provider_identity",
                "_artifact_store_identity",
                "_seal",
            }
        ),
    )


def _combined_checkpoint_digest(
    checkpoint: PaperKineticFixedCameraCombinedCheckpoint,
) -> str:
    return _digest_parts(
        CHECKPOINT_PROVENANCE,
        CHECKPOINT_SCHEMA,
        checkpoint.material_checkpoint.generation_digest,
        checkpoint.combined_sgd_policy.generation_digest,
        checkpoint.provider_generation_digest,
        checkpoint.world_generation_digest,
        checkpoint.sites_content_digest,
        checkpoint.initializer_generation_digest,
        checkpoint.geometry_generation_parent_digest,
        checkpoint.geometry_generation_id,
        checkpoint.last_authorization_generation_digest,
        checkpoint.last_step_generation_id,
        checkpoint.last_update_policy_generation_digest,
        checkpoint.geometry_update_count,
        checkpoint.cold_recompile_manifest.generation_digest,
        checkpoint.cold_recompile_seal_generation_digest,
        checkpoint.positions0_content_digest,
        checkpoint.velocities_content_digest,
        checkpoint.weight_coefficients_content_digest,
        checkpoint.live_state_logical_tensor_bytes_at_checkpoint,
        checkpoint.state_checkpoint_logical_tensor_bytes,
        checkpoint.state_checkpoint_payload_peak_logical_tensor_bytes,
        checkpoint.checkpoint_tensor_bytes,
        checkpoint.persistent_frame_tensor_bytes,
        checkpoint.persistent_sample_tensor_bytes,
        checkpoint.persistent_target_tensor_bytes,
        checkpoint.persistent_prediction_tensor_bytes,
        checkpoint.optimizer_history_tensor_bytes,
        checkpoint.camera_ray_parameter_tensor_bytes,
        checkpoint.compiled_tensor_bytes,
        checkpoint.combined_checkpoint_restore_integrated,
        checkpoint.production_trainer_integrated,
        checkpoint.allocator_peak_measured,
    )


def _ready_generation_digest(
    ready: PaperKineticFixedCameraReadyGeneration,
) -> str:
    return _digest_parts(
        READY_PROVENANCE,
        ready.state.generation_digest,
        ready.provider.generation_digest,
        ready.update_receipt.generation_digest,
        ready.recompile_receipt.generation_digest,
        ready.manifest.generation_digest,
        ready.next_step_claimed,
    )


def _restore_receipt_digest(
    receipt: PaperKineticFixedCameraCombinedRestoreReceipt,
) -> str:
    return _digest_parts(
        RESTORE_PROVENANCE,
        *(
            value
            for name, value in receipt.__dict__.items()
            if name not in {"generation_digest", "_seal"}
        ),
    )


def _restored_ready_generation_digest(
    ready: PaperKineticFixedCameraRestoredReadyGeneration,
) -> str:
    return _digest_parts(
        RESTORED_READY_PROVENANCE,
        ready.state.generation_digest,
        ready.provider.generation_digest,
        ready.restore_receipt.generation_digest,
        ready.recompile_receipt.generation_digest,
        ready.manifest.generation_digest,
        ready.next_step_claimed,
    )


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
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
        bool(tensor.requires_grad),
    )


def _tensor_bytes(*tensors: torch.Tensor) -> int:
    storages: dict[tuple[str, int], int] = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), int(storage.data_ptr()))] = int(storage.nbytes())
    return sum(storages.values())


def _tensor_content_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(repr(tuple(value.shape)).encode("utf-8"))
    digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def _require_distinct_storage(*tensors: torch.Tensor) -> None:
    identities = tuple(
        (str(tensor.device), int(tensor.untyped_storage().data_ptr()))
        for tensor in tensors
    )
    if len(set(identities)) != len(identities):
        raise ValueError("combined state/bar tensors must not alias")


def _require_sha256(value: str, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


__all__ = [
    "CHECKPOINT_PROVENANCE",
    "CHECKPOINT_SCHEMA",
    "READY_PROVENANCE",
    "RECOMPILE_PROVENANCE",
    "RESTORED_READY_PROVENANCE",
    "RESTORE_PROVENANCE",
    "RUNTIME_STATUS",
    "STATE_PROVENANCE",
    "UPDATE_PROVENANCE",
    "PaperKineticFixedCameraColdRecompileManifest",
    "PaperKineticFixedCameraColdRecompileReceipt",
    "PaperKineticFixedCameraColdRecompileRequest",
    "PaperKineticFixedCameraCombinedCheckpoint",
    "PaperKineticFixedCameraCombinedRestoreReceipt",
    "PaperKineticFixedCameraCombinedSGDPolicy",
    "PaperKineticFixedCameraCombinedState",
    "PaperKineticFixedCameraCombinedTransactionFailure",
    "PaperKineticFixedCameraCombinedUpdateReceipt",
    "PaperKineticFixedCameraReadyGeneration",
    "PaperKineticFixedCameraRestoredReadyGeneration",
    "apply_paper_kinetic_fixed_camera_combined_sgd_transaction",
    "checkpoint_paper_kinetic_fixed_camera_combined_state",
    "claim_paper_kinetic_fixed_camera_ready_generation_for_next_step",
    "claim_paper_kinetic_fixed_camera_restored_ready_generation_for_next_step",
    "claim_paper_kinetic_fixed_camera_restored_ready_generation_for_lazy_native_next_step",
    "paper_kinetic_fixed_camera_cold_recompile_manifest_from_payload",
    "paper_kinetic_fixed_camera_combined_checkpoint_from_payload",
    "prepare_paper_kinetic_fixed_camera_cold_recompile_manifest",
    "prepare_paper_kinetic_fixed_camera_selected_tracks_cold_recompile_manifest",
    "prepare_paper_kinetic_fixed_camera_combined_state",
    "restore_paper_kinetic_fixed_camera_combined_generation",
    "restore_paper_kinetic_fixed_camera_combined_generation_from_payload",
]
