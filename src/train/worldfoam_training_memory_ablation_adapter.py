"""Production adapter for the fixed-camera WorldFoam memory ablation.

This module is deliberately narrower than the fresh-process producer.  It
constructs the checked-in procedural world and direct selected-pixel target
source, executes one lazy full-geometry step, consumes its exact device/CPU
gradient receipt in one manual-SGD/cold-recompile transaction, and projects
only receipt-backed fields into a JSON-safe row.

Process RSS, MPS allocator peaks, native-extension hashes, watchdog evidence,
and publication eligibility remain producer-owned.  In particular this module
does not rename streamed samples as camera work, infer compiler work from
configuration, or report logical byte bounds as measured peaks.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import weakref
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from camera import CameraSpec
from kinetic_compiled_cpu_artifact_store import (
    PaperKineticCompiledCpuArtifactStore,
    PaperKineticCompiledCpuArtifactStorePolicy,
)
from kinetic_dense_cached_native_material_request import (
    MPS_DEVICE_COMPLETION_FENCE_PROVENANCE,
    synchronize_mps_device_completion_fence,
)
from kinetic_lazy_native_material_step import (
    TARGET_FRAME_STREAM_ONCE,
    PaperKineticLazyNativeMemoryPolicy,
    paper_kinetic_observation_manifest_digest,
    prepare_paper_kinetic_lazy_native_trainer_state,
)
from kinetic_power_word_compiler import AffineKineticPowerSites
from paper_kinetic_active_track_program_factory import (
    PaperKineticActiveP0TrackProgramFactoryConfig,
    prepare_paper_kinetic_active_p0_track_program_factory,
)
from paper_kinetic_compiled_framewise_full_geometry_control import (
    prepare_paper_kinetic_compiled_framewise_program_provider,
    run_paper_kinetic_compiled_framewise_full_geometry_control,
)
from paper_kinetic_fixed_camera_combined_state import (
    PaperKineticFixedCameraCombinedSGDPolicy,
    checkpoint_paper_kinetic_fixed_camera_combined_state,
    claim_paper_kinetic_fixed_camera_restored_ready_generation_for_lazy_native_next_step,
    prepare_paper_kinetic_fixed_camera_combined_state,
    restore_paper_kinetic_fixed_camera_combined_generation_from_payload,
)
from paper_kinetic_fixed_site_material_device_bridge import (
    snapshot_paper_kinetic_fixed_site_material_to_device,
)
from paper_kinetic_fixed_site_material_state import (
    PaperKineticFixedSiteMaterialParameterization,
    PaperKineticFixedSiteMaterialSGDPolicy,
    prepare_paper_kinetic_fixed_site_material_state,
)
from paper_kinetic_lazy_full_geometry_device_bridge import (
    apply_paper_kinetic_lazy_full_geometry_combined_sgd_transaction,
    claim_paper_kinetic_lazy_full_geometry_ready_generation_for_next_step,
    seal_paper_kinetic_lazy_full_geometry_device_gradient_receipt,
)
from paper_kinetic_lazy_full_geometry_step import (
    FUSED_UNION_V2,
    STAGED_SPARSE,
    PaperKineticLazyFullGeometryMemoryPolicy,
    run_paper_kinetic_lazy_native_full_geometry_step,
)
from paper_kinetic_lazy_program_bundles import (
    PaperKineticObservation,
    PaperKineticWorldInitializationRequest,
    prepare_paper_kinetic_lazy_program_bundle_provider,
)
from paper_kinetic_world_initializer import (
    prepare_paper_kinetic_p0_material_initialization,
)
from powerfoam_training_data import (
    PowerFoamRayProvider,
    PowerFoamSelectedPixelRead,
    PowerFoamTargetProvider,
)


ADAPTER_PROVENANCE = "worldfoam-training-memory-ablation-adapter-v1"
TARGET_SOURCE_PROVENANCE = (
    "worldfoam-training-memory-procedural-direct-selected-pixels-v1"
)
OBSERVATION_SOURCE_PROVENANCE = (
    "worldfoam-training-memory-replayable-selected-observations-v1"
)

_DRIVER_TENSOR_INPUT_KEYS = (
    "positions0_f64_cpu",
    "velocities_f64_cpu",
    "weight_coefficients_f64_cpu",
    "site_rgba_f32_cpu",
    "background_rgb_f32_cpu",
    "full_physical_time_grid_f64_cpu",
    "selected_frame_indices_i64_cpu",
    "selected_physical_times_f64_cpu",
    "track_ids_i64_cpu",
    "pixel_ids_i64_cpu",
    "rows_i64_cpu",
    "columns_i64_cpu",
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(int(tensor.numel()) * int(tensor.element_size()) for tensor in tensors)


def _take_driver_tensor_inputs(inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Transfer every driver-created tensor out of the caller-owned mapping."""

    result: dict[str, torch.Tensor] = {}
    for key in _DRIVER_TENSOR_INPUT_KEYS:
        value = inputs.pop(key, None)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"driver tensor input {key} is missing or changed")
        result[key] = value
    if any(isinstance(value, torch.Tensor) for value in inputs.values()):
        raise ValueError("unclassified driver tensor input would remain strongly rooted")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(repr(tuple(int(item) for item in value.shape)).encode("utf-8"))
    digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def _combined_state_content_digest(state: Any) -> str:
    state.material_state.assert_current()
    return _digest_parts(
        "worldfoam-training-memory-combined-state-content-v1",
        state.geometry_update_count,
        _tensor_digest(state.material_state.raw_color_f32),
        _tensor_digest(state.material_state.raw_density_f32),
        _tensor_digest(state.positions0_f64),
        _tensor_digest(state.velocities_f64),
        _tensor_digest(state.weight_coefficients_f64),
    )


def _parity_gradient_digest(payload: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {
            "loss": payload["loss"],
            "material_gradient": payload["material_gradient"],
            "geometry_gradient": payload["geometry_gradient"],
        }
    )


def _flat_finite_values(*tensors: torch.Tensor) -> list[float]:
    result: list[float] = []
    for tensor in tensors:
        value = tensor.detach().to(device="cpu").reshape(-1)
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError("parity tensor contains a non-finite value")
        result.extend(float(item) for item in value.tolist())
    return result


def _public_receipt_fields(receipt: Any) -> dict[str, Any]:
    """Project one sealed receipt without exposing its in-process seal."""

    fields = getattr(receipt, "__dict__", None)
    if not isinstance(fields, dict):
        raise TypeError("ablation receipt must expose dataclass fields")
    return {key: value for key, value in fields.items() if key != "_seal"}


def _teacher_rgb(
    *, row: int, column: int, width: int, height: int, physical_time: float
) -> tuple[float, float, float]:
    u = (column + 0.5) / float(width)
    v = (row + 0.5) / float(height)
    return (
        0.5 + 0.25 * math.sin(2.0 * math.pi * (u + 0.15 * physical_time)),
        0.5
        + 0.25
        * math.sin(
            2.0 * math.pi * (v - 0.10 * physical_time) + 2.0 * math.pi / 3.0
        ),
        0.5
        + 0.25
        * math.sin(
            2.0 * math.pi * (u + v + 0.05 * physical_time)
            + 4.0 * math.pi / 3.0
        ),
    )


@dataclass(frozen=True)
class _ProceduralDirectSelectedPixelTargetSource:
    """Nonresident target source whose only allocating API returns [N,3]."""

    frame_times: tuple[float, ...]
    height: int
    width: int
    target_generation_id: str
    view_count: int = 1
    source_provenance: str = TARGET_SOURCE_PROVENANCE

    @property
    def frame_count(self) -> int:
        return len(self.frame_times)

    def select_view_frames(
        self,
        view_indices: tuple[int, ...],
        frame_indices: tuple[int, ...],
    ) -> torch.Tensor:
        del view_indices, frame_indices
        raise RuntimeError(
            "paper-memory target source forbids full-frame materialization"
        )

    def select_view_frame_pixels_cpu(
        self,
        view_indices: tuple[int, ...],
        frame_indices: tuple[int, ...],
        pixel_indices: tuple[int, ...],
        *,
        maximum_source_decode_tensor_bytes: int,
    ) -> PowerFoamSelectedPixelRead:
        count = len(pixel_indices)
        if count < 1 or len(view_indices) != count or len(frame_indices) != count:
            raise ValueError("selected-pixel target request is empty or ragged")
        required_bytes = count * 3 * 4
        if required_bytes > maximum_source_decode_tensor_bytes:
            raise MemoryError(
                "procedural selected-pixel output exceeds its pre-allocation budget"
            )
        if any(view_index != 0 for view_index in view_indices):
            raise IndexError("procedural selected-pixel request left its sole view")
        output = torch.empty((count, 3), dtype=torch.float32, device="cpu")
        for index, (frame_index, pixel_index) in enumerate(
            zip(frame_indices, pixel_indices, strict=True)
        ):
            if not 0 <= frame_index < self.frame_count:
                raise IndexError("procedural selected-pixel frame left the dataset")
            if not 0 <= pixel_index < self.height * self.width:
                raise IndexError("procedural selected-pixel coordinate left the image")
            row, column = divmod(pixel_index, self.width)
            rgb = _teacher_rgb(
                row=row,
                column=column,
                width=self.width,
                height=self.height,
                physical_time=self.frame_times[frame_index],
            )
            output[index, 0] = rgb[0]
            output[index, 1] = rgb[1]
            output[index, 2] = rgb[2]
        return PowerFoamSelectedPixelRead.seal(
            output,
            selection_mode="direct_pixels",
            source_provenance=self.source_provenance,
            source_visible_peak_logical_tensor_bytes_upper_bound=required_bytes,
        )

    def residency(self) -> dict[str, Any]:
        return {
            "source_kind": "deterministic_procedural_direct_selected_pixels",
            "source_device": "cpu",
            "logical_bytes": self.frame_count * 3 * self.height * self.width * 4,
            "resident_bytes": 0,
            "full_source_resident": False,
            "disk_lazy_decode": False,
            "full_frame_api_fail_closed": True,
            "target_generation_id": self.target_generation_id,
        }


@dataclass(frozen=True)
class _ReplayableSelectedObservationSource:
    """O(P+F) manifest yielding canonical ``(view,pixel,frame,id)`` rows."""

    pixel_ids: tuple[int, ...]
    frame_indices: tuple[int, ...]
    dataset_frame_count: int
    image_pixel_count: int
    expected_observation_count: int
    generation_digest: str
    provenance: str = OBSERVATION_SOURCE_PROVENANCE

    def __iter__(self) -> Iterator[PaperKineticObservation]:
        emitted = 0
        for track_position, pixel_id in enumerate(self.pixel_ids):
            for frame_index in self.frame_indices:
                emitted += 1
                yield PaperKineticObservation(
                    observation_id=(
                        track_position * self.dataset_frame_count + frame_index
                    ),
                    view_index=0,
                    frame_index=frame_index,
                    pixel_index=pixel_id,
                )
        if emitted != self.expected_observation_count:
            raise ArithmeticError("selected observation replay coverage changed")

    @classmethod
    def seal(
        cls,
        *,
        pixel_ids: Sequence[int],
        frame_indices: Sequence[int],
        dataset_frame_count: int,
        image_pixel_count: int,
    ) -> _ReplayableSelectedObservationSource:
        pixels = tuple(int(value) for value in pixel_ids)
        frames = tuple(int(value) for value in frame_indices)
        if (
            not pixels
            or tuple(sorted(set(pixels))) != pixels
            or pixels[0] < 0
            or pixels[-1] >= image_pixel_count
            or not frames
            or tuple(sorted(set(frames))) != frames
            or frames[0] < 0
            or frames[-1] >= dataset_frame_count
        ):
            raise ValueError("selected observation axes are not canonical")
        count = len(pixels) * len(frames)
        digest = _digest_parts(
            OBSERVATION_SOURCE_PROVENANCE,
            pixels,
            frames,
            dataset_frame_count,
            image_pixel_count,
            count,
        )
        return cls(
            pixel_ids=pixels,
            frame_indices=frames,
            dataset_frame_count=dataset_frame_count,
            image_pixel_count=image_pixel_count,
            expected_observation_count=count,
            generation_digest=digest,
        )


@dataclass
class _OwnedInputWorldInitializer:
    positions0_f64_cpu: torch.Tensor | None = field(repr=False)
    velocities_f64_cpu: torch.Tensor | None = field(repr=False)
    weight_coefficients_f64_cpu: torch.Tensor | None = field(repr=False)
    dataset_generation_digest: str
    expected_view_count: int
    expected_frame_count: int
    expected_height: int
    expected_width: int
    generation_digest: str
    provenance: str = "worldfoam-training-memory-owned-input-world-v1"
    consumed: bool = False

    def initialize_world(
        self, request: PaperKineticWorldInitializationRequest
    ) -> AffineKineticPowerSites:
        request.assert_self_consistent()
        if self.consumed:
            raise RuntimeError("ablation world initializer is one-shot")
        if (
            request.dataset_generation_digest != self.dataset_generation_digest
            or request.view_count != self.expected_view_count
            or request.frame_count != self.expected_frame_count
            or request.height != self.expected_height
            or request.width != self.expected_width
            or request.initializer_generation_digest != self.generation_digest
        ):
            raise ValueError("ablation world initializer received a foreign request")
        tensors = (
            self.positions0_f64_cpu,
            self.velocities_f64_cpu,
            self.weight_coefficients_f64_cpu,
        )
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise ValueError("ablation world initializer lost its geometry")
        result = AffineKineticPowerSites(
            positions0=tensors[0],  # type: ignore[arg-type]
            velocities=tensors[1],  # type: ignore[arg-type]
            weight_coefficients=tensors[2],  # type: ignore[arg-type]
        )
        self.positions0_f64_cpu = None
        self.velocities_f64_cpu = None
        self.weight_coefficients_f64_cpu = None
        self.consumed = True
        return result


def _camera_from_config(config: Mapping[str, Any]) -> CameraSpec:
    camera = _mapping(config.get("camera"), name="camera")
    program = _mapping(camera.get("program"), name="camera.program")
    if program.get("kind") != "fixed_pinhole_world_to_camera_v1":
        raise ValueError("ablation requires its fixed pinhole camera program")
    flat = tuple(
        _finite_float(value, name="camera world-to-camera entry")
        for value in program.get("world_to_camera_row_major", ())
    )
    if len(flat) != 16:
        raise ValueError("camera world-to-camera matrix must have 16 entries")
    world_to_camera = torch.tensor(flat, dtype=torch.float64).reshape(4, 4)
    camera_to_world = torch.linalg.inv(world_to_camera).contiguous()
    if not bool(torch.isfinite(camera_to_world).all().item()):
        raise ValueError("camera inverse is non-finite")
    return CameraSpec(
        fx=_finite_float(program.get("fx"), name="camera.fx"),
        fy=_finite_float(program.get("fy"), name="camera.fy"),
        cx=_finite_float(program.get("cx"), name="camera.cx"),
        cy=_finite_float(program.get("cy"), name="camera.cy"),
        camera_to_world=camera_to_world,
        lens_model="pinhole",
        distortion=None,
    )


def _validate_driver_target_chunks(
    inputs: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    frame_indices: tuple[int, ...],
    pixel_ids: tuple[int, ...],
    rows: tuple[int, ...],
    columns: tuple[int, ...],
    frame_times: tuple[float, ...],
) -> str:
    """Consume the driver's bounded target stream once and bind its contents."""

    factory = inputs.get("target_chunk_factory")
    if not callable(factory):
        raise TypeError("driver inputs lack a replayable target chunk factory")
    maximum = _positive_int(
        config["target_source"]["maximum_resident_observations"],
        name="target maximum resident observations",
    )
    expected_count = len(frame_indices) * len(pixel_ids)
    digest = hashlib.sha256()
    observed = 0
    for chunk in factory():
        chunk = _mapping(chunk, name="driver target chunk")
        records = tuple(chunk.get("records", ()))
        if (
            not records
            or len(records) > maximum
            or chunk.get("observation_count") != len(records)
            or chunk.get("logical_target_tensor_bytes") != len(records) * 3 * 4
            or chunk.get("full_frame_materialized") is not False
        ):
            raise ValueError("driver target chunk violates its bounded contract")
        for record in records:
            record = _mapping(record, name="driver target record")
            frame_position, track_position = divmod(observed, len(pixel_ids))
            if frame_position >= len(frame_indices):
                raise ValueError("driver target stream emitted extra records")
            frame_index = frame_indices[frame_position]
            expected_rgb = _teacher_rgb(
                row=rows[track_position],
                column=columns[track_position],
                width=int(config["image"]["width"]),
                height=int(config["image"]["height"]),
                physical_time=frame_times[frame_index],
            )
            if (
                record.get("dataset_frame_index") != frame_index
                or record.get("pixel_id") != pixel_ids[track_position]
                or record.get("row") != rows[track_position]
                or record.get("column") != columns[track_position]
                or tuple(record.get("rgb", ())) != expected_rgb
            ):
                raise ValueError("driver target stream content/order changed")
            encoded = json.dumps(
                dict(record),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            observed += 1
    if observed != expected_count:
        raise ValueError("driver target stream coverage changed")
    return digest.hexdigest()


def _selected_track_manifest(provider: Any, pixel_ids: tuple[int, ...]) -> Any:
    """Call the public sparse-track cold-recompile helper, never H*W fallback."""

    import paper_kinetic_fixed_camera_combined_state as combined_state

    helper = getattr(
        combined_state,
        "prepare_paper_kinetic_fixed_camera_selected_tracks_cold_recompile_manifest",
        None,
    )
    if not callable(helper):
        raise RuntimeError(
            "selected-track cold-recompile manifest helper is not source-complete; "
            "the dense H*W manifest is forbidden for this P=512 ablation"
        )
    manifest = helper(
        provider,
        selected_track_ids_by_view=((0, pixel_ids),),
        maximum_tracks_per_request=provider.maximum_tracks_per_bundle,
        maximum_request_count=len(pixel_ids),
        maximum_track_id_logical_bytes=len(pixel_ids) * 3 * 8,
    )
    if manifest.track_count != len(pixel_ids):
        raise ArithmeticError("selected-track cold manifest changed its coverage")
    return manifest


def _build_policies(
    config: Mapping[str, Any], *, reverse_mode: str, selected_track_count: int
) -> tuple[
    PaperKineticCompiledCpuArtifactStorePolicy,
    PaperKineticLazyNativeMemoryPolicy,
    PaperKineticLazyFullGeometryMemoryPolicy,
    PaperKineticFixedCameraCombinedSGDPolicy,
]:
    spatial = _mapping(config.get("spatial_streaming"), name="spatial_streaming")
    limits = _mapping(config.get("memory_limits_bytes"), name="memory limits")
    bridge_bound = _positive_int(
        limits.get("maximum_bridge_visible_peak"),
        name="maximum bridge-visible peak",
    )
    state_bound = _positive_int(
        limits.get("maximum_frame_invariant_live_logical"),
        name="maximum frame-invariant logical state",
    )
    artifact_bound = _positive_int(
        limits.get("maximum_artifact_store_resident"),
        name="maximum artifact-store resident bytes",
    )
    maximum_observations_per_chunk = _positive_int(
        spatial.get("maximum_observations_per_chunk"),
        name="maximum observations per chunk",
    )
    selected_target_bound = maximum_observations_per_chunk * 3 * 4
    store_policy = PaperKineticCompiledCpuArtifactStorePolicy(
        maximum_entries=_positive_int(
            spatial.get("artifact_store_maximum_entries"),
            name="artifact store maximum entries",
        ),
        maximum_resident_accounted_bytes=artifact_bound,
    )
    lazy_policy = PaperKineticLazyNativeMemoryPolicy(
        max_global_material_and_bar_tensor_bytes=state_bound,
        max_bundle_observation_count=maximum_observations_per_chunk,
        max_lane_resident_logical_tensor_bytes=bridge_bound,
        max_active_node_and_vjp_tensor_bytes=bridge_bound,
        max_decoded_frame_scratch_tensor_bytes=selected_target_bound,
        max_selected_frame_target_tensor_bytes=selected_target_bound,
        max_sample_launch_tensor_bytes=bridge_bound,
        max_coordinator_visible_live_tensor_bytes=bridge_bound,
        target_frame_access_mode=TARGET_FRAME_STREAM_ONCE,
        max_step_target_frame_cache_tensor_bytes=0,
    )
    geometry_policy = PaperKineticLazyFullGeometryMemoryPolicy(
        maximum_global_geometry_bar_logical_tensor_bytes=state_bound,
        maximum_geometry_bridge_visible_peak_logical_tensor_bytes=bridge_bound,
        maximum_fused_union_transaction_scratch_tensor_bytes=(
            bridge_bound if reverse_mode == FUSED_UNION_V2 else 0
        ),
    )
    learning_rate = _finite_float(
        config["optimizer"]["learning_rate"], name="optimizer learning rate"
    )
    combined_policy = PaperKineticFixedCameraCombinedSGDPolicy(
        position_learning_rate=learning_rate,
        velocity_learning_rate=learning_rate,
        weight_learning_rate=learning_rate,
        maximum_absolute_position_update=1.0,
        maximum_absolute_velocity_update=1.0,
        maximum_absolute_weight_update=1.0,
        maximum_absolute_position_value=16.0,
        maximum_absolute_velocity_value=16.0,
        maximum_absolute_weight_value=16.0,
        maximum_combined_state_logical_tensor_bytes=state_bound,
        maximum_update_candidate_logical_tensor_bytes=state_bound,
        maximum_candidate_world_geometry_clone_logical_tensor_bytes=state_bound,
        maximum_update_validation_scratch_logical_tensor_bytes=state_bound,
        maximum_old_candidate_authorization_logical_tensor_bytes=state_bound,
        maximum_checkpoint_logical_tensor_bytes=state_bound,
        maximum_state_checkpoint_logical_tensor_bytes=state_bound,
        maximum_state_checkpoint_payload_logical_tensor_bytes=state_bound,
        maximum_transaction_tracked_logical_and_store_accounted_bytes=(
            int(config["memory_limits_bytes"]["maximum_worker_process_group_rss"])
        ),
        maximum_recompile_request_count=selected_track_count,
        maximum_recompile_track_id_logical_bytes=selected_track_count * 3 * 8,
        maximum_artifact_accounted_bytes=artifact_bound,
    )
    return store_policy, lazy_policy, geometry_policy, combined_policy


def _prepare_initial_generation(
    context: Mapping[str, Any],
) -> dict[str, Any]:
    config = _mapping(context.get("config"), name="ablation config")
    raw_inputs = context.get("inputs")
    if not isinstance(raw_inputs, dict):
        raise TypeError("ablation inputs must be an ownership-transfer dictionary")
    inputs = raw_inputs
    image = _mapping(config.get("image"), name="image")
    height = _positive_int(image.get("height"), name="image.height")
    width = _positive_int(image.get("width"), name="image.width")
    frame_times = tuple(
        float(value)
        for value in inputs["full_physical_time_grid_f64_cpu"].tolist()
    )
    selected_frames = tuple(
        int(value) for value in inputs["selected_frame_indices_i64_cpu"].tolist()
    )
    pixel_ids = tuple(int(value) for value in inputs["pixel_ids_i64_cpu"].tolist())
    rows = tuple(int(value) for value in inputs["rows_i64_cpu"].tolist())
    columns = tuple(int(value) for value in inputs["columns_i64_cpu"].tolist())
    target_stream_manifest = _validate_driver_target_chunks(
        inputs,
        config,
        frame_indices=selected_frames,
        pixel_ids=pixel_ids,
        rows=rows,
        columns=columns,
        frame_times=frame_times,
    )
    tensor_inputs = _take_driver_tensor_inputs(inputs)
    device = torch.device(str(context.get("backend")))
    if device.type != "mps" and context.get("allow_cpu_fake_native") is not True:
        raise ValueError("production adapter requires MPS unless CPU fake-native is explicit")

    dataset_generation_digest = _canonical_sha256(
        {
            "adapter": ADAPTER_PROVENANCE,
            "input_program_sha256": inputs["input_program_sha256"],
            "target_generation_id": inputs["target_generation_id"],
            "target_stream_manifest_sha256": target_stream_manifest,
        }
    )
    initializer_digest = _canonical_sha256(
        {
            "adapter": ADAPTER_PROVENANCE,
            "role": "owned-world-initializer",
            "dataset_generation_digest": dataset_generation_digest,
            "compiled_world_sha256": inputs["compiled_world_sha256"],
        }
    )
    initializer = _OwnedInputWorldInitializer(
        positions0_f64_cpu=tensor_inputs["positions0_f64_cpu"],
        velocities_f64_cpu=tensor_inputs["velocities_f64_cpu"],
        weight_coefficients_f64_cpu=tensor_inputs["weight_coefficients_f64_cpu"],
        dataset_generation_digest=dataset_generation_digest,
        expected_view_count=1,
        expected_frame_count=len(frame_times),
        expected_height=height,
        expected_width=width,
        generation_digest=initializer_digest,
    )
    camera = _camera_from_config(config)
    target_source = _ProceduralDirectSelectedPixelTargetSource(
        frame_times=frame_times,
        height=height,
        width=width,
        target_generation_id=str(inputs["target_generation_id"]),
    )
    target_provider = PowerFoamTargetProvider(source=target_source, device=device)
    ray_provider = PowerFoamRayProvider(
        cameras=(tuple(camera for _ in frame_times),),
        height=height,
        width=width,
        device=device,
    )
    compiler = _mapping(config.get("compiler"), name="compiler")
    factory = prepare_paper_kinetic_active_p0_track_program_factory(
        PaperKineticActiveP0TrackProgramFactoryConfig(
            near=_finite_float(compiler.get("near"), name="compiler.near"),
            far=_finite_float(compiler.get("far"), name="compiler.far"),
            node_count=_positive_int(
                compiler.get("node_count"), name="compiler.node_count"
            ),
            maximum_sites_per_track_compile=_positive_int(
                compiler.get("maximum_sites_per_track_compile"),
                name="compiler.maximum_sites_per_track_compile",
            ),
            maximum_charts_per_track=_positive_int(
                compiler.get("maximum_charts_per_track"),
                name="compiler.maximum_charts_per_track",
            ),
            maximum_owner_runs_per_chart=_positive_int(
                compiler.get("maximum_owner_runs_per_chart"),
                name="compiler.maximum_owner_runs_per_chart",
            ),
            rank_selection_provenance=str(
                compiler.get("rank_selection_provenance", "")
            ),
        )
    )
    spatial = _mapping(config.get("spatial_streaming"), name="spatial_streaming")
    provider = prepare_paper_kinetic_lazy_program_bundle_provider(
        dataset_generation_digest=dataset_generation_digest,
        target_provider=target_provider,
        ray_provider=ray_provider,
        frame_times=frame_times,
        height=height,
        width=width,
        maximum_tracks_per_bundle=_positive_int(
            spatial.get("maximum_tracks_per_request"),
            name="maximum tracks per request",
        ),
        maximum_observations_per_bundle=_positive_int(
            spatial.get("maximum_observations_per_chunk"),
            name="maximum observations per chunk",
        ),
        maximum_rows_per_native_block=_positive_int(
            spatial.get("maximum_rows_per_native_block"),
            name="maximum rows per native block",
        ),
        world_initializer=initializer,
        program_factory=factory,
    )
    if not initializer.consumed:
        raise ArithmeticError("provider did not consume the owned world initializer")
    reverse_mode = str(context.get("mode"))
    store_policy, lazy_policy, geometry_policy, combined_policy = _build_policies(
        config,
        reverse_mode=reverse_mode,
        selected_track_count=len(pixel_ids),
    )
    control_precompile_wall_time_seconds = 0.0
    if context.get("worker_kind") == "control":
        request_count = math.ceil(
            len(pixel_ids) / int(provider.maximum_tracks_per_bundle)
        )
        artifact_bound = int(
            config["memory_limits_bytes"]["maximum_artifact_store_resident"]
        )
        store_policy = PaperKineticCompiledCpuArtifactStorePolicy(
            maximum_entries=request_count,
            maximum_resident_accounted_bytes=artifact_bound,
        )
        store = PaperKineticCompiledCpuArtifactStore(store_policy)
        precompile_started = time.perf_counter()
        provider = prepare_paper_kinetic_compiled_framewise_program_provider(
            provider,
            store,
            selected_track_ids=pixel_ids,
            maximum_artifact_accounted_bytes_per_entry=(
                artifact_bound // request_count
            ),
        )
        control_precompile_wall_time_seconds = (
            time.perf_counter() - precompile_started
        )
    else:
        store = PaperKineticCompiledCpuArtifactStore(store_policy)
    material_seed = tensor_inputs["site_rgba_f32_cpu"]
    material_initialization = prepare_paper_kinetic_p0_material_initialization(
        material_seed,
        provider.world.sites,
        initializer_generation_digest=initializer_digest,
        source_material_seed_digest=_canonical_sha256(
            {
                "target": inputs["target_generation_id"],
                "world": inputs["compiled_world_sha256"],
                "material_model": config["material"]["model"],
            }
        ),
    )
    learning_rate = _finite_float(
        config["optimizer"]["learning_rate"], name="optimizer learning rate"
    )
    material_state = prepare_paper_kinetic_fixed_site_material_state(
        material_initialization,
        provider.world,
        parameterization=PaperKineticFixedSiteMaterialParameterization(),
        optimizer_policy=PaperKineticFixedSiteMaterialSGDPolicy(
            color_learning_rate=learning_rate,
            density_learning_rate=learning_rate,
        ),
        device="cpu",
        maximum_material_state_logical_tensor_bytes=int(
            config["memory_limits_bytes"]["maximum_frame_invariant_live_logical"]
        ),
    )
    # The provider/state now own the only live geometry/material generation.
    # Release the driver's unused background/time/index tensors and the seed
    # views before any compile or optimizer lifecycle measurement begins.
    tensor_inputs.clear()
    del material_seed, material_initialization
    state = prepare_paper_kinetic_fixed_camera_combined_state(
        material_state,
        provider,
        store,
        maximum_combined_state_logical_tensor_bytes=(
            combined_policy.maximum_combined_state_logical_tensor_bytes
        ),
    )
    observations = _ReplayableSelectedObservationSource.seal(
        pixel_ids=pixel_ids,
        frame_indices=selected_frames,
        dataset_frame_count=len(frame_times),
        image_pixel_count=height * width,
    )
    if observations.expected_observation_count != inputs["expected_observation_count"]:
        raise ArithmeticError("selected observation count changed")
    manifest_digest = paper_kinetic_observation_manifest_digest(observations)
    cold_manifest = (
        None
        if context.get("worker_kind") == "control"
        else _selected_track_manifest(provider, pixel_ids)
    )
    return {
        "config": config,
        "inputs": inputs,
        "device": device,
        "provider": provider,
        "store": store,
        "store_policy": store_policy,
        "state": state,
        "observations": observations,
        "observation_manifest_digest": manifest_digest,
        "cold_manifest": cold_manifest,
        "lazy_policy": lazy_policy,
        "geometry_policy": geometry_policy,
        "combined_policy": combined_policy,
        "pixel_ids": pixel_ids,
        "selected_frames": selected_frames,
        "target_stream_manifest_sha256": target_stream_manifest,
        "dataset_generation_digest": dataset_generation_digest,
        "factory": factory,
        "retained_driver_input_tensor_bytes": 0,
        "control_precompile_wall_time_seconds": (
            control_precompile_wall_time_seconds
        ),
    }


def _run_one_primary_transaction(
    context: Mapping[str, Any], generation: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(generation, dict):
        raise TypeError("primary generation must support explicit ownership transfer")
    state = generation["state"]
    provider = generation["provider"]
    store = generation["store"]
    device: torch.device = generation["device"]
    config = generation["config"]
    transaction_started = time.perf_counter()
    if device.type == "mps":
        completion_fence = synchronize_mps_device_completion_fence
        completion_provenance = MPS_DEVICE_COMPLETION_FENCE_PROVENANCE
    else:
        completion_fence = lambda: None
        completion_provenance = "cpu-synchronous-fake-native-v1"
    background_generation_id = _canonical_sha256(
        {
            "adapter": ADAPTER_PROVENANCE,
            "background": config["material"]["background_rgb"],
        }
    )
    snapshot = snapshot_paper_kinetic_fixed_site_material_to_device(
        state.material_state,
        background_rgb_f32_cpu=torch.tensor(
            config["material"]["background_rgb"], dtype=torch.float32
        ),
        background_generation_id=background_generation_id,
        device=device,
        device_completion_fence=completion_fence,
        device_completion_fence_provenance=completion_provenance,
    )
    trainer_state = generation.get("trainer_state")
    if trainer_state is None:
        trainer_state = prepare_paper_kinetic_lazy_native_trainer_state(
            provider,
            device=device,
            initial_step_index=state.geometry_update_count,
        )
    material_bar = torch.empty_like(snapshot.site_rgba_f32_device)
    position_bar = torch.empty_like(state.positions0_f64, device="cpu")
    velocity_bar = torch.empty_like(state.velocities_f64, device="cpu")
    weight_bar = torch.empty_like(state.weight_coefficients_f64, device="cpu")
    captures: list[Any] = []
    core_started = time.perf_counter()
    result = run_paper_kinetic_lazy_native_full_geometry_step(
        trainer_state,
        provider,
        generation["observations"],
        step_index=state.geometry_update_count,
        expected_observation_count=(
            generation["observations"].expected_observation_count
        ),
        expected_observation_manifest_digest=(
            generation["observation_manifest_digest"]
        ),
        loss_normalization_id=_digest_parts(
            ADAPTER_PROVENANCE,
            "global-rgb-mean",
            generation["observations"].expected_observation_count * 3,
        ),
        material_generation_id=state.material_state.material_generation_id,
        geometry_generation_id=state.geometry_generation_id,
        background_generation_id=background_generation_id,
        global_site_rgba_f32=snapshot.site_rgba_f32_device,
        global_grad_site_rgba_f32=material_bar,
        grad_positions0_f64_cpu=position_bar,
        grad_velocities_f64_cpu=velocity_bar,
        grad_weight_coefficients_f64_cpu=weight_bar,
        background_rgb_f32=snapshot.background_rgb_f32_device,
        native_ops=context["native_ops"],
        maximum_samples_per_launch=int(
            config["spatial_streaming"]["maximum_samples_per_launch"]
        ),
        memory_policy=generation["lazy_policy"],
        full_geometry_memory_policy=generation["geometry_policy"],
        reverse_mode=str(context["mode"]),
        optimizer_update=captures.append,
        cone_tolerance=float(config["compiler"]["cone_tolerance"]),
    )
    core_step_wall_time = time.perf_counter() - core_started
    if captures != [result]:
        raise ArithmeticError("full-geometry coordinator issued the wrong callback")
    bridge = seal_paper_kinetic_lazy_full_geometry_device_gradient_receipt(
        state,
        provider,
        store,
        snapshot,
        result,
    )
    emit_parity = (
        len(generation["selected_frames"])
        == int(config["optimizer"]["lifecycle_frame_count"])
        and context["mode"] in {STAGED_SPARSE, FUSED_UNION_V2}
    )
    parity_gradient = None
    if emit_parity:
        parity_gradient = {
            "material_gradient": _flat_finite_values(
                bridge.grad_site_rgba_f32_cpu  # type: ignore[arg-type]
            ),
            "geometry_gradient": _flat_finite_values(
                bridge.grad_positions0_f64_cpu,  # type: ignore[arg-type]
                bridge.grad_velocities_f64_cpu,  # type: ignore[arg-type]
                bridge.grad_weight_coefficients_f64_cpu,  # type: ignore[arg-type]
            ),
        }
    # Transfer the current generation into the one-shot update call.  Keeping
    # the same state/provider/store in the caller mapping would retain the
    # retired tensors and compiled store after promotion.
    for key, expected in (
        ("state", state),
        ("provider", provider),
        ("store", store),
    ):
        if generation.pop(key) is not expected:
            raise ValueError(f"generation ownership for {key} changed")
    generation.pop("trainer_state", None)
    runtime_measurements: dict[str, float] = {}
    ready = apply_paper_kinetic_lazy_full_geometry_combined_sgd_transaction(
        state,
        provider,
        store,
        bridge,
        policy=generation["combined_policy"],
        cold_recompile_manifest=generation["cold_manifest"],
        fresh_store_policy=generation["store_policy"],
        runtime_measurements=runtime_measurements,
    )
    ready.assert_current()
    if set(runtime_measurements) != {"cold_cpu_compile_wall_time_seconds"}:
        raise ArithmeticError("combined transaction omitted cold-compile timing")
    update = ready.update_receipt.accounting()
    if parity_gradient is not None:
        parity_gradient["parameters_after_step"] = _flat_finite_values(
            ready.state.material_state.raw_color_f32,
            ready.state.material_state.raw_density_f32,
            ready.state.positions0_f64,
            ready.state.velocities_f64,
            ready.state.weight_coefficients_f64,
        )
        parity_gradient["loss"] = float(update["loss"])
    full_transaction_wall_time = time.perf_counter() - transaction_started
    return {
        "ready": ready,
        "update": update,
        "step_wall_time_seconds": full_transaction_wall_time,
        "core_forward_backward_wall_time_seconds": core_step_wall_time,
        "cold_cpu_compile_wall_time_seconds": runtime_measurements[
            "cold_cpu_compile_wall_time_seconds"
        ],
        "parity_payload": parity_gradient,
    }


def _run_control_transaction(
    context: Mapping[str, Any], generation: Mapping[str, Any]
) -> dict[str, Any]:
    """Run the compile-once, framewise-replay control through the same native core."""

    transaction_started = time.perf_counter()
    config = generation["config"]
    state = generation["state"]
    provider = generation["provider"]
    store = generation["store"]
    device: torch.device = generation["device"]
    if device.type == "mps":
        completion_fence = synchronize_mps_device_completion_fence
        completion_provenance = MPS_DEVICE_COMPLETION_FENCE_PROVENANCE
    else:
        completion_fence = lambda: None
        completion_provenance = "cpu-synchronous-fake-native-v1"
    background_generation_id = _canonical_sha256(
        {
            "adapter": ADAPTER_PROVENANCE,
            "background": config["material"]["background_rgb"],
        }
    )
    snapshot = snapshot_paper_kinetic_fixed_site_material_to_device(
        state.material_state,
        background_rgb_f32_cpu=torch.tensor(
            config["material"]["background_rgb"], dtype=torch.float32
        ),
        background_generation_id=background_generation_id,
        device=device,
        device_completion_fence=completion_fence,
        device_completion_fence_provenance=completion_provenance,
    )
    result = run_paper_kinetic_compiled_framewise_full_geometry_control(
        state,
        provider,
        store,
        selected_frame_indices=generation["selected_frames"],
        selected_track_ids=generation["pixel_ids"],
        global_site_rgba_f32=snapshot.site_rgba_f32_device,
        background_rgb_f32=snapshot.background_rgb_f32_device,
        background_generation_id=background_generation_id,
        native_ops=context["native_ops"],
        maximum_samples_per_launch=int(
            config["spatial_streaming"]["maximum_samples_per_launch"]
        ),
        cone_tolerance=float(config["compiler"]["cone_tolerance"]),
        memory_policy=generation["lazy_policy"],
        full_geometry_memory_policy=generation["geometry_policy"],
        combined_sgd_policy=generation["combined_policy"],
        device_completion_fence=completion_fence,
        device_completion_fence_provenance=completion_provenance,
        emit_parity_payload=(len(generation["selected_frames"]) == 8),
    )
    result.assert_current()
    return {
        "result": result,
        "control_transaction_wall_time_seconds": (
            time.perf_counter() - transaction_started
        ),
    }


_CONTROL_LOGICAL_ACCOUNTING_KEYS = (
    "global_material_bar_logical_tensor_bytes",
    "global_geometry_bar_logical_tensor_bytes",
    "global_bar_and_loss_logical_tensor_bytes",
    "frame_material_bar_logical_tensor_bytes",
    "frame_geometry_bar_logical_tensor_bytes",
    "frame_material_readback_and_loss_logical_tensor_bytes",
    "frame_coordinator_visible_logical_tensor_bytes_upper_bound",
    "frame_geometry_bridge_visible_logical_tensor_bytes_upper_bound",
    "compiled_program_store_resident_accounted_bytes",
    "compiled_program_store_peak_resident_accounted_bytes",
    "combined_live_state_logical_tensor_bytes",
    "maximum_frame_local_logical_tensor_bytes_upper_bound",
    "expensive_live_logical_and_accounted_peak_upper_bound_bytes",
    "peak_lane_resident_logical_tensor_bytes",
    "peak_active_node_state_tensor_bytes",
    "peak_sample_launch_tensor_bytes",
    "peak_decoded_frame_scratch_upper_bound_bytes",
    "peak_selected_frame_target_tensor_upper_bound_bytes",
    "peak_coordinator_visible_live_tensor_upper_bound_bytes",
    "maximum_geometry_bridge_visible_peak_logical_tensor_bytes",
    "persistent_frame_tensor_bytes",
    "persistent_sample_tensor_bytes",
    "persistent_target_tensor_bytes",
    "persistent_prediction_tensor_bytes",
    "optimizer_history_tensor_bytes",
)


def _control_row(
    context: Mapping[str, Any], generation: Mapping[str, Any], transaction: Mapping[str, Any]
) -> dict[str, Any]:
    """Keep the fair control's three sealed receipts literal in the evidence row."""

    config = generation["config"]
    inputs = generation["inputs"]
    state = generation["state"]
    provider = generation["provider"]
    result = transaction["result"]
    precompile_receipt = _public_receipt_fields(result.precompile_receipt)
    update_receipt = _public_receipt_fields(result.update_receipt)
    accounting = dict(result.accounting)
    missing_logical = tuple(
        key for key in _CONTROL_LOGICAL_ACCOUNTING_KEYS if key not in accounting
    )
    if missing_logical:
        raise KeyError(
            "compiled-framewise accounting omitted logical fields: "
            + ", ".join(missing_logical)
        )
    finite_update_fields = (
        "loss",
        "material_gradient_l2_norm",
        "position_gradient_l2_norm",
        "velocity_gradient_l2_norm",
        "weight_gradient_l2_norm",
        "raw_color_parameter_delta_l2_norm",
        "raw_density_parameter_delta_l2_norm",
        "positions0_parameter_delta_l2_norm",
        "velocities_parameter_delta_l2_norm",
        "weight_coefficients_parameter_delta_l2_norm",
    )
    if not all(
        math.isfinite(float(update_receipt[key]))
        for key in finite_update_fields
    ):
        raise FloatingPointError("compiled-framewise update receipt is non-finite")
    control_result_receipt = {
        key: getattr(result, key)
        for key in (
            "provenance",
            "runtime_status",
            "native_runtime_verified",
            "allocator_peak_measured",
            "generation_digest",
        )
    }
    structure = {
        "image_height": int(config["image"]["height"]),
        "image_width": int(config["image"]["width"]),
        "dataset_frame_count": int(config["temporal_grid"]["dataset_frame_count"]),
        "world_site_count": state.site_count,
        "weight_coefficient_count": int(state.weight_coefficients_f64.shape[1]),
        "fixed_track_count": len(generation["pixel_ids"]),
        "selected_frame_count": len(generation["selected_frames"]),
        "expected_observation_count": (
            generation["observations"].expected_observation_count
        ),
        "loss_element_count": inputs["loss_element_count"],
        "node_count": int(config["compiler"]["node_count"]),
        "maximum_sites_per_track_compile": int(
            config["compiler"]["maximum_sites_per_track_compile"]
        ),
        "compile_certification_mode": config["compiler"]["certification_mode"],
        "track_manifest_sha256": inputs["track_manifest_sha256"],
        "camera_program_sha256": inputs["camera_program_sha256"],
        "target_teacher_sha256": inputs["target_generation_id"],
        "target_stream_manifest_sha256": generation[
            "target_stream_manifest_sha256"
        ],
        "observation_manifest_sha256": generation[
            "observation_manifest_digest"
        ],
        "compiled_world_sha256": inputs["compiled_world_sha256"],
        "physical_grid_sha256": inputs["physical_grid_sha256"],
        "camera_grid_sha256": inputs["camera_grid_sha256"],
        "spatial_block_manifest_sha256": inputs[
            "spatial_block_manifest_sha256"
        ],
        "provider_generation_sha256": provider.generation_digest,
        "factory_generation_sha256": generation["factory"].generation_digest,
        "optimizer_policy_sha256": generation[
            "combined_policy"
        ].generation_digest,
        "loss": config["target_source"]["loss"],
    }
    return {
        "execution": {
            "adapter_provenance": ADAPTER_PROVENANCE,
            "control_result_receipt": control_result_receipt,
            "update_receipt": update_receipt,
            "adapter_measurements": {
                "continuous_precompile_measurement_count": 1,
                "continuous_precompile_wall_time_seconds": generation[
                    "control_precompile_wall_time_seconds"
                ],
                "control_transaction_measurement_count": 1,
                "control_transaction_wall_time_seconds": transaction[
                    "control_transaction_wall_time_seconds"
                ],
            },
        },
        "structure": structure,
        "work": {
            "precompile_receipt": precompile_receipt,
            "accounting": accounting,
        },
        "memory": {
            "logical_accounting": {
                key: accounting[key] for key in _CONTROL_LOGICAL_ACCOUNTING_KEYS
            },
            "measured_peak_fields_producer_owned": True,
            "logical_bounds_are_measured_peaks": False,
        },
        "quality": {
            "finite": True,
            "loss": update_receipt["loss"],
            "post_update_loss_measured": False,
        },
        "preflight": dict(_mapping(context.get("preflight"), name="control preflight")),
    }


def _checkpoint_transaction(
    context: Mapping[str, Any],
    generation: Mapping[str, Any],
    transaction: Mapping[str, Any],
) -> dict[str, Any]:
    ready = transaction["ready"]
    checkpoint = checkpoint_paper_kinetic_fixed_camera_combined_state(
        ready.state,
        ready.provider,
        ready.artifact_store,
        manifest=ready.manifest,
        recompile_receipt=ready.recompile_receipt,
        policy=generation["combined_policy"],
        initializer_generation_digest=ready.provider.initializer_generation_digest,
    )
    checkpoint.assert_current()
    output_dir = Path(str(context.get("worker_output_dir", ""))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (
        "worldfoam_combined_f8_"
        f"repeat_{int(context.get('repeat_index', 0))}_step1.pt"
    )
    torch.save(checkpoint.payload(), path)
    digest = _file_sha256(path)
    result = {
        "checkpoint_path": str(path),
        "checkpoint_sha256": digest,
        "checkpoint_generation_digest": checkpoint.generation_digest,
        "checkpoint_tensor_bytes": checkpoint.checkpoint_tensor_bytes,
        "live_state_logical_tensor_bytes_at_checkpoint": (
            checkpoint.live_state_logical_tensor_bytes_at_checkpoint
        ),
        "state_checkpoint_logical_tensor_bytes": (
            checkpoint.state_checkpoint_logical_tensor_bytes
        ),
        "state_checkpoint_payload_peak_logical_tensor_bytes": (
            checkpoint.state_checkpoint_payload_peak_logical_tensor_bytes
        ),
    }
    del checkpoint
    return result


def _next_generation(
    generation: Mapping[str, Any], transaction: Mapping[str, Any]
) -> dict[str, Any]:
    ready = transaction["ready"]
    trainer_state = (
        claim_paper_kinetic_lazy_full_geometry_ready_generation_for_next_step(
            ready,
            caller_retained_untracked_logical_and_accounted_bytes=0,
            device=generation["device"],
        )
    )
    return {
        **dict(generation),
        "provider": ready.provider,
        "store": ready.artifact_store,
        "state": ready.state,
        "trainer_state": trainer_state,
        "cold_manifest": _selected_track_manifest(
            ready.provider,
            generation["pixel_ids"],
        ),
    }


_LIFECYCLE_DELTA_KEYS = (
    "raw_color_parameter_delta_l2_norm",
    "raw_density_parameter_delta_l2_norm",
    "positions0_parameter_delta_l2_norm",
    "velocities_parameter_delta_l2_norm",
    "weight_coefficients_parameter_delta_l2_norm",
)


def _transaction_python_evidence(
    transaction: Mapping[str, Any], *, name: str
) -> dict[str, Any]:
    """Copy one completed transaction into tensor-free lifecycle evidence."""

    ready = transaction.get("ready")
    if ready is None:
        raise ValueError(f"{name} transaction lost its ready generation")
    parity = _mapping(
        transaction.get("parity_payload"), name=f"{name} parity payload"
    )
    update = _mapping(transaction.get("update"), name=f"{name} update")
    evidence = {
        "loss_pre_update": float(update["loss"]),
        "gradient_sha256": _parity_gradient_digest(parity),
        "parameters_after_step_sha256": _canonical_sha256(
            parity["parameters_after_step"]
        ),
        "state_sha256": _combined_state_content_digest(ready.state),
        "update_receipt_sha256": str(update["generation_digest"]),
        "parameter_delta_l2": {
            key: float(update[key]) for key in _LIFECYCLE_DELTA_KEYS
        },
    }
    # The receipt generation digest intentionally binds provider/store lineage,
    # which differs across fresh processes and after a valid checkpoint restore.
    # This portable content digest instead binds the complete trainable
    # parameter vector and the optimizer inputs/deltas.
    evidence["update_content_sha256"] = _canonical_sha256(
        {
            key: evidence[key]
            for key in (
                "loss_pre_update",
                "gradient_sha256",
                "parameters_after_step_sha256",
                "parameter_delta_l2",
            )
        }
    )
    return evidence


def _prepare_restart_generation(
    context: Mapping[str, Any], payload: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _mapping(context.get("config"), name="restart config")
    raw_inputs = context.get("inputs")
    if not isinstance(raw_inputs, dict):
        raise TypeError("restart inputs must be an ownership-transfer dictionary")
    inputs = raw_inputs
    height = int(config["image"]["height"])
    width = int(config["image"]["width"])
    frame_times = tuple(
        float(value)
        for value in inputs["full_physical_time_grid_f64_cpu"].tolist()
    )
    selected_frames = tuple(
        int(value) for value in inputs["selected_frame_indices_i64_cpu"].tolist()
    )
    pixel_ids = tuple(int(value) for value in inputs["pixel_ids_i64_cpu"].tolist())
    rows = tuple(int(value) for value in inputs["rows_i64_cpu"].tolist())
    columns = tuple(int(value) for value in inputs["columns_i64_cpu"].tolist())
    target_stream_manifest = _validate_driver_target_chunks(
        inputs,
        config,
        frame_indices=selected_frames,
        pixel_ids=pixel_ids,
        rows=rows,
        columns=columns,
        frame_times=frame_times,
    )
    # Restart reconstructs world/material exclusively from the checkpoint.
    # Drop the driver's otherwise-unused procedural initialization tensors
    # before cloning/restoring the checkpoint payload.
    unused_tensor_inputs = _take_driver_tensor_inputs(inputs)
    unused_tensor_inputs.clear()
    device = torch.device(str(context.get("backend")))
    target_provider = PowerFoamTargetProvider(
        source=_ProceduralDirectSelectedPixelTargetSource(
            frame_times=frame_times,
            height=height,
            width=width,
            target_generation_id=str(inputs["target_generation_id"]),
        ),
        device=device,
    )
    camera = _camera_from_config(config)
    ray_provider = PowerFoamRayProvider(
        cameras=(tuple(camera for _ in frame_times),),
        height=height,
        width=width,
        device=device,
    )
    compiler = config["compiler"]
    factory = prepare_paper_kinetic_active_p0_track_program_factory(
        PaperKineticActiveP0TrackProgramFactoryConfig(
            near=float(compiler["near"]),
            far=float(compiler["far"]),
            node_count=int(compiler["node_count"]),
            maximum_sites_per_track_compile=int(
                compiler["maximum_sites_per_track_compile"]
            ),
            maximum_charts_per_track=int(compiler["maximum_charts_per_track"]),
            maximum_owner_runs_per_chart=int(
                compiler["maximum_owner_runs_per_chart"]
            ),
            rank_selection_provenance=str(compiler["rank_selection_provenance"]),
        )
    )
    store_policy, lazy_policy, geometry_policy, combined_policy = _build_policies(
        config,
        reverse_mode=FUSED_UNION_V2,
        selected_track_count=len(pixel_ids),
    )
    restored = restore_paper_kinetic_fixed_camera_combined_generation_from_payload(
        payload,
        expected_world_site_count=int(config["procedural_world"]["site_count"]),
        expected_combined_sgd_policy=combined_policy,
        target_provider=target_provider,
        ray_provider=ray_provider,
        frame_times=frame_times,
        maximum_observations_per_bundle=int(
            config["spatial_streaming"]["maximum_observations_per_chunk"]
        ),
        maximum_rows_per_native_block=int(
            config["spatial_streaming"]["maximum_rows_per_native_block"]
        ),
        program_factory=factory,
        fresh_store_policy=store_policy,
        device="cpu",
    )
    restore_receipt = {
        name: value
        for name, value in restored.restore_receipt.__dict__.items()
        if name != "_seal"
    }
    payload.clear()
    trainer_state = (
        claim_paper_kinetic_fixed_camera_restored_ready_generation_for_lazy_native_next_step(
            restored,
            caller_retained_untracked_logical_and_accounted_bytes=0,
            device=device,
        )
    )
    observations = _ReplayableSelectedObservationSource.seal(
        pixel_ids=pixel_ids,
        frame_indices=selected_frames,
        dataset_frame_count=len(frame_times),
        image_pixel_count=height * width,
    )
    manifest_digest = paper_kinetic_observation_manifest_digest(observations)
    generation = {
        "config": config,
        "inputs": inputs,
        "device": device,
        "provider": restored.provider,
        "store": restored.artifact_store,
        "store_policy": store_policy,
        "state": restored.state,
        "trainer_state": trainer_state,
        "observations": observations,
        "observation_manifest_digest": manifest_digest,
        "cold_manifest": restored.manifest,
        "lazy_policy": lazy_policy,
        "geometry_policy": geometry_policy,
        "combined_policy": combined_policy,
        "pixel_ids": pixel_ids,
        "selected_frames": selected_frames,
        "target_stream_manifest_sha256": target_stream_manifest,
        "factory": factory,
        "retained_driver_input_tensor_bytes": 0,
    }
    return generation, restore_receipt


def _run_restart_worker(context: Mapping[str, Any]) -> Mapping[str, Any]:
    config = _mapping(context.get("config"), name="restart config")
    if (
        context.get("mode") != FUSED_UNION_V2
        or int(context.get("frame_count", 0))
        != int(config["optimizer"]["lifecycle_frame_count"])
    ):
        raise ValueError("auxiliary lifecycle worker requires fused lifecycle F")
    fresh_inputs_factory = context.get("fresh_inputs_factory")
    if not callable(fresh_inputs_factory):
        raise TypeError("auxiliary lifecycle worker requires a fresh input factory")

    # This auxiliary worker is deliberately not a scaling row.  It owns the
    # full step1/checkpoint/step2 reference and restart comparison so that the
    # independently measured primary F8 worker remains exactly one step.
    first_generation = _prepare_initial_generation(context)
    first = _run_one_primary_transaction(context, first_generation)
    first_evidence = _transaction_python_evidence(first, name="auxiliary step1")
    checkpoint = _checkpoint_transaction(context, first_generation, first)
    second_generation = _next_generation(first_generation, first)
    first.pop("ready")
    first.clear()
    first_generation.clear()

    uninterrupted = _run_one_primary_transaction(context, second_generation)
    uninterrupted_evidence = _transaction_python_evidence(
        uninterrupted, name="uninterrupted step2"
    )
    if not uninterrupted_evidence["loss_pre_update"] < first_evidence[
        "loss_pre_update"
    ]:
        raise ArithmeticError(
            "auxiliary step1 SGD did not lower the step2 pre-update loss"
        )

    # Prove that no first-world owner remains before reconstructing the
    # checkpoint world.  The evidence copied above contains only Python
    # scalars, strings, and tuples/lists of scalars.
    uninterrupted_ready = uninterrupted["ready"]
    released_world_refs = tuple(
        weakref.ref(value)
        for value in (
            uninterrupted_ready,
            uninterrupted_ready.state,
            uninterrupted_ready.provider,
            uninterrupted_ready.artifact_store,
        )
    )
    del uninterrupted_ready
    uninterrupted.clear()
    second_generation.clear()
    if any(reference() is not None for reference in released_world_refs):
        raise RuntimeError(
            "auxiliary uninterrupted world remained live before checkpoint restore"
        )

    checkpoint_path = Path(str(checkpoint["checkpoint_path"])).resolve()
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    if checkpoint_sha256 != checkpoint["checkpoint_sha256"]:
        raise ValueError("auxiliary lifecycle checkpoint file digest changed")
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise TypeError("auxiliary checkpoint file must contain one mapping payload")
    fresh_inputs = fresh_inputs_factory()
    if not isinstance(fresh_inputs, dict):
        raise TypeError("auxiliary fresh input factory must return a dictionary")
    restore_context = {**dict(context), "inputs": fresh_inputs}
    restored_generation, restore_receipt = _prepare_restart_generation(
        restore_context, loaded
    )
    restored = _run_one_primary_transaction(restore_context, restored_generation)
    restored_evidence = _transaction_python_evidence(
        restored, name="restored step2"
    )
    for key in (
        "loss_pre_update",
        "gradient_sha256",
        "parameters_after_step_sha256",
        "state_sha256",
        "update_content_sha256",
        "parameter_delta_l2",
    ):
        if uninterrupted_evidence[key] != restored_evidence[key]:
            raise ArithmeticError(
                f"auxiliary restored step2 {key} differs from uninterrupted step2"
            )
    restored.clear()
    restored_generation.clear()
    return {
        "native_ops_used": context["native_ops"],
        "restart_result": {
            **checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
            "auxiliary_step_1": first_evidence,
            "uninterrupted_step_2": uninterrupted_evidence,
            "restored_step_2": restored_evidence,
            "restore_receipt": restore_receipt,
            "auxiliary_optimizer_mutation_count": 3,
            "uninterrupted_process_optimizer_mutation_count": 2,
            "fresh_restart_optimizer_mutation_count": 1,
            "maximum_simultaneously_retained_world_count": 1,
            "uninterrupted_world_released_before_restore": True,
            "lifecycle_executed_outside_primary_scaling_worker": True,
        },
    }


def _primary_row(
    context: Mapping[str, Any],
    generation: Mapping[str, Any],
    transaction: Mapping[str, Any],
) -> dict[str, Any]:
    config = generation["config"]
    inputs = generation["inputs"]
    update = transaction["update"]
    core = dict(update["core_accounting"])
    compiler_accounting_keys = (
        "compile_track_count",
        "compiler_work_receipt_count",
        "compiler_work_receipt_bundle_count",
        "compiler_work_receipt_chain_link_count",
        "root_complement_witness_count",
        "candidate_source_attempt_count",
        "all_site_witness_check_count",
        "unique_pair_difference_count",
        "per_witness_candidate_bound_verified",
        "exhaustive_triple_enumeration_used",
        "requested_frame_sampling_used",
        "active_compiler_accounting_complete",
        "all_track_receipt_digests_verified",
        "compiler_work_receipt_provenance",
        "compiler_work_receipt_chain_digest",
        "retained_compiled_program_count",
        "retained_compiler_receipt_entry_count",
        "retained_compiler_tensor_bytes",
        "compiler_receipt_state_scaling",
    )
    missing_compiler_fields = tuple(
        key for key in compiler_accounting_keys if key not in core
    )
    if missing_compiler_fields:
        raise KeyError(
            "lazy core omitted compiler receipt fields: "
            + ", ".join(missing_compiler_fields)
        )
    if (
        core["active_compiler_accounting_complete"] is not True
        or core["all_track_receipt_digests_verified"] is not True
        or core["per_witness_candidate_bound_verified"] is not True
        or core["exhaustive_triple_enumeration_used"] is not False
        or core["requested_frame_sampling_used"] is not False
        or any(
            core[key] != 0
            for key in (
                "retained_compiled_program_count",
                "retained_compiler_receipt_entry_count",
                "retained_compiler_tensor_bytes",
            )
        )
    ):
        raise ArithmeticError("active-compiler whole-step receipt is incomplete")
    ready = transaction["ready"]
    gradient_update = {
        key: update[key]
        for key in (
            "grad_site_rgba_l2_norm",
            "grad_site_rgba_max_abs",
            "grad_positions0_l2_norm",
            "grad_positions0_max_abs",
            "grad_velocities_l2_norm",
            "grad_velocities_max_abs",
            "grad_weight_coefficients_l2_norm",
            "grad_weight_coefficients_max_abs",
            "raw_color_parameter_delta_l2_norm",
            "raw_color_parameter_delta_max_abs",
            "raw_density_parameter_delta_l2_norm",
            "raw_density_parameter_delta_max_abs",
            "positions0_parameter_delta_l2_norm",
            "positions0_parameter_delta_max_abs",
            "velocities_parameter_delta_l2_norm",
            "velocities_parameter_delta_max_abs",
            "weight_coefficients_parameter_delta_l2_norm",
            "weight_coefficients_parameter_delta_max_abs",
        )
    }
    required_nonzero = tuple(
        value
        for key, value in gradient_update.items()
        if key.endswith("_l2_norm")
    )
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in required_nonzero):
        raise ArithmeticError("ablation requires nonzero material and geometry updates")
    total_delta = math.sqrt(
        sum(
            float(gradient_update[key]) ** 2
            for key in (
                "raw_color_parameter_delta_l2_norm",
                "raw_density_parameter_delta_l2_norm",
                "positions0_parameter_delta_l2_norm",
                "velocities_parameter_delta_l2_norm",
                "weight_coefficients_parameter_delta_l2_norm",
            )
        )
    )
    state_accounting = ready.state.accounting(
        requested_frame_count=len(generation["selected_frames"])
    )
    execution = {
        "adapter_provenance": ADAPTER_PROVENANCE,
        "real_native_spatial_block_coordinator": True,
        "fake_native_backend": False,
        "native_runtime_executed": True,
        "core_native_runtime_verified": bool(core["native_runtime_verified"]),
        "full_geometry_trainable": True,
        "material_trainable": True,
        "fixed_camera": True,
        "all_competitor_active_owner_certified": True,
        "heuristic_spatial_culling_used": False,
        "post_certification_compact_device_lowering": True,
        "direct_selected_pixel_target_stream": (
            core["selected_pixel_read_acceptance_capable"] is True
            and core["full_frame_target_materialization_count"] == 0
        ),
        "full_frame_target_materialization_used": (
            core["full_frame_target_materialization_count"] != 0
        ),
        "cpu_optimizer_mutation_count": 1,
        "geometry_mutation_count": 1,
        "material_device_gradient_receipt_count": 1,
        "geometry_d2h_receipt_count": update["geometry_d2h_receipt_count"],
        "combined_optimizer_authorization_count": 1,
        "gradient_update": gradient_update,
        "cpu_optimizer_parameter_delta_l2": total_delta,
        "step_wall_time_seconds": transaction["step_wall_time_seconds"],
        "core_forward_backward_wall_time_seconds": transaction[
            "core_forward_backward_wall_time_seconds"
        ],
        "cold_cpu_compile_measured": True,
        "cold_cpu_compile_measurement_count": 1,
        "cold_cpu_compile_wall_time_seconds": transaction[
            "cold_cpu_compile_wall_time_seconds"
        ],
        "bridge_receipt_generation_digest": update[
            "bridge_receipt_generation_digest"
        ],
        "geometry_d2h_receipt_generation_digests": update[
            "geometry_d2h_receipt_generation_digests"
        ],
        "authorization_generation_digest": update[
            "authorization_generation_digest"
        ],
        "combined_update_receipt_generation_digest": update[
            "generation_digest"
        ],
        "stale_provider_store_retirement_count": update[
            "stale_provider_store_retirement_count"
        ],
        "provider_store_retirement_receipt_chain_sha256": update[
            "provider_store_retirement_receipt_chain_sha256"
        ],
        "fresh_selected_track_recompile_count": update[
            "fresh_full_interval_recompile_count"
        ],
        "fresh_selected_track_recompile_receipt_sha256": update[
            "fresh_full_interval_recompile_receipt_generation_digest"
        ],
        "fresh_selected_track_recompile_request_count": update[
            "cold_compiled_request_count"
        ],
        "autograd_graph_retained": False,
        "repeat_changes_world_or_data": False,
        "dataset_is_procedural_synthetic": True,
        "public_quality_evidence": False,
        "worker_measurement_scope": "single_optimizer_step_scaling_row_v2",
        "worker_measurement_covers_checkpoint_and_uninterrupted_step_2": False,
        "parity_payload_scope": "single_step_pre_update_gradient_and_post_update_parameters",
    }
    structure = {
        "image_height": int(config["image"]["height"]),
        "image_width": int(config["image"]["width"]),
        "dataset_frame_count": int(config["temporal_grid"]["dataset_frame_count"]),
        "world_site_count": ready.state.site_count,
        "weight_coefficient_count": int(
            ready.state.weight_coefficients_f64.shape[1]
        ),
        "fixed_track_count": len(generation["pixel_ids"]),
        "selected_frame_count": len(generation["selected_frames"]),
        "expected_observation_count": (
            generation["observations"].expected_observation_count
        ),
        "loss_element_count": inputs["loss_element_count"],
        "global_loss_denominator": core["global_loss_element_count"],
        "node_count": int(config["compiler"]["node_count"]),
        "maximum_sites_per_track_compile": int(
            config["compiler"]["maximum_sites_per_track_compile"]
        ),
        "compile_certification_mode": config["compiler"]["certification_mode"],
        "track_manifest_sha256": inputs["track_manifest_sha256"],
        "camera_program_sha256": inputs["camera_program_sha256"],
        "target_teacher_sha256": inputs["target_generation_id"],
        "target_stream_manifest_sha256": generation[
            "target_stream_manifest_sha256"
        ],
        "observation_manifest_sha256": generation[
            "observation_manifest_digest"
        ],
        "compiled_world_sha256": inputs["compiled_world_sha256"],
        "physical_grid_sha256": inputs["physical_grid_sha256"],
        "camera_grid_sha256": inputs["camera_grid_sha256"],
        "spatial_block_manifest_sha256": inputs[
            "spatial_block_manifest_sha256"
        ],
        "provider_generation_before_sha256": update[
            "old_provider_generation_digest"
        ],
        "provider_generation_after_sha256": update[
            "new_provider_generation_digest"
        ],
        "factory_generation_sha256": generation["factory"].generation_digest,
        "selected_track_recompile_manifest_sha256": generation[
            "cold_manifest"
        ].generation_digest,
        "optimizer_policy_sha256": generation[
            "combined_policy"
        ].generation_digest,
        "loss": config["target_source"]["loss"],
    }
    work = {
        "core_accounting": core,
        **{key: core[key] for key in compiler_accounting_keys},
        "spatial_bundle_count": core["spatial_bundle_count"],
        "eligible_native_block_count": core["eligible_native_block_count"],
        "active_native_block_count": core["active_native_block_count"],
        "native_node_forward_launch_count": core[
            "native_node_forward_launch_count"
        ],
        "native_material_word_vjp_launch_count": core[
            "native_material_word_vjp_launch_count"
        ],
        "native_full_geometry_vjp_launch_count": core[
            "native_full_geometry_vjp_launch_count"
        ],
        "native_fused_union_v2_transaction_count": core[
            "native_fused_union_v2_transaction_count"
        ],
        "ordered_word_node_interactions": core[
            "ordered_word_node_interactions"
        ],
        "streamed_sample_count": core["streamed_sample_count"],
        "sample_to_node_linear_interactions": core[
            "sample_to_node_linear_interactions"
        ],
        "sample_to_node_dense_fallback_interactions": core[
            "sample_to_node_dense_fallback_interactions"
        ],
        "selected_pixel_read_call_count": core["selected_pixel_read_call_count"],
        "direct_selected_pixel_observation_count": core[
            "direct_selected_pixel_observation_count"
        ],
        "camera_ray_slice_work_count": core["camera_ray_slice_work_count"],
        "camera_ray_slice_scalar_count": core[
            "camera_ray_slice_scalar_count"
        ],
        "fresh_selected_track_recompile_request_count": update[
            "cold_compiled_request_count"
        ],
        "fresh_selected_track_recompile_track_count": generation[
            "cold_manifest"
        ].track_count,
    }
    memory = {
        "retained_driver_input_tensor_bytes": generation[
            "retained_driver_input_tensor_bytes"
        ],
        "combined_live_state_logical_tensor_bytes": state_accounting[
            "total_persistent_tensor_bytes"
        ],
        "material_state_logical_tensor_bytes": state_accounting[
            "material_state_tensor_bytes"
        ],
        "trainable_geometry_state_logical_tensor_bytes": state_accounting[
            "geometry_tensor_bytes"
        ],
        "global_material_bar_logical_tensor_bytes": core[
            "caller_global_material_bar_tensor_bytes"
        ],
        "global_geometry_bar_logical_tensor_bytes": core[
            "global_cpu_geometry_bar_logical_tensor_bytes"
        ],
        "peak_lane_resident_logical_tensor_bytes": core[
            "peak_lane_resident_logical_tensor_bytes"
        ],
        "peak_active_node_state_logical_tensor_bytes": core[
            "peak_active_node_state_tensor_bytes"
        ],
        "peak_sample_launch_logical_tensor_bytes": core[
            "peak_sample_launch_tensor_bytes"
        ],
        "peak_coordinator_visible_logical_tensor_upper_bound_bytes": core[
            "peak_coordinator_visible_live_tensor_upper_bound_bytes"
        ],
        "maximum_geometry_bridge_visible_logical_tensor_bytes": core[
            "maximum_geometry_bridge_visible_peak_logical_tensor_bytes"
        ],
        "combined_update_authorization_logical_tensor_bytes": update[
            "authorization_logical_tensor_bytes"
        ],
        "combined_update_transaction_tracked_logical_and_store_accounted_upper_bound_bytes": update[
            "transaction_tracked_logical_and_store_accounted_bytes_upper_bound"
        ],
        "persistent_frame_tensor_bytes": core["persistent_frame_tensor_bytes"],
        "persistent_sample_tensor_bytes": core["persistent_sample_tensor_bytes"],
        "persistent_target_tensor_bytes": core["persistent_target_tensor_bytes"],
        "persistent_prediction_tensor_bytes": core[
            "persistent_prediction_tensor_bytes"
        ],
        "optimizer_history_tensor_bytes": state_accounting[
            "optimizer_history_tensor_bytes"
        ],
        "measured_peak_fields_producer_owned": True,
        "logical_bounds_are_measured_peaks": False,
    }
    quality = {
        "finite": True,
        "loss_before_update": float(update["loss"]),
        "post_update_loss_measured": False,
        "all_material_and_geometry_gradient_l2_norms_nonzero": True,
        "all_material_and_geometry_parameter_delta_l2_norms_nonzero": True,
    }
    return {
        "execution": execution,
        "structure": structure,
        "work": work,
        "memory": memory,
        "quality": quality,
    }


def run_worldfoam_training_memory_ablation_adapter(
    context: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Execute one receipt-backed primary, control, or restart transaction."""

    context = _mapping(context, name="ablation adapter context")
    worker_kind = str(context.get("worker_kind", ""))
    if worker_kind == "restart":
        if context.get("native_ops") is None:
            raise ValueError("restart adapter requires producer-attested native ops")
        return _run_restart_worker(context)
    if worker_kind == "control":
        if context.get("native_ops") is None:
            raise ValueError("control adapter requires producer-attested native ops")
        config = _mapping(context.get("config"), name="control config")
        if context.get("mode") != config["ablation"]["control_mode"]:
            raise ValueError("control adapter requires the checked-in control mode")
        generation = _prepare_initial_generation(context)
        control_transaction = _run_control_transaction(context, generation)
        control_result = control_transaction["result"]
        result: dict[str, Any] = {
            "native_ops_used": context["native_ops"],
            "row": _control_row(context, generation, control_transaction),
        }
        if control_result.parity_payload is not None:
            result["parity_payload"] = dict(control_result.parity_payload)
        return result
    if worker_kind != "primary":
        raise ValueError("ablation adapter worker_kind must be primary, control, or restart")
    mode = str(context.get("mode", ""))
    if mode not in {STAGED_SPARSE, FUSED_UNION_V2}:
        raise ValueError("primary ablation mode must be staged_sparse or fused_union_v2")
    if context.get("native_ops") is None:
        raise ValueError("ablation adapter requires producer-attested native ops")
    generation = _prepare_initial_generation(context)
    transaction = _run_one_primary_transaction(context, generation)
    row = _primary_row(context, generation, transaction)
    parity_payload = transaction["parity_payload"]
    result: dict[str, Any] = {
        "native_ops_used": context["native_ops"],
        "row": row,
    }
    if parity_payload is not None:
        # Detach the bounded parity vectors from the transaction-owned result;
        # lifecycle work, when requested, runs in a separate auxiliary worker.
        result["parity_payload"] = dict(parity_payload)
    return result


__all__ = (
    "ADAPTER_PROVENANCE",
    "run_worldfoam_training_memory_ablation_adapter",
)
