"""Sealed production inputs for the WorldFoam G4 public-quality executors.

The public row worker owns datasets and protocol scheduling.  WorldFoam needs
additional retained-depth objects that Gaussian executors must not know about:
split-specific mapped targets and calibrated rays, a deterministic kinetic
world initializer, the exact compiler factory, and every optimizer/memory
policy.  This module is the neutral hand-off boundary between those owners.

There are deliberately no defaults here.  In particular, a route executor may
not silently replace a missing public provider with the procedural G6 memory
fixture or choose a more permissive memory budget at runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
from paper_kinetic_fixed_camera_combined_state import (
    PaperKineticFixedCameraCombinedSGDPolicy,
)
from paper_kinetic_fixed_site_material_state import (
    PaperKineticFixedSiteMaterialParameterization,
    PaperKineticFixedSiteMaterialSGDPolicy,
)
from paper_kinetic_lazy_full_geometry_step import (
    FULL_GEOMETRY_REVERSE_MODES,
    PaperKineticLazyFullGeometryMemoryPolicy,
)
from paper_kinetic_runtime_paths import ensure_worldfoam_lane2_research_path
from paper_kinetic_world_initializer import (
    PaperKineticPointCloudWorldInitializer,
)
from powerfoam_training_data import (
    MappedRgb8PowerFoamTargetSource,
    PowerFoamRayProvider,
    PowerFoamTargetProvider,
)


ensure_worldfoam_lane2_research_path()

from kinetic_compiled_cpu_artifact_store import (  # noqa: E402
    PaperKineticCompiledCpuArtifactStorePolicy,
)
from kinetic_lazy_native_material_step import (  # noqa: E402
    PaperKineticLazyNativeMemoryPolicy,
)


INPUT_SCHEMA_VERSION = 1
INPUT_PROVENANCE = "worldfoam-g4-public-quality-inputs-v1"
_INPUT_SEAL = object()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _component_generation(component: Any, *, name: str) -> str:
    generation = getattr(component, "generation_digest", None)
    if not _is_sha256(generation):
        raise ValueError(f"{name} lacks a sealed generation digest")
    return str(generation)


def _policy_payload(policy: Any, *, name: str) -> Any:
    validator = getattr(policy, "assert_valid", None)
    if callable(validator):
        try:
            validator()
        except TypeError:
            # Full-geometry policy is validated separately against its route.
            pass
    payload = getattr(policy, "payload", None)
    if callable(payload):
        return payload()
    fields = getattr(policy, "__dict__", None)
    if not isinstance(fields, dict):
        raise TypeError(f"{name} does not expose immutable policy fields")
    return dict(fields)


@dataclass(frozen=True)
class WorldFoamPublicQualityInputs(Mapping[str, Any]):
    """Immutable, owner-bound mapping consumed by both WorldFoam routes."""

    schema_version: int
    dataset_owner_identity: int
    sample_id: str
    dataset_generation_digest: str
    heldout_dataset_generation_digest: str
    dataset_capability_sha256: str
    initialization_sha256: str
    compiler_sha256: str
    same_representation_group: str
    target_provider: PowerFoamTargetProvider = field(repr=False)
    ray_provider: PowerFoamRayProvider = field(repr=False)
    heldout_target_provider: PowerFoamTargetProvider = field(repr=False)
    heldout_ray_provider: PowerFoamRayProvider = field(repr=False)
    frame_times: tuple[float, ...]
    world_initializer: PaperKineticPointCloudWorldInitializer = field(repr=False)
    program_factory: Any = field(repr=False)
    background_rgb_f32_cpu: torch.Tensor = field(repr=False)
    artifact_store_policy: PaperKineticCompiledCpuArtifactStorePolicy
    lazy_memory_policy: PaperKineticLazyNativeMemoryPolicy
    full_geometry_memory_policy: PaperKineticLazyFullGeometryMemoryPolicy
    combined_sgd_policy: PaperKineticFixedCameraCombinedSGDPolicy
    material_parameterization: PaperKineticFixedSiteMaterialParameterization
    material_sgd_policy: PaperKineticFixedSiteMaterialSGDPolicy
    maximum_material_state_logical_tensor_bytes: int
    maximum_tracks_per_bundle: int
    maximum_observations_per_bundle: int
    maximum_rows_per_native_block: int
    maximum_samples_per_launch: int
    maximum_artifact_accounted_bytes_per_entry: int
    cone_tolerance: float
    shared_reverse_mode: str
    input_generation_digest: str
    _object_identities: tuple[int, ...] = field(repr=False)
    _seal: object = field(repr=False)

    FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "schema_version",
        "dataset_owner_identity",
        "sample_id",
        "dataset_generation_digest",
        "heldout_dataset_generation_digest",
        "dataset_capability_sha256",
        "initialization_sha256",
        "compiler_sha256",
        "same_representation_group",
        "target_provider",
        "ray_provider",
        "heldout_target_provider",
        "heldout_ray_provider",
        "frame_times",
        "world_initializer",
        "program_factory",
        "background_rgb_f32_cpu",
        "artifact_store_policy",
        "lazy_memory_policy",
        "full_geometry_memory_policy",
        "combined_sgd_policy",
        "material_parameterization",
        "material_sgd_policy",
        "maximum_material_state_logical_tensor_bytes",
        "maximum_tracks_per_bundle",
        "maximum_observations_per_bundle",
        "maximum_rows_per_native_block",
        "maximum_samples_per_launch",
        "maximum_artifact_accounted_bytes_per_entry",
        "cone_tolerance",
        "shared_reverse_mode",
        "input_generation_digest",
    )

    def __getitem__(self, key: str) -> Any:
        if key not in self.FIELD_NAMES:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.FIELD_NAMES)

    def __len__(self) -> int:
        return len(self.FIELD_NAMES)

    def assert_current(self, *, dataset: Any, context: Any | None = None) -> None:
        if (
            self._seal is not _INPUT_SEAL
            or self.schema_version != INPUT_SCHEMA_VERSION
            or self.dataset_owner_identity != id(dataset)
            or self._object_identities != _input_object_identities(self)
            or self.input_generation_digest != _input_digest(self)
        ):
            raise ValueError("WorldFoam public-quality input seal or owner changed")
        _validate_inputs(self)
        if getattr(dataset, "sample_id", None) != self.sample_id:
            raise ValueError("WorldFoam inputs belong to another public dataset")
        if context is None:
            return
        protocol = context.protocol
        if (
            protocol.dataset.sample_id != self.sample_id
            or protocol.dataset.frame_count != len(self.frame_times)
            or tuple(protocol.dataset.train_cameras)
            and self.target_provider.view_count
            != len(protocol.dataset.train_cameras)
            or self.heldout_target_provider.view_count
            != len(protocol.dataset.heldout_cameras)
            or self.same_representation_group
            != context.route_spec["same_representation_group"]
            or self.initialization_sha256
            != context.scene_receipt["initialization_sha256"]
            or self.compiler_sha256 != context.scene_receipt["compiler_sha256"]
            or self.dataset_capability_sha256
            != context.dataset_capability["capability_sha256"]
        ):
            raise ValueError("WorldFoam inputs drifted from the frozen row context")


def seal_worldfoam_public_training_inputs(
    *,
    dataset: Any,
    sample_id: str,
    dataset_generation_digest: str,
    heldout_dataset_generation_digest: str,
    dataset_capability_sha256: str,
    initialization_sha256: str,
    compiler_sha256: str,
    same_representation_group: str,
    target_provider: PowerFoamTargetProvider,
    ray_provider: PowerFoamRayProvider,
    heldout_target_provider: PowerFoamTargetProvider,
    heldout_ray_provider: PowerFoamRayProvider,
    frame_times: tuple[float, ...],
    world_initializer: PaperKineticPointCloudWorldInitializer,
    program_factory: Any,
    background_rgb_f32_cpu: torch.Tensor,
    artifact_store_policy: PaperKineticCompiledCpuArtifactStorePolicy,
    lazy_memory_policy: PaperKineticLazyNativeMemoryPolicy,
    full_geometry_memory_policy: PaperKineticLazyFullGeometryMemoryPolicy,
    combined_sgd_policy: PaperKineticFixedCameraCombinedSGDPolicy,
    material_parameterization: PaperKineticFixedSiteMaterialParameterization,
    material_sgd_policy: PaperKineticFixedSiteMaterialSGDPolicy,
    maximum_material_state_logical_tensor_bytes: int,
    maximum_tracks_per_bundle: int,
    maximum_observations_per_bundle: int,
    maximum_rows_per_native_block: int,
    maximum_samples_per_launch: int,
    maximum_artifact_accounted_bytes_per_entry: int,
    cone_tolerance: float,
    shared_reverse_mode: str,
) -> WorldFoamPublicQualityInputs:
    """Seal the exact split providers, compiler, optimizer, and memory limits."""

    provisional = WorldFoamPublicQualityInputs(
        schema_version=INPUT_SCHEMA_VERSION,
        dataset_owner_identity=id(dataset),
        sample_id=str(sample_id),
        dataset_generation_digest=str(dataset_generation_digest),
        heldout_dataset_generation_digest=str(heldout_dataset_generation_digest),
        dataset_capability_sha256=str(dataset_capability_sha256),
        initialization_sha256=str(initialization_sha256),
        compiler_sha256=str(compiler_sha256),
        same_representation_group=str(same_representation_group),
        target_provider=target_provider,
        ray_provider=ray_provider,
        heldout_target_provider=heldout_target_provider,
        heldout_ray_provider=heldout_ray_provider,
        frame_times=tuple(float(value) for value in frame_times),
        world_initializer=world_initializer,
        program_factory=program_factory,
        background_rgb_f32_cpu=background_rgb_f32_cpu,
        artifact_store_policy=artifact_store_policy,
        lazy_memory_policy=lazy_memory_policy,
        full_geometry_memory_policy=full_geometry_memory_policy,
        combined_sgd_policy=combined_sgd_policy,
        material_parameterization=material_parameterization,
        material_sgd_policy=material_sgd_policy,
        maximum_material_state_logical_tensor_bytes=(
            maximum_material_state_logical_tensor_bytes
        ),
        maximum_tracks_per_bundle=maximum_tracks_per_bundle,
        maximum_observations_per_bundle=maximum_observations_per_bundle,
        maximum_rows_per_native_block=maximum_rows_per_native_block,
        maximum_samples_per_launch=maximum_samples_per_launch,
        maximum_artifact_accounted_bytes_per_entry=(
            maximum_artifact_accounted_bytes_per_entry
        ),
        cone_tolerance=float(cone_tolerance),
        shared_reverse_mode=str(shared_reverse_mode),
        input_generation_digest="",
        _object_identities=(),
        _seal=_INPUT_SEAL,
    )
    object.__setattr__(
        provisional,
        "_object_identities",
        _input_object_identities(provisional),
    )
    object.__setattr__(
        provisional,
        "input_generation_digest",
        _input_digest(provisional),
    )
    provisional.assert_current(dataset=dataset)
    return provisional


def _validate_inputs(value: WorldFoamPublicQualityInputs) -> None:
    for name in (
        "dataset_generation_digest",
        "heldout_dataset_generation_digest",
        "dataset_capability_sha256",
        "initialization_sha256",
        "compiler_sha256",
        "input_generation_digest",
    ):
        if not _is_sha256(getattr(value, name)):
            raise ValueError(f"WorldFoam input {name} is not a SHA-256 digest")
    if not value.sample_id.strip() or not value.same_representation_group.strip():
        raise ValueError("WorldFoam sample/representation identity must be nonempty")
    if (
        not value.frame_times
        or any(not math.isfinite(time) for time in value.frame_times)
        or tuple(sorted(value.frame_times)) != value.frame_times
    ):
        raise ValueError("WorldFoam physical frame times must be finite and ordered")
    providers = (
        ("target_provider", value.target_provider, PowerFoamTargetProvider),
        ("ray_provider", value.ray_provider, PowerFoamRayProvider),
        (
            "heldout_target_provider",
            value.heldout_target_provider,
            PowerFoamTargetProvider,
        ),
        ("heldout_ray_provider", value.heldout_ray_provider, PowerFoamRayProvider),
    )
    for name, provider, expected_type in providers:
        if type(provider) is not expected_type:
            raise TypeError(f"{name} must use the exact production provider type")
        if provider.frame_count != len(value.frame_times):
            raise ValueError(f"{name} frame grid changed")
    for target in (value.target_provider, value.heldout_target_provider):
        if type(target.source) is not MappedRgb8PowerFoamTargetSource:
            raise TypeError("G4 targets must use the mapped selected-pixel source")
        residency = target.residency()
        if (
            residency.get("resident_bytes") != 0
            or residency.get("source_kind") != "mapped_rgb8_pixel_time_v1"
            or not target.native_selected_pixel_method_available
            or residency.get("mapping_lifetime") != "one_selected_pixel_read"
        ):
            raise MemoryError("G4 target provider is not disk-mapped and pixel-bounded")
    dimensions = {
        (provider.height, provider.width)
        for provider in (
            value.target_provider,
            value.ray_provider,
            value.heldout_target_provider,
            value.heldout_ray_provider,
        )
    }
    if len(dimensions) != 1:
        raise ValueError("WorldFoam train/heldout provider dimensions differ")
    if type(value.world_initializer) is not PaperKineticPointCloudWorldInitializer:
        raise TypeError("G4 requires the deterministic point-cloud initializer")
    value.world_initializer.assert_current()
    _component_generation(value.program_factory, name="program_factory")
    tensor = value.background_rgb_f32_cpu
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device.type != "cpu"
        or tensor.dtype != torch.float32
        or tuple(tensor.shape) != (3,)
        or not tensor.is_contiguous()
        or tensor.requires_grad
        or not bool(torch.isfinite(tensor).all().item())
    ):
        raise ValueError("WorldFoam background must be contiguous CPU float32 [3]")
    if type(value.artifact_store_policy) is not PaperKineticCompiledCpuArtifactStorePolicy:
        raise TypeError("WorldFoam artifact-store policy type changed")
    if type(value.lazy_memory_policy) is not PaperKineticLazyNativeMemoryPolicy:
        raise TypeError("WorldFoam lazy memory policy type changed")
    value.lazy_memory_policy.assert_valid()
    if type(value.full_geometry_memory_policy) is not PaperKineticLazyFullGeometryMemoryPolicy:
        raise TypeError("WorldFoam full-geometry policy type changed")
    value.full_geometry_memory_policy.assert_valid(
        reverse_mode=value.shared_reverse_mode
    )
    if type(value.combined_sgd_policy) is not PaperKineticFixedCameraCombinedSGDPolicy:
        raise TypeError("WorldFoam combined SGD policy type changed")
    value.combined_sgd_policy.assert_valid()
    if type(value.material_parameterization) is not PaperKineticFixedSiteMaterialParameterization:
        raise TypeError("WorldFoam material parameterization type changed")
    value.material_parameterization.assert_valid()
    if type(value.material_sgd_policy) is not PaperKineticFixedSiteMaterialSGDPolicy:
        raise TypeError("WorldFoam material SGD policy type changed")
    value.material_sgd_policy.assert_valid()
    for name in (
        "maximum_material_state_logical_tensor_bytes",
        "maximum_tracks_per_bundle",
        "maximum_observations_per_bundle",
        "maximum_rows_per_native_block",
        "maximum_samples_per_launch",
        "maximum_artifact_accounted_bytes_per_entry",
    ):
        raw = getattr(value, name)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
            raise ValueError(f"WorldFoam {name} must be a positive integer")
    if (
        value.shared_reverse_mode not in FULL_GEOMETRY_REVERSE_MODES
        or not math.isfinite(value.cone_tolerance)
        or value.cone_tolerance <= 0.0
    ):
        raise ValueError("WorldFoam reverse mode or cone tolerance is invalid")


def _input_object_identities(
    value: WorldFoamPublicQualityInputs,
) -> tuple[int, ...]:
    return tuple(
        id(component)
        for component in (
            value.target_provider,
            value.target_provider.source,
            value.ray_provider,
            value.ray_provider.cameras,
            value.heldout_target_provider,
            value.heldout_target_provider.source,
            value.heldout_ray_provider,
            value.heldout_ray_provider.cameras,
            value.world_initializer,
            value.program_factory,
            value.background_rgb_f32_cpu,
            value.artifact_store_policy,
            value.lazy_memory_policy,
            value.full_geometry_memory_policy,
            value.combined_sgd_policy,
            value.material_parameterization,
            value.material_sgd_policy,
        )
    )


def _input_digest(value: WorldFoamPublicQualityInputs) -> str:
    return _canonical_sha256(
        {
            "provenance": INPUT_PROVENANCE,
            "schema_version": value.schema_version,
            "sample_id": value.sample_id,
            "dataset_generation_digest": value.dataset_generation_digest,
            "heldout_dataset_generation_digest": (
                value.heldout_dataset_generation_digest
            ),
            "dataset_capability_sha256": value.dataset_capability_sha256,
            "initialization_sha256": value.initialization_sha256,
            "compiler_sha256": value.compiler_sha256,
            "same_representation_group": value.same_representation_group,
            "train_target_residency": value.target_provider.residency(),
            "heldout_target_residency": value.heldout_target_provider.residency(),
            "train_camera_shape": (
                value.ray_provider.view_count,
                value.ray_provider.frame_count,
                value.ray_provider.height,
                value.ray_provider.width,
            ),
            "heldout_camera_shape": (
                value.heldout_ray_provider.view_count,
                value.heldout_ray_provider.frame_count,
                value.heldout_ray_provider.height,
                value.heldout_ray_provider.width,
            ),
            "frame_times": value.frame_times,
            "initializer_generation_digest": (
                value.world_initializer.generation_digest
            ),
            "factory_generation_digest": _component_generation(
                value.program_factory,
                name="program_factory",
            ),
            "background": tuple(float(item) for item in value.background_rgb_f32_cpu),
            "artifact_store_policy": _policy_payload(
                value.artifact_store_policy,
                name="artifact_store_policy",
            ),
            "lazy_memory_policy": _policy_payload(
                value.lazy_memory_policy,
                name="lazy_memory_policy",
            ),
            "full_geometry_memory_policy": _policy_payload(
                value.full_geometry_memory_policy,
                name="full_geometry_memory_policy",
            ),
            "combined_sgd_policy": _policy_payload(
                value.combined_sgd_policy,
                name="combined_sgd_policy",
            ),
            "material_parameterization": _policy_payload(
                value.material_parameterization,
                name="material_parameterization",
            ),
            "material_sgd_policy": _policy_payload(
                value.material_sgd_policy,
                name="material_sgd_policy",
            ),
            "maximum_material_state_logical_tensor_bytes": (
                value.maximum_material_state_logical_tensor_bytes
            ),
            "maximum_tracks_per_bundle": value.maximum_tracks_per_bundle,
            "maximum_observations_per_bundle": (
                value.maximum_observations_per_bundle
            ),
            "maximum_rows_per_native_block": value.maximum_rows_per_native_block,
            "maximum_samples_per_launch": value.maximum_samples_per_launch,
            "maximum_artifact_accounted_bytes_per_entry": (
                value.maximum_artifact_accounted_bytes_per_entry
            ),
            "cone_tolerance": value.cone_tolerance,
            "shared_reverse_mode": value.shared_reverse_mode,
        }
    )


__all__ = (
    "INPUT_PROVENANCE",
    "INPUT_SCHEMA_VERSION",
    "WorldFoamPublicQualityInputs",
    "seal_worldfoam_public_training_inputs",
)
