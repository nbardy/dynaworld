"""Memory-bounded material-only optimizer steps for native WorldFoam.

This module is the narrow production seam between the source-only compiler
contracts and the native fixed-word lifecycle.  Geometry, power weights, ray
programs, owner words, and compact chart schedules are immutable.  Only
caller-owned raw-density and raw-color ``Parameter`` objects advance. They are
decoded with softplus and sigmoid before native replay, and native physical
bars are returned through the exact manual chain rule.

One logical step is normalized by ``global_tracks * global_samples * 3`` and
streams every registered spatial block through ``B_p x K`` staging.  Native
RGBA bars are scattered into one global ``CompactSpatialStepLedger`` and then
assigned directly to the two material parameters.  No autograd render tape is
constructed.

The owner-only binding deliberately does *not* certify transfer/Jacobian
approximation error after a material update.  Every result from this module is
therefore explicitly non-paper evidence.  A frozen checkpoint must take the
separate strict continuous-certification path before it can support a paper
claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from compact_lie_schedule import CompactLieWorldSchedule
from material_parameterization import WorldFoamMaterialParameterization
from native_track_adapter import (
    NativeFixedWordP0TopologyCacheKey,
    NativeFixedWordP0ValidatedTopologyToken,
    assert_native_fixed_word_p0_validated_topology_token,
    consume_native_fixed_word_p0_track_block_result,
    estimate_native_fixed_word_p0_topology_token_resident_bytes,
    execute_native_fixed_word_p0_track_block,
    native_fixed_word_p0_topology_cache_key,
    resolve_native_fixed_word_p0_ops,
)
from powerfoam_track_staging import PowerFoamTrackStagingPlan
from prepared_track_block import PreparedWorldFoamTrackBlock
from staged_compiled_lie_adjoint import (
    CompactSpatialGradientBuffers,
    allocate_compact_spatial_gradient_buffers,
    begin_compact_spatial_step_v2,
    finalize_compact_spatial_step,
)

_TRAINING_BINDING_MODE = "training_owner_topology_only"


@dataclass(frozen=True)
class WorldFoamMaterialTrainingBlock:
    """One immutable contiguous track block and its owner-only capability."""

    block_id: str
    topology: PreparedWorldFoamTrackBlock
    schedule: CompactLieWorldSchedule
    owner_binding: Any
    _topology_tensor_signatures: tuple[tuple[object, ...], ...] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.block_id.strip():
            raise ValueError("material-training block_id must be nonempty")
        self.schedule.assert_current()
        _assert_owner_only_training_binding(self.owner_binding)
        _validate_block_identity(self)
        object.__setattr__(
            self,
            "_topology_tensor_signatures",
            tuple(_tensor_signature(tensor) for tensor in _topology_tensors(self.topology)),
        )

    @property
    def track_start(self) -> int:
        return int(self.topology.source_track_ids[0].item())

    @property
    def track_end(self) -> int:
        return int(self.topology.source_track_ids[-1].item()) + 1

    def assert_current(self) -> None:
        self.schedule.assert_current()
        _assert_owner_only_training_binding(self.owner_binding)
        if tuple(_tensor_signature(tensor) for tensor in _topology_tensors(self.topology)) != (
            self._topology_tensor_signatures
        ):
            raise ValueError("material-training block topology changed after preparation")
        _validate_block_identity(self)


@dataclass(frozen=True)
class WorldFoamMaterialTrainingProgram:
    """Reusable immutable program shared by all material optimizer steps."""

    staging_plan: PowerFoamTrackStagingPlan
    blocks: tuple[WorldFoamMaterialTrainingBlock, ...]
    site_geometry: torch.Tensor
    ray_coefficients: torch.Tensor
    background_rgb: tuple[float, float, float]
    replay_config: Any
    parameterization: WorldFoamMaterialParameterization
    sample_block_size: int
    global_track_count: int
    global_sample_count: int
    loss_normalization_id: str
    immutable_generation_id: str
    _immutable_tensor_signatures: tuple[tuple[object, ...], ...] = field(
        repr=False,
    )
    binding_mode: str = field(default=_TRAINING_BINDING_MODE, init=False)
    owner_identity_certified: bool = field(default=True, init=False)
    transfer_jacobian_certified: bool = field(default=False, init=False)
    approximation_error_certified: bool = field(default=False, init=False)
    paper_evidence_eligible: bool = field(default=False, init=False)

    @property
    def expected_blocks(self) -> tuple[tuple[str, int, int], ...]:
        return tuple((block.block_id, block.track_start, block.track_end) for block in self.blocks)

    @property
    def expected_block_schedule_generations(self) -> tuple[tuple[str, str], ...]:
        return tuple((block.block_id, block.schedule.generation_digest) for block in self.blocks)

    @property
    def global_loss_element_count(self) -> int:
        return self.global_track_count * self.global_sample_count * 3

    def assert_current(self) -> None:
        if (
            self.binding_mode != _TRAINING_BINDING_MODE
            or not self.owner_identity_certified
            or self.transfer_jacobian_certified
            or self.approximation_error_certified
            or self.paper_evidence_eligible
        ):
            raise ValueError("material-training program cannot claim strict or paper certification")
        if tuple(_tensor_signature(tensor) for tensor in _program_immutable_tensors(self)) != (
            self._immutable_tensor_signatures
        ):
            raise ValueError("material-training geometry, rays, or staging identities changed")
        for block in self.blocks:
            block.assert_current()
        if _program_generation_id(self) != self.immutable_generation_id:
            raise ValueError("material-training immutable program generation changed")


@dataclass(frozen=True)
class WorldFoamMaterialPersistentMemoryReport:
    """Unique persistent tensor storage reachable from one live program.

    Category byte counts are independently storage-deduplicated. The program
    total deduplicates again across categories. Target-provider residency is
    reported through its source contract because lazy providers need not expose
    a tensor. Python objects, JSON strings, allocator metadata/reservations,
    optimizer state, and native/transient launch buffers are outside this
    source-level report.
    """

    block_count: int
    unique_schedule_count: int
    program_global_model_staging_tensor_bytes: int
    retained_block_topology_tensor_bytes: int
    retained_training_binding_private_tensor_bytes: int
    unique_schedule_tensor_bytes: int
    cross_category_shared_tensor_storage_bytes: int
    unique_program_tensor_storage_bytes: int
    target_provider_residency_available: bool
    target_provider_residency: dict[str, Any]
    target_provider_resident_bytes: int
    total_source_level_persistent_bytes: int
    tensor_storage_deduplicated: bool = True
    retained_compiled_cpu_atlas_block_count: int = 0
    excluded_byte_classes: tuple[str, ...] = (
        "python_objects",
        "json_strings",
        "allocator_metadata_and_reservations",
        "optimizer_state",
        "native_transient_buffers",
    )


@dataclass(frozen=True)
class WorldFoamNativeTopologyCachePolicy:
    """Explicit device-topology residency budget for a material session.

    ``max_live_topology_tensor_bytes`` bounds cached tokens plus the one token
    being prepared. It is checked from source topology before native allocation
    and again against the sealed token. Cached tokens use LRU eviction under
    both the entry and byte caps. Setting either cache cap to zero disables
    retention while still allowing one bounded live token.
    """

    max_cached_entries: int
    max_cached_tensor_bytes: int
    max_live_topology_tensor_bytes: int

    def assert_valid(self) -> None:
        values = (
            self.max_cached_entries,
            self.max_cached_tensor_bytes,
            self.max_live_topology_tensor_bytes,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("native topology cache limits must be integers")
        if self.max_cached_entries < 0 or self.max_cached_tensor_bytes < 0:
            raise ValueError("native topology cache limits must be nonnegative")
        if self.max_live_topology_tensor_bytes < 1:
            raise ValueError("native topology live-residency limit must be positive")


@dataclass
class WorldFoamMaterialTrainingSession:
    """Caller-owned material parameters plus reusable global gradient bars."""

    program: WorldFoamMaterialTrainingProgram
    raw_density: torch.nn.Parameter
    raw_color: torch.nn.Parameter
    site_density: torch.Tensor
    site_color: torch.Tensor
    raw_density_gradient: torch.Tensor
    raw_color_gradient: torch.Tensor
    gradients: CompactSpatialGradientBuffers
    parameter_layout_signatures: tuple[tuple[object, ...], ...]
    physical_material_layout_signatures: tuple[tuple[object, ...], ...]
    gradient_layout_signatures: tuple[tuple[object, ...], ...]
    raw_parameter_versions: tuple[int, int]
    physical_material_versions: tuple[int, int]
    native_topology_cache_policy: WorldFoamNativeTopologyCachePolicy
    native_topology_token_cache: OrderedDict[
        NativeFixedWordP0TopologyCacheKey,
        NativeFixedWordP0ValidatedTopologyToken,
    ] = field(default_factory=OrderedDict)
    native_topology_cache_peak_resident_tensor_bytes: int = 0
    native_topology_peak_live_tensor_bytes: int = 0
    native_topology_cache_eviction_count: int = 0
    native_topology_cache_skip_count: int = 0
    steps_completed: int = 0

    @property
    def gradient_storage_pointers(self) -> tuple[int, ...]:
        return tuple(
            tensor.untyped_storage().data_ptr()
            for tensor in (*self.gradients.tensors, self.raw_density_gradient, self.raw_color_gradient)
        )

    def assert_current(self) -> None:
        self.program.assert_current()
        self.native_topology_cache_policy.assert_valid()
        _validate_raw_material_parameters(
            self.program,
            self.raw_density,
            self.raw_color,
        )
        if tuple(_tensor_layout_signature(tensor) for tensor in (self.raw_density, self.raw_color)) != (
            self.parameter_layout_signatures
        ):
            raise ValueError("caller-owned material parameter storage or layout changed")
        if (self.raw_density._version, self.raw_color._version) != self.raw_parameter_versions:
            raise ValueError("caller-owned raw material parameters changed outside the optimizer step")
        if tuple(_tensor_layout_signature(tensor) for tensor in (self.site_density, self.site_color)) != (
            self.physical_material_layout_signatures
        ):
            raise ValueError("decoded physical material storage or layout changed")
        if (self.site_density._version, self.site_color._version) != self.physical_material_versions:
            raise ValueError("decoded physical materials changed outside their parameterization")
        _validate_physical_materials(self.program, self.site_density, self.site_color)
        expected_shapes = (
            tuple(self.program.site_geometry[:, :4].shape),
            tuple(self.program.site_geometry[:, 4].shape),
            tuple(self.site_density.shape),
            tuple(self.site_color.shape),
            tuple(self.raw_density_gradient.shape),
            tuple(self.raw_color_gradient.shape),
        )
        all_gradients = (*self.gradients.tensors, self.raw_density_gradient, self.raw_color_gradient)
        if tuple(tuple(tensor.shape) for tensor in all_gradients) != expected_shapes:
            raise ValueError("reusable global material gradient buffers changed shape")
        if tuple(_tensor_layout_signature(tensor) for tensor in all_gradients) != (self.gradient_layout_signatures):
            raise ValueError("reusable global material gradient buffer storage changed")
        if not isinstance(self.native_topology_token_cache, OrderedDict):
            raise TypeError("material-training topology cache must preserve LRU order")
        if len(self.native_topology_token_cache) > self.native_topology_cache_policy.max_cached_entries:
            raise ValueError("material-training topology cache exceeded its explicit entry budget")
        cache_bytes = _native_topology_cache_resident_tensor_bytes(self.native_topology_token_cache)
        if cache_bytes > self.native_topology_cache_policy.max_cached_tensor_bytes:
            raise ValueError("material-training topology cache exceeded its explicit tensor-byte budget")
        if cache_bytes > self.native_topology_cache_policy.max_live_topology_tensor_bytes:
            raise ValueError("material-training topology cache exceeded total live topology budget")
        if self.native_topology_cache_peak_resident_tensor_bytes < cache_bytes:
            raise ValueError("material-training topology cache peak accounting is smaller than live state")
        if (
            self.native_topology_peak_live_tensor_bytes
            > self.native_topology_cache_policy.max_live_topology_tensor_bytes
        ):
            raise ValueError("material-training live topology residency exceeded its explicit byte budget")
        blocks_by_id = {block.block_id: block for block in self.program.blocks}
        for key, cached in self.native_topology_token_cache.items():
            if key != cached.cache_key or key.block_id not in blocks_by_id:
                raise ValueError("material-training topology cache identity was corrupted")
            block = blocks_by_id[key.block_id]
            assert_native_fixed_word_p0_validated_topology_token(
                cached,
                block_id=block.block_id,
                prepared=block,
                certificate_binding=block.owner_binding,
                native_ops=cached.native_ops,
                device=self.program.site_geometry.device,
                immutable_generation_id=self.program.immutable_generation_id,
            )


@dataclass(frozen=True)
class WorldFoamMaterialTrainingStepResult:
    """Detached diagnostics for one completed non-paper optimizer step."""

    step_index: int
    loss: float
    density_gradient_norm: float
    color_gradient_norm: float
    raw_density_gradient_norm: float
    raw_color_gradient_norm: float
    frozen_geometry_gradient_norm: float
    frozen_weight_gradient_norm: float
    immutable_generation_id: str
    accounting: dict[str, int | float | str | bool]
    binding_mode: str = _TRAINING_BINDING_MODE
    transfer_jacobian_certified: bool = False
    approximation_error_certified: bool = False
    paper_evidence_eligible: bool = False


def prepare_worldfoam_material_training_program(
    *,
    staging_plan: PowerFoamTrackStagingPlan,
    blocks: tuple[WorldFoamMaterialTrainingBlock, ...],
    site_geometry: torch.Tensor,
    ray_coefficients: torch.Tensor,
    background_rgb: torch.Tensor | tuple[float, float, float] | list[float],
    replay_config: Any,
    sample_block_size: int,
    loss_normalization_id: str = "worldfoam-material-logical-step",
    parameterization: WorldFoamMaterialParameterization | None = None,
) -> WorldFoamMaterialTrainingProgram:
    """Seal the immutable, template-free program used by repeated steps."""

    if not blocks:
        raise ValueError("material-training program requires at least one spatial block")
    if sample_block_size < 1:
        raise ValueError("sample_block_size must be positive")
    if not loss_normalization_id.strip():
        raise ValueError("loss_normalization_id must be nonempty")
    normalized_parameterization = WorldFoamMaterialParameterization() if parameterization is None else parameterization
    if not isinstance(normalized_parameterization, WorldFoamMaterialParameterization):
        raise TypeError("parameterization must be a WorldFoamMaterialParameterization")
    normalized_parameterization.assert_valid()
    geometry = torch.as_tensor(site_geometry)
    rays = torch.as_tensor(ray_coefficients)
    if geometry.ndim != 2 or tuple(geometry.shape[1:]) != (5,):
        raise ValueError("frozen site_geometry must have shape [site_count,5]")
    if rays.ndim != 2 or tuple(rays.shape[1:]) != (12,):
        raise ValueError("frozen ray_coefficients must have shape [global_tracks,12]")
    if geometry.dtype != torch.float32 or rays.dtype != torch.float32:
        raise ValueError("native material-training geometry and rays must use float32")
    if geometry.device != rays.device:
        raise ValueError("frozen geometry and rays must share one device")
    if geometry.requires_grad or rays.requires_grad:
        raise ValueError("geometry, weights, and rays must remain frozen in material-only training")
    if not bool(torch.isfinite(geometry).all().item()) or not bool(torch.isfinite(rays).all().item()):
        raise ValueError("frozen geometry and rays must be finite")

    global_track_count, global_sample_count = _global_view_track_shape(staging_plan)
    if tuple(rays.shape) != (global_track_count, 12):
        raise ValueError("ray_coefficients must contain one affine program per global view-pixel track")
    expected_start = 0
    seen_block_ids: set[str] = set()
    for block in blocks:
        block.assert_current()
        if block.block_id in seen_block_ids:
            raise ValueError("material-training spatial block ids must be unique")
        seen_block_ids.add(block.block_id)
        if block.track_start != expected_start or block.track_end <= block.track_start:
            raise ValueError("material-training blocks must form one ordered half-open track tiling")
        if block.schedule.global_track_count != global_track_count:
            raise ValueError("every compact schedule must use the global view-track count")
        if block.schedule.selection_provenance == "extracted_from_selected_adaptive_atlas":
            raise ValueError("material training requires a predeclared compact spec schedule")
        if global_track_count != staging_plan.track_count:
            pixel_count = staging_plan.track_count
            if block.track_start // pixel_count != (block.track_end - 1) // pixel_count:
                raise ValueError("a material-training spatial block cannot cross a view-major boundary")
        expected_start = block.track_end
    if expected_start != global_track_count:
        raise ValueError("material-training blocks must cover every global track exactly once")

    background = tuple(float(value) for value in torch.as_tensor(background_rgb).reshape(-1).tolist())
    if len(background) != 3 or not all(math.isfinite(value) for value in background):
        raise ValueError("background_rgb must contain three finite values")
    if not hasattr(replay_config, "near") or not hasattr(replay_config, "far"):
        raise ValueError("replay_config must expose near and far")
    if not math.isfinite(float(replay_config.near)) or not math.isfinite(float(replay_config.far)):
        raise ValueError("replay_config near/far must be finite")

    immutable_tensors = (
        geometry,
        rays,
        staging_plan.pixel_indices,
        staging_plan.sample_indices,
        staging_plan.sample_times,
    )
    provisional = WorldFoamMaterialTrainingProgram(
        staging_plan=staging_plan,
        blocks=blocks,
        site_geometry=geometry,
        ray_coefficients=rays,
        background_rgb=background,
        replay_config=replay_config,
        parameterization=normalized_parameterization,
        sample_block_size=int(sample_block_size),
        global_track_count=global_track_count,
        global_sample_count=global_sample_count,
        loss_normalization_id=loss_normalization_id,
        immutable_generation_id="",
        _immutable_tensor_signatures=tuple(_tensor_signature(tensor) for tensor in immutable_tensors),
    )
    program = replace(
        provisional,
        immutable_generation_id=_program_generation_id(provisional),
    )
    program.assert_current()
    return program


def report_worldfoam_material_training_program_persistent_memory(
    program: WorldFoamMaterialTrainingProgram,
) -> WorldFoamMaterialPersistentMemoryReport:
    """Measure persistent source-level storage from an actual sealed program."""

    if not isinstance(program, WorldFoamMaterialTrainingProgram):
        raise TypeError("persistent memory reporting requires a WorldFoamMaterialTrainingProgram")
    program.assert_current()
    global_tensors = _program_global_model_staging_tensors(program)
    topology_tensors = tuple(tensor for block in program.blocks for tensor in _topology_tensors(block.topology))
    binding_tensors = tuple(
        tensor for block in program.blocks for tensor in _training_binding_private_tensors(block.owner_binding)
    )
    unique_schedules = _unique_program_schedules(program)
    schedule_tensors = tuple(
        tensor
        for schedule in unique_schedules
        for chart in schedule.charts
        for tensor in (chart.node_times, chart.fit_matrix, chart.barycentric_weights)
    )
    category_bytes = (
        _unique_tensor_storage_bytes(global_tensors),
        _unique_tensor_storage_bytes(topology_tensors),
        _unique_tensor_storage_bytes(binding_tensors),
        _unique_tensor_storage_bytes(schedule_tensors),
    )
    unique_program_bytes = _unique_tensor_storage_bytes(
        (*global_tensors, *topology_tensors, *binding_tensors, *schedule_tensors)
    )
    category_sum = sum(category_bytes)
    if unique_program_bytes > category_sum:
        raise ArithmeticError("globally deduplicated tensor storage exceeded category storage")
    residency_available, residency, target_resident_bytes = _target_provider_residency(
        program.staging_plan.target_provider
    )
    return WorldFoamMaterialPersistentMemoryReport(
        block_count=len(program.blocks),
        unique_schedule_count=len(unique_schedules),
        program_global_model_staging_tensor_bytes=category_bytes[0],
        retained_block_topology_tensor_bytes=category_bytes[1],
        retained_training_binding_private_tensor_bytes=category_bytes[2],
        unique_schedule_tensor_bytes=category_bytes[3],
        cross_category_shared_tensor_storage_bytes=category_sum - unique_program_bytes,
        unique_program_tensor_storage_bytes=unique_program_bytes,
        target_provider_residency_available=residency_available,
        target_provider_residency=residency,
        target_provider_resident_bytes=target_resident_bytes,
        total_source_level_persistent_bytes=unique_program_bytes + target_resident_bytes,
    )


@torch.no_grad()
def bind_worldfoam_material_parameters(
    program: WorldFoamMaterialTrainingProgram,
    *,
    raw_density: torch.nn.Parameter,
    raw_color: torch.nn.Parameter,
    native_topology_cache_policy: WorldFoamNativeTopologyCachePolicy,
) -> WorldFoamMaterialTrainingSession:
    """Attach raw material Parameters and allocate decoded values/bars once."""

    program.assert_current()
    if not isinstance(native_topology_cache_policy, WorldFoamNativeTopologyCachePolicy):
        raise TypeError("native_topology_cache_policy must be explicit")
    native_topology_cache_policy.assert_valid()
    _validate_raw_material_parameters(program, raw_density, raw_color)
    site_density = torch.empty_like(raw_density, requires_grad=False)
    site_color = torch.empty_like(raw_color, requires_grad=False)
    program.parameterization.decode_density_(site_density, raw_density)
    program.parameterization.decode_color_(site_color, raw_color)
    _validate_physical_materials(program, site_density, site_color)
    gradients = allocate_compact_spatial_gradient_buffers(
        site_geometry=program.site_geometry,
        site_density=site_density,
        site_color=site_color,
    )
    raw_density_gradient = torch.zeros_like(raw_density, requires_grad=False)
    raw_color_gradient = torch.zeros_like(raw_color, requires_grad=False)
    session = WorldFoamMaterialTrainingSession(
        program=program,
        raw_density=raw_density,
        raw_color=raw_color,
        site_density=site_density,
        site_color=site_color,
        raw_density_gradient=raw_density_gradient,
        raw_color_gradient=raw_color_gradient,
        gradients=gradients,
        parameter_layout_signatures=tuple(_tensor_layout_signature(tensor) for tensor in (raw_density, raw_color)),
        physical_material_layout_signatures=tuple(
            _tensor_layout_signature(tensor) for tensor in (site_density, site_color)
        ),
        gradient_layout_signatures=tuple(
            _tensor_layout_signature(tensor)
            for tensor in (*gradients.tensors, raw_density_gradient, raw_color_gradient)
        ),
        raw_parameter_versions=(raw_density._version, raw_color._version),
        physical_material_versions=(site_density._version, site_color._version),
        native_topology_cache_policy=native_topology_cache_policy,
    )
    session.assert_current()
    return session


@torch.no_grad()
def run_worldfoam_material_training_step(
    session: WorldFoamMaterialTrainingSession,
    optimizer: torch.optim.Optimizer,
    *,
    native_ops: Any | None = None,
    device_synchronize: Callable[[torch.device], None] | None = None,
) -> WorldFoamMaterialTrainingStepResult:
    """Stream all ``B_p x K`` blocks, assign manual bars, and step once."""

    session.assert_current()
    _validate_material_only_optimizer(session, optimizer)
    optimizer.zero_grad(set_to_none=True)
    program = session.program
    ledger = begin_compact_spatial_step_v2(
        schedule=program.blocks[0].schedule,
        site_geometry=program.site_geometry,
        ray_coefficients=program.ray_coefficients,
        site_density=session.site_density,
        site_color=session.site_color,
        gradients=session.gradients,
        global_track_count=program.global_track_count,
        global_frame_count=program.global_sample_count,
        loss_normalization_id=program.loss_normalization_id,
        expected_blocks=program.expected_blocks,
        expected_block_schedule_generations=program.expected_block_schedule_generations,
    )
    sample_block_count = 0
    chart_count = 0
    device_barrier_count = 0
    sample_weight_evaluations: set[str] = set()
    sample_weight_linear_interactions = 0
    sample_weight_dense_fallback_interactions = 0
    sample_weight_exact_node_rows = 0
    sample_weight_dense_fallback_rows = 0
    peak_staged_target_bytes = 0
    peak_staged_explicit_ray_bytes = 0
    peak_staged_sample_time_bytes = 0
    topology_cache_hit_count = 0
    topology_cache_miss_count = 0
    topology_cache_evictions_before = session.native_topology_cache_eviction_count
    topology_cache_skips_before = session.native_topology_cache_skip_count
    peak_preflight_token_bytes = 0
    peak_actual_token_bytes = 0
    resolved_native_ops = resolve_native_fixed_word_p0_ops(native_ops)
    for block in program.blocks:
        preflight_token_bytes = estimate_native_fixed_word_p0_topology_token_resident_bytes(block)
        peak_preflight_token_bytes = max(peak_preflight_token_bytes, preflight_token_bytes)
        if preflight_token_bytes > session.native_topology_cache_policy.max_live_topology_tensor_bytes:
            raise ValueError(
                "native topology token preflight exceeds max_live_topology_tensor_bytes: "
                f"block={block.block_id!r}, estimated={preflight_token_bytes}, "
                f"budget={session.native_topology_cache_policy.max_live_topology_tensor_bytes}"
            )
        topology_cache_key = native_fixed_word_p0_topology_cache_key(
            block_id=block.block_id,
            prepared=block,
            certificate_binding=block.owner_binding,
            native_ops=resolved_native_ops,
            device=program.site_geometry.device,
            immutable_generation_id=program.immutable_generation_id,
        )
        cached_topology = session.native_topology_token_cache.get(topology_cache_key)
        if cached_topology is None:
            topology_cache_miss_count += 1
            _evict_native_topology_tokens_for_allocation(
                session,
                required_tensor_bytes=preflight_token_bytes,
            )
        else:
            topology_cache_hit_count += 1
            session.native_topology_token_cache.move_to_end(topology_cache_key)
        predicted_live_bytes = _native_topology_cache_resident_tensor_bytes(
            session.native_topology_token_cache
        ) + (preflight_token_bytes if cached_topology is None else 0)
        session.native_topology_peak_live_tensor_bytes = max(
            session.native_topology_peak_live_tensor_bytes,
            predicted_live_bytes,
        )
        native_result = execute_native_fixed_word_p0_track_block(
            ledger,
            block_id=block.block_id,
            prepared=block,
            staging_plan=program.staging_plan,
            certificate_binding=block.owner_binding,
            background_rgb=program.background_rgb,
            replay_config=program.replay_config,
            sample_block_size=program.sample_block_size,
            native_ops=resolved_native_ops,
            validated_topology_token=cached_topology,
            immutable_generation_id=program.immutable_generation_id,
            max_in_flight_sample_blocks=1,
            device_synchronize=device_synchronize,
        )
        if native_result.geometry_vjp_executed or native_result.grad_site_geometry_f32 is not None:
            raise ValueError("material training must use the native RGBA-only reverse capability")
        validated_topology = native_result.validated_topology_token
        if validated_topology is None or validated_topology.cache_key != topology_cache_key:
            raise ValueError("material-training native adapter returned no matching topology token")
        actual_token_bytes = validated_topology.resident_tensor_bytes
        peak_actual_token_bytes = max(peak_actual_token_bytes, actual_token_bytes)
        if actual_token_bytes > preflight_token_bytes:
            raise ValueError(
                "sealed native topology token exceeded its conservative preflight: "
                f"block={block.block_id!r}, actual={actual_token_bytes}, "
                f"estimated={preflight_token_bytes}"
            )
        actual_live_bytes = _native_topology_cache_resident_tensor_bytes(
            session.native_topology_token_cache
        ) + (actual_token_bytes if cached_topology is None else 0)
        session.native_topology_peak_live_tensor_bytes = max(
            session.native_topology_peak_live_tensor_bytes,
            actual_live_bytes,
        )
        if actual_live_bytes > session.native_topology_cache_policy.max_live_topology_tensor_bytes:
            raise ArithmeticError("native topology live residency exceeded its preflighted byte budget")
        sample_block_count += native_result.sample_block_count
        chart_count += native_result.chart_count
        device_barrier_count += native_result.device_barrier_count
        sample_weight_evaluations.add(native_result.sample_weight_evaluation)
        sample_weight_linear_interactions += native_result.sample_weight_linear_interactions
        sample_weight_dense_fallback_interactions += native_result.sample_weight_dense_fallback_interactions
        sample_weight_exact_node_rows += native_result.sample_weight_exact_node_rows
        sample_weight_dense_fallback_rows += native_result.sample_weight_dense_fallback_rows
        peak_staged_target_bytes = max(
            peak_staged_target_bytes,
            native_result.peak_staged_target_bytes,
        )
        peak_staged_explicit_ray_bytes = max(
            peak_staged_explicit_ray_bytes,
            native_result.peak_staged_explicit_ray_bytes,
        )
        peak_staged_sample_time_bytes = max(
            peak_staged_sample_time_bytes,
            native_result.peak_staged_sample_time_bytes,
        )
        consume_native_fixed_word_p0_track_block_result(ledger, native_result)
        del native_result
        if cached_topology is None:
            _retain_native_topology_token_if_budgeted(
                session,
                topology_cache_key,
                validated_topology,
            )
        del validated_topology

    final = finalize_compact_spatial_step(ledger)
    density_gradient = final.gradients.grad_site_density
    color_gradient = final.gradients.grad_site_color
    for name, gradient in (("density", density_gradient), ("color", color_gradient)):
        if not bool(torch.isfinite(gradient).all().item()):
            raise ValueError(f"native {name} gradient is non-finite")
    program.parameterization.density_vjp_(
        session.raw_density_gradient,
        session.raw_density,
        density_gradient,
    )
    program.parameterization.color_vjp_(
        session.raw_color_gradient,
        session.site_color,
        color_gradient,
    )
    session.raw_density.grad = session.raw_density_gradient
    session.raw_color.grad = session.raw_color_gradient
    loss = float(final.loss.item())
    if not math.isfinite(loss):
        raise ValueError("native material loss is non-finite")
    density_gradient_norm = float(torch.linalg.vector_norm(density_gradient).item())
    color_gradient_norm = float(torch.linalg.vector_norm(color_gradient).item())
    raw_density_gradient_norm = float(torch.linalg.vector_norm(session.raw_density_gradient).item())
    raw_color_gradient_norm = float(torch.linalg.vector_norm(session.raw_color_gradient).item())
    frozen_geometry_gradient_norm = float(torch.linalg.vector_norm(final.gradients.grad_site_geometry).item())
    frozen_weight_gradient_norm = float(torch.linalg.vector_norm(final.gradients.grad_site_weight).item())
    if frozen_geometry_gradient_norm != 0.0 or frozen_weight_gradient_norm != 0.0:
        raise ValueError("material-only training produced a nonzero frozen geometry bar")
    optimizer.step()
    if not bool(torch.isfinite(session.raw_density).all().item()) or not bool(
        torch.isfinite(session.raw_color).all().item()
    ):
        raise ValueError("material-only optimizer produced non-finite raw parameters")
    program.parameterization.decode_density_(session.site_density, session.raw_density)
    program.parameterization.decode_color_(session.site_color, session.raw_color)
    session.raw_parameter_versions = (
        session.raw_density._version,
        session.raw_color._version,
    )
    session.physical_material_versions = (
        session.site_density._version,
        session.site_color._version,
    )
    session.assert_current()
    session.steps_completed += 1
    accounting: dict[str, int | float | str | bool] = {
        **final.accounting,
        "mode": _TRAINING_BINDING_MODE,
        "optimizer_step_index": session.steps_completed,
        "sample_block_count": sample_block_count,
        "sample_weight_evaluation": "+".join(sorted(sample_weight_evaluations)),
        "sample_weight_common_path_complexity": "O(spatial_blocks*F*J)",
        "sample_weight_dense_fallback_complexity": ("O(spatial_blocks*F_fallback*J^2)"),
        "sample_weight_spatial_block_count": len(program.blocks),
        "sample_weight_linear_interactions": sample_weight_linear_interactions,
        "sample_weight_dense_fallback_interactions": (sample_weight_dense_fallback_interactions),
        "sample_weight_exact_node_rows": sample_weight_exact_node_rows,
        "sample_weight_dense_fallback_rows": sample_weight_dense_fallback_rows,
        "chart_block_count": chart_count,
        "device_barrier_count": device_barrier_count,
        "sample_payload_layout": "target_only",
        "peak_staged_target_bytes": peak_staged_target_bytes,
        "peak_staged_explicit_ray_bytes": peak_staged_explicit_ray_bytes,
        "peak_staged_sample_time_bytes": peak_staged_sample_time_bytes,
        "chart_or_global_sample_time_clone_bytes": 0,
        "explicit_ray_staging_omitted": peak_staged_explicit_ray_bytes == 0,
        "native_material_refresh_count": len(program.blocks),
        "native_topology_prepare_count": topology_cache_miss_count,
        "native_topology_cache_hit_count": topology_cache_hit_count,
        "native_topology_cache_miss_count": topology_cache_miss_count,
        "native_topology_cache_entry_count": len(session.native_topology_token_cache),
        "native_topology_cache_resident_tensor_bytes": _native_topology_cache_resident_tensor_bytes(
            session.native_topology_token_cache
        ),
        "native_topology_cache_peak_resident_tensor_bytes": (
            session.native_topology_cache_peak_resident_tensor_bytes
        ),
        "native_topology_cache_max_entries": (
            session.native_topology_cache_policy.max_cached_entries
        ),
        "native_topology_cache_max_tensor_bytes": (
            session.native_topology_cache_policy.max_cached_tensor_bytes
        ),
        "native_topology_max_live_tensor_bytes": (
            session.native_topology_cache_policy.max_live_topology_tensor_bytes
        ),
        "native_topology_peak_live_tensor_bytes": session.native_topology_peak_live_tensor_bytes,
        "native_topology_peak_preflight_token_tensor_bytes": peak_preflight_token_bytes,
        "native_topology_peak_actual_token_tensor_bytes": peak_actual_token_bytes,
        "native_topology_cache_eviction_count": (
            session.native_topology_cache_eviction_count - topology_cache_evictions_before
        ),
        "native_topology_cache_skip_count": (
            session.native_topology_cache_skip_count - topology_cache_skips_before
        ),
        "native_topology_cache_bounded_by_explicit_policy": True,
        "native_topology_cache_bounded_by_spatial_block_count": False,
        "cpu_compact_atlas_compile_count_per_step": 0,
        "prepared_block_compile_count_per_step": 0,
        "binding_construction_compiled_snapshot_count_lower_bound": len(program.blocks),
        "retained_compiled_cpu_atlas_block_count": 0,
        "lightweight_topology_schedule_block_count": len(program.blocks),
        "immutable_topology_reused": True,
        "compact_spec_schedule_reused": True,
        "global_gradient_buffers_reused": True,
        "density_parameterization": "softplus",
        "color_parameterization": "sigmoid",
        "manual_parameter_chain_rule": True,
        "geometry_vjp_executed": False,
        "material_only_reverse_tensor_bytes_omitted": sum(
            16 * block.topology.incidence_count + 20 * block.topology.boundary_count + 20 * block.topology.site_count
            for block in program.blocks
        ),
        "geometry_optimizer_gradient_assigned": False,
        "weight_optimizer_gradient_assigned": False,
        "ray_optimizer_gradient_assigned": False,
        "transfer_jacobian_certified": False,
        "approximation_error_certified": False,
        "paper_evidence_eligible": False,
        "immutable_generation_id": program.immutable_generation_id,
    }
    return WorldFoamMaterialTrainingStepResult(
        step_index=session.steps_completed,
        loss=loss,
        density_gradient_norm=density_gradient_norm,
        color_gradient_norm=color_gradient_norm,
        raw_density_gradient_norm=raw_density_gradient_norm,
        raw_color_gradient_norm=raw_color_gradient_norm,
        frozen_geometry_gradient_norm=frozen_geometry_gradient_norm,
        frozen_weight_gradient_norm=frozen_weight_gradient_norm,
        immutable_generation_id=program.immutable_generation_id,
        accounting=accounting,
    )


def _validate_block_identity(block: WorldFoamMaterialTrainingBlock) -> None:
    topology = block.topology
    if topology.track_count < 1 or topology.site_count < 1:
        raise ValueError("material-training topology must contain tracks and sites")
    expected_track_ids = torch.arange(
        int(topology.source_track_ids[0].item()),
        int(topology.source_track_ids[-1].item()) + 1,
        dtype=topology.source_track_ids.dtype,
        device=topology.source_track_ids.device,
    )
    if not torch.equal(topology.source_track_ids, expected_track_ids):
        raise ValueError("material-training topology tracks must be contiguous and ordered")
    binding = block.owner_binding
    if int(getattr(binding, "site_count", topology.site_count)) != topology.site_count:
        raise ValueError("owner-only binding and compact topology disagree on site count")
    binding_charts = tuple(getattr(binding, "charts", ()))
    if len(binding_charts) != block.schedule.chart_count:
        raise ValueError("owner-only binding and compact schedule disagree on chart count")
    for chart, expected in zip(binding_charts, block.schedule.charts, strict=True):
        if (
            float(chart.t_min) != expected.t_min
            or float(chart.t_max) != expected.t_max
            or float(chart.near) != expected.near
            or float(chart.far) != expected.far
            or int(chart.node_count) != expected.node_count
        ):
            raise ValueError("owner-only binding and compact spec schedule disagree")


def _assert_owner_only_training_binding(binding: Any) -> None:
    assert_current = getattr(binding, "assert_current", None)
    if not callable(assert_current):
        raise TypeError("material training requires a sealed current owner-only binding")
    assert_current()
    required_true = (
        "owner_identity_certified",
        "geometry_rays_immutable",
        "live_site_rgba_refresh_allowed",
    )
    required_false = (
        "transfer_jacobian_certified",
        "approximation_error_certified",
        "paper_evidence_eligible",
        "runtime_floating_point_roundoff_certified",
    )
    if getattr(binding, "binding_mode", "") != _TRAINING_BINDING_MODE:
        raise ValueError("material training requires the owner-topology-only binding mode")
    if any(not bool(getattr(binding, name, False)) for name in required_true):
        raise ValueError("owner-only training binding is missing an immutable capability fact")
    if any(bool(getattr(binding, name, True)) for name in required_false):
        raise ValueError("owner-only material binding must remain uncertified and non-paper")
    for method_name in (
        "assert_prepared_immutable",
        "assert_native_topology",
        "assert_native_world",
        "assert_native_chart",
        "validate_sample_times",
    ):
        if not callable(getattr(binding, method_name, None)):
            raise TypeError(f"owner-only material binding is missing {method_name}")
    digest = str(getattr(binding, "canonical_digest", ""))
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("owner-only material binding must expose a canonical SHA-256 digest")


def _validate_raw_material_parameters(
    program: WorldFoamMaterialTrainingProgram,
    raw_density: torch.nn.Parameter,
    raw_color: torch.nn.Parameter,
) -> None:
    if not isinstance(raw_density, torch.nn.Parameter) or not isinstance(raw_color, torch.nn.Parameter):
        raise TypeError("raw_density and raw_color must be caller-owned torch Parameters")
    if not raw_density.is_leaf or not raw_color.is_leaf or not raw_density.requires_grad or not raw_color.requires_grad:
        raise ValueError("raw material parameters must be trainable leaf Parameters")
    site_count = int(program.site_geometry.shape[0])
    if tuple(raw_density.shape) != (site_count,) or tuple(raw_color.shape) != (site_count, 3):
        raise ValueError("raw material parameter shapes must be [site_count] and [site_count,3]")
    if (
        raw_density.dtype != torch.float32
        or raw_color.dtype != torch.float32
        or raw_density.device != program.site_geometry.device
        or raw_color.device != program.site_geometry.device
    ):
        raise ValueError("raw material parameters must match the frozen float32 world device")
    if raw_density.untyped_storage().data_ptr() == raw_color.untyped_storage().data_ptr():
        raise ValueError("raw density and color Parameters must own distinct storage")
    if not bool(torch.isfinite(raw_density).all().item()) or not bool(torch.isfinite(raw_color).all().item()):
        raise ValueError("raw material parameters must be finite")


def _validate_physical_materials(
    program: WorldFoamMaterialTrainingProgram,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
) -> None:
    site_count = int(program.site_geometry.shape[0])
    if tuple(site_density.shape) != (site_count,) or tuple(site_color.shape) != (site_count, 3):
        raise ValueError("decoded physical material shapes must be [site_count] and [site_count,3]")
    if site_density.requires_grad or site_color.requires_grad:
        raise ValueError("decoded physical materials must not retain an autograd graph")
    if not bool(torch.isfinite(site_density).all().item()) or not bool(torch.isfinite(site_color).all().item()):
        raise ValueError("decoded physical materials must be finite")
    if bool(torch.any(site_density < program.parameterization.minimum_density).item()):
        raise ValueError("decoded physical density left its nonnegative parameterization")
    if bool(torch.any((site_color < 0.0) | (site_color > 1.0)).item()):
        raise ValueError("decoded physical color left the sigmoid unit interval")


def _validate_material_only_optimizer(
    session: WorldFoamMaterialTrainingSession,
    optimizer: torch.optim.Optimizer,
) -> None:
    optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if len(optimizer_parameters) != 2 or {id(parameter) for parameter in optimizer_parameters} != {
        id(session.raw_density),
        id(session.raw_color),
    }:
        raise ValueError("material-only optimizer must contain exactly the caller raw-density and raw-color Parameters")


def _global_view_track_shape(plan: PowerFoamTrackStagingPlan) -> tuple[int, int]:
    frame_count = plan.target_provider.frame_count
    views = torch.div(plan.sample_indices, frame_count, rounding_mode="floor")
    active_views = tuple(sorted({int(view) for view in views.tolist()}))
    if not active_views:
        raise ValueError("material-training staging plan contains no active views")
    reference_frames = None
    reference_times = None
    for view in active_views:
        positions = torch.nonzero(views == view, as_tuple=False).reshape(-1)
        order = torch.argsort(plan.sample_times.index_select(0, positions), stable=True)
        positions = positions.index_select(0, order)
        frames = torch.remainder(plan.sample_indices.index_select(0, positions), frame_count)
        times = plan.sample_times.index_select(0, positions)
        if reference_frames is None:
            reference_frames = frames
            reference_times = times
        elif not torch.equal(frames, reference_frames) or not torch.equal(times, reference_times):
            raise ValueError("material-training views must form one rectangular frame/time grid")
    assert reference_times is not None
    return plan.track_count * len(active_views), int(reference_times.numel())


def _program_global_model_staging_tensors(
    program: WorldFoamMaterialTrainingProgram,
) -> tuple[torch.Tensor, ...]:
    tensors = list(_program_immutable_tensors(program))
    ray_provider = program.staging_plan.ray_provider
    for cameras in ray_provider.cameras:
        for camera in cameras:
            for value in (
                camera.fx,
                camera.fy,
                camera.cx,
                camera.cy,
                camera.camera_to_world,
                camera.distortion,
            ):
                if torch.is_tensor(value):
                    tensors.append(value)
    return tuple(tensors)


def _training_binding_private_tensors(binding: Any) -> tuple[torch.Tensor, ...]:
    bound_tensors = getattr(binding, "_bound_tensors", None)
    barycentric_weights = getattr(binding, "_sample_barycentric_weights", None)
    if not isinstance(bound_tensors, tuple) or not isinstance(barycentric_weights, tuple):
        raise TypeError(
            "exact persistent memory reporting requires binding _bound_tensors and _sample_barycentric_weights tuples"
        )
    tensors = (*bound_tensors, *barycentric_weights)
    if any(not torch.is_tensor(tensor) for tensor in tensors):
        raise TypeError("training-binding private tensor payload contains a non-tensor value")
    return tensors


def _unique_program_schedules(
    program: WorldFoamMaterialTrainingProgram,
) -> tuple[CompactLieWorldSchedule, ...]:
    schedules = []
    seen_ids: set[int] = set()
    for block in program.blocks:
        schedule_id = id(block.schedule)
        if schedule_id not in seen_ids:
            seen_ids.add(schedule_id)
            schedules.append(block.schedule)
    return tuple(schedules)


def _unique_tensor_storage_bytes(tensors: tuple[torch.Tensor, ...]) -> int:
    storages: dict[tuple[str, int, int], int] = {}
    for tensor in tensors:
        if not torch.is_tensor(tensor):
            raise TypeError("persistent memory accounting accepts tensors only")
        storage = tensor.untyped_storage()
        storage_bytes = int(storage.nbytes())
        key = (str(tensor.device), int(storage.data_ptr()), storage_bytes)
        storages.setdefault(key, storage_bytes)
    return sum(storages.values())


def _target_provider_residency(
    target_provider: Any,
) -> tuple[bool, dict[str, Any], int]:
    residency_method = getattr(target_provider, "residency", None)
    if not callable(residency_method):
        return False, {}, 0
    raw_residency = residency_method()
    if not isinstance(raw_residency, Mapping):
        raise TypeError("target-provider residency() must return a mapping")
    residency = dict(raw_residency)
    resident_bytes = residency.get("resident_bytes")
    if isinstance(resident_bytes, bool) or not isinstance(resident_bytes, int):
        raise TypeError("target-provider residency must contain integer resident_bytes")
    if resident_bytes < 0:
        raise ValueError("target-provider resident_bytes must be nonnegative")
    return True, residency, resident_bytes


def _program_immutable_tensors(program: WorldFoamMaterialTrainingProgram) -> tuple[torch.Tensor, ...]:
    return (
        program.site_geometry,
        program.ray_coefficients,
        program.staging_plan.pixel_indices,
        program.staging_plan.sample_indices,
        program.staging_plan.sample_times,
    )


def _program_generation_id(program: WorldFoamMaterialTrainingProgram) -> str:
    payload = {
        "schema": "worldfoam-material-training-program-v1",
        "binding_mode": program.binding_mode,
        "global_track_count": program.global_track_count,
        "global_sample_count": program.global_sample_count,
        "loss_normalization_id": program.loss_normalization_id,
        "sample_block_size": program.sample_block_size,
        "background_rgb": program.background_rgb,
        "near": float(program.replay_config.near),
        "far": float(program.replay_config.far),
        "parameterization": {
            "density": "softplus",
            "density_beta": program.parameterization.density_beta,
            "density_threshold": program.parameterization.density_threshold,
            "minimum_density": program.parameterization.minimum_density,
            "color": "sigmoid",
        },
        "blocks": [
            {
                "block_id": block.block_id,
                "track_start": block.track_start,
                "track_end": block.track_end,
                "schedule_generation": block.schedule.generation_digest,
                "owner_binding": block.owner_binding.canonical_digest,
            }
            for block in program.blocks
        ],
        "staging": [
            _tensor_content_digest(program.staging_plan.pixel_indices),
            _tensor_content_digest(program.staging_plan.sample_indices),
            _tensor_content_digest(program.staging_plan.sample_times),
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _topology_tensors(topology: PreparedWorldFoamTrackBlock) -> tuple[torch.Tensor, ...]:
    return (
        topology.source_track_ids,
        topology.source_boundary_ids,
        topology.source_site_ids,
        topology.word_offsets_i32,
        topology.word_owner_i32,
        topology.word_left_incidence_i32,
        topology.word_right_incidence_i32,
        topology.track_incidence_offsets_i32,
        topology.incidence_boundary_i32,
        topology.boundary_site_pairs_i32,
    )


def _native_topology_cache_resident_tensor_bytes(
    cache: Mapping[
        NativeFixedWordP0TopologyCacheKey,
        NativeFixedWordP0ValidatedTopologyToken,
    ],
) -> int:
    """Conservative sum; separate compact block tokens should not alias."""

    return sum(token.resident_tensor_bytes for token in cache.values())


def _evict_native_topology_tokens_for_allocation(
    session: WorldFoamMaterialTrainingSession,
    *,
    required_tensor_bytes: int,
) -> None:
    """Make room before a native token allocation, not after its peak."""

    if required_tensor_bytes < 0:
        raise ValueError("native topology allocation bytes must be nonnegative")
    cache = session.native_topology_token_cache
    live_budget = session.native_topology_cache_policy.max_live_topology_tensor_bytes
    while cache and _native_topology_cache_resident_tensor_bytes(cache) + required_tensor_bytes > live_budget:
        cache.popitem(last=False)
        session.native_topology_cache_eviction_count += 1
    if _native_topology_cache_resident_tensor_bytes(cache) + required_tensor_bytes > live_budget:
        raise ValueError("native topology allocation cannot fit the explicit live-residency budget")


def _retain_native_topology_token_if_budgeted(
    session: WorldFoamMaterialTrainingSession,
    key: NativeFixedWordP0TopologyCacheKey,
    token: NativeFixedWordP0ValidatedTopologyToken,
) -> None:
    """Retain one sealed token under entry, cache-byte, and live-byte caps."""

    policy = session.native_topology_cache_policy
    token_bytes = token.resident_tensor_bytes
    retention_byte_limit = min(
        policy.max_cached_tensor_bytes,
        policy.max_live_topology_tensor_bytes,
    )
    if policy.max_cached_entries == 0 or retention_byte_limit == 0 or token_bytes > retention_byte_limit:
        session.native_topology_cache_skip_count += 1
        return
    cache = session.native_topology_token_cache
    for stale_key in tuple(cache):
        if stale_key.block_id == key.block_id:
            del cache[stale_key]
            session.native_topology_cache_eviction_count += 1
    while cache and (
        len(cache) + 1 > policy.max_cached_entries
        or _native_topology_cache_resident_tensor_bytes(cache) + token_bytes > retention_byte_limit
    ):
        cache.popitem(last=False)
        session.native_topology_cache_eviction_count += 1
    if len(cache) + 1 > policy.max_cached_entries:
        raise ArithmeticError("native topology token cannot fit an ostensibly valid entry budget")
    if _native_topology_cache_resident_tensor_bytes(cache) + token_bytes > retention_byte_limit:
        raise ArithmeticError("native topology token cannot fit an ostensibly valid byte budget")
    cache[key] = token
    resident = _native_topology_cache_resident_tensor_bytes(cache)
    session.native_topology_cache_peak_resident_tensor_bytes = max(
        session.native_topology_cache_peak_resident_tensor_bytes,
        resident,
    )


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (*_tensor_layout_signature(tensor), tensor._version)


def _tensor_layout_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        tensor.untyped_storage().data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        str(tensor.dtype),
        str(tensor.device),
    )


def _tensor_content_digest(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().to(device="cpu").contiguous()
    header = json.dumps(
        {"dtype": str(cpu.dtype), "shape": list(cpu.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + cpu.numpy().tobytes(order="C")).hexdigest()


__all__ = [
    "WorldFoamMaterialParameterization",
    "WorldFoamNativeTopologyCachePolicy",
    "WorldFoamMaterialPersistentMemoryReport",
    "WorldFoamMaterialTrainingBlock",
    "WorldFoamMaterialTrainingProgram",
    "WorldFoamMaterialTrainingSession",
    "WorldFoamMaterialTrainingStepResult",
    "bind_worldfoam_material_parameters",
    "prepare_worldfoam_material_training_program",
    "report_worldfoam_material_training_program_persistent_memory",
    "run_worldfoam_material_training_step",
]
