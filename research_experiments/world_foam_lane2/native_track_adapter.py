"""Bridge pixel-track staging to the certified native fixed-word lifecycle.

This is deliberately a narrow adapter.  It supports the representation that
the current native kernel actually consumes: one affine ray program per track.
A rectangular fixed-camera multi-view grid is expanded into canonical
view-major ``(view,pixel)`` tracks and processed one view-local spatial block
at a time.  Moving cameras still fail closed until a certified camera-gauge
compiler produces their native track program.

The adapter owns sample-to-chart assignment and never accepts interpolation
weights.  The sealed native certificate derives those weights from sample
times.  Targets are staged one ``B_p x K`` block at a time, predictions are
discarded after each launch, and compact native gradients can be index-added
into the existing caller-owned ``CompactSpatialStepLedger``.
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from powerfoam_track_staging import (
    PowerFoamTrackStageBlock,
    PowerFoamTrackStagingPlan,
    PowerFoamTrackTargetStageBlock,
)
from prepared_track_block import accumulate_prepared_rows_, gather_prepared_rows
from staged_compiled_lie_adjoint import (
    CompactSpatialStepLedger,
    _assert_compact_spatial_step_current,
    _assert_prepared_schedule_matches_ledger,
    _tensor_signature,
)


class NativeTrackAdapterUnavailableError(ValueError):
    """The selected staging contract has no certified native representation."""


_INFLIGHT_LEDGER_ATTRIBUTE = "_native_fixed_word_p0_inflight_block_id"
_VALIDATED_TOPOLOGY_TOKEN_SEAL = object()


@dataclass(frozen=True)
class _ChartSamplePartition:
    chart_index: int
    global_start: int
    global_end: int


@dataclass(frozen=True)
class _ResolvedStagingLayout:
    plan: PowerFoamTrackStagingPlan
    stage_track_start: int
    stage_track_end: int
    view_factor: int


@dataclass(frozen=True)
class NativeFixedWordP0TopologyCacheKey:
    """Identity of one reusable material-training topology token.

    The native token contains device-resident CSR and therefore cannot be
    reused merely because a block id matches.  The native implementation,
    device, whole immutable program generation, binding, topology snapshot,
    and chart schedule must all still be identical.
    """

    block_id: str
    native_ops_identity: int
    device: str
    immutable_generation_id: str
    binding_digest: str
    topology_generation_id: str
    schedule_generation_digest: str


@dataclass(frozen=True)
class NativeFixedWordP0ValidatedTopologyToken:
    """Sealed reusable device topology plus its compact gather identities."""

    cache_key: NativeFixedWordP0TopologyCacheKey
    native_token: Any = field(repr=False)
    native_ops: Any = field(repr=False)
    certificate_binding: Any = field(repr=False)
    source_site_ids_i64: torch.Tensor = field(repr=False)
    source_track_ids_i64: torch.Tensor = field(repr=False)
    native_tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    source_id_tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    _seal: object = field(repr=False)

    @property
    def resident_tensor_bytes(self) -> int:
        """Unique tensor storage retained by this validated device token."""

        return _unique_tensor_storage_bytes(
            (
                *_native_topology_token_tensors(self.native_token),
                self.source_site_ids_i64,
                self.source_track_ids_i64,
            )
        )


@dataclass(frozen=True)
class NativeFixedWordP0TrackBlockResult:
    """Compact native result ready for the global spatial gradient ledger."""

    block_id: str
    prepared: Any
    certificate_binding: Any
    loss: torch.Tensor
    grad_site_geometry_f32: torch.Tensor | None
    grad_site_rgba_f32: torch.Tensor
    geometry_vjp_executed: bool
    loss_normalization_id: str
    sample_partition_generation_id: str
    global_track_count: int
    global_sample_count: int
    sample_block_count: int
    chart_count: int
    device_barrier_count: int
    sample_payload_layout: str
    peak_staged_target_bytes: int
    peak_staged_explicit_ray_bytes: int
    peak_staged_sample_time_bytes: int
    sample_weight_evaluation: str
    sample_weight_linear_interactions: int
    sample_weight_dense_fallback_interactions: int
    sample_weight_exact_node_rows: int
    sample_weight_dense_fallback_rows: int
    tensor_signatures: tuple[tuple[object, ...], ...]
    source_site_ids_i64: torch.Tensor = field(repr=False)
    source_site_ids_signature: tuple[object, ...] = field(repr=False)
    validated_topology_token: NativeFixedWordP0ValidatedTopologyToken | None = field(repr=False)
    _ledger: CompactSpatialStepLedger = field(repr=False)

    @property
    def resident_output_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.loss,
                self.grad_site_rgba_f32,
                self.grad_site_geometry_f32,
            )
            if tensor is not None
        )


def execute_native_fixed_word_p0_track_block(
    ledger: CompactSpatialStepLedger,
    *,
    block_id: str,
    prepared: Any,
    staging_plan: PowerFoamTrackStagingPlan,
    certificate_binding: Any,
    background_rgb: torch.Tensor | tuple[float, float, float] | list[float],
    replay_config: Any,
    sample_block_size: int,
    native_ops: Any | None = None,
    validated_topology_token: NativeFixedWordP0ValidatedTopologyToken | None = None,
    immutable_generation_id: str | None = None,
    max_in_flight_sample_blocks: int = 1,
    device_synchronize: Callable[[torch.device], None] | None = None,
    physical_length_epsilon: float = 1.0e-8,
    cone_tolerance: float = 1.0e-6,
) -> NativeFixedWordP0TrackBlockResult:
    """Execute one certified spatial block while staging only ``B_p x K`` data.

    ``ledger`` supplies both the live accelerator parameters and the one global
    RGB denominator. ``prepared`` supplies compact topology and a certified
    chart schedule. Strict evaluation uses a full compact snapshot; the
    material-only path may pass a lightweight immutable training block and
    therefore retains no compiled CPU atlas per spatial block.
    Strict evaluation binds every world value. The explicit material-training
    capability instead binds immutable topology/geometry/rays/schedules while
    permitting a new live RGBA tensor each step; that path is non-paper and has
    no transfer/Jacobian approximation-error certificate.
    """

    _assert_compact_spatial_step_current(ledger)
    if ledger.finalized:
        raise ValueError("compact spatial step was already finalized")
    track_start, track_end = _registered_track_range(ledger, block_id)
    if block_id in ledger.consumed_block_ids:
        raise ValueError("compact spatial block was already consumed")
    if getattr(ledger, _INFLIGHT_LEDGER_ATTRIBUTE, None) is not None:
        raise ValueError("consume and drop the previous native spatial result before executing another block")
    if sample_block_size < 1:
        raise ValueError("sample_block_size must be positive")
    if max_in_flight_sample_blocks != 1:
        raise ValueError(
            "the native adapter currently requires max_in_flight_sample_blocks=1 so Bp x K residency is bounded"
        )
    _assert_binding_matches_prepared(certificate_binding, prepared)
    material_training_mode = getattr(certificate_binding, "binding_mode", "") == "training_owner_topology_only"
    _assert_prepared_schedule_matches_ledger(
        ledger,
        prepared,
        block_id=block_id,
        legacy_error="native track block uses a different global atlas template",
        validate_prepared_current=not material_training_mode,
    )
    expected_track_ids = torch.arange(
        track_start,
        track_end,
        dtype=prepared.topology.source_track_ids.dtype,
        device=prepared.topology.source_track_ids.device,
    )
    if not torch.equal(prepared.topology.source_track_ids, expected_track_ids):
        raise ValueError("certified compact topology does not match its registered track range")
    staging = _resolve_staging_layout(
        staging_plan,
        ledger,
        track_start=track_start,
        track_end=track_end,
    )

    ordered_plan, partitions = _ordered_chart_staging_plan(
        staging.plan,
        prepared,
        device=ledger.source_tensors[0].device,
    )
    if not partitions:
        raise ValueError("native track block requires at least one populated certified chart")
    first_range = next(
        _chart_sample_blocks(
            partitions[0],
            sample_block_size=sample_block_size,
            block_id=block_id,
        )
    )
    first_stage: PowerFoamTrackStageBlock | PowerFoamTrackTargetStageBlock | None = None
    if not material_training_mode:
        first_stage = ordered_plan.stage(
            track_start=staging.stage_track_start,
            track_end=staging.stage_track_end,
            sample_start=first_range[1],
            sample_end=first_range[2],
            require_affine_ray_program=True,
        )

    geometry, global_rays, density, color = ledger.source_tensors
    if any(tensor.dtype != torch.float32 for tensor in (geometry, global_rays, density, color)):
        raise ValueError("native fixed-word live world tensors must use float32")
    if len({tensor.device for tensor in (geometry, global_rays, density, color)}) != 1:
        raise ValueError("native fixed-word live world tensors must share one device")
    native = resolve_native_fixed_word_p0_ops(native_ops)
    if validated_topology_token is not None:
        if not material_training_mode:
            raise ValueError("reusable topology tokens are restricted to material training")
        if immutable_generation_id is None:
            raise ValueError("reusable topology tokens require an immutable program generation")
        assert_native_fixed_word_p0_validated_topology_token(
            validated_topology_token,
            block_id=block_id,
            prepared=prepared,
            certificate_binding=certificate_binding,
            native_ops=native,
            device=geometry.device,
            immutable_generation_id=immutable_generation_id,
        )
        topology_token = validated_topology_token.native_token
        source_site_ids = validated_topology_token.source_site_ids_i64
        source_track_ids = validated_topology_token.source_track_ids_i64
        reusable_topology = validated_topology_token
    else:
        topology = prepared.topology
        topology_token = native.prepare_fixed_word_p0_topology_token(
            topology.word_offsets_i32.to(device=geometry.device, dtype=torch.int32),
            topology.word_owner_i32.to(device=geometry.device, dtype=torch.int32),
            topology.word_left_incidence_i32.to(device=geometry.device, dtype=torch.int32),
            topology.word_right_incidence_i32.to(device=geometry.device, dtype=torch.int32),
            topology.track_incidence_offsets_i32.to(device=geometry.device, dtype=torch.int32),
            topology.incidence_boundary_i32.to(device=geometry.device, dtype=torch.int32),
            topology.boundary_site_pairs_i32.to(device=geometry.device, dtype=torch.int32),
            track_count=topology.track_count,
            site_count=topology.site_count,
            certificate_binding=certificate_binding,
        )
        source_site_ids = topology.source_site_ids.to(
            device=geometry.device,
            dtype=torch.long,
        )
        source_track_ids = topology.source_track_ids.to(
            device=geometry.device,
            dtype=torch.long,
        )
        reusable_topology = None
        if material_training_mode and immutable_generation_id is not None:
            reusable_topology = _seal_native_fixed_word_p0_validated_topology_token(
                topology_token,
                block_id=block_id,
                prepared=prepared,
                certificate_binding=certificate_binding,
                native_ops=native,
                device=geometry.device,
                immutable_generation_id=immutable_generation_id,
                source_site_ids_i64=source_site_ids,
                source_track_ids_i64=source_track_ids,
            )
    compact_sites = gather_prepared_rows(geometry, source_site_ids)
    compact_rays = gather_prepared_rows(global_rays, source_track_ids)
    compact_density = gather_prepared_rows(density, source_site_ids)
    compact_color = gather_prepared_rows(color, source_site_ids)
    compact_rgba = torch.cat((compact_color, compact_density[:, None]), dim=1).contiguous()
    if material_training_mode:
        ordered_plan.assert_fixed_camera_affine_coefficients(
            compact_rays,
            track_start=staging.stage_track_start,
            track_end=staging.stage_track_end,
        )
        first_stage = ordered_plan.stage_targets(
            track_start=staging.stage_track_start,
            track_end=staging.stage_track_end,
            sample_start=first_range[1],
            sample_end=first_range[2],
        )
    assert first_stage is not None
    _validate_staged_camera_block(
        first_stage,
        ordered_plan,
        track_start=staging.stage_track_start,
        track_end=staging.stage_track_end,
        sample_start=first_range[1],
        sample_end=first_range[2],
        compact_rays=compact_rays,
        validate_static_camera_program=not material_training_mode,
        global_track_count=ledger.global_track_count,
        global_sample_count=ledger.global_frame_count,
        view_factor=staging.view_factor,
    )

    world_token = native.refresh_fixed_word_p0_world_token(
        topology_token,
        compact_sites,
        compact_rgba,
        compact_rays,
        replay_config,
        physical_length_epsilon=physical_length_epsilon,
        cone_tolerance=cone_tolerance,
    )
    expected_chart_partitions = tuple(
        (
            certificate_binding.charts[partition.chart_index].chart_digest,
            partition.global_start,
            partition.global_end,
        )
        for partition in partitions
    )
    sample_partition_generation_id = _sample_partition_generation_id(
        ordered_plan,
        loss_normalization_id=ledger.loss_normalization_id,
        global_track_count=ledger.global_track_count,
        global_sample_count=ledger.global_frame_count,
    )
    world_grad_init = (
        native.fixed_word_p0_lie_material_world_grad_init_launch_only
        if material_training_mode
        else native.fixed_word_p0_lie_world_grad_init_launch_only
    )
    world_grad = world_grad_init(
        world_token,
        expected_chart_partitions=expected_chart_partitions,
        global_track_count=ledger.global_track_count,
        global_sample_count=ledger.global_frame_count,
        global_loss_element_count=ledger.global_loss_element_count,
        loss_normalization_id=ledger.loss_normalization_id,
        sample_partition_generation_id=sample_partition_generation_id,
    )
    background = (
        torch.as_tensor(
            background_rgb,
            dtype=torch.float32,
            device=geometry.device,
        )
        .reshape(3)
        .contiguous()
    )
    block_loss = torch.zeros((), dtype=torch.float32, device=geometry.device)
    sample_block_count = 0
    device_barrier_count = 0
    sample_weight_evaluations: set[str] = set()
    sample_weight_linear_interactions = 0
    sample_weight_dense_fallback_interactions = 0
    sample_weight_exact_node_rows = 0
    sample_weight_dense_fallback_rows = 0
    peak_staged_target_bytes = 0
    peak_staged_explicit_ray_bytes = 0
    peak_staged_sample_time_bytes = 0
    for partition in partitions:
        runtime_chart = _prepared_charts(prepared)[partition.chart_index]
        chart_token = native.prepare_fixed_word_p0_chart_token(
            world_token,
            _chart_node_times(runtime_chart).to(device=geometry.device, dtype=torch.float32),
            chart_index=partition.chart_index,
        )
        certified_chart = certificate_binding.charts[partition.chart_index]
        if (
            chart_token.world is not world_token
            or chart_token.chart_index != partition.chart_index
            or chart_token.chart_generation_id != certified_chart.chart_digest
            or chart_token.node_count != _chart_node_count(runtime_chart)
        ):
            raise ValueError("native chart token does not match the sealed compact chart identity")
        sample_state = native.prepare_fixed_word_p0_sample_state_token(
            chart_token,
            global_track_count=ledger.global_track_count,
            global_sample_count=ledger.global_frame_count,
            global_sample_start=partition.global_start,
            global_sample_end=partition.global_end,
            global_loss_element_count=ledger.global_loss_element_count,
            loss_normalization_id=ledger.loss_normalization_id,
            sample_partition_generation_id=sample_partition_generation_id,
            sample_block_size=sample_block_size,
        )
        for sample_block_id, sample_start, sample_end in _chart_sample_blocks(
            partition,
            sample_block_size=sample_block_size,
            block_id=block_id,
        ):
            if sample_start == first_range[1] and sample_end == first_range[2] and partition is partitions[0]:
                staged = first_stage
            else:
                if material_training_mode:
                    staged = ordered_plan.stage_targets(
                        track_start=staging.stage_track_start,
                        track_end=staging.stage_track_end,
                        sample_start=sample_start,
                        sample_end=sample_end,
                    )
                else:
                    staged = ordered_plan.stage(
                        track_start=staging.stage_track_start,
                        track_end=staging.stage_track_end,
                        sample_start=sample_start,
                        sample_end=sample_end,
                        require_affine_ray_program=True,
                    )
                _validate_staged_camera_block(
                    staged,
                    ordered_plan,
                    track_start=staging.stage_track_start,
                    track_end=staging.stage_track_end,
                    sample_start=sample_start,
                    sample_end=sample_end,
                    compact_rays=compact_rays,
                    validate_static_camera_program=False,
                    global_track_count=ledger.global_track_count,
                    global_sample_count=ledger.global_frame_count,
                    view_factor=staging.view_factor,
                )
            peak_staged_target_bytes = max(
                peak_staged_target_bytes,
                int(staged.accounting["target_bytes"]),
            )
            peak_staged_explicit_ray_bytes = max(
                peak_staged_explicit_ray_bytes,
                int(staged.accounting["ray_bytes"]),
            )
            block_sample_times_f64 = ordered_plan.sample_times[sample_start:sample_end].to(
                device="cpu", dtype=torch.float64
            )
            peak_staged_sample_time_bytes = max(
                peak_staged_sample_time_bytes,
                block_sample_times_f64.numel() * block_sample_times_f64.element_size(),
            )
            sample_block = native.prepare_fixed_word_p0_sample_block_token(
                sample_state,
                staged.targets,
                background,
                sample_t_f64=block_sample_times_f64,
                sample_block_id=sample_block_id,
                global_sample_start=sample_start,
                global_sample_end=sample_end,
            )
            sample_weight_evaluations.add(sample_block.sample_weight_evaluation)
            sample_weight_linear_interactions += sample_block.sample_weight_linear_interactions
            sample_weight_dense_fallback_interactions += sample_block.sample_weight_dense_fallback_interactions
            sample_weight_exact_node_rows += sample_block.sample_weight_exact_node_rows
            sample_weight_dense_fallback_rows += sample_block.sample_weight_dense_fallback_rows
            native.fixed_word_p0_lie_sample_accumulate_loss_only_launch_only(
                sample_block,
                sample_state,
            )
            _synchronize_device(geometry.device, device_synchronize)
            device_barrier_count += 1
            del sample_block, staged, block_sample_times_f64
            if first_stage is not None and sample_start == first_range[1] and sample_end == first_range[2]:
                first_stage = None
            sample_block_count += 1
        block_loss.add_(sample_state.loss_f32)
        if material_training_mode:
            native.fixed_word_p0_lie_material_node_vjp_accumulate_launch_only(
                chart_token,
                sample_state,
                world_grad,
            )
        else:
            native.fixed_word_p0_lie_node_vjp_accumulate_launch_only(
                chart_token,
                sample_state,
                world_grad,
            )
        _synchronize_device(geometry.device, device_synchronize)
        device_barrier_count += 1
        del sample_state, chart_token
    if material_training_mode:
        grad_site_rgba = native.fixed_word_p0_lie_material_world_grad_finalize_launch_only(world_grad)
        grad_sites = None
    else:
        native.fixed_word_p0_sparse_mobius_boundary_finalize_launch_only(world_grad)
        grad_sites = native.fixed_word_p0_site_geometry_finalize_launch_only(world_grad)
        grad_site_rgba = world_grad.grad_site_rgba_f32
    geometry_vjp_executed = grad_sites is not None
    outputs = (block_loss, grad_site_rgba, *((grad_sites,) if grad_sites is not None else ()))
    result = NativeFixedWordP0TrackBlockResult(
        block_id=block_id,
        prepared=prepared,
        certificate_binding=certificate_binding,
        loss=block_loss,
        grad_site_geometry_f32=grad_sites,
        grad_site_rgba_f32=grad_site_rgba,
        geometry_vjp_executed=geometry_vjp_executed,
        loss_normalization_id=ledger.loss_normalization_id,
        sample_partition_generation_id=sample_partition_generation_id,
        global_track_count=ledger.global_track_count,
        global_sample_count=ledger.global_frame_count,
        sample_block_count=sample_block_count,
        chart_count=len(partitions),
        device_barrier_count=device_barrier_count,
        sample_payload_layout=("target_only" if material_training_mode else "target_plus_explicit_rays"),
        peak_staged_target_bytes=peak_staged_target_bytes,
        peak_staged_explicit_ray_bytes=peak_staged_explicit_ray_bytes,
        peak_staged_sample_time_bytes=peak_staged_sample_time_bytes,
        sample_weight_evaluation="+".join(sorted(sample_weight_evaluations)),
        sample_weight_linear_interactions=sample_weight_linear_interactions,
        sample_weight_dense_fallback_interactions=(sample_weight_dense_fallback_interactions),
        sample_weight_exact_node_rows=sample_weight_exact_node_rows,
        sample_weight_dense_fallback_rows=sample_weight_dense_fallback_rows,
        tensor_signatures=tuple(_tensor_signature(tensor) for tensor in outputs),
        source_site_ids_i64=source_site_ids,
        source_site_ids_signature=_tensor_signature(source_site_ids),
        validated_topology_token=reusable_topology,
        _ledger=ledger,
    )
    setattr(ledger, _INFLIGHT_LEDGER_ATTRIBUTE, block_id)
    return result


def consume_native_fixed_word_p0_track_block_result(
    ledger: CompactSpatialStepLedger,
    result: NativeFixedWordP0TrackBlockResult,
) -> None:
    """Index-add a certified native block into existing caller-owned bars."""

    _assert_compact_spatial_step_current(ledger)
    if result._ledger is not ledger:
        raise ValueError("native track result belongs to a different global gradient ledger")
    if getattr(ledger, _INFLIGHT_LEDGER_ATTRIBUTE, None) != result.block_id:
        raise ValueError("native track result is not the ledger's one registered in-flight block")
    if ledger.finalized:
        raise ValueError("compact spatial step was already finalized")
    track_start, track_end = _registered_track_range(ledger, result.block_id)
    if result.block_id in ledger.consumed_block_ids:
        raise ValueError("compact spatial block was already consumed")
    _assert_binding_matches_prepared(result.certificate_binding, result.prepared)
    if result.loss_normalization_id != ledger.loss_normalization_id:
        raise ValueError("native track result uses a different global loss normalization")
    if (
        result.global_track_count != ledger.global_track_count
        or result.global_sample_count != ledger.global_frame_count
    ):
        raise ValueError("native track result uses different global counts")
    expected_track_ids = torch.arange(
        track_start,
        track_end,
        dtype=result.prepared.topology.source_track_ids.dtype,
        device=result.prepared.topology.source_track_ids.device,
    )
    if not torch.equal(result.prepared.topology.source_track_ids, expected_track_ids):
        raise ValueError("native track result does not match its registered track range")
    outputs = (
        result.loss,
        result.grad_site_rgba_f32,
        *((result.grad_site_geometry_f32,) if result.grad_site_geometry_f32 is not None else ()),
    )
    if tuple(_tensor_signature(tensor) for tensor in outputs) != result.tensor_signatures:
        raise ValueError("native track result tensors changed before global accumulation")
    site_count = result.prepared.topology.site_count
    if result.geometry_vjp_executed != (result.grad_site_geometry_f32 is not None):
        raise ValueError("native geometry VJP accounting disagrees with its output payload")
    if result.geometry_vjp_executed:
        assert result.grad_site_geometry_f32 is not None
        if tuple(result.grad_site_geometry_f32.shape) != (site_count, 5):
            raise ValueError("native site geometry gradient must have shape [compact_sites,5]")
    else:
        if getattr(result.certificate_binding, "binding_mode", "") != "training_owner_topology_only":
            raise ValueError("only material training may omit the native geometry VJP")
        _assert_frozen_geometry_bars_zero(ledger)
    if tuple(result.grad_site_rgba_f32.shape) != (site_count, 4):
        raise ValueError("native site RGBA gradient must have shape [compact_sites,4]")
    if result.loss.ndim != 0:
        raise ValueError("native track result loss must be scalar")
    material_training_mode = getattr(result.certificate_binding, "binding_mode", "") == "training_owner_topology_only"
    expected_payload_layout = "target_only" if material_training_mode else "target_plus_explicit_rays"
    if result.sample_payload_layout != expected_payload_layout:
        raise ValueError("native sample payload layout disagrees with its binding capability")
    if result.peak_staged_target_bytes < 1:
        raise ValueError("native track result staged no target payload")
    if result.peak_staged_sample_time_bytes < 1:
        raise ValueError("native track result staged no bounded sample-time payload")
    if material_training_mode and result.peak_staged_explicit_ray_bytes != 0:
        raise ValueError("material training retained an explicit staged ray payload")
    if not material_training_mode and result.peak_staged_explicit_ray_bytes < 1:
        raise ValueError("strict native evaluation omitted its explicit staged ray payload")
    if not result.sample_weight_evaluation.startswith("verified_fit_derived_second_form_barycentric"):
        raise ValueError("native track result has unverified interpolation-weight provenance")
    if (
        result.sample_weight_linear_interactions < 1
        or result.sample_weight_dense_fallback_interactions < 0
        or result.sample_weight_exact_node_rows < 0
        or result.sample_weight_dense_fallback_rows < 0
        or result.sample_weight_exact_node_rows + result.sample_weight_dense_fallback_rows > result.global_sample_count
    ):
        raise ValueError("native track result interpolation-weight accounting is invalid")
    for tensor in outputs:
        if tensor.dtype != ledger.loss.dtype or tensor.device != ledger.loss.device:
            raise ValueError("native outputs must match the caller-owned gradient ledger")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError("native outputs must be finite")

    source_site_ids = result.source_site_ids_i64
    if _tensor_signature(source_site_ids) != result.source_site_ids_signature:
        raise ValueError("native track result compact source-site identity changed")
    if source_site_ids.dtype != torch.long or source_site_ids.device != ledger.loss.device:
        raise ValueError("native track result compact source-site identity is on the wrong device")
    if result.geometry_vjp_executed:
        assert result.grad_site_geometry_f32 is not None
        accumulate_prepared_rows_(
            ledger.gradients.grad_site_geometry,
            result.grad_site_geometry_f32[:, :4],
            source_site_ids,
        )
        accumulate_prepared_rows_(
            ledger.gradients.grad_site_weight,
            result.grad_site_geometry_f32[:, 4],
            source_site_ids,
        )
    accumulate_prepared_rows_(
        ledger.gradients.grad_site_color,
        result.grad_site_rgba_f32[:, :3],
        source_site_ids,
    )
    accumulate_prepared_rows_(
        ledger.gradients.grad_site_density,
        result.grad_site_rgba_f32[:, 3],
        source_site_ids,
    )
    if not result.geometry_vjp_executed:
        _assert_frozen_geometry_bars_zero(ledger)
    ledger.loss.add_(result.loss)
    ledger.compact_site_rows_accumulated += int(source_site_ids.numel())
    ledger.consumed_block_ids.add(result.block_id)
    ledger.state_tensor_signatures = tuple(
        _tensor_signature(tensor) for tensor in (*ledger.gradients.tensors, ledger.loss)
    )
    delattr(ledger, _INFLIGHT_LEDGER_ATTRIBUTE)
    _assert_compact_spatial_step_current(ledger)


def _assert_frozen_geometry_bars_zero(ledger: CompactSpatialStepLedger) -> None:
    """Protect the material-only promise that omitted bars stay untouched."""

    for name, tensor in (
        ("geometry", ledger.gradients.grad_site_geometry),
        ("weight", ledger.gradients.grad_site_weight),
    ):
        if bool(torch.any(tensor != 0.0).item()):
            raise ValueError(f"material-only native result found a nonzero frozen {name} bar")


def _registered_track_range(ledger: CompactSpatialStepLedger, block_id: str) -> tuple[int, int]:
    if not block_id.strip():
        raise ValueError("block_id must be nonempty")
    matches = tuple(record for record in ledger.expected_blocks if record[0] == block_id)
    if len(matches) != 1:
        raise ValueError("native spatial block id is not registered")
    return matches[0][1], matches[0][2]


def _assert_binding_matches_prepared(certificate_binding: Any, prepared: Any) -> None:
    if not hasattr(certificate_binding, "assert_current"):
        raise ValueError("native track adapter requires a sealed current certificate binding")
    certificate_binding.assert_current()
    binding_mode = getattr(certificate_binding, "binding_mode", "")
    if binding_mode == "strict_frozen_evaluation":
        prepared.assert_current()
        if getattr(certificate_binding, "_prepared", None) is not prepared:
            raise ValueError("continuous certificate binding belongs to a different compact prepared snapshot")
    elif binding_mode == "training_owner_topology_only":
        if (
            bool(getattr(certificate_binding, "paper_evidence_eligible", True))
            or bool(getattr(certificate_binding, "transfer_jacobian_certified", True))
            or bool(getattr(certificate_binding, "approximation_error_certified", True))
        ):
            raise ValueError("material-training binding must remain uncertified and non-paper")
        if getattr(prepared, "owner_binding", None) is certificate_binding:
            prepared.assert_current()
        else:
            assert_prepared_immutable = getattr(certificate_binding, "assert_prepared_immutable", None)
            if not callable(assert_prepared_immutable):
                raise ValueError("material-training binding cannot validate immutable prepared state")
            assert_prepared_immutable(prepared)
    else:
        raise ValueError("native track adapter received an unknown binding mode")
    if not str(getattr(certificate_binding, "canonical_digest", "")).strip():
        raise ValueError("native fixed-word binding has no canonical digest")
    if len(getattr(certificate_binding, "charts", ())) != len(_prepared_charts(prepared)):
        raise ValueError("native fixed-word binding and compact atlas disagree on chart count")


def _resolve_staging_layout(
    plan: PowerFoamTrackStagingPlan,
    ledger: CompactSpatialStepLedger,
    *,
    track_start: int,
    track_end: int,
) -> _ResolvedStagingLayout:
    """Select one canonical view-major track row without staging other views."""

    frame_count = plan.target_provider.frame_count
    views = torch.div(plan.sample_indices, frame_count, rounding_mode="floor")
    active_views = tuple(sorted({int(view) for view in views.tolist()}))
    if not active_views:
        raise ValueError("native track staging requires at least one selected camera view")
    if len(active_views) == 1:
        if plan.track_count != ledger.global_track_count:
            raise ValueError("single-view staging plan and global ledger disagree on track count")
        if plan.sample_count != ledger.global_frame_count:
            raise ValueError("single-view staging plan and global ledger disagree on sample count")
        return _ResolvedStagingLayout(
            plan=plan,
            stage_track_start=track_start,
            stage_track_end=track_end,
            view_factor=1,
        )

    view_positions = []
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
            raise NativeTrackAdapterUnavailableError(
                "mixed fixed views must form one rectangular frame/time grid before view-track expansion"
            )
        view_positions.append(positions)
    if reference_frames is None or reference_times is None:
        raise ValueError("view-track expansion found no temporal samples")
    global_pixel_count = plan.track_count
    temporal_sample_count = int(reference_times.numel())
    if ledger.global_track_count != len(active_views) * global_pixel_count:
        raise ValueError("view-track ledger must contain selected_views * global_pixels tracks")
    if ledger.global_frame_count != temporal_sample_count:
        raise ValueError("view-track ledger sample count must equal the rectangular temporal grid")
    view_local_index = track_start // global_pixel_count
    if view_local_index >= len(active_views):
        raise ValueError("registered view-track range starts outside the selected view-major grid")
    view_track_start = view_local_index * global_pixel_count
    view_track_end = view_track_start + global_pixel_count
    if track_end > view_track_end:
        raise NativeTrackAdapterUnavailableError(
            "one native spatial block cannot cross a canonical view-major track boundary"
        )
    pixel_start = track_start - view_track_start
    pixel_end = track_end - view_track_start
    positions = view_positions[view_local_index]
    view_plan = PowerFoamTrackStagingPlan(
        target_provider=plan.target_provider,
        ray_provider=plan.ray_provider,
        pixel_indices=plan.pixel_indices,
        sample_indices=plan.sample_indices.index_select(0, positions),
        height=plan.height,
        width=plan.width,
        sample_times=plan.sample_times.index_select(0, positions),
        device=plan.device,
    )
    return _ResolvedStagingLayout(
        plan=view_plan,
        stage_track_start=pixel_start,
        stage_track_end=pixel_end,
        view_factor=len(active_views),
    )


def _ordered_chart_staging_plan(
    plan: PowerFoamTrackStagingPlan,
    prepared: Any,
    *,
    device: torch.device,
) -> tuple[PowerFoamTrackStagingPlan, tuple[_ChartSamplePartition, ...]]:
    charts = _prepared_charts(prepared)
    assignments = tuple(_chart_for_time(float(time), charts) for time in plan.sample_times.tolist())
    chart_buckets: list[list[int]] = [[] for _ in charts]
    for sample_index, chart_index in enumerate(assignments):
        chart_buckets[chart_index].append(sample_index)
    permutation = [sample_index for bucket in chart_buckets for sample_index in bucket]
    order = torch.tensor(permutation, dtype=torch.long)
    ordered_assignments = tuple(assignments[index] for index in permutation)
    ordered = PowerFoamTrackStagingPlan(
        target_provider=plan.target_provider,
        ray_provider=plan.ray_provider,
        pixel_indices=plan.pixel_indices,
        sample_indices=plan.sample_indices.index_select(0, order),
        height=plan.height,
        width=plan.width,
        sample_times=plan.sample_times.index_select(0, order),
        device=device,
    )
    partitions = []
    cursor = 0
    while cursor < ordered.sample_count:
        chart_index = ordered_assignments[cursor]
        stop = cursor + 1
        while stop < ordered.sample_count and ordered_assignments[stop] == chart_index:
            stop += 1
        partitions.append(
            _ChartSamplePartition(
                chart_index=chart_index,
                global_start=cursor,
                global_end=stop,
            )
        )
        cursor = stop
    return ordered, tuple(partitions)


def _chart_sample_blocks(
    partition: _ChartSamplePartition,
    *,
    sample_block_size: int,
    block_id: str,
):
    """Yield the deterministic K partition without retaining O(F/K) records."""

    for start in range(partition.global_start, partition.global_end, sample_block_size):
        end = min(start + sample_block_size, partition.global_end)
        yield f"{block_id}:chart-{partition.chart_index}:samples-{start}-{end}", start, end


def _chart_for_time(time: float, charts: tuple[Any, ...]) -> int:
    matches = []
    for index, chart in enumerate(charts):
        t_min = _chart_t_min(chart)
        t_max = _chart_t_max(chart)
        if time >= t_min and (time <= t_max if index + 1 == len(charts) else time < t_max):
            matches.append(index)
    if len(matches) != 1:
        raise NativeTrackAdapterUnavailableError(f"sample time {time!r} is not owned by exactly one certified chart")
    return matches[0]


def _prepared_charts(prepared: Any) -> tuple[Any, ...]:
    world_snapshot = getattr(prepared, "world_snapshot", None)
    if world_snapshot is not None:
        charts = tuple(world_snapshot.atlas.charts)
    else:
        schedule = getattr(prepared, "schedule", None)
        charts = tuple(getattr(schedule, "charts", ()))
    if not charts:
        raise ValueError("native fixed-word block requires at least one chart")
    return charts


def _chart_node_times(chart: Any) -> torch.Tensor:
    transfer_atlas = getattr(chart, "transfer_atlas", None)
    value = transfer_atlas.node_times if transfer_atlas is not None else getattr(chart, "node_times", None)
    if not isinstance(value, torch.Tensor):
        raise TypeError("native fixed-word chart has no tensor node schedule")
    return value


def _chart_node_count(chart: Any) -> int:
    value = getattr(chart, "node_count", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError("native fixed-word chart node_count must be an integer >= 2")
    return value


def _chart_t_min(chart: Any) -> float:
    transfer_atlas = getattr(chart, "transfer_atlas", None)
    return float(transfer_atlas.t_min if transfer_atlas is not None else chart.t_min)


def _chart_t_max(chart: Any) -> float:
    transfer_atlas = getattr(chart, "transfer_atlas", None)
    return float(transfer_atlas.t_max if transfer_atlas is not None else chart.t_max)


def native_fixed_word_p0_topology_cache_key(
    *,
    block_id: str,
    prepared: Any,
    certificate_binding: Any,
    native_ops: Any,
    device: torch.device,
    immutable_generation_id: str,
) -> NativeFixedWordP0TopologyCacheKey:
    """Build the exact lookup key for one material-training topology token."""

    if getattr(certificate_binding, "binding_mode", "") != "training_owner_topology_only":
        raise ValueError("native topology caching is restricted to material training")
    values = {
        "block_id": block_id,
        "immutable_generation_id": immutable_generation_id,
        "binding_digest": str(getattr(certificate_binding, "canonical_digest", "")),
        "topology_generation_id": str(getattr(certificate_binding, "topology_snapshot_generation", "")),
        "schedule_generation_digest": str(getattr(getattr(prepared, "schedule", None), "generation_digest", "")),
    }
    for name, value in values.items():
        if not value.strip():
            raise ValueError(f"native topology cache key requires nonempty {name}")
    return NativeFixedWordP0TopologyCacheKey(
        block_id=block_id,
        native_ops_identity=id(native_ops),
        device=str(torch.device(device)),
        immutable_generation_id=immutable_generation_id,
        binding_digest=values["binding_digest"],
        topology_generation_id=values["topology_generation_id"],
        schedule_generation_digest=values["schedule_generation_digest"],
    )


def estimate_native_fixed_word_p0_topology_token_resident_bytes(prepared: Any) -> int:
    """Conservative exact-shape preflight for one validated native token.

    The native token owns seven CSR/source topology tensors plus a separately
    materialized active-boundary pair table.  The validated wrapper also owns
    compact int64 source-site and source-track maps.  This estimate is checked
    against the actual sealed token after preparation; it exists so an
    oversized spatial block can fail before allocating device topology.
    """

    topology = getattr(prepared, "topology", None)
    if topology is None:
        raise TypeError("native topology byte estimation requires prepared.topology")
    native_source_names = (
        "word_offsets_i32",
        "word_owner_i32",
        "word_left_incidence_i32",
        "word_right_incidence_i32",
        "track_incidence_offsets_i32",
        "incidence_boundary_i32",
        "boundary_site_pairs_i32",
    )
    native_sources = tuple(getattr(topology, name, None) for name in native_source_names)
    if any(not torch.is_tensor(tensor) for tensor in native_sources):
        raise TypeError("prepared topology does not expose the complete native CSR payload")
    if any(tensor.dtype != torch.int32 for tensor in native_sources):
        raise ValueError("prepared native CSR topology must use int32")
    boundary_pairs = native_sources[-1]
    source_site_ids = getattr(topology, "source_site_ids", None)
    source_track_ids = getattr(topology, "source_track_ids", None)
    if not torch.is_tensor(source_site_ids) or not torch.is_tensor(source_track_ids):
        raise TypeError("prepared topology does not expose compact source identities")
    return int(
        4 * sum(tensor.numel() for tensor in native_sources)
        + 4 * boundary_pairs.numel()
        + 8 * (source_site_ids.numel() + source_track_ids.numel())
    )


def assert_native_fixed_word_p0_validated_topology_token(
    validated: NativeFixedWordP0ValidatedTopologyToken,
    *,
    block_id: str,
    prepared: Any,
    certificate_binding: Any,
    native_ops: Any,
    device: torch.device,
    immutable_generation_id: str,
) -> None:
    """Fail closed unless a cached token still has its exact sealed identity."""

    if not isinstance(validated, NativeFixedWordP0ValidatedTopologyToken):
        raise TypeError("cached native topology must be a validated topology token")
    if validated._seal is not _VALIDATED_TOPOLOGY_TOKEN_SEAL:
        raise ValueError("cached native topology token was not sealed by the adapter")
    expected_key = native_fixed_word_p0_topology_cache_key(
        block_id=block_id,
        prepared=prepared,
        certificate_binding=certificate_binding,
        native_ops=native_ops,
        device=device,
        immutable_generation_id=immutable_generation_id,
    )
    if validated.cache_key != expected_key:
        raise ValueError("cached native topology token identity is stale or mismatched")
    if validated.native_ops is not native_ops:
        raise ValueError("cached native topology token belongs to different native ops")
    if validated.certificate_binding is not certificate_binding:
        raise ValueError("cached native topology token belongs to a different binding")
    token_binding = getattr(validated.native_token, "certificate_binding", None)
    if token_binding is not certificate_binding:
        raise ValueError("cached native topology payload belongs to a different binding")
    topology = prepared.topology
    if (
        int(getattr(validated.native_token, "track_count", -1)) != topology.track_count
        or int(getattr(validated.native_token, "site_count", -1)) != topology.site_count
    ):
        raise ValueError("cached native topology payload has different compact dimensions")
    token_generation = getattr(validated.native_token, "topology_generation_id", None)
    if token_generation is not None and str(token_generation) != expected_key.topology_generation_id:
        raise ValueError("cached native topology payload has a stale topology generation")
    token_binding_digest = getattr(validated.native_token, "training_binding_digest", None)
    if token_binding_digest is not None and str(token_binding_digest) != expected_key.binding_digest:
        raise ValueError("cached native topology payload has a different binding digest")
    token_binding_mode = getattr(validated.native_token, "binding_mode", None)
    if token_binding_mode is not None and token_binding_mode != "training_owner_topology_only":
        raise ValueError("cached native topology payload is not a material-training token")
    native_tensors = _native_topology_token_tensors(validated.native_token)
    if tuple(_tensor_signature(tensor) for tensor in native_tensors) != (validated.native_tensor_signatures):
        raise ValueError("cached native topology device tensors changed after validation")
    source_ids = (validated.source_site_ids_i64, validated.source_track_ids_i64)
    if tuple(_tensor_signature(tensor) for tensor in source_ids) != (validated.source_id_tensor_signatures):
        raise ValueError("cached native topology source identities changed after validation")
    if any(tensor.device != torch.device(device) for tensor in (*native_tensors, *source_ids)):
        raise ValueError("cached native topology tensors are on a different device")
    if any(tensor.dtype != torch.long for tensor in source_ids):
        raise ValueError("cached native topology source identities must use int64")
    if (
        validated.source_site_ids_i64.numel() != topology.site_count
        or validated.source_track_ids_i64.numel() != topology.track_count
    ):
        raise ValueError("cached native topology source identities have different dimensions")
    if any(tensor.dtype != torch.int32 for tensor in native_tensors):
        raise ValueError("cached native topology device tensors must use int32")


def _seal_native_fixed_word_p0_validated_topology_token(
    native_token: Any,
    *,
    block_id: str,
    prepared: Any,
    certificate_binding: Any,
    native_ops: Any,
    device: torch.device,
    immutable_generation_id: str,
    source_site_ids_i64: torch.Tensor,
    source_track_ids_i64: torch.Tensor,
) -> NativeFixedWordP0ValidatedTopologyToken:
    key = native_fixed_word_p0_topology_cache_key(
        block_id=block_id,
        prepared=prepared,
        certificate_binding=certificate_binding,
        native_ops=native_ops,
        device=device,
        immutable_generation_id=immutable_generation_id,
    )
    native_tensors = _native_topology_token_tensors(native_token)
    validated = NativeFixedWordP0ValidatedTopologyToken(
        cache_key=key,
        native_token=native_token,
        native_ops=native_ops,
        certificate_binding=certificate_binding,
        source_site_ids_i64=source_site_ids_i64,
        source_track_ids_i64=source_track_ids_i64,
        native_tensor_signatures=tuple(_tensor_signature(tensor) for tensor in native_tensors),
        source_id_tensor_signatures=tuple(
            _tensor_signature(tensor) for tensor in (source_site_ids_i64, source_track_ids_i64)
        ),
        _seal=_VALIDATED_TOPOLOGY_TOKEN_SEAL,
    )
    assert_native_fixed_word_p0_validated_topology_token(
        validated,
        block_id=block_id,
        prepared=prepared,
        certificate_binding=certificate_binding,
        native_ops=native_ops,
        device=device,
        immutable_generation_id=immutable_generation_id,
    )
    return validated


def _native_topology_token_tensors(native_token: Any) -> tuple[torch.Tensor, ...]:
    names = (
        "word_offsets_i32",
        "word_owner_i32",
        "word_left_incidence_i32",
        "word_right_incidence_i32",
        "track_incidence_offsets_i32",
        "incidence_boundary_i32",
        "boundary_site_pairs_i32",
        "active_boundary_site_pairs_i32",
    )
    present = tuple(hasattr(native_token, name) for name in names)
    if any(present):
        if not all(present):
            raise TypeError("native topology token exposes an incomplete tensor payload")
        tensors = tuple(getattr(native_token, name) for name in names)
    else:
        tensors = tuple(getattr(native_token, "tensors", ()))
    if not tensors or any(not torch.is_tensor(tensor) for tensor in tensors):
        raise TypeError("native topology token exposes no immutable tensor payload")
    return tensors


def _unique_tensor_storage_bytes(tensors: tuple[torch.Tensor, ...]) -> int:
    storages: dict[tuple[str, int, int], int] = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storage_bytes = int(storage.nbytes())
        storages.setdefault(
            (str(tensor.device), int(storage.data_ptr()), storage_bytes),
            storage_bytes,
        )
    return sum(storages.values())


def _validate_staged_camera_block(
    staged: PowerFoamTrackStageBlock | PowerFoamTrackTargetStageBlock,
    plan: PowerFoamTrackStagingPlan,
    *,
    track_start: int,
    track_end: int,
    sample_start: int,
    sample_end: int,
    compact_rays: torch.Tensor,
    validate_static_camera_program: bool,
    global_track_count: int,
    global_sample_count: int,
    view_factor: int,
) -> None:
    if staged.normalization.global_track_count != plan.track_count:
        raise ValueError("staged block changed the global track normalization")
    if staged.normalization.global_sample_count != plan.sample_count:
        raise ValueError("staged block changed the global sample normalization")
    if staged.normalization.global_rgb_element_count != plan.track_count * plan.sample_count * 3:
        raise ValueError("staged block changed the global RGB denominator")
    if global_track_count != plan.track_count * view_factor:
        raise ValueError("view-track factorization changed the global track count")
    if global_sample_count != plan.sample_count:
        raise ValueError("view-track factorization changed the global sample count")
    if global_track_count * global_sample_count * 3 != (staged.normalization.global_rgb_element_count * view_factor):
        raise ValueError("view-track factorization changed the global RGB denominator")
    if staged.normalization.block_track_count != track_end - track_start:
        raise ValueError("staged block track count does not match its spatial partition")
    if staged.normalization.block_sample_count != sample_end - sample_start:
        raise ValueError("staged block sample count does not match its temporal partition")
    for name, actual, expected in (
        ("pixel", staged.pixel_indices, plan.pixel_indices[track_start:track_end]),
        ("sample", staged.sample_indices, plan.sample_indices[sample_start:sample_end]),
        ("time", staged.sample_times, plan.sample_times[sample_start:sample_end]),
    ):
        if not torch.equal(actual.to(device="cpu"), expected.to(device="cpu")):
            raise ValueError(f"staged {name} identities do not match the certified adapter partition")
    if not validate_static_camera_program:
        return
    if not isinstance(staged, PowerFoamTrackStageBlock):
        raise TypeError("strict native evaluation requires explicit staged rays")
    program = staged.affine_ray_program
    if program is None:
        reason = staged.affine_ray_program_unavailable_reason or "no exact affine ray program"
        raise NativeTrackAdapterUnavailableError(reason)
    if int(program.view_indices.numel()) != 1 or not bool(torch.all(program.sample_program_indices == 0).item()):
        raise NativeTrackAdapterUnavailableError(
            "native fixed-word track blocks require one affine camera-program row per pixel track"
        )
    if not torch.equal(program.sample_times.to(device="cpu"), staged.sample_times.to(device="cpu")):
        raise ValueError("staged affine camera program changed its sample-time identity")
    coefficients = program.coefficients[0]
    if coefficients.dtype != compact_rays.dtype or coefficients.device != compact_rays.device:
        raise ValueError("staged affine camera program must match live ray dtype and device")
    if not torch.equal(coefficients, compact_rays):
        raise ValueError("staged affine camera program does not match the certified live track rays")


def _synchronize_device(
    device: torch.device,
    callback: Callable[[torch.device], None] | None,
) -> None:
    """Bound in-flight K blocks before target/prediction references are dropped."""

    if callback is not None:
        callback(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _sample_partition_generation_id(
    plan: PowerFoamTrackStagingPlan,
    *,
    loss_normalization_id: str,
    global_track_count: int,
    global_sample_count: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(loss_normalization_id.encode("utf-8"))
    digest.update(str(global_track_count).encode("ascii"))
    digest.update(str(global_sample_count).encode("ascii"))
    frame_indices = torch.remainder(plan.sample_indices, plan.target_provider.frame_count)
    digest.update(frame_indices.to(device="cpu", dtype=torch.int64).contiguous().numpy().tobytes())
    digest.update(plan.sample_times.to(device="cpu", dtype=torch.float64).contiguous().numpy().tobytes())
    return f"worldfoam-track-samples:{digest.hexdigest()}"


def resolve_native_fixed_word_p0_ops(native_ops: Any | None = None) -> Any:
    if native_ops is not None:
        return native_ops
    try:
        return importlib.import_module("torch_world_foam_lane2_fused_slab")
    except ImportError as error:
        raise RuntimeError(
            "world_foam_lane2_fused_slab_v0 is not importable; add the built variant to sys.path"
        ) from error


__all__ = [
    "NativeFixedWordP0TrackBlockResult",
    "NativeFixedWordP0TopologyCacheKey",
    "NativeFixedWordP0ValidatedTopologyToken",
    "NativeTrackAdapterUnavailableError",
    "assert_native_fixed_word_p0_validated_topology_token",
    "consume_native_fixed_word_p0_track_block_result",
    "execute_native_fixed_word_p0_track_block",
    "estimate_native_fixed_word_p0_topology_token_resident_bytes",
    "native_fixed_word_p0_topology_cache_key",
    "resolve_native_fixed_word_p0_ops",
]
