"""End-to-end CPU/fake-native kinetic WorldFoam paper-step bridge.

This module closes the source-level integration path from arbitrary paper
observations to one material optimizer authorization.  It deliberately remains
CPU/fake-native: the native word forward and VJP are injected through the
already sealed :mod:`kinetic_native_equal_rank_runtime_adapter` ABI, while the
row-ragged sample reduction is a direct CPU implementation of the Metal
kernel's affine-Lie arithmetic.

The important scheduling decision is *spatial block outer, temporal chunk
inner*.  Request-local losses are returned to the generic paper coordinator as
soon as their bounded target block has been reduced.  Material gradients are
different: every native block's node cotangent accumulates across all temporal
requests for the current spatial block, intermediate coordinator requests
carry a zero compact bar, and the native ordered-word VJP runs exactly once at
the spatial block's final request.  Consequently:

* sample work is ``O(sum_samples J_selected)`` and may grow with requested F;
* ordered-word forward/reverse work is ``O(sum_active_blocks J_b W_b)`` and
  does not grow when F is densified at fixed compiled worlds;
* node charts/cotangents are retained only for the current spatial block, so
  peak expensive state is
  ``max_spatial_bundle sum_{q in bundle} rows_q J_q`` rather than a sum over
  every spatial block;
* one caller-owned global ``[S,4]`` material bar and one global RGB denominator
  are preserved by :mod:`paper_ragged_material_bar_coordinator`.

The default material step calls the material-only native VJP and allocates no
``[J,W]`` physical-length cotangent.  Geometry training remains an explicit
open path through the separate full-VJP runtime adapter; this module does
**not** claim stable-stratum length-to-geometry VJP or native Metal completion.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from kinetic_native_equal_rank_runtime_adapter import (
    KineticNativeEqualRankRuntimeBlock,
    KineticNativeEqualRankWorld,
    execute_kinetic_native_equal_rank_material_node_vjp,
    refresh_kinetic_native_equal_rank_world,
)
from paper_kinetic_ragged_sample_plan import (
    PaperKineticRowRaggedSampleBlock,
    iter_paper_kinetic_row_ragged_request_blocks,
)
from paper_kinetic_union_local_bar_assembly import (
    PaperKineticActiveNativeBlockWork,
    PaperKineticUnionLocalRequestWork,
    PaperKineticUnionLocalSpatialBundle,
    begin_paper_kinetic_union_local_bar_assembly,
    consume_paper_kinetic_union_local_native_contribution,
    finalize_paper_kinetic_union_local_bar_assembly,
    prepare_paper_kinetic_union_local_request_work,
    seal_paper_kinetic_native_block_vjp_contribution,
)
from paper_ragged_material_bar_coordinator import (
    PaperRaggedMaterialBarRequest,
    PaperRaggedMaterialBarStepResult,
    PaperRaggedMaterialSpatialBlock,
    PaperRaggedMaterialViewProgram,
    begin_paper_ragged_material_bar_step,
    consume_paper_ragged_compact_material_bar_result,
    finalize_paper_ragged_material_bar_step,
    seal_paper_ragged_compact_material_bar_result,
    stage_next_paper_ragged_material_bar_request,
)
from paper_ragged_track_staging import PaperRaggedTrackBatch

CPU_FAKE_NATIVE_STATUS = "cpu_fake_native/source_integration_only"
DEFERRED_WORK_PROVENANCE = "paper-kinetic-deferred-union-work-v1"

_LANE_SEAL = object()
_DEFERRED_WORK_SEAL = object()


@dataclass(frozen=True)
class PaperKineticCPUFakeNativeSpatialLane:
    """Cold-sealed runtime set for one generic coordinator spatial block."""

    spatial_block: PaperRaggedMaterialSpatialBlock = field(repr=False)
    bundle: PaperKineticUnionLocalSpatialBundle = field(repr=False)
    runtimes: tuple[KineticNativeEqualRankRuntimeBlock, ...] = field(repr=False)
    runtime_block_generation_digests: tuple[str, ...]
    generation_id: str
    _spatial_block_identity: int = field(repr=False)
    _bundle_identity: int = field(repr=False)
    _runtime_identities: tuple[int, ...] = field(repr=False)
    runtime_status: str = CPU_FAKE_NATIVE_STATUS
    _seal: object = field(default=None, repr=False)

    @property
    def view_index(self) -> int:
        return self.spatial_block.view_index

    @property
    def block_id(self) -> str:
        return self.spatial_block.block_id

    def runtime_for_digest(self, digest: str) -> KineticNativeEqualRankRuntimeBlock:
        matches = tuple(runtime for runtime in self.runtimes if runtime.payload.block.generation_digest == digest)
        if len(matches) != 1:
            raise ValueError("deferred kinetic lane has no unique runtime for native block")
        return matches[0]

    def assert_current(self) -> None:
        """Warm identity/layout validation; source contents were sealed cold."""

        if self._seal is not _LANE_SEAL or self.runtime_status != CPU_FAKE_NATIVE_STATUS:
            raise ValueError("deferred kinetic spatial lane is not a sealed CPU/fake-native lane")
        if (
            id(self.spatial_block) != self._spatial_block_identity
            or id(self.bundle) != self._bundle_identity
            or tuple(id(runtime) for runtime in self.runtimes) != self._runtime_identities
            or self.spatial_block.world_token is not self.bundle
            or self.spatial_block.world_generation_id != self.bundle.generation_digest
            or self.spatial_block.view_index != self.bundle.view_index
            or self.spatial_block.global_site_count != self.bundle.global_site_count
            or tuple(range(self.spatial_block.track_start, self.spatial_block.track_end)) != self.bundle.track_ids
            or self.runtime_block_generation_digests
            != tuple(binding.native_block_generation_digest for binding in self.bundle.native_blocks)
        ):
            raise ValueError("deferred kinetic spatial lane identity/provenance changed")
        self.spatial_block.assert_current()
        self.bundle.assert_warm_layout()
        for runtime in self.runtimes:
            runtime.assert_warm_layout()


def prepare_paper_kinetic_cpu_fake_native_spatial_lane(
    spatial_block: PaperRaggedMaterialSpatialBlock,
    bundle: PaperKineticUnionLocalSpatialBundle,
    *,
    runtimes: Sequence[KineticNativeEqualRankRuntimeBlock],
) -> PaperKineticCPUFakeNativeSpatialLane:
    """Cold-bind every union-local native block to one CPU runtime."""

    if not isinstance(spatial_block, PaperRaggedMaterialSpatialBlock):
        raise TypeError("spatial_block must be PaperRaggedMaterialSpatialBlock")
    if not isinstance(bundle, PaperKineticUnionLocalSpatialBundle):
        raise TypeError("bundle must be PaperKineticUnionLocalSpatialBundle")
    spatial_block.assert_current()
    bundle.assert_cold_current()
    if spatial_block.world_token is not bundle:
        raise ValueError("kinetic spatial block must use its union-local bundle as world token")
    if spatial_block.world_generation_id != bundle.generation_digest:
        raise ValueError("kinetic spatial block and union-local bundle generation differ")
    if tuple(range(spatial_block.track_start, spatial_block.track_end)) != bundle.track_ids:
        raise ValueError("kinetic spatial block tracks differ from the union-local bundle")

    normalized = tuple(runtimes)
    if not normalized or len({id(runtime) for runtime in normalized}) != len(normalized):
        raise ValueError("kinetic spatial lane requires unique nonempty runtimes")
    runtime_by_digest: dict[str, KineticNativeEqualRankRuntimeBlock] = {}
    for runtime in normalized:
        if not isinstance(runtime, KineticNativeEqualRankRuntimeBlock):
            raise TypeError("runtimes must contain KineticNativeEqualRankRuntimeBlock values")
        runtime.assert_warm_layout()
        if runtime.device.type != "cpu":
            raise ValueError("this integration proof accepts CPU/fake-native runtimes only")
        if runtime.global_site_count != bundle.global_site_count:
            raise ValueError("kinetic runtime changed the bundle global material table")
        digest = runtime.payload.block.generation_digest
        if digest in runtime_by_digest:
            raise ValueError("kinetic spatial lane received a duplicate native runtime block")
        runtime_by_digest[digest] = runtime

    expected_digests = tuple(binding.native_block_generation_digest for binding in bundle.native_blocks)
    if set(runtime_by_digest) != set(expected_digests):
        raise ValueError("kinetic spatial lane must bind every and only union-local native block")
    canonical = tuple(runtime_by_digest[digest] for digest in expected_digests)
    for binding, runtime in zip(bundle.native_blocks, canonical, strict=True):
        runtime_ids = tuple(int(value) for value in runtime.source_site_ids_i64.tolist())
        if runtime_ids != binding.compact_source_site_ids:
            raise ValueError("kinetic runtime compact source map differs from union-local binding")

    generation = _digest_parts(
        CPU_FAKE_NATIVE_STATUS,
        spatial_block.block_id,
        spatial_block.site_mapping_id,
        bundle.generation_digest,
        tuple(runtime.generation_id for runtime in canonical),
    )
    result = PaperKineticCPUFakeNativeSpatialLane(
        spatial_block=spatial_block,
        bundle=bundle,
        runtimes=canonical,
        runtime_block_generation_digests=expected_digests,
        generation_id=generation,
        _spatial_block_identity=id(spatial_block),
        _bundle_identity=id(bundle),
        _runtime_identities=tuple(id(runtime) for runtime in canonical),
        _seal=_LANE_SEAL,
    )
    result.assert_current()
    return result


@dataclass(frozen=True)
class PaperKineticDeferredUnionLocalRequestWork(PaperKineticUnionLocalRequestWork):
    """Aggregate sample coverage carried by the final coordinator request.

    This intentionally subclasses the union assembler's work token: the
    assembler still owns compact-to-union provenance and exactly-once native
    contributions, while this token strengthens ``active_blocks`` from one
    temporal request to the complete spatial-block temporal coverage.
    """

    deferred_request_count: int = 0
    covered_local_sample_start: int = 0
    covered_local_sample_end: int = 0
    carrier_request_sample_count: int = 0
    request_coverage_digest: str = ""
    provenance: str = DEFERRED_WORK_PROVENANCE
    _sealed_generation_digest: str = field(default="", repr=False)
    _deferred_seal: object = field(default=None, repr=False)

    def assert_warm_layout(self) -> None:
        if (
            self._deferred_seal is not _DEFERRED_WORK_SEAL
            or self.provenance != DEFERRED_WORK_PROVENANCE
            or self.generation_digest != self._sealed_generation_digest
            or id(self.bundle) != self._bundle_identity
            or id(self.sampler) != self._sampler_identity
            or id(self.request) != self._request_identity
            or self.sampler is not self.bundle.sampler
            or self.request.world_token is not self.bundle
            or self.maximum_samples_per_launch < 1
            or self.deferred_request_count < 1
            or self.covered_local_sample_start != 0
            or self.covered_local_sample_end != self.request.group.observation_count
            or self.request.local_sample_end != self.covered_local_sample_end
            or self.carrier_request_sample_count
            != self.request.block.track_count * (self.request.local_sample_end - self.request.local_sample_start)
            or self.total_sample_count
            != self.request.block.track_count * (self.covered_local_sample_end - self.covered_local_sample_start)
            or not self.request_coverage_digest.strip()
            or not self.active_blocks
            or self.persistent_sample_tensor_bytes != 0
            or self.persistent_target_tensor_bytes != 0
            or self.persistent_interpolation_weight_tensor_bytes != 0
            or self.retained_sample_partition_records != 0
            or any(active.sample_chunk_count < 1 or active.sample_count < 1 for active in self.active_blocks)
            or sum(active.sample_count for active in self.active_blocks) != self.total_sample_count
        ):
            raise ValueError("deferred kinetic union work coverage/provenance changed")
        self.bundle.assert_warm_layout()
        self.request.assert_current()
        canonical = tuple(binding.native_block_generation_digest for binding in self.bundle.native_blocks)
        active = tuple(block.native_block_generation_digest for block in self.active_blocks)
        if active != tuple(digest for digest in canonical if digest in set(active)):
            raise ValueError("deferred kinetic active native blocks are not canonical")
        if len(set(active)) != len(active):
            raise ValueError("deferred kinetic work contains duplicate native blocks")


@dataclass(frozen=True)
class PaperKineticCPUFakeNativeStepAccounting:
    requested_observation_count: int
    global_track_count: int
    global_loss_element_count: int
    spatial_block_count: int
    temporal_request_count: int
    sample_kernel_call_count: int
    streamed_sample_count: int
    sample_to_node_interactions: int
    native_active_block_count: int
    native_node_forward_invocation_count: int
    native_word_vjp_invocation_count: int
    ordered_run_node_interactions: int
    retained_topology_runtime_tensor_bytes: int
    peak_staged_target_tensor_bytes: int
    peak_sample_launch_tensor_bytes: int
    peak_spatial_node_state_tensor_bytes: int
    native_length_bar_tensor_bytes: int
    native_length_bar_callback_count: int
    caller_global_material_bar_tensor_bytes: int
    caller_global_material_bar_count: int
    optimizer_update_authorization_count: int
    maximum_observations_per_request: int
    maximum_samples_per_launch: int
    global_denominator_preserved: bool
    block_major_temporal_streaming: bool
    node_forward_depends_on_requested_frames: bool
    native_word_vjp_depends_on_requested_frames: bool
    global_common_temporal_refinement_used: bool
    persistent_frame_tensor_bytes: int
    persistent_target_tensor_bytes: int
    persistent_prediction_tensor_bytes: int
    geometry_length_bar_delivered: bool
    geometry_parameter_vjp_implemented: bool
    allocator_peak_measured: bool
    runtime_status: str
    sample_work_scaling: str
    word_work_scaling: str


@dataclass(frozen=True)
class PaperKineticCPUFakeNativeStepResult:
    step: PaperRaggedMaterialBarStepResult
    accounting: PaperKineticCPUFakeNativeStepAccounting


@dataclass
class _NativeDeferredState:
    runtime: KineticNativeEqualRankRuntimeBlock
    world: KineticNativeEqualRankWorld
    grad_node_chart_f32: torch.Tensor

    @property
    def base_tensor_bytes(self) -> int:
        return self.world.logical_world_tensor_bytes + _tensor_bytes(self.grad_node_chart_f32)


class _SpatialCoverageAccumulator:
    """O(native blocks) deferred coverage state; never retains request tensors."""

    def __init__(self, lane: PaperKineticCPUFakeNativeSpatialLane) -> None:
        self.lane = lane
        self.request_count = 0
        self.total_sample_count = 0
        self.counts: dict[str, list[int]] = {}
        self.next_local_sample_start = 0
        self._digest = hashlib.sha256()
        self._digest.update(DEFERRED_WORK_PROVENANCE.encode("utf-8"))

    def consume_request(
        self,
        request: PaperRaggedMaterialBarRequest,
        work: PaperKineticUnionLocalRequestWork,
        observed: dict[str, tuple[int, int]],
    ) -> None:
        work.assert_warm_layout()
        if work.request is not request or work.bundle is not self.lane.bundle:
            raise ValueError("deferred sample coverage received foreign request work")
        if request.local_sample_start != self.next_local_sample_start:
            raise ValueError("deferred sample coverage has a temporal gap or overlap")
        expected = {
            block.native_block_generation_digest: (
                block.sample_chunk_count,
                block.sample_count,
            )
            for block in work.active_blocks
        }
        if observed != expected:
            raise ValueError("streamed kinetic sample blocks did not exactly match request work")
        for digest, (chunk_count, sample_count) in observed.items():
            counts = self.counts.setdefault(digest, [0, 0])
            counts[0] += chunk_count
            counts[1] += sample_count
        self.request_count += 1
        self.total_sample_count += work.total_sample_count
        self.next_local_sample_start = request.local_sample_end
        self._digest.update(request.request_generation_id.encode("utf-8"))
        self._digest.update(work.generation_digest.encode("utf-8"))
        for digest, counts in observed.items():
            self._digest.update(digest.encode("utf-8"))
            self._digest.update(repr(counts).encode("utf-8"))

    def seal(
        self,
        carrier_request: PaperRaggedMaterialBarRequest,
        *,
        maximum_samples_per_launch: int,
    ) -> PaperKineticDeferredUnionLocalRequestWork:
        if self.next_local_sample_start != carrier_request.group.observation_count:
            raise ValueError("deferred kinetic work cannot seal before full temporal coverage")
        expected_total = carrier_request.block.track_count * carrier_request.group.observation_count
        if self.total_sample_count != expected_total:
            raise ValueError("deferred kinetic work sample count differs from spatial coverage")
        active = tuple(
            PaperKineticActiveNativeBlockWork(
                native_block_generation_digest=binding.native_block_generation_digest,
                sample_chunk_count=self.counts[binding.native_block_generation_digest][0],
                sample_count=self.counts[binding.native_block_generation_digest][1],
            )
            for binding in self.lane.bundle.native_blocks
            if binding.native_block_generation_digest in self.counts
        )
        coverage_digest = self._digest.hexdigest()
        generation = _digest_parts(
            DEFERRED_WORK_PROVENANCE,
            self.lane.generation_id,
            carrier_request.request_generation_id,
            coverage_digest,
            self.request_count,
            self.total_sample_count,
            tuple(
                (
                    block.native_block_generation_digest,
                    block.sample_chunk_count,
                    block.sample_count,
                )
                for block in active
            ),
        )
        result = PaperKineticDeferredUnionLocalRequestWork(
            bundle=self.lane.bundle,
            sampler=self.lane.bundle.sampler,
            request=carrier_request,
            maximum_samples_per_launch=maximum_samples_per_launch,
            active_blocks=active,
            total_sample_count=self.total_sample_count,
            generation_digest=generation,
            _bundle_identity=id(self.lane.bundle),
            _sampler_identity=id(self.lane.bundle.sampler),
            _request_identity=id(carrier_request),
            deferred_request_count=self.request_count,
            covered_local_sample_start=0,
            covered_local_sample_end=carrier_request.group.observation_count,
            carrier_request_sample_count=(
                carrier_request.block.track_count
                * (carrier_request.local_sample_end - carrier_request.local_sample_start)
            ),
            request_coverage_digest=coverage_digest,
            _sealed_generation_digest=generation,
            _deferred_seal=_DEFERRED_WORK_SEAL,
        )
        result.assert_warm_layout()
        return result


@torch.no_grad()
def run_paper_kinetic_cpu_fake_native_material_step(
    batch: PaperRaggedTrackBatch,
    *,
    programs: Sequence[PaperRaggedMaterialViewProgram],
    lanes: Sequence[PaperKineticCPUFakeNativeSpatialLane],
    global_site_rgba_f32: torch.Tensor,
    global_grad_site_rgba_f32: torch.Tensor,
    background_rgb_f32: torch.Tensor,
    maximum_observations_per_request: int,
    maximum_samples_per_launch: int,
    optimizer_update: Callable[[PaperRaggedMaterialBarStepResult], None],
) -> PaperKineticCPUFakeNativeStepResult:
    """Run one complete block-major paper step through the fake-native ABI."""

    if not isinstance(batch, PaperRaggedTrackBatch):
        raise TypeError("batch must be PaperRaggedTrackBatch")
    _require_positive_int(
        maximum_observations_per_request,
        name="maximum_observations_per_request",
    )
    _require_positive_int(maximum_samples_per_launch, name="maximum_samples_per_launch")
    if not callable(optimizer_update):
        raise TypeError("optimizer_update must be callable")
    _require_cpu_f32(global_site_rgba_f32, name="global_site_rgba_f32", columns=4)
    _require_cpu_f32(
        global_grad_site_rgba_f32,
        name="global_grad_site_rgba_f32",
        columns=4,
    )
    if tuple(global_site_rgba_f32.shape) != tuple(global_grad_site_rgba_f32.shape):
        raise ValueError("global material and material bar must have identical shapes")
    if _same_storage(global_site_rgba_f32, global_grad_site_rgba_f32):
        raise ValueError("global material and material bar must not alias")
    _require_cpu_f32(background_rgb_f32, name="background_rgb_f32", shape=(3,))

    normalized_programs = tuple(programs)
    normalized_lanes = tuple(lanes)
    lane_by_key: dict[tuple[int, str], PaperKineticCPUFakeNativeSpatialLane] = {}
    runtime_ids: set[int] = set()
    for lane in normalized_lanes:
        if not isinstance(lane, PaperKineticCPUFakeNativeSpatialLane):
            raise TypeError("lanes must contain PaperKineticCPUFakeNativeSpatialLane values")
        lane.assert_current()
        key = (lane.view_index, lane.block_id)
        if key in lane_by_key:
            raise ValueError("deferred kinetic runner received duplicate spatial lanes")
        if runtime_ids.intersection(id(runtime) for runtime in lane.runtimes):
            raise ValueError("one native runtime cannot belong to multiple spatial lanes")
        runtime_ids.update(id(runtime) for runtime in lane.runtimes)
        lane_by_key[key] = lane
    expected_keys = {
        (program.view_index, block.block_id) for program in normalized_programs for block in program.blocks
    }
    if set(lane_by_key) != expected_keys:
        raise ValueError("deferred kinetic runner requires one lane for every coordinator block")
    if any(lane.bundle.global_site_count != int(global_site_rgba_f32.shape[0]) for lane in normalized_lanes):
        raise ValueError("kinetic lanes disagree with the caller global material table")

    ledger = begin_paper_ragged_material_bar_step(
        batch,
        programs=normalized_programs,
        global_grad_site_rgba_f32=global_grad_site_rgba_f32,
    )
    temporal_request_count = 0
    sample_kernel_call_count = 0
    streamed_sample_count = 0
    sample_to_node_interactions = 0
    native_active_block_count = 0
    native_forward_count = 0
    native_vjp_count = 0
    ordered_run_node_interactions = 0
    peak_sample_launch_bytes = 0
    peak_spatial_node_state_bytes = 0

    groups_by_view = {group.view_index: group for group in batch.groups}
    programs_by_view = {program.view_index: program for program in normalized_programs}
    for view_index in sorted(programs_by_view):
        group = groups_by_view[view_index]
        for spatial_block in programs_by_view[view_index].blocks:
            lane = lane_by_key[(view_index, spatial_block.block_id)]
            coverage = _SpatialCoverageAccumulator(lane)
            states: dict[str, _NativeDeferredState] = {}
            union_bar = torch.zeros(
                (lane.bundle.union_site_count, 4),
                dtype=torch.float32,
                device="cpu",
            )
            request_loss = torch.zeros((1,), dtype=torch.float32, device="cpu")
            assembly_loss = torch.zeros((1,), dtype=torch.float32, device="cpu")
            zero_loss = torch.zeros((1,), dtype=torch.float32, device="cpu")

            for local_start in range(
                0,
                group.observation_count,
                maximum_observations_per_request,
            ):
                local_end = min(
                    group.observation_count,
                    local_start + maximum_observations_per_request,
                )
                request = stage_next_paper_ragged_material_bar_request(
                    ledger,
                    view_index=view_index,
                    block_id=spatial_block.block_id,
                    local_sample_start=local_start,
                    local_sample_end=local_end,
                )
                request_work = prepare_paper_kinetic_union_local_request_work(
                    lane.bundle,
                    request,
                    maximum_samples_per_launch=maximum_samples_per_launch,
                )
                request_loss.zero_()
                observed_mutable: dict[str, list[int]] = {}
                for sample_block in iter_paper_kinetic_row_ragged_request_blocks(
                    lane.bundle.sampler,
                    request,
                    maximum_samples_per_launch=maximum_samples_per_launch,
                ):
                    digest = sample_block.native_block_generation_digest
                    state = states.get(digest)
                    if state is None:
                        runtime = lane.runtime_for_digest(digest)
                        compact_material = global_site_rgba_f32.index_select(
                            0,
                            runtime.source_site_ids_i64,
                        ).contiguous()
                        world = refresh_kinetic_native_equal_rank_world(
                            runtime,
                            compact_material,
                        )
                        state = _NativeDeferredState(
                            runtime=runtime,
                            world=world,
                            grad_node_chart_f32=torch.zeros_like(world.node_chart_f32),
                        )
                        states[digest] = state
                        native_forward_count += 1
                        ordered_run_node_interactions += runtime.node_count * runtime.word_count
                    interactions = _accumulate_cpu_ragged_sample_block(
                        state,
                        sample_block,
                        background_rgb_f32=background_rgb_f32,
                        loss_f32=request_loss,
                    )
                    counts = observed_mutable.setdefault(digest, [0, 0])
                    counts[0] += 1
                    counts[1] += sample_block.sample_count
                    sample_kernel_call_count += 1
                    streamed_sample_count += sample_block.sample_count
                    sample_to_node_interactions += interactions
                    peak_sample_launch_bytes = max(
                        peak_sample_launch_bytes,
                        sample_block.retained_tensor_bytes,
                    )
                    current_state_bytes = _tensor_bytes(union_bar) + sum(
                        value.base_tensor_bytes for value in states.values()
                    )
                    peak_spatial_node_state_bytes = max(
                        peak_spatial_node_state_bytes,
                        current_state_bytes,
                    )
                    del sample_block

                observed = {digest: (counts[0], counts[1]) for digest, counts in observed_mutable.items()}
                coverage.consume_request(request, request_work, observed)
                temporal_request_count += 1
                is_final_request = local_end == group.observation_count
                if not is_final_request:
                    zero_result = seal_paper_ragged_compact_material_bar_result(
                        request,
                        loss_f32=request_loss,
                        grad_compact_site_rgba_f32=union_bar,
                    )
                    consume_paper_ragged_compact_material_bar_result(
                        ledger,
                        request,
                        zero_result,
                    )
                    del zero_result, request_work, request
                    continue

                deferred_work = coverage.seal(
                    request,
                    maximum_samples_per_launch=maximum_samples_per_launch,
                )
                assembly = begin_paper_kinetic_union_local_bar_assembly(
                    deferred_work,
                    grad_union_site_rgba_f32=union_bar,
                    loss_f32=assembly_loss,
                )
                native_active_block_count += deferred_work.active_native_block_count
                for native_index, active in enumerate(deferred_work.active_blocks):
                    state = states.get(active.native_block_generation_digest)
                    if state is None:
                        raise ValueError("deferred native block has samples but no node state")
                    compact_bar = torch.empty(
                        (state.runtime.compact_site_count, 4),
                        dtype=torch.float32,
                        device="cpu",
                    )
                    native_vjp = execute_kinetic_native_equal_rank_material_node_vjp(
                        state.world,
                        state.grad_node_chart_f32,
                        compact_grad_site_rgba_f32=compact_bar,
                    )
                    native_vjp_count += 1
                    peak_spatial_node_state_bytes = max(
                        peak_spatial_node_state_bytes,
                        _tensor_bytes(union_bar, compact_bar)
                        + sum(value.base_tensor_bytes for value in states.values()),
                    )
                    contribution = seal_paper_kinetic_native_block_vjp_contribution(
                        assembly,
                        native_vjp_result=native_vjp,
                        loss_f32=request_loss if native_index == 0 else zero_loss,
                        reduced_sample_chunk_count=active.sample_chunk_count,
                        reduced_sample_count=active.sample_count,
                    )
                    consume_paper_kinetic_union_local_native_contribution(
                        assembly,
                        contribution,
                    )
                    del states[active.native_block_generation_digest]
                    del contribution, native_vjp, compact_bar, state

                if states:
                    raise ValueError("deferred native state remained after exact block coverage")
                compact_result = finalize_paper_kinetic_union_local_bar_assembly(assembly)
                consume_paper_ragged_compact_material_bar_result(
                    ledger,
                    request,
                    compact_result,
                )
                del compact_result, assembly, deferred_work, request_work, request

            if states:
                raise ValueError("spatial block ended with unreversed native node state")

    authorization = finalize_paper_ragged_material_bar_step(ledger)
    optimizer_callback_count = 0

    def _authorized_update(result: PaperRaggedMaterialBarStepResult) -> None:
        nonlocal optimizer_callback_count
        optimizer_callback_count += 1
        optimizer_update(result)

    authorization.consume(_authorized_update)
    if optimizer_callback_count != 1:
        raise ArithmeticError("kinetic paper step did not invoke exactly one optimizer callback")
    if not (native_forward_count == native_vjp_count == native_active_block_count):
        raise ArithmeticError("kinetic paper step violated once-per-active-native-block execution")
    if streamed_sample_count != batch.pixel_count * batch.observation_count:
        raise ArithmeticError("kinetic paper step sample coverage differs from P*B")

    retained_runtime_bytes = sum(
        runtime.memory_accounting(batch.observation_count).unique_retained_tensor_bytes
        for lane in normalized_lanes
        for runtime in lane.runtimes
    )
    step = authorization.result
    accounting = PaperKineticCPUFakeNativeStepAccounting(
        requested_observation_count=batch.observation_count,
        global_track_count=batch.pixel_count,
        global_loss_element_count=batch.global_rgb_element_count,
        spatial_block_count=len(normalized_lanes),
        temporal_request_count=temporal_request_count,
        sample_kernel_call_count=sample_kernel_call_count,
        streamed_sample_count=streamed_sample_count,
        sample_to_node_interactions=sample_to_node_interactions,
        native_active_block_count=native_active_block_count,
        native_node_forward_invocation_count=native_forward_count,
        native_word_vjp_invocation_count=native_vjp_count,
        ordered_run_node_interactions=ordered_run_node_interactions,
        retained_topology_runtime_tensor_bytes=retained_runtime_bytes,
        peak_staged_target_tensor_bytes=int(step.accounting["peak_staged_target_bytes"]),
        peak_sample_launch_tensor_bytes=peak_sample_launch_bytes,
        peak_spatial_node_state_tensor_bytes=peak_spatial_node_state_bytes,
        native_length_bar_tensor_bytes=0,
        native_length_bar_callback_count=0,
        caller_global_material_bar_tensor_bytes=_tensor_bytes(global_grad_site_rgba_f32),
        caller_global_material_bar_count=1,
        optimizer_update_authorization_count=1,
        maximum_observations_per_request=maximum_observations_per_request,
        maximum_samples_per_launch=maximum_samples_per_launch,
        global_denominator_preserved=True,
        block_major_temporal_streaming=True,
        node_forward_depends_on_requested_frames=False,
        native_word_vjp_depends_on_requested_frames=False,
        global_common_temporal_refinement_used=False,
        persistent_frame_tensor_bytes=0,
        persistent_target_tensor_bytes=0,
        persistent_prediction_tensor_bytes=0,
        geometry_length_bar_delivered=False,
        geometry_parameter_vjp_implemented=False,
        allocator_peak_measured=False,
        runtime_status=CPU_FAKE_NATIVE_STATUS,
        sample_work_scaling="O(sum_streamed_samples * selected_J)",
        word_work_scaling="O(sum_active_native_blocks J_b * W_b), independent of requested F",
    )
    return PaperKineticCPUFakeNativeStepResult(step=step, accounting=accounting)


@torch.no_grad()
def _accumulate_cpu_ragged_sample_block(
    state: _NativeDeferredState,
    block: PaperKineticRowRaggedSampleBlock,
    *,
    background_rgb_f32: torch.Tensor,
    loss_f32: torch.Tensor,
    cone_tolerance: float = 1.0e-5,
) -> int:
    """CPU double of the loss-only row-ragged Metal sample kernel."""

    block.assert_warm_layout()
    if block.native_block_generation_digest != state.runtime.payload.block.generation_digest:
        raise ValueError("ragged sample block belongs to a different native world")
    if (
        block.row_count != state.world.row_count
        or block.node_count != state.world.node_count
        or block.loss_scale != 1.0 / float(block.global_loss_element_count)
    ):
        raise ValueError("ragged sample block changed native row/rank or global denominator")
    _require_cpu_f32(loss_f32, name="request_loss_f32", shape=(1,))
    row_ids = block.sample_row_i32.to(dtype=torch.int64)
    selected_nodes = state.world.node_chart_f32.index_select(0, row_ids)
    chart = torch.sum(selected_nodes * block.sample_to_node_f32[:, :, None], dim=1)
    kappa = chart[:, 0]
    velocity = chart[:, 1:]
    cone_violation = torch.maximum(
        torch.maximum(-kappa, -torch.amin(velocity, dim=1)),
        torch.amax(velocity, dim=1) - kappa,
    )
    if not bool(torch.isfinite(chart).all().item()) or bool(torch.any(cone_violation > cone_tolerance).item()):
        raise ValueError("interpolated ragged affine-Lie chart left the physical cone")
    phi, phi_prime = _lie_phi_and_derivative_f32(kappa)
    beta = torch.exp(-kappa)
    prediction = phi[:, None] * velocity + beta[:, None] * background_rgb_f32
    residual = prediction - block.target_rgb_f32
    if not bool(torch.isfinite(residual).all().item()):
        raise ValueError("ragged kinetic prediction/target residual is nonfinite")
    loss_f32.add_(residual.square().sum() * block.loss_scale)
    grad_prediction = (2.0 * block.loss_scale) * residual
    grad_beta = torch.sum(grad_prediction * background_rgb_f32, dim=1)
    grad_chart = torch.cat(
        (
            (-beta * grad_beta + phi_prime * torch.sum(velocity * grad_prediction, dim=1))[:, None],
            phi[:, None] * grad_prediction,
        ),
        dim=1,
    )
    node_contribution = block.sample_to_node_f32[:, :, None] * grad_chart[:, None, :]
    state.grad_node_chart_f32.index_add_(0, row_ids, node_contribution)
    return block.sample_count * block.node_count


def _lie_phi_and_derivative_f32(kappa: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    small = torch.abs(kappa) < 1.0e-4
    k2 = kappa.square()
    k3 = k2 * kappa
    k4 = k3 * kappa
    k5 = k4 * kappa
    k6 = k5 * kappa
    phi_series = 1.0 - 0.5 * kappa + k2 / 6.0 - k3 / 24.0 + k4 / 120.0 - k5 / 720.0 + k6 / 5040.0
    derivative_series = -0.5 + kappa / 3.0 - k2 / 8.0 + k3 / 30.0 - k4 / 144.0 + k5 / 840.0
    safe_kappa = torch.where(small, torch.ones_like(kappa), kappa)
    numerator = -torch.expm1(-kappa)
    phi = torch.where(small, phi_series, numerator / safe_kappa)
    derivative = torch.where(
        small,
        derivative_series,
        (kappa * torch.exp(-kappa) - numerator) / safe_kappa.square(),
    )
    return phi, derivative


def _require_cpu_f32(
    tensor: torch.Tensor,
    *,
    name: str,
    columns: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if (
        tensor.device.type != "cpu"
        or tensor.dtype != torch.float32
        or not tensor.is_contiguous()
        or tensor.requires_grad
    ):
        raise ValueError(f"{name} must be detached contiguous CPU float32")
    if shape is not None and tuple(tensor.shape) != shape:
        raise ValueError(f"{name} has the wrong shape")
    if columns is not None and (tensor.ndim != 2 or int(tensor.shape[1]) != columns):
        raise ValueError(f"{name} must have shape [N,{columns}]")


def _same_storage(first: torch.Tensor, second: torch.Tensor) -> bool:
    return first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()


def _tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _digest_parts(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _require_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


__all__ = [
    "CPU_FAKE_NATIVE_STATUS",
    "PaperKineticCPUFakeNativeSpatialLane",
    "PaperKineticCPUFakeNativeStepAccounting",
    "PaperKineticCPUFakeNativeStepResult",
    "PaperKineticDeferredUnionLocalRequestWork",
    "prepare_paper_kinetic_cpu_fake_native_spatial_lane",
    "run_paper_kinetic_cpu_fake_native_material_step",
]
