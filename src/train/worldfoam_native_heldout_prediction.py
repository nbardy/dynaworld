"""Bounded real-native heldout RGB prediction for kinetic WorldFoam.

This module is the evaluation-side counterpart of the lazy shared-adjoint
trainer.  It accepts an already sealed heldout provider and a caller-owned
global material tensor, compiles one bounded spatial bundle at a time, and
streams the provider's existing ragged sample blocks through the post-103
prediction-returning Metal operator.

There is deliberately no CPU renderer, fake-native implementation, or
procedural fallback here.  MPS admission requires all of the following:

* the canonical WorldFoam native module and its source-fresh memory-light ABI;
* the exact prediction operator schema registered by the compiled extension;
* the project's selected-kernel launch-domain attestation; and
* a sealed device-wide completion receipt before every CPU transfer or
  predecessor release.

The returned CPU tensor is in the caller's exact observation order.  Device
prediction, target, sample, and frame tensors are not retained after return.
If a completion fence fails, one bounded stage is quarantined and every later
call fails closed until process restart.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch

from paper_kinetic_lazy_program_bundles import (
    PaperKineticLazyProgramBundleProvider,
    PaperKineticObservation,
    prepare_paper_kinetic_lazy_bundle_construction_lifetime_slot,
)
from paper_kinetic_runtime_paths import ensure_worldfoam_lane2_research_path
from paper_kinetic_sparse_sample_blocks import (
    iter_paper_kinetic_sparse_sample_blocks,
    prepare_paper_kinetic_sparse_sample_plan,
)

ensure_worldfoam_lane2_research_path()

from kinetic_native_equal_rank_runtime_adapter import (  # noqa: E402
    refresh_kinetic_native_equal_rank_world_into,
)
from kinetic_native_lazy_bundle_lane import (  # noqa: E402
    materialize_paper_kinetic_native_lazy_bundle_lane,
    prepare_paper_kinetic_native_lazy_bundle_lane_construction_lifetime,
)
from kinetic_sealed_completion_fence import (  # noqa: E402
    PaperKineticCompletionFenceReceipt,
    PaperKineticCompletionLaunchEpoch,
    PaperKineticCompletionSubjectBinding,
    PaperKineticCompletionUnknownError,
    PaperKineticSealedCompletionFence,
    prepare_paper_kinetic_completion_subject_binding,
    prepare_paper_kinetic_sealed_completion_fence,
)


PREDICTION_PROVENANCE = "worldfoam-native-heldout-prediction-v1"
PREDICTION_RESULT_PROVENANCE = "worldfoam-native-heldout-prediction-result-v1"
PREDICTION_RECEIPT_PROVENANCE = "worldfoam-native-heldout-prediction-receipt-v1"
PREDICTION_SUBJECT_KIND = "worldfoam-native-heldout-prediction-stage-v1"

_NATIVE_NAMESPACE = "world_foam_lane2_fused_slab_v0"
_PREDICTION_OP_NAME = "kinetic_ragged_p0_lie_sample_accumulate_launch_only"
_PREDICTION_PREPARE_NAME = "prepare_kinetic_ragged_p0_lie_sample_block"
_PREDICTION_SCHEMA = (
    "world_foam_lane2_fused_slab_v0::"
    "kinetic_ragged_p0_lie_sample_accumulate_launch_only("
    "Tensor node_chart_f32, Tensor sample_row_i32, "
    "Tensor sample_to_node_f32, Tensor target_rgb_f32, "
    "Tensor background_rgb_f32, Tensor(a!) loss_f32, "
    "Tensor(b!) grad_node_chart_f32, Tensor(c!) cone_diagnostic_i32, "
    "Tensor config_i32, Tensor config_f32, int row_count, int node_count, "
    "int sample_count) -> Tensor"
)
PREDICTION_ABI_SCHEMA_SHA256 = hashlib.sha256(
    _PREDICTION_SCHEMA.encode("utf-8")
).hexdigest()

_UNKNOWN_COMPLETION_QUARANTINE: list[_PredictionFenceStage] = []


@dataclass(frozen=True)
class WorldFoamNativeHeldoutPredictionReceipt:
    """Primitive-only receipt for one exact bounded prediction request."""

    provider_generation_digest: str
    material_generation_digest: str
    observation_order_digest: str
    observation_count: int
    unique_observation_id_count: int
    bundle_count: int
    native_runtime_count: int
    active_native_runtime_count: int
    node_forward_launch_count: int
    native_prediction_launch_count: int
    native_sample_count: int
    rasterized_sample_count: int
    selected_pixel_read_call_count: int
    selected_pixel_read_observation_count: int
    direct_selected_pixel_observation_count: int
    bounded_region_selected_pixel_observation_count: int
    full_frame_fallback_observation_count: int
    full_frame_target_materialization_count: int
    bounded_region_target_materialization_count: int
    mapped_selected_pixel_read_call_count: int
    mapping_closed_before_return_count: int
    bundle_construction_fence_count: int
    empty_bundle_iteration_fence_count: int
    lane_construction_fence_count: int
    sample_completion_fence_count: int
    sealed_device_completion_fence_count: int
    device_to_cpu_prediction_transfer_count: int
    cone_diagnostic: tuple[int, int, int]
    peak_lane_resident_logical_tensor_bytes: int
    peak_cached_node_world_logical_tensor_bytes: int
    peak_sample_launch_logical_tensor_bytes: int
    peak_device_prediction_tensor_bytes: int
    peak_source_visible_target_read_logical_tensor_bytes_upper_bound: int
    peak_transient_mapped_address_space_bytes: int
    returned_cpu_prediction_tensor_bytes: int
    maximum_returned_cpu_prediction_tensor_bytes: int
    backend_provenance: str
    completion_scope: str
    generation_digest: str
    schema_version: int = 1
    provenance: str = PREDICTION_RECEIPT_PROVENANCE
    real_native_execution: bool = True
    fake_native_execution: bool = False
    cpu_prediction_fallback_used: bool = False
    compiled_prediction_schema_verified: bool = True
    exact_observation_id_coverage: bool = True
    prediction_in_input_observation_order: bool = True
    selected_pixel_read_acceptance_capable: bool = True
    all_selected_pixel_mappings_closed_before_return: bool = True
    global_loss_denominator_preserved: bool = True
    persistent_frame_tensor_bytes: int = 0
    persistent_device_target_tensor_bytes: int = 0
    persistent_device_sample_tensor_bytes: int = 0
    persistent_device_prediction_tensor_bytes: int = 0
    retained_native_lane_count_after_return: int = 0
    measurement_is_simulated: bool = False
    allocator_peak_measured: bool = False

    def assert_current(self) -> None:
        integer_fields = (
            self.observation_count,
            self.unique_observation_id_count,
            self.bundle_count,
            self.native_runtime_count,
            self.active_native_runtime_count,
            self.node_forward_launch_count,
            self.native_prediction_launch_count,
            self.native_sample_count,
            self.rasterized_sample_count,
            self.selected_pixel_read_call_count,
            self.selected_pixel_read_observation_count,
            self.direct_selected_pixel_observation_count,
            self.bounded_region_selected_pixel_observation_count,
            self.full_frame_fallback_observation_count,
            self.full_frame_target_materialization_count,
            self.bounded_region_target_materialization_count,
            self.mapped_selected_pixel_read_call_count,
            self.mapping_closed_before_return_count,
            self.bundle_construction_fence_count,
            self.empty_bundle_iteration_fence_count,
            self.lane_construction_fence_count,
            self.sample_completion_fence_count,
            self.sealed_device_completion_fence_count,
            self.device_to_cpu_prediction_transfer_count,
            self.peak_lane_resident_logical_tensor_bytes,
            self.peak_cached_node_world_logical_tensor_bytes,
            self.peak_sample_launch_logical_tensor_bytes,
            self.peak_device_prediction_tensor_bytes,
            self.peak_source_visible_target_read_logical_tensor_bytes_upper_bound,
            self.peak_transient_mapped_address_space_bytes,
            self.returned_cpu_prediction_tensor_bytes,
            self.maximum_returned_cpu_prediction_tensor_bytes,
        )
        if (
            self.schema_version != 1
            or self.provenance != PREDICTION_RECEIPT_PROVENANCE
            or not _is_sha256(self.provider_generation_digest)
            or not _is_sha256(self.material_generation_digest)
            or not _is_sha256(self.observation_order_digest)
            or not _is_sha256(self.generation_digest)
            or any(type(value) is not int or value < 0 for value in integer_fields)
            or self.observation_count < 1
            or self.observation_count != self.unique_observation_id_count
            or self.observation_count != self.native_sample_count
            or self.observation_count != self.rasterized_sample_count
            or self.observation_count
            != self.selected_pixel_read_observation_count
            or self.observation_count
            != self.direct_selected_pixel_observation_count
            + self.bounded_region_selected_pixel_observation_count
            or self.native_prediction_launch_count
            != self.sample_completion_fence_count
            or self.device_to_cpu_prediction_transfer_count
            != self.native_prediction_launch_count
            or self.node_forward_launch_count != self.active_native_runtime_count
            or self.active_native_runtime_count > self.native_runtime_count
            or self.bundle_count != self.lane_construction_fence_count
            or self.bundle_construction_fence_count
            != self.bundle_count + self.empty_bundle_iteration_fence_count
            or self.empty_bundle_iteration_fence_count != 1
            or self.sealed_device_completion_fence_count
            != self.bundle_construction_fence_count
            + self.lane_construction_fence_count
            + self.sample_completion_fence_count
            or len(self.cone_diagnostic) != 3
            or any(type(value) is not int or value < 0 for value in self.cone_diagnostic)
            or self.returned_cpu_prediction_tensor_bytes
            != self.observation_count * 3 * 4
            or self.returned_cpu_prediction_tensor_bytes
            > self.maximum_returned_cpu_prediction_tensor_bytes
            or self.peak_device_prediction_tensor_bytes < 3 * 4
            or self.full_frame_fallback_observation_count != 0
            or self.full_frame_target_materialization_count != 0
            or self.mapping_closed_before_return_count
            != self.mapped_selected_pixel_read_call_count
            or not self.backend_provenance.strip()
            or self.completion_scope != "mps/torch-device-wide"
            or not self.real_native_execution
            or self.fake_native_execution
            or self.cpu_prediction_fallback_used
            or not self.compiled_prediction_schema_verified
            or not self.exact_observation_id_coverage
            or not self.prediction_in_input_observation_order
            or not self.selected_pixel_read_acceptance_capable
            or not self.all_selected_pixel_mappings_closed_before_return
            or not self.global_loss_denominator_preserved
            or self.persistent_frame_tensor_bytes != 0
            or self.persistent_device_target_tensor_bytes != 0
            or self.persistent_device_sample_tensor_bytes != 0
            or self.persistent_device_prediction_tensor_bytes != 0
            or self.retained_native_lane_count_after_return != 0
            or self.measurement_is_simulated
            or self.allocator_peak_measured
            or self.generation_digest != _prediction_receipt_digest(self)
        ):
            raise ValueError("native heldout prediction receipt changed")

    def accounting(self) -> Mapping[str, int | bool | str | tuple[int, int, int]]:
        self.assert_current()
        return MappingProxyType(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "generation_digest"
            }
            | {"generation_digest": self.generation_digest}
        )


@dataclass(frozen=True)
class WorldFoamNativeHeldoutPrediction:
    """Exact CPU RGB in input order plus a tensor-free native receipt."""

    rgb_f32_cpu: torch.Tensor = field(repr=False)
    receipt: WorldFoamNativeHeldoutPredictionReceipt
    generation_digest: str
    _rgb_identity: int = field(repr=False)
    _rgb_signature: tuple[object, ...] = field(repr=False)
    provenance: str = PREDICTION_RESULT_PROVENANCE

    def assert_current(self) -> None:
        self.receipt.assert_current()
        if (
            self.provenance != PREDICTION_RESULT_PROVENANCE
            or not isinstance(self.rgb_f32_cpu, torch.Tensor)
            or self.rgb_f32_cpu.device.type != "cpu"
            or self.rgb_f32_cpu.dtype != torch.float32
            or tuple(self.rgb_f32_cpu.shape)
            != (self.receipt.observation_count, 3)
            or not self.rgb_f32_cpu.is_contiguous()
            or id(self.rgb_f32_cpu) != self._rgb_identity
            or _tensor_signature(self.rgb_f32_cpu) != self._rgb_signature
            or not _is_sha256(self.generation_digest)
            or self.generation_digest != _digest_parts(
                PREDICTION_RESULT_PROVENANCE,
                self.receipt.generation_digest,
                self._rgb_signature,
            )
        ):
            raise ValueError("native heldout prediction result changed")


@dataclass
class _PredictionFenceStage:
    """Stable prelaunch subject retaining every root until receipt use."""

    capability: PaperKineticSealedCompletionFence = field(repr=False)
    stage: str
    ordinal: int
    owner_generation_digest: str
    roots: list[Any] = field(default_factory=list, repr=False)
    generation_digest: str = ""
    binding: PaperKineticCompletionSubjectBinding | None = field(
        default=None,
        repr=False,
    )
    epoch: PaperKineticCompletionLaunchEpoch | None = field(
        default=None,
        repr=False,
    )
    receipt: PaperKineticCompletionFenceReceipt | None = field(
        default=None,
        repr=False,
    )
    phase: str = "installed"

    def begin(self) -> None:
        if self.phase != "installed" or self.binding is not None or self.epoch is not None:
            raise RuntimeError("prediction completion stage was already opened")
        self.generation_digest = _digest_parts(
            PREDICTION_SUBJECT_KIND,
            self.owner_generation_digest,
            self.stage,
            self.ordinal,
            id(self),
        )
        self.binding = prepare_paper_kinetic_completion_subject_binding(
            self.capability,
            self,
            kind=PREDICTION_SUBJECT_KIND,
            subject_generation_digest=self.generation_digest,
        )
        self.epoch = self.capability.register_launch(
            stage=self.stage,
            launch_generation_digest=_digest_parts(
                PREDICTION_PROVENANCE,
                self.generation_digest,
                "registered-before-transfer-or-dispatch",
            ),
            subject_binding=self.binding,
        )
        self.phase = "registered"

    def fence(self) -> PaperKineticCompletionFenceReceipt:
        if self.phase != "registered" or self.epoch is None:
            raise RuntimeError("prediction completion stage is not registered")
        try:
            self.receipt = self.capability.fence(self.epoch)
        except PaperKineticCompletionUnknownError:
            self.phase = "completion_unknown"
            if not any(value is self for value in _UNKNOWN_COMPLETION_QUARANTINE):
                _UNKNOWN_COMPLETION_QUARANTINE.append(self)
            raise
        self.phase = "fenced"
        return self.receipt

    def consume(self, *, consumer: str) -> None:
        if (
            self.phase != "fenced"
            or self.binding is None
            or self.receipt is None
        ):
            raise RuntimeError("prediction completion stage has no fenced receipt")
        try:
            self.receipt.consume_for_subject(
                self.capability,
                self.binding,
                subject=self,
                consumer=consumer,
            )
        except BaseException:
            self.phase = "receipt_consumption_failed"
            if not any(value is self for value in _UNKNOWN_COMPLETION_QUARANTINE):
                _UNKNOWN_COMPLETION_QUARANTINE.append(self)
            raise
        self.phase = "consumed"

    def settle_after_error(self) -> None:
        """Drain a registered stage once; quarantine if completion is unknown."""

        if self.phase == "registered":
            self.fence()
        if self.phase == "fenced":
            self.consume(consumer="native-heldout-prediction-error-settlement")
        if self.phase == "consumed":
            self.roots.clear()


@torch.no_grad()
def predict_worldfoam_native_heldout_observations(
    provider: PaperKineticLazyProgramBundleProvider,
    observations: Sequence[PaperKineticObservation],
    *,
    global_site_rgba_f32: torch.Tensor,
    material_generation_digest: str,
    native_ops: Any,
    background_rgb_f32: torch.Tensor,
    maximum_samples_per_launch: int,
    maximum_source_decode_tensor_bytes: int,
    maximum_lane_resident_logical_tensor_bytes: int,
    maximum_returned_cpu_prediction_tensor_bytes: int,
    cone_tolerance: float = 1.0e-5,
) -> WorldFoamNativeHeldoutPrediction:
    """Render exact observations with bounded real-native WorldFoam.

    ``observations`` may be in any order, but every ``observation_id`` must be
    unique.  The provider canonicalizes work internally; predictions are
    scattered back to this exact input order before return.
    """

    if _UNKNOWN_COMPLETION_QUARANTINE:
        raise RuntimeError(
            "a prior native prediction completion is unknown; process restart "
            "is required before another WorldFoam launch"
        )
    _validate_prediction_request(
        provider,
        observations,
        global_site_rgba_f32=global_site_rgba_f32,
        material_generation_digest=material_generation_digest,
        native_ops=native_ops,
        background_rgb_f32=background_rgb_f32,
        maximum_samples_per_launch=maximum_samples_per_launch,
        maximum_source_decode_tensor_bytes=maximum_source_decode_tensor_bytes,
        maximum_lane_resident_logical_tensor_bytes=(
            maximum_lane_resident_logical_tensor_bytes
        ),
        maximum_returned_cpu_prediction_tensor_bytes=(
            maximum_returned_cpu_prediction_tensor_bytes
        ),
        cone_tolerance=cone_tolerance,
    )
    selected = tuple(observations)
    observation_order_digest = _observation_order_digest(selected)
    prepare_prediction = getattr(native_ops, _PREDICTION_PREPARE_NAME)
    launch_prediction = getattr(native_ops, _PREDICTION_OP_NAME)
    input_position_by_observation_id = {
        observation.observation_id: position
        for position, observation in enumerate(selected)
    }
    output = torch.empty((len(selected), 3), dtype=torch.float32, device="cpu")
    covered = torch.zeros((len(selected),), dtype=torch.bool, device="cpu")
    owner_generation_digest = _digest_parts(
        PREDICTION_PROVENANCE,
        provider.generation_digest,
        material_generation_digest,
        observation_order_digest,
        _tensor_signature(global_site_rgba_f32),
        _tensor_signature(background_rgb_f32),
        maximum_samples_per_launch,
        maximum_source_decode_tensor_bytes,
        maximum_lane_resident_logical_tensor_bytes,
        maximum_returned_cpu_prediction_tensor_bytes,
        float(cone_tolerance),
        id(getattr(prepare_prediction, "__func__", prepare_prediction)),
        id(getattr(launch_prediction, "__func__", launch_prediction)),
    )
    capability = prepare_paper_kinetic_sealed_completion_fence(
        native_ops,
        device=global_site_rgba_f32.device,
        owner_generation_digest=owner_generation_digest,
        dispatch_anchor=global_site_rgba_f32,
    )
    construction_slot = prepare_paper_kinetic_lazy_bundle_construction_lifetime_slot()
    bundle_iterator = provider.iter_spatial_bundles(
        selected,
        device=global_site_rgba_f32.device,
        construction_lifetime_slot=construction_slot,
    )

    counters = {
        "bundle_count": 0,
        "native_runtime_count": 0,
        "active_native_runtime_count": 0,
        "node_forward_launch_count": 0,
        "native_prediction_launch_count": 0,
        "native_sample_count": 0,
        "selected_pixel_read_call_count": 0,
        "selected_pixel_read_observation_count": 0,
        "direct_selected_pixel_observation_count": 0,
        "bounded_region_selected_pixel_observation_count": 0,
        "full_frame_fallback_observation_count": 0,
        "full_frame_target_materialization_count": 0,
        "bounded_region_target_materialization_count": 0,
        "mapped_selected_pixel_read_call_count": 0,
        "mapping_closed_before_return_count": 0,
        "bundle_construction_fence_count": 0,
        "empty_bundle_iteration_fence_count": 0,
        "lane_construction_fence_count": 0,
        "sample_completion_fence_count": 0,
        "device_to_cpu_prediction_transfer_count": 0,
    }
    peaks = {
        "lane": 0,
        "world": 0,
        "sample": 0,
        "prediction": 0,
        "target_read": 0,
        "mapped_address_space": 0,
    }
    cone_diagnostic = [0, 0, 0]
    stage_ordinal = 0

    try:
        while True:
            construction_stage = _PredictionFenceStage(
                capability=capability,
                stage="heldout-bundle-construction",
                ordinal=stage_ordinal,
                owner_generation_digest=owner_generation_digest,
                roots=[
                    provider,
                    bundle_iterator,
                    construction_slot,
                    global_site_rgba_f32,
                    background_rgb_f32,
                ],
            )
            stage_ordinal += 1
            construction_stage.begin()
            try:
                bundle = next(bundle_iterator)
            except StopIteration:
                construction_stage.fence()
                construction_stage.consume(
                    consumer="native-heldout-empty-bundle-iteration"
                )
                construction_stage.roots.clear()
                counters["bundle_construction_fence_count"] += 1
                counters["empty_bundle_iteration_fence_count"] += 1
                break
            except BaseException as error:
                _settle_stage_or_raise(construction_stage, error)
                raise
            try:
                construction_stage.roots.append(bundle)
                spatial_lifetime = construction_slot.active_lifetime
                if spatial_lifetime is None:
                    raise RuntimeError(
                        "accelerator bundle construction did not publish its lifetime"
                    )
                construction_stage.roots.append(spatial_lifetime)
                construction_stage.fence()
                construction_slot.assert_active_releasable_after_consumed_receipt()
                spatial_lifetime.assert_accelerator_transfer_releasable_after_completion_fence(
                    bundle.spatial_bundle
                )
                construction_stage.consume(
                    consumer="native-heldout-bundle-transfer-release"
                )
                spatial_lifetime._commit_transfer_predecessors_after_consumed_receipt()
                construction_slot.complete(spatial_lifetime)
                bundle.assert_cold_current(provider)
                construction_stage.roots.clear()
                counters["bundle_construction_fence_count"] += 1
            except BaseException as error:
                _settle_stage_or_raise(construction_stage, error)
                raise

            lane_lifetime = prepare_paper_kinetic_native_lazy_bundle_lane_construction_lifetime(
                bundle,
                provider,
                native_ops,
                device=global_site_rgba_f32.device,
                backend_provenance=capability.backend_provenance,
                max_resident_logical_tensor_bytes=(
                    maximum_lane_resident_logical_tensor_bytes
                ),
            )
            lane_stage = _PredictionFenceStage(
                capability=capability,
                stage="heldout-native-lane-construction",
                ordinal=stage_ordinal,
                owner_generation_digest=owner_generation_digest,
                roots=[
                    provider,
                    bundle,
                    lane_lifetime,
                    global_site_rgba_f32,
                    background_rgb_f32,
                ],
            )
            stage_ordinal += 1
            lane_stage.begin()
            try:
                lane = materialize_paper_kinetic_native_lazy_bundle_lane(
                    lane_lifetime
                )
                lane_stage.roots.append(lane)
                lane_stage.fence()
                lane.assert_cold_current(provider)
                lane_stage.consume(consumer="native-heldout-lane-admission")
                lane_stage.roots.clear()
            except BaseException as error:
                _settle_stage_or_raise(lane_stage, error)
                raise

            counters["bundle_count"] += 1
            counters["lane_construction_fence_count"] += 1
            counters["native_runtime_count"] += lane.native_runtime_count
            peaks["lane"] = max(
                peaks["lane"],
                lane.resident_logical_tensor_bytes,
            )
            plan = prepare_paper_kinetic_sparse_sample_plan(
                bundle,
                provider,
                global_loss_element_count=len(selected) * 3,
                loss_normalization_id=owner_generation_digest,
                maximum_samples_per_launch=maximum_samples_per_launch,
            )
            sample_stream = iter_paper_kinetic_sparse_sample_blocks(
                plan,
                maximum_source_decode_tensor_bytes=(
                    maximum_source_decode_tensor_bytes
                ),
                require_explicit_transfer_settlement=True,
            )
            world_by_native_digest: dict[str, tuple[Any, ...]] = {}
            covered_in_bundle = 0
            try:
                while covered_in_bundle < plan.observation_count:
                    sample_stage = _PredictionFenceStage(
                        capability=capability,
                        stage="heldout-native-prediction-sample",
                        ordinal=stage_ordinal,
                        owner_generation_digest=owner_generation_digest,
                        roots=[
                            provider,
                            bundle,
                            lane,
                            plan,
                            sample_stream,
                            world_by_native_digest,
                            global_site_rgba_f32,
                            background_rgb_f32,
                        ],
                    )
                    stage_ordinal += 1
                    sample_stage.begin()
                    try:
                        sample_block = next(sample_stream)
                        sample_lifetime = sample_stream.active_lifetime_for(
                            sample_block
                        )
                        sample_stage.roots.extend(
                            (sample_block, sample_lifetime)
                        )
                        digest = sample_block.native_block_generation_digest
                        cached = world_by_native_digest.get(digest)
                        if cached is None:
                            runtime = lane.runtime_for_native_block_digest(digest)
                            compact_material = global_site_rgba_f32.index_select(
                                0,
                                runtime.source_site_ids_i64,
                            ).contiguous()
                            node_chart_out = torch.empty(
                                (runtime.row_count, runtime.node_count, 4),
                                dtype=torch.float32,
                                device=global_site_rgba_f32.device,
                            )
                            sample_stage.roots.extend(
                                (runtime, compact_material, node_chart_out)
                            )
                            world = refresh_kinetic_native_equal_rank_world_into(
                                runtime,
                                compact_material,
                                node_chart_out,
                            )
                            cached = (
                                runtime,
                                compact_material,
                                node_chart_out,
                                world,
                            )
                            world_by_native_digest[digest] = cached
                            sample_stage.roots.append(world)
                            counters["node_forward_launch_count"] += 1
                            counters["active_native_runtime_count"] += 1
                            peaks["world"] = max(
                                peaks["world"],
                                sum(
                                    _tensor_bytes((entry[1], entry[2]))
                                    for entry in world_by_native_digest.values()
                                ),
                            )
                        world = cached[3]
                        loss_f32 = torch.zeros(
                            (1,),
                            dtype=torch.float32,
                            device=global_site_rgba_f32.device,
                        )
                        grad_node_chart_f32 = torch.zeros_like(
                            world.node_chart_f32
                        )
                        cone_diagnostic_i32 = torch.zeros(
                            (3,),
                            dtype=torch.int32,
                            device=global_site_rgba_f32.device,
                        )
                        sample_stage.roots.extend(
                            (
                                loss_f32,
                                grad_node_chart_f32,
                                cone_diagnostic_i32,
                            )
                        )
                        prepared = prepare_prediction(
                            world.node_chart_f32,
                            sample_block.sample_row_i32,
                            sample_block.sample_to_node_f32,
                            sample_block.target_rgb_f32,
                            background_rgb_f32,
                            loss_scale=sample_block.loss_scale,
                            cone_tolerance=cone_tolerance,
                        )
                        sample_stage.roots.append(prepared)
                        prediction_device = launch_prediction(
                            prepared,
                            loss_f32,
                            grad_node_chart_f32,
                            cone_diagnostic_i32,
                        )
                        sample_stage.roots.append(prediction_device)
                        _require_prediction_output(
                            prediction_device,
                            sample_block=sample_block,
                            target_rgb_f32=sample_block.target_rgb_f32,
                        )
                        sample_stage.fence()

                        prediction_cpu = prediction_device.detach().to(
                            device="cpu",
                            dtype=torch.float32,
                        ).contiguous()
                        diagnostic_cpu = cone_diagnostic_i32.detach().to(
                            device="cpu",
                            dtype=torch.int32,
                        ).contiguous()
                        if (
                            tuple(prediction_cpu.shape)
                            != (sample_block.sample_count, 3)
                            or not bool(torch.isfinite(prediction_cpu).all().item())
                        ):
                            raise FloatingPointError(
                                "native WorldFoam prediction returned nonfinite RGB"
                            )
                        flat_indices = sample_block.flat_sample_index_i64
                        output_positions = torch.tensor(
                            [
                                input_position_by_observation_id[
                                    bundle.observations[int(local_index)].observation.observation_id
                                ]
                                for local_index in flat_indices.tolist()
                            ],
                            dtype=torch.int64,
                            device="cpu",
                        )
                        if (
                            output_positions.numel() != sample_block.sample_count
                            or torch.unique(output_positions).numel()
                            != output_positions.numel()
                            or bool(covered.index_select(0, output_positions).any().item())
                        ):
                            raise ValueError(
                                "native prediction sample coverage is duplicate or foreign"
                            )
                        sample_stream.assert_active_releasable_after_consumed_receipt(
                            sample_block,
                            expected_lifetime=sample_lifetime,
                        )
                        output.index_copy_(
                            0,
                            output_positions,
                            prediction_cpu,
                        )
                        covered.index_fill_(0, output_positions, True)
                        sample_stage.consume(
                            consumer="native-heldout-prediction-transfer-release"
                        )
                        sample_stream._commit_active_release_after_consumed_receipt(
                            expected_lifetime=sample_lifetime,
                        )
                        sample_stage.roots.clear()
                    except BaseException as error:
                        _settle_stage_or_raise(sample_stage, error)
                        raise

                    count = sample_block.sample_count
                    covered_in_bundle += count
                    counters["native_prediction_launch_count"] += 1
                    counters["native_sample_count"] += count
                    counters["sample_completion_fence_count"] += 1
                    counters["device_to_cpu_prediction_transfer_count"] += 1
                    peaks["sample"] = max(
                        peaks["sample"],
                        _unique_tensor_bytes(
                            (
                                *sample_block._tensors(),
                                prepared.sample_row_i32,
                                prepared.sample_to_node_f32,
                                prepared.target_rgb_f32,
                                prepared.background_rgb_f32,
                                prepared.config_i32,
                                prepared.config_f32,
                                loss_f32,
                                grad_node_chart_f32,
                                cone_diagnostic_i32,
                            )
                        ),
                    )
                    peaks["prediction"] = max(
                        peaks["prediction"],
                        _tensor_bytes((prediction_device,)),
                    )
                    diagnostic_values = tuple(
                        int(value) for value in diagnostic_cpu.tolist()
                    )
                    for index, value in enumerate(diagnostic_values):
                        if value < 0:
                            raise ValueError(
                                "native cone diagnostic contained a negative count"
                            )
                        cone_diagnostic[index] += value
                    del (
                        cached,
                        cone_diagnostic_i32,
                        diagnostic_cpu,
                        flat_indices,
                        grad_node_chart_f32,
                        loss_f32,
                        output_positions,
                        prediction_cpu,
                        prediction_device,
                        prepared,
                        sample_block,
                        sample_lifetime,
                        sample_stage,
                        world,
                    )
                if covered_in_bundle != plan.observation_count:
                    raise ArithmeticError(
                        "native prediction stream lost bundle observation coverage"
                    )
                target_accounting = sample_stream.target_read_accounting()
                _require_acceptance_capable_target_accounting(
                    target_accounting,
                    expected_observation_count=plan.observation_count,
                )
                _accumulate_target_accounting(counters, peaks, target_accounting)
            finally:
                if (
                    sample_stream.active_transfer_lifetime is None
                    and not _UNKNOWN_COMPLETION_QUARANTINE
                ):
                    sample_stream.close()
            del lane, lane_lifetime, plan, sample_stream, world_by_native_digest
            del bundle, spatial_lifetime
    finally:
        close = getattr(bundle_iterator, "close", None)
        if callable(close) and not _UNKNOWN_COMPLETION_QUARANTINE:
            close()

    if construction_slot.active_lifetime is not None:
        raise RuntimeError("native prediction retained a bundle construction lifetime")
    if not bool(covered.all().item()):
        missing = tuple(
            selected[index].observation_id
            for index in torch.nonzero(~covered, as_tuple=False).flatten().tolist()
        )
        raise ValueError(f"native prediction omitted observation ids: {missing}")
    if counters["native_sample_count"] != len(selected):
        raise ArithmeticError("native prediction sample count changed")

    receipt_without_digest = WorldFoamNativeHeldoutPredictionReceipt(
        provider_generation_digest=provider.generation_digest,
        material_generation_digest=material_generation_digest,
        observation_order_digest=observation_order_digest,
        observation_count=len(selected),
        unique_observation_id_count=len(input_position_by_observation_id),
        rasterized_sample_count=counters["native_sample_count"],
        cone_diagnostic=tuple(cone_diagnostic),
        peak_lane_resident_logical_tensor_bytes=peaks["lane"],
        peak_cached_node_world_logical_tensor_bytes=peaks["world"],
        peak_sample_launch_logical_tensor_bytes=peaks["sample"],
        peak_device_prediction_tensor_bytes=peaks["prediction"],
        peak_source_visible_target_read_logical_tensor_bytes_upper_bound=(
            peaks["target_read"]
        ),
        peak_transient_mapped_address_space_bytes=peaks["mapped_address_space"],
        returned_cpu_prediction_tensor_bytes=_tensor_bytes((output,)),
        maximum_returned_cpu_prediction_tensor_bytes=(
            maximum_returned_cpu_prediction_tensor_bytes
        ),
        backend_provenance=capability.backend_provenance,
        completion_scope=capability.completion_scope,
        sealed_device_completion_fence_count=(
            capability.successful_fence_count
        ),
        generation_digest="",
        **counters,
    )
    receipt = WorldFoamNativeHeldoutPredictionReceipt(
        **{
            **receipt_without_digest.__dict__,
            "generation_digest": _prediction_receipt_digest(
                receipt_without_digest
            ),
        }
    )
    receipt.assert_current()
    rgb_signature = _tensor_signature(output)
    result = WorldFoamNativeHeldoutPrediction(
        rgb_f32_cpu=output,
        receipt=receipt,
        generation_digest=_digest_parts(
            PREDICTION_RESULT_PROVENANCE,
            receipt.generation_digest,
            rgb_signature,
        ),
        _rgb_identity=id(output),
        _rgb_signature=rgb_signature,
    )
    result.assert_current()
    return result


def _validate_prediction_request(
    provider: PaperKineticLazyProgramBundleProvider,
    observations: Sequence[PaperKineticObservation],
    *,
    global_site_rgba_f32: torch.Tensor,
    material_generation_digest: str,
    native_ops: Any,
    background_rgb_f32: torch.Tensor,
    maximum_samples_per_launch: int,
    maximum_source_decode_tensor_bytes: int,
    maximum_lane_resident_logical_tensor_bytes: int,
    maximum_returned_cpu_prediction_tensor_bytes: int,
    cone_tolerance: float,
) -> None:
    if not isinstance(provider, PaperKineticLazyProgramBundleProvider):
        raise TypeError("native prediction requires a kinetic provider")
    if not isinstance(observations, Sequence):
        raise TypeError("native prediction observations must be a bounded sequence")
    if not observations or any(
        not isinstance(value, PaperKineticObservation) for value in observations
    ):
        raise ValueError("native prediction requires exact paper observations")
    observation_ids = tuple(value.observation_id for value in observations)
    if len(set(observation_ids)) != len(observation_ids):
        raise ValueError("native prediction requires every observation_id exactly once")
    physical_observations = tuple(
        (value.view_index, value.frame_index, value.pixel_index)
        for value in observations
    )
    if len(set(physical_observations)) != len(physical_observations):
        raise ValueError(
            "native prediction cannot rasterize one physical observation twice"
        )
    image_pixel_count = provider.height * provider.width
    if any(
        value.view_index >= provider.view_count
        or value.frame_index >= provider.frame_count
        or value.pixel_index >= image_pixel_count
        for value in observations
    ):
        raise IndexError("native prediction observation leaves the heldout provider")
    for value, name in (
        (maximum_samples_per_launch, "maximum_samples_per_launch"),
        (
            maximum_source_decode_tensor_bytes,
            "maximum_source_decode_tensor_bytes",
        ),
        (
            maximum_lane_resident_logical_tensor_bytes,
            "maximum_lane_resident_logical_tensor_bytes",
        ),
        (
            maximum_returned_cpu_prediction_tensor_bytes,
            "maximum_returned_cpu_prediction_tensor_bytes",
        ),
    ):
        _require_positive_int(value, name=name)
    required_output_bytes = len(observations) * 3 * 4
    if required_output_bytes > maximum_returned_cpu_prediction_tensor_bytes:
        raise MemoryError(
            "native prediction CPU result exceeds its explicit byte budget "
            "before allocation"
        )
    if not _is_sha256(material_generation_digest):
        raise ValueError("material_generation_digest must be a SHA-256 digest")
    if not math.isfinite(float(cone_tolerance)) or float(cone_tolerance) < 0.0:
        raise ValueError("cone_tolerance must be finite and nonnegative")
    provider.assert_current()
    _require_mps_tensor(
        global_site_rgba_f32,
        name="global_site_rgba_f32",
        dtype=torch.float32,
        shape=(provider.world.site_count, 4),
    )
    _require_mps_tensor(
        background_rgb_f32,
        name="background_rgb_f32",
        dtype=torch.float32,
        shape=(3,),
    )
    assert_worldfoam_native_heldout_prediction_abi(native_ops)


def assert_worldfoam_native_heldout_prediction_abi(native_ops: Any) -> None:
    """Allocation-free attestation of the exact post-103 prediction ABI."""

    base_attestation = getattr(
        native_ops,
        "assert_kinetic_memory_light_compiled_abi_registered",
        None,
    )
    if not callable(base_attestation):
        raise RuntimeError(
            "native heldout prediction requires the compiled memory-light ABI attestation"
        )
    if base_attestation() is not None:
        raise TypeError("compiled memory-light ABI attestation returned a value")
    for name in (_PREDICTION_PREPARE_NAME, _PREDICTION_OP_NAME):
        if not callable(getattr(native_ops, name, None)):
            raise RuntimeError(f"native heldout prediction callable is missing: {name}")
    qualified = f"{_NATIVE_NAMESPACE}::{_PREDICTION_OP_NAME}"
    try:
        handle = torch._C._dispatch_find_schema_or_throw(qualified, "")
    except RuntimeError as error:
        raise RuntimeError(
            f"required post-103 prediction schema is not registered: {qualified}"
        ) from error
    if str(handle.schema()) != _PREDICTION_SCHEMA:
        raise RuntimeError(
            "compiled post-103 prediction schema differs from its source contract"
        )
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        qualified,
        "CompositeExplicitAutograd",
    ):
        raise RuntimeError(
            "compiled post-103 prediction operator has no dispatch kernel"
        )


def _require_prediction_output(
    prediction: Any,
    *,
    sample_block: Any,
    target_rgb_f32: torch.Tensor,
) -> None:
    if (
        not isinstance(prediction, torch.Tensor)
        or prediction.device.type != "mps"
        or prediction.dtype != torch.float32
        or tuple(prediction.shape) != (sample_block.sample_count, 3)
        or not prediction.is_contiguous()
        or _same_storage(prediction, target_rgb_f32)
    ):
        raise ValueError(
            "post-103 WorldFoam prediction must return independent contiguous "
            "MPS float32 [sample_count,3] RGB"
        )


def _require_acceptance_capable_target_accounting(
    accounting: Mapping[str, int | bool | str],
    *,
    expected_observation_count: int,
) -> None:
    if (
        accounting.get("selected_pixel_read_acceptance_capable") is not True
        or int(accounting.get("selected_pixel_read_observation_count", -1))
        != expected_observation_count
        or int(accounting.get("full_frame_fallback_observation_count", -1)) != 0
        or int(accounting.get("full_frame_target_materialization_count", -1)) != 0
        or accounting.get("all_selected_pixel_mappings_closed_before_return")
        is not True
    ):
        raise RuntimeError(
            "heldout prediction target source used a non-public or full-frame fallback"
        )


def _accumulate_target_accounting(
    counters: dict[str, int],
    peaks: dict[str, int],
    accounting: Mapping[str, int | bool | str],
) -> None:
    for name in (
        "selected_pixel_read_call_count",
        "selected_pixel_read_observation_count",
        "direct_selected_pixel_observation_count",
        "bounded_region_selected_pixel_observation_count",
        "full_frame_fallback_observation_count",
        "full_frame_target_materialization_count",
        "bounded_region_target_materialization_count",
        "mapped_selected_pixel_read_call_count",
        "mapping_closed_before_return_count",
    ):
        counters[name] += int(accounting[name])
    peaks["target_read"] = max(
        peaks["target_read"],
        int(
            accounting[
                "peak_source_visible_target_read_logical_tensor_bytes_upper_bound"
            ]
        ),
    )
    peaks["mapped_address_space"] = max(
        peaks["mapped_address_space"],
        int(accounting["peak_transient_mapped_address_space_bytes"]),
    )


def _settle_stage_or_raise(
    stage: _PredictionFenceStage,
    original_error: BaseException,
) -> None:
    try:
        stage.settle_after_error()
    except BaseException as settlement_error:
        raise settlement_error from original_error


def _prediction_receipt_digest(
    receipt: WorldFoamNativeHeldoutPredictionReceipt,
) -> str:
    return _digest_parts(
        PREDICTION_RECEIPT_PROVENANCE,
        tuple(
            (name, getattr(receipt, name))
            for name in receipt.__dataclass_fields__
            if name != "generation_digest"
        ),
    )


def _observation_order_digest(
    observations: Sequence[PaperKineticObservation],
) -> str:
    return _digest_parts(
        PREDICTION_PROVENANCE,
        "input-observation-order-v1",
        tuple(observation.sample_identity for observation in observations),
    )


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        id(tensor),
        str(tensor.device),
        str(tensor.dtype),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        int(tensor.storage_offset()),
        int(tensor.untyped_storage().data_ptr()),
        int(tensor._version),
    )


def _tensor_bytes(tensors: Sequence[torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _unique_tensor_bytes(tensors: Sequence[torch.Tensor]) -> int:
    unique = {id(tensor): tensor for tensor in tensors}
    return _tensor_bytes(tuple(unique.values()))


def _same_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


def _require_mps_tensor(
    tensor: Any,
    *,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device.type != "mps"
        or tensor.dtype != dtype
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            f"{name} must be contiguous MPS {dtype} with shape {shape}"
        )


def _require_positive_int(value: int, *, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


__all__ = [
    "PREDICTION_ABI_SCHEMA_SHA256",
    "PREDICTION_PROVENANCE",
    "PREDICTION_RECEIPT_PROVENANCE",
    "PREDICTION_RESULT_PROVENANCE",
    "WorldFoamNativeHeldoutPrediction",
    "WorldFoamNativeHeldoutPredictionReceipt",
    "assert_worldfoam_native_heldout_prediction_abi",
    "predict_worldfoam_native_heldout_observations",
]
