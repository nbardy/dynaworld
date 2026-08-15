"""Bounded sparse paper targets for the native kinetic material executor.

The lazy program provider emits exact ``(view, frame, pixel)`` ray records,
not a padded ``tracks x times`` rectangle.  This module performs the matching
data-side operation:

* each unique selected ``(view, frame)`` issues one sealed selected-pixel read
  per bounded bundle;
* only explicitly requested RGB rows are returned by an acceptance-capable
  direct/bounded-region source;
* compatibility full-frame fallback and the legacy bounded frame cache remain
  explicit, non-acceptance-capable modes with materialization receipts;
* exact chart dispatch selects the native row and its row-local interpolation
  weights;
* every launch is a sealed :class:`PaperKineticRowRaggedSampleBlock` accepted
  directly by :class:`KineticNativeMaterialStepExecutor`;
* one caller-supplied global RGB denominator is preserved across every launch.

The plan itself owns no target, ray, sample, or interpolation tensor.  Without
a cache, the iterator retains one sealed selected-pixel read and at most one
yielded launch tensor set.  The source receives an explicit decode/mapping
budget before allocation.  A supplied compatibility step cache retains only
frames admitted by its explicit byte budget and has no eviction or unbounded
fallback, but cannot satisfy the direct-selected-pixel paper gate.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction

import torch
from paper_kinetic_runtime_paths import ensure_worldfoam_lane2_research_path

ensure_worldfoam_lane2_research_path()

from kinetic_multichart_transfer_program import (  # noqa: E402
    dispatch_prevalidated_kinetic_chart_index,
)
from paper_kinetic_lazy_program_bundles import (  # noqa: E402
    PaperKineticLazyProgramBundle,
    PaperKineticLazyProgramBundleProvider,
)
from paper_kinetic_ragged_sample_plan import (  # noqa: E402
    PaperKineticRowBinding,
    PaperKineticRowRaggedSampleBlock,
    seal_paper_kinetic_row_ragged_sample_block,
)
from paper_kinetic_step_target_frame_cache import (  # noqa: E402
    PaperKineticStepTargetFrameCache,
)
from powerfoam_training_data import PowerFoamSelectedPixelRead  # noqa: E402

SPARSE_SAMPLE_PROVENANCE = "paper-kinetic-sparse-sample-blocks-v2"

_PLAN_SEAL = object()
_MATERIALIZATION_LIFETIME_SEAL = object()


@dataclass
class PaperKineticSparseSampleMaterializationLifetime:
    """One transfer's CPU predecessors installed before the first `.to`."""

    plan: PaperKineticSparseSamplePlan = field(repr=False)
    native_block: object = field(repr=False)
    entries: tuple[tuple[int, int, PaperKineticRowBinding], ...] = field(
        repr=False
    )
    selected_pixel_read: PowerFoamSelectedPixelRead | None = field(repr=False)
    selected_frame_targets_f32: torch.Tensor | None = field(repr=False)
    weights_f64: torch.Tensor | None = field(repr=False)
    sample_rows_i32_cpu: torch.Tensor | None = field(repr=False)
    record_indices_i64_cpu: torch.Tensor | None = field(repr=False)
    target_indices_i64_cpu: torch.Tensor | None = field(repr=False)
    selected_targets_cpu_f32: torch.Tensor | None = field(
        default=None,
        repr=False,
    )
    weights_transfer_f32: torch.Tensor | None = field(default=None, repr=False)
    weights_f32: torch.Tensor | None = field(default=None, repr=False)
    target_transfer_f32: torch.Tensor | None = field(default=None, repr=False)
    target_rgb_f32: torch.Tensor | None = field(default=None, repr=False)
    sample_block: PaperKineticRowRaggedSampleBlock | None = field(
        default=None,
        repr=False,
    )
    current_transfer_source: torch.Tensor | None = field(default=None, repr=False)
    exact_node_row_count: int = 0
    dense_fallback_row_count: int = 0
    linear_weight_interactions: int = 0
    dense_fallback_interactions: int = 0
    dispatch_generation_digest: str = ""
    phase: str = "installed"
    released_after_completion_fence: bool = False
    _plan_identity: int = field(default=0, repr=False)
    _native_block_identity: int = field(default=0, repr=False)
    _source_tensor_identities: tuple[int, ...] = field(default=(), repr=False)
    _seal: object = field(default=None, repr=False)

    def _source_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            tensor
            for tensor in (
                self.selected_frame_targets_f32,
                self.weights_f64,
                self.sample_rows_i32_cpu,
                self.record_indices_i64_cpu,
                self.target_indices_i64_cpu,
            )
            if isinstance(tensor, torch.Tensor)
        )

    def assert_retained(self) -> None:
        if self.selected_pixel_read is not None:
            self.selected_pixel_read.assert_valid(
                expected_observation_count=int(
                    self.selected_frame_targets_f32.shape[0]
                ),
                full_frame_tensor_bytes=(
                    self.plan.provider.height
                    * self.plan.provider.width
                    * 3
                    * 4
                ),
            )
        if (
            self._seal is not _MATERIALIZATION_LIFETIME_SEAL
            or self.released_after_completion_fence
            or self.phase not in {"installed", "transferring", "materialized"}
            or id(self.plan) != self._plan_identity
            or id(self.native_block) != self._native_block_identity
            or tuple(id(tensor) for tensor in self._source_tensors())
            != self._source_tensor_identities
            or not self.entries
            or not self.dispatch_generation_digest.strip()
            or (
                self.selected_pixel_read is not None
                and self.selected_pixel_read.rgb_f32_cpu
                is not self.selected_frame_targets_f32
            )
            or (
                self.current_transfer_source is not None
                and all(
                    self.current_transfer_source is not tensor
                    for tensor in (
                        *self._source_tensors(),
                        self.selected_targets_cpu_f32,
                    )
                )
            )
            or (self.phase == "materialized")
            != isinstance(self.sample_block, PaperKineticRowRaggedSampleBlock)
        ):
            raise ValueError("sparse sample materialization lifetime changed")
        if self.sample_block is not None:
            self.sample_block.assert_warm_layout()

    def release_after_completion_fence(self) -> None:
        """Drop transfer/sample roots only after caller-proven completion."""

        if self.plan.bundle.spatial_bundle.device.type != "cpu":
            raise RuntimeError(
                "authority-free sparse sample release is CPU-only; "
                "accelerator release requires an exact consumed receipt"
            )
        self.assert_releasable_after_consumed_receipt()
        self._commit_release_after_consumed_receipt()

    def assert_releasable_after_consumed_receipt(self) -> None:
        """Validate every root before consuming a sealed completion receipt."""

        if type(self) is not PaperKineticSparseSampleMaterializationLifetime:
            raise TypeError("sparse sample materialization lifetime type changed")
        self.assert_retained()

    def _commit_release_after_consumed_receipt(self) -> None:
        """Assignment-only tail used after exact completion authority is spent."""

        self.selected_pixel_read = None
        self.selected_frame_targets_f32 = None
        self.weights_f64 = None
        self.sample_rows_i32_cpu = None
        self.record_indices_i64_cpu = None
        self.target_indices_i64_cpu = None
        self.selected_targets_cpu_f32 = None
        self.weights_transfer_f32 = None
        self.weights_f32 = None
        self.target_transfer_f32 = None
        self.target_rgb_f32 = None
        self.sample_block = None
        self.current_transfer_source = None
        self.phase = "released"
        self.released_after_completion_fence = True


class PaperKineticSparseSampleBlockStream:
    """Iterator owning at most one materialization lifetime at a time."""

    def __init__(
        self,
        plan: PaperKineticSparseSamplePlan,
        *,
        target_frame_cache: PaperKineticStepTargetFrameCache | None,
        maximum_source_decode_tensor_bytes: int,
        require_explicit_transfer_settlement: bool,
    ) -> None:
        self.plan = plan
        self.target_frame_cache = target_frame_cache
        self.maximum_source_decode_tensor_bytes = int(
            maximum_source_decode_tensor_bytes
        )
        self.require_explicit_transfer_settlement = (
            require_explicit_transfer_settlement
        )
        self._active_transfer_lifetime: (
            PaperKineticSparseSampleMaterializationLifetime | None
        ) = None
        self._closed = False
        self._failed = False
        self._selected_pixel_read_call_count = 0
        self._selected_pixel_observation_count = 0
        self._direct_selected_pixel_observation_count = 0
        self._bounded_region_selected_pixel_observation_count = 0
        self._full_frame_fallback_observation_count = 0
        self._full_frame_target_materialization_count = 0
        self._bounded_region_target_materialization_count = 0
        self._peak_full_frame_materialization_tensor_bytes = 0
        self._peak_bounded_region_materialization_tensor_bytes = 0
        self._peak_source_visible_target_read_tensor_bytes = 0
        self._peak_transient_mapped_address_space_bytes = 0
        self._mapped_selected_pixel_read_call_count = 0
        self._mapping_closed_before_return_count = 0
        self._selection_modes: set[str] = set()
        self._source_provenances: set[str] = set()
        self._iterator = _iter_paper_kinetic_sparse_sample_blocks(self)

    def _record_selected_pixel_read(
        self,
        read: PowerFoamSelectedPixelRead,
    ) -> None:
        read.assert_valid(
            full_frame_tensor_bytes=(
                self.plan.provider.height
                * self.plan.provider.width
                * 3
                * 4
            )
        )
        count = read.observation_count
        self._selected_pixel_read_call_count += 1
        self._selected_pixel_observation_count += count
        self._selection_modes.add(read.selection_mode)
        self._source_provenances.add(read.source_provenance)
        if read.selection_mode == "direct_pixels":
            self._direct_selected_pixel_observation_count += count
        elif read.selection_mode == "certified_bounded_region":
            self._bounded_region_selected_pixel_observation_count += count
        else:
            self._full_frame_fallback_observation_count += count
        self._full_frame_target_materialization_count += (
            read.full_frame_materialization_count
        )
        self._bounded_region_target_materialization_count += (
            read.bounded_region_materialization_count
        )
        self._peak_full_frame_materialization_tensor_bytes = max(
            self._peak_full_frame_materialization_tensor_bytes,
            read.maximum_full_frame_materialization_tensor_bytes,
        )
        self._peak_bounded_region_materialization_tensor_bytes = max(
            self._peak_bounded_region_materialization_tensor_bytes,
            read.maximum_bounded_region_materialization_tensor_bytes,
        )
        self._peak_source_visible_target_read_tensor_bytes = max(
            self._peak_source_visible_target_read_tensor_bytes,
            read.source_visible_peak_logical_tensor_bytes_upper_bound,
        )
        self._peak_transient_mapped_address_space_bytes = max(
            self._peak_transient_mapped_address_space_bytes,
            read.transient_mapped_address_space_bytes,
        )
        if read.transient_mapped_address_space_bytes > 0:
            self._mapped_selected_pixel_read_call_count += 1
            if read.mapping_closed_before_return:
                self._mapping_closed_before_return_count += 1

    def _record_full_frame_cache_access(
        self,
        *,
        observation_count: int,
        newly_materialized_frame_count: int,
    ) -> None:
        if observation_count < 1 or newly_materialized_frame_count not in {0, 1}:
            raise ValueError("target-frame cache access receipt is invalid")
        frame_bytes = (
            self.plan.provider.height
            * self.plan.provider.width
            * 3
            * 4
        )
        self._selected_pixel_read_call_count += 1
        self._selected_pixel_observation_count += observation_count
        self._full_frame_fallback_observation_count += observation_count
        self._full_frame_target_materialization_count += (
            newly_materialized_frame_count
        )
        self._peak_full_frame_materialization_tensor_bytes = max(
            self._peak_full_frame_materialization_tensor_bytes,
            frame_bytes if newly_materialized_frame_count else 0,
        )
        self._peak_source_visible_target_read_tensor_bytes = max(
            self._peak_source_visible_target_read_tensor_bytes,
            observation_count * 3 * 4
            + (frame_bytes if newly_materialized_frame_count else 0),
        )
        self._selection_modes.add("full_frame_cache")
        self._source_provenances.add(
            "paper_kinetic_step_target_frame_cache/full_frame_v1"
        )

    def target_read_accounting(self) -> dict[str, int | bool | str]:
        """Return primitive-only receipts after all yielded transfers settle."""

        if self._active_transfer_lifetime is not None:
            raise RuntimeError(
                "selected-target accounting requires the active transfer to settle"
            )
        if len(self._selection_modes) == 1:
            mode = next(iter(self._selection_modes))
        elif not self._selection_modes:
            mode = "none"
        else:
            mode = "mixed"
        return {
            "selected_pixel_read_mode": mode,
            "selected_pixel_read_source_provenance_digest": _digest_parts(
                tuple(sorted(self._source_provenances))
            ),
            "selected_pixel_read_source_provenance_count": len(
                self._source_provenances
            ),
            "selected_pixel_read_call_count": self._selected_pixel_read_call_count,
            "selected_pixel_read_observation_count": (
                self._selected_pixel_observation_count
            ),
            "selected_pixel_read_acceptance_capable": (
                self._selected_pixel_read_call_count > 0
                and self._selection_modes
                <= {"direct_pixels", "certified_bounded_region"}
                and self._full_frame_target_materialization_count == 0
                and self._full_frame_fallback_observation_count == 0
            ),
            "direct_selected_pixel_observation_count": (
                self._direct_selected_pixel_observation_count
            ),
            "bounded_region_selected_pixel_observation_count": (
                self._bounded_region_selected_pixel_observation_count
            ),
            "full_frame_fallback_observation_count": (
                self._full_frame_fallback_observation_count
            ),
            "full_frame_target_materialization_count": (
                self._full_frame_target_materialization_count
            ),
            "bounded_region_target_materialization_count": (
                self._bounded_region_target_materialization_count
            ),
            "peak_full_frame_materialization_tensor_bytes": (
                self._peak_full_frame_materialization_tensor_bytes
            ),
            "peak_bounded_region_materialization_tensor_bytes": (
                self._peak_bounded_region_materialization_tensor_bytes
            ),
            "peak_source_visible_target_read_logical_tensor_bytes_upper_bound": (
                self._peak_source_visible_target_read_tensor_bytes
            ),
            "peak_transient_mapped_address_space_bytes": (
                self._peak_transient_mapped_address_space_bytes
            ),
            "mapped_selected_pixel_read_call_count": (
                self._mapped_selected_pixel_read_call_count
            ),
            "mapping_closed_before_return_count": (
                self._mapping_closed_before_return_count
            ),
            "all_selected_pixel_mappings_closed_before_return": (
                self._mapping_closed_before_return_count
                == self._mapped_selected_pixel_read_call_count
            ),
            "target_source_decode_budget_enforced_before_allocation": True,
        }

    @property
    def active_transfer_lifetime(
        self,
    ) -> PaperKineticSparseSampleMaterializationLifetime | None:
        return self._active_transfer_lifetime

    def __iter__(self) -> PaperKineticSparseSampleBlockStream:
        return self

    def __next__(self) -> PaperKineticRowRaggedSampleBlock:
        if self._closed:
            raise StopIteration
        if self._failed:
            raise RuntimeError(
                "sparse sample stream failed; retry would duplicate a transfer"
            )
        active = self._active_transfer_lifetime
        if active is not None:
            if self.require_explicit_transfer_settlement:
                raise RuntimeError(
                    "sparse sample transfer must be settled before advancing"
                )
            if self.plan.bundle.spatial_bundle.device.type != "cpu":
                raise RuntimeError(
                    "accelerator sparse transfer requires explicit settlement"
                )
            active.release_after_completion_fence()
            self._active_transfer_lifetime = None
        try:
            lifetime = next(self._iterator)
        except StopIteration:
            self._closed = True
            raise
        except BaseException:
            self._failed = True
            raise
        block = lifetime.sample_block
        if not isinstance(block, PaperKineticRowRaggedSampleBlock):
            self._failed = True
            raise ValueError("sparse sample iterator lost its materialized block")
        return block

    def active_lifetime_for(
        self,
        sample_block: PaperKineticRowRaggedSampleBlock,
    ) -> PaperKineticSparseSampleMaterializationLifetime:
        active = self._active_transfer_lifetime
        if active is None or active.sample_block is not sample_block:
            raise ValueError("sample block has no active transfer lifetime")
        active.assert_retained()
        return active

    def release_active_after_completion_fence(
        self,
        sample_block: PaperKineticRowRaggedSampleBlock | None = None,
    ) -> None:
        if self.plan.bundle.spatial_bundle.device.type != "cpu":
            raise RuntimeError(
                "authority-free sparse stream release is CPU-only; "
                "accelerator release requires an exact consumed receipt"
            )
        active = self._active_transfer_lifetime
        if active is None:
            return
        if sample_block is not None and active.sample_block is not sample_block:
            raise ValueError("completion fence belongs to another sample block")
        active.release_after_completion_fence()
        self._active_transfer_lifetime = None

    def assert_active_releasable_after_consumed_receipt(
        self,
        sample_block: PaperKineticRowRaggedSampleBlock | None = None,
        expected_lifetime: (
            PaperKineticSparseSampleMaterializationLifetime | None
        ) = None,
    ) -> PaperKineticSparseSampleMaterializationLifetime:
        """Validate the exact active transfer before receipt consumption."""

        if type(self) is not PaperKineticSparseSampleBlockStream:
            raise TypeError("sparse sample stream type changed")
        active = self._active_transfer_lifetime
        if active is None:
            raise RuntimeError("sparse sample stream has no active transfer")
        if expected_lifetime is not None and active is not expected_lifetime:
            raise ValueError("active sparse sample lifetime identity changed")
        if type(active) is not PaperKineticSparseSampleMaterializationLifetime:
            raise TypeError("active sparse sample lifetime type changed")
        if sample_block is not None and active.sample_block is not sample_block:
            raise ValueError("completion receipt belongs to another sample block")
        active.assert_releasable_after_consumed_receipt()
        return active

    def _commit_active_release_after_consumed_receipt(
        self,
        expected_lifetime: (
            PaperKineticSparseSampleMaterializationLifetime | None
        ) = None,
    ) -> None:
        """Release the prevalidated active transfer without another check."""

        active = self._active_transfer_lifetime
        if expected_lifetime is not None:
            active = expected_lifetime
        elif self.plan.bundle.spatial_bundle.device.type != "cpu":
            raise RuntimeError(
                "accelerator sparse release commit requires its exact lifetime"
            )
        active._commit_release_after_consumed_receipt()
        self._active_transfer_lifetime = None

    def close(self) -> None:
        if self._closed:
            return
        active = self._active_transfer_lifetime
        if active is not None:
            if self.require_explicit_transfer_settlement:
                raise RuntimeError(
                    "cannot close sparse sample stream with an unsettled transfer"
                )
            if self.plan.bundle.spatial_bundle.device.type != "cpu":
                raise RuntimeError(
                    "cannot implicitly release an accelerator sparse transfer"
                )
            active.release_after_completion_fence()
            self._active_transfer_lifetime = None
        self._iterator.close()
        self._closed = True


@dataclass(frozen=True)
class PaperKineticSparseSampleMemoryReport:
    observation_count: int
    unique_selected_frame_count: int
    maximum_samples_per_launch: int
    maximum_node_count: int
    persistent_target_tensor_bytes: int
    persistent_ray_tensor_bytes: int
    persistent_sample_tensor_bytes: int
    persistent_interpolation_weight_tensor_bytes: int
    dense_track_time_tensor_bytes: int
    decoded_frame_scratch_upper_bound_bytes: int
    direct_selected_pixel_source_visible_upper_bound_bytes: int
    maximum_selected_observations_per_frame: int
    selected_frame_target_tensor_upper_bound_bytes: int
    launch_tensor_upper_bound_bytes: int
    selected_target_python_scalar_count: int
    target_python_allocator_bytes_measured: bool
    allocator_peak_measured: bool


@dataclass(frozen=True)
class PaperKineticSparseSamplePlan:
    """Tensor-free exact coverage plan for one lazy spatial bundle."""

    provider: PaperKineticLazyProgramBundleProvider = field(repr=False)
    bundle: PaperKineticLazyProgramBundle = field(repr=False)
    global_loss_element_count: int
    caller_loss_normalization_id: str
    maximum_samples_per_launch: int
    observation_generation_digests: tuple[str, ...]
    unique_selected_frames: tuple[tuple[int, int], ...]
    native_block_generation_digests: tuple[str, ...]
    generation_digest: str
    _provider_identity: int = field(repr=False)
    _bundle_identity: int = field(repr=False)
    provenance: str = SPARSE_SAMPLE_PROVENANCE
    persistent_target_tensor_bytes: int = 0
    persistent_ray_tensor_bytes: int = 0
    persistent_sample_tensor_bytes: int = 0
    persistent_interpolation_weight_tensor_bytes: int = 0
    dense_track_time_tensor_bytes: int = 0
    _seal: object = field(default=None, repr=False)

    @property
    def observation_count(self) -> int:
        return self.bundle.observation_count

    @property
    def loss_scale(self) -> float:
        return 1.0 / float(self.global_loss_element_count)

    @property
    def block_loss_normalization_id(self) -> str:
        return f"{self.caller_loss_normalization_id}:{self.generation_digest}"

    def memory_report(self) -> PaperKineticSparseSampleMemoryReport:
        maximum_node_count = max(row.node_count for row in self.bundle.sampler.rows)
        frame_counts: dict[tuple[int, int], int] = {}
        for record in self.bundle.observations:
            frame = (
                record.observation.view_index,
                record.observation.frame_index,
            )
            frame_counts[frame] = frame_counts.get(frame, 0) + 1
        maximum_selected_observations_per_frame = max(frame_counts.values())
        # The resident/direct source uses three int64 request vectors, two
        # int64 row/column temporaries, and one float32 RGB result per selected
        # observation.  Other native selected-pixel sources enforce the caller
        # supplied source-decode cap before allocation and report their exact
        # source-visible upper bound in the sealed read receipt.
        direct_selected_pixel_source_visible_upper_bound_bytes = (
            maximum_selected_observations_per_frame * (5 * 8 + 3 * 4)
        )
        sample_bytes = 4 + maximum_node_count * 4 + 3 * 4 + 8
        return PaperKineticSparseSampleMemoryReport(
            observation_count=self.observation_count,
            unique_selected_frame_count=len(self.unique_selected_frames),
            maximum_samples_per_launch=self.maximum_samples_per_launch,
            maximum_node_count=maximum_node_count,
            persistent_target_tensor_bytes=0,
            persistent_ray_tensor_bytes=0,
            persistent_sample_tensor_bytes=0,
            persistent_interpolation_weight_tensor_bytes=0,
            dense_track_time_tensor_bytes=0,
            decoded_frame_scratch_upper_bound_bytes=(self.provider.height * self.provider.width * 3 * 4),
            direct_selected_pixel_source_visible_upper_bound_bytes=(
                direct_selected_pixel_source_visible_upper_bound_bytes
            ),
            maximum_selected_observations_per_frame=(maximum_selected_observations_per_frame),
            selected_frame_target_tensor_upper_bound_bytes=(maximum_selected_observations_per_frame * 3 * 4),
            launch_tensor_upper_bound_bytes=(self.maximum_samples_per_launch * sample_bytes),
            selected_target_python_scalar_count=0,
            target_python_allocator_bytes_measured=False,
            allocator_peak_measured=False,
        )

    def accounting(self) -> dict[str, int | bool | str]:
        report = self.memory_report()
        return {
            "provenance": self.provenance,
            "observation_count": self.observation_count,
            "unique_selected_frame_count": len(self.unique_selected_frames),
            "maximum_samples_per_launch": self.maximum_samples_per_launch,
            "global_loss_element_count": self.global_loss_element_count,
            "loss_scale": self.loss_scale,
            "persistent_target_tensor_bytes": 0,
            "persistent_ray_tensor_bytes": 0,
            "persistent_sample_tensor_bytes": 0,
            "persistent_interpolation_weight_tensor_bytes": 0,
            "dense_track_time_tensor_bytes": 0,
            "decoded_frame_scratch_upper_bound_bytes": (report.decoded_frame_scratch_upper_bound_bytes),
            "direct_selected_pixel_source_visible_upper_bound_bytes": (
                report.direct_selected_pixel_source_visible_upper_bound_bytes
            ),
            "maximum_selected_observations_per_frame": (report.maximum_selected_observations_per_frame),
            "selected_frame_target_tensor_upper_bound_bytes": (report.selected_frame_target_tensor_upper_bound_bytes),
            "launch_tensor_upper_bound_bytes": report.launch_tensor_upper_bound_bytes,
            "read_each_bundle_local_selected_frame_at_most_once": True,
            "frame_major_selected_pixel_streaming": True,
            "frame_major_target_streaming": True,
            "whole_bundle_target_tuple_retained": False,
            "selected_target_python_scalar_count": 0,
            "explicit_selected_pixels_only": True,
            "cartesian_padding_observation_count": 0,
            "executor_compatible_sample_block": True,
            "provider_retains_yielded_sample_blocks": False,
            "allocator_peak_measured": False,
        }

    def assert_current(self) -> None:
        if (
            self._seal is not _PLAN_SEAL
            or self.provenance != SPARSE_SAMPLE_PROVENANCE
            or id(self.provider) != self._provider_identity
            or id(self.bundle) != self._bundle_identity
            or self.global_loss_element_count < self.observation_count * 3
            or not self.caller_loss_normalization_id.strip()
            or self.maximum_samples_per_launch < 1
            or self.observation_generation_digests
            != tuple(record.generation_digest for record in self.bundle.observations)
            or self.unique_selected_frames
            != tuple(
                sorted(
                    {
                        (
                            record.observation.view_index,
                            record.observation.frame_index,
                        )
                        for record in self.bundle.observations
                    }
                )
            )
            or self.native_block_generation_digests
            != tuple(binding.native_block_generation_digest for binding in self.bundle.spatial_bundle.native_blocks)
            or self.persistent_target_tensor_bytes != 0
            or self.persistent_ray_tensor_bytes != 0
            or self.persistent_sample_tensor_bytes != 0
            or self.persistent_interpolation_weight_tensor_bytes != 0
            or self.dense_track_time_tensor_bytes != 0
        ):
            raise ValueError("paper kinetic sparse sample plan metadata changed")
        self.bundle.assert_cold_current(self.provider)
        if self.generation_digest != _plan_digest(self):
            raise ValueError("paper kinetic sparse sample plan generation changed")

    def assert_complete_launch_coverage(
        self,
        blocks: Sequence[PaperKineticRowRaggedSampleBlock],
    ) -> None:
        """Cold diagnostic for a retained test/debug launch sequence."""

        self.assert_current()
        selected = tuple(blocks)
        if not selected:
            raise ValueError("paper kinetic sparse launch coverage is empty")
        covered = []
        for block in selected:
            block.assert_cold_current(self.bundle.sampler)
            if (
                block.global_loss_element_count != self.global_loss_element_count
                or block.loss_normalization_id != self.block_loss_normalization_id
                or block.native_block_generation_digest not in self.native_block_generation_digests
                or block.sample_count > self.maximum_samples_per_launch
            ):
                raise ValueError("paper kinetic sparse launch has foreign provenance")
            covered.extend(int(value) for value in block.flat_sample_index_i64.tolist())
        if sorted(covered) != list(range(self.observation_count)):
            raise ValueError("paper kinetic sparse launches have missing or duplicate coverage")


def prepare_paper_kinetic_sparse_sample_plan(
    bundle: PaperKineticLazyProgramBundle,
    provider: PaperKineticLazyProgramBundleProvider,
    *,
    global_loss_element_count: int,
    loss_normalization_id: str,
    maximum_samples_per_launch: int,
) -> PaperKineticSparseSamplePlan:
    """Seal one sparse bundle without decoding any target."""

    if not isinstance(bundle, PaperKineticLazyProgramBundle):
        raise TypeError("paper kinetic sparse planning requires a lazy bundle")
    if not isinstance(provider, PaperKineticLazyProgramBundleProvider):
        raise TypeError("paper kinetic sparse planning requires its lazy provider")
    bundle.assert_cold_current(provider)
    _require_positive_int(
        global_loss_element_count,
        name="global_loss_element_count",
    )
    _require_positive_int(
        maximum_samples_per_launch,
        name="maximum_samples_per_launch",
    )
    if global_loss_element_count < bundle.observation_count * 3:
        raise ValueError("global RGB denominator is smaller than this sparse bundle")
    if not isinstance(loss_normalization_id, str) or not loss_normalization_id.strip():
        raise ValueError("loss_normalization_id must be nonempty")
    provisional = PaperKineticSparseSamplePlan(
        provider=provider,
        bundle=bundle,
        global_loss_element_count=global_loss_element_count,
        caller_loss_normalization_id=loss_normalization_id,
        maximum_samples_per_launch=maximum_samples_per_launch,
        observation_generation_digests=tuple(record.generation_digest for record in bundle.observations),
        unique_selected_frames=tuple(
            sorted({(record.observation.view_index, record.observation.frame_index) for record in bundle.observations})
        ),
        native_block_generation_digests=tuple(
            binding.native_block_generation_digest for binding in bundle.spatial_bundle.native_blocks
        ),
        generation_digest="",
        _provider_identity=id(provider),
        _bundle_identity=id(bundle),
        _seal=_PLAN_SEAL,
    )
    result = replace(provisional, generation_digest=_plan_digest(provisional))
    result.assert_current()
    return result


def iter_paper_kinetic_sparse_sample_blocks(
    plan: PaperKineticSparseSamplePlan,
    *,
    target_frame_cache: PaperKineticStepTargetFrameCache | None = None,
    maximum_source_decode_tensor_bytes: int | None = None,
    require_explicit_transfer_settlement: bool = False,
) -> PaperKineticSparseSampleBlockStream:
    """Stream one sealed selected-pixel read through native chunks at a time.

    Launches may interleave native block identities across frames.  The native
    executor's step session accumulates every launch into the corresponding
    resident node-chart cotangent and performs one VJP per active block only
    after this iterator is exhausted.  Consequently, target residency is one
    sealed selected-pixel read plus one launch payload.  Any compatibility
    full-frame materialization is surfaced by the stream receipt.
    """

    if not isinstance(plan, PaperKineticSparseSamplePlan):
        raise TypeError("paper kinetic sparse launch iteration requires a plan")
    plan.assert_current()
    if target_frame_cache is not None:
        if not isinstance(
            target_frame_cache,
            PaperKineticStepTargetFrameCache,
        ):
            raise TypeError("target_frame_cache has the wrong type")
        target_frame_cache.assert_open_current(plan.provider)
    if maximum_source_decode_tensor_bytes is None:
        report = plan.memory_report()
        maximum_source_decode_tensor_bytes = max(
            report.direct_selected_pixel_source_visible_upper_bound_bytes,
            report.decoded_frame_scratch_upper_bound_bytes
            + report.maximum_selected_observations_per_frame
            * (2 * 3 * 4 + 4 * 8),
        )
    if (
        isinstance(maximum_source_decode_tensor_bytes, bool)
        or not isinstance(maximum_source_decode_tensor_bytes, int)
        or maximum_source_decode_tensor_bytes < 1
    ):
        raise ValueError("selected-pixel source-decode budget must be positive")
    if not isinstance(require_explicit_transfer_settlement, bool):
        raise TypeError("require_explicit_transfer_settlement must be bool")
    if (
        plan.bundle.spatial_bundle.device.type != "cpu"
        and not require_explicit_transfer_settlement
    ):
        raise RuntimeError(
            "accelerator sparse sample streaming requires explicit transfer settlement"
        )
    return PaperKineticSparseSampleBlockStream(
        plan,
        target_frame_cache=target_frame_cache,
        maximum_source_decode_tensor_bytes=maximum_source_decode_tensor_bytes,
        require_explicit_transfer_settlement=(
            require_explicit_transfer_settlement
        ),
    )


def _iter_paper_kinetic_sparse_sample_blocks(
    stream: PaperKineticSparseSampleBlockStream,
) -> Iterator[PaperKineticSparseSampleMaterializationLifetime]:
    plan = stream.plan
    target_frame_cache = stream.target_frame_cache
    records_by_frame = _record_indices_by_frame(plan)
    dispatched_count = 0
    for (view_index, frame_index), record_indices in records_by_frame:
        selected_targets, selected_pixel_read = _decode_selected_frame_targets(
            stream,
            plan,
            view_index=view_index,
            frame_index=frame_index,
            record_indices=record_indices,
            target_frame_cache=target_frame_cache,
        )
        entries = _dispatch_sparse_frame_records(plan, record_indices)
        for native_digest in plan.native_block_generation_digests:
            native_entries = entries.get(native_digest, ())
            for start in range(
                0,
                len(native_entries),
                plan.maximum_samples_per_launch,
            ):
                chunk = native_entries[start : start + plan.maximum_samples_per_launch]
                lifetime = _prepare_sparse_materialization_lifetime(
                    plan,
                    native_digest,
                    chunk,
                    selected_targets,
                    selected_pixel_read=selected_pixel_read,
                )
                if stream._active_transfer_lifetime is not None:
                    raise RuntimeError(
                        "sparse stream already owns a transfer lifetime"
                    )
                stream._active_transfer_lifetime = lifetime
                _materialize_sparse_launch(lifetime)
                dispatched_count += len(chunk)
                # Yield the owner, not a second direct reference to its
                # sample block.  Settlement clears ``lifetime.sample_block``
                # before the caller advances the public stream, so the
                # suspended generator cannot keep the previous payload alive.
                yield lifetime
        del selected_pixel_read, selected_targets, entries
    if dispatched_count != plan.observation_count:
        raise ArithmeticError("paper kinetic sparse dispatch lost exact observation coverage")


def _record_indices_by_frame(
    plan: PaperKineticSparseSamplePlan,
) -> tuple[tuple[tuple[int, int], tuple[int, ...]], ...]:
    records_by_frame: dict[tuple[int, int], list[int]] = {}
    for record_index, record in enumerate(plan.bundle.observations):
        key = (record.observation.view_index, record.observation.frame_index)
        records_by_frame.setdefault(key, []).append(record_index)
    if tuple(sorted(records_by_frame)) != plan.unique_selected_frames:
        raise ValueError("paper kinetic sparse target frame provenance changed")
    if sum(len(selected) for selected in records_by_frame.values()) != plan.observation_count:
        raise ArithmeticError("paper kinetic sparse frame partition lost exact observation coverage")
    return tuple((frame, tuple(selected)) for frame, selected in sorted(records_by_frame.items()))


def _decode_selected_frame_targets(
    stream: PaperKineticSparseSampleBlockStream,
    plan: PaperKineticSparseSamplePlan,
    *,
    view_index: int,
    frame_index: int,
    record_indices: tuple[int, ...],
    target_frame_cache: PaperKineticStepTargetFrameCache | None,
) -> tuple[torch.Tensor, PowerFoamSelectedPixelRead | None]:
    if not record_indices:
        raise ValueError("paper kinetic sparse target frame cannot be empty")

    pixel_values = tuple(
        plan.bundle.observations[record_index].observation.pixel_index
        for record_index in record_indices
    )
    if target_frame_cache is None:
        provider = plan.provider.target_provider
        selected_pixel_read = provider.select_view_frame_pixels_cpu(
            (view_index,) * len(pixel_values),
            (frame_index,) * len(pixel_values),
            pixel_values,
            maximum_source_decode_tensor_bytes=(
                stream.maximum_source_decode_tensor_bytes
            ),
        )
        selected_pixel_read.assert_valid(
            expected_observation_count=len(record_indices),
            full_frame_tensor_bytes=(
                plan.provider.height * plan.provider.width * 3 * 4
            ),
        )
        stream._record_selected_pixel_read(selected_pixel_read)
        selected_targets = selected_pixel_read.rgb_f32_cpu
    else:
        pixels = torch.tensor(
            pixel_values,
            dtype=torch.int64,
            device="cpu",
        )
        cache_before = target_frame_cache.accounting()
        frame = target_frame_cache.get_frame(
            plan.provider,
            view_index=view_index,
            frame_index=frame_index,
        )
        selected_targets = frame.gather_pixels(
            target_frame_cache,
            plan.provider,
            pixels,
        )
        del frame
        cache_after = target_frame_cache.accounting()
        newly_materialized = int(cache_after["decode_count"]) - int(
            cache_before["decode_count"]
        )
        if newly_materialized not in {0, 1}:
            raise ArithmeticError("target-frame cache decode receipt changed")
        stream._record_full_frame_cache_access(
            observation_count=len(record_indices),
            newly_materialized_frame_count=newly_materialized,
        )
        selected_pixel_read = None
        del pixels
    if not bool(torch.isfinite(selected_targets).all().item()):
        raise ValueError("paper kinetic sparse selected targets are nonfinite")
    return selected_targets, selected_pixel_read


def _dispatch_sparse_frame_records(
    plan: PaperKineticSparseSamplePlan,
    record_indices: tuple[int, ...],
) -> dict[str, tuple[tuple[int, int, PaperKineticRowBinding], ...]]:
    sampler = plan.bundle.sampler
    row_by_identity = {row.row_identity: row for row in sampler.rows}
    first_row_by_track = {
        track_id: next(row for row in sampler.rows if row.track_id == track_id) for track_id in sampler.track_ids
    }
    entries: dict[str, list[tuple[int, int, PaperKineticRowBinding]]] = {}
    for target_index, record_index in enumerate(record_indices):
        record = plan.bundle.observations[record_index]
        track_id = record.observation.pixel_index
        try:
            first_row = first_row_by_track[track_id]
        except KeyError as error:
            raise ValueError("paper kinetic sparse record has no compiled track") from error
        chart_index = dispatch_prevalidated_kinetic_chart_index(
            first_row.program,
            Fraction.from_float(record.sample_time),
            expected_generation_digest=first_row.program_generation_digest,
        )
        try:
            row = row_by_identity[(track_id, chart_index)]
        except KeyError as error:
            raise ValueError("paper kinetic sparse dispatch selected an unbound row") from error
        entries.setdefault(row.native_block_generation_digest, []).append((record_index, target_index, row))
    foreign = set(entries) - set(plan.native_block_generation_digests)
    if foreign:
        raise ValueError("paper kinetic sparse dispatch selected a foreign native block")
    return {digest: tuple(values) for digest, values in entries.items()}


def _prepare_sparse_materialization_lifetime(
    plan: PaperKineticSparseSamplePlan,
    native_digest: str,
    entries: tuple[tuple[int, int, PaperKineticRowBinding], ...],
    selected_frame_targets: torch.Tensor,
    *,
    selected_pixel_read: PowerFoamSelectedPixelRead | None,
) -> PaperKineticSparseSampleMaterializationLifetime:
    if not entries:
        raise ValueError("paper kinetic sparse launch cannot be empty")
    native_blocks = tuple(
        block
        for bucket in plan.bundle.sampler.lowering.buckets
        for block in bucket.blocks
        if block.generation_digest == native_digest
    )
    if len(native_blocks) != 1:
        raise ValueError("paper kinetic sparse launch has no unique native block")
    native_block = native_blocks[0]
    sample_count = len(entries)
    weights_f64 = torch.empty(
        (sample_count, native_block.node_count),
        dtype=torch.float64,
        device="cpu",
    )
    sample_rows = torch.empty(sample_count, dtype=torch.int32, device="cpu")
    record_indices = torch.empty(sample_count, dtype=torch.int64, device="cpu")
    exact_rows = 0
    fallback_rows = 0
    linear_interactions = 0
    fallback_interactions = 0
    entries_by_row: dict[tuple[int, int], list[int]] = {}
    for sample_index, (record_index, _target_index, row) in enumerate(entries):
        sample_rows[sample_index] = row.native_local_row_index
        record_indices[sample_index] = record_index
        entries_by_row.setdefault(row.row_identity, []).append(sample_index)
    for row_identity, sample_indices in entries_by_row.items():
        row = next(row for row in plan.bundle.sampler.rows if row.row_identity == row_identity)
        times = torch.tensor(
            [plan.bundle.observations[entries[sample_index][0]].sample_time for sample_index in sample_indices],
            dtype=torch.float64,
            device="cpu",
        )
        interpolation = row.program.charts[row.chart_index].schedule.sample_to_node_weights(times)
        destination = torch.tensor(sample_indices, dtype=torch.int64, device="cpu")
        weights_f64.index_copy_(0, destination, interpolation.weights)
        exact_rows += interpolation.exact_node_row_count
        fallback_rows += interpolation.dense_fallback_row_count
        linear_interactions += interpolation.linear_weight_interactions
        fallback_interactions += interpolation.dense_fallback_interactions

    target_indices = torch.tensor(
        [target_index for _record_index, target_index, _row in entries],
        dtype=torch.int64,
        device="cpu",
    )
    dispatch_digest = _digest_parts(
        SPARSE_SAMPLE_PROVENANCE,
        plan.generation_digest,
        native_digest,
        tuple(
            (
                record_index,
                plan.bundle.observations[record_index].generation_digest,
                row.global_row_index,
                row.native_local_row_index,
            )
            for record_index, _target_index, row in entries
        ),
    )
    sources = (
        selected_frame_targets,
        weights_f64,
        sample_rows,
        record_indices,
        target_indices,
    )
    lifetime = PaperKineticSparseSampleMaterializationLifetime(
        plan=plan,
        native_block=native_block,
        entries=entries,
        selected_pixel_read=selected_pixel_read,
        selected_frame_targets_f32=selected_frame_targets,
        weights_f64=weights_f64,
        sample_rows_i32_cpu=sample_rows,
        record_indices_i64_cpu=record_indices,
        target_indices_i64_cpu=target_indices,
        exact_node_row_count=exact_rows,
        dense_fallback_row_count=fallback_rows,
        linear_weight_interactions=linear_interactions,
        dense_fallback_interactions=fallback_interactions,
        dispatch_generation_digest=dispatch_digest,
        _plan_identity=id(plan),
        _native_block_identity=id(native_block),
        _source_tensor_identities=tuple(id(tensor) for tensor in sources),
        _seal=_MATERIALIZATION_LIFETIME_SEAL,
    )
    lifetime.assert_retained()
    return lifetime


def _materialize_sparse_launch(
    lifetime: PaperKineticSparseSampleMaterializationLifetime,
) -> PaperKineticRowRaggedSampleBlock:
    lifetime.assert_retained()
    if lifetime.phase != "installed":
        raise ValueError("sparse sample materialization lifetime was already used")
    lifetime.phase = "transferring"
    device = lifetime.plan.bundle.spatial_bundle.device
    lifetime.current_transfer_source = lifetime.weights_f64
    lifetime.weights_transfer_f32 = lifetime.weights_f64.to(
        device=device,
        dtype=torch.float32,
    )
    lifetime.weights_f32 = lifetime.weights_transfer_f32.contiguous()
    lifetime.selected_targets_cpu_f32 = (
        lifetime.selected_frame_targets_f32.index_select(
            0,
            lifetime.target_indices_i64_cpu,
        ).contiguous()
    )
    lifetime.current_transfer_source = lifetime.selected_targets_cpu_f32
    lifetime.target_transfer_f32 = lifetime.selected_targets_cpu_f32.to(
        device=device,
        dtype=torch.float32,
    )
    lifetime.target_rgb_f32 = lifetime.target_transfer_f32.contiguous()
    lifetime.current_transfer_source = None
    block = seal_paper_kinetic_row_ragged_sample_block(
        lifetime.plan.bundle.sampler,
        native_block_generation_digest=(
            lifetime.native_block.generation_digest
        ),
        sample_row_i32=lifetime.sample_rows_i32_cpu,
        sample_to_node_f32=lifetime.weights_f32,
        target_rgb_f32=lifetime.target_rgb_f32,
        flat_sample_index_i64=lifetime.record_indices_i64_cpu,
        global_loss_element_count=lifetime.plan.global_loss_element_count,
        loss_normalization_id=lifetime.plan.block_loss_normalization_id,
        exact_node_row_count=lifetime.exact_node_row_count,
        dense_fallback_row_count=lifetime.dense_fallback_row_count,
        linear_weight_interactions=lifetime.linear_weight_interactions,
        dense_fallback_interactions=lifetime.dense_fallback_interactions,
        dispatch_generation_digest=lifetime.dispatch_generation_digest,
    )
    block.assert_cold_current(lifetime.plan.bundle.sampler)
    lifetime.sample_block = block
    lifetime.phase = "materialized"
    lifetime.assert_retained()
    return block


def _plan_digest(plan: PaperKineticSparseSamplePlan) -> str:
    return _digest_parts(
        SPARSE_SAMPLE_PROVENANCE,
        plan.provider.generation_digest,
        plan.bundle.generation_digest,
        plan.global_loss_element_count,
        plan.caller_loss_normalization_id,
        plan.maximum_samples_per_launch,
        plan.observation_generation_digests,
        plan.unique_selected_frames,
        plan.native_block_generation_digests,
        0,
        0,
        0,
        0,
        0,
    )


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _require_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


__all__ = [
    "SPARSE_SAMPLE_PROVENANCE",
    "PaperKineticSparseSampleBlockStream",
    "PaperKineticSparseSampleMaterializationLifetime",
    "PaperKineticSparseSampleMemoryReport",
    "PaperKineticSparseSamplePlan",
    "iter_paper_kinetic_sparse_sample_blocks",
    "prepare_paper_kinetic_sparse_sample_plan",
]
