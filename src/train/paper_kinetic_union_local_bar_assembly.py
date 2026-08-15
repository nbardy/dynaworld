"""Bounded union-local material-bar assembly for kinetic WorldFoam.

One paper spatial request can touch several heterogeneous equal-rank native
blocks.  Each native block has its own compact source-site table, while the
outer paper coordinator accepts exactly one compact bar for the spatial
request.  This module closes that join without creating a request-sized
``[global_site_count, 4]`` buffer:

* cold preparation seals the sorted union of source ids used by the spatial
  bundle and one compact-to-union map per native block;
* a request manifest replays only exact chart dispatch and retains one pair of
  integer counts per active native block, never a sample/target/weight tensor;
* one caller-owned ``[union_site_count, 4]`` bar and one caller-owned scalar
  loss are zeroed once;
* every expected native block contributes one compact native VJP result;
  ``index_add_`` merges repeated source sites into the union bar;
* finalization seals that same union bar directly as the generic
  :class:`PaperRaggedCompactMaterialBarResult` consumed by the outer
  coordinator.

Warm validation is identity/layout/version only.  It performs no content
hash, CPU copy, scalar extraction, list conversion, or tensor allocation.
Exact time dispatch is deliberately a cold request-planning operation.  The
proof boundary is also explicit: this module proves assembly coverage and one
accepted VJP result per active native block; the injected sample reducer and
native adapter remain responsible for proving that their returned VJP is the
derivative of the streamed node-bar reduction.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import torch
from paper_kinetic_runtime_paths import ensure_worldfoam_lane2_research_path

ensure_worldfoam_lane2_research_path()

from kinetic_multichart_transfer_program import dispatch_prevalidated_kinetic_chart_index  # noqa: E402
from paper_kinetic_ragged_sample_plan import PaperKineticRowRaggedSampler  # noqa: E402
from paper_ragged_material_bar_coordinator import (  # noqa: E402
    PaperRaggedCompactMaterialBarResult,
    PaperRaggedMaterialBarRequest,
    seal_paper_ragged_compact_material_bar_result,
)

UNION_LOCAL_PROVENANCE = "paper-kinetic-union-local-material-bar-v1"
WARM_VALIDATION_KIND = "identity_shape_stride_dtype_device_version_only"
PROOF_BOUNDARY = (
    "assembly proves one compact contribution for every cold-dispatched active native block; "
    "the injected sample reducer/native adapter proves chunk-to-node reduction and VJP numerics"
)

_BINDING_SEAL = object()
_BUNDLE_SEAL = object()
_BUNDLE_CONSTRUCTION_LIFETIME_SEAL = object()
_COLD_RECEIPT_LIFETIME_SEAL = object()
_WORK_SEAL = object()
_CONTRIBUTION_SEAL = object()
_ASSEMBLY_SEAL = object()


@dataclass
class PaperKineticUnionLocalColdReceiptLifetime:
    """Own one cold union-map device-to-host receipt until completion.

    ``source_tensors_i64`` are installed before the first transfer.  Each
    detached transfer source, raw device-to-host result, and contiguous CPU
    destination is then published into this caller-visible object before the
    next potentially synchronizing conversion or content check.  A failed or
    interrupted receipt therefore remains rooted by the bundle construction
    lifetime instead of depending on Python temporary lifetime.  Successfully
    compared destinations are released one at a time, so transient receipt
    storage is bounded by the largest union/map tensor rather than their sum.
    """

    bundle_identity: int
    bundle_generation_digest: str
    source_tensors_i64: tuple[torch.Tensor, ...] = field(repr=False)
    expected_int_tuples: tuple[tuple[int, ...], ...] = field(repr=False)
    source_tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    current_transfer_source: torch.Tensor | None = field(default=None, repr=False)
    current_raw_device_to_host_result: torch.Tensor | None = field(
        default=None,
        repr=False,
    )
    current_cpu_destination_tensor: torch.Tensor | None = field(
        default=None,
        repr=False,
    )
    current_converted_int_tuple: tuple[int, ...] | None = field(
        default=None,
        repr=False,
    )
    current_source_index: int | None = None
    validated_source_count: int = 0
    validated_content_digest: str = ""
    phase: str = "installed"
    _source_tensor_identities: tuple[int, ...] = field(default=(), repr=False)
    _seal: object = field(default=None, repr=False)

    @property
    def source_tensor_count(self) -> int:
        return len(self.source_tensor_signatures)

    def assert_retained(self) -> None:
        source_count = len(self.source_tensor_signatures)
        current_roots = (
            self.current_transfer_source,
            self.current_raw_device_to_host_result,
            self.current_cpu_destination_tensor,
            self.current_converted_int_tuple,
        )
        if (
            self._seal is not _COLD_RECEIPT_LIFETIME_SEAL
            or self.bundle_identity < 1
            or not self.bundle_generation_digest.strip()
            or self.phase
            not in {
                "installed",
                "transferring",
                "validating",
                "validated",
                "retired",
            }
            or (
                self.phase != "retired"
                and len(self.source_tensors_i64) != source_count
            )
            or (
                self.phase != "retired"
                and tuple(id(tensor) for tensor in self.source_tensors_i64)
                != self._source_tensor_identities
            )
            or (
                self.phase != "retired"
                and tuple(_warm_tensor_signature(tensor) for tensor in self.source_tensors_i64)
                != self.source_tensor_signatures
            )
            or (
                self.phase != "retired"
                and source_count != len(self.expected_int_tuples)
            )
            or not 0 <= self.validated_source_count <= source_count
            or (
                self.current_raw_device_to_host_result is not None
                and self.current_transfer_source is None
            )
            or (
                self.current_cpu_destination_tensor is not None
                and self.current_raw_device_to_host_result is None
            )
            or (
                self.current_converted_int_tuple is not None
                and self.current_cpu_destination_tensor is None
            )
            or (self.current_source_index is None) != all(root is None for root in current_roots)
            or (
                self.phase == "installed"
                and (
                    self.validated_source_count != 0
                    or self.current_source_index is not None
                    or self.validated_content_digest != ""
                )
            )
            or (
                self.phase in {"validating", "validated"}
                and (
                    self.validated_source_count != source_count
                    or self.current_source_index is not None
                )
            )
            or (
                self.phase == "validated"
                and not self.validated_content_digest.strip()
            )
            or (
                self.current_source_index is not None
                and (
                    self.phase != "transferring"
                    or self.current_source_index < 0
                    or self.current_source_index >= source_count
                )
            )
            or (
                self.phase == "retired"
                and (
                    self.source_tensors_i64 != ()
                    or self.expected_int_tuples != ()
                    or self._source_tensor_identities != ()
                    or any(root is not None for root in current_roots)
                    or self.current_source_index is not None
                    or not self.validated_content_digest.strip()
                )
            )
        ):
            raise ValueError("union-local cold receipt lifetime changed")

    def assert_for_bundle(self, bundle: PaperKineticUnionLocalSpatialBundle) -> None:
        self.assert_retained()
        bundle_sources = (
            bundle.source_site_ids_i64,
            *(binding.compact_to_union_i64 for binding in bundle.native_blocks),
        )
        if (
            self.bundle_identity != id(bundle)
            or self.bundle_generation_digest != bundle.generation_digest
            or tuple(id(tensor) for tensor in self.source_tensors_i64)
            != tuple(id(tensor) for tensor in bundle_sources)
        ):
            raise ValueError("union-local cold receipt is foreign to the spatial bundle")

    def retire_after_proven_completion_boundary(self) -> None:
        """Release transfer roots only after validation proved completion."""

        self.assert_retained()
        if self.phase == "retired":
            return
        if self.phase != "validated":
            raise ValueError("union-local cold receipt has no proven completion boundary")
        self.source_tensors_i64 = ()
        self.expected_int_tuples = ()
        self._source_tensor_identities = ()
        self.current_transfer_source = None
        self.current_raw_device_to_host_result = None
        self.current_cpu_destination_tensor = None
        self.current_converted_int_tuple = None
        self.current_source_index = None
        self.phase = "retired"
        self.assert_retained()


@dataclass
class PaperKineticUnionLocalSpatialBundleConstructionLifetime:
    """Roots every union-map predecessor before its first device copy."""

    sampler: PaperKineticRowRaggedSampler = field(repr=False)
    track_ids: tuple[int, ...]
    selected_blocks: tuple[Any, ...] = field(repr=False)
    union_source_site_ids: tuple[int, ...]
    compact_to_union_by_block: tuple[tuple[int, ...], ...]
    source_tensors_i64_cpu: tuple[torch.Tensor, ...] = field(repr=False)
    device: torch.device
    transferred_tensors: list[torch.Tensor] = field(default_factory=list, repr=False)
    transfer_intermediates: list[torch.Tensor] = field(default_factory=list, repr=False)
    bindings: list[PaperKineticUnionLocalNativeBlockBinding] = field(
        default_factory=list,
        repr=False,
    )
    current_transfer_source: torch.Tensor | None = field(default=None, repr=False)
    cold_receipt_lifetime: PaperKineticUnionLocalColdReceiptLifetime | None = field(
        default=None,
        repr=False,
    )
    cold_receipt_install_count: int = 0
    cold_receipt_retirement_count: int = 0
    bundle_identity: int = 0
    bundle_generation_digest: str = ""
    phase: str = "installed"
    _sampler_identity: int = field(default=0, repr=False)
    _selected_block_identities: tuple[int, ...] = field(default=(), repr=False)
    _source_tensor_identities: tuple[int, ...] = field(default=(), repr=False)
    _seal: object = field(default=None, repr=False)

    def assert_retained(self) -> None:
        outstanding_cold_receipts = (
            self.cold_receipt_install_count - self.cold_receipt_retirement_count
        )
        if (
            self._seal is not _BUNDLE_CONSTRUCTION_LIFETIME_SEAL
            or self.phase
            not in {
                "installed",
                "transferring",
                "materialized",
                "settled",
                "retired",
            }
            or id(self.sampler) != self._sampler_identity
            or (
                self.phase != "retired"
                and tuple(id(block) for block in self.selected_blocks)
                != self._selected_block_identities
            )
            or tuple(id(tensor) for tensor in self.source_tensors_i64_cpu)
            != self._source_tensor_identities
            or (
                self.phase not in {"settled", "retired"}
                and len(self.source_tensors_i64_cpu)
                != len(self.selected_blocks) + 1
            )
            or (
                self.phase in {"settled", "retired"}
                and self.source_tensors_i64_cpu != ()
            )
            or (
                self.phase != "retired"
                and len(self.compact_to_union_by_block)
                != len(self.selected_blocks)
            )
            or (
                self.phase != "retired"
                and len(self.transferred_tensors) > len(self.selected_blocks) + 1
            )
            or len(self.transfer_intermediates)
            > 3 * len(self.source_tensors_i64_cpu)
            or (
                self.phase != "retired"
                and len(self.bindings) > len(self.selected_blocks)
            )
            or (
                self.current_transfer_source is not None
                and all(
                    self.current_transfer_source is not tensor
                    for tensor in self.source_tensors_i64_cpu
                )
            )
            or (self.phase in {"materialized", "settled", "retired"})
            != (self.bundle_identity > 0 and self.bundle_generation_digest != "")
            or outstanding_cold_receipts not in {0, 1}
            or (
                self.cold_receipt_lifetime is None
                and (
                    self.cold_receipt_install_count != 0
                    or self.cold_receipt_retirement_count != 0
                )
            )
            or (
                self.cold_receipt_lifetime is not None
                and (
                    self.cold_receipt_lifetime.bundle_identity != self.bundle_identity
                    or self.cold_receipt_lifetime.bundle_generation_digest
                    != self.bundle_generation_digest
                    or (
                        self.cold_receipt_lifetime.phase == "retired"
                        and outstanding_cold_receipts != 0
                    )
                    or (
                        self.cold_receipt_lifetime.phase != "retired"
                        and outstanding_cold_receipts != 1
                    )
                )
            )
        ):
            raise ValueError("union-local construction lifetime changed")
        if self.cold_receipt_lifetime is not None:
            self.cold_receipt_lifetime.assert_retained()

    def install_cold_receipt_lifetime(
        self,
        receipt: PaperKineticUnionLocalColdReceiptLifetime,
    ) -> None:
        """Publish one receipt before any of its device-to-host work."""

        self.assert_retained()
        receipt.assert_retained()
        if self.phase not in {"materialized", "settled"}:
            raise ValueError("union-local bundle is not ready for a cold receipt")
        if (
            self.cold_receipt_lifetime is not None
            and self.cold_receipt_lifetime.phase != "retired"
        ):
            raise RuntimeError("union-local cold receipt lifetime is already active")
        if (
            receipt.phase != "installed"
            or receipt.bundle_identity != self.bundle_identity
            or receipt.bundle_generation_digest != self.bundle_generation_digest
        ):
            raise ValueError("union-local cold receipt lifetime is foreign")
        self.cold_receipt_lifetime = receipt
        self.cold_receipt_install_count += 1
        self.assert_retained()

    def retire_cold_receipt_after_proven_completion_boundary(
        self,
        receipt: PaperKineticUnionLocalColdReceiptLifetime,
    ) -> None:
        self.assert_retained()
        if self.cold_receipt_lifetime is not receipt:
            raise ValueError("union-local cold receipt retirement is foreign")
        receipt.retire_after_proven_completion_boundary()
        self.cold_receipt_retirement_count += 1
        self.assert_retained()

    def release_transfer_predecessors_after_completion_fence(self) -> None:
        """Drop CPU/raw duplicate roots after caller-proven completion."""

        if self.device.type != "cpu":
            raise RuntimeError(
                "authority-free union transfer release is CPU-only; "
                "accelerator release requires an exact consumed subject receipt"
            )
        self.assert_transfer_predecessors_releasable_after_consumed_receipt()
        if self.phase == "settled":
            return
        self._commit_transfer_predecessors_after_consumed_receipt()
        self.assert_retained()

    def assert_transfer_predecessors_releasable_after_consumed_receipt(self) -> None:
        """Validate transfer roots before consuming exact completion authority."""

        if type(self) is not PaperKineticUnionLocalSpatialBundleConstructionLifetime:
            raise TypeError("union-local construction lifetime type changed")
        self.assert_retained()
        if self.phase not in {"materialized", "settled"}:
            raise ValueError("union-local construction is not materialized")
        if type(self.transfer_intermediates) is not list:
            raise ValueError("union-local transfer intermediates changed type")

    def assert_accelerator_transfer_releasable_after_completion_fence(
        self,
        bundle: PaperKineticUnionLocalSpatialBundle,
    ) -> None:
        """Prove one source-to-device map copy without a device readback.

        The exact CPU sources and the exact device destinations remain rooted
        by this lifetime until the outer coordinator has obtained its sealed
        backend completion receipt.  Content is certified from the CPU source
        tuples and the construction operation; warm destination validation is
        identity/shape/stride/dtype/device/version only.  No accelerator value
        is copied to CPU and no device scalar is read.
        """

        self.assert_transfer_predecessors_releasable_after_consumed_receipt()
        if self.device.type == "cpu":
            raise ValueError(
                "accelerator union-map settlement requires a non-CPU device"
            )
        if self.phase != "materialized":
            raise ValueError(
                "accelerator union-map settlement requires pending materialized transfers"
            )
        if (
            not isinstance(bundle, PaperKineticUnionLocalSpatialBundle)
            or bundle._construction_lifetime is not self
            or self.bundle_identity != id(bundle)
            or self.bundle_generation_digest != bundle.generation_digest
            or self.cold_receipt_lifetime is not None
            or self.cold_receipt_install_count != 0
            or self.cold_receipt_retirement_count != 0
        ):
            raise ValueError(
                "accelerator union-map settlement received a foreign or read-back bundle"
            )
        expected_sources = (
            self.union_source_site_ids,
            *self.compact_to_union_by_block,
        )
        if len(self.source_tensors_i64_cpu) != len(expected_sources):
            raise ValueError("accelerator union-map CPU source count changed")
        for tensor, expected in zip(
            self.source_tensors_i64_cpu,
            expected_sources,
            strict=True,
        ):
            if (
                tensor.device.type != "cpu"
                or tensor.dtype != torch.int64
                or not tensor.is_contiguous()
                or tuple(int(value) for value in tensor.tolist()) != expected
            ):
                raise ValueError("accelerator union-map CPU source content changed")
        destinations = (
            bundle.source_site_ids_i64,
            *(binding.compact_to_union_i64 for binding in bundle.native_blocks),
        )
        if (
            len(self.transferred_tensors) != len(destinations)
            or any(
                transferred is not destination
                for transferred, destination in zip(
                    self.transferred_tensors,
                    destinations,
                    strict=True,
                )
            )
        ):
            raise ValueError("accelerator union-map destination identity changed")
        bundle.assert_warm_layout()

    def _commit_transfer_predecessors_after_consumed_receipt(self) -> None:
        """Assignment-only transfer-root release after a receipt is consumed."""

        self.source_tensors_i64_cpu = ()
        self._source_tensor_identities = ()
        self.transfer_intermediates.clear()
        self.current_transfer_source = None
        self.phase = "settled"

    def retire_after_completion_fence(self) -> None:
        """Break duplicate roots and poison the spent bundle after its fence."""

        if self.device.type != "cpu":
            raise RuntimeError(
                "authority-free union retirement is CPU-only; "
                "accelerator retirement requires an exact consumed subject receipt"
            )
        self.assert_retirable_after_consumed_receipt()
        if self.phase == "retired":
            return
        self._commit_retire_after_consumed_receipt()
        self.assert_retained()

    def assert_retirable_after_consumed_receipt(self) -> None:
        """Validate every mutable construction root before receipt consumption."""

        if type(self) is not PaperKineticUnionLocalSpatialBundleConstructionLifetime:
            raise TypeError("union-local construction lifetime type changed")
        self.assert_retained()
        if self.phase not in {"materialized", "settled", "retired"}:
            raise ValueError("union-local construction is not retirable")
        for name, value in (
            ("transfer_intermediates", self.transfer_intermediates),
            ("transferred_tensors", self.transferred_tensors),
            ("bindings", self.bindings),
        ):
            if type(value) is not list:
                raise ValueError(f"union-local {name} changed type")

    def _commit_retire_after_consumed_receipt(self) -> None:
        """Assignment/list-clear-only retirement after authority is spent."""

        self.source_tensors_i64_cpu = ()
        self._source_tensor_identities = ()
        self.transfer_intermediates.clear()
        self.current_transfer_source = None
        self.selected_blocks = ()
        self._selected_block_identities = ()
        self.compact_to_union_by_block = ()
        self.transferred_tensors.clear()
        self.bindings.clear()
        self.phase = "retired"


@dataclass(frozen=True)
class PaperKineticUnionLocalNativeBlockBinding:
    """One native compact source table mapped into the spatial union."""

    native_block_generation_digest: str
    compact_source_site_ids: tuple[int, ...]
    compact_to_union_i64: torch.Tensor = field(repr=False)
    mapping_generation_digest: str
    warm_tensor_signature: tuple[object, ...] = field(repr=False)
    _seal: object = field(default=None, repr=False)

    @property
    def compact_site_count(self) -> int:
        return len(self.compact_source_site_ids)

    @property
    def mapping_tensor_bytes(self) -> int:
        return self.compact_to_union_i64.numel() * self.compact_to_union_i64.element_size()

    def assert_warm_layout(self, *, device: torch.device, union_site_count: int) -> None:
        if self._seal is not _BINDING_SEAL:
            raise ValueError("union-local native binding was not sealed by its preparer")
        if (
            not self.native_block_generation_digest.strip()
            or not self.mapping_generation_digest.strip()
            or self.compact_site_count < 1
            or tuple(sorted(set(self.compact_source_site_ids))) != self.compact_source_site_ids
            or min(self.compact_source_site_ids) < 0
        ):
            raise ValueError("union-local native binding metadata changed")
        if _warm_tensor_signature(self.compact_to_union_i64) != self.warm_tensor_signature:
            raise ValueError("union-local compact-to-union mapping identity/layout/version changed")
        _require_warm_tensor(
            self.compact_to_union_i64,
            name="compact_to_union_i64",
            device=device,
            dtype=torch.int64,
            shape=(self.compact_site_count,),
        )
        if union_site_count < self.compact_site_count:
            raise ValueError("union-local mapping cannot fit inside its declared union")


@dataclass(frozen=True)
class PaperKineticUnionLocalMemoryReport:
    """Logical tensor bytes; allocator storage/peak and Python heap are unmeasured."""

    requested_frame_count: int
    native_block_count: int
    union_site_count: int
    summed_native_compact_site_count: int
    union_source_site_id_tensor_bytes: int
    compact_to_union_mapping_tensor_bytes: int
    persistent_mapping_tensor_bytes: int
    request_union_material_bar_bytes: int
    maximum_native_compact_material_bar_bytes: int
    request_loss_tensor_bytes: int
    per_request_global_material_bar_bytes: int
    persistent_frame_tensor_bytes: int
    persistent_sample_tensor_bytes: int
    persistent_target_tensor_bytes: int
    persistent_prediction_tensor_bytes: int
    descriptor_canonical_metadata_bytes: int
    allocator_storage_bytes_measured: bool
    allocator_peak_measured: bool
    python_object_bytes_measured: bool


@dataclass(frozen=True)
class PaperKineticUnionLocalSpatialBundle:
    """Cold-sealed source-site union for one view-local spatial track set."""

    sampler: PaperKineticRowRaggedSampler = field(repr=False)
    view_index: int
    track_ids: tuple[int, ...]
    global_site_count: int
    union_source_site_ids: tuple[int, ...]
    source_site_ids_i64: torch.Tensor = field(repr=False)
    native_blocks: tuple[PaperKineticUnionLocalNativeBlockBinding, ...]
    descriptor_canonical_metadata_bytes: int
    generation_digest: str
    warm_source_site_signature: tuple[object, ...] = field(repr=False)
    _sampler_identity: int = field(repr=False)
    _construction_lifetime: (
        PaperKineticUnionLocalSpatialBundleConstructionLifetime
    ) = field(repr=False)
    _construction_lifetime_identity: int = field(repr=False)
    provenance: str = UNION_LOCAL_PROVENANCE
    warm_validation_kind: str = WARM_VALIDATION_KIND
    warm_validation_device_to_host_syncs: int = 0
    warm_validation_tensor_allocations: int = 0
    requested_frame_sampling_used_for_compile: bool = False
    global_common_temporal_refinement_used: bool = False
    persistent_frame_tensor_bytes: int = 0
    persistent_sample_tensor_bytes: int = 0
    persistent_target_tensor_bytes: int = 0
    persistent_prediction_tensor_bytes: int = 0
    _seal: object = field(default=None, repr=False)

    @property
    def device(self) -> torch.device:
        return self.source_site_ids_i64.device

    @property
    def union_site_count(self) -> int:
        return len(self.union_source_site_ids)

    @property
    def native_block_count(self) -> int:
        return len(self.native_blocks)

    @property
    def compact_to_union_by_block(self) -> tuple[tuple[int, ...], ...]:
        """Cold-sealed CPU metadata; tensor equality still needs certification."""

        return self._construction_lifetime.compact_to_union_by_block

    @property
    def persistent_mapping_tensor_bytes(self) -> int:
        return _tensor_bytes((self.source_site_ids_i64,)) + sum(
            binding.mapping_tensor_bytes for binding in self.native_blocks
        )

    def binding_for_digest(
        self,
        native_block_generation_digest: str,
    ) -> PaperKineticUnionLocalNativeBlockBinding:
        matches = tuple(
            binding
            for binding in self.native_blocks
            if binding.native_block_generation_digest == native_block_generation_digest
        )
        if len(matches) != 1:
            raise ValueError("native contribution is foreign to the sealed union-local bundle")
        return matches[0]

    def memory_report(self, requested_frame_count: int) -> PaperKineticUnionLocalMemoryReport:
        _require_positive_int(requested_frame_count, name="requested_frame_count")
        union_ids_bytes = _tensor_bytes((self.source_site_ids_i64,))
        mapping_bytes = sum(binding.mapping_tensor_bytes for binding in self.native_blocks)
        return PaperKineticUnionLocalMemoryReport(
            requested_frame_count=requested_frame_count,
            native_block_count=self.native_block_count,
            union_site_count=self.union_site_count,
            summed_native_compact_site_count=sum(
                binding.compact_site_count for binding in self.native_blocks
            ),
            union_source_site_id_tensor_bytes=union_ids_bytes,
            compact_to_union_mapping_tensor_bytes=mapping_bytes,
            persistent_mapping_tensor_bytes=union_ids_bytes + mapping_bytes,
            request_union_material_bar_bytes=self.union_site_count * 4 * 4,
            maximum_native_compact_material_bar_bytes=max(
                binding.compact_site_count * 4 * 4 for binding in self.native_blocks
            ),
            request_loss_tensor_bytes=4,
            per_request_global_material_bar_bytes=0,
            persistent_frame_tensor_bytes=0,
            persistent_sample_tensor_bytes=0,
            persistent_target_tensor_bytes=0,
            persistent_prediction_tensor_bytes=0,
            descriptor_canonical_metadata_bytes=self.descriptor_canonical_metadata_bytes,
            allocator_storage_bytes_measured=False,
            allocator_peak_measured=False,
            python_object_bytes_measured=False,
        )

    def assert_warm_layout(self) -> None:
        """Validate the persistent seal without reading tensor contents."""

        if self._seal is not _BUNDLE_SEAL:
            raise ValueError("union-local spatial bundle was not sealed by its preparer")
        if (
            self.provenance != UNION_LOCAL_PROVENANCE
            or self.warm_validation_kind != WARM_VALIDATION_KIND
            or self.warm_validation_device_to_host_syncs != 0
            or self.warm_validation_tensor_allocations != 0
            or self.requested_frame_sampling_used_for_compile
            or self.global_common_temporal_refinement_used
            or self.persistent_frame_tensor_bytes != 0
            or self.persistent_sample_tensor_bytes != 0
            or self.persistent_target_tensor_bytes != 0
            or self.persistent_prediction_tensor_bytes != 0
            or id(self.sampler) != self._sampler_identity
            or id(self._construction_lifetime)
            != self._construction_lifetime_identity
            or self._construction_lifetime.phase
            not in {"materialized", "settled"}
            or self._construction_lifetime.bundle_identity != id(self)
            or self._construction_lifetime.bundle_generation_digest
            != self.generation_digest
            or self.view_index != self.sampler.view_index
            or self.global_site_count != self.sampler.lowering.global_site_count
            or not self.track_ids
            or tuple(sorted(set(self.track_ids))) != self.track_ids
            or not self.union_source_site_ids
            or tuple(sorted(set(self.union_source_site_ids))) != self.union_source_site_ids
            or self.union_site_count > self.global_site_count
            or not self.native_blocks
        ):
            raise ValueError("union-local spatial bundle metadata/memory contract changed")
        self._construction_lifetime.assert_retained()
        self.sampler.assert_warm_layout()
        if _warm_tensor_signature(self.source_site_ids_i64) != self.warm_source_site_signature:
            raise ValueError("union-local source union identity/layout/version changed")
        _require_warm_tensor(
            self.source_site_ids_i64,
            name="union_source_site_ids_i64",
            device=self.device,
            dtype=torch.int64,
            shape=(self.union_site_count,),
        )
        seen: set[str] = set()
        for binding in self.native_blocks:
            binding.assert_warm_layout(device=self.device, union_site_count=self.union_site_count)
            if binding.native_block_generation_digest in seen:
                raise ValueError("union-local bundle contains a duplicate native block")
            seen.add(binding.native_block_generation_digest)

    def assert_current(self) -> None:
        """Coordinator-compatible warm-validation alias."""

        self.assert_warm_layout()

    def assert_accelerator_transfer_pending(self) -> None:
        """Validate the pre-fence accelerator copy without reading it back."""

        self.assert_warm_layout()
        lifetime = self._construction_lifetime
        if (
            self.device.type == "cpu"
            or lifetime.device != self.device
            or lifetime.phase != "materialized"
            or lifetime.cold_receipt_lifetime is not None
            or lifetime.cold_receipt_install_count != 0
            or lifetime.cold_receipt_retirement_count != 0
        ):
            raise ValueError(
                "union-local accelerator transfer is not a pending no-readback copy"
            )
        lifetime.assert_accelerator_transfer_releasable_after_completion_fence(
            self
        )

    def assert_accelerator_cold_current_after_settlement(self) -> None:
        """Cold-admit a completed accelerator copy with zero device readback.

        The caller must first consume the exact outer completion receipt and
        commit the construction lifetime's transfer-predecessor release.  This
        method then proves the settled identity/layout/generation relation; it
        never inspects accelerator contents.
        """

        self.assert_warm_layout()
        lifetime = self._construction_lifetime
        if (
            self.device.type == "cpu"
            or lifetime.device != self.device
            or lifetime.phase != "settled"
            or lifetime.source_tensors_i64_cpu != ()
            or lifetime.transfer_intermediates
            or lifetime.current_transfer_source is not None
            or lifetime.cold_receipt_lifetime is not None
            or lifetime.cold_receipt_install_count != 0
            or lifetime.cold_receipt_retirement_count != 0
            or self.generation_digest != _bundle_digest(self)
        ):
            raise ValueError(
                "union-local accelerator transfer lacks exact settled cold admission"
            )

    def assert_cold_current(
        self,
        *,
        receipt_lifetime: PaperKineticUnionLocalColdReceiptLifetime | None = None,
    ) -> None:
        """Opt-in content/provenance recertification at a cold boundary.

        CPU tensors can be inspected synchronously without a transfer lease.
        An accelerator tensor requires an explicitly installed receipt lifetime;
        implicit device-to-host temporaries are deliberately rejected.
        """

        self.sampler.assert_cold_current()
        self.assert_warm_layout()
        expected_maps = self._construction_lifetime.compact_to_union_by_block
        expected_contents = (self.union_source_site_ids, *expected_maps)
        if receipt_lifetime is None:
            if self.device.type != "cpu":
                self.assert_accelerator_cold_current_after_settlement()
                return
            for source_index, (tensor, expected) in enumerate(
                zip(
                    (
                        self.source_site_ids_i64,
                        *(binding.compact_to_union_i64 for binding in self.native_blocks),
                    ),
                    expected_contents,
                    strict=True,
                )
            ):
                observed = _cpu_int_tuple_without_transfer(tensor)
                _validate_one_cold_union_map_receipt(
                    self,
                    source_index=source_index,
                    observed=observed,
                    expected=expected,
                )
        else:
            if self._construction_lifetime.cold_receipt_lifetime is not receipt_lifetime:
                raise ValueError("union-local cold receipt was not caller-visible")
            receipt_lifetime.assert_for_bundle(self)
            if receipt_lifetime.phase != "installed":
                raise ValueError("union-local cold receipt was already consumed")
            if receipt_lifetime.expected_int_tuples != expected_contents:
                raise ValueError("union-local cold receipt expectation changed")
            _materialize_and_validate_cold_receipt_contents(self, receipt_lifetime)
            receipt_lifetime.phase = "validating"
            receipt_lifetime.assert_retained()
        if self.generation_digest != _bundle_digest(self):
            raise ValueError("union-local spatial bundle generation changed")
        if receipt_lifetime is not None:
            receipt_lifetime.phase = "validated"
            receipt_lifetime.assert_retained()


def install_paper_kinetic_union_local_cold_receipt_lifetime(
    bundle: PaperKineticUnionLocalSpatialBundle,
) -> PaperKineticUnionLocalColdReceiptLifetime:
    """Install a caller-visible receipt before its first device-to-host copy."""

    if not isinstance(bundle, PaperKineticUnionLocalSpatialBundle):
        raise TypeError("cold receipt preparation requires a union-local spatial bundle")
    bundle.assert_warm_layout()
    expected_maps = bundle._construction_lifetime.compact_to_union_by_block
    sources = (
        bundle.source_site_ids_i64,
        *(binding.compact_to_union_i64 for binding in bundle.native_blocks),
    )
    receipt = PaperKineticUnionLocalColdReceiptLifetime(
        bundle_identity=id(bundle),
        bundle_generation_digest=bundle.generation_digest,
        source_tensors_i64=sources,
        expected_int_tuples=(bundle.union_source_site_ids, *expected_maps),
        source_tensor_signatures=tuple(_warm_tensor_signature(tensor) for tensor in sources),
        _source_tensor_identities=tuple(id(tensor) for tensor in sources),
        _seal=_COLD_RECEIPT_LIFETIME_SEAL,
    )
    receipt.assert_for_bundle(bundle)
    bundle._construction_lifetime.install_cold_receipt_lifetime(receipt)
    receipt.assert_for_bundle(bundle)
    return receipt


def certify_paper_kinetic_union_local_spatial_bundle_cold_current(
    bundle: PaperKineticUnionLocalSpatialBundle,
) -> PaperKineticUnionLocalColdReceiptLifetime:
    """Cold-certify through a bounded caller-visible receipt and retire it."""

    receipt = install_paper_kinetic_union_local_cold_receipt_lifetime(bundle)
    bundle.assert_cold_current(receipt_lifetime=receipt)
    bundle._construction_lifetime.retire_cold_receipt_after_proven_completion_boundary(
        receipt
    )
    return receipt


def prepare_paper_kinetic_union_local_spatial_bundle_construction_lifetime(
    sampler: PaperKineticRowRaggedSampler,
    *,
    track_ids: Sequence[int],
    device: torch.device | str,
) -> PaperKineticUnionLocalSpatialBundleConstructionLifetime:
    """Install CPU predecessors before any union-local device tensor exists."""

    if not isinstance(sampler, PaperKineticRowRaggedSampler):
        raise TypeError("union-local preparation requires PaperKineticRowRaggedSampler")
    sampler.assert_cold_current()
    normalized_tracks = tuple(sorted(int(track_id) for track_id in track_ids))
    if not normalized_tracks or len(set(normalized_tracks)) != len(normalized_tracks):
        raise ValueError("union-local track ids must be unique and nonempty")
    if any(track_id not in sampler.track_ids for track_id in normalized_tracks):
        raise ValueError("union-local track id has no compiled kinetic program")
    selected_rows = tuple(row for row in sampler.rows if row.track_id in set(normalized_tracks))
    selected_digests = {
        row.native_block_generation_digest for row in selected_rows
    }
    selected_blocks = tuple(
        block
        for bucket in sampler.lowering.buckets
        for block in bucket.blocks
        if block.generation_digest in selected_digests
    )
    if not selected_blocks:
        raise ValueError("union-local spatial bundle selected no native blocks")
    union_ids = tuple(
        sorted({source_id for block in selected_blocks for source_id in block.source_site_ids})
    )
    union_position = {source_id: index for index, source_id in enumerate(union_ids)}
    resolved_device = torch.device(device)
    compact_to_union_by_block = tuple(
        tuple(union_position[source_id] for source_id in block.source_site_ids)
        for block in selected_blocks
    )
    source_tensors = (
        torch.tensor(union_ids, dtype=torch.int64, device="cpu").contiguous(),
        *(
            torch.tensor(values, dtype=torch.int64, device="cpu").contiguous()
            for values in compact_to_union_by_block
        ),
    )
    lifetime = PaperKineticUnionLocalSpatialBundleConstructionLifetime(
        sampler=sampler,
        track_ids=normalized_tracks,
        selected_blocks=selected_blocks,
        union_source_site_ids=union_ids,
        compact_to_union_by_block=compact_to_union_by_block,
        source_tensors_i64_cpu=source_tensors,
        device=resolved_device,
        _sampler_identity=id(sampler),
        _selected_block_identities=tuple(id(block) for block in selected_blocks),
        _source_tensor_identities=tuple(id(tensor) for tensor in source_tensors),
        _seal=_BUNDLE_CONSTRUCTION_LIFETIME_SEAL,
    )
    lifetime.assert_retained()
    return lifetime


def materialize_paper_kinetic_union_local_spatial_bundle(
    lifetime: PaperKineticUnionLocalSpatialBundleConstructionLifetime,
) -> PaperKineticUnionLocalSpatialBundle:
    """Materialize one preinstalled union-local lifetime exactly once."""

    if not isinstance(
        lifetime,
        PaperKineticUnionLocalSpatialBundleConstructionLifetime,
    ):
        raise TypeError(
            "lifetime must be PaperKineticUnionLocalSpatialBundleConstructionLifetime"
        )
    lifetime.assert_retained()
    if lifetime.phase != "installed":
        raise ValueError("union-local construction lifetime was already used")
    lifetime.phase = "transferring"
    for source in lifetime.source_tensors_i64_cpu:
        lifetime.current_transfer_source = source
        transferred = _copy_or_alias_i64_with_construction_lifetime(
            lifetime,
            source,
        )
        lifetime.transferred_tensors.append(transferred)
    lifetime.current_transfer_source = None
    union_tensor = lifetime.transferred_tensors[0]
    bindings = []
    for block, compact_to_union, mapping_tensor in zip(
        lifetime.selected_blocks,
        lifetime.compact_to_union_by_block,
        lifetime.transferred_tensors[1:],
        strict=True,
    ):
        bindings.append(
            PaperKineticUnionLocalNativeBlockBinding(
                native_block_generation_digest=block.generation_digest,
                compact_source_site_ids=block.source_site_ids,
                compact_to_union_i64=mapping_tensor,
                mapping_generation_digest=_mapping_digest(
                    sampler_generation_digest=lifetime.sampler.generation_digest,
                    native_block_generation_digest=block.generation_digest,
                    compact_source_site_ids=block.source_site_ids,
                    union_source_site_ids=lifetime.union_source_site_ids,
                    compact_to_union=compact_to_union,
                ),
                warm_tensor_signature=_warm_tensor_signature(mapping_tensor),
                _seal=_BINDING_SEAL,
            )
        )
    binding_tuple = tuple(bindings)
    metadata = repr(
        (
            UNION_LOCAL_PROVENANCE,
            lifetime.sampler.generation_digest,
            lifetime.sampler.view_index,
            lifetime.track_ids,
            lifetime.sampler.lowering.global_site_count,
            lifetime.union_source_site_ids,
            tuple(
                (
                    binding.native_block_generation_digest,
                    binding.compact_source_site_ids,
                    binding.mapping_generation_digest,
                )
                for binding in binding_tuple
            ),
        )
    ).encode("utf-8")
    provisional = PaperKineticUnionLocalSpatialBundle(
        sampler=lifetime.sampler,
        view_index=lifetime.sampler.view_index,
        track_ids=lifetime.track_ids,
        global_site_count=lifetime.sampler.lowering.global_site_count,
        union_source_site_ids=lifetime.union_source_site_ids,
        source_site_ids_i64=union_tensor,
        native_blocks=binding_tuple,
        descriptor_canonical_metadata_bytes=len(metadata),
        generation_digest="",
        warm_source_site_signature=_warm_tensor_signature(union_tensor),
        _sampler_identity=id(lifetime.sampler),
        _construction_lifetime=lifetime,
        _construction_lifetime_identity=id(lifetime),
        _seal=_BUNDLE_SEAL,
    )
    result = _replace_bundle_generation(provisional, _bundle_digest(provisional))
    lifetime.bindings.extend(binding_tuple)
    lifetime.bundle_identity = id(result)
    lifetime.bundle_generation_digest = result.generation_digest
    lifetime.phase = "materialized"
    lifetime.assert_retained()
    if lifetime.device.type == "cpu":
        certify_paper_kinetic_union_local_spatial_bundle_cold_current(result)
    else:
        # The caller registered the outer bundle-materialization epoch before
        # entering this function.  Keep both CPU predecessors and device
        # destinations rooted until that exact receipt is consumed; a device
        # readback here would add a hidden synchronization and duplicate map.
        result.assert_accelerator_transfer_pending()
    return result


def prepare_paper_kinetic_union_local_spatial_bundle(
    sampler: PaperKineticRowRaggedSampler,
    *,
    track_ids: Sequence[int],
    device: torch.device | str,
    construction_lifetime: (
        PaperKineticUnionLocalSpatialBundleConstructionLifetime | None
    ) = None,
) -> PaperKineticUnionLocalSpatialBundle:
    """Legacy one-call CPU preparation; accelerators require two-phase use."""

    lifetime = construction_lifetime
    if lifetime is None:
        lifetime = prepare_paper_kinetic_union_local_spatial_bundle_construction_lifetime(
            sampler,
            track_ids=track_ids,
            device=device,
        )
        if lifetime.device.type != "cpu":
            raise RuntimeError(
                "accelerator union-local preparation requires a caller-retained "
                "construction lifetime"
            )
    else:
        lifetime.assert_retained()
        if (
            lifetime.sampler is not sampler
            or lifetime.track_ids
            != tuple(sorted(int(track_id) for track_id in track_ids))
            or lifetime.device != torch.device(device)
        ):
            raise ValueError("union-local construction lifetime is foreign")
    result = materialize_paper_kinetic_union_local_spatial_bundle(lifetime)
    if lifetime.device.type == "cpu":
        lifetime.release_transfer_predecessors_after_completion_fence()
    return result


@dataclass(frozen=True)
class PaperKineticActiveNativeBlockWork:
    """O(active native blocks) request work, never O(samples)."""

    native_block_generation_digest: str
    sample_chunk_count: int
    sample_count: int


@dataclass(frozen=True)
class PaperKineticUnionLocalRequestWork:
    """Cold exact-dispatch manifest for one coordinator request."""

    bundle: PaperKineticUnionLocalSpatialBundle = field(repr=False)
    sampler: PaperKineticRowRaggedSampler = field(repr=False)
    request: PaperRaggedMaterialBarRequest = field(repr=False)
    maximum_samples_per_launch: int
    active_blocks: tuple[PaperKineticActiveNativeBlockWork, ...]
    total_sample_count: int
    generation_digest: str
    _bundle_identity: int = field(repr=False)
    _sampler_identity: int = field(repr=False)
    _request_identity: int = field(repr=False)
    persistent_sample_tensor_bytes: int = 0
    persistent_target_tensor_bytes: int = 0
    persistent_interpolation_weight_tensor_bytes: int = 0
    retained_sample_partition_records: int = 0
    _seal: object = field(default=None, repr=False)

    @property
    def active_native_block_count(self) -> int:
        return len(self.active_blocks)

    def assert_warm_layout(self) -> None:
        if self._seal is not _WORK_SEAL:
            raise ValueError("union-local request work was not sealed by its planner")
        if (
            id(self.bundle) != self._bundle_identity
            or id(self.sampler) != self._sampler_identity
            or id(self.request) != self._request_identity
            or self.sampler is not self.bundle.sampler
            or self.request.world_token is not self.bundle
            or self.maximum_samples_per_launch < 1
            or not self.active_blocks
            or self.total_sample_count != self.request.block.track_count
            * (self.request.local_sample_end - self.request.local_sample_start)
            or self.persistent_sample_tensor_bytes != 0
            or self.persistent_target_tensor_bytes != 0
            or self.persistent_interpolation_weight_tensor_bytes != 0
            or self.retained_sample_partition_records != 0
            or any(
                block.sample_chunk_count < 1 or block.sample_count < 1
                for block in self.active_blocks
            )
            or sum(block.sample_count for block in self.active_blocks) != self.total_sample_count
        ):
            raise ValueError("union-local request work metadata/memory contract changed")
        self.bundle.assert_warm_layout()
        self.request.assert_current()
        expected_digests = tuple(
            binding.native_block_generation_digest for binding in self.bundle.native_blocks
        )
        active_digests = tuple(
            block.native_block_generation_digest for block in self.active_blocks
        )
        if active_digests != tuple(digest for digest in expected_digests if digest in set(active_digests)):
            raise ValueError("union-local active blocks are not in canonical bundle order")
        if len(set(active_digests)) != len(active_digests):
            raise ValueError("union-local request work contains duplicate active blocks")


def prepare_paper_kinetic_union_local_request_work(
    bundle: PaperKineticUnionLocalSpatialBundle,
    request: PaperRaggedMaterialBarRequest,
    *,
    maximum_samples_per_launch: int,
) -> PaperKineticUnionLocalRequestWork:
    """Cold-dispatch counts without retaining frame/sample tensors."""

    _require_positive_int(maximum_samples_per_launch, name="maximum_samples_per_launch")
    bundle.assert_warm_layout()
    request.assert_current()
    if request.world_token is not bundle:
        raise ValueError("union-local request belongs to a different spatial bundle")
    if request.view_index != bundle.view_index:
        raise ValueError("union-local request and bundle belong to different views")
    request_tracks = tuple(int(value) for value in request.staged.pixel_indices.tolist())
    if tuple(sorted(request_tracks)) != bundle.track_ids:
        raise ValueError("union-local request tracks differ from the cold spatial bundle")
    times = tuple(float(value) for value in request.staged.sample_times.tolist())
    if not times or not all(math.isfinite(time) for time in times):
        raise ValueError("union-local request times must be finite and nonempty")
    row_by_identity = {row.row_identity: row for row in bundle.sampler.rows}
    first_row_by_track = {
        track_id: next(row for row in bundle.sampler.rows if row.track_id == track_id)
        for track_id in request_tracks
    }
    sample_counts: dict[str, int] = {}
    for track_id in request_tracks:
        first_row = first_row_by_track[track_id]
        for time in times:
            chart_index = dispatch_prevalidated_kinetic_chart_index(
                first_row.program,
                Fraction.from_float(time),
                expected_generation_digest=first_row.program_generation_digest,
            )
            try:
                row = row_by_identity[(track_id, chart_index)]
            except KeyError as error:
                raise ValueError("union-local dispatch selected an unbound kinetic row") from error
            bundle.binding_for_digest(row.native_block_generation_digest)
            sample_counts[row.native_block_generation_digest] = (
                sample_counts.get(row.native_block_generation_digest, 0) + 1
            )
    active_blocks = tuple(
        PaperKineticActiveNativeBlockWork(
            native_block_generation_digest=binding.native_block_generation_digest,
            sample_chunk_count=(
                sample_counts[binding.native_block_generation_digest]
                + maximum_samples_per_launch
                - 1
            )
            // maximum_samples_per_launch,
            sample_count=sample_counts[binding.native_block_generation_digest],
        )
        for binding in bundle.native_blocks
        if binding.native_block_generation_digest in sample_counts
    )
    total_samples = len(request_tracks) * len(times)
    generation = _digest_parts(
        UNION_LOCAL_PROVENANCE,
        "request-work",
        bundle.generation_digest,
        request.request_generation_id,
        request.loss_normalization_id,
        maximum_samples_per_launch,
        tuple(
            (
                block.native_block_generation_digest,
                block.sample_chunk_count,
                block.sample_count,
            )
            for block in active_blocks
        ),
        total_samples,
    )
    result = PaperKineticUnionLocalRequestWork(
        bundle=bundle,
        sampler=bundle.sampler,
        request=request,
        maximum_samples_per_launch=maximum_samples_per_launch,
        active_blocks=active_blocks,
        total_sample_count=total_samples,
        generation_digest=generation,
        _bundle_identity=id(bundle),
        _sampler_identity=id(bundle.sampler),
        _request_identity=id(request),
        _seal=_WORK_SEAL,
    )
    result.assert_warm_layout()
    return result


@dataclass(frozen=True)
class PaperKineticNativeBlockVJPContribution:
    """One native compact bar sealed to one request/work entry."""

    assembly_generation_id: str
    request_generation_id: str
    work_generation_digest: str
    native_block_generation_digest: str
    reduced_sample_chunk_count: int
    reduced_sample_count: int
    native_node_vjp_invocation_count: int
    loss_f32: torch.Tensor = field(repr=False)
    grad_compact_site_rgba_f32: torch.Tensor = field(repr=False)
    native_vjp_result: Any = field(repr=False)
    warm_tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    native_vjp_identity: int
    _seal: object = field(default=None, repr=False)

    def assert_warm_layout(
        self,
        assembly: PaperKineticUnionLocalBarAssembly,
    ) -> None:
        if self._seal is not _CONTRIBUTION_SEAL:
            raise ValueError("union-local contribution was not sealed by its factory")
        if (
            self.assembly_generation_id != assembly.generation_id
            or self.request_generation_id != assembly.work.request.request_generation_id
            or self.work_generation_digest != assembly.work.generation_digest
            or self.native_node_vjp_invocation_count != 1
            or id(self.native_vjp_result) != self.native_vjp_identity
        ):
            raise ValueError("union-local contribution has foreign request/VJP provenance")
        _assert_native_vjp_current(self.native_vjp_result)
        tensors = (self.loss_f32, self.grad_compact_site_rgba_f32)
        if tuple(_warm_tensor_signature(tensor) for tensor in tensors) != self.warm_tensor_signatures:
            raise ValueError("union-local contribution tensor identity/layout/version changed")
        binding = assembly.work.bundle.binding_for_digest(
            self.native_block_generation_digest
        )
        device = assembly.grad_union_site_rgba_f32.device
        _require_warm_tensor(
            self.loss_f32,
            name="native_block_loss_f32",
            device=device,
            dtype=torch.float32,
            shape=(1,),
        )
        _require_warm_tensor(
            self.grad_compact_site_rgba_f32,
            name="native_block_grad_compact_site_rgba_f32",
            device=device,
            dtype=torch.float32,
            shape=(binding.compact_site_count, 4),
        )


@dataclass
class PaperKineticUnionLocalBarAssembly:
    """One request-local union accumulator and exact native-block ledger."""

    work: PaperKineticUnionLocalRequestWork
    grad_union_site_rgba_f32: torch.Tensor
    loss_f32: torch.Tensor
    generation_id: str
    state_tensor_signatures: tuple[tuple[object, ...], ...]
    next_active_block_index: int = 0
    consumed_native_block_count: int = 0
    consumed_sample_chunk_count: int = 0
    consumed_sample_count: int = 0
    union_bar_zero_count: int = 1
    loss_zero_count: int = 1
    finalized: bool = False
    _seal: object = field(default=None, repr=False)

    @property
    def accounting(self) -> dict[str, int | bool | str]:
        bundle = self.work.bundle
        return {
            "native_block_count": bundle.native_block_count,
            "active_native_block_count": self.work.active_native_block_count,
            "consumed_native_block_count": self.consumed_native_block_count,
            "expected_sample_chunk_count": sum(
                block.sample_chunk_count for block in self.work.active_blocks
            ),
            "consumed_sample_chunk_count": self.consumed_sample_chunk_count,
            "expected_sample_count": self.work.total_sample_count,
            "consumed_sample_count": self.consumed_sample_count,
            "union_site_count": bundle.union_site_count,
            "summed_native_compact_site_count": sum(
                binding.compact_site_count for binding in bundle.native_blocks
            ),
            "persistent_mapping_tensor_bytes": bundle.persistent_mapping_tensor_bytes,
            "union_material_bar_tensor_bytes": _tensor_bytes(
                (self.grad_union_site_rgba_f32,)
            ),
            "loss_tensor_bytes": _tensor_bytes((self.loss_f32,)),
            "per_request_global_material_bar_bytes": 0,
            "adapter_allocated_union_material_bar_bytes": 0,
            "adapter_allocated_global_material_bar_bytes": 0,
            "union_bar_zero_count": self.union_bar_zero_count,
            "loss_zero_count": self.loss_zero_count,
            "native_vjp_result_count": self.consumed_native_block_count,
            "cross_native_duplicate_sites_sum_with_index_add": True,
            "persistent_frame_tensor_bytes": 0,
            "persistent_sample_tensor_bytes": 0,
            "persistent_target_tensor_bytes": 0,
            "persistent_prediction_tensor_bytes": 0,
            "retained_sample_partition_records": 0,
            "global_common_temporal_refinement_used": False,
            "warm_validation_device_to_host_syncs": 0,
            "warm_validation_tensor_allocations": 0,
            "allocator_storage_bytes_measured": False,
            "allocator_peak_measured": False,
            "proof_boundary": PROOF_BOUNDARY,
        }


@torch.no_grad()
def begin_paper_kinetic_union_local_bar_assembly(
    work: PaperKineticUnionLocalRequestWork,
    *,
    grad_union_site_rgba_f32: torch.Tensor,
    loss_f32: torch.Tensor,
) -> PaperKineticUnionLocalBarAssembly:
    """Zero the caller-owned union bar/loss once and open the exact ledger."""

    if not isinstance(work, PaperKineticUnionLocalRequestWork):
        raise TypeError("union-local assembly requires PaperKineticUnionLocalRequestWork")
    work.assert_warm_layout()
    device = work.bundle.device
    _require_warm_tensor(
        grad_union_site_rgba_f32,
        name="grad_union_site_rgba_f32",
        device=device,
        dtype=torch.float32,
        shape=(work.bundle.union_site_count, 4),
    )
    _require_warm_tensor(
        loss_f32,
        name="union_local_loss_f32",
        device=device,
        dtype=torch.float32,
        shape=(1,),
    )
    if _same_storage(grad_union_site_rgba_f32, loss_f32):
        raise ValueError("union-local gradient and loss buffers must not alias")
    grad_union_site_rgba_f32.zero_()
    loss_f32.zero_()
    generation = (
        f"{work.generation_digest}:assembly:"
        f"{id(grad_union_site_rgba_f32)}:{id(loss_f32)}"
    )
    result = PaperKineticUnionLocalBarAssembly(
        work=work,
        grad_union_site_rgba_f32=grad_union_site_rgba_f32,
        loss_f32=loss_f32,
        generation_id=generation,
        state_tensor_signatures=tuple(
            _warm_tensor_signature(tensor)
            for tensor in (grad_union_site_rgba_f32, loss_f32)
        ),
        _seal=_ASSEMBLY_SEAL,
    )
    _assert_assembly_warm_layout(result)
    return result


def seal_paper_kinetic_native_block_vjp_contribution(
    assembly: PaperKineticUnionLocalBarAssembly,
    *,
    native_vjp_result: Any,
    loss_f32: torch.Tensor,
    reduced_sample_chunk_count: int,
    reduced_sample_count: int,
) -> PaperKineticNativeBlockVJPContribution:
    """Seal one compact-only native VJP result against its expected work."""

    _assert_assembly_warm_layout(assembly)
    if assembly.finalized:
        raise ValueError("union-local assembly was already finalized")
    native_digest, compact_bar = _native_vjp_identity(native_vjp_result)
    expected = _expected_next_work(assembly)
    if native_digest != expected.native_block_generation_digest:
        raise ValueError("native contribution is duplicate, out of order, or foreign")
    if (
        reduced_sample_chunk_count != expected.sample_chunk_count
        or reduced_sample_count != expected.sample_count
    ):
        raise ValueError("native contribution did not reduce every expected sample chunk")
    if not isinstance(loss_f32, torch.Tensor):
        raise TypeError("native block loss must be a tensor")
    binding = assembly.work.bundle.binding_for_digest(native_digest)
    _require_warm_tensor(
        loss_f32,
        name="native_block_loss_f32",
        device=assembly.grad_union_site_rgba_f32.device,
        dtype=torch.float32,
        shape=(1,),
    )
    _require_warm_tensor(
        compact_bar,
        name="native_vjp_compact_material_bar",
        device=assembly.grad_union_site_rgba_f32.device,
        dtype=torch.float32,
        shape=(binding.compact_site_count, 4),
    )
    if _same_storage(compact_bar, assembly.grad_union_site_rgba_f32) or _same_storage(
        loss_f32, assembly.loss_f32
    ):
        raise ValueError("native contribution must not alias the union accumulators")
    result = PaperKineticNativeBlockVJPContribution(
        assembly_generation_id=assembly.generation_id,
        request_generation_id=assembly.work.request.request_generation_id,
        work_generation_digest=assembly.work.generation_digest,
        native_block_generation_digest=native_digest,
        reduced_sample_chunk_count=reduced_sample_chunk_count,
        reduced_sample_count=reduced_sample_count,
        native_node_vjp_invocation_count=1,
        loss_f32=loss_f32,
        grad_compact_site_rgba_f32=compact_bar,
        native_vjp_result=native_vjp_result,
        warm_tensor_signatures=tuple(
            _warm_tensor_signature(tensor) for tensor in (loss_f32, compact_bar)
        ),
        native_vjp_identity=id(native_vjp_result),
        _seal=_CONTRIBUTION_SEAL,
    )
    result.assert_warm_layout(assembly)
    return result


@torch.no_grad()
def consume_paper_kinetic_union_local_native_contribution(
    assembly: PaperKineticUnionLocalBarAssembly,
    contribution: PaperKineticNativeBlockVJPContribution,
) -> None:
    """Merge one expected compact bar; repeated union rows sum exactly."""

    _assert_assembly_warm_layout(assembly)
    if assembly.finalized:
        raise ValueError("union-local assembly was already finalized")
    if not isinstance(contribution, PaperKineticNativeBlockVJPContribution):
        raise TypeError("union-local assembly requires a sealed native contribution")
    contribution.assert_warm_layout(assembly)
    expected = _expected_next_work(assembly)
    if contribution.native_block_generation_digest != expected.native_block_generation_digest:
        raise ValueError("native contribution is duplicate, out of order, or foreign")
    binding = assembly.work.bundle.binding_for_digest(
        contribution.native_block_generation_digest
    )
    assembly.grad_union_site_rgba_f32.index_add_(
        0,
        binding.compact_to_union_i64,
        contribution.grad_compact_site_rgba_f32,
    )
    assembly.loss_f32.add_(contribution.loss_f32)
    assembly.next_active_block_index += 1
    assembly.consumed_native_block_count += 1
    assembly.consumed_sample_chunk_count += contribution.reduced_sample_chunk_count
    assembly.consumed_sample_count += contribution.reduced_sample_count
    assembly.state_tensor_signatures = tuple(
        _warm_tensor_signature(tensor)
        for tensor in (assembly.grad_union_site_rgba_f32, assembly.loss_f32)
    )
    _assert_assembly_warm_layout(assembly)


def finalize_paper_kinetic_union_local_bar_assembly(
    assembly: PaperKineticUnionLocalBarAssembly,
) -> PaperRaggedCompactMaterialBarResult:
    """Fail on missing work, then seal the union bar directly for coordination."""

    _assert_assembly_warm_layout(assembly)
    if assembly.finalized:
        raise ValueError("union-local assembly was already finalized")
    if assembly.next_active_block_index != assembly.work.active_native_block_count:
        raise ValueError("union-local assembly cannot finalize with missing native contributions")
    expected_chunks = sum(block.sample_chunk_count for block in assembly.work.active_blocks)
    if (
        assembly.consumed_native_block_count != assembly.work.active_native_block_count
        or assembly.consumed_sample_chunk_count != expected_chunks
        or assembly.consumed_sample_count != assembly.work.total_sample_count
    ):
        raise ValueError("union-local assembly coverage accounting is incomplete")
    result = seal_paper_ragged_compact_material_bar_result(
        assembly.work.request,
        loss_f32=assembly.loss_f32,
        grad_compact_site_rgba_f32=assembly.grad_union_site_rgba_f32,
    )
    assembly.finalized = True
    return result


def _assert_assembly_warm_layout(assembly: PaperKineticUnionLocalBarAssembly) -> None:
    if not isinstance(assembly, PaperKineticUnionLocalBarAssembly) or assembly._seal is not _ASSEMBLY_SEAL:
        raise ValueError("union-local assembly was not sealed by its opener")
    assembly.work.assert_warm_layout()
    tensors = (assembly.grad_union_site_rgba_f32, assembly.loss_f32)
    if tuple(_warm_tensor_signature(tensor) for tensor in tensors) != assembly.state_tensor_signatures:
        raise ValueError("union-local accumulator identity/layout/version changed outside assembly")
    device = assembly.work.bundle.device
    _require_warm_tensor(
        assembly.grad_union_site_rgba_f32,
        name="grad_union_site_rgba_f32",
        device=device,
        dtype=torch.float32,
        shape=(assembly.work.bundle.union_site_count, 4),
    )
    _require_warm_tensor(
        assembly.loss_f32,
        name="union_local_loss_f32",
        device=device,
        dtype=torch.float32,
        shape=(1,),
    )
    if (
        assembly.next_active_block_index < 0
        or assembly.next_active_block_index > assembly.work.active_native_block_count
        or assembly.consumed_native_block_count != assembly.next_active_block_index
        or assembly.union_bar_zero_count != 1
        or assembly.loss_zero_count != 1
    ):
        raise ValueError("union-local assembly coverage/zeroing state changed")


def _expected_next_work(
    assembly: PaperKineticUnionLocalBarAssembly,
) -> PaperKineticActiveNativeBlockWork:
    if assembly.next_active_block_index >= assembly.work.active_native_block_count:
        raise ValueError("native contribution is duplicate or exceeds expected active blocks")
    return assembly.work.active_blocks[assembly.next_active_block_index]


def _native_vjp_identity(native_vjp_result: Any) -> tuple[str, torch.Tensor]:
    _assert_native_vjp_current(native_vjp_result)
    world = getattr(native_vjp_result, "world", None)
    runtime = getattr(world, "runtime", None)
    payload = getattr(runtime, "payload", None)
    block = getattr(payload, "block", None)
    digest = getattr(block, "generation_digest", None)
    compact_bar = getattr(native_vjp_result, "grad_compact_site_rgba_f32", None)
    global_bar = getattr(native_vjp_result, "grad_global_site_rgba_f32", None)
    if not isinstance(digest, str) or not digest.strip() or not isinstance(compact_bar, torch.Tensor):
        raise TypeError("native VJP result lacks sealed equal-rank block/bar provenance")
    if global_bar is not None:
        raise ValueError("union-local assembly requires compact-only VJP; per-block global bar is forbidden")
    source_ids = getattr(runtime, "source_site_ids_i64", None)
    if not isinstance(source_ids, torch.Tensor):
        raise TypeError("native VJP runtime lacks its compact source-site map")
    return digest, compact_bar


def _assert_native_vjp_current(native_vjp_result: Any) -> None:
    assert_current = getattr(native_vjp_result, "assert_current", None)
    if not callable(assert_current):
        assert_current = getattr(native_vjp_result, "assert_warm_layout", None)
    if not callable(assert_current):
        raise TypeError("native VJP result must expose warm provenance validation")
    assert_current()


def _replace_bundle_generation(
    bundle: PaperKineticUnionLocalSpatialBundle,
    generation_digest: str,
) -> PaperKineticUnionLocalSpatialBundle:
    return PaperKineticUnionLocalSpatialBundle(
        sampler=bundle.sampler,
        view_index=bundle.view_index,
        track_ids=bundle.track_ids,
        global_site_count=bundle.global_site_count,
        union_source_site_ids=bundle.union_source_site_ids,
        source_site_ids_i64=bundle.source_site_ids_i64,
        native_blocks=bundle.native_blocks,
        descriptor_canonical_metadata_bytes=bundle.descriptor_canonical_metadata_bytes,
        generation_digest=generation_digest,
        warm_source_site_signature=bundle.warm_source_site_signature,
        _sampler_identity=bundle._sampler_identity,
        _construction_lifetime=bundle._construction_lifetime,
        _construction_lifetime_identity=bundle._construction_lifetime_identity,
        _seal=_BUNDLE_SEAL,
    )


def _bundle_digest(bundle: PaperKineticUnionLocalSpatialBundle) -> str:
    return _digest_parts(
        UNION_LOCAL_PROVENANCE,
        bundle.sampler.generation_digest,
        bundle.view_index,
        bundle.track_ids,
        bundle.global_site_count,
        bundle.union_source_site_ids,
        tuple(
            (
                binding.native_block_generation_digest,
                binding.compact_source_site_ids,
                binding.mapping_generation_digest,
            )
            for binding in bundle.native_blocks
        ),
        bundle.descriptor_canonical_metadata_bytes,
        False,
        0,
        0,
        0,
        0,
    )


def _mapping_digest(
    *,
    sampler_generation_digest: str,
    native_block_generation_digest: str,
    compact_source_site_ids: tuple[int, ...],
    union_source_site_ids: tuple[int, ...],
    compact_to_union: tuple[int, ...],
) -> str:
    return _digest_parts(
        UNION_LOCAL_PROVENANCE,
        "compact-to-union",
        sampler_generation_digest,
        native_block_generation_digest,
        compact_source_site_ids,
        union_source_site_ids,
        compact_to_union,
    )


def _cpu_int_tuple_without_transfer(tensor: torch.Tensor) -> tuple[int, ...]:
    """Convert an already-CPU tensor; never hide a device-to-host copy."""

    if tensor.device.type != "cpu":
        raise RuntimeError("cold union-map conversion requires an explicit transfer receipt")
    return tuple(int(value) for value in tensor.detach().tolist())


def _materialize_and_validate_cold_receipt_contents(
    bundle: PaperKineticUnionLocalSpatialBundle,
    lifetime: PaperKineticUnionLocalColdReceiptLifetime,
) -> None:
    """Validate one rooted CPU destination at a time, then drop its payload."""

    lifetime.assert_retained()
    if lifetime.phase != "installed":
        raise ValueError("union-local cold receipt was already consumed")
    lifetime.phase = "transferring"
    lifetime.validated_content_digest = _digest_parts(
        UNION_LOCAL_PROVENANCE,
        "cold-device-to-host-receipt",
        bundle.generation_digest,
        "begin",
    )
    for source_index, (source, expected) in enumerate(
        zip(
            lifetime.source_tensors_i64,
            lifetime.expected_int_tuples,
            strict=True,
        )
    ):
        lifetime.current_transfer_source = source.detach()
        lifetime.current_source_index = source_index
        # The caller-visible lifetime owns the device source before ``to``.
        # If ``to`` raises before returning, no destination exists to publish;
        # the source remains rooted.  If it returns, assign that result here
        # before requesting any contiguous view or Python conversion.
        lifetime.current_raw_device_to_host_result = lifetime.current_transfer_source.to(
            device="cpu",
            dtype=torch.int64,
            non_blocking=False,
        )
        lifetime.current_cpu_destination_tensor = (
            lifetime.current_raw_device_to_host_result.contiguous()
        )
        converted = tuple(
            int(value)
            for value in lifetime.current_cpu_destination_tensor.tolist()
        )
        lifetime.current_converted_int_tuple = converted
        _validate_one_cold_union_map_receipt(
            bundle,
            source_index=source_index,
            observed=converted,
            expected=expected,
        )
        lifetime.validated_content_digest = _digest_parts(
            lifetime.validated_content_digest,
            source_index,
            converted,
        )
        lifetime.validated_source_count += 1
        # ``non_blocking=False`` requests a synchronous D2H copy, and
        # ``tolist()`` cannot return until the CPU destination is readable.
        # Exact comparison above is therefore the per-map completion boundary.
        lifetime.current_transfer_source = None
        lifetime.current_raw_device_to_host_result = None
        lifetime.current_cpu_destination_tensor = None
        lifetime.current_converted_int_tuple = None
        lifetime.current_source_index = None
        del converted
    lifetime.assert_retained()


def _validate_one_cold_union_map_receipt(
    bundle: PaperKineticUnionLocalSpatialBundle,
    *,
    source_index: int,
    observed: tuple[int, ...],
    expected: tuple[int, ...],
) -> None:
    if observed != expected:
        if source_index == 0:
            raise ValueError("union-local source union content changed")
        raise ValueError("union-local compact-to-union mapping content changed")
    if source_index == 0:
        return
    binding = bundle.native_blocks[source_index - 1]
    if binding.mapping_generation_digest != _mapping_digest(
        sampler_generation_digest=bundle.sampler.generation_digest,
        native_block_generation_digest=binding.native_block_generation_digest,
        compact_source_site_ids=binding.compact_source_site_ids,
        union_source_site_ids=bundle.union_source_site_ids,
        compact_to_union=expected,
    ):
        raise ValueError("union-local compact mapping generation changed")


def _copy_or_alias_i64_with_construction_lifetime(
    lifetime: PaperKineticUnionLocalSpatialBundleConstructionLifetime,
    source: torch.Tensor,
) -> torch.Tensor:
    """Publish each transfer result before requesting a contiguous view."""

    if (
        source.device == lifetime.device
        and source.dtype == torch.int64
        and source.is_contiguous()
        and not source.requires_grad
    ):
        return source
    detached = source.detach()
    lifetime.transfer_intermediates.append(detached)
    transferred = detached.to(device=lifetime.device, dtype=torch.int64)
    lifetime.transfer_intermediates.append(transferred)
    contiguous = transferred.contiguous()
    if contiguous is not transferred:
        lifetime.transfer_intermediates.append(contiguous)
    return contiguous


def _warm_tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        tensor.untyped_storage().data_ptr(),
        tensor._version,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.dtype,
        tensor.device,
    )


def _require_warm_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device != device
        or tensor.dtype != dtype
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
        or tensor.requires_grad
    ):
        raise ValueError(f"{name} has the wrong detached contiguous device/dtype/shape")


def _same_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


def _tensor_bytes(tensors: tuple[torch.Tensor, ...]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


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
    "PROOF_BOUNDARY",
    "PaperKineticActiveNativeBlockWork",
    "PaperKineticNativeBlockVJPContribution",
    "PaperKineticUnionLocalColdReceiptLifetime",
    "PaperKineticUnionLocalBarAssembly",
    "PaperKineticUnionLocalMemoryReport",
    "PaperKineticUnionLocalNativeBlockBinding",
    "PaperKineticUnionLocalRequestWork",
    "PaperKineticUnionLocalSpatialBundle",
    "PaperKineticUnionLocalSpatialBundleConstructionLifetime",
    "UNION_LOCAL_PROVENANCE",
    "begin_paper_kinetic_union_local_bar_assembly",
    "certify_paper_kinetic_union_local_spatial_bundle_cold_current",
    "consume_paper_kinetic_union_local_native_contribution",
    "finalize_paper_kinetic_union_local_bar_assembly",
    "install_paper_kinetic_union_local_cold_receipt_lifetime",
    "materialize_paper_kinetic_union_local_spatial_bundle",
    "prepare_paper_kinetic_union_local_request_work",
    "prepare_paper_kinetic_union_local_spatial_bundle",
    "prepare_paper_kinetic_union_local_spatial_bundle_construction_lifetime",
    "seal_paper_kinetic_native_block_vjp_contribution",
]
