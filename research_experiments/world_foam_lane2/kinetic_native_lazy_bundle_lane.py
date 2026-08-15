"""Cold preparation of one droppable native lane from one lazy paper bundle.

The lazy program provider deliberately yields only one bounded spatial bundle
at a time.  This module is the matching native cold boundary: it materializes
exactly one runtime for every equal-rank block in that bundle, preserves the
compiler's canonical bucket/block order, and binds one
``KineticNativeMaterialStepExecutor``.  The returned object owns references to
the supplied bundle, those runtimes, that executor, and the bounded construction
lifetime that rooted their transfer predecessors.  It does not retain the
provider separately and can therefore be dropped or evicted as one unit.

Logical resident bytes are computed over unique reachable tensor *objects*.
They are exact for that definition and do not double-count aliases reached
through the bundle, runtimes, and executor.  They are not allocator-storage,
Python-heap, or peak-memory measurements.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import torch
from kinetic_native_equal_rank_lowering import (
    iter_materialize_kinetic_native_equal_rank_blocks,
)
from kinetic_native_equal_rank_runtime_adapter import (
    KineticNativeEqualRankRuntimeBlock,
    KineticNativeEqualRankRuntimeConstructionLifetime,
    prepare_kinetic_native_equal_rank_runtime_construction_lifetime,
    prepare_kinetic_native_equal_rank_runtime_block,
)
from kinetic_native_material_step_executor import (
    KineticNativeMaterialStepExecutor,
    prepare_kinetic_native_material_step_executor,
)
from paper_kinetic_lazy_program_bundles import (
    PaperKineticLazyProgramBundle,
    PaperKineticLazyProgramBundleProvider,
)

LANE_PROVENANCE = "paper-kinetic-native-lazy-bundle-lane-v2"
LANE_STATUS = "native_ops_bound/source_runtime_unverified"
_LANE_SEAL = object()
_LANE_CONSTRUCTION_LIFETIME_SEAL = object()


@dataclass
class PaperKineticNativeLazyBundleLaneConstructionLifetime:
    """Caller-retained roots installed before any runtime device transfer."""

    bundle: PaperKineticLazyProgramBundle = field(repr=False)
    provider: PaperKineticLazyProgramBundleProvider | None = field(repr=False)
    native_ops: Any = field(repr=False)
    device: torch.device
    backend_provenance: str
    maximum_resident_logical_tensor_bytes: int | None
    preflight_logical_tensor_bytes: int
    lowering: Any = field(repr=False)
    sources: tuple[Any, ...] = field(repr=False)
    expected_blocks: tuple[Any, ...] = field(repr=False)
    expected_block_generation_digests: tuple[str, ...]
    payloads: list[Any] = field(default_factory=list, repr=False)
    runtime_lifetimes: list[
        KineticNativeEqualRankRuntimeConstructionLifetime
    ] = field(default_factory=list, repr=False)
    runtimes: list[KineticNativeEqualRankRuntimeBlock] = field(
        default_factory=list,
        repr=False,
    )
    executor: KineticNativeMaterialStepExecutor | None = field(
        default=None,
        repr=False,
    )
    current_payload: Any = field(default=None, repr=False)
    current_runtime_lifetime: (
        KineticNativeEqualRankRuntimeConstructionLifetime | None
    ) = field(default=None, repr=False)
    phase: str = "installed"
    provider_identity: int = 0
    provider_generation_digest: str = ""
    bundle_generation_digest: str = ""
    _bundle_identity: int = field(default=0, repr=False)
    _native_ops_identity: int = field(default=0, repr=False)
    _seal: object = field(default=None, repr=False)

    def assert_retained(self) -> None:
        if (
            self._seal is not _LANE_CONSTRUCTION_LIFETIME_SEAL
            or self.phase not in {"installed", "materializing", "transferred"}
            or id(self.bundle) != self._bundle_identity
            or self.bundle.generation_digest != self.bundle_generation_digest
            or id(self.native_ops) != self._native_ops_identity
            or not self.backend_provenance.strip()
            or self.provider_identity < 1
            or not self.provider_generation_digest.strip()
            or len(self.payloads) > len(self.expected_blocks)
            or len(self.runtime_lifetimes) > len(self.expected_blocks)
            or len(self.runtimes) > len(self.runtime_lifetimes)
            or any(
                payload.block is not block
                for payload, block in zip(
                    self.payloads,
                    self.expected_blocks,
                    strict=False,
                )
            )
            or any(
                runtime.payload is not payload
                for runtime, payload in zip(
                    self.runtimes,
                    self.payloads,
                    strict=False,
                )
            )
            or (self.phase == "transferred")
            != (self.provider is None and self.executor is not None)
        ):
            raise ValueError("native lazy lane construction lifetime changed")
        if self.provider is not None:
            if (
                id(self.provider) != self.provider_identity
                or self.provider.generation_digest
                != self.provider_generation_digest
            ):
                raise ValueError("native lazy lane construction provider changed")
        for lifetime in self.runtime_lifetimes:
            lifetime.assert_retained()


@dataclass(frozen=True)
class PaperKineticNativeLazyBundleLaneMemory:
    """Exact logical tensor-object bytes for one resident lane."""

    requested_observation_count: int
    native_runtime_count: int
    bundle_logical_tensor_object_count: int
    bundle_logical_tensor_bytes: int
    runtime_additional_logical_tensor_object_count: int
    runtime_additional_logical_tensor_bytes: int
    executor_additional_logical_tensor_object_count: int
    executor_additional_logical_tensor_bytes: int
    resident_logical_tensor_object_count: int
    resident_logical_tensor_bytes: int
    persistent_frame_tensor_bytes: int
    persistent_sample_tensor_bytes: int
    persistent_target_tensor_bytes: int
    persistent_prediction_tensor_bytes: int
    intended_maximum_live_native_lane_count: int = 1
    one_live_lane_enforced_by_lane_object: bool = False
    retained_provider_count: int = 0
    runtime_count_equals_sampler_lowering_block_count: bool = True
    canonical_native_block_order_preserved: bool = True
    requested_observation_count_affects_resident_bytes: bool = False
    droppable_as_one_lane_unit: bool = True
    logical_byte_definition: str = "unique_reachable_tensor_objects_numel_times_element_size"
    allocator_storage_bytes_measured: bool = False
    allocator_peak_measured: bool = False
    python_object_bytes_measured: bool = False


@dataclass(frozen=True)
class PaperKineticNativeLazyBundleLane:
    """One bundle-scoped native residency unit, suitable for an outer LRU."""

    bundle: PaperKineticLazyProgramBundle = field(repr=False)
    runtimes: tuple[KineticNativeEqualRankRuntimeBlock, ...] = field(repr=False)
    executor: KineticNativeMaterialStepExecutor = field(repr=False)
    device: torch.device
    backend_provenance: str
    provider_identity: int
    provider_generation_digest: str
    bundle_identity: int
    bundle_generation_digest: str
    runtime_identities: tuple[int, ...]
    runtime_generation_ids: tuple[str, ...]
    canonical_native_block_generation_digests: tuple[str, ...]
    executor_identity: int
    executor_generation_id: str
    bundle_logical_tensor_object_count: int
    bundle_logical_tensor_bytes: int
    runtime_additional_logical_tensor_object_count: int
    runtime_additional_logical_tensor_bytes: int
    executor_additional_logical_tensor_object_count: int
    executor_additional_logical_tensor_bytes: int
    resident_logical_tensor_object_count: int
    resident_logical_tensor_bytes: int
    generation_digest: str
    _sealed_generation_digest: str = field(repr=False)
    _construction_lifetime: (
        PaperKineticNativeLazyBundleLaneConstructionLifetime
    ) = field(repr=False)
    _construction_lifetime_identity: int = field(repr=False)
    provenance: str = LANE_PROVENANCE
    runtime_status: str = LANE_STATUS
    retained_provider_count: int = 0
    persistent_frame_tensor_bytes: int = 0
    persistent_sample_tensor_bytes: int = 0
    persistent_target_tensor_bytes: int = 0
    persistent_prediction_tensor_bytes: int = 0
    native_runtime_verified: bool = False
    allocator_storage_bytes_measured: bool = False
    allocator_peak_measured: bool = False
    python_object_bytes_measured: bool = False
    _seal: object = field(default=None, repr=False)

    @property
    def native_runtime_count(self) -> int:
        return len(self.runtimes)

    @property
    def resident_tensor_bytes(self) -> int:
        """LRU-facing alias for exact resident logical tensor-object bytes."""

        return self.resident_logical_tensor_bytes

    @property
    def view_index(self) -> int:
        return self.bundle.view_index

    @property
    def bundle_index(self) -> int:
        return self.bundle.bundle_index

    def runtime_for_native_block_digest(
        self,
        native_block_generation_digest: str,
    ) -> KineticNativeEqualRankRuntimeBlock:
        matches = tuple(
            runtime
            for runtime in self.runtimes
            if runtime.payload.block.generation_digest
            == native_block_generation_digest
        )
        if len(matches) != 1:
            raise ValueError("native lazy lane has no unique runtime for block digest")
        return matches[0]

    def assert_warm_layout(self) -> None:
        """Validate lane-local identity/layout without retaining the provider."""

        if self._seal is not _LANE_SEAL:
            raise ValueError("native lazy bundle lane was not sealed by its preparer")
        expected_blocks = _canonical_native_blocks(self.bundle)
        expected_digests = tuple(block.generation_digest for block in expected_blocks)
        if (
            self.provenance != LANE_PROVENANCE
            or self.runtime_status != LANE_STATUS
            or not self.backend_provenance.strip()
            or self.device != self.bundle.spatial_bundle.device
            or self.provider_identity != self.bundle._provider_identity
            or self.provider_generation_digest
            != self.bundle.provider_generation_digest
            or id(self.bundle) != self.bundle_identity
            or self.bundle.generation_digest != self.bundle_generation_digest
            or tuple(id(runtime) for runtime in self.runtimes)
            != self.runtime_identities
            or tuple(runtime.generation_id for runtime in self.runtimes)
            != self.runtime_generation_ids
            or expected_digests
            != self.canonical_native_block_generation_digests
            or tuple(
                runtime.payload.block.generation_digest
                for runtime in self.runtimes
            )
            != expected_digests
            or any(
                runtime.payload.block is not block
                for runtime, block in zip(
                    self.runtimes,
                    expected_blocks,
                    strict=True,
                )
            )
            or id(self.executor) != self.executor_identity
            or self.executor.generation_id != self.executor_generation_id
            or tuple(id(binding.runtime) for binding in self.executor.bindings)
            != self.runtime_identities
            or self.retained_provider_count != 0
            or self.persistent_frame_tensor_bytes != 0
            or self.persistent_sample_tensor_bytes != 0
            or self.persistent_target_tensor_bytes != 0
            or self.persistent_prediction_tensor_bytes != 0
            or self.native_runtime_verified
            or self.allocator_storage_bytes_measured
            or self.allocator_peak_measured
            or self.python_object_bytes_measured
            or self.generation_digest != self._sealed_generation_digest
            or self.generation_digest != _lane_generation_digest(self)
            or id(self._construction_lifetime)
            != self._construction_lifetime_identity
            or self._construction_lifetime.phase != "transferred"
            or tuple(
                id(runtime)
                for runtime in self._construction_lifetime.runtimes
            )
            != self.runtime_identities
            or self._construction_lifetime.executor is not self.executor
        ):
            raise ValueError("native lazy bundle lane identity/memory contract changed")
        self._construction_lifetime.assert_retained()
        self.bundle.spatial_bundle.assert_warm_layout()
        for runtime in self.runtimes:
            runtime.assert_warm_layout()
        self.executor.assert_current()

    def assert_cold_current(
        self,
        provider: PaperKineticLazyProgramBundleProvider,
    ) -> None:
        """Revalidate bundle/provider provenance at an explicit cold boundary."""

        if not isinstance(provider, PaperKineticLazyProgramBundleProvider):
            raise TypeError("provider must be PaperKineticLazyProgramBundleProvider")
        if (
            id(provider) != self.provider_identity
            or provider.generation_digest != self.provider_generation_digest
        ):
            raise ValueError("native lazy lane belongs to a different bundle provider")
        self.bundle.assert_cold_current(provider)
        self.assert_warm_layout()
        observed = _resident_tensor_accounting(
            self.bundle,
            self.runtimes,
            self.executor,
            self._construction_lifetime,
        )
        if observed != self._stored_tensor_accounting():
            raise ValueError("native lazy bundle lane resident logical bytes changed")

    def assert_current(
        self,
        provider: PaperKineticLazyProgramBundleProvider,
    ) -> None:
        """Provider-aware alias for the required pre-use cold check."""

        self.assert_cold_current(provider)

    def memory_report(
        self,
        requested_observation_count: int,
    ) -> PaperKineticNativeLazyBundleLaneMemory:
        self.assert_warm_layout()
        _require_positive_int(
            requested_observation_count,
            name="requested_observation_count",
        )
        return PaperKineticNativeLazyBundleLaneMemory(
            requested_observation_count=requested_observation_count,
            native_runtime_count=self.native_runtime_count,
            bundle_logical_tensor_object_count=(
                self.bundle_logical_tensor_object_count
            ),
            bundle_logical_tensor_bytes=self.bundle_logical_tensor_bytes,
            runtime_additional_logical_tensor_object_count=(
                self.runtime_additional_logical_tensor_object_count
            ),
            runtime_additional_logical_tensor_bytes=(
                self.runtime_additional_logical_tensor_bytes
            ),
            executor_additional_logical_tensor_object_count=(
                self.executor_additional_logical_tensor_object_count
            ),
            executor_additional_logical_tensor_bytes=(
                self.executor_additional_logical_tensor_bytes
            ),
            resident_logical_tensor_object_count=(
                self.resident_logical_tensor_object_count
            ),
            resident_logical_tensor_bytes=self.resident_logical_tensor_bytes,
            persistent_frame_tensor_bytes=0,
            persistent_sample_tensor_bytes=0,
            persistent_target_tensor_bytes=0,
            persistent_prediction_tensor_bytes=0,
        )

    def _stored_tensor_accounting(self) -> tuple[int, ...]:
        return (
            self.bundle_logical_tensor_object_count,
            self.bundle_logical_tensor_bytes,
            self.runtime_additional_logical_tensor_object_count,
            self.runtime_additional_logical_tensor_bytes,
            self.executor_additional_logical_tensor_object_count,
            self.executor_additional_logical_tensor_bytes,
            self.resident_logical_tensor_object_count,
            self.resident_logical_tensor_bytes,
        )


def prepare_paper_kinetic_native_lazy_bundle_lane_construction_lifetime(
    bundle: PaperKineticLazyProgramBundle,
    provider: PaperKineticLazyProgramBundleProvider,
    native_ops: Any,
    *,
    device: torch.device | str,
    backend_provenance: str,
    max_resident_logical_tensor_bytes: int | None = None,
) -> PaperKineticNativeLazyBundleLaneConstructionLifetime:
    """Validate and install every lane predecessor before device work."""

    if not isinstance(bundle, PaperKineticLazyProgramBundle):
        raise TypeError("bundle must be PaperKineticLazyProgramBundle")
    if not isinstance(provider, PaperKineticLazyProgramBundleProvider):
        raise TypeError("provider must be PaperKineticLazyProgramBundleProvider")
    if not isinstance(backend_provenance, str) or not backend_provenance.strip():
        raise ValueError("backend_provenance must be nonempty")
    bundle.assert_cold_current(provider)
    resolved_device = torch.device(device)
    if resolved_device != bundle.spatial_bundle.device:
        raise ValueError("native lazy lane device must match its union-local bundle")
    preflight_bytes = estimate_paper_kinetic_native_lazy_bundle_lane_resident_bytes(
        bundle,
        device=resolved_device,
    )
    if max_resident_logical_tensor_bytes is not None:
        _require_positive_int(
            max_resident_logical_tensor_bytes,
            name="max_resident_logical_tensor_bytes",
        )
        if preflight_bytes > max_resident_logical_tensor_bytes:
            raise ValueError(
                "native lazy lane preflight exceeds its explicit logical tensor budget: "
                f"estimated={preflight_bytes}, budget={max_resident_logical_tensor_bytes}"
            )
    lowering = bundle.sampler.lowering
    sources = bundle.sampler.sources
    expected_blocks = _canonical_native_blocks(bundle)
    spatial_digests = tuple(
        binding.native_block_generation_digest
        for binding in bundle.spatial_bundle.native_blocks
    )
    expected_digests = tuple(block.generation_digest for block in expected_blocks)
    if spatial_digests != expected_digests:
        raise ValueError("lazy bundle union-local block order differs from its lowering")
    lifetime = PaperKineticNativeLazyBundleLaneConstructionLifetime(
        bundle=bundle,
        provider=provider,
        native_ops=native_ops,
        device=resolved_device,
        backend_provenance=backend_provenance,
        maximum_resident_logical_tensor_bytes=max_resident_logical_tensor_bytes,
        preflight_logical_tensor_bytes=preflight_bytes,
        lowering=lowering,
        sources=tuple(sources),
        expected_blocks=expected_blocks,
        expected_block_generation_digests=expected_digests,
        provider_identity=id(provider),
        provider_generation_digest=provider.generation_digest,
        bundle_generation_digest=bundle.generation_digest,
        _bundle_identity=id(bundle),
        _native_ops_identity=id(native_ops),
        _seal=_LANE_CONSTRUCTION_LIFETIME_SEAL,
    )
    lifetime.assert_retained()
    return lifetime


def materialize_paper_kinetic_native_lazy_bundle_lane(
    lifetime: PaperKineticNativeLazyBundleLaneConstructionLifetime,
) -> PaperKineticNativeLazyBundleLane:
    """Materialize one preinstalled lane lifetime exactly once."""

    if not isinstance(
        lifetime,
        PaperKineticNativeLazyBundleLaneConstructionLifetime,
    ):
        raise TypeError(
            "lifetime must be PaperKineticNativeLazyBundleLaneConstructionLifetime"
        )
    lifetime.assert_retained()
    if lifetime.phase != "installed":
        raise ValueError("native lazy lane construction lifetime was already used")
    provider = lifetime.provider
    if not isinstance(provider, PaperKineticLazyProgramBundleProvider):
        raise ValueError("native lazy lane construction lost its provider")
    lifetime.phase = "materializing"

    payloads = tuple(
        iter_materialize_kinetic_native_equal_rank_blocks(
            lifetime.lowering,
            lifetime.sources,
        )
    )
    lifetime.payloads.extend(payloads)
    if (
        len(payloads) != len(lifetime.expected_blocks)
        or any(
            payload.block is not block
            for payload, block in zip(
                payloads,
                lifetime.expected_blocks,
                strict=True,
            )
        )
    ):
        raise ValueError("native payload materialization changed canonical block coverage")
    for payload in payloads:
        runtime_lifetime = (
            prepare_kinetic_native_equal_rank_runtime_construction_lifetime(
                payload,
                lowering=lifetime.lowering,
                sources=lifetime.sources,
                native_ops=lifetime.native_ops,
                device=lifetime.device,
            )
        )
        lifetime.current_payload = payload
        lifetime.current_runtime_lifetime = runtime_lifetime
        lifetime.runtime_lifetimes.append(runtime_lifetime)
        runtime = prepare_kinetic_native_equal_rank_runtime_block(
            payload,
            lowering=lifetime.lowering,
            sources=lifetime.sources,
            native_ops=lifetime.native_ops,
            device=lifetime.device,
            construction_lifetime=runtime_lifetime,
        )
        lifetime.runtimes.append(runtime)
        lifetime.current_runtime_lifetime = None
        lifetime.current_payload = None
    runtimes = tuple(lifetime.runtimes)
    if len({id(runtime) for runtime in runtimes}) != len(lifetime.expected_blocks):
        raise ValueError("native lazy lane did not create one unique runtime per block")
    executor = prepare_kinetic_native_material_step_executor(
        lifetime.native_ops,
        tuple((runtime, lifetime.bundle.sampler) for runtime in runtimes),
        backend_provenance=lifetime.backend_provenance,
    )
    lifetime.executor = executor
    # The completed lane must not retain the provider.  Drop that construction
    # root before measuring and sealing resident state; measuring first counted
    # provider-only tensors that were released immediately afterwards, so the
    # stored accounting disagreed with the live lane and could exceed the
    # otherwise conservative preflight.
    lifetime.provider = None
    lifetime.phase = "transferred"
    lifetime.assert_retained()
    tensor_accounting = _resident_tensor_accounting(
        lifetime.bundle,
        runtimes,
        executor,
        lifetime,
    )
    provisional = PaperKineticNativeLazyBundleLane(
        bundle=lifetime.bundle,
        runtimes=runtimes,
        executor=executor,
        device=lifetime.device,
        backend_provenance=lifetime.backend_provenance,
        provider_identity=id(provider),
        provider_generation_digest=provider.generation_digest,
        bundle_identity=id(lifetime.bundle),
        bundle_generation_digest=lifetime.bundle.generation_digest,
        runtime_identities=tuple(id(runtime) for runtime in runtimes),
        runtime_generation_ids=tuple(runtime.generation_id for runtime in runtimes),
        canonical_native_block_generation_digests=(
            lifetime.expected_block_generation_digests
        ),
        executor_identity=id(executor),
        executor_generation_id=executor.generation_id,
        bundle_logical_tensor_object_count=tensor_accounting[0],
        bundle_logical_tensor_bytes=tensor_accounting[1],
        runtime_additional_logical_tensor_object_count=tensor_accounting[2],
        runtime_additional_logical_tensor_bytes=tensor_accounting[3],
        executor_additional_logical_tensor_object_count=tensor_accounting[4],
        executor_additional_logical_tensor_bytes=tensor_accounting[5],
        resident_logical_tensor_object_count=tensor_accounting[6],
        resident_logical_tensor_bytes=tensor_accounting[7],
        generation_digest="",
        _sealed_generation_digest="",
        _construction_lifetime=lifetime,
        _construction_lifetime_identity=id(lifetime),
        _seal=_LANE_SEAL,
    )
    generation_digest = _lane_generation_digest(provisional)
    lane = PaperKineticNativeLazyBundleLane(
        **{
            **provisional.__dict__,
            "generation_digest": generation_digest,
            "_sealed_generation_digest": generation_digest,
        }
    )
    lane.assert_cold_current(provider)
    if lane.resident_logical_tensor_bytes > lifetime.preflight_logical_tensor_bytes:
        raise ArithmeticError(
            "native lazy lane resident bytes exceeded its conservative preflight"
        )
    if (
        lifetime.maximum_resident_logical_tensor_bytes is not None
        and lane.resident_logical_tensor_bytes
        > lifetime.maximum_resident_logical_tensor_bytes
    ):
        raise ArithmeticError("native lazy lane exceeded its explicit logical tensor budget")
    return lane


def prepare_paper_kinetic_native_lazy_bundle_lane(
    bundle: PaperKineticLazyProgramBundle,
    provider: PaperKineticLazyProgramBundleProvider,
    native_ops: Any,
    *,
    device: torch.device | str,
    backend_provenance: str,
    max_resident_logical_tensor_bytes: int | None = None,
) -> PaperKineticNativeLazyBundleLane:
    """Legacy one-call CPU preparation; accelerators require two-phase use."""

    lifetime = prepare_paper_kinetic_native_lazy_bundle_lane_construction_lifetime(
        bundle,
        provider,
        native_ops,
        device=device,
        backend_provenance=backend_provenance,
        max_resident_logical_tensor_bytes=max_resident_logical_tensor_bytes,
    )
    if lifetime.device.type != "cpu":
        raise RuntimeError(
            "accelerator lane preparation requires caller-retained two-phase "
            "construction lifetime"
        )
    return materialize_paper_kinetic_native_lazy_bundle_lane(lifetime)


def estimate_paper_kinetic_native_lazy_bundle_lane_resident_bytes(
    bundle: PaperKineticLazyProgramBundle,
    *,
    device: torch.device | str,
) -> int:
    """Conservative pre-allocation logical-byte bound for one native lane.

    The bound includes tensors already reachable from the lazy bundle, every
    materialized equal-rank payload, and a complete additional runtime launch
    copy for every block even when the runtime can alias payload tensors on the
    requested device.  Native allocator reservations and opaque tensor caches
    inside a third-party ops object remain outside this source-level bound.
    """

    if not isinstance(bundle, PaperKineticLazyProgramBundle):
        raise TypeError("bundle must be PaperKineticLazyProgramBundle")
    resolved_device = torch.device(device)
    if resolved_device != bundle.spatial_bundle.device:
        raise ValueError("native lazy lane estimate device must match the bundle")
    bundle_tensors = _reachable_tensors((bundle,))
    bundle_bytes = _tensor_map_bytes(bundle_tensors, set(bundle_tensors))
    lowering = bundle.sampler.lowering
    payload_bytes = lowering.total_materialized_block_tensor_bytes
    runtime_copy_upper_bound = sum(
        8 * len(block.source_site_ids)
        + 4 * (block.row_count + 1)
        + 4 * block.word_count
        + 4 * block.node_count * block.word_count
        + 4 * 4
        + 4  # device config_f32
        + 4  # retained CPU epsilon predecessor until lane completion fence
        for bucket in lowering.buckets
        for block in bucket.blocks
    )
    return bundle_bytes + payload_bytes + runtime_copy_upper_bound


def _canonical_native_blocks(
    bundle: PaperKineticLazyProgramBundle,
) -> tuple[Any, ...]:
    return tuple(
        block
        for bucket in bundle.sampler.lowering.buckets
        for block in bucket.blocks
    )


def _resident_tensor_accounting(
    bundle: PaperKineticLazyProgramBundle,
    runtimes: Sequence[KineticNativeEqualRankRuntimeBlock],
    executor: KineticNativeMaterialStepExecutor,
    construction_lifetime: (
        PaperKineticNativeLazyBundleLaneConstructionLifetime
    ),
) -> tuple[int, ...]:
    bundle_tensors = _reachable_tensors((bundle,))
    runtime_tensors = _reachable_tensors(tuple(runtimes))
    executor_tensors = _reachable_tensors((executor, construction_lifetime))
    bundle_ids = set(bundle_tensors)
    runtime_additional_ids = set(runtime_tensors) - bundle_ids
    resident_before_executor = bundle_ids | set(runtime_tensors)
    executor_additional_ids = set(executor_tensors) - resident_before_executor
    resident_ids = resident_before_executor | set(executor_tensors)
    return (
        len(bundle_ids),
        _tensor_map_bytes(bundle_tensors, bundle_ids),
        len(runtime_additional_ids),
        _tensor_map_bytes(runtime_tensors, runtime_additional_ids),
        len(executor_additional_ids),
        _tensor_map_bytes(executor_tensors, executor_additional_ids),
        len(resident_ids),
        _tensor_map_bytes(
            {**bundle_tensors, **runtime_tensors, **executor_tensors},
            resident_ids,
        ),
    )


def _reachable_tensors(roots: tuple[Any, ...]) -> dict[int, torch.Tensor]:
    tensors: dict[int, torch.Tensor] = {}
    visited: set[int] = set()

    def visit(value: Any) -> None:
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(value, torch.Tensor):
            tensors[identity] = value
            return
        if is_dataclass(value) and not isinstance(value, type):
            for descriptor in fields(value):
                visit(getattr(value, descriptor.name))
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(key)
                visit(item)
            return
        if isinstance(value, (tuple, list, set, frozenset)):
            for item in value:
                visit(item)

    visit(roots)
    return tensors


def _tensor_map_bytes(
    tensors: Mapping[int, torch.Tensor],
    identities: set[int],
) -> int:
    return sum(
        tensors[identity].numel() * tensors[identity].element_size()
        for identity in identities
    )


def _lane_generation_digest(lane: PaperKineticNativeLazyBundleLane) -> str:
    return _digest_parts(
        LANE_PROVENANCE,
        lane.backend_provenance,
        str(lane.device),
        lane.provider_identity,
        lane.provider_generation_digest,
        lane.bundle_identity,
        lane.bundle_generation_digest,
        lane.runtime_identities,
        lane.runtime_generation_ids,
        lane.canonical_native_block_generation_digests,
        lane.executor_identity,
        lane.executor_generation_id,
        lane._construction_lifetime_identity,
        tuple(id(item) for item in lane._construction_lifetime.runtime_lifetimes),
        lane._stored_tensor_accounting(),
        0,
        False,
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
    "LANE_PROVENANCE",
    "LANE_STATUS",
    "PaperKineticNativeLazyBundleLane",
    "PaperKineticNativeLazyBundleLaneConstructionLifetime",
    "PaperKineticNativeLazyBundleLaneMemory",
    "estimate_paper_kinetic_native_lazy_bundle_lane_resident_bytes",
    "materialize_paper_kinetic_native_lazy_bundle_lane",
    "prepare_paper_kinetic_native_lazy_bundle_lane",
    "prepare_paper_kinetic_native_lazy_bundle_lane_construction_lifetime",
]
