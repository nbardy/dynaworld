"""Warm-safe native ABI adapter for one equal-rank kinetic block.

The equal-rank lowering owns the cold mathematical/provenance proof.  This
module turns one sealed ``KineticNativeEqualRankBlockPayload`` into the exact
precompiled-length forward/VJP launch contract while keeping that proof off
the warm path:

* preparation reruns full source/content validation and may copy tensors;
* warm validation checks only object identity, tensor identity/layout/version,
  scalar metadata, and native callable identity;
* material refresh consumes caller-owned compact ``[RGB,density]`` rows;
* reverse consumes a caller-owned compact bar and can scatter it directly into
  one caller-owned global bar with ``index_add_``;
* no frame/sample/target/prediction tensor is retained by the adapter.

The legacy native forward returns a node-chart tensor and remains an oracle.
The lazy production path uses a distinct suffixed into-output ABI so its caller
can allocate and root ``[R,J,4]`` before enqueue.  Full geometry reverse
additionally returns a ``[J,W]`` physical-length bar; material-only reverse uses
a separate ABI and allocates no such bar.  These are execution outputs, not
validation allocations.  Logical tensor-byte accounting is exact; allocator
storage/peak and Python-object bytes are explicitly unmeasured.

This remains ``source_only/native_runtime_unverified``.  CPU tests inject a
differentiable fake-native executor; they do not establish Metal build, parity,
allocator, bandwidth, or trainer integration.

The suffixed fused direct full-VJP is a separate, unpromoted fixed-camera lane.
Its low-level preparer is intentionally only structural.  The adapter below is
the provenance-bearing entrance: immediately before launch it revalidates the
sealed world/runtime/payload, live lowering and chart sources, and
compiler-issued continuous owner-topology certificate digests against that
live content.  Immediately after launch it fences and accepts the native
scalar validation receipt before returning any result to its caller.  It does
not rerun the active all-site compiler.  The staged sparse geometry reducer
remains the correctness oracle and trainer promotion is still pending.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from kinetic_native_equal_rank_lowering import (
    WARM_VALIDATION_KIND,
    KineticNativeEqualRankBlockPayload,
    KineticNativeEqualRankChartSource,
    KineticNativeEqualRankLowering,
)

ADAPTER_PROVENANCE = "kinetic-native-equal-rank-runtime-adapter-v1"
RUNTIME_STATUS = "source_only/native_runtime_unverified"
FORWARD_OP_NAME = "kinetic_precompiled_length_p0_lie_node_forward_launch_only"
FORWARD_INTO_OP_NAME = (
    "kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1"
)
VJP_OP_NAME = "kinetic_precompiled_length_p0_lie_node_vjp_accumulate_launch_only"
MATERIAL_VJP_OP_NAME = (
    "kinetic_precompiled_length_p0_lie_material_node_vjp_accumulate_launch_only"
)
FUSED_PREPARE_OP_NAME = "prepare_kinetic_fused_direct_full_vjp_v1"
FUSED_VJP_OP_NAME = "kinetic_fused_direct_full_vjp_accumulate_launch_only_v1"
FUSED_STATUS_INIT_OP_NAME = (
    "kinetic_fused_direct_full_vjp_validation_status_init_v1"
)
FUSED_VJP_RUNTIME_STATUS = (
    "source_only/native_runtime_unverified/fixed_camera_fused_v1_unpromoted"
)
FUSED_UNION_PREPARE_OP_NAME = "prepare_kinetic_fused_union_full_vjp_v2"
FUSED_UNION_VJP_OP_NAME = "kinetic_fused_union_full_vjp_accumulate_launch_only_v2"
FUSED_UNION_STATUS_INIT_OP_NAME = (
    "kinetic_fused_union_full_vjp_validation_status_init_v2"
)
FUSED_UNION_VJP_RUNTIME_STATUS = (
    "source_only/native_runtime_unverified/fixed_camera_fused_union_v2_unpromoted"
)
_FUSED_RAW_PREPARED_TENSOR_NAMES = (
    "word_offsets_i32",
    "word_owner_i32",
    "source_site_ids_i64",
    "node_physical_length_f32",
    "site_rgba_f32",
    "node_chart_f32",
    "row_node_time_f32",
    "row_near_far_f32",
    "row_ray_coeff_f32",
    "compact_positions0_f32",
    "compact_velocities_f32",
    "compact_weight_coefficients_f32",
    "config_i32",
    "config_f32",
)

_RUNTIME_BLOCK_SEAL = object()
_RUNTIME_CONSTRUCTION_LIFETIME_SEAL = object()
_WORLD_SEAL = object()
_VJP_RESULT_SEAL = object()
_MATERIAL_VJP_RESULT_SEAL = object()
_FUSED_VJP_SEAL = object()
_FUSED_VJP_TRANSACTION_SEAL = object()
_FUSED_VJP_TRANSACTION_RESULT_SEAL = object()
_FUSED_UNION_VJP_TRANSACTION_SEAL = object()
_FUSED_UNION_VJP_TRANSACTION_RESULT_SEAL = object()
_FUSED_UNION_VJP_CONSTRUCTION_LIFETIME_SEAL = object()

_FUSED_VJP_RESTART_REQUIRED_QUARANTINE: list[Any] = []
_FUSED_UNION_REJECTED_ROOT_QUARANTINE: Any | None = None


def _assert_fused_process_not_quarantined() -> None:
    """Forbid every later fused adapter operation after unknown completion."""

    if (
        _FUSED_VJP_RESTART_REQUIRED_QUARANTINE
        or _FUSED_UNION_REJECTED_ROOT_QUARANTINE is not None
    ):
        raise RuntimeError(
            "a prior fused transaction is quarantined; process restart is required"
        )


def _retain_fused_union_rejected_roots(state: Any) -> None:
    """Keep exactly one fail-stop root carrier; later fused work is forbidden."""

    global _FUSED_UNION_REJECTED_ROOT_QUARANTINE
    if _FUSED_UNION_REJECTED_ROOT_QUARANTINE is None:
        _FUSED_UNION_REJECTED_ROOT_QUARANTINE = state


@dataclass(frozen=True)
class KineticNativeEqualRankRuntimeMemory:
    """Logical retained bytes; never an allocator or Python-heap claim."""

    requested_frame_count: int
    source_payload_tensor_bytes: int
    runtime_launch_tensor_bytes: int
    runtime_launch_aliased_payload_tensor_bytes: int
    runtime_owned_persistent_tensor_bytes: int
    unique_retained_tensor_bytes: int
    source_payload_identity_domain_tensor_bytes: int
    source_payload_topology_tensor_bytes: int
    source_payload_node_physical_length_tensor_bytes: int
    runtime_config_f32_tensor_bytes: int
    persistent_frame_tensor_bytes: int
    persistent_sample_tensor_bytes: int
    persistent_target_tensor_bytes: int
    persistent_prediction_tensor_bytes: int
    dense_row_by_global_time_tensor_bytes: int
    allocator_storage_bytes_measured: bool
    allocator_peak_measured: bool
    python_object_bytes_measured: bool


@dataclass
class KineticNativeEqualRankRuntimeConstructionLifetime:
    """Preinstalled roots for every runtime tensor device transfer.

    The owner must create and retain this carrier before materialization.  A
    transfer that raises can therefore never make its CPU/source predecessor
    unreachable: the current source, all earlier returned destinations, and
    the complete payload/lowering/source provenance remain rooted here.
    """

    payload: KineticNativeEqualRankBlockPayload = field(repr=False)
    lowering: KineticNativeEqualRankLowering = field(repr=False)
    sources: tuple[KineticNativeEqualRankChartSource, ...] = field(repr=False)
    native_ops: Any = field(repr=False)
    device: torch.device
    requested_physical_length_epsilon: float
    physical_length_epsilon: float
    epsilon_f32_cpu: torch.Tensor = field(repr=False)
    source_tensors: tuple[torch.Tensor, ...] = field(repr=False)
    native_abi_identity: tuple[tuple[str, int], ...]
    transferred_tensors: list[tuple[torch.Tensor, bool]] = field(
        default_factory=list,
        repr=False,
    )
    transfer_intermediates: list[torch.Tensor] = field(
        default_factory=list,
        repr=False,
    )
    current_transfer_source: torch.Tensor | None = field(default=None, repr=False)
    runtime: KineticNativeEqualRankRuntimeBlock | None = field(
        default=None,
        repr=False,
    )
    phase: str = "installed"
    _payload_identity: int = field(default=0, repr=False)
    _lowering_identity: int = field(default=0, repr=False)
    _source_identities: tuple[int, ...] = field(default=(), repr=False)
    _native_ops_identity: int = field(default=0, repr=False)
    _source_tensor_identities: tuple[int, ...] = field(default=(), repr=False)
    _seal: object = field(default=None, repr=False)

    def assert_retained(self) -> None:
        if (
            self._seal is not _RUNTIME_CONSTRUCTION_LIFETIME_SEAL
            or self.phase not in {"installed", "transferring", "materialized"}
            or id(self.payload) != self._payload_identity
            or id(self.lowering) != self._lowering_identity
            or tuple(id(source) for source in self.sources)
            != self._source_identities
            or id(self.native_ops) != self._native_ops_identity
            or tuple(id(tensor) for tensor in self.source_tensors)
            != self._source_tensor_identities
            or len(self.transferred_tensors) > len(self.source_tensors) + 1
            or len(self.transfer_intermediates)
            > 3 * len(self.source_tensors) + 1
            or any(
                not isinstance(tensor, torch.Tensor)
                or not isinstance(owned, bool)
                for tensor, owned in self.transferred_tensors
            )
            or any(
                not isinstance(tensor, torch.Tensor)
                for tensor in self.transfer_intermediates
            )
            or (
                self.current_transfer_source is not None
                and all(
                    self.current_transfer_source is not tensor
                    for tensor in (*self.source_tensors, self.epsilon_f32_cpu)
                )
            )
            or (self.phase == "materialized")
            != isinstance(self.runtime, KineticNativeEqualRankRuntimeBlock)
        ):
            raise ValueError("native runtime construction lifetime changed")
        if self.runtime is not None:
            self.runtime.assert_warm_layout()


@dataclass(frozen=True)
class KineticNativeEqualRankFusedDirectFullVjpV1Memory:
    """Exact logical bytes retained by the sealed fused launch token."""

    aliased_runtime_topology_tensor_bytes: int
    aliased_runtime_world_tensor_bytes: int
    owned_row_payload_tensor_bytes: int
    owned_config_tensor_bytes: int
    retained_launch_tensor_bytes: int
    unique_retained_launch_tensor_bytes: int
    owned_persistent_tensor_bytes: int
    persistent_frame_tensor_bytes: int = 0
    persistent_sample_tensor_bytes: int = 0
    persistent_target_tensor_bytes: int = 0
    persistent_prediction_tensor_bytes: int = 0
    dense_row_by_global_time_tensor_bytes: int = 0
    allocator_storage_bytes_measured: bool = False
    allocator_peak_measured: bool = False
    python_object_bytes_measured: bool = False


@dataclass(frozen=True)
class KineticNativeEqualRankRuntimeBlock:
    """Cold-sealed launch state for one bounded equal-rank block."""

    payload: KineticNativeEqualRankBlockPayload = field(repr=False)
    native_ops: Any = field(repr=False)
    device: torch.device
    global_site_count: int
    physical_length_epsilon: float
    source_site_ids_i64: torch.Tensor = field(repr=False)
    word_offsets_i32: torch.Tensor = field(repr=False)
    word_owner_i32: torch.Tensor = field(repr=False)
    node_physical_length_f32: torch.Tensor = field(repr=False)
    config_i32: torch.Tensor = field(repr=False)
    config_f32: torch.Tensor = field(repr=False)
    launch_tensor_owned: tuple[bool, ...] = field(repr=False)
    warm_tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    native_ops_identity: int
    native_abi_identity: tuple[tuple[str, int], ...]
    payload_identity: int
    payload_generation_id: str
    generation_id: str
    _sealed_generation_id: str = field(repr=False)
    adapter_provenance: str = ADAPTER_PROVENANCE
    runtime_status: str = RUNTIME_STATUS
    warm_validation_kind: str = WARM_VALIDATION_KIND
    warm_validation_device_to_host_syncs: int = 0
    warm_validation_tensor_allocations: int = 0
    requested_frame_sampling_used: bool = False
    frame_or_sample_axis_retained: bool = False
    target_or_prediction_retained: bool = False
    native_runtime_verified: bool = False
    _seal: object = field(default=None, repr=False)

    @property
    def row_count(self) -> int:
        return self.payload.row_count

    @property
    def node_count(self) -> int:
        return self.payload.node_count

    @property
    def word_count(self) -> int:
        return self.payload.word_count

    @property
    def compact_site_count(self) -> int:
        return self.payload.compact_site_count

    def _launch_tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.source_site_ids_i64,
            self.word_offsets_i32,
            self.word_owner_i32,
            self.node_physical_length_f32,
            self.config_i32,
            self.config_f32,
        )

    def memory_accounting(self, requested_frame_count: int) -> KineticNativeEqualRankRuntimeMemory:
        if (
            isinstance(requested_frame_count, bool)
            or not isinstance(requested_frame_count, int)
            or requested_frame_count < 1
        ):
            raise ValueError("requested_frame_count must be positive")
        payload_accounting = self.payload.byte_accounting
        launch_tensors = self._launch_tensors()
        launch_bytes = _tensor_bytes(launch_tensors)
        aliased_bytes = sum(
            _tensor_bytes((tensor,))
            for tensor, owned in zip(launch_tensors, self.launch_tensor_owned, strict=True)
            if not owned
        )
        owned_bytes = launch_bytes - aliased_bytes
        return KineticNativeEqualRankRuntimeMemory(
            requested_frame_count=requested_frame_count,
            source_payload_tensor_bytes=self.payload.retained_tensor_bytes,
            runtime_launch_tensor_bytes=launch_bytes,
            runtime_launch_aliased_payload_tensor_bytes=aliased_bytes,
            runtime_owned_persistent_tensor_bytes=owned_bytes,
            unique_retained_tensor_bytes=self.payload.retained_tensor_bytes + owned_bytes,
            source_payload_identity_domain_tensor_bytes=int(
                payload_accounting["identity_domain_tensor_bytes"]
            ),
            source_payload_topology_tensor_bytes=int(payload_accounting["topology_tensor_bytes"]),
            source_payload_node_physical_length_tensor_bytes=int(
                payload_accounting["node_physical_length_tensor_bytes"]
            ),
            runtime_config_f32_tensor_bytes=_tensor_bytes((self.config_f32,)),
            persistent_frame_tensor_bytes=0,
            persistent_sample_tensor_bytes=0,
            persistent_target_tensor_bytes=0,
            persistent_prediction_tensor_bytes=0,
            dense_row_by_global_time_tensor_bytes=0,
            allocator_storage_bytes_measured=False,
            allocator_peak_measured=False,
            python_object_bytes_measured=False,
        )

    def assert_warm_layout(self) -> None:
        """Fail closed without reading tensor contents or allocating tensors."""

        if self._seal is not _RUNTIME_BLOCK_SEAL:
            raise ValueError("equal-rank runtime block was not sealed by its preparer")
        if (
            self.adapter_provenance != ADAPTER_PROVENANCE
            or self.runtime_status != RUNTIME_STATUS
            or self.warm_validation_kind != WARM_VALIDATION_KIND
            or self.warm_validation_device_to_host_syncs != 0
            or self.warm_validation_tensor_allocations != 0
            or self.requested_frame_sampling_used
            or self.frame_or_sample_axis_retained
            or self.target_or_prediction_retained
            or self.native_runtime_verified
        ):
            raise ValueError("equal-rank runtime warm/source contract changed")
        if (
            id(self.payload) != self.payload_identity
            or self.payload.generation_digest != self.payload_generation_id
        ):
            raise ValueError("equal-rank runtime source payload identity changed")
        self.payload.assert_warm_layout()
        if id(self.native_ops) != self.native_ops_identity:
            raise ValueError("equal-rank runtime block belongs to different native ops")
        if _native_abi_identity(self.native_ops) != self.native_abi_identity:
            raise ValueError("equal-rank runtime native ABI identity changed")
        if (
            self.global_site_count < 1
            or self.compact_site_count > self.global_site_count
            or not math.isfinite(self.physical_length_epsilon)
            or self.physical_length_epsilon < 0.0
            or self.generation_id != self._sealed_generation_id
        ):
            raise ValueError("equal-rank runtime scalar/generation metadata changed")
        tensors = self._launch_tensors()
        if len(tensors) != len(self.warm_tensor_signatures) or any(
            _warm_tensor_signature(tensor) != signature
            for tensor, signature in zip(tensors, self.warm_tensor_signatures, strict=True)
        ):
            raise ValueError("equal-rank runtime launch tensor identity/layout/version changed")
        _require_warm_tensor(
            self.source_site_ids_i64,
            name="source_site_ids_i64",
            device=self.device,
            dtype=torch.int64,
            shape=(self.compact_site_count,),
        )
        _require_warm_tensor(
            self.word_offsets_i32,
            name="word_offsets_i32",
            device=self.device,
            dtype=torch.int32,
            shape=(self.row_count + 1,),
        )
        _require_warm_tensor(
            self.word_owner_i32,
            name="word_owner_i32",
            device=self.device,
            dtype=torch.int32,
            shape=(self.word_count,),
        )
        _require_warm_tensor(
            self.node_physical_length_f32,
            name="node_physical_length_f32",
            device=self.device,
            dtype=torch.float32,
            shape=(self.node_count, self.word_count),
        )
        _require_warm_tensor(
            self.config_i32,
            name="config_i32",
            device=self.device,
            dtype=torch.int32,
            shape=(4,),
        )
        _require_warm_tensor(
            self.config_f32,
            name="config_f32",
            device=self.device,
            dtype=torch.float32,
            shape=(1,),
        )


@dataclass(frozen=True)
class KineticNativeEqualRankWorld:
    """One caller-owned compact material snapshot and native node chart."""

    runtime: KineticNativeEqualRankRuntimeBlock = field(repr=False)
    compact_site_rgba_f32: torch.Tensor = field(repr=False)
    node_chart_f32: torch.Tensor = field(repr=False)
    warm_tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    runtime_identity: int
    generation_id: str
    _sealed_generation_id: str = field(repr=False)
    adapter_provenance: str = ADAPTER_PROVENANCE
    runtime_status: str = RUNTIME_STATUS
    compact_material_caller_owned: bool = True
    native_node_chart_output_allocated: bool = True
    native_runtime_verified: bool = False
    _seal: object = field(default=None, repr=False)

    @property
    def row_count(self) -> int:
        return self.runtime.row_count

    @property
    def node_count(self) -> int:
        return self.runtime.node_count

    @property
    def compact_site_count(self) -> int:
        return self.runtime.compact_site_count

    @property
    def source_site_ids_i64(self) -> torch.Tensor:
        return self.runtime.source_site_ids_i64

    @property
    def generation_digest(self) -> str:
        """Compatibility name used by the outer ragged coordinator."""

        return self.generation_id

    @property
    def logical_world_tensor_bytes(self) -> int:
        return _tensor_bytes((self.compact_site_rgba_f32, self.node_chart_f32))

    @property
    def memory_accounting(self) -> dict[str, int | bool | str]:
        return {
            "compact_site_rgba_tensor_bytes": _tensor_bytes((self.compact_site_rgba_f32,)),
            "node_chart_tensor_bytes": _tensor_bytes((self.node_chart_f32,)),
            "logical_world_tensor_bytes": self.logical_world_tensor_bytes,
            "adapter_allocated_compact_material_tensor_bytes": 0,
            "native_forward_output_tensor_bytes": _tensor_bytes((self.node_chart_f32,)),
            "persistent_frame_tensor_bytes": 0,
            "persistent_sample_tensor_bytes": 0,
            "persistent_target_tensor_bytes": 0,
            "persistent_prediction_tensor_bytes": 0,
            "allocator_storage_bytes_measured": False,
            "allocator_peak_measured": False,
            "compact_material_caller_owned": True,
            "native_runtime_verified": False,
            "runtime_status": RUNTIME_STATUS,
        }

    def assert_warm_layout(self) -> None:
        """Validate the world without a content read, hash, or tensor allocation."""

        if self._seal is not _WORLD_SEAL:
            raise ValueError("equal-rank world was not sealed by its refresher")
        if (
            self.adapter_provenance != ADAPTER_PROVENANCE
            or self.runtime_status != RUNTIME_STATUS
            or not self.compact_material_caller_owned
            or not self.native_node_chart_output_allocated
            or self.native_runtime_verified
            or id(self.runtime) != self.runtime_identity
            or self.generation_id != self._sealed_generation_id
        ):
            raise ValueError("equal-rank world warm/source contract changed")
        self.runtime.assert_warm_layout()
        tensors = (self.compact_site_rgba_f32, self.node_chart_f32)
        if tuple(_warm_tensor_signature(tensor) for tensor in tensors) != self.warm_tensor_signatures:
            raise ValueError("equal-rank world tensor identity/layout/version changed")
        _require_warm_tensor(
            self.compact_site_rgba_f32,
            name="compact_site_rgba_f32",
            device=self.runtime.device,
            dtype=torch.float32,
            shape=(self.compact_site_count, 4),
        )
        _require_warm_tensor(
            self.node_chart_f32,
            name="node_chart_f32",
            device=self.runtime.device,
            dtype=torch.float32,
            shape=(self.row_count, self.node_count, 4),
        )

    def assert_current(self) -> None:
        """Coordinator-compatible alias for warm validation."""

        self.assert_warm_layout()


@dataclass(frozen=True)
class KineticNativeEqualRankFusedDirectFullVjpV1:
    """Cold-sealed admission token for the unpromoted fixed-camera fused VJP."""

    world: KineticNativeEqualRankWorld = field(repr=False)
    lowering: KineticNativeEqualRankLowering = field(repr=False)
    sources: tuple[KineticNativeEqualRankChartSource, ...] = field(repr=False)
    fused_ops: Any = field(repr=False)
    raw_prepared: Any = field(repr=False)
    continuous_owner_certificate_digests: tuple[str, ...]
    row_identity_digests: tuple[str, ...]
    world_identity: int
    lowering_identity: int
    lowering_generation_id: str
    source_identities: tuple[int, ...]
    fused_ops_identity: int
    fused_abi_identity: tuple[tuple[str, int], ...]
    raw_prepared_identity: int
    memory: KineticNativeEqualRankFusedDirectFullVjpV1Memory
    generation_id: str
    _sealed_generation_id: str = field(repr=False)
    adapter_provenance: str = ADAPTER_PROVENANCE
    runtime_status: str = FUSED_VJP_RUNTIME_STATUS
    camera_mode: str = "fixed"
    ray_cotangent_surface_exposed: bool = False
    staged_sparse_oracle_retained: bool = True
    trainer_promotion_complete: bool = False
    requested_frame_sampling_used: bool = False
    frame_or_sample_axis_retained: bool = False
    target_or_prediction_retained: bool = False
    native_runtime_verified: bool = False
    _seal: object = field(default=None, repr=False)

    def assert_cold_current(self) -> None:
        """Revalidate compiler/world/certificate binding immediately prelaunch."""

        if self._seal is not _FUSED_VJP_SEAL:
            raise ValueError("fused fixed-camera VJP token was not sealed by its adapter")
        if (
            self.adapter_provenance != ADAPTER_PROVENANCE
            or self.runtime_status != FUSED_VJP_RUNTIME_STATUS
            or self.camera_mode != "fixed"
            or self.ray_cotangent_surface_exposed
            or not self.staged_sparse_oracle_retained
            or self.trainer_promotion_complete
            or self.requested_frame_sampling_used
            or self.frame_or_sample_axis_retained
            or self.target_or_prediction_retained
            or self.native_runtime_verified
            or id(self.world) != self.world_identity
            or id(self.lowering) != self.lowering_identity
            or self.lowering.generation_digest != self.lowering_generation_id
            or tuple(id(source) for source in self.sources) != self.source_identities
            or id(self.fused_ops) != self.fused_ops_identity
            or id(self.raw_prepared) != self.raw_prepared_identity
        ):
            raise ValueError("fused fixed-camera VJP provenance contract changed")
        self.world.assert_warm_layout()
        self.world.runtime.payload.assert_cold_current(self.lowering, self.sources)
        if _fused_abi_identity(self.fused_ops) != self.fused_abi_identity:
            raise ValueError("fused fixed-camera VJP callable identity changed")
        rows, _row_sources, certificate_digests, _sites = _fused_block_sources(
            self.world.runtime,
            self.lowering,
            self.sources,
        )
        if (
            tuple(row.row_identity_digest for row in rows) != self.row_identity_digests
            or certificate_digests != self.continuous_owner_certificate_digests
            or getattr(self.raw_prepared, "runtime_status", None)
            != "raw_fixed_camera_source_only_until_native_rebuild_and_sparse_oracle_parity"
            or getattr(self.raw_prepared, "row_count", None) != self.world.row_count
            or getattr(self.raw_prepared, "node_count", None) != self.world.node_count
            or getattr(self.raw_prepared, "word_count", None)
            != self.world.runtime.word_count
            or getattr(self.raw_prepared, "compact_site_count", None)
            != self.world.compact_site_count
            or getattr(self.raw_prepared, "global_site_count", None)
            != self.world.runtime.global_site_count
            or self.memory != _fused_vjp_memory(self.raw_prepared)
        ):
            raise ValueError("fused fixed-camera VJP raw/compiler binding changed")
        if self.generation_id != self._sealed_generation_id or self.generation_id != (
            _fused_generation_id(
                world=self.world,
                lowering=self.lowering,
                sources=self.sources,
                fused_ops=self.fused_ops,
                raw_prepared=self.raw_prepared,
                certificate_digests=self.continuous_owner_certificate_digests,
                row_identity_digests=self.row_identity_digests,
            )
        ):
            raise ValueError("fused fixed-camera VJP generation changed")

    def assert_current(self) -> None:
        self.assert_cold_current()


@dataclass
class _KineticNativeEqualRankFusedDirectFullVjpV1TransactionState:
    """Mutable one-shot launch state; retained globally if completion is unknown."""

    prepared_blocks: tuple[KineticNativeEqualRankFusedDirectFullVjpV1, ...]
    grad_node_chart_f32_by_block: tuple[torch.Tensor, ...]
    grad_compact_site_rgba_f32_by_block: tuple[torch.Tensor, ...]
    grad_global_positions0_f32: torch.Tensor | None
    grad_global_velocities_f32: torch.Tensor | None
    grad_global_weight_coefficients_f32: torch.Tensor | None
    validation_status_i32: torch.Tensor | None = None
    device_completion_fence: Callable[[], None] | None = None
    consumed: bool = False
    launch_attempt_count: int = 0
    launch_result_count: int = 0
    completion_fence_call_count: int = 0
    settled: bool = False
    accepted: bool = False
    quarantined: bool = False
    completion_unknown: bool = False
    validation_reason_mask: int | None = None
    failure: BaseException | None = field(default=None, repr=False)
    failure_traceback: Any = field(default=None, repr=False)
    completion_failure: BaseException | None = field(default=None, repr=False)
    completion_failure_traceback: Any = field(default=None, repr=False)


@dataclass(frozen=True)
class KineticNativeEqualRankFusedDirectFullVjpV1Transaction:
    """Cold-prepared, token-owned zero scratch for one one-shot block sequence.

    The token rejects duplicate canonical block generations, binds the exact
    prepared order and all launch tensors, owns fresh zero output scratch, and
    can be consumed only once.  It does not prove that this ordered sequence is
    the executor session's complete active manifest; that authority lives above
    this adapter.
    """

    _state: _KineticNativeEqualRankFusedDirectFullVjpV1TransactionState = field(
        repr=False
    )
    active_block_generation_ids: tuple[str, ...]
    prepared_block_generation_ids: tuple[str, ...]
    prepared_block_identities: tuple[int, ...]
    node_bar_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    output_bar_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    compact_output_scratch_tensor_bytes: int
    global_output_scratch_tensor_bytes: int
    total_output_scratch_tensor_bytes: int
    output_scratch_tensor_byte_budget: int
    output_scratch_tensor_count: int
    generation_id: str
    output_scratch_owned_by_token: bool = True
    exact_zero_output_scratch_allocated: bool = True
    duplicate_active_block_generations_rejected: bool = True
    active_manifest_coverage_certified: bool = False
    single_use_scratch_generation_certified: bool = True
    hidden_output_alias_absence_certified: bool = False
    allocator_storage_bytes_measured: bool = False
    trainer_promotion_complete: bool = False
    runtime_status: str = FUSED_VJP_RUNTIME_STATUS
    _seal: object = field(default=None, repr=False)

    def assert_ready(self) -> None:
        """Cold-revalidate all token bindings before the irreversible consume."""

        _assert_fused_transaction_ready(self)


@dataclass(frozen=True)
class KineticNativeEqualRankFusedDirectFullVjpV1TransactionResult:
    """Accepted all-block receipt for the unpromoted fixed-camera fused VJP.

    This receipt exists to make the launch ordering auditable.  One shared
    four-byte status is cleared once, every block's validation-only dispatch
    proves its visible output scratch is finite and exactly zero, every block
    performs its status-gated accumulation, and every compact output ledger plus
    the shared global ledgers are finalized on the same serialized stream.  One
    final fence accepts the complete request; there is deliberately no host
    fence between the three device phases.  A postwrite rejection means the
    token-owned bars are disposable scratch, not byte-for-byte rolled back
    state.  Hidden aliases, exact active-manifest coverage, and optimizer
    fail-atomicity remain higher-layer obligations.  The accepted output bars
    are retained for a separate out-of-place commit gate; no status or raw phase
    result is retained.
    """

    grad_compact_site_rgba_f32_by_block: tuple[torch.Tensor, ...] = field(
        repr=False
    )
    grad_global_positions0_f32: torch.Tensor = field(repr=False)
    grad_global_velocities_f32: torch.Tensor = field(repr=False)
    grad_global_weight_coefficients_f32: torch.Tensor = field(repr=False)
    output_bar_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    active_block_generation_ids: tuple[str, ...]
    transaction_generation_id: str
    validation_reason_mask: int
    validation_status_tensor_bytes_during_transaction: int
    retained_output_tensor_bytes: int
    retained_device_tensor_count: int
    block_count: int
    validation_launch_count: int
    accumulation_launch_count: int
    finalization_launch_count: int
    compact_ledger_validation_count: int
    shared_global_ledger_validation_count: int
    compact_ledger_finalization_count: int
    shared_global_ledger_finalization_count: int
    device_completion_fence_call_count: int
    device_completion_fence_provenance: str
    retained_validation_status_tensor_bytes: int = 0
    requested_frame_sampling_used: bool = False
    frame_or_sample_axis_retained: bool = False
    length_cotangent_allocated: bool = False
    all_blocks_validated_before_first_write: bool = True
    zero_initialized_output_scratch_validated: bool = True
    all_output_ledgers_finalized_before_acceptance: bool = True
    accepted_final_output_ledgers_finite: bool = True
    prospective_atomic_sum_bound_certified: bool = False
    postwrite_failure_byte_rollback_certified: bool = False
    hidden_output_alias_absence_certified: bool = False
    active_manifest_coverage_certified: bool = False
    single_use_scratch_generation_certified: bool = True
    optimizer_fail_atomicity_certified: bool = False
    rejected_scratch_quarantine_required: bool = True
    native_runtime_verified: bool = False
    trainer_promotion_complete: bool = False
    runtime_status: str = FUSED_VJP_RUNTIME_STATUS
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        """Fail closed if this accepted, unpromoted receipt is forged or stale."""

        output_bars = (
            *self.grad_compact_site_rgba_f32_by_block,
            self.grad_global_positions0_f32,
            self.grad_global_velocities_f32,
            self.grad_global_weight_coefficients_f32,
        )
        if (
            self._seal is not _FUSED_VJP_TRANSACTION_RESULT_SEAL
            or len(self.grad_compact_site_rgba_f32_by_block) != self.block_count
            or len(self.active_block_generation_ids) != self.block_count
            or len(set(self.active_block_generation_ids)) != self.block_count
            or not self.transaction_generation_id.strip()
            or any(not isinstance(tensor, torch.Tensor) for tensor in output_bars)
            or any(
                tensor.dtype != torch.float32
                or tensor.device != output_bars[0].device
                or not tensor.is_contiguous()
                for tensor in output_bars
            )
            or any(
                tensor.ndim != 2
                or tensor.shape[0] < 1
                or tensor.shape[1] != 4
                for tensor in self.grad_compact_site_rgba_f32_by_block
            )
            or self.grad_global_positions0_f32.ndim != 2
            or self.grad_global_positions0_f32.shape[0] < 1
            or self.grad_global_positions0_f32.shape[1] != 3
            or self.grad_global_velocities_f32.shape
            != self.grad_global_positions0_f32.shape
            or self.grad_global_weight_coefficients_f32.ndim != 2
            or self.grad_global_weight_coefficients_f32.shape[0]
            != self.grad_global_positions0_f32.shape[0]
            or not 1 <= self.grad_global_weight_coefficients_f32.shape[1] <= 3
            or tuple(_warm_tensor_signature(tensor) for tensor in output_bars)
            != self.output_bar_signatures
            or len(
                {
                    tensor.untyped_storage().data_ptr()
                    for tensor in output_bars
                }
            )
            != len(output_bars)
            or self.validation_reason_mask != 0
            or self.validation_status_tensor_bytes_during_transaction != 4
            or self.retained_validation_status_tensor_bytes != 0
            or self.retained_output_tensor_bytes != _tensor_bytes(output_bars)
            or self.retained_device_tensor_count != self.block_count + 3
            or self.block_count < 1
            or self.validation_launch_count != self.block_count
            or self.accumulation_launch_count != self.block_count
            or self.finalization_launch_count != self.block_count
            or self.compact_ledger_validation_count != self.block_count
            or self.shared_global_ledger_validation_count != 1
            or self.compact_ledger_finalization_count != self.block_count
            or self.shared_global_ledger_finalization_count != 1
            or self.device_completion_fence_call_count != 1
            or not isinstance(self.device_completion_fence_provenance, str)
            or not self.device_completion_fence_provenance.strip()
            or self.requested_frame_sampling_used
            or self.frame_or_sample_axis_retained
            or self.length_cotangent_allocated
            or not self.all_blocks_validated_before_first_write
            or not self.zero_initialized_output_scratch_validated
            or not self.all_output_ledgers_finalized_before_acceptance
            or not self.accepted_final_output_ledgers_finite
            or self.prospective_atomic_sum_bound_certified
            or self.postwrite_failure_byte_rollback_certified
            or self.hidden_output_alias_absence_certified
            or self.active_manifest_coverage_certified
            or not self.single_use_scratch_generation_certified
            or self.optimizer_fail_atomicity_certified
            or not self.rejected_scratch_quarantine_required
            or self.native_runtime_verified
            or self.trainer_promotion_complete
            or self.runtime_status != FUSED_VJP_RUNTIME_STATUS
        ):
            raise ValueError("fused fixed-camera transaction receipt changed or was forged")


@dataclass
class KineticNativeEqualRankFusedUnionFullVjpV2ConstructionLifetime:
    """Caller-installed owner for partial raw tokens and output allocations."""

    prepared_blocks: tuple[KineticNativeEqualRankFusedDirectFullVjpV1, ...] = field(
        repr=False
    )
    grad_node_chart_f32_by_block: tuple[torch.Tensor, ...] = field(repr=False)
    spatial_bundle: Any = field(repr=False)
    active_block_manifest_generation_id: str
    max_transaction_scratch_tensor_bytes: int
    active_block_generation_ids: tuple[str, ...]
    compact_to_geometry_output_by_block: tuple[tuple[int, ...], ...]
    thresholds_f32_by_block: tuple[tuple[float, ...], ...]
    union_abi_identity: tuple[tuple[str, int], ...]
    required_output_bar_tensor_bytes: int
    required_validation_status_tensor_bytes: int
    required_transaction_scratch_tensor_bytes: int
    union_site_count: int
    weight_coefficient_count: int
    prepared_block_identities: tuple[int, ...]
    node_bar_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    spatial_bundle_identity: int
    spatial_bundle_generation_digest: str
    union_identity_signature: tuple[object, ...] = field(repr=False)
    compact_map_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    compact_map_generation_digests: tuple[str, ...]
    raw_union_blocks: list[Any | None] = field(default_factory=list, repr=False)
    output_tensors: list[torch.Tensor | None] = field(default_factory=list, repr=False)
    current_raw_block_index: int | None = None
    transaction: Any = field(default=None, repr=False)
    phase: str = "installed"
    construction_completion_fence_call_count: int = 0
    construction_completion_fence_provenance: str | None = None
    quarantined: bool = False
    completion_unknown: bool = False
    settled: bool = False
    failure: BaseException | None = field(default=None, repr=False)
    completion_failure: BaseException | None = field(default=None, repr=False)
    _seal: object = field(default=None, repr=False)

    def assert_retained(self) -> None:
        if self._seal is not _FUSED_UNION_VJP_CONSTRUCTION_LIFETIME_SEAL:
            raise ValueError("union-v2 construction lifetime seal changed")
        if self.phase == "released":
            if (
                self.prepared_blocks
                or self.grad_node_chart_f32_by_block
                or self.spatial_bundle is not None
                or self.active_block_generation_ids
                or self.compact_to_geometry_output_by_block
                or self.thresholds_f32_by_block
                or self.union_abi_identity
                or self.prepared_block_identities
                or self.node_bar_signatures
                or self.union_identity_signature
                or self.compact_map_signatures
                or self.compact_map_generation_digests
                or self.raw_union_blocks
                or self.output_tensors
                or self.current_raw_block_index is not None
                or self.transaction is not None
                or self.quarantined
                or self.completion_unknown
                or self.settled
            ):
                raise ValueError("released union-v2 construction lifetime retained roots")
            return
        block_count = len(self.prepared_blocks)
        bindings = tuple(self.spatial_bundle.native_blocks)
        if (
            self.phase
            not in {"installed", "materializing", "transferred", "quarantined"}
            or block_count < 1
            or len(self.grad_node_chart_f32_by_block) != block_count
            or len(self.active_block_generation_ids) != block_count
            or len(self.compact_to_geometry_output_by_block) != block_count
            or len(self.thresholds_f32_by_block) != block_count
            or self.required_validation_status_tensor_bytes != 4
            or self.required_transaction_scratch_tensor_bytes
            != self.required_output_bar_tensor_bytes + 4
            or self.required_transaction_scratch_tensor_bytes
            > self.max_transaction_scratch_tensor_bytes
            or len(self.prepared_block_identities) != block_count
            or tuple(id(block) for block in self.prepared_blocks)
            != self.prepared_block_identities
            or id(self.spatial_bundle) != self.spatial_bundle_identity
            or self.spatial_bundle.generation_digest
            != self.spatial_bundle_generation_digest
            or self.active_block_manifest_generation_id
            != self.spatial_bundle_generation_digest
            or self.union_identity_signature
            != _warm_tensor_signature(self.spatial_bundle.source_site_ids_i64)
            or self.compact_map_signatures
            != tuple(
                _warm_tensor_signature(binding.compact_to_union_i64)
                for binding in bindings
            )
            or self.compact_map_generation_digests
            != tuple(binding.mapping_generation_digest for binding in bindings)
            or self.node_bar_signatures
            != tuple(
                _warm_tensor_signature(tensor)
                for tensor in self.grad_node_chart_f32_by_block
            )
            or len(self.raw_union_blocks) != block_count
            or len(self.output_tensors) != block_count + 3
            or any(
                tensor is not None and not isinstance(tensor, torch.Tensor)
                for tensor in self.output_tensors
            )
            or (
                self.current_raw_block_index is not None
                and not 0 <= self.current_raw_block_index < block_count
            )
            or self.construction_completion_fence_call_count not in {0, 1}
            or (
                self.construction_completion_fence_call_count == 1
                and (
                    not isinstance(self.construction_completion_fence_provenance, str)
                    or not self.construction_completion_fence_provenance.strip()
                )
            )
            or self.quarantined != (self.phase == "quarantined")
            or self.completion_unknown and not self.quarantined
            or self.settled and not self.quarantined
            or (
                self.phase == "installed"
                and (
                    any(raw is not None for raw in self.raw_union_blocks)
                    or any(tensor is not None for tensor in self.output_tensors)
                )
            )
            or (
                self.phase == "transferred"
                and (
                    any(raw is None for raw in self.raw_union_blocks)
                    or any(
                        not isinstance(tensor, torch.Tensor)
                        for tensor in self.output_tensors
                    )
                    or _tensor_bytes(
                        tuple(
                            tensor
                            for tensor in self.output_tensors
                            if isinstance(tensor, torch.Tensor)
                        )
                    )
                    != self.required_output_bar_tensor_bytes
                )
            )
            or (
                self.phase == "transferred"
                and not isinstance(
                    self.transaction,
                    KineticNativeEqualRankFusedUnionFullVjpV2Transaction,
                )
            )
            or (self.phase == "installed" and self.transaction is not None)
        ):
            raise ValueError("union-v2 construction lifetime changed or lost roots")


@dataclass
class _KineticNativeEqualRankFusedUnionFullVjpV2TransactionState:
    """One-shot roots; failure retains this whole object as quarantine."""

    prepared_blocks: tuple[KineticNativeEqualRankFusedDirectFullVjpV1, ...]
    spatial_bundle: Any
    construction_lifetime: Any
    raw_union_blocks: tuple[Any, ...]
    grad_node_chart_f32_by_block: tuple[torch.Tensor, ...]
    grad_compact_site_rgba_f32_by_block: tuple[torch.Tensor, ...]
    grad_union_positions0_f32: torch.Tensor | None
    grad_union_velocities_f32: torch.Tensor | None
    grad_union_weight_coefficients_f32: torch.Tensor | None
    resident_union_source_site_ids_i64: torch.Tensor | None
    union_transfer_predecessor: torch.Tensor | None
    validation_status_i32: torch.Tensor | None = None
    device_completion_fence: Callable[[], None] | None = None
    consumed: bool = False
    settled: bool = False
    accepted: bool = False
    quarantined: bool = False
    completion_unknown: bool = False
    completion_fence_call_count: int = 0
    validation_launch_count: int = 0
    accumulation_launch_count: int = 0
    finalization_launch_count: int = 0
    compact_ledger_validation_count: int = 0
    shared_union_ledger_validation_count: int = 0
    compact_ledger_finalization_count: int = 0
    shared_union_ledger_finalization_count: int = 0
    validation_reason_mask: int | None = None
    failure: BaseException | None = field(default=None, repr=False)
    completion_failure: BaseException | None = field(default=None, repr=False)


@dataclass(frozen=True)
class KineticNativeEqualRankFusedUnionFullVjpV2Transaction:
    """Exact all-block ``P_b=P_U Q_b`` transaction with union-local scratch."""

    _state: _KineticNativeEqualRankFusedUnionFullVjpV2TransactionState = field(
        repr=False
    )
    active_block_manifest_generation_id: str
    spatial_bundle_identity: int
    spatial_bundle_generation_digest: str
    active_block_generation_ids: tuple[str, ...]
    prepared_block_generation_ids: tuple[str, ...]
    prepared_block_identities: tuple[int, ...]
    geometry_output_source_site_ids: tuple[int, ...]
    compact_to_geometry_output_by_block: tuple[tuple[int, ...], ...]
    raw_union_block_identities: tuple[int, ...]
    union_abi_identity: tuple[tuple[str, int], ...]
    union_identity_signature: tuple[object, ...] = field(repr=False)
    compact_map_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    compact_map_generation_digests: tuple[str, ...]
    node_bar_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    output_bar_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    output_bar_scratch_tensor_bytes: int
    validation_status_tensor_bytes_during_execution: int
    total_transaction_scratch_tensor_bytes: int
    transaction_scratch_tensor_byte_budget: int
    union_site_count: int
    block_count: int
    generation_id: str
    exact_active_block_manifest_certified: bool = True
    exact_union_identity_certified: bool = True
    exact_factorization_certified: bool = True
    output_geometry_index_space: str = "request_union"
    material_output_index_space: str = "block_compact"
    union_material_finiteness_certified: bool = False
    single_use: bool = True
    persistent_or_global_write_authorized: bool = False
    optimizer_write_authorized: bool = False
    bounded_batch_q: int | None = None
    trainer_promotion_complete: bool = False
    runtime_status: str = FUSED_UNION_VJP_RUNTIME_STATUS
    _seal: object = field(default=None, repr=False)

    def assert_ready(self) -> None:
        _assert_fused_union_transaction_ready(self)


@dataclass
class _KineticNativeEqualRankFusedUnionFullVjpV2ResultState:
    compact_bars: tuple[torch.Tensor, ...]
    union_position_bar: torch.Tensor | None
    union_velocity_bar: torch.Tensor | None
    union_weight_bar: torch.Tensor | None
    consumed: bool = False


@dataclass(frozen=True)
class KineticNativeEqualRankFusedUnionFullVjpV2TransactionResult:
    """Accepted receipt whose bars may be removed exactly once."""

    _state: _KineticNativeEqualRankFusedUnionFullVjpV2ResultState = field(
        repr=False
    )
    geometry_output_source_site_ids: tuple[int, ...]
    active_block_manifest_generation_id: str
    active_block_generation_ids: tuple[str, ...]
    transaction_generation_id: str
    output_bar_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    validation_reason_mask: int
    validation_status_tensor_bytes_during_transaction: int
    retained_validation_status_tensor_bytes: int
    retained_output_tensor_bytes: int
    retained_device_tensor_count: int
    block_count: int
    union_site_count: int
    validation_launch_count: int
    accumulation_launch_count: int
    finalization_launch_count: int
    compact_ledger_validation_count: int
    shared_union_ledger_validation_count: int
    compact_ledger_finalization_count: int
    shared_union_ledger_finalization_count: int
    device_completion_fence_call_count: int
    device_completion_fence_provenance: str
    exact_active_block_manifest_certified: bool = True
    exact_union_identity_certified: bool = True
    exact_factorization_certified: bool = True
    output_geometry_index_space: str = "request_union"
    material_output_index_space: str = "block_compact"
    union_material_finiteness_certified: bool = False
    persistent_or_global_write_performed: bool = False
    optimizer_write_performed: bool = False
    transfer_predecessors_released_after_proven_fence: bool = True
    native_runtime_verified: bool = False
    trainer_promotion_complete: bool = False
    runtime_status: str = FUSED_UNION_VJP_RUNTIME_STATUS
    _seal: object = field(default=None, repr=False)

    def assert_current(self) -> None:
        state = self._state
        union_bars = (
            state.union_position_bar,
            state.union_velocity_bar,
            state.union_weight_bar,
        )
        bars = (*state.compact_bars, *union_bars)
        if (
            self._seal is not _FUSED_UNION_VJP_TRANSACTION_RESULT_SEAL
            or state.consumed
            or any(not isinstance(tensor, torch.Tensor) for tensor in bars)
            or any(
                tensor.dtype != torch.float32
                or tensor.device != bars[0].device
                or not tensor.is_contiguous()
                for tensor in bars
            )
            or any(
                tensor.ndim != 2 or tensor.shape[0] < 1 or tensor.shape[1] != 4
                for tensor in state.compact_bars
            )
            or state.union_position_bar.shape != (self.union_site_count, 3)
            or state.union_velocity_bar.shape != (self.union_site_count, 3)
            or state.union_weight_bar.ndim != 2
            or state.union_weight_bar.shape[0] != self.union_site_count
            or not 1 <= state.union_weight_bar.shape[1] <= 3
            or tuple(_warm_tensor_signature(tensor) for tensor in bars)
            != self.output_bar_signatures
            or len({tensor.untyped_storage().data_ptr() for tensor in bars})
            != len(bars)
            or self.validation_reason_mask != 0
            or self.validation_status_tensor_bytes_during_transaction != 4
            or self.retained_validation_status_tensor_bytes != 0
            or self.block_count != len(state.compact_bars)
            or self.block_count != len(self.active_block_generation_ids)
            or self.block_count < 1
            or len(set(self.active_block_generation_ids)) != self.block_count
            or not self.active_block_manifest_generation_id.strip()
            or not self.transaction_generation_id.strip()
            or self.union_site_count != len(self.geometry_output_source_site_ids)
            or self.geometry_output_source_site_ids
            != tuple(sorted(set(self.geometry_output_source_site_ids)))
            or self.retained_output_tensor_bytes != _tensor_bytes(bars)
            or self.retained_device_tensor_count != self.block_count + 3
            or self.validation_launch_count != self.block_count
            or self.accumulation_launch_count != self.block_count
            or self.finalization_launch_count != self.block_count
            or self.compact_ledger_validation_count != self.block_count
            or self.shared_union_ledger_validation_count != 1
            or self.compact_ledger_finalization_count != self.block_count
            or self.shared_union_ledger_finalization_count != 1
            or self.device_completion_fence_call_count != 1
            or not self.device_completion_fence_provenance.strip()
            or not self.exact_active_block_manifest_certified
            or not self.exact_union_identity_certified
            or not self.exact_factorization_certified
            or self.output_geometry_index_space != "request_union"
            or self.material_output_index_space != "block_compact"
            or self.union_material_finiteness_certified
            or self.persistent_or_global_write_performed
            or self.optimizer_write_performed
            or not self.transfer_predecessors_released_after_proven_fence
            or self.native_runtime_verified
            or self.trainer_promotion_complete
            or self.runtime_status != FUSED_UNION_VJP_RUNTIME_STATUS
        ):
            raise ValueError("union-v2 accepted receipt changed, was consumed, or was forged")

    def consume_bars_once(
        self,
    ) -> tuple[
        tuple[int, ...],
        tuple[torch.Tensor, ...],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Transfer union-local bars once; this method performs no global write."""

        self.assert_current()
        state = self._state
        assert isinstance(state.union_position_bar, torch.Tensor)
        assert isinstance(state.union_velocity_bar, torch.Tensor)
        assert isinstance(state.union_weight_bar, torch.Tensor)
        result = (
            self.geometry_output_source_site_ids,
            state.compact_bars,
            state.union_position_bar,
            state.union_velocity_bar,
            state.union_weight_bar,
        )
        state.compact_bars = ()
        state.union_position_bar = None
        state.union_velocity_bar = None
        state.union_weight_bar = None
        state.consumed = True
        return result


@dataclass(frozen=True)
class KineticNativeEqualRankVJPResult:
    """Caller-owned material bars plus the native bounded length-bar output."""

    world: KineticNativeEqualRankWorld = field(repr=False)
    grad_compact_site_rgba_f32: torch.Tensor
    grad_global_site_rgba_f32: torch.Tensor | None
    grad_node_physical_length_f32: torch.Tensor
    warm_tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    world_identity: int
    accounting: dict[str, int | bool | str]
    adapter_provenance: str = ADAPTER_PROVENANCE
    runtime_status: str = RUNTIME_STATUS
    compact_bar_caller_owned: bool = True
    global_bar_caller_owned: bool = True
    compact_bar_zeroed_before_native_vjp: bool = True
    global_bar_allocated_by_adapter: bool = False
    geometry_length_bar_returned: bool = True
    geometry_parameter_vjp_implemented: bool = False
    native_runtime_verified: bool = False
    _seal: object = field(default=None, repr=False)

    def _tensors(self) -> tuple[torch.Tensor, ...]:
        if self.grad_global_site_rgba_f32 is None:
            return (self.grad_compact_site_rgba_f32, self.grad_node_physical_length_f32)
        return (
            self.grad_compact_site_rgba_f32,
            self.grad_global_site_rgba_f32,
            self.grad_node_physical_length_f32,
        )

    def assert_warm_layout(self) -> None:
        """Validate result identities/layouts without reading device values."""

        if self._seal is not _VJP_RESULT_SEAL:
            raise ValueError("equal-rank VJP result was not sealed by its executor")
        if (
            self.adapter_provenance != ADAPTER_PROVENANCE
            or self.runtime_status != RUNTIME_STATUS
            or not self.compact_bar_caller_owned
            or not self.global_bar_caller_owned
            or not self.compact_bar_zeroed_before_native_vjp
            or self.global_bar_allocated_by_adapter
            or not self.geometry_length_bar_returned
            or self.geometry_parameter_vjp_implemented
            or self.native_runtime_verified
            or id(self.world) != self.world_identity
        ):
            raise ValueError("equal-rank VJP result warm/source contract changed")
        self.world.assert_warm_layout()
        tensors = self._tensors()
        if tuple(_warm_tensor_signature(tensor) for tensor in tensors) != self.warm_tensor_signatures:
            raise ValueError("equal-rank VJP result tensor identity/layout/version changed")
        device = self.world.runtime.device
        _require_warm_tensor(
            self.grad_compact_site_rgba_f32,
            name="grad_compact_site_rgba_f32",
            device=device,
            dtype=torch.float32,
            shape=(self.world.compact_site_count, 4),
        )
        if self.grad_global_site_rgba_f32 is not None:
            _require_warm_tensor(
                self.grad_global_site_rgba_f32,
                name="grad_global_site_rgba_f32",
                device=device,
                dtype=torch.float32,
                shape=(self.world.runtime.global_site_count, 4),
            )
        _require_warm_tensor(
            self.grad_node_physical_length_f32,
            name="grad_node_physical_length_f32",
            device=device,
            dtype=torch.float32,
            shape=(self.world.node_count, self.world.runtime.word_count),
        )
        if self.accounting != _vjp_accounting(
            self.world,
            self.grad_compact_site_rgba_f32,
            self.grad_global_site_rgba_f32,
            self.grad_node_physical_length_f32,
        ):
            raise ValueError("equal-rank VJP logical-byte accounting changed")

    def assert_current(self) -> None:
        self.assert_warm_layout()


@dataclass(frozen=True)
class KineticNativeEqualRankMaterialVJPResult:
    """Material-only reverse result with no allocated ``[J,W]`` length bar."""

    world: KineticNativeEqualRankWorld = field(repr=False)
    grad_compact_site_rgba_f32: torch.Tensor
    grad_global_site_rgba_f32: torch.Tensor | None
    warm_tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    world_identity: int
    accounting: dict[str, int | bool | str]
    adapter_provenance: str = ADAPTER_PROVENANCE
    runtime_status: str = RUNTIME_STATUS
    compact_bar_caller_owned: bool = True
    global_bar_caller_owned: bool = True
    compact_bar_zeroed_before_native_vjp: bool = True
    global_bar_allocated_by_adapter: bool = False
    geometry_length_bar_returned: bool = False
    geometry_parameter_vjp_implemented: bool = False
    native_runtime_verified: bool = False
    _seal: object = field(default=None, repr=False)

    def _tensors(self) -> tuple[torch.Tensor, ...]:
        if self.grad_global_site_rgba_f32 is None:
            return (self.grad_compact_site_rgba_f32,)
        return (self.grad_compact_site_rgba_f32, self.grad_global_site_rgba_f32)

    def assert_warm_layout(self) -> None:
        if self._seal is not _MATERIAL_VJP_RESULT_SEAL:
            raise ValueError("equal-rank material VJP result was not sealed by its executor")
        if (
            self.adapter_provenance != ADAPTER_PROVENANCE
            or self.runtime_status != RUNTIME_STATUS
            or not self.compact_bar_caller_owned
            or not self.global_bar_caller_owned
            or not self.compact_bar_zeroed_before_native_vjp
            or self.global_bar_allocated_by_adapter
            or self.geometry_length_bar_returned
            or self.geometry_parameter_vjp_implemented
            or self.native_runtime_verified
            or id(self.world) != self.world_identity
        ):
            raise ValueError("equal-rank material VJP result warm/source contract changed")
        self.world.assert_warm_layout()
        tensors = self._tensors()
        if tuple(_warm_tensor_signature(tensor) for tensor in tensors) != self.warm_tensor_signatures:
            raise ValueError("equal-rank material VJP result tensor identity/layout/version changed")
        _require_warm_tensor(
            self.grad_compact_site_rgba_f32,
            name="grad_compact_site_rgba_f32",
            device=self.world.runtime.device,
            dtype=torch.float32,
            shape=(self.world.compact_site_count, 4),
        )
        if self.grad_global_site_rgba_f32 is not None:
            _require_warm_tensor(
                self.grad_global_site_rgba_f32,
                name="grad_global_site_rgba_f32",
                device=self.world.runtime.device,
                dtype=torch.float32,
                shape=(self.world.runtime.global_site_count, 4),
            )
        if self.accounting != _material_vjp_accounting(
            self.world,
            self.grad_compact_site_rgba_f32,
            self.grad_global_site_rgba_f32,
        ):
            raise ValueError("equal-rank material VJP logical-byte accounting changed")

    def assert_current(self) -> None:
        self.assert_warm_layout()


def prepare_kinetic_native_equal_rank_runtime_construction_lifetime(
    payload: KineticNativeEqualRankBlockPayload,
    *,
    lowering: KineticNativeEqualRankLowering,
    sources: Sequence[KineticNativeEqualRankChartSource],
    native_ops: Any,
    device: torch.device | str,
    physical_length_epsilon: float = 1.0e-8,
) -> KineticNativeEqualRankRuntimeConstructionLifetime:
    """Install all runtime-transfer predecessors before the first copy."""

    if not isinstance(payload, KineticNativeEqualRankBlockPayload):
        raise TypeError("payload must be KineticNativeEqualRankBlockPayload")
    if not isinstance(lowering, KineticNativeEqualRankLowering):
        raise TypeError("lowering must be KineticNativeEqualRankLowering")
    payload.assert_cold_current(lowering, sources)
    resolved_device = torch.device(device)
    native_abi_identity = _require_native_ops(
        native_ops,
        device=resolved_device,
    )
    if isinstance(physical_length_epsilon, bool):
        raise TypeError("physical_length_epsilon must be a finite nonnegative float")
    requested_epsilon = float(physical_length_epsilon)
    if not math.isfinite(requested_epsilon) or requested_epsilon < 0.0:
        raise ValueError("physical_length_epsilon must be finite and nonnegative")
    epsilon_f32_cpu = torch.tensor([requested_epsilon], dtype=torch.float32)
    epsilon = float(epsilon_f32_cpu[0].item())
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("physical_length_epsilon must remain finite and nonnegative in float32")
    if bool(torch.any(payload.node_physical_length_f32 <= epsilon).item()):
        raise ValueError("physical node lengths must remain strictly above epsilon")

    source_tensors = (
        payload.source_site_ids_i64,
        payload.word_offsets_i32,
        payload.word_owner_i32,
        payload.node_physical_length_f32,
        payload.config_i32,
    )
    lifetime = KineticNativeEqualRankRuntimeConstructionLifetime(
        payload=payload,
        lowering=lowering,
        sources=tuple(sources),
        native_ops=native_ops,
        device=resolved_device,
        requested_physical_length_epsilon=requested_epsilon,
        physical_length_epsilon=epsilon,
        epsilon_f32_cpu=epsilon_f32_cpu,
        source_tensors=source_tensors,
        native_abi_identity=native_abi_identity,
        _payload_identity=id(payload),
        _lowering_identity=id(lowering),
        _source_identities=tuple(id(source) for source in sources),
        _native_ops_identity=id(native_ops),
        _source_tensor_identities=tuple(id(tensor) for tensor in source_tensors),
        _seal=_RUNTIME_CONSTRUCTION_LIFETIME_SEAL,
    )
    lifetime.assert_retained()
    return lifetime


def materialize_kinetic_native_equal_rank_runtime_block(
    lifetime: KineticNativeEqualRankRuntimeConstructionLifetime,
) -> KineticNativeEqualRankRuntimeBlock:
    """Perform one lifetime-rooted runtime materialization exactly once."""

    if not isinstance(
        lifetime,
        KineticNativeEqualRankRuntimeConstructionLifetime,
    ):
        raise TypeError(
            "lifetime must be KineticNativeEqualRankRuntimeConstructionLifetime"
        )
    lifetime.assert_retained()
    if lifetime.phase != "installed" or lifetime.transferred_tensors:
        raise ValueError("native runtime construction lifetime was already used")
    lifetime.phase = "transferring"
    for source in lifetime.source_tensors:
        lifetime.current_transfer_source = source
        copied = _copy_or_alias_tensor_with_lifetime(
            lifetime,
            source,
            device=lifetime.device,
            dtype=source.dtype,
        )
        lifetime.transferred_tensors.append(copied)
    lifetime.current_transfer_source = lifetime.epsilon_f32_cpu
    epsilon_transfer = lifetime.epsilon_f32_cpu.to(device=lifetime.device)
    lifetime.transfer_intermediates.append(epsilon_transfer)
    lifetime.transferred_tensors.append(
        (
            epsilon_transfer,
            True,
        )
    )
    lifetime.current_transfer_source = None
    launch_tensors = tuple(value[0] for value in lifetime.transferred_tensors)
    launch_owned = tuple(value[1] for value in lifetime.transferred_tensors)
    generation_id = _runtime_generation_id(
        payload=lifetime.payload,
        device=lifetime.device,
        global_site_count=lifetime.lowering.global_site_count,
        physical_length_epsilon=lifetime.physical_length_epsilon,
        native_ops_identity=id(lifetime.native_ops),
        native_abi_identity=lifetime.native_abi_identity,
    )
    result = KineticNativeEqualRankRuntimeBlock(
        payload=lifetime.payload,
        native_ops=lifetime.native_ops,
        device=lifetime.device,
        global_site_count=lifetime.lowering.global_site_count,
        physical_length_epsilon=lifetime.physical_length_epsilon,
        source_site_ids_i64=launch_tensors[0],
        word_offsets_i32=launch_tensors[1],
        word_owner_i32=launch_tensors[2],
        node_physical_length_f32=launch_tensors[3],
        config_i32=launch_tensors[4],
        config_f32=launch_tensors[5],
        launch_tensor_owned=launch_owned,
        warm_tensor_signatures=tuple(_warm_tensor_signature(tensor) for tensor in launch_tensors),
        native_ops_identity=id(lifetime.native_ops),
        native_abi_identity=lifetime.native_abi_identity,
        payload_identity=id(lifetime.payload),
        payload_generation_id=lifetime.payload.generation_digest,
        generation_id=generation_id,
        _sealed_generation_id=generation_id,
        _seal=_RUNTIME_BLOCK_SEAL,
    )
    result.assert_warm_layout()
    lifetime.runtime = result
    lifetime.phase = "materialized"
    lifetime.assert_retained()
    return result


def prepare_kinetic_native_equal_rank_runtime_block(
    payload: KineticNativeEqualRankBlockPayload,
    *,
    lowering: KineticNativeEqualRankLowering,
    sources: Sequence[KineticNativeEqualRankChartSource],
    native_ops: Any,
    device: torch.device | str,
    physical_length_epsilon: float = 1.0e-8,
    construction_lifetime: (
        KineticNativeEqualRankRuntimeConstructionLifetime | None
    ) = None,
) -> KineticNativeEqualRankRuntimeBlock:
    """Cold-validate and materialize one launch-shaped runtime block.

    Accelerator owners that require failure quarantine must preinstall and
    retain ``construction_lifetime``.  The legacy one-call CPU path remains
    available for existing structural callers.
    """

    lifetime = construction_lifetime
    if lifetime is None:
        lifetime = prepare_kinetic_native_equal_rank_runtime_construction_lifetime(
            payload,
            lowering=lowering,
            sources=sources,
            native_ops=native_ops,
            device=device,
            physical_length_epsilon=physical_length_epsilon,
        )
    else:
        lifetime.assert_retained()
        if (
            lifetime.payload is not payload
            or lifetime.lowering is not lowering
            or tuple(id(source) for source in lifetime.sources)
            != tuple(id(source) for source in sources)
            or lifetime.native_ops is not native_ops
            or lifetime.device != torch.device(device)
            or lifetime.requested_physical_length_epsilon
            != float(physical_length_epsilon)
        ):
            raise ValueError("native runtime construction lifetime is foreign")
    return materialize_kinetic_native_equal_rank_runtime_block(lifetime)


@torch.no_grad()
def refresh_kinetic_native_equal_rank_world(
    runtime: KineticNativeEqualRankRuntimeBlock,
    compact_site_rgba_f32: torch.Tensor,
) -> KineticNativeEqualRankWorld:
    """Run native node forward from a caller-owned compact material buffer."""

    if not isinstance(runtime, KineticNativeEqualRankRuntimeBlock):
        raise TypeError("runtime must be KineticNativeEqualRankRuntimeBlock")
    runtime.assert_warm_layout()
    _require_warm_tensor(
        compact_site_rgba_f32,
        name="compact_site_rgba_f32",
        device=runtime.device,
        dtype=torch.float32,
        shape=(runtime.compact_site_count, 4),
    )
    node_chart = getattr(runtime.native_ops, FORWARD_OP_NAME)(
        runtime.word_offsets_i32,
        runtime.word_owner_i32,
        runtime.node_physical_length_f32,
        compact_site_rgba_f32,
        runtime.config_i32,
        runtime.config_f32,
        track_count=runtime.row_count,
        node_count=runtime.node_count,
    )
    if not isinstance(node_chart, torch.Tensor):
        raise TypeError("equal-rank native forward must return one tensor")
    _require_warm_tensor(
        node_chart,
        name="native node_chart_f32",
        device=runtime.device,
        dtype=torch.float32,
        shape=(runtime.row_count, runtime.node_count, 4),
    )
    if _same_storage(node_chart, compact_site_rgba_f32):
        raise ValueError("native node chart must not alias caller compact material")
    generation_id = _world_generation_id(runtime, compact_site_rgba_f32, node_chart)
    result = KineticNativeEqualRankWorld(
        runtime=runtime,
        compact_site_rgba_f32=compact_site_rgba_f32,
        node_chart_f32=node_chart,
        warm_tensor_signatures=tuple(
            _warm_tensor_signature(tensor) for tensor in (compact_site_rgba_f32, node_chart)
        ),
        runtime_identity=id(runtime),
        generation_id=generation_id,
        _sealed_generation_id=generation_id,
        _seal=_WORLD_SEAL,
    )
    result.assert_warm_layout()
    return result


@torch.no_grad()
def refresh_kinetic_native_equal_rank_world_into(
    runtime: KineticNativeEqualRankRuntimeBlock,
    compact_site_rgba_f32: torch.Tensor,
    node_chart_out_f32: torch.Tensor,
) -> KineticNativeEqualRankWorld:
    """Launch the lifetime-safe caller-owned-output forward ABI.

    The legacy return-allocating forward above remains a correctness oracle.
    It cannot provide an async lifetime guarantee when a native callable
    enqueues work, obtains its internally allocated return tensor, and then
    raises before returning it to Python.  This distinct ABI instead requires
    the caller to allocate and retain ``node_chart_out_f32`` *before* launch;
    the native callable may only fill that exact buffer and must return
    ``None``.  Native binding/shader support is a separate, still-unverified
    gate; this adapter never implements the contract by wrapping the unsafe
    return-allocating operation.
    """

    if not isinstance(runtime, KineticNativeEqualRankRuntimeBlock):
        raise TypeError("runtime must be KineticNativeEqualRankRuntimeBlock")
    runtime.assert_warm_layout()
    _require_warm_tensor(
        compact_site_rgba_f32,
        name="compact_site_rgba_f32",
        device=runtime.device,
        dtype=torch.float32,
        shape=(runtime.compact_site_count, 4),
    )
    _require_warm_tensor(
        node_chart_out_f32,
        name="node_chart_out_f32",
        device=runtime.device,
        dtype=torch.float32,
        shape=(runtime.row_count, runtime.node_count, 4),
    )
    if _same_storage(node_chart_out_f32, compact_site_rgba_f32):
        raise ValueError("caller node chart output must not alias compact material")
    forward_into = getattr(runtime.native_ops, FORWARD_INTO_OP_NAME, None)
    if not callable(forward_into):
        raise RuntimeError(
            "lifetime-safe native forward requires the distinct caller-owned "
            f"output ABI {FORWARD_INTO_OP_NAME}; the return-allocating oracle "
            "cannot be used for accelerator promotion"
        )
    returned = forward_into(
        runtime.word_offsets_i32,
        runtime.word_owner_i32,
        runtime.node_physical_length_f32,
        compact_site_rgba_f32,
        runtime.config_i32,
        runtime.config_f32,
        node_chart_out_f32,
        track_count=runtime.row_count,
        node_count=runtime.node_count,
    )
    if returned is not None:
        raise TypeError("caller-owned native forward must return None")
    generation_id = _world_generation_id(
        runtime,
        compact_site_rgba_f32,
        node_chart_out_f32,
    )
    result = KineticNativeEqualRankWorld(
        runtime=runtime,
        compact_site_rgba_f32=compact_site_rgba_f32,
        node_chart_f32=node_chart_out_f32,
        warm_tensor_signatures=tuple(
            _warm_tensor_signature(tensor)
            for tensor in (compact_site_rgba_f32, node_chart_out_f32)
        ),
        runtime_identity=id(runtime),
        generation_id=generation_id,
        _sealed_generation_id=generation_id,
        _seal=_WORLD_SEAL,
    )
    result.assert_warm_layout()
    return result


def prepare_kinetic_native_equal_rank_fused_direct_full_vjp_v1(
    world: KineticNativeEqualRankWorld,
    *,
    lowering: KineticNativeEqualRankLowering,
    sources: Sequence[KineticNativeEqualRankChartSource],
    minimum_absolute_cut_denominator: float = 1.0e-7,
    minimum_cut_cosine: float = 1.0e-8,
    minimum_coordinate_length: float = 1.0e-8,
    minimum_ray_speed: float = 1.0e-7,
    depth_closure_relative_tolerance: float = 2.0e-5,
    active_tie_relative_tolerance: float = 2.0e-5,
) -> KineticNativeEqualRankFusedDirectFullVjpV1:
    """Cold-bind live equal-rank provenance to the raw fixed-camera v1 ABI.

    This is the provenance-bearing entry into the suffixed fused lane.
    It revalidates compiler-generated continuous owner-topology certificates
    against live chart/program contents, derives every row-local payload from
    the sealed block/world, and then delegates structural packing to the raw
    variant preparer.  It does not rerun the active all-site compiler, select
    the operator for training, or claim native parity.  Publication promotion
    requires active-compiler provenance; exhaustive provenance remains an
    oracle-only input.
    """

    _assert_fused_process_not_quarantined()
    if not isinstance(world, KineticNativeEqualRankWorld):
        raise TypeError("world must be KineticNativeEqualRankWorld")
    if not isinstance(lowering, KineticNativeEqualRankLowering):
        raise TypeError("lowering must be KineticNativeEqualRankLowering")
    world.assert_warm_layout()
    normalized_sources = tuple(sources)
    world.runtime.payload.assert_cold_current(lowering, normalized_sources)
    selected_fused_ops = world.runtime.native_ops
    fused_abi_identity = _require_fused_ops(selected_fused_ops)
    rows, row_sources, certificate_digests, sites = _fused_block_sources(
        world.runtime,
        lowering,
        normalized_sources,
    )
    source_ids_cpu = world.runtime.payload.source_site_ids_i64
    row_node_time_f32 = torch.stack(
        tuple(
            source.program.charts[source.chart_index].schedule.node_times
            for source in row_sources
        )
    ).detach().to(device=world.runtime.device, dtype=torch.float32).contiguous()
    row_near_far_f32 = torch.stack(
        (
            world.runtime.payload.row_near_f64,
            world.runtime.payload.row_far_f64,
        ),
        dim=1,
    ).detach().to(device=world.runtime.device, dtype=torch.float32).contiguous()
    row_ray_coeff_f32 = torch.stack(
        tuple(source.program.binding.ray_coefficients for source in row_sources)
    ).detach().to(device=world.runtime.device, dtype=torch.float32).contiguous()
    compact_positions0_f32 = sites.positions0.index_select(
        0,
        source_ids_cpu,
    ).detach().to(device=world.runtime.device, dtype=torch.float32).contiguous()
    compact_velocities_f32 = sites.velocities.index_select(
        0,
        source_ids_cpu,
    ).detach().to(device=world.runtime.device, dtype=torch.float32).contiguous()
    compact_weight_coefficients_f32 = sites.weight_coefficients.index_select(
        0,
        source_ids_cpu,
    ).detach().to(device=world.runtime.device, dtype=torch.float32).contiguous()
    raw_prepared = getattr(selected_fused_ops, FUSED_PREPARE_OP_NAME)(
        world.runtime.word_offsets_i32,
        world.runtime.word_owner_i32,
        world.runtime.source_site_ids_i64,
        world.runtime.node_physical_length_f32,
        world.compact_site_rgba_f32,
        world.node_chart_f32,
        row_node_time_f32,
        row_near_far_f32,
        row_ray_coeff_f32,
        compact_positions0_f32,
        compact_velocities_f32,
        compact_weight_coefficients_f32,
        global_site_count=world.runtime.global_site_count,
        physical_length_epsilon=world.runtime.physical_length_epsilon,
        minimum_absolute_cut_denominator=minimum_absolute_cut_denominator,
        minimum_cut_cosine=minimum_cut_cosine,
        minimum_coordinate_length=minimum_coordinate_length,
        minimum_ray_speed=minimum_ray_speed,
        depth_closure_relative_tolerance=depth_closure_relative_tolerance,
        active_tie_relative_tolerance=active_tie_relative_tolerance,
    )
    row_identity_digests = tuple(row.row_identity_digest for row in rows)
    generation_id = _fused_generation_id(
        world=world,
        lowering=lowering,
        sources=normalized_sources,
        fused_ops=selected_fused_ops,
        raw_prepared=raw_prepared,
        certificate_digests=certificate_digests,
        row_identity_digests=row_identity_digests,
    )
    result = KineticNativeEqualRankFusedDirectFullVjpV1(
        world=world,
        lowering=lowering,
        sources=normalized_sources,
        fused_ops=selected_fused_ops,
        raw_prepared=raw_prepared,
        continuous_owner_certificate_digests=certificate_digests,
        row_identity_digests=row_identity_digests,
        world_identity=id(world),
        lowering_identity=id(lowering),
        lowering_generation_id=lowering.generation_digest,
        source_identities=tuple(id(source) for source in normalized_sources),
        fused_ops_identity=id(selected_fused_ops),
        fused_abi_identity=fused_abi_identity,
        raw_prepared_identity=id(raw_prepared),
        memory=_fused_vjp_memory(raw_prepared),
        generation_id=generation_id,
        _sealed_generation_id=generation_id,
        _seal=_FUSED_VJP_SEAL,
    )
    result.assert_cold_current()
    return result


@torch.no_grad()
def execute_kinetic_native_equal_rank_fused_direct_full_vjp_v1(
    prepared: KineticNativeEqualRankFusedDirectFullVjpV1,
    grad_node_chart_f32: torch.Tensor,
    grad_site_rgba_f32: torch.Tensor,
    grad_global_positions0_f32: torch.Tensor,
    grad_global_velocities_f32: torch.Tensor,
    grad_global_weight_coefficients_f32: torch.Tensor,
) -> Any:
    """Launch legacy one-block fused v1 and accept its scalar receipt.

    The raw suffixed ABI deliberately returns four caller-owned aliases plus a
    one-element int32 validation receipt.  This provenance-bearing boundary
    must consume ``accepted_bars()`` before exposing the result: merely
    receiving the aliases is not evidence that the validation grid admitted
    the launch.  A rejected launch raises here, before any higher-level request
    can commit the bars into its world accumulator.  This compatibility route
    does not provide the transaction token's single-use, manifest binding, or
    abort-quarantine contract and is not trainer-authorizable.
    """

    _assert_fused_process_not_quarantined()
    if not isinstance(prepared, KineticNativeEqualRankFusedDirectFullVjpV1):
        raise TypeError("prepared must be KineticNativeEqualRankFusedDirectFullVjpV1")
    prepared.assert_cold_current()
    raw_result = getattr(prepared.fused_ops, FUSED_VJP_OP_NAME)(
        prepared.raw_prepared,
        grad_node_chart_f32,
        grad_site_rgba_f32,
        grad_global_positions0_f32,
        grad_global_velocities_f32,
        grad_global_weight_coefficients_f32,
    )
    accept = getattr(raw_result, "accepted_bars", None)
    if not callable(accept):
        raise RuntimeError(
            "fused fixed-camera VJP result has no validation-receipt acceptance boundary"
        )
    accepted_bars = accept()
    expected_bars = (
        grad_site_rgba_f32,
        grad_global_positions0_f32,
        grad_global_velocities_f32,
        grad_global_weight_coefficients_f32,
    )
    if (
        not isinstance(accepted_bars, tuple)
        or len(accepted_bars) != len(expected_bars)
        or any(
            returned is not expected
            for returned, expected in zip(
                accepted_bars,
                expected_bars,
                strict=True,
            )
        )
    ):
        raise RuntimeError(
            "accepted fused fixed-camera VJP bars must be the four exact caller tensors"
        )
    raw_bar_names = (
        "grad_site_rgba_f32",
        "grad_global_positions0_f32",
        "grad_global_velocities_f32",
        "grad_global_weight_coefficients_f32",
    )
    if any(
        getattr(raw_result, name, None) is not expected
        for name, expected in zip(raw_bar_names, expected_bars, strict=True)
    ):
        raise RuntimeError(
            "fused fixed-camera VJP public bars must be the four exact caller tensors"
        )
    if (
        getattr(raw_result, "accumulation_enqueued", None) is not True
        or getattr(raw_result, "finalization_enqueued", None) is not True
        or getattr(raw_result, "shared_status_reused", None) is not False
        or getattr(raw_result, "runtime_status", None)
        != "raw_fixed_camera_source_only_until_native_rebuild_and_sparse_oracle_parity"
    ):
        raise RuntimeError(
            "sealed combined fused fixed-camera VJP returned the wrong phase/status contract"
        )
    validation_status = getattr(raw_result, "validation_status_i32", None)
    if (
        not isinstance(validation_status, torch.Tensor)
        or validation_status.device != prepared.world.runtime.device
        or validation_status.dtype != torch.int32
        or tuple(validation_status.shape) != (1,)
        or not validation_status.is_contiguous()
        or validation_status.numel() * validation_status.element_size() != 4
    ):
        raise RuntimeError(
            "accepted fused fixed-camera VJP lost its scalar int32 validation receipt"
        )
    if int(validation_status.item()) != 0:
        raise RuntimeError(
            "accepted fused fixed-camera VJP retained a nonzero validation reason mask"
        )
    prepared.assert_cold_current()
    return raw_result


def _assert_fused_transaction_block_sequence(
    blocks: tuple[KineticNativeEqualRankFusedDirectFullVjpV1, ...],
) -> None:
    if not blocks or any(
        not isinstance(block, KineticNativeEqualRankFusedDirectFullVjpV1)
        for block in blocks
    ):
        raise TypeError("transaction requires sealed fused v1 blocks")
    for block in blocks:
        block.assert_cold_current()
    active_ids = tuple(
        block.world.runtime.payload.block.generation_digest for block in blocks
    )
    if len(set(active_ids)) != len(active_ids) or len({id(block) for block in blocks}) != len(
        blocks
    ):
        raise ValueError("transaction blocks contain a duplicate active generation")
    first = blocks[0]
    if any(
        block.fused_ops is not first.fused_ops
        or block.fused_abi_identity != first.fused_abi_identity
        or block.lowering is not first.lowering
        or block.lowering_identity != first.lowering_identity
        or block.lowering_generation_id != first.lowering_generation_id
        or block.lowering.site_namespace_digest != first.lowering.site_namespace_digest
        or block.source_identities != first.source_identities
        or block.world.runtime.device != first.world.runtime.device
        or block.world.runtime.global_site_count
        != first.world.runtime.global_site_count
        or getattr(block.raw_prepared, "weight_coefficient_count", None)
        != getattr(first.raw_prepared, "weight_coefficient_count", None)
        for block in blocks[1:]
    ):
        raise ValueError("transaction blocks do not share one sealed geometry namespace")


def _fused_transaction_output_bars(
    state: _KineticNativeEqualRankFusedDirectFullVjpV1TransactionState,
) -> tuple[torch.Tensor, ...]:
    globals_ = (
        state.grad_global_positions0_f32,
        state.grad_global_velocities_f32,
        state.grad_global_weight_coefficients_f32,
    )
    if any(not isinstance(tensor, torch.Tensor) for tensor in globals_):
        raise ValueError("transaction global scratch was released or forged")
    return (*state.grad_compact_site_rgba_f32_by_block, *globals_)


def _fused_transaction_generation_id(
    transaction: KineticNativeEqualRankFusedDirectFullVjpV1Transaction,
) -> str:
    return ":".join(
        (
            ADAPTER_PROVENANCE,
            "fused-direct-full-vjp-transaction-v1",
            repr(transaction.active_block_generation_ids),
            repr(transaction.prepared_block_generation_ids),
            repr(transaction.prepared_block_identities),
            repr(transaction.node_bar_signatures),
            repr(transaction.output_bar_signatures),
            str(transaction.total_output_scratch_tensor_bytes),
            str(transaction.output_scratch_tensor_byte_budget),
            str(id(transaction._state)),
        )
    )


def _assert_fused_transaction_ready(
    transaction: KineticNativeEqualRankFusedDirectFullVjpV1Transaction,
) -> None:
    state = transaction._state
    blocks = state.prepared_blocks
    _assert_fused_transaction_block_sequence(blocks)
    active_ids = tuple(
        block.world.runtime.payload.block.generation_digest for block in blocks
    )
    output_bars = _fused_transaction_output_bars(state)
    all_bars = (*state.grad_node_chart_f32_by_block, *output_bars)
    if any(not isinstance(tensor, torch.Tensor) for tensor in all_bars):
        raise TypeError("transaction launch values must remain tensors")
    for block, tensor in zip(
        blocks,
        state.grad_node_chart_f32_by_block,
        strict=True,
    ):
        _require_warm_tensor(
            tensor,
            name="transaction_grad_node_chart_f32",
            device=block.world.runtime.device,
            dtype=torch.float32,
            shape=(block.world.row_count, block.world.node_count, 4),
        )
    output_storage = tuple(tensor.untyped_storage().data_ptr() for tensor in output_bars)
    node_storage = tuple(
        tensor.untyped_storage().data_ptr()
        for tensor in state.grad_node_chart_f32_by_block
    )
    raw_storage = {
        tensor.untyped_storage().data_ptr()
        for block in blocks
        for tensor in _fused_raw_prepared_tensors(block.raw_prepared)
    }
    if (
        transaction._seal is not _FUSED_VJP_TRANSACTION_SEAL
        or state.consumed
        or state.launch_attempt_count != 0
        or state.validation_status_i32 is not None
        or state.settled
        or state.quarantined
        or transaction.active_block_generation_ids != active_ids
        or transaction.prepared_block_generation_ids
        != tuple(block.generation_id for block in blocks)
        or transaction.prepared_block_identities != tuple(id(block) for block in blocks)
        or len(state.grad_node_chart_f32_by_block) != len(blocks)
        or tuple(
            _warm_tensor_signature(tensor)
            for tensor in state.grad_node_chart_f32_by_block
        )
        != transaction.node_bar_signatures
        or tuple(_warm_tensor_signature(tensor) for tensor in output_bars)
        != transaction.output_bar_signatures
        or len(set(output_storage)) != len(output_storage)
        or len(set(node_storage)) != len(node_storage)
        or set(output_storage) & (set(node_storage) | raw_storage)
        or set(node_storage) & raw_storage
        or transaction.compact_output_scratch_tensor_bytes
        != _tensor_bytes(state.grad_compact_site_rgba_f32_by_block)
        or transaction.global_output_scratch_tensor_bytes
        != _tensor_bytes(output_bars[-3:])
        or transaction.total_output_scratch_tensor_bytes != _tensor_bytes(output_bars)
        or transaction.total_output_scratch_tensor_bytes
        > transaction.output_scratch_tensor_byte_budget
        or transaction.output_scratch_tensor_count != len(blocks) + 3
        or not transaction.output_scratch_owned_by_token
        or not transaction.exact_zero_output_scratch_allocated
        or not transaction.duplicate_active_block_generations_rejected
        or transaction.active_manifest_coverage_certified
        or not transaction.single_use_scratch_generation_certified
        or transaction.hidden_output_alias_absence_certified
        or transaction.allocator_storage_bytes_measured
        or transaction.trainer_promotion_complete
        or transaction.runtime_status != FUSED_VJP_RUNTIME_STATUS
        or transaction.generation_id != _fused_transaction_generation_id(transaction)
    ):
        raise ValueError("fused transaction scratch token changed, aliased, or was consumed")


@torch.no_grad()
def prepare_kinetic_native_equal_rank_fused_direct_full_vjp_transaction_v1(
    prepared_blocks: Sequence[KineticNativeEqualRankFusedDirectFullVjpV1],
    grad_node_chart_f32_by_block: Sequence[torch.Tensor],
    *,
    max_output_scratch_tensor_bytes: int,
) -> KineticNativeEqualRankFusedDirectFullVjpV1Transaction:
    """Allocate and seal fresh zero scratch for one ordered one-shot transaction."""

    _assert_fused_process_not_quarantined()
    blocks = tuple(prepared_blocks)
    node_bars = tuple(grad_node_chart_f32_by_block)
    _assert_fused_transaction_block_sequence(blocks)
    if len(node_bars) != len(blocks) or any(
        not isinstance(tensor, torch.Tensor) for tensor in node_bars
    ):
        raise TypeError("transaction requires one node cotangent tensor per block")
    for block, tensor in zip(blocks, node_bars, strict=True):
        _require_warm_tensor(
            tensor,
            name="transaction_grad_node_chart_f32",
            device=block.world.runtime.device,
            dtype=torch.float32,
            shape=(block.world.row_count, block.world.node_count, 4),
        )
    device = blocks[0].world.runtime.device
    global_site_count = blocks[0].world.runtime.global_site_count
    weight_count = int(getattr(blocks[0].raw_prepared, "weight_coefficient_count"))
    required_output_bytes = 16 * sum(
        block.world.compact_site_count for block in blocks
    ) + 4 * global_site_count * (6 + weight_count)
    if (
        isinstance(max_output_scratch_tensor_bytes, bool)
        or not isinstance(max_output_scratch_tensor_bytes, int)
        or max_output_scratch_tensor_bytes < required_output_bytes
    ):
        raise ValueError("transaction output scratch exceeds its logical byte budget")
    compact_bars = tuple(
        torch.zeros((block.world.compact_site_count, 4), dtype=torch.float32, device=device)
        for block in blocks
    )
    global_bars = (
        torch.zeros((global_site_count, 3), dtype=torch.float32, device=device),
        torch.zeros((global_site_count, 3), dtype=torch.float32, device=device),
        torch.zeros((global_site_count, weight_count), dtype=torch.float32, device=device),
    )
    state = _KineticNativeEqualRankFusedDirectFullVjpV1TransactionState(
        prepared_blocks=blocks,
        grad_node_chart_f32_by_block=node_bars,
        grad_compact_site_rgba_f32_by_block=compact_bars,
        grad_global_positions0_f32=global_bars[0],
        grad_global_velocities_f32=global_bars[1],
        grad_global_weight_coefficients_f32=global_bars[2],
    )
    output_bars = (*compact_bars, *global_bars)
    transaction = KineticNativeEqualRankFusedDirectFullVjpV1Transaction(
        _state=state,
        active_block_generation_ids=tuple(
            block.world.runtime.payload.block.generation_digest for block in blocks
        ),
        prepared_block_generation_ids=tuple(block.generation_id for block in blocks),
        prepared_block_identities=tuple(id(block) for block in blocks),
        node_bar_signatures=tuple(_warm_tensor_signature(tensor) for tensor in node_bars),
        output_bar_signatures=tuple(_warm_tensor_signature(tensor) for tensor in output_bars),
        compact_output_scratch_tensor_bytes=_tensor_bytes(compact_bars),
        global_output_scratch_tensor_bytes=_tensor_bytes(global_bars),
        total_output_scratch_tensor_bytes=_tensor_bytes(output_bars),
        output_scratch_tensor_byte_budget=max_output_scratch_tensor_bytes,
        output_scratch_tensor_count=len(output_bars),
        generation_id="",
        _seal=_FUSED_VJP_TRANSACTION_SEAL,
    )
    object.__setattr__(transaction, "generation_id", _fused_transaction_generation_id(transaction))
    transaction.assert_ready()
    return transaction


@torch.no_grad()
def execute_kinetic_native_equal_rank_fused_direct_full_vjp_transaction_v1(
    transaction: KineticNativeEqualRankFusedDirectFullVjpV1Transaction,
    *,
    device_completion_fence: Callable[[], None],
    device_completion_fence_provenance: str,
) -> KineticNativeEqualRankFusedDirectFullVjpV1TransactionResult:
    """Validate every active block before the first fused atomic write.

    This is the provenance-bearing split-phase boundary needed by a request
    spanning multiple equal-rank blocks.  Validation dispatches share one
    reason mask; only the first scans the shared global geometry ledgers, while
    every block scans its distinct compact material ledger.  Admission requires
    every visible scratch element to be finite and exactly zero.  All
    validations, all guarded accumulations, and all postwrite ledger finalizers
    are enqueued in that order on the same stream; there is no host read or
    fence between them.  The caller-supplied completion fence runs once, after
    every launch, before the receipt is accepted.  A finalizer rejection
    quarantines mutated scratch; it does not claim byte-for-byte rollback,
    hidden-alias absence, exact active-manifest coverage, optimizer integration,
    or optimizer fail-atomicity.
    """

    _assert_fused_process_not_quarantined()
    if not isinstance(
        transaction,
        KineticNativeEqualRankFusedDirectFullVjpV1Transaction,
    ):
        raise TypeError("transaction must be a prepared fused v1 transaction token")
    transaction.assert_ready()
    if not callable(device_completion_fence):
        raise TypeError("device_completion_fence must be callable")
    if not isinstance(device_completion_fence_provenance, str) or not (
        device_completion_fence_provenance.strip()
    ):
        raise ValueError("device_completion_fence_provenance must be nonempty")
    state = transaction._state
    blocks = state.prepared_blocks
    node_bars = state.grad_node_chart_f32_by_block
    compact_bars = state.grad_compact_site_rgba_f32_by_block
    first = blocks[0]
    device = first.world.runtime.device
    global_bars = (
        state.grad_global_positions0_f32,
        state.grad_global_velocities_f32,
        state.grad_global_weight_coefficients_f32,
    )
    launch_bars = (*compact_bars, *global_bars)
    if any(not isinstance(tensor, torch.Tensor) for tensor in launch_bars):
        raise TypeError("fused fixed-camera transaction scratch was released")
    launch_bar_storage = tuple(
        tensor.untyped_storage().data_ptr() for tensor in launch_bars
    )
    node_bar_storage = tuple(
        tensor.untyped_storage().data_ptr() for tensor in node_bars
    )
    raw_prepared_storage = {
        tensor.untyped_storage().data_ptr()
        for block in blocks
        for tensor in _fused_raw_prepared_tensors(block.raw_prepared)
    }
    init_status = getattr(first.fused_ops, FUSED_STATUS_INIT_OP_NAME, None)
    launch = getattr(first.fused_ops, FUSED_VJP_OP_NAME, None)
    if not callable(init_status) or not callable(launch):
        raise TypeError(
            "fused fixed-camera transaction requires status-init and split-phase wrappers"
        )
    state.device_completion_fence = device_completion_fence
    state.consumed = True
    state.launch_attempt_count += 1
    try:
        validation_status = init_status(first.world.compact_site_rgba_f32)
        if (
            not isinstance(validation_status, torch.Tensor)
            or validation_status.device != device
            or validation_status.dtype != torch.int32
            or tuple(validation_status.shape) != (1,)
            or not validation_status.is_contiguous()
            or validation_status.numel() * validation_status.element_size() != 4
            or validation_status.untyped_storage().data_ptr()
            in set(node_bar_storage) | set(launch_bar_storage) | raw_prepared_storage
        ):
            raise RuntimeError(
                "fused fixed-camera transaction status initializer lost its four-byte ABI"
            )
    except BaseException as error:
        state.failure = error
        state.failure_traceback = error.__traceback__
        state.quarantined = True
        state.completion_fence_call_count += 1
        try:
            init_abort_returned = device_completion_fence()
        except BaseException as completion_error:
            state.completion_unknown = True
            state.completion_failure = completion_error
            state.completion_failure_traceback = completion_error.__traceback__
            _FUSED_VJP_RESTART_REQUIRED_QUARANTINE.append(state)
            raise RuntimeError(
                "fused status initialization completion is unknown; restart required and scratch remains quarantined"
            ) from error
        state.settled = True
        if init_abort_returned is not None:
            raise RuntimeError(
                "fused status initialization abort fence violated its contract; scratch is quarantined"
            ) from error
        raise RuntimeError(
            "fused status initialization failed; device settled and scratch quarantined"
        ) from error
    state.launch_result_count += 1
    state.validation_status_i32 = validation_status

    def assert_split_result(
        raw_result: Any,
        *,
        compact_bar: torch.Tensor,
        accumulation_enqueued: bool,
        finalization_enqueued: bool,
    ) -> None:
        expected_bars = (compact_bar, *global_bars)
        returned_status = getattr(raw_result, "validation_status_i32", None)
        raw_bar_names = (
            "grad_site_rgba_f32",
            "grad_global_positions0_f32",
            "grad_global_velocities_f32",
            "grad_global_weight_coefficients_f32",
        )
        if (
            any(
                getattr(raw_result, name, None) is not expected
                for name, expected in zip(
                    raw_bar_names,
                    expected_bars,
                    strict=True,
                )
            )
            or getattr(raw_result, "accumulation_enqueued", None)
            is not accumulation_enqueued
            or getattr(raw_result, "finalization_enqueued", None)
            is not finalization_enqueued
            or getattr(raw_result, "shared_status_reused", None) is not True
            or getattr(raw_result, "runtime_status", None)
            != "raw_fixed_camera_source_only_until_native_rebuild_and_sparse_oracle_parity"
            or not isinstance(returned_status, torch.Tensor)
            or not _same_exact_view(returned_status, validation_status)
        ):
            raise RuntimeError(
                "fused fixed-camera split phase returned a foreign result/status contract"
            )

    try:
        for phase in ("validate", "accumulate", "finalize"):
            for block_index, (block, node_bar, compact_bar) in enumerate(
                zip(blocks, node_bars, compact_bars, strict=True)
            ):
                state.launch_attempt_count += 1
                phase_kwargs = {}
                if phase == "validate":
                    phase_kwargs["validate_shared_global_ledgers"] = block_index == 0
                elif phase == "finalize":
                    phase_kwargs["finalize_shared_global_ledgers"] = block_index == 0
                raw_result = launch(
                    block.raw_prepared,
                    node_bar,
                    compact_bar,
                    *global_bars,
                    validation_status_i32=validation_status,
                    launch_phase=phase,
                    **phase_kwargs,
                )
                state.launch_result_count += 1
                assert_split_result(
                    raw_result,
                    compact_bar=compact_bar,
                    accumulation_enqueued=phase == "accumulate",
                    finalization_enqueued=phase == "finalize",
                )
                del raw_result
    except BaseException as error:
        state.failure = error
        state.failure_traceback = error.__traceback__
        state.quarantined = True
        state.completion_fence_call_count += 1
        try:
            abort_returned = device_completion_fence()
        except BaseException as completion_error:
            state.completion_unknown = True
            state.completion_failure = completion_error
            state.completion_failure_traceback = completion_error.__traceback__
            _FUSED_VJP_RESTART_REQUIRED_QUARANTINE.append(state)
            raise RuntimeError(
                "fused transaction completion is unknown; restart required and scratch remains quarantined"
            ) from error
        state.settled = True
        if abort_returned is not None:
            raise RuntimeError(
                "fused transaction abort fence violated its contract; scratch is quarantined"
            ) from error
        state.validation_status_i32 = None
        raise RuntimeError(
            "fused transaction launch failed after enqueue; device settled and scratch quarantined"
        ) from error

    # The completion callback is an external Python capability.  Bind every
    # mutable transaction-owned reference and tensor version immediately before
    # granting that capability so a callback cannot rewrite accepted scratch or
    # the shared status after the device finalizers have run.
    state.completion_fence_call_count += 1
    pre_fence_callback_snapshot = _fused_completion_callback_snapshot(
        transaction,
        state,
        node_bars=node_bars,
        launch_bars=launch_bars,
        validation_status=validation_status,
    )
    try:
        returned = device_completion_fence()
    except BaseException as completion_error:
        state.quarantined = True
        state.completion_unknown = True
        state.completion_failure = completion_error
        state.completion_failure_traceback = completion_error.__traceback__
        _FUSED_VJP_RESTART_REQUIRED_QUARANTINE.append(state)
        raise RuntimeError(
            "fused transaction completion is unknown; restart required and scratch remains quarantined"
        ) from completion_error
    if returned is not None:
        state.settled = True
        state.quarantined = True
        raise RuntimeError("fused transaction completion fence returned a value; scratch quarantined")
    state.settled = True
    try:
        post_fence_callback_snapshot = _fused_completion_callback_snapshot(
            transaction,
            state,
            node_bars=node_bars,
            launch_bars=launch_bars,
            validation_status=validation_status,
        )
    except BaseException as snapshot_error:
        callback_error = RuntimeError(
            "fused transaction completion callback corrupted bound transaction state; scratch quarantined"
        )
        _quarantine_fused_completion_callback_mutation(
            state,
            blocks=blocks,
            node_bars=node_bars,
            compact_bars=compact_bars,
            global_bars=global_bars,
            error=callback_error,
        )
        raise callback_error from snapshot_error
    if post_fence_callback_snapshot != pre_fence_callback_snapshot:
        callback_error = RuntimeError(
            "fused transaction completion callback mutated bound transaction state; scratch quarantined"
        )
        _quarantine_fused_completion_callback_mutation(
            state,
            blocks=blocks,
            node_bars=node_bars,
            compact_bars=compact_bars,
            global_bars=global_bars,
            error=callback_error,
        )
        raise callback_error
    try:
        reason_mask = int(validation_status.item())
    except BaseException as status_error:
        state.quarantined = True
        state.failure = status_error
        state.failure_traceback = status_error.__traceback__
        raise RuntimeError("fused transaction status acceptance failed; scratch quarantined") from status_error
    state.validation_reason_mask = reason_mask
    if reason_mask != 0:
        state.quarantined = True
        state.validation_status_i32 = None
        raise RuntimeError(
            f"fused all-block transaction rejected its scratch with reason mask 0x{reason_mask:02x}"
        )
    try:
        for block in blocks:
            block.assert_cold_current()
    except BaseException as provenance_error:
        state.quarantined = True
        state.failure = provenance_error
        state.failure_traceback = provenance_error.__traceback__
        state.validation_status_i32 = None
        raise RuntimeError("fused transaction provenance changed; scratch quarantined") from provenance_error
    output_bar_signatures = tuple(_warm_tensor_signature(tensor) for tensor in launch_bars)
    try:
        result = KineticNativeEqualRankFusedDirectFullVjpV1TransactionResult(
            grad_compact_site_rgba_f32_by_block=compact_bars,
            grad_global_positions0_f32=global_bars[0],
            grad_global_velocities_f32=global_bars[1],
            grad_global_weight_coefficients_f32=global_bars[2],
            output_bar_signatures=output_bar_signatures,
            active_block_generation_ids=transaction.active_block_generation_ids,
            transaction_generation_id=transaction.generation_id,
            validation_reason_mask=reason_mask,
            validation_status_tensor_bytes_during_transaction=4,
            retained_output_tensor_bytes=_tensor_bytes(launch_bars),
            retained_device_tensor_count=len(launch_bars),
            block_count=len(blocks),
            validation_launch_count=len(blocks),
            accumulation_launch_count=len(blocks),
            finalization_launch_count=len(blocks),
            compact_ledger_validation_count=len(blocks),
            shared_global_ledger_validation_count=1,
            compact_ledger_finalization_count=len(blocks),
            shared_global_ledger_finalization_count=1,
            device_completion_fence_call_count=1,
            device_completion_fence_provenance=device_completion_fence_provenance,
            _seal=_FUSED_VJP_TRANSACTION_RESULT_SEAL,
        )
        result.assert_current()
    except BaseException as receipt_error:
        state.quarantined = True
        state.failure = receipt_error
        state.failure_traceback = receipt_error.__traceback__
        state.validation_status_i32 = None
        raise RuntimeError("fused transaction receipt sealing failed; scratch quarantined") from receipt_error
    state.accepted = True
    state.validation_status_i32 = None
    state.device_completion_fence = None
    state.prepared_blocks = ()
    state.grad_node_chart_f32_by_block = ()
    state.grad_compact_site_rgba_f32_by_block = ()
    state.grad_global_positions0_f32 = None
    state.grad_global_velocities_f32 = None
    state.grad_global_weight_coefficients_f32 = None
    return result


def _fused_union_abi_identity(fused_ops: Any) -> tuple[tuple[str, int], ...]:
    identities = []
    for name in (
        FUSED_UNION_PREPARE_OP_NAME,
        FUSED_UNION_STATUS_INIT_OP_NAME,
        FUSED_UNION_VJP_OP_NAME,
    ):
        value = getattr(fused_ops, name, None)
        implementation = getattr(value, "__func__", value)
        identities.append((name, id(implementation)))
    return tuple(identities)


def _certify_union_local_spatial_bundle_cold_current(spatial_bundle: Any) -> None:
    """Use the bundle-owned bounded D2H receipt for exact union-map reads."""

    from paper_kinetic_union_local_bar_assembly import (
        certify_paper_kinetic_union_local_spatial_bundle_cold_current,
    )

    certify_paper_kinetic_union_local_spatial_bundle_cold_current(spatial_bundle)


def _require_fused_union_ops(fused_ops: Any) -> tuple[tuple[str, int], ...]:
    identity = _fused_union_abi_identity(fused_ops)
    if any(not callable(getattr(fused_ops, name, None)) for name, _ in identity):
        raise TypeError("fused_ops does not expose the complete suffixed union-v2 ABI")
    return identity


def _fused_union_raw_prepared_tensors(raw: Any) -> tuple[torch.Tensor, ...]:
    direct = getattr(raw, "direct_v1_oracle", None)
    tensors = (
        *_fused_raw_prepared_tensors(direct),
        getattr(raw, "compact_to_geometry_output_i64", None),
        getattr(raw, "geometry_output_source_site_ids_i64", None),
        getattr(raw, "config_i32", None),
    )
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("raw union-v2 token has an invalid tensor contract")
    if getattr(raw, "mapping_tensor_owned_by_preparer", None) != (False, False):
        raise ValueError(
            "sealed union transaction requires resident bundle maps; raw-owned asynchronous copies are forbidden"
        )
    if getattr(direct, "tensor_owned_by_preparer", None) != (False,) * 12 + (
        True,
        True,
    ):
        raise ValueError("union-v2 inherited v1 tensors changed ownership")
    return tensors


def _fused_union_output_bars(
    state: _KineticNativeEqualRankFusedUnionFullVjpV2TransactionState,
) -> tuple[torch.Tensor, ...]:
    union_bars = (
        state.grad_union_positions0_f32,
        state.grad_union_velocities_f32,
        state.grad_union_weight_coefficients_f32,
    )
    if any(not isinstance(tensor, torch.Tensor) for tensor in union_bars):
        raise ValueError("union-v2 transaction scratch was released or forged")
    return (*state.grad_compact_site_rgba_f32_by_block, *union_bars)


def _fused_union_transaction_generation_id(
    transaction: KineticNativeEqualRankFusedUnionFullVjpV2Transaction,
) -> str:
    return ":".join(
        (
            ADAPTER_PROVENANCE,
            "fused-union-full-vjp-all-block-transaction-v2",
            transaction.active_block_manifest_generation_id,
            transaction.spatial_bundle_generation_digest,
            repr(transaction.active_block_generation_ids),
            repr(transaction.prepared_block_generation_ids),
            repr(transaction.geometry_output_source_site_ids),
            repr(transaction.compact_to_geometry_output_by_block),
            repr(transaction.compact_map_generation_digests),
            repr(transaction.union_abi_identity),
            str(transaction.output_bar_scratch_tensor_bytes),
            str(transaction.validation_status_tensor_bytes_during_execution),
            str(transaction.total_transaction_scratch_tensor_bytes),
            str(transaction.transaction_scratch_tensor_byte_budget),
            str(id(transaction._state)),
        )
    )


def _assert_fused_union_transaction_ready(
    transaction: KineticNativeEqualRankFusedUnionFullVjpV2Transaction,
) -> None:
    _assert_fused_process_not_quarantined()
    state = transaction._state
    blocks = state.prepared_blocks
    raw_blocks = state.raw_union_blocks
    bundle = state.spatial_bundle
    construction_lifetime = state.construction_lifetime
    if not isinstance(
        construction_lifetime,
        KineticNativeEqualRankFusedUnionFullVjpV2ConstructionLifetime,
    ):
        raise TypeError("union-v2 transaction lost its construction lifetime")
    construction_lifetime.assert_retained()
    _assert_fused_transaction_block_sequence(blocks)
    _certify_union_local_spatial_bundle_cold_current(bundle)
    active_ids = tuple(
        block.world.runtime.payload.block.generation_digest for block in blocks
    )
    bindings = tuple(bundle.native_blocks)
    maps = tuple(binding.compact_to_union_i64 for binding in bindings)
    output_bars = _fused_union_output_bars(state)
    all_bars = (*state.grad_node_chart_f32_by_block, *output_bars)
    raw_tensors = tuple(
        tensor for raw in raw_blocks for tensor in _fused_union_raw_prepared_tensors(raw)
    )
    raw_storage = {tensor.untyped_storage().data_ptr() for tensor in raw_tensors}
    bar_storage = tuple(tensor.untyped_storage().data_ptr() for tensor in all_bars)
    if (
        transaction._seal is not _FUSED_UNION_VJP_TRANSACTION_SEAL
        or state.consumed
        or state.settled
        or state.quarantined
        or state.validation_status_i32 is not None
        or id(bundle) != transaction.spatial_bundle_identity
        or construction_lifetime.phase != "transferred"
        or construction_lifetime.transaction is not transaction
        or bundle.generation_digest != transaction.spatial_bundle_generation_digest
        or transaction.active_block_manifest_generation_id != bundle.generation_digest
        or transaction.active_block_generation_ids != active_ids
        or active_ids
        != tuple(binding.native_block_generation_digest for binding in bindings)
        or transaction.prepared_block_generation_ids
        != tuple(block.generation_id for block in blocks)
        or transaction.prepared_block_identities != tuple(id(block) for block in blocks)
        or transaction.raw_union_block_identities != tuple(id(raw) for raw in raw_blocks)
        or transaction.union_abi_identity != _fused_union_abi_identity(blocks[0].fused_ops)
        or transaction.geometry_output_source_site_ids != bundle.union_source_site_ids
        or transaction.compact_to_geometry_output_by_block
        != bundle.compact_to_union_by_block
        or transaction.compact_map_generation_digests
        != tuple(binding.mapping_generation_digest for binding in bindings)
        or state.resident_union_source_site_ids_i64 is not bundle.source_site_ids_i64
        or transaction.union_identity_signature
        != _warm_tensor_signature(bundle.source_site_ids_i64)
        or transaction.compact_map_signatures
        != tuple(_warm_tensor_signature(tensor) for tensor in maps)
        or transaction.node_bar_signatures
        != tuple(_warm_tensor_signature(tensor) for tensor in state.grad_node_chart_f32_by_block)
        or transaction.output_bar_signatures
        != tuple(_warm_tensor_signature(tensor) for tensor in output_bars)
        or len(set(bar_storage)) != len(bar_storage)
        or set(bar_storage) & raw_storage
        or state.union_transfer_predecessor is not None
        or transaction.output_bar_scratch_tensor_bytes != _tensor_bytes(output_bars)
        or transaction.validation_status_tensor_bytes_during_execution != 4
        or transaction.total_transaction_scratch_tensor_bytes
        != transaction.output_bar_scratch_tensor_bytes + 4
        or transaction.total_transaction_scratch_tensor_bytes
        > transaction.transaction_scratch_tensor_byte_budget
        or transaction.union_site_count != bundle.union_site_count
        or transaction.block_count != len(blocks)
        or not transaction.exact_active_block_manifest_certified
        or not transaction.exact_union_identity_certified
        or not transaction.exact_factorization_certified
        or transaction.output_geometry_index_space != "request_union"
        or transaction.material_output_index_space != "block_compact"
        or transaction.union_material_finiteness_certified
        or not transaction.single_use
        or transaction.persistent_or_global_write_authorized
        or transaction.optimizer_write_authorized
        or transaction.bounded_batch_q is not None
        or transaction.trainer_promotion_complete
        or transaction.runtime_status != FUSED_UNION_VJP_RUNTIME_STATUS
        or transaction.generation_id != _fused_union_transaction_generation_id(transaction)
    ):
        raise ValueError("union-v2 all-block transaction changed, aliased, or was consumed")
    for block, raw, binding, node_bar in zip(
        blocks, raw_blocks, bindings, state.grad_node_chart_f32_by_block, strict=True
    ):
        _require_warm_tensor(
            node_bar,
            name="union_v2_grad_node_chart_f32",
            device=block.world.runtime.device,
            dtype=torch.float32,
            shape=(block.world.row_count, block.world.node_count, 4),
        )
        if (
            raw.geometry_output_source_site_ids_i64 is not bundle.source_site_ids_i64
            or raw.compact_to_geometry_output_i64 is not binding.compact_to_union_i64
            or raw.union_site_count != bundle.union_site_count
            or raw.global_site_count != bundle.global_site_count
        ):
            raise ValueError("raw union-v2 block escaped its sealed bundle identity")


@torch.no_grad()
def prepare_kinetic_native_equal_rank_fused_union_full_vjp_construction_lifetime_v2(
    prepared_blocks: Sequence[KineticNativeEqualRankFusedDirectFullVjpV1],
    grad_node_chart_f32_by_block: Sequence[torch.Tensor],
    *,
    spatial_bundle: Any,
    active_block_manifest_generation_id: str,
    max_transaction_scratch_tensor_bytes: int,
) -> KineticNativeEqualRankFusedUnionFullVjpV2ConstructionLifetime:
    """Validate every scalar/map/count before the first device allocation."""

    _assert_fused_process_not_quarantined()
    from paper_kinetic_union_local_bar_assembly import (
        PaperKineticUnionLocalSpatialBundle,
    )

    if not isinstance(spatial_bundle, PaperKineticUnionLocalSpatialBundle):
        raise TypeError("spatial_bundle must be a sealed union-local spatial bundle")
    _certify_union_local_spatial_bundle_cold_current(spatial_bundle)
    if (
        not isinstance(active_block_manifest_generation_id, str)
        or active_block_manifest_generation_id != spatial_bundle.generation_digest
    ):
        raise ValueError("active manifest generation must exactly equal the sealed bundle")
    blocks = tuple(prepared_blocks)
    node_bars = tuple(grad_node_chart_f32_by_block)
    _assert_fused_transaction_block_sequence(blocks)
    active_ids = tuple(
        block.world.runtime.payload.block.generation_digest for block in blocks
    )
    bindings = tuple(spatial_bundle.native_blocks)
    if active_ids != tuple(binding.native_block_generation_digest for binding in bindings):
        raise ValueError("prepared blocks are not the exact canonical active bundle order")
    if len(node_bars) != len(blocks):
        raise ValueError("union-v2 requires one node cotangent per active block")
    first = blocks[0]
    if (
        spatial_bundle.device != first.world.runtime.device
        or spatial_bundle.global_site_count != first.world.runtime.global_site_count
    ):
        raise ValueError("union-v2 bundle and fused blocks do not share one world/device")
    union_ids = spatial_bundle.source_site_ids_i64
    union_storage = union_ids.untyped_storage().data_ptr()
    union_abi_identity = _require_fused_union_ops(first.fused_ops)
    union_count = spatial_bundle.union_site_count
    weight_count = int(first.raw_prepared.weight_coefficient_count)
    if union_count < 1 or not 1 <= weight_count <= 3:
        raise ValueError("union-v2 U/C dimensions are invalid")
    required_output_bar_bytes = 16 * sum(
        block.world.compact_site_count for block in blocks
    ) + (
        4 * union_count * (6 + weight_count)
    )
    required_transaction_bytes = required_output_bar_bytes + 4
    if (
        isinstance(max_transaction_scratch_tensor_bytes, bool)
        or not isinstance(max_transaction_scratch_tensor_bytes, int)
        or max_transaction_scratch_tensor_bytes < required_transaction_bytes
    ):
        raise ValueError("union-v2 bars plus sticky status exceed the transaction budget")
    compact_maps: list[tuple[int, ...]] = []
    thresholds_by_block: list[tuple[float, ...]] = []
    for block, binding, node_bar, compact_map in zip(
        blocks,
        bindings,
        node_bars,
        spatial_bundle.compact_to_union_by_block,
        strict=True,
    ):
        _require_warm_tensor(
            node_bar,
            name="union_v2_grad_node_chart_f32",
            device=first.world.runtime.device,
            dtype=torch.float32,
            shape=(block.world.row_count, block.world.node_count, 4),
        )
        raw_v1 = block.raw_prepared
        compact_ids = tuple(int(value) for value in raw_v1.source_site_ids_i64.tolist())
        if (
            compact_ids != binding.compact_source_site_ids
            or len(compact_map) != block.world.compact_site_count
            or any(index < 0 or index >= union_count for index in compact_map)
            or tuple(spatial_bundle.union_source_site_ids[index] for index in compact_map)
            != compact_ids
            or len(
                {
                    raw_v1.source_site_ids_i64.untyped_storage().data_ptr(),
                    binding.compact_to_union_i64.untyped_storage().data_ptr(),
                    union_storage,
                }
            )
            != 3
        ):
            raise ValueError("union-v2 block does not prove P_b=P_U Q_b")
        thresholds = tuple(float(value) for value in raw_v1.config_f32.tolist())
        if (
            len(thresholds) != 7
            or any(not math.isfinite(value) or value < 0.0 for value in thresholds)
            or thresholds[0] == 0.0
            or thresholds[1] == 0.0
            or thresholds[2] == 0.0
            or thresholds[5] == 0.0
            or thresholds[5] > 1.0
            or thresholds[6] == 0.0
            or int(raw_v1.weight_coefficient_count) != weight_count
        ):
            raise ValueError("union-v2 inherited threshold/C scalars are invalid")
        compact_maps.append(compact_map)
        thresholds_by_block.append(thresholds)
    lifetime = KineticNativeEqualRankFusedUnionFullVjpV2ConstructionLifetime(
        prepared_blocks=blocks,
        grad_node_chart_f32_by_block=node_bars,
        spatial_bundle=spatial_bundle,
        active_block_manifest_generation_id=active_block_manifest_generation_id,
        max_transaction_scratch_tensor_bytes=max_transaction_scratch_tensor_bytes,
        active_block_generation_ids=active_ids,
        compact_to_geometry_output_by_block=tuple(compact_maps),
        thresholds_f32_by_block=tuple(thresholds_by_block),
        union_abi_identity=union_abi_identity,
        required_output_bar_tensor_bytes=required_output_bar_bytes,
        required_validation_status_tensor_bytes=4,
        required_transaction_scratch_tensor_bytes=required_transaction_bytes,
        union_site_count=union_count,
        weight_coefficient_count=weight_count,
        prepared_block_identities=tuple(id(block) for block in blocks),
        node_bar_signatures=tuple(_warm_tensor_signature(tensor) for tensor in node_bars),
        spatial_bundle_identity=id(spatial_bundle),
        spatial_bundle_generation_digest=spatial_bundle.generation_digest,
        union_identity_signature=_warm_tensor_signature(union_ids),
        compact_map_signatures=tuple(
            _warm_tensor_signature(binding.compact_to_union_i64) for binding in bindings
        ),
        compact_map_generation_digests=tuple(
            binding.mapping_generation_digest for binding in bindings
        ),
        raw_union_blocks=[None] * len(blocks),
        output_tensors=[None] * (len(blocks) + 3),
        _seal=_FUSED_UNION_VJP_CONSTRUCTION_LIFETIME_SEAL,
    )
    lifetime.assert_retained()
    return lifetime


def _settle_failed_fused_union_construction(
    lifetime: KineticNativeEqualRankFusedUnionFullVjpV2ConstructionLifetime,
    *,
    fence: Callable[[], None],
    fence_provenance: str,
    error: BaseException,
) -> None:
    lifetime.phase = "quarantined"
    lifetime.quarantined = True
    lifetime.failure = error
    lifetime.construction_completion_fence_call_count += 1
    lifetime.construction_completion_fence_provenance = fence_provenance
    quarantine_roots = (
        lifetime,
        lifetime.prepared_blocks,
        lifetime.grad_node_chart_f32_by_block,
        lifetime.spatial_bundle,
        tuple(raw for raw in lifetime.raw_union_blocks if raw is not None),
        tuple(
            tensor
            for tensor in lifetime.output_tensors
            if isinstance(tensor, torch.Tensor)
        ),
        lifetime.transaction,
    )
    _retain_fused_union_rejected_roots(quarantine_roots)
    before = (
        id(lifetime),
        lifetime.phase,
        tuple(id(block) for block in lifetime.prepared_blocks),
        tuple(id(tensor) for tensor in lifetime.grad_node_chart_f32_by_block),
        id(lifetime.spatial_bundle),
        tuple(id(raw) for raw in lifetime.raw_union_blocks),
        tuple(id(tensor) for tensor in lifetime.output_tensors),
        id(lifetime.transaction),
        lifetime.current_raw_block_index,
        lifetime.construction_completion_fence_call_count,
        lifetime.construction_completion_fence_provenance,
        lifetime.quarantined,
        lifetime.completion_unknown,
        id(lifetime.failure),
    )
    try:
        returned = fence()
    except BaseException as completion_error:
        lifetime.completion_unknown = True
        lifetime.completion_failure = completion_error
        _FUSED_VJP_RESTART_REQUIRED_QUARANTINE.append(quarantine_roots)
        raise RuntimeError(
            "union-v2 partial construction completion is unknown; restart required"
        ) from completion_error
    after = (
        id(lifetime),
        lifetime.phase,
        tuple(id(block) for block in lifetime.prepared_blocks),
        tuple(id(tensor) for tensor in lifetime.grad_node_chart_f32_by_block),
        id(lifetime.spatial_bundle),
        tuple(id(raw) for raw in lifetime.raw_union_blocks),
        tuple(id(tensor) for tensor in lifetime.output_tensors),
        id(lifetime.transaction),
        lifetime.current_raw_block_index,
        lifetime.construction_completion_fence_call_count,
        lifetime.construction_completion_fence_provenance,
        lifetime.quarantined,
        lifetime.completion_unknown,
        id(lifetime.failure),
    )
    if after != before:
        raise RuntimeError(
            "union-v2 partial construction fence mutated bound roots; quarantine retained"
        ) from error
    lifetime.settled = True
    if returned is not None:
        raise RuntimeError(
            "union-v2 partial construction fence returned a value; roots quarantined"
        ) from error


@torch.no_grad()
def materialize_kinetic_native_equal_rank_fused_union_full_vjp_transaction_v2(
    lifetime: KineticNativeEqualRankFusedUnionFullVjpV2ConstructionLifetime,
    *,
    construction_completion_fence: Callable[[], None],
    construction_completion_fence_provenance: str,
) -> KineticNativeEqualRankFusedUnionFullVjpV2Transaction:
    """Publish each returned device object before the next allocation."""

    _assert_fused_process_not_quarantined()
    if not isinstance(
        lifetime,
        KineticNativeEqualRankFusedUnionFullVjpV2ConstructionLifetime,
    ):
        raise TypeError("lifetime must be a union-v2 construction lifetime")
    lifetime.assert_retained()
    if lifetime.phase != "installed":
        raise ValueError("union-v2 construction lifetime was already used")
    if not callable(construction_completion_fence):
        raise TypeError("construction_completion_fence must be callable")
    if (
        not isinstance(construction_completion_fence_provenance, str)
        or not construction_completion_fence_provenance.strip()
    ):
        raise ValueError("construction_completion_fence_provenance must be nonempty")
    lifetime.phase = "materializing"
    blocks = lifetime.prepared_blocks
    node_bars = lifetime.grad_node_chart_f32_by_block
    bundle = lifetime.spatial_bundle
    bindings = tuple(bundle.native_blocks)
    union_ids = bundle.source_site_ids_i64
    raw_prepare = getattr(blocks[0].fused_ops, FUSED_UNION_PREPARE_OP_NAME)
    try:
        for block_index, (block, binding, thresholds) in enumerate(
            zip(blocks, bindings, lifetime.thresholds_f32_by_block, strict=True)
        ):
            lifetime.current_raw_block_index = block_index
            raw_v1 = block.raw_prepared
            raw = raw_prepare(
                raw_v1.word_offsets_i32,
                raw_v1.word_owner_i32,
                raw_v1.source_site_ids_i64,
                binding.compact_to_union_i64,
                union_ids,
                raw_v1.node_physical_length_f32,
                raw_v1.site_rgba_f32,
                raw_v1.node_chart_f32,
                raw_v1.row_node_time_f32,
                raw_v1.row_near_far_f32,
                raw_v1.row_ray_coeff_f32,
                raw_v1.compact_positions0_f32,
                raw_v1.compact_velocities_f32,
                raw_v1.compact_weight_coefficients_f32,
                global_site_count=bundle.global_site_count,
                physical_length_epsilon=thresholds[0],
                minimum_absolute_cut_denominator=thresholds[1],
                minimum_ray_speed=thresholds[2],
                depth_closure_relative_tolerance=thresholds[3],
                active_tie_relative_tolerance=thresholds[4],
                minimum_cut_cosine=thresholds[5],
                minimum_coordinate_length=thresholds[6],
            )
            lifetime.raw_union_blocks[block_index] = raw
            _fused_union_raw_prepared_tensors(raw)
            if (
                raw.geometry_output_source_site_ids_i64 is not union_ids
                or raw.compact_to_geometry_output_i64 is not binding.compact_to_union_i64
            ):
                raise ValueError("union-v2 raw preparer copied a sealed resident identity")
        lifetime.current_raw_block_index = None
        device = blocks[0].world.runtime.device
        for block_index, block in enumerate(blocks):
            compact_bar = torch.zeros(
                (block.world.compact_site_count, 4),
                dtype=torch.float32,
                device=device,
            )
            lifetime.output_tensors[block_index] = compact_bar
        for union_bar_index, shape in enumerate((
            (lifetime.union_site_count, 3),
            (lifetime.union_site_count, 3),
            (lifetime.union_site_count, lifetime.weight_coefficient_count),
        )):
            union_bar = torch.zeros(shape, dtype=torch.float32, device=device)
            lifetime.output_tensors[len(blocks) + union_bar_index] = union_bar
        if any(raw is None for raw in lifetime.raw_union_blocks) or any(
            not isinstance(tensor, torch.Tensor) for tensor in lifetime.output_tensors
        ):
            raise RuntimeError("union-v2 construction publication slots are incomplete")
        block_count = len(blocks)
        compact_bars = tuple(lifetime.output_tensors[:block_count])
        union_bars = tuple(lifetime.output_tensors[block_count:])
        output_bars = (*compact_bars, *union_bars)
        raw_blocks = tuple(lifetime.raw_union_blocks)
        state = _KineticNativeEqualRankFusedUnionFullVjpV2TransactionState(
            prepared_blocks=blocks,
            spatial_bundle=bundle,
            construction_lifetime=lifetime,
            raw_union_blocks=raw_blocks,
            grad_node_chart_f32_by_block=node_bars,
            grad_compact_site_rgba_f32_by_block=compact_bars,
            grad_union_positions0_f32=union_bars[0],
            grad_union_velocities_f32=union_bars[1],
            grad_union_weight_coefficients_f32=union_bars[2],
            resident_union_source_site_ids_i64=union_ids,
            union_transfer_predecessor=None,
        )
        transaction = KineticNativeEqualRankFusedUnionFullVjpV2Transaction(
            _state=state,
            active_block_manifest_generation_id=(
                lifetime.active_block_manifest_generation_id
            ),
            spatial_bundle_identity=lifetime.spatial_bundle_identity,
            spatial_bundle_generation_digest=lifetime.spatial_bundle_generation_digest,
            active_block_generation_ids=lifetime.active_block_generation_ids,
            prepared_block_generation_ids=tuple(block.generation_id for block in blocks),
            prepared_block_identities=lifetime.prepared_block_identities,
            geometry_output_source_site_ids=bundle.union_source_site_ids,
            compact_to_geometry_output_by_block=(
                lifetime.compact_to_geometry_output_by_block
            ),
            raw_union_block_identities=tuple(id(raw) for raw in raw_blocks),
            union_abi_identity=lifetime.union_abi_identity,
            union_identity_signature=lifetime.union_identity_signature,
            compact_map_signatures=lifetime.compact_map_signatures,
            compact_map_generation_digests=lifetime.compact_map_generation_digests,
            node_bar_signatures=lifetime.node_bar_signatures,
            output_bar_signatures=tuple(
                _warm_tensor_signature(tensor) for tensor in output_bars
            ),
            output_bar_scratch_tensor_bytes=_tensor_bytes(output_bars),
            validation_status_tensor_bytes_during_execution=4,
            total_transaction_scratch_tensor_bytes=_tensor_bytes(output_bars) + 4,
            transaction_scratch_tensor_byte_budget=(
                lifetime.max_transaction_scratch_tensor_bytes
            ),
            union_site_count=lifetime.union_site_count,
            block_count=block_count,
            generation_id="",
            _seal=_FUSED_UNION_VJP_TRANSACTION_SEAL,
        )
        lifetime.transaction = transaction
        object.__setattr__(
            transaction,
            "generation_id",
            _fused_union_transaction_generation_id(transaction),
        )
        lifetime.phase = "transferred"
        transaction.assert_ready()
        return transaction
    except BaseException as construction_error:
        _settle_failed_fused_union_construction(
            lifetime,
            fence=construction_completion_fence,
            fence_provenance=construction_completion_fence_provenance,
            error=construction_error,
        )
        raise RuntimeError(
            "union-v2 partial construction failed; all published roots quarantined"
        ) from construction_error


def prepare_kinetic_native_equal_rank_fused_union_full_vjp_transaction_v2(
    lifetime: KineticNativeEqualRankFusedUnionFullVjpV2ConstructionLifetime,
    *,
    construction_completion_fence: Callable[[], None],
    construction_completion_fence_provenance: str,
) -> KineticNativeEqualRankFusedUnionFullVjpV2Transaction:
    """Two-phase compatibility name; the lifetime must already be caller-owned."""

    return materialize_kinetic_native_equal_rank_fused_union_full_vjp_transaction_v2(
        lifetime,
        construction_completion_fence=construction_completion_fence,
        construction_completion_fence_provenance=(
            construction_completion_fence_provenance
        ),
    )


def _fused_union_callback_snapshot(
    transaction: KineticNativeEqualRankFusedUnionFullVjpV2Transaction,
    validation_status: torch.Tensor,
) -> tuple[object, ...]:
    state = transaction._state
    construction_lifetime = state.construction_lifetime
    return (
        id(state),
        id(state.spatial_bundle),
        id(construction_lifetime),
        construction_lifetime.phase,
        tuple(id(raw) for raw in construction_lifetime.raw_union_blocks),
        tuple(id(tensor) for tensor in construction_lifetime.output_tensors),
        id(construction_lifetime.transaction),
        construction_lifetime.current_raw_block_index,
        construction_lifetime.construction_completion_fence_call_count,
        construction_lifetime.quarantined,
        construction_lifetime.completion_unknown,
        construction_lifetime.settled,
        state.spatial_bundle.generation_digest,
        tuple(
            binding.mapping_generation_digest
            for binding in state.spatial_bundle.native_blocks
        ),
        tuple(id(block) for block in state.prepared_blocks),
        tuple(id(raw) for raw in state.raw_union_blocks),
        tuple(
            _callback_tensor_binding(tensor)
            for raw in state.raw_union_blocks
            for tensor in _fused_union_raw_prepared_tensors(raw)
        ),
        tuple(_callback_tensor_binding(tensor) for tensor in state.grad_node_chart_f32_by_block),
        tuple(_callback_tensor_binding(tensor) for tensor in _fused_union_output_bars(state)),
        _callback_tensor_binding(validation_status),
        id(state.validation_status_i32),
        id(state.device_completion_fence),
        id(state.resident_union_source_site_ids_i64),
        state.consumed,
        state.accepted,
        state.quarantined,
        state.completion_unknown,
        state.completion_fence_call_count,
        state.validation_launch_count,
        state.accumulation_launch_count,
        state.finalization_launch_count,
        state.compact_ledger_validation_count,
        state.shared_union_ledger_validation_count,
        state.compact_ledger_finalization_count,
        state.shared_union_ledger_finalization_count,
        id(state.failure) if state.failure is not None else None,
        id(state.completion_failure)
        if state.completion_failure is not None
        else None,
        transaction.generation_id,
        _fused_union_transaction_generation_id(transaction),
        transaction._seal,
    )


def _settle_failed_fused_union_transaction(
    state: _KineticNativeEqualRankFusedUnionFullVjpV2TransactionState,
    fence: Callable[[], None],
    error: BaseException,
) -> None:
    state.consumed = True
    state.quarantined = True
    state.failure = error
    _retain_fused_union_rejected_roots(state)
    state.completion_fence_call_count += 1
    try:
        returned = fence()
    except BaseException as completion_error:
        state.completion_unknown = True
        state.completion_failure = completion_error
        _FUSED_VJP_RESTART_REQUIRED_QUARANTINE.append(state)
        raise RuntimeError(
            "union-v2 completion is unknown; process restart is required and every root remains quarantined"
        ) from completion_error
    state.settled = True
    if returned is not None:
        raise RuntimeError(
            "union-v2 failure fence returned a value; every transaction root remains quarantined"
        ) from error


@torch.no_grad()
def execute_kinetic_native_equal_rank_fused_union_full_vjp_transaction_v2(
    transaction: KineticNativeEqualRankFusedUnionFullVjpV2Transaction,
    *,
    device_completion_fence: Callable[[], None],
    device_completion_fence_provenance: str,
) -> KineticNativeEqualRankFusedUnionFullVjpV2TransactionResult:
    """Validate all, accumulate all, finalize all, then accept after one fence."""

    _assert_fused_process_not_quarantined()
    if not isinstance(transaction, KineticNativeEqualRankFusedUnionFullVjpV2Transaction):
        raise TypeError("transaction must be a sealed union-v2 transaction")
    transaction.assert_ready()
    if not callable(device_completion_fence):
        raise TypeError("device_completion_fence must be callable")
    if not isinstance(device_completion_fence_provenance, str) or not device_completion_fence_provenance.strip():
        raise ValueError("device_completion_fence_provenance must be nonempty")
    state = transaction._state
    construction_lifetime = state.construction_lifetime
    state.device_completion_fence = device_completion_fence
    state.consumed = True
    blocks = state.prepared_blocks
    raw_blocks = state.raw_union_blocks
    node_bars = state.grad_node_chart_f32_by_block
    compact_bars = state.grad_compact_site_rgba_f32_by_block
    output_bars = _fused_union_output_bars(state)
    union_bars = output_bars[-3:]
    first = blocks[0]
    init_status = getattr(first.fused_ops, FUSED_UNION_STATUS_INIT_OP_NAME, None)
    launch = getattr(first.fused_ops, FUSED_UNION_VJP_OP_NAME, None)
    bound_storage = {
        tensor.untyped_storage().data_ptr()
        for tensor in (
            *node_bars,
            *output_bars,
            *tuple(
                tensor
                for raw in raw_blocks
                for tensor in _fused_union_raw_prepared_tensors(raw)
            ),
        )
    }
    try:
        validation_status = init_status(first.world.compact_site_rgba_f32)
        if (
            not isinstance(validation_status, torch.Tensor)
            or validation_status.device != first.world.runtime.device
            or validation_status.dtype != torch.int32
            or tuple(validation_status.shape) != (1,)
            or not validation_status.is_contiguous()
            or validation_status.numel() * validation_status.element_size() != 4
            or validation_status.untyped_storage().data_ptr() in bound_storage
        ):
            raise RuntimeError("union-v2 status initializer lost its four-byte ABI")
        state.validation_status_i32 = validation_status
        for phase in ("validate", "accumulate", "finalize"):
            for block_index, (raw, node_bar, compact_bar) in enumerate(
                zip(raw_blocks, node_bars, compact_bars, strict=True)
            ):
                kwargs: dict[str, bool] = {}
                if phase == "validate":
                    kwargs["validate_shared_union_ledgers"] = block_index == 0
                elif phase == "finalize":
                    kwargs["finalize_shared_union_ledgers"] = block_index == 0
                raw_result = launch(
                    raw,
                    node_bar,
                    compact_bar,
                    *union_bars,
                    validation_status_i32=validation_status,
                    launch_phase=phase,
                    **kwargs,
                )
                if (
                    getattr(raw_result, "grad_site_rgba_f32", None) is not compact_bar
                    or getattr(raw_result, "grad_union_positions0_f32", None) is not union_bars[0]
                    or getattr(raw_result, "grad_union_velocities_f32", None) is not union_bars[1]
                    or getattr(raw_result, "grad_union_weight_coefficients_f32", None) is not union_bars[2]
                    or not _same_exact_view(raw_result.validation_status_i32, validation_status)
                    or raw_result.accumulation_enqueued is not (phase == "accumulate")
                    or raw_result.finalization_enqueued is not (phase == "finalize")
                    or raw_result.shared_status_reused is not True
                    or raw_result.geometry_output_index_space != "request_union"
                    or raw_result.runtime_status
                    != "raw_union_v2_source_only_until_native_rebuild_v1_sparse_parity_and_allocator_evidence"
                ):
                    raise RuntimeError("union-v2 split phase returned a foreign contract")
                if phase == "validate":
                    state.validation_launch_count += 1
                    state.compact_ledger_validation_count += 1
                    state.shared_union_ledger_validation_count += int(block_index == 0)
                elif phase == "accumulate":
                    state.accumulation_launch_count += 1
                else:
                    state.finalization_launch_count += 1
                    state.compact_ledger_finalization_count += 1
                    state.shared_union_ledger_finalization_count += int(block_index == 0)
                del raw_result
    except BaseException as launch_error:
        _settle_failed_fused_union_transaction(state, device_completion_fence, launch_error)
        raise RuntimeError("union-v2 launch failed; every root remains quarantined") from launch_error
    state.completion_fence_call_count += 1
    before = _fused_union_callback_snapshot(transaction, validation_status)
    try:
        returned = device_completion_fence()
    except BaseException as completion_error:
        state.quarantined = True
        state.completion_unknown = True
        state.completion_failure = completion_error
        _FUSED_VJP_RESTART_REQUIRED_QUARANTINE.append(state)
        raise RuntimeError(
            "union-v2 completion is unknown; process restart is required and every root remains quarantined"
        ) from completion_error
    state.settled = True
    if returned is not None:
        state.quarantined = True
        _retain_fused_union_rejected_roots(state)
        raise RuntimeError("union-v2 completion fence returned a value; roots quarantined")
    try:
        after = _fused_union_callback_snapshot(transaction, validation_status)
    except BaseException as snapshot_error:
        state.quarantined = True
        state.failure = snapshot_error
        _retain_fused_union_rejected_roots(state)
        raise RuntimeError("union-v2 completion callback corrupted bound roots") from snapshot_error
    if after != before:
        state.quarantined = True
        state.failure = RuntimeError("union-v2 completion callback mutated bound roots")
        _retain_fused_union_rejected_roots(state)
        raise state.failure
    try:
        reason_mask = int(validation_status.item())
        state.validation_reason_mask = reason_mask
        if reason_mask != 0:
            raise RuntimeError(
                f"union-v2 all-block transaction rejected scratch with reason mask 0x{reason_mask:03x}"
            )
        construction_lifetime.assert_retained()
        _certify_union_local_spatial_bundle_cold_current(state.spatial_bundle)
        for block, raw in zip(blocks, raw_blocks, strict=True):
            block.assert_cold_current()
            _fused_union_raw_prepared_tensors(raw)
        if (
            state.validation_launch_count != len(blocks)
            or state.accumulation_launch_count != len(blocks)
            or state.finalization_launch_count != len(blocks)
            or state.compact_ledger_validation_count != len(blocks)
            or state.shared_union_ledger_validation_count != 1
            or state.compact_ledger_finalization_count != len(blocks)
            or state.shared_union_ledger_finalization_count != 1
        ):
            raise RuntimeError("union-v2 split-phase launch accounting changed")
    except BaseException as acceptance_error:
        state.quarantined = True
        state.failure = acceptance_error
        _retain_fused_union_rejected_roots(state)
        raise RuntimeError("union-v2 post-fence acceptance failed; roots quarantined") from acceptance_error
    try:
        result_state = _KineticNativeEqualRankFusedUnionFullVjpV2ResultState(
            compact_bars=compact_bars,
            union_position_bar=union_bars[0],
            union_velocity_bar=union_bars[1],
            union_weight_bar=union_bars[2],
        )
        result = KineticNativeEqualRankFusedUnionFullVjpV2TransactionResult(
            _state=result_state,
            geometry_output_source_site_ids=transaction.geometry_output_source_site_ids,
            active_block_manifest_generation_id=transaction.active_block_manifest_generation_id,
            active_block_generation_ids=transaction.active_block_generation_ids,
            transaction_generation_id=transaction.generation_id,
            output_bar_signatures=tuple(
                _warm_tensor_signature(tensor) for tensor in output_bars
            ),
            validation_reason_mask=reason_mask,
            validation_status_tensor_bytes_during_transaction=4,
            retained_validation_status_tensor_bytes=0,
            retained_output_tensor_bytes=_tensor_bytes(output_bars),
            retained_device_tensor_count=len(output_bars),
            block_count=len(blocks),
            union_site_count=transaction.union_site_count,
            validation_launch_count=state.validation_launch_count,
            accumulation_launch_count=state.accumulation_launch_count,
            finalization_launch_count=state.finalization_launch_count,
            compact_ledger_validation_count=state.compact_ledger_validation_count,
            shared_union_ledger_validation_count=(
                state.shared_union_ledger_validation_count
            ),
            compact_ledger_finalization_count=state.compact_ledger_finalization_count,
            shared_union_ledger_finalization_count=(
                state.shared_union_ledger_finalization_count
            ),
            device_completion_fence_call_count=1,
            device_completion_fence_provenance=device_completion_fence_provenance,
            material_output_index_space="block_compact",
            union_material_finiteness_certified=False,
            _seal=_FUSED_UNION_VJP_TRANSACTION_RESULT_SEAL,
        )
        result.assert_current()
    except BaseException as receipt_error:
        state.quarantined = True
        state.failure = receipt_error
        _retain_fused_union_rejected_roots(state)
        raise RuntimeError("union-v2 receipt sealing failed; roots quarantined") from receipt_error
    try:
        construction_lifetime.transaction = None
        construction_lifetime.raw_union_blocks.clear()
        construction_lifetime.output_tensors.clear()
        construction_lifetime.prepared_blocks = ()
        construction_lifetime.grad_node_chart_f32_by_block = ()
        construction_lifetime.spatial_bundle = None
        construction_lifetime.active_block_generation_ids = ()
        construction_lifetime.compact_to_geometry_output_by_block = ()
        construction_lifetime.thresholds_f32_by_block = ()
        construction_lifetime.union_abi_identity = ()
        construction_lifetime.prepared_block_identities = ()
        construction_lifetime.node_bar_signatures = ()
        construction_lifetime.union_identity_signature = ()
        construction_lifetime.compact_map_signatures = ()
        construction_lifetime.compact_map_generation_digests = ()
        construction_lifetime.current_raw_block_index = None
        construction_lifetime.phase = "released"
        construction_lifetime.assert_retained()
    except BaseException as release_error:
        state.quarantined = True
        state.failure = release_error
        _retain_fused_union_rejected_roots(state)
        raise RuntimeError(
            "union-v2 accepted construction-root release failed; transaction quarantined"
        ) from release_error
    state.accepted = True
    state.validation_status_i32 = None
    state.device_completion_fence = None
    state.construction_lifetime = None
    state.prepared_blocks = ()
    state.raw_union_blocks = ()
    state.grad_node_chart_f32_by_block = ()
    state.grad_compact_site_rgba_f32_by_block = ()
    state.grad_union_positions0_f32 = None
    state.grad_union_velocities_f32 = None
    state.grad_union_weight_coefficients_f32 = None
    state.resident_union_source_site_ids_i64 = None
    state.union_transfer_predecessor = None
    state.spatial_bundle = None
    return result


@torch.no_grad()
def execute_kinetic_native_equal_rank_node_vjp(
    world: KineticNativeEqualRankWorld,
    grad_node_chart_f32: torch.Tensor,
    *,
    compact_grad_site_rgba_f32: torch.Tensor,
    global_grad_site_rgba_f32: torch.Tensor | None = None,
) -> KineticNativeEqualRankVJPResult:
    """Run one bounded VJP and optionally scatter into one global bar.

    ``compact_grad_site_rgba_f32`` is caller-owned scratch.  It is reset before
    launch so the optional global scatter always adds exactly this invocation's
    contribution, never a prior invocation's prefix.  Reusing the same global
    accumulator across blocks therefore sums repeated global site ids exactly.
    """

    if not isinstance(world, KineticNativeEqualRankWorld):
        raise TypeError("world must be KineticNativeEqualRankWorld")
    world.assert_warm_layout()
    runtime = world.runtime
    _require_warm_tensor(
        grad_node_chart_f32,
        name="grad_node_chart_f32",
        device=runtime.device,
        dtype=torch.float32,
        shape=(runtime.row_count, runtime.node_count, 4),
    )
    _require_warm_tensor(
        compact_grad_site_rgba_f32,
        name="compact_grad_site_rgba_f32",
        device=runtime.device,
        dtype=torch.float32,
        shape=(runtime.compact_site_count, 4),
    )
    if any(
        _same_storage(compact_grad_site_rgba_f32, tensor)
        for tensor in (
            world.compact_site_rgba_f32,
            world.node_chart_f32,
            grad_node_chart_f32,
        )
    ):
        raise ValueError("compact material bar must not alias world or node-gradient state")
    if global_grad_site_rgba_f32 is not None:
        _require_warm_tensor(
            global_grad_site_rgba_f32,
            name="global_grad_site_rgba_f32",
            device=runtime.device,
            dtype=torch.float32,
            shape=(runtime.global_site_count, 4),
        )
        if any(
            _same_storage(global_grad_site_rgba_f32, tensor)
            for tensor in (
                compact_grad_site_rgba_f32,
                world.compact_site_rgba_f32,
                world.node_chart_f32,
                grad_node_chart_f32,
            )
        ):
            raise ValueError("global material bar must not alias compact/world/node-gradient state")
    grad_node_signature = _warm_tensor_signature(grad_node_chart_f32)
    compact_grad_site_rgba_f32.zero_()
    native_result = getattr(runtime.native_ops, VJP_OP_NAME)(
        runtime.word_offsets_i32,
        runtime.word_owner_i32,
        runtime.node_physical_length_f32,
        world.compact_site_rgba_f32,
        world.node_chart_f32,
        grad_node_chart_f32,
        compact_grad_site_rgba_f32,
        runtime.config_i32,
        runtime.config_f32,
        track_count=runtime.row_count,
        node_count=runtime.node_count,
    )
    if not isinstance(native_result, tuple) or len(native_result) != 2:
        raise TypeError("equal-rank native VJP must return (aliased material bar, length bar)")
    returned_compact_bar, grad_lengths = native_result
    if not isinstance(returned_compact_bar, torch.Tensor) or not isinstance(grad_lengths, torch.Tensor):
        raise TypeError("equal-rank native VJP outputs must be tensors")
    if _warm_tensor_signature(grad_node_chart_f32) != grad_node_signature:
        raise ValueError("equal-rank native VJP mutated its read-only node-gradient input")
    _require_warm_tensor(
        returned_compact_bar,
        name="native returned compact material bar",
        device=runtime.device,
        dtype=torch.float32,
        shape=(runtime.compact_site_count, 4),
    )
    if not _same_exact_view(returned_compact_bar, compact_grad_site_rgba_f32):
        raise ValueError("equal-rank native VJP must alias the supplied compact material bar")
    _require_warm_tensor(
        grad_lengths,
        name="native grad_node_physical_length_f32",
        device=runtime.device,
        dtype=torch.float32,
        shape=(runtime.node_count, runtime.word_count),
    )
    if any(
        _same_storage(grad_lengths, tensor)
        for tensor in (
            compact_grad_site_rgba_f32,
            world.compact_site_rgba_f32,
            world.node_chart_f32,
            grad_node_chart_f32,
        )
    ) or (
        global_grad_site_rgba_f32 is not None
        and _same_storage(grad_lengths, global_grad_site_rgba_f32)
    ):
        raise ValueError("native length bar must own storage distinct from launch/material state")
    if global_grad_site_rgba_f32 is not None:
        global_grad_site_rgba_f32.index_add_(
            0,
            runtime.source_site_ids_i64,
            compact_grad_site_rgba_f32,
        )
    tensors = (
        (compact_grad_site_rgba_f32, grad_lengths)
        if global_grad_site_rgba_f32 is None
        else (compact_grad_site_rgba_f32, global_grad_site_rgba_f32, grad_lengths)
    )
    result = KineticNativeEqualRankVJPResult(
        world=world,
        grad_compact_site_rgba_f32=compact_grad_site_rgba_f32,
        grad_global_site_rgba_f32=global_grad_site_rgba_f32,
        grad_node_physical_length_f32=grad_lengths,
        warm_tensor_signatures=tuple(_warm_tensor_signature(tensor) for tensor in tensors),
        world_identity=id(world),
        accounting=_vjp_accounting(
            world,
            compact_grad_site_rgba_f32,
            global_grad_site_rgba_f32,
            grad_lengths,
        ),
        _seal=_VJP_RESULT_SEAL,
    )
    result.assert_warm_layout()
    return result


@torch.no_grad()
def execute_kinetic_native_equal_rank_material_node_vjp(
    world: KineticNativeEqualRankWorld,
    grad_node_chart_f32: torch.Tensor,
    *,
    compact_grad_site_rgba_f32: torch.Tensor,
    global_grad_site_rgba_f32: torch.Tensor | None = None,
) -> KineticNativeEqualRankMaterialVJPResult:
    """Run material-only reverse without allocating the geometry ``[J,W]`` bar."""

    if not isinstance(world, KineticNativeEqualRankWorld):
        raise TypeError("world must be KineticNativeEqualRankWorld")
    world.assert_warm_layout()
    runtime = world.runtime
    for tensor, name, shape in (
        (
            grad_node_chart_f32,
            "grad_node_chart_f32",
            (runtime.row_count, runtime.node_count, 4),
        ),
        (
            compact_grad_site_rgba_f32,
            "compact_grad_site_rgba_f32",
            (runtime.compact_site_count, 4),
        ),
    ):
        _require_warm_tensor(
            tensor,
            name=name,
            device=runtime.device,
            dtype=torch.float32,
            shape=shape,
        )
    if any(
        _same_storage(compact_grad_site_rgba_f32, tensor)
        for tensor in (world.compact_site_rgba_f32, world.node_chart_f32, grad_node_chart_f32)
    ):
        raise ValueError("compact material bar must not alias world or node-gradient state")
    if global_grad_site_rgba_f32 is not None:
        _require_warm_tensor(
            global_grad_site_rgba_f32,
            name="global_grad_site_rgba_f32",
            device=runtime.device,
            dtype=torch.float32,
            shape=(runtime.global_site_count, 4),
        )
        if any(
            _same_storage(global_grad_site_rgba_f32, tensor)
            for tensor in (
                compact_grad_site_rgba_f32,
                world.compact_site_rgba_f32,
                world.node_chart_f32,
                grad_node_chart_f32,
            )
        ):
            raise ValueError("global material bar must not alias compact/world/node-gradient state")
    grad_node_signature = _warm_tensor_signature(grad_node_chart_f32)
    compact_grad_site_rgba_f32.zero_()
    returned_compact_bar = getattr(runtime.native_ops, MATERIAL_VJP_OP_NAME)(
        runtime.word_offsets_i32,
        runtime.word_owner_i32,
        runtime.node_physical_length_f32,
        world.compact_site_rgba_f32,
        world.node_chart_f32,
        grad_node_chart_f32,
        compact_grad_site_rgba_f32,
        runtime.config_i32,
        runtime.config_f32,
        track_count=runtime.row_count,
        node_count=runtime.node_count,
    )
    if not isinstance(returned_compact_bar, torch.Tensor):
        raise TypeError("equal-rank native material VJP must return one tensor")
    if _warm_tensor_signature(grad_node_chart_f32) != grad_node_signature:
        raise ValueError("equal-rank native material VJP mutated its read-only node-gradient input")
    _require_warm_tensor(
        returned_compact_bar,
        name="native returned compact material bar",
        device=runtime.device,
        dtype=torch.float32,
        shape=(runtime.compact_site_count, 4),
    )
    if not _same_exact_view(returned_compact_bar, compact_grad_site_rgba_f32):
        raise ValueError("equal-rank native material VJP must alias the supplied compact material bar")
    if global_grad_site_rgba_f32 is not None:
        global_grad_site_rgba_f32.index_add_(
            0,
            runtime.source_site_ids_i64,
            compact_grad_site_rgba_f32,
        )
    tensors = (
        (compact_grad_site_rgba_f32,)
        if global_grad_site_rgba_f32 is None
        else (compact_grad_site_rgba_f32, global_grad_site_rgba_f32)
    )
    result = KineticNativeEqualRankMaterialVJPResult(
        world=world,
        grad_compact_site_rgba_f32=compact_grad_site_rgba_f32,
        grad_global_site_rgba_f32=global_grad_site_rgba_f32,
        warm_tensor_signatures=tuple(_warm_tensor_signature(tensor) for tensor in tensors),
        world_identity=id(world),
        accounting=_material_vjp_accounting(
            world,
            compact_grad_site_rgba_f32,
            global_grad_site_rgba_f32,
        ),
        _seal=_MATERIAL_VJP_RESULT_SEAL,
    )
    result.assert_warm_layout()
    return result


def _material_vjp_accounting(
    world: KineticNativeEqualRankWorld,
    compact_bar: torch.Tensor,
    global_bar: torch.Tensor | None,
) -> dict[str, int | bool | str]:
    compact_bytes = _tensor_bytes((compact_bar,))
    global_bytes = 0 if global_bar is None else _tensor_bytes((global_bar,))
    return {
        "row_count": world.row_count,
        "compiler_node_count": world.node_count,
        "ordered_run_count": world.runtime.word_count,
        "ordered_run_node_interactions": world.node_count * world.runtime.word_count,
        "compact_site_count": world.compact_site_count,
        "global_site_count": world.runtime.global_site_count,
        "compact_material_bar_tensor_bytes": compact_bytes,
        "global_material_bar_tensor_bytes": global_bytes,
        "node_physical_length_bar_tensor_bytes": 0,
        "caller_owned_material_bar_tensor_bytes": compact_bytes + global_bytes,
        "logical_result_tensor_bytes": compact_bytes + global_bytes,
        "adapter_allocated_compact_material_bar_bytes": 0,
        "adapter_allocated_global_material_bar_bytes": 0,
        "native_vjp_output_length_bar_bytes": 0,
        "global_scatter_performed": global_bar is not None,
        "global_scatter_kind": "index_add/cross_block_duplicate_ids_sum",
        "compact_bar_zeroed_before_native_vjp": True,
        "requested_frame_count_used": 0,
        "persistent_frame_tensor_bytes": 0,
        "persistent_sample_tensor_bytes": 0,
        "persistent_target_tensor_bytes": 0,
        "persistent_prediction_tensor_bytes": 0,
        "frame_by_run_reverse_state_allocated": False,
        "reverse_scaling": "O(J * W_block)",
        "geometry_length_bar_returned": False,
        "geometry_parameter_vjp_implemented": False,
        "allocator_storage_bytes_measured": False,
        "allocator_peak_measured": False,
        "native_runtime_verified": False,
        "runtime_status": RUNTIME_STATUS,
    }


def _vjp_accounting(
    world: KineticNativeEqualRankWorld,
    compact_bar: torch.Tensor,
    global_bar: torch.Tensor | None,
    length_bar: torch.Tensor,
) -> dict[str, int | bool | str]:
    compact_bytes = _tensor_bytes((compact_bar,))
    global_bytes = 0 if global_bar is None else _tensor_bytes((global_bar,))
    length_bytes = _tensor_bytes((length_bar,))
    return {
        "row_count": world.row_count,
        "compiler_node_count": world.node_count,
        "ordered_run_count": world.runtime.word_count,
        "ordered_run_node_interactions": world.node_count * world.runtime.word_count,
        "compact_site_count": world.compact_site_count,
        "global_site_count": world.runtime.global_site_count,
        "compact_material_bar_tensor_bytes": compact_bytes,
        "global_material_bar_tensor_bytes": global_bytes,
        "node_physical_length_bar_tensor_bytes": length_bytes,
        "caller_owned_material_bar_tensor_bytes": compact_bytes + global_bytes,
        "logical_result_tensor_bytes": compact_bytes + global_bytes + length_bytes,
        "adapter_allocated_compact_material_bar_bytes": 0,
        "adapter_allocated_global_material_bar_bytes": 0,
        "native_vjp_output_length_bar_bytes": length_bytes,
        "global_scatter_performed": global_bar is not None,
        "global_scatter_kind": "index_add/cross_block_duplicate_ids_sum",
        "compact_bar_zeroed_before_native_vjp": True,
        "requested_frame_count_used": 0,
        "persistent_frame_tensor_bytes": 0,
        "persistent_sample_tensor_bytes": 0,
        "persistent_target_tensor_bytes": 0,
        "persistent_prediction_tensor_bytes": 0,
        "frame_by_run_reverse_state_allocated": False,
        "reverse_scaling": "O(J * W_block)",
        "geometry_length_bar_returned": True,
        "geometry_parameter_vjp_implemented": False,
        "allocator_storage_bytes_measured": False,
        "allocator_peak_measured": False,
        "native_runtime_verified": False,
        "runtime_status": RUNTIME_STATUS,
    }


def _fused_raw_prepared_tensors(raw_prepared: Any) -> tuple[torch.Tensor, ...]:
    tensors = tuple(
        getattr(raw_prepared, name, None)
        for name in _FUSED_RAW_PREPARED_TENSOR_NAMES
    )
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("raw fused fixed-camera token has an invalid tensor contract")
    return tensors


def _fused_vjp_memory(raw_prepared: Any) -> KineticNativeEqualRankFusedDirectFullVjpV1Memory:
    tensors = _fused_raw_prepared_tensors(raw_prepared)
    if len({tensor.untyped_storage().data_ptr() for tensor in tensors}) != len(tensors):
        raise ValueError("fused fixed-camera retained-byte accounting requires distinct tensors")
    ownership = getattr(raw_prepared, "tensor_owned_by_preparer", None)
    expected_raw_ownership = (False,) * 12 + (True, True)
    if ownership != expected_raw_ownership:
        raise ValueError(
            "sealed fused adapter requires resident topology aliases and only raw-owned configs"
        )
    topology_bytes = _tensor_bytes(tensors[:4])
    world_bytes = _tensor_bytes(tensors[4:6])
    row_payload_bytes = _tensor_bytes(tensors[6:12])
    config_bytes = _tensor_bytes(tensors[12:14])
    retained_bytes = topology_bytes + world_bytes + row_payload_bytes + config_bytes
    owned_bytes = row_payload_bytes + config_bytes
    if (
        getattr(raw_prepared, "retained_logical_tensor_bytes", None) != retained_bytes
        or getattr(raw_prepared, "preparer_owned_logical_tensor_bytes", None) != config_bytes
        or any(
            getattr(raw_prepared, name, None) != 0
            for name in (
                "persistent_frame_tensor_bytes",
                "persistent_sample_tensor_bytes",
                "persistent_target_tensor_bytes",
                "persistent_prediction_tensor_bytes",
            )
        )
    ):
        raise ValueError("raw fused fixed-camera logical-byte accounting changed")
    return KineticNativeEqualRankFusedDirectFullVjpV1Memory(
        aliased_runtime_topology_tensor_bytes=topology_bytes,
        aliased_runtime_world_tensor_bytes=world_bytes,
        owned_row_payload_tensor_bytes=row_payload_bytes,
        owned_config_tensor_bytes=config_bytes,
        retained_launch_tensor_bytes=retained_bytes,
        unique_retained_launch_tensor_bytes=retained_bytes,
        owned_persistent_tensor_bytes=owned_bytes,
    )


def _fused_block_sources(
    runtime: KineticNativeEqualRankRuntimeBlock,
    lowering: KineticNativeEqualRankLowering,
    sources: tuple[KineticNativeEqualRankChartSource, ...],
) -> tuple[tuple[Any, ...], tuple[KineticNativeEqualRankChartSource, ...], tuple[str, ...], Any]:
    """Resolve one sealed payload block to live rows, sources, and certificates."""

    from kinetic_native_equal_rank_sparse_geometry_reduction import (
        validate_kinetic_native_equal_rank_continuous_owner_certificate,
    )

    runtime.payload.assert_cold_current(lowering, sources)
    rows_by_index = {row.global_row_index: row for row in lowering.rows}
    rows = tuple(rows_by_index[index] for index in runtime.payload.block.global_row_indices)
    sources_by_identity = {source.row_identity: source for source in sources}
    if len(sources_by_identity) != len(sources):
        raise ValueError("fused fixed-camera VJP sources contain duplicate row identities")
    try:
        row_sources = tuple(sources_by_identity[row.row_identity] for row in rows)
    except KeyError as error:
        raise ValueError("fused fixed-camera VJP block lost a live chart source") from error
    for row, source in zip(rows, row_sources, strict=True):
        chart = source.program.charts[source.chart_index]
        topology = source.lowering.charts[source.chart_index]
        if (
            source.program.generation_digest != row.kinetic_program_generation_digest
            or source.lowering.generation_digest != row.topology_lowering_generation_digest
            or chart.node_count != row.node_count
            or chart.run_count != row.word_count
            or tuple(chart.owner_word) != row.owner_word
            or topology.owner_word != row.owner_word
            or topology.payload_digest != row.chart_payload_digest
            or topology.node_physical_lengths_digest != row.node_physical_lengths_digest
        ):
            raise ValueError("fused fixed-camera VJP row source/program provenance changed")
    sites = row_sources[0].program.binding.sites
    if any(source.program.binding.sites is not sites for source in row_sources):
        raise ValueError("one fused fixed-camera block cannot mix kinetic site tables")
    if sites.site_count != runtime.global_site_count:
        raise ValueError("fused fixed-camera geometry namespace changed")
    certificate_digests = tuple(
        validate_kinetic_native_equal_rank_continuous_owner_certificate(source, row)
        for source, row in zip(row_sources, rows, strict=True)
    )
    return rows, row_sources, certificate_digests, sites


def _require_fused_ops(fused_ops: Any) -> tuple[tuple[str, int], ...]:
    if fused_ops is None:
        raise TypeError("runtime.native_ops must expose the suffixed fused v1 wrappers")
    identity = _fused_abi_identity(fused_ops)
    for name, _implementation_id in identity:
        if not callable(getattr(fused_ops, name, None)):
            raise TypeError(f"fused_ops does not expose callable {name}")
    return identity


def _fused_abi_identity(fused_ops: Any) -> tuple[tuple[str, int], ...]:
    identities = []
    for name in (
        FUSED_PREPARE_OP_NAME,
        FUSED_STATUS_INIT_OP_NAME,
        FUSED_VJP_OP_NAME,
    ):
        callable_value = getattr(fused_ops, name, None)
        implementation = getattr(callable_value, "__func__", callable_value)
        identities.append((name, id(implementation)))
    return tuple(identities)


def _fused_generation_id(
    *,
    world: KineticNativeEqualRankWorld,
    lowering: KineticNativeEqualRankLowering,
    sources: tuple[KineticNativeEqualRankChartSource, ...],
    fused_ops: Any,
    raw_prepared: Any,
    certificate_digests: tuple[str, ...],
    row_identity_digests: tuple[str, ...],
) -> str:
    return ":".join(
        (
            ADAPTER_PROVENANCE,
            "fixed-camera-fused-direct-full-vjp-v1",
            world.generation_id,
            lowering.generation_digest,
            repr(tuple(id(source) for source in sources)),
            repr(_fused_abi_identity(fused_ops)),
            str(id(raw_prepared)),
            repr(row_identity_digests),
            repr(certificate_digests),
        )
    )


def _require_native_ops(
    native_ops: Any,
    *,
    device: torch.device,
) -> tuple[tuple[str, int], ...]:
    if native_ops is None:
        raise TypeError("native_ops must be injected explicitly")
    identity = _native_abi_identity(native_ops)
    for name, _implementation_id in identity:
        if not callable(getattr(native_ops, name, None)):
            raise TypeError(f"native_ops does not expose callable {name}")
    if device.type == "mps":
        attestation = getattr(
            native_ops,
            "assert_kinetic_memory_light_compiled_abi_registered",
            None,
        )
        if not callable(attestation):
            raise TypeError(
                "MPS native_ops must expose compiled kinetic ABI attestation"
            )
        attestation()
    return identity


def _native_abi_identity(native_ops: Any) -> tuple[tuple[str, int], ...]:
    identities = []
    for name in (FORWARD_OP_NAME, VJP_OP_NAME, MATERIAL_VJP_OP_NAME):
        callable_value = getattr(native_ops, name, None)
        implementation = getattr(callable_value, "__func__", callable_value)
        identities.append((name, id(implementation)))
    return tuple(identities)


def _copy_or_alias_tensor_with_lifetime(
    lifetime: KineticNativeEqualRankRuntimeConstructionLifetime,
    tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, bool]:
    """Publish each returned transfer object before the next operation."""

    if (
        tensor.device == device
        and tensor.dtype == dtype
        and tensor.layout == torch.strided
        and tensor.is_contiguous()
        and not tensor.requires_grad
    ):
        return tensor, False
    detached = tensor.detach()
    lifetime.transfer_intermediates.append(detached)
    transferred = detached.to(device=device, dtype=dtype)
    lifetime.transfer_intermediates.append(transferred)
    contiguous = transferred.contiguous()
    if contiguous is not transferred:
        lifetime.transfer_intermediates.append(contiguous)
    return contiguous, True


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
        or tensor.layout != torch.strided
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
        or tensor.requires_grad
    ):
        raise ValueError(f"{name} has an invalid device/dtype/layout/shape")


def _quarantine_fused_completion_callback_mutation(
    state: _KineticNativeEqualRankFusedDirectFullVjpV1TransactionState,
    *,
    blocks: Sequence[KineticNativeEqualRankFusedDirectFullVjpV1],
    node_bars: Sequence[torch.Tensor],
    compact_bars: Sequence[torch.Tensor],
    global_bars: Sequence[torch.Tensor],
    error: BaseException,
) -> None:
    """Restore every settled scratch root and reject callback-authored state."""

    state.prepared_blocks = tuple(blocks)
    state.grad_node_chart_f32_by_block = tuple(node_bars)
    state.grad_compact_site_rgba_f32_by_block = tuple(compact_bars)
    (
        state.grad_global_positions0_f32,
        state.grad_global_velocities_f32,
        state.grad_global_weight_coefficients_f32,
    ) = tuple(global_bars)
    state.validation_status_i32 = None
    state.device_completion_fence = None
    state.consumed = True
    state.completion_fence_call_count = 1
    state.settled = True
    state.accepted = False
    state.quarantined = True
    state.completion_unknown = False
    state.validation_reason_mask = None
    state.failure = error
    state.failure_traceback = error.__traceback__
    state.completion_failure = None
    state.completion_failure_traceback = None


def _fused_completion_callback_snapshot(
    transaction: KineticNativeEqualRankFusedDirectFullVjpV1Transaction,
    state: _KineticNativeEqualRankFusedDirectFullVjpV1TransactionState,
    *,
    node_bars: Sequence[torch.Tensor],
    launch_bars: Sequence[torch.Tensor],
    validation_status: torch.Tensor,
) -> tuple[object, ...]:
    """Bind callback-visible transaction structure, storage, and lifecycle."""

    return (
        id(transaction),
        id(transaction._state),
        id(state),
        tuple(id(block) for block in state.prepared_blocks),
        tuple(id(tensor) for tensor in state.grad_node_chart_f32_by_block),
        tuple(id(tensor) for tensor in state.grad_compact_site_rgba_f32_by_block),
        id(state.grad_global_positions0_f32),
        id(state.grad_global_velocities_f32),
        id(state.grad_global_weight_coefficients_f32),
        id(state.validation_status_i32),
        id(state.device_completion_fence),
        state.consumed,
        state.launch_attempt_count,
        state.launch_result_count,
        state.completion_fence_call_count,
        state.accepted,
        state.quarantined,
        state.completion_unknown,
        state.validation_reason_mask,
        id(state.failure) if state.failure is not None else None,
        id(state.failure_traceback) if state.failure_traceback is not None else None,
        id(state.completion_failure)
        if state.completion_failure is not None
        else None,
        id(state.completion_failure_traceback)
        if state.completion_failure_traceback is not None
        else None,
        tuple(_callback_tensor_binding(tensor) for tensor in node_bars),
        tuple(_callback_tensor_binding(tensor) for tensor in launch_bars),
        _callback_tensor_binding(validation_status),
        transaction.active_block_generation_ids,
        transaction.prepared_block_generation_ids,
        transaction.prepared_block_identities,
        transaction.node_bar_signatures,
        transaction.output_bar_signatures,
        transaction.compact_output_scratch_tensor_bytes,
        transaction.global_output_scratch_tensor_bytes,
        transaction.total_output_scratch_tensor_bytes,
        transaction.output_scratch_tensor_byte_budget,
        transaction.output_scratch_tensor_count,
        transaction.generation_id,
        transaction.output_scratch_owned_by_token,
        transaction.exact_zero_output_scratch_allocated,
        transaction.duplicate_active_block_generations_rejected,
        transaction.active_manifest_coverage_certified,
        transaction.single_use_scratch_generation_certified,
        transaction.hidden_output_alias_absence_certified,
        transaction.allocator_storage_bytes_measured,
        transaction.trainer_promotion_complete,
        transaction.runtime_status,
        transaction._seal,
    )


def _callback_tensor_binding(tensor: torch.Tensor) -> tuple[object, ...]:
    """Include storage identity so same-shaped ``.data`` rebinding is visible."""

    return (
        *_warm_tensor_signature(tensor),
        tensor.untyped_storage().data_ptr(),
        tensor.untyped_storage().nbytes(),
        tensor.storage_offset(),
    )


def _warm_tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        id(tensor),
        int(getattr(tensor, "_version", 0)),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        tensor.layout,
        tensor.requires_grad,
        tensor.is_contiguous(),
    )


def _same_storage(first: torch.Tensor, second: torch.Tensor) -> bool:
    return first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()


def _same_exact_view(first: torch.Tensor, second: torch.Tensor) -> bool:
    return (
        _same_storage(first, second)
        and first.storage_offset() == second.storage_offset()
        and tuple(first.shape) == tuple(second.shape)
        and tuple(first.stride()) == tuple(second.stride())
        and first.dtype == second.dtype
        and first.device == second.device
    )


def _tensor_bytes(tensors: Sequence[torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _runtime_generation_id(
    *,
    payload: KineticNativeEqualRankBlockPayload,
    device: torch.device,
    global_site_count: int,
    physical_length_epsilon: float,
    native_ops_identity: int,
    native_abi_identity: tuple[tuple[str, int], ...],
) -> str:
    return ":".join(
        (
            ADAPTER_PROVENANCE,
            payload.generation_digest,
            str(device),
            str(global_site_count),
            physical_length_epsilon.hex(),
            str(native_ops_identity),
            repr(native_abi_identity),
        )
    )


def _world_generation_id(
    runtime: KineticNativeEqualRankRuntimeBlock,
    compact_site_rgba_f32: torch.Tensor,
    node_chart_f32: torch.Tensor,
) -> str:
    return ":".join(
        (
            runtime.generation_id,
            "world",
            str(id(compact_site_rgba_f32)),
            str(getattr(compact_site_rgba_f32, "_version", 0)),
            str(id(node_chart_f32)),
            str(getattr(node_chart_f32, "_version", 0)),
        )
    )


__all__ = [
    "ADAPTER_PROVENANCE",
    "FORWARD_INTO_OP_NAME",
    "FORWARD_OP_NAME",
    "FUSED_PREPARE_OP_NAME",
    "FUSED_STATUS_INIT_OP_NAME",
    "FUSED_UNION_PREPARE_OP_NAME",
    "FUSED_UNION_STATUS_INIT_OP_NAME",
    "FUSED_UNION_VJP_OP_NAME",
    "FUSED_UNION_VJP_RUNTIME_STATUS",
    "FUSED_VJP_OP_NAME",
    "FUSED_VJP_RUNTIME_STATUS",
    "MATERIAL_VJP_OP_NAME",
    "KineticNativeEqualRankFusedDirectFullVjpV1",
    "KineticNativeEqualRankFusedDirectFullVjpV1Memory",
    "KineticNativeEqualRankFusedDirectFullVjpV1Transaction",
    "KineticNativeEqualRankFusedDirectFullVjpV1TransactionResult",
    "KineticNativeEqualRankFusedUnionFullVjpV2ConstructionLifetime",
    "KineticNativeEqualRankFusedUnionFullVjpV2Transaction",
    "KineticNativeEqualRankFusedUnionFullVjpV2TransactionResult",
    "KineticNativeEqualRankMaterialVJPResult",
    "KineticNativeEqualRankRuntimeBlock",
    "KineticNativeEqualRankRuntimeConstructionLifetime",
    "KineticNativeEqualRankRuntimeMemory",
    "KineticNativeEqualRankVJPResult",
    "KineticNativeEqualRankWorld",
    "RUNTIME_STATUS",
    "VJP_OP_NAME",
    "execute_kinetic_native_equal_rank_fused_direct_full_vjp_v1",
    "execute_kinetic_native_equal_rank_fused_direct_full_vjp_transaction_v1",
    "execute_kinetic_native_equal_rank_fused_union_full_vjp_transaction_v2",
    "execute_kinetic_native_equal_rank_material_node_vjp",
    "execute_kinetic_native_equal_rank_node_vjp",
    "materialize_kinetic_native_equal_rank_fused_union_full_vjp_transaction_v2",
    "prepare_kinetic_native_equal_rank_fused_direct_full_vjp_v1",
    "prepare_kinetic_native_equal_rank_fused_direct_full_vjp_transaction_v1",
    "prepare_kinetic_native_equal_rank_fused_union_full_vjp_construction_lifetime_v2",
    "prepare_kinetic_native_equal_rank_fused_union_full_vjp_transaction_v2",
    "prepare_kinetic_native_equal_rank_runtime_block",
    "prepare_kinetic_native_equal_rank_runtime_construction_lifetime",
    "materialize_kinetic_native_equal_rank_runtime_block",
    "refresh_kinetic_native_equal_rank_world",
    "refresh_kinetic_native_equal_rank_world_into",
]
