"""Source-only adapter for frame-free kinetic precompiled-length native ops.

The adapter binds one provenance-current kinetic topology chart to persistent
device CSR, compact-site, configuration, and ``[J,R]`` physical-length
tensors.  A material refresh evaluates affine-Lie node charts ``[1,J,4]``.
Its explicit reverse accepts arbitrary Lie-chart cotangents, accumulates
compact ``[RGB,density]`` bars, returns the bounded ``[J,R]`` length tape, and
scatters material bars into a caller-sized global table.

This module is intentionally ``source_only/runtime_unverified``.  It does not
import or rebuild the native extension, retain requested samples or frames, or
claim trainer/session integration.  Native ops and device are injected so the
same boundary can be tested with a CPU behavioral double while production is
still gated on real Metal parity.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

import torch
from kinetic_native_topology_lowering import KineticNativeTopologyChartPayload

ADAPTER_PROVENANCE = "kinetic-native-precompiled-length-source-adapter-v1"
RUNTIME_STATUS = "source_only/runtime_unverified"
FORWARD_OP_NAME = "kinetic_precompiled_length_p0_lie_node_forward_launch_only"
VJP_OP_NAME = "kinetic_precompiled_length_p0_lie_node_vjp_accumulate_launch_only"

_TOPOLOGY_TOKEN_SEAL = object()
_WORLD_TOKEN_SEAL = object()
_VJP_RESULT_SEAL = object()


@dataclass(frozen=True)
class KineticNativePrecompiledLengthTopologyToken:
    """Persistent device topology/config for one lowered kinetic chart."""

    payload: KineticNativeTopologyChartPayload = field(repr=False)
    native_ops: Any = field(repr=False)
    device: torch.device
    native_ops_identity: int
    native_abi_identity: tuple[tuple[str, int], ...]
    physical_length_epsilon: float
    source_site_ids_i64: torch.Tensor = field(repr=False)
    word_offsets_i32: torch.Tensor = field(repr=False)
    word_owner_i32: torch.Tensor = field(repr=False)
    node_physical_length_f32: torch.Tensor = field(repr=False)
    config_i32: torch.Tensor = field(repr=False)
    config_f32: torch.Tensor = field(repr=False)
    tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    generation_digest: str
    adapter_provenance: str = ADAPTER_PROVENANCE
    runtime_status: str = RUNTIME_STATUS
    source_only: bool = True
    runtime_verified: bool = False
    native_execution_ready: bool = False
    requested_frame_sampling_used: bool = False
    frame_or_sample_axis_retained: bool = False
    _seal: object = field(default=None, repr=False)

    @property
    def track_count(self) -> int:
        return 1

    @property
    def node_count(self) -> int:
        return self.payload.spec.node_count

    @property
    def run_count(self) -> int:
        return self.payload.spec.run_count

    @property
    def compact_site_count(self) -> int:
        return self.payload.topology.site_count

    @property
    def resident_tensor_bytes(self) -> int:
        return _tensor_bytes(self._persistent_tensors())

    @property
    def persistent_sample_time_tensor_bytes(self) -> int:
        return 0

    @property
    def persistent_frame_or_sample_tensor_bytes(self) -> int:
        return 0

    def _persistent_tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.source_site_ids_i64,
            self.word_offsets_i32,
            self.word_owner_i32,
            self.node_physical_length_f32,
            self.config_i32,
            self.config_f32,
        )

    def assert_current(self) -> None:
        if self._seal is not _TOPOLOGY_TOKEN_SEAL:
            raise ValueError("kinetic native topology token was not sealed by the adapter")
        self.payload.assert_current()
        if (
            self.adapter_provenance != ADAPTER_PROVENANCE
            or self.runtime_status != RUNTIME_STATUS
            or not self.source_only
            or self.runtime_verified
            or self.native_execution_ready
            or self.requested_frame_sampling_used
            or self.frame_or_sample_axis_retained
        ):
            raise ValueError("kinetic native topology source/runtime contract changed")
        _require_native_ops(
            self.native_ops,
            device=self.device,
            attest_compiled=False,
        )
        if self.native_ops_identity != id(self.native_ops):
            raise ValueError("kinetic native topology token belongs to different native ops")
        if self.native_abi_identity != _native_abi_identity(self.native_ops):
            raise ValueError("kinetic native topology token has a stale native ABI identity")
        if self.device != torch.device(self.device):
            raise ValueError("kinetic native topology token has an invalid device")
        expected_config = (self.track_count, self.node_count, self.compact_site_count, self.run_count)
        if (
            not math.isfinite(self.physical_length_epsilon)
            or self.physical_length_epsilon < 0.0
            or tuple(self.config_i32.shape) != (4,)
            or tuple(self.config_f32.shape) != (1,)
        ):
            raise ValueError("kinetic native topology config metadata is invalid")
        tensors = self._persistent_tensors()
        if tuple(_tensor_signature(tensor) for tensor in tensors) != self.tensor_signatures:
            raise ValueError("kinetic native topology/config tensors changed after preparation")
        _require_device_tensor(
            self.source_site_ids_i64,
            name="source_site_ids_i64",
            device=self.device,
            dtype=torch.int64,
            shape=(self.compact_site_count,),
        )
        _require_device_tensor(
            self.word_offsets_i32,
            name="word_offsets_i32",
            device=self.device,
            dtype=torch.int32,
            shape=(2,),
        )
        _require_device_tensor(
            self.word_owner_i32,
            name="word_owner_i32",
            device=self.device,
            dtype=torch.int32,
            shape=(self.run_count,),
        )
        _require_device_tensor(
            self.node_physical_length_f32,
            name="node_physical_length_f32",
            device=self.device,
            dtype=torch.float32,
            shape=(self.node_count, self.run_count),
        )
        _require_device_tensor(
            self.config_i32,
            name="config_i32",
            device=self.device,
            dtype=torch.int32,
            shape=(4,),
        )
        _require_device_tensor(
            self.config_f32,
            name="config_f32",
            device=self.device,
            dtype=torch.float32,
            shape=(1,),
        )
        if self.generation_digest != _topology_generation_digest(
            self.payload,
            device=self.device,
            native_ops_identity=self.native_ops_identity,
            native_abi_identity=self.native_abi_identity,
            expected_config=expected_config,
            physical_length_epsilon=self.physical_length_epsilon,
        ):
            raise ValueError("kinetic native topology generation is stale or mismatched")


@dataclass(frozen=True)
class KineticNativePrecompiledLengthWorldToken:
    """One compact material snapshot and native Lie-node evaluation."""

    topology: KineticNativePrecompiledLengthTopologyToken = field(repr=False)
    site_rgba_f32: torch.Tensor = field(repr=False)
    node_chart_f32: torch.Tensor = field(repr=False)
    global_site_count: int
    compact_material_digest: str
    node_chart_digest: str
    tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    generation_digest: str
    adapter_provenance: str = ADAPTER_PROVENANCE
    runtime_status: str = RUNTIME_STATUS
    source_only: bool = True
    runtime_verified: bool = False
    native_execution_ready: bool = False
    requested_frame_sampling_used: bool = False
    frame_or_sample_axis_retained: bool = False
    _seal: object = field(default=None, repr=False)

    @property
    def node_count(self) -> int:
        return self.topology.node_count

    @property
    def run_count(self) -> int:
        return self.topology.run_count

    @property
    def compact_site_count(self) -> int:
        return self.topology.compact_site_count

    @property
    def resident_tensor_bytes(self) -> int:
        return self.topology.resident_tensor_bytes + _tensor_bytes((self.site_rgba_f32, self.node_chart_f32))

    def assert_current(self) -> None:
        if self._seal is not _WORLD_TOKEN_SEAL:
            raise ValueError("kinetic native world token was not sealed by the adapter")
        self.topology.assert_current()
        if (
            self.adapter_provenance != ADAPTER_PROVENANCE
            or self.runtime_status != RUNTIME_STATUS
            or not self.source_only
            or self.runtime_verified
            or self.native_execution_ready
            or self.requested_frame_sampling_used
            or self.frame_or_sample_axis_retained
        ):
            raise ValueError("kinetic native world source/runtime contract changed")
        if self.global_site_count < 1:
            raise ValueError("kinetic native world requires a positive global site count")
        _require_device_tensor(
            self.site_rgba_f32,
            name="site_rgba_f32",
            device=self.topology.device,
            dtype=torch.float32,
            shape=(self.compact_site_count, 4),
        )
        _require_device_tensor(
            self.node_chart_f32,
            name="node_chart_f32",
            device=self.topology.device,
            dtype=torch.float32,
            shape=(1, self.node_count, 4),
        )
        tensors = (self.site_rgba_f32, self.node_chart_f32)
        if tuple(_tensor_signature(tensor) for tensor in tensors) != self.tensor_signatures:
            raise ValueError("kinetic native world tensors changed after refresh")
        if self.generation_digest != _world_generation_digest(
            self.topology.generation_digest,
            self.global_site_count,
            self.compact_material_digest,
            self.node_chart_digest,
        ):
            raise ValueError("kinetic native world generation is stale or mismatched")


@dataclass(frozen=True)
class KineticNativePrecompiledLengthAdapterVJP:
    """Native material scatter plus the bounded geometry length-bar seam."""

    world: KineticNativePrecompiledLengthWorldToken = field(repr=False)
    grad_compact_site_rgba_f32: torch.Tensor
    grad_global_site_rgba_f32: torch.Tensor
    grad_node_physical_length_f32: torch.Tensor
    accounting: dict[str, int | str | bool]
    tensor_signatures: tuple[tuple[object, ...], ...] = field(repr=False)
    adapter_provenance: str = ADAPTER_PROVENANCE
    runtime_status: str = RUNTIME_STATUS
    source_only: bool = True
    runtime_verified: bool = False
    native_execution_ready: bool = False
    requested_frame_sampling_used: bool = False
    frame_or_sample_axis_retained: bool = False
    geometry_vjp_implemented: bool = False
    _seal: object = field(default=None, repr=False)

    @property
    def resident_output_bytes(self) -> int:
        return _tensor_bytes(
            (
                self.grad_compact_site_rgba_f32,
                self.grad_global_site_rgba_f32,
                self.grad_node_physical_length_f32,
            )
        )

    def assert_current(self) -> None:
        if self._seal is not _VJP_RESULT_SEAL:
            raise ValueError("kinetic native VJP result was not sealed by the adapter")
        self.world.assert_current()
        if (
            self.adapter_provenance != ADAPTER_PROVENANCE
            or self.runtime_status != RUNTIME_STATUS
            or not self.source_only
            or self.runtime_verified
            or self.native_execution_ready
            or self.requested_frame_sampling_used
            or self.frame_or_sample_axis_retained
            or self.geometry_vjp_implemented
        ):
            raise ValueError("kinetic native VJP source/runtime contract changed")
        device = self.world.topology.device
        _require_device_tensor(
            self.grad_compact_site_rgba_f32,
            name="grad_compact_site_rgba_f32",
            device=device,
            dtype=torch.float32,
            shape=(self.world.compact_site_count, 4),
        )
        _require_device_tensor(
            self.grad_global_site_rgba_f32,
            name="grad_global_site_rgba_f32",
            device=device,
            dtype=torch.float32,
            shape=(self.world.global_site_count, 4),
        )
        _require_device_tensor(
            self.grad_node_physical_length_f32,
            name="grad_node_physical_length_f32",
            device=device,
            dtype=torch.float32,
            shape=(self.world.node_count, self.world.run_count),
        )
        tensors = (
            self.grad_compact_site_rgba_f32,
            self.grad_global_site_rgba_f32,
            self.grad_node_physical_length_f32,
        )
        if tuple(_tensor_signature(tensor) for tensor in tensors) != self.tensor_signatures:
            raise ValueError("kinetic native VJP result tensors changed after execution")
        expected_accounting = _vjp_accounting(self.world)
        if self.accounting != expected_accounting:
            raise ValueError("kinetic native VJP accounting changed")


def prepare_kinetic_native_precompiled_length_topology_token(
    payload: KineticNativeTopologyChartPayload,
    *,
    device: torch.device | str,
    native_ops: Any,
    physical_length_epsilon: float = 1.0e-8,
) -> KineticNativePrecompiledLengthTopologyToken:
    """Seal one payload into persistent native launch tensors."""

    if not isinstance(payload, KineticNativeTopologyChartPayload):
        raise TypeError("payload must be KineticNativeTopologyChartPayload")
    payload.assert_current()
    resolved_device = torch.device(device)
    native_abi_identity = _require_native_ops(
        native_ops,
        device=resolved_device,
    )
    if isinstance(physical_length_epsilon, bool):
        raise TypeError("physical_length_epsilon must be a finite nonnegative float")
    epsilon_f32 = torch.tensor(float(physical_length_epsilon), dtype=torch.float32)
    if not bool(torch.isfinite(epsilon_f32).item()) or float(epsilon_f32.item()) < 0.0:
        raise ValueError("physical_length_epsilon must be finite and nonnegative in float32")
    epsilon = float(epsilon_f32.item())

    topology = payload.topology
    if topology.track_count != 1:
        raise ValueError("kinetic precompiled-length native ABI currently requires one track")
    expected_offsets = torch.tensor([0, payload.spec.run_count], dtype=torch.int32)
    if not torch.equal(topology.word_offsets_i32, expected_offsets):
        raise ValueError("kinetic topology must contain one complete compact owner word")
    if (
        topology.word_owner_i32.numel() != payload.spec.run_count
        or bool(torch.any(topology.word_owner_i32 < 0).item())
        or bool(torch.any(topology.word_owner_i32 >= topology.site_count).item())
    ):
        raise ValueError("kinetic topology compact owner word is malformed")

    lengths_f32_cpu = payload.node_physical_lengths.to(dtype=torch.float32).clone().contiguous()
    if not bool(torch.isfinite(lengths_f32_cpu).all().item()) or bool(torch.any(lengths_f32_cpu <= epsilon).item()):
        raise ValueError("float32 node physical lengths must be finite and strictly above epsilon")
    config_values = (1, payload.spec.node_count, topology.site_count, payload.spec.run_count)
    source_site_ids = _persistent_device_copy(
        topology.source_site_ids,
        device=resolved_device,
        dtype=torch.int64,
    )
    word_offsets = _persistent_device_copy(
        topology.word_offsets_i32,
        device=resolved_device,
        dtype=torch.int32,
    )
    word_owner = _persistent_device_copy(
        topology.word_owner_i32,
        device=resolved_device,
        dtype=torch.int32,
    )
    node_lengths = _persistent_device_copy(
        lengths_f32_cpu,
        device=resolved_device,
        dtype=torch.float32,
    )
    config_i32 = _persistent_device_copy(
        torch.tensor(config_values, dtype=torch.int32),
        device=resolved_device,
        dtype=torch.int32,
    )
    config_f32 = _persistent_device_copy(
        epsilon_f32.reshape(1),
        device=resolved_device,
        dtype=torch.float32,
    )
    tensors = (
        source_site_ids,
        word_offsets,
        word_owner,
        node_lengths,
        config_i32,
        config_f32,
    )
    result = KineticNativePrecompiledLengthTopologyToken(
        payload=payload,
        native_ops=native_ops,
        device=resolved_device,
        native_ops_identity=id(native_ops),
        native_abi_identity=native_abi_identity,
        physical_length_epsilon=epsilon,
        source_site_ids_i64=source_site_ids,
        word_offsets_i32=word_offsets,
        word_owner_i32=word_owner,
        node_physical_length_f32=node_lengths,
        config_i32=config_i32,
        config_f32=config_f32,
        tensor_signatures=tuple(_tensor_signature(tensor) for tensor in tensors),
        generation_digest=_topology_generation_digest(
            payload,
            device=resolved_device,
            native_ops_identity=id(native_ops),
            native_abi_identity=native_abi_identity,
            expected_config=config_values,
            physical_length_epsilon=epsilon,
        ),
        _seal=_TOPOLOGY_TOKEN_SEAL,
    )
    result.assert_current()
    return result


@torch.no_grad()
def refresh_kinetic_native_precompiled_length_world_token(
    topology: KineticNativePrecompiledLengthTopologyToken,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
) -> KineticNativePrecompiledLengthWorldToken:
    """Pack compact RGBA and invoke the source-level Lie-node forward ABI."""

    if not isinstance(topology, KineticNativePrecompiledLengthTopologyToken):
        raise TypeError("topology must be KineticNativePrecompiledLengthTopologyToken")
    topology.assert_current()
    density = torch.as_tensor(
        site_density,
        dtype=torch.float32,
        device=topology.device,
    ).detach()
    color = torch.as_tensor(
        site_color,
        dtype=torch.float32,
        device=topology.device,
    ).detach()
    if density.ndim != 1 or density.numel() < 1:
        raise ValueError("site_density must have shape [S] with S >= 1")
    if tuple(color.shape) != (int(density.numel()), 3):
        raise ValueError("site_color must have shape [S,3]")
    if (
        not bool(torch.isfinite(density).all().item())
        or not bool(torch.isfinite(color).all().item())
        or bool(torch.any(density < 0.0).item())
        or bool(torch.any(color < 0.0).item())
        or bool(torch.any(color > 1.0).item())
    ):
        raise ValueError("site material must be finite with density >= 0 and color in [0,1]")
    global_site_count = int(density.numel())
    source_ids_cpu = topology.payload.topology.source_site_ids
    if int(source_ids_cpu.max().item()) >= global_site_count:
        raise ValueError("compact topology references a site outside the global material table")
    compact_density = density.index_select(0, topology.source_site_ids_i64)
    compact_color = color.index_select(0, topology.source_site_ids_i64)
    site_rgba = torch.cat((compact_color, compact_density[:, None]), dim=1).detach().contiguous()
    node_chart = getattr(topology.native_ops, FORWARD_OP_NAME)(
        topology.word_offsets_i32,
        topology.word_owner_i32,
        topology.node_physical_length_f32,
        site_rgba,
        topology.config_i32,
        topology.config_f32,
        track_count=topology.track_count,
        node_count=topology.node_count,
    )
    if not isinstance(node_chart, torch.Tensor):
        raise TypeError("kinetic native forward must return one tensor")
    _require_device_tensor(
        node_chart,
        name="native node_chart_f32",
        device=topology.device,
        dtype=torch.float32,
        shape=(topology.track_count, topology.node_count, 4),
        require_finite=True,
    )
    node_chart = node_chart.detach()
    material_digest = _tensor_digest(site_rgba)
    node_digest = _tensor_digest(node_chart)
    result = KineticNativePrecompiledLengthWorldToken(
        topology=topology,
        site_rgba_f32=site_rgba,
        node_chart_f32=node_chart,
        global_site_count=global_site_count,
        compact_material_digest=material_digest,
        node_chart_digest=node_digest,
        tensor_signatures=tuple(_tensor_signature(tensor) for tensor in (site_rgba, node_chart)),
        generation_digest=_world_generation_digest(
            topology.generation_digest,
            global_site_count,
            material_digest,
            node_digest,
        ),
        _seal=_WORLD_TOKEN_SEAL,
    )
    result.assert_current()
    return result


@torch.no_grad()
def execute_kinetic_native_precompiled_length_node_vjp(
    world: KineticNativePrecompiledLengthWorldToken,
    grad_node_chart: torch.Tensor,
    *,
    global_grad_site_rgba_f32: torch.Tensor | None = None,
) -> KineticNativePrecompiledLengthAdapterVJP:
    """Invoke native Lie-node VJP, then scatter compact bars globally."""

    if not isinstance(world, KineticNativePrecompiledLengthWorldToken):
        raise TypeError("world must be KineticNativePrecompiledLengthWorldToken")
    world.assert_current()
    grad = torch.as_tensor(
        grad_node_chart,
        dtype=torch.float32,
        device=world.topology.device,
    ).detach()
    if tuple(grad.shape) != (world.node_count, 4) or not bool(torch.isfinite(grad).all().item()):
        raise ValueError("grad_node_chart must be finite with shape [J,4]")
    grad_native = grad.clone().contiguous().unsqueeze(0)
    grad_compact = torch.zeros_like(world.site_rgba_f32)
    native_result = getattr(world.topology.native_ops, VJP_OP_NAME)(
        world.topology.word_offsets_i32,
        world.topology.word_owner_i32,
        world.topology.node_physical_length_f32,
        world.site_rgba_f32,
        world.node_chart_f32,
        grad_native,
        grad_compact,
        world.topology.config_i32,
        world.topology.config_f32,
        track_count=world.topology.track_count,
        node_count=world.node_count,
    )
    if not isinstance(native_result, tuple) or len(native_result) != 2:
        raise TypeError("kinetic native VJP must return (aliased RGBA bar, length bar)")
    returned_grad_compact, grad_lengths = native_result
    if not isinstance(returned_grad_compact, torch.Tensor) or not isinstance(grad_lengths, torch.Tensor):
        raise TypeError("kinetic native VJP outputs must be tensors")
    _require_device_tensor(
        returned_grad_compact,
        name="native grad_site_rgba_f32",
        device=world.topology.device,
        dtype=torch.float32,
        shape=(world.compact_site_count, 4),
        require_finite=True,
    )
    if returned_grad_compact.data_ptr() != grad_compact.data_ptr():
        raise ValueError("kinetic native VJP must alias the supplied compact RGBA accumulator")
    _require_device_tensor(
        grad_lengths,
        name="native grad_node_physical_length_f32",
        device=world.topology.device,
        dtype=torch.float32,
        shape=(world.node_count, world.run_count),
        require_finite=True,
    )
    if global_grad_site_rgba_f32 is None:
        global_grad = torch.zeros(
            (world.global_site_count, 4),
            dtype=torch.float32,
            device=world.topology.device,
        )
    else:
        if not isinstance(global_grad_site_rgba_f32, torch.Tensor):
            raise TypeError("global_grad_site_rgba_f32 must be a tensor")
        global_grad = global_grad_site_rgba_f32
        _require_device_tensor(
            global_grad,
            name="global_grad_site_rgba_f32",
            device=world.topology.device,
            dtype=torch.float32,
            shape=(world.global_site_count, 4),
            require_finite=True,
        )
    global_grad.index_add_(
        0,
        world.topology.source_site_ids_i64,
        returned_grad_compact,
    )
    tensors = (returned_grad_compact, global_grad, grad_lengths)
    result = KineticNativePrecompiledLengthAdapterVJP(
        world=world,
        grad_compact_site_rgba_f32=returned_grad_compact,
        grad_global_site_rgba_f32=global_grad,
        grad_node_physical_length_f32=grad_lengths,
        accounting=_vjp_accounting(world),
        tensor_signatures=tuple(_tensor_signature(tensor) for tensor in tensors),
        _seal=_VJP_RESULT_SEAL,
    )
    result.assert_current()
    return result


def _require_native_ops(
    native_ops: Any,
    *,
    device: torch.device,
    attest_compiled: bool = True,
) -> tuple[tuple[str, int], ...]:
    if native_ops is None:
        raise TypeError("native_ops must be injected explicitly")
    for name in (FORWARD_OP_NAME, VJP_OP_NAME):
        if not callable(getattr(native_ops, name, None)):
            raise TypeError(f"native_ops does not expose callable {name}")
    if device.type == "mps" and attest_compiled:
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
    return _native_abi_identity(native_ops)


def _native_abi_identity(native_ops: Any) -> tuple[tuple[str, int], ...]:
    identities = []
    for name in (FORWARD_OP_NAME, VJP_OP_NAME):
        callable_value = getattr(native_ops, name, None)
        implementation = getattr(callable_value, "__func__", callable_value)
        identities.append((name, id(implementation)))
    return tuple(identities)


def _persistent_device_copy(
    value: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return value.detach().to(device=device, dtype=dtype).clone().contiguous()


def _require_device_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    require_finite: bool = False,
) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device != device
        or tensor.dtype != dtype
        or tuple(tensor.shape) != shape
        or not tensor.is_contiguous()
    ):
        raise ValueError(f"{name} must be contiguous {device} {dtype} with shape {shape}")
    if require_finite and not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite")


def _topology_generation_digest(
    payload: KineticNativeTopologyChartPayload,
    *,
    device: torch.device,
    native_ops_identity: int,
    native_abi_identity: tuple[tuple[str, int], ...],
    expected_config: tuple[int, int, int, int],
    physical_length_epsilon: float,
) -> str:
    return _digest_parts(
        ADAPTER_PROVENANCE,
        RUNTIME_STATUS,
        payload.spec.payload_digest,
        str(device),
        native_ops_identity,
        native_abi_identity,
        expected_config,
        physical_length_epsilon,
    )


def _world_generation_digest(
    topology_generation_digest: str,
    global_site_count: int,
    compact_material_digest: str,
    node_chart_digest: str,
) -> str:
    return _digest_parts(
        ADAPTER_PROVENANCE,
        topology_generation_digest,
        global_site_count,
        compact_material_digest,
        node_chart_digest,
    )


def _vjp_accounting(
    world: KineticNativePrecompiledLengthWorldToken,
) -> dict[str, int | str | bool]:
    compact_bytes = world.site_rgba_f32.numel() * world.site_rgba_f32.element_size()
    length_bytes = (
        world.topology.node_physical_length_f32.numel() * world.topology.node_physical_length_f32.element_size()
    )
    return {
        "compiler_node_count": world.node_count,
        "ordered_run_count": world.run_count,
        "ordered_run_node_interactions": world.node_count * world.run_count,
        "compact_site_count": world.compact_site_count,
        "compact_site_rgba_bytes": compact_bytes,
        "node_physical_length_bytes": length_bytes,
        "node_chart_bytes": world.node_chart_f32.numel() * world.node_chart_f32.element_size(),
        "compact_reverse_output_bytes": compact_bytes + length_bytes,
        "global_site_rgba_bytes": world.global_site_count * 4 * torch.tensor([], dtype=torch.float32).element_size(),
        "requested_frame_count_used": 0,
        "persistent_sample_time_tensor_bytes": 0,
        "persistent_frame_or_sample_tensor_bytes": 0,
        "frame_by_run_reverse_state_allocated": False,
        "reverse_scaling": "O(J * R)",
        "geometry_vjp_implemented": False,
        "source_only": True,
        "runtime_verified": False,
        "native_execution_ready": False,
    }


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        tuple(tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
        tuple(tensor.stride()),
        int(tensor.data_ptr()),
        int(getattr(tensor, "_version", 0)),
    )


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    return _digest_parts(
        "tensor-v1",
        tuple(value.shape),
        str(value.dtype),
        value.numpy().tobytes(order="C"),
    )


def _tensor_bytes(tensors: tuple[torch.Tensor, ...]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


__all__ = [
    "ADAPTER_PROVENANCE",
    "FORWARD_OP_NAME",
    "KineticNativePrecompiledLengthAdapterVJP",
    "KineticNativePrecompiledLengthTopologyToken",
    "KineticNativePrecompiledLengthWorldToken",
    "RUNTIME_STATUS",
    "VJP_OP_NAME",
    "execute_kinetic_native_precompiled_length_node_vjp",
    "prepare_kinetic_native_precompiled_length_topology_token",
    "refresh_kinetic_native_precompiled_length_world_token",
]
