"""CPU contract oracle for native kinetic precompiled-length node replay.

The proposed native seam consumes one structurally lowered kinetic chart:

* fixed-word CSR in :class:`PreparedWorldFoamTrackBlock` form;
* a compact ``J``-node schedule;
* precompiled physical lengths ``[J,R]``; and
* compact site RGBA ordered by ``topology.source_site_ids``.

It emits affine-Lie node charts ``[J,4] = [kappa,v_rgb]`` and manually
reverses Lie-node cotangents into compact site RGBA plus physical-length bars. There is no
requested-frame/sample input or retained frame tape. Geometry remains behind
the length VJP seam and native execution remains unavailable in this CPU
oracle.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from kinetic_native_topology_lowering import KineticNativeTopologyChartPayload
from transfer_lie_chart import lie_chart_word_cotangents

DTYPE = torch.float64
CONTRACT_PROVENANCE = "kinetic-native-precompiled-node-length-p0-contract-v1"


@dataclass(frozen=True)
class KineticNativePrecompiledLengthWorld:
    """One immutable compact material snapshot and its native Lie nodes."""

    payload: KineticNativeTopologyChartPayload
    compact_site_rgba: torch.Tensor
    node_charts: torch.Tensor
    global_site_count: int
    compact_material_digest: str
    world_generation_digest: str
    tensor_signatures: tuple[tuple[object, ...], ...]
    contract_provenance: str = CONTRACT_PROVENANCE
    requested_frame_sampling_used: bool = False
    frame_or_sample_axis_retained: bool = False
    native_execution_ready: bool = False
    geometry_vjp_implemented: bool = False

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
        return (
            self.payload.retained_tensor_bytes
            + self.compact_site_rgba.numel() * self.compact_site_rgba.element_size()
            + self.node_charts.numel() * self.node_charts.element_size()
        )

    def assert_current(self) -> None:
        self.payload.assert_current()
        if (
            self.contract_provenance != CONTRACT_PROVENANCE
            or self.requested_frame_sampling_used
            or self.frame_or_sample_axis_retained
            or self.native_execution_ready
            or self.geometry_vjp_implemented
        ):
            raise ValueError("kinetic precompiled-length world contract changed")
        if self.global_site_count < self.compact_site_count:
            raise ValueError("global site count is smaller than the compact topology")
        if tuple(self.compact_site_rgba.shape) != (self.compact_site_count, 4):
            raise ValueError("compact site RGBA must have shape [compact_site_count,4]")
        if tuple(self.node_charts.shape) != (self.node_count, 4):
            raise ValueError("kinetic Lie node charts must have shape [J,4]")
        tensors = (self.compact_site_rgba, self.node_charts)
        if tuple(_tensor_signature(tensor) for tensor in tensors) != self.tensor_signatures:
            raise ValueError("kinetic precompiled-length world tensors changed after refresh")
        if _tensor_digest(self.compact_site_rgba) != self.compact_material_digest:
            raise ValueError("kinetic compact material digest mismatch")
        expected_nodes = _ordered_node_charts(
            self.payload,
            self.compact_site_rgba,
        )
        if not torch.equal(expected_nodes, self.node_charts):
            raise ValueError("kinetic precompiled-length Lie nodes changed provenance")
        if self.world_generation_digest != _world_digest(
            self.payload.spec.payload_digest,
            self.global_site_count,
            self.compact_material_digest,
            _tensor_digest(self.node_charts),
        ):
            raise ValueError("kinetic precompiled-length world generation mismatch")


@dataclass(frozen=True)
class KineticNativePrecompiledLengthVJP:
    """Manual node reverse with no requested-frame/sample state."""

    world: KineticNativePrecompiledLengthWorld
    grad_compact_site_rgba: torch.Tensor
    grad_node_physical_lengths: torch.Tensor
    accounting: dict[str, int | str | bool]
    requested_frame_sampling_used: bool = False
    frame_or_sample_axis_retained: bool = False
    geometry_vjp_implemented: bool = False


def refresh_kinetic_native_precompiled_length_world(
    payload: KineticNativeTopologyChartPayload,
    site_density: torch.Tensor,
    site_color: torch.Tensor,
) -> KineticNativePrecompiledLengthWorld:
    """Pack global material rows into compact native order and replay nodes."""

    if not isinstance(payload, KineticNativeTopologyChartPayload):
        raise TypeError("payload must be KineticNativeTopologyChartPayload")
    payload.assert_current()
    density = torch.as_tensor(site_density, dtype=DTYPE, device="cpu").detach()
    color = torch.as_tensor(site_color, dtype=DTYPE, device="cpu").detach()
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
    source_site_ids = payload.topology.source_site_ids
    if int(source_site_ids.max().item()) >= int(density.numel()):
        raise ValueError("compact topology references a site outside the global material table")
    compact_density = density.index_select(0, source_site_ids).clone().contiguous()
    compact_color = color.index_select(0, source_site_ids).clone().contiguous()
    compact_rgba = torch.cat((compact_color, compact_density[:, None]), dim=1).contiguous()
    node_charts = _ordered_node_charts(payload, compact_rgba)
    material_digest = _tensor_digest(compact_rgba)
    result = KineticNativePrecompiledLengthWorld(
        payload=payload,
        compact_site_rgba=compact_rgba,
        node_charts=node_charts,
        global_site_count=int(density.numel()),
        compact_material_digest=material_digest,
        world_generation_digest=_world_digest(
            payload.spec.payload_digest,
            int(density.numel()),
            material_digest,
            _tensor_digest(node_charts),
        ),
        tensor_signatures=tuple(_tensor_signature(tensor) for tensor in (compact_rgba, node_charts)),
    )
    result.assert_current()
    return result


@torch.no_grad()
def kinetic_native_precompiled_length_node_vjp(
    world: KineticNativePrecompiledLengthWorld,
    grad_node_chart: torch.Tensor,
) -> KineticNativePrecompiledLengthVJP:
    """Reverse Lie ``[J,4]`` bars into compact RGBA and ``[J,R]`` lengths."""

    if not isinstance(world, KineticNativePrecompiledLengthWorld):
        raise TypeError("world must be KineticNativePrecompiledLengthWorld")
    world.assert_current()
    grad = torch.as_tensor(
        grad_node_chart,
        dtype=DTYPE,
        device="cpu",
    ).detach()
    if tuple(grad.shape) != tuple(world.node_charts.shape) or not bool(torch.isfinite(grad).all().item()):
        raise ValueError("grad_node_chart must be finite with shape [J,4]")

    owners = _compact_word_owners(world.payload)
    lengths = world.payload.node_physical_lengths
    rgba = world.compact_site_rgba
    grad_rgba = torch.zeros_like(rgba)
    grad_lengths = torch.zeros_like(lengths)
    raw_node_transfers = _ordered_node_transfers(world.payload, rgba)
    for node_index in range(world.node_count):
        total = raw_node_transfers[node_index]
        total_moment = total[1:]
        moment_bar, kappa_bar = lie_chart_word_cotangents(
            world.node_charts[node_index, 0],
            total_moment,
            grad[node_index],
        )
        prefix_beta = torch.ones((), dtype=DTYPE)
        prefix_moment = torch.zeros(3, dtype=DTYPE)
        for run_index, owner in enumerate(owners):
            density = rgba[owner, 3]
            color = rgba[owner, :3]
            length = lengths[node_index, run_index]
            optical_depth = density * length
            beta = torch.exp(-optical_depth)
            alpha = -torch.expm1(-optical_depth)
            optical_depth_bar = (
                torch.dot(
                    moment_bar,
                    prefix_moment + prefix_beta * color - total_moment,
                )
                + kappa_bar
            )
            grad_rgba[owner, :3] += prefix_beta * alpha * moment_bar
            grad_rgba[owner, 3] += length * optical_depth_bar
            grad_lengths[node_index, run_index] = density * optical_depth_bar
            prefix_moment = prefix_moment + prefix_beta * alpha * color
            prefix_beta = prefix_beta * beta

    accounting: dict[str, int | str | bool] = {
        "compiler_node_count": world.node_count,
        "ordered_run_count": world.run_count,
        "ordered_run_node_interactions": world.node_count * world.run_count,
        "compact_site_count": world.compact_site_count,
        "compact_site_rgba_bytes": rgba.numel() * rgba.element_size(),
        "node_physical_length_bytes": lengths.numel() * lengths.element_size(),
        "node_chart_bytes": world.node_charts.numel() * world.node_charts.element_size(),
        "reverse_output_bytes": grad_rgba.numel() * grad_rgba.element_size()
        + grad_lengths.numel() * grad_lengths.element_size(),
        "requested_frame_count_used": 0,
        "persistent_sample_time_tensor_bytes": 0,
        "persistent_frame_or_sample_tensor_bytes": 0,
        "frame_by_run_reverse_state_allocated": False,
        "reverse_scaling": "O(J * R)",
        "geometry_vjp_implemented": False,
    }
    return KineticNativePrecompiledLengthVJP(
        world=world,
        grad_compact_site_rgba=grad_rgba,
        grad_node_physical_lengths=grad_lengths,
        accounting=accounting,
    )


def _ordered_node_transfers(
    payload: KineticNativeTopologyChartPayload,
    compact_site_rgba: torch.Tensor,
) -> torch.Tensor:
    return _ordered_node_statistics(payload, compact_site_rgba)[0]


def _ordered_node_charts(
    payload: KineticNativeTopologyChartPayload,
    compact_site_rgba: torch.Tensor,
) -> torch.Tensor:
    return _ordered_node_statistics(payload, compact_site_rgba)[1]


def _ordered_node_statistics(
    payload: KineticNativeTopologyChartPayload,
    compact_site_rgba: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    owners = _compact_word_owners(payload)
    transfer_rows = []
    chart_rows = []
    for node_lengths in payload.node_physical_lengths:
        kappa_total = torch.zeros((), dtype=DTYPE)
        beta_total = torch.ones((), dtype=DTYPE)
        moment_total = torch.zeros(3, dtype=DTYPE)
        for run_index, owner in enumerate(owners):
            density = compact_site_rgba[owner, 3]
            color = compact_site_rgba[owner, :3]
            optical_depth = density * node_lengths[run_index]
            beta = torch.exp(-optical_depth)
            alpha = -torch.expm1(-optical_depth)
            kappa_total = kappa_total + optical_depth
            moment_total = moment_total + beta_total * alpha * color
            beta_total = beta_total * beta
        transfer_rows.append(torch.cat((beta_total.reshape(1), moment_total)))
        chart_rows.append(
            torch.cat(
                (
                    kappa_total.reshape(1),
                    _lie_inverse_phi(kappa_total) * moment_total,
                )
            )
        )
    transfers = torch.stack(transfer_rows).contiguous()
    charts = torch.stack(chart_rows).contiguous()
    if not bool(torch.isfinite(transfers).all().item()) or not bool(torch.isfinite(charts).all().item()):
        raise ValueError("kinetic node transfer contains nonfinite values")
    return transfers, charts


def _lie_inverse_phi(kappa: torch.Tensor) -> torch.Tensor:
    small = kappa.abs() < 1.0e-4
    kappa2 = kappa * kappa
    kappa4 = kappa2 * kappa2
    kappa6 = kappa4 * kappa2
    series = 1.0 + 0.5 * kappa + kappa2 / 12.0 - kappa4 / 720.0 + kappa6 / 30240.0
    denominator = -torch.expm1(-kappa)
    safe_denominator = torch.where(small, torch.ones_like(denominator), denominator)
    return torch.where(small, series, kappa / safe_denominator)


def _compact_word_owners(
    payload: KineticNativeTopologyChartPayload,
) -> tuple[int, ...]:
    offsets = payload.topology.word_offsets_i32
    if tuple(offsets.shape) != (2,) or tuple(offsets.tolist()) != (
        0,
        payload.spec.run_count,
    ):
        raise ValueError("kinetic native topology must contain one complete word")
    owners = tuple(int(owner) for owner in payload.topology.word_owner_i32.tolist())
    if len(owners) != payload.spec.run_count or any(not 0 <= owner < payload.topology.site_count for owner in owners):
        raise ValueError("kinetic native compact owner word is malformed")
    return owners


def _world_digest(
    payload_digest: str,
    global_site_count: int,
    compact_material_digest: str,
    node_transfer_digest: str,
) -> str:
    return _digest_parts(
        CONTRACT_PROVENANCE,
        payload_digest,
        global_site_count,
        compact_material_digest,
        node_transfer_digest,
    )


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        tuple(tensor.shape),
        str(tensor.dtype),
        int(getattr(tensor, "_version", 0)),
        _tensor_digest(tensor),
    )


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    return _digest_parts(
        "tensor-v1",
        tuple(value.shape),
        str(value.dtype),
        value.numpy().tobytes(order="C"),
    )


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


__all__ = [
    "CONTRACT_PROVENANCE",
    "KineticNativePrecompiledLengthVJP",
    "KineticNativePrecompiledLengthWorld",
    "kinetic_native_precompiled_length_node_vjp",
    "refresh_kinetic_native_precompiled_length_world",
]
