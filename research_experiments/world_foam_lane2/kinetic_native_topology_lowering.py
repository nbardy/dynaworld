"""Frame-free structural lowering from kinetic charts to native WorldFoam ABIs.

Every kinetic owner chart lowers exactly to the existing one-track fixed-word
CSR and compact affine-Lie schedule types.  The lowering additionally carries
the chart's precompiled ``[J,R]`` physical lengths, which are the sufficient
world-side data for a material-only node replay.

This module deliberately stops before native execution.  A source-level native
ABI now accepts the precompiled node lengths directly, so general affine
kinetic sites no longer need a fabricated static ``[S,5]`` table.  Runtime
readiness remains false until that extension is rebuilt and its real Metal
forward/VJP are parity-tested and integrated with the trainer/session path.

The program descriptor retains no tensors.  One chart payload can be
materialized at a time, and its storage is ``O(R + J^2 + J R)`` with no
requested-sample or frame axis.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
from compact_lie_schedule import (
    CompactLieChartSpec,
    CompactLieWorldSchedule,
    compact_lie_world_schedule_from_specs,
)
from compiled_transfer_adjoint import (
    FAR_CUT_ID,
    NEAR_CUT_ID,
    make_stable_cell_word,
)
from kinetic_multichart_transfer_program import (
    KineticMultiChartP0Program,
)
from prepared_track_block import (
    PreparedWorldFoamTrackBlock,
    prepare_worldfoam_track_block,
)

DTYPE = torch.float64
LOWERING_PROVENANCE = "kinetic-native-topology-schedule-structural-lowering-v2"
NATIVE_RUNTIME_REMAINING_GATE = (
    "precompiled-node-length source ABI exists; native execution still requires "
    "an extension rebuild, real Metal forward/VJP parity, and trainer/session integration"
)


@dataclass(frozen=True)
class KineticNativeTopologyChartSpec:
    """Tensor-free identity for one streamable kinetic chart payload."""

    chart_index: int
    owner_word: tuple[int, ...]
    t_min: float
    t_max: float
    near: float
    far: float
    node_count: int
    run_count: int
    topology_content_digest: str
    schedule_generation_digest: str
    node_physical_lengths_digest: str
    owner_topology_certificate_digest: str
    payload_digest: str
    retained_payload_tensor_bytes: int
    native_execution_ready: bool = False
    native_fixed_word_topology_abi_lowered: bool = True
    native_compact_schedule_abi_lowered: bool = True
    native_node_physical_length_source_abi_available: bool = True
    native_node_physical_length_runtime_verified: bool = False
    static_s5_site_geometry_fabricated: bool = False
    requested_frame_sampling_used: bool = False
    remaining_native_runtime_gate: str = NATIVE_RUNTIME_REMAINING_GATE

    def assert_self_consistent(self) -> None:
        if self.chart_index < 0 or not self.owner_word:
            raise ValueError("kinetic native chart spec needs a nonempty owner word")
        if (
            not math.isfinite(self.t_min)
            or not math.isfinite(self.t_max)
            or self.t_max <= self.t_min
            or not math.isfinite(self.near)
            or not math.isfinite(self.far)
            or self.far <= self.near
        ):
            raise ValueError("kinetic native chart spec has an invalid domain")
        if self.node_count < 2 or self.run_count != len(self.owner_word):
            raise ValueError("kinetic native chart spec rank/run metadata is invalid")
        for digest in (
            self.topology_content_digest,
            self.schedule_generation_digest,
            self.node_physical_lengths_digest,
            self.owner_topology_certificate_digest,
            self.payload_digest,
        ):
            _require_sha256(digest)
        if self.retained_payload_tensor_bytes < 1:
            raise ValueError("kinetic native chart payload must retain structural tensors")
        if (
            self.native_execution_ready
            or not self.native_fixed_word_topology_abi_lowered
            or not self.native_compact_schedule_abi_lowered
            or not self.native_node_physical_length_source_abi_available
            or self.native_node_physical_length_runtime_verified
            or self.static_s5_site_geometry_fabricated
            or self.requested_frame_sampling_used
            or self.remaining_native_runtime_gate != NATIVE_RUNTIME_REMAINING_GATE
        ):
            raise ValueError("kinetic native chart execution-boundary contract changed")


@dataclass(frozen=True)
class KineticNativeTopologyChartPayload:
    """One chart's existing native CSR/schedule plus kinetic node lengths."""

    spec: KineticNativeTopologyChartSpec
    topology: PreparedWorldFoamTrackBlock
    schedule: CompactLieWorldSchedule
    node_physical_lengths: torch.Tensor
    tensor_signatures: tuple[tuple[object, ...], ...]

    @property
    def retained_tensor_bytes(self) -> int:
        return (
            self.topology.resident_bytes
            + self.schedule.resident_bytes
            + self.node_physical_lengths.numel() * self.node_physical_lengths.element_size()
        )

    @property
    def persistent_sample_time_tensor_bytes(self) -> int:
        return 0

    @property
    def persistent_frame_or_chart_sample_state_bytes(self) -> int:
        return 0

    @property
    def dense_sample_by_chart_tensor_bytes(self) -> int:
        return 0

    def assert_current(self) -> None:
        self.spec.assert_self_consistent()
        self.schedule.assert_current()
        tensors = (*_topology_tensors(self.topology), self.node_physical_lengths)
        if tuple(_tensor_signature(tensor) for tensor in tensors) != self.tensor_signatures:
            raise ValueError("kinetic native chart payload tensors changed after lowering")
        if self.topology.track_count != 1:
            raise ValueError("direct kinetic lowering currently represents exactly one ray track")
        if self.schedule.global_track_count != 1 or self.schedule.chart_count != 1:
            raise ValueError("one kinetic topology chart must lower to one native schedule chart")
        if tuple(self.node_physical_lengths.shape) != (
            self.spec.node_count,
            self.spec.run_count,
        ):
            raise ValueError("kinetic native node-length payload has the wrong shape")
        if (
            self.node_physical_lengths.dtype != DTYPE
            or self.node_physical_lengths.device.type != "cpu"
            or not self.node_physical_lengths.is_contiguous()
            or not bool(torch.isfinite(self.node_physical_lengths).all().item())
            or bool(torch.any(self.node_physical_lengths <= 0.0).item())
        ):
            raise ValueError("kinetic native node lengths must be positive CPU float64")
        if self.schedule.generation_digest != self.spec.schedule_generation_digest:
            raise ValueError("kinetic native chart schedule changed provenance")
        if _topology_digest(self.topology) != self.spec.topology_content_digest:
            raise ValueError("kinetic native chart CSR changed provenance")
        if _tensor_digest(self.node_physical_lengths) != (self.spec.node_physical_lengths_digest):
            raise ValueError("kinetic native node lengths changed provenance")
        if self.retained_tensor_bytes != self.spec.retained_payload_tensor_bytes:
            raise ValueError("kinetic native retained-byte accounting changed")
        if (
            _payload_digest(
                self.spec,
                topology_digest=self.spec.topology_content_digest,
                schedule_digest=self.spec.schedule_generation_digest,
                node_lengths_digest=self.spec.node_physical_lengths_digest,
            )
            != self.spec.payload_digest
        ):
            raise ValueError("kinetic native chart payload digest mismatch")


@dataclass(frozen=True)
class KineticNativeTopologyLoweringMemoryReport:
    """Requested-frame-labelled report whose retained bytes are F-invariant."""

    requested_frame_count: int
    descriptor_tensor_bytes: int
    source_kinetic_program_tensor_bytes: int
    maximum_resident_chart_payload_bytes: int
    maximum_in_process_retained_tensor_bytes: int
    persistent_sample_time_tensor_bytes: int
    persistent_frame_or_chart_sample_state_bytes: int
    dense_sample_by_chart_tensor_bytes: int
    frame_dependent_retained_tensor_bytes: int


@dataclass(frozen=True)
class KineticNativeTopologyLowering:
    """Provenance-sealed, tensor-free descriptor for streamed chart payloads."""

    source_content_digest: str
    owner_program_semantic_digest: str
    kinetic_program_generation_digest: str
    compiler_provenance: str
    seam_policy_id: str
    charts: tuple[KineticNativeTopologyChartSpec, ...]
    event_guard_digests: tuple[str, ...]
    generation_digest: str
    lowering_provenance: str = LOWERING_PROVENANCE
    exact_binary_sample_dispatch: bool = True
    descriptor_tensor_bytes: int = 0
    source_kinetic_program_tensor_bytes: int = 0
    persistent_sample_time_tensor_bytes: int = 0
    persistent_frame_or_chart_sample_state_bytes: int = 0
    dense_sample_by_chart_tensor_bytes: int = 0
    frame_dependent_retained_tensor_bytes: int = 0
    requested_frame_sampling_used: bool = False
    native_execution_ready: bool = False
    static_s5_site_geometry_fabricated: bool = False
    native_node_physical_length_source_abi_available: bool = True
    native_node_physical_length_runtime_verified: bool = False
    remaining_native_runtime_gate: str = NATIVE_RUNTIME_REMAINING_GATE

    @property
    def chart_count(self) -> int:
        return len(self.charts)

    @property
    def maximum_resident_chart_payload_bytes(self) -> int:
        return max(chart.retained_payload_tensor_bytes for chart in self.charts)

    @property
    def maximum_in_process_retained_tensor_bytes(self) -> int:
        """Current CPU path: the bound source program plus one chart payload."""

        return self.source_kinetic_program_tensor_bytes + self.maximum_resident_chart_payload_bytes

    def memory_report(
        self,
        requested_frame_count: int,
    ) -> KineticNativeTopologyLoweringMemoryReport:
        if (
            isinstance(requested_frame_count, bool)
            or not isinstance(requested_frame_count, int)
            or requested_frame_count < 1
        ):
            raise ValueError("requested_frame_count must be a positive integer")
        return KineticNativeTopologyLoweringMemoryReport(
            requested_frame_count=requested_frame_count,
            descriptor_tensor_bytes=self.descriptor_tensor_bytes,
            source_kinetic_program_tensor_bytes=self.source_kinetic_program_tensor_bytes,
            maximum_resident_chart_payload_bytes=(self.maximum_resident_chart_payload_bytes),
            maximum_in_process_retained_tensor_bytes=(self.maximum_in_process_retained_tensor_bytes),
            persistent_sample_time_tensor_bytes=(self.persistent_sample_time_tensor_bytes),
            persistent_frame_or_chart_sample_state_bytes=(self.persistent_frame_or_chart_sample_state_bytes),
            dense_sample_by_chart_tensor_bytes=self.dense_sample_by_chart_tensor_bytes,
            frame_dependent_retained_tensor_bytes=(self.frame_dependent_retained_tensor_bytes),
        )

    def assert_current(self, program: KineticMultiChartP0Program) -> None:
        if not isinstance(program, KineticMultiChartP0Program):
            raise TypeError("program must be KineticMultiChartP0Program")
        program.assert_current()
        expected_source = (
            program.binding.source_content_digest,
            program.binding.program_semantic_digest,
            program.generation_digest,
            program.binding.compiler_provenance,
            program.seam_policy_id,
        )
        observed_source = (
            self.source_content_digest,
            self.owner_program_semantic_digest,
            self.kinetic_program_generation_digest,
            self.compiler_provenance,
            self.seam_policy_id,
        )
        if observed_source != expected_source:
            raise ValueError("kinetic native lowering source/program provenance changed")
        if (
            self.lowering_provenance != LOWERING_PROVENANCE
            or not self.exact_binary_sample_dispatch
            or self.descriptor_tensor_bytes != 0
            or self.source_kinetic_program_tensor_bytes != _source_program_tensor_bytes(program)
            or self.persistent_sample_time_tensor_bytes != 0
            or self.persistent_frame_or_chart_sample_state_bytes != 0
            or self.dense_sample_by_chart_tensor_bytes != 0
            or self.frame_dependent_retained_tensor_bytes != 0
            or self.requested_frame_sampling_used
            or self.native_execution_ready
            or not self.native_node_physical_length_source_abi_available
            or self.native_node_physical_length_runtime_verified
            or self.static_s5_site_geometry_fabricated
            or self.remaining_native_runtime_gate != NATIVE_RUNTIME_REMAINING_GATE
        ):
            raise ValueError("kinetic native lowering execution/memory contract changed")
        expected_charts = tuple(_describe_chart(program, chart_index) for chart_index in range(program.chart_count))
        if self.charts != expected_charts:
            raise ValueError("kinetic native lowering chart descriptors changed")
        expected_guards = tuple(_event_guard_digest(guard) for guard in program.binding.program.active_event_guards)
        if self.event_guard_digests != expected_guards:
            raise ValueError("kinetic native lowering event-guard provenance changed")
        if self.generation_digest != _lowering_digest(
            source=expected_source,
            charts=expected_charts,
            event_guard_digests=expected_guards,
            source_program_tensor_bytes=self.source_kinetic_program_tensor_bytes,
        ):
            raise ValueError("kinetic native lowering generation digest mismatch")


def lower_kinetic_multichart_to_native_topology(
    program: KineticMultiChartP0Program,
) -> KineticNativeTopologyLowering:
    """Describe streamable native CSR/schedule payloads without retaining them."""

    if not isinstance(program, KineticMultiChartP0Program):
        raise TypeError("program must be KineticMultiChartP0Program")
    program.assert_current()
    source = (
        program.binding.source_content_digest,
        program.binding.program_semantic_digest,
        program.generation_digest,
        program.binding.compiler_provenance,
        program.seam_policy_id,
    )
    charts = tuple(_describe_chart(program, chart_index) for chart_index in range(program.chart_count))
    event_guard_digests = tuple(_event_guard_digest(guard) for guard in program.binding.program.active_event_guards)
    result = KineticNativeTopologyLowering(
        source_content_digest=source[0],
        owner_program_semantic_digest=source[1],
        kinetic_program_generation_digest=source[2],
        compiler_provenance=source[3],
        seam_policy_id=source[4],
        charts=charts,
        event_guard_digests=event_guard_digests,
        source_kinetic_program_tensor_bytes=_source_program_tensor_bytes(program),
        generation_digest=_lowering_digest(
            source=source,
            charts=charts,
            event_guard_digests=event_guard_digests,
            source_program_tensor_bytes=_source_program_tensor_bytes(program),
        ),
    )
    result.assert_current(program)
    return result


def materialize_kinetic_native_topology_chart(
    lowering: KineticNativeTopologyLowering,
    program: KineticMultiChartP0Program,
    chart_index: int,
) -> KineticNativeTopologyChartPayload:
    """Materialize one frame-free chart payload after validating provenance."""

    if not isinstance(lowering, KineticNativeTopologyLowering):
        raise TypeError("lowering must be KineticNativeTopologyLowering")
    lowering.assert_current(program)
    if isinstance(chart_index, bool) or not isinstance(chart_index, int):
        raise TypeError("chart_index must be an integer")
    if not 0 <= chart_index < lowering.chart_count:
        raise ValueError("chart_index leaves the kinetic native lowering")
    topology, schedule, lengths = _lower_chart_payload_tensors(program, chart_index)
    payload = KineticNativeTopologyChartPayload(
        spec=lowering.charts[chart_index],
        topology=topology,
        schedule=schedule,
        node_physical_lengths=lengths,
        tensor_signatures=tuple(_tensor_signature(tensor) for tensor in (*_topology_tensors(topology), lengths)),
    )
    payload.assert_current()
    return payload


def _describe_chart(
    program: KineticMultiChartP0Program,
    chart_index: int,
) -> KineticNativeTopologyChartSpec:
    chart = program.charts[chart_index]
    topology, schedule, lengths = _lower_chart_payload_tensors(program, chart_index)
    topology_digest = _topology_digest(topology)
    node_lengths_digest = _tensor_digest(lengths)
    owner_certificate = _digest_parts(
        "kinetic-native-owner-topology-certificate-v1",
        program.binding.source_content_digest,
        program.binding.program_semantic_digest,
        program.generation_digest,
        chart.chart_id,
        chart.owner_word,
        chart.schedule.t_min,
        chart.schedule.t_max,
        chart.right_closed,
    )
    values = dict(
        chart_index=chart_index,
        owner_word=chart.owner_word,
        t_min=chart.schedule.t_min,
        t_max=chart.schedule.t_max,
        near=chart.schedule.near,
        far=chart.schedule.far,
        node_count=chart.node_count,
        run_count=chart.run_count,
        topology_content_digest=topology_digest,
        schedule_generation_digest=schedule.generation_digest,
        node_physical_lengths_digest=node_lengths_digest,
        owner_topology_certificate_digest=owner_certificate,
        retained_payload_tensor_bytes=(
            topology.resident_bytes + schedule.resident_bytes + lengths.numel() * lengths.element_size()
        ),
    )
    provisional = KineticNativeTopologyChartSpec(**values, payload_digest="")
    result = KineticNativeTopologyChartSpec(
        **values,
        payload_digest=_payload_digest(
            provisional,
            topology_digest=topology_digest,
            schedule_digest=schedule.generation_digest,
            node_lengths_digest=node_lengths_digest,
        ),
    )
    result.assert_self_consistent()
    return result


def _lower_chart_payload_tensors(
    program: KineticMultiChartP0Program,
    chart_index: int,
) -> tuple[PreparedWorldFoamTrackBlock, CompactLieWorldSchedule, torch.Tensor]:
    chart = program.charts[chart_index]
    run_count = chart.run_count
    boundary_pairs = (
        torch.tensor(
            list(zip(chart.owner_word[:-1], chart.owner_word[1:], strict=True)),
            dtype=torch.int64,
        )
        if run_count > 1
        else torch.empty((0, 2), dtype=torch.int64)
    )
    left_cut_ids = (NEAR_CUT_ID, *range(run_count - 1))
    right_cut_ids = (*range(run_count - 1), FAR_CUT_ID)
    word = make_stable_cell_word(
        chart.owner_word,
        left_cut_ids,
        right_cut_ids,
    )
    topology = prepare_worldfoam_track_block(
        (word,),
        boundary_pairs,
        site_count=program.binding.sites.site_count,
        track_start=0,
        track_end=1,
    )
    schedule = compact_lie_world_schedule_from_specs(
        (
            CompactLieChartSpec(
                t_min=chart.schedule.t_min,
                t_max=chart.schedule.t_max,
                near=chart.schedule.near,
                far=chart.schedule.far,
                node_count=chart.node_count,
                chart="lie",
            ),
        ),
        global_track_count=1,
        selection_provenance=(f"{LOWERING_PROVENANCE}:{program.generation_digest}:{chart_index}"),
    )
    lowered_chart = schedule.charts[0]
    if (
        not torch.equal(lowered_chart.node_times, chart.schedule.node_times)
        or not torch.equal(lowered_chart.fit_matrix, chart.schedule.fit_matrix)
        or not torch.equal(
            lowered_chart.barycentric_weights,
            chart.schedule.barycentric_weights,
        )
    ):
        raise ValueError("kinetic chart schedule cannot lower exactly to native ABI")
    lengths = chart.node_physical_lengths.detach().clone().contiguous()
    return topology, schedule, lengths


def _payload_digest(
    spec: KineticNativeTopologyChartSpec,
    *,
    topology_digest: str,
    schedule_digest: str,
    node_lengths_digest: str,
) -> str:
    return _digest_parts(
        "kinetic-native-topology-chart-payload-v2",
        spec.chart_index,
        spec.owner_word,
        spec.t_min,
        spec.t_max,
        spec.near,
        spec.far,
        spec.node_count,
        spec.run_count,
        topology_digest,
        schedule_digest,
        node_lengths_digest,
        spec.owner_topology_certificate_digest,
        spec.retained_payload_tensor_bytes,
        True,
        False,
        NATIVE_RUNTIME_REMAINING_GATE,
    )


def _lowering_digest(
    *,
    source: tuple[str, str, str, str, str],
    charts: tuple[KineticNativeTopologyChartSpec, ...],
    event_guard_digests: tuple[str, ...],
    source_program_tensor_bytes: int,
) -> str:
    return _digest_parts(
        "kinetic-native-topology-lowering-v2",
        source,
        source_program_tensor_bytes,
        tuple(chart.payload_digest for chart in charts),
        event_guard_digests,
        LOWERING_PROVENANCE,
        True,
        False,
        NATIVE_RUNTIME_REMAINING_GATE,
    )


def _source_program_tensor_bytes(program: KineticMultiChartP0Program) -> int:
    """Count tensors retained by the current in-memory kinetic source token."""

    binding = program.binding
    tensors = [
        binding.sites.positions0,
        binding.sites.velocities,
        binding.sites.weight_coefficients,
        binding.ray_coefficients,
    ]
    for chart in binding.program.charts:
        representative = chart.representative_word
        tensors.extend(
            (
                representative.word.owners,
                representative.word.left_cut_ids,
                representative.word.right_cut_ids,
                representative.boundary_site_pairs,
            )
        )
    return program.structural_tensor_bytes + sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _event_guard_digest(guard: object) -> str:
    return _digest_parts(
        "kinetic-native-event-guard-provenance-v1",
        guard.guard_id,
        guard.lower_bound,
        guard.upper_bound,
        guard.exact,
        tuple(
            (
                source.kind,
                source.site_ids,
                source.polynomial.coefficients,
                source.derivation,
                source.analytic_guard_only,
                multiplicity,
            )
            for source, multiplicity in zip(
                guard.sources,
                guard.source_multiplicities,
                strict=True,
            )
        ),
        guard.left_owner_word,
        guard.right_owner_word,
        guard.active_owner_change,
    )


def _topology_digest(topology: PreparedWorldFoamTrackBlock) -> str:
    return _digest_parts(
        "prepared-worldfoam-track-block-v1",
        *(_tensor_digest(tensor) for tensor in _topology_tensors(topology)),
    )


def _topology_tensors(
    topology: PreparedWorldFoamTrackBlock,
) -> tuple[torch.Tensor, ...]:
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


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    value = tensor.detach().to(device="cpu").contiguous()
    return (
        tuple(value.shape),
        str(value.dtype),
        int(getattr(tensor, "_version", 0)),
        _tensor_digest(value),
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


def _require_sha256(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("kinetic native lowering digest must be lowercase SHA-256")


__all__ = [
    "KineticNativeTopologyChartPayload",
    "KineticNativeTopologyChartSpec",
    "KineticNativeTopologyLowering",
    "KineticNativeTopologyLoweringMemoryReport",
    "LOWERING_PROVENANCE",
    "NATIVE_RUNTIME_REMAINING_GATE",
    "lower_kinetic_multichart_to_native_topology",
    "materialize_kinetic_native_topology_chart",
]
