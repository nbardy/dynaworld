from __future__ import annotations

import inspect
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import paper_kinetic_union_local_bar_assembly as union_assembly
import pytest
import torch
from camera import CameraSpec
from kinetic_multichart_transfer_program import compile_kinetic_multichart_p0_program
from kinetic_native_equal_rank_lowering import (
    kinetic_native_equal_rank_chart_sources_for_track,
    lower_kinetic_native_equal_rank_buckets,
)
from kinetic_native_topology_lowering import lower_kinetic_multichart_to_native_topology
from kinetic_owner_chart_compiler import compile_exact_kinetic_owner_charts
from kinetic_power_word_compiler import AffineKineticPowerSites
from paper_kinetic_ragged_sample_plan import iter_paper_kinetic_row_ragged_request_blocks
from paper_kinetic_union_local_bar_assembly import (
    begin_paper_kinetic_union_local_bar_assembly,
    consume_paper_kinetic_union_local_native_contribution,
    finalize_paper_kinetic_union_local_bar_assembly,
    install_paper_kinetic_union_local_cold_receipt_lifetime,
    materialize_paper_kinetic_union_local_spatial_bundle,
    prepare_paper_kinetic_union_local_request_work,
    prepare_paper_kinetic_union_local_spatial_bundle,
    prepare_paper_kinetic_union_local_spatial_bundle_construction_lifetime,
    seal_paper_kinetic_native_block_vjp_contribution,
)
from paper_ragged_material_bar_coordinator import (
    begin_paper_ragged_material_bar_step,
    consume_paper_ragged_compact_material_bar_result,
    finalize_paper_ragged_material_bar_step,
    prepare_paper_ragged_material_spatial_block,
    prepare_paper_ragged_material_view_program,
    stage_next_paper_ragged_material_bar_request,
)
from paper_ragged_track_staging import adapt_paper_spacetime_batch_to_track_groups
from paper_training_types import SpacetimeBatch, SpacetimeSample
from powerfoam_training_data import PowerFoamRayProvider, PowerFoamTargetProvider


class _Targets:
    view_count = 1
    frame_count = 4
    height = 1
    width = 3

    def select_view_frames(
        self,
        view_indices: tuple[int, ...],
        frame_indices: tuple[int, ...],
    ) -> torch.Tensor:
        return torch.stack(
            [
                torch.full(
                    (3, self.height, self.width),
                    float(frame + 1),
                    dtype=torch.float32,
                )
                for _view, frame in zip(view_indices, frame_indices, strict=True)
            ]
        )

    def residency(self) -> dict[str, Any]:
        return {
            "source_kind": "union_local_cpu_fixture",
            "source_device": "cpu",
            "logical_bytes": self.frame_count * 3 * self.height * self.width * 4,
            "resident_bytes": 0,
            "full_source_resident": False,
            "disk_lazy_decode": False,
        }


def _sites() -> AffineKineticPowerSites:
    slopes = [(0, 0), (-2, 0)]
    intercepts = [(0, 0, 0), (1, -1, 0)]
    positions = []
    velocities = []
    weights = []
    for (slope0, slope1), (bias0, bias1, bias2) in zip(
        slopes,
        intercepts,
        strict=True,
    ):
        position = -Fraction(slope0) / 2
        velocity = -Fraction(slope1) / 2
        positions.append((position, Fraction(0), Fraction(0)))
        velocities.append((velocity, Fraction(0), Fraction(0)))
        weights.append(
            (
                position * position - Fraction(bias0),
                2 * position * velocity - Fraction(bias1),
                velocity * velocity - Fraction(bias2),
            )
        )
    # Two deliberately distant sites remain in the global material table but
    # never own a segment in this request.  The test therefore distinguishes
    # the bounded source union from a request-sized global-site bar.
    positions.extend(
        (
            (Fraction(100), Fraction(0), Fraction(0)),
            (Fraction(200), Fraction(0), Fraction(0)),
        )
    )
    velocities.extend(
        (
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(0)),
        )
    )
    weights.extend(
        (
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(0)),
        )
    )
    return AffineKineticPowerSites(
        positions0=torch.tensor(
            [[float(value) for value in row] for row in positions],
            dtype=torch.float64,
        ),
        velocities=torch.tensor(
            [[float(value) for value in row] for row in velocities],
            dtype=torch.float64,
        ),
        weight_coefficients=torch.tensor(
            [[float(value) for value in row] for row in weights],
            dtype=torch.float64,
        ),
    )


def _kinetic_program(
    sites: AffineKineticPowerSites,
    *,
    ray_origin_x: int,
    node_count: int,
):
    ray = torch.tensor(
        [ray_origin_x, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        dtype=torch.float64,
    )
    owner_program = compile_exact_kinetic_owner_charts(
        sites,
        ray,
        t_min=-2,
        t_max=2,
        near=0,
        far=1,
    )
    return compile_kinetic_multichart_p0_program(
        owner_program,
        sites,
        ray,
        node_count=node_count,
    )


def _sampler():
    sites = _sites()
    programs = (
        _kinetic_program(sites, ray_origin_x=-2, node_count=3),
        _kinetic_program(sites, ray_origin_x=-1, node_count=4),
        _kinetic_program(sites, ray_origin_x=0, node_count=5),
    )
    sources = tuple(
        source
        for track_id, program in enumerate(programs)
        for source in kinetic_native_equal_rank_chart_sources_for_track(
            track_id,
            program,
            lowering=lower_kinetic_multichart_to_native_topology(program),
        )
    )
    lowering = lower_kinetic_native_equal_rank_buckets(
        tuple(reversed(sources)),
        maximum_rows_per_block=2,
    )
    from paper_kinetic_ragged_sample_plan import prepare_paper_kinetic_row_ragged_sampler

    return prepare_paper_kinetic_row_ragged_sampler(
        view_index=0,
        lowering=lowering,
        sources=sources,
    )


def test_union_local_two_phase_construction_installs_cpu_sources_first() -> None:
    sampler = _sampler()
    lifetime = (
        prepare_paper_kinetic_union_local_spatial_bundle_construction_lifetime(
            sampler,
            track_ids=(0, 1, 2),
            device="cpu",
        )
    )

    lifetime.assert_retained()
    assert lifetime.phase == "installed"
    assert lifetime.transferred_tensors == []
    assert lifetime.transfer_intermediates == []
    assert lifetime.bundle_identity == 0
    assert all(
        tensor.device.type == "cpu"
        for tensor in lifetime.source_tensors_i64_cpu
    )

    bundle = materialize_paper_kinetic_union_local_spatial_bundle(lifetime)

    lifetime.assert_retained()
    assert lifetime.phase == "materialized"
    assert lifetime.bundle_identity == id(bundle)
    assert lifetime.bundle_generation_digest == bundle.generation_digest
    assert bundle._construction_lifetime is lifetime
    assert lifetime.cold_receipt_install_count == 1
    assert lifetime.cold_receipt_retirement_count == 1
    assert lifetime.cold_receipt_lifetime is not None
    assert lifetime.cold_receipt_lifetime.phase == "retired"
    assert lifetime.cold_receipt_lifetime.current_transfer_source is None
    assert lifetime.cold_receipt_lifetime.current_raw_device_to_host_result is None
    assert lifetime.cold_receipt_lifetime.current_cpu_destination_tensor is None
    assert len(lifetime.transferred_tensors) == len(
        lifetime.source_tensors_i64_cpu
    )
    with pytest.raises(ValueError, match="already used"):
        materialize_paper_kinetic_union_local_spatial_bundle(lifetime)
    original_device = lifetime.device
    lifetime.device = torch.device("mps")
    with pytest.raises(RuntimeError, match="union transfer release is CPU-only"):
        lifetime.release_transfer_predecessors_after_completion_fence()
    with pytest.raises(RuntimeError, match="union retirement is CPU-only"):
        lifetime.retire_after_completion_fence()
    assert lifetime.phase == "materialized"
    lifetime.device = original_device
    lifetime.release_transfer_predecessors_after_completion_fence()
    assert lifetime.phase == "settled"
    assert lifetime.source_tensors_i64_cpu == ()
    assert lifetime.transfer_intermediates == []
    bundle.assert_warm_layout()
    lifetime.retire_after_completion_fence()
    assert lifetime.phase == "retired"
    assert lifetime.transferred_tensors == []
    assert lifetime.bindings == []
    with pytest.raises(ValueError, match="metadata/memory contract changed"):
        bundle.assert_warm_layout()


def test_accelerator_union_maps_cold_admit_from_fenced_copy_without_readback(
    monkeypatch,
) -> None:
    """A meta device stands in for a non-CPU destination in this source gate.

    The test protects the behavior that matters for MPS: construction retains
    exact CPU sources plus exact destinations until the outer fence, and cold
    admission after settlement never calls the legacy device-to-host receipt.
    """

    sampler = _sampler()
    lifetime = (
        prepare_paper_kinetic_union_local_spatial_bundle_construction_lifetime(
            sampler,
            track_ids=(0, 1, 2),
            device="meta",
        )
    )
    readback_calls = 0

    def reject_readback(*_args, **_kwargs):
        nonlocal readback_calls
        readback_calls += 1
        raise AssertionError("accelerator union-map cold admission read back data")

    monkeypatch.setattr(
        union_assembly,
        "_materialize_and_validate_cold_receipt_contents",
        reject_readback,
    )
    bundle = materialize_paper_kinetic_union_local_spatial_bundle(lifetime)

    bundle.assert_accelerator_transfer_pending()
    lifetime.assert_accelerator_transfer_releasable_after_completion_fence(
        bundle
    )
    assert lifetime.phase == "materialized"
    assert lifetime.source_tensors_i64_cpu
    assert lifetime.cold_receipt_lifetime is None
    assert readback_calls == 0

    # This assignment-only commit is authorized by the outer sealed receipt in
    # production; the unit gate exercises the post-consumption transition.
    lifetime._commit_transfer_predecessors_after_consumed_receipt()
    bundle.assert_accelerator_cold_current_after_settlement()
    bundle.assert_cold_current()
    assert lifetime.phase == "settled"
    assert lifetime.source_tensors_i64_cpu == ()
    assert lifetime.transfer_intermediates == []
    assert readback_calls == 0


def test_cold_union_map_receipt_is_installed_validated_and_retired_explicitly() -> None:
    sampler = _sampler()
    bundle = prepare_paper_kinetic_union_local_spatial_bundle(
        sampler,
        track_ids=(0, 1, 2),
        device="cpu",
    )
    construction = bundle._construction_lifetime
    receipt = install_paper_kinetic_union_local_cold_receipt_lifetime(bundle)

    assert construction.cold_receipt_lifetime is receipt
    assert construction.cold_receipt_install_count == 2
    assert construction.cold_receipt_retirement_count == 1
    assert receipt.phase == "installed"
    assert receipt.current_transfer_source is None
    assert receipt.current_raw_device_to_host_result is None
    assert receipt.current_cpu_destination_tensor is None
    with pytest.raises(ValueError, match="no proven completion boundary"):
        construction.retire_cold_receipt_after_proven_completion_boundary(receipt)

    bundle.assert_cold_current(receipt_lifetime=receipt)

    receipt.assert_for_bundle(bundle)
    assert receipt.phase == "validated"
    assert receipt.source_tensor_count == bundle.native_block_count + 1
    assert receipt.validated_source_count == receipt.source_tensor_count
    assert receipt.current_transfer_source is None
    assert receipt.current_raw_device_to_host_result is None
    assert receipt.current_cpu_destination_tensor is None
    assert receipt.current_converted_int_tuple is None
    assert receipt.validated_content_digest
    with pytest.raises(ValueError, match="already consumed"):
        bundle.assert_cold_current(receipt_lifetime=receipt)

    construction.retire_cold_receipt_after_proven_completion_boundary(receipt)
    construction.assert_retained()
    assert receipt.phase == "retired"
    assert receipt.source_tensor_count == bundle.native_block_count + 1
    assert receipt.source_tensors_i64 == ()
    assert receipt.current_transfer_source is None
    assert receipt.current_raw_device_to_host_result is None
    assert receipt.current_cpu_destination_tensor is None
    assert receipt.current_converted_int_tuple is None
    assert construction.cold_receipt_install_count == 2
    assert construction.cold_receipt_retirement_count == 2


def test_cold_union_map_validation_failure_keeps_all_receipt_roots(monkeypatch) -> None:
    sampler = _sampler()
    bundle = prepare_paper_kinetic_union_local_spatial_bundle(
        sampler,
        track_ids=(0, 1, 2),
        device="cpu",
    )
    construction = bundle._construction_lifetime
    receipt = install_paper_kinetic_union_local_cold_receipt_lifetime(bundle)
    monkeypatch.setattr(
        union_assembly,
        "_mapping_digest",
        lambda **_kwargs: "synthetic-stale-mapping-digest",
    )

    with pytest.raises(ValueError, match="mapping generation changed"):
        bundle.assert_cold_current(receipt_lifetime=receipt)

    construction.assert_retained()
    receipt.assert_for_bundle(bundle)
    assert construction.cold_receipt_lifetime is receipt
    assert receipt.phase == "transferring"
    # The stale digest is detected on the first compact map.  The completed
    # union destination was already retired, while only the failing current map
    # and its predecessors remain rooted.
    assert receipt.validated_source_count == 1
    assert receipt.current_source_index == 1
    assert receipt.current_transfer_source is not None
    assert receipt.current_raw_device_to_host_result is not None
    assert receipt.current_cpu_destination_tensor is not None
    assert receipt.current_converted_int_tuple is not None
    with pytest.raises(RuntimeError, match="already active"):
        install_paper_kinetic_union_local_cold_receipt_lifetime(bundle)


def _request_case():
    sampler = _sampler()
    bundle = prepare_paper_kinetic_union_local_spatial_bundle(
        sampler,
        track_ids=(0, 1, 2),
        device="cpu",
    )
    source = _Targets()
    target_provider = PowerFoamTargetProvider(source=source, device=torch.device("cpu"))
    cameras = (
        tuple(
            CameraSpec(
                fx=3.0,
                fy=3.0,
                cx=1.5,
                cy=0.5,
                camera_to_world=torch.eye(4),
            )
            for _frame in range(source.frame_count)
        ),
    )
    ray_provider = PowerFoamRayProvider(
        cameras=cameras,
        height=source.height,
        width=source.width,
        device=torch.device("cpu"),
    )
    paper_batch = SpacetimeBatch(
        samples=tuple(
            SpacetimeSample(view_index=0, frame_index=frame)
            for frame in range(source.frame_count)
        ),
        epoch=0,
        batch_index=0,
        completes_epoch=False,
    )
    adapted = adapt_paper_spacetime_batch_to_track_groups(
        paper_batch,
        target_provider=target_provider,
        ray_provider=ray_provider,
        frame_times=torch.tensor([-2.0, -1.0, 1.0, 2.0]),
        height=1,
        width=3,
        device="cpu",
        loss_normalization_id="union-local-paper-step",
    )
    spatial_block = prepare_paper_ragged_material_spatial_block(
        block_id="view-0-tracks-0-3",
        view_index=0,
        track_start=0,
        track_end=3,
        world_token=bundle,
        world_generation_id=bundle.generation_digest,
        source_site_ids=bundle.source_site_ids_i64,
        global_site_count=bundle.global_site_count,
        device="cpu",
    )
    view_program = prepare_paper_ragged_material_view_program(
        view_index=0,
        global_track_count=3,
        global_site_count=bundle.global_site_count,
        blocks=(spatial_block,),
    )
    outer = begin_paper_ragged_material_bar_step(
        adapted,
        programs=(view_program,),
        global_grad_site_rgba_f32=torch.zeros(
            (bundle.global_site_count, 4),
            dtype=torch.float32,
        ),
    )
    request = stage_next_paper_ragged_material_bar_request(
        outer,
        view_index=0,
        block_id=spatial_block.block_id,
        local_sample_start=0,
        local_sample_end=4,
    )
    work = prepare_paper_kinetic_union_local_request_work(
        bundle,
        request,
        maximum_samples_per_launch=2,
    )
    return sampler, bundle, outer, request, work


@dataclass
class _FakeBlock:
    generation_digest: str


@dataclass
class _FakePayload:
    block: _FakeBlock
    generation_digest: str = "distinct-payload-generation"


@dataclass
class _FakeRuntime:
    payload: _FakePayload
    source_site_ids_i64: torch.Tensor


@dataclass
class _FakeWorld:
    runtime: _FakeRuntime


@dataclass
class _FakeVJP:
    world: _FakeWorld
    grad_compact_site_rgba_f32: torch.Tensor
    grad_global_site_rgba_f32: torch.Tensor | None = None
    stale: bool = False

    def assert_current(self) -> None:
        if self.stale:
            raise ValueError("fake native VJP is stale")


def _fake_vjp(binding, value: float) -> _FakeVJP:
    compact_bar = torch.full(
        (binding.compact_site_count, 4),
        value,
        dtype=torch.float32,
    )
    return _FakeVJP(
        world=_FakeWorld(
            runtime=_FakeRuntime(
                payload=_FakePayload(
                    _FakeBlock(binding.native_block_generation_digest)
                ),
                source_site_ids_i64=torch.tensor(
                    binding.compact_source_site_ids,
                    dtype=torch.int64,
                ),
            )
        ),
        grad_compact_site_rgba_f32=compact_bar,
    )


def test_union_local_assembly_emits_one_compact_bar_without_request_global_bar() -> None:
    sampler, bundle, outer, request, work = _request_case()
    sample_blocks = tuple(
        iter_paper_kinetic_row_ragged_request_blocks(
            sampler,
            request,
            maximum_samples_per_launch=2,
        )
    )
    observed = {}
    for block in sample_blocks:
        entry = observed.setdefault(block.native_block_generation_digest, [0, 0])
        entry[0] += 1
        entry[1] += block.sample_count
    assert tuple(
        (
            block.native_block_generation_digest,
            block.sample_chunk_count,
            block.sample_count,
        )
        for block in work.active_blocks
    ) == tuple(
        (binding.native_block_generation_digest, *observed[binding.native_block_generation_digest])
        for binding in bundle.native_blocks
        if binding.native_block_generation_digest in observed
    )

    union_scratch = torch.full((bundle.union_site_count, 4), 99.0, dtype=torch.float32)
    loss_scratch = torch.full((1,), 99.0, dtype=torch.float32)
    assembly = begin_paper_kinetic_union_local_bar_assembly(
        work,
        grad_union_site_rgba_f32=union_scratch,
        loss_f32=loss_scratch,
    )
    expected_union = torch.zeros_like(union_scratch)
    expected_loss = torch.zeros_like(loss_scratch)
    for work_index, active in enumerate(work.active_blocks):
        binding = bundle.binding_for_digest(active.native_block_generation_digest)
        vjp = _fake_vjp(binding, float(work_index + 1))
        block_loss = torch.tensor([float(10 + work_index)], dtype=torch.float32)
        contribution = seal_paper_kinetic_native_block_vjp_contribution(
            assembly,
            native_vjp_result=vjp,
            loss_f32=block_loss,
            reduced_sample_chunk_count=active.sample_chunk_count,
            reduced_sample_count=active.sample_count,
        )
        expected_union.index_add_(
            0,
            binding.compact_to_union_i64,
            vjp.grad_compact_site_rgba_f32,
        )
        expected_loss.add_(block_loss)
        consume_paper_kinetic_union_local_native_contribution(assembly, contribution)

    compact_result = finalize_paper_kinetic_union_local_bar_assembly(assembly)
    assert compact_result.request is request
    assert compact_result.grad_compact_site_rgba_f32 is union_scratch
    assert compact_result.loss_f32 is loss_scratch
    torch.testing.assert_close(union_scratch, expected_union)
    torch.testing.assert_close(loss_scratch, expected_loss)

    consume_paper_ragged_compact_material_bar_result(outer, request, compact_result)
    authorization = finalize_paper_ragged_material_bar_step(outer)
    torch.testing.assert_close(
        authorization.result.grad_global_site_rgba_f32.index_select(
            0,
            bundle.source_site_ids_i64,
        ),
        expected_union,
    )
    assert torch.count_nonzero(
        authorization.result.grad_global_site_rgba_f32[2:]
    ) == 0
    torch.testing.assert_close(authorization.result.loss_f32, expected_loss)

    accounting = assembly.accounting
    assert bundle.union_site_count < bundle.global_site_count
    assert accounting["consumed_native_block_count"] == work.active_native_block_count
    assert accounting["native_vjp_result_count"] == work.active_native_block_count
    assert accounting["per_request_global_material_bar_bytes"] == 0
    assert accounting["adapter_allocated_union_material_bar_bytes"] == 0
    assert accounting["adapter_allocated_global_material_bar_bytes"] == 0
    assert accounting["union_bar_zero_count"] == accounting["loss_zero_count"] == 1
    assert accounting["cross_native_duplicate_sites_sum_with_index_add"] is True
    assert accounting["persistent_frame_tensor_bytes"] == 0
    assert accounting["persistent_sample_tensor_bytes"] == 0
    assert accounting["persistent_target_tensor_bytes"] == 0
    assert accounting["retained_sample_partition_records"] == 0


def test_union_local_duplicate_missing_foreign_and_partial_chunk_work_fail_closed() -> None:
    _sampler_value, bundle, _outer, _request, work = _request_case()
    assembly = begin_paper_kinetic_union_local_bar_assembly(
        work,
        grad_union_site_rgba_f32=torch.zeros((bundle.union_site_count, 4)),
        loss_f32=torch.zeros((1,)),
    )
    first = work.active_blocks[0]
    binding = bundle.binding_for_digest(first.native_block_generation_digest)
    with pytest.raises(ValueError, match="every expected sample chunk"):
        seal_paper_kinetic_native_block_vjp_contribution(
            assembly,
            native_vjp_result=_fake_vjp(binding, 1.0),
            loss_f32=torch.zeros((1,)),
            reduced_sample_chunk_count=first.sample_chunk_count - 1,
            reduced_sample_count=first.sample_count,
        )

    foreign = _fake_vjp(binding, 1.0)
    foreign.world.runtime.payload.block.generation_digest = "foreign-native-block"
    with pytest.raises(ValueError, match="duplicate, out of order, or foreign"):
        seal_paper_kinetic_native_block_vjp_contribution(
            assembly,
            native_vjp_result=foreign,
            loss_f32=torch.zeros((1,)),
            reduced_sample_chunk_count=first.sample_chunk_count,
            reduced_sample_count=first.sample_count,
        )

    contribution = seal_paper_kinetic_native_block_vjp_contribution(
        assembly,
        native_vjp_result=_fake_vjp(binding, 1.0),
        loss_f32=torch.zeros((1,)),
        reduced_sample_chunk_count=first.sample_chunk_count,
        reduced_sample_count=first.sample_count,
    )
    consume_paper_kinetic_union_local_native_contribution(assembly, contribution)
    with pytest.raises(ValueError, match="duplicate, out of order, or foreign"):
        consume_paper_kinetic_union_local_native_contribution(assembly, contribution)
    with pytest.raises(ValueError, match="missing native contributions"):
        finalize_paper_kinetic_union_local_bar_assembly(assembly)


def test_union_local_persistent_bytes_are_frame_invariant_and_warm_checks_do_not_sync(
    monkeypatch,
) -> None:
    _sampler_value, bundle, _outer, _request, work = _request_case()
    small = bundle.memory_report(4)
    huge = bundle.memory_report(10_000_000)
    for name in small.__dataclass_fields__:
        if name != "requested_frame_count":
            assert getattr(small, name) == getattr(huge, name)
    assert small.persistent_mapping_tensor_bytes == (
        small.union_source_site_id_tensor_bytes
        + small.compact_to_union_mapping_tensor_bytes
    )
    assert small.request_union_material_bar_bytes == bundle.union_site_count * 16
    assert small.per_request_global_material_bar_bytes == 0
    assert small.persistent_frame_tensor_bytes == 0
    assert small.persistent_sample_tensor_bytes == 0
    assert small.persistent_target_tensor_bytes == 0

    assembly = begin_paper_kinetic_union_local_bar_assembly(
        work,
        grad_union_site_rgba_f32=torch.zeros((bundle.union_site_count, 4)),
        loss_f32=torch.zeros((1,)),
    )

    def forbid(*_args, **_kwargs):
        raise AssertionError("warm union-local validation synchronized or allocated")

    monkeypatch.setattr(torch.Tensor, "cpu", forbid)
    monkeypatch.setattr(torch.Tensor, "item", forbid)
    monkeypatch.setattr(torch.Tensor, "tolist", forbid)
    monkeypatch.setattr(torch, "empty", forbid)
    monkeypatch.setattr(torch, "zeros", forbid)
    monkeypatch.setattr(torch, "tensor", forbid)
    monkeypatch.setattr(torch, "as_tensor", forbid)
    bundle.assert_warm_layout()
    work.assert_warm_layout()
    union_assembly._assert_assembly_warm_layout(assembly)

    for function in (
        bundle.assert_warm_layout,
        work.assert_warm_layout,
        union_assembly._assert_assembly_warm_layout,
        consume_paper_kinetic_union_local_native_contribution,
        finalize_paper_kinetic_union_local_bar_assembly,
    ):
        source = inspect.getsource(function)
        for forbidden_text in (".cpu(", ".item(", ".tolist(", "_digest_parts("):
            assert forbidden_text not in source
