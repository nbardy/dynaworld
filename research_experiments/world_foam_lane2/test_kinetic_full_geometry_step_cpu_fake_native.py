from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
import torch
import kinetic_full_geometry_step_cpu_fake_native as full_geometry_step
from kinetic_full_geometry_step_cpu_fake_native import (
    begin_paper_kinetic_full_geometry_request,
    consume_paper_kinetic_full_geometry_native_block,
    finalize_and_consume_paper_kinetic_full_geometry_request,
)
from kinetic_native_material_step_executor import (
    prepare_kinetic_native_material_step_executor,
)
from kinetic_stable_stratum_vjp import (
    kinetic_p0_node_physical_length_geometry_vjp,
    make_frozen_kinetic_owner_word,
)
from paper_kinetic_ragged_sample_plan import (
    iter_paper_kinetic_row_ragged_request_blocks,
    iter_paper_kinetic_row_ragged_sample_blocks,
)
from paper_kinetic_union_local_bar_assembly import (
    prepare_paper_kinetic_union_local_request_work,
)
from paper_ragged_material_bar_coordinator import (
    begin_paper_ragged_material_bar_step,
    finalize_paper_ragged_material_bar_step,
    stage_next_paper_ragged_material_bar_request,
)
from test_kinetic_ragged_paper_step_cpu_fake_native import (
    _adapted,
    _compiled_case,
    _node_charts,
    _phi,
)


@dataclass
class _Fence:
    calls: int = 0
    wrong_return: bool = False

    def __call__(self):
        self.calls += 1
        return "not-none" if self.wrong_return else None


_SAMPLE_COMPLETION_FENCE_PROVENANCE = (
    "cpu-contract-double/full-geometry-sample-completion-fence-v1"
)


def _settle_sample(session, lifetime) -> None:
    receipt = session.settle_sample_accumulate(
        lifetime,
        device_completion_fence=lambda: None,
        device_completion_fence_provenance=(
            _SAMPLE_COMPLETION_FENCE_PROVENANCE
        ),
    )
    receipt.assert_current()


def _ray_keys(work):
    active = {
        block.native_block_generation_digest for block in work.active_blocks
    }
    return tuple(
        sorted(
            {
                (work.bundle.view_index, row.track_id)
                for row in work.sampler.rows
                if row.native_block_generation_digest in active
            }
        )
    )


def _run_request(
    compiled,
    adapted,
    *,
    sample_launch_size: int,
    alias_global_position_velocity: bool = False,
    commit_failure_probe: dict[str, object] | None = None,
    replay_first_consumed_execution: bool = False,
):
    lane = compiled.lane
    global_material = torch.zeros_like(compiled.global_site_rgba_f32)
    ledger = begin_paper_ragged_material_bar_step(
        adapted,
        programs=(compiled.view_program,),
        global_grad_site_rgba_f32=global_material,
    )
    request = stage_next_paper_ragged_material_bar_request(
        ledger,
        view_index=0,
        block_id=lane.block_id,
        local_sample_start=0,
        local_sample_end=adapted.observation_count,
    )
    work = prepare_paper_kinetic_union_local_request_work(
        lane.bundle,
        request,
        maximum_samples_per_launch=sample_launch_size,
    )
    executor = prepare_kinetic_native_material_step_executor(
        compiled.native_ops,
        tuple((runtime, work.sampler) for runtime in lane.runtimes),
        backend_provenance="cpu-contract-double/full-geometry-request",
    )
    session = executor.begin_step(
        step_generation_id=request.request_generation_id,
        requested_observation_count=work.total_sample_count,
    )
    runtime_by_digest = {
        runtime.payload.block.generation_digest: runtime
        for runtime in lane.runtimes
    }
    sites = work.sampler.rows[0].program.binding.sites
    keys = _ray_keys(work)
    request_positions = torch.empty_like(sites.positions0)
    request_velocities = torch.empty_like(sites.velocities)
    request_weights = torch.empty_like(sites.weight_coefficients)
    request_rays = torch.empty((len(keys), 12), dtype=torch.float64)
    assembly = begin_paper_kinetic_full_geometry_request(
        work,
        session,
        grad_union_site_rgba_f32=torch.empty(
            (lane.bundle.union_site_count, 4),
            dtype=torch.float32,
        ),
        loss_f32=torch.empty((1,), dtype=torch.float32),
        grad_positions0_f64=request_positions,
        grad_velocities_f64=request_velocities,
        grad_weight_coefficients_f64=request_weights,
        ray_bar_keys=keys,
        grad_track_ray_coefficients_f64=request_rays,
    )
    fence = _Fence()
    block_receipts = []
    cone = torch.zeros((3,), dtype=torch.int32)
    current_digest = None
    current_runtime = None
    current_token = None
    current_node_bar = None
    current_block_loss = None
    maximum_live_native_block_count = 0

    def finish_current_block() -> None:
        nonlocal current_digest
        nonlocal current_runtime
        nonlocal current_token
        nonlocal current_node_bar
        nonlocal current_block_loss
        if current_digest is None:
            return
        execution = session.launch_full_geometry_vjp(
            current_token,
            current_node_bar,
            compact_grad_site_rgba_f32=torch.empty(
                (current_runtime.compact_site_count, 4),
                dtype=torch.float32,
            ),
        )
        block_receipts.append(
            consume_paper_kinetic_full_geometry_native_block(
                assembly,
                session,
                execution,
                loss_f32=current_block_loss,
                device_completion_fence=fence,
                device_completion_fence_provenance="cpu-fake-request-fence-v1",
                maximum_geometry_bridge_visible_peak_logical_tensor_bytes=10_000_000,
            )
        )
        assert execution.consumed and execution.native_vjp_result is None
        if replay_first_consumed_execution and len(block_receipts) == 1:
            with pytest.raises(ValueError, match="no longer owns its native result"):
                consume_paper_kinetic_full_geometry_native_block(
                    assembly,
                    session,
                    execution,
                    loss_f32=current_block_loss,
                    device_completion_fence=fence,
                    device_completion_fence_provenance="cpu-fake-replay-fence-v1",
                    maximum_geometry_bridge_visible_peak_logical_tensor_bytes=(
                        10_000_000
                    ),
                )
            assembly.assert_open()
        current_digest = None
        current_runtime = None
        current_token = None
        current_node_bar = None
        current_block_loss = None

    for block in iter_paper_kinetic_row_ragged_request_blocks(
        work.sampler,
        request,
        maximum_samples_per_launch=sample_launch_size,
    ):
        digest = block.native_block_generation_digest
        if current_digest != digest:
            finish_current_block()
            current_digest = digest
            current_runtime = runtime_by_digest[digest]
            compact_material = compiled.global_site_rgba_f32.index_select(
                0,
                current_runtime.source_site_ids_i64,
            ).contiguous()
            current_token = session.launch_node_forward(
                current_runtime,
                compact_material,
            )
            current_node_bar = torch.zeros_like(current_token.world.node_chart_f32)
            current_block_loss = torch.zeros((1,), dtype=torch.float32)
            maximum_live_native_block_count = max(
                maximum_live_native_block_count,
                1,
            )
        lifetime = session.launch_sample_accumulate(
            current_token,
            block,
            sampler=work.sampler,
            background_rgb_f32=compiled.background_rgb_f32,
            loss_f32=current_block_loss,
            grad_node_chart_f32=current_node_bar,
            cone_diagnostic_i32=cone,
        )
        _settle_sample(session, lifetime)
    finish_current_block()
    assert maximum_live_native_block_count == 1
    assert current_token is current_node_bar is current_block_loss is None
    telemetry = session.seal()
    telemetry.assert_current()
    assert telemetry.reverse_mode == "full_geometry"
    assert telemetry.streamed_sample_count == work.total_sample_count
    assert (
        telemetry.native_full_geometry_vjp_launch_count
        == work.active_native_block_count
    )

    global_positions = torch.zeros_like(sites.positions0)
    global_velocities = (
        global_positions
        if alias_global_position_velocity
        else torch.zeros_like(sites.velocities)
    )
    global_weights = torch.zeros_like(sites.weight_coefficients)
    global_rays = torch.zeros((len(keys), 12), dtype=torch.float64)
    request_local_tensors = assembly._tensors()
    try:
        receipt = finalize_and_consume_paper_kinetic_full_geometry_request(
            assembly,
            ledger,
            request,
            executor_telemetry=telemetry,
            global_grad_positions0_f64=global_positions,
            global_grad_velocities_f64=global_velocities,
            global_grad_weight_coefficients_f64=global_weights,
            global_ray_bar_keys=keys,
            global_grad_track_ray_coefficients_f64=global_rays,
        )
    except BaseException:
        if commit_failure_probe is not None:
            commit_failure_probe.update(
                assembly=assembly,
                ledger=ledger,
                request=request,
                request_local_tensors=request_local_tensors,
                global_tensors=(
                    ledger.global_grad_site_rgba_f32,
                    ledger.loss_f32,
                    global_positions,
                    global_velocities,
                    global_weights,
                    global_rays,
                ),
            )
        raise
    assert assembly.finalized and not assembly.poisoned
    assert assembly.material is None and assembly.request_references_released
    assert assembly.block_receipts == tuple(block_receipts)
    for block_receipt in block_receipts:
        block_receipt.assert_current()
    authorization = finalize_paper_ragged_material_bar_step(ledger)
    callback_calls = 0

    def update(_result) -> None:
        nonlocal callback_calls
        callback_calls += 1

    authorization.consume(update)
    assert callback_calls == 1
    return (
        authorization.result,
        global_positions,
        global_velocities,
        global_weights,
        global_rays,
        receipt,
        tuple(block_receipts),
        fence,
    )


def _direct_full_oracle(compiled, adapted, ray_bar_keys):
    global_rgba = compiled.global_site_rgba_f32.detach().clone().requires_grad_(True)
    payloads = {
        payload.block.generation_digest: payload for payload in compiled.payloads
    }
    lengths = {
        digest: payload.node_physical_length_f32.detach().clone().requires_grad_(True)
        for digest, payload in payloads.items()
    }
    nodes = {
        digest: _node_charts(
            payload.word_offsets_i32,
            payload.word_owner_i32,
            lengths[digest],
            global_rgba.index_select(0, payload.source_site_ids_i64),
        )
        for digest, payload in payloads.items()
    }
    group = adapted.groups[0]
    staged = group.stage_targets(
        track_start=0,
        track_end=adapted.pixel_count,
        sample_start=0,
        sample_end=group.observation_count,
    )
    loss = torch.zeros((), dtype=torch.float32)
    for block in iter_paper_kinetic_row_ragged_sample_blocks(
        compiled.sampler,
        staged,
        loss_normalization_id=adapted.loss_normalization_id,
        maximum_samples_per_launch=7,
    ):
        selected = nodes[block.native_block_generation_digest].index_select(
            0,
            block.sample_row_i32.to(torch.int64),
        )
        chart = torch.sum(selected * block.sample_to_node_f32[:, :, None], dim=1)
        kappa = chart[:, 0]
        prediction = (
            _phi(kappa)[:, None] * chart[:, 1:]
            + torch.exp(-kappa)[:, None] * compiled.background_rgb_f32
        )
        loss = loss + (prediction - block.target_rgb_f32).square().sum() * block.loss_scale
    length_items = tuple(lengths.items())
    grads = torch.autograd.grad(
        loss,
        (global_rgba,) + tuple(value for _digest, value in length_items),
    )
    length_bars = {
        digest: bar
        for (digest, _value), bar in zip(length_items, grads[1:], strict=True)
    }
    sites = compiled.sampler.rows[0].program.binding.sites
    positions = torch.zeros_like(sites.positions0)
    velocities = torch.zeros_like(sites.velocities)
    weights = torch.zeros_like(sites.weight_coefficients)
    rays = torch.zeros((len(ray_bar_keys), 12), dtype=torch.float64)
    ray_position = {key: index for index, key in enumerate(ray_bar_keys)}
    rows_by_index = {
        row.global_row_index: row for row in compiled.sampler.lowering.rows
    }
    for payload in compiled.payloads:
        digest = payload.block.generation_digest
        bindings = tuple(
            sorted(
                (
                    row
                    for row in compiled.sampler.rows
                    if row.native_block_generation_digest == digest
                ),
                key=lambda row: row.native_local_row_index,
            )
        )
        row_specs = tuple(rows_by_index[index] for index in payload.block.global_row_indices)
        word_start = 0
        for binding, row in zip(bindings, row_specs, strict=True):
            word_end = word_start + row.word_count
            chart = binding.program.charts[binding.chart_index]
            topology = binding.source.lowering.charts[binding.chart_index]
            result = kinetic_p0_node_physical_length_geometry_vjp(
                sites,
                binding.program.binding.ray_coefficients,
                chart.schedule.node_times,
                (make_frozen_kinetic_owner_word(row.owner_word),),
                length_bars[digest][:, word_start:word_end].to(torch.float64),
                near=row.near,
                far=row.far,
                continuous_topology_certificate_id=(
                    topology.owner_topology_certificate_digest
                ),
                node_physical_length_cotangent_provenance_id="direct-request-oracle",
            )
            positions.add_(result.grad_positions0)
            velocities.add_(result.grad_velocities)
            weights.add_(result.grad_weight_coefficients)
            rays[ray_position[(compiled.sampler.view_index, binding.track_id)]].add_(
                result.grad_ray_coefficients
            )
            word_start = word_end
    return loss.reshape(1), grads[0], positions, velocities, weights, rays


def test_full_request_matches_direct_multirow_oracle_and_global_coordinator() -> None:
    compiled = _compiled_case()
    adapted = _adapted(7)
    before_full = compiled.native_ops.vjp_calls
    before_material = compiled.native_ops.material_vjp_calls
    actual = _run_request(compiled, adapted, sample_launch_size=3)
    material, positions, velocities, weights, rays, receipt, blocks, fence = actual
    oracle = _direct_full_oracle(compiled, adapted, receipt.ray_bar_keys)
    active = int(receipt.accounting["active_native_block_count"])

    assert active > 1
    assert compiled.native_ops.vjp_calls - before_full == active
    assert compiled.native_ops.material_vjp_calls - before_material == 0
    assert fence.calls == len(blocks) == active
    assert all(block.execution_consumed for block in blocks)
    assert all(len(block.sample_manifest_digest) == 64 for block in blocks)
    assert len(str(receipt.accounting["ordered_block_receipt_digest"])) == 64
    assert len(str(receipt.accounting["ordered_sample_manifest_digest"])) == 64
    assert len(str(receipt.accounting["executor_telemetry_generation_digest"])) == 64
    assert receipt.accounting["loss_normalization_id"] == adapted.loss_normalization_id
    assert int(receipt.accounting["global_loss_element_count"]) == (
        adapted.pixel_count * adapted.observation_count * 3
    )
    assert int(receipt.accounting["request_total_sample_count"]) == (
        adapted.pixel_count * adapted.observation_count
    )
    assert int(receipt.accounting["expected_sample_chunk_count"]) == sum(
        block.reduced_sample_chunk_count for block in blocks
    )
    assert receipt.accounting["request_references_released_before_receipt_return"]
    assert not hasattr(receipt, "request")
    assert receipt.material_routed_through_existing_union_and_global_coordinators
    assert receipt.cpu_fake_native_only
    assert not receipt.accelerator_tensor_admission_allowed
    assert receipt.runtime_status == full_geometry_step.FULL_GEOMETRY_STATUS
    assert receipt.accounting["cpu_only_entry_and_finalization_enforced"]
    assert not receipt.accounting["accelerator_tensor_admission_allowed"]
    assert not receipt.frame_sample_target_prediction_or_native_state_retained
    assert receipt.accounting["geometry_validation_scaling"] == "O(sum_rows J * S * R_row)"
    assert int(receipt.accounting["dense_global_site_accumulation_elements"]) > 0
    tensors = (
        material.loss_f32,
        material.grad_global_site_rgba_f32,
        positions,
        velocities,
        weights,
        rays,
    )
    for result, expected in zip(tensors, oracle, strict=True):
        torch.testing.assert_close(result, expected, rtol=8e-5, atol=8e-6)
    receipt.assert_current()


def test_global_geometry_destinations_must_own_distinct_storage() -> None:
    with pytest.raises(ValueError, match="must not alias"):
        _run_request(
            _compiled_case(),
            _adapted(3),
            sample_launch_size=2,
            alias_global_position_velocity=True,
        )


def test_equal_count_foreign_executor_session_cannot_bind_request() -> None:
    compiled = _compiled_case()
    adapted = _adapted(3)
    lane = compiled.lane
    ledger = begin_paper_ragged_material_bar_step(
        adapted,
        programs=(compiled.view_program,),
        global_grad_site_rgba_f32=torch.zeros_like(compiled.global_site_rgba_f32),
    )
    request = stage_next_paper_ragged_material_bar_request(
        ledger,
        view_index=0,
        block_id=lane.block_id,
        local_sample_start=0,
        local_sample_end=adapted.observation_count,
    )
    work = prepare_paper_kinetic_union_local_request_work(
        lane.bundle,
        request,
        maximum_samples_per_launch=2,
    )
    executor = prepare_kinetic_native_material_step_executor(
        compiled.native_ops,
        tuple((runtime, work.sampler) for runtime in lane.runtimes),
        backend_provenance="cpu-contract-double/foreign-full-geometry-request",
    )
    foreign = executor.begin_step(
        step_generation_id="equal-count-foreign-request",
        requested_observation_count=work.total_sample_count,
    )
    sites = work.sampler.rows[0].program.binding.sites
    keys = _ray_keys(work)
    with pytest.raises(ValueError, match="different or active request"):
        begin_paper_kinetic_full_geometry_request(
            work,
            foreign,
            grad_union_site_rgba_f32=torch.empty(
                (lane.bundle.union_site_count, 4),
                dtype=torch.float32,
            ),
            loss_f32=torch.empty((1,), dtype=torch.float32),
            grad_positions0_f64=torch.empty_like(sites.positions0),
            grad_velocities_f64=torch.empty_like(sites.velocities),
            grad_weight_coefficients_f64=torch.empty_like(
                sites.weight_coefficients
            ),
            ray_bar_keys=keys,
            grad_track_ray_coefficients_f64=torch.empty(
                (len(keys), 12),
                dtype=torch.float64,
            ),
        )
    accelerator_bundle = replace(
        work.bundle,
        source_site_ids_i64=torch.empty_like(
            work.bundle.source_site_ids_i64,
            device="meta",
        ),
    )
    accelerator_work = replace(work, bundle=accelerator_bundle)
    with pytest.raises(ValueError, match="CPU/fake-native-only"):
        begin_paper_kinetic_full_geometry_request(
            accelerator_work,
            foreign,
            grad_union_site_rgba_f32=torch.empty(
                (lane.bundle.union_site_count, 4),
                dtype=torch.float32,
            ),
            loss_f32=torch.empty((1,), dtype=torch.float32),
            grad_positions0_f64=torch.empty_like(sites.positions0),
            grad_velocities_f64=torch.empty_like(sites.velocities),
            grad_weight_coefficients_f64=torch.empty_like(
                sites.weight_coefficients
            ),
            ray_bar_keys=keys,
            grad_track_ray_coefficients_f64=torch.empty(
                (len(keys), 12),
                dtype=torch.float64,
            ),
        )


def test_consumed_block_receipt_cannot_forge_equal_count_provenance() -> None:
    result = _run_request(_compiled_case(), _adapted(3), sample_launch_size=2)
    block = result[6][0]
    forged = replace(
        block,
        sample_manifest_digest="0" * 64,
    )
    with pytest.raises(ValueError, match="block receipt changed"):
        forged.assert_current()


def test_equal_count_consumed_execution_cannot_replay() -> None:
    _run_request(
        _compiled_case(),
        _adapted(3),
        sample_launch_size=2,
        replay_first_consumed_execution=True,
    )


def test_post_start_commit_failure_zeros_and_poison_all_global_bars(monkeypatch) -> None:
    original_consume = (
        full_geometry_step.consume_paper_ragged_compact_material_bar_result
    )
    consume_calls = 0

    def consume_then_fail(*args, **kwargs) -> None:
        nonlocal consume_calls
        original_consume(*args, **kwargs)
        consume_calls += 1
        raise RuntimeError("injected post-material-commit failure")

    monkeypatch.setattr(
        full_geometry_step,
        "consume_paper_ragged_compact_material_bar_result",
        consume_then_fail,
    )
    probe: dict[str, object] = {}
    with pytest.raises(RuntimeError, match="injected post-material-commit failure"):
        _run_request(
            _compiled_case(),
            _adapted(3),
            sample_launch_size=2,
            commit_failure_probe=probe,
        )

    assembly = probe["assembly"]
    ledger = probe["ledger"]
    assert consume_calls == 1
    assert assembly.poisoned and not assembly.finalized
    assert assembly.material is None and assembly.request_references_released
    assert ledger.active_request is None
    assert ledger.finalized and getattr(ledger, "_full_geometry_poisoned", False)
    for tensor in probe["request_local_tensors"] + probe["global_tensors"]:
        assert torch.count_nonzero(tensor).item() == 0


def test_chunk_k_and_requested_f_do_not_multiply_reverse_or_selected_bar_bytes() -> None:
    compiled = _compiled_case()
    adapted = _adapted(9)
    first = _run_request(compiled, adapted, sample_launch_size=1)
    second = _run_request(compiled, adapted, sample_launch_size=11)
    for left, right in zip(first[:5], second[:5], strict=True):
        left_tensors = (
            (left.loss_f32, left.grad_global_site_rgba_f32)
            if hasattr(left, "loss_f32")
            else (left,)
        )
        right_tensors = (
            (right.loss_f32, right.grad_global_site_rgba_f32)
            if hasattr(right, "loss_f32")
            else (right,)
        )
        for left_tensor, right_tensor in zip(left_tensors, right_tensors, strict=True):
            torch.testing.assert_close(left_tensor, right_tensor, rtol=8e-5, atol=8e-6)
    first_receipt = first[5]
    second_receipt = second[5]
    for key in (
        "active_native_block_count",
        "native_full_vjp_invocation_count",
        "device_completion_fence_call_count",
        "differentiable_word_reverse_interactions",
        "dense_global_site_accumulation_elements",
        "all_site_owner_validation_evaluations",
        "maximum_native_length_bar_tensor_bytes",
        "maximum_geometry_bridge_visible_tensor_bytes",
    ):
        assert first_receipt.accounting[key] == second_receipt.accounting[key]
    small = first_receipt.memory_report(2)
    large = first_receipt.memory_report(1_000_000)
    assert {
        key: value for key, value in small.__dict__.items() if key != "requested_frame_count"
    } == {
        key: value for key, value in large.__dict__.items() if key != "requested_frame_count"
    }
    assert not small.hot_native_block_state_tensor_bytes_included
    assert not small.whole_step_peak_measured
    assert not first_receipt.accounting["whole_step_peak_measured"]

    denser = _run_request(compiled, _adapted(13), sample_launch_size=5)
    for key in (
        "active_native_block_count",
        "native_full_vjp_invocation_count",
        "device_completion_fence_call_count",
        "maximum_native_length_bar_tensor_bytes",
    ):
        assert denser[5].accounting[key] == second_receipt.accounting[key]
    assert int(denser[5].accounting["reduced_sample_count"]) > int(
        second_receipt.accounting["reduced_sample_count"]
    )


def test_fence_failure_poison_clears_request_bars_before_global_consumption() -> None:
    compiled = _compiled_case()
    adapted = _adapted(3)
    lane = compiled.lane
    ledger = begin_paper_ragged_material_bar_step(
        adapted,
        programs=(compiled.view_program,),
        global_grad_site_rgba_f32=torch.zeros_like(compiled.global_site_rgba_f32),
    )
    request = stage_next_paper_ragged_material_bar_request(
        ledger,
        view_index=0,
        block_id=lane.block_id,
        local_sample_start=0,
        local_sample_end=adapted.observation_count,
    )
    work = prepare_paper_kinetic_union_local_request_work(
        lane.bundle,
        request,
        maximum_samples_per_launch=99,
    )
    first = work.active_blocks[0]
    executor = prepare_kinetic_native_material_step_executor(
        compiled.native_ops,
        tuple((runtime, work.sampler) for runtime in lane.runtimes),
        backend_provenance="cpu-contract-double/failing-full-geometry-request",
    )
    session = executor.begin_step(
        step_generation_id=request.request_generation_id,
        requested_observation_count=work.total_sample_count,
    )
    sites = work.sampler.rows[0].program.binding.sites
    keys = _ray_keys(work)
    local_tensors = (
        torch.full((lane.bundle.union_site_count, 4), 7.0, dtype=torch.float32),
        torch.full((1,), 7.0, dtype=torch.float32),
        torch.full_like(sites.positions0, 7.0),
        torch.full_like(sites.velocities, 7.0),
        torch.full_like(sites.weight_coefficients, 7.0),
        torch.full((len(keys), 12), 7.0, dtype=torch.float64),
    )
    assembly = begin_paper_kinetic_full_geometry_request(
        work,
        session,
        grad_union_site_rgba_f32=local_tensors[0],
        loss_f32=local_tensors[1],
        grad_positions0_f64=local_tensors[2],
        grad_velocities_f64=local_tensors[3],
        grad_weight_coefficients_f64=local_tensors[4],
        ray_bar_keys=keys,
        grad_track_ray_coefficients_f64=local_tensors[5],
    )
    runtime = lane.runtime_for_digest(first.native_block_generation_digest)
    token = session.launch_node_forward(
        runtime,
        compiled.global_site_rgba_f32.index_select(
            0,
            runtime.source_site_ids_i64,
        ).contiguous(),
    )
    node_bar = torch.zeros_like(token.world.node_chart_f32)
    block_loss = torch.zeros((1,), dtype=torch.float32)
    cone = torch.zeros((3,), dtype=torch.int32)
    selected_sample_count = 0
    selected_chunk_count = 0
    for sample_block in iter_paper_kinetic_row_ragged_request_blocks(
        work.sampler,
        request,
        maximum_samples_per_launch=99,
    ):
        if (
            sample_block.native_block_generation_digest
            != first.native_block_generation_digest
        ):
            continue
        lifetime = session.launch_sample_accumulate(
            token,
            sample_block,
            sampler=work.sampler,
            background_rgb_f32=compiled.background_rgb_f32,
            loss_f32=block_loss,
            grad_node_chart_f32=node_bar,
            cone_diagnostic_i32=cone,
        )
        _settle_sample(session, lifetime)
        selected_chunk_count += 1
        selected_sample_count += sample_block.sample_count
    assert selected_chunk_count == first.sample_chunk_count
    assert selected_sample_count == first.sample_count
    execution = session.launch_full_geometry_vjp(
        token,
        node_bar,
        compact_grad_site_rgba_f32=torch.empty(
            (runtime.compact_site_count, 4),
            dtype=torch.float32,
        ),
    )
    fence = _Fence(wrong_return=True)
    with pytest.raises(ValueError, match="receipt changed or is foreign"):
        consume_paper_kinetic_full_geometry_native_block(
            assembly,
            session,
            execution,
            loss_f32=block_loss.clone(),
            device_completion_fence=fence,
            device_completion_fence_provenance="failing-test-fence",
            maximum_geometry_bridge_visible_peak_logical_tensor_bytes=10_000_000,
        )
    assert fence.calls == 0
    assembly.assert_open()
    with pytest.raises(TypeError, match="must return None"):
        consume_paper_kinetic_full_geometry_native_block(
            assembly,
            session,
            execution,
            loss_f32=block_loss,
            device_completion_fence=fence,
            device_completion_fence_provenance="failing-test-fence",
            maximum_geometry_bridge_visible_peak_logical_tensor_bytes=10_000_000,
        )
    assert fence.calls == 2
    assert assembly.poisoned and not assembly.finalized
    assert assembly.material is None and assembly.request_references_released
    assert session._failed and not session._abort_release_completed
    assert not execution.consumed and execution.native_vjp_result is not None
    assert ledger.active_request is request
    for tensor in local_tensors:
        assert torch.count_nonzero(tensor).item() == 0
    with pytest.raises(ValueError, match="poisoned"):
        assembly.assert_open()
    cleanup_fence = _Fence()
    session.abort(
        device_completion_fence=cleanup_fence,
        device_completion_fence_provenance="cpu-fake-test-cleanup-fence-v1",
    )
    assert cleanup_fence.calls == 1
    assert session._abort_release_completed
    assert execution.consumed and execution.native_vjp_result is None
